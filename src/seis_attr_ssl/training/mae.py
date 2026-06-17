"""Stage 1 strict MAE pretraining engine."""

from __future__ import annotations

import math
import re
import shutil
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import torch

import seis_attr_ssl
from seis_attr_ssl.attributes import MVP_ATTRIBUTE_REGISTRY
from seis_attr_ssl.data import NopimsAttributePretrainDataset, read_manifest_json
from seis_attr_ssl.losses import mae_pretraining_loss
from seis_attr_ssl.models.mae import StrictAttributeSetMAE3D
from seis_attr_ssl.training.checkpoint import load_checkpoint, save_checkpoint
from seis_attr_ssl.training.collate import move_batch_to_device
from seis_attr_ssl.training.dataloaders import build_mae_dataloader
from seis_attr_ssl.training.diagnostics import (
	build_mae_nonfinite_diagnostic,
	summarize_tensor,
	write_json_diagnostic,
)
from seis_attr_ssl.training.logging import print_epoch_metrics
from seis_attr_ssl.visualization.mae_debug import (
	MaeDebugVisualizationConfig,
	save_mae_debug_visualization_pngs,
)

_MANIFEST_BUILD_HINT = (
	'Build NOPIMS manifests with '
	'`python proc/build_nopims_manifests.py --config '
	'proc/configs/build_nopims_manifests.yaml`.'
)


@dataclass(frozen=True)
class MaeTrainingState:
	"""Summary state returned from one MAE training epoch."""

	epoch: int
	global_step: int
	metrics: dict[str, float]
	amp_enabled: bool


@dataclass(frozen=True)
class MaeStepState:
	"""State captured immediately after one MAE optimizer step."""

	epoch: int
	batch_index: int
	global_step: int
	metrics: dict[str, float]
	amp_enabled: bool


StepCallback = Callable[[MaeStepState], None]


def train_mae_one_epoch(  # noqa: C901, PLR0912, PLR0913, PLR0915
	*,
	model: torch.nn.Module,
	dataloader: torch.utils.data.DataLoader,
	optimizer: torch.optim.Optimizer,
	device: torch.device,
	epoch: int,
	patch_size_xyz: tuple[int, int, int],
	loss_config: Mapping[str, object],
	amp_enabled: bool = False,
	scaler: torch.amp.GradScaler | None = None,
	global_step: int = 0,
	max_steps: int | None = None,
	diagnostics_dir: Path | None = None,
	grad_clip_norm: float | None = None,
	visualization_config: MaeDebugVisualizationConfig | None = None,
	step_callback: StepCallback | None = None,
) -> MaeTrainingState:
	"""Train ``model`` for one epoch and return averaged loss metrics."""
	model.train()
	totals: dict[str, float] = {}
	batches = 0
	epoch_visualization_batches = 0

	for batch_index, raw_batch in enumerate(dataloader):
		if max_steps is not None and batches >= max_steps:
			break
		batch = move_batch_to_device(raw_batch, device)
		optimizer.zero_grad(set_to_none=True)

		with torch.amp.autocast('cuda', enabled=amp_enabled):
			output = model(cast('Mapping[str, torch.Tensor]', batch))
			losses = mae_pretraining_loss(
				pred_patches=_required_tensor(output, 'pred_patches'),
				target=_required_tensor(batch, 'target'),
				spatial_mask=_required_tensor(batch, 'spatial_mask'),
				target_valid=_required_tensor(batch, 'target_valid'),
				dropped_attribute_mask=_required_tensor(
					batch,
					'dropped_attribute_mask',
				),
				local_valid_mask=_optional_tensor(batch, 'local_valid_mask'),
				valid_patch_min_fraction=_float_config(
					loss_config,
					'valid_patch_min_fraction',
					0.5,
				),
				patch_size_xyz=patch_size_xyz,
				reconstruction=_loss_mode(loss_config.get('reconstruction', 'huber')),
				huber_delta=_float_config(loss_config, 'huber_delta', 1.0),
				dropped_attribute_weight=_float_config(
					loss_config,
					'dropped_attribute_weight',
					0.25,
				),
				gradient_weight=_float_config(loss_config, 'gradient_weight', 0.05),
				family_balanced=_bool_config(
					loss_config,
					'family_balanced',
					default=True,
				),
			)
			loss = losses['loss']

		if not torch.isfinite(loss).all():
			if diagnostics_dir is not None:
				diagnostic_path = write_json_diagnostic(
					build_mae_nonfinite_diagnostic(
						global_step=global_step,
						epoch=epoch,
						batch_index=batch_index,
						batch=batch,
						output=output,
						losses=losses,
						torch_amp_enabled=amp_enabled,
					),
					diagnostics_dir / f'nonfinite_mae_step_{global_step:08d}.json',
				)
				msg = (
					f'non-finite MAE loss at epoch {epoch}, step {global_step}; '
					f'diagnostic written to {diagnostic_path}'
				)
			else:
				msg = f'non-finite MAE loss at epoch {epoch}, step {global_step}'
			raise FloatingPointError(msg)

		current_grad_norm: float | None = None
		if amp_enabled:
			if scaler is None:
				msg = 'scaler is required when amp_enabled is true'
				raise ValueError(msg)
			scaler.scale(loss).backward()
			if grad_clip_norm is not None:
				scaler.unscale_(optimizer)
				grad_norm = torch.nn.utils.clip_grad_norm_(
					model.parameters(),
					grad_clip_norm,
				)
				_check_finite_grad_norm(
					grad_norm=grad_norm,
					model=model,
					global_step=global_step,
					epoch=epoch,
					batch_index=batch_index,
					batch=batch,
					output=output,
					losses=losses,
					amp_enabled=amp_enabled,
					diagnostics_dir=diagnostics_dir,
				)
				current_grad_norm = _grad_norm_float(grad_norm)
				totals['grad_norm'] = totals.get('grad_norm', 0.0) + current_grad_norm
			scaler.step(optimizer)
			scaler.update()
		else:
			loss.backward()
			if grad_clip_norm is not None:
				grad_norm = torch.nn.utils.clip_grad_norm_(
					model.parameters(),
					grad_clip_norm,
				)
				_check_finite_grad_norm(
					grad_norm=grad_norm,
					model=model,
					global_step=global_step,
					epoch=epoch,
					batch_index=batch_index,
					batch=batch,
					output=output,
					losses=losses,
					amp_enabled=amp_enabled,
					diagnostics_dir=diagnostics_dir,
				)
				current_grad_norm = _grad_norm_float(grad_norm)
				totals['grad_norm'] = totals.get('grad_norm', 0.0) + current_grad_norm
			optimizer.step()

		step_metrics: dict[str, float] = {}
		for key, value in losses.items():
			metric = float(value.detach().cpu().item())
			step_metrics[key] = metric
			totals[key] = totals.get(key, 0.0) + metric
		if current_grad_norm is not None:
			step_metrics['grad_norm'] = current_grad_norm
		batches += 1
		global_step += 1
		if visualization_config is not None:
			epoch_triggered = _mae_debug_epoch_triggered(
				config=visualization_config,
				epoch=epoch,
				epoch_visualization_batches=epoch_visualization_batches,
			)
			step_triggered = _mae_debug_step_triggered(
				config=visualization_config,
				global_step=global_step,
			)
			if epoch_triggered or step_triggered:
				_save_mae_debug_visualization(
					batch=batch,
					model_output=output,
					patch_size_xyz=patch_size_xyz,
					epoch=epoch,
					global_step=global_step,
					config=visualization_config,
				)
				if epoch_triggered:
					epoch_visualization_batches += 1
		if step_callback is not None:
			step_callback(
				MaeStepState(
					epoch=epoch,
					batch_index=batch_index,
					global_step=global_step,
					metrics=step_metrics,
					amp_enabled=amp_enabled,
				),
			)

	if batches == 0:
		msg = 'dataloader produced no batches'
		raise ValueError(msg)

	return MaeTrainingState(
		epoch=epoch,
		global_step=global_step,
		metrics={key: total / batches for key, total in totals.items()},
		amp_enabled=amp_enabled,
	)


def run_mae_pretraining(
	config: Mapping[str, object],
	*,
	resume: str | Path | None = None,
) -> Path:
	"""Run strict MAE pretraining from ``config`` and return the last checkpoint."""
	_validate_no_f3_pretraining_config(config)
	manifests = read_manifest_json(_manifest_train_path(config))
	train_config = _mapping(config, 'train')
	model_config = _mapping(config, 'model')
	paths_config = _mapping(config, 'paths')
	loss_config = _mapping(config, 'loss')

	device = _resolve_device(train_config)
	torch.manual_seed(_int_config(train_config, 'seed', 42))

	samples_per_epoch = _optional_int_config(train_config, 'samples_per_epoch')
	dataset = NopimsAttributePretrainDataset.from_config(
		manifests,
		config,
		samples_per_epoch=samples_per_epoch,
	)
	dataloader = build_mae_dataloader(
		dataset,
		batch_size=_int_config(train_config, 'batch_size', 1),
		num_workers=_nonnegative_int_config(train_config, 'num_workers', 0),
		shuffle=_bool_config(train_config, 'shuffle', default=True),
		seed=_int_config(train_config, 'seed', 42),
	)

	model = _build_model(model_config, config).to(device)
	optimizer = torch.optim.AdamW(
		model.parameters(),
		lr=_float_config(train_config, 'lr', 1.0e-4),
		weight_decay=_float_config(train_config, 'weight_decay', 0.05),
	)
	amp_enabled = (
		_bool_config(train_config, 'amp', default=False)
		and device.type == 'cuda'
		and torch.cuda.is_available()
	)
	scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled) if amp_enabled else None
	start_epoch = 1
	global_step = 0
	if resume is not None:
		payload = load_checkpoint(resume, map_location=device)
		start_epoch, global_step = _restore_mae_checkpoint(
			payload=payload,
			model=model,
			optimizer=optimizer,
			scaler=scaler,
			amp_enabled=amp_enabled,
		)

	output_root = Path(_str_config(paths_config, 'output_root'))
	diagnostics_dir = _resolve_diagnostics_dir(train_config, output_root)
	epochs = _int_config(train_config, 'epochs', 1)
	max_steps = _optional_int_config(train_config, 'max_steps')
	checkpoint_every_steps = _optional_int_config(
		train_config,
		'checkpoint_every_steps',
	)
	grad_clip_norm = _optional_positive_float_config(train_config, 'grad_clip_norm')
	visualization_config = _mae_debug_visualization_config(config, paths_config)
	state: MaeTrainingState | None = MaeTrainingState(
		epoch=start_epoch - 1,
		global_step=global_step,
		metrics={},
		amp_enabled=amp_enabled,
	)
	checkpoint_path: Path | None = None
	for epoch in range(start_epoch, epochs + 1):
		set_epoch = getattr(dataset, 'set_epoch', None)
		if callable(set_epoch):
			set_epoch(epoch - 1)
		remaining_steps = None
		if max_steps is not None:
			remaining_steps = max_steps - state.global_step
			if remaining_steps <= 0:
				break

		def save_step_checkpoint(step_state: MaeStepState) -> None:
			nonlocal checkpoint_path
			if (
				checkpoint_every_steps is None
				or step_state.global_step % checkpoint_every_steps != 0
			):
				return
			checkpoint_path = _save_mae_checkpoint(
				output_root / f'mae_step_{step_state.global_step:08d}.pt',
				model=model,
				optimizer=optimizer,
				epoch=step_state.epoch,
				config=config,
				metrics=step_state.metrics,
				global_step=step_state.global_step,
				amp_enabled=step_state.amp_enabled,
				scaler=scaler,
				checkpoint_kind='step',
				batch_index=step_state.batch_index,
			)

		state = train_mae_one_epoch(
			model=model,
			dataloader=dataloader,
			optimizer=optimizer,
			device=device,
			epoch=epoch,
			patch_size_xyz=_xyz_config(model_config, 'patch_size'),
			loss_config=loss_config,
			amp_enabled=amp_enabled,
			scaler=scaler,
			global_step=state.global_step,
			max_steps=remaining_steps,
			diagnostics_dir=diagnostics_dir,
			grad_clip_norm=grad_clip_norm,
			visualization_config=visualization_config,
			step_callback=save_step_checkpoint,
		)
		print_epoch_metrics(epoch, state.metrics)
		checkpoint_path = _save_mae_checkpoint(
			output_root / f'mae_epoch_{epoch:04d}.pt',
			model=model,
			optimizer=optimizer,
			epoch=epoch,
			config=config,
			metrics={**state.metrics, 'amp_enabled': float(state.amp_enabled)},
			global_step=state.global_step,
			amp_enabled=state.amp_enabled,
			scaler=scaler,
			checkpoint_kind='epoch',
			batch_index=None,
		)
		if max_steps is not None and state.global_step >= max_steps:
			break

	if checkpoint_path is None:
		msg = 'no MAE training epochs were run'
		raise ValueError(msg)
	return checkpoint_path


def _mae_debug_visualization_config(
	config: Mapping[str, object],
	paths_config: Mapping[str, object],
) -> MaeDebugVisualizationConfig | None:
	visualization = config.get('visualization')
	if visualization is None:
		return None
	if not isinstance(visualization, Mapping):
		msg = f'visualization must be a mapping; got {visualization!r}'
		raise TypeError(msg)
	mae_debug = visualization.get('mae_debug')
	if mae_debug is None:
		return None
	if not isinstance(mae_debug, Mapping):
		msg = f'visualization.mae_debug must be a mapping; got {mae_debug!r}'
		raise TypeError(msg)
	if not _bool_config(mae_debug, 'enabled', default=False):
		return None

	output_dir_value = mae_debug.get('output_dir')
	if output_dir_value is None:
		output_dir = (
			Path(_str_config(paths_config, 'output_root')).parent
			/ 'visualizations'
			/ 'mae_debug'
		)
	elif isinstance(output_dir_value, str):
		output_dir = Path(output_dir_value)
	else:
		msg = (
			'visualization.mae_debug.output_dir must be a string or null; '
			f'got {output_dir_value!r}'
		)
		raise TypeError(msg)

	return MaeDebugVisualizationConfig(
		output_dir=output_dir,
		attributes=_string_tuple_config(
			mae_debug,
			'attributes',
			default=tuple(MVP_ATTRIBUTE_REGISTRY.names),
		),
		every_n_steps=_optional_int_config(mae_debug, 'every_n_steps'),
		every_n_epochs=_optional_int_config(mae_debug, 'every_n_epochs'),
		max_batches_per_trigger=_int_config(
			mae_debug,
			'max_batches_per_trigger',
			1,
		),
		max_samples_per_batch=_int_config(
			mae_debug,
			'max_samples_per_batch',
			1,
		),
		fail_on_error=_bool_config(mae_debug, 'fail_on_error', default=True),
		columns=_string_tuple_config(
			mae_debug,
			'columns',
			default=(
				'input',
				'masked_input',
				'target',
				'prediction',
				'abs_error',
			),
		),
		xy_slice_index=_optional_any_int_config(mae_debug, 'xy_slice_index'),
		xz_slice_y_index=_optional_any_int_config(mae_debug, 'xz_slice_y_index'),
		grid_mode=_str_config_with_default(mae_debug, 'grid_mode', 'auto'),
		dpi=_int_config(mae_debug, 'dpi', 160),
		panel_width=_float_config(mae_debug, 'panel_width', 3.2),
		panel_height=_float_config(mae_debug, 'panel_height', 2.8),
		clip_percentiles=_float_pair_config(
			mae_debug,
			'clip_percentiles',
			default=(1.0, 99.0),
		),
		use_known_ranges=_bool_config(mae_debug, 'use_known_ranges', default=True),
		mask_invalid_values=_bool_config(
			mae_debug,
			'mask_invalid_values',
			default=True,
		),
		show_valid_mask_panel=_bool_config(
			mae_debug,
			'show_valid_mask_panel',
			default=True,
		),
		show_spatial_mask_panel=_bool_config(
			mae_debug,
			'show_spatial_mask_panel',
			default=True,
		),
		invalid_color=_str_config_with_default(
			mae_debug,
			'invalid_color',
			'lightgray',
		),
	)


def _mae_debug_epoch_triggered(
	*,
	config: MaeDebugVisualizationConfig,
	epoch: int,
	epoch_visualization_batches: int,
) -> bool:
	if config.every_n_epochs is None:
		return False
	return (
		epoch % config.every_n_epochs == 0
		and epoch_visualization_batches < config.max_batches_per_trigger
	)


def _mae_debug_step_triggered(
	*,
	config: MaeDebugVisualizationConfig,
	global_step: int,
) -> bool:
	if config.every_n_steps is None:
		return False
	return global_step % config.every_n_steps == 0


def _save_mae_debug_visualization(  # noqa: PLR0913
	*,
	batch: Mapping[str, torch.Tensor | object],
	model_output: Mapping[str, torch.Tensor | object],
	patch_size_xyz: tuple[int, int, int],
	epoch: int,
	global_step: int,
	config: MaeDebugVisualizationConfig,
) -> None:
	try:
		save_mae_debug_visualization_pngs(
			batch=batch,
			model_output=model_output,
			patch_size_xyz=patch_size_xyz,
			epoch=epoch,
			global_step=global_step,
			config=config,
			max_samples=config.max_samples_per_batch,
		)
	except Exception as exc:
		if config.fail_on_error:
			raise
		warnings.warn(
			f'MAE debug visualization failed at epoch={epoch} '
			f'global_step={global_step}: {exc}',
			RuntimeWarning,
			stacklevel=2,
		)


def _save_mae_checkpoint(  # noqa: PLR0913
	path: Path,
	*,
	model: torch.nn.Module,
	optimizer: torch.optim.Optimizer,
	epoch: int,
	config: Mapping[str, object],
	metrics: Mapping[str, float],
	global_step: int,
	amp_enabled: bool,
	scaler: torch.amp.GradScaler | None,
	checkpoint_kind: Literal['step', 'epoch'],
	batch_index: int | None,
) -> Path:
	checkpoint_path = save_checkpoint(
		path,
		model=model,
		optimizer=optimizer,
		epoch=epoch,
		config=config,
		package_version=getattr(seis_attr_ssl, '__version__', None),
		metrics=metrics,
		global_step=global_step,
		amp_enabled=amp_enabled,
		scaler=scaler,
		training_state={
			'schema_version': 1,
			'checkpoint_kind': checkpoint_kind,
			'batch_index': batch_index,
		},
	)
	shutil.copy2(checkpoint_path, checkpoint_path.parent / 'mae_latest.pt')
	return checkpoint_path


def _restore_mae_checkpoint(
	*,
	payload: Mapping[str, object],
	model: torch.nn.Module,
	optimizer: torch.optim.Optimizer,
	scaler: torch.amp.GradScaler | None,
	amp_enabled: bool,
) -> tuple[int, int]:
	_validate_resume_payload(payload)
	model.load_state_dict(payload['model_state_dict'])
	optimizer.load_state_dict(payload['optimizer_state_dict'])
	if amp_enabled and payload.get('scaler_state_dict') is not None:
		if scaler is None:
			msg = 'scaler is required when amp_enabled is true'
			raise ValueError(msg)
		scaler.load_state_dict(payload['scaler_state_dict'])
	return int(payload['epoch']) + 1, int(payload.get('global_step', 0))


def _validate_resume_payload(payload: Mapping[str, object]) -> None:
	if 'model_state_dict' not in payload:
		msg = 'resume checkpoint is missing model_state_dict'
		raise ValueError(msg)
	if 'optimizer_state_dict' not in payload:
		msg = 'resume checkpoint is missing optimizer_state_dict'
		raise ValueError(msg)
	stage = _checkpoint_stage(payload)
	if stage is not None and stage != 'pretrain_mae':
		msg = f'resume checkpoint stage must be pretrain_mae; got {stage!r}'
		raise ValueError(msg)


def _checkpoint_stage(payload: Mapping[str, object]) -> object | None:
	if 'stage' in payload:
		return payload.get('stage')
	config = payload.get('config')
	if isinstance(config, Mapping):
		return config.get('stage')
	return None


def _build_model(
	model_config: Mapping[str, object],
	config: Mapping[str, object],
) -> StrictAttributeSetMAE3D:
	data_config = _mapping(config, 'data')
	return StrictAttributeSetMAE3D(
		num_attributes=len(MVP_ATTRIBUTE_REGISTRY.specs),
		attribute_groups=MVP_ATTRIBUTE_REGISTRY.groups,
		patch_size_xyz=_xyz_config(model_config, 'patch_size'),
		encoder_dim=_int_config(model_config, 'encoder_dim', 384),
		encoder_depth=_int_config(model_config, 'encoder_depth', 8),
		encoder_heads=_int_config(model_config, 'encoder_heads', 6),
		decoder_dim=_int_config(model_config, 'decoder_dim', 256),
		decoder_depth=_int_config(model_config, 'decoder_depth', 4),
		decoder_heads=_int_config(model_config, 'decoder_heads', 4),
		num_context_tokens=_int_config(model_config, 'num_context_tokens', 8),
		context_token_min_valid_fraction=_fraction_config(
			model_config,
			'context_token_min_valid_fraction',
			0.5,
		),
		use_context=_bool_config(data_config, 'use_context', default=True),
	)


def _manifest_train_path(config: Mapping[str, object]) -> Path:
	manifests = config.get('manifests')
	if not isinstance(manifests, Mapping):
		msg = _manifest_path_error('manifests.train is required')
		raise TypeError(msg)
	if 'train' not in manifests:
		msg = _manifest_path_error('manifests.train is required')
		raise ValueError(msg)
	path_value = manifests.get('train')
	if not isinstance(path_value, str) or not path_value:
		msg = _manifest_path_error(
			f'manifests.train must be a non-empty string; got {path_value!r}',
		)
		raise ValueError(msg)
	path = Path(path_value)
	if not path.is_file():
		msg = _manifest_path_error(f'manifests.train does not exist: {path}')
		raise FileNotFoundError(msg)
	return path


def _manifest_path_error(reason: str) -> str:
	return f'{reason}. {_MANIFEST_BUILD_HINT}'


def _resolve_device(train_config: Mapping[str, object]) -> torch.device:
	device_name = train_config.get('device')
	if device_name is None or device_name == 'auto':
		return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	if not isinstance(device_name, str):
		msg = f'train.device must be a string; got {device_name!r}'
		raise TypeError(msg)
	if device_name not in {'cpu', 'cuda'}:
		msg = 'train.device must be "auto", "cpu", or "cuda"'
		raise ValueError(msg)
	device = torch.device(device_name)
	if device.type == 'cuda' and not torch.cuda.is_available():
		msg = 'train.device requested CUDA, but CUDA is not available'
		raise ValueError(msg)
	return device


def _resolve_diagnostics_dir(
	train_config: Mapping[str, object],
	output_root: Path,
) -> Path:
	value = train_config.get('diagnostics_dir')
	if value is None:
		return output_root / 'diagnostics'
	if not isinstance(value, str):
		msg = f'train.diagnostics_dir must be a string; got {value!r}'
		raise TypeError(msg)
	path = Path(value)
	if path.is_absolute():
		return path
	return output_root / path


def _validate_no_f3_pretraining_config(value: object, path: str = 'config') -> None:
	if isinstance(value, Mapping):
		for key, child in value.items():
			key_text = str(key)
			if key_text.lower() in {'f3', 'f3_root', 'f3_dataset', 'f3_data'}:
				msg = (
					'F3 settings are not allowed in pretraining config: '
					f'{path}.{key_text}'
				)
				raise ValueError(msg)
			_validate_no_f3_pretraining_config(child, f'{path}.{key_text}')
	elif isinstance(value, list | tuple):
		for index, child in enumerate(value):
			_validate_no_f3_pretraining_config(child, f'{path}[{index}]')
	elif isinstance(value, str) and _looks_like_f3_path(value):
		msg = f'F3 paths are not allowed in pretraining config: {path}'
		raise ValueError(msg)


def _looks_like_f3_path(value: str) -> bool:
	parts = [part for part in re.split(r'[\\/]+', value.lower()) if part]
	return any(part == 'f3' or part.startswith('f3_') for part in parts)


def _check_finite_grad_norm(  # noqa: PLR0913
	*,
	grad_norm: torch.Tensor,
	model: torch.nn.Module,
	global_step: int,
	epoch: int,
	batch_index: int,
	batch: Mapping[str, object],
	output: Mapping[str, object],
	losses: Mapping[str, torch.Tensor],
	amp_enabled: bool,
	diagnostics_dir: Path | None,
) -> None:
	if _is_finite_grad_norm(grad_norm):
		return

	if diagnostics_dir is not None:
		diagnostic = build_mae_nonfinite_diagnostic(
			global_step=global_step,
			epoch=epoch,
			batch_index=batch_index,
			batch=batch,
			output=output,
			losses=losses,
			torch_amp_enabled=amp_enabled,
		)
		diagnostic['grad_norm'] = summarize_tensor(grad_norm)
		diagnostic['gradients'] = _summarize_gradients(model)
		diagnostic_path = write_json_diagnostic(
			diagnostic,
			diagnostics_dir / f'nonfinite_mae_grad_step_{global_step:08d}.json',
		)
		msg = (
			f'non-finite MAE gradient norm at epoch {epoch}, step {global_step}; '
			f'diagnostic written to {diagnostic_path}'
		)
	else:
		msg = f'non-finite MAE gradient norm at epoch {epoch}, step {global_step}'
	raise FloatingPointError(msg)


def _is_finite_grad_norm(grad_norm: torch.Tensor) -> bool:
	return bool(torch.isfinite(grad_norm.detach()).all().cpu().item())


def _grad_norm_float(grad_norm: torch.Tensor) -> float:
	return float(grad_norm.detach().float().cpu().item())


def _summarize_gradients(model: torch.nn.Module) -> dict[str, object]:
	return {
		name: summarize_tensor(parameter.grad)
		for name, parameter in model.named_parameters()
	}


def _required_tensor(
	mapping: Mapping[str, object],
	key: str,
) -> torch.Tensor:
	value = mapping[key]
	if not isinstance(value, torch.Tensor):
		msg = f'{key} must be a torch.Tensor; got {type(value).__name__}'
		raise TypeError(msg)
	return value


def _optional_tensor(
	mapping: Mapping[str, object],
	key: str,
) -> torch.Tensor | None:
	value = mapping.get(key)
	if value is None:
		return None
	if not isinstance(value, torch.Tensor):
		msg = f'{key} must be a torch.Tensor when present; got {type(value).__name__}'
		raise TypeError(msg)
	return value


def _mapping(parent: Mapping[str, object], key: str) -> Mapping[str, object]:
	value = parent.get(key)
	if not isinstance(value, Mapping):
		msg = f'{key} must be a mapping'
		raise TypeError(msg)
	return value


def _str_config(parent: Mapping[str, object], key: str) -> str:
	value = parent.get(key)
	if not isinstance(value, str):
		msg = f'{key} must be a string; got {value!r}'
		raise TypeError(msg)
	return value


def _int_config(parent: Mapping[str, object], key: str, default: int) -> int:
	value = parent.get(key, default)
	if not isinstance(value, int) or isinstance(value, bool):
		msg = f'{key} must be an integer; got {value!r}'
		raise TypeError(msg)
	if value <= 0:
		msg = f'{key} must be positive; got {value!r}'
		raise ValueError(msg)
	return value


def _nonnegative_int_config(
	parent: Mapping[str, object],
	key: str,
	default: int,
) -> int:
	value = parent.get(key, default)
	if not isinstance(value, int) or isinstance(value, bool):
		msg = f'{key} must be an integer; got {value!r}'
		raise TypeError(msg)
	if value < 0:
		msg = f'{key} must be nonnegative; got {value!r}'
		raise ValueError(msg)
	return value


def _optional_int_config(parent: Mapping[str, object], key: str) -> int | None:
	value = parent.get(key)
	if value is None:
		return None
	return _int_config(parent, key, 1)


def _optional_any_int_config(parent: Mapping[str, object], key: str) -> int | None:
	value = parent.get(key)
	if value is None:
		return None
	if not isinstance(value, int) or isinstance(value, bool):
		msg = f'{key} must be an integer; got {value!r}'
		raise TypeError(msg)
	return value


def _float_config(parent: Mapping[str, object], key: str, default: float) -> float:
	value = parent.get(key, default)
	if not isinstance(value, float | int) or isinstance(value, bool):
		msg = f'{key} must be a float; got {value!r}'
		raise TypeError(msg)
	return float(value)


def _float_pair_config(
	parent: Mapping[str, object],
	key: str,
	*,
	default: tuple[float, float],
) -> tuple[float, float]:
	value = parent.get(key)
	if value is None:
		return default
	if (
		not isinstance(value, Sequence)
		or isinstance(value, str | bytes)
		or len(value) != 2
	):
		msg = f'{key} must be a length-2 float sequence; got {value!r}'
		raise TypeError(msg)
	left, right = value
	if (
		not isinstance(left, float | int)
		or isinstance(left, bool)
		or not isinstance(right, float | int)
		or isinstance(right, bool)
	):
		msg = f'{key} must be a length-2 float sequence; got {value!r}'
		raise TypeError(msg)
	return float(left), float(right)


def _optional_positive_float_config(
	parent: Mapping[str, object],
	key: str,
) -> float | None:
	value = parent.get(key)
	if value is None:
		return None
	if not isinstance(value, float | int) or isinstance(value, bool):
		msg = f'{key} must be a float; got {value!r}'
		raise TypeError(msg)
	number = float(value)
	if not math.isfinite(number) or number <= 0.0:
		msg = f'{key} must be finite and positive; got {value!r}'
		raise ValueError(msg)
	return number


def _fraction_config(parent: Mapping[str, object], key: str, default: float) -> float:
	value = _float_config(parent, key, default)
	if not 0.0 < value <= 1.0:
		msg = f'{key} must be in (0, 1]; got {value!r}'
		raise ValueError(msg)
	return value


def _bool_config(
	parent: Mapping[str, object],
	key: str,
	*,
	default: bool,
) -> bool:
	value = parent.get(key, default)
	if not isinstance(value, bool):
		msg = f'{key} must be a bool; got {value!r}'
		raise TypeError(msg)
	return value


def _str_config_with_default(
	parent: Mapping[str, object],
	key: str,
	default: str,
) -> str:
	value = parent.get(key, default)
	if not isinstance(value, str):
		msg = f'{key} must be a string; got {value!r}'
		raise TypeError(msg)
	return value


def _string_tuple_config(
	parent: Mapping[str, object],
	key: str,
	*,
	default: tuple[str, ...],
) -> tuple[str, ...]:
	value = parent.get(key)
	if value is None:
		return default
	if not isinstance(value, list | tuple) or not all(
		isinstance(item, str) for item in value
	):
		msg = f'{key} must be a sequence of strings; got {value!r}'
		raise TypeError(msg)
	return tuple(value)


def _xyz_config(parent: Mapping[str, object], key: str) -> tuple[int, int, int]:
	value = parent.get(key)
	if (
		not isinstance(value, list | tuple)
		or len(value) != 3
		or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
	):
		msg = f'{key} must be a length-3 integer sequence; got {value!r}'
		raise TypeError(msg)
	xyz = tuple(cast('tuple[int, int, int]', value))
	if any(item <= 0 for item in xyz):
		msg = f'{key} values must be positive; got {xyz!r}'
		raise ValueError(msg)
	return xyz


def _loss_mode(value: object) -> Literal['huber', 'mse']:
	if value not in {'huber', 'mse'}:
		msg = f'reconstruction must be "huber" or "mse"; got {value!r}'
		raise ValueError(msg)
	return cast('Literal["huber", "mse"]', value)


__all__ = [
	'MaeStepState',
	'MaeTrainingState',
	'run_mae_pretraining',
	'train_mae_one_epoch',
]
