"""Stage 1 strict MAE pretraining engine."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import torch

import seis_attr_ssl
from seis_attr_ssl.attributes import MVP_ATTRIBUTE_REGISTRY
from seis_attr_ssl.data import NopimsAttributePretrainDataset, read_manifest_json
from seis_attr_ssl.losses import mae_pretraining_loss
from seis_attr_ssl.models.mae import StrictAttributeSetMAE3D
from seis_attr_ssl.training.checkpoint import save_checkpoint
from seis_attr_ssl.training.collate import move_batch_to_device
from seis_attr_ssl.training.dataloaders import build_mae_dataloader
from seis_attr_ssl.training.diagnostics import (
	build_mae_nonfinite_diagnostic,
	write_json_diagnostic,
)
from seis_attr_ssl.training.logging import print_epoch_metrics

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


def train_mae_one_epoch(  # noqa: PLR0913
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
) -> MaeTrainingState:
	"""Train ``model`` for one epoch and return averaged loss metrics."""
	model.train()
	totals: dict[str, float] = {}
	batches = 0

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

		if amp_enabled:
			if scaler is None:
				msg = 'scaler is required when amp_enabled is true'
				raise ValueError(msg)
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()
		else:
			loss.backward()
			optimizer.step()

		for key, value in losses.items():
			totals[key] = totals.get(key, 0.0) + float(value.detach().cpu().item())
		batches += 1
		global_step += 1

	if batches == 0:
		msg = 'dataloader produced no batches'
		raise ValueError(msg)

	return MaeTrainingState(
		epoch=epoch,
		global_step=global_step,
		metrics={key: total / batches for key, total in totals.items()},
		amp_enabled=amp_enabled,
	)


def run_mae_pretraining(config: Mapping[str, object]) -> Path:
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

	output_root = Path(_str_config(paths_config, 'output_root'))
	diagnostics_dir = _resolve_diagnostics_dir(train_config, output_root)
	epochs = _int_config(train_config, 'epochs', 1)
	max_steps = _optional_int_config(train_config, 'max_steps')
	state: MaeTrainingState | None = None
	checkpoint_path: Path | None = None
	for epoch in range(1, epochs + 1):
		set_epoch = getattr(dataset, 'set_epoch', None)
		if callable(set_epoch):
			set_epoch(epoch - 1)
		remaining_steps = None
		if max_steps is not None:
			previous_steps = 0 if state is None else state.global_step
			remaining_steps = max_steps - previous_steps
			if remaining_steps <= 0:
				break
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
			global_step=0 if state is None else state.global_step,
			max_steps=remaining_steps,
			diagnostics_dir=diagnostics_dir,
		)
		print_epoch_metrics(epoch, state.metrics)
		checkpoint_path = save_checkpoint(
			output_root / f'mae_epoch_{epoch:04d}.pt',
			model=model,
			optimizer=optimizer,
			epoch=epoch,
			config=config,
			package_version=getattr(seis_attr_ssl, '__version__', None),
			metrics={**state.metrics, 'amp_enabled': float(state.amp_enabled)},
		)
		if max_steps is not None and state.global_step >= max_steps:
			break

	if checkpoint_path is None:
		msg = 'train.epochs must be positive'
		raise ValueError(msg)
	return checkpoint_path


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


def _required_tensor(
	mapping: Mapping[str, object],
	key: str,
) -> torch.Tensor:
	value = mapping[key]
	if not isinstance(value, torch.Tensor):
		msg = f'{key} must be a torch.Tensor; got {type(value).__name__}'
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


def _float_config(parent: Mapping[str, object], key: str, default: float) -> float:
	value = parent.get(key, default)
	if not isinstance(value, float | int) or isinstance(value, bool):
		msg = f'{key} must be a float; got {value!r}'
		raise TypeError(msg)
	return float(value)


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
	'MaeTrainingState',
	'run_mae_pretraining',
	'train_mae_one_epoch',
]
