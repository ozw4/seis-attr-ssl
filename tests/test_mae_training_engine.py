from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from seis_attr_ssl.attributes import MVP_ATTRIBUTE_REGISTRY
from seis_attr_ssl.data import (
	AttributeVolumeRecord,
	SurveyManifest,
	write_manifest_json,
)
from seis_attr_ssl.data.pretrain_dataset import NopimsAttributePretrainDataset
from seis_attr_ssl.training import load_checkpoint
from seis_attr_ssl.training.collate import mae_collate_fn
from seis_attr_ssl.training.mae import run_mae_pretraining, train_mae_one_epoch


class _TinyMaeModel(torch.nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.weight = torch.nn.Parameter(torch.tensor(1.0))

	def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
		target = batch['target']
		pred = self.weight * torch.zeros(
			(
				target.shape[0],
				8,
				target.shape[1],
				8,
			),
			dtype=target.dtype,
			device=target.device,
		)
		return {'pred_patches': pred}


def _write_tiny_manifest(root: Path) -> Path:
	records: dict[str, AttributeVolumeRecord] = {}
	shape_xyz = (4, 4, 4)
	attribute_dir = root / 'attributes'
	attribute_dir.mkdir(parents=True, exist_ok=True)
	for spec in MVP_ATTRIBUTE_REGISTRY.specs:
		array = np.full(shape_xyz, float(spec.id + 1), dtype=np.float32)
		path = attribute_dir / f'{spec.name}.npy'
		np.save(path, array)
		records[spec.name] = AttributeVolumeRecord(
			survey_id='tiny',
			attribute_name=spec.name,
			path=Path('attributes') / f'{spec.name}.npy',
			shape_xyz=shape_xyz,
			dtype='float32',
			grid_order=('x', 'y', 'z'),
			is_memmap_safe=True,
		)

	manifest = SurveyManifest(
		survey_id='tiny',
		root=root,
		attribute_volumes=records,
		shape_xyz=shape_xyz,
	)
	manifest_path = root / 'manifest.json'
	write_manifest_json([manifest], manifest_path)
	return manifest_path


def _tiny_config(tmp_path: Path) -> dict[str, object]:
	manifest_path = _write_tiny_manifest(tmp_path / 'survey')
	return {
		'project': {'name': 'SeisAttrSSL', 'package': 'seis_attr_ssl'},
		'paths': {'nopims_root': str(tmp_path), 'output_root': str(tmp_path / 'runs')},
		'manifests': {'train': str(manifest_path)},
		'data': {
			'grid_order': ['x', 'y', 'z'],
			'volume_format': 'npy_memmap',
			'base_seismic_kind': 'dip_steered_median_filtered',
			'local_crop_size': [4, 4, 4],
			'context_crop_size': [4, 4, 4],
			'context_downsample': 1,
			'local_attribute_halo': [0, 0, 0],
			'context_attribute_halo': [0, 0, 0],
			'require_full_halo_inside_volume': True,
			'use_context': False,
		},
		'attributes': {
			'names': list(MVP_ATTRIBUTE_REGISTRY.names),
			'groups': dict(MVP_ATTRIBUTE_REGISTRY.groups),
		},
		'normalization': {
			'pre_attribute': {
				'clipping_percentiles': [0.5, 99.5],
				'center': 'median',
				'scale': 'iqr',
				'epsilon': 1.0e-6,
				'smooth_time_depth_trend_correction': False,
				'trace_wise_agc': False,
				'patch_wise_zscore': False,
			},
		},
		'stage': 'pretrain_mae',
		'masking': {
			'spatial_mask_ratio': 0.5,
			'spatial_mask_mode': 'block',
			'block_size_tokens': [1, 1, 1],
			'min_input_attributes': 2,
			'max_input_attributes': 4,
			'attribute_dropout_prob': 0.0,
			'group_dropout_prob': 0.0,
		},
		'model': {
			'name': 'strict_attribute_set_mae3d',
			'patch_size': [2, 2, 2],
			'encoder_dim': 16,
			'encoder_depth': 1,
			'encoder_heads': 4,
			'decoder_dim': 16,
			'decoder_depth': 1,
			'decoder_heads': 4,
			'num_context_tokens': 1,
		},
		'loss': {
			'reconstruction': 'mse',
			'huber_delta': 1.0,
			'dropped_attribute_weight': 0.25,
			'gradient_weight': 0.0,
			'family_balanced': True,
		},
		'train': {
			'batch_size': 1,
			'epochs': 1,
			'lr': 1.0e-4,
			'weight_decay': 0.0,
			'amp': False,
			'seed': 7,
			'device': 'cpu',
			'shuffle': False,
			'samples_per_epoch': 1,
		},
	}


def test_one_epoch_cpu_training_writes_loadable_checkpoint(tmp_path: Path) -> None:
	checkpoint_path = run_mae_pretraining(_tiny_config(tmp_path))

	assert checkpoint_path.is_file()
	checkpoint = load_checkpoint(checkpoint_path, map_location='cpu')
	assert checkpoint['epoch'] == 1
	assert checkpoint['global_step'] == 1
	assert checkpoint['amp_enabled'] is False
	assert checkpoint['scaler_state_dict'] is None
	assert checkpoint['training_state']['schema_version'] == 1
	assert checkpoint['training_state']['checkpoint_kind'] == 'epoch'
	assert checkpoint['training_state']['batch_index'] is None
	assert checkpoint['package_version'] == '0.1.0'
	assert checkpoint['model_state_dict']
	assert checkpoint['optimizer_state_dict']
	loss = checkpoint['metrics']['loss']
	assert np.isfinite(loss)


def test_step_interval_checkpoints_and_latest_pointer(tmp_path: Path) -> None:
	cfg = _tiny_config(tmp_path)
	cfg['train']['samples_per_epoch'] = 2
	cfg['train']['max_steps'] = 2
	cfg['train']['checkpoint_every_steps'] = 1

	checkpoint_path = run_mae_pretraining(cfg)

	output_root = Path(cfg['paths']['output_root'])
	step_1_path = output_root / 'mae_step_00000001.pt'
	step_2_path = output_root / 'mae_step_00000002.pt'
	latest_path = output_root / 'mae_latest.pt'
	assert checkpoint_path.name == 'mae_epoch_0001.pt'
	assert step_1_path.is_file()
	assert step_2_path.is_file()
	assert latest_path.is_file()

	step_2 = load_checkpoint(step_2_path, map_location='cpu')
	assert step_2['global_step'] == 2
	assert step_2['training_state']['schema_version'] == 1
	assert step_2['training_state']['checkpoint_kind'] == 'step'
	assert step_2['training_state']['batch_index'] == 1

	epoch_checkpoint = load_checkpoint(checkpoint_path, map_location='cpu')
	assert epoch_checkpoint['training_state']['checkpoint_kind'] == 'epoch'
	assert epoch_checkpoint['training_state']['batch_index'] is None

	latest = load_checkpoint(latest_path, map_location='cpu')
	assert latest['global_step'] == 2


def test_step_checkpoints_are_not_written_when_interval_is_unset(
	tmp_path: Path,
) -> None:
	cfg = _tiny_config(tmp_path)
	cfg['train']['samples_per_epoch'] = 2
	cfg['train']['max_steps'] = 2

	run_mae_pretraining(cfg)

	output_root = Path(cfg['paths']['output_root'])
	assert not list(output_root.glob('mae_step_*.pt'))
	assert (output_root / 'mae_latest.pt').is_file()


def test_run_mae_pretraining_resumes_from_checkpoint_epoch_boundary(
	tmp_path: Path,
) -> None:
	cfg = _tiny_config(tmp_path)
	checkpoint_path = run_mae_pretraining(cfg)

	resume_cfg = deepcopy(cfg)
	resume_cfg['train']['epochs'] = 2
	resumed_checkpoint_path = run_mae_pretraining(
		resume_cfg,
		resume=checkpoint_path,
	)

	checkpoint = load_checkpoint(resumed_checkpoint_path, map_location='cpu')
	assert resumed_checkpoint_path.name == 'mae_epoch_0002.pt'
	assert checkpoint['epoch'] == 2
	assert checkpoint['global_step'] == 2
	assert checkpoint['amp_enabled'] is False
	assert checkpoint['scaler_state_dict'] is None
	assert checkpoint['optimizer_state_dict']


def test_run_mae_pretraining_resume_validates_checkpoint_payload(
	tmp_path: Path,
) -> None:
	cfg = _tiny_config(tmp_path)
	checkpoint_path = run_mae_pretraining(cfg)
	payload = load_checkpoint(checkpoint_path, map_location='cpu')

	missing_model_path = tmp_path / 'missing-model.pt'
	missing_model = dict(payload)
	missing_model.pop('model_state_dict')
	torch.save(missing_model, missing_model_path)

	with pytest.raises(ValueError, match='model_state_dict'):
		run_mae_pretraining(cfg, resume=missing_model_path)

	missing_optimizer_path = tmp_path / 'missing-optimizer.pt'
	missing_optimizer = dict(payload)
	missing_optimizer.pop('optimizer_state_dict')
	torch.save(missing_optimizer, missing_optimizer_path)

	with pytest.raises(ValueError, match='optimizer_state_dict'):
		run_mae_pretraining(cfg, resume=missing_optimizer_path)

	wrong_stage_path = tmp_path / 'wrong-stage.pt'
	wrong_stage = dict(payload)
	wrong_stage['config'] = {**wrong_stage['config'], 'stage': 'finetune_f3'}
	torch.save(wrong_stage, wrong_stage_path)

	with pytest.raises(ValueError, match='pretrain_mae'):
		run_mae_pretraining(cfg, resume=wrong_stage_path)


def test_one_epoch_cpu_training_records_grad_norm_when_clipping_enabled(
	tmp_path: Path,
) -> None:
	cfg = _tiny_config(tmp_path)
	cfg['train']['grad_clip_norm'] = 1.0

	checkpoint_path = run_mae_pretraining(cfg)
	checkpoint = load_checkpoint(checkpoint_path, map_location='cpu')

	assert 'grad_norm' in checkpoint['metrics']
	assert np.isfinite(checkpoint['metrics']['grad_norm'])


def test_run_mae_pretraining_sets_dataset_epoch_each_epoch(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	calls: list[int] = []
	original = NopimsAttributePretrainDataset.set_epoch

	def spy_set_epoch(self: NopimsAttributePretrainDataset, epoch: int) -> None:
		calls.append(epoch)
		original(self, epoch)

	monkeypatch.setattr(NopimsAttributePretrainDataset, 'set_epoch', spy_set_epoch)
	cfg = _tiny_config(tmp_path)
	cfg['train']['epochs'] = 2

	run_mae_pretraining(cfg)

	assert calls == [0, 1]


def test_checkpoint_save_load_round_trip(tmp_path: Path) -> None:
	checkpoint_path = run_mae_pretraining(_tiny_config(tmp_path))

	checkpoint = load_checkpoint(checkpoint_path, map_location=torch.device('cpu'))

	assert checkpoint['config']['stage'] == 'pretrain_mae'
	assert checkpoint['config']['train']['epochs'] == 1
	assert checkpoint['metrics']['loss'] >= 0.0


def test_config_with_f3_path_or_key_is_rejected_for_pretraining(tmp_path: Path) -> None:
	cfg = _tiny_config(tmp_path)
	cfg['manifests']['train'] = str(tmp_path / 'F3' / 'manifest.json')

	with pytest.raises(ValueError, match='F3'):
		run_mae_pretraining(cfg)

	cfg = _tiny_config(tmp_path)
	cfg['paths']['f3_root'] = str(tmp_path / 'f3')

	with pytest.raises(ValueError, match='F3'):
		run_mae_pretraining(cfg)


def test_missing_manifest_train_path_explains_how_to_build(tmp_path: Path) -> None:
	cfg = deepcopy(_tiny_config(tmp_path))
	cfg.pop('manifests')

	with pytest.raises(TypeError, match=r'proc/build_nopims_manifests\.py'):
		run_mae_pretraining(cfg)

	cfg = deepcopy(_tiny_config(tmp_path))
	cfg['manifests'] = {}

	with pytest.raises(ValueError, match=r'manifests\.train is required'):
		run_mae_pretraining(cfg)


def test_amp_flag_on_cpu_does_not_enable_cuda_amp(tmp_path: Path) -> None:
	cfg = deepcopy(_tiny_config(tmp_path))
	cfg['train']['amp'] = True

	checkpoint_path = run_mae_pretraining(cfg)
	checkpoint = load_checkpoint(checkpoint_path, map_location='cpu')

	assert checkpoint['metrics']['amp_enabled'] == 0.0
	assert checkpoint['amp_enabled'] is False
	assert checkpoint['scaler_state_dict'] is None


def test_nonfinite_mae_loss_writes_diagnostic_json(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	def nan_loss(**_: object) -> dict[str, torch.Tensor]:
		return {
			'loss': torch.tensor(float('nan')),
			'loss_reconstruction': torch.tensor(float('nan')),
			'loss_dropped_attribute': torch.tensor(0.0),
			'loss_gradient': torch.tensor(float('inf')),
		}

	monkeypatch.setattr('seis_attr_ssl.training.mae.mae_pretraining_loss', nan_loss)
	sample = _mae_sample(
		attribute_ids=(0, 2),
		coords={
			'survey_id': 'survey-a',
			'local_start_xyz': (1, 2, 3),
			'local_compute_start_xyz': (1, 2, 3),
			'context_compute_start_xyz': None,
		},
	)
	dataloader = torch.utils.data.DataLoader(
		[sample],
		batch_size=1,
		collate_fn=mae_collate_fn,
	)
	model = _TinyMaeModel()
	optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
	diagnostics_dir = tmp_path / 'diagnostics'

	with pytest.raises(FloatingPointError, match='diagnostic written to'):
		train_mae_one_epoch(
			model=model,
			dataloader=dataloader,
			optimizer=optimizer,
			device=torch.device('cpu'),
			epoch=3,
			patch_size_xyz=(2, 2, 2),
			loss_config={'reconstruction': 'mse'},
			global_step=1042,
			diagnostics_dir=diagnostics_dir,
		)

	diagnostic_path = diagnostics_dir / 'nonfinite_mae_step_00001042.json'
	text = diagnostic_path.read_text(encoding='utf-8')
	assert 'NaN' not in text
	assert 'Infinity' not in text

	payload = json.loads(text)
	assert payload['global_step'] == 1042
	assert payload['epoch'] == 3
	assert payload['batch_index'] == 0
	assert payload['coords'] == [
		{
			'survey_id': 'survey-a',
			'local_start_xyz': [1, 2, 3],
			'local_compute_start_xyz': [1, 2, 3],
			'context_compute_start_xyz': None,
		},
	]
	assert payload['attribute_ids'] == [[0, 2]]
	assert set(payload['losses']) >= {
		'loss',
		'loss_reconstruction',
		'loss_dropped_attribute',
		'loss_gradient',
	}
	assert payload['losses']['loss'] == {
		'value': None,
		'finite': False,
		'repr': 'nan',
	}
	assert set(payload['tensors']) >= {
		'x',
		'target',
		'context',
		'pred_patches',
		'target_valid',
		'spatial_mask',
		'dropped_attribute_mask',
	}


def test_run_mae_pretraining_resolves_diagnostics_dir(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	def nan_loss(**_: object) -> dict[str, torch.Tensor]:
		return {
			'loss': torch.tensor(float('nan')),
			'loss_reconstruction': torch.tensor(float('nan')),
			'loss_dropped_attribute': torch.tensor(0.0),
			'loss_gradient': torch.tensor(0.0),
		}

	monkeypatch.setattr('seis_attr_ssl.training.mae.mae_pretraining_loss', nan_loss)
	cases = (
		('default', None, Path('runs') / 'diagnostics'),
		('relative', 'custom-diagnostics', Path('runs') / 'custom-diagnostics'),
		(
			'absolute',
			str(tmp_path / 'absolute-diagnostics'),
			tmp_path / 'absolute-diagnostics',
		),
	)
	for case_name, diagnostics_dir, expected_dir in cases:
		case_root = tmp_path / case_name
		cfg = _tiny_config(case_root)
		if diagnostics_dir is not None:
			cfg['train']['diagnostics_dir'] = diagnostics_dir
		expected_path = (
			expected_dir
			if expected_dir.is_absolute()
			else case_root / expected_dir
		) / 'nonfinite_mae_step_00000000.json'

		with pytest.raises(FloatingPointError, match='diagnostic written to'):
			run_mae_pretraining(cfg)

		assert expected_path.is_file()


def test_grad_clip_norm_calls_torch_clip_on_cpu(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	calls: list[float] = []

	def fake_clip_grad_norm_(
		parameters: object,
		max_norm: float,
	) -> torch.Tensor:
		list(parameters)
		calls.append(max_norm)
		return torch.tensor(0.25)

	monkeypatch.setattr(torch.nn.utils, 'clip_grad_norm_', fake_clip_grad_norm_)
	sample = _mae_sample(
		attribute_ids=(0, 2),
		coords={
			'survey_id': 'survey-a',
			'local_start_xyz': (1, 2, 3),
			'local_compute_start_xyz': (1, 2, 3),
			'context_compute_start_xyz': None,
		},
	)
	dataloader = torch.utils.data.DataLoader(
		[sample],
		batch_size=1,
		collate_fn=mae_collate_fn,
	)
	model = _TinyMaeModel()
	optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

	state = train_mae_one_epoch(
		model=model,
		dataloader=dataloader,
		optimizer=optimizer,
		device=torch.device('cpu'),
		epoch=1,
		patch_size_xyz=(2, 2, 2),
		loss_config={'reconstruction': 'mse'},
		grad_clip_norm=1.0,
	)

	assert calls == [1.0]
	assert state.metrics['grad_norm'] == pytest.approx(0.25)


def _mae_sample(
	*,
	attribute_ids: tuple[int, ...],
	coords: dict[str, object],
) -> dict[str, object]:
	channel_count = len(attribute_ids)
	target_channel_count = len(MVP_ATTRIBUTE_REGISTRY.specs)
	return {
		'x': np.ones((channel_count, 4, 4, 4), dtype=np.float32),
		'target': np.ones((target_channel_count, 4, 4, 4), dtype=np.float32),
		'attribute_ids': np.asarray(attribute_ids, dtype=np.int64),
		'spatial_mask': np.ones((2, 2, 2), dtype=np.bool_),
		'visible_spatial_mask': np.zeros((2, 2, 2), dtype=np.bool_),
		'attribute_input_mask': np.ones(target_channel_count, dtype=np.bool_),
		'attribute_target_mask': np.ones(target_channel_count, dtype=np.bool_),
		'dropped_attribute_mask': np.zeros(
			target_channel_count,
			dtype=np.bool_,
		),
		'target_valid': np.ones(target_channel_count, dtype=np.bool_),
		'coords': coords,
	}
