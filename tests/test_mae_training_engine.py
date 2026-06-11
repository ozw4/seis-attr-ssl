from __future__ import annotations

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
from seis_attr_ssl.training import load_checkpoint
from seis_attr_ssl.training.mae import run_mae_pretraining


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
	assert checkpoint['package_version'] == '0.1.0'
	assert checkpoint['model_state_dict']
	assert checkpoint['optimizer_state_dict']
	loss = checkpoint['metrics']['loss']
	assert np.isfinite(loss)


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
