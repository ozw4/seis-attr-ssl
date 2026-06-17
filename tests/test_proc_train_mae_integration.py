from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from seis_attr_ssl.attributes import MVP_ATTRIBUTE_REGISTRY
from seis_attr_ssl.data import (
	AttributeVolumeRecord,
	SurveyManifest,
	write_manifest_json,
)
from seis_attr_ssl.training import load_checkpoint
from tests.helpers import run_python_proc


@pytest.mark.heavy
@pytest.mark.skipif(
	os.environ.get('SEIS_ATTR_SSL_RUN_HEAVY_TESTS') != '1',
	reason='heavy subprocess MAE training smoke test',
)
def test_train_mae_proc_one_step_cpu_run_writes_checkpoint(tmp_path: Path) -> None:
	manifest_path = _write_synthetic_manifest(tmp_path / 'survey')
	config_path = tmp_path / 'mae.yaml'
	output_root = tmp_path / 'runs'
	config = _synthetic_config(tmp_path, manifest_path)
	config['train']['epochs'] = 2
	config_path.write_text(
		yaml.safe_dump(config),
		encoding='utf-8',
	)

	result = run_python_proc(
		Path('proc/train_mae.py'),
		'--config',
		config_path,
		'--device',
		'cpu',
		'--max-steps',
		'1',
		'--output-root',
		output_root,
		timeout=120,
	)

	assert result.returncode == 0, result.stderr
	checkpoint_path = output_root / 'mae_epoch_0001.pt'
	assert checkpoint_path.is_file()
	assert f'checkpoint: {checkpoint_path}' in result.stdout
	checkpoint = load_checkpoint(checkpoint_path, map_location='cpu')
	assert checkpoint['epoch'] == 1
	assert checkpoint['config']['train']['device'] == 'cpu'
	assert checkpoint['config']['train']['max_steps'] == 1
	assert np.isfinite(checkpoint['metrics']['loss'])

	resume_result = run_python_proc(
		Path('proc/train_mae.py'),
		'--config',
		config_path,
		'--device',
		'cpu',
		'--max-steps',
		'2',
		'--output-root',
		output_root,
		'--resume',
		checkpoint_path,
		timeout=120,
	)

	assert resume_result.returncode == 0, resume_result.stderr
	resumed_checkpoint_path = output_root / 'mae_epoch_0002.pt'
	assert resumed_checkpoint_path.is_file()
	assert f'checkpoint: {resumed_checkpoint_path}' in resume_result.stdout
	resumed_checkpoint = load_checkpoint(resumed_checkpoint_path, map_location='cpu')
	assert resumed_checkpoint['epoch'] == 2
	assert resumed_checkpoint['global_step'] == 2
	assert resumed_checkpoint['config']['train']['max_steps'] == 2


def test_train_mae_proc_missing_manifest_explains_how_to_build(
	tmp_path: Path,
) -> None:
	config_path = tmp_path / 'mae.yaml'
	config_path.write_text(
		yaml.safe_dump(_synthetic_config(tmp_path, tmp_path / 'missing.json')),
		encoding='utf-8',
	)

	result = run_python_proc(
		Path('proc/train_mae.py'),
		'--config',
		config_path,
		'--device',
		'cpu',
		'--max-steps',
		'1',
	)

	assert result.returncode != 0
	assert 'manifests.train does not exist' in result.stderr
	assert 'proc/build_nopims_manifests.py' in result.stderr


def test_train_mae_proc_missing_resume_checkpoint_fails(tmp_path: Path) -> None:
	config_path = tmp_path / 'mae.yaml'
	config_path.write_text(
		yaml.safe_dump(_synthetic_config(tmp_path, tmp_path / 'missing.json')),
		encoding='utf-8',
	)

	result = run_python_proc(
		Path('proc/train_mae.py'),
		'--config',
		config_path,
		'--device',
		'cpu',
		'--max-steps',
		'1',
		'--resume',
		tmp_path / 'missing-checkpoint.pt',
	)

	assert result.returncode != 0
	assert 'resume checkpoint does not exist' in result.stderr


def test_train_mae_proc_missing_manifest_train_key_explains_how_to_build(
	tmp_path: Path,
) -> None:
	config = _synthetic_config(tmp_path, tmp_path / 'manifest.json')
	config['manifests'] = {}
	config_path = tmp_path / 'mae.yaml'
	config_path.write_text(yaml.safe_dump(config), encoding='utf-8')

	result = run_python_proc(
		Path('proc/train_mae.py'),
		'--config',
		config_path,
		'--device',
		'cpu',
		'--max-steps',
		'1',
	)

	assert result.returncode != 0
	assert 'manifests.train is required' in result.stderr
	assert 'proc/build_nopims_manifests.py' in result.stderr


def _write_synthetic_manifest(root: Path) -> Path:
	records: dict[str, AttributeVolumeRecord] = {}
	shape_xyz = (128, 128, 128)
	attribute_dir = root / 'attributes'
	attribute_dir.mkdir(parents=True, exist_ok=True)
	for spec in MVP_ATTRIBUTE_REGISTRY.specs[:4]:
		path = attribute_dir / f'{spec.name}.npy'
		array = np.lib.format.open_memmap(
			path,
			mode='w+',
			dtype='float32',
			shape=shape_xyz,
		)
		array[...] = float(spec.id + 1)
		del array
		records[spec.name] = AttributeVolumeRecord(
			survey_id='synthetic',
			attribute_name=spec.name,
			path=Path('attributes') / f'{spec.name}.npy',
			shape_xyz=shape_xyz,
			dtype='float32',
			grid_order=('x', 'y', 'z'),
			is_memmap_safe=True,
		)

	manifest = SurveyManifest(
		survey_id='synthetic',
		root=root,
		attribute_volumes=records,
		shape_xyz=shape_xyz,
	)
	manifest_path = root / 'manifest.json'
	write_manifest_json([manifest], manifest_path)
	return manifest_path


def _synthetic_config(tmp_path: Path, manifest_path: Path) -> dict[str, object]:
	return {
		'project': {'name': 'SeisAttrSSL', 'package': 'seis_attr_ssl'},
		'paths': {'nopims_root': str(tmp_path), 'output_root': str(tmp_path / 'runs')},
		'manifests': {'train': str(manifest_path)},
		'data': {
			'grid_order': ['x', 'y', 'z'],
			'volume_format': 'npy_memmap',
			'base_seismic_kind': 'dip_steered_median_filtered',
			'local_crop_size': [128, 128, 128],
			'context_crop_size': [512, 512, 512],
			'context_downsample': 4,
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
			'patch_size': [32, 32, 32],
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
			'device': 'auto',
			'seed': 7,
			'shuffle': False,
			'samples_per_epoch': 1,
		},
	}
