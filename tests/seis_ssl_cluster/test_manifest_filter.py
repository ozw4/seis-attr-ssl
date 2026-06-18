from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from seis_ssl_cluster.data import (
	make_survey_id_from_path,
	read_manifest_json,
	scan_nopims_amplitude_manifests_from_path_list,
	write_manifest_json,
)
from tests.helpers import run_python_proc


def test_synthetic_normalization_qc_integration_writes_clean_outputs(
	tmp_path: Path,
) -> None:
	nopims_root = tmp_path / 'NOPIMS'
	stable = _write_volume(
		nopims_root / 'stable' / 'base.npy',
		np.arange(64, dtype=np.float32).reshape((4, 4, 4)),
	)
	unstable = _write_volume(
		nopims_root / 'unstable' / 'base.npy',
		np.ones((4, 4, 4), dtype=np.float32),
	)
	source_split = tmp_path / 'artifacts' / 'registry' / 'splits' / 'nopims' / (
		'pretrain_v1'
	) / 'train_npy_paths.txt'
	source_split.parent.mkdir(parents=True, exist_ok=True)
	source_split.write_text(
		'stable/base.npy\nunstable/base.npy\n',
		encoding='utf-8',
	)
	stats_dir = (
		tmp_path
		/ 'artifacts'
		/ 'registry'
		/ 'normalization_stats'
		/ 'nopims'
		/ 'pretrain_v1'
	)
	manifest_path = (
		tmp_path
		/ 'artifacts'
		/ 'registry'
		/ 'manifests'
		/ 'nopims'
		/ 'pretrain_v1'
		/ 'nopims_amplitude_manifests.json'
	)
	result = scan_nopims_amplitude_manifests_from_path_list(
		nopims_root,
		source_split,
		stats_dir,
	)
	manifest_path.parent.mkdir(parents=True, exist_ok=True)
	write_manifest_json(result.manifests, manifest_path)
	original_manifest_text = manifest_path.read_text(encoding='utf-8')
	original_split_text = source_split.read_text(encoding='utf-8')

	prepare_config = _base_config('prepare_nopims_normalization_stats', nopims_root)
	prepare_config['manifests'] = {'train': str(manifest_path)}
	prepare_config['normalization'] = {
		'clipping_percentiles': [0.5, 99.5],
		'epsilon': 1.0e-6,
		'max_samples': 1000000,
		'seed': 42,
		'smooth_time_depth_trend_correction': False,
		'trace_wise_agc': False,
		'patch_wise_zscore': False,
	}
	prepare_config_path = tmp_path / 'prepare.yaml'
	prepare_config_path.write_text(yaml.safe_dump(prepare_config), encoding='utf-8')

	prepare = run_python_proc(
		Path('proc/seis_ssl_cluster/prepare_nopims_normalization_stats.py'),
		'--config',
		prepare_config_path,
	)

	assert prepare.returncode == 0, prepare.stderr
	assert all(
		manifest.amplitude.normalization_stats_path.is_file()
		for manifest in result.manifests
	)
	assert stats_dir in result.manifests[0].amplitude.normalization_stats_path.parents

	clean_manifest = (
		tmp_path
		/ 'artifacts'
		/ 'registry'
		/ 'manifests'
		/ 'nopims'
		/ 'pretrain_v1_clean'
		/ 'nopims_amplitude_manifests.json'
	)
	clean_split = (
		tmp_path
		/ 'artifacts'
		/ 'registry'
		/ 'splits'
		/ 'nopims'
		/ 'pretrain_v1_clean'
		/ 'train_npy_paths.txt'
	)
	qc_json = (
		tmp_path
		/ 'artifacts'
		/ 'registry'
		/ 'qc'
		/ 'nopims'
		/ 'pretrain_v1'
		/ 'normalization_stats_qc.json'
	)
	excluded = qc_json.parent / 'excluded_surveys.txt'
	filter_config = _base_config('filter_manifest_by_normalization_qc', nopims_root)
	filter_config['manifests'] = {
		'input': str(manifest_path),
		'output': str(clean_manifest),
	}
	filter_config['splits'] = {
		'input': str(source_split),
		'output': str(clean_split),
	}
	filter_config['qc'] = {
		'output_json': str(qc_json),
		'excluded_surveys': str(excluded),
		'min_iqr': 1.0e-4,
		'max_normalized_abs': 1.0e6,
	}
	filter_config_path = tmp_path / 'filter.yaml'
	filter_config_path.write_text(yaml.safe_dump(filter_config), encoding='utf-8')

	filtered = run_python_proc(
		Path('proc/seis_ssl_cluster/filter_manifest_by_normalization_qc.py'),
		'--config',
		filter_config_path,
	)

	assert filtered.returncode == 0, filtered.stderr
	unstable_id = make_survey_id_from_path(unstable, nopims_root)
	stable_id = make_survey_id_from_path(stable, nopims_root)
	report = json.loads(qc_json.read_text(encoding='utf-8'))
	assert report['source_manifest_path'] == str(manifest_path)
	assert report['source_split_path'] == str(source_split)
	assert report['per_survey_reason_codes'][unstable_id] == ['small_iqr']
	assert excluded.read_text(encoding='utf-8') == f'{unstable_id}\n'
	assert manifest_path.read_text(encoding='utf-8') == original_manifest_text
	assert source_split.read_text(encoding='utf-8') == original_split_text
	assert clean_split.read_text(encoding='utf-8') == 'stable/base.npy\n'
	assert [manifest.survey_id for manifest in read_manifest_json(clean_manifest)] == [
		stable_id,
	]


def _write_volume(path: Path, values: np.ndarray) -> Path:
	path.parent.mkdir(parents=True, exist_ok=True)
	np.save(path, values)
	return path


def _base_config(stage: str, nopims_root: Path) -> dict[str, object]:
	return {
		'stage': stage,
		'paths': {
			'nopims_root': str(nopims_root),
			'artifact_root': str(nopims_root.parent / 'artifacts'),
		},
		'data': {
			'grid_order': ['x', 'y', 'z'],
			'volume_format': 'npy_memmap',
			'input_channels': 1,
			'target_channels': 1,
			'use_context': False,
			'local_crop_size': [128, 128, 128],
		},
		'model': {
			'name': 'amp_mae3d',
			'in_channels': 1,
			'out_channels': 1,
			'patch_size': [8, 8, 8],
		},
		'masking': {
			'spatial_mask_ratio': 0.75,
			'spatial_mask_mode': 'block',
			'block_size_tokens': [2, 2, 2],
		},
		'train': {
			'batch_size': 4,
			'samples_per_epoch': 10000,
			'epochs': 100,
			'num_workers': 8,
			'amp': False,
		},
	}
