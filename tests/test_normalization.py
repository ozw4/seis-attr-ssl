from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from seis_attr_ssl.config import load_config
from seis_attr_ssl.data.normalization import (
	SurveyNormalizationStats,
	compute_normalization_stats,
	load_normalization_stats,
	normalize_amplitude,
	write_normalization_stats,
)
from seis_attr_ssl.data.schema import (
	BaseSeismicVolumeRecord,
	SurveyManifest,
	write_manifest_json,
)
from tests.helpers import run_python_proc

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _stats() -> SurveyNormalizationStats:
	return SurveyNormalizationStats(
		survey_id='survey-a',
		source_path=Path('base.npy'),
		grid_order=('x', 'y', 'z'),
		clip_low_percentile=0.5,
		clip_high_percentile=99.5,
		clip_low=-2.0,
		clip_high=6.0,
		median=2.0,
		iqr=2.0,
		eps=1.0e-6,
	)


def test_normalize_amplitude_clips_and_robust_scales() -> None:
	crop = np.asarray([-10.0, -2.0, 2.0, 6.0, 10.0], dtype=np.float32)

	normalized = normalize_amplitude(crop, _stats())

	expected = np.asarray([-2.0, -2.0, 0.0, 2.0, 2.0], dtype=np.float32)
	np.testing.assert_allclose(normalized, expected, rtol=0.0, atol=1.0e-5)


def test_normalize_amplitude_preserves_xyz_shape_and_order() -> None:
	crop = np.arange(2 * 3 * 4, dtype=np.float32).reshape((2, 3, 4))
	stats = SurveyNormalizationStats(
		survey_id='survey-a',
		source_path=Path('base.npy'),
		grid_order=('x', 'y', 'z'),
		clip_low_percentile=0.5,
		clip_high_percentile=99.5,
		clip_low=0.0,
		clip_high=100.0,
		median=0.0,
		iqr=1.0,
		eps=1.0e-6,
	)

	normalized = normalize_amplitude(crop, stats)

	assert normalized.shape == (2, 3, 4)
	np.testing.assert_allclose(normalized[1, 2, 3], crop[1, 2, 3] / 1.000001)


def test_load_normalization_stats_reads_required_json_fields(tmp_path: Path) -> None:
	path = tmp_path / 'normalization_stats.json'
	stats = _stats()
	write_normalization_stats(stats, path)

	loaded = load_normalization_stats(path)

	assert loaded == SurveyNormalizationStats(
		survey_id='survey-a',
		source_path=Path('base.npy'),
		grid_order=('x', 'y', 'z'),
		clip_low_percentile=0.5,
		clip_high_percentile=99.5,
		clip_low=-2.0,
		clip_high=6.0,
		median=2.0,
		iqr=2.0,
		eps=1.0e-6,
	)


def test_compute_normalization_stats_samples_memmap_deterministically(
	tmp_path: Path,
) -> None:
	path = tmp_path / 'volume.npy'
	volume = np.arange(1000, dtype=np.float32).reshape((10, 10, 10))
	np.save(path, volume)

	first = compute_normalization_stats(
		path,
		survey_id='survey-a',
		max_samples=100,
		seed=7,
	)
	second = compute_normalization_stats(
		path,
		survey_id='survey-a',
		max_samples=100,
		seed=7,
	)
	other = compute_normalization_stats(
		path,
		survey_id='survey-a',
		max_samples=100,
		seed=8,
	)

	assert first == second
	assert first != other
	assert first.grid_order == ('x', 'y', 'z')
	assert first.source_path == path


def test_compute_normalization_stats_full_volume_values(tmp_path: Path) -> None:
	path = tmp_path / 'volume.npy'
	volume = np.arange(8, dtype=np.float32).reshape((2, 2, 2))
	np.save(path, volume)

	stats = compute_normalization_stats(
		path,
		survey_id='survey-a',
		max_samples=None,
	)

	assert stats.clip_low == np.percentile(volume, 0.5)
	assert stats.clip_high == np.percentile(volume, 99.5)
	assert stats.median == np.percentile(volume, 50.0)
	assert stats.iqr == np.percentile(volume, 75.0) - np.percentile(volume, 25.0)


def test_load_normalization_stats_rejects_legacy_center_scale(
	tmp_path: Path,
) -> None:
	path = tmp_path / 'normalization_stats.json'
	path.write_text(json.dumps({'center': 0.0, 'scale': 1.0}), encoding='utf-8')

	with pytest.raises(TypeError, match='survey_id'):
		load_normalization_stats(path)


def test_prepare_nopims_normalization_stats_cli_writes_sidecars(
	tmp_path: Path,
) -> None:
	manifest_path, config_path, stats_paths, source_paths = _write_bulk_stats_inputs(
		tmp_path,
	)

	result = run_python_proc(
		Path('proc/prepare_nopims_normalization_stats.py'),
		'--config',
		config_path,
	)

	assert result.returncode == 0, result.stderr
	assert f'normalization_stats.manifest_path: {manifest_path}' in result.stdout
	assert 'normalization_stats.written_files: 2' in result.stdout
	loaded = [load_normalization_stats(path) for path in stats_paths]
	assert [stats.survey_id for stats in loaded] == ['survey-a', 'survey-b']
	assert [stats.source_path for stats in loaded] == source_paths


def test_prepare_nopims_normalization_stats_skips_and_overwrites_existing(
	tmp_path: Path,
) -> None:
	_, config_path, stats_paths, source_paths = _write_bulk_stats_inputs(tmp_path)
	first_result = run_python_proc(
		Path('proc/prepare_nopims_normalization_stats.py'),
		'--config',
		config_path,
	)
	assert first_result.returncode == 0, first_result.stderr
	original = load_normalization_stats(stats_paths[0])
	np.save(source_paths[0], np.arange(8, 16, dtype=np.float32).reshape((2, 2, 2)))

	skip_result = run_python_proc(
		Path('proc/prepare_nopims_normalization_stats.py'),
		'--config',
		config_path,
	)

	assert skip_result.returncode == 0, skip_result.stderr
	assert 'normalization_stats.written_files: 0' in skip_result.stdout
	assert 'normalization_stats.skipped_existing_files: 2' in skip_result.stdout
	assert load_normalization_stats(stats_paths[0]) == original

	overwrite_result = run_python_proc(
		Path('proc/prepare_nopims_normalization_stats.py'),
		'--config',
		config_path,
		'--overwrite',
	)

	assert overwrite_result.returncode == 0, overwrite_result.stderr
	assert 'normalization_stats.written_files: 2' in overwrite_result.stdout
	updated = load_normalization_stats(stats_paths[0])
	assert updated != original
	assert updated.source_path == source_paths[0]


def test_prepare_nopims_normalization_stats_dry_run_reports_counts(
	tmp_path: Path,
) -> None:
	_, config_path, stats_paths, _ = _write_bulk_stats_inputs(tmp_path)
	write_normalization_stats(_stats(), stats_paths[0])

	result = run_python_proc(
		Path('proc/prepare_nopims_normalization_stats.py'),
		'--config',
		config_path,
		'--dry-run',
	)

	assert result.returncode == 0, result.stderr
	assert 'normalization_stats.manifest_entries: 2' in result.stdout
	assert 'normalization_stats.existing_files: 1' in result.stdout
	assert 'normalization_stats.missing_files: 1' in result.stdout
	assert 'normalization_stats.max_samples: 1000' in result.stdout
	assert 'normalization_stats.seed: 42' in result.stdout
	assert 'normalization_stats.overwrite: false' in result.stdout
	assert 'normalization_stats.compute: skipped' in result.stdout


def test_prepare_nopims_normalization_stats_dry_run_missing_manifest_is_actionable(
	tmp_path: Path,
) -> None:
	manifest_path = tmp_path / 'manifests' / 'missing.json'
	config_path = _write_missing_manifest_stats_config(tmp_path, manifest_path)

	result = run_python_proc(
		Path('proc/prepare_nopims_normalization_stats.py'),
		'--config',
		config_path,
		'--dry-run',
	)

	assert result.returncode == 0, result.stderr
	assert result.stderr == ''
	assert f'normalization_stats.manifest_path: {manifest_path}' in result.stdout
	assert 'normalization_stats.manifest_exists: false' in result.stdout
	assert 'normalization_stats.compute: skipped' in result.stdout
	assert 'manifest does not exist' in result.stdout
	assert 'proc/build_nopims_manifests.py' in result.stdout


def test_prepare_nopims_normalization_stats_missing_manifest_error_is_actionable(
	tmp_path: Path,
) -> None:
	manifest_path = tmp_path / 'manifests' / 'missing.json'
	config_path = _write_missing_manifest_stats_config(tmp_path, manifest_path)

	result = run_python_proc(
		Path('proc/prepare_nopims_normalization_stats.py'),
		'--config',
		config_path,
	)

	assert result.returncode != 0
	assert f'manifests.train does not exist: {manifest_path}' in result.stderr
	assert 'proc/build_nopims_manifests.py' in result.stderr


def _write_bulk_stats_inputs(
	tmp_path: Path,
) -> tuple[Path, Path, list[Path], list[Path]]:
	nopims_root = tmp_path / 'NOPIMS'
	first_path = _write_manifest_volume(
		nopims_root / 'survey-a' / 'seismic' / 'base.npy',
		np.arange(8, dtype=np.float32).reshape((2, 2, 2)),
	)
	second_path = _write_manifest_volume(
		nopims_root / 'survey-b' / 'seismic' / 'base.npy',
		np.arange(8, 16, dtype=np.float32).reshape((2, 2, 2)),
	)
	manifest_path = tmp_path / 'manifests' / 'nopims_base_seismic_manifests.json'
	manifest_path.parent.mkdir(parents=True, exist_ok=True)
	write_manifest_json(
		[
			_manifest('survey-a', first_path),
			_manifest('survey-b', second_path),
		],
		manifest_path,
	)

	config = load_config(PROJECT_ROOT / 'proc/configs/mvp_prepare_nopims_stats.yaml')
	config['paths']['nopims_root'] = str(nopims_root)
	config['manifests']['train'] = str(manifest_path)
	config['normalization_stats']['max_samples'] = 1000
	config_path = tmp_path / 'prepare_nopims_stats.yaml'
	config_path.write_text(yaml.safe_dump(config), encoding='utf-8')
	return (
		manifest_path,
		config_path,
		[
			first_path.with_suffix('.normalization_stats.json'),
			second_path.with_suffix('.normalization_stats.json'),
		],
		[first_path, second_path],
	)


def _write_missing_manifest_stats_config(tmp_path: Path, manifest_path: Path) -> Path:
	config = load_config(PROJECT_ROOT / 'proc/configs/mvp_prepare_nopims_stats.yaml')
	config['manifests']['train'] = str(manifest_path)
	config_path = tmp_path / 'prepare_nopims_stats.yaml'
	config_path.write_text(yaml.safe_dump(config), encoding='utf-8')
	return config_path


def _write_manifest_volume(path: Path, values: np.ndarray) -> Path:
	path.parent.mkdir(parents=True, exist_ok=True)
	np.save(path, values)
	return path


def _manifest(survey_id: str, path: Path) -> SurveyManifest:
	return SurveyManifest(
		survey_id=survey_id,
		root=path.parent,
		attribute_volumes={},
		shape_xyz=(2, 2, 2),
		base_seismic=BaseSeismicVolumeRecord(
			survey_id=survey_id,
			path=path,
			kind='dip_steered_median_filtered',
			shape_xyz=(2, 2, 2),
			dtype='float32',
			grid_order=('x', 'y', 'z'),
			normalization_stats_path=path.with_suffix(
				'.normalization_stats.json',
			),
		),
	)
