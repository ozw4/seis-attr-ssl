from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from seis_attr_ssl.data.normalization import (
	SurveyNormalizationStats,
	compute_normalization_stats,
	load_normalization_stats,
	normalize_amplitude,
	write_normalization_stats,
)


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
