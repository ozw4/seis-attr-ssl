from __future__ import annotations

import json
from pathlib import Path

import pytest

from seis_attr_ssl.data.normalization import (
	SurveyNormalizationStats,
	write_normalization_stats,
)
from seis_attr_ssl.data.normalization_qc import (
	NormalizationStatsQcReport,
	NormalizationStatsQcThresholds,
	evaluate_normalization_stats,
	evaluate_normalization_stats_file,
	normalization_qc_report_to_dict,
)


def _stats(  # noqa: PLR0913
	*,
	survey_id: str = 'survey-a',
	clip_low: float = -2.0,
	clip_high: float = 6.0,
	median: float = 2.0,
	iqr: float = 2.0,
	eps: float = 1.0e-6,
) -> SurveyNormalizationStats:
	return SurveyNormalizationStats(
		survey_id=survey_id,
		source_path=Path(f'{survey_id}.npy'),
		grid_order=('x', 'y', 'z'),
		clip_low_percentile=0.5,
		clip_high_percentile=99.5,
		clip_low=clip_low,
		clip_high=clip_high,
		median=median,
		iqr=iqr,
		eps=eps,
	)


def test_evaluate_normalization_stats_passes_valid_stats() -> None:
	item = evaluate_normalization_stats(
		_stats(),
		stats_path='survey-a.normalization_stats.json',
		thresholds=NormalizationStatsQcThresholds(),
	)

	assert item.status == 'pass'
	assert item.exclude_reasons == ()
	assert item.non_finite_fields == ()
	assert item.iqr == pytest.approx(2.0)
	assert item.norm_abs_max == pytest.approx(2.0)


def test_evaluate_normalization_stats_excludes_small_iqr() -> None:
	item = evaluate_normalization_stats(
		_stats(clip_low=-1.0e-5, clip_high=1.0e-5, median=0.0, iqr=1.0e-8),
		stats_path='survey-a.normalization_stats.json',
		thresholds=NormalizationStatsQcThresholds(iqr_min=1.0e-6),
	)

	assert item.status == 'exclude'
	assert item.exclude_reasons == ('small_iqr',)


def test_evaluate_stats_reports_small_iqr_and_large_norm_abs_max() -> None:
	item = evaluate_normalization_stats(
		_stats(clip_low=-1.0, clip_high=1.0, median=0.0, iqr=1.0e-8, eps=1.0e-8),
		stats_path='survey-a.normalization_stats.json',
		thresholds=NormalizationStatsQcThresholds(
			iqr_min=1.0e-6,
			norm_abs_max=1.0e4,
		),
	)

	assert item.status == 'exclude'
	assert item.exclude_reasons == ('small_iqr', 'large_norm_abs_max')
	assert item.norm_abs_max == pytest.approx(5.0e7)


def test_evaluate_normalization_stats_excludes_non_finite_stats() -> None:
	item = evaluate_normalization_stats(
		_stats(median=float('nan')),
		stats_path='survey-a.normalization_stats.json',
		thresholds=NormalizationStatsQcThresholds(),
	)

	assert item.status == 'exclude'
	assert item.exclude_reasons == ('non_finite_stats',)
	assert item.non_finite_fields == ('median', 'norm_abs_max')
	assert item.norm_abs_max is None


def test_evaluate_stats_file_reports_missing_stats(tmp_path: Path) -> None:
	path = tmp_path / 'missing.normalization_stats.json'

	item = evaluate_normalization_stats_file(
		path,
		survey_id='survey-a',
		source_path=tmp_path / 'survey-a.npy',
		thresholds=NormalizationStatsQcThresholds(),
	)

	assert item.status == 'exclude'
	assert item.survey_id == 'survey-a'
	assert item.source_path == tmp_path / 'survey-a.npy'
	assert item.exclude_reasons == ('missing_stats',)
	assert item.error is not None


def test_evaluate_stats_file_reports_invalid_stats(tmp_path: Path) -> None:
	path = tmp_path / 'survey-a.normalization_stats.json'
	path.write_text('{"survey_id": ', encoding='utf-8')

	item = evaluate_normalization_stats_file(
		path,
		survey_id='survey-a',
		thresholds=NormalizationStatsQcThresholds(),
	)

	assert item.status == 'exclude'
	assert item.exclude_reasons == ('invalid_stats',)
	assert item.error is not None


def test_normalization_qc_report_to_dict_is_strict_json_compatible(
	tmp_path: Path,
) -> None:
	valid_path = tmp_path / 'survey-a.normalization_stats.json'
	invalid_path = tmp_path / 'survey-b.normalization_stats.json'
	write_normalization_stats(_stats(survey_id='survey-a'), valid_path)
	write_normalization_stats(
		_stats(survey_id='survey-b', median=float('nan')),
		invalid_path,
	)
	items = (
		evaluate_normalization_stats_file(
			invalid_path,
			thresholds=NormalizationStatsQcThresholds(),
		),
		evaluate_normalization_stats_file(
			valid_path,
			thresholds=NormalizationStatsQcThresholds(),
		),
	)
	report = NormalizationStatsQcReport(
		schema_version=1,
		thresholds=NormalizationStatsQcThresholds(),
		items=items,
	)

	report_dict = normalization_qc_report_to_dict(report)

	json.dumps(report_dict, allow_nan=False)
	assert report_dict['excluded_surveys'] == ['survey-b']
	assert report_dict['counts'] == {
		'total': 2,
		'passed': 1,
		'excluded': 1,
		'missing_stats': 0,
		'invalid_stats': 0,
		'non_finite_stats': 1,
		'small_iqr': 0,
		'large_norm_abs_max': 0,
	}
	assert [survey['survey_id'] for survey in report_dict['surveys']] == [
		'survey-a',
		'survey-b',
	]


def test_report_counts_and_excluded_surveys_are_deterministic() -> None:
	thresholds = NormalizationStatsQcThresholds(iqr_min=1.0e-6, norm_abs_max=1.0e4)
	items = (
		evaluate_normalization_stats(
			_stats(survey_id='survey-c', median=float('nan')),
			stats_path='survey-c.normalization_stats.json',
			thresholds=thresholds,
		),
		evaluate_normalization_stats(
			_stats(survey_id='survey-a'),
			stats_path='survey-a.normalization_stats.json',
			thresholds=thresholds,
		),
		evaluate_normalization_stats(
			_stats(
				survey_id='survey-b',
				clip_low=-1.0,
				clip_high=1.0,
				median=0.0,
				iqr=1.0e-8,
				eps=1.0e-8,
			),
			stats_path='survey-b.normalization_stats.json',
			thresholds=thresholds,
		),
	)
	report = NormalizationStatsQcReport(
		schema_version=1,
		thresholds=thresholds,
		items=items,
	)

	report_dict = normalization_qc_report_to_dict(report)

	assert report_dict['excluded_surveys'] == ['survey-b', 'survey-c']
	assert report_dict['counts'] == {
		'total': 3,
		'passed': 1,
		'excluded': 2,
		'missing_stats': 0,
		'invalid_stats': 0,
		'non_finite_stats': 1,
		'small_iqr': 1,
		'large_norm_abs_max': 1,
	}
