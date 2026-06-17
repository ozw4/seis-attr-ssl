"""Quality-control checks for survey normalization statistics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from seis_attr_ssl.data.normalization import (
	SurveyNormalizationStats,
	load_normalization_stats,
)

QC_SCHEMA_VERSION = 1

_EXCLUDE_REASON_ORDER = (
	'missing_stats',
	'invalid_stats',
	'non_finite_stats',
	'small_iqr',
	'large_norm_abs_max',
)
_FINITE_STAT_FIELDS = ('clip_low', 'clip_high', 'median', 'iqr', 'eps')
_STATS_FILENAME_SUFFIX = '.normalization_stats.json'


@dataclass(frozen=True)
class NormalizationStatsQcThresholds:
	"""Thresholds used to exclude suspect normalization statistics."""

	iqr_min: float = 1.0e-6
	norm_abs_max: float = 1.0e4


@dataclass(frozen=True)
class NormalizationStatsQcItem:
	"""QC outcome for one survey normalization stats file."""

	survey_id: str
	stats_path: Path
	source_path: Path | None
	status: Literal['pass', 'exclude']
	exclude_reasons: tuple[str, ...]
	non_finite_fields: tuple[str, ...]
	iqr: float | None
	norm_abs_max: float | None
	error: str | None = None


@dataclass(frozen=True)
class NormalizationStatsQcReport:
	"""QC report for a collection of survey normalization stats."""

	schema_version: int
	thresholds: NormalizationStatsQcThresholds
	items: tuple[NormalizationStatsQcItem, ...]

	def __post_init__(self) -> None:
		"""Keep report item order stable for deterministic downstream JSON."""
		items = tuple(
			sorted(self.items, key=lambda item: (item.survey_id, str(item.stats_path)))
		)
		object.__setattr__(self, 'items', items)


def evaluate_normalization_stats(
	stats: SurveyNormalizationStats,
	*,
	stats_path: str | Path,
	thresholds: NormalizationStatsQcThresholds,
) -> NormalizationStatsQcItem:
	"""Evaluate in-memory survey normalization stats against QC thresholds."""
	return _evaluate_loaded_normalization_stats(
		stats,
		stats_path=Path(stats_path),
		source_path=stats.source_path,
		thresholds=thresholds,
	)


def evaluate_normalization_stats_file(
	stats_path: str | Path,
	*,
	survey_id: str | None = None,
	source_path: str | Path | None = None,
	thresholds: NormalizationStatsQcThresholds,
) -> NormalizationStatsQcItem:
	"""Load and evaluate a stats JSON file without raising on QC failures."""
	path = Path(stats_path)
	resolved_source_path = None if source_path is None else Path(source_path)
	fallback_survey_id = survey_id or _survey_id_from_stats_path(path)

	if not path.exists():
		return _excluded_item(
			survey_id=fallback_survey_id,
			stats_path=path,
			source_path=resolved_source_path,
			exclude_reasons=('missing_stats',),
			error=f'normalization stats file does not exist: {path}',
		)

	try:
		stats = load_normalization_stats(path)
	except Exception as exc:  # noqa: BLE001 - QC must report invalid files, not stop.
		return _excluded_item(
			survey_id=fallback_survey_id,
			stats_path=path,
			source_path=resolved_source_path,
			exclude_reasons=('invalid_stats',),
			error=str(exc),
		)

	return _evaluate_loaded_normalization_stats(
		stats,
		stats_path=path,
		source_path=resolved_source_path or stats.source_path,
		thresholds=thresholds,
	)


def normalization_qc_report_to_dict(
	report: NormalizationStatsQcReport,
) -> dict[str, object]:
	"""Convert a normalization QC report to a strict-JSON-compatible dict."""
	counts: dict[str, int] = {
		'total': len(report.items),
		'passed': sum(item.status == 'pass' for item in report.items),
		'excluded': sum(item.status == 'exclude' for item in report.items),
	}
	for reason in _EXCLUDE_REASON_ORDER:
		counts[reason] = sum(reason in item.exclude_reasons for item in report.items)

	excluded_surveys = [
		item.survey_id for item in report.items if item.status == 'exclude'
	]
	return {
		'schema_version': report.schema_version,
		'thresholds': {
			'iqr_min': _json_float_or_none(report.thresholds.iqr_min),
			'norm_abs_max': _json_float_or_none(report.thresholds.norm_abs_max),
		},
		'counts': counts,
		'excluded_surveys': excluded_surveys,
		'surveys': [_qc_item_to_dict(item) for item in report.items],
	}


def _evaluate_loaded_normalization_stats(
	stats: SurveyNormalizationStats,
	*,
	stats_path: Path,
	source_path: Path | None,
	thresholds: NormalizationStatsQcThresholds,
) -> NormalizationStatsQcItem:
	non_finite_fields = [
		field
		for field in _FINITE_STAT_FIELDS
		if not math.isfinite(float(getattr(stats, field)))
	]
	iqr = stats.iqr if math.isfinite(stats.iqr) else None
	denominator = stats.iqr + stats.eps
	norm_abs_max = _compute_norm_abs_max(stats, denominator=denominator)
	if (
		norm_abs_max is None
		and math.isfinite(denominator)
		and denominator > 0.0
	):
		non_finite_fields.append('norm_abs_max')

	reasons: list[str] = []
	if non_finite_fields or not math.isfinite(denominator) or denominator <= 0.0:
		reasons.append('non_finite_stats')
	elif stats.iqr < thresholds.iqr_min:
		reasons.append('small_iqr')

	if norm_abs_max is None:
		if 'non_finite_stats' not in reasons:
			reasons.append('non_finite_stats')
	elif norm_abs_max > thresholds.norm_abs_max:
		reasons.append('large_norm_abs_max')

	exclude_reasons = tuple(
		reason for reason in _EXCLUDE_REASON_ORDER if reason in reasons
	)
	return NormalizationStatsQcItem(
		survey_id=stats.survey_id,
		stats_path=stats_path,
		source_path=source_path,
		status='exclude' if exclude_reasons else 'pass',
		exclude_reasons=exclude_reasons,
		non_finite_fields=tuple(non_finite_fields),
		iqr=iqr,
		norm_abs_max=norm_abs_max,
		error=None,
	)


def _compute_norm_abs_max(
	stats: SurveyNormalizationStats,
	*,
	denominator: float,
) -> float | None:
	if not math.isfinite(denominator) or denominator <= 0.0:
		return None
	norm_abs_max = max(
		abs((stats.clip_low - stats.median) / denominator),
		abs((stats.clip_high - stats.median) / denominator),
	)
	if not math.isfinite(norm_abs_max):
		return None
	return norm_abs_max


def _excluded_item(
	*,
	survey_id: str,
	stats_path: Path,
	source_path: Path | None,
	exclude_reasons: tuple[str, ...],
	error: str,
) -> NormalizationStatsQcItem:
	return NormalizationStatsQcItem(
		survey_id=survey_id,
		stats_path=stats_path,
		source_path=source_path,
		status='exclude',
		exclude_reasons=exclude_reasons,
		non_finite_fields=(),
		iqr=None,
		norm_abs_max=None,
		error=error,
	)


def _survey_id_from_stats_path(path: Path) -> str:
	name = path.name
	if name.endswith(_STATS_FILENAME_SUFFIX):
		return name[: -len(_STATS_FILENAME_SUFFIX)]
	return path.stem


def _qc_item_to_dict(item: NormalizationStatsQcItem) -> dict[str, object]:
	return {
		'survey_id': item.survey_id,
		'status': item.status,
		'exclude_reasons': list(item.exclude_reasons),
		'stats_path': str(item.stats_path),
		'source_path': None if item.source_path is None else str(item.source_path),
		'iqr': _json_float_or_none(item.iqr),
		'norm_abs_max': _json_float_or_none(item.norm_abs_max),
		'non_finite_fields': list(item.non_finite_fields),
		'error': item.error,
	}


def _json_float_or_none(value: float | None) -> float | None:
	if value is None or not math.isfinite(value):
		return None
	return float(value)


__all__ = [
	'NormalizationStatsQcItem',
	'NormalizationStatsQcReport',
	'NormalizationStatsQcThresholds',
	'evaluate_normalization_stats',
	'evaluate_normalization_stats_file',
	'normalization_qc_report_to_dict',
]
