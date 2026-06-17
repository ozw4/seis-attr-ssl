"""Filter survey manifests and path-lists from normalization stats QC."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from seis_attr_ssl.data.normalization_qc import (
	QC_SCHEMA_VERSION,
	NormalizationStatsQcReport,
	NormalizationStatsQcThresholds,
	evaluate_normalization_stats_file,
)
from seis_attr_ssl.data.path_list import make_survey_id_from_path

if TYPE_CHECKING:
	from collections.abc import Sequence

	from seis_attr_ssl.data.schema import SurveyManifest

_STATS_SUFFIX = '.normalization_stats.json'


@dataclass(frozen=True)
class FilteredManifestStatsQcResult:
	"""Clean manifest/path-list output derived from stats QC."""

	report: NormalizationStatsQcReport
	excluded_surveys: tuple[str, ...]
	clean_manifests: tuple[SurveyManifest, ...]
	clean_path_entries: tuple[str, ...]


def filter_manifests_by_stats_qc(
	manifests: Sequence[SurveyManifest],
	path_entries: Sequence[str],
	*,
	nopims_root: str | Path,
	thresholds: NormalizationStatsQcThresholds,
	stats_dir: str | Path | None = None,
) -> FilteredManifestStatsQcResult:
	"""Run normalization stats QC and keep only passing surveys."""
	manifest_by_survey = _manifest_by_survey_id(manifests)
	path_survey_ids = tuple(
		_survey_id_from_path_entry(entry, nopims_root) for entry in path_entries
	)
	unknown_survey_ids = sorted(set(path_survey_ids) - set(manifest_by_survey))
	if unknown_survey_ids:
		msg = (
			'path-list contains survey_id values not present in manifest: '
			f'{", ".join(unknown_survey_ids)}'
		)
		raise ValueError(msg)

	items = tuple(
		evaluate_normalization_stats_file(
			_stats_path_for_manifest(manifest, stats_dir=stats_dir),
			survey_id=manifest.survey_id,
			source_path=(
				None if manifest.base_seismic is None else manifest.base_seismic.path
			),
			thresholds=thresholds,
		)
		for manifest in manifests
	)
	report = NormalizationStatsQcReport(
		schema_version=QC_SCHEMA_VERSION,
		thresholds=thresholds,
		items=items,
	)
	excluded = frozenset(
		item.survey_id for item in report.items if item.status == 'exclude'
	)
	return FilteredManifestStatsQcResult(
		report=report,
		excluded_surveys=tuple(sorted(excluded)),
		clean_manifests=tuple(
			manifest for manifest in manifests if manifest.survey_id not in excluded
		),
		clean_path_entries=tuple(
			entry
			for entry, survey_id in zip(path_entries, path_survey_ids, strict=True)
			if survey_id not in excluded
		),
	)


def _manifest_by_survey_id(
	manifests: Sequence[SurveyManifest],
) -> dict[str, SurveyManifest]:
	manifest_by_survey: dict[str, SurveyManifest] = {}
	for manifest in manifests:
		if manifest.survey_id in manifest_by_survey:
			msg = f'duplicate manifest survey_id: {manifest.survey_id}'
			raise ValueError(msg)
		manifest_by_survey[manifest.survey_id] = manifest
	return manifest_by_survey


def _survey_id_from_path_entry(
	entry: str,
	nopims_root: str | Path,
) -> str:
	root = Path(nopims_root)
	path = Path(entry)
	if not path.is_absolute():
		path = root / path
	return make_survey_id_from_path(path, root)


def _stats_path_for_manifest(
	manifest: SurveyManifest,
	*,
	stats_dir: str | Path | None,
) -> Path:
	if manifest.base_seismic is None:
		msg = f'manifest {manifest.survey_id!r} has no base_seismic'
		raise ValueError(msg)

	if stats_dir is not None:
		stats_dir_path = Path(stats_dir)
		candidate = stats_dir_path / f'{manifest.survey_id}{_STATS_SUFFIX}'
		if candidate.exists():
			return candidate

	stats_path = manifest.base_seismic.normalization_stats_path
	if stats_path.is_absolute():
		return stats_path
	return manifest.root / stats_path


__all__ = [
	'FilteredManifestStatsQcResult',
	'filter_manifests_by_stats_qc',
]
