"""Build NOPIMS survey manifests from base seismic or attribute `.npy` volumes."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from seis_attr_ssl.attributes import MVP_ATTRIBUTE_REGISTRY, AttributeRegistry
from seis_attr_ssl.data.path_list import (
	make_survey_id_from_path,
	resolve_npy_path_list,
)
from seis_attr_ssl.data.schema import (
	BASE_SEISMIC_DTYPE_FLOAT32,
	BASE_SEISMIC_KIND_DIP_STEERED_MEDIAN_FILTERED,
	GRID_ORDER_XYZ,
	AttributeVolumeRecord,
	BaseSeismicVolumeRecord,
	SurveyManifest,
	write_manifest_json,
)
from seis_attr_ssl.data.volume_store import inspect_npy_volume


@dataclass(frozen=True)
class ManifestBuildSummary:
	"""Compact summary of a manifest scan."""

	survey_count: int
	attribute_volume_count: int
	missing_attributes_by_survey: dict[str, tuple[str, ...]]
	unknown_attribute_counts: dict[str, int]
	base_seismic_count: int = 0
	output_path: Path | None = None


@dataclass(frozen=True)
class ManifestBuildResult:
	"""Manifest scan result with ignored-file accounting."""

	manifests: list[SurveyManifest]
	unknown_attribute_counts: dict[str, int]

	def summary(
		self,
		registry: AttributeRegistry = MVP_ATTRIBUTE_REGISTRY,
		output_path: Path | None = None,
	) -> ManifestBuildSummary:
		"""Return aggregate counts and per-survey missing attributes."""
		return summarize_manifests(
			self.manifests,
			registry=registry,
			unknown_attribute_counts=self.unknown_attribute_counts,
			output_path=output_path,
		)


def build_nopims_manifests(
	nopims_root: Path,
	output_path: Path,
	scan_pattern: str,
	registry: AttributeRegistry = MVP_ATTRIBUTE_REGISTRY,
	*,
	require_all_attributes: bool = False,
) -> list[SurveyManifest]:
	"""Scan NOPIMS `.npy` attribute volumes and write a JSON manifest."""
	result = scan_nopims_manifests(
		nopims_root=nopims_root,
		scan_pattern=scan_pattern,
		registry=registry,
		require_all_attributes=require_all_attributes,
	)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	write_manifest_json(result.manifests, output_path)
	return result.manifests


def build_nopims_base_seismic_manifests(  # noqa: PLR0913
	nopims_root: Path,
	output_path: Path,
	scan_pattern: str,
	base_seismic_kind: str = BASE_SEISMIC_KIND_DIP_STEERED_MEDIAN_FILTERED,
	base_seismic_name_hints: tuple[str, ...] = ('dip', 'median', 'filtered'),
	normalization_stats_name: str = 'normalization_stats.json',
) -> list[SurveyManifest]:
	"""Scan NOPIMS base seismic `.npy` volumes and write a JSON manifest."""
	result = scan_nopims_base_seismic_manifests(
		nopims_root=nopims_root,
		scan_pattern=scan_pattern,
		base_seismic_kind=base_seismic_kind,
		base_seismic_name_hints=base_seismic_name_hints,
		normalization_stats_name=normalization_stats_name,
	)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	write_manifest_json(result.manifests, output_path)
	return result.manifests


def build_nopims_base_seismic_manifests_from_path_list(
	nopims_root: Path,
	output_path: Path,
	input_path_list: Path,
	base_seismic_kind: str = BASE_SEISMIC_KIND_DIP_STEERED_MEDIAN_FILTERED,
	normalization_stats_suffix: str = '.normalization_stats.json',
) -> list[SurveyManifest]:
	"""Build NOPIMS base seismic manifests from an explicit `.npy` path-list."""
	result = scan_nopims_base_seismic_manifests_from_path_list(
		nopims_root=nopims_root,
		input_path_list=input_path_list,
		base_seismic_kind=base_seismic_kind,
		normalization_stats_suffix=normalization_stats_suffix,
	)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	write_manifest_json(result.manifests, output_path)
	return result.manifests


def scan_nopims_manifests(
	nopims_root: Path,
	scan_pattern: str,
	registry: AttributeRegistry = MVP_ATTRIBUTE_REGISTRY,
	*,
	require_all_attributes: bool = False,
) -> ManifestBuildResult:
	"""Scan NOPIMS `.npy` attribute volumes without writing JSON."""
	root = Path(nopims_root)
	if not root.is_dir():
		msg = f'NOPIMS root does not exist or is not a directory: {root}'
		raise FileNotFoundError(msg)

	registry_names = set(registry.names)
	survey_records: dict[str, dict[str, AttributeVolumeRecord]] = {}
	survey_roots: dict[str, Path] = {}
	unknown_attributes: Counter[str] = Counter()

	for path in _iter_npy_paths(root, scan_pattern):
		relative = path.relative_to(root)
		if len(relative.parts) < 2:
			unknown_attributes[path.stem] += 1
			continue

		survey_id = relative.parts[0]
		attribute_name = _identify_attribute_name(relative, registry_names)
		if attribute_name is None:
			unknown_attributes[path.stem] += 1
			continue

		info = inspect_npy_volume(path)
		record = AttributeVolumeRecord(
			survey_id=survey_id,
			attribute_name=attribute_name,
			path=path,
			shape_xyz=info.shape_xyz,
			dtype=info.dtype,
			grid_order=GRID_ORDER_XYZ,
			is_memmap_safe=True,
		)
		records = survey_records.setdefault(survey_id, {})
		if attribute_name in records:
			msg = (
				f'duplicate attribute {attribute_name!r} for survey {survey_id!r}: '
				f'{records[attribute_name].path} and {path}'
			)
			raise ValueError(msg)
		records[attribute_name] = record
		survey_roots.setdefault(survey_id, root / survey_id)

	manifests = [
		_build_survey_manifest(
			survey_id=survey_id,
			survey_root=survey_roots[survey_id],
			records=records,
			registry=registry,
			require_all_attributes=require_all_attributes,
		)
		for survey_id, records in sorted(survey_records.items())
	]

	return ManifestBuildResult(
		manifests=manifests,
		unknown_attribute_counts=dict(sorted(unknown_attributes.items())),
	)


def scan_nopims_base_seismic_manifests(
	nopims_root: Path,
	scan_pattern: str,
	base_seismic_kind: str = BASE_SEISMIC_KIND_DIP_STEERED_MEDIAN_FILTERED,
	base_seismic_name_hints: tuple[str, ...] = ('dip', 'median', 'filtered'),
	normalization_stats_name: str = 'normalization_stats.json',
) -> ManifestBuildResult:
	"""Scan NOPIMS `.npy` base seismic volumes without writing JSON."""
	root = Path(nopims_root)
	if not root.is_dir():
		msg = f'NOPIMS root does not exist or is not a directory: {root}'
		raise FileNotFoundError(msg)

	survey_records: dict[str, BaseSeismicVolumeRecord] = {}
	unknown_bases: Counter[str] = Counter()
	hints = tuple(hint.lower() for hint in base_seismic_name_hints)
	for path in _iter_npy_paths(root, scan_pattern):
		relative = path.relative_to(root)
		if len(relative.parts) < 2 or not _matches_base_seismic_hints(relative, hints):
			unknown_bases[path.stem] += 1
			continue

		survey_id = relative.parts[0]
		if survey_id in survey_records:
			msg = (
				f'duplicate base seismic volume for survey {survey_id!r}: '
				f'{survey_records[survey_id].path} and {path}'
			)
			raise ValueError(msg)
		info = inspect_npy_volume(path)
		survey_root = root / survey_id
		survey_records[survey_id] = BaseSeismicVolumeRecord(
			survey_id=survey_id,
			path=path,
			kind=base_seismic_kind,
			shape_xyz=info.shape_xyz,
			dtype=info.dtype,
			grid_order=GRID_ORDER_XYZ,
			normalization_stats_path=survey_root / normalization_stats_name,
		)

	manifests = [
		SurveyManifest(
			survey_id=survey_id,
			root=root / survey_id,
			attribute_volumes={},
			shape_xyz=record.shape_xyz,
			base_seismic=record,
		)
		for survey_id, record in sorted(survey_records.items())
	]
	for manifest in manifests:
		manifest.validate_consistent_shapes()

	return ManifestBuildResult(
		manifests=manifests,
		unknown_attribute_counts=dict(sorted(unknown_bases.items())),
	)


def scan_nopims_base_seismic_manifests_from_path_list(
	nopims_root: Path,
	input_path_list: Path,
	base_seismic_kind: str = BASE_SEISMIC_KIND_DIP_STEERED_MEDIAN_FILTERED,
	normalization_stats_suffix: str = '.normalization_stats.json',
) -> ManifestBuildResult:
	"""Build base seismic manifests from a user-maintained `.npy` path-list."""
	root = Path(nopims_root)
	paths = resolve_npy_path_list(input_path_list, root)

	manifests: list[SurveyManifest] = []
	survey_ids: dict[str, Path] = {}
	for path in paths:
		survey_id = make_survey_id_from_path(path, root)
		if survey_id in survey_ids:
			msg = (
				f'duplicate generated survey_id {survey_id!r}: '
				f'{survey_ids[survey_id]} and {path}'
			)
			raise ValueError(msg)
		survey_ids[survey_id] = path

		info = inspect_npy_volume(path)
		if info.dtype != BASE_SEISMIC_DTYPE_FLOAT32:
			msg = (
				f'base seismic dtype must be {BASE_SEISMIC_DTYPE_FLOAT32!r}; '
				f'got {info.dtype!r}: {path}'
			)
			raise ValueError(msg)

		record = BaseSeismicVolumeRecord(
			survey_id=survey_id,
			path=path,
			kind=base_seismic_kind,
			shape_xyz=info.shape_xyz,
			dtype=info.dtype,
			grid_order=GRID_ORDER_XYZ,
			normalization_stats_path=_normalization_stats_path(
				path,
				normalization_stats_suffix,
			),
		)
		manifest = SurveyManifest(
			survey_id=survey_id,
			root=path.parent,
			attribute_volumes={},
			shape_xyz=record.shape_xyz,
			base_seismic=record,
		)
		manifest.validate_consistent_shapes()
		manifests.append(manifest)

	return ManifestBuildResult(
		manifests=manifests,
		unknown_attribute_counts={},
	)


def summarize_manifests(
	manifests: list[SurveyManifest],
	registry: AttributeRegistry = MVP_ATTRIBUTE_REGISTRY,
	unknown_attribute_counts: dict[str, int] | None = None,
	output_path: Path | None = None,
) -> ManifestBuildSummary:
	"""Summarize manifest coverage in deterministic survey and attribute order."""
	missing = {
		manifest.survey_id: manifest.missing_attributes(registry)
		for manifest in sorted(manifests, key=lambda item: item.survey_id)
		if manifest.missing_attributes(registry)
	}
	return ManifestBuildSummary(
		survey_count=len(manifests),
		attribute_volume_count=sum(
			len(manifest.attribute_volumes) for manifest in manifests
		),
		missing_attributes_by_survey=missing,
		unknown_attribute_counts=dict(sorted((unknown_attribute_counts or {}).items())),
		base_seismic_count=sum(
			1 for manifest in manifests if manifest.base_seismic is not None
		),
		output_path=output_path,
	)


def _iter_npy_paths(root: Path, scan_pattern: str) -> list[Path]:
	return sorted(
		(
			path
			for path in root.glob(scan_pattern)
			if path.is_file() and path.suffix == '.npy'
		),
		key=lambda path: path.relative_to(root).as_posix(),
	)


def _identify_attribute_name(
	relative_path: Path,
	registry_names: set[str],
) -> str | None:
	if relative_path.stem in registry_names:
		return relative_path.stem

	for part in reversed(relative_path.parts[1:-1]):
		if part in registry_names:
			return part
	return None


def _matches_base_seismic_hints(relative_path: Path, hints: tuple[str, ...]) -> bool:
	if not hints:
		return True
	text = relative_path.as_posix().lower()
	return all(hint in text for hint in hints)


def _normalization_stats_path(path: Path, suffix: str) -> Path:
	if not suffix.startswith('.'):
		msg = f'normalization_stats_suffix must start with ".": {suffix!r}'
		raise ValueError(msg)
	return path.with_suffix(suffix)


def _build_survey_manifest(
	survey_id: str,
	survey_root: Path,
	records: dict[str, AttributeVolumeRecord],
	registry: AttributeRegistry,
	*,
	require_all_attributes: bool,
) -> SurveyManifest:
	if not records:
		msg = f'survey {survey_id!r} has no known attribute volumes'
		raise ValueError(msg)

	ordered_records = {
		name: records[name] for name in registry.names if name in records
	}
	shape_xyz = next(iter(ordered_records.values())).shape_xyz
	manifest = SurveyManifest(
		survey_id=survey_id,
		root=survey_root,
		attribute_volumes=ordered_records,
		shape_xyz=shape_xyz,
	)
	manifest.validate_consistent_shapes()

	missing = manifest.missing_attributes(registry)
	if missing and require_all_attributes:
		msg = f'survey {survey_id!r} is missing required attributes: {missing!r}'
		raise ValueError(msg)

	return manifest


__all__ = [
	'ManifestBuildResult',
	'ManifestBuildSummary',
	'build_nopims_base_seismic_manifests',
	'build_nopims_base_seismic_manifests_from_path_list',
	'build_nopims_manifests',
	'scan_nopims_base_seismic_manifests',
	'scan_nopims_base_seismic_manifests_from_path_list',
	'scan_nopims_manifests',
	'summarize_manifests',
]
