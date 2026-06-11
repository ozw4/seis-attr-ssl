"""Typed data contracts for survey manifests and dataset samples."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, TypedDict, cast

from seis_attr_ssl.attributes import MVP_ATTRIBUTE_REGISTRY, AttributeRegistry

GRID_ORDER_XYZ: tuple[str, str, str] = ('x', 'y', 'z')
BASE_SEISMIC_KIND_DIP_STEERED_MEDIAN_FILTERED = 'dip_steered_median_filtered'
BASE_SEISMIC_DTYPE_FLOAT32 = 'float32'
TensorLike: TypeAlias = object


@dataclass(frozen=True)
class BaseSeismicVolumeRecord:
	"""Manifest record for one source seismic volume used for on-the-fly attributes."""

	survey_id: str
	path: Path
	kind: str
	shape_xyz: tuple[int, int, int]
	dtype: str
	grid_order: tuple[str, str, str]
	normalization_stats_path: Path

	def validate(self) -> None:
		"""Validate fixed MVP base-seismic metadata."""
		_validate_grid_order(self.grid_order, 'base_seismic.grid_order')
		_validate_shape_xyz(self.shape_xyz, 'base_seismic.shape_xyz')
		if self.path.suffix != '.npy':
			msg = f'base_seismic_path must point to a .npy file: {self.path}'
			raise ValueError(msg)
		if self.kind != BASE_SEISMIC_KIND_DIP_STEERED_MEDIAN_FILTERED:
			msg = (
				'base_seismic_kind must be '
				f'{BASE_SEISMIC_KIND_DIP_STEERED_MEDIAN_FILTERED!r}; got '
				f'{self.kind!r}'
			)
			raise ValueError(msg)
		if self.dtype != BASE_SEISMIC_DTYPE_FLOAT32:
			msg = (
				f'base_seismic dtype must be {BASE_SEISMIC_DTYPE_FLOAT32!r}; '
				f'got {self.dtype!r}'
			)
			raise ValueError(msg)


@dataclass(frozen=True)
class AttributeVolumeRecord:
	"""Manifest record for one generated seismic attribute volume."""

	survey_id: str
	attribute_name: str
	path: Path
	shape_xyz: tuple[int, int, int]
	dtype: str
	grid_order: tuple[str, str, str]
	is_memmap_safe: bool


@dataclass(frozen=True)
class SurveyManifest:
	"""Attribute volume manifest for one survey."""

	survey_id: str
	root: Path
	attribute_volumes: dict[str, AttributeVolumeRecord]
	shape_xyz: tuple[int, int, int]
	grid_order: tuple[str, str, str] = GRID_ORDER_XYZ
	base_seismic: BaseSeismicVolumeRecord | None = None

	def has_all_mvp_attributes(
		self,
		registry: AttributeRegistry = MVP_ATTRIBUTE_REGISTRY,
	) -> bool:
		"""Return whether all registry attributes are present."""
		return not self.missing_attributes(registry)

	def missing_attributes(
		self,
		registry: AttributeRegistry = MVP_ATTRIBUTE_REGISTRY,
	) -> tuple[str, ...]:
		"""Return missing registry attributes in stable registry order."""
		if self.base_seismic is not None:
			return ()
		return tuple(
			name for name in registry.names if name not in self.attribute_volumes
		)

	def get_attribute(self, name: str) -> AttributeVolumeRecord:
		"""Return an attribute volume record by name."""
		try:
			return self.attribute_volumes[name]
		except KeyError as exc:
			msg = f'unknown manifest attribute: {name!r}'
			raise KeyError(msg) from exc

	def validate_consistent_shapes(self) -> None:
		"""Validate that manifest records match manifest shape and grid order."""
		_validate_grid_order(self.grid_order, 'manifest.grid_order')
		_validate_shape_xyz(self.shape_xyz, 'manifest.shape_xyz')
		if self.base_seismic is not None:
			self.base_seismic.validate()
			if self.base_seismic.shape_xyz != self.shape_xyz:
				msg = (
					f'base seismic shape {self.base_seismic.shape_xyz!r} does not '
					f'match manifest shape {self.shape_xyz!r}'
				)
				raise ValueError(msg)
			if self.base_seismic.grid_order != self.grid_order:
				msg = (
					f'base seismic grid order {self.base_seismic.grid_order!r} does '
					f'not match manifest grid order {self.grid_order!r}'
				)
				raise ValueError(msg)
		for name, record in self.attribute_volumes.items():
			if record.shape_xyz != self.shape_xyz:
				msg = (
					f'attribute {name!r} shape {record.shape_xyz!r} does not match '
					f'manifest shape {self.shape_xyz!r}'
				)
				raise ValueError(msg)
			if record.grid_order != self.grid_order:
				msg = (
					f'attribute {name!r} grid order {record.grid_order!r} does not '
					f'match manifest grid order {self.grid_order!r}'
				)
				raise ValueError(msg)


@dataclass(frozen=True)
class CropRequest:
	"""Spatial request for a local crop and optional context crop."""

	survey_id: str
	start_xyz: tuple[int, int, int]
	size_xyz: tuple[int, int, int]
	context_size_xyz: tuple[int, int, int] | None
	context_downsample: int


class UnlabeledPretrainingSample(TypedDict):
	"""Dataset sample contract for unlabeled external pretraining."""

	x: TensorLike
	target: TensorLike
	attribute_ids: TensorLike
	target_attribute_ids: TensorLike
	valid_attributes: TensorLike
	target_valid: TensorLike
	coords: dict[str, object]
	context: TensorLike | None
	context_valid_mask: TensorLike | None
	local_valid_mask: TensorLike


def survey_manifest_to_dict(manifest: SurveyManifest) -> dict[str, object]:
	"""Convert a survey manifest to a JSON-compatible dictionary."""
	payload: dict[str, object] = {
		'survey_id': manifest.survey_id,
		'root': str(manifest.root),
		'shape_xyz': list(manifest.shape_xyz),
		'grid_order': list(manifest.grid_order),
	}
	if manifest.base_seismic is not None:
		payload.update(_base_seismic_record_to_manifest_fields(manifest.base_seismic))
	if manifest.attribute_volumes:
		payload['attribute_volumes'] = {
			name: _attribute_record_to_dict(record)
			for name, record in manifest.attribute_volumes.items()
		}
	return payload


def survey_manifest_from_dict(data: Mapping[str, object]) -> SurveyManifest:
	"""Build a survey manifest from a dictionary loaded from JSON."""
	attribute_volumes = _optional_mapping(data, 'attribute_volumes')
	records = {
		name: _attribute_record_from_dict(_require_nested_mapping(raw, name))
		for name, raw in attribute_volumes.items()
	}
	base_seismic = _base_seismic_record_from_manifest_fields(data)
	manifest = SurveyManifest(
		survey_id=_require_str(data, 'survey_id'),
		root=Path(str(data.get('root', '.'))),
		attribute_volumes=records,
		shape_xyz=_require_int_tuple3(data, 'shape_xyz'),
		grid_order=_require_str_tuple3(data, 'grid_order'),
		base_seismic=base_seismic,
	)
	manifest.validate_consistent_shapes()
	if not manifest.attribute_volumes and manifest.base_seismic is None:
		msg = (
			f'survey {manifest.survey_id!r} must define base seismic metadata or '
			'attribute volumes'
		)
		raise ValueError(msg)
	return manifest


def write_manifest_json(manifests: Sequence[SurveyManifest], path: Path) -> None:
	"""Write survey manifests to a JSON file."""
	payload = [survey_manifest_to_dict(manifest) for manifest in manifests]
	path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def read_manifest_json(path: Path) -> list[SurveyManifest]:
	"""Read survey manifests from a JSON file."""
	data = json.loads(path.read_text(encoding='utf-8'))
	if not isinstance(data, list):
		msg = f'manifest JSON must contain a list; got {type(data).__name__}'
		raise TypeError(msg)
	return [
		survey_manifest_from_dict(_require_nested_mapping(item, 'manifest'))
		for item in data
	]


def _attribute_record_to_dict(record: AttributeVolumeRecord) -> dict[str, object]:
	return {
		'survey_id': record.survey_id,
		'attribute_name': record.attribute_name,
		'path': str(record.path),
		'shape_xyz': list(record.shape_xyz),
		'dtype': record.dtype,
		'grid_order': list(record.grid_order),
		'is_memmap_safe': record.is_memmap_safe,
	}


def _base_seismic_record_to_manifest_fields(
	record: BaseSeismicVolumeRecord,
) -> dict[str, object]:
	return {
		'base_seismic_path': str(record.path),
		'base_seismic_kind': record.kind,
		'dtype': record.dtype,
		'normalization_stats_path': str(record.normalization_stats_path),
	}


def _base_seismic_record_from_manifest_fields(
	data: Mapping[str, object],
) -> BaseSeismicVolumeRecord | None:
	if 'base_seismic_path' not in data:
		return None
	record = BaseSeismicVolumeRecord(
		survey_id=_require_str(data, 'survey_id'),
		path=Path(_require_str(data, 'base_seismic_path')),
		kind=_require_str(data, 'base_seismic_kind'),
		shape_xyz=_require_int_tuple3(data, 'shape_xyz'),
		dtype=_require_str(data, 'dtype'),
		grid_order=_require_str_tuple3(data, 'grid_order'),
		normalization_stats_path=Path(_require_str(data, 'normalization_stats_path')),
	)
	record.validate()
	return record


def _attribute_record_from_dict(data: Mapping[str, object]) -> AttributeVolumeRecord:
	return AttributeVolumeRecord(
		survey_id=_require_str(data, 'survey_id'),
		attribute_name=_require_str(data, 'attribute_name'),
		path=Path(_require_str(data, 'path')),
		shape_xyz=_require_int_tuple3(data, 'shape_xyz'),
		dtype=_require_str(data, 'dtype'),
		grid_order=_require_str_tuple3(data, 'grid_order'),
		is_memmap_safe=_require_bool(data, 'is_memmap_safe'),
	)


def _optional_mapping(data: Mapping[str, object], key: str) -> Mapping[str, object]:
	if key not in data:
		return {}
	return _require_mapping(data, key)


def _require_mapping(data: Mapping[str, object], key: str) -> Mapping[str, object]:
	value = data[key]
	if not isinstance(value, Mapping):
		msg = f'{key!r} must be a mapping; got {type(value).__name__}'
		raise TypeError(msg)
	return cast('Mapping[str, object]', value)


def _require_nested_mapping(value: object, label: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		msg = f'{label!r} must be a mapping; got {type(value).__name__}'
		raise TypeError(msg)
	return cast('Mapping[str, object]', value)


def _require_str(data: Mapping[str, object], key: str) -> str:
	value = data[key]
	if not isinstance(value, str):
		msg = f'{key!r} must be a string; got {type(value).__name__}'
		raise TypeError(msg)
	return value


def _require_bool(data: Mapping[str, object], key: str) -> bool:
	value = data[key]
	if not isinstance(value, bool):
		msg = f'{key!r} must be a bool; got {type(value).__name__}'
		raise TypeError(msg)
	return value


def _require_int_tuple3(
	data: Mapping[str, object],
	key: str,
) -> tuple[int, int, int]:
	value = data[key]
	if (
		not isinstance(value, Sequence)
		or isinstance(value, str)
		or len(value) != 3
		or not all(isinstance(item, int) for item in value)
	):
		msg = f'{key!r} must be a length-3 integer sequence; got {value!r}'
		raise TypeError(msg)
	xyz = cast('tuple[int, int, int]', tuple(value))
	_validate_shape_xyz(xyz, key)
	return xyz


def _require_str_tuple3(
	data: Mapping[str, object],
	key: str,
) -> tuple[str, str, str]:
	value = data[key]
	if (
		not isinstance(value, Sequence)
		or isinstance(value, str)
		or len(value) != 3
		or not all(isinstance(item, str) for item in value)
	):
		msg = f'{key!r} must be a length-3 string sequence; got {value!r}'
		raise TypeError(msg)
	grid_order = cast('tuple[str, str, str]', tuple(value))
	_validate_grid_order(grid_order, key)
	return grid_order


def _validate_shape_xyz(shape_xyz: tuple[int, int, int], label: str) -> None:
	if any(axis <= 0 for axis in shape_xyz):
		msg = f'{label} values must be positive; got {shape_xyz!r}'
		raise ValueError(msg)


def _validate_grid_order(grid_order: tuple[str, str, str], label: str) -> None:
	if grid_order != GRID_ORDER_XYZ:
		msg = f'{label} must be {GRID_ORDER_XYZ!r}; got {grid_order!r}'
		raise ValueError(msg)


__all__ = [
	'BASE_SEISMIC_KIND_DIP_STEERED_MEDIAN_FILTERED',
	'GRID_ORDER_XYZ',
	'AttributeVolumeRecord',
	'BaseSeismicVolumeRecord',
	'CropRequest',
	'SurveyManifest',
	'TensorLike',
	'UnlabeledPretrainingSample',
	'read_manifest_json',
	'survey_manifest_from_dict',
	'survey_manifest_to_dict',
	'write_manifest_json',
]
