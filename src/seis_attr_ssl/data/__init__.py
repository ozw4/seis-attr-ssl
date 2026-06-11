"""Data loading and sampling components."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from seis_attr_ssl.data.schema import (
		GRID_ORDER_XYZ,
		AttributeVolumeRecord,
		CropRequest,
		SurveyManifest,
		TensorLike,
		UnlabeledPretrainingSample,
		read_manifest_json,
		survey_manifest_from_dict,
		survey_manifest_to_dict,
		write_manifest_json,
	)
	from seis_attr_ssl.data.volume_store import (
		NpyMemmapVolumeStore,
		NpyVolumeInfo,
		inspect_npy_volume,
	)

__all__ = [
	'GRID_ORDER_XYZ',
	'AttributeVolumeRecord',
	'CropRequest',
	'SurveyManifest',
	'TensorLike',
	'UnlabeledPretrainingSample',
	'NpyMemmapVolumeStore',
	'NpyVolumeInfo',
	'inspect_npy_volume',
	'read_manifest_json',
	'survey_manifest_from_dict',
	'survey_manifest_to_dict',
	'write_manifest_json',
]

_SCHEMA_EXPORTS = {
	'GRID_ORDER_XYZ',
	'AttributeVolumeRecord',
	'CropRequest',
	'SurveyManifest',
	'TensorLike',
	'UnlabeledPretrainingSample',
	'read_manifest_json',
	'survey_manifest_from_dict',
	'survey_manifest_to_dict',
	'write_manifest_json',
}

_VOLUME_STORE_EXPORTS = {
	'NpyMemmapVolumeStore',
	'NpyVolumeInfo',
	'inspect_npy_volume',
}


def __getattr__(name: str) -> object:
	"""Lazily expose data schema objects."""
	if name in _SCHEMA_EXPORTS:
		return getattr(import_module('seis_attr_ssl.data.schema'), name)
	if name in _VOLUME_STORE_EXPORTS:
		return getattr(import_module('seis_attr_ssl.data.volume_store'), name)
	msg = f'module {__name__!r} has no attribute {name!r}'
	raise AttributeError(msg)
