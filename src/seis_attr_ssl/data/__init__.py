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

__all__ = [
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
]


def __getattr__(name: str) -> object:
	"""Lazily expose data schema objects."""
	if name in __all__:
		return getattr(import_module('seis_attr_ssl.data.schema'), name)
	msg = f'module {__name__!r} has no attribute {name!r}'
	raise AttributeError(msg)
