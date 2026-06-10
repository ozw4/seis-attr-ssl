"""Explicit validation for MVP configuration dictionaries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias, TypeVar

from seis_attr_ssl.config.schema import (
	DISALLOWED_PRETRAINING_KEYS,
	EXPECTED_ATTRIBUTE_GROUPS,
	EXPECTED_ATTRIBUTES,
	EXPECTED_CONTEXT_CROP_SIZE,
	EXPECTED_CONTEXT_DOWNSAMPLE,
	EXPECTED_GRID_ORDER,
	EXPECTED_LOCAL_CROP_SIZE,
	EXPECTED_VOLUME_FORMAT,
	F3_ALLOWED_STAGES,
	KNOWN_STAGES,
)

Config: TypeAlias = dict[str, object]
_T = TypeVar('_T', bound=Mapping[str, object])


def validate_config(config: _T) -> _T:
	"""Validate a loaded MVP configuration and return it unchanged."""
	if not isinstance(config, Mapping):
		msg = 'config must be a mapping'
		raise TypeError(msg)

	stage = _validate_stage(config)
	if stage not in F3_ALLOWED_STAGES:
		_reject_f3_pretraining_config(config)

	data = _required_mapping(config, 'data')
	_validate_equal(data, 'grid_order', EXPECTED_GRID_ORDER)
	_validate_equal(data, 'volume_format', EXPECTED_VOLUME_FORMAT)
	_validate_equal(data, 'local_crop_size', EXPECTED_LOCAL_CROP_SIZE)
	_validate_equal(data, 'context_crop_size', EXPECTED_CONTEXT_CROP_SIZE)
	_validate_equal(data, 'context_downsample', EXPECTED_CONTEXT_DOWNSAMPLE)
	_validate_context_downsample(data)

	attributes = _required_mapping(config, 'attributes')
	_validate_attributes(attributes)

	normalization = _required_mapping(config, 'normalization')
	pre_attribute = _required_mapping(normalization, 'pre_attribute')
	for key in (
		'smooth_time_depth_trend_correction',
		'trace_wise_agc',
		'patch_wise_zscore',
	):
		_validate_equal(
			pre_attribute,
			key,
			expected=False,
			prefix='normalization.pre_attribute',
		)

	paths = _required_mapping(config, 'paths')
	_validate_nopims_root(paths)

	return config


def _validate_stage(config: Mapping[str, object]) -> str:
	stage = config.get('stage')
	if stage not in KNOWN_STAGES:
		msg = f'stage must be one of {sorted(KNOWN_STAGES)!r}; got {stage!r}'
		raise ValueError(msg)
	return str(stage)


def _required_mapping(parent: Mapping[str, object], key: str) -> Mapping[str, object]:
	value = parent.get(key)
	if not isinstance(value, Mapping):
		msg = f'{key} must be a mapping'
		raise TypeError(msg)
	return value


def _validate_equal(
	parent: Mapping[str, object],
	key: str,
	expected: object,
	*,
	prefix: str = 'data',
) -> None:
	actual = parent.get(key)
	if actual != expected:
		msg = f'{prefix}.{key} must be {expected!r}; got {actual!r}'
		raise ValueError(msg)


def _validate_context_downsample(data: Mapping[str, object]) -> None:
	context_crop_size = data.get('context_crop_size')
	context_downsample = data.get('context_downsample')
	local_crop_size = data.get('local_crop_size')

	if not isinstance(context_crop_size, list) or not isinstance(local_crop_size, list):
		msg = 'data.context_crop_size and data.local_crop_size must be lists'
		raise TypeError(msg)
	if not isinstance(context_downsample, int):
		msg = 'data.context_downsample must be an integer'
		raise TypeError(msg)

	downsampled = [size // context_downsample for size in context_crop_size]
	if any(size % context_downsample != 0 for size in context_crop_size):
		msg = 'data.context_crop_size must be divisible by data.context_downsample'
		raise ValueError(msg)
	if downsampled != local_crop_size:
		msg = (
			'data.context_crop_size / data.context_downsample '
			f'must equal data.local_crop_size; got {downsampled!r} '
			f'and {local_crop_size!r}'
		)
		raise ValueError(msg)


def _validate_attributes(attributes: Mapping[str, object]) -> None:
	names = attributes.get('names')
	if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
		msg = 'attributes.names must be a list of attribute names'
		raise ValueError(msg)

	if names != EXPECTED_ATTRIBUTES:
		msg = f'attributes.names must equal {EXPECTED_ATTRIBUTES!r}; got {names!r}'
		raise ValueError(msg)

	groups = attributes.get('groups')
	if not isinstance(groups, Mapping):
		msg = 'attributes.groups must be a mapping'
		raise TypeError(msg)

	if groups != EXPECTED_ATTRIBUTE_GROUPS:
		msg = (
			'attributes.groups must equal the MVP attribute group mapping; '
			f'got {dict(groups)!r}'
		)
		raise ValueError(msg)


def _validate_nopims_root(paths: Mapping[str, object]) -> None:
	nopims_root = paths.get('nopims_root')
	if nopims_root is None:
		return
	if not isinstance(nopims_root, str):
		msg = f'paths.nopims_root must be a string; got {nopims_root!r}'
		raise TypeError(msg)


def _reject_f3_pretraining_config(value: object, path: str = 'config') -> None:
	if isinstance(value, Mapping):
		for key, child in value.items():
			key_text = str(key)
			if key_text.lower() in DISALLOWED_PRETRAINING_KEYS:
				msg = (
					'F3 settings are not allowed in pretraining config: '
					f'{path}.{key_text}'
				)
				raise ValueError(msg)
			_reject_f3_pretraining_config(child, f'{path}.{key_text}')
	elif isinstance(value, list):
		for index, child in enumerate(value):
			_reject_f3_pretraining_config(child, f'{path}[{index}]')
