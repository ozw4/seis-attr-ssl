"""Explicit validation for MVP configuration dictionaries."""

from __future__ import annotations

import re
from collections.abc import Mapping
from numbers import Integral, Real
from typing import TypeAlias, TypeVar

from seis_attr_ssl.config.schema import (
	BASE_SEISMIC_REQUIRED_STAGES,
	DISALLOWED_PRETRAINING_KEYS,
	EXPECTED_ATTRIBUTE_GROUPS,
	EXPECTED_ATTRIBUTES,
	EXPECTED_BASE_SEISMIC_KIND,
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
	if stage in BASE_SEISMIC_REQUIRED_STAGES or 'base_seismic_kind' in data:
		_validate_equal(data, 'base_seismic_kind', EXPECTED_BASE_SEISMIC_KIND)
	_validate_optional_npy_path(data, 'base_seismic_path')
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

	if 'train' in config:
		_validate_train(_required_mapping(config, 'train'))

	if 'model' in config:
		_validate_model(_required_mapping(config, 'model'))

	if stage in {'pretrain_mae', 'dense_adaptation'} and 'masking' in config:
		_validate_masking(_required_mapping(config, 'masking'))

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


def _validate_optional_npy_path(parent: Mapping[str, object], key: str) -> None:
	value = parent.get(key)
	if value is None:
		return
	if not isinstance(value, str):
		msg = f'data.{key} must be a string; got {value!r}'
		raise TypeError(msg)
	if not value.endswith('.npy'):
		msg = f'data.{key} must point to a .npy file; got {value!r}'
		raise ValueError(msg)


def _validate_masking(masking: Mapping[str, object]) -> None:
	_validate_probability(masking, 'spatial_mask_ratio')
	_validate_equal(masking, 'spatial_mask_mode', 'block', prefix='masking')
	_validate_xyz_positive_ints(masking, 'block_size_tokens')
	min_input_attributes = _validate_positive_int(
		masking,
		'min_input_attributes',
	)
	max_input_attributes = _validate_positive_int(
		masking,
		'max_input_attributes',
	)
	if min_input_attributes > max_input_attributes:
		msg = (
			'masking.min_input_attributes must be less than or equal to '
			'masking.max_input_attributes'
		)
		raise ValueError(msg)
	_validate_probability(masking, 'attribute_dropout_prob')
	_validate_probability(masking, 'group_dropout_prob')


def _validate_train(train: Mapping[str, object]) -> None:
	if 'max_steps' in train:
		_validate_positive_int(train, 'max_steps', prefix='train')
	if 'samples_per_epoch' in train:
		_validate_positive_int(train, 'samples_per_epoch', prefix='train')
	if 'num_workers' in train:
		_validate_nonnegative_int(train, 'num_workers', prefix='train')
	if 'shuffle' in train:
		_validate_bool(train, 'shuffle', prefix='train')


def _validate_model(model: Mapping[str, object]) -> None:
	if 'context_token_min_valid_fraction' in model:
		_validate_unit_fraction(
			model,
			'context_token_min_valid_fraction',
			prefix='model',
		)


def _validate_probability(parent: Mapping[str, object], key: str) -> float:
	value = parent.get(key)
	if isinstance(value, bool) or not isinstance(value, Real):
		msg = f'masking.{key} must be a real number; got {value!r}'
		raise TypeError(msg)
	probability = float(value)
	if not 0.0 <= probability < 1.0:
		msg = f'masking.{key} must be in [0, 1); got {probability!r}'
		raise ValueError(msg)
	return probability


def _validate_unit_fraction(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> float:
	value = parent.get(key)
	if isinstance(value, bool) or not isinstance(value, Real):
		msg = f'{prefix}.{key} must be a real number; got {value!r}'
		raise TypeError(msg)
	fraction = float(value)
	if not 0.0 < fraction <= 1.0:
		msg = f'{prefix}.{key} must be in (0, 1]; got {fraction!r}'
		raise ValueError(msg)
	return fraction


def _validate_positive_int(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str = 'masking',
) -> int:
	value = parent.get(key)
	if isinstance(value, bool) or not isinstance(value, Integral):
		msg = f'{prefix}.{key} must be an integer; got {value!r}'
		raise TypeError(msg)
	count = int(value)
	if count <= 0:
		msg = f'{prefix}.{key} must be positive; got {count!r}'
		raise ValueError(msg)
	return count


def _validate_nonnegative_int(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> int:
	value = parent.get(key)
	if isinstance(value, bool) or not isinstance(value, Integral):
		msg = f'{prefix}.{key} must be an integer; got {value!r}'
		raise TypeError(msg)
	count = int(value)
	if count < 0:
		msg = f'{prefix}.{key} must be nonnegative; got {count!r}'
		raise ValueError(msg)
	return count


def _validate_bool(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> bool:
	value = parent.get(key)
	if not isinstance(value, bool):
		msg = f'{prefix}.{key} must be a bool; got {value!r}'
		raise TypeError(msg)
	return value


def _validate_xyz_positive_ints(parent: Mapping[str, object], key: str) -> None:
	value = parent.get(key)
	if (
		isinstance(value, str)
		or not isinstance(value, list)
		or len(value) != 3
		or any(
			isinstance(item, bool) or not isinstance(item, Integral)
			for item in value
		)
	):
		msg = f'masking.{key} must be a length-3 integer list; got {value!r}'
		raise TypeError(msg)
	if any(int(item) <= 0 for item in value):
		msg = f'masking.{key} values must be positive; got {value!r}'
		raise ValueError(msg)


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
	elif isinstance(value, str) and _looks_like_f3_path(value):
		msg = f'F3 paths are not allowed in pretraining config: {path}'
		raise ValueError(msg)


def _looks_like_f3_path(value: str) -> bool:
	parts = [part for part in re.split(r'[\\/]+', value.lower()) if part]
	return any(part == 'f3' or part.startswith('f3_') for part in parts)
