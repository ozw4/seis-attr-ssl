"""Explicit validation for MVP configuration dictionaries."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import TypeAlias, TypeVar

from seis_attr_ssl.config.schema import (
	BASE_SEISMIC_REQUIRED_STAGES,
	DISALLOWED_PRETRAINING_KEYS,
	EXPECTED_ATTRIBUTE_GROUPS,
	EXPECTED_ATTRIBUTE_MODE,
	EXPECTED_ATTRIBUTES,
	EXPECTED_BASE_SEISMIC_KIND,
	EXPECTED_GRID_ORDER,
	EXPECTED_VOLUME_FORMAT,
	F3_ALLOWED_STAGES,
	KNOWN_STAGES,
)
from seis_attr_ssl.data.downsample import normalize_downsample_xyz

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
	if 'attribute_mode' in data:
		_validate_equal(data, 'attribute_mode', EXPECTED_ATTRIBUTE_MODE)
	_validate_optional_npy_path(data, 'base_seismic_path')
	local_crop_size, context_crop_size, context_downsample = (
		_validate_data_geometry(data)
	)
	if stage == 'pretrain_mae':
		_validate_attribute_halo_config(
			data,
			local_crop_size,
			context_crop_size,
			context_downsample,
		)

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
		model = _required_mapping(config, 'model')
		_validate_model(model)
		_validate_local_crop_patch_size(local_crop_size, model)

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


def _validate_data_geometry(
	data: Mapping[str, object],
) -> tuple[
	tuple[int, int, int],
	tuple[int, int, int] | None,
	tuple[int, int, int] | None,
]:
	local_crop_size = _validate_xyz_positive_ints(
		data,
		'local_crop_size',
		prefix='data',
	)
	use_context = _validate_bool(data, 'use_context', prefix='data')

	context_crop_size = None
	context_downsample = None
	if use_context and (
		'context_crop_size' not in data or 'context_downsample' not in data
	):
		msg = (
			'data.context_crop_size and data.context_downsample are required '
			'when data.use_context is true'
		)
		raise ValueError(msg)
	if use_context or 'context_crop_size' in data:
		context_crop_size = _validate_xyz_positive_ints(
			data,
			'context_crop_size',
			prefix='data',
		)
	if use_context or 'context_downsample' in data:
		context_downsample = _validate_context_downsample(data)
	if use_context:
		_validate_context_geometry(
			local_crop_size,
			context_crop_size,
			context_downsample,
		)
	if 'local_attribute_halo' in data:
		_validate_xyz_nonnegative_ints(
			data,
			'local_attribute_halo',
			prefix='data',
		)
	if 'context_attribute_halo' in data:
		_validate_xyz_nonnegative_ints(
			data,
			'context_attribute_halo',
			prefix='data',
		)
	if 'require_full_halo_inside_volume' in data:
		_validate_bool(data, 'require_full_halo_inside_volume', prefix='data')
	return local_crop_size, context_crop_size, context_downsample


def _validate_context_downsample(
	data: Mapping[str, object],
) -> tuple[int, int, int]:
	value = data.get('context_downsample')
	try:
		return normalize_downsample_xyz(value)  # type: ignore[arg-type]
	except TypeError as exc:
		msg = (
			'data.context_downsample must be a positive integer or list; '
			f'got {value!r}'
		)
		raise TypeError(msg) from exc
	except ValueError as exc:
		msg = f'data.context_downsample must be positive; got {value!r}'
		raise ValueError(msg) from exc


def _validate_context_geometry(
	local_crop_size: tuple[int, int, int],
	context_crop_size: tuple[int, int, int],
	context_downsample: tuple[int, int, int],
) -> None:
	if any(
		context_size % downsample != 0
		for context_size, downsample in zip(
			context_crop_size,
			context_downsample,
			strict=True,
		)
	):
		msg = (
			'data.context_crop_size must be divisible by '
			f'data.context_downsample; got {context_crop_size!r} and '
			f'{context_downsample!r}'
		)
		raise ValueError(msg)

	downsampled = tuple(
		context_size // downsample
		for context_size, downsample in zip(
			context_crop_size,
			context_downsample,
			strict=True,
		)
	)
	if downsampled != local_crop_size:
		msg = (
			'data.context_crop_size / data.context_downsample '
			f'must equal data.local_crop_size; got {downsampled!r} '
			f'and {local_crop_size!r}'
		)
		raise ValueError(msg)


def _validate_attribute_halo_config(
	data: Mapping[str, object],
	local_crop_size: tuple[int, int, int],
	context_crop_size: tuple[int, int, int] | None,
	context_downsample: tuple[int, int, int] | None,
) -> None:
	local_halo = _validate_xyz_nonnegative_ints(
		data,
		'local_attribute_halo',
		prefix='data',
	)
	_validate_bool(data, 'require_full_halo_inside_volume', prefix='data')
	_validate_compute_crop_size(local_crop_size, local_halo, 'local')
	if context_crop_size is not None and context_downsample is not None:
		context_halo = _validate_xyz_nonnegative_ints(
			data,
			'context_attribute_halo',
			prefix='data',
		)
		downsampled_context_size = tuple(
			size // downsample
			for size, downsample in zip(
				context_crop_size,
				context_downsample,
				strict=True,
			)
		)
		_validate_compute_crop_size(
			downsampled_context_size,
			context_halo,
			'context',
		)


def _validate_compute_crop_size(
	crop_size: Sequence[int],
	halo_xyz: tuple[int, int, int],
	name: str,
) -> None:
	compute_size = [
		size + 2 * halo
		for size, halo in zip(crop_size, halo_xyz, strict=True)
	]
	if any(size <= 0 for size in compute_size):
		msg = f'data.{name} attribute compute crop size must be positive'
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
	if 'grad_clip_norm' in train:
		_validate_optional_positive_float(train, 'grad_clip_norm', prefix='train')


def _validate_model(model: Mapping[str, object]) -> None:
	if 'context_token_min_valid_fraction' in model:
		_validate_unit_fraction(
			model,
			'context_token_min_valid_fraction',
			prefix='model',
		)
	if 'patch_size' in model:
		_validate_xyz_positive_ints(model, 'patch_size', prefix='model')


def _validate_local_crop_patch_size(
	local_crop_size: tuple[int, int, int],
	model: Mapping[str, object],
) -> None:
	if 'patch_size' not in model:
		return
	patch_size = _validate_xyz_positive_ints(model, 'patch_size', prefix='model')
	if any(
		crop_axis % patch_axis != 0
		for crop_axis, patch_axis in zip(local_crop_size, patch_size, strict=True)
	):
		msg = (
			'data.local_crop_size must be divisible by model.patch_size; '
			f'got {local_crop_size!r} and {patch_size!r}'
		)
		raise ValueError(msg)


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


def _validate_optional_positive_float(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> float | None:
	value = parent.get(key)
	if value is None:
		return None
	if isinstance(value, bool) or not isinstance(value, Real):
		msg = f'{prefix}.{key} must be a real number; got {value!r}'
		raise TypeError(msg)
	number = float(value)
	if not math.isfinite(number) or number <= 0.0:
		msg = f'{prefix}.{key} must be finite and positive; got {value!r}'
		raise ValueError(msg)
	return number


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


def _validate_xyz_positive_ints(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str = 'masking',
) -> tuple[int, int, int]:
	value = _require_xyz_list(parent, key, prefix=prefix)
	if any(int(item) <= 0 for item in value):
		msg = f'{prefix}.{key} values must be positive; got {value!r}'
		raise ValueError(msg)
	return tuple(value)


def _validate_xyz_nonnegative_ints(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> tuple[int, int, int]:
	value = _require_xyz_list(parent, key, prefix=prefix)
	if any(item < 0 for item in value):
		msg = f'{prefix}.{key} values must be nonnegative; got {value!r}'
		raise ValueError(msg)
	return tuple(value)


def _require_xyz_list(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> list[int]:
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
		msg = f'{prefix}.{key} must be a length-3 integer list; got {value!r}'
		raise TypeError(msg)
	return [int(item) for item in value]


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
