"""PyTorch collation helpers for MAE pretraining batches."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
	from collections.abc import Mapping, Sequence


def mae_collate_fn(
	samples: Sequence[Mapping[str, object]],
) -> dict[str, torch.Tensor | object]:
	"""Collate variable-attribute MAE samples into a padded tensor batch."""
	if not samples:
		msg = 'samples must contain at least one sample'
		raise ValueError(msg)

	x_arrays = [_require_array(sample, 'x') for sample in samples]
	cmax = max(int(x.shape[0]) for x in x_arrays)
	batch: dict[str, torch.Tensor | object] = {
		'x': _collate_padded_channels(x_arrays, cmax),
		'attribute_ids': _collate_attribute_ids(samples, cmax),
		'attribute_valid_mask': _collate_attribute_valid_mask(samples, cmax),
		'coords': [sample.get('coords') for sample in samples],
	}

	for key in (
		'target',
		'spatial_mask',
		'visible_spatial_mask',
		'attribute_input_mask',
		'attribute_target_mask',
		'dropped_attribute_mask',
		'target_valid',
	):
		batch[key] = _stack_arrays(samples, key)

	local_valid_values = [sample.get('local_valid_mask') for sample in samples]
	if any(value is not None for value in local_valid_values):
		if any(value is None for value in local_valid_values):
			msg = (
				'local_valid_mask must be present for every sample or omitted '
				'for every sample'
			)
			raise ValueError(msg)
		batch['local_valid_mask'] = _stack_arrays(samples, 'local_valid_mask')
	else:
		batch['local_valid_mask'] = None

	context_values = [sample.get('context') for sample in samples]
	if any(value is not None for value in context_values):
		if any(value is None for value in context_values):
			msg = 'context must be present for every sample or omitted for every sample'
			raise ValueError(msg)
		context_arrays = [_as_array(value, 'context') for value in context_values]
		batch['context'] = _collate_padded_channels(context_arrays, cmax)
		batch['context_valid_mask'] = _stack_arrays(samples, 'context_valid_mask')
	else:
		batch['context'] = None
		batch['context_valid_mask'] = None

	return batch


def move_batch_to_device(
	batch: Mapping[str, object],
	device: torch.device,
) -> dict[str, object]:
	"""Move tensor values in a batch to ``device`` while preserving metadata."""
	return {
		key: value.to(device) if isinstance(value, torch.Tensor) else value
		for key, value in batch.items()
	}


def _collate_padded_channels(
	arrays: Sequence[np.ndarray],
	channels: int,
) -> torch.Tensor:
	first_shape = arrays[0].shape[1:]
	for array in arrays:
		if array.ndim < 1:
			msg = (
				'channel-padded arrays must have at least 1 dimension; '
				f'got {array.ndim}'
			)
			raise ValueError(msg)
		if array.shape[1:] != first_shape:
			msg = (
				'all channel-padded arrays must share trailing shape; '
				f'got {array.shape[1:]!r}, expected {first_shape!r}'
			)
			raise ValueError(msg)
	tensor = torch.zeros(
		(len(arrays), channels, *first_shape),
		dtype=_torch_dtype(arrays[0]),
	)
	for index, array in enumerate(arrays):
		tensor[index, : array.shape[0]] = _to_tensor(array)
	return tensor


def _collate_attribute_ids(
	samples: Sequence[Mapping[str, object]],
	channels: int,
) -> torch.Tensor:
	ids = torch.full((len(samples), channels), -1, dtype=torch.long)
	for index, sample in enumerate(samples):
		attribute_ids = _to_tensor(_require_array(sample, 'attribute_ids')).long()
		ids[index, : attribute_ids.numel()] = attribute_ids
	return ids


def _collate_attribute_valid_mask(
	samples: Sequence[Mapping[str, object]],
	channels: int,
) -> torch.Tensor:
	mask = torch.zeros((len(samples), channels), dtype=torch.bool)
	for index, sample in enumerate(samples):
		if sample.get('valid_attributes') is None:
			count = _require_array(sample, 'attribute_ids').shape[0]
			mask[index, :count] = True
		else:
			valid = _to_tensor(_require_array(sample, 'valid_attributes')).bool()
			mask[index, : valid.numel()] = valid
	return mask


def _stack_arrays(
	samples: Sequence[Mapping[str, object]],
	key: str,
) -> torch.Tensor:
	arrays = [_require_array(sample, key) for sample in samples]
	return torch.stack([_to_tensor(array) for array in arrays], dim=0)


def _require_array(sample: Mapping[str, object], key: str) -> np.ndarray:
	try:
		value = sample[key]
	except KeyError as exc:
		msg = f'sample is missing required key {key!r}'
		raise KeyError(msg) from exc
	return _as_array(value, key)


def _as_array(value: object, key: str) -> np.ndarray:
	if not isinstance(value, np.ndarray):
		msg = f'{key} must be a NumPy array; got {type(value).__name__}'
		raise TypeError(msg)
	return value


def _to_tensor(array: np.ndarray) -> torch.Tensor:
	return torch.as_tensor(array, dtype=_torch_dtype(array))


def _torch_dtype(array: np.ndarray) -> torch.dtype:
	if np.issubdtype(array.dtype, np.floating):
		return torch.float32
	if np.issubdtype(array.dtype, np.bool_):
		return torch.bool
	if np.issubdtype(array.dtype, np.integer):
		return torch.long
	msg = f'unsupported NumPy dtype for collation: {array.dtype}'
	raise TypeError(msg)


__all__ = ['mae_collate_fn', 'move_batch_to_device']
