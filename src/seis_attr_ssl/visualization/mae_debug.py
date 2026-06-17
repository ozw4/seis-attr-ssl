"""Array utilities for MAE debug visualizations."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import torch

ArrayLike: TypeAlias = torch.Tensor | np.ndarray


def unpatchify_mae_predictions(
	pred_patches: ArrayLike,
	*,
	token_grid_shape: tuple[int, int, int],
	patch_size_xyz: tuple[int, int, int],
) -> np.ndarray:
	"""Convert MAE patch predictions from ``[B, N, A, PV]`` to ``[B, A, X, Y, Z]``."""
	pred_array = _as_numpy(pred_patches, 'pred_patches')
	if pred_array.ndim != 4:
		msg = (
			'pred_patches must be a 4D array with shape [B, N, A, patch_volume]; '
			f'got shape={pred_array.shape!r}'
		)
		raise ValueError(msg)

	tx_size, ty_size, tz_size = _validate_positive_int_triple(
		token_grid_shape,
		'token_grid_shape',
	)
	px_size, py_size, pz_size = _validate_positive_int_triple(
		patch_size_xyz,
		'patch_size_xyz',
	)
	batch_size, num_tokens, num_attributes, patch_volume = pred_array.shape
	expected_num_tokens = tx_size * ty_size * tz_size
	expected_patch_volume = px_size * py_size * pz_size
	if num_tokens != expected_num_tokens or patch_volume != expected_patch_volume:
		msg = (
			'pred_patches shape must match token_grid_shape and patch_size_xyz; '
			f'got shape={pred_array.shape!r}, '
			f'token_grid_shape={token_grid_shape!r}, '
			f'patch_size_xyz={patch_size_xyz!r}, '
			f'expected_num_tokens={expected_num_tokens}, '
			f'expected_patch_volume={expected_patch_volume}'
		)
		raise ValueError(msg)

	return (
		pred_array.reshape(
			batch_size,
			tx_size,
			ty_size,
			tz_size,
			num_attributes,
			px_size,
			py_size,
			pz_size,
		)
		.transpose(0, 4, 1, 5, 2, 6, 3, 7)
		.reshape(
			batch_size,
			num_attributes,
			tx_size * px_size,
			ty_size * py_size,
			tz_size * pz_size,
		)
	)


def upsample_token_mask_to_voxels(
	spatial_mask: ArrayLike,
	*,
	patch_size_xyz: tuple[int, int, int],
) -> np.ndarray:
	"""Convert token masks from ``[B, TX, TY, TZ]`` to voxel masks."""
	mask_array = _as_numpy(spatial_mask, 'spatial_mask')
	if mask_array.ndim != 4:
		msg = (
			'spatial_mask must be a 4D array with shape [B, TX, TY, TZ]; '
			f'got shape={mask_array.shape!r}'
		)
		raise ValueError(msg)

	px_size, py_size, pz_size = _validate_positive_int_triple(
		patch_size_xyz,
		'patch_size_xyz',
	)
	return (
		mask_array.astype(bool, copy=False)
		.repeat(px_size, axis=1)
		.repeat(py_size, axis=2)
		.repeat(pz_size, axis=3)
	)


def build_dense_model_input_for_attribute(
	*,
	x: ArrayLike,
	attribute_ids: ArrayLike,
	attr_id: int,
) -> tuple[np.ndarray | None, np.ndarray]:
	"""Return dense input channel values and per-sample presence for one attribute."""
	x_array = _as_numpy(x, 'x')
	attribute_id_array = _as_numpy(attribute_ids, 'attribute_ids')
	if x_array.ndim != 5:
		msg = (
			'x must be a 5D array with shape [B, C, X, Y, Z]; '
			f'got shape={x_array.shape!r}'
		)
		raise ValueError(msg)
	if attribute_id_array.ndim != 2:
		msg = (
			'attribute_ids must be a 2D array with shape [B, C]; '
			f'got shape={attribute_id_array.shape!r}'
		)
		raise ValueError(msg)
	if attribute_id_array.shape != x_array.shape[:2]:
		msg = (
			'attribute_ids shape must match x batch/channel dimensions; '
			f'got attribute_ids.shape={attribute_id_array.shape!r}, '
			f'x.shape={x_array.shape!r}'
		)
		raise ValueError(msg)

	if attr_id < 0:
		return None, np.zeros(x_array.shape[0], dtype=bool)

	presence = np.any(attribute_id_array == attr_id, axis=1)
	if not np.any(presence):
		return None, presence

	dense = np.zeros(
		(x_array.shape[0], *x_array.shape[2:]),
		dtype=x_array.dtype,
	)
	for batch_index in np.flatnonzero(presence):
		channel_index = int(
			np.flatnonzero(attribute_id_array[batch_index] == attr_id)[0],
		)
		dense[batch_index] = x_array[batch_index, channel_index]

	return dense, presence


def apply_visual_invalid_mask(
	image: np.ndarray,
	valid_mask: np.ndarray | None,
) -> np.ma.MaskedArray | np.ndarray:
	"""Mask invalid voxels for display without modifying numeric image values."""
	if valid_mask is None:
		return image
	return np.ma.masked_where(~np.asarray(valid_mask, dtype=bool), image, copy=True)


def _as_numpy(value: ArrayLike, name: str) -> np.ndarray:
	if isinstance(value, np.ndarray):
		return value

	if isinstance(value, torch.Tensor):
		return value.detach().cpu().numpy()

	msg = f'{name} must be a torch.Tensor or np.ndarray; got {type(value).__name__}'
	raise TypeError(msg)


def _validate_positive_int_triple(
	value: tuple[int, int, int],
	name: str,
) -> tuple[int, int, int]:
	if (
		not isinstance(value, tuple)
		or len(value) != 3
		or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
		or any(item <= 0 for item in value)
	):
		msg = f'{name} must be a positive integer triple; got {value!r}'
		raise ValueError(msg)
	return value


__all__ = [
	'apply_visual_invalid_mask',
	'build_dense_model_input_for_attribute',
	'unpatchify_mae_predictions',
	'upsample_token_mask_to_voxels',
]
