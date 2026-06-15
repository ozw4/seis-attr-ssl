"""Dependency-light downsampling utilities for context crops."""

from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
	from collections.abc import Sequence


def downsample_context_mean(
	volume: np.ndarray,
	factor: int | Sequence[int],
) -> np.ndarray:
	"""Downsample a 3D context volume with mean pooling over cubic blocks."""
	factor_xyz = _validate_downsample_inputs(volume, factor)
	x_size, y_size, z_size = volume.shape
	x_factor, y_factor, z_factor = factor_xyz
	blocked = volume.astype(np.float32, copy=False).reshape(
		x_size // x_factor,
		x_factor,
		y_size // y_factor,
		y_factor,
		z_size // z_factor,
		z_factor,
	)
	return blocked.mean(axis=(1, 3, 5), dtype=np.float32).astype(
		np.float32,
		copy=False,
	)


def downsample_context_masked_mean(
	volume: np.ndarray,
	valid_mask: np.ndarray,
	factor: int | Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
	"""Downsample a 3D context volume by averaging valid voxels in each block."""
	factor_xyz = _validate_downsample_inputs(volume, factor)
	if valid_mask.shape != volume.shape:
		msg = (
			f'valid_mask shape must match volume shape; '
			f'got {valid_mask.shape!r} and {volume.shape!r}'
		)
		raise ValueError(msg)

	x_size, y_size, z_size = volume.shape
	x_factor, y_factor, z_factor = factor_xyz
	block_shape = (
		x_size // x_factor,
		x_factor,
		y_size // y_factor,
		y_factor,
		z_size // z_factor,
		z_factor,
	)
	blocked_volume = np.where(
		valid_mask,
		volume.astype(np.float32, copy=False),
		np.float32(0.0),
	).reshape(block_shape)
	blocked_mask = valid_mask.reshape(block_shape)
	valid_sum = blocked_volume.sum(axis=(1, 3, 5), dtype=np.float32)
	valid_count = blocked_mask.sum(axis=(1, 3, 5), dtype=np.int64)
	downsampled = np.zeros(valid_sum.shape, dtype=np.float32)
	np.divide(
		valid_sum,
		valid_count,
		out=downsampled,
		where=valid_count > 0,
	)
	return downsampled, valid_count > 0


def normalize_downsample_xyz(value: int | Sequence[int]) -> tuple[int, int, int]:
	"""Normalize scalar or `[x, y, z]` downsample factors to an XYZ triple."""
	if isinstance(value, bool):
		msg = f'downsample must be a positive integer or triple; got {value!r}'
		raise TypeError(msg)
	if isinstance(value, Integral):
		downsample = int(value)
		if downsample <= 0:
			msg = f'downsample must be positive; got {downsample!r}'
			raise ValueError(msg)
		return (downsample, downsample, downsample)
	if isinstance(value, str):
		msg = (
			'downsample must be a positive integer or length-3 integer sequence; '
			f'got {value!r}'
		)
		raise TypeError(msg)
	try:
		axes = tuple(value)
	except TypeError as exc:
		msg = (
			'downsample must be a positive integer or length-3 integer sequence; '
			f'got {value!r}'
		)
		raise TypeError(msg) from exc
	if len(axes) != 3:
		msg = (
			'downsample must be a positive integer or length-3 integer sequence; '
			f'got {value!r}'
		)
		raise TypeError(msg)
	if any(
		isinstance(axis, bool) or not isinstance(axis, Integral)
		for axis in axes
	):
		msg = (
			'downsample must be a positive integer or length-3 integer sequence; '
			f'got {value!r}'
		)
		raise TypeError(msg)
	downsample_xyz = tuple(int(axis) for axis in axes)
	if any(axis <= 0 for axis in downsample_xyz):
		msg = f'downsample values must be positive; got {downsample_xyz!r}'
		raise ValueError(msg)
	return downsample_xyz


def _validate_downsample_inputs(
	volume: np.ndarray,
	factor: int | Sequence[int],
) -> tuple[int, int, int]:
	factor_xyz = normalize_downsample_xyz(factor)
	if volume.ndim != 3:
		msg = f'volume must be a 3D [x, y, z] array; got ndim={volume.ndim}'
		raise ValueError(msg)
	if any(
		axis % factor_axis != 0
		for axis, factor_axis in zip(volume.shape, factor_xyz, strict=True)
	):
		msg = (
			f'volume shape must be divisible by factor in all dimensions; '
			f'got shape={volume.shape!r}, factor={factor!r}'
		)
		raise ValueError(msg)
	return factor_xyz


__all__ = [
	'downsample_context_masked_mean',
	'downsample_context_mean',
	'normalize_downsample_xyz',
]
