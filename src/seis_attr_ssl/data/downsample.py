"""Dependency-light downsampling utilities for context crops."""

from __future__ import annotations

from numbers import Integral

import numpy as np


def downsample_context_mean(volume: np.ndarray, factor: int) -> np.ndarray:
	"""Downsample a 3D context volume with mean pooling over cubic blocks."""
	_validate_downsample_inputs(volume, factor)
	x_size, y_size, z_size = volume.shape
	blocked = volume.astype(np.float32, copy=False).reshape(
		x_size // factor,
		factor,
		y_size // factor,
		factor,
		z_size // factor,
		factor,
	)
	return blocked.mean(axis=(1, 3, 5), dtype=np.float32).astype(
		np.float32,
		copy=False,
	)


def downsample_context_masked_mean(
	volume: np.ndarray,
	valid_mask: np.ndarray,
	factor: int,
) -> tuple[np.ndarray, np.ndarray]:
	"""Downsample a 3D context volume by averaging valid voxels in each block."""
	_validate_downsample_inputs(volume, factor)
	if valid_mask.shape != volume.shape:
		msg = (
			f'valid_mask shape must match volume shape; '
			f'got {valid_mask.shape!r} and {volume.shape!r}'
		)
		raise ValueError(msg)

	x_size, y_size, z_size = volume.shape
	block_shape = (
		x_size // factor,
		factor,
		y_size // factor,
		factor,
		z_size // factor,
		factor,
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


def _validate_downsample_inputs(volume: np.ndarray, factor: int) -> None:
	if not isinstance(factor, Integral):
		msg = f'factor must be an integer; got {factor!r}'
		raise TypeError(msg)
	if factor <= 0:
		msg = f'factor must be positive; got {factor!r}'
		raise ValueError(msg)
	if volume.ndim != 3:
		msg = f'volume must be a 3D [x, y, z] array; got ndim={volume.ndim}'
		raise ValueError(msg)
	if any(axis % factor != 0 for axis in volume.shape):
		msg = (
			f'volume shape must be divisible by factor in all dimensions; '
			f'got shape={volume.shape!r}, factor={factor!r}'
		)
		raise ValueError(msg)


__all__ = ['downsample_context_masked_mean', 'downsample_context_mean']
