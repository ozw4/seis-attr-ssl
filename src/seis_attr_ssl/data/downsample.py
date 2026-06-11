"""Dependency-light downsampling utilities for context crops."""

from __future__ import annotations

from numbers import Integral

import numpy as np


def downsample_context_mean(volume: np.ndarray, factor: int) -> np.ndarray:
	"""Downsample a 3D context volume with mean pooling over cubic blocks."""
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


__all__ = ['downsample_context_mean']
