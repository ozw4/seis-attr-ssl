"""Spatial token-mask generation for 3D MAE grids."""

from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
	from collections.abc import Sequence


def compute_token_grid_shape(
	local_crop_size_xyz: Sequence[int],
	patch_size_xyz: Sequence[int],
) -> tuple[int, int, int]:
	"""Return token-grid shape in ``[x, y, z]`` order."""
	local_crop_size = _validate_xyz_ints(
		'local_crop_size_xyz',
		local_crop_size_xyz,
	)
	patch_size = _validate_xyz_ints('patch_size_xyz', patch_size_xyz)

	token_grid_shape = []
	for crop_size, patch in zip(local_crop_size, patch_size, strict=True):
		if crop_size % patch != 0:
			msg = (
				'local_crop_size_xyz must be exactly divisible by '
				f'patch_size_xyz; got {local_crop_size!r} and {patch_size!r}'
			)
			raise ValueError(msg)
		token_grid_shape.append(crop_size // patch)

	return tuple(token_grid_shape)


def generate_spatial_block_mask(
	token_grid_shape_xyz: Sequence[int],
	mask_ratio: float,
	block_size_tokens_xyz: Sequence[int],
	rng: np.random.Generator,
) -> np.ndarray:
	"""Return bool mask ``[tx, ty, tz]`` where ``True`` is a reconstruction target."""
	token_grid_shape = _validate_xyz_ints(
		'token_grid_shape_xyz',
		token_grid_shape_xyz,
	)
	block_size = _validate_xyz_ints(
		'block_size_tokens_xyz',
		block_size_tokens_xyz,
	)
	if not 0 <= mask_ratio < 1:
		msg = f'mask_ratio must be in [0, 1); got {mask_ratio!r}'
		raise ValueError(msg)
	if not isinstance(rng, np.random.Generator):
		msg = f'rng must be a NumPy Generator; got {type(rng).__name__}'
		raise TypeError(msg)

	mask = np.zeros(token_grid_shape, dtype=np.bool_)
	num_tokens = int(np.prod(token_grid_shape))
	target_masked_tokens = min(round(mask_ratio * num_tokens), num_tokens - 1)
	if target_masked_tokens == 0:
		return mask

	max_attempts = max(1000, target_masked_tokens * 10)
	for _ in range(max_attempts):
		if int(mask.sum()) >= target_masked_tokens:
			break
		start = tuple(
			int(rng.integers(0, max(dim - block + 1, 1)))
			for dim, block in zip(token_grid_shape, block_size, strict=True)
		)
		stop = tuple(
			min(offset + block, dim)
			for offset, block, dim in zip(
				start,
				block_size,
				token_grid_shape,
				strict=True,
			)
		)
		mask[
			start[0] : stop[0],
			start[1] : stop[1],
			start[2] : stop[2],
		] = True

	masked_count = int(mask.sum())
	if masked_count < target_masked_tokens:
		unmasked = np.flatnonzero(np.logical_not(mask))
		remaining = min(target_masked_tokens - masked_count, unmasked.size)
		if remaining > 0:
			extra = rng.choice(unmasked, size=remaining, replace=False)
			mask.reshape(-1)[extra] = True
	elif masked_count > target_masked_tokens:
		masked = np.flatnonzero(mask)
		to_unmask = rng.choice(
			masked,
			size=masked_count - target_masked_tokens,
			replace=False,
		)
		mask.reshape(-1)[to_unmask] = False

	if bool(mask.all()):
		visible_index = int(rng.integers(0, num_tokens))
		mask.reshape(-1)[visible_index] = False

	return mask


def _validate_xyz_ints(field_name: str, value: Sequence[int]) -> tuple[int, int, int]:
	if len(value) != 3:
		msg = f'{field_name} must contain exactly 3 values; got {len(value)}'
		raise ValueError(msg)

	values = tuple(value)
	for item in values:
		if isinstance(item, bool) or not isinstance(item, Integral):
			msg = f'{field_name} values must be positive integers; got {value!r}'
			raise TypeError(msg)
		if item <= 0:
			msg = f'{field_name} values must be positive integers; got {value!r}'
			raise ValueError(msg)

	return values


__all__ = ['compute_token_grid_shape', 'generate_spatial_block_mask']
