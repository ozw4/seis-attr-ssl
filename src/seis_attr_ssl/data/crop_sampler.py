"""Coordinate sampling utilities for `[x, y, z]` crop requests."""

from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
	from collections.abc import Sequence

	import numpy as np

from seis_attr_ssl.data.downsample import normalize_downsample_xyz
from seis_attr_ssl.data.schema import CropRequest

XYZ = tuple[int, int, int]


def compute_centered_start(center_xyz: Sequence[int], size_xyz: Sequence[int]) -> XYZ:
	"""Return the crop start that places ``size_xyz`` around ``center_xyz``."""
	center = _validate_xyz(center_xyz, 'center_xyz')
	size = _validate_positive_xyz(size_xyz, 'size_xyz')
	return tuple(
		center_axis - size_axis // 2
		for center_axis, size_axis in zip(center, size, strict=True)
	)


def compute_local_compute_size_xyz(
	local_crop_size_xyz: Sequence[int],
	local_attribute_halo_xyz: Sequence[int],
) -> XYZ:
	"""Return source crop size needed to compute local payload attributes."""
	local_crop_size = _validate_positive_xyz(
		local_crop_size_xyz,
		'local_crop_size_xyz',
	)
	local_attribute_halo = _validate_nonnegative_xyz(
		local_attribute_halo_xyz,
		'local_attribute_halo_xyz',
	)
	return tuple(
		local_axis + 2 * halo_axis
		for local_axis, halo_axis in zip(
			local_crop_size,
			local_attribute_halo,
			strict=True,
		)
	)


def compute_context_compute_size_xyz(
	context_crop_size_xyz: Sequence[int],
	context_downsample_xyz: Sequence[int],
	context_attribute_halo_xyz: Sequence[int],
) -> XYZ:
	"""Return source crop size needed to compute low-resolution context attributes."""
	context_crop_size = _validate_positive_xyz(
		context_crop_size_xyz,
		'context_crop_size_xyz',
	)
	context_downsample = _validate_positive_xyz(
		context_downsample_xyz,
		'context_downsample_xyz',
	)
	context_attribute_halo = _validate_nonnegative_xyz(
		context_attribute_halo_xyz,
		'context_attribute_halo_xyz',
	)
	return tuple(
		context_axis + 2 * halo_axis * downsample_axis
		for context_axis, downsample_axis, halo_axis in zip(
			context_crop_size,
			context_downsample,
			context_attribute_halo,
			strict=True,
		)
	)


def compute_required_full_halo_size_xyz(  # noqa: PLR0913
	local_crop_size_xyz: Sequence[int],
	local_attribute_halo_xyz: Sequence[int],
	*,
	use_context: bool,
	context_crop_size_xyz: Sequence[int] | None = None,
	context_downsample_xyz: Sequence[int] | None = None,
	context_attribute_halo_xyz: Sequence[int] | None = None,
) -> XYZ:
	"""Return minimum source volume size for full local/context halo crops."""
	local_compute_size = compute_local_compute_size_xyz(
		local_crop_size_xyz,
		local_attribute_halo_xyz,
	)
	if not use_context:
		return local_compute_size
	if (
		context_crop_size_xyz is None
		or context_downsample_xyz is None
		or context_attribute_halo_xyz is None
	):
		msg = 'context geometry is required when use_context=True'
		raise ValueError(msg)
	context_compute_size = compute_context_compute_size_xyz(
		context_crop_size_xyz,
		context_downsample_xyz,
		context_attribute_halo_xyz,
	)
	return tuple(
		max(local_axis, context_axis)
		for local_axis, context_axis in zip(
			local_compute_size,
			context_compute_size,
			strict=True,
		)
	)


def sample_random_center(
	shape_xyz: Sequence[int],
	margin_xyz: Sequence[int],
	rng: np.random.Generator,
) -> XYZ:
	"""Sample a center point in `[x, y, z]` order with the requested margins."""
	shape = _validate_positive_xyz(shape_xyz, 'shape_xyz')
	margin = _validate_nonnegative_xyz(margin_xyz, 'margin_xyz')
	return tuple(
		_sample_axis_center(axis_shape, axis_margin, rng)
		for axis_shape, axis_margin in zip(shape, margin, strict=True)
	)


def sample_random_local_crop(
	shape_xyz: Sequence[int],
	local_size_xyz: Sequence[int],
	rng: np.random.Generator,
) -> CropRequest:
	"""Sample a local crop, preferring fully in-bounds requests when possible."""
	shape = _validate_positive_xyz(shape_xyz, 'shape_xyz')
	local_size = _validate_positive_xyz(local_size_xyz, 'local_size_xyz')
	start = tuple(
		_sample_axis_start(axis_shape, axis_size, rng)
		for axis_shape, axis_size in zip(shape, local_size, strict=True)
	)
	return CropRequest(
		survey_id='',
		start_xyz=start,
		size_xyz=local_size,
		context_size_xyz=None,
		context_downsample=1,
	)


def expand_request_with_halo(
	request: CropRequest,
	halo_xyz: Sequence[int],
) -> tuple[CropRequest, tuple[slice, slice, slice]]:
	"""Return a compute request expanded by halo and slices to recover payload."""
	halo = _validate_nonnegative_xyz(halo_xyz, 'halo_xyz')
	size = _validate_positive_xyz(request.size_xyz, 'request.size_xyz')
	start = _validate_xyz(request.start_xyz, 'request.start_xyz')
	compute_start = tuple(
		start_axis - halo_axis
		for start_axis, halo_axis in zip(start, halo, strict=True)
	)
	compute_size = tuple(
		size_axis + 2 * halo_axis
		for size_axis, halo_axis in zip(size, halo, strict=True)
	)
	payload_slices = tuple(
		slice(halo_axis, halo_axis + size_axis)
		for halo_axis, size_axis in zip(halo, size, strict=True)
	)
	return (
		CropRequest(
			survey_id=request.survey_id,
			start_xyz=compute_start,
			size_xyz=compute_size,
			context_size_xyz=request.context_size_xyz,
			context_downsample=request.context_downsample,
		),
		cast('tuple[slice, slice, slice]', payload_slices),
	)


def sample_random_local_crop_with_margin(
	shape_xyz: Sequence[int],
	local_size_xyz: Sequence[int],
	margin_xyz: Sequence[int],
	rng: np.random.Generator,
) -> CropRequest:
	"""Sample local payload crop so payload+margin is inside the volume."""
	return sample_random_local_crop_with_margins(
		shape_xyz,
		local_size_xyz,
		margin_xyz,
		margin_xyz,
		rng,
	)


def sample_random_local_crop_with_margins(
	shape_xyz: Sequence[int],
	local_size_xyz: Sequence[int],
	margin_left_xyz: Sequence[int],
	margin_right_xyz: Sequence[int],
	rng: np.random.Generator,
) -> CropRequest:
	"""Sample local payload crop so asymmetric margins are inside the volume."""
	shape = _validate_positive_xyz(shape_xyz, 'shape_xyz')
	local_size = _validate_positive_xyz(local_size_xyz, 'local_size_xyz')
	margin_left = _validate_nonnegative_xyz(margin_left_xyz, 'margin_left_xyz')
	margin_right = _validate_nonnegative_xyz(margin_right_xyz, 'margin_right_xyz')
	if any(
		shape_axis < left_axis + size_axis + right_axis
		for shape_axis, size_axis, left_axis, right_axis in zip(
			shape,
			local_size,
			margin_left,
			margin_right,
			strict=True,
		)
	):
		return sample_random_local_crop(shape, local_size, rng)

	start = tuple(
		int(rng.integers(left_axis, shape_axis - size_axis - right_axis + 1))
		for shape_axis, size_axis, left_axis, right_axis in zip(
			shape,
			local_size,
			margin_left,
			margin_right,
			strict=True,
		)
	)
	return CropRequest(
		survey_id='',
		start_xyz=start,
		size_xyz=local_size,
		context_size_xyz=None,
		context_downsample=1,
	)


def make_context_request(
	local_request: CropRequest,
	context_size_xyz: Sequence[int],
	context_downsample: int | Sequence[int],
) -> CropRequest:
	"""Return a context crop request centered on ``local_request``."""
	context_size = _validate_positive_xyz(context_size_xyz, 'context_size_xyz')
	downsample = _validate_downsample(context_downsample)

	center = tuple(
		start_axis + size_axis // 2
		for start_axis, size_axis in zip(
			local_request.start_xyz,
			local_request.size_xyz,
			strict=True,
		)
	)
	return CropRequest(
		survey_id=local_request.survey_id,
		start_xyz=compute_centered_start(center, context_size),
		size_xyz=context_size,
		context_size_xyz=None,
		context_downsample=downsample,
	)


def _sample_axis_center(
	axis_shape: int,
	axis_margin: int,
	rng: np.random.Generator,
) -> int:
	if axis_shape <= axis_margin * 2:
		return int(rng.integers(0, axis_shape))
	high = axis_shape - axis_margin + 1 if axis_margin > 0 else axis_shape
	return int(rng.integers(axis_margin, high))


def _sample_axis_start(
	axis_shape: int,
	axis_size: int,
	rng: np.random.Generator,
) -> int:
	if axis_shape >= axis_size:
		return int(rng.integers(0, axis_shape - axis_size + 1))

	center = int(rng.integers(0, axis_shape))
	start = center - axis_size // 2
	return max(axis_shape - axis_size, min(start, 0))


def _validate_xyz(value: Sequence[int], name: str) -> XYZ:
	if (
		isinstance(value, str)
		or len(value) != 3
		or not all(
			not isinstance(axis, bool) and isinstance(axis, Integral)
			for axis in value
		)
	):
		msg = f'{name} must be a length-3 integer sequence; got {value!r}'
		raise TypeError(msg)
	return cast('XYZ', tuple(int(axis) for axis in value))


def _validate_positive_xyz(value: Sequence[int], name: str) -> XYZ:
	xyz = _validate_xyz(value, name)
	if any(axis <= 0 for axis in xyz):
		msg = f'{name} values must be positive; got {xyz!r}'
		raise ValueError(msg)
	return xyz


def _validate_nonnegative_xyz(value: Sequence[int], name: str) -> XYZ:
	xyz = _validate_xyz(value, name)
	if any(axis < 0 for axis in xyz):
		msg = f'{name} values must be nonnegative; got {xyz!r}'
		raise ValueError(msg)
	return xyz


def _validate_downsample(value: int | Sequence[int]) -> int | XYZ:
	if isinstance(value, bool):
		msg = f'context_downsample must be a positive integer or triple; got {value!r}'
		raise TypeError(msg)
	if isinstance(value, Integral):
		downsample = int(value)
		if downsample <= 0:
			msg = f'context_downsample must be positive; got {downsample!r}'
			raise ValueError(msg)
		return downsample
	return normalize_downsample_xyz(value)


__all__ = [
	'compute_centered_start',
	'compute_context_compute_size_xyz',
	'compute_local_compute_size_xyz',
	'compute_required_full_halo_size_xyz',
	'expand_request_with_halo',
	'make_context_request',
	'sample_random_center',
	'sample_random_local_crop',
	'sample_random_local_crop_with_margin',
	'sample_random_local_crop_with_margins',
]
