from __future__ import annotations

import numpy as np
import pytest

from seis_attr_ssl.data.crop_sampler import (
	compute_centered_start,
	expand_request_with_halo,
	make_context_request,
	sample_random_center,
	sample_random_local_crop,
	sample_random_local_crop_with_margin,
)
from seis_attr_ssl.data.downsample import (
	downsample_context_masked_mean,
	downsample_context_mean,
)
from seis_attr_ssl.data.schema import CropRequest

LOCAL_SIZE = (128, 128, 128)
CONTEXT_SIZE = (512, 512, 512)


def _center(
	request_start: tuple[int, int, int],
	request_size: tuple[int, int, int],
) -> tuple[int, int, int]:
	return tuple(
		start_axis + size_axis // 2
		for start_axis, size_axis in zip(request_start, request_size, strict=True)
	)


def test_compute_centered_start_uses_xyz_order() -> None:
	assert compute_centered_start((100, 200, 300), (10, 20, 30)) == (95, 190, 285)


def test_sample_random_center_is_reproducible() -> None:
	first = sample_random_center(
		(1024, 1024, 768),
		(64, 64, 64),
		np.random.default_rng(123),
	)
	second = sample_random_center(
		(1024, 1024, 768),
		(64, 64, 64),
		np.random.default_rng(123),
	)

	assert first == second
	assert all(
		64 <= axis < limit - 64 + 1
		for axis, limit in zip(first, (1024, 1024, 768), strict=True)
	)


def test_sample_random_local_crop_is_in_bounds_for_large_volume() -> None:
	request = sample_random_local_crop(
		(1024, 1024, 768),
		LOCAL_SIZE,
		np.random.default_rng(42),
	)

	assert request.size_xyz == LOCAL_SIZE
	assert request.context_size_xyz is None
	assert request.context_downsample == 1
	assert all(axis >= 0 for axis in request.start_xyz)
	assert all(
		start_axis + size_axis <= shape_axis
		for start_axis, size_axis, shape_axis in zip(
			request.start_xyz,
			request.size_xyz,
			(1024, 1024, 768),
			strict=True,
		)
	)


def test_sample_random_local_crop_handles_exact_volume_size() -> None:
	request = sample_random_local_crop(
		(128, 128, 128),
		LOCAL_SIZE,
		np.random.default_rng(42),
	)

	assert request.start_xyz == (0, 0, 0)
	assert request.size_xyz == LOCAL_SIZE


def test_sample_random_local_crop_marks_padding_by_out_of_bounds_request() -> None:
	request = sample_random_local_crop(
		(64, 128, 256),
		LOCAL_SIZE,
		np.random.default_rng(42),
	)

	assert request.size_xyz == LOCAL_SIZE
	assert request.start_xyz[0] <= 0
	assert request.start_xyz[0] + request.size_xyz[0] >= 64
	assert request.start_xyz[1] == 0
	assert 0 <= request.start_xyz[2] <= 128


def test_expand_request_with_halo_returns_compute_request_and_payload_slices() -> None:
	request = CropRequest(
		survey_id='survey-a',
		start_xyz=(100, 100, 100),
		size_xyz=LOCAL_SIZE,
		context_size_xyz=None,
		context_downsample=1,
	)

	compute_request, payload_slices = expand_request_with_halo(
		request,
		(16, 16, 64),
	)

	assert compute_request.survey_id == 'survey-a'
	assert compute_request.start_xyz == (84, 84, 36)
	assert compute_request.size_xyz == (160, 160, 256)
	assert payload_slices == (
		slice(16, 144),
		slice(16, 144),
		slice(64, 192),
	)


def test_expand_request_with_zero_halo_keeps_payload_request() -> None:
	request = CropRequest(
		survey_id='survey-a',
		start_xyz=(10, 20, 30),
		size_xyz=(32, 32, 32),
		context_size_xyz=None,
		context_downsample=1,
	)

	compute_request, payload_slices = expand_request_with_halo(request, (0, 0, 0))

	assert compute_request == request
	assert payload_slices == (slice(0, 32), slice(0, 32), slice(0, 32))


def test_sample_random_local_crop_with_margin_is_in_bounds_when_possible() -> None:
	request = sample_random_local_crop_with_margin(
		(512, 512, 512),
		LOCAL_SIZE,
		(16, 16, 64),
		np.random.default_rng(42),
	)

	assert request.size_xyz == LOCAL_SIZE
	assert all(
		start_axis - margin_axis >= 0
		for start_axis, margin_axis in zip(
			request.start_xyz,
			(16, 16, 64),
			strict=True,
		)
	)
	assert all(
		start_axis + size_axis + margin_axis <= shape_axis
		for start_axis, size_axis, margin_axis, shape_axis in zip(
			request.start_xyz,
			request.size_xyz,
			(16, 16, 64),
			(512, 512, 512),
			strict=True,
		)
	)


def test_sample_random_local_crop_with_margin_falls_back_for_small_volume() -> None:
	request = sample_random_local_crop_with_margin(
		(64, 128, 256),
		LOCAL_SIZE,
		(16, 16, 64),
		np.random.default_rng(42),
	)

	assert isinstance(request, CropRequest)
	assert request.size_xyz == LOCAL_SIZE


def test_local_crop_sampling_is_reproducible() -> None:
	first = sample_random_local_crop(
		(1024, 1024, 768),
		LOCAL_SIZE,
		np.random.default_rng(7),
	)
	second = sample_random_local_crop(
		(1024, 1024, 768),
		LOCAL_SIZE,
		np.random.default_rng(7),
	)

	assert first == second


def test_make_context_request_matches_local_center() -> None:
	local = sample_random_local_crop(
		(1024, 1024, 768),
		LOCAL_SIZE,
		np.random.default_rng(4),
	)

	context = make_context_request(local, CONTEXT_SIZE, 4)

	assert context.size_xyz == CONTEXT_SIZE
	assert context.context_downsample == 4
	assert _center(context.start_xyz, context.size_xyz) == _center(
		local.start_xyz,
		local.size_xyz,
	)


def test_downsample_context_mean_shape_dtype_and_values() -> None:
	volume = np.arange(4 * 4 * 4, dtype=np.float32).reshape((4, 4, 4))

	downsampled = downsample_context_mean(volume, 2)

	expected = volume.reshape(2, 2, 2, 2, 2, 2).mean(axis=(1, 3, 5))
	assert downsampled.shape == (2, 2, 2)
	assert downsampled.dtype == np.float32
	np.testing.assert_array_equal(downsampled, expected.astype(np.float32))


def test_downsample_context_masked_mean_matches_mean_when_fully_valid() -> None:
	volume = np.arange(4 * 4 * 4, dtype=np.float32).reshape((4, 4, 4))
	valid_mask = np.ones_like(volume, dtype=bool)

	downsampled, downsampled_mask = downsample_context_masked_mean(
		volume,
		valid_mask,
		2,
	)

	np.testing.assert_array_equal(downsampled, downsample_context_mean(volume, 2))
	np.testing.assert_array_equal(
		downsampled_mask,
		np.ones((2, 2, 2), dtype=bool),
	)


def test_downsample_context_masked_mean_ignores_invalid_padding() -> None:
	volume = np.ones((4, 4, 4), dtype=np.float32)
	valid_mask = np.zeros((4, 4, 4), dtype=bool)
	valid_mask[:2, :, :] = True

	downsampled, downsampled_mask = downsample_context_masked_mean(
		volume,
		valid_mask,
		4,
	)

	np.testing.assert_array_equal(
		downsampled,
		np.ones((1, 1, 1), dtype=np.float32),
	)
	np.testing.assert_array_equal(downsampled_mask, np.ones((1, 1, 1), dtype=bool))


def test_downsample_context_masked_mean_marks_all_invalid_blocks() -> None:
	volume = np.ones((4, 4, 4), dtype=np.float32)
	valid_mask = np.zeros((4, 4, 4), dtype=bool)

	downsampled, downsampled_mask = downsample_context_masked_mean(
		volume,
		valid_mask,
		4,
	)

	np.testing.assert_array_equal(
		downsampled,
		np.zeros((1, 1, 1), dtype=np.float32),
	)
	np.testing.assert_array_equal(downsampled_mask, np.zeros((1, 1, 1), dtype=bool))


def test_downsample_context_mean_rejects_non_divisible_shape() -> None:
	with pytest.raises(ValueError, match='divisible'):
		downsample_context_mean(np.zeros((5, 4, 4), dtype=np.float32), 2)
