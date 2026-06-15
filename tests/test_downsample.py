from __future__ import annotations

import numpy as np
import pytest

from seis_attr_ssl.data.downsample import (
	downsample_context_masked_mean,
	downsample_context_mean,
	normalize_downsample_xyz,
)


def test_normalize_downsample_xyz_accepts_scalar_and_axis_wise_values() -> None:
	assert normalize_downsample_xyz(2) == (2, 2, 2)
	assert normalize_downsample_xyz([2, 2, 4]) == (2, 2, 4)


@pytest.mark.parametrize('value', [True, 0, -1, [2, 2], [2, 2, 0], [2, True, 4]])
def test_normalize_downsample_xyz_rejects_invalid_values(
	value: object,
) -> None:
	with pytest.raises((TypeError, ValueError)):
		normalize_downsample_xyz(value)  # type: ignore[arg-type]


def test_downsample_context_mean_accepts_axis_wise_factor() -> None:
	volume = np.arange(4 * 4 * 8, dtype=np.float32).reshape((4, 4, 8))

	downsampled = downsample_context_mean(volume, (2, 2, 4))

	expected = volume.reshape(2, 2, 2, 2, 2, 4).mean(axis=(1, 3, 5))
	assert downsampled.shape == (2, 2, 2)
	np.testing.assert_array_equal(downsampled, expected.astype(np.float32))


def test_downsample_context_masked_mean_accepts_axis_wise_factor() -> None:
	volume = np.arange(4 * 4 * 8, dtype=np.float32).reshape((4, 4, 8))
	valid_mask = np.ones_like(volume, dtype=bool)
	valid_mask[:2, :2, :4] = False

	downsampled, downsampled_mask = downsample_context_masked_mean(
		volume,
		valid_mask,
		(2, 2, 4),
	)

	assert downsampled.shape == (2, 2, 2)
	assert downsampled_mask.shape == (2, 2, 2)
	assert downsampled[0, 0, 0] == 0.0
	assert not downsampled_mask[0, 0, 0]
	np.testing.assert_array_equal(
		downsampled_mask[1:, :, :],
		np.ones_like(downsampled_mask[1:, :, :], dtype=bool),
	)
