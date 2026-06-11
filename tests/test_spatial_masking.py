from __future__ import annotations

import numpy as np
import pytest

from seis_attr_ssl.masking import (
	compute_token_grid_shape,
	generate_spatial_block_mask,
)


def test_compute_token_grid_shape_default_mvp_grid() -> None:
	assert compute_token_grid_shape([128, 128, 128], [8, 8, 8]) == (16, 16, 16)


def test_generate_spatial_block_mask_is_deterministic_for_seed() -> None:
	first = generate_spatial_block_mask(
		(16, 16, 16),
		0.75,
		(2, 2, 2),
		np.random.default_rng(123),
	)
	second = generate_spatial_block_mask(
		(16, 16, 16),
		0.75,
		(2, 2, 2),
		np.random.default_rng(123),
	)

	np.testing.assert_array_equal(first, second)


def test_generate_spatial_block_mask_returns_bool_mask_with_expected_shape() -> None:
	mask = generate_spatial_block_mask(
		(16, 16, 16),
		0.75,
		(2, 2, 2),
		np.random.default_rng(123),
	)

	assert mask.dtype == np.bool_
	assert mask.shape == (16, 16, 16)


def test_generate_spatial_block_mask_approximately_matches_default_ratio() -> None:
	mask = generate_spatial_block_mask(
		(16, 16, 16),
		0.75,
		(2, 2, 2),
		np.random.default_rng(123),
	)

	assert 0.74 <= float(mask.mean()) <= 0.76


def test_generate_spatial_block_mask_trims_large_overshoots() -> None:
	mask = generate_spatial_block_mask(
		(4, 4, 4),
		0.10,
		(4, 4, 4),
		np.random.default_rng(123),
	)

	assert 0.08 <= float(mask.mean()) <= 0.12
	assert np.any(np.logical_not(mask))


def test_generate_spatial_block_mask_keeps_at_least_one_visible_token() -> None:
	mask = generate_spatial_block_mask(
		(2, 2, 2),
		0.99,
		(2, 2, 2),
		np.random.default_rng(123),
	)

	assert np.any(np.logical_not(mask))


def test_compute_token_grid_shape_rejects_non_divisible_sizes() -> None:
	with pytest.raises(ValueError, match='exactly divisible'):
		compute_token_grid_shape([128, 128, 127], [8, 8, 8])


@pytest.mark.parametrize('mask_ratio', [-0.1, 1.0, 1.1])
def test_generate_spatial_block_mask_rejects_invalid_ratios(mask_ratio: float) -> None:
	with pytest.raises(ValueError, match='mask_ratio must be in'):
		generate_spatial_block_mask(
			(16, 16, 16),
			mask_ratio,
			(2, 2, 2),
			np.random.default_rng(123),
		)


@pytest.mark.parametrize('block_size', [(0, 2, 2), (-1, 2, 2), (2, 2)])
def test_generate_spatial_block_mask_rejects_invalid_block_sizes(
	block_size: tuple[int, ...],
) -> None:
	with pytest.raises((TypeError, ValueError), match='block_size_tokens_xyz'):
		generate_spatial_block_mask(
			(16, 16, 16),
			0.75,
			block_size,
			np.random.default_rng(123),
		)
