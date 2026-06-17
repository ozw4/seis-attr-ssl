from __future__ import annotations

import numpy as np
import pytest
import torch

from seis_attr_ssl.visualization.mae_debug import (
	apply_visual_invalid_mask,
	build_dense_model_input_for_attribute,
	unpatchify_mae_predictions,
	upsample_token_mask_to_voxels,
)


def test_unpatchify_mae_predictions_produces_expected_dense_values() -> None:
	pred_patches = np.array(
		[
			[
				[
					[0, 1, 10, 11, 100, 101, 110, 111],
					[1000, 1001, 1010, 1011, 1100, 1101, 1110, 1111],
				],
				[
					[200, 201, 210, 211, 300, 301, 310, 311],
					[1200, 1201, 1210, 1211, 1300, 1301, 1310, 1311],
				],
			],
		],
	)

	dense = unpatchify_mae_predictions(
		pred_patches,
		token_grid_shape=(2, 1, 1),
		patch_size_xyz=(2, 2, 2),
	)

	expected_attr0 = np.array(
		[
			[
				[[0, 1], [10, 11]],
				[[100, 101], [110, 111]],
				[[200, 201], [210, 211]],
				[[300, 301], [310, 311]],
			],
		],
	)
	expected_attr1 = expected_attr0 + 1000
	assert dense.shape == (1, 2, 4, 2, 2)
	np.testing.assert_array_equal(dense[:, 0], expected_attr0)
	np.testing.assert_array_equal(dense[:, 1], expected_attr1)


def test_unpatchify_mae_predictions_detaches_torch_inputs() -> None:
	pred_patches = torch.arange(16, dtype=torch.float32, requires_grad=True).reshape(
		1,
		2,
		1,
		8,
	)

	dense = unpatchify_mae_predictions(
		pred_patches,
		token_grid_shape=(2, 1, 1),
		patch_size_xyz=(2, 2, 2),
	)

	assert isinstance(dense, np.ndarray)
	assert dense.shape == (1, 1, 4, 2, 2)


def test_upsample_token_mask_to_voxels_repeats_along_all_axes() -> None:
	spatial_mask = np.array(
		[
			[
				[[True, False]],
				[[False, True]],
			],
		],
	)

	voxel_mask = upsample_token_mask_to_voxels(
		spatial_mask,
		patch_size_xyz=(2, 3, 2),
	)

	expected = (
		spatial_mask.repeat(2, axis=1).repeat(3, axis=2).repeat(2, axis=3)
	)
	assert voxel_mask.dtype == np.bool_
	assert voxel_mask.shape == (1, 4, 3, 4)
	np.testing.assert_array_equal(voxel_mask, expected)


def test_build_dense_model_input_for_attribute_returns_present_channel() -> None:
	x = np.arange(2 * 3 * 2 * 2 * 2, dtype=np.float32).reshape(2, 3, 2, 2, 2)
	attribute_ids = np.array([[4, 1, -1], [2, 4, 3]])

	dense, presence = build_dense_model_input_for_attribute(
		x=x,
		attribute_ids=attribute_ids,
		attr_id=4,
	)

	assert dense is not None
	np.testing.assert_array_equal(presence, np.array([True, True]))
	np.testing.assert_array_equal(dense[0], x[0, 0])
	np.testing.assert_array_equal(dense[1], x[1, 1])


def test_build_dense_model_input_for_attribute_handles_absent_attribute() -> None:
	x = np.zeros((2, 2, 2, 2, 2), dtype=np.float32)
	attribute_ids = np.array([[0, 1], [1, -1]])

	dense, presence = build_dense_model_input_for_attribute(
		x=x,
		attribute_ids=attribute_ids,
		attr_id=4,
	)

	assert dense is None
	np.testing.assert_array_equal(presence, np.array([False, False]))


def test_build_dense_model_input_for_attribute_ignores_padded_minus_one() -> None:
	x = np.ones((1, 2, 2, 2, 2), dtype=np.float32)
	attribute_ids = np.array([[-1, 2]])

	dense, presence = build_dense_model_input_for_attribute(
		x=x,
		attribute_ids=attribute_ids,
		attr_id=-1,
	)

	assert dense is None
	np.testing.assert_array_equal(presence, np.array([False]))


def test_apply_visual_invalid_mask_does_not_mutate_image() -> None:
	image = np.arange(4, dtype=np.float32).reshape(2, 2)
	valid_mask = np.array([[True, False], [False, True]])

	masked = apply_visual_invalid_mask(image, valid_mask)

	assert isinstance(masked, np.ma.MaskedArray)
	np.testing.assert_array_equal(image, np.array([[0, 1], [2, 3]], dtype=np.float32))
	np.testing.assert_array_equal(masked.mask, ~valid_mask)
	np.testing.assert_array_equal(masked.data, image)


def test_apply_visual_invalid_mask_returns_image_when_mask_is_none() -> None:
	image = np.zeros((2, 2), dtype=np.float32)

	assert apply_visual_invalid_mask(image, None) is image


def test_unpatchify_mae_predictions_rejects_mismatched_shape() -> None:
	pred_patches = np.zeros((1, 3, 2, 8), dtype=np.float32)

	with pytest.raises(ValueError, match='expected_num_tokens=2'):
		unpatchify_mae_predictions(
			pred_patches,
			token_grid_shape=(2, 1, 1),
			patch_size_xyz=(2, 2, 2),
		)
