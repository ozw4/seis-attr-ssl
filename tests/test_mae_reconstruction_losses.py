from __future__ import annotations

import pytest
import torch

from seis_attr_ssl.losses import (
	mae_pretraining_loss,
	masked_patch_reconstruction_loss,
)
from seis_attr_ssl.models.mae.patching import patchify_3d

PATCH_SIZE_XYZ = (2, 2, 2)


def _target() -> torch.Tensor:
	return torch.arange(1 * 3 * 4 * 4 * 4, dtype=torch.float32).reshape(
		1,
		3,
		4,
		4,
		4,
	)


def _spatial_mask() -> torch.Tensor:
	mask = torch.zeros((1, 2, 2, 2), dtype=torch.bool)
	mask[0, 0, 0, 0] = True
	return mask


def _target_valid() -> torch.Tensor:
	return torch.ones((1, 3), dtype=torch.bool)


def _dropped_attribute_mask() -> torch.Tensor:
	return torch.zeros((1, 3), dtype=torch.bool)


def test_zero_loss_when_prediction_equals_target() -> None:
	target = _target()
	pred_patches = patchify_3d(target, PATCH_SIZE_XYZ).requires_grad_()

	losses = mae_pretraining_loss(
		pred_patches=pred_patches,
		target=target,
		spatial_mask=_spatial_mask(),
		target_valid=_target_valid(),
		dropped_attribute_mask=_dropped_attribute_mask(),
		patch_size_xyz=PATCH_SIZE_XYZ,
	)

	assert losses['loss'] == torch.tensor(0.0)
	assert losses['loss_reconstruction'] == torch.tensor(0.0)
	assert losses['loss_dropped_attribute'] == torch.tensor(0.0)
	assert losses['loss_gradient'] == torch.tensor(0.0)


def test_masked_spatial_patches_only_contribute_to_reconstruction_loss() -> None:
	target = torch.zeros((1, 3, 4, 4, 4))
	pred_patches = patchify_3d(target, PATCH_SIZE_XYZ)
	pred_patches[:, 1] = 10.0

	visible_error_loss = masked_patch_reconstruction_loss(
		pred_patches=pred_patches,
		target=target,
		spatial_mask=_spatial_mask(),
		target_valid=_target_valid(),
		patch_size_xyz=PATCH_SIZE_XYZ,
		reconstruction='mse',
	)

	pred_patches[:, 0] = 10.0
	masked_error_loss = masked_patch_reconstruction_loss(
		pred_patches=pred_patches,
		target=target,
		spatial_mask=_spatial_mask(),
		target_valid=_target_valid(),
		patch_size_xyz=PATCH_SIZE_XYZ,
		reconstruction='mse',
	)

	assert visible_error_loss == torch.tensor(0.0)
	assert masked_error_loss > 0


def test_invalid_target_attributes_are_ignored() -> None:
	target = torch.zeros((1, 3, 4, 4, 4))
	pred_patches = patchify_3d(target, PATCH_SIZE_XYZ)
	pred_patches[:, 0, 1] = 10.0
	target_valid = torch.tensor([[True, False, True]])

	loss = masked_patch_reconstruction_loss(
		pred_patches=pred_patches,
		target=target,
		spatial_mask=_spatial_mask(),
		target_valid=target_valid,
		patch_size_xyz=PATCH_SIZE_XYZ,
		reconstruction='mse',
	)

	assert loss == torch.tensor(0.0)


def test_dropped_attribute_term_is_nonzero_when_visible_drop_is_wrong() -> None:
	target = torch.zeros((1, 3, 4, 4, 4))
	pred_patches = patchify_3d(target, PATCH_SIZE_XYZ)
	pred_patches[:, 1, 1] = 2.0
	dropped_attribute_mask = torch.tensor([[False, True, False]])

	losses = mae_pretraining_loss(
		pred_patches=pred_patches,
		target=target,
		spatial_mask=_spatial_mask(),
		target_valid=_target_valid(),
		dropped_attribute_mask=dropped_attribute_mask,
		patch_size_xyz=PATCH_SIZE_XYZ,
		reconstruction='mse',
		dropped_attribute_weight=1.0,
		gradient_weight=0.0,
	)

	assert losses['loss_reconstruction'] == torch.tensor(0.0)
	assert losses['loss_dropped_attribute'] > 0
	assert losses['loss'] == losses['loss_dropped_attribute']


def test_loss_raises_on_no_valid_targets() -> None:
	target = _target()
	pred_patches = patchify_3d(target, PATCH_SIZE_XYZ)

	with pytest.raises(ValueError, match='no valid reconstruction targets'):
		mae_pretraining_loss(
			pred_patches=pred_patches,
			target=target,
			spatial_mask=_spatial_mask(),
			target_valid=torch.zeros((1, 3), dtype=torch.bool),
			dropped_attribute_mask=_dropped_attribute_mask(),
			patch_size_xyz=PATCH_SIZE_XYZ,
		)


def test_gradients_flow_to_pred_patches() -> None:
	target = torch.zeros((1, 3, 4, 4, 4))
	pred_patches = (patchify_3d(target, PATCH_SIZE_XYZ) + 0.5).requires_grad_()

	losses = mae_pretraining_loss(
		pred_patches=pred_patches,
		target=target,
		spatial_mask=_spatial_mask(),
		target_valid=_target_valid(),
		dropped_attribute_mask=_dropped_attribute_mask(),
		patch_size_xyz=PATCH_SIZE_XYZ,
		reconstruction='mse',
	)

	losses['loss'].backward()

	assert pred_patches.grad is not None
	assert pred_patches.grad.abs().sum() > 0


@pytest.mark.parametrize('reconstruction', ['huber', 'mse'])
def test_huber_and_mse_modes_both_work(reconstruction: str) -> None:
	target = torch.zeros((1, 3, 4, 4, 4))
	pred_patches = patchify_3d(target, PATCH_SIZE_XYZ) + 1.0

	losses = mae_pretraining_loss(
		pred_patches=pred_patches,
		target=target,
		spatial_mask=_spatial_mask(),
		target_valid=_target_valid(),
		dropped_attribute_mask=_dropped_attribute_mask(),
		patch_size_xyz=PATCH_SIZE_XYZ,
		reconstruction=reconstruction,
	)

	assert losses['loss'] > 0
