"""MAE reconstruction losses for strict Stage 1 pretraining."""

from __future__ import annotations

from typing import Literal

import torch

from seis_attr_ssl.attributes import MVP_ATTRIBUTE_REGISTRY
from seis_attr_ssl.losses.gradient import gradient_loss_xyz
from seis_attr_ssl.models.mae.patching import patchify_3d

LossMode = Literal['huber', 'mse']


def masked_patch_reconstruction_loss(  # noqa: PLR0913
	*,
	pred_patches: torch.Tensor,
	target: torch.Tensor,
	spatial_mask: torch.Tensor,
	target_valid: torch.Tensor,
	patch_size_xyz: tuple[int, int, int],
	local_valid_mask: torch.Tensor | None = None,
	valid_patch_min_fraction: float = 0.5,
	reconstruction: LossMode = 'huber',
	huber_delta: float = 1.0,
	family_balanced: bool = True,
) -> torch.Tensor:
	"""Return reconstruction loss on masked spatial patches and valid attributes."""
	target_patches = _aligned_target_patches(pred_patches, target, patch_size_xyz)
	_validate_spatial_mask(spatial_mask, pred_patches)
	_validate_bool_mask(
		target_valid,
		'target_valid',
		(pred_patches.shape[0], pred_patches.shape[2]),
	)
	_validate_same_device(pred_patches, target, spatial_mask, target_valid)

	selection = spatial_mask.reshape(
		pred_patches.shape[0],
		pred_patches.shape[1],
	)
	valid_patch_mask = _local_valid_patch_mask(
		local_valid_mask=local_valid_mask,
		pred_patches=pred_patches,
		target=target,
		patch_size_xyz=patch_size_xyz,
		valid_patch_min_fraction=valid_patch_min_fraction,
	)
	if valid_patch_mask is not None:
		selection = selection & valid_patch_mask
	selection = (
		selection.unsqueeze(-1).unsqueeze(-1)
		& target_valid.unsqueeze(1).unsqueeze(-1)
	)
	return _weighted_mean(
		_elementwise_loss(
			pred_patches,
			target_patches.to(dtype=pred_patches.dtype),
			reconstruction,
			huber_delta,
		),
		selection,
		_attribute_family_weights(
			pred_patches.shape[2],
			device=pred_patches.device,
			dtype=pred_patches.dtype,
			family_balanced=family_balanced,
		),
		empty='zero' if local_valid_mask is not None else 'raise',
	)


def dropped_attribute_reconstruction_loss(  # noqa: PLR0913
	*,
	pred_patches: torch.Tensor,
	target: torch.Tensor,
	target_valid: torch.Tensor,
	dropped_attribute_mask: torch.Tensor,
	patch_size_xyz: tuple[int, int, int],
	local_valid_mask: torch.Tensor | None = None,
	valid_patch_min_fraction: float = 0.5,
	reconstruction: LossMode = 'huber',
	huber_delta: float = 1.0,
	family_balanced: bool = True,
) -> torch.Tensor:
	"""Return loss for dropped valid attributes across all spatial patches."""
	target_patches = _aligned_target_patches(pred_patches, target, patch_size_xyz)
	_validate_bool_mask(
		target_valid,
		'target_valid',
		(pred_patches.shape[0], pred_patches.shape[2]),
	)
	_validate_bool_mask(
		dropped_attribute_mask,
		'dropped_attribute_mask',
		(pred_patches.shape[0], pred_patches.shape[2]),
	)
	_validate_same_device(pred_patches, target, target_valid, dropped_attribute_mask)

	valid_patch_mask = _local_valid_patch_mask(
		local_valid_mask=local_valid_mask,
		pred_patches=pred_patches,
		target=target,
		patch_size_xyz=patch_size_xyz,
		valid_patch_min_fraction=valid_patch_min_fraction,
	)
	if valid_patch_mask is None:
		valid_patch_mask = torch.ones(
			(pred_patches.shape[0], pred_patches.shape[1]),
			dtype=torch.bool,
			device=pred_patches.device,
		)
	selection = (
		valid_patch_mask.unsqueeze(-1).unsqueeze(-1)
		& (target_valid & dropped_attribute_mask).unsqueeze(1).unsqueeze(-1)
	)
	return _weighted_mean(
		_elementwise_loss(
			pred_patches,
			target_patches.to(dtype=pred_patches.dtype),
			reconstruction,
			huber_delta,
		),
		selection,
		_attribute_family_weights(
			pred_patches.shape[2],
			device=pred_patches.device,
			dtype=pred_patches.dtype,
			family_balanced=family_balanced,
		),
		empty='zero',
	)


def mae_pretraining_loss(  # noqa: PLR0913
	*,
	pred_patches: torch.Tensor,
	target: torch.Tensor,
	spatial_mask: torch.Tensor,
	target_valid: torch.Tensor,
	dropped_attribute_mask: torch.Tensor,
	patch_size_xyz: tuple[int, int, int],
	reconstruction: LossMode = 'huber',
	huber_delta: float = 1.0,
	dropped_attribute_weight: float = 0.25,
	gradient_weight: float = 0.05,
	family_balanced: bool = True,
	local_valid_mask: torch.Tensor | None = None,
	valid_patch_min_fraction: float = 0.5,
) -> dict[str, torch.Tensor]:
	"""Return total strict MAE pretraining loss and component scalars."""
	loss_reconstruction = masked_patch_reconstruction_loss(
		pred_patches=pred_patches,
		target=target,
		spatial_mask=spatial_mask,
		target_valid=target_valid,
		patch_size_xyz=patch_size_xyz,
		local_valid_mask=local_valid_mask,
		valid_patch_min_fraction=valid_patch_min_fraction,
		reconstruction=reconstruction,
		huber_delta=huber_delta,
		family_balanced=family_balanced,
	)
	loss_dropped_attribute = dropped_attribute_reconstruction_loss(
		pred_patches=pred_patches,
		target=target,
		target_valid=target_valid,
		dropped_attribute_mask=dropped_attribute_mask,
		patch_size_xyz=patch_size_xyz,
		local_valid_mask=local_valid_mask,
		valid_patch_min_fraction=valid_patch_min_fraction,
		reconstruction=reconstruction,
		huber_delta=huber_delta,
		family_balanced=family_balanced,
	)
	loss_gradient = gradient_loss_xyz(
		pred_patches=pred_patches,
		target=target,
		spatial_mask=spatial_mask,
		target_valid=target_valid,
		patch_size_xyz=patch_size_xyz,
		local_valid_mask=local_valid_mask,
		valid_patch_min_fraction=valid_patch_min_fraction,
		reconstruction=reconstruction,
		huber_delta=huber_delta,
		family_balanced=family_balanced,
	)
	loss = (
		loss_reconstruction
		+ dropped_attribute_weight * loss_dropped_attribute
		+ gradient_weight * loss_gradient
	)
	return {
		'loss': loss,
		'loss_reconstruction': loss_reconstruction,
		'loss_dropped_attribute': loss_dropped_attribute,
		'loss_gradient': loss_gradient,
	}


def _aligned_target_patches(
	pred_patches: torch.Tensor,
	target: torch.Tensor,
	patch_size_xyz: tuple[int, int, int],
) -> torch.Tensor:
	if pred_patches.ndim != 4:
		msg = (
			'pred_patches must be a 4D tensor with shape '
			f'[B, N, A, patch_volume]; got {tuple(pred_patches.shape)!r}'
		)
		raise ValueError(msg)
	if target.ndim != 5:
		msg = (
			'target must be a 5D tensor with shape [B, A, X, Y, Z]; '
			f'got {tuple(target.shape)!r}'
		)
		raise ValueError(msg)
	target_patches = patchify_3d(target, patch_size_xyz)
	if target_patches.shape != pred_patches.shape:
		msg = (
			'patchified target must match pred_patches shape; '
			f'got {tuple(target_patches.shape)!r} and {tuple(pred_patches.shape)!r}'
		)
		raise ValueError(msg)
	return target_patches


def _elementwise_loss(
	pred: torch.Tensor,
	target: torch.Tensor,
	reconstruction: LossMode,
	huber_delta: float,
) -> torch.Tensor:
	if reconstruction == 'mse':
		return (pred - target).square()
	if reconstruction == 'huber':
		if huber_delta <= 0:
			msg = f'huber_delta must be positive; got {huber_delta!r}'
			raise ValueError(msg)
		return torch.nn.functional.huber_loss(
			pred,
			target,
			reduction='none',
			delta=huber_delta,
		)
	msg = f'reconstruction must be "huber" or "mse"; got {reconstruction!r}'
	raise ValueError(msg)


def _local_valid_patch_mask(
	*,
	local_valid_mask: torch.Tensor | None,
	pred_patches: torch.Tensor,
	target: torch.Tensor,
	patch_size_xyz: tuple[int, int, int],
	valid_patch_min_fraction: float,
) -> torch.Tensor | None:
	if local_valid_mask is None:
		return None
	_validate_local_valid_mask(local_valid_mask, target)
	_validate_same_device(pred_patches, target, local_valid_mask)
	if not 0.0 <= valid_patch_min_fraction <= 1.0:
		msg = (
			'valid_patch_min_fraction must be between 0.0 and 1.0; '
			f'got {valid_patch_min_fraction!r}'
		)
		raise ValueError(msg)
	patches = patchify_3d(
		local_valid_mask.unsqueeze(1).to(dtype=pred_patches.dtype),
		patch_size_xyz,
	)
	valid_fraction = patches.squeeze(2).mean(dim=-1)
	if valid_fraction.shape != (pred_patches.shape[0], pred_patches.shape[1]):
		msg = (
			'patchified local_valid_mask must match pred_patches patch grid; '
			f'got {tuple(valid_fraction.shape)!r} and '
			f'{(pred_patches.shape[0], pred_patches.shape[1])!r}'
		)
		raise ValueError(msg)
	return valid_fraction > valid_patch_min_fraction


def _validate_local_valid_mask(
	local_valid_mask: torch.Tensor,
	target: torch.Tensor,
) -> None:
	if local_valid_mask.dtype != torch.bool:
		msg = (
			'local_valid_mask must have dtype torch.bool; '
			f'got {local_valid_mask.dtype!r}'
		)
		raise TypeError(msg)
	expected_shape = (
		target.shape[0],
		target.shape[2],
		target.shape[3],
		target.shape[4],
	)
	if tuple(local_valid_mask.shape) != expected_shape:
		msg = (
			f'local_valid_mask shape must be {expected_shape!r}; '
			f'got {tuple(local_valid_mask.shape)!r}'
		)
		raise ValueError(msg)


def _weighted_mean(
	loss: torch.Tensor,
	selection: torch.Tensor,
	attr_weights: torch.Tensor,
	*,
	empty: Literal['raise', 'zero'],
) -> torch.Tensor:
	weight = selection.to(dtype=loss.dtype) * attr_weights.reshape(
		1,
		1,
		attr_weights.shape[0],
		1,
	)
	weight = weight.expand_as(loss)
	denominator = weight.sum()
	if bool(denominator.detach().eq(0).item()):
		if empty == 'zero':
			return loss.sum() * 0.0
		msg = 'no valid reconstruction targets'
		raise ValueError(msg)
	return (loss * weight).sum() / denominator


def _validate_bool_mask(
	mask: torch.Tensor,
	name: str,
	shape: tuple[int, ...],
) -> None:
	if mask.dtype != torch.bool:
		msg = f'{name} must have dtype torch.bool; got {mask.dtype!r}'
		raise TypeError(msg)
	if tuple(mask.shape) != shape:
		msg = f'{name} shape must be {shape!r}; got {tuple(mask.shape)!r}'
		raise ValueError(msg)


def _validate_spatial_mask(
	spatial_mask: torch.Tensor,
	pred_patches: torch.Tensor,
) -> None:
	if spatial_mask.dtype != torch.bool:
		msg = f'spatial_mask must have dtype torch.bool; got {spatial_mask.dtype!r}'
		raise TypeError(msg)
	if spatial_mask.ndim != 4:
		msg = (
			'spatial_mask must be a 4D tensor with shape [B, TX, TY, TZ]; '
			f'got {tuple(spatial_mask.shape)!r}'
		)
		raise ValueError(msg)
	if spatial_mask.shape[0] != pred_patches.shape[0]:
		msg = (
			'spatial_mask batch dimension must match pred_patches; '
			f'got {spatial_mask.shape[0]} and {pred_patches.shape[0]}'
		)
		raise ValueError(msg)
	num_spatial_patches = spatial_mask.reshape(spatial_mask.shape[0], -1).shape[1]
	if num_spatial_patches != pred_patches.shape[1]:
		msg = (
			'spatial_mask grid must match pred_patches patch count; '
			f'got {tuple(spatial_mask.shape[1:])!r} and {pred_patches.shape[1]}'
		)
		raise ValueError(msg)


def _validate_same_device(*tensors: torch.Tensor) -> None:
	devices = {tensor.device for tensor in tensors}
	if len(devices) != 1:
		device_names = sorted(map(str, devices))
		msg = f'all tensors must be on the same device; got {device_names!r}'
		raise ValueError(msg)


def _attribute_family_weights(
	num_attributes: int,
	*,
	device: torch.device,
	dtype: torch.dtype,
	family_balanced: bool,
) -> torch.Tensor:
	if not family_balanced:
		return torch.ones(num_attributes, device=device, dtype=dtype)
	if num_attributes > len(MVP_ATTRIBUTE_REGISTRY.specs):
		msg = (
			'family-balanced losses require attributes from the fixed MVP registry; '
			f'got {num_attributes} attributes'
		)
		raise ValueError(msg)

	groups = tuple(
		MVP_ATTRIBUTE_REGISTRY.spec(index).group for index in range(num_attributes)
	)
	group_counts = {group: groups.count(group) for group in dict.fromkeys(groups)}
	weights = [
		num_attributes / (len(group_counts) * group_counts[group])
		for group in groups
	]
	return torch.tensor(weights, device=device, dtype=dtype)


__all__ = [
	'dropped_attribute_reconstruction_loss',
	'mae_pretraining_loss',
	'masked_patch_reconstruction_loss',
]
