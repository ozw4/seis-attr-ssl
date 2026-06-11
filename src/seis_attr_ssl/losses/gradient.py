"""Gradient-domain losses for MAE reconstruction targets."""

from __future__ import annotations

from typing import Literal

import torch

from seis_attr_ssl.attributes import MVP_ATTRIBUTE_REGISTRY
from seis_attr_ssl.models.mae.patching import unpatchify_3d

LossMode = Literal['huber', 'mse']


def gradient_loss_xyz(  # noqa: PLR0913
	*,
	pred_patches: torch.Tensor,
	target: torch.Tensor,
	spatial_mask: torch.Tensor,
	target_valid: torch.Tensor,
	patch_size_xyz: tuple[int, int, int],
	reconstruction: LossMode = 'huber',
	huber_delta: float = 1.0,
	family_balanced: bool = True,
) -> torch.Tensor:
	"""Return finite-difference XYZ loss over masked spatial targets."""
	_validate_prediction_and_target(pred_patches, target)
	_validate_bool_mask(
		spatial_mask,
		'spatial_mask',
		(pred_patches.shape[0], *spatial_mask.shape[1:]),
	)
	_validate_bool_mask(
		target_valid,
		'target_valid',
		(pred_patches.shape[0], pred_patches.shape[2]),
	)
	_validate_same_device(pred_patches, target, spatial_mask, target_valid)

	grid_size_xyz = _grid_size_from_mask(spatial_mask, pred_patches.shape[1])
	pred_volume = unpatchify_3d(pred_patches, patch_size_xyz, grid_size_xyz)
	if pred_volume.shape != target.shape:
		msg = (
			'unpatchified pred_patches must match target shape; '
			f'got {tuple(pred_volume.shape)!r} and {tuple(target.shape)!r}'
		)
		raise ValueError(msg)

	volume_mask = _spatial_mask_to_volume(spatial_mask, patch_size_xyz)
	attr_weights = _attribute_family_weights(
		pred_patches.shape[2],
		device=pred_patches.device,
		dtype=pred_patches.dtype,
		family_balanced=family_balanced,
	)

	numerator = pred_patches.sum() * 0.0
	denominator = pred_patches.new_tensor(0.0)
	for dim in (2, 3, 4):
		pred_grad = pred_volume.diff(dim=dim)
		target_grad = target.diff(dim=dim)
		loss = _elementwise_loss(
			pred_grad,
			target_grad.to(dtype=pred_patches.dtype),
			reconstruction,
			huber_delta,
		)
		axis_mask = _neighbor_mask(volume_mask, dim)
		selected = axis_mask.unsqueeze(1) & target_valid.reshape(
			target_valid.shape[0],
			target_valid.shape[1],
			1,
			1,
			1,
		)
		weight = selected.to(dtype=loss.dtype) * attr_weights.reshape(
			1,
			attr_weights.shape[0],
			1,
			1,
			1,
		)
		numerator = numerator + (loss * weight).sum()
		denominator = denominator + weight.sum()

	if bool(denominator.detach().eq(0).item()):
		return pred_patches.sum() * 0.0
	return numerator / denominator


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


def _validate_prediction_and_target(
	pred_patches: torch.Tensor,
	target: torch.Tensor,
) -> None:
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
	if (
		pred_patches.shape[0] != target.shape[0]
		or pred_patches.shape[2] != target.shape[1]
	):
		msg = (
			'pred_patches and target batch/attribute dimensions must match; '
			f'got {tuple(pred_patches.shape)!r} and {tuple(target.shape)!r}'
		)
		raise ValueError(msg)


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


def _validate_same_device(*tensors: torch.Tensor) -> None:
	devices = {tensor.device for tensor in tensors}
	if len(devices) != 1:
		device_names = sorted(map(str, devices))
		msg = f'all tensors must be on the same device; got {device_names!r}'
		raise ValueError(msg)


def _grid_size_from_mask(
	spatial_mask: torch.Tensor,
	num_patches: int,
) -> tuple[int, int, int]:
	if spatial_mask.ndim != 4:
		msg = (
			'spatial_mask must be a 4D tensor with shape [B, TX, TY, TZ]; '
			f'got {tuple(spatial_mask.shape)!r}'
		)
		raise ValueError(msg)
	grid_size_xyz = (
		int(spatial_mask.shape[1]),
		int(spatial_mask.shape[2]),
		int(spatial_mask.shape[3]),
	)
	if grid_size_xyz[0] * grid_size_xyz[1] * grid_size_xyz[2] != num_patches:
		msg = (
			'spatial_mask grid must match pred_patches patch count; '
			f'got grid_size_xyz={grid_size_xyz!r}, num_patches={num_patches}'
		)
		raise ValueError(msg)
	return grid_size_xyz


def _spatial_mask_to_volume(
	spatial_mask: torch.Tensor,
	patch_size_xyz: tuple[int, int, int],
) -> torch.Tensor:
	px_size, py_size, pz_size = patch_size_xyz
	return (
		spatial_mask.repeat_interleave(px_size, dim=1)
		.repeat_interleave(py_size, dim=2)
		.repeat_interleave(pz_size, dim=3)
	)


def _neighbor_mask(mask: torch.Tensor, dim: int) -> torch.Tensor:
	head = [slice(None)] * mask.ndim
	tail = [slice(None)] * mask.ndim
	head[dim - 1] = slice(1, None)
	tail[dim - 1] = slice(None, -1)
	return mask[tuple(head)] & mask[tuple(tail)]


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


__all__ = ['gradient_loss_xyz']
