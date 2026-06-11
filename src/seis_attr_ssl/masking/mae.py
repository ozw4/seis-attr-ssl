"""High-level MAE masking plan builder."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from seis_attr_ssl.masking.attributes import sample_attribute_input_mask
from seis_attr_ssl.masking.schema import MaskingPlan
from seis_attr_ssl.masking.spatial import (
	compute_token_grid_shape,
	generate_spatial_block_mask,
)
from seis_attr_ssl.masking.validation import (
	registry_attribute_ids,
	validate_masking_plan,
)

if TYPE_CHECKING:
	from collections.abc import Mapping, Sequence


def build_mae_masking_plan(  # noqa: PLR0913
	available_attribute_ids: Sequence[int],
	target_valid: np.ndarray,
	local_crop_size_xyz: Sequence[int],
	patch_size_xyz: Sequence[int],
	spatial_mask_ratio: float,
	block_size_tokens_xyz: Sequence[int],
	min_input_attributes: int,
	max_input_attributes: int,
	attribute_dropout_prob: float,
	group_dropout_prob: float,
	attribute_groups: Mapping[str, str],
	rng: np.random.Generator,
) -> MaskingPlan:
	"""Build and validate the strict MVP MAE masking contract for one sample."""
	target_attribute_ids = registry_attribute_ids()
	attribute_target_mask = _validate_target_valid(
		target_valid,
		expected_length=len(target_attribute_ids),
	)
	token_grid_shape = compute_token_grid_shape(
		local_crop_size_xyz,
		patch_size_xyz,
	)
	spatial_mask = generate_spatial_block_mask(
		token_grid_shape,
		spatial_mask_ratio,
		block_size_tokens_xyz,
		rng,
	)
	attribute_input_mask = sample_attribute_input_mask(
		available_attribute_ids,
		target_attribute_ids,
		attribute_groups,
		min_input_attributes,
		max_input_attributes,
		attribute_dropout_prob,
		group_dropout_prob,
		rng,
	)

	plan = MaskingPlan(
		spatial_mask=spatial_mask,
		visible_spatial_mask=np.logical_not(spatial_mask),
		attribute_input_mask=attribute_input_mask,
		attribute_target_mask=attribute_target_mask,
		dropped_attribute_mask=np.logical_and(
			attribute_target_mask,
			np.logical_not(attribute_input_mask),
		),
		input_attribute_ids=np.flatnonzero(attribute_input_mask).astype(np.int64),
		target_attribute_ids=target_attribute_ids,
	)
	validate_masking_plan(plan)
	return plan


def _validate_target_valid(
	target_valid: np.ndarray,
	*,
	expected_length: int,
) -> np.ndarray:
	if not isinstance(target_valid, np.ndarray):
		msg = f'target_valid must be a NumPy array; got {type(target_valid).__name__}'
		raise TypeError(msg)
	if target_valid.dtype != np.bool_:
		msg = f'target_valid dtype must be bool; got {target_valid.dtype}'
		raise TypeError(msg)
	if target_valid.shape != (expected_length,):
		msg = (
			f'target_valid shape must equal ({expected_length},); '
			f'got {target_valid.shape!r}'
		)
		raise ValueError(msg)
	return target_valid.copy()


__all__ = ['build_mae_masking_plan']
