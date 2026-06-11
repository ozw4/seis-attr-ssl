from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from seis_attr_ssl.attributes import MVP_ATTRIBUTE_REGISTRY
from seis_attr_ssl.masking import (
	MaskingPlan,
	build_mae_masking_plan,
	validate_masking_plan,
)


def _build_plan(seed: int = 123, target_valid: np.ndarray | None = None) -> MaskingPlan:
	if target_valid is None:
		target_valid = np.ones(len(MVP_ATTRIBUTE_REGISTRY.specs), dtype=np.bool_)
	return build_mae_masking_plan(
		available_attribute_ids=tuple(np.flatnonzero(target_valid)),
		target_valid=target_valid,
		local_crop_size_xyz=(128, 128, 128),
		patch_size_xyz=(8, 8, 8),
		spatial_mask_ratio=0.75,
		block_size_tokens_xyz=(2, 2, 2),
		min_input_attributes=4,
		max_input_attributes=10,
		attribute_dropout_prob=0.30,
		group_dropout_prob=0.20,
		attribute_groups=MVP_ATTRIBUTE_REGISTRY.groups,
		rng=np.random.default_rng(seed),
	)


def test_default_config_produces_expected_token_grid_and_spatial_ratio() -> None:
	plan = _build_plan()

	assert plan.spatial_mask.shape == (16, 16, 16)
	assert 0.74 <= float(plan.spatial_mask.mean()) <= 0.76


def test_visible_spatial_mask_is_spatial_mask_complement() -> None:
	plan = _build_plan()

	np.testing.assert_array_equal(
		plan.visible_spatial_mask,
		np.logical_not(plan.spatial_mask),
	)


def test_missing_target_valid_propagates_to_attribute_target_mask() -> None:
	target_valid = np.asarray(
		[True, True, True, True, False, False, False, False, False, False],
		dtype=np.bool_,
	)
	plan = _build_plan(target_valid=target_valid)

	np.testing.assert_array_equal(plan.attribute_target_mask, target_valid)
	assert set(plan.input_attribute_ids).issubset(set(np.flatnonzero(target_valid)))


def test_dropped_attribute_mask_marks_valid_targets_withheld_from_input() -> None:
	target_valid = np.ones(len(MVP_ATTRIBUTE_REGISTRY.specs), dtype=np.bool_)
	target_valid[5] = False

	plan = _build_plan(seed=17, target_valid=target_valid)

	np.testing.assert_array_equal(
		plan.dropped_attribute_mask,
		np.logical_and(target_valid, np.logical_not(plan.attribute_input_mask)),
	)


def test_input_and_target_attribute_ids_follow_mvp_registry_order() -> None:
	plan = _build_plan(seed=17)

	np.testing.assert_array_equal(
		plan.input_attribute_ids,
		np.flatnonzero(plan.attribute_input_mask).astype(np.int64),
	)
	np.testing.assert_array_equal(
		plan.target_attribute_ids,
		np.arange(len(MVP_ATTRIBUTE_REGISTRY.specs), dtype=np.int64),
	)


def test_build_mae_masking_plan_is_deterministic_for_same_seed() -> None:
	first = _build_plan(seed=999)
	second = _build_plan(seed=999)

	for field_name in MaskingPlan.__dataclass_fields__:
		np.testing.assert_array_equal(
			getattr(first, field_name),
			getattr(second, field_name),
		)


def test_build_mae_masking_plan_returns_bool_masks() -> None:
	plan = _build_plan()

	assert plan.spatial_mask.dtype == np.bool_
	assert plan.visible_spatial_mask.dtype == np.bool_
	assert plan.attribute_input_mask.dtype == np.bool_
	assert plan.attribute_target_mask.dtype == np.bool_
	assert plan.dropped_attribute_mask.dtype == np.bool_


def test_validation_catches_inconsistent_mae_masks() -> None:
	plan = _build_plan()
	invalid = replace(
		plan,
		dropped_attribute_mask=np.zeros_like(plan.dropped_attribute_mask),
	)

	with pytest.raises(ValueError, match='dropped_attribute_mask must equal'):
		validate_masking_plan(invalid)


def test_build_mae_masking_plan_requires_explicit_bool_target_valid() -> None:
	with pytest.raises(TypeError, match='target_valid dtype must be bool'):
		_build_plan(target_valid=np.ones(len(MVP_ATTRIBUTE_REGISTRY.specs)))
