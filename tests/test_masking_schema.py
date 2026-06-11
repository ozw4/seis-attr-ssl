from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from seis_attr_ssl.attributes import MVP_ATTRIBUTE_REGISTRY
from seis_attr_ssl.masking import (
	MaskingPlan,
	registry_attribute_ids,
	validate_masking_plan,
)


def _valid_plan() -> MaskingPlan:
	spatial_mask = np.zeros((2, 3, 4), dtype=np.bool_)
	spatial_mask[0, 1, 2] = True
	attribute_input_mask = np.array(
		[True, False, True, True, False, False, True, True, True, False],
		dtype=np.bool_,
	)
	attribute_target_mask = np.ones(len(MVP_ATTRIBUTE_REGISTRY.specs), dtype=np.bool_)
	return MaskingPlan(
		spatial_mask=spatial_mask,
		visible_spatial_mask=np.logical_not(spatial_mask),
		attribute_input_mask=attribute_input_mask,
		attribute_target_mask=attribute_target_mask,
		dropped_attribute_mask=np.logical_and(
			attribute_target_mask,
			np.logical_not(attribute_input_mask),
		),
		input_attribute_ids=np.flatnonzero(attribute_input_mask).astype(np.int64),
		target_attribute_ids=registry_attribute_ids(),
	)


def test_valid_masking_plan_passes_validation() -> None:
	plan = _valid_plan()

	validate_masking_plan(plan)

	assert plan.spatial_mask[0, 1, 2]
	assert not bool(plan.visible_spatial_mask[0, 1, 2])
	assert plan.dropped_attribute_mask[1]
	assert plan.input_attribute_ids.tolist() == [0, 2, 3, 6, 7, 8]
	assert plan.target_attribute_ids.tolist() == list(range(10))


def test_input_attribute_ids_use_registry_ids_not_mask_positions() -> None:
	registry = SimpleNamespace(
		specs=(
			SimpleNamespace(id=10),
			SimpleNamespace(id=20),
			SimpleNamespace(id=30),
		),
	)
	spatial_mask = np.zeros((1, 1, 1), dtype=np.bool_)
	attribute_input_mask = np.array([False, True, True], dtype=np.bool_)
	attribute_target_mask = np.ones(3, dtype=np.bool_)
	plan = MaskingPlan(
		spatial_mask=spatial_mask,
		visible_spatial_mask=np.logical_not(spatial_mask),
		attribute_input_mask=attribute_input_mask,
		attribute_target_mask=attribute_target_mask,
		dropped_attribute_mask=np.logical_and(
			attribute_target_mask,
			np.logical_not(attribute_input_mask),
		),
		input_attribute_ids=np.array([20, 30], dtype=np.int64),
		target_attribute_ids=np.array([10, 20, 30], dtype=np.int64),
	)

	validate_masking_plan(plan, registry=registry)


def test_masking_plan_requires_numpy_masks() -> None:
	plan = replace(_valid_plan(), spatial_mask=[False])

	with pytest.raises(TypeError, match='spatial_mask must be a NumPy array'):
		validate_masking_plan(plan)


def test_masking_plan_requires_bool_mask_dtype() -> None:
	plan = replace(
		_valid_plan(),
		attribute_input_mask=np.ones(len(MVP_ATTRIBUTE_REGISTRY.specs), dtype=np.int64),
	)

	with pytest.raises(TypeError, match='attribute_input_mask dtype must be bool'):
		validate_masking_plan(plan)


def test_masking_plan_requires_3d_spatial_masks() -> None:
	plan = replace(_valid_plan(), spatial_mask=np.zeros((2, 3), dtype=np.bool_))

	with pytest.raises(ValueError, match='spatial_mask must be 3D'):
		validate_masking_plan(plan)


def test_masking_plan_requires_1d_attribute_masks() -> None:
	plan = replace(
		_valid_plan(),
		attribute_target_mask=np.ones((2, 5), dtype=np.bool_),
	)

	with pytest.raises(ValueError, match='attribute_target_mask must be 1D'):
		validate_masking_plan(plan)


def test_masking_plan_requires_matching_spatial_shapes() -> None:
	plan = replace(
		_valid_plan(),
		visible_spatial_mask=np.ones((2, 3, 5), dtype=np.bool_),
	)

	with pytest.raises(ValueError, match='visible_spatial_mask shape'):
		validate_masking_plan(plan)


def test_masking_plan_requires_visible_mask_to_be_spatial_inverse() -> None:
	plan = replace(
		_valid_plan(),
		visible_spatial_mask=np.ones((2, 3, 4), dtype=np.bool_),
	)

	with pytest.raises(ValueError, match='visible_spatial_mask must equal'):
		validate_masking_plan(plan)


def test_masking_plan_requires_attribute_masks_to_match_registry_length() -> None:
	plan = replace(
		_valid_plan(),
		dropped_attribute_mask=np.zeros(9, dtype=np.bool_),
	)

	with pytest.raises(ValueError, match='dropped_attribute_mask shape'):
		validate_masking_plan(plan)


def test_masking_plan_requires_dropped_attribute_convention() -> None:
	plan = replace(
		_valid_plan(),
		dropped_attribute_mask=np.zeros(
			len(MVP_ATTRIBUTE_REGISTRY.specs),
			dtype=np.bool_,
		),
	)

	with pytest.raises(ValueError, match='dropped_attribute_mask must equal'):
		validate_masking_plan(plan)


def test_masking_plan_requires_input_ids_from_input_mask() -> None:
	plan = replace(
		_valid_plan(),
		input_attribute_ids=np.array([2, 0, 3, 6, 7, 8], dtype=np.int64),
	)

	with pytest.raises(ValueError, match='input_attribute_ids'):
		validate_masking_plan(plan)


def test_masking_plan_requires_int64_attribute_ids() -> None:
	plan = replace(
		_valid_plan(),
		input_attribute_ids=np.array([0, 2, 3, 6, 7, 8], dtype=np.int32),
	)

	with pytest.raises(TypeError, match='input_attribute_ids dtype must be int64'):
		validate_masking_plan(plan)


def test_masking_plan_requires_target_ids_in_registry_order() -> None:
	plan = replace(
		_valid_plan(),
		target_attribute_ids=np.array([1, 0, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64),
	)

	with pytest.raises(ValueError, match='target_attribute_ids'):
		validate_masking_plan(plan)
