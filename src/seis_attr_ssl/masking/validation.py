"""Validation helpers for masking plans."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from seis_attr_ssl.attributes import MVP_ATTRIBUTE_REGISTRY

if TYPE_CHECKING:
	from seis_attr_ssl.attributes import AttributeRegistry
	from seis_attr_ssl.masking.schema import MaskingPlan


def validate_masking_plan(
	plan: MaskingPlan,
	registry: AttributeRegistry = MVP_ATTRIBUTE_REGISTRY,
) -> None:
	"""Validate mask arrays, shapes, and registry-ID conventions."""
	_validate_bool_mask('spatial_mask', plan.spatial_mask, ndim=3)
	_validate_bool_mask('visible_spatial_mask', plan.visible_spatial_mask, ndim=3)
	_validate_bool_mask('attribute_input_mask', plan.attribute_input_mask, ndim=1)
	_validate_bool_mask('attribute_target_mask', plan.attribute_target_mask, ndim=1)
	_validate_bool_mask('dropped_attribute_mask', plan.dropped_attribute_mask, ndim=1)
	_validate_int64_array('input_attribute_ids', plan.input_attribute_ids, ndim=1)
	_validate_int64_array('target_attribute_ids', plan.target_attribute_ids, ndim=1)

	if plan.visible_spatial_mask.shape != plan.spatial_mask.shape:
		msg = (
			'visible_spatial_mask shape must equal spatial_mask shape; '
			f'got {plan.visible_spatial_mask.shape!r} and {plan.spatial_mask.shape!r}'
		)
		raise ValueError(msg)

	if not np.array_equal(plan.visible_spatial_mask, np.logical_not(plan.spatial_mask)):
		msg = 'visible_spatial_mask must equal ~spatial_mask'
		raise ValueError(msg)

	num_attrs = len(registry.specs)
	for field_name in (
		'attribute_input_mask',
		'attribute_target_mask',
		'dropped_attribute_mask',
	):
		mask = getattr(plan, field_name)
		if mask.shape != (num_attrs,):
			msg = (
				f'{field_name} shape must equal ({num_attrs},) for registry; '
				f'got {mask.shape!r}'
			)
			raise ValueError(msg)

	expected_dropped = np.logical_and(
		plan.attribute_target_mask,
		np.logical_not(plan.attribute_input_mask),
	)
	if not np.array_equal(plan.dropped_attribute_mask, expected_dropped):
		msg = (
			'dropped_attribute_mask must equal '
			'attribute_target_mask & ~attribute_input_mask'
		)
		raise ValueError(msg)

	expected_input_ids = registry_attribute_ids(registry)[plan.attribute_input_mask]
	if not np.array_equal(plan.input_attribute_ids, expected_input_ids):
		msg = (
			'input_attribute_ids must be sorted registry IDs selected by '
			'attribute_input_mask'
		)
		raise ValueError(msg)

	expected_target_ids = registry_attribute_ids(registry)
	if not np.array_equal(plan.target_attribute_ids, expected_target_ids):
		msg = 'target_attribute_ids must match registry order'
		raise ValueError(msg)


def registry_attribute_ids(
	registry: AttributeRegistry = MVP_ATTRIBUTE_REGISTRY,
) -> np.ndarray:
	"""Return registry IDs in stable registry order as int64."""
	return np.asarray([spec.id for spec in registry.specs], dtype=np.int64)


def _validate_bool_mask(field_name: str, value: object, *, ndim: int) -> None:
	if not isinstance(value, np.ndarray):
		msg = f'{field_name} must be a NumPy array; got {type(value).__name__}'
		raise TypeError(msg)
	if value.dtype != np.bool_:
		msg = f'{field_name} dtype must be bool; got {value.dtype}'
		raise TypeError(msg)
	if value.ndim != ndim:
		msg = f'{field_name} must be {ndim}D; got {value.ndim}D'
		raise ValueError(msg)


def _validate_int64_array(field_name: str, value: object, *, ndim: int) -> None:
	if not isinstance(value, np.ndarray):
		msg = f'{field_name} must be a NumPy array; got {type(value).__name__}'
		raise TypeError(msg)
	if value.dtype != np.int64:
		msg = f'{field_name} dtype must be int64; got {value.dtype}'
		raise TypeError(msg)
	if value.ndim != ndim:
		msg = f'{field_name} must be {ndim}D; got {value.ndim}D'
		raise ValueError(msg)


__all__ = ['registry_attribute_ids', 'validate_masking_plan']
