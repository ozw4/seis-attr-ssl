"""Attribute subset sampling for MVP pretraining batches."""

from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np

from seis_attr_ssl.attributes import MVP_ATTRIBUTE_REGISTRY

if TYPE_CHECKING:
	from collections.abc import Sequence

AMPLITUDE_ATTRIBUTE_ID = MVP_ATTRIBUTE_REGISTRY.name_to_id('amplitude_norm')
MVP_ATTRIBUTE_IDS = tuple(spec.id for spec in MVP_ATTRIBUTE_REGISTRY.specs)
_MVP_ATTRIBUTE_ID_SET = set(MVP_ATTRIBUTE_IDS)


def sample_attribute_subset(
	available_attribute_ids: Sequence[int],
	min_input_attributes: int,
	max_input_attributes: int,
	rng: np.random.Generator,
) -> tuple[int, ...]:
	"""Sample a registry-ordered input attribute subset.

	The MVP amplitude channel is always included when present. If fewer than
	``min_input_attributes`` are available, a clear error is raised.
	"""
	available = _validate_available_attribute_ids(available_attribute_ids)
	min_count = _validate_count(min_input_attributes, 'min_input_attributes')
	max_count = _validate_count(max_input_attributes, 'max_input_attributes')
	if min_count > max_count:
		msg = (
			'min_input_attributes must be less than or equal to '
			f'max_input_attributes; got {min_count!r} > {max_count!r}'
		)
		raise ValueError(msg)
	if len(available) < min_count:
		msg = (
			'not enough available attributes to sample input subset: '
			f'available={available!r}, min_input_attributes={min_count!r}'
		)
		raise ValueError(msg)

	target_count = int(rng.integers(min_count, min(max_count, len(available)) + 1))
	selected: set[int] = set()
	if AMPLITUDE_ATTRIBUTE_ID in available:
		selected.add(AMPLITUDE_ATTRIBUTE_ID)

	remaining = tuple(id_ for id_ in available if id_ not in selected)
	needed = target_count - len(selected)
	if needed > 0:
		sampled = rng.choice(
			np.asarray(remaining, dtype=np.int64),
			size=needed,
			replace=False,
		)
		selected.update(int(id_) for id_ in sampled)

	return tuple(id_ for id_ in available if id_ in selected)


def _validate_available_attribute_ids(
	available_attribute_ids: Sequence[int],
) -> tuple[int, ...]:
	if isinstance(available_attribute_ids, str):
		msg = 'available_attribute_ids must be a sequence of integer IDs'
		raise TypeError(msg)

	ids: list[int] = []
	seen: set[int] = set()
	for raw_id in available_attribute_ids:
		if not isinstance(raw_id, Integral):
			msg = f'attribute ID must be an integer; got {raw_id!r}'
			raise TypeError(msg)
		id_ = int(raw_id)
		if id_ not in _MVP_ATTRIBUTE_ID_SET:
			msg = f'unknown MVP attribute ID: {id_!r}'
			raise ValueError(msg)
		if id_ in seen:
			msg = f'available_attribute_ids must be unique; duplicate {id_!r}'
			raise ValueError(msg)
		seen.add(id_)
		ids.append(id_)

	return tuple(sorted(ids))


def _validate_count(value: int, name: str) -> int:
	if not isinstance(value, Integral):
		msg = f'{name} must be an integer; got {value!r}'
		raise TypeError(msg)
	count = int(value)
	if count <= 0:
		msg = f'{name} must be positive; got {count!r}'
		raise ValueError(msg)
	return count


__all__ = [
	'AMPLITUDE_ATTRIBUTE_ID',
	'MVP_ATTRIBUTE_IDS',
	'sample_attribute_subset',
]
