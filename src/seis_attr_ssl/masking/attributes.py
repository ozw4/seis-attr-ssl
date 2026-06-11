"""Attribute-level masking samplers for strict attribute-set MAE."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral, Real

import numpy as np

from seis_attr_ssl.attributes import MVP_ATTRIBUTE_REGISTRY

AMPLITUDE_ATTRIBUTE_ID = MVP_ATTRIBUTE_REGISTRY.name_to_id('amplitude_norm')
MVP_ATTRIBUTE_IDS = tuple(spec.id for spec in MVP_ATTRIBUTE_REGISTRY.specs)
_MVP_ATTRIBUTE_ID_SET = set(MVP_ATTRIBUTE_IDS)


def build_attribute_group_index(
	attribute_names: Sequence[str],
	attribute_groups: Mapping[str, str],
) -> dict[str, tuple[int, ...]]:
	"""Build group-to-registry-position index from ordered attribute names."""
	names = _validate_attribute_names(attribute_names)
	if not isinstance(attribute_groups, Mapping):
		msg = 'attribute_groups must be a mapping from attribute name to group name'
		raise TypeError(msg)

	group_keys = set(attribute_groups)
	name_keys = set(names)
	if group_keys != name_keys:
		missing = tuple(sorted(name_keys - group_keys))
		extra = tuple(sorted(group_keys - name_keys))
		msg = (
			'attribute_groups must contain exactly the target attribute names; '
			f'missing={missing!r}, extra={extra!r}'
		)
		raise ValueError(msg)

	group_to_ids: dict[str, list[int]] = {}
	for id_, name in enumerate(names):
		group = attribute_groups[name]
		if not isinstance(group, str) or not group:
			msg = f'attribute group for {name!r} must be a non-empty string'
			raise ValueError(msg)
		group_to_ids.setdefault(group, []).append(id_)

	return {group: tuple(ids) for group, ids in group_to_ids.items()}


def sample_attribute_input_mask(  # noqa: PLR0913
	available_attribute_ids: Sequence[int],
	target_attribute_ids: Sequence[int],
	attribute_groups: Mapping[str, str],
	min_input_attributes: int,
	max_input_attributes: int,
	attribute_dropout_prob: float,
	group_dropout_prob: float,
	rng: np.random.Generator,
	always_include_attribute_ids: Sequence[int] = (AMPLITUDE_ATTRIBUTE_ID,),
	attribute_names: Sequence[str] | None = None,
) -> np.ndarray:
	"""Return bool ``[num_attrs]`` where ``True`` means input-visible."""
	target_ids = _validate_target_attribute_ids(target_attribute_ids)
	target_id_set = set(target_ids)
	available = _validate_id_sequence(
		available_attribute_ids,
		'available_attribute_ids',
		target_id_set=target_id_set,
		sort=True,
	)
	always_include = _validate_id_sequence(
		always_include_attribute_ids,
		'always_include_attribute_ids',
		target_id_set=target_id_set,
		sort=True,
	)
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
			'not enough available attributes to sample input mask: '
			f'available={available!r}, min_input_attributes={min_count!r}'
		)
		raise ValueError(msg)

	available_set = set(available)
	missing_required = tuple(
		id_ for id_ in always_include if id_ not in available_set
	)
	if missing_required:
		names = _attribute_names_for_ids(target_ids, attribute_names)
		required_names = tuple(names[id_] for id_ in missing_required)
		msg = (
			'required always-include attributes are unavailable: '
			f'ids={missing_required!r}, names={required_names!r}'
		)
		raise ValueError(msg)
	if len(always_include) > max_count:
		msg = (
			'always_include_attribute_ids cannot fit within max_input_attributes: '
			f'always_include={always_include!r}, max_input_attributes={max_count!r}'
		)
		raise ValueError(msg)

	attribute_dropout = _validate_probability(
		attribute_dropout_prob,
		'attribute_dropout_prob',
	)
	group_dropout = _validate_probability(group_dropout_prob, 'group_dropout_prob')
	names = _attribute_names_for_ids(target_ids, attribute_names)
	group_index = build_attribute_group_index(names, attribute_groups)

	selected = set(available)
	required = set(always_include)
	_apply_group_dropout(selected, required, group_index, min_count, group_dropout, rng)
	_apply_attribute_dropout(selected, required, min_count, attribute_dropout, rng)
	_refill_to_minimum(selected, available, min_count, rng)
	_trim_to_maximum(selected, required, max_count, rng)

	mask = np.zeros(len(target_ids), dtype=np.bool_)
	for id_ in selected:
		mask[id_] = True
	return mask


def _apply_group_dropout(  # noqa: PLR0913
	selected: set[int],
	required: set[int],
	group_index: Mapping[str, tuple[int, ...]],
	min_count: int,
	group_dropout_prob: float,
	rng: np.random.Generator,
) -> None:
	for group_ids in group_index.values():
		if rng.random() >= group_dropout_prob:
			continue
		droppable = [
			id_ for id_ in group_ids if id_ in selected and id_ not in required
		]
		if droppable and len(selected) - len(droppable) >= min_count:
			selected.difference_update(droppable)


def _apply_attribute_dropout(
	selected: set[int],
	required: set[int],
	min_count: int,
	attribute_dropout_prob: float,
	rng: np.random.Generator,
) -> None:
	for id_ in sorted(selected - required):
		if len(selected) <= min_count:
			break
		if rng.random() < attribute_dropout_prob:
			selected.remove(id_)


def _refill_to_minimum(
	selected: set[int],
	available: tuple[int, ...],
	min_count: int,
	rng: np.random.Generator,
) -> None:
	needed = min_count - len(selected)
	if needed <= 0:
		return
	candidates = np.asarray(
		[id_ for id_ in available if id_ not in selected],
		dtype=np.int64,
	)
	if len(candidates) < needed:
		msg = (
			'not enough available attributes remain to satisfy '
			f'min_input_attributes={min_count!r}'
		)
		raise ValueError(msg)
	sampled = rng.choice(candidates, size=needed, replace=False)
	selected.update(int(id_) for id_ in sampled)


def _trim_to_maximum(
	selected: set[int],
	required: set[int],
	max_count: int,
	rng: np.random.Generator,
) -> None:
	remove_count = len(selected) - max_count
	if remove_count <= 0:
		return
	removable = np.asarray(sorted(selected - required), dtype=np.int64)
	if len(removable) < remove_count:
		msg = (
			'not enough non-required attributes can be dropped to satisfy '
			f'max_input_attributes={max_count!r}'
		)
		raise ValueError(msg)
	removed = rng.choice(removable, size=remove_count, replace=False)
	selected.difference_update(int(id_) for id_ in removed)


def _validate_attribute_names(attribute_names: Sequence[str]) -> tuple[str, ...]:
	if isinstance(attribute_names, str):
		msg = 'attribute_names must be a sequence of strings'
		raise TypeError(msg)
	names = tuple(attribute_names)
	if not names or not all(isinstance(name, str) for name in names):
		msg = f'attribute_names must be a non-empty string sequence; got {names!r}'
		raise ValueError(msg)
	if len(set(names)) != len(names):
		msg = f'attribute_names must be unique; got {names!r}'
		raise ValueError(msg)
	return names


def _validate_target_attribute_ids(
	target_attribute_ids: Sequence[int],
) -> tuple[int, ...]:
	target_ids = _validate_id_sequence(
		target_attribute_ids,
		'target_attribute_ids',
		target_id_set=None,
		sort=False,
	)
	expected = tuple(range(len(target_ids)))
	if target_ids != expected:
		msg = (
			'target_attribute_ids must be contiguous registry IDs in order; '
			f'expected {expected!r}, got {target_ids!r}'
		)
		raise ValueError(msg)
	return target_ids


def _validate_id_sequence(
	attribute_ids: Sequence[int],
	name: str,
	*,
	target_id_set: set[int] | None,
	sort: bool,
) -> tuple[int, ...]:
	if isinstance(attribute_ids, str):
		msg = f'{name} must be a sequence of integer IDs'
		raise TypeError(msg)
	ids: list[int] = []
	seen: set[int] = set()
	for raw_id in attribute_ids:
		if isinstance(raw_id, bool) or not isinstance(raw_id, Integral):
			msg = f'{name} values must be integers; got {raw_id!r}'
			raise TypeError(msg)
		id_ = int(raw_id)
		if target_id_set is not None and id_ not in target_id_set:
			msg = f'unknown target attribute ID in {name}: {id_!r}'
			raise ValueError(msg)
		if id_ in seen:
			msg = f'{name} must be unique; duplicate {id_!r}'
			raise ValueError(msg)
		seen.add(id_)
		ids.append(id_)
	if sort:
		return tuple(sorted(ids))
	return tuple(ids)


def _validate_count(value: int, name: str) -> int:
	if isinstance(value, bool) or not isinstance(value, Integral):
		msg = f'{name} must be an integer; got {value!r}'
		raise TypeError(msg)
	count = int(value)
	if count <= 0:
		msg = f'{name} must be positive; got {count!r}'
		raise ValueError(msg)
	return count


def _validate_probability(value: float, name: str) -> float:
	if isinstance(value, bool) or not isinstance(value, Real):
		msg = f'{name} must be a real number; got {value!r}'
		raise TypeError(msg)
	probability = float(value)
	if not 0.0 <= probability < 1.0:
		msg = f'{name} must be in [0, 1); got {probability!r}'
		raise ValueError(msg)
	return probability


def _attribute_names_for_ids(
	target_ids: tuple[int, ...],
	attribute_names: Sequence[str] | None,
) -> tuple[str, ...]:
	if attribute_names is not None:
		names = _validate_attribute_names(attribute_names)
		if len(names) != len(target_ids):
			msg = (
				'attribute_names length must match target_attribute_ids; '
				f'got {len(names)!r} and {len(target_ids)!r}'
			)
			raise ValueError(msg)
		return names

	if set(target_ids).issubset(_MVP_ATTRIBUTE_ID_SET):
		return tuple(MVP_ATTRIBUTE_REGISTRY.id_to_name(id_) for id_ in target_ids)

	msg = (
		'attribute_names must be provided when target_attribute_ids are not '
		'MVP attribute IDs'
	)
	raise ValueError(msg)


__all__ = [
	'AMPLITUDE_ATTRIBUTE_ID',
	'MVP_ATTRIBUTE_IDS',
	'build_attribute_group_index',
	'sample_attribute_input_mask',
]
