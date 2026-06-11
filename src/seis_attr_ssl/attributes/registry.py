"""Typed registry for MVP seismic attributes."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

from seis_attr_ssl.config.schema import EXPECTED_ATTRIBUTE_GROUPS, EXPECTED_ATTRIBUTES

if TYPE_CHECKING:
	from collections.abc import Mapping, Sequence


@dataclass(frozen=True)
class AttributeSpec:
	"""Single attribute definition with a stable channel ID."""

	name: str
	id: int
	group: str
	dtype: str = 'float32'
	bounded_range: tuple[float, float] | None = None
	description: str | None = None


@dataclass(frozen=True)
class AttributeRegistry:
	"""Immutable ordered collection of attribute specifications."""

	specs: tuple[AttributeSpec, ...]
	_name_to_spec: Mapping[str, AttributeSpec] = field(
		init=False,
		repr=False,
		compare=False,
	)
	_id_to_spec: Mapping[int, AttributeSpec] = field(
		init=False,
		repr=False,
		compare=False,
	)
	_groups: Mapping[str, str] = field(init=False, repr=False, compare=False)

	def __post_init__(self) -> None:
		"""Build immutable lookup maps and validate registry shape."""
		specs = tuple(self.specs)
		names = tuple(spec.name for spec in specs)
		ids = tuple(spec.id for spec in specs)

		if len(set(names)) != len(names):
			msg = f'attribute names must be unique; got {names!r}'
			raise ValueError(msg)
		if len(set(ids)) != len(ids):
			msg = f'attribute IDs must be unique; got {ids!r}'
			raise ValueError(msg)
		expected_ids = tuple(range(len(specs)))
		if ids != expected_ids:
			msg = f'attribute IDs must be contiguous and ordered; got {ids!r}'
			raise ValueError(msg)

		object.__setattr__(self, 'specs', specs)
		object.__setattr__(
			self,
			'_name_to_spec',
			MappingProxyType({spec.name: spec for spec in specs}),
		)
		object.__setattr__(
			self,
			'_id_to_spec',
			MappingProxyType({spec.id: spec for spec in specs}),
		)
		object.__setattr__(
			self,
			'_groups',
			MappingProxyType({spec.name: spec.group for spec in specs}),
		)

	@property
	def names(self) -> tuple[str, ...]:
		"""Attribute names in stable channel order."""
		return tuple(spec.name for spec in self.specs)

	@property
	def groups(self) -> Mapping[str, str]:
		"""Immutable mapping from attribute name to group name."""
		return self._groups

	def name_to_id(self, name: str) -> int:
		"""Return the stable ID for an attribute name."""
		try:
			return self._name_to_spec[name].id
		except KeyError as exc:
			msg = f'unknown attribute name: {name!r}'
			raise KeyError(msg) from exc

	def id_to_name(self, id_: int) -> str:
		"""Return the attribute name for a stable ID."""
		try:
			return self._id_to_spec[id_].name
		except KeyError as exc:
			msg = f'unknown attribute ID: {id_!r}'
			raise KeyError(msg) from exc

	def spec(self, name_or_id: str | int) -> AttributeSpec:
		"""Return an attribute specification by name or stable ID."""
		if isinstance(name_or_id, str):
			try:
				return self._name_to_spec[name_or_id]
			except KeyError as exc:
				msg = f'unknown attribute name: {name_or_id!r}'
				raise KeyError(msg) from exc
		if isinstance(name_or_id, int):
			try:
				return self._id_to_spec[name_or_id]
			except KeyError as exc:
				msg = f'unknown attribute ID: {name_or_id!r}'
				raise KeyError(msg) from exc

		msg = f'attribute spec key must be str or int; got {type(name_or_id).__name__}'
		raise TypeError(msg)

	def group_names(self) -> tuple[str, ...]:
		"""Return group names in first-seen registry order."""
		seen: set[str] = set()
		ordered_groups: list[str] = []
		for spec in self.specs:
			if spec.group not in seen:
				ordered_groups.append(spec.group)
				seen.add(spec.group)
		return tuple(ordered_groups)

	def ids_for_group(self, group: str) -> tuple[int, ...]:
		"""Return stable IDs for all attributes in a group."""
		ids = tuple(spec.id for spec in self.specs if spec.group == group)
		if not ids:
			msg = f'unknown attribute group: {group!r}'
			raise KeyError(msg)
		return ids

	def validate_names(self, names: Sequence[str]) -> None:
		"""Validate that names exactly match registry order."""
		actual = tuple(names)
		if actual != self.names:
			msg = f'attribute names must equal {self.names!r}; got {actual!r}'
			raise ValueError(msg)

	def validate_groups(self, groups: Mapping[str, str]) -> None:
		"""Validate that groups exactly match registry group mapping."""
		actual = dict(groups)
		expected = dict(self.groups)
		if actual != expected:
			msg = f'attribute groups must equal {expected!r}; got {actual!r}'
			raise ValueError(msg)


MVP_ATTRIBUTE_REGISTRY = AttributeRegistry(
	specs=tuple(
		AttributeSpec(
			name=name,
			id=index,
			group=EXPECTED_ATTRIBUTE_GROUPS[name],
		)
		for index, name in enumerate(EXPECTED_ATTRIBUTES)
	),
)

MVP_ATTRIBUTE_REGISTRY.validate_names(EXPECTED_ATTRIBUTES)
MVP_ATTRIBUTE_REGISTRY.validate_groups(EXPECTED_ATTRIBUTE_GROUPS)

__all__ = ['MVP_ATTRIBUTE_REGISTRY', 'AttributeRegistry', 'AttributeSpec']
