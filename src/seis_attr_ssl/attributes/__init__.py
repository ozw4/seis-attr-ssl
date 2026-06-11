"""Seismic attribute generation components."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from seis_attr_ssl.attributes.registry import (
		MVP_ATTRIBUTE_REGISTRY,
		AttributeRegistry,
		AttributeSpec,
	)

__all__ = ['MVP_ATTRIBUTE_REGISTRY', 'AttributeRegistry', 'AttributeSpec']


def __getattr__(name: str) -> object:
	"""Lazily expose attribute registry objects."""
	if name in __all__:
		return getattr(import_module('seis_attr_ssl.attributes.registry'), name)
	msg = f'module {__name__!r} has no attribute {name!r}'
	raise AttributeError(msg)
