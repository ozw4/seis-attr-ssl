"""Visualization helpers for seismic attribute workflows."""

from __future__ import annotations

import importlib

__all__ = [
	'OnTheFlyAttributeCompareConfig',
	'save_on_the_fly_attribute_comparison_pngs',
]


def __getattr__(name: str) -> object:
	if name in __all__:
		module = importlib.import_module(
			'seis_attr_ssl.visualization.attribute_on_the_fly_compare',
		)
		return getattr(module, name)
	msg = f'module {__name__!r} has no attribute {name!r}'
	raise AttributeError(msg)
