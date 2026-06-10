"""Configuration loading and validation utilities."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from seis_attr_ssl.config.io import load_config
	from seis_attr_ssl.config.validate import validate_config

__all__ = ['load_config', 'validate_config']


def __getattr__(name: str) -> object:
	if name == 'load_config':
		return import_module('seis_attr_ssl.config.io').load_config
	if name == 'validate_config':
		return import_module('seis_attr_ssl.config.validate').validate_config
	msg = f'module {__name__!r} has no attribute {name!r}'
	raise AttributeError(msg)
