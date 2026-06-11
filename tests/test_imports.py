from __future__ import annotations

import importlib

import seis_attr_ssl

MODULES = (
	'seis_attr_ssl',
	'seis_attr_ssl.attributes',
	'seis_attr_ssl.config',
	'seis_attr_ssl.data',
	'seis_attr_ssl.evaluation',
	'seis_attr_ssl.inference',
	'seis_attr_ssl.losses',
	'seis_attr_ssl.masking',
	'seis_attr_ssl.models',
	'seis_attr_ssl.models.common',
	'seis_attr_ssl.models.mae',
	'seis_attr_ssl.models.tokenizers',
	'seis_attr_ssl.training',
	'seis_attr_ssl.utils',
)


def test_public_modules_import() -> None:
	for module_name in MODULES:
		importlib.import_module(module_name)


def test_package_version_is_non_empty() -> None:
	assert seis_attr_ssl.__version__
