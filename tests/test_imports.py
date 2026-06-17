from __future__ import annotations

import builtins
import importlib
import sys

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


def test_mae_training_import_does_not_require_matplotlib(monkeypatch) -> None:
	real_import = builtins.__import__

	def guarded_import(name, *args, **kwargs):
		if name == 'matplotlib' or name.startswith('matplotlib.'):
			msg = f'imported plotting dependency while importing {name!r}'
			raise AssertionError(msg)
		return real_import(name, *args, **kwargs)

	monkeypatch.delattr(seis_attr_ssl, 'training', raising=False)
	monkeypatch.delattr(seis_attr_ssl, 'visualization', raising=False)
	for module_name in (
		'seis_attr_ssl.training',
		'seis_attr_ssl.training.mae',
		'seis_attr_ssl.visualization',
		'seis_attr_ssl.visualization.mae_debug',
		'seis_attr_ssl.visualization.attribute_on_the_fly_compare',
	):
		monkeypatch.delitem(sys.modules, module_name, raising=False)

	monkeypatch.setattr(builtins, '__import__', guarded_import)

	importlib.import_module('seis_attr_ssl.training.mae')


def test_package_version_is_non_empty() -> None:
	assert seis_attr_ssl.__version__
