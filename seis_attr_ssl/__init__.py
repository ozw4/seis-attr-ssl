"""SeisAttrSSL package path bridge for repository-root commands."""

from __future__ import annotations

from pathlib import Path

_SRC_PACKAGE = Path(__file__).resolve().parent.parent / 'src' / 'seis_attr_ssl'
if not _SRC_PACKAGE.is_dir():
	msg = f'cannot find src-layout package at {_SRC_PACKAGE}'
	raise ImportError(msg)

__path__ = [str(_SRC_PACKAGE)]
__version__ = '0.1.0'

__all__ = ['__version__']
