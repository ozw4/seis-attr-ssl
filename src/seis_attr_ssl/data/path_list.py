"""Path-list utilities for explicit NOPIMS `.npy` training manifests."""

from __future__ import annotations

import re
from pathlib import Path

_SURVEY_ID_ALLOWED = re.compile(r'[^A-Za-z0-9_.-]')


def load_npy_path_list(path_list: str | Path) -> list[str]:
	"""Load non-empty, non-comment entries from a plain text path-list file."""
	source = Path(path_list)
	if not source.is_file():
		msg = f'path-list file does not exist: {source}'
		raise FileNotFoundError(msg)

	entries: list[str] = []
	for line in source.read_text(encoding='utf-8').splitlines():
		entry = line.strip()
		if not entry or entry.startswith('#'):
			continue
		entries.append(entry)
	return entries


def resolve_npy_path_list(
	path_list: str | Path,
	nopims_root: str | Path,
) -> list[Path]:
	"""Resolve and validate `.npy` paths from a path-list file."""
	root = Path(nopims_root)
	paths: list[Path] = []
	seen: dict[Path, Path] = {}
	for entry in load_npy_path_list(path_list):
		path = Path(entry)
		if not path.is_absolute():
			path = root / path
		if path.suffix != '.npy':
			msg = f'path-list entry must have .npy suffix: {path}'
			raise ValueError(msg)
		if not path.is_file():
			msg = f'path-list entry does not exist: {path}'
			raise FileNotFoundError(msg)
		key = path.resolve(strict=True)
		if key in seen:
			msg = f'duplicate path-list entry: {seen[key]} and {path}'
			raise ValueError(msg)
		seen[key] = path
		paths.append(path)
	return paths


def make_survey_id_from_path(path: str | Path, nopims_root: str | Path) -> str:
	"""Create a deterministic survey id from an `.npy` path."""
	volume_path = Path(path)
	root = Path(nopims_root)
	try:
		source = volume_path.relative_to(root)
	except ValueError:
		source = volume_path

	source_without_suffix = source.with_suffix('')
	parts = source_without_suffix.parts
	raw_id = '__'.join(parts)
	survey_id = _SURVEY_ID_ALLOWED.sub('_', raw_id)
	if not survey_id:
		msg = f'could not derive survey_id from path: {volume_path}'
		raise ValueError(msg)
	return survey_id


__all__ = [
	'load_npy_path_list',
	'make_survey_id_from_path',
	'resolve_npy_path_list',
]
