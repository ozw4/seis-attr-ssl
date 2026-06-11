"""Shared helpers for thin procedure entrypoints."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from seis_attr_ssl.config import load_config, validate_config


def parse_config_args(
	description: str,
	default_config: str | Path | None = None,
) -> argparse.Namespace:
	"""Parse common config and dry-run arguments."""
	parser = argparse.ArgumentParser(description=description)
	parser.add_argument(
		'--config',
		type=Path,
		default=Path(default_config) if default_config is not None else None,
		required=default_config is None,
		help='Path to a YAML configuration file.',
	)
	parser.add_argument(
		'--dry-run',
		action='store_true',
		help='Validate the config and print a run summary without executing.',
	)
	return parser.parse_args()


def print_config_summary(cfg: Mapping[str, Any]) -> None:
	"""Print a compact summary of validated configuration values."""
	project = _mapping(cfg.get('project'))
	paths = _mapping(cfg.get('paths'))
	data = _mapping(cfg.get('data'))
	attributes = _mapping(cfg.get('attributes'))
	masking = _mapping(cfg.get('masking'))
	model = _mapping(cfg.get('model'))
	train = _mapping(cfg.get('train'))

	rows: list[tuple[str, Any]] = [
		('stage', cfg.get('stage')),
		('project.name', project.get('name')),
		('paths.nopims_root', paths.get('nopims_root')),
		('paths.output_root', paths.get('output_root')),
		('data.grid_order', data.get('grid_order')),
		('data.local_crop_size', data.get('local_crop_size')),
		('data.context_crop_size', data.get('context_crop_size')),
		('data.context_downsample', data.get('context_downsample')),
		('attributes.names count', _count(attributes.get('names'))),
		('attributes.groups count', _count(attributes.get('groups'))),
	]

	if 'name' in model:
		rows.append(('model.name', model.get('name')))
	rows.extend(
		(f'masking.{key}', masking.get(key))
		for key in (
			'spatial_mask_ratio',
			'spatial_mask_mode',
			'block_size_tokens',
			'min_input_attributes',
			'max_input_attributes',
			'attribute_dropout_prob',
			'group_dropout_prob',
		)
		if key in masking
	)
	if 'batch_size' in train:
		rows.append(('train.batch_size', train.get('batch_size')))
	if 'epochs' in train:
		rows.append(('train.epochs', train.get('epochs')))
	if 'device' in train:
		rows.append(('train.device', train.get('device')))
	if 'max_steps' in train:
		rows.append(('train.max_steps', train.get('max_steps')))

	for key, value in rows:
		print(f'{key}: {_format_value(value)}')


def run_config_entrypoint(
	description: str,
	default_config: str | Path | None = None,
) -> None:
	"""Run the common dry-run flow for procedure entrypoints."""
	args = parse_config_args(description, default_config)
	if not args.dry_run:
		message = (
			'Execution is not implemented yet. '
			'Use --dry-run to validate the config and print a summary.'
		)
		raise SystemExit(message)

	config = validate_config(load_config(args.config))
	print_config_summary(config)


def _mapping(value: object) -> Mapping[str, Any]:
	if isinstance(value, Mapping):
		return value
	return {}


def _count(value: object) -> int:
	if isinstance(value, Mapping | Sequence) and not isinstance(value, str | bytes):
		return len(value)
	return 0


def _format_value(value: object) -> str:
	if isinstance(value, list):
		return ', '.join(str(item) for item in value)
	if value is None:
		return 'null'
	return str(value)


__all__ = [
	'parse_config_args',
	'print_config_summary',
	'run_config_entrypoint',
]
