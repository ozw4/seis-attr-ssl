"""YAML IO for SeisAttrSSL configuration files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from seis_attr_ssl.config.schema import DEFAULT_NOPIMS_ROOT
from seis_attr_ssl.config.validate import validate_config


def load_config(path: str | Path) -> dict[str, object]:
	"""Load a YAML config file and apply lightweight defaults."""
	config_path = Path(path)
	with config_path.open(encoding='utf-8') as file_obj:
		loaded = yaml.safe_load(file_obj)

	if not isinstance(loaded, dict):
		msg = f'config file must contain a mapping: {config_path}'
		raise TypeError(msg)

	paths = loaded.setdefault('paths', {})
	if not isinstance(paths, dict):
		msg = 'paths must be a mapping'
		raise TypeError(msg)
	paths.setdefault('nopims_root', DEFAULT_NOPIMS_ROOT)

	return loaded


def main() -> None:
	"""Load, validate, and print a compact summary for one config file."""
	parser = argparse.ArgumentParser(
		description='Validate a SeisAttrSSL config YAML file.',
	)
	parser.add_argument('config_path', type=Path)
	args = parser.parse_args()

	config = validate_config(load_config(args.config_path))
	summary = {
		'project': config.get('project', {}),
		'paths': config.get('paths', {}),
		'data': config.get('data', {}),
		'attributes': {
			'names': config.get('attributes', {}).get('names', []),
			'count': len(config.get('attributes', {}).get('names', [])),
		},
	}
	print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
	main()
