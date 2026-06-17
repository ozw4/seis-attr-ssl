"""Generate on-the-fly MVP attribute comparison PNGs from a NumPy volume."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from seis_attr_ssl.attributes.on_the_fly import (  # noqa: E402
	attribute_generation_config_from_mapping,
)
from seis_attr_ssl.visualization.attribute_on_the_fly_compare import (  # noqa: E402
	OnTheFlyAttributeCompareConfig,
	save_on_the_fly_attribute_comparison_pngs,
)

DEFAULT_CONFIG = (
	Path(__file__).resolve().parent
	/ 'configs'
	/ 'visualize_attribute_on_the_fly_compare.yaml'
)


def main() -> None:
	"""Parse the YAML config and write comparison PNGs."""
	parser = argparse.ArgumentParser(
		description=(
			'Generate on-the-fly MVP attributes from an amplitude volume and '
			'save XY/XZ comparison PNGs.'
		),
	)
	parser.add_argument(
		'--config',
		type=Path,
		default=DEFAULT_CONFIG,
		help='Path to YAML config.',
	)
	args = parser.parse_args()

	cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
	if not isinstance(cfg, Mapping):
		msg = f'config must be a mapping: {args.config}'
		raise TypeError(msg)

	input_cfg = _required_mapping(cfg, 'input')
	output_cfg = _required_mapping(cfg, 'output')
	attr_cfg = _optional_mapping(cfg, 'attributes')
	vis_cfg = _optional_mapping(cfg, 'visualization')

	xy_path, xz_path = save_on_the_fly_attribute_comparison_pngs(
		_required_value(input_cfg, 'amplitude_npy'),
		_required_value(output_cfg, 'out_dir'),
		stem=str(output_cfg.get('stem', 'attribute_compare')),
		normalization_stats_json=input_cfg.get('normalization_stats_json'),
		assume_normalized=bool(input_cfg.get('assume_normalized', False)),
		attribute_names=attr_cfg.get('names'),  # type: ignore[arg-type]
		config=OnTheFlyAttributeCompareConfig(
			xy_slice_index=_optional_int(vis_cfg.get('xy_slice_index')),
			xz_slice_y_index=_optional_int(vis_cfg.get('xz_slice_y_index')),
			xy_z_window=int(vis_cfg.get('xy_z_window', 256)),
			xz_y_window=int(vis_cfg.get('xz_y_window', 64)),
			clip_percentiles=tuple(
				float(value) for value in vis_cfg.get('clip_percentiles', (1.0, 99.0))
			),  # type: ignore[arg-type]
			show_raw_amplitude=bool(vis_cfg.get('show_raw_amplitude', True)),
			use_known_ranges=bool(vis_cfg.get('use_known_ranges', True)),
			dpi=int(vis_cfg.get('dpi', 200)),
			figure_height=float(vis_cfg.get('figure_height', 4.0)),
			panel_width=float(vis_cfg.get('panel_width', 4.0)),
			xy_aspect=str(vis_cfg.get('xy_aspect', 'equal')),
			xz_aspect=str(vis_cfg.get('xz_aspect', 'auto')),
			grid_mode=str(vis_cfg.get('grid_mode', 'auto')),
		),
		attribute_generation_config=attribute_generation_config_from_mapping(
			cfg.get('attribute_generation'),
		),
	)
	print(f'wrote xy png: {xy_path}')
	print(f'wrote xz png: {xz_path}')


def _required_mapping(parent: Mapping[str, object], key: str) -> Mapping[str, object]:
	value = parent.get(key)
	if not isinstance(value, Mapping):
		msg = f'{key} must be a mapping'
		raise TypeError(msg)
	return value


def _optional_mapping(parent: Mapping[str, object], key: str) -> Mapping[str, object]:
	value = parent.get(key, {})
	if not isinstance(value, Mapping):
		msg = f'{key} must be a mapping'
		raise TypeError(msg)
	return value


def _required_value(parent: Mapping[str, object], key: str) -> object:
	value = parent.get(key)
	if value is None:
		msg = f'{key} is required'
		raise ValueError(msg)
	return value


def _optional_int(value: object) -> int | None:
	if value is None:
		return None
	return int(value)


if __name__ == '__main__':
	main()
