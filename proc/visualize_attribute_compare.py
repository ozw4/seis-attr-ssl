from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from seis_attr_ssl.visualization.attribute_compare import (
	SliceComparisonConfig,
	save_attribute_comparison_pngs,
)


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description='Visualize amplitude vs attribute volumes.'
	)
	parser.add_argument(
		'--config',
		type=Path,
		required=True,
		help='Path to YAML config.',
	)
	return parser.parse_args()


def _load_npy(path: str | Path) -> np.ndarray:
	return np.load(path, mmap_mode='r')


def main() -> None:
	args = _parse_args()
	cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))

	amplitude_path = Path(cfg['input']['amplitude_npy'])
	attribute_paths = {
		name: Path(path) for name, path in cfg['input']['attribute_npys'].items()
	}
	out_dir = Path(cfg['output']['out_dir'])
	stem = cfg['output'].get('stem', 'attribute_compare')

	vis_cfg = SliceComparisonConfig(
		xy_slice_index=cfg['visualization'].get('xy_slice_index'),
		xz_slice_y_index=cfg['visualization'].get('xz_slice_y_index'),
		clip_percentiles=tuple(
			cfg['visualization'].get('clip_percentiles', [1.0, 99.0])
		),
		dpi=int(cfg['visualization'].get('dpi', 200)),
		figure_height=float(cfg['visualization'].get('figure_height', 4.0)),
		panel_width=float(cfg['visualization'].get('panel_width', 4.0)),
		xz_aspect=str(cfg['visualization'].get('xz_aspect', 'auto')),
		xy_aspect=str(cfg['visualization'].get('xy_aspect', 'equal')),
	)

	amplitude_xyz = _load_npy(amplitude_path)
	attribute_volumes_xyz = {
		name: _load_npy(path) for name, path in attribute_paths.items()
	}

	xy_path, xz_path = save_attribute_comparison_pngs(
		amplitude_xyz=amplitude_xyz,
		attribute_volumes_xyz=attribute_volumes_xyz,
		out_dir=out_dir,
		config=vis_cfg,
		stem=stem,
	)

	print(f'wrote xy png: {xy_path}')
	print(f'wrote xz png: {xz_path}')


if __name__ == '__main__':
	main()
