from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

Array3D = np.ndarray


@dataclass(frozen=True)
class SliceComparisonConfig:
	xy_slice_index: int | None = None
	xz_slice_y_index: int | None = None
	clip_percentiles: tuple[float, float] = (1.0, 99.0)
	dpi: int = 200
	figure_height: float = 4.0
	panel_width: float = 4.0
	xz_aspect: str = 'auto'  # "auto" 推奨
	xy_aspect: str = 'equal'  # xy は equal で見やすい


def _validate_same_shape(
	amplitude_xyz: Array3D,
	attribute_volumes_xyz: Mapping[str, Array3D],
) -> None:
	if amplitude_xyz.ndim != 3:
		raise ValueError(
			f'amplitude volume must be 3D; got shape={amplitude_xyz.shape}'
		)

	base_shape = amplitude_xyz.shape
	for name, vol in attribute_volumes_xyz.items():
		if vol.ndim != 3:
			raise ValueError(f"attribute '{name}' must be 3D; got shape={vol.shape}")
		if vol.shape != base_shape:
			raise ValueError(
				f"attribute '{name}' shape mismatch: expected {base_shape}, got {vol.shape}"
			)


def _resolve_indices(
	shape_xyz: tuple[int, int, int],
	xy_slice_index: int | None,
	xz_slice_y_index: int | None,
) -> tuple[int, int]:
	nx, ny, nz = shape_xyz

	z_idx = nz // 2 if xy_slice_index is None else xy_slice_index
	y_idx = ny // 2 if xz_slice_y_index is None else xz_slice_y_index

	if not (0 <= z_idx < nz):
		raise ValueError(f'xy_slice_index out of range: {z_idx}, valid=[0, {nz - 1}]')
	if not (0 <= y_idx < ny):
		raise ValueError(f'xz_slice_y_index out of range: {y_idx}, valid=[0, {ny - 1}]')

	return z_idx, y_idx


def _robust_limits(
	slice_2d: np.ndarray, q_low: float, q_high: float
) -> tuple[float, float]:
	finite = slice_2d[np.isfinite(slice_2d)]
	if finite.size == 0:
		return 0.0, 1.0

	vmin = float(np.percentile(finite, q_low))
	vmax = float(np.percentile(finite, q_high))

	if np.isclose(vmin, vmax):
		center = float(np.mean(finite))
		half = float(np.std(finite))
		if half == 0.0:
			half = 1.0
		return center - half, center + half

	return vmin, vmax


def _xy_slice(volume_xyz: Array3D, z_idx: int) -> np.ndarray:
	# [x, y, z] -> xy表示用に [y, x]
	return volume_xyz[:, :, z_idx].T


def _xz_slice(volume_xyz: Array3D, y_idx: int) -> np.ndarray:
	# [x, y, z] -> xz表示用に [z, x]
	return volume_xyz[:, y_idx, :].T


def _plot_panel_row(
	slices: list[np.ndarray],
	names: list[str],
	title_prefix: str,
	out_path: Path,
	config: SliceComparisonConfig,
	aspect: str,
) -> None:
	n = len(slices)
	fig, axes = plt.subplots(
		1,
		n,
		figsize=(config.panel_width * n, config.figure_height),
		squeeze=False,
	)
	axes = axes[0]

	q_low, q_high = config.clip_percentiles

	for ax, name, img in zip(axes, names, slices, strict=True):
		vmin, vmax = _robust_limits(img, q_low, q_high)
		cmap = 'gray' if name == 'amplitude' else 'viridis'

		im = ax.imshow(
			img,
			cmap=cmap,
			vmin=vmin,
			vmax=vmax,
			origin='upper',
			aspect=aspect,
		)
		ax.set_title(name)
		ax.set_xlabel('x')
		ax.set_ylabel('y/z')
		fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

	fig.suptitle(title_prefix)
	fig.tight_layout()
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_path, dpi=config.dpi, bbox_inches='tight')
	plt.close(fig)


def save_attribute_comparison_pngs(
	amplitude_xyz: Array3D,
	attribute_volumes_xyz: Mapping[str, Array3D],
	out_dir: Path,
	config: SliceComparisonConfig = SliceComparisonConfig(),
	stem: str = 'attribute_compare',
) -> tuple[Path, Path]:
	"""Save two PNGs:
	  - <stem>_xy.png
	  - <stem>_xz.png

	amplitude_xyz and all attribute volumes must be in [x, y, z] order.
	"""
	_validate_same_shape(amplitude_xyz, attribute_volumes_xyz)

	z_idx, y_idx = _resolve_indices(
		shape_xyz=tuple(amplitude_xyz.shape),
		xy_slice_index=config.xy_slice_index,
		xz_slice_y_index=config.xz_slice_y_index,
	)

	names = ['amplitude', *attribute_volumes_xyz.keys()]

	xy_slices = [_xy_slice(amplitude_xyz, z_idx)] + [
		_xy_slice(vol, z_idx) for vol in attribute_volumes_xyz.values()
	]
	xz_slices = [_xz_slice(amplitude_xyz, y_idx)] + [
		_xz_slice(vol, y_idx) for vol in attribute_volumes_xyz.values()
	]

	xy_path = out_dir / f'{stem}_xy.png'
	xz_path = out_dir / f'{stem}_xz.png'

	_plot_panel_row(
		slices=xy_slices,
		names=names,
		title_prefix=f'XY slice comparison (z={z_idx})',
		out_path=xy_path,
		config=config,
		aspect=config.xy_aspect,
	)
	_plot_panel_row(
		slices=xz_slices,
		names=names,
		title_prefix=f'XZ slice comparison (y={y_idx})',
		out_path=xz_path,
		config=config,
		aspect=config.xz_aspect,
	)

	return xy_path, xz_path
