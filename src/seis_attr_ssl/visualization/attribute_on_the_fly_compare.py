"""On-the-fly MVP attribute slice comparison PNGs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from seis_attr_ssl.attributes import MVP_ATTRIBUTE_REGISTRY
from seis_attr_ssl.attributes.on_the_fly import (
	AttributeGenerationConfig,
	generate_mvp_attributes_for_payload,
	normalize_base_seismic,
)
from seis_attr_ssl.data.normalization import (
	SurveyNormalizationStats,
	load_normalization_stats,
)

if TYPE_CHECKING:
	from collections.abc import Sequence


@dataclass(frozen=True)
class OnTheFlyAttributeCompareConfig:
	"""Configuration for on-the-fly attribute slice comparison PNGs."""

	xy_slice_index: int | None = None
	xz_slice_y_index: int | None = None
	xy_z_window: int = 256
	xz_y_window: int = 64
	clip_percentiles: tuple[float, float] = (1.0, 99.0)
	show_raw_amplitude: bool = True
	use_known_ranges: bool = True
	show_valid_mask: bool = True
	mask_invalid_values: bool = True
	invalid_color: str = 'lightgray'
	dpi: int = 200
	figure_height: float = 4.0
	panel_width: float = 4.0
	xy_aspect: str = 'equal'
	xz_aspect: str = 'auto'
	grid_mode: str = 'auto'


@dataclass(frozen=True)
class _Panel:
	name: str
	image: np.ndarray
	valid_mask: np.ndarray | None = None


_KNOWN_RANGES: dict[str, tuple[float, float]] = {
	'phase_sin': (-1.0, 1.0),
	'phase_cos': (-1.0, 1.0),
	'spectral_low_ratio': (0.0, 1.0),
	'spectral_mid_ratio': (0.0, 1.0),
	'spectral_high_ratio': (0.0, 1.0),
	'coherence': (0.0, 1.0),
	'glcm_homogeneity': (0.0, 1.0),
	'valid_mask': (0.0, 1.0),
}


def save_on_the_fly_attribute_comparison_pngs(  # noqa: PLR0913
	amplitude_npy: str | Path,
	out_dir: str | Path,
	*,
	stem: str = 'attribute_compare',
	normalization_stats_json: str | Path | None = None,
	assume_normalized: bool = False,
	attribute_names: Sequence[str] | None = None,
	config: OnTheFlyAttributeCompareConfig | None = None,
	attribute_generation_config: AttributeGenerationConfig | None = None,
) -> tuple[Path, Path]:
	"""Generate MVP attributes on the fly from one amplitude volume and save PNGs.

	The input amplitude volume must be a 3D NumPy array in [x, y, z] order.
	If assume_normalized is false, normalization_stats_json is required.
	"""
	amplitude_path = Path(amplitude_npy)
	output_dir = Path(out_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	volume = np.load(amplitude_path, mmap_mode='r')
	_validate_volume(volume, amplitude_path)
	config = config or OnTheFlyAttributeCompareConfig()

	if assume_normalized:
		stats = None
	elif normalization_stats_json is None:
		msg = 'normalization_stats_json is required unless assume_normalized=true'
		raise ValueError(msg)
	else:
		stats = load_normalization_stats(normalization_stats_json)

	names = _resolve_attribute_names(attribute_names)
	ids = [MVP_ATTRIBUTE_REGISTRY.name_to_id(name) for name in names]
	attribute_generation_config = (
		attribute_generation_config or AttributeGenerationConfig()
	)
	attribute_generation_config.validate()

	xy_png = output_dir / f'{stem}_xy.png'
	xz_png = output_dir / f'{stem}_xz.png'
	_save_xy_png(
		volume,
		stats,
		assume_normalized=assume_normalized,
		names=names,
		ids=ids,
		out_path=xy_png,
		config=config,
		attribute_generation_config=attribute_generation_config,
	)
	_save_xz_png(
		volume,
		stats,
		assume_normalized=assume_normalized,
		names=names,
		ids=ids,
		out_path=xz_png,
		config=config,
		attribute_generation_config=attribute_generation_config,
	)
	return xy_png, xz_png


def _save_xy_png(  # noqa: PLR0913
	volume: np.ndarray,
	stats: SurveyNormalizationStats | None,
	*,
	assume_normalized: bool,
	names: Sequence[str],
	ids: Sequence[int],
	out_path: Path,
	config: OnTheFlyAttributeCompareConfig,
	attribute_generation_config: AttributeGenerationConfig,
) -> None:
	_nx, _ny, nz = volume.shape
	z_idx = nz // 2 if config.xy_slice_index is None else int(config.xy_slice_index)
	if not 0 <= z_idx < nz:
		msg = f'xy_slice_index out of range: {z_idx}, valid=[0, {nz - 1}]'
		raise ValueError(msg)

	z_start, z_stop, z_local = _centered_interval(
		center=z_idx,
		size=config.xy_z_window,
		limit=nz,
	)
	raw_crop = np.asarray(volume[:, :, z_start:z_stop], dtype=np.float32)
	norm_crop = _normalize_crop(
		raw_crop,
		stats,
		assume_normalized=assume_normalized,
	)
	payload_slices = (
		slice(0, norm_crop.shape[0]),
		slice(0, norm_crop.shape[1]),
		slice(0, norm_crop.shape[2]),
	)
	result = generate_mvp_attributes_for_payload(
		norm_crop,
		payload_slices,
		config=attribute_generation_config,
	)

	valid_2d = result.voxel_valid_mask[:, :, z_local].T
	panels = _build_panels(
		raw_amplitude=raw_crop[:, :, z_local].T,
		attribute_images=[
			(name, result.attributes[attr_id, :, :, z_local].T)
			for name, attr_id in zip(names, ids, strict=True)
		],
		valid_mask_2d=valid_2d,
		config=config,
	)

	_plot_panels(
		panels,
		title=(
			f'XY slice on-the-fly attributes (z={z_idx}, z_window={z_stop - z_start})'
		),
		out_path=out_path,
		xlabel='x',
		ylabel='y',
		aspect=config.xy_aspect,
		config=config,
	)


def _save_xz_png(  # noqa: PLR0913
	volume: np.ndarray,
	stats: SurveyNormalizationStats | None,
	*,
	assume_normalized: bool,
	names: Sequence[str],
	ids: Sequence[int],
	out_path: Path,
	config: OnTheFlyAttributeCompareConfig,
	attribute_generation_config: AttributeGenerationConfig,
) -> None:
	_nx, ny, _nz = volume.shape
	y_idx = ny // 2 if config.xz_slice_y_index is None else int(config.xz_slice_y_index)
	if not 0 <= y_idx < ny:
		msg = f'xz_slice_y_index out of range: {y_idx}, valid=[0, {ny - 1}]'
		raise ValueError(msg)

	y_start, y_stop, y_local = _centered_interval(
		center=y_idx,
		size=config.xz_y_window,
		limit=ny,
	)
	raw_crop = np.asarray(volume[:, y_start:y_stop, :], dtype=np.float32)
	norm_crop = _normalize_crop(
		raw_crop,
		stats,
		assume_normalized=assume_normalized,
	)
	payload_slices = (
		slice(0, norm_crop.shape[0]),
		slice(0, norm_crop.shape[1]),
		slice(0, norm_crop.shape[2]),
	)
	result = generate_mvp_attributes_for_payload(
		norm_crop,
		payload_slices,
		config=attribute_generation_config,
	)

	valid_2d = result.voxel_valid_mask[:, y_local, :].T
	panels = _build_panels(
		raw_amplitude=raw_crop[:, y_local, :].T,
		attribute_images=[
			(name, result.attributes[attr_id, :, y_local, :].T)
			for name, attr_id in zip(names, ids, strict=True)
		],
		valid_mask_2d=valid_2d,
		config=config,
	)

	_plot_panels(
		panels,
		title=(
			f'XZ slice on-the-fly attributes (y={y_idx}, y_window={y_stop - y_start})'
		),
		out_path=out_path,
		xlabel='x',
		ylabel='z',
		aspect=config.xz_aspect,
		config=config,
	)


def _normalize_crop(
	raw_crop: np.ndarray,
	stats: SurveyNormalizationStats | None,
	*,
	assume_normalized: bool,
) -> np.ndarray:
	if assume_normalized:
		return np.asarray(raw_crop, dtype=np.float32)
	if stats is None:
		msg = 'stats must be provided when assume_normalized=false'
		raise ValueError(msg)
	return normalize_base_seismic(raw_crop, stats)


def _centered_interval(
	*,
	center: int,
	size: int,
	limit: int,
) -> tuple[int, int, int]:
	if size <= 0:
		msg = f'window size must be positive; got {size!r}'
		raise ValueError(msg)
	stop = min(limit, max(size, center + size // 2 + 1))
	start = max(0, stop - size)
	stop = min(limit, start + size)
	return start, stop, center - start


def _build_panels(
	*,
	raw_amplitude: np.ndarray,
	attribute_images: Sequence[tuple[str, np.ndarray]],
	valid_mask_2d: np.ndarray,
	config: OnTheFlyAttributeCompareConfig,
) -> list[_Panel]:
	valid_mask = np.asarray(valid_mask_2d, dtype=bool)
	panels: list[_Panel] = []
	if config.show_raw_amplitude:
		panels.append(_Panel('raw_amplitude', raw_amplitude, valid_mask))
	for name, image in attribute_images:
		panels.append(_Panel(name, image, valid_mask))
	if config.show_valid_mask:
		# valid_mask panel convention: 1 = valid voxel, 0 = invalid voxel.
		panels.append(_Panel('valid_mask', valid_mask.astype(np.float32)))
	return panels


def _plot_panels(  # noqa: PLR0913
	panels: Sequence[_Panel],
	title: str,
	out_path: Path,
	xlabel: str,
	ylabel: str,
	aspect: str,
	config: OnTheFlyAttributeCompareConfig,
) -> None:
	n_panels = len(panels)
	nrows, ncols = _resolve_grid(n_panels, config.grid_mode)
	fig, axes = plt.subplots(
		nrows,
		ncols,
		figsize=(config.panel_width * ncols, config.figure_height * nrows),
		squeeze=False,
	)
	flat_axes = list(axes.ravel())
	for ax, panel in zip(flat_axes, panels, strict=False):
		image = _display_image(panel, config)
		vmin, vmax = _display_limits(
			panel.name,
			image,
			config.clip_percentiles,
			use_known_ranges=config.use_known_ranges,
		)
		im = ax.imshow(
			image,
			origin='upper',
			aspect=aspect,
			cmap=_cmap_for_panel(panel, config),
			vmin=vmin,
			vmax=vmax,
		)
		ax.set_title(panel.name)
		ax.set_xlabel(xlabel)
		ax.set_ylabel(ylabel)
		fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
	for ax in flat_axes[n_panels:]:
		ax.axis('off')
	fig.suptitle(title)
	fig.tight_layout()
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_path, dpi=config.dpi, bbox_inches='tight')
	plt.close(fig)


def _resolve_grid(n_panels: int, grid_mode: str) -> tuple[int, int]:
	mode = str(grid_mode).strip().lower()
	if mode == 'auto':
		if n_panels <= 0:
			msg = f'n_panels must be positive; got {n_panels!r}'
			raise ValueError(msg)
		if n_panels == 10:
			return 2, 5
		if 11 <= n_panels <= 12:
			return 3, 4
		ncols = math.ceil(math.sqrt(n_panels))
		nrows = math.ceil(n_panels / ncols)
		return nrows, ncols
	try:
		rows_str, cols_str = mode.split('x', maxsplit=1)
		nrows = int(rows_str)
		ncols = int(cols_str)
	except Exception as exc:
		msg = f"invalid grid_mode={grid_mode!r}; expected 'auto' or like '2x5'"
		raise ValueError(msg) from exc
	if nrows <= 0 or ncols <= 0:
		msg = f'grid_mode must use positive integers; got {grid_mode!r}'
		raise ValueError(msg)
	if nrows * ncols < n_panels:
		msg = (
			f'grid_mode={grid_mode!r} has only {nrows * ncols} slots but '
			f'{n_panels} panels are required'
		)
		raise ValueError(msg)
	return nrows, ncols


def _display_limits(
	name: str,
	image: np.ndarray,
	clip_percentiles: tuple[float, float],
	*,
	use_known_ranges: bool,
) -> tuple[float | None, float | None]:
	if use_known_ranges and name in _KNOWN_RANGES:
		return _KNOWN_RANGES[name]

	finite = np.ma.masked_invalid(image).compressed()
	if finite.size == 0:
		return None, None
	vmin, vmax = np.percentile(finite, clip_percentiles)
	if np.isclose(vmin, vmax):
		center = float(np.mean(finite))
		half = float(np.std(finite)) or 1.0
		return center - half, center + half
	return float(vmin), float(vmax)


def _display_image(
	panel: _Panel,
	config: OnTheFlyAttributeCompareConfig,
) -> np.ndarray | np.ma.MaskedArray:
	if (
		not config.mask_invalid_values
		or panel.valid_mask is None
		or panel.name == 'valid_mask'
	):
		return panel.image
	return np.ma.masked_where(~panel.valid_mask, panel.image)


def _cmap_for_panel(
	panel: _Panel,
	config: OnTheFlyAttributeCompareConfig,
) -> str | plt.Colormap:
	if panel.name == 'valid_mask':
		return 'gray'
	name = panel.name
	cmap_name = 'twilight' if name.startswith('phase_') else 'viridis'
	if (
		not config.mask_invalid_values
		or panel.valid_mask is None
		or panel.valid_mask.all()
	):
		return cmap_name
	cmap = plt.get_cmap(cmap_name).copy()
	cmap.set_bad(config.invalid_color)
	return cmap


def _resolve_attribute_names(
	attribute_names: Sequence[str] | None,
) -> tuple[str, ...]:
	names = (
		MVP_ATTRIBUTE_REGISTRY.names
		if attribute_names is None
		else tuple(attribute_names)
	)
	for name in names:
		MVP_ATTRIBUTE_REGISTRY.name_to_id(name)
	return names


def _validate_volume(volume: np.ndarray, path: Path) -> None:
	if volume.ndim != 3:
		msg = f'volume must be 3D [x, y, z]: {path}; got shape={volume.shape!r}'
		raise ValueError(msg)
	if not np.issubdtype(volume.dtype, np.number):
		msg = f'volume must be numeric: {path}; got dtype={volume.dtype!r}'
		raise TypeError(msg)


__all__ = [
	'OnTheFlyAttributeCompareConfig',
	'save_on_the_fly_attribute_comparison_pngs',
]
