"""Array utilities and PNG renderer for MAE debug visualizations."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import torch

from seis_attr_ssl.attributes import MVP_ATTRIBUTE_REGISTRY

if TYPE_CHECKING:
	from pathlib import Path

ArrayLike: TypeAlias = torch.Tensor | np.ndarray


@dataclass(frozen=True)
class MaeDebugVisualizationConfig:
	"""Configuration for MAE debug visualization PNGs."""

	output_dir: Path
	attributes: tuple[str, ...]
	every_n_steps: int | None = None
	every_n_epochs: int | None = 1
	max_batches_per_trigger: int = 1
	max_samples_per_batch: int = 1
	fail_on_error: bool = True
	columns: tuple[str, ...] = (
		'input',
		'masked_input',
		'target',
		'prediction',
		'abs_error',
	)
	xy_slice_index: int | None = None
	xz_slice_y_index: int | None = None
	grid_mode: str = 'auto'
	dpi: int = 160
	panel_width: float = 3.2
	panel_height: float = 2.8
	clip_percentiles: tuple[float, float] = (1.0, 99.0)
	use_known_ranges: bool = True
	mask_invalid_values: bool = True
	show_valid_mask_panel: bool = True
	show_spatial_mask_panel: bool = True
	invalid_color: str = 'lightgray'


@dataclass(frozen=True)
class _Panel:
	title: str
	image: np.ndarray
	valid_mask: np.ndarray | None = None
	range_name: str | None = None


_KNOWN_RANGES: dict[str, tuple[float, float]] = {
	'phase_sin': (-1.0, 1.0),
	'phase_cos': (-1.0, 1.0),
	'spectral_low_ratio': (0.0, 1.0),
	'spectral_mid_ratio': (0.0, 1.0),
	'spectral_high_ratio': (0.0, 1.0),
	'coherence': (0.0, 1.0),
	'glcm_homogeneity': (0.0, 1.0),
	'local_valid_mask': (0.0, 1.0),
	'spatial_mask_voxel': (0.0, 1.0),
}

_DEFAULT_COLUMNS = {
	'input',
	'masked_input',
	'target',
	'prediction',
	'abs_error',
}


def unpatchify_mae_predictions(
	pred_patches: ArrayLike,
	*,
	token_grid_shape: tuple[int, int, int],
	patch_size_xyz: tuple[int, int, int],
) -> np.ndarray:
	"""Convert MAE patch predictions from ``[B, N, A, PV]`` to ``[B, A, X, Y, Z]``."""
	pred_array = _as_numpy(pred_patches, 'pred_patches')
	if pred_array.ndim != 4:
		msg = (
			'pred_patches must be a 4D array with shape [B, N, A, patch_volume]; '
			f'got shape={pred_array.shape!r}'
		)
		raise ValueError(msg)

	tx_size, ty_size, tz_size = _validate_positive_int_triple(
		token_grid_shape,
		'token_grid_shape',
	)
	px_size, py_size, pz_size = _validate_positive_int_triple(
		patch_size_xyz,
		'patch_size_xyz',
	)
	batch_size, num_tokens, num_attributes, patch_volume = pred_array.shape
	expected_num_tokens = tx_size * ty_size * tz_size
	expected_patch_volume = px_size * py_size * pz_size
	if num_tokens != expected_num_tokens or patch_volume != expected_patch_volume:
		msg = (
			'pred_patches shape must match token_grid_shape and patch_size_xyz; '
			f'got shape={pred_array.shape!r}, '
			f'token_grid_shape={token_grid_shape!r}, '
			f'patch_size_xyz={patch_size_xyz!r}, '
			f'expected_num_tokens={expected_num_tokens}, '
			f'expected_patch_volume={expected_patch_volume}'
		)
		raise ValueError(msg)

	return (
		pred_array.reshape(
			batch_size,
			tx_size,
			ty_size,
			tz_size,
			num_attributes,
			px_size,
			py_size,
			pz_size,
		)
		.transpose(0, 4, 1, 5, 2, 6, 3, 7)
		.reshape(
			batch_size,
			num_attributes,
			tx_size * px_size,
			ty_size * py_size,
			tz_size * pz_size,
		)
	)


def upsample_token_mask_to_voxels(
	spatial_mask: ArrayLike,
	*,
	patch_size_xyz: tuple[int, int, int],
) -> np.ndarray:
	"""Convert token masks from ``[B, TX, TY, TZ]`` to voxel masks."""
	mask_array = _as_numpy(spatial_mask, 'spatial_mask')
	if mask_array.ndim != 4:
		msg = (
			'spatial_mask must be a 4D array with shape [B, TX, TY, TZ]; '
			f'got shape={mask_array.shape!r}'
		)
		raise ValueError(msg)

	px_size, py_size, pz_size = _validate_positive_int_triple(
		patch_size_xyz,
		'patch_size_xyz',
	)
	return (
		mask_array.astype(bool, copy=False)
		.repeat(px_size, axis=1)
		.repeat(py_size, axis=2)
		.repeat(pz_size, axis=3)
	)


def build_dense_model_input_for_attribute(
	*,
	x: ArrayLike,
	attribute_ids: ArrayLike,
	attr_id: int,
) -> tuple[np.ndarray | None, np.ndarray]:
	"""Return dense input channel values and per-sample presence for one attribute."""
	x_array = _as_numpy(x, 'x')
	attribute_id_array = _as_numpy(attribute_ids, 'attribute_ids')
	if x_array.ndim != 5:
		msg = (
			'x must be a 5D array with shape [B, C, X, Y, Z]; '
			f'got shape={x_array.shape!r}'
		)
		raise ValueError(msg)
	if attribute_id_array.ndim != 2:
		msg = (
			'attribute_ids must be a 2D array with shape [B, C]; '
			f'got shape={attribute_id_array.shape!r}'
		)
		raise ValueError(msg)
	if attribute_id_array.shape != x_array.shape[:2]:
		msg = (
			'attribute_ids shape must match x batch/channel dimensions; '
			f'got attribute_ids.shape={attribute_id_array.shape!r}, '
			f'x.shape={x_array.shape!r}'
		)
		raise ValueError(msg)

	if attr_id < 0:
		return None, np.zeros(x_array.shape[0], dtype=bool)

	presence = np.any(attribute_id_array == attr_id, axis=1)
	if not np.any(presence):
		return None, presence

	dense = np.zeros(
		(x_array.shape[0], *x_array.shape[2:]),
		dtype=x_array.dtype,
	)
	for batch_index in np.flatnonzero(presence):
		channel_index = int(
			np.flatnonzero(attribute_id_array[batch_index] == attr_id)[0],
		)
		dense[batch_index] = x_array[batch_index, channel_index]

	return dense, presence


def apply_visual_invalid_mask(
	image: np.ndarray,
	valid_mask: np.ndarray | None,
) -> np.ma.MaskedArray | np.ndarray:
	"""Mask invalid voxels for display without modifying numeric image values."""
	if valid_mask is None:
		return image
	return np.ma.masked_where(~np.asarray(valid_mask, dtype=bool), image, copy=True)


def save_mae_debug_visualization_pngs(  # noqa: PLR0913
	*,
	batch: Mapping[str, torch.Tensor | object],
	model_output: Mapping[str, torch.Tensor | object],
	patch_size_xyz: tuple[int, int, int],
	epoch: int,
	global_step: int,
	config: MaeDebugVisualizationConfig,
	max_samples: int = 1,
	prefix: str = 'mae_debug',
) -> list[Path]:
	"""Save XY and XZ MAE debug PNGs and return the created paths."""
	target = _required_array(batch, 'target')
	x = _required_array(batch, 'x')
	attribute_ids = _required_array(batch, 'attribute_ids')
	pred_patches = _required_array(model_output, 'pred_patches')
	_validate_target(target)
	_validate_nonnegative_int(epoch, 'epoch')
	_validate_nonnegative_int(global_step, 'global_step')
	_validate_nonnegative_int(max_samples, 'max_samples')
	_validate_positive_float(config.panel_width, 'panel_width')
	_validate_positive_float(config.panel_height, 'panel_height')
	_validate_positive_int(config.dpi, 'dpi')

	token_grid_shape = _resolve_token_grid_shape(
		model_output.get('token_grid_shape'),
		target.shape[2:],
		patch_size_xyz,
	)
	prediction = unpatchify_mae_predictions(
		pred_patches,
		token_grid_shape=token_grid_shape,
		patch_size_xyz=patch_size_xyz,
	)
	if prediction.shape != target.shape:
		msg = (
			'prediction shape must match target shape after unpatchify; '
			f'got prediction.shape={prediction.shape!r}, target.shape={target.shape!r}'
		)
		raise ValueError(msg)

	spatial_mask = _optional_spatial_mask(batch, model_output)
	spatial_mask_voxel = None
	if spatial_mask is not None:
		spatial_mask_voxel = upsample_token_mask_to_voxels(
			spatial_mask,
			patch_size_xyz=patch_size_xyz,
		)
	local_valid_mask = _optional_array(batch, 'local_valid_mask')
	if local_valid_mask is not None:
		_validate_mask_shape(
			local_valid_mask, target.shape[0], target.shape[2:], 'local_valid_mask'
		)
		local_valid_mask = local_valid_mask.astype(bool, copy=False)
	if spatial_mask_voxel is not None:
		_validate_mask_shape(
			spatial_mask_voxel,
			target.shape[0],
			target.shape[2:],
			'spatial_mask_voxel',
		)

	attribute_names = _resolve_attribute_names(config.attributes)
	_validate_columns(config.columns)
	config.output_dir.mkdir(parents=True, exist_ok=True)
	sample_count = min(max_samples, target.shape[0])
	created: list[Path] = []
	for sample_index in range(sample_count):
		name_prefix = f'{prefix}_' if prefix else ''
		stem = (
			f'{name_prefix}epoch_{epoch:04d}_step_{global_step:06d}_'
			f'sample_{sample_index:02d}'
		)
		xy_path = config.output_dir / f'{stem}_xy.png'
		xz_path = config.output_dir / f'{stem}_xz.png'
		_save_slice_png(
			out_path=xy_path,
			view='xy',
			sample_index=sample_index,
			target=target,
			x=x,
			attribute_ids=attribute_ids,
			prediction=prediction,
			spatial_mask_voxel=spatial_mask_voxel,
			local_valid_mask=local_valid_mask,
			attribute_names=attribute_names,
			config=config,
			epoch=epoch,
			global_step=global_step,
			coords=batch.get('coords'),
		)
		_save_slice_png(
			out_path=xz_path,
			view='xz',
			sample_index=sample_index,
			target=target,
			x=x,
			attribute_ids=attribute_ids,
			prediction=prediction,
			spatial_mask_voxel=spatial_mask_voxel,
			local_valid_mask=local_valid_mask,
			attribute_names=attribute_names,
			config=config,
			epoch=epoch,
			global_step=global_step,
			coords=batch.get('coords'),
		)
		created.extend([xy_path, xz_path])
	return created


def _save_slice_png(  # noqa: PLR0913
	*,
	out_path: Path,
	view: str,
	sample_index: int,
	target: np.ndarray,
	x: np.ndarray,
	attribute_ids: np.ndarray,
	prediction: np.ndarray,
	spatial_mask_voxel: np.ndarray | None,
	local_valid_mask: np.ndarray | None,
	attribute_names: Sequence[str],
	config: MaeDebugVisualizationConfig,
	epoch: int,
	global_step: int,
	coords: object,
) -> None:
	volume_shape = target.shape[2:]
	slice_index = _resolve_slice_index(view, volume_shape, config)
	panels = _build_slice_panels(
		view=view,
		slice_index=slice_index,
		sample_index=sample_index,
		target=target,
		x=x,
		attribute_ids=attribute_ids,
		prediction=prediction,
		spatial_mask_voxel=spatial_mask_voxel,
		local_valid_mask=local_valid_mask,
		attribute_names=attribute_names,
		config=config,
	)
	_plot_panels(
		panels,
		title=_figure_title(
			view=view,
			slice_index=slice_index,
			sample_index=sample_index,
			epoch=epoch,
			global_step=global_step,
			coords=coords,
		),
		out_path=out_path,
		xlabel='x',
		ylabel='y' if view == 'xy' else 'z',
		aspect='equal' if view == 'xy' else 'auto',
		config=config,
	)


def _build_slice_panels(  # noqa: PLR0913
	*,
	view: str,
	slice_index: int,
	sample_index: int,
	target: np.ndarray,
	x: np.ndarray,
	attribute_ids: np.ndarray,
	prediction: np.ndarray,
	spatial_mask_voxel: np.ndarray | None,
	local_valid_mask: np.ndarray | None,
	attribute_names: Sequence[str],
	config: MaeDebugVisualizationConfig,
) -> list[_Panel]:
	valid_slice = _slice_mask(
		local_valid_mask[sample_index] if local_valid_mask is not None else None,
		view=view,
		slice_index=slice_index,
	)
	spatial_slice = _slice_mask(
		spatial_mask_voxel[sample_index] if spatial_mask_voxel is not None else None,
		view=view,
		slice_index=slice_index,
	)
	panels: list[_Panel] = []
	for attr_name in attribute_names:
		attr_id = MVP_ATTRIBUTE_REGISTRY.name_to_id(attr_name)
		input_volume, input_presence = build_dense_model_input_for_attribute(
			x=x,
			attribute_ids=attribute_ids,
			attr_id=attr_id,
		)
		input_present = bool(input_presence[sample_index])
		panels.extend(
			[
				_panel_for_column(
					column=column,
					attr_name=attr_name,
					attr_id=attr_id,
					sample_index=sample_index,
					view=view,
					slice_index=slice_index,
					target=target,
					prediction=prediction,
					input_volume=input_volume,
					input_present=input_present,
					spatial_slice=spatial_slice,
					valid_slice=valid_slice,
				)
				for column in config.columns
			],
		)
	if config.show_valid_mask_panel and valid_slice is not None:
		panels.append(
			_Panel(
				'local_valid_mask',
				valid_slice.astype(np.float32),
				range_name='local_valid_mask',
			),
		)
	if config.show_spatial_mask_panel and spatial_slice is not None:
		panels.append(
			_Panel(
				'spatial_mask_voxel',
				spatial_slice.astype(np.float32),
				range_name='spatial_mask_voxel',
			),
		)
	return panels


def _panel_for_column(  # noqa: PLR0911, PLR0913
	*,
	column: str,
	attr_name: str,
	attr_id: int,
	sample_index: int,
	view: str,
	slice_index: int,
	target: np.ndarray,
	prediction: np.ndarray,
	input_volume: np.ndarray | None,
	input_present: bool,
	spatial_slice: np.ndarray | None,
	valid_slice: np.ndarray | None,
) -> _Panel:
	if column == 'input':
		if input_volume is None or not input_present:
			return _blank_panel(
				f'{attr_name}\ninput (dropped)', target.shape[2:], view=view
			)
		return _Panel(
			f'{attr_name}\ninput',
			_slice_image(
				input_volume[sample_index], view=view, slice_index=slice_index
			),
			valid_mask=valid_slice,
			range_name=attr_name,
		)
	if column == 'masked_input':
		if input_volume is None or not input_present:
			return _blank_panel(
				f'{attr_name}\nmasked_input', target.shape[2:], view=view
			)
		mask = valid_slice
		if spatial_slice is not None:
			mask = ~spatial_slice if mask is None else mask & ~spatial_slice
		return _Panel(
			f'{attr_name}\nmasked_input',
			_slice_image(
				input_volume[sample_index], view=view, slice_index=slice_index
			),
			valid_mask=mask,
			range_name=attr_name,
		)
	if column == 'target':
		return _Panel(
			f'{attr_name}\ntarget',
			_slice_image(
				target[sample_index, attr_id], view=view, slice_index=slice_index
			),
			valid_mask=valid_slice,
			range_name=attr_name,
		)
	if column == 'prediction':
		return _Panel(
			f'{attr_name}\nprediction',
			_slice_image(
				prediction[sample_index, attr_id],
				view=view,
				slice_index=slice_index,
			),
			valid_mask=valid_slice,
			range_name=attr_name,
		)
	if column == 'abs_error':
		return _Panel(
			f'{attr_name}\nabs_error',
			_slice_image(
				np.abs(
					prediction[sample_index, attr_id] - target[sample_index, attr_id]
				),
				view=view,
				slice_index=slice_index,
			),
			valid_mask=valid_slice,
			range_name='abs_error',
		)
	msg = f'unknown MAE debug column: {column!r}'
	raise ValueError(msg)


def _plot_panels(  # noqa: PLR0913
	panels: Sequence[_Panel],
	title: str,
	out_path: Path,
	xlabel: str,
	ylabel: str,
	aspect: str,
	config: MaeDebugVisualizationConfig,
) -> None:
	n_panels = len(panels)
	nrows, ncols = _resolve_grid(n_panels, len(config.columns), config.grid_mode)
	fig, axes = plt.subplots(
		nrows,
		ncols,
		figsize=(config.panel_width * ncols, config.panel_height * nrows),
		squeeze=False,
	)
	flat_axes = list(axes.ravel())
	for ax, panel in zip(flat_axes, panels, strict=False):
		image = _display_image(panel, config)
		vmin, vmax = _display_limits(
			panel.range_name or panel.title,
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
		ax.set_title(panel.title, fontsize=8)
		ax.set_xlabel(xlabel, fontsize=8)
		ax.set_ylabel(ylabel, fontsize=8)
		ax.tick_params(labelsize=7)
		fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
	for ax in flat_axes[n_panels:]:
		ax.axis('off')
	fig.suptitle(title, fontsize=10)
	fig.tight_layout()
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_path, dpi=config.dpi, bbox_inches='tight')
	plt.close(fig)


def _resolve_grid(
	n_panels: int,
	num_columns: int,
	grid_mode: str,
) -> tuple[int, int]:
	mode = str(grid_mode).strip().lower()
	if mode == 'auto':
		if n_panels <= 0:
			msg = f'n_panels must be positive; got {n_panels!r}'
			raise ValueError(msg)
		ncols = max(1, num_columns)
		nrows = math.ceil(n_panels / ncols)
		return nrows, ncols
	try:
		rows_str, cols_str = mode.split('x', maxsplit=1)
		nrows = int(rows_str)
		ncols = int(cols_str)
	except Exception as exc:
		msg = f"invalid grid_mode={grid_mode!r}; expected 'auto' or like '8x5'"
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
	image: np.ndarray | np.ma.MaskedArray,
	clip_percentiles: tuple[float, float],
	*,
	use_known_ranges: bool,
) -> tuple[float | None, float | None]:
	if use_known_ranges and name in _KNOWN_RANGES:
		return _KNOWN_RANGES[name]
	finite = np.ma.masked_invalid(image).compressed()
	if finite.size == 0:
		return None, None
	if name == 'abs_error':
		return 0.0, float(np.percentile(finite, 99.0))
	vmin, vmax = np.percentile(finite, clip_percentiles)
	if np.isclose(vmin, vmax):
		center = float(np.mean(finite))
		half = float(np.std(finite)) or 1.0
		return center - half, center + half
	return float(vmin), float(vmax)


def _display_image(
	panel: _Panel,
	config: MaeDebugVisualizationConfig,
) -> np.ndarray | np.ma.MaskedArray:
	if not config.mask_invalid_values:
		return panel.image
	if panel.valid_mask is None:
		return panel.image
	return apply_visual_invalid_mask(panel.image, panel.valid_mask)


def _cmap_for_panel(
	panel: _Panel,
	config: MaeDebugVisualizationConfig,
) -> str | plt.Colormap:
	name = panel.range_name or panel.title
	if name in {'local_valid_mask', 'spatial_mask_voxel'}:
		return 'gray'
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


def _slice_image(
	volume: np.ndarray,
	*,
	view: str,
	slice_index: int,
) -> np.ndarray:
	if view == 'xy':
		return np.asarray(volume[:, :, slice_index]).T
	if view == 'xz':
		return np.asarray(volume[:, slice_index, :]).T
	msg = f'unknown view: {view!r}'
	raise ValueError(msg)


def _slice_mask(
	mask: np.ndarray | None,
	*,
	view: str,
	slice_index: int,
) -> np.ndarray | None:
	if mask is None:
		return None
	return _slice_image(mask, view=view, slice_index=slice_index).astype(
		bool, copy=False
	)


def _blank_panel(title: str, volume_shape: Sequence[int], *, view: str) -> _Panel:
	x_size, y_size, z_size = volume_shape
	height = y_size if view == 'xy' else z_size
	return _Panel(
		title,
		np.zeros((height, x_size), dtype=np.float32),
		valid_mask=np.zeros((height, x_size), dtype=bool),
	)


def _resolve_slice_index(
	view: str,
	volume_shape: Sequence[int],
	config: MaeDebugVisualizationConfig,
) -> int:
	_x_size, y_size, z_size = volume_shape
	if view == 'xy':
		index = (
			z_size // 2 if config.xy_slice_index is None else int(config.xy_slice_index)
		)
		limit = z_size
		name = 'xy_slice_index'
	elif view == 'xz':
		index = (
			y_size // 2
			if config.xz_slice_y_index is None
			else int(config.xz_slice_y_index)
		)
		limit = y_size
		name = 'xz_slice_y_index'
	else:
		msg = f'unknown view: {view!r}'
		raise ValueError(msg)
	if not 0 <= index < limit:
		msg = f'{name} out of range: {index}, valid=[0, {limit - 1}]'
		raise ValueError(msg)
	return index


def _figure_title(  # noqa: PLR0913
	*,
	view: str,
	slice_index: int,
	sample_index: int,
	epoch: int,
	global_step: int,
	coords: object,
) -> str:
	title = (
		f'MAE debug {view.upper()} sample={sample_index} '
		f'epoch={epoch:04d} step={global_step:06d} '
		f'{"z" if view == "xy" else "y"}={slice_index}'
	)
	coord = _sample_coords(coords, sample_index)
	if coord is None:
		return title
	pieces: list[str] = []
	if (survey_id := coord.get('survey_id')) is not None:
		pieces.append(f'survey={survey_id}')
	if (local_start := coord.get('local_start_xyz')) is not None:
		pieces.append(f'local_start_xyz={local_start}')
	if not pieces:
		return title
	return f'{title} | {", ".join(pieces)}'


def _sample_coords(coords: object, sample_index: int) -> Mapping[str, object] | None:
	if isinstance(coords, Mapping):
		return coords
	if (
		isinstance(coords, Sequence)
		and not isinstance(coords, str)
		and sample_index < len(coords)
		and isinstance(coords[sample_index], Mapping)
	):
		return coords[sample_index]
	return None


def _resolve_attribute_names(names: Sequence[str]) -> tuple[str, ...]:
	resolved = tuple(names)
	if not resolved:
		msg = 'config.attributes must contain at least one attribute name'
		raise ValueError(msg)
	for name in resolved:
		MVP_ATTRIBUTE_REGISTRY.name_to_id(name)
	return resolved


def _validate_columns(columns: Sequence[str]) -> None:
	if not columns:
		msg = 'config.columns must contain at least one column'
		raise ValueError(msg)
	unknown = sorted(set(columns) - _DEFAULT_COLUMNS)
	if unknown:
		msg = f'unknown MAE debug columns: {unknown!r}'
		raise ValueError(msg)


def _resolve_token_grid_shape(
	value: object,
	volume_shape: Sequence[int],
	patch_size_xyz: tuple[int, int, int],
) -> tuple[int, int, int]:
	px_size, py_size, pz_size = _validate_positive_int_triple(
		patch_size_xyz,
		'patch_size_xyz',
	)
	if value is not None:
		if isinstance(value, torch.Tensor):
			value = value.detach().cpu().tolist()
		if isinstance(value, np.ndarray):
			value = value.tolist()
		if not isinstance(value, Sequence):
			msg = f'token_grid_shape must be a sequence; got {type(value).__name__}'
			raise TypeError(msg)
		return _validate_positive_int_triple(
			tuple(int(item) for item in value), 'token_grid_shape'
		)
	x_size, y_size, z_size = tuple(int(item) for item in volume_shape)
	if x_size % px_size or y_size % py_size or z_size % pz_size:
		msg = (
			'target volume shape must be divisible by patch_size_xyz when '
			'token_grid_shape is absent; '
			f'got volume_shape={tuple(volume_shape)!r}, '
			f'patch_size_xyz={patch_size_xyz!r}'
		)
		raise ValueError(msg)
	return x_size // px_size, y_size // py_size, z_size // pz_size


def _required_array(
	mapping: Mapping[str, object],
	key: str,
) -> np.ndarray:
	if key not in mapping:
		msg = f'missing required key: {key!r}'
		raise KeyError(msg)
	return _as_numpy(mapping[key], key)


def _optional_array(
	mapping: Mapping[str, object],
	key: str,
) -> np.ndarray | None:
	value = mapping.get(key)
	if value is None:
		return None
	return _as_numpy(value, key)


def _optional_spatial_mask(
	batch: Mapping[str, object],
	model_output: Mapping[str, object],
) -> np.ndarray | None:
	value = batch.get('spatial_mask')
	if value is None:
		value = model_output.get('spatial_mask')
	if value is None:
		return None
	return _as_numpy(value, 'spatial_mask')


def _validate_target(target: np.ndarray) -> None:
	if target.ndim != 5:
		msg = (
			'target must be a 5D array with shape [B, A, X, Y, Z]; '
			f'got shape={target.shape!r}'
		)
		raise ValueError(msg)


def _validate_mask_shape(
	mask: np.ndarray,
	batch_size: int,
	volume_shape: Sequence[int],
	name: str,
) -> None:
	expected = (batch_size, *tuple(volume_shape))
	if mask.shape != expected:
		msg = f'{name} shape must be {expected!r}; got {mask.shape!r}'
		raise ValueError(msg)


def _validate_nonnegative_int(value: int, name: str) -> int:
	if not isinstance(value, int) or isinstance(value, bool) or value < 0:
		msg = f'{name} must be a non-negative integer; got {value!r}'
		raise ValueError(msg)
	return value


def _validate_positive_int(value: int, name: str) -> int:
	if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
		msg = f'{name} must be a positive integer; got {value!r}'
		raise ValueError(msg)
	return value


def _validate_positive_float(value: float, name: str) -> float:
	if not isinstance(value, (float, int)) or isinstance(value, bool) or value <= 0:
		msg = f'{name} must be positive; got {value!r}'
		raise ValueError(msg)
	return float(value)


def _as_numpy(value: ArrayLike, name: str) -> np.ndarray:
	if isinstance(value, np.ndarray):
		return value

	if isinstance(value, torch.Tensor):
		return value.detach().cpu().numpy()

	msg = f'{name} must be a torch.Tensor or np.ndarray; got {type(value).__name__}'
	raise TypeError(msg)


def _validate_positive_int_triple(
	value: tuple[int, int, int],
	name: str,
) -> tuple[int, int, int]:
	if (
		not isinstance(value, tuple)
		or len(value) != 3
		or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
		or any(item <= 0 for item in value)
	):
		msg = f'{name} must be a positive integer triple; got {value!r}'
		raise ValueError(msg)
	return value


__all__ = [
	'MaeDebugVisualizationConfig',
	'apply_visual_invalid_mask',
	'build_dense_model_input_for_attribute',
	'save_mae_debug_visualization_pngs',
	'unpatchify_mae_predictions',
	'upsample_token_mask_to_voxels',
]
