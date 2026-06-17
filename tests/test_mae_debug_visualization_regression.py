from __future__ import annotations

import numpy as np
import pytest
import torch

from seis_attr_ssl.attributes import MVP_ATTRIBUTE_REGISTRY
from seis_attr_ssl.models.mae.patching import patchify_3d
from seis_attr_ssl.visualization import mae_debug
from seis_attr_ssl.visualization.mae_debug import (
	MaeDebugVisualizationConfig,
	save_mae_debug_visualization_pngs,
)

PATCH_SIZE_XYZ = (4, 4, 4)
TOKEN_GRID_SHAPE = (2, 2, 2)
LOCAL_SHAPE_XYZ = (8, 8, 8)
ATTRIBUTE_NAMES = MVP_ATTRIBUTE_REGISTRY.names[:4]


def test_dropped_visualization_attribute_keeps_target_prediction_and_error(
	tmp_path,
	monkeypatch,
) -> None:
	calls = _capture_plot_calls(monkeypatch)
	target = _target()
	selected_attr_name = ATTRIBUTE_NAMES[2]

	save_mae_debug_visualization_pngs(
		batch=_batch(
			target=target,
			input_attribute_ids=(0, 1),
		),
		model_output=_model_output(target + 0.5),
		patch_size_xyz=PATCH_SIZE_XYZ,
		epoch=0,
		global_step=0,
		config=_config(tmp_path, attributes=(selected_attr_name,)),
	)

	input_panel = _panel(calls, 'xy', f'{selected_attr_name}\ninput (dropped)')
	masked_input_panel = _panel(
		calls,
		'xy',
		f'{selected_attr_name}\nmasked_input',
	)
	target_panel = _panel(calls, 'xy', f'{selected_attr_name}\ntarget')
	prediction_panel = _panel(calls, 'xy', f'{selected_attr_name}\nprediction')
	error_panel = _panel(calls, 'xy', f'{selected_attr_name}\nabs_error')

	np.testing.assert_array_equal(input_panel.image, np.zeros((8, 8)))
	np.testing.assert_array_equal(masked_input_panel.image, np.zeros((8, 8)))
	assert not input_panel.valid_mask.any()
	assert not masked_input_panel.valid_mask.any()
	assert target_panel.image.any()
	assert prediction_panel.image.any()
	np.testing.assert_allclose(error_panel.image, np.full((8, 8), 0.5))


def test_spatial_mask_only_masks_masked_input_panel(tmp_path, monkeypatch) -> None:
	calls = _capture_plot_calls(monkeypatch)
	target = _target()
	spatial_mask = torch.zeros((1, *TOKEN_GRID_SHAPE), dtype=torch.bool)
	spatial_mask[0, 0, 0, 1] = True
	attr_name = ATTRIBUTE_NAMES[0]

	save_mae_debug_visualization_pngs(
		batch=_batch(
			target=target,
			input_attribute_ids=(0, 1, 2, 3),
			spatial_mask=spatial_mask,
		),
		model_output=_model_output(target + 0.5),
		patch_size_xyz=PATCH_SIZE_XYZ,
		epoch=0,
		global_step=1,
		config=_config(tmp_path, attributes=(attr_name,), xy_slice_index=4),
	)

	masked_input_panel = _panel(calls, 'xy', f'{attr_name}\nmasked_input')
	target_panel = _panel(calls, 'xy', f'{attr_name}\ntarget')
	prediction_panel = _panel(calls, 'xy', f'{attr_name}\nprediction')
	expected_mask = np.ones((8, 8), dtype=bool)
	expected_mask[:4, :4] = False

	np.testing.assert_array_equal(masked_input_panel.valid_mask, expected_mask)
	np.testing.assert_array_equal(target_panel.valid_mask, np.ones((8, 8), dtype=bool))
	np.testing.assert_array_equal(
		prediction_panel.valid_mask,
		np.ones((8, 8), dtype=bool),
	)


def test_local_valid_mask_masks_deep_region_in_rendered_panels(
	tmp_path,
	monkeypatch,
) -> None:
	calls = _capture_plot_calls(monkeypatch)
	target = _target()
	local_valid_mask = torch.ones((1, *LOCAL_SHAPE_XYZ), dtype=torch.bool)
	local_valid_mask[:, :, :, 4:] = False
	attr_name = ATTRIBUTE_NAMES[1]

	save_mae_debug_visualization_pngs(
		batch=_batch(
			target=target,
			input_attribute_ids=(0, 1, 2, 3),
			local_valid_mask=local_valid_mask,
		),
		model_output=_model_output(target + 0.5),
		patch_size_xyz=PATCH_SIZE_XYZ,
		epoch=0,
		global_step=2,
		config=_config(tmp_path, attributes=(attr_name,), xz_slice_y_index=4),
	)

	expected_valid_slice = np.ones((8, 8), dtype=bool)
	expected_valid_slice[4:, :] = False
	for title in (
		f'{attr_name}\ntarget',
		f'{attr_name}\nprediction',
		f'{attr_name}\nabs_error',
	):
		display_image = _display_image(calls, 'xz', title)
		assert isinstance(display_image, np.ma.MaskedArray)
		np.testing.assert_array_equal(display_image.mask, ~expected_valid_slice)

	valid_mask_panel = _panel(calls, 'xz', 'local_valid_mask')
	np.testing.assert_array_equal(
		valid_mask_panel.image,
		expected_valid_slice.astype(np.float32),
	)


def test_context_disabled_batch_still_renders(tmp_path, monkeypatch) -> None:
	calls = _capture_plot_calls(monkeypatch)
	target = _target()

	paths = save_mae_debug_visualization_pngs(
		batch=_batch(
			target=target,
			input_attribute_ids=(0, 1, 2, 3),
			context=None,
			context_valid_mask=None,
		),
		model_output=_model_output(target + 0.5),
		patch_size_xyz=PATCH_SIZE_XYZ,
		epoch=0,
		global_step=3,
		config=_config(tmp_path, attributes=(ATTRIBUTE_NAMES[0],)),
	)

	assert len(paths) == 2
	assert len(calls) == 2


def test_prediction_patch_shape_mismatch_raises_clear_value_error(tmp_path) -> None:
	target = _target()
	bad_pred_patches = torch.zeros((1, 7, 4, 64), dtype=torch.float32)

	with pytest.raises(ValueError, match='expected_num_tokens=8'):
		save_mae_debug_visualization_pngs(
			batch=_batch(
				target=target,
				input_attribute_ids=(0, 1, 2, 3),
			),
			model_output={
				'pred_patches': bad_pred_patches,
				'token_grid_shape': TOKEN_GRID_SHAPE,
			},
			patch_size_xyz=PATCH_SIZE_XYZ,
			epoch=0,
			global_step=4,
			config=_config(tmp_path, attributes=(ATTRIBUTE_NAMES[0],)),
		)


def _target() -> torch.Tensor:
	values = torch.arange(
		4 * np.prod(LOCAL_SHAPE_XYZ),
		dtype=torch.float32,
	).reshape(1, 4, *LOCAL_SHAPE_XYZ)
	return values / 100.0


def _batch(  # noqa: PLR0913
	*,
	target: torch.Tensor,
	input_attribute_ids: tuple[int, ...],
	spatial_mask: torch.Tensor | None = None,
	local_valid_mask: torch.Tensor | None = None,
	context: torch.Tensor | None = None,
	context_valid_mask: torch.Tensor | None = None,
) -> dict[str, object]:
	ids = torch.tensor([input_attribute_ids], dtype=torch.long)
	return {
		'x': target[:, list(input_attribute_ids)].clone(),
		'target': target,
		'attribute_ids': ids,
		'spatial_mask': spatial_mask,
		'local_valid_mask': (
			torch.ones((1, *LOCAL_SHAPE_XYZ), dtype=torch.bool)
			if local_valid_mask is None
			else local_valid_mask
		),
		'context': context,
		'context_valid_mask': context_valid_mask,
	}


def _model_output(prediction: torch.Tensor) -> dict[str, object]:
	return {
		'pred_patches': patchify_3d(prediction, PATCH_SIZE_XYZ),
		'token_grid_shape': TOKEN_GRID_SHAPE,
	}


def _config(
	output_dir,
	*,
	attributes: tuple[str, ...],
	xy_slice_index: int | None = None,
	xz_slice_y_index: int | None = None,
) -> MaeDebugVisualizationConfig:
	return MaeDebugVisualizationConfig(
		output_dir=output_dir,
		attributes=attributes,
		xy_slice_index=xy_slice_index,
		xz_slice_y_index=xz_slice_y_index,
		dpi=40,
		panel_width=1.4,
		panel_height=1.2,
	)


def _capture_plot_calls(monkeypatch):
	calls = []

	def fake_plot(panels, **kwargs):
		calls.append(
			{
				'title': kwargs['title'],
				'panels': list(panels),
				'display_images': [
					mae_debug._display_image(panel, kwargs['config'])  # noqa: SLF001
					for panel in panels
				],
			},
		)

	monkeypatch.setattr(mae_debug, '_plot_panels', fake_plot)
	return calls


def _call(calls, view: str):
	return next(call for call in calls if f'MAE debug {view.upper()}' in call['title'])


def _panel(calls, view: str, title: str):
	call = _call(calls, view)
	return next(panel for panel in call['panels'] if panel.title == title)


def _display_image(calls, view: str, title: str):
	call = _call(calls, view)
	for panel, display_image in zip(
		call['panels'],
		call['display_images'],
		strict=True,
	):
		if panel.title == title:
			return display_image
	msg = f'missing panel: {title!r}'
	raise AssertionError(msg)
