from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from seis_attr_ssl.data import (
	AttributeGenerationConfig,
	AttributeGenerationResult,
	ZeroAmplitudeMaskConfig,
)
from seis_attr_ssl.visualization.attribute_on_the_fly_compare import (
	OnTheFlyAttributeCompareConfig,
	_build_panels,
	_display_image,
	_resolve_grid,
	save_on_the_fly_attribute_comparison_pngs,
)


def test_visualization_passes_attribute_generation_config_to_on_the_fly_generator(
	tmp_path: Path,
	monkeypatch,
) -> None:
	volume_path = tmp_path / 'volume.npy'
	np.save(volume_path, np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6))
	attribute_generation_config = AttributeGenerationConfig(spectral_local_window_z=9)
	calls: list[AttributeGenerationConfig | None] = []

	def fake_generate_mvp_attributes_for_payload(
		amp_norm_compute: np.ndarray,
		payload_slices_xyz: tuple[slice, slice, slice],
		*,
		valid_mask=None,
		config: AttributeGenerationConfig | None = None,
	) -> AttributeGenerationResult:
		del valid_mask
		calls.append(config)
		payload_shape = tuple(
			len(range(*payload_slice.indices(size)))
			for payload_slice, size in zip(
				payload_slices_xyz,
				amp_norm_compute.shape,
				strict=True,
			)
		)
		return AttributeGenerationResult(
			attributes=np.zeros((10, *payload_shape), dtype=np.float32),
			attribute_valid=np.ones(10, dtype=bool),
			voxel_valid_mask=np.ones(payload_shape, dtype=bool),
		)

	monkeypatch.setattr(
		'seis_attr_ssl.visualization.attribute_on_the_fly_compare.'
		'generate_mvp_attributes_for_payload',
		fake_generate_mvp_attributes_for_payload,
	)

	xy_png, xz_png = save_on_the_fly_attribute_comparison_pngs(
		volume_path,
		tmp_path / 'out',
		assume_normalized=True,
		attribute_names=('spectral_low_ratio',),
		config=OnTheFlyAttributeCompareConfig(
			xy_slice_index=3,
			xz_slice_y_index=2,
			xy_z_window=5,
			xz_y_window=3,
			show_raw_amplitude=False,
		),
		attribute_generation_config=attribute_generation_config,
	)

	assert calls == [attribute_generation_config, attribute_generation_config]
	assert xy_png.exists()
	assert xz_png.exists()


def test_visualization_writes_zero_mask_pngs_for_xy_and_xz(tmp_path: Path) -> None:
	volume = np.ones((7, 6, 9), dtype=np.float32)
	volume += np.linspace(0.0, 0.5, volume.shape[2], dtype=np.float32)
	volume[:, :, 4] = 0.0
	volume[3, 2, :] = 0.0
	volume_path = tmp_path / 'volume.npy'
	np.save(volume_path, volume)

	xy_png, xz_png = save_on_the_fly_attribute_comparison_pngs(
		volume_path,
		tmp_path / 'out',
		assume_normalized=True,
		attribute_names=('amplitude_norm',),
		config=OnTheFlyAttributeCompareConfig(
			xy_slice_index=4,
			xz_slice_y_index=2,
			xy_z_window=9,
			xz_y_window=5,
			show_raw_amplitude=True,
			show_valid_mask=True,
			mask_invalid_values=True,
		),
		attribute_generation_config=AttributeGenerationConfig(
			phase_reflect_pad_z=2,
			instantaneous_frequency_smooth_z=3,
			spectral_local_window_z=3,
			zero_mask=ZeroAmplitudeMaskConfig(
				z_sample_influence_radius=1,
				xy_trace_influence_radius=1,
			),
		),
	)

	assert xy_png.exists()
	assert xz_png.exists()


def test_visualization_valid_mask_panel_uses_displayed_local_slice(
	tmp_path: Path,
	monkeypatch,
) -> None:
	volume_path = tmp_path / 'volume.npy'
	np.save(volume_path, np.ones((3, 5, 7), dtype=np.float32))
	captured_masks: list[np.ndarray] = []

	def fake_generate_mvp_attributes_for_payload(
		amp_norm_compute: np.ndarray,
		payload_slices_xyz: tuple[slice, slice, slice],
		*,
		valid_mask=None,
		config: AttributeGenerationConfig | None = None,
	) -> AttributeGenerationResult:
		del valid_mask, config
		assert payload_slices_xyz == tuple(
			slice(0, size) for size in amp_norm_compute.shape
		)
		mask = np.ones(amp_norm_compute.shape, dtype=bool)
		if amp_norm_compute.shape == (3, 5, 3):
			mask[:, :, 0] = False
			mask[1, 2, 1] = False
		else:
			assert amp_norm_compute.shape == (3, 3, 7)
			mask[:, 0, :] = False
			mask[2, 1, 4] = False
		return AttributeGenerationResult(
			attributes=np.zeros((10, *amp_norm_compute.shape), dtype=np.float32),
			attribute_valid=np.ones(10, dtype=bool),
			voxel_valid_mask=mask,
		)

	def fake_plot_panels(panels, *args, **kwargs) -> None:
		del args, kwargs
		valid_panel = next(panel for panel in panels if panel.name == 'valid_mask')
		captured_masks.append(valid_panel.image)

	monkeypatch.setattr(
		'seis_attr_ssl.visualization.attribute_on_the_fly_compare.'
		'generate_mvp_attributes_for_payload',
		fake_generate_mvp_attributes_for_payload,
	)
	monkeypatch.setattr(
		'seis_attr_ssl.visualization.attribute_on_the_fly_compare._plot_panels',
		fake_plot_panels,
	)

	save_on_the_fly_attribute_comparison_pngs(
		volume_path,
		tmp_path / 'out',
		assume_normalized=True,
		attribute_names=('amplitude_norm',),
		config=OnTheFlyAttributeCompareConfig(
			xy_slice_index=5,
			xz_slice_y_index=3,
			xy_z_window=3,
			xz_y_window=3,
			show_valid_mask=True,
		),
	)

	expected_xy = np.ones((3, 5), dtype=bool)
	expected_xy[1, 2] = False
	expected_xz = np.ones((3, 7), dtype=bool)
	expected_xz[2, 4] = False
	np.testing.assert_array_equal(captured_masks[0], expected_xy.T.astype(np.float32))
	np.testing.assert_array_equal(captured_masks[1], expected_xz.T.astype(np.float32))


def test_build_panels_adds_valid_mask_and_masks_invalid_values() -> None:
	valid_mask = np.array([[True, False], [True, True]])
	config = OnTheFlyAttributeCompareConfig(
		show_raw_amplitude=False,
		show_valid_mask=True,
		mask_invalid_values=True,
	)

	panels = _build_panels(
		raw_amplitude=np.ones((2, 2), dtype=np.float32),
		attribute_images=[
			('amplitude_norm', np.arange(4, dtype=np.float32).reshape(2, 2)),
		],
		valid_mask_2d=valid_mask,
		config=config,
	)

	assert [panel.name for panel in panels] == ['amplitude_norm', 'valid_mask']
	np.testing.assert_array_equal(panels[1].image, valid_mask.astype(np.float32))
	masked = _display_image(panels[0], config)
	assert np.ma.isMaskedArray(masked)
	np.testing.assert_array_equal(np.ma.getmaskarray(masked), ~valid_mask)

	disabled_config = OnTheFlyAttributeCompareConfig(
		show_raw_amplitude=False,
		show_valid_mask=False,
		mask_invalid_values=False,
	)
	disabled_panels = _build_panels(
		raw_amplitude=np.ones((2, 2), dtype=np.float32),
		attribute_images=[
			('amplitude_norm', np.arange(4, dtype=np.float32).reshape(2, 2)),
		],
		valid_mask_2d=valid_mask,
		config=disabled_config,
	)
	assert [panel.name for panel in disabled_panels] == ['amplitude_norm']
	assert (
		_display_image(disabled_panels[0], disabled_config)
		is disabled_panels[0].image
	)


def test_auto_grid_prefers_attribute_comparison_shapes() -> None:
	assert _resolve_grid(10, 'auto') == (2, 5)
	assert _resolve_grid(11, 'auto') == (3, 4)
	assert _resolve_grid(12, 'auto') == (3, 4)


def test_default_visualization_config_pins_revised_attribute_qc_defaults() -> None:
	config_path = Path('proc/configs/visualize_attribute_on_the_fly_compare.yaml')
	cfg = yaml.safe_load(config_path.read_text(encoding='utf-8'))

	assert cfg['attribute_generation'] == {
		'phase_reflect_pad_z': 64,
		'phase_taper_fraction': 0.05,
		'instantaneous_frequency_smooth_z': 5,
		'instantaneous_frequency_envelope_quantile': 0.05,
		'instantaneous_frequency_clip_percentile': 99.5,
		'spectral_local_window_z': 65,
		'spectral_remove_dc': True,
		'zero_mask': {
			'enabled': True,
			'zero_atol': 0.0,
			'z_sample_influence_radius': 64,
			'xy_trace_influence_radius': 1,
			'z_trace_influence_radius': 0,
		},
	}
	assert cfg['visualization']['grid_mode'] == 'auto'
	assert cfg['visualization']['show_raw_amplitude'] is True
	assert cfg['visualization']['show_valid_mask'] is True
	assert cfg['visualization']['mask_invalid_values'] is True
	assert cfg['visualization']['invalid_color'] == 'lightgray'
	assert cfg['visualization']['use_known_ranges'] is True
