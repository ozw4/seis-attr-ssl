from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from seis_attr_ssl.data import AttributeGenerationConfig, AttributeGenerationResult
from seis_attr_ssl.visualization.attribute_on_the_fly_compare import (
	OnTheFlyAttributeCompareConfig,
	save_on_the_fly_attribute_comparison_pngs,
)

if TYPE_CHECKING:
	from pathlib import Path


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
