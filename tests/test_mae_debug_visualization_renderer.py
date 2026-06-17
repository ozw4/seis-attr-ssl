from __future__ import annotations

import torch

from seis_attr_ssl.attributes import MVP_ATTRIBUTE_REGISTRY
from seis_attr_ssl.models.mae.patching import patchify_3d
from seis_attr_ssl.visualization.mae_debug import (
	MaeDebugVisualizationConfig,
	save_mae_debug_visualization_pngs,
)


def test_save_mae_debug_visualization_pngs_writes_xy_and_xz(
	tmp_path,
) -> None:
	patch_size_xyz = (2, 2, 2)
	num_attributes = len(MVP_ATTRIBUTE_REGISTRY.specs)
	target = torch.arange(
		num_attributes * 4 * 4 * 4,
		dtype=torch.float32,
	).reshape(1, num_attributes, 4, 4, 4)
	x = target[:, :1].clone()
	attribute_ids = torch.tensor(
		[[MVP_ATTRIBUTE_REGISTRY.name_to_id('amplitude_norm')]],
		dtype=torch.long,
	)
	spatial_mask = torch.zeros((1, 2, 2, 2), dtype=torch.bool)
	spatial_mask[0, 0, 0, 0] = True
	local_valid_mask = torch.ones((1, 4, 4, 4), dtype=torch.bool)
	local_valid_mask[:, 0, :, :] = False
	pred_patches = patchify_3d(target + 0.5, patch_size_xyz)
	config = MaeDebugVisualizationConfig(
		output_dir=tmp_path,
		attributes=('amplitude_norm', 'phase_sin'),
		dpi=40,
		panel_width=1.8,
		panel_height=1.5,
	)

	paths = save_mae_debug_visualization_pngs(
		batch={
			'x': x,
			'target': target,
			'attribute_ids': attribute_ids,
			'spatial_mask': spatial_mask,
			'local_valid_mask': local_valid_mask,
			'coords': [
				{
					'survey_id': 'survey-a',
					'local_start_xyz': (1, 2, 3),
				},
			],
		},
		model_output={
			'pred_patches': pred_patches,
			'token_grid_shape': (2, 2, 2),
		},
		patch_size_xyz=patch_size_xyz,
		epoch=1,
		global_step=7,
		config=config,
	)

	assert paths == [
		tmp_path / 'mae_debug_epoch_0001_step_000007_sample_00_xy.png',
		tmp_path / 'mae_debug_epoch_0001_step_000007_sample_00_xz.png',
	]
	for path in paths:
		assert path.exists()
		assert path.stat().st_size > 0
