from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from proc.seis_ssl_cluster.visualize_clusters import run_cluster_visualization
from seis_ssl_cluster.visualization.clusters import (
	ClusterSliceRequest,
	save_cluster_slice_pngs,
	stable_cluster_colors,
)

if TYPE_CHECKING:
	from pathlib import Path


pytest.importorskip('matplotlib')


def test_xy_and_xz_cluster_pngs_are_created(tmp_path: Path) -> None:
	labels = np.arange(3 * 4 * 5, dtype=np.int32).reshape(3, 4, 5) % 3
	labels[0, :, :] = -1
	amplitude = np.linspace(-1.0, 1.0, labels.size, dtype=np.float32).reshape(
		labels.shape,
	)

	created = save_cluster_slice_pngs(
		labels,
		survey_id='survey',
		k=3,
		mode='voxel',
		output_dir=tmp_path,
		slices=ClusterSliceRequest(xy_slices=(2,), xz_slices=(1,)),
		amplitude=amplitude,
		amplitude_alpha=0.25,
	)

	assert [path.name for path in created] == [
		'survey_k3_xy_z2.png',
		'survey_k3_xz_y1.png',
	]
	assert all(path.is_file() and path.stat().st_size > 0 for path in created)


def test_cluster_colormap_is_stable_for_same_k() -> None:
	first = stable_cluster_colors(4)
	second = stable_cluster_colors(4)

	np.testing.assert_array_equal(first.colors, second.colors)


def test_proc_visualization_writes_token_and_voxel_modes_separately(
	tmp_path: Path,
) -> None:
	input_dir = tmp_path / 'cluster_run'
	output_dir = tmp_path / 'figures'
	labels_dir = input_dir / 'labels' / 'k2'
	labels_dir.mkdir(parents=True)
	token_labels = np.array(
		[
			[[0, 1], [1, -1]],
			[[1, 0], [-1, 0]],
		],
		dtype=np.int32,
	)
	np.save(labels_dir / 'survey.cluster_labels_token.npy', token_labels)

	result = run_cluster_visualization(
		{
			'clustering': {'input_dir': str(input_dir)},
			'visualization': {
				'output_dir': str(output_dir),
				'modes': ['token', 'voxel'],
				'reconstruct_voxel': True,
				'xy_slices': [1],
				'xz_slices': [0],
				'summaries': {'enabled': False},
				'amplitude_underlay': {'enabled': False},
			},
		},
	)

	assert result['png_count'] == 4
	assert (output_dir / 'token' / 'survey_k2_xy_z1.png').is_file()
	assert (output_dir / 'token' / 'survey_k2_xz_y0.png').is_file()
	assert (output_dir / 'voxel' / 'survey_k2_xy_z1.png').is_file()
	assert (output_dir / 'voxel' / 'survey_k2_xz_y0.png').is_file()
