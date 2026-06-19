from __future__ import annotations

import json
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


plt = pytest.importorskip('matplotlib.pyplot')


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


def test_amplitude_underlay_changes_visible_cluster_pixels(tmp_path: Path) -> None:
	labels = np.zeros((4, 4, 1), dtype=np.int32)
	flat = np.linspace(-1.0, 1.0, labels.size, dtype=np.float32)
	first_amplitude = np.zeros_like(labels, dtype=np.float32)
	second_amplitude = flat.reshape(labels.shape)

	first = save_cluster_slice_pngs(
		labels,
		survey_id='survey',
		k=1,
		mode='voxel',
		output_dir=tmp_path / 'first',
		slices=ClusterSliceRequest(xy_slices=(0,)),
		amplitude=first_amplitude,
		amplitude_alpha=0.65,
	)[0]
	second = save_cluster_slice_pngs(
		labels,
		survey_id='survey',
		k=1,
		mode='voxel',
		output_dir=tmp_path / 'second',
		slices=ClusterSliceRequest(xy_slices=(0,)),
		amplitude=second_amplitude,
		amplitude_alpha=0.65,
	)[0]

	assert not np.array_equal(plt.imread(first), plt.imread(second))


def test_proc_visualization_token_mode_uses_amplitude_underlay(
	tmp_path: Path,
) -> None:
	first = _run_token_underlay_visualization(
		tmp_path / 'first',
		np.zeros((4, 4, 1), dtype=np.float32),
	)
	second = _run_token_underlay_visualization(
		tmp_path / 'second',
		np.linspace(-1.0, 1.0, 16, dtype=np.float32).reshape(4, 4, 1),
	)

	assert not np.array_equal(plt.imread(first), plt.imread(second))


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


@pytest.mark.parametrize('slice_value', [1.9, True])
def test_proc_visualization_rejects_non_integer_slice_values(
	tmp_path: Path,
	slice_value: object,
) -> None:
	with pytest.raises(TypeError, match=r'visualization\.xy_slices'):
		run_cluster_visualization(
			{
				'clustering': {'input_dir': str(tmp_path / 'cluster_run')},
				'visualization': {
					'output_dir': str(tmp_path / 'figures'),
					'xy_slices': [slice_value],
					'xz_slices': [0],
				},
			},
		)


def _run_token_underlay_visualization(root: Path, amplitude: np.ndarray) -> Path:
	input_dir = root / 'cluster_run'
	output_dir = root / 'figures'
	labels_dir = input_dir / 'labels' / 'k1'
	labels_dir.mkdir(parents=True)
	np.save(
		labels_dir / 'survey.cluster_labels_token.npy',
		np.zeros((2, 2, 1), dtype=np.int32),
	)
	amplitude_path = root / 'survey.npy'
	np.save(amplitude_path, amplitude)
	(labels_dir / 'survey.cluster_label_metadata.json').write_text(
		json.dumps(
			{
				'source_amplitude_path': str(amplitude_path),
				'patch_size': [2, 2, 1],
			},
		)
		+ '\n',
		encoding='utf-8',
	)

	result = run_cluster_visualization(
		{
			'clustering': {'input_dir': str(input_dir)},
			'visualization': {
				'output_dir': str(output_dir),
				'modes': ['token'],
				'xy_slices': [0],
				'xz_slices': [],
				'summaries': {'enabled': False},
				'amplitude_underlay': {'enabled': True, 'alpha': 0.8},
			},
		},
	)

	assert result == {'png_count': 1, 'voxel_count': 0, 'summary_count': 0}
	return output_dir / 'token' / 'survey_k1_xy_z0.png'
