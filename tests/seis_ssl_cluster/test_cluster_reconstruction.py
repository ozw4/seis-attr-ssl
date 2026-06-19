from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

from seis_ssl_cluster.clustering.reconstruct import (
	reconstruct_labels_for_survey,
	reconstruct_voxel_labels,
)
from seis_ssl_cluster.clustering.summaries import (
	ClusterSummaryInput,
	write_cluster_summaries,
)

if TYPE_CHECKING:
	from pathlib import Path


def test_token_labels_upsample_to_clipped_voxel_shape_and_keep_invalid(
	tmp_path: Path,
) -> None:
	token_labels = np.array(
		[
			[[0, -1], [1, 2]],
			[[1, 2], [-1, 0]],
		],
		dtype=np.int32,
	)
	voxel_path = tmp_path / 'survey.cluster_labels_voxel.npy'

	voxels = reconstruct_voxel_labels(
		token_labels,
		patch_size_xyz=(2, 2, 2),
		volume_shape_xyz=(3, 4, 3),
		output_path=voxel_path,
	)

	assert voxel_path.is_file()
	assert voxels.shape == (3, 4, 3)
	assert voxels[0, 0, 0] == 0
	assert voxels[0, 0, 2] == -1
	assert voxels[0, 3, 0] == 1
	assert voxels[2, 3, 0] == -1
	np.testing.assert_array_equal(np.load(voxel_path), voxels)


def test_reconstruct_labels_for_survey_uses_embedding_metadata_shape(
	tmp_path: Path,
) -> None:
	labels_dir = tmp_path / 'labels' / 'k3'
	embedding_dir = tmp_path / 'embeddings'
	labels_dir.mkdir(parents=True)
	embedding_dir.mkdir()
	token_path = labels_dir / 'survey.cluster_labels_token.npy'
	embedding_metadata_path = embedding_dir / 'survey.embedding_metadata.json'
	label_metadata_path = labels_dir / 'survey.cluster_label_metadata.json'
	np.save(token_path, np.zeros((2, 2, 1), dtype=np.int32))
	embedding_metadata_path.write_text(
		json.dumps(
			{
				'patch_size': [2, 3, 4],
				'volume_shape_xyz': [3, 5, 4],
			},
		)
		+ '\n',
		encoding='utf-8',
	)
	label_metadata_path.write_text(
		json.dumps(
			{
				'embedding_input': {
					'metadata_path': str(embedding_metadata_path),
				},
			},
		)
		+ '\n',
		encoding='utf-8',
	)

	result = reconstruct_labels_for_survey(
		token_path,
		metadata_path=label_metadata_path,
	)

	assert result.voxel_labels_path == labels_dir / 'survey.cluster_labels_voxel.npy'
	assert np.load(result.voxel_labels_path).shape == (3, 5, 4)


def test_cluster_summary_counts_equal_valid_assigned_tokens(tmp_path: Path) -> None:
	labels_path = tmp_path / 'survey.cluster_labels_token.npy'
	embeddings_path = tmp_path / 'survey.embeddings.npy'
	np.save(
		labels_path,
		np.array([[[0, 1], [-1, 1]], [[2, -1], [2, 2]]], dtype=np.int32),
	)
	np.save(embeddings_path, np.ones((2, 2, 2, 3), dtype=np.float32))

	artifacts = write_cluster_summaries(
		[
			ClusterSummaryInput(
				survey_id='survey',
				labels_path=labels_path,
				embeddings_path=embeddings_path,
			),
		],
		k=3,
		output_dir=tmp_path / 'summary',
	)

	payload = json.loads(artifacts.json_path.read_text(encoding='utf-8'))
	assert payload['total_valid_token_count'] == 6
	assert payload['total_invalid_token_count'] == 2
	assert [row['token_count'] for row in payload['clusters']] == [1, 2, 3]
	assert artifacts.csv_path.is_file()
	assert artifacts.png_path.is_file()
