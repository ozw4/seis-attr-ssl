from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from seis_ssl_cluster.clustering.kmeans import cluster_embeddings, run_minibatch_kmeans

if TYPE_CHECKING:
	from pathlib import Path


def test_two_compatible_survey_artifacts_cluster_successfully(
	tmp_path: Path,
) -> None:
	input_dir = tmp_path / 'embeddings'
	output_dir = tmp_path / 'clusters'
	input_dir.mkdir()
	_write_artifact(input_dir, 'survey_a', offset=0.0)
	_write_artifact(input_dir, 'survey_b', offset=10.0)

	results = cluster_embeddings(
		input_dir,
		output_dir,
		k_values=[2],
		random_state=0,
		batch_size=2,
	)

	assert len(results) == 1
	result = results[0]
	assert result.k == 2
	for survey_id in ('survey_a', 'survey_b'):
		labels = np.load(result.label_paths[survey_id])
		assert labels.shape == (1, 2, 2)
		assert set(np.unique(labels)).issubset({0, 1})
	metadata = json.loads(result.metadata_path.read_text(encoding='utf-8'))
	assert metadata['inputs'][0]['metadata_path'].endswith(
		'survey_a.embedding_metadata.json',
	)
	assert metadata['inputs'][0]['metadata_sha256']
	assert metadata['inputs'][1]['embedding_metadata']['survey_id'] == 'survey_b'


def test_duplicate_k_values_are_rejected(tmp_path: Path) -> None:
	input_dir = tmp_path / 'embeddings'
	input_dir.mkdir()
	_write_artifact(input_dir, 'survey_a')

	with pytest.raises(ValueError, match='duplicate k_values'):
		cluster_embeddings(input_dir, tmp_path / 'clusters', k_values=[2, 2])


def test_run_minibatch_kmeans_rejects_duplicate_k_values_from_config(
	tmp_path: Path,
) -> None:
	config = {
		'embeddings': {'input_dir': str(tmp_path / 'embeddings')},
		'clustering': {
			'output_dir': str(tmp_path / 'clusters'),
			'k_values': [2, 2],
		},
	}

	with pytest.raises(ValueError, match='duplicate k_values'):
		run_minibatch_kmeans(config)


def test_prediction_non_finite_error_names_survey(tmp_path: Path) -> None:
	input_dir = tmp_path / 'embeddings'
	input_dir.mkdir()
	_write_artifact(input_dir, 'survey_a')
	features = np.ones((1, 2, 2, 2), dtype=np.float32)
	features[0, 0, 0, 0] = np.nan
	_write_artifact(input_dir, 'survey_bad', embeddings=features)

	with pytest.raises(ValueError, match=r'survey_bad.*sampled features.*non-finite'):
		cluster_embeddings(input_dir, tmp_path / 'clusters', k_values=[2])


def _write_artifact(
	root: Path,
	survey_id: str,
	*,
	offset: float = 0.0,
	embeddings: np.ndarray | None = None,
) -> None:
	if embeddings is None:
		embeddings = np.array(
			[
				[
					[[offset, 0.0], [offset + 0.1, 0.0]],
					[[offset + 9.0, 0.0], [offset + 9.1, 0.0]],
				],
			],
			dtype=np.float32,
		)
	valid_tokens = np.ones(embeddings.shape[:-1], dtype=bool)
	np.save(root / f'{survey_id}.embeddings.npy', embeddings)
	np.save(root / f'{survey_id}.valid_tokens.npy', valid_tokens)
	(root / f'{survey_id}.embedding_metadata.json').write_text(
		json.dumps(_metadata(survey_id, embeddings.shape[:-1]), sort_keys=True),
		encoding='utf-8',
	)


def _metadata(
	survey_id: str,
	token_grid_shape: tuple[int, int, int],
) -> dict[str, object]:
	return {
		'survey_id': survey_id,
		'source_amplitude_path': f'/data/{survey_id}.npy',
		'checkpoint_path': '/checkpoints/model.pt',
		'checkpoint_sha256': 'checkpoint-a',
		'model_geometry': {
			'name': 'amp_mae3d',
			'in_channels': 1,
			'out_channels': 1,
			'patch_size': [2, 2, 2],
			'encoder_dim': 2,
			'encoder_depth': 1,
			'encoder_heads': 1,
			'decoder_dim': 2,
			'decoder_depth': 1,
			'decoder_heads': 1,
		},
		'volume_shape_xyz': [axis * 2 for axis in token_grid_shape],
		'token_grid_shape_xyz': list(token_grid_shape),
		'window_size_xyz': [4, 4, 4],
		'overlap_xyz': [0, 0, 0],
		'patch_size_xyz': [2, 2, 2],
		'embedding_dim': 2,
		'output_dtype': 'float32',
		'normalization_stats_path': f'/stats/{survey_id}.json',
		'min_token_valid_fraction': 1.0,
		'zero_mask': {
			'enabled': True,
			'zero_atol': 0.0,
			'z_sample_influence_radius': 1,
			'xy_trace_influence_radius': 0,
		},
	}
