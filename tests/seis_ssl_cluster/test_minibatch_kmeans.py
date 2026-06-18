from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

from seis_ssl_cluster.clustering import run_embedding_clustering

if TYPE_CHECKING:
	from pathlib import Path


def test_run_embedding_clustering_writes_deterministic_labels_for_multiple_k(
	tmp_path: Path,
) -> None:
	input_dir = tmp_path / 'embeddings'
	first_output = tmp_path / 'clusters-a'
	second_output = tmp_path / 'clusters-b'
	input_dir.mkdir()
	_write_embedding_artifacts(
		input_dir,
		'survey_a',
		embeddings=np.array(
			[
				[[[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]]],
				[[[0.0, 1.0, 0.0], [0.0, 0.9, 0.1]]],
			],
			dtype=np.float32,
		),
		valid=np.array([[[True, True]], [[True, False]]]),
	)
	_write_embedding_artifacts(
		input_dir,
		'survey_b',
		embeddings=np.array(
			[
				[[[0.0, 0.0, 1.0], [0.1, 0.0, 0.9]]],
				[[[1.0, 0.1, 0.0], [0.0, 1.0, 0.1]]],
			],
			dtype=np.float32,
		),
		valid=np.array([[[True, True]], [[True, True]]]),
	)

	first = run_embedding_clustering(_config(input_dir, first_output))
	second = run_embedding_clustering(_config(input_dir, second_output))

	assert [result.k for result in first.results] == [2, 3]
	assert [result.k for result in second.results] == [2, 3]
	assert first.sample.sample_count == 7
	assert first.sample.total_valid_count == 7
	for output_dir in (first_output, second_output):
		for k in (2, 3):
			assert (output_dir / 'models' / f'k{k}' / 'preprocessor.joblib').is_file()
			assert (output_dir / 'models' / f'k{k}' / 'kmeans.joblib').is_file()
			assert (output_dir / 'models' / f'k{k}' / 'cluster_centers.npy').is_file()
			metadata = json.loads(
				(output_dir / 'models' / f'k{k}' / 'clustering_metadata.json')
				.read_text(encoding='utf-8'),
			)
			assert metadata['sample']['count'] == 7
			assert metadata['invalid_token_count'] == 1

	survey_a_first = np.load(
		first_output / 'labels' / 'k2' / 'survey_a.cluster_labels_token.npy',
	)
	survey_a_second = np.load(
		second_output / 'labels' / 'k2' / 'survey_a.cluster_labels_token.npy',
	)
	np.testing.assert_array_equal(survey_a_first, survey_a_second)
	assert survey_a_first.shape == (2, 1, 2)
	assert survey_a_first[1, 0, 1] == -1
	assert np.all(survey_a_first[np.array([[[True, True]], [[True, False]]])] >= 0)

	for result in first.results:
		assert sum(result.cluster_counts.values()) == 7
		assert result.invalid_token_count == 1


def _config(input_dir: Path, output_dir: Path) -> dict[str, object]:
	return {
		'embeddings': {'input_dir': str(input_dir)},
		'clustering': {
			'output_dir': str(output_dir),
			'embedding_normalization': 'l2',
			'pca': {
				'enabled': True,
				'n_components': 2,
				'whiten': False,
			},
			'sample_tokens': 100,
			'method': 'minibatch_kmeans',
			'k_values': [2, 3],
			'minibatch_size': 4,
			'seed': 42,
		},
	}


def _write_embedding_artifacts(
	root: Path,
	survey_id: str,
	*,
	embeddings: np.ndarray,
	valid: np.ndarray,
) -> None:
	np.save(root / f'{survey_id}.embeddings.npy', embeddings)
	np.save(root / f'{survey_id}.valid_tokens.npy', valid.astype(np.bool_))
	(root / f'{survey_id}.embedding_metadata.json').write_text(
		json.dumps({'survey_id': survey_id}) + '\n',
		encoding='utf-8',
	)
