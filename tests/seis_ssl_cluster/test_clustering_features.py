from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from seis_ssl_cluster.clustering.features import (
	discover_embedding_artifacts,
	iter_prediction_feature_batches,
	sample_features,
	validate_embedding_compatibility,
)

if TYPE_CHECKING:
	from pathlib import Path


def test_compatible_artifacts_share_signature(tmp_path: Path) -> None:
	_write_artifact(tmp_path, 'survey_a')
	_write_artifact(tmp_path, 'survey_b', token_grid_shape=(1, 3, 2))

	artifacts = discover_embedding_artifacts(tmp_path)
	signature = validate_embedding_compatibility(artifacts)

	assert signature['checkpoint_sha256'] == 'checkpoint-a'
	assert signature['embedding_dim'] == 2
	assert {artifact.survey_id for artifact in artifacts} == {'survey_a', 'survey_b'}


def test_different_checkpoint_hashes_are_rejected(tmp_path: Path) -> None:
	_write_artifact(tmp_path, 'survey_a')
	_write_artifact(tmp_path, 'survey_b', checkpoint_sha256='checkpoint-b')

	artifacts = discover_embedding_artifacts(tmp_path)
	with pytest.raises(ValueError, match=r'survey_a.*survey_b.*checkpoint_sha256'):
		validate_embedding_compatibility(artifacts)


def test_different_patch_or_model_geometry_is_rejected(tmp_path: Path) -> None:
	_write_artifact(tmp_path, 'survey_a')
	metadata = _metadata('survey_b')
	metadata['model_geometry'] = {
		**metadata['model_geometry'],
		'patch_size': [2, 2, 1],
	}
	metadata['patch_size_xyz'] = [2, 2, 1]
	_write_artifact(tmp_path, 'survey_b', metadata=metadata)

	artifacts = discover_embedding_artifacts(tmp_path)
	with pytest.raises(ValueError, match=r'model_geometry.*patch_size'):
		validate_embedding_compatibility(artifacts)


def test_different_window_overlap_or_zero_mask_contract_is_rejected(
	tmp_path: Path,
) -> None:
	_write_artifact(tmp_path, 'survey_a')
	metadata = _metadata('survey_b')
	metadata['window_size_xyz'] = [4, 4, 2]
	metadata['overlap_xyz'] = [2, 0, 0]
	metadata['zero_mask'] = {**metadata['zero_mask'], 'enabled': False}
	_write_artifact(tmp_path, 'survey_b', metadata=metadata)

	artifacts = discover_embedding_artifacts(tmp_path)
	with pytest.raises(ValueError, match=r'window_size.*overlap.*zero_mask'):
		validate_embedding_compatibility(artifacts)


def test_survey_specific_token_grid_shapes_remain_allowed(tmp_path: Path) -> None:
	_write_artifact(tmp_path, 'survey_a', token_grid_shape=(1, 2, 2))
	_write_artifact(tmp_path, 'survey_b', token_grid_shape=(2, 2, 2))

	artifacts = discover_embedding_artifacts(tmp_path)
	validate_embedding_compatibility(artifacts)


def test_sampled_features_report_survey_for_non_finite_values(
	tmp_path: Path,
) -> None:
	_write_artifact(tmp_path, 'survey_a')
	features = np.ones((1, 2, 2, 2), dtype=np.float32)
	features[0, 0, 0, 1] = np.nan
	_write_artifact(tmp_path, 'survey_bad', embeddings=features)

	artifacts = discover_embedding_artifacts(tmp_path)
	with pytest.raises(ValueError, match=r'survey_bad.*sampled features.*non-finite'):
		sample_features(artifacts, random_state=0)


def test_prediction_batches_report_survey_for_non_finite_values(
	tmp_path: Path,
) -> None:
	features = np.ones((1, 2, 2, 2), dtype=np.float32)
	features[0, 1, 0, 0] = np.inf
	_write_artifact(tmp_path, 'survey_bad', embeddings=features)
	artifact = discover_embedding_artifacts(tmp_path)[0]

	with pytest.raises(
		ValueError,
		match=r'survey_bad.*prediction features.*non-finite',
	):
		list(iter_prediction_feature_batches(artifact, batch_size=2))


def _write_artifact(  # noqa: PLR0913
	root: Path,
	survey_id: str,
	*,
	token_grid_shape: tuple[int, int, int] = (1, 2, 2),
	checkpoint_sha256: str = 'checkpoint-a',
	metadata: dict[str, object] | None = None,
	embeddings: np.ndarray | None = None,
) -> None:
	if embeddings is None:
		values = np.arange(np.prod((*token_grid_shape, 2)), dtype=np.float32)
		embeddings = values.reshape(*token_grid_shape, 2)
	metadata = metadata or _metadata(
		survey_id,
		token_grid_shape=tuple(int(axis) for axis in embeddings.shape[:-1]),
		checkpoint_sha256=checkpoint_sha256,
	)
	valid_tokens = np.ones(embeddings.shape[:-1], dtype=bool)
	np.save(root / f'{survey_id}.embeddings.npy', embeddings)
	np.save(root / f'{survey_id}.valid_tokens.npy', valid_tokens)
	(root / f'{survey_id}.embedding_metadata.json').write_text(
		json.dumps(metadata, sort_keys=True),
		encoding='utf-8',
	)


def _metadata(
	survey_id: str,
	*,
	token_grid_shape: tuple[int, int, int] = (1, 2, 2),
	checkpoint_sha256: str = 'checkpoint-a',
) -> dict[str, object]:
	return {
		'survey_id': survey_id,
		'source_amplitude_path': f'/data/{survey_id}.npy',
		'checkpoint_path': '/checkpoints/model.pt',
		'checkpoint_sha256': checkpoint_sha256,
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
