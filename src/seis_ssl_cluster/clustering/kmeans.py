"""Mini-batch k-means clustering for embedding artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Protocol, cast

import numpy as np

from seis_ssl_cluster.clustering.features import (
	SurveyEmbeddingArtifact,
	discover_embedding_artifacts,
	iter_prediction_feature_batches,
	sample_features,
	validate_embedding_compatibility,
)
from seis_ssl_cluster.clustering.writer import (
	build_clustering_metadata,
	write_cluster_labels,
	write_clustering_metadata,
)


class _Predictor(Protocol):
	def predict(self, x: np.ndarray) -> np.ndarray:
		"""Return cluster labels for a feature batch."""


@dataclass(frozen=True)
class KMeansClusteringResult:
	"""Output metadata for one requested k value."""

	k: int
	output_dir: Path
	metadata_path: Path
	label_paths: dict[str, Path]


def cluster_embeddings(  # noqa: PLR0913
	input_dir: str | Path,
	output_dir: str | Path,
	*,
	k_values: Sequence[int],
	survey_ids: Sequence[str] | None = None,
	max_samples_per_survey: int | None = None,
	batch_size: int = 4096,
	random_state: int | None = 0,
) -> list[KMeansClusteringResult]:
	"""Cluster compatible survey embedding artifacts for each requested k."""
	normalized_k_values = _validate_k_values(k_values)
	artifacts = discover_embedding_artifacts(input_dir, survey_ids)
	signature = validate_embedding_compatibility(artifacts)
	features = sample_features(
		artifacts,
		max_samples_per_survey=max_samples_per_survey,
		random_state=random_state,
	)
	root = Path(output_dir)
	results: list[KMeansClusteringResult] = []
	for k in normalized_k_values:
		if features.shape[0] < k:
			msg = (
				f'k={k} requires at least {k} sampled features; '
				f'got {features.shape[0]}'
			)
			raise ValueError(msg)
		from sklearn.cluster import MiniBatchKMeans  # noqa: PLC0415

		model = MiniBatchKMeans(
			n_clusters=k,
			random_state=random_state,
			batch_size=max(batch_size, k),
			n_init='auto',
		)
		model.fit(features)
		k_output_dir = root / f'k_{k}'
		label_paths = _predict_and_write_labels(
			model,
			artifacts,
			k_output_dir,
			batch_size=batch_size,
		)
		metadata = build_clustering_metadata(
			artifacts,
			k=k,
			compatibility_signature=signature,
			label_paths=label_paths,
			inertia=float(model.inertia_),
			n_iter=int(model.n_iter_),
			random_state=random_state,
		)
		metadata_path = write_clustering_metadata(
			k_output_dir / 'clustering_metadata.json',
			metadata,
		)
		results.append(
			KMeansClusteringResult(
				k=k,
				output_dir=k_output_dir,
				metadata_path=metadata_path,
				label_paths=label_paths,
			),
		)
	return results


def run_minibatch_kmeans(
	config: Mapping[str, object],
	*,
	random_state: int | None = 0,
) -> list[KMeansClusteringResult]:
	"""Run mini-batch k-means from a validated cluster config mapping."""
	embeddings = _required_mapping(config, 'embeddings')
	clustering = _required_mapping(config, 'clustering')
	return cluster_embeddings(
		_required_path(embeddings, 'input_dir', 'embeddings'),
		_required_path(clustering, 'output_dir', 'clustering'),
		k_values=_k_values_from_config(clustering),
		survey_ids=_optional_str_sequence(clustering.get('survey_ids')),
		max_samples_per_survey=_optional_positive_int(
			clustering.get('max_samples_per_survey'),
			'clustering.max_samples_per_survey',
		),
		batch_size=_positive_int(
			clustering.get('batch_size', 4096),
			'clustering.batch_size',
		),
		random_state=random_state,
	)


def _predict_and_write_labels(
	model: _Predictor,
	artifacts: Sequence[SurveyEmbeddingArtifact],
	output_dir: Path,
	*,
	batch_size: int,
) -> dict[str, Path]:
	label_paths: dict[str, Path] = {}
	for artifact in artifacts:
		valid_tokens = np.load(artifact.valid_tokens_path, mmap_mode='r')
		labels = np.full(valid_tokens.shape, -1, dtype=np.int32)
		flat_labels = labels.reshape(-1)
		for batch in iter_prediction_feature_batches(artifact, batch_size=batch_size):
			flat_labels[batch.flat_indices] = model.predict(batch.features)
		label_paths[artifact.survey_id] = write_cluster_labels(
			output_dir / f'{artifact.survey_id}.clusters.npy',
			labels,
		)
	return label_paths


def _validate_k_values(k_values: Sequence[int]) -> list[int]:
	if isinstance(k_values, str) or not k_values:
		msg = f'k_values must be a non-empty sequence of integers; got {k_values!r}'
		raise ValueError(msg)
	normalized = [_positive_int(k, 'k_values') for k in k_values]
	if len(set(normalized)) != len(normalized):
		msg = f'duplicate k_values are not allowed: {normalized!r}'
		raise ValueError(msg)
	return normalized


def _k_values_from_config(clustering: Mapping[str, object]) -> list[int]:
	value = clustering.get('k_values')
	if value is None:
		value = clustering.get('num_clusters')
	if isinstance(value, Sequence) and not isinstance(value, str | bytes):
		return _validate_k_values(cast('Sequence[int]', value))
	return _validate_k_values([_positive_int(value, 'clustering.num_clusters')])


def _required_mapping(
	parent: Mapping[str, object],
	key: str,
) -> Mapping[str, object]:
	value = parent.get(key)
	if not isinstance(value, Mapping):
		msg = f'{key} must be a mapping'
		raise TypeError(msg)
	return cast('Mapping[str, object]', value)


def _required_path(parent: Mapping[str, object], key: str, prefix: str) -> Path:
	value = parent.get(key)
	if not isinstance(value, str) or not value:
		msg = f'{prefix}.{key} must be a non-empty string; got {value!r}'
		raise TypeError(msg)
	return Path(value)


def _optional_str_sequence(value: object) -> Sequence[str] | None:
	if value is None:
		return None
	if isinstance(value, str) or not isinstance(value, Sequence):
		msg = f'survey_ids must be a sequence of strings; got {value!r}'
		raise TypeError(msg)
	if not all(isinstance(item, str) and item for item in value):
		msg = f'survey_ids must contain only non-empty strings; got {value!r}'
		raise TypeError(msg)
	return cast('Sequence[str]', value)


def _optional_positive_int(value: object, name: str) -> int | None:
	if value is None:
		return None
	return _positive_int(value, name)


def _positive_int(value: object, name: str) -> int:
	if isinstance(value, bool) or not isinstance(value, Integral):
		msg = f'{name} must be an integer; got {value!r}'
		raise TypeError(msg)
	integer = int(value)
	if integer <= 0:
		msg = f'{name} must be positive; got {integer!r}'
		raise ValueError(msg)
	return integer


__all__ = [
	'KMeansClusteringResult',
	'cluster_embeddings',
	'run_minibatch_kmeans',
]
