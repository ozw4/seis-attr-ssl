"""Embedding artifact loading and feature-batch validation for clustering."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, cast

import numpy as np

from seis_ssl_cluster.embedding.writer import file_sha256

JsonMapping: TypeAlias = Mapping[str, object]
CompatibilitySignature: TypeAlias = dict[str, object]

_METADATA_SUFFIX = '.embedding_metadata.json'
_FIELD_SOURCES: tuple[tuple[str, tuple[str, ...]], ...] = (
	('checkpoint_sha256', ('checkpoint_sha256',)),
	('model_geometry', ('model_geometry',)),
	('patch_size', ('patch_size_xyz', 'patch_size')),
	('window_size', ('window_size_xyz', 'window_size')),
	('overlap', ('overlap_xyz', 'overlap')),
	('min_token_valid_fraction', ('min_token_valid_fraction',)),
	('zero_mask', ('zero_mask',)),
	('embedding_dim', ('embedding_dim',)),
)


@dataclass(frozen=True)
class SurveyEmbeddingArtifact:
	"""Resolved files and parsed metadata for one survey embedding artifact."""

	survey_id: str
	embeddings_path: Path
	valid_tokens_path: Path
	metadata_path: Path
	metadata_sha256: str
	metadata: JsonMapping
	compatibility_signature: CompatibilitySignature


@dataclass(frozen=True)
class PredictionFeatureBatch:
	"""One finite prediction batch from a survey embedding artifact."""

	artifact: SurveyEmbeddingArtifact
	flat_indices: np.ndarray
	features: np.ndarray


def discover_embedding_artifacts(
	input_dir: str | Path,
	survey_ids: Sequence[str] | None = None,
) -> list[SurveyEmbeddingArtifact]:
	"""Parse embedding metadata for selected surveys before array loading."""
	root = Path(input_dir)
	if survey_ids is None:
		metadata_paths = sorted(root.glob(f'*{_METADATA_SUFFIX}'))
		if not metadata_paths:
			msg = f'no embedding metadata files found in {root}'
			raise FileNotFoundError(msg)
		return [_artifact_from_metadata_path(path) for path in metadata_paths]
	return [load_embedding_artifact(root, survey_id) for survey_id in survey_ids]


def load_embedding_artifact(
	input_dir: str | Path,
	survey_id: str,
) -> SurveyEmbeddingArtifact:
	"""Parse one survey's embedding metadata and resolve its arrays."""
	root = Path(input_dir)
	return _artifact_from_metadata_path(root / f'{survey_id}{_METADATA_SUFFIX}')


def compatibility_signature(metadata: JsonMapping) -> CompatibilitySignature:
	"""Return the representation-defining compatibility signature."""
	signature: CompatibilitySignature = {}
	for field, source_keys in _FIELD_SOURCES:
		signature[field] = _metadata_value(metadata, field, source_keys)
	return signature


def validate_embedding_compatibility(
	artifacts: Sequence[SurveyEmbeddingArtifact],
) -> CompatibilitySignature:
	"""Require all surveys to share the same compatibility signature."""
	if not artifacts:
		msg = 'at least one embedding artifact is required'
		raise ValueError(msg)
	reference = artifacts[0]
	reference_signature = reference.compatibility_signature
	mismatches: list[str] = []
	for artifact in artifacts[1:]:
		differing_fields = [
			field
			for field, value in artifact.compatibility_signature.items()
			if reference_signature.get(field) != value
		]
		if differing_fields:
			fields = ', '.join(differing_fields)
			mismatches.append(
				f'{reference.survey_id} vs {artifact.survey_id}: {fields}',
			)
	if mismatches:
		msg = 'incompatible embedding artifacts; differing fields: ' + '; '.join(
			mismatches,
		)
		raise ValueError(msg)
	return dict(reference_signature)


def sample_features(
	artifacts: Sequence[SurveyEmbeddingArtifact],
	*,
	max_samples_per_survey: int | None = None,
	random_state: int | np.random.Generator | None = None,
) -> np.ndarray:
	"""Sample finite valid-token features from compatible embedding artifacts."""
	validate_embedding_compatibility(artifacts)
	rng = _rng(random_state)
	feature_batches = [
		_sample_survey_features(
			artifact,
			max_samples_per_survey=max_samples_per_survey,
			rng=rng,
		)
		for artifact in artifacts
	]
	if not feature_batches:
		msg = 'no embedding features were sampled'
		raise ValueError(msg)
	return np.concatenate(feature_batches, axis=0).astype(np.float32, copy=False)


def iter_prediction_feature_batches(
	artifact: SurveyEmbeddingArtifact,
	*,
	batch_size: int,
) -> Iterator[PredictionFeatureBatch]:
	"""Yield finite valid-token feature batches for prediction."""
	if batch_size <= 0:
		msg = f'batch_size must be positive; got {batch_size!r}'
		raise ValueError(msg)
	embeddings, valid_tokens = _load_arrays(artifact)
	flat_embeddings = embeddings.reshape(-1, embeddings.shape[-1])
	valid_indices = np.flatnonzero(valid_tokens.reshape(-1))
	for start in range(0, len(valid_indices), batch_size):
		flat_indices = valid_indices[start : start + batch_size]
		features = np.asarray(flat_embeddings[flat_indices], dtype=np.float32)
		_require_finite(features, artifact.survey_id, 'prediction features')
		yield PredictionFeatureBatch(artifact, flat_indices, features)


def _artifact_from_metadata_path(metadata_path: Path) -> SurveyEmbeddingArtifact:
	if not metadata_path.is_file():
		msg = f'missing embedding metadata: {metadata_path}'
		raise FileNotFoundError(msg)
	survey_id = _survey_id_from_metadata_path(metadata_path)
	metadata = _read_metadata(metadata_path)
	root = metadata_path.parent
	artifact = SurveyEmbeddingArtifact(
		survey_id=survey_id,
		embeddings_path=root / f'{survey_id}.embeddings.npy',
		valid_tokens_path=root / f'{survey_id}.valid_tokens.npy',
		metadata_path=metadata_path,
		metadata_sha256=file_sha256(metadata_path),
		metadata=metadata,
		compatibility_signature=compatibility_signature(metadata),
	)
	_require_file(artifact.embeddings_path, 'embedding array')
	_require_file(artifact.valid_tokens_path, 'valid-token array')
	return artifact


def _read_metadata(path: Path) -> JsonMapping:
	payload = json.loads(path.read_text(encoding='utf-8'))
	if not isinstance(payload, Mapping):
		msg = f'embedding metadata must be a JSON object: {path}'
		raise TypeError(msg)
	return cast('JsonMapping', payload)


def _survey_id_from_metadata_path(path: Path) -> str:
	name = path.name
	if not name.endswith(_METADATA_SUFFIX):
		msg = f'embedding metadata filename must end with {_METADATA_SUFFIX}: {path}'
		raise ValueError(msg)
	return name[: -len(_METADATA_SUFFIX)]


def _metadata_value(
	metadata: JsonMapping,
	field: str,
	source_keys: Sequence[str],
) -> object:
	for key in source_keys:
		if key in metadata:
			return metadata[key]
	msg = f'embedding metadata is missing compatibility field {field!r}'
	raise KeyError(msg)


def _sample_survey_features(
	artifact: SurveyEmbeddingArtifact,
	*,
	max_samples_per_survey: int | None,
	rng: np.random.Generator,
) -> np.ndarray:
	embeddings, valid_tokens = _load_arrays(artifact)
	flat_embeddings = embeddings.reshape(-1, embeddings.shape[-1])
	valid_indices = np.flatnonzero(valid_tokens.reshape(-1))
	if len(valid_indices) == 0:
		msg = f'survey {artifact.survey_id} has no valid embedding tokens'
		raise ValueError(msg)
	if max_samples_per_survey is not None:
		if max_samples_per_survey <= 0:
			msg = (
				'max_samples_per_survey must be positive; '
				f'got {max_samples_per_survey!r}'
			)
			raise ValueError(msg)
		if len(valid_indices) > max_samples_per_survey:
			valid_indices = rng.choice(
				valid_indices,
				size=max_samples_per_survey,
				replace=False,
			)
	features = np.asarray(flat_embeddings[valid_indices], dtype=np.float32)
	_require_finite(features, artifact.survey_id, 'sampled features')
	return features


def _load_arrays(
	artifact: SurveyEmbeddingArtifact,
) -> tuple[np.ndarray, np.ndarray]:
	embeddings = np.load(artifact.embeddings_path, mmap_mode='r')
	valid_tokens = np.load(artifact.valid_tokens_path, mmap_mode='r')
	if embeddings.ndim != 4:
		msg = (
			f'survey {artifact.survey_id} embeddings must have shape '
			f'(x, y, z, dim); got {embeddings.shape!r}'
		)
		raise ValueError(msg)
	if valid_tokens.shape != embeddings.shape[:-1]:
		msg = (
			f'survey {artifact.survey_id} valid_tokens shape '
			f'{valid_tokens.shape!r} does not match embeddings token grid '
			f'{embeddings.shape[:-1]!r}'
		)
		raise ValueError(msg)
	expected_dim = int(artifact.compatibility_signature['embedding_dim'])
	if embeddings.shape[-1] != expected_dim:
		msg = (
			f'survey {artifact.survey_id} embedding dimension '
			f'{embeddings.shape[-1]!r} does not match metadata {expected_dim!r}'
		)
		raise ValueError(msg)
	return embeddings, np.asarray(valid_tokens, dtype=bool)


def _require_finite(features: np.ndarray, survey_id: str, label: str) -> None:
	if not np.isfinite(features).all():
		msg = f'survey {survey_id} {label} contain non-finite values'
		raise ValueError(msg)


def _require_file(path: Path, label: str) -> None:
	if not path.is_file():
		msg = f'missing {label}: {path}'
		raise FileNotFoundError(msg)


def _rng(
	random_state: int | np.random.Generator | None,
) -> np.random.Generator:
	if isinstance(random_state, np.random.Generator):
		return random_state
	return np.random.default_rng(random_state)


__all__ = [
	'CompatibilitySignature',
	'PredictionFeatureBatch',
	'SurveyEmbeddingArtifact',
	'compatibility_signature',
	'discover_embedding_artifacts',
	'iter_prediction_feature_batches',
	'load_embedding_artifact',
	'sample_features',
	'validate_embedding_compatibility',
]
