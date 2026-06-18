"""Writers for clustering labels and provenance metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
	from collections.abc import Mapping, Sequence

	from seis_ssl_cluster.clustering.features import (
		CompatibilitySignature,
		SurveyEmbeddingArtifact,
	)


def write_cluster_labels(path: str | Path, labels: np.ndarray) -> Path:
	"""Write one survey cluster-label grid."""
	output_path = Path(path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	np.save(output_path, np.asarray(labels, dtype=np.int32))
	return output_path


def build_clustering_metadata(  # noqa: PLR0913
	artifacts: Sequence[SurveyEmbeddingArtifact],
	*,
	k: int,
	compatibility_signature: CompatibilitySignature,
	label_paths: Mapping[str, Path],
	inertia: float,
	n_iter: int,
	random_state: int | None,
) -> dict[str, object]:
	"""Build deterministic clustering metadata with input provenance."""
	return {
		'algorithm': 'minibatch_kmeans',
		'num_clusters': int(k),
		'compatibility_signature': compatibility_signature,
		'inertia': float(inertia),
		'n_iter': int(n_iter),
		'random_state': random_state,
		'inputs': [
			{
				'survey_id': artifact.survey_id,
				'embeddings_path': str(artifact.embeddings_path),
				'valid_tokens_path': str(artifact.valid_tokens_path),
				'metadata_path': str(artifact.metadata_path),
				'metadata_sha256': artifact.metadata_sha256,
				'embedding_metadata': dict(artifact.metadata),
			}
			for artifact in artifacts
		],
		'outputs': [
			{
				'survey_id': survey_id,
				'labels_path': str(label_paths[survey_id]),
			}
			for survey_id in sorted(label_paths)
		],
	}


def write_clustering_metadata(
	path: str | Path,
	metadata: Mapping[str, object],
) -> Path:
	"""Write deterministic clustering metadata JSON."""
	metadata_path = Path(path)
	metadata_path.parent.mkdir(parents=True, exist_ok=True)
	tmp_path = metadata_path.with_name(f'{metadata_path.name}.tmp')
	tmp_path.write_text(
		json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + '\n',
		encoding='utf-8',
	)
	tmp_path.replace(metadata_path)
	return metadata_path


__all__ = [
	'build_clustering_metadata',
	'write_cluster_labels',
	'write_clustering_metadata',
]
