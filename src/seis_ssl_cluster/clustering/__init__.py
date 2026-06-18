"""Clustering components for seismic SSL clustering."""

import importlib

from seis_ssl_cluster.clustering.features import (
	EmbeddingInput,
	FeatureBatch,
	count_valid_tokens,
	discover_embedding_inputs,
	embedding_dim,
	embedding_input_metadata,
	extract_token_features,
	file_sha256,
	iter_valid_feature_batches,
	load_valid_tokens,
	open_embedding_array,
	valid_flat_indices,
)
from seis_ssl_cluster.clustering.sampling import (
	SampledTokens,
	sample_valid_embedding_tokens,
)

_KMEANS_EXPORTS = {
	'ClusteringRunResult',
	'ClusteringSettings',
	'KClusteringResult',
	'PCASettings',
	'clustering_settings_from_config',
	'fit_minibatch_kmeans',
	'fit_preprocessor',
	'run_embedding_clustering',
}
_WRITER_EXPORTS = {
	'SurveyLabelResult',
	'write_json',
	'write_labels_for_k',
	'write_model_artifacts',
}

__all__ = [
	'ClusteringRunResult',
	'ClusteringSettings',
	'EmbeddingInput',
	'FeatureBatch',
	'KClusteringResult',
	'PCASettings',
	'SampledTokens',
	'SurveyLabelResult',
	'clustering_settings_from_config',
	'count_valid_tokens',
	'discover_embedding_inputs',
	'embedding_dim',
	'embedding_input_metadata',
	'extract_token_features',
	'file_sha256',
	'fit_minibatch_kmeans',
	'fit_preprocessor',
	'iter_valid_feature_batches',
	'load_valid_tokens',
	'open_embedding_array',
	'run_embedding_clustering',
	'sample_valid_embedding_tokens',
	'valid_flat_indices',
	'write_json',
	'write_labels_for_k',
	'write_model_artifacts',
]


def __getattr__(name: str) -> object:
	"""Lazily import optional clustering dependencies."""
	if name in _KMEANS_EXPORTS:
		kmeans = importlib.import_module('seis_ssl_cluster.clustering.kmeans')
		return getattr(kmeans, name)
	if name in _WRITER_EXPORTS:
		writer = importlib.import_module('seis_ssl_cluster.clustering.writer')
		return getattr(writer, name)
	msg = f'module {__name__!r} has no attribute {name!r}'
	raise AttributeError(msg)
