"""Data loading and sampling components."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from seis_attr_ssl.data.attribute_generation import (
		AttributeGenerationConfig,
		AttributeGenerationResult,
		generate_mvp_attributes,
	)
	from seis_attr_ssl.data.attribute_subset import (
		AMPLITUDE_ATTRIBUTE_ID,
		MVP_ATTRIBUTE_IDS,
		sample_attribute_subset,
	)
	from seis_attr_ssl.data.crop_sampler import (
		compute_centered_start,
		make_context_request,
		sample_random_center,
		sample_random_local_crop,
	)
	from seis_attr_ssl.data.downsample import downsample_context_mean
	from seis_attr_ssl.data.manifest_builder import (
		ManifestBuildResult,
		ManifestBuildSummary,
		build_nopims_base_seismic_manifests,
		build_nopims_manifests,
		scan_nopims_base_seismic_manifests,
		scan_nopims_manifests,
		summarize_manifests,
	)
	from seis_attr_ssl.data.normalization import (
		SurveyNormalizationStats,
		compute_normalization_stats,
		load_normalization_stats,
		normalize_amplitude,
		write_normalization_stats,
	)
	from seis_attr_ssl.data.pretrain_dataset import NopimsAttributePretrainDataset
	from seis_attr_ssl.data.schema import (
		BASE_SEISMIC_KIND_DIP_STEERED_MEDIAN_FILTERED,
		GRID_ORDER_XYZ,
		AttributeVolumeRecord,
		BaseSeismicVolumeRecord,
		CropRequest,
		SurveyManifest,
		TensorLike,
		UnlabeledPretrainingSample,
		read_manifest_json,
		survey_manifest_from_dict,
		survey_manifest_to_dict,
		write_manifest_json,
	)
	from seis_attr_ssl.data.volume_store import (
		NpyMemmapVolumeStore,
		NpyVolumeInfo,
		inspect_npy_volume,
	)

__all__ = [
	'AMPLITUDE_ATTRIBUTE_ID',
	'BASE_SEISMIC_KIND_DIP_STEERED_MEDIAN_FILTERED',
	'GRID_ORDER_XYZ',
	'MVP_ATTRIBUTE_IDS',
	'AttributeGenerationConfig',
	'AttributeGenerationResult',
	'AttributeVolumeRecord',
	'BaseSeismicVolumeRecord',
	'CropRequest',
	'ManifestBuildResult',
	'ManifestBuildSummary',
	'NopimsAttributePretrainDataset',
	'NpyMemmapVolumeStore',
	'NpyVolumeInfo',
	'SurveyManifest',
	'SurveyNormalizationStats',
	'TensorLike',
	'UnlabeledPretrainingSample',
	'build_nopims_base_seismic_manifests',
	'build_nopims_manifests',
	'compute_centered_start',
	'compute_normalization_stats',
	'downsample_context_mean',
	'generate_mvp_attributes',
	'inspect_npy_volume',
	'load_normalization_stats',
	'make_context_request',
	'normalize_amplitude',
	'read_manifest_json',
	'sample_attribute_subset',
	'sample_random_center',
	'sample_random_local_crop',
	'scan_nopims_base_seismic_manifests',
	'scan_nopims_manifests',
	'summarize_manifests',
	'survey_manifest_from_dict',
	'survey_manifest_to_dict',
	'write_manifest_json',
	'write_normalization_stats',
]

_MANIFEST_BUILDER_EXPORTS = {
	'ManifestBuildResult',
	'ManifestBuildSummary',
	'build_nopims_base_seismic_manifests',
	'build_nopims_manifests',
	'scan_nopims_base_seismic_manifests',
	'scan_nopims_manifests',
	'summarize_manifests',
}

_CROP_SAMPLER_EXPORTS = {
	'compute_centered_start',
	'make_context_request',
	'sample_random_center',
	'sample_random_local_crop',
}

_ATTRIBUTE_SUBSET_EXPORTS = {
	'AMPLITUDE_ATTRIBUTE_ID',
	'MVP_ATTRIBUTE_IDS',
	'sample_attribute_subset',
}

_ATTRIBUTE_GENERATION_EXPORTS = {
	'AttributeGenerationConfig',
	'AttributeGenerationResult',
	'generate_mvp_attributes',
}

_DOWNSAMPLE_EXPORTS = {
	'downsample_context_mean',
}

_PRETRAIN_DATASET_EXPORTS = {
	'NopimsAttributePretrainDataset',
}

_NORMALIZATION_EXPORTS = {
	'SurveyNormalizationStats',
	'compute_normalization_stats',
	'load_normalization_stats',
	'normalize_amplitude',
	'write_normalization_stats',
}

_SCHEMA_EXPORTS = {
	'GRID_ORDER_XYZ',
	'AttributeVolumeRecord',
	'BASE_SEISMIC_KIND_DIP_STEERED_MEDIAN_FILTERED',
	'BaseSeismicVolumeRecord',
	'CropRequest',
	'SurveyManifest',
	'TensorLike',
	'UnlabeledPretrainingSample',
	'read_manifest_json',
	'survey_manifest_from_dict',
	'survey_manifest_to_dict',
	'write_manifest_json',
}

_VOLUME_STORE_EXPORTS = {
	'NpyMemmapVolumeStore',
	'NpyVolumeInfo',
	'inspect_npy_volume',
}

_EXPORT_MODULES = {
	**dict.fromkeys(_MANIFEST_BUILDER_EXPORTS, 'seis_attr_ssl.data.manifest_builder'),
	**dict.fromkeys(_CROP_SAMPLER_EXPORTS, 'seis_attr_ssl.data.crop_sampler'),
	**dict.fromkeys(
		_ATTRIBUTE_GENERATION_EXPORTS,
		'seis_attr_ssl.data.attribute_generation',
	),
	**dict.fromkeys(_ATTRIBUTE_SUBSET_EXPORTS, 'seis_attr_ssl.data.attribute_subset'),
	**dict.fromkeys(_DOWNSAMPLE_EXPORTS, 'seis_attr_ssl.data.downsample'),
	**dict.fromkeys(_NORMALIZATION_EXPORTS, 'seis_attr_ssl.data.normalization'),
	**dict.fromkeys(_PRETRAIN_DATASET_EXPORTS, 'seis_attr_ssl.data.pretrain_dataset'),
	**dict.fromkeys(_SCHEMA_EXPORTS, 'seis_attr_ssl.data.schema'),
	**dict.fromkeys(_VOLUME_STORE_EXPORTS, 'seis_attr_ssl.data.volume_store'),
}


def __getattr__(name: str) -> object:
	"""Lazily expose data schema objects."""
	try:
		module_name = _EXPORT_MODULES[name]
	except KeyError as exc:
		msg = f'module {__name__!r} has no attribute {name!r}'
		raise AttributeError(msg) from exc
	return getattr(import_module(module_name), name)
