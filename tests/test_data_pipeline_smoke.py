from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from seis_attr_ssl.attributes import MVP_ATTRIBUTE_REGISTRY
from seis_attr_ssl.data import (
	NopimsAttributePretrainDataset,
	build_nopims_base_seismic_manifests,
	compute_normalization_stats,
	read_manifest_json,
	write_normalization_stats,
)

if TYPE_CHECKING:
	from pathlib import Path

SHAPE_XYZ = (32, 32, 32)
LOCAL_SIZE_XYZ = (8, 8, 8)
CONTEXT_SIZE_XYZ = (16, 16, 16)
CONTEXT_DOWNSAMPLE = 2
EXPECTED_SAMPLE_KEYS = {
	'x',
	'target',
	'attribute_ids',
	'spatial_mask',
	'visible_spatial_mask',
	'attribute_input_mask',
	'attribute_target_mask',
	'dropped_attribute_mask',
	'target_attribute_ids',
	'valid_attributes',
	'target_valid',
	'coords',
	'context',
	'context_valid_mask',
	'local_valid_mask',
}


def _build_synthetic_nopims_tree(root: Path) -> None:
	for survey_index, survey_id in enumerate(('survey_001', 'survey_002')):
		path = root / survey_id / 'seismic' / 'dip_steered_median_filtered.npy'
		path.parent.mkdir(parents=True, exist_ok=True)
		volume = (
			np.arange(np.prod(SHAPE_XYZ), dtype=np.float32).reshape(SHAPE_XYZ)
			+ np.float32(survey_index)
		)
		np.save(path, volume)
		stats = compute_normalization_stats(
			path,
			survey_id=survey_id,
			max_samples=None,
		)
		write_normalization_stats(
			stats,
			root / survey_id / 'normalization_stats.json',
		)


def test_synthetic_nopims_data_pipeline_smoke(tmp_path: Path) -> None:  # noqa: PLR0915
	nopims_root = tmp_path / 'NOPIMS'
	manifest_path = tmp_path / 'manifests' / 'nopims_base_seismic_manifests.json'
	_build_synthetic_nopims_tree(nopims_root)

	manifests = build_nopims_base_seismic_manifests(
		nopims_root=nopims_root,
		output_path=manifest_path,
		scan_pattern='**/*.npy',
	)
	loaded_manifests = read_manifest_json(manifest_path)

	assert loaded_manifests == manifests
	assert [manifest.survey_id for manifest in loaded_manifests] == [
		'survey_001',
		'survey_002',
	]
	for manifest in loaded_manifests:
		assert manifest.base_seismic is not None
		assert manifest.attribute_volumes == {}
		assert manifest.missing_attributes() == ()
		assert manifest.shape_xyz == SHAPE_XYZ

	dataset = NopimsAttributePretrainDataset(
		loaded_manifests,
		local_crop_size_xyz=LOCAL_SIZE_XYZ,
		context_crop_size_xyz=CONTEXT_SIZE_XYZ,
		context_downsample=CONTEXT_DOWNSAMPLE,
		min_input_attributes=4,
		max_input_attributes=6,
		seed=17,
		samples_per_epoch=4,
	)

	for sample in (dataset[0], dataset[1]):
		assert set(sample) == EXPECTED_SAMPLE_KEYS
		x = sample['x']
		target = sample['target']
		attribute_ids = sample['attribute_ids']
		spatial_mask = sample['spatial_mask']
		visible_spatial_mask = sample['visible_spatial_mask']
		attribute_input_mask = sample['attribute_input_mask']
		attribute_target_mask = sample['attribute_target_mask']
		dropped_attribute_mask = sample['dropped_attribute_mask']
		target_attribute_ids = sample['target_attribute_ids']
		valid_attributes = sample['valid_attributes']
		context = sample['context']
		context_valid_mask = sample['context_valid_mask']
		local_valid_mask = sample['local_valid_mask']

		assert isinstance(x, np.ndarray)
		assert isinstance(target, np.ndarray)
		assert isinstance(attribute_ids, np.ndarray)
		assert isinstance(spatial_mask, np.ndarray)
		assert isinstance(visible_spatial_mask, np.ndarray)
		assert isinstance(attribute_input_mask, np.ndarray)
		assert isinstance(attribute_target_mask, np.ndarray)
		assert isinstance(dropped_attribute_mask, np.ndarray)
		assert isinstance(target_attribute_ids, np.ndarray)
		assert isinstance(valid_attributes, np.ndarray)
		assert isinstance(context, np.ndarray)
		assert isinstance(context_valid_mask, np.ndarray)
		assert isinstance(local_valid_mask, np.ndarray)

		assert x.dtype == np.float32
		assert target.dtype == np.float32
		assert attribute_ids.dtype == np.int64
		assert spatial_mask.dtype == np.bool_
		assert visible_spatial_mask.dtype == np.bool_
		assert attribute_input_mask.dtype == np.bool_
		assert attribute_target_mask.dtype == np.bool_
		assert dropped_attribute_mask.dtype == np.bool_
		assert target_attribute_ids.dtype == np.int64
		assert valid_attributes.dtype == np.bool_
		target_valid = sample['target_valid']
		assert isinstance(target_valid, np.ndarray)
		assert target_valid.dtype == np.bool_

		assert 4 <= len(attribute_ids) <= 6
		assert spatial_mask.shape == (1, 1, 1)
		assert visible_spatial_mask.shape == spatial_mask.shape
		assert attribute_input_mask.shape == (len(MVP_ATTRIBUTE_REGISTRY.specs),)
		assert attribute_target_mask.shape == attribute_input_mask.shape
		assert dropped_attribute_mask.shape == attribute_input_mask.shape
		assert x.shape == (len(attribute_ids), *LOCAL_SIZE_XYZ)
		assert target.shape == (
			len(MVP_ATTRIBUTE_REGISTRY.specs),
			*LOCAL_SIZE_XYZ,
		)
		assert context.shape == (len(attribute_ids), *LOCAL_SIZE_XYZ)
		assert context_valid_mask.shape == LOCAL_SIZE_XYZ
		assert local_valid_mask.shape == LOCAL_SIZE_XYZ

		np.testing.assert_array_equal(
			target_attribute_ids,
			np.arange(len(MVP_ATTRIBUTE_REGISTRY.specs), dtype=np.int64),
		)
		np.testing.assert_array_equal(
			attribute_ids,
			np.flatnonzero(attribute_input_mask).astype(np.int64),
		)
		np.testing.assert_array_equal(
			visible_spatial_mask,
			np.logical_not(spatial_mask),
		)
		np.testing.assert_array_equal(
			sample['target_valid'],
			np.ones(len(MVP_ATTRIBUTE_REGISTRY.specs), dtype=bool),
		)
		np.testing.assert_array_equal(attribute_target_mask, sample['target_valid'])
		np.testing.assert_array_equal(
			dropped_attribute_mask,
			np.logical_and(attribute_target_mask, np.logical_not(attribute_input_mask)),
		)
		np.testing.assert_array_equal(
			valid_attributes,
			np.ones(len(attribute_ids), dtype=bool),
		)
		assert set(attribute_ids).issubset(set(target_attribute_ids))
		assert MVP_ATTRIBUTE_REGISTRY.name_to_id('amplitude_norm') in set(attribute_ids)
