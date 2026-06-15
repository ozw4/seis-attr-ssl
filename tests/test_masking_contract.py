from __future__ import annotations

from pathlib import Path

import numpy as np

from seis_attr_ssl.attributes import MVP_ATTRIBUTE_REGISTRY
from seis_attr_ssl.data import AttributeVolumeRecord, SurveyManifest
from seis_attr_ssl.data.pretrain_dataset import NopimsAttributePretrainDataset

LOCAL_SIZE_XYZ = (2, 2, 2)
CONTEXT_SIZE_XYZ = (4, 4, 4)
CONTEXT_DOWNSAMPLE = 2
PATCH_SIZE_XYZ = (1, 1, 1)
TOKEN_GRID_SHAPE = (2, 2, 2)
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


def _write_synthetic_manifest(root: Path) -> SurveyManifest:
	attribute_names = MVP_ATTRIBUTE_REGISTRY.names[:2]
	records: dict[str, AttributeVolumeRecord] = {}
	for name in attribute_names:
		path = root / 'attributes' / f'{name}.npy'
		path.parent.mkdir(parents=True, exist_ok=True)
		np.save(path, np.full(LOCAL_SIZE_XYZ, 7.0, dtype=np.float32))
		records[name] = AttributeVolumeRecord(
			survey_id='survey-a',
			attribute_name=name,
			path=Path('attributes') / f'{name}.npy',
			shape_xyz=LOCAL_SIZE_XYZ,
			dtype='float32',
			grid_order=('x', 'y', 'z'),
			is_memmap_safe=True,
		)

	return SurveyManifest(
		survey_id='survey-a',
		root=root,
		attribute_volumes=records,
		shape_xyz=LOCAL_SIZE_XYZ,
	)


def test_pretrain_sample_masking_contract(tmp_path: Path) -> None:
	manifest = _write_synthetic_manifest(tmp_path / 'survey-a')
	dataset = NopimsAttributePretrainDataset(
		[manifest],
		local_crop_size_xyz=LOCAL_SIZE_XYZ,
		context_crop_size_xyz=CONTEXT_SIZE_XYZ,
		context_downsample=CONTEXT_DOWNSAMPLE,
		require_full_halo_inside_volume=False,
		patch_size_xyz=PATCH_SIZE_XYZ,
		min_input_attributes=2,
		max_input_attributes=2,
		seed=123,
	)

	sample = dataset[0]

	assert set(sample) == EXPECTED_SAMPLE_KEYS

	x = sample['x']
	spatial_mask = sample['spatial_mask']
	visible_spatial_mask = sample['visible_spatial_mask']
	attribute_input_mask = sample['attribute_input_mask']
	attribute_target_mask = sample['attribute_target_mask']
	dropped_attribute_mask = sample['dropped_attribute_mask']
	attribute_ids = sample['attribute_ids']
	context = sample['context']
	context_valid_mask = sample['context_valid_mask']

	assert isinstance(x, np.ndarray)
	assert isinstance(spatial_mask, np.ndarray)
	assert isinstance(visible_spatial_mask, np.ndarray)
	assert isinstance(attribute_input_mask, np.ndarray)
	assert isinstance(attribute_target_mask, np.ndarray)
	assert isinstance(dropped_attribute_mask, np.ndarray)
	assert isinstance(attribute_ids, np.ndarray)
	assert isinstance(context, np.ndarray)
	assert isinstance(context_valid_mask, np.ndarray)

	assert spatial_mask.shape == TOKEN_GRID_SHAPE
	assert visible_spatial_mask.shape == TOKEN_GRID_SHAPE
	assert attribute_input_mask.shape == (len(MVP_ATTRIBUTE_REGISTRY.specs),)
	assert attribute_target_mask.shape == (len(MVP_ATTRIBUTE_REGISTRY.specs),)
	assert dropped_attribute_mask.shape == (len(MVP_ATTRIBUTE_REGISTRY.specs),)

	np.testing.assert_array_equal(visible_spatial_mask, np.logical_not(spatial_mask))
	np.testing.assert_array_equal(
		dropped_attribute_mask,
		np.logical_and(attribute_target_mask, np.logical_not(attribute_input_mask)),
	)
	np.testing.assert_array_equal(
		attribute_ids,
		np.flatnonzero(attribute_input_mask).astype(np.int64),
	)
	assert x.shape[0] == attribute_ids.shape[0]

	np.testing.assert_array_equal(
		attribute_target_mask,
		np.asarray(
			[True, True, False, False, False, False, False, False, False, False],
			dtype=bool,
		),
	)
	np.testing.assert_array_equal(
		context,
		np.full((2, *LOCAL_SIZE_XYZ), 7.0, dtype=np.float32),
	)
	np.testing.assert_array_equal(
		context_valid_mask,
		np.ones(LOCAL_SIZE_XYZ, dtype=bool),
	)
