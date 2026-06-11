from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from seis_attr_ssl.attributes import MVP_ATTRIBUTE_REGISTRY
from seis_attr_ssl.data import AttributeVolumeRecord, SurveyManifest
from seis_attr_ssl.data.attribute_subset import sample_attribute_subset
from seis_attr_ssl.data.pretrain_dataset import NopimsAttributePretrainDataset

LOCAL_SIZE = (4, 4, 4)
CONTEXT_SIZE = (8, 8, 8)
CONTEXT_DOWNSAMPLE = 2


def _write_manifest(
	root: Path,
	attribute_names: tuple[str, ...],
	shape_xyz: tuple[int, int, int] = (10, 10, 10),
	fill_value: float | None = None,
) -> SurveyManifest:
	records: dict[str, AttributeVolumeRecord] = {}
	for name in attribute_names:
		id_ = MVP_ATTRIBUTE_REGISTRY.name_to_id(name)
		path = root / 'attributes' / f'{name}.npy'
		path.parent.mkdir(parents=True, exist_ok=True)
		value = float(id_) if fill_value is None else fill_value
		array = np.full(shape_xyz, value, dtype=np.float32)
		np.save(path, array)
		records[name] = AttributeVolumeRecord(
			survey_id='survey-a',
			attribute_name=name,
			path=Path('attributes') / f'{name}.npy',
			shape_xyz=shape_xyz,
			dtype='float32',
			grid_order=('x', 'y', 'z'),
			is_memmap_safe=True,
		)

	return SurveyManifest(
		survey_id='survey-a',
		root=root,
		attribute_volumes=records,
		shape_xyz=shape_xyz,
	)


def _dataset(
	manifest: SurveyManifest,
	**kwargs: object,
) -> NopimsAttributePretrainDataset:
	options = {
		'local_crop_size_xyz': LOCAL_SIZE,
		'context_crop_size_xyz': CONTEXT_SIZE,
		'context_downsample': CONTEXT_DOWNSAMPLE,
		'min_input_attributes': 2,
		'max_input_attributes': 4,
		'seed': 123,
	}
	options.update(kwargs)
	return NopimsAttributePretrainDataset([manifest], **options)


def test_sample_attribute_subset_includes_amplitude_and_registry_orders() -> None:
	subset = sample_attribute_subset(
		[4, 0, 2, 1],
		min_input_attributes=3,
		max_input_attributes=3,
		rng=np.random.default_rng(5),
	)

	assert subset[0] == MVP_ATTRIBUTE_REGISTRY.name_to_id('amplitude_norm')
	assert subset == tuple(sorted(subset))
	assert len(subset) == 3


def test_sample_attribute_subset_raises_when_too_few_available() -> None:
	with pytest.raises(ValueError, match='not enough available attributes'):
		sample_attribute_subset([0], 2, 4, np.random.default_rng(5))


def test_pretrain_dataset_sample_contract_shapes_and_order(tmp_path: Path) -> None:
	manifest = _write_manifest(tmp_path / 'survey-a', MVP_ATTRIBUTE_REGISTRY.names)
	dataset = _dataset(manifest)

	sample = dataset[0]

	assert set(sample) == {
		'x',
		'target',
		'attribute_ids',
		'target_attribute_ids',
		'valid_attributes',
		'target_valid',
		'coords',
		'context',
		'context_valid_mask',
		'local_valid_mask',
	}
	x = sample['x']
	target = sample['target']
	attribute_ids = sample['attribute_ids']
	assert isinstance(x, np.ndarray)
	assert isinstance(target, np.ndarray)
	assert isinstance(attribute_ids, np.ndarray)
	assert x.shape == (len(attribute_ids), *LOCAL_SIZE)
	assert target.shape == (len(MVP_ATTRIBUTE_REGISTRY.specs), *LOCAL_SIZE)
	assert sample['context'].shape == (len(attribute_ids), *LOCAL_SIZE)
	assert sample['context_valid_mask'].shape == LOCAL_SIZE
	assert sample['local_valid_mask'].shape == LOCAL_SIZE
	assert target.dtype == np.float32
	assert x.dtype == np.float32
	np.testing.assert_array_equal(
		sample['target_attribute_ids'],
		np.arange(len(MVP_ATTRIBUTE_REGISTRY.specs), dtype=np.int64),
	)
	np.testing.assert_array_equal(sample['target_valid'], np.ones(10, dtype=bool))
	for row, id_ in enumerate(attribute_ids):
		np.testing.assert_array_equal(x[row], target[id_])


def test_pretrain_dataset_is_deterministic_for_seed_and_index(tmp_path: Path) -> None:
	manifest = _write_manifest(tmp_path / 'survey-a', MVP_ATTRIBUTE_REGISTRY.names)
	first = _dataset(manifest, samples_per_epoch=3)
	second = _dataset(manifest, samples_per_epoch=3)

	first_sample = first[1]
	second_sample = second[1]

	for key in ('x', 'target', 'attribute_ids', 'target_valid', 'local_valid_mask'):
		np.testing.assert_array_equal(first_sample[key], second_sample[key])
	assert first_sample['coords'] == second_sample['coords']


def test_pretrain_dataset_marks_missing_targets_but_samples_available_subset(
	tmp_path: Path,
) -> None:
	names = MVP_ATTRIBUTE_REGISTRY.names[:4]
	manifest = _write_manifest(tmp_path / 'survey-a', names)
	dataset = _dataset(manifest, max_input_attributes=3, use_context=False)

	sample = dataset[0]

	np.testing.assert_array_equal(
		sample['target_valid'],
		np.asarray([True, True, True, True, False, False, False, False, False, False]),
	)
	assert set(sample['attribute_ids']).issubset(set(range(4)))
	assert sample['context'] is None
	assert sample['context_valid_mask'] is None


def test_pretrain_dataset_requires_amplitude_norm(tmp_path: Path) -> None:
	manifest = _write_manifest(tmp_path / 'survey-a', MVP_ATTRIBUTE_REGISTRY.names[1:4])

	with pytest.raises(ValueError, match='amplitude_norm'):
		_dataset(manifest)


def test_pretrain_dataset_context_downsample_ignores_boundary_padding(
	tmp_path: Path,
) -> None:
	names = MVP_ATTRIBUTE_REGISTRY.names[:2]
	manifest = _write_manifest(
		tmp_path / 'survey-a',
		names,
		shape_xyz=(2, 2, 2),
		fill_value=7.0,
	)
	dataset = _dataset(
		manifest,
		local_crop_size_xyz=(2, 2, 2),
		context_crop_size_xyz=(4, 4, 4),
		context_downsample=2,
		min_input_attributes=2,
		max_input_attributes=2,
	)

	sample = dataset[0]

	np.testing.assert_array_equal(
		sample['context'],
		np.full((2, 2, 2, 2), 7.0, dtype=np.float32),
	)
	np.testing.assert_array_equal(
		sample['context_valid_mask'],
		np.ones((2, 2, 2), dtype=bool),
	)
