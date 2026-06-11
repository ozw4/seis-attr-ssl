from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from seis_attr_ssl.attributes import MVP_ATTRIBUTE_REGISTRY
from seis_attr_ssl.config import load_config
from seis_attr_ssl.data import (
	AttributeVolumeRecord,
	BaseSeismicVolumeRecord,
	CropRequest,
	SurveyManifest,
)
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


def _manifest_metadata(
	root: Path,
	attribute_names: tuple[str, ...],
	shape_xyz: tuple[int, int, int],
) -> SurveyManifest:
	return SurveyManifest(
		survey_id='survey-a',
		root=root,
		attribute_volumes={
			name: AttributeVolumeRecord(
				survey_id='survey-a',
				attribute_name=name,
				path=Path('attributes') / f'{name}.npy',
				shape_xyz=shape_xyz,
				dtype='float32',
				grid_order=('x', 'y', 'z'),
				is_memmap_safe=True,
			)
			for name in attribute_names
		},
		shape_xyz=shape_xyz,
	)


def _write_base_manifest(
	root: Path,
	shape_xyz: tuple[int, int, int] = (10, 10, 10),
) -> SurveyManifest:
	seismic_path = root / 'dip_steered_median_filtered.npy'
	seismic_path.parent.mkdir(parents=True, exist_ok=True)
	np.save(
		seismic_path,
		np.arange(np.prod(shape_xyz), dtype=np.float32).reshape(shape_xyz),
	)
	stats_path = root / 'normalization_stats.json'
	stats_path.write_text(
		(
			'{'
			'"survey_id": "survey-a", '
			f'"source_path": "{seismic_path}", '
			'"grid_order": ["x", "y", "z"], '
			'"clip_low_percentile": 0.5, '
			'"clip_high_percentile": 99.5, '
			'"clip_low": 0.0, '
			'"clip_high": 999.0, '
			'"median": 0.0, '
			'"iqr": 10.0, '
			'"eps": 1.0e-6'
			'}'
		),
		encoding='utf-8',
	)
	return SurveyManifest(
		survey_id='survey-a',
		root=root,
		attribute_volumes={},
		shape_xyz=shape_xyz,
		base_seismic=BaseSeismicVolumeRecord(
			survey_id='survey-a',
			path=Path('dip_steered_median_filtered.npy'),
			kind='dip_steered_median_filtered',
			shape_xyz=shape_xyz,
			dtype='float32',
			grid_order=('x', 'y', 'z'),
			normalization_stats_path=Path('normalization_stats.json'),
		),
	)


def _dataset(
	manifest: SurveyManifest,
	**kwargs: object,
) -> NopimsAttributePretrainDataset:
	options = {
		'local_crop_size_xyz': LOCAL_SIZE,
		'context_crop_size_xyz': CONTEXT_SIZE,
		'context_downsample': CONTEXT_DOWNSAMPLE,
		'patch_size_xyz': LOCAL_SIZE,
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
	x = sample['x']
	target = sample['target']
	attribute_ids = sample['attribute_ids']
	attribute_input_mask = sample['attribute_input_mask']
	attribute_target_mask = sample['attribute_target_mask']
	dropped_attribute_mask = sample['dropped_attribute_mask']
	spatial_mask = sample['spatial_mask']
	visible_spatial_mask = sample['visible_spatial_mask']
	assert isinstance(x, np.ndarray)
	assert isinstance(target, np.ndarray)
	assert isinstance(attribute_ids, np.ndarray)
	assert isinstance(attribute_input_mask, np.ndarray)
	assert isinstance(attribute_target_mask, np.ndarray)
	assert isinstance(dropped_attribute_mask, np.ndarray)
	assert isinstance(spatial_mask, np.ndarray)
	assert isinstance(visible_spatial_mask, np.ndarray)
	assert x.shape == (len(attribute_ids), *LOCAL_SIZE)
	assert target.shape == (len(MVP_ATTRIBUTE_REGISTRY.specs), *LOCAL_SIZE)
	assert spatial_mask.shape == (1, 1, 1)
	assert visible_spatial_mask.shape == spatial_mask.shape
	assert attribute_input_mask.shape == (len(MVP_ATTRIBUTE_REGISTRY.specs),)
	assert attribute_target_mask.shape == attribute_input_mask.shape
	assert dropped_attribute_mask.shape == attribute_input_mask.shape
	assert spatial_mask.dtype == np.bool_
	assert visible_spatial_mask.dtype == np.bool_
	assert attribute_input_mask.dtype == np.bool_
	assert attribute_target_mask.dtype == np.bool_
	assert dropped_attribute_mask.dtype == np.bool_
	assert sample['context'].shape == (len(attribute_ids), *LOCAL_SIZE)
	assert sample['context_valid_mask'].shape == LOCAL_SIZE
	assert sample['local_valid_mask'].shape == LOCAL_SIZE
	assert target.dtype == np.float32
	assert x.dtype == np.float32
	np.testing.assert_array_equal(
		sample['target_attribute_ids'],
		np.arange(len(MVP_ATTRIBUTE_REGISTRY.specs), dtype=np.int64),
	)
	np.testing.assert_array_equal(
		attribute_ids,
		np.flatnonzero(attribute_input_mask).astype(np.int64),
	)
	np.testing.assert_array_equal(sample['target_valid'], np.ones(10, dtype=bool))
	np.testing.assert_array_equal(attribute_target_mask, sample['target_valid'])
	np.testing.assert_array_equal(visible_spatial_mask, np.logical_not(spatial_mask))
	np.testing.assert_array_equal(
		dropped_attribute_mask,
		np.logical_and(attribute_target_mask, np.logical_not(attribute_input_mask)),
	)
	for row, id_ in enumerate(attribute_ids):
		np.testing.assert_array_equal(x[row], target[id_])


def test_pretrain_dataset_generates_attributes_from_base_seismic(
	tmp_path: Path,
) -> None:
	manifest = _write_base_manifest(tmp_path / 'survey-a')
	dataset = _dataset(manifest, use_context=False)

	sample = dataset[0]

	assert sample['target'].shape == (len(MVP_ATTRIBUTE_REGISTRY.specs), *LOCAL_SIZE)
	assert sample['x'].shape == (len(sample['attribute_ids']), *LOCAL_SIZE)
	np.testing.assert_array_equal(
		sample['target_valid'],
		np.ones(len(MVP_ATTRIBUTE_REGISTRY.specs), dtype=bool),
	)
	np.testing.assert_array_equal(
		sample['local_valid_mask'],
		np.ones(LOCAL_SIZE, dtype=bool),
	)
	for row, id_ in enumerate(sample['attribute_ids']):
		np.testing.assert_array_equal(sample['x'][row], sample['target'][id_])


def test_pretrain_dataset_reads_base_target_with_local_halo(
	tmp_path: Path,
) -> None:
	manifest = _write_base_manifest(tmp_path / 'survey-a', shape_xyz=(10, 10, 10))
	dataset = _dataset(
		manifest,
		use_context=False,
		local_attribute_halo_xyz=(1, 1, 2),
	)

	sample = dataset[0]
	coords = sample['coords']
	local_start = coords['local_start_xyz']
	local_compute_start = coords['local_compute_start_xyz']

	assert sample['target'].shape == (len(MVP_ATTRIBUTE_REGISTRY.specs), *LOCAL_SIZE)
	assert sample['x'].shape == (len(sample['attribute_ids']), *LOCAL_SIZE)
	assert sample['local_valid_mask'].shape == LOCAL_SIZE
	assert coords['local_attribute_halo_xyz'] == (1, 1, 2)
	assert coords['local_compute_size_xyz'] == (6, 6, 8)
	assert local_compute_start == (
		local_start[0] - 1,
		local_start[1] - 1,
		local_start[2] - 2,
	)
	assert bool(sample['local_valid_mask'].all())
	np.testing.assert_array_equal(
		sample['target_valid'],
		np.ones(len(MVP_ATTRIBUTE_REGISTRY.specs), dtype=bool),
	)


def test_pretrain_dataset_halo_sampling_falls_back_for_small_base_volume(
	tmp_path: Path,
) -> None:
	manifest = _write_base_manifest(tmp_path / 'survey-a', shape_xyz=(3, 3, 3))
	dataset = _dataset(
		manifest,
		use_context=False,
		local_crop_size_xyz=LOCAL_SIZE,
		local_attribute_halo_xyz=(2, 2, 2),
	)

	sample = dataset[0]

	assert sample['target'].shape == (len(MVP_ATTRIBUTE_REGISTRY.specs), *LOCAL_SIZE)
	assert sample['local_valid_mask'].shape == LOCAL_SIZE
	assert sample['coords']['local_compute_size_xyz'] == (8, 8, 8)
	assert not bool(sample['local_valid_mask'].all())


def test_pretrain_dataset_requires_base_normalization_stats(tmp_path: Path) -> None:
	manifest = _write_base_manifest(tmp_path / 'survey-a')
	(manifest.root / 'normalization_stats.json').unlink()

	with pytest.raises(FileNotFoundError, match='normalization stats'):
		_dataset(manifest, use_context=False)


def test_pretrain_dataset_requires_base_seismic_file(tmp_path: Path) -> None:
	manifest = _write_base_manifest(tmp_path / 'survey-a')
	(manifest.root / 'dip_steered_median_filtered.npy').unlink()

	with pytest.raises(FileNotFoundError, match='base seismic file'):
		_dataset(manifest, use_context=False)


def test_pretrain_dataset_default_mae_spatial_mask_shape_and_ratio(
	tmp_path: Path,
) -> None:
	manifest = _write_manifest(
		tmp_path / 'survey-a',
		MVP_ATTRIBUTE_REGISTRY.names,
		shape_xyz=(128, 128, 128),
	)
	dataset = _dataset(
		manifest,
		local_crop_size_xyz=(128, 128, 128),
		patch_size_xyz=(8, 8, 8),
		use_context=False,
	)

	sample = dataset[0]
	spatial_mask = sample['spatial_mask']

	assert spatial_mask.shape == (16, 16, 16)
	assert 0.74 <= float(spatial_mask.mean()) <= 0.76


def test_pretrain_dataset_from_config_wires_masking_values(tmp_path: Path) -> None:
	cfg = load_config(Path('proc/configs/mvp_mae.yaml'))
	manifest = _manifest_metadata(
		tmp_path / 'survey-a',
		MVP_ATTRIBUTE_REGISTRY.names,
		shape_xyz=(128, 128, 128),
	)

	dataset = NopimsAttributePretrainDataset.from_config([manifest], cfg)

	assert dataset.local_crop_size_xyz == (128, 128, 128)
	assert dataset.local_attribute_halo_xyz == (16, 16, 64)
	assert dataset.require_full_halo_inside_volume is True
	assert dataset.context_crop_size_xyz == (512, 512, 512)
	assert dataset.context_downsample == 4
	assert dataset.context_attribute_halo_xyz == (8, 8, 16)
	assert dataset.use_context is True
	assert dataset.patch_size_xyz == (8, 8, 8)
	assert dataset.spatial_mask_ratio == 0.75
	assert dataset.spatial_mask_mode == 'block'
	assert dataset.block_size_tokens_xyz == (2, 2, 2)
	assert dataset.min_input_attributes == 4
	assert dataset.max_input_attributes == 10
	assert dataset.attribute_dropout_prob == 0.30
	assert dataset.group_dropout_prob == 0.20
	assert dataset.seed == 42


def test_pretrain_dataset_mvp_halo_sample_contract(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	manifest = _manifest_metadata(
		tmp_path / 'survey-a',
		MVP_ATTRIBUTE_REGISTRY.names,
		shape_xyz=(1024, 1024, 1024),
	)

	def fake_read_target(
		self: NopimsAttributePretrainDataset,
		manifest: SurveyManifest,
		local_request: CropRequest,
	) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		del self, manifest, local_request
		target = np.broadcast_to(
			np.arange(len(MVP_ATTRIBUTE_REGISTRY.specs), dtype=np.float32).reshape(
				-1,
				1,
				1,
				1,
			),
			(len(MVP_ATTRIBUTE_REGISTRY.specs), 128, 128, 128),
		)
		return (
			target,
			np.ones(len(MVP_ATTRIBUTE_REGISTRY.specs), dtype=bool),
			np.ones((128, 128, 128), dtype=bool),
		)

	def fake_read_context(
		self: NopimsAttributePretrainDataset,
		manifest: SurveyManifest,
		local_request: CropRequest,
		input_ids: tuple[int, ...],
	) -> tuple[np.ndarray, np.ndarray]:
		del self, manifest, local_request
		context = np.broadcast_to(
			np.asarray(input_ids, dtype=np.float32).reshape(-1, 1, 1, 1),
			(len(input_ids), 128, 128, 128),
		)
		return context, np.ones((128, 128, 128), dtype=bool)

	monkeypatch.setattr(
		NopimsAttributePretrainDataset,
		'_read_target',
		fake_read_target,
	)
	monkeypatch.setattr(
		NopimsAttributePretrainDataset,
		'_read_context',
		fake_read_context,
	)
	dataset = NopimsAttributePretrainDataset(
		[manifest],
		local_crop_size_xyz=(128, 128, 128),
		local_attribute_halo_xyz=(16, 16, 64),
		require_full_halo_inside_volume=True,
		context_crop_size_xyz=(512, 512, 512),
		context_downsample=4,
		context_attribute_halo_xyz=(8, 8, 16),
		patch_size_xyz=(8, 8, 8),
		min_input_attributes=4,
		max_input_attributes=4,
		seed=7,
	)

	sample = dataset[0]
	coords = sample['coords']

	assert sample['target'].shape == (10, 128, 128, 128)
	assert sample['local_valid_mask'].shape == (128, 128, 128)
	assert coords['local_compute_size_xyz'] == (160, 160, 256)
	assert coords['local_attribute_halo_xyz'] == (16, 16, 64)
	assert sample['context'].shape == (4, 128, 128, 128)
	assert sample['context_valid_mask'].shape == (128, 128, 128)
	assert coords['context_compute_size_xyz'] == (576, 576, 640)
	assert coords['context_lowres_compute_size_xyz'] == (144, 144, 160)
	assert coords['context_attribute_halo_xyz'] == (8, 8, 16)
	assert coords['context_downsample'] == 4


def test_pretrain_dataset_context_compute_request_uses_lowres_halo(
	tmp_path: Path,
) -> None:
	manifest = _manifest_metadata(
		tmp_path / 'survey-a',
		MVP_ATTRIBUTE_REGISTRY.names,
		shape_xyz=(1024, 1024, 1024),
	)
	dataset = NopimsAttributePretrainDataset(
		[manifest],
		local_crop_size_xyz=(128, 128, 128),
		context_crop_size_xyz=(512, 512, 512),
		context_downsample=4,
		context_attribute_halo_xyz=(8, 8, 16),
	)
	local_request = CropRequest(
		survey_id='survey-a',
		start_xyz=(256, 256, 256),
		size_xyz=(128, 128, 128),
		context_size_xyz=None,
		context_downsample=1,
	)

	_, compute_request, lowres_payload_slices = dataset._context_requests(  # noqa: SLF001
		local_request,
	)

	assert compute_request.size_xyz == (576, 576, 640)
	assert tuple(
		size // dataset.context_downsample
		for size in compute_request.size_xyz
	) == (144, 144, 160)
	assert lowres_payload_slices == (
		slice(8, 136),
		slice(8, 136),
		slice(16, 144),
	)


def test_pretrain_dataset_context_halo_trims_to_payload_mask(
	tmp_path: Path,
) -> None:
	manifest = _write_base_manifest(tmp_path / 'survey-a', shape_xyz=(12, 12, 12))
	dataset = _dataset(
		manifest,
		local_crop_size_xyz=(4, 4, 4),
		context_crop_size_xyz=(8, 8, 8),
		context_downsample=2,
		context_attribute_halo_xyz=(1, 1, 2),
		patch_size_xyz=(4, 4, 4),
		min_input_attributes=2,
		max_input_attributes=2,
	)

	sample = dataset[0]

	assert sample['context'].shape == (2, 4, 4, 4)
	assert sample['context_valid_mask'].shape == (4, 4, 4)
	assert sample['coords']['context_attribute_halo_xyz'] == (1, 1, 2)
	assert sample['coords']['context_compute_size_xyz'] == (12, 12, 16)
	assert sample['coords']['context_lowres_compute_size_xyz'] == (6, 6, 8)


def test_pretrain_dataset_zero_context_halo_keeps_payload_request(
	tmp_path: Path,
) -> None:
	names = MVP_ATTRIBUTE_REGISTRY.names[:2]
	manifest = _write_manifest(
		tmp_path / 'survey-a',
		names,
		shape_xyz=(8, 8, 8),
		fill_value=3.0,
	)
	dataset = _dataset(
		manifest,
		local_crop_size_xyz=(4, 4, 4),
		context_crop_size_xyz=(8, 8, 8),
		context_downsample=2,
		context_attribute_halo_xyz=(0, 0, 0),
		patch_size_xyz=(4, 4, 4),
		min_input_attributes=2,
		max_input_attributes=2,
	)

	sample = dataset[0]

	assert sample['context'].shape == (2, 4, 4, 4)
	assert sample['context_valid_mask'].shape == (4, 4, 4)
	assert sample['coords']['context_compute_size_xyz'] == (8, 8, 8)
	assert sample['coords']['context_lowres_compute_size_xyz'] == (4, 4, 4)
	np.testing.assert_array_equal(
		sample['context'][:, sample['context_valid_mask']],
		np.full((2, int(sample['context_valid_mask'].sum())), 3.0, dtype=np.float32),
	)
	np.testing.assert_array_equal(
		sample['context'][:, np.logical_not(sample['context_valid_mask'])],
		np.zeros(
			(2, int(np.logical_not(sample['context_valid_mask']).sum())),
			dtype=np.float32,
		),
	)


def test_pretrain_dataset_is_deterministic_for_seed_and_index(tmp_path: Path) -> None:
	manifest = _write_manifest(tmp_path / 'survey-a', MVP_ATTRIBUTE_REGISTRY.names)
	first = _dataset(manifest, samples_per_epoch=3)
	second = _dataset(manifest, samples_per_epoch=3)
	first.set_epoch(0)
	second.set_epoch(0)

	first_sample = first[1]
	second_sample = second[1]

	for key in (
		'x',
		'target',
		'attribute_ids',
		'spatial_mask',
		'visible_spatial_mask',
		'attribute_input_mask',
		'attribute_target_mask',
		'dropped_attribute_mask',
		'target_valid',
		'local_valid_mask',
	):
		np.testing.assert_array_equal(first_sample[key], second_sample[key])
	assert first_sample['coords'] == second_sample['coords']


def test_pretrain_dataset_epoch_zero_repeated_calls_are_reproducible(
	tmp_path: Path,
) -> None:
	manifest = _write_manifest(tmp_path / 'survey-a', MVP_ATTRIBUTE_REGISTRY.names)
	dataset = _dataset(manifest, samples_per_epoch=3)
	dataset.set_epoch(0)

	first_sample = dataset[0]
	second_sample = dataset[0]

	for key in (
		'x',
		'target',
		'attribute_ids',
		'spatial_mask',
		'visible_spatial_mask',
		'attribute_input_mask',
		'attribute_target_mask',
		'dropped_attribute_mask',
		'target_valid',
		'local_valid_mask',
	):
		np.testing.assert_array_equal(first_sample[key], second_sample[key])
	assert first_sample['coords'] == second_sample['coords']


def test_pretrain_dataset_epoch_changes_stochastic_sample(tmp_path: Path) -> None:
	manifest = _write_manifest(
		tmp_path / 'survey-a',
		MVP_ATTRIBUTE_REGISTRY.names,
		shape_xyz=(12, 12, 12),
	)
	dataset = _dataset(
		manifest,
		patch_size_xyz=(2, 2, 2),
		max_input_attributes=8,
		samples_per_epoch=3,
	)

	dataset.set_epoch(0)
	epoch_zero = dataset[0]
	dataset.set_epoch(1)
	epoch_one = dataset[0]

	changed = (
		epoch_zero['coords']['local_start_xyz']
		!= epoch_one['coords']['local_start_xyz']
		or not np.array_equal(epoch_zero['spatial_mask'], epoch_one['spatial_mask'])
		or not np.array_equal(epoch_zero['attribute_ids'], epoch_one['attribute_ids'])
	)
	assert changed


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
		patch_size_xyz=(2, 2, 2),
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
