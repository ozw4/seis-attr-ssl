from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from seis_attr_ssl.attributes import MVP_ATTRIBUTE_REGISTRY
from seis_attr_ssl.data import (
	AttributeGenerationResult,
	BaseSeismicVolumeRecord,
	SurveyManifest,
	center_trim_attribute_result,
	generate_mvp_attributes,
	generate_mvp_attributes_for_payload,
)
from seis_attr_ssl.data.pretrain_dataset import NopimsAttributePretrainDataset


def test_generate_mvp_attributes_constant_cube_contract() -> None:
	amp = np.ones((3, 4, 8), dtype=np.float32)

	result = generate_mvp_attributes(amp)

	assert result.attributes.shape == (10, *amp.shape)
	assert result.attributes.dtype == np.float32
	assert result.attribute_valid.dtype == np.bool_
	assert result.voxel_valid_mask.shape == amp.shape
	np.testing.assert_array_equal(result.attribute_valid, np.ones(10, dtype=bool))
	np.testing.assert_allclose(result.attributes[0], amp)
	np.testing.assert_allclose(result.attributes[1], 0.0, atol=1.0e-6)
	np.testing.assert_allclose(result.attributes[2], 1.0, atol=1.0e-6)
	np.testing.assert_allclose(result.attributes[3], 0.0, atol=1.0e-6)
	np.testing.assert_allclose(result.attributes[4], 1.0, atol=1.0e-5)
	np.testing.assert_allclose(result.attributes[7], 1.0, atol=1.0e-6)
	np.testing.assert_allclose(result.attributes[8], 0.0, atol=1.0e-6)
	np.testing.assert_allclose(result.attributes[9], 1.0, atol=1.0e-6)
	assert np.isfinite(result.attributes).all()


def test_generate_mvp_attributes_ramp_along_z_is_finite() -> None:
	z = np.linspace(-1.0, 1.0, 9, dtype=np.float32)
	amp = np.broadcast_to(z.reshape(1, 1, -1), (4, 3, z.size)).copy()

	result = generate_mvp_attributes(amp)

	np.testing.assert_allclose(result.attributes[0], amp)
	assert np.isfinite(result.attributes).all()
	assert (result.attributes[3] >= 0.0).all()
	assert (result.attributes[4:7] >= 0.0).all()
	assert (result.attributes[4:7] <= 1.0).all()
	np.testing.assert_allclose(
		result.attributes[4] + result.attributes[5] + result.attributes[6],
		1.0,
		atol=1.0e-5,
	)


def test_generate_mvp_attributes_sinusoid_has_nonzero_frequency() -> None:
	z_size = 32
	z = np.arange(z_size, dtype=np.float32)
	trace = np.sin(2.0 * np.pi * 2.0 * z / z_size).astype(np.float32)
	amp = np.broadcast_to(trace.reshape(1, 1, -1), (2, 3, z_size)).copy()

	result = generate_mvp_attributes(amp)

	assert np.isfinite(result.attributes).all()
	assert float(result.attributes[3].mean()) > 0.0
	assert float(result.attributes[4].mean()) > float(result.attributes[5].mean())
	assert float(result.attributes[4].mean()) > float(result.attributes[6].mean())


def test_generate_mvp_attributes_zeroes_invalid_padded_voxels() -> None:
	amp = np.arange(4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4)
	valid_mask = np.ones_like(amp, dtype=bool)
	valid_mask[-1] = False

	result = generate_mvp_attributes(amp, valid_mask=valid_mask)

	np.testing.assert_array_equal(result.voxel_valid_mask, valid_mask)
	assert np.isfinite(result.attributes).all()
	assert not result.attributes[:, ~valid_mask].any()


def test_generate_mvp_attributes_for_payload_trims_compute_crop(monkeypatch) -> None:
	compute_shape = (160, 160, 256)
	payload_slices = (slice(16, 144), slice(16, 144), slice(64, 192))
	amp = np.broadcast_to(np.zeros((1, 1, 1), dtype=np.float32), compute_shape)
	valid_mask = np.ones(compute_shape, dtype=bool)

	def fake_generate(
		amp_norm: np.ndarray,
		*,
		valid_mask: np.ndarray | None = None,
		config: object | None = None,
	) -> AttributeGenerationResult:
		del config
		assert amp_norm.shape == compute_shape
		assert valid_mask is not None
		attributes = np.broadcast_to(
			np.arange(10, dtype=np.float32).reshape(10, 1, 1, 1),
			(10, *compute_shape),
		)
		return AttributeGenerationResult(
			attributes=attributes,
			attribute_valid=np.ones(10, dtype=bool),
			voxel_valid_mask=valid_mask,
		)

	monkeypatch.setattr(
		'seis_attr_ssl.attributes.on_the_fly.generate_mvp_attributes',
		fake_generate,
	)

	result = generate_mvp_attributes_for_payload(
		amp,
		payload_slices,
		valid_mask=valid_mask,
	)

	assert result.attributes.shape == (10, 128, 128, 128)
	assert result.attributes.dtype == np.float32
	assert result.voxel_valid_mask.shape == (128, 128, 128)
	assert result.voxel_valid_mask.dtype == np.bool_
	np.testing.assert_array_equal(result.attributes[:, 0, 0, 0], np.arange(10))


def test_generate_mvp_attributes_for_payload_trims_valid_mask() -> None:
	amp = np.arange(6 * 6 * 8, dtype=np.float32).reshape(6, 6, 8)
	valid_mask = np.ones_like(amp, dtype=bool)
	valid_mask[0] = False
	valid_mask[2, 4, 5] = False
	payload_slices = (slice(1, 5), slice(2, 6), slice(3, 7))

	result = generate_mvp_attributes_for_payload(
		amp,
		payload_slices,
		valid_mask=valid_mask,
	)

	assert result.attributes.shape == (10, 4, 4, 4)
	assert result.voxel_valid_mask.shape == (4, 4, 4)
	assert not result.voxel_valid_mask[1, 2, 2]
	assert not result.attributes[:, 1, 2, 2].any()
	np.testing.assert_array_equal(
		result.voxel_valid_mask,
		valid_mask[payload_slices],
	)


@pytest.mark.parametrize(
	'payload_slices',
	[
		(slice(0, 2), slice(0, 2)),
		(slice(None, 2), slice(0, 2), slice(0, 2)),
		(slice(0, None), slice(0, 2), slice(0, 2)),
		(slice(-1, 2), slice(0, 2), slice(0, 2)),
		(slice(0, 5), slice(0, 2), slice(0, 2)),
		(slice(1, 1), slice(0, 2), slice(0, 2)),
		(slice(0, 2, 2), slice(0, 2), slice(0, 2)),
		(0, slice(0, 2), slice(0, 2)),
	],
)
def test_center_trim_attribute_result_rejects_invalid_slices(
	payload_slices: tuple[slice, ...],
) -> None:
	result = AttributeGenerationResult(
		attributes=np.ones((10, 4, 4, 4), dtype=np.float32),
		attribute_valid=np.ones(10, dtype=bool),
		voxel_valid_mask=np.ones((4, 4, 4), dtype=bool),
	)

	with pytest.raises(ValueError, match='payload_slices_xyz'):
		center_trim_attribute_result(result, payload_slices)


def test_pretrain_dataset_generates_context_attributes_after_downsampling(
	tmp_path: Path,
	monkeypatch,
) -> None:
	manifest = _write_base_manifest(tmp_path / 'survey-a', (8, 8, 8))
	calls: list[tuple[int, int, int]] = []

	def fake_generate(
		amp_norm: np.ndarray,
		*,
		valid_mask: np.ndarray | None = None,
		config: object | None = None,
	) -> AttributeGenerationResult:
		del config
		calls.append(amp_norm.shape)
		assert amp_norm.shape == (4, 4, 4)
		mask = np.ones(amp_norm.shape, dtype=bool) if valid_mask is None else valid_mask
		attributes = np.stack(
			[
				np.full(amp_norm.shape, spec.id, dtype=np.float32)
				for spec in MVP_ATTRIBUTE_REGISTRY.specs
			],
			axis=0,
		)
		return AttributeGenerationResult(
			attributes=attributes,
			attribute_valid=np.ones(len(MVP_ATTRIBUTE_REGISTRY.specs), dtype=bool),
			voxel_valid_mask=mask,
		)

	monkeypatch.setattr(
		'seis_attr_ssl.data.pretrain_dataset.generate_mvp_attributes',
		fake_generate,
	)
	dataset = NopimsAttributePretrainDataset(
		[manifest],
		local_crop_size_xyz=(4, 4, 4),
		context_crop_size_xyz=(8, 8, 8),
		context_downsample=2,
		patch_size_xyz=(4, 4, 4),
		min_input_attributes=2,
		max_input_attributes=2,
		seed=1,
	)

	sample = dataset[0]

	assert calls == [(4, 4, 4), (4, 4, 4)]
	assert sample['context'].shape == (2, 4, 4, 4)


def _write_base_manifest(
	root: Path,
	shape_xyz: tuple[int, int, int],
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
