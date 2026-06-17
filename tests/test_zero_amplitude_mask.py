from __future__ import annotations

import numpy as np

from seis_attr_ssl.attributes.on_the_fly import (
	AttributeGenerationConfig,
	attribute_generation_config_from_mapping,
)
from seis_attr_ssl.attributes.zero_mask import (
	ZeroAmplitudeMaskConfig,
	compute_zero_amplitude_invalid_mask,
	detect_all_zero_traces,
	detect_all_zero_z_samples,
)


def test_detect_one_all_zero_z_plane() -> None:
	amplitude = np.ones((3, 4, 5), dtype=np.float32)
	amplitude[:, :, 2] = 0.0

	zero_z = detect_all_zero_z_samples(amplitude)

	np.testing.assert_array_equal(
		zero_z,
		np.array([False, False, True, False, False]),
	)


def test_detect_one_all_zero_trace() -> None:
	amplitude = np.ones((3, 4, 5), dtype=np.float32)
	amplitude[1, 2, :] = 0.0

	zero_traces = detect_all_zero_traces(amplitude)

	expected = np.zeros((3, 4), dtype=bool)
	expected[1, 2] = True
	np.testing.assert_array_equal(zero_traces, expected)


def test_padding_only_invalid_voxels_do_not_seed_zero_dilation() -> None:
	amplitude = np.ones((4, 3, 5), dtype=np.float32)
	amplitude[0, :, :] = 0.0
	valid_mask = np.ones_like(amplitude, dtype=bool)
	valid_mask[0, :, :] = False

	zero_traces = detect_all_zero_traces(amplitude, valid_mask=valid_mask)
	invalid = compute_zero_amplitude_invalid_mask(
		amplitude,
		valid_mask=valid_mask,
		config=ZeroAmplitudeMaskConfig(
			z_sample_influence_radius=0,
			xy_trace_influence_radius=1,
		),
	)

	assert not zero_traces.any()
	assert invalid[0, :, :].all()
	assert not invalid[1:, :, :].any()


def test_z_plane_influence_radius_expands_along_z_only() -> None:
	amplitude = np.ones((3, 4, 7), dtype=np.float32)
	amplitude[:, :, 3] = 0.0

	invalid = compute_zero_amplitude_invalid_mask(
		amplitude,
		config=ZeroAmplitudeMaskConfig(
			z_sample_influence_radius=1,
			xy_trace_influence_radius=0,
		),
	)

	expected = np.zeros_like(amplitude, dtype=bool)
	expected[:, :, 2:5] = True
	np.testing.assert_array_equal(invalid, expected)


def test_zero_trace_influence_radius_expands_in_xy_over_full_z() -> None:
	amplitude = np.ones((5, 5, 4), dtype=np.float32)
	amplitude[2, 2, :] = 0.0

	invalid = compute_zero_amplitude_invalid_mask(
		amplitude,
		config=ZeroAmplitudeMaskConfig(
			z_sample_influence_radius=0,
			xy_trace_influence_radius=1,
		),
	)

	expected = np.zeros_like(amplitude, dtype=bool)
	expected[1:4, 1:4, :] = True
	np.testing.assert_array_equal(invalid, expected)


def test_disabled_zero_mask_returns_all_false_invalid_mask() -> None:
	amplitude = np.zeros((2, 3, 4), dtype=np.float32)

	invalid = compute_zero_amplitude_invalid_mask(
		amplitude,
		config=ZeroAmplitudeMaskConfig(enabled=False),
	)

	assert invalid.dtype == np.bool_
	assert invalid.shape == amplitude.shape
	assert not invalid.any()


def test_nonzero_data_returns_all_false_invalid_mask() -> None:
	amplitude = np.ones((2, 3, 4), dtype=np.float32)

	invalid = compute_zero_amplitude_invalid_mask(
		amplitude,
		config=ZeroAmplitudeMaskConfig(
			z_sample_influence_radius=1,
			xy_trace_influence_radius=1,
		),
	)

	assert not invalid.any()


def test_attribute_generation_config_parses_nested_zero_mask_mapping() -> None:
	config = attribute_generation_config_from_mapping(
		{
			'zero_mask': {
				'enabled': False,
				'zero_atol': 1.0e-5,
				'z_sample_influence_radius': 3,
				'xy_trace_influence_radius': 2,
				'z_trace_influence_radius': 1,
			},
		},
	)

	assert config == AttributeGenerationConfig(
		zero_mask=ZeroAmplitudeMaskConfig(
			enabled=False,
			zero_atol=1.0e-5,
			z_sample_influence_radius=3,
			xy_trace_influence_radius=2,
			z_trace_influence_radius=1,
		),
	)
