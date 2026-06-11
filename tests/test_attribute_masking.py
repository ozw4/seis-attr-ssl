from __future__ import annotations

import numpy as np
import pytest

from seis_attr_ssl.attributes import MVP_ATTRIBUTE_REGISTRY
from seis_attr_ssl.masking import (
	build_attribute_group_index,
	registry_attribute_ids,
	sample_attribute_input_mask,
)


def _sample(  # noqa: PLR0913
	available_attribute_ids: list[int] | tuple[int, ...] | None = None,
	*,
	min_input_attributes: int = 4,
	max_input_attributes: int = 10,
	attribute_dropout_prob: float = 0.0,
	group_dropout_prob: float = 0.0,
	seed: int = 123,
) -> np.ndarray:
	if available_attribute_ids is None:
		available_attribute_ids = tuple(range(10))
	return sample_attribute_input_mask(
		available_attribute_ids,
		registry_attribute_ids(),
		MVP_ATTRIBUTE_REGISTRY.groups,
		min_input_attributes,
		max_input_attributes,
		attribute_dropout_prob,
		group_dropout_prob,
		np.random.default_rng(seed),
	)


def test_build_attribute_group_index_uses_default_mvp_groups() -> None:
	group_index = build_attribute_group_index(
		MVP_ATTRIBUTE_REGISTRY.names,
		MVP_ATTRIBUTE_REGISTRY.groups,
	)

	assert group_index == {
		'waveform': (0,),
		'phase': (1, 2),
		'frequency': (3,),
		'spectral': (4, 5, 6),
		'discontinuity': (7,),
		'texture': (8, 9),
	}


def test_sample_attribute_input_mask_includes_amplitude_by_default() -> None:
	mask = _sample(max_input_attributes=4, seed=7)

	assert mask.dtype == np.bool_
	assert mask.shape == (len(MVP_ATTRIBUTE_REGISTRY.specs),)
	assert bool(mask[MVP_ATTRIBUTE_REGISTRY.name_to_id('amplitude_norm')])
	assert int(mask.sum()) == 4


def test_group_dropout_can_drop_spectral_group_as_unit() -> None:
	mask = _sample(
		min_input_attributes=4,
		max_input_attributes=10,
		group_dropout_prob=0.999999,
		seed=11,
	)

	spectral_ids = MVP_ATTRIBUTE_REGISTRY.ids_for_group('spectral')
	assert not np.any(mask[list(spectral_ids)])
	assert int(mask.sum()) >= 4


def test_attribute_input_mask_respects_min_and_max_counts() -> None:
	for seed in range(10):
		mask = _sample(
			min_input_attributes=3,
			max_input_attributes=5,
			attribute_dropout_prob=0.75,
			group_dropout_prob=0.25,
			seed=seed,
		)

		assert 3 <= int(mask.sum()) <= 5


def test_attribute_input_mask_is_deterministic_for_same_seed() -> None:
	first = _sample(
		attribute_dropout_prob=0.5,
		group_dropout_prob=0.25,
		seed=99,
	)
	second = _sample(
		attribute_dropout_prob=0.5,
		group_dropout_prob=0.25,
		seed=99,
	)

	np.testing.assert_array_equal(first, second)


def test_unavailable_attributes_are_never_selected() -> None:
	available = (0, 1, 2, 7, 8)
	mask = _sample(
		available,
		min_input_attributes=2,
		max_input_attributes=5,
		attribute_dropout_prob=0.4,
		group_dropout_prob=0.4,
		seed=33,
	)

	assert set(np.flatnonzero(mask)).issubset(available)


@pytest.mark.parametrize(
	('attribute_dropout_prob', 'group_dropout_prob'),
	[(-0.1, 0.0), (1.0, 0.0), (0.0, -0.1), (0.0, 1.0)],
)
def test_invalid_dropout_probabilities_raise(
	attribute_dropout_prob: float,
	group_dropout_prob: float,
) -> None:
	with pytest.raises(ValueError, match='must be in \\[0, 1\\)'):
		_sample(
			attribute_dropout_prob=attribute_dropout_prob,
			group_dropout_prob=group_dropout_prob,
		)


def test_missing_amplitude_norm_raises() -> None:
	with pytest.raises(ValueError, match='amplitude_norm'):
		_sample(
			available_attribute_ids=(1, 2, 3, 4),
			min_input_attributes=2,
			max_input_attributes=4,
		)


def test_unknown_attribute_ids_raise() -> None:
	with pytest.raises(ValueError, match='unknown target attribute ID'):
		_sample(available_attribute_ids=(0, 1, 10))


def test_unknown_group_mapping_raises() -> None:
	groups = dict(MVP_ATTRIBUTE_REGISTRY.groups)
	groups.pop('coherence')

	with pytest.raises(ValueError, match='attribute_groups'):
		sample_attribute_input_mask(
			tuple(range(10)),
			registry_attribute_ids(),
			groups,
			4,
			10,
			0.0,
			0.0,
			np.random.default_rng(1),
		)
