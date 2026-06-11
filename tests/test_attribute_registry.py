from __future__ import annotations

from pathlib import Path

import pytest

from seis_attr_ssl.attributes import MVP_ATTRIBUTE_REGISTRY
from seis_attr_ssl.config import load_config, validate_config
from seis_attr_ssl.config.schema import EXPECTED_ATTRIBUTE_GROUPS, EXPECTED_ATTRIBUTES


def test_mvp_registry_has_fixed_length_and_order() -> None:
	assert len(MVP_ATTRIBUTE_REGISTRY.specs) == 10
	assert MVP_ATTRIBUTE_REGISTRY.names == tuple(EXPECTED_ATTRIBUTES)
	assert MVP_ATTRIBUTE_REGISTRY.name_to_id('amplitude_norm') == 0
	assert MVP_ATTRIBUTE_REGISTRY.id_to_name(9) == 'glcm_homogeneity'


def test_mvp_registry_spectral_group_ids() -> None:
	assert MVP_ATTRIBUTE_REGISTRY.ids_for_group('spectral') == (4, 5, 6)


def test_invalid_name_lookup_raises_clear_key_error() -> None:
	with pytest.raises(KeyError, match='unknown attribute name'):
		MVP_ATTRIBUTE_REGISTRY.name_to_id('not_an_attribute')


def test_invalid_id_lookup_raises_clear_key_error() -> None:
	with pytest.raises(KeyError, match='unknown attribute ID'):
		MVP_ATTRIBUTE_REGISTRY.id_to_name(10)


def test_reordered_attribute_names_raise_clear_value_error() -> None:
	reordered = (
		'phase_sin',
		'amplitude_norm',
		*MVP_ATTRIBUTE_REGISTRY.names[2:],
	)

	with pytest.raises(ValueError, match='attribute names'):
		MVP_ATTRIBUTE_REGISTRY.validate_names(reordered)


def test_mvp_registry_matches_config_schema_constants() -> None:
	assert MVP_ATTRIBUTE_REGISTRY.names == tuple(EXPECTED_ATTRIBUTES)
	assert dict(MVP_ATTRIBUTE_REGISTRY.groups) == EXPECTED_ATTRIBUTE_GROUPS

	for name in EXPECTED_ATTRIBUTES:
		spec = MVP_ATTRIBUTE_REGISTRY.spec(name)
		assert spec.id == EXPECTED_ATTRIBUTES.index(name)
		assert spec.group == EXPECTED_ATTRIBUTE_GROUPS[name]


def test_mvp_config_still_validates() -> None:
	validate_config(load_config(Path('proc/configs/mvp_mae.yaml')))
