from __future__ import annotations

from pathlib import Path

import pytest

from seis_attr_ssl.attributes import MVP_ATTRIBUTE_REGISTRY
from seis_attr_ssl.data import (
	GRID_ORDER_XYZ,
	AttributeVolumeRecord,
	SurveyManifest,
	read_manifest_json,
	survey_manifest_from_dict,
	survey_manifest_to_dict,
	write_manifest_json,
)

SURVEY_ROOT = Path('nopims/survey-a')


def _record(
	attribute_name: str,
	shape_xyz: tuple[int, int, int] = (8, 9, 10),
) -> AttributeVolumeRecord:
	return AttributeVolumeRecord(
		survey_id='survey-a',
		attribute_name=attribute_name,
		path=Path('attrs') / f'{attribute_name}.npy',
		shape_xyz=shape_xyz,
		dtype='float32',
		grid_order=GRID_ORDER_XYZ,
		is_memmap_safe=True,
	)


def _manifest(*attribute_names: str) -> SurveyManifest:
	return SurveyManifest(
		survey_id='survey-a',
		root=SURVEY_ROOT,
		attribute_volumes={name: _record(name) for name in attribute_names},
		shape_xyz=(8, 9, 10),
	)


def test_manifest_json_round_trip_preserves_records(tmp_path: Path) -> None:
	manifest = _manifest('amplitude_norm', 'phase_sin')
	path = tmp_path / 'manifests.json'

	write_manifest_json([manifest], path)

	loaded = read_manifest_json(path)
	assert loaded == [manifest]
	assert loaded[0].root == SURVEY_ROOT
	assert loaded[0].get_attribute('phase_sin').path == Path('attrs/phase_sin.npy')
	assert loaded[0].get_attribute('phase_sin').shape_xyz == (8, 9, 10)
	assert loaded[0].get_attribute('phase_sin').dtype == 'float32'


def test_manifest_dict_round_trip_preserves_grid_order() -> None:
	manifest = _manifest('amplitude_norm')

	loaded = survey_manifest_from_dict(survey_manifest_to_dict(manifest))

	assert loaded.grid_order == ('x', 'y', 'z')
	assert loaded.get_attribute('amplitude_norm').grid_order == ('x', 'y', 'z')


def test_missing_attributes_are_reported_in_registry_order() -> None:
	manifest = _manifest('phase_cos', 'amplitude_norm')
	expected = tuple(MVP_ATTRIBUTE_REGISTRY.names[1:2]) + tuple(
		MVP_ATTRIBUTE_REGISTRY.names[3:],
	)

	assert manifest.has_all_mvp_attributes() is False
	assert manifest.missing_attributes() == expected


def test_manifest_with_all_mvp_attributes_reports_complete() -> None:
	manifest = _manifest(*MVP_ATTRIBUTE_REGISTRY.names)

	assert manifest.has_all_mvp_attributes() is True
	assert manifest.missing_attributes() == ()


def test_validate_consistent_shapes_fails_clearly() -> None:
	manifest = SurveyManifest(
		survey_id='survey-a',
		root=SURVEY_ROOT,
		attribute_volumes={'amplitude_norm': _record('amplitude_norm', (1, 9, 10))},
		shape_xyz=(8, 9, 10),
	)

	with pytest.raises(ValueError, match=r'amplitude_norm.*shape'):
		manifest.validate_consistent_shapes()


def test_get_attribute_success_and_failure() -> None:
	manifest = _manifest('amplitude_norm')

	assert manifest.get_attribute('amplitude_norm').attribute_name == 'amplitude_norm'
	with pytest.raises(KeyError, match='unknown manifest attribute'):
		manifest.get_attribute('phase_sin')


def test_grid_order_default_is_xyz() -> None:
	manifest = _manifest('amplitude_norm')

	assert GRID_ORDER_XYZ == ('x', 'y', 'z')
	assert manifest.grid_order == ('x', 'y', 'z')
