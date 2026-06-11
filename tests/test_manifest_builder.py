from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from seis_attr_ssl.attributes import MVP_ATTRIBUTE_REGISTRY
from seis_attr_ssl.config import load_config
from seis_attr_ssl.data import (
	build_nopims_manifests,
	read_manifest_json,
	scan_nopims_manifests,
)
from tests.helpers import run_python_proc

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_volume(
	root: Path,
	survey_id: str,
	relative_path: str,
	shape: tuple[int, int, int] = (4, 5, 6),
	dtype: str = 'float32',
) -> Path:
	path = root / survey_id / relative_path
	path.parent.mkdir(parents=True, exist_ok=True)
	np.save(path, np.zeros(shape, dtype=dtype))
	return path


def test_build_nopims_manifests_writes_ordered_manifest(tmp_path: Path) -> None:
	nopims_root = tmp_path / 'NOPIMS'
	_write_volume(nopims_root, 'survey_b', 'attributes/amplitude_norm.npy')
	_write_volume(nopims_root, 'survey_a', 'attributes/phase_sin.npy')
	_write_volume(nopims_root, 'survey_a', 'attributes/amplitude_norm.npy')
	output_path = tmp_path / 'manifests' / 'nopims_manifests.json'

	manifests = build_nopims_manifests(
		nopims_root=nopims_root,
		output_path=output_path,
		scan_pattern='**/*.npy',
	)

	assert [manifest.survey_id for manifest in manifests] == ['survey_a', 'survey_b']
	assert output_path.is_file()
	loaded = read_manifest_json(output_path)
	assert loaded == manifests
	assert list(loaded[0].attribute_volumes) == ['amplitude_norm', 'phase_sin']
	record = loaded[0].get_attribute('amplitude_norm')
	assert record.shape_xyz == (4, 5, 6)
	assert record.dtype == 'float32'


def test_scan_supports_stem_and_parent_attribute_patterns(tmp_path: Path) -> None:
	nopims_root = tmp_path / 'NOPIMS'
	_write_volume(nopims_root, 'survey_a', 'anything/coherence.npy')
	_write_volume(nopims_root, 'survey_a', 'phase_cos/volume.npy')

	result = scan_nopims_manifests(nopims_root, '**/*.npy')
	manifest = result.manifests[0]

	assert list(manifest.attribute_volumes) == ['phase_cos', 'coherence']
	assert manifest.get_attribute('phase_cos').path.name == 'volume.npy'


def test_scan_summary_reports_missing_and_unknown_attributes(tmp_path: Path) -> None:
	nopims_root = tmp_path / 'NOPIMS'
	_write_volume(nopims_root, 'survey_a', 'attributes/amplitude_norm.npy')
	_write_volume(nopims_root, 'survey_a', 'attributes/not_in_registry.npy')

	result = scan_nopims_manifests(nopims_root, '**/*.npy')
	summary = result.summary()

	assert summary.survey_count == 1
	assert summary.attribute_volume_count == 1
	assert summary.unknown_attribute_counts == {'not_in_registry': 1}
	assert summary.missing_attributes_by_survey['survey_a'] == tuple(
		MVP_ATTRIBUTE_REGISTRY.names[1:],
	)


def test_shape_mismatch_raises_clear_error(tmp_path: Path) -> None:
	nopims_root = tmp_path / 'NOPIMS'
	_write_volume(nopims_root, 'survey_a', 'attributes/amplitude_norm.npy')
	_write_volume(
		nopims_root,
		'survey_a',
		'attributes/phase_sin.npy',
		shape=(4, 5, 7),
	)

	with pytest.raises(ValueError, match=r'phase_sin.*shape'):
		scan_nopims_manifests(nopims_root, '**/*.npy')


def test_require_all_attributes_raises_for_incomplete_survey(tmp_path: Path) -> None:
	nopims_root = tmp_path / 'NOPIMS'
	_write_volume(nopims_root, 'survey_a', 'attributes/amplitude_norm.npy')

	with pytest.raises(ValueError, match='missing required attributes'):
		build_nopims_manifests(
			nopims_root=nopims_root,
			output_path=tmp_path / 'manifest.json',
			scan_pattern='**/*.npy',
			require_all_attributes=True,
		)


def test_build_nopims_manifests_cli_writes_json(tmp_path: Path) -> None:
	nopims_root = tmp_path / 'NOPIMS'
	_write_volume(nopims_root, 'survey_a', 'attributes/amplitude_norm.npy')
	output_dir = tmp_path / 'manifests'
	config = load_config(PROJECT_ROOT / 'proc/configs/build_nopims_manifests.yaml')
	config['paths']['nopims_root'] = str(nopims_root)
	config['manifest']['output_dir'] = str(output_dir)
	config_path = tmp_path / 'build_nopims_manifests.yaml'
	config_path.write_text(yaml.safe_dump(config), encoding='utf-8')

	result = run_python_proc(
		Path('proc/build_nopims_manifests.py'),
		'--config',
		config_path,
	)

	assert result.returncode == 0, result.stderr
	output_path = output_dir / 'nopims_manifests.json'
	assert output_path.is_file()
	assert read_manifest_json(output_path)[0].survey_id == 'survey_a'
	assert 'manifest.missing_attributes.survey_a:' in result.stdout
