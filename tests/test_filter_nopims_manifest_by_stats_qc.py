from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from seis_attr_ssl.data.normalization import (
	SurveyNormalizationStats,
	write_normalization_stats,
)
from seis_attr_ssl.data.schema import (
	BaseSeismicVolumeRecord,
	SurveyManifest,
	read_manifest_json,
	write_manifest_json,
)
from tests.helpers import run_python_proc

SCRIPT = Path('proc/filter_nopims_manifest_by_stats_qc.py')


def test_filter_nopims_manifest_by_stats_qc_writes_clean_outputs(
	tmp_path: Path,
) -> None:
	root = tmp_path / 'NOPIMS'
	manifest_path = tmp_path / 'manifest.json'
	path_list = tmp_path / 'train_npy_paths.txt'
	outputs = _outputs(tmp_path)
	for survey_id in ('survey-a', 'survey-b', 'survey-c'):
		_write_volume(root, survey_id)
	_write_stats(root, 'survey-a')
	_write_stats(root, 'survey-b', iqr=1.0e-8, clip_low=-1.0e-8, clip_high=1.0e-8)
	_write_stats(root, 'survey-c')
	write_manifest_json(
		[
			_manifest(root, 'survey-c'),
			_manifest(root, 'survey-a'),
			_manifest(root, 'survey-b'),
		],
		manifest_path,
	)
	path_list.write_text(
		'# source order\nsurvey-b.npy\n\nsurvey-a.npy\nsurvey-c.npy\n',
		encoding='utf-8',
	)

	result = run_python_proc(
		SCRIPT,
		*_args(root, manifest_path, path_list, outputs),
	)

	assert result.returncode == 0, result.stderr
	assert 'normalization_qc.total_surveys: 3' in result.stdout
	assert 'normalization_qc.passed_surveys: 2' in result.stdout
	assert 'normalization_qc.excluded_surveys: 1' in result.stdout
	assert 'normalization_qc.write: true' in result.stdout
	assert outputs['excluded_surveys'].read_text(encoding='utf-8') == 'survey-b\n'
	report = json.loads(outputs['qc_json'].read_text(encoding='utf-8'))
	assert report['excluded_surveys'] == ['survey-b']
	assert report['counts']['small_iqr'] == 1
	clean_manifest = read_manifest_json(outputs['manifest'])
	assert [manifest.survey_id for manifest in clean_manifest] == [
		'survey-c',
		'survey-a',
	]
	assert outputs['path_list'].read_text(encoding='utf-8').splitlines() == [
		'survey-a.npy',
		'survey-c.npy',
	]


def test_filter_nopims_manifest_by_stats_qc_dry_run_writes_no_files(
	tmp_path: Path,
) -> None:
	root = tmp_path / 'NOPIMS'
	manifest_path = tmp_path / 'manifest.json'
	path_list = tmp_path / 'train_npy_paths.txt'
	outputs = _outputs(tmp_path)
	_write_volume(root, 'survey-a')
	_write_stats(root, 'survey-a')
	write_manifest_json([_manifest(root, 'survey-a')], manifest_path)
	path_list.write_text('survey-a.npy\n', encoding='utf-8')

	result = run_python_proc(
		SCRIPT,
		*_args(root, manifest_path, path_list, outputs),
		'--dry-run',
	)

	assert result.returncode == 0, result.stderr
	assert 'normalization_qc.write: false' in result.stdout
	assert not outputs['qc_json'].exists()
	assert not outputs['excluded_surveys'].exists()
	assert not outputs['manifest'].exists()
	assert not outputs['path_list'].exists()


def test_filter_nopims_manifest_by_stats_qc_missing_stats_excludes_survey(
	tmp_path: Path,
) -> None:
	root = tmp_path / 'NOPIMS'
	manifest_path = tmp_path / 'manifest.json'
	path_list = tmp_path / 'train_npy_paths.txt'
	outputs = _outputs(tmp_path)
	_write_volume(root, 'survey-a')
	write_manifest_json([_manifest(root, 'survey-a')], manifest_path)
	path_list.write_text('survey-a.npy\n', encoding='utf-8')

	result = run_python_proc(
		SCRIPT,
		*_args(root, manifest_path, path_list, outputs),
	)

	assert result.returncode == 0, result.stderr
	report = json.loads(outputs['qc_json'].read_text(encoding='utf-8'))
	assert report['excluded_surveys'] == ['survey-a']
	assert report['counts']['missing_stats'] == 1
	assert read_manifest_json(outputs['manifest']) == []
	assert outputs['path_list'].read_text(encoding='utf-8') == ''


def test_filter_nopims_manifest_by_stats_qc_excludes_manifest_survey_on_id_mismatch(
	tmp_path: Path,
) -> None:
	root = tmp_path / 'NOPIMS'
	manifest_path = tmp_path / 'manifest.json'
	path_list = tmp_path / 'train_npy_paths.txt'
	outputs = _outputs(tmp_path)
	for survey_id in ('survey-a', 'survey-b'):
		_write_volume(root, survey_id)
	_write_stats(
		root,
		'survey-a',
		stats_survey_id='survey-b',
		iqr=1.0e-8,
		clip_low=-1.0e-8,
		clip_high=1.0e-8,
	)
	_write_stats(root, 'survey-b')
	write_manifest_json(
		[_manifest(root, 'survey-a'), _manifest(root, 'survey-b')],
		manifest_path,
	)
	path_list.write_text('survey-a.npy\nsurvey-b.npy\n', encoding='utf-8')

	result = run_python_proc(
		SCRIPT,
		*_args(root, manifest_path, path_list, outputs),
	)

	assert result.returncode == 0, result.stderr
	report = json.loads(outputs['qc_json'].read_text(encoding='utf-8'))
	assert report['excluded_surveys'] == ['survey-a']
	assert [survey['survey_id'] for survey in report['surveys']] == [
		'survey-a',
		'survey-b',
	]
	clean_manifest = read_manifest_json(outputs['manifest'])
	assert [manifest.survey_id for manifest in clean_manifest] == ['survey-b']
	assert outputs['path_list'].read_text(encoding='utf-8') == 'survey-b.npy\n'


def test_filter_nopims_manifest_by_stats_qc_fail_if_empty(
	tmp_path: Path,
) -> None:
	root = tmp_path / 'NOPIMS'
	manifest_path = tmp_path / 'manifest.json'
	path_list = tmp_path / 'train_npy_paths.txt'
	outputs = _outputs(tmp_path)
	_write_volume(root, 'survey-a')
	write_manifest_json([_manifest(root, 'survey-a')], manifest_path)
	path_list.write_text('survey-a.npy\n', encoding='utf-8')

	result = run_python_proc(
		SCRIPT,
		*_args(root, manifest_path, path_list, outputs),
		'--fail-if-empty',
	)

	assert result.returncode != 0
	assert 'clean manifest or clean path-list is empty' in result.stderr
	assert not outputs['qc_json'].exists()
	assert not outputs['excluded_surveys'].exists()
	assert not outputs['manifest'].exists()
	assert not outputs['path_list'].exists()


def test_filter_nopims_manifest_by_stats_qc_fails_on_unregistered_path_survey(
	tmp_path: Path,
) -> None:
	root = tmp_path / 'NOPIMS'
	manifest_path = tmp_path / 'manifest.json'
	path_list = tmp_path / 'train_npy_paths.txt'
	outputs = _outputs(tmp_path)
	_write_volume(root, 'survey-a')
	_write_stats(root, 'survey-a')
	write_manifest_json([_manifest(root, 'survey-a')], manifest_path)
	path_list.write_text('survey-a.npy\nsurvey-z.npy\n', encoding='utf-8')

	result = run_python_proc(
		SCRIPT,
		*_args(root, manifest_path, path_list, outputs),
	)

	assert result.returncode != 0
	assert 'not present in manifest: survey-z' in result.stderr


def test_filter_nopims_manifest_by_stats_qc_prefers_stats_dir(
	tmp_path: Path,
) -> None:
	root = tmp_path / 'NOPIMS'
	stats_dir = tmp_path / 'stats'
	manifest_path = tmp_path / 'manifest.json'
	path_list = tmp_path / 'train_npy_paths.txt'
	outputs = _outputs(tmp_path)
	_write_volume(root, 'survey-a')
	_write_stats(root, 'survey-a', iqr=1.0e-8, clip_low=-1.0e-8, clip_high=1.0e-8)
	_write_stats(stats_dir, 'survey-a')
	write_manifest_json([_manifest(root, 'survey-a')], manifest_path)
	path_list.write_text('survey-a.npy\n', encoding='utf-8')

	result = run_python_proc(
		SCRIPT,
		*_args(root, manifest_path, path_list, outputs),
		'--stats-dir',
		stats_dir,
	)

	assert result.returncode == 0, result.stderr
	report = json.loads(outputs['qc_json'].read_text(encoding='utf-8'))
	assert report['excluded_surveys'] == []
	assert outputs['path_list'].read_text(encoding='utf-8') == 'survey-a.npy\n'


def _outputs(tmp_path: Path) -> dict[str, Path]:
	return {
		'qc_json': tmp_path / 'out' / 'qc' / 'normalization_stats_qc.json',
		'excluded_surveys': tmp_path / 'out' / 'qc' / 'excluded_surveys.txt',
		'manifest': tmp_path / 'out' / 'manifests' / 'manifest.json',
		'path_list': tmp_path / 'out' / 'splits' / 'train_npy_paths.txt',
	}


def _args(
	root: Path,
	manifest_path: Path,
	path_list: Path,
	outputs: dict[str, Path],
) -> tuple[object, ...]:
	return (
		'--manifest',
		manifest_path,
		'--input-path-list',
		path_list,
		'--nopims-root',
		root,
		'--output-qc-json',
		outputs['qc_json'],
		'--output-excluded-surveys',
		outputs['excluded_surveys'],
		'--output-manifest',
		outputs['manifest'],
		'--output-path-list',
		outputs['path_list'],
		'--iqr-min-threshold',
		'1.0e-6',
		'--norm-abs-max-threshold',
		'1.0e4',
	)


def _write_volume(root: Path, survey_id: str) -> Path:
	path = root / f'{survey_id}.npy'
	path.parent.mkdir(parents=True, exist_ok=True)
	np.save(path, np.ones((2, 2, 2), dtype=np.float32))
	return path


def _write_stats(  # noqa: PLR0913
	root: Path,
	survey_id: str,
	*,
	clip_low: float = -2.0,
	clip_high: float = 6.0,
	iqr: float = 2.0,
	stats_survey_id: str | None = None,
) -> Path:
	path = root / f'{survey_id}.normalization_stats.json'
	stats_survey_id = stats_survey_id or survey_id
	write_normalization_stats(
		SurveyNormalizationStats(
			survey_id=stats_survey_id,
			source_path=root / f'{stats_survey_id}.npy',
			grid_order=('x', 'y', 'z'),
			clip_low_percentile=0.5,
			clip_high_percentile=99.5,
			clip_low=clip_low,
			clip_high=clip_high,
			median=2.0,
			iqr=iqr,
			eps=1.0e-6,
		),
		path,
	)
	return path


def _manifest(root: Path, survey_id: str) -> SurveyManifest:
	return SurveyManifest(
		survey_id=survey_id,
		root=root,
		attribute_volumes={},
		shape_xyz=(2, 2, 2),
		base_seismic=BaseSeismicVolumeRecord(
			survey_id=survey_id,
			path=root / f'{survey_id}.npy',
			kind='dip_steered_median_filtered',
			shape_xyz=(2, 2, 2),
			dtype='float32',
			grid_order=('x', 'y', 'z'),
			normalization_stats_path=Path(
				f'{survey_id}.normalization_stats.json',
			),
		),
	)
