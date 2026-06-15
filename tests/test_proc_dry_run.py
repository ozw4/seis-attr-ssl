from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers import run_python_proc

PROC_SCRIPTS = (
	Path('proc/build_nopims_manifests.py'),
	Path('proc/prepare_normalization_stats.py'),
	Path('proc/prepare_nopims_normalization_stats.py'),
	Path('proc/generate_attributes.py'),
	Path('proc/train_mae.py'),
	Path('proc/train_dense_adapt.py'),
	Path('proc/train_finetune.py'),
	Path('proc/eval_f3.py'),
	Path('proc/infer_volume.py'),
)


@pytest.mark.parametrize('script_path', PROC_SCRIPTS)
def test_proc_script_help_exits_zero(script_path: Path) -> None:
	result = run_python_proc(script_path, '--help')

	assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('script_path', PROC_SCRIPTS)
def test_proc_script_dry_run_exits_zero_and_prints_stage(
	script_path: Path,
) -> None:
	result = run_python_proc(script_path, '--dry-run')

	assert result.returncode == 0, result.stderr
	assert 'stage:' in result.stdout


def test_train_mae_dry_run_prints_masking_settings() -> None:
	result = run_python_proc(Path('proc/train_mae.py'), '--dry-run')

	assert result.returncode == 0, result.stderr
	assert 'masking.spatial_mask_ratio: 0.75' in result.stdout
	assert 'masking.spatial_mask_mode: block' in result.stdout
	assert 'masking.block_size_tokens: 2, 2, 2' in result.stdout
	assert 'train.samples_per_epoch: 10000' in result.stdout
	assert 'train.num_workers: 4' in result.stdout
	assert 'train.shuffle: true' in result.stdout


@pytest.mark.parametrize(
	('args', 'expected_stderr'),
	[
		(('--max-steps', '-1'), 'train.max_steps must be positive'),
		(('--output-root', 'F3/run'), 'F3 paths are not allowed'),
	],
)
def test_train_mae_dry_run_validates_cli_overrides(
	args: tuple[str, ...],
	expected_stderr: str,
) -> None:
	result = run_python_proc(Path('proc/train_mae.py'), '--dry-run', *args)

	assert result.returncode != 0
	assert expected_stderr in result.stderr
	assert 'stage:' not in result.stdout


def test_prepare_nopims_normalization_stats_dry_run_missing_manifest_is_actionable(
	tmp_path: Path,
) -> None:
	config_path = tmp_path / 'mae.yaml'
	config_path.write_text(
		"""
project:
  name: SeisAttrSSL
  package: seis_attr_ssl
paths:
  nopims_root: /tmp/NOPIMS/
  output_root: runs
manifests:
  train: /tmp/NOPIMS/manifests/missing.json
data:
  grid_order: [x, y, z]
  volume_format: npy_memmap
  base_seismic_kind: dip_steered_median_filtered
  attribute_mode: on_the_fly
  local_crop_size: [128, 128, 128]
  context_crop_size: [256, 256, 512]
  context_downsample: [2, 2, 4]
  local_attribute_halo: [16, 16, 64]
  context_attribute_halo: [8, 8, 16]
  require_full_halo_inside_volume: true
  use_context: true
attributes:
  names:
    - amplitude_norm
    - phase_sin
    - phase_cos
    - instantaneous_frequency
    - spectral_low_ratio
    - spectral_mid_ratio
    - spectral_high_ratio
    - coherence
    - glcm_contrast
    - glcm_homogeneity
  groups:
    amplitude_norm: waveform
    phase_sin: phase
    phase_cos: phase
    instantaneous_frequency: frequency
    spectral_low_ratio: spectral
    spectral_mid_ratio: spectral
    spectral_high_ratio: spectral
    coherence: discontinuity
    glcm_contrast: texture
    glcm_homogeneity: texture
normalization:
  pre_attribute:
    clipping_percentiles: [0.5, 99.5]
    center: median
    scale: iqr
    epsilon: 1.0e-6
    smooth_time_depth_trend_correction: false
    trace_wise_agc: false
    patch_wise_zscore: false
stage: pretrain_mae
masking:
  spatial_mask_ratio: 0.75
  spatial_mask_mode: block
  block_size_tokens: [2, 2, 2]
  min_input_attributes: 4
  max_input_attributes: 10
  attribute_dropout_prob: 0.30
  group_dropout_prob: 0.20
model:
  patch_size: [8, 8, 8]
train:
  batch_size: 1
""",
		encoding='utf-8',
	)

	result = run_python_proc(
		Path('proc/prepare_nopims_normalization_stats.py'),
		'--config',
		config_path,
		'--dry-run',
	)

	assert result.returncode == 0, result.stderr
	assert 'normalization_stats.manifest_exists: false' in result.stdout
	assert 'normalization_stats.compute: skipped' in result.stdout
	assert 'proc/build_nopims_manifests.py' in result.stdout
	assert 'Traceback' not in result.stderr


def test_prepare_nopims_normalization_stats_missing_manifest_fails_actionably(
	tmp_path: Path,
) -> None:
	config_path = tmp_path / 'mae.yaml'
	config_path.write_text(
		Path('proc/configs/mvp_mae.yaml').read_text(encoding='utf-8').replace(
			'/home/dcuser/data/NOPIMS/manifests/nopims_base_seismic_manifests.json',
			str(tmp_path / 'missing.json'),
		),
		encoding='utf-8',
	)

	result = run_python_proc(
		Path('proc/prepare_nopims_normalization_stats.py'),
		'--config',
		config_path,
	)

	assert result.returncode != 0
	assert 'manifests.train does not exist' in result.stderr
	assert 'proc/build_nopims_manifests.py' in result.stderr
