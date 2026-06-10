from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from seis_attr_ssl.config import load_config, validate_config


def test_loads_valid_mvp_config() -> None:
	cfg = load_config(Path('proc/configs/mvp_mae.yaml'))

	validate_config(cfg)

	assert cfg['project']['package'] == 'seis_attr_ssl'
	assert cfg['paths']['nopims_root'] == '/home/dcuser/data/NOPIMS/'
	assert cfg['data']['grid_order'] == ['x', 'y', 'z']
	assert cfg['data']['local_crop_size'] == [128, 128, 128]
	assert len(cfg['attributes']['names']) == 10


def test_load_config_applies_nopims_root_default(tmp_path: Path) -> None:
	config_path = tmp_path / 'minimal.yaml'
	config_path.write_text(
		"""
project:
  name: SeisAttrSSL
  package: seis_attr_ssl
paths:
  output_root: runs
data:
  grid_order: [x, y, z]
  volume_format: npy_memmap
  base_seismic_kind: dip_steered_median_filtered
  local_crop_size: [128, 128, 128]
  context_crop_size: [512, 512, 512]
  context_downsample: 4
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
""",
		encoding='utf-8',
	)

	cfg = load_config(config_path)

	assert cfg['paths']['nopims_root'] == '/home/dcuser/data/NOPIMS/'
	validate_config(cfg)


def test_explicit_custom_nopims_root_is_allowed() -> None:
	cfg = _valid_config()
	cfg['paths']['nopims_root'] = '/mnt/nopims/'

	validate_config(cfg)

	assert cfg['paths']['nopims_root'] == '/mnt/nopims/'


def test_invalid_grid_order_raises_clear_value_error() -> None:
	cfg = _valid_config()
	cfg['data']['grid_order'] = ['z', 'y', 'x']

	with pytest.raises(ValueError, match='data\\.grid_order'):
		validate_config(cfg)


def test_invalid_crop_downsample_combination_raises_clear_value_error() -> None:
	cfg = _valid_config()
	cfg['data']['context_crop_size'] = [256, 512, 512]

	with pytest.raises(ValueError, match='data\\.context_crop_size'):
		validate_config(cfg)


def test_missing_attribute_group_raises_clear_value_error() -> None:
	cfg = _valid_config()
	del cfg['attributes']['groups']['coherence']

	with pytest.raises(ValueError, match='attributes\\.groups'):
		validate_config(cfg)


def test_f3_pretraining_setting_is_rejected() -> None:
	cfg = _valid_config()
	cfg['data']['f3_root'] = 'f3'

	with pytest.raises(ValueError, match='F3 settings are not allowed'):
		validate_config(cfg)


def _valid_config() -> dict:
	return deepcopy(load_config(Path('proc/configs/mvp_mae.yaml')))
