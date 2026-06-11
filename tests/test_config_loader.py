from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from seis_attr_ssl.config import load_config, validate_config
from tests.helpers import run_python_proc

DEFAULT_CONFIGS = [
	Path('proc/configs/build_nopims_manifests.yaml'),
	Path('proc/configs/mvp_prepare_stats.yaml'),
	Path('proc/configs/generate_attributes_nopims.yaml'),
	Path('proc/configs/mvp_mae.yaml'),
	Path('proc/configs/mvp_dense_adapt.yaml'),
	Path('proc/configs/mvp_finetune_f3.yaml'),
	Path('proc/configs/mvp_eval_f3.yaml'),
	Path('proc/configs/mvp_infer_volume.yaml'),
]


def test_loads_valid_mvp_config() -> None:
	cfg = load_config(Path('proc/configs/mvp_mae.yaml'))

	validate_config(cfg)

	assert cfg['stage'] == 'pretrain_mae'
	assert cfg['project']['package'] == 'seis_attr_ssl'
	assert cfg['paths']['nopims_root'] == '/home/dcuser/data/NOPIMS/'
	assert (
		cfg['manifests']['train']
		== '/home/dcuser/data/NOPIMS/manifests/nopims_base_seismic_manifests.json'
	)
	assert cfg['data']['grid_order'] == ['x', 'y', 'z']
	assert cfg['data']['local_crop_size'] == [128, 128, 128]
	assert len(cfg['attributes']['names']) == 10
	assert cfg['train']['samples_per_epoch'] == 10000
	assert cfg['train']['num_workers'] == 4
	assert cfg['train']['shuffle'] is True


@pytest.mark.parametrize('config_path', DEFAULT_CONFIGS)
def test_default_mvp_configs_load_and_validate(config_path: Path) -> None:
	cfg = load_config(config_path)

	validate_config(cfg)

	assert cfg['paths']['nopims_root'] == '/home/dcuser/data/NOPIMS/'
	assert cfg['data']['grid_order'] == ['x', 'y', 'z']
	assert (
		cfg['normalization']['pre_attribute'][
			'smooth_time_depth_trend_correction'
		]
		is False
	)


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
stage: pretrain_mae
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


def test_invalid_base_seismic_path_raises_clear_value_error() -> None:
	cfg = _valid_config()
	cfg['data']['base_seismic_path'] = '/home/dcuser/data/NOPIMS/survey/seismic.dat'

	with pytest.raises(ValueError, match='data\\.base_seismic_path'):
		validate_config(cfg)


def test_base_seismic_kind_is_not_required_for_f3_stage() -> None:
	cfg = load_config(Path('proc/configs/mvp_finetune_f3.yaml'))
	del cfg['data']['base_seismic_kind']

	validate_config(cfg)


def test_base_seismic_kind_is_required_for_pretraining_stage() -> None:
	cfg = _valid_config()
	del cfg['data']['base_seismic_kind']

	with pytest.raises(ValueError, match='data\\.base_seismic_kind'):
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


def test_mvp_infer_volume_config_loads_and_validates() -> None:
	cfg = load_config(Path('proc/configs/mvp_infer_volume.yaml'))

	validate_config(cfg)

	assert cfg['stage'] == 'infer_volume'
	assert cfg['model']['output_probabilities'] is True
	assert cfg['inference']['output_dir'] == 'runs/inference'


def test_infer_volume_dry_run_prints_infer_volume_stage() -> None:
	result = run_python_proc(Path('proc/infer_volume.py'), '--dry-run')

	assert result.returncode == 0, result.stderr
	assert 'stage: infer_volume' in result.stdout


def test_reordered_attribute_names_raise_clear_value_error() -> None:
	cfg = _valid_config()
	cfg['attributes']['names'] = [
		'phase_sin',
		'amplitude_norm',
		*cfg['attributes']['names'][2:],
	]

	with pytest.raises(ValueError, match='attributes\\.names'):
		validate_config(cfg)


def test_changed_attribute_group_value_raises_clear_value_error() -> None:
	cfg = _valid_config()
	cfg['attributes']['groups']['coherence'] = 'texture'

	with pytest.raises(ValueError, match='attributes\\.groups'):
		validate_config(cfg)


def test_unknown_stage_raises_clear_value_error() -> None:
	cfg = _valid_config()
	cfg['stage'] = 'unknown_stage'

	with pytest.raises(ValueError, match='stage'):
		validate_config(cfg)


def test_f3_pretraining_setting_is_rejected() -> None:
	cfg = _valid_config()
	cfg['data']['f3_root'] = 'f3'

	with pytest.raises(ValueError, match='F3 settings are not allowed'):
		validate_config(cfg)


def test_invalid_masking_ratio_raises_clear_value_error() -> None:
	cfg = _valid_config()
	cfg['masking']['spatial_mask_ratio'] = 1.0

	with pytest.raises(ValueError, match='masking\\.spatial_mask_ratio'):
		validate_config(cfg)


def test_invalid_masking_mode_raises_clear_value_error() -> None:
	cfg = _valid_config()
	cfg['masking']['spatial_mask_mode'] = 'random'

	with pytest.raises(ValueError, match='masking\\.spatial_mask_mode'):
		validate_config(cfg)


def test_invalid_masking_attribute_bounds_raise_clear_value_error() -> None:
	cfg = _valid_config()
	cfg['masking']['min_input_attributes'] = 11
	cfg['masking']['max_input_attributes'] = 10

	with pytest.raises(ValueError, match='masking\\.min_input_attributes'):
		validate_config(cfg)


@pytest.mark.parametrize(
	('key', 'value', 'error_type', 'match'),
	[
		('samples_per_epoch', 0, ValueError, 'train\\.samples_per_epoch'),
		('samples_per_epoch', True, TypeError, 'train\\.samples_per_epoch'),
		('num_workers', -1, ValueError, 'train\\.num_workers'),
		('num_workers', False, TypeError, 'train\\.num_workers'),
		('shuffle', 'true', TypeError, 'train\\.shuffle'),
	],
)
def test_invalid_train_runtime_fields_raise_clear_error(
	key: str,
	value: object,
	error_type: type[Exception],
	match: str,
) -> None:
	cfg = _valid_config()
	cfg['train'][key] = value

	with pytest.raises(error_type, match=match):
		validate_config(cfg)


@pytest.mark.parametrize(
	('value', 'error_type'),
	[
		(0.0, ValueError),
		(1.1, ValueError),
		(False, TypeError),
	],
)
def test_invalid_context_token_min_valid_fraction_raises_clear_error(
	value: object,
	error_type: type[Exception],
) -> None:
	cfg = _valid_config()
	cfg['model']['context_token_min_valid_fraction'] = value

	with pytest.raises(
		error_type,
		match='model\\.context_token_min_valid_fraction',
	):
		validate_config(cfg)


def _valid_config() -> dict:
	return deepcopy(load_config(Path('proc/configs/mvp_mae.yaml')))
