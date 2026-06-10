"""Fixed MVP configuration values."""

from __future__ import annotations

from typing import Final

DEFAULT_NOPIMS_ROOT: Final = '/home/dcuser/data/NOPIMS/'

EXPECTED_GRID_ORDER: Final = ['x', 'y', 'z']
EXPECTED_VOLUME_FORMAT: Final = 'npy_memmap'
EXPECTED_LOCAL_CROP_SIZE: Final = [128, 128, 128]
EXPECTED_CONTEXT_CROP_SIZE: Final = [512, 512, 512]
EXPECTED_CONTEXT_DOWNSAMPLE: Final = 4

EXPECTED_ATTRIBUTES: Final = [
	'amplitude_norm',
	'phase_sin',
	'phase_cos',
	'instantaneous_frequency',
	'spectral_low_ratio',
	'spectral_mid_ratio',
	'spectral_high_ratio',
	'coherence',
	'glcm_contrast',
	'glcm_homogeneity',
]

EXPECTED_ATTRIBUTE_GROUPS: Final = {
	'amplitude_norm': 'waveform',
	'phase_sin': 'phase',
	'phase_cos': 'phase',
	'instantaneous_frequency': 'frequency',
	'spectral_low_ratio': 'spectral',
	'spectral_mid_ratio': 'spectral',
	'spectral_high_ratio': 'spectral',
	'coherence': 'discontinuity',
	'glcm_contrast': 'texture',
	'glcm_homogeneity': 'texture',
}

DISALLOWED_PRETRAINING_KEYS: Final = {'f3', 'f3_root', 'f3_dataset', 'f3_data'}
F3_ALLOWED_STAGES: Final = {'finetune_f3', 'eval_f3'}
