from __future__ import annotations

from typing import TYPE_CHECKING

from seis_attr_ssl.training.mae import run_mae_pretraining
from tests.test_mae_training_engine import _tiny_config

if TYPE_CHECKING:
	from pathlib import Path


def test_mae_training_writes_debug_visualization_pngs(tmp_path: Path) -> None:
	cfg = _tiny_config(tmp_path)
	cfg['paths']['output_root'] = str(tmp_path / 'run' / 'checkpoints')
	cfg['train']['max_steps'] = 1
	cfg['visualization'] = {
		'mae_debug': {
			'enabled': True,
			'output_dir': None,
			'every_n_steps': 1,
			'every_n_epochs': None,
			'max_batches_per_trigger': 1,
			'max_samples_per_batch': 1,
			'attributes': ['amplitude_norm'],
			'columns': ['target', 'prediction'],
			'dpi': 30,
			'panel_width': 2.0,
			'panel_height': 1.6,
			'show_valid_mask_panel': False,
			'show_spatial_mask_panel': False,
		},
	}

	checkpoint_path = run_mae_pretraining(cfg)

	visualization_dir = tmp_path / 'run' / 'visualizations' / 'mae_debug'
	assert checkpoint_path.is_file()
	assert visualization_dir.is_dir()
	assert len(list(visualization_dir.glob('*_xy.png'))) == 1
	assert len(list(visualization_dir.glob('*_xz.png'))) == 1


def test_mae_training_disabled_debug_visualization_creates_no_directory(
	tmp_path: Path,
) -> None:
	cfg = _tiny_config(tmp_path)
	cfg['paths']['output_root'] = str(tmp_path / 'run' / 'checkpoints')
	cfg['train']['max_steps'] = 1
	cfg['visualization'] = {
		'mae_debug': {
			'enabled': False,
			'output_dir': None,
			'every_n_steps': 1,
		},
	}

	checkpoint_path = run_mae_pretraining(cfg)

	assert checkpoint_path.is_file()
	assert not (tmp_path / 'run' / 'visualizations' / 'mae_debug').exists()
