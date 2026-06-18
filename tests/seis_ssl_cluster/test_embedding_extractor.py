from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from seis_ssl_cluster.data.normalization import (
	SurveyNormalizationStats,
	write_normalization_stats,
)
from seis_ssl_cluster.data.schema import (
	AmplitudeVolumeRecord,
	SurveyManifest,
	write_manifest_json,
)
from seis_ssl_cluster.embedding.extractor import (
	extraction_settings_from_config,
	run_embedding_extraction,
)
from seis_ssl_cluster.models.mae import AmplitudeMAE3D
from seis_ssl_cluster.training.checkpoint import save_checkpoint

if TYPE_CHECKING:
	from pathlib import Path


def test_run_embedding_extraction_hashes_checkpoint_once(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	context = _write_extraction_context(tmp_path, survey_count=2)
	calls: list[Path] = []

	def fake_file_sha256(path: Path) -> str:
		calls.append(path)
		return 'abc123'

	monkeypatch.setattr(
		'seis_ssl_cluster.embedding.extractor.file_sha256',
		fake_file_sha256,
	)

	results = run_embedding_extraction(context.config, device='cpu')

	assert len(results) == 2
	assert calls == [context.checkpoint_path]
	for result in results:
		assert result.metadata_path.is_file()
		metadata = json.loads(result.metadata_path.read_text(encoding='utf-8'))
		assert metadata['checkpoint_sha256'] == 'abc123'


def test_checkpoint_zero_mask_drives_metadata_and_valid_tokens(
	tmp_path: Path,
) -> None:
	zero_mask = {
		'enabled': True,
		'zero_atol': 0.0,
		'z_sample_influence_radius': 0,
		'xy_trace_influence_radius': 0,
	}
	context = _write_extraction_context(
		tmp_path,
		survey_count=1,
		zero_mask=zero_mask,
	)

	result = run_embedding_extraction(context.config, device='cpu')[0]

	metadata = json.loads(result.metadata_path.read_text(encoding='utf-8'))
	assert metadata['zero_mask'] == zero_mask
	valid_tokens = np.load(result.valid_tokens_path)
	assert valid_tokens.shape == (2, 2, 2)
	assert not valid_tokens[:, :, 0].any()
	assert valid_tokens[:, :, 1].all()


def test_conflicting_extraction_zero_mask_override_is_rejected(tmp_path: Path) -> None:
	context = _write_extraction_context(
		tmp_path,
		survey_count=1,
		zero_mask={
			'enabled': True,
			'zero_atol': 0.0,
			'z_sample_influence_radius': 0,
			'xy_trace_influence_radius': 0,
		},
	)
	config = dict(context.config)
	config['data'] = {
		**context.config['data'],
		'zero_mask': {
			'enabled': True,
			'zero_atol': 0.0,
			'z_sample_influence_radius': 1,
			'xy_trace_influence_radius': 0,
		},
	}

	with pytest.raises(ValueError, match='zero_mask override must match checkpoint'):
		run_embedding_extraction(config, device='cpu')


def test_integer_output_dtype_is_rejected(tmp_path: Path) -> None:
	context = _write_extraction_context(tmp_path, survey_count=1)
	config = dict(context.config)
	config['embedding'] = {**context.config['embedding'], 'output_dtype': 'int16'}

	with pytest.raises(TypeError, match='floating-point NumPy dtype'):
		extraction_settings_from_config(config)


def test_full_model_geometry_is_present_in_metadata(tmp_path: Path) -> None:
	context = _write_extraction_context(tmp_path, survey_count=1)

	result = run_embedding_extraction(context.config, device='cpu')[0]

	metadata = json.loads(result.metadata_path.read_text(encoding='utf-8'))
	assert metadata['model_geometry'] == {
		'name': 'amp_mae3d',
		'in_channels': 1,
		'out_channels': 1,
		'patch_size': [2, 2, 2],
		'encoder_dim': 4,
		'encoder_depth': 1,
		'encoder_heads': 1,
		'decoder_dim': 4,
		'decoder_depth': 1,
		'decoder_heads': 1,
	}
	assert metadata['checkpoint_sha256']


class _ExtractionContext:
	def __init__(
		self,
		*,
		config: dict[str, object],
		checkpoint_path: Path,
	) -> None:
		self.config = config
		self.checkpoint_path = checkpoint_path


def _write_extraction_context(
	tmp_path: Path,
	*,
	survey_count: int,
	zero_mask: dict[str, object] | None = None,
) -> _ExtractionContext:
	model_config = {
		'name': 'amp_mae3d',
		'in_channels': 1,
		'out_channels': 1,
		'patch_size': [2, 2, 2],
		'encoder_dim': 4,
		'encoder_depth': 1,
		'encoder_heads': 1,
		'decoder_dim': 4,
		'decoder_depth': 1,
		'decoder_heads': 1,
	}
	checkpoint_config: dict[str, object] = {
		'stage': 'train_amp_mae',
		'data': {
			'grid_order': ['x', 'y', 'z'],
			'volume_format': 'npy_memmap',
			'input_channels': 1,
			'target_channels': 1,
			'use_context': False,
			'local_crop_size': [4, 4, 4],
		},
		'model': model_config,
		'masking': {
			'spatial_mask_ratio': 0.75,
			'spatial_mask_mode': 'block',
			'block_size_tokens': [1, 1, 1],
		},
		'train': {'device': 'cpu'},
	}
	if zero_mask is not None:
		checkpoint_config['data'] = {
			**checkpoint_config['data'],
			'zero_mask': zero_mask,
		}
	model = AmplitudeMAE3D(
		in_channels=1,
		out_channels=1,
		patch_size_xyz=(2, 2, 2),
		encoder_dim=4,
		encoder_depth=1,
		encoder_heads=1,
		decoder_dim=4,
		decoder_depth=1,
		decoder_heads=1,
	)
	optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
	checkpoint_path = save_checkpoint(
		tmp_path / 'checkpoint.pt',
		model=model,
		optimizer=optimizer,
		epoch=1,
		config=checkpoint_config,
	)
	manifest_path = tmp_path / 'manifests.json'
	write_manifest_json(_manifests(tmp_path, survey_count), manifest_path)
	config = {
		'stage': 'extract_embeddings',
		'paths': {
			'nopims_root': str(tmp_path / 'nopims'),
			'artifact_root': str(tmp_path / 'artifacts'),
		},
		'manifests': {'input': str(manifest_path)},
		'embeddings': {
			'checkpoint': str(checkpoint_path),
			'output_dir': str(tmp_path / 'embeddings'),
		},
		'embedding': {
			'window_size': [4, 4, 4],
			'overlap': [0, 0, 0],
			'output_dtype': 'float32',
			'batch_size': 2,
			'min_token_valid_fraction': 1.0,
		},
		'data': {
			'grid_order': ['x', 'y', 'z'],
			'volume_format': 'npy_memmap',
			'input_channels': 1,
			'target_channels': 1,
			'use_context': False,
			'local_crop_size': [4, 4, 4],
		},
		'model': model_config,
		'masking': {
			'spatial_mask_ratio': 0.75,
			'spatial_mask_mode': 'block',
			'block_size_tokens': [1, 1, 1],
		},
		'train': {
			'batch_size': 1,
			'samples_per_epoch': 1,
			'epochs': 1,
			'num_workers': 1,
			'amp': False,
		},
	}
	return _ExtractionContext(config=config, checkpoint_path=checkpoint_path)


def _manifests(tmp_path: Path, survey_count: int) -> list[SurveyManifest]:
	manifests = []
	for index in range(survey_count):
		survey_id = f'survey-{index}'
		volume_path = tmp_path / f'{survey_id}.npy'
		volume = np.ones((4, 4, 4), dtype=np.float32)
		volume[:, :, 1] = 0.0
		np.save(volume_path, volume)
		stats_path = tmp_path / f'{survey_id}.stats.json'
		write_normalization_stats(
			SurveyNormalizationStats(
				survey_id=survey_id,
				source_path=volume_path,
				grid_order=('x', 'y', 'z'),
				clip_low_percentile=0.0,
				clip_high_percentile=100.0,
				clip_low=0.0,
				clip_high=1.0,
				median=0.0,
				iqr=1.0,
			),
			stats_path,
		)
		manifests.append(
			SurveyManifest(
				survey_id=survey_id,
				root=tmp_path,
				amplitude=AmplitudeVolumeRecord(
					survey_id=survey_id,
					path=volume_path,
					shape_xyz=(4, 4, 4),
					dtype='float32',
					grid_order=('x', 'y', 'z'),
					normalization_stats_path=stats_path,
				),
			),
		)
	return manifests
