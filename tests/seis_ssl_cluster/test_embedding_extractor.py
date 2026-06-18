from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import torch

from seis_ssl_cluster.data import (
	GRID_ORDER_XYZ,
	AmplitudeVolumeRecord,
	SurveyManifest,
	SurveyNormalizationStats,
	write_manifest_json,
	write_normalization_stats,
)
from seis_ssl_cluster.embedding import run_embedding_extraction
from seis_ssl_cluster.models.mae import AmplitudeMAE3D

if TYPE_CHECKING:
	from pathlib import Path


def test_embedding_extraction_writes_deterministic_nondivisible_outputs(
	tmp_path: Path,
) -> None:
	config = _write_fixture(tmp_path)

	first = run_embedding_extraction(config, device='cpu')
	embeddings_path = first[0].embeddings_path
	valid_tokens_path = first[0].valid_tokens_path
	metadata_path = first[0].metadata_path
	first_embeddings = np.load(embeddings_path)
	first_valid = np.load(valid_tokens_path)

	second = run_embedding_extraction(config, device='cpu')
	second_embeddings = np.load(second[0].embeddings_path)
	second_valid = np.load(second[0].valid_tokens_path)

	assert first[0].skipped is False
	assert second[0].skipped is False
	assert first_embeddings.shape == (3, 3, 4, 12)
	assert first_embeddings.dtype == np.float16
	assert first_valid.shape == (3, 3, 4)
	assert first_valid.dtype == np.bool_
	assert first_valid.any()
	np.testing.assert_array_equal(first_embeddings, second_embeddings)
	np.testing.assert_array_equal(first_valid, second_valid)

	metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
	assert metadata['source_amplitude_path'].endswith('amplitude.npy')
	assert metadata['checkpoint_path'].endswith('mae.pt')
	assert metadata['checkpoint_sha256']
	assert metadata['patch_size'] == [2, 2, 2]
	assert metadata['token_grid_shape'] == [3, 3, 4]
	assert metadata['window_size'] == [4, 4, 4]
	assert metadata['overlap'] == [2, 2, 2]
	assert metadata['output_dtype'] == 'float16'
	assert metadata['min_token_valid_fraction'] == 0.5


def test_embedding_extraction_skip_existing_uses_matching_metadata(
	tmp_path: Path,
) -> None:
	config = _write_fixture(tmp_path)
	run_embedding_extraction(config, device='cpu')

	result = run_embedding_extraction(config, skip_existing=True, device='cpu')

	assert result[0].skipped is True


def _write_fixture(tmp_path: Path) -> dict[str, object]:
	survey_root = tmp_path / 'survey-a'
	survey_root.mkdir()
	volume_path = survey_root / 'amplitude.npy'
	volume = np.arange(5 * 6 * 7, dtype=np.float32).reshape(5, 6, 7)
	volume[0, 0, :] = 0.0
	np.save(volume_path, volume)
	stats_path = survey_root / 'stats.json'
	write_normalization_stats(
		SurveyNormalizationStats(
			survey_id='survey-a',
			source_path=volume_path,
			grid_order=GRID_ORDER_XYZ,
			clip_low_percentile=0.0,
			clip_high_percentile=100.0,
			clip_low=-1000.0,
			clip_high=1000.0,
			median=0.0,
			iqr=100.0,
		),
		stats_path,
	)
	manifest = SurveyManifest(
		survey_id='survey-a',
		root=survey_root,
		amplitude=AmplitudeVolumeRecord(
			survey_id='survey-a',
			path=volume_path,
			shape_xyz=tuple(int(axis) for axis in volume.shape),
			dtype='float32',
			grid_order=GRID_ORDER_XYZ,
			normalization_stats_path=stats_path,
		),
	)
	manifest_path = tmp_path / 'manifest.json'
	write_manifest_json([manifest], manifest_path)
	checkpoint_path = tmp_path / 'mae.pt'
	model_config = {
		'name': 'amp_mae3d',
		'in_channels': 1,
		'out_channels': 1,
		'patch_size': [2, 2, 2],
		'encoder_dim': 12,
		'encoder_depth': 1,
		'encoder_heads': 3,
		'decoder_dim': 12,
		'decoder_depth': 1,
		'decoder_heads': 3,
	}
	torch.manual_seed(7)
	model = AmplitudeMAE3D(
		in_channels=1,
		out_channels=1,
		patch_size_xyz=(2, 2, 2),
		encoder_dim=12,
		encoder_depth=1,
		encoder_heads=3,
		decoder_dim=12,
		decoder_depth=1,
		decoder_heads=3,
	)
	torch.save(
		{
			'model_state_dict': model.state_dict(),
			'config': {'stage': 'train_amp_mae', 'model': model_config},
		},
		checkpoint_path,
	)
	return {
		'stage': 'extract_embeddings',
		'paths': {
			'nopims_root': str(tmp_path),
			'artifact_root': str(tmp_path / 'artifacts'),
		},
		'manifests': {'input': str(manifest_path)},
		'embeddings': {
			'checkpoint': str(checkpoint_path),
			'output_dir': str(tmp_path / 'embeddings'),
		},
		'embedding': {
			'window_size': [4, 4, 4],
			'overlap': [2, 2, 2],
			'output_dtype': 'float16',
			'batch_size': 2,
			'min_token_valid_fraction': 0.5,
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
			'spatial_mask_ratio': 0.5,
			'spatial_mask_mode': 'block',
			'block_size_tokens': [1, 1, 1],
		},
		'zero_mask': {'enabled': False},
		'train': {
			'batch_size': 1,
			'samples_per_epoch': 1,
			'epochs': 1,
			'amp': False,
			'device': 'cpu',
		},
	}
