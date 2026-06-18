from __future__ import annotations

import numpy as np
import torch

from seis_ssl_cluster.training import build_mae_dataloader


def test_build_mae_dataloader_uses_amplitude_collate_and_persistent_workers() -> None:
	dataloader = build_mae_dataloader(
		[_sample(), _sample()],
		batch_size=2,
		num_workers=1,
		shuffle=False,
	)

	assert dataloader.persistent_workers is True
	batches = list(dataloader)
	assert len(batches) == 1
	batch = batches[0]

	assert batch['x'].shape == (2, 1, 2, 2, 2)
	assert batch['x'].dtype == torch.float32
	assert batch['spatial_mask'].dtype == torch.bool
	assert batch['coords'] == [{'survey_id': 'survey'}, {'survey_id': 'survey'}]


def test_build_mae_dataloader_disables_persistent_workers_without_workers() -> None:
	dataloader = build_mae_dataloader(
		[_sample()],
		batch_size=1,
		num_workers=0,
		shuffle=False,
	)

	assert dataloader.persistent_workers is False


def _sample() -> dict[str, object]:
	x = np.ones((1, 2, 2, 2), dtype=np.float32)
	return {
		'x': x,
		'target': x.copy(),
		'spatial_mask': np.ones((1, 1, 1), dtype=np.bool_),
		'visible_spatial_mask': np.zeros((1, 1, 1), dtype=np.bool_),
		'local_valid_mask': np.ones((2, 2, 2), dtype=np.bool_),
		'coords': {'survey_id': 'survey'},
	}
