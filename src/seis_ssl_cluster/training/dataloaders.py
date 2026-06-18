"""DataLoader builders for amplitude MAE pretraining."""

from __future__ import annotations

import torch

from seis_ssl_cluster.training.collate import mae_collate_fn


def build_mae_dataloader(
	dataset: object,
	*,
	batch_size: int,
	num_workers: int = 0,
	shuffle: bool = True,
	seed: int = 42,
) -> torch.utils.data.DataLoader:
	"""Build a deterministic amplitude MAE DataLoader."""
	generator = torch.Generator()
	generator.manual_seed(seed)
	return torch.utils.data.DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
		persistent_workers=num_workers > 0,
		collate_fn=mae_collate_fn,
		generator=generator,
	)


__all__ = ['build_mae_dataloader']
