"""DataLoader builders for MAE pretraining."""

from __future__ import annotations

import torch

from seis_attr_ssl.training.collate import mae_collate_fn


def build_mae_dataloader(
	dataset: object,
	*,
	batch_size: int,
	num_workers: int = 0,
	shuffle: bool = True,
	seed: int = 42,
) -> torch.utils.data.DataLoader:
	"""Build a deterministic MAE DataLoader with the project collate function."""
	generator = torch.Generator()
	generator.manual_seed(seed)
	return torch.utils.data.DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
		collate_fn=mae_collate_fn,
		generator=generator,
	)


__all__ = ['build_mae_dataloader']
