"""Checkpoint IO for MAE pretraining."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(  # noqa: PLR0913
	path: str | Path,
	*,
	model: torch.nn.Module,
	optimizer: torch.optim.Optimizer,
	epoch: int,
	config: Mapping[str, object],
	package_version: str | None = None,
	metrics: Mapping[str, float] | None = None,
) -> Path:
	"""Write a training checkpoint and return its path."""
	checkpoint_path = Path(path)
	checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
	payload: dict[str, object] = {
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'epoch': int(epoch),
		'config': _to_plain_value(config),
		'package_version': package_version,
	}
	if metrics is not None:
		payload['metrics'] = dict(metrics)
	torch.save(payload, checkpoint_path)
	return checkpoint_path


def load_checkpoint(
	path: str | Path,
	map_location: str | torch.device | None = None,
) -> dict[str, Any]:
	"""Load a checkpoint payload from disk."""
	return torch.load(Path(path), map_location=map_location, weights_only=False)


def _to_plain_value(value: object) -> object:
	if isinstance(value, Mapping):
		return {str(key): _to_plain_value(child) for key, child in value.items()}
	if isinstance(value, list | tuple):
		return [_to_plain_value(child) for child in value]
	if isinstance(value, Path):
		return str(value)
	return value


__all__ = ['load_checkpoint', 'save_checkpoint']
