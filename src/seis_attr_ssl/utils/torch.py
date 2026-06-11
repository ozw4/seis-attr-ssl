"""Small PyTorch utility helpers."""

from __future__ import annotations

import torch


def set_torch_seed(seed: int) -> None:
	"""Seed PyTorch random number generators."""
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def get_default_device(device: str | None = None) -> torch.device:
	"""Return an explicit device, or CUDA when available."""
	if device is not None:
		return torch.device(device)
	if torch.cuda.is_available():
		return torch.device('cuda')
	return torch.device('cpu')


def count_trainable_parameters(module: torch.nn.Module) -> int:
	"""Count trainable parameters in a PyTorch module."""
	return sum(
		parameter.numel()
		for parameter in module.parameters()
		if parameter.requires_grad
	)


__all__ = ['count_trainable_parameters', 'get_default_device', 'set_torch_seed']
