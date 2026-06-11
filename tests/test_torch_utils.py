from __future__ import annotations

import torch

from seis_attr_ssl.utils.torch import (
	count_trainable_parameters,
	get_default_device,
	set_torch_seed,
)


def test_get_default_device_accepts_explicit_cpu() -> None:
	assert get_default_device('cpu') == torch.device('cpu')


def test_set_torch_seed_does_not_error() -> None:
	set_torch_seed(42)


def test_count_trainable_parameters_counts_only_trainable_parameters() -> None:
	module = torch.nn.Sequential(
		torch.nn.Linear(2, 3),
		torch.nn.Linear(3, 1),
	)
	for parameter in module[1].parameters():
		parameter.requires_grad = False

	assert count_trainable_parameters(module) == 9
