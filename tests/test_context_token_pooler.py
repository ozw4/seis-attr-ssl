from __future__ import annotations

import pytest
import torch

from seis_attr_ssl.models.mae import ContextTokenPooler


def test_context_token_pooler_returns_expected_shape() -> None:
	pooler = ContextTokenPooler(
		embed_dim=8,
		num_context_tokens=3,
		num_heads=2,
	)
	context_tokens = torch.randn((2, 5, 8))

	output = pooler(context_tokens)

	assert output.shape == (2, 3, 8)


def test_masked_invalid_context_tokens_do_not_affect_output() -> None:
	pooler = ContextTokenPooler(
		embed_dim=4,
		num_context_tokens=2,
		num_heads=2,
	)
	pooler.eval()
	context_tokens = torch.randn((1, 3, 4))
	context_token_valid_mask = torch.tensor([[True, True, False]])
	changed_invalid_tokens = context_tokens.clone()
	changed_invalid_tokens[:, 2] = 10_000.0

	output = pooler(context_tokens, context_token_valid_mask)
	changed_output = pooler(changed_invalid_tokens, context_token_valid_mask)

	assert torch.allclose(changed_output, output)


def test_all_invalid_context_token_mask_raises_value_error() -> None:
	pooler = ContextTokenPooler(
		embed_dim=4,
		num_context_tokens=2,
		num_heads=2,
	)
	context_tokens = torch.randn((2, 3, 4))
	context_token_valid_mask = torch.tensor(
		[[True, False, False], [False, False, False]],
	)

	with pytest.raises(ValueError, match='at least one valid context token'):
		pooler(context_tokens, context_token_valid_mask)


def test_context_token_pooler_gradients_flow() -> None:
	pooler = ContextTokenPooler(
		embed_dim=8,
		num_context_tokens=3,
		num_heads=2,
	)
	context_tokens = torch.randn((2, 5, 8), requires_grad=True)
	context_token_valid_mask = torch.tensor(
		[[True, True, False, True, True], [True, False, True, True, False]],
	)

	loss = pooler(context_tokens, context_token_valid_mask).square().sum()
	loss.backward()

	assert context_tokens.grad is not None
	assert pooler.query_tokens.grad is not None
	assert pooler.attention.in_proj_weight.grad is not None
