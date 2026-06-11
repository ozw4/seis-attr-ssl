from __future__ import annotations

import torch

from seis_attr_ssl.models.common import TransformerStack


def test_transformer_stack_preserves_shape() -> None:
	stack = TransformerStack(embed_dim=8, num_heads=2, depth=2)
	tokens = torch.randn((3, 5, 8))

	output = stack(tokens)

	assert output.shape == (3, 5, 8)


def test_transformer_stack_is_deterministic_in_eval_mode() -> None:
	stack = TransformerStack(embed_dim=8, num_heads=2, depth=2, dropout=0.5)
	stack.eval()
	tokens = torch.randn((2, 4, 8))

	first = stack(tokens)
	second = stack(tokens)

	assert torch.equal(first, second)


def test_transformer_stack_accepts_key_padding_mask() -> None:
	stack = TransformerStack(embed_dim=8, num_heads=2, depth=1)
	tokens = torch.randn((2, 4, 8))
	key_padding_mask = torch.tensor(
		[[False, False, True, True], [False, True, False, True]],
	)

	output = stack(tokens, key_padding_mask)

	assert output.shape == tokens.shape
