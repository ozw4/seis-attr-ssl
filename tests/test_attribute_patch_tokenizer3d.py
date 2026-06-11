from __future__ import annotations

import pytest
import torch

from seis_attr_ssl.models.tokenizers import AttributePatchTokenizer3D


def _make_tokenizer() -> AttributePatchTokenizer3D:
	return AttributePatchTokenizer3D(
		num_attributes=4,
		attribute_groups=('waveform', 'phase', 'phase', 'texture'),
		patch_size_xyz=(2, 2, 2),
		embed_dim=5,
	)


def test_attribute_patch_tokenizer3d_returns_expected_shape() -> None:
	tokenizer = _make_tokenizer()
	x = torch.zeros((2, 3, 4, 4, 4))
	attribute_ids = torch.tensor([[0, 1, 2], [1, 2, 3]])

	output = tokenizer(x, attribute_ids)

	assert output['tokens'].shape == (2, 8, 5)
	assert output['token_grid_shape'] == (2, 2, 2)


def test_attribute_patch_tokenizer3d_accepts_variable_attribute_counts() -> None:
	tokenizer = _make_tokenizer()
	x = torch.zeros((2, 3, 4, 4, 4))
	attribute_ids = torch.tensor([[0, 1, 2], [1, 2, 3]])
	attribute_valid_mask = torch.tensor(
		[[True, True, False], [True, False, False]],
	)

	output = tokenizer(x, attribute_ids, attribute_valid_mask)

	assert output['tokens'].shape == (2, 8, 5)


def test_invalid_padded_attributes_do_not_change_fused_token_result() -> None:
	tokenizer = _make_tokenizer()
	tokenizer.eval()
	valid_x = torch.arange(8, dtype=torch.float32).reshape(1, 1, 2, 2, 2)
	padded_x = torch.cat(
		[valid_x, torch.full((1, 1, 2, 2, 2), 10_000.0)],
		dim=1,
	)

	valid_output = tokenizer(valid_x, torch.tensor([[0]]))
	padded_output = tokenizer(
		padded_x,
		torch.tensor([[0, -1]]),
		torch.tensor([[True, False]]),
	)

	assert torch.allclose(padded_output['tokens'], valid_output['tokens'])


@pytest.mark.parametrize('attribute_ids', [torch.tensor([[4]]), torch.tensor([[-1]])])
def test_attribute_patch_tokenizer3d_rejects_invalid_attribute_ids(
	attribute_ids: torch.Tensor,
) -> None:
	tokenizer = _make_tokenizer()
	x = torch.zeros((1, 1, 2, 2, 2))

	with pytest.raises(ValueError, match='attribute_ids must be in range'):
		tokenizer(x, attribute_ids)


def test_attribute_patch_tokenizer3d_is_deterministic_in_eval_mode() -> None:
	tokenizer = _make_tokenizer()
	tokenizer.eval()
	x = torch.randn((2, 3, 4, 4, 4))
	attribute_ids = torch.tensor([[0, 1, 2], [1, 2, 3]])
	attribute_valid_mask = torch.tensor(
		[[True, True, False], [True, False, True]],
	)

	first = tokenizer(x, attribute_ids, attribute_valid_mask)['tokens']
	second = tokenizer(x, attribute_ids, attribute_valid_mask)['tokens']

	assert torch.equal(first, second)


def test_attribute_patch_tokenizer3d_gradients_flow() -> None:
	tokenizer = _make_tokenizer()
	x = torch.randn((2, 2, 4, 4, 4))
	attribute_ids = torch.tensor([[0, 1], [2, 3]])

	loss = tokenizer(x, attribute_ids)['tokens'].sum()
	loss.backward()

	assert tokenizer.patch_projection.weight.grad is not None
	assert tokenizer.patch_projection.bias.grad is not None
	assert tokenizer.attribute_embedding.weight.grad is not None
	assert tokenizer.group_embedding is not None
	assert tokenizer.group_embedding.weight.grad is not None
