from __future__ import annotations

import pytest
import torch

from seis_attr_ssl.models.mae import (
	build_3d_sincos_position_embedding,
	restore_decoder_sequence,
	select_visible_tokens,
)


def test_position_embedding_shape_and_deterministic_values() -> None:
	first = build_3d_sincos_position_embedding((2, 2, 1), 12)
	second = build_3d_sincos_position_embedding((2, 2, 1), 12)

	assert first.shape == (4, 12)
	assert torch.equal(first, second)
	assert torch.allclose(
		first[0],
		torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]),
	)


def test_select_visible_tokens_respects_xyz_flatten_order() -> None:
	tokens = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1)
	pos = torch.arange(100, 108, dtype=torch.float32).reshape(8, 1)
	visible_spatial_mask = torch.zeros((1, 2, 2, 2), dtype=torch.bool)
	visible_spatial_mask[0, 0, 1, 1] = True
	visible_spatial_mask[0, 1, 0, 0] = True

	visible_tokens, visible_pos, valid_mask = select_visible_tokens(
		tokens,
		pos,
		visible_spatial_mask,
	)

	assert visible_tokens.squeeze(-1).tolist() == [[3.0, 4.0]]
	assert visible_pos.squeeze(-1).tolist() == [[103.0, 104.0]]
	assert valid_mask.tolist() == [[True, True]]


def test_select_visible_tokens_pads_variable_visible_counts() -> None:
	tokens = torch.arange(16, dtype=torch.float32).reshape(2, 8, 1)
	pos = torch.arange(8, dtype=torch.float32).reshape(8, 1)
	visible_spatial_mask = torch.zeros((2, 2, 2, 2), dtype=torch.bool)
	visible_spatial_mask[0, 0, 0, 0] = True
	visible_spatial_mask[1, 0, 0, 0] = True
	visible_spatial_mask[1, 1, 1, 1] = True

	visible_tokens, visible_pos, valid_mask = select_visible_tokens(
		tokens,
		pos,
		visible_spatial_mask,
	)

	assert visible_tokens.squeeze(-1).tolist() == [[0.0, 0.0], [8.0, 15.0]]
	assert visible_pos.squeeze(-1).tolist() == [[0.0, 0.0], [0.0, 7.0]]
	assert valid_mask.tolist() == [[True, False], [True, True]]


def test_all_masked_sample_raises_value_error() -> None:
	tokens = torch.zeros((2, 8, 4))
	pos = torch.zeros((8, 4))
	visible_spatial_mask = torch.ones((2, 2, 2, 2), dtype=torch.bool)
	visible_spatial_mask[1] = False

	with pytest.raises(ValueError, match='at least one visible spatial token'):
		select_visible_tokens(tokens, pos, visible_spatial_mask)


def test_restore_decoder_sequence_scatter_and_adds_positions() -> None:
	visible_tokens = torch.tensor([[[10.0], [20.0]]])
	pos = torch.arange(4, dtype=torch.float32).reshape(4, 1)
	visible_spatial_mask = torch.tensor(
		[[[[False, True], [False, True]]]],
		dtype=torch.bool,
	)
	mask_token = torch.tensor([-1.0])

	decoder_tokens, masked_token_mask = restore_decoder_sequence(
		visible_tokens,
		pos,
		visible_spatial_mask,
		mask_token,
	)

	assert decoder_tokens.squeeze(-1).tolist() == [[-1.0, 11.0, 1.0, 23.0]]
	assert masked_token_mask.tolist() == [[True, False, True, False]]
