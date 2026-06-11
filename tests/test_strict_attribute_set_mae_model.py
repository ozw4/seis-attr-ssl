from __future__ import annotations

import pytest
import torch

from seis_attr_ssl.models.mae import StrictAttributeSetMAE3D


def _make_batch(
	*,
	batch_size: int = 2,
	channels: int = 3,
	use_context: bool = False,
	valid_attributes: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
	x = torch.randn((batch_size, channels, 16, 16, 16))
	attribute_ids = torch.arange(channels).unsqueeze(0).expand(batch_size, -1).clone()
	spatial_mask = torch.zeros((batch_size, 4, 4, 4), dtype=torch.bool)
	spatial_mask[:, 0, 0, 0] = True
	spatial_mask[:, 1, 1, 1] = True
	batch = {
		'x': x,
		'attribute_ids': attribute_ids,
		'spatial_mask': spatial_mask,
		'visible_spatial_mask': ~spatial_mask,
	}
	if valid_attributes is not None:
		batch['valid_attributes'] = valid_attributes
	if use_context:
		batch['context'] = torch.randn_like(x)
		batch['context_valid_mask'] = torch.ones(
			(batch_size, 16, 16, 16),
			dtype=torch.bool,
		)
	return batch


def _make_model(*, use_context: bool = False) -> StrictAttributeSetMAE3D:
	return StrictAttributeSetMAE3D(
		patch_size_xyz=(4, 4, 4),
		encoder_dim=32,
		encoder_depth=1,
		encoder_heads=4,
		decoder_dim=16,
		decoder_depth=1,
		decoder_heads=4,
		num_context_tokens=2,
		use_context=use_context,
	)


def test_forward_pass_without_context() -> None:
	model = _make_model()
	out = model(_make_batch())

	assert out['pred_patches'].shape == (2, 64, 10, 64)
	assert out['decoder_tokens'].shape == (2, 64, 16)
	assert out['spatial_mask'].shape == (2, 4, 4, 4)
	assert out['token_grid_shape'] == (4, 4, 4)


def test_forward_pass_with_context() -> None:
	model = _make_model(use_context=True)
	out = model(_make_batch(use_context=True))

	assert out['pred_patches'].shape == (2, 64, 10, 64)
	assert out['encoded_tokens'].shape == (2, 62, 32)
	assert out['decoder_tokens'].shape == (2, 64, 16)


def test_encoder_receives_only_visible_local_tokens(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	model = _make_model()
	batch = _make_batch(batch_size=1)
	captured: dict[str, torch.Tensor] = {}

	def capture_encoder(
		tokens: torch.Tensor,
		key_padding_mask: torch.Tensor | None = None,
	) -> torch.Tensor:
		captured['tokens'] = tokens
		assert key_padding_mask is not None
		captured['key_padding_mask'] = key_padding_mask
		return tokens

	monkeypatch.setattr(model.encoder, 'forward', capture_encoder)
	model(batch)

	assert captured['tokens'].shape == (1, 62, 32)
	assert captured['key_padding_mask'].shape == (1, 62)
	assert not captured['key_padding_mask'].any()


def test_smoke_output_patch_shape_matches_issue_example() -> None:
	model = StrictAttributeSetMAE3D(
		patch_size_xyz=(4, 4, 4),
		encoder_dim=64,
		encoder_depth=2,
		encoder_heads=4,
		decoder_dim=32,
		decoder_depth=1,
		decoder_heads=4,
		num_context_tokens=2,
	)
	out = model(_make_batch(use_context=True))

	assert out['pred_patches'].shape == (2, 64, 10, 64)


def test_gradients_flow_from_pred_patches_sum() -> None:
	model = _make_model(use_context=True)
	out = model(_make_batch(use_context=True))

	out['pred_patches'].sum().backward()

	assert model.local_tokenizer.patch_projection.weight.grad is not None
	assert model.encoder.layers[0].attention.in_proj_weight.grad is not None
	assert model.decoder.layers[0].attention.in_proj_weight.grad is not None
	assert model.prediction_head.weight.grad is not None


def test_variable_input_attribute_counts_via_padded_mask() -> None:
	valid_attributes = torch.tensor(
		[[True, True, False], [True, False, False]],
		dtype=torch.bool,
	)
	batch = _make_batch(valid_attributes=valid_attributes)
	batch['attribute_ids'] = torch.tensor([[0, 1, -1], [2, -1, -1]])
	model = _make_model()

	out = model(batch)

	assert out['pred_patches'].shape == (2, 64, 10, 64)


@pytest.mark.parametrize(
	('spatial_mask', 'visible_spatial_mask', 'match'),
	[
		(
			torch.ones((1, 4, 4, 4), dtype=torch.bool),
			torch.zeros((1, 4, 4, 4), dtype=torch.bool),
			'at least one visible spatial token',
		),
		(
			torch.zeros((1, 4, 4, 4), dtype=torch.bool),
			torch.zeros((1, 4, 4, 4), dtype=torch.bool),
			'visible_spatial_mask must equal',
		),
	],
)
def test_all_masked_or_all_invisible_spatial_mask_raises(
	spatial_mask: torch.Tensor,
	visible_spatial_mask: torch.Tensor,
	match: str,
) -> None:
	batch = _make_batch(batch_size=1)
	batch['spatial_mask'] = spatial_mask
	batch['visible_spatial_mask'] = visible_spatial_mask
	model = _make_model()

	with pytest.raises(ValueError, match=match):
		model(batch)
