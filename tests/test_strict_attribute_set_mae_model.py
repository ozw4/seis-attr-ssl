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


def _make_model(
	*,
	use_context: bool = False,
	context_token_min_valid_fraction: float = 0.5,
) -> StrictAttributeSetMAE3D:
	return StrictAttributeSetMAE3D(
		patch_size_xyz=(4, 4, 4),
		encoder_dim=32,
		encoder_depth=1,
		encoder_heads=4,
		decoder_dim=16,
		decoder_depth=1,
		decoder_heads=4,
		num_context_tokens=2,
		context_token_min_valid_fraction=context_token_min_valid_fraction,
		use_context=use_context,
	)


def _capture_context_token_valid_mask(
	model: StrictAttributeSetMAE3D,
	context_valid_mask: torch.Tensor,
	monkeypatch: pytest.MonkeyPatch,
) -> torch.Tensor | None:
	captured: dict[str, torch.Tensor | None] = {}

	def capture_pooler(
		context_tokens: torch.Tensor,
		context_token_valid_mask: torch.Tensor | None = None,
	) -> torch.Tensor:
		captured['mask'] = context_token_valid_mask
		return torch.zeros(
			(context_tokens.shape[0], model.num_context_tokens, model.encoder_dim),
			dtype=context_tokens.dtype,
			device=context_tokens.device,
		)

	monkeypatch.setattr(model.context_pooler, 'forward', capture_pooler)
	context = torch.randn((context_valid_mask.shape[0], 3, 8, 4, 4))
	attribute_ids = torch.arange(3).unsqueeze(0).expand(context.shape[0], -1).clone()
	spatial_mask = torch.zeros((context.shape[0], 2, 1, 1), dtype=torch.bool)
	spatial_mask[:, 0, 0, 0] = True

	model(
		{
			'x': torch.randn_like(context),
			'attribute_ids': attribute_ids,
			'spatial_mask': spatial_mask,
			'visible_spatial_mask': ~spatial_mask,
			'context': context,
			'context_valid_mask': context_valid_mask,
		},
	)

	return captured['mask']


def test_forward_pass_without_context() -> None:
	model = _make_model()
	out = model(_make_batch())

	assert out['pred_patches'].shape == (2, 64, 10, 64)
	assert out['decoder_tokens'].shape == (2, 64, 16)
	assert out['spatial_mask'].shape == (2, 4, 4, 4)
	assert out['token_grid_shape'] == (4, 4, 4)


def test_forward_pass_with_explicit_none_context() -> None:
	model = _make_model(use_context=True)
	batch = _make_batch()
	batch['context'] = None
	batch['context_valid_mask'] = None

	out = model(batch)

	assert out['pred_patches'].shape == (2, 64, 10, 64)
	assert out['decoder_tokens'].shape == (2, 64, 16)


def test_forward_pass_with_context() -> None:
	model = _make_model(use_context=True)
	out = model(_make_batch(use_context=True))

	assert out['pred_patches'].shape == (2, 64, 10, 64)
	assert out['encoded_tokens'].shape == (2, 62, 32)
	assert out['decoder_tokens'].shape == (2, 64, 16)


def test_forward_pass_with_context_matching_variable_local_shape() -> None:
	model = _make_model(use_context=True)
	batch = _make_batch(batch_size=1, channels=2, use_context=True)
	batch['x'] = torch.randn((1, 2, 8, 8, 16))
	batch['context'] = torch.randn_like(batch['x'])
	batch['attribute_ids'] = torch.tensor([[0, 1]])
	batch['spatial_mask'] = torch.zeros((1, 2, 2, 4), dtype=torch.bool)
	batch['spatial_mask'][:, 0, 0, 0] = True
	batch['visible_spatial_mask'] = ~batch['spatial_mask']
	batch['context_valid_mask'] = torch.ones((1, 8, 8, 16), dtype=torch.bool)

	out = model(batch)

	assert out['pred_patches'].shape == (1, 16, 10, 64)
	assert out['token_grid_shape'] == (2, 2, 4)


def test_context_shape_mismatch_raises_clear_value_error() -> None:
	model = _make_model(use_context=True)
	batch = _make_batch(batch_size=1, use_context=True)
	batch['context'] = torch.randn((1, 3, 8, 16, 16))
	batch['context_valid_mask'] = torch.ones((1, 8, 16, 16), dtype=torch.bool)

	with pytest.raises(ValueError, match='context payload shape must match local x'):
		model(batch)


def test_fully_valid_context_mask_keeps_all_context_tokens_valid(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	model = _make_model(use_context=True)
	context_valid_mask = torch.ones((1, 8, 4, 4), dtype=torch.bool)

	context_token_valid_mask = _capture_context_token_valid_mask(
		model,
		context_valid_mask,
		monkeypatch,
	)

	assert context_token_valid_mask is not None
	assert context_token_valid_mask.tolist() == [[True, True]]


def test_boundary_context_token_below_valid_fraction_is_invalid(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	model = _make_model(use_context=True)
	context_valid_mask = torch.ones((1, 8, 4, 4), dtype=torch.bool)
	context_valid_mask[:, :4] = False
	context_valid_mask[:, :1] = True

	context_token_valid_mask = _capture_context_token_valid_mask(
		model,
		context_valid_mask,
		monkeypatch,
	)

	assert context_token_valid_mask is not None
	assert context_token_valid_mask.tolist() == [[False, True]]


def test_context_token_above_valid_fraction_threshold_is_valid(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	model = _make_model(use_context=True)
	context_valid_mask = torch.ones((1, 8, 4, 4), dtype=torch.bool)
	context_valid_mask[:, :4] = False
	context_valid_mask[:, :3] = True

	context_token_valid_mask = _capture_context_token_valid_mask(
		model,
		context_valid_mask,
		monkeypatch,
	)

	assert context_token_valid_mask is not None
	assert context_token_valid_mask.tolist() == [[True, True]]


def test_all_invalid_context_tokens_raise_clear_value_error() -> None:
	model = _make_model(use_context=True)
	batch = _make_batch(batch_size=1, use_context=True)
	batch['context_valid_mask'] = torch.zeros((1, 16, 16, 16), dtype=torch.bool)

	with pytest.raises(ValueError, match='at least one valid context token'):
		model(batch)


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
