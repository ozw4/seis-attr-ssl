"""Strict attribute-set 3D masked autoencoder model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from seis_attr_ssl.attributes import MVP_ATTRIBUTE_REGISTRY
from seis_attr_ssl.models.common import TransformerStack
from seis_attr_ssl.models.mae.context import ContextTokenPooler
from seis_attr_ssl.models.mae.patching import patchify_3d
from seis_attr_ssl.models.mae.positional_encoding import (
	build_3d_sincos_position_embedding,
	restore_decoder_sequence,
	select_visible_tokens,
)
from seis_attr_ssl.models.tokenizers import AttributePatchTokenizer3D

if TYPE_CHECKING:
	from collections.abc import Mapping, Sequence


class StrictAttributeSetMAE3D(nn.Module):
	"""Masked autoencoder for variable 3D seismic attribute sets."""

	def __init__(  # noqa: PLR0913
		self,
		*,
		num_attributes: int = 10,
		attribute_groups: Mapping[str, str] | None = None,
		patch_size_xyz: tuple[int, int, int] = (8, 8, 8),
		encoder_dim: int = 384,
		encoder_depth: int = 8,
		encoder_heads: int = 6,
		decoder_dim: int = 256,
		decoder_depth: int = 4,
		decoder_heads: int = 4,
		num_context_tokens: int = 8,
		use_context: bool = True,
	) -> None:
		"""Initialize tokenizers, transformer stacks, and prediction head."""
		super().__init__()
		self.num_attributes = _validate_positive_int(num_attributes, 'num_attributes')
		self.patch_size_xyz = _validate_patch_size(patch_size_xyz)
		self.encoder_dim = _validate_positive_int(encoder_dim, 'encoder_dim')
		self.decoder_dim = _validate_positive_int(decoder_dim, 'decoder_dim')
		self.num_context_tokens = _validate_positive_int(
			num_context_tokens,
			'num_context_tokens',
		)
		if not isinstance(use_context, bool):
			msg = f'use_context must be bool; got {use_context!r}'
			raise TypeError(msg)
		self.use_context = use_context
		self.patch_volume = (
			self.patch_size_xyz[0] * self.patch_size_xyz[1] * self.patch_size_xyz[2]
		)

		tokenizer_attribute_groups = _normalize_attribute_groups(
			attribute_groups,
			self.num_attributes,
		)
		use_group_embedding = tokenizer_attribute_groups is not None
		self.local_tokenizer = AttributePatchTokenizer3D(
			num_attributes=self.num_attributes,
			attribute_groups=tokenizer_attribute_groups,
			patch_size_xyz=self.patch_size_xyz,
			embed_dim=self.encoder_dim,
			use_group_embedding=use_group_embedding,
		)
		self.context_tokenizer = AttributePatchTokenizer3D(
			num_attributes=self.num_attributes,
			attribute_groups=tokenizer_attribute_groups,
			patch_size_xyz=self.patch_size_xyz,
			embed_dim=self.encoder_dim,
			use_group_embedding=use_group_embedding,
		)
		self.context_pooler = ContextTokenPooler(
			embed_dim=self.encoder_dim,
			num_context_tokens=self.num_context_tokens,
			num_heads=encoder_heads,
		)
		self.context_pos = nn.Parameter(
			torch.empty(self.num_context_tokens, self.encoder_dim),
		)

		self.encoder = TransformerStack(
			embed_dim=self.encoder_dim,
			num_heads=encoder_heads,
			depth=encoder_depth,
		)
		self.encoder_to_decoder = nn.Linear(self.encoder_dim, self.decoder_dim)
		self.mask_token = nn.Parameter(torch.empty(self.decoder_dim))
		self.decoder_context_pos = nn.Parameter(
			torch.empty(self.num_context_tokens, self.decoder_dim),
		)
		self.decoder = TransformerStack(
			embed_dim=self.decoder_dim,
			num_heads=decoder_heads,
			depth=decoder_depth,
		)
		self.prediction_head = nn.Linear(
			self.decoder_dim,
			self.num_attributes * self.patch_volume,
		)
		self.reset_parameters()

	def reset_parameters(self) -> None:
		"""Initialize learned mask and context position tokens."""
		nn.init.normal_(self.context_pos, std=0.02)
		nn.init.normal_(self.decoder_context_pos, std=0.02)
		nn.init.normal_(self.mask_token, std=0.02)

	def forward(
		self,
		batch: Mapping[str, torch.Tensor],
	) -> dict[str, torch.Tensor | tuple[int, int, int]]:
		"""Return full-grid MAE patch predictions for the provided batch."""
		x = _required_tensor(batch, 'x')
		attribute_ids = _required_tensor(batch, 'attribute_ids')
		spatial_mask = _required_tensor(batch, 'spatial_mask')
		visible_spatial_mask = _required_tensor(batch, 'visible_spatial_mask')
		attribute_valid_mask = _optional_attribute_valid_mask(batch)

		local_output = self.local_tokenizer(x, attribute_ids, attribute_valid_mask)
		local_tokens = _as_tensor(local_output['tokens'], 'tokens')
		token_grid_shape = _as_grid_shape(local_output['token_grid_shape'])
		_validate_spatial_masks(
			spatial_mask,
			visible_spatial_mask,
			local_tokens.shape[0],
			token_grid_shape,
			local_tokens.device,
		)

		encoder_pos = build_3d_sincos_position_embedding(
			token_grid_shape,
			self.encoder_dim,
		).to(device=local_tokens.device, dtype=local_tokens.dtype)
		visible_tokens, visible_pos, visible_valid_mask = select_visible_tokens(
			local_tokens,
			encoder_pos,
			visible_spatial_mask,
		)
		encoder_tokens = visible_tokens + visible_pos
		encoder_key_padding_mask = ~visible_valid_mask

		context_token_count = 0
		if self.use_context and batch.get('context') is not None:
			context_tokens = self._encode_context(
				_as_tensor(batch['context'], 'context'),
				attribute_ids,
				attribute_valid_mask,
				batch.get('context_valid_mask'),
			)
			context_tokens = context_tokens + self.context_pos.to(
				device=context_tokens.device,
				dtype=context_tokens.dtype,
			).unsqueeze(0)
			context_token_count = int(context_tokens.shape[1])
			encoder_tokens = torch.cat([encoder_tokens, context_tokens], dim=1)
			context_padding = torch.zeros(
				(context_tokens.shape[0], context_tokens.shape[1]),
				dtype=torch.bool,
				device=context_tokens.device,
			)
			encoder_key_padding_mask = torch.cat(
				[encoder_key_padding_mask, context_padding],
				dim=1,
			)

		encoded = self.encoder(encoder_tokens, encoder_key_padding_mask)
		encoded_local = encoded[:, : visible_tokens.shape[1]]
		decoder_visible = self.encoder_to_decoder(encoded_local)
		decoder_pos = build_3d_sincos_position_embedding(
			token_grid_shape,
			self.decoder_dim,
		).to(device=decoder_visible.device, dtype=decoder_visible.dtype)
		decoder_tokens, _masked_token_mask = restore_decoder_sequence(
			decoder_visible,
			decoder_pos,
			visible_spatial_mask,
			self.mask_token.to(dtype=decoder_visible.dtype),
		)

		if context_token_count:
			encoded_context = encoded[:, -context_token_count:]
			decoder_context = self.encoder_to_decoder(encoded_context)
			decoder_context = decoder_context + self.decoder_context_pos[
				:context_token_count
			].to(
				device=decoder_context.device,
				dtype=decoder_context.dtype,
			).unsqueeze(0)
			decoder_input = torch.cat([decoder_tokens, decoder_context], dim=1)
			decoded = self.decoder(decoder_input)[:, : decoder_tokens.shape[1]]
		else:
			decoded = self.decoder(decoder_tokens)

		pred_patches = self.prediction_head(decoded).reshape(
			x.shape[0],
			decoder_tokens.shape[1],
			self.num_attributes,
			self.patch_volume,
		)
		return {
			'pred_patches': pred_patches,
			'encoded_tokens': encoded_local,
			'decoder_tokens': decoded,
			'spatial_mask': spatial_mask,
			'token_grid_shape': token_grid_shape,
		}

	def _encode_context(
		self,
		context: torch.Tensor,
		attribute_ids: torch.Tensor,
		attribute_valid_mask: torch.Tensor | None,
		context_valid_mask: torch.Tensor | None,
	) -> torch.Tensor:
		"""Tokenize and pool context volumes into fixed encoder tokens."""
		context_output = self.context_tokenizer(
			context,
			attribute_ids,
			attribute_valid_mask,
		)
		context_tokens = _as_tensor(context_output['tokens'], 'tokens')
		context_token_valid_mask = _context_token_valid_mask(
			context_valid_mask,
			context,
			self.patch_size_xyz,
		)
		return self.context_pooler(context_tokens, context_token_valid_mask)


def _required_tensor(
	batch: Mapping[str, torch.Tensor],
	key: str,
) -> torch.Tensor:
	try:
		return batch[key]
	except KeyError as exc:
		msg = f'batch is missing required key {key!r}'
		raise KeyError(msg) from exc


def _optional_attribute_valid_mask(
	batch: Mapping[str, torch.Tensor],
) -> torch.Tensor | None:
	if batch.get('valid_attributes') is not None:
		return batch['valid_attributes']
	if batch.get('attribute_valid_mask') is not None:
		return batch['attribute_valid_mask']
	return None


def _as_tensor(value: torch.Tensor | tuple[int, int, int], name: str) -> torch.Tensor:
	if not isinstance(value, torch.Tensor):
		msg = f'{name} must be a tensor; got {type(value).__name__}'
		raise TypeError(msg)
	return value


def _as_grid_shape(value: torch.Tensor | tuple[int, int, int]) -> tuple[int, int, int]:
	if (
		not isinstance(value, tuple)
		or len(value) != 3
		or any(not isinstance(item, int) for item in value)
	):
		msg = f'token_grid_shape must be an integer triple; got {value!r}'
		raise TypeError(msg)
	return value


def _validate_spatial_masks(
	spatial_mask: torch.Tensor,
	visible_spatial_mask: torch.Tensor,
	batch_size: int,
	token_grid_shape: tuple[int, int, int],
	device: torch.device,
) -> None:
	expected_shape = (batch_size, *token_grid_shape)
	for name, mask in (
		('spatial_mask', spatial_mask),
		('visible_spatial_mask', visible_spatial_mask),
	):
		if mask.ndim != 4 or tuple(mask.shape) != expected_shape:
			msg = (
				f'{name} must have shape [B, TX, TY, TZ]; '
				f'got shape={tuple(mask.shape)!r}, expected={expected_shape!r}'
			)
			raise ValueError(msg)
		if mask.dtype != torch.bool:
			msg = f'{name} dtype must be bool; got {mask.dtype}'
			raise TypeError(msg)
		if mask.device != device:
			msg = (
				f'{name} must be on the same device as x; '
				f'got mask_device={mask.device}, x_device={device}'
			)
			raise ValueError(msg)
	if not torch.equal(visible_spatial_mask, ~spatial_mask):
		msg = 'visible_spatial_mask must equal ~spatial_mask'
		raise ValueError(msg)


def _context_token_valid_mask(
	context_valid_mask: torch.Tensor | None,
	context: torch.Tensor,
	patch_size_xyz: tuple[int, int, int],
) -> torch.Tensor | None:
	if context_valid_mask is None:
		return None
	if context_valid_mask.ndim != 4 or tuple(context_valid_mask.shape) != (
		context.shape[0],
		context.shape[2],
		context.shape[3],
		context.shape[4],
	):
		msg = (
			'context_valid_mask must have shape [B, X, Y, Z] matching context; '
			f'got shape={tuple(context_valid_mask.shape)!r}'
		)
		raise ValueError(msg)
	if context_valid_mask.dtype != torch.bool:
		msg = f'context_valid_mask dtype must be bool; got {context_valid_mask.dtype}'
		raise TypeError(msg)
	if context_valid_mask.device != context.device:
		msg = (
			'context_valid_mask must be on the same device as context; '
			f'got mask_device={context_valid_mask.device}, '
			f'context_device={context.device}'
		)
		raise ValueError(msg)
	context_patches = patchify_3d(
		context_valid_mask.unsqueeze(1).to(dtype=context.dtype),
		patch_size_xyz,
	)
	return context_patches.squeeze(2).bool().any(dim=-1)


def _normalize_attribute_groups(
	attribute_groups: Mapping[str, str] | None,
	num_attributes: int,
) -> Sequence[str] | None:
	if attribute_groups is None:
		return None
	if num_attributes != len(MVP_ATTRIBUTE_REGISTRY.specs):
		msg = (
			'attribute_groups with string keys require the MVP attribute count; '
			f'got num_attributes={num_attributes!r}'
		)
		raise ValueError(msg)
	MVP_ATTRIBUTE_REGISTRY.validate_groups(attribute_groups)
	return tuple(attribute_groups[name] for name in MVP_ATTRIBUTE_REGISTRY.names)


def _validate_positive_int(value: int, name: str) -> int:
	if not isinstance(value, int) or isinstance(value, bool):
		msg = f'{name} must be an integer; got {value!r}'
		raise TypeError(msg)
	if value <= 0:
		msg = f'{name} must be positive; got {value!r}'
		raise ValueError(msg)
	return value


def _validate_patch_size(value: tuple[int, int, int]) -> tuple[int, int, int]:
	if (
		not isinstance(value, tuple)
		or len(value) != 3
		or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
		or any(item <= 0 for item in value)
	):
		msg = f'patch_size_xyz must be a positive integer triple; got {value!r}'
		raise ValueError(msg)
	return value

__all__ = ['StrictAttributeSetMAE3D']
