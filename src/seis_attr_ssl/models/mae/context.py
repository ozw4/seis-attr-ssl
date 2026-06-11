"""Context token pooling for local-plus-context masked autoencoders."""

from __future__ import annotations

import torch
from torch import nn


class ContextTokenPooler(nn.Module):
	"""Pool spatial context tokens into a fixed set of learned context tokens."""

	def __init__(
		self,
		*,
		embed_dim: int,
		num_context_tokens: int,
		num_heads: int = 4,
	) -> None:
		"""Initialize learned context queries and cross-attention."""
		super().__init__()
		self.embed_dim = _validate_positive_int(embed_dim, 'embed_dim')
		self.num_context_tokens = _validate_positive_int(
			num_context_tokens,
			'num_context_tokens',
		)
		self.num_heads = _validate_positive_int(num_heads, 'num_heads')
		if self.embed_dim % self.num_heads != 0:
			msg = (
				'embed_dim must be divisible by num_heads; '
				f'got embed_dim={self.embed_dim!r}, num_heads={self.num_heads!r}'
			)
			raise ValueError(msg)

		self.query_tokens = nn.Parameter(
			torch.empty(self.num_context_tokens, self.embed_dim),
		)
		self.attention = nn.MultiheadAttention(
			self.embed_dim,
			self.num_heads,
			batch_first=True,
		)
		self.norm = nn.LayerNorm(self.embed_dim)
		self.reset_parameters()

	def reset_parameters(self) -> None:
		"""Reset learnable query tokens."""
		nn.init.normal_(self.query_tokens, std=0.02)

	def forward(
		self,
		context_tokens: torch.Tensor,
		context_token_valid_mask: torch.Tensor | None = None,
	) -> torch.Tensor:
		"""Return pooled context tokens with shape ``[B, K, D]``.

		``context_token_valid_mask`` is expected to already be reduced to token
		shape ``[B, N_context]``, for example by reducing a voxel-level
		``context_valid_mask`` over the same patch grid used to tokenize context.
		"""
		batch_size, num_context_tokens = _validate_context_tokens(
			context_tokens,
			self.embed_dim,
		)
		valid_mask = _validate_context_token_valid_mask(
			context_token_valid_mask,
			batch_size,
			num_context_tokens,
			context_tokens.device,
		)
		if valid_mask is not None and not valid_mask.any(dim=1).all():
			msg = 'each sample must contain at least one valid context token'
			raise ValueError(msg)

		query_tokens = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
		pooled_tokens, _attention_weights = self.attention(
			query_tokens,
			context_tokens,
			context_tokens,
			key_padding_mask=None if valid_mask is None else ~valid_mask,
			need_weights=False,
		)
		return self.norm(pooled_tokens)


def _validate_positive_int(value: int, name: str) -> int:
	if not isinstance(value, int) or isinstance(value, bool):
		msg = f'{name} must be an integer; got {value!r}'
		raise TypeError(msg)
	if value <= 0:
		msg = f'{name} must be positive; got {value!r}'
		raise ValueError(msg)
	return value


def _validate_context_tokens(
	context_tokens: torch.Tensor,
	embed_dim: int,
) -> tuple[int, int]:
	if context_tokens.ndim != 3:
		msg = (
			'context_tokens must be a 3D tensor with shape [B, N_context, D]; '
			f'got shape={tuple(context_tokens.shape)!r}'
		)
		raise ValueError(msg)

	batch_size, num_context_tokens, token_embed_dim = context_tokens.shape
	if token_embed_dim != embed_dim:
		msg = (
			'context_tokens last dimension must equal embed_dim; '
			f'got shape={tuple(context_tokens.shape)!r}, embed_dim={embed_dim!r}'
		)
		raise ValueError(msg)
	if num_context_tokens <= 0:
		msg = 'context_tokens must contain at least one token'
		raise ValueError(msg)
	return int(batch_size), int(num_context_tokens)


def _validate_context_token_valid_mask(
	context_token_valid_mask: torch.Tensor | None,
	batch_size: int,
	num_context_tokens: int,
	device: torch.device,
) -> torch.Tensor | None:
	if context_token_valid_mask is None:
		return None
	if context_token_valid_mask.ndim != 2 or tuple(
		context_token_valid_mask.shape,
	) != (
		batch_size,
		num_context_tokens,
	):
		msg = (
			'context_token_valid_mask must have shape [B, N_context]; '
			f'got shape={tuple(context_token_valid_mask.shape)!r}, '
			f'expected={(batch_size, num_context_tokens)!r}'
		)
		raise ValueError(msg)
	if context_token_valid_mask.dtype != torch.bool:
		msg = (
			'context_token_valid_mask dtype must be bool; '
			f'got dtype={context_token_valid_mask.dtype}'
		)
		raise TypeError(msg)
	if context_token_valid_mask.device != device:
		msg = (
			'context_token_valid_mask must be on the same device as context_tokens; '
			f'got mask_device={context_token_valid_mask.device}, '
			f'tokens_device={device}'
		)
		raise ValueError(msg)
	return context_token_valid_mask


__all__ = ['ContextTokenPooler']
