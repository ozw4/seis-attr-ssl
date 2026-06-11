"""Attribute-set patch tokenizer for strict 3D MAE inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn

from seis_attr_ssl.models.mae.patching import compute_num_patches, patchify_3d


class AttributePatchTokenizer3D(nn.Module):
	"""Patchify selected attribute volumes and fuse them into spatial tokens."""

	def __init__(  # noqa: PLR0913
		self,
		*,
		num_attributes: int,
		attribute_groups: Sequence[str] | Mapping[int, str] | None,
		patch_size_xyz: tuple[int, int, int],
		embed_dim: int,
		use_group_embedding: bool = True,
		fusion: str = 'mean',
	) -> None:
		"""Initialize the tokenizer projection and attribute embeddings."""
		super().__init__()
		if not isinstance(num_attributes, int) or isinstance(num_attributes, bool):
			msg = f'num_attributes must be an integer; got {num_attributes!r}'
			raise TypeError(msg)
		if num_attributes <= 0:
			msg = f'num_attributes must be positive; got {num_attributes!r}'
			raise ValueError(msg)
		if not isinstance(embed_dim, int) or isinstance(embed_dim, bool):
			msg = f'embed_dim must be an integer; got {embed_dim!r}'
			raise TypeError(msg)
		if embed_dim <= 0:
			msg = f'embed_dim must be positive; got {embed_dim!r}'
			raise ValueError(msg)
		if fusion != 'mean':
			msg = f"fusion must be 'mean'; got {fusion!r}"
			raise ValueError(msg)

		patch_volume = _validate_patch_size(patch_size_xyz)
		self.num_attributes = num_attributes
		self.patch_size_xyz = patch_size_xyz
		self.embed_dim = embed_dim
		self.fusion = fusion
		self.patch_projection = nn.Linear(patch_volume, embed_dim)
		self.attribute_embedding = nn.Embedding(num_attributes, embed_dim)

		if use_group_embedding:
			group_ids, num_groups = _build_attribute_group_ids(
				attribute_groups,
				num_attributes,
			)
			self.group_embedding: nn.Embedding | None = nn.Embedding(
				num_groups,
				embed_dim,
			)
			self.register_buffer(
				'attribute_group_ids',
				torch.tensor(group_ids, dtype=torch.long),
				persistent=False,
			)
		else:
			self.group_embedding = None
			self.register_buffer(
				'attribute_group_ids',
				torch.empty((0,), dtype=torch.long),
				persistent=False,
			)

	def forward(
		self,
		x: torch.Tensor,
		attribute_ids: torch.Tensor,
		attribute_valid_mask: torch.Tensor | None = None,
	) -> dict[str, torch.Tensor | tuple[int, int, int]]:
		"""Return fused spatial tokens and their XYZ token grid shape."""
		if x.ndim != 5:
			msg = (
				'x must be a 5D tensor with shape [B, C, X, Y, Z]; '
				f'got shape={tuple(x.shape)!r}'
			)
			raise ValueError(msg)

		batch_size, channels, x_size, y_size, z_size = x.shape
		_validate_attribute_ids_shape(attribute_ids, batch_size, channels)
		valid_mask = _validate_attribute_valid_mask(
			attribute_valid_mask,
			batch_size,
			channels,
			attribute_ids.device,
		)
		_validate_attribute_ids_range(
			attribute_ids,
			self.num_attributes,
			valid_mask,
		)
		if not valid_mask.any(dim=1).all():
			msg = 'each sample must contain at least one valid attribute'
			raise ValueError(msg)

		safe_attribute_ids = attribute_ids.masked_fill(~valid_mask, 0).long()
		grid_shape = compute_num_patches(
			(int(x_size), int(y_size), int(z_size)),
			self.patch_size_xyz,
		)[:3]
		if not valid_mask.all():
			return {
				'tokens': self._tokenize_padded_attributes(
					x,
					safe_attribute_ids,
					valid_mask,
				),
				'token_grid_shape': grid_shape,
			}

		tokens = self._tokenize_dense_attributes(x, safe_attribute_ids)
		return {'tokens': tokens, 'token_grid_shape': grid_shape}

	def _tokenize_dense_attributes(
		self,
		x: torch.Tensor,
		attribute_ids: torch.Tensor,
	) -> torch.Tensor:
		patches = patchify_3d(x, self.patch_size_xyz)
		attribute_tokens = self.patch_projection(patches)
		attribute_tokens = attribute_tokens + self.attribute_embedding(
			attribute_ids,
		).unsqueeze(1)

		if self.group_embedding is not None:
			group_ids = self.attribute_group_ids.to(attribute_ids.device)[
				attribute_ids
			]
			attribute_tokens = attribute_tokens + self.group_embedding(
				group_ids,
			).unsqueeze(1)

		return attribute_tokens.mean(dim=2)

	def _tokenize_padded_attributes(
		self,
		x: torch.Tensor,
		attribute_ids: torch.Tensor,
		valid_mask: torch.Tensor,
	) -> torch.Tensor:
		sample_tokens = []
		for sample_x, sample_attribute_ids, sample_valid_mask in zip(
			x,
			attribute_ids,
			valid_mask,
			strict=True,
		):
			valid_x = sample_x[sample_valid_mask].unsqueeze(0)
			valid_attribute_ids = sample_attribute_ids[sample_valid_mask].unsqueeze(0)
			valid_tokens = self._tokenize_dense_attributes(
				valid_x,
				valid_attribute_ids,
			)
			sample_tokens.append(
				valid_tokens.squeeze(0),
			)
		return torch.stack(sample_tokens, dim=0)


def _validate_patch_size(patch_size_xyz: tuple[int, int, int]) -> int:
	if (
		not isinstance(patch_size_xyz, tuple)
		or len(patch_size_xyz) != 3
		or any(
			not isinstance(item, int) or isinstance(item, bool)
			for item in patch_size_xyz
		)
		or any(item <= 0 for item in patch_size_xyz)
	):
		msg = (
			'patch_size_xyz must be a positive integer triple; '
			f'got {patch_size_xyz!r}'
		)
		raise ValueError(msg)
	px_size, py_size, pz_size = patch_size_xyz
	return px_size * py_size * pz_size


def _build_attribute_group_ids(
	attribute_groups: Sequence[str] | Mapping[int, str] | None,
	num_attributes: int,
) -> tuple[tuple[int, ...], int]:
	if attribute_groups is None:
		msg = 'attribute_groups must be provided when use_group_embedding=True'
		raise ValueError(msg)

	if isinstance(attribute_groups, Mapping):
		group_by_attribute = _group_sequence_from_mapping(
			attribute_groups,
			num_attributes,
		)
	else:
		group_by_attribute = tuple(attribute_groups)
		if len(group_by_attribute) != num_attributes:
			msg = (
				'attribute_groups sequence length must equal num_attributes; '
				f'got len={len(group_by_attribute)!r}, '
				f'num_attributes={num_attributes!r}'
			)
			raise ValueError(msg)

	group_to_id: dict[str, int] = {}
	group_ids: list[int] = []
	for attribute_index, group in enumerate(group_by_attribute):
		if not isinstance(group, str) or not group:
			msg = (
				'attribute group names must be non-empty strings; '
				f'got attribute_index={attribute_index!r}, group={group!r}'
			)
			raise ValueError(msg)
		group_id = group_to_id.setdefault(group, len(group_to_id))
		group_ids.append(group_id)

	return tuple(group_ids), len(group_to_id)


def _group_sequence_from_mapping(
	attribute_groups: Mapping[int, str],
	num_attributes: int,
) -> tuple[str, ...]:
	expected_keys = set(range(num_attributes))
	actual_keys = set(attribute_groups)
	if actual_keys != expected_keys:
		missing = tuple(sorted(expected_keys - actual_keys))
		extra = tuple(sorted(actual_keys - expected_keys))
		msg = (
			'attribute_groups mapping must contain exactly attribute IDs '
			f'0..{num_attributes - 1}; missing={missing!r}, extra={extra!r}'
		)
		raise ValueError(msg)
	return tuple(attribute_groups[index] for index in range(num_attributes))


def _validate_attribute_ids_shape(
	attribute_ids: torch.Tensor,
	batch_size: int,
	channels: int,
) -> None:
	if attribute_ids.ndim != 2 or tuple(attribute_ids.shape) != (
		batch_size,
		channels,
	):
		msg = (
			'attribute_ids must have shape [B, C] matching x; '
			f'got shape={tuple(attribute_ids.shape)!r}, '
			f'expected={(batch_size, channels)!r}'
		)
		raise ValueError(msg)
	if (
		attribute_ids.dtype == torch.bool
		or torch.is_floating_point(attribute_ids)
		or torch.is_complex(attribute_ids)
	):
		msg = (
			'attribute_ids must have an integer dtype; '
			f'got dtype={attribute_ids.dtype}'
		)
		raise TypeError(msg)


def _validate_attribute_valid_mask(
	attribute_valid_mask: torch.Tensor | None,
	batch_size: int,
	channels: int,
	device: torch.device,
) -> torch.Tensor:
	if attribute_valid_mask is None:
		return torch.ones((batch_size, channels), dtype=torch.bool, device=device)
	if attribute_valid_mask.ndim != 2 or tuple(attribute_valid_mask.shape) != (
		batch_size,
		channels,
	):
		msg = (
			'attribute_valid_mask must have shape [B, C] matching x; '
			f'got shape={tuple(attribute_valid_mask.shape)!r}, '
			f'expected={(batch_size, channels)!r}'
		)
		raise ValueError(msg)
	if attribute_valid_mask.dtype != torch.bool:
		msg = (
			'attribute_valid_mask dtype must be bool; '
			f'got dtype={attribute_valid_mask.dtype}'
		)
		raise TypeError(msg)
	if attribute_valid_mask.device != device:
		msg = (
			'attribute_valid_mask must be on the same device as attribute_ids; '
			f'got mask_device={attribute_valid_mask.device}, ids_device={device}'
		)
		raise ValueError(msg)
	return attribute_valid_mask


def _validate_attribute_ids_range(
	attribute_ids: torch.Tensor,
	num_attributes: int,
	valid_mask: torch.Tensor,
) -> None:
	valid_attribute_ids = attribute_ids[valid_mask]
	if valid_attribute_ids.numel() == 0:
		return
	if (
		valid_attribute_ids.min().item() < 0
		or valid_attribute_ids.max().item() >= num_attributes
	):
		msg = (
			'attribute_ids must be in range [0, num_attributes); '
			f'got num_attributes={num_attributes!r}'
		)
		raise ValueError(msg)


__all__ = ['AttributePatchTokenizer3D']
