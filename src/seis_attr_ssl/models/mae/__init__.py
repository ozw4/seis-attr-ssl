"""Masked autoencoder model components."""

from importlib import import_module
from typing import TYPE_CHECKING

from seis_attr_ssl.models.mae.context import ContextTokenPooler
from seis_attr_ssl.models.mae.positional_encoding import (
	build_3d_sincos_position_embedding,
	restore_decoder_sequence,
	select_visible_tokens,
)

if TYPE_CHECKING:
	from seis_attr_ssl.models.mae.model import StrictAttributeSetMAE3D

__all__ = [
	'ContextTokenPooler',
	'StrictAttributeSetMAE3D',
	'build_3d_sincos_position_embedding',
	'restore_decoder_sequence',
	'select_visible_tokens',
]


def __getattr__(name: str) -> object:
	"""Lazily expose the full MAE model without creating import cycles."""
	if name == 'StrictAttributeSetMAE3D':
		return import_module('seis_attr_ssl.models.mae.model').StrictAttributeSetMAE3D
	msg = f'module {__name__!r} has no attribute {name!r}'
	raise AttributeError(msg)
