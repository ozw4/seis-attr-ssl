"""Public data-layer entry points for on-the-fly MVP attribute generation."""

from __future__ import annotations

from seis_attr_ssl.attributes.on_the_fly import (
	AttributeGenerationConfig,
	AttributeGenerationResult,
	generate_mvp_attributes,
)

__all__ = [
	'AttributeGenerationConfig',
	'AttributeGenerationResult',
	'generate_mvp_attributes',
]
