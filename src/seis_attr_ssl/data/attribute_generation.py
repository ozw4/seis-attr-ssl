"""Public data-layer entry points for on-the-fly MVP attribute generation."""

from __future__ import annotations

from seis_attr_ssl.attributes.on_the_fly import (
	AttributeGenerationConfig,
	AttributeGenerationResult,
	center_trim_attribute_result,
	generate_mvp_attributes,
	generate_mvp_attributes_for_payload,
)

__all__ = [
	'AttributeGenerationConfig',
	'AttributeGenerationResult',
	'center_trim_attribute_result',
	'generate_mvp_attributes',
	'generate_mvp_attributes_for_payload',
]
