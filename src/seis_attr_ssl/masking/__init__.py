"""Masking utilities and shared MAE mask contracts."""

from seis_attr_ssl.masking.schema import MaskingPlan
from seis_attr_ssl.masking.spatial import (
    compute_token_grid_shape,
    generate_spatial_block_mask,
)
from seis_attr_ssl.masking.validation import (
    registry_attribute_ids,
    validate_masking_plan,
)

__all__ = [
    'MaskingPlan',
    'compute_token_grid_shape',
    'generate_spatial_block_mask',
    'registry_attribute_ids',
    'validate_masking_plan',
]
