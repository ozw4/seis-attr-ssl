"""Masking utilities and shared MAE mask contracts."""

from seis_attr_ssl.masking.attributes import (
    AMPLITUDE_ATTRIBUTE_ID,
    MVP_ATTRIBUTE_IDS,
    build_attribute_group_index,
    sample_attribute_input_mask,
)
from seis_attr_ssl.masking.mae import build_mae_masking_plan
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
    'AMPLITUDE_ATTRIBUTE_ID',
    'MVP_ATTRIBUTE_IDS',
    'MaskingPlan',
    'build_attribute_group_index',
    'build_mae_masking_plan',
    'compute_token_grid_shape',
    'generate_spatial_block_mask',
    'registry_attribute_ids',
    'sample_attribute_input_mask',
    'validate_masking_plan',
]
