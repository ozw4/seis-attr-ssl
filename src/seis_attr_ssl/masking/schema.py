"""Common masking schema for strict 3D attribute-set MAE.

Grid order is always ``[x, y, z]`` for spatial token masks.

Boolean mask conventions:

- ``spatial_mask[x, y, z] == True`` means the spatial token is masked and
  should be reconstructed by MAE.
- ``visible_spatial_mask[x, y, z] == True`` means the spatial token is visible
  to the encoder.
- ``attribute_input_mask[a] == True`` means registry attribute ``a`` is
  available as model input for this sample.
- ``attribute_target_mask[a] == True`` means registry attribute ``a`` can be
  used as a reconstruction target.
- ``dropped_attribute_mask[a] == True`` means registry attribute ``a`` is valid
  as a target but was withheld from input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	import numpy as np


@dataclass(frozen=True)
class MaskingPlan:
	"""Mask contract shared by masking samplers, datasets, and MAE training."""

	spatial_mask: np.ndarray
	visible_spatial_mask: np.ndarray
	attribute_input_mask: np.ndarray
	attribute_target_mask: np.ndarray
	dropped_attribute_mask: np.ndarray
	input_attribute_ids: np.ndarray
	target_attribute_ids: np.ndarray


__all__ = ['MaskingPlan']
