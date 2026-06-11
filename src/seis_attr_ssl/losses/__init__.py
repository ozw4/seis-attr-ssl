"""Loss functions."""

from seis_attr_ssl.losses.gradient import gradient_loss_xyz
from seis_attr_ssl.losses.mae_reconstruction import (
	dropped_attribute_reconstruction_loss,
	mae_pretraining_loss,
	masked_patch_reconstruction_loss,
)

__all__ = [
	'dropped_attribute_reconstruction_loss',
	'gradient_loss_xyz',
	'mae_pretraining_loss',
	'masked_patch_reconstruction_loss',
]
