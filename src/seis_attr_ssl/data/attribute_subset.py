"""Attribute subset sampling for MVP pretraining batches."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from seis_attr_ssl.attributes import MVP_ATTRIBUTE_REGISTRY
from seis_attr_ssl.masking.attributes import (
	AMPLITUDE_ATTRIBUTE_ID,
	MVP_ATTRIBUTE_IDS,
	sample_attribute_input_mask,
)

if TYPE_CHECKING:
	from collections.abc import Sequence


def sample_attribute_subset(
	available_attribute_ids: Sequence[int],
	min_input_attributes: int,
	max_input_attributes: int,
	rng: np.random.Generator,
) -> tuple[int, ...]:
	"""Sample a registry-ordered input attribute subset.

	The MVP amplitude channel is required and always included. If fewer than
	``min_input_attributes`` are available, a clear error is raised.
	"""
	input_mask = sample_attribute_input_mask(
		available_attribute_ids,
		MVP_ATTRIBUTE_IDS,
		MVP_ATTRIBUTE_REGISTRY.groups,
		min_input_attributes,
		max_input_attributes,
		attribute_dropout_prob=0.0,
		group_dropout_prob=0.0,
		rng=rng,
	)
	return tuple(int(id_) for id_ in np.flatnonzero(input_mask))


__all__ = [
	'AMPLITUDE_ATTRIBUTE_ID',
	'MVP_ATTRIBUTE_IDS',
	'sample_attribute_subset',
]
