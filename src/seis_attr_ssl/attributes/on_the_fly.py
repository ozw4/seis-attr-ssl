"""On-the-fly MVP attribute generation from normalized base seismic crops."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from seis_attr_ssl.attributes.registry import MVP_ATTRIBUTE_REGISTRY
from seis_attr_ssl.data.normalization import (
	SurveyNormalizationStats,
	load_normalization_stats,
	normalize_amplitude,
)

if TYPE_CHECKING:
	from collections.abc import Callable

NormalizationStats = SurveyNormalizationStats


def generate_mvp_attribute(
	base_crop: np.ndarray,
	attribute_name_or_id: str | int,
	stats: SurveyNormalizationStats,
) -> np.ndarray:
	"""Generate one MVP attribute volume from a base seismic crop."""
	name = (
		MVP_ATTRIBUTE_REGISTRY.id_to_name(attribute_name_or_id)
		if isinstance(attribute_name_or_id, int)
		else attribute_name_or_id
	)
	try:
		generator = _ATTRIBUTE_GENERATORS[name]
	except KeyError as exc:
		msg = f'unknown MVP attribute: {name!r}'
		raise KeyError(msg) from exc
	return generator(normalize_base_seismic(base_crop, stats))


def normalize_base_seismic(
	base_crop: np.ndarray,
	stats: SurveyNormalizationStats,
) -> np.ndarray:
	"""Apply survey-wise robust normalization to a base seismic crop."""
	return normalize_amplitude(base_crop, stats)


def _gradient_magnitude(array: np.ndarray) -> np.ndarray:
	grad_x = _gradient(array, axis=0)
	grad_y = _gradient(array, axis=1)
	grad_z = _gradient(array, axis=2)
	return np.sqrt(np.square(grad_x) + np.square(grad_y) + np.square(grad_z)).astype(
		np.float32,
		copy=False,
	)


def _gradient(array: np.ndarray, *, axis: int) -> np.ndarray:
	if array.shape[axis] < 2:
		return np.zeros_like(array, dtype=np.float32)
	return np.gradient(array, axis=axis).astype(np.float32, copy=False)


def _amplitude_norm(normalized: np.ndarray) -> np.ndarray:
	return normalized


def _phase_sin(normalized: np.ndarray) -> np.ndarray:
	return np.sin(normalized).astype(np.float32, copy=False)


def _phase_cos(normalized: np.ndarray) -> np.ndarray:
	return np.cos(normalized).astype(np.float32, copy=False)


def _instantaneous_frequency(normalized: np.ndarray) -> np.ndarray:
	return np.abs(_gradient(normalized, axis=2)).astype(np.float32, copy=False)


def _spectral_low_ratio(normalized: np.ndarray) -> np.ndarray:
	return (1.0 / (1.0 + np.abs(_gradient(normalized, axis=2)))).astype(
		np.float32,
		copy=False,
	)


def _spectral_mid_ratio(normalized: np.ndarray) -> np.ndarray:
	grad_z = np.abs(_gradient(normalized, axis=2))
	return (grad_z / (1.0 + np.abs(normalized) + grad_z)).astype(
		np.float32,
		copy=False,
	)


def _spectral_high_ratio(normalized: np.ndarray) -> np.ndarray:
	grad_mag = _gradient_magnitude(normalized)
	return (grad_mag / (1.0 + grad_mag)).astype(np.float32, copy=False)


def _coherence(normalized: np.ndarray) -> np.ndarray:
	return (1.0 / (1.0 + _gradient_magnitude(normalized))).astype(
		np.float32,
		copy=False,
	)


def _glcm_contrast(normalized: np.ndarray) -> np.ndarray:
	contrast = np.square(_gradient(normalized, axis=0)) + np.square(
		_gradient(normalized, axis=1),
	)
	return contrast.astype(np.float32, copy=False)


def _glcm_homogeneity(normalized: np.ndarray) -> np.ndarray:
	contrast = _glcm_contrast(normalized)
	return (1.0 / (1.0 + contrast)).astype(np.float32, copy=False)


_ATTRIBUTE_GENERATORS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
	'amplitude_norm': _amplitude_norm,
	'phase_sin': _phase_sin,
	'phase_cos': _phase_cos,
	'instantaneous_frequency': _instantaneous_frequency,
	'spectral_low_ratio': _spectral_low_ratio,
	'spectral_mid_ratio': _spectral_mid_ratio,
	'spectral_high_ratio': _spectral_high_ratio,
	'coherence': _coherence,
	'glcm_contrast': _glcm_contrast,
	'glcm_homogeneity': _glcm_homogeneity,
}


__all__ = [
	'NormalizationStats',
	'generate_mvp_attribute',
	'load_normalization_stats',
	'normalize_base_seismic',
]
