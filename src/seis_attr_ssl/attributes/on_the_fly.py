"""On-the-fly MVP attribute generation from normalized base seismic crops.

The MVP generator intentionally uses deterministic, dependency-light
approximations. Phase and instantaneous frequency come from a NumPy FFT Hilbert
transform along the z axis. Spectral ratios are whole-trace low/mid/high FFT
energy fractions broadcast back over each trace. Coherence is a local
finite-difference similarity proxy, and GLCM texture channels are quantized
finite-difference contrast/homogeneity proxies rather than full co-occurrence
matrix estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from seis_attr_ssl.attributes.registry import MVP_ATTRIBUTE_REGISTRY
from seis_attr_ssl.data.normalization import (
	SurveyNormalizationStats,
	load_normalization_stats,
	normalize_amplitude,
)

NormalizationStats = SurveyNormalizationStats


@dataclass(frozen=True)
class AttributeGenerationConfig:
	"""Configuration for deterministic MVP attribute approximations."""

	eps: float = 1.0e-6
	spectral_low_max_fraction: float = 0.25
	spectral_mid_max_fraction: float = 0.60
	glcm_levels: int = 16
	output_clip: float = 1.0e6

	def validate(self) -> None:
		"""Validate numeric generation settings."""
		if not isinstance(self.eps, Real) or self.eps <= 0.0:
			msg = f'eps must be a positive real number; got {self.eps!r}'
			raise ValueError(msg)
		if (
			not isinstance(self.spectral_low_max_fraction, Real)
			or not isinstance(self.spectral_mid_max_fraction, Real)
			or not 0.0 < self.spectral_low_max_fraction < 1.0
			or not self.spectral_low_max_fraction
			< self.spectral_mid_max_fraction
			< 1.0
		):
			msg = (
				'spectral band fractions must satisfy '
				'0 < low < mid < 1; got '
				f'{self.spectral_low_max_fraction!r}, '
				f'{self.spectral_mid_max_fraction!r}'
			)
			raise ValueError(msg)
		if not isinstance(self.glcm_levels, Integral) or self.glcm_levels < 2:
			msg = f'glcm_levels must be an integer >= 2; got {self.glcm_levels!r}'
			raise ValueError(msg)
		if not isinstance(self.output_clip, Real) or self.output_clip <= 0.0:
			msg = (
				'output_clip must be a positive real number; '
				f'got {self.output_clip!r}'
			)
			raise ValueError(msg)


@dataclass(frozen=True)
class AttributeGenerationResult:
	"""Generated MVP attributes and validity metadata."""

	attributes: np.ndarray
	attribute_valid: np.ndarray
	voxel_valid_mask: np.ndarray


def generate_mvp_attributes(
	amp_norm: np.ndarray,
	*,
	valid_mask: np.ndarray | None = None,
	config: AttributeGenerationConfig | None = None,
) -> AttributeGenerationResult:
	"""Generate all MVP attributes from one normalized [x, y, z] amplitude crop."""
	cfg = config or AttributeGenerationConfig()
	cfg.validate()
	amplitude, voxel_valid_mask = _prepare_amplitude(amp_norm, valid_mask)

	phase_sin, phase_cos, instantaneous_frequency = _phase_attributes(
		amplitude,
		cfg,
	)
	spectral_low, spectral_mid, spectral_high = _spectral_ratios(amplitude, cfg)
	coherence = _coherence(amplitude)
	glcm_contrast = _glcm_contrast(amplitude, voxel_valid_mask, cfg)
	glcm_homogeneity = _glcm_homogeneity(glcm_contrast)

	attributes = np.stack(
		[
			amplitude,
			phase_sin,
			phase_cos,
			instantaneous_frequency,
			spectral_low,
			spectral_mid,
			spectral_high,
			coherence,
			glcm_contrast,
			glcm_homogeneity,
		],
		axis=0,
	)
	attributes = _sanitize(attributes, cfg)
	attributes *= voxel_valid_mask[np.newaxis, ...]
	return AttributeGenerationResult(
		attributes=attributes.astype(np.float32, copy=False),
		attribute_valid=np.ones(len(MVP_ATTRIBUTE_REGISTRY.specs), dtype=bool),
		voxel_valid_mask=voxel_valid_mask,
	)


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
		attribute_id = MVP_ATTRIBUTE_REGISTRY.name_to_id(name)
	except KeyError as exc:
		msg = f'unknown MVP attribute: {name!r}'
		raise KeyError(msg) from exc
	return generate_mvp_attributes(normalize_base_seismic(base_crop, stats)).attributes[
		attribute_id
	]


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


def _coherence(normalized: np.ndarray) -> np.ndarray:
	return (1.0 / (1.0 + _gradient_magnitude(normalized))).astype(
		np.float32,
		copy=False,
	)


def _glcm_contrast(
	normalized: np.ndarray,
	valid_mask: np.ndarray | None = None,
	config: AttributeGenerationConfig | None = None,
) -> np.ndarray:
	cfg = config or AttributeGenerationConfig()
	cfg.validate()
	mask = (
		np.ones(normalized.shape, dtype=bool)
		if valid_mask is None
		else np.asarray(valid_mask, dtype=bool)
	)
	quantized = _quantize_for_texture(normalized, mask, cfg)
	contrast = (
		np.square(_gradient(quantized, axis=0))
		+ np.square(_gradient(quantized, axis=1))
		+ np.square(_gradient(quantized, axis=2))
	)
	return contrast.astype(np.float32, copy=False)


def _glcm_homogeneity(normalized: np.ndarray) -> np.ndarray:
	contrast = normalized
	return (1.0 / (1.0 + contrast)).astype(np.float32, copy=False)


def _prepare_amplitude(
	amp_norm: np.ndarray,
	valid_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
	amplitude = np.asarray(amp_norm, dtype=np.float32)
	if amplitude.ndim != 3:
		msg = f'amp_norm must be a 3D [x, y, z] array; got ndim={amplitude.ndim}'
		raise ValueError(msg)
	if any(axis <= 0 for axis in amplitude.shape):
		msg = (
			'amp_norm shape must be non-empty in all dimensions; '
			f'got {amplitude.shape!r}'
		)
		raise ValueError(msg)
	if valid_mask is None:
		voxel_valid_mask = np.ones(amplitude.shape, dtype=bool)
	else:
		voxel_valid_mask = np.asarray(valid_mask, dtype=bool)
		if voxel_valid_mask.shape != amplitude.shape:
			msg = (
				'valid_mask shape must match amp_norm shape; got '
				f'{voxel_valid_mask.shape!r} and {amplitude.shape!r}'
			)
			raise ValueError(msg)
	amplitude = np.nan_to_num(
		amplitude,
		nan=0.0,
		posinf=0.0,
		neginf=0.0,
	).astype(np.float32, copy=False)
	amplitude = np.where(voxel_valid_mask, amplitude, np.float32(0.0))
	return amplitude.astype(np.float32, copy=False), voxel_valid_mask


def _phase_attributes(
	amplitude: np.ndarray,
	config: AttributeGenerationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	analytic = _hilbert_z(amplitude)
	phase = np.unwrap(np.angle(analytic), axis=2).astype(np.float32, copy=False)
	phase_sin = np.sin(phase).astype(np.float32, copy=False)
	phase_cos = np.cos(phase).astype(np.float32, copy=False)
	if amplitude.shape[2] < 2:
		instantaneous_frequency = np.zeros_like(amplitude, dtype=np.float32)
	else:
		instantaneous_frequency = (
			np.abs(_gradient(phase, axis=2)) / np.float32(2.0 * np.pi)
		).astype(np.float32, copy=False)
	return (
		_sanitize(phase_sin, config),
		_sanitize(phase_cos, config),
		_sanitize(instantaneous_frequency, config),
	)


def _hilbert_z(amplitude: np.ndarray) -> np.ndarray:
	z_size = amplitude.shape[2]
	spectrum = np.fft.fft(amplitude, axis=2)
	multiplier = np.zeros(z_size, dtype=np.float32)
	if z_size % 2 == 0:
		multiplier[0] = 1.0
		multiplier[z_size // 2] = 1.0
		multiplier[1 : z_size // 2] = 2.0
	else:
		multiplier[0] = 1.0
		multiplier[1 : (z_size + 1) // 2] = 2.0
	return np.fft.ifft(spectrum * multiplier.reshape(1, 1, z_size), axis=2)


def _spectral_ratios(
	amplitude: np.ndarray,
	config: AttributeGenerationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	z_size = amplitude.shape[2]
	if z_size < 2:
		zeros = np.zeros_like(amplitude, dtype=np.float32)
		return zeros, zeros, zeros

	spectrum = np.fft.rfft(amplitude, axis=2)
	power = (
		np.square(spectrum.real) + np.square(spectrum.imag)
	).astype(np.float32, copy=False)
	frequencies = np.fft.rfftfreq(z_size, d=1.0)
	relative = frequencies / frequencies[-1]
	low_mask = relative <= config.spectral_low_max_fraction
	mid_mask = (relative > config.spectral_low_max_fraction) & (
		relative <= config.spectral_mid_max_fraction
	)
	high_mask = relative > config.spectral_mid_max_fraction
	total = power.sum(axis=2, keepdims=True, dtype=np.float32)
	total = total + np.float32(config.eps)
	low = power[..., low_mask].sum(axis=2, keepdims=True, dtype=np.float32) / total
	mid = power[..., mid_mask].sum(axis=2, keepdims=True, dtype=np.float32) / total
	high = power[..., high_mask].sum(axis=2, keepdims=True, dtype=np.float32) / total
	return (
		np.broadcast_to(low, amplitude.shape).astype(np.float32, copy=False),
		np.broadcast_to(mid, amplitude.shape).astype(np.float32, copy=False),
		np.broadcast_to(high, amplitude.shape).astype(np.float32, copy=False),
	)


def _quantize_for_texture(
	amplitude: np.ndarray,
	valid_mask: np.ndarray,
	config: AttributeGenerationConfig,
) -> np.ndarray:
	valid_values = amplitude[valid_mask]
	if valid_values.size == 0:
		return np.zeros_like(amplitude, dtype=np.float32)
	minimum = np.float32(valid_values.min())
	maximum = np.float32(valid_values.max())
	if float(maximum - minimum) <= config.eps:
		return np.zeros_like(amplitude, dtype=np.float32)
	levels = np.float32(config.glcm_levels - 1)
	scaled = (amplitude - minimum) / np.float32(maximum - minimum + config.eps)
	quantized = np.rint(np.clip(scaled, 0.0, 1.0) * levels) / levels
	return np.where(valid_mask, quantized.astype(np.float32), np.float32(0.0))


def _sanitize(
	array: np.ndarray,
	config: AttributeGenerationConfig,
) -> np.ndarray:
	return np.nan_to_num(
		np.clip(array, -config.output_clip, config.output_clip),
		nan=0.0,
		posinf=config.output_clip,
		neginf=-config.output_clip,
	).astype(np.float32, copy=False)


__all__ = [
	'AttributeGenerationConfig',
	'AttributeGenerationResult',
	'NormalizationStats',
	'generate_mvp_attribute',
	'generate_mvp_attributes',
	'load_normalization_stats',
	'normalize_base_seismic',
]
