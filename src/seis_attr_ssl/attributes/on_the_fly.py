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
from operator import index

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
	phase_reflect_pad_z: int = 64
	phase_taper_fraction: float = 0.05
	instantaneous_frequency_smooth_z: int = 5
	instantaneous_frequency_envelope_quantile: float = 0.05
	instantaneous_frequency_clip_percentile: float = 99.5
	spectral_local_window_z: int = 65
	spectral_remove_dc: bool = True

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
		_validate_nonnegative_int(self.phase_reflect_pad_z, 'phase_reflect_pad_z')
		if (
			not isinstance(self.phase_taper_fraction, Real)
			or not 0.0 <= self.phase_taper_fraction < 0.5
		):
			msg = (
				'phase_taper_fraction must satisfy 0.0 <= fraction < 0.5; '
				f'got {self.phase_taper_fraction!r}'
			)
			raise ValueError(msg)
		_validate_odd_positive_int(
			self.instantaneous_frequency_smooth_z,
			'instantaneous_frequency_smooth_z',
		)
		if (
			not isinstance(self.instantaneous_frequency_envelope_quantile, Real)
			or not 0.0 <= self.instantaneous_frequency_envelope_quantile < 0.5
		):
			msg = (
				'instantaneous_frequency_envelope_quantile must satisfy '
				'0.0 <= quantile < 0.5; got '
				f'{self.instantaneous_frequency_envelope_quantile!r}'
			)
			raise ValueError(msg)
		if (
			not isinstance(self.instantaneous_frequency_clip_percentile, Real)
			or not 50.0 <= self.instantaneous_frequency_clip_percentile <= 100.0
		):
			msg = (
				'instantaneous_frequency_clip_percentile must satisfy '
				'50.0 <= percentile <= 100.0; got '
				f'{self.instantaneous_frequency_clip_percentile!r}'
			)
			raise ValueError(msg)
		_validate_odd_positive_int(
			self.spectral_local_window_z,
			'spectral_local_window_z',
		)
		if self.spectral_local_window_z < 3:
			msg = (
				'spectral_local_window_z must be an odd integer >= 3; '
				f'got {self.spectral_local_window_z!r}'
			)
			raise ValueError(msg)
		if not isinstance(self.spectral_remove_dc, bool):
			msg = f'spectral_remove_dc must be a bool; got {self.spectral_remove_dc!r}'
			raise TypeError(msg)


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


def center_trim_attribute_result(
	result: AttributeGenerationResult,
	payload_slices_xyz: tuple[slice, slice, slice],
) -> AttributeGenerationResult:
	"""Trim generated attributes and voxel_valid_mask to payload slices."""
	if result.attributes.ndim != 4:
		msg = (
			'result.attributes must be a 4D [attribute, x, y, z] array; '
			f'got ndim={result.attributes.ndim}'
		)
		raise ValueError(msg)
	compute_shape_xyz = result.attributes.shape[1:]
	if result.voxel_valid_mask.shape != compute_shape_xyz:
		msg = (
			'result.voxel_valid_mask shape must match result.attributes spatial '
			f'shape; got {result.voxel_valid_mask.shape!r} and {compute_shape_xyz!r}'
		)
		raise ValueError(msg)
	slices_xyz = _validate_payload_slices_xyz(
		payload_slices_xyz,
		compute_shape_xyz,
	)
	return AttributeGenerationResult(
		attributes=result.attributes[(slice(None), *slices_xyz)],
		attribute_valid=result.attribute_valid,
		voxel_valid_mask=result.voxel_valid_mask[slices_xyz],
	)


def generate_mvp_attributes_for_payload(
	amp_norm_compute: np.ndarray,
	payload_slices_xyz: tuple[slice, slice, slice],
	*,
	valid_mask: np.ndarray | None = None,
	config: AttributeGenerationConfig | None = None,
) -> AttributeGenerationResult:
	"""Generate MVP attributes on a compute crop and return only payload slices."""
	result = generate_mvp_attributes(
		amp_norm_compute,
		valid_mask=valid_mask,
		config=config,
	)
	return center_trim_attribute_result(result, payload_slices_xyz)


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


def _validate_payload_slices_xyz(
	payload_slices_xyz: tuple[slice, slice, slice],
	compute_shape_xyz: tuple[int, int, int],
) -> tuple[slice, slice, slice]:
	try:
		slices_xyz = tuple(payload_slices_xyz)
	except TypeError as exc:
		msg = 'payload_slices_xyz must be a sequence of three slices'
		raise ValueError(msg) from exc
	if len(slices_xyz) != 3:
		msg = f'payload_slices_xyz must contain three slices; got {len(slices_xyz)}'
		raise ValueError(msg)
	for axis, (payload_slice, axis_size) in enumerate(
		zip(slices_xyz, compute_shape_xyz, strict=True),
	):
		if type(payload_slice) is not slice:
			msg = (
				'payload_slices_xyz must contain only slice objects; '
				f'axis {axis} got {type(payload_slice).__name__}'
			)
			raise ValueError(msg)
		if payload_slice.start is None or payload_slice.stop is None:
			msg = (
				'payload_slices_xyz slices must have explicit start and stop; '
				f'axis {axis} got {payload_slice!r}'
			)
			raise ValueError(msg)
		if payload_slice.step not in (None, 1):
			msg = (
				'payload_slices_xyz slices must have step None or 1; '
				f'axis {axis} got {payload_slice!r}'
			)
			raise ValueError(msg)
		try:
			start = index(payload_slice.start)
			stop = index(payload_slice.stop)
		except TypeError as exc:
			msg = (
				'payload_slices_xyz slice start and stop must be integers; '
				f'axis {axis} got {payload_slice!r}'
			)
			raise ValueError(msg) from exc
		if start < 0 or stop > axis_size:
			msg = (
				'payload_slices_xyz slices must be within the compute crop; '
				f'axis {axis} got {payload_slice!r} for size {axis_size}'
			)
			raise ValueError(msg)
		if stop <= start:
			msg = (
				'payload_slices_xyz slices must be non-empty; '
				f'axis {axis} got {payload_slice!r}'
			)
			raise ValueError(msg)
	return slices_xyz


def _validate_odd_positive_int(value: int, name: str) -> int:
	"""Return value as int after validating it is odd and positive."""
	if isinstance(value, bool):
		msg = f'{name} must be an odd positive integer; got {value!r}'
		raise TypeError(msg)
	try:
		integer = index(value)
	except TypeError as exc:
		msg = f'{name} must be an odd positive integer; got {value!r}'
		raise TypeError(msg) from exc
	if integer < 1 or integer % 2 == 0:
		msg = f'{name} must be an odd positive integer; got {value!r}'
		raise ValueError(msg)
	return integer


def _validate_nonnegative_int(value: int, name: str) -> int:
	"""Return value as int after validating it is non-negative."""
	if isinstance(value, bool):
		msg = f'{name} must be a non-negative integer; got {value!r}'
		raise TypeError(msg)
	try:
		integer = index(value)
	except TypeError as exc:
		msg = f'{name} must be a non-negative integer; got {value!r}'
		raise TypeError(msg) from exc
	if integer < 0:
		msg = f'{name} must be a non-negative integer; got {value!r}'
		raise ValueError(msg)
	return integer


def _safe_percentile(values: np.ndarray, percentile: float, default: float) -> float:
	"""Compute a percentile over finite values, falling back when none exist."""
	array = np.asarray(values, dtype=np.float32)
	finite = array[np.isfinite(array)]
	if finite.size == 0:
		return float(default)
	return float(np.percentile(finite, percentile))


def _smooth_z_mean(array: np.ndarray, window_z: int) -> np.ndarray:
	"""Mean-smooth a 3D [x, y, z] array along z while preserving shape."""
	window = _validate_odd_positive_int(window_z, 'window_z')
	values = np.asarray(array, dtype=np.float32)
	if values.ndim != 3:
		msg = f'array must be a 3D [x, y, z] array; got ndim={values.ndim}'
		raise ValueError(msg)
	if values.shape[2] == 0 or window == 1:
		return values.astype(np.float32, copy=True)
	pad = window // 2
	padded = np.pad(values, ((0, 0), (0, 0), (pad, pad)), mode='edge')
	zero = np.zeros((*padded.shape[:2], 1), dtype=np.float32)
	cumsum = np.cumsum(
		np.concatenate((zero, padded), axis=2),
		axis=2,
		dtype=np.float32,
	)
	smoothed = (cumsum[..., window:] - cumsum[..., :-window]) / np.float32(window)
	return smoothed.astype(np.float32, copy=False)


def _hann_taper_z(array: np.ndarray, fraction: float) -> np.ndarray:
	"""Apply a symmetric Hann taper to the leading/trailing z edges."""
	values = np.asarray(array, dtype=np.float32)
	if values.ndim != 3:
		msg = f'array must be a 3D [x, y, z] array; got ndim={values.ndim}'
		raise ValueError(msg)
	if not isinstance(fraction, Real) or not 0.0 <= fraction < 0.5:
		msg = f'fraction must satisfy 0.0 <= fraction < 0.5; got {fraction!r}'
		raise ValueError(msg)
	z_size = values.shape[2]
	taper_size = int(np.floor(z_size * float(fraction)))
	if taper_size < 1:
		return values.astype(np.float32, copy=True)
	window = np.ones(z_size, dtype=np.float32)
	edge = np.hanning((2 * taper_size) + 2).astype(np.float32)[1 : taper_size + 1]
	window[:taper_size] = edge
	window[-taper_size:] = edge[::-1]
	return (values * window.reshape(1, 1, z_size)).astype(np.float32, copy=False)


def _reflect_pad_z(array: np.ndarray, pad_z: int) -> tuple[np.ndarray, int]:
	"""Reflect-pad a 3D [x, y, z] array along z and return the applied pad."""
	pad = _validate_nonnegative_int(pad_z, 'pad_z')
	values = np.asarray(array, dtype=np.float32)
	if values.ndim != 3:
		msg = f'array must be a 3D [x, y, z] array; got ndim={values.ndim}'
		raise ValueError(msg)
	if pad == 0 or values.shape[2] == 0:
		return values.astype(np.float32, copy=True), 0
	mode = 'reflect' if values.shape[2] > 1 else 'edge'
	padded = np.pad(values, ((0, 0), (0, 0), (pad, pad)), mode=mode)
	return padded.astype(np.float32, copy=False), pad


def _unpad_z(array: np.ndarray, pad_z: int) -> np.ndarray:
	"""Remove symmetric z padding from a 3D [x, y, z] array."""
	pad = _validate_nonnegative_int(pad_z, 'pad_z')
	values = np.asarray(array)
	if values.ndim != 3:
		msg = f'array must be a 3D [x, y, z] array; got ndim={values.ndim}'
		raise ValueError(msg)
	if pad == 0:
		return values.copy()
	if values.shape[2] < 2 * pad:
		msg = (
			'array z dimension is too small to remove requested padding; '
			f'got z={values.shape[2]} and pad_z={pad}'
		)
		raise ValueError(msg)
	return values[..., pad:-pad]


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
	analytic = _hilbert_z(amplitude, config)
	phase = np.unwrap(np.angle(analytic), axis=2).astype(np.float32, copy=False)
	envelope = np.abs(analytic).astype(np.float32, copy=False)
	phase_sin = np.sin(phase).astype(np.float32, copy=False)
	phase_cos = np.cos(phase).astype(np.float32, copy=False)
	if amplitude.shape[2] < 2:
		instantaneous_frequency = np.zeros_like(amplitude, dtype=np.float32)
	else:
		instantaneous_frequency = (
			np.abs(_gradient(phase, axis=2)) / np.float32(2.0 * np.pi)
		).astype(np.float32, copy=False)
	instantaneous_frequency = _smooth_z_mean(
		instantaneous_frequency,
		config.instantaneous_frequency_smooth_z,
	)
	envelope_threshold = _safe_percentile(
		envelope,
		config.instantaneous_frequency_envelope_quantile * 100.0,
		default=np.inf,
	)
	if np.isfinite(envelope_threshold):
		instantaneous_frequency = np.where(
			envelope >= np.float32(envelope_threshold),
			instantaneous_frequency,
			np.float32(0.0),
		)
	else:
		instantaneous_frequency = np.zeros_like(amplitude, dtype=np.float32)
	valid_if = instantaneous_frequency[
		np.isfinite(instantaneous_frequency) & (instantaneous_frequency >= 0.0)
	]
	if valid_if.size == 0:
		instantaneous_frequency = np.zeros_like(amplitude, dtype=np.float32)
	else:
		clip_value = np.float32(
			np.percentile(
				valid_if,
				config.instantaneous_frequency_clip_percentile,
			),
		)
		if not np.isfinite(clip_value) or clip_value < 0.0:
			instantaneous_frequency = np.zeros_like(amplitude, dtype=np.float32)
		else:
			instantaneous_frequency = np.clip(
				instantaneous_frequency,
				0.0,
				clip_value,
			).astype(np.float32, copy=False)
	return (
		_sanitize(phase_sin, config),
		_sanitize(phase_cos, config),
		_sanitize(instantaneous_frequency, config),
	)


def _hilbert_z(
	amplitude: np.ndarray,
	config: AttributeGenerationConfig,
) -> np.ndarray:
	values = np.asarray(amplitude, dtype=np.float32)
	if values.ndim != 3:
		msg = f'amplitude must be a 3D [x, y, z] array; got ndim={values.ndim}'
		raise ValueError(msg)
	z_size = values.shape[2]
	effective_pad = min(config.phase_reflect_pad_z, max(z_size - 1, 0))
	if effective_pad == 0:
		return _hilbert_z_unpadded(values)

	padded, applied_pad = _reflect_pad_z(values, effective_pad)
	if config.phase_taper_fraction > 0.0:
		padded = _hann_taper_z(padded, config.phase_taper_fraction)
	analytic = _unpad_z(_hilbert_z_unpadded(padded), applied_pad)
	if analytic.shape != values.shape:
		msg = (
			'Hilbert analytic signal shape must match input shape; '
			f'got {analytic.shape!r} and {values.shape!r}'
		)
		raise RuntimeError(msg)
	return analytic


def _hilbert_z_unpadded(amplitude: np.ndarray) -> np.ndarray:
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
	'center_trim_attribute_result',
	'generate_mvp_attribute',
	'generate_mvp_attributes',
	'generate_mvp_attributes_for_payload',
	'load_normalization_stats',
	'normalize_base_seismic',
]
