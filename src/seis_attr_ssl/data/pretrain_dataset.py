"""Unlabeled MVP attribute-volume dataset for MAE pretraining."""

from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np

from seis_attr_ssl.attributes import MVP_ATTRIBUTE_REGISTRY, AttributeRegistry
from seis_attr_ssl.data.attribute_subset import (
	AMPLITUDE_ATTRIBUTE_ID,
)
from seis_attr_ssl.data.crop_sampler import (
	make_context_request,
	sample_random_local_crop,
)
from seis_attr_ssl.data.downsample import downsample_context_masked_mean
from seis_attr_ssl.data.volume_store import NpyMemmapVolumeStore
from seis_attr_ssl.masking import sample_attribute_input_mask

if TYPE_CHECKING:
	from collections.abc import Sequence
	from pathlib import Path

	from seis_attr_ssl.data.schema import CropRequest, SurveyManifest

XYZ = tuple[int, int, int]


class NopimsAttributePretrainDataset:
	"""Return deterministic random local/context crops from NOPIMS manifests."""

	def __init__(  # noqa: D107, PLR0913
		self,
		manifests: Sequence[SurveyManifest],
		registry: AttributeRegistry = MVP_ATTRIBUTE_REGISTRY,
		local_crop_size_xyz: Sequence[int] = (128, 128, 128),
		context_crop_size_xyz: Sequence[int] = (512, 512, 512),
		context_downsample: int = 4,
		use_context: bool = True,  # noqa: FBT001, FBT002
		min_input_attributes: int = 4,
		max_input_attributes: int = 10,
		attribute_dropout_prob: float = 0.0,
		group_dropout_prob: float = 0.0,
		seed: int = 42,
		samples_per_epoch: int | None = None,
	) -> None:
		self.manifests = tuple(manifests)
		if not self.manifests:
			msg = 'manifests must contain at least one survey'
			raise ValueError(msg)

		self.registry = registry
		self.local_crop_size_xyz = _validate_xyz(
			local_crop_size_xyz,
			'local_crop_size_xyz',
		)
		self.context_crop_size_xyz = _validate_xyz(
			context_crop_size_xyz,
			'context_crop_size_xyz',
		)
		self.context_downsample = _validate_positive_int(
			context_downsample,
			'context_downsample',
		)
		self.use_context = bool(use_context)
		self.min_input_attributes = _validate_positive_int(
			min_input_attributes,
			'min_input_attributes',
		)
		self.max_input_attributes = _validate_positive_int(
			max_input_attributes,
			'max_input_attributes',
		)
		if self.min_input_attributes > self.max_input_attributes:
			msg = (
				'min_input_attributes must be less than or equal to '
				'max_input_attributes'
			)
			raise ValueError(msg)
		self.attribute_dropout_prob = float(attribute_dropout_prob)
		self.group_dropout_prob = float(group_dropout_prob)

		self.seed = _validate_nonnegative_int(seed, 'seed')
		if samples_per_epoch is None:
			self.samples_per_epoch = len(self.manifests)
		else:
			self.samples_per_epoch = _validate_positive_int(
				samples_per_epoch,
				'samples_per_epoch',
			)

		if self.use_context:
			_validate_context_geometry(
				self.local_crop_size_xyz,
				self.context_crop_size_xyz,
				self.context_downsample,
			)

		self._store = NpyMemmapVolumeStore()
		self._target_attribute_ids = np.asarray(
			[spec.id for spec in self.registry.specs],
			dtype=np.int64,
		)
		self._validate_manifests()

	def __len__(self) -> int:
		"""Return configured epoch length."""
		return self.samples_per_epoch

	def __getitem__(self, index: int) -> dict[str, object]:
		"""Return one MVP-compatible sample dictionary."""
		if not isinstance(index, Integral):
			msg = f'index must be an integer; got {index!r}'
			raise TypeError(msg)
		index = int(index)
		if index < 0:
			index += len(self)
		if index < 0 or index >= len(self):
			msg = f'index out of range: {index!r}'
			raise IndexError(msg)

		manifest = self.manifests[index % len(self.manifests)]
		rng = self._rng_for_index(index)
		local_request = sample_random_local_crop(
			manifest.shape_xyz,
			self.local_crop_size_xyz,
			rng,
		)
		available_ids = self._available_attribute_ids(manifest)
		attribute_input_mask = sample_attribute_input_mask(
			available_ids,
			self._target_attribute_ids,
			self.registry.groups,
			self.min_input_attributes,
			self.max_input_attributes,
			self.attribute_dropout_prob,
			self.group_dropout_prob,
			rng,
		)
		input_ids = tuple(
			int(id_) for id_ in self._target_attribute_ids[attribute_input_mask]
		)

		target, target_valid, local_valid_mask = self._read_target(
			manifest,
			local_request,
		)
		x = np.stack(
			[target[id_] for id_ in input_ids],
			axis=0,
		).astype(np.float32, copy=False)

		context, context_valid_mask = self._read_context(
			manifest,
			local_request,
			input_ids,
		)

		return {
			'x': x,
			'target': target,
			'attribute_ids': np.asarray(input_ids, dtype=np.int64),
			'attribute_input_mask': attribute_input_mask,
			'target_attribute_ids': self._target_attribute_ids.copy(),
			'valid_attributes': np.ones(len(input_ids), dtype=bool),
			'target_valid': target_valid,
			'coords': {
				'survey_id': manifest.survey_id,
				'local_start_xyz': local_request.start_xyz,
				'local_size_xyz': local_request.size_xyz,
				'context_size_xyz': (
					self.context_crop_size_xyz if self.use_context else None
				),
				'context_downsample': (
					self.context_downsample if self.use_context else None
				),
			},
			'context': context,
			'context_valid_mask': context_valid_mask,
			'local_valid_mask': local_valid_mask,
		}

	def _rng_for_index(self, index: int) -> np.random.Generator:
		seed_sequence = np.random.SeedSequence([self.seed, index])
		return np.random.default_rng(seed_sequence)

	def _validate_manifests(self) -> None:
		amplitude_name = self.registry.id_to_name(AMPLITUDE_ATTRIBUTE_ID)
		for manifest in self.manifests:
			manifest.validate_consistent_shapes()
			if amplitude_name not in manifest.attribute_volumes:
				msg = (
					f'survey {manifest.survey_id!r} is missing required '
					f'attribute {amplitude_name!r}'
				)
				raise ValueError(msg)
			available_ids = self._available_attribute_ids(manifest)
			if len(available_ids) < self.min_input_attributes:
				msg = (
					f'survey {manifest.survey_id!r} has {len(available_ids)} '
					'available MVP attributes, fewer than '
					f'min_input_attributes={self.min_input_attributes!r}'
				)
				raise ValueError(msg)

	def _available_attribute_ids(self, manifest: SurveyManifest) -> tuple[int, ...]:
		return tuple(
			spec.id
			for spec in self.registry.specs
			if spec.name in manifest.attribute_volumes
		)

	def _read_target(
		self,
		manifest: SurveyManifest,
		local_request: CropRequest,
	) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		target = np.zeros(
			(len(self.registry.specs), *self.local_crop_size_xyz),
			dtype=np.float32,
		)
		target_valid = np.zeros(len(self.registry.specs), dtype=bool)
		local_valid_mask: np.ndarray | None = None

		for spec in self.registry.specs:
			record = manifest.attribute_volumes.get(spec.name)
			if record is None:
				continue
			crop, valid_mask = self._store.read_crop_with_padding(
				_resolve_record_path(manifest, record.path),
				local_request.start_xyz,
				local_request.size_xyz,
			)
			target[spec.id] = crop.astype(np.float32, copy=False)
			target_valid[spec.id] = True
			if local_valid_mask is None:
				local_valid_mask = valid_mask

		if local_valid_mask is None:
			local_valid_mask = np.zeros(self.local_crop_size_xyz, dtype=bool)
		return target, target_valid, local_valid_mask

	def _read_context(
		self,
		manifest: SurveyManifest,
		local_request: CropRequest,
		input_ids: tuple[int, ...],
	) -> tuple[np.ndarray | None, np.ndarray | None]:
		if not self.use_context:
			return None, None

		context_request = make_context_request(
			local_request,
			self.context_crop_size_xyz,
			self.context_downsample,
		)
		context_volumes: list[np.ndarray] = []
		context_valid_mask: np.ndarray | None = None
		for id_ in input_ids:
			name = self.registry.id_to_name(id_)
			record = manifest.get_attribute(name)
			crop, valid_mask = self._store.read_crop_with_padding(
				_resolve_record_path(manifest, record.path),
				context_request.start_xyz,
				context_request.size_xyz,
			)
			context_volume, attribute_valid_mask = downsample_context_masked_mean(
				crop,
				valid_mask,
				self.context_downsample,
			)
			context_volumes.append(context_volume)
			if context_valid_mask is None:
				context_valid_mask = attribute_valid_mask

		return (
			np.stack(context_volumes, axis=0).astype(np.float32, copy=False),
			context_valid_mask,
		)


def _resolve_record_path(manifest: SurveyManifest, path: Path) -> Path:
	if path.is_absolute():
		return path
	return manifest.root / path


def _validate_xyz(value: Sequence[int], name: str) -> XYZ:
	if (
		isinstance(value, str)
		or len(value) != 3
		or not all(isinstance(axis, Integral) for axis in value)
	):
		msg = f'{name} must be a length-3 integer sequence; got {value!r}'
		raise TypeError(msg)
	xyz = tuple(int(axis) for axis in value)
	if any(axis <= 0 for axis in xyz):
		msg = f'{name} values must be positive; got {xyz!r}'
		raise ValueError(msg)
	return xyz


def _validate_positive_int(value: int, name: str) -> int:
	if not isinstance(value, Integral):
		msg = f'{name} must be an integer; got {value!r}'
		raise TypeError(msg)
	count = int(value)
	if count <= 0:
		msg = f'{name} must be positive; got {count!r}'
		raise ValueError(msg)
	return count


def _validate_nonnegative_int(value: int, name: str) -> int:
	if not isinstance(value, Integral):
		msg = f'{name} must be an integer; got {value!r}'
		raise TypeError(msg)
	count = int(value)
	if count < 0:
		msg = f'{name} must be nonnegative; got {count!r}'
		raise ValueError(msg)
	return count


def _validate_context_geometry(
	local_crop_size_xyz: XYZ,
	context_crop_size_xyz: XYZ,
	context_downsample: int,
) -> None:
	if any(axis % context_downsample != 0 for axis in context_crop_size_xyz):
		msg = (
			'context_crop_size_xyz must be divisible by context_downsample; '
			f'got {context_crop_size_xyz!r} and {context_downsample!r}'
		)
		raise ValueError(msg)
	context_output = tuple(axis // context_downsample for axis in context_crop_size_xyz)
	if context_output != local_crop_size_xyz:
		msg = (
			'downsampled context shape must match local crop size; '
			f'got {context_output!r} and {local_crop_size_xyz!r}'
		)
		raise ValueError(msg)


__all__ = ['NopimsAttributePretrainDataset']
