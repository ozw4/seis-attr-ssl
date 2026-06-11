"""Unlabeled MVP attribute-volume dataset for MAE pretraining."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral, Real
from typing import TYPE_CHECKING

import numpy as np

from seis_attr_ssl.attributes import MVP_ATTRIBUTE_REGISTRY, AttributeRegistry
from seis_attr_ssl.attributes.on_the_fly import (
	generate_mvp_attributes,
	normalize_base_seismic,
)
from seis_attr_ssl.data.attribute_subset import (
	AMPLITUDE_ATTRIBUTE_ID,
)
from seis_attr_ssl.data.crop_sampler import (
	make_context_request,
	sample_random_local_crop,
)
from seis_attr_ssl.data.downsample import downsample_context_masked_mean
from seis_attr_ssl.data.normalization import (
	SurveyNormalizationStats,
	load_normalization_stats,
)
from seis_attr_ssl.data.volume_store import NpyMemmapVolumeStore
from seis_attr_ssl.masking import build_mae_masking_plan

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
		patch_size_xyz: Sequence[int] = (8, 8, 8),
		spatial_mask_ratio: float = 0.75,
		spatial_mask_mode: str = 'block',
		block_size_tokens_xyz: Sequence[int] = (2, 2, 2),
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
		self.patch_size_xyz = _validate_xyz(patch_size_xyz, 'patch_size_xyz')
		self.spatial_mask_ratio = _validate_probability(
			spatial_mask_ratio,
			'spatial_mask_ratio',
		)
		self.spatial_mask_mode = _validate_spatial_mask_mode(spatial_mask_mode)
		self.block_size_tokens_xyz = _validate_xyz(
			block_size_tokens_xyz,
			'block_size_tokens_xyz',
		)
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
		self.epoch = 0
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
		self._normalization_stats: dict[Path, SurveyNormalizationStats] = {}
		self._validate_manifests()

	@classmethod
	def from_config(
		cls,
		manifests: Sequence[SurveyManifest],
		config: Mapping[str, object],
		registry: AttributeRegistry = MVP_ATTRIBUTE_REGISTRY,
		*,
		samples_per_epoch: int | None = None,
	) -> NopimsAttributePretrainDataset:
		"""Build a pretrain dataset from validated MVP config sections."""
		data = _require_config_mapping(config, 'data')
		model = _require_config_mapping(config, 'model')
		masking = _require_config_mapping(config, 'masking')
		train = _require_config_mapping(config, 'train')

		return cls(
			manifests,
			registry=registry,
			local_crop_size_xyz=data['local_crop_size'],
			context_crop_size_xyz=data['context_crop_size'],
			context_downsample=data['context_downsample'],
			use_context=data['use_context'],
			patch_size_xyz=model['patch_size'],
			spatial_mask_ratio=masking['spatial_mask_ratio'],
			spatial_mask_mode=masking['spatial_mask_mode'],
			block_size_tokens_xyz=masking['block_size_tokens'],
			min_input_attributes=masking['min_input_attributes'],
			max_input_attributes=masking['max_input_attributes'],
			attribute_dropout_prob=masking['attribute_dropout_prob'],
			group_dropout_prob=masking['group_dropout_prob'],
			seed=train['seed'],
			samples_per_epoch=samples_per_epoch,
		)

	def __len__(self) -> int:
		"""Return configured epoch length."""
		return self.samples_per_epoch

	def set_epoch(self, epoch: int) -> None:
		"""Set the sampling epoch used to seed deterministic sample draws."""
		self.epoch = _validate_nonnegative_int(epoch, 'epoch')

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
		target, target_valid, local_valid_mask = self._read_target(
			manifest,
			local_request,
		)
		masking_plan = build_mae_masking_plan(
			available_attribute_ids=available_ids,
			target_valid=target_valid,
			local_crop_size_xyz=self.local_crop_size_xyz,
			patch_size_xyz=self.patch_size_xyz,
			spatial_mask_ratio=self.spatial_mask_ratio,
			block_size_tokens_xyz=self.block_size_tokens_xyz,
			min_input_attributes=self.min_input_attributes,
			max_input_attributes=self.max_input_attributes,
			attribute_dropout_prob=self.attribute_dropout_prob,
			group_dropout_prob=self.group_dropout_prob,
			attribute_groups=self.registry.groups,
			rng=rng,
		)
		input_ids = tuple(int(id_) for id_ in masking_plan.input_attribute_ids)
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
			'spatial_mask': masking_plan.spatial_mask,
			'visible_spatial_mask': masking_plan.visible_spatial_mask,
			'attribute_input_mask': masking_plan.attribute_input_mask,
			'attribute_target_mask': masking_plan.attribute_target_mask,
			'dropped_attribute_mask': masking_plan.dropped_attribute_mask,
			'target_attribute_ids': masking_plan.target_attribute_ids,
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
		seed_sequence = np.random.SeedSequence([self.seed, self.epoch, index])
		return np.random.default_rng(seed_sequence)

	def _validate_manifests(self) -> None:
		amplitude_name = self.registry.id_to_name(AMPLITUDE_ATTRIBUTE_ID)
		for manifest in self.manifests:
			manifest.validate_consistent_shapes()
			if manifest.base_seismic is not None:
				base_path = _resolve_manifest_path(manifest, manifest.base_seismic.path)
				if not base_path.is_file():
					msg = (
						f'survey {manifest.survey_id!r} base seismic file '
						f'does not exist: {base_path}'
					)
					raise FileNotFoundError(msg)
				stats_path = _resolve_manifest_path(
					manifest,
					manifest.base_seismic.normalization_stats_path,
				)
				if not stats_path.is_file():
					msg = (
						f'survey {manifest.survey_id!r} normalization stats file '
						f'does not exist: {stats_path}'
					)
					raise FileNotFoundError(msg)
			elif amplitude_name not in manifest.attribute_volumes:
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
		if manifest.base_seismic is not None:
			return tuple(spec.id for spec in self.registry.specs)
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

		if manifest.base_seismic is not None:
			base_crop, local_valid_mask = self._store.read_crop_with_padding(
				_resolve_manifest_path(manifest, manifest.base_seismic.path),
				local_request.start_xyz,
				local_request.size_xyz,
			)
			stats = self._stats_for_manifest(manifest)
			result = generate_mvp_attributes(
				normalize_base_seismic(base_crop, stats),
				valid_mask=local_valid_mask,
			)
			return result.attributes, result.attribute_valid, result.voxel_valid_mask

		for spec in self.registry.specs:
			record = manifest.attribute_volumes.get(spec.name)
			if record is None:
				continue
			crop, valid_mask = self._store.read_crop_with_padding(
				_resolve_manifest_path(manifest, record.path),
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
		if manifest.base_seismic is not None:
			base_crop, valid_mask = self._store.read_crop_with_padding(
				_resolve_manifest_path(manifest, manifest.base_seismic.path),
				context_request.start_xyz,
				context_request.size_xyz,
			)
			stats = self._stats_for_manifest(manifest)
			normalized_context, context_valid_mask = downsample_context_masked_mean(
				normalize_base_seismic(base_crop, stats),
				valid_mask,
				self.context_downsample,
			)
			context_result = generate_mvp_attributes(
				normalized_context,
				valid_mask=context_valid_mask,
			)
			ids = np.asarray(input_ids, dtype=np.int64)
			return (
				context_result.attributes[ids].astype(np.float32, copy=False),
				context_result.voxel_valid_mask,
			)

		context_volumes: list[np.ndarray] = []
		context_valid_mask: np.ndarray | None = None
		for id_ in input_ids:
			name = self.registry.id_to_name(id_)
			record = manifest.get_attribute(name)
			crop, valid_mask = self._store.read_crop_with_padding(
				_resolve_manifest_path(manifest, record.path),
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

	def _stats_for_manifest(self, manifest: SurveyManifest) -> SurveyNormalizationStats:
		if manifest.base_seismic is None:
			msg = f'survey {manifest.survey_id!r} has no base seismic record'
			raise ValueError(msg)
		path = _resolve_manifest_path(
			manifest,
			manifest.base_seismic.normalization_stats_path,
		)
		if path not in self._normalization_stats:
			self._normalization_stats[path] = load_normalization_stats(path)
		return self._normalization_stats[path]


def _resolve_manifest_path(manifest: SurveyManifest, path: Path) -> Path:
	if path.is_absolute():
		return path
	return manifest.root / path


def _require_config_mapping(
	config: Mapping[str, object],
	key: str,
) -> Mapping[str, object]:
	value = config[key]
	if not isinstance(value, Mapping):
		msg = f'config.{key} must be a mapping'
		raise TypeError(msg)
	return value


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
	if isinstance(value, bool) or not isinstance(value, Integral):
		msg = f'{name} must be an integer; got {value!r}'
		raise TypeError(msg)
	count = int(value)
	if count <= 0:
		msg = f'{name} must be positive; got {count!r}'
		raise ValueError(msg)
	return count


def _validate_nonnegative_int(value: int, name: str) -> int:
	if isinstance(value, bool) or not isinstance(value, Integral):
		msg = f'{name} must be an integer; got {value!r}'
		raise TypeError(msg)
	count = int(value)
	if count < 0:
		msg = f'{name} must be nonnegative; got {count!r}'
		raise ValueError(msg)
	return count


def _validate_probability(value: float, name: str) -> float:
	if isinstance(value, bool) or not isinstance(value, Real):
		msg = f'{name} must be a real number; got {value!r}'
		raise TypeError(msg)
	probability = float(value)
	if not 0.0 <= probability < 1.0:
		msg = f'{name} must be in [0, 1); got {probability!r}'
		raise ValueError(msg)
	return probability


def _validate_spatial_mask_mode(value: str) -> str:
	if value != 'block':
		msg = f"spatial_mask_mode must be 'block'; got {value!r}"
		raise ValueError(msg)
	return value


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
