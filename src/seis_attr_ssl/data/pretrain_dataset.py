"""Unlabeled MVP attribute-volume dataset for MAE pretraining."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import TYPE_CHECKING, cast

import numpy as np

from seis_attr_ssl.attributes import MVP_ATTRIBUTE_REGISTRY, AttributeRegistry
from seis_attr_ssl.attributes.on_the_fly import (
	AttributeGenerationConfig,
	attribute_generation_config_from_mapping,
	generate_mvp_attributes_for_payload,
	normalize_base_seismic,
)
from seis_attr_ssl.data.attribute_subset import (
	AMPLITUDE_ATTRIBUTE_ID,
)
from seis_attr_ssl.data.crop_sampler import (
	compute_context_compute_size_xyz,
	compute_local_compute_size_xyz,
	compute_required_full_halo_size_xyz,
	expand_request_with_halo,
	make_context_request,
	sample_random_local_crop,
	sample_random_local_crop_with_margins,
)
from seis_attr_ssl.data.downsample import (
	downsample_context_masked_mean,
	normalize_downsample_xyz,
)
from seis_attr_ssl.data.normalization import (
	SurveyNormalizationStats,
	load_normalization_stats,
)
from seis_attr_ssl.data.volume_store import NpyMemmapVolumeStore
from seis_attr_ssl.masking import build_mae_masking_plan

if TYPE_CHECKING:
	from pathlib import Path

	from seis_attr_ssl.data.schema import CropRequest, SurveyManifest

XYZ = tuple[int, int, int]
ContextDownsample = int | XYZ


class NopimsAttributePretrainDataset:
	"""Return deterministic random local/context crops from NOPIMS manifests."""

	def __init__(  # noqa: D107, PLR0913
		self,
		manifests: Sequence[SurveyManifest],
		registry: AttributeRegistry = MVP_ATTRIBUTE_REGISTRY,
		local_crop_size_xyz: Sequence[int] = (128, 128, 128),
		local_attribute_halo_xyz: Sequence[int] = (16, 16, 64),
		require_full_halo_inside_volume: bool = True,  # noqa: FBT001, FBT002
		context_crop_size_xyz: Sequence[int] = (256, 256, 512),
		context_downsample: int | Sequence[int] = (2, 2, 4),
		context_attribute_halo_xyz: Sequence[int] = (8, 8, 16),
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
		attribute_generation_config: AttributeGenerationConfig | None = None,
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
		self.local_attribute_halo_xyz = _validate_nonnegative_xyz(
			local_attribute_halo_xyz,
			'local_attribute_halo_xyz',
		)
		self.require_full_halo_inside_volume = _validate_bool(
			require_full_halo_inside_volume,
			'require_full_halo_inside_volume',
		)
		self.context_crop_size_xyz = _validate_xyz(
			context_crop_size_xyz,
			'context_crop_size_xyz',
		)
		self.context_downsample = _validate_downsample(
			context_downsample,
			'context_downsample',
		)
		self.context_downsample_xyz = _downsample_xyz(self.context_downsample)
		self.context_attribute_halo_xyz = _validate_nonnegative_xyz(
			context_attribute_halo_xyz,
			'context_attribute_halo_xyz',
		)
		self.use_context = bool(use_context)
		self.local_compute_size_xyz = compute_local_compute_size_xyz(
			self.local_crop_size_xyz,
			self.local_attribute_halo_xyz,
		)
		self.context_compute_size_xyz = compute_context_compute_size_xyz(
			self.context_crop_size_xyz,
			self.context_downsample_xyz,
			self.context_attribute_halo_xyz,
		)
		self.required_full_halo_size_xyz = compute_required_full_halo_size_xyz(
			self.local_crop_size_xyz,
			self.local_attribute_halo_xyz,
			use_context=self.use_context,
			context_crop_size_xyz=self.context_crop_size_xyz,
			context_downsample_xyz=self.context_downsample_xyz,
			context_attribute_halo_xyz=self.context_attribute_halo_xyz,
		)
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
		self.attribute_generation_config = (
			attribute_generation_config or AttributeGenerationConfig()
		)
		self.attribute_generation_config.validate()

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
				self.context_downsample_xyz,
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
		context_kwargs: dict[str, object] = {}
		if 'context_crop_size' in data:
			context_kwargs['context_crop_size_xyz'] = data['context_crop_size']
		if 'context_downsample' in data:
			context_kwargs['context_downsample'] = data['context_downsample']
		if 'context_attribute_halo' in data:
			context_kwargs['context_attribute_halo_xyz'] = data[
				'context_attribute_halo'
			]

		return cls(
			manifests,
			registry=registry,
			local_crop_size_xyz=data['local_crop_size'],
			local_attribute_halo_xyz=data['local_attribute_halo'],
			require_full_halo_inside_volume=data['require_full_halo_inside_volume'],
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
			attribute_generation_config=attribute_generation_config_from_mapping(
				config.get('attribute_generation'),
			),
			**context_kwargs,
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
		if self.require_full_halo_inside_volume:
			margin_left_xyz, margin_right_xyz = self._sampling_margins_xyz()
			local_request = sample_random_local_crop_with_margins(
				manifest.shape_xyz,
				self.local_crop_size_xyz,
				margin_left_xyz,
				margin_right_xyz,
				rng,
			)
		else:
			local_request = sample_random_local_crop(
				manifest.shape_xyz,
				self.local_crop_size_xyz,
				rng,
			)
		local_compute_request, _ = expand_request_with_halo(
			local_request,
			self.local_attribute_halo_xyz,
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
		context_compute_request: CropRequest | None = None
		if self.use_context:
			_, context_compute_request, _ = self._context_requests(local_request)

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
				'local_attribute_halo_xyz': self.local_attribute_halo_xyz,
				'local_compute_start_xyz': local_compute_request.start_xyz,
				'local_compute_size_xyz': local_compute_request.size_xyz,
				'context_size_xyz': (
					self.context_crop_size_xyz if self.use_context else None
				),
				'context_downsample': (
					self.context_downsample if self.use_context else None
				),
				'context_downsample_xyz': (
					self.context_downsample_xyz if self.use_context else None
				),
				'context_attribute_halo_xyz': (
					self.context_attribute_halo_xyz if self.use_context else None
				),
				'context_compute_start_xyz': (
					context_compute_request.start_xyz
					if context_compute_request is not None
					else None
				),
				'context_compute_size_xyz': (
					context_compute_request.size_xyz
					if context_compute_request is not None
					else None
				),
				'context_lowres_compute_size_xyz': (
					tuple(
						size_axis // downsample_axis
						for size_axis, downsample_axis in zip(
							context_compute_request.size_xyz,
							self.context_downsample_xyz,
							strict=True,
						)
					)
					if context_compute_request is not None
					else None
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
			if self.require_full_halo_inside_volume:
				self._validate_full_halo_volume_size(manifest)

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
			compute_request, payload_slices = expand_request_with_halo(
				local_request,
				self.local_attribute_halo_xyz,
			)
			base_crop, compute_valid_mask = self._store.read_crop_with_padding(
				_resolve_manifest_path(manifest, manifest.base_seismic.path),
				compute_request.start_xyz,
				compute_request.size_xyz,
			)
			stats = self._stats_for_manifest(manifest)
			result = generate_mvp_attributes_for_payload(
				normalize_base_seismic(base_crop, stats),
				payload_slices,
				valid_mask=compute_valid_mask,
				config=self.attribute_generation_config,
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

		_, context_compute_request, lowres_payload_slices = self._context_requests(
			local_request,
		)
		if manifest.base_seismic is not None:
			base_crop, valid_mask = self._store.read_crop_with_padding(
				_resolve_manifest_path(manifest, manifest.base_seismic.path),
				context_compute_request.start_xyz,
				context_compute_request.size_xyz,
			)
			stats = self._stats_for_manifest(manifest)
			normalized_context, context_valid_mask = downsample_context_masked_mean(
				normalize_base_seismic(base_crop, stats),
				valid_mask,
				self.context_downsample_xyz,
			)
			context_result = generate_mvp_attributes_for_payload(
				normalized_context,
				lowres_payload_slices,
				valid_mask=context_valid_mask,
				config=self.attribute_generation_config,
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
				context_compute_request.start_xyz,
				context_compute_request.size_xyz,
			)
			context_volume, attribute_valid_mask = downsample_context_masked_mean(
				crop,
				valid_mask,
				self.context_downsample_xyz,
			)
			context_volumes.append(context_volume[lowres_payload_slices])
			if context_valid_mask is None:
				context_valid_mask = attribute_valid_mask[lowres_payload_slices]

		return (
			np.stack(context_volumes, axis=0).astype(np.float32, copy=False),
			context_valid_mask,
		)

	def _context_requests(
		self,
		local_request: CropRequest,
	) -> tuple[CropRequest, CropRequest, tuple[slice, slice, slice]]:
		context_request = make_context_request(
			local_request,
			self.context_crop_size_xyz,
			self.context_downsample,
		)
		source_halo_xyz = self._context_source_halo_xyz()
		context_compute_request, _ = expand_request_with_halo(
			context_request,
			source_halo_xyz,
		)
		lowres_payload_slices = tuple(
			slice(halo_axis, halo_axis + size_axis // downsample_axis)
			for halo_axis, size_axis, downsample_axis in zip(
				self.context_attribute_halo_xyz,
				self.context_crop_size_xyz,
				self.context_downsample_xyz,
				strict=True,
			)
		)
		return (
			context_request,
			context_compute_request,
			lowres_payload_slices,
		)

	def _validate_full_halo_volume_size(self, manifest: SurveyManifest) -> None:
		required = self._required_sampling_volume_size_xyz()
		if all(
			shape_axis >= required_axis
			for shape_axis, required_axis in zip(
				manifest.shape_xyz,
				required,
				strict=True,
			)
		):
			return
		msg = (
			f'survey {manifest.survey_id!r} shape {list(manifest.shape_xyz)} '
			f'is smaller than required full-halo size {list(required)}. '
			'Either remove it from the path-list, reduce context geometry, '
			'or set use_context=false.'
		)
		raise ValueError(msg)

	def _required_sampling_volume_size_xyz(self) -> XYZ:
		margin_left_xyz, margin_right_xyz = self._sampling_margins_xyz()
		return tuple(
			local_axis + left_axis + right_axis
			for local_axis, left_axis, right_axis in zip(
				self.local_crop_size_xyz,
				margin_left_xyz,
				margin_right_xyz,
				strict=True,
			)
		)

	def _sampling_margins_xyz(self) -> tuple[XYZ, XYZ]:
		if not self.use_context:
			return self.local_attribute_halo_xyz, self.local_attribute_halo_xyz

		source_halo_xyz = self._context_source_halo_xyz()
		axes = zip(
			self.local_attribute_halo_xyz,
			self.local_crop_size_xyz,
			self.context_crop_size_xyz,
			source_halo_xyz,
			strict=True,
		)
		margin_left_xyz = tuple(
			max(
				local_halo,
				_context_payload_margin_left_axis(
					local_size,
					context_size,
					source_halo,
				),
			)
			for local_halo, local_size, context_size, source_halo in axes
		)
		axes = zip(
			self.local_attribute_halo_xyz,
			self.local_crop_size_xyz,
			self.context_crop_size_xyz,
			source_halo_xyz,
			strict=True,
		)
		margin_right_xyz = tuple(
			max(
				local_halo,
				_context_payload_margin_right_axis(
					local_size,
					context_size,
					source_halo,
				),
			)
			for local_halo, local_size, context_size, source_halo in axes
		)
		return cast('XYZ', margin_left_xyz), cast('XYZ', margin_right_xyz)

	def _context_source_halo_xyz(self) -> XYZ:
		source_halo_xyz = tuple(
			halo_axis * downsample_axis
			for halo_axis, downsample_axis in zip(
				self.context_attribute_halo_xyz,
				self.context_downsample_xyz,
				strict=True,
			)
		)
		return cast('XYZ', source_halo_xyz)

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
		or not all(
			not isinstance(axis, bool) and isinstance(axis, Integral)
			for axis in value
		)
	):
		msg = f'{name} must be a length-3 integer sequence; got {value!r}'
		raise TypeError(msg)
	xyz = tuple(int(axis) for axis in value)
	if any(axis <= 0 for axis in xyz):
		msg = f'{name} values must be positive; got {xyz!r}'
		raise ValueError(msg)
	return xyz


def _validate_nonnegative_xyz(value: Sequence[int], name: str) -> XYZ:
	if (
		isinstance(value, str)
		or len(value) != 3
		or not all(
			not isinstance(axis, bool) and isinstance(axis, Integral)
			for axis in value
		)
	):
		msg = f'{name} must be a length-3 integer sequence; got {value!r}'
		raise TypeError(msg)
	xyz = tuple(int(axis) for axis in value)
	if any(axis < 0 for axis in xyz):
		msg = f'{name} values must be nonnegative; got {xyz!r}'
		raise ValueError(msg)
	return xyz


def _validate_bool(value: object, name: str) -> bool:
	if not isinstance(value, bool):
		msg = f'{name} must be a bool; got {value!r}'
		raise TypeError(msg)
	return value


def _validate_positive_int(value: int, name: str) -> int:
	if isinstance(value, bool) or not isinstance(value, Integral):
		msg = f'{name} must be an integer; got {value!r}'
		raise TypeError(msg)
	count = int(value)
	if count <= 0:
		msg = f'{name} must be positive; got {count!r}'
		raise ValueError(msg)
	return count


def _validate_downsample(value: int | Sequence[int], name: str) -> ContextDownsample:
	if isinstance(value, bool):
		msg = f'{name} must be a positive integer or triple; got {value!r}'
		raise TypeError(msg)
	if isinstance(value, Integral):
		count = int(value)
		if count <= 0:
			msg = f'{name} must be positive; got {count!r}'
			raise ValueError(msg)
		return count
	try:
		return normalize_downsample_xyz(value)
	except (TypeError, ValueError) as exc:
		msg = f'{name} must be a positive integer or triple; got {value!r}'
		raise type(exc)(msg) from exc


def _downsample_xyz(value: ContextDownsample) -> XYZ:
	return normalize_downsample_xyz(value)


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
	context_downsample_xyz: XYZ,
) -> None:
	if any(
		context_axis % downsample_axis != 0
		for context_axis, downsample_axis in zip(
			context_crop_size_xyz,
			context_downsample_xyz,
			strict=True,
		)
	):
		msg = (
			'context_crop_size_xyz must be divisible by context_downsample; '
			f'got {context_crop_size_xyz!r} and {context_downsample_xyz!r}'
		)
		raise ValueError(msg)
	context_output = tuple(
		context_axis // downsample_axis
		for context_axis, downsample_axis in zip(
			context_crop_size_xyz,
			context_downsample_xyz,
			strict=True,
		)
	)
	if context_output != local_crop_size_xyz:
		msg = (
			'downsampled context shape must match local crop size; '
			f'got {context_output!r} and {local_crop_size_xyz!r}'
		)
		raise ValueError(msg)


def _context_payload_margin_left_axis(
	local_size_axis: int,
	context_size_axis: int,
	source_halo_axis: int,
) -> int:
	return max(
		0,
		context_size_axis // 2
		+ source_halo_axis
		- local_size_axis // 2,
	)


def _context_payload_margin_right_axis(
	local_size_axis: int,
	context_size_axis: int,
	source_halo_axis: int,
) -> int:
	return max(
		0,
		context_size_axis
		- context_size_axis // 2
		+ source_halo_axis
		- (local_size_axis - local_size_axis // 2),
	)


__all__ = ['NopimsAttributePretrainDataset']
