"""Compute normalization stats sidecars for NOPIMS manifest volumes."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[1] / 'src'
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from seis_attr_ssl.config import load_config, validate_config  # noqa: E402
from seis_attr_ssl.data.normalization import (  # noqa: E402
	compute_normalization_stats,
	write_normalization_stats,
)
from seis_attr_ssl.data.schema import (  # noqa: E402
	BASE_SEISMIC_DTYPE_FLOAT32,
	GRID_ORDER_XYZ,
	BaseSeismicVolumeRecord,
	SurveyManifest,
	read_manifest_json,
)
from seis_attr_ssl.utils.cli import print_config_summary  # noqa: E402

DEFAULT_CONFIG = (
	Path(__file__).resolve().parent / 'configs' / 'mvp_prepare_nopims_stats.yaml'
)
MANIFEST_BUILD_HINT = (
	'hint: build the manifest with `python proc/build_nopims_manifests.py --config '
	'proc/configs/build_nopims_manifests.yaml`'
)


@dataclass(frozen=True)
class NormalizationTarget:
	"""Validated manifest target for one sidecar stats file."""

	survey_id: str
	base_seismic: BaseSeismicVolumeRecord

	@property
	def output_path(self) -> Path:
		"""Stats sidecar path from the manifest entry."""
		return self.base_seismic.normalization_stats_path


def main() -> None:
	"""Compute missing NOPIMS normalization stats sidecars."""
	args = _parse_args()
	config = validate_config(load_config(args.config))
	manifest_path = _manifest_path(config)
	normalization = _required_mapping(config, 'normalization')
	pre_attribute = _required_mapping(normalization, 'pre_attribute')
	stats_cfg = _required_mapping(config, 'normalization_stats')
	max_samples = _optional_positive_int(stats_cfg, 'max_samples')
	seed = _required_int(stats_cfg, 'seed')
	clip_low, clip_high = _required_percentiles(pre_attribute)
	eps = _required_float(pre_attribute, 'epsilon')

	if not manifest_path.is_file():
		if args.dry_run:
			print_config_summary(config)
			print(f'normalization_stats.manifest_path: {manifest_path}')
			print('normalization_stats.manifest_exists: false')
			print('normalization_stats.manifest_entries: 0')
			print(f'normalization_stats.max_samples: {max_samples}')
			print(f'normalization_stats.seed: {seed}')
			print(f'normalization_stats.overwrite: {str(args.overwrite).lower()}')
			print('normalization_stats.compute: skipped')
			print(f'normalization_stats.message: manifest does not exist: {manifest_path}')
			print(MANIFEST_BUILD_HINT)
			return
		msg = f'manifests.train does not exist: {manifest_path}. {MANIFEST_BUILD_HINT}'
		raise FileNotFoundError(msg)

	manifests = read_manifest_json(manifest_path)
	targets = [_normalization_target(manifest) for manifest in manifests]
	existing_count = sum(target.output_path.is_file() for target in targets)
	missing_count = len(targets) - existing_count

	if args.dry_run:
		print_config_summary(config)
		print(f'normalization_stats.manifest_path: {manifest_path}')
		print('normalization_stats.manifest_exists: true')
		print(f'normalization_stats.manifest_entries: {len(targets)}')
		print(f'normalization_stats.existing_files: {existing_count}')
		print(f'normalization_stats.missing_files: {missing_count}')
		print(f'normalization_stats.max_samples: {max_samples}')
		print(f'normalization_stats.seed: {seed}')
		print(f'normalization_stats.overwrite: {str(args.overwrite).lower()}')
		print('normalization_stats.compute: skipped')
		return

	written_count = 0
	skipped_count = 0
	for target in targets:
		if target.output_path.is_file() and not args.overwrite:
			skipped_count += 1
			continue
		stats = compute_normalization_stats(
			target.base_seismic.path,
			survey_id=target.survey_id,
			grid_order=target.base_seismic.grid_order,
			clip_low_percentile=clip_low,
			clip_high_percentile=clip_high,
			max_samples=max_samples,
			seed=seed,
			eps=eps,
		)
		write_normalization_stats(stats, target.output_path)
		written_count += 1

	print(f'normalization_stats.manifest_path: {manifest_path}')
	print(f'normalization_stats.manifest_entries: {len(targets)}')
	print(f'normalization_stats.written_files: {written_count}')
	print(f'normalization_stats.skipped_existing_files: {skipped_count}')


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description='Prepare normalization stats sidecars for NOPIMS manifests.',
	)
	parser.add_argument(
		'--config',
		type=Path,
		default=DEFAULT_CONFIG,
		help='Path to a YAML configuration file.',
	)
	parser.add_argument(
		'--dry-run',
		action='store_true',
		help='Validate inputs and print a run summary without writing stats.',
	)
	parser.add_argument(
		'--overwrite',
		action='store_true',
		help='Recompute stats files even when sidecars already exist.',
	)
	return parser.parse_args()


def _manifest_path(config: Mapping[str, object]) -> Path:
	manifests = _required_mapping(config, 'manifests')
	return Path(_required_str(manifests, 'train'))


def _normalization_target(manifest: SurveyManifest) -> NormalizationTarget:
	base_seismic = manifest.base_seismic
	if base_seismic is None:
		msg = f'manifest {manifest.survey_id!r} has no base_seismic'
		raise ValueError(msg)
	if not base_seismic.path.is_file():
		msg = f'base_seismic.path does not exist: {base_seismic.path}'
		raise FileNotFoundError(msg)
	if base_seismic.dtype != BASE_SEISMIC_DTYPE_FLOAT32:
		msg = (
			f'base_seismic.dtype must be {BASE_SEISMIC_DTYPE_FLOAT32!r}; '
			f'got {base_seismic.dtype!r}'
		)
		raise ValueError(msg)
	if base_seismic.grid_order != GRID_ORDER_XYZ:
		msg = (
			f'base_seismic.grid_order must be {list(GRID_ORDER_XYZ)!r}; '
			f'got {list(base_seismic.grid_order)!r}'
		)
		raise ValueError(msg)
	return NormalizationTarget(
		survey_id=manifest.survey_id,
		base_seismic=base_seismic,
	)


def _required_mapping(parent: Mapping[str, object], key: str) -> Mapping[str, Any]:
	value = parent.get(key)
	if not isinstance(value, Mapping):
		msg = f'{key} must be a mapping'
		raise TypeError(msg)
	return value


def _required_str(parent: Mapping[str, object], key: str) -> str:
	value = parent.get(key)
	if not isinstance(value, str) or not value:
		msg = f'{key} must be a non-empty string; got {value!r}'
		raise TypeError(msg)
	return value


def _required_int(parent: Mapping[str, object], key: str) -> int:
	value = parent.get(key)
	if isinstance(value, bool) or not isinstance(value, int):
		msg = f'{key} must be an integer; got {value!r}'
		raise TypeError(msg)
	return value


def _optional_positive_int(parent: Mapping[str, object], key: str) -> int | None:
	value = parent.get(key)
	if value is None:
		return None
	if isinstance(value, bool) or not isinstance(value, int):
		msg = f'{key} must be an integer or null; got {value!r}'
		raise TypeError(msg)
	if value <= 0:
		msg = f'{key} must be positive when provided; got {value!r}'
		raise ValueError(msg)
	return value


def _required_float(parent: Mapping[str, object], key: str) -> float:
	value = parent.get(key)
	if isinstance(value, bool) or not isinstance(value, int | float):
		msg = f'{key} must be numeric; got {value!r}'
		raise TypeError(msg)
	return float(value)


def _required_percentiles(parent: Mapping[str, object]) -> tuple[float, float]:
	value = parent.get('clipping_percentiles')
	if (
		not isinstance(value, list)
		or len(value) != 2
		or any(
			isinstance(item, bool) or not isinstance(item, int | float)
			for item in value
		)
	):
		msg = f'clipping_percentiles must contain two numbers; got {value!r}'
		raise TypeError(msg)
	low, high = (float(value[0]), float(value[1]))
	if not 0.0 <= low < high <= 100.0:
		msg = f'clipping_percentiles must satisfy 0 <= low < high <= 100; got {value!r}'
		raise ValueError(msg)
	return low, high


if __name__ == '__main__':
	main()
