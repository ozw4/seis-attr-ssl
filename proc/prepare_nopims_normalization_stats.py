"""Compute normalization stats for NOPIMS base-seismic manifest entries."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[1] / 'src'
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from seis_attr_ssl.config import load_config, validate_config  # noqa: E402
from seis_attr_ssl.data import SurveyManifest, read_manifest_json  # noqa: E402
from seis_attr_ssl.data.normalization import (  # noqa: E402
	compute_normalization_stats,
	write_normalization_stats,
)
from seis_attr_ssl.utils.cli import (  # noqa: E402
	parse_config_args,
	print_config_summary,
)

DEFAULT_CONFIG = Path(__file__).resolve().parent / 'configs' / 'mvp_mae.yaml'
MANIFEST_BUILD_HINT = (
	'build the manifest with `python proc/build_nopims_manifests.py --config '
	'proc/configs/build_nopims_manifests.yaml`'
)


def main() -> None:
	"""Compute NOPIMS normalization stats or print a dry-run summary."""
	args = parse_config_args(
		'Prepare normalization stats for NOPIMS base-seismic manifests.',
		DEFAULT_CONFIG,
	)
	config = validate_config(load_config(args.config))
	normalization = _required_mapping(config, 'normalization')
	pre_attribute = _required_mapping(normalization, 'pre_attribute')
	manifest_path = _manifest_train_path(config)
	clip_low, clip_high = _required_percentiles(pre_attribute)
	max_samples = _optional_positive_int(
		_optional_mapping(config, 'normalization_stats'),
		'max_samples',
	)
	seed = _optional_int(
		_optional_mapping(config, 'normalization_stats'),
		'seed',
		default=42,
	)
	eps = _required_float(pre_attribute, 'epsilon')

	if not manifest_path.is_file():
		message = _missing_manifest_message(manifest_path)
		if args.dry_run:
			print_config_summary(config)
			print(f'normalization_stats.manifest_path: {manifest_path}')
			print('normalization_stats.manifest_exists: false')
			print('normalization_stats.compute: skipped')
			print(f'hint: {MANIFEST_BUILD_HINT}')
			return
		raise FileNotFoundError(message)

	manifests = read_manifest_json(manifest_path)
	if args.dry_run:
		print_config_summary(config)
		print(f'normalization_stats.manifest_path: {manifest_path}')
		print('normalization_stats.manifest_exists: true')
		_print_manifest_counts(manifests)
		print('normalization_stats.compute: skipped')
		return

	for manifest in manifests:
		if manifest.base_seismic is None:
			msg = (
				f'manifest survey {manifest.survey_id!r} is missing '
				'base seismic metadata'
			)
			raise ValueError(msg)
		stats = compute_normalization_stats(
			manifest.root / manifest.base_seismic.path,
			survey_id=manifest.survey_id,
			grid_order=manifest.grid_order,
			clip_low_percentile=clip_low,
			clip_high_percentile=clip_high,
			max_samples=max_samples,
			seed=seed,
			eps=eps,
		)
		output_path = manifest.root / manifest.base_seismic.normalization_stats_path
		write_normalization_stats(stats, output_path)
		print(f'wrote normalization stats: {output_path}')


def _manifest_train_path(config: Mapping[str, object]) -> Path:
	manifests = config.get('manifests')
	if not isinstance(manifests, Mapping):
		msg = _manifest_path_error('manifests.train is required')
		raise TypeError(msg)
	path_value = manifests.get('train')
	if not isinstance(path_value, str) or not path_value:
		msg = _manifest_path_error(
			f'manifests.train must be a non-empty string; got {path_value!r}',
		)
		raise ValueError(msg)
	return Path(path_value)


def _manifest_path_error(reason: str) -> str:
	return f'{reason}. {MANIFEST_BUILD_HINT}.'


def _missing_manifest_message(manifest_path: Path) -> str:
	return _manifest_path_error(f'manifests.train does not exist: {manifest_path}')


def _print_manifest_counts(manifests: Sequence[SurveyManifest]) -> None:
	base_seismic_count = sum(
		manifest.base_seismic is not None for manifest in manifests
	)
	stats_existing_count = sum(
		manifest.base_seismic is not None
		and (manifest.root / manifest.base_seismic.normalization_stats_path).is_file()
		for manifest in manifests
	)
	print(f'normalization_stats.survey_count: {len(manifests)}')
	print(f'normalization_stats.base_seismic_count: {base_seismic_count}')
	print(f'normalization_stats.existing_stats_count: {stats_existing_count}')


def _required_mapping(parent: Mapping[str, object], key: str) -> Mapping[str, Any]:
	value = parent.get(key)
	if not isinstance(value, Mapping):
		msg = f'{key} must be a mapping'
		raise TypeError(msg)
	return value


def _optional_mapping(
	parent: Mapping[str, object],
	key: str,
) -> Mapping[str, object]:
	value = parent.get(key, {})
	if not isinstance(value, Mapping):
		msg = f'{key} must be a mapping'
		raise TypeError(msg)
	return value


def _optional_int(
	parent: Mapping[str, object],
	key: str,
	*,
	default: int,
) -> int:
	value = parent.get(key, default)
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
