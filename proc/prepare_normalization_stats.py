"""Compute survey-wise normalization stats for a base seismic `.npy` volume."""

from __future__ import annotations

import sys
from collections.abc import Mapping
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
from seis_attr_ssl.utils.cli import (  # noqa: E402
	parse_config_args,
	print_config_summary,
)

DEFAULT_CONFIG = (
	Path(__file__).resolve().parent / 'configs' / 'mvp_prepare_stats.yaml'
)


def main() -> None:
	"""Compute normalization stats or print the configured target."""
	args = parse_config_args(
		'Prepare survey-wise normalization stats for a seismic volume.',
		DEFAULT_CONFIG,
	)
	config = validate_config(load_config(args.config))
	data = _required_mapping(config, 'data')
	normalization = _required_mapping(config, 'normalization')
	pre_attribute = _required_mapping(normalization, 'pre_attribute')
	stats_cfg = _required_mapping(config, 'normalization_stats')
	source_path = Path(_required_str(data, 'base_seismic_path'))
	output_path = Path(_required_str(stats_cfg, 'output_path'))
	survey_id = _required_str(stats_cfg, 'survey_id')
	max_samples = _optional_positive_int(stats_cfg, 'max_samples')
	seed = _required_int(stats_cfg, 'seed')
	clip_low, clip_high = _required_percentiles(pre_attribute)
	eps = _required_float(pre_attribute, 'epsilon')

	if args.dry_run:
		print_config_summary(config)
		print(f'normalization_stats.survey_id: {survey_id}')
		print(f'normalization_stats.source_path: {source_path}')
		print(f'normalization_stats.output_path: {output_path}')
		print(f'normalization_stats.max_samples: {max_samples}')
		print(f'normalization_stats.seed: {seed}')
		print('normalization_stats.compute: skipped')
		return

	stats = compute_normalization_stats(
		source_path,
		survey_id=survey_id,
		grid_order=data['grid_order'],
		clip_low_percentile=clip_low,
		clip_high_percentile=clip_high,
		max_samples=max_samples,
		seed=seed,
		eps=eps,
	)
	write_normalization_stats(stats, output_path)
	print(f'wrote normalization stats: {output_path}')


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
