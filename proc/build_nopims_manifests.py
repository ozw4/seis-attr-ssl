"""Build NOPIMS manifests from configured base seismic `.npy` volumes."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[1] / 'src'
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from seis_attr_ssl.config import load_config, validate_config  # noqa: E402
from seis_attr_ssl.data import (  # noqa: E402
	ManifestBuildSummary,
	scan_nopims_base_seismic_manifests,
	write_manifest_json,
)
from seis_attr_ssl.utils.cli import (  # noqa: E402
	parse_config_args,
	print_config_summary,
)

DEFAULT_CONFIG = (
	Path(__file__).resolve().parent / 'configs' / 'build_nopims_manifests.yaml'
)
DEFAULT_OUTPUT_NAME = 'nopims_base_seismic_manifests.json'


def main() -> None:
	"""Build NOPIMS manifests or print a dry-run summary."""
	args = parse_config_args(
		'Build NOPIMS manifest files from configured seismic volumes.',
		DEFAULT_CONFIG,
	)
	config = validate_config(load_config(args.config))
	paths = _required_mapping(config, 'paths')
	manifest_cfg = _required_mapping(config, 'manifest')
	nopims_root = Path(_required_str(paths, 'nopims_root'))
	data = _required_mapping(config, 'data')
	output_path = _manifest_output_path(manifest_cfg)
	scan_pattern = _required_str(manifest_cfg, 'scan_pattern')
	base_seismic_kind = _required_str(data, 'base_seismic_kind')
	base_seismic_name_hints = tuple(_required_str_list(
		manifest_cfg,
		'base_seismic_name_hints',
	))
	normalization_stats_name = _required_str(
		manifest_cfg,
		'normalization_stats_name',
	)

	if args.dry_run:
		print_config_summary(config)
		_print_manifest_target(nopims_root, output_path, scan_pattern)
		print('manifest scan: skipped')
		return

	result = scan_nopims_base_seismic_manifests(
		nopims_root,
		scan_pattern,
		base_seismic_kind=base_seismic_kind,
		base_seismic_name_hints=base_seismic_name_hints,
		normalization_stats_name=normalization_stats_name,
	)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	write_manifest_json(result.manifests, output_path)
	_print_manifest_summary(result.summary(output_path=output_path))
	print(f'wrote manifest: {output_path}')


def _manifest_output_path(manifest_cfg: Mapping[str, Any]) -> Path:
	output_dir = Path(_required_str(manifest_cfg, 'output_dir'))
	return output_dir / DEFAULT_OUTPUT_NAME


def _required_mapping(parent: Mapping[str, object], key: str) -> Mapping[str, Any]:
	value = parent.get(key)
	if not isinstance(value, Mapping):
		msg = f'{key} must be a mapping'
		raise TypeError(msg)
	return value


def _required_str(parent: Mapping[str, object], key: str) -> str:
	value = parent.get(key)
	if not isinstance(value, str):
		msg = f'{key} must be a string; got {value!r}'
		raise TypeError(msg)
	return value


def _required_str_list(parent: Mapping[str, object], key: str) -> list[str]:
	value = parent.get(key)
	if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
		msg = f'{key} must be a list of strings; got {value!r}'
		raise TypeError(msg)
	return value


def _print_manifest_target(
	nopims_root: Path,
	output_path: Path,
	scan_pattern: str,
) -> None:
	print(f'manifest.nopims_root: {nopims_root}')
	print(f'manifest.output_path: {output_path}')
	print(f'manifest.scan_pattern: {scan_pattern}')


def _print_manifest_summary(summary: ManifestBuildSummary) -> None:
	print(f'manifest.survey_count: {summary.survey_count}')
	print(f'manifest.base_seismic_count: {summary.base_seismic_count}')
	print(f'manifest.attribute_volume_count: {summary.attribute_volume_count}')
	if summary.missing_attributes_by_survey:
		for survey_id, missing in summary.missing_attributes_by_survey.items():
			print(f'manifest.missing_attributes.{survey_id}: {", ".join(missing)}')
	else:
		print('manifest.missing_attributes: none')

	if summary.unknown_attribute_counts:
		for name, count in summary.unknown_attribute_counts.items():
			print(f'manifest.unknown_attribute.{name}: {count}')
	else:
		print('manifest.unknown_attributes: none')

if __name__ == '__main__':
	main()
