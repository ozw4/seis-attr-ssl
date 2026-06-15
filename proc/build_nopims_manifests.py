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
	scan_nopims_base_seismic_manifests_from_path_list,
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
	)/
	config = validate_config(load_config(args.config))
	paths = _required_mapping(config, 'paths')
	manifest_cfg = _required_mapping(config, 'manifest')
	nopims_root = Path(_required_str(paths, 'nopims_root'))
	data = _required_mapping(config, 'data')
	output_path = _manifest_output_path(manifest_cfg)
	input_path_list = Path(_required_str(manifest_cfg, 'input_path_list'))
	base_seismic_kind = _required_str(data, 'base_seismic_kind')
	normalization_stats_suffix = _optional_str(
		manifest_cfg,
		'normalization_stats_suffix',
		'.normalization_stats.json',
	)

	if args.dry_run:
		print_config_summary(config)
		_print_manifest_target(nopims_root, input_path_list, output_path)
		print('manifest scan: skipped')
		return

	result = scan_nopims_base_seismic_manifests_from_path_list(
		nopims_root=nopims_root,
		input_path_list=input_path_list,
		base_seismic_kind=base_seismic_kind,
		normalization_stats_suffix=normalization_stats_suffix,
	)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	write_manifest_json(result.manifests, output_path)
	_print_manifest_summary(result.summary(output_path=output_path))
	print(f'wrote manifest: {output_path}')


def _manifest_output_path(manifest_cfg: Mapping[str, Any]) -> Path:
	output_dir = Path(_required_str(manifest_cfg, 'output_dir'))
	output_name = _optional_str(manifest_cfg, 'output_name', DEFAULT_OUTPUT_NAME)
	return output_dir / output_name


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


def _optional_str(parent: Mapping[str, object], key: str, default: str) -> str:
	value = parent.get(key, default)
	if not isinstance(value, str):
		msg = f'{key} must be a string; got {value!r}'
		raise TypeError(msg)
	return value


def _print_manifest_target(
	nopims_root: Path,
	input_path_list: Path,
	output_path: Path,
) -> None:
	print(f'manifest.nopims_root: {nopims_root}')
	print(f'manifest.input_path_list: {input_path_list}')
	print(f'manifest.output_path: {output_path}')


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
