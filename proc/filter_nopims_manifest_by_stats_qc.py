"""Generate clean NOPIMS manifest/path-list files from stats QC."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / 'src'
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from seis_attr_ssl.data.manifest_filter import (  # noqa: E402
	FilteredManifestStatsQcResult,
	filter_manifests_by_stats_qc,
)
from seis_attr_ssl.data.normalization_qc import (  # noqa: E402
	NormalizationStatsQcThresholds,
	normalization_qc_report_to_dict,
)
from seis_attr_ssl.data.path_list import load_npy_path_list  # noqa: E402
from seis_attr_ssl.data.schema import (  # noqa: E402
	read_manifest_json,
	write_manifest_json,
)


def main() -> None:
	"""Run stats QC and write clean manifest/path-list outputs."""
	args = _parse_args()
	manifests = read_manifest_json(args.manifest)
	path_entries = load_npy_path_list(args.input_path_list)
	thresholds = NormalizationStatsQcThresholds(
		iqr_min=args.iqr_min_threshold,
		norm_abs_max=args.norm_abs_max_threshold,
	)
	result = filter_manifests_by_stats_qc(
		manifests,
		path_entries,
		nopims_root=args.nopims_root,
		thresholds=thresholds,
		stats_dir=args.stats_dir,
	)

	write_outputs = not args.dry_run
	_print_summary(args, result, write=write_outputs)
	if args.fail_if_empty and (
		not result.clean_manifests or not result.clean_path_entries
	):
		msg = 'clean manifest or clean path-list is empty'
		raise SystemExit(msg)
	if not write_outputs:
		return

	_write_qc_json(result, args.output_qc_json)
	_write_excluded_surveys(result, args.output_excluded_surveys)
	args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
	write_manifest_json(result.clean_manifests, args.output_manifest)
	_write_clean_path_list(result, args.output_path_list)


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description='Filter NOPIMS manifests and path-lists using stats QC.',
	)
	parser.add_argument(
		'--manifest',
		type=Path,
		required=True,
		help='Input survey manifest JSON.',
	)
	parser.add_argument(
		'--input-path-list',
		type=Path,
		required=True,
		help='Input `.npy` path-list.',
	)
	parser.add_argument(
		'--nopims-root',
		type=Path,
		required=True,
		help='NOPIMS root used to derive survey_id values from path-list entries.',
	)
	parser.add_argument(
		'--output-qc-json',
		type=Path,
		required=True,
		help='Output normalization stats QC report JSON.',
	)
	parser.add_argument(
		'--output-excluded-surveys',
		type=Path,
		required=True,
		help='Output text file of excluded survey_id values.',
	)
	parser.add_argument(
		'--output-manifest',
		type=Path,
		required=True,
		help='Output clean manifest JSON.',
	)
	parser.add_argument(
		'--output-path-list',
		type=Path,
		required=True,
		help='Output clean path-list.',
	)
	parser.add_argument(
		'--stats-dir',
		type=Path,
		default=None,
		help=(
			'Optional stats directory with {survey_id}.normalization_stats.json files.'
		),
	)
	parser.add_argument(
		'--iqr-min-threshold',
		type=float,
		default=1.0e-6,
		help='Minimum accepted IQR before excluding a survey.',
	)
	parser.add_argument(
		'--norm-abs-max-threshold',
		type=float,
		default=1.0e4,
		help='Maximum accepted normalized absolute clip endpoint.',
	)
	parser.add_argument(
		'--dry-run',
		action='store_true',
		help='Run QC and print summary without writing files.',
	)
	parser.add_argument(
		'--fail-if-empty',
		action='store_true',
		help='Exit non-zero if the clean manifest or clean path-list is empty.',
	)
	return parser.parse_args()


def _print_summary(
	args: argparse.Namespace,
	result: FilteredManifestStatsQcResult,
	*,
	write: bool,
) -> None:
	print(f'normalization_qc.manifest_path: {args.manifest}')
	print(f'normalization_qc.input_path_list: {args.input_path_list}')
	print(f'normalization_qc.total_surveys: {len(result.report.items)}')
	print(
		'normalization_qc.passed_surveys: '
		f'{sum(item.status == "pass" for item in result.report.items)}',
	)
	print(f'normalization_qc.excluded_surveys: {len(result.excluded_surveys)}')
	print(f'normalization_qc.clean_manifest: {args.output_manifest}')
	print(f'normalization_qc.clean_path_list: {args.output_path_list}')
	print(f'normalization_qc.write: {str(write).lower()}')


def _write_qc_json(
	result: FilteredManifestStatsQcResult,
	output_path: Path,
) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(
		json.dumps(
			normalization_qc_report_to_dict(result.report),
			indent=2,
			allow_nan=False,
		),
		encoding='utf-8',
	)


def _write_excluded_surveys(
	result: FilteredManifestStatsQcResult,
	output_path: Path,
) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	text = ''.join(f'{survey_id}\n' for survey_id in result.excluded_surveys)
	output_path.write_text(text, encoding='utf-8')


def _write_clean_path_list(
	result: FilteredManifestStatsQcResult,
	output_path: Path,
) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	text = ''.join(f'{entry}\n' for entry in result.clean_path_entries)
	output_path.write_text(text, encoding='utf-8')


if __name__ == '__main__':
	main()
