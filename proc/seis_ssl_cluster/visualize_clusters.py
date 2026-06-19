"""Thin entrypoint for amplitude-only cluster visualization."""

from __future__ import annotations

import importlib
import json
import sys
from argparse import ArgumentParser
from collections.abc import Mapping, Sequence
from numbers import Integral
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2] / 'src'
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from seis_ssl_cluster.config import load_config, validate_config  # noqa: E402
from seis_ssl_cluster.utils.cli import print_config_summary  # noqa: E402

DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[1]
	/ 'configs'
	/ 'seis_ssl_cluster'
	/ 'visualize_clusters.yaml'
)


def main() -> None:
	"""Run amplitude-only cluster visualization or print a dry-run summary."""
	parser = ArgumentParser(description='Visualize amplitude-only clusters.')
	parser.add_argument(
		'--config',
		type=Path,
		default=DEFAULT_CONFIG,
		help='Path to a YAML configuration file.',
	)
	parser.add_argument(
		'--dry-run',
		action='store_true',
		help='Validate the config and print a run summary without executing.',
	)
	args = parser.parse_args()

	config = validate_config(load_config(args.config))
	if args.dry_run:
		print_config_summary(config)
		print('execution: dry-run; visualization skipped')
		return

	result = run_cluster_visualization(config)
	print(
		f'visualization: wrote {result["png_count"]} PNG(s), '
		f'{result["voxel_count"]} voxel label file(s), '
		f'and {result["summary_count"]} summary set(s)',
	)


def run_cluster_visualization(config: Mapping[str, object]) -> dict[str, int]:
	"""Run cluster-map reconstruction, summaries, and configured PNG rendering."""
	reconstruct = importlib.import_module('seis_ssl_cluster.clustering.reconstruct')
	summaries = importlib.import_module('seis_ssl_cluster.clustering.summaries')
	clusters = importlib.import_module('seis_ssl_cluster.visualization.clusters')
	clustering = _required_mapping(config, 'clustering')
	visualization = _required_mapping(config, 'visualization')
	input_dir = _required_path(clustering, 'input_dir', 'clustering')
	output_dir = _required_path(visualization, 'output_dir', 'visualization')
	reconstruct_voxel = _bool(
		visualization.get('reconstruct_voxel', False),
		'visualization.reconstruct_voxel',
	)
	modes = _modes(visualization.get('modes', ['token']))
	slice_request = clusters.ClusterSliceRequest(
		xy_slices=_int_tuple(visualization.get('xy_slices', ()), 'xy_slices'),
		xz_slices=_int_tuple(visualization.get('xz_slices', ()), 'xz_slices'),
	)
	dpi = _positive_int(visualization.get('dpi', 160), 'visualization.dpi')
	invalid_color = str(visualization.get('invalid_color', 'lightgray'))
	underlay_cfg = _optional_mapping(visualization, 'amplitude_underlay')
	underlay_enabled = _bool(
		underlay_cfg.get('enabled', False),
		'visualization.amplitude_underlay.enabled',
	)
	underlay_alpha = _fraction(
		underlay_cfg.get('alpha', 0.35),
		'visualization.amplitude_underlay.alpha',
	)
	summary_cfg = _optional_mapping(visualization, 'summaries')
	summaries_enabled = _bool(
		summary_cfg.get('enabled', True),
		'visualization.summaries.enabled',
	)
	include_amplitude = _bool(
		summary_cfg.get('include_amplitude_norm', False),
		'visualization.summaries.include_amplitude_norm',
	)

	png_count = 0
	voxel_count = 0
	summary_count = 0
	for k_dir in _label_k_dirs(input_dir):
		k = int(k_dir.name.removeprefix('k'))
		summary_inputs = []
		for token_path in sorted(k_dir.glob('*.cluster_labels_token.npy')):
			survey_id = token_path.name.removesuffix('.cluster_labels_token.npy')
			metadata_path = k_dir / f'{survey_id}.cluster_label_metadata.json'
			metadata = _load_metadata(metadata_path)
			embedding_metadata = _embedding_metadata(metadata)
			embedding_input = metadata.get('embedding_input')
			embeddings_path = None
			if isinstance(embedding_input, Mapping):
				value = embedding_input.get('embeddings_path')
				if isinstance(value, str):
					embeddings_path = Path(value)
			summary_inputs.append(
				summaries.ClusterSummaryInput(
					survey_id=survey_id,
					labels_path=token_path,
					metadata_path=metadata_path,
					embeddings_path=embeddings_path,
				),
			)
			token_labels = np.load(token_path, mmap_mode='r')
			amplitude = (
				_open_amplitude(embedding_metadata) if underlay_enabled else None
			)
			if 'token' in modes:
				token_amplitude = _amplitude_underlay_for_labels(
					amplitude,
					token_labels,
					embedding_metadata,
				)
				png_count += len(
					clusters.save_cluster_slice_pngs(
						token_labels,
						survey_id=survey_id,
						k=k,
						mode='token',
						output_dir=output_dir / 'token',
						slices=slice_request,
						amplitude=token_amplitude,
						amplitude_alpha=underlay_alpha,
						invalid_color=invalid_color,
						dpi=dpi,
					),
				)
			voxel_path = k_dir / f'{survey_id}.cluster_labels_voxel.npy'
			if reconstruct_voxel or 'voxel' in modes:
				reconstruct.reconstruct_labels_for_survey(
					token_path,
					metadata_path=metadata_path,
					write_voxel_labels=True,
				)
				voxel_count += 1
			if 'voxel' in modes:
				voxel_labels = np.load(voxel_path, mmap_mode='r')
				png_count += len(
					clusters.save_cluster_slice_pngs(
						voxel_labels,
						survey_id=survey_id,
						k=k,
						mode='voxel',
						output_dir=output_dir / 'voxel',
						slices=slice_request,
						amplitude=amplitude,
						amplitude_alpha=underlay_alpha,
						invalid_color=invalid_color,
						dpi=dpi,
					),
				)
		if summaries_enabled and summary_inputs:
			summaries.write_cluster_summaries(
				summary_inputs,
				k=k,
				output_dir=output_dir / f'k{k}',
				include_amplitude_norm=include_amplitude,
			)
			summary_count += 1
	return {
		'png_count': png_count,
		'voxel_count': voxel_count,
		'summary_count': summary_count,
	}


def _label_k_dirs(input_dir: Path) -> list[Path]:
	labels_root = input_dir / 'labels'
	if not labels_root.is_dir():
		msg = f'clustering labels directory does not exist: {labels_root}'
		raise FileNotFoundError(msg)
	k_dirs = [
		path
		for path in labels_root.iterdir()
		if path.is_dir()
		and path.name.startswith('k')
		and path.name.removeprefix('k').isdigit()
	]
	if not k_dirs:
		msg = f'no k label directories found under {labels_root}'
		raise ValueError(msg)
	return sorted(k_dirs, key=lambda path: int(path.name.removeprefix('k')))


def _load_metadata(path: Path) -> dict[str, object]:
	if not path.is_file():
		return {}
	payload = json.loads(path.read_text(encoding='utf-8'))
	if not isinstance(payload, dict):
		return {}
	return payload


def _embedding_metadata(label_metadata: Mapping[str, object]) -> dict[str, object]:
	embedding_input = label_metadata.get('embedding_input')
	if not isinstance(embedding_input, Mapping):
		return label_metadata.copy()
	metadata_path = embedding_input.get('metadata_path')
	if not isinstance(metadata_path, str) or not Path(metadata_path).is_file():
		return label_metadata.copy()
	payload = json.loads(Path(metadata_path).read_text(encoding='utf-8'))
	if not isinstance(payload, dict):
		return label_metadata.copy()
	return {**payload, **label_metadata}


def _open_amplitude(metadata: Mapping[str, object]) -> np.ndarray | None:
	value = metadata.get('source_amplitude_path')
	if not isinstance(value, str) or not Path(value).is_file():
		return None
	array = np.load(value, mmap_mode='r')
	if array.ndim != 3:
		return None
	return array


def _amplitude_underlay_for_labels(
	amplitude: np.ndarray | None,
	labels: np.ndarray,
	metadata: Mapping[str, object],
) -> np.ndarray | None:
	if amplitude is None:
		return None
	if amplitude.shape == labels.shape:
		return amplitude
	patch = _metadata_xyz(metadata.get('patch_size', (1, 1, 1)), 'patch_size')
	padded_shape = tuple(
		label_axis * patch_axis
		for label_axis, patch_axis in zip(labels.shape, patch, strict=True)
	)
	if any(
		amplitude_axis > padded_axis
		for amplitude_axis, padded_axis in zip(
			amplitude.shape,
			padded_shape,
			strict=True,
		)
	):
		msg = (
			'amplitude underlay shape is incompatible with token labels; '
			f'got amplitude={amplitude.shape!r}, labels={labels.shape!r}, '
			f'patch_size={patch!r}'
		)
		raise ValueError(msg)
	return _downsample_amplitude_to_tokens(amplitude, labels.shape, patch)


def _downsample_amplitude_to_tokens(
	amplitude: np.ndarray,
	token_shape: tuple[int, int, int],
	patch: tuple[int, int, int],
) -> np.ndarray:
	underlay = np.empty(token_shape, dtype=np.float32)
	for token_x in range(token_shape[0]):
		x_start = token_x * patch[0]
		x_stop = min(x_start + patch[0], amplitude.shape[0])
		for token_y in range(token_shape[1]):
			y_start = token_y * patch[1]
			y_stop = min(y_start + patch[1], amplitude.shape[1])
			for token_z in range(token_shape[2]):
				z_start = token_z * patch[2]
				z_stop = min(z_start + patch[2], amplitude.shape[2])
				values = np.asarray(
					amplitude[x_start:x_stop, y_start:y_stop, z_start:z_stop],
					dtype=np.float32,
				)
				finite = values[np.isfinite(values)]
				underlay[token_x, token_y, token_z] = (
					float(finite.mean()) if finite.size else np.nan
				)
	return underlay


def _metadata_xyz(value: object, name: str) -> tuple[int, int, int]:
	if (
		not isinstance(value, Sequence)
		or isinstance(value, str)
		or len(value) != 3
		or any(
			isinstance(item, bool) or not isinstance(item, Integral)
			for item in value
		)
	):
		msg = f'{name} must be a length-3 integer sequence; got {value!r}'
		raise TypeError(msg)
	xyz = tuple(int(item) for item in value)
	if any(item <= 0 for item in xyz):
		msg = f'{name} values must be positive; got {xyz!r}'
		raise ValueError(msg)
	return xyz


def _required_mapping(
	parent: Mapping[str, object],
	key: str,
) -> Mapping[str, object]:
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


def _required_path(
	parent: Mapping[str, object],
	key: str,
	prefix: str,
) -> Path:
	value = parent.get(key)
	if not isinstance(value, str) or not value:
		msg = f'{prefix}.{key} must be a non-empty string'
		raise TypeError(msg)
	return Path(value)


def _int_tuple(value: object, name: str) -> tuple[int, ...]:
	if value is None:
		return ()
	if not isinstance(value, Sequence) or isinstance(value, str):
		msg = f'visualization.{name} must be a sequence of integers'
		raise TypeError(msg)
	if any(isinstance(item, bool) or not isinstance(item, Integral) for item in value):
		msg = f'visualization.{name} must be a sequence of integers'
		raise TypeError(msg)
	return tuple(int(item) for item in value)


def _modes(value: object) -> tuple[str, ...]:
	if isinstance(value, str):
		modes = (value,)
	elif isinstance(value, Sequence):
		modes = tuple(str(item) for item in value)
	else:
		msg = f'visualization.modes must be a string or sequence; got {value!r}'
		raise TypeError(msg)
	unknown = sorted(set(modes) - {'token', 'voxel'})
	if unknown:
		msg = f'unknown visualization modes: {unknown!r}'
		raise ValueError(msg)
	return modes


def _bool(value: object, name: str) -> bool:
	if not isinstance(value, bool):
		msg = f'{name} must be a boolean; got {value!r}'
		raise TypeError(msg)
	return value


def _positive_int(value: object, name: str) -> int:
	if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
		msg = f'{name} must be a positive integer; got {value!r}'
		raise ValueError(msg)
	return int(value)


def _fraction(value: object, name: str) -> float:
	if not isinstance(value, int | float) or isinstance(value, bool):
		msg = f'{name} must be a number; got {value!r}'
		raise TypeError(msg)
	fraction = float(value)
	if fraction < 0.0 or fraction > 1.0:
		msg = f'{name} must be in [0, 1]; got {value!r}'
		raise ValueError(msg)
	return fraction


if __name__ == '__main__':
	main()
