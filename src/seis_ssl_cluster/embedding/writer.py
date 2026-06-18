"""Output path, metadata, and memmap helpers for embedding extraction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EmbeddingOutputPaths:
	"""Output files for one survey embedding extraction."""

	embeddings: Path
	valid_tokens: Path
	metadata: Path
	sum_tmp: Path
	count_tmp: Path


def output_paths(output_dir: str | Path, survey_id: str) -> EmbeddingOutputPaths:
	"""Return deterministic output paths for one survey."""
	root = Path(output_dir)
	return EmbeddingOutputPaths(
		embeddings=root / f'{survey_id}.embeddings.npy',
		valid_tokens=root / f'{survey_id}.valid_tokens.npy',
		metadata=root / f'{survey_id}.embedding_metadata.json',
		sum_tmp=root / f'.{survey_id}.embedding_sum.float32.npy',
		count_tmp=root / f'.{survey_id}.embedding_count.uint32.npy',
	)


def prepare_outputs(
	paths: EmbeddingOutputPaths,
	metadata: dict[str, object],
	*,
	skip_existing: bool,
) -> bool:
	"""Return true when matching existing outputs should be skipped."""
	existing_outputs = [
		path
		for path in (paths.embeddings, paths.valid_tokens, paths.metadata)
		if path.exists()
	]
	if not existing_outputs:
		paths.embeddings.parent.mkdir(parents=True, exist_ok=True)
		return False
	if not metadata_matches(paths.metadata, metadata):
		msg = (
			'existing embedding output metadata does not match current settings: '
			f'{paths.metadata}'
		)
		raise ValueError(msg)
	return skip_existing and paths.embeddings.exists() and paths.valid_tokens.exists()


def create_merge_memmaps(
	paths: EmbeddingOutputPaths,
	*,
	token_grid_shape_xyz: tuple[int, int, int],
	embedding_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
	"""Create restartable temporary memmaps for embedding sums and counts."""
	paths.sum_tmp.parent.mkdir(parents=True, exist_ok=True)
	sum_array = np.lib.format.open_memmap(
		paths.sum_tmp,
		mode='w+',
		dtype=np.float32,
		shape=(*token_grid_shape_xyz, embedding_dim),
	)
	count_array = np.lib.format.open_memmap(
		paths.count_tmp,
		mode='w+',
		dtype=np.uint32,
		shape=token_grid_shape_xyz,
	)
	sum_array[...] = 0.0
	count_array[...] = 0
	return sum_array, count_array


def write_metadata(path: str | Path, metadata: dict[str, object]) -> None:
	"""Write deterministic extraction metadata."""
	metadata_path = Path(path)
	metadata_path.parent.mkdir(parents=True, exist_ok=True)
	metadata_path.write_text(
		json.dumps(metadata, indent=2, sort_keys=True) + '\n',
		encoding='utf-8',
	)


def metadata_matches(path: str | Path, metadata: dict[str, object]) -> bool:
	"""Return true when an existing metadata JSON matches exactly."""
	metadata_path = Path(path)
	if not metadata_path.is_file():
		return False
	try:
		existing = json.loads(metadata_path.read_text(encoding='utf-8'))
	except json.JSONDecodeError:
		return False
	return existing == metadata


def cleanup_temp_outputs(paths: EmbeddingOutputPaths) -> None:
	"""Remove temporary merge arrays left after a successful run."""
	for path in (paths.sum_tmp, paths.count_tmp):
		if path.exists():
			path.unlink()


def file_sha256(path: str | Path) -> str:
	"""Return the SHA-256 hex digest for a file."""
	digest = hashlib.sha256()
	with Path(path).open('rb') as file_obj:
		for block in iter(lambda: file_obj.read(1024 * 1024), b''):
			digest.update(block)
	return digest.hexdigest()


__all__ = [
	'EmbeddingOutputPaths',
	'cleanup_temp_outputs',
	'create_merge_memmaps',
	'file_sha256',
	'metadata_matches',
	'output_paths',
	'prepare_outputs',
	'write_metadata',
]
