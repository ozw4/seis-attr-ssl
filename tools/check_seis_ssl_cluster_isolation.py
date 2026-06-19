"""Validate that the Seis SSL Cluster MVP can be extracted cleanly."""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import yaml

EXPECTED_STAGES = {
	'build_nopims_manifests',
	'prepare_nopims_normalization_stats',
	'filter_manifest_by_normalization_qc',
	'train_amp_mae',
	'extract_embeddings',
	'cluster_embeddings',
	'visualize_clusters',
}
MANIFEST_PATH = Path('docs/seis_ssl_cluster_extraction_manifest.txt')
REQUIRED_MANIFEST_ENTRIES = (
	'README.md',
	'pyproject.toml',
	'docs/seis_ssl_cluster_mvp_spec.md',
	'docs/seis_ssl_cluster_runbook.md',
	'docs/seis_ssl_cluster_artifact_layout.md',
	'docs/seis_ssl_cluster_extraction_manifest.txt',
	'src/seis_ssl_cluster/',
	'proc/seis_ssl_cluster/',
	'proc/configs/seis_ssl_cluster/',
	'tests/seis_ssl_cluster/',
	'tools/check_seis_ssl_cluster_isolation.py',
)
EXTRACTION_ROOTS = (
	Path('src/seis_ssl_cluster'),
	Path('proc/seis_ssl_cluster'),
	Path('proc/configs/seis_ssl_cluster'),
	Path('tests/seis_ssl_cluster'),
)


def main() -> None:
	"""Run all extraction isolation checks."""
	repo_root = Path(__file__).resolve().parents[1]
	errors: list[str] = []
	errors.extend(_legacy_import_errors(repo_root / 'src' / 'seis_ssl_cluster'))
	errors.extend(_legacy_import_errors(repo_root / 'proc' / 'seis_ssl_cluster'))
	errors.extend(_config_errors(repo_root / 'proc' / 'configs' / 'seis_ssl_cluster'))
	errors.extend(_manifest_errors(repo_root))
	errors.extend(_existing_package_runtime_errors(repo_root / 'src' / 'seis_attr_ssl'))

	if errors:
		for error in errors:
			print(f'ERROR: {error}', file=sys.stderr)
		raise SystemExit(1)
	print('seis_ssl_cluster isolation checks passed')


def _legacy_import_errors(root: Path) -> list[str]:
	return [
		f'{_rel(path)} imports legacy package {imported}'
		for path in _python_files(root)
		for imported in _imported_modules(path)
		if imported == 'seis_attr_ssl' or imported.startswith('seis_attr_ssl.')
	]


def _existing_package_runtime_errors(root: Path) -> list[str]:
	return [
		f'{_rel(path)} imports new MVP package {imported}'
		for path in _python_files(root)
		for imported in _imported_modules(path)
		if imported == 'seis_ssl_cluster'
		or imported.startswith('seis_ssl_cluster.')
	]


def _config_errors(config_dir: Path) -> list[str]:
	errors = []
	for path in sorted(config_dir.glob('*.yaml')):
		text = path.read_text(encoding='utf-8')
		if 'seis_attr_ssl' in text:
			errors.append(f'{_rel(path)} references seis_attr_ssl')
		if '/workspace/artifacts/ssl' in text:
			errors.append(f'{_rel(path)} references legacy artifact root')
		payload = yaml.safe_load(text)
		if not isinstance(payload, Mapping):
			errors.append(f'{_rel(path)} is not a YAML mapping')
			continue
		stage = payload.get('stage')
		if stage not in EXPECTED_STAGES:
			errors.append(f'{_rel(path)} has unexpected stage {stage!r}')
		paths = payload.get('paths')
		if not isinstance(paths, Mapping):
			errors.append(f'{_rel(path)} has no paths mapping')
			continue
		artifact_root = paths.get('artifact_root')
		if artifact_root != '/workspace/artifacts/seis_ssl_cluster':
			errors.append(
				f'{_rel(path)} paths.artifact_root must be '
				'/workspace/artifacts/seis_ssl_cluster',
			)
	return errors


def _manifest_errors(repo_root: Path) -> list[str]:
	path = repo_root / MANIFEST_PATH
	if not path.is_file():
		return [f'missing extraction manifest: {MANIFEST_PATH}']
	entries = _manifest_entries(path)
	errors = [
		f'extraction manifest missing {entry}'
		for entry in REQUIRED_MANIFEST_ENTRIES
		if entry not in entries
	]
	for required_path in _required_extraction_paths(repo_root):
		relative = required_path.relative_to(repo_root).as_posix()
		if not _manifest_covers(relative, entries):
			errors.append(f'extraction manifest does not cover {relative}')
	return errors


def _manifest_entries(path: Path) -> tuple[str, ...]:
	entries = []
	for line in path.read_text(encoding='utf-8').splitlines():
		value = line.strip()
		if not value or value.startswith('#'):
			continue
		entries.append(value)
	return tuple(entries)


def _required_extraction_paths(repo_root: Path) -> tuple[Path, ...]:
	paths = [
		repo_root / 'README.md',
		repo_root / 'pyproject.toml',
		repo_root / MANIFEST_PATH,
		repo_root / 'docs' / 'seis_ssl_cluster_mvp_spec.md',
		repo_root / 'docs' / 'seis_ssl_cluster_runbook.md',
		repo_root / 'docs' / 'seis_ssl_cluster_artifact_layout.md',
		repo_root / 'tools' / 'check_seis_ssl_cluster_isolation.py',
	]
	for root in EXTRACTION_ROOTS:
		paths.extend(
			path
			for path in (repo_root / root).rglob('*')
			if path.is_file()
		)
	return tuple(sorted(paths))


def _manifest_covers(relative_path: str, entries: Sequence[str]) -> bool:
	for entry in entries:
		if entry.endswith('/') and relative_path.startswith(entry):
			return True
		if relative_path == entry:
			return True
	return False


def _python_files(root: Path) -> Iterable[Path]:
	return sorted(path for path in root.rglob('*.py') if path.is_file())


def _imported_modules(path: Path) -> tuple[str, ...]:
	tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
	imports = []
	for node in ast.walk(tree):
		if isinstance(node, ast.Import):
			imports.extend(alias.name for alias in node.names)
		elif isinstance(node, ast.ImportFrom):
			imports.append(node.module or '')
	return tuple(imports)


def _rel(path: Path) -> str:
	return path.relative_to(Path(__file__).resolve().parents[1]).as_posix()


if __name__ == '__main__':
	main()
