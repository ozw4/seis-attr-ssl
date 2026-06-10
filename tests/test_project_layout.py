from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATHS = (
	Path('src/seis_attr_ssl/'),
	Path('proc/'),
	Path('proc/configs/'),
	Path('docs/'),
	Path('docs/mvp_spec.md'),
	Path('pyproject.toml'),
	Path('README.md'),
)


def test_required_project_layout_paths_exist() -> None:
	for relative_path in REQUIRED_PATHS:
		assert (PROJECT_ROOT / relative_path).exists()
