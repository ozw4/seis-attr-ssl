"""Dry-run entrypoint for NOPIMS attribute generation."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / 'src'
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from seis_attr_ssl.utils.cli import run_config_entrypoint  # noqa: E402

DEFAULT_CONFIG = (
	Path(__file__).resolve().parent / 'configs' / 'generate_attributes_nopims.yaml'
)


def main() -> None:
	"""Validate the attribute-generation config or report pending execution."""
	run_config_entrypoint(
		'Generate configured seismic attributes for NOPIMS volumes.',
		DEFAULT_CONFIG,
	)


if __name__ == '__main__':
	main()
