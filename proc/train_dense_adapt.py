"""Dry-run entrypoint for dense adaptation training."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / 'src'
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from seis_attr_ssl.utils.cli import run_config_entrypoint  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parent / 'configs' / 'mvp_dense_adapt.yaml'


def main() -> None:
	"""Validate the dense adaptation config or report pending training."""
	run_config_entrypoint(
		'Train the dense adaptation decoder from a pretrained MAE.',
		DEFAULT_CONFIG,
	)


if __name__ == '__main__':
	main()
