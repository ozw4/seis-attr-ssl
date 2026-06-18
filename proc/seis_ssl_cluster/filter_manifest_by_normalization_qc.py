"""Thin entrypoint for amplitude-only normalization QC manifest filtering."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / 'src'
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from seis_ssl_cluster.utils.cli import run_pending_entrypoint  # noqa: E402

DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[1]
	/ 'configs'
	/ 'seis_ssl_cluster'
	/ 'filter_manifest_by_normalization_qc.yaml'
)


def main() -> None:
	"""Validate config and report pending manifest QC filtering."""
	run_pending_entrypoint(
		'Filter amplitude-only manifests by normalization QC.',
		DEFAULT_CONFIG,
	)


if __name__ == '__main__':
	main()
