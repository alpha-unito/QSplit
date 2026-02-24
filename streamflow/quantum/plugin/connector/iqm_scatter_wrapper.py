from __future__ import annotations

import sys

from .iqm_wms_control import install_iqm_runtime_hooks


def main() -> int:
    install_iqm_runtime_hooks()
    from qsplit.cwl.cli.scatter import main as scatter_main

    sys.argv = ["cli_scatter", *sys.argv[1:]]
    scatter_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
