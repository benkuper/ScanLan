from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .release_matrix import (
    evaluate_release_matrix,
    load_release_evidence,
    load_release_requirements,
    write_default_promotion,
    write_release_matrix_report,
)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        prog="scanlan-release-matrix",
        description="Evaluate ScanLan V2 release evidence and fail closed on missing scenarios.",
    )
    command.add_argument("--requirements", type=Path, default=None)
    command.add_argument("--evidence", type=Path, nargs="+", required=True)
    command.add_argument("--report", type=Path, required=True)
    command.add_argument("--promote-defaults", type=Path, default=None)
    command.add_argument(
        "--no-verify-artifacts",
        action="store_true",
        help="Diagnostic-only: validate declared digests without reading artifact bytes.",
    )
    return command


def main(arguments: Sequence[str] | None = None) -> int:
    options = parser().parse_args(arguments)
    requirements = load_release_requirements(options.requirements)
    evidence = load_release_evidence(options.evidence)
    report = evaluate_release_matrix(
        requirements=requirements,
        evidence_records=evidence,
        verify_artifacts=not options.no_verify_artifacts,
    )
    write_release_matrix_report(options.report, report)
    if options.promote_defaults is not None:
        write_default_promotion(options.promote_defaults, report)
    print(json.dumps(report, separators=(",", ":"), allow_nan=False))
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
