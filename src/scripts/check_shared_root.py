#!/usr/bin/env python3
"""Verify the seven required Jynx repo root entries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REQUIRED_DIRS = ("artifacts", "docs", "src", "schemas", "tests", "tournaments")
REQUIRED_FILES = ("README.md",)


def check_root(root: Path) -> list[str]:
    errors: list[str] = []
    for name in REQUIRED_DIRS:
        path = root / name
        if not path.is_dir():
            errors.append(f"missing directory: {name}/")
    for name in REQUIRED_FILES:
        path = root / name
        if not path.is_file():
            errors.append(f"missing file: {name}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check Jynx shared repository root layout.")
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Repository root to inspect (default: current directory)",
    )
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    errors = check_root(root)
    if errors:
        for err in errors:
            print(f"FAIL: {err}", file=sys.stderr)
        return 1
    print(f"OK: {root} has all seven required root entries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
