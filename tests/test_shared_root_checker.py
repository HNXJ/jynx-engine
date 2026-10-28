import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "scripts" / "check_shared_root.py"


def test_shared_root_passes_on_repo_root():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(ROOT)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_shared_root_rejects_missing_directory(tmp_path):
    (tmp_path / "README.md").write_text("# tmp\n", encoding="utf-8")
    for name in ("artifacts", "docs", "src", "schemas", "tests", "tournaments"):
        (tmp_path / name).mkdir()
    (tmp_path / "schemas").rmdir()
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "schemas" in result.stderr
