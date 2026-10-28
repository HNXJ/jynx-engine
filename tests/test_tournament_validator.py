import copy
import json
from pathlib import Path

import pytest

from jynx_protocol.tournament_validator import validate_tournament, validate_tournament_file

ROOT = Path(__file__).resolve().parents[1]
VALID = json.loads((ROOT / "schemas/fixtures/tournament.valid.json").read_text(encoding="utf-8"))
TOURNAMENT_DIR = ROOT / "tournaments" / "neuro-omission-growth"


def test_example_tournament_yaml_passes():
    result = validate_tournament_file(TOURNAMENT_DIR / "tournament.yaml")
    assert result.ok, result.issues


def test_rejects_missing_plugin():
    doc = copy.deepcopy(VALID)
    doc["rules_contract"] = {
        "logical_id": "jynx.rules.unknown-contract",
        "version_range": "^0.1.0",
    }
    result = validate_tournament(doc, tournament_dir=TOURNAMENT_DIR)
    assert not result.ok
    assert any(i.code == "missing_plugin" for i in result.issues)


def test_rejects_dependency_cycle():
    doc = copy.deepcopy(VALID)
    doc["helper_dependency_dag"]["jynx.rules.growth-gate"] = ["jynx.judge.neuro-growth-rubric"]
    result = validate_tournament(doc, tournament_dir=TOURNAMENT_DIR)
    assert not result.ok
    assert any(i.code == "dependency_cycle" for i in result.issues)


def test_rejects_incompatible_protocol_version():
    doc = copy.deepcopy(VALID)
    doc["protocol_version"] = "2.0.0"
    doc["rules_contract"]["version_range"] = "0.1.0"
    result = validate_tournament(doc, tournament_dir=TOURNAMENT_DIR)
    assert not result.ok
    assert any(i.code == "incompatible_contract" for i in result.issues)


def test_rejects_unavailable_held_out():
    doc = copy.deepcopy(VALID)
    doc["held_out_evidence"][0]["commitment_ref"] = "does-not-exist"
    result = validate_tournament(doc, tournament_dir=TOURNAMENT_DIR)
    assert not result.ok
    assert any(i.code == "held_out_unavailable" for i in result.issues)


def test_rejects_prohibited_absolute_path_reference():
    doc = copy.deepcopy(VALID)
    doc["rules_contract"]["logical_id"] = "/etc/passwd"
    result = validate_tournament(doc, tournament_dir=TOURNAMENT_DIR)
    assert not result.ok
    assert any(i.code == "prohibited_path" for i in result.issues)
