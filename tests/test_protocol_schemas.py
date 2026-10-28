import json
from pathlib import Path

import pytest

from jynx_protocol.schema_loader import validate_instance

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "schemas" / "fixtures"

SCHEMAS = {
    "evidence_receipt": "EvidenceReceipt.schema.json",
    "judgment": "Judgment.schema.json",
    "move": "Move.schema.json",
    "round_manifest": "RoundManifest.schema.json",
    "tournament": "Tournament.schema.json",
}


@pytest.mark.parametrize("stem,schema_file", list(SCHEMAS.items()))
def test_valid_fixtures_pass_schema(stem, schema_file):
    data = json.loads((FIXTURES / f"{stem}.valid.json").read_text(encoding="utf-8"))
    errors = validate_instance(schema_file, data)
    assert errors == []


@pytest.mark.parametrize("stem,schema_file", list(SCHEMAS.items()))
def test_malformed_fixtures_fail_schema(stem, schema_file):
    data = json.loads((FIXTURES / f"{stem}.malformed.json").read_text(encoding="utf-8"))
    errors = validate_instance(schema_file, data)
    assert errors, f"expected schema errors for {stem}.malformed.json"


def test_round_manifest_pins_versions_and_digests():
    data = json.loads((FIXTURES / "round_manifest.valid.json").read_text(encoding="utf-8"))
    assert data["immutable"] is True
    pin = data["pinned_components"][0]
    assert pin["commit_sha"] == "0123456789abcdef0123456789abcdef01234567"
    assert pin["content_digest"]["algorithm"] == "sha256"
    assert len(pin["content_digest"]["hex"]) == 64


def test_evidence_receipt_immutable_id_and_provenance():
    data = json.loads((FIXTURES / "evidence_receipt.valid.json").read_text(encoding="utf-8"))
    assert data["evidence_id"].startswith("ev:")
    assert data["provenance"]["actors"][0]["roles"] == ["proposer"]


def test_judgment_typed_outcome_and_rubric():
    data = json.loads((FIXTURES / "judgment.valid.json").read_text(encoding="utf-8"))
    assert data["outcome"]["kind"] == "PASS"
    assert data["rubric"]["version"] == "0.1.0"
    assert data["evidence_refs"]
