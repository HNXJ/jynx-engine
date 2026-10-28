from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from jynx_protocol.schema_loader import validate_instance
from jynx_protocol.semver import (
    ValidationResult,
    build_adjacency_from_tournament,
    contains_prohibited_path,
    detect_cycle,
    logical_refs,
    ranges_compatible,
    registry_index,
    satisfies_range,
)


def load_tournament_document(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Tournament document must be a mapping")
    return data


def load_held_out_registry(tournament_dir: Path) -> dict[str, Any]:
    registry_path = tournament_dir / "sealed" / "held_out_registry.json"
    if not registry_path.is_file():
        return {"commitments": []}
    return json.loads(registry_path.read_text(encoding="utf-8"))


def validate_tournament(
    doc: dict[str, Any],
    *,
    tournament_dir: Path | None = None,
) -> ValidationResult:
    result = ValidationResult(ok=True)

    schema_errors = validate_instance("Tournament.schema.json", doc)
    for err in schema_errors:
        result.add("schema", err)

    if not result.ok and all(i.code == "schema" for i in result.issues):
        # Continue semantic checks when partial doc still parseable
        pass

    protocol_version = doc.get("protocol_version", "")
    rules = doc.get("rules_contract") or {}
    if rules and not satisfies_range(protocol_version, rules.get("version_range", "")):
        result.add(
            "incompatible_contract",
            f"protocol_version {protocol_version!r} does not satisfy rules_contract "
            f"range {rules.get('version_range')!r}",
            "rules_contract.version_range",
        )

    idx = registry_index(doc)
    for path, ref in logical_refs(doc):
        if not ref:
            result.add("missing_reference", "missing logical ref", path)
            continue
        if contains_prohibited_path(ref):
            result.add(
                "prohibited_path",
                f"logical_id must not be a path or URL: {ref.get('logical_id')!r}",
                path,
            )
        lid = ref.get("logical_id")
        plugin = idx.get(lid)
        if not plugin:
            result.add(
                "missing_plugin",
                f"no plugin_registry entry provides logical_id {lid!r}",
                path,
            )
            continue
        if not ranges_compatible(ref.get("version_range", ""), plugin.get("version_range", "")):
            result.add(
                "incompatible_contract",
                f"version_range {ref.get('version_range')!r} incompatible with plugin "
                f"{plugin.get('logical_id')!r} range {plugin.get('version_range')!r}",
                path,
            )

    adj = build_adjacency_from_tournament(doc)
    cycle = detect_cycle(adj)
    if cycle:
        result.add(
            "dependency_cycle",
            "dependency cycle detected: " + " -> ".join(cycle),
            "helper_dependency_dag",
        )

    registry = load_held_out_registry(tournament_dir) if tournament_dir else {"commitments": []}
    commitments = {
        c.get("commitment_ref"): c for c in registry.get("commitments") or [] if c.get("commitment_ref")
    }
    for i, held in enumerate(doc.get("held_out_evidence") or []):
        cref = held.get("commitment_ref")
        if not cref or cref not in commitments:
            result.add(
                "held_out_unavailable",
                f"held-out commitment_ref {cref!r} not registered for judge access",
                f"held_out_evidence[{i}].commitment_ref",
            )
            continue
        reg = commitments[cref]
        if reg.get("evidence_id") != held.get("evidence_id"):
            result.add(
                "held_out_unavailable",
                "held-out evidence_id does not match sealed registry",
                f"held_out_evidence[{i}].evidence_id",
            )
        reg_digest = (reg.get("content_digest") or {}).get("hex")
        doc_digest = (held.get("content_digest") or {}).get("hex")
        if reg_digest != doc_digest:
            result.add(
                "held_out_unavailable",
                "held-out content_digest does not match sealed registry commitment",
                f"held_out_evidence[{i}].content_digest",
            )
        if reg.get("player_accessible") is True:
            result.add(
                "held_out_unavailable",
                "held-out evidence marked player_accessible in registry (must be false)",
                f"held_out_evidence[{i}]",
            )

    return result


def validate_tournament_file(path: Path) -> ValidationResult:
    doc = load_tournament_document(path)
    tournament_dir = path.parent if path.name.startswith("tournament") else path.parent
    return validate_tournament(doc, tournament_dir=tournament_dir)
