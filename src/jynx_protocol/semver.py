from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass
class ValidationIssue:
    code: str
    message: str
    path: str = ""


@dataclass
class ValidationResult:
    ok: bool
    issues: list[ValidationIssue] = field(default_factory=list)

    def add(self, code: str, message: str, path: str = "") -> None:
        self.ok = False
        self.issues.append(ValidationIssue(code=code, message=message, path=path))


_SEMVER = re.compile(
    r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<prerelease>[0-9A-Za-z.-]+))?(?:\+(?P<build>[0-9A-Za-z.-]+))?$"
)


def parse_version(version: str) -> tuple[int, int, int] | None:
    m = _SEMVER.match(version.strip())
    if not m:
        return None
    return int(m.group("major")), int(m.group("minor")), int(m.group("patch"))


def satisfies_range(version: str, version_range: str) -> bool:
    """Minimal semver range support: exact, ^X.Y.Z, ~X.Y.Z."""
    v = parse_version(version)
    if v is None:
        return False
    vr = version_range.strip()
    if vr.startswith("^"):
        base = parse_version(vr[1:])
        if base is None:
            return False
        maj, mino, _ = base
        if v[0] != maj:
            return False
        if maj == 0:
            return v[1] == mino and v >= base
        return v >= base
    if vr.startswith("~"):
        base = parse_version(vr[1:])
        if base is None:
            return False
        return v[0] == base[0] and v[1] == base[1] and v >= base
    exact = parse_version(vr)
    return exact == v if exact else False


def ranges_compatible(requested: str, provided: str) -> bool:
    """True if some version could satisfy both ranges (conservative check)."""
    if requested.strip() == provided.strip():
        return True
    # Sample pinned versions from requested range
    candidates: list[str] = []
    if requested.startswith("^"):
        base = parse_version(requested[1:])
        if base:
            candidates.append(f"{base[0]}.{base[1]}.{base[2]}")
    elif requested.startswith("~"):
        base = parse_version(requested[1:])
        if base:
            candidates.append(f"{base[0]}.{base[1]}.{base[2]}")
    else:
        if parse_version(requested):
            candidates.append(requested)
    return any(satisfies_range(c, provided) for c in candidates)


def detect_cycle(adjacency: dict[str, list[str]]) -> list[str] | None:
    visited: set[str] = set()
    stack: set[str] = set()
    path: list[str] = []

    def dfs(node: str) -> list[str] | None:
        visited.add(node)
        stack.add(node)
        path.append(node)
        for nxt in adjacency.get(node, []):
            if nxt not in visited:
                found = dfs(nxt)
                if found:
                    return found
            elif nxt in stack:
                cycle_start = path.index(nxt)
                return path[cycle_start:] + [nxt]
        path.pop()
        stack.remove(node)
        return None

    for node in adjacency:
        if node not in visited:
            found = dfs(node)
            if found:
                return found
    return None


def build_adjacency_from_tournament(doc: dict[str, Any]) -> dict[str, list[str]]:
    adj: dict[str, list[str]] = {}
    dag = doc.get("helper_dependency_dag") or {}
    for node, deps in dag.items():
        adj.setdefault(node, list(deps or []))
    for plugin in doc.get("plugin_registry") or []:
        lid = plugin.get("logical_id")
        if lid:
            adj.setdefault(lid, list(plugin.get("depends_on") or []))
    return adj


def registry_index(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    idx: dict[str, dict[str, Any]] = {}
    for plugin in doc.get("plugin_registry") or []:
        lid = plugin.get("logical_id")
        if lid:
            idx[lid] = plugin
        for provided in plugin.get("provides") or []:
            idx[provided] = plugin
    return idx


def logical_refs(doc: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    yield "rules_contract", doc.get("rules_contract") or {}
    if doc.get("arena_contract"):
        yield "arena_contract", doc["arena_contract"]
    if doc.get("engine_contract"):
        yield "engine_contract", doc["engine_contract"]
    for i, st in enumerate(doc.get("science_targets") or []):
        yield f"science_targets[{i}]", st
    for i, obj in enumerate(doc.get("objectives") or []):
        if obj.get("science_target"):
            yield f"objectives[{i}].science_target", obj["science_target"]
    for i, jdef in enumerate(doc.get("judges") or []):
        yield f"judges[{i}].judge", jdef.get("judge") or {}
        for r, rubric in enumerate(jdef.get("rubrics") or []):
            yield f"judges[{i}].rubrics[{r}]", rubric


def contains_prohibited_path(ref: dict[str, Any]) -> bool:
    lid = ref.get("logical_id", "")
    if not isinstance(lid, str):
        return False
    if lid.startswith("/") or lid.startswith(".") or "://" in lid:
        return True
    if "\\" in lid:
        return True
    return False
