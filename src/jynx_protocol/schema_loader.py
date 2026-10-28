from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def schemas_dir() -> Path:
    return repo_root() / "schemas"


@lru_cache(maxsize=1)
def schema_store() -> dict[str, Any]:
    store: dict[str, Any] = {}
    root = schemas_dir()
    for path in sorted(root.rglob("*.schema.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        schema_id = data.get("$id")
        if not schema_id:
            raise SchemaError(f"Schema missing $id: {path}")
        store[schema_id] = data
    return store


def validator_for_name(schema_name: str) -> Draft202012Validator:
    path = schemas_dir() / schema_name
    data = json.loads(path.read_text(encoding="utf-8"))
    store = schema_store()
    cls = validator_for(data)
    return cls(data, registry=_registry_from_store(store))


def _registry_from_store(store: dict[str, Any]):
    from referencing import Registry, Resource

    resources = [Resource.from_contents(v) for v in store.values()]
    return Registry().with_resources((r.id(), r) for r in resources if r.id() is not None)


def validate_instance(schema_filename: str, instance: Any) -> list[str]:
    validator = validator_for_name(schema_filename)
    errors = sorted(validator.iter_errors(instance), key=lambda e: list(e.path))
    return [f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in errors]
