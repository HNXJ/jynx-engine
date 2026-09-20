# src/jbiophysic/models/__init__.py
"""
Models tier: orchestration of biophysical hierarchies, simulation pipelines, and
training loops.

Imports that require optional heavy simulators such as Jaxley are deliberately
lazy/guarded so pure-JAX reduced models and TFNE-Izhikevich utilities can be
used in minimal environments.
"""
from jbiophysic.common.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "build_cortical_hierarchy",
    "build_11_area_hierarchy",
    "construct_column",
    "run_simulation",
]


def _missing_optional(name: str, err: Exception):
    def _raiser(*args, **kwargs):
        raise ImportError(
            f"{name} requires an optional dependency or module that is not available: {err}"
        ) from err

    return _raiser


try:
    from .builders.hierarchy import build_cortical_hierarchy, build_11_area_hierarchy
except Exception as exc:  # pragma: no cover - exercised only when optional deps missing
    build_cortical_hierarchy = _missing_optional("build_cortical_hierarchy", exc)
    build_11_area_hierarchy = _missing_optional("build_11_area_hierarchy", exc)

try:
    from .builders.populations import construct_column
except Exception as exc:  # pragma: no cover
    construct_column = _missing_optional("construct_column", exc)

try:
    from .simulation.run import run_simulation
except Exception as exc:  # pragma: no cover
    run_simulation = _missing_optional("run_simulation", exc)
