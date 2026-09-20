"""Configuration objects for TFNE-Izhikevich point-process networks."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class TFNEIzhikevichConfig:
    n_neurons: int = 96
    dt: float = 0.1
    # Add other fields as needed for the smoke test
