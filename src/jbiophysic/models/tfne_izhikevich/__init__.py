"""TFNE-Izhikevich point-process network prototype.

Public API:
    TFNEIzhikevichConfig
    TFNEIzhikevichNetwork
    build_tfne_izhikevich_network
"""
from .config import TFNEIzhikevichConfig
from .grid import TFNEGrid, build_grid
from .network import (
    TFNEIzhikevichNetwork,
    TFNEIzhikevichResult,
    TFNEIzhikevichStructure,
    build_tfne_izhikevich_network,
)

__all__ = [
    "TFNEIzhikevichConfig",
    "TFNEGrid",
    "build_grid",
    "TFNEIzhikevichNetwork",
    "TFNEIzhikevichResult",
    "TFNEIzhikevichStructure",
    "build_tfne_izhikevich_network",
]
