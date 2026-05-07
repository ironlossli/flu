# src/spectra/models/backbones/__init__.py
"""
Lightweight init to avoid importing optional/backbone-specific deps at import time.
Prevents errors like 'No module named deepqmc.torchext' when only using EGNN.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Make local 'schnetpack' importable if it exists but isn't a package (__init__.py missing)
_pkg_dir = Path(__file__).resolve().parent
_sp_dir = _pkg_dir / "schnetpack"
if _sp_dir.is_dir():
    _sp = str(_sp_dir)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

__all__ = [
    "AE_EGNN",
    "E_GCL",
    "PaiNN",
    "NeuralNetworkPotential",
    "MessagePassPaiNN",
    "UpdatePaiNN",
    "BesselBasis",
    "CosineCutoff",
]

def __getattr__(name: str):
    if name == "AE_EGNN":
        return importlib.import_module(".egnn_autoencoder", __name__).AE_EGNN
    if name == "E_GCL":
        return importlib.import_module(".egnn_layers", __name__).E_GCL
    if name == "PaiNN":
        return importlib.import_module(".painn", __name__).PaiNN
    if name == "NeuralNetworkPotential":
        return importlib.import_module(".schnet", __name__).NeuralNetworkPotential
    if name == "MessagePassPaiNN":
        return importlib.import_module(".message", __name__).MessagePassPaiNN
    if name == "UpdatePaiNN":
        return importlib.import_module(".update", __name__).UpdatePaiNN
    if name == "BesselBasis":
        return importlib.import_module(".helper", __name__).BesselBasis
    if name == "CosineCutoff":
        return importlib.import_module(".helper", __name__).CosineCutoff
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return sorted(list(globals().keys()) + __all__)

# For static type checkers / IDEs
if TYPE_CHECKING:
    from .egnn_autoencoder import AE_EGNN
    from .egnn_layers import E_GCL
    from .painn import PaiNN
    from .schnet import NeuralNetworkPotential
    from .message import MessagePassPaiNN
    from .update import UpdatePaiNN
    from .helper import BesselBasis, CosineCutoff
