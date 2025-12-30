"""
RNK: Recursive Neuro-Knowledge Architecture

A sample-efficient, compute-efficient alternative to Transformers
that achieves coherent text generation through latent planning.
"""

from .model import RNK
from .encoder import ChunkEncoder
from .mamba import MambaSSM as StateSpaceModule  # Mamba replaces FastState+SlowMemory
from .hrm import HRM, HRMLayer
from .neuro_symbolic import NeuroSymbolicRefiner
from .decoder import Decoder

__version__ = "0.1.0"
__all__ = [
    "RNK",
    "ChunkEncoder",
    "StateSpaceModule",  # Now MambaSSM
    "HRM",
    "HRMLayer",
    "NeuroSymbolicRefiner",
    "Decoder",
]
