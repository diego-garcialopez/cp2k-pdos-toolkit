"""
pdos_toolkit
============

A small, correct, and reasonably memory-efficient toolkit for post-processing
Projected Density of States (PDOS) output from CP2K.

Public API:
    PDOSProcessor  -- main class: load files, compute broadened PDOS, save, plot.
"""

from .pdos_processor import PDOSProcessor, HARTREE_TO_EV

__all__ = ["PDOSProcessor", "HARTREE_TO_EV"]
__version__ = "1.0.0"
