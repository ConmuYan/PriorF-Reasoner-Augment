"""Project-wide exception hierarchy."""

from __future__ import annotations


class LGHGCLError(Exception):
    """Base exception for LG-HGCL."""


class DataError(LGHGCLError):
    """Raised when data loading or preprocessing fails."""


class ModelError(LGHGCLError):
    """Raised when model creation or forward pass fails."""


class TrainingError(LGHGCLError):
    """Raised when training fails."""
