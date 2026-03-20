from __future__ import annotations


class VAOError(Exception):
    """Base error for VAO."""


class OpenSmileNotFoundError(VAOError, FileNotFoundError):
    """Raised when SMILExtract cannot be located."""


class OpenSmileRunError(VAOError, RuntimeError):
    """Raised when SMILExtract fails."""
