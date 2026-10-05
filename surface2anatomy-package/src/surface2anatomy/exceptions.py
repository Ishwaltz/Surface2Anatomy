"""
Surface2Anatomy Custom Exception Hierarchy.
All exceptions derive from Surface2AnatomyError and provide clear, actionable diagnostics.
"""

from typing import List, Optional

class Surface2AnatomyError(Exception):
    """Base exception for all Surface2Anatomy errors."""
    pass

class UnsupportedFormatError(Surface2AnatomyError):
    """Raised when an unsupported file format or 2D image is supplied."""
    pass

class InvalidSurfaceError(Surface2AnatomyError):
    """Raised when the surface geometry is physically or mathematically degenerate."""
    pass

class TooFewPointsError(InvalidSurfaceError):
    """Raised when the 3D surface contains fewer points than the required minimum threshold."""
    def __init__(self, count: int, minimum: int = 64):
        super().__init__(
            f"Surface geometry contains only {count} points. Minimum required is {minimum} points."
        )
        self.count = count
        self.minimum = minimum

class AmbiguousUnitsError(Surface2AnatomyError):
    """Raised when the physical coordinate scale cannot be determined unambiguously."""
    def __init__(self, message: str, max_dimension: float):
        super().__init__(
            f"{message} (Detected bounding box max span: {max_dimension:.2f}). "
            "Please specify units explicitly via units='mm', units='cm', or units='m'."
        )
        self.max_dimension = max_dimension

class UnknownTargetError(Surface2AnatomyError):
    """Raised when an anatomical target query cannot be resolved to any supported landmark."""
    def __init__(self, target: str, suggestions: Optional[List[str]] = None):
        msg = f"Unknown anatomical target '{target}'."
        if suggestions:
            msg += "\n\nDid you mean:"
            for s in suggestions:
                msg += f"\n  - {s}"
        msg += "\n\nRun `surface2anatomy list-targets` or call `surface2anatomy.list_targets()` to view all supported targets."
        super().__init__(msg)
        self.target = target
        self.suggestions = suggestions or []

class ModelDownloadError(Surface2AnatomyError):
    """Raised when downloading pretrained model weights fails."""
    pass

class ChecksumError(Surface2AnatomyError):
    """Raised when downloaded weights do not match the frozen cryptographic SHA256 checksum."""
    def __init__(self, filename: str, expected: str, actual: str):
        super().__init__(
            f"Cryptographic SHA256 verification failed for '{filename}'.\n"
            f"  Expected: {expected}\n"
            f"  Actual:   {actual}\n"
            "The corrupted file was removed. Please retry the download or check network integrity."
        )
        self.filename = filename
        self.expected = expected
        self.actual = actual

class ModelNotFoundError(Surface2AnatomyError):
    """Raised when required model checkpoint files are not present in local cache."""
    pass

class ModelLoadError(Surface2AnatomyError):
    """Raised when a PyTorch checkpoint fails to load or deserialize."""
    pass

class AlignmentError(Surface2AnatomyError):
    """Raised when external canonical alignment fails or produces non-finite coordinates."""
    pass

class PreprocessingError(Surface2AnatomyError):
    """Raised when surface preprocessing encounters an unrecoverable validation error."""
    pass
