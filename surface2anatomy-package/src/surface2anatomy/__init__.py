"""
Surface2Anatomy: Target-Conditioned 3D Internal Anatomy Localization from External Body Surface Geometry.
"""

from surface2anatomy._version import __version__
from surface2anatomy.constants import (
    NUM_POINTS,
    S_GLOBAL_MM,
    NUM_ORGANS_ARCH,
    NUM_PRIMARY_BENCHMARK_TARGETS,
    TOTAL_CATALOG_TARGETS,
    DEFAULT_SAMPLING_SEED,
    FROZEN_SEEDS,
    COORDINATE_FRAME_DESC,
    RUNTIME_IMAGING_DESC
)
from surface2anatomy.exceptions import (
    Surface2AnatomyError,
    UnsupportedFormatError,
    InvalidSurfaceError,
    TooFewPointsError,
    AmbiguousUnitsError,
    UnknownTargetError,
    ModelDownloadError,
    ChecksumError,
    ModelNotFoundError,
    ModelLoadError,
    AlignmentError,
    PreprocessingError
)
from surface2anatomy.targets import (
    CANONICAL_TARGET_NAMES,
    PRIMARY_BENCHMARK_TARGETS,
    TARGET_CATEGORIES,
    list_targets
)
from surface2anatomy.synonyms import resolve_target
from surface2anatomy.results import (
    SinglePrediction,
    PredictionResult,
    PreprocessingMetadata
)
from surface2anatomy.uncertainty import compute_ensemble_disagreement_mm
from surface2anatomy.io import load_surface
from surface2anatomy.sampling import sample_4096_points
from surface2anatomy.model import SurfaceAnatomyModel, RemoteSurfaceAnatomyModel

__all__ = [
    "__version__",
    "SurfaceAnatomyModel",
    "RemoteSurfaceAnatomyModel",
    "list_targets",
    "resolve_target",
    "load_surface",
    "sample_4096_points",
    "compute_ensemble_disagreement_mm",
    "SinglePrediction",
    "PredictionResult",
    "PreprocessingMetadata",
    "CANONICAL_TARGET_NAMES",
    "PRIMARY_BENCHMARK_TARGETS",
    "TARGET_CATEGORIES",
    "NUM_POINTS",
    "S_GLOBAL_MM",
    "Surface2AnatomyError",
    "UnsupportedFormatError",
    "InvalidSurfaceError",
    "TooFewPointsError",
    "AmbiguousUnitsError",
    "UnknownTargetError",
    "ModelDownloadError",
    "ChecksumError",
    "ModelNotFoundError",
    "ModelLoadError",
    "AlignmentError",
    "PreprocessingError",
]
