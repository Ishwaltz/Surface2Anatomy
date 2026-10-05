"""
Surface2Anatomy Canonical Constants and Coordinate System Definitions.

Coordinate Convention:
- +X: Patient Anatomical Right
- -X: Patient Anatomical Left
- +Y: Patient Anterior
- -Y: Patient Posterior
- +Z: Patient Superior (Cranial)
- -Z: Patient Inferior (Caudal)

Origin (0, 0, 0): Lower thoracic / T12 vertebral level along midsagittal and midcoronal symmetry axes.
Global Normalization Scale: S_global = 500.0 mm.
Input Surface Geometry: 4,096 points x 3 coordinates.
Surface Memory Tokens: 320 tokens (256 local SA2 + 64 coarse SA3).
"""

from typing import Tuple

# Physical & Model Parameters
NUM_POINTS: int = 4096
S_GLOBAL_MM: float = 500.0
NUM_ORGANS_ARCH: int = 117
NUM_PRIMARY_BENCHMARK_TARGETS: int = 104
TOTAL_CATALOG_TARGETS: int = 121

# Default Sampling & Ensemble Seeds
DEFAULT_SAMPLING_SEED: int = 42
FROZEN_SEEDS: Tuple[int, ...] = (42, 43, 44)

# Coordinate Frame Description
COORDINATE_FRAME_DESC: str = "+X = Patient Right, +Y = Patient Anterior, +Z = Patient Superior"
RUNTIME_IMAGING_DESC: str = "External surface only (zero CT/MRI voxel requirements at runtime)"
