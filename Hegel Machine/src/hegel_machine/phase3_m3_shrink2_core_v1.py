"""Target-free sparse-registry constants for M3 shrink step 2.

The child keeps every numeric identity allocated by the parent DSL.  Shrink
step 2 changes admission only: RationalParameterId 0, 2, 4, and 6 become
permanent tombstones while 1, 3, and 5 remain active.  The three-bit code
space is neither compressed nor renumbered and code point 7 stays reserved.
"""

from __future__ import annotations

from typing import Final

from .phase3_m3_dsl_core_v1 import AGGREGATE_MAP_IDS


PARENT_DSL_VERSION: Final = "hegel-old-dsl-v1.1.0"
PARENT_FREEZE_VERSION: Final = "hegel-freeze-p2b-p3-v1.1.2"
DSL_VERSION: Final = "hegel-old-dsl-v1.2.0"
FREEZE_VERSION: Final = "hegel-freeze-p2b-p3-v1.2.0"
HUMAN_AMENDMENT_ID: Final = "hegel-freeze-p2b-p3-v1.2.0-shrink-step2"
SHRINK_STEP_ID: Final = (
    "SHRINK_STEP_2_REDUCE_RATIONAL_PARAMETER_TO_NEG1_ZERO_POS1"
)

RATIONAL_PARAMETER_REGISTRY_NAMESPACE: Final = "RationalParameterId/v1"
RATIONAL_PARAMETER_CODE_WIDTH_BITS: Final = 3
RATIONAL_PARAMETER_ALLOCATED_ID_COUNT: Final = 7
RATIONAL_PARAMETER_CODE_SPACE_SIZE: Final = 8
ACTIVE_RATIONAL_PARAMETER_IDS: Final = (1, 3, 5)
TOMBSTONED_RATIONAL_PARAMETER_IDS: Final = (0, 2, 4, 6)
RESERVED_RATIONAL_PARAMETER_IDS: Final = (7,)
REMOVED_RATIONAL_PARAMETER_ERROR: Final = "REJECT_REMOVED_RATIONAL_PARAMETER"
UNKNOWN_RATIONAL_PARAMETER_ERROR: Final = "REJECT_REGISTRY_INDEX_OUT_OF_RANGE"

# Shrink step 1 is inherited without changing or reallocating its registry.
ACTIVE_AGGREGATE_IDS: Final = (0, 1, 5)
TOMBSTONED_AGGREGATE_IDS: Final = (2, 3, 4)
RATIONAL_ACTIVE_AGGREGATE_IDS: Final = (0, 5)
RATIONAL_ACTIVE_AGGREGATE_NAMES: Final = tuple(
    AGGREGATE_MAP_IDS[numeric_id]
    for numeric_id in RATIONAL_ACTIVE_AGGREGATE_IDS
)
REMOVED_AGGREGATE_ERROR: Final = "REJECT_REMOVED_AGGREGATE_MAP"


__all__ = [
    "ACTIVE_AGGREGATE_IDS",
    "ACTIVE_RATIONAL_PARAMETER_IDS",
    "DSL_VERSION",
    "FREEZE_VERSION",
    "HUMAN_AMENDMENT_ID",
    "PARENT_DSL_VERSION",
    "PARENT_FREEZE_VERSION",
    "RATIONAL_PARAMETER_ALLOCATED_ID_COUNT",
    "RATIONAL_PARAMETER_CODE_SPACE_SIZE",
    "RATIONAL_PARAMETER_CODE_WIDTH_BITS",
    "RATIONAL_PARAMETER_REGISTRY_NAMESPACE",
    "RATIONAL_ACTIVE_AGGREGATE_IDS",
    "RATIONAL_ACTIVE_AGGREGATE_NAMES",
    "REMOVED_AGGREGATE_ERROR",
    "REMOVED_RATIONAL_PARAMETER_ERROR",
    "RESERVED_RATIONAL_PARAMETER_IDS",
    "SHRINK_STEP_ID",
    "TOMBSTONED_AGGREGATE_IDS",
    "TOMBSTONED_RATIONAL_PARAMETER_IDS",
    "UNKNOWN_RATIONAL_PARAMETER_ERROR",
]
