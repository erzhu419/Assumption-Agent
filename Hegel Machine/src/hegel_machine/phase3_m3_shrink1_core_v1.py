"""Target-free sparse-registry constants for M3 shrink step 1."""

from __future__ import annotations

from typing import Final


DSL_VERSION: Final = "hegel-old-dsl-v1.1.0"
REGISTRY_WIDTH: Final = 6
ACTIVE_AGGREGATE_IDS: Final = (0, 1, 5)
TOMBSTONED_AGGREGATE_IDS: Final = (2, 3, 4)
TOMBSTONED_AGGREGATE_NAMES: Final = ("mean_v1", "min_v1", "max_v1")
REMOVED_AGGREGATE_ERROR: Final = "REJECT_REMOVED_AGGREGATE_MAP"


__all__ = [
    "ACTIVE_AGGREGATE_IDS",
    "DSL_VERSION",
    "REGISTRY_WIDTH",
    "REMOVED_AGGREGATE_ERROR",
    "TOMBSTONED_AGGREGATE_IDS",
    "TOMBSTONED_AGGREGATE_NAMES",
]
