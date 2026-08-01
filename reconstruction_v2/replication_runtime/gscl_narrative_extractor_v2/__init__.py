"""Hierarchical closed-choice GSCL narrative extraction runtime v2.

Version 2 is a new implementation lineage.  It does not alter or reinterpret
the frozen v1 runtime or its evidence.
"""

from .contract import (
    ERROR_TAXONOMY,
    HIERARCHICAL_WIRE_SCHEMA,
    ClosedChoiceV2Abstention,
    ClosedChoiceV2Error,
    ErrorCategory,
)
from .closed_choice import (
    ClosedChoiceV2Decision,
    select_hierarchical_qualification_only,
)

__all__ = [
    "ERROR_TAXONOMY",
    "HIERARCHICAL_WIRE_SCHEMA",
    "ClosedChoiceV2Abstention",
    "ClosedChoiceV2Decision",
    "ClosedChoiceV2Error",
    "ErrorCategory",
    "select_hierarchical_qualification_only",
]
