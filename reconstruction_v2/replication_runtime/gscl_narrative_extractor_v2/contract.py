"""Non-content error contract for hierarchical closed-choice extraction."""

from __future__ import annotations

from enum import Enum
from types import MappingProxyType
from typing import Mapping


VERSION = "gscl_narrative_extractor_v2_contract_v1"
HIERARCHICAL_WIRE_SCHEMA = "gscl.narrative.hierarchical_selection.v2"


class ErrorCategory(str, Enum):
    AUTHORITY = "authority"
    CATALOG = "catalog"
    CONTEXT = "context"
    CUDA = "cuda"
    MODEL = "model"
    PARSER = "parser"
    SELECTION = "selection"
    TOKEN_BOUNDARY = "token_boundary"
    VERIFIER = "verifier"


ERROR_TAXONOMY = MappingProxyType(
    {
        "V2_AUTHORITY_INVALID": ErrorCategory.AUTHORITY,
        "V2_CATALOG_DOCUMENT_TOKEN_COUNT_UNSUPPORTED": (
            ErrorCategory.CATALOG
        ),
        "V2_CATALOG_EPISODE_CAPACITY_UNSUPPORTED": (
            ErrorCategory.CATALOG
        ),
        "V2_CATALOG_SENTENCE_CAPACITY_UNSUPPORTED": (
            ErrorCategory.CATALOG
        ),
        "V2_CATALOG_SENTENCE_TOKEN_COUNT_UNSUPPORTED": (
            ErrorCategory.CATALOG
        ),
        "V2_CATALOG_SPAN_GROUNDING_INVALID": (
            ErrorCategory.CATALOG
        ),
        "V2_CONTEXT_TOKEN_LIMIT_EXCEEDED": ErrorCategory.CONTEXT,
        "V2_CUDA_RUNTIME_UNAVAILABLE": ErrorCategory.CUDA,
        "V2_MODEL_FORWARD_FAILED": ErrorCategory.MODEL,
        "V2_MODEL_SCORE_BATCH_INVALID": ErrorCategory.MODEL,
        "V2_PARSER_REJECTED": ErrorCategory.PARSER,
        "V2_PLAN_CANDIDATE_SET_INVALID": ErrorCategory.VERIFIER,
        "V2_PLAN_NO_RELATION_SELECTED": ErrorCategory.SELECTION,
        "V2_PLAN_RELATION_CAPACITY_EXCEEDED": (
            ErrorCategory.CATALOG
        ),
        "V2_TOKEN_BOUNDARY_INVALID": ErrorCategory.TOKEN_BOUNDARY,
        "V2_VERIFIER_REJECTED": ErrorCategory.VERIFIER,
        "V2_WIRE_COVERAGE_INVALID": ErrorCategory.VERIFIER,
        "V2_WIRE_CANONICAL_MISMATCH": ErrorCategory.VERIFIER,
        "V2_WIRE_ENDPOINT_OVERLAP": ErrorCategory.VERIFIER,
        "V2_WIRE_ENDPOINT_REF_INVALID": ErrorCategory.VERIFIER,
        "V2_WIRE_ENDPOINT_SELECTION_MISSING": (
            ErrorCategory.VERIFIER
        ),
        "V2_WIRE_FIELDS_INVALID": ErrorCategory.VERIFIER,
        "V2_WIRE_ORDER_INVALID": ErrorCategory.VERIFIER,
        "V2_WIRE_REFERENCE_INVALID": ErrorCategory.VERIFIER,
        "V2_WIRE_OBJECT_OWNERSHIP_INVALID": (
            ErrorCategory.VERIFIER
        ),
        "V2_WIRE_SENTENCE_COVERAGE_INVALID": (
            ErrorCategory.VERIFIER
        ),
        "V2_WIRE_SPAN_OVERLAP_INVALID": ErrorCategory.VERIFIER,
    }
)


class ClosedChoiceV2Error(RuntimeError):
    """Stable error whose identifier never embeds narrative content."""

    def __init__(self, issue_id: str) -> None:
        if issue_id not in ERROR_TAXONOMY:
            raise ValueError("v2_error_issue_id_unknown")
        self.issue_id = issue_id
        self.category = ERROR_TAXONOMY[issue_id]
        super().__init__(issue_id)


class ClosedChoiceV2Abstention(ClosedChoiceV2Error):
    """Typed representability/context abstention, not a runtime failure."""

    def __init__(
        self, issue_id: str, *, before_model_forward: bool
    ) -> None:
        if ERROR_TAXONOMY.get(issue_id) not in {
            ErrorCategory.CATALOG,
            ErrorCategory.CONTEXT,
            ErrorCategory.SELECTION,
        }:
            raise ValueError("v2_abstention_category_invalid")
        self.before_model_forward = before_model_forward
        super().__init__(issue_id)


def classify_external_failure(issue_id: str) -> ClosedChoiceV2Error:
    """Construct a typed failure for exact runtimes/supervisors."""

    return ClosedChoiceV2Error(issue_id)


def non_content_failure_record(
    error: ClosedChoiceV2Error,
) -> Mapping[str, object]:
    """Preserve the specific safe taxonomy without narrative content."""

    if not isinstance(error, ClosedChoiceV2Error):
        raise ClosedChoiceV2Error("V2_AUTHORITY_INVALID")
    return MappingProxyType(
        {
            "error_category": error.category.value,
            "error_code": error.issue_id,
            "generation_valid": False,
            "pre_model_abstention": (
                error.before_model_forward
                if isinstance(error, ClosedChoiceV2Abstention)
                else False
            ),
        }
    )


__all__ = [
    "ERROR_TAXONOMY",
    "HIERARCHICAL_WIRE_SCHEMA",
    "ClosedChoiceV2Abstention",
    "ClosedChoiceV2Error",
    "ErrorCategory",
    "VERSION",
    "classify_external_failure",
    "non_content_failure_record",
]
