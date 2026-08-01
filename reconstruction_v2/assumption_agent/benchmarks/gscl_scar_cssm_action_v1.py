"""Label-blind SCAR CSSM action formation over one opaque item.

The action former owns no source file, label pack, scorer, filesystem, or
network capability.  It receives one already-validated opaque action item,
forms the semantic control before attempting structural extraction, extracts
each base-side background exactly once, and reuses those two local graphs for
the frozen base/system-swap variants.

All returned content is private measurement material.  Nothing in this
module is a safe aggregate or an effect claim.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import re
from typing import Any, Callable, Mapping
import unicodedata

from assumption_agent.generalized_structural_correspondence_v1 import (
    strict_content_hash,
)
from assumption_agent.gscl_slot_graph_binder_v1 import (
    BoundSlotGraphV1,
    SemanticMatrixResultV1,
    TextEncoder,
    bind_relation_set_to_slots_v1,
    semantic_slot_score_matrix_v1,
)
from assumption_agent.gscl_slot_set_mapping_v1 import (
    ChoiceDisposition,
    MappingArm,
    MappingProposalV1,
    SemanticSlotScoreMatrixV1,
    SlotGraphV1,
    build_slot_graph_v1,
    map_slot_graphs_v1,
)
from replication_runtime.gscl_narrative_extractor_v2 import (
    bounded_set_consumer,
    document_envelope,
)


VERSION = "gscl_scar_cssm_action_v1"
ARM_IDS = (
    "semantic_only",
    "flat_structural",
    "full_no_composition",
    "full_with_length2_composition",
    "full_with_length2_composition_target_color_shuffle",
)
VARIANT_NAMES = ("base", "system_swap")
STRUCTURAL_ERROR_CODES = frozenset(
    {
        "BOUNDED_CONSUMER_TYPED_FAILURE",
        "DOCUMENT_EXTRACTOR_TYPED_FAILURE",
        "INTERNAL_TYPED_FAILURE",
        "NO_FEASIBLE_INJECTIVE_PAIR_SET",
        "PROPOSAL_CONSTRUCTION_TYPED_FAILURE",
        "SLOT_BINDER_TYPED_FAILURE",
    }
)

_ITEM_TOKEN = re.compile(r"scar-item-v1-[0-9a-f]{64}\Z")
_SLOT_TOKEN = re.compile(r"scar-slot-v1-[0-9a-f]{64}\Z")
_IDENTITY_OPERATOR_ID = (
    "ori_keep.pol_keep.slots_identity"
)


class ScarCssmActionError(RuntimeError):
    def __init__(self, issue_id: str) -> None:
        self.issue_id = issue_id
        super().__init__(issue_id)


class ScarCssmActionInfrastructureError(ScarCssmActionError):
    """A runtime/programming failure that must invalidate the formal run."""


_INFRASTRUCTURE_LEAF_ERRORS = frozenset(
    {
        "DOCUMENT_LEAF_RUNTIME_FAILED",
        "V2_CUDA_RUNTIME_UNAVAILABLE",
        "V2_MODEL_FORWARD_FAILED",
        "V2_MODEL_SCORE_BATCH_INVALID",
    }
)


DocumentSelector = Callable[
    [str], document_envelope.NarrativeDocumentEnvelopeV1
]


def _side_labels(side: Mapping[str, Any]) -> dict[str, str]:
    if (
        type(side) is not dict
        or set(side) != {"background", "slots", "system"}
        or not isinstance(side["background"], str)
        or not side["background"].strip()
        or not isinstance(side["system"], str)
        or not side["system"].strip()
        or type(side["slots"]) is not list
        or not 2 <= len(side["slots"]) <= 14
    ):
        raise ScarCssmActionError("SCAR_ACTION_SIDE_INVALID")
    rows: dict[str, str] = {}
    for slot in side["slots"]:
        if (
            type(slot) is not dict
            or set(slot) != {"opaque_slot_id", "surface"}
            or not isinstance(slot["opaque_slot_id"], str)
            or _SLOT_TOKEN.fullmatch(slot["opaque_slot_id"]) is None
            or not isinstance(slot["surface"], str)
            or not slot["surface"].strip()
            or slot["opaque_slot_id"] in rows
        ):
            raise ScarCssmActionError("SCAR_ACTION_SLOT_INVALID")
        rows[slot["opaque_slot_id"]] = slot["surface"].strip()
    return rows


def _validate_item(item: Mapping[str, Any]) -> tuple[
    str, Mapping[str, Any], Mapping[str, Any], dict[str, str], dict[str, str]
]:
    if (
        type(item) is not dict
        or set(item) != {"item_token", "variants"}
        or not isinstance(item["item_token"], str)
        or _ITEM_TOKEN.fullmatch(item["item_token"]) is None
        or type(item["variants"]) is not dict
        or tuple(item["variants"]) != VARIANT_NAMES
    ):
        raise ScarCssmActionError("SCAR_ACTION_ITEM_INVALID")
    base = item["variants"]["base"]
    swapped = item["variants"]["system_swap"]
    if (
        type(base) is not dict
        or set(base) != {"left", "right"}
        or type(swapped) is not dict
        or set(swapped) != {"left", "right"}
        or swapped["left"] != base["right"]
        or swapped["right"] != base["left"]
    ):
        raise ScarCssmActionError("SCAR_ACTION_VARIANT_INVALID")
    left_labels = _side_labels(base["left"])
    right_labels = _side_labels(base["right"])
    if len(left_labels) != len(right_labels) or set(left_labels) & set(right_labels):
        raise ScarCssmActionError("SCAR_ACTION_SIDE_SLOT_SET_INVALID")
    return item["item_token"], base["left"], base["right"], left_labels, right_labels


def _has_normalized_collision(labels: Mapping[str, str]) -> bool:
    normalized = [
        unicodedata.normalize("NFKC", value).casefold()
        for value in labels.values()
    ]
    return len(set(normalized)) != len(normalized)


def _reject_infrastructure_leaf_failure(
    envelope: document_envelope.NarrativeDocumentEnvelopeV1,
) -> None:
    if type(envelope) is not document_envelope.NarrativeDocumentEnvelopeV1:
        raise ScarCssmActionInfrastructureError(
            "SCAR_ACTION_DOCUMENT_RESULT_INVALID"
        )
    if any(
        row.error_code in _INFRASTRUCTURE_LEAF_ERRORS
        for row in envelope.segments
    ):
        raise ScarCssmActionInfrastructureError(
            "SCAR_ACTION_RUNTIME_INFRASTRUCTURE_FAILURE"
        )


def _empty_semantic_graph(
    labels: Mapping[str, str], *, side: str, item_token: str
) -> SlotGraphV1:
    evidence = {
        slot_id: strict_content_hash(
            {
                "control": "semantic_surface_only_empty_graph_carrier",
                "normalized_surface_sha256": hashlib.sha256(
                    unicodedata.normalize("NFKC", label)
                    .casefold()
                    .encode("utf-8", errors="strict")
                ).hexdigest(),
                "side": side,
                "slot_id": slot_id,
            }
        )
        for slot_id, label in labels.items()
    }
    return build_slot_graph_v1(
        slot_labels=labels,
        slot_evidence_bindings=evidence,
        relations=(),
        extractor_binding_sha256=strict_content_hash(
            {
                "control": "semantic_surface_only_empty_graph_carrier",
                "item_token": item_token,
                "side": side,
                "version": VERSION,
            }
        ),
        coverage_complete=False,
    )


def _transpose_matrix(
    matrix: SemanticSlotScoreMatrixV1,
) -> SemanticSlotScoreMatrixV1:
    return SemanticSlotScoreMatrixV1.from_mapping(
        {(right, left): score for left, right, score in matrix.rows}
    )


def _proposal_pairs(
    proposal: MappingProposalV1,
    source_graph: SlotGraphV1,
    target_graph: SlotGraphV1,
) -> list[list[str]]:
    if len(proposal.target_indices) != len(source_graph.slots):
        raise ScarCssmActionError("SCAR_PROPOSAL_INVALID")
    return [
        [left.slot_id, target_graph.slots[target_index].slot_id]
        for left, target_index in zip(
            source_graph.slots, proposal.target_indices, strict=True
        )
    ]


def _choice_payload(
    *,
    result: object,
    arm: MappingArm,
    source_graph: SlotGraphV1,
    target_graph: SlotGraphV1,
) -> dict[str, object]:
    choice = result.choice(arm)
    if choice.disposition is ChoiceDisposition.ABSTAIN:
        return {"disposition": "ABSTAIN", "error_code": None, "pairs": None}
    proposal = next(
        row for row in result.proposals if row.proposal_hash == choice.proposal_hash
    )
    return {
        "disposition": "ANSWER",
        "error_code": None,
        "pairs": _proposal_pairs(proposal, source_graph, target_graph),
    }


def _error_payload(code: str) -> dict[str, object]:
    if code not in STRUCTURAL_ERROR_CODES:
        raise ScarCssmActionError("SCAR_ERROR_CODE_INVALID")
    return {"disposition": "ERROR", "error_code": code, "pairs": None}


def _pools(
    result: object,
    source_graph: SlotGraphV1,
    target_graph: SlotGraphV1,
) -> dict[str, list[list[list[str]]]]:
    semantic: set[tuple[tuple[str, str], ...]] = set()
    structural: set[tuple[tuple[str, str], ...]] = set()
    for proposal in result.proposals:
        pairs = tuple(
            tuple(pair)
            for pair in _proposal_pairs(proposal, source_graph, target_graph)
        )
        if (
            proposal.operator_id == _IDENTITY_OPERATOR_ID
            and "semantic_kbest" in proposal.origins
        ):
            semantic.add(pairs)
        if "structure_kbest" in proposal.origins:
            structural.add(pairs)
    return {
        "semantic_kbest": [
            [list(pair) for pair in proposal] for proposal in sorted(semantic)
        ],
        "structure_kbest": [
            [list(pair) for pair in proposal] for proposal in sorted(structural)
        ],
    }


def _empty_arm_diagnostic() -> dict[str, object]:
    return {
        "flat_structural_score": None,
        "incidence_match_count": 0,
        "incidence_total_count": 0,
        "length2_composition_verified": False,
        "length2_path_count": 0,
        "length2_path_total_count": 0,
        "proposal_hash": None,
        "selected_operator": None,
        "semantic_score": None,
        "semantic_origin_count": 0,
        "structural_origin_count": 0,
        "typed_incidence_verified": False,
    }


def _arm_diagnostic(result: object, arm: MappingArm) -> dict[str, object]:
    choice = result.choice(arm)
    if choice.disposition is ChoiceDisposition.ABSTAIN:
        return _empty_arm_diagnostic()
    proposal = next(
        row for row in result.proposals if row.proposal_hash == choice.proposal_hash
    )
    return {
        "flat_structural_score": proposal.flat_structural_score,
        "incidence_match_count": proposal.typed_incidence_matched,
        "incidence_total_count": proposal.typed_incidence_total,
        "length2_composition_verified": (
            proposal.length2_composition_verified
        ),
        "length2_path_count": proposal.length2_path_matched,
        "length2_path_total_count": proposal.length2_path_total,
        "proposal_hash": proposal.proposal_hash,
        "selected_operator": proposal.operator_id,
        "semantic_score": proposal.semantic_score,
        "semantic_origin_count": int("semantic_kbest" in proposal.origins),
        "structural_origin_count": int("structure_kbest" in proposal.origins),
        "typed_incidence_verified": proposal.typed_incidence_verified,
    }


def _binder_diagnostic(bound: BoundSlotGraphV1) -> dict[str, object]:
    incident = {
        slot_id
        for relation in bound.graph.relations
        for slot_id in (relation.slot0_id, relation.slot1_id)
    }
    return {
        "coverage_disposition": bound.receipt["relation_set_disposition"],
        "dropped_edge_count": bound.receipt["dropped_relation_count"],
        "endpoint_count": bound.receipt["endpoint_count"],
        "retained_edge_count": bound.receipt["retained_relation_count"],
        "self_loop_count": sum(
            row.slot0_id == row.slot1_id for row in bound.graph.relations
        ),
        "unbound_count": bound.receipt["unbound_endpoint_count"],
        "zero_degree_count": sum(
            row.slot_id not in incident for row in bound.graph.slots
        ),
    }


def _receipt_archive_entry(raw: bytes) -> dict[str, object]:
    """Persist one canonical safe receipt and its exact byte commitment."""

    try:
        value = json.loads(raw.decode("ascii"))
        canonical = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (UnicodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ScarCssmActionInfrastructureError(
            "SCAR_ACTION_RECEIPT_ARCHIVE_INVALID"
        ) from exc
    trailing_lf = raw == canonical + b"\n"
    if type(value) is not dict or (raw != canonical and not trailing_lf):
        raise ScarCssmActionInfrastructureError(
            "SCAR_ACTION_RECEIPT_ARCHIVE_INVALID"
        )
    return {
        "receipt": value,
        "receipt_sha256": hashlib.sha256(raw).hexdigest(),
        "trailing_lf": trailing_lf,
    }


def _leaf_receipt_archive(
    envelope: document_envelope.NarrativeDocumentEnvelopeV1,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for outcome in envelope.segments:
        decision = outcome.leaf_decision
        if decision is None:
            continue
        entry = _receipt_archive_entry(decision.receipt_bytes)
        if entry["receipt_sha256"] != outcome.leaf_receipt_sha256:
            raise ScarCssmActionInfrastructureError(
                "SCAR_ACTION_RECEIPT_ARCHIVE_INVALID"
            )
        rows.append(
            {
                "canonical_completion": decision.canonical_completion,
                "canonical_completion_sha256": hashlib.sha256(
                    decision.canonical_completion.encode("utf-8")
                ).hexdigest(),
                "leaf_receipt": entry,
                "segment_id": outcome.plan.segment_id,
                "wire_completion": decision.wire_completion,
                "wire_completion_sha256": hashlib.sha256(
                    decision.wire_completion.encode("utf-8")
                ).hexdigest(),
            }
        )
    return rows


def _side_receipt_archive(
    *,
    envelope: document_envelope.NarrativeDocumentEnvelopeV1,
    relation_set: bounded_set_consumer.BoundedNarrativeRelationSetV1,
    bound: BoundSlotGraphV1,
) -> dict[str, object]:
    signature = relation_set.relation_set_signature_bytes
    return {
        "binder": {
            "endpoint_bindings": [
                row.private_payload() for row in bound.endpoint_bindings
            ],
            "receipt": _receipt_archive_entry(bound.receipt_bytes),
        },
        "bounded_set": {
            "coverage": [asdict(row) for row in relation_set.coverage],
            "receipt": _receipt_archive_entry(relation_set.receipt_bytes),
            "relation_set_signature_ascii": (
                None if signature is None else signature.decode("ascii")
            ),
            "relation_set_signature_sha256": (
                relation_set.relation_set_signature_sha256
            ),
            "units": [asdict(row) for row in relation_set.units],
        },
        "document_envelope": {
            "leaf_records": _leaf_receipt_archive(envelope),
            "receipt": _receipt_archive_entry(envelope.receipt_bytes),
        },
        "slot_graph": {
            "coverage_complete": bound.graph.coverage_complete,
            "extractor_binding_sha256": bound.graph.extractor_binding_sha256,
            "graph_evidence_binding_sha256": (
                bound.graph.graph_evidence_binding_sha256
            ),
            "receipt": _receipt_archive_entry(bound.graph.receipt_bytes),
            "relations": [
                row.private_payload() for row in bound.graph.relations
            ],
            "slots": [row.private_payload() for row in bound.graph.slots],
        },
    }


def _mapping_receipt_archive(
    *,
    semantic_result: object,
    structural_result: object,
    shuffled_result: object,
) -> dict[str, object]:
    def archive(result: object) -> dict[str, object]:
        return {
            "assignment_subproblems_solved": (
                result.assignment_subproblems_solved
            ),
            "choices": [row.safe_payload() for row in result.choices],
            "proposals": [
                {**row._body(), "proposal_hash": row.proposal_hash}
                for row in result.proposals
            ],
            "receipt": _receipt_archive_entry(result.receipt_bytes),
            "target_color_shuffle_effective": (
                result.target_color_shuffle_effective
            ),
        }

    return {
        "semantic_mapping": archive(semantic_result),
        "structural_mapping": archive(structural_result),
        "target_color_shuffle_mapping": archive(shuffled_result),
    }


def _premodel_failure_receipt_archive(error_code: str) -> dict[str, object]:
    return {
        "availability": "PREMODEL_TYPED_FAILURE",
        "error_code": error_code,
        "semantic_matrix": None,
        "sides": {"left": None, "right": None},
        "variants": {"base": None, "system_swap": None},
    }


def _unavailable_variant_diagnostic() -> dict[str, object]:
    return {
        "arms": {arm_id: _empty_arm_diagnostic() for arm_id in ARM_IDS},
        "left_binder": None,
        "left_graph_receipt_sha256": None,
        "mapping_receipt_sha256_by_arm": {arm_id: None for arm_id in ARM_IDS},
        "right_binder": None,
        "right_graph_receipt_sha256": None,
        "structural_diagnostics_available": False,
        "target_color_shuffle_effective": None,
    }


def _variant_diagnostic(
    *,
    semantic_result: object,
    structural_result: object,
    shuffled_result: object,
    left_bound: BoundSlotGraphV1,
    right_bound: BoundSlotGraphV1,
) -> dict[str, object]:
    semantic_hash = hashlib.sha256(semantic_result.receipt_bytes).hexdigest()
    structural_hash = hashlib.sha256(structural_result.receipt_bytes).hexdigest()
    shuffled_hash = hashlib.sha256(shuffled_result.receipt_bytes).hexdigest()
    return {
        "arms": {
            "semantic_only": _arm_diagnostic(
                semantic_result, MappingArm.SEMANTIC_ONLY
            ),
            "flat_structural": _arm_diagnostic(
                structural_result, MappingArm.FLAT_STRUCTURAL
            ),
            "full_no_composition": _arm_diagnostic(
                structural_result, MappingArm.FULL_NO_COMPOSITION
            ),
            "full_with_length2_composition": _arm_diagnostic(
                structural_result,
                MappingArm.FULL_WITH_LENGTH2_COMPOSITION,
            ),
            "full_with_length2_composition_target_color_shuffle": (
                _arm_diagnostic(
                    shuffled_result,
                    MappingArm.FULL_WITH_LENGTH2_COMPOSITION,
                )
            ),
        },
        "left_binder": _binder_diagnostic(left_bound),
        "left_graph_receipt_sha256": hashlib.sha256(
            left_bound.graph.receipt_bytes
        ).hexdigest(),
        "mapping_receipt_sha256_by_arm": {
            "semantic_only": semantic_hash,
            "flat_structural": structural_hash,
            "full_no_composition": structural_hash,
            "full_with_length2_composition": structural_hash,
            "full_with_length2_composition_target_color_shuffle": shuffled_hash,
        },
        "right_binder": _binder_diagnostic(right_bound),
        "right_graph_receipt_sha256": hashlib.sha256(
            right_bound.graph.receipt_bytes
        ).hexdigest(),
        "structural_diagnostics_available": True,
        "target_color_shuffle_effective": (
            shuffled_result.target_color_shuffle_effective
        ),
    }


def _variant_from_results(
    *,
    semantic_result: object,
    structural_result: object | None,
    shuffled_result: object | None,
    source_graph: SlotGraphV1,
    target_graph: SlotGraphV1,
    structural_error_code: str | None,
) -> tuple[dict[str, object], dict[str, object]]:
    arms: dict[str, object] = {
        "semantic_only": _choice_payload(
            result=semantic_result,
            arm=MappingArm.SEMANTIC_ONLY,
            source_graph=source_graph,
            target_graph=target_graph,
        )
    }
    if structural_result is None or shuffled_result is None:
        code = structural_error_code or "INTERNAL_TYPED_FAILURE"
        for arm_id in ARM_IDS[1:]:
            arms[arm_id] = _error_payload(code)
        pools: dict[str, object] = {
            "semantic_kbest": _pools(
                semantic_result, source_graph, target_graph
            )["semantic_kbest"],
            "structure_kbest": [],
        }
    else:
        for arm in (
            MappingArm.FLAT_STRUCTURAL,
            MappingArm.FULL_NO_COMPOSITION,
            MappingArm.FULL_WITH_LENGTH2_COMPOSITION,
        ):
            arms[arm.value] = _choice_payload(
                result=structural_result,
                arm=arm,
                source_graph=source_graph,
                target_graph=target_graph,
            )
        arms[
            "full_with_length2_composition_target_color_shuffle"
        ] = _choice_payload(
            result=shuffled_result,
            arm=MappingArm.FULL_WITH_LENGTH2_COMPOSITION,
            source_graph=source_graph,
            target_graph=target_graph,
        )
        pools = _pools(structural_result, source_graph, target_graph)
    return {"arms": arms}, pools


def form_scar_cssm_item_action_v1(
    item: Mapping[str, Any],
    *,
    document_selector: DocumentSelector,
    encoder: TextEncoder,
    encoder_binding_sha256: str,
) -> dict[str, object]:
    """Form one opaque item's two-variant private action exactly once."""

    item_token, left, right, left_labels, right_labels = _validate_item(item)
    if _has_normalized_collision(left_labels) or _has_normalized_collision(
        right_labels
    ):
        error = "SLOT_BINDER_TYPED_FAILURE"
        variants = {
            variant: {
                "arms": {arm: _error_payload(error) for arm in ARM_IDS}
            }
            for variant in VARIANT_NAMES
        }
        pools = {
            variant: {"semantic_kbest": [], "structure_kbest": []}
            for variant in VARIANT_NAMES
        }
        return {
            "diagnostics": {
                variant: _unavailable_variant_diagnostic()
                for variant in VARIANT_NAMES
            },
            "execution": {
                "document_call_count": 0,
                "error_code": error,
                "structural_status": "TYPED_FAILURE",
            },
            "item_token": item_token,
            "private_mechanism_receipts": (
                _premodel_failure_receipt_archive(error)
            ),
            "proposal_pools": pools,
            "variants": variants,
        }

    semantic_matrix: SemanticMatrixResultV1 = semantic_slot_score_matrix_v1(
        source_slot_labels=left_labels,
        target_slot_labels=right_labels,
        encoder=encoder,
        encoder_binding_sha256=encoder_binding_sha256,
    )
    semantic_left = _empty_semantic_graph(
        left_labels, side="left", item_token=item_token
    )
    semantic_right = _empty_semantic_graph(
        right_labels, side="right", item_token=item_token
    )
    semantic_base = map_slot_graphs_v1(
        semantic_left, semantic_right, semantic_matrix.matrix
    )
    semantic_swap = map_slot_graphs_v1(
        semantic_right,
        semantic_left,
        _transpose_matrix(semantic_matrix.matrix),
    )

    structural_base = structural_swap = None
    shuffled_base = shuffled_swap = None
    bound_left: BoundSlotGraphV1 | None = None
    bound_right: BoundSlotGraphV1 | None = None
    structural_error: str | None = None
    document_calls = 0
    if not callable(document_selector):
        raise ScarCssmActionInfrastructureError(
            "SCAR_ACTION_DOCUMENT_SELECTOR_INVALID"
        )
    document_calls += 1
    left_envelope = document_selector(left["background"])
    document_calls += 1
    right_envelope = document_selector(right["background"])
    _reject_infrastructure_leaf_failure(left_envelope)
    _reject_infrastructure_leaf_failure(right_envelope)
    left_set = bounded_set_consumer.consume_document_envelope(left_envelope)
    right_set = bounded_set_consumer.consume_document_envelope(right_envelope)
    if (
        left_set.disposition
        is bounded_set_consumer.SetConsumerDisposition.TYPED_FAILURE_BLOCKED
        or right_set.disposition
        is bounded_set_consumer.SetConsumerDisposition.TYPED_FAILURE_BLOCKED
    ):
        raise ScarCssmActionInfrastructureError(
            "SCAR_ACTION_TYPED_FAILURE_BLOCKED"
        )
    else:
        bound_left = bind_relation_set_to_slots_v1(
            left_set,
            slot_labels=left_labels,
            encoder=encoder,
            encoder_binding_sha256=encoder_binding_sha256,
        )
        bound_right = bind_relation_set_to_slots_v1(
            right_set,
            slot_labels=right_labels,
            encoder=encoder,
            encoder_binding_sha256=encoder_binding_sha256,
        )
        # This four-result block is an atomic action barrier: any unexpected
        # exception propagates and no item output is returned or persisted.
        structural_base = map_slot_graphs_v1(
            bound_left.graph, bound_right.graph, semantic_matrix.matrix
        )
        structural_swap = map_slot_graphs_v1(
            bound_right.graph,
            bound_left.graph,
            _transpose_matrix(semantic_matrix.matrix),
        )
        shuffled_base = map_slot_graphs_v1(
            bound_left.graph,
            bound_right.graph,
            semantic_matrix.matrix,
            target_color_shuffle=True,
        )
        shuffled_swap = map_slot_graphs_v1(
            bound_right.graph,
            bound_left.graph,
            _transpose_matrix(semantic_matrix.matrix),
            target_color_shuffle=True,
        )

    if bound_left is None or bound_right is None:
        base_source, base_target = semantic_left, semantic_right
        swap_source, swap_target = semantic_right, semantic_left
    else:
        base_source, base_target = bound_left.graph, bound_right.graph
        swap_source, swap_target = bound_right.graph, bound_left.graph
    base_prediction, base_pools = _variant_from_results(
        semantic_result=semantic_base,
        structural_result=structural_base,
        shuffled_result=shuffled_base,
        source_graph=base_source,
        target_graph=base_target,
        structural_error_code=structural_error,
    )
    swap_prediction, swap_pools = _variant_from_results(
        semantic_result=semantic_swap,
        structural_result=structural_swap,
        shuffled_result=shuffled_swap,
        source_graph=swap_source,
        target_graph=swap_target,
        structural_error_code=structural_error,
    )
    diagnostics = {
        "base": _unavailable_variant_diagnostic(),
        "system_swap": _unavailable_variant_diagnostic(),
    }
    if (
        structural_error is None
        and bound_left is not None
        and bound_right is not None
        and structural_base is not None
        and structural_swap is not None
        and shuffled_base is not None
        and shuffled_swap is not None
    ):
        diagnostics = {
            "base": _variant_diagnostic(
                semantic_result=semantic_base,
                structural_result=structural_base,
                shuffled_result=shuffled_base,
                left_bound=bound_left,
                right_bound=bound_right,
            ),
            "system_swap": _variant_diagnostic(
                semantic_result=semantic_swap,
                structural_result=structural_swap,
                shuffled_result=shuffled_swap,
                left_bound=bound_right,
                right_bound=bound_left,
            ),
        }
    if (
        bound_left is None
        or bound_right is None
        or structural_base is None
        or structural_swap is None
        or shuffled_base is None
        or shuffled_swap is None
    ):
        raise ScarCssmActionInfrastructureError(
            "SCAR_ACTION_RECEIPT_ARCHIVE_INCOMPLETE"
        )
    private_mechanism_receipts = {
        "availability": "COMPLETE",
        "error_code": None,
        "semantic_matrix": {
            "receipt": _receipt_archive_entry(
                semantic_matrix.receipt_bytes
            ),
            "rows": [list(row) for row in semantic_matrix.matrix.rows],
        },
        "sides": {
            "left": _side_receipt_archive(
                envelope=left_envelope,
                relation_set=left_set,
                bound=bound_left,
            ),
            "right": _side_receipt_archive(
                envelope=right_envelope,
                relation_set=right_set,
                bound=bound_right,
            ),
        },
        "variants": {
            "base": _mapping_receipt_archive(
                semantic_result=semantic_base,
                structural_result=structural_base,
                shuffled_result=shuffled_base,
            ),
            "system_swap": _mapping_receipt_archive(
                semantic_result=semantic_swap,
                structural_result=structural_swap,
                shuffled_result=shuffled_swap,
            ),
        },
    }
    return {
        "diagnostics": diagnostics,
        "execution": {
            "document_call_count": document_calls,
            "error_code": structural_error,
            "structural_status": (
                "EXECUTED_WITHOUT_TYPED_FAILURE"
                if structural_error is None
                else "TYPED_FAILURE"
            ),
        },
        "item_token": item_token,
        "private_mechanism_receipts": private_mechanism_receipts,
        "proposal_pools": {
            "base": base_pools,
            "system_swap": swap_pools,
        },
        "variants": {
            "base": base_prediction,
            "system_swap": swap_prediction,
        },
    }


__all__ = [
    "ARM_IDS",
    "DocumentSelector",
    "STRUCTURAL_ERROR_CODES",
    "ScarCssmActionError",
    "ScarCssmActionInfrastructureError",
    "VARIANT_NAMES",
    "VERSION",
    "form_scar_cssm_item_action_v1",
]
