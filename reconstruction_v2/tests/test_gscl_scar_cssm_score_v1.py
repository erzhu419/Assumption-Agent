from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import runpy
from typing import Any

import pytest

from assumption_agent.benchmarks import gscl_scar_cssm_action_v1 as action
from assumption_agent.benchmarks import gscl_scar_cssm_score_v1 as scorer
from assumption_agent.benchmarks import gscl_scar_cssm_source_v1 as source


_SECRET = bytes(range(source.HMAC_SECRET_BYTES))
_STUDY_ID = "SCAR_CSSM_SCORE_SYNTHETIC_V1"


def _source_row(
    source_id: int,
    *,
    domains: tuple[str, str],
    mappings: tuple[tuple[str, str], ...],
) -> source._SourceRow:  # noqa: SLF001
    raw = f"synthetic-private-row-{source_id}".encode("ascii")
    return source._SourceRow(  # noqa: SLF001
        raw_line_sha256=hashlib.sha256(raw).hexdigest(),
        source_id=source_id,
        system_a=f"left system {source_id}",
        system_b=f"right system {source_id}",
        system_a_domain=domains[0],
        system_b_domain=domains[1],
        system_a_background=f"left background {source_id}",
        system_b_background=f"right background {source_id}",
        mappings=mappings,
    )


def _bound_fixture_packs():
    # ID 24 is in the frozen ambiguous-secondary ID set.  The fixture is not
    # and cannot be validated as the official source; it only exercises the
    # scorer's schema and arithmetic.
    rows = (
        _source_row(
            1,
            domains=("same", "same"),
            mappings=(("alpha", "one"), ("beta", "two")),
        ),
        _source_row(
            24,
            domains=("left-domain", "right-domain"),
            mappings=(("gamma", "three"), ("delta", "four")),
        ),
    )
    action_core, label_core = source._build_core_packs(  # noqa: SLF001
        rows, secret=_SECRET, study_id=_STUDY_ID
    )
    return source._finish_packs(  # noqa: SLF001
        action_core, label_core, secret=_SECRET, study_id=_STUDY_ID
    )


def _label_index(label_pack):
    return {item["item_token"]: item for item in label_pack["items"]}


def _wrong_bijection(pair_set: list[list[str]]) -> list[list[str]]:
    targets = [pair[1] for pair in pair_set]
    rotated = targets[1:] + targets[:1]
    return [[pair[0], target] for pair, target in zip(pair_set, rotated, strict=True)]


def _answer(pairs: list[list[str]]) -> dict[str, Any]:
    return {"disposition": "ANSWER", "pairs": pairs, "error_code": None}


def _abstain() -> dict[str, Any]:
    return {"disposition": "ABSTAIN", "pairs": None, "error_code": None}


def _error() -> dict[str, Any]:
    return {
        "disposition": "ERROR",
        "pairs": None,
        "error_code": "PROPOSAL_CONSTRUCTION_TYPED_FAILURE",
    }


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _variant_diagnostic(
    *, variant_name: str, variant: dict[str, Any]
) -> dict[str, Any]:
    semantic_hash = _hash(f"{variant_name}-semantic-mapping")
    structural_hash = _hash(f"{variant_name}-structural-mapping")
    shuffle_hash = _hash(f"{variant_name}-shuffle-mapping")
    mapping_hashes = {
        "semantic_only": semantic_hash,
        "flat_structural": structural_hash,
        "full_no_composition": structural_hash,
        "full_with_length2_composition": structural_hash,
        "full_with_length2_composition_target_color_shuffle": shuffle_hash,
    }
    arm_diagnostics = {}
    for arm_id in scorer.ARM_IDS:
        answered = variant["arms"][arm_id]["disposition"] == "ANSWER"
        arm_diagnostics[arm_id] = {
            "selected_operator": (
                "ori_keep.pol_keep.slots_identity" if answered else None
            ),
            "semantic_origin_count": 1 if answered else 0,
            "structural_origin_count": (
                0 if arm_id == "semantic_only" or not answered else 1
            ),
            "incidence_match_count": 2 if answered else 0,
            "incidence_total_count": 2 if answered else 0,
            "length2_path_count": 1 if answered else 0,
            "length2_path_total_count": 1 if answered else 0,
            "typed_incidence_verified": answered,
            "length2_composition_verified": answered,
            "proposal_hash": (
                _hash(f"{variant_name}-{arm_id}-proposal") if answered else None
            ),
            "semantic_score": 100 if answered else None,
            "flat_structural_score": 2 if answered else None,
        }
    return {
        "structural_diagnostics_available": True,
        "target_color_shuffle_effective": True,
        "left_binder": {
            "coverage_disposition": "PARTIAL_SELECTED_SET",
            "unbound_count": 1,
            "dropped_edge_count": 1,
            "retained_edge_count": 2,
            "zero_degree_count": 0,
            "endpoint_count": 4,
            "self_loop_count": 0,
        },
        "right_binder": {
            "coverage_disposition": "COMPLETE_SELECTED_SET",
            "unbound_count": 0,
            "dropped_edge_count": 0,
            "retained_edge_count": 3,
            "zero_degree_count": 0,
            "endpoint_count": 4,
            "self_loop_count": 0,
        },
        "left_graph_receipt_sha256": _hash(f"{variant_name}-left-graph"),
        "right_graph_receipt_sha256": _hash(f"{variant_name}-right-graph"),
        "mapping_receipt_sha256_by_arm": mapping_hashes,
        "arms": arm_diagnostics,
    }


def _receipt_entry(
    schema: str, label: str, *, extras: dict[str, Any] | None = None
) -> dict[str, Any]:
    body: dict[str, Any] = {"schema": schema, "marker": label}
    if extras:
        body.update(extras)
    receipt = {**body, "self_sha256": scorer._content_hash(body)}  # noqa: SLF001
    raw = scorer._canonical_bytes(receipt)  # noqa: SLF001
    return {
        "receipt": receipt,
        "receipt_sha256": hashlib.sha256(raw).hexdigest(),
        "trailing_lf": False,
    }


def _side_bundle(
    *, variant_label: str, side_ids: list[str], retained_count: int
) -> dict[str, Any]:
    slots = [
        {
            "slot_id": slot_id,
            "normalized_label_sha256": _hash(f"{slot_id}-normalized"),
            "evidence_binding_sha256": _hash(f"{slot_id}-evidence"),
        }
        for slot_id in sorted(side_ids)
    ]
    relations = [
        {
            "relation_id": f"r.{index}",
            "slot0_id": slots[index % len(slots)]["slot_id"],
            "slot1_id": slots[(index + 1) % len(slots)]["slot_id"],
            "generator_kind": "relation",
            "polarity": "neutral",
            "temporal_orientation": "none",
            "causal_orientation": "none",
            "evidence_binding_sha256": _hash(
                f"{variant_label}-relation-{index}"
            ),
        }
        for index in range(retained_count)
    ]
    return {
        "document_envelope": {
            "receipt": _receipt_entry(
                scorer.DOCUMENT_ENVELOPE_RECEIPT_SCHEMA,
                f"{variant_label}-document",
            ),
            "leaf_records": [],
        },
        "bounded_set": {
            "coverage": [],
            "units": [],
            "relation_set_signature_ascii": None,
            "relation_set_signature_sha256": None,
            "receipt": _receipt_entry(
                scorer.BOUNDED_SET_RECEIPT_SCHEMA,
                f"{variant_label}-bounded",
            ),
        },
        "binder": {
            "endpoint_bindings": [],
            "receipt": _receipt_entry(
                scorer.BINDER_RECEIPT_SCHEMA, f"{variant_label}-binder"
            ),
        },
        "slot_graph": {
            "slots": slots,
            "relations": relations,
            "coverage_complete": False,
            "extractor_binding_sha256": _hash(
                f"{variant_label}-extractor"
            ),
            "graph_evidence_binding_sha256": _hash(
                f"{variant_label}-graph-evidence"
            ),
            "receipt": _receipt_entry(
                scorer.SLOT_GRAPH_RECEIPT_SCHEMA,
                f"{variant_label}-graph",
            ),
        },
    }


def _mapping_bundle(label: str, *, arity: int, shuffle: bool) -> dict[str, Any]:
    proposal_body = {
        "flat_structural_score": 2,
        "injective_verified": True,
        "length2_composition_verified": True,
        "length2_path_matched": 1,
        "length2_path_total": 1,
        "operator_id": "ori_keep.pol_keep.slots_identity",
        "origins": ["semantic_kbest", "structure_kbest"],
        "semantic_score": 100,
        "target_indices": list(range(arity)),
        "typed_incidence_matched": 2,
        "typed_incidence_total": 2,
        "typed_incidence_verified": True,
    }
    proposal = {
        **proposal_body,
        "proposal_hash": scorer._content_hash(proposal_body),  # noqa: SLF001
    }
    choices = [
        {
            "arm": arm_id,
            "disposition": "SELECTED",
            "proposal_hash": proposal["proposal_hash"],
            "reason_ids": [],
        }
        for arm_id in scorer.MAPPING_ARM_IDS
    ]
    return {
        "assignment_subproblems_solved": 1,
        "choices": choices,
        "proposals": [proposal],
        "receipt": _receipt_entry(scorer.MAPPING_RECEIPT_SCHEMA, label),
        "target_color_shuffle_effective": shuffle,
    }


def _complete_private_bundle(
    action_item: dict[str, Any], diagnostics: dict[str, Any]
) -> dict[str, Any]:
    left_ids = sorted(
        slot["opaque_slot_id"]
        for slot in action_item["variants"]["base"]["left"]["slots"]
    )
    right_ids = sorted(
        slot["opaque_slot_id"]
        for slot in action_item["variants"]["base"]["right"]["slots"]
    )
    matrix_rows = [
        [left, right, 1_000_000] for left in left_ids for right in right_ids
    ]
    matrix_receipt = _receipt_entry(
        scorer.SEMANTIC_MATRIX_RECEIPT_SCHEMA,
        "semantic-matrix",
        extras={"matrix_commitment": scorer._content_hash(matrix_rows)},  # noqa: SLF001
    )
    sides = {
        "left": _side_bundle(
            variant_label="left", side_ids=left_ids, retained_count=2
        ),
        "right": _side_bundle(
            variant_label="right", side_ids=right_ids, retained_count=3
        ),
    }
    variants: dict[str, Any] = {}
    for variant_name in scorer.VARIANT_NAMES:
        semantic = _mapping_bundle(
            f"{variant_name}-semantic", arity=len(left_ids), shuffle=False
        )
        structural = _mapping_bundle(
            f"{variant_name}-structural", arity=len(left_ids), shuffle=False
        )
        shuffled = _mapping_bundle(
            f"{variant_name}-shuffle", arity=len(left_ids), shuffle=True
        )
        variants[variant_name] = {
            "semantic_mapping": semantic,
            "structural_mapping": structural,
            "target_color_shuffle_mapping": shuffled,
        }
        diagnostic = diagnostics[variant_name]
        left_side = sides["left" if variant_name == "base" else "right"]
        right_side = sides["right" if variant_name == "base" else "left"]
        diagnostic["left_graph_receipt_sha256"] = left_side["slot_graph"][
            "receipt"
        ]["receipt_sha256"]
        diagnostic["right_graph_receipt_sha256"] = right_side["slot_graph"][
            "receipt"
        ]["receipt_sha256"]
        diagnostic["mapping_receipt_sha256_by_arm"] = {
            "semantic_only": semantic["receipt"]["receipt_sha256"],
            "flat_structural": structural["receipt"]["receipt_sha256"],
            "full_no_composition": structural["receipt"]["receipt_sha256"],
            "full_with_length2_composition": structural["receipt"][
                "receipt_sha256"
            ],
            "full_with_length2_composition_target_color_shuffle": shuffled[
                "receipt"
            ]["receipt_sha256"],
        }
    return {
        "availability": "COMPLETE",
        "error_code": None,
        "semantic_matrix": {"receipt": matrix_receipt, "rows": matrix_rows},
        "sides": sides,
        "variants": variants,
    }


def _prediction_rows(action_pack, label_pack):
    labels = _label_index(label_pack)
    rows = []
    for action_item in action_pack["items"]:
        token = action_item["item_token"]
        label = labels[token]
        variants: dict[str, Any] = {}
        pools: dict[str, Any] = {}
        for variant_name in scorer.VARIANT_NAMES:
            gold = copy.deepcopy(label["gold_pairs"][variant_name])
            wrong = _wrong_bijection(gold)
            variants[variant_name] = {
                "arms": {
                    "semantic_only": _answer(wrong),
                    "flat_structural": _answer(wrong),
                    "full_no_composition": _answer(wrong),
                    "full_with_length2_composition": _answer(gold),
                    "full_with_length2_composition_target_color_shuffle": (
                        _answer(wrong)
                    ),
                }
            }
            pools[variant_name] = {
                "semantic_kbest": [wrong],
                "structure_kbest": [wrong, gold],
            }
        diagnostics = {
            variant_name: _variant_diagnostic(
                variant_name=variant_name, variant=variants[variant_name]
            )
            for variant_name in scorer.VARIANT_NAMES
        }
        private_receipts = _complete_private_bundle(action_item, diagnostics)
        rows.append(
            {
                "item_token": token,
                "variants": variants,
                "proposal_pools": pools,
                "diagnostics": diagnostics,
                "private_mechanism_receipts": private_receipts,
                "execution": {
                    "structural_status": "EXECUTED_WITHOUT_TYPED_FAILURE",
                    "error_code": None,
                    "document_call_count": 2,
                },
            }
        )
    return rows


def _fixture():
    action_pack, label_pack = _bound_fixture_packs()
    predictions = scorer._seal_scar_cssm_prediction_pack_for_test_v1(  # noqa: SLF001
        action_pack,
        items=_prediction_rows(action_pack, label_pack),
        secret=_SECRET,
        study_id=_STUDY_ID,
        expected_case_count=2,
    )
    return action_pack, label_pack, predictions


def _score(action_pack, label_pack, predictions):
    return scorer._score_scar_cssm_fixture_v1(  # noqa: SLF001
        action_pack,
        label_pack,
        predictions,
        secret=_SECRET,
        study_id=_STUDY_ID,
        expected_primary_count=1,
        expected_ambiguous_count=1,
    )


def _rehash_prediction(pack: dict[str, Any]) -> None:
    body = {key: value for key, value in pack.items() if key != "self_sha256"}
    pack["self_sha256"] = scorer._content_hash(body)  # noqa: SLF001


def test_exact_five_arm_and_prediction_pack_contract() -> None:
    action, label, prediction = _fixture()
    assert scorer.ARM_IDS == (
        "semantic_only",
        "flat_structural",
        "full_no_composition",
        "full_with_length2_composition",
        "full_with_length2_composition_target_color_shuffle",
    )
    assert set(prediction) == {
        "schema",
        "study_id",
        "source_action_commitment_sha256",
        "arm_ids",
        "variant_names",
        "items",
        "self_sha256",
    }
    assert prediction["source_action_commitment_sha256"] == action[
        "action_commitment_sha256"
    ]
    assert prediction["items"] == sorted(
        prediction["items"], key=lambda row: row["item_token"]
    )
    assert set(prediction["items"][0]) == {
        "item_token",
        "variants",
        "proposal_pools",
        "execution",
        "diagnostics",
        "private_mechanism_receipts",
    }
    assert label["label_commitment_sha256"] != action[
        "action_commitment_sha256"
    ]


def test_prediction_seal_authenticates_action_without_opening_labels() -> None:
    action, label = _bound_fixture_packs()
    with pytest.raises(
        scorer.ScarCssmScoreError,
        match="SCAR_SCORE_ACTION_SECRET_BINDING_INVALID",
    ):
        scorer._seal_scar_cssm_prediction_pack_for_test_v1(  # noqa: SLF001
            action,
            items=_prediction_rows(action, label),
            secret=b"x" * source.HMAC_SECRET_BYTES,
            study_id=_STUDY_ID,
            expected_case_count=2,
        )


def test_primary_metrics_effects_mechanism_and_strata_are_exact() -> None:
    result = _score(*_fixture())
    primary = result.safe_aggregate["cohorts"]["primary_unique_slot"]
    assert primary["effect_authority"] is True
    assert primary["case_count"] == 1
    assert primary["variant_count"] == 2
    assert primary["arms"]["semantic_only"]["pair_micro_accuracy"] == 0.0
    assert primary["arms"]["semantic_only"]["item_macro_pair_f1"] == 0.0
    full = primary["arms"]["full_with_length2_composition"]
    assert full["pair_micro_accuracy"] == 1.0
    assert full["item_macro_pair_f1"] == 1.0
    assert full["strict_exact_rate"] == 1.0
    assert full["answer_coverage"] == 1.0
    assert full["base_swap_consistency"] == 1.0
    assert primary["mechanism"] == {
        "variant_count": 2,
        "semantic_pool_complete_reference_mapping_recall": 0.0,
        "structure_pool_complete_reference_mapping_recall": 1.0,
        "structure_only_added_pool_complete_reference_mapping_recall": 1.0,
    }
    assert primary["private_receipt_archive"]["receipt_counts"] == {
        "semantic_matrix": 1,
        "document_envelope": 2,
        "leaf": 0,
        "bounded_set": 2,
        "binder": 2,
        "slot_graph": 2,
        "mapping": 6,
        "total": 15,
    }
    assert primary["private_receipt_archive"]["complete_rate"] == 1.0
    diagnostic = primary["execution_diagnostics"]
    assert diagnostic["structural_diagnostics_available_rate"] == 1.0
    assert diagnostic["target_color_shuffle_effective_rate_among_available"] == 1.0
    assert diagnostic["binder_totals"] == {
        "unbound_count": 2,
        "dropped_edge_count": 2,
        "retained_edge_count": 10,
        "zero_degree_count": 0,
        "endpoint_count": 16,
        "self_loop_count": 0,
    }
    assert diagnostic["arms"]["semantic_only"][
        "semantic_origin_rate_among_selected"
    ] == 1.0
    assert diagnostic["arms"]["semantic_only"]["incidence_match_rate"] == 1.0
    assert diagnostic["arms"]["semantic_only"]["length2_path_match_rate"] == 1.0
    assert diagnostic["arms"]["semantic_only"][
        "typed_incidence_verified_rate_among_selected"
    ] == 1.0
    assert diagnostic["binder_unbound_endpoint_rate"] == 0.125
    assert primary["stratified"]["domain_relation"]["intra"]["case_count"] == 1
    assert primary["stratified"]["domain_relation"]["cross"]["case_count"] == 0
    assert primary["stratified"]["domain_relation"]["cross"][
        "full_minus_semantic_pair_f1_secondary"
    ] is None
    assert primary["stratified"]["arity"]["2"]["case_count"] == 1
    for effect in primary["paired_effects"].values():
        assert effect["mean_difference"] == 1.0
        assert effect["bootstrap_confidence_interval"] == [1.0, 1.0]
        assert effect["bootstrap_samples"] == scorer.BOOTSTRAP_SAMPLES
        assert effect["strict_exact_paired_effect"]["mean_difference"] == 1.0
        assert effect["strict_exact_paired_effect"][
            "effect_authority"
        ] == "SECONDARY_DESCRIPTIVE_ONLY"
    primary_effect = primary["paired_effects"][scorer.PRIMARY_EFFECT_NAME]
    assert primary_effect["effect_authority"] == "SOLE_PRIMARY_CONFIRMATORY"
    assert primary_effect["passes_primary_success_rule"] is True
    assert primary["primary_effect_disposition"] == "PASS"
    assert primary["confirmatory_contract"] == {
        "primary_arm_id": scorer.PRIMARY_ARM_ID,
        "primary_comparator_arm_id": scorer.PRIMARY_COMPARATOR_ID,
        "primary_endpoint": scorer.PRIMARY_ENDPOINT,
        "primary_success_rule": scorer.PRIMARY_SUCCESS_RULE,
        "multiplicity": "single_predeclared_primary_comparison_no_adjustment",
        "sampling_unit": (
            "scar_primary_item_with_base_and_system_swap_averaged_within_item"
        ),
        "population_scope": (
            "frozen_scar_primary_cohort_only_no_population_generalization"
        ),
        "secondary_endpoints_do_not_change_primary_disposition": True,
    }
    assert result.safe_aggregate["primary_effect_disposition"] == "PASS"


def test_ambiguous_cohort_is_secondary_diagnostic_only() -> None:
    result = _score(*_fixture())
    ambiguous = result.safe_aggregate["cohorts"]["ambiguous_secondary"]
    assert ambiguous["effect_authority"] is False
    assert ambiguous["paired_effects"] is None
    assert ambiguous["disposition"] == "SECONDARY_EXECUTION_DIAGNOSTIC_ONLY"
    assert ambiguous["case_count"] == 1
    assert ambiguous["stratified"]["domain_relation"]["cross"]["case_count"] == 1


def test_abstain_and_error_are_zero_not_missing() -> None:
    action, label, prediction = _fixture()
    primary_token = next(
        item["item_token"]
        for item in label["items"]
        if item["strata"]["cohort"] == "primary_unique_slot"
    )
    row = next(
        item
        for item in prediction["items"]
        if item["item_token"] == primary_token
    )
    row["variants"]["base"]["arms"]["semantic_only"] = _abstain()
    row["variants"]["system_swap"]["arms"]["semantic_only"] = _error()
    for variant_name in scorer.VARIANT_NAMES:
        row["diagnostics"][variant_name]["arms"]["semantic_only"] = {
            "selected_operator": None,
            "semantic_origin_count": 0,
            "structural_origin_count": 0,
            "incidence_match_count": 0,
            "incidence_total_count": 0,
            "length2_path_count": 0,
            "length2_path_total_count": 0,
            "typed_incidence_verified": False,
            "length2_composition_verified": False,
            "proposal_hash": None,
            "semantic_score": None,
            "flat_structural_score": None,
        }
    _rehash_prediction(prediction)
    primary = _score(action, label, prediction).safe_aggregate["cohorts"][
        "primary_unique_slot"
    ]["arms"]["semantic_only"]
    assert primary["pair_micro_accuracy"] == 0.0
    assert primary["item_macro_pair_f1"] == 0.0
    assert primary["answer_coverage"] == 0.0
    assert primary["both_variant_answer_coverage"] == 0.0
    assert primary["base_swap_consistency"] is None
    assert primary["base_swap_consistency_coverage"] == 0.0


@pytest.mark.parametrize(
    "mutator",
    [
        lambda answer, left, right: answer.update({"pairs": answer["pairs"][:-1]}),
        lambda answer, left, right: answer.update(
            {"pairs": [[left[0], right[0]], [left[1], right[0]]]}
        ),
        lambda answer, left, right: answer.update(
            {"pairs": [[right[0], left[0]], [right[1], left[1]]]}
        ),
    ],
)
def test_answer_must_be_complete_injective_and_side_correct(mutator) -> None:
    action, label = _bound_fixture_packs()
    rows = _prediction_rows(action, label)
    row = rows[0]
    action_by_token = {value["item_token"]: value for value in action["items"]}
    action_item = action_by_token[row["item_token"]]
    left = [
        value["opaque_slot_id"]
        for value in action_item["variants"]["base"]["left"]["slots"]
    ]
    right = [
        value["opaque_slot_id"]
        for value in action_item["variants"]["base"]["right"]["slots"]
    ]
    answer = row["variants"]["base"]["arms"]["semantic_only"]
    mutator(answer, left, right)
    with pytest.raises(scorer.ScarCssmScoreError):
        scorer._seal_scar_cssm_prediction_pack_for_test_v1(  # noqa: SLF001
            action,
            items=rows,
            secret=_SECRET,
            study_id=_STUDY_ID,
            expected_case_count=2,
        )


def test_proposal_pool_requires_unique_complete_bijections() -> None:
    action, label = _bound_fixture_packs()
    rows = _prediction_rows(action, label)
    pool = rows[0]["proposal_pools"]["base"]["semantic_kbest"]
    pool.append(copy.deepcopy(pool[0]))
    with pytest.raises(
        scorer.ScarCssmScoreError, match="SCAR_SCORE_PROPOSAL_POOL_INVALID"
    ):
        scorer._seal_scar_cssm_prediction_pack_for_test_v1(  # noqa: SLF001
            action,
            items=rows,
            secret=_SECRET,
            study_id=_STUDY_ID,
            expected_case_count=2,
        )


@pytest.mark.parametrize(
    "execution",
    [
        {
            "structural_status": "EXECUTED_WITHOUT_TYPED_FAILURE",
            "error_code": None,
            "document_call_count": 1,
        },
        {
            "structural_status": "TYPED_FAILURE",
            "error_code": None,
            "document_call_count": 1,
        },
        {
            "structural_status": "UNKNOWN",
            "error_code": "INTERNAL_TYPED_FAILURE",
            "document_call_count": 2,
        },
    ],
)
def test_execution_contract_is_fixed(execution) -> None:
    action, label = _bound_fixture_packs()
    rows = _prediction_rows(action, label)
    rows[0]["execution"] = execution
    with pytest.raises(scorer.ScarCssmScoreError, match="SCAR_SCORE_EXECUTION_INVALID"):
        scorer._seal_scar_cssm_prediction_pack_for_test_v1(  # noqa: SLF001
            action,
            items=rows,
            secret=_SECRET,
            study_id=_STUDY_ID,
            expected_case_count=2,
        )


def test_unavailable_diagnostics_require_exact_null_and_zero_shape() -> None:
    action, label = _bound_fixture_packs()
    rows = _prediction_rows(action, label)
    empty_arm = {
        "selected_operator": None,
        "semantic_origin_count": 0,
        "structural_origin_count": 0,
        "incidence_match_count": 0,
        "incidence_total_count": 0,
        "length2_path_count": 0,
        "length2_path_total_count": 0,
        "typed_incidence_verified": False,
        "length2_composition_verified": False,
        "proposal_hash": None,
        "semantic_score": None,
        "flat_structural_score": None,
    }
    unavailable = {
        "structural_diagnostics_available": False,
        "target_color_shuffle_effective": None,
        "left_binder": None,
        "right_binder": None,
        "left_graph_receipt_sha256": None,
        "right_graph_receipt_sha256": None,
        "mapping_receipt_sha256_by_arm": {
            arm_id: None for arm_id in scorer.ARM_IDS
        },
        "arms": {arm_id: dict(empty_arm) for arm_id in scorer.ARM_IDS},
    }
    rows[0]["diagnostics"] = {
        variant_name: copy.deepcopy(unavailable)
        for variant_name in scorer.VARIANT_NAMES
    }
    for variant_name in scorer.VARIANT_NAMES:
        rows[0]["variants"][variant_name]["arms"] = {
            arm_id: {
                "disposition": "ERROR",
                "pairs": None,
                "error_code": "SLOT_BINDER_TYPED_FAILURE",
            }
            for arm_id in scorer.ARM_IDS
        }
        rows[0]["proposal_pools"][variant_name] = {
            "semantic_kbest": [],
            "structure_kbest": [],
        }
    rows[0]["execution"] = {
        "structural_status": "TYPED_FAILURE",
        "error_code": "SLOT_BINDER_TYPED_FAILURE",
        "document_call_count": 0,
    }
    rows[0]["private_mechanism_receipts"] = {
        "availability": "PREMODEL_TYPED_FAILURE",
        "error_code": "SLOT_BINDER_TYPED_FAILURE",
        "semantic_matrix": None,
        "sides": {"left": None, "right": None},
        "variants": {"base": None, "system_swap": None},
    }
    pack = scorer._seal_scar_cssm_prediction_pack_for_test_v1(  # noqa: SLF001
        action,
        items=rows,
        secret=_SECRET,
        study_id=_STUDY_ID,
        expected_case_count=2,
    )
    sealed_row = next(
        item for item in pack["items"] if item["item_token"] == rows[0]["item_token"]
    )
    assert sealed_row["diagnostics"]["base"][
        "structural_diagnostics_available"
    ] is False

    rows[0]["private_mechanism_receipts"]["sides"]["left"] = {}
    with pytest.raises(
        scorer.ScarCssmScoreError, match="SCAR_SCORE_PRIVATE_RECEIPT_INVALID"
    ):
        scorer._seal_scar_cssm_prediction_pack_for_test_v1(  # noqa: SLF001
            action,
            items=rows,
            secret=_SECRET,
            study_id=_STUDY_ID,
            expected_case_count=2,
        )


def test_prediction_pack_is_exact_canonical_and_tamper_evident() -> None:
    action, label, prediction = _fixture()
    prediction["unexpected"] = True
    with pytest.raises(scorer.ScarCssmScoreError):
        _score(action, label, prediction)

    action, label, prediction = _fixture()
    prediction["items"].reverse()
    _rehash_prediction(prediction)
    with pytest.raises(
        scorer.ScarCssmScoreError, match="SCAR_SCORE_PREDICTION_COVERAGE_INVALID"
    ):
        _score(action, label, prediction)

    action, label, prediction = _fixture()
    prediction["source_action_commitment_sha256"] = "0" * 64
    _rehash_prediction(prediction)
    with pytest.raises(
        scorer.ScarCssmScoreError, match="SCAR_SCORE_PREDICTION_PACK_INVALID"
    ):
        _score(action, label, prediction)


def test_base_swap_consistency_has_separate_coverage() -> None:
    action, label, prediction = _fixture()
    primary_token = next(
        item["item_token"]
        for item in label["items"]
        if item["strata"]["cohort"] == "primary_unique_slot"
    )
    row = next(
        item
        for item in prediction["items"]
        if item["item_token"] == primary_token
    )
    # Make swap correct while base remains the wrong bijection.  Both are
    # ANSWER, so consistency coverage is one and consistency itself is zero.
    gold = _label_index(label)[primary_token]["gold_pairs"]["system_swap"]
    row["variants"]["system_swap"]["arms"]["semantic_only"] = _answer(gold)
    _rehash_prediction(prediction)
    metric = _score(action, label, prediction).safe_aggregate["cohorts"][
        "primary_unique_slot"
    ]["arms"]["semantic_only"]
    assert metric["base_swap_consistency_coverage"] == 1.0
    assert metric["base_swap_consistency"] == 0.0
    assert metric["item_macro_pair_f1"] == 0.5


def test_bootstrap_is_fixed_seed_and_repeat_exact() -> None:
    first = _score(*_fixture())
    second = _score(*_fixture())
    assert first.private_result == second.private_result
    assert first.safe_aggregate == second.safe_aggregate
    effects = first.safe_aggregate["cohorts"]["primary_unique_slot"][
        "paired_effects"
    ]
    assert [effects[key]["bootstrap_seed"] for key in effects] == [
        scorer.BOOTSTRAP_SEED + index for index in range(4)
    ]


def test_only_full_minus_semantic_controls_primary_disposition() -> None:
    arm_metrics = {
        arm_id: {
            "mean_variant_pair_f1": 1.0,
            "both_variants_strict_exact": True,
        }
        for arm_id in scorer.ARM_IDS
    }
    effects = scorer._paired_effects(  # noqa: SLF001
        [{"arm_case_metrics": arm_metrics}]
    )
    primary = effects[scorer.PRIMARY_EFFECT_NAME]
    assert primary["bootstrap_confidence_interval"] == [0.0, 0.0]
    assert primary["passes_primary_success_rule"] is False
    for name, effect in effects.items():
        if name == scorer.PRIMARY_EFFECT_NAME:
            continue
        assert effect["effect_authority"] == (
            "SECONDARY_MECHANISM_DIAGNOSTIC_ONLY"
        )
        assert effect["primary_success_rule_applies"] is False
        assert effect["passes_primary_success_rule"] is None


def test_safe_aggregate_has_no_private_case_or_content_values() -> None:
    action, label, prediction = _fixture()
    result = _score(action, label, prediction)
    encoded = json.dumps(result.safe_aggregate, sort_keys=True)
    for forbidden in (
        "scar-item-v1-",
        "scar-slot-v1-",
        '"item_token"',
        '"opaque_slot_id"',
        '"background"',
        '"surface"',
        '"gold_pairs"',
        '"per_item"',
        '"proposal_hash"',
        '"left_graph_receipt_sha256"',
        '"right_graph_receipt_sha256"',
        '"mapping_receipt_sha256_by_arm"',
    ):
        assert forbidden not in encoded
    assert "scar-item-v1-" in json.dumps(result.private_result)
    assert result.safe_aggregate["access_counts"] == {
        "source_file_access_count": 0,
        "model_call_count": 0,
        "network_call_count": 0,
        "api_call_count": 0,
        "online_evaluator_call_count": 0,
        "offline_scorer_call_count": 1,
    }


def test_label_or_action_tamper_fails_closed() -> None:
    action, label, prediction = _fixture()
    label["items"][0]["strata"]["arity"] += 1
    with pytest.raises(scorer.ScarCssmScoreError):
        _score(action, label, prediction)

    action, label, prediction = _fixture()
    action["items"][0]["variants"]["base"]["left"]["system"] += " tamper"
    with pytest.raises(scorer.ScarCssmScoreError):
        _score(action, label, prediction)


def test_public_seal_and_score_reject_tiny_fixture_as_nonofficial() -> None:
    action, label = _bound_fixture_packs()
    rows = _prediction_rows(action, label)
    with pytest.raises(scorer.ScarCssmScoreError):
        scorer.seal_scar_cssm_prediction_pack_v1(
            action,
            items=rows,
            secret=_SECRET,
            study_id=_STUDY_ID,
        )
    prediction = scorer._seal_scar_cssm_prediction_pack_for_test_v1(  # noqa: SLF001
        action,
        items=rows,
        secret=_SECRET,
        study_id=_STUDY_ID,
        expected_case_count=2,
    )
    with pytest.raises(
        scorer.ScarCssmScoreError,
        match="SCAR_SCORE_SOURCE_BINDING_INVALID__",
    ):
        scorer.score_scar_cssm_predictions_v1(
            action,
            label,
            prediction,
            secret=_SECRET,
            study_id=_STUDY_ID,
        )


def test_all_arms_and_both_variants_are_mandatory() -> None:
    action, label = _bound_fixture_packs()
    rows = _prediction_rows(action, label)
    del rows[0]["variants"]["base"]["arms"]["flat_structural"]
    with pytest.raises(
        scorer.ScarCssmScoreError, match="SCAR_SCORE_PREDICTION_ARMS_INVALID"
    ):
        scorer._seal_scar_cssm_prediction_pack_for_test_v1(  # noqa: SLF001
            action,
            items=rows,
            secret=_SECRET,
            study_id=_STUDY_ID,
            expected_case_count=2,
        )


def test_diagnostic_denominators_and_verification_are_fail_closed() -> None:
    action, label = _bound_fixture_packs()
    rows = _prediction_rows(action, label)
    diagnostic = rows[0]["diagnostics"]["base"]["arms"][
        "full_with_length2_composition"
    ]
    diagnostic["incidence_total_count"] = 1
    with pytest.raises(
        scorer.ScarCssmScoreError, match="SCAR_SCORE_DIAGNOSTICS_INVALID"
    ):
        scorer._seal_scar_cssm_prediction_pack_for_test_v1(  # noqa: SLF001
            action,
            items=rows,
            secret=_SECRET,
            study_id=_STUDY_ID,
            expected_case_count=2,
        )


def test_actual_action_complete_and_premodel_outputs_fit_scorer_contract() -> None:
    helpers = runpy.run_path(
        str(Path(__file__).with_name("test_gscl_scar_cssm_action_v1.py"))
    )
    item = helpers["_item"]()
    formed = action.form_scar_cssm_item_action_v1(
        item,
        document_selector=helpers["_Selector"](),
        encoder=helpers["_ExactTextEncoder"](),
        encoder_binding_sha256=helpers["_ENCODER"],
    )
    assert scorer._normalize_prediction_item(  # noqa: SLF001
        formed, action_item=item
    ) == formed

    collision = helpers["_item"]()
    collision["variants"]["base"]["left"]["slots"][0]["surface"] = "K"
    collision["variants"]["base"]["left"]["slots"][1]["surface"] = "K"
    collision["variants"]["system_swap"]["right"] = collision["variants"][
        "base"
    ]["left"]
    failed = action.form_scar_cssm_item_action_v1(
        collision,
        document_selector=helpers["_Selector"](),
        encoder=helpers["_ExactTextEncoder"](),
        encoder_binding_sha256=helpers["_ENCODER"],
    )
    assert failed["private_mechanism_receipts"]["availability"] == (
        "PREMODEL_TYPED_FAILURE"
    )
    assert scorer._normalize_prediction_item(  # noqa: SLF001
        failed, action_item=collision
    ) == failed
