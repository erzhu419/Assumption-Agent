"""Golden, source-free tests for the SCAR repair mechanism primitives."""

from __future__ import annotations

import copy
from fractions import Fraction

import pytest

from assumption_agent import gscl_scar_cssm_repair_mechanisms_v2 as subject
from assumption_agent.gscl_scar_cssm_repair_contract_v2 import content_hash


_FORMAL_RESULT_SELF = (
    "a5f5411545e8d98386889be606dc6b21db04be606fbf2d258f879b54cc49340b"
)


def _hash(label: str) -> str:
    return content_hash({"fixture": label})


def _graph(prefix: str, *, with_relations: bool = True) -> dict[str, object]:
    slots = [
        {
            "slot_id": f"{prefix}{index}",
            "normalized_label_sha256": _hash(f"{prefix}-label-{index}"),
            "evidence_binding_sha256": _hash(f"{prefix}-slot-{index}"),
        }
        for index in range(3 if with_relations else 2)
    ]
    relations = (
        [
            {
                "relation_id": "rel.alpha",
                "slot0_id": f"{prefix}0",
                "slot1_id": f"{prefix}1",
                "generator_kind": "relation",
                "polarity": "positive",
                "temporal_orientation": "forward",
                "causal_orientation": "none",
                "evidence_binding_sha256": _hash(f"{prefix}-rel-a"),
            },
            {
                "relation_id": "rel.beta",
                "slot0_id": f"{prefix}1",
                "slot1_id": f"{prefix}2",
                "generator_kind": "causal",
                "polarity": "negative",
                "temporal_orientation": "none",
                "causal_orientation": "forward",
                "evidence_binding_sha256": _hash(f"{prefix}-rel-b"),
            },
        ]
        if with_relations
        else []
    )
    return {
        "slots": slots,
        "relations": relations,
        "coverage_complete": False,
        "extractor_binding_sha256": _hash(f"{prefix}-extractor"),
        "graph_evidence_binding_sha256": _hash(f"{prefix}-graph"),
        "receipt": {},
    }


def _proposal(*, empty: bool = False) -> dict[str, object]:
    body: dict[str, object] = {
        "flat_structural_score": 0,
        "injective_verified": True,
        "length2_composition_verified": not empty,
        "length2_path_matched": 0 if empty else 1,
        "length2_path_total": 0 if empty else 1,
        "operator_id": "ori_keep.pol_keep.slots_identity",
        "origins": ["semantic_kbest"] if empty else [
            "semantic_kbest",
            "structure_kbest",
        ],
        "semantic_score": 50 if empty else 123,
        "target_indices": [0, 1] if empty else [0, 1, 2],
        "typed_incidence_matched": 0 if empty else 4,
        "typed_incidence_total": 0 if empty else 4,
        "typed_incidence_verified": not empty,
    }
    return {**body, "proposal_hash": content_hash(body)}


def test_exact_stratified_fold_digest_and_per_stratum_modulo_five() -> None:
    rows = (
        subject.StratifiedFoldRow("item.a", "cross_domain", 2),
        subject.StratifiedFoldRow("item.b", "cross_domain", 2),
        subject.StratifiedFoldRow("item.c", "cross_domain", 2),
        subject.StratifiedFoldRow("item.d", "cross_domain", 2),
        subject.StratifiedFoldRow("item.e", "intra_domain", 2),
        subject.StratifiedFoldRow("item.f", "cross_domain", 3),
        subject.StratifiedFoldRow("item.a", "cross_domain", 2),
    )
    expected_digests = {
        "item.a": "25d028d51f35a87a73c97baa3c6d415777e964c825e118e0f8fb51512b1ea763",
        "item.b": "031f759bd8f96ba92c26e46cc850ccb2a154148d91789725d573ed7b1610167b",
        "item.c": "f26c6f192f8f9d503bb9df9041c77c604620cbd5d896f738aadf74df0179f938",
        "item.d": "96c3e4aa81694895187638a9c04ae70b04d0052ec1d9d2b0bddace45e0484a3e",
        "item.e": "98bbe428a60fa055d2cc58d7e3895df25a1de9e78b6f4f252ed3c75d94a8b30c",
        "item.f": "976c6c46ebef3606a80d79a85f39deff8c16fa67095b9491d5467f944748efc6",
    }
    expected_folds = {
        "item.a": 1,
        "item.b": 0,
        "item.c": 3,
        "item.d": 2,
        # These are each first in a different stratum.
        "item.e": 0,
        "item.f": 0,
    }

    assignments = subject.assign_stratified_folds(
        rows, formal_result_self_sha256=_FORMAL_RESULT_SELF
    )
    for assignment in assignments:
        assert assignment.assignment_digest == expected_digests[
            assignment.canonical_item_id
        ]
        assert assignment.fold_index == expected_folds[
            assignment.canonical_item_id
        ]
    assert assignments[0].fold_index == assignments[-1].fold_index

    reordered = subject.assign_stratified_folds(
        tuple(reversed(rows)), formal_result_self_sha256=_FORMAL_RESULT_SELF
    )
    assert {
        row.canonical_item_id: (row.assignment_digest, row.fold_index)
        for row in reordered
    } == {
        row.canonical_item_id: (row.assignment_digest, row.fold_index)
        for row in assignments
    }


def test_candidate_rank_clips_first_then_uses_semantic_and_hash_ties() -> None:
    candidates = (
        subject.CandidateRankInput("A", 2.0, 10, "d" * 64),
        subject.CandidateRankInput("B", 1.0, 11, "c" * 64),
        subject.CandidateRankInput("C", 1.0, 11, "a" * 64),
        subject.CandidateRankInput("D", 0.9, 999, "b" * 64),
    )
    assert tuple(row.payload for row in subject.rank_candidates(candidates)) == (
        "C",
        "B",
        "A",
        "D",
    )
    assert subject.rank_candidates(tuple(reversed(candidates))) == tuple(
        subject.rank_candidates(candidates)
    )


def test_null_stage_relation_orders_are_exact_golden_sha256() -> None:
    graph = _graph("t")
    expected = {
        "COLOR": (
            (
                "rel.alpha",
                "3ef2cea6eafcb849675b6a84f2e446cc235813e8b35f11ddb1e53439885f1d53",
            ),
            (
                "rel.beta",
                "f4ad6be13bd0499eb555e6a94bc0a598cf99bb64c8e6ddc5674d92108233cba7",
            ),
        ),
        "ROLE": (
            (
                "rel.alpha",
                "850360c87c7a50fc67d7c66ebd0949c6fb6c359f997cfb220ead27e4e1c44d2c",
            ),
            (
                "rel.beta",
                "a7ff52d82399b19eab1132bf906186d4e60ea4078e06aabf4d17b2a21b669f65",
            ),
        ),
        "SIGN": (
            (
                "rel.alpha",
                "99e1845a6c95f54def40b61fb86289b88ccb55a6d2098636039e6258b0da602e",
            ),
            (
                "rel.beta",
                "ef6c1ef57053ddc71e9b5088adb64a2c2790c68bc098a6adb226a2b78f6d6433",
            ),
        ),
    }
    for stage in subject.NULL_STAGES:
        actual = subject.stage_relation_order(
            "item.null.golden", graph, replicate_index=0, stage=stage
        )
        assert tuple((row.relation_id, row.digest) for row in actual) == expected[
            stage
        ]


def test_null_transform_order_uses_archived_sign_fields_and_half_edges_role() -> None:
    graph = _graph("t")
    before = copy.deepcopy(graph)
    transformed = subject.apply_null_package_transform(
        "item.null.golden", graph, replicate_index=0
    )
    assert graph == before
    assert transformed[0] == {
        "relation_id": "rel.alpha",
        # ROLE order puts alpha first, so exactly one of two edges is swapped.
        "slot0_id": "t1",
        "slot1_id": "t0",
        # COLOR and SIGN each left-rotate their exact archived fields.
        "generator_kind": "causal",
        "polarity": "negative",
        "temporal_orientation": "none",
        "causal_orientation": "forward",
        "evidence_binding_sha256": _hash("t-rel-a"),
    }
    assert transformed[1] == {
        "relation_id": "rel.beta",
        "slot0_id": "t1",
        "slot1_id": "t2",
        "generator_kind": "relation",
        "polarity": "positive",
        "temporal_orientation": "forward",
        "causal_orientation": "none",
        "evidence_binding_sha256": _hash("t-rel-b"),
    }


def test_null_recomputation_and_32_replicate_mean_are_exact_and_deterministic() -> None:
    source = _graph("s")
    target = _graph("t")
    proposal = _proposal()
    assert proposal["proposal_hash"] == (
        "b6df0c137c9f83f32fa880ff7897ed02d195cdf37b41f3cc913f503903578dfa"
    )
    original = subject.recompute_structural_features(source, target, proposal)
    assert (
        original.flat_structural_score,
        original.typed_incidence_matched,
        original.typed_incidence_total,
        original.f04_flat_structural_score_per_slot,
        original.f05_typed_incidence_match_rate,
        original.f06_typed_incidence_total_per_slot,
        original.f07_zero_incidence_support,
    ) == (0, 4, 4, Fraction(0), Fraction(1), Fraction(4, 3), Fraction(0))

    first = subject.build_null_package_mean(
        "item.null.golden", source, target, (proposal,)
    )
    second = subject.build_null_package_mean(
        "item.null.golden", source, target, (proposal,)
    )
    assert first == second
    assert len(first) == 1
    assert first[0] == subject.NullProposalMean(
        proposal_hash=str(proposal["proposal_hash"]),
        f04_flat_structural_score_per_slot=Fraction(-2),
        f05_typed_incidence_match_rate=Fraction(1, 4),
        f06_typed_incidence_total_per_slot=Fraction(4, 3),
        f07_zero_incidence_support=Fraction(0),
    )


def test_zero_relation_null_is_retained_as_ineffective_without_resampling() -> None:
    source = _graph("s", with_relations=False)
    target = _graph("t", with_relations=False)
    proposal = _proposal(empty=True)
    assert subject.apply_null_package_transform(
        "item.empty", target, replicate_index=31
    ) == ()
    assert subject.stage_relation_order(
        "item.empty", target, replicate_index=31, stage="COLOR"
    ) == ()
    assert subject.build_null_package_mean(
        "item.empty", source, target, (proposal,)
    ) == (
        subject.NullProposalMean(
            proposal_hash=str(proposal["proposal_hash"]),
            f04_flat_structural_score_per_slot=Fraction(0),
            f05_typed_incidence_match_rate=Fraction(0),
            f06_typed_incidence_total_per_slot=Fraction(0),
            f07_zero_incidence_support=Fraction(1),
        ),
    )


def test_fold_candidate_and_null_inputs_fail_closed() -> None:
    with pytest.raises(subject.ScarRepairMechanismError):
        subject.assign_stratified_folds(
            (subject.StratifiedFoldRow("item", "cross_domain", 2),),
            formal_result_self_sha256="bad",
        )
    with pytest.raises(subject.ScarRepairMechanismError):
        subject.assign_stratified_folds(
            (
                subject.StratifiedFoldRow("item", "cross_domain", 2),
                subject.StratifiedFoldRow("item", "intra_domain", 2),
            ),
            formal_result_self_sha256=_FORMAL_RESULT_SELF,
        )
    with pytest.raises(subject.ScarRepairMechanismError):
        subject.rank_candidates(
            (subject.CandidateRankInput(None, float("nan"), 0, "a" * 64),)
        )

    graph = _graph("t")
    malformed_graph = copy.deepcopy(graph)
    relation = malformed_graph["relations"][0]
    relation["temporal"] = relation.pop("temporal_orientation")
    with pytest.raises(subject.ScarRepairMechanismError):
        subject.stage_relation_order(
            "item", malformed_graph, replicate_index=0, stage="COLOR"
        )
    with pytest.raises(subject.ScarRepairMechanismError):
        subject.apply_null_package_transform(
            "item", graph, replicate_index=32
        )

    source = _graph("s")
    proposal = _proposal()
    tampered_hash = dict(proposal)
    tampered_hash["proposal_hash"] = "0" * 64
    with pytest.raises(subject.ScarRepairMechanismError):
        subject.build_null_package_mean(
            "item", source, graph, (tampered_hash,)
        )
    mismatched_archive = dict(proposal)
    body = dict(mismatched_archive)
    body.pop("proposal_hash")
    body["typed_incidence_matched"] = 3
    mismatched_archive = {**body, "proposal_hash": content_hash(body)}
    with pytest.raises(
        subject.ScarRepairMechanismError,
        match="SCAR_REPAIR_NULL_ARCHIVE_MISMATCH",
    ):
        subject.build_null_package_mean(
            "item", source, graph, (mismatched_archive,)
        )
