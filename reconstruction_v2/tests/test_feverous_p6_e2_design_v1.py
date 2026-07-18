from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "manifests" / "feverous_p6_e2_evaluator_design_v1.json"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _semantic_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    declared = body.pop(field)
    raw = json.dumps(
        body,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    assert isinstance(declared, str)
    assert hashlib.sha256(raw).hexdigest() == declared
    return declared


def test_design_self_hash_and_source_qualifications_are_bound() -> None:
    design = _load(DESIGN)
    assert _semantic_hash(design, "design_sha256") == (
        "6193646baca9e35820a5d157bc248012fbd478c89a45db7d879295c4d64f0181"
    )
    bindings = design["source_bindings"]
    assert isinstance(bindings, dict)
    sources = {
        "source_audit_sha256": (
            "feverous_p6_e2_source_audit_v1.json",
            "audit_sha256",
        ),
        "annotation_qualification_sha256": (
            "feverous_annotation_source_qualification_v1.json",
            "qualification_sha256",
        ),
        "wikipedia_qualification_sha256": (
            "feverous_wikipedia_source_qualification_v1.json",
            "qualification_sha256",
        ),
    }
    for binding, (name, field) in sources.items():
        observed = _load(ROOT / "manifests" / name)
        assert bindings[binding] == observed[field]


def test_four_family_blocks_and_single_closed_corpus_are_fixed() -> None:
    design = _load(DESIGN)
    blocks = design["block_contract"]
    corpus = design["corpus_contract"]
    assert isinstance(blocks, dict) and isinstance(corpus, dict)
    families = blocks["native_confirmatory_families"]
    assert isinstance(families, list) and len(families) == 4
    assert blocks["total_TRAIN_items"] == 288
    total = 0
    for name in ("A_form", "F_search", "A_hold", "M_search"):
        block = blocks[name]
        assert isinstance(block, dict)
        assert block["total"] == block["per_family"] * len(families)
        assert block["per_family"] % 2 == 0
        total += block["total"]
    assert total == blocks["total_TRAIN_items"]
    assert corpus["atomic_unit_count"] == 8192
    assert corpus["shared_across_arms_and_blocks"] is True


def test_agent_does_not_consume_hippo_candidates_and_e2_has_one_fit() -> None:
    design = _load(DESIGN)
    action = design["action_contract"]
    evaluator = design["evaluator_contract"]
    assert isinstance(action, dict) and isinstance(evaluator, dict)
    assert action["hipporag_output_as_agent_candidate_or_feature"] is False
    assert action["action_registry"] == [
        "R0_DENSE5",
        "R1_P6_DIRECT_B2",
        "R2_P6_PATH1_B2",
        "R3_P6_PATH2_B2",
    ]
    assert evaluator["E2"]["ridge"] == (
        "weighted_closed_form_ridge_lambda_exactly_1_no_intercept_no_hyperparameter_search"
    )
    assert evaluator["E2"]["single_final_fit"].startswith(
        "after_the_descriptive_cross_fit_fit_once"
    )
    assert evaluator["F_policy"]["freeze_before_A_hold"] is True


def test_dev_and_late_label_boundaries_are_not_optional() -> None:
    design = _load(DESIGN)
    dev = design["development_boundary"]
    acquisition = design["acquisition_contract"]
    promotion = design["promotion_and_claim_contract"]
    assert isinstance(dev, dict)
    assert dev["DEV_claim_label_evidence_or_challenge_read_by_formal_TRAIN_controller"] is False
    assert dev["not_a_failure_backup"] is True
    assert acquisition["F_search_gold_pack_created"] is False
    assert promotion["no_promotion"] == (
        "retain_E0_epoch_and_leave_M_search_unopened"
    )
