from __future__ import annotations

import json
from pathlib import Path

from hegel_machine.hashing import stable_hash


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "v2_scar_negative_evidence_binding_v1.json"


def test_v2_negative_binding_is_self_addressed_and_does_not_advance_m3() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    manifest_id = payload.pop("manifest_id")
    assert manifest_id == stable_hash(
        payload,
        prefix="v2_scar_negative_binding_",
    )

    binding = payload["hegel_binding"]
    assert binding["current_freeze_version"] == "hegel-freeze-p2b-p3-v1.1.1"
    assert binding["current_child_state_at_binding"] == "NOT_RUN"
    assert binding["formal_roots_advanced_by_this_record"] is False
    assert binding["m3_gate_count_advanced_by_this_record"] == 0
    assert binding["legacy_snapshot_mutated"] is False


def test_v2_negative_claim_boundary_is_not_a_22_or_13_prior_verdict() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    boundary = payload["claim_boundary"]
    assert boundary["all_13_or_22_meta_priors_falsified"] is False
    assert boundary["formal_m2_5_blocked"] is False
    assert boundary["formal_m3_closure_blocked"] is False
    assert boundary["phase3b_effect_interpretation_requires_new_controls"] is True

    result = payload["scar_formal_negative"]
    assert result["commit"] == "4861b2d88ef7e85fb62f32e3d2e1f5c78afe9529"
    assert result["primary"]["disposition"] == "FAIL"
    assert result["interpretation"]["protocol_execution_valid"] is True
    assert (
        result["interpretation"][
            "generalized_counterpoint_abstract_family_falsified"
        ]
        is False
    )
