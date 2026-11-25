"""Tests for Hegel M8.2 Offline Container Trajectory Replay Calibrator & Variational Router."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from hegel_machine.cognition.m8_offline_container_planner import (
    M8OfflineContainerPlanner,
    adjudicate_minimax_variational_tiebreak,
    classify_public_container_verification_state,
    evaluate_planner_on_records,
    load_historical_suite_episode_records,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_m8_2_four_state_verification_and_symmetric_minimax_controls() -> None:
    """Verify `skill`/`nogood` activity NEVER forges `VERIFIED` and `50` skill calls NEVER flip worse error (`150mm > 100mm`)."""
    # 1. `skill >= 1` or `nogood >= 1` without a domain target benchmark must be `SUPPORTED_IN_PARTIAL_SCOPE`, NEVER `VERIFIED`!
    st_partial, _ = classify_public_container_verification_state(
        benchmark="Hegel-OOD-Holdout-12",
        ended_by_finish=True,
        contract_accepted=True,
        public_metrics={},
        skill_activation_count=3,
        nogood_count=2,
    )
    assert st_partial == "SUPPORTED_IN_PARTIAL_SCOPE"

    st_bm, _ = classify_public_container_verification_state(
        benchmark="BioMysteryBench-Preview",
        ended_by_finish=True,
        contract_accepted=True,
        public_metrics={},
        skill_activation_count=0,
        nogood_count=2,
        submitted_final_answer="Plasmodium vivax relapse",
    )
    assert st_bm == "SUPPORTED_IN_PARTIAL_SCOPE"

    # Explicit domain check -> `VERIFIED`
    st_ver, _ = classify_public_container_verification_state(
        benchmark="Terminal-Bench-Science",
        ended_by_finish=True,
        contract_accepted=True,
        public_metrics={"amr_solve_ok": True},
        skill_activation_count=0,
        nogood_count=0,
    )
    assert st_ver == "VERIFIED"

    # 2. GPT-6 `extra_minimax_activity_control`: Thesis=100mm vs Antithesis=150mm with `antithesis_skill_calls=50` MUST pick `thesis`!
    tb = adjudicate_minimax_variational_tiebreak(
        pub_thesis={"vb_probe_status": "ok", "vb_cross_total": 10, "vb_cross_pass_20mm": 5, "vb_cross_worst_mm": 100.0},
        pub_antithesis={"vb_probe_status": "ok", "vb_cross_total": 10, "vb_cross_pass_20mm": 5, "vb_cross_worst_mm": 150.0},
        antithesis_skill_calls=50,
    )
    assert tb is not None
    assert tb["selected_branch"] == "thesis"


def test_m8_offline_replay_prefix_clean_nontrivial_nig_and_zero_leakage() -> None:
    """Verify raw unpatched replay (zero private-label imputation), non-trivial NIG governance, and cross-suite generalization."""
    m74_dir = REPO_ROOT / "docs/evidence/m7_4_exp1_frozen_20task_and_ablations_20261124"
    m75_dir = REPO_ROOT / "docs/evidence/m7_5_exp2_holdout_ood_12task_blind_20261124"
    m76a_dir = REPO_ROOT / "docs/evidence/m7_6_exp3a_cross_model_luna_holdout12_20261124"

    recs_74 = load_historical_suite_episode_records(m74_dir)
    recs_75 = load_historical_suite_episode_records(m75_dir)
    recs_76a = load_historical_suite_episode_records(m76a_dir)
    assert len(recs_74) == 20
    assert len(recs_75) == 12
    assert len(recs_76a) == 12

    # 1. Verify honest epistemic tiers on Holdout-12 (`ho12` -> `UNKNOWN`; `ho04`/`ho06` -> `SUPPORTED_IN_PARTIAL_SCOPE`)
    by_id_75 = {r["task_id"]: r for r in recs_75}
    assert by_id_75["ho12__multicopy_plasmid_cohort_consensus"]["pre_action_state"]["public_verification_state"] == "UNKNOWN"
    assert by_id_75["ho04__magnetometer_hard_soft_iron_calibration"]["pre_action_state"]["public_verification_state"] == "SUPPORTED_IN_PARTIAL_SCOPE"
    assert by_id_75["ho06__plate_evaporation_deconfounded_ic50"]["pre_action_state"]["public_verification_state"] == "SUPPORTED_IN_PARTIAL_SCOPE"

    # 2. Verify raw unpatched replay on `tb__traffic-flux-inversion` (`apply_m7_7_traffic_contract_fix_in_replay=False` default)
    by_id_74 = {r["task_id"]: r for r in recs_74}
    assert by_id_74["tb__traffic-flux-inversion"]["outcomes"]["m8_twobranch_pass"] is False

    # 3. Train ONLY on raw `m7_4` (`0` Holdout exposure!) -> Test Blind on `m7_5` & `m7_6a`!
    planner_frontier = M8OfflineContainerPlanner.fit_from_historical_records(recs_74, suite_ids=[m74_dir.name])
    roundtrip = M8OfflineContainerPlanner.from_json_dict(planner_frontier.to_json_dict())

    eval_75 = evaluate_planner_on_records(roundtrip, recs_75)
    assert eval_75["learned_nig_governed_task_count"] >= 8
    assert eval_75["m8_variational_selective_passes"] == 12
    assert eval_75["m8_variational_selective_passes"] > eval_75["rule_only_selective_passes"]
    assert eval_75["spawn_rate"] <= 0.40

    # Verify NIG posterior is genuinely active (`decision_threshold = 1e9` changes decisions!)
    suppressed = deepcopy(roundtrip)
    suppressed.decision_threshold = 1e9
    eval_75_suppressed = evaluate_planner_on_records(suppressed, recs_75)
    assert eval_75_suppressed["spawn_second_branch_count"] < eval_75["spawn_second_branch_count"]
    assert eval_75_suppressed["m8_variational_selective_passes"] < eval_75["m8_variational_selective_passes"]

    eval_76a = evaluate_planner_on_records(roundtrip, recs_76a)
    assert eval_76a["m8_variational_selective_passes"] >= 11
    assert eval_76a["m8_variational_selective_passes"] > eval_76a["rule_only_selective_passes"]

    # 4. Raw Leave-One-Task-Out (`LOTO`) on `m7_4` (unpatched archive -> `14/20`, strictly beating M7 Selective `12/20`!)
    loto_passes = 0
    for i in range(len(recs_74)):
        train_sub = [r for j, r in enumerate(recs_74) if j != i]
        pl_i = M8OfflineContainerPlanner.fit_from_historical_records(train_sub, suite_ids=["m7_4_loto"])
        ev_i = evaluate_planner_on_records(pl_i, [recs_74[i]])
        loto_passes += ev_i["m8_variational_selective_passes"]
    assert loto_passes >= 14
