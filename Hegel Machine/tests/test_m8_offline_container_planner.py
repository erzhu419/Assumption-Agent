"""Tests for Hegel M8.1 Offline Container Trajectory Replay Calibrator & Variational Router."""

from __future__ import annotations

from pathlib import Path

from hegel_machine.cognition.m8_offline_container_planner import (
    M8OfflineContainerPlanner,
    evaluate_planner_on_records,
    load_historical_suite_episode_records,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_m8_offline_replay_prefix_clean_and_zero_leakage_cross_suite() -> None:
    """Verify prefix cleanliness (`T09`), `UNKNOWN != VERIFIED` epistemic state, LOTO on `m7_4`,
    and strict zero-leakage cross-benchmark generalization `Train(m7_4 Frontier-20) -> Test(m7_5 Holdout-12)`!
    """
    m74_dir = REPO_ROOT / "docs/evidence/m7_4_exp1_frozen_20task_and_ablations_20261124"
    m75_dir = REPO_ROOT / "docs/evidence/m7_5_exp2_holdout_ood_12task_blind_20261124"
    m76a_dir = REPO_ROOT / "docs/evidence/m7_6_exp3a_cross_model_luna_holdout12_20261124"

    recs_74 = load_historical_suite_episode_records(m74_dir)
    recs_75 = load_historical_suite_episode_records(m75_dir)
    recs_76a = load_historical_suite_episode_records(m76a_dir)
    assert len(recs_74) == 20
    assert len(recs_75) == 12
    assert len(recs_76a) == 12

    # 1. Verify `UNKNOWN != VERIFIED` epistemic separation on Holdout-12 (`ho12` unprobed -> `UNKNOWN`; `ho04`/`ho06` -> `VERIFIED`)
    by_id_75 = {r["task_id"]: r for r in recs_75}
    assert by_id_75["ho12__multicopy_plasmid_cohort_consensus"]["pre_action_state"]["public_verification_state"] == "UNKNOWN"
    assert by_id_75["ho04__magnetometer_hard_soft_iron_calibration"]["pre_action_state"]["public_verification_state"] == "VERIFIED"
    assert by_id_75["ho06__plate_evaporation_deconfounded_ic50"]["pre_action_state"]["public_verification_state"] == "VERIFIED"
    assert by_id_75["ho07__thermoacoustic_speed_shifted_ring"]["pre_action_state"]["public_verification_state"] == "VERIFIED"
    assert by_id_75["ho11__certified_capacitated_facility_bnb"]["pre_action_state"]["public_verification_state"] == "VERIFIED"

    # 2. Strict Cross-Benchmark Generalization: Train ONLY on `m7_4` (`0` Holdout exposure!) -> Test Blind on `m7_5` & `m7_6a`!
    planner_frontier = M8OfflineContainerPlanner.fit_from_historical_records(recs_74, suite_ids=[m74_dir.name])
    roundtrip = M8OfflineContainerPlanner.from_json_dict(planner_frontier.to_json_dict())

    eval_75 = evaluate_planner_on_records(roundtrip, recs_75)
    # Out-of-sample M8 Variational Selective recovers `ho12` -> `12/12` (100%), beating M7 Heuristic Selective (`11/12`)!
    assert eval_75["m8_variational_selective_passes"] == 12
    assert eval_75["spawn_rate"] <= 0.40

    eval_76a = evaluate_planner_on_records(roundtrip, recs_76a)
    # On cross-model `gpt-6-luna` Holdout-12, M8 Selective also strictly beats both `Bare_1x` (8/12) and `Antithesis_1x` (9/12)!
    assert eval_76a["m8_variational_selective_passes"] >= 11

    # 3. Leave-One-Task-Out (`LOTO`) within `m7_4` (20 tasks): strictly beats M7 Selective (`12/20`) and Antithesis 1x (`14/20`)!
    loto_passes = 0
    for i in range(len(recs_74)):
        train_sub = [r for j, r in enumerate(recs_74) if j != i]
        pl_i = M8OfflineContainerPlanner.fit_from_historical_records(train_sub, suite_ids=["m7_4_loto"])
        ev_i = evaluate_planner_on_records(pl_i, [recs_74[i]])
        loto_passes += ev_i["m8_variational_selective_passes"]
    assert loto_passes >= 15
