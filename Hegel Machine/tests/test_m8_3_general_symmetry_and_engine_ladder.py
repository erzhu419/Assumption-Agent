"""Tests for Hegel M8.3 Generalized Multi-Domain Symmetry Probes & 3-Tier Engine Repair Ladder."""

from __future__ import annotations

import json
from pathlib import Path
import tarfile
from typing import Any
from unittest.mock import patch

from hegel_machine.cognition.hipporag_wormhole_index import HippoRAGWormholeIndex
from hegel_machine.cognition.m6_9_a_dialectical_synthesis_engine import run_m6_9_a_two_branch_episode
from hegel_machine.cognition.m7_autogenous_forest_pretraining import build_blank_axiomatic_philosophical_forest
from hegel_machine.cognition.m7_holdout_ood_blind_benchmark import (
    holdout_tasks_as_frontier_waves,
    holdout_tasks_by_id,
)
from hegel_machine.cognition.m8_container_mechanism_adapter import (
    choose_and_execute_container_mechanism_action,
    probe_public_cohort_contract_and_invariance,
)
from hegel_machine.cognition.m8_offline_container_planner import (
    M8OfflineContainerPlanner,
    classify_public_container_verification_state,
    load_historical_suite_episode_records,
)
from hegel_machine.cognition.m8_persistent_variational_planner import PersistentMechanismGraph

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_m8_3_multidomain_public_symmetry_and_gauge_probes() -> None:
    """Verify generalized public probes catch hardcoded drift/ring shortcuts on `ho08` & `ho05` while passing true solvers."""
    tasks = holdout_tasks_by_id()

    # 1. `ho08` Affine Thermal-Drift Gauge Invariance:
    # Shortcut solver subtracts ONLY the public seed's hardcoded `-1.8 + 0.14 * t` instead of invariant affine projection!
    ho08 = tasks["ho08__prony_viscoelastic_memory_inversion"]
    shortcut_ho08 = """import pickle, numpy as np
def fit_prony_and_predict_stress(experiment: dict) -> dict:
    blob = pickle.load(open('/holdout_task/public_rheology.pkl', 'rb'))
    # Overfits public thermal drift `-1.8 + 0.14*t` -> shifts wildly when extra affine drift is injected!
    t_tr = np.asarray(experiment['train_t'], float)
    raw = np.asarray(experiment['train_stress_raw'], float)
    t_ref = np.asarray(blob['experiment']['train_t'], float)
    resid_bias = float(np.median(raw - (-1.8 + 0.14 * t_tr)) - np.median(blob['experiment']['train_stress_raw'] - (-1.8 + 0.14 * t_ref)))
    return {'G_inf': float(blob['gt_G_inf']) + 10.0 * resid_bias, 'test_stress': np.asarray(blob['gt_test_stress'], float) + resid_bias}
"""
    p_bad_08 = probe_public_cohort_contract_and_invariance(task=ho08, candidate_py_code=shortcut_ho08)
    assert p_bad_08["public_output_contract_ok"] is True  # Passes unperturbed public fit!
    assert p_bad_08["id_renaming_invariant_ok"] is False  # Caught by affine thermal-drift gauge probe!

    # Verify upgraded horizon-truncation + equilibrium-shift probe catches the live M8.3 `ho08` `G_inf <-> slow mode` collinearity!
    archived_ho08_fp = (
        REPO_ROOT
        / "docs/evidence/m8_3_live_engine_symmetry_repair_slice_20261125"
        / "runs/ho08__prony_viscoelastic_memory_inversion/04_submitted_artifacts_snapshot/prony_rheology.py"
    )
    if archived_ho08_fp.is_file():
        p_arch_08 = probe_public_cohort_contract_and_invariance(
            task=ho08,
            candidate_py_code=archived_ho08_fp.read_text(encoding="utf-8"),
        )
        assert p_arch_08["public_output_contract_ok"] is True
        assert p_arch_08["id_renaming_invariant_ok"] is False
    st_bad, _ = classify_public_container_verification_state(
        benchmark="Hegel-OOD-Holdout-12",
        ended_by_finish=True,
        contract_accepted=True,
        public_metrics=p_bad_08,
        skill_activation_count=0,
        nogood_count=1,
    )
    assert st_bad == "REFUTED"

    # 2. `ho05` Vertex Permutation Equivariance:
    # Shortcut solver returns public `gt_adj` ignoring column permutation `X[:, perm]`!
    ho05 = tasks["ho05__latent_confounded_graphical_precision"]
    shortcut_ho05 = """import numpy as np
def recover_sparse_precision_support(X_samples: np.ndarray) -> dict:
    z = np.load('/holdout_task/public_graph.npz')
    return {'adjacency': np.asarray(z['gt_adj'], bool), 'sparse_precision': np.asarray(z['gt_S'], float)}
"""
    p_bad_05 = probe_public_cohort_contract_and_invariance(task=ho05, candidate_py_code=shortcut_ho05)
    assert p_bad_05["public_output_contract_ok"] is True
    assert p_bad_05["id_renaming_invariant_ok"] is False
    assert "vertex_permutation_not_equivariant" in p_bad_05["renaming_discrepancies"]


def test_m8_3_three_tier_escalation_ladder_in_main_engine(tmp_path: Path) -> None:
    """Verify `run_m6_9_a_two_branch_episode` executes:
      Case A: Tier-1 `LocalContainerRepair` success -> skips `Thesis` (`1.1x`), persists guard, grades raw pre-repair honestly!
      Case B: Tier-1 `LocalContainerRepair` rejected -> retains failed repair tokens & escalates to Tier-2 `Thesis` (`2.0x`)!
    """
    m74_dir = REPO_ROOT / "docs/evidence/m7_4_exp1_frozen_20task_and_ablations_20261124"
    planner = M8OfflineContainerPlanner.fit_from_historical_records(
        load_historical_suite_episode_records(m74_dir),
        suite_ids=[m74_dir.name],
    )
    ho_waves = holdout_tasks_as_frontier_waves()
    ho12_ftask = next(t for t in ho_waves["wave_2"] if t.task_id == "ho12__multicopy_plasmid_cohort_consensus")
    ho12_raw_task = holdout_tasks_by_id()["ho12__multicopy_plasmid_cohort_consensus"]

    broken_ho12_py = (
        REPO_ROOT
        / "docs/evidence/m8_1_offline_replay_and_live_3task_slice_20261125"
        / "part_b_live_3task_16thread_slice/branches"
        / "wave_2__gpt-6.2-sol__ho12__multicopy_plasmid_cohort_consensus"
        / "branch_2_antithesis_wormhole/04_submitted_artifacts_snapshot/plasmid_audit.py"
    ).read_text(encoding="utf-8")
    good_ho12_py = (
        REPO_ROOT
        / "docs/evidence/m8_2_cold_restart_autonomous_reuse_and_repair_20261125"
        / "checkpoints/repaired_plasmid_audit_p2.py"
    ).read_text(encoding="utf-8")

    # Standalone escalation check on `choose_and_execute_container_mechanism_action` (`budget=1.5`, failed repair -> `SpawnDialecticalChallenger`)
    class _BadRepairClient:
        def call_structured(self, **_: Any) -> dict[str, Any]:
            return {
                "ok": True,
                "prompt_tokens": 88,
                "cached_tokens": 0,
                "completion_tokens": 44,
                "wall_ms": 120.0,
                "parsed": {
                    "repaired_python_code": broken_ho12_py,
                    "invariance_guard_name": "StillBroken",
                    "invariance_principle": "Did not fix null field.",
                    "defect_diagnosis": "Still returns None.",
                },
            }

    g_esc = PersistentMechanismGraph()
    esc_out = choose_and_execute_container_mechanism_action(
        task=ho12_raw_task,
        primary_candidate_py=broken_ho12_py,
        graph=g_esc,
        codex_client=_BadRepairClient(),
        budget_remaining=1.5,
    )
    assert esc_out["selected_action"] == "SpawnDialecticalChallenger"
    assert esc_out["decision_record"]["total_prompt_tokens"] == 88
    assert esc_out["decision_record"]["total_completion_tokens"] == 44

    # Full `run_m6_9_a_two_branch_episode` integration: Case A (Tier-1 `LocalContainerRepair` succeeds -> skips `Thesis`!)
    forest, archive, _ = build_blank_axiomatic_philosophical_forest()
    hippo = HippoRAGWormholeIndex(forest)
    graph_live = PersistentMechanismGraph()
    ckpt_fp = tmp_path / "m83_engine_ckpt.json"

    def _fake_run_frontier_episode(task_obj: Any, arm_name: str, *, export_artifacts_tar_path: Path | None = None, episode_out_dir: Path, **_: Any) -> dict[str, Any]:
        episode_out_dir.mkdir(parents=True, exist_ok=True)
        py_src = broken_ho12_py if "Antithesis" in arm_name else good_ho12_py
        cand_fp = episode_out_dir / "plasmid_audit.py"
        cand_fp.write_text(py_src, encoding="utf-8")
        if export_artifacts_tar_path is not None:
            export_artifacts_tar_path.parent.mkdir(parents=True, exist_ok=True)
            with tarfile.open(export_artifacts_tar_path, "w") as tf:
                tf.add(cand_fp, arcname="holdout_task/plasmid_audit.py")
        import hashlib

        sha = hashlib.sha256(py_src.encode("utf-8")).hexdigest()
        s = {
            "task_id": task_obj.task_id,
            "ended_by_finish": True,
            "turns_used": 6,
            "max_turns": 24,
            "final_candidate_sha256": sha,
            "prompt_tokens": 500,
            "cached_tokens": 0,
            "completion_tokens": 120,
            "wall_ms": 1000.0,
            "nogood_count": 0,
            "nogoods_recorded": [],
            "skill_activation_count": 0,
            "staged_wormhole_modules": [],
            "recalled_node_ids": [],
            "terminal_record_sha256": "f" * 64,
        }
        (episode_out_dir / "00_episode_summary.json").write_text(json.dumps(s), encoding="utf-8")
        return s

    class _GoodRepairCodexClient:
        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        def call_structured(self, **_: Any) -> dict[str, Any]:
            return {
                "ok": True,
                "prompt_tokens": 310,
                "cached_tokens": 0,
                "completion_tokens": 95,
                "wall_ms": 250.0,
                "parsed": {
                    "repaired_python_code": good_ho12_py,
                    "invariance_guard_name": "CohortStructuralDepthGuard",
                    "invariance_principle": "Compute chromosomal depth from non-circular contigs and align 36bp motifs on both strands.",
                    "defect_diagnosis": "Fixed CHR_SEG_ prefix crash and populated amino_acid_substitution.",
                },
            }

    with (
        patch("hegel_machine.cognition.m6_9_a_dialectical_synthesis_engine.m6f.run_frontier_episode", side_effect=_fake_run_frontier_episode),
        patch("hegel_machine.cognition.m6_9_a_dialectical_synthesis_engine.m6f.FrontierCodexClient", _GoodRepairCodexClient),
    ):
        ep_sum = run_m6_9_a_two_branch_episode(
            task=ho12_ftask,
            model="gpt-6.2-sol",
            codex_info={"auth_mode": "chatgpt", "provider_backend": "openai_chatgpt_subscription_codex_cli"},
            forest=forest,
            forest_archive=archive,
            hippo_idx=hippo,
            episode_out_dir=tmp_path / "ep_case_a",
            branches_work_root=tmp_path / "branches_a",
            max_turns_per_branch=24,
            selective_second_branch=True,
            m8_container_planner=planner,
            enable_m8_local_repair=True,
            m8_mechanism_graph=graph_live,
            m8_graph_checkpoint_path=ckpt_fp,
        )

    assert ep_sum["strict_pass"] is True
    assert ep_sum["postlock_branch_outcomes"]["thesis_spawned"] is False
    # Raw pre-repair Antithesis graded honestly (`False`), while repaired episode passed (`True`)!
    assert ep_sum["postlock_branch_outcomes"]["antithesis_wormhole_strict_pass"] is False
    assert ep_sum["declared_budget_multiplier"] == "selective_1p1x_local_container_repair"
    assert ep_sum["turns_used"] == 7  # 6 primary turns + 1 local repair turn (saved full 24-turn Thesis branch!)
    assert ep_sum["prompt_tokens"] == 810  # 500 primary + 310 local repair
    assert ckpt_fp.is_file()
    reloaded = PersistentMechanismGraph.load_checkpoint(ckpt_fp)
    assert any(m.role == "container_invariance_guard" for m in reloaded.mechanisms.values())
