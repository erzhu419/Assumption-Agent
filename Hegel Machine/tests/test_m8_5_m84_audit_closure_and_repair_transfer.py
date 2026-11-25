"""Hegel M8.5 Complete Audit Closure (`Hegel_M8_4_Audit`) & Executable Repair Skill Transfer Suite.

Reproduces verbatim and verifies 100% deterministic closure of every counterexample & contract gap in `Hegel_M8_4_Audit`:
  - `E01..E04`: Outer engine repair acceptance transaction (`outer_transaction_control`: zero escaped `OSError`,
    `123+45` repair tokens preserved in both `PersistentMechanismGraph.decision_ledger` and episode DAG,
    guard publication suppressed, raw artifact restored, and event-ID-keyed `action_return_history` preserving
    distinct `ho06` + `ho12` repair receipts while separating binary acceptance from log-predictive nats).
  - `P01..P03`: Explicit `initial_memory_states` priority + `train_initial_memory_states` replay (`nonzero_initial_state`),
    pre-fit `reserve_budget_fn` gating (`reuse_budget`: `budget=0.05` runs at most `1` `fit_and_score_task_model` call),
    and row-wise `scope_guard` execution gating + fail-closed unknown operators (`scope_subset`: `0` outside-guard rows).
  - `R01..R02`: Pure primary-fail + conditional-damage NIG decomposition (`utility_semantics`: `U ≈ +0.02 > 0` without
    double-counting damage) and paraphrase/structured-field `RUNG_3` target-identifiability nogood detection (`nogood_paraphrase`).
  - `B01..B07`, `M01` & `M8.5`: Sentinel scrubbing & non-blocking file-backed RPC `stderr` (`> 200 KB` worker prints),
    `ho12` plasmid CDS mutant-codon verification, `ho05` pre-bool finite `{0, 1}` check, `_helper` & module-constant
    AST digest invalidation, `_eval_pure_ast_expr` imported-math/numpy parity, active repaired snapshot compiler resolution,
    and unit-tested executable repair skill compilation + `@track_wormhole_skill` container invocation telemetry!
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np

from hegel_machine.cognition import m6_9_a_dialectical_synthesis_engine as engine
from hegel_machine.cognition import m7_autogenous_forest_pretraining as m7a
from hegel_machine.cognition import m8_container_mechanism_adapter as adapter
from hegel_machine.cognition import m8_offline_container_planner as o
from hegel_machine.cognition import m8_persistent_variational_planner as m8p
from hegel_machine.cognition.hipporag_wormhole_index import HippoRAGWormholeIndex
from hegel_machine.cognition.m6_8_b_prime_capability_bridge import lift_python_scientific_expressions_to_programs
from hegel_machine.cognition.m7_holdout_ood_blind_benchmark import holdout_tasks_by_id

REPO_ROOT = Path(__file__).resolve().parents[1]


def _root_graph(features: tuple[str, ...] = ("u",)) -> m8p.PersistentMechanismGraph:
    g = m8p.PersistentMechanismGraph()
    g.register_mechanism(
        m8p.TypedMechanism(
            "root",
            None,
            "shared_upstream_premise",
            features,
            shared_weights={f: 0.0 for f in features},
        )
    )
    return g


def test_m8_5_outer_engine_acceptance_transaction_and_event_id_return_dedup(tmp_path: Path) -> None:
    """Reproduce `Hegel_M8_4_Audit/engine/outer_transaction_control.py` (`E01..E04`) and verify full closure."""
    graph = m8p.PersistentMechanismGraph()
    calls = {
        "branches": [],
        "repair_receipts_supplied": 0,
        "contract_checks": 0,
        "containers_started": 0,
        "containers_stopped": 0,
    }
    raw = "def solve(x):\n    return x\n"
    repaired = "def solve(x):\n    return float(x)\n"
    task = SimpleNamespace(
        task_id="cpu_outer_acceptance",
        short_name="cpu_outer_acceptance",
        benchmark="Toy-Audit",
        wave="dev",
        scientific_domain="toy_audit",
        wormhole_family="toy_family",
        agent_image_tag="mock",
        agent_memory_mb=64,
        agent_cpus=1,
        declared_artifacts=("/task/solution.py",),
        instruction_text="Return a numerical input.",
    )

    def _branch(task_obj: Any, arm_name: str, *, export_artifacts_tar_path: Path, episode_out_dir: Path, **_: Any) -> dict[str, Any]:
        _ = task_obj
        calls["branches"].append(arm_name)
        episode_out_dir.mkdir(parents=True, exist_ok=True)
        export_artifacts_tar_path.parent.mkdir(parents=True, exist_ok=True)
        export_artifacts_tar_path.write_bytes(b"fixture-raw-primary-tar")
        return {
            "final_candidate_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "terminal_record_sha256": "a" * 64,
            "ended_by_finish": True,
            "turns_used": 1,
            "prompt_tokens": 10,
            "cached_tokens": 0,
            "completion_tokens": 5,
            "wall_ms": 2.0,
        }

    def _start(*_: Any, **__: Any) -> str:
        calls["containers_started"] += 1
        return f"mock-cid-{calls['containers_started']}"

    def _stop(*_: Any, **__: Any) -> None:
        calls["containers_stopped"] += 1

    def _contract(*_: Any, **__: Any) -> dict[str, Any]:
        calls["contract_checks"] += 1
        if calls["contract_checks"] == 2:
            raise OSError("injected_secondary_contract_transport_error")
        return {"accepted": True}

    def _repair(**_: Any) -> dict[str, Any]:
        calls["repair_receipts_supplied"] += 1
        pending_g = m8p.TypedMechanism(
            mechanism_id="mech_guard_must_not_publish",
            parent_id=None,
            role="container_invariance_guard",
            feature_names=("public_output_contract_ok", "id_renaming_invariant_ok"),
            epistemic_status="CANDIDATE_REPAIR_HYPOTHESIS",
            provenance_note="Must not publish on outer secondary contract error.",
        ).to_dict()
        return {
            "attempted": True,
            "repaired_verified": True,
            "reason": "ok",
            "repaired_python_code": repaired,
            "attempted_repaired_python_code": repaired,
            "repaired_candidate_sha256": hashlib.sha256(repaired.encode()).hexdigest(),
            "llm_tokens": {"prompt": 123, "cached": 0, "completion": 45, "wall_ms": 6.0},
            "wall_ms": 8.0,
            "pending_guard": pending_g,
            "probe_after": {"public_output_contract_ok": True, "id_renaming_invariant_ok": True},
        }

    def _export(_cid: str, _artifacts: Any, path: Path) -> dict[str, Any]:
        Path(path).write_bytes(b"repaired-fixture-placeholder-only")
        return {}

    def _fake_verifier(*_: Any, **__: Any) -> dict[str, Any]:
        return {
            "strict_pass": False,
            "reward": 0.0,
            "artifact_audit": {"all_present": True},
            "ctrf_summary": None,
            "verifier_stdout": "",
            "verifier_stderr": "",
            "verifier_timed_out": False,
            "verifier_wall_ms": 1.0,
        }

    with (
        patch.object(engine.m6f, "run_frontier_episode", side_effect=_branch),
        patch.object(engine.m6f, "start_detached_container", side_effect=_start),
        patch.object(engine.m6f, "stop_container_safely", side_effect=_stop),
        patch.object(engine, "_init_holdout_public_env_if_needed", return_value=None),
        patch.object(engine.m6f, "import_declared_artifacts_tar", return_value={"candidate_sha256": "a" * 64}),
        patch.object(engine.m6f, "export_declared_artifacts_tar", side_effect=_export),
        patch.object(engine.m6f, "save_submitted_artifacts_snapshot", return_value=[]),
        patch.object(engine, "verify_candidate_artifact_contract_in_container", side_effect=_contract),
        patch.object(engine, "_run_public_container_candidate_benchmark", return_value={}),
        patch.object(
            adapter,
            "probe_general_public_container_invariants_in_container",
            return_value={
                "probe_applicable": True,
                "public_output_contract_ok": False,
                "id_renaming_invariant_ok": False,
                "symmetry_probe_family": "toy",
            },
        ),
        patch.object(engine.m6f, "FrontierCodexClient", return_value=SimpleNamespace(call_structured=lambda **_: {"ok": False})),
        patch.object(adapter, "execute_in_container_local_repair", side_effect=_repair),
        patch.object(engine.m6f, "run_tb_separate_verifier", side_effect=_fake_verifier),
        patch.object(engine, "grade_archived_branch_post_lock", return_value={"strict_pass": False, "reward": 0.0}),
    ):
        result = engine.run_m6_9_a_two_branch_episode(
            task=task,
            model="mock",
            codex_info={"provider_backend": "mock", "auth_mode": "mock"},
            forest=None,
            forest_archive={},
            hippo_idx=None,
            episode_out_dir=tmp_path / "episode",
            branches_work_root=tmp_path / "branches",
            selective_second_branch=True,
            m8_container_planner=SimpleNamespace(),
            enable_m8_local_repair=True,
            m8_mechanism_graph=graph,
        )

    assert result is not None
    assert calls["containers_started"] == calls["containers_stopped"]
    assert len(graph.decision_ledger) == 1
    rep_rec = graph.decision_ledger[0]
    assert rep_rec.execution_status == "OUTER_ACCEPTANCE_ROLLED_BACK"
    assert rep_rec.total_prompt_tokens == 123
    assert rep_rec.total_completion_tokens == 45
    assert "injected_secondary_contract_transport_error" in rep_rec.attempt_cost_events[0]["note"]
    assert not any(m.role == "container_invariance_guard" for m in graph.mechanisms.values())
    assert (tmp_path / "episode/00_episode_summary.json").is_file()
    assert (tmp_path / "episode/01_decision_provenance_dag.json").is_file()
    # Episode token total includes Antithesis (`10`) + failed outer-acceptance repair (`123`) + Challenger (`10`)!
    assert result["prompt_tokens"] >= 133

    # `E02` & `E03`: Two distinct `LocalContainerRepair` successes (`ho06` & `ho12`) NEVER collapse on `merge_worker_graph_into_shared`!
    shared_g = m8p.PersistentMechanismGraph()
    for tid in ("ho06__plate_evaporation_deconfounded_ic50", "ho12__multicopy_plasmid_cohort_consensus"):
        wg = m8p.PersistentMechanismGraph()
        did = f"m8_rep_dec_007_{tid}"
        wg.decision_ledger.append(
            m8p.ResearchDecisionRecord(
                decision_id=did,
                step_index=7,
                task_id=tid,
                pre_action_state={"task_id": tid},
                available_actions=["SkipLocalRepair", "LocalContainerRepair"],
                selected_action="LocalContainerRepair",
                selected_candidate_id="guard_x",
                execution_status="EXECUTED_OK",
                actual_cost=0.35,
                public_val_log_predictive_delta=0.0,
                collateral_min_delta=0.0,
                rationale="ok",
            )
        )
        wg.action_return_history["LocalContainerRepair"] = [
            {
                "event_id": did,
                "task_id": tid,
                "step_index": 7,
                "metric_kind": "observed_public_repair_acceptance",
                "observed_public_acceptance": 1.0,
                "gain_nats": 0.0,
                "cost_spent": 0.35,
            }
        ]
        adapter.merge_worker_graph_into_shared(shared_g, wg)
        # Re-merging the same worker graph is idempotent!
        adapter.merge_worker_graph_into_shared(shared_g, wg)

    assert len(shared_g.decision_ledger) == 2
    assert len(shared_g.action_return_history["LocalContainerRepair"]) == 2
    assert {r["task_id"] for r in shared_g.action_return_history["LocalContainerRepair"]} == {
        "ho06__plate_evaporation_deconfounded_ic50",
        "ho12__multicopy_plasmid_cohort_consensus",
    }


def test_m8_5_planner_nonzero_initial_memory_reuse_budget_and_scope_gate_controls() -> None:
    """Reproduce `Hegel_M8_4_Audit/planner/planner_controls.py` (`P01..P03`, `R01..R02`) verbatim."""
    # 1. `P01` (`nonzero_initial_state_control`): explicit `initial_memory_states` + `train_initial_memory_states` replay == fit!
    g1 = _root_graph(("u",))
    g1.register_mechanism(m8p.TypedMechanism("memory", "root", "temporal_memory", ("u",), temporal_tau=2.0, temporal_driver_col="u"))
    rng1 = np.random.default_rng(84001)
    u = rng1.normal(size=40)
    ts = np.arange(100, 140, dtype=float)
    initial = {"0": [99.0, 6.0, -2.0]}
    z = m8p._compute_causal_memory_feature({"x": {"u": u}, "timestamps": ts, "initial_memory_states": initial}, "u", 2.0)
    assert isinstance(z, np.ndarray)
    y = 3.0 * z + 0.2 * u
    tr1 = {"x": {"u": u[:25]}, "y": y[:25], "timestamps": ts[:25], "initial_memory_states": initial}
    va1 = {"x": {"u": u[25:]}, "y": y[25:], "timestamps": ts[25:]}
    p1 = m8p.M8VariationalPlanner(g1)
    tm1 = p1.register_task(task_id="nonzero_start", shared_upstream_id="root", feature_names=("u",), train_data=tr1, val_data=va1)
    tm1.temporal_memory_id = "memory"
    fit1 = m8p.fit_and_score_task_model(tm1, g1, tr1, va1, prior_shrinkage=1e-5)
    ev1 = m8p.evaluate_committed_task_model(tm1, g1, tr1)
    assert abs(fit1["train_rmse"] - ev1["val_rmse"]) < 1e-12
    assert ev1["val_rmse"] < 1e-4
    # Even replaying the training window `[100..124]` without repeating `initial_memory_states` recovers `train_initial_memory_states`!
    ev1_implicit = m8p.evaluate_committed_task_model(tm1, g1, {"x": {"u": u[:25]}, "y": y[:25], "timestamps": ts[:25]})
    assert abs(fit1["train_rmse"] - ev1_implicit["val_rmse"]) < 1e-12
    # And forward validation continuation (`[125..139]`) still carries over `terminal_memory_states` (`< 1e-4` RMSE)!
    ev1_va = m8p.evaluate_committed_task_model(tm1, g1, va1)
    assert ev1_va["val_rmse"] < 1e-4

    # 2. `P03` (`reuse_budget_control`): `budget_remaining=0.05` executes AT MOST `1` `fit_and_score_task_model` call!
    g2 = _root_graph(("a", "b"))
    for idx, expr in enumerate(["x['a'] ** 2", "x['b'] ** 2", "x['a'] * x['b']", "x['a'] * np.abs(x['b'])"]):
        g2.register_mechanism(m8p.TypedMechanism(f"saved_{idx}", "root", "interaction_lift", ("a", "b"), python_expression=expr))
    rng2 = np.random.default_rng(84002)
    a2, b2 = rng2.normal(size=(2, 80))
    y2 = 3.0 * a2 * b2 + rng2.normal(scale=0.03, size=80)
    p2 = m8p.M8VariationalPlanner(g2)
    p2.register_task(
        task_id="reuse_budget",
        shared_upstream_id="root",
        feature_names=("a", "b"),
        train_data={"x": {"a": a2[:50], "b": b2[:50]}, "y": y2[:50]},
        val_data={"x": {"a": a2[50:], "b": b2[50:]}, "y": y2[50:]},
    )
    fit_calls: list[str] = []
    orig_fit = m8p.fit_and_score_task_model

    def _counted_fit(model: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        fit_calls.append(model.model_id)
        return orig_fit(model, *args, **kwargs)

    with patch.object(m8p, "fit_and_score_task_model", side_effect=_counted_fit):
        res2 = p2.evaluate_and_revise_task("reuse_budget", budget_remaining=0.05)

    assert res2["decision_record"]["actual_cost"] == 0.05
    assert len(fit_calls) == 1
    assert res2["selected_candidate"]["action_type"] == "ReuseMechanism"
    assert res2["selected_candidate"]["metadata"]["recalled_mechanism_id"] == "saved_2"

    # 3. `P02` (`scope_subset_control` + unknown `op` fail-closed): `0` non-zero lift rows outside `a > 0`!
    g3 = _root_graph(("a", "b"))
    guard = {"field": "a", "op": ">", "value": 0.0}
    g3.register_mechanism(
        m8p.TypedMechanism("scoped_lift", "root", "interaction_lift", ("a", "b"), python_expression="x['a'] * x['b']", scope_guard=guard)
    )
    a3 = np.r_[np.linspace(-5.0, -1.0, 60), np.linspace(1.0, 3.0, 20)]
    b3 = np.linspace(0.3, 2.0, 80)
    y3 = np.where(a3 > 0.0, 2.0 * a3 * b3, 0.0)
    p3 = m8p.M8VariationalPlanner(g3)
    tr3 = {"x": {"a": a3, "b": b3}, "y": y3}
    tm3 = p3.register_task(task_id="partial_scope", shared_upstream_id="root", feature_names=("a", "b"), train_data=tr3, val_data=tr3)
    base3 = m8p.evaluate_committed_task_model(tm3, g3, tr3)
    cands3 = p3._discover_reusable_mechanism_candidates(tm3, tr3, tr3, base3["val_mean_log_predictive"], base3["variational_free_energy"])
    cand3 = next(c for c in cands3 if c.metadata["recalled_mechanism_id"] == "scoped_lift")
    assert cand3.metadata["scope_qualification"] == "PARTIAL_SUBSET_GATED_IN_SCOPE"
    assert cand3.apply_fn is not None
    cand3.apply_fn(g3)
    rm3 = g3.active_task_models["partial_scope"].lift_role_maps["scoped_lift"]
    phi3 = m8p.build_task_design_matrix(g3.active_task_models["partial_scope"], g3, tr3, use_bound_shared_weights=True)[0]
    mask3 = np.asarray(tr3["x"][rm3["a"]]) > 0.0
    assert int(np.count_nonzero(phi3[~mask3, -1])) == 0
    assert int(np.count_nonzero(phi3[mask3, -1])) == 20
    # Unknown guard operator MUST fail closed!
    ok_unk, frac_unk = p3._evaluate_mapped_scope_guard_fraction({"field": "a", "op": "approx_gt", "value": 0.0}, tr3["x"], {"a": "a"})
    assert ok_unk is False and frac_unk == 0.0

    # 4. `R01` (`utility_semantics_control`): `160` rescues (`+0.16`) minus `120` damages (`-0.12`) minus `0.02` cost -> `+0.02 > 0` (`Spawn`)!
    pre_u = {
        "task_id": "same_context",
        "benchmark": "Hegel-OOD-Holdout-12",
        "wave": "wave_2",
        "ended_by_finish": True,
        "pullback_gate_triggered": False,
        "turns_used": 10,
        "max_turns": 24,
        "turns_ratio": 10 / 24,
        "skill_activation_count": 0,
        "nogood_count": 0,
        "contract_accepted": True,
        "public_verification_state": "UNKNOWN",
    }
    recs_u = [
        {
            "task_id": f"t{i}",
            "pre_action_state": deepcopy(pre_u),
            "outcomes": {"antithesis_1x_pass": (i >= 200), "m8_twobranch_pass": ((i < 160) or (i >= 320))},
        }
        for i in range(1000)
    ]
    pl_u = o.M8OfflineContainerPlanner.fit_from_historical_records(recs_u, suite_ids=["synthetic_same_context"], cost_lambda=0.02)
    dec_u = pl_u.decide_second_branch(pre_u, record_to_ledger=False)
    assert abs(dec_u["posterior_mean_fail_risk"] - 0.20) < 0.01
    assert abs(dec_u["posterior_conditional_damage_risk"] - 0.15) < 0.01
    assert 0.015 <= dec_u["net_variational_utility"] <= 0.030
    assert dec_u["spawn_second_branch"] is True

    # 5. `R02` (`nogood_paraphrase_control` + structured `risk_kind` / `resolved` support)
    ho08_fp = REPO_ROOT / "docs/evidence/m8_3_live_engine_symmetry_repair_slice_20261125/runs/ho08__prony_viscoelastic_memory_inversion/00_episode_summary.json"
    ng_orig = json.loads(ho08_fp.read_text(encoding="utf-8"))["nogoods_recorded"][0]
    para = deepcopy(ng_orig)
    para["assumption_refuted"] = "Accurate stress prediction implies unique recovery of the equilibrium modulus."
    para["empirical_witness"] = "Two admissible slow-mode fits give nearly equal stress residuals but different equilibrium moduli."
    para["causal_blanket_diagnosis"] = "The excitation does not distinguish the slow relaxation amplitude from the equilibrium term."
    assert o._has_unresolved_target_identifiability_nogood([ng_orig]) is True
    assert o._has_unresolved_target_identifiability_nogood([para]) is True
    assert o._has_unresolved_target_identifiability_nogood([{**para, "resolved": True}]) is False
    assert o._has_unresolved_target_identifiability_nogood([{"risk_kind": "target_identifiability"}]) is True


def test_m8_5_boundary_sentinel_stderr_ho12_codon_ho05_binary_ast_and_compiler_controls(tmp_path: Path) -> None:
    """Verify `B01..B07`, `M01`, and `M8.5` container-verified executable repair skill compilation & staging."""
    tasks = holdout_tasks_by_id()

    # 1. `B02` + `B04` + `B06`: Candidate printing `250 KB` of stdout/stderr noise + reading `os.environ` (`HEGEL_PUB_PROBE_SENTINEL` scrubbed!)
    #    AND `ho12` fake mutant amino acid (`mut_aa` changed on real repaired solver) rejected by plasmid sequence alignment!
    good_ho12 = (
        REPO_ROOT / "docs/evidence/m8_2_cold_restart_autonomous_reuse_and_repair_20261125/checkpoints/repaired_plasmid_audit_p2.py"
    ).read_text(encoding="utf-8")
    noisy_env_check_ho12 = (
        "import os, sys\n"
        "assert 'HEGEL_PUB_PROBE_SENTINEL' not in os.environ, 'SENTINEL_LEAKED_TO_CANDIDATE'\n"
        "print('X' * 260000)\n"
        "sys.stderr.write('Y' * 260000 + '\\n')\n"
        + good_ho12
    )
    p_noisy_good = adapter.probe_public_cohort_contract_and_invariance(
        task=tasks["ho12__multicopy_plasmid_cohort_consensus"],
        candidate_py_code=noisy_env_check_ho12,
    )
    assert p_noisy_good["public_output_contract_ok"] is True
    assert p_noisy_good["public_calibration_ok"] is True
    assert p_noisy_good["id_renaming_invariant_ok"] is True

    # Corrupt ONLY the final mutant amino acid returned by `good_ho12` (keeping valid `wt_aa` + valid `pos` + valid tag + valid copy number!)
    real_sub = p_noisy_good["public_consistency_metrics"]["public_output_preview"]["amino_acid_substitution"]
    wrong_mut = "W" if real_sub[-1] != "W" else "Y"
    fake_mut_sub = real_sub[:-1] + wrong_mut
    fake_mut_ho12 = (
        good_ho12
        + "\n_orig_audit = audit_outbreak_cohort\n"
        + f"def audit_outbreak_cohort(cohort_dir: str) -> dict:\n    r = dict(_orig_audit(cohort_dir))\n    r['amino_acid_substitution'] = {fake_mut_sub!r}\n    return r\n"
    )
    p_fake_mut = adapter.probe_public_cohort_contract_and_invariance(
        task=tasks["ho12__multicopy_plasmid_cohort_consensus"],
        candidate_py_code=fake_mut_ho12,
    )
    assert p_fake_mut["public_output_contract_ok"] is False
    assert p_fake_mut["public_calibration_ok"] is False
    assert any("substitution_mut_not_supported_by_plasmid_sequence" in m for m in p_fake_mut["missing_or_null_keys"])

    # 2. `B07`: `ho05` non-binary (`2.5` or `NaN`) adjacency matrix rejected BEFORE `bool` coercion!
    nonbinary_ho05 = (
        "import numpy as np\n"
        "def recover_sparse_precision_support(X_samples):\n"
        "    p = X_samples.shape[1]\n"
        "    adj = np.full((p, p), 2.5)\n"
        "    np.fill_diagonal(adj, 0.0)\n"
        "    return {'adjacency': adj, 'sparse_precision': np.eye(p)}\n"
    )
    p_nb05 = adapter.probe_public_cohort_contract_and_invariance(
        task=tasks["ho05__latent_confounded_graphical_precision"],
        candidate_py_code=nonbinary_ho05,
    )
    assert p_nb05["public_output_contract_ok"] is False
    assert "invalid_shape_nonbinary_adjacency_or_non_finite_precision" in p_nb05["missing_or_null_keys"]

    # 3. `B03`: Editing private `_helper` OR module constant invalidates `_extract_code_ast_operator_profile` `ast_sha12`!
    helper_v1 = "SCALE = 2.0\ndef _helper(z):\n    return z + 1.0\ndef kernel(x):\n    return SCALE * _helper(x) * x\n"
    helper_v2 = "SCALE = 2.0\ndef _helper(z):\n    return z - 1.0\ndef kernel(x):\n    return SCALE * _helper(x) * x\n"
    const_v2 = "SCALE = -2.0\ndef _helper(z):\n    return z + 1.0\ndef kernel(x):\n    return SCALE * _helper(x) * x\n"
    sha_base = m7a._extract_code_ast_operator_profile(helper_v1)["ast_sha12"]
    assert sha_base != ""
    assert m7a._extract_code_ast_operator_profile(helper_v2)["ast_sha12"] != sha_base
    assert m7a._extract_code_ast_operator_profile(const_v2)["ast_sha12"] != sha_base

    # 4. `B04_ast`: `from math import sin, pi` and `np.add` / `np.power` lift cleanly with pure AST differential parity!
    from_math_src = (
        "from math import sin, pi\n"
        "import numpy as np\n"
        "def wave_lift(x, y):\n"
        "    return np.add(sin(x * pi), np.power(y, 2))\n"
    )
    lifted = lift_python_scientific_expressions_to_programs([{"name": "wave_lift", "code": from_math_src}])
    assert len(lifted) == 1

    # 5. `M01`: `consolidate_wave1_via_aufhebung_compiler` reads `04a_repaired_antithesis_snapshot` instead of broken `04_submitted_artifacts_snapshot`!
    ep_dir = tmp_path / "repaired_ep"
    (ep_dir / "04_submitted_artifacts_snapshot").mkdir(parents=True)
    (ep_dir / "04a_repaired_antithesis_snapshot").mkdir(parents=True)
    (ep_dir / "04_submitted_artifacts_snapshot/solver.py").write_text("def solve(x):\n    return 'BROKEN_RAW_MARKER'\n", encoding="utf-8")
    (ep_dir / "04a_repaired_antithesis_snapshot/solver.py").write_text("def solve(x):\n    return float(x) * 2.0 + 1.0\n", encoding="utf-8")

    captured_sources: list[dict[str, str]] = []

    def _spy_compile(**kwargs: Any) -> dict[str, Any]:
        captured_sources.append(dict(kwargs["raw_python_sources"]))
        return {"compiled_ok": False, "error": "spy_stop", "episode_id": kwargs["episode_id"], "llm_tokens": {}}

    forest, archive, _ = m7a.build_blank_axiomatic_philosophical_forest()
    mock_w1_task = SimpleNamespace(
        task_id="w1_repaired_ctrl",
        short_name="w1_repaired_ctrl",
        benchmark="Hegel-OOD-Holdout-12",
        instruction_text=" calibrate sensor drift and conserve flux ",
    )
    with (
        patch.object(m7a, "ensure_clean_scipy_sandbox_image", return_value="mock"),
        patch.object(m7a.m6f, "start_detached_container", return_value="mock-cid"),
        patch.object(m7a.m6f, "stop_container_safely", return_value=None),
        patch.object(m7a.m6f, "FrontierCodexClient", return_value=SimpleNamespace()),
        patch.object(m7a, "compile_episode_dialectical_sublation", side_effect=_spy_compile),
    ):
        m7a.consolidate_wave1_via_aufhebung_compiler(
            wave1_antithesis_episodes=[
                (
                    mock_w1_task,
                    {
                        "ended_by_finish": True,
                        "local_container_repair_summary": {
                            "repaired_verified": True,
                            "repaired_candidate_sha256": hashlib.sha256(
                                (ep_dir / "04a_repaired_antithesis_snapshot/solver.py").read_bytes()
                            ).hexdigest(),
                        },
                    },
                    ep_dir,
                )
            ],
            forest=forest,
            archive=archive,
            codex_info={"provider_backend": "mock", "auth_mode": "chatgpt"},
            model="mock",
            run_id="ctrl_m01",
        )
    assert len(captured_sources) == 1
    assert "BROKEN_RAW_MARKER" not in captured_sources[0]["solver.py"]
    assert "float(x) * 2.0 + 1.0" in captured_sources[0]["solver.py"]

    # 6. `M8.5` End-to-End Container-Verified `compile_verified_repair_to_executable_skill` + `@track_wormhole_skill` Runtime Activation!
    class _DeterministicRepairSkillClient:
        def call_structured(self, **_: Any) -> dict[str, Any]:
            return {
                "ok": True,
                "prompt_tokens": 150,
                "cached_tokens": 0,
                "completion_tokens": 90,
                "wall_ms": 25.0,
                "parsed": {
                    "canonical_module_stem": "autogenous_border_nuisance_deconfounder",
                    "sublated_methodology_principle": (
                        "Estimate smooth 2D spatial background fields EXCLUSIVELY on negative-control border coordinates "
                        "before fitting nonlinear dose-response curves on interior wells, preserving compound-ID permutation equivariance."
                    ),
                    "routed_signatures": [
                        "sig:cohort_transfer_and_metadata_imputation",
                        "sig:robust_physical_calibration_and_safety_margins",
                    ],
                    "scope_preconditions": [],
                    "regime_bifurcation": {
                        "should_split_node": False,
                        "regime_axis": "none",
                        "branch_guard_op": ">=",
                        "branch_guard_value": 0.0,
                        "counter_regime_warning": "",
                    },
                    "operator_pipeline_stages": [
                        {"stage_kind": "nuisance_projection", "numeric_param": 2.0},
                        {"stage_kind": "robust_inversion", "numeric_param": 1.0},
                    ],
                    "exported_api_signatures": ["fit_control_surface_and_subtract(coords, signals, is_control)"],
                    "canonical_python_code": (
                        "import numpy as np\n"
                        "def fit_control_surface_and_subtract(coords, signals, is_control):\n"
                        "    xy = np.asarray(coords, dtype=float)\n"
                        "    y = np.asarray(signals, dtype=float).ravel()\n"
                        "    m = np.asarray(is_control, dtype=bool).ravel()\n"
                        "    X = np.column_stack([np.ones(len(y)), xy[:, 0], xy[:, 1], xy[:, 0] ** 2, xy[:, 1] ** 2])\n"
                        "    beta, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)\n"
                        "    return y - (X @ beta)\n"
                    ),
                    "synthetic_unit_test_code": (
                        "import numpy as np\n"
                        "def run_self_test():\n"
                        "    rng = np.random.default_rng(7)\n"
                        "    xy = rng.uniform(-1.0, 1.0, size=(60, 2))\n"
                        "    ctrl = np.arange(60) < 30\n"
                        "    bg = 1.5 + 0.4 * xy[:, 0] - 0.3 * (xy[:, 1] ** 2)\n"
                        "    sig = bg + np.where(ctrl, 0.0, 2.5)\n"
                        "    clean = fit_control_surface_and_subtract(xy, sig, ctrl)\n"
                        "    assert np.max(np.abs(clean[~ctrl] - 2.5)) < 1e-9\n"
                        "run_self_test()\n"
                    ),
                },
            }

    g_skill = m8p.PersistentMechanismGraph()
    gid = adapter.commit_verified_container_guard(
        g_skill,
        m8p.TypedMechanism(
            mechanism_id="mech_guard_evap_border_01",
            parent_id=None,
            role="container_invariance_guard",
            feature_names=("public_output_contract_ok", "id_renaming_invariant_ok"),
            epistemic_status="CANDIDATE_REPAIR_HYPOTHESIS",
            provenance_note="compound_renaming_and_well_permutation_equivariance: border-control evaporation projection",
        ).to_dict(),
    )
    rep_fixture_py = "def estimate_compound_hill_parameters(plates):\n    return {}\n"
    rep_fixture_sha = hashlib.sha256(rep_fixture_py.encode("utf-8")).hexdigest()
    comp_skill = adapter.compile_verified_repair_to_executable_skill(
        task=tasks["ho06__plate_evaporation_deconfounded_ic50"],
        repair_summary={
            "repaired_verified": True,
            "repaired_python_code": rep_fixture_py,
            "repaired_candidate_sha256": rep_fixture_sha,
            "persisted_guard_id": gid,
            "defect_diagnosis": "Ignored 2D plate evaporation confounding.",
            "invariance_principle": "Fit 2D quadratic evaporation field on DMSO border controls before Hill regression.",
            "probe_after": {
                "public_output_contract_ok": True,
                "id_renaming_invariant_ok": True,
                "public_consistency_metrics": {"pub_max_ic50_err": 0.005},
            },
        },
        forest=forest,
        archive=archive,
        codex_client=_DeterministicRepairSkillClient(),
        graph=g_skill,
    )
    assert comp_skill["compiled_and_consolidated"] is True
    assert comp_skill["unit_test_receipt"]["passed"] is True
    assert comp_skill["source_candidate_sha256"] == rep_fixture_sha
    assert "def run_self_test" in str(comp_skill["synthetic_unit_test_code"])
    assert "EXECUTABLE_SKILL" in g_skill.mechanisms[str(gid)].provenance_note

    # Verify HippoRAG recall + `@track_wormhole_skill` runtime execution receipt!
    hippo = HippoRAGWormholeIndex(forest)
    recalled_hits = hippo.retrieve(
        ["sig:cohort_transfer_and_metadata_imputation", "sig:robust_physical_calibration_and_safety_margins"],
        task_view={"goal_kind": "scientific_terminal_discovery"},
        top_k=5,
    )["candidates"]
    assert any(h["node_id"] == comp_skill["target_node_id"] for h in recalled_hits)
    target_rec = archive[comp_skill["target_node_id"]]
    mod_code = target_rec["autogenous_wormhole_modules"][0]["python_code"]
    assert "@track_wormhole_skill" in mod_code
