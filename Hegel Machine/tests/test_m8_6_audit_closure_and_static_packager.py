"""Hegel M8.5.1 + M8.6 Regression & Static Dependency Closure Packager Verification Suite.

Covers:
  1. `M85-CACHE` (`partial_role_search_control.py`): budget-truncated role-map search never poisons untried role maps.
  2. `M85-SCOPE` & `M85-TIME` (`planner_controls.py`): partial-domain `np.log(x['a'])` under `a > 0` guard (even with
     `np.errstate(invalid='raise', divide='raise')`) and exact interior temporal slice boundary reconstruction.
  3. `M85-TXN` & `M85-EVENT` (`transaction_and_event_controls.py`): atomic `graph` + `forest` rollback when `save_checkpoint`
     raises after guard registration, plus distinct `run_id` event identity across independent same-task runs.
  4. `M85-SOURCE`, `M85-BIND`, `M85-TESTSRC`, `M85-ROUTER`: strict repaired-snapshot SHA-256 provenance, unit-test source
     retention, UCB utility unit labeling, and legacy untyped `action_return_history` tagging.
  5. `M86-CLOSURE`: all 5 mechanical static dependency closure controls (module constants + import aliases + lexical
     collision immunity, `@track_wormhole_skill` stripping, unexecuted conditional branch helpers without truncation,
     fail-closed dynamic/wildcard/decorator rejection, and real Docker container clean-room execution after deleting
     `/tmp/hegel_wormhole_skills` with `VerificationReceipt` SHA-256 identity equality).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from hegel_machine.cognition import (
    m6_9_a_dialectical_synthesis_engine as engine,
    m7_autogenous_forest_pretraining as m7a,
    m8_container_mechanism_adapter as adapter,
    m8_offline_container_planner as o_planner,
    m8_persistent_variational_planner as m_planner,
)
from hegel_machine.cognition.m6_4_executable_wormhole_library import (
    auto_package_wormhole_imports_in_container,
    find_uninlined_wormhole_imports_in_container,
    package_self_contained_wormhole_dependencies,
)
from hegel_machine.cognition.m6_8_b_prime_capability_bridge import (
    verify_candidate_artifact_contract_in_container,
)
from hegel_machine.cognition.m6_frontier_real_benchmarks import (
    compute_declared_artifacts_digest,
    exec_container_bash,
    start_detached_container,
    stop_container_safely,
)
from hegel_machine.cognition.m7_holdout_ood_blind_benchmark import (
    CLEAN_SCIPY_SANDBOX_IMAGE,
    ensure_clean_scipy_sandbox_image,
)
from hegel_machine.cognition.wormhole_forest import PhilosophicalForest


def _root_graph(features: tuple[str, ...] = ("u",)) -> m_planner.PersistentMechanismGraph:
    g = m_planner.PersistentMechanismGraph()
    g.register_mechanism(
        m_planner.TypedMechanism(
            "root",
            None,
            "shared_upstream_premise",
            features,
            shared_weights={f: 0.0 for f in features},
        )
    )
    return g


def test_m8_5_1_partial_role_search_budget_does_not_poison_untried_role_maps() -> None:
    """Reproduces `Hegel_M8_5_Audit/planner/partial_role_search_control.py` verbatim (`M85-CACHE`)."""

    def _make():
        g = _root_graph(("a", "b"))
        g.register_mechanism(
            m_planner.TypedMechanism("square_role", "root", "interaction_lift", ("s",), python_expression="x['s'] ** 2")
        )
        vals = np.linspace(-2.0, 2.0, 9)
        a, b = [x.ravel() for x in np.meshgrid(vals, vals)]
        tr = {"x": {"a": a, "b": b}, "y": 4.0 * a * a + 3.0 * b * b}
        va = {"x": {"a": a, "b": b}, "y": 4.0 * np.mean(a * a) + 3.0 * b * b}
        p = m_planner.M8VariationalPlanner(g)
        tm = p.register_task(
            task_id="partial_role_budget",
            shared_upstream_id="root",
            feature_names=("a", "b"),
            train_data=tr,
            val_data=va,
        )
        base = m_planner.evaluate_committed_task_model(tm, g, va)
        return g, p, tm, tr, va, base

    g, p, tm, tr, va, base = _make()
    remaining = [0.05]
    debited: list[str] = []

    def _reserve(act: str, cost: float, note: str) -> bool:
        _ = act
        if remaining[0] + 1e-9 < cost:
            return False
        remaining[0] -= cost
        debited.append(note)
        return True

    limited = p._discover_reusable_mechanism_candidates(
        tm, tr, va, base["val_mean_log_predictive"], base["variational_free_energy"], reserve_budget_fn=_reserve
    )
    assert len(debited) == 1
    assert len(limited) == 1
    assert limited[0].metadata["role_map"] == {"s": "a"}
    assert limited[0].val_log_predictive_gain < 0.0

    # Mechanism-level `partial_role_budget@v1:n81` MUST NOT be present after truncated search!
    neg_set = g.mechanisms["square_role"].inapplicable_task_ids
    assert "partial_role_budget@v1:n81" not in neg_set
    assert any("rm=" in k for k in neg_set)

    # Retrying with budget restored MUST evaluate untried `{'s': 'b'}` and match the complete search!
    retry = p._discover_reusable_mechanism_candidates(
        tm, tr, va, base["val_mean_log_predictive"], base["variational_free_energy"]
    )
    _, p2, tm2, tr2, va2, base2 = _make()
    complete = p2._discover_reusable_mechanism_candidates(
        tm2, tr2, va2, base2["val_mean_log_predictive"], base2["variational_free_energy"]
    )
    assert len(retry) == 1 and len(complete) == 1
    assert retry[0].metadata["role_map"] == {"s": "b"}
    assert abs(retry[0].val_log_predictive_gain - complete[0].val_log_predictive_gain) < 1e-12
    assert retry[0].val_log_predictive_gain > 0.35


def test_m8_5_1_scoped_math_domain_and_interior_temporal_replay() -> None:
    """Reproduces `Hegel_M8_5_Audit/planner/planner_controls.py` (`M85-SCOPE` & `M85-TIME`)."""
    # 1. `M85-SCOPE`: `np.log(x['a'])` under `scope_guard: a > 0` with 60 negative rows and strict `errstate('raise')`
    g_sc = _root_graph(("a",))
    guard = {"field": "a", "op": ">", "value": 0.0}
    g_sc.register_mechanism(
        m_planner.TypedMechanism("positive_log", "root", "interaction_lift", ("a",), python_expression="np.log(x['a'])", scope_guard=guard)
    )
    a = np.r_[np.linspace(-5.0, -1.0, 60), np.linspace(1.0, 4.0, 20)]
    y = np.zeros_like(a)
    y[a > 0] = 3.0 * np.log(a[a > 0])
    ds = {"x": {"a": a}, "y": y}
    p_sc = m_planner.M8VariationalPlanner(g_sc)
    tm_sc = p_sc.register_task(task_id="scoped_math", shared_upstream_id="root", feature_names=("a",), train_data=ds, val_data=ds)
    base_sc = m_planner.evaluate_committed_task_model(tm_sc, g_sc, ds)

    with np.errstate(invalid="raise", divide="raise"):
        cands = p_sc._discover_reusable_mechanism_candidates(
            tm_sc, ds, ds, base_sc["val_mean_log_predictive"], base_sc["variational_free_energy"]
        )
        assert sum(c.metadata.get("recalled_mechanism_id") == "positive_log" for c in cands) == 1
        assert cands[0].val_log_predictive_gain > 1.0
        rev_out = p_sc.evaluate_and_revise_task("scoped_math", budget_remaining=0.5)
        assert rev_out["selected_candidate"]["action_type"] == "ReuseMechanism"
        post_eval = m_planner.evaluate_committed_task_model(g_sc.active_task_models["scoped_math"], g_sc, ds)
        assert post_eval["val_rmse"] < 1e-2

    # 2. `M85-TIME`: Interior temporal window `ts[10:25]` reconstructs exact boundary `[109.0, z[9], u[9]]`
    g_t = _root_graph(("u",))
    g_t.register_mechanism(
        m_planner.TypedMechanism("memory", "root", "temporal_memory", ("u",), temporal_tau=2.0, temporal_driver_col="u")
    )
    rng = np.random.default_rng(84001)
    u = rng.normal(size=40)
    ts = np.arange(100, 140, dtype=float)
    initial = {"0": [99.0, 6.0, -2.0]}
    z = m_planner._compute_causal_memory_feature({"x": {"u": u}, "timestamps": ts, "initial_memory_states": initial}, "u", 2.0)
    assert isinstance(z, np.ndarray)
    y_t = 3.0 * z + 0.2 * u
    tr_t = {"x": {"u": u[:25]}, "y": y_t[:25], "timestamps": ts[:25], "initial_memory_states": initial}
    va_t = {"x": {"u": u[25:]}, "y": y_t[25:], "timestamps": ts[25:]}
    p_t = m_planner.M8VariationalPlanner(g_t)
    tm_t = p_t.register_task(task_id="interior_replay", shared_upstream_id="root", feature_names=("u",), train_data=tr_t, val_data=va_t)
    tm_t.temporal_memory_id = "memory"
    m_planner.fit_and_score_task_model(tm_t, g_t, tr_t, va_t, prior_shrinkage=1e-5, commit_to_model=True)

    k = 10
    piece = {"x": {"u": u[k:25]}, "y": y_t[k:25], "timestamps": ts[k:25]}
    right_seed = {"0": [float(ts[k - 1]), float(z[k - 1]), float(u[k - 1])]}
    inferred = m_planner._resolve_eval_initial_memory_states(tm_t, piece)
    assert inferred == right_seed
    replay = m_planner.evaluate_committed_task_model(tm_t, g_t, piece)
    with_seed = m_planner.evaluate_committed_task_model(tm_t, g_t, {**piece, "initial_memory_states": right_seed})
    assert abs(replay["val_rmse"] - with_seed["val_rmse"]) < 1e-12
    assert replay["val_rmse"] < 1e-5

    # Roundtrip serialization preserves `train_step_memory_states`; missing step history on interior slice raises `ValueError`
    tm_rt = m_planner.TaskModel.from_dict(tm_t.to_dict())
    assert m_planner.evaluate_committed_task_model(tm_rt, g_t, piece)["val_rmse"] < 1e-5
    tm_legacy = m_planner.TaskModel.from_dict({**tm_t.to_dict(), "train_step_memory_states": {}})
    with pytest.raises(ValueError, match="Interior temporal replay"):
        m_planner.evaluate_committed_task_model(tm_legacy, g_t, piece)


def test_m8_5_1_checkpoint_commit_rollback_and_independent_run_event_identity() -> None:
    """Reproduces `Hegel_M8_5_Audit/engine/transaction_and_event_controls.py` verbatim (`M85-TXN` & `M85-EVENT`)."""

    def _run_case(run_id: str, *, checkpoint_fails_after_guard_registration: bool) -> tuple[m_planner.PersistentMechanismGraph, dict[str, Any]]:
        graph = m_planner.PersistentMechanismGraph()
        # Pre-seed an existing mechanism to prove rollback restores prior state without wiping existing graph entries!
        graph.register_mechanism(
            m_planner.TypedMechanism(
                mechanism_id="mech_preexisting_keep",
                parent_id=None,
                role="container_invariance_guard",
                feature_names=("public_output_contract_ok",),
                epistemic_status="SUPPORTED_IN_SCOPE",
            )
        )
        raw = b"raw-fixture"
        repaired = b"repaired-fixture"
        counters = {"branches": 0, "checkpoint_calls": 0}
        task = SimpleNamespace(
            task_id="numeric_same_task",
            short_name="numeric_same_task",
            benchmark="Toy-Audit",
            wave="dev",
            scientific_domain="numeric",
            wormhole_family="toy",
            agent_image_tag="mock",
            agent_memory_mb=64,
            agent_cpus=1,
            declared_artifacts=("/task/solution.py",),
            instruction_text="Return a real input.",
        )
        gid = "mech_guard_numeric_fixture"
        pending = m_planner.TypedMechanism(
            mechanism_id=gid,
            parent_id=None,
            role="container_invariance_guard",
            feature_names=("public_output_contract_ok",),
            epistemic_status="CANDIDATE_REPAIR_HYPOTHESIS",
            provenance_note="Pure numeric audit fixture.",
        ).to_dict()

        def _branch(task_obj, arm_name, *, export_artifacts_tar_path, episode_out_dir, **kwargs):
            _ = (task_obj, arm_name, kwargs)
            counters["branches"] += 1
            episode_out_dir.mkdir(parents=True, exist_ok=True)
            export_artifacts_tar_path.parent.mkdir(parents=True, exist_ok=True)
            export_artifacts_tar_path.write_bytes(raw)
            return {
                "final_candidate_sha256": hashlib.sha256(raw).hexdigest(),
                "terminal_record_sha256": "a" * 64,
                "ended_by_finish": True,
                "turns_used": 1,
                "prompt_tokens": 10,
                "cached_tokens": 0,
                "completion_tokens": 5,
                "wall_ms": 2.0,
            }

        def _repair(**kwargs):
            _ = kwargs
            return {
                "attempted": True,
                "repaired_verified": True,
                "reason": "ok",
                "repaired_python_code": "def f(x):\n    return float(x)\n",
                "attempted_repaired_python_code": "def f(x):\n    return float(x)\n",
                "llm_tokens": {"prompt": 123, "cached": 0, "completion": 45, "wall_ms": 6.0},
                "wall_ms": 8.0,
                "pending_guard": pending,
                "probe_after": {"public_output_contract_ok": True, "id_renaming_invariant_ok": True},
            }

        def _export(cid, artifacts, target):
            _ = (cid, artifacts)
            Path(target).write_bytes(repaired)
            return {}

        actual_save = graph.save_checkpoint

        def _save_checkpoint(target):
            counters["checkpoint_calls"] += 1
            if checkpoint_fails_after_guard_registration and counters["checkpoint_calls"] == 1:
                assert gid in graph.mechanisms
                raise OSError("injected_checkpoint_write_failure_after_registration")
            return actual_save(target)

        verdict = {
            "strict_pass": False,
            "reward": 0.0,
            "artifact_audit": {"all_present": True},
            "ctrf_summary": None,
            "verifier_stdout": "",
            "verifier_stderr": "",
            "verifier_timed_out": False,
            "verifier_wall_ms": 1.0,
        }
        with tempfile.TemporaryDirectory(prefix="m86_txn_") as td:
            temp = Path(td)
            ckpt = temp / "graph.json"
            with (
                patch.object(engine.m6f, "run_frontier_episode", side_effect=_branch),
                patch.object(engine.m6f, "start_detached_container", return_value="mock-container"),
                patch.object(engine.m6f, "stop_container_safely", return_value=None),
                patch.object(engine, "_init_holdout_public_env_if_needed", return_value=None),
                patch.object(engine.m6f, "import_declared_artifacts_tar", return_value={"candidate_sha256": "a" * 64}),
                patch.object(engine.m6f, "export_declared_artifacts_tar", side_effect=_export),
                patch.object(engine.m6f, "save_submitted_artifacts_snapshot", return_value=[]),
                patch.object(engine, "verify_candidate_artifact_contract_in_container", return_value={"accepted": True}),
                patch.object(engine, "_run_public_container_candidate_benchmark", return_value={}),
                patch.object(
                    adapter,
                    "probe_general_public_container_invariants_in_container",
                    return_value={
                        "probe_applicable": True,
                        "public_output_contract_ok": False,
                        "id_renaming_invariant_ok": False,
                        "symmetry_probe_family": "numeric",
                    },
                ),
                patch.object(engine.m6f, "FrontierCodexClient", return_value=SimpleNamespace(call_structured=lambda **kw: {"ok": False})),
                patch.object(adapter, "execute_in_container_local_repair", side_effect=_repair),
                patch.object(engine, "should_spawn_dialectical_second_branch", return_value={"spawn_second_branch": False, "selected_action": "CommitPrimary1x"}),
                patch.object(engine.m6f, "run_tb_separate_verifier", return_value=verdict),
                patch.object(engine, "grade_archived_branch_post_lock", return_value={"strict_pass": False, "reward": 0.0}),
                patch.object(graph, "save_checkpoint", side_effect=_save_checkpoint),
            ):
                result = engine.run_m6_9_a_two_branch_episode(
                    task=task,
                    model="mock",
                    codex_info={"provider_backend": "mock", "auth_mode": "mock"},
                    forest=None,
                    forest_archive={},
                    hippo_idx=None,
                    episode_out_dir=temp / "episode",
                    branches_work_root=temp / "branches",
                    run_id=run_id,
                    selective_second_branch=True,
                    m8_container_planner=SimpleNamespace(),
                    enable_m8_local_repair=True,
                    m8_mechanism_graph=graph,
                    m8_graph_checkpoint_path=ckpt,
                )
            rec = graph.decision_ledger[0]
            cold = m_planner.PersistentMechanismGraph.load_checkpoint(ckpt)
            info = {
                "run_id": run_id,
                "decision_id": rec.decision_id,
                "status": rec.execution_status,
                "usage": {"prompt": rec.total_prompt_tokens, "completion": rec.total_completion_tokens},
                "repaired_verified": result["local_container_repair_summary"]["repaired_verified"],
                "reported_persisted_guard_id": result["local_container_repair_summary"].get("persisted_guard_id"),
                "guard_still_present_in_graph": gid in graph.mechanisms,
                "guard_present_after_cold_reload": gid in cold.mechanisms,
                "preexisting_preserved_after_cold_reload": "mech_preexisting_keep" in cold.mechanisms,
                "episode_completed": (temp / "episode/00_episode_summary.json").is_file(),
            }
        return graph, info

    _, rollback = _run_case("rollback_audit", checkpoint_fails_after_guard_registration=True)
    assert rollback["status"] == "OUTER_ACCEPTANCE_ROLLED_BACK"
    assert rollback["reported_persisted_guard_id"] is None
    assert rollback["guard_still_present_in_graph"] is False
    assert rollback["guard_present_after_cold_reload"] is False
    assert rollback["preexisting_preserved_after_cold_reload"] is True
    assert rollback["usage"] == {"prompt": 123, "completion": 45}
    assert rollback["episode_completed"] is True

    one, one_info = _run_case("independent_run_one", checkpoint_fails_after_guard_registration=False)
    two, two_info = _run_case("independent_run_two", checkpoint_fails_after_guard_registration=False)
    assert one_info["decision_id"] != two_info["decision_id"]

    shared = m_planner.PersistentMechanismGraph()
    adapter.merge_worker_graph_into_shared(shared, one)
    adapter.merge_worker_graph_into_shared(shared, two)
    assert len(shared.decision_ledger) == 2
    assert len(shared.action_return_history.get("LocalContainerRepair", [])) == 2
    # Idempotence when re-merging worker `one`
    adapter.merge_worker_graph_into_shared(shared, one)
    assert len(shared.decision_ledger) == 2
    assert len(shared.action_return_history.get("LocalContainerRepair", [])) == 2


def test_m8_5_1_repair_provenance_binding_unit_test_retention_and_router_contracts(tmp_path: Path) -> None:
    """Verifies `M85-SOURCE`, `M85-BIND`, `M85-TESTSRC`, `M85-ROUTER`, and legacy `action_return_history` tagging."""
    # 1. `M85-ROUTER`: UCB utility unit + legacy v3 mixed-target guard
    pre = {
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
    recs = [
        {"task_id": f"t{i}", "pre_action_state": dict(pre), "outcomes": {"antithesis_1x_pass": i >= 200, "m8_twobranch_pass": i < 160 or i >= 320}}
        for i in range(1000)
    ]
    pl = o_planner.M8OfflineContainerPlanner.fit_from_historical_records(recs, suite_ids=["finite_control"], cost_lambda=0.02)
    dec = pl.decide_second_branch(pre, record_to_ledger=False, run_id="run_alpha")
    dec_b = pl.decide_second_branch(pre, record_to_ledger=False, run_id="run_beta")
    assert dec["utility_unit"] == "ucb_adjusted_net_accuracy_gain_estimate"
    assert dec["decision_record"]["decision_id"] != dec_b["decision_record"]["decision_id"]

    legacy_dict = pl.to_json_dict()
    legacy_dict.pop("fail_nig_fit", None)
    with pytest.raises(ValueError, match="fail_nig_fit"):
        o_planner.M8OfflineContainerPlanner.from_json_dict(legacy_dict)
    assert o_planner.M8OfflineContainerPlanner.from_json_dict(legacy_dict, allow_legacy_mixed_target=True) is not None

    # 2. Legacy untyped `action_return_history` row tagging on reload
    g_leg = _root_graph(("u",))
    g_leg.sync_authoritative_state_to_forest()
    g_leg.forest.questions[m_planner.M8_FOREST_STATE_KEY]["action_return_history"] = {
        "LocalContainerRepair": [{"gain_nats": 1.0, "cost_spent": 0.35}]
    }
    rehydrated = m_planner.PersistentMechanismGraph(g_leg.forest)
    assert rehydrated.action_return_history["LocalContainerRepair"][0]["metric_kind"] == "legacy_untyped_m84_record"

    # 3. `M85-BIND`: `compile_verified_repair_to_executable_skill` rejects mismatched SHA-256 or missing `probe_after`
    task_stub = SimpleNamespace(
        task_id="ho06__plate_evaporation_deconfounded_ic50",
        short_name="plate_evaporation_deconfounded_ic50",
        declared_artifacts=("/holdout_task/assay_ic50.py",),
        instruction_text="Estimate deconfounded IC50.",
    )
    good_src = "def f(x):\n    return float(x) * 2.0\n"
    good_sha = hashlib.sha256(good_src.encode("utf-8")).hexdigest()
    rej_sha = adapter.compile_verified_repair_to_executable_skill(
        task=task_stub,
        repair_summary={
            "repaired_verified": True,
            "repaired_python_code": good_src,
            "repaired_candidate_sha256": "0" * 64,
            "probe_after": {"public_output_contract_ok": True, "id_renaming_invariant_ok": True},
        },
        forest=PhilosophicalForest(),
        archive={},
        codex_client=SimpleNamespace(),
    )
    assert rej_sha["compiled_and_consolidated"] is False
    assert "sha256_mismatch" in rej_sha["reason"]

    rej_probe = adapter.compile_verified_repair_to_executable_skill(
        task=task_stub,
        repair_summary={
            "repaired_verified": True,
            "repaired_python_code": good_src,
            "repaired_candidate_sha256": good_sha,
            "probe_after": {"public_output_contract_ok": True, "id_renaming_invariant_ok": False},
        },
        forest=PhilosophicalForest(),
        archive={},
        codex_client=SimpleNamespace(),
    )
    assert rej_probe["compiled_and_consolidated"] is False
    assert rej_probe["reason"] == "unverified_public_probe_after_receipt"

    # 4. `M85-SOURCE`: `consolidate_wave1_via_aufhebung_compiler` skips when `repaired_verified=True` but `04a_repaired` is missing!
    ep_dir = tmp_path / "ep_missing_rep"
    (ep_dir / "04_submitted_artifacts_snapshot").mkdir(parents=True)
    (ep_dir / "04_submitted_artifacts_snapshot" / "broken.py").write_text("def broken():\n    return 0\n", encoding="utf-8")
    t_w1 = SimpleNamespace(
        task_id="tb__toy_w1",
        short_name="toy_w1",
        benchmark="Terminal-Bench-Science",
        instruction_text="Toy wave1 task.",
    )
    sum_w1 = {
        "ended_by_finish": True,
        "local_container_repair_summary": {
            "repaired_verified": True,
            "repaired_candidate_sha256": good_sha,
        },
    }
    with (
        patch.object(m7a, "ensure_clean_scipy_sandbox_image", return_value="mock"),
        patch.object(m7a.m6f, "start_detached_container", return_value="mock-cid"),
        patch.object(m7a.m6f, "stop_container_safely", return_value=None),
        patch.object(m7a.m6f, "FrontierCodexClient", return_value=SimpleNamespace()),
    ):
        w1_out = m7a.consolidate_wave1_via_aufhebung_compiler(
            wave1_antithesis_episodes=[(t_w1, sum_w1, ep_dir)],
            forest=PhilosophicalForest(),
            archive={},
            codex_info={"provider_backend": "mock", "auth_mode": "mock"},
            model="mock",
            run_id="test_m85_src",
        )
    assert w1_out["consolidated_event_count"] == 0
    assert w1_out["events"][0]["status"] == "skipped_missing_or_mismatched_repaired_snapshot"
    assert w1_out["events"][0]["source_qualification_status"] == "missing_verified_repaired_snapshot"


def test_m8_6_static_dependency_packager_all_five_mechanical_controls_and_container_cleanroom() -> None:
    """Verifies all 5 `M8.6` static dependency closure controls + clean-room Docker container verification."""
    skill_mod_code = '''"""Autogenous Dose-Response Skill Fixture (`from __future__` + helpers + constants + branch helper)."""
from __future__ import annotations
import math
import numpy as np
from scipy.optimize import least_squares as _lsq
from . import track_wormhole_skill

HUBER_DELTA = 1.345
DERIVED_FLOOR = 1e-6 * HUBER_DELTA


def _huber_scale(r: np.ndarray, floor: float = DERIVED_FLOOR) -> np.ndarray:
    return np.maximum(np.abs(r), floor)


def _rare_branch_solver(x: np.ndarray) -> float:
    return float(np.median(x) + HUBER_DELTA)


def _unused_dead_helper(x: np.ndarray) -> float:
    return float(np.sum(x) * 999.0)


@track_wormhole_skill("autogenous_dose_fixture")
def calibrate_shifted_curve(
    doses: np.ndarray,
    responses: np.ndarray,
    *,
    use_rare_branch: bool = False,
) -> dict[str, float]:
    if use_rare_branch:
        return {"ic50": _rare_branch_solver(doses), "slope": 1.0}
    x = np.asarray(doses, dtype=float)
    y = np.asarray(responses, dtype=float)
    res = _lsq(lambda p: (p[0] + p[1] * x - y) / _huber_scale(y - np.mean(y)), x0=np.array([0.0, 1.0]))
    return {"ic50": float(res.x[0]), "slope": float(res.x[1]) + 0.0 * math.pi}
'''
    staged = {"autogenous_dose_fixture": skill_mod_code}

    # Candidate intentionally defines its OWN `_huber_scale` and `HUBER_DELTA` to test lexical collision immunity!
    cand_src = '''"""Candidate module importing staged wormhole skill."""
import numpy as np
from hegel_wormhole_skills.autogenous_dose_fixture import calibrate_shifted_curve as _skill_cal

HUBER_DELTA = -999.0


def _huber_scale(z):
    raise RuntimeError("Candidate local _huber_scale must NEVER collide with skill internal _huber_scale!")


def run_assay(doses, responses, rare=False):
    return _skill_cal(doses, responses, use_rare_branch=rare)
'''
    pkg = package_self_contained_wormhole_dependencies(cand_src, staged)
    assert pkg["had_skill_imports"] is True
    assert pkg["ok"] is True
    assert "hegel_wormhole_skills" not in pkg["packaged_python_code"]
    assert "track_wormhole_skill" not in pkg["packaged_python_code"]
    inc_defs = set(pkg["included_definitions"]["autogenous_dose_fixture"])
    assert {"HUBER_DELTA", "DERIVED_FLOOR", "_huber_scale", "_rare_branch_solver", "calibrate_shifted_curve"}.issubset(inc_defs)
    # Unused helper `_unused_dead_helper` is pruned by exact reachability while unexecuted `_rare_branch_solver` IS kept!
    assert "_unused_dead_helper" not in inc_defs

    # Fail-closed negative controls: wildcard, dynamic import, missing symbol, unsupported semantic decorator
    assert not package_self_contained_wormhole_dependencies(
        "from hegel_wormhole_skills.autogenous_dose_fixture import *\n", staged
    )["ok"]
    assert not package_self_contained_wormhole_dependencies(
        "import importlib\nm = importlib.import_module('hegel_wormhole_skills.autogenous_dose_fixture')\n", staged
    )["ok"]
    assert not package_self_contained_wormhole_dependencies(
        "from hegel_wormhole_skills.autogenous_dose_fixture import nonexistent_fn\n", staged
    )["ok"]
    bad_dec_mod = "from . import track_wormhole_skill\n@unknown_ stateful\ndef f(x): return x\n".replace("unknown_ stateful", "unknown_stateful_dec")
    assert not package_self_contained_wormhole_dependencies(
        "from hegel_wormhole_skills.bad_mod import f\n", {"bad_mod": bad_dec_mod}
    )["ok"]

    # Control 4 & 5: Real Docker container (`hegel_clean_scipy_sandbox:v1`) clean-room verification AFTER deleting `/tmp/hegel_wormhole_skills`!
    ensure_clean_scipy_sandbox_image()
    cid = start_detached_container(CLEAN_SCIPY_SANDBOX_IMAGE, memory_limit="1024m", cpus="2", network_mode="none")
    try:
        exec_container_bash(cid, "mkdir -p /holdout_task /tmp/hegel_wormhole_skills", timeout_sec=10)
        with tempfile.TemporaryDirectory(prefix="m86_docker_test_") as td:
            raw_fp = Path(td) / "assay_ic50.py"
            raw_fp.write_text(cand_src, encoding="utf-8")
            subprocess.run(["docker", "cp", str(raw_fp), f"{cid}:/holdout_task/assay_ic50.py"], check=True, timeout=20)

        assert find_uninlined_wormhole_imports_in_container(cid, ("/holdout_task/assay_ic50.py",)) == ["/holdout_task/assay_ic50.py"]
        pre_sha = compute_declared_artifacts_digest(cid, ("/holdout_task/assay_ic50.py",))

        auto_res = auto_package_wormhole_imports_in_container(cid, ("/holdout_task/assay_ic50.py",), staged)
        assert auto_res["ok"] is True
        assert len(auto_res["packaged_files"]) == 1
        assert auto_res["packaged_files"][0]["pre_package_sha256"] != auto_res["packaged_files"][0]["post_package_sha256"]

        # Physically wipe `/tmp/hegel_wormhole_skills` inside container!
        exec_container_bash(cid, "rm -rf /tmp/hegel_wormhole_skills", timeout_sec=10)
        assert find_uninlined_wormhole_imports_in_container(cid, ("/holdout_task/assay_ic50.py",)) == []

        packaged_digest = compute_declared_artifacts_digest(cid, ("/holdout_task/assay_ic50.py",))
        assert pre_sha != packaged_digest
        chk = verify_candidate_artifact_contract_in_container(
            cid,
            ("/holdout_task/assay_ic50.py",),
            packaged_digest,
            required_entrypoints=("run_assay",),
        )
        assert chk["accepted"] is True, f"Clean-room contract verification failed: {chk}"
        assert chk["computed_candidate_sha256"] == packaged_digest

        # Execute both normal path and unexecuted conditional branch (`rare=True`) with `PYTHONPATH=""` from `/`!
        ex_run = exec_container_bash(
            cid,
            "cd / && PYTHONPATH='' python3 -I -c \""
            "import sys, numpy as np; sys.path.insert(0, '/holdout_task'); import assay_ic50; "
            "x = np.linspace(0.0, 4.0, 25); y = 2.5 + 1.8 * x; "
            "r0 = assay_ic50.run_assay(x, y, rare=False); "
            "r1 = assay_ic50.run_assay(x, y, rare=True); "
            "assert abs(r0['ic50'] - 2.5) < 1e-4 and abs(r0['slope'] - 1.8) < 1e-4; "
            "assert abs(r1['ic50'] - (2.0 + 1.345)) < 1e-6; "
            "assert assay_ic50.HUBER_DELTA == -999.0; "
            "print('CLEANROOM_PACKAGED_EXEC_OK')\"",
            timeout_sec=25,
        )
        assert ex_run["exit_code"] == 0, ex_run
        assert "CLEANROOM_PACKAGED_EXEC_OK" in ex_run["stdout"]
    finally:
        stop_container_safely(cid)
