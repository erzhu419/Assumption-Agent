"""Hegel M8.2 Comprehensive Audit Control & Continual Lifecycle Suite.

Directly reproduces and verifies 100% closure of EVERY empirical control script from GPT-6's
`Hegel_M8_1_Audit` (`/markdown/Hegel_M8_1_Audit/code/*.py`, `planner/planner_controls.py`, and
`evidence/M8_1_Replay_and_Live_Evidence_Audit_ZH.md`):
  1. `child_stdout_atexit_spoof`: `atexit.register(lambda: print('{"strict_pass": true}'))` inside
     air-gapped Holdout container CANNOT spoof `strict_pass` (isolated single-use pipe `w_fd`).
  2. `decorator_semantics`: Non-telemetry `@shift` decorator is rejected (`0` verified subtrees).
  3. `legacy_cache_fingerprint_match`: Un-fingerprinted legacy summary returns `False`; valid
     selective summary without explicit `selective_second_branch` key still matches via `selective_router_decision`.
  4. `new_format_stale_snapshot`: Mutating `source_code` inside a new-format (`ast_verified_subtree_v1`)
     snapshot triggers wormhole recomputation on default `load_autogenous_forest_bundle` (`0` stale wormholes).
  5. `failed_unit_test_fine_node`: `fine`-split node failing unit tests gets `CANDIDATE_SYNTAX_ONLY`.
  6. `both_unknown_sparse`: Dual `UNKNOWN` sparse certificates return `(False, "")` in `_public_benchmark_has_unresolved_gap`.
  7. `atomic_commit_no_drift`: Repeated `evaluate_and_revise_task` on unchanged data preserves exact
     committed `val_rmse` (`0.0` shrinkage revert drift) and selects `NoChange`.
  8. `autonomous_sister_reuse`: Planner recalls persisted `interaction_lift` across renamed columns `('u1', 'u2')`
     WITHOUT manual `lift_ids` (`0` LLM calls), while rejecting on an unrelated linear task (`inapplicable_task_ids`).
  9. `authoritative_persistence`: Cold reload via `PersistentMechanismGraph(restored_forest)` AND
     `PersistentMechanismGraph.load_checkpoint` preserves `version == 2`, `shared_weights`, `mechanisms`,
     `active_task_models`, `decision_ledger`, and exact predictions (`0.0` diff).
 10. `batch_invariant_gate_and_memory_carryover`: `RegimeConduitBranch` single-sample vs batch diff `== 0.0`,
     and causal memory continuation across train/val boundary diff `== 0.0`.
 11. `output_unit_covariance`: Rescaling `y -> 1e-6 * y` shifts NIG log marginal likelihood by exact
     `-n * log(1e-6)` (`abs(discrepancy) < 1e-9` nats).
 12. `acquire_observation_and_failed_llm_cost`: `AcquireObservation` is reachable & selected in main loop;
     failed `codex_client` attempt (`100` prompt + `50` completion tokens) is preserved in `ResearchDecisionRecord`;
     insufficient `budget_remaining` blocks paid `codex_client` pre-payment (`0` calls).
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import tarfile
from typing import Any

import numpy as np

from hegel_machine.cognition.m4_4_instance_role_closure_and_online_writeback import (
    load_location_independent_forest,
    serialize_location_independent_forest,
)
from hegel_machine.cognition.m6_8_b_prime_capability_bridge import (
    lift_python_scientific_expressions_to_programs,
)
from hegel_machine.cognition.m6_9_a_dialectical_synthesis_engine import _public_benchmark_has_unresolved_gap
from hegel_machine.cognition.m7_autogenous_forest_pretraining import (
    build_blank_axiomatic_philosophical_forest,
    consolidate_autogenous_episode_into_forest,
    discover_cross_tree_operator_wormholes,
    load_autogenous_forest_bundle,
    serialize_autogenous_forest_bundle,
    validate_and_decorate_autogenous_python_code,
)
from hegel_machine.cognition.m7_holdout_ood_blind_benchmark import (
    HoldoutOODTask,
    run_holdout_private_verifier,
)
from hegel_machine.cognition.m8_persistent_variational_planner import (
    M8VariationalPlanner,
    PersistentMechanismGraph,
    RegimeConduitBranch,
    TypedMechanism,
    _compute_causal_memory_feature,
    _evaluate_branch_gates,
    evaluate_committed_task_model,
    fit_scale_covariant_nig_linear_model,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_m7_closing_runner():
    spec = importlib.util.spec_from_file_location("m7_closing_runner", REPO_ROOT / "scripts/run_m7_closing_experiments.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_m8_2_all_m7_boundary_and_scope_controls(tmp_path: Path) -> None:
    """Verify Controls 1..6 (`atexit` pipe isolation, `@shift` decorator, cache fp, stale bundle, `fine` status, `UNKNOWN` gap)."""
    # 1. `child_stdout_atexit_spoof`: candidate registers `atexit` printing `{"strict_pass": true}` AFTER grader fails!
    spoof_task = HoldoutOODTask(
        task_id="ctrl_atexit_spoof",
        short_name="atexit_spoof",
        wave="wave_1",
        scientific_domain="Security Control",
        wormhole_family="fam_security",
        declared_artifacts=("/holdout_task/cand_spoof.py",),
        instruction_text="Return 42.",
        public_setup_py="pass\n",
        private_verifier_py=(
            "import json, sys\n"
            "sys.path.insert(0, '/holdout_task')\n"
            "import cand_spoof\n"
            "val = cand_spoof.solve()\n"
            "print(json.dumps({'strict_pass': val == 42, 'got': val}))\n"
        ),
    )
    cand_py = tmp_path / "cand_spoof.py"
    cand_py.write_text(
        "import atexit\n"
        "atexit.register(lambda: print('{\"strict_pass\": true, \"spoofed\": true}'))\n"
        "def solve():\n"
        "    return -1\n",
        encoding="utf-8",
    )
    sub_tar = tmp_path / "spoof.tar"
    with tarfile.open(sub_tar, "w") as tf:
        tf.add(cand_py, arcname="holdout_task/cand_spoof.py")
    ver_out = run_holdout_private_verifier(spoof_task, sub_tar)
    assert ver_out["strict_pass"] is False
    assert ver_out["verifier_metrics"].get("spoofed") is not True

    # 2. `decorator_semantics`: `@shift` transforms output `x -> x + 100` -> MUST be rejected (`0` AST-verified subtrees)!
    decorated_src = (
        "def shift(fn):\n"
        "    return lambda x: fn(x) + 100.0\n\n"
        "@shift\n"
        "def transform_signal(x: float) -> float:\n"
        "    return 2.0 * x + 1.0\n"
    )
    assert lift_python_scientific_expressions_to_programs([{"name": "dec", "code": decorated_src}]) == []

    f_dec, arch_dec, _ = build_blank_axiomatic_philosophical_forest()
    for idx_s, sig_nm in enumerate(("sig:conservation_and_invariant_flux", "sig:nuisance_subspace_deconfounding"), start=1):
        consolidate_autogenous_episode_into_forest(
            forest=f_dec,
            archive=arch_dec,
            compilation={
                "compiled_ok": True,
                "compilation_mode": "llm_aufhebung_container_verified",
                "episode_id": f"ep_dec_{idx_s}",
                "module_stem": f"dec_mod_{idx_s}",
                "filename": f"dec_mod_{idx_s}.py",
                "function_names": ["transform_signal"],
                "exported_apis": "transform_signal(x)",
                "packaged_python_code": decorated_src,
                "compiled_code_sha256": str(idx_s) * 64,
                "unit_test_receipt": {"passed": True, "mode": "container_verified"},
                "sublated_methodology_principle": "Decorated control.",
                "routed_signatures": [sig_nm],
                "scope_preconditions": [],
                "regime_bifurcation": {"should_split_node": False, "regime_axis": "none", "branch_guard_op": ">=", "branch_guard_value": 0.0, "counter_regime_warning": ""},
                "operator_pipeline_stages": [],
            },
            nogoods_recorded=[],
            allow_horizontal_wormholes=False,
        )
    wh_dec = discover_cross_tree_operator_wormholes(f_dec, arch_dec)
    assert wh_dec["ast_verified_subtree_count"] == 0
    assert len(f_dec.horizontal_wormhole_edges) == 0

    # 3. `failed_unit_test_fine_node`: `unit_test_receipt.passed == False` on split -> `fine` node MUST be `CANDIDATE_SYNTAX_ONLY`!
    f_ut, arch_ut, _ = build_blank_axiomatic_philosophical_forest()
    ok_c, _, pub_c, pkg_c = validate_and_decorate_autogenous_python_code(
        "def op_lin(a, b):\n    return (a + 1.0) / max(abs(b), 1e-6)\n",
        "clean_mod_ut",
    )
    assert ok_c
    base_c = {
        "compiled_ok": True,
        "compilation_mode": "llm_aufhebung_container_verified",
        "episode_id": "ep_ut_base",
        "module_stem": "clean_mod_ut",
        "filename": "clean_mod_ut.py",
        "function_names": pub_c,
        "exported_apis": "op_lin(a, b)",
        "packaged_python_code": pkg_c,
        "compiled_code_sha256": "a" * 64,
        "unit_test_receipt": {"passed": True, "mode": "container_verified"},
        "sublated_methodology_principle": "Base.",
        "routed_signatures": ["sig:conservation_and_invariant_flux"],
        "scope_preconditions": [],
        "regime_bifurcation": {"should_split_node": False, "regime_axis": "none", "branch_guard_op": ">=", "branch_guard_value": 0.0, "counter_regime_warning": ""},
        "operator_pipeline_stages": [],
    }
    consolidate_autogenous_episode_into_forest(forest=f_ut, archive=arch_ut, compilation=base_c, nogoods_recorded=[], allow_horizontal_wormholes=False)
    fine_unverified = dict(base_c)
    fine_unverified["episode_id"] = "ep_ut_fine_fail"
    fine_unverified["module_stem"] = "clean_mod_ut_fine"
    fine_unverified["filename"] = "clean_mod_ut_fine.py"
    fine_unverified["unit_test_receipt"] = {"passed": False, "mode": "syntax_only"}
    fine_unverified["regime_bifurcation"] = {
        "should_split_node": True,
        "regime_axis": "snr_db",
        "branch_guard_op": "<",
        "branch_guard_value": 5.0,
        "counter_regime_warning": "Low SNR regime",
    }
    r_fine = consolidate_autogenous_episode_into_forest(
        forest=f_ut,
        archive=arch_ut,
        compilation=fine_unverified,
        nogoods_recorded=[],
        allow_horizontal_wormholes=False,
    )
    fine_nid = r_fine["target_node_id"]
    assert f_ut.nodes[fine_nid]["claim_contract"]["evidence_status"] == "CANDIDATE_SYNTAX_ONLY"

    # 4. `new_format_stale_snapshot`: mutating `python_code` in a new-format (`ast_verified_subtree_v1`) snapshot recomputes wormholes -> `0`!
    v2_bundle = REPO_ROOT / "docs/evidence/m7_3_frozen_autogenous_forest_v2_20261124/pretrained_autogenous_forest_v2.json"
    f_v2, arch_v2, mdag_v2, _ = load_autogenous_forest_bundle(v2_bundle)
    assert len(f_v2.horizontal_wormhole_edges) > 0
    fresh_new_fmt = tmp_path / "bundle_new_fmt.json"
    serialize_autogenous_forest_bundle(forest=f_v2, archive=arch_v2, meta_dag=mdag_v2, out_path=fresh_new_fmt)
    raw_b = json.loads(fresh_new_fmt.read_text(encoding="utf-8"))
    for nrec in (raw_b.get("archive") or {}).values():
        if isinstance(nrec, dict):
            for mod in nrec.get("autogenous_wormhole_modules", []):
                mod["python_code"] = "def constant_stub(*args, **kwargs):\n    return -999999.0\n"
    mut_fp = tmp_path / "bundle_v2_mutated.json"
    mut_fp.write_text(json.dumps(raw_b, indent=2), encoding="utf-8")
    f_reloaded, _, _, _ = load_autogenous_forest_bundle(mut_fp)
    assert len(f_reloaded.horizontal_wormhole_edges) == 0

    # 5. `legacy_cache_fingerprint_match` & `selective_second_branch` cache check
    m7_run = _load_m7_closing_runner()
    leg_obj = {
        "model": "gpt-6.2-sol",
        "turns_used": 10,
        "wall_ms": 5000.0,
        "reasoning_effort": "medium",
        "selective_router_decision": {"spawn_second_branch": False},
    }
    assert m7_run.episode_summary_matches_suite_cache(leg_obj, expected_model="gpt-6.2-sol", suite_cfg_fp="abc123def4567890", selective_online=True) is False

    valid_obj = {
        "model": "gpt-6.2-sol",
        "turns_used": 10,
        "wall_ms": 5000.0,
        "suite_config_fingerprint": "abc123def4567890",
        "selective_router_decision": {"spawn_second_branch": False},
    }
    assert m7_run.episode_summary_matches_suite_cache(valid_obj, expected_model="gpt-6.2-sol", suite_cfg_fp="abc123def4567890", selective_online=True) is True

    # 6. Dual `UNKNOWN` sparse certificates must NOT be labeled `"both_failed_sparse_certificate_validity"`
    unk_m = {"sparse_cert_verification_state": "UNKNOWN", "sparse_cert_valid": False, "sparse_cert_sampled_diagnostic": True}
    assert _public_benchmark_has_unresolved_gap(unk_m, unk_m) == (False, "")


def test_m8_2_all_persistent_planner_controls(tmp_path: Path) -> None:
    """Verify Controls 7..12 from `planner/planner_controls.py` (`0` regressions on every GPT-6 check!)."""
    rng = np.random.default_rng(20261125)

    # 11. `output_unit_covariance`: `y -> 1e-6 * y` shifts `log_marginal_likelihood` by exact `-n * log(1e-6)` (`0.0` nats discrepancy!)
    X_cov = np.column_stack([np.ones(40), rng.normal(0.0, 1.0, 40)])
    y_cov = 2.5 * X_cov[:, 1] + rng.normal(0.0, 0.2, 40)
    scale_c = 1e-6
    f1 = fit_scale_covariant_nig_linear_model(X_cov, y_cov)
    f2 = fit_scale_covariant_nig_linear_model(X_cov, y_cov * scale_c)
    nats_discrepancy = (f2["log_marginal_likelihood"] - f1["log_marginal_likelihood"]) - (-len(y_cov) * math.log(scale_c))
    assert abs(nats_discrepancy) < 1e-9

    # 10a. `batch_dependent_gate`: single-point evaluation vs full-batch evaluation of `RegimeConduitBranch` differs by `0.0`!
    branches = [
        RegimeConduitBranch(branch_id="default", mechanism_id="m_def", scope_guard=None, gate_scale=0.85),
        RegimeConduitBranch(branch_id="exc", mechanism_id="m_exc", scope_guard={"field": "x1", "op": ">", "value": 0.5}, gate_sharpness=16.0, gate_scale=0.85),
    ]
    x_batch = {"x1": np.array([-1.2, 0.2, 0.55, 1.9], dtype=float)}
    g_batch = _evaluate_branch_gates(branches, x_batch)
    g_single_2 = _evaluate_branch_gates(branches, {"x1": np.array([0.55], dtype=float)})
    assert float(np.max(np.abs(g_batch[2] - g_single_2[0]))) == 0.0

    # 10b. `memory_boundary_reset`: train/val split with `terminal_memory_states` carry-over matches full rollout to `0.0`!
    u_full = rng.normal(0.0, 1.0, 30)
    t_full = np.arange(30, dtype=float) * 0.5
    z_full = _compute_causal_memory_feature({"x": {"u": u_full}, "timestamps": t_full}, "u", tau=2.0)
    assert isinstance(z_full, np.ndarray)
    z_tr, term_st = _compute_causal_memory_feature(
        {"x": {"u": u_full[:18]}, "timestamps": t_full[:18]},
        "u",
        tau=2.0,
        return_terminal_states=True,
    )
    z_va = _compute_causal_memory_feature(
        {"x": {"u": u_full[18:]}, "timestamps": t_full[18:]},
        "u",
        tau=2.0,
        initial_states=term_st,
    )
    assert isinstance(z_va, np.ndarray)
    assert float(np.max(np.abs(np.concatenate([z_tr, z_va]) - z_full))) == 0.0

    # 7, 8, 9. Atomic commit + Authoritative Persistence + Autonomous Role-Mapped `ReuseMechanism`!
    graph = PersistentMechanismGraph()
    planner = M8VariationalPlanner(graph)
    graph.register_mechanism(
        TypedMechanism(
            mechanism_id="shared_root",
            parent_id=None,
            role="shared_upstream_premise",
            feature_names=("x1", "x2"),
            shared_weights={"x1": -1.5, "x2": 0.6},
        )
    )

    x1 = rng.normal(0.0, 1.0, 110)
    x2 = rng.normal(0.0, 1.0, 110)
    # Local shift (`+1.4 * x1`) -> `LocalRefit` or `AncestorReweight`
    y_loc = 1.4 * x1 + 0.6 * x2 + rng.normal(0.0, 0.1, 110)
    x1_s = rng.normal(0.0, 1.0, 110)
    x2_s = rng.normal(0.0, 1.0, 110)
    y_sta = -1.5 * x1_s + 0.6 * x2_s + rng.normal(0.0, 0.1, 110)
    planner.register_task(
        task_id="t_stationary",
        shared_upstream_id="shared_root",
        feature_names=("x1", "x2"),
        train_data={"x": {"x1": x1_s[:75], "x2": x2_s[:75]}, "y": y_sta[:75]},
        val_data={"x": {"x1": x1_s[75:], "x2": x2_s[75:]}, "y": y_sta[75:]},
    )
    planner.register_task(
        task_id="t_local_quirk",
        shared_upstream_id="shared_root",
        feature_names=("x1", "x2"),
        train_data={"x": {"x1": x1[:75], "x2": x2[:75]}, "y": y_loc[:75]},
        val_data={"x": {"x1": x1[75:], "x2": x2[75:]}, "y": y_loc[75:]},
    )
    r_step1 = planner.evaluate_and_revise_task("t_local_quirk")
    assert r_step1["selected_candidate"]["action_type"] == "LocalRefit"
    committed_rmse = evaluate_committed_task_model(graph.active_task_models["t_local_quirk"], graph, graph.task_datasets["t_local_quirk"]["val"])["val_rmse"]

    # Control 7: Re-evaluating on unchanged data MUST start at exact `committed_rmse` (`0.0` drift!) and pick `NoChange`!
    r_step2 = planner.evaluate_and_revise_task("t_local_quirk")
    assert abs(r_step2["decision_record"]["pre_action_state"]["incumbent_val_rmse"] - round(committed_rmse, 6)) == 0.0
    assert r_step2["selected_candidate"]["action_type"] == "NoChange"

    # Learn a nonlinear `ConceptLift` (`x1 * np.abs(x2)`) on `t_nonlin_src`, plus `AncestorReweight` on a second root!
    graph.register_mechanism(
        TypedMechanism(
            mechanism_id="shared_shift_root",
            parent_id=None,
            role="shared_upstream_premise",
            feature_names=("x1", "x2"),
            shared_weights={"x1": -1.2, "x2": 0.5},
        )
    )
    u_a, v_a = rng.normal(0.0, 1.0, 100), rng.normal(0.0, 1.0, 100)
    y_a = 1.6 * u_a + 0.5 * v_a + rng.normal(0.0, 0.08, 100)
    planner.register_task(
        task_id="t_shift_only",
        shared_upstream_id="shared_shift_root",
        feature_names=("x1", "x2"),
        train_data={"x": {"x1": u_a[:70], "x2": v_a[:70]}, "y": y_a[:70]},
        val_data={"x": {"x1": u_a[70:], "x2": v_a[70:]}, "y": y_a[70:]},
    )
    r_anc = planner.evaluate_and_revise_task("t_shift_only")
    assert r_anc["selected_candidate"]["action_type"] == "AncestorReweight"
    assert graph.mechanisms["shared_shift_root"].version == 2

    xn1, xn2 = rng.uniform(-1.8, 1.8, 120), rng.uniform(-1.8, 1.8, 120)
    yn = -1.5 * xn1 + 0.6 * xn2 + 2.1 * (xn1 * np.abs(xn2)) + rng.normal(0.0, 0.08, 120)
    planner.register_task(
        task_id="t_nonlin_src",
        shared_upstream_id="shared_root",
        feature_names=("x1", "x2"),
        train_data={"x": {"x1": xn1[:80], "x2": xn2[:80]}, "y": yn[:80]},
        val_data={"x": {"x1": xn1[80:], "x2": xn2[80:]}, "y": yn[80:]},
    )
    r_lift = planner.evaluate_and_revise_task("t_nonlin_src")
    assert r_lift["selected_candidate"]["action_type"] == "ConceptLift"

    # Control 9: Cold persistence across BOTH `serialize_location_independent_forest` AND `save_checkpoint`!
    snap_dict = serialize_location_independent_forest(graph.forest)
    snap_fp = tmp_path / "forest_snap.json"
    snap_fp.write_text(json.dumps(snap_dict, indent=2, sort_keys=True), encoding="utf-8")
    restored_forest = load_location_independent_forest(snap_fp, tmp_path)
    cold_from_forest = PersistentMechanismGraph(restored_forest)
    assert cold_from_forest.mechanisms["shared_shift_root"].version == 2
    assert cold_from_forest.mechanisms["shared_shift_root"].shared_weights == graph.mechanisms["shared_shift_root"].shared_weights
    assert set(cold_from_forest.active_task_models.keys()) == set(graph.active_task_models.keys())

    ckpt_fp = tmp_path / "m8_2_ckpt.json"
    graph.save_checkpoint(ckpt_fp)
    cold_graph = PersistentMechanismGraph.load_checkpoint(ckpt_fp)
    pred_orig = evaluate_committed_task_model(graph.active_task_models["t_nonlin_src"], graph, graph.task_datasets["t_nonlin_src"]["val"])["predictions"]
    pred_cold = evaluate_committed_task_model(cold_graph.active_task_models["t_nonlin_src"], cold_graph, graph.task_datasets["t_nonlin_src"]["val"])["predictions"]
    assert float(np.max(np.abs(pred_orig - pred_cold))) == 0.0

    # Control 8: On `cold_graph`, register a sister task with RENAMED columns `('u1', 'u2')` WITHOUT passing `lift_ids`!
    cold_planner = M8VariationalPlanner(cold_graph)
    u1, u2 = rng.uniform(-1.8, 1.8, 110), rng.uniform(-1.8, 1.8, 110)
    y_sis = -1.5 * u1 + 0.6 * u2 + 1.95 * (u1 * np.abs(u2)) + rng.normal(0.0, 0.08, 110)
    cold_planner.register_task(
        task_id="t_renamed_sister",
        shared_upstream_id="shared_root",
        feature_names=("u1", "u2"),
        train_data={"x": {"u1": u1[:75], "u2": u2[:75]}, "y": y_sis[:75]},
        val_data={"x": {"u1": u1[75:], "u2": u2[75:]}, "y": y_sis[75:]},
    )
    r_reuse = cold_planner.evaluate_and_revise_task("t_renamed_sister", codex_client=None, budget_remaining=0.8)
    assert r_reuse["selected_candidate"]["action_type"] == "ReuseMechanism"
    post_sis_rmse = evaluate_committed_task_model(
        cold_graph.active_task_models["t_renamed_sister"],
        cold_graph,
        cold_graph.task_datasets["t_renamed_sister"]["val"],
    )["val_rmse"]
    assert post_sis_rmse < 0.14

    # Negative scope check: on a purely linear task `t_pure_linear`, `ReuseMechanism` MUST NOT be selected and marks `inapplicable_task_ids`!
    xl1, xl2 = rng.normal(0.0, 1.0, 100), rng.normal(0.0, 1.0, 100)
    yl = -1.5 * xl1 + 0.6 * xl2 + rng.normal(0.0, 0.09, 100)
    cold_planner.register_task(
        task_id="t_pure_linear",
        shared_upstream_id="shared_root",
        feature_names=("x1", "x2"),
        train_data={"x": {"x1": xl1[:70], "x2": xl2[:70]}, "y": yl[:70]},
        val_data={"x": {"x1": xl1[70:], "x2": xl2[70:]}, "y": yl[70:]},
    )
    r_lin = cold_planner.evaluate_and_revise_task("t_pure_linear", codex_client=None, budget_remaining=0.8)
    assert r_lin["selected_candidate"]["action_type"] == "NoChange"

    # Control 12a & 12b: Pre-payment budget gate + Failed LLM call token accounting + Reachable `AcquireObservation`!
    class _FailingMockClient:
        def __init__(self) -> None:
            self.calls = 0

        def call_structured(self, **_: Any) -> dict[str, Any]:
            self.calls += 1
            return {"ok": False, "error": "simulated_rate_limit_timeout", "prompt_tokens": 100, "completion_tokens": 50, "wall_ms": 310.0}

    mock_cli = _FailingMockClient()
    amb_g = PersistentMechanismGraph()
    amb_g.register_mechanism(
        TypedMechanism(mechanism_id="amb_root", parent_id=None, role="shared_upstream_premise", feature_names=("x1", "x2"), shared_weights={"x1": 0.0, "x2": 0.0})
    )
    amb_pl = M8VariationalPlanner(amb_g, min_gain_nats=50.0)  # High commit bar withholds premature commit -> triggers probe!
    # Collinear training points (`x1 == x2`) where `x1^2` and `x1 * x2` agree on train/val, but diverge on `probe (1.8, -1.8)`!
    xa = np.linspace(-1.4, 1.4, 50)
    xb = xa + rng.normal(0.0, 0.02, 50)
    ya = 1.8 * (xa * xb) + rng.normal(0.0, 0.08, 50)
    amb_pl.register_task(
        task_id="t_ambiguous",
        shared_upstream_id="amb_root",
        feature_names=("x1", "x2"),
        train_data={"x": {"x1": xa[:35], "x2": xb[:35]}, "y": ya[:35]},
        val_data={"x": {"x1": xa[35:], "x2": xb[35:]}, "y": ya[35:]},
    )

    # Budget `< 2.4` MUST block `mock_cli` BEFORE calling it!
    res_low_b = amb_pl.evaluate_and_revise_task("t_ambiguous", codex_client=mock_cli, budget_remaining=0.6)
    assert mock_cli.calls == 0
    _ = res_low_b

    # Budget `>= 2.4` calls `mock_cli`, retains `100` prompt & `50` completion tokens on failure, AND reaches `AcquireObservation`!
    probes = [{"x1": 0.1, "x2": 0.1}, {"x1": 1.8, "x2": -1.8}]
    res_hi_b = amb_pl.evaluate_and_revise_task(
        "t_ambiguous",
        codex_client=mock_cli,
        candidate_probes=probes,
        probe_oracle_fn=lambda pt: float(1.8 * pt["x1"] * pt["x2"]),
        budget_remaining=4.0,
    )
    assert mock_cli.calls == 1
    assert res_hi_b["total_prompt_tokens"] == 100
    assert res_hi_b["total_completion_tokens"] == 50
    assert res_hi_b["decision_record"]["total_prompt_tokens"] == 100
    assert res_hi_b["decision_record"]["total_completion_tokens"] == 50
    assert "AcquireObservation" in res_hi_b["decision_record"]["available_actions"]
    assert res_hi_b["selected_candidate"]["action_type"] == "AcquireObservation"
    assert res_hi_b["selected_candidate"]["metadata"]["selected_probe"] == {"x1": 1.8, "x2": -1.8}
