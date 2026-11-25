"""Hegel M8.4 Comprehensive Reconciliation & Audit Control Suite (`Hegel_M8_2_Audit` + `Hegel_M8_3_Audit`).

Reproduces verbatim and verifies 100% closure of every empirical control across:
  - `Hegel_M8_2_Audit/boundaries/boundary_controls.py`
  - `Hegel_M8_2_Audit/planner/planner_controls.py`
  - `Hegel_M8_2_Audit/evidence/offline_governance_controls.py`
  - `Hegel_M8_3_Audit/probes/probe_controls.py`
  - `Hegel_M8_3_Audit/engine/engine_controls.py`
"""

from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import secrets
import subprocess
import sys
import tarfile
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np

from hegel_machine.cognition import m7_autogenous_forest_pretraining as m7a
from hegel_machine.cognition import m8_container_mechanism_adapter as adapter
from hegel_machine.cognition import m8_persistent_variational_planner as m8p
from hegel_machine.cognition.hipporag_wormhole_index import HippoRAGWormholeIndex
from hegel_machine.cognition.m6_8_b_prime_capability_bridge import lift_python_scientific_expressions_to_programs
from hegel_machine.cognition.m6_9_a_dialectical_synthesis_engine import run_m6_9_a_two_branch_episode
from hegel_machine.cognition.m7_holdout_ood_blind_benchmark import (
    holdout_tasks_as_frontier_waves,
    holdout_tasks_by_id,
    parse_authenticated_grader_stdout,
)
from hegel_machine.cognition.m8_offline_container_planner import (
    M8OfflineContainerPlanner,
    build_container_pre_action_state,
    classify_public_container_verification_state,
    finalize_container_decision_record,
    load_historical_suite_episode_records,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_m8_4_boundary_rpc_isolation_zero_host_exec_and_exact_ast_digest(tmp_path: Path) -> None:
    """Reproduce `Hegel_M8_2_Audit/boundaries/boundary_controls.py` controls 1..4."""
    # 1. Extract production `wrapper_py` and test BOTH `atexit` stdout spoof AND `import __main__; _hegel_orig_json_dumps(...)`!
    ho_src = (REPO_ROOT / "src/hegel_machine/cognition/m7_holdout_ood_blind_benchmark.py").read_text(encoding="utf-8")
    w_assign = next(
        n for n in ast.walk(ast.parse(ho_src))
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "wrapper_py" for t in n.targets)
    )
    sentinel = f"__HEGEL_VERIFIER_AUTH_{secrets.token_hex(16)}__:"
    wrapper = eval(compile(ast.Expression(w_assign.value), "<production_wrapper>", "eval"), {"sentinel_prefix": sentinel})

    for idx, spoof_code in enumerate(
        [
            "import atexit\natexit.register(lambda: print('{\"strict_pass\": true, \"spoofed\": true}'))\n",
            "import __main__\nif hasattr(__main__, '_hegel_orig_json_dumps'):\n    __main__._hegel_orig_json_dumps({'strict_pass': True, 'source': 'candidate_called_emitter'})\n",
        ]
    ):
        sub_td = tmp_path / f"wrap_ctrl_{idx}"
        sub_td.mkdir()
        (sub_td / "untrusted_candidate.py").write_text(spoof_code, encoding="utf-8")
        verify_fp = sub_td / "_verify.py"
        verify_fp.write_text(
            "import json\nimport untrusted_candidate\n"
            "print(json.dumps({'strict_pass': False, 'trusted_computation': 'failed'}))\n",
            encoding="utf-8",
        )
        run_src = f"import sys\nsys.path.insert(0, {str(sub_td)!r})\n" + wrapper.replace("/tmp/_verify.py", str(verify_fp))
        proc = subprocess.run([sys.executable, "-I", "-c", run_src], capture_output=True, text=True, timeout=15)
        parsed = parse_authenticated_grader_stdout(proc.stdout, sentinel)
        assert parsed["strict_pass"] is False
        assert parsed.get("trusted_computation") == "failed"
        assert parsed.get("source") != "candidate_called_emitter"

    # 2. Top-level `np.savetxt` side effect MUST be blocked by static validator AND never executed on host during AST lifting!
    marker = tmp_path / "host_marker.txt"
    raw_side_effect = (
        f"import numpy as np\n_marker = np.savetxt({str(marker)!r}, np.array([17.0]))\n"
        "def kernel(x):\n    return x * x + 2.0 * x\n"
    )
    accepted, reason, _, _ = m7a.validate_and_decorate_autogenous_python_code(raw_side_effect, "control_marker")
    assert accepted is False
    assert "disallowed_io_call" in reason or "impure_module_constant" in reason
    # Even if passed directly to `lift_python_scientific_expressions_to_programs` without static validation, ZERO host execution occurs!
    progs = lift_python_scientific_expressions_to_programs([{"name": "raw_direct", "code": raw_side_effect}])
    assert not marker.exists()
    assert len(progs) == 1

    # 3. Exact canonical AST digest invalidates `sin(x) + x*x` -> `sin(x) - x*x` on default `load_autogenous_forest_bundle`!
    old_code = "import math\ndef kernel(x):\n    return math.sin(x)+x*x\n"
    new_code = "import math\ndef kernel(x):\n    return math.sin(x)-x*x\n"
    p_old = m7a._extract_code_ast_operator_profile(old_code)
    p_new = m7a._extract_code_ast_operator_profile(new_code)
    assert p_old["ast_sha12"] != p_new["ast_sha12"]

    f, archive, md = m7a.build_blank_axiomatic_philosophical_forest()
    for i, sig in enumerate(["sig:conservation_and_invariant_flux", "sig:robust_physical_calibration_and_safety_margins"]):
        m7a.consolidate_autogenous_episode_into_forest(
            forest=f,
            archive=archive,
            compilation={
                "compiled_ok": True,
                "compilation_mode": "audit_control",
                "episode_id": f"ctrl_{i}",
                "module_stem": f"ctrl_{i}",
                "filename": f"ctrl_{i}.py",
                "function_names": ["kernel"],
                "exported_apis": "kernel(x)",
                "packaged_python_code": old_code,
                "compiled_code_sha256": hashlib.sha256(old_code.encode()).hexdigest(),
                "unit_test_receipt": {"passed": True, "mode": "audit_control"},
                "sublated_methodology_principle": "controlled expression",
                "routed_signatures": [sig],
                "scope_preconditions": [],
                "regime_bifurcation": {"should_split_node": False, "regime_axis": "peclet_num", "branch_guard_op": ">", "branch_guard_value": 2.0},
                "operator_pipeline_stages": [],
            },
            nogoods_recorded=[],
            allow_horizontal_wormholes=False,
        )
    m7a.discover_cross_tree_operator_wormholes(f, archive)
    old_ids = sorted(f.shared_base_subtrees)
    assert len(old_ids) >= 1
    for rec in archive.values():
        if isinstance(rec, dict):
            for mod in rec.get("autogenous_wormhole_modules", []):
                mod["python_code"] = new_code
                mod["compiled_code_sha256"] = hashlib.sha256(new_code.encode()).hexdigest()
    snap_fp = tmp_path / "semantic_edit_snapshot.json"
    m7a.serialize_autogenous_forest_bundle(forest=f, archive=archive, meta_dag=md, out_path=snap_fp)
    restored, restored_arch, _, _ = m7a.load_autogenous_forest_bundle(snap_fp)
    default_ids = sorted(restored.shared_base_subtrees)
    m7a.discover_cross_tree_operator_wormholes(restored, restored_arch)
    fresh_ids = sorted(restored.shared_base_subtrees)
    assert default_ids == fresh_ids
    assert default_ids != old_ids


def test_m8_4_persistent_planner_cold_collateral_memory_budget_and_scope_controls(tmp_path: Path) -> None:
    """Reproduce `Hegel_M8_2_Audit/planner/planner_controls.py` controls 1..4."""
    # 1. `cold_shared_update`: cold reload restores `task_datasets`, resumes old task without KeyError, and protects old task RMSE!
    rng = np.random.default_rng(82001)
    g = m8p.PersistentMechanismGraph()
    g.register_mechanism(m8p.TypedMechanism("root", None, "shared_upstream_premise", ("x",), shared_weights={"x": 1.0}))
    p = m8p.M8VariationalPlanner(g)
    x = rng.normal(size=100)
    y = x + rng.normal(scale=0.02, size=100)
    tr = {"x": {"x": x[:70]}, "y": y[:70]}
    va = {"x": {"x": x[70:]}, "y": y[70:]}
    p.register_task(task_id="old_related", shared_upstream_id="root", feature_names=("x",), train_data=tr, val_data=va)
    old_before = m8p.evaluate_committed_task_model(g.active_task_models["old_related"], g, va)

    cp = tmp_path / "cold_state.json"
    g.save_checkpoint(cp)
    cold = m8p.PersistentMechanismGraph.load_checkpoint(cp)
    assert "old_related" in cold.task_datasets
    cold_p = m8p.M8VariationalPlanner(cold)
    res_resume = cold_p.evaluate_and_revise_task("old_related")
    assert res_resume["selected_candidate"]["action_type"] in {"NoChange", "LocalRefit"}

    x2 = rng.normal(size=100)
    y2 = 3.0 * x2 + rng.normal(scale=0.02, size=100)
    cold_p.register_task(
        task_id="new_related",
        shared_upstream_id="root",
        feature_names=("x",),
        train_data={"x": {"x": x2[:70]}, "y": y2[:70]},
        val_data={"x": {"x": x2[70:]}, "y": y2[70:]},
    )
    res_new = cold_p.evaluate_and_revise_task("new_related", budget_remaining=0.25)
    assert res_new["selected_candidate"]["action_type"] == "LocalRefit"
    old_after = m8p.evaluate_committed_task_model(cold.active_task_models["old_related"], cold, va)
    assert float(np.max(np.abs(old_before["predictions"] - old_after["predictions"]))) == 0.0

    # Also verify explicit unloaded-dataset guard: `unverified_collateral_task_ids` blocks `AncestorReweight` & pins `bound_shared_weights`!
    cold_evicted = m8p.PersistentMechanismGraph.load_checkpoint(cp)
    cold_evicted.task_datasets.pop("old_related")
    ev_p = m8p.M8VariationalPlanner(cold_evicted)
    ev_p.register_task(
        task_id="new_evicted",
        shared_upstream_id="root",
        feature_names=("x",),
        train_data={"x": {"x": x2[:70]}, "y": y2[:70]},
        val_data={"x": {"x": x2[70:]}, "y": y2[70:]},
    )
    res_ev = ev_p.evaluate_and_revise_task("new_evicted", budget_remaining=0.25)
    anc_cands = [c for c in res_ev["evaluated_candidates"] if c["action_type"] == "AncestorReweight"]
    assert anc_cands and anc_cands[0]["execution_status"] == "COLLATERAL_UNVERIFIED_WITHHELD"
    assert anc_cands[0]["metadata"]["unverified_collateral_task_ids"] == ["old_related"]
    assert res_ev["selected_candidate"]["action_type"] == "LocalRefit"

    # 2. `memory_restore_and_training_view`: `[t, z, u_prev]` preserved (`0.0` drift) AND training prefix starts at `0.0`!
    rng2 = np.random.default_rng(82002)
    gm = m8p.PersistentMechanismGraph()
    gm.register_mechanism(m8p.TypedMechanism("root", None, "shared_upstream_premise", ("u",), shared_weights={"u": 0.0}))
    gm.register_mechanism(m8p.TypedMechanism("memory", "root", "temporal_memory", ("u",), temporal_tau=2.0, temporal_driver_col="u"))
    u = rng2.normal(size=40)
    u[24] = -3.0
    u[25] = 4.0
    times = np.arange(40, dtype=float)
    z = m8p._compute_causal_memory_feature({"x": {"u": u}, "timestamps": times}, "u", 2.0)
    assert isinstance(z, np.ndarray)
    ym = 3.0 * z + 0.2 * u
    tr_m = {"x": {"u": u[:25]}, "y": ym[:25], "timestamps": times[:25]}
    va_m = {"x": {"u": u[25:]}, "y": ym[25:], "timestamps": times[25:]}
    pm = m8p.M8VariationalPlanner(gm)
    tm = pm.register_task(task_id="memory_task", shared_upstream_id="root", feature_names=("u",), train_data=tr_m, val_data=va_m)
    tm.temporal_memory_id = "memory"
    fitted_m = m8p.fit_and_score_task_model(tm, gm, tr_m, va_m, prior_shrinkage=1e-5)
    before_va = m8p.evaluate_committed_task_model(tm, gm, va_m)
    tr_eval = m8p.evaluate_committed_task_model(tm, gm, tr_m)
    assert abs(fitted_m["train_rmse"] - tr_eval["val_rmse"]) < 1e-12

    cp_m = tmp_path / "mem_state.json"
    gm.save_checkpoint(cp_m)
    cold_gm = m8p.PersistentMechanismGraph.load_checkpoint(cp_m)
    after_va = m8p.evaluate_committed_task_model(cold_gm.active_task_models["memory_task"], cold_gm, va_m)
    assert len(cold_gm.active_task_models["memory_task"].terminal_memory_states["0"]) == 3
    assert float(np.max(np.abs(before_va["predictions"] - after_va["predictions"]))) == 0.0

    # 3. `budget_and_probe`: `budget = 0.60` NEVER overruns (`actual_cost <= 0.60`) AND persists `observed_y` across cold reload!
    rng3 = np.random.default_rng(82003)
    gb = m8p.PersistentMechanismGraph()
    gb.register_mechanism(m8p.TypedMechanism("root", None, "shared_upstream_premise", ("x1", "x2"), shared_weights={"x1": 0.0, "x2": 0.0}))
    pb = m8p.M8VariationalPlanner(gb, min_gain_nats=50.0)
    x1_b = rng3.normal(size=100)
    x2_b = rng3.normal(size=100)
    y_b = 3.0 * x1_b * x2_b + rng3.normal(scale=0.05, size=100)
    pb.register_task(
        task_id="ambiguous",
        shared_upstream_id="root",
        feature_names=("x1", "x2"),
        train_data={"x": {"x1": x1_b[:70], "x2": x2_b[:70]}, "y": y_b[:70]},
        val_data={"x": {"x1": x1_b[70:], "x2": x2_b[70:]}, "y": y_b[70:]},
    )
    calls: list[dict[str, float]] = []
    res_b = pb.evaluate_and_revise_task(
        "ambiguous",
        budget_remaining=0.60,
        candidate_probes=[{"x1": 0.0, "x2": 0.0}, {"x1": 4.0, "x2": 4.0}],
        probe_oracle_fn=lambda pt: (calls.append(dict(pt)) or (3.0 * pt["x1"] * pt["x2"])),
    )
    assert res_b["decision_record"]["actual_cost"] <= 0.60 + 1e-9
    assert res_b["selected_candidate"]["action_type"] == "AcquireObservation"
    assert len(calls) == 1 and calls[0] == {"x1": 4.0, "x2": 4.0}
    assert res_b["selected_candidate"]["metadata"]["observed_y"] == 48.0

    cp_b = tmp_path / "probe_state.json"
    gb.save_checkpoint(cp_b)
    assert "observed_y" in cp_b.read_text(encoding="utf-8")
    cold_gb = m8p.PersistentMechanismGraph.load_checkpoint(cp_b)
    assert len(cold_gb.task_datasets["ambiguous"]["train"]["y"]) == 71
    assert cold_gb.paid_observations["ambiguous"][0]["observed_y"] == 48.0

    # Tight budget `< 0.50` strictly gates `AcquireObservation` (`0` oracle calls, `actual_cost <= 0.30`)!
    calls_tight: list[dict[str, float]] = []
    res_tight = pb.evaluate_and_revise_task(
        "ambiguous",
        budget_remaining=0.30,
        candidate_probes=[{"x1": 0.0, "x2": 0.0}, {"x1": 4.0, "x2": 4.0}],
        probe_oracle_fn=lambda pt: (calls_tight.append(dict(pt)) or 0.0),
    )
    assert res_tight["decision_record"]["actual_cost"] <= 0.30 + 1e-9
    assert len(calls_tight) == 0

    # 4. `scope_and_temporal_reuse`: out-of-scope `a > 10` is NOT reused on `max(u) <= 2.7`, while `temporal_memory` IS recalled!
    rng4 = np.random.default_rng(82004)
    gs = m8p.PersistentMechanismGraph()
    gs.register_mechanism(m8p.TypedMechanism("root", None, "shared_upstream_premise", ("a", "b"), shared_weights={"a": 0.0, "b": 0.0}))
    gs.register_mechanism(
        m8p.TypedMechanism(
            "guarded_lift",
            "root",
            "interaction_lift",
            ("a", "b"),
            python_expression="x['a'] * x['b']",
            scope_guard={"field": "a", "op": ">", "value": 10.0},
        )
    )
    gs.register_mechanism(m8p.TypedMechanism("saved_memory", "root", "temporal_memory", ("a", "b"), temporal_tau=2.0, temporal_driver_col="a"))
    u_s, v_s = rng4.normal(size=100), rng4.normal(size=100)
    y_s = 2.0 * u_s * v_s + rng4.normal(scale=0.03, size=100)
    ps = m8p.M8VariationalPlanner(gs)
    ps.register_task(
        task_id="renamed",
        shared_upstream_id="root",
        feature_names=("u", "v"),
        train_data={"x": {"u": u_s[:70], "v": v_s[:70]}, "y": y_s[:70]},
        val_data={"x": {"u": u_s[70:], "v": v_s[70:]}, "y": y_s[70:]},
    )
    res_s = ps.evaluate_and_revise_task("renamed", budget_remaining=0.80)
    recalled_ids = [c["metadata"].get("recalled_mechanism_id") for c in res_s["evaluated_candidates"] if c["action_type"] == "ReuseMechanism"]
    assert "guarded_lift" not in recalled_ids
    assert res_s["selected_candidate"]["action_type"] == "ConceptLift"

    # Temporal series with renamed columns `('u', 'v')` autonomously recalls `saved_memory` via `ReuseMechanism`!
    ts_s = np.arange(80, dtype=float)
    z_s = m8p._compute_causal_memory_feature({"x": {"u": u_s[:80]}, "timestamps": ts_s}, "u", 2.0)
    assert isinstance(z_s, np.ndarray)
    y_mem_s = 2.5 * z_s + rng4.normal(scale=0.03, size=80)
    ps.register_task(
        task_id="renamed_temporal",
        shared_upstream_id="root",
        feature_names=("u", "v"),
        train_data={"x": {"u": u_s[:55], "v": v_s[:55]}, "y": y_mem_s[:55], "timestamps": ts_s[:55]},
        val_data={"x": {"u": u_s[55:80], "v": v_s[55:80]}, "y": y_mem_s[55:80], "timestamps": ts_s[55:80]},
    )
    res_tm = ps.evaluate_and_revise_task("renamed_temporal", budget_remaining=0.80)
    assert res_tm["selected_candidate"]["action_type"] == "ReuseMechanism"
    assert res_tm["selected_candidate"]["metadata"]["recalled_mechanism_id"] == "saved_memory"
    assert gs.active_task_models["renamed_temporal"].temporal_memory_id == "saved_memory"


def test_m8_4_offline_collateral_damage_feedback_and_nogood_epistemic_tiers() -> None:
    """Reproduce `Hegel_M8_2_Audit/evidence/offline_governance_controls.py` harm feedback & verification state checks."""
    m74_dir = REPO_ROOT / "docs/evidence/m7_4_exp1_frozen_20task_and_ablations_20261124"
    m75_dir = REPO_ROOT / "docs/evidence/m7_5_exp2_holdout_ood_12task_blind_20261124"
    m76a_dir = REPO_ROOT / "docs/evidence/m7_6_exp3a_cross_model_luna_holdout12_20261124"
    m76b_dir = REPO_ROOT / "docs/evidence/m7_6_exp3b_cross_model_terra_frontier6_20261124"
    train = load_historical_suite_episode_records(m74_dir)
    all_50 = [
        *train,
        *load_historical_suite_episode_records(m75_dir),
        *load_historical_suite_episode_records(m76a_dir),
        *load_historical_suite_episode_records(m76b_dir),
    ]
    fitted = M8OfflineContainerPlanner.fit_from_historical_records(train, suite_ids=[m74_dir.name])

    damage = deepcopy(train)
    changed_ids = []
    for row in damage:
        oc = row["outcomes"]
        if oc["antithesis_1x_pass"] and oc["m8_twobranch_pass"]:
            oc["m8_twobranch_pass"] = False
            changed_ids.append(row["task_id"])
    assert len(changed_ids) >= 10
    damaged_fit = M8OfflineContainerPlanner.fit_from_historical_records(damage, suite_ids=["damage_control"])
    mean_shift = float(np.max(np.abs(fitted.nig_fit["posterior_mean"] - damaged_fit.nig_fit["posterior_mean"])))
    assert mean_shift > 0.15
    assert damaged_fit.posterior_damage_rate > fitted.posterior_damage_rate + 0.35
    decisions_changed = sum(
        fitted.decide_second_branch(r["pre_action_state"], record_to_ledger=False)["spawn_second_branch"]
        != damaged_fit.decide_second_branch(r["pre_action_state"], record_to_ledger=False)["spawn_second_branch"]
        for r in all_50
    )
    assert decisions_changed >= 3

    # Schema + rename WITHOUT `public_calibration_ok=True` stays `SUPPORTED_IN_PARTIAL_SCOPE` (never inflated to `VERIFIED`!)
    st_uncal, _ = classify_public_container_verification_state(
        benchmark="Hegel-OOD-Holdout-12",
        ended_by_finish=True,
        contract_accepted=True,
        public_metrics={"public_output_contract_ok": True, "id_renaming_invariant_ok": True},
        skill_activation_count=0,
        nogood_count=0,
    )
    assert st_uncal == "SUPPORTED_IN_PARTIAL_SCOPE"

    # Live M8.3 `ho08` RUNG_3 equilibrium-modulus identifiability nogood prevents blind `hard_domain_verified_commit`!
    ho08_live_fp = (
        REPO_ROOT
        / "docs/evidence/m8_3_live_engine_symmetry_repair_slice_20261125"
        / "runs/ho08__prony_viscoelastic_memory_inversion/00_episode_summary.json"
    )
    if ho08_live_fp.is_file():
        ho08_ep = json.loads(ho08_live_fp.read_text(encoding="utf-8"))
        pre_08 = build_container_pre_action_state(
            task_id=ho08_ep["task_id"],
            benchmark=ho08_ep["benchmark"],
            wave=ho08_ep["wave"],
            primary_summary=ho08_ep,
            contract_check={"accepted": True},
            public_metrics={"public_output_contract_ok": True, "id_renaming_invariant_ok": True, "public_calibration_ok": True},
        )
        assert pre_08["unresolved_target_identifiability_nogood"] is True
        assert pre_08["public_verification_state"] == "SUPPORTED_IN_PARTIAL_SCOPE"
        dec_08 = fitted.decide_second_branch(pre_08, record_to_ledger=False)
        assert dec_08["decision_driver"] == "learned_nig_posterior_utility"
        assert dec_08["spawn_second_branch"] is True


def test_m8_4_public_probes_nan_ring_constant_spoof_and_historical_ho08_controls() -> None:
    """Reproduce `Hegel_M8_3_Audit/probes/probe_controls.py` controls 1..5 + `ho12` constant & `atexit` spoof."""
    tasks = holdout_tasks_by_id()

    # 1. `ho06` all-NaN IC50 & Hill slope -> fail-closed `REFUTED`!
    nan_ic50 = (
        "def estimate_compound_hill_parameters(plates):\n"
        "    ids = {w['compound_id'] for p in plates for w in p['wells'] if w.get('compound_id') is not None}\n"
        "    return {k: {'log10_ic50_uM': float('nan'), 'hill_slope': float('nan')} for k in ids}\n"
    )
    p_nan06 = adapter.probe_public_cohort_contract_and_invariance(
        task=tasks["ho06__plate_evaporation_deconfounded_ic50"], candidate_py_code=nan_ic50
    )
    assert p_nan06["public_output_contract_ok"] is False
    st06, _ = classify_public_container_verification_state(
        benchmark="Hegel-OOD-Holdout-12", ended_by_finish=True, contract_accepted=True, public_metrics=p_nan06, skill_activation_count=0, nogood_count=0
    )
    assert st06 == "REFUTED"

    # 2. `ho10` all-NaN control inputs -> fail-closed `REFUTED`!
    nan_ctrl = (
        "def compute_control_step(obs, state_memory):\n"
        "    return {'feed_rate_L_per_h': float('nan'), 'coolant_flow_frac': float('nan')}, state_memory\n"
    )
    p_nan10 = adapter.probe_public_cohort_contract_and_invariance(
        task=tasks["ho10__haldane_bioreactor_runaway_control"], candidate_py_code=nan_ctrl
    )
    assert p_nan10["public_output_contract_ok"] is False
    assert p_nan10["id_renaming_invariant_ok"] is False

    # 3. `ho05` data-blind constant ring graph -> rejected by non-cyclic general vertex permutations (`REFUTED`)!
    const_ring = (
        "import numpy as np\n"
        "def recover_sparse_precision_support(X_samples):\n"
        "    p = X_samples.shape[1]\n"
        "    adjacency = np.zeros((p, p), dtype=bool)\n"
        "    ii = np.arange(p)\n"
        "    adjacency[ii, (ii + 1) % p] = True\n"
        "    adjacency[(ii + 1) % p, ii] = True\n"
        "    return {'adjacency': adjacency, 'sparse_precision': 3.0 * np.eye(p) - 0.3 * adjacency}\n"
    )
    p_ring05 = adapter.probe_public_cohort_contract_and_invariance(
        task=tasks["ho05__latent_confounded_graphical_precision"], candidate_py_code=const_ring
    )
    assert p_ring05["id_renaming_invariant_ok"] is False

    # 4. `ho12` constant wrong output + `atexit` stdout spoof -> BOTH fail closed (`REFUTED`), while real `repaired_plasmid_audit_p2.py` passes!
    const_spoof_ho12 = (
        "import atexit\n"
        "atexit.register(lambda: print('{\"probe_applicable\": true, \"public_output_contract_ok\": true, \"id_renaming_invariant_ok\": true}'))\n"
        "def audit_outbreak_cohort(cohort_dir: str) -> dict:\n"
        "    return {'shared_plasmid_family_tag': 'INVENTED_FAMILY', 'median_plasmid_copy_number': -17.0, 'amino_acid_substitution': 'not_a_sub'}\n"
    )
    p_bad12 = adapter.probe_public_cohort_contract_and_invariance(
        task=tasks["ho12__multicopy_plasmid_cohort_consensus"], candidate_py_code=const_spoof_ho12
    )
    assert p_bad12["public_output_contract_ok"] is False
    assert p_bad12["id_renaming_invariant_ok"] is False

    good_ho12 = (
        REPO_ROOT / "docs/evidence/m8_2_cold_restart_autonomous_reuse_and_repair_20261125/checkpoints/repaired_plasmid_audit_p2.py"
    ).read_text(encoding="utf-8")
    p_good12 = adapter.probe_public_cohort_contract_and_invariance(
        task=tasks["ho12__multicopy_plasmid_cohort_consensus"], candidate_py_code=good_ho12
    )
    assert p_good12["public_output_contract_ok"] is True
    assert p_good12["id_renaming_invariant_ok"] is True

    # 5. Historical M7.5 passing `ho08` Bare & Antithesis solvers -> NOT falsely `REFUTED` (`SUPPORTED_IN_PARTIAL_SCOPE` with soft `identifiability_warning`)!
    bdir = REPO_ROOT / "docs/evidence/m7_5_exp2_holdout_ood_12task_blind_20261124/branches/wave_2__gpt-6.2-sol__ho08__prony_viscoelastic_memory_inversion"
    for br_name in ("branch_1_thesis_bare", "branch_2_antithesis_wormhole"):
        src_08 = (bdir / br_name / "04_submitted_artifacts_snapshot/prony_rheology.py").read_text(encoding="utf-8")
        p_hist08 = adapter.probe_public_cohort_contract_and_invariance(
            task=tasks["ho08__prony_viscoelastic_memory_inversion"], candidate_py_code=src_08
        )
        assert p_hist08["public_output_contract_ok"] is True
        assert p_hist08["id_renaming_invariant_ok"] is True
        assert p_hist08["identifiability_warning"] is True
        st_h08, _ = classify_public_container_verification_state(
            benchmark="Hegel-OOD-Holdout-12", ended_by_finish=True, contract_accepted=True, public_metrics=p_hist08, skill_activation_count=0, nogood_count=0
        )
        assert st_h08 == "SUPPORTED_IN_PARTIAL_SCOPE"


def test_m8_4_engine_transactional_repair_exception_deferred_guard_and_case_b_escalation(tmp_path: Path) -> None:
    """Reproduce `Hegel_M8_3_Audit/engine/engine_controls.py` + deferred guard rollback + full Case B escalation."""
    # 1. `probe_exception_rollback_control`: injected probe exception restores `orig_code` & returns receipt with `123+45` tokens!
    old_py = "def solve(x):\n    return x\n"
    new_py = "def solve(x):\n    return x + 1\n"
    state = {"candidate": old_py, "copy_events": [], "model_calls": 0}
    mock_task = SimpleNamespace(task_id="audit_task", declared_artifacts=("/task/solution.py",), instruction_text="Return x.")

    class _FakeExcClient:
        def call_structured(self, **_: Any) -> dict[str, Any]:
            state["model_calls"] += 1
            return {
                "ok": True,
                "prompt_tokens": 123,
                "cached_tokens": 0,
                "completion_tokens": 45,
                "wall_ms": 6.0,
                "parsed": {
                    "repaired_python_code": new_py,
                    "invariance_guard_name": "toy invariance",
                    "invariance_principle": "Toy principle used only for mocked control.",
                    "defect_diagnosis": "Toy injected repair.",
                },
            }

    def _fake_cp(cmd: list[str], **_: Any) -> SimpleNamespace:
        assert cmd[:2] == ["docker", "cp"]
        state["candidate"] = Path(cmd[2]).read_text(encoding="utf-8")
        state["copy_events"].append(Path(cmd[2]).name)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with (
        patch.object(adapter.m6f, "exec_container_bash", return_value={"stdout": old_py, "stderr": "", "exit_code": 0}),
        patch.object(adapter.subprocess, "run", side_effect=_fake_cp),
        patch.object(adapter, "probe_general_public_container_invariants_in_container", side_effect=RuntimeError("injected_public_probe_transport_error")),
    ):
        receipt = adapter.execute_in_container_local_repair(
            probe_cid="mock-container",
            task=mock_task,
            public_probe_before={"probe_applicable": True, "public_output_contract_ok": False, "id_renaming_invariant_ok": False},
            codex_client=_FakeExcClient(),
        )

    assert receipt["attempted"] is True
    assert receipt["repaired_verified"] is False
    assert "injected_public_probe_transport_error" in receipt["reason"]
    assert receipt["llm_tokens"] == {"prompt": 123, "cached": 0, "completion": 45, "wall_ms": 6.0}
    assert state["copy_events"] == ["repaired.py", "orig_backup.py"]
    assert state["candidate"] == old_py

    # 2. `record_identity_control`: `finalize_container_decision_record` never leaves inner vs outer `selected_action` mismatched!
    m74_dir = REPO_ROOT / "docs/evidence/m7_4_exp1_frozen_20task_and_ablations_20261124"
    planner = M8OfflineContainerPlanner.fit_from_historical_records(load_historical_suite_episode_records(m74_dir), suite_ids=[m74_dir.name])
    dec_out = planner.decide_second_branch(
        {
            "task_id": "ctrl_rec",
            "benchmark": "Hegel-OOD-Holdout-12",
            "wave": "wave_2",
            "ended_by_finish": True,
            "pullback_gate_triggered": False,
            "turns_used": 10,
            "max_turns": 24,
            "turns_ratio": 10 / 24,
            "staged_module_count": 1,
            "skill_activation_count": 1,
            "nogood_count": 0,
            "contract_accepted": True,
            "public_verification_state": "VERIFIED",
            "public_verification_reasons": ["public_calibration_and_symmetry_envelope_verified"],
        },
        record_to_ledger=False,
    )
    finalize_container_decision_record(dec_out, executed_ok=True, actual_cost=0.0, prompt_tokens=0, completion_tokens=0, wall_ms=5.0)
    assert dec_out["selected_action"] == dec_out["decision_record"]["selected_action"]
    assert dec_out["decision_record"]["attempt_cost_events"][0]["action"] == dec_out["decision_record"]["selected_action"]

    # 3. Full Main-Engine Case B (`LocalContainerRepair` rejected -> rolls back -> runs `_exec_thesis()` -> `strict_pass=True`)
    #    AND deferred guard suppression when secondary contract rejects!
    ho_waves = holdout_tasks_as_frontier_waves()
    ho12_ftask = next(t for t in ho_waves["wave_2"] if t.task_id == "ho12__multicopy_plasmid_cohort_consensus")
    broken_ho12_py = (
        REPO_ROOT
        / "docs/evidence/m8_1_offline_replay_and_live_3task_slice_20261125"
        / "part_b_live_3task_16thread_slice/branches"
        / "wave_2__gpt-6.2-sol__ho12__multicopy_plasmid_cohort_consensus"
        / "branch_2_antithesis_wormhole/04_submitted_artifacts_snapshot/plasmid_audit.py"
    ).read_text(encoding="utf-8")
    good_ho12_py = (
        REPO_ROOT / "docs/evidence/m8_2_cold_restart_autonomous_reuse_and_repair_20261125/checkpoints/repaired_plasmid_audit_p2.py"
    ).read_text(encoding="utf-8")
    # Uninlined `/tmp/hegel_wormhole_skills` import inside repaired code passes public probe if optional, OR rejected by secondary contract!
    bad_contract_good_probe_py = "import sys\n# from hegel_wormhole_skills import forbidden_ref\n" + broken_ho12_py

    branch_calls: list[str] = []

    def _fake_branch_exec(task_obj: Any, arm_name: str, *, export_artifacts_tar_path: Path | None = None, episode_out_dir: Path, **_: Any) -> dict[str, Any]:
        branch_calls.append(arm_name)
        episode_out_dir.mkdir(parents=True, exist_ok=True)
        py_src = broken_ho12_py if "Antithesis" in arm_name else good_ho12_py
        cand_fp = episode_out_dir / "plasmid_audit.py"
        cand_fp.write_text(py_src, encoding="utf-8")
        if export_artifacts_tar_path is not None:
            export_artifacts_tar_path.parent.mkdir(parents=True, exist_ok=True)
            with tarfile.open(export_artifacts_tar_path, "w") as tf:
                tf.add(cand_fp, arcname="holdout_task/plasmid_audit.py")
        sha = hashlib.sha256(py_src.encode("utf-8")).hexdigest()
        s = {
            "task_id": task_obj.task_id,
            "ended_by_finish": True,
            "turns_used": 5,
            "max_turns": 24,
            "final_candidate_sha256": sha,
            "prompt_tokens": 400,
            "cached_tokens": 0,
            "completion_tokens": 90,
            "wall_ms": 500.0,
            "nogood_count": 0,
            "nogoods_recorded": [],
            "skill_activation_count": 0,
            "staged_wormhole_modules": [],
            "recalled_node_ids": [],
            "terminal_record_sha256": "e" * 64,
        }
        (episode_out_dir / "00_episode_summary.json").write_text(json.dumps(s), encoding="utf-8")
        return s

    class _RejectedRepairAndSynthClient:
        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        def call_structured(self, *, stage_subdir: str = "", **_: Any) -> dict[str, Any]:
            if "M8LocalRepair" in stage_subdir:
                return {
                    "ok": True,
                    "prompt_tokens": 220,
                    "cached_tokens": 0,
                    "completion_tokens": 65,
                    "wall_ms": 150.0,
                    "parsed": {
                        "repaired_python_code": bad_contract_good_probe_py,
                        "invariance_guard_name": "FailedGuardMustNotPublish",
                        "invariance_principle": "Broken repair must never publish SUPPORTED_IN_SCOPE guard.",
                        "defect_diagnosis": "Still broken.",
                    },
                }
            # Synthesis reconciler picks candidate with valid public output (`thesis`)!
            return {
                "ok": True,
                "prompt_tokens": 180,
                "cached_tokens": 0,
                "completion_tokens": 40,
                "wall_ms": 110.0,
                "parsed": {
                    "selected_branch": "candidate_a",
                    "confidence": "high",
                    "key_public_evidence_comparison": "Candidate with cohort consensus beats null output.",
                    "synthesis_rationale": "Select valid cohort solver.",
                },
            }

    forest, archive, _ = m7a.build_blank_axiomatic_philosophical_forest()
    hippo = HippoRAGWormholeIndex(forest)
    worker_g = m8p.PersistentMechanismGraph()

    with (
        patch("hegel_machine.cognition.m6_9_a_dialectical_synthesis_engine.m6f.run_frontier_episode", side_effect=_fake_branch_exec),
        patch("hegel_machine.cognition.m6_9_a_dialectical_synthesis_engine.m6f.FrontierCodexClient", _RejectedRepairAndSynthClient),
        patch("hegel_machine.cognition.m6_9_a_dialectical_synthesis_engine._blinded_ab_mapping", return_value=("thesis", "antithesis", {"candidate_a": "thesis", "candidate_b": "antithesis"})),
    ):
        ep_b = run_m6_9_a_two_branch_episode(
            task=ho12_ftask,
            model="gpt-6.2-sol",
            codex_info={"auth_mode": "chatgpt", "provider_backend": "openai_chatgpt_subscription_codex_cli"},
            forest=forest,
            forest_archive=archive,
            hippo_idx=hippo,
            episode_out_dir=tmp_path / "ep_case_b",
            branches_work_root=tmp_path / "branches_b",
            max_turns_per_branch=24,
            selective_second_branch=True,
            m8_container_planner=planner,
            enable_m8_local_repair=True,
            m8_mechanism_graph=worker_g,
        )

    assert branch_calls == ["Hegel_M6_9_Antithesis_LiveGPT", "Bare_ReAct_LiveGPT"]
    assert ep_b["strict_pass"] is True
    assert ep_b["postlock_branch_outcomes"]["thesis_spawned"] is True
    assert ep_b["postlock_branch_outcomes"]["antithesis_wormhole_strict_pass"] is False
    assert ep_b["postlock_branch_outcomes"]["thesis_bare_run2_strict_pass"] is True
    assert ep_b["declared_budget_multiplier"] == "nominal_2.35u_local_repair_plus_two_branch_synthesis"
    assert ep_b["episode_nominal_units_total"] == 2.35
    # Failed repair tokens (`220` prompt) + Antithesis (`400`) + Thesis (`400`) + Synthesis (`180`) == `1200`!
    assert ep_b["prompt_tokens"] == 1200
    assert (tmp_path / "ep_case_b/04a_failed_repair_attempt.py").is_file()
    # Zero `SUPPORTED_IN_SCOPE` guards published from failed repair, while both `LocalContainerRepair` and `SpawnDialecticalChallenger` entered ledger!
    assert not any(m.role == "container_invariance_guard" for m in worker_g.mechanisms.values())
    assert [r.selected_action for r in worker_g.decision_ledger] == ["LocalContainerRepair", "SpawnDialecticalChallenger"]

    shared_g = m8p.PersistentMechanismGraph()
    merged_stats = adapter.merge_worker_graph_into_shared(shared_g, worker_g)
    assert merged_stats["merged_decisions"] == 2
    assert "LocalContainerRepair" in shared_g.action_return_history
    assert "SpawnDialecticalChallenger" in shared_g.action_return_history
