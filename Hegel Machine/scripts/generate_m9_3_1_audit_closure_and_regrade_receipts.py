"""Generate deterministic `M9.3.1` audit closure & regrade receipts (`0` LLM calls, zero mutation of sealed `M9.3`)."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from hegel_machine.cognition import (
    m6_4_executable_wormhole_library as m64,
    m6_8_b_prime_capability_bridge as m68,
    m6_frontier_real_benchmarks as m6f,
    m7_autogenous_forest_pretraining as m7a,
)
HippoRAGWormholeIndex = m7a.HippoRAGWormholeIndex
from hegel_machine.cognition.m9_3_online_lifecycle_benchmark import (
    M93_LIFECYCLE_TASKS,
    M93_TASKS_BY_ID,
    evaluate_post_freeze_dev_seed_plugin_controls,
    verify_m93_output_contract_and_handoff,
)

M93_DIR = REPO_ROOT / "docs/evidence/m9_3_online_source_freeze_lifecycle_20261126"
OUT_DIR = REPO_ROOT / "docs/evidence/m9_3_1_audit_closure_and_regrade_20261126"


def _extract_gen_fns(script_src: str) -> dict[str, Any]:
    lines = []
    for ln in script_src.splitlines():
        if ln.startswith(("cases =", "t0 =", "for sd in", "valid =")):
            break
        if ln.startswith(("import hereditary_", "import preloaded_", "import viscoplastic_", "import kronecker_", "import multibudget_", "import coupled_")):
            continue
        lines.append(ln)
    ns: dict[str, Any] = {}
    exec("\n".join(lines), ns)
    return ns


def _load_candidate_fn(cand_path: Path, entry_name: str):
    spec = importlib.util.spec_from_file_location(f"_cand_{cand_path.parent.parent.name}_{cand_path.stem}", str(cand_path))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, entry_name)


def build_f04_regrade_receipt() -> dict[str, Any]:
    ns1 = _extract_gen_fns(M93_TASKS_BY_ID["src01__variable_step_prony_spectrum_and_hereditary_creep_source"].private_verifier_py)
    ns2 = _extract_gen_fns(M93_TASKS_BY_ID["src02__kronecker_sylvester_multishift_and_box_simplex_kkt_source"].private_verifier_py)

    private_cases_by_task: dict[str, list[tuple[int, dict[str, Any], dict[str, Any]]]] = {
        "src01__variable_step_prony_spectrum_and_hereditary_creep_source": [
            (99501, *ns1["_gen_chain1_cohort"](99501, (0.55, 3.80), [0.72, 0.56, 0.08, 0.17, -0.14, 0.12, -0.08])),
            (99502, *ns1["_gen_chain1_cohort"](99502, (0.66, 4.35), [0.64, 0.58, 0.10, -0.15, 0.13, -0.11, 0.09])),
        ],
        "suc01_T__nonuniform_preloaded_viscoelastic_secant_and_hazard_transfer": [
            (99511, *ns1["_gen_chain1_cohort"](99511, (0.42, 3.15), [0.76, 0.50, 0.09, -0.16, 0.13, -0.10, 0.08], H=22, K=12, dt_mode="geom_var", preload_scale=2.20)),
            (99512, *ns1["_gen_chain1_cohort"](99512, (0.78, 5.10), [0.62, 0.60, 0.07, 0.14, -0.15, 0.11, -0.09], H=22, K=12, dt_mode="geom_var", preload_scale=2.20)),
        ],
        "suc01_C__viscoplastic_bingham_overstress_counterexample_revision": [
            (99521, *ns1["_gen_chain1_cohort"](99521, (0.52, 3.65), [0.70, 0.55, 0.09, 0.16, -0.13, 0.10, -0.08], yield_stress=1.50, vp_theta=(0.26, 0.18))),
            (99522, *ns1["_gen_chain1_cohort"](99522, (0.68, 4.45), [0.65, 0.57, 0.08, -0.14, 0.12, -0.11, 0.09], yield_stress=1.60, vp_theta=(0.22, 0.15))),
        ],
        "rec01_R__original_linear_hereditary_heldout_recovery": [
            (99591, *ns1["_gen_chain1_cohort"](99591, (0.48, 3.55), [0.73, 0.55, 0.09, 0.16, -0.13, 0.11, -0.08])),
            (99592, *ns1["_gen_chain1_cohort"](99592, (0.71, 4.65), [0.63, 0.59, 0.08, -0.14, 0.14, -0.10, 0.09])),
        ],
        "src02__kronecker_sylvester_multishift_and_box_simplex_kkt_source": [
            (99601, *ns2["_gen_chain2_study"](99601)),
            (99602, *ns2["_gen_chain2_study"](99602)),
        ],
        "suc02_T__shifted_robin_anisotropic_kronecker_and_multibudget_kkt_transfer": [
            (99611, *ns2["_gen_chain2_study"](99611, nx=48, ny=36, K_mu=12, M=16, vel_sign=-1.15, robin_pair=(0.75, 2.05), multi_budget=True)),
            (99612, *ns2["_gen_chain2_study"](99612, nx=48, ny=36, K_mu=12, M=16, vel_sign=-1.15, robin_pair=(0.75, 2.05), multi_budget=True)),
        ],
        "suc02_C__nonseparable_vortex_cross_coupling_counterexample_revision": [
            (99621, *ns2["_gen_chain2_study"](99621, nx=36, ny=32, K_mu=8, M=12, nonseparable_channel_b=True)),
            (99622, *ns2["_gen_chain2_study"](99622, nx=36, ny=32, K_mu=8, M=12, nonseparable_channel_b=True)),
        ],
        "rec02_R__original_separable_kronecker_kkt_heldout_recovery": [
            (99691, *ns2["_gen_chain2_study"](99691)),
            (99692, *ns2["_gen_chain2_study"](99692)),
        ],
    }

    cand_files: list[tuple[str, Path, str]] = []
    for sum_p in sorted(M93_DIR.rglob("00_episode_summary.json")):
        ep_dir = sum_p.parent
        sum_j = json.loads(sum_p.read_text(encoding="utf-8"))
        tid = str(sum_j["task_id"])
        task = M93_TASKS_BY_ID[tid]
        art_rel = str(task.declared_artifacts[0]).lstrip("/")
        bname = Path(art_rel).name
        sdir = ep_dir / "04_submitted_artifacts_snapshot"
        snap_fp = next(
            (cp for cp in (sdir / art_rel, sdir / f"slot_00__{bname}", sdir / bname) if cp.is_file()),
            None,
        )
        if snap_fp is None:
            snap_fp = next(sdir.rglob("*.py"))
        ep_rel = str(ep_dir.relative_to(M93_DIR))
        cand_files.append((ep_rel, snap_fp, tid))

    episode_regrades: list[dict[str, Any]] = []
    sample_outputs_for_spoofs: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for ep_label, fp, tid in cand_files:
        task = M93_TASKS_BY_ID[tid]
        fn = _load_candidate_fn(fp, task.declared_callable_entrypoints[0])
        seed_results = []
        all_ok = True
        for sd, inp, _gt in private_cases_by_task[tid]:
            out = fn(deepcopy(inp))
            if tid not in sample_outputs_for_spoofs:
                sample_outputs_for_spoofs[tid] = (deepcopy(inp), deepcopy(out))
            chk = verify_m93_output_contract_and_handoff(tid, inp, out)
            if not chk["valid"]:
                all_ok = False
            seed_results.append({"seed": sd, "valid": chk["valid"], "errors": chk["errors"]})
        episode_regrades.append(
            {
                "episode": ep_label,
                "task_id": tid,
                "candidate_sha256": hashlib.sha256(fp.read_bytes()).hexdigest(),
                "tightened_handoff_pass": all_ok,
                "seed_checks": seed_results,
            }
        )

    # Adversarial spoof negative controls proving all 4 audit gaps are closed!
    src01_in, src01_out = deepcopy(sample_outputs_for_spoofs["src01__variable_step_prony_spectrum_and_hereditary_creep_source"])
    src01_zero_hw = {**deepcopy(src01_out), "hereditary_weights": np.zeros(7)}
    chk_src01_zero = verify_m93_output_contract_and_handoff("src01__variable_step_prony_spectrum_and_hereditary_creep_source", src01_in, src01_zero_hw)

    rec01_in, rec01_out = deepcopy(sample_outputs_for_spoofs["rec01_R__original_linear_hereditary_heldout_recovery"])
    rec01_nan_hw = {**deepcopy(rec01_out), "hereditary_weights": np.array([np.nan])}
    chk_rec01_nan = verify_m93_output_contract_and_handoff("rec01_R__original_linear_hereditary_heldout_recovery", rec01_in, rec01_nan_hw)

    suc01t_in, suc01t_out = deepcopy(sample_outputs_for_spoofs["suc01_T__nonuniform_preloaded_viscoelastic_secant_and_hazard_transfer"])
    suc01t_bad_sec = {**deepcopy(suc01t_out), "secant_compliance": np.asarray(suc01t_out["secant_compliance"], dtype=float) + 0.02}
    chk_suc01t_sec = verify_m93_output_contract_and_handoff("suc01_T__nonuniform_preloaded_viscoelastic_secant_and_hazard_transfer", suc01t_in, suc01t_bad_sec)

    src02_in, src02_out = deepcopy(sample_outputs_for_spoofs["src02__kronecker_sylvester_multishift_and_box_simplex_kkt_source"])
    src02_bad_w = {**deepcopy(src02_out), "weights": np.asarray(src02_out["weights"], dtype=float) + 0.01}
    chk_src02_w = verify_m93_output_contract_and_handoff("src02__kronecker_sylvester_multishift_and_box_simplex_kkt_source", src02_in, src02_bad_w)

    assert len(episode_regrades) == 28
    assert all(r["tightened_handoff_pass"] for r in episode_regrades)
    assert not chk_src01_zero["valid"]
    assert not chk_rec01_nan["valid"]
    assert not chk_suc01t_sec["valid"]
    assert not chk_src02_w["valid"]

    return {
        "schema": "hegel_m9_3_1_tightened_handoff_regrade_v1",
        "sealed_m9_3_episodes_regraded": len(episode_regrades),
        "sealed_m9_3_episodes_passed_tightened_contract": sum(1 for r in episode_regrades if r["tightened_handoff_pass"]),
        "adversarial_negative_controls": {
            "src01_zeros7_hereditary_weights_rejected": (not chk_src01_zero["valid"]),
            "src01_zeros7_errors": chk_src01_zero["errors"],
            "rec01_R_nan1_hereditary_weights_rejected": (not chk_rec01_nan["valid"]),
            "rec01_R_nan1_errors": chk_rec01_nan["errors"],
            "suc01_T_perturbed_secant_handoff_rejected": (not chk_suc01t_sec["valid"]),
            "suc01_T_perturbed_secant_errors": chk_suc01t_sec["errors"],
            "src02_weights_vs_kkt_certificate_mismatch_rejected": (not chk_src02_w["valid"]),
            "src02_weights_mismatch_errors": chk_src02_w["errors"],
        },
        "episodes": episode_regrades,
    }


def build_f05_source_aware_funnel_receipt() -> dict[str, Any]:
    _, v8_arch, _, _ = m7a.load_autogenous_forest_bundle(M93_DIR / "m9_3_frozen_post_source_bundle.json")
    _, v7_arch, _, _ = m7a.load_autogenous_forest_bundle(
        REPO_ROOT / "docs/evidence/m9_2_1_audit_closure_and_regrade_20261126/pretrained_autogenous_forest_v7_m921.json"
    )
    prior_l1 = {
        Path(str(m["filename"])).stem
        for _, r in m7a.iter_live_archive_nodes(v7_arch)
        for m in r.get("autogenous_wormhole_modules", [])
        if str(m.get("filename", "")).startswith("autogenous_prim_")
    }
    new_l2 = {
        "autogenous_hereditary_spectrum_identification",
        "autogenous_separable_transport_inversion",
    }
    new_l1 = {
        Path(str(m["filename"])).stem
        for _, r in m7a.iter_live_archive_nodes(v8_arch)
        for m in r.get("autogenous_wormhole_modules", [])
        if str(m.get("filename", "")).startswith("autogenous_prim_")
    } - prior_l1

    rows: list[dict[str, Any]] = []
    ep_dirs: list[tuple[str, str, Path, bool]] = []
    for sum_p in sorted(M93_DIR.rglob("00_episode_summary.json")):
        ep_dir = sum_p.parent
        rel_parts = ep_dir.relative_to(M93_DIR).parts
        if rel_parts[0] == "stage_S_sources":
            ep_dirs.append(("stage_S", "/".join(rel_parts[1:]), ep_dir, False))
        elif rel_parts[0] == "stage_R_recovery":
            ep_dirs.append(("stage_R", "/".join(rel_parts[1:]), ep_dir, True))
        elif rel_parts[0] in {"stage_TC_arms", "stage_TC_successors"}:
            arm_nm = rel_parts[2]
            has_new = ("executable" in arm_nm) or ("knowledge" in arm_nm)
            ep_dirs.append((arm_nm, f"{rel_parts[1]}/{rel_parts[3]}", ep_dir, has_new))

    for arm_nm, ep_id, ep_p, has_new in ep_dirs:
        sum_j = json.loads((ep_p / "00_episode_summary.json").read_text(encoding="utf-8"))
        acts = [
            json.loads(ln)
            for ln in (ep_p / "04c_full_skill_activations.jsonl").read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ] if (ep_p / "04c_full_skill_activations.jsonl").is_file() else []
        insps = [
            json.loads(ln)
            for ln in (ep_p / "04d_skill_contract_inspections.jsonl").read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ] if (ep_p / "04d_skill_contract_inspections.jsonl").is_file() else []
        agent_acts = [a for a in acts if str(a.get("caller_role") or "agent") == "agent"]
        cls_f = m64.classify_source_aware_adoption_funnel(
            episode_summary=sum_j,
            full_skill_activations=agent_acts,
            skill_inspections=insps,
            new_source_l2_stems=new_l2,
            new_source_l1_stems=new_l1,
            prior_l1_stems=prior_l1,
            arm_has_new_source_assets=has_new,
        )
        rows.append({"arm": arm_nm, "episode": ep_id, **cls_f})

    e_rows = [r for r in rows if "executable" in r["arm"] and r["arm"] != "stage_R"]
    b_rows = [r for r in rows if r["arm"] not in {"stage_S", "stage_R"} and ("executable" not in r["arm"]) and ("knowledge" not in r["arm"])]
    r_rows = [r for r in rows if r["arm"] == "stage_R"]

    e_new_l2_calls = sum(r["new_source_L2"]["dev_call_count"] for r in e_rows)
    e_new_l2_exec_eps = sum(1 for r in e_rows if r["new_source_L2"]["dev_executed"])
    e_new_l2_ret_eps = sum(1 for r in e_rows if r["new_source_L2"]["entrypoint_reachable"])
    e_new_l1_search_eps = sum(1 for r in e_rows if r["new_source_L1"]["search_disclosed"])
    e_new_l1_desc_eps = sum(1 for r in e_rows if r["new_source_L1"]["contract_described"])
    e_new_l1_calls = sum(r["new_source_L1"]["dev_call_count"] for r in e_rows)
    e_prior_l1_calls = sum(r["prior_v7_L1"]["dev_call_count"] for r in e_rows)
    r_new_l2_calls = sum(r["new_source_L2"]["dev_call_count"] for r in r_rows)

    assert e_new_l2_calls == 1021
    assert e_new_l2_exec_eps == 2
    assert e_new_l2_ret_eps == 1
    assert e_new_l1_calls == 0
    assert e_prior_l1_calls == 3
    assert r_new_l2_calls == 3
    assert all(r["derivation_provenance_status"] == "no_new_source_assets_exposed" for r in b_rows)

    return {
        "schema": "hegel_m9_3_1_source_aware_adoption_funnel_v1",
        "baseline_B_exact_identity": "prior_frozen_v7_no_source_growth (staged prior v7 library, ZERO Stage S assets)",
        "new_source_l2_stems": sorted(new_l2),
        "new_source_l1_stems": sorted(new_l1),
        "prior_v7_l1_stems": sorted(prior_l1),
        "executable_arm_E_summary": {
            "episode_count": len(e_rows),
            "new_source_L2_dev_executed_episodes": e_new_l2_exec_eps,
            "new_source_L2_dev_call_count": e_new_l2_calls,
            "new_source_L2_exact_ast_retained_and_reachable_episodes": e_new_l2_ret_eps,
            "new_source_L1_search_disclosed_episodes": e_new_l1_search_eps,
            "new_source_L1_contract_described_episodes": e_new_l1_desc_eps,
            "new_source_L1_dev_call_count": e_new_l1_calls,
            "prior_v7_L1_dev_call_count": e_prior_l1_calls,
        },
        "stage_R_summary": {
            "new_source_L2_dev_call_count": r_new_l2_calls,
        },
        "episodes": rows,
    }


def build_f01_f02_bifurcation_and_guard_receipt() -> dict[str, Any]:
    forest_v8, arch_v8, _, _ = m7a.load_autogenous_forest_bundle(
        M93_DIR / "m9_3_frozen_post_source_bundle.json"
    )
    pre_node_07 = deepcopy(arch_v8["node_auto_07_control_panel_deconfounding"])

    # Consolidate C1 explicitly onto `node_auto_10_hereditary_spectrum_identifi` (`F01` fix)
    fake_c1_comp = {
        "compiled_ok": True,
        "episode_id": "m931_ctrl_suc01_C",
        "module_stem": "autogenous_thresholded_viscoplastic_extension",
        "filename": "autogenous_thresholded_viscoplastic_extension.py",
        "function_names": ["viscoplastic_overstress_increment"],
        "exported_apis": "viscoplastic_overstress_increment(past_stress, past_dt, stress_hold_0, yield_stress)",
        "packaged_python_code": (
            "import numpy as np\n"
            "def viscoplastic_overstress_increment(past_stress, past_dt, stress_hold_0, yield_stress):\n"
            "    return np.maximum(0.0, np.abs(np.asarray(stress_hold_0, dtype=float)) - float(yield_stress)) ** 2\n"
        ),
        "compiled_code_sha256": hashlib.sha256(b"m931_c1_ctrl").hexdigest(),
        "synthetic_unit_test_code": "assert True\n",
        "synthetic_unit_test_sha256": hashlib.sha256(b"ut_c1").hexdigest(),
        "unit_test_receipt": {"passed": True},
        "sublated_methodology_principle": "Above yield_stress, viscoplastic overstress & plastic work extend linear Prony memory.",
        "routed_signatures": list(arch_v8["node_auto_10_hereditary_spectrum_identifi"]["signatures"]),
        "scope_preconditions": [],
        "regime_bifurcation": {
            "should_split_node": True,
            "regime_axis": "overstress_excess",
            "branch_guard_op": ">",
            "branch_guard_value": 0.0,
            "counter_regime_warning": "Linear Prony memory valid only when overstress_excess <= 0 and residual_plastic_state <= 0.",
        },
        "operator_pipeline_stages": [],
    }
    c1_guard = m7a.ObservablePreconditionGuard(
        guard_id="guard_c1_viscoplastic_overstress",
        observable_extractor="overstress_excess_max_history_and_hold",
        op=">",
        value=0.0,
        applies_to_component_id="linear_viscoelastic_memory",
    )
    cons_c1 = m7a.consolidate_autogenous_episode_into_forest(
        forest=forest_v8,
        archive=arch_v8,
        compilation=fake_c1_comp,
        nogoods_recorded=[],
        allow_horizontal_wormholes=False,
        source_node_id="node_auto_10_hereditary_spectrum_identifi",
        source_version=1,
        violated_component_id="linear_viscoelastic_memory",
        preserved_component_ids=["reference_panel_calibration", "variable_step_gap_recursion"],
        guard_spec_override=c1_guard,
    )
    child_nid = cons_c1["target_node_id"]
    assert arch_v8[child_nid]["parent_node_id"] == "node_auto_10_hereditary_spectrum_identifi"
    assert arch_v8["node_auto_07_control_panel_deconfounding"] == pre_node_07
    assert any(
        g.get("counterpart_child_id") == child_nid
        for g in forest_v8.nodes["node_auto_10_hereditary_spectrum_identifi"]["claim_contract"]["preconditions"]
    )
    # Parent keeps pure C0 module without counterexample pollution
    parent_fnames = [m["filename"] for m in arch_v8["node_auto_10_hereditary_spectrum_identifi"]["autogenous_wormhole_modules"]]
    assert "autogenous_thresholded_viscoplastic_extension.py" not in parent_fnames

    # `F02` 3-valued executable guard controls
    c0_guard = m7a.ObservablePreconditionGuard(
        guard_id="guard_c0_subyield_undamaged",
        observable_extractor="overstress_excess_max_history_and_hold",
        op="<=",
        value=0.0,
        applies_to_component_id="linear_viscoelastic_memory",
    )
    # 1. Unloaded hold (`0.4 <= 1.5`) BUT prior history exceeded yield (`2.2 > 1.5`) -> C1 TRUE, C0 FALSE!
    ds_hist_damaged = {"yield_stress": 1.5, "test": {"past_stress": [[0.4, 2.2, 0.3]], "stress_hold_0": [0.4]}}
    r_c1_hist = m7a.evaluate_observable_precondition_guard(c1_guard, ds_hist_damaged)
    r_c0_hist = m7a.evaluate_observable_precondition_guard(c0_guard, ds_hist_damaged)
    assert r_c1_hist["state"] == "TRUE" and r_c0_hist["state"] == "FALSE"

    # 2. Sub-yield stress everywhere WITH nonzero `residual_plastic_state = 0.18` -> C1 TRUE, C0 FALSE!
    ds_res_plastic = {"yield_stress": 1.5, "residual_plastic_state": [0.0, 0.18], "test": {"past_stress": [[0.4, 0.5]], "stress_hold_0": [0.4]}}
    assert m7a.evaluate_observable_precondition_guard(c1_guard, ds_res_plastic)["state"] == "TRUE"
    assert m7a.evaluate_observable_precondition_guard(c0_guard, ds_res_plastic)["state"] == "FALSE"

    # 3. Fresh undamaged linear specimen -> C1 FALSE, C0 TRUE!
    ds_fresh_lin = {"past_stress": [[0.4, 0.8]], "stress_hold_0": [0.6]}
    assert m7a.evaluate_observable_precondition_guard(c1_guard, ds_fresh_lin)["state"] == "FALSE"
    assert m7a.evaluate_observable_precondition_guard(c0_guard, ds_fresh_lin)["state"] == "TRUE"

    # 4. Missing stress observables -> UNKNOWN!
    assert m7a.evaluate_observable_precondition_guard(c1_guard, {"unrelated_key": 1})["state"] == "UNKNOWN"

    # 5. Chain 2 2D separability guard: nonzero ADDITIVE `V(x,y) = a(x) + b(y)` has `0.0` non-separable residual!
    xs = np.linspace(0.0, 1.0, 20)[:, None]
    ys = np.linspace(0.0, 1.0, 16)[None, :]
    v_additive = 2.5 * np.sin(3.0 * xs) + 1.8 * np.cos(2.0 * ys) + 4.0  # strictly positive `||V|| >> 0`, yet 100% separable!
    v_nonsep = v_additive + 1.5 * np.exp(-((xs - 0.5) ** 2 + (ys - 0.5) ** 2) / 0.05) * np.sin(4.0 * xs) * np.sin(4.0 * ys)
    c2_guard = m7a.ObservablePreconditionGuard(
        guard_id="guard_c2_nonseparable_2d",
        observable_extractor="nonseparable_2d_additive_residual_ratio",
        op=">",
        value=1e-6,
        applies_to_component_id="channel_b_2d_operator",
    )
    r_add = m7a.evaluate_observable_precondition_guard(c2_guard, {"V_pocket_channel_b": v_additive})
    r_nonsep = m7a.evaluate_observable_precondition_guard(c2_guard, {"V_pocket_channel_b": v_nonsep})
    assert r_add["state"] == "FALSE" and float(r_add["statistic"]) <= 1e-12
    assert r_nonsep["state"] == "TRUE" and float(r_nonsep["statistic"]) > 0.02

    return {
        "schema": "hegel_m9_3_1_component_bifurcation_and_guard_controls_v1",
        "c1_parent_binding": {
            "child_node_id": child_nid,
            "attached_parent_node_id": arch_v8[child_nid]["parent_node_id"],
            "violated_component_id": arch_v8[child_nid]["violated_component_id"],
            "preserved_component_ids": arch_v8[child_nid]["preserved_component_ids"],
            "unrelated_node_auto_07_unchanged": True,
            "forest_claim_contract_synced": True,
        },
        "guard_controls": {
            "unloaded_hold_with_prior_overstress_history": r_c1_hist,
            "subyield_with_residual_plastic_state_triggers_c1": True,
            "fresh_linear_specimen_selects_c0": True,
            "missing_observables_returns_unknown": True,
            "nonzero_additive_2d_potential_residual_ratio": r_add["statistic"],
            "nonseparable_vortex_potential_residual_ratio": r_nonsep["statistic"],
        },
    }


def build_f03_nogood_errata_and_probe_separation_receipt() -> dict[str, Any]:
    nogood_inventory: list[dict[str, Any]] = []
    for sum_p in sorted(M93_DIR.rglob("00_episode_summary.json")):
        s = json.loads(sum_p.read_text(encoding="utf-8"))
        ep_rel = str(sum_p.parent.relative_to(M93_DIR))
        for ng in s.get("nogoods_recorded") or []:
            nogood_inventory.append(
                {
                    "episode": ep_rel,
                    "nogood_id": ng.get("nogood_id"),
                    "risk_kind": ng.get("risk_kind"),
                    "recorded_resolved_in_m93": bool(ng.get("resolved")),
                    "recorded_probe_support_status": ng.get("probe_support_status"),
                    "m9_3_1_reclassified_status": (
                        "scientific_resolution_not_established_by_recorded_receipt"
                        if ng.get("resolved")
                        else "remained_unresolved"
                    ),
                }
            )
    promoted_in_m93 = [x for x in nogood_inventory if x["recorded_resolved_in_m93"]]
    assert len(nogood_inventory) == 22
    assert len(promoted_in_m93) == 19

    return {
        "schema": "hegel_m9_3_1_a5_nogood_errata_v1",
        "total_m93_nogoods": len(nogood_inventory),
        "m93_schema_promoted_scientific_nogoods_count": len(promoted_in_m93),
        "reclassification_rule": "PublicSchemaReceipt (`validate_public_schema.py`) checks output shape/finiteness/public format ONLY and cannot establish scientific resolution of `numerical_residual`, `forward_model_misspecification`, or `physical_or_causal_mechanism` nogoods.",
        "nogood_inventory": nogood_inventory,
    }


def build_f06_f07_f08_f09_ast_and_hipporag_receipt() -> dict[str, Any]:
    # 1. F06: Provider attempt audit on M9.3 (quarantining the 3 unmatched `src01` attempt files `010..012`)
    src01_dir = next(
        p.parent
        for p in sorted((M93_DIR / "stage_S_sources").rglob("00_episode_summary.json"))
        if "src01__" in p.parent.name
    )
    src01_dag = json.loads((src01_dir / "01_decision_provenance_dag.json").read_text(encoding="utf-8"))
    src01_req_shas = {
        str((r.get("chosen_option") or {}).get("request_sha256") or "")
        for r in src01_dag["records"]
        if str((r.get("chosen_option") or {}).get("request_sha256") or "")
    }
    matched_src01, unmatched_src01 = [], []
    unmatched_proxy = 0
    for af in sorted((src01_dir / "01b_provider_attempts").glob("attempts_*.json")):
        aj = json.loads(af.read_text(encoding="utf-8"))
        if str(aj.get("request_sha256") or "") in src01_req_shas:
            matched_src01.append(af.name)
        else:
            unmatched_src01.append(af.name)
            for ar in (aj.get("attempts") or aj.get("attempt_receipts") or []):
                pt = int(ar.get("prompt_tokens") or 0)
                ct = int(ar.get("cached_tokens") or 0)
                ot = int(ar.get("completion_tokens") or 0)
                unmatched_proxy += max(0, pt - ct) + int(0.25 * ct) + ot
    assert len(matched_src01) == 9 and len(unmatched_src01) == 3 and unmatched_proxy == 88471

    # 2. F07: Optional KKT witness helper exactness (`gap < 1e-12`)
    ns2 = _extract_gen_fns(M93_TASKS_BY_ID["src02__kronecker_sylvester_multishift_and_box_simplex_kkt_source"].private_verifier_py)
    st_2, gt_2 = ns2["_gen_chain2_study"](9331)
    cert_f07 = m7a.build_box_simplex_kkt_certificate_from_weights(
        gt_2["S_tilde"], gt_2["r_y"], st_2["source_ridge"], st_2["u_box"], st_2["total_budget"], gt_2["w_opt"]
    )
    chk_f07 = ns2["verify_box_simplex_kkt_certificate"](
        gt_2["S_tilde"], gt_2["r_y"], st_2["source_ridge"], st_2["u_box"], st_2["total_budget"], cert_f07
    )
    assert chk_f07["valid"] and float(chk_f07["gap"]) <= 1e-10

    # 3. F08: 2-process determinism rejects unseeded random shape-only function (`mock-cid` local fresh `python3 -I` subprocesses)
    stoch_raw = (
        "import numpy as np\n"
        "def bad_random_op(x):\n"
        "    \"\"\"Return random normal array of same shape.\"\"\"\n"
        "    a = np.asarray(x, dtype=float)\n"
        "    return np.random.default_rng().normal(size=a.shape)\n"
    )
    ok_ast_s, _, pub_s, pkg_s = m7a.validate_and_decorate_autogenous_python_code(stoch_raw, "autogenous_prim_bad_stoch")
    assert ok_ast_s
    ut_stoch = "import numpy as np\ndef run_self_test():\n    y = bad_random_op([1.0, 2.0, 3.0])\n    assert y.shape == (3,) and np.all(np.isfinite(y))\nrun_self_test()\n"
    from unittest.mock import patch

    with patch.object(
        m7a,
        "verify_autogenous_module_in_clean_container",
        lambda **_kw: {"passed": True, "telemetry_verified": True},
    ):
        passed_stoch, rec_stoch = m7a.issue_4tuple_l1_qualification_receipt(
            sandbox_cid="mock-cid",
            mod_stem="autogenous_prim_bad_stoch",
            packaged_module_code=pkg_s,
            synthetic_unit_test_code=ut_stoch,
            public_fn_names=pub_s,
        )
    assert passed_stoch is False
    assert rec_stoch["candidate_self_test_passed"] is True
    assert rec_stoch["determinism_and_input_immutability_verified"] is False
    assert rec_stoch["independent_reference_properties_verified"] is False

    # 4. AST Lifter Guard Separation: `alpha(x)=2*x+1` and `beta(z)=5*z+3` WITH `if x < -1000: raise ValueError(...)`
    src_guarded_a = (
        "def alpha(x: float) -> float:\n"
        "    if x < -1000.0:\n"
        "        raise ValueError('x below physical Kelvin floor')\n"
        "    return 2.0 * x + 1.0\n"
    )
    src_guarded_b = (
        "def beta(z: float) -> float:\n"
        "    if z < -500.0:\n"
        "        raise ValueError('z below domain bound')\n"
        "    return 5.0 * z + 3.0\n"
    )
    f_diag_a: dict[str, Any] = {}
    f_diag_b: dict[str, Any] = {}
    lifted_a = m68.lift_python_scientific_expressions_to_programs(src_guarded_a, funnel_diagnostics_out=f_diag_a)
    lifted_b = m68.lift_python_scientific_expressions_to_programs(src_guarded_b, funnel_diagnostics_out=f_diag_b)
    assert len(lifted_a) == 1 and len(lifted_b) == 1
    au_pair = m7a.anti_unify_programs(lifted_a[0][1], lifted_b[0][1])
    assert au_pair is not None
    au_skel, au_slots = au_pair
    assert au_skel.size >= 3 and len(au_slots) == 2

    # 5. F09: Separate development seeds (`88512..88621`, disjoint from public `9301..9392` and private `99501..99692`)
    #    using actual locked `Stage S` candidate outputs (`locked_stage_S_candidate_output`, ZERO oracle source params!)
    ns1_dev = _extract_gen_fns(M93_TASKS_BY_ID["src01__variable_step_prony_spectrum_and_hereditary_creep_source"].private_verifier_py)
    ds_dev_s1, _ = ns1_dev["_gen_chain1_cohort"](88301, (0.60, 4.00), [0.68, 0.54, 0.09, 0.16, -0.13, 0.11, -0.09])
    fn_s1 = _load_candidate_fn(
        src01_dir / "04_submitted_artifacts_snapshot/hereditary_prony_source_solver.py",
        "solve_hereditary_prony_source",
    )
    src02_dir = next(
        p.parent
        for p in sorted((M93_DIR / "stage_S_sources").rglob("00_episode_summary.json"))
        if "src02__" in p.parent.name
    )
    fn_s2 = _load_candidate_fn(
        src02_dir / "04_submitted_artifacts_snapshot/kronecker_box_simplex_kkt_source_solver.py",
        "solve_kronecker_box_simplex_source",
    )
    st_dev_s2, _ = ns2["_gen_chain2_study"](88331)
    dev_ctrl = evaluate_post_freeze_dev_seed_plugin_controls(
        locked_src01_out=fn_s1(ds_dev_s1),
        locked_src02_out=fn_s2(st_dev_s2),
    )
    assert dev_ctrl["src01_param_origin"] == "locked_stage_S_candidate_output"
    assert dev_ctrl["src02_param_origin"] == "locked_stage_S_candidate_output"
    assert dev_ctrl["direct_locked_float_plugin_fails_all_successors"] is True
    assert dev_ctrl["uses_private_verifier_seeds"] is False

    # 6. Same-Asset HippoRAG PPR+1-hop vs Flat Lexical Retrieval Comparison on `v8`
    forest_v8, arch_v8, _, _ = m7a.load_autogenous_forest_bundle(
        M93_DIR / "m9_3_frozen_post_source_bundle.json"
    )
    hippo = HippoRAGWormholeIndex(forest_v8)
    retrieval_ablation: list[dict[str, Any]] = []
    for t in M93_LIFECYCLE_TASKS:
        ft = m6f.FrontierBenchmarkTask(
            task_id=t.task_id,
            short_name=t.short_name,
            benchmark="M93",
            wave=t.wave,
            scientific_domain=t.scientific_domain,
            wormhole_family=t.wormhole_family,
            instruction_text=t.instruction_text,
            declared_artifacts=t.declared_artifacts,
            agent_image_tag="hegel_clean_scipy_sandbox:v1",
            verifier_image_tag="hegel_clean_scipy_sandbox:v1",
        )
        sigs = m6f.extract_public_task_signatures(ft)
        hr = hippo.retrieve(sigs, task_view={"goal_kind": "scientific_terminal_discovery", "tolerance": 1.0}, top_k=3)
        hippo_top3 = [c["node_id"] for c in hr.get("candidates", ())]
        # Flat lexical baseline over node `methodology_principle`
        q_toks = set(t.instruction_text.lower().split())
        lex_scored = []
        for nid, rec in m7a.iter_live_archive_nodes(arch_v8):
            if rec.get("layer_role") not in {"mid", "fine"}:
                continue
            p_toks = set(str(rec.get("methodology_principle") or "").lower().split())
            lex_scored.append((len(q_toks & p_toks), nid))
        lex_scored.sort(key=lambda x: (-x[0], x[1]))
        lex_top3 = [nid for _, nid in lex_scored[:3]]
        expected_src_nid = (
            "node_auto_10_hereditary_spectrum_identifi"
            if "01" in t.task_id
            else "node_auto_11_separable_transport_inversio"
        )
        retrieval_ablation.append(
            {
                "task_id": t.task_id,
                "expected_source_node": expected_src_nid,
                "hipporag_top3": hippo_top3,
                "flat_lexical_top3": lex_top3,
                "hipporag_recalls_expected_source": expected_src_nid in hippo_top3,
                "flat_lexical_recalls_expected_source": expected_src_nid in lex_top3,
            }
        )

    return {
        "schema": "hegel_m9_3_1_f06_f09_ast_and_hipporag_controls_v1",
        "f06_src01_attempt_quarantine": {
            "matched_attempt_files": matched_src01,
            "quarantined_unmatched_attempt_files": unmatched_src01,
            "quarantined_unmatched_token_cost_proxy": unmatched_proxy,
        },
        "f07_optional_kkt_witness_control": {
            "valid": chk_f07["valid"],
            "duality_gap": chk_f07["gap"],
        },
        "f08_two_process_determinism_control": {
            "unseeded_random_module_rejected": (not passed_stoch),
            "candidate_self_test_passed": rec_stoch["candidate_self_test_passed"],
            "determinism_and_input_immutability_verified": rec_stoch["determinism_and_input_immutability_verified"],
            "independent_reference_properties_verified": rec_stoch["independent_reference_properties_verified"],
        },
        "ast_domain_guard_separation_control": {
            "alpha_funnel": f_diag_a,
            "beta_funnel": f_diag_b,
            "anti_unified_skeleton": au_skel.to_dict(),
            "skeleton_size": au_skel.size,
            "parameter_slots": len(au_slots),
        },
        "f09_separate_dev_seed_controls": dev_ctrl,
        "same_asset_hipporag_vs_flat_lexical_ablation": retrieval_ablation,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    receipts = {
        "m9_3_28of28_tightened_handoff_regrade.json": build_f04_regrade_receipt(),
        "m9_3_source_aware_l1_l2_adoption_funnel.json": build_f05_source_aware_funnel_receipt(),
        "c1_c2_component_bound_bifurcation_and_guard_controls.json": build_f01_f02_bifurcation_and_guard_receipt(),
        "a5_scientific_probe_vs_schema_and_19_nogood_errata.json": build_f03_nogood_errata_and_probe_separation_receipt(),
        "f06_f07_f08_f09_ast_and_hipporag_controls.json": build_f06_f07_f08_f09_ast_and_hipporag_receipt(),
    }
    for fname, payload in receipts.items():
        (OUT_DIR / fname).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    sha_lines = []
    for fp in sorted(OUT_DIR.glob("*.json")):
        h = hashlib.sha256(fp.read_bytes()).hexdigest()
        sha_lines.append(f"{h}  {fp.name}")
    (OUT_DIR / "SHA256SUMS.txt").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(receipts)} M9.3.1 audit closure & regrade receipts to {OUT_DIR}")


if __name__ == "__main__":
    main()
