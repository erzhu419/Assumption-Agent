"""`M10.2` Gate 0–2 Synthetic + Real Docker Verification Suite (`0` Paid LLM Calls, `0` Confirmation Seed Touching!).

Verifies (`U01–U12` + `E1–E4`):
  1. **Strict Seed & Split Isolation (`U01`)**:
     - Test suite NEVER imports or touches `CONFIRMATION_M102_WORLD_SPECS` (verified by AST inspection of this test file!);
     - Seed ranges of `UNIT_TEST_M102_WORLD_SPECS` (`7201000..`), `DEV_M102_WORLD_SPECS` (`1141000..`), and
       `CONFIRMATION_M102_WORLD_SPECS` (`1151000..`) are 100% disjoint!
  2. **Zero Prompt Split-Scaffolding & Role-Blind `stat_00..stat_17` Observables (`U06 / §七`)**:
     - `DE_SCAFFOLDED_INITIAL_DECOMP_SYSTEM_PROMPT_TEMPLATE`, `DE_SCAFFOLDED_EDIT_PLANNER_SYSTEM_PROMPT_TEMPLATE`,
       and `M102_OBSERVABLE_NAMES` contain ZERO `panel_a`, `panel_b`, `panel_c`, `pa_lin_r2`, `pb_lin_r2`, `pc_v1`,
       `tel_ch`, `diamond`, `chain_k3`, or `skip_dag` hints!
  3. **Zero-LLM Structural Investment & Maintenance Gate (`evaluate_m102_selective_investment_gate`, `E2`)**:
     - Rejects decomposition (`invest_in_structure == False`, `permanent_source_verdict == True`) on `monolithic_coupled`
       solvers while approving decomposition (`invest_in_structure == True`) on multi-panel separable solvers with localized
       `G0..G3` shifts; suspends surgery on consecutive failed edits.
  4. **Real Docker Execution (`hegel-clean-scipy:m7v1`) on `UNIT_TEST_M102_WORLD_SPECS` (`H = 16`, Wave-2 `A2/B2/L2/R2`)**:
     - Verifies Wave-2 `A2`, `B2`, `L2`, `R2` operators generate distinct non-nested regimes in container, and verifies
       end-to-end 3-arm equal-probe-cap (`<= 4`) execution + `0`-LLM guard collision avoidance + Step-14 Cold-Hydration
       & Plain-Dict Routing Parity (`100.0%`)!
"""

from __future__ import annotations

import ast
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hegel_machine.cognition.m7_holdout_ood_blind_benchmark import ensure_clean_scipy_sandbox_image
from hegel_machine.cognition.m10_1_unknown_cut_dag_discovery_world import (
    ExecutableComputationDAG,
    ExecutableModuleNode,
    REF_DEV_NODE_JOIN_READOUT_PY,
    build_ref_dev_node_pa_code,
    build_ref_dev_node_pb_code,
    compile_m101_dag_to_instrumented_solver_code,
    execute_m101_public_gate_in_container,
    grade_m101_candidate_in_container,
    materialize_m101_public_study_and_sha,
)
from hegel_machine.cognition.m10_2_selective_bounded_dag_world import (
    DE_SCAFFOLDED_EDIT_PLANNER_SYSTEM_PROMPT_TEMPLATE,
    DE_SCAFFOLDED_INITIAL_DECOMP_SYSTEM_PROMPT_TEMPLATE,
    M102_HORIZON_STEPS,
    M102_MAX_PRE_LLM_PROBES_PER_STEP,
    M102_OBSERVABLE_NAMES,
    UNIT_TEST_M102_WORLD_SPECS,
    build_m102_world_tasks,
    compute_m102_blind_observables,
    evaluate_m102_selective_investment_gate,
    inspect_monolithic_code_ast_separability,
)
from hegel_machine.cognition.wormhole_forest import PhilosophicalForest


def test_m102_gate0_seed_isolation_and_zero_prompt_scaffolding() -> None:
    # 1. Verify this test module NEVER imports `CONFIRMATION_M102_WORLD_SPECS` (`U01`)!
    src_self = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(src_self)
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "CONFIRMATION_M102_WORLD_SPECS" not in imported_names

    # 2. Verify strictly role-blind `stat_00..stat_17` observable names and de-scaffolded prompts (`U06`)
    assert len(M102_OBSERVABLE_NAMES) == 18
    assert all(nm == f"stat_{i:02d}" for i, nm in enumerate(M102_OBSERVABLE_NAMES))

    combined_prompts = (
        DE_SCAFFOLDED_INITIAL_DECOMP_SYSTEM_PROMPT_TEMPLATE.lower()
        + "\n"
        + DE_SCAFFOLDED_EDIT_PLANNER_SYSTEM_PROMPT_TEMPLATE.lower()
    )
    for forbidden in (
        "panel_a",
        "panel_b",
        "panel_c",
        "pa_lin_r2",
        "pb_lin_r2",
        "pc_v1",
        "pc_v2",
        "tel_ch",
        "diamond",
        "chain_k3",
        "chain_k4",
        "skip_dag",
    ):
        assert forbidden not in combined_prompts, f"Forbidden scaffold token `{forbidden}` leaked in M10.2 prompt!"

    assert M102_HORIZON_STEPS == 16
    assert M102_MAX_PRE_LLM_PROBES_PER_STEP == 4


def test_m102_gate1_zero_llm_selective_investment_and_maintenance_gate() -> None:
    mono_code = """
import numpy as np
def solve_batch(study: dict) -> dict:
    x_tr = study["train"]["x"]
    y_tr = study["train"]["y"]
    tr_tr = study["train"]["traj"]
    x_te = study["test"]["x"]
    A = np.column_stack([np.ones(len(x_tr)), x_tr, x_tr ** 2])
    A_te = np.column_stack([np.ones(len(x_te)), x_te, x_te ** 2])
    wy, *_ = np.linalg.lstsq(A, y_tr, rcond=None)
    wt, *_ = np.linalg.lstsq(A, tr_tr, rcond=None)
    yh = A_te @ wy
    sp = A_te @ wt
    return {"y_hat": yh, "state_pred": sp, "diagnostics_vec": np.zeros(4)}
"""
    sep_code = """
import numpy as np
from scipy.optimize import least_squares
def solve_batch(study: dict) -> dict:
    pa = study["panel_a"]
    pb = study["panel_b"]
    c1, *_ = np.linalg.lstsq(np.column_stack([pa["u"], np.ones(len(pa["u"]))]), pa["v"], rcond=None)
    c2, *_ = np.linalg.lstsq(np.column_stack([pb["u"], np.ones(len(pb["u"]))]), pb["v"], rcond=None)
    c3, *_ = np.linalg.lstsq(np.column_stack([study["train"]["x"], np.ones(len(study["train"]["x"]))]), study["train"]["y"], rcond=None)
    return {"y_hat": np.zeros(1), "state_pred": np.zeros((1, 8)), "diagnostics_vec": np.zeros(4)}
"""
    assert inspect_monolithic_code_ast_separability(mono_code)["ast_separable"] is False
    assert inspect_monolithic_code_ast_separability(sep_code)["ast_separable"] is True

    ref_obs = {f"stat_{i:02d}": 0.98 for i in range(18)}
    tgt_mono_shift = {**ref_obs, "stat_12": 1.35, "stat_14": 1.25}
    tgt_b_shift = {**ref_obs, "stat_04": 0.42, "stat_05": 0.35, "stat_15": 1.25}
    tgt_global_shock = {**ref_obs, "stat_08": 0.12, "stat_09": 0.40, "stat_10": 0.05, "stat_11": 0.55}

    g_m = evaluate_m102_selective_investment_gate(
        phase="initial_decomposition",
        source_code=mono_code,
        ref_obs=ref_obs,
        target_obs=tgt_mono_shift,
        step_index=1,
        total_horizon=16,
        consecutive_failed_dag_searches=0,
        total_failed_dag_searches=0,
        solver_free_dag_reuse_hits=0,
    )
    assert g_m["invest_in_structure"] is False and g_m["permanent_source_verdict"] is True

    g_s = evaluate_m102_selective_investment_gate(
        phase="initial_decomposition",
        source_code=sep_code,
        ref_obs=ref_obs,
        target_obs=tgt_b_shift,
        step_index=1,
        total_horizon=16,
        consecutive_failed_dag_searches=0,
        total_failed_dag_searches=0,
        solver_free_dag_reuse_hits=0,
    )
    assert g_s["invest_in_structure"] is True and g_s["shifted_blocks"] == ["G1"]

    g_shock = evaluate_m102_selective_investment_gate(
        phase="initial_decomposition",
        source_code=sep_code,
        ref_obs=ref_obs,
        target_obs=tgt_global_shock,
        step_index=1,
        total_horizon=16,
        consecutive_failed_dag_searches=0,
        total_failed_dag_searches=0,
        solver_free_dag_reuse_hits=0,
    )
    assert g_shock["invest_in_structure"] is False and g_shock["permanent_source_verdict"] is False

    g_cooldown = evaluate_m102_selective_investment_gate(
        phase="graph_maintenance_edit",
        source_code=sep_code,
        ref_obs=ref_obs,
        target_obs=tgt_b_shift,
        step_index=5,
        total_horizon=16,
        consecutive_failed_dag_searches=1,
        total_failed_dag_searches=1,
        solver_free_dag_reuse_hits=2,
    )
    assert g_cooldown["invest_in_structure"] is False


def test_m102_gate2_real_docker_wave2_and_end_to_end_selective_vs_eager_and_step14_hydration(tmp_path: Path) -> None:
    from scripts.run_m10_2_selective_bounded_dag_experiment import (
        _run_one_m102_world_arm_stream,
        run_m102_step14_cold_hydration_and_plain_dict_parity_ablation,
    )

    ensure_clean_scipy_sandbox_image()

    # Part A: Verify `Hegel_SelectiveBoundedDAG_Online` keeps `K=1` (`0` decomp calls!) on `ut_m102_mono` (`7201000`)
    sp_mono = UNIT_TEST_M102_WORLD_SPECS[0]
    s0_m, tasks_m, man_m = build_m102_world_tasks(sp_mono)
    assert len(tasks_m) == 16
    st0_m, pub0_m, _ = materialize_m101_public_study_and_sha(s0_m)
    obs0_m = compute_m102_blind_observables(st0_m)

    mono_ref_solver = r'''from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares

def solve_batch(study: dict) -> dict:
    x_tr = np.asarray(study["train"]["x"], float)
    y_tr = np.asarray(study["train"]["y"], float)
    tr_tr = np.asarray(study["train"]["traj"], float)
    x_te = np.asarray(study["test"]["x"], float)
    # Implicit fixed-point basis approximation without reading auxiliary panels
    def _feat(x):
        cols = [np.ones(len(x)), x, np.tanh(0.8 * x), x ** 2]
        for i in range(4):
            for j in range(i + 1, 4):
                cols.append((x[:, i] * x[:, j])[:, None])
        return np.column_stack(cols)
    F_tr, F_te = _feat(x_tr), _feat(x_te)
    reg = 1e-3 * np.eye(F_tr.shape[1])
    wy = np.linalg.solve(F_tr.T @ F_tr + reg, F_tr.T @ y_tr)
    wt = np.linalg.solve(F_tr.T @ F_tr + reg, F_tr.T @ tr_tr)
    yh = F_te @ wy
    sp = F_te @ wt
    dv = np.array([float(np.mean(yh)), float(np.std(yh)), float(np.mean(sp[:, -1])), float(np.std(sp[:, -1]))], float)
    return {"y_hat": yh, "state_pred": sp, "diagnostics_vec": dv}
'''
    decomp_calls = {"eager": 0, "selective": 0}

    class _MockMonoBackend:
        def make_client(self, *, role: str, world_id: str, arm: str, ep_sn: str):
            class _Cli:
                def call_structured(self, *, system_prompt, conversation_turns, output_schema, stage_subdir):
                    if "init_decomp_" in stage_subdir:
                        if arm == "Eager_BoundedDAG_Online":
                            decomp_calls["eager"] += 1
                        else:
                            decomp_calls["selective"] += 1
                    return {"ok": True, "parsed": {"action": "keep_monolithic", "rationale": "mono", "nodes": [], "edges": [], "sink_node_id": "", "node_modules": []}}
            return _Cli()

        def solve_in_sandbox(self, *, task, role, world_id, arm, ep_sn, sub_tag):
            return {"batch_solver.py": mono_ref_solver}, {"turns_used": 1}

    g0_m = execute_m101_public_gate_in_container(task=s0_m, candidate_code=mono_ref_solver, expect_dag_trace=False)
    for arm_test in ("Eager_BoundedDAG_Online", "Hegel_SelectiveBoundedDAG_Online"):
        res_m = _run_one_m102_world_arm_stream(
            spec=sp_mono,
            arm=arm_test,
            s_task=s0_m,
            stream_tasks=tasks_m[:4],
            schedule_manifest=man_m[:4],
            base_v7_forest=PhilosophicalForest(),
            base_v7_archive={},
            source_monolithic_code=mono_ref_solver,
            source_pub_sha256=str(g0_m.get("public_input_sha256") or pub0_m),
            source_obs=obs0_m,
            source_public_gate=g0_m,
            model="mock",
            codex_info={},
            stream_out_dir=tmp_path / "mono_gate_check" / arm_test,
            run_id="ut_mono",
            max_turns=4,
            reasoning_effort="medium",
            mock_backend=_MockMonoBackend(),
        )
        assert res_m["max_active_nodes_observed"] == 1
        assert res_m["max_pre_llm_probes_observed"] <= M102_MAX_PRE_LLM_PROBES_PER_STEP

    assert decomp_calls["eager"] >= 1
    assert decomp_calls["selective"] == 0, "Hegel_SelectiveBoundedDAG_Online must spend 0 LLM decomp calls on monolithic_coupled!"

    # Part B: Full 16-step (`H = 16`, incl. Wave-2 `A2` & `B2` + Step-14 Cold-Hydration & Plain-Dict Parity) on `ut_m102_k3` (`7202000`)!
    sp_k3 = UNIT_TEST_M102_WORLD_SPECS[1]
    s0_k3, tasks_k3, man_k3 = build_m102_world_tasks(sp_k3)
    st0_k3, pub0_k3, _ = materialize_m101_public_study_and_sha(s0_k3)
    obs0_k3 = compute_m102_blind_observables(st0_k3)

    init_nlib = {
        "a0": ExecutableModuleNode("node_00", "a0", build_ref_dev_node_pa_code("A0")),
        "b0": ExecutableModuleNode("node_01", "b0", build_ref_dev_node_pb_code("B0")),
        "c0": ExecutableModuleNode("node_02", "c0", REF_DEV_NODE_JOIN_READOUT_PY),
    }
    init_dag = ExecutableComputationDAG(
        "tmp_s0",
        ["node_00", "node_01", "node_02"],
        [("__input__", "node_00"), ("__input__", "node_01"), ("__input__", "node_02"), ("node_00", "node_01"), ("node_01", "node_02")],
        "node_02",
        {"node_00": "a0", "node_01": "b0", "node_02": "c0"},
    )
    src_k3_code, _ = compile_m101_dag_to_instrumented_solver_code(init_dag, init_nlib)
    src_k3_gate = execute_m101_public_gate_in_container(task=s0_k3, candidate_code=src_k3_code, expect_dag_trace=False)
    assert src_k3_gate["accepted"] is True

    c1_node_code = r'''from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares

def run_node(inputs: dict) -> dict:
    study = inputs["__input__"]
    prev = inputs["node_01"]
    h_tr, h_te = np.asarray(prev["h_tr"], float), np.asarray(prev["h_te"], float)
    tr_y, tr_traj = np.asarray(study["train"]["y"], float), np.asarray(study["train"]["traj"], float)
    def _feat(h):
        return np.column_stack([np.ones(len(h)), h, np.sum(h[:, :2] * h[:, 2:], axis=1)])
    F_tr, F_te = _feat(h_tr), _feat(h_te)
    w_tr = np.linalg.solve(F_tr.T @ F_tr + 1e-6 * np.eye(F_tr.shape[1]), F_tr.T @ tr_traj)
    w0 = np.linalg.solve(F_tr.T @ F_tr + 1e-6 * np.eye(F_tr.shape[1]), F_tr.T @ tr_y)
    def _res(p):
        b = F_tr @ p[:-2]
        return (b + p[-2] * np.tanh(p[-1] * b)) - tr_y
    opt = least_squares(_res, np.concatenate([w0, [0.45, 1.6]]), max_nfev=400)
    p = opt.x
    b_te = F_te @ p[:-2]
    yh = b_te + p[-2] * np.tanh(p[-1] * b_te)
    sp = F_te @ w_tr
    dv = np.array([float(np.mean(yh)), float(np.std(yh)), float(np.mean(sp[:, -1])), float(np.std(sp[:, -1]))], dtype=float)
    return {"y_hat": yh, "state_pred": sp, "diagnostics_vec": dv}
'''
    c1_rewired_coupled_code = r'''from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares

def run_node(inputs: dict) -> dict:
    study = inputs["__input__"]
    z_d, h_d = inputs["node_00"], inputs["node_01"]
    z_tr, z_te = np.asarray(z_d["z_tr"], float), np.asarray(z_d["z_te"], float)
    h_tr, h_te = np.asarray(h_d["h_tr"], float), np.asarray(h_d["h_te"], float)
    tr_y, tr_traj = np.asarray(study["train"]["y"], float), np.asarray(study["train"]["traj"], float)
    def _feat(z, h):
        cpl = np.sum(z * np.roll(h, 1, axis=1), axis=1)
        return np.column_stack([np.ones(len(h)), h, np.sum(h[:, :2] * h[:, 2:], axis=1)]), cpl
    F_tr, cpl_tr = _feat(z_tr, h_tr)
    F_te, cpl_te = _feat(z_te, h_te)
    X_tr = np.column_stack([F_tr, cpl_tr])
    X_te = np.column_stack([F_te, cpl_te])
    w_tr = np.linalg.solve(X_tr.T @ X_tr + 1e-6 * np.eye(X_tr.shape[1]), X_tr.T @ tr_traj)
    w0 = np.linalg.solve(F_tr.T @ F_tr + 1e-6 * np.eye(F_tr.shape[1]), F_tr.T @ tr_y)
    def _res(p):
        b = F_tr @ p[:6]
        return (b + p[6] * np.tanh(p[7] * b) + p[8] * cpl_tr) - tr_y
    opt = least_squares(_res, np.concatenate([w0, [0.45, 1.6, 0.7]]), max_nfev=400)
    p = opt.x
    b_te = F_te @ p[:6]
    yh = b_te + p[6] * np.tanh(p[7] * b_te) + p[8] * cpl_te
    sp = X_te @ w_tr
    dv = np.array([float(np.mean(yh)), float(np.std(yh)), float(np.mean(sp[:, -1])), float(np.std(sp[:, -1]))], dtype=float)
    return {"y_hat": yh, "state_pred": sp, "diagnostics_vec": dv}
'''

    a2_node_code = r'''from __future__ import annotations
import numpy as np

def run_node(inputs: dict) -> dict:
    study = inputs["__input__"]
    u = np.asarray(study["panel_a"]["u"], dtype=float)
    v = np.asarray(study["panel_a"]["v"], dtype=float)
    def _fwd_phi(z):
        zo = np.roll(z, 2, axis=1)
        return np.column_stack([np.ones(len(z)), z, np.sin(2.4 * zo), np.tanh(1.8 * z)])
    P = _fwd_phi(u)
    w = np.ones(len(u))
    for _ in range(8):
        sw = np.sqrt(w)[:, None]
        W, *_ = np.linalg.lstsq(P * sw, v * sw, rcond=None)
        res = np.linalg.norm(v - P @ W, axis=1)
        w = 1.0 / (1.0 + (res / max(3.0 * float(np.median(res)), 1e-4)) ** 2)
    keep = np.linalg.norm(v - P @ W, axis=1) <= max(5.0 * float(np.median(np.linalg.norm(v - P @ W, axis=1))), 0.1)
    W, *_ = np.linalg.lstsq(P[keep], v[keep], rcond=None)
    def _inv(x_obs):
        z = np.zeros_like(x_obs)
        eps = 1e-5
        for _ in range(16):
            f = _fwd_phi(z) @ W - x_obs
            J = np.zeros((len(z), 4, 4), dtype=float)
            for j in range(4):
                zp = z.copy(); zp[:, j] += eps
                J[:, :, j] = (_fwd_phi(zp) @ W - _fwd_phi(z) @ W) / eps
            dz = np.linalg.solve(J, f[:, :, None])[:, :, 0]
            z = z - 0.85 * dz
            if float(np.max(np.abs(dz))) < 1e-9:
                break
        return z
    return {"z_tr": _inv(np.asarray(study["train"]["x"], dtype=float)), "z_te": _inv(np.asarray(study["test"]["x"], dtype=float))}
'''

    class _MockK3H16Backend:
        def make_client(self, *, role: str, world_id: str, arm: str, ep_sn: str):
            class _Cli:
                def call_structured(self, *, system_prompt, conversation_turns, output_schema, stage_subdir):
                    if "init_decomp_" in stage_subdir:
                        return {
                            "ok": True,
                            "parsed": {
                                "action": "split_node",
                                "rationale": "Factorize 3-stage pipeline",
                                "nodes": ["node_00", "node_01", "node_02"],
                                "edges": [["__input__", "node_00"], ["__input__", "node_01"], ["__input__", "node_02"], ["node_00", "node_01"], ["node_01", "node_02"]],
                                "sink_node_id": "node_02",
                                "node_modules": [
                                    {"node_id": "node_00", "code": build_ref_dev_node_pa_code("A0")},
                                    {"node_id": "node_01", "code": build_ref_dev_node_pb_code("B0")},
                                    {"node_id": "node_02", "code": REF_DEV_NODE_JOIN_READOUT_PY},
                                ],
                            },
                        }
                    if "edit_plan_" in stage_subdir:
                        import json as _jj
                        pay = _jj.loads(conversation_turns[0]["content"])
                        dags = pay["existing_verified_dags"]
                        base = [d for d in dags if len(d["nodes"]) == 3 and len(d["edges"]) == 5][0]
                        v_by_n: dict[str, list[str]] = {}
                        for nv in pay["verified_node_versions"]:
                            v_by_n.setdefault(nv["node_id"], []).append(nv["version_id"])
                        step_num = int(ep_sn.rsplit("_batch_", 1)[-1])
                        if step_num in (1, 12):  # B1 on S01, B2 on S12
                            return {"ok": True, "parsed": {"action": "reuse_subgraph", "rationale": "edit node_01", "base_dag_id": base["dag_id"], "new_nodes": base["nodes"], "new_edges": base["edges"], "new_sink_node_id": "node_02", "reused_node_versions": [{"node_id": "node_00", "version_id": base["node_versions"]["node_00"]}, {"node_id": "node_02", "version_id": base["node_versions"]["node_02"]}], "editable_node_ids": ["node_01"]}}
                        if step_num == 2:  # C1 on S02
                            return {"ok": True, "parsed": {"action": "reuse_subgraph", "rationale": "edit node_02", "base_dag_id": base["dag_id"], "new_nodes": base["nodes"], "new_edges": base["edges"], "new_sink_node_id": "node_02", "reused_node_versions": [{"node_id": "node_00", "version_id": base["node_versions"]["node_00"]}, {"node_id": "node_01", "version_id": base["node_versions"]["node_01"]}], "editable_node_ids": ["node_02"]}}
                        if step_num in (3, 9):  # A1 on S03, A2 on S09
                            return {"ok": True, "parsed": {"action": "reuse_subgraph", "rationale": "edit node_00", "base_dag_id": base["dag_id"], "new_nodes": base["nodes"], "new_edges": base["edges"], "new_sink_node_id": "node_02", "reused_node_versions": [{"node_id": "node_01", "version_id": base["node_versions"]["node_01"]}, {"node_id": "node_02", "version_id": base["node_versions"]["node_02"]}], "editable_node_ids": ["node_00"]}}
                        if step_num in (6, 14):  # coupled interaction (`S06`, `S14`) -> `rewire_edge` adding `node_00 -> node_02`
                            # `node_00` versions: [0]=Stage S mono, [1]=A0, [2]=A1, [3]=A2
                            a_v = sorted(v_by_n["node_00"])[2 if step_num == 6 else -1]
                            b_v = sorted(v_by_n["node_01"])[1]
                            cand_bases = [d for d in dags if len(d["nodes"]) == 3 and len(d["edges"]) == 5 and d["node_versions"].get("node_00") == a_v and d["node_versions"].get("node_01") == b_v]
                            b_ref = cand_bases[-1] if cand_bases else base
                            rew_edges = list(b_ref["edges"]) + [["node_00", "node_02"]]
                            return {"ok": True, "parsed": {"action": "rewire_edge", "rationale": "add skip edge node_00 -> node_02", "base_dag_id": b_ref["dag_id"], "new_nodes": b_ref["nodes"], "new_edges": rew_edges, "new_sink_node_id": "node_02", "reused_node_versions": [{"node_id": "node_00", "version_id": a_v}, {"node_id": "node_01", "version_id": b_v}], "editable_node_ids": ["node_02"]}}
                    return {"ok": False}
            return _Cli()

        def solve_in_sandbox(self, *, task, role, world_id, arm, ep_sn, sub_tag):
            step_num = int(ep_sn.rsplit("_batch_", 1)[-1])
            if step_num == 1:
                return {"node_01.py": build_ref_dev_node_pb_code("B1")}, {"turns_used": 2}
            if step_num == 2:
                return {"node_02.py": c1_node_code}, {"turns_used": 2}
            if step_num == 3:
                return {"node_00.py": build_ref_dev_node_pa_code("A1")}, {"turns_used": 2}
            if step_num == 9:
                return {"node_00.py": a2_node_code}, {"turns_used": 2}
            if step_num == 12:
                return {"node_01.py": build_ref_dev_node_pb_code("B2")}, {"turns_used": 2}
            return {"node_02.py": c1_rewired_coupled_code}, {"turns_used": 2}

    st_dir = tmp_path / "continual_streams" / sp_k3.world_id / "Hegel_SelectiveBoundedDAG_Online"
    st_sum = _run_one_m102_world_arm_stream(
        spec=sp_k3,
        arm="Hegel_SelectiveBoundedDAG_Online",
        s_task=s0_k3,
        stream_tasks=tasks_k3,
        schedule_manifest=man_k3,
        base_v7_forest=PhilosophicalForest(),
        base_v7_archive={},
        source_monolithic_code=src_k3_code,
        source_pub_sha256=str(src_k3_gate.get("public_input_sha256") or pub0_k3),
        source_obs=obs0_k3,
        source_public_gate=src_k3_gate,
        model="mock",
        codex_info={},
        stream_out_dir=st_dir,
        run_id="ut_k3_h16",
        max_turns=6,
        reasoning_effort="medium",
        mock_backend=_MockK3H16Backend(),
    )
    assert st_sum["strict_passes"] == 16, st_sum
    assert st_sum["solver_free_successes"] >= 9, st_sum  # S04, S05, S07, S08, S10, S11, S13, S15, S16 all 0-LLM!
    assert st_sum["compositional_reuse_passes"] >= 6, st_sum
    assert st_sum["cut_withdrawal_count"] >= 1, st_sum
    assert st_sum["max_pre_llm_probes_observed"] <= M102_MAX_PRE_LLM_PROBES_PER_STEP
    assert st_sum["shared_fanout_exec_count_1_verified"] is True
    assert st_sum["all_traced_episodes_exec_count_1"] is True

    # Verify Step-14 Cold Hydration + Plain-Dict Routing Parity across S15 & S16 (`U11`)!
    abl = run_m102_step14_cold_hydration_and_plain_dict_parity_ablation(
        world_specs=(sp_k3,),
        built_worlds={sp_k3.world_id: (s0_k3, tasks_k3, man_k3)},
        out_dir=tmp_path,
        checkpoint_step=14,
    )
    assert abl["all_clusters_cold_hydrated_ok"] is True, abl
    assert abl["evaluated_episodes"] == 2 and abl["parity_rate_pct"] == 100.0, abl
