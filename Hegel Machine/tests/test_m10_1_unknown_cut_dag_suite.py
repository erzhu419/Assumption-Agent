"""`M10.1` Gate 0–2 Deterministic + Real Docker Verification Suite for Bounded Unknown-Cut DAG Discovery.

Verifies (`0` paid LLM calls):
  1. Topology bounds & anti-fake-graph enforcement (`1 <= K <= 4`, acyclic, `depth <= 4`, `max_fanout <= 2`,
     cycle rejection, dead-orphan rejection, fan-out > 2 rejection, anti-identity-shell rejection).
  2. Real Docker execution (`hegel-clean-scipy:m7v1`) across:
     - `K = 3` chain (`chain_k3_variable_cut`, `dev_m101_c02`): exact pass on matching modes `(A0, B0, C0)` AND
       strict scientific rejection when stage mode shifts to non-nested `B1` (`S01`), THEN exact `0`-LLM pass
       when swapping ONLY `node_01` to `B1` AND composing unseen `(A1, B1, C0)` (`S04`)!
     - `K = 4` diamond shared-prefix (`diamond_k4_shared_prefix`, `dev_m101_c03`): runtime instrumentation proves
       `active_node_count == 4`, `max_fanout == 2`, `shared_fanout_nodes == ['node_00']`, **`exec_count['node_00'] == 1`**
       (executed ONCE, NOT duplicated!), passes BOTH public 56/16 gate AND official private verifier, AND passes
       unseen `(A0, L1, R1, C0)` (`S04`) by recombining `L1` and `R1` nodes at `0` LLM calls!
     - `K = 4` Stratum B held-out skip-DAG (`asymmetric_skip_dag_k4`, `conf_m101_b02`): passes with skip edge
       `node_00 -> node_03` (`fan-out = 2`, `exec_count == 1`).
  3. Strict 3-way external `ProbeReceiptStore` FK on `CutWithdrawalMergeReceipt` (`T01`):
     - Positive control passes; forged `prior_positive_probe_id` is strictly rejected!
  4. Zero topology / role-label leakage in blind task instructions and neutral observable names (`T09`).
"""

from __future__ import annotations

from pathlib import Path
import sys
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from hegel_machine.cognition.m7_holdout_ood_blind_benchmark import ensure_clean_scipy_sandbox_image
from hegel_machine.cognition.m8_persistent_variational_planner import PersistentMechanismGraph
from hegel_machine.cognition.m9_6_autonomous_structure_continual_world import ProbeReceiptStore
from hegel_machine.cognition.m10_1_unknown_cut_dag_discovery_world import (
    BLIND_M101_INSTRUCTION_TEXT,
    CONFIRMATION_M101_WORLD_SPECS,
    CutWithdrawalMergeReceipt,
    DEV_M101_WORLD_SPECS,
    ExecutableComputationDAG,
    ExecutableModuleNode,
    M101_OBSERVABLE_NAMES,
    REF_DEV_NODE_JOIN_READOUT_PY,
    build_m101_world_tasks,
    build_ref_dev_node_pa_code,
    build_ref_dev_node_pb_code,
    build_ref_dev_node_pc_code,
    compile_m101_dag_to_instrumented_solver_code,
    compute_m101_neutral_observables,
    execute_m101_public_gate_in_container,
    grade_m101_candidate_in_container,
    materialize_m101_public_study_and_sha,
    verify_m101_cut_withdrawal_external_fk,
)
from hegel_machine.cognition.wormhole_forest import PhilosophicalForest


def test_m101_gate0_topology_bounds_and_neutral_blindness() -> None:
    # 1. Legal K=1 monolithic
    d_mono = ExecutableComputationDAG(
        dag_id="dag_00",
        nodes=["node_00"],
        edges=[("__input__", "node_00")],
        sink_node_id="node_00",
        node_versions={"node_00": "nv_00"},
    )
    info = d_mono.validate_topology()
    assert info["active_node_count"] == 1 and info["depth"] == 1 and info["max_fanout"] == 0

    # 2. Legal K=4 diamond (`fan-out = 2`, `depth = 3`)
    d_dia = ExecutableComputationDAG(
        dag_id="dag_01",
        nodes=["node_00", "node_01", "node_02", "node_03"],
        edges=[
            ("__input__", "node_00"),
            ("__input__", "node_01"),
            ("__input__", "node_02"),
            ("__input__", "node_03"),
            ("node_00", "node_01"),
            ("node_00", "node_02"),
            ("node_01", "node_03"),
            ("node_02", "node_03"),
        ],
        sink_node_id="node_03",
        node_versions={"node_00": "nv_00", "node_01": "nv_01", "node_02": "nv_02", "node_03": "nv_03"},
    )
    info_d = d_dia.validate_topology()
    assert info_d["active_node_count"] == 4 and info_d["depth"] == 3 and info_d["max_fanout"] == 2
    assert info_d["shared_fanout_nodes"] == ["node_00"] and info_d["has_branching_dag"] is True

    # 3. Illegal: cycle
    d_cyc = ExecutableComputationDAG(
        dag_id="dag_bad_cyc",
        nodes=["node_00", "node_01"],
        edges=[("__input__", "node_00"), ("node_00", "node_01"), ("node_01", "node_00")],
        sink_node_id="node_01",
        node_versions={"node_00": "nv_00", "node_01": "nv_01"},
    )
    with pytest.raises(ValueError, match="cycle"):
        d_cyc.validate_topology()

    # 4. Illegal: fan-out = 3 (> 2)
    d_fo3 = ExecutableComputationDAG(
        dag_id="dag_bad_fo3",
        nodes=["node_00", "node_01", "node_02", "node_03"],
        edges=[
            ("__input__", "node_00"),
            ("node_00", "node_01"),
            ("node_00", "node_02"),
            ("node_00", "node_03"),
            ("node_01", "node_03"),
            ("node_02", "node_03"),
        ],
        sink_node_id="node_03",
        node_versions={"node_00": "nv_00", "node_01": "nv_01", "node_02": "nv_02", "node_03": "nv_03"},
    )
    with pytest.raises(ValueError, match="fan-out"):
        d_fo3.validate_topology()

    # 5. Illegal: > 4 active nodes
    d_5 = ExecutableComputationDAG(
        dag_id="dag_bad_5",
        nodes=[f"node_0{i}" for i in range(5)],
        edges=[("__input__", "node_00"), ("node_00", "node_01"), ("node_01", "node_02"), ("node_02", "node_03"), ("node_03", "node_04")],
        sink_node_id="node_04",
        node_versions={f"node_0{i}": f"nv_0{i}" for i in range(5)},
    )
    with pytest.raises(ValueError, match="active node count"):
        d_5.validate_topology()

    # 6. Zero topology/role leakage in blind instruction text & neutral observable names
    low = BLIND_M101_INSTRUCTION_TEXT.lower()
    for forbidden in ("diamond", "chain_k3", "chain_k4", "skip_dag", "monolithic", "upstream", "downstream", "calibrate_upstream", "solve_downstream"):
        assert forbidden not in low
    assert all(nm.startswith("obs_") for nm in M101_OBSERVABLE_NAMES)
    assert len(DEV_M101_WORLD_SPECS) == 4
    assert len(CONFIRMATION_M101_WORLD_SPECS) == 8
    assert sum(1 for s in CONFIRMATION_M101_WORLD_SPECS if s.stratum == "stratum_a_seen_topology") == 4
    assert sum(1 for s in CONFIRMATION_M101_WORLD_SPECS if s.stratum == "stratum_b_unseen_topology") == 4


def test_m101_gate1_three_way_external_probe_fk_on_cut_withdrawal(tmp_path: Path) -> None:
    forest = PhilosophicalForest()
    graph = PersistentMechanismGraph(forest=forest)
    pstore = ProbeReceiptStore(tmp_path / "probe_receipts", graph=graph)

    e1 = pstore.persist_probe_receipt(
        probe_id="p_prior_01",
        receipt={"accepted": True, "execution_status": "completed"},
        probe_kind="joint_task_probe",
        executed_utc="2026-11-28T06:00:00+00:00",
    )
    e2 = pstore.persist_probe_receipt(
        probe_id="p_pres_02",
        receipt={"healthy": True, "execution_status": "completed"},
        probe_kind="direct_component_probe",
        executed_utc="2026-11-28T06:00:05+00:00",
    )
    e3 = pstore.persist_probe_receipt(
        probe_id="p_fail_03",
        receipt={"accepted": False, "execution_status": "completed"},
        probe_kind="joint_task_probe",
        executed_utc="2026-11-28T06:00:10+00:00",
    )
    e4 = pstore.persist_probe_receipt(
        probe_id="p_new_04",
        receipt={"accepted": True, "execution_status": "completed"},
        probe_kind="joint_task_probe",
        executed_utc="2026-11-28T06:00:20+00:00",
    )
    wd = CutWithdrawalMergeReceipt(
        receipt_id="",
        world_id="dev_m101_c02",
        old_dag_id="dag_01",
        new_dag_id="dag_02",
        action="merge_nodes",
        prior_positive_probe_id="p_prior_01",
        prior_positive_receipt_sha256=e1["receipt_content_sha256"],
        preserved_node_probe_id="p_pres_02",
        preserved_node_receipt_sha256=e2["receipt_content_sha256"],
        failing_joint_probe_id="p_fail_03",
        failing_joint_receipt_sha256=e3["receipt_content_sha256"],
        new_passing_probe_id="p_new_04",
        new_passing_receipt_sha256=e4["receipt_content_sha256"],
        preserved_old_dag_id="dag_01",
        expanded_guard_features=["obs_00_pa_lin_r2", "obs_14_tel_ch0"],
    )
    wd.receipt_id = wd.compute_expected_receipt_id()
    assert verify_m101_cut_withdrawal_external_fk(wd, pstore)["verified"] is True

    # Negative control: forged nonexistent prior_positive_probe_id MUST fail!
    wd_bad = CutWithdrawalMergeReceipt.from_dict({**wd.to_dict(), "prior_positive_probe_id": "audit_no_such_probe"})
    wd_bad.receipt_id = wd_bad.compute_expected_receipt_id()
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_m101_cut_withdrawal_external_fk(wd_bad, pstore)


def test_m101_gate2_real_docker_chain_k3_diamond_k4_exec_count_1_and_anti_identity_shell() -> None:
    ensure_clean_scipy_sandbox_image()

    # 1. Test K=3 chain (`dev_m101_c02`): verify non-nested rejection on B1 shift + 0-LLM compositional reuse on S04 (A1, B1, C0)!
    sp_k3 = DEV_M101_WORLD_SPECS[1]
    s0_k3, tasks_k3, _ = build_m101_world_tasks(sp_k3)
    st0, pub0, _ = materialize_m101_public_study_and_sha(s0_k3)
    obs0 = compute_m101_neutral_observables(st0)
    assert len(obs0) == len(M101_OBSERVABLE_NAMES)

    n_lib = {
        "nv_a0": ExecutableModuleNode("node_00", "nv_a0", build_ref_dev_node_pa_code("A0")),
        "nv_a1": ExecutableModuleNode("node_00", "nv_a1", build_ref_dev_node_pa_code("A1")),
        "nv_b0": ExecutableModuleNode("node_01", "nv_b0", build_ref_dev_node_pb_code("B0")),
        "nv_b1": ExecutableModuleNode("node_01", "nv_b1", build_ref_dev_node_pb_code("B1")),
        "nv_rd": ExecutableModuleNode("node_02", "nv_rd", REF_DEV_NODE_JOIN_READOUT_PY),
    }
    dag_k3_base = ExecutableComputationDAG(
        dag_id="dag_k3_00",
        nodes=["node_00", "node_01", "node_02"],
        edges=[("__input__", "node_00"), ("__input__", "node_01"), ("__input__", "node_02"), ("node_00", "node_01"), ("node_01", "node_02")],
        sink_node_id="node_02",
        node_versions={"node_00": "nv_a0", "node_01": "nv_b0", "node_02": "nv_rd"},
    )
    code_k3_base, _ = compile_m101_dag_to_instrumented_solver_code(dag_k3_base, n_lib)
    g_s0 = execute_m101_public_gate_in_container(task=s0_k3, candidate_code=code_k3_base, expect_dag_trace=True)
    assert g_s0["accepted"] is True and g_s0["dag_runtime_trace"]["all_exec_count_eq_1"] is True, g_s0
    pv_s0 = grade_m101_candidate_in_container(task=s0_k3, candidate_code=code_k3_base)
    assert pv_s0["strict_pass"] is True, pv_s0

    # Non-nested check: base `(A0, B0, C0)` MUST FAIL on `S01 (A0, B1, C0)`!
    g_s1_fail = execute_m101_public_gate_in_container(task=tasks_k3[0], candidate_code=code_k3_base, expect_dag_trace=True)
    assert g_s1_fail["accepted"] is False and g_s1_fail["failure_kind"] == "scientific_or_numerical", g_s1_fail

    # Zero-LLM compositional reuse on unseen `S04 (A1, B1, C0)` by recombining `(nv_a1, nv_b1, nv_rd)`!
    dag_k3_s4 = ExecutableComputationDAG(
        dag_id="dag_k3_s4",
        nodes=["node_00", "node_01", "node_02"],
        edges=dag_k3_base.edges,
        sink_node_id="node_02",
        node_versions={"node_00": "nv_a1", "node_01": "nv_b1", "node_02": "nv_rd"},
    )
    code_k3_s4, _ = compile_m101_dag_to_instrumented_solver_code(dag_k3_s4, n_lib)
    g_s4 = execute_m101_public_gate_in_container(task=tasks_k3[3], candidate_code=code_k3_s4, expect_dag_trace=True)
    assert g_s4["accepted"] is True, g_s4
    pv_s4 = grade_m101_candidate_in_container(task=tasks_k3[3], candidate_code=code_k3_s4)
    assert pv_s4["strict_pass"] is True, pv_s4

    # 2. Test K=4 Diamond shared-prefix (`dev_m101_c03`, `S04 = (A0, L1, R1, C0)`) proving `exec_count['node_00'] == 1`!
    sp_dia = DEV_M101_WORLD_SPECS[2]
    _, tasks_dia, _ = build_m101_world_tasks(sp_dia)
    dia_lib = {
        "nv_p0": ExecutableModuleNode("node_00", "nv_p0", build_ref_dev_node_pa_code("A0")),
        "nv_l1": ExecutableModuleNode("node_01", "nv_l1", build_ref_dev_node_pc_code("v1", "L1")),
        "nv_r1": ExecutableModuleNode("node_02", "nv_r1", build_ref_dev_node_pc_code("v2", "R1")),
        "nv_jn": ExecutableModuleNode("node_03", "nv_jn", REF_DEV_NODE_JOIN_READOUT_PY),
    }
    dag_dia = ExecutableComputationDAG(
        dag_id="dag_dia_01",
        nodes=["node_00", "node_01", "node_02", "node_03"],
        edges=[
            ("__input__", "node_00"),
            ("__input__", "node_01"),
            ("__input__", "node_02"),
            ("__input__", "node_03"),
            ("node_00", "node_01"),
            ("node_00", "node_02"),
            ("node_01", "node_03"),
            ("node_02", "node_03"),
        ],
        sink_node_id="node_03",
        node_versions={"node_00": "nv_p0", "node_01": "nv_l1", "node_02": "nv_r1", "node_03": "nv_jn"},
    )
    code_dia, _ = compile_m101_dag_to_instrumented_solver_code(dag_dia, dia_lib)
    g_dia = execute_m101_public_gate_in_container(task=tasks_dia[3], candidate_code=code_dia, expect_dag_trace=True)
    assert g_dia["accepted"] is True, g_dia
    tr_d = g_dia["dag_runtime_trace"]
    assert tr_d["active_node_count"] == 4 and tr_d["max_fanout"] == 2
    assert tr_d["shared_fanout_nodes"] == ["node_00"] and tr_d["exec_counts"]["node_00"] == 1
    pv_dia = grade_m101_candidate_in_container(task=tasks_dia[3], candidate_code=code_dia)
    assert pv_dia["strict_pass"] is True, pv_dia

    # 3. Test Stratum B unseen subclasses: `conf_m101_b01` (K=4 depth-4 chain) and `conf_m101_b02` (K=4 skip-DAG fan-out=2)!
    sp_k4 = [s for s in CONFIRMATION_M101_WORLD_SPECS if s.topology_class == "chain_k4_deep_cascade"][0]
    s0_k4, _, _ = build_m101_world_tasks(sp_k4)
    k4_lib = {
        "nv_a0": ExecutableModuleNode("node_00", "nv_a0", build_ref_dev_node_pa_code("A0")),
        "nv_b0": ExecutableModuleNode("node_01", "nv_b0", build_ref_dev_node_pb_code("B0")),
        "nv_h0": ExecutableModuleNode("node_02", "nv_h0", build_ref_dev_node_pc_code("v1", "H0")),
        "nv_rd": ExecutableModuleNode("node_03", "nv_rd", REF_DEV_NODE_JOIN_READOUT_PY),
    }
    dag_k4 = ExecutableComputationDAG(
        dag_id="dag_k4_00",
        nodes=["node_00", "node_01", "node_02", "node_03"],
        edges=[
            ("__input__", "node_00"), ("__input__", "node_01"), ("__input__", "node_02"), ("__input__", "node_03"),
            ("node_00", "node_01"), ("node_01", "node_02"), ("node_02", "node_03"),
        ],
        sink_node_id="node_03",
        node_versions={"node_00": "nv_a0", "node_01": "nv_b0", "node_02": "nv_h0", "node_03": "nv_rd"},
    )
    code_k4, _ = compile_m101_dag_to_instrumented_solver_code(dag_k4, k4_lib)
    g_k4 = execute_m101_public_gate_in_container(task=s0_k4, candidate_code=code_k4, expect_dag_trace=True)
    assert g_k4["accepted"] is True and g_k4["dag_runtime_trace"]["depth"] == 4, g_k4
    pv_k4 = grade_m101_candidate_in_container(task=s0_k4, candidate_code=code_k4)
    assert pv_k4["strict_pass"] is True, pv_k4

    sp_skip = [s for s in CONFIRMATION_M101_WORLD_SPECS if s.topology_class == "asymmetric_skip_dag_k4"][0]
    s0_skip, _, _ = build_m101_world_tasks(sp_skip)
    dag_skip = ExecutableComputationDAG(
        dag_id="dag_skip_00",
        nodes=["node_00", "node_01", "node_02", "node_03"],
        edges=[
            ("__input__", "node_00"), ("__input__", "node_01"), ("__input__", "node_02"), ("__input__", "node_03"),
            ("node_00", "node_01"), ("node_01", "node_02"), ("node_02", "node_03"), ("node_00", "node_03"),
        ],
        sink_node_id="node_03",
        node_versions={"node_00": "nv_a0", "node_01": "nv_b0", "node_02": "nv_h0", "node_03": "nv_rd"},
    )
    code_skip, _ = compile_m101_dag_to_instrumented_solver_code(dag_skip, k4_lib)
    g_skip = execute_m101_public_gate_in_container(task=s0_skip, candidate_code=code_skip, expect_dag_trace=True)
    assert g_skip["accepted"] is True and g_skip["dag_runtime_trace"]["exec_counts"]["node_00"] == 1, g_skip
    pv_skip = grade_m101_candidate_in_container(task=s0_skip, candidate_code=code_skip)
    assert pv_skip["strict_pass"] is True, pv_skip

    # 4. Anti-identity-shell negative control: identity pass-through node MUST be rejected by instrumented runtime!
    shell_lib = {
        "nv_shell": ExecutableModuleNode("node_00", "nv_shell", "def run_node(inputs):\n    return inputs['__input__']\n"),
        "nv_sink": ExecutableModuleNode("node_01", "nv_sink", "import numpy as np\ndef run_node(inputs):\n    st = inputs['node_00']\n    n = len(st['test']['x'])\n    return {'y_hat': np.zeros(n), 'state_pred': np.zeros((n, 8)), 'diagnostics_vec': np.zeros(4)}\n"),
    }
    dag_shell = ExecutableComputationDAG(
        dag_id="dag_shell_bad",
        nodes=["node_00", "node_01"],
        edges=[("__input__", "node_00"), ("node_00", "node_01")],
        sink_node_id="node_01",
        node_versions={"node_00": "nv_shell", "node_01": "nv_sink"},
    )
    code_shell, _ = compile_m101_dag_to_instrumented_solver_code(dag_shell, shell_lib)
    g_shell = execute_m101_public_gate_in_container(task=s0_k3, candidate_code=code_shell, expect_dag_trace=True)
    assert g_shell["accepted"] is False and g_shell["failure_kind"] == "dag_invariant_violation"


def test_m101_gate2_end_to_end_stream_runner_and_plain_dict_parity(tmp_path: Path) -> None:
    """End-to-end Gate 2 test of `_run_one_m101_world_arm_stream` + `ProbeReceiptStore` + 4-link FK `CutWithdrawalMergeReceipt`
    + `0`-LLM compositional replay (`S04, S05`) + `rewire_edge` withdrawal (`S06`) + colliding revisits (`S07, S08`)
    + Step-06 Plain-Dict Parity Ablation in REAL Docker (`0` paid LLM calls!).
    """
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from scripts.run_m10_1_unknown_cut_dag_experiment import (
        _run_one_m101_world_arm_stream,
        run_m101_step06_plain_dict_routing_parity_ablation,
    )

    ensure_clean_scipy_sandbox_image()
    sp = DEV_M101_WORLD_SPECS[1]  # `dev_m101_c02`: `chain_k3_variable_cut`
    s0_task, tasks, manifest = build_m101_world_tasks(sp)
    st0, pub0_sha, _ = materialize_m101_public_study_and_sha(s0_task)
    obs0 = compute_m101_neutral_observables(st0)

    # Initial monolithic solver for Stage S (`A0, B0, C0`) built from ref nodes
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
    src_mono_code, _ = compile_m101_dag_to_instrumented_solver_code(init_dag, init_nlib)
    src_gate = execute_m101_public_gate_in_container(task=s0_task, candidate_code=src_mono_code, expect_dag_trace=False)
    assert src_gate["accepted"] is True

    # Reference readout node for `C1` (non-linear tanh readout) & cross-coupled rewired readout (`node_00 -> node_02`)
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

    class _MockM101Backend:
        def make_client(self, *, role: str, world_id: str, arm: str, ep_sn: str):
            class _Cli:
                def call_structured(self, *, system_prompt, conversation_turns, output_schema, stage_subdir):
                    if "init_decomp_" in stage_subdir:
                        return {
                            "ok": True,
                            "parsed": {
                                "action": "split_node",
                                "rationale": "Factorize source solver into 3-node chain",
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
                        base = [d for d in dags if len(d["nodes"]) == 3][0]
                        v_by_n = {}
                        for nv in pay["verified_node_versions"]:
                            v_by_n.setdefault(nv["node_id"], []).append(nv["version_id"])
                        if ep_sn.endswith("01"):
                            return {"ok": True, "parsed": {"action": "reuse_subgraph", "rationale": "update node_01", "base_dag_id": base["dag_id"], "new_nodes": base["nodes"], "new_edges": base["edges"], "new_sink_node_id": "node_02", "reused_node_versions": [{"node_id": "node_00", "version_id": base["node_versions"]["node_00"]}, {"node_id": "node_02", "version_id": base["node_versions"]["node_02"]}], "editable_node_ids": ["node_01"]}}
                        if ep_sn.endswith("02"):
                            return {"ok": True, "parsed": {"action": "reuse_subgraph", "rationale": "update node_02", "base_dag_id": base["dag_id"], "new_nodes": base["nodes"], "new_edges": base["edges"], "new_sink_node_id": "node_02", "reused_node_versions": [{"node_id": "node_00", "version_id": base["node_versions"]["node_00"]}, {"node_id": "node_01", "version_id": base["node_versions"]["node_01"]}], "editable_node_ids": ["node_02"]}}
                        if ep_sn.endswith("03"):
                            return {"ok": True, "parsed": {"action": "reuse_subgraph", "rationale": "update node_00", "base_dag_id": base["dag_id"], "new_nodes": base["nodes"], "new_edges": base["edges"], "new_sink_node_id": "node_02", "reused_node_versions": [{"node_id": "node_01", "version_id": base["node_versions"]["node_01"]}, {"node_id": "node_02", "version_id": base["node_versions"]["node_02"]}], "editable_node_ids": ["node_00"]}}
                        # Step 06 interaction -> `rewire_edge` adding `node_00 -> node_02` while reusing A1 & B1!
                        a1_v = sorted(v_by_n["node_00"])[-1]
                        b1_v = sorted(v_by_n["node_01"])[-1]
                        rew_edges = list(base["edges"]) + [["node_00", "node_02"]]
                        return {"ok": True, "parsed": {"action": "rewire_edge", "rationale": "add direct skip edge node_00 -> node_02 for interaction", "base_dag_id": base["dag_id"], "new_nodes": base["nodes"], "new_edges": rew_edges, "new_sink_node_id": "node_02", "reused_node_versions": [{"node_id": "node_00", "version_id": a1_v}, {"node_id": "node_01", "version_id": b1_v}], "editable_node_ids": ["node_02"]}}
                    return {"ok": False}
            return _Cli()

        def solve_in_sandbox(self, *, task, role, world_id, arm, ep_sn, sub_tag):
            if ep_sn.endswith("01"):
                return {"node_01.py": build_ref_dev_node_pb_code("B1")}, {"turns_used": 2}
            if ep_sn.endswith("02"):
                return {"node_02.py": c1_node_code}, {"turns_used": 2}
            if ep_sn.endswith("03"):
                return {"node_00.py": build_ref_dev_node_pa_code("A1")}, {"turns_used": 2}
            return {"node_02.py": c1_rewired_coupled_code}, {"turns_used": 2}

        def induce_guards(self, *, new_id, existing_ids, pos_map, neg_map, feature_names):
            # Deterministic large-margin guard partition using high-spread neutral observables
            all_ids = list(existing_ids) + [new_id]
            guards = {}
            for bid in all_ids:
                pw = pos_map[bid][0]
                clauses = []
                # Use neutral observables separating A0/A1, B0/B1, C0/C1 cleanly
                if pw["obs_00_pa_lin_r2"] < 0.85:
                    clauses.append({"feature_name": "obs_00_pa_lin_r2", "op": "<=", "threshold": 0.85})
                else:
                    clauses.append({"feature_name": "obs_00_pa_lin_r2", "op": ">=", "threshold": 0.85})
                if pw["obs_04_pb_lin_r2"] < 0.85:
                    clauses.append({"feature_name": "obs_04_pb_lin_r2", "op": "<=", "threshold": 0.85})
                else:
                    clauses.append({"feature_name": "obs_04_pb_lin_r2", "op": ">=", "threshold": 0.85})
                if pw["obs_15_tel_ch1"] > 0.95:
                    clauses.append({"feature_name": "obs_15_tel_ch1", "op": ">=", "threshold": 0.95})
                else:
                    clauses.append({"feature_name": "obs_15_tel_ch1", "op": "<=", "threshold": 0.95})
                guards[bid] = {"all_of": clauses}
            return {
                "accepted": True,
                "new_branch_guard": guards[new_id],
                "existing_branch_guards": {bid: guards[bid] for bid in existing_ids},
            }

    st_dir = tmp_path / "continual_streams" / sp.world_id / "Hegel_UnknownCut_GraphDiscovery_Online"
    st_sum = _run_one_m101_world_arm_stream(
        spec=sp,
        arm="Hegel_UnknownCut_GraphDiscovery_Online",
        s_task=s0_task,
        stream_tasks=tasks,
        schedule_manifest=manifest,
        base_v7_forest=PhilosophicalForest(),
        base_v7_archive={},
        source_monolithic_code=src_mono_code,
        source_pub_sha256=pub0_sha,
        source_obs=obs0,
        source_public_gate=src_gate,
        model="mock",
        codex_info={},
        stream_out_dir=st_dir,
        run_id="gate2_mock",
        max_turns=6,
        reasoning_effort="medium",
        mock_backend=_MockM101Backend(),
    )
    assert st_sum["strict_passes"] == 8, st_sum
    assert st_sum["zero_turn_passed_closures"] == 4, st_sum  # S04, S05, S07, S08 all 0-LLM!
    assert st_sum["cut_withdrawal_count"] == 1, st_sum
    assert st_sum["max_active_nodes_observed"] == 3 and st_sum["all_traced_episodes_exec_count_1"] is True
    assert st_sum["shared_fanout_exec_count_1_verified"] is True  # `rewire_edge` added `node_00 -> node_02` -> fanout=2, exec_count==1!
    assert st_sum["probe_receipt_store_count"] >= 8

    # Run Step-06 plain-dict routing parity ablation
    abl = run_m101_step06_plain_dict_routing_parity_ablation(
        world_specs=(sp,),
        built_worlds={sp.world_id: (s0_task, tasks, manifest)},
        out_dir=tmp_path,
    )
    assert abl["evaluated_episodes"] == 2 and abl["parity_rate_pct"] == 100.0, abl

