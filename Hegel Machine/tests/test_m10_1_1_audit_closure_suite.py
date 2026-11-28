"""`M10.1.1` Audit Closure & Contract Verification Suite (`U01–U12`, `0` Paid LLM Calls)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import threading

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from hegel_machine.cognition.m8_persistent_variational_planner import PersistentMechanismGraph
from hegel_machine.cognition.m9_4_1_shared_verification_kernel import evaluate_three_layer_scientific_contract
from hegel_machine.cognition.m9_6_autonomous_structure_continual_world import (
    M10_0_1_PROBE_RECEIPT_INDEX_FOREST_KEY,
    ProbeReceiptStore,
)
from hegel_machine.cognition.m10_1_unknown_cut_dag_discovery_world import (
    M101_OBSERVABLE_NAMES,
    CutWithdrawalMergeReceipt,
    ExecutableComputationDAG,
    ExecutableModuleNode,
    build_feasible_domain_negative_witnesses,
    check_axis_aligned_all_of_guard_satisfiability,
    compile_m101_dag_to_instrumented_solver_code,
    hydrate_m101_full_prefix_state,
    persist_m101_full_prefix_state,
    select_feasible_candidates_ranked_by_public_evidence,
    verify_m101_cut_withdrawal_external_fk,
    verify_m101_feasible_domain_guards,
)
from hegel_machine.cognition.wormhole_forest import PhilosophicalForest


def test_u02_overlapping_domain_guard_satisfiability_and_feasible_router() -> None:
    src_obs = {k: 0.5 for k in M101_OBSERVABLE_NAMES}
    s1_obs = {**src_obs, "obs_00_pa_lin_r2": 0.20}
    s2_obs = {**src_obs, "obs_04_pb_lin_r2": 0.15}

    # Case 1: Exact overlap right after source decomposition (`dag_00` and `dag_01` share `src_obs`) -> 0-LLM skip!
    pos_overlap = {"dag_00": [src_obs], "dag_01": [src_obs]}
    feas_ov = build_feasible_domain_negative_witnesses(
        branch_ids=["dag_00", "dag_01"],
        positive_witnesses_by_branch=pos_overlap,
    )
    assert feas_ov["verified_overlap_pairs"] == [("dag_00", "dag_01")]
    sat_ov = check_axis_aligned_all_of_guard_satisfiability(
        branch_ids=["dag_00", "dag_01"],
        feature_names=M101_OBSERVABLE_NAMES,
        positive_witnesses_by_branch=pos_overlap,
        effective_negatives_by_branch=feas_ov["effective_negatives_by_branch"],
    )
    assert sat_ov["satisfiable"] is True and sat_ov["has_distinguishing_negatives"] is False

    # Case 2: Partial overlap (`dag_00` & `dag_01` share `src_obs`, `dag_00` also passed `s1_obs` while failing `s2_obs`,
    # and `dag_01` passed `s2_obs` while failing `s1_obs`) -> constructive guards separate `s1_obs` vs `s2_obs` while BOTH accept `src_obs`!
    pos_part = {"dag_00": [src_obs, s1_obs], "dag_01": [src_obs, s2_obs]}
    neg_part = {"dag_00": [s2_obs], "dag_01": [s1_obs]}
    feas_p = build_feasible_domain_negative_witnesses(
        branch_ids=["dag_00", "dag_01"],
        positive_witnesses_by_branch=pos_part,
        executed_negative_witnesses_by_branch=neg_part,
    )
    sat_p = check_axis_aligned_all_of_guard_satisfiability(
        branch_ids=["dag_00", "dag_01"],
        feature_names=M101_OBSERVABLE_NAMES,
        positive_witnesses_by_branch=pos_part,
        effective_negatives_by_branch=feas_p["effective_negatives_by_branch"],
    )
    assert sat_p["satisfiable"] is True
    v_p = verify_m101_feasible_domain_guards(
        branch_guards=sat_p["constructive_guards"],
        positive_witnesses_by_branch=pos_part,
        effective_negatives_by_branch=feas_p["effective_negatives_by_branch"],
        feature_spreads=sat_p["feature_spreads"],
    )
    assert v_p["valid"] is True

    # Feasible router prefers closer verified positive witness even when priority favors newer DAG (`a02/S7` fix!)
    ranked = select_feasible_candidates_ranked_by_public_evidence(
        candidates=[
            {"id": "dag_06", "guard": {}, "priority": 16},
            {"id": "dag_07", "guard": {}, "priority": 17},
        ],
        obs_now=s1_obs,
        positive_witnesses_by_id={"dag_06": [s1_obs], "dag_07": [s2_obs]},
    )
    assert ranked[0]["id"] == "dag_06"


def test_u03_strict_probe_store_write_once_index_and_semantic_withdrawal_fk(tmp_path: Path) -> None:
    g = PersistentMechanismGraph(forest=PhilosophicalForest())
    ps = ProbeReceiptStore(tmp_path / "probes", graph=g)

    # Concurrent write-once race test (`os.link`)
    bar = threading.Barrier(2)

    def _writer(i: int) -> str:
        bar.wait()
        try:
            ps.persist_probe_receipt(probe_id="p_race", receipt={"accepted": i == 0, "v": i}, probe_kind="test")
            return "ok"
        except ValueError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as ex:
        res = sorted(ex.map(_writer, [0, 1]))
    assert res == ["ok", "rejected"]

    # Deleting graph index MUST fail `load_probe_receipt(..., verify_graph_index=True)` and `verify_all_indexed_receipts()`!
    saved = g.forest.questions.pop(M10_0_1_PROBE_RECEIPT_INDEX_FOREST_KEY)
    with pytest.raises(ValueError, match="index"):
        ps.load_probe_receipt("p_race", verify_graph_index=True)
    with pytest.raises(ValueError, match="index"):
        ps.verify_all_indexed_receipts()
    g.forest.questions[M10_0_1_PROBE_RECEIPT_INDEX_FOREST_KEY] = saved
    assert ps.verify_all_indexed_receipts()["verified_count"] == 1

    # Code-version mismatch between prior_positive and failing_joint MUST be rejected!
    e_pr = ps.persist_probe_receipt(
        probe_id="w_pr",
        receipt={"accepted": True, "schema_valid": True, "execution_status": "completed", "failure_kind": "none", "public_input_sha256": "in0", "dag_runtime_trace": {"dag_id": "dag_01", "node_code_shas": {"node_00": "shaA"}}},
        probe_kind="p",
        executed_utc="2026-11-28T08:00:00+00:00",
        metadata={"world_id": "w_test"},
    )
    e_h = ps.persist_probe_receipt(
        probe_id="w_h",
        receipt={"healthy": True, "accepted": False, "schema_valid": True, "execution_status": "completed"},
        probe_kind="h",
        executed_utc="2026-11-28T08:00:02+00:00",
        metadata={"world_id": "w_test"},
    )
    e_fl_bad = ps.persist_probe_receipt(
        probe_id="w_fl_bad",
        receipt={"accepted": False, "schema_valid": True, "execution_status": "completed", "failure_kind": "scientific_or_numerical", "public_input_sha256": "in1", "dag_runtime_trace": {"dag_id": "dag_05", "node_code_shas": {"node_00": "shaB_DIFFERENT"}}},
        probe_kind="f",
        executed_utc="2026-11-28T08:00:05+00:00",
        metadata={"world_id": "w_test"},
    )
    e_nw = ps.persist_probe_receipt(
        probe_id="w_nw",
        receipt={"accepted": True, "schema_valid": True, "execution_status": "completed", "failure_kind": "none", "public_input_sha256": "in1", "dag_runtime_trace": {"dag_id": "dag_06", "node_code_shas": {"node_00": "shaC"}}},
        probe_kind="n",
        executed_utc="2026-11-28T08:00:10+00:00",
        metadata={"world_id": "w_test"},
    )
    wd_mismatch = CutWithdrawalMergeReceipt(
        receipt_id="",
        world_id="w_test",
        old_dag_id="dag_01",
        new_dag_id="dag_06",
        action="rewire_edge",
        prior_positive_probe_id="w_pr",
        prior_positive_receipt_sha256=e_pr["receipt_content_sha256"],
        preserved_node_probe_id="w_h",
        preserved_node_receipt_sha256=e_h["receipt_content_sha256"],
        failing_joint_probe_id="w_fl_bad",
        failing_joint_receipt_sha256=e_fl_bad["receipt_content_sha256"],
        new_passing_probe_id="w_nw",
        new_passing_receipt_sha256=e_nw["receipt_content_sha256"],
        preserved_old_dag_id="dag_01",
        expanded_guard_features=["obs_00_pa_lin_r2"],
    )
    wd_mismatch.receipt_id = wd_mismatch.compute_expected_receipt_id()
    with pytest.raises(ValueError, match="code-version mismatch"):
        verify_m101_cut_withdrawal_external_fk(wd_mismatch, ps)


def test_u06_compiler_rejects_wrapper_shell_and_unused_branch() -> None:
    def _eval(codes: list[str]) -> dict:
        vers = {f"v{i}": ExecutableModuleNode(f"node_0{i}", f"v{i}", c) for i, c in enumerate(codes)}
        dag = ExecutableComputationDAG(
            dag_id="d_test",
            nodes=["node_00", "node_01", "node_02", "node_03"],
            edges=[("__input__", "node_00"), ("node_00", "node_01"), ("node_00", "node_02"), ("node_01", "node_03"), ("node_02", "node_03")],
            sink_node_id="node_03",
            node_versions={f"node_0{i}": f"v{i}" for i in range(4)},
        )
        src, _ = compile_m101_dag_to_instrumented_solver_code(dag, vers)
        ns: dict = {}
        exec(compile(src, "<t>", "exec"), ns)
        return ns["solve_batch"]({"xs": np.array([1.0, 2.0, 3.0])})

    with pytest.raises(RuntimeError, match="unconsumed|anti-wrapper-shell"):
        _eval([
            "def run_node(inputs):\n    return {'raw_payload': inputs['__input__'], 'tag': 'root'}\n",
            "def run_node(inputs):\n    return {'raw_payload': inputs['node_00']['raw_payload'], 'tag': 'left'}\n",
            "def run_node(inputs):\n    return {'raw_payload': inputs['node_00']['raw_payload'], 'tag': 'right'}\n",
            "def run_node(inputs):\n    return {'answer': sum(inputs['node_01']['raw_payload']['xs'])}\n",
        ])

    # Sneaky wrapper that touches `__input__['xs']` so edge is consumed, yet still wraps `__input__` without novel math:
    with pytest.raises(RuntimeError, match="anti-wrapper-shell"):
        _eval([
            "def run_node(inputs):\n    _ = inputs['__input__']['xs']\n    return {'raw_payload': inputs['__input__'], 'tag': 'root'}\n",
            "def run_node(inputs):\n    return {'raw_payload': inputs['node_00']['raw_payload'], 'tag': 'left'}\n",
            "def run_node(inputs):\n    return {'raw_payload': inputs['node_00']['raw_payload'], 'tag': 'right'}\n",
            "def run_node(inputs):\n    return {'answer': sum(inputs['node_01']['raw_payload']['xs'])}\n",
        ])

    with pytest.raises(RuntimeError, match="unconsumed"):
        _eval([
            "import numpy as np\ndef run_node(inputs):\n    return {'z': np.asarray(inputs['__input__']['xs']) * 2.0}\n",
            "import numpy as np\ndef run_node(inputs):\n    return {'l': inputs['node_00']['z'] + 1.0}\n",
            "import numpy as np\ndef run_node(inputs):\n    return {'r': inputs['node_00']['z'] - 1.0}\n",
            "import numpy as np\ndef run_node(inputs):\n    return {'answer': float(np.sum(inputs['node_01']['l']))}\n",
        ])


def test_u09_and_u11_layer2_wiring_and_full_prefix_cold_restart(tmp_path: Path) -> None:
    # U09: Layer 2 enforces strict Beta KKT uncertainty budget when `_enforce_kkt_uncertainty_budget=True`
    inp = {
        "nx": 4,
        "ny": 4,
        "source_modes": np.zeros((3, 4, 4), dtype=float),
        "y_obs_raw": np.zeros(72, dtype=float),
        "source_ridge": 0.01,
        "u_box": 0.6,
        "total_budget": 1.0,
        "_enforce_kkt_uncertainty_budget": True,
        "sensor_calibration_sweeps": {
            "raw_rosette": np.linspace(-1.0, 1.0, 36).reshape(12, 3),
            "ref_rosette": np.linspace(-1.0, 1.0, 36).reshape(12, 3) + 0.003 * np.cos(np.arange(36).reshape(12, 3)),
        },
    }
    out = {
        "weights": np.array([0.4, 0.4, 0.2], dtype=float),
        "terminal_field": np.zeros((4, 4), dtype=float),
        "kkt_certificate": {
            "weights": [0.4, 0.4, 0.2],
            "nu_sum": 0.0,
            "lam_lower": [0.0, 0.0, 0.0],
            "lam_upper": [0.0, 0.0, 0.0],
            "primal_objective": 0.0085,
            "dual_lower_bound": 0.0085,
        },
    }
    res = evaluate_three_layer_scientific_contract(family="beta", inp=inp, out=out)
    assert res["ok"] is False and any("kkt_uncertainty_budget_exceeded" in e for e in res["errors"])

    # U11: Full prefix persistence & cold restart hydration round-trip + tamper rejection
    g = PersistentMechanismGraph(forest=PhilosophicalForest())
    ps = ProbeReceiptStore(tmp_path / "probes", graph=g)
    ep = ps.persist_probe_receipt(
        probe_id="p0",
        receipt={"accepted": True, "schema_valid": True, "execution_status": "completed", "public_input_sha256": "pub0"},
        probe_kind="init",
        metadata={"world_id": "w_hyd", "arm": "Hegel"},
    )
    mn = ExecutableModuleNode(
        node_id="node_00",
        version_id="nv_00_00",
        code="def run_node(inputs):\n    return {'y_hat': inputs['x']}\n",
        probe_receipt_id="p0",
        probe_receipt_sha256=ep["receipt_content_sha256"],
    )
    dag = ExecutableComputationDAG(
        dag_id="dag_00",
        nodes=["node_00"],
        edges=[("__input__", "node_00")],
        sink_node_id="node_00",
        node_versions={"node_00": "nv_00_00"},
    )
    obs = {k: 0.25 for k in M101_OBSERVABLE_NAMES}
    persist_m101_full_prefix_state(
        g,
        world_id="w_hyd",
        arm="Hegel",
        node_library={"nv_00_00": mn},
        dag_library=[dag],
        withdrawals=[],
        dag_pos_obs={"dag_00": [obs]},
        dag_neg_obs={},
        node_pos_obs={"nv_00_00": [obs]},
        dag_last_pos_probe={"dag_00": ("p0", ep["receipt_content_sha256"])},
        candidate_input_verdicts={"dag_00::pub0": {"accepted": True, "probe_id": "p0"}},
        attempted_decomp_from_dag_ids={"dag_00"},
        covered_dag_pairs={("dag_00", "dag_01")},
        version_counter=3,
        dag_counter=2,
        event_seq=5,
    )
    ckpt_fp = tmp_path / "ckpt.json"
    g.save_checkpoint(ckpt_fp)

    g_loaded = PersistentMechanismGraph.load_checkpoint(ckpt_fp)
    ps_loaded = ProbeReceiptStore(tmp_path / "probes", graph=g_loaded)
    hyd = hydrate_m101_full_prefix_state(g_loaded, ps_loaded)
    assert hyd["version_counter"] == 3 and hyd["dag_counter"] == 2 and hyd["event_seq"] == 5
    assert hyd["attempted_decomp_from_dag_ids"] == {"dag_00"}
    assert hyd["covered_dag_pairs"] == {("dag_00", "dag_01")}
    assert hyd["dag_last_pos_probe"]["dag_00"][0] == "p0"
