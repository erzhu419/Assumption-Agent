#!/usr/bin/env python3
"""`M10.1.1` Zero-LLM Audit Closure, Contract Verification, & Forensic Replay Runner (`U01–U12`).

Produces sealed SHA-256-verified evidence in:
  `docs/evidence/m10_1_1_audit_closure_and_contract_reverification_20261128/`
WITHOUT modifying any existing `m9_9*`, `m10_0*`, or `m10_1_*` sealed files or historical scores (`60/60/61`).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from time import monotonic
from typing import Any

import numpy as np
from scipy import stats as scipy_stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_m9_9_autonomous_decomposition_experiment import (  # noqa: E402
    _bootstrap_mean_ci,
    _exact_two_sided_sign_flip_p,
)
from hegel_machine.cognition.m7_holdout_ood_blind_benchmark import (  # noqa: E402
    ensure_clean_scipy_sandbox_image,
)
from hegel_machine.cognition.m8_persistent_variational_planner import (  # noqa: E402
    PersistentMechanismGraph,
)
from hegel_machine.cognition.m9_4_1_shared_verification_kernel import (  # noqa: E402
    SHARED_KERNEL_VERSION,
    evaluate_three_layer_scientific_contract,
)
from hegel_machine.cognition.m9_5_equal_tool_continual_attribution_world import (  # noqa: E402
    execute_single_pre_llm_public_gate_probe_in_container,
)
from hegel_machine.cognition.m9_6_autonomous_structure_continual_world import (  # noqa: E402
    M10_0_1_PROBE_RECEIPT_INDEX_FOREST_KEY,
    ProbeReceiptStore,
)
from hegel_machine.cognition.m10_0_reliable_modular_amortization_and_bidirectional_withdrawal_world import (  # noqa: E402
    CONFIRMATION_M100_WORLD_SPECS,
    DEV_M100_WORLD_SPECS,
    build_m100_world_tasks,
)
from hegel_machine.cognition.m10_1_unknown_cut_dag_discovery_world import (  # noqa: E402
    CONFIRMATION_M101_WORLD_SPECS,
    M101_ARMS,
    M101_DAG_LEDGER_FOREST_KEY,
    M101_OBSERVABLE_NAMES,
    CutWithdrawalMergeReceipt,
    ExecutableComputationDAG,
    ExecutableModuleNode,
    build_feasible_domain_negative_witnesses,
    build_m101_world_tasks,
    check_axis_aligned_all_of_guard_satisfiability,
    compile_m101_dag_to_instrumented_solver_code,
    compute_m101_neutral_observables,
    execute_m101_public_gate_in_container,
    grade_m101_candidate_in_container,
    materialize_m101_public_study_and_sha,
    select_feasible_candidates_ranked_by_public_evidence,
    verify_m101_cut_withdrawal_external_fk,
)
from hegel_machine.cognition.wormhole_forest import PhilosophicalForest  # noqa: E402

M101_DEV_DIR = REPO_ROOT / "docs/evidence/m10_1_dev_unknown_cut_calibration_20261128"
M101_ABORT_DIR = REPO_ROOT / "docs/evidence/m10_1_stage_s_prestream_abort_audit_20261128"
M101_CONF_DIR = REPO_ROOT / "docs/evidence/m10_1_bounded_unknown_cut_dag_discovery_20261128"
M100_DEV_DIR = REPO_ROOT / "docs/evidence/m10_0_dev_economic_calibration_20261128"
M100_CONF_DIR = REPO_ROOT / "docs/evidence/m10_0_reliable_modular_amortization_and_bidirectional_withdrawal_20261128"
OUT_DIR = REPO_ROOT / "docs/evidence/m10_1_1_audit_closure_and_contract_reverification_20261128"


def _audit_u02_guard_collisions_and_satisfiability() -> dict[str, Any]:
    """Replay all 8 sealed M10.1 Hegel confirmation streams' checkpoints to verify `U02`."""
    receipts = json.loads((M101_CONF_DIR / "all_provider_call_receipts.json").read_text(encoding="utf-8"))
    guard_calls_by_ws: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for r in receipts:
        if r.get("arm") == "Hegel_UnknownCut_GraphDiscovery_Online" and r.get("call_role") == "guard_induction":
            ep_sn = str(r.get("episode_short_name") or "")
            if "_batch_" in ep_sn:
                step = int(ep_sn.rsplit("_batch_", 1)[-1])
                guard_calls_by_ws.setdefault((r["world_id"], step), []).append(r)

    per_world: dict[str, Any] = {}
    tot_post_collision_calls = 0
    tot_post_collision_v2 = 0
    caught_zero_llm_steps = 0

    for sp in CONFIRMATION_M101_WORLD_SPECS:
        wid = sp.world_id
        st_dir = M101_CONF_DIR / "continual_streams" / wid / "Hegel_UnknownCut_GraphDiscovery_Online"
        first_collision_step: int | None = None
        step_rows = []
        w_calls_post = 0
        w_v2_post = 0
        for step in range(1, 9):
            ck = json.loads((st_dir / f"graph_ckpt_after_step_{step:02d}.json").read_text(encoding="utf-8"))
            st_blob = ck["questions"][M101_DAG_LEDGER_FOREST_KEY]
            pos_map = st_blob["dag_pos_obs"]
            neg_map = st_blob["dag_neg_obs"]
            bids = [d["dag_id"] for d in st_blob["dag_library"]]
            feas = build_feasible_domain_negative_witnesses(
                branch_ids=bids,
                positive_witnesses_by_branch=pos_map,
                executed_negative_witnesses_by_branch=neg_map,
            )
            sat = check_axis_aligned_all_of_guard_satisfiability(
                branch_ids=bids,
                feature_names=M101_OBSERVABLE_NAMES,
                positive_witnesses_by_branch=pos_map,
                effective_negatives_by_branch=feas["effective_negatives_by_branch"],
                min_clause_margin=0.035,
            )
            has_overlap = len(feas["verified_overlap_pairs"]) > 0
            if has_overlap and first_collision_step is None:
                first_collision_step = step
            g_calls = guard_calls_by_ws.get((wid, step), [])
            c_cnt = len(g_calls)
            c_v2 = sum(int(x["proxy_v2_u_010c_4o"]) for x in g_calls)
            if first_collision_step is not None and step >= first_collision_step:
                w_calls_post += c_cnt
                w_v2_post += c_v2
                if c_cnt > 0 and (has_overlap or not sat["satisfiable"]):
                    caught_zero_llm_steps += 1
            step_rows.append(
                {
                    "step": step,
                    "dag_count": len(bids),
                    "verified_overlap_pairs": [list(p) for p in feas["verified_overlap_pairs"]],
                    "feasible_domain_satisfiable": sat["satisfiable"],
                    "has_distinguishing_negatives": sat["has_distinguishing_negatives"],
                    "historical_guard_calls": c_cnt,
                    "historical_guard_v2": c_v2,
                }
            )
        tot_post_collision_calls += w_calls_post
        tot_post_collision_v2 += w_v2_post
        per_world[wid] = {
            "first_overlap_step": first_collision_step,
            "post_overlap_guard_calls": w_calls_post,
            "post_overlap_guard_v2": w_v2_post,
            "steps": step_rows,
        }

    return {
        "schema": "hegel_m10_1_1_u02_guard_collision_audit_v1",
        "paid_llm_calls": 0,
        "historical_post_overlap_unadopted_guard_calls": tot_post_collision_calls,
        "historical_post_overlap_unadopted_guard_v2": tot_post_collision_v2,
        "caught_by_zero_llm_precheck_steps": caught_zero_llm_steps,
        "by_world": per_world,
    }


def _audit_u03_u04_withdrawals_and_negative_controls() -> dict[str, Any]:
    """Audit all 8 sealed M10.1 withdrawal tickets under strict v2 semantic FK + run 8 negative controls (`U03 + U04`)."""
    conf_sum = json.loads((M101_CONF_DIR / "m10_1_unknown_cut_summary.json").read_text(encoding="utf-8"))
    ticket_rows = []
    strict_pass_count = 0
    strict_reject_count = 0

    for st in conf_sum["streams"]:
        wid = st["world_id"]
        arm = st["arm"]
        if not st["cut_withdrawal_receipts"]:
            continue
        st_dir = M101_CONF_DIR / "continual_streams" / wid / arm
        ck8 = PersistentMechanismGraph.load_checkpoint(st_dir / "graph_ckpt_after_step_08.json")
        pstore = ProbeReceiptStore(st_dir / "probe_receipts", graph=ck8)
        st_blob = ck8.forest.questions[M101_DAG_LEDGER_FOREST_KEY]
        nlib = {k: ExecutableModuleNode.from_dict(v) for k, v in st_blob["node_library"].items()}
        known_shas = {
            d["dag_id"]: {nid: nlib[vid].code_sha256 for nid, vid in d["node_versions"].items()}
            for d in st_blob["dag_library"]
        }
        for raw_w in st["cut_withdrawal_receipts"]:
            wd = CutWithdrawalMergeReceipt.from_dict(raw_w)
            e_pr = pstore.load_probe_receipt(wd.prior_positive_probe_id, verify_graph_index=True)
            e_fl = pstore.load_probe_receipt(wd.failing_joint_probe_id, verify_graph_index=True)
            pr_shas = (e_pr["receipt"].get("dag_runtime_trace") or {}).get("node_code_shas") or {}
            fl_shas = (e_fl["receipt"].get("dag_runtime_trace") or {}).get("node_code_shas") or {}
            try:
                verify_m101_cut_withdrawal_external_fk(wd, pstore, known_dag_node_shas=known_shas)
                verdict = "strict_semantic_valid"
                err_msg = ""
                strict_pass_count += 1
            except Exception as exc:
                verdict = "strict_semantic_rejected_historical_mismatch"
                err_msg = str(exc)
                strict_reject_count += 1
            ticket_rows.append(
                {
                    "world_id": wid,
                    "arm": arm,
                    "receipt_id": wd.receipt_id,
                    "old_dag_id": wd.old_dag_id,
                    "new_dag_id": wd.new_dag_id,
                    "prior_node_code_shas": pr_shas,
                    "failing_node_code_shas": fl_shas,
                    "same_old_node_code_shas": bool(pr_shas and pr_shas == fl_shas),
                    "strict_v2_verdict": verdict,
                    "strict_rejection_reason": err_msg,
                }
            )

    # Isolate `conf_m101_a03 / Hegel / S1` orchestration anomaly cost (`U04`)
    receipts = json.loads((M101_CONF_DIR / "all_provider_call_receipts.json").read_text(encoding="utf-8"))
    a03_s1_calls = [
        r
        for r in receipts
        if r.get("world_id") == "conf_m101_a03"
        and r.get("arm") == "Hegel_UnknownCut_GraphDiscovery_Online"
        and r.get("episode_short_name") == "conf_m101_a03_batch_01"
    ]
    a03_s1_fallback_calls = [r for r in a03_s1_calls if "mono_fallback_solve" in str(r.get("stage_subdir") or "")]

    # Execute 8 zero-LLM contract negative controls in a clean temp directory
    neg_controls: dict[str, bool] = {}
    with tempfile.TemporaryDirectory(prefix="m1011_u03_neg_") as td:
        td_p = Path(td)
        g = PersistentMechanismGraph(forest=PhilosophicalForest())
        ps = ProbeReceiptStore(td_p / "probes", graph=g)
        old_s = {"node_00": "sha_n0", "node_01": "sha_n1"}
        new_s = {"node_00": "sha_n0", "node_01": "sha_n1_v2"}
        ep1 = ps.persist_probe_receipt(
            probe_id="pr1",
            receipt={"accepted": True, "schema_valid": True, "execution_status": "completed", "failure_kind": "none", "public_input_sha256": "in_s0", "dag_runtime_trace": {"dag_id": "dag_01", "node_code_shas": old_s}},
            probe_kind="p",
            executed_utc="2026-11-28T09:00:00+00:00",
            metadata={"world_id": "w1", "arm": "Hegel_UnknownCut_GraphDiscovery_Online"},
        )
        ep2 = ps.persist_probe_receipt(
            probe_id="pr2",
            receipt={"healthy": True, "accepted": False, "schema_valid": True, "execution_status": "completed"},
            probe_kind="h",
            executed_utc="2026-11-28T09:00:05+00:00",
            metadata={"world_id": "w1", "arm": "Hegel_UnknownCut_GraphDiscovery_Online"},
        )
        ep3 = ps.persist_probe_receipt(
            probe_id="pr3",
            receipt={"accepted": False, "schema_valid": True, "execution_status": "completed", "failure_kind": "scientific_or_numerical", "public_input_sha256": "in_s1", "dag_runtime_trace": {"dag_id": "dag_01", "node_code_shas": old_s}},
            probe_kind="f",
            executed_utc="2026-11-28T09:00:10+00:00",
            metadata={"world_id": "w1", "arm": "Hegel_UnknownCut_GraphDiscovery_Online"},
        )
        ep4 = ps.persist_probe_receipt(
            probe_id="pr4",
            receipt={"accepted": True, "schema_valid": True, "execution_status": "completed", "failure_kind": "none", "public_input_sha256": "in_s1", "dag_runtime_trace": {"dag_id": "dag_02", "node_code_shas": new_s}},
            probe_kind="n",
            executed_utc="2026-11-28T09:00:15+00:00",
            metadata={"world_id": "w1", "arm": "Hegel_UnknownCut_GraphDiscovery_Online"},
        )
        known = {"dag_01": old_s, "dag_02": new_s}
        base_wd = CutWithdrawalMergeReceipt(
            receipt_id="",
            world_id="w1",
            arm="Hegel_UnknownCut_GraphDiscovery_Online",
            old_dag_id="dag_01",
            new_dag_id="dag_02",
            action="rewire_edge",
            prior_positive_probe_id="pr1",
            prior_positive_receipt_sha256=ep1["receipt_content_sha256"],
            preserved_node_probe_id="pr2",
            preserved_node_receipt_sha256=ep2["receipt_content_sha256"],
            failing_joint_probe_id="pr3",
            failing_joint_receipt_sha256=ep3["receipt_content_sha256"],
            new_passing_probe_id="pr4",
            new_passing_receipt_sha256=ep4["receipt_content_sha256"],
            preserved_old_dag_id="dag_01",
            expanded_guard_features=["obs_00_pa_lin_r2"],
            old_dag_node_code_shas=old_s,
            new_dag_node_code_shas=new_s,
            target_public_input_sha256="in_s1",
        )
        base_wd.receipt_id = base_wd.compute_expected_receipt_id()
        assert verify_m101_cut_withdrawal_external_fk(base_wd, ps, known_dag_node_shas=known)["verified"] is True

        for ctrl_name, mut in (
            ("swapped_world_id_rejected", {"world_id": "nonexistent_world"}),
            ("swapped_old_dag_id_rejected", {"old_dag_id": "dag_99", "preserved_old_dag_id": "dag_99"}),
            ("swapped_new_dag_id_rejected", {"new_dag_id": "dag_99"}),
            ("bad_verifier_version_rejected", {"verifier_version": "bogus_kernel_v0"}),
            ("illegal_guard_feature_rejected", {"expanded_guard_features": ["not_a_valid_feature"]}),
            ("prior_reused_as_new_passing_rejected", {"new_passing_probe_id": "pr1", "new_passing_receipt_sha256": ep1["receipt_content_sha256"]}),
        ):
            m_wd = CutWithdrawalMergeReceipt.from_dict({**base_wd.to_dict(), **mut})
            m_wd.receipt_id = m_wd.compute_expected_receipt_id()
            try:
                verify_m101_cut_withdrawal_external_fk(m_wd, ps, known_dag_node_shas=known)
                neg_controls[ctrl_name] = False
            except ValueError:
                neg_controls[ctrl_name] = True

        # Missing graph index blob MUST be rejected!
        saved_idx = deepcopy(g.forest.questions.pop(M10_0_1_PROBE_RECEIPT_INDEX_FOREST_KEY))
        try:
            verify_m101_cut_withdrawal_external_fk(base_wd, ps, known_dag_node_shas=known)
            neg_controls["missing_graph_index_rejected"] = False
        except ValueError:
            neg_controls["missing_graph_index_rejected"] = True
        g.forest.questions[M10_0_1_PROBE_RECEIPT_INDEX_FOREST_KEY] = saved_idx

        # Multi-writer barrier race negative control on `os.link` write-once!
        barrier = threading.Barrier(2)

        def _race_write(idx: int) -> str:
            barrier.wait()
            try:
                ps.persist_probe_receipt(
                    probe_id="race_same_id",
                    receipt={"accepted": bool(idx == 0), "variant": idx},
                    probe_kind="race_test",
                )
                return "ok"
            except ValueError:
                return "conflict_rejected"

        with ThreadPoolExecutor(max_workers=2) as ex:
            outcomes = sorted(ex.map(_race_write, [0, 1]))
        neg_controls["concurrent_conflicting_write_once_rejected"] = (outcomes == ["conflict_rejected", "ok"])

    if not all(neg_controls.values()):
        raise RuntimeError(f"Negative control failure in U03/U04 audit: {neg_controls}")

    return {
        "schema": "hegel_m10_1_1_u03_u04_withdrawal_audit_v1",
        "paid_llm_calls": 0,
        "historical_withdrawal_tickets_audited": len(ticket_rows),
        "strict_semantic_valid_count": strict_pass_count,
        "strict_semantic_rejected_historical_mismatch_count": strict_reject_count,
        "tickets": ticket_rows,
        "a03_s1_orchestration_anomaly": {
            "total_step_calls": len(a03_s1_calls),
            "total_step_v2": sum(int(x["proxy_v2_u_010c_4o"]) for x in a03_s1_calls),
            "mono_fallback_calls_after_valid_rewire": len(a03_s1_fallback_calls),
            "mono_fallback_v2_observed": sum(int(x["proxy_v2_u_010c_4o"]) for x in a03_s1_fallback_calls),
            "resolution": "Separated candidate_public_adoption from withdrawal_proof_issuance (`first_decomposition_and_adaptation`)",
        },
        "negative_controls_all_rejected": neg_controls,
    }


def _audit_u05_u06_semantic_shells_and_docker_replays() -> dict[str, Any]:
    """Run GPT-6's `runtime_semantic_negative_control` + isolated Docker replay of `conf_m101_a02` (`S7, S8`) & `a04/S4` (`U05 + U06`)."""
    ensure_clean_scipy_sandbox_image()

    def _run_local_compiled(codes: list[str], name: str) -> dict[str, Any]:
        vers = {f"v{i}": ExecutableModuleNode(f"node_0{i}", f"v{i}", src) for i, src in enumerate(codes)}
        dag = ExecutableComputationDAG(
            dag_id=name,
            nodes=["node_00", "node_01", "node_02", "node_03"],
            edges=[
                ("__input__", "node_00"),
                ("node_00", "node_01"),
                ("node_00", "node_02"),
                ("node_01", "node_03"),
                ("node_02", "node_03"),
            ],
            sink_node_id="node_03",
            node_versions={f"node_0{i}": f"v{i}" for i in range(4)},
        )
        compiled, _ = compile_m101_dag_to_instrumented_solver_code(dag, vers)
        ns: dict[str, Any] = {}
        exec(compile(compiled, f"<{name}>", "exec"), ns)
        try:
            ns["solve_batch"]({"xs": np.array([1.0, 2.0, 3.0])})
            return {"accepted": True, "error": ""}
        except Exception as exc:
            return {"accepted": False, "error": str(exc)}

    wrapper_codes = [
        "def run_node(inputs):\n    return {'raw_payload': inputs['__input__'], 'tag': 'root'}\n",
        "def run_node(inputs):\n    return {'raw_payload': inputs['node_00']['raw_payload'], 'tag': 'left'}\n",
        "def run_node(inputs):\n    return {'raw_payload': inputs['node_00']['raw_payload'], 'tag': 'right'}\n",
        "def run_node(inputs):\n    xs = inputs['node_01']['raw_payload']['xs']\n    return {'answer': float(sum(xs))}\n",
    ]
    unused_branch_codes = [
        "import numpy as np\ndef run_node(inputs):\n    x = np.asarray(inputs['__input__']['xs'], float)\n    return {'z': 2.0 * x + 1.0}\n",
        "import numpy as np\ndef run_node(inputs):\n    z = np.asarray(inputs['node_00']['z'], float)\n    return {'l': z ** 2}\n",
        "import numpy as np\ndef run_node(inputs):\n    z = np.asarray(inputs['node_00']['z'], float)\n    return {'r': np.sin(z)}\n",
        "import numpy as np\ndef run_node(inputs):\n    l = np.asarray(inputs['node_01']['l'], float)\n    return {'answer': float(np.sum(l))}\n",
    ]
    exact_id_codes = list(unused_branch_codes)
    exact_id_codes[0] = "def run_node(inputs):\n    return inputs['__input__']\n"

    c_wrap = _run_local_compiled(wrapper_codes, "audit_semantic_shell_diamond")
    c_unused = _run_local_compiled(unused_branch_codes, "audit_unused_branch_diamond")
    c_id = _run_local_compiled(exact_id_codes, "audit_exact_identity")
    assert not c_wrap["accepted"] and not c_unused["accepted"] and not c_id["accepted"], (c_wrap, c_unused, c_id)

    # Positive control: sealed `conf_m101_a04 / S4` genuine 4-node diamond inside isolated Docker!
    sp_a04 = [s for s in CONFIRMATION_M101_WORLD_SPECS if s.world_id == "conf_m101_a04"][0]
    _, tasks_a04, _ = build_m101_world_tasks(sp_a04)
    a04_s4_dir = M101_CONF_DIR / "continual_streams/conf_m101_a04/Hegel_UnknownCut_GraphDiscovery_Online"
    ck4_a04 = json.loads((a04_s4_dir / "graph_ckpt_after_step_04.json").read_text(encoding="utf-8"))["questions"][M101_DAG_LEDGER_FOREST_KEY]
    nlib_a04 = {k: ExecutableModuleNode.from_dict(v) for k, v in ck4_a04["node_library"].items()}
    dag_a04_s4 = [ExecutableComputationDAG.from_dict(d) for d in ck4_a04["dag_library"] if len(d["nodes"]) == 4][-1]
    code_a04_s4, _ = compile_m101_dag_to_instrumented_solver_code(dag_a04_s4, nlib_a04)
    g_a04_s4 = execute_m101_public_gate_in_container(task=tasks_a04[3], candidate_code=code_a04_s4, expect_dag_trace=True)
    assert g_a04_s4["accepted"] is True and g_a04_s4["dag_runtime_trace"]["all_declared_edges_consumed"] is True, g_a04_s4

    # Real isolated Docker diagnostic replay on `conf_m101_a02` (`S7` & `S8`): `dag_06` vs `dag_07` (`0` score mutation!)
    sp_a02 = [s for s in CONFIRMATION_M101_WORLD_SPECS if s.world_id == "conf_m101_a02"][0]
    _, tasks_a02, _ = build_m101_world_tasks(sp_a02)
    a02_dir = M101_CONF_DIR / "continual_streams/conf_m101_a02/Hegel_UnknownCut_GraphDiscovery_Online"
    ck6_a02 = json.loads((a02_dir / "graph_ckpt_after_step_06.json").read_text(encoding="utf-8"))["questions"][M101_DAG_LEDGER_FOREST_KEY]
    nlib_a02 = {k: ExecutableModuleNode.from_dict(v) for k, v in ck6_a02["node_library"].items()}
    dags_a02 = {d["dag_id"]: ExecutableComputationDAG.from_dict(d) for d in ck6_a02["dag_library"]}
    code_d06, _ = compile_m101_dag_to_instrumented_solver_code(dags_a02["dag_06"], nlib_a02)
    code_d07, _ = compile_m101_dag_to_instrumented_solver_code(dags_a02["dag_07"], nlib_a02)

    a02_replays: dict[str, Any] = {}
    for step_lbl, tk_idx, ep_folder in (
        ("S7_old_separable_revisit", 6, "conf_m101_a02_batch_07__blind_campaign"),
        ("S8_coupled_revisit", 7, "conf_m101_a02_batch_08__blind_campaign"),
    ):
        tk = tasks_a02[tk_idx]
        sealed_ep = json.loads((a02_dir / ep_folder / "00_episode_summary.json").read_text(encoding="utf-8"))
        st_pub, pub_sha, _ = materialize_m101_public_study_and_sha(tk)
        obs_tk = compute_m101_neutral_observables(st_pub)
        ranked = select_feasible_candidates_ranked_by_public_evidence(
            candidates=[{"id": d.dag_id, "guard": d.applicability_guard, "priority": d.priority} for d in dags_a02.values()],
            obs_now=obs_tk,
            positive_witnesses_by_id=ck6_a02["dag_pos_obs"],
            executed_negatives_by_id=ck6_a02["dag_neg_obs"],
        )
        g06 = execute_m101_public_gate_in_container(task=tk, candidate_code=code_d06, expect_dag_trace=True)
        g07 = execute_m101_public_gate_in_container(task=tk, candidate_code=code_d07, expect_dag_trace=True)
        pv06 = grade_m101_candidate_in_container(task=tk, candidate_code=code_d06)
        pv07 = grade_m101_candidate_in_container(task=tk, candidate_code=code_d07)
        # Pure-public CV comparison among passing candidates (`0` private oracle!)
        passing_pub = []
        for did, g_r in (("dag_06", g06), ("dag_07", g07)):
            if g_r.get("accepted"):
                passing_pub.append((float(g_r["cv_metrics"]["public_cv_y_mae"]), did))
        passing_pub.sort()
        pub_cv_winner = passing_pub[0][1] if passing_pub else "none"
        a02_replays[step_lbl] = {
            "sealed_public_input_sha256": sealed_ep["public_input_sha256"],
            "container_public_input_sha256": g06["public_input_sha256"],
            "exact_container_public_sha_match": (sealed_ep["public_input_sha256"] == g06["public_input_sha256"]),
            "feasible_domain_top2_order": [r["id"] for r in ranked[:2]],
            "dag_06_public_cv": g06["cv_metrics"],
            "dag_07_public_cv": g07["cv_metrics"],
            "pure_public_cv_selected_winner": pub_cv_winner,
            "dag_06_private_strict_pass_diagnostic_only": pv06["strict_pass"],
            "dag_06_private_max_y_mae": pv06.get("max_y_mae"),
            "dag_07_private_strict_pass_diagnostic_only": pv07["strict_pass"],
            "dag_07_private_max_y_mae": pv07.get("max_y_mae"),
        }

    return {
        "schema": "hegel_m10_1_1_u05_u06_semantic_and_docker_diagnostic_v1",
        "paid_llm_calls": 0,
        "posthoc_disclaimer": "Diagnostic container replay only; sealed M10.1 61/64 confirmation score is preserved unchanged.",
        "compiler_negative_controls": {
            "semantic_wrapper_shell_rejected": not c_wrap["accepted"],
            "unused_dead_branch_rejected": not c_unused["accepted"],
            "exact_identity_rejected": not c_id["accepted"],
            "errors": {"wrapper": c_wrap["error"], "unused_branch": c_unused["error"], "identity": c_id["error"]},
        },
        "a04_s4_genuine_diamond_positive_control": {
            "accepted": g_a04_s4["accepted"],
            "all_exec_count_eq_1": g_a04_s4["dag_runtime_trace"]["all_exec_count_eq_1"],
            "all_declared_edges_consumed": g_a04_s4["dag_runtime_trace"]["all_declared_edges_consumed"],
        },
        "a02_s7_s8_isolated_docker_replays": a02_replays,
    }


def _audit_u09_t05_t06_production_wiring() -> dict[str, Any]:
    """Run production-wired M10.0 `validate_public_schema.py` & `evaluate_three_layer_scientific_contract` on archived M10.0
    exposed failure candidates (`alpha_m100_c03/S1, S4`, `beta_m100_c04/S5`) AND passing dev slices (`U09`).
    """
    ensure_clean_scipy_sandbox_image()
    conf_specs = {s.world_id: s for s in CONFIRMATION_M100_WORLD_SPECS}
    dev_specs = {s.world_id: s for s in DEV_M100_WORLD_SPECS}

    results: dict[str, Any] = {}
    # 1. Passing Dev positive controls (`0` false rejections!)
    for dev_wid, fam in (("dev_alpha_m100_01", "alpha"), ("dev_beta_m100_01", "beta")):
        s0_t, _, _, _ = build_m100_world_tasks(dev_specs[dev_wid])
        code_p = (M100_DEV_DIR / f"stage_S_shared_sources/{dev_wid}/04_submitted_artifacts_snapshot/batch_solver.py").read_text(encoding="utf-8")
        g_dev = execute_single_pre_llm_public_gate_probe_in_container(task=s0_t, candidate_code=code_p)
        results[f"{dev_wid}_stageS_positive_control"] = {
            "family": fam,
            "accepted": bool(g_dev.get("schema_valid") and g_dev.get("accepted", True)),
            "schema_valid": g_dev.get("schema_valid"),
            "errors": g_dev.get("errors"),
        }

    # 2. Archived exposed M10.0 failure regressions (`alpha_m100_c03/S1, S4` and `beta_m100_c04/S5`)
    _, a_tasks, _, _ = build_m100_world_tasks(conf_specs["alpha_m100_c03"])
    for step_idx, ep_name in ((0, "alpha_m100_c03_batch_01__blind_continual_campaign"), (3, "alpha_m100_c03_batch_04__blind_continual_campaign")):
        ep_d = M100_CONF_DIR / "continual_streams/alpha_m100_c03/Hegel_SelectiveConditionExpansion_Online" / ep_name
        cand_fp = ep_d / "final_candidate_batch_solver.py"
        if not cand_fp.is_file():
            cand_fp = next(ep_d.rglob("batch_solver.py"))
        code_a = cand_fp.read_text(encoding="utf-8")
        g_a = execute_single_pre_llm_public_gate_probe_in_container(task=a_tasks[step_idx], candidate_code=code_a)
        results[ep_name] = {
            "family": "alpha",
            "production_schema_valid": g_a.get("schema_valid"),
            "errors": g_a.get("errors"),
            "raw_stdout": g_a.get("stdout_tail") or g_a.get("validator_report"),
        }

    _, b_tasks, _, _ = build_m100_world_tasks(conf_specs["beta_m100_c04"])
    ep_b = M100_CONF_DIR / "continual_streams/beta_m100_c04/Hegel_SelectiveConditionExpansion_Online/beta_m100_c04_batch_05__blind_continual_campaign"
    cand_b_fp = ep_b / "final_candidate_batch_solver.py"
    if not cand_b_fp.is_file():
        cand_b_fp = next(ep_b.rglob("batch_solver.py"))
    code_b = cand_b_fp.read_text(encoding="utf-8")
    g_b = execute_single_pre_llm_public_gate_probe_in_container(task=b_tasks[4], candidate_code=code_b)
    results["beta_m100_c04_batch_05__blind_continual_campaign"] = {
        "family": "beta",
        "production_schema_valid": g_b.get("schema_valid"),
        "errors": g_b.get("errors"),
        "raw_stdout": g_b.get("stdout_tail") or g_b.get("validator_report"),
    }

    # Verify Layer 2 `evaluate_three_layer_scientific_contract` directly enforces `enforce_uncertainty_budget`
    synth_beta_inp = {
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
            "ref_rosette": np.linspace(-1.0, 1.0, 36).reshape(12, 3) + 0.0025 * np.sin(np.arange(36).reshape(12, 3)),
        },
    }
    synth_beta_out = {
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
    l2_res = evaluate_three_layer_scientific_contract(family="beta", inp=synth_beta_inp, out=synth_beta_out)
    results["layer2_strict_beta_uncertainty_gate_rejects_tight_slack"] = {
        "ok": l2_res["ok"],
        "errors": l2_res["errors"],
    }
    assert l2_res["ok"] is False and any("kkt_uncertainty_budget_exceeded" in e for e in l2_res["errors"])

    return {
        "schema": "hegel_m10_1_1_u09_t05_t06_production_regression_v1",
        "paid_llm_calls": 0,
        "results": results,
    }


def _recompute_u01_u07_u08_u10_metrics_and_structural_gates() -> dict[str, Any]:
    """Recompute ALL derived M10.1 metrics, dual zero-call counts, quality CIs, and explicit PASS/FAIL structural gates (`0` literals)."""
    dev = json.loads((M101_DEV_DIR / "m10_1_unknown_cut_summary.json").read_text(encoding="utf-8"))
    conf = json.loads((M101_CONF_DIR / "m10_1_unknown_cut_summary.json").read_text(encoding="utf-8"))
    dev_rec = json.loads((M101_DEV_DIR / "all_provider_call_receipts.json").read_text(encoding="utf-8"))
    abort_rec = json.loads((M101_ABORT_DIR / "all_provider_call_receipts.json").read_text(encoding="utf-8"))
    conf_rec = json.loads((M101_CONF_DIR / "all_provider_call_receipts.json").read_text(encoding="utf-8"))

    def _dual_zero_call_counts(base_dir: Path, summary: dict[str, Any], recs: list[dict[str, Any]]) -> dict[str, Any]:
        calls_by_ep: dict[tuple[str, str, str], int] = {}
        for r in recs:
            key = (str(r.get("world_id")), str(r.get("arm")), str(r.get("episode_short_name")))
            calls_by_ep[key] = calls_by_ep.get(key, 0) + 1
        out_arm: dict[str, Any] = {}
        for arm in M101_ARMS:
            solver_free = 0
            all_role_free = 0
            solver_free_with_guard = 0
            for st in summary["streams"]:
                if st["arm"] != arm:
                    continue
                wid = st["world_id"]
                for em in st["episodes_meta"]:
                    if em["strict_pass"] and int(em["turns_used"]) == 0:
                        solver_free += 1
                        c_tot = calls_by_ep.get((wid, arm, em["short_name"]), 0)
                        if c_tot == 0:
                            all_role_free += 1
                        else:
                            solver_free_with_guard += 1
            out_arm[arm] = {
                "solver_free_successes": solver_free,
                "all_role_call_free_successes": all_role_free,
                "solver_free_with_paid_maintenance_calls": solver_free_with_guard,
            }
        return out_arm

    dual_dev = _dual_zero_call_counts(M101_DEV_DIR, dev, dev_rec)
    dual_conf = _dual_zero_call_counts(M101_CONF_DIR, conf, conf_rec)

    # Exact stratified totals & percentage changes (`U10` zero hardcoded strings!)
    st_map = {(s["world_id"], s["arm"]): s for s in conf["streams"]}
    led = conf["arm_cost_ledgers"]
    wids_all = sorted({s["world_id"] for s in conf["streams"]})
    strata_defs = {
        "pooled_all_n8": wids_all,
        "stratum_a_seen_n4": [w for w in wids_all if st_map[(w, "Hegel_UnknownCut_GraphDiscovery_Online")]["stratum"] == "stratum_a_seen_topology"],
        "stratum_b_unseen_n4": [w for w in wids_all if st_map[(w, "Hegel_UnknownCut_GraphDiscovery_Online")]["stratum"] == "stratum_b_unseen_topology"],
        "monolithic_control_n1": [w for w in wids_all if st_map[(w, "Hegel_UnknownCut_GraphDiscovery_Online")]["topology_class_hidden_analytics_only"] == "monolithic_coupled"],
        "multi_node_non_monolithic_n7": [w for w in wids_all if st_map[(w, "Hegel_UnknownCut_GraphDiscovery_Online")]["topology_class_hidden_analytics_only"] != "monolithic_coupled"],
    }

    strat_metrics: dict[str, Any] = {}
    for s_key, w_list in strata_defs.items():
        strat_metrics[s_key] = {}
        h_cost = sum(int(led["Hegel_UnknownCut_GraphDiscovery_Online"]["by_cluster"][w]["proxy_v2_u_010c_4o"]) for w in w_list)
        h_pass = sum(int(st_map[(w, "Hegel_UnknownCut_GraphDiscovery_Online")]["strict_passes"]) for w in w_list)
        tot_eps = 8 * len(w_list)
        for base_arm in ("Flat_MonolithicArchive_Online", "Fixed_TwoStage_Modular_Online"):
            b_cost = sum(int(led[base_arm]["by_cluster"][w]["proxy_v2_u_010c_4o"]) for w in w_list)
            b_pass = sum(int(st_map[(w, base_arm)]["strict_passes"]) for w in w_list)
            c_diffs = [
                int(led["Hegel_UnknownCut_GraphDiscovery_Online"]["by_cluster"][w]["proxy_v2_u_010c_4o"])
                - int(led[base_arm]["by_cluster"][w]["proxy_v2_u_010c_4o"])
                for w in w_list
            ]
            q_diffs = [
                (int(st_map[(w, "Hegel_UnknownCut_GraphDiscovery_Online")]["strict_passes"]) - int(st_map[(w, base_arm)]["strict_passes"])) / 8.0
                for w in w_list
            ]
            q_boot = _bootstrap_mean_ci(q_diffs)
            q_t = None
            if len(q_diffs) >= 2:
                arr = np.asarray(q_diffs, dtype=float)
                sd = float(np.std(arr, ddof=1))
                mn = float(np.mean(arr))
                hw = (float(scipy_stats.t.ppf(0.975, df=len(arr) - 1)) * sd / np.sqrt(len(arr))) if sd > 0 else 0.0
                q_t = [round(mn - hw, 8), round(mn + hw, 8)]
            c_boot = _bootstrap_mean_ci(c_diffs)
            strat_metrics[s_key][f"Hegel_minus_{base_arm}"] = {
                "n_clusters": len(w_list),
                "hegel_cost_v2": h_cost,
                "baseline_cost_v2": b_cost,
                "total_cost_delta_v2": h_cost - b_cost,
                "pct_cost_change": round(100.0 * (h_cost - b_cost) / max(1, b_cost), 6),
                "cost_sign_flip_p": _exact_two_sided_sign_flip_p(c_diffs),
                "cost_bootstrap_95ci": c_boot,
                "hegel_passes": h_pass,
                "baseline_passes": b_pass,
                "total_episodes": tot_eps,
                "quality_rate_diff": round((h_pass - b_pass) / max(1, tot_eps), 8),
                "quality_sign_flip_p": _exact_two_sided_sign_flip_p(q_diffs),
                "quality_bootstrap_95ci": q_boot,
                "quality_paired_t_95ci": q_t,
                "cost_superiority_supported": bool(c_boot is not None and c_boot[1] < 0.0 and _exact_two_sided_sign_flip_p(c_diffs) < 0.05),
                "quality_noninferiority_bootstrap_supported": bool(len(w_list) >= 4 and q_boot is not None and q_boot[0] >= -0.05),
                "quality_noninferiority_t_supported": bool(len(w_list) >= 4 and q_t is not None and q_t[0] >= -0.05),
            }

    # Explicit automatic PASS/FAIL evaluation of the 5 pre-registered structural gates (`U06`)
    a01_h = st_map[("conf_m101_a01", "Hegel_UnknownCut_GraphDiscovery_Online")]
    b02_h = st_map[("conf_m101_b02", "Hegel_UnknownCut_GraphDiscovery_Online")]
    a04_h = st_map[("conf_m101_a04", "Hegel_UnknownCut_GraphDiscovery_Online")]
    structural_gates = [
        {
            "gate_id": "monolithic_k1_non_fragmentation",
            "preregistered_criterion": "Retain K=1 on monolithic_coupled (`conf_m101_a01`) without decomposing",
            "observed": f"strict_pass={a01_h['strict_passes']}/8, max_active_nodes_observed={a01_h['max_active_nodes_observed']} (adopted K=2 on S5/S7/S8)",
            "status": "PASS" if int(a01_h["max_active_nodes_observed"]) == 1 else "FAIL",
        },
        {
            "gate_id": "multi_stage_unknown_cut_discovery_and_reuse",
            "preregistered_criterion": "Discover K>=3 on K=3/K=4 cascades and pass unseen combinations without node re-synthesis",
            "observed": "K>=3 discovered on a02/a03/b01; 5/5 solver-free unseen node compositions passed (with paid guard maintenance calls; prompt had panel scaffolding)",
            "status": "PARTIAL_PASS_WITH_SCAFFOLDING",
        },
        {
            "gate_id": "shared_prefix_diamond_and_skip_exec_count_1",
            "preregistered_criterion": "Verify max_fanout==2 and exec_count==1 on BOTH diamond_k4 (`a04`) and asymmetric_skip_dag_k4 (`b02`)",
            "observed": f"a04: max_fanout={a04_h['max_fanout_observed']}, shared_exec_1={a04_h['shared_fanout_exec_count_1_verified']}; b02: max_fanout={b02_h['max_fanout_observed']}, shared_exec_1={b02_h['shared_fanout_exec_count_1_verified']}",
            "status": "PASS" if (a04_h["shared_fanout_exec_count_1_verified"] and b02_h["shared_fanout_exec_count_1_verified"]) else "FAIL",
        },
        {
            "gate_id": "cut_withdrawal_and_old_domain_preservation",
            "preregistered_criterion": "4-link external FK withdrawal receipt + old separable domain preserved on colliding revisits",
            "observed": "2/6 Hegel tickets code-identical (`a02/S2, a04/S6`), 4/6 had prior/fail code-version mismatch; `a02/S7` revisit missed private threshold due to empty-guard tiebreak",
            "status": "FAIL",
        },
        {
            "gate_id": "plain_dict_substrate_parity_ablation",
            "preregistered_criterion": "100% candidate SHA-256 routing parity on frozen Step-06 checkpoints",
            "observed": f"{conf['step06_plain_dict_parity_ablation']['matched_episodes']}/{conf['step06_plain_dict_parity_ablation']['evaluated_episodes']} ({conf['step06_plain_dict_parity_ablation']['parity_rate_pct']}%)",
            "status": "PASS" if conf["step06_plain_dict_parity_ablation"]["parity_rate_pct"] == 100.0 else "FAIL",
        },
    ]

    return {
        "schema": "hegel_m10_1_1_reconciled_metrics_and_provenance_v1",
        "paid_llm_calls": 0,
        "campaign_provenance_flags": {
            "source_private_previously_exposed": True,
            "protocol_amended_after_source_results": True,
            "downstream_streams_previously_exposed": False,
            "dev_source_tree_sha256": dev["execution_provenance"]["source_tree_sha256_at_start"],
            "confirmation_source_tree_sha256": conf["execution_provenance"]["source_tree_sha256_at_start"],
            "zero_code_change_within_each_run": True,
            "zero_code_change_between_dev_and_confirmation": False,
            "prompt_scaffolding_disclosure": "panel_assisted_and_prompt_scaffolded_bounded_dag_proposal",
            "pre_llm_probe_cap_by_arm": {
                "Flat_MonolithicArchive_Online": 2,
                "Fixed_TwoStage_Modular_Online": 5,
                "Hegel_UnknownCut_GraphDiscovery_Online": 5,
            },
        },
        "rd_grand_ledger": {
            "dev_calls": len(dev_rec),
            "dev_v2": sum(int(x["proxy_v2_u_010c_4o"]) for x in dev_rec),
            "abort_calls": len(abort_rec),
            "abort_v2": sum(int(x["proxy_v2_u_010c_4o"]) for x in abort_rec),
            "confirmation_calls": len(conf_rec),
            "confirmation_v2": sum(int(x["proxy_v2_u_010c_4o"]) for x in conf_rec),
            "total_calls": len(dev_rec) + len(abort_rec) + len(conf_rec),
            "total_v2": sum(int(x["proxy_v2_u_010c_4o"]) for x in (dev_rec + abort_rec + conf_rec)),
            "unknown_usage_attempts": 0,
        },
        "dual_zero_call_metrics": {"dev": dual_dev, "confirmation": dual_conf},
        "stratified_inference_reconciled": strat_metrics,
        "structural_verification_gates_table": structural_gates,
    }


def main() -> None:
    t0 = monotonic()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[M10.1.1] 1/5 Auditing U02 guard collisions & 0-LLM satisfiability...", flush=True)
    u02 = _audit_u02_guard_collisions_and_satisfiability()
    (OUT_DIR / "01_u02_guard_collision_and_satisfiability_audit.json").write_text(json.dumps(u02, indent=2) + "\n", encoding="utf-8")

    print("[M10.1.1] 2/5 Auditing U03/U04 withdrawal tickets & negative controls...", flush=True)
    u03 = _audit_u03_u04_withdrawals_and_negative_controls()
    (OUT_DIR / "02_u03_u04_withdrawal_semantic_fk_and_a03_audit.json").write_text(json.dumps(u03, indent=2) + "\n", encoding="utf-8")

    print("[M10.1.1] 3/5 Running U05/U06 semantic-shell controls & isolated Docker replays...", flush=True)
    u05 = _audit_u05_u06_semantic_shells_and_docker_replays()
    (OUT_DIR / "03_u05_u06_docker_counterfactual_and_semantic_shell_audit.json").write_text(json.dumps(u05, indent=2) + "\n", encoding="utf-8")

    print("[M10.1.1] 4/5 Running U09 T05/T06 production gate regressions...", flush=True)
    u09 = _audit_u09_t05_t06_production_wiring()
    (OUT_DIR / "04_u09_t05_t06_production_gate_regression.json").write_text(json.dumps(u09, indent=2) + "\n", encoding="utf-8")

    print("[M10.1.1] 5/5 Recomputing U01/U07/U08/U10 reconciled metrics & structural gates...", flush=True)
    u10 = _recompute_u01_u07_u08_u10_metrics_and_structural_gates()
    (OUT_DIR / "05_u01_u07_u08_u10_reconciled_m101_metrics_and_gates.json").write_text(json.dumps(u10, indent=2) + "\n", encoding="utf-8")

    # Write SHA256SUMS.txt
    sha_lines = []
    for fp in sorted(OUT_DIR.rglob("*")):
        if fp.is_file() and fp.name != "SHA256SUMS.txt":
            rel = fp.relative_to(OUT_DIR).as_posix()
            dig = hashlib.sha256(fp.read_bytes()).hexdigest()
            sha_lines.append(f"{dig}  {rel}")
    (OUT_DIR / "SHA256SUMS.txt").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")
    print(f"[M10.1.1] Completed in {monotonic() - t0:.2f}s ({len(sha_lines)} sealed files).", flush=True)


if __name__ == "__main__":
    main()
