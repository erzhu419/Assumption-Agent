"""Automated Verification Suite for `M9.4.1` (`Gates A–D`: Shared 7-Coeff Kernel, Strict 3-Valued Guards, Identity Chain, AST Guards, Stratified F08, F05)."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

from hegel_machine.cognition import (
    m6_4_executable_wormhole_library as m64,
    m6_8_b_prime_capability_bridge as m68,
    m6_frontier_real_benchmarks as m6f,
    m7_autogenous_forest_pretraining as m7a,
    m9_4_1_shared_verification_kernel as m941k,
    m9_4_continual_mechanism_binding_world as m94,
)
from hegel_machine.cognition.language import DomainViolationError, EvaluationContext
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.wormhole_forest import PhilosophicalForest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_m941_gate_a_shared_kernel_rejects_zeros_and_million_tail_and_isolates_worker() -> None:
    """Verify BOTH `zeros(7)` AND `hw[2:7] = 1_000_000` are rejected by shared kernel & container isolated-worker probe!"""
    import pickle
    import subprocess

    task = m94.M94_CHAIN_ALPHA_SOURCE
    src_fp = (
        REPO_ROOT
        / "docs/evidence/m9_4_continual_mechanism_binding_world_20261126"
        / "stage_S_sources/attempt_01/alpha_s00__polymer_coupon_creep_baseline/04_submitted_artifacts_snapshot/alpha_s00_solver.py"
    )
    src_code = src_fp.read_text(encoding="utf-8")

    # Container-level test of `execute_host_trusted_scientific_probe` (`hegel_host_isolated_worker_scientific_probe_v4`)
    ft = m94.m94_task_to_frontier_task(task)
    cid = m6f.start_detached_container(m7a.CLEAN_SCIPY_SANDBOX_IMAGE, memory_limit="2048m", cpus="2", network_mode="none")
    try:
        with tempfile.TemporaryDirectory(prefix="m941_c01_") as td:
            td_p = Path(td)
            (td_p / "_setup.py").write_text(task.public_setup_py, encoding="utf-8")
            m6f.exec_container_bash(cid, "mkdir -p /holdout_task", timeout_sec=10)
            subprocess.run(["docker", "cp", str(td_p / "_setup.py"), f"{cid}:/tmp/_setup.py"], check=True, timeout=15)
            m6f.exec_container_bash(cid, "python3 -I /tmp/_setup.py && rm -f /tmp/_setup.py", timeout_sec=45)
            sh_r = m6f.exec_container_bash(
                cid,
                "python3 -I -c \"import hashlib,json; ps=['/holdout_task/public_alpha_s00.pkl','/holdout_task/validate_public_schema.py']; "
                "print(json.dumps({p:hashlib.sha256(open(p,'rb').read()).hexdigest() for p in ps}))\"",
                timeout_sec=15,
            )
            pub_shas = json.loads(sh_r["stdout"].strip().splitlines()[-1])

            # 1. Genuine unmutated solver -> PASSES with scoped `verified_claim_ids`!
            (td_p / "good.py").write_text(src_code, encoding="utf-8")
            subprocess.run(["docker", "cp", str(td_p / "good.py"), f"{cid}:/holdout_task/alpha_s00_solver.py"], check=True, timeout=15)
            good_sha = m6f.compute_declared_artifacts_digest(cid, ft.declared_artifacts)
            rec_good = m6f.execute_host_trusted_scientific_probe(
                agent_cid=cid, task=ft, current_candidate_sha256=good_sha, public_setup_files_sha256=pub_shas
            )
            assert rec_good is not None and rec_good["worker_process_isolated"] is True
            assert "full_7coeff_hereditary_handoff" in rec_good["verified_claim_ids"]
            assert rec_good["verified_scope"] != "task_scientific_invariants"

            # 2. Mutation A (`hw = zeros(7)`) -> REJECTED (`None`) by isolated-worker public scientific probe!
            zero_mut = src_code + "\n_orig = solve_alpha_s00\ndef solve_alpha_s00(ds):\n    o = _orig(ds); o['hereditary_weights'] = np.zeros(7); return o\n"
            (td_p / "zero.py").write_text(zero_mut, encoding="utf-8")
            subprocess.run(["docker", "cp", str(td_p / "zero.py"), f"{cid}:/holdout_task/alpha_s00_solver.py"], check=True, timeout=15)
            zero_sha = m6f.compute_declared_artifacts_digest(cid, ft.declared_artifacts)
            assert (
                m6f.execute_host_trusted_scientific_probe(
                    agent_cid=cid, task=ft, current_candidate_sha256=zero_sha, public_setup_files_sha256=pub_shas
                )
                is None
            )

            # 3. Mutation B (`hw[2:7] = 1_000_000`) -> REJECTED by BOTH public scientific probe AND private verifier!
            tail_mut = (
                src_code
                + "\n_orig = solve_alpha_s00\ndef solve_alpha_s00(ds):\n    o = _orig(ds); o['hereditary_weights'] = np.asarray(o['hereditary_weights']).copy(); "
                "o['hereditary_weights'][2:] = 1000000.0; return o\n"
            )
            (td_p / "tail.py").write_text(tail_mut, encoding="utf-8")
            subprocess.run(["docker", "cp", str(td_p / "tail.py"), f"{cid}:/holdout_task/alpha_s00_solver.py"], check=True, timeout=15)
            tail_sha = m6f.compute_declared_artifacts_digest(cid, ft.declared_artifacts)
            assert (
                m6f.execute_host_trusted_scientific_probe(
                    agent_cid=cid, task=ft, current_candidate_sha256=tail_sha, public_setup_files_sha256=pub_shas
                )
                is None
            )
            priv_tail = m94.grade_m94_artifact_map(task, {"/holdout_task/alpha_s00_solver.py": tail_mut})
            assert priv_tail["strict_pass"] is False
    finally:
        m6f.stop_container_safely(cid)


def test_m941_gate_b_guards_binding_identity_ast_f08_and_f05() -> None:
    # 1. Fail-closed 3-valued guard checks
    ag = m7a.ObservablePreconditionGuard(
        guard_id="g_a",
        observable_extractor="overstress_excess_max_history_and_hold",
        op="<=",
        value=0.0,
        applies_to_component_id="linear_viscoelastic_memory",
    )
    assert m7a.evaluate_observable_precondition_guard(ag, {"past_stress": [[100.0]], "stress_hold_0": [100.0]})["state"] == "UNKNOWN"
    assert m7a.evaluate_observable_precondition_guard(ag, {"past_stress": [[float("nan"), 5.0]], "stress_hold_0": [0.2], "yield_stress": 1.0})["state"] == "UNKNOWN"
    assert m7a.evaluate_observable_precondition_guard(ag, {"stress_hold_0": [0.2], "yield_stress": 1.0})["state"] == "UNKNOWN"
    assert m7a.evaluate_observable_precondition_guard(ag, {"past_stress": [[0.2]], "stress_hold_0": [0.2], "yield_stress": 1.0, "residual_plastic_state": [0.3]})["state"] == "FALSE"
    assert m7a.evaluate_observable_precondition_guard(ag, {"past_stress": [[0.2]], "stress_hold_0": [0.2], "yield_stress": 1.0})["state"] == "TRUE"

    # 2. Binding identity & foreign-key enforcement
    graph, _, reg = m94.initialize_continual_mechanism_graph_with_sentinel(PhilosophicalForest())
    with pytest.raises(ValueError):
        m94.render_bound_executable_adapter(
            binding=m94.MechanismExecutableBinding(
                binding_id="b_bad",
                chain_id="alpha",
                branch_tag="C0_base",
                mechanism_id="m_bad",
                mechanism_version=1,
                forest_node_id="n0",
                component_id="c0",
                preserved_component_ids=[],
                violated_component_id=None,
                module_stem="s",
                module_digest="0" * 64,
                function_name="fn",
                qualification_id="hegel_l1_qual_4tuple_v3",
                parameter_mechanism_id="alpha_shared_loadcell_calibration",
                parameter_version=1,
                parameterized_closure_code="def fn(d): return {}\n",
                parameterized_entrypoint="fn",
                parameterized_closure_sha256="f" * 64,
            ),
            target_entrypoint="out_fn",
            param_payload={"stress_gains": [1]*4, "stress_biases": [0]*4},
        )

    # 3. AST domain-guard propagation into lifted Program & cross-tree anti-unification
    s1 = "def f1(x):\n    if x < -1000.0:\n        raise ValueError('bad')\n    return 2.0 * x + 1.0\n"
    s2 = "def f2(x):\n    if x > 500.0:\n        raise ValueError('bad')\n    return 2.0 * x + 3.0\n"
    p1 = m68.lift_python_scientific_expressions_to_programs(s1)[0][1]
    with pytest.raises(DomainViolationError):
        EvaluationContext(Observation({"x_0": -1001.0}, 0.0)).evaluate(p1)

    from hegel_machine.cognition.wormhole_forest import ClaimContract, ObservationMap, ScaleMap, TaskView

    forest, archive, _ = m7a.build_blank_axiomatic_philosophical_forest()
    for nid, tid, sig in (("node_m1", "tree_A01_conservation", "sig:a"), ("node_m2", "tree_A02_deconfounding", "sig:b")):
        forest.register_node(
            node_id=nid,
            tree_id=tid,
            layer_role="mid",
            task_view=TaskView("tv", ["x"], "y", "present_snapshot", []),
            observation_map=ObservationMap("y", temporal_readout_mode="present_snapshot"),
            scale_map=ScaleMap("identity", preserved_queries=["q"], closure_scope="exact"),
            claim_contract=ClaimContract(f"cc_{nid}", "p", [], {}, {}, 1.0, "SUPPORTED_IN_SCOPE", "local_solver_guidance"),
            probe_signatures=[sig],
        )
    archive["node_m1"] = {"tree_id": "tree_A01_conservation", "layer_role": "mid", "signatures": ["sig:a"], "autogenous_wormhole_modules": [{"filename": "m1.py", "python_code": s1}]}
    archive["node_m2"] = {"tree_id": "tree_A02_deconfounding", "layer_role": "mid", "signatures": ["sig:b"], "autogenous_wormhole_modules": [{"filename": "m2.py", "python_code": s2}]}
    wh_res = m7a.discover_cross_tree_operator_wormholes(forest, archive)
    assert isinstance(wh_res, dict) and len(forest.shared_base_subtrees) >= 1
    sub_obj = next(iter(forest.shared_base_subtrees.values()))
    skel_p = m68.Program.from_dict(sub_obj["executable_skeleton_dict"])
    assert skel_p.op == "domain_guard"
    with pytest.raises(DomainViolationError):
        EvaluationContext(Observation({"x_0": -1001.0}, 0.0)).evaluate(skel_p)
    with pytest.raises(DomainViolationError):
        EvaluationContext(Observation({"x_0": 600.0}, 0.0)).evaluate(skel_p)

    # 4. Stratified per-function F08 qualification rejects 24 stable + 1 unseeded public function
    raw_q = "import numpy as np\ndef stable(x): return np.asarray(x, dtype=float) * 2.0\ndef unseeded(x): return np.random.default_rng().normal(size=np.asarray(x).shape)\n"
    ok_a, _, fns_q, pkg_q = m7a.validate_and_decorate_autogenous_python_code(raw_q, "autogenous_prim_strat")
    assert ok_a
    ut_q = "import numpy as np\nfor i in range(24):\n    stable([float(i), 1.0])\nunseeded([1.0, 2.0])\n"
    with patch.object(m7a, "verify_autogenous_module_in_clean_container", lambda **kw: {"passed": True, "telemetry_verified": True}):
        ok_p, rec_p = m7a.issue_4tuple_l1_qualification_receipt(
            sandbox_cid="mock-cid", mod_stem="autogenous_prim_strat", packaged_module_code=pkg_q, synthetic_unit_test_code=ut_q, public_fn_names=fns_q
        )
    assert ok_p is False and rec_p["determinism_and_input_immutability_verified"] is False

    # 5. F05 exact-AND-reachable funnel check
    f_out = m64.classify_source_aware_adoption_funnel(
        episode_summary={"skill_adoption_funnel": {"final_selected_artifact_retention": {"final_retained_modules": ["ns.py"], "entrypoint_reachable_functions": ["ns.fn"], "all_retained_ast_match_source": False}}},
        full_skill_activations=[],
        skill_inspections=[],
        new_source_l2_stems={"ns"},
        new_source_l1_stems=set(),
    )
    assert f_out["any_new_source_exact_retained_and_reachable"] is False
    assert f_out["derivation_provenance_status"] == "adapted_closure_reachable"


def test_m941_gate_c_immediate_attempt_mirror_and_dag_append() -> None:
    import subprocess as _sp

    with tempfile.TemporaryDirectory(prefix="m941_mirror_") as td:
        td_p = Path(td)
        mir_dir = td_p / "immediate_raw_attempts"
        cli = m6f.FrontierCodexClient(
            codex_info={
                "available": True,
                "auth_mode": "chatgpt",
                "executable": "mock_codex",
                "is_windows_exe_under_wsl": False,
                "stage_root_wsl": str(td_p / "stage"),
                "stage_root_cli_str": str(td_p / "stage"),
            },
            model="gpt-6.2-sol",
            timeout_seconds=1,
        )
        cli.immediate_archive_dir = mir_dir  # type: ignore[attr-defined]

        # Simulate a timeout on attempt 1 followed by process crash (exception) on attempt 2:
        # Attempt 1 MUST already be persisted in `immediate_raw_attempts/`!
        call_n = {"n": 0}

        def _fake_run(*a: object, **kw: object) -> object:
            call_n["n"] += 1
            if call_n["n"] == 1:
                raise _sp.TimeoutExpired(cmd="codex", timeout=1, output='{"type":"turn.completed","usage":{"input_tokens":123,"cached_input_tokens":0,"output_tokens":7}}\n')
            raise RuntimeError("simulated_mid_run_kill")

        with patch.object(m6f.subprocess, "run", _fake_run), patch.object(m6f, "sleep", lambda _s: None):
            with pytest.raises(RuntimeError, match="simulated_mid_run_kill"):
                cli.call_structured(
                    stage_subdir="ep_crash",
                    output_schema={"type": "object"},
                    system_prompt="sys",
                    conversation_turns=[{"role": "user", "content": "hi"}],
                )
        mirrored_files = list(mir_dir.glob("*__attempts_*.json"))
        assert len(mirrored_files) == 1
        saved = json.loads(mirrored_files[0].read_text(encoding="utf-8"))
        assert saved["attempt_count"] == 1 and saved["cumulative_usage"]["prompt_tokens"] == 123

        ep_d = td_p / "ep_dag"
        ep_d.mkdir(parents=True, exist_ok=True)
        chk1 = m6f.append_post_episode_verification_to_dag(
            ep_d,
            run_id="r1",
            task_id="t1",
            arm="armA",
            model="gpt-6.2-sol",
            candidate_sha256="b" * 64,
            public_gate_receipt={"ok": True, "verified_claim_ids": ["c1"]},
            verifier_report={"strict_pass": True},
        )
        assert chk1["valid"] is True
        dag_j = json.loads((ep_d / "01_decision_provenance_dag.json").read_text(encoding="utf-8"))
        assert [r["decision_type"] for r in dag_j["records"]] == [
            "D_CANDIDATE_SEALED",
            "D_PUBLIC_SCIENTIFIC_GATE",
            "D_FINAL_PRIVATE_GRADE",
        ]

        # Non-empty existing DAG with non-zero cumulative measurement/compute/token costs MUST preserve exact conservation!
        ep_d2 = td_p / "ep_dag_nonempty"
        ep_d2.mkdir(parents=True, exist_ok=True)
        led0 = m6f.DecisionProvenanceLedger(run_id="r2", task_id="t2", arm="armB", model="gpt-6.2-sol", provider_backend="codex")
        d0 = led0.record_decision(
            decision_type="D0_BOOT",
            parent_decision_id=None,
            evidence_inputs={},
            candidates_evaluated=[],
            chosen_option={"boot": True},
            rejection_reasons={},
            rationale_summary="boot",
            measurement_delta=1.5,
            compute_delta=2.25,
            prompt_tokens_delta=4500,
            cached_tokens_delta=1000,
            completion_tokens_delta=320,
            wall_ms_delta=850.5,
        )
        led0.record_decision(
            decision_type="D1_TURN",
            parent_decision_id=d0,
            evidence_inputs={},
            candidates_evaluated=[],
            chosen_option={"turn": 1},
            rejection_reasons={},
            rationale_summary="turn1",
            measurement_delta=1.0,
            compute_delta=1.75,
            prompt_tokens_delta=6200,
            cached_tokens_delta=4000,
            completion_tokens_delta=510,
            wall_ms_delta=1240.25,
        )
        (ep_d2 / "01_decision_provenance_dag.json").write_text(
            json.dumps({"provenance_verification": m6f.verify_decision_provenance_chain(led0.records), "records": led0.records}, indent=2),
            encoding="utf-8",
        )
        chk2 = m6f.append_post_episode_verification_to_dag(
            ep_d2,
            run_id="r2",
            task_id="t2",
            arm="armB",
            model="gpt-6.2-sol",
            candidate_sha256="c" * 64,
            public_gate_receipt={"ok": True, "verified_claim_ids": ["c2"], "wall_ms": 15.0},
            verifier_report={"strict_pass": True, "verifier_wall_ms": 25.0},
        )
        assert chk2["valid"] is True and chk2["decision_count"] == 5
