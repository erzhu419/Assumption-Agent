#!/usr/bin/env python3
"""End-to-End 32-Way Concurrent `gpt-6.2-sol` Runner for `M9.5` (`8` Independent Confirmation Worlds x `3` Equal-Tool Arms).

Design & Audit Guarantees (`Hegel-M95-EqualTool-Continual`):
  - **`32`-Way Parallel GPT CLI Pool**: Global `BoundedSemaphore(32)` + `24` concurrent continual streams (`8` independent
    confirmation worlds x `3` arms x `8` steps = `192` stream episodes + `8` independent blind `Stage S` sources = `200` episodes).
  - **Non-Overwritable Unique Output Bundle (`E07`)** + **`source_tree_py_sha256` Start/End Invariance (`E08`)**.
  - **100% Immediate Atomic Raw Attempt Mirror (`E04`, `E05`)**: Every `call_structured` invocation (episode AND compiler!)
    atomically mirrors its raw `attempts_*.json` into `<out_dir>/provider_raw_attempts_mirror/` on every attempt/retry/timeout.
  - **Unified Appended Provenance DAG (`E09`)**: Every episode (`200 / 200`) appends `D_CANDIDATE_SEALED -> D_PUBLIC_SCIENTIFIC_GATE -> D_FINAL_PRIVATE_GRADE`.
  - **Symmetric Post-Episode Consolidation (`E06`)**: Identical non-terminal (`step < 8`) consolidation rule across all 3 arms,
    with separate accounting of `solver_token_cost_proxy` vs `consolidation_token_cost_proxy` vs `total_lifecycle_token_cost_proxy`.
  - **World-Clustered Paired Statistics (`n = 8` Independent Worlds, Exact `2^8 = 256` Sign-Flip Permutation $p$-Values)**.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile
import threading
from time import monotonic
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from hegel_machine.cognition import (
    m6_4_executable_wormhole_library as m64,
    m6_frontier_real_benchmarks as m6f,
    m7_autogenous_forest_pretraining as m7a,
)
from hegel_machine.cognition.hipporag_wormhole_index import HippoRAGWormholeIndex
from hegel_machine.cognition.m7_holdout_ood_blind_benchmark import ensure_clean_scipy_sandbox_image
from hegel_machine.cognition.m8_persistent_variational_planner import (
    M8VariationalPlanner,
    PersistentMechanismGraph,
)
from hegel_machine.cognition.m9_4_continual_mechanism_binding_world import (
    M94BindingRegistry,
    MechanismExecutableBinding,
    grade_m94_artifact_map,
)
from hegel_machine.cognition.m9_5_equal_tool_continual_attribution_world import (
    CONFIRMATION_WORLD_SPECS,
    M95_ARMS,
    M95WorldSpec,
    build_world_tasks,
    execute_single_pre_llm_public_gate_probe_in_container,
    expected_public_output_keys,
    initialize_m95_wing_scoped_mechanism_graph,
    m95_task_to_frontier_task,
    run_equal_tool_pre_llm_trial,
    run_m95_wing_scoped_calibration_update,
    update_flat_parameter_snapshot_from_public_item,
)

DEFAULT_V7_BUNDLE = REPO_ROOT / "docs/evidence/m9_2_1_audit_closure_and_regrade_20261126/pretrained_autogenous_forest_v7_m921.json"
DEFAULT_OUT_DIR = REPO_ROOT / "docs/evidence/m9_5_equal_tool_continual_attribution_run2_20261127"


def _git_and_source_tree_state() -> tuple[str, bool, str]:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    st = subprocess.run(["git", "status", "--porcelain", "."], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    h = hashlib.sha256()
    for sub in ("src", "scripts", "tests"):
        for fp in sorted((REPO_ROOT / sub).rglob("*.py")):
            if fp.is_file():
                rel = fp.relative_to(REPO_ROOT).as_posix()
                h.update(rel.encode("utf-8") + b"\0" + hashlib.sha256(fp.read_bytes()).digest())
    return sha, bool(st), h.hexdigest()


def _materialize_public_item_host(task: Any) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="m95_hpub_") as td:
        td_p = Path(td)
        patched = task.public_setup_py.replace("/holdout_task", str(td_p))
        ns: dict[str, Any] = {}
        exec(compile(patched, f"<pub_{task.short_name}>", "exec"), ns)
        return pickle.loads((td_p / f"public_{task.short_name}.pkl").read_bytes())[0]


def _extract_submitted_py(ep_dir: Path, task: Any) -> dict[str, str]:
    sdir = ep_dir / "04_submitted_artifacts_snapshot"
    art_rel = str(task.declared_artifacts[0]).lstrip("/")
    bname = Path(art_rel).name
    hit = next((cp for cp in (sdir / art_rel, sdir / f"slot_00__{bname}", sdir / bname) if cp.is_file()), None)
    if hit is None:
        py_files = sorted(sdir.rglob("*.py"))
        hit = py_files[0] if py_files else None
    if hit is None:
        return {}
    return {str(task.declared_artifacts[0]): hit.read_text(encoding="utf-8")}


def _summarize_public_calibration_sweep(family: str, pub_item: dict[str, Any], short_name: str) -> dict[str, Any]:
    if family == "alpha":
        cal = pub_item["dataset"]["calibration_load_cells"]
        return {
            "episode_short_name": short_name,
            "lab_id": np.asarray(cal["lab_id"], dtype=int).tolist(),
            "cmd_stress": np.round(np.asarray(cal["cmd_stress"], dtype=float), 6).tolist(),
            "true_stress": np.round(np.asarray(cal["true_stress"], dtype=float), 6).tolist(),
        }
    cal = pub_item["study"]["sensor_calibration_sweeps"]
    return {
        "episode_short_name": short_name,
        "bank_id": np.asarray(cal["bank_id"], dtype=int).tolist(),
        "ref_signal": np.round(np.asarray(cal["ref_signal"], dtype=float), 6).tolist(),
        "raw_readout": np.round(np.asarray(cal["raw_readout"], dtype=float), 6).tolist(),
    }


def _inject_equal_info_archive_into_task(
    task: Any,
    prior_archive_entries: list[dict[str, Any]],
    prior_calibration_sweeps: list[dict[str, Any]],
    pre_llm_diagnostic: dict[str, Any] | None = None,
) -> tuple[Any, str]:
    """Stage `/tmp/hegel_continual_archive/` identically across ALL 3 ARMS (`100%` equal historical code + calibration sweep access!)."""
    extra_lines = [
        "\n# Materialize equal-information continual stream archive in `/tmp/hegel_continual_archive/`",
        "os.makedirs('/tmp/hegel_continual_archive', exist_ok=True)",
    ]
    manifest_rows = []
    for ent in prior_archive_entries:
        ep_tag = str(ent["short_name"])
        code_txt = str(ent["submitted_code"])
        extra_lines.append(f"open('/tmp/hegel_continual_archive/{ep_tag}_solver.py', 'w').write({code_txt!r})")
        manifest_rows.append(
            {
                "episode_short_name": ep_tag,
                "task_id": ent["task_id"],
                "entrypoint": ent["entrypoint"],
                "output_schema_keys": ent["output_schema_keys"],
                "solver_path": f"/tmp/hegel_continual_archive/{ep_tag}_solver.py",
            }
        )
    extra_lines.append(f"open('/tmp/hegel_continual_archive/archive_manifest.json', 'w').write({json.dumps(manifest_rows, indent=2)!r})")
    extra_lines.append(
        f"open('/tmp/hegel_continual_archive/archive_calibration_history.json', 'w').write({json.dumps(prior_calibration_sweeps, indent=2)!r})"
    )
    if pre_llm_diagnostic is not None:
        extra_lines.append(
            f"open('/tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json', 'w').write({json.dumps(pre_llm_diagnostic, indent=2)!r})"
        )

    header = (
        "CONTINUAL STREAM ARCHIVE (`/tmp/hegel_continual_archive/`):\n"
        f"- Prior verified stream deliverables: {[r['solver_path'] for r in manifest_rows]}.\n"
        "- Manifest: `/tmp/hegel_continual_archive/archive_manifest.json`.\n"
        "- Full historical calibration sweeps from all prior batches: `/tmp/hegel_continual_archive/archive_calibration_history.json`."
    )
    if pre_llm_diagnostic is not None:
        header += f"\n- Pre-LLM Public Gate Diagnostic (`/tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json`): `{json.dumps(pre_llm_diagnostic)}`."

    aug_task = type(task)(
        **{
            **task.__dict__,
            "public_setup_py": task.public_setup_py + "\n" + "\n".join(extra_lines) + "\n",
        }
    )
    return aug_task, header


def _exact_two_sided_sign_flip_p_value(diffs: list[float]) -> float:
    arr = np.asarray(diffs, dtype=float)
    n = len(arr)
    if n == 0:
        return 1.0
    obs = abs(float(np.mean(arr)))
    if obs <= 1e-15:
        return 1.0
    extreme = 0
    total = 1 << n
    for signs in itertools.product((-1.0, 1.0), repeat=n):
        m_val = abs(float(np.mean(arr * np.asarray(signs, dtype=float))))
        if m_val >= obs - 1e-12:
            extreme += 1
    return round(float(extreme) / float(total), 6)


def _bootstrap_mean_ci(diffs: list[float], seed: int = 95099, n_boot: int = 10000) -> tuple[float, float]:
    arr = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = np.mean(arr[idx], axis=1)
    return round(float(np.percentile(means, 2.5)), 2), round(float(np.percentile(means, 97.5)), 2)


def _run_one_world_arm_stream(
    *,
    world_spec: M95WorldSpec,
    arm: str,
    stream_tasks: tuple[Any, ...],
    schedule_manifest: list[dict[str, Any]],
    base_forest: Any,
    base_archive: dict[str, dict[str, Any]],
    base_graph_ckpt_path: Path,
    source_archive_entry: dict[str, Any],
    source_cal_summary: dict[str, Any],
    source_forest_nid: str,
    model: str,
    codex_info: dict[str, Any],
    stream_out_dir: Path,
    run_id: str,
    max_turns: int,
    reasoning_effort: str,
) -> dict[str, Any]:
    stream_t0 = monotonic()
    fam = world_spec.family
    wid = world_spec.world_id
    archive = deepcopy(base_archive)
    graph = PersistentMechanismGraph.load_checkpoint(base_graph_ckpt_path) if arm == "Full_Mechanism_Graph" else None
    forest = graph.forest if (arm == "Full_Mechanism_Graph" and graph is not None) else deepcopy(base_forest)
    planner = M8VariationalPlanner(graph, min_gain_nats=0.25, collateral_tolerance_nats=-0.05) if graph is not None else None
    registry = M94BindingRegistry(graph) if graph is not None else None

    flat_snap: dict[str, Any] = (
        {"stress_gains": [1.0, 1.0, 1.0, 1.0], "stress_biases": [0.0, 0.0, 0.0, 0.0]}
        if fam == "alpha"
        else {"bank_gains": [1.0, 1.0, 1.0, 1.0], "bank_offsets": [0.0, 0.0, 0.0, 0.0]}
    )
    prior_entries: list[dict[str, Any]] = [deepcopy(source_archive_entry)]
    prior_cal_sweeps: list[dict[str, Any]] = [deepcopy(source_cal_summary)]
    episode_results: list[dict[str, Any]] = []
    stream_compiler_receipts: list[dict[str, Any]] = []

    comp_client = m6f.FrontierCodexClient(
        codex_info=codex_info, model=model, reasoning_effort=reasoning_effort, timeout_seconds=300
    )
    n_steps = len(stream_tasks)

    for step_i, (task, m_info) in enumerate(zip(stream_tasks, schedule_manifest), start=1):
        ep_t0 = monotonic()
        ep_dir = stream_out_dir / task.task_id
        ep_dir.mkdir(parents=True, exist_ok=True)
        pub_item = _materialize_public_item_host(task)
        req_keys = expected_public_output_keys(task)

        m8_update_rec: dict[str, Any] | None = None
        if arm == "Flat_ParameterizedCache":
            flat_snap = update_flat_parameter_snapshot_from_public_item(
                family=fam, public_item=pub_item, prior_snapshot=flat_snap
            )
        elif arm == "Full_Mechanism_Graph":
            assert graph is not None and planner is not None and registry is not None
            m8_update_rec = run_m95_wing_scoped_calibration_update(
                family=fam,
                episode_id=f"{wid}_{task.short_name}",
                public_item=pub_item,
                graph=graph,
                planner=planner,
                registry=registry,
            )

        # Shared Equal-Tool Pre-LLM Trial (`max_pre_llm_probes = 1` across ALL 3 arms!)
        probe_t0 = monotonic()
        pre_trial = run_equal_tool_pre_llm_trial(
            arm=arm,
            family=fam,
            task=task,
            public_item=pub_item,
            prior_archive_entries=prior_entries,
            flat_param_snapshot=(flat_snap if arm == "Flat_ParameterizedCache" else None),
            graph=graph,
            registry=registry,
        )
        probe_wall_ms = round((monotonic() - probe_t0) * 1000.0, 2)

        if pre_trial.get("accepted_pre_llm") is True:
            cand_code = str(pre_trial["candidate_code"])
            art_path = str(task.declared_artifacts[0])
            bname = Path(art_path).name
            sdir = ep_dir / "04_submitted_artifacts_snapshot"
            sdir.mkdir(parents=True, exist_ok=True)
            (sdir / f"slot_00__{bname}").write_text(cand_code, encoding="utf-8")

            priv_ver = grade_m94_artifact_map(task, {art_path: cand_code})
            cand_sha = hashlib.sha256(cand_code.encode("utf-8")).hexdigest()
            dag_chk = m6f.append_post_episode_verification_to_dag(
                ep_dir,
                run_id=run_id,
                task_id=task.task_id,
                arm=arm,
                model=model,
                candidate_sha256=cand_sha,
                public_gate_receipt=(pre_trial.get("probe_result") or {}).get("scientific_probe_receipt"),
                verifier_report=priv_ver,
            )
            wall_ms = round((monotonic() - ep_t0) * 1000.0, 2)
            summary = {
                "run_id": run_id,
                "world_id": wid,
                "family": fam,
                "arm": arm,
                "stream_step": step_i,
                "causal_event": m_info["causal_event"],
                "event_category": m_info["event_category"],
                "task_id": task.task_id,
                "short_name": task.short_name,
                "execution_mode": "pre_llm_equal_tool_gate_accepted",
                "strict_pass": bool(priv_ver.get("strict_pass")),
                "reward": 1.0 if priv_ver.get("strict_pass") else 0.0,
                "turns_used": 0,
                "ended_by_finish": True,
                "prompt_tokens": 0,
                "cached_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "token_cost_proxy": 0,
                "solver_token_cost_proxy": 0,
                "consolidation_token_cost_proxy": 0,
                "total_lifecycle_token_cost_proxy": 0,
                "pre_llm_probe_wall_ms": probe_wall_ms,
                "consolidation_wall_ms": 0.0,
                "wall_ms": wall_ms,
                "m8_wing_scoped_update": m8_update_rec,
                "pre_llm_trial_receipt": {k: v for k, v in pre_trial.items() if k != "candidate_code"},
                "provenance_dag_verification": dag_chk,
                "provider_attempt_accounting": {
                    "logical_llm_calls_total": 0,
                    "logical_agent_decision_calls": 0,
                    "logical_llm_error_calls": 0,
                    "total_provider_attempts": 0,
                    "usage_unknown_attempt_count": 0,
                    "cost_accounting_status": "complete",
                    "archived_attempt_files": [],
                    "quarantined_unmatched_attempt_files": [],
                },
                "verifier_report": priv_ver,
            }
            (ep_dir / "00_episode_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            (ep_dir / "03_official_verifier_report.json").write_text(json.dumps(priv_ver, indent=2) + "\n", encoding="utf-8")
            prior_entries.append(
                {
                    "short_name": task.short_name,
                    "task_id": task.task_id,
                    "entrypoint": task.declared_callable_entrypoints[0],
                    "submitted_code": cand_code,
                    "output_schema_keys": req_keys,
                }
            )
            prior_cal_sweeps.append(_summarize_public_calibration_sweep(fam, pub_item, task.short_name))
            if graph is not None:
                graph.save_checkpoint(stream_out_dir / f"graph_ckpt_after_step_{step_i:02d}.json")
            episode_results.append(summary)
            print(
                f"  [M9.5 {wid} | {arm} | step {step_i}/{n_steps} {task.short_name} ({m_info['causal_event']})] "
                f"PRE_LLM_DIRECT | pass={summary['strict_pass']} | turns=0 | wall={wall_ms:.0f}ms",
                flush=True,
            )
            continue

        # Fallback to Live `gpt-6.2-sol` Episode (`18` turns, Equal-Information Archive + On-Demand Executable Library!)
        diag_for_prompt = {
            "arm": arm,
            "pre_llm_reason": pre_trial.get("reason"),
            "parameter_payload": pre_trial.get("parameter_payload"),
            "guard_evaluations": pre_trial.get("guard_evaluations"),
        }
        aug_task, ctx_hdr = _inject_equal_info_archive_into_task(task, prior_entries, prior_cal_sweeps, diag_for_prompt)
        ft = m95_task_to_frontier_task(aug_task, continual_context_block=ctx_hdr)
        ep_run_tag = f"{run_id}__{wid}__{arm}"

        ep_sum = m6f.run_frontier_episode(
            ft,
            "Hegel_M6_9_Antithesis_LiveGPT",
            model=model,
            codex_info=codex_info,
            forest=forest,
            forest_archive=archive,
            hippo_idx=HippoRAGWormholeIndex(forest),
            episode_out_dir=ep_dir,
            run_id=ep_run_tag,
            max_turns=max_turns,
            cmd_timeout_sec=180,
            reasoning_effort=reasoning_effort,
            defer_official_verifier=True,
            enable_auto_skill_packaging=True,
            allow_l1_primitives=True,
            disclosure_mode="on_demand_executable",
            staging_policy="net_value_gated",
        )
        sub_map = _extract_submitted_py(ep_dir, task)
        priv_ver = grade_m94_artifact_map(task, sub_map) if sub_map else {"strict_pass": False, "error": "missing_submitted_artifact"}
        cand_txt = next(iter(sub_map.values()), "")
        cand_sha = hashlib.sha256(cand_txt.encode("utf-8")).hexdigest() if cand_txt else ""
        dag_chk = m6f.append_post_episode_verification_to_dag(
            ep_dir,
            run_id=ep_run_tag,
            task_id=task.task_id,
            arm=arm,
            model=model,
            candidate_sha256=cand_sha,
            public_gate_receipt={"ended_by_finish": bool(ep_sum.get("ended_by_finish")), "verified_claim_ids": ["agent_public_gate_finish"]},
            verifier_report=priv_ver,
        )

        # Symmetric Non-Terminal Consolidation Rule (`step_i < n_steps` IDENTICAL across ALL 3 arms!)
        cons_proxy = 0
        cons_wall_ms = 0.0
        if sub_map and step_i < n_steps:
            c_t0 = monotonic()
            ut_cid = m6f.start_detached_container(m7a.CLEAN_SCIPY_SANDBOX_IMAGE, memory_limit="1536m", cpus="2", network_mode="none")
            try:
                sigs = m6f.extract_public_task_signatures(ft)
                comp_res = m7a.compile_episode_dialectical_sublation(
                    episode_id=f"{ep_run_tag}__{task.short_name}",
                    source_slug=task.short_name,
                    public_task_brief=task.instruction_text,
                    inferred_signatures=sigs,
                    raw_python_sources={Path(k).name: v for k, v in sub_map.items()},
                    nogoods_recorded=ep_sum.get("nogoods_recorded") or [],
                    parent_node_summary=archive.get(source_forest_nid),
                    client=comp_client,
                    sandbox_cid=ut_cid,
                    stage_subdir=f"{ep_run_tag}__{task.short_name}_comp",
                )
                c_recs = comp_res.get("provider_attempt_receipts") or []
                stream_compiler_receipts.extend(c_recs)
                for cr in c_recs:
                    p_tk = int(cr.get("prompt_tokens", 0) or 0)
                    ca_tk = int(cr.get("cached_tokens", 0) or 0)
                    o_tk = int(cr.get("completion_tokens", 0) or 0)
                    cons_proxy += max(0, p_tk - ca_tk) + (ca_tk // 10) + 4 * o_tk

                c0_falsified = bool(
                    arm == "Full_Mechanism_Graph"
                    and any(
                        ev.get("branch_tag") == "C0_base"
                        and (ev.get("guard_result") or {}).get("state") == "FALSE"
                        and ev.get("schema_compatible") is True
                        for ev in (pre_trial.get("guard_evaluations") or [])
                    )
                )
                if arm == "Full_Mechanism_Graph" and c0_falsified and graph is not None and registry is not None:
                    ext_guard = (
                        m7a.ObservablePreconditionGuard(
                            guard_id=f"guard_{wid}_c1_{task.short_name}",
                            observable_extractor="overstress_excess_max_history_and_hold",
                            op=">",
                            value=0.0,
                            applies_to_component_id="linear_viscoelastic_memory",
                        )
                        if fam == "alpha"
                        else m7a.ObservablePreconditionGuard(
                            guard_id=f"guard_{wid}_c1_{task.short_name}",
                            observable_extractor="nonseparable_2d_additive_residual_ratio",
                            op=">",
                            value=1e-6,
                            applies_to_component_id="separable_kronecker_transport",
                        )
                    )
                    if comp_res.get("compiled_ok"):
                        comp_res["regime_bifurcation"] = {**(comp_res.get("regime_bifurcation") or {}), "should_split_node": True}
                        cons_r = m7a.consolidate_autogenous_episode_into_forest(
                            forest=forest,
                            archive=archive,
                            compilation=comp_res,
                            nogoods_recorded=ep_sum.get("nogoods_recorded") or [],
                            allow_horizontal_wormholes=False,
                            source_node_id=source_forest_nid,
                            source_version=1,
                            violated_component_id=ext_guard.applies_to_component_id,
                            preserved_component_ids=(
                                ["reference_panel_calibration", "variable_step_gap_recursion"]
                                if fam == "alpha"
                                else ["sensor_bank_calibration", "nuisance_profiling", "box_simplex_kkt_certification"]
                            ),
                            guard_spec_override=ext_guard,
                        )
                        c1_fnid = str(cons_r["target_node_id"])
                        c1_stem = str(comp_res["module_stem"])
                        c1_dig = str(comp_res["compiled_code_sha256"])
                        c1_fn = comp_res["function_names"][0] if comp_res.get("function_names") else task.declared_callable_entrypoints[0]
                        c1_qual = "hegel_l2_aufhebung_v2"
                    else:
                        c1_fnid = source_forest_nid
                        c1_stem = task.short_name
                        c1_dig = cand_sha
                        c1_fn = task.declared_callable_entrypoints[0]
                        c1_qual = "hegel_public_scientific_gate_v4"

                    comp_id = "alpha_facility_composite_calibration" if fam == "alpha" else "beta_facility_composite_calibration"
                    if f"bind_{wid}_c1" not in registry.bindings:
                        c1_gate = execute_single_pre_llm_public_gate_probe_in_container(task=task, candidate_code=cand_txt)
                        if c1_gate["accepted"]:
                            registry.register_binding(
                                MechanismExecutableBinding(
                                    binding_id=f"bind_{wid}_c1",
                                    chain_id=fam,
                                    branch_tag="C1_extension",
                                    mechanism_id=f"mech_{wid}_c1",
                                    mechanism_version=1,
                                    forest_node_id=c1_fnid,
                                    component_id=("viscoplastic_overstress_extension" if fam == "alpha" else "vortex_coupled_dispersion_extension"),
                                    preserved_component_ids=(
                                        ["reference_panel_calibration", "variable_step_gap_recursion"]
                                        if fam == "alpha"
                                        else ["sensor_bank_calibration", "nuisance_profiling", "box_simplex_kkt_certification"]
                                    ),
                                    violated_component_id=ext_guard.applies_to_component_id,
                                    module_stem=c1_stem,
                                    module_digest=c1_dig,
                                    function_name=c1_fn,
                                    qualification_id=c1_qual,
                                    parameter_mechanism_id=comp_id,
                                    parameter_version=int(graph.mechanisms[comp_id].version),
                                    precondition_guard=ext_guard.to_dict(),
                                    output_schema_keys=req_keys,
                                    parameterized_closure_code=cand_txt,
                                    parameterized_entrypoint=task.declared_callable_entrypoints[0],
                                    parameterized_closure_sha256=cand_sha,
                                ),
                                reason=f"component_split_consolidated_on_{task.short_name}",
                            )
                elif comp_res.get("compiled_ok"):
                    m7a.consolidate_autogenous_episode_into_forest(
                        forest=forest,
                        archive=archive,
                        compilation=comp_res,
                        nogoods_recorded=ep_sum.get("nogoods_recorded") or [],
                        allow_horizontal_wormholes=False,
                    )
            finally:
                m6f.stop_container_safely(ut_cid)
            cons_wall_ms = round((monotonic() - c_t0) * 1000.0, 2)

        solver_proxy = int(ep_sum.get("token_cost_proxy", 0) or 0)
        ep_sum["world_id"] = wid
        ep_sum["family"] = fam
        ep_sum["arm"] = arm
        ep_sum["stream_step"] = step_i
        ep_sum["causal_event"] = m_info["causal_event"]
        ep_sum["event_category"] = m_info["event_category"]
        ep_sum["execution_mode"] = "llm_synthesis_after_equal_tool_gate"
        ep_sum["strict_pass"] = bool(priv_ver.get("strict_pass"))
        ep_sum["reward"] = 1.0 if priv_ver.get("strict_pass") else 0.0
        ep_sum["solver_token_cost_proxy"] = solver_proxy
        ep_sum["consolidation_token_cost_proxy"] = cons_proxy
        ep_sum["total_lifecycle_token_cost_proxy"] = solver_proxy + cons_proxy
        ep_sum["pre_llm_probe_wall_ms"] = probe_wall_ms
        ep_sum["consolidation_wall_ms"] = cons_wall_ms
        ep_sum["m8_wing_scoped_update"] = m8_update_rec
        ep_sum["pre_llm_trial_receipt"] = {k: v for k, v in pre_trial.items() if k != "candidate_code"}
        ep_sum["provenance_dag_verification"] = dag_chk
        ep_sum["verifier_report"] = priv_ver
        (ep_dir / "00_episode_summary.json").write_text(json.dumps(ep_sum, indent=2) + "\n", encoding="utf-8")
        (ep_dir / "03_official_verifier_report.json").write_text(json.dumps(priv_ver, indent=2) + "\n", encoding="utf-8")

        if cand_txt:
            prior_entries.append(
                {
                    "short_name": task.short_name,
                    "task_id": task.task_id,
                    "entrypoint": task.declared_callable_entrypoints[0],
                    "submitted_code": cand_txt,
                    "output_schema_keys": req_keys,
                }
            )
        prior_cal_sweeps.append(_summarize_public_calibration_sweep(fam, pub_item, task.short_name))
        if graph is not None:
            graph.save_checkpoint(stream_out_dir / f"graph_ckpt_after_step_{step_i:02d}.json")
        episode_results.append(ep_sum)
        print(
            f"  [M9.5 {wid} | {arm} | step {step_i}/{n_steps} {task.short_name} ({m_info['causal_event']})] "
            f"LLM_FALLBACK | pass={ep_sum['strict_pass']} | turns={ep_sum.get('turns_used')} | solver_proxy={solver_proxy} | cons_proxy={cons_proxy}",
            flush=True,
        )

    return {
        "world_id": wid,
        "family": fam,
        "arm": arm,
        "wall_sec": round(monotonic() - stream_t0, 2),
        "strict_passes": sum(1 for r in episode_results if r.get("strict_pass")),
        "episode_count": len(episode_results),
        "zero_turn_pre_llm_closures": sum(1 for r in episode_results if int(r.get("turns_used", 0) or 0) == 0 and r.get("strict_pass")),
        "total_turns_used": sum(int(r.get("turns_used", 0) or 0) for r in episode_results),
        "solver_token_cost_proxy": sum(int(r.get("solver_token_cost_proxy", 0) or 0) for r in episode_results),
        "consolidation_token_cost_proxy": sum(int(r.get("consolidation_token_cost_proxy", 0) or 0) for r in episode_results),
        "total_lifecycle_token_cost_proxy": sum(int(r.get("total_lifecycle_token_cost_proxy", 0) or 0) for r in episode_results),
        "binding_events": (registry.binding_events if registry is not None else []),
        "episodes": [
            {
                "stream_step": r["stream_step"],
                "causal_event": r["causal_event"],
                "event_category": r["event_category"],
                "task_id": r["task_id"],
                "short_name": r.get("short_name") or Path(str(r["task_id"])).name.split("__")[0],
                "execution_mode": r["execution_mode"],
                "strict_pass": r["strict_pass"],
                "turns_used": int(r.get("turns_used", 0) or 0),
                "solver_token_cost_proxy": int(r.get("solver_token_cost_proxy", 0) or 0),
                "consolidation_token_cost_proxy": int(r.get("consolidation_token_cost_proxy", 0) or 0),
                "total_lifecycle_token_cost_proxy": int(r.get("total_lifecycle_token_cost_proxy", 0) or 0),
                "pre_llm_probe_wall_ms": float(r.get("pre_llm_probe_wall_ms", 0.0) or 0.0),
                "wall_ms": float(r.get("wall_ms", 0.0) or 0.0),
                "m8_wing_scoped_update": r.get("m8_wing_scoped_update"),
                "provenance_dag_valid": bool((r.get("provenance_dag_verification") or {}).get("valid")),
            }
            for r in episode_results
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--v7-bundle", type=Path, default=DEFAULT_V7_BUNDLE)
    ap.add_argument("--model", type=str, default="gpt-6.2-sol")
    ap.add_argument("--reasoning-effort", type=str, default="medium")
    ap.add_argument("--num-worlds", type=int, default=8, choices=[2, 4, 6, 8])
    ap.add_argument("--max-turns", type=int, default=18)
    ap.add_argument("--max-stream-workers", type=int, default=32)
    ap.add_argument("--max-codex-concurrency", type=int, default=32)
    args = ap.parse_args()

    os.environ["HEGEL_MAX_CODEX_CONCURRENCY"] = str(args.max_codex_concurrency)
    out_dir: Path = args.out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory `{out_dir}` (`E07` unique-run-dir rule)!")

    commit_start, dirty_start, src_sha_start = _git_and_source_tree_state()
    if dirty_start:
        raise RuntimeError("Refusing to start M9.5 confirmation experiment on a dirty working tree!")

    out_dir.mkdir(parents=True, exist_ok=True)
    mirror_dir = out_dir / "provider_raw_attempts_mirror"
    mirror_dir.mkdir(parents=True, exist_ok=True)

    ensure_clean_scipy_sandbox_image()
    m64.ensure_runtime_qualification_context(sandbox_cid=None)

    codex_info = m6f.discover_chatgpt_subscription_codex_cli()
    if not (codex_info.get("available") and codex_info.get("auth_mode") == "chatgpt"):
        raise RuntimeError(f"Official ChatGPT Codex CLI unavailable: {codex_info}")

    half_n = args.num_worlds // 2
    selected_worlds: list[M95WorldSpec] = (
        list(CONFIRMATION_WORLD_SPECS[:half_n]) + list(CONFIRMATION_WORLD_SPECS[4 : 4 + half_n])
    )
    print(
        f"[M9.5 INIT] Commit={commit_start[:12]} | SrcSHA={src_sha_start[:12]} | Worlds={len(selected_worlds)} "
        f"| Streams={len(selected_worlds)*len(M95_ARMS)} | CodexConcurrency={args.max_codex_concurrency}",
        flush=True,
    )

    cli_sem = threading.BoundedSemaphore(max(1, int(args.max_codex_concurrency)))
    master_lock = threading.Lock()
    master_provider_receipts: list[dict[str, Any]] = []
    _orig_call = m6f.FrontierCodexClient.call_structured

    def _throttled_and_mirrored_call(self_cli: Any, **kw: Any) -> dict[str, Any]:
        self_cli.immediate_archive_dir = mirror_dir
        with cli_sem:
            res = _orig_call(self_cli, **kw)
        with master_lock:
            master_provider_receipts.append(
                {
                    "stage_subdir": str(kw.get("stage_subdir") or ""),
                    "call_uuid": str(res.get("call_uuid") or ""),
                    "attempts_ledger_filename": str(res.get("attempts_ledger_filename") or ""),
                    "mirrored_raw_filename": f"{res.get('call_uuid')}__{res.get('attempts_ledger_filename')}",
                    "ok": bool(res.get("ok")),
                    "attempt_count": int(res.get("attempt_count", 1) or 1),
                    "usage_unknown_attempt_count": int(res.get("usage_unknown_attempt_count", 0) or 0),
                    "cost_accounting_status": str(res.get("cost_accounting_status") or "complete"),
                    "prompt_tokens": int(res.get("prompt_tokens", 0) or 0),
                    "cached_tokens": int(res.get("cached_tokens", 0) or 0),
                    "completion_tokens": int(res.get("completion_tokens", 0) or 0),
                    "wall_ms": float(res.get("wall_ms", 0.0) or 0.0),
                }
            )
        return res

    m6f.FrontierCodexClient.call_structured = _throttled_and_mirrored_call  # type: ignore[method-assign]
    run_id = "m9_5_equal_tool_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    t0_all = monotonic()

    base_v7_forest, base_v7_archive, base_v7_dag, _ = m7a.load_autogenous_forest_bundle(args.v7_bundle)
    built_worlds = {w.world_id: build_world_tasks(w) for w in selected_worlds}

    # =========================================================================
    # PHASE 1: Independent Blind `Stage S` Source Formation Per World (`8`-Way Parallel!)
    # =========================================================================
    src_out_root = out_dir / "stage_S_sources"

    def _run_world_source(w_spec: M95WorldSpec) -> dict[str, Any]:
        s_task, _, _ = built_worlds[w_spec.world_id]
        for att_i in (1, 2):
            s_dir = src_out_root / w_spec.world_id / f"attempt_{att_i:02d}" / s_task.task_id
            ft_s = m95_task_to_frontier_task(s_task)
            ep_a = deepcopy(base_v7_archive)
            ep_a["__autogenous_blank_seed__"] = {"enabled": True}
            ep_s = m6f.run_frontier_episode(
                ft_s,
                "Hegel_M6_9_Antithesis_LiveGPT",
                model=args.model,
                codex_info=codex_info,
                forest=deepcopy(base_v7_forest),
                forest_archive=ep_a,
                hippo_idx=HippoRAGWormholeIndex(base_v7_forest),
                episode_out_dir=s_dir,
                run_id=f"{run_id}__{w_spec.world_id}__stage_S_att{att_i:02d}",
                max_turns=args.max_turns,
                cmd_timeout_sec=180,
                reasoning_effort=args.reasoning_effort,
                defer_official_verifier=True,
                enable_auto_skill_packaging=True,
                allow_l1_primitives=True,
                disclosure_mode="on_demand_executable",
                staging_policy="net_value_gated",
            )
            sub_s = _extract_submitted_py(s_dir, s_task)
            ver_s = grade_m94_artifact_map(s_task, sub_s) if sub_s else {"strict_pass": False, "error": "missing_submitted_artifact"}
            cand_s_txt = next(iter(sub_s.values()), "")
            cand_s_sha = hashlib.sha256(cand_s_txt.encode("utf-8")).hexdigest() if cand_s_txt else ""
            dag_s = m6f.append_post_episode_verification_to_dag(
                s_dir,
                run_id=run_id,
                task_id=s_task.task_id,
                arm="Stage_S_Shared_World_Source",
                model=args.model,
                candidate_sha256=cand_s_sha,
                public_gate_receipt={"ended_by_finish": bool(ep_s.get("ended_by_finish")), "verified_claim_ids": ["stage_s_public_finish"]},
                verifier_report=ver_s,
            )
            ep_s["strict_pass"] = bool(ver_s.get("strict_pass"))
            ep_s["provenance_dag_verification"] = dag_s
            ep_s["verifier_report"] = ver_s
            (s_dir / "00_episode_summary.json").write_text(json.dumps(ep_s, indent=2) + "\n", encoding="utf-8")
            (s_dir / "03_official_verifier_report.json").write_text(json.dumps(ver_s, indent=2) + "\n", encoding="utf-8")
            print(
                f"  [M9.5 STAGE S] {w_spec.world_id} (att={att_i}) | pass={ep_s['strict_pass']} | turns={ep_s['turns_used']} | proxy={ep_s['token_cost_proxy']}",
                flush=True,
            )
            if ep_s["strict_pass"]:
                # Compile world-isolated Stage S L2 skill & freeze initial wing-scoped graph checkpoint BEFORE stream starts!
                w_forest = deepcopy(base_v7_forest)
                w_archive = deepcopy(base_v7_archive)
                comp_cli = m6f.FrontierCodexClient(
                    codex_info=codex_info, model=args.model, reasoning_effort=args.reasoning_effort, timeout_seconds=300
                )
                ut_cid = m6f.start_detached_container(m7a.CLEAN_SCIPY_SANDBOX_IMAGE, memory_limit="1536m", cpus="2", network_mode="none")
                try:
                    sigs_s = m6f.extract_public_task_signatures(ft_s)
                    c_res = m7a.compile_episode_dialectical_sublation(
                        episode_id=f"m95_src_{w_spec.world_id}",
                        source_slug=s_task.short_name,
                        public_task_brief=s_task.instruction_text,
                        inferred_signatures=sigs_s,
                        raw_python_sources={Path(k).name: v for k, v in sub_s.items()},
                        nogoods_recorded=ep_s.get("nogoods_recorded") or [],
                        parent_node_summary=None,
                        client=comp_cli,
                        sandbox_cid=ut_cid,
                        stage_subdir=f"{run_id}__{w_spec.world_id}__src_l2",
                    )
                    if c_res.get("compiled_ok"):
                        c_res["regime_bifurcation"] = {**(c_res.get("regime_bifurcation") or {}), "should_split_node": False}
                        cons_s = m7a.consolidate_autogenous_episode_into_forest(
                            forest=w_forest,
                            archive=w_archive,
                            compilation=c_res,
                            nogoods_recorded=ep_s.get("nogoods_recorded") or [],
                            allow_horizontal_wormholes=True,
                        )
                        src_nid = str(cons_s["target_node_id"])
                        src_stem = str(c_res["module_stem"])
                        src_dig = str(c_res["compiled_code_sha256"])
                        src_fn = c_res["function_names"][0] if c_res.get("function_names") else s_task.declared_callable_entrypoints[0]
                        src_qual = "hegel_l2_aufhebung_v2"
                    else:
                        src_nid = "root_A01_conservation_and_operator_symmetry"
                        src_stem = s_task.short_name
                        src_dig = cand_s_sha
                        src_fn = s_task.declared_callable_entrypoints[0]
                        src_qual = "hegel_public_scientific_gate_v4"
                finally:
                    m6f.stop_container_safely(ut_cid)

                init_g, _, init_reg = initialize_m95_wing_scoped_mechanism_graph(family=w_spec.family, forest=deepcopy(w_forest))
                comp_id = "alpha_facility_composite_calibration" if w_spec.family == "alpha" else "beta_facility_composite_calibration"
                c0_guard = (
                    m7a.ObservablePreconditionGuard(
                        guard_id=f"guard_{w_spec.world_id}_c0",
                        observable_extractor="overstress_excess_max_history_and_hold",
                        op="<=",
                        value=0.0,
                        applies_to_component_id="linear_viscoelastic_memory",
                    )
                    if w_spec.family == "alpha"
                    else m7a.ObservablePreconditionGuard(
                        guard_id=f"guard_{w_spec.world_id}_c0",
                        observable_extractor="nonseparable_2d_additive_residual_ratio",
                        op="<=",
                        value=1e-6,
                        applies_to_component_id="separable_kronecker_transport",
                    )
                )
                uni_keys = expected_public_output_keys(s_task)
                init_reg.register_binding(
                    MechanismExecutableBinding(
                        binding_id=f"bind_{w_spec.world_id}_c0",
                        chain_id=w_spec.family,
                        branch_tag="C0_base",
                        mechanism_id=f"mech_{w_spec.world_id}_c0",
                        mechanism_version=1,
                        forest_node_id=src_nid,
                        component_id=c0_guard.applies_to_component_id,
                        preserved_component_ids=(
                            ["reference_panel_calibration", "variable_step_gap_recursion"]
                            if w_spec.family == "alpha"
                            else ["sensor_bank_calibration", "nuisance_profiling", "box_simplex_kkt_certification"]
                        ),
                        violated_component_id=None,
                        module_stem=src_stem,
                        module_digest=src_dig,
                        function_name=src_fn,
                        qualification_id=src_qual,
                        parameter_mechanism_id=comp_id,
                        parameter_version=1,
                        precondition_guard=c0_guard.to_dict(),
                        output_schema_keys=uni_keys,
                        parameterized_closure_code=cand_s_txt,
                        parameterized_entrypoint=s_task.declared_callable_entrypoints[0],
                        parameterized_closure_sha256=cand_s_sha,
                    ),
                    reason=f"stage_S_freeze_{w_spec.world_id}",
                )
                w_freeze_dir = out_dir / "world_freezes" / w_spec.world_id
                w_freeze_dir.mkdir(parents=True, exist_ok=True)
                ckpt_p = w_freeze_dir / "initial_mechanism_graph_ckpt.json"
                ckpt_meta = init_g.save_checkpoint(ckpt_p)
                pub_s_item = _materialize_public_item_host(s_task)
                return {
                    "world_id": w_spec.world_id,
                    "ep_summary": ep_s,
                    "forest": w_forest,
                    "archive": w_archive,
                    "graph_ckpt_path": ckpt_p,
                    "graph_ckpt_meta": ckpt_meta,
                    "source_nid": src_nid,
                    "source_archive_entry": {
                        "short_name": s_task.short_name,
                        "task_id": s_task.task_id,
                        "entrypoint": s_task.declared_callable_entrypoints[0],
                        "submitted_code": cand_s_txt,
                        "output_schema_keys": uni_keys,
                    },
                    "source_cal_summary": _summarize_public_calibration_sweep(w_spec.family, pub_s_item, s_task.short_name),
                }
        raise RuntimeError(f"Stage S source for `{w_spec.world_id}` failed private verifier across 2 blind attempts!")

    world_init_by_id: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=len(selected_worlds)) as pool_s:
        futs_s = [pool_s.submit(_run_world_source, w) for w in selected_worlds]
        for f_done in as_completed(futs_s):
            res_w = f_done.result()
            world_init_by_id[res_w["world_id"]] = res_w

    # =========================================================================
    # PHASE 2: `24` Parallel Independent Continual Streams (`8` Worlds x `3` Arms)
    # =========================================================================
    stream_jobs = [(w, arm_nm) for w in selected_worlds for arm_nm in M95_ARMS]
    stream_summaries: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.max_stream_workers))) as pool_st:
        futs_st = []
        for w_spec, arm_nm in stream_jobs:
            w_init = world_init_by_id[w_spec.world_id]
            _, s_tasks, s_man = built_worlds[w_spec.world_id]
            futs_st.append(
                pool_st.submit(
                    _run_one_world_arm_stream,
                    world_spec=w_spec,
                    arm=arm_nm,
                    stream_tasks=s_tasks,
                    schedule_manifest=s_man,
                    base_forest=w_init["forest"],
                    base_archive=w_init["archive"],
                    base_graph_ckpt_path=w_init["graph_ckpt_path"],
                    source_archive_entry=w_init["source_archive_entry"],
                    source_cal_summary=w_init["source_cal_summary"],
                    source_forest_nid=w_init["source_nid"],
                    model=args.model,
                    codex_info=codex_info,
                    stream_out_dir=out_dir / "continual_streams" / w_spec.world_id / arm_nm,
                    run_id=run_id,
                    max_turns=args.max_turns,
                    reasoning_effort=args.reasoning_effort,
                )
            )
        for fd in as_completed(futs_st):
            stream_summaries.append(fd.result())

    stream_summaries.sort(key=lambda r: (r["world_id"], M95_ARMS.index(r["arm"])))

    # =========================================================================
    # PHASE 3: World-Clustered Paired Statistics & Forensic Integrity Assertions
    # =========================================================================
    shutil.rmtree(out_dir / "_codex_stage", ignore_errors=True)
    # Check git/source cleanliness BEFORE writing summary into `docs/evidence/...`? Note `out_dir` is untracked in git,
    # and `_git_and_source_tree_state` hashes ONLY `src/`, `scripts/`, `tests/`!
    commit_end, _, src_sha_end = _git_and_source_tree_state()
    tracked_dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no", "."],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert commit_start == commit_end, f"Git commit changed mid-run: {commit_start} -> {commit_end}"
    assert src_sha_start == src_sha_end, f"Python source tree SHA-256 mutated mid-run: {src_sha_start} -> {src_sha_end}"
    assert not tracked_dirty, f"Tracked files mutated mid-run: {tracked_dirty}"

    mirrored_raw_files = sorted(p.name for p in mirror_dir.glob("*__attempts_*.json"))
    assert len(mirrored_raw_files) == len(master_provider_receipts), (
        f"Raw provider attempt mirror mismatch: mirrored={len(mirrored_raw_files)} != receipts={len(master_provider_receipts)}"
    )

    by_world_arm = {(s["world_id"], s["arm"]): s for s in stream_summaries}
    world_ids_ord = [w.world_id for w in selected_worlds]

    arm_agg: dict[str, Any] = {}
    for arm_nm in M95_ARMS:
        a_streams = [by_world_arm[(wid, arm_nm)] for wid in world_ids_ord]
        a_eps = [ep for s in a_streams for ep in s["episodes"]]
        by_event = {}
        for ev_k in ("T", "N", "G1", "C", "R", "C_prime", "G2", "M"):
            ev_eps = [e for e in a_eps if e["causal_event"] == ev_k]
            by_event[ev_k] = {
                "strict_passes": sum(1 for e in ev_eps if e["strict_pass"]),
                "episode_count": len(ev_eps),
                "zero_turn_closures": sum(1 for e in ev_eps if e["turns_used"] == 0 and e["strict_pass"]),
                "mean_turns": round(float(np.mean([e["turns_used"] for e in ev_eps])), 2) if ev_eps else 0.0,
                "solver_token_cost_proxy": sum(int(e["solver_token_cost_proxy"]) for e in ev_eps),
                "total_lifecycle_token_cost_proxy": sum(int(e["total_lifecycle_token_cost_proxy"]) for e in ev_eps),
            }
        by_cat = {}
        for cat_k in ("immediate_cal_negative_control", "adaptation_discovery", "continual_history_guard_fanout"):
            c_eps = [e for e in a_eps if e["event_category"] == cat_k]
            by_cat[cat_k] = {
                "strict_passes": sum(1 for e in c_eps if e["strict_pass"]),
                "episode_count": len(c_eps),
                "zero_turn_closures": sum(1 for e in c_eps if e["turns_used"] == 0 and e["strict_pass"]),
                "solver_token_cost_proxy": sum(int(e["solver_token_cost_proxy"]) for e in c_eps),
                "total_lifecycle_token_cost_proxy": sum(int(e["total_lifecycle_token_cost_proxy"]) for e in c_eps),
            }
        arm_agg[arm_nm] = {
            "world_count": len(a_streams),
            "episode_count": len(a_eps),
            "strict_passes": sum(1 for e in a_eps if e["strict_pass"]),
            "pass_rate": round(sum(1 for e in a_eps if e["strict_pass"]) / max(len(a_eps), 1), 4),
            "zero_turn_pre_llm_closures": sum(1 for e in a_eps if e["turns_used"] == 0 and e["strict_pass"]),
            "total_turns_used": sum(int(e["turns_used"]) for e in a_eps),
            "mean_turns_per_episode": round(float(np.mean([e["turns_used"] for e in a_eps])), 2),
            "solver_token_cost_proxy": sum(int(s["solver_token_cost_proxy"]) for s in a_streams),
            "consolidation_token_cost_proxy": sum(int(s["consolidation_token_cost_proxy"]) for s in a_streams),
            "total_lifecycle_token_cost_proxy": sum(int(s["total_lifecycle_token_cost_proxy"]) for s in a_streams),
            "total_wall_sec_sum_across_streams": round(sum(float(s["wall_sec"]) for s in a_streams), 2),
            "by_causal_event": by_event,
            "by_event_category": by_cat,
        }

    def _paired_world_comparison(arm_treat: str, arm_base: str) -> dict[str, Any]:
        solver_diffs = []
        lifecycle_diffs = []
        turn_diffs = []
        pass_diffs = []
        per_world = []
        for wid in world_ids_ord:
            st_t = by_world_arm[(wid, arm_treat)]
            st_b = by_world_arm[(wid, arm_base)]
            sd = int(st_t["solver_token_cost_proxy"]) - int(st_b["solver_token_cost_proxy"])
            ld = int(st_t["total_lifecycle_token_cost_proxy"]) - int(st_b["total_lifecycle_token_cost_proxy"])
            td = int(st_t["total_turns_used"]) - int(st_b["total_turns_used"])
            pd = int(st_t["strict_passes"]) - int(st_b["strict_passes"])
            solver_diffs.append(float(sd))
            lifecycle_diffs.append(float(ld))
            turn_diffs.append(float(td))
            pass_diffs.append(float(pd))
            per_world.append(
                {
                    "world_id": wid,
                    "solver_proxy_diff": sd,
                    "lifecycle_proxy_diff": ld,
                    "turns_diff": td,
                    "strict_passes_diff": pd,
                }
            )
        tot_t_sol = sum(int(by_world_arm[(w, arm_treat)]["solver_token_cost_proxy"]) for w in world_ids_ord)
        tot_b_sol = sum(int(by_world_arm[(w, arm_base)]["solver_token_cost_proxy"]) for w in world_ids_ord)
        tot_t_lif = sum(int(by_world_arm[(w, arm_treat)]["total_lifecycle_token_cost_proxy"]) for w in world_ids_ord)
        tot_b_lif = sum(int(by_world_arm[(w, arm_base)]["total_lifecycle_token_cost_proxy"]) for w in world_ids_ord)
        return {
            "treatment": arm_treat,
            "baseline": arm_base,
            "n_independent_worlds": len(world_ids_ord),
            "worlds_with_lower_solver_proxy": sum(1 for d in solver_diffs if d < 0),
            "worlds_with_lower_lifecycle_proxy": sum(1 for d in lifecycle_diffs if d < 0),
            "solver_proxy_total_delta": tot_t_sol - tot_b_sol,
            "solver_proxy_relative_pct": round((tot_t_sol - tot_b_sol) / max(tot_b_sol, 1) * 100.0, 2),
            "solver_proxy_mean_world_delta": round(float(np.mean(solver_diffs)), 2),
            "solver_proxy_world_bootstrap_95ci": list(_bootstrap_mean_ci(solver_diffs)),
            "solver_proxy_exact_sign_flip_two_sided_p": _exact_two_sided_sign_flip_p_value(solver_diffs),
            "lifecycle_proxy_total_delta": tot_t_lif - tot_b_lif,
            "lifecycle_proxy_relative_pct": round((tot_t_lif - tot_b_lif) / max(tot_b_lif, 1) * 100.0, 2),
            "lifecycle_proxy_mean_world_delta": round(float(np.mean(lifecycle_diffs)), 2),
            "lifecycle_proxy_world_bootstrap_95ci": list(_bootstrap_mean_ci(lifecycle_diffs)),
            "lifecycle_proxy_exact_sign_flip_two_sided_p": _exact_two_sided_sign_flip_p_value(lifecycle_diffs),
            "turns_mean_world_delta": round(float(np.mean(turn_diffs)), 2),
            "turns_exact_sign_flip_two_sided_p": _exact_two_sided_sign_flip_p_value(turn_diffs),
            "per_world_deltas": per_world,
        }

    paired_stats = {
        "Full_vs_Archive": _paired_world_comparison("Full_Mechanism_Graph", "Archive_AutoReplay"),
        "Full_vs_Flat": _paired_world_comparison("Full_Mechanism_Graph", "Flat_ParameterizedCache"),
        "Flat_vs_Archive": _paired_world_comparison("Flat_ParameterizedCache", "Archive_AutoReplay"),
    }

    all_dag_files = sorted(out_dir.rglob("01_decision_provenance_dag.json"))
    dag_valid_count = sum(1 for fp in all_dag_files if json.loads(fp.read_text(encoding="utf-8"))["provenance_verification"]["valid"])

    master_summary = {
        "schema": "hegel_m9_5_equal_tool_continual_attribution_summary_v1",
        "run_id": run_id,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "max_codex_concurrency": args.max_codex_concurrency,
        "max_stream_workers": args.max_stream_workers,
        "code_commit_at_start": commit_start,
        "code_commit_at_end": commit_end,
        "source_tree_py_sha256_at_start": src_sha_start,
        "source_tree_py_sha256_at_end": src_sha_end,
        "tracked_tree_dirty_at_end": bool(tracked_dirty),
        "wall_clock_sec": round(monotonic() - t0_all, 2),
        "world_specifications": [
            {
                "world_id": w.world_id,
                "family": w.family,
                "world_seed": w.world_seed,
                "schedule_events": list(w.schedule_events),
                "v2_gains": list(w.v2_gains),
                "v2_offsets": list(w.v2_offsets),
                "v3_wing_a_gains": list(w.v3_wing_a_gains),
                "v3_wing_a_offsets": list(w.v3_wing_a_offsets),
            }
            for w in selected_worlds
        ],
        "stage_S_sources": {
            wid: {
                "task_id": world_init_by_id[wid]["ep_summary"]["task_id"],
                "strict_pass": world_init_by_id[wid]["ep_summary"]["strict_pass"],
                "turns_used": world_init_by_id[wid]["ep_summary"]["turns_used"],
                "token_cost_proxy": world_init_by_id[wid]["ep_summary"]["token_cost_proxy"],
                "initial_graph_ckpt_sha256": world_init_by_id[wid]["graph_ckpt_meta"]["checkpoint_sha256"],
            }
            for wid in world_ids_ord
        },
        "arm_comparison": arm_agg,
        "world_clustered_paired_statistics": paired_stats,
        "forensic_integrity": {
            "total_episodes_with_provenance_dag": len(all_dag_files),
            "valid_appended_provenance_dag_count": dag_valid_count,
            "logical_llm_calls_all_stages": len(master_provider_receipts),
            "mirrored_raw_attempt_files_count": len(mirrored_raw_files),
            "total_provider_attempts_all_stages": sum(int(r["attempt_count"]) for r in master_provider_receipts),
            "usage_unknown_attempt_count_all_stages": sum(int(r["usage_unknown_attempt_count"]) for r in master_provider_receipts),
            "cost_accounting_status": (
                "partial_unknown_usage"
                if any(int(r["usage_unknown_attempt_count"]) > 0 for r in master_provider_receipts)
                else "complete"
            ),
        },
        "streams": stream_summaries,
    }
    (out_dir / "m9_5_continual_summary.json").write_text(json.dumps(master_summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "all_provider_call_receipts.json").write_text(json.dumps(master_provider_receipts, indent=2) + "\n", encoding="utf-8")

    sha_rows = []
    for fp in sorted(out_dir.rglob("*")):
        if fp.is_file() and fp.name != "SHA256SUMS.txt":
            rel_p = fp.relative_to(out_dir).as_posix()
            sha_rows.append(f"{hashlib.sha256(fp.read_bytes()).hexdigest()}  {rel_p}")
    (out_dir / "SHA256SUMS.txt").write_text("\n".join(sha_rows) + "\n", encoding="utf-8")
    print(json.dumps({"arm_comparison": arm_agg, "paired_statistics": paired_stats}, indent=2), flush=True)


if __name__ == "__main__":
    main()
