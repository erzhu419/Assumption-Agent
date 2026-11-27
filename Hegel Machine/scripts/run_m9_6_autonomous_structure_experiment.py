#!/usr/bin/env python3
"""End-to-End 32-Way Parallel `gpt-6.2-sol` Runner for `M9.6`:
Autonomous Guard & Multi-Parent Structure Induction + Non-Markovian `U` + Cross-Domain Wormhole Continual Benchmark.

Head-to-Head Steelmanned Comparison (`8` Disjoint Confirmation Worlds Across `4` Disjoint Cross-Family Pairs):
  1. `Flat_Multiversion_Validated` (`Steelmanned Baseline` with 100% of `M9.5.1` Fixes):
     - Retains ALL `public_gate_verified` canonical solvers `{C0, C1, ...}` (`F01` 1-hop canonical export, `0` double-wrap!).
     - Validated per-wing `2/3–1/3` holdout-MSE OLS (`F03` — immune to Wing-B EMI spikes!).
     - Identical prefix-locked `0444` `readonly_dependencies` (`F02`) and `max_pre_llm_probes = 1` equal probe budget.
     - Pays **ZERO** forest-compilation or guard-induction token tax (`0` Stage-S L2/L1 compile charged to `Flat`, `0` in-stream compile!).
  2. `Full_Mechanism_Graph_Autonomous` (`Hegel Autonomous Structure Architecture`):
     - **Zero Author-Hardcoded Guards (`§4.1 + §4.5`)**: Starts `C0` unconstrained (`precondition_guard = None`); when falsified on a
       counterexample, `gpt-6.2-sol` proposes `(c0_guard, c1_guard, violated_component_id, dependent_upstream_mechanism_ids)` over a
       12-feature bank containing myopic & Frobenius-norm distractors, gated strictly by Contrastive Public Witnesses + Perturbation Sensitivity!
     - **True Multi-Parent `(wing_a, wing_b)` `PersistentMechanismGraph` (`§4.2`)** + selective isolation (`aux_wing_b_monitor` & `sentinel` `0.0` drift).
     - **Cross-Domain Horizontal Operator Wormholes (`§4.4`, `allow_horizontal_wormholes = True`)** across paired `(alpha, beta)` `Stage S` sources!
     - Pays **100%** of its own `stream_compile` + `autonomous_guard_induction` (and `stage_S_compile_and_l1` in Scope 2) token costs!
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
from hegel_machine.cognition.cost_policy import (
    COST_POLICY_VERSION,
    aggregate_call_usages,
    score_single_call_usage,
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
    execute_single_pre_llm_public_gate_probe_in_container,
    make_continual_archive_entry,
    update_flat_validated_wing_parameter_snapshot_from_public_item,
)
from hegel_machine.cognition.m9_6_autonomous_structure_continual_world import (
    CONFIRMATION_M96_WORLD_SPECS,
    M96_ARMS,
    M96WorldSpec,
    assemble_multiparent_parameter_payload,
    build_m96_world_tasks,
    compute_generic_public_structural_features,
    expected_m96_public_output_keys,
    initialize_m96_multiparent_mechanism_graph,
    m96_task_to_frontier_task,
    propose_and_contrastively_validate_autonomous_split,
    run_m96_equal_capability_pre_llm_trial,
    run_m96_multiparent_calibration_update,
)

DEFAULT_V7_BUNDLE = REPO_ROOT / "docs/evidence/m9_2_1_audit_closure_and_regrade_20261126/pretrained_autogenous_forest_v7_m921.json"
DEFAULT_OUT_DIR = REPO_ROOT / "docs/evidence/m9_6_autonomous_guard_multiparent_continual_run2_20261127"

_CALL_CONTEXT = threading.local()


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
    with tempfile.TemporaryDirectory(prefix="m96_hpub_") as td:
        td_p = Path(td)
        ns: dict[str, Any] = {}
        exec(compile(task.public_setup_py.replace("/holdout_task", str(td_p)), f"<pub_{task.short_name}>", "exec"), ns)
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
    """Stage `/tmp/hegel_continual_archive/` identically across BOTH arms (`100%` equal historical code + prefix calibration sweeps!)."""
    extra_lines = [
        "\n# Materialize equal-information continual stream archive in `/tmp/hegel_continual_archive/`",
        "os.makedirs('/tmp/hegel_continual_archive', exist_ok=True)",
    ]
    manifest_rows = []
    for ent in prior_archive_entries:
        if ent.get("public_qualification_status", "public_gate_verified") != "public_gate_verified":
            continue
        ep_tag = str(ent["short_name"])
        code_txt = str(ent.get("canonical_algorithm_code") or ent["submitted_code"])
        extra_lines.append(f"open('/tmp/hegel_continual_archive/{ep_tag}_solver.py', 'w').write({code_txt!r})")
        manifest_rows.append(
            {
                "episode_short_name": ep_tag,
                "task_id": ent["task_id"],
                "entrypoint": ent.get("canonical_entrypoint") or ent["entrypoint"],
                "output_schema_keys": ent["output_schema_keys"],
                "input_coordinate_contract": "raw_measured",
                "solver_path": f"/tmp/hegel_continual_archive/{ep_tag}_solver.py",
            }
        )
    extra_lines.append(f"open('/tmp/hegel_continual_archive/archive_manifest.json', 'w').write({json.dumps(manifest_rows, indent=2)!r})")
    cal_json = json.dumps(prior_calibration_sweeps, indent=2)
    extra_lines.append("if os.path.exists('/tmp/hegel_continual_archive/archive_calibration_history.json'): os.chmod('/tmp/hegel_continual_archive/archive_calibration_history.json', 0o644)")
    extra_lines.append(f"open('/tmp/hegel_continual_archive/archive_calibration_history.json', 'w').write({cal_json!r})")
    extra_lines.append("os.chmod('/tmp/hegel_continual_archive/archive_calibration_history.json', 0o444)")
    if pre_llm_diagnostic is not None:
        extra_lines.append(
            f"open('/tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json', 'w').write({json.dumps(pre_llm_diagnostic, indent=2)!r})"
        )

    header = (
        "CONTINUAL STREAM ARCHIVE (`/tmp/hegel_continual_archive/`):\n"
        f"- Prior publicly-verified canonical stream solvers (`raw_measured` inputs): {[r['solver_path'] for r in manifest_rows]}.\n"
        "- Manifest: `/tmp/hegel_continual_archive/archive_manifest.json`.\n"
        "- Read-only (`0444`) prefix calibration sweeps (`<= t-1`, also available in private verifier): `/tmp/hegel_continual_archive/archive_calibration_history.json`."
    )
    if pre_llm_diagnostic is not None:
        header += f"\n- Pre-LLM Public Gate Diagnostic: `{json.dumps(pre_llm_diagnostic)}`."

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
    extreme = sum(
        1 for signs in itertools.product((-1.0, 1.0), repeat=n) if abs(float(np.mean(arr * np.asarray(signs, dtype=float)))) >= obs - 1e-12
    )
    return round(float(extreme) / float(1 << n), 7)


def _bootstrap_mean_ci(diffs: list[float], seed: int = 96099, n_boot: int = 10000) -> tuple[float, float]:
    arr = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = np.mean(arr[idx], axis=1)
    return round(float(np.percentile(means, 2.5)), 2), round(float(np.percentile(means, 97.5)), 2)


def _run_one_m96_world_arm_stream(
    *,
    world_spec: M96WorldSpec,
    arm: str,
    stream_tasks: tuple[Any, ...],
    schedule_manifest: list[dict[str, Any]],
    pair_forest: Any,
    pair_archive: dict[str, dict[str, Any]],
    base_v7_forest: Any,
    base_v7_archive: dict[str, dict[str, Any]],
    base_graph_ckpt_path: Path,
    source_archive_entry: dict[str, Any],
    source_cal_summary: dict[str, Any],
    source_forest_nid: str,
    source_structural_features: dict[str, float | None],
    new_pair_skill_stems: set[str],
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

    # `Flat_Multiversion_Validated` uses frozen base v7 (`0` cross-domain Stage S forest growth, `0` compilation overhead!);
    # `Full_Mechanism_Graph_Autonomous` uses the pair-compiled forest (`allow_horizontal_wormholes = True`) + multi-parent graph!
    if arm == "Full_Mechanism_Graph_Autonomous":
        graph = PersistentMechanismGraph.load_checkpoint(base_graph_ckpt_path)
        forest = graph.forest
        archive = deepcopy(pair_archive)
        planner = M8VariationalPlanner(graph, min_gain_nats=0.25, collateral_tolerance_nats=-0.05)
        registry = M94BindingRegistry(graph)
    else:
        graph = None
        forest = deepcopy(base_v7_forest)
        archive = deepcopy(base_v7_archive)
        archive["__autogenous_blank_seed__"] = {"enabled": True}
        planner = None
        registry = None

    flat_snap: dict[str, Any] = (
        {"stress_gains": [1.0, 1.0, 1.0, 1.0], "stress_biases": [0.0, 0.0, 0.0, 0.0]}
        if fam == "alpha"
        else {"bank_gains": [1.0, 1.0, 1.0, 1.0], "bank_offsets": [0.0, 0.0, 0.0, 0.0]}
    )
    prior_entries: list[dict[str, Any]] = [deepcopy(source_archive_entry)]
    prior_cal_sweeps: list[dict[str, Any]] = [deepcopy(source_cal_summary)]
    positive_c0_witness_features: list[dict[str, float | None]] = [deepcopy(source_structural_features)]
    negative_c0_witness_features: list[dict[str, float | None]] = []
    uni_non_t_keys = list(source_archive_entry["output_schema_keys"])

    episode_results: list[dict[str, Any]] = []
    autonomous_induction_events: list[dict[str, Any]] = []

    comp_client = m6f.FrontierCodexClient(
        codex_info=codex_info, model=model, reasoning_effort=reasoning_effort, timeout_seconds=300
    )
    n_steps = len(stream_tasks)
    candidate_upstream_ids = (
        ["alpha_wing_a_calibration", "alpha_wing_b_calibration", "aux_bench_c_calibration", "sentinel_shared_acoustic_root"]
        if fam == "alpha"
        else ["beta_wing_a_calibration", "beta_wing_b_calibration", "aux_bench_c_calibration", "sentinel_shared_acoustic_root"]
    )

    for step_i, (task, m_info) in enumerate(zip(stream_tasks, schedule_manifest), start=1):
        ep_t0 = monotonic()
        ep_dir = stream_out_dir / task.task_id
        ep_dir.mkdir(parents=True, exist_ok=True)
        pub_item = _materialize_public_item_host(task)
        req_keys = expected_m96_public_output_keys(task)

        m8_update_rec: dict[str, Any] | None = None
        flat_wing_decisions: dict[str, Any] | None = None
        if arm == "Flat_Multiversion_Validated":
            snap_upd = update_flat_validated_wing_parameter_snapshot_from_public_item(
                family=fam, public_item=pub_item, prior_snapshot=flat_snap
            )
            flat_wing_decisions = snap_upd.get("wing_validation_decisions")
            flat_snap = {k: v for k, v in snap_upd.items() if k != "wing_validation_decisions"}
        else:
            assert graph is not None and planner is not None and registry is not None
            m8_update_rec = run_m96_multiparent_calibration_update(
                family=fam,
                episode_id=f"{wid}_{task.short_name}",
                public_item=pub_item,
                graph=graph,
                planner=planner,
                registry=registry,
            )

        cal_hist_json = json.dumps(prior_cal_sweeps, indent=2)
        readonly_deps = {"/tmp/hegel_continual_archive/archive_calibration_history.json": cal_hist_json}
        readonly_shas = {k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in readonly_deps.items()}

        probe_t0 = monotonic()
        pre_trial = run_m96_equal_capability_pre_llm_trial(
            arm=arm,
            family=fam,
            task=task,
            public_item=pub_item,
            prior_archive_entries=prior_entries,
            flat_param_snapshot=(flat_snap if arm == "Flat_Multiversion_Validated" else None),
            graph=graph,
            registry=registry,
            readonly_dependencies=readonly_deps,
            expected_dependency_sha256=readonly_shas,
        )
        probe_wall_ms = round((monotonic() - probe_t0) * 1000.0, 2)

        if pre_trial.get("accepted_pre_llm") is True:
            cand_code = str(pre_trial["candidate_code"])
            can_code_sel = str(pre_trial.get("selected_canonical_algorithm_code") or cand_code)
            can_ep_sel = str(pre_trial.get("selected_canonical_entrypoint") or task.declared_callable_entrypoints[0])
            art_path = str(task.declared_artifacts[0])
            bname = Path(art_path).name
            sdir = ep_dir / "04_submitted_artifacts_snapshot"
            sdir.mkdir(parents=True, exist_ok=True)
            (sdir / f"slot_00__{bname}").write_text(cand_code, encoding="utf-8")

            priv_ver = grade_m94_artifact_map(
                task,
                {art_path: cand_code},
                readonly_dependencies=readonly_deps,
                expected_dependency_sha256=readonly_shas,
            )
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

            if arm == "Full_Mechanism_Graph_Autonomous" and sorted(req_keys) == sorted(uni_non_t_keys):
                cur_feats = pre_trial.get("public_structural_features")
                if isinstance(cur_feats, dict):
                    if pre_trial.get("selected_branch_tag") == "C0_base":
                        positive_c0_witness_features.append(deepcopy(cur_feats))
                    elif pre_trial.get("selected_branch_tag") == "C1_extension":
                        negative_c0_witness_features.append(deepcopy(cur_feats))

            wall_ms = round((monotonic() - ep_t0) * 1000.0, 2)
            summary = {
                "run_id": run_id,
                "world_id": wid,
                "pair_cluster_id": world_spec.pair_cluster_id,
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
                "solver_cost_proxy_v1": 0,
                "solver_cost_proxy_v2": 0,
                "consolidation_cost_proxy_v1": 0,
                "consolidation_cost_proxy_v2": 0,
                "guard_induction_cost_proxy_v1": 0,
                "guard_induction_cost_proxy_v2": 0,
                "in_stream_total_cost_proxy_v1": 0,
                "in_stream_total_cost_proxy_v2": 0,
                "pre_llm_probe_wall_ms": probe_wall_ms,
                "wall_ms": wall_ms,
                "readonly_dependencies_sha256": readonly_shas,
                "flat_wing_validation_decisions": flat_wing_decisions,
                "m8_multiparent_update": m8_update_rec,
                "pre_llm_trial_receipt": {
                    k: v for k, v in pre_trial.items() if k not in {"candidate_code", "selected_canonical_algorithm_code"}
                },
                "provenance_dag_verification": dag_chk,
                "verifier_report": priv_ver,
            }
            (ep_dir / "00_episode_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            (ep_dir / "03_official_verifier_report.json").write_text(json.dumps(priv_ver, indent=2) + "\n", encoding="utf-8")

            prior_entries.append(
                make_continual_archive_entry(
                    short_name=task.short_name,
                    task_id=task.task_id,
                    canonical_entrypoint=can_ep_sel,
                    canonical_algorithm_code=can_code_sel,
                    output_schema_keys=req_keys,
                    public_qualification_status="public_gate_verified",
                    delivered_artifact_code=cand_code,
                    delivered_entrypoint=task.declared_callable_entrypoints[0],
                )
            )
            prior_cal_sweeps.append(_summarize_public_calibration_sweep(fam, pub_item, task.short_name))
            if graph is not None:
                graph.save_checkpoint(stream_out_dir / f"graph_ckpt_after_step_{step_i:02d}.json")
            episode_results.append(summary)
            print(
                f"  [M9.6 {wid} | {arm} | step {step_i}/{n_steps} {task.short_name} ({m_info['causal_event']})] "
                f"PRE_LLM_DIRECT ({pre_trial.get('selected_branch_tag') or pre_trial.get('selected_source_short_name')}) "
                f"| pass={summary['strict_pass']} | turns=0 | wall={wall_ms:.0f}ms",
                flush=True,
            )
            continue

        # Fallback to Live `gpt-6.2-sol` Episode (`18` turns)
        diag_for_prompt = {
            "arm": arm,
            "pre_llm_reason": pre_trial.get("reason"),
            "parameter_payload": pre_trial.get("parameter_payload"),
            "guard_evaluations": pre_trial.get("guard_evaluations"),
        }
        aug_task, ctx_hdr = _inject_equal_info_archive_into_task(task, prior_entries, prior_cal_sweeps, diag_for_prompt)
        ft = m96_task_to_frontier_task(
            aug_task,
            continual_context_block=ctx_hdr,
            allowed_readonly_dependencies=tuple(sorted(readonly_deps.keys())),
        )
        ep_run_tag = f"{run_id}__{wid}__{arm}"

        _CALL_CONTEXT.role = "stream_solve"
        _CALL_CONTEXT.world_id = wid
        _CALL_CONTEXT.pair_cluster_id = world_spec.pair_cluster_id
        _CALL_CONTEXT.arm = arm
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
            enable_auto_skill_packaging=(arm == "Full_Mechanism_Graph_Autonomous"),
            allow_l1_primitives=(arm == "Full_Mechanism_Graph_Autonomous"),
            disclosure_mode="on_demand_executable",
            staging_policy="net_value_gated",
        )
        sub_map = _extract_submitted_py(ep_dir, task)
        cand_txt = next(iter(sub_map.values()), "")
        cand_sha = hashlib.sha256(cand_txt.encode("utf-8")).hexdigest() if cand_txt else ""

        pub_gate_post = (
            execute_single_pre_llm_public_gate_probe_in_container(
                task=task,
                candidate_code=cand_txt,
                readonly_dependencies=readonly_deps,
                expected_dependency_sha256=readonly_shas,
            )
            if cand_txt
            else {"accepted": False}
        )
        priv_ver = (
            grade_m94_artifact_map(
                task,
                sub_map,
                readonly_dependencies=readonly_deps,
                expected_dependency_sha256=readonly_shas,
            )
            if sub_map
            else {"strict_pass": False, "error": "missing_submitted_artifact"}
        )
        dag_chk = m6f.append_post_episode_verification_to_dag(
            ep_dir,
            run_id=ep_run_tag,
            task_id=task.task_id,
            arm=arm,
            model=model,
            candidate_sha256=cand_sha,
            public_gate_receipt=(
                pub_gate_post.get("scientific_probe_receipt")
                or {"ended_by_finish": bool(ep_sum.get("ended_by_finish")), "verified_claim_ids": ["agent_public_gate_finish"]}
            ),
            verifier_report=priv_ver,
        )

        # Cross-Domain Adoption Funnel on `T` (`§4.4`)
        cross_domain_funnel = None
        if m_info["causal_event"] == "T":
            cross_domain_funnel = m64.classify_source_aware_adoption_funnel(
                episode_summary=ep_sum,
                full_skill_activations=ep_sum.get("full_skill_activations") or [],
                skill_inspections=ep_sum.get("skill_inspections") or [],
                new_source_l2_stems=new_pair_skill_stems,
                new_source_l1_stems=new_pair_skill_stems,
            )

        cons_v1 = cons_v2 = 0
        ind_v1 = ind_v2 = 0
        ind_receipt: dict[str, Any] | None = None

        # `Full_Mechanism_Graph_Autonomous` ONLY pays `Aufhebung` + Autonomous Guard/Dependency Induction costs!
        # (`Flat_Multiversion_Validated` pays `0` compilation/guard tax!)
        if arm == "Full_Mechanism_Graph_Autonomous" and sub_map and graph is not None and registry is not None:
            c1_fnid = source_forest_nid
            c1_stem = task.short_name
            c1_dig = cand_sha
            c1_fn = task.declared_callable_entrypoints[0]
            c1_qual = "hegel_public_scientific_gate_v4"

            if step_i < n_steps:
                ut_cid = m6f.start_detached_container(m7a.CLEAN_SCIPY_SANDBOX_IMAGE, memory_limit="1536m", cpus="2", network_mode="none")
                try:
                    _CALL_CONTEXT.role = "stream_compile"
                    comp_client._hegel_call_meta = {"role": "stream_compile", "world_id": wid, "pair_cluster_id": world_spec.pair_cluster_id, "arm": arm}  # type: ignore[attr-defined]
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
                    for cr in comp_res.get("provider_attempt_receipts") or []:
                        su = score_single_call_usage(
                            prompt_tokens=int(cr.get("prompt_tokens", 0) or 0),
                            cached_tokens=int(cr.get("cached_tokens", 0) or 0),
                            completion_tokens=int(cr.get("completion_tokens", 0) or 0),
                        )
                        cons_v1 += su["proxy_v1_u_025c_1o"]
                        cons_v2 += su["proxy_v2_u_010c_4o"]

                    if comp_res.get("compiled_ok"):
                        cons_r = m7a.consolidate_autogenous_episode_into_forest(
                            forest=forest,
                            archive=archive,
                            compilation=comp_res,
                            nogoods_recorded=ep_sum.get("nogoods_recorded") or [],
                            allow_horizontal_wormholes=True,
                        )
                        c1_fnid = str(cons_r["target_node_id"])
                        c1_stem = str(comp_res["module_stem"])
                        c1_dig = str(comp_res["compiled_code_sha256"])
                        c1_fn = comp_res["function_names"][0] if comp_res.get("function_names") else c1_fn
                        c1_qual = "hegel_l2_aufhebung_v2"
                finally:
                    m6f.stop_container_safely(ut_cid)

            # Autonomous Guard & Multi-Parent Induction (`0` host `if family == ...` guard hardcoding!)
            # Triggered whenever `C0` was schema-compatible (`req_keys == uni_non_t_keys`) yet failed pre-LLM AND `cand_txt` passed public gate!
            if pub_gate_post.get("accepted") and sorted(req_keys) == sorted(uni_non_t_keys) and step_i < n_steps:
                all_wings = candidate_upstream_ids[:2]
                cur_pay = assemble_multiparent_parameter_payload(family=fam, graph=graph, parent_mechanism_ids=all_wings)
                cur_feat_vec = compute_generic_public_structural_features(pub_item, cur_pay)
                negative_c0_witness_features.append(deepcopy(cur_feat_vec))

                c0_bind = registry.bindings[f"bind_{wid}_c0"]
                _CALL_CONTEXT.role = "autonomous_guard_induction"
                comp_client._hegel_call_meta = {"role": "autonomous_guard_induction", "world_id": wid, "pair_cluster_id": world_spec.pair_cluster_id, "arm": arm}  # type: ignore[attr-defined]
                ind_res = propose_and_contrastively_validate_autonomous_split(
                    client=comp_client,
                    stage_subdir=f"{ep_run_tag}__{task.short_name}",
                    family=fam,
                    task_instruction_text=task.instruction_text,
                    c0_code=c0_bind.canonical_algorithm_code,
                    c1_code=cand_txt,
                    c1_entrypoint=task.declared_callable_entrypoints[0],
                    positive_c0_witness_features=positive_c0_witness_features,
                    negative_c0_witness_features=negative_c0_witness_features,
                    current_public_item=pub_item,
                    current_param_payload={k: v for k, v in cur_pay.items() if k != "linked_parents"},
                    candidate_upstream_ids=candidate_upstream_ids,
                )
                for gr in ind_res.get("provider_call_receipts") or []:
                    su_g = score_single_call_usage(
                        prompt_tokens=int(gr.get("prompt_tokens", 0) or 0),
                        cached_tokens=int(gr.get("cached_tokens", 0) or 0),
                        completion_tokens=int(gr.get("completion_tokens", 0) or 0),
                    )
                    ind_v1 += su_g["proxy_v1_u_025c_1o"]
                    ind_v2 += su_g["proxy_v2_u_010c_4o"]

                ind_receipt = {k: v for k, v in ind_res.items() if k != "provider_call_receipts"}
                autonomous_induction_events.append(
                    {"step": step_i, "causal_event": m_info["causal_event"], "task_id": task.task_id, **ind_receipt}
                )

                if ind_res.get("accepted"):
                    val_parents = list(ind_res["validated_parent_mechanism_ids"])
                    primary_p = val_parents[0]
                    # Update `C0` with contrastively-validated guard (`structure_origin = 'publicly_contrastively_validated'`)!
                    c0_bind.precondition_guard = dict(ind_res["c0_guard"])
                    c0_bind.structure_origin = "publicly_contrastively_validated"
                    c0_bind.parent_parameter_mechanism_ids = val_parents

                    q_rec_c1 = registry.issue_qualification_receipt(
                        qualification_protocol=c1_qual,
                        verified_code_sha256=cand_sha,
                        forest_node_id=c1_fnid,
                        contract_schema_sha256=hashlib.sha256(json.dumps(sorted(req_keys)).encode("utf-8")).hexdigest(),
                        issued_reason=f"autonomous_c1_split_{task.short_name}",
                    )
                    c1_mech_id = f"mech_{wid}_c1"
                    if c1_mech_id in graph.mechanisms:
                        graph.mechanisms[c1_mech_id].parent_ids = tuple(val_parents)
                    registry.register_binding(
                        MechanismExecutableBinding(
                            binding_id=f"bind_{wid}_c1",
                            chain_id=fam,
                            branch_tag="C1_extension",
                            mechanism_id=c1_mech_id,
                            mechanism_version=1,
                            forest_node_id=c1_fnid,
                            component_id=str(ind_res["violated_component_id"]) + "_extension",
                            preserved_component_ids=list(ind_res["preserved_component_ids"]),
                            violated_component_id=str(ind_res["violated_component_id"]),
                            module_stem=c1_stem,
                            module_digest=c1_dig,
                            function_name=c1_fn,
                            qualification_id=c1_qual,
                            qualification_protocol=c1_qual,
                            qualification_receipt_id=q_rec_c1.receipt_id,
                            parameter_mechanism_id=primary_p,
                            parent_parameter_mechanism_ids=val_parents,
                            parameter_version=int(graph.mechanisms[primary_p].version),
                            structure_origin="publicly_contrastively_validated",
                            precondition_guard=dict(ind_res["c1_guard"]),
                            output_schema_keys=req_keys,
                            canonical_algorithm_code=cand_txt,
                            canonical_entrypoint=task.declared_callable_entrypoints[0],
                            input_coordinate_contract="raw_measured",
                            parameterized_closure_code=cand_txt,
                            parameterized_entrypoint=task.declared_callable_entrypoints[0],
                            parameterized_closure_sha256=cand_sha,
                        ),
                        reason=f"autonomous_contrastive_split_on_{task.short_name}",
                        strict_receipt_check=True,
                    )

        sol_sc = score_single_call_usage(
            prompt_tokens=int(ep_sum.get("prompt_tokens", 0) or 0),
            cached_tokens=int(ep_sum.get("cached_tokens", 0) or 0),
            completion_tokens=int(ep_sum.get("completion_tokens", 0) or 0),
        )
        sol_v1 = int(sol_sc["proxy_v1_u_025c_1o"])
        sol_v2 = int(ep_sum.get("token_cost_proxy", 0) or sol_sc["proxy_v2_u_010c_4o"])
        ep_sum["world_id"] = wid
        ep_sum["pair_cluster_id"] = world_spec.pair_cluster_id
        ep_sum["family"] = fam
        ep_sum["arm"] = arm
        ep_sum["stream_step"] = step_i
        ep_sum["causal_event"] = m_info["causal_event"]
        ep_sum["event_category"] = m_info["event_category"]
        ep_sum["execution_mode"] = "llm_synthesis_after_equal_tool_gate"
        ep_sum["strict_pass"] = bool(priv_ver.get("strict_pass"))
        ep_sum["reward"] = 1.0 if priv_ver.get("strict_pass") else 0.0
        ep_sum["solver_cost_proxy_v1"] = sol_v1
        ep_sum["solver_cost_proxy_v2"] = sol_v2
        ep_sum["consolidation_cost_proxy_v1"] = cons_v1
        ep_sum["consolidation_cost_proxy_v2"] = cons_v2
        ep_sum["guard_induction_cost_proxy_v1"] = ind_v1
        ep_sum["guard_induction_cost_proxy_v2"] = ind_v2
        ep_sum["in_stream_total_cost_proxy_v1"] = sol_v1 + cons_v1 + ind_v1
        ep_sum["in_stream_total_cost_proxy_v2"] = sol_v2 + cons_v2 + ind_v2
        ep_sum["pre_llm_probe_wall_ms"] = probe_wall_ms
        ep_sum["readonly_dependencies_sha256"] = readonly_shas
        ep_sum["public_qualification_status"] = "public_gate_verified" if pub_gate_post.get("accepted") else "public_gate_failed"
        ep_sum["flat_wing_validation_decisions"] = flat_wing_decisions
        ep_sum["m8_multiparent_update"] = m8_update_rec
        ep_sum["autonomous_split_induction"] = ind_receipt
        ep_sum["cross_domain_adoption_funnel"] = cross_domain_funnel
        ep_sum["pre_llm_trial_receipt"] = {k: v for k, v in pre_trial.items() if k != "candidate_code"}
        ep_sum["provenance_dag_verification"] = dag_chk
        ep_sum["verifier_report"] = priv_ver
        (ep_dir / "00_episode_summary.json").write_text(json.dumps(ep_sum, indent=2) + "\n", encoding="utf-8")
        (ep_dir / "03_official_verifier_report.json").write_text(json.dumps(priv_ver, indent=2) + "\n", encoding="utf-8")

        if cand_txt:
            prior_entries.append(
                make_continual_archive_entry(
                    short_name=task.short_name,
                    task_id=task.task_id,
                    canonical_entrypoint=task.declared_callable_entrypoints[0],
                    canonical_algorithm_code=cand_txt,
                    output_schema_keys=req_keys,
                    public_qualification_status=("public_gate_verified" if pub_gate_post.get("accepted") else "public_gate_failed"),
                )
            )
        prior_cal_sweeps.append(_summarize_public_calibration_sweep(fam, pub_item, task.short_name))
        if graph is not None:
            graph.save_checkpoint(stream_out_dir / f"graph_ckpt_after_step_{step_i:02d}.json")
        episode_results.append(ep_sum)
        print(
            f"  [M9.6 {wid} | {arm} | step {step_i}/{n_steps} {task.short_name} ({m_info['causal_event']})] "
            f"LLM_FALLBACK | pass={ep_sum['strict_pass']} | turns={ep_sum.get('turns_used')} "
            f"| sol_v2={sol_v2} | comp_v2={cons_v2} | guard_v2={ind_v2}",
            flush=True,
        )

    return {
        "world_id": wid,
        "pair_cluster_id": world_spec.pair_cluster_id,
        "family": fam,
        "arm": arm,
        "wall_sec": round(monotonic() - stream_t0, 2),
        "strict_passes": sum(1 for r in episode_results if r.get("strict_pass")),
        "episode_count": len(episode_results),
        "zero_turn_pre_llm_closures": sum(1 for r in episode_results if int(r.get("turns_used", 0) or 0) == 0 and r.get("strict_pass")),
        "total_turns_used": sum(int(r.get("turns_used", 0) or 0) for r in episode_results),
        "solver_cost_proxy_v1": sum(int(r.get("solver_cost_proxy_v1", 0) or 0) for r in episode_results),
        "solver_cost_proxy_v2": sum(int(r.get("solver_cost_proxy_v2", 0) or 0) for r in episode_results),
        "consolidation_cost_proxy_v1": sum(int(r.get("consolidation_cost_proxy_v1", 0) or 0) for r in episode_results),
        "consolidation_cost_proxy_v2": sum(int(r.get("consolidation_cost_proxy_v2", 0) or 0) for r in episode_results),
        "guard_induction_cost_proxy_v1": sum(int(r.get("guard_induction_cost_proxy_v1", 0) or 0) for r in episode_results),
        "guard_induction_cost_proxy_v2": sum(int(r.get("guard_induction_cost_proxy_v2", 0) or 0) for r in episode_results),
        "in_stream_total_cost_proxy_v1": sum(int(r.get("in_stream_total_cost_proxy_v1", 0) or 0) for r in episode_results),
        "in_stream_total_cost_proxy_v2": sum(int(r.get("in_stream_total_cost_proxy_v2", 0) or 0) for r in episode_results),
        "autonomous_induction_events": autonomous_induction_events,
        "binding_events": (registry.binding_events if registry is not None else []),
        "final_bindings": ({k: {bk: bv for bk, bv in b.to_dict().items() if "code" not in bk} for k, b in registry.bindings.items()} if registry is not None else {}),
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
                "solver_cost_proxy_v1": int(r.get("solver_cost_proxy_v1", 0) or 0),
                "solver_cost_proxy_v2": int(r.get("solver_cost_proxy_v2", 0) or 0),
                "consolidation_cost_proxy_v2": int(r.get("consolidation_cost_proxy_v2", 0) or 0),
                "guard_induction_cost_proxy_v2": int(r.get("guard_induction_cost_proxy_v2", 0) or 0),
                "in_stream_total_cost_proxy_v1": int(r.get("in_stream_total_cost_proxy_v1", 0) or 0),
                "in_stream_total_cost_proxy_v2": int(r.get("in_stream_total_cost_proxy_v2", 0) or 0),
                "selected_branch_tag": (r.get("pre_llm_trial_receipt") or {}).get("selected_branch_tag"),
                "selected_structure_origin": (r.get("pre_llm_trial_receipt") or {}).get("selected_structure_origin"),
                "m8_multiparent_update": r.get("m8_multiparent_update"),
                "cross_domain_adoption_funnel": r.get("cross_domain_adoption_funnel"),
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
    ap.add_argument("--num-worlds", type=int, default=8, choices=[2, 4, 8])
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
        raise RuntimeError("Refusing to start M9.6 confirmation experiment on a dirty working tree!")

    out_dir.mkdir(parents=True, exist_ok=True)
    mirror_dir = out_dir / "provider_raw_attempts_mirror"
    mirror_dir.mkdir(parents=True, exist_ok=True)

    ensure_clean_scipy_sandbox_image()
    m64.ensure_runtime_qualification_context(sandbox_cid=None)

    codex_info = m6f.discover_chatgpt_subscription_codex_cli()
    if not (codex_info.get("available") and codex_info.get("auth_mode") == "chatgpt"):
        raise RuntimeError(f"Official ChatGPT Codex CLI unavailable: {codex_info}")

    half_n = args.num_worlds // 2
    selected_worlds: list[M96WorldSpec] = (
        list(CONFIRMATION_M96_WORLD_SPECS[:half_n]) + list(CONFIRMATION_M96_WORLD_SPECS[4 : 4 + half_n])
    )
    print(
        f"[M9.6 INIT] Commit={commit_start[:12]} | SrcSHA={src_sha_start[:12]} | Worlds={len(selected_worlds)} "
        f"| Streams={len(selected_worlds)*len(M96_ARMS)} | CodexConcurrency={args.max_codex_concurrency}",
        flush=True,
    )

    cli_sem = threading.BoundedSemaphore(max(1, int(args.max_codex_concurrency)))
    master_lock = threading.Lock()
    master_provider_receipts: list[dict[str, Any]] = []
    active_counter = {"cur": 0, "peak": 0}
    _orig_call = m6f.FrontierCodexClient.call_structured

    def _throttled_and_mirrored_call(self_cli: Any, **kw: Any) -> dict[str, Any]:
        self_cli.immediate_archive_dir = mirror_dir
        cli_meta = getattr(self_cli, "_hegel_call_meta", None) or {}
        role_tag = str(cli_meta.get("role") or getattr(_CALL_CONTEXT, "role", "unspecified"))
        wid_tag = str(cli_meta.get("world_id") or getattr(_CALL_CONTEXT, "world_id", ""))
        pair_tag = str(cli_meta.get("pair_cluster_id") or getattr(_CALL_CONTEXT, "pair_cluster_id", ""))
        arm_tag = str(cli_meta.get("arm") or getattr(_CALL_CONTEXT, "arm", ""))
        with cli_sem:
            with master_lock:
                active_counter["cur"] += 1
                if active_counter["cur"] > active_counter["peak"]:
                    active_counter["peak"] = active_counter["cur"]
            try:
                res = _orig_call(self_cli, **kw)
            finally:
                with master_lock:
                    active_counter["cur"] -= 1
        sc = score_single_call_usage(
            prompt_tokens=int(res.get("prompt_tokens", 0) or 0),
            cached_tokens=int(res.get("cached_tokens", 0) or 0),
            completion_tokens=int(res.get("completion_tokens", 0) or 0),
            wall_ms=float(res.get("wall_ms", 0.0) or 0.0),
        )
        with master_lock:
            master_provider_receipts.append(
                {
                    "call_role": role_tag,
                    "world_id": wid_tag,
                    "pair_cluster_id": pair_tag,
                    "arm": arm_tag,
                    "stage_subdir": str(kw.get("stage_subdir") or ""),
                    "call_uuid": str(res.get("call_uuid") or ""),
                    "attempts_ledger_filename": str(res.get("attempts_ledger_filename") or ""),
                    "mirrored_raw_filename": f"{res.get('call_uuid')}__{res.get('attempts_ledger_filename')}",
                    "ok": bool(res.get("ok")),
                    "attempt_count": int(res.get("attempt_count", 1) or 1),
                    "usage_unknown_attempt_count": int(res.get("usage_unknown_attempt_count", 0) or 0),
                    "cost_accounting_status": str(res.get("cost_accounting_status") or "complete"),
                    **sc,
                }
            )
        return res

    m6f.FrontierCodexClient.call_structured = _throttled_and_mirrored_call  # type: ignore[method-assign]
    run_id = "m9_6_auto_struct_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    t0_all = monotonic()

    base_v7_forest, base_v7_archive, _, _ = m7a.load_autogenous_forest_bundle(args.v7_bundle)
    built_worlds = {w.world_id: build_m96_world_tasks(w) for w in selected_worlds}

    # =========================================================================
    # PHASE 1A: Blind `Stage S` Source Episodes (`8`-Way Parallel!)
    # =========================================================================
    src_out_root = out_dir / "stage_S_sources"

    def _run_world_source_episode(w_spec: M96WorldSpec) -> dict[str, Any]:
        s_task, _, _ = built_worlds[w_spec.world_id]
        for att_i in (1, 2):
            s_dir = src_out_root / w_spec.world_id / f"attempt_{att_i:02d}" / s_task.task_id
            ft_s = m96_task_to_frontier_task(s_task)
            ep_a = deepcopy(base_v7_archive)
            ep_a["__autogenous_blank_seed__"] = {"enabled": True}
            _CALL_CONTEXT.role = "stage_S_solve"
            _CALL_CONTEXT.world_id = w_spec.world_id
            _CALL_CONTEXT.pair_cluster_id = w_spec.pair_cluster_id
            _CALL_CONTEXT.arm = "Stage_S_Shared_Source"
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
                arm="Stage_S_Shared_Source",
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
                f"  [M9.6 STAGE S] {w_spec.world_id} ({w_spec.pair_cluster_id}, att={att_i}) | pass={ep_s['strict_pass']} | turns={ep_s['turns_used']} | proxy={ep_s['token_cost_proxy']}",
                flush=True,
            )
            if ep_s["strict_pass"]:
                return {
                    "world_spec": w_spec,
                    "s_task": s_task,
                    "ft_s": ft_s,
                    "ep_s": ep_s,
                    "sub_s": sub_s,
                    "cand_s_txt": cand_s_txt,
                    "cand_s_sha": cand_s_sha,
                }
        raise RuntimeError(f"Stage S source for `{w_spec.world_id}` failed private verifier across 2 blind attempts!")

    raw_src_by_wid: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=len(selected_worlds)) as pool_s:
        futs_s = [pool_s.submit(_run_world_source_episode, w) for w in selected_worlds]
        for fd in as_completed(futs_s):
            rw = fd.result()
            raw_src_by_wid[rw["world_spec"].world_id] = rw

    # =========================================================================
    # PHASE 1B: Pair-Scoped Cross-Domain `L2 + L1 + Wormhole` Freeze (`allow_horizontal_wormholes = True`)
    # (`4`-Way Parallel Across Disjoint Pairs `pair_01..pair_04`; `0` Author-Seeded Guards!)
    # =========================================================================
    pairs_map: dict[str, list[M96WorldSpec]] = {}
    for w in selected_worlds:
        pairs_map.setdefault(w.pair_cluster_id, []).append(w)

    def _compile_pair_and_freeze_worlds(pair_id: str, pair_worlds: list[M96WorldSpec]) -> dict[str, dict[str, Any]]:
        p_forest = deepcopy(base_v7_forest)
        p_archive = deepcopy(base_v7_archive)
        v7_prim_stems = {
            Path(str(m.get("filename") or "")).stem
            for _, rec in m7a.iter_live_archive_nodes(base_v7_archive)
            for m in (rec.get("autogenous_wormhole_modules") or [])
            if str(m.get("filename") or "").startswith("autogenous_prim_")
        }
        comp_cli = m6f.FrontierCodexClient(
            codex_info=codex_info, model=args.model, reasoning_effort=args.reasoning_effort, timeout_seconds=300
        )
        comp_cli._hegel_call_meta = {"role": "stage_S_compile_and_l1", "world_id": f"{pair_id}_src", "pair_cluster_id": pair_id, "arm": "Full_Mechanism_Graph_Autonomous"}  # type: ignore[attr-defined]
        new_stems: set[str] = set()
        src_meta_by_wid: dict[str, dict[str, Any]] = {}
        ut_cid = m6f.start_detached_container(m7a.CLEAN_SCIPY_SANDBOX_IMAGE, memory_limit="2048m", cpus="2", network_mode="none")
        try:
            for w_spec in sorted(pair_worlds, key=lambda x: x.world_id):
                comp_cli._hegel_call_meta = {"role": "stage_S_compile_and_l1", "world_id": w_spec.world_id, "pair_cluster_id": pair_id, "arm": "Full_Mechanism_Graph_Autonomous"}  # type: ignore[attr-defined]
                rw = raw_src_by_wid[w_spec.world_id]
                s_task = rw["s_task"]
                sigs_s = m6f.extract_public_task_signatures(rw["ft_s"])
                c_res = m7a.compile_episode_dialectical_sublation(
                    episode_id=f"m96_src_{w_spec.world_id}",
                    source_slug=s_task.short_name,
                    public_task_brief=s_task.instruction_text,
                    inferred_signatures=sigs_s,
                    raw_python_sources={Path(k).name: v for k, v in rw["sub_s"].items()},
                    nogoods_recorded=rw["ep_s"].get("nogoods_recorded") or [],
                    parent_node_summary=None,
                    client=comp_cli,
                    sandbox_cid=ut_cid,
                    stage_subdir=f"{run_id}__{w_spec.world_id}__src_l2",
                )
                if c_res.get("compiled_ok"):
                    c_res["regime_bifurcation"] = {**(c_res.get("regime_bifurcation") or {}), "should_split_node": False}
                    cons_s = m7a.consolidate_autogenous_episode_into_forest(
                        forest=p_forest,
                        archive=p_archive,
                        compilation=c_res,
                        nogoods_recorded=rw["ep_s"].get("nogoods_recorded") or [],
                        allow_horizontal_wormholes=True,
                    )
                    src_nid = str(cons_s["target_node_id"])
                    src_stem = str(c_res["module_stem"])
                    src_dig = str(c_res["compiled_code_sha256"])
                    src_fn = c_res["function_names"][0] if c_res.get("function_names") else s_task.declared_callable_entrypoints[0]
                    src_qual = "hegel_l2_aufhebung_v2"
                    new_stems.add(src_stem)
                else:
                    src_nid = next(iter(p_forest.nodes.keys()))
                    src_stem = s_task.short_name
                    src_dig = rw["cand_s_sha"]
                    src_fn = s_task.declared_callable_entrypoints[0]
                    src_qual = "hegel_public_scientific_gate_v4"
                src_meta_by_wid[w_spec.world_id] = {
                    "src_nid": src_nid,
                    "src_stem": src_stem,
                    "src_dig": src_dig,
                    "src_fn": src_fn,
                    "src_qual": src_qual,
                }
        finally:
            m6f.stop_container_safely(ut_cid)

        # Compile online L1 compositional primitives across the `(alpha, beta)` source pair (`allow_horizontal_wormholes = True`)!
        comp_cli._hegel_call_meta = {"role": "stage_S_compile_and_l1", "world_id": f"{pair_id}_l1", "pair_cluster_id": pair_id, "arm": "Full_Mechanism_Graph_Autonomous"}  # type: ignore[attr-defined]
        m7a.sublate_autogenous_forest_into_compositional_primitives(
            forest=p_forest,
            archive=p_archive,
            codex_client=comp_cli,
            run_id=f"{run_id}__{pair_id}__src_l1",
            require_pretraining_only=False,
            allow_horizontal_wormholes=True,
            skip_already_compiled_sources=True,
            max_workers=2,
            include_executable_checker_guidance=True,
            only_source_episode_ids={f"m96_src_{w.world_id}" for w in pair_worlds},
        )
        for _, rec_a in m7a.iter_live_archive_nodes(p_archive):
            for mod_a in rec_a.get("autogenous_wormhole_modules") or []:
                fn_a = str(mod_a.get("filename") or "")
                if fn_a.startswith("autogenous_prim_"):
                    st_a = Path(fn_a).stem
                    if st_a not in v7_prim_stems:
                        new_stems.add(st_a)
        m7a.discover_cross_tree_operator_wormholes(p_forest, p_archive)

        out_by_w: dict[str, dict[str, Any]] = {}
        for w_spec in pair_worlds:
            rw = raw_src_by_wid[w_spec.world_id]
            s_task = rw["s_task"]
            sm = src_meta_by_wid[w_spec.world_id]
            init_g, _, init_reg = initialize_m96_multiparent_mechanism_graph(family=w_spec.family, forest=deepcopy(p_forest))
            wing_parents = (
                ["alpha_wing_a_calibration", "alpha_wing_b_calibration"]
                if w_spec.family == "alpha"
                else ["beta_wing_a_calibration", "beta_wing_b_calibration"]
            )
            uni_keys = expected_m96_public_output_keys(s_task)
            q_rec_s = init_reg.issue_qualification_receipt(
                qualification_protocol=sm["src_qual"],
                verified_code_sha256=rw["cand_s_sha"],
                forest_node_id=sm["src_nid"],
                contract_schema_sha256=hashlib.sha256(json.dumps(sorted(uni_keys)).encode("utf-8")).hexdigest(),
                issued_reason=f"stage_S_public_verified_{w_spec.world_id}",
            )
            # ZERO AUTHOR-SEEDED GUARD (`precondition_guard = {}`, `structure_origin = 'unconstrained_initial_source'`)!
            init_reg.register_binding(
                MechanismExecutableBinding(
                    binding_id=f"bind_{w_spec.world_id}_c0",
                    chain_id=w_spec.family,
                    branch_tag="C0_base",
                    mechanism_id=f"mech_{w_spec.world_id}_c0",
                    mechanism_version=1,
                    forest_node_id=sm["src_nid"],
                    component_id="base_constitutive_mechanism",
                    preserved_component_ids=[],
                    violated_component_id=None,
                    module_stem=sm["src_stem"],
                    module_digest=sm["src_dig"],
                    function_name=sm["src_fn"],
                    qualification_id=sm["src_qual"],
                    qualification_protocol=sm["src_qual"],
                    qualification_receipt_id=q_rec_s.receipt_id,
                    parameter_mechanism_id=wing_parents[0],
                    parent_parameter_mechanism_ids=wing_parents,
                    parameter_version=1,
                    structure_origin="unconstrained_initial_source",
                    precondition_guard={},
                    output_schema_keys=uni_keys,
                    canonical_algorithm_code=rw["cand_s_txt"],
                    canonical_entrypoint=s_task.declared_callable_entrypoints[0],
                    input_coordinate_contract="raw_measured",
                    parameterized_closure_code=rw["cand_s_txt"],
                    parameterized_entrypoint=s_task.declared_callable_entrypoints[0],
                    parameterized_closure_sha256=rw["cand_s_sha"],
                ),
                reason=f"stage_S_unconstrained_multiparent_freeze_{w_spec.world_id}",
                strict_receipt_check=True,
            )
            w_freeze_dir = out_dir / "world_freezes" / w_spec.world_id
            w_freeze_dir.mkdir(parents=True, exist_ok=True)
            ckpt_p = w_freeze_dir / "initial_multiparent_graph_ckpt.json"
            ckpt_meta = init_g.save_checkpoint(ckpt_p)
            pub_s_item = _materialize_public_item_host(s_task)
            nom_pay = {"stress_gains": [1.0]*4, "stress_biases": [0.0]*4} if w_spec.family == "alpha" else {"bank_gains": [1.0]*4, "bank_offsets": [0.0]*4}
            out_by_w[w_spec.world_id] = {
                "world_id": w_spec.world_id,
                "pair_cluster_id": pair_id,
                "ep_summary": rw["ep_s"],
                "pair_forest": deepcopy(p_forest),
                "pair_archive": deepcopy(p_archive),
                "horizontal_wormhole_count": len(p_forest.horizontal_wormhole_edges),
                "new_pair_skill_stems": set(new_stems),
                "graph_ckpt_path": ckpt_p,
                "graph_ckpt_meta": ckpt_meta,
                "source_nid": sm["src_nid"],
                "source_structural_features": compute_generic_public_structural_features(pub_s_item, nom_pay),
                "source_archive_entry": make_continual_archive_entry(
                    short_name=s_task.short_name,
                    task_id=s_task.task_id,
                    canonical_entrypoint=s_task.declared_callable_entrypoints[0],
                    canonical_algorithm_code=rw["cand_s_txt"],
                    output_schema_keys=uni_keys,
                    public_qualification_status="public_gate_verified",
                ),
                "source_cal_summary": _summarize_public_calibration_sweep(w_spec.family, pub_s_item, s_task.short_name),
            }
        print(
            f"  [M9.6 PAIR FREEZE] {pair_id} | new_skills={sorted(new_stems)} | wormholes={len(p_forest.horizontal_wormhole_edges)}",
            flush=True,
        )
        return out_by_w

    world_init_by_id: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=len(pairs_map)) as pool_p:
        futs_p = [pool_p.submit(_compile_pair_and_freeze_worlds, pid, pws) for pid, pws in sorted(pairs_map.items())]
        for fp_done in as_completed(futs_p):
            world_init_by_id.update(fp_done.result())

    # =========================================================================
    # PHASE 2: `16` Parallel Independent Continual Streams (`8` Worlds x `2` Arms)
    # =========================================================================
    stream_jobs = [(w, arm_nm) for w in selected_worlds for arm_nm in M96_ARMS]
    stream_summaries: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.max_stream_workers))) as pool_st:
        futs_st = []
        for w_spec, arm_nm in stream_jobs:
            w_init = world_init_by_id[w_spec.world_id]
            _, s_tasks, s_man = built_worlds[w_spec.world_id]
            futs_st.append(
                pool_st.submit(
                    _run_one_m96_world_arm_stream,
                    world_spec=w_spec,
                    arm=arm_nm,
                    stream_tasks=s_tasks,
                    schedule_manifest=s_man,
                    pair_forest=w_init["pair_forest"],
                    pair_archive=w_init["pair_archive"],
                    base_v7_forest=base_v7_forest,
                    base_v7_archive=base_v7_archive,
                    base_graph_ckpt_path=w_init["graph_ckpt_path"],
                    source_archive_entry=w_init["source_archive_entry"],
                    source_cal_summary=w_init["source_cal_summary"],
                    source_forest_nid=w_init["source_nid"],
                    source_structural_features=w_init["source_structural_features"],
                    new_pair_skill_stems=w_init["new_pair_skill_stems"],
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

    stream_summaries.sort(key=lambda r: (r["world_id"], M96_ARMS.index(r["arm"])))

    # =========================================================================
    # PHASE 3: Forensic Assertions, 3-Scope CostPolicy & World/Pair Statistics
    # =========================================================================
    shutil.rmtree(out_dir / "_codex_stage", ignore_errors=True)
    commit_end, _, src_sha_end = _git_and_source_tree_state()
    tracked_dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no", "."],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert commit_start == commit_end, f"Git commit mutated mid-run: {commit_start} -> {commit_end}"
    assert src_sha_start == src_sha_end, f"Python source SHA-256 mutated mid-run: {src_sha_start} -> {src_sha_end}"
    assert not tracked_dirty, f"Tracked tree dirty at end: {tracked_dirty}"

    mirrored_raw_files = sorted(p.name for p in mirror_dir.glob("*__attempts_*.json"))
    assert len(mirrored_raw_files) == len(master_provider_receipts), (
        f"Raw mirror mismatch: {len(mirrored_raw_files)} != {len(master_provider_receipts)}"
    )

    by_world_arm = {(s["world_id"], s["arm"]): s for s in stream_summaries}
    world_ids_ord = [w.world_id for w in selected_worlds]

    # Exact 3-scope CostPolicy aggregation directly from `master_provider_receipts` (`call_role` tagged!)
    stage_s_solve_agg = aggregate_call_usages([r for r in master_provider_receipts if r["call_role"] == "stage_S_solve"])
    stage_s_comp_agg = aggregate_call_usages([r for r in master_provider_receipts if r["call_role"] == "stage_S_compile_and_l1"])

    arm_agg: dict[str, Any] = {}
    for arm_nm in M96_ARMS:
        a_streams = [by_world_arm[(wid, arm_nm)] for wid in world_ids_ord]
        a_eps = [ep for s in a_streams for ep in s["episodes"]]
        arm_calls_solve = [r for r in master_provider_receipts if r["arm"] == arm_nm and r["call_role"] == "stream_solve"]
        arm_calls_comp = [r for r in master_provider_receipts if r["arm"] == arm_nm and r["call_role"] == "stream_compile"]
        arm_calls_guard = [r for r in master_provider_receipts if r["arm"] == arm_nm and r["call_role"] == "autonomous_guard_induction"]
        in_stream_agg = aggregate_call_usages(arm_calls_solve + arm_calls_comp + arm_calls_guard)
        # Scope 2: `Flat` pays ONLY `stage_S_solve` + its own stream; `Full` pays `stage_S_solve` + 100% of `stage_S_compile_and_l1` + its own stream!
        standalone_calls = (
            [r for r in master_provider_receipts if r["call_role"] == "stage_S_solve"]
            + ([r for r in master_provider_receipts if r["call_role"] == "stage_S_compile_and_l1"] if arm_nm == "Full_Mechanism_Graph_Autonomous" else [])
            + arm_calls_solve
            + arm_calls_comp
            + arm_calls_guard
        )
        standalone_agg = aggregate_call_usages(standalone_calls)

        by_event = {}
        for ev_k in ("N", "G1", "T", "C", "R", "U", "G2", "M"):
            ev_eps = [e for e in a_eps if e["causal_event"] == ev_k]
            by_event[ev_k] = {
                "strict_passes": sum(1 for e in ev_eps if e["strict_pass"]),
                "episode_count": len(ev_eps),
                "zero_turn_closures": sum(1 for e in ev_eps if e["turns_used"] == 0 and e["strict_pass"]),
                "mean_turns": round(float(np.mean([e["turns_used"] for e in ev_eps])), 2) if ev_eps else 0.0,
                "in_stream_cost_proxy_v1": sum(int(e["in_stream_total_cost_proxy_v1"]) for e in ev_eps),
                "in_stream_cost_proxy_v2": sum(int(e["in_stream_total_cost_proxy_v2"]) for e in ev_eps),
            }
        arm_agg[arm_nm] = {
            "world_count": len(a_streams),
            "episode_count": len(a_eps),
            "strict_passes": sum(1 for e in a_eps if e["strict_pass"]),
            "pass_rate": round(sum(1 for e in a_eps if e["strict_pass"]) / max(len(a_eps), 1), 4),
            "zero_turn_pre_llm_closures": sum(1 for e in a_eps if e["turns_used"] == 0 and e["strict_pass"]),
            "total_turns_used": sum(int(e["turns_used"]) for e in a_eps),
            "mean_turns_per_episode": round(float(np.mean([e["turns_used"] for e in a_eps])), 2),
            "scope_1_in_stream_solve_only": aggregate_call_usages(arm_calls_solve),
            "scope_1_in_stream_compile_only": aggregate_call_usages(arm_calls_comp),
            "scope_1_in_stream_guard_induction_only": aggregate_call_usages(arm_calls_guard),
            "scope_1_in_stream_total": in_stream_agg,
            "scope_2_standalone_with_stage_S": standalone_agg,
            "by_causal_event": by_event,
        }

    # World-clustered (`n = 8`) AND Pair-clustered (`n = 4` completely disjoint pairs!) paired statistics (`Full - Flat`)
    v1_diffs, v2_diffs, turn_diffs, pass_diffs, zt_diffs = [], [], [], [], []
    per_world_rows = []
    for wid in world_ids_ord:
        st_f = by_world_arm[(wid, "Full_Mechanism_Graph_Autonomous")]
        st_b = by_world_arm[(wid, "Flat_Multiversion_Validated")]
        dv1 = int(st_f["in_stream_total_cost_proxy_v1"]) - int(st_b["in_stream_total_cost_proxy_v1"])
        dv2 = int(st_f["in_stream_total_cost_proxy_v2"]) - int(st_b["in_stream_total_cost_proxy_v2"])
        dt = int(st_f["total_turns_used"]) - int(st_b["total_turns_used"])
        dp = int(st_f["strict_passes"]) - int(st_b["strict_passes"])
        dzt = int(st_f["zero_turn_pre_llm_closures"]) - int(st_b["zero_turn_pre_llm_closures"])
        v1_diffs.append(float(dv1)); v2_diffs.append(float(dv2)); turn_diffs.append(float(dt)); pass_diffs.append(float(dp)); zt_diffs.append(float(dzt))
        per_world_rows.append(
            {
                "world_id": wid,
                "pair_cluster_id": st_f["pair_cluster_id"],
                "in_stream_v1_diff": dv1,
                "in_stream_v2_diff": dv2,
                "turns_diff": dt,
                "strict_passes_diff": dp,
                "zero_turn_closures_diff": dzt,
            }
        )

    pair_rows = []
    for pid in sorted(pairs_map.keys()):
        p_wids = [w.world_id for w in pairs_map[pid]]
        pv1 = sum(int(by_world_arm[(w, "Full_Mechanism_Graph_Autonomous")]["in_stream_total_cost_proxy_v1"]) - int(by_world_arm[(w, "Flat_Multiversion_Validated")]["in_stream_total_cost_proxy_v1"]) for w in p_wids)
        pv2 = sum(int(by_world_arm[(w, "Full_Mechanism_Graph_Autonomous")]["in_stream_total_cost_proxy_v2"]) - int(by_world_arm[(w, "Flat_Multiversion_Validated")]["in_stream_total_cost_proxy_v2"]) for w in p_wids)
        pp = sum(int(by_world_arm[(w, "Full_Mechanism_Graph_Autonomous")]["strict_passes"]) - int(by_world_arm[(w, "Flat_Multiversion_Validated")]["strict_passes"]) for w in p_wids)
        pair_rows.append({"pair_cluster_id": pid, "worlds": p_wids, "in_stream_v1_diff": pv1, "in_stream_v2_diff": pv2, "strict_passes_diff": pp})

    tot_f_v1 = arm_agg["Full_Mechanism_Graph_Autonomous"]["scope_1_in_stream_total"]["proxy_v1_u_025c_1o"]
    tot_b_v1 = arm_agg["Flat_Multiversion_Validated"]["scope_1_in_stream_total"]["proxy_v1_u_025c_1o"]
    tot_f_v2 = arm_agg["Full_Mechanism_Graph_Autonomous"]["scope_1_in_stream_total"]["proxy_v2_u_010c_4o"]
    tot_b_v2 = arm_agg["Flat_Multiversion_Validated"]["scope_1_in_stream_total"]["proxy_v2_u_010c_4o"]
    tot_f_s2_v2 = arm_agg["Full_Mechanism_Graph_Autonomous"]["scope_2_standalone_with_stage_S"]["proxy_v2_u_010c_4o"]
    tot_b_s2_v2 = arm_agg["Flat_Multiversion_Validated"]["scope_2_standalone_with_stage_S"]["proxy_v2_u_010c_4o"]

    paired_stats = {
        "treatment": "Full_Mechanism_Graph_Autonomous",
        "baseline": "Flat_Multiversion_Validated",
        "n_worlds": len(world_ids_ord),
        "n_disjoint_pair_clusters": len(pair_rows),
        "worlds_with_lower_in_stream_v1": sum(1 for d in v1_diffs if d < 0),
        "worlds_with_lower_in_stream_v2": sum(1 for d in v2_diffs if d < 0),
        "disjoint_pairs_with_lower_in_stream_v2": sum(1 for r in pair_rows if r["in_stream_v2_diff"] < 0),
        "in_stream_v1_total_delta": tot_f_v1 - tot_b_v1,
        "in_stream_v1_relative_pct": round((tot_f_v1 - tot_b_v1) / max(tot_b_v1, 1) * 100.0, 2),
        "in_stream_v1_two_sided_sign_flip_p": _exact_two_sided_sign_flip_p_value(v1_diffs),
        "in_stream_v2_total_delta": tot_f_v2 - tot_b_v2,
        "in_stream_v2_relative_pct": round((tot_f_v2 - tot_b_v2) / max(tot_b_v2, 1) * 100.0, 2),
        "in_stream_v2_world_bootstrap_95ci": list(_bootstrap_mean_ci(v2_diffs)),
        "in_stream_v2_two_sided_sign_flip_p": _exact_two_sided_sign_flip_p_value(v2_diffs),
        "standalone_with_stage_S_v2_relative_pct": round((tot_f_s2_v2 - tot_b_s2_v2) / max(tot_b_s2_v2, 1) * 100.0, 2),
        "turns_mean_world_delta": round(float(np.mean(turn_diffs)), 2),
        "turns_two_sided_sign_flip_p": _exact_two_sided_sign_flip_p_value(turn_diffs),
        "zero_turn_closures_total_delta": int(round(sum(zt_diffs))),
        "zero_turn_closures_two_sided_sign_flip_p": _exact_two_sided_sign_flip_p_value(zt_diffs),
        "strict_passes_total_delta": int(round(sum(pass_diffs))),
        "strict_passes_two_sided_sign_flip_p": _exact_two_sided_sign_flip_p_value(pass_diffs),
        "per_world_deltas": per_world_rows,
        "per_disjoint_pair_deltas": pair_rows,
    }

    # Autonomous Structure Induction Funnel (`§4.1 + §4.3 + §4.5`)
    full_streams = [by_world_arm[(w, "Full_Mechanism_Graph_Autonomous")] for w in world_ids_ord]
    all_ind_events = [ev for s in full_streams for ev in s["autonomous_induction_events"]]
    u_eps_full = [e for s in full_streams for e in s["episodes"] if e["causal_event"] == "U"]
    u_eps_flat = [e for w in world_ids_ord for e in by_world_arm[(w, "Flat_Multiversion_Validated")]["episodes"] if e["causal_event"] == "U"]
    structure_funnel = {
        "author_hardcoded_regime_guards_in_runner": 0,
        "total_autonomous_split_inductions_triggered": len(all_ind_events),
        "contrastively_and_sensitivity_validated_splits": sum(1 for e in all_ind_events if e.get("accepted")),
        "rejected_split_proposals": sum(1 for e in all_ind_events if not e.get("accepted")),
        "downstream_zero_turn_closures_via_validated_guards": sum(
            1
            for s in full_streams
            for e in s["episodes"]
            if e["turns_used"] == 0 and e["strict_pass"] and e.get("selected_structure_origin") == "publicly_contrastively_validated"
        ),
        "false_accept_public_pass_private_fail_count": sum(
            1 for s in full_streams for e in s["episodes"] if e["turns_used"] == 0 and not e["strict_pass"]
        ),
        "non_markovian_U_zero_turn_closures_Full": sum(1 for e in u_eps_full if e["turns_used"] == 0 and e["strict_pass"]),
        "non_markovian_U_zero_turn_closures_Flat": sum(1 for e in u_eps_flat if e["turns_used"] == 0 and e["strict_pass"]),
        "non_markovian_U_strict_passes_Full": sum(1 for e in u_eps_full if e["strict_pass"]),
        "non_markovian_U_strict_passes_Flat": sum(1 for e in u_eps_flat if e["strict_pass"]),
        "induced_guards_by_world": {
            s["world_id"]: [
                {"step": ev["step"], "event": ev["causal_event"], "accepted": ev["accepted"], "c0_guard": ev.get("c0_guard"), "c1_guard": ev.get("c1_guard"), "parents": ev.get("validated_parent_mechanism_ids")}
                for ev in s["autonomous_induction_events"]
            ]
            for s in full_streams
        },
    }

    all_dag_files = sorted(out_dir.rglob("01_decision_provenance_dag.json"))
    dag_valid_count = sum(1 for fp in all_dag_files if json.loads(fp.read_text(encoding="utf-8"))["provenance_verification"]["valid"])

    master_summary = {
        "schema": "hegel_m9_6_autonomous_structure_continual_summary_v1",
        "cost_policy_version": COST_POLICY_VERSION,
        "run_id": run_id,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "configured_codex_concurrency_cap": args.max_codex_concurrency,
        "peak_concurrent_active_codex_calls_observed": active_counter["peak"],
        "configured_stream_workers": args.max_stream_workers,
        "code_commit_at_start": commit_start,
        "code_commit_at_end": commit_end,
        "source_tree_py_sha256_at_start": src_sha_start,
        "source_tree_py_sha256_at_end": src_sha_end,
        "tracked_tree_dirty_at_end": bool(tracked_dirty),
        "wall_clock_sec": round(monotonic() - t0_all, 2),
        "stage_S_cost_ledger": {
            "shared_stage_S_solve_8worlds": stage_s_solve_agg,
            "full_arm_only_stage_S_l2_l1_compile": stage_s_comp_agg,
        },
        "scope_3_campaign_physical_spend_all_calls": aggregate_call_usages(master_provider_receipts),
        "arm_comparison": arm_agg,
        "paired_statistics_Full_vs_Flat": paired_stats,
        "autonomous_structure_induction_funnel": structure_funnel,
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
    (out_dir / "m9_6_continual_summary.json").write_text(json.dumps(master_summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "all_provider_call_receipts.json").write_text(json.dumps(master_provider_receipts, indent=2) + "\n", encoding="utf-8")

    sha_rows = []
    for fp in sorted(out_dir.rglob("*")):
        if fp.is_file() and fp.name != "SHA256SUMS.txt":
            sha_rows.append(f"{hashlib.sha256(fp.read_bytes()).hexdigest()}  {fp.relative_to(out_dir).as_posix()}")
    (out_dir / "SHA256SUMS.txt").write_text("\n".join(sha_rows) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "arm_comparison": {
                    k: {
                        "strict_passes": v["strict_passes"],
                        "zero_turn_closures": v["zero_turn_pre_llm_closures"],
                        "in_stream_v1": v["scope_1_in_stream_total"]["proxy_v1_u_025c_1o"],
                        "in_stream_v2": v["scope_1_in_stream_total"]["proxy_v2_u_010c_4o"],
                        "standalone_v2": v["scope_2_standalone_with_stage_S"]["proxy_v2_u_010c_4o"],
                    }
                    for k, v in arm_agg.items()
                },
                "paired_statistics": paired_stats,
                "autonomous_structure_funnel": structure_funnel,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
