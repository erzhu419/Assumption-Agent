#!/usr/bin/env python3
"""End-to-End `32`-Way Parallel `gpt-6.2-sol` CLI Runner for `M9.7`:
Non-Nested Multi-Conflict Lineage, Strict Present-Matched Counterfactual Memory (`R0` vs `U_iso`),
Anonymized Permuted Multi-Parent Calibration + Wired Nuisance Parent, and Demand-Gated Consolidation.

Cluster-Independent Pre-Registered Protocol (`n = 6` Completely Independent Source-World Clusters):
  - `3` Independent Alpha Clusters (`alpha_m97_c01..c03`) + `3` Independent Beta Clusters (`beta_m97_c01..c03`).
  - `7` Anonymous Continual Steps per Cluster (`42` continual episodes per arm = `84` stream episodes + `6` Stage S = `90` episodes):
      `Step 1 (C1) -> Step 2 (R0) -> Step 3 (C2) -> Step 4 (R1) -> Step 5 (U_iso) -> Step 6 (G_perm) -> Step 7 (M_couple)`.
  - Head-to-Head Unhandicapped Comparison:
      1. `Flat_EqualInfo_Multiversion`: 100% of `M9.5.1 + M9.6.1` fixes (`F01` 1-hop canonical export, `F02` `0444` prefix
         history, `F03` validated `2/3–1/3` per-channel OLS, full `/tmp/hegel_continual_archive/` access with complete freedom
         to write unified/piecewise/conditional solvers, `max_pre_llm_probes = 1`, `0` compilation tax).
      2. `Hegel_DemandGated_Graph`: Immutable versioned branch lineage (`b00, b01, b02`), evidence-bound `SolverEvaluationWitness`
         (`0` fabricated negative witnesses), `EmpiricalCoverageReceipt` check before guard induction, neutral primitive
         observable DSL (`obs_01..obs_08` + Kleene `all_of`), containerized permuted-parent sensitivity (`cal_node_p0..p3`
         including wired `aux_telemetry` nuisance `cal_node_p2`), AND **M8 Demand-Gated Consolidation** (`0` unconditional
         post-episode L2 compilation when canonical bindings cover the contract, while retaining V7 L1/L2 skills!).
  - Exact Call-UUID Hierarchical Cost Conservation (`N01`) asserted across `episodes == events == clusters == arm` for BOTH `v1` & `v2`!
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
    assert_hierarchical_ledger_conservation,
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
)
from hegel_machine.cognition.m9_6_autonomous_structure_continual_world import (
    ImmutableBranchLineageController,
)
from hegel_machine.cognition.m9_7_non_nested_counterfactual_lineage_world import (
    CONFIRMATION_M97_WORLD_SPECS,
    M97_ARMS,
    M97WorldSpec,
    assemble_m97_parameter_payload,
    build_m97_world_tasks,
    compute_m97_primitive_observables,
    evaluate_empirical_coverage_on_branch_witnesses,
    expected_m97_public_output_keys,
    initialize_m97_permuted_parent_graph,
    m97_task_to_frontier_task,
    propose_and_validate_m97_multibranch_split,
    run_m97_equal_capability_pre_llm_trial,
    update_m97_flat_validated_channel_snapshot,
    update_m97_permuted_calibration_graph,
)

DEFAULT_V7_BUNDLE = REPO_ROOT / "docs/evidence/m9_2_1_audit_closure_and_regrade_20261126/pretrained_autogenous_forest_v7_m921.json"
DEFAULT_OUT_DIR = REPO_ROOT / "docs/evidence/m9_7_non_nested_counterfactual_lineage_run3_20261127"
ABORTED_RUN_MANIFESTS = (
    REPO_ROOT / "docs/evidence/m9_7_aborted_run1_openblas_setup_timeout_20261127/00_aborted_run1_forensic_manifest.json",
    REPO_ROOT / "docs/evidence/m9_7_aborted_run2_beta_self_guard_witness_kind_20261127/00_aborted_run2_forensic_manifest.json",
)

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


def _docker_image_digest(image_name: str) -> str:
    r = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", image_name], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else f"image:{image_name}"


def _materialize_public_item_and_sha(task: Any) -> tuple[dict[str, Any], str]:
    with tempfile.TemporaryDirectory(prefix="m97_hpub_") as td:
        td_p = Path(td)
        ns: dict[str, Any] = {}
        exec(compile(task.public_setup_py.replace("/holdout_task", str(td_p)), f"<pub_{task.short_name}>", "exec"), ns)
        raw_b = (td_p / f"public_{task.short_name}.pkl").read_bytes()
        return pickle.loads(raw_b)[0], hashlib.sha256(raw_b).hexdigest()


def _extract_submitted_py(ep_dir: Path, task: Any) -> dict[str, str]:
    sdir = ep_dir / "04_submitted_artifacts_snapshot"
    art_rel = str(task.declared_artifacts[0]).lstrip("/")
    bname = Path(art_rel).name
    hit = next((cp for cp in (sdir / art_rel, sdir / f"slot_00__{bname}", sdir / bname) if cp.is_file()), None)
    if hit is None:
        py_files = [p for p in sorted(sdir.rglob("*.py")) if p.name.endswith(bname)] or sorted(sdir.rglob("*.py"))
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
    """Identical `/tmp/hegel_continual_archive/` staging across BOTH arms (`Flat` is 100% free to write piecewise/unified solvers!)."""
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
        f"- Prior publicly-verified canonical stream solvers (`raw_measured` inputs; you may inspect, reuse, or compose piecewise dispatchers freely): {[r['solver_path'] for r in manifest_rows]}.\n"
        "- Manifest: `/tmp/hegel_continual_archive/archive_manifest.json`.\n"
        "- Read-only (`0444`) prefix calibration sweeps (`<= t-1`, also available in private verifier): `/tmp/hegel_continual_archive/archive_calibration_history.json`."
    )
    if pre_llm_diagnostic is not None:
        header += f"\n- Pre-LLM Public Gate Diagnostic: `{json.dumps(pre_llm_diagnostic)}`."
    aug_task = type(task)(**{**task.__dict__, "public_setup_py": task.public_setup_py + "\n" + "\n".join(extra_lines) + "\n"})
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
        1 for signs in itertools.product((-1.0, 1.0), repeat=n)
        if abs(float(np.mean(arr * np.asarray(signs, dtype=float)))) >= obs - 1e-12
    )
    return round(float(extreme) / float(1 << n), 7)


def _bootstrap_mean_ci(diffs: list[float], seed: int = 97099, n_boot: int = 10000) -> list[float]:
    arr = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = np.mean(arr[idx], axis=1)
    return [round(float(np.percentile(means, 2.5)), 2), round(float(np.percentile(means, 97.5)), 2)]


def _evaluate_demand_gated_l2_consolidation_policy(
    *,
    remaining_steps: int,
    canonical_binding_covers_contract: bool,
    estimated_compile_v2_cost: int = 75000,
) -> dict[str, Any]:
    """`M9.6.1 / M9.7` `N08` Demand-Gated Consolidation Policy:
    Evaluates `E[N_compatible_llm_reuse] * E[Delta_C_solve] - C_compile`.
    When `canonical_binding_covers_contract == True`, future same-contract episodes are served `0`-turn by
    `MechanismExecutableBinding` + `ImmutableBranchLineageController` (`E[N_compatible_llm_reuse] == 0`), so
    synchronous post-episode L2 compilation has strictly negative expected net value (`-75,000 v2`) and is deferred!
    """
    exp_llm_reuse = 0.0 if (remaining_steps <= 0 or canonical_binding_covers_contract) else float(remaining_steps) * 0.25
    exp_net_v2 = round(exp_llm_reuse * 45000.0 - float(estimated_compile_v2_cost), 2)
    return {
        "policy": "m8_expected_net_value_demand_gate_v1",
        "effective_rule_on_same_contract_stream": "disable_eager_l2_when_canonical_binding_covers_contract",
        "remaining_steps": remaining_steps,
        "canonical_binding_covers_contract": canonical_binding_covers_contract,
        "expected_compatible_llm_reuse_episodes": exp_llm_reuse,
        "estimated_compile_cost_v2": estimated_compile_v2_cost,
        "expected_net_value_v2": exp_net_v2,
        "should_compile_synchronously": bool(exp_net_v2 > 0.0),
    }


def _run_one_m97_world_arm_stream(
    *,
    spec: M97WorldSpec,
    arm: str,
    s_task: Any,
    stream_tasks: tuple[Any, ...],
    schedule_manifest: list[dict[str, Any]],
    base_v7_forest: Any,
    base_v7_archive: dict[str, dict[str, Any]],
    prior_v7_l1_stems: set[str],
    base_graph_ckpt_path: Path,
    parent_wiring_meta: dict[str, Any],
    source_archive_entry: dict[str, Any],
    source_cal_summary: dict[str, Any],
    source_forest_nid: str,
    source_pub_sha256: str,
    source_observables: dict[str, float | None],
    env_image_digest: str,
    model: str,
    codex_info: dict[str, Any],
    stream_out_dir: Path,
    run_id: str,
    max_turns: int,
    reasoning_effort: str,
) -> dict[str, Any]:
    stream_t0 = monotonic()
    fam = spec.family
    wid = spec.world_id
    uni_keys = expected_m97_public_output_keys(fam)

    forest = deepcopy(base_v7_forest)
    archive = deepcopy(base_v7_archive)
    archive["__autogenous_blank_seed__"] = {"enabled": True}

    if arm == "Hegel_DemandGated_Graph":
        graph = PersistentMechanismGraph.load_checkpoint(base_graph_ckpt_path)
        forest = graph.forest
        planner = M8VariationalPlanner(graph, min_gain_nats=0.25, collateral_tolerance_nats=-0.05)
        registry = M94BindingRegistry(graph)
        lineage_ctrl = ImmutableBranchLineageController(world_id=wid, family=fam, graph=graph, registry=registry)
        b0 = registry.bindings[f"bind_{wid}_b00"]
        lineage_ctrl.record_tested_witness(
            binding=b0,
            public_input_sha256=source_pub_sha256,
            passed=True,
            failure_kind="none",
            features=source_observables,
            probe_wall_ms=0.0,
        )
        # Keep representative positive witness task per branch for `EmpiricalCoverageReceipt` checks (`§5`)!
        rep_positive_witness_by_bid: dict[str, list[tuple[str, Any, dict[str, Any], dict[str, str], dict[str, str]]]] = {
            b0.binding_id: [
                (
                    f"wit_src_{wid}",
                    s_task,
                    ({"stress_gains": [1.0] * 4, "stress_biases": [0.0] * 4} if fam == "alpha" else {"bank_gains": [1.0] * 4, "bank_offsets": [0.0] * 4}),
                    {},
                    {},
                )
            ]
        }
    else:
        graph = None
        planner = None
        registry = None
        lineage_ctrl = None
        rep_positive_witness_by_bid = {}

    flat_snap: dict[str, Any] = (
        {"stress_gains": [1.0, 1.0, 1.0, 1.0], "stress_biases": [0.0, 0.0, 0.0, 0.0]}
        if fam == "alpha"
        else {"bank_gains": [1.0, 1.0, 1.0, 1.0], "bank_offsets": [0.0, 0.0, 0.0, 0.0]}
    )
    prior_entries: list[dict[str, Any]] = [deepcopy(source_archive_entry)]
    prior_cal_sweeps: list[dict[str, Any]] = [deepcopy(source_cal_summary)]

    episode_results: list[dict[str, Any]] = []
    lineage_induction_events: list[dict[str, Any]] = []
    coverage_receipts_log: list[dict[str, Any]] = []
    consolidation_gate_decisions: list[dict[str, Any]] = []

    comp_client = m6f.FrontierCodexClient(
        codex_info=codex_info, model=model, reasoning_effort=reasoning_effort, timeout_seconds=300
    )
    n_steps = len(stream_tasks)

    for step_i, (task, m_info) in enumerate(zip(stream_tasks, schedule_manifest), start=1):
        ep_t0 = monotonic()
        ep_dir = stream_out_dir / task.task_id
        ep_dir.mkdir(parents=True, exist_ok=True)
        pub_item, pub_sha = _materialize_public_item_and_sha(task)

        m8_update_rec: dict[str, Any] | None = None
        if arm == "Flat_EqualInfo_Multiversion":
            flat_snap = update_m97_flat_validated_channel_snapshot(
                family=fam, public_item=pub_item, prior_snapshot=flat_snap
            )
        else:
            assert graph is not None and planner is not None and registry is not None
            m8_update_rec = update_m97_permuted_calibration_graph(
                spec=spec,
                episode_id=f"{wid}_{task.short_name}",
                public_item=pub_item,
                graph=graph,
                planner=planner,
                registry=registry,
            )

        cal_hist_json = json.dumps(prior_cal_sweeps, indent=2)
        readonly_deps = {"/tmp/hegel_continual_archive/archive_calibration_history.json": cal_hist_json}
        readonly_shas = {k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in readonly_deps.items()}

        pre_trial = run_m97_equal_capability_pre_llm_trial(
            arm=arm,
            spec=spec,
            task=task,
            public_item=pub_item,
            public_input_sha256=pub_sha,
            prior_archive_entries=prior_entries,
            flat_param_snapshot=(flat_snap if arm == "Flat_EqualInfo_Multiversion" else None),
            lineage_ctrl=lineage_ctrl,
            graph=graph,
            registry=registry,
            readonly_dependencies=readonly_deps,
            expected_dependency_sha256=readonly_shas,
        )

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
            wall_ms = round((monotonic() - ep_t0) * 1000.0, 2)
            summary = {
                "run_id": run_id,
                "world_id": wid,
                "cluster_id": spec.cluster_id,
                "family": fam,
                "arm": arm,
                "stream_step": step_i,
                "causal_event": m_info["causal_event"],
                "true_regime": m_info["true_regime"],
                "event_category": m_info["event_category"],
                "task_id": task.task_id,
                "short_name": task.short_name,
                "execution_mode": "pre_llm_equal_tool_gate_accepted",
                "strict_pass": bool(priv_ver.get("strict_pass")),
                "reward": 1.0 if priv_ver.get("strict_pass") else 0.0,
                "turns_used": 0,
                "ended_by_finish": True,
                "pre_llm_probe_wall_ms": float(pre_trial.get("probe_wall_ms", 0.0)),
                "wall_ms": wall_ms,
                "readonly_dependencies_sha256": readonly_shas,
                "m8_permuted_parent_update": m8_update_rec,
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
                    output_schema_keys=uni_keys,
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
                f"  [M9.7 {wid} | {arm} | step {step_i}/{n_steps} {task.short_name} ({m_info['causal_event']})] "
                f"PRE_LLM_DIRECT ({pre_trial.get('selected_binding_id') or pre_trial.get('selected_source_short_name')}) "
                f"| pass={summary['strict_pass']} | turns=0 | probe={pre_trial.get('probe_wall_ms', 0):.0f}ms",
                flush=True,
            )
            continue

        # Fallback to Live `gpt-6.2-sol` Episode (`18` turns)
        diag_for_prompt = {
            "arm": arm,
            "pre_llm_reason": pre_trial.get("reason"),
            "parameter_payload": pre_trial.get("parameter_payload"),
        }
        aug_task, ctx_hdr = _inject_equal_info_archive_into_task(task, prior_entries, prior_cal_sweeps, diag_for_prompt)
        ft = m97_task_to_frontier_task(
            aug_task,
            continual_context_block=ctx_hdr,
            allowed_readonly_dependencies=tuple(sorted(readonly_deps.keys())),
        )
        ep_run_tag = f"{run_id}__{wid}__{arm}__{task.short_name}"

        _CALL_CONTEXT.role = "stream_solve"
        _CALL_CONTEXT.world_id = wid
        _CALL_CONTEXT.cluster_id = spec.cluster_id
        _CALL_CONTEXT.arm = arm
        _CALL_CONTEXT.episode_short_name = task.short_name
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
            enable_auto_skill_packaging=(arm == "Hegel_DemandGated_Graph"),
            allow_l1_primitives=(arm == "Hegel_DemandGated_Graph"),
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
        funnel_rec = m64.load_and_classify_episode_source_aware_funnel(
            episode_dir=ep_dir,
            episode_summary=ep_sum,
            new_source_l2_stems=set(),
            new_source_l1_stems=set(),
            prior_l1_stems=prior_v7_l1_stems,
            arm_has_new_source_assets=False,
        )

        ind_receipt: dict[str, Any] | None = None
        cov_receipts_this_step: list[dict[str, Any]] = []
        cons_gate: dict[str, Any] | None = None

        if arm == "Hegel_DemandGated_Graph" and pub_gate_post.get("accepted") and cand_txt and graph is not None and registry is not None and lineage_ctrl is not None:
            # 1. Demand-Gated Consolidation Check (`N08`: turns off negative-EV eager post-episode L2 compile!)
            cons_gate = _evaluate_demand_gated_l2_consolidation_policy(
                remaining_steps=(n_steps - step_i),
                canonical_binding_covers_contract=True,
            )
            consolidation_gate_decisions.append({"step": step_i, "short_name": task.short_name, **cons_gate})

            # 2. `EmpiricalCoverageReceipt` check (`§5`, `0` LLM calls, explicit CPU/wall-ms accounting!)
            failed_bid = str(pre_trial.get("selected_binding_id") or f"bind_{wid}_b00")
            existing_bs = sorted(
                [b for b in registry.bindings.values() if b.chain_id == fam and b.qualification_status == "verified"],
                key=lambda x: x.binding_id,
            )
            next_b_idx = lineage_ctrl._branch_seq
            placeholder_new_bid = f"bind_{wid}_b{next_b_idx:02d}"

            all_covered = True
            for old_b in existing_bs:
                rep_wits = rep_positive_witness_by_bid.get(old_b.binding_id, [])[:1]
                cov_r = evaluate_empirical_coverage_on_branch_witnesses(
                    family=fam,
                    new_binding_id=placeholder_new_bid,
                    new_canonical_code=cand_txt,
                    new_entrypoint=task.declared_callable_entrypoints[0],
                    old_binding=old_b,
                    old_positive_tasks_and_payloads=rep_wits,
                )
                lineage_ctrl.record_coverage_receipt(cov_r)
                cov_receipts_this_step.append(cov_r.to_dict())
                coverage_receipts_log.append(cov_r.to_dict())
                if cov_r.status != "supported_on_witnesses":
                    all_covered = False

            cur_pay = assemble_m97_parameter_payload(spec=spec, graph=graph, parent_ids=["cal_node_p0", "cal_node_p1"])
            cur_obs = compute_m97_primitive_observables(fam, pub_item, cur_pay)
            ev_sha = hashlib.sha256(json.dumps(pub_gate_post.get("scientific_probe_receipt") or {}, sort_keys=True).encode("utf-8")).hexdigest()

            if all_covered and step_i < n_steps:
                # If `cand_txt` genuinely empirically covered ALL prior branches (`piecewise/unified solver`!),
                # register `cand_txt` as higher-priority covering branch WITHOUT deleting older branches (`0` guard LLM calls!)
                new_b = lineage_ctrl.mint_immutable_conflict_branch(
                    failed_binding_id=failed_bid,
                    new_canonical_code=cand_txt,
                    new_entrypoint=task.declared_callable_entrypoints[0],
                    forest_node_id=source_forest_nid,
                    qualification_protocol="hegel_public_scientific_gate_v4",
                    output_schema_keys=uni_keys,
                    validated_parent_ids=["cal_node_p0", "cal_node_p1"],
                    new_applicability_guard={},
                    updated_failed_branch_applicability_guard=None,
                    violated_component_id="empirically_covered_superset",
                    preserved_component_ids=[],
                    selection_priority=50 + step_i,
                    environment_digest=env_image_digest,
                    evidence_digest=ev_sha,
                )
                lineage_ctrl.record_tested_witness(
                    binding=new_b,
                    public_input_sha256=pub_sha,
                    passed=True,
                    failure_kind="none",
                    features=cur_obs,
                )
                rep_positive_witness_by_bid[new_b.binding_id] = [(f"wit_{task.short_name}", task, cur_pay, readonly_deps, readonly_shas)]
            elif step_i < n_steps:
                # Non-covering conflict (`cov_r.status == 'refuted'`): trigger multi-branch guard & containerized parent sensitivity induction!
                pos_by_br: dict[str, list[dict[str, float | None]]] = {
                    b.binding_id: [w.public_structural_features for w in lineage_ctrl.positive_witnesses_for(b.binding_id)]
                    for b in existing_bs
                }
                pos_by_br[placeholder_new_bid] = [cur_obs]

                _CALL_CONTEXT.role = "autonomous_guard_induction"
                _CALL_CONTEXT.world_id = wid
                _CALL_CONTEXT.cluster_id = spec.cluster_id
                _CALL_CONTEXT.arm = arm
                _CALL_CONTEXT.episode_short_name = task.short_name
                comp_client._hegel_call_meta = {  # type: ignore[attr-defined]
                    "role": "autonomous_guard_induction",
                    "world_id": wid,
                    "cluster_id": spec.cluster_id,
                    "arm": arm,
                    "episode_short_name": task.short_name,
                }
                cov_pairs_ok = {
                    (cr.new_binding_id, cr.old_binding_id)
                    for cr in lineage_ctrl.coverage_receipts
                    if cr.status == "supported_on_witnesses"
                }
                ind_res = propose_and_validate_m97_multibranch_split(
                    client=comp_client,
                    stage_subdir=f"{ep_run_tag}__ind",
                    spec=spec,
                    task_instruction_text=task.instruction_text,
                    existing_bindings=existing_bs,
                    new_branch_id_placeholder=placeholder_new_bid,
                    new_code=cand_txt,
                    new_entrypoint=task.declared_callable_entrypoints[0],
                    positive_witness_features_by_branch=pos_by_br,
                    current_public_item=pub_item,
                    current_param_payload=cur_pay,
                    parent_wiring_meta=parent_wiring_meta,
                    covered_branch_pairs=cov_pairs_ok,
                )
                ind_receipt = {k: v for k, v in ind_res.items() if k != "provider_call_receipts"}
                lineage_induction_events.append(
                    {"step": step_i, "causal_event": m_info["causal_event"], "short_name": task.short_name, **ind_receipt}
                )
                if ind_res.get("accepted"):
                    val_pids = sorted(ind_res["validated_parent_ids"])
                    new_b = lineage_ctrl.mint_immutable_conflict_branch(
                        failed_binding_id=failed_bid,
                        new_canonical_code=cand_txt,
                        new_entrypoint=task.declared_callable_entrypoints[0],
                        forest_node_id=source_forest_nid,
                        qualification_protocol="hegel_public_scientific_gate_v4",
                        output_schema_keys=uni_keys,
                        validated_parent_ids=val_pids,
                        new_applicability_guard=dict(ind_res["new_branch_guard"]),
                        updated_failed_branch_applicability_guard=None,
                        violated_component_id=str(ind_res["violated_component_id"]),
                        preserved_component_ids=list(ind_res["preserved_component_ids"]),
                        selection_priority=10 + step_i,
                        environment_digest=env_image_digest,
                        evidence_digest=ev_sha,
                    )
                    # Update applicability guards of all existing branches non-destructively (`0` code overwrite!)
                    for ex_bid, ex_g in (ind_res.get("existing_branch_guards") or {}).items():
                        if ex_bid in registry.bindings:
                            registry.bindings[ex_bid].applicability_guard = deepcopy(ex_g)
                            registry.bindings[ex_bid].precondition_guard = deepcopy(ex_g)
                    registry.sync_to_graph()
                    lineage_ctrl.record_tested_witness(
                        binding=new_b,
                        public_input_sha256=pub_sha,
                        passed=True,
                        failure_kind="none",
                        features=cur_obs,
                    )
                    rep_positive_witness_by_bid[new_b.binding_id] = [
                        (f"wit_{task.short_name}", task, cur_pay, readonly_deps, readonly_shas)
                    ]

        ep_sum["world_id"] = wid
        ep_sum["cluster_id"] = spec.cluster_id
        ep_sum["family"] = fam
        ep_sum["arm"] = arm
        ep_sum["stream_step"] = step_i
        ep_sum["causal_event"] = m_info["causal_event"]
        ep_sum["true_regime"] = m_info["true_regime"]
        ep_sum["event_category"] = m_info["event_category"]
        ep_sum["short_name"] = task.short_name
        ep_sum["execution_mode"] = "llm_synthesis_after_equal_tool_gate"
        ep_sum["strict_pass"] = bool(priv_ver.get("strict_pass"))
        ep_sum["reward"] = 1.0 if priv_ver.get("strict_pass") else 0.0
        ep_sum["pre_llm_probe_wall_ms"] = float(pre_trial.get("probe_wall_ms", 0.0))
        ep_sum["readonly_dependencies_sha256"] = readonly_shas
        ep_sum["public_qualification_status"] = "public_gate_verified" if pub_gate_post.get("accepted") else "public_gate_failed"
        ep_sum["m8_permuted_parent_update"] = m8_update_rec
        ep_sum["demand_consolidation_gate"] = cons_gate
        ep_sum["empirical_coverage_receipts"] = cov_receipts_this_step
        ep_sum["multibranch_split_induction"] = ind_receipt
        ep_sum["source_aware_adoption_funnel"] = funnel_rec
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
                    output_schema_keys=uni_keys,
                    public_qualification_status=("public_gate_verified" if pub_gate_post.get("accepted") else "public_gate_failed"),
                )
            )
        prior_cal_sweeps.append(_summarize_public_calibration_sweep(fam, pub_item, task.short_name))
        if graph is not None:
            graph.save_checkpoint(stream_out_dir / f"graph_ckpt_after_step_{step_i:02d}.json")
        episode_results.append(ep_sum)
        print(
            f"  [M9.7 {wid} | {arm} | step {step_i}/{n_steps} {task.short_name} ({m_info['causal_event']})] "
            f"LLM_FALLBACK | pass={ep_sum['strict_pass']} | turns={ep_sum.get('turns_used')} "
            f"| ind_accepted={ind_receipt.get('accepted') if ind_receipt else None}",
            flush=True,
        )

    return {
        "world_id": wid,
        "cluster_id": spec.cluster_id,
        "family": fam,
        "arm": arm,
        "wall_sec": round(monotonic() - stream_t0, 2),
        "strict_passes": sum(1 for r in episode_results if r.get("strict_pass")),
        "episode_count": len(episode_results),
        "zero_llm_episodes_total": sum(1 for r in episode_results if int(r.get("turns_used", 0) or 0) == 0),
        "zero_turn_passed_closures": sum(1 for r in episode_results if int(r.get("turns_used", 0) or 0) == 0 and r.get("strict_pass")),
        "total_turns_used": sum(int(r.get("turns_used", 0) or 0) for r in episode_results),
        "total_pre_llm_probe_wall_ms": round(sum(float(r.get("pre_llm_probe_wall_ms", 0.0) or 0.0) for r in episode_results), 2),
        "lineage_induction_events": lineage_induction_events,
        "empirical_coverage_receipts": coverage_receipts_log,
        "consolidation_gate_decisions": consolidation_gate_decisions,
        "witness_ledger": ([w.to_dict() for w in lineage_ctrl.witness_ledger] if lineage_ctrl is not None else []),
        "final_bindings": (
            {k: {bk: bv for bk, bv in b.to_dict().items() if "code" not in bk} for k, b in registry.bindings.items()}
            if registry is not None
            else {}
        ),
        "episodes_meta": [
            {
                "stream_step": r["stream_step"],
                "causal_event": r["causal_event"],
                "true_regime": r["true_regime"],
                "event_category": r["event_category"],
                "task_id": r["task_id"],
                "short_name": r["short_name"],
                "execution_mode": r["execution_mode"],
                "strict_pass": r["strict_pass"],
                "turns_used": int(r.get("turns_used", 0) or 0),
                "pre_llm_probe_wall_ms": float(r.get("pre_llm_probe_wall_ms", 0.0) or 0.0),
                "selected_binding_id": (r.get("pre_llm_trial_receipt") or {}).get("selected_binding_id"),
                "selected_source_short_name": (r.get("pre_llm_trial_receipt") or {}).get("selected_source_short_name"),
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
    ap.add_argument("--num-worlds", type=int, default=6, choices=[2, 4, 6])
    ap.add_argument("--max-turns", type=int, default=18)
    ap.add_argument("--max-stream-workers", type=int, default=32)
    ap.add_argument("--max-codex-concurrency", type=int, default=32)
    args = ap.parse_args()

    os.environ["HEGEL_MAX_CODEX_CONCURRENCY"] = str(args.max_codex_concurrency)
    out_dir: Path = args.out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory `{out_dir}`!")

    commit_start, dirty_start, src_sha_start = _git_and_source_tree_state()
    if dirty_start:
        raise RuntimeError("Refusing to start M9.7 confirmation experiment on a dirty working tree!")

    out_dir.mkdir(parents=True, exist_ok=True)
    mirror_dir = out_dir / "provider_raw_attempts_mirror"
    mirror_dir.mkdir(parents=True, exist_ok=True)

    ensure_clean_scipy_sandbox_image()
    m64.ensure_runtime_qualification_context(sandbox_cid=None)
    env_image_digest = _docker_image_digest(m7a.CLEAN_SCIPY_SANDBOX_IMAGE)

    codex_info = m6f.discover_chatgpt_subscription_codex_cli()
    if not (codex_info.get("available") and codex_info.get("auth_mode") == "chatgpt"):
        raise RuntimeError(f"Official ChatGPT Codex CLI unavailable: {codex_info}")

    half_n = args.num_worlds // 2
    selected_worlds: list[M97WorldSpec] = (
        list(CONFIRMATION_M97_WORLD_SPECS[:half_n]) + list(CONFIRMATION_M97_WORLD_SPECS[3 : 3 + half_n])
    )
    print(
        f"[M9.7 INIT] Commit={commit_start[:12]} | SrcSHA={src_sha_start[:12]} | IndependentClusters={len(selected_worlds)} "
        f"| Streams={len(selected_worlds)*len(M97_ARMS)} | CodexConcurrency={args.max_codex_concurrency}",
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
        cid_tag = str(cli_meta.get("cluster_id") or getattr(_CALL_CONTEXT, "cluster_id", ""))
        arm_tag = str(cli_meta.get("arm") or getattr(_CALL_CONTEXT, "arm", ""))
        ep_sn_tag = str(cli_meta.get("episode_short_name") or getattr(_CALL_CONTEXT, "episode_short_name", ""))
        mono_t0 = monotonic()
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
        mono_t1 = monotonic()
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
                    "cluster_id": cid_tag,
                    "arm": arm_tag,
                    "episode_short_name": ep_sn_tag,
                    "stage_subdir": str(kw.get("stage_subdir") or ""),
                    "call_uuid": str(res.get("call_uuid") or ""),
                    "attempts_ledger_filename": str(res.get("attempts_ledger_filename") or ""),
                    "mirrored_raw_filename": f"{res.get('call_uuid')}__{res.get('attempts_ledger_filename')}",
                    "monotonic_start_sec": round(mono_t0, 4),
                    "monotonic_end_sec": round(mono_t1, 4),
                    "ok": bool(res.get("ok")),
                    "attempt_count": int(res.get("attempt_count", 1) or 1),
                    "usage_unknown_attempt_count": int(res.get("usage_unknown_attempt_count", 0) or 0),
                    "cost_accounting_status": str(res.get("cost_accounting_status") or "complete"),
                    **sc,
                }
            )
        return res

    m6f.FrontierCodexClient.call_structured = _throttled_and_mirrored_call  # type: ignore[method-assign]
    run_id = "m9_7_lineage_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    t0_all = monotonic()

    base_v7_forest, base_v7_archive, _, _ = m7a.load_autogenous_forest_bundle(args.v7_bundle)
    prior_v7_l1_stems = {
        Path(str(m.get("filename") or "")).stem
        for _, rec in m7a.iter_live_archive_nodes(base_v7_archive)
        for m in (rec.get("autogenous_wormhole_modules") or [])
        if str(m.get("filename") or "").startswith("autogenous_prim_")
    }
    built_worlds = {w.world_id: build_m97_world_tasks(w) for w in selected_worlds}

    # =========================================================================
    # PHASE 1: Independent Cluster `Stage S` Sources + Initial `b00` Freeze (`0` Cross-Cluster Coupling!)
    # =========================================================================
    src_out_root = out_dir / "stage_S_sources"

    def _run_and_freeze_cluster_source(w_spec: M97WorldSpec) -> dict[str, Any]:
        s_task, _, _ = built_worlds[w_spec.world_id]
        for att_i in (1, 2):
            s_dir = src_out_root / w_spec.world_id / f"attempt_{att_i:02d}" / s_task.task_id
            ft_s = m97_task_to_frontier_task(s_task)
            ep_a = deepcopy(base_v7_archive)
            ep_a["__autogenous_blank_seed__"] = {"enabled": True}
            _CALL_CONTEXT.role = "stage_S_solve"
            _CALL_CONTEXT.world_id = w_spec.world_id
            _CALL_CONTEXT.cluster_id = w_spec.cluster_id
            _CALL_CONTEXT.arm = "Stage_S_Shared_Source"
            _CALL_CONTEXT.episode_short_name = s_task.short_name
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
            cand_s_txt = next(iter(sub_s.values()), "")
            cand_s_sha = hashlib.sha256(cand_s_txt.encode("utf-8")).hexdigest() if cand_s_txt else ""
            pub_gate_s = execute_single_pre_llm_public_gate_probe_in_container(task=s_task, candidate_code=cand_s_txt) if cand_s_txt else {"accepted": False}
            ver_s = grade_m94_artifact_map(s_task, sub_s) if sub_s else {"strict_pass": False, "error": "missing_submitted_artifact"}
            dag_s = m6f.append_post_episode_verification_to_dag(
                s_dir,
                run_id=run_id,
                task_id=s_task.task_id,
                arm="Stage_S_Shared_Source",
                model=args.model,
                candidate_sha256=cand_s_sha,
                public_gate_receipt=pub_gate_s.get("scientific_probe_receipt") or {"ended_by_finish": bool(ep_s.get("ended_by_finish"))},
                verifier_report=ver_s,
            )
            ep_s["strict_pass"] = bool(ver_s.get("strict_pass"))
            ep_s["provenance_dag_verification"] = dag_s
            ep_s["verifier_report"] = ver_s
            (s_dir / "00_episode_summary.json").write_text(json.dumps(ep_s, indent=2) + "\n", encoding="utf-8")
            (s_dir / "03_official_verifier_report.json").write_text(json.dumps(ver_s, indent=2) + "\n", encoding="utf-8")
            print(
                f"  [M9.7 STAGE S] {w_spec.world_id} ({w_spec.cluster_id}, att={att_i}) | pass={ep_s['strict_pass']} | turns={ep_s['turns_used']}",
                flush=True,
            )
            if ep_s["strict_pass"] and pub_gate_s.get("accepted"):
                src_nid = next(iter(base_v7_forest.nodes.keys()))
                init_g, _, init_reg, wiring_meta = initialize_m97_permuted_parent_graph(
                    spec=w_spec, forest=deepcopy(base_v7_forest)
                )
                uni_keys = expected_m97_public_output_keys(w_spec.family)
                ev_s_sha = hashlib.sha256(
                    json.dumps(pub_gate_s.get("scientific_probe_receipt") or {}, sort_keys=True).encode("utf-8")
                ).hexdigest()
                q_rec_s = init_reg.issue_qualification_receipt(
                    qualification_protocol="hegel_public_scientific_gate_v4",
                    verified_code_sha256=cand_s_sha,
                    forest_node_id=src_nid,
                    contract_schema_sha256=hashlib.sha256(json.dumps(sorted(uni_keys)).encode("utf-8")).hexdigest(),
                    environment_digest=env_image_digest,
                    evidence_digest=ev_s_sha,
                    issued_reason=f"stage_S_verified_{w_spec.world_id}",
                )
                init_reg.register_binding(
                    MechanismExecutableBinding(
                        binding_id=f"bind_{w_spec.world_id}_b00",
                        chain_id=w_spec.family,
                        branch_tag="C0_base",
                        mechanism_id=f"mech_{w_spec.world_id}_b00",
                        mechanism_version=1,
                        forest_node_id=src_nid,
                        component_id="base_constitutive_mechanism",
                        preserved_component_ids=[],
                        violated_component_id=None,
                        module_stem=s_task.short_name,
                        module_digest=cand_s_sha,
                        function_name=s_task.declared_callable_entrypoints[0],
                        qualification_id="hegel_public_scientific_gate_v4",
                        qualification_protocol="hegel_public_scientific_gate_v4",
                        qualification_receipt_id=q_rec_s.receipt_id,
                        qualification_status="verified",
                        parameter_mechanism_id="cal_node_p0",
                        parent_parameter_mechanism_ids=["cal_node_p0", "cal_node_p1"],
                        parent_parameter_versions={"cal_node_p0": 1, "cal_node_p1": 1},
                        parameter_version=1,
                        structure_origin="unconstrained_initial_source",
                        environment_digest=env_image_digest,
                        evidence_receipt_id=ev_s_sha,
                        precondition_guard={},
                        applicability_guard={},
                        selection_priority=5,
                        output_schema_keys=uni_keys,
                        canonical_algorithm_code=cand_s_txt,
                        canonical_entrypoint=s_task.declared_callable_entrypoints[0],
                        canonical_algorithm_sha256=cand_s_sha,
                        input_coordinate_contract="raw_measured",
                        parameterized_closure_code=cand_s_txt,
                        parameterized_entrypoint=s_task.declared_callable_entrypoints[0],
                        parameterized_closure_sha256=cand_s_sha,
                    ),
                    reason=f"stage_S_unconstrained_freeze_{w_spec.world_id}",
                    strict_receipt_check=True,
                )
                w_freeze_dir = out_dir / "world_freezes" / w_spec.world_id
                w_freeze_dir.mkdir(parents=True, exist_ok=True)
                ckpt_p = w_freeze_dir / "initial_permuted_parent_graph_ckpt.json"
                init_g.save_checkpoint(ckpt_p)
                pub_s_item, pub_s_sha = _materialize_public_item_and_sha(s_task)
                nom_pay = {"stress_gains": [1.0] * 4, "stress_biases": [0.0] * 4} if w_spec.family == "alpha" else {"bank_gains": [1.0] * 4, "bank_offsets": [0.0] * 4}
                return {
                    "world_id": w_spec.world_id,
                    "s_task": s_task,
                    "graph_ckpt_path": ckpt_p,
                    "parent_wiring_meta": wiring_meta,
                    "source_nid": src_nid,
                    "source_pub_sha256": pub_s_sha,
                    "source_observables": compute_m97_primitive_observables(w_spec.family, pub_s_item, nom_pay),
                    "source_archive_entry": make_continual_archive_entry(
                        short_name=s_task.short_name,
                        task_id=s_task.task_id,
                        canonical_entrypoint=s_task.declared_callable_entrypoints[0],
                        canonical_algorithm_code=cand_s_txt,
                        output_schema_keys=uni_keys,
                        public_qualification_status="public_gate_verified",
                    ),
                    "source_cal_summary": _summarize_public_calibration_sweep(w_spec.family, pub_s_item, s_task.short_name),
                }
        raise RuntimeError(f"Stage S source for `{w_spec.world_id}` failed verification across 2 attempts!")

    world_init_by_id: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=len(selected_worlds)) as pool_s:
        futs_s = [pool_s.submit(_run_and_freeze_cluster_source, w) for w in selected_worlds]
        for fd in as_completed(futs_s):
            rw = fd.result()
            world_init_by_id[rw["world_id"]] = rw

    # =========================================================================
    # PHASE 2: `12` Parallel Independent Continual Streams (`6` Clusters x `2` Arms)
    # =========================================================================
    stream_jobs = [(w, arm_nm) for w in selected_worlds for arm_nm in M97_ARMS]
    stream_summaries: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.max_stream_workers))) as pool_st:
        futs_st = []
        for w_spec, arm_nm in stream_jobs:
            w_init = world_init_by_id[w_spec.world_id]
            s_tk, s_tasks, s_man = built_worlds[w_spec.world_id]
            futs_st.append(
                pool_st.submit(
                    _run_one_m97_world_arm_stream,
                    spec=w_spec,
                    arm=arm_nm,
                    s_task=s_tk,
                    stream_tasks=s_tasks,
                    schedule_manifest=s_man,
                    base_v7_forest=base_v7_forest,
                    base_v7_archive=base_v7_archive,
                    prior_v7_l1_stems=prior_v7_l1_stems,
                    base_graph_ckpt_path=w_init["graph_ckpt_path"],
                    parent_wiring_meta=w_init["parent_wiring_meta"],
                    source_archive_entry=w_init["source_archive_entry"],
                    source_cal_summary=w_init["source_cal_summary"],
                    source_forest_nid=w_init["source_nid"],
                    source_pub_sha256=w_init["source_pub_sha256"],
                    source_observables=w_init["source_observables"],
                    env_image_digest=env_image_digest,
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

    stream_summaries.sort(key=lambda r: (r["world_id"], M97_ARMS.index(r["arm"])))

    # =========================================================================
    # PHASE 3: Strict Call-UUID CostPolicy Conservation (`N01`) & Independent Cluster Statistics (`n = 6`)
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

    world_ids_ord = [w.world_id for w in selected_worlds]
    by_world_arm = {(s["world_id"], s["arm"]): s for s in stream_summaries}
    in_stream_roles = {"stream_solve", "stream_compile", "autonomous_guard_induction"}

    stage_s_solve_agg = aggregate_call_usages(
        [r for r in master_provider_receipts if r["call_role"] == "stage_S_solve"], deduplicate_uuid=True
    )

    arm_agg: dict[str, Any] = {}
    cluster_ledgers_by_arm: dict[str, dict[str, dict[str, Any]]] = {a: {} for a in M97_ARMS}

    for arm_nm in M97_ARMS:
        a_streams = [by_world_arm[(wid, arm_nm)] for wid in world_ids_ord]
        a_eps = [ep for s in a_streams for ep in s["episodes_meta"]]
        arm_calls_solve = [r for r in master_provider_receipts if r["arm"] == arm_nm and r["call_role"] == "stream_solve"]
        arm_calls_comp = [r for r in master_provider_receipts if r["arm"] == arm_nm and r["call_role"] == "stream_compile"]
        arm_calls_guard = [r for r in master_provider_receipts if r["arm"] == arm_nm and r["call_role"] == "autonomous_guard_induction"]
        arm_in_stream_calls = arm_calls_solve + arm_calls_comp + arm_calls_guard
        in_stream_ref = aggregate_call_usages(arm_in_stream_calls, deduplicate_uuid=True)

        # Partition strictly by `episode_short_name`, `causal_event`, and `cluster_id` (`== world_id`) — ZERO `token_cost_proxy`!
        ep_ledgers: dict[str, dict[str, Any]] = {}
        for ep in a_eps:
            sn = ep["short_name"]
            ep_calls = [r for r in arm_in_stream_calls if r["episode_short_name"] == sn]
            ep_led = aggregate_call_usages(ep_calls, deduplicate_uuid=True)
            ep["cost_ledger"] = ep_led
            ep_ledgers[sn] = ep_led

        event_ledgers: dict[str, dict[str, Any]] = {}
        by_event_summary: dict[str, Any] = {}
        for ev_k in ("C1", "R0", "C2", "R1", "U_iso", "G_perm", "M_couple"):
            ev_sns = {e["short_name"] for e in a_eps if e["causal_event"] == ev_k}
            ev_led = aggregate_call_usages([r for r in arm_in_stream_calls if r["episode_short_name"] in ev_sns], deduplicate_uuid=True)
            event_ledgers[ev_k] = ev_led
            ev_eps = [e for e in a_eps if e["causal_event"] == ev_k]
            by_event_summary[ev_k] = {
                "strict_passes": sum(1 for e in ev_eps if e["strict_pass"]),
                "episode_count": len(ev_eps),
                "zero_llm_episodes": sum(1 for e in ev_eps if e["turns_used"] == 0),
                "zero_turn_passed_closures": sum(1 for e in ev_eps if e["turns_used"] == 0 and e["strict_pass"]),
                "mean_turns": round(float(np.mean([e["turns_used"] for e in ev_eps])), 2) if ev_eps else 0.0,
                "cost_ledger": ev_led,
            }

        for s in a_streams:
            wid = s["world_id"]
            c_led = aggregate_call_usages([r for r in arm_in_stream_calls if r["world_id"] == wid], deduplicate_uuid=True)
            s["cost_ledger"] = c_led
            cluster_ledgers_by_arm[arm_nm][wid] = c_led

        # `N01` Fail-Closed Conservation Assertion (`sum(episodes) == sum(events) == sum(clusters) == arm_in_stream_ref`)!
        cons_chk = assert_hierarchical_ledger_conservation(
            reference_arm_ledger=in_stream_ref,
            named_partition_ledgers={
                "episodes": ep_ledgers.values(),
                "events": event_ledgers.values(),
                "independent_clusters": cluster_ledgers_by_arm[arm_nm].values(),
            },
            context_label=arm_nm,
        )

        standalone_agg = aggregate_call_usages(
            [r for r in master_provider_receipts if r["call_role"] == "stage_S_solve"] + arm_in_stream_calls,
            deduplicate_uuid=True,
        )
        arm_agg[arm_nm] = {
            "independent_cluster_count": len(a_streams),
            "episode_count": len(a_eps),
            "strict_passes": sum(1 for e in a_eps if e["strict_pass"]),
            "pass_rate": round(sum(1 for e in a_eps if e["strict_pass"]) / max(len(a_eps), 1), 4),
            "zero_llm_episodes_total": sum(1 for e in a_eps if e["turns_used"] == 0),
            "zero_turn_passed_closures": sum(1 for e in a_eps if e["turns_used"] == 0 and e["strict_pass"]),
            "total_turns_used": sum(int(e["turns_used"]) for e in a_eps),
            "mean_turns_per_episode": round(float(np.mean([e["turns_used"] for e in a_eps])), 2),
            "total_pre_llm_probe_wall_ms": round(sum(float(e["pre_llm_probe_wall_ms"]) for e in a_eps), 2),
            "scope_1_in_stream_solve_only": aggregate_call_usages(arm_calls_solve, deduplicate_uuid=True),
            "scope_1_in_stream_compile_only": aggregate_call_usages(arm_calls_comp, deduplicate_uuid=True),
            "scope_1_in_stream_guard_induction_only": aggregate_call_usages(arm_calls_guard, deduplicate_uuid=True),
            "scope_1_in_stream_total": in_stream_ref,
            "scope_2_standalone_with_stage_S": standalone_agg,
            "ledger_conservation_verification": cons_chk,
            "by_causal_event": by_event_summary,
        }

    # Primary Cluster-Independent Paired Inference (`n = 6` Independent Clusters: `Hegel - Flat`)
    v1_diffs, v2_diffs, turn_diffs, pass_diffs, zt_diffs = [], [], [], [], []
    per_cluster_rows = []
    for wid in world_ids_ord:
        st_h = by_world_arm[(wid, "Hegel_DemandGated_Graph")]
        st_f = by_world_arm[(wid, "Flat_EqualInfo_Multiversion")]
        lh = cluster_ledgers_by_arm["Hegel_DemandGated_Graph"][wid]
        lf = cluster_ledgers_by_arm["Flat_EqualInfo_Multiversion"][wid]
        dv1 = int(lh["proxy_v1_u_025c_1o"]) - int(lf["proxy_v1_u_025c_1o"])
        dv2 = int(lh["proxy_v2_u_010c_4o"]) - int(lf["proxy_v2_u_010c_4o"])
        dt = int(st_h["total_turns_used"]) - int(st_f["total_turns_used"])
        dp = int(st_h["strict_passes"]) - int(st_f["strict_passes"])
        dzt = int(st_h["zero_turn_passed_closures"]) - int(st_f["zero_turn_passed_closures"])
        v1_diffs.append(float(dv1)); v2_diffs.append(float(dv2)); turn_diffs.append(float(dt)); pass_diffs.append(float(dp)); zt_diffs.append(float(dzt))
        per_cluster_rows.append(
            {
                "world_id": wid,
                "cluster_id": st_h["cluster_id"],
                "family": st_h["family"],
                "in_stream_v1_diff": dv1,
                "in_stream_v2_diff": dv2,
                "turns_diff": dt,
                "strict_passes_diff": dp,
                "zero_turn_passed_closures_diff": dzt,
            }
        )

    tot_h_v1 = arm_agg["Hegel_DemandGated_Graph"]["scope_1_in_stream_total"]["proxy_v1_u_025c_1o"]
    tot_f_v1 = arm_agg["Flat_EqualInfo_Multiversion"]["scope_1_in_stream_total"]["proxy_v1_u_025c_1o"]
    tot_h_v2 = arm_agg["Hegel_DemandGated_Graph"]["scope_1_in_stream_total"]["proxy_v2_u_010c_4o"]
    tot_f_v2 = arm_agg["Flat_EqualInfo_Multiversion"]["scope_1_in_stream_total"]["proxy_v2_u_010c_4o"]
    tot_h_s2_v2 = arm_agg["Hegel_DemandGated_Graph"]["scope_2_standalone_with_stage_S"]["proxy_v2_u_010c_4o"]
    tot_f_s2_v2 = arm_agg["Flat_EqualInfo_Multiversion"]["scope_2_standalone_with_stage_S"]["proxy_v2_u_010c_4o"]

    paired_stats = {
        "treatment": "Hegel_DemandGated_Graph",
        "baseline": "Flat_EqualInfo_Multiversion",
        "primary_inferential_unit": "independent_source_world_cluster",
        "n_independent_clusters": len(world_ids_ord),
        "clusters_with_lower_in_stream_v1": sum(1 for d in v1_diffs if d < 0),
        "clusters_with_lower_in_stream_v2": sum(1 for d in v2_diffs if d < 0),
        "in_stream_v1_total_delta": tot_h_v1 - tot_f_v1,
        "in_stream_v1_relative_pct": round((tot_h_v1 - tot_f_v1) / max(tot_f_v1, 1) * 100.0, 2),
        "in_stream_v1_two_sided_sign_flip_p": _exact_two_sided_sign_flip_p_value(v1_diffs),
        "in_stream_v2_total_delta": tot_h_v2 - tot_f_v2,
        "in_stream_v2_relative_pct": round((tot_h_v2 - tot_f_v2) / max(tot_f_v2, 1) * 100.0, 2),
        "in_stream_v2_cluster_bootstrap_95ci": _bootstrap_mean_ci(v2_diffs),
        "in_stream_v2_two_sided_sign_flip_p": _exact_two_sided_sign_flip_p_value(v2_diffs),
        "standalone_with_stage_S_v2_relative_pct": round((tot_h_s2_v2 - tot_f_s2_v2) / max(tot_f_s2_v2, 1) * 100.0, 2),
        "turns_mean_cluster_delta": round(float(np.mean(turn_diffs)), 2),
        "turns_two_sided_sign_flip_p": _exact_two_sided_sign_flip_p_value(turn_diffs),
        "zero_turn_passed_closures_total_delta": int(round(sum(zt_diffs))),
        "zero_turn_passed_closures_two_sided_sign_flip_p": _exact_two_sided_sign_flip_p_value(zt_diffs),
        "strict_passes_total_delta": int(round(sum(pass_diffs))),
        "strict_passes_two_sided_sign_flip_p": _exact_two_sided_sign_flip_p_value(pass_diffs),
        "per_independent_cluster_deltas": per_cluster_rows,
    }

    hegel_streams = [by_world_arm[(w, "Hegel_DemandGated_Graph")] for w in world_ids_ord]
    flat_streams = [by_world_arm[(w, "Flat_EqualInfo_Multiversion")] for w in world_ids_ord]
    all_ind = [ev for s in hegel_streams for ev in s["lineage_induction_events"]]
    all_cov = [cr for s in hegel_streams for cr in s["empirical_coverage_receipts"]]

    lineage_funnel = {
        "total_empirical_coverage_checks_run": len(all_cov),
        "empirical_coverage_refuted_count": sum(1 for c in all_cov if c["status"] == "refuted"),
        "empirical_coverage_supported_count": sum(1 for c in all_cov if c["status"] == "supported_on_witnesses"),
        "total_multibranch_inductions_triggered": len(all_ind),
        "accepted_multibranch_inductions": sum(1 for e in all_ind if e.get("accepted")),
        "synchronous_in_stream_l2_compilations_executed": sum(
            1 for s in hegel_streams for g in s["consolidation_gate_decisions"] if g["should_compile_synchronously"]
        ),
        "branches_retained_at_end_by_world": {s["world_id"]: sorted(s["final_bindings"].keys()) for s in hegel_streams},
        "r0_zero_turn_passed_Hegel_vs_Flat": [
            sum(1 for s in hegel_streams for e in s["episodes_meta"] if e["causal_event"] == "R0" and e["turns_used"] == 0 and e["strict_pass"]),
            sum(1 for s in flat_streams for e in s["episodes_meta"] if e["causal_event"] == "R0" and e["turns_used"] == 0 and e["strict_pass"]),
        ],
        "r1_post_c2_restoration_zero_turn_passed_Hegel_vs_Flat": [
            sum(1 for s in hegel_streams for e in s["episodes_meta"] if e["causal_event"] == "R1" and e["turns_used"] == 0 and e["strict_pass"]),
            sum(1 for s in flat_streams for e in s["episodes_meta"] if e["causal_event"] == "R1" and e["turns_used"] == 0 and e["strict_pass"]),
        ],
        "u_iso_strict_counterfactual_zero_turn_passed_Hegel_vs_Flat": [
            sum(1 for s in hegel_streams for e in s["episodes_meta"] if e["causal_event"] == "U_iso" and e["turns_used"] == 0 and e["strict_pass"]),
            sum(1 for s in flat_streams for e in s["episodes_meta"] if e["causal_event"] == "U_iso" and e["turns_used"] == 0 and e["strict_pass"]),
        ],
        "induced_guards_and_parents_by_world": {
            s["world_id"]: [
                {
                    "step": ev["step"],
                    "event": ev["causal_event"],
                    "accepted": ev["accepted"],
                    "validated_parents": ev.get("validated_parent_ids"),
                    "new_branch_guard": ev.get("new_branch_guard"),
                    "existing_branch_guards": ev.get("existing_branch_guards"),
                }
                for ev in s["lineage_induction_events"]
            ]
            for s in hegel_streams
        },
    }

    all_dag_files = sorted(out_dir.rglob("01_decision_provenance_dag.json"))
    dag_valid_count = sum(1 for fp in all_dag_files if json.loads(fp.read_text(encoding="utf-8"))["provenance_verification"]["valid"])

    scope3_this_run = aggregate_call_usages(master_provider_receipts, deduplicate_uuid=True)
    aborted_ledgers = []
    for mp_ab in ABORTED_RUN_MANIFESTS:
        if mp_ab.is_file():
            ab_j = json.loads(mp_ab.read_text(encoding="utf-8"))
            ab_sp = ab_j.get("aborted_physical_spend") or ab_j.get("aborted_run1_physical_spend")
            if isinstance(ab_sp, dict):
                aborted_ledgers.append(ab_sp)
    from hegel_machine.cognition.cost_policy import sum_aggregated_ledgers

    scope3_plus_grand = sum_aggregated_ledgers([*aborted_ledgers, scope3_this_run]) if aborted_ledgers else scope3_this_run

    master_summary = {
        "schema": "hegel_m9_7_non_nested_counterfactual_lineage_summary_v1",
        "cost_policy_version": COST_POLICY_VERSION,
        "run_id": run_id,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "docker_sandbox_image_digest": env_image_digest,
        "configured_codex_concurrency_cap": args.max_codex_concurrency,
        "peak_concurrent_active_codex_calls_observed": active_counter["peak"],
        "configured_stream_workers": args.max_stream_workers,
        "code_commit_at_start": commit_start,
        "code_commit_at_end": commit_end,
        "source_tree_py_sha256_at_start": src_sha_start,
        "source_tree_py_sha256_at_end": src_sha_end,
        "tracked_tree_dirty_at_end": bool(tracked_dirty),
        "wall_clock_sec": round(monotonic() - t0_all, 2),
        "stage_S_cost_ledger": {"shared_stage_S_solve_6clusters": stage_s_solve_agg},
        "scope_3_campaign_physical_spend_all_calls": scope3_this_run,
        "scope_3_plus_grand_campaign_spend_including_aborted_runs": scope3_plus_grand,
        "arm_comparison": arm_agg,
        "paired_statistics_Hegel_vs_Flat_n6_clusters": paired_stats,
        "multibranch_lineage_and_counterfactual_funnel": lineage_funnel,
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
    (out_dir / "m9_7_continual_summary.json").write_text(json.dumps(master_summary, indent=2) + "\n", encoding="utf-8")
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
                        "zero_turn_passed_closures": v["zero_turn_passed_closures"],
                        "total_turns": v["total_turns_used"],
                        "in_stream_v1": v["scope_1_in_stream_total"]["proxy_v1_u_025c_1o"],
                        "in_stream_v2": v["scope_1_in_stream_total"]["proxy_v2_u_010c_4o"],
                        "standalone_v2": v["scope_2_standalone_with_stage_S"]["proxy_v2_u_010c_4o"],
                    }
                    for k, v in arm_agg.items()
                },
                "paired_statistics_n6_clusters": paired_stats,
                "lineage_funnel": lineage_funnel,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
