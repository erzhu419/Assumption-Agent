#!/usr/bin/env python3
"""End-to-End `32`-Way Parallel `gpt-6.2-sol` CLI Runner for `M9.8`:
Blind Residual-Mismatch Discovery, Strict Byte-Identical Counterfactual History (`R0 == U_iso`),
and Two-Stage Factored Mechanism Graph vs Online `Flat_ConditionalTable_Online` (`n = 8` Independent Clusters).

Pre-Registered Experimental Design (`A01–A17` Closed):
  - `n = 8` Completely Unexposed Confirmation Source-World Clusters (`alpha_m98_c01..c04`, `beta_m98_c01..c04`).
  - `3` Unhandicapped Online Continual Arms (`7` steps per cluster × `8` clusters = `56` episodes per arm = `168` continual
    episodes + `8` shared `Stage S` episodes = `176` live episodes):
      1. `Flat_LatestFirst`: Equal-info archive + shared `2/3–1/3` stratified OLS + `0444` prefix logs (`max_pre_llm_probes = 1`).
      2. `Flat_ConditionalTable_Online`: Live online conditional version table (`0` `PersistentMechanismGraph`) with its own
         live `gpt-6.2-sol` Kleene guard induction over all 12 generic observables (`up_stat_01..04 + dn_stat_01..08`) +
         `EmpiricalCoverageReceipt` checks (`max_pre_llm_probes = 1`).
      3. `Hegel_FactoredMechanismGraph`: Two-layer causal `PersistentMechanismGraph` (`Upstream {u0, u1} -> Downstream {d0, d1, d2}`)
         with causal parent holdout-residual locus attribution (`upstream_transducer` vs `downstream_constitutive`), checkpoint-persisted
         `witness_ledger` & `EmpiricalCoverageReceipt`s (`A10`), and zero-shot cross-factor composition (`u1 o d1`, `u1 o d2`).
  - Executed `HistoryMasked_CurrentOnly` Ablation Control (`A03`, `0` LLM calls): Proves in Docker that masking
    `archive_apparatus_log.json` on byte-identical `(R0, U_iso)` forces `<= 50%` pass ceiling (`8/16`), vs `16/16` history-aware!
  - Exact Call-UUID Hierarchical Cost Conservation (`N01`) asserted across `episodes == events == clusters == arm` (`v1` & `v2`).
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
import shutil
import subprocess
import sys
import threading
from time import monotonic
from typing import Any

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

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
from hegel_machine.cognition.m9_4_1_shared_verification_kernel import SHARED_KERNEL_VERSION
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
    EmpiricalCoverageReceipt,
    ImmutableBranchLineageController,
)
from hegel_machine.cognition.m9_8_blind_mismatch_and_selective_dependency_world import (
    CONFIRMATION_M98_WORLD_SPECS,
    M98_ALL_OBSERVABLE_NAMES,
    M98_ARMS,
    M98_DOWNSTREAM_OBSERVABLE_NAMES,
    M98_UPSTREAM_OBSERVABLE_NAMES,
    M98WorldSpec,
    assemble_m98_graph_diagonal_payload,
    build_m98_apparatus_log_entry,
    build_m98_default_u0_transducer_code,
    build_m98_upstream_transducer_synthesis_task,
    build_m98_world_tasks,
    compose_m98_two_stage_solver_code,
    compute_m98_combined_observables,
    compute_m98_downstream_observables,
    compute_m98_upstream_observables,
    diagnose_m98_factored_conflict_locus,
    expected_m98_public_output_keys,
    initialize_m98_factored_mechanism_graph,
    m98_task_to_frontier_task,
    materialize_m98_public_item_and_sha,
    propose_and_validate_m98_guard_partition,
    run_m98_equal_capability_pre_llm_trial,
    update_m98_calibration_snapshot_and_graph,
)

DEFAULT_V7_BUNDLE = REPO_ROOT / "docs/evidence/m9_2_1_audit_closure_and_regrade_20261126/pretrained_autogenous_forest_v7_m921.json"
DEFAULT_OUT_DIR = REPO_ROOT / "docs/evidence/m9_8_blind_mismatch_factored_graph_20261127"

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


def _extract_submitted_files(ep_dir: Path) -> dict[str, str]:
    sdir = ep_dir / "04_submitted_artifacts_snapshot"
    out: dict[str, str] = {}
    if not sdir.is_dir():
        return out
    for fp in sorted(sdir.rglob("*.py")):
        bname = fp.name.split("__", 1)[-1] if "__" in fp.name else fp.name
        out[bname] = fp.read_text(encoding="utf-8")
    return out


def _inject_equal_info_archive_into_m98_task(
    task: Any,
    prior_archive_entries: list[dict[str, Any]],
    prefix_apparatus_logs: list[dict[str, Any]],
    pre_llm_diagnostic: dict[str, Any] | None = None,
) -> tuple[Any, str]:
    """Identical `/tmp/hegel_continual_archive/` staging across ALL 3 online arms (`0` information asymmetry!)."""
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
                "solver_path": f"/tmp/hegel_continual_archive/{ep_tag}_solver.py",
            }
        )
    extra_lines.append(f"open('/tmp/hegel_continual_archive/archive_manifest.json', 'w').write({json.dumps(manifest_rows, indent=2)!r})")
    log_json = json.dumps(prefix_apparatus_logs, indent=2)
    extra_lines.append(
        "if os.path.exists('/tmp/hegel_continual_archive/archive_apparatus_log.json'): "
        "os.chmod('/tmp/hegel_continual_archive/archive_apparatus_log.json', 0o644)"
    )
    extra_lines.append(f"open('/tmp/hegel_continual_archive/archive_apparatus_log.json', 'w').write({log_json!r})")
    extra_lines.append("os.chmod('/tmp/hegel_continual_archive/archive_apparatus_log.json', 0o444)")
    if pre_llm_diagnostic is not None:
        extra_lines.append(
            f"open('/tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json', 'w').write({json.dumps(pre_llm_diagnostic, indent=2)!r})"
        )
    header = (
        "CONTINUAL STREAM ARCHIVE (`/tmp/hegel_continual_archive/`):\n"
        f"- Prior publicly-verified canonical solvers (you may inspect, reuse, or compose freely): {[r['solver_path'] for r in manifest_rows]}.\n"
        "- Manifest: `/tmp/hegel_continual_archive/archive_manifest.json`.\n"
        "- Read-only (`0444`) prefix apparatus equilibration log (`<= t-1`, also mounted in private verifier): "
        "`/tmp/hegel_continual_archive/archive_apparatus_log.json`."
    )
    if pre_llm_diagnostic is not None:
        header += f"\n- Pre-LLM Public Gate Diagnostic: `{json.dumps(pre_llm_diagnostic)}`."
    aug_task = type(task)(**{**task.__dict__, "public_setup_py": task.public_setup_py + "\n" + "\n".join(extra_lines) + "\n"})
    return aug_task, header


def _evaluate_empirical_coverage_m98(
    *,
    new_id: str,
    new_composed_builder: Any,
    old_id: str,
    old_solver_sha: str,
    old_parent_vers: dict[str, int],
    old_witness_specs: list[tuple[str, Any, dict[str, Any], dict[str, str], dict[str, str]]],
) -> EmpiricalCoverageReceipt:
    """Execute `new_composed_builder(w_task, w_pay)` inside air-gapped Docker container on `old_id`'s positive witness (`0` LLM calls)."""
    t0 = monotonic()
    passed = 0
    w_ids, w_inp_shas, p_vers, p_pay_shas = [], [], [], []
    new_sha = ""
    for wit_id, w_task, w_pay, ro_deps, ro_shas in old_witness_specs:
        cand_c, raw_new_sha = new_composed_builder(w_task, w_pay)
        new_sha = raw_new_sha
        w_ids.append(wit_id)
        w_inp_shas.append(hashlib.sha256(w_task.public_setup_py.encode("utf-8")).hexdigest())
        p_vers.append(dict(old_parent_vers))
        p_pay_shas.append(hashlib.sha256(json.dumps(w_pay, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest())
        pr = execute_single_pre_llm_public_gate_probe_in_container(
            task=w_task, candidate_code=cand_c, readonly_dependencies=ro_deps, expected_dependency_sha256=ro_shas
        )
        if pr["accepted"]:
            passed += 1
    el_ms = round((monotonic() - t0) * 1000.0, 2)
    n_ev = len(old_witness_specs)
    status = "unknown" if n_ev == 0 else ("supported_on_witnesses" if passed == n_ev else "refuted")
    rec = EmpiricalCoverageReceipt(
        receipt_id="",
        new_binding_id=new_id,
        new_solver_sha256=(new_sha or "0" * 64),
        old_binding_id=old_id,
        old_solver_sha256=old_solver_sha,
        witness_ids=w_ids,
        parameter_versions_by_witness=p_vers,
        verifier_version=SHARED_KERNEL_VERSION,
        status=status,
        passed_witness_count=passed,
        evaluated_witness_count=n_ev,
        total_cpu_probe_wall_ms=el_ms,
        mean_probe_wall_ms_ratio=1.0,
        witness_input_sha256s=w_inp_shas,
        parameter_payload_sha256s=p_pay_shas,
    )
    rec.receipt_id = rec.compute_expected_receipt_id()
    rec.verify_integrity()
    return rec


def _exact_two_sided_sign_flip_p(diffs: list[float]) -> float:
    arr = np.asarray(diffs, dtype=float)
    n = len(arr)
    if n == 0:
        return 1.0
    obs = abs(float(np.mean(arr)))
    if obs <= 1e-15:
        return 1.0
    ext = sum(
        1
        for signs in itertools.product((-1.0, 1.0), repeat=n)
        if abs(float(np.mean(arr * np.asarray(signs, dtype=float)))) >= obs - 1e-12
    )
    return round(float(ext) / float(1 << n), 7)


def _bootstrap_mean_ci(diffs: list[float], seed: int = 98099, n_boot: int = 10000) -> list[float]:
    arr = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = np.mean(arr[idx], axis=1)
    return [round(float(np.percentile(means, 2.5)), 2), round(float(np.percentile(means, 97.5)), 2)]


def _run_one_m98_world_arm_stream(
    *,
    spec: M98WorldSpec,
    arm: str,
    s_task: Any,
    stream_tasks: tuple[Any, ...],
    schedule_manifest: list[dict[str, Any]],
    apparatus_logs: list[dict[str, Any]],
    base_v7_forest: Any,
    base_v7_archive: dict[str, dict[str, Any]],
    base_graph_ckpt_path: Path,
    source_archive_entry: dict[str, Any],
    source_forest_nid: str,
    source_pub_sha256: str,
    source_up_obs: dict[str, float | None],
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
    uni_keys = expected_m98_public_output_keys(fam)
    u0_code = build_m98_default_u0_transducer_code(fam)

    forest = deepcopy(base_v7_forest)
    archive = deepcopy(base_v7_archive)
    archive["__autogenous_blank_seed__"] = {"enabled": True}

    flat_snap: dict[str, Any] = {"gains": [1.0, 1.0, 1.0, 1.0], "offsets": [0.0, 0.0, 0.0, 0.0]}
    prior_entries: list[dict[str, Any]] = [deepcopy(source_archive_entry)]

    # Arm-specific online state
    flat_table: list[dict[str, Any]] | None = None
    flat_pos_obs_by_row: dict[str, list[dict[str, float | None]]] = {}
    flat_neg_obs_by_row: dict[str, list[dict[str, float | None]]] = {}
    flat_rep_wit_by_row: dict[str, list[tuple[str, Any, dict[str, Any], dict[str, str], dict[str, str]]]] = {}
    flat_cov_pairs: set[tuple[str, str]] = set()

    s_init_log = [build_m98_apparatus_log_entry(spec=spec, step_index=-1, short_name=f"{wid}_init", next_downstream_regime="D0")]
    if arm == "Flat_ConditionalTable_Online":
        src_code = str(source_archive_entry["canonical_algorithm_code"])
        src_ep = str(source_archive_entry["canonical_entrypoint"])
        r0_id = f"row_{wid}_00"
        flat_table = [
            {
                "row_id": r0_id,
                "guard": {},
                "canonical_code": src_code,
                "entrypoint": src_ep,
                "code_sha256": hashlib.sha256(src_code.encode("utf-8")).hexdigest(),
                "priority": 10,
                "output_schema_keys": list(uni_keys),
                "uses_raw_bridge": False,
            }
        ]
        s_comb_obs = {**source_up_obs, **compute_m98_downstream_observables(s_init_log)}
        flat_pos_obs_by_row[r0_id] = [s_comb_obs]
        flat_neg_obs_by_row[r0_id] = []
        flat_rep_wit_by_row[r0_id] = [(f"wit_src_{wid}", s_task, {"gains": [1.0] * 4, "offsets": [0.0] * 4}, {}, {})]

    if arm == "Hegel_FactoredMechanismGraph":
        graph = PersistentMechanismGraph.load_checkpoint(base_graph_ckpt_path)
        forest = graph.forest
        registry = M94BindingRegistry(graph)
        up_ctrl = ImmutableBranchLineageController(world_id=f"{wid}_up", family=f"{fam}_up", graph=graph, registry=registry)
        dn_ctrl = ImmutableBranchLineageController(world_id=f"{wid}_dn", family=f"{fam}_dn", graph=graph, registry=registry)
        u0_b = registry.bindings[f"bind_{wid}_up_b00"]
        d0_b = registry.bindings[f"bind_{wid}_dn_b00"]
        up_ctrl.record_tested_witness(
            binding=u0_b, public_input_sha256=source_pub_sha256, passed=True, failure_kind="none", features=source_up_obs
        )
        s_dn_obs = compute_m98_downstream_observables(s_init_log)
        dn_ctrl.record_tested_witness(
            binding=d0_b, public_input_sha256=source_pub_sha256, passed=True, failure_kind="none", features=s_dn_obs
        )
        dn_rep_wit_by_bid: dict[str, list[tuple[str, Any, dict[str, Any], dict[str, str], dict[str, str]]]] = {
            d0_b.binding_id: [(f"wit_src_{wid}", s_task, {"gains": [1.0] * 4, "offsets": [0.0] * 4}, {}, {})]
        }
    else:
        graph = None
        registry = None
        up_ctrl = None
        dn_ctrl = None
        dn_rep_wit_by_bid = {}

    comp_client = m6f.FrontierCodexClient(
        codex_info=codex_info, model=model, reasoning_effort=reasoning_effort, timeout_seconds=300
    )

    episode_results: list[dict[str, Any]] = []
    guard_induction_events: list[dict[str, Any]] = []
    coverage_receipts_log: list[dict[str, Any]] = []
    n_steps = len(stream_tasks)

    for step_i, (task, m_info) in enumerate(zip(stream_tasks, schedule_manifest), start=1):
        ep_t0 = monotonic()
        ep_dir = stream_out_dir / task.task_id
        ep_dir.mkdir(parents=True, exist_ok=True)
        pub_item, pub_sha = materialize_m98_public_item_and_sha(task)

        # Prefix apparatus logs `0 .. step_i - 1` (`<= t - 1`, `0` future leakage!)
        prefix_logs = apparatus_logs[:step_i]
        log_json = json.dumps(prefix_logs, indent=2)
        readonly_deps = {"/tmp/hegel_continual_archive/archive_apparatus_log.json": log_json}
        readonly_shas = {k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in readonly_deps.items()}

        cal_upd = update_m98_calibration_snapshot_and_graph(
            spec=spec,
            episode_id=f"{wid}_{task.short_name}",
            public_item=pub_item,
            flat_snapshot=(flat_snap if arm != "Hegel_FactoredMechanismGraph" else None),
            graph=graph,
            registry=registry,
        )
        if "flat_snapshot" in cal_upd:
            flat_snap = cal_upd["flat_snapshot"]

        pre_trial = run_m98_equal_capability_pre_llm_trial(
            arm=arm,
            spec=spec,
            task=task,
            public_item=pub_item,
            public_input_sha256=pub_sha,
            prefix_apparatus_logs=prefix_logs,
            prior_archive_entries=prior_entries,
            flat_param_snapshot=(flat_snap if arm != "Hegel_FactoredMechanismGraph" else None),
            flat_conditional_table=flat_table,
            up_ctrl=up_ctrl,
            dn_ctrl=dn_ctrl,
            graph=graph,
            registry=registry,
            readonly_dependencies=readonly_deps,
            expected_dependency_sha256=readonly_shas,
        )

        if pre_trial.get("accepted_pre_llm") is True:
            cand_code = str(pre_trial["candidate_code"])
            can_code_sel = str(pre_trial.get("selected_canonical_algorithm_code") or cand_code)
            can_ep_sel = str(pre_trial.get("selected_canonical_entrypoint") or "solve_batch")
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
            if arm == "Flat_ConditionalTable_Online":
                sel_rid = str(pre_trial["selected_row_id"])
                flat_pos_obs_by_row.setdefault(sel_rid, []).append(pre_trial["observables"])

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
                "calibration_update_receipt": cal_upd.get("graph_receipt"),
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
                    delivered_entrypoint="solve_batch",
                )
            )
            if graph is not None:
                graph.save_checkpoint(stream_out_dir / f"graph_ckpt_after_step_{step_i:02d}.json")
            episode_results.append(summary)
            print(
                f"  [M9.8 {wid} | {arm} | step {step_i}/{n_steps} {task.short_name} ({m_info['causal_event']})] "
                f"PRE_LLM_DIRECT | pass={summary['strict_pass']} | turns=0 | probe={pre_trial.get('probe_wall_ms', 0):.0f}ms",
                flush=True,
            )
            continue

        # Record negative witness for `Flat_ConditionalTable_Online` when scientific/domain failure occurred
        if arm == "Flat_ConditionalTable_Online" and pre_trial.get("failure_kind") in {"scientific_or_numerical", "domain_precondition"}:
            flat_neg_obs_by_row.setdefault(str(pre_trial["selected_row_id"]), []).append(pre_trial["observables"])

        # Determine synthesis task (for `Hegel_FactoredMechanismGraph`, check causal locus!)
        locus = (pre_trial.get("conflict_locus_diagnosis") or {}).get("locus", "downstream_constitutive")
        is_hegel_upstream_synth = bool(arm == "Hegel_FactoredMechanismGraph" and locus == "upstream_transducer")

        if is_hegel_upstream_synth and registry is not None:
            dn_bid_sel = str(pre_trial["selected_downstream_binding_id"])
            dn_b_obj = registry.bindings[dn_bid_sel]
            synth_base_task = build_m98_upstream_transducer_synthesis_task(
                base_task=task,
                family=fam,
                downstream_code=(dn_b_obj.canonical_algorithm_code or dn_b_obj.parameterized_closure_code),
                downstream_entrypoint=(dn_b_obj.canonical_entrypoint or dn_b_obj.parameterized_entrypoint),
            )
        else:
            synth_base_task = task

        diag_for_prompt = {
            "arm": arm,
            "pre_llm_reason": pre_trial.get("reason"),
            "conflict_locus": (locus if arm == "Hegel_FactoredMechanismGraph" else "monolithic_task"),
        }
        aug_task, ctx_hdr = _inject_equal_info_archive_into_m98_task(synth_base_task, prior_entries, prefix_logs, diag_for_prompt)
        ft = m98_task_to_frontier_task(
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
            enable_auto_skill_packaging=(arm == "Hegel_FactoredMechanismGraph"),
            allow_l1_primitives=(arm == "Hegel_FactoredMechanismGraph"),
            disclosure_mode="on_demand_executable",
            staging_policy="net_value_gated",
        )
        sub_files = _extract_submitted_files(ep_dir)
        up_mod_txt = sub_files.get("upstream_transducer.py", "")
        batch_mod_txt = sub_files.get("batch_solver.py", "")

        # Build the authoritative standalone `batch_solver.py` artifact for grading!
        if is_hegel_upstream_synth and up_mod_txt and registry is not None and graph is not None:
            dn_b_obj = registry.bindings[str(pre_trial["selected_downstream_binding_id"])]
            graded_cand_txt = compose_m98_two_stage_solver_code(
                upstream_code=up_mod_txt,
                upstream_entrypoint="calibrate_upstream",
                downstream_code=(dn_b_obj.canonical_algorithm_code or dn_b_obj.parameterized_closure_code),
                downstream_entrypoint=(dn_b_obj.canonical_entrypoint or dn_b_obj.parameterized_entrypoint),
                param_payload=assemble_m98_graph_diagonal_payload(spec, graph),
            )
            canonical_saved_txt = up_mod_txt
            canonical_saved_ep = "calibrate_upstream"
        else:
            graded_cand_txt = batch_mod_txt
            canonical_saved_txt = batch_mod_txt
            canonical_saved_ep = "solve_batch"
            is_hegel_upstream_synth = False  # Fallback if agent only edited `batch_solver.py`

        cand_sha = hashlib.sha256(graded_cand_txt.encode("utf-8")).hexdigest() if graded_cand_txt else ""
        pub_gate_post = (
            execute_single_pre_llm_public_gate_probe_in_container(
                task=task,
                candidate_code=graded_cand_txt,
                readonly_dependencies=readonly_deps,
                expected_dependency_sha256=readonly_shas,
            )
            if graded_cand_txt
            else {"accepted": False}
        )
        priv_ver = (
            grade_m94_artifact_map(
                task,
                {str(task.declared_artifacts[0]): graded_cand_txt},
                readonly_dependencies=readonly_deps,
                expected_dependency_sha256=readonly_shas,
            )
            if graded_cand_txt
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

        ind_receipt: dict[str, Any] | None = None
        cov_this_step: list[dict[str, Any]] = []

        # Online Guard & Lineage Update (`Flat_ConditionalTable_Online` vs `Hegel_FactoredMechanismGraph`)
        if pub_gate_post.get("accepted") and graded_cand_txt and step_i < n_steps:
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

            if arm == "Flat_ConditionalTable_Online" and flat_table is not None:
                new_rid = f"row_{wid}_{len(flat_table):02d}"
                new_raw_sha = hashlib.sha256(batch_mod_txt.encode("utf-8")).hexdigest()
                comb_now = compute_m98_combined_observables(fam, pub_item, prefix_logs)
                uses_raw = bool(float(comb_now.get("up_stat_01_diagonal_ols_r2") or 1.0) < 0.96)
                all_cov = True
                for old_r in flat_table:
                    rep_w = flat_rep_wit_by_row.get(old_r["row_id"], [])[:1]
                    cov_r = _evaluate_empirical_coverage_m98(
                        new_id=new_rid,
                        new_composed_builder=lambda _wt, wp, _ur=uses_raw: (
                            compose_m98_two_stage_solver_code(
                                upstream_code=u0_code,
                                upstream_entrypoint="calibrate_upstream",
                                downstream_code=batch_mod_txt,
                                downstream_entrypoint="solve_batch",
                                param_payload=({"gains": [1.0] * 4, "offsets": [0.0] * 4} if _ur else wp),
                            ),
                            new_raw_sha,
                        ),
                        old_id=old_r["row_id"],
                        old_solver_sha=old_r["code_sha256"],
                        old_parent_vers={"flat": 1},
                        old_witness_specs=rep_w,
                    )
                    cov_this_step.append(cov_r.to_dict())
                    coverage_receipts_log.append(cov_r.to_dict())
                    if cov_r.status == "supported_on_witnesses":
                        flat_cov_pairs.add((new_rid, old_r["row_id"]))
                    else:
                        all_cov = False

                if all_cov:
                    flat_table.append(
                        {
                            "row_id": new_rid,
                            "guard": {},
                            "canonical_code": batch_mod_txt,
                            "entrypoint": "solve_batch",
                            "code_sha256": new_raw_sha,
                            "priority": 50 + step_i,
                            "output_schema_keys": list(uni_keys),
                            "uses_raw_bridge": uses_raw,
                        }
                    )
                else:
                    ex_ids = [r["row_id"] for r in flat_table]
                    pos_map = {rid: list(flat_pos_obs_by_row.get(rid, [])) for rid in ex_ids}
                    pos_map[new_rid] = [comb_now]
                    ind_res = propose_and_validate_m98_guard_partition(
                        client=comp_client,
                        stage_subdir=f"{ep_run_tag}__flat_gind",
                        scope_label="flat_conditional_table_12feat",
                        feature_names=M98_ALL_OBSERVABLE_NAMES,
                        existing_branch_ids=ex_ids,
                        new_branch_id_placeholder=new_rid,
                        positive_witness_features_by_branch=pos_map,
                        negative_witness_features_by_branch=flat_neg_obs_by_row,
                        covered_branch_pairs=flat_cov_pairs,
                    )
                    ind_receipt = {k: v for k, v in ind_res.items() if k != "provider_call_receipts"}
                    guard_induction_events.append({"step": step_i, "event": m_info["causal_event"], "layer": "flat_table", **ind_receipt})
                    if ind_res.get("accepted"):
                        for r_obj in flat_table:
                            if r_obj["row_id"] in ind_res["existing_branch_guards"]:
                                r_obj["guard"] = deepcopy(ind_res["existing_branch_guards"][r_obj["row_id"]])
                        flat_table.append(
                            {
                                "row_id": new_rid,
                                "guard": deepcopy(ind_res["new_branch_guard"]),
                                "canonical_code": batch_mod_txt,
                                "entrypoint": "solve_batch",
                                "code_sha256": new_raw_sha,
                                "priority": 10 + step_i,
                                "output_schema_keys": list(uni_keys),
                                "uses_raw_bridge": uses_raw,
                            }
                        )
                flat_pos_obs_by_row[new_rid] = [comb_now]
                flat_rep_wit_by_row[new_rid] = [
                    (
                        f"wit_{task.short_name}",
                        task,
                        ({"gains": [1.0] * 4, "offsets": [0.0] * 4} if uses_raw else deepcopy(flat_snap)),
                        readonly_deps,
                        readonly_shas,
                    )
                ]

            elif arm == "Hegel_FactoredMechanismGraph" and graph is not None and registry is not None and up_ctrl is not None and dn_ctrl is not None:
                ev_sha = hashlib.sha256(json.dumps(pub_gate_post.get("scientific_probe_receipt") or {}, sort_keys=True).encode("utf-8")).hexdigest()
                if is_hegel_upstream_synth:
                    failed_up_id = str(pre_trial["selected_upstream_binding_id"])
                    ex_ups = sorted(
                        [b for b in registry.bindings.values() if b.chain_id == f"{fam}_up" and b.qualification_status == "verified"],
                        key=lambda x: x.binding_id,
                    )
                    ph_up = f"bind_{wid}_up_b{up_ctrl._branch_seq:02d}"
                    up_obs_now = compute_m98_upstream_observables(fam, pub_item)
                    pos_up = {b.binding_id: [w.public_structural_features for w in up_ctrl.positive_witnesses_for(b.binding_id)] for b in ex_ups}
                    pos_up[ph_up] = [up_obs_now]
                    neg_up = {b.binding_id: [w.public_structural_features for w in up_ctrl.scientific_negative_witnesses_for(b.binding_id)] for b in ex_ups}
                    ind_res = propose_and_validate_m98_guard_partition(
                        client=comp_client,
                        stage_subdir=f"{ep_run_tag}__up_gind",
                        scope_label="upstream_transducer_layer",
                        feature_names=M98_UPSTREAM_OBSERVABLE_NAMES,
                        existing_branch_ids=[b.binding_id for b in ex_ups],
                        new_branch_id_placeholder=ph_up,
                        positive_witness_features_by_branch=pos_up,
                        negative_witness_features_by_branch=neg_up,
                    )
                    ind_receipt = {k: v for k, v in ind_res.items() if k != "provider_call_receipts"}
                    guard_induction_events.append({"step": step_i, "event": m_info["causal_event"], "layer": "upstream_transducer", **ind_receipt})
                    if ind_res.get("accepted"):
                        new_u = up_ctrl.mint_immutable_conflict_branch(
                            failed_binding_id=failed_up_id,
                            new_canonical_code=up_mod_txt,
                            new_entrypoint="calibrate_upstream",
                            forest_node_id=source_forest_nid,
                            qualification_protocol="hegel_public_scientific_gate_v4",
                            output_schema_keys=["calibrated_input"],
                            validated_parent_ids=["cal_node_p0", "cal_node_p1"],
                            new_applicability_guard=dict(ind_res["new_branch_guard"]),
                            updated_failed_branch_applicability_guard=None,
                            violated_component_id="diagonal_channel_independence",
                            preserved_component_ids=["downstream_constitutive_laws"],
                            selection_priority=20 + step_i,
                            environment_digest=env_image_digest,
                            evidence_digest=ev_sha,
                        )
                        for ex_bid, ex_g in (ind_res.get("existing_branch_guards") or {}).items():
                            if ex_bid in registry.bindings:
                                registry.bindings[ex_bid].applicability_guard = deepcopy(ex_g)
                                registry.bindings[ex_bid].precondition_guard = deepcopy(ex_g)
                        registry.sync_to_graph()
                        up_ctrl.record_tested_witness(
                            binding=new_u, public_input_sha256=pub_sha, passed=True, failure_kind="none", features=up_obs_now
                        )
                else:
                    failed_dn_id = str(pre_trial["selected_downstream_binding_id"])
                    ex_dns = sorted(
                        [b for b in registry.bindings.values() if b.chain_id == f"{fam}_dn" and b.qualification_status == "verified"],
                        key=lambda x: x.binding_id,
                    )
                    ph_dn = f"bind_{wid}_dn_b{dn_ctrl._branch_seq:02d}"
                    dn_raw_sha = hashlib.sha256(batch_mod_txt.encode("utf-8")).hexdigest()
                    all_cov_dn = True
                    for old_d in ex_dns:
                        rep_w = dn_rep_wit_by_bid.get(old_d.binding_id, [])[:1]
                        cov_r = _evaluate_empirical_coverage_m98(
                            new_id=ph_dn,
                            new_composed_builder=lambda _wt, wp: (
                                compose_m98_two_stage_solver_code(
                                    upstream_code=u0_code,
                                    upstream_entrypoint="calibrate_upstream",
                                    downstream_code=batch_mod_txt,
                                    downstream_entrypoint="solve_batch",
                                    param_payload=wp,
                                ),
                                dn_raw_sha,
                            ),
                            old_id=old_d.binding_id,
                            old_solver_sha=old_d.authoritative_executed_code_sha256(),
                            old_parent_vers=dict(old_d.parent_parameter_versions),
                            old_witness_specs=rep_w,
                        )
                        dn_ctrl.record_coverage_receipt(cov_r)
                        cov_this_step.append(cov_r.to_dict())
                        coverage_receipts_log.append(cov_r.to_dict())
                        if cov_r.status != "supported_on_witnesses":
                            all_cov_dn = False

                    dn_obs_now = compute_m98_downstream_observables(prefix_logs)
                    if all_cov_dn:
                        new_d = dn_ctrl.mint_immutable_conflict_branch(
                            failed_binding_id=failed_dn_id,
                            new_canonical_code=batch_mod_txt,
                            new_entrypoint="solve_batch",
                            forest_node_id=source_forest_nid,
                            qualification_protocol="hegel_public_scientific_gate_v4",
                            output_schema_keys=uni_keys,
                            validated_parent_ids=["constitutive_root_dn"],
                            new_applicability_guard={},
                            updated_failed_branch_applicability_guard=None,
                            violated_component_id="empirically_covering_downstream",
                            preserved_component_ids=[],
                            selection_priority=50 + step_i,
                            environment_digest=env_image_digest,
                            evidence_digest=ev_sha,
                        )
                        dn_ctrl.record_tested_witness(
                            binding=new_d, public_input_sha256=pub_sha, passed=True, failure_kind="none", features=dn_obs_now
                        )
                        dn_rep_wit_by_bid[new_d.binding_id] = [
                            (f"wit_{task.short_name}", task, assemble_m98_graph_diagonal_payload(spec, graph), readonly_deps, readonly_shas)
                        ]
                    else:
                        pos_dn = {b.binding_id: [w.public_structural_features for w in dn_ctrl.positive_witnesses_for(b.binding_id)] for b in ex_dns}
                        pos_dn[ph_dn] = [dn_obs_now]
                        neg_dn = {b.binding_id: [w.public_structural_features for w in dn_ctrl.scientific_negative_witnesses_for(b.binding_id)] for b in ex_dns}
                        cov_pairs_dn = {(cr.new_binding_id, cr.old_binding_id) for cr in dn_ctrl.coverage_receipts if cr.status == "supported_on_witnesses"}
                        ind_res = propose_and_validate_m98_guard_partition(
                            client=comp_client,
                            stage_subdir=f"{ep_run_tag}__dn_gind",
                            scope_label="downstream_constitutive_layer",
                            feature_names=M98_DOWNSTREAM_OBSERVABLE_NAMES,
                            existing_branch_ids=[b.binding_id for b in ex_dns],
                            new_branch_id_placeholder=ph_dn,
                            positive_witness_features_by_branch=pos_dn,
                            negative_witness_features_by_branch=neg_dn,
                            covered_branch_pairs=cov_pairs_dn,
                        )
                        ind_receipt = {k: v for k, v in ind_res.items() if k != "provider_call_receipts"}
                        guard_induction_events.append({"step": step_i, "event": m_info["causal_event"], "layer": "downstream_constitutive", **ind_receipt})
                        if ind_res.get("accepted"):
                            new_d = dn_ctrl.mint_immutable_conflict_branch(
                                failed_binding_id=failed_dn_id,
                                new_canonical_code=batch_mod_txt,
                                new_entrypoint="solve_batch",
                                forest_node_id=source_forest_nid,
                                qualification_protocol="hegel_public_scientific_gate_v4",
                                output_schema_keys=uni_keys,
                                validated_parent_ids=["constitutive_root_dn"],
                                new_applicability_guard=dict(ind_res["new_branch_guard"]),
                                updated_failed_branch_applicability_guard=None,
                                violated_component_id="constitutive_regime_shift",
                                preserved_component_ids=["upstream_transducer"],
                                selection_priority=10 + step_i,
                                environment_digest=env_image_digest,
                                evidence_digest=ev_sha,
                            )
                            for ex_bid, ex_g in (ind_res.get("existing_branch_guards") or {}).items():
                                if ex_bid in registry.bindings:
                                    registry.bindings[ex_bid].applicability_guard = deepcopy(ex_g)
                                    registry.bindings[ex_bid].precondition_guard = deepcopy(ex_g)
                            registry.sync_to_graph()
                            dn_ctrl.record_tested_witness(
                                binding=new_d, public_input_sha256=pub_sha, passed=True, failure_kind="none", features=dn_obs_now
                            )
                            dn_rep_wit_by_bid[new_d.binding_id] = [
                                (f"wit_{task.short_name}", task, assemble_m98_graph_diagonal_payload(spec, graph), readonly_deps, readonly_shas)
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
        ep_sum["execution_mode"] = ("llm_upstream_transducer_synthesis" if is_hegel_upstream_synth else "llm_synthesis_after_equal_tool_gate")
        ep_sum["strict_pass"] = bool(priv_ver.get("strict_pass"))
        ep_sum["reward"] = 1.0 if priv_ver.get("strict_pass") else 0.0
        ep_sum["pre_llm_probe_wall_ms"] = float(pre_trial.get("probe_wall_ms", 0.0))
        ep_sum["readonly_dependencies_sha256"] = readonly_shas
        ep_sum["public_qualification_status"] = "public_gate_verified" if pub_gate_post.get("accepted") else "public_gate_failed"
        ep_sum["calibration_update_receipt"] = cal_upd.get("graph_receipt")
        ep_sum["empirical_coverage_receipts"] = cov_this_step
        ep_sum["guard_induction_receipt"] = ind_receipt
        ep_sum["pre_llm_trial_receipt"] = {k: v for k, v in pre_trial.items() if k != "candidate_code"}
        ep_sum["provenance_dag_verification"] = dag_chk
        ep_sum["verifier_report"] = priv_ver
        (ep_dir / "00_episode_summary.json").write_text(json.dumps(ep_sum, indent=2) + "\n", encoding="utf-8")
        (ep_dir / "03_official_verifier_report.json").write_text(json.dumps(priv_ver, indent=2) + "\n", encoding="utf-8")

        if graded_cand_txt and not is_hegel_upstream_synth:
            prior_entries.append(
                make_continual_archive_entry(
                    short_name=task.short_name,
                    task_id=task.task_id,
                    canonical_entrypoint=canonical_saved_ep,
                    canonical_algorithm_code=canonical_saved_txt,
                    output_schema_keys=uni_keys,
                    public_qualification_status=("public_gate_verified" if pub_gate_post.get("accepted") else "public_gate_failed"),
                )
            )
        if graph is not None:
            graph.save_checkpoint(stream_out_dir / f"graph_ckpt_after_step_{step_i:02d}.json")
        episode_results.append(ep_sum)
        print(
            f"  [M9.8 {wid} | {arm} | step {step_i}/{n_steps} {task.short_name} ({m_info['causal_event']})] "
            f"{ep_sum['execution_mode']} | pass={ep_sum['strict_pass']} | turns={ep_sum.get('turns_used')} "
            f"| gind={ind_receipt.get('accepted') if ind_receipt else None}",
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
        "total_coverage_probe_wall_ms": round(sum(float(c["total_cpu_probe_wall_ms"]) for c in coverage_receipts_log), 2),
        "guard_induction_events": guard_induction_events,
        "empirical_coverage_receipts": coverage_receipts_log,
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
            }
            for r in episode_results
        ],
    }


def run_m98_history_masked_ablation_on_sealed_streams(
    *,
    built_worlds: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    """`A03` Executed `HistoryMasked_CurrentOnly` Control (`0` LLM calls, real Docker private verification):
    Evaluates `Hegel_FactoredMechanismGraph` at `R0` (`Step 2`) and `U_iso` (`Step 5`) using the frozen prefix checkpoint
    `graph_ckpt_after_step_04.json` when `archive_apparatus_log.json` is masked (`[]`)!
    """
    rows = []
    for spec in CONFIRMATION_M98_WORLD_SPECS:
        wid = spec.world_id
        _, tasks, manifest, _ = built_worlds[wid]
        r0_idx = next(i for i, m in enumerate(manifest) if m["causal_event"] == "R0")
        ui_idx = next(i for i, m in enumerate(manifest) if m["causal_event"] == "U_iso")
        r0_t, ui_t = tasks[r0_idx], tasks[ui_idx]
        _, r0_sha = materialize_m98_public_item_and_sha(r0_t)
        _, ui_sha = materialize_m98_public_item_and_sha(ui_t)
        assert r0_sha == ui_sha and r0_t.instruction_text == ui_t.instruction_text

        ck4 = out_dir / "continual_streams" / wid / "Hegel_FactoredMechanismGraph" / "graph_ckpt_after_step_04.json"
        g4 = PersistentMechanismGraph.load_checkpoint(ck4)
        r4 = M94BindingRegistry(g4)
        up4 = ImmutableBranchLineageController(world_id=f"{wid}_up", family=f"{spec.family}_up", graph=g4, registry=r4)
        dn4 = ImmutableBranchLineageController(world_id=f"{wid}_dn", family=f"{spec.family}_dn", graph=g4, registry=r4)

        for ev_nm, tk in (("R0", r0_t), ("U_iso", ui_t)):
            pub_item, pub_sha = materialize_m98_public_item_and_sha(tk)
            # Masked history (`prefix_apparatus_logs = []`) -> `dn_stat_01..08` are all `None` -> `UNKNOWN` -> identical fallback!
            up_obs = compute_m98_upstream_observables(spec.family, pub_item)
            up_sel = up4.select_branch_deterministically(req_schema_keys=["calibrated_input"], feature_evaluator=lambda _b: up_obs)
            c_up = up_sel["selected"] or sorted(
                [b for b in r4.bindings.values() if b.chain_id == f"{spec.family}_up" and b.qualification_status == "verified"],
                key=lambda x: (-int(x.selection_priority), x.binding_id),
            )[0]
            dn_ver = sorted(
                [b for b in r4.bindings.values() if b.chain_id == f"{spec.family}_dn" and b.qualification_status == "verified"],
                key=lambda x: (-int(x.selection_priority), x.binding_id),
            )
            c_dn = dn_ver[0]
            cand = compose_m98_two_stage_solver_code(
                upstream_code=(c_up.canonical_algorithm_code or c_up.parameterized_closure_code),
                upstream_entrypoint=(c_up.canonical_entrypoint or c_up.parameterized_entrypoint),
                downstream_code=(c_dn.canonical_algorithm_code or c_dn.parameterized_closure_code),
                downstream_entrypoint=(c_dn.canonical_entrypoint or c_dn.parameterized_entrypoint),
                param_payload=assemble_m98_graph_diagonal_payload(spec, g4),
            )
            ro_masked = {"/tmp/hegel_continual_archive/archive_apparatus_log.json": "[]"}
            ro_shas = {k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in ro_masked.items()}
            ver = grade_m94_artifact_map(tk, {str(tk.declared_artifacts[0]): cand}, readonly_dependencies=ro_masked, expected_dependency_sha256=ro_shas)
            rows.append(
                {
                    "world_id": wid,
                    "family": spec.family,
                    "event": ev_nm,
                    "public_batch_pkl_sha256": pub_sha,
                    "masked_selected_downstream": c_dn.binding_id,
                    "masked_private_strict_pass": bool(ver.get("strict_pass")),
                }
            )
    return {
        "schema": "hegel_m9_8_a03_history_masked_current_only_ablation_v1",
        "paid_model_calls": 0,
        "evidence_provenance": "executed",
        "all_8_clusters_byte_identical_public_pkl_and_prompt": True,
        "total_twin_episodes": len(rows),
        "history_masked_strict_passes": sum(1 for r in rows if r["masked_private_strict_pass"]),
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--v7-bundle", type=Path, default=DEFAULT_V7_BUNDLE)
    ap.add_argument("--model", type=str, default="gpt-6.2-sol")
    ap.add_argument("--reasoning-effort", type=str, default="medium")
    ap.add_argument("--max-turns", type=int, default=18)
    ap.add_argument("--max-codex-concurrency", type=int, default=32)
    ap.add_argument("--max-stream-workers", type=int, default=24)
    args = ap.parse_args()

    os.environ["HEGEL_MAX_CODEX_CONCURRENCY"] = str(args.max_codex_concurrency)
    out_dir: Path = args.out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory `{out_dir}`!")

    git_sha, is_dirty, src_tree_sha = _git_and_source_tree_state()
    if is_dirty:
        raise RuntimeError("Refusing to start M9.8 confirmation experiment on a dirty working tree!")

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_mirror_dir = out_dir / "provider_raw_attempts_mirror"
    raw_mirror_dir.mkdir(parents=True, exist_ok=True)

    ensure_clean_scipy_sandbox_image()
    m64.ensure_runtime_qualification_context(sandbox_cid=None)
    env_img_digest = _docker_image_digest(m7a.CLEAN_SCIPY_SANDBOX_IMAGE)

    codex_info = m6f.discover_chatgpt_subscription_codex_cli()
    if not (codex_info.get("available") and codex_info.get("auth_mode") == "chatgpt"):
        raise RuntimeError(f"Official ChatGPT Codex CLI unavailable: {codex_info}")

    # Mirror 100% of raw provider call attempts (`call_uuid` conserved & concurrency bounded at `32`!)
    cli_sem = threading.BoundedSemaphore(max(1, int(args.max_codex_concurrency)))
    all_receipts: list[dict[str, Any]] = []
    receipts_lock = threading.Lock()
    active_counter = {"cur": 0, "peak": 0}
    orig_call_structured = m6f.FrontierCodexClient.call_structured

    def _wrapped_call_structured(self_cli: Any, **c_kwargs: Any) -> dict[str, Any]:
        self_cli.immediate_archive_dir = raw_mirror_dir
        meta = getattr(self_cli, "_hegel_call_meta", None) or {}
        role = str(meta.get("role") or getattr(_CALL_CONTEXT, "role", "unknown"))
        wid = str(meta.get("world_id") or getattr(_CALL_CONTEXT, "world_id", "unknown"))
        cid = str(meta.get("cluster_id") or getattr(_CALL_CONTEXT, "cluster_id", "unknown"))
        arm = str(meta.get("arm") or getattr(_CALL_CONTEXT, "arm", "Stage_S_Shared_Source"))
        ep_sn = str(meta.get("episode_short_name") or getattr(_CALL_CONTEXT, "episode_short_name", "unknown"))
        mono_t0 = monotonic()
        with cli_sem:
            with receipts_lock:
                active_counter["cur"] += 1
                if active_counter["cur"] > active_counter["peak"]:
                    active_counter["peak"] = active_counter["cur"]
            try:
                res = orig_call_structured(self_cli, **c_kwargs)
            finally:
                with receipts_lock:
                    active_counter["cur"] -= 1
        mono_t1 = monotonic()
        sc = score_single_call_usage(
            prompt_tokens=int(res.get("prompt_tokens", 0) or 0),
            cached_tokens=int(res.get("cached_tokens", 0) or 0),
            completion_tokens=int(res.get("completion_tokens", 0) or 0),
            wall_ms=float(res.get("wall_ms", 0.0) or 0.0),
        )
        cu_uuid = str(res.get("call_uuid") or "")
        att_fn = str(res.get("attempts_ledger_filename") or "")
        unk_cnt = int(res.get("usage_unknown_attempt_count", 0) or 0)
        rec_row = {
            "call_uuid": cu_uuid,
            "world_id": wid,
            "cluster_id": cid,
            "arm": arm,
            "episode_short_name": ep_sn,
            "call_role": role,
            "stage_subdir": str(c_kwargs.get("stage_subdir") or ""),
            "attempts_ledger_filename": att_fn,
            "mirrored_raw_filename": f"{cu_uuid}__{att_fn}",
            "monotonic_start_sec": round(mono_t0, 4),
            "monotonic_end_sec": round(mono_t1, 4),
            "ok": bool(res.get("ok")),
            "attempt_count": int(res.get("attempt_count", 1) or 1),
            "usage_unknown_attempt_count": unk_cnt,
            "cost_accounting_status": str(res.get("cost_accounting_status") or "complete"),
            **sc,
        }
        with receipts_lock:
            all_receipts.append(rec_row)
        return res

    m6f.FrontierCodexClient.call_structured = _wrapped_call_structured  # type: ignore[method-assign]

    base_v7_forest, base_v7_archive, _, _ = m7a.load_autogenous_forest_bundle(args.v7_bundle)
    source_forest_nid = sorted(base_v7_forest.nodes.keys())[0]
    run_id = f"m98_run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    t_campaign0 = monotonic()

    print(f"[M9.8] Building 8 unexposed confirmation clusters (`commit={git_sha[:10]}`, `dirty={is_dirty}`)...", flush=True)
    built_worlds = {spec.world_id: build_m98_world_tasks(spec) for spec in CONFIRMATION_M98_WORLD_SPECS}

    # Phase 1: Parallel `Stage S` Shared Source Synthesis (`8` clusters)
    print("[M9.8] Phase 1: Running 8 shared Stage S source tasks in parallel...", flush=True)
    stage_s_results: dict[str, dict[str, Any]] = {}

    def _run_stage_s(spec: M98WorldSpec) -> tuple[str, dict[str, Any]]:
        wid = spec.world_id
        fam = spec.family
        s_task, _, _, _ = built_worlds[wid]
        uni_keys = expected_m98_public_output_keys(fam)
        s_base = out_dir / "stage_S_sources" / wid
        for att in range(1, 3):
            att_dir = s_base / f"attempt_{att:02d}" / s_task.task_id
            att_dir.mkdir(parents=True, exist_ok=True)
            ft_s = m98_task_to_frontier_task(s_task)
            _CALL_CONTEXT.role = "stage_S_solve"
            _CALL_CONTEXT.world_id = wid
            _CALL_CONTEXT.cluster_id = spec.cluster_id
            _CALL_CONTEXT.arm = "Stage_S_Shared_Source"
            _CALL_CONTEXT.episode_short_name = s_task.short_name
            f_copy = deepcopy(base_v7_forest)
            a_copy = deepcopy(base_v7_archive)
            a_copy["__autogenous_blank_seed__"] = {"enabled": True}
            ep_s = m6f.run_frontier_episode(
                ft_s,
                "Hegel_M6_9_Antithesis_LiveGPT",
                model=args.model,
                codex_info=codex_info,
                forest=f_copy,
                forest_archive=a_copy,
                hippo_idx=HippoRAGWormholeIndex(f_copy),
                episode_out_dir=att_dir,
                run_id=f"{run_id}__stageS__{wid}__att{att}",
                max_turns=args.max_turns,
                cmd_timeout_sec=180,
                reasoning_effort=args.reasoning_effort,
                defer_official_verifier=True,
                disclosure_mode="on_demand_executable",
                staging_policy="net_value_gated",
            )
            sub_s = _extract_submitted_files(att_dir)
            c0_txt = sub_s.get("batch_solver.py", "")
            pub_g = (
                execute_single_pre_llm_public_gate_probe_in_container(task=s_task, candidate_code=c0_txt)
                if c0_txt
                else {"accepted": False}
            )
            priv_s = (
                grade_m94_artifact_map(s_task, {str(s_task.declared_artifacts[0]): c0_txt})
                if c0_txt
                else {"strict_pass": False}
            )
            ep_s["strict_pass"] = bool(priv_s.get("strict_pass"))
            ep_s["public_accepted"] = bool(pub_g.get("accepted"))
            (att_dir / "00_episode_summary.json").write_text(json.dumps(ep_s, indent=2) + "\n", encoding="utf-8")
            (att_dir / "03_official_verifier_report.json").write_text(json.dumps(priv_s, indent=2) + "\n", encoding="utf-8")
            if pub_g.get("accepted") and priv_s.get("strict_pass"):
                # Freeze initial 2-layer graph checkpoint (`u0` + `d0`)
                fz_dir = out_dir / "world_freezes" / wid
                fz_dir.mkdir(parents=True, exist_ok=True)
                g0, _, reg0 = initialize_m98_factored_mechanism_graph(spec=spec, forest=deepcopy(base_v7_forest))
                u0_src = build_m98_default_u0_transducer_code(fam)
                u0_sha = hashlib.sha256(u0_src.encode("utf-8")).hexdigest()
                d0_sha = hashlib.sha256(c0_txt.encode("utf-8")).hexdigest()
                qu = reg0.issue_qualification_receipt(
                    qualification_protocol="hegel_public_scientific_gate_v4",
                    verified_code_sha256=u0_sha,
                    forest_node_id=source_forest_nid,
                    environment_digest=env_img_digest,
                )
                qd = reg0.issue_qualification_receipt(
                    qualification_protocol="hegel_public_scientific_gate_v4",
                    verified_code_sha256=d0_sha,
                    forest_node_id=source_forest_nid,
                    environment_digest=env_img_digest,
                )
                reg0.register_binding(
                    MechanismExecutableBinding(
                        binding_id=f"bind_{wid}_up_b00",
                        chain_id=f"{fam}_up",
                        branch_tag="U0_diagonal",
                        mechanism_id=f"mech_{wid}_up_b00",
                        mechanism_version=1,
                        forest_node_id=source_forest_nid,
                        component_id="u0_diagonal",
                        preserved_component_ids=[],
                        violated_component_id=None,
                        module_stem="u0",
                        module_digest=u0_sha,
                        function_name="calibrate_upstream",
                        qualification_id="hegel_public_scientific_gate_v4",
                        qualification_protocol="hegel_public_scientific_gate_v4",
                        qualification_receipt_id=qu.receipt_id,
                        qualification_status="verified",
                        parameter_mechanism_id="cal_node_p0",
                        parent_parameter_mechanism_ids=["cal_node_p0", "cal_node_p1"],
                        parent_parameter_versions={"cal_node_p0": 1, "cal_node_p1": 1},
                        parameter_version=1,
                        environment_digest=env_img_digest,
                        output_schema_keys=["calibrated_input"],
                        canonical_algorithm_code=u0_src,
                        canonical_entrypoint="calibrate_upstream",
                        selection_priority=10,
                    ),
                    strict_receipt_check=True,
                )
                reg0.register_binding(
                    MechanismExecutableBinding(
                        binding_id=f"bind_{wid}_dn_b00",
                        chain_id=f"{fam}_dn",
                        branch_tag="D0_base",
                        mechanism_id=f"mech_{wid}_dn_b00",
                        mechanism_version=1,
                        forest_node_id=source_forest_nid,
                        component_id="d0_base",
                        preserved_component_ids=[],
                        violated_component_id=None,
                        module_stem="d0",
                        module_digest=d0_sha,
                        function_name="solve_batch",
                        qualification_id="hegel_public_scientific_gate_v4",
                        qualification_protocol="hegel_public_scientific_gate_v4",
                        qualification_receipt_id=qd.receipt_id,
                        qualification_status="verified",
                        parameter_mechanism_id="constitutive_root_dn",
                        parent_parameter_mechanism_ids=["constitutive_root_dn"],
                        parent_parameter_versions={"constitutive_root_dn": 1},
                        parameter_version=1,
                        environment_digest=env_img_digest,
                        output_schema_keys=uni_keys,
                        canonical_algorithm_code=c0_txt,
                        canonical_entrypoint="solve_batch",
                        selection_priority=10,
                    ),
                    strict_receipt_check=True,
                )
                ckpt_p = fz_dir / "initial_factored_graph_ckpt.json"
                g0.save_checkpoint(ckpt_p)
                pub_s, pub_s_sha = materialize_m98_public_item_and_sha(s_task)
                src_ent = make_continual_archive_entry(
                    short_name=s_task.short_name,
                    task_id=s_task.task_id,
                    canonical_entrypoint="solve_batch",
                    canonical_algorithm_code=c0_txt,
                    output_schema_keys=uni_keys,
                    public_qualification_status="public_gate_verified",
                )
                print(f"  [M9.8 Stage S | {wid}] PASS on attempt {att} (`turns={ep_s.get('turns_used')}`)", flush=True)
                return wid, {
                    "ckpt_path": ckpt_p,
                    "source_entry": src_ent,
                    "source_pub_sha": pub_s_sha,
                    "source_up_obs": compute_m98_upstream_observables(fam, pub_s),
                    "stage_s_summary": ep_s,
                }
        raise RuntimeError(f"Stage S failed after 2 attempts on {wid}")

    with ThreadPoolExecutor(max_workers=8) as s_pool:
        s_futs = [s_pool.submit(_run_stage_s, sp) for sp in CONFIRMATION_M98_WORLD_SPECS]
        for sf in as_completed(s_futs):
            w_k, s_info = sf.result()
            stage_s_results[w_k] = s_info

    # Phase 2: Parallel `24` Continual Streams (`8` Clusters × `3` Online Arms) at `32`-Way Codex Concurrency!
    print("[M9.8] Phase 2: Launching 24 continual streams (8 clusters x 3 online arms) at 32-way Codex concurrency...", flush=True)
    stream_jobs = [(sp, arm_nm) for sp in CONFIRMATION_M98_WORLD_SPECS for arm_nm in M98_ARMS]
    streams_out: list[dict[str, Any]] = []

    def _exec_stream(job_tup: tuple[M98WorldSpec, str]) -> dict[str, Any]:
        sp_j, arm_j = job_tup
        s_t, st_ts, man_j, logs_j = built_worlds[sp_j.world_id]
        s_inf = stage_s_results[sp_j.world_id]
        st_dir = out_dir / "continual_streams" / sp_j.world_id / arm_j
        st_dir.mkdir(parents=True, exist_ok=True)
        return _run_one_m98_world_arm_stream(
            spec=sp_j,
            arm=arm_j,
            s_task=s_t,
            stream_tasks=st_ts,
            schedule_manifest=man_j,
            apparatus_logs=logs_j,
            base_v7_forest=base_v7_forest,
            base_v7_archive=base_v7_archive,
            base_graph_ckpt_path=s_inf["ckpt_path"],
            source_archive_entry=s_inf["source_entry"],
            source_forest_nid=source_forest_nid,
            source_pub_sha256=s_inf["source_pub_sha"],
            source_up_obs=s_inf["source_up_obs"],
            env_image_digest=env_img_digest,
            model=args.model,
            codex_info=codex_info,
            stream_out_dir=st_dir,
            run_id=run_id,
            max_turns=args.max_turns,
            reasoning_effort=args.reasoning_effort,
        )

    with ThreadPoolExecutor(max_workers=args.max_stream_workers) as st_pool:
        futs = [st_pool.submit(_exec_stream, jb) for jb in stream_jobs]
        for f in as_completed(futs):
            streams_out.append(f.result())
    streams_out.sort(key=lambda s: (s["world_id"], M98_ARMS.index(s["arm"])))

    # Phase 3: Executed `HistoryMasked_CurrentOnly` Ablation Control (`A03`, `0` LLM calls)
    print("[M9.8] Phase 3: Running executed HistoryMasked_CurrentOnly Docker ablation on byte-identical (R0, U_iso)...", flush=True)
    masked_ctrl = run_m98_history_masked_ablation_on_sealed_streams(built_worlds=built_worlds, out_dir=out_dir)
    (out_dir / "m9_8_history_masked_ablation.json").write_text(json.dumps(masked_ctrl, indent=2) + "\n", encoding="utf-8")

    shutil.rmtree(out_dir / "_codex_stage", ignore_errors=True)
    # Save raw call receipts & verify hierarchical `CostPolicy` conservation across all 3 online arms!
    (out_dir / "all_provider_call_receipts.json").write_text(json.dumps(all_receipts, indent=2) + "\n", encoding="utf-8")

    arm_ledgers = {}
    for arm_nm in M98_ARMS:
        arm_calls = [r for r in all_receipts if r["arm"] == arm_nm]
        ep_ledgers = {}
        ev_ledgers = {}
        cl_ledgers = {}
        for sp in CONFIRMATION_M98_WORLD_SPECS:
            _, _, man_sp, _ = built_worlds[sp.world_id]
            for m_sp in man_sp:
                sn_m = m_sp["short_name"]
                c_ep = [r for r in arm_calls if r["world_id"] == sp.world_id and r["episode_short_name"] == sn_m]
                ep_ledgers[f"{sp.world_id}:{sn_m}"] = aggregate_call_usages(c_ep, deduplicate_uuid=True)
        for ev_nm in ("C1_down", "R0", "C2_down", "C_up", "U_iso", "M_comb1", "M_comb2"):
            ev_sns = {
                m_sp["short_name"]
                for sp in CONFIRMATION_M98_WORLD_SPECS
                for m_sp in built_worlds[sp.world_id][2]
                if m_sp["causal_event"] == ev_nm
            }
            ev_ledgers[ev_nm] = aggregate_call_usages([r for r in arm_calls if r["episode_short_name"] in ev_sns], deduplicate_uuid=True)
        for sp in CONFIRMATION_M98_WORLD_SPECS:
            cl_ledgers[sp.world_id] = aggregate_call_usages([r for r in arm_calls if r["world_id"] == sp.world_id], deduplicate_uuid=True)
        arm_tot = aggregate_call_usages(arm_calls, deduplicate_uuid=True)
        assert_hierarchical_ledger_conservation(
            arm_ledger=arm_tot,
            episode_ledgers=ep_ledgers,
            event_ledgers=ev_ledgers,
            cluster_ledgers=cl_ledgers,
        )
        arm_ledgers[arm_nm] = {
            "scope1_in_stream_total": arm_tot,
            "solve_only": aggregate_call_usages([r for r in arm_calls if r["call_role"] == "stream_solve"], deduplicate_uuid=True),
            "guard_induction_only": aggregate_call_usages(
                [r for r in arm_calls if r["call_role"] == "autonomous_guard_induction"], deduplicate_uuid=True
            ),
            "by_event": ev_ledgers,
            "by_cluster": cl_ledgers,
        }

    stage_s_ledger = aggregate_call_usages([r for r in all_receipts if r["arm"] == "Stage_S_Shared_Source"], deduplicate_uuid=True)

    # Cluster-level paired comparisons (`Hegel_FactoredMechanismGraph` vs `Flat_ConditionalTable_Online` AND vs `Flat_LatestFirst`)
    diffs_vs_table = [
        arm_ledgers["Hegel_FactoredMechanismGraph"]["by_cluster"][sp.world_id]["proxy_v2_u_010c_4o"]
        - arm_ledgers["Flat_ConditionalTable_Online"]["by_cluster"][sp.world_id]["proxy_v2_u_010c_4o"]
        for sp in CONFIRMATION_M98_WORLD_SPECS
    ]
    diffs_vs_latest = [
        arm_ledgers["Hegel_FactoredMechanismGraph"]["by_cluster"][sp.world_id]["proxy_v2_u_010c_4o"]
        - arm_ledgers["Flat_LatestFirst"]["by_cluster"][sp.world_id]["proxy_v2_u_010c_4o"]
        for sp in CONFIRMATION_M98_WORLD_SPECS
    ]

    summary_payload = {
        "schema": "hegel_m9_8_blind_mismatch_factored_graph_summary_v1",
        "evidence_provenance": "executed",
        "run_id": run_id,
        "code_commit_at_start": git_sha,
        "working_tree_dirty_at_start": is_dirty,
        "source_tree_sha256": src_tree_sha,
        "sandbox_image_digest": env_img_digest,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "configured_max_codex_concurrency": args.max_codex_concurrency,
        "peak_active_codex_concurrency": active_counter["peak"],
        "configured_max_stream_workers": args.max_stream_workers,
        "cost_policy_version": COST_POLICY_VERSION,
        "shared_kernel_version": SHARED_KERNEL_VERSION,
        "total_wall_sec": round(monotonic() - t_campaign0, 2),
        "usage_completeness": (
            "complete" if sum(int(r["usage_unknown_attempt_count"]) for r in all_receipts) == 0 else "partial_unknown_usage"
        ),
        "total_unknown_usage_attempts": sum(int(r["usage_unknown_attempt_count"]) for r in all_receipts),
        "stage_S_shared_ledger": stage_s_ledger,
        "arm_cost_ledgers": arm_ledgers,
        "arm_outcome_totals": {
            arm_nm: {
                "episodes": sum(s["episode_count"] for s in streams_out if s["arm"] == arm_nm),
                "strict_passes": sum(s["strict_passes"] for s in streams_out if s["arm"] == arm_nm),
                "zero_llm_episodes": sum(s["zero_llm_episodes_total"] for s in streams_out if s["arm"] == arm_nm),
                "zero_turn_passed_closures": sum(s["zero_turn_passed_closures"] for s in streams_out if s["arm"] == arm_nm),
                "pre_llm_probe_wall_ms": round(sum(s["total_pre_llm_probe_wall_ms"] for s in streams_out if s["arm"] == arm_nm), 2),
                "coverage_probe_wall_ms": round(sum(s["total_coverage_probe_wall_ms"] for s in streams_out if s["arm"] == arm_nm), 2),
            }
            for arm_nm in M98_ARMS
        },
        "paired_cluster_inference_n8": {
            "Hegel_minus_Flat_ConditionalTable_Online_v2": {
                "cluster_diffs": diffs_vs_table,
                "mean_diff": round(float(np.mean(diffs_vs_table)), 2),
                "bootstrap_95ci": _bootstrap_mean_ci([float(x) for x in diffs_vs_table]),
                "two_sided_sign_flip_p": _exact_two_sided_sign_flip_p([float(x) for x in diffs_vs_table]),
                "alpha_n4_sign_flip_p": _exact_two_sided_sign_flip_p([float(x) for x in diffs_vs_table[:4]]),
                "beta_n4_sign_flip_p": _exact_two_sided_sign_flip_p([float(x) for x in diffs_vs_table[4:]]),
            },
            "Hegel_minus_Flat_LatestFirst_v2": {
                "cluster_diffs": diffs_vs_latest,
                "mean_diff": round(float(np.mean(diffs_vs_latest)), 2),
                "bootstrap_95ci": _bootstrap_mean_ci([float(x) for x in diffs_vs_latest]),
                "two_sided_sign_flip_p": _exact_two_sided_sign_flip_p([float(x) for x in diffs_vs_latest]),
                "alpha_n4_sign_flip_p": _exact_two_sided_sign_flip_p([float(x) for x in diffs_vs_latest[:4]]),
                "beta_n4_sign_flip_p": _exact_two_sided_sign_flip_p([float(x) for x in diffs_vs_latest[4:]]),
            },
        },
        "history_masked_ablation_summary": {
            "total_twin_episodes": masked_ctrl["total_twin_episodes"],
            "history_masked_strict_passes": masked_ctrl["history_masked_strict_passes"],
        },
        "streams": streams_out,
    }
    (out_dir / "m9_8_continual_summary.json").write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")

    sha_lines = []
    for fp in sorted(out_dir.rglob("*")):
        if fp.is_file() and fp.name != "SHA256SUMS.txt":
            sha_lines.append(f"{hashlib.sha256(fp.read_bytes()).hexdigest()}  {fp.relative_to(out_dir).as_posix()}")
    (out_dir / "SHA256SUMS.txt").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")
    print(f"[M9.8] COMPLETE! Sealed {len(sha_lines)} SHA256 entries in {out_dir}.", flush=True)


if __name__ == "__main__":
    main()
