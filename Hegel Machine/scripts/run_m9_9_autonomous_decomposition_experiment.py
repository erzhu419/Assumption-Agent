#!/usr/bin/env python3
"""Run or finalize the M9.9 equal-tool continual benchmark.

Fresh runs record their public recovery budget, formation and repair calls,
per-episode probe times, and paired source-cluster outcomes. Finalization
retains the measurements and configuration from the recorded execution.
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
import subprocess
import sys
import threading
from time import monotonic
from typing import Any
import uuid

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
from hegel_machine.cognition.m8_persistent_variational_planner import PersistentMechanismGraph
from hegel_machine.cognition.m9_4_1_shared_verification_kernel import SHARED_KERNEL_VERSION
from hegel_machine.cognition.m9_4_continual_mechanism_binding_world import (
    M94BindingRegistry,
    MechanismExecutableBinding,
    grade_m94_artifact_map,
)
from hegel_machine.cognition.m9_5_equal_tool_continual_attribution_world import (
    make_continual_archive_entry,
)
from hegel_machine.cognition.m9_6_autonomous_structure_continual_world import (
    EmpiricalCoverageReceipt,
    ImmutableBranchLineageController,
    classify_public_probe_failure,
    compute_canonical_numeric_item_sha256,
    evaluate_m96_autonomous_guard,
)
from hegel_machine.cognition.m9_8_blind_mismatch_and_selective_dependency_world import (
    assemble_m98_graph_diagonal_payload,
    build_m98_upstream_transducer_synthesis_task,
    compose_m98_two_stage_solver_code,
    compute_m98_combined_observables,
    compute_m98_downstream_observables,
    compute_m98_upstream_observables,
    diagnose_m98_factored_conflict_locus,
    expected_m98_public_output_keys,
    initialize_m98_factored_mechanism_graph,
    materialize_m98_public_item_and_sha,
    update_m98_calibration_snapshot_and_graph,
)
from hegel_machine.cognition.m9_9_autonomous_decomposition_and_interaction_withdrawal_world import (
    CONFIRMATION_M99_WORLD_SPECS,
    DEV_M99_WORLD_SPECS,
    execute_single_pre_llm_public_gate_probe_in_container,
    materialize_m99_public_item_and_sha,
    M99_ALL_OBSERVABLE_NAMES,
    M99_ARMS,
    M99_DOWNSTREAM_OBSERVABLE_NAMES,
    M99_UPSTREAM_OBSERVABLE_NAMES,
    M99WorldSpec,
    SeparabilityWithdrawalReceipt,
    build_m99_downstream_canonical_synthesis_task,
    build_m99_world_tasks,
    check_witness_feature_collision,
    issue_measured_withdrawal as _issue_measured_withdrawal,
    inspect_archived_withdrawal_lineage,
    hydrate_separability_withdrawals_from_graph,
    m99_task_to_frontier_task,
    persist_separability_withdrawals_to_graph,
    propose_and_validate_m99_guard_partition as propose_and_validate_m98_guard_partition,
    propose_and_verify_m99_autonomous_two_stage_factorization,
    update_m99_calibration_snapshot_and_graph,
)

DEFAULT_V7_BUNDLE = REPO_ROOT / "docs/evidence/m9_2_1_audit_closure_and_regrade_20261126/pretrained_autogenous_forest_v7_m921.json"
DEFAULT_OUT_DIR = REPO_ROOT / "docs/evidence/m9_9_autonomous_decomposition_and_interaction_withdrawal_20261128"

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


def _execution_fields(start: dict, end: dict, calls: list[dict]) -> dict:
    # Older bundles have a provider stage prefix but no execution clock or live tree manifest.
    ids = {row["stage_subdir"].split("__", 1)[0] for row in calls if row.get("stage_subdir")}
    observed_id = next(iter(ids)) if len(ids) == 1 else None
    if start.get("execution_run_id") and ids and ids != {start["execution_run_id"]}:
        raise ValueError("Provider call prefixes disagree with the execution manifest")
    run_id = start.get("execution_run_id") or observed_id
    if end.get("execution_run_id") and end["execution_run_id"] != run_id:
        raise ValueError("Completion belongs to a different execution")
    if len(ids) > 1:
        raise ValueError("Mixed provider runs in the same execution")
    return {
        "run_id": run_id,
        "wall_sec": end.get("execution_elapsed_wall_sec"),
        "execution_provenance": {**start, **end,
            "identity_source": "execution_manifest" if start.get("execution_run_id") else ("provider_call_prefix" if observed_id else None),
            "code_commit_at_start": start.get("code_commit_at_start"),
            "source_tree_sha256_at_start": start.get("source_tree_sha256_at_start"),
            "working_tree_dirty_at_start": start.get("working_tree_dirty_at_start")},
    }


def _extract_submitted_files(ep_dir: Path) -> dict[str, str]:
    sdir = ep_dir / "04_submitted_artifacts_snapshot"
    out: dict[str, str] = {}
    if not sdir.is_dir():
        return out
    for fp in sorted(sdir.rglob("*.py")):
        bname = fp.name.split("__", 1)[-1] if "__" in fp.name else fp.name
        out[bname] = fp.read_text(encoding="utf-8")
    return out


def _inject_equal_info_archive_into_m99_task(
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
        if ent.get("public_qualification_status") != "public_gate_verified":
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
        f"- Prior publicly-verified solvers: {[r['solver_path'] for r in manifest_rows]}.\n"
        "- Manifest: `/tmp/hegel_continual_archive/archive_manifest.json`.\n"
        "- Read-only (`0444`) prefix apparatus equilibration log (`<= t-1`, also mounted in private verifier): "
        "`/tmp/hegel_continual_archive/archive_apparatus_log.json` (when `item['campaign_diagnostics']` is `None` on a blind campaign, "
        "compare the latest entry of `archive_apparatus_log.json` against prior campaigns' entries to identify the active chamber setting!)."
    )
    if pre_llm_diagnostic is not None:
        header += f"\n- Pre-LLM Public Gate Diagnostic: `{json.dumps(pre_llm_diagnostic)}`."
    aug_task = type(task)(**{**task.__dict__, "public_setup_py": task.public_setup_py + "\n" + "\n".join(extra_lines) + "\n"})
    return aug_task, header


def _evaluate_empirical_coverage_m99(
    *,
    new_id: str,
    new_candidate_builder: Any,
    old_id: str,
    old_solver_sha: str,
    old_parent_vers: dict[str, int],
    old_witness_specs: list[tuple[str, Any, dict[str, Any], dict[str, str], dict[str, str]]],
) -> EmpiricalCoverageReceipt:
    t0 = monotonic()
    passed = 0
    has_counterexample = False
    w_ids, w_inp_shas, w_setup_shas, w_num_shas, dep_shas, p_ids, p_vers, p_pay_shas = [], [], [], [], [], [], [], []
    new_sha = ""
    for wit_id, w_task, w_pay, ro_deps, ro_shas in old_witness_specs:
        cand_c, raw_new_sha = new_candidate_builder(w_task, w_pay)
        new_sha = raw_new_sha
        w_item, w_pkl_sha = materialize_m99_public_item_and_sha(w_task)
        w_ids.append(wit_id)
        w_inp_shas.append(w_pkl_sha)
        w_setup_shas.append(hashlib.sha256(w_task.public_setup_py.encode("utf-8")).hexdigest())
        w_num_shas.append(compute_canonical_numeric_item_sha256(w_item))
        dep_shas.append(
            hashlib.sha256(json.dumps(ro_shas or {}, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        )
        p_ids.append(f"cprobe_{uuid.uuid4().hex[:16]}")
        p_vers.append(dict(old_parent_vers))
        p_pay_shas.append(hashlib.sha256(json.dumps(w_pay, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest())
        pr = execute_single_pre_llm_public_gate_probe_in_container(
            task=w_task, candidate_code=cand_c, readonly_dependencies=ro_deps, expected_dependency_sha256=ro_shas
        )
        if pr["accepted"]:
            passed += 1
        elif classify_public_probe_failure(pr)["tier"] != "INFRA_FAILURE":
            has_counterexample = True
    el_ms = round((monotonic() - t0) * 1000.0, 2)
    n_ev = len(old_witness_specs)
    status = "refuted" if has_counterexample else ("supported_on_witnesses" if n_ev and passed == n_ev else "unknown")
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
        receipt_schema_version=2,
        witness_setup_source_sha256s=w_setup_shas,
        witness_canonical_numeric_sha256s=w_num_shas,
        dependency_bundle_sha256s=dep_shas,
        probe_ids=p_ids,
    )
    rec.receipt_id = rec.compute_expected_receipt_id()
    rec.verify_integrity()
    return rec


def _guard_uses_upstream_features(guard: dict[str, Any] | None) -> bool:
    if not isinstance(guard, dict):
        return False
    for cl in guard.get("all_of") or []:
        if isinstance(cl, dict) and str(cl.get("feature_name", "")) in M99_UPSTREAM_OBSERVABLE_NAMES:
            return True
    return False


def _plain_table_select_row(
    table: list[dict[str, Any]], req_keys: list[str], obs: dict[str, float | None]
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    ordered = sorted(table, key=lambda r: (-int(r["priority"]), str(r["row_id"])))
    applicable = []
    evals = []
    any_unk = False
    for r in ordered:
        if sorted(r["output_schema_keys"]) != sorted(req_keys):
            evals.append({"row_id": r["row_id"], "guard_state": "SCHEMA_MISMATCH"})
            continue
        st = str(evaluate_m96_autonomous_guard(r.get("guard"), obs, allow_open_feature_keys=True)["state"])
        evals.append({"row_id": r["row_id"], "priority": r["priority"], "guard_state": st})
        if st == "TRUE":
            applicable.append(r)
        elif st == "UNKNOWN":
            any_unk = True
    if applicable:
        return applicable[0], ("selected_from_overlap_by_priority" if len(applicable) > 1 else "selected_unique_applicable"), evals
    return ordered[0], ("fallback_unknown" if any_unk else "fallback_no_applicable"), evals


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


def _bootstrap_mean_ci(diffs: list[float], seed: int = 99099, n_boot: int = 10000) -> list[float]:
    arr = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = np.mean(arr[idx], axis=1)
    return [round(float(np.percentile(means, 2.5)), 2), round(float(np.percentile(means, 97.5)), 2)]


def _register_learned_binding(
    *,
    registry: M94BindingRegistry,
    world_id: str,
    binding_id: str,
    chain_id: str,
    branch_tag: str,
    forest_node_id: str,
    code_txt: str,
    entrypoint: str,
    parent_ids: list[str],
    graph: PersistentMechanismGraph,
    output_schema_keys: list[str],
    applicability_guard: dict[str, Any],
    selection_priority: int,
    env_image_digest: str,
    structure_origin: str = "publicly_contrastively_validated",
) -> MechanismExecutableBinding:
    sha = hashlib.sha256(code_txt.encode("utf-8")).hexdigest()
    q = registry.issue_qualification_receipt(
        qualification_protocol="hegel_public_scientific_gate_v4",
        verified_code_sha256=sha,
        forest_node_id=forest_node_id,
        environment_digest=env_image_digest,
        issued_reason=f"learned_{branch_tag}",
    )
    p_vers = {pid: int(graph.mechanisms[pid].version) for pid in parent_ids}
    b = MechanismExecutableBinding(
        binding_id=binding_id,
        chain_id=chain_id,
        branch_tag=branch_tag,
        mechanism_id=f"mech_{binding_id}",
        mechanism_version=1,
        forest_node_id=forest_node_id,
        component_id=branch_tag,
        preserved_component_ids=[],
        violated_component_id=None,
        module_stem=branch_tag,
        module_digest=sha,
        function_name=entrypoint,
        qualification_id="hegel_public_scientific_gate_v4",
        qualification_protocol="hegel_public_scientific_gate_v4",
        qualification_receipt_id=q.receipt_id,
        qualification_status="verified",
        parameter_mechanism_id=parent_ids[0],
        parent_parameter_mechanism_ids=list(parent_ids),
        parent_parameter_versions=p_vers,
        parameter_version=p_vers[parent_ids[0]],
        structure_origin=structure_origin,
        environment_digest=env_image_digest,
        precondition_guard=deepcopy(applicability_guard),
        applicability_guard=deepcopy(applicability_guard),
        selection_priority=selection_priority,
        output_schema_keys=list(output_schema_keys),
        canonical_algorithm_code=code_txt,
        canonical_entrypoint=entrypoint,
        canonical_algorithm_sha256=sha,
        input_coordinate_contract="raw_measured",
        parameterized_closure_code=code_txt,
        parameterized_entrypoint=entrypoint,
        parameterized_closure_sha256=sha,
    )
    registry.register_binding(b, reason=f"register_{binding_id}", strict_receipt_check=True)
    return b


def _bounded_public_recovery(
    *, trial, family, task, public_item, numeric_sha, dependencies, dependency_shas, payload,
    primary_code, upstream_candidates, selected_up, compose_options, prior_entries, archive_context, schema,
):
    """Spend at most one alternate-component attempt and one unprojected archive candidate per episode."""
    if trial["accepted_pre_llm"] or classify_public_probe_failure(trial["probe_result"])["tier"] == "INFRA_FAILURE":
        return
    attempts = trial["recovery_attempts"] = []
    original_probe = trial["probe_result"]

    def joint(kind, source_id, code, context):
        target, item_sha, deps, shas, _ = context
        begin = monotonic()
        result = execute_single_pre_llm_public_gate_probe_in_container(
            task=target, candidate_code=code, readonly_dependencies=deps, expected_dependency_sha256=shas,
        )
        elapsed = round((monotonic() - begin) * 1000.0, 2)
        trial["pre_llm_probes_used"] += 1
        trial["probe_wall_ms"] += elapsed
        attempts.append({"kind": kind, "source_id": source_id, "target_task_id": target.task_id,
                         "numeric_input_sha256": item_sha, "candidate_sha256": hashlib.sha256(code.encode()).hexdigest(),
                         "wall_ms": elapsed, "probe_result": result})
        return result

    def accept(mode, result, code):
        trial.update(accepted_pre_llm=True, candidate_code=code, primary_probe_result=original_probe,
                     probe_result=result, recovery_mode=mode, failure_kind="none")

    current = (task, numeric_sha, dependencies, dependency_shas, payload)
    if trial["locus"] == "upstream_transducer" and upstream_candidates:
        row_id, code, ep, binding = upstream_candidates[0]
        direct = diagnose_m98_factored_conflict_locus(
            family, public_item, chosen_up_code=code, chosen_up_entrypoint=ep, diag_payload=payload,
            force_direct_component_probe=True,
        )
        trial["localization_probes_used"] += int(bool(direct.get("component_probe_executed")))
        trial["localization_probe_wall_ms"] += float(direct.get("probe_wall_ms") or 0)
        attempts.append({"kind": "alternative_upstream_direct", "source_id": row_id,
                         "numeric_input_sha256": numeric_sha, "candidate_sha256": hashlib.sha256(code.encode()).hexdigest(),
                         "component_probe": direct})
        rmse = direct.get("chosen_up_holdout_rmse")
        if (direct.get("execution_status") == "completed" and direct.get("component_probe_executed")
                and direct.get("candidate_code_sha256") == hashlib.sha256(code.encode()).hexdigest()
                and direct.get("locus") == "downstream_constitutive" and rmse is not None
                and np.isfinite(float(rmse)) and 0 <= float(rmse) <= 0.03):
            composed = compose_m98_two_stage_solver_code(
                upstream_code=code, upstream_entrypoint=ep, downstream_code=trial["selected_downstream_code"],
                downstream_entrypoint=trial["selected_downstream_ep"], param_payload=payload,
                upstream_binding=binding, **compose_options,
            )
            result = joint("alternative_upstream_joint", row_id, composed, current)
            if result["accepted"]:
                accept("alternative_upstream_joint", result, composed)
                trial.update(primary_upstream_id=trial["selected_upstream_id"], selected_upstream_id=row_id,
                             selected_upstream_code=code, selected_upstream_ep=ep)
                return
            if classify_public_probe_failure(result)["tier"] == "INFRA_FAILURE":
                return

    # Rank candidates by a public rule. Prefer an as-yet unrepresented source over
    # rechecking a monolith whose decomposition is already available.
    eligible = [r for r in reversed(prior_entries)
                if r.get("module_projection_status") in {"unprojected", "verified_on_source_and_target", "decomposed"}
                and r["public_qualification_status"] in {"public_gate_verified", "public_gate_failed"}
                and r["input_coordinate_contract"] == "raw_measured"
                and r["canonical_entrypoint"] == "solve_batch"
                and set(r["output_schema_keys"]) == set(schema)
                and (r["canonical_algorithm_code"] != primary_code or (
                    selected_up and r["module_projection_status"] != "decomposed"
                    and trial.get("conflict_locus_diagnosis", {}).get("locus") == "downstream_constitutive"))]
    entry = min(eligible, key=lambda r: (r["module_projection_status"] == "decomposed",
                                        r["public_qualification_status"] != "public_gate_verified"), default=None)
    if entry is None:
        trial["archive_recovery_status"] = "missing_eligible_public_solver"
        return
    source_id, archived = entry["task_id"], entry["canonical_algorithm_code"]
    if hashlib.sha256(archived.encode()).hexdigest() != entry["canonical_sha256"]:
        raise ValueError("Archived raw solver SHA-256 does not match its qualified identity")
    trial["archive_source_public_status"] = entry["public_qualification_status"]
    trial["archive_recovery_status"] = "unknown_projection"
    # A prior raw solver is only a hypothesis for the canonical downstream interface.
    # Qualify its composition on its original witness as well as the current target.
    source = archive_context.get(source_id)
    direct = trial.get("conflict_locus_diagnosis", {})
    can_project = bool(entry["module_projection_status"] != "decomposed" and source and source[1] != numeric_sha and selected_up and
                       direct.get("component_probe_executed") and direct.get("execution_status") == "completed"
                       and direct.get("locus") == "downstream_constitutive")

    def projected(pay):
        up_code, up_ep, up_binding = selected_up
        return compose_m98_two_stage_solver_code(
            upstream_code=up_code, upstream_entrypoint=up_ep, downstream_code=archived,
            downstream_entrypoint="solve_batch", param_payload=pay,
            upstream_binding=up_binding, graph=compose_options.get("graph"), registry=compose_options.get("registry"),
        )

    cached = entry.get("downstream_projection")
    prior_code = projected(source[4]) if can_project else None
    source_candidate_sha = hashlib.sha256(prior_code.encode()).hexdigest() if prior_code else None
    qualified_source = bool(can_project and cached and cached["source_candidate_sha256"] == source_candidate_sha
                            and cached["source_numeric_sha256"] == source[1]
                            and cached["source_dependency_sha256"] == source[3])
    if qualified_source:
        proposal = projected(payload)
        result = joint("archive_projection_target", source_id, proposal, current)
        if result["accepted"]:
            accept("archive_projection_target", result, proposal)
            trial["archive_recovery_status"] = "verified_on_source_and_target"
            trial["projection_source_receipt"] = cached
            return
        if classify_public_probe_failure(result)["tier"] == "INFRA_FAILURE":
            return
    if archived != primary_code:
        result = joint("archive_raw", source_id, archived, current)
        if result["accepted"]:
            accept("archive_raw", result, archived)
            trial["archive_recovery_status"] = "verified_raw_fallback"
            return
        if classify_public_probe_failure(result)["tier"] == "INFRA_FAILURE":
            return
    if not can_project or qualified_source:
        return
    old_result = joint("archive_projection_source", source_id, prior_code, source)
    if old_result["accepted"]:
        proposal = projected(payload)
        result = joint("archive_projection_target", source_id, proposal, current)
        if result["accepted"]:
            accept("archive_projection_target", result, proposal)
            trial["archive_recovery_status"] = entry["module_projection_status"] = "verified_on_source_and_target"
            trial["projection_source_receipt"] = entry["downstream_projection"] = {
                "source_task_id": source_id, "source_numeric_sha256": source[1],
                "source_candidate_sha256": source_candidate_sha, "source_dependency_sha256": dict(source[3]),
                "downstream_source_sha256": entry["canonical_sha256"], "downstream_entrypoint": "solve_batch",
                "qualifying_target_task_id": task.task_id, "source_gate_result": old_result,
            }


def _run_one_m99_world_arm_stream(
    *,
    spec: M99WorldSpec,
    arm: str,
    s_task: Any,
    stream_tasks: tuple[Any, ...],
    schedule_manifest: list[dict[str, Any]],
    apparatus_logs: list[dict[str, Any]],
    base_v7_forest: Any,
    base_v7_archive: dict[str, dict[str, Any]],
    source_monolithic_code: str,
    source_archive_entry: dict[str, Any],
    source_forest_nid: str,
    source_pub_item: dict[str, Any],
    source_pub_sha256: str,
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
    compat_spec = spec.to_m98_compat_spec()
    uni_keys = expected_m98_public_output_keys(fam)

    forest = deepcopy(base_v7_forest)
    archive = deepcopy(base_v7_archive)
    archive["__autogenous_blank_seed__"] = {"enabled": True}
    prior_entries: list[dict[str, Any]] = [deepcopy(source_archive_entry)]
    prior_entries[0]["module_projection_status"] = "unprojected"
    archive_context: dict[str, tuple] = {}
    flat_snap: dict[str, Any] = {"gains": [1.0, 1.0, 1.0, 1.0], "offsets": [0.0, 0.0, 0.0, 0.0]}

    # `Stage S` genuine measured observables (`apparatus_logs[:1]` was visible before `Stage S`!)
    s0_prefix_logs = apparatus_logs[:1]
    s0_ro_deps = {"/tmp/hegel_continual_archive/archive_apparatus_log.json": json.dumps(s0_prefix_logs, indent=2)}
    s0_ro_shas = {k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in s0_ro_deps.items()}
    s0_up_obs = compute_m98_upstream_observables(fam, source_pub_item)
    s0_dn_obs = compute_m98_downstream_observables(s0_prefix_logs)
    s0_comb_obs = {**s0_up_obs, **s0_dn_obs}
    s0_sha = hashlib.sha256(source_monolithic_code.encode("utf-8")).hexdigest()
    s0_canon_num_sha = compute_canonical_numeric_item_sha256(source_pub_item)

    archive_context[s_task.task_id] = (s_task, s0_canon_num_sha, s0_ro_deps, s0_ro_shas, deepcopy(flat_snap))

    # Arm 1: `Flat_MonolithicConditionalTable_Online`
    mono_table: list[dict[str, Any]] = [
        {
            "row_id": "mono_r00_S",
            "guard": {},
            "canonical_code": source_monolithic_code,
            "entrypoint": "solve_batch",
            "code_sha256": s0_sha,
            "priority": 10,
            "output_schema_keys": list(uni_keys),
        }
    ]
    mono_pos_obs: dict[str, list[dict[str, float | None]]] = {"mono_r00_S": [s0_comb_obs]}
    mono_neg_obs: dict[str, list[dict[str, float | None]]] = {}
    mono_cov_pairs: set[tuple[str, str]] = set()
    mono_rep_wit: dict[str, list[tuple[str, Any, dict[str, Any], dict[str, str], dict[str, str]]]] = {
        "mono_r00_S": [("wit_S", s_task, {"gains": [1.0] * 4, "offsets": [0.0] * 4}, s0_ro_deps, s0_ro_shas)]
    }

    # Arm 2: `Flat_PlainModuleRegistry_Online` (starts UNFACTORED!)
    plain_is_factored = False
    plain_up_table: list[dict[str, Any]] = []
    plain_dn_table: list[dict[str, Any]] = []
    plain_up_pos: dict[str, list[dict[str, float | None]]] = {}
    plain_up_neg: dict[str, list[dict[str, float | None]]] = {}
    plain_dn_pos: dict[str, list[dict[str, float | None]]] = {}
    plain_dn_neg: dict[str, list[dict[str, float | None]]] = {}
    plain_dn_cov_pairs: set[tuple[str, str]] = set()
    plain_dn_rep_wit: dict[str, list[tuple[str, Any, dict[str, Any], dict[str, str], dict[str, str]]]] = {}

    # Arm 3: `Hegel_AutonomousInteractionGraph_Online` (starts UNFACTORED!)
    hegel_is_factored = False
    graph: PersistentMechanismGraph | None = None
    registry: M94BindingRegistry | None = None
    up_ctrl: ImmutableBranchLineageController | None = None
    dn_ctrl: ImmutableBranchLineageController | None = None
    withdrawal_receipts: list[SeparabilityWithdrawalReceipt] = []
    hegel_cross_scoped_dn_bids: set[str] = set()
    hegel_dn_joint_pos_obs: dict[str, list[dict[str, float | None]]] = {}
    hegel_dn_joint_neg_obs: dict[str, list[dict[str, float | None]]] = {}
    hegel_dn_rep_wit: dict[str, list[tuple[str, Any, dict[str, Any], dict[str, str], dict[str, str]]]] = {}

    if arm == "Hegel_AutonomousInteractionGraph_Online":
        graph, _, registry = initialize_m98_factored_mechanism_graph(spec=compat_spec, forest=forest)
        up_ctrl = ImmutableBranchLineageController(world_id=f"{wid}_up", family=f"{fam}_up", graph=graph, registry=registry)
        dn_ctrl = ImmutableBranchLineageController(world_id=f"{wid}_dn", family=f"{fam}_dn", graph=graph, registry=registry)
        graph.save_checkpoint(stream_out_dir / "graph_ckpt_after_step_00.json")

    episode_results: list[dict[str, Any]] = []
    coverage_receipts_log: list[dict[str, Any]] = []
    guard_induction_events: list[dict[str, Any]] = []
    decomposition_events: list[dict[str, Any]] = []

    n_steps = len(stream_tasks)
    for step_i, (task, m_info) in enumerate(zip(stream_tasks, schedule_manifest), start=1):
        ep_dir = stream_out_dir / task.task_id
        ep_dir.mkdir(parents=True, exist_ok=True)
        ep_run_tag = f"{run_id}__{wid}__{arm}__step{step_i:02d}"

        prefix_logs = apparatus_logs[: step_i + 1]
        readonly_deps = {"/tmp/hegel_continual_archive/archive_apparatus_log.json": json.dumps(prefix_logs, indent=2)}
        readonly_shas = {k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in readonly_deps.items()}

        input_start = monotonic()
        pub_item, pub_sha = materialize_m99_public_item_and_sha(task)
        input_wall_ms = round((monotonic() - input_start) * 1000, 2)
        canon_num_sha = compute_canonical_numeric_item_sha256(pub_item)
        setup_src_sha = hashlib.sha256(task.public_setup_py.encode("utf-8")).hexdigest()
        dep_bundle_sha = hashlib.sha256(json.dumps(readonly_shas, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

        cal_upd = update_m99_calibration_snapshot_and_graph(
            spec=compat_spec,
            episode_id=f"{wid}_{arm}_{task.short_name}",
            public_item=pub_item,
            flat_snapshot=flat_snap,
            graph=graph,
            registry=registry,
        )
        flat_snap = cal_upd["flat_snapshot"]
        up_obs_now = compute_m98_upstream_observables(fam, pub_item)
        dn_obs_now = compute_m98_downstream_observables(prefix_logs)
        comb_obs_now = {**up_obs_now, **dn_obs_now}

        # =====================================================================
        # Step A: Equal-Capability Pre-LLM Public Gate Probe (`max_pre_llm_probes = 1`)
        # =====================================================================
        loc_wall_ms = 0.0
        loc_probes_used = 0
        decomp_wall_ms = 0.0
        decomp_probes_used = 0
        withdrawal_this_step: dict[str, Any] | None = None

        if arm == "Flat_MonolithicConditionalTable_Online":
            chosen_row, sel_mode, _ = _plain_table_select_row(mono_table, uni_keys, comb_obs_now)
            cand_code = str(chosen_row["canonical_code"])
            t_p0 = monotonic()
            pr = execute_single_pre_llm_public_gate_probe_in_container(
                task=task, candidate_code=cand_code, readonly_dependencies=readonly_deps, expected_dependency_sha256=readonly_shas
            )
            pr_ms = round((monotonic() - t_p0) * 1000.0, 2)
            cls = classify_public_probe_failure(pr)
            if not pr["accepted"] and cls["tier"] != "INFRA_FAILURE":
                mono_neg_obs.setdefault(chosen_row["row_id"], []).append(comb_obs_now)
            pre_trial: dict[str, Any] = {
                "arm": arm,
                "accepted_pre_llm": bool(pr["accepted"]),
                "pre_llm_probes_used": 1,
                "probe_wall_ms": pr_ms,
                "localization_probes_used": 0,
                "localization_probe_wall_ms": 0.0,
                "selection_mode": sel_mode,
                "selected_row_id": chosen_row["row_id"],
                "failure_kind": cls["failure_kind"],
                "candidate_code": (cand_code if pr["accepted"] else None),
                "probe_result": pr,
                "locus": "monolithic_task",
            }

        elif arm == "Flat_PlainModuleRegistry_Online":
            if not plain_is_factored:
                cand_code = source_monolithic_code
                t_p0 = monotonic()
                pr = execute_single_pre_llm_public_gate_probe_in_container(
                    task=task, candidate_code=cand_code, readonly_dependencies=readonly_deps, expected_dependency_sha256=readonly_shas
                )
                pr_ms = round((monotonic() - t_p0) * 1000.0, 2)
                cls = classify_public_probe_failure(pr)
                pre_trial = {
                    "arm": arm,
                    "accepted_pre_llm": bool(pr["accepted"]),
                    "pre_llm_probes_used": 1,
                    "probe_wall_ms": pr_ms,
                    "localization_probes_used": 0,
                    "localization_probe_wall_ms": 0.0,
                    "selection_mode": "pre_factorization_monolithic_s0",
                    "failure_kind": cls["failure_kind"],
                    "candidate_code": (cand_code if pr["accepted"] else None),
                    "probe_result": pr,
                    "locus": "pre_factorization_monolithic",
                }
            else:
                up_r, up_m, _ = _plain_table_select_row(plain_up_table, ["calibrated_input"], up_obs_now)
                dn_r, dn_m, _ = _plain_table_select_row(plain_dn_table, uni_keys, dn_obs_now)
                cand_code = compose_m98_two_stage_solver_code(
                    upstream_code=str(up_r["canonical_code"]),
                    upstream_entrypoint=str(up_r["entrypoint"]),
                    downstream_code=str(dn_r["canonical_code"]),
                    downstream_entrypoint=str(dn_r["entrypoint"]),
                    param_payload=flat_snap,
                )
                t_p0 = monotonic()
                pr = execute_single_pre_llm_public_gate_probe_in_container(
                    task=task, candidate_code=cand_code, readonly_dependencies=readonly_deps, expected_dependency_sha256=readonly_shas
                )
                pr_ms = round((monotonic() - t_p0) * 1000.0, 2)
                cls = classify_public_probe_failure(pr)
                locus_str = "none"
                locus_diag: dict[str, Any] = {}
                if pr["accepted"]:
                    plain_up_pos.setdefault(up_r["row_id"], []).append(up_obs_now)
                    plain_dn_pos.setdefault(dn_r["row_id"], []).append(dn_obs_now)
                elif cls["tier"] == "INFRA_FAILURE":
                    locus_str = "unknown_infra"
                else:
                    locus_diag = diagnose_m98_factored_conflict_locus(
                        fam,
                        pub_item,
                        chosen_up_code=str(up_r["canonical_code"]),
                        chosen_up_entrypoint=str(up_r["entrypoint"]),
                        diag_payload=flat_snap,
                        force_direct_component_probe=True,
                    )
                    loc_wall_ms = round(float(locus_diag.get("probe_wall_ms") or 0.0), 2)
                    loc_probes_used = 1 if locus_diag.get("component_probe_executed") else 0
                    locus_str = str(locus_diag["locus"])
                    if locus_str == "upstream_transducer":
                        plain_up_neg.setdefault(up_r["row_id"], []).append(up_obs_now)
                    elif locus_str == "downstream_constitutive":
                        plain_up_pos.setdefault(up_r["row_id"], []).append(up_obs_now)
                        plain_dn_neg.setdefault(dn_r["row_id"], []).append(dn_obs_now)
                pre_trial = {
                    "arm": arm,
                    "accepted_pre_llm": bool(pr["accepted"]),
                    "pre_llm_probes_used": 1,
                    "probe_wall_ms": pr_ms,
                    "localization_probes_used": loc_probes_used,
                    "localization_probe_wall_ms": loc_wall_ms,
                    "upstream_selection_mode": up_m,
                    "downstream_selection_mode": dn_m,
                    "selected_upstream_id": up_r["row_id"],
                    "selected_downstream_id": dn_r["row_id"],
                    "selected_upstream_code": up_r["canonical_code"],
                    "selected_upstream_ep": up_r["entrypoint"],
                    "selected_downstream_code": dn_r["canonical_code"],
                    "selected_downstream_ep": dn_r["entrypoint"],
                    "conflict_locus_diagnosis": locus_diag,
                    "failure_kind": cls["failure_kind"],
                    "candidate_code": (cand_code if pr["accepted"] else None),
                    "probe_result": pr,
                    "locus": locus_str,
                }

        else:
            # Arm 3: `Hegel_AutonomousInteractionGraph_Online`
            assert graph is not None and registry is not None and up_ctrl is not None and dn_ctrl is not None
            if not hegel_is_factored:
                cand_code = source_monolithic_code
                t_p0 = monotonic()
                pr = execute_single_pre_llm_public_gate_probe_in_container(
                    task=task, candidate_code=cand_code, readonly_dependencies=readonly_deps, expected_dependency_sha256=readonly_shas
                )
                pr_ms = round((monotonic() - t_p0) * 1000.0, 2)
                cls = classify_public_probe_failure(pr)
                pre_trial = {
                    "arm": arm,
                    "accepted_pre_llm": bool(pr["accepted"]),
                    "pre_llm_probes_used": 1,
                    "probe_wall_ms": pr_ms,
                    "localization_probes_used": 0,
                    "localization_probe_wall_ms": 0.0,
                    "selection_mode": "pre_factorization_monolithic_s0",
                    "failure_kind": cls["failure_kind"],
                    "candidate_code": (cand_code if pr["accepted"] else None),
                    "probe_result": pr,
                    "locus": "pre_factorization_monolithic",
                }
            else:
                up_sel = up_ctrl.select_branch_deterministically(req_schema_keys=["calibrated_input"], feature_evaluator=lambda _b: up_obs_now)
                chosen_up: MechanismExecutableBinding = up_sel["selected"] or sorted(
                    [b for b in registry.bindings.values() if b.chain_id == f"{fam}_up" and b.qualification_status == "verified"],
                    key=lambda x: (-int(x.selection_priority), x.binding_id),
                )[0]
                up_mode = str(up_sel["status"])

                dn_sel = dn_ctrl.select_branch_deterministically(
                    req_schema_keys=uni_keys,
                    feature_evaluator=lambda b: (
                        comb_obs_now
                        if (b.binding_id in hegel_cross_scoped_dn_bids or _guard_uses_upstream_features(b.applicability_guard))
                        else dn_obs_now
                    ),
                )
                chosen_dn: MechanismExecutableBinding = dn_sel["selected"] or sorted(
                    [b for b in registry.bindings.values() if b.chain_id == f"{fam}_dn" and b.qualification_status == "verified"],
                    key=lambda x: (-int(x.selection_priority), x.binding_id),
                )[0]
                dn_mode = str(dn_sel["status"])

                diag_pay = assemble_m98_graph_diagonal_payload(compat_spec, graph)
                cand_code = compose_m98_two_stage_solver_code(
                    upstream_code=(chosen_up.canonical_algorithm_code or chosen_up.parameterized_closure_code),
                    upstream_entrypoint=(chosen_up.canonical_entrypoint or chosen_up.parameterized_entrypoint),
                    downstream_code=(chosen_dn.canonical_algorithm_code or chosen_dn.parameterized_closure_code),
                    downstream_entrypoint=(chosen_dn.canonical_entrypoint or chosen_dn.parameterized_entrypoint),
                    param_payload=diag_pay,
                    upstream_binding=chosen_up,
                    downstream_binding=chosen_dn,
                    graph=graph,
                    registry=registry,
                )
                cand_sha = hashlib.sha256(cand_code.encode("utf-8")).hexdigest()
                t_p0 = monotonic()
                pr = execute_single_pre_llm_public_gate_probe_in_container(
                    task=task, candidate_code=cand_code, readonly_dependencies=readonly_deps, expected_dependency_sha256=readonly_shas
                )
                pr_ms = round((monotonic() - t_p0) * 1000.0, 2)
                cls = classify_public_probe_failure(pr)
                locus_str = "none"
                locus_diag = {}

                if pr["accepted"]:
                    up_ctrl.record_tested_witness(
                        binding=chosen_up,
                        public_input_sha256=pub_sha,
                        passed=True,
                        failure_kind="none",
                        features=up_obs_now,
                        probe_wall_ms=pr_ms,
                        execution_status="completed",
                        evidence_origin="joint_task_probe",
                        setup_source_sha256=setup_src_sha,
                        canonical_numeric_sha256=canon_num_sha,
                        dependency_bundle_sha256=dep_bundle_sha,
                    )
                    dn_ctrl.record_tested_witness(
                        binding=chosen_dn,
                        public_input_sha256=pub_sha,
                        passed=True,
                        failure_kind="none",
                        features=(comb_obs_now if chosen_dn.binding_id in hegel_cross_scoped_dn_bids else dn_obs_now),
                        probe_wall_ms=pr_ms,
                        execution_status="completed",
                        evidence_origin="joint_task_probe",
                        setup_source_sha256=setup_src_sha,
                        canonical_numeric_sha256=canon_num_sha,
                        dependency_bundle_sha256=dep_bundle_sha,
                    )
                    hegel_dn_joint_pos_obs.setdefault(chosen_dn.binding_id, []).append(comb_obs_now)
                elif cls["tier"] == "INFRA_FAILURE":
                    locus_str = "unknown_infra"
                else:
                    locus_diag = diagnose_m98_factored_conflict_locus(
                        fam,
                        pub_item,
                        chosen_up_code=(chosen_up.canonical_algorithm_code or chosen_up.parameterized_closure_code),
                        chosen_up_entrypoint=(chosen_up.canonical_entrypoint or chosen_up.parameterized_entrypoint),
                        diag_payload=diag_pay,
                        force_direct_component_probe=True,
                    )
                    loc_wall_ms = round(float(locus_diag.get("probe_wall_ms") or 0.0), 2)
                    loc_probes_used = 1 if locus_diag.get("component_probe_executed") else 0
                    locus_str = str(locus_diag["locus"])
                    if locus_str == "upstream_transducer" and locus_diag.get("execution_status") == "completed":
                        up_ctrl.record_tested_witness(
                            binding=chosen_up,
                            public_input_sha256=pub_sha,
                            passed=False,
                            failure_kind=str(locus_diag.get("failure_kind") or "scientific_or_numerical"),
                            features=up_obs_now,
                            probe_wall_ms=loc_wall_ms,
                            execution_status="completed",
                            evidence_origin="direct_component_probe",
                            metric_value=locus_diag.get("chosen_up_holdout_rmse"),
                            probe_id=str(locus_diag.get("probe_id") or ""),
                            setup_source_sha256=setup_src_sha,
                            canonical_numeric_sha256=canon_num_sha,
                            dependency_bundle_sha256=dep_bundle_sha,
                        )
                    elif locus_str == "downstream_constitutive" and locus_diag.get("execution_status") == "completed":
                        up_ctrl.record_tested_witness(
                            binding=chosen_up,
                            public_input_sha256=pub_sha,
                            passed=True,
                            failure_kind="none",
                            features=up_obs_now,
                            probe_wall_ms=loc_wall_ms,
                            execution_status="completed",
                            evidence_origin="direct_component_probe",
                            metric_value=locus_diag.get("chosen_up_holdout_rmse"),
                            probe_id=str(locus_diag.get("probe_id") or ""),
                            setup_source_sha256=setup_src_sha,
                            canonical_numeric_sha256=canon_num_sha,
                            dependency_bundle_sha256=dep_bundle_sha,
                        )
                        joint_probe_id = f"joint_{cand_sha}_{locus_diag['probe_id']}"
                        dn_ctrl.record_tested_witness(
                            binding=chosen_dn,
                            public_input_sha256=pub_sha,
                            probe_id=joint_probe_id,
                            passed=False,
                            failure_kind=cls["failure_kind"],
                            features=(comb_obs_now if chosen_dn.binding_id in hegel_cross_scoped_dn_bids else dn_obs_now),
                            probe_wall_ms=pr_ms,
                            execution_status="completed",
                            evidence_origin="joint_task_probe",
                            setup_source_sha256=setup_src_sha,
                            canonical_numeric_sha256=canon_num_sha,
                            dependency_bundle_sha256=dep_bundle_sha,
                        )
                        hegel_dn_joint_neg_obs.setdefault(chosen_dn.binding_id, []).append(comb_obs_now)

                        sw_rec = _issue_measured_withdrawal(
                            world_id=wid, family=fam, downstream_selected=dn_sel["selected"] is not None,
                            upstream=chosen_up, downstream=chosen_dn, downstream_features=dn_obs_now,
                            downstream_ledger=dn_ctrl.witness_ledger, direct=locus_diag, gate=pr, candidate_sha=cand_sha,
                        )
                        if sw_rec is not None:
                            locus_str = "cross_factor_interaction_withdrawal"
                            withdrawal_receipts.append(sw_rec)
                            persist_separability_withdrawals_to_graph(graph, wid, withdrawal_receipts)
                            hegel_cross_scoped_dn_bids.add(chosen_dn.binding_id)
                            withdrawal_this_step = sw_rec.to_dict()

                pre_trial = {
                    "arm": arm,
                    "accepted_pre_llm": bool(pr["accepted"]),
                    "pre_llm_probes_used": 1,
                    "probe_wall_ms": pr_ms,
                    "localization_probes_used": loc_probes_used,
                    "localization_probe_wall_ms": loc_wall_ms,
                    "upstream_selection_mode": up_mode,
                    "downstream_selection_mode": dn_mode,
                    "selected_upstream_id": chosen_up.binding_id,
                    "selected_downstream_id": chosen_dn.binding_id,
                    "selected_upstream_code": (chosen_up.canonical_algorithm_code or chosen_up.parameterized_closure_code),
                    "selected_upstream_ep": (chosen_up.canonical_entrypoint or chosen_up.parameterized_entrypoint),
                    "selected_downstream_code": (chosen_dn.canonical_algorithm_code or chosen_dn.parameterized_closure_code),
                    "selected_downstream_ep": (chosen_dn.canonical_entrypoint or chosen_dn.parameterized_entrypoint),
                    "conflict_locus_diagnosis": locus_diag,
                    "separability_withdrawal_receipt": withdrawal_this_step,
                    "failure_kind": cls["failure_kind"],
                    "candidate_code": (cand_code if pr["accepted"] else None),
                    "probe_result": pr,
                    "locus": locus_str,
                }

        # =====================================================================
        # Identical public recovery budget for the two modular arms. A flat solver can also use raw archive replay.
        options = {}
        alternatives = []
        selected_up = None
        pay = flat_snap
        if not pre_trial["accepted_pre_llm"] and "selected_upstream_id" in pre_trial:
            if arm == "Flat_PlainModuleRegistry_Online":
                for row in sorted(plain_up_table, key=lambda r: (-int(r["priority"]), r["row_id"])):
                    if row["row_id"] != up_r["row_id"] and set(row["output_schema_keys"]) == {"calibrated_input"}:
                        alternatives.append((row["row_id"], row["canonical_code"], row["entrypoint"], None))
                selected_up = (up_r["canonical_code"], up_r["entrypoint"], None)
            else:
                pay = diag_pay
                options = dict(downstream_binding=chosen_dn, graph=graph, registry=registry)
                for b in sorted(registry.bindings.values(), key=lambda x: (-int(x.selection_priority), x.binding_id)):
                    if (b.chain_id == chosen_up.chain_id and b.binding_id != chosen_up.binding_id and
                            b.qualification_status == "verified" and set(b.output_schema_keys) == {"calibrated_input"}):
                        alternatives.append((b.binding_id, b.canonical_algorithm_code or b.parameterized_closure_code,
                                             b.canonical_entrypoint or b.parameterized_entrypoint, b))
                selected_up = (pre_trial["selected_upstream_code"], pre_trial["selected_upstream_ep"], chosen_up)
        _bounded_public_recovery(
            trial=pre_trial, family=fam, task=task, public_item=pub_item, numeric_sha=canon_num_sha,
            dependencies=readonly_deps, dependency_shas=readonly_shas, payload=pay, primary_code=cand_code,
            upstream_candidates=alternatives, selected_up=selected_up, compose_options=options,
            prior_entries=prior_entries, archive_context=archive_context, schema=uni_keys,
        )
        loc_wall_ms = pre_trial["localization_probe_wall_ms"]
        loc_probes_used = pre_trial["localization_probes_used"]

        # Step B: Zero-Turn Closure OR Live LLM Synthesis (`100%` Equal Tools!)
        # =====================================================================
        if pre_trial.get("accepted_pre_llm"):
            cand_txt = str(pre_trial["candidate_code"])
            sdir = ep_dir / "04_submitted_artifacts_snapshot"
            sdir.mkdir(parents=True, exist_ok=True)
            (sdir / f"slot_00__{Path(task.declared_artifacts[0]).name}").write_text(cand_txt, encoding="utf-8")
            t_v0 = monotonic()
            priv_ver = grade_m94_artifact_map(
                task,
                {str(task.declared_artifacts[0]): cand_txt},
                readonly_dependencies=readonly_deps,
                expected_dependency_sha256=readonly_shas,
            )
            priv_ms = round((monotonic() - t_v0) * 1000.0, 2)
            ep_sum = {
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
                "execution_mode": "zero_llm_public_gate_verified_reuse",
                "turns_used": 0,
                "ended_by_finish": True,
                "strict_pass": bool(priv_ver.get("strict_pass")),
                "reward": 1.0 if priv_ver.get("strict_pass") else 0.0,
                "input_materialization_wall_ms": input_wall_ms,
                "pre_llm_probe_wall_ms": float(pre_trial.get("probe_wall_ms", 0.0)),
                "post_llm_probe_wall_ms": 0.0,
                "localization_probe_wall_ms": loc_wall_ms,
                "decomposition_probe_wall_ms": 0.0,
                "official_verifier_wall_ms": priv_ms,
                "readonly_dependencies_sha256": readonly_shas,
                "pre_llm_trial_receipt": {
                    k: v
                    for k, v in pre_trial.items()
                    if k not in {"candidate_code", "selected_upstream_code", "selected_downstream_code"}
                },
                "verifier_report": priv_ver,
            }
            (ep_dir / "00_episode_summary.json").write_text(json.dumps(ep_sum, indent=2) + "\n", encoding="utf-8")
            (ep_dir / "03_official_verifier_report.json").write_text(json.dumps(priv_ver, indent=2) + "\n", encoding="utf-8")
            if graph is not None:
                graph.save_checkpoint(stream_out_dir / f"graph_ckpt_after_step_{step_i:02d}.json")
            episode_results.append(ep_sum)
            print(
                f"  [M9.9 {wid} | {arm} | step {step_i}/{n_steps} {task.short_name} ({m_info['causal_event']})] "
                f"0-LLM REUSE | pass={ep_sum['strict_pass']}",
                flush=True,
            )
            continue

        # Equal-prompt construction across `Flat_PlainModuleRegistry_Online` and `Hegel_AutonomousInteractionGraph_Online` (`R02`!)
        is_modular_arm = arm in {"Flat_PlainModuleRegistry_Online", "Hegel_AutonomousInteractionGraph_Online"}
        is_factored_now = plain_is_factored if arm == "Flat_PlainModuleRegistry_Online" else hegel_is_factored
        repair_mode = "monolithic_solve"

        if is_modular_arm and is_factored_now and pre_trial["locus"] == "upstream_transducer":
            repair_mode = "upstream_module_solve"
            base_synth_task = build_m98_upstream_transducer_synthesis_task(
                base_task=task,
                family=fam,
                downstream_code=str(pre_trial["selected_downstream_code"]),
                downstream_entrypoint=str(pre_trial["selected_downstream_ep"]),
            )
        elif is_modular_arm and is_factored_now and pre_trial["locus"] in {"downstream_constitutive", "cross_factor_interaction_withdrawal"}:
            repair_mode = "downstream_module_solve"
            pay_for_dn = flat_snap if arm == "Flat_PlainModuleRegistry_Online" else assemble_m98_graph_diagonal_payload(compat_spec, graph)
            base_synth_task = build_m99_downstream_canonical_synthesis_task(
                base_task=task,
                family=fam,
                upstream_code=str(pre_trial["selected_upstream_code"]),
                upstream_entrypoint=str(pre_trial["selected_upstream_ep"]),
                param_payload=pay_for_dn,
            )
        else:
            base_synth_task = task

        diag_for_prompt = {
            "pre_llm_gate_accepted": False,
            "repair_target": repair_mode,
            "failure_kind": pre_trial.get("failure_kind"),
        }
        aug_task, ctx_hdr = _inject_equal_info_archive_into_m99_task(base_synth_task, prior_entries, prefix_logs, diag_for_prompt)
        ft = m99_task_to_frontier_task(
            aug_task,
            continual_context_block=ctx_hdr,
            allowed_readonly_dependencies=tuple(sorted(readonly_deps.keys())),
        )

        _CALL_CONTEXT.role = "stream_solve"
        _CALL_CONTEXT.world_id = wid
        _CALL_CONTEXT.cluster_id = spec.cluster_id
        _CALL_CONTEXT.arm = arm
        _CALL_CONTEXT.episode_short_name = task.short_name

        # STRICT `R03` EQUAL PERMISSIONS: `enable_auto_skill_packaging = False` AND `allow_l1_primitives = False` FOR ALL 3 ARMS!
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
            enable_auto_skill_packaging=False,
            allow_l1_primitives=False,
            disclosure_mode="on_demand_executable",
            staging_policy="net_value_gated",
        )

        sub_files = _extract_submitted_files(ep_dir)
        up_mod_txt = sub_files.get("upstream_transducer.py", "")
        dn_mod_txt = sub_files.get("downstream_solver.py", "")
        batch_mod_txt = sub_files.get("batch_solver.py", "")

        if repair_mode == "upstream_module_solve" and up_mod_txt:
            pay_u = flat_snap if arm == "Flat_PlainModuleRegistry_Online" else assemble_m98_graph_diagonal_payload(compat_spec, graph)
            graded_cand_txt = compose_m98_two_stage_solver_code(
                upstream_code=up_mod_txt,
                upstream_entrypoint="calibrate_upstream",
                downstream_code=str(pre_trial["selected_downstream_code"]),
                downstream_entrypoint=str(pre_trial["selected_downstream_ep"]),
                param_payload=pay_u,
            )
            archive_code_txt, archive_ep = up_mod_txt, "calibrate_upstream"
        elif repair_mode == "downstream_module_solve" and dn_mod_txt:
            pay_d = flat_snap if arm == "Flat_PlainModuleRegistry_Online" else assemble_m98_graph_diagonal_payload(compat_spec, graph)
            graded_cand_txt = compose_m98_two_stage_solver_code(
                upstream_code=str(pre_trial["selected_upstream_code"]),
                upstream_entrypoint=str(pre_trial["selected_upstream_ep"]),
                downstream_code=dn_mod_txt,
                downstream_entrypoint="solve_downstream",
                param_payload=pay_d,
            )
            archive_code_txt, archive_ep = dn_mod_txt, "solve_downstream"
        else:
            graded_cand_txt = batch_mod_txt
            archive_code_txt, archive_ep = batch_mod_txt, "solve_batch"

        t_pg = monotonic()
        pub_gate_post = (
            execute_single_pre_llm_public_gate_probe_in_container(
                task=task, candidate_code=graded_cand_txt, readonly_dependencies=readonly_deps, expected_dependency_sha256=readonly_shas
            )
            if graded_cand_txt
            else {"accepted": False}
        )
        post_pub_ms = round((monotonic() - t_pg) * 1000.0, 2)

        t_v0 = monotonic()
        priv_ver = (
            grade_m94_artifact_map(
                task,
                {str(task.declared_artifacts[0]): graded_cand_txt},
                readonly_dependencies=readonly_deps,
                expected_dependency_sha256=readonly_shas,
            )
            if graded_cand_txt
            else {"strict_pass": False, "error": "MISSING_CANDIDATE"}
        )
        priv_ms = round((monotonic() - t_v0) * 1000.0, 2)

        cov_this_step: list[dict[str, Any]] = []
        ind_receipt: dict[str, Any] | None = None
        decomp_receipt_summary: dict[str, Any] | None = None

        # =====================================================================
        # Step C: Online Consolidation (`Monolithic` vs `PlainModuleRegistry` vs `Hegel_Graph`)
        # =====================================================================
        if pub_gate_post.get("accepted") and graded_cand_txt:
            comp_client = m6f.FrontierCodexClient(
                codex_info=codex_info,
                model=model,
                reasoning_effort=reasoning_effort,
                timeout_seconds=300,
            )

            if arm == "Flat_MonolithicConditionalTable_Online":
                comp_client._hegel_call_meta = {
                    "role": "autonomous_guard_induction",
                    "world_id": wid,
                    "cluster_id": spec.cluster_id,
                    "arm": arm,
                    "episode_short_name": task.short_name,
                }
                new_rid = f"mono_r{step_i:02d}_{m_info['causal_event']}"
                new_sha = hashlib.sha256(batch_mod_txt.encode("utf-8")).hexdigest()
                all_cov = True
                for old_r in mono_table:
                    rep_w = mono_rep_wit.get(old_r["row_id"], [])[:1]
                    cov_r = _evaluate_empirical_coverage_m99(
                        new_id=new_rid,
                        new_candidate_builder=lambda _wt, _wp: (batch_mod_txt, new_sha),
                        old_id=old_r["row_id"],
                        old_solver_sha=old_r["code_sha256"],
                        old_parent_vers={"flat_cal": step_i},
                        old_witness_specs=rep_w,
                    )
                    cov_this_step.append(cov_r.to_dict())
                    coverage_receipts_log.append(cov_r.to_dict())
                    if cov_r.status == "supported_on_witnesses":
                        mono_cov_pairs.add((new_rid, old_r["row_id"]))
                    else:
                        all_cov = False
                if all_cov:
                    mono_table.append(
                        {
                            "row_id": new_rid,
                            "guard": {},
                            "canonical_code": batch_mod_txt,
                            "entrypoint": "solve_batch",
                            "code_sha256": new_sha,
                            "priority": 50 + step_i,
                            "output_schema_keys": list(uni_keys),
                        }
                    )
                else:
                    ex_ids = [r["row_id"] for r in mono_table]
                    pos_m = {rid: list(mono_pos_obs.get(rid, [])) for rid in ex_ids}
                    pos_m[new_rid] = [comb_obs_now]
                    if check_witness_feature_collision(pos_m, mono_cov_pairs)["has_collision"]:
                        ind_receipt = {"accepted": False, "attempts_used": 0, "skipped_infeasible_collision": True}
                        mono_table.append(
                            {
                                "row_id": new_rid,
                                "guard": deepcopy(mono_table[-1]["guard"]),
                                "canonical_code": batch_mod_txt,
                                "entrypoint": "solve_batch",
                                "code_sha256": new_sha,
                                "priority": 10 + step_i,
                                "output_schema_keys": list(uni_keys),
                            }
                        )
                    else:
                        ind_res = propose_and_validate_m98_guard_partition(
                            client=comp_client,
                            stage_subdir=f"{ep_run_tag}__mono_gind",
                            scope_label="monolithic_conditional_table_12feat",
                            feature_names=M99_ALL_OBSERVABLE_NAMES,
                            existing_branch_ids=ex_ids,
                            new_branch_id_placeholder=new_rid,
                            positive_witness_features_by_branch=pos_m,
                            negative_witness_features_by_branch=mono_neg_obs,
                            covered_branch_pairs=mono_cov_pairs,
                        )
                        ind_receipt = {k: v for k, v in ind_res.items() if k != "provider_call_receipts"}
                        guard_induction_events.append({"step": step_i, "event": m_info["causal_event"], "layer": "monolithic_table", **ind_receipt})
                        if ind_res.get("accepted"):
                            for r_obj in mono_table:
                                if r_obj["row_id"] in ind_res["existing_branch_guards"]:
                                    r_obj["guard"] = deepcopy(ind_res["existing_branch_guards"][r_obj["row_id"]])
                            mono_table.append(
                                {
                                    "row_id": new_rid,
                                    "guard": deepcopy(ind_res["new_branch_guard"]),
                                    "canonical_code": batch_mod_txt,
                                    "entrypoint": "solve_batch",
                                    "code_sha256": new_sha,
                                    "priority": 10 + step_i,
                                    "output_schema_keys": list(uni_keys),
                                }
                            )
                mono_pos_obs[new_rid] = [comb_obs_now]
                mono_rep_wit[new_rid] = [(f"wit_{task.short_name}", task, deepcopy(flat_snap), readonly_deps, readonly_shas)]

            elif arm == "Flat_PlainModuleRegistry_Online":
                if not plain_is_factored:
                    # Step 1 (`E1_down`): Autonomous two-stage factorization + initial `(d0, d1)` guard induction!
                    comp_client._hegel_call_meta = {
                        "role": "autonomous_decomposition_induction",
                        "world_id": wid,
                        "cluster_id": spec.cluster_id,
                        "arm": arm,
                        "episode_short_name": task.short_name,
                    }
                    dec_res = propose_and_verify_m99_autonomous_two_stage_factorization(
                        client=comp_client,
                        stage_subdir=f"{ep_run_tag}__plain_decomp",
                        family=fam,
                        s0_monolithic_code=source_monolithic_code,
                        e1_monolithic_code=batch_mod_txt,
                        s0_task=s_task,
                        s0_public_item=source_pub_item,
                        s0_pay={"gains": [1.0] * 4, "offsets": [0.0] * 4},
                        s0_ro_deps=s0_ro_deps, s0_ro_shas=s0_ro_shas,
                        e1_task=task,
                        e1_public_item=pub_item,
                        e1_pay=flat_snap,
                        e1_ro_deps=readonly_deps,
                        e1_ro_shas=readonly_shas,
                    )
                    decomp_wall_ms = float(dec_res.get("total_probe_wall_ms", 0.0))
                    decomp_probes_used = len(dec_res.get("verification_probes") or [])
                    decomp_receipt_summary = {
                        k: v
                        for k, v in dec_res.items()
                        if k not in {"provider_call_receipts", "upstream_code", "downstream_d0_code", "downstream_d1_code"}
                    }
                    decomposition_events.append({"step": step_i, "event": m_info["causal_event"], **decomp_receipt_summary})
                    if dec_res.get("accepted"):
                        u0_c = str(dec_res["upstream_code"])
                        d0_c = str(dec_res["downstream_d0_code"])
                        d1_c = str(dec_res["downstream_d1_code"])
                        plain_is_factored = True
                        plain_up_table.append(
                            {
                                "row_id": "up_r00_u0",
                                "guard": {},
                                "canonical_code": u0_c,
                                "entrypoint": "calibrate_upstream",
                                "code_sha256": hashlib.sha256(u0_c.encode()).hexdigest(),
                                "priority": 10,
                                "output_schema_keys": ["calibrated_input"],
                            }
                        )
                        plain_up_pos["up_r00_u0"] = [s0_up_obs, up_obs_now]

                        m22_p = dec_res.get("matrix_2x2") or {}
                        e1_d0_p = m22_p.get("e1_d0") or {}
                        s0_d1_p = m22_p.get("s0_d1") or {}
                        d0_pos_p = [s0_dn_obs] + ([dn_obs_now] if e1_d0_p.get("accepted") else [])
                        d1_pos_p = [dn_obs_now] + ([s0_dn_obs] if s0_d1_p.get("accepted") else [])
                        d0_neg_p = (
                            [dn_obs_now]
                            if (
                                e1_d0_p.get("execution_status") == "completed"
                                and not e1_d0_p.get("accepted")
                                and e1_d0_p.get("failure_kind") in {"scientific_or_numerical", "domain_precondition"}
                            )
                            else []
                        )
                        d1_neg_p = (
                            [s0_dn_obs]
                            if (
                                s0_d1_p.get("execution_status") == "completed"
                                and not s0_d1_p.get("accepted")
                                and s0_d1_p.get("failure_kind") in {"scientific_or_numerical", "domain_precondition"}
                            )
                            else []
                        )
                        if e1_d0_p.get("accepted"):
                            plain_dn_cov_pairs.add(("dn_r00_d0", "dn_r01_d1"))
                        if s0_d1_p.get("accepted"):
                            plain_dn_cov_pairs.add(("dn_r01_d1", "dn_r00_d0"))

                        if e1_d0_p.get("accepted") or s0_d1_p.get("accepted"):
                            g_d0, g_d1 = {}, {}
                            ind_receipt = {"accepted": True, "attempts_used": 0, "shared_or_covering_from_2x2": True}
                            guard_induction_events.append({"step": step_i, "event": m_info["causal_event"], "layer": "plain_downstream", **ind_receipt})
                        else:
                            comp_client._hegel_call_meta["role"] = "autonomous_guard_induction"
                            ind_res = propose_and_validate_m98_guard_partition(
                                client=comp_client,
                                stage_subdir=f"{ep_run_tag}__plain_d0d1_gind",
                                scope_label="plain_downstream_table_8feat",
                                feature_names=M99_DOWNSTREAM_OBSERVABLE_NAMES,
                                existing_branch_ids=["dn_r00_d0"],
                                new_branch_id_placeholder="dn_r01_d1",
                                positive_witness_features_by_branch={"dn_r00_d0": d0_pos_p, "dn_r01_d1": d1_pos_p},
                                negative_witness_features_by_branch={"dn_r00_d0": d0_neg_p, "dn_r01_d1": d1_neg_p},
                                covered_branch_pairs=plain_dn_cov_pairs,
                            )
                            ind_receipt = {k: v for k, v in ind_res.items() if k != "provider_call_receipts"}
                            guard_induction_events.append({"step": step_i, "event": m_info["causal_event"], "layer": "plain_downstream", **ind_receipt})
                            g_d0 = deepcopy(ind_res["existing_branch_guards"]["dn_r00_d0"]) if ind_res.get("accepted") else {}
                            g_d1 = deepcopy(ind_res["new_branch_guard"]) if ind_res.get("accepted") else {}
                        plain_dn_table.extend(
                            [
                                {
                                    "row_id": "dn_r00_d0",
                                    "guard": g_d0,
                                    "canonical_code": d0_c,
                                    "entrypoint": "solve_downstream",
                                    "code_sha256": hashlib.sha256(d0_c.encode()).hexdigest(),
                                    "priority": (20 if e1_d0_p.get("accepted") else 10),
                                    "output_schema_keys": list(uni_keys),
                                },
                                {
                                    "row_id": "dn_r01_d1",
                                    "guard": g_d1,
                                    "canonical_code": d1_c,
                                    "entrypoint": "solve_downstream",
                                    "code_sha256": hashlib.sha256(d1_c.encode()).hexdigest(),
                                    "priority": (21 if s0_d1_p.get("accepted") else 11),
                                    "output_schema_keys": list(uni_keys),
                                },
                            ]
                        )
                        plain_dn_pos["dn_r00_d0"] = d0_pos_p
                        plain_dn_neg["dn_r00_d0"] = d0_neg_p
                        plain_dn_pos["dn_r01_d1"] = d1_pos_p
                        plain_dn_neg["dn_r01_d1"] = d1_neg_p
                        plain_dn_rep_wit["dn_r00_d0"] = [("wit_S", s_task, {"gains": [1.0] * 4, "offsets": [0.0] * 4}, s0_ro_deps, s0_ro_shas)]
                        plain_dn_rep_wit["dn_r01_d1"] = [(f"wit_{task.short_name}", task, deepcopy(flat_snap), readonly_deps, readonly_shas)]

                elif repair_mode == "upstream_module_solve" and up_mod_txt:
                    comp_client._hegel_call_meta = {
                        "role": "autonomous_guard_induction",
                        "world_id": wid,
                        "cluster_id": spec.cluster_id,
                        "arm": arm,
                        "episode_short_name": task.short_name,
                    }
                    new_uid = f"up_r{step_i:02d}_u1"
                    ex_uids = [r["row_id"] for r in plain_up_table]
                    pos_u = {rid: list(plain_up_pos.get(rid, [])) for rid in ex_uids}
                    pos_u[new_uid] = [up_obs_now]
                    ind_res = propose_and_validate_m98_guard_partition(
                        client=comp_client,
                        stage_subdir=f"{ep_run_tag}__plain_up_gind",
                        scope_label="plain_upstream_table_4feat",
                        feature_names=M99_UPSTREAM_OBSERVABLE_NAMES,
                        existing_branch_ids=ex_uids,
                        new_branch_id_placeholder=new_uid,
                        positive_witness_features_by_branch=pos_u,
                        negative_witness_features_by_branch=plain_up_neg,
                    )
                    ind_receipt = {k: v for k, v in ind_res.items() if k != "provider_call_receipts"}
                    guard_induction_events.append({"step": step_i, "event": m_info["causal_event"], "layer": "plain_upstream", **ind_receipt})
                    if ind_res.get("accepted"):
                        for r_obj in plain_up_table:
                            if r_obj["row_id"] in ind_res["existing_branch_guards"]:
                                r_obj["guard"] = deepcopy(ind_res["existing_branch_guards"][r_obj["row_id"]])
                        plain_up_table.append(
                            {
                                "row_id": new_uid,
                                "guard": deepcopy(ind_res["new_branch_guard"]),
                                "canonical_code": up_mod_txt,
                                "entrypoint": "calibrate_upstream",
                                "code_sha256": hashlib.sha256(up_mod_txt.encode()).hexdigest(),
                                "priority": 20 + step_i,
                                "output_schema_keys": ["calibrated_input"],
                            }
                        )
                    plain_up_pos[new_uid] = [up_obs_now]

                elif repair_mode == "downstream_module_solve" and dn_mod_txt:
                    comp_client._hegel_call_meta = {
                        "role": "autonomous_guard_induction",
                        "world_id": wid,
                        "cluster_id": spec.cluster_id,
                        "arm": arm,
                        "episode_short_name": task.short_name,
                    }
                    new_did = f"dn_r{step_i:02d}_{m_info['causal_event']}"
                    new_dsha = hashlib.sha256(dn_mod_txt.encode("utf-8")).hexdigest()
                    u0_base_code = str(plain_up_table[0]["canonical_code"])
                    all_cov = True
                    for old_d in plain_dn_table:
                        rep_w = plain_dn_rep_wit.get(old_d["row_id"], [])[:1]
                        cov_r = _evaluate_empirical_coverage_m99(
                            new_id=new_did,
                            new_candidate_builder=lambda _wt, wp: (
                                compose_m98_two_stage_solver_code(
                                    upstream_code=u0_base_code,
                                    upstream_entrypoint="calibrate_upstream",
                                    downstream_code=dn_mod_txt,
                                    downstream_entrypoint="solve_downstream",
                                    param_payload=wp,
                                ),
                                new_dsha,
                            ),
                            old_id=old_d["row_id"],
                            old_solver_sha=old_d["code_sha256"],
                            old_parent_vers={"plain_dn": 1},
                            old_witness_specs=rep_w,
                        )
                        cov_this_step.append(cov_r.to_dict())
                        coverage_receipts_log.append(cov_r.to_dict())
                        if cov_r.status == "supported_on_witnesses":
                            plain_dn_cov_pairs.add((new_did, old_d["row_id"]))
                        else:
                            all_cov = False
                    if all_cov:
                        plain_dn_table.append(
                            {
                                "row_id": new_did,
                                "guard": {},
                                "canonical_code": dn_mod_txt,
                                "entrypoint": "solve_downstream",
                                "code_sha256": new_dsha,
                                "priority": 50 + step_i,
                                "output_schema_keys": list(uni_keys),
                            }
                        )
                    else:
                        ex_dids = [r["row_id"] for r in plain_dn_table]
                        pos_d = {rid: list(plain_dn_pos.get(rid, [])) for rid in ex_dids}
                        pos_d[new_did] = [dn_obs_now]
                        col_chk = check_witness_feature_collision(pos_d, plain_dn_cov_pairs)
                        if col_chk["has_collision"]:
                            # Factorized `dn_obs` collision (`(U0, D2)` vs `(U1, D2)`): `0` doomed guard calls wasted!
                            failed_d_row = next((r for r in plain_dn_table if r["row_id"] == pre_trial["selected_downstream_id"]), plain_dn_table[-1])
                            ind_receipt = {
                                "accepted": False,
                                "attempts_used": 0,
                                "skipped_infeasible_dn_obs_collision": True,
                                "colliding_pairs": col_chk["colliding_pairs"],
                            }
                            guard_induction_events.append({"step": step_i, "event": m_info["causal_event"], "layer": "plain_downstream", **ind_receipt})
                            plain_dn_table.append(
                                {
                                    "row_id": new_did,
                                    "guard": deepcopy(failed_d_row["guard"]),
                                    "canonical_code": dn_mod_txt,
                                    "entrypoint": "solve_downstream",
                                    "code_sha256": new_dsha,
                                    "priority": 10 + step_i,
                                    "output_schema_keys": list(uni_keys),
                                }
                            )
                        else:
                            ind_res = propose_and_validate_m98_guard_partition(
                                client=comp_client,
                                stage_subdir=f"{ep_run_tag}__plain_dn_gind",
                                scope_label="plain_downstream_table_8feat",
                                feature_names=M99_DOWNSTREAM_OBSERVABLE_NAMES,
                                existing_branch_ids=ex_dids,
                                new_branch_id_placeholder=new_did,
                                positive_witness_features_by_branch=pos_d,
                                negative_witness_features_by_branch=plain_dn_neg,
                                covered_branch_pairs=plain_dn_cov_pairs,
                            )
                            ind_receipt = {k: v for k, v in ind_res.items() if k != "provider_call_receipts"}
                            guard_induction_events.append({"step": step_i, "event": m_info["causal_event"], "layer": "plain_downstream", **ind_receipt})
                            if ind_res.get("accepted"):
                                for r_obj in plain_dn_table:
                                    if r_obj["row_id"] in ind_res["existing_branch_guards"]:
                                        r_obj["guard"] = deepcopy(ind_res["existing_branch_guards"][r_obj["row_id"]])
                                plain_dn_table.append(
                                    {
                                        "row_id": new_did,
                                        "guard": deepcopy(ind_res["new_branch_guard"]),
                                        "canonical_code": dn_mod_txt,
                                        "entrypoint": "solve_downstream",
                                        "code_sha256": new_dsha,
                                        "priority": 10 + step_i,
                                        "output_schema_keys": list(uni_keys),
                                    }
                                )
                    plain_dn_pos[new_did] = [dn_obs_now]
                    plain_dn_rep_wit[new_did] = [(f"wit_{task.short_name}", task, deepcopy(flat_snap), readonly_deps, readonly_shas)]

            elif arm == "Hegel_AutonomousInteractionGraph_Online" and graph is not None and registry is not None and up_ctrl is not None and dn_ctrl is not None:
                ev_sha = hashlib.sha256(json.dumps(pub_gate_post.get("scientific_probe_receipt") or {}, sort_keys=True).encode("utf-8")).hexdigest()
                if not hegel_is_factored:
                    # Step 1 (`E1_down`): IDENTICAL autonomous two-stage factorization + initial `(d0, d1)` guard induction!
                    comp_client._hegel_call_meta = {
                        "role": "autonomous_decomposition_induction",
                        "world_id": wid,
                        "cluster_id": spec.cluster_id,
                        "arm": arm,
                        "episode_short_name": task.short_name,
                    }
                    diag_pay_now = assemble_m98_graph_diagonal_payload(compat_spec, graph)
                    dec_res = propose_and_verify_m99_autonomous_two_stage_factorization(
                        client=comp_client,
                        stage_subdir=f"{ep_run_tag}__hegel_decomp",
                        family=fam,
                        s0_monolithic_code=source_monolithic_code,
                        e1_monolithic_code=batch_mod_txt,
                        s0_task=s_task,
                        s0_public_item=source_pub_item,
                        s0_pay={"gains": [1.0] * 4, "offsets": [0.0] * 4},
                        s0_ro_deps=s0_ro_deps, s0_ro_shas=s0_ro_shas,
                        e1_task=task,
                        e1_public_item=pub_item,
                        e1_pay=diag_pay_now,
                        e1_ro_deps=readonly_deps,
                        e1_ro_shas=readonly_shas,
                    )
                    decomp_wall_ms = float(dec_res.get("total_probe_wall_ms", 0.0))
                    decomp_probes_used = len(dec_res.get("verification_probes") or [])
                    decomp_receipt_summary = {
                        k: v
                        for k, v in dec_res.items()
                        if k not in {"provider_call_receipts", "upstream_code", "downstream_d0_code", "downstream_d1_code"}
                    }
                    decomposition_events.append({"step": step_i, "event": m_info["causal_event"], **decomp_receipt_summary})
                    if dec_res.get("accepted"):
                        u0_c = str(dec_res["upstream_code"])
                        d0_c = str(dec_res["downstream_d0_code"])
                        d1_c = str(dec_res["downstream_d1_code"])
                        hegel_is_factored = True

                        b_u0 = _register_learned_binding(
                            registry=registry,
                            world_id=wid,
                            binding_id=f"bind_{wid}_up_b00",
                            chain_id=f"{fam}_up",
                            branch_tag="U0_learned",
                            forest_node_id=source_forest_nid,
                            code_txt=u0_c,
                            entrypoint="calibrate_upstream",
                            parent_ids=["cal_node_p0", "cal_node_p1"],
                            graph=graph,
                            output_schema_keys=["calibrated_input"],
                            applicability_guard={},
                            selection_priority=10,
                            env_image_digest=env_image_digest,
                        )
                        m22_h = dec_res.get("matrix_2x2") or {}
                        up_pr_h = dec_res.get("upstream_probe_receipt") or {}
                        s0_d0_h = m22_h.get("s0_d0") or {}
                        e1_d1_h = m22_h.get("e1_d1") or {}
                        e1_d0_h = m22_h.get("e1_d0") or {}
                        s0_d1_h = m22_h.get("s0_d1") or {}

                        up_ctrl._branch_seq = 1
                        up_ctrl.record_tested_witness_from_probe_receipt(
                            binding=b_u0,
                            probe_receipt=s0_d0_h,
                            public_input_sha256=source_pub_sha256,
                            features=s0_up_obs,
                            evidence_origin="joint_task_probe",
                            canonical_numeric_sha256=s0_canon_num_sha,
                        )
                        up_ctrl.record_tested_witness_from_probe_receipt(
                            binding=b_u0,
                            probe_receipt=up_pr_h,
                            public_input_sha256=pub_sha,
                            features=up_obs_now,
                            evidence_origin="direct_component_probe",
                            canonical_numeric_sha256=canon_num_sha,
                        )

                        bid_d0 = f"bind_{wid}_dn_b00"
                        bid_d1 = f"bind_{wid}_dn_b01"
                        d0_pos_h = [s0_dn_obs] + ([dn_obs_now] if e1_d0_h.get("accepted") else [])
                        d1_pos_h = [dn_obs_now] + ([s0_dn_obs] if s0_d1_h.get("accepted") else [])
                        d0_neg_h = (
                            [dn_obs_now]
                            if (
                                e1_d0_h.get("execution_status") == "completed"
                                and not e1_d0_h.get("accepted")
                                and e1_d0_h.get("failure_kind") in {"scientific_or_numerical", "domain_precondition"}
                            )
                            else []
                        )
                        d1_neg_h = (
                            [s0_dn_obs]
                            if (
                                s0_d1_h.get("execution_status") == "completed"
                                and not s0_d1_h.get("accepted")
                                and s0_d1_h.get("failure_kind") in {"scientific_or_numerical", "domain_precondition"}
                            )
                            else []
                        )
                        cov_init_h: set[tuple[str, str]] = set()
                        if e1_d0_h.get("accepted"):
                            cov_init_h.add((bid_d0, bid_d1))
                        if s0_d1_h.get("accepted"):
                            cov_init_h.add((bid_d1, bid_d0))

                        if cov_init_h:
                            g_d0, g_d1 = {}, {}
                            ind_receipt = {"accepted": True, "attempts_used": 0, "shared_or_covering_from_2x2": True}
                            guard_induction_events.append({"step": step_i, "event": m_info["causal_event"], "layer": "hegel_downstream", **ind_receipt})
                        else:
                            comp_client._hegel_call_meta["role"] = "autonomous_guard_induction"
                            ind_res = propose_and_validate_m98_guard_partition(
                                client=comp_client,
                                stage_subdir=f"{ep_run_tag}__hegel_d0d1_gind",
                                scope_label="hegel_downstream_layer_8feat",
                                feature_names=M99_DOWNSTREAM_OBSERVABLE_NAMES,
                                existing_branch_ids=[bid_d0],
                                new_branch_id_placeholder=bid_d1,
                                positive_witness_features_by_branch={bid_d0: d0_pos_h, bid_d1: d1_pos_h},
                                negative_witness_features_by_branch={bid_d0: d0_neg_h, bid_d1: d1_neg_h},
                                covered_branch_pairs=cov_init_h,
                            )
                            ind_receipt = {k: v for k, v in ind_res.items() if k != "provider_call_receipts"}
                            guard_induction_events.append({"step": step_i, "event": m_info["causal_event"], "layer": "hegel_downstream", **ind_receipt})
                            g_d0 = deepcopy(ind_res["existing_branch_guards"][bid_d0]) if ind_res.get("accepted") else {}
                            g_d1 = deepcopy(ind_res["new_branch_guard"]) if ind_res.get("accepted") else {}

                        b_d0 = _register_learned_binding(
                            registry=registry,
                            world_id=wid,
                            binding_id=bid_d0,
                            chain_id=f"{fam}_dn",
                            branch_tag="D0_learned",
                            forest_node_id=source_forest_nid,
                            code_txt=d0_c,
                            entrypoint="solve_downstream",
                            parent_ids=["constitutive_root_dn"],
                            graph=graph,
                            output_schema_keys=uni_keys,
                            applicability_guard=g_d0,
                            selection_priority=(20 if e1_d0_h.get("accepted") else 10),
                            env_image_digest=env_image_digest,
                        )
                        b_d1 = _register_learned_binding(
                            registry=registry,
                            world_id=wid,
                            binding_id=bid_d1,
                            chain_id=f"{fam}_dn",
                            branch_tag="D1_learned",
                            forest_node_id=source_forest_nid,
                            code_txt=d1_c,
                            entrypoint="solve_downstream",
                            parent_ids=["constitutive_root_dn"],
                            graph=graph,
                            output_schema_keys=uni_keys,
                            applicability_guard=g_d1,
                            selection_priority=(21 if s0_d1_h.get("accepted") else 11),
                            env_image_digest=env_image_digest,
                        )
                        dn_ctrl._branch_seq = 2
                        dn_ctrl.record_tested_witness_from_probe_receipt(
                            binding=b_d0,
                            probe_receipt=s0_d0_h,
                            public_input_sha256=source_pub_sha256,
                            features=s0_dn_obs,
                            evidence_origin="joint_task_probe",
                            canonical_numeric_sha256=s0_canon_num_sha,
                        )
                        dn_ctrl.record_tested_witness_from_probe_receipt(
                            binding=b_d0,
                            probe_receipt=e1_d0_h,
                            public_input_sha256=pub_sha,
                            features=dn_obs_now,
                            evidence_origin="joint_task_probe",
                            canonical_numeric_sha256=canon_num_sha,
                        )
                        dn_ctrl.record_tested_witness_from_probe_receipt(
                            binding=b_d1,
                            probe_receipt=e1_d1_h,
                            public_input_sha256=pub_sha,
                            features=dn_obs_now,
                            evidence_origin="joint_task_probe",
                            canonical_numeric_sha256=canon_num_sha,
                        )
                        dn_ctrl.record_tested_witness_from_probe_receipt(
                            binding=b_d1,
                            probe_receipt=s0_d1_h,
                            public_input_sha256=source_pub_sha256,
                            features=s0_dn_obs,
                            evidence_origin="joint_task_probe",
                            canonical_numeric_sha256=s0_canon_num_sha,
                        )
                        hegel_dn_joint_pos_obs[bid_d0] = [s0_comb_obs] + ([comb_obs_now] if e1_d0_h.get("accepted") else [])
                        hegel_dn_joint_neg_obs[bid_d0] = ([comb_obs_now] if d0_neg_h else [])
                        hegel_dn_joint_pos_obs[bid_d1] = [comb_obs_now] + ([s0_comb_obs] if s0_d1_h.get("accepted") else [])
                        hegel_dn_joint_neg_obs[bid_d1] = ([s0_comb_obs] if d1_neg_h else [])
                        hegel_dn_rep_wit[bid_d0] = [("wit_S", s_task, {"gains": [1.0] * 4, "offsets": [0.0] * 4}, s0_ro_deps, s0_ro_shas)]
                        hegel_dn_rep_wit[bid_d1] = [(f"wit_{task.short_name}", task, deepcopy(diag_pay_now), readonly_deps, readonly_shas)]

                elif repair_mode == "upstream_module_solve" and up_mod_txt:
                    comp_client._hegel_call_meta = {
                        "role": "autonomous_guard_induction",
                        "world_id": wid,
                        "cluster_id": spec.cluster_id,
                        "arm": arm,
                        "episode_short_name": task.short_name,
                    }
                    failed_up_id = str(pre_trial["selected_upstream_id"])
                    ex_ups = sorted(
                        [b for b in registry.bindings.values() if b.chain_id == f"{fam}_up" and b.qualification_status == "verified"],
                        key=lambda x: x.binding_id,
                    )
                    ph_up = f"bind_{wid}_up_b{up_ctrl._branch_seq:02d}"
                    pos_up = {b.binding_id: [w.public_structural_features for w in up_ctrl.positive_witnesses_for(b.binding_id)] for b in ex_ups}
                    pos_up[ph_up] = [up_obs_now]
                    neg_up = {b.binding_id: [w.public_structural_features for w in up_ctrl.scientific_negative_witnesses_for(b.binding_id)] for b in ex_ups}
                    ind_res = propose_and_validate_m98_guard_partition(
                        client=comp_client,
                        stage_subdir=f"{ep_run_tag}__hegel_up_gind",
                        scope_label="hegel_upstream_layer_4feat",
                        feature_names=M99_UPSTREAM_OBSERVABLE_NAMES,
                        existing_branch_ids=[b.binding_id for b in ex_ups],
                        new_branch_id_placeholder=ph_up,
                        positive_witness_features_by_branch=pos_up,
                        negative_witness_features_by_branch=neg_up,
                    )
                    ind_receipt = {k: v for k, v in ind_res.items() if k != "provider_call_receipts"}
                    guard_induction_events.append({"step": step_i, "event": m_info["causal_event"], "layer": "hegel_upstream", **ind_receipt})
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
                            violated_component_id="upstream_transducer_shift",
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
                            binding=new_u,
                            public_input_sha256=pub_sha,
                            passed=True,
                            failure_kind="none",
                            features=up_obs_now,
                            execution_status="completed",
                            evidence_origin="joint_task_probe",
                            canonical_numeric_sha256=canon_num_sha,
                        )

                elif repair_mode == "downstream_module_solve" and dn_mod_txt:
                    comp_client._hegel_call_meta = {
                        "role": "autonomous_guard_induction",
                        "world_id": wid,
                        "cluster_id": spec.cluster_id,
                        "arm": arm,
                        "episode_short_name": task.short_name,
                    }
                    failed_dn_id = str(pre_trial["selected_downstream_id"])
                    ex_dns = sorted(
                        [b for b in registry.bindings.values() if b.chain_id == f"{fam}_dn" and b.qualification_status == "verified"],
                        key=lambda x: x.binding_id,
                    )
                    ph_dn = f"bind_{wid}_dn_b{dn_ctrl._branch_seq:02d}"
                    dn_raw_sha = hashlib.sha256(dn_mod_txt.encode("utf-8")).hexdigest()
                    u_cur_code = str(pre_trial["selected_upstream_code"])
                    all_cov_dn = True
                    for old_d in ex_dns:
                        rep_w = hegel_dn_rep_wit.get(old_d.binding_id, [])[:1]
                        cov_r = _evaluate_empirical_coverage_m99(
                            new_id=ph_dn,
                            new_candidate_builder=lambda _wt, wp: (
                                compose_m98_two_stage_solver_code(
                                    upstream_code=u_cur_code,
                                    upstream_entrypoint="calibrate_upstream",
                                    downstream_code=dn_mod_txt,
                                    downstream_entrypoint="solve_downstream",
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

                    is_cross_withdrawal = pre_trial["locus"] == "cross_factor_interaction_withdrawal"
                    if is_cross_withdrawal:
                        # Selective Cross-Factor Parent Edge Expansion: ONLY `{failed_dn_id, ph_dn}` receive `up_obs ∪ dn_obs`
                        # and multi-parent `["constitutive_root_dn", "cal_node_p0", "cal_node_p1"]`, while `d0, d1` stay factorized!
                        ind_res = propose_and_validate_m98_guard_partition(
                            client=comp_client,
                            stage_subdir=f"{ep_run_tag}__hegel_cross_withdrawal_gind",
                            scope_label="selective_cross_factor_interaction_12feat",
                            feature_names=M99_ALL_OBSERVABLE_NAMES,
                            existing_branch_ids=[b.binding_id for b in ex_dns],
                            new_branch_id_placeholder=ph_dn,
                            positive_witness_features_by_branch={
                                **{b.binding_id: list(hegel_dn_joint_pos_obs.get(b.binding_id, [])) for b in ex_dns},
                                ph_dn: [comb_obs_now],
                            },
                            negative_witness_features_by_branch={
                                b.binding_id: list(hegel_dn_joint_neg_obs.get(b.binding_id, [])) for b in ex_dns
                            },
                            covered_branch_pairs={(cr.new_binding_id, cr.old_binding_id) for cr in dn_ctrl.coverage_receipts if cr.status == "supported_on_witnesses"},
                        )
                        ind_receipt = {k: v for k, v in ind_res.items() if k != "provider_call_receipts"}
                        guard_induction_events.append(
                            {"step": step_i, "event": m_info["causal_event"], "layer": "hegel_selective_cross_interaction", **ind_receipt}
                        )
                        if ind_res.get("accepted"):
                            new_d = dn_ctrl.mint_immutable_conflict_branch(
                                failed_binding_id=failed_dn_id,
                                new_canonical_code=dn_mod_txt,
                                new_entrypoint="solve_downstream",
                                forest_node_id=source_forest_nid,
                                qualification_protocol="hegel_public_scientific_gate_v4",
                                output_schema_keys=uni_keys,
                                validated_parent_ids=["constitutive_root_dn", "cal_node_p0", "cal_node_p1"],
                                new_applicability_guard=dict(ind_res["new_branch_guard"]),
                                updated_failed_branch_applicability_guard=dict(ind_res["existing_branch_guards"][failed_dn_id]),
                                violated_component_id="cross_factor_separability_withdrawn",
                                preserved_component_ids=["upstream_transducer", "separable_d0_d1"],
                                selection_priority=15 + step_i,
                                environment_digest=env_image_digest,
                                evidence_digest=ev_sha,
                            )
                            hegel_cross_scoped_dn_bids.add(new_d.binding_id)
                            for ex_bid, ex_g in (ind_res.get("existing_branch_guards") or {}).items():
                                if ex_bid in registry.bindings:
                                    registry.bindings[ex_bid].applicability_guard = deepcopy(ex_g)
                                    registry.bindings[ex_bid].precondition_guard = deepcopy(ex_g)
                                    if _guard_uses_upstream_features(ex_g):
                                        hegel_cross_scoped_dn_bids.add(ex_bid)
                                        cur_pids = list(registry.bindings[ex_bid].parent_parameter_mechanism_ids or [])
                                        cur_pvers = dict(registry.bindings[ex_bid].parent_parameter_versions or {})
                                        for pid_extra in ("cal_node_p0", "cal_node_p1"):
                                            if pid_extra not in cur_pids:
                                                cur_pids.append(pid_extra)
                                            if pid_extra in graph.mechanisms:
                                                cur_pvers[pid_extra] = int(graph.mechanisms[pid_extra].version)
                                        registry.bindings[ex_bid].parent_parameter_mechanism_ids = cur_pids
                                        registry.bindings[ex_bid].parent_parameter_versions = cur_pvers
                                        registry.bindings[ex_bid].parameter_epoch_sha256 = hashlib.sha256(
                                            json.dumps(cur_pvers, sort_keys=True).encode("utf-8")
                                        ).hexdigest()
                                        m_id = registry.bindings[ex_bid].mechanism_id
                                        if m_id in graph.mechanisms:
                                            graph.mechanisms[m_id].parent_ids = tuple(cur_pids)
                            registry.sync_to_graph()
                            dn_ctrl.record_tested_witness(
                                binding=new_d,
                                public_input_sha256=pub_sha,
                                passed=True,
                                failure_kind="none",
                                features=comb_obs_now,
                                execution_status="completed",
                                evidence_origin="joint_task_probe",
                                canonical_numeric_sha256=canon_num_sha,
                            )
                            hegel_dn_joint_pos_obs[new_d.binding_id] = [comb_obs_now]
                            hegel_dn_rep_wit[new_d.binding_id] = [
                                (f"wit_{task.short_name}", task, assemble_m98_graph_diagonal_payload(compat_spec, graph), readonly_deps, readonly_shas)
                            ]
                    elif all_cov_dn:
                        new_d = dn_ctrl.mint_immutable_conflict_branch(
                            failed_binding_id=failed_dn_id,
                            new_canonical_code=dn_mod_txt,
                            new_entrypoint="solve_downstream",
                            forest_node_id=source_forest_nid,
                            qualification_protocol="hegel_public_scientific_gate_v4",
                            output_schema_keys=uni_keys,
                            validated_parent_ids=["constitutive_root_dn"],
                            new_applicability_guard={},
                            updated_failed_branch_applicability_guard=None,
                            violated_component_id="covering_downstream",
                            preserved_component_ids=["upstream_transducer"],
                            selection_priority=50 + step_i,
                            environment_digest=env_image_digest,
                            evidence_digest=ev_sha,
                        )
                        dn_ctrl.record_tested_witness(
                            binding=new_d,
                            public_input_sha256=pub_sha,
                            passed=True,
                            failure_kind="none",
                            features=dn_obs_now,
                            execution_status="completed",
                            evidence_origin="joint_task_probe",
                            canonical_numeric_sha256=canon_num_sha,
                        )
                        hegel_dn_joint_pos_obs[new_d.binding_id] = [comb_obs_now]
                        hegel_dn_rep_wit[new_d.binding_id] = [
                            (f"wit_{task.short_name}", task, assemble_m98_graph_diagonal_payload(compat_spec, graph), readonly_deps, readonly_shas)
                        ]
                    else:
                        pos_dn = {b.binding_id: [w.public_structural_features for w in dn_ctrl.positive_witnesses_for(b.binding_id)] for b in ex_dns}
                        pos_dn[ph_dn] = [dn_obs_now]
                        neg_dn = {b.binding_id: [w.public_structural_features for w in dn_ctrl.scientific_negative_witnesses_for(b.binding_id)] for b in ex_dns}
                        cov_pairs_dn = {(cr.new_binding_id, cr.old_binding_id) for cr in dn_ctrl.coverage_receipts if cr.status == "supported_on_witnesses"}
                        ind_res = propose_and_validate_m98_guard_partition(
                            client=comp_client,
                            stage_subdir=f"{ep_run_tag}__hegel_dn_gind",
                            scope_label="hegel_downstream_layer_8feat",
                            feature_names=M99_DOWNSTREAM_OBSERVABLE_NAMES,
                            existing_branch_ids=[b.binding_id for b in ex_dns],
                            new_branch_id_placeholder=ph_dn,
                            positive_witness_features_by_branch=pos_dn,
                            negative_witness_features_by_branch=neg_dn,
                            covered_branch_pairs=cov_pairs_dn,
                        )
                        ind_receipt = {k: v for k, v in ind_res.items() if k != "provider_call_receipts"}
                        guard_induction_events.append({"step": step_i, "event": m_info["causal_event"], "layer": "hegel_downstream", **ind_receipt})
                        if ind_res.get("accepted"):
                            new_d = dn_ctrl.mint_immutable_conflict_branch(
                                failed_binding_id=failed_dn_id,
                                new_canonical_code=dn_mod_txt,
                                new_entrypoint="solve_downstream",
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
                                binding=new_d,
                                public_input_sha256=pub_sha,
                                passed=True,
                                failure_kind="none",
                                features=dn_obs_now,
                                execution_status="completed",
                                evidence_origin="joint_task_probe",
                                canonical_numeric_sha256=canon_num_sha,
                            )
                            hegel_dn_joint_pos_obs[new_d.binding_id] = [comb_obs_now]
                            hegel_dn_rep_wit[new_d.binding_id] = [
                                (f"wit_{task.short_name}", task, assemble_m98_graph_diagonal_payload(compat_spec, graph), readonly_deps, readonly_shas)
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
        ep_sum["execution_mode"] = f"llm_{repair_mode}"
        ep_sum["strict_pass"] = bool(priv_ver.get("strict_pass"))
        ep_sum["reward"] = 1.0 if priv_ver.get("strict_pass") else 0.0
        ep_sum["input_materialization_wall_ms"] = input_wall_ms
        ep_sum["pre_llm_probe_wall_ms"] = float(pre_trial.get("probe_wall_ms", 0.0))
        ep_sum["post_llm_probe_wall_ms"] = post_pub_ms
        ep_sum["localization_probe_wall_ms"] = loc_wall_ms
        ep_sum["decomposition_probe_wall_ms"] = decomp_wall_ms
        ep_sum["official_verifier_wall_ms"] = priv_ms
        ep_sum["readonly_dependencies_sha256"] = readonly_shas
        ep_sum["public_qualification_status"] = "public_gate_verified" if pub_gate_post.get("accepted") else "public_gate_failed"
        ep_sum["calibration_update_receipt"] = cal_upd.get("graph_receipt")
        ep_sum["empirical_coverage_receipts"] = cov_this_step
        ep_sum["guard_induction_receipt"] = ind_receipt
        ep_sum["autonomous_decomposition_receipt"] = decomp_receipt_summary
        ep_sum["pre_llm_trial_receipt"] = {
            k: v
            for k, v in pre_trial.items()
            if k not in {"candidate_code", "selected_upstream_code", "selected_downstream_code"}
        }
        ep_sum["verifier_report"] = priv_ver
        (ep_dir / "00_episode_summary.json").write_text(json.dumps(ep_sum, indent=2) + "\n", encoding="utf-8")
        (ep_dir / "03_official_verifier_report.json").write_text(json.dumps(priv_ver, indent=2) + "\n", encoding="utf-8")

        if graded_cand_txt and repair_mode != "upstream_module_solve":
            prior_entries.append(
                make_continual_archive_entry(
                    short_name=task.short_name,
                    task_id=task.task_id,
                    canonical_entrypoint=archive_ep,
                    canonical_algorithm_code=archive_code_txt,
                    output_schema_keys=uni_keys,
                    public_qualification_status=("public_gate_verified" if pub_gate_post.get("accepted") else "public_gate_failed"),
                )
            )
            projected = bool(decomp_receipt_summary and decomp_receipt_summary.get("accepted"))
            prior_entries[-1]["module_projection_status"] = "decomposed" if projected else "unprojected"
            if projected:
                prior_entries[0]["module_projection_status"] = "decomposed"
            historical_payload = assemble_m98_graph_diagonal_payload(compat_spec, graph) if graph is not None else flat_snap
            archive_context[task.task_id] = (task, canon_num_sha, readonly_deps, readonly_shas, deepcopy(historical_payload))
        if graph is not None:
            graph.save_checkpoint(stream_out_dir / f"graph_ckpt_after_step_{step_i:02d}.json")
        episode_results.append(ep_sum)
        print(
            f"  [M9.9 {wid} | {arm} | step {step_i}/{n_steps} {task.short_name} ({m_info['causal_event']})] "
            f"{ep_sum['execution_mode']} | pass={ep_sum['strict_pass']} | turns={ep_sum.get('turns_used')} "
            f"| gind={ind_receipt.get('accepted') if ind_receipt else None}",
            flush=True,
        )

    st_sum = {
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
        "total_input_materialization_wall_ms": round(sum(r["input_materialization_wall_ms"] for r in episode_results), 2),
        "total_pre_llm_probe_wall_ms": round(sum(float(r.get("pre_llm_probe_wall_ms", 0.0) or 0.0) for r in episode_results), 2),
        "total_post_llm_probe_wall_ms": round(sum(float(r.get("post_llm_probe_wall_ms", 0.0) or 0.0) for r in episode_results), 2),
        "total_localization_probe_wall_ms": round(sum(float(r.get("localization_probe_wall_ms", 0.0) or 0.0) for r in episode_results), 2),
        "total_decomposition_probe_wall_ms": round(sum(float(r.get("decomposition_probe_wall_ms", 0.0) or 0.0) for r in episode_results), 2),
        "total_coverage_probe_wall_ms": round(sum(float(c["total_cpu_probe_wall_ms"]) for c in coverage_receipts_log), 2),
        "total_official_verifier_wall_ms": round(sum(float(r.get("official_verifier_wall_ms", 0.0) or 0.0) for r in episode_results), 2),
        "separability_withdrawal_receipts": [w.to_dict() for w in withdrawal_receipts],
        "autonomous_decomposition_events": decomposition_events,
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
                "localization_probe_wall_ms": float(r.get("localization_probe_wall_ms", 0.0) or 0.0),
                "decomposition_probe_wall_ms": float(r.get("decomposition_probe_wall_ms", 0.0) or 0.0),
            }
            for r in episode_results
        ],
    }
    (stream_out_dir / "00_stream_summary.json").write_text(json.dumps(st_sum, indent=2) + "\n", encoding="utf-8")
    return st_sum


def run_m99_factorized_ceiling_ablation_on_sealed_streams(
    *,
    world_specs: tuple[M99WorldSpec, ...],
    built_worlds: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    """Executed `0`-LLM Frozen-Library Constrained `dn`-Only vs Selective-Joint Routing Ablation (`Q10/Q13` scoped):
    Evaluates at frozen prefix checkpoint `graph_ckpt_after_step_05.json` (`0` future leakage!):
      1. `factorized_fixed_D2_separable`: `u_selected o d2_sep` (design-time ideal `50.0%` when both `d2_sep, d2_coupled` exist; abstains when no separable branch is identified).
      2. `factorized_fixed_D2_coupled`: `u_selected o d2_coupled` (design-time ideal `50.0%`).
      3. `hegel_selective_cross_factor_graph`: routes using learned `up + dn` guards on `{d2_sep, d2_coupled}` (design-time ideal `100.0%`; observed `12/16 = 75.0%` on M9.9 confirmation).
    """
    rows = []
    manifest_path = out_dir / "execution_manifest.json"
    run_manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    historical_protocol = ((out_dir / "m9_9_continual_summary.json").is_file()
                           and run_manifest.get("policy_public_input_source") != "sandbox_public_pickle")
    for spec in world_specs:
        wid = spec.world_id
        compat = spec.to_m98_compat_spec()
        _, tasks, manifest, logs = built_worlds[wid]
        e6_idx = next(i for i, m in enumerate(manifest) if m["causal_event"] == "E6_separable_revisit")
        e7_idx = next(i for i, m in enumerate(manifest) if m["causal_event"] == "E7_coupled_revisit")

        ck5 = out_dir / "continual_streams" / wid / "Hegel_AutonomousInteractionGraph_Online" / "graph_ckpt_after_step_05.json"
        g5_base = PersistentMechanismGraph.load_checkpoint(ck5)
        rg5_base = M94BindingRegistry(g5_base)
        sw_list = hydrate_separability_withdrawals_from_graph(g5_base, rg5_base, wid, quarantine_unverified=True)
        basis = "measured_host_rejection" if sw_list else "absent"
        if not sw_list:
            archived = inspect_archived_withdrawal_lineage(g5_base, rg5_base, wid)
            if archived and all(r.joint_gate_receipt is None and not r.joint_probe_id for r in archived):
                sw_list = archived
                basis = "historical_lineage_only"
            elif archived:
                basis = "pending_measured_host_rejection"
        use_historical_input = historical_protocol or basis == "historical_lineage_only"
        policy_input = "historical_host_reconstruction" if use_historical_input else "sandbox_public_pickle"
        materialize = materialize_m98_public_item_and_sha if use_historical_input else materialize_m99_public_item_and_sha
        dn_all = sorted(
            [b for b in rg5_base.bindings.values() if b.chain_id == f"{spec.family}_dn" and b.qualification_status == "verified"],
            key=lambda b: (-int(b.selection_priority), b.binding_id),
        )
        d2_cpl_bid = dn_all[0].binding_id if dn_all else None
        d2_sep_bid = sw_list[-1].downstream_binding_id if sw_list else None
        cross_bids = {d2_sep_bid, d2_cpl_bid} if sw_list else set()

        for ev_nm, s_idx in (("E6_separable_revisit", e6_idx), ("E7_coupled_revisit", e7_idx)):
            tk = tasks[s_idx]
            pref_logs = logs[: s_idx + 2]
            ro_deps = {"/tmp/hegel_continual_archive/archive_apparatus_log.json": json.dumps(pref_logs, indent=2)}
            ro_shas = {k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in ro_deps.items()}
            pub_item, pub_sha = materialize(tk)
            g_step = deepcopy(g5_base)
            rg_step = M94BindingRegistry(g_step)
            update_m99_calibration_snapshot_and_graph(
                spec=compat, episode_id=f"{wid}_{ev_nm}_abl", public_item=pub_item, graph=g_step, registry=rg_step
            )
            pay = assemble_m98_graph_diagonal_payload(compat, g_step)
            up_c = ImmutableBranchLineageController(world_id=f"{wid}_up", family=f"{spec.family}_up", graph=g_step, registry=rg_step)
            dn_c = ImmutableBranchLineageController(world_id=f"{wid}_dn", family=f"{spec.family}_dn", graph=g_step, registry=rg_step)

            up_obs = compute_m98_upstream_observables(spec.family, pub_item)
            dn_obs = compute_m98_downstream_observables(pref_logs)
            comb_obs = {**up_obs, **dn_obs}
            c_up = up_c.select_branch_deterministically(req_schema_keys=["calibrated_input"], feature_evaluator=lambda _b: up_obs)["selected"]
            c_dn_hg = dn_c.select_branch_deterministically(
                req_schema_keys=expected_m98_public_output_keys(spec.family),
                feature_evaluator=lambda b: (
                    comb_obs
                    if (b.binding_id in cross_bids or _guard_uses_upstream_features(b.applicability_guard))
                    else dn_obs
                ),
            )["selected"]

            for pol_nm, b_dn in (
                ("factorized_fixed_D2_separable", rg_step.bindings.get(d2_sep_bid)),
                ("factorized_fixed_D2_coupled", rg_step.bindings.get(d2_cpl_bid)),
                ("hegel_selective_cross_factor_graph", c_dn_hg),
            ):
                if c_up is None or b_dn is None:
                    rows.append({
                        "policy": pol_nm, "world_id": wid, "family": spec.family, "event": ev_nm,
                        "withdrawal_evidence_basis": basis, "policy_input_source": policy_input,
                        "prefix_checkpoint": ck5.name, "public_batch_pkl_sha256": pub_sha,
                        "downstream_observables": dn_obs,
                        "selected_upstream": c_up.binding_id if c_up else None,
                        "selected_downstream": b_dn.binding_id if b_dn else None,
                        "candidate_sha256": None, "private_strict_pass": None,
                        "execution_status": "abstained_missing_prefix_binding",
                        "verifier_wall_ms": 0.0, "official_verifier_receipt": None,
                    })
                    continue
                cand = compose_m98_two_stage_solver_code(
                    upstream_code=(c_up.canonical_algorithm_code or c_up.parameterized_closure_code),
                    upstream_entrypoint=(c_up.canonical_entrypoint or c_up.parameterized_entrypoint),
                    downstream_code=(b_dn.canonical_algorithm_code or b_dn.parameterized_closure_code),
                    downstream_entrypoint=(b_dn.canonical_entrypoint or b_dn.parameterized_entrypoint),
                    param_payload=pay,
                )
                t_v = monotonic()
                ver = grade_m94_artifact_map(
                    tk, {str(tk.declared_artifacts[0]): cand}, readonly_dependencies=ro_deps, expected_dependency_sha256=ro_shas
                )
                rows.append(
                    {
                        "policy": pol_nm,
                        "world_id": wid,
                        "family": spec.family,
                        "event": ev_nm,
                        "withdrawal_evidence_basis": basis, "policy_input_source": policy_input,
                        "prefix_checkpoint": "graph_ckpt_after_step_05.json",
                        "public_batch_pkl_sha256": pub_sha,
                        "downstream_observables": dn_obs,
                        "selected_upstream": c_up.binding_id,
                        "selected_downstream": b_dn.binding_id,
                        "candidate_sha256": hashlib.sha256(cand.encode("utf-8")).hexdigest(),
                        "private_strict_pass": bool(ver.get("strict_pass")),
                        "execution_status": "graded",
                        "verifier_wall_ms": round((monotonic() - t_v) * 1000.0, 2),
                        "official_verifier_receipt": ver,
                    }
                )

    summary = {}
    for pol in ("factorized_fixed_D2_separable", "factorized_fixed_D2_coupled", "hegel_selective_cross_factor_graph"):
        sub = [r for r in rows if r["policy"] == pol]
        summary[pol] = {
            "total_episodes": len(sub),
            "abstained_missing_prefix_binding": sum(r["private_strict_pass"] is None for r in sub),
            "strict_passes": sum(1 for r in sub if r["private_strict_pass"]),
            "e6_passes": sum(1 for r in sub if r["event"] == "E6_separable_revisit" and r["private_strict_pass"]),
            "e7_passes": sum(1 for r in sub if r["event"] == "E7_coupled_revisit" and r["private_strict_pass"]),
            "pass_rate_pct": round(100.0 * sum(1 for r in sub if r["private_strict_pass"]) / max(1, len(sub)), 2),
        }
    return {
        "schema": "hegel_m9_9_factorized_ceiling_vs_selective_interaction_ablation_v2",
        "paid_model_calls": 0,
        "evidence_provenance": "executed",
        "dn_obs_byte_identical_across_E6_and_E7": all(
            len({json.dumps(row["downstream_observables"], sort_keys=True)
                 for row in rows if row["world_id"] == spec.world_id}) == 1 for spec in world_specs
        ),
        "summary_by_policy": summary,
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--v7-bundle", type=Path, default=DEFAULT_V7_BUNDLE)
    ap.add_argument("--split", type=str, default="confirmation", choices=["dev", "confirmation"])
    ap.add_argument("--model", type=str, default="gpt-6.2-sol")
    ap.add_argument("--reasoning-effort", type=str, default="medium")
    ap.add_argument("--max-turns", type=int, default=18)
    ap.add_argument("--max-codex-concurrency", type=int, default=32)
    ap.add_argument("--max-stream-workers", type=int, default=24)
    ap.add_argument("--finalize-existing", action="store_true")
    args = ap.parse_args()

    selected_specs: tuple[M99WorldSpec, ...] = DEV_M99_WORLD_SPECS if args.split == "dev" else CONFIRMATION_M99_WORLD_SPECS
    os.environ["HEGEL_MAX_CODEX_CONCURRENCY"] = str(args.max_codex_concurrency)
    out_dir: Path = args.out_dir.resolve()
    if not args.finalize_existing and out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory `{out_dir}`!")

    git_sha_start, is_dirty_start, src_tree_sha_start = _git_and_source_tree_state()
    if is_dirty_start and args.split == "confirmation" and not args.finalize_existing:
        raise RuntimeError("Refusing to start M9.9 confirmation experiment on a dirty working tree!")

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_mirror_dir = out_dir / "provider_raw_attempts_mirror"
    raw_mirror_dir.mkdir(parents=True, exist_ok=True)
    campaign_t0 = monotonic()
    run_id = f"m99_{args.split}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    exec_manifest_path = out_dir / "execution_manifest.json"
    end_path = out_dir / "execution_completion.json"
    abl_path = out_dir / "m9_9_factorized_ceiling_ablation.json"
    abl_manifest_path = out_dir / "ablation_manifest.json"
    env_img_digest = None
    if not args.finalize_existing or not abl_path.is_file():
        ensure_clean_scipy_sandbox_image()
        m64.ensure_runtime_qualification_context(sandbox_cid=None)
        env_img_digest = _docker_image_digest(m7a.CLEAN_SCIPY_SANDBOX_IMAGE)
    codex_info = {}
    if not args.finalize_existing:
        codex_info = m6f.discover_chatgpt_subscription_codex_cli()
        if not (codex_info.get("available") and codex_info.get("auth_mode") == "chatgpt"):
            raise RuntimeError(f"Official ChatGPT Codex CLI unavailable: {codex_info}")
        exec_manifest_init = {
            "schema": "hegel_execution_provenance_manifest_v1",
            "execution_run_id": run_id, "split": args.split,
            "model": args.model, "reasoning_effort": args.reasoning_effort,
            "configured_max_codex_concurrency": args.max_codex_concurrency,
            "public_recovery_budget": {
                "max_pre_llm_probes_all_arms": 4, "max_direct_component_probes_per_modular_arm": 2,
                "max_alternative_upstreams": 1, "max_archived_raw_solvers": 1,
                "max_projection_source_target_pairs": 1,
            },
            "sandbox_image_digest": env_img_digest, "shared_kernel_version": SHARED_KERNEL_VERSION,
            "cost_policy_version": COST_POLICY_VERSION, "max_turns": args.max_turns,
            "input_forest_sha256": hashlib.sha256(args.v7_bundle.read_bytes()).hexdigest(),
            "world_ids": [spec.world_id for spec in selected_specs],
            "code_commit_at_start": git_sha_start, "working_tree_dirty_at_start": bool(is_dirty_start),
            "policy_public_input_source": "sandbox_public_pickle",
            "source_tree_sha256_at_start": src_tree_sha_start,
            "clock_domain": "monotonic_elapsed_plus_utc_iso8601",
            "utc_started_at": datetime.now(timezone.utc).isoformat(),
        }
        with exec_manifest_path.open("x", encoding="utf-8") as stream:
            json.dump(exec_manifest_init, stream, indent=2)
            stream.write("\n")

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
        with cli_sem:
            mono_t0 = monotonic()
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
            wall_ms=float(res.get("wall_ms", 0) or 0.0),
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
            "usage_unknown_attempt_count": unk_cnt,
            **sc,
        }
        with receipts_lock:
            all_receipts.append(rec_row)
        return res

    if not args.finalize_existing:
        m6f.FrontierCodexClient.call_structured = _wrapped_call_structured
        v7_forest, v7_archive, _, _ = m7a.load_autogenous_forest_bundle(args.v7_bundle.resolve())
        source_nid = sorted(v7_forest.nodes.keys())[0]

    built_worlds: dict[str, Any] = {spec.world_id: build_m99_world_tasks(spec) for spec in selected_specs}

    ablation_t0 = None
    rebuilt_streams = set()
    previous_path = out_dir / "m9_9_continual_summary.json"
    previous = json.loads(previous_path.read_text()) if args.finalize_existing and previous_path.is_file() else {}
    if args.finalize_existing:
        wid_to_cid = {sp.world_id: sp.cluster_id for sp in selected_specs}
        rec_path = out_dir / "all_provider_call_receipts.json"
        if rec_path.exists():
            all_receipts = json.loads(rec_path.read_text(encoding="utf-8"))
        else:
            all_receipts = []
            for mf in sorted(raw_mirror_dir.glob("*.json")):
                rj = json.loads(mf.read_text(encoding="utf-8"))
                ss = str(rj.get("stage_subdir") or "")
                parts = ss.split("__")
                if "__stageS__" in ss:
                    wid_r = parts[2]
                    arm_r = "Stage_S_Shared_Source"
                    ep_r = f"{wid_r}_s00"
                    role_r = "stage_S_solve"
                else:
                    wid_r = parts[1]
                    arm_r = parts[2]
                    step_num = int(parts[3].replace("step", ""))
                    ep_r = f"{wid_r}_batch_{step_num:02d}"
                    if "_decomp_att" in ss:
                        role_r = "autonomous_decomposition_induction"
                    elif "_gind_att" in ss:
                        role_r = "autonomous_guard_induction"
                    else:
                        role_r = "stream_solve"
                cu = rj.get("cumulative_usage") or {}
                w_ms = round(sum(float(a.get("wall_ms", 0.0) or 0.0) for a in (rj.get("attempts") or [])), 2)
                sc = score_single_call_usage(
                    prompt_tokens=int(cu.get("prompt_tokens", 0) or 0),
                    cached_tokens=int(cu.get("cached_tokens", 0) or 0),
                    completion_tokens=int(cu.get("completion_tokens", 0) or 0),
                    wall_ms=w_ms,
                )
                t_s = datetime.fromisoformat(str(rj["started_at_utc"])).timestamp()
                t_e = datetime.fromisoformat(str(rj["ended_at_utc"])).timestamp()
                cu_id = str(rj["call_uuid"])
                att_fn = mf.name.split("__", 1)[1] if "__" in mf.name else mf.name
                all_receipts.append(
                    {
                        "call_uuid": cu_id,
                        "world_id": wid_r,
                        "cluster_id": wid_to_cid[wid_r],
                        "arm": arm_r,
                        "episode_short_name": ep_r,
                        "call_role": role_r,
                        "stage_subdir": ss,
                        "attempts_ledger_filename": att_fn,
                        "mirrored_raw_filename": mf.name,
                        "utc_start_epoch_sec": round(t_s, 4),
                        "utc_end_epoch_sec": round(t_e, 4),
                        "usage_unknown_attempt_count": int(rj.get("usage_unknown_attempt_count", 0) or 0),
                        **sc,
                    }
                )
            all_receipts.sort(key=lambda x: (x["utc_start_epoch_sec"], x["call_uuid"]))
        uuids = {row.get("call_uuid") for row in all_receipts}
        expected_count = previous.get("total_logical_calls", len(all_receipts))
        if not all_receipts or None in uuids or "" in uuids or len(uuids) != len(all_receipts) or len(uuids) != expected_count:
            raise ValueError("Missing or duplicated provider receipts during finalization")
        if not rec_path.exists():
            rec_path.write_text(json.dumps(all_receipts, indent=2) + "\n", encoding="utf-8")

        abl_path = out_dir / "m9_9_factorized_ceiling_ablation.json"
        if abl_path.exists():
            ceiling_ablation = json.loads(abl_path.read_text(encoding="utf-8"))
        else:
            print("[M9.9] Running 0-LLM Factorized-Ceiling vs Selective-Interaction-Graph Docker ablation...", flush=True)
            ablation_t0 = monotonic()
            ceiling_ablation = run_m99_factorized_ceiling_ablation_on_sealed_streams(
                world_specs=selected_specs, built_worlds=built_worlds, out_dir=out_dir
            )
            ablation_wall_sec = monotonic() - ablation_t0
            abl_path.write_text(json.dumps(ceiling_ablation, indent=2) + "\n", encoding="utf-8")

        # Recompute exact peak active concurrency from recorded `[monotonic_start_sec, monotonic_end_sec]` intervals
        ev_pts: list[tuple[float, int]] = []
        for r in all_receipts:
            ev_pts.append((float(r.get("monotonic_start_sec", r.get("utc_start_epoch_sec"))), 1))
            ev_pts.append((float(r.get("monotonic_end_sec", r.get("utc_end_epoch_sec"))), -1))
        ev_pts.sort(key=lambda x: (x[0], x[1]))
        cur_c = 0
        for _, d_c in ev_pts:
            cur_c += d_c
            if cur_c > active_counter["peak"]:
                active_counter["peak"] = cur_c
        stream_summaries = []
        for sp in selected_specs:
            wid = sp.world_id
            _, tasks, manifest, _ = built_worlds[wid]
            for arm_nm in M99_ARMS:
                st_dir = out_dir / "continual_streams" / wid / arm_nm
                if (st_dir / "00_stream_summary.json").exists():
                    stream_summaries.append(json.loads((st_dir / "00_stream_summary.json").read_text(encoding="utf-8")))
                    continue
                ep_rows = []
                cov_all, gind_all, decomp_all, sw_all = [], [], [], []
                for tk in tasks:
                    ep_j = json.loads((st_dir / tk.task_id / "00_episode_summary.json").read_text(encoding="utf-8"))
                    ep_rows.append(ep_j)
                    cov_all.extend(ep_j.get("empirical_coverage_receipts") or [])
                    if ep_j.get("guard_induction_receipt") is not None:
                        gind_all.append({"step": ep_j["stream_step"], "event": ep_j["causal_event"], **ep_j["guard_induction_receipt"]})
                    if ep_j.get("autonomous_decomposition_receipt") is not None:
                        decomp_all.append({"step": ep_j["stream_step"], "event": ep_j["causal_event"], **ep_j["autonomous_decomposition_receipt"]})
                    sw_r = (ep_j.get("pre_llm_trial_receipt") or {}).get("separability_withdrawal_receipt")
                    if sw_r is not None:
                        sw_all.append(sw_r)
                st_sum = {
                    "world_id": wid,
                    "cluster_id": sp.cluster_id,
                    "family": sp.family,
                    "arm": arm_nm,
                    "wall_sec": None,  # Original stream clock unavailable after recovery.
                    "strict_passes": sum(1 for r in ep_rows if r.get("strict_pass")),
                    "episode_count": len(ep_rows),
                    "zero_llm_episodes_total": sum(1 for r in ep_rows if int(r.get("turns_used", 0) or 0) == 0),
                    "zero_turn_passed_closures": sum(1 for r in ep_rows if int(r.get("turns_used", 0) or 0) == 0 and r.get("strict_pass")),
                    "total_turns_used": sum(int(r.get("turns_used", 0) or 0) for r in ep_rows),
                    "total_input_materialization_wall_ms": (round(sum(r["input_materialization_wall_ms"] for r in ep_rows), 2)
                        if all("input_materialization_wall_ms" in r for r in ep_rows) else None),
                    "total_pre_llm_probe_wall_ms": round(sum(float(r.get("pre_llm_probe_wall_ms", 0.0) or 0.0) for r in ep_rows), 2),
                    "total_post_llm_probe_wall_ms": round(sum(float(r.get("post_llm_probe_wall_ms", 0.0) or 0.0) for r in ep_rows), 2),
                    "total_localization_probe_wall_ms": round(sum(float(r.get("localization_probe_wall_ms", 0.0) or 0.0) for r in ep_rows), 2),
                    "total_decomposition_probe_wall_ms": round(sum(float(r.get("decomposition_probe_wall_ms", 0.0) or 0.0) for r in ep_rows), 2),
                    "total_coverage_probe_wall_ms": round(sum(float(c["total_cpu_probe_wall_ms"]) for c in cov_all), 2),
                    "total_official_verifier_wall_ms": round(sum(float(r.get("official_verifier_wall_ms", 0.0) or 0.0) for r in ep_rows), 2),
                    "separability_withdrawal_receipts": sw_all,
                    "autonomous_decomposition_events": decomp_all,
                    "guard_induction_events": gind_all,
                    "empirical_coverage_receipts": cov_all,
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
                            "localization_probe_wall_ms": float(r.get("localization_probe_wall_ms", 0.0) or 0.0),
                            "decomposition_probe_wall_ms": float(r.get("decomposition_probe_wall_ms", 0.0) or 0.0),
                        }
                        for r in ep_rows
                    ],
                }
                rebuilt_streams.add((wid, arm_nm))
                stream_summaries.append(st_sum)
    else:
        # =========================================================================
        # Stage S: Shared Unfactored Baseline Source Synthesis (`0` Host-Seeded `u0`/`d0`!)
        # =========================================================================
        print(f"[M9.9] Starting Stage S unfactored baseline synthesis across {len(selected_specs)} worlds (`concurrency={args.max_codex_concurrency}`)...", flush=True)
        stage_s_results: dict[str, dict[str, Any]] = {}

        def _solve_one_stage_s(spec: M99WorldSpec) -> tuple[str, dict[str, Any]]:
            wid = spec.world_id
            s_task, _, _, logs = built_worlds[wid]
            s_pref_logs = logs[:1]
            s_ro_deps = {"/tmp/hegel_continual_archive/archive_apparatus_log.json": json.dumps(s_pref_logs, indent=2)}
            s_ro_shas = {k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in s_ro_deps.items()}
            aug_s_task, s_hdr = _inject_equal_info_archive_into_m99_task(s_task, [], s_pref_logs, None)
            ft_s = m99_task_to_frontier_task(
                aug_s_task,
                continual_context_block=s_hdr,
                allowed_readonly_dependencies=tuple(sorted(s_ro_deps.keys())),
            )
            s_root = out_dir / "stage_S_sources" / wid
            s_root.mkdir(parents=True, exist_ok=True)
            for att_s in range(1, 3):
                ep_d = s_root / f"attempt_{att_s:02d}" / s_task.task_id
                ep_d.mkdir(parents=True, exist_ok=True)
                _CALL_CONTEXT.role = "stage_S_solve"
                _CALL_CONTEXT.world_id = wid
                _CALL_CONTEXT.cluster_id = spec.cluster_id
                _CALL_CONTEXT.arm = "Stage_S_Shared_Source"
                _CALL_CONTEXT.episode_short_name = s_task.short_name
                f_cp = deepcopy(v7_forest)
                a_cp = deepcopy(v7_archive)
                a_cp["__autogenous_blank_seed__"] = {"enabled": True}
                ep_res = m6f.run_frontier_episode(
                    ft_s,
                    "Hegel_M6_9_Antithesis_LiveGPT",
                    model=args.model,
                    codex_info=codex_info,
                    forest=f_cp,
                    forest_archive=a_cp,
                    hippo_idx=HippoRAGWormholeIndex(f_cp),
                    episode_out_dir=ep_d,
                    run_id=f"{run_id}__stageS__{wid}__att{att_s}",
                    max_turns=args.max_turns,
                    cmd_timeout_sec=180,
                    reasoning_effort=args.reasoning_effort,
                    defer_official_verifier=True,
                    enable_auto_skill_packaging=False,
                    allow_l1_primitives=False,
                    disclosure_mode="on_demand_executable",
                    staging_policy="net_value_gated",
                )
                sub = _extract_submitted_files(ep_d)
                code_s = sub.get("batch_solver.py", "")
                pub_g = (
                    execute_single_pre_llm_public_gate_probe_in_container(
                        task=s_task, candidate_code=code_s, readonly_dependencies=s_ro_deps, expected_dependency_sha256=s_ro_shas
                    )
                    if code_s
                    else {"accepted": False}
                )
                priv_s = (
                    grade_m94_artifact_map(
                        s_task,
                        {str(s_task.declared_artifacts[0]): code_s},
                        readonly_dependencies=s_ro_deps,
                        expected_dependency_sha256=s_ro_shas,
                    )
                    if code_s
                    else {"strict_pass": False}
                )
                (ep_d / "03_official_verifier_report.json").write_text(json.dumps(priv_s, indent=2) + "\n", encoding="utf-8")
                if pub_g.get("accepted"):
                    input_start = monotonic()
                    pub_item_s, pub_sha_s = materialize_m99_public_item_and_sha(s_task)
                    pub_g["input_materialization_wall_ms"] = round((monotonic() - input_start) * 1000, 2)
                (ep_d / "stage_s_public_gate.json").write_text(json.dumps(pub_g, indent=2) + "\n")
                if pub_g.get("accepted"):
                    ent = make_continual_archive_entry(
                        short_name=s_task.short_name,
                        task_id=s_task.task_id,
                        canonical_entrypoint="solve_batch",
                        canonical_algorithm_code=code_s,
                        output_schema_keys=expected_m98_public_output_keys(spec.family),
                        public_qualification_status="public_gate_verified",
                    )
                    print(f"  [M9.9 Stage S | {wid}] PASS on attempt {att_s} (`turns={ep_res.get('turns_used')}`)", flush=True)
                    return wid, {
                        "code": code_s,
                        "archive_entry": ent,
                        "pub_item": pub_item_s,
                        "pub_sha": pub_sha_s,
                        "verifier_report": priv_s,
                    }
            raise RuntimeError(f"Stage S failed to produce a verified baseline solver for `{wid}`!")

        with ThreadPoolExecutor(max_workers=len(selected_specs)) as s_pool:
            futs_s = [s_pool.submit(_solve_one_stage_s, sp) for sp in selected_specs]
            for f in as_completed(futs_s):
                wid_k, s_val = f.result()
                stage_s_results[wid_k] = s_val

        # =========================================================================
        # Continual Streams: `3` Equal-Tool Arms × `n` Worlds in Parallel (`max-codex-concurrency = 32`)
        # =========================================================================
        stream_jobs = [(sp, arm_nm) for sp in selected_specs for arm_nm in M99_ARMS]
        print(f"[M9.9] Launching {len(stream_jobs)} continual streams (`3` arms x `{len(selected_specs)}` worlds)...", flush=True)
        stream_summaries = []

        def _run_stream_job(job: tuple[M99WorldSpec, str]) -> dict[str, Any]:
            sp, arm_nm = job
            wid = sp.world_id
            s_task, tasks, manifest, logs = built_worlds[wid]
            s_info = stage_s_results[wid]
            st_dir = out_dir / "continual_streams" / wid / arm_nm
            st_dir.mkdir(parents=True, exist_ok=True)
            return _run_one_m99_world_arm_stream(
                spec=sp,
                arm=arm_nm,
                s_task=s_task,
                stream_tasks=tasks,
                schedule_manifest=manifest,
                apparatus_logs=logs,
                base_v7_forest=v7_forest,
                base_v7_archive=v7_archive,
                source_monolithic_code=s_info["code"],
                source_archive_entry=s_info["archive_entry"],
                source_forest_nid=source_nid,
                source_pub_item=s_info["pub_item"],
                source_pub_sha256=s_info["pub_sha"],
                env_image_digest=env_img_digest,
                model=args.model,
                codex_info=codex_info,
                stream_out_dir=st_dir,
                run_id=run_id,
                max_turns=args.max_turns,
                reasoning_effort=args.reasoning_effort,
            )

        with ThreadPoolExecutor(max_workers=min(len(stream_jobs), int(args.max_stream_workers))) as pool:
            futs = [pool.submit(_run_stream_job, jb) for jb in stream_jobs]
            for f in as_completed(futs):
                stream_summaries.append(f.result())

        stream_order = {(sp.world_id, arm_nm): idx for idx, (sp, arm_nm) in enumerate(stream_jobs)}
        stream_summaries.sort(key=lambda s: stream_order[(s["world_id"], s["arm"])])

        end_commit, end_dirty, end_tree = _git_and_source_tree_state()
        completion = dict(execution_run_id=run_id, code_commit_at_end=end_commit,
            working_tree_dirty_at_end=end_dirty, source_tree_sha256_at_end=end_tree,
            source_tree_unchanged_during_execution=src_tree_sha_start == end_tree,
            execution_elapsed_wall_sec=round(monotonic()-campaign_t0, 2),
            utc_finished_at=datetime.now(timezone.utc).isoformat())
        with end_path.open("x", encoding="utf-8") as stream:
            json.dump(completion, stream, indent=2)
            stream.write("\n")
        # Run `0`-LLM Factorized-Ceiling vs Selective-Interaction-Graph Control on `{E6, E7}`
        print("[M9.9] Running 0-LLM Factorized-Ceiling vs Selective-Interaction-Graph Docker ablation...", flush=True)
        ablation_t0 = monotonic()
        ceiling_ablation = run_m99_factorized_ceiling_ablation_on_sealed_streams(
            world_specs=selected_specs, built_worlds=built_worlds, out_dir=out_dir
        )
        ablation_wall_sec = monotonic() - ablation_t0
        (out_dir / "m9_9_factorized_ceiling_ablation.json").write_text(json.dumps(ceiling_ablation, indent=2) + "\n", encoding="utf-8")
        (out_dir / "all_provider_call_receipts.json").write_text(json.dumps(all_receipts, indent=2) + "\n", encoding="utf-8")
    total_unknown_attempts = sum(int(r.get("usage_unknown_attempt_count", 0) or 0) for r in all_receipts)

    stage_s_calls = [r for r in all_receipts if r["arm"] == "Stage_S_Shared_Source"]
    stage_s_ledger = aggregate_call_usages(stage_s_calls)

    arm_cost_ledgers: dict[str, Any] = {}
    arm_outcome_totals: dict[str, Any] = {}
    ev_names = [m["causal_event"] for m in built_worlds[selected_specs[0].world_id][2]]
    sn_to_ev = {
        m["short_name"]: m["causal_event"]
        for sp in selected_specs
        for m in built_worlds[sp.world_id][2]
    }

    # Resolve zero-call episodes from provider receipts across all roles, including guard induction.
    billed_episodes = {(row["world_id"], row["arm"], row["episode_short_name"]) for row in all_receipts}
    for stream in stream_summaries:
        key = (stream["world_id"], stream["arm"])
        zero_call_episodes = [ep for ep in stream["episodes_meta"] if (*key, ep["short_name"]) not in billed_episodes]
        stream["zero_llm_episodes_total"] = len(zero_call_episodes)
        stream["zero_turn_passed_closures"] = sum(bool(ep["strict_pass"]) for ep in zero_call_episodes)
        if not args.finalize_existing or key in rebuilt_streams:
            stream_file = out_dir.joinpath("continual_streams", *key, "00_stream_summary.json")
            stream_file.write_text(json.dumps(stream, indent=2) + "\n", encoding="utf-8")
    for arm_nm in M99_ARMS:
        a_calls = [r for r in all_receipts if r["arm"] == arm_nm]
        tot_led = aggregate_call_usages(a_calls)
        solve_led = aggregate_call_usages([r for r in a_calls if r["call_role"] == "stream_solve"])
        decomp_led = aggregate_call_usages([r for r in a_calls if r["call_role"] == "autonomous_decomposition_induction"])
        guard_led = aggregate_call_usages([r for r in a_calls if r["call_role"] == "autonomous_guard_induction"])

        by_ep = {
            ep_sn: aggregate_call_usages([r for r in a_calls if r["episode_short_name"] == ep_sn])
            for sp in selected_specs
            for ep_sn in [m["short_name"] for m in built_worlds[sp.world_id][2]]
        }
        by_ev = {
            ev: aggregate_call_usages([r for r in a_calls if sn_to_ev.get(r["episode_short_name"]) == ev])
            for ev in ev_names
        }
        by_cl = {
            sp.world_id: aggregate_call_usages([r for r in a_calls if r["world_id"] == sp.world_id])
            for sp in selected_specs
        }
        assert_hierarchical_ledger_conservation(
            reference_arm_ledger=tot_led,
            named_partition_ledgers={
                "episodes": list(by_ep.values()),
                "events": list(by_ev.values()),
                "clusters": list(by_cl.values()),
                "roles": [solve_led, decomp_led, guard_led],
            },
            context_label=arm_nm,
        )

        arm_cost_ledgers[arm_nm] = {
            "scope1_in_stream_total": tot_led,
            "solve_only": solve_led,
            "decomposition_induction_only": decomp_led,
            "guard_induction_only": guard_led,
            "by_event": by_ev,
            "by_cluster": by_cl,
        }
        sts = [s for s in stream_summaries if s["arm"] == arm_nm]
        arm_outcome_totals[arm_nm] = {
            "strict_passes": sum(int(s["strict_passes"]) for s in sts),
            "episode_count": sum(int(s["episode_count"]) for s in sts),
            "zero_turn_passed_closures": sum(int(s["zero_turn_passed_closures"]) for s in sts),
            "separability_withdrawals": sum(len(s["separability_withdrawal_receipts"]) for s in sts),
            "input_materialization_wall_ms": (round(sum(s["total_input_materialization_wall_ms"] for s in sts), 2)
                if all(s.get("total_input_materialization_wall_ms") is not None for s in sts) else None),
            "pre_llm_probe_wall_ms": round(sum(float(s["total_pre_llm_probe_wall_ms"]) for s in sts), 2),
            "post_llm_probe_wall_ms": round(sum(float(s["total_post_llm_probe_wall_ms"]) for s in sts), 2),
            "localization_probe_wall_ms": round(sum(float(s["total_localization_probe_wall_ms"]) for s in sts), 2),
            "decomposition_probe_wall_ms": round(sum(float(s["total_decomposition_probe_wall_ms"]) for s in sts), 2),
            "coverage_probe_wall_ms": round(sum(float(s["total_coverage_probe_wall_ms"]) for s in sts), 2),
            "official_verifier_wall_ms": round(sum(float(s["total_official_verifier_wall_ms"]) for s in sts), 2),
        }

    def _pair_stats(base_arm: str) -> dict[str, Any]:
        diffs_all = [
            float(arm_cost_ledgers["Hegel_AutonomousInteractionGraph_Online"]["by_cluster"][sp.world_id]["proxy_v2_u_010c_4o"])
            - float(arm_cost_ledgers[base_arm]["by_cluster"][sp.world_id]["proxy_v2_u_010c_4o"])
            for sp in selected_specs
        ]
        diffs_alpha = [
            float(arm_cost_ledgers["Hegel_AutonomousInteractionGraph_Online"]["by_cluster"][sp.world_id]["proxy_v2_u_010c_4o"])
            - float(arm_cost_ledgers[base_arm]["by_cluster"][sp.world_id]["proxy_v2_u_010c_4o"])
            for sp in selected_specs
            if sp.family == "alpha"
        ]
        diffs_beta = [
            float(arm_cost_ledgers["Hegel_AutonomousInteractionGraph_Online"]["by_cluster"][sp.world_id]["proxy_v2_u_010c_4o"])
            - float(arm_cost_ledgers[base_arm]["by_cluster"][sp.world_id]["proxy_v2_u_010c_4o"])
            for sp in selected_specs
            if sp.family == "beta"
        ]
        qual_diffs = [
            float(
                next(s["strict_passes"] for s in stream_summaries if s["world_id"] == sp.world_id and s["arm"] == "Hegel_AutonomousInteractionGraph_Online")
                - next(s["strict_passes"] for s in stream_summaries if s["world_id"] == sp.world_id and s["arm"] == base_arm)
            )
            for sp in selected_specs
        ]
        zllm_diffs = [
            float(
                next(s["zero_turn_passed_closures"] for s in stream_summaries if s["world_id"] == sp.world_id and s["arm"] == "Hegel_AutonomousInteractionGraph_Online")
                - next(s["zero_turn_passed_closures"] for s in stream_summaries if s["world_id"] == sp.world_id and s["arm"] == base_arm)
            )
            for sp in selected_specs
        ]
        return {
            "cluster_v2_diffs": [round(x, 2) for x in diffs_all],
            "mean_diff": round(float(np.mean(diffs_all)), 2),
            "bootstrap_95ci": _bootstrap_mean_ci(diffs_all),
            "two_sided_sign_flip_p": _exact_two_sided_sign_flip_p(diffs_all),
            "alpha_sign_flip_p": _exact_two_sided_sign_flip_p(diffs_alpha),
            "beta_sign_flip_p": _exact_two_sided_sign_flip_p(diffs_beta),
            "quality_strict_pass_cluster_diffs": [int(x) for x in qual_diffs],
            "quality_two_sided_sign_flip_p": _exact_two_sided_sign_flip_p(qual_diffs),
            "zero_llm_closure_cluster_diffs": [int(x) for x in zllm_diffs],
            "zero_llm_two_sided_sign_flip_p": _exact_two_sided_sign_flip_p(zllm_diffs),
        }

    git_sha_end, is_dirty_end, src_tree_sha_end = _git_and_source_tree_state()
    start = json.loads(exec_manifest_path.read_text()) if exec_manifest_path.is_file() else {}
    end = json.loads(end_path.read_text()) if end_path.is_file() else {}
    execution = _execution_fields(start, end, all_receipts)
    if ablation_t0 is not None:
        ablation_manifest = dict(code_commit=git_sha_start, source_tree_sha256=src_tree_sha_start,
            source_run_id=execution["run_id"], checkpoint_prefix="step_05",
            result_file=abl_path.name, result_sha256=hashlib.sha256(abl_path.read_bytes()).hexdigest(),
            sandbox_image_digest=env_img_digest, elapsed_wall_sec=ablation_wall_sec)
        abl_manifest_path.write_text(json.dumps(ablation_manifest, indent=2)+"\n")
    ablation_provenance = (json.loads(abl_manifest_path.read_text()) if abl_manifest_path.is_file()
                           else dict(result_file=abl_path.name, code_commit=None, source_tree_sha256=None))
    if abl_manifest_path.is_file():
        if ablation_provenance.get("source_run_id") != execution["run_id"]:
            raise ValueError("Ablation manifest belongs to a different execution")
        if ablation_provenance.get("result_sha256") != hashlib.sha256(abl_path.read_bytes()).hexdigest():
            raise ValueError("Ablation result no longer matches its manifest")
    fin_manifest = None
    if args.finalize_existing:
        fin_manifest = dict(finalization_run_id="finalize_"+run_id, finalization_commit=git_sha_end,
            finalization_source_tree_sha256=src_tree_sha_end, finalization_working_tree_dirty=is_dirty_end,
            finalization_elapsed_wall_sec=round(monotonic()-campaign_t0, 2),
            execution_manifest=exec_manifest_path.name if start else None,
            execution_completion=end_path.name if end_path.is_file() else None,
            ablation_manifest=abl_manifest_path.name if abl_manifest_path.is_file() else None,
            reused_ablation=ablation_t0 is None, call_receipts="all_provider_call_receipts.json")
        (out_dir / "finalization_manifest.json").write_text(json.dumps(fin_manifest, indent=2)+"\n")
    summary = {
        "schema": "hegel_m9_9_autonomous_decomposition_and_interaction_withdrawal_summary_v2",
        **execution,
        "split": args.split,
        "evidence_provenance": ("executed_with_post_finalization" if args.finalize_existing else "executed"),
        "equal_tool_contract": {
            "enable_auto_skill_packaging_all_arms": False,
            "allow_l1_primitives_all_arms": False,
            **(start.get("public_recovery_budget") or {"max_pre_llm_probes_all_arms": 1}),
            "host_seeded_u0_or_d0": False,
            "modular_repair_prompt_parity_plain_vs_hegel": True,
        },
        "finalizer_provenance": fin_manifest,
        "paid_llm_frozen_commit": previous.get("paid_llm_frozen_commit") or (
            start.get("code_commit_at_start") if start.get("working_tree_dirty_at_start") is False else None),
        **{field: execution["execution_provenance"].get(field) for field in (
            "code_commit_at_start", "code_commit_at_end", "working_tree_dirty_at_start",
            "source_tree_sha256_at_start", "source_tree_sha256_at_end")},
        "source_tree_unchanged_during_run": end.get("source_tree_unchanged_during_execution"),
        "ablation_provenance": ablation_provenance,
        "configuration_source": "execution_manifest" if start else ("historical_summary" if previous else None),
        **{field: start.get(field, previous.get(field)) for field in ("sandbox_image_digest", "shared_kernel_version",
            "cost_policy_version", "model", "reasoning_effort", "configured_max_codex_concurrency")},
        "peak_active_codex_concurrency": int(active_counter["peak"]),
        "usage_completeness": ("complete" if total_unknown_attempts == 0 else "partial_unknown_usage"),
        "total_unknown_usage_attempts": total_unknown_attempts,
        "total_logical_calls": len(all_receipts),
        "stage_S_shared_ledger": stage_s_ledger,
        "arm_outcome_totals": arm_outcome_totals,
        "arm_cost_ledgers": arm_cost_ledgers,
        "paired_cluster_inference": {
            "Hegel_minus_Flat_PlainModuleRegistry_Online": _pair_stats("Flat_PlainModuleRegistry_Online"),
            "Hegel_minus_Flat_MonolithicConditionalTable_Online": _pair_stats("Flat_MonolithicConditionalTable_Online"),
        },
        "factorized_ceiling_ablation_summary": ceiling_ablation["summary_by_policy"],
        "streams": stream_summaries,
    }
    (out_dir / "m9_9_continual_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    sha_lines = []
    for fp in sorted(out_dir.rglob("*")):
        if fp.is_file() and fp.name != "SHA256SUMS.txt":
            rel = fp.relative_to(out_dir).as_posix()
            sha_lines.append(f"{hashlib.sha256(fp.read_bytes()).hexdigest()}  {rel}")
    (out_dir / "SHA256SUMS.txt").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "run_id": execution["run_id"],
                "split": args.split,
                "total_calls": len(all_receipts),
                "peak_concurrency": active_counter["peak"],
                "unknown_usage_attempts": total_unknown_attempts,
                "arm_outcome_totals": arm_outcome_totals,
                "paired_cluster_inference": summary["paired_cluster_inference"],
                "sha256_entries": len(sha_lines),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
