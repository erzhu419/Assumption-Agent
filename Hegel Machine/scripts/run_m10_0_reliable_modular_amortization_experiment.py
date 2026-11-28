#!/usr/bin/env python3
"""Live Execution & Pre-Registered Economic Break-Even ($K^*$) Runner for Hegel `M10.0`:
Reliable Modular Program Knowledge (`B1–B3`), Bidirectional Separability Withdrawal (`sep_first` & `coupled_first`),
and `3 × 3` Continual Amortization Against BOTH `Flat_PlainModuleRegistry_Online` AND `Flat_MonolithicConditionalTable_Online`.

Enforces ALL Audit Contracts from `Hegel_M9_9_Audit` (`Q01–Q14` & `01_Implementation_and_Experiment_Plan_ZH.md` §3–§8):
  - `C1 / Q05`: Write-once `execution_manifest.json` persisted BEFORE the first paid LLM call, strictly separated from
    `ablation_manifest.json` and `finalization_manifest.json`.
  - `A1 / Q01`: Full `2×2` container decomposition verification + `record_tested_witness_from_probe_receipt` (`0` unexecuted witnesses!).
  - `A2 / Q07`: Full 6-link foreign-key & temporal witness-ledger verification on `SeparabilityWithdrawalReceipt`.
  - `B1 / Q02`: Shared Alpha public pooled `train+test` `ref_coupons` swap invariance gate (`require_pooled_coupon_check=True`
    across ALL 3 arms, `0` private truth!).
  - `B2 / Q03`: Symmetric `1`-alternative verified-sibling upstream fallback (`direct 56/24 probe -> joint 3-layer public gate`,
    `100%` timed) across `Flat_PlainModuleRegistry_Online` and `Hegel_SelectiveConditionExpansion_Online`.
  - `B3 / Q04`: Symmetric unfactored monolithic fallback + lazy on-demand retroactive downstream projection (`100%` metered
    in `Scope-1` as `lazy_retroactive_projection`) across `Plain` and `Hegel`.
  - `D2 / §7–§8`: Step-by-step cumulative paired cost curves `ΔC_mono(K)` & `ΔC_plain(K)` (`K = 1..10`), empirical
    quality-constrained break-even step `K*` (`"not_reached"` when right-censored), pre-registered quality non-inferiority
    margin `delta_NI = -0.05` (`-0.5` episode/cluster), and separate cost / zero-LLM / quality exact sign-flip `p`-values.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import threading
from time import monotonic
from typing import Any
import uuid

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scripts.run_m9_9_autonomous_decomposition_experiment import _exact_two_sided_sign_flip_p, _bootstrap_mean_ci

from hegel_machine.cognition import (  # noqa: E402
    m6_4_executable_wormhole_library as m64,
    m6_frontier_real_benchmarks as m6f,
    m7_autogenous_forest_pretraining as m7a,
)
from hegel_machine.cognition.cost_policy import (  # noqa: E402
    COST_POLICY_VERSION,
    aggregate_call_usages,
    assert_hierarchical_ledger_conservation,
    score_single_call_usage,
)
from hegel_machine.cognition.hipporag_wormhole_index import HippoRAGWormholeIndex
from hegel_machine.cognition.m7_holdout_ood_blind_benchmark import ensure_clean_scipy_sandbox_image
from hegel_machine.cognition.m8_persistent_variational_planner import (  # noqa: E402
    PersistentMechanismGraph,
)
from hegel_machine.cognition.m9_4_1_shared_verification_kernel import (  # noqa: E402
    SHARED_KERNEL_VERSION,
)
from hegel_machine.cognition.m9_4_continual_mechanism_binding_world import (  # noqa: E402
    M94BindingRegistry,
    MechanismExecutableBinding,
    grade_m94_artifact_map,
)
from hegel_machine.cognition.m9_5_equal_tool_continual_attribution_world import (  # noqa: E402
    execute_single_pre_llm_public_gate_probe_in_container,
    make_continual_archive_entry,
)
from hegel_machine.cognition.m9_6_autonomous_structure_continual_world import (  # noqa: E402
    EmpiricalCoverageReceipt,
    ImmutableBranchLineageController,
    classify_public_probe_failure,
    joint_scientific_receipt_matches,
    compute_canonical_numeric_item_sha256,
    evaluate_m96_autonomous_guard,
    joint_scientific_receipt_matches,
)
from hegel_machine.cognition.m9_8_blind_mismatch_and_selective_dependency_world import (  # noqa: E402
    assemble_m98_graph_diagonal_payload,
    build_m98_upstream_transducer_synthesis_task,
    compose_m98_two_stage_solver_code,
    compute_m98_downstream_observables,
    diagnose_m98_factored_conflict_locus,
    expected_m98_public_output_keys,
    initialize_m98_factored_mechanism_graph,
)
from hegel_machine.cognition.m9_9_autonomous_decomposition_and_interaction_withdrawal_world import (  # noqa: E402
    issue_measured_withdrawal,
    materialize_m99_public_item_and_sha,
    propose_and_verify_m99_autonomous_two_stage_factorization,
    update_m99_calibration_snapshot_and_graph,
)
from hegel_machine.cognition.m10_0_reliable_modular_amortization_and_bidirectional_withdrawal_world import (  # noqa: E402
    CONFIRMATION_M100_WORLD_SPECS,
    DEV_M100_WORLD_SPECS,
    M100_ALL_OBSERVABLE_NAMES,
    M100_ARMS,
    M100_DOWNSTREAM_OBSERVABLE_NAMES,
    M100_UPSTREAM_OBSERVABLE_NAMES,
    M100WorldSpec,
    SeparabilityWithdrawalReceipt,
    build_m100_world_tasks,
    build_m99_downstream_canonical_synthesis_task,
    check_witness_feature_collision,
    compute_m100_upstream_observables,
    hydrate_separability_withdrawals_from_graph,
    m100_task_to_frontier_task,
    persist_separability_withdrawals_to_graph,
    propose_and_validate_m99_guard_partition,
    propose_and_verify_lazy_retroactive_downstream_projection,
)

DEFAULT_V7_BUNDLE = REPO_ROOT / "docs/evidence/m9_2_1_audit_closure_and_regrade_20261126/pretrained_autogenous_forest_v7_m921.json"
DEFAULT_DEV_OUT_DIR = REPO_ROOT / "docs/evidence/m10_0_dev_economic_calibration_20261128"
DEFAULT_CONF_OUT_DIR = REPO_ROOT / "docs/evidence/m10_0_reliable_modular_amortization_and_bidirectional_withdrawal_20261128"

# Pre-registered quality non-inferiority margin (`01_Implementation_and_Experiment_Plan_ZH.md` §8.2):
# `Hegel` pass rate minus `Monolithic` pass rate >= -0.05 (`-0.5` episode per 10-step cluster, i.e. at most 4 fewer passes across 80 episodes).
PREREGISTERED_QUALITY_NONINFERIORITY_MARGIN_PER_EPISODE = -0.05

_CALL_CONTEXT = threading.local()


def paired_cluster_sign_flip_and_bootstrap(diffs):
    return {"mean_diff": round(float(np.mean(diffs)), 6) if diffs else None,
            "two_sided_sign_flip_p": _exact_two_sided_sign_flip_p(diffs) if diffs else None,
            "bootstrap_95ci": _bootstrap_mean_ci(diffs) if diffs else None}


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
    if r.returncode or not r.stdout.strip().startswith("sha256:"):
        raise RuntimeError(f"Could not determine the Docker image identity for {image_name}")
    return r.stdout.strip()


def _extract_submitted_files(ep_dir: Path) -> dict[str, str]:
    sdir = ep_dir / "04_submitted_artifacts_snapshot"
    out: dict[str, str] = {}
    if not sdir.is_dir():
        return out
    for fp in sorted(sdir.rglob("*.py")):
        bname = fp.name.split("__", 1)[-1] if "__" in fp.name else fp.name
        out[bname] = fp.read_text(encoding="utf-8")
    return out


def _inject_equal_info_archive_into_m100_task(
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

    from hegel_machine.cognition.m7_holdout_ood_blind_benchmark import HoldoutOODTask

    aug = HoldoutOODTask(
        **{
            **task.__dict__,
            "public_setup_py": task.public_setup_py + "\n" + "\n".join(extra_lines) + "\n",
        }
    )
    ctx_hdr = (
        "CONTINUAL CAMPAIGN ARCHIVE & FACILITY TELEMETRY LOG:\n"
        "- Prior verified campaign solvers: `/tmp/hegel_continual_archive/archive_manifest.json`\n"
        "- Chronological prefix apparatus equilibration telemetry (`<= current_step`): `/tmp/hegel_continual_archive/archive_apparatus_log.json` (`0444`)\n"
        "- Pre-LLM public gate diagnostic: `/tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json`\n"
    )
    return aug, ctx_hdr


def _plain_table_select_row(
    rows: list[dict[str, Any]],
    req_keys: list[str],
    features: dict[str, float | None],
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    """Deterministic Kleene `all_of` selector for `Plain` / `Monolithic` tables; also returns ordered non-selected verified rows."""
    compat = [r for r in rows if set(req_keys).issubset(set(r.get("output_schema_keys") or []))]
    matched = []
    for r in compat:
        ev = evaluate_m96_autonomous_guard(r.get("guard") or {}, features, allow_open_feature_keys=True)
        if ev["state"] == "TRUE":
            matched.append(r)
    matched.sort(key=lambda r: (-int(r.get("priority", 0)), str(r["row_id"])))
    compat_sorted = sorted(compat, key=lambda r: (-int(r.get("priority", 0)), str(r["row_id"])))
    if matched:
        chosen = matched[0]
        alts = [r for r in compat_sorted if r["row_id"] != chosen["row_id"]]
        return chosen, ("unique_branch" if len(matched) == 1 else "priority_tiebreak"), alts
    chosen = compat_sorted[0]
    alts = [r for r in compat_sorted if r["row_id"] != chosen["row_id"]]
    return chosen, "fallback_highest_priority", alts


def _guard_uses_upstream_features(guard_spec: dict[str, Any] | None) -> bool:
    if not guard_spec:
        return False
    if isinstance(guard_spec.get("all_of"), list):
        return any(_guard_uses_upstream_features(c) for c in guard_spec["all_of"])
    return str(guard_spec.get("feature_name") or "").startswith("up_stat_")


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
) -> MechanismExecutableBinding:
    sha = hashlib.sha256(code_txt.encode("utf-8")).hexdigest()
    qrec = registry.issue_qualification_receipt(
        qualification_protocol="hegel_public_scientific_gate_v4",
        verified_code_sha256=sha,
        forest_node_id=forest_node_id,
        environment_digest=env_image_digest,
    )
    pvers = {pid: int(graph.mechanisms[pid].version) for pid in parent_ids}
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
        module_stem=binding_id,
        module_digest=sha,
        function_name=entrypoint,
        qualification_id="hegel_public_scientific_gate_v4",
        qualification_receipt_id=qrec.receipt_id,
        parameter_mechanism_id=parent_ids[0],
        parameter_version=pvers[parent_ids[0]],
        parent_parameter_mechanism_ids=list(parent_ids),
        parent_parameter_versions=pvers,
        canonical_algorithm_code=code_txt,
        canonical_entrypoint=entrypoint,
        output_schema_keys=list(output_schema_keys),
        applicability_guard=deepcopy(applicability_guard),
        precondition_guard=deepcopy(applicability_guard),
        selection_priority=selection_priority,
    )
    registry.register_binding(b, strict_receipt_check=True)
    return b


def _execute_bound_gate(*, task, candidate_code, readonly_dependencies=None, expected_dependency_sha256=None,
                        require_pooled_coupon_check=False):
    try:
        gate = execute_single_pre_llm_public_gate_probe_in_container(
            task=task, candidate_code=candidate_code, readonly_dependencies=readonly_dependencies,
            expected_dependency_sha256=expected_dependency_sha256, require_pooled_coupon_check=require_pooled_coupon_check,
        )
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        return {"accepted": False, "setup_ok": False, "timed_out": isinstance(exc, subprocess.TimeoutExpired),
                "infrastructure_error": str(exc)}
    tier = classify_public_probe_failure(gate)["tier"]
    if tier == "INFRA_FAILURE":
        return {**gate, "accepted": False}
    item, public_sha = materialize_m99_public_item_and_sha(task)
    numeric_sha = compute_canonical_numeric_item_sha256(item)
    dependencies = expected_dependency_sha256 or {}
    assets = gate.get("public_setup_files_sha256") or {}
    candidate_sha = hashlib.sha256(candidate_code.encode()).hexdigest()
    if (gate.get("candidate_raw_sha256") != candidate_sha
            or gate.get("setup_source_sha256") != hashlib.sha256(task.public_setup_py.encode()).hexdigest()
            or assets.get("/holdout_task/public_batch.pkl") != public_sha
            or gate.get("readonly_dependencies_sha256") != dependencies
            or any(assets.get(k) != v for k, v in dependencies.items())
            or (tier != "SCHEMA_FAILURE" and not joint_scientific_receipt_matches(gate, candidate_sha, public_sha, numeric_sha))):
        return {**gate, "accepted": False, "setup_ok": False, "infrastructure_error": "unbound_public_probe"}
    return gate


def _record_direct_execution(controller, binding, diagnostic, features, public_sha, numeric_sha, setup_sha, dependencies_sha):
    return controller.record_tested_witness_from_probe_receipt(
        binding=binding, probe_receipt={
            **diagnostic, "public_input_sha256": public_sha, "canonical_numeric_sha256": numeric_sha,
            "setup_source_sha256": setup_sha, "dependency_bundle_sha256": dependencies_sha,
        },
        public_input_sha256=public_sha, features=features, evidence_origin="direct_component_probe",
        canonical_numeric_sha256=numeric_sha, setup_source_sha256=setup_sha, dependency_bundle_sha256=dependencies_sha,
    )


def _evaluate_empirical_coverage_m100(
    *,
    family: str,
    new_id: str,
    new_candidate_builder: Any,
    old_id: str,
    old_solver_sha: str,
    old_parent_vers: dict[str, int],
    old_witness_specs: list[tuple[str, Any, dict[str, Any], dict[str, str], dict[str, str]]],
) -> EmpiricalCoverageReceipt:
    if not old_witness_specs:
        rec = EmpiricalCoverageReceipt(
            receipt_id="",
            new_binding_id=new_id,
            new_solver_sha256="0" * 64,
            old_binding_id=old_id,
            old_solver_sha256=old_solver_sha,
            witness_ids=[],
            parameter_versions_by_witness=[],
            verifier_version=SHARED_KERNEL_VERSION,
            status="unknown",
            passed_witness_count=0,
            evaluated_witness_count=0,
            total_cpu_probe_wall_ms=0.0,
            mean_probe_wall_ms_ratio=1.0,
            witness_input_sha256s=[],
            parameter_payload_sha256s=[],
        )
        rec.receipt_id = rec.compute_expected_receipt_id()
        return rec

    passed = 0
    tot_ms = 0.0
    w_ids: list[str] = []
    p_vers: list[dict[str, int]] = []
    in_shas: list[str] = []
    pay_shas: list[str] = []
    new_sha = "0" * 64
    has_rejection = False
    setup_shas, numeric_shas, dependency_shas, probe_ids = [], [], [], []
    req_coupon = family == "alpha"

    for w_id, w_task, w_pay, w_ro_deps, w_ro_shas in old_witness_specs:
        cand_txt, raw_sha = new_candidate_builder(w_task, w_pay)
        new_sha = raw_sha
        item, p_sha = materialize_m99_public_item_and_sha(w_task)
        t0 = monotonic()
        pr = _execute_bound_gate(
            task=w_task,
            candidate_code=cand_txt,
            readonly_dependencies=w_ro_deps,
            expected_dependency_sha256=w_ro_shas,
            require_pooled_coupon_check=req_coupon,
        )
        dt_ms = (monotonic() - t0) * 1000.0
        tot_ms += dt_ms
        w_ids.append(w_id)
        p_vers.append(dict(old_parent_vers))
        in_shas.append(p_sha)
        setup_shas.append(hashlib.sha256(w_task.public_setup_py.encode()).hexdigest())
        numeric_shas.append(compute_canonical_numeric_item_sha256(item))
        dependency_shas.append(hashlib.sha256(json.dumps(w_ro_shas, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
        probe_ids.append(f"cprobe_{uuid.uuid4().hex[:16]}")
        pay_shas.append(hashlib.sha256(json.dumps(w_pay, sort_keys=True).encode("utf-8")).hexdigest())
        if pr.get("accepted"):
            passed += 1
        elif classify_public_probe_failure(pr)["tier"] in {"SCIENTIFIC_FAILURE", "DOMAIN_REJECTION", "SCHEMA_FAILURE"}:
            has_rejection = True

    ev_cnt = len(old_witness_specs)
    status = "refuted" if has_rejection else ("supported_on_witnesses" if ev_cnt and passed == ev_cnt else "unknown")
    rec = EmpiricalCoverageReceipt(
        receipt_id="",
        new_binding_id=new_id,
        new_solver_sha256=new_sha,
        old_binding_id=old_id,
        old_solver_sha256=old_solver_sha,
        witness_ids=w_ids,
        parameter_versions_by_witness=p_vers,
        verifier_version=SHARED_KERNEL_VERSION,
        status=status,
        passed_witness_count=passed,
        evaluated_witness_count=ev_cnt,
        total_cpu_probe_wall_ms=round(tot_ms, 2),
        mean_probe_wall_ms_ratio=1.0,
        witness_input_sha256s=in_shas,
        parameter_payload_sha256s=pay_shas, receipt_schema_version=2,
        witness_setup_source_sha256s=setup_shas, witness_canonical_numeric_sha256s=numeric_shas,
        dependency_bundle_sha256s=dependency_shas, probe_ids=probe_ids,
    )
    rec.receipt_id = rec.compute_expected_receipt_id()
    return rec


def _dn_obs_distance(a: dict[str, float | None], b: dict[str, float | None]) -> float:
    diffs = [
        abs(float(a[k]) - float(b[k]))
        for k in M100_DOWNSTREAM_OBSERVABLE_NAMES
        if a.get(k) is not None and b.get(k) is not None
    ]
    return max(diffs) if diffs else 999.0


def _replay_one_matching_monolith(pool, observations, task, deps, shas, require_coupon):
    """One raw archive attempt on the target, shared by both modular arms before paid extraction."""
    matched = next((row for row in pool if not row["projected"]
                    and _dn_obs_distance(row["dn_obs"], observations) <= 0.35), None)
    if matched is None:
        return None, None
    code = matched["code"]
    sha = hashlib.sha256(code.encode()).hexdigest()
    source_gate = matched["source_gate_receipt"]
    if not (source_gate.get("accepted") and joint_scientific_receipt_matches(
            source_gate, sha, matched["pub_sha"], matched["canon_num_sha"])):
        raise ValueError("Unfactored monolith does not match its verified source program and input")
    start = monotonic()
    try:
        probe = _execute_bound_gate(task=task, candidate_code=code, readonly_dependencies=deps,
                                    expected_dependency_sha256=shas, require_pooled_coupon_check=require_coupon)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        probe = {"accepted": False, "setup_ok": False, "timed_out": isinstance(exc, subprocess.TimeoutExpired),
                 "infrastructure_error": str(exc)}
    classification = classify_public_probe_failure(probe)
    receipt = {"source_task_id": matched["task"].task_id, "source_input_sha256": matched["pub_sha"],
               "source_numeric_sha256": matched["canon_num_sha"], "candidate_sha256": sha,
               "target_task_id": task.task_id, "probe_result": probe, **classification,
               "wall_ms": round((monotonic() - start) * 1000, 2)}
    return matched, receipt


def _run_stage_s_shared_source(
    *,
    spec: M100WorldSpec,
    s_task: Any,
    s0_prefix_logs: list[dict[str, Any]],
    base_v7_forest: Any,
    base_v7_archive: Any,
    codex_info: dict[str, Any],
    model: str,
    reasoning_effort: str,
    out_dir: Path,
    run_id: str,
    max_turns: int,
) -> dict[str, Any]:
    """Synthesize unfactored monolithic `Stage S` (`(U0, D0)`) ONCE per cluster (`0` host-seeded `u0/d0`!)."""
    wid = spec.world_id
    s_dir = out_dir / "stage_S_shared_sources" / wid
    s_dir.mkdir(parents=True, exist_ok=True)
    ro_deps = {"/tmp/hegel_continual_archive/archive_apparatus_log.json": json.dumps(s0_prefix_logs, indent=2)}
    ro_shas = {k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in ro_deps.items()}
    aug_s, ctx_s = _inject_equal_info_archive_into_m100_task(s_task, [], s0_prefix_logs, None)
    ft_s = m100_task_to_frontier_task(aug_s, continual_context_block=ctx_s, allowed_readonly_dependencies=tuple(sorted(ro_deps.keys())))

    _CALL_CONTEXT.role = "stage_S_solve"
    _CALL_CONTEXT.world_id = wid
    _CALL_CONTEXT.cluster_id = spec.cluster_id
    _CALL_CONTEXT.arm = "Stage_S_Shared_Source"
    _CALL_CONTEXT.episode_short_name = s_task.short_name

    forest = deepcopy(base_v7_forest)
    archive = deepcopy(base_v7_archive)
    ep_sum = m6f.run_frontier_episode(
        ft_s,
        "Hegel_M6_9_Antithesis_LiveGPT",
        model=model,
        codex_info=codex_info,
        forest=forest,
        forest_archive=archive,
        hippo_idx=HippoRAGWormholeIndex(forest),
        episode_out_dir=s_dir,
        run_id=f"{run_id}__stageS__{wid}",
        max_turns=max_turns,
        cmd_timeout_sec=180,
        reasoning_effort=reasoning_effort,
        defer_official_verifier=True,
        enable_auto_skill_packaging=False,
        allow_l1_primitives=False,
        disclosure_mode="on_demand_executable",
        staging_policy="net_value_gated",
    )
    sub = _extract_submitted_files(s_dir)
    s_code = sub.get("batch_solver.py", "")
    pub_gate = (
        _execute_bound_gate(
            task=s_task,
            candidate_code=s_code,
            readonly_dependencies=ro_deps,
            expected_dependency_sha256=ro_shas,
            require_pooled_coupon_check=(spec.family == "alpha"),
        )
        if s_code
        else {"accepted": False}
    )
    priv_ver = (
        grade_m94_artifact_map(
            s_task, {str(s_task.declared_artifacts[0]): s_code}, readonly_dependencies=ro_deps, expected_dependency_sha256=ro_shas
        )
        if s_code
        else {"strict_pass": False}
    )
    s_pub_item, s_pub_sha = materialize_m99_public_item_and_sha(s_task)
    source_sha = hashlib.sha256(s_code.encode()).hexdigest()
    admitted = bool(pub_gate.get("accepted") and joint_scientific_receipt_matches(
        pub_gate, source_sha, s_pub_sha, compute_canonical_numeric_item_sha256(s_pub_item)))
    (s_dir / "stage_s_public_gate.json").write_text(json.dumps(pub_gate, indent=2) + "\n")
    (s_dir / "03_official_verifier_report.json").write_text(json.dumps(priv_ver, indent=2) + "\n")
    uni_keys = expected_m98_public_output_keys(spec.family)
    s_ent = make_continual_archive_entry(
        short_name=s_task.short_name,
        task_id=s_task.task_id,
        canonical_entrypoint="solve_batch",
        canonical_algorithm_code=s_code,
        output_schema_keys=uni_keys,
        public_qualification_status=("public_gate_verified" if admitted else "public_gate_failed"),
    )
    rec = {
        "world_id": wid,
        "cluster_id": spec.cluster_id,
        "family": spec.family,
        "order_mode": spec.order_mode,
        "short_name": s_task.short_name,
        "turns_used": int(ep_sum.get("turns_used", 0) or 0),
        "public_gate_accepted": bool(pub_gate.get("accepted")),
        "admitted": admitted, "admission_basis": "bound_public_gate",
        "strict_pass": bool(priv_ver.get("strict_pass")),
        "source_code_sha256": hashlib.sha256(s_code.encode("utf-8")).hexdigest(),
        "public_input_sha256": s_pub_sha,
    }
    (s_dir / "00_stage_S_summary.json").write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    print(
        f"[M10.0 Stage S | {wid} ({spec.family}, {spec.order_mode})] pass={rec['strict_pass']} | pub_gate={rec['public_gate_accepted']} | turns={rec['turns_used']}",
        flush=True,
    )
    return {
        "summary": rec,
        "source_monolithic_code": s_code,
        "source_archive_entry": s_ent,
        "source_pub_item": s_pub_item,
        "source_pub_sha256": s_pub_sha,
    }


def _run_one_m100_world_arm_stream(
    *,
    spec: M100WorldSpec,
    arm: str,
    s_task: Any,
    stream_tasks: tuple[Any, ...],
    schedule_manifest: list[dict[str, Any]],
    apparatus_logs: list[dict[str, Any]],
    base_v7_forest: Any,
    base_v7_archive: Any,
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
    wid = spec.world_id
    fam = spec.family
    req_coupon = fam == "alpha"
    compat_spec = spec.to_m98_compat_spec()
    uni_keys = expected_m98_public_output_keys(fam)
    stream_t0 = monotonic()

    forest = deepcopy(base_v7_forest)
    archive = deepcopy(base_v7_archive)
    prior_entries: list[dict[str, Any]] = [deepcopy(source_archive_entry)]
    flat_snap: dict[str, Any] = {"gains": [1.0, 1.0, 1.0, 1.0], "offsets": [0.0, 0.0, 0.0, 0.0]}

    s0_prefix_logs = apparatus_logs[:1]
    s0_ro_deps = {"/tmp/hegel_continual_archive/archive_apparatus_log.json": json.dumps(s0_prefix_logs, indent=2)}
    s0_ro_shas = {k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in s0_ro_deps.items()}
    s0_up_obs = compute_m100_upstream_observables(fam, source_pub_item)
    s0_dn_obs = compute_m98_downstream_observables(s0_prefix_logs)
    s0_comb_obs = {**s0_up_obs, **s0_dn_obs}
    s0_sha = hashlib.sha256(source_monolithic_code.encode("utf-8")).hexdigest()
    s0_canon_num_sha = compute_canonical_numeric_item_sha256(source_pub_item)

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

    # Shared unfactored pre-decomposition monolithic fallback pool (`B3 / Q04`: symmetric across `Plain` and `Hegel`!)
    unfactored_mono_pool: list[dict[str, Any]] = []

    # Arm 2: `Flat_PlainModuleRegistry_Online`
    plain_is_factored = False
    plain_up_table: list[dict[str, Any]] = []
    plain_dn_table: list[dict[str, Any]] = []
    plain_up_pos: dict[str, list[dict[str, float | None]]] = {}
    plain_up_neg: dict[str, list[dict[str, float | None]]] = {}
    plain_dn_pos: dict[str, list[dict[str, float | None]]] = {}
    plain_dn_neg: dict[str, list[dict[str, float | None]]] = {}
    plain_dn_cov_pairs: set[tuple[str, str]] = set()
    plain_dn_rep_wit: dict[str, list[tuple[str, Any, dict[str, Any], dict[str, str], dict[str, str]]]] = {}

    # Arm 3: `Hegel_SelectiveConditionExpansion_Online`
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

    if arm == "Hegel_SelectiveConditionExpansion_Online":
        graph, _, registry = initialize_m98_factored_mechanism_graph(spec=compat_spec, forest=forest)
        up_ctrl = ImmutableBranchLineageController(world_id=f"{wid}_up", family=f"{fam}_up", graph=graph, registry=registry)
        dn_ctrl = ImmutableBranchLineageController(world_id=f"{wid}_dn", family=f"{fam}_dn", graph=graph, registry=registry)
        graph.save_checkpoint(stream_out_dir / "graph_ckpt_after_step_00.json")

    episode_results: list[dict[str, Any]] = []
    coverage_receipts_log: list[dict[str, Any]] = []
    guard_induction_events: list[dict[str, Any]] = []
    decomposition_events: list[dict[str, Any]] = []
    lazy_projection_events: list[dict[str, Any]] = []

    n_steps = len(stream_tasks)
    for step_i, (task, m_info) in enumerate(zip(stream_tasks, schedule_manifest), start=1):
        ep_dir = stream_out_dir / task.task_id
        ep_dir.mkdir(parents=True, exist_ok=True)
        ep_run_tag = f"{run_id}__{wid}__{arm}__step{step_i:02d}"

        prefix_logs = apparatus_logs[: step_i + 1]
        readonly_deps = {"/tmp/hegel_continual_archive/archive_apparatus_log.json": json.dumps(prefix_logs, indent=2)}
        readonly_shas = {k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in readonly_deps.items()}

        pub_item, pub_sha = materialize_m99_public_item_and_sha(task)
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
        up_obs_now = compute_m100_upstream_observables(fam, pub_item)
        dn_obs_now = compute_m98_downstream_observables(prefix_logs)
        comb_obs_now = {**up_obs_now, **dn_obs_now}

        # =====================================================================
        # Step A: Equal-Capability Pre-LLM Public Gate + Symmetric `B2`/`B3` Recovery
        # =====================================================================
        loc_wall_ms = 0.0
        loc_probes_used = 0
        decomp_wall_ms = 0.0
        decomp_probes_used = 0
        withdrawal_this_step: dict[str, Any] | None = None
        lazy_proj_this_step: dict[str, Any] | None = None
        raw_archive_receipt = None

        if arm == "Flat_MonolithicConditionalTable_Online":
            chosen_row, sel_mode, _ = _plain_table_select_row(mono_table, uni_keys, comb_obs_now)
            cand_code = str(chosen_row["canonical_code"])
            t_p0 = monotonic()
            pr = _execute_bound_gate(
                task=task,
                candidate_code=cand_code,
                readonly_dependencies=readonly_deps,
                expected_dependency_sha256=readonly_shas,
                require_pooled_coupon_check=req_coupon,
            )
            pr_ms = round((monotonic() - t_p0) * 1000.0, 2)
            cls = classify_public_probe_failure(pr)
            if cls["tier"] in {"SCIENTIFIC_FAILURE", "DOMAIN_REJECTION"}:
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
                pr = _execute_bound_gate(
                    task=task,
                    candidate_code=cand_code,
                    readonly_dependencies=readonly_deps,
                    expected_dependency_sha256=readonly_shas,
                    require_pooled_coupon_check=req_coupon,
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
                up_r, up_m, alt_up_rows = _plain_table_select_row(plain_up_table, ["calibrated_input"], up_obs_now)
                dn_r, dn_m, _ = _plain_table_select_row(plain_dn_table, uni_keys, dn_obs_now)
                cand_code = compose_m98_two_stage_solver_code(
                    upstream_code=str(up_r["canonical_code"]),
                    upstream_entrypoint=str(up_r["entrypoint"]),
                    downstream_code=str(dn_r["canonical_code"]),
                    downstream_entrypoint=str(dn_r["entrypoint"]),
                    param_payload=flat_snap,
                )
                t_p0 = monotonic()
                pr = _execute_bound_gate(
                    task=task,
                    candidate_code=cand_code,
                    readonly_dependencies=readonly_deps,
                    expected_dependency_sha256=readonly_shas,
                    require_pooled_coupon_check=req_coupon,
                )
                pr_ms = round((monotonic() - t_p0) * 1000.0, 2)
                pre_probes_used = 1
                cls = classify_public_probe_failure(pr)
                locus_str = "none"
                locus_diag: dict[str, Any] = {}
                sibling_fb_used = False

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
                    loc_wall_ms += round(float(locus_diag.get("probe_wall_ms") or 0.0), 2)
                    loc_probes_used += 1 if locus_diag.get("component_probe_executed") else 0
                    locus_str = str(locus_diag["locus"])
                    if locus_str == "upstream_transducer":
                        plain_up_neg.setdefault(up_r["row_id"], []).append(up_obs_now)
                        # `B2` (`Q03` Symmetric `1`-Alternative Verified-Sibling Upstream Fallback):
                        if alt_up_rows:
                            alt_u = alt_up_rows[0]
                            alt_diag = diagnose_m98_factored_conflict_locus(
                                fam,
                                pub_item,
                                chosen_up_code=str(alt_u["canonical_code"]),
                                chosen_up_entrypoint=str(alt_u["entrypoint"]),
                                diag_payload=flat_snap,
                                force_direct_component_probe=True,
                            )
                            loc_wall_ms += round(float(alt_diag.get("probe_wall_ms") or 0.0), 2)
                            loc_probes_used += 1 if alt_diag.get("component_probe_executed") else 0
                            alt_rmse = alt_diag.get("chosen_up_holdout_rmse")
                            if (
                                alt_diag.get("execution_status") == "completed"
                                and alt_rmse is not None
                                and 0.0 <= float(alt_rmse) <= 0.03
                            ):
                                alt_cand = compose_m98_two_stage_solver_code(
                                    upstream_code=str(alt_u["canonical_code"]),
                                    upstream_entrypoint=str(alt_u["entrypoint"]),
                                    downstream_code=str(dn_r["canonical_code"]),
                                    downstream_entrypoint=str(dn_r["entrypoint"]),
                                    param_payload=flat_snap,
                                )
                                t_alt = monotonic()
                                pr_alt = _execute_bound_gate(
                                    task=task,
                                    candidate_code=alt_cand,
                                    readonly_dependencies=readonly_deps,
                                    expected_dependency_sha256=readonly_shas,
                                    require_pooled_coupon_check=req_coupon,
                                )
                                pr_ms = round(pr_ms + (monotonic() - t_alt) * 1000.0, 2)
                                pre_probes_used += 1
                                up_r = alt_u
                                up_m = "sibling_1alt_fallback"
                                sibling_fb_used = True
                                if pr_alt["accepted"]:
                                    pr = pr_alt
                                    cand_code = alt_cand
                                    locus_str = "none"
                                    plain_up_pos.setdefault(alt_u["row_id"], []).append(up_obs_now)
                                    plain_dn_pos.setdefault(dn_r["row_id"], []).append(dn_obs_now)
                                else:
                                    cls_alt = classify_public_probe_failure(pr_alt)
                                    pr = pr_alt
                                    cls = cls_alt
                                    locus_diag = alt_diag
                                    locus_str = "downstream_constitutive" if cls_alt["tier"] != "INFRA_FAILURE" else "unknown_infra"
                                    plain_up_pos.setdefault(alt_u["row_id"], []).append(up_obs_now)
                                    if locus_str == "downstream_constitutive":
                                        plain_dn_neg.setdefault(dn_r["row_id"], []).append(dn_obs_now)
                            else:
                                plain_up_neg.setdefault(alt_u["row_id"], []).append(up_obs_now)
                    elif locus_str == "downstream_constitutive":
                        plain_up_pos.setdefault(up_r["row_id"], []).append(up_obs_now)
                        plain_dn_neg.setdefault(dn_r["row_id"], []).append(dn_obs_now)

                    # `B3` (`Q04` Symmetric Unfactored Monolithic Fallback + Lazy On-Demand Downstream Projection):
                    if (not pr["accepted"]) and locus_str == "downstream_constitutive" and unfactored_mono_pool:
                        matched_mono, raw_archive_receipt = _replay_one_matching_monolith(
                            unfactored_mono_pool, dn_obs_now, task, readonly_deps, readonly_shas, req_coupon,
                        )
                        if raw_archive_receipt is not None:
                            pre_probes_used += 1
                            pr_ms = round(pr_ms + raw_archive_receipt["wall_ms"], 2)
                            raw_gate = raw_archive_receipt["probe_result"]
                            if raw_gate["accepted"]:
                                raw_archive_receipt["preceding_module_gate"] = pr
                                cand_code, pr, cls = matched_mono["code"], raw_gate, classify_public_probe_failure(raw_gate)
                                locus_str = "none"
                            elif raw_archive_receipt["tier"] == "INFRA_FAILURE":
                                locus_str = "unknown_infra"
                        if matched_mono is not None and not pr["accepted"] and locus_str == "downstream_constitutive":
                            proj_client = m6f.FrontierCodexClient(codex_info=codex_info, model=model, reasoning_effort=reasoning_effort, timeout_seconds=300)
                            proj_client._hegel_call_meta = {
                                "role": "lazy_retroactive_projection",
                                "world_id": wid,
                                "cluster_id": spec.cluster_id,
                                "arm": arm,
                                "episode_short_name": task.short_name,
                            }
                            proj_res = propose_and_verify_lazy_retroactive_downstream_projection(
                                client=proj_client,
                                stage_subdir=f"{ep_run_tag}__plain_lazyproj",
                                family=fam,
                                u0_code=str(plain_up_table[0]["canonical_code"]),
                                current_up_code=str(up_r["canonical_code"]),
                                current_up_entrypoint=str(up_r["entrypoint"]),
                                current_pay=flat_snap,
                                prior_monolithic_code=str(matched_mono["code"]),
                                prior_task=matched_mono["task"],
                                prior_pay=matched_mono["pay"],
                                prior_ro_deps=matched_mono["ro_deps"],
                                prior_ro_shas=matched_mono["ro_shas"],
                                current_task=task,
                                current_ro_deps=readonly_deps,
                                current_ro_shas=readonly_shas,
                            )
                            decomp_wall_ms += float(proj_res.get("total_probe_wall_ms", 0.0))
                            lazy_proj_this_step = {k: v for k, v in proj_res.items() if k not in {"provider_call_receipts", "downstream_code", "composed_current_code"}}
                            lazy_projection_events.append({"step": step_i, "event": m_info["causal_event"], **lazy_proj_this_step})
                            if proj_res.get("accepted"):
                                matched_mono["projected"] = True
                                dn_proj_c = str(proj_res["downstream_code"])
                                new_did = f"dn_r_proj_{matched_mono['short_name']}"
                                plain_dn_table.append(
                                    {
                                        "row_id": new_did,
                                        "guard": {},
                                        "canonical_code": dn_proj_c,
                                        "entrypoint": "solve_downstream",
                                        "code_sha256": hashlib.sha256(dn_proj_c.encode()).hexdigest(),
                                        "priority": 12,
                                        "output_schema_keys": list(uni_keys),
                                    }
                                )
                                plain_dn_pos[new_did] = [matched_mono["dn_obs"], dn_obs_now]
                                plain_dn_rep_wit[new_did] = [(f"wit_{matched_mono['short_name']}", matched_mono["task"], matched_mono["pay"], matched_mono["ro_deps"], matched_mono["ro_shas"])]
                                pr = proj_res["current_probe_receipt"]
                                cand_code = str(proj_res["composed_current_code"])
                                dn_r = plain_dn_table[-1]
                                dn_m = "lazy_retroactive_projection"
                                locus_str = "none"

                pre_trial = {
                    "arm": arm,
                    "accepted_pre_llm": bool(pr["accepted"]),
                    "pre_llm_probes_used": pre_probes_used,
                    "probe_wall_ms": pr_ms,
                    "localization_probes_used": loc_probes_used,
                    "localization_probe_wall_ms": round(loc_wall_ms, 2),
                    "sibling_fallback_used": sibling_fb_used,
                    "lazy_projection_receipt": lazy_proj_this_step,
                    "raw_archive_replay": raw_archive_receipt,
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
            # Arm 3: `Hegel_SelectiveConditionExpansion_Online`
            assert graph is not None and registry is not None and up_ctrl is not None and dn_ctrl is not None
            if not hegel_is_factored:
                cand_code = source_monolithic_code
                t_p0 = monotonic()
                pr = _execute_bound_gate(
                    task=task,
                    candidate_code=cand_code,
                    readonly_dependencies=readonly_deps,
                    expected_dependency_sha256=readonly_shas,
                    require_pooled_coupon_check=req_coupon,
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
                all_up_verified = sorted(
                    [b for b in registry.bindings.values() if b.chain_id == f"{fam}_up" and b.qualification_status == "verified"],
                    key=lambda x: (-int(x.selection_priority), x.binding_id),
                )
                up_sel = up_ctrl.select_branch_deterministically(req_schema_keys=["calibrated_input"], feature_evaluator=lambda _b: up_obs_now)
                chosen_up: MechanismExecutableBinding = up_sel["selected"] or all_up_verified[0]
                up_mode = str(up_sel["status"])
                alt_up_bindings = [b for b in all_up_verified if b.binding_id != chosen_up.binding_id]

                all_dn_verified = sorted(
                    [b for b in registry.bindings.values() if b.chain_id == f"{fam}_dn" and b.qualification_status == "verified"],
                    key=lambda x: (-int(x.selection_priority), x.binding_id),
                )
                dn_sel = dn_ctrl.select_branch_deterministically(
                    req_schema_keys=uni_keys,
                    feature_evaluator=lambda b: (
                        comb_obs_now
                        if (b.binding_id in hegel_cross_scoped_dn_bids or _guard_uses_upstream_features(b.applicability_guard))
                        else dn_obs_now
                    ),
                )
                chosen_dn: MechanismExecutableBinding = dn_sel["selected"] or all_dn_verified[0]
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
                pid_main = f"jprobe_{uuid.uuid4().hex[:16]}"
                t_p0 = monotonic()
                pr = _execute_bound_gate(
                    task=task,
                    candidate_code=cand_code,
                    readonly_dependencies=readonly_deps,
                    expected_dependency_sha256=readonly_shas,
                    require_pooled_coupon_check=req_coupon,
                )
                pr_ms = round((monotonic() - t_p0) * 1000.0, 2)
                pre_probes_used = 1
                cls = classify_public_probe_failure(pr)
                locus_str = "none"
                locus_diag = {}
                sibling_fb_used = False

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
                        probe_id=pid_main,
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
                        probe_id=pid_main,
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
                    loc_wall_ms += round(float(locus_diag.get("probe_wall_ms") or 0.0), 2)
                    loc_probes_used += 1 if locus_diag.get("component_probe_executed") else 0
                    locus_str = str(locus_diag["locus"])
                    if locus_str == "upstream_transducer" and locus_diag.get("execution_status") == "completed":
                        _record_direct_execution(
                            up_ctrl, chosen_up, locus_diag, up_obs_now, pub_sha, canon_num_sha, setup_src_sha, dep_bundle_sha,
                        )
                        # `B2` (`Q03` Symmetric `1`-Alternative Verified-Sibling Upstream Fallback):
                        if alt_up_bindings:
                            alt_ub = alt_up_bindings[0]
                            alt_diag = diagnose_m98_factored_conflict_locus(
                                fam,
                                pub_item,
                                chosen_up_code=(alt_ub.canonical_algorithm_code or alt_ub.parameterized_closure_code),
                                chosen_up_entrypoint=(alt_ub.canonical_entrypoint or alt_ub.parameterized_entrypoint),
                                diag_payload=diag_pay,
                                force_direct_component_probe=True,
                            )
                            loc_wall_ms += round(float(alt_diag.get("probe_wall_ms") or 0.0), 2)
                            loc_probes_used += 1 if alt_diag.get("component_probe_executed") else 0
                            alt_rmse = alt_diag.get("chosen_up_holdout_rmse")
                            if (
                                alt_diag.get("execution_status") == "completed"
                                and alt_rmse is not None
                                and 0.0 <= float(alt_rmse) <= 0.03
                            ):
                                _record_direct_execution(
                                    up_ctrl, alt_ub, alt_diag, up_obs_now, pub_sha, canon_num_sha, setup_src_sha, dep_bundle_sha,
                                )
                                alt_cand = compose_m98_two_stage_solver_code(
                                    upstream_code=(alt_ub.canonical_algorithm_code or alt_ub.parameterized_closure_code),
                                    upstream_entrypoint=(alt_ub.canonical_entrypoint or alt_ub.parameterized_entrypoint),
                                    downstream_code=(chosen_dn.canonical_algorithm_code or chosen_dn.parameterized_closure_code),
                                    downstream_entrypoint=(chosen_dn.canonical_entrypoint or chosen_dn.parameterized_entrypoint),
                                    param_payload=diag_pay,
                                    upstream_binding=alt_ub,
                                    downstream_binding=chosen_dn,
                                    graph=graph,
                                    registry=registry,
                                )
                                pid_alt = f"jprobe_{uuid.uuid4().hex[:16]}"
                                t_alt = monotonic()
                                pr_alt = _execute_bound_gate(
                                    task=task,
                                    candidate_code=alt_cand,
                                    readonly_dependencies=readonly_deps,
                                    expected_dependency_sha256=readonly_shas,
                                    require_pooled_coupon_check=req_coupon,
                                )
                                w_alt_ms = round((monotonic() - t_alt) * 1000.0, 2)
                                pr_ms = round(pr_ms + w_alt_ms, 2)
                                pre_probes_used += 1
                                chosen_up = alt_ub
                                up_mode = "sibling_1alt_fallback"
                                sibling_fb_used = True
                                cand_sha = hashlib.sha256(alt_cand.encode("utf-8")).hexdigest()
                                if pr_alt["accepted"]:
                                    pr = pr_alt
                                    cand_code = alt_cand
                                    locus_str = "none"
                                    dn_ctrl.record_tested_witness(
                                        binding=chosen_dn,
                                        public_input_sha256=pub_sha,
                                        passed=True,
                                        failure_kind="none",
                                        features=(comb_obs_now if chosen_dn.binding_id in hegel_cross_scoped_dn_bids else dn_obs_now),
                                        probe_wall_ms=w_alt_ms,
                                        execution_status="completed",
                                        evidence_origin="joint_task_probe",
                                        probe_id=pid_alt,
                                        setup_source_sha256=setup_src_sha,
                                        canonical_numeric_sha256=canon_num_sha,
                                        dependency_bundle_sha256=dep_bundle_sha,
                                    )
                                    hegel_dn_joint_pos_obs.setdefault(chosen_dn.binding_id, []).append(comb_obs_now)
                                else:
                                    cls_alt = classify_public_probe_failure(pr_alt)
                                    pr = pr_alt
                                    cls = cls_alt
                                    pid_main = pid_alt
                                    locus_diag = alt_diag
                                    locus_str = "downstream_constitutive" if cls_alt["tier"] != "INFRA_FAILURE" else "unknown_infra"
                            elif alt_diag.get("execution_status") == "completed":
                                _record_direct_execution(
                                    up_ctrl, alt_ub, alt_diag, up_obs_now, pub_sha, canon_num_sha, setup_src_sha, dep_bundle_sha,
                                )

                    if locus_str == "downstream_constitutive" and locus_diag.get("execution_status") == "completed":
                        if not sibling_fb_used:
                            _record_direct_execution(
                                up_ctrl, chosen_up, locus_diag, up_obs_now, pub_sha, canon_num_sha, setup_src_sha, dep_bundle_sha,
                            )
                        dn_ctrl.record_tested_witness(
                            binding=chosen_dn,
                            public_input_sha256=pub_sha,
                            passed=False,
                            failure_kind=cls["failure_kind"],
                            features=(comb_obs_now if chosen_dn.binding_id in hegel_cross_scoped_dn_bids else dn_obs_now),
                            probe_wall_ms=pr_ms,
                            execution_status="completed",
                            evidence_origin="joint_task_probe",
                            probe_id=pid_main,
                            setup_source_sha256=setup_src_sha,
                            canonical_numeric_sha256=canon_num_sha,
                            dependency_bundle_sha256=dep_bundle_sha,
                        )
                        if cls["tier"] in {"SCIENTIFIC_FAILURE", "DOMAIN_REJECTION"}:
                            hegel_dn_joint_neg_obs.setdefault(chosen_dn.binding_id, []).append(comb_obs_now)

                        sw_rec = issue_measured_withdrawal(
                            world_id=wid, family=fam, downstream_selected=dn_sel["selected"] is not None,
                            upstream=chosen_up, downstream=chosen_dn, downstream_features=dn_obs_now,
                            downstream_ledger=dn_ctrl.witness_ledger, direct=locus_diag, gate=pr,
                            candidate_sha=cand_sha, expanded_feature_names=M100_ALL_OBSERVABLE_NAMES,
                        )
                        if sw_rec is not None:
                            locus_str = "cross_factor_interaction_withdrawal"
                            withdrawal_receipts.append(sw_rec)
                            persist_separability_withdrawals_to_graph(graph, wid, withdrawal_receipts)
                            hegel_cross_scoped_dn_bids.add(chosen_dn.binding_id)
                            withdrawal_this_step = sw_rec.to_dict()

                    # `B3` (`Q04` Symmetric Unfactored Monolithic Fallback + Lazy On-Demand Downstream Projection):
                    if (not pr["accepted"]) and locus_str == "downstream_constitutive" and unfactored_mono_pool:
                        matched_mono, raw_archive_receipt = _replay_one_matching_monolith(
                            unfactored_mono_pool, dn_obs_now, task, readonly_deps, readonly_shas, req_coupon,
                        )
                        if raw_archive_receipt is not None:
                            pre_probes_used += 1
                            pr_ms = round(pr_ms + raw_archive_receipt["wall_ms"], 2)
                            raw_gate = raw_archive_receipt["probe_result"]
                            if raw_gate["accepted"]:
                                raw_archive_receipt["preceding_module_gate"] = pr
                                cand_code, pr, cls = matched_mono["code"], raw_gate, classify_public_probe_failure(raw_gate)
                                locus_str = "none"
                            elif raw_archive_receipt["tier"] == "INFRA_FAILURE":
                                locus_str = "unknown_infra"
                        if matched_mono is not None and not pr["accepted"] and locus_str == "downstream_constitutive":
                            proj_client = m6f.FrontierCodexClient(codex_info=codex_info, model=model, reasoning_effort=reasoning_effort, timeout_seconds=300)
                            proj_client._hegel_call_meta = {
                                "role": "lazy_retroactive_projection",
                                "world_id": wid,
                                "cluster_id": spec.cluster_id,
                                "arm": arm,
                                "episode_short_name": task.short_name,
                            }
                            u0_b = registry.bindings[f"bind_{wid}_up_b00"]
                            proj_res = propose_and_verify_lazy_retroactive_downstream_projection(
                                client=proj_client,
                                stage_subdir=f"{ep_run_tag}__hegel_lazyproj",
                                family=fam,
                                u0_code=(u0_b.canonical_algorithm_code or u0_b.parameterized_closure_code),
                                current_up_code=(chosen_up.canonical_algorithm_code or chosen_up.parameterized_closure_code),
                                current_up_entrypoint=(chosen_up.canonical_entrypoint or chosen_up.parameterized_entrypoint),
                                current_pay=diag_pay,
                                prior_monolithic_code=str(matched_mono["code"]),
                                prior_task=matched_mono["task"],
                                prior_pay=matched_mono["pay"],
                                prior_ro_deps=matched_mono["ro_deps"],
                                prior_ro_shas=matched_mono["ro_shas"],
                                current_task=task,
                                current_ro_deps=readonly_deps,
                                current_ro_shas=readonly_shas,
                            )
                            decomp_wall_ms += float(proj_res.get("total_probe_wall_ms", 0.0))
                            lazy_proj_this_step = {k: v for k, v in proj_res.items() if k not in {"provider_call_receipts", "downstream_code", "composed_current_code"}}
                            lazy_projection_events.append({"step": step_i, "event": m_info["causal_event"], **lazy_proj_this_step})
                            if proj_res.get("accepted"):
                                matched_mono["projected"] = True
                                dn_proj_c = str(proj_res["downstream_code"])
                                new_d_proj = dn_ctrl.mint_immutable_conflict_branch(
                                    failed_binding_id=chosen_dn.binding_id,
                                    new_canonical_code=dn_proj_c,
                                    new_entrypoint="solve_downstream",
                                    forest_node_id=source_forest_nid,
                                    qualification_protocol="hegel_public_scientific_gate_v4",
                                    output_schema_keys=uni_keys,
                                    validated_parent_ids=["constitutive_root_dn"],
                                    new_applicability_guard={},
                                    updated_failed_branch_applicability_guard=None,
                                    violated_component_id="lazy_retroactive_projected_downstream",
                                    preserved_component_ids=["upstream_transducer"],
                                    selection_priority=12,
                                    environment_digest=env_image_digest,
                                    evidence_digest=hashlib.sha256(dn_proj_c.encode()).hexdigest(),
                                )
                                dn_ctrl.record_tested_witness_from_probe_receipt(
                                    binding=new_d_proj,
                                    probe_receipt=proj_res["prior_probe_receipt"],
                                    public_input_sha256=matched_mono["pub_sha"],
                                    features=matched_mono["dn_obs"],
                                    evidence_origin="joint_task_probe",
                                    canonical_numeric_sha256=matched_mono["canon_num_sha"],
                                )
                                dn_ctrl.record_tested_witness_from_probe_receipt(
                                    binding=new_d_proj,
                                    probe_receipt=proj_res["current_probe_receipt"],
                                    public_input_sha256=pub_sha,
                                    features=dn_obs_now,
                                    evidence_origin="joint_task_probe",
                                    canonical_numeric_sha256=canon_num_sha,
                                )
                                hegel_dn_joint_pos_obs[new_d_proj.binding_id] = [matched_mono["comb_obs"], comb_obs_now]
                                hegel_dn_rep_wit[new_d_proj.binding_id] = [
                                    (f"wit_{matched_mono['short_name']}", matched_mono["task"], matched_mono["pay"], matched_mono["ro_deps"], matched_mono["ro_shas"])
                                ]
                                pr = proj_res["current_probe_receipt"]
                                cand_code = str(proj_res["composed_current_code"])
                                chosen_dn = new_d_proj
                                dn_mode = "lazy_retroactive_projection"
                                locus_str = "none"

                pre_trial = {
                    "arm": arm,
                    "accepted_pre_llm": bool(pr["accepted"]),
                    "pre_llm_probes_used": pre_probes_used,
                    "probe_wall_ms": pr_ms,
                    "localization_probes_used": loc_probes_used,
                    "localization_probe_wall_ms": round(loc_wall_ms, 2),
                    "sibling_fallback_used": sibling_fb_used,
                    "lazy_projection_receipt": lazy_proj_this_step,
                    "raw_archive_replay": raw_archive_receipt,
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
        # Step B: Zero-Turn Closure (or Lazy-Projected Closure) vs Live Episode Synthesis
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
            used_lazy_llm = lazy_proj_this_step is not None
            raw_reuse = bool(raw_archive_receipt and raw_archive_receipt["probe_result"]["accepted"])
            ep_sum = {
                "world_id": wid,
                "cluster_id": spec.cluster_id,
                "family": fam,
                "order_mode": spec.order_mode,
                "arm": arm,
                "stream_step": step_i,
                "causal_event": m_info["causal_event"],
                "true_regime": m_info["true_regime"],
                "event_category": m_info["event_category"],
                "task_id": task.task_id,
                "short_name": task.short_name,
                "execution_mode": ("lazy_retroactive_projection_reuse" if used_lazy_llm else
                                   "zero_llm_raw_archive_reuse" if raw_reuse else "zero_llm_public_gate_verified_reuse"),
                "turns_used": (int(lazy_proj_this_step.get("attempts_used", 1)) if used_lazy_llm else 0),
                "ended_by_finish": True,
                "strict_pass": bool(priv_ver.get("strict_pass")),
                "reward": 1.0 if priv_ver.get("strict_pass") else 0.0,
                "pre_llm_probe_wall_ms": float(pre_trial.get("probe_wall_ms", 0.0)),
                "post_llm_probe_wall_ms": 0.0,
                "localization_probe_wall_ms": round(loc_wall_ms, 2),
                "decomposition_probe_wall_ms": round(decomp_wall_ms, 2),
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
                f"  [M10.0 {wid} | {arm} | step {step_i}/{n_steps} ({m_info['causal_event']})] "
                f"{ep_sum['execution_mode']} | pass={ep_sum['strict_pass']}",
                flush=True,
            )
            continue

        is_modular_arm = arm in {"Flat_PlainModuleRegistry_Online", "Hegel_SelectiveConditionExpansion_Online"}
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
        aug_task, ctx_hdr = _inject_equal_info_archive_into_m100_task(base_synth_task, prior_entries, prefix_logs, diag_for_prompt)
        ft = m100_task_to_frontier_task(
            aug_task,
            continual_context_block=ctx_hdr,
            allowed_readonly_dependencies=tuple(sorted(readonly_deps.keys())),
        )

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

        pid_post = f"pprobe_{uuid.uuid4().hex[:16]}"
        t_pg = monotonic()
        pub_gate_post = (
            _execute_bound_gate(
                task=task,
                candidate_code=graded_cand_txt,
                readonly_dependencies=readonly_deps,
                expected_dependency_sha256=readonly_shas,
                require_pooled_coupon_check=req_coupon,
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
        # Step C: Online Consolidation (`Monolithic` vs `Plain` vs `Hegel`)
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
                    cov_r = _evaluate_empirical_coverage_m100(
                        family=fam,
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
                        ind_res = propose_and_validate_m99_guard_partition(
                            client=comp_client,
                            stage_subdir=f"{ep_run_tag}__mono_gind",
                            scope_label="monolithic_conditional_table_15feat",
                            feature_names=M100_ALL_OBSERVABLE_NAMES,
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
                    decomp_wall_ms += float(dec_res.get("total_probe_wall_ms", 0.0))
                    decomp_probes_used = len(dec_res.get("verification_probes") or [])
                    decomp_receipt_summary = {
                        k: v
                        for k, v in dec_res.items()
                        if k not in {"provider_call_receipts", "upstream_code", "downstream_d0_code", "downstream_d1_code"}
                    }
                    decomposition_events.append({"step": step_i, "event": m_info["causal_event"], **decomp_receipt_summary})
                    if not dec_res.get("accepted"):
                        # `B3 / Q04`: Retain verified unfactored monolithic solution so it is never lost if decomposition defers!
                        unfactored_mono_pool.append(
                            {
                                "short_name": task.short_name,
                                "code": batch_mod_txt,
                                "source_gate_receipt": deepcopy(pub_gate_post),
                                "dn_obs": dn_obs_now,
                                "comb_obs": comb_obs_now,
                                "task": task,
                                "pay": deepcopy(flat_snap),
                                "pub_sha": pub_sha,
                                "canon_num_sha": canon_num_sha,
                                "ro_deps": readonly_deps,
                                "ro_shas": readonly_shas,
                                "projected": False,
                            }
                        )
                    else:
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
                            ind_res = propose_and_validate_m99_guard_partition(
                                client=comp_client,
                                stage_subdir=f"{ep_run_tag}__plain_d0d1_gind",
                                scope_label="plain_downstream_table_8feat",
                                feature_names=M100_DOWNSTREAM_OBSERVABLE_NAMES,
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
                    new_uid = f"up_r{step_i:02d}_{m_info['causal_event']}"
                    ex_uids = [r["row_id"] for r in plain_up_table]
                    pos_u = {rid: list(plain_up_pos.get(rid, [])) for rid in ex_uids}
                    pos_u[new_uid] = [up_obs_now]
                    ind_res = propose_and_validate_m99_guard_partition(
                        client=comp_client,
                        stage_subdir=f"{ep_run_tag}__plain_up_gind",
                        scope_label="plain_upstream_table_7feat",
                        feature_names=M100_UPSTREAM_OBSERVABLE_NAMES,
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
                    else:
                        plain_up_table.append(
                            {
                                "row_id": new_uid,
                                "guard": {},
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
                    u_cur_code = str(pre_trial["selected_upstream_code"])
                    all_cov = True
                    for old_d in plain_dn_table:
                        rep_w = plain_dn_rep_wit.get(old_d["row_id"], [])[:1]
                        cov_r = _evaluate_empirical_coverage_m100(
                            family=fam,
                            new_id=new_did,
                            new_candidate_builder=lambda _wt, wp: (
                                compose_m98_two_stage_solver_code(
                                    upstream_code=u_cur_code,
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
                            ind_res = propose_and_validate_m99_guard_partition(
                                client=comp_client,
                                stage_subdir=f"{ep_run_tag}__plain_dn_gind",
                                scope_label="plain_downstream_table_8feat",
                                feature_names=M100_DOWNSTREAM_OBSERVABLE_NAMES,
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

            elif arm == "Hegel_SelectiveConditionExpansion_Online" and graph is not None and registry is not None and up_ctrl is not None and dn_ctrl is not None:
                ev_sha = hashlib.sha256(json.dumps(pub_gate_post.get("scientific_probe_receipt") or {}, sort_keys=True).encode("utf-8")).hexdigest()
                if not hegel_is_factored:
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
                    decomp_wall_ms += float(dec_res.get("total_probe_wall_ms", 0.0))
                    decomp_probes_used = len(dec_res.get("verification_probes") or [])
                    decomp_receipt_summary = {
                        k: v
                        for k, v in dec_res.items()
                        if k not in {"provider_call_receipts", "upstream_code", "downstream_d0_code", "downstream_d1_code"}
                    }
                    decomposition_events.append({"step": step_i, "event": m_info["causal_event"], **decomp_receipt_summary})
                    if not dec_res.get("accepted"):
                        unfactored_mono_pool.append(
                            {
                                "short_name": task.short_name,
                                "code": batch_mod_txt,
                                "source_gate_receipt": deepcopy(pub_gate_post),
                                "dn_obs": dn_obs_now,
                                "comb_obs": comb_obs_now,
                                "task": task,
                                "pay": deepcopy(diag_pay_now),
                                "pub_sha": pub_sha,
                                "canon_num_sha": canon_num_sha,
                                "ro_deps": readonly_deps,
                                "ro_shas": readonly_shas,
                                "projected": False,
                            }
                        )
                    else:
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
                            ind_res = propose_and_validate_m99_guard_partition(
                                client=comp_client,
                                stage_subdir=f"{ep_run_tag}__hegel_d0d1_gind",
                                scope_label="hegel_downstream_layer_8feat",
                                feature_names=M100_DOWNSTREAM_OBSERVABLE_NAMES,
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
                        if e1_d0_h.get("execution_status") in {"completed", "timeout", "infrastructure_error"}:
                            dn_ctrl.record_tested_witness_from_probe_receipt(
                                binding=b_d0,
                                probe_receipt=e1_d0_h,
                                public_input_sha256=pub_sha,
                                features=dn_obs_now,
                                evidence_origin="joint_task_probe",
                                setup_source_sha256=setup_src_sha,
                                canonical_numeric_sha256=canon_num_sha,
                                dependency_bundle_sha256=dep_bundle_sha,
                            )
                        dn_ctrl.record_tested_witness_from_probe_receipt(
                            binding=b_d1,
                            probe_receipt=e1_d1_h,
                            public_input_sha256=pub_sha,
                            features=dn_obs_now,
                            evidence_origin="joint_task_probe",
                            setup_source_sha256=setup_src_sha,
                            canonical_numeric_sha256=canon_num_sha,
                            dependency_bundle_sha256=dep_bundle_sha,
                        )
                        if s0_d1_h.get("execution_status") in {"completed", "timeout", "infrastructure_error"}:
                            dn_ctrl.record_tested_witness_from_probe_receipt(
                                binding=b_d1,
                                probe_receipt=s0_d1_h,
                                public_input_sha256=source_pub_sha256,
                                features=s0_dn_obs,
                                evidence_origin="joint_task_probe",
                                canonical_numeric_sha256=s0_canon_num_sha,
                            )
                        hegel_dn_joint_pos_obs[bid_d0] = [s0_comb_obs] + ([comb_obs_now] if e1_d0_h.get("accepted") else [])
                        hegel_dn_joint_pos_obs[bid_d1] = [comb_obs_now] + ([s0_comb_obs] if s0_d1_h.get("accepted") else [])
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
                    ind_res = propose_and_validate_m99_guard_partition(
                        client=comp_client,
                        stage_subdir=f"{ep_run_tag}__hegel_up_gind",
                        scope_label="hegel_upstream_layer_7feat",
                        feature_names=M100_UPSTREAM_OBSERVABLE_NAMES,
                        existing_branch_ids=[b.binding_id for b in ex_ups],
                        new_branch_id_placeholder=ph_up,
                        positive_witness_features_by_branch=pos_up,
                        negative_witness_features_by_branch=neg_up,
                    )
                    ind_receipt = {k: v for k, v in ind_res.items() if k != "provider_call_receipts"}
                    guard_induction_events.append({"step": step_i, "event": m_info["causal_event"], "layer": "hegel_upstream", **ind_receipt})
                    new_u_guard = dict(ind_res["new_branch_guard"]) if ind_res.get("accepted") else {}
                    new_u = up_ctrl.mint_immutable_conflict_branch(
                        failed_binding_id=failed_up_id,
                        new_canonical_code=up_mod_txt,
                        new_entrypoint="calibrate_upstream",
                        forest_node_id=source_forest_nid,
                        qualification_protocol="hegel_public_scientific_gate_v4",
                        output_schema_keys=["calibrated_input"],
                        validated_parent_ids=["cal_node_p0", "cal_node_p1"],
                        new_applicability_guard=new_u_guard,
                        updated_failed_branch_applicability_guard=None,
                        violated_component_id="upstream_transducer_shift",
                        preserved_component_ids=["downstream_constitutive_laws"],
                        selection_priority=20 + step_i,
                        environment_digest=env_image_digest,
                        evidence_digest=ev_sha,
                    )
                    if ind_res.get("accepted"):
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
                        probe_wall_ms=post_pub_ms,
                        execution_status="completed",
                        evidence_origin="joint_task_probe",
                        probe_id=pid_post,
                        setup_source_sha256=setup_src_sha,
                        canonical_numeric_sha256=canon_num_sha,
                        dependency_bundle_sha256=dep_bundle_sha,
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
                        cov_r = _evaluate_empirical_coverage_m100(
                            family=fam,
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
                        ind_res = propose_and_validate_m99_guard_partition(
                            client=comp_client,
                            stage_subdir=f"{ep_run_tag}__hegel_cross_withdrawal_gind",
                            scope_label="selective_cross_factor_interaction_15feat",
                            feature_names=M100_ALL_OBSERVABLE_NAMES,
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
                                probe_wall_ms=post_pub_ms,
                                execution_status="completed",
                                evidence_origin="joint_task_probe",
                                probe_id=pid_post,
                                setup_source_sha256=setup_src_sha,
                                canonical_numeric_sha256=canon_num_sha,
                                dependency_bundle_sha256=dep_bundle_sha,
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
                            probe_wall_ms=post_pub_ms,
                            execution_status="completed",
                            evidence_origin="joint_task_probe",
                            probe_id=pid_post,
                            setup_source_sha256=setup_src_sha,
                            canonical_numeric_sha256=canon_num_sha,
                            dependency_bundle_sha256=dep_bundle_sha,
                        )
                        hegel_dn_joint_pos_obs[new_d.binding_id] = [comb_obs_now]
                        hegel_dn_rep_wit[new_d.binding_id] = [
                            (f"wit_{task.short_name}", task, assemble_m98_graph_diagonal_payload(compat_spec, graph), readonly_deps, readonly_shas)
                        ]
                    else:
                        use_joint_15 = any(b.binding_id in hegel_cross_scoped_dn_bids for b in ex_dns)
                        pos_dn = (
                            {**{b.binding_id: list(hegel_dn_joint_pos_obs.get(b.binding_id, [])) for b in ex_dns}, ph_dn: [comb_obs_now]}
                            if use_joint_15
                            else {**{b.binding_id: [w.public_structural_features for w in dn_ctrl.positive_witnesses_for(b.binding_id)] for b in ex_dns}, ph_dn: [dn_obs_now]}
                        )
                        neg_dn = (
                            {b.binding_id: list(hegel_dn_joint_neg_obs.get(b.binding_id, [])) for b in ex_dns}
                            if use_joint_15
                            else {b.binding_id: [w.public_structural_features for w in dn_ctrl.scientific_negative_witnesses_for(b.binding_id)] for b in ex_dns}
                        )
                        cov_pairs_dn = {(cr.new_binding_id, cr.old_binding_id) for cr in dn_ctrl.coverage_receipts if cr.status == "supported_on_witnesses"}
                        ind_res = propose_and_validate_m99_guard_partition(
                            client=comp_client,
                            stage_subdir=f"{ep_run_tag}__hegel_dn_gind",
                            scope_label=("hegel_downstream_layer_15feat" if use_joint_15 else "hegel_downstream_layer_8feat"),
                            feature_names=(M100_ALL_OBSERVABLE_NAMES if use_joint_15 else M100_DOWNSTREAM_OBSERVABLE_NAMES),
                            existing_branch_ids=[b.binding_id for b in ex_dns],
                            new_branch_id_placeholder=ph_dn,
                            positive_witness_features_by_branch=pos_dn,
                            negative_witness_features_by_branch=neg_dn,
                            covered_branch_pairs=cov_pairs_dn,
                        )
                        ind_receipt = {k: v for k, v in ind_res.items() if k != "provider_call_receipts"}
                        guard_induction_events.append({"step": step_i, "event": m_info["causal_event"], "layer": "hegel_downstream", **ind_receipt})
                        if ind_res.get("accepted"):
                            new_g_dn = dict(ind_res["new_branch_guard"])
                            val_pids = ["constitutive_root_dn", "cal_node_p0", "cal_node_p1"] if _guard_uses_upstream_features(new_g_dn) else ["constitutive_root_dn"]
                            new_d = dn_ctrl.mint_immutable_conflict_branch(
                                failed_binding_id=failed_dn_id,
                                new_canonical_code=dn_mod_txt,
                                new_entrypoint="solve_downstream",
                                forest_node_id=source_forest_nid,
                                qualification_protocol="hegel_public_scientific_gate_v4",
                                output_schema_keys=uni_keys,
                                validated_parent_ids=val_pids,
                                new_applicability_guard=new_g_dn,
                                updated_failed_branch_applicability_guard=None,
                                violated_component_id="constitutive_regime_shift",
                                preserved_component_ids=["upstream_transducer"],
                                selection_priority=10 + step_i,
                                environment_digest=env_image_digest,
                                evidence_digest=ev_sha,
                            )
                            if _guard_uses_upstream_features(new_g_dn):
                                hegel_cross_scoped_dn_bids.add(new_d.binding_id)
                            for ex_bid, ex_g in (ind_res.get("existing_branch_guards") or {}).items():
                                if ex_bid in registry.bindings:
                                    registry.bindings[ex_bid].applicability_guard = deepcopy(ex_g)
                                    registry.bindings[ex_bid].precondition_guard = deepcopy(ex_g)
                                    if _guard_uses_upstream_features(ex_g):
                                        hegel_cross_scoped_dn_bids.add(ex_bid)
                            registry.sync_to_graph()
                            dn_ctrl.record_tested_witness(
                                binding=new_d,
                                public_input_sha256=pub_sha,
                                passed=True,
                                failure_kind="none",
                                features=(comb_obs_now if new_d.binding_id in hegel_cross_scoped_dn_bids else dn_obs_now),
                                probe_wall_ms=post_pub_ms,
                                execution_status="completed",
                                evidence_origin="joint_task_probe",
                                probe_id=pid_post,
                                setup_source_sha256=setup_src_sha,
                                canonical_numeric_sha256=canon_num_sha,
                                dependency_bundle_sha256=dep_bundle_sha,
                            )
                            hegel_dn_joint_pos_obs[new_d.binding_id] = [comb_obs_now]
                            hegel_dn_rep_wit[new_d.binding_id] = [
                                (f"wit_{task.short_name}", task, assemble_m98_graph_diagonal_payload(compat_spec, graph), readonly_deps, readonly_shas)
                            ]

        ep_sum["world_id"] = wid
        ep_sum["cluster_id"] = spec.cluster_id
        ep_sum["family"] = fam
        ep_sum["order_mode"] = spec.order_mode
        ep_sum["arm"] = arm
        ep_sum["stream_step"] = step_i
        ep_sum["causal_event"] = m_info["causal_event"]
        ep_sum["true_regime"] = m_info["true_regime"]
        ep_sum["event_category"] = m_info["event_category"]
        ep_sum["short_name"] = task.short_name
        ep_sum["execution_mode"] = f"llm_{repair_mode}"
        ep_sum["strict_pass"] = bool(priv_ver.get("strict_pass"))
        ep_sum["reward"] = 1.0 if priv_ver.get("strict_pass") else 0.0
        ep_sum["pre_llm_probe_wall_ms"] = float(pre_trial.get("probe_wall_ms", 0.0))
        ep_sum["post_llm_probe_wall_ms"] = post_pub_ms
        ep_sum["localization_probe_wall_ms"] = round(loc_wall_ms, 2)
        ep_sum["decomposition_probe_wall_ms"] = round(decomp_wall_ms, 2)
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
        if graph is not None:
            graph.save_checkpoint(stream_out_dir / f"graph_ckpt_after_step_{step_i:02d}.json")
        episode_results.append(ep_sum)
        print(
            f"  [M10.0 {wid} | {arm} | step {step_i}/{n_steps} ({m_info['causal_event']})] "
            f"{ep_sum['execution_mode']} | pass={ep_sum['strict_pass']} | turns={ep_sum.get('turns_used')} "
            f"| gind={ind_receipt.get('accepted') if ind_receipt else None}",
            flush=True,
        )

    st_sum = {
        "world_id": wid,
        "cluster_id": spec.cluster_id,
        "family": fam,
        "order_mode": spec.order_mode,
        "arm": arm,
        "wall_sec": round(monotonic() - stream_t0, 2),
        "strict_passes": sum(1 for r in episode_results if r.get("strict_pass")),
        "episode_count": len(episode_results),
        "zero_llm_episodes_total": sum(1 for r in episode_results if int(r.get("turns_used", 0) or 0) == 0),
        "zero_turn_passed_closures": sum(1 for r in episode_results if int(r.get("turns_used", 0) or 0) == 0 and r.get("strict_pass")),
        "total_turns_used": sum(int(r.get("turns_used", 0) or 0) for r in episode_results),
        "total_pre_llm_probe_wall_ms": round(sum(float(r.get("pre_llm_probe_wall_ms", 0.0) or 0.0) for r in episode_results), 2),
        "total_post_llm_probe_wall_ms": round(sum(float(r.get("post_llm_probe_wall_ms", 0.0) or 0.0) for r in episode_results), 2),
        "total_localization_probe_wall_ms": round(sum(float(r.get("localization_probe_wall_ms", 0.0) or 0.0) for r in episode_results), 2),
        "total_decomposition_probe_wall_ms": round(sum(float(r.get("decomposition_probe_wall_ms", 0.0) or 0.0) for r in episode_results), 2),
        "total_coverage_probe_wall_ms": round(sum(float(c["total_cpu_probe_wall_ms"]) for c in coverage_receipts_log), 2),
        "total_official_verifier_wall_ms": round(sum(float(r.get("official_verifier_wall_ms", 0.0) or 0.0) for r in episode_results), 2),
        "separability_withdrawal_receipts": [w.to_dict() for w in withdrawal_receipts],
        "autonomous_decomposition_events": decomposition_events,
        "lazy_retroactive_projection_events": lazy_projection_events,
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


def run_m100_step08_constrained_routing_and_plain_dict_ablation(
    *,
    world_specs: tuple[M100WorldSpec, ...],
    built_worlds: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    """Executed `0`-LLM Frozen Prefix (immediately before the two revisits) Constrained `dn`-Only vs Selective-Joint Routing
    AND Plain-Dict Same-Feature Candidate-SHA Parity Control (`Q10 / Q13` compliant!).
    """
    rows = []
    plain_dict_matches = 0
    plain_dict_evaluated = 0
    parity_rows = []
    for spec in world_specs:
        wid = spec.world_id
        compat = spec.to_m98_compat_spec()
        _, tasks, manifest, logs = built_worlds[wid]
        prefix_steps = len(tasks) - 2
        ck8 = out_dir / "continual_streams" / wid / "Hegel_SelectiveConditionExpansion_Online" / f"graph_ckpt_after_step_{prefix_steps:02d}.json"
        if not ck8.is_file():
            for row in manifest[prefix_steps:]:
                parity_rows.append({"world_id": wid, "event": row["causal_event"], "status": "missing_prefix_graph"})
                for policy in ("dn_only_fixed_prior_d2", "dn_only_fixed_latest_d2", "hegel_selective_condition_expansion"):
                    rows.append({"world_id": wid, "family": spec.family, "order_mode": spec.order_mode,
                                 "policy": policy, "event": row["causal_event"], "prefix_checkpoint": ck8.name,
                                 "candidate_sha256": None, "private_strict_pass": None,
                                 "execution_status": "abstained_missing_prefix_graph", "verifier_wall_ms": 0.0})
            continue
        g8_base = PersistentMechanismGraph.load_checkpoint(ck8)
        rg8_base = M94BindingRegistry(g8_base)
        sw_list = hydrate_separability_withdrawals_from_graph(g8_base, rg8_base, wid, strict_ledger_fk=True)
        dn_all = sorted(
            [b for b in rg8_base.bindings.values() if b.chain_id == f"{spec.family}_dn" and b.qualification_status == "verified"],
            key=lambda b: (-int(b.selection_priority), b.binding_id),
        )
        # Identify the two colliding `D2` branches (`sw_list[-1].downstream_binding_id` and the post-withdrawal branch)
        d2_prior_bid = sw_list[-1].downstream_binding_id if sw_list else None
        post_bindings = []
        if sw_list:
            prefix_dn = ImmutableBranchLineageController(world_id=f"{wid}_dn", family=f"{spec.family}_dn", graph=g8_base, registry=rg8_base)
            withdrawal = sw_list[-1]
            for binding in dn_all:
                if binding.binding_id == d2_prior_bid:
                    continue
                if any(w.public_input_sha256 == withdrawal.counterexample_public_input_sha256
                       and w.canonical_numeric_sha256 == withdrawal.counterexample_canonical_numeric_sha256
                       and w.is_verified_positive_witness() and w.evidence_origin == "joint_task_probe"
                       for w in prefix_dn.positive_witnesses_for(binding.binding_id)):
                    post_bindings.append(binding.binding_id)
        d2_post_bid = post_bindings[0] if post_bindings else None
        cross_bids = {b for b in (d2_prior_bid, d2_post_bid) if b is not None} if sw_list else set()

        # Plain JSON dict view for `Q10` substrate-independence parity check
        ck8_raw = json.loads(ck8.read_text(encoding="utf-8"))
        b_dict = ck8_raw["questions"]["__m9_4_mechanism_executable_bindings__"]["bindings"]
        dn_rows_dict = sorted(
            [v for v in b_dict.values() if v["chain_id"] == f"{spec.family}_dn" and v["qualification_status"] == "verified"],
            key=lambda v: (-int(v["selection_priority"]), v["binding_id"]),
        )
        up_rows_dict = sorted(
            [v for v in b_dict.values() if v["chain_id"] == f"{spec.family}_up" and v["qualification_status"] == "verified"],
            key=lambda v: (-int(v["selection_priority"]), v["binding_id"]),
        )
        flat_snap = {"gains": [1.0] * 4, "offsets": [0.0] * 4}
        mechs = ck8_raw["questions"]["__m8_authoritative_persistent_state__"]["mechanisms"]
        for node, channels in (("cal_node_p0", spec.channel_perm[:2]), ("cal_node_p1", spec.channel_perm[2:])):
            weights = mechs[node]["shared_weights"]
            for ch in channels:
                flat_snap["gains"][ch] = float(weights[f"g_ch{ch}"])
                flat_snap["offsets"][ch] = float(weights[f"b_ch{ch}"])

        for s_idx in range(prefix_steps, len(tasks)):
            tk = tasks[s_idx]
            ev_nm = manifest[s_idx]["causal_event"]
            pref_logs = logs[: s_idx + 2]
            ro_deps = {"/tmp/hegel_continual_archive/archive_apparatus_log.json": json.dumps(pref_logs, indent=2)}
            ro_shas = {k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in ro_deps.items()}
            pub_item, pub_sha = materialize_m99_public_item_and_sha(tk)

            g_step = deepcopy(g8_base)
            rg_step = M94BindingRegistry(g_step)
            update_m99_calibration_snapshot_and_graph(spec=compat, episode_id=f"{wid}_{ev_nm}_abl", public_item=pub_item, graph=g_step, registry=rg_step)
            pay = assemble_m98_graph_diagonal_payload(compat, g_step)
            up_c = ImmutableBranchLineageController(world_id=f"{wid}_up", family=f"{spec.family}_up", graph=g_step, registry=rg_step)
            dn_c = ImmutableBranchLineageController(world_id=f"{wid}_dn", family=f"{spec.family}_dn", graph=g_step, registry=rg_step)

            up_obs = compute_m100_upstream_observables(spec.family, pub_item)
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

            # Plain dict same-joint-features parity check
            snap_ev = update_m99_calibration_snapshot_and_graph(spec=compat, episode_id=f"{wid}_{ev_nm}_pd", public_item=pub_item, flat_snapshot=deepcopy(flat_snap))["flat_snapshot"]
            def select_plain(rows, features, required_schema):
                for row in rows:
                    if sorted(row["output_schema_keys"]) != sorted(required_schema):
                        continue
                    guard = row.get("applicability_guard") or row.get("precondition_guard")
                    if evaluate_m96_autonomous_guard(guard, features, allow_open_feature_keys=True)["state"] == "TRUE":
                        return row
                return None
            sel_u_d = select_plain(up_rows_dict, up_obs, ["calibrated_input"])
            sel_d_d = select_plain(dn_rows_dict, comb_obs, expected_m98_public_output_keys(spec.family))
            parity = {"world_id": wid, "event": ev_nm, "public_batch_pkl_sha256": pub_sha,
                      "status": "abstention", "plain_upstream": sel_u_d and sel_u_d["binding_id"],
                      "plain_downstream": sel_d_d and sel_d_d["binding_id"],
                      "graph_upstream": c_up and c_up.binding_id, "graph_downstream": c_dn_hg and c_dn_hg.binding_id}
            parity_rows.append(parity)
            if sel_u_d and sel_d_d and c_up is not None and c_dn_hg is not None:
                plain_dict_evaluated += 1
                cand_pd = compose_m98_two_stage_solver_code(
                    upstream_code=sel_u_d["canonical_algorithm_code"],
                    upstream_entrypoint=sel_u_d["canonical_entrypoint"],
                    downstream_code=sel_d_d["canonical_algorithm_code"],
                    downstream_entrypoint=sel_d_d["canonical_entrypoint"],
                    param_payload=snap_ev,
                )
                cand_hg_ref = compose_m98_two_stage_solver_code(
                    upstream_code=(c_up.canonical_algorithm_code or c_up.parameterized_closure_code),
                    upstream_entrypoint=(c_up.canonical_entrypoint or c_up.parameterized_entrypoint),
                    downstream_code=(c_dn_hg.canonical_algorithm_code or c_dn_hg.parameterized_closure_code),
                    downstream_entrypoint=(c_dn_hg.canonical_entrypoint or c_dn_hg.parameterized_entrypoint),
                    param_payload=pay,
                )
                parity.update(plain_candidate_sha256=hashlib.sha256(cand_pd.encode()).hexdigest(),
                              graph_candidate_sha256=hashlib.sha256(cand_hg_ref.encode()).hexdigest(), status="evaluated")
                if (sel_u_d["binding_id"] == c_up.binding_id and sel_d_d["binding_id"] == c_dn_hg.binding_id
                        and parity["plain_candidate_sha256"] == parity["graph_candidate_sha256"]):
                    plain_dict_matches += 1

            for pol_nm, b_dn in (
                ("dn_only_fixed_prior_d2", rg_step.bindings.get(d2_prior_bid) if d2_prior_bid else None),
                ("dn_only_fixed_latest_d2", rg_step.bindings.get(d2_post_bid) if d2_post_bid else None),
                ("hegel_selective_condition_expansion", c_dn_hg),
            ):
                if c_up is None or b_dn is None:
                    rows.append(
                        {
                            "policy": pol_nm,
                            "world_id": wid,
                            "family": spec.family,
                            "order_mode": spec.order_mode,
                            "event": ev_nm,
                            "prefix_checkpoint": ck8.name,
                            "public_batch_pkl_sha256": pub_sha,
                            "selected_upstream": c_up.binding_id if c_up else None,
                            "selected_downstream": b_dn.binding_id if b_dn else None,
                            "candidate_sha256": None,
                            "private_strict_pass": None,
                            "execution_status": "abstained_missing_prefix_binding",
                            "verifier_wall_ms": 0.0,
                        }
                    )
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
                        "order_mode": spec.order_mode,
                        "event": ev_nm,
                        "prefix_checkpoint": ck8.name,
                        "public_batch_pkl_sha256": pub_sha,
                        "selected_upstream": c_up.binding_id,
                        "selected_downstream": b_dn.binding_id,
                        "candidate_sha256": hashlib.sha256(cand.encode("utf-8")).hexdigest(),
                        "private_strict_pass": bool(ver.get("strict_pass")),
                        "execution_status": "completed",
                        "verifier_wall_ms": round((monotonic() - t_v) * 1000.0, 2),
                    }
                )

    summary = {}
    for pol in ("dn_only_fixed_prior_d2", "dn_only_fixed_latest_d2", "hegel_selective_condition_expansion"):
        sub = [r for r in rows if r["policy"] == pol]
        summary[pol] = {
            "total_episodes": len(sub),
            "strict_passes": sum(1 for r in sub if r["private_strict_pass"]),
            "abstained": sum(r["private_strict_pass"] is None for r in sub),
            "pass_rate_pct": round(100.0 * sum(1 for r in sub if r["private_strict_pass"]) / max(1, len(sub)), 2),
        }
    return {
        "schema": "hegel_m10_0_step08_constrained_routing_and_plain_dict_ablation_v1",
        "paid_model_calls": 0,
        "scope_note": "Prefix before the final two revisits. Control selection and composition use the same learned bindings, current measured inputs and checkpoint parameters.",
        "plain_dict_same_joint_features_parity": {
            "matched_episodes": plain_dict_matches,
            "total_episodes": len(parity_rows), "rows": parity_rows,
            "evaluated_episodes": plain_dict_evaluated,
        },
        "summary_by_policy": summary,
        "rows": rows,
    }


def compute_empirical_cumulative_curves_and_k_star(
    *,
    world_specs: tuple[M100WorldSpec, ...],
    all_receipts: list[dict[str, Any]],
    stream_summaries: list[dict[str, Any]],
    n_steps: int = 10,
) -> dict[str, Any]:
    """Compute step-by-step (`K = 1..10`) cumulative paired cost curves `ΔC(K)` and empirical quality-constrained
    break-even step `K*` (`"not_reached"` when right-censored, `01_Implementation_and_Experiment_Plan_ZH.md` §7)!
    """
    wids = [sp.world_id for sp in world_specs]
    st_by_key = {(s["world_id"], s["arm"]): s for s in stream_summaries}

    # Map each call receipt to `step_num in 1..n_steps`
    step_cost: dict[tuple[str, str, int], int] = {}
    for r in all_receipts:
        arm = r["arm"]
        if arm not in M100_ARMS:
            continue
        ep_sn = str(r["episode_short_name"])
        if "_batch_" not in ep_sn:
            continue
        k_num = int(ep_sn.rsplit("_batch_", 1)[-1])
        key = (r["world_id"], arm, k_num)
        step_cost[key] = step_cost.get(key, 0) + int(r["proxy_v2_u_010c_4o"])

    curves = {}
    for base_arm in ("Flat_MonolithicConditionalTable_Online", "Flat_PlainModuleRegistry_Online"):
        step_rows = []
        for k in range(1, n_steps + 1):
            hg_cum_c = [sum(step_cost.get((w, "Hegel_SelectiveConditionExpansion_Online", s), 0) for s in range(1, k + 1)) for w in wids]
            bs_cum_c = [sum(step_cost.get((w, base_arm, s), 0) for s in range(1, k + 1)) for w in wids]
            diffs = [float(h - b) for h, b in zip(hg_cum_c, bs_cum_c)]
            hg_pass_k = sum(
                1 for w in wids for ep in st_by_key[(w, "Hegel_SelectiveConditionExpansion_Online")]["episodes_meta"][:k] if ep["strict_pass"]
            )
            bs_pass_k = sum(1 for w in wids for ep in st_by_key[(w, base_arm)]["episodes_meta"][:k] if ep["strict_pass"])
            tot_eps_k = len(wids) * k
            pass_rate_diff = (hg_pass_k - bs_pass_k) / float(tot_eps_k)
            step_rows.append(
                {
                    "step_K": k,
                    "hegel_cum_scope1_v2_total": int(sum(hg_cum_c)),
                    "baseline_cum_scope1_v2_total": int(sum(bs_cum_c)),
                    "mean_cluster_delta_v2": round(float(np.mean(diffs)), 2),
                    "total_delta_v2": int(sum(hg_cum_c) - sum(bs_cum_c)),
                    "cluster_wins_hegel_cheaper": sum(1 for d in diffs if d < 0),
                    "hegel_cum_strict_passes": hg_pass_k,
                    "baseline_cum_strict_passes": bs_pass_k,
                    "total_episodes_at_K": tot_eps_k,
                    "pass_rate_diff_hegel_minus_base": round(pass_rate_diff, 6),
                    "quality_noninferior_at_K": bool(pass_rate_diff >= PREREGISTERED_QUALITY_NONINFERIORITY_MARGIN_PER_EPISODE),
                }
            )
        # Empirical `K*`: smallest `K` where `total_delta_v2 < 0` AND remains `< 0` through `n_steps` AND satisfies quality non-inferiority at `n_steps`!
        k_star: int | str = "not_reached"
        if step_rows[-1]["quality_noninferior_at_K"]:
            for idx_k in range(len(step_rows)):
                if all(r["total_delta_v2"] < 0 for r in step_rows[idx_k:]):
                    k_star = step_rows[idx_k]["step_K"]
                    break
        missing_usage = sum(int(r.get("usage_unknown_attempt_count", 0)) for r in all_receipts
                            if r["arm"] in {"Hegel_SelectiveConditionExpansion_Online", base_arm})
        if missing_usage:
            k_star = "unknown_partial_usage"
        curves[f"Hegel_minus_{base_arm}"] = {
            "empirical_quality_constrained_K_star": k_star,
            "preregistered_quality_noninferiority_margin_per_episode": PREREGISTERED_QUALITY_NONINFERIORITY_MARGIN_PER_EPISODE,
            "by_step_K": step_rows,
        }
    return curves


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--v7-bundle", type=Path, default=DEFAULT_V7_BUNDLE)
    ap.add_argument("--split", type=str, default="dev", choices=["dev", "confirmation"])
    ap.add_argument("--grid-size", choices=["2x3", "3x3"], help="One development grid for this run")
    ap.add_argument("--order-mode", choices=["sep_first", "coupled_first"], help="One development arrival order for this run")
    ap.add_argument("--model", type=str, default="gpt-6.2-sol")
    ap.add_argument("--reasoning-effort", type=str, default="medium")
    ap.add_argument("--max-turns", type=int, default=16)
    ap.add_argument("--max-codex-concurrency", type=int, default=32)
    ap.add_argument("--max-stream-workers", type=int, default=24)
    ap.add_argument("--finalize-existing", action="store_true")
    args = ap.parse_args()

    if args.split != "dev" and (args.grid_size or args.order_mode):
        ap.error("--grid-size and --order-mode are development controls")
    native_specs = DEV_M100_WORLD_SPECS if args.split == "dev" else CONFIRMATION_M100_WORLD_SPECS
    default_dir = DEFAULT_DEV_OUT_DIR if args.split == "dev" else DEFAULT_CONF_OUT_DIR
    if args.grid_size or args.order_mode:
        default_dir = default_dir.with_name(f"{default_dir.name}_{args.grid_size or '3x3'}_{args.order_mode or 'declared_order'}")
    out_dir: Path = (args.out_dir or default_dir).resolve()
    exec_manifest_path = out_dir / "execution_manifest.json"
    exec_done_path = out_dir / "execution_completion.json"
    if args.finalize_existing:
        start = json.loads(exec_manifest_path.read_bytes())
        if args.split != start["split"]:
            ap.error("Requested split differs from the paid execution manifest")
        saved = start.get("development_condition") or {}
        for key in ("grid_size", "order_mode"):
            requested = getattr(args, key)
            if requested is not None and requested != saved.get(key):
                ap.error(f"{key} differs from the paid execution manifest")
            setattr(args, key, saved.get(key))
        completion = json.loads(exec_done_path.read_bytes())
        if completion["execution_run_id"] != start["execution_run_id"]:
            raise ValueError("Execution manifests refer to different runs")
    paired_options = {k: val for k, val in (("grid_size", args.grid_size), ("order_mode", args.order_mode)) if val}
    selected_specs = tuple(replace(sp, order_mode=args.order_mode) if args.order_mode else sp for sp in native_specs)
    condition = {"grid_size": args.grid_size, "order_mode": args.order_mode,
                 "effective_grid": args.grid_size or "3x3", "independent_unit": "source_cluster",
                 "source_clusters": {sp.world_id: sp.cluster_id for sp in native_specs},
                 "effective_orders": {sp.world_id: sp.order_mode for sp in selected_specs}}
    if len({sp.cluster_id for sp in native_specs}) != len(native_specs):
        raise ValueError("Each invocation must have one condition per independent source cluster")

    os.environ["HEGEL_MAX_CODEX_CONCURRENCY"] = str(args.max_codex_concurrency)

    if not args.finalize_existing and out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory `{out_dir}`!")

    git_sha_start, is_dirty_start, src_tree_sha_start = _git_and_source_tree_state()
    if is_dirty_start and args.split == "confirmation" and not args.finalize_existing:
        raise RuntimeError("Refusing to start M10.0 confirmation experiment on a dirty working tree!")

    codex_info = {}
    if not args.finalize_existing:
        ensure_clean_scipy_sandbox_image()
        m64.ensure_runtime_qualification_context(sandbox_cid=None)
        env_img_digest = _docker_image_digest(m7a.CLEAN_SCIPY_SANDBOX_IMAGE)
        codex_info = m6f.discover_chatgpt_subscription_codex_cli()
        if not (codex_info.get("available") and codex_info.get("auth_mode") == "chatgpt"):
            raise RuntimeError("Official ChatGPT Codex CLI unavailable for paid execution")
    else:
        env_img_digest = start["sandbox_image_digest"]

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_mirror_dir = out_dir / "provider_raw_attempts_mirror"
    raw_mirror_dir.mkdir(parents=True, exist_ok=True)
    exec_manifest_path = out_dir / "execution_manifest.json"
    exec_done_path = out_dir / "execution_completion.json"
    run_id = f"m100_{args.split}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"

    # `C1 / Q05`: Write-once immutable start-time `execution_manifest.json` BEFORE any paid LLM call!
    if not args.finalize_existing:
        exec_manifest = {
            "schema": "hegel_execution_provenance_manifest_v2",
            "execution_run_id": run_id,
            "development_condition": condition if args.split == "dev" else None,
            "policy_public_input_source": "sandbox_public_pickle",
            "max_turns": args.max_turns, "max_stream_workers": args.max_stream_workers,
            "v7_bundle_sha256": hashlib.sha256(args.v7_bundle.read_bytes()).hexdigest(),
            "split": args.split,
            "code_commit_at_start": git_sha_start,
            "working_tree_dirty_at_start": bool(is_dirty_start),
            "source_tree_sha256_at_start": src_tree_sha_start,
            "sandbox_image_digest": env_img_digest,
            "shared_kernel_version": SHARED_KERNEL_VERSION,
            "cost_policy_version": COST_POLICY_VERSION,
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "configured_max_codex_concurrency": int(args.max_codex_concurrency),
            "max_matched_raw_archive_replays_before_projection_per_modular_arm_per_episode": 1,
            "clock_domain": "monotonic_elapsed_plus_utc_iso8601",
            "utc_started_at": datetime.now(timezone.utc).isoformat(),
        }
        with exec_manifest_path.open("x", encoding="utf-8") as f:
            json.dump(exec_manifest, f, indent=2)
            f.write("\n")

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
        m6f.FrontierCodexClient.call_structured = _wrapped_call_structured  # type: ignore[method-assign]
    campaign_t0 = monotonic()

    if not args.finalize_existing:
        v7_forest, v7_archive, _, _ = m7a.load_autogenous_forest_bundle(args.v7_bundle.resolve())
        source_nid = sorted(v7_forest.nodes.keys())[0]
    built_worlds = {spec.world_id: build_m100_world_tasks(spec, **paired_options) for spec in native_specs}

    if not args.finalize_existing:
        stage_s_results: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=len(selected_specs)) as pool:
            futs = {
                pool.submit(
                    _run_stage_s_shared_source,
                    spec=sp,
                    s_task=built_worlds[sp.world_id][0],
                    s0_prefix_logs=built_worlds[sp.world_id][3][:1],
                    base_v7_forest=v7_forest,
                    base_v7_archive=v7_archive,
                    codex_info=codex_info,
                    model=args.model,
                    reasoning_effort=args.reasoning_effort,
                    out_dir=out_dir,
                    run_id=run_id,
                    max_turns=args.max_turns,
                ): sp.world_id
                for sp in selected_specs
            }
            for fut in as_completed(futs):
                stage_s_results[futs[fut]] = fut.result()
        missing = [wid for wid, result in stage_s_results.items() if not result["summary"]["admitted"]]
        if missing:
            (out_dir / "all_provider_call_receipts.json").write_text(json.dumps(all_receipts, indent=2) + "\n")
            raise RuntimeError(f"Stage S public qualification failed for {missing}. Source receipts retained.")

        stream_jobs = [(sp, arm_nm) for sp in selected_specs for arm_nm in M100_ARMS]
        stream_summaries: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=min(int(args.max_stream_workers), len(stream_jobs))) as pool:
            s_futs = []
            for sp, arm_nm in stream_jobs:
                wid = sp.world_id
                s_tk, st_tks, s_man, a_logs = built_worlds[wid]
                st_dir = out_dir / "continual_streams" / wid / arm_nm
                st_dir.mkdir(parents=True, exist_ok=True)
                s_res = stage_s_results[wid]
                s_futs.append(
                    pool.submit(
                        _run_one_m100_world_arm_stream,
                        spec=sp,
                        arm=arm_nm,
                        s_task=s_tk,
                        stream_tasks=st_tks,
                        schedule_manifest=s_man,
                        apparatus_logs=a_logs,
                        base_v7_forest=v7_forest,
                        base_v7_archive=v7_archive,
                        source_monolithic_code=s_res["source_monolithic_code"],
                        source_archive_entry=s_res["source_archive_entry"],
                        source_forest_nid=source_nid,
                        source_pub_item=s_res["source_pub_item"],
                        source_pub_sha256=s_res["source_pub_sha256"],
                        env_image_digest=env_img_digest,
                        model=args.model,
                        codex_info=codex_info,
                        stream_out_dir=st_dir,
                        run_id=run_id,
                        max_turns=args.max_turns,
                        reasoning_effort=args.reasoning_effort,
                    )
                )
            for fut in as_completed(s_futs):
                stream_summaries.append(fut.result())

        stream_summaries.sort(key=lambda s: (s["world_id"], M100_ARMS.index(s["arm"])))
        all_receipts.sort(key=lambda x: (x["monotonic_start_sec"], x["call_uuid"]))
        (out_dir / "all_provider_call_receipts.json").write_text(json.dumps(all_receipts, indent=2) + "\n", encoding="utf-8")

        git_sha_end, is_dirty_end, src_tree_sha_end = _git_and_source_tree_state()
        exec_done = {
            "execution_run_id": run_id,
            "utc_ended_at": datetime.now(timezone.utc).isoformat(),
            "execution_elapsed_wall_sec": round(monotonic() - campaign_t0, 2),
            "peak_active_codex_concurrency": int(active_counter["peak"]),
            "code_commit_at_end": git_sha_end,
            "working_tree_dirty_at_end": bool(is_dirty_end),
            "source_tree_sha256_at_end": src_tree_sha_end,
            "source_tree_unchanged_during_paid_execution": bool(src_tree_sha_end == src_tree_sha_start),
        }
        exec_done_path.write_text(json.dumps(exec_done, indent=2) + "\n", encoding="utf-8")
    else:
        all_receipts = json.loads((out_dir / "all_provider_call_receipts.json").read_text(encoding="utf-8"))
        stage_s_results = {
            sp.world_id: {"summary": json.loads((out_dir / "stage_S_shared_sources" / sp.world_id / "00_stage_S_summary.json").read_text(encoding="utf-8"))}
            for sp in selected_specs
        }
        stream_summaries = [
            json.loads((out_dir / "continual_streams" / sp.world_id / arm_nm / "00_stream_summary.json").read_text(encoding="utf-8"))
            for sp in selected_specs
            for arm_nm in M100_ARMS
        ]

    # Separate `ablation_manifest.json` (`0` paid LLM calls!)
    abl_path = out_dir / "m10_0_step08_routing_ablation.json"
    abl_manifest_path = out_dir / "ablation_manifest.json"
    if abl_path.exists():
        routing_ablation = json.loads(abl_path.read_text(encoding="utf-8"))
    else:
        t_ab0 = monotonic()
        print("[M10.0] Running 0-LLM Step-08 Constrained Routing & Plain-Dict Parity Ablation...", flush=True)
        routing_ablation = run_m100_step08_constrained_routing_and_plain_dict_ablation(
            world_specs=selected_specs, built_worlds=built_worlds, out_dir=out_dir
        )
        abl_path.write_text(json.dumps(routing_ablation, indent=2) + "\n", encoding="utf-8")
        abl_manifest_path.write_text(
            json.dumps(
                {
                    "schema": "hegel_ablation_provenance_manifest_v1",
                    "code_commit": git_sha_start,
                    "source_tree_sha256": src_tree_sha_start,
                    "elapsed_wall_sec": round(monotonic() - t_ab0, 2),
                    "utc_completed_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    exec_start_obj = json.loads(exec_manifest_path.read_text(encoding="utf-8"))
    exec_done_obj = json.loads(exec_done_path.read_text(encoding="utf-8"))
    if args.finalize_existing:
        (out_dir / "finalization_manifest.json").write_text(
            json.dumps(
                {
                    "schema": "hegel_finalization_provenance_manifest_v1",
                    "finalizer_run_id": run_id,
                    "finalizer_code_commit": git_sha_start,
                    "finalizer_source_tree_sha256": src_tree_sha_start,
                    "finalizer_elapsed_wall_sec": round(monotonic() - campaign_t0, 2),
                    "utc_completed_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    stage_s_calls = [r for r in all_receipts if r["arm"] == "Stage_S_Shared_Source"]
    stage_s_ledger = {
        **aggregate_call_usages(stage_s_calls),
        "usage_unknown_attempt_count": sum(int(r.get("usage_unknown_attempt_count", 0) or 0) for r in stage_s_calls),
    }

    arm_ledgers: dict[str, Any] = {}
    arm_outcomes: dict[str, Any] = {}
    for arm_nm in M100_ARMS:
        a_calls = [r for r in all_receipts if r["arm"] == arm_nm]
        tot_led = aggregate_call_usages(a_calls)
        solve_led = aggregate_call_usages([r for r in a_calls if r["call_role"] == "stream_solve"])
        decomp_led = aggregate_call_usages([r for r in a_calls if r["call_role"] == "autonomous_decomposition_induction"])
        lazy_led = aggregate_call_usages([r for r in a_calls if r["call_role"] == "lazy_retroactive_projection"])
        guard_led = aggregate_call_usages([r for r in a_calls if r["call_role"] == "autonomous_guard_induction"])

        by_cluster: dict[str, Any] = {}
        for sp in selected_specs:
            wc = [r for r in a_calls if r["world_id"] == sp.world_id]
            by_cluster[sp.world_id] = {
                **aggregate_call_usages(wc),
                "usage_unknown_attempt_count": sum(int(r.get("usage_unknown_attempt_count", 0) or 0) for r in wc),
            }
        assert_hierarchical_ledger_conservation(
            reference_arm_ledger=tot_led, named_partition_ledgers={"clusters": list(by_cluster.values()),
                "roles": [solve_led, decomp_led, lazy_led, guard_led]}, context_label=arm_nm,
        )

        arm_ledgers[arm_nm] = {
            "scope1_in_stream_total": {
                **tot_led,
                "usage_unknown_attempt_count": sum(int(r.get("usage_unknown_attempt_count", 0) or 0) for r in a_calls),
            },
            "solve_only": solve_led,
            "decomposition_induction_only": decomp_led,
            "lazy_retroactive_projection_only": lazy_led,
            "guard_induction_only": guard_led,
            "by_cluster": by_cluster,
        }
        sub_s = [s for s in stream_summaries if s["arm"] == arm_nm]
        arm_outcomes[arm_nm] = {
            "strict_passes": sum(int(s["strict_passes"]) for s in sub_s),
            "episode_count": sum(int(s["episode_count"]) for s in sub_s),
            "zero_llm_episodes_total": sum(int(s["zero_llm_episodes_total"]) for s in sub_s),
            "zero_turn_passed_closures": sum(int(s["zero_turn_passed_closures"]) for s in sub_s),
            "total_turns_used": sum(int(s["total_turns_used"]) for s in sub_s),
            "separability_withdrawals": sum(len(s["separability_withdrawal_receipts"]) for s in sub_s),
            "pre_llm_probe_wall_ms": round(sum(float(s["total_pre_llm_probe_wall_ms"]) for s in sub_s), 2),
            "post_llm_probe_wall_ms": round(sum(float(s["total_post_llm_probe_wall_ms"]) for s in sub_s), 2),
            "localization_probe_wall_ms": round(sum(float(s["total_localization_probe_wall_ms"]) for s in sub_s), 2),
            "decomposition_probe_wall_ms": round(sum(float(s["total_decomposition_probe_wall_ms"]) for s in sub_s), 2),
            "coverage_probe_wall_ms": round(sum(float(s["total_coverage_probe_wall_ms"]) for s in sub_s), 2),
            "official_verifier_wall_ms": round(sum(float(s["total_official_verifier_wall_ms"]) for s in sub_s), 2),
        }

    grand_tot = aggregate_call_usages(all_receipts)
    assert_hierarchical_ledger_conservation(
        reference_arm_ledger=grand_tot,
        named_partition_ledgers={"arms_and_source": [stage_s_ledger] + [arm_ledgers[a]["scope1_in_stream_total"] for a in M100_ARMS]},
        context_label="M100_grand_total",
    )

    # Paired cluster-level inference (`n = len(selected_specs)` independent clusters!)
    wids = [sp.world_id for sp in selected_specs]
    alpha_wids = [sp.world_id for sp in selected_specs if sp.family == "alpha"]
    beta_wids = [sp.world_id for sp in selected_specs if sp.family == "beta"]
    sep_first_wids = [sp.world_id for sp in selected_specs if sp.order_mode == "sep_first"]
    cpl_first_wids = [sp.world_id for sp in selected_specs if sp.order_mode == "coupled_first"]

    hg_by_c = arm_ledgers["Hegel_SelectiveConditionExpansion_Online"]["by_cluster"]
    st_by_key = {(s["world_id"], s["arm"]): s for s in stream_summaries}

    paired_stats: dict[str, Any] = {}
    for base_arm in ("Flat_PlainModuleRegistry_Online", "Flat_MonolithicConditionalTable_Online"):
        bs_by_c = arm_ledgers[base_arm]["by_cluster"]
        c_diffs = [float(hg_by_c[w]["proxy_v2_u_010c_4o"]) - float(bs_by_c[w]["proxy_v2_u_010c_4o"]) for w in wids]
        a_diffs = [float(hg_by_c[w]["proxy_v2_u_010c_4o"]) - float(bs_by_c[w]["proxy_v2_u_010c_4o"]) for w in alpha_wids]
        b_diffs = [float(hg_by_c[w]["proxy_v2_u_010c_4o"]) - float(bs_by_c[w]["proxy_v2_u_010c_4o"]) for w in beta_wids]
        sf_diffs = [float(hg_by_c[w]["proxy_v2_u_010c_4o"]) - float(bs_by_c[w]["proxy_v2_u_010c_4o"]) for w in sep_first_wids]
        cf_diffs = [float(hg_by_c[w]["proxy_v2_u_010c_4o"]) - float(bs_by_c[w]["proxy_v2_u_010c_4o"]) for w in cpl_first_wids]
        q_diffs = [
            float(st_by_key[(w, "Hegel_SelectiveConditionExpansion_Online")]["strict_passes"]) - float(st_by_key[(w, base_arm)]["strict_passes"])
            for w in wids
        ]
        q_rate_diffs = [d / len(built_worlds[w][1]) for w, d in zip(wids, q_diffs)]
        z_diffs = [
            float(st_by_key[(w, "Hegel_SelectiveConditionExpansion_Online")]["zero_turn_passed_closures"])
            - float(st_by_key[(w, base_arm)]["zero_turn_passed_closures"])
            for w in wids
        ]
        q_boot = paired_cluster_sign_flip_and_bootstrap(q_rate_diffs)
        paired_stats[f"Hegel_minus_{base_arm}"] = {
            **paired_cluster_sign_flip_and_bootstrap(c_diffs),
            "alpha_cost_sign_flip_p": paired_cluster_sign_flip_and_bootstrap(a_diffs)["two_sided_sign_flip_p"],
            "beta_cost_sign_flip_p": paired_cluster_sign_flip_and_bootstrap(b_diffs)["two_sided_sign_flip_p"],
            "sep_first_mean_diff_v2": round(float(np.mean(sf_diffs)), 2) if sf_diffs else None,
            "coupled_first_mean_diff_v2": round(float(np.mean(cf_diffs)), 2) if cf_diffs else None,
            "quality_strict_pass_cluster_diffs": q_diffs,
            "quality_two_sided_sign_flip_p": paired_cluster_sign_flip_and_bootstrap(q_diffs)["two_sided_sign_flip_p"],
            "quality_pass_rate_diff_per_episode": round(float(np.mean(q_rate_diffs)), 6),
            "quality_pass_rate_diff_bootstrap_95ci": q_boot["bootstrap_95ci"],
            "meets_preregistered_quality_noninferiority_point": bool(
                float(np.mean(q_rate_diffs)) >= PREREGISTERED_QUALITY_NONINFERIORITY_MARGIN_PER_EPISODE
            ),
            "zero_llm_closure_cluster_diffs": z_diffs,
            "zero_llm_two_sided_sign_flip_p": paired_cluster_sign_flip_and_bootstrap(z_diffs)["two_sided_sign_flip_p"],
        }

    cum_curves = compute_empirical_cumulative_curves_and_k_star(
        world_specs=selected_specs, all_receipts=all_receipts, stream_summaries=stream_summaries, n_steps=len(built_worlds[wids[0]][1])
    )
    total_unknown_attempts = sum(int(r.get("usage_unknown_attempt_count", 0) or 0) for r in all_receipts)

    summary = {
        "schema": "hegel_m10_0_reliable_modular_amortization_and_bidirectional_withdrawal_summary_v1",
        "execution_run_id": exec_start_obj["execution_run_id"],
        "split": args.split,
        "execution_provenance": {**exec_start_obj, **exec_done_obj},
        "development_condition": exec_start_obj.get("development_condition"),
        "independent_source_cluster_count": len({sp.cluster_id for sp in selected_specs}),
        "sandbox_image_digest": exec_start_obj["sandbox_image_digest"],
        "shared_kernel_version": exec_start_obj["shared_kernel_version"],
        "cost_policy_version": exec_start_obj["cost_policy_version"],
        "model": exec_start_obj["model"],
        "reasoning_effort": exec_start_obj["reasoning_effort"],
        "configured_max_codex_concurrency": exec_start_obj["configured_max_codex_concurrency"],
        "peak_active_codex_concurrency": exec_done_obj["peak_active_codex_concurrency"],
        "execution_elapsed_wall_sec": exec_done_obj["execution_elapsed_wall_sec"],
        "usage_completeness": ("complete" if total_unknown_attempts == 0 else "partial_unknown_usage"),
        "total_usage_unknown_attempt_count": total_unknown_attempts,
        "equal_tool_flags": {
            "enable_auto_skill_packaging": False,
            "allow_l1_primitives": False,
            "require_alpha_pooled_ref_coupon_invariance": True,
            "max_sibling_upstream_fallback_candidates": 1,
            "lazy_retroactive_downstream_projection": True,
            "min_guard_clause_margin": 0.035,
        },
        "total_logical_calls": len(all_receipts),
        "grand_physical_ledger": grand_tot,
        "stage_S_shared_ledger": stage_s_ledger,
        "stage_S_summaries": {wid: stage_s_results[wid]["summary"] for wid in wids},
        "arm_outcome_totals": arm_outcomes,
        "arm_cost_ledgers": arm_ledgers,
        "paired_cluster_inference": paired_stats,
        "cumulative_amortization_curves_and_K_star": cum_curves,
        "step08_routing_and_plain_dict_ablation_summary": {
            "plain_dict_same_joint_features_parity": routing_ablation["plain_dict_same_joint_features_parity"],
            "summary_by_policy": routing_ablation["summary_by_policy"],
        },
        "streams": stream_summaries,
    }
    (out_dir / "m10_0_continual_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    sha_lines = []
    for fp in sorted(out_dir.rglob("*")):
        if fp.is_file() and fp.name != "SHA256SUMS.txt":
            rel = fp.relative_to(out_dir).as_posix()
            sha_lines.append(f"{hashlib.sha256(fp.read_bytes()).hexdigest()}  {rel}")
    (out_dir / "SHA256SUMS.txt").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "execution_run_id": summary["execution_run_id"],
                "split": args.split,
                "total_calls": len(all_receipts),
                "unknown_attempts": total_unknown_attempts,
                "peak_concurrency": summary["peak_active_codex_concurrency"],
                "arm_outcomes": arm_outcomes,
                "K_star_vs_Monolithic": cum_curves["Hegel_minus_Flat_MonolithicConditionalTable_Online"]["empirical_quality_constrained_K_star"],
                "K_star_vs_Plain": cum_curves["Hegel_minus_Flat_PlainModuleRegistry_Online"]["empirical_quality_constrained_K_star"],
                "paired_cluster_inference": paired_stats,
            },
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
    original_call = m6f.FrontierCodexClient.call_structured
    try:
        _main()
    finally:
        m6f.FrontierCodexClient.call_structured = original_call


if __name__ == "__main__":
    main()
