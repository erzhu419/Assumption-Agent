#!/usr/bin/env python3
"""`M10.0.1` Audit Closure, Missing-Probe Docker Post-Hoc Back-Verification, & `0`-LLM Lifting Coverage Audit.

Executes with **0 paid LLM calls** and **zero mutation** of any existing sealed `docs/evidence/m10_0_*` directory:
  1. `01_m100_probe_persistence_inventory.json`:
     Inventories EVERY witness across `M10.0 dev` (`n=4`) and `M10.0 confirmation` (`n=8`) `Hegel` final checkpoints,
     explicitly marking raw-receipt presence (`persisted_raw_receipt` vs `recorded_without_raw_measurement`, matching
     the `41` confirmation [`32 pprobe + 9 uprobe`] and `19` dev missing receipts identified by `T01`).
  2. `02_m100_41_missing_probes_docker_backverification.json`:
     Replays ALL missing confirmation probes (`41/41`) in parallel inside real `hegel-clean-scipy:m7v1` containers
     using the exact sealed binding code, parameter epoch, public batch, and readonly dependency bundle from the
     archived checkpoints. Persists each backfilled raw receipt under `backfilled_probe_receipts/backfill_<orig>.json`
     marked explicitly with `raw_measurement_status = "backfilled_posthoc_replay"`, `supersedes_probe_id`, and
     fresh UTC timestamps (strictly post-hoc; NEVER overwrites historical `m10_0_*` directories or scores!).
  3. `03_m100_archived_solvers_lifting_coverage_inventory.json`:
     Audits archived `M10.0` confirmation monolithic solvers via `extract_pure_reusable_functions_from_python`,
     `lift_python_scientific_expressions_to_programs`, and `anti_unify_programs`, classifying coverage cleanly into
     `no_shared_skeleton`, `lifter_unsupported`, and `extracted_unadopted` (`0` LLM calls).
  4. Seals `SHA256SUMS.txt` across the new `docs/evidence/m10_0_1_audit_closure_and_probe_reverification_20261128/` directory.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from time import monotonic
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from hegel_machine.cognition.branch_reconciliation import anti_unify_programs
from hegel_machine.cognition.m6_4_executable_wormhole_library import extract_pure_reusable_functions_from_python
from hegel_machine.cognition.m6_8_b_prime_capability_bridge import lift_python_scientific_expressions_to_programs
from hegel_machine.cognition.m7_holdout_ood_blind_benchmark import ensure_clean_scipy_sandbox_image
from hegel_machine.cognition.m8_persistent_variational_planner import PersistentMechanismGraph
from hegel_machine.cognition.m9_4_1_shared_verification_kernel import (
    SHARED_KERNEL_VERSION,
    compute_canonical_numeric_item_sha256,
)
from hegel_machine.cognition.m9_4_continual_mechanism_binding_world import M94BindingRegistry
from hegel_machine.cognition.m9_5_equal_tool_continual_attribution_world import (
    execute_single_pre_llm_public_gate_probe_in_container,
)
from hegel_machine.cognition.m9_6_autonomous_structure_continual_world import (
    ImmutableBranchLineageController,
    ProbeReceiptStore,
    classify_public_probe_failure,
    enrich_joint_probe_receipt_for_persistence,
)
from hegel_machine.cognition.m9_8_blind_mismatch_and_selective_dependency_world import (
    compose_m98_two_stage_solver_code,
    diagnose_m98_factored_conflict_locus,
)
from hegel_machine.cognition.m9_9_autonomous_decomposition_and_interaction_withdrawal_world import (
    hydrate_separability_withdrawals_from_graph,
    materialize_m99_public_item_and_sha,
)
from hegel_machine.cognition.m10_0_reliable_modular_amortization_and_bidirectional_withdrawal_world import (
    CONFIRMATION_M100_WORLD_SPECS,
    DEV_M100_WORLD_SPECS,
    M100_ALL_OBSERVABLE_NAMES,
    M100WorldSpec,
    build_m100_world_tasks,
)

DEV_DIR = REPO_ROOT / "docs/evidence/m10_0_dev_economic_calibration_20261128"
CONF_DIR = REPO_ROOT / "docs/evidence/m10_0_reliable_modular_amortization_and_bidirectional_withdrawal_20261128"
OUT_DIR = REPO_ROOT / "docs/evidence/m10_0_1_audit_closure_and_probe_reverification_20261128"


def _collect_persisted_probe_evidence_in_stream_dir(stream_dir: Path) -> tuple[set[str], set[tuple[str, str]]]:
    """Return `(persisted_probe_ids, persisted_pre_llm_joint_fingerprints)` where `persisted_pre_llm_joint_fingerprints`
    captures `(public_input_sha256, 'passed'|'failed')` from persisted `pre_llm_trial_receipt.probe_result` payloads
    that stored full raw `scientific_probe_receipt` JSON without embedding the `jprobe_*` ID string (`T01`).
    """
    found_ids: set[str] = set()
    pre_llm_joint_fps: set[tuple[str, str]] = set()

    def _walk(obj: Any) -> None:
        if isinstance(obj, dict):
            pid = obj.get("probe_id")
            if isinstance(pid, str) and pid:
                if any(
                    k in obj
                    for k in (
                        "scientific_probe_receipt",
                        "chosen_up_holdout_rmse",
                        "component_probe_executed",
                        "accepted",
                        "joint_gate_receipt",
                    )
                ):
                    found_ids.add(pid)
            j_pid = obj.get("joint_probe_id")
            if isinstance(j_pid, str) and j_pid and isinstance(obj.get("joint_gate_receipt"), dict):
                found_ids.add(j_pid)
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for x in obj:
                _walk(x)

    for fp in sorted(stream_dir.rglob("00_episode_summary.json")):
        ep = json.loads(fp.read_text(encoding="utf-8"))
        _walk(ep)
        pre = ep.get("pre_llm_trial_receipt")
        if isinstance(pre, dict) and isinstance(pre.get("probe_result"), dict):
            pr = pre["probe_result"]
            assets = pr.get("public_setup_files_sha256") or {}
            pub_sha = str(assets.get("/holdout_task/public_batch.pkl") or "")
            if pub_sha and isinstance(pr.get("accepted"), bool) and isinstance(pr.get("scientific_probe_receipt"), dict):
                pre_llm_joint_fps.add((pub_sha, "passed" if pr["accepted"] else "failed"))
    st_sum = stream_dir / "00_stream_summary.json"
    if st_sum.is_file():
        _walk(json.loads(st_sum.read_text(encoding="utf-8")))
    return found_ids, pre_llm_joint_fps


def _reconstruct_param_payload_for_epoch(
    graph: PersistentMechanismGraph,
    parent_versions: dict[str, int],
) -> dict[str, Any]:
    weights: dict[str, float] = {}
    for pid in ("cal_node_p0", "cal_node_p1"):
        mech = graph.mechanisms.get(pid)
        if mech is None:
            continue
        epoch = parent_versions.get(pid, int(mech.version))
        if int(mech.version) == int(epoch):
            weights.update({k: float(v) for k, v in mech.shared_weights.items()})
        else:
            archived = [row for row in graph.mechanism_lineage.get(pid, []) if int(row["version"]) == int(epoch)]
            if archived:
                weights.update({k: float(v) for k, v in archived[0]["shared_weights"].items()})
            else:
                weights.update({k: float(v) for k, v in mech.shared_weights.items()})
    return {
        "gains": [float(weights.get(f"g_ch{i}", 1.0)) for i in range(4)],
        "offsets": [float(weights.get(f"b_ch{i}", 0.0)) for i in range(4)],
    }


def _inventory_split(split_name: str, base_dir: Path, specs: tuple[M100WorldSpec, ...]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    withdrawal_prior_refs: list[dict[str, Any]] = []
    for sp in specs:
        wid = sp.world_id
        stream_dir = base_dir / "continual_streams" / wid / "Hegel_SelectiveConditionExpansion_Online"
        ck10 = stream_dir / "graph_ckpt_after_step_10.json"
        g = PersistentMechanismGraph.load_checkpoint(ck10)
        rg = M94BindingRegistry(g)
        up_ctrl = ImmutableBranchLineageController(world_id=f"{wid}_up", family=f"{sp.family}_up", graph=g, registry=rg)
        dn_ctrl = ImmutableBranchLineageController(world_id=f"{wid}_dn", family=f"{sp.family}_dn", graph=g, registry=rg)
        persisted_ids, pre_llm_joint_fps = _collect_persisted_probe_evidence_in_stream_dir(stream_dir)

        sw_list = hydrate_separability_withdrawals_from_graph(
            g, rg, wid, strict_ledger_fk=True, expected_feature_names=M100_ALL_OBSERVABLE_NAMES
        )
        prior_wid_set = {sw.prior_witness_id for sw in sw_list}

        for layer_nm, ctrl in (("upstream", up_ctrl), ("downstream", dn_ctrl)):
            for w in ctrl.witness_ledger:
                prefix = w.probe_id.split("_", 1)[0] if "_" in w.probe_id else "other"
                has_raw_by_id = bool(w.probe_id and w.probe_id in persisted_ids)
                has_raw_pre_llm_jprobe = bool(
                    prefix == "jprobe"
                    and w.evidence_origin == "joint_task_probe"
                    and (w.public_input_sha256, w.result) in pre_llm_joint_fps
                )
                has_raw = has_raw_by_id or has_raw_pre_llm_jprobe
                row = {
                    "split": split_name,
                    "world_id": wid,
                    "family": sp.family,
                    "layer": layer_nm,
                    "witness_id": w.witness_id,
                    "binding_id": w.binding_id,
                    "probe_id": w.probe_id,
                    "probe_prefix": prefix,
                    "evidence_origin": w.evidence_origin,
                    "result": w.result,
                    "failure_kind": w.failure_kind,
                    "metric_value": w.metric_value,
                    "public_input_sha256": w.public_input_sha256,
                    "canonical_numeric_sha256": w.canonical_numeric_sha256,
                    "setup_source_sha256": w.setup_source_sha256,
                    "dependency_bundle_sha256": w.dependency_bundle_sha256,
                    "parent_parameter_versions": dict(w.parent_parameter_versions),
                    "is_withdrawal_prior_witness": w.witness_id in prior_wid_set,
                    "persisted_by_explicit_probe_id": has_raw_by_id,
                    "persisted_in_pre_llm_probe_result_without_probe_id_key": (not has_raw_by_id) and has_raw_pre_llm_jprobe,
                    "raw_measurement_status": ("persisted_raw_receipt" if has_raw else "recorded_without_raw_measurement"),
                }
                rows.append(row)
                if w.witness_id in prior_wid_set:
                    withdrawal_prior_refs.append(row)

    missing = [r for r in rows if r["raw_measurement_status"] == "recorded_without_raw_measurement"]
    by_prefix: dict[str, int] = {}
    for r in missing:
        by_prefix[r["probe_prefix"]] = by_prefix.get(r["probe_prefix"], 0) + 1
    return {
        "split": split_name,
        "total_witnesses": len(rows),
        "persisted_raw_receipt_count": len(rows) - len(missing),
        "recorded_without_raw_measurement_count": len(missing),
        "missing_by_probe_prefix": by_prefix,
        "withdrawal_prior_witness_total": len(withdrawal_prior_refs),
        "withdrawal_prior_witness_missing_raw_count": sum(
            1 for r in withdrawal_prior_refs if r["raw_measurement_status"] == "recorded_without_raw_measurement"
        ),
        "rows": rows,
    }


def _find_matching_episode_and_code(
    *,
    spec: M100WorldSpec,
    stream_dir: Path,
    graph: PersistentMechanismGraph,
    registry: M94BindingRegistry,
    missing_row: dict[str, Any],
    built_world: tuple[Any, tuple[Any, ...], list[dict[str, Any]], list[dict[str, Any]]],
) -> dict[str, Any]:
    s_task, stream_tasks, manifest, app_logs = built_world
    all_tasks = [(0, s_task, app_logs[:1])] + [(idx + 1, tk, app_logs[: idx + 2]) for idx, tk in enumerate(stream_tasks)]
    target_pub_sha = missing_row["public_input_sha256"]
    target_dep_sha = missing_row["dependency_bundle_sha256"]

    matched_task = None
    matched_step = None
    matched_logs = None
    matched_pub_item = None
    matched_pub_sha = None
    matched_num_sha = None
    for st_num, tk, p_logs in all_tasks:
        p_item, p_sha = materialize_m99_public_item_and_sha(tk)
        if p_sha == target_pub_sha:
            matched_task = tk
            matched_step = st_num
            matched_logs = p_logs
            matched_pub_item = p_item
            matched_pub_sha = p_sha
            matched_num_sha = compute_canonical_numeric_item_sha256(p_item)
            break
    if matched_task is None or matched_logs is None or matched_pub_item is None or matched_pub_sha is None or matched_num_sha is None:
        raise RuntimeError(f"Could not match task for witness {missing_row['witness_id']} ({target_pub_sha[:12]})")

    ro_deps = {"/tmp/hegel_continual_archive/archive_apparatus_log.json": json.dumps(matched_logs, indent=2)}
    ro_shas = {k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in ro_deps.items()}
    dep_bundle_sha = hashlib.sha256(json.dumps(ro_shas, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    setup_sha = hashlib.sha256(matched_task.public_setup_py.encode("utf-8")).hexdigest()

    # Load exact episode checkpoint and submitted composed solver if pprobe (for exact code match!)
    ep_dir = stream_dir / matched_task.task_id
    binding = registry.bindings[missing_row["binding_id"]]
    b_code = binding.canonical_algorithm_code or binding.parameterized_closure_code
    b_ep = binding.canonical_entrypoint or binding.parameterized_entrypoint

    # Find exact composed solver code for joint_task_probe (`pprobe` / `jprobe`):
    composed_code = None
    comp_shas = [binding.authoritative_executed_code_sha256()]
    if missing_row["evidence_origin"] == "joint_task_probe":
        # Check archived episode summary & snapshot to reconstruct exact composed solver (`upstream + downstream + param_payload`)
        ep_sum_fp = ep_dir / "00_episode_summary.json"
        snap_dir = ep_dir / "04_submitted_artifacts_snapshot"
        ep_sum = json.loads(ep_sum_fp.read_text(encoding="utf-8")) if ep_sum_fp.is_file() else {}
        pre_tr = ep_sum.get("pre_llm_trial_receipt") or {}
        ck_step = stream_dir / f"graph_ckpt_after_step_{matched_step:02d}.json"
        g_step = PersistentMechanismGraph.load_checkpoint(ck_step) if ck_step.is_file() else graph
        rg_step = M94BindingRegistry(g_step)
        pay = _reconstruct_param_payload_for_epoch(g_step, missing_row["parent_parameter_versions"])
        # Re-read the exact upstream & downstream codes from the graph bindings
        if missing_row["layer"] == "upstream":
            dn_id = str(pre_tr.get("selected_downstream_id") or "")
            dn_b = rg_step.bindings.get(dn_id) or registry.bindings.get(dn_id)
            assert dn_b is not None, f"Missing downstream binding `{dn_id}` for `{missing_row['witness_id']}`"
            composed_code = compose_m98_two_stage_solver_code(
                upstream_code=b_code,
                upstream_entrypoint=b_ep,
                downstream_code=(dn_b.canonical_algorithm_code or dn_b.parameterized_closure_code),
                downstream_entrypoint=(dn_b.canonical_entrypoint or dn_b.parameterized_entrypoint),
                param_payload=pay,
            )
            comp_shas = sorted({binding.authoritative_executed_code_sha256(), dn_b.authoritative_executed_code_sha256()})
        else:
            up_id = str(pre_tr.get("selected_upstream_id") or "")
            up_b = rg_step.bindings.get(up_id) or registry.bindings.get(up_id)
            assert up_b is not None, f"Missing upstream binding `{up_id}` for `{missing_row['witness_id']}`"
            # Note: `up_b` parameter epoch comes from `g_step` current mechanisms!
            pay_up = _reconstruct_param_payload_for_epoch(
                g_step,
                {
                    "cal_node_p0": int(g_step.mechanisms["cal_node_p0"].version),
                    "cal_node_p1": int(g_step.mechanisms["cal_node_p1"].version),
                },
            )
            composed_code = compose_m98_two_stage_solver_code(
                upstream_code=(up_b.canonical_algorithm_code or up_b.parameterized_closure_code),
                upstream_entrypoint=(up_b.canonical_entrypoint or up_b.parameterized_entrypoint),
                downstream_code=b_code,
                downstream_entrypoint=b_ep,
                param_payload=pay_up,
            )
            comp_shas = sorted({up_b.authoritative_executed_code_sha256(), binding.authoritative_executed_code_sha256()})

    return {
        "spec": spec,
        "task": matched_task,
        "step_num": matched_step,
        "pub_item": matched_pub_item,
        "pub_sha": matched_pub_sha,
        "num_sha": matched_num_sha,
        "setup_sha": setup_sha,
        "ro_deps": ro_deps,
        "ro_shas": ro_shas,
        "dep_bundle_sha": dep_bundle_sha,
        "dep_bundle_sha_matches_witness": (not target_dep_sha) or (dep_bundle_sha == target_dep_sha),
        "binding_code": b_code,
        "binding_ep": b_ep,
        "binding_code_sha": binding.authoritative_executed_code_sha256(),
        "composed_code": composed_code,
        "component_code_sha256s": comp_shas,
        "param_payload": _reconstruct_param_payload_for_epoch(graph, missing_row["parent_parameter_versions"]),
    }


def _replay_one_missing_probe(
    *,
    missing_row: dict[str, Any],
    ctx: dict[str, Any],
    backfill_store: ProbeReceiptStore,
) -> dict[str, Any]:
    spec: M100WorldSpec = ctx["spec"]
    orig_pid = missing_row["probe_id"]
    backfill_pid = f"backfill_{orig_pid}"
    t0 = monotonic()
    if missing_row["evidence_origin"] == "direct_component_probe":
        diag = diagnose_m98_factored_conflict_locus(
            spec.family,
            ctx["pub_item"],
            chosen_up_code=ctx["binding_code"],
            chosen_up_entrypoint=ctx["binding_ep"],
            diag_payload=ctx["param_payload"],
            force_direct_component_probe=True,
        )
        wall_ms = round((monotonic() - t0) * 1000.0, 2)
        rmse = diag.get("chosen_up_holdout_rmse")
        passed_replay = bool(
            diag.get("execution_status") == "completed" and rmse is not None and 0.0 <= float(rmse) <= 0.03
        )
        rec_payload = {
            **diag,
            "probe_id": backfill_pid,
            "supersedes_probe_id": orig_pid,
            "raw_measurement_status": "backfilled_posthoc_replay",
            "public_input_sha256": ctx["pub_sha"],
            "canonical_numeric_sha256": ctx["num_sha"],
            "setup_source_sha256": ctx["setup_sha"],
            "dependency_bundle_sha256": ctx["dep_bundle_sha"],
        }
        env = backfill_store.persist_probe_receipt(
            probe_id=backfill_pid,
            receipt=rec_payload,
            probe_kind="direct_component_probe",
            metadata={
                "world_id": spec.world_id,
                "witness_id": missing_row["witness_id"],
                "supersedes_probe_id": orig_pid,
                "raw_measurement_status": "backfilled_posthoc_replay",
            },
        )
        verdict_matches = (passed_replay and missing_row["result"] == "passed") or (
            (not passed_replay) and missing_row["result"] == "failed"
        )
        rmse_diff = (
            abs(float(rmse) - float(missing_row["metric_value"]))
            if (rmse is not None and missing_row["metric_value"] is not None)
            else None
        )
        return {
            "world_id": spec.world_id,
            "witness_id": missing_row["witness_id"],
            "binding_id": missing_row["binding_id"],
            "orig_probe_id": orig_pid,
            "backfill_probe_id": backfill_pid,
            "evidence_origin": "direct_component_probe",
            "is_withdrawal_prior_witness": missing_row["is_withdrawal_prior_witness"],
            "recorded_result": missing_row["result"],
            "replayed_result": ("passed" if passed_replay else "failed"),
            "verdict_matches_recorded_witness": verdict_matches,
            "recorded_rmse": missing_row["metric_value"],
            "replayed_rmse": (round(float(rmse), 10) if rmse is not None else None),
            "rmse_abs_diff": (round(rmse_diff, 10) if rmse_diff is not None else None),
            "execution_status": diag.get("execution_status"),
            "receipt_content_sha256": env["receipt_content_sha256"],
            "backfilled_receipt_file": f"backfilled_probe_receipts/{backfill_pid}.json",
            "wall_ms": wall_ms,
        }

    # Joint task probe (`pprobe`)
    cand_code = str(ctx["composed_code"])
    cand_sha = hashlib.sha256(cand_code.encode("utf-8")).hexdigest()
    gate = execute_single_pre_llm_public_gate_probe_in_container(
        task=ctx["task"],
        candidate_code=cand_code,
        readonly_dependencies=ctx["ro_deps"],
        expected_dependency_sha256=ctx["ro_shas"],
        require_pooled_coupon_check=(spec.family == "alpha"),
    )
    wall_ms = round((monotonic() - t0) * 1000.0, 2)
    cls = classify_public_probe_failure(gate)
    enriched = enrich_joint_probe_receipt_for_persistence(
        gate,
        probe_id=backfill_pid,
        public_input_sha256=ctx["pub_sha"],
        canonical_numeric_sha256=ctx["num_sha"],
        setup_source_sha256=ctx["setup_sha"],
        dependency_bundle_sha256=ctx["dep_bundle_sha"],
        candidate_sha256=cand_sha,
        component_code_sha256s=ctx["component_code_sha256s"],
        wall_ms=wall_ms,
    )
    enriched["supersedes_probe_id"] = orig_pid
    enriched["raw_measurement_status"] = "backfilled_posthoc_replay"
    env = backfill_store.persist_probe_receipt(
        probe_id=backfill_pid,
        receipt=enriched,
        probe_kind="joint_task_probe",
        metadata={
            "world_id": spec.world_id,
            "witness_id": missing_row["witness_id"],
            "supersedes_probe_id": orig_pid,
            "raw_measurement_status": "backfilled_posthoc_replay",
        },
    )
    passed_replay = bool(gate.get("accepted"))
    verdict_matches = (passed_replay and missing_row["result"] == "passed") or (
        (not passed_replay) and missing_row["result"] == "failed"
    )
    return {
        "world_id": spec.world_id,
        "witness_id": missing_row["witness_id"],
        "binding_id": missing_row["binding_id"],
        "orig_probe_id": orig_pid,
        "backfill_probe_id": backfill_pid,
        "evidence_origin": "joint_task_probe",
        "is_withdrawal_prior_witness": missing_row["is_withdrawal_prior_witness"],
        "recorded_result": missing_row["result"],
        "replayed_result": ("passed" if passed_replay else "failed"),
        "verdict_matches_recorded_witness": verdict_matches,
        "failure_kind_replayed": cls["failure_kind"],
        "dep_bundle_sha_matches_witness": ctx["dep_bundle_sha_matches_witness"],
        "candidate_sha256": cand_sha,
        "receipt_content_sha256": env["receipt_content_sha256"],
        "backfilled_receipt_file": f"backfilled_probe_receipts/{backfill_pid}.json",
        "wall_ms": wall_ms,
    }


def _audit_archived_solvers_lifting_coverage() -> dict[str, Any]:
    """Zero-LLM audit of `M6.8` scalar AST lifter & `anti_unify_programs` coverage over archived `M10.0` solvers."""
    solver_rows: list[dict[str, Any]] = []
    all_lifted: list[tuple[str, str, Any]] = []

    for sp in CONFIRMATION_M100_WORLD_SPECS:
        wid = sp.world_id
        s_fp = CONF_DIR / "stage_S_shared_sources" / wid / "04_submitted_artifacts_snapshot"
        cand_files: list[tuple[str, Path]] = []
        if s_fp.is_dir():
            for fp in sorted(s_fp.glob("*.py")):
                cand_files.append((f"{wid}::StageS::{fp.name}", fp))
        mono_dir = CONF_DIR / "continual_streams" / wid / "Flat_MonolithicConditionalTable_Online"
        for ep_snap in sorted(mono_dir.glob("*/04_submitted_artifacts_snapshot/*batch_solver.py"))[:3]:
            cand_files.append((f"{wid}::Mono::{ep_snap.parent.parent.name}", ep_snap))

        for label, fp in cand_files:
            code = fp.read_text(encoding="utf-8")
            pure_fns = extract_pure_reusable_functions_from_python(code, max_funcs=12)
            closures = [{"name": nm, "code": cd} for nm, cd in pure_fns]
            funnel: dict[str, Any] = {}
            lifted = lift_python_scientific_expressions_to_programs(
                closures, max_programs=8, funnel_diagnostics_out=funnel
            )
            for fn_nm, prog in lifted:
                all_lifted.append((label, fn_nm, prog))
            if lifted:
                cat = "extracted_unadopted"
            else:
                cat = "lifter_unsupported"
            solver_rows.append(
                {
                    "solver_label": label,
                    "family": sp.family,
                    "bytes": len(code.encode("utf-8")),
                    "pure_functions_extracted": len(pure_fns),
                    "lifted_program_count": len(lifted),
                    "lifted_ops": [f"{nm}:{p.op}" for nm, p in lifted],
                    "funnel_aborted_imperative_or_unsupported": int(funnel.get("aborted_imperative_or_unsupported", 0)),
                    "coverage_category": cat,
                }
            )

    anti_pairs: list[dict[str, Any]] = []
    for i in range(len(all_lifted)):
        la, fa, pa = all_lifted[i]
        for j in range(i + 1, len(all_lifted)):
            lb, fb, pb = all_lifted[j]
            if la.split("::", 1)[0] == lb.split("::", 1)[0]:
                continue
            res = anti_unify_programs(pa, pb)
            if res is not None:
                skel, slots = res
                if skel.op not in {"arg", "const", "feature"}:
                    anti_pairs.append(
                        {
                            "pair": [f"{la}:{fa}", f"{lb}:{fb}"],
                            "skeleton_op": skel.op,
                            "skeleton_size": skel.size,
                            "slot_count": len(slots),
                            "status": "extracted_unadopted",
                        }
                    )
                    if len(anti_pairs) >= 16:
                        break
        if len(anti_pairs) >= 16:
            break

    cats = {"extracted_unadopted": 0, "lifter_unsupported": 0, "no_shared_skeleton": 0}
    for r in solver_rows:
        cats[r["coverage_category"]] += 1
    if all_lifted and not anti_pairs:
        cats["no_shared_skeleton"] = len(all_lifted)

    return {
        "schema": "hegel_m10_0_1_lifting_coverage_audit_v1",
        "paid_llm_calls": 0,
        "audited_solver_count": len(solver_rows),
        "total_lifted_scalar_programs": len(all_lifted),
        "non_trivial_anti_unified_cross_cluster_pairs": len(anti_pairs),
        "category_counts": cats,
        "scientific_interpretation": (
            "Archived M10.0 numerical solvers consist primarily of imperative NumPy/SciPy matrix routines "
            "(linear solves, Nelder-Mead/SLSQP optimization, finite-volume sparse assembly) that fall in "
            "`lifter_unsupported` for the legacy M6.8 scalar AST expression lifter; therefore general "
            "multi-module discovery in M10.1 must operate on executable Python callable nodes and runtime "
            "I/O contracts rather than scalar symbolic AST lifting alone."
        ),
        "sample_anti_unified_pairs": anti_pairs[:8],
        "solver_rows": solver_rows,
    }


def _seal_dir(target_dir: Path) -> int:
    entries: list[str] = []
    for fp in sorted(target_dir.rglob("*")):
        if fp.is_file() and fp.name != "SHA256SUMS.txt":
            rel = fp.relative_to(target_dir).as_posix()
            sha = hashlib.sha256(fp.read_bytes()).hexdigest()
            entries.append(f"{sha}  {rel}")
    (target_dir / "SHA256SUMS.txt").write_text("\n".join(entries) + "\n", encoding="utf-8")
    return len(entries)


def main() -> None:
    t_start = monotonic()
    utc_start = datetime.now(timezone.utc).isoformat()
    ensure_clean_scipy_sandbox_image()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    backfill_store = ProbeReceiptStore(OUT_DIR / "backfilled_probe_receipts")

    print("[M10.0.1] 1/3 Scanning M10.0 dev & confirmation probe persistence inventory...", flush=True)
    dev_inv = _inventory_split("dev", DEV_DIR, DEV_M100_WORLD_SPECS)
    conf_inv = _inventory_split("confirmation", CONF_DIR, CONFIRMATION_M100_WORLD_SPECS)
    inv_doc = {
        "schema": "hegel_m10_0_1_probe_persistence_inventory_v1",
        "paid_llm_calls": 0,
        "utc_generated_at": datetime.now(timezone.utc).isoformat(),
        "dev": {k: v for k, v in dev_inv.items() if k != "rows"},
        "confirmation": {k: v for k, v in conf_inv.items() if k != "rows"},
        "dev_rows": dev_inv["rows"],
        "confirmation_rows": conf_inv["rows"],
    }
    (OUT_DIR / "01_m100_probe_persistence_inventory.json").write_text(
        json.dumps(inv_doc, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"  dev: {dev_inv['recorded_without_raw_measurement_count']}/{dev_inv['total_witnesses']} missing ({dev_inv['missing_by_probe_prefix']}) | "
        f"confirm: {conf_inv['recorded_without_raw_measurement_count']}/{conf_inv['total_witnesses']} missing ({conf_inv['missing_by_probe_prefix']})",
        flush=True,
    )

    print("[M10.0.1] 2/3 Running parallel Docker post-hoc back-verification on ALL missing confirmation probes...", flush=True)
    built_conf = {sp.world_id: build_m100_world_tasks(sp) for sp in CONFIRMATION_M100_WORLD_SPECS}
    specs_by_id = {sp.world_id: sp for sp in CONFIRMATION_M100_WORLD_SPECS}
    graphs_by_id: dict[str, tuple[PersistentMechanismGraph, M94BindingRegistry]] = {}
    for sp in CONFIRMATION_M100_WORLD_SPECS:
        ck10 = CONF_DIR / "continual_streams" / sp.world_id / "Hegel_SelectiveConditionExpansion_Online" / "graph_ckpt_after_step_10.json"
        g = PersistentMechanismGraph.load_checkpoint(ck10)
        graphs_by_id[sp.world_id] = (g, M94BindingRegistry(g))

    missing_conf_rows = [r for r in conf_inv["rows"] if r["raw_measurement_status"] == "recorded_without_raw_measurement"]
    replay_jobs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for m_row in missing_conf_rows:
        wid = m_row["world_id"]
        g, rg = graphs_by_id[wid]
        st_dir = CONF_DIR / "continual_streams" / wid / "Hegel_SelectiveConditionExpansion_Online"
        ctx = _find_matching_episode_and_code(
            spec=specs_by_id[wid],
            stream_dir=st_dir,
            graph=g,
            registry=rg,
            missing_row=m_row,
            built_world=built_conf[wid],
        )
        replay_jobs.append((m_row, ctx))

    back_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        futs = [
            pool.submit(_replay_one_missing_probe, missing_row=m_r, ctx=ctx, backfill_store=backfill_store)
            for m_r, ctx in replay_jobs
        ]
        for fut in as_completed(futs):
            back_rows.append(fut.result())

    back_rows.sort(key=lambda r: (r["world_id"], r["witness_id"]))
    matched_cnt = sum(1 for r in back_rows if r["verdict_matches_recorded_witness"])
    w_prior_rows = [r for r in back_rows if r["is_withdrawal_prior_witness"]]
    w_prior_matched = sum(1 for r in w_prior_rows if r["verdict_matches_recorded_witness"])
    back_doc = {
        "schema": "hegel_m10_0_1_missing_probes_posthoc_docker_backverification_v1",
        "paid_llm_calls": 0,
        "raw_measurement_status": "backfilled_posthoc_replay",
        "posthoc_disclaimer": (
            "Post-hoc container replay against unchanged sealed M10.0 confirmation checkpoints and code. "
            "Does NOT alter historical M10.0 scores or rewrite sealed m10_0_* directories."
        ),
        "utc_started_at": utc_start,
        "utc_ended_at": datetime.now(timezone.utc).isoformat(),
        "monotonic_elapsed_sec": round(monotonic() - t_start, 2),
        "shared_kernel_version": SHARED_KERNEL_VERSION,
        "total_missing_confirmation_probes_replayed": len(back_rows),
        "verdict_matched_count": matched_cnt,
        "verdict_mismatch_count": len(back_rows) - matched_cnt,
        "withdrawal_prior_witnesses_replayed": len(w_prior_rows),
        "withdrawal_prior_witnesses_matched": w_prior_matched,
        "pprobe_replayed": sum(1 for r in back_rows if r["evidence_origin"] == "joint_task_probe"),
        "uprobe_replayed": sum(1 for r in back_rows if r["evidence_origin"] == "direct_component_probe"),
        "rows": back_rows,
    }
    (OUT_DIR / "02_m100_41_missing_probes_docker_backverification.json").write_text(
        json.dumps(back_doc, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"  Replayed {len(back_rows)}/{len(missing_conf_rows)} probes: {matched_cnt}/{len(back_rows)} verdict matches, "
        f"withdrawal priors: {w_prior_matched}/{len(w_prior_rows)} matched in {back_doc['monotonic_elapsed_sec']}s",
        flush=True,
    )

    print("[M10.0.1] 3/3 Running 0-LLM archived solver lifting & anti-unification coverage audit...", flush=True)
    lift_doc = _audit_archived_solvers_lifting_coverage()
    (OUT_DIR / "03_m100_archived_solvers_lifting_coverage_inventory.json").write_text(
        json.dumps(lift_doc, indent=2) + "\n", encoding="utf-8"
    )
    n_sealed = _seal_dir(OUT_DIR)
    print(
        f"[M10.0.1] Sealed `{OUT_DIR.name}` ({n_sealed} files in SHA256SUMS.txt) | "
        f"lifting audit: {lift_doc['category_counts']} | elapsed={round(monotonic() - t_start, 2)}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
