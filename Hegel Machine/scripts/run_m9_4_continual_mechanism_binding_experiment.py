"""End-to-End Live `gpt-6.2-sol` Continual Runner for `M9.4` (`MechanismExecutableBinding` vs `Strong_Archive_Reconstruct`).

Protocol (`Hegel-M94-Continual`, Zero Leakage, Hidden Schedule, Equal Information, 32-Way Concurrent CLI Pool):
  1. Pre-flight clean-tree provenance capture (`code_commit_at_start`) + global `BoundedSemaphore(max_codex_concurrency=32)`.
  2. Shared `Stage S` Blind Source Formation (`alpha_s00`, `beta_s00`) -> `Aufhebung` `L2` + optional-witness `L1` (`F07`) ->
     freeze `autogenous_forest_v9_m94_post_stage_S.json` + initial `PersistentMechanismGraph` (`sentinel` + `C0` bindings @ `param_v1`)
     BEFORE any continual stream episode begins!
  3. `8` Parallel Independent Continual Streams (`2` Replicates x `2` Equal-Info Arms x `2` Chains), where within each stream
     `ep01 -> ep02 -> ep03 -> ep04` runs strictly sequentially on persistent state:
       - `Strong_Archive_Reconstruct`: Full equal-information archive (`/tmp/hegel_continual_archive/` containing all prior
         verified `.py` deliverables + public metrics, plus online-growing `L1/L2` `/tmp/hegel_wormhole_skills/`), `18`-turn `gpt-6.2-sol`.
       - `Mechanism_Bound_Continual`: Identical assets, disclosures, and budget, plus unified `M8` `PersistentMechanismGraph`
         (`AncestorReweight` shared parameter versioning with `0.0` `sentinel` drift), component-scoped `F01` `LocalSplit` on
         guard falsification, and uniform task-ID-blind 3-valued `ObservablePreconditionGuard` + `ScientificProbeReceipt` routing!
  4. Complete run-scoped UUID provider attempt ledger (`all_provider_call_receipts.json`) + `SHA256SUMS.txt`.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
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
from hegel_machine.cognition.m7_holdout_ood_blind_benchmark import (
    ensure_clean_scipy_sandbox_image,
)
from hegel_machine.cognition.m8_persistent_variational_planner import PersistentMechanismGraph
from hegel_machine.cognition.m9_4_continual_mechanism_binding_world import (
    M94_CHAIN_ALPHA_SOURCE,
    M94_CHAIN_ALPHA_STREAM,
    M94_CHAIN_BETA_SOURCE,
    M94_CHAIN_BETA_STREAM,
    M94BindingRegistry,
    MechanismExecutableBinding,
    evaluate_bound_mechanism_candidate_in_container,
    grade_m94_artifact_map as run_holdout_private_verifier,
    initialize_continual_mechanism_graph_with_sentinel,
    m94_task_to_frontier_task,
    run_m8_shared_calibration_update_from_public_dataset,
)

DEFAULT_V7_BUNDLE = REPO_ROOT / "docs/evidence/m9_2_1_audit_closure_and_regrade_20261126/pretrained_autogenous_forest_v7_m921.json"
DEFAULT_OUT_DIR = REPO_ROOT / "docs/evidence/m9_4_continual_mechanism_binding_world_20261126"

ARMS = (
    "Strong_Archive_Reconstruct",
    "Mechanism_Bound_Continual",
)


def _git_state() -> tuple[str, bool]:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    st = subprocess.run(["git", "status", "--porcelain", "."], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    return sha, bool(st)


def _materialize_public_item_host(task: Any) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="m94_host_pub_") as td:
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


def _inject_equal_info_archive_into_task(
    task: Any,
    prior_archive_entries: list[dict[str, Any]],
    bound_state_json: dict[str, Any] | None = None,
) -> tuple[Any, str]:
    """Stage `/tmp/hegel_continual_archive/` identically inside the episode container for `Strong_Archive_Reconstruct`
    and `Mechanism_Bound_Continual` (`100%` equal historical code + metadata access!).
    """
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
                "solver_path": f"/tmp/hegel_continual_archive/{ep_tag}_solver.py",
                "public_schema_valid": ent.get("public_schema_valid", True),
            }
        )
    extra_lines.append(
        f"open('/tmp/hegel_continual_archive/archive_manifest.json', 'w').write({json.dumps(manifest_rows, indent=2)!r})"
    )
    if bound_state_json is not None:
        extra_lines.append(
            f"open('/tmp/hegel_continual_archive/m94_bound_mechanism_state.json', 'w').write({json.dumps(bound_state_json, indent=2)!r})"
        )

    header = (
        "CONTINUAL STREAM ARCHIVE (`/tmp/hegel_continual_archive/`):\n"
        f"- Prior verified stream deliverables available for inspection/reuse: {[r['solver_path'] for r in manifest_rows]}.\n"
        "- Manifest: `/tmp/hegel_continual_archive/archive_manifest.json`."
    )
    if bound_state_json is not None:
        header += (
            "\n- Persistent Mechanism–Executable Binding State (`/tmp/hegel_continual_archive/m94_bound_mechanism_state.json`): "
            f"`shared_parameter_mechanism={bound_state_json.get('parameter_mechanism_id')}@v{bound_state_json.get('parameter_version')}`, "
            f"`parameter_payload={json.dumps(bound_state_json.get('parameter_payload'))}`, "
            f"`guard_evaluations={json.dumps(bound_state_json.get('guard_evaluations'))}`."
        )

    aug_task = type(task)(
        **{
            **task.__dict__,
            "public_setup_py": task.public_setup_py + "\n" + "\n".join(extra_lines) + "\n",
        }
    )
    return aug_task, header


def _run_one_continual_stream(
    *,
    rep_idx: int,
    arm: str,
    chain_id: str,
    base_forest: Any,
    base_archive: dict[str, dict[str, Any]],
    base_graph_ckpt_path: Path,
    source_archive_entry: dict[str, Any],
    source_forest_nid: str,
    model: str,
    codex_info: dict[str, Any],
    stream_out_dir: Path,
    run_id: str,
    max_turns: int,
    reasoning_effort: str,
) -> dict[str, Any]:
    stream_t0 = monotonic()
    stream_tasks = M94_CHAIN_ALPHA_STREAM if chain_id == "alpha" else M94_CHAIN_BETA_STREAM
    archive = deepcopy(base_archive)
    graph = PersistentMechanismGraph.load_checkpoint(base_graph_ckpt_path)
    forest = graph.forest if arm == "Mechanism_Bound_Continual" else deepcopy(base_forest)
    planner_min_gain = 0.25
    from hegel_machine.cognition.m8_persistent_variational_planner import M8VariationalPlanner

    planner = M8VariationalPlanner(graph, min_gain_nats=planner_min_gain, collateral_tolerance_nats=-0.05)
    registry = M94BindingRegistry(graph)

    prior_entries: list[dict[str, Any]] = [deepcopy(source_archive_entry)]
    episode_results: list[dict[str, Any]] = []
    stream_compiler_receipts: list[dict[str, Any]] = []

    comp_client = m6f.FrontierCodexClient(
        codex_info=codex_info,
        model=model,
        reasoning_effort=reasoning_effort,
        timeout_seconds=300,
    )

    for step_i, task in enumerate(stream_tasks, start=1):
        ep_t0 = monotonic()
        ep_dir = stream_out_dir / task.task_id
        ep_dir.mkdir(parents=True, exist_ok=True)
        pub_item = _materialize_public_item_host(task)

        m8_update_rec: dict[str, Any] | None = None
        bound_trial_rec: dict[str, Any] | None = None
        bound_state_for_prompt: dict[str, Any] | None = None

        if arm == "Mechanism_Bound_Continual":
            # Step 1: Genuine `M8` `AncestorReweight` / `NoChange` shared parameter check (`sentinel` drift == 0.0!)
            m8_update_rec = run_m8_shared_calibration_update_from_public_dataset(
                chain_id=chain_id,
                episode_id=f"rep{rep_idx:02d}_{task.short_name}",
                public_item=pub_item,
                graph=graph,
                planner=planner,
                registry=registry,
            )
            # Step 2: Uniform task-ID-blind 3-valued `ObservablePreconditionGuard` + `ScientificProbeReceipt` bound trial!
            bound_trial_rec = evaluate_bound_mechanism_candidate_in_container(
                task=task,
                chain_id=chain_id,
                graph=graph,
                registry=registry,
                public_item=pub_item,
            )
            if bound_trial_rec.get("accepted_bound_execution") is True:
                cand_code = str(bound_trial_rec["candidate_code"])
                art_path = str(task.declared_artifacts[0])
                bname = Path(art_path).name
                sdir = ep_dir / "04_submitted_artifacts_snapshot"
                sdir.mkdir(parents=True, exist_ok=True)
                (sdir / f"slot_00__{bname}").write_text(cand_code, encoding="utf-8")

                # Grade on held-out private verifier strictly AFTER candidate is sealed!
                priv_ver = run_holdout_private_verifier(task, {art_path: cand_code})
                wall_ms = round((monotonic() - ep_t0) * 1000.0, 2)
                summary = {
                    "run_id": run_id,
                    "replicate": rep_idx,
                    "arm": arm,
                    "chain_id": chain_id,
                    "stream_step": step_i,
                    "task_id": task.task_id,
                    "short_name": task.short_name,
                    "execution_mode": "guard_routed_bound_mechanism_closure",
                    "strict_pass": bool(priv_ver.get("strict_pass")),
                    "reward": 1.0 if priv_ver.get("strict_pass") else 0.0,
                    "turns_used": 0,
                    "ended_by_finish": True,
                    "prompt_tokens": 0,
                    "cached_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "token_cost_proxy": 0,
                    "wall_ms": wall_ms,
                    "m8_shared_parameter_update": m8_update_rec,
                    "bound_mechanism_routing_receipt": {
                        k: v for k, v in bound_trial_rec.items() if k != "candidate_code"
                    },
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
                (ep_dir / "00_episode_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
                (ep_dir / "03_official_verifier_report.json").write_text(json.dumps(priv_ver, indent=2), encoding="utf-8")
                prior_entries.append(
                    {
                        "short_name": task.short_name,
                        "task_id": task.task_id,
                        "submitted_code": cand_code,
                        "public_schema_valid": True,
                    }
                )
                episode_results.append(summary)
                print(
                    f"  [M9.4 STREAM r{rep_idx} | {arm} | {chain_id} | step {step_i} {task.short_name}] "
                    f"BOUND_CLOSURE_DIRECT | pass={summary['strict_pass']} | turns=0 | param_v={m8_update_rec['shared_mechanism_version']} | wall={wall_ms:.0f}ms",
                    flush=True,
                )
                graph.save_checkpoint(stream_out_dir / f"graph_ckpt_after_step_{step_i:02d}.json")
                continue

            bound_state_for_prompt = {
                "parameter_mechanism_id": bound_trial_rec.get("parameter_mechanism_id"),
                "parameter_version": bound_trial_rec.get("parameter_version"),
                "parameter_payload": bound_trial_rec.get("parameter_payload"),
                "pre_llm_bound_trial_reason": bound_trial_rec.get("reason"),
                "guard_evaluations": bound_trial_rec.get("guard_evaluations"),
            }

        # Live `gpt-6.2-sol` Episode Execution (Equal-Information Archive + On-Demand Executable Library!)
        aug_task, ctx_hdr = _inject_equal_info_archive_into_task(task, prior_entries, bound_state_for_prompt)
        ft = m94_task_to_frontier_task(aug_task, continual_context_block=ctx_hdr)
        hippo = HippoRAGWormholeIndex(forest)
        ep_run_tag = f"{run_id}__r{rep_idx:02d}__{arm}__{chain_id}"

        ep_sum = m6f.run_frontier_episode(
            ft,
            "Hegel_M6_9_Antithesis_LiveGPT",
            model=model,
            codex_info=codex_info,
            forest=forest,
            forest_archive=archive,
            hippo_idx=hippo,
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
        priv_ver = (
            run_holdout_private_verifier(task, sub_map)
            if sub_map
            else {"strict_pass": False, "error": "missing_submitted_artifact"}
        )
        ep_sum["strict_pass"] = bool(priv_ver.get("strict_pass"))
        ep_sum["reward"] = 1.0 if priv_ver.get("strict_pass") else 0.0
        ep_sum["execution_mode"] = (
            "llm_synthesis_after_guard_or_interface_gate" if arm == "Mechanism_Bound_Continual" else "llm_archive_reconstruct"
        )
        ep_sum["replicate"] = rep_idx
        ep_sum["arm"] = arm
        ep_sum["chain_id"] = chain_id
        ep_sum["stream_step"] = step_i
        ep_sum["m8_shared_parameter_update"] = m8_update_rec
        ep_sum["bound_mechanism_routing_receipt"] = bound_trial_rec
        ep_sum["verifier_report"] = priv_ver
        (ep_dir / "00_episode_summary.json").write_text(json.dumps(ep_sum, indent=2), encoding="utf-8")
        (ep_dir / "03_official_verifier_report.json").write_text(json.dumps(priv_ver, indent=2), encoding="utf-8")
        print(
            f"  [M9.4 STREAM r{rep_idx} | {arm} | {chain_id} | step {step_i} {task.short_name}] "
            f"{ep_sum['execution_mode']} | pass={ep_sum['strict_pass']} | turns={ep_sum.get('turns_used')} | proxy={ep_sum.get('token_cost_proxy')}",
            flush=True,
        )

        if sub_map:
            cand_txt = next(iter(sub_map.values()))
            prior_entries.append(
                {
                    "short_name": task.short_name,
                    "task_id": task.task_id,
                    "submitted_code": cand_txt,
                    "public_schema_valid": True,
                }
            )
            # Online post-episode consolidation (using ONLY public completion `ended_by_finish == True`, ZERO private peek!)
            c0_guard_falsified = bool(
                bound_trial_rec
                and any(
                    ev.get("branch_tag") == "C0_base" and (ev.get("guard_result") or {}).get("state") == "FALSE"
                    for ev in (bound_trial_rec.get("guard_evaluations") or [])
                )
            )
            if ep_sum.get("ended_by_finish") and (c0_guard_falsified or arm == "Strong_Archive_Reconstruct"):
                ut_cid = m6f.start_detached_container(
                    m7a.CLEAN_SCIPY_SANDBOX_IMAGE, memory_limit="1536m", cpus="2", network_mode="none"
                )
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
                    stream_compiler_receipts.extend(comp_res.get("provider_attempt_receipts") or [])
                    if comp_res.get("compiled_ok"):
                        if arm == "Mechanism_Bound_Continual" and c0_guard_falsified:
                            # `F01` + `F02`: Component-scoped bifurcation explicitly onto `source_forest_nid`!
                            ext_guard = (
                                m7a.ObservablePreconditionGuard(
                                    guard_id=f"guard_{chain_id}_c1_{task.short_name}",
                                    observable_extractor="overstress_excess_max_history_and_hold",
                                    op=">",
                                    value=0.0,
                                    applies_to_component_id="linear_viscoelastic_memory",
                                )
                                if chain_id == "alpha"
                                else m7a.ObservablePreconditionGuard(
                                    guard_id=f"guard_{chain_id}_c1_{task.short_name}",
                                    observable_extractor="nonseparable_2d_additive_residual_ratio",
                                    op=">",
                                    value=1e-6,
                                    applies_to_component_id="separable_kronecker_transport",
                                )
                            )
                            comp_res["regime_bifurcation"] = {
                                **(comp_res.get("regime_bifurcation") or {}),
                                "should_split_node": True,
                            }
                            cons_r = m7a.consolidate_autogenous_episode_into_forest(
                                forest=forest,
                                archive=archive,
                                compilation=comp_res,
                                nogoods_recorded=ep_sum.get("nogoods_recorded") or [],
                                allow_horizontal_wormholes=False,
                                source_node_id=source_forest_nid,
                                source_version=1,
                                violated_component_id=(
                                    "linear_viscoelastic_memory" if chain_id == "alpha" else "separable_kronecker_transport"
                                ),
                                preserved_component_ids=(
                                    ["reference_panel_calibration", "variable_step_gap_recursion"]
                                    if chain_id == "alpha"
                                    else ["sensor_bank_calibration", "nuisance_profiling", "box_simplex_kkt_certification"]
                                ),
                                guard_spec_override=ext_guard,
                            )
                            shared_id = (
                                "alpha_shared_loadcell_calibration"
                                if chain_id == "alpha"
                                else "beta_shared_sensor_bank_calibration"
                            )
                            registry.register_binding(
                                MechanismExecutableBinding(
                                    binding_id=f"bind_{chain_id}_c1_{task.short_name}",
                                    chain_id=chain_id,
                                    branch_tag="C1_extension",
                                    mechanism_id=f"mech_{chain_id}_c1",
                                    mechanism_version=1,
                                    forest_node_id=str(cons_r["target_node_id"]),
                                    component_id=(
                                        "viscoplastic_overstress_extension"
                                        if chain_id == "alpha"
                                        else "nonseparable_channel_b_extension"
                                    ),
                                    preserved_component_ids=(
                                        ["reference_panel_calibration", "variable_step_gap_recursion"]
                                        if chain_id == "alpha"
                                        else ["sensor_bank_calibration", "nuisance_profiling", "box_simplex_kkt_certification"]
                                    ),
                                    violated_component_id=(
                                        "linear_viscoelastic_memory" if chain_id == "alpha" else "separable_kronecker_transport"
                                    ),
                                    module_stem=str(comp_res["module_stem"]),
                                    module_digest=str(comp_res["compiled_code_sha256"]),
                                    function_name=(comp_res["function_names"][0] if comp_res.get("function_names") else "solve"),
                                    qualification_id="hegel_l2_aufhebung_v2",
                                    parameter_mechanism_id=shared_id,
                                    parameter_version=int(graph.mechanisms[shared_id].version),
                                    precondition_guard=ext_guard.to_dict(),
                                    output_schema_keys=(
                                        ["estimated_taus", "viscoplastic_weights", "predicted_strain"]
                                        if chain_id == "alpha"
                                        else ["weights", "terminal_field_channel_a", "terminal_field_channel_b", "kkt_certificate"]
                                    ),
                                    parameterized_closure_code=cand_txt,
                                    parameterized_entrypoint=task.declared_callable_entrypoints[0],
                                    parameterized_closure_sha256=hashlib.sha256(cand_txt.encode("utf-8")).hexdigest(),
                                ),
                                reason=f"component_split_consolidated_on_{task.short_name}",
                            )
                        elif arm == "Strong_Archive_Reconstruct":
                            # Equal-capability online growth for `A_raw` without structured binding/guard graph
                            m7a.consolidate_autogenous_episode_into_forest(
                                forest=forest,
                                archive=archive,
                                compilation=comp_res,
                                nogoods_recorded=ep_sum.get("nogoods_recorded") or [],
                                allow_horizontal_wormholes=False,
                            )
                finally:
                    m6f.stop_container_safely(ut_cid)

        if arm == "Mechanism_Bound_Continual":
            graph.save_checkpoint(stream_out_dir / f"graph_ckpt_after_step_{step_i:02d}.json")
        episode_results.append(ep_sum)

    return {
        "replicate": rep_idx,
        "arm": arm,
        "chain_id": chain_id,
        "wall_sec": round(monotonic() - stream_t0, 2),
        "strict_passes": sum(1 for r in episode_results if r.get("strict_pass")),
        "episode_count": len(episode_results),
        "total_turns_used": sum(int(r.get("turns_used", 0) or 0) for r in episode_results),
        "total_token_cost_proxy": sum(int(r.get("token_cost_proxy", 0) or 0) for r in episode_results),
        "compiler_provider_receipts": stream_compiler_receipts,
        "binding_events": (registry.binding_events if arm == "Mechanism_Bound_Continual" else []),
        "episodes": [
            {
                "stream_step": r["stream_step"],
                "task_id": r["task_id"],
                "short_name": r.get("short_name") or Path(str(r["task_id"])).name.split("__")[0],
                "execution_mode": r["execution_mode"],
                "strict_pass": r["strict_pass"],
                "turns_used": r.get("turns_used", 0),
                "token_cost_proxy": r.get("token_cost_proxy", 0),
                "wall_ms": r.get("wall_ms", 0.0),
                "provider_attempt_accounting": r.get("provider_attempt_accounting"),
                "m8_shared_parameter_update": r.get("m8_shared_parameter_update"),
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
    ap.add_argument("--replicates", type=int, default=2)
    ap.add_argument("--max-turns", type=int, default=18)
    ap.add_argument("--max-stream-workers", type=int, default=8)
    ap.add_argument("--max-codex-concurrency", type=int, default=32)
    args = ap.parse_args()

    out_dir: Path = args.out_dir.resolve()
    commit_start, dirty_start = _git_state()
    if dirty_start:
        raise RuntimeError("Refusing to start M9.4 live experiment on a dirty working tree!")
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_clean_scipy_sandbox_image()
    m64.ensure_runtime_qualification_context(sandbox_cid=None)

    codex_info = m6f.discover_chatgpt_subscription_codex_cli()
    if not (codex_info.get("available") and codex_info.get("auth_mode") == "chatgpt"):
        raise RuntimeError(f"Official ChatGPT Codex CLI unavailable: {codex_info}")
    print(f"[M9.4 INIT] Commit={commit_start[:12]} | Dirty={dirty_start} | Model={args.model} | Concurrency={args.max_codex_concurrency}", flush=True)

    # Global 32-way concurrency throttle + master run-scoped provider call ledger (`F06`)
    cli_sem = threading.BoundedSemaphore(max(1, int(args.max_codex_concurrency)))
    master_receipts_lock = threading.Lock()
    master_provider_receipts: list[dict[str, Any]] = []
    _orig_call = m6f.FrontierCodexClient.call_structured

    def _throttled_and_metered_call(self_cli: Any, **kw: Any) -> dict[str, Any]:
        with cli_sem:
            res = _orig_call(self_cli, **kw)
        with master_receipts_lock:
            master_provider_receipts.append(
                {
                    "stage_subdir": str(kw.get("stage_subdir") or ""),
                    "call_uuid": str(res.get("call_uuid") or ""),
                    "attempts_ledger_filename": str(res.get("attempts_ledger_filename") or ""),
                    "attempts_ledger_path": str(res.get("attempts_ledger_path") or ""),
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

    m6f.FrontierCodexClient.call_structured = _throttled_and_metered_call  # type: ignore[method-assign]
    run_id = "m9_4_continual_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    t0_all = monotonic()

    # =========================================================================
    # PHASE 1: Blind `Stage S` Source Formation (`alpha_s00`, `beta_s00`) & Freeze
    # =========================================================================
    forest_s, archive_s, meta_dag_s, _ = m7a.load_autogenous_forest_bundle(args.v7_bundle)
    src_out_root = out_dir / "stage_S_sources"
    src_tasks = [("alpha", M94_CHAIN_ALPHA_SOURCE), ("beta", M94_CHAIN_BETA_SOURCE)]
    src_results: dict[str, dict[str, Any]] = {}

    def _run_src(chain_k: str, s_task: Any) -> tuple[str, dict[str, Any], dict[str, str]]:
        for att_i in (1, 2):
            s_dir = src_out_root / f"attempt_{att_i:02d}" / s_task.task_id
            ft_s = m94_task_to_frontier_task(s_task)
            ep_a = deepcopy(archive_s)
            ep_a["__autogenous_blank_seed__"] = {"enabled": True}
            ep_s = m6f.run_frontier_episode(
                ft_s,
                "Hegel_M6_9_Antithesis_LiveGPT",
                model=args.model,
                codex_info=codex_info,
                forest=deepcopy(forest_s),
                forest_archive=ep_a,
                hippo_idx=HippoRAGWormholeIndex(forest_s),
                episode_out_dir=s_dir,
                run_id=f"{run_id}__stage_S_att{att_i:02d}",
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
            ver_s = run_holdout_private_verifier(s_task, sub_s) if sub_s else {"strict_pass": False, "error": "missing_submitted_artifact"}
            ep_s["strict_pass"] = bool(ver_s.get("strict_pass"))
            ep_s["verifier_report"] = ver_s
            (s_dir / "00_episode_summary.json").write_text(json.dumps(ep_s, indent=2), encoding="utf-8")
            (s_dir / "03_official_verifier_report.json").write_text(json.dumps(ver_s, indent=2), encoding="utf-8")
            print(f"  [M9.4 STAGE S] {s_task.short_name} (att={att_i}) | pass={ep_s['strict_pass']} | turns={ep_s['turns_used']} | proxy={ep_s['token_cost_proxy']}", flush=True)
            if ep_s["strict_pass"]:
                return chain_k, ep_s, sub_s
        raise RuntimeError(f"Stage S source `{chain_k}` failed private verifier across 2 blind attempts!")

    src_sub_maps: dict[str, dict[str, str]] = {}
    with ThreadPoolExecutor(max_workers=2) as pool_s:
        futs_s = [pool_s.submit(_run_src, ck, st) for ck, st in src_tasks]
        for f_s in as_completed(futs_s):
            ck, ep_s, sub_s = f_s.result()
            src_results[ck] = ep_s
            src_sub_maps[ck] = sub_s
            if not ep_s["strict_pass"]:
                raise RuntimeError(f"Stage S source `{ck}` failed private verifier: {ep_s['verifier_report']}")

    # Compile `Stage S` `L2` + optional-witness `L1` (`F07`) and freeze BEFORE any continual stream starts!
    comp_cli_s = m6f.FrontierCodexClient(
        codex_info=codex_info, model=args.model, reasoning_effort=args.reasoning_effort, timeout_seconds=300
    )
    source_node_by_chain: dict[str, str] = {}
    source_comp_by_chain: dict[str, dict[str, Any]] = {}
    ut_cid_s = m6f.start_detached_container(m7a.CLEAN_SCIPY_SANDBOX_IMAGE, memory_limit="1536m", cpus="2", network_mode="none")
    try:
        for ck, s_task in src_tasks:
            ft_s = m94_task_to_frontier_task(s_task)
            sigs_s = m6f.extract_public_task_signatures(ft_s)
            c_res = m7a.compile_episode_dialectical_sublation(
                episode_id=f"m94_src_{ck}",
                source_slug=s_task.short_name,
                public_task_brief=s_task.instruction_text,
                inferred_signatures=sigs_s,
                raw_python_sources={Path(k).name: v for k, v in src_sub_maps[ck].items()},
                nogoods_recorded=src_results[ck].get("nogoods_recorded") or [],
                parent_node_summary=None,
                client=comp_cli_s,
                sandbox_cid=ut_cid_s,
                stage_subdir=f"{run_id}__src_{ck}_l2",
            )
            c_res["regime_bifurcation"] = {**(c_res.get("regime_bifurcation") or {}), "should_split_node": False}
            cons_s = m7a.consolidate_autogenous_episode_into_forest(
                forest=forest_s,
                archive=archive_s,
                compilation=c_res,
                nogoods_recorded=src_results[ck].get("nogoods_recorded") or [],
                allow_horizontal_wormholes=True,
            )
            source_node_by_chain[ck] = str(cons_s["target_node_id"])
            source_comp_by_chain[ck] = c_res
    finally:
        m6f.stop_container_safely(ut_cid_s)

    l1_sub_res = m7a.sublate_autogenous_forest_into_compositional_primitives(
        forest=forest_s,
        archive=archive_s,
        codex_client=comp_cli_s,
        run_id=f"{run_id}__stage_S_l1",
        require_pretraining_only=False,
        allow_horizontal_wormholes=True,
        max_workers=2,
        include_executable_checker_guidance=True,
        include_optional_witness_guidance=True,
        only_source_episode_ids={"m94_src_alpha", "m94_src_beta"},
    )

    freeze_dir = out_dir / "stage_S_frozen_bundle"
    freeze_dir.mkdir(parents=True, exist_ok=True)
    v9_bundle_path = freeze_dir / "autogenous_forest_v9_m94_post_stage_S.json"
    v9_meta = m7a.serialize_autogenous_forest_bundle(
        forest=forest_s,
        archive=archive_s,
        meta_dag=meta_dag_s,
        out_path=v9_bundle_path,
        extra_metadata={"stage_S_l1_summary": {k: v for k, v in l1_sub_res.items() if k != "events"}},
    )

    # Freeze initial `PersistentMechanismGraph` + `M94BindingRegistry` (`sentinel` + `C0` @ `param_v1`)
    init_graph, _, init_reg = initialize_continual_mechanism_graph_with_sentinel(deepcopy(forest_s))
    init_reg.register_binding(
        MechanismExecutableBinding(
            binding_id="bind_alpha_c0",
            chain_id="alpha",
            branch_tag="C0_base",
            mechanism_id="mech_alpha_c0",
            mechanism_version=1,
            forest_node_id=source_node_by_chain["alpha"],
            component_id="linear_viscoelastic_memory",
            preserved_component_ids=["reference_panel_calibration", "variable_step_gap_recursion"],
            violated_component_id=None,
            module_stem=str(source_comp_by_chain["alpha"]["module_stem"]),
            module_digest=str(source_comp_by_chain["alpha"]["compiled_code_sha256"]),
            function_name=(source_comp_by_chain["alpha"]["function_names"][0] if source_comp_by_chain["alpha"].get("function_names") else "solve"),
            qualification_id="hegel_l1_qual_4tuple_v2",
            parameter_mechanism_id="alpha_shared_loadcell_calibration",
            parameter_version=1,
            precondition_guard=m7a.ObservablePreconditionGuard(
                guard_id="guard_alpha_c0_subyield",
                observable_extractor="overstress_excess_max_history_and_hold",
                op="<=",
                value=0.0,
                applies_to_component_id="linear_viscoelastic_memory",
            ).to_dict(),
            output_schema_keys=["estimated_taus", "hereditary_weights", "predicted_strain", "initial_rate_pred"],
            parameterized_closure_code=next(iter(src_sub_maps["alpha"].values())),
            parameterized_entrypoint="solve_alpha_s00",
            parameterized_closure_sha256=hashlib.sha256(next(iter(src_sub_maps["alpha"].values())).encode("utf-8")).hexdigest(),
        ),
        reason="stage_S_freeze_alpha_c0",
    )
    init_reg.register_binding(
        MechanismExecutableBinding(
            binding_id="bind_beta_c0",
            chain_id="beta",
            branch_tag="C0_base",
            mechanism_id="mech_beta_c0",
            mechanism_version=1,
            forest_node_id=source_node_by_chain["beta"],
            component_id="separable_kronecker_transport",
            preserved_component_ids=["sensor_bank_calibration", "nuisance_profiling", "box_simplex_kkt_certification"],
            violated_component_id=None,
            module_stem=str(source_comp_by_chain["beta"]["module_stem"]),
            module_digest=str(source_comp_by_chain["beta"]["compiled_code_sha256"]),
            function_name=(source_comp_by_chain["beta"]["function_names"][0] if source_comp_by_chain["beta"].get("function_names") else "solve"),
            qualification_id="hegel_l1_qual_4tuple_v2",
            parameter_mechanism_id="beta_shared_sensor_bank_calibration",
            parameter_version=1,
            precondition_guard=m7a.ObservablePreconditionGuard(
                guard_id="guard_beta_c0_separable",
                observable_extractor="nonseparable_2d_additive_residual_ratio",
                op="<=",
                value=1e-6,
                applies_to_component_id="separable_kronecker_transport",
            ).to_dict(),
            output_schema_keys=["weights", "terminal_field", "kkt_certificate"],
            parameterized_closure_code=next(iter(src_sub_maps["beta"].values())),
            parameterized_entrypoint="solve_beta_s00",
            parameterized_closure_sha256=hashlib.sha256(next(iter(src_sub_maps["beta"].values())).encode("utf-8")).hexdigest(),
        ),
        reason="stage_S_freeze_beta_c0",
    )
    base_graph_ckpt = freeze_dir / "m9_4_initial_mechanism_graph_checkpoint.json"
    graph_ckpt_meta = init_graph.save_checkpoint(base_graph_ckpt)

    freeze_receipt = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "v9_forest_bundle": v9_meta,
        "initial_mechanism_graph_checkpoint": graph_ckpt_meta,
        "source_nodes": source_node_by_chain,
        "l1_primitive_sublation": l1_sub_res,
    }
    (freeze_dir / "pre_stream_freeze_receipt.json").write_text(json.dumps(freeze_receipt, indent=2), encoding="utf-8")

    # =========================================================================
    # PHASE 2: `8` Parallel Independent Continual Streams (`ep01 -> ep02 -> ep03 -> ep04`)
    # =========================================================================
    src_arch_entries = {
        "alpha": {
            "short_name": "alpha_s00",
            "task_id": M94_CHAIN_ALPHA_SOURCE.task_id,
            "submitted_code": next(iter(src_sub_maps["alpha"].values())),
            "public_schema_valid": True,
        },
        "beta": {
            "short_name": "beta_s00",
            "task_id": M94_CHAIN_BETA_SOURCE.task_id,
            "submitted_code": next(iter(src_sub_maps["beta"].values())),
            "public_schema_valid": True,
        },
    }

    stream_specs = [
        (rep_i, arm_nm, ch_id)
        for rep_i in range(1, int(args.replicates) + 1)
        for arm_nm in ARMS
        for ch_id in ("alpha", "beta")
    ]
    stream_summaries: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.max_stream_workers))) as pool_st:
        futs_st = {
            pool_st.submit(
                _run_one_continual_stream,
                rep_idx=rep_i,
                arm=arm_nm,
                chain_id=ch_id,
                base_forest=forest_s,
                base_archive=archive_s,
                base_graph_ckpt_path=base_graph_ckpt,
                source_archive_entry=src_arch_entries[ch_id],
                source_forest_nid=source_node_by_chain[ch_id],
                model=args.model,
                codex_info=codex_info,
                stream_out_dir=out_dir / "continual_streams" / f"rep_{rep_i:02d}" / arm_nm / f"chain_{ch_id}",
                run_id=run_id,
                max_turns=args.max_turns,
                reasoning_effort=args.reasoning_effort,
            ): (rep_i, arm_nm, ch_id)
            for rep_i, arm_nm, ch_id in stream_specs
        }
        for f_done in as_completed(futs_st):
            stream_summaries.append(f_done.result())

    stream_summaries.sort(key=lambda r: (r["replicate"], r["arm"], r["chain_id"]))
    commit_end, dirty_end = _git_state()

    arm_agg: dict[str, dict[str, Any]] = {}
    for arm_nm in ARMS:
        a_streams = [s for s in stream_summaries if s["arm"] == arm_nm]
        a_eps = [ep for s in a_streams for ep in s["episodes"]]
        unk_cnt = sum(int((ep.get("provider_attempt_accounting") or {}).get("usage_unknown_attempt_count", 0) or 0) for ep in a_eps)
        by_step = {}
        for st_idx in (1, 2, 3, 4):
            st_eps = [ep for ep in a_eps if ep["stream_step"] == st_idx]
            by_step[f"step_{st_idx}"] = {
                "strict_passes": sum(1 for e in st_eps if e["strict_pass"]),
                "episode_count": len(st_eps),
                "mean_turns": round(float(np.mean([e["turns_used"] for e in st_eps])), 2) if st_eps else 0.0,
                "total_token_cost_proxy": sum(int(e["token_cost_proxy"]) for e in st_eps),
                "guard_routed_direct_count": sum(1 for e in st_eps if e["execution_mode"] == "guard_routed_bound_mechanism_closure"),
            }
        arm_agg[arm_nm] = {
            "strict_passes": sum(1 for e in a_eps if e["strict_pass"]),
            "episode_count": len(a_eps),
            "pass_rate": round(sum(1 for e in a_eps if e["strict_pass"]) / max(len(a_eps), 1), 4),
            "total_turns_used": sum(int(e["turns_used"]) for e in a_eps),
            "mean_turns_per_episode": round(float(np.mean([e["turns_used"] for e in a_eps])), 2) if a_eps else 0.0,
            "total_token_cost_proxy": sum(int(e["token_cost_proxy"]) for e in a_eps),
            "usage_unknown_attempt_count": unk_cnt,
            "cost_accounting_status": "partial_unknown_usage" if unk_cnt > 0 else "complete",
            "by_stream_step": by_step,
        }

    master_summary = {
        "schema": "hegel_m9_4_continual_mechanism_binding_summary_v1",
        "run_id": run_id,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "max_codex_concurrency": args.max_codex_concurrency,
        "code_commit_at_start": commit_start,
        "code_commit_at_end": commit_end,
        "code_dirty_at_start": dirty_start,
        "code_dirty_at_end": dirty_end,
        "wall_clock_sec": round(monotonic() - t0_all, 2),
        "stage_S_sources": {
            ck: {
                "task_id": src_results[ck]["task_id"],
                "strict_pass": src_results[ck]["strict_pass"],
                "turns_used": src_results[ck]["turns_used"],
                "token_cost_proxy": src_results[ck]["token_cost_proxy"],
            }
            for ck in ("alpha", "beta")
        },
        "pre_stream_freeze": {
            "v9_bundle_sha256": v9_meta["bundle_sha256"],
            "initial_graph_checkpoint_sha256": graph_ckpt_meta["checkpoint_sha256"],
            "newly_compiled_l1_count": l1_sub_res["newly_compiled_count"],
        },
        "arm_comparison": arm_agg,
        "provider_ledger_totals": {
            "logical_llm_calls_all_stages": len(master_provider_receipts),
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
    (out_dir / "m9_4_continual_summary.json").write_text(json.dumps(master_summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "all_provider_call_receipts.json").write_text(
        json.dumps(master_provider_receipts, indent=2) + "\n", encoding="utf-8"
    )

    # Remove transient `_codex_stage` AFTER copying whitelisted attempt ledgers so sealed bundle stays compact & pure
    import shutil

    shutil.rmtree(out_dir / "_codex_stage", ignore_errors=True)
    sha_rows = []
    for fp in sorted(out_dir.rglob("*")):
        if fp.is_file() and fp.name != "SHA256SUMS.txt":
            rel_p = fp.relative_to(out_dir).as_posix()
            sha_rows.append(f"{hashlib.sha256(fp.read_bytes()).hexdigest()}  {rel_p}")
    (out_dir / "SHA256SUMS.txt").write_text("\n".join(sha_rows) + "\n", encoding="utf-8")
    print(json.dumps(master_summary["arm_comparison"], indent=2))


if __name__ == "__main__":
    main()
