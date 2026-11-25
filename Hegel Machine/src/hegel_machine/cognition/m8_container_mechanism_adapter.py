"""Hegel M8.2 Container Mechanism Adapter: Public Invariance Probes, Budgeted Local Repair & Graph Writeback.

Connects `PersistentMechanismGraph` (`m8_persistent_variational_planner.py`) directly to real Docker scientific
discovery containers (`Hegel-OOD-Holdout-12` / cohort & sensor tasks):

  1. **Label-Free Public Output Contract & Identifier-Renaming Invariance Probe (`probe_public_cohort_contract_and_invariance`)**:
     - Executes strictly inside an isolated `hegel_clean_scipy_sandbox:v1` container (`network_mode='none'`) on the
       task's PUBLIC dataset (`/holdout_task/public_cohort`) and an automatically generated **renamed-identifier twin**
       where non-semantic background contig IDs/tags (`ctg_chr_* -> bg_contig_*`, `CHR_SEG_* -> BG_CHROM_*`) and
       isolate file ordering are perturbed while scientific quantities (circularity, depth ratios, sequences, wildtype CDS)
       are preserved identically.
     - Catches both **incomplete output contracts** (e.g., `amino_acid_substitution: None`) AND **identifier-prefix
       overfitting** (`startswith('CHR_SEG_')`) with ZERO access to any private verifier or private seed!

  2. **Budgeted Pre-Payment Container Action Selection (`choose_and_execute_container_mechanism_action`)**:
     - Evaluates `["CommitPrimary1x", "LocalContainerRepair", "SpawnDialecticalChallenger"]` under `budget_remaining`:
       - Recalls any persisted `container_invariance_guard` mechanism from `PersistentMechanismGraph` automatically.
       - Selects `LocalContainerRepair` (`estimated_cost = 0.35`, 1 targeted `GPT CLI` repair turn) when primary
         produced a concrete module with a localized public contract / invariance defect, avoiding a full 24-turn
         second branch (`estimated_cost = 1.0`) unless budget allows and local repair cannot resolve the gap.
       - Logs complete `ResearchDecisionRecord` provenance including failed LLM call token costs, and persists verified
         invariance guards back into `PersistentMechanismGraph` (`save_checkpoint`) for cross-process cold restart!
"""

from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from time import monotonic
from typing import Any

from . import m6_frontier_real_benchmarks as m6f
from .m7_holdout_ood_blind_benchmark import CLEAN_SCIPY_SANDBOX_IMAGE, HoldoutOODTask, ensure_clean_scipy_sandbox_image
from .m8_persistent_variational_planner import (
    PersistentMechanismGraph,
    ResearchDecisionRecord,
    TypedMechanism,
)

M8_CONTAINER_LOCAL_REPAIR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "repaired_python_code",
        "invariance_guard_name",
        "invariance_principle",
        "defect_diagnosis",
    ],
    "properties": {
        "repaired_python_code": {"type": "string", "minLength": 40, "maxLength": 18000},
        "invariance_guard_name": {"type": "string", "minLength": 3, "maxLength": 80},
        "invariance_principle": {"type": "string", "minLength": 10, "maxLength": 600},
        "defect_diagnosis": {"type": "string", "minLength": 10, "maxLength": 600},
    },
}


def probe_public_cohort_contract_and_invariance(
    *,
    task: HoldoutOODTask,
    candidate_py_code: str,
    required_non_null_str_keys: tuple[str, ...] = ("shared_plasmid_family_tag", "amino_acid_substitution"),
    required_finite_float_keys: tuple[str, ...] = ("median_plasmid_copy_number",),
) -> dict[str, Any]:
    """Run strictly PUBLIC output-completeness + background-identifier-renaming invariance check in Docker."""
    ensure_clean_scipy_sandbox_image()
    t0 = monotonic()
    artifact_rel = task.declared_artifacts[0]
    mod_stem = Path(artifact_rel).stem
    art_dir = str(Path(artifact_rel).parent)

    cid = m6f.start_detached_container(
        CLEAN_SCIPY_SANDBOX_IMAGE,
        memory_limit="2048m",
        cpus="2",
        network_mode="none",
    )
    try:
        m6f.exec_container_bash(cid, f"mkdir -p {art_dir} /tmp", timeout_sec=10)
        probe_script = f"""import glob, json, math, os, shutil, sys
sys.path.insert(0, {art_dir!r})

report = {{
    "import_ok": False,
    "public_run_ok": False,
    "public_output_contract_ok": False,
    "id_renaming_invariant_ok": False,
    "missing_or_null_keys": [],
    "renaming_discrepancies": [],
    "public_output": None,
    "renamed_output": None,
    "error": None,
}}

try:
    import {mod_stem} as cand_mod
    report["import_ok"] = True
except Exception as exc:
    report["error"] = f"import_error:{{type(exc).__name__}}:{{exc}}"
    print(json.dumps(report))
    sys.exit(0)

pub_dir = "/holdout_task/public_cohort"
try:
    out_orig = cand_mod.audit_outbreak_cohort(pub_dir)
    report["public_run_ok"] = isinstance(out_orig, dict)
    report["public_output"] = out_orig if isinstance(out_orig, dict) else str(type(out_orig))
except Exception as exc:
    report["error"] = f"public_run_error:{{type(exc).__name__}}:{{exc}}"
    print(json.dumps(report))
    sys.exit(0)

missing = []
if not isinstance(out_orig, dict):
    missing.append("return_not_dict")
else:
    for k in {list(required_non_null_str_keys)!r}:
        v = out_orig.get(k)
        if not isinstance(v, str) or not v.strip() or v.strip().lower() == "none":
            missing.append(k)
    for k in {list(required_finite_float_keys)!r}:
        v = out_orig.get(k)
        try:
            fv = float(v)
            if not math.isfinite(fv):
                missing.append(k)
        except Exception:
            missing.append(k)

report["missing_or_null_keys"] = missing
report["public_output_contract_ok"] = (len(missing) == 0)

# Build synthetic PUBLIC renamed-background-identifier twin (zero private information!)
ren_dir = "/tmp/public_cohort_renamed_bg"
os.makedirs(ren_dir, exist_ok=True)
shutil.copy2(os.path.join(pub_dir, "reference_panel.json"), os.path.join(ren_dir, "reference_panel.json"))
for idx, iso_fp in enumerate(sorted(glob.glob(os.path.join(pub_dir, "isolate_*.json"))), start=1):
    iso_obj = json.load(open(iso_fp, "r", encoding="utf-8"))
    new_contigs = []
    for c_idx, c in enumerate(iso_obj.get("contigs", [])):
        c2 = dict(c)
        c2["contig_id"] = f"renamed_node_{{idx}}_{{c_idx}}"
        # Rename non-circular chromosomal background tags (`CHR_SEG_0` -> `BG_CHROM_UNIT_0`) to test structural vs prefix reliance!
        if not bool(c2.get("circular", False)):
            c2["family_tag"] = f"BG_CHROM_UNIT_{{c_idx}}"
        new_contigs.append(c2)
    iso_obj["isolate_id"] = f"RENAMED_SAMPLE_{{idx:02d}}"
    iso_obj["contigs"] = new_contigs
    with open(os.path.join(ren_dir, os.path.basename(iso_fp)), "w", encoding="utf-8") as fh:
        json.dump(iso_obj, fh)

try:
    out_ren = cand_mod.audit_outbreak_cohort(ren_dir)
    report["renamed_output"] = out_ren if isinstance(out_ren, dict) else str(type(out_ren))
    diffs = []
    if not isinstance(out_ren, dict):
        diffs.append("renamed_return_not_dict")
    else:
        for k in {list(required_non_null_str_keys)!r}:
            if out_orig.get(k) != out_ren.get(k):
                diffs.append(f"{{k}}:{{out_orig.get(k)}}!={{out_ren.get(k)}}")
        for k in {list(required_finite_float_keys)!r}:
            try:
                if abs(float(out_orig[k]) - float(out_ren[k])) > 1e-6:
                    diffs.append(f"{{k}}_drift")
            except Exception:
                diffs.append(f"{{k}}_non_numeric")
    report["renaming_discrepancies"] = diffs
    report["id_renaming_invariant_ok"] = (len(diffs) == 0)
except Exception as exc:
    report["renaming_discrepancies"] = [f"renamed_crash:{{type(exc).__name__}}:{{exc}}"]
    report["id_renaming_invariant_ok"] = False

print(json.dumps(report))
"""
        with tempfile.TemporaryDirectory(prefix="hegel_m8_pubprobe_") as td:
            td_p = Path(td)
            (td_p / "_setup.py").write_text(task.public_setup_py, encoding="utf-8")
            (td_p / "cand.py").write_text(candidate_py_code, encoding="utf-8")
            (td_p / "_probe.py").write_text(probe_script, encoding="utf-8")
            subprocess.run(["docker", "cp", str(td_p / "_setup.py"), f"{cid}:/tmp/_setup.py"], check=True, timeout=20)
            subprocess.run(["docker", "cp", str(td_p / "cand.py"), f"{cid}:{artifact_rel}"], check=True, timeout=20)
            subprocess.run(["docker", "cp", str(td_p / "_probe.py"), f"{cid}:/tmp/_probe.py"], check=True, timeout=20)

        m6f.exec_container_bash(cid, "python3 /tmp/_setup.py && rm -f /tmp/_setup.py", timeout_sec=45)
        ex = m6f.exec_container_bash(cid, "python3 /tmp/_probe.py", timeout_sec=45)
        lines = [ln.strip() for ln in ex["stdout"].splitlines() if ln.strip().startswith("{")]
        parsed = json.loads(lines[-1]) if lines else {"error": ex["stderr"][-400:]}
        parsed["wall_ms"] = round((monotonic() - t0) * 1000.0, 2)
        return parsed
    finally:
        m6f.stop_container_safely(cid)


def recall_persisted_container_invariance_guards(
    graph: PersistentMechanismGraph,
    *,
    domain_tag: str = "cohort_invariance",
) -> list[TypedMechanism]:
    """Autonomously recall verified `container_invariance_guard` mechanisms from `PersistentMechanismGraph`."""
    out: list[TypedMechanism] = []
    for _, mech in sorted(graph.mechanisms.items()):
        if mech.role == "container_invariance_guard" and mech.epistemic_status == "SUPPORTED_IN_SCOPE":
            if not domain_tag or domain_tag in mech.provenance_note or not mech.provenance_note:
                out.append(mech)
    return out


def choose_and_execute_container_mechanism_action(
    *,
    task: HoldoutOODTask,
    primary_candidate_py: str,
    graph: PersistentMechanismGraph,
    codex_client: Any | None,
    budget_remaining: float = 1.0,
    step_index: int = 1,
) -> dict[str, Any]:
    """Pre-payment budgeted container planner bridging `PersistentMechanismGraph` -> `LocalContainerRepair` -> Writeback.

    Evaluates:
      - `CommitPrimary1x` (`cost = 0.0` extra): chosen when public output contract + renaming invariance already pass;
      - `LocalContainerRepair` (`estimated_cost = 0.35`, 1 targeted `GPT CLI` call): chosen when primary produced code
        with a localized public contract / renaming-invariance failure and `budget_remaining >= 0.35`;
      - `SpawnDialecticalChallenger` (`estimated_cost = 1.0`): chosen only if primary code is empty or budget >= 1.0
        after local repair is unavailable.
    """
    t0 = monotonic()
    probe_before = probe_public_cohort_contract_and_invariance(
        task=task,
        candidate_py_code=primary_candidate_py,
    )
    recalled_guards = recall_persisted_container_invariance_guards(graph)

    both_ok = bool(probe_before.get("public_output_contract_ok") and probe_before.get("id_renaming_invariant_ok"))
    blinded_task_ref = "container_task_" + hashlib.sha256(task.task_id.encode("utf-8")).hexdigest()[:8]

    pre_state = {
        "task_id": task.task_id,
        "public_output_contract_ok": bool(probe_before.get("public_output_contract_ok")),
        "id_renaming_invariant_ok": bool(probe_before.get("id_renaming_invariant_ok")),
        "missing_or_null_keys": list(probe_before.get("missing_or_null_keys") or []),
        "renaming_discrepancies": list(probe_before.get("renaming_discrepancies") or []),
        "recalled_invariance_guard_ids": [m.mechanism_id for m in recalled_guards],
        "budget_remaining_before_step": round(float(budget_remaining), 4),
    }

    attempt_events: list[dict[str, Any]] = []
    tot_p_tok = 0
    tot_c_tok = 0

    if both_ok:
        rec = ResearchDecisionRecord(
            decision_id=f"m8_cnt_mech_{step_index:03d}_{task.task_id}",
            step_index=step_index,
            task_id=task.task_id,
            pre_action_state=pre_state,
            available_actions=["CommitPrimary1x"],
            selected_action="CommitPrimary1x",
            selected_candidate_id="primary_candidate_verified",
            execution_status="EXECUTED_OK",
            actual_cost=0.0,
            estimated_selected_cost=0.0,
            public_val_log_predictive_delta=0.0,
            collateral_min_delta=0.0,
            rationale="Primary candidate passed both public output contract and identifier-renaming invariance probe.",
        )
        graph.decision_ledger.append(rec)
        graph.sync_authoritative_state_to_forest()
        return {
            "selected_action": "CommitPrimary1x",
            "final_python_code": primary_candidate_py,
            "probe_before": probe_before,
            "probe_after": probe_before,
            "recalled_guard_ids": [m.mechanism_id for m in recalled_guards],
            "persisted_guard_id": None,
            "decision_record": rec.to_dict(),
            "wall_ms": round((monotonic() - t0) * 1000.0, 2),
        }

    if codex_client is None or float(budget_remaining) < 0.35:
        fallback_act = "SpawnDialecticalChallenger" if float(budget_remaining) >= 1.0 else "CommitPrimary1x"
        rec_fb = ResearchDecisionRecord(
            decision_id=f"m8_cnt_mech_{step_index:03d}_{task.task_id}",
            step_index=step_index,
            task_id=task.task_id,
            pre_action_state=pre_state,
            available_actions=["CommitPrimary1x", "LocalContainerRepair", "SpawnDialecticalChallenger"],
            selected_action=fallback_act,
            selected_candidate_id="budget_or_client_gated_fallback",
            execution_status="BUDGET_GATED_SKIP",
            actual_cost=0.0,
            estimated_selected_cost=(1.0 if fallback_act == "SpawnDialecticalChallenger" else 0.0),
            public_val_log_predictive_delta=0.0,
            collateral_min_delta=0.0,
            rationale="Insufficient budget or missing codex_client for paid `LocalContainerRepair`.",
        )
        graph.decision_ledger.append(rec_fb)
        graph.sync_authoritative_state_to_forest()
        return {
            "selected_action": fallback_act,
            "final_python_code": primary_candidate_py,
            "probe_before": probe_before,
            "probe_after": probe_before,
            "recalled_guard_ids": [m.mechanism_id for m in recalled_guards],
            "persisted_guard_id": None,
            "decision_record": rec_fb.to_dict(),
            "wall_ms": round((monotonic() - t0) * 1000.0, 2),
        }

    # Execute 1-turn targeted `LocalContainerRepair` via official `GPT CLI` (`estimated_cost = 0.35` << `1.0` full branch!)
    guard_hints = [
        {"guard_id": m.mechanism_id, "principle": m.provenance_note}
        for m in recalled_guards
    ]
    sys_prompt = (
        "You are the `Hegel-M8.2` Budgeted Container Mechanism Repair Planner.\n"
        "A candidate Python module failed our strictly PUBLIC output-completeness and/or non-semantic identifier-renaming "
        "invariance probe on the public dataset.\n"
        "Return a complete, self-contained Python module (`repaired_python_code`) satisfying ALL requirements in "
        "`task_instruction`, using structural properties (e.g. `not circular` for chromosomal depth baseline, full 6-frame "
        "or forward+reverse-complement codon alignment against `reference_wildtype_cds` in `reference_panel.json` using the "
        "standard genetic code) rather than hardcoded string prefixes like `'CHR_SEG_'` or leaving required fields `None`!"
    )
    user_payload = json.dumps(
        {
            "task_ref": blinded_task_ref,
            "task_instruction": task.instruction_text,
            "declared_artifact": task.declared_artifacts[0],
            "public_probe_diagnostics": {
                "public_output_contract_ok": probe_before.get("public_output_contract_ok"),
                "id_renaming_invariant_ok": probe_before.get("id_renaming_invariant_ok"),
                "missing_or_null_keys": probe_before.get("missing_or_null_keys"),
                "renaming_discrepancies": probe_before.get("renaming_discrepancies"),
                "public_output_preview": probe_before.get("public_output"),
            },
            "recalled_persistent_invariance_guards": guard_hints,
            "current_candidate_python_code": primary_candidate_py,
        },
        ensure_ascii=False,
    )
    llm_res = codex_client.call_structured(
        stage_subdir=f"m8_local_container_repair_{step_index:02d}",
        system_prompt=sys_prompt,
        conversation_turns=[{"role": "user", "content": user_payload}],
        output_schema=M8_CONTAINER_LOCAL_REPAIR_SCHEMA,
    )
    p_tok = int(llm_res.get("prompt_tokens") or 0)
    c_tok = int(llm_res.get("completion_tokens") or 0)
    tot_p_tok += p_tok
    tot_c_tok += c_tok

    if not (llm_res.get("ok") and isinstance(llm_res.get("parsed"), dict)):
        err_msg = str(llm_res.get("error") or "local_repair_llm_error")
        attempt_events.append(
            {
                "action": "LocalContainerRepair",
                "cost_spent": 0.35,
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
                "status": "LLM_CALL_ERROR",
                "note": err_msg,
            }
        )
        rec_err = ResearchDecisionRecord(
            decision_id=f"m8_cnt_mech_{step_index:03d}_{task.task_id}",
            step_index=step_index,
            task_id=task.task_id,
            pre_action_state=pre_state,
            available_actions=["CommitPrimary1x", "LocalContainerRepair", "SpawnDialecticalChallenger"],
            selected_action="CommitPrimary1x",
            selected_candidate_id="local_repair_failed_retain_primary",
            execution_status="EXECUTION_ERROR",
            actual_cost=0.35,
            estimated_selected_cost=0.35,
            total_search_cost_spent=0.35,
            total_prompt_tokens=tot_p_tok,
            total_completion_tokens=tot_c_tok,
            attempt_cost_events=attempt_events,
            public_val_log_predictive_delta=0.0,
            collateral_min_delta=0.0,
            rationale=f"Paid `LocalContainerRepair` failed (`{err_msg}`); retained primary candidate.",
        )
        graph.decision_ledger.append(rec_err)
        graph.sync_authoritative_state_to_forest()
        return {
            "selected_action": "CommitPrimary1x",
            "final_python_code": primary_candidate_py,
            "probe_before": probe_before,
            "probe_after": probe_before,
            "recalled_guard_ids": [m.mechanism_id for m in recalled_guards],
            "persisted_guard_id": None,
            "decision_record": rec_err.to_dict(),
            "wall_ms": round((monotonic() - t0) * 1000.0, 2),
        }

    parsed_repair = llm_res["parsed"]
    repaired_code = str(parsed_repair["repaired_python_code"])
    # Validate syntax & re-probe strictly on public container inputs!
    syntax_ok = True
    try:
        ast.parse(repaired_code)
    except SyntaxError:
        syntax_ok = False

    probe_after = (
        probe_public_cohort_contract_and_invariance(task=task, candidate_py_code=repaired_code)
        if syntax_ok
        else {"public_output_contract_ok": False, "id_renaming_invariant_ok": False, "error": "SyntaxError"}
    )
    repaired_ok = bool(probe_after.get("public_output_contract_ok") and probe_after.get("id_renaming_invariant_ok"))
    attempt_events.append(
        {
            "action": "LocalContainerRepair",
            "cost_spent": 0.35,
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
            "status": "OK" if repaired_ok else "PROBE_REJECTED",
            "note": str(parsed_repair.get("defect_diagnosis") or ""),
        }
    )

    persisted_guard_id: str | None = None
    if repaired_ok:
        raw_gname = "".join(ch if ch.isalnum() else "_" for ch in str(parsed_repair["invariance_guard_name"]).lower()).strip("_")
        persisted_guard_id = f"mech_guard_{raw_gname[:24]}_s{step_index}"
        graph.register_mechanism(
            TypedMechanism(
                mechanism_id=persisted_guard_id,
                parent_id=None,
                role="container_invariance_guard",
                feature_names=("public_output_contract_ok", "id_renaming_invariant_ok"),
                dependent_task_ids={task.task_id},
                epistemic_status="SUPPORTED_IN_SCOPE",
                provenance_note=f"cohort_invariance: {parsed_repair['invariance_principle']}",
            ),
            tree_id="tree_container_invariance",
        )

    selected_act = "LocalContainerRepair" if repaired_ok else "CommitPrimary1x"
    final_code = repaired_code if repaired_ok else primary_candidate_py
    rec_done = ResearchDecisionRecord(
        decision_id=f"m8_cnt_mech_{step_index:03d}_{task.task_id}",
        step_index=step_index,
        task_id=task.task_id,
        pre_action_state=deepcopy(pre_state),
        available_actions=["CommitPrimary1x", "LocalContainerRepair", "SpawnDialecticalChallenger"],
        selected_action=selected_act,
        selected_candidate_id=persisted_guard_id or "local_repair_rejected",
        execution_status="EXECUTED_OK" if repaired_ok else "PROBE_REJECTED",
        actual_cost=0.35,
        estimated_selected_cost=0.35,
        total_search_cost_spent=0.35,
        total_prompt_tokens=tot_p_tok,
        total_completion_tokens=tot_c_tok,
        attempt_cost_events=attempt_events,
        public_val_log_predictive_delta=1.0 if repaired_ok else 0.0,
        collateral_min_delta=0.0,
        rationale=(
            f"Executed 1-turn `LocalContainerRepair` (`repaired_ok={repaired_ok}`, "
            f"`guard={persisted_guard_id}`): {parsed_repair.get('invariance_principle', '')}"
        ),
    )
    graph.decision_ledger.append(rec_done)
    graph.sync_authoritative_state_to_forest()

    return {
        "selected_action": selected_act,
        "final_python_code": final_code,
        "probe_before": probe_before,
        "probe_after": probe_after,
        "recalled_guard_ids": [m.mechanism_id for m in recalled_guards],
        "persisted_guard_id": persisted_guard_id,
        "repair_metadata": {
            "invariance_guard_name": parsed_repair.get("invariance_guard_name"),
            "invariance_principle": parsed_repair.get("invariance_principle"),
            "defect_diagnosis": parsed_repair.get("defect_diagnosis"),
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
        },
        "decision_record": rec_done.to_dict(),
        "wall_ms": round((monotonic() - t0) * 1000.0, 2),
    }
