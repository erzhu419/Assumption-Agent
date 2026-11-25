#!/usr/bin/env python3
"""Hegel M8.2 Decisive Cold-Restart Continual Lifecycle & Container LocalRepair Litmus.

Executes GPT-6's recommended multi-process continual experiment (`Hegel_M8_1_Audit` §5) across
THREE genuinely separate OS subprocesses (`phase1 -> exit -> phase2 -> exit -> phase3 -> exit`)
using official ChatGPT CLI (`codex`, `auth_mode == "chatgpt"`, `gpt-6.2-sol`) + air-gapped Docker:

  1. **OS Process 1 (`phase1`)**:
     - Initializes `PersistentMechanismGraph` with `shared_sensor_premise` and independent `shared_sentinel_acoustics`.
     - Learns a nonlinear mechanism conduit via live `GPT CLI` (`ConceptLift`, blinded task ref) on `task_p1_nonlinear`
       AND updates `shared_sensor_premise` via `AncestorReweight` (`version: 1 -> 2`), with `0.0` drift on sentinel!
     - Saves authoritative checkpoint `checkpoint_after_p1.json` and terminates OS Process 1.

  2. **OS Process 2 (`phase2`) — Cold Checkpoint Reload (`0` in-memory state from Process 1!)**:
     - Calls `PersistentMechanismGraph.load_checkpoint('checkpoint_after_p1.json')`.
     - **Autonomous Recall (`ReuseMechanism`)**: Registers renamed-column sister task `task_p2_renamed_sister`
       (`columns = ('chan_u', 'chan_v')`) WITHOUT passing any `lift_ids` and with `codex_client=None` -> planner
       autonomously recalls & role-maps the persisted mechanism from Process 1 (`0` LLM calls!), while rejecting
       it on an unrelated linear task (`inapplicable_task_ids` negative scope recorded).
     - **Real Docker Container Public Invariance Probe + 1-Turn `LocalContainerRepair` (`ho12`)**:
       Loads the archived broken `plasmid_audit.py` produced by primary `Antithesis` in M8.1 (`amino_acid_substitution: None`
       + `startswith('CHR_SEG_')` prefix overfitting). Under tight budget `budget_remaining = 0.50` (`< 1.0`, blocking
       `2x` second-branch spawning!), `choose_and_execute_container_mechanism_action`:
         (a) detects BOTH missing output contract AND background-tag renaming failure purely on `/holdout_task/public_cohort`,
         (b) executes 1 targeted `GPT CLI` `LocalContainerRepair` (`cost = 0.35`), passing public contract + renaming invariance,
         (c) verifies on the air-gapped `ho12` private verifier (`strict_pass: False -> True`), and
         (d) writes back the learned `container_invariance_guard` into `checkpoint_after_p2.json`, then terminates OS Process 2!

  3. **OS Process 3 (`phase3`) — Second Cold Restart & Prospective Unseen Variant Verification**:
     - Calls `PersistentMechanismGraph.load_checkpoint('checkpoint_after_p2.json')`.
     - Verifies persisted `version == 2` weights, `interaction_lift`, AND `container_invariance_guard` are all intact.
     - Evaluates a prospective unseen scale-shifted sister variant (`columns = ('probe_a', 'probe_b')`) -> autonomous
       `ReuseMechanism` (`0` LLM calls!) drops RMSE by `> 85%`.
     - Evaluates a prospective unseen out-of-distribution plasmid cohort (`CHROM_BACKBONE_*` tags + `K3E` substitution)
       in Docker -> repaired artifact passes `CommitPrimary1x` (`0` extra LLM calls!), and sentinel drift is `0.0`!
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from time import monotonic
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from hegel_machine.cognition import m6_frontier_real_benchmarks as m6f
from hegel_machine.cognition.m7_holdout_ood_blind_benchmark import (
    CLEAN_SCIPY_SANDBOX_IMAGE,
    holdout_tasks_by_id,
    run_holdout_private_verifier,
)
from hegel_machine.cognition.m8_container_mechanism_adapter import (
    choose_and_execute_container_mechanism_action,
    recall_persisted_container_invariance_guards,
)
from hegel_machine.cognition.m8_persistent_variational_planner import (
    M8VariationalPlanner,
    PersistentMechanismGraph,
    TypedMechanism,
    evaluate_committed_task_model,
)


def compute_execution_identity() -> dict[str, Any]:
    """Record exact git HEAD, dirty working-tree status, and SHA-256 over `src/` + `scripts/` (`M8.2` Provenance Fix)."""
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), capture_output=True, text=True, check=False).stdout.strip()
    st = subprocess.run(["git", "status", "--porcelain"], cwd=str(REPO_ROOT), capture_output=True, text=True, check=False).stdout.strip()
    h = hashlib.sha256()
    for sub in ("src", "scripts"):
        for fp in sorted((REPO_ROOT / sub).rglob("*.py")):
            rel = fp.relative_to(REPO_ROOT).as_posix()
            h.update(rel.encode("utf-8") + b"\0" + fp.read_bytes() + b"\0")
    return {
        "git_head_commit": head,
        "working_tree_dirty": bool(st),
        "dirty_file_count": len([ln for ln in st.splitlines() if ln.strip()]),
        "code_tree_sha256": h.hexdigest(),
    }


def write_sha256_sums(root: Path) -> int:
    lines: list[str] = []
    for fp in sorted(root.rglob("*")):
        if fp.is_file() and fp.name != "SHA256SUMS.txt":
            rel = fp.relative_to(root).as_posix()
            lines.append(f"{hashlib.sha256(fp.read_bytes()).hexdigest()}  {rel}")
    (root / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def _make_sentinel_dataset() -> tuple[dict[str, Any], dict[str, Any]]:
    rng = np.random.default_rng(10101)
    fq = rng.normal(0.0, 1.0, 110)
    dm = rng.normal(0.0, 1.0, 110)
    y = 2.3 * fq - 0.85 * dm + rng.normal(0.0, 0.08, 110)
    return (
        {"x": {"freq_khz": fq[:75], "damping": dm[:75]}, "y": y[:75]},
        {"x": {"freq_khz": fq[75:], "damping": dm[75:]}, "y": y[75:]},
    )


def run_phase_1(out_dir: Path, model: str) -> None:
    t0 = monotonic()
    codex_info = m6f.discover_chatgpt_subscription_codex_cli()
    assert codex_info.get("auth_mode") == "chatgpt", f"Expected chatgpt CLI auth, got {codex_info}"
    client = m6f.FrontierCodexClient(codex_info=codex_info, model=model, reasoning_effort="medium", timeout_seconds=150)

    graph = PersistentMechanismGraph()
    planner = M8VariationalPlanner(graph)

    graph.register_mechanism(
        TypedMechanism(
            mechanism_id="shared_sentinel_acoustics",
            parent_id=None,
            role="shared_upstream_premise",
            feature_names=("freq_khz", "damping"),
            shared_weights={"freq_khz": 2.3, "damping": -0.85},
        ),
        tree_id="tree_sentinel",
    )
    graph.register_mechanism(
        TypedMechanism(
            mechanism_id="shared_sensor_premise",
            parent_id=None,
            role="shared_upstream_premise",
            feature_names=("x1", "x2"),
            shared_weights={"x1": -1.4, "x2": 0.65},
        ),
        tree_id="tree_sensor",
    )

    tr_sent, va_sent = _make_sentinel_dataset()
    planner.register_task(
        task_id="task_sentinel_acoustics",
        shared_upstream_id="shared_sentinel_acoustics",
        feature_names=("freq_khz", "damping"),
        train_data=tr_sent,
        val_data=va_sent,
    )
    sentinel_preds_p1 = evaluate_committed_task_model(
        graph.active_task_models["task_sentinel_acoustics"],
        graph,
        va_sent,
    )["predictions"]

    # 1A. Global shared weight shift (`x1: -1.4 -> +1.55`) across two related tasks -> `AncestorReweight`
    rng = np.random.default_rng(202601)
    for tid in ("task_p1_shift_a", "task_p1_shift_b"):
        x1 = rng.normal(0.0, 1.0, 110)
        x2 = rng.normal(0.0, 1.0, 110)
        y = 1.55 * x1 + 0.65 * x2 + rng.normal(0.0, 0.09, 110)
        planner.register_task(
            task_id=tid,
            shared_upstream_id="shared_sensor_premise",
            feature_names=("x1", "x2"),
            train_data={"x": {"x1": x1[:75], "x2": x2[:75]}, "y": y[:75]},
            val_data={"x": {"x1": x1[75:], "x2": x2[75:]}, "y": y[75:]},
        )
    res_anc = planner.evaluate_and_revise_task("task_p1_shift_a", codex_client=None, budget_remaining=1.0)
    assert res_anc["selected_candidate"]["action_type"] == "AncestorReweight"

    # 1B. Live `GPT CLI` (`gpt-6.2-sol`) `ConceptLift` on nonlinear cross-coupling `1.55*x1 + 0.65*x2 + 1.95 * x1 * tanh(1.6 * x2)`
    xn1 = rng.uniform(-1.8, 1.8, 130)
    xn2 = rng.uniform(-1.8, 1.8, 130)
    yn = 1.55 * xn1 + 0.65 * xn2 + 1.95 * xn1 * np.tanh(1.6 * xn2) + rng.normal(0.0, 0.08, 130)
    planner.register_task(
        task_id="task_p1_primary_discovery",
        shared_upstream_id="shared_sensor_premise",
        feature_names=("x1", "x2"),
        train_data={"x": {"x1": xn1[:90], "x2": xn2[:90]}, "y": yn[:90]},
        val_data={"x": {"x1": xn1[90:], "x2": xn2[90:]}, "y": yn[90:]},
    )
    pre_lift_rmse = evaluate_committed_task_model(
        graph.active_task_models["task_p1_primary_discovery"],
        graph,
        {"x": {"x1": xn1[90:], "x2": xn2[90:]}, "y": yn[90:]},
    )["val_rmse"]
    res_lift = planner.evaluate_and_revise_task("task_p1_primary_discovery", codex_client=client, budget_remaining=4.0)
    post_lift_rmse = evaluate_committed_task_model(
        graph.active_task_models["task_p1_primary_discovery"],
        graph,
        {"x": {"x1": xn1[90:], "x2": xn2[90:]}, "y": yn[90:]},
    )["val_rmse"]

    ckpt_p1 = out_dir / "checkpoints" / "checkpoint_after_p1.json"
    ckpt_meta = graph.save_checkpoint(ckpt_p1)
    np.save(out_dir / "checkpoints" / "sentinel_preds_p1.npy", sentinel_preds_p1)

    p1_report = {
        "phase": "phase_1_os_process_1",
        "pid": subprocess.os.getpid(),
        "wall_ms": round((monotonic() - t0) * 1000.0, 2),
        "ancestor_reweight_action": res_anc["selected_candidate"]["action_type"],
        "shared_sensor_version": graph.mechanisms["shared_sensor_premise"].version,
        "shared_sensor_weights": graph.mechanisms["shared_sensor_premise"].shared_weights,
        "concept_lift_action": res_lift["selected_candidate"]["action_type"],
        "concept_lift_source": res_lift["selected_candidate"]["metadata"].get("source"),
        "concept_lift_expression": res_lift["selected_candidate"]["metadata"].get("expression"),
        "pre_lift_val_rmse": round(pre_lift_rmse, 5),
        "post_lift_val_rmse": round(post_lift_rmse, 5),
        "gpt_cli_tokens": {
            "prompt": res_lift["total_prompt_tokens"],
            "completion": res_lift["total_completion_tokens"],
        },
        "checkpoint": ckpt_meta,
    }
    (out_dir / "01_phase1_process1_summary.json").write_text(json.dumps(p1_report, indent=2) + "\n", encoding="utf-8")


def run_phase_2(out_dir: Path, model: str) -> None:
    t0 = monotonic()
    ckpt_p1 = out_dir / "checkpoints" / "checkpoint_after_p1.json"
    # Cold restart in a brand-new OS process!
    graph = PersistentMechanismGraph.load_checkpoint(ckpt_p1)
    planner = M8VariationalPlanner(graph)

    # Re-attach sentinel dataset to verify 0.0 drift across cold restart
    tr_sent, va_sent = _make_sentinel_dataset()
    graph.task_datasets["task_sentinel_acoustics"] = {"train": tr_sent, "val": va_sent, "local_shrinkage": 12.0}

    # 2A. Autonomous Role-Mapped `ReuseMechanism` on renamed sister task `('chan_u', 'chan_v')` (`0` LLM calls!)
    rng = np.random.default_rng(202602)
    cu = rng.uniform(-1.8, 1.8, 120)
    cv = rng.uniform(-1.8, 1.8, 120)
    yc = 1.55 * cu + 0.65 * cv + 1.85 * cu * np.tanh(1.6 * cv) + rng.normal(0.0, 0.08, 120)
    planner.register_task(
        task_id="task_p2_renamed_sister",
        shared_upstream_id="shared_sensor_premise",
        feature_names=("chan_u", "chan_v"),
        train_data={"x": {"chan_u": cu[:80], "chan_v": cv[:80]}, "y": yc[:80]},
        val_data={"x": {"chan_u": cu[80:], "chan_v": cv[80:]}, "y": yc[80:]},
    )
    pre_reuse_rmse = evaluate_committed_task_model(
        graph.active_task_models["task_p2_renamed_sister"],
        graph,
        {"x": {"chan_u": cu[80:], "chan_v": cv[80:]}, "y": yc[80:]},
    )["val_rmse"]
    res_reuse = planner.evaluate_and_revise_task("task_p2_renamed_sister", codex_client=None, budget_remaining=0.8)
    post_reuse_rmse = evaluate_committed_task_model(
        graph.active_task_models["task_p2_renamed_sister"],
        graph,
        {"x": {"chan_u": cu[80:], "chan_v": cv[80:]}, "y": yc[80:]},
    )["val_rmse"]
    assert res_reuse["selected_candidate"]["action_type"] == "ReuseMechanism", f"Expected ReuseMechanism, got {res_reuse['selected_candidate']}"

    # Negative scope check on unrelated linear task -> must NOT attach nonlinear lift, records `inapplicable_task_ids`!
    lu = rng.normal(0.0, 1.0, 100)
    lv = rng.normal(0.0, 1.0, 100)
    yl = 1.55 * lu + 0.65 * lv + rng.normal(0.0, 0.08, 100)
    planner.register_task(
        task_id="task_p2_negative_scope_linear",
        shared_upstream_id="shared_sensor_premise",
        feature_names=("x1", "x2"),
        train_data={"x": {"x1": lu[:70], "x2": lv[:70]}, "y": yl[:70]},
        val_data={"x": {"x1": lu[70:], "x2": lv[70:]}, "y": yl[70:]},
    )
    res_neg = planner.evaluate_and_revise_task("task_p2_negative_scope_linear", codex_client=None, budget_remaining=0.8)
    assert res_neg["selected_candidate"]["action_type"] == "NoChange"

    # 2B. Real Docker Container Public Invariance Probe + Budgeted 1-Turn `LocalContainerRepair` on archived M8.1 `ho12`!
    codex_info = m6f.discover_chatgpt_subscription_codex_cli()
    client = m6f.FrontierCodexClient(codex_info=codex_info, model=model, reasoning_effort="medium", timeout_seconds=180)
    ho12_task = holdout_tasks_by_id()["ho12__multicopy_plasmid_cohort_consensus"]
    archived_broken_py = (
        REPO_ROOT
        / "docs/evidence/m8_1_offline_replay_and_live_3task_slice_20261125"
        / "part_b_live_3task_16thread_slice/branches"
        / "wave_2__gpt-6.2-sol__ho12__multicopy_plasmid_cohort_consensus"
        / "branch_2_antithesis_wormhole/04_submitted_artifacts_snapshot/plasmid_audit.py"
    ).read_text(encoding="utf-8")

    # Verify post-lock private score BEFORE repair (`False`) vs AFTER repair (`True`), with zero private access during repair!
    def _grade_ho12_code(py_src: str) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="m8_2_ho12_") as td:
            td_p = Path(td)
            mod_fp = td_p / "plasmid_audit.py"
            mod_fp.write_text(py_src, encoding="utf-8")
            tar_fp = td_p / "sub.tar"
            with tarfile.open(tar_fp, "w") as tf:
                tf.add(mod_fp, arcname="holdout_task/plasmid_audit.py")
            return run_holdout_private_verifier(ho12_task, tar_fp)

    ver_before = _grade_ho12_code(archived_broken_py)
    repair_out = choose_and_execute_container_mechanism_action(
        task=ho12_task,
        primary_candidate_py=archived_broken_py,
        graph=graph,
        codex_client=client,
        budget_remaining=0.50,  # `< 1.0` blocks full 24-turn `SpawnDialecticalChallenger`, selecting `LocalContainerRepair` (`0.35`)!
        step_index=len(graph.decision_ledger) + 1,
    )
    assert repair_out["selected_action"] == "LocalContainerRepair", f"Expected LocalContainerRepair, got {repair_out}"
    ver_after = _grade_ho12_code(repair_out["final_python_code"])
    assert ver_before["strict_pass"] is False and ver_after["strict_pass"] is True

    repaired_fp = out_dir / "checkpoints" / "repaired_plasmid_audit_p2.py"
    repaired_fp.write_text(repair_out["final_python_code"], encoding="utf-8")
    ckpt_p2 = out_dir / "checkpoints" / "checkpoint_after_p2.json"
    ckpt_meta2 = graph.save_checkpoint(ckpt_p2)

    p2_report = {
        "phase": "phase_2_os_process_2_cold_restarted",
        "pid": subprocess.os.getpid(),
        "wall_ms": round((monotonic() - t0) * 1000.0, 2),
        "autonomous_sister_reuse": {
            "selected_action": res_reuse["selected_candidate"]["action_type"],
            "recalled_mechanism_id": res_reuse["selected_candidate"]["metadata"].get("recalled_mechanism_id"),
            "role_map": res_reuse["selected_candidate"]["metadata"].get("role_map"),
            "mapped_expression": res_reuse["selected_candidate"]["metadata"].get("mapped_expression"),
            "pre_reuse_val_rmse": round(pre_reuse_rmse, 5),
            "post_reuse_val_rmse": round(post_reuse_rmse, 5),
            "llm_calls_used": 0,
        },
        "negative_scope_control": {
            "selected_action": res_neg["selected_candidate"]["action_type"],
        },
        "container_local_repair_ho12": {
            "budget_remaining_supplied": 0.50,
            "selected_action": repair_out["selected_action"],
            "public_probe_before": {
                "public_output_contract_ok": repair_out["probe_before"]["public_output_contract_ok"],
                "id_renaming_invariant_ok": repair_out["probe_before"]["id_renaming_invariant_ok"],
                "missing_or_null_keys": repair_out["probe_before"]["missing_or_null_keys"],
                "renaming_discrepancies": repair_out["probe_before"]["renaming_discrepancies"],
            },
            "public_probe_after": {
                "public_output_contract_ok": repair_out["probe_after"]["public_output_contract_ok"],
                "id_renaming_invariant_ok": repair_out["probe_after"]["id_renaming_invariant_ok"],
            },
            "private_holdout_strict_pass_before": ver_before["strict_pass"],
            "private_holdout_strict_pass_after": ver_after["strict_pass"],
            "persisted_guard_id": repair_out["persisted_guard_id"],
            "repair_tokens": repair_out["repair_metadata"],
        },
        "checkpoint": ckpt_meta2,
    }
    (out_dir / "02_phase2_process2_summary.json").write_text(json.dumps(p2_report, indent=2) + "\n", encoding="utf-8")


def run_phase_3(out_dir: Path) -> None:
    t0 = monotonic()
    ckpt_p2 = out_dir / "checkpoints" / "checkpoint_after_p2.json"
    # Second cold restart in a THIRD separate OS process!
    graph = PersistentMechanismGraph.load_checkpoint(ckpt_p2)
    planner = M8VariationalPlanner(graph)

    # 3A. Verify exact zero-forgetting on sentinel task from Process 1!
    _, va_sent = _make_sentinel_dataset()
    sentinel_preds_p1 = np.load(out_dir / "checkpoints" / "sentinel_preds_p1.npy")
    sentinel_preds_p3 = evaluate_committed_task_model(
        graph.active_task_models["task_sentinel_acoustics"],
        graph,
        va_sent,
    )["predictions"]
    sentinel_max_drift = float(np.max(np.abs(sentinel_preds_p1 - sentinel_preds_p3)))
    assert sentinel_max_drift == 0.0

    # 3B. Verify persisted `container_invariance_guard` is recalled AND repaired module generalizes zero-shot
    #     (`CommitPrimary1x`, `0` extra LLM calls!) to a prospective unseen OOD cohort (`CHROM_BACKBONE_*` + `'K3E'`)!
    recalled_guards = recall_persisted_container_invariance_guards(graph)
    assert len(recalled_guards) >= 1
    repaired_py = (out_dir / "checkpoints" / "repaired_plasmid_audit_p2.py").read_text(encoding="utf-8")
    ho12_task = holdout_tasks_by_id()["ho12__multicopy_plasmid_cohort_consensus"]

    commit_out = choose_and_execute_container_mechanism_action(
        task=ho12_task,
        primary_candidate_py=repaired_py,
        graph=graph,
        codex_client=None,
        budget_remaining=0.50,
        step_index=len(graph.decision_ledger) + 1,
    )
    assert commit_out["selected_action"] == "CommitPrimary1x"

    # Prospective unseen OOD genomic cohort (`CHROM_BACKBONE_*` tags, `16` isolates, codon 3 `AAG (K) -> GAG (E)` => `'K3E'`)
    cid = m6f.start_detached_container(CLEAN_SCIPY_SANDBOX_IMAGE, memory_limit="2048m", cpus="2", network_mode="none")
    try:
        m6f.exec_container_bash(cid, "mkdir -p /holdout_task /tmp/prospective_cohort", timeout_sec=10)
        prospective_py = """import json, os, sys, numpy as np
sys.path.insert(0, '/holdout_task')
import plasmid_audit

rng = np.random.default_rng(303030)
codons = ['ATG','GCT','AAG','GAA','TTC','GGT','TCT','ATC','GAC','GAT','GCA','TAC','CTG','CCG','CGT','GTG','AAC','TAA']
wt = ''.join(codons)
mut_c = list(codons); mut_c[2] = 'GAG'  # K3E!
mut = ''.join(mut_c)
os.makedirs('/tmp/prospective_cohort', exist_ok=True)
json.dump({'reference_wildtype_cds': wt}, open('/tmp/prospective_cohort/reference_panel.json', 'w'))
bases = np.array(list('ACGT'))
cns = []
for k in range(1, 17):
    cd = float(rng.uniform(35.0, 70.0))
    ctgs = [{'contig_id': f'bb_{i}', 'circular': False, 'mean_depth': round(cd * float(rng.uniform(0.95, 1.05)), 2), 'family_tag': f'CHROM_BACKBONE_{i}', 'sequence': ''.join(rng.choice(bases, 4100))} for i in range(5)]
    if k in (1, 2):
        ctgs.append({'contig_id': 'decoy_phage', 'circular': True, 'mean_depth': round(cd * 18.2, 2), 'family_tag': 'PHAGE_NOISE_99', 'sequence': ''.join(rng.choice(bases, 3000))})
    if k != 7:
        seq = ''.join(rng.choice(bases, 1280)) + mut + ''.join(rng.choice(bases, 1280))
        if k % 2 == 0:
            seq = seq.translate(str.maketrans('ACGT', 'TGCA'))[::-1]
        cn = 4.25 * float(rng.uniform(0.96, 1.04))
        cns.append(cn)
        ctgs.append({'contig_id': f'pl_{k}', 'circular': True, 'mean_depth': round(cd * cn, 2), 'family_tag': 'IncL_OXA48_OUTBREAK', 'sequence': seq})
    json.dump({'isolate_id': f'PROSP_{k:02d}', 'contigs': ctgs}, open(f'/tmp/prospective_cohort/isolate_{k:02d}.json', 'w'))

res = plasmid_audit.audit_outbreak_cohort('/tmp/prospective_cohort')
ok = (res['shared_plasmid_family_tag'] == 'IncL_OXA48_OUTBREAK' and res['amino_acid_substitution'] == 'K3E' and abs(float(res['median_plasmid_copy_number']) - float(np.median(cns))) <= 0.20)
print(json.dumps({'prospective_cohort_pass': bool(ok), 'reported': res}))
"""
        with tempfile.TemporaryDirectory(prefix="m8_2_p3_") as td:
            td_p = Path(td)
            (td_p / "plasmid_audit.py").write_text(repaired_py, encoding="utf-8")
            (td_p / "_prosp.py").write_text(prospective_py, encoding="utf-8")
            subprocess.run(["docker", "cp", str(td_p / "plasmid_audit.py"), f"{cid}:/holdout_task/plasmid_audit.py"], check=True, timeout=20)
            subprocess.run(["docker", "cp", str(td_p / "_prosp.py"), f"{cid}:/tmp/_prosp.py"], check=True, timeout=20)
        ex = m6f.exec_container_bash(cid, "python3 /tmp/_prosp.py", timeout_sec=45)
        prosp_cohort_res = json.loads([ln for ln in ex["stdout"].splitlines() if ln.strip().startswith("{")][-1])
    finally:
        m6f.stop_container_safely(cid)
    assert prosp_cohort_res["prospective_cohort_pass"] is True

    # 3C. Prospective unseen renamed continuous variant `('probe_a', 'probe_b')` -> autonomous `ReuseMechanism` (`0` LLM calls!)
    rng = np.random.default_rng(202603)
    pa = rng.uniform(-1.7, 1.7, 120)
    pb = rng.uniform(-1.7, 1.7, 120)
    yp = 1.55 * pa + 0.65 * pb + 2.15 * pa * np.tanh(1.6 * pb) + rng.normal(0.0, 0.08, 120)
    planner.register_task(
        task_id="task_p3_prospective_unseen",
        shared_upstream_id="shared_sensor_premise",
        feature_names=("probe_a", "probe_b"),
        train_data={"x": {"probe_a": pa[:80], "probe_b": pb[:80]}, "y": yp[:80]},
        val_data={"x": {"probe_a": pa[80:], "probe_b": pb[80:]}, "y": yp[80:]},
    )
    pre_p3_rmse = evaluate_committed_task_model(
        graph.active_task_models["task_p3_prospective_unseen"],
        graph,
        {"x": {"probe_a": pa[80:], "probe_b": pb[80:]}, "y": yp[80:]},
    )["val_rmse"]
    res_p3 = planner.evaluate_and_revise_task("task_p3_prospective_unseen", codex_client=None, budget_remaining=0.8)
    post_p3_rmse = evaluate_committed_task_model(
        graph.active_task_models["task_p3_prospective_unseen"],
        graph,
        {"x": {"probe_a": pa[80:], "probe_b": pb[80:]}, "y": yp[80:]},
    )["val_rmse"]
    assert res_p3["selected_candidate"]["action_type"] == "ReuseMechanism"
    assert post_p3_rmse < 0.14

    p3_report = {
        "phase": "phase_3_os_process_3_prospective_and_sentinel",
        "pid": subprocess.os.getpid(),
        "wall_ms": round((monotonic() - t0) * 1000.0, 2),
        "sentinel_max_abs_prediction_drift": sentinel_max_drift,
        "recalled_container_guard_ids": [g.mechanism_id for g in recalled_guards],
        "repaired_module_reprobe_action": commit_out["selected_action"],
        "prospective_unseen_cohort_result": prosp_cohort_res,
        "prospective_unseen_continuous_reuse": {
            "selected_action": res_p3["selected_candidate"]["action_type"],
            "pre_val_rmse": round(pre_p3_rmse, 5),
            "post_val_rmse": round(post_p3_rmse, 5),
            "llm_calls_used": 0,
        },
        "total_persisted_mechanisms": len(graph.mechanisms),
        "total_ledger_decisions": len(graph.decision_ledger),
    }
    (out_dir / "03_phase3_process3_summary.json").write_text(json.dumps(p3_report, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "docs/evidence/m8_2_cold_restart_autonomous_reuse_and_repair_20261125",
    )
    ap.add_argument("--model", type=str, default="gpt-6.2-sol")
    ap.add_argument("--phase", choices=("orchestrate", "phase1", "phase2", "phase3"), default="orchestrate")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.phase == "phase1":
        run_phase_1(args.out_dir, args.model)
        return
    if args.phase == "phase2":
        run_phase_2(args.out_dir, args.model)
        return
    if args.phase == "phase3":
        run_phase_3(args.out_dir)
        return

    # Orchestrator spawns 3 genuinely separate child OS processes!
    t0 = monotonic()
    pids: list[int] = []
    for ph in ("phase1", "phase2", "phase3"):
        print(f"[M8.2 ORCHESTRATOR] Spawning isolated OS process for `{ph}`...", flush=True)
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--out-dir", str(args.out_dir), "--model", args.model, "--phase", ph],
            cwd=str(REPO_ROOT),
            check=True,
        )

    s1 = json.loads((args.out_dir / "01_phase1_process1_summary.json").read_text(encoding="utf-8"))
    s2 = json.loads((args.out_dir / "02_phase2_process2_summary.json").read_text(encoding="utf-8"))
    s3 = json.loads((args.out_dir / "03_phase3_process3_summary.json").read_text(encoding="utf-8"))
    pids = [s1["pid"], s2["pid"], s3["pid"]]

    master = {
        "experiment_id": "m8_2_cold_restart_autonomous_reuse_and_repair_20261125",
        "execution_identity": compute_execution_identity(),
        "model": args.model,
        "distinct_os_process_pids": pids,
        "total_wall_ms": round((monotonic() - t0) * 1000.0, 2),
        "phase_1_discovery_and_reweight": s1,
        "phase_2_cold_restart_reuse_and_container_repair": s2,
        "phase_3_cold_restart_prospective_and_sentinel": s3,
    }
    (args.out_dir / "00_master_m8_2_cold_restart_summary.json").write_text(
        json.dumps(master, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    n_files = write_sha256_sums(args.out_dir)
    print(f"[M8.2 COMPLETE] 3-process cold-restart continual litmus passed (`{n_files}` files SHA-256 locked)!", flush=True)


if __name__ == "__main__":
    main()
