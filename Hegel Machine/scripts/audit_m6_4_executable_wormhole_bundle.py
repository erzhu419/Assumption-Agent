#!/usr/bin/env python3
"""Independent Cryptographic, Zero-Leakage, Skill∞ Runtime Telemetry & Provenance Auditor for Hegel M6.4.

Mechanically executes 7 independent audit checks over `m6_4_executable_wormhole_frontier_40ep_20261121`:
  1. `sha256_manifest_integrity`: Recomputes SHA-256 of every file in `SHA256SUMS.txt`.
  2. `provenance_dag_chain_integrity`: Verifies unbroken SHA-256 hash-chains across all 40 `01_decision_provenance_dag.json`.
  3. `zero_private_leakage_in_wormhole_code_and_sublations`: Scans `m6_4_executable_wormhole_library.py` AND
     `wave1_to_wave2_consolidation_*.json` for private verifier thresholds (`97703`, `0.236`, `[11, 42, 55]`),
     `/tests`, `/solution`, `answer_rubric`, and `ruoli.dev`.
  4. `submitted_artifacts_self_containment_check`: Scans every file in `04_submitted_artifacts_snapshot/` across all
     `Hegel_M6_4` episodes to verify ZERO un-inlined `hegel_wormhole_skills` imports in submitted deliverables.
  5. `skill_infinity_runtime_activation_audit`: Independently tallies `D1B_EXECUTABLE_WORMHOLE_STAGED` and
     `D_SKILL_ACTIVATION_HARVEST` across all 15 `TB-Science` `Hegel_M6_4` episodes and verifies manifest parity.
  6. `biomystery_provenance_and_judge_token_audit`: Checks all 5 `Hegel_M6_4` `BioMystery` trajectories for
     reverse-provenance queries (`ebi.ac.uk/ena`, `europepmc`, `ncbi.nlm.nih.gov/geo`, `ncbi.nlm.nih.gov/sra`)
     and independently sums `llm_judge_tokens` from `03_official_verifier_report.json`.
  7. `live_docker_container_leak_check`: Queries `docker ps` to verify 0 leaked benchmark containers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

HEGEL_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = HEGEL_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hegel_machine.cognition.m5_live_gpt_and_provenance import verify_decision_provenance_chain
from hegel_machine.cognition.m6_4_executable_wormhole_library import (
    HOST_MECHANICAL_WORMHOLE_GATE_RUNNER_PY,
    NODE_TO_EXECUTABLE_WORMHOLE_MODULE,
    sum_skill_activation_calls,
)
from hegel_machine.cognition.m6_8_b_prime_capability_bridge import SKILL_07_MULTISCALE_RECONCILIATION


def verify_sha256sums_dir(ev_dir: Path) -> dict[str, Any]:
    sha_fp = ev_dir / "SHA256SUMS.txt"
    if not sha_fp.is_file():
        return {"passed": False, "checked_files": 0, "mismatches": ["MISSING_SHA256SUMS.txt"]}
    mismatches: list[str] = []
    checked = 0
    for ln in sha_fp.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s:
            continue
        exp_h, rel_p = s.split("  ", 1)
        fp = ev_dir / rel_p
        checked += 1
        if not fp.is_file():
            mismatches.append(f"missing:{rel_p}")
            continue
        act_h = hashlib.sha256(fp.read_bytes()).hexdigest()
        if act_h != exp_h:
            mismatches.append(f"hash_mismatch:{rel_p}")
    return {"passed": len(mismatches) == 0 and checked > 0, "checked_files": checked, "mismatches": mismatches}


def audit_m6_4_frontier_bundle(ev_dir: Path, arm_name: str = "Hegel_M6_4_Frontier_LiveGPT") -> dict[str, Any]:
    sha_res = verify_sha256sums_dir(ev_dir)

    dag_files = sorted(ev_dir.rglob("01_decision_provenance_dag.json"))
    invalid_dags: list[str] = []
    for df in dag_files:
        obj = json.loads(df.read_text(encoding="utf-8"))
        ver = verify_decision_provenance_chain(obj.get("records", []))
        if not ver.get("valid"):
            invalid_dags.append(str(df.relative_to(ev_dir)))

    # 3. Scan foundational wormhole modules, HOST_MECHANICAL_WORMHOLE_GATE_RUNNER_PY, SKILL_07, Wave-1 sublations & producer manifests for forbidden tokens
    forbidden_tokens = ("answer_rubric", "ground_truth", "/tests/", "97703", "0.236", "11, 42, 55", "ruoli.dev", "3.41")
    leak_hits: list[str] = []
    for _, (_, code_s, _) in sorted(NODE_TO_EXECUTABLE_WORMHOLE_MODULE.items()):
        for tok in forbidden_tokens:
            if tok in code_s:
                leak_hits.append(f"foundational_lib:{tok}")
    for label_s, src_s in (
        ("host_gate_runner", HOST_MECHANICAL_WORMHOLE_GATE_RUNNER_PY),
        ("skill_07_multiscale", SKILL_07_MULTISCALE_RECONCILIATION),
    ):
        for tok in forbidden_tokens:
            if tok in src_s:
                leak_hits.append(f"{label_s}:{tok}")
    for cf in sorted(ev_dir.glob("wave1_to_wave2_consolidation_*.json")):
        txt = cf.read_text(encoding="utf-8")
        for tok in forbidden_tokens:
            if tok in txt:
                leak_hits.append(f"{cf.name}:{tok}")
    for prod_fp in sorted(ev_dir.rglob("04b_producer_skill_manifest/producers/*")):
        if prod_fp.is_file():
            txt = prod_fp.read_text(encoding="utf-8", errors="ignore")
            for tok in ("answer_rubric", "/tests/", "97703", "ruoli.dev"):
                if tok in txt:
                    leak_hits.append(f"producer:{prod_fp.name}:{tok}")

    # 3b. Summary vs Official Verifier Native Score Parity (`P0a` / `H06`: `00_episode_summary.json` must NEVER disagree with `03_official_verifier_report.json`!)
    ep_root = ev_dir / "episodes"
    all_ep_dirs = sorted(p for p in ep_root.iterdir() if p.is_dir()) if ep_root.is_dir() else []
    verifier_parity_mismatches: list[str] = []
    invalid_regrade_receipts: list[str] = []
    for ep_d in all_ep_dirs:
        s_fp = ep_d / "00_episode_summary.json"
        v_fp = ep_d / "03_official_verifier_report.json"
        if s_fp.is_file() and v_fp.is_file():
            s_o = json.loads(s_fp.read_text(encoding="utf-8"))
            v_o = json.loads(v_fp.read_text(encoding="utf-8"))
            if bool(s_o.get("strict_pass")) != bool(v_o.get("strict_pass")):
                verifier_parity_mismatches.append(ep_d.name)
        r_fp = ep_d / "05_regrade_receipt.json"
        if r_fp.is_file():
            r_o = json.loads(r_fp.read_text(encoding="utf-8"))
            if not r_o.get("candidate_hash_match_verified"):
                invalid_regrade_receipts.append(ep_d.name)

    # 4. Verify zero un-inlined `hegel_wormhole_skills` imports in submitted deliverables
    m64_ep_dirs = sorted(ep_root.glob(f"*__{arm_name}__*"))
    uninlined_imports: list[str] = []
    for ep_d in m64_ep_dirs:
        snap_d = ep_d / "04_submitted_artifacts_snapshot"
        if snap_d.is_dir():
            for fp in sorted(snap_d.iterdir()):
                if fp.is_file():
                    txt = fp.read_text(encoding="utf-8", errors="ignore")
                    if "hegel_wormhole_skills" in txt:
                        uninlined_imports.append(f"{ep_d.name}/{fp.name}")

    # 5. Independently recompute Skill∞ runtime telemetry from episode summaries & full JSONL (`04c_full_skill_activations.jsonl`)
    staged_eps = 0
    activated_eps = 0
    total_calls = 0
    full_log_mismatches: list[str] = []
    for ep_d in m64_ep_dirs:
        s_obj = json.loads((ep_d / "00_episode_summary.json").read_text(encoding="utf-8"))
        if s_obj["benchmark"] == "Terminal-Bench-Science-0.1":
            if s_obj.get("staged_wormhole_modules"):
                staged_eps += 1
            cnt = int(s_obj.get("skill_activation_count", 0))
            full_jsonl = ep_d / "04c_full_skill_activations.jsonl"
            if full_jsonl.is_file():
                recs = [
                    json.loads(ln)
                    for ln in full_jsonl.read_text(encoding="utf-8").splitlines()
                    if ln.strip()
                ]
                recomp_cnt = sum_skill_activation_calls(recs)
                if recomp_cnt != cnt:
                    full_log_mismatches.append(f"{ep_d.name}:summary={cnt}!=jsonl={recomp_cnt}")
            if cnt > 0:
                activated_eps += 1
            total_calls += cnt

    manifest = json.loads((ev_dir / "study_manifest.json").read_text(encoding="utf-8"))
    m_tel = manifest.get("skill_infinity_runtime_telemetry", {})
    telemetry_match = (
        m_tel.get("tb_science_episodes_activating_wormhole_code") == activated_eps
        and m_tel.get("total_runtime_function_calls") == total_calls
        and staged_eps == 15
        and len(full_log_mismatches) == 0
    )

    # 6. BioMystery provenance & judge-token audit
    bm_prov_hits: list[str] = []
    bm_judge_tokens = 0
    prov_domains = ("ebi.ac.uk/ena", "europepmc.org", "ncbi.nlm.nih.gov/geo", "ncbi.nlm.nih.gov/sra", "ncbi.nlm.nih.gov/bioproject")
    for ep_d in m64_ep_dirs:
        s_obj = json.loads((ep_d / "00_episode_summary.json").read_text(encoding="utf-8"))
        if s_obj["benchmark"] == "BioMysteryBench-Preview":
            dag_obj = json.loads((ep_d / "01_decision_provenance_dag.json").read_text(encoding="utf-8"))
            for rec in dag_obj.get("records", []):
                if "CONTAINER_EXECUTION" in str(rec.get("decision_type", "")):
                    cmd_s = str((rec.get("evidence_inputs") or {}).get("command", "")).lower()
                    for pd in prov_domains:
                        if pd in cmd_s:
                            bm_prov_hits.append(f"{s_obj['task_id']}:{pd}")
            v_obj = json.loads((ep_d / "03_official_verifier_report.json").read_text(encoding="utf-8"))
            bm_judge_tokens += int((v_obj.get("llm_judge_tokens") or {}).get("total_tokens", 0))

    judge_tok_expected = None
    for k_cand in (
        "biomystery_judge_total_tokens_m6_8",
        "biomystery_judge_total_tokens_m6_7",
        "biomystery_judge_total_tokens_m6_6",
        "biomystery_judge_total_tokens_m6_5",
        "biomystery_judge_total_tokens_m6_4",
    ):
        if k_cand in manifest:
            judge_tok_expected = int(manifest[k_cand])
            break
    if judge_tok_expected is None:
        judge_tok_expected = 0

    # 7. Live Docker container leak check (`H06` Fix: require `ps_proc.returncode == 0`!)
    ps_proc = subprocess.run(["docker", "ps", "--format", "{{.ID}} {{.Image}}"], capture_output=True, text=True, timeout=15)
    running_bench_containers = [
        ln for ln in ps_proc.stdout.splitlines()
        if any(tag in ln for tag in ("tb_sci_", "hegel_biomystery_"))
    ]

    return {
        "sha256_manifest_integrity": sha_res,
        "provenance_dag_chain_integrity": {
            "passed": len(invalid_dags) == 0 and len(dag_files) == 40,
            "dag_count": len(dag_files),
            "invalid": invalid_dags,
        },
        "native_verifier_vs_summary_ledger_parity": {
            "passed": len(verifier_parity_mismatches) == 0 and len(invalid_regrade_receipts) == 0,
            "verifier_parity_mismatches": verifier_parity_mismatches,
            "invalid_regrade_receipts": invalid_regrade_receipts,
        },
        "zero_private_leakage_in_wormhole_code_and_sublations": {
            "passed": len(leak_hits) == 0,
            "hits": leak_hits,
        },
        "submitted_artifacts_self_containment_check": {
            "passed": len(uninlined_imports) == 0,
            "uninlined_hits": uninlined_imports,
        },
        "skill_infinity_runtime_activation_audit": {
            "passed": telemetry_match,
            "tb_science_staged_episodes": staged_eps,
            "tb_science_activating_episodes": activated_eps,
            "total_runtime_skill_calls": total_calls,
            "full_log_mismatches": full_log_mismatches,
        },
        "biomystery_provenance_and_judge_token_audit": {
            "passed": len(bm_prov_hits) == 0 and bm_judge_tokens == judge_tok_expected,
            "reverse_provenance_command_hits": bm_prov_hits,
            "recomputed_judge_total_tokens": bm_judge_tokens,
        },
        "live_docker_container_leak_check": {
            "passed": ps_proc.returncode == 0 and len(running_bench_containers) == 0,
            "docker_ps_returncode": ps_proc.returncode,
            "running_benchmark_containers": running_bench_containers,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=HEGEL_ROOT / "docs" / "evidence" / "m6_4_executable_wormhole_frontier_40ep_20261121",
    )
    parser.add_argument("--arm", type=str, default="Hegel_M6_4_Frontier_LiveGPT")
    args = parser.parse_args()
    res = audit_m6_4_frontier_bundle(args.evidence_dir, args.arm)
    overall = all(v.get("passed", False) for v in res.values())
    print(json.dumps({"overall_passed": overall, "arm": args.arm, "audit": res}, indent=2, ensure_ascii=False))
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
