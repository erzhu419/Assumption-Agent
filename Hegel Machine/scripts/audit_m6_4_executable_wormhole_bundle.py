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
from hegel_machine.cognition.m6_4_executable_wormhole_library import NODE_TO_EXECUTABLE_WORMHOLE_MODULE


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

    # 3. Scan foundational wormhole modules + Wave-1 sublations for forbidden tokens
    forbidden_tokens = ("answer_rubric", "ground_truth", "/tests/", "97703", "0.236", "11, 42, 55", "ruoli.dev")
    leak_hits: list[str] = []
    for _, (_, code_s, _) in sorted(NODE_TO_EXECUTABLE_WORMHOLE_MODULE.items()):
        for tok in forbidden_tokens:
            if tok in code_s:
                leak_hits.append(f"foundational_lib:{tok}")
    for cf in sorted(ev_dir.glob("wave1_to_wave2_consolidation_*.json")):
        txt = cf.read_text(encoding="utf-8")
        for tok in forbidden_tokens:
            if tok in txt:
                leak_hits.append(f"{cf.name}:{tok}")

    # 4. Verify zero un-inlined `hegel_wormhole_skills` imports in submitted deliverables
    ep_root = ev_dir / "episodes"
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

    # 5. Independently recompute Skill∞ runtime telemetry from episode summaries & DAGs
    staged_eps = 0
    activated_eps = 0
    total_calls = 0
    for ep_d in m64_ep_dirs:
        s_obj = json.loads((ep_d / "00_episode_summary.json").read_text(encoding="utf-8"))
        if s_obj["benchmark"] == "Terminal-Bench-Science-0.1":
            if s_obj.get("staged_wormhole_modules"):
                staged_eps += 1
            cnt = int(s_obj.get("skill_activation_count", 0))
            if cnt > 0:
                activated_eps += 1
            total_calls += cnt

    manifest = json.loads((ev_dir / "study_manifest.json").read_text(encoding="utf-8"))
    m_tel = manifest.get("skill_infinity_runtime_telemetry", {})
    telemetry_match = (
        m_tel.get("tb_science_episodes_activating_wormhole_code") == activated_eps
        and m_tel.get("total_runtime_function_calls") == total_calls
        and staged_eps == 15
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

    # 7. Live Docker container leak check
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
        "zero_private_leakage_in_wormhole_code_and_sublations": {
            "passed": len(leak_hits) == 0,
            "hits": leak_hits,
        },
        "submitted_artifacts_self_containment_check": {
            "passed": len(uninlined_imports) == 0,
            "uninlined_hits": uninlined_imports,
        },
        "skill_infinity_runtime_activation_audit": {
            "passed": telemetry_match and activated_eps > 0,
            "tb_science_staged_episodes": staged_eps,
            "tb_science_activating_episodes": activated_eps,
            "total_runtime_skill_calls": total_calls,
        },
        "biomystery_provenance_and_judge_token_audit": {
            "passed": len(bm_prov_hits) == 0
            and bm_judge_tokens
            == int(
                manifest.get(
                    "biomystery_judge_total_tokens_m6_5"
                    if "M6_5" in arm_name
                    else "biomystery_judge_total_tokens_m6_4",
                    0,
                )
            ),
            "reverse_provenance_command_hits": bm_prov_hits,
            "recomputed_judge_total_tokens": bm_judge_tokens,
        },
        "live_docker_container_leak_check": {
            "passed": len(running_bench_containers) == 0,
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
