#!/usr/bin/env python3
"""Offline scope controls on disposable copies; never changes sealed source evidence.

The Docker check is explicitly stubbed to an empty successful response. This is
NOT a live Docker audit. A new local checksum manifest is generated for each
synthetic fixture so the controls test semantic coverage separately from hashes.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get("HEGEL_AUDIT_REPO", str(ROOT.parent / "repo_m691_4001bef56" / "Hegel Machine")))
SOURCE = REPO / "docs/evidence/m6_9_1_a_two_branch_frontier_40ep_20261123"
ARM = "Hegel_M6_9_A_TwoBranch_LiveGPT"
sys.path.insert(0, str(REPO / "src"))
spec = importlib.util.spec_from_file_location("auditor_under_review", REPO / "scripts/audit_m6_4_executable_wormhole_bundle.py")
auditor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auditor)


def audit(path: Path) -> dict:
    with patch.object(auditor.subprocess, "run", return_value=subprocess.CompletedProcess(["docker", "ps"], 0, "", "")):
        data = auditor.audit_m6_4_frontier_bundle(path, ARM)
    return {"all_checks_passed_with_docker_stub": all(v["passed"] for v in data.values()), "checks": data}


def reseal(path: Path) -> None:
    entries = []
    for p in sorted(path.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS.txt":
            entries.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(path).as_posix()}")
    (path / "SHA256SUMS.txt").write_text("\n".join(entries) + "\n")


def repair_record_hashes(data: dict) -> None:
    prev = "0" * 64
    for rec in data["records"]:
        rec["prev_record_sha256"] = prev
        core = {k: v for k, v in rec.items() if k != "record_sha256"}
        rec["record_sha256"] = hashlib.sha256(json.dumps(core, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        prev = rec["record_sha256"]


def main() -> None:
    command_count = 0
    command_hits = []
    terms = ("ebi.ac.uk/ena", "europepmc.org", "ncbi.nlm.nih.gov/geo", "ncbi.nlm.nih.gov/sra", "ncbi.nlm.nih.gov/bioproject", "answer_rubric", "/tests/", "ground_truth")
    for fp in sorted((SOURCE / "branch_trajectories").rglob("01_decision_provenance_dag.json")):
        for rec in json.loads(fp.read_text())["records"]:
            if "CONTAINER_EXECUTION" not in rec.get("decision_type", ""):
                continue
            command_count += 1
            cmd = str(rec.get("evidence_inputs", {}).get("command", "")).lower()
            for term in terms:
                if term in cmd:
                    command_hits.append({"file": str(fp.relative_to(SOURCE)), "decision_seq": rec["decision_seq"], "term": term})
    result = {
        "source_commit": "4001bef56ffa77520c248b853e7fed3535ee9fe4",
        "scope": "Saved evidence and disposable synthetic copies; no online model, container, or scientific rerun.",
        "docker_check_stubbed": True,
        "actual_branch_literal_command_scan": {"commands_scanned": command_count, "terms": terms, "hits": command_hits, "scope": "Literal trace scan only; absence is not a proof against all indirect access."},
        "original_checkout": audit(SOURCE),
        "controls": [],
    }
    with tempfile.TemporaryDirectory(prefix="m691_auditor_scope_") as td:
        fixture = Path(td) / "evidence"
        shutil.copytree(SOURCE, fixture)
        reseal(fixture)
        result["resealed_fixture_baseline"] = audit(fixture)

        # Missing successful-call traces are skipped, not rejected.
        logs = sorted(fixture.rglob("04c_full_skill_activations.jsonl"))
        backup = {p: p.read_bytes() for p in logs}
        for p in logs:
            p.unlink()
        reseal(fixture)
        result["controls"].append({"id": "AUD01_missing_activation_logs", "removed_logs": len(logs), **audit(fixture)})
        for p, content in backup.items():
            p.write_bytes(content)

        # A missing top-level verifier report is also skipped.
        ep = next((fixture / "episodes").glob("*Hegel*tb__clinical-metadata-recovery"))
        vf = ep / "03_official_verifier_report.json"
        original = vf.read_bytes()
        vf.unlink()
        reseal(fixture)
        result["controls"].append({"id": "AUD02_missing_official_report", "removed_file": str(vf.relative_to(fixture)), **audit(fixture)})
        vf.write_bytes(original)

        # Branch post-lock native outcomes are outside the parity check.
        vf = next((fixture / "branch_trajectories").glob("*tb__amr-poisson-optimize/branch_2*/03_postlock_branch_verifier_report.json"))
        original = vf.read_bytes()
        v = json.loads(original)
        old_pass = v["strict_pass"]
        v["strict_pass"] = not old_pass
        vf.write_text(json.dumps(v, indent=2))
        reseal(fixture)
        result["controls"].append({"id": "AUD03_branch_score_parity", "changed_file": str(vf.relative_to(fixture)), "old_strict_pass": old_pass, "new_strict_pass": v["strict_pass"], **audit(fixture)})
        vf.write_bytes(original)

        # Insert a detectable string into a saved branch command, without
        # executing it. Rehash the DAG to isolate scanning coverage.
        df = next((fixture / "branch_trajectories").glob("*bm__hb024/branch_2*/01_decision_provenance_dag.json"))
        original = df.read_bytes()
        d = json.loads(original)
        rec = next(r for r in d["records"] if "CONTAINER_EXECUTION" in r["decision_type"])
        rec["evidence_inputs"]["command"] = "curl https://europepmc.org/SYNTHETIC_OFFLINE_CONTROL_NEVER_EXECUTED"
        repair_record_hashes(d)
        df.write_text(json.dumps(d, indent=2, ensure_ascii=False))
        reseal(fixture)
        result["controls"].append({"id": "AUD04_branch_provenance_command_not_scanned", "changed_file": str(df.relative_to(fixture)), "synthetic_marker_executed": False, **audit(fixture)})
        df.write_bytes(original)

    dest = Path(__file__).with_name("auditor_scope_controls_results.json")
    dest.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"original_sha_check": result["original_checkout"]["checks"]["sha256_manifest_integrity"], "baseline": result["resealed_fixture_baseline"]["all_checks_passed_with_docker_stub"], "controls": [{"id": c["id"], "all_checks_passed_with_docker_stub": c["all_checks_passed_with_docker_stub"]} for c in result["controls"]]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
