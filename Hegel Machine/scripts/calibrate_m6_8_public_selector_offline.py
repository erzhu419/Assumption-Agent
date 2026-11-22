#!/usr/bin/env python3
"""Offline Developmental Verification for `P0b` (Historical Producer Recovery) & `P1b` (Public Selector Regret).

STRICT EPISTEMIC LABELING (`02_plan_and_implementation_changes.md`):
  - Part A (`historical_producer_recovery`) verifies capability retention & producer extraction over sealed
    `M6.3`–`M6.7` provenance logs (including CSV/TSV/NPZ/JSON tasks whose `/tmp` scripts were previously lost).
  - Part B (`offline_public_selector_calibration`) evaluates `select_candidate_on_public_evidence_only` across
    archived `M6.3`–`M6.7` candidate trajectories using ONLY public trajectory/contract observables during
    selection, checking `03_official_verifier_report.json` strictly AFTER selection to measure `oracle_coverage`,
    `selected_native_score`, `selection_regret`, and `false_promotion_rate`.
  - Neither Part A nor Part B is reported as a live blind benchmark generalization score!
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
from typing import Any

HEGEL_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = HEGEL_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hegel_machine.cognition.m6_4_executable_wormhole_library import extract_pure_reusable_functions_from_python
from hegel_machine.cognition.m6_8_b_prime_capability_bridge import (
    recover_historical_producers_from_sealed_dag,
    select_candidate_on_public_evidence_only,
)

ARCHIVED_RUNS: list[tuple[str, str]] = [
    ("M6.3", "m6_3_frontier_28turn_40ep_20261121"),
    ("M6.4", "m6_4_executable_wormhole_frontier_40ep_20261121"),
    ("M6.5", "m6_5_nogood_gated_pull_frontier_40ep_20261122"),
    ("M6.6", "m6_6_unified_voluntary_pull_frontier_40ep_20261122"),
    ("M6.7", "m6_7_host_mechanical_wormhole_frontier_40ep_20261122"),
]


def _extract_public_candidate_features(ep_dir: Path, version_tag: str) -> tuple[dict[str, Any], bool]:
    """Extract ONLY public episode observables (zero access to `03_official_verifier_report` for features!).
    Returns `(public_features_dict, native_verifier_strict_pass_for_posthoc_regret_only)`.
    """
    s_obj = json.loads((ep_dir / "00_episode_summary.json").read_text(encoding="utf-8"))
    v_obj = json.loads((ep_dir / "03_official_verifier_report.json").read_text(encoding="utf-8"))
    native_pass = bool(v_obj.get("strict_pass"))

    is_bm = s_obj.get("benchmark") == "BioMysteryBench-Preview"
    if is_bm:
        ans = str(s_obj.get("submitted_final_answer") or "").strip()
        contract_ok = len(ans) > 0
    else:
        snaps = list(s_obj.get("saved_artifact_snapshots") or [])
        # Note: NPZ tasks (`traffic`) had 0 text snapshots in M6.3-M6.7; infer presence from trajectory/ended_by_finish
        contract_ok = len(snaps) > 0 or bool(s_obj.get("ended_by_finish"))

    ended_ok = bool(s_obj.get("ended_by_finish"))
    unverified_late = not ended_ok
    gate_reps = " ".join(str(x) for x in (s_obj.get("host_wormhole_gate_reports") or []))
    # Uncalibrated coercive probe warnings from M6.7 (`MUST apply Feng-Ngan` or unnested `POST-SELECTION`)
    coercive_or_snoop = ("MUST apply" in gate_reps) or ("POST_SELECTION_F1_INFLATION" in gate_reps and version_tag == "M6.7")

    # Public self-timing on runtime optimization tasks (from trajectory stdout, NOT verifier!)
    traj_fp = ep_dir / "02_human_readable_trajectory.md"
    self_sec = None
    if traj_fp.is_file() and "amr" in str(s_obj.get("task_id", "")):
        txt = traj_fp.read_text(encoding="utf-8", errors="ignore")
        ms = [float(m.group(1)) for m in re.finditer(r"elapsed[=:\s]+([0-9]+\.[0-9]+)\s*s", txt, re.IGNORECASE)]
        if ms:
            self_sec = ms[-1]

    pub = {
        "candidate_id": f"{version_tag}:{s_obj['task_id']}",
        "version_tag": version_tag,
        "contract_accepted": contract_ok,
        "ended_by_verified_commit": ended_ok,
        "has_unverified_late_edit": unverified_late,
        "has_post_selection_snoop_warning": coercive_or_snoop,
        "public_slice_regressions": 1 if coercive_or_snoop else 0,
        "public_honest_cv_score": None,
        "public_self_timed_median_sec": self_sec,
    }
    return pub, native_pass


def run_offline_calibration(evidence_root: Path | None = None) -> dict[str, Any]:
    ev_root = evidence_root or (HEGEL_ROOT / "docs" / "evidence")
    by_task: dict[str, list[tuple[dict[str, Any], bool, Path]]] = {}
    recovery_records: list[dict[str, Any]] = []

    for ver_tag, dir_name in ARCHIVED_RUNS:
        ep_root = ev_root / dir_name / "episodes"
        if not ep_root.is_dir():
            continue
        for ep_d in sorted(ep_root.glob("*__Hegel_*__*")):
            pub_feat, nat_pass = _extract_public_candidate_features(ep_d, ver_tag)
            s_obj = json.loads((ep_d / "00_episode_summary.json").read_text(encoding="utf-8"))
            t_id = str(s_obj["task_id"])
            by_task.setdefault(t_id, []).append((pub_feat, nat_pass, ep_d))

            if nat_pass and s_obj.get("benchmark") == "Terminal-Bench-Science-0.1":
                rec = recover_historical_producers_from_sealed_dag(ep_d)
                py_blob = "\n\n".join(
                    v["code"] for k, v in rec.get("recovered_sources", {}).items() if k.endswith(".py")
                )
                pure_fns = extract_pure_reusable_functions_from_python(py_blob, max_funcs=8) if py_blob else []
                recovery_records.append(
                    {
                        "version": ver_tag,
                        "task_id": t_id,
                        "recovered_file_count": rec["recovered_file_count"],
                        "recovered_tmp_or_workspace_scripts": [
                            p for p in rec["recovered_paths"] if not p.startswith("snapshot::")
                        ],
                        "pure_function_closure_count": len(pure_fns),
                    }
                )

    oracle_native = 0
    selected_native = 0
    incumbent_m63_native = 0
    false_promotions = 0
    promotions_attempted = 0
    per_task_audit: list[dict[str, Any]] = []

    for t_id, items in sorted(by_task.items()):
        cands = [x[0] for x in items]
        truths = [x[1] for x in items]
        any_pass = any(truths)
        if any_pass:
            oracle_native += 1
        inc_pass = bool(truths[0])
        if inc_pass:
            incumbent_m63_native += 1

        sel = select_candidate_on_public_evidence_only(cands, incumbent_index=0)
        sel_idx = int(sel["selected_index"])
        sel_pass = bool(truths[sel_idx])
        if sel_pass:
            selected_native += 1
        if sel_idx != 0:
            promotions_attempted += 1
            if inc_pass and (not sel_pass):
                false_promotions += 1

        per_task_audit.append(
            {
                "task_id": t_id,
                "oracle_any_native_pass": any_pass,
                "incumbent_m63_native_pass": inc_pass,
                "selected_version": cands[sel_idx]["version_tag"],
                "selected_native_pass": sel_pass,
                "selection_mode": sel["selection_mode"],
            }
        )

    n_tasks = len(by_task)
    report = {
        "scope": "DEVELOPMENTAL_OFFLINE_CALIBRATION_ONLY — NOT_A_BLIND_BENCHMARK_SCORE",
        "tasks_evaluated": n_tasks,
        "p0b_historical_producer_recovery": {
            "passing_tb_episodes_scanned": len(recovery_records),
            "episodes_with_recovered_producers": sum(1 for r in recovery_records if r["recovered_file_count"] > 0),
            "episodes_with_recovered_tmp_scripts": sum(
                1 for r in recovery_records if len(r["recovered_tmp_or_workspace_scripts"]) > 0
            ),
            "total_pure_function_closures_recovered": sum(r["pure_function_closure_count"] for r in recovery_records),
            "sample_recoveries": recovery_records[:12],
        },
        "p1b_public_selector_regret_vs_oracle": {
            "oracle_native_coverage": f"{oracle_native}/{n_tasks}",
            "oracle_native_count": oracle_native,
            "incumbent_m63_native_count": incumbent_m63_native,
            "selected_native_count": selected_native,
            "selection_regret_vs_oracle": oracle_native - selected_native,
            "net_gain_vs_incumbent": selected_native - incumbent_m63_native,
            "promotions_attempted": promotions_attempted,
            "false_promotions": false_promotions,
            "false_promotion_rate": round(false_promotions / max(1, promotions_attempted), 4),
            "per_task_selection": per_task_audit,
        },
    }
    return report


def main() -> int:
    rep = run_offline_calibration()
    out_fp = HEGEL_ROOT / "docs" / "evidence" / "m6_8_offline_producer_and_selector_calibration_20261122.json"
    out_fp.write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
