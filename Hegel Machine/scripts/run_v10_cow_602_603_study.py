"""Run authentic mounted COW census and non-destructive v10 comparison on 602 & 603.

Must be executed in the mount namespace where ``/mnt/hegel-snapshot-cow`` is mounted.
Strictly obeys AGENTS.md:
- Never modifies, overwrites, or appends to any closed historical directory on
  ``/mnt/hegel-snapshot-cow`` (verifies byte size and ``st_mtime_ns`` before & after).
- Uses ``checkpoint_storage.write_json_checkpoint`` for the new separately-ledgered
  output directory ``/mnt/hegel-snapshot-cow/hegel-v10-cow-602-603-successor-20261116``,
  verifying physical footprint < 5 MB.
- Updates ``docs/evidence/v10_world_structure_20261116/paid_position_round1_census_20261116.json``
  with ``cow_mounted: true`` and writes
  ``docs/evidence/v10_world_structure_20261116/v10_cow_602_603_new_ledger_20261116.json``.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
COW_ROOT = Path("/mnt/hegel-snapshot-cow")
COW_602_DIR = COW_ROOT / "hegel-v9-composed-abc-20261101" / "secondary"
COW_603_DIR = COW_ROOT / "hegel-v9-position-method-successor-20261107"
NEW_COW_OUTPUT = COW_ROOT / "hegel-v10-cow-602-603-successor-20261116"
EVIDENCE_DIR = REPO_ROOT / "docs" / "evidence" / "v10_world_structure_20261116"

sys.path[:0] = [
    str(REPO_ROOT / "src"),
    str(REPO_ROOT),
    str(COW_603_DIR / "runtime" / "scripts"),
]

from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.checkpoint_storage import write_json_checkpoint
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.world_structure_commitments import (
    consolidate_world_structure_commitments,
)
from scripts.run_v10_world_structure_study import run_cow_paid_position_census
from run_partial_role_parent import evaluate_reports, load
from run_partial_role_successors import program_knowledge


CLOSED_GUARD_FILES = (
    COW_ROOT / "hegel-v8-paid-interval-confirmation-20261029" / "receipt.json",
    COW_ROOT / "hegel-v9-composed-abc-20261101" / "receipt.json",
    COW_602_DIR / "receipt.json",
    COW_603_DIR / "receipt.json",
    COW_ROOT / "hegel-v10-paid-position-parent-return-20261114" / "receipt.json",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _snapshot_guards() -> dict[str, dict[str, int]]:
    snap = {}
    for p in CLOSED_GUARD_FILES:
        st = p.stat()
        snap[str(p)] = {"size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}
    return snap


def _disk_free_mb(path: Path) -> float:
    st = os.statvfs(path)
    return float(st.f_bavail * st.f_frsize) / (1024.0 * 1024.0)


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for fn in files:
            fp = Path(root) / fn
            if not fp.is_symlink():
                total += fp.stat().st_size
    return total


def _summarize_v10_actions(actions: list[dict]) -> dict:
    audits = [a for a in actions if a.get("action") == "audit_transition_premises"]
    consolidations = [
        a for a in actions if a.get("action") == "consolidate_world_structure_commitments"
    ]
    applications = [
        a for a in actions if a.get("action") == "apply_world_structure_commitments"
    ]
    return {
        "consolidate_calls": len(consolidations),
        "audit_calls": len(audits),
        "last_audit": audits[-1] if audits else None,
        "commitment_applications": applications,
    }


def _run_arm(
    arm_name: str,
    task: dict,
    base_rc: dict,
    measurement_budget: int,
    library,
    *,
    enable_v10: bool,
    out_dir: Path,
) -> tuple[dict, object]:
    rc = dict(base_rc)
    rc.update(
        world_structure_commitments=enable_v10,
        transition_premise_audit=enable_v10,
        expanded_premise_revisions=enable_v10,
    )
    lib_copy = deepcopy(library)
    agent = UnifiedResearchAgent(
        ResearchConfig(**rc),
        measurement_budget=measurement_budget,
        library=lib_copy,
    )
    t0 = time.monotonic()
    with AnonymousObjectLab(**task) as lab:
        report = agent.run(lab)
    elapsed = time.monotonic() - t0

    predictor_available = bool(report.get("program") is not None)
    arm_dir = out_dir / arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    if predictor_available:
        frozen = {
            key: report[key]
            for key in ("program", "fallback", "selected_channels", "selected_kind", "status")
        }
        write_json_checkpoint(arm_dir / "frozen_predictor.json", frozen)

    compact_summary = {
        "arm": arm_name,
        "seed": task["seed"],
        "enable_v10": enable_v10,
        "status": report["status"],
        "predictor_available": predictor_available,
        "selected_kind": report.get("selected_kind"),
        "selected_channels": report.get("selected_channels"),
        "confirmation_loss": report.get("confirmation_loss"),
        "costs": report["costs"],
        "model_calls": 0,
        "reported_tokens": 0,
        "wall_seconds": round(elapsed, 3),
        "v10_diagnostics": _summarize_v10_actions(report.get("actions", [])),
        "meta_priors_after_count": len(lib_copy.meta_priors),
    }
    write_json_checkpoint(arm_dir / "summary.json", compact_summary)
    return compact_summary, lib_copy


def _evaluate_arm_predictors(task: dict, terminal_spec: dict, out_dir: Path, arms: list[str]) -> dict:
    frozen_reports = {}
    for arm in arms:
        fp = out_dir / arm / "frozen_predictor.json"
        if fp.exists():
            frozen_reports[arm] = json.loads(fp.read_text(encoding="utf-8"))
    if not frozen_reports:
        return {
            "arms": {
                arm: {"prediction_pass": False, "prediction_measured": False}
                for arm in arms
            },
            "shared_actual_costs": {"observation_cost": 0, "measurement_cost": 0, "compute_cost": 0},
        }
    term = deepcopy(terminal_spec)
    term["random_seed"] = task["seed"] + term.pop("random_seed_offset")
    term["background_seed"] = task["seed"] + term.pop("background_seed_offset")
    term_dir = out_dir / "terminal"
    eval_summary = evaluate_reports(task, frozen_reports, term, term_dir)
    # Remove bulky raw terminal observation dump on COW after scoring while keeping summary.json
    for bulky in ("paid_observations.json", "raw_predictions.json"):
        bp = term_dir / bulky
        if bp.exists():
            bp.unlink()
    for arm in arms:
        if arm not in eval_summary["arms"]:
            eval_summary["arms"][arm] = {
                "prediction_pass": False,
                "prediction_measured": False,
                "reason": "no_predictor_delivered",
            }
    return eval_summary


def main() -> None:
    if not (COW_ROOT.exists() and COW_603_DIR.exists() and COW_602_DIR.exists()):
        raise RuntimeError("/mnt/hegel-snapshot-cow is not mounted in this namespace")

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    guards_before = _snapshot_guards()
    free_mb_before = _disk_free_mb(COW_ROOT)

    print("[1/3] Running authentic mounted read-only COW census...", flush=True)
    census_payload = run_cow_paid_position_census()
    assert census_payload["cow_mounted"] is True
    census_path = EVIDENCE_DIR / "paid_position_round1_census_20261116.json"
    census_path.write_text(json.dumps(census_payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"      Census complete: cow_mounted={census_payload['cow_mounted']}, "
        f"workspaces_screened={census_payload['workspaces_screened']}, "
        f"artifacts={census_payload['artifacts_with_position_state']}, "
        f"non_source_len2={census_payload['independent_two_owner_scenes_count']}",
        flush=True,
    )

    if NEW_COW_OUTPUT.exists():
        shutil.rmtree(NEW_COW_OUTPUT)
    NEW_COW_OUTPUT.mkdir(parents=True, exist_ok=True)

    print("[2/3] Restoring frozen historical v8 KnowledgeLibrary (read-only)...", flush=True)
    t_restore = time.monotonic()
    reg602 = load(COW_602_DIR / "registration.json")
    reg603 = load(COW_603_DIR / "registration.json")
    hist_rec602 = load(COW_602_DIR / "receipt.json")
    hist_rec603 = load(COW_603_DIR / "receipt.json")
    base_library = program_knowledge(
        Path(reg603["knowledge_root"]), reg603.get("knowledge_repair_ref")
    )
    restore_seconds = round(time.monotonic() - t_restore, 3)
    initial_comms = consolidate_world_structure_commitments(base_library.meta_priors)
    print(
        f"      Restored in {restore_seconds}s: meta_priors={len(base_library.meta_priors)}, "
        f"consolidated_commitments={len(initial_comms)}",
        flush=True,
    )

    # --- Benchmark 602 (seed 106110602, obs=320, compute=300,000, meas=6,144) ---
    print("[3/3] Running controlled comparison on 106110602 (602) and 106110603 (603)...", flush=True)
    task602 = reg602["tasks"][0]
    out602 = NEW_COW_OUTPUT / "runs" / str(task602["seed"])
    base602_sum, _ = _run_arm(
        "baseline_without_v10",
        task602,
        reg602["research_config"],
        reg602["limits"]["measurement_cost"],
        base_library,
        enable_v10=False,
        out_dir=out602,
    )
    v10_602_sum, lib_after_602 = _run_arm(
        "v10_from_v8_archive",
        task602,
        reg602["research_config"],
        reg602["limits"]["measurement_cost"],
        base_library,
        enable_v10=True,
        out_dir=out602,
    )
    term602 = _evaluate_arm_predictors(
        task602,
        reg602["terminal"],
        out602,
        ["baseline_without_v10", "v10_from_v8_archive"],
    )
    print(
        f"      [602] baseline={base602_sum['status']} "
        f"(pass={term602['arms']['baseline_without_v10'].get('prediction_pass')}) | "
        f"v10={v10_602_sum['status']} "
        f"(pass={term602['arms']['v10_from_v8_archive'].get('prediction_pass')}, "
        f"rand_mse={term602['arms']['v10_from_v8_archive'].get('random_mse')}, "
        f"step_mse={term602['arms']['v10_from_v8_archive'].get('public_control_step_mse')}, "
        f"obs={v10_602_sum['costs']['observation_cost']}, "
        f"compute={v10_602_sum['costs']['compute_cost']})",
        flush=True,
    )

    # Check consolidated commitments after 602!
    comms_after_602 = consolidate_world_structure_commitments(lib_after_602.meta_priors)

    # --- Benchmark 603 (seed 106110603, obs=256, compute=200,000, meas=6,144) ---
    task603 = reg603["tasks"][0]
    out603 = NEW_COW_OUTPUT / "runs" / str(task603["seed"])
    base603_sum, _ = _run_arm(
        "baseline_without_v10",
        task603,
        reg603["research_config"],
        reg603["limits"]["measurement_cost"],
        base_library,
        enable_v10=False,
        out_dir=out603,
    )
    v10_603_sum, _ = _run_arm(
        "v10_from_v8_archive",
        task603,
        reg603["research_config"],
        reg603["limits"]["measurement_cost"],
        base_library,
        enable_v10=True,
        out_dir=out603,
    )
    # Keep only supported directional priors + instrument/object knowledge for clean successor test,
    # AND test raw full lib_after_602 directly!
    v10_603_seq_sum, _ = _run_arm(
        "v10_sequential_after_602",
        task603,
        reg603["research_config"],
        reg603["limits"]["measurement_cost"],
        lib_after_602,
        enable_v10=True,
        out_dir=out603,
    )
    term603 = _evaluate_arm_predictors(
        task603,
        reg603["terminal"],
        out603,
        ["baseline_without_v10", "v10_from_v8_archive", "v10_sequential_after_602"],
    )
    print(
        f"      [603] baseline={base603_sum['status']} "
        f"(pass={term603['arms']['baseline_without_v10'].get('prediction_pass')}) | "
        f"v10_from_v8={v10_603_sum['status']} "
        f"(pass={term603['arms']['v10_from_v8_archive'].get('prediction_pass')}, "
        f"rand_mse={term603['arms']['v10_from_v8_archive'].get('random_mse')}, "
        f"step_mse={term603['arms']['v10_from_v8_archive'].get('public_control_step_mse')}, "
        f"obs={v10_603_sum['costs']['observation_cost']}, "
        f"compute={v10_603_sum['costs']['compute_cost']}) | "
        f"v10_seq_after_602={v10_603_seq_sum['status']} "
        f"(pass={term603['arms']['v10_sequential_after_602'].get('prediction_pass')}, "
        f"obs={v10_603_seq_sum['costs']['observation_cost']}, "
        f"compute={v10_603_seq_sum['costs']['compute_cost']})",
        flush=True,
    )

    guards_after = _snapshot_guards()
    assert guards_before == guards_after, "FATAL: closed historical receipt modified!"

    receipt_payload = {
        "generated_at_utc": _utc_now(),
        "study": "v10_cow_602_603_non_destructive_comparison",
        "model_calls": 0,
        "reported_tokens": 0,
        "cow_integrity_verification": {
            "closed_historical_files_checked": len(guards_before),
            "closed_historical_files_unchanged": guards_before == guards_after,
            "guards_snapshot": guards_after,
            "new_cow_directory": str(NEW_COW_OUTPUT),
        },
        "inherited_source_library": {
            "knowledge_root": reg603["knowledge_root"],
            "knowledge_repair_ref": reg603.get("knowledge_repair_ref"),
            "restore_wall_seconds": restore_seconds,
            "meta_priors_count": len(base_library.meta_priors),
            "consolidated_commitments": [
                {
                    "commitment_id": c["commitment_id"],
                    "source_concept_ids": c["source_concept_ids"],
                    "source_arities": c["source_arities"],
                    "radius_parameters": c["radius_parameters"],
                    "role_terms_count": len(c["role_terms"]),
                }
                for c in initial_comms
            ],
        },
        "benchmark_602_seed_106110602": {
            "registered_limits": reg602["limits"],
            "task": task602,
            "historical_closed_3arm_outcomes": hist_rec602["arms"],
            "new_controlled_arms": {
                "baseline_without_v10": {
                    **base602_sum,
                    "terminal_evaluation": term602["arms"]["baseline_without_v10"],
                },
                "v10_from_v8_archive": {
                    **v10_602_sum,
                    "terminal_evaluation": term602["arms"]["v10_from_v8_archive"],
                },
            },
            "shared_terminal_costs": term602["shared_actual_costs"],
            "commitments_after_602_count": len(comms_after_602),
            "commitments_after_602_top": {
                "commitment_id": comms_after_602[0]["commitment_id"],
                "source_arities": comms_after_602[0]["source_arities"],
                "generation": comms_after_602[0].get("generation", 0),
                "radius_parameters": comms_after_602[0]["radius_parameters"],
            }
            if comms_after_602
            else None,
        },
        "benchmark_603_seed_106110603": {
            "registered_limits": reg603["limits"],
            "task": task603,
            "historical_closed_3arm_outcomes": hist_rec603["outcomes"],
            "new_controlled_arms": {
                "baseline_without_v10": {
                    **base603_sum,
                    "terminal_evaluation": term603["arms"]["baseline_without_v10"],
                },
                "v10_from_v8_archive": {
                    **v10_603_sum,
                    "terminal_evaluation": term603["arms"]["v10_from_v8_archive"],
                },
                "v10_sequential_after_602": {
                    **v10_603_seq_sum,
                    "terminal_evaluation": term603["arms"]["v10_sequential_after_602"],
                },
            },
            "shared_terminal_costs": term603["shared_actual_costs"],
        },
    }

    write_json_checkpoint(NEW_COW_OUTPUT / "receipt.json", receipt_payload)
    cow_footprint_bytes = _dir_size_bytes(NEW_COW_OUTPUT)
    free_mb_after = _disk_free_mb(COW_ROOT)
    receipt_payload["cow_integrity_verification"]["new_cow_directory_bytes"] = cow_footprint_bytes
    receipt_payload["cow_integrity_verification"]["new_cow_directory_mb"] = round(
        cow_footprint_bytes / (1024.0 * 1024.0), 4
    )
    receipt_payload["cow_integrity_verification"]["cow_free_mb_before"] = round(free_mb_before, 2)
    receipt_payload["cow_integrity_verification"]["cow_free_mb_after"] = round(free_mb_after, 2)
    assert cow_footprint_bytes < 5 * 1024 * 1024, "COW footprint exceeded 5 MB budget!"
    write_json_checkpoint(NEW_COW_OUTPUT / "receipt.json", receipt_payload)

    local_receipt = EVIDENCE_DIR / "v10_cow_602_603_new_ledger_20261116.json"
    local_receipt.write_text(json.dumps(receipt_payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"Saved receipt to {local_receipt} and {NEW_COW_OUTPUT / 'receipt.json'} "
        f"(COW footprint = {cow_footprint_bytes} bytes, closed files unchanged = True).",
        flush=True,
    )


if __name__ == "__main__":
    main()
