"""Execute Preregistered Hegel V17.6 (M4.4) Study: Full `Hegel_M4_3_Audit` Invariant Closure,
Self-Verifying Relocatable Forest Snapshots, 4-Seed 11-Arm Continual Benchmark + All-Seed Gaussian
Fork Receipts, and Stateful Online Write-Back & Multi-Cue Continual Stream (`M5` Bridge).

Writes full static evidence bundle by default to:
  `docs/evidence/v17_6_m4_4_instance_role_closure_and_online_writeback_20261120/`
and supports `--out-dir <path>` for clean read-only external audit replays without touching
any committed `docs/evidence/*` or `markdown/*Audit*/` directories!
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from typing import Any

from hegel_machine.cognition.m4_2_unified_closure_and_pi_lifecycle import (
    run_interleaved_zero_fit_dispatcher_stream,
)
from hegel_machine.cognition.m4_3_channel_bound_lifecycle_and_audit_closure import (
    build_m4_3_continual_forest,
    run_m4_3_channel_bound_continual_study,
)
from hegel_machine.cognition.m4_4_instance_role_closure_and_online_writeback import (
    load_location_independent_forest,
    run_m4_3_audit_closure_verification_suite,
    run_m4_4_online_writeback_and_multicue_stream,
    serialize_location_independent_forest,
    verify_forest_snapshot_digests,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVID_DIR = REPO_ROOT / "docs" / "evidence" / "v17_6_m4_4_instance_role_closure_and_online_writeback_20261120"

M4_4_SEEDS = (20_301_120, 20_311_120, 20_321_120, 20_331_120)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relocate_transient_summary_paths(obj: Any, tmp_lifecycle_dir: Path) -> Any:
    """Clean any transient `/tmp/...` diagnostic workspace paths in summary dicts (never in forest contracts!)."""
    prefix = tmp_lifecycle_dir.resolve().as_posix()
    if isinstance(obj, str):
        norm = obj.replace("\\", "/")
        if norm.startswith(prefix + "/"):
            sub = norm[len(prefix) + 1 :]
            for folder in ("stage0_metered_sources", "persisted_stage1_authored_workspaces", "stage2_lifecycle"):
                if sub.startswith(folder + "/"):
                    return f"persisted_authored_workspaces/{sub}"
        return obj
    if isinstance(obj, list):
        return [_relocate_transient_summary_paths(x, tmp_lifecycle_dir) for x in obj]
    if isinstance(obj, dict):
        return {k: _relocate_transient_summary_paths(v, tmp_lifecycle_dir) for k, v in obj.items()}
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Hegel V17.6 (M4.4) Audit Closure + Continual Benchmark + Online Write-Back Study"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_EVID_DIR,
        help="Output directory for evidence bundle (use /tmp/... for read-only audit replay)",
    )
    args = parser.parse_args()
    out_dir: Path = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    with TemporaryDirectory(prefix="hegel-v17-6-run-") as td:
        tmp_work = Path(td)

        # Part 1: Full `Hegel_M4_3_Audit` Counterfactual Closure Suite
        part1_audit = run_m4_3_audit_closure_verification_suite(tmp_work / "part1_m43_audit_closure")

        # Part 2: Full 4-Seed 11-Arm Continual Lifecycle Benchmark + All-Seed Gaussian Fork Receipts
        part2_study = run_m4_3_channel_bound_continual_study(
            tmp_work / "part2_m44_benchmark",
            seeds=M4_4_SEEDS,
            onset_pilot_noise_std=0.0020,
        )

        # Part 3: Stateful Online Write-Back, Equivalence Pooling & Multi-Cue Continual Stream (`M5` Bridge)
        built_for_part3 = build_m4_3_continual_forest(tmp_work / "part3_base_lifecycle")
        part3_online = run_m4_4_online_writeback_and_multicue_stream(
            tmp_work / "part3_online_stream",
            built_for_part3,
        )

        lifecycle_dir = tmp_work / "part2_m44_benchmark" / "continual_lifecycle"
        persisted_dst = out_dir / "persisted_authored_workspaces"
        if persisted_dst.exists():
            shutil.rmtree(persisted_dst)
        persisted_dst.mkdir(parents=True, exist_ok=True)

        ignore_pyc = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")
        for sub_name in ("stage0_metered_sources", "persisted_stage1_authored_workspaces"):
            sub_src = lifecycle_dir / sub_name
            if sub_src.exists():
                shutil.copytree(sub_src, persisted_dst / sub_name, ignore=ignore_pyc)
        s2_reval_src = lifecycle_dir / "stage2_lifecycle" / "stage2_pi_s_reverified"
        if s2_reval_src.exists():
            (persisted_dst / "stage2_lifecycle").mkdir(parents=True, exist_ok=True)
            shutil.copytree(
                s2_reval_src,
                persisted_dst / "stage2_lifecycle" / "stage2_pi_s_reverified",
                ignore=ignore_pyc,
            )

        snaps = part2_study.pop("forest_snapshots")
        snapshots_dir = out_dir / "relocatable_forest_snapshots"
        if snapshots_dir.exists():
            shutil.rmtree(snapshots_dir)
        snapshots_dir.mkdir(parents=True, exist_ok=True)

        for snap_key, fname in (
            ("true_stage0_forest", "true_stage0_forest_snapshot.json"),
            ("unmaintained_post_stage1_forest", "unmaintained_post_stage1_forest_snapshot.json"),
            ("maintained_post_stage2_forest", "post_stage2_forest_snapshot.json"),
        ):
            # ZERO path rewriting on serialized contracts/certificates (`semantic_digest` stays 100% self-verifying!)
            snap_json = serialize_location_independent_forest(snaps[snap_key])
            (snapshots_dir / fname).write_text(json.dumps(snap_json, indent=2, sort_keys=True), encoding="utf-8")

        reval_cert_json = deepcopy(snaps["stage2_pi_s_reverified_certificate"])
        (snapshots_dir / "stage2_pi_s_reverified_certificate.json").write_text(
            json.dumps(reval_cert_json, indent=2, sort_keys=True), encoding="utf-8"
        )

        part2_receipts = part2_study.pop("execution_receipts")
        part2_fork_receipts = part2_study.pop("fork_execution_receipts")
        part2_clean = _relocate_transient_summary_paths(part2_study, lifecycle_dir)
        part3_receipts = part3_online.pop("stream_receipts")

    # Verify BOTH raw JSON digest self-consistency AND cold reload AFTER TemporaryDirectory deletion!
    relocated_digest_checks: dict[str, Any] = {}
    for fname in (
        "true_stage0_forest_snapshot.json",
        "unmaintained_post_stage1_forest_snapshot.json",
        "post_stage2_forest_snapshot.json",
    ):
        sp_p = out_dir / "relocatable_forest_snapshots" / fname
        disk_chk = verify_forest_snapshot_digests(json.loads(sp_p.read_text(encoding="utf-8")))
        mem_chk = verify_forest_snapshot_digests(load_location_independent_forest(sp_p, out_dir))
        assert disk_chk["semantic_digest_mismatch_count"] == 0, f"Disk digest mismatch in {fname}: {disk_chk}"
        assert disk_chk["certificate_digest_mismatch_count"] == 0, f"Cert digest mismatch in {fname}: {disk_chk}"
        assert len(disk_chk["missing_expected_instance_nodes"]) == 0, f"Missing instance digest in {fname}: {disk_chk}"
        assert mem_chk["semantic_digest_mismatch_count"] == 0
        relocated_digest_checks[fname] = {"disk_json": disk_chk, "reloaded_memory": mem_chk}

    reloaded_post_s2 = load_location_independent_forest(
        out_dir / "relocatable_forest_snapshots" / "post_stage2_forest_snapshot.json",
        out_dir,
    )
    relocated_r3_verify = run_interleaved_zero_fit_dispatcher_stream(reloaded_post_s2)
    assert relocated_r3_verify["all_routes_exact"] is True
    assert relocated_r3_verify["any_fit_called"] is False
    assert relocated_r3_verify["unauthorized_cleanly_blocked"] is True

    part2_clean["relocated_post_stage2_cold_reload_and_digest_verified"] = {
        "all_routes_exact": relocated_r3_verify["all_routes_exact"],
        "any_fit_called": relocated_r3_verify["any_fit_called"],
        "unauthorized_cleanly_blocked": relocated_r3_verify["unauthorized_cleanly_blocked"],
        "snapshot_digest_verification": relocated_digest_checks,
    }

    (out_dir / "part1_m4_3_audit_closure_verification.json").write_text(
        json.dumps(part1_audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out_dir / "part2_m4_4_continual_benchmark_summary.json").write_text(
        json.dumps(part2_clean, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out_dir / "part2_m4_4_execution_receipts.jsonl").write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in part2_receipts) + "\n", encoding="utf-8"
    )
    (out_dir / "part2_m4_4_fork_noise_receipts.jsonl").write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in part2_fork_receipts) + "\n", encoding="utf-8"
    )
    (out_dir / "part3_m4_4_online_writeback_stream_summary.json").write_text(
        json.dumps(part3_online, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out_dir / "part3_m4_4_online_writeback_receipts.jsonl").write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in part3_receipts) + "\n", encoding="utf-8"
    )

    for pyc_dir in list(out_dir.rglob("__pycache__")):
        shutil.rmtree(pyc_dir, ignore_errors=True)

    elapsed = round(time.perf_counter() - t0, 2)
    arms = part2_clean["stage3_arms"]
    manifest = {
        "version": "V17.6_M4.4",
        "elapsed_wall_seconds": elapsed,
        "test_seeds": list(M4_4_SEEDS),
        "onset_pilot_noise_std": part2_clean["onset_pilot_noise_std"],
        "relocated_snapshot_cold_r3_exact": relocated_r3_verify["all_routes_exact"],
        "relocated_snapshot_semantic_digest_mismatches": sum(
            v["disk_json"]["semantic_digest_mismatch_count"] for v in relocated_digest_checks.values()
        ),
        "max_prefix_confirm_overlap_across_all_arms": max(
            st["max_prefix_confirm_overlap"] for st in arms.values()
        ),
        "hegel_stage3_mean_composite": arms["Hegel_Unified_Continual_Forest"]["mean_composite_cost"],
        "compiled_cuetable_stage3_mean_composite": arms["Compiled_Relational_CueTable_SameForest"]["mean_composite_cost"],
        "no_maintenance_stage3_mean_composite": arms["Hegel_NoMaintenance_SameHistory"]["mean_composite_cost"],
        "best_pilot_rmse_stage3_mean_composite": arms["Ordinary_AllHistory_BestPilotRMSE"]["mean_composite_cost"],
        "frozen_stage0_open_rediscovery_mean_composite": arms["Frozen_After_Stage0_OpenRediscovery"]["mean_composite_cost"],
        "lifecycle_breakeven": part2_clean["lifecycle_breakeven"],
        "part3_online_writeback_summary": {
            "net_savings_vs_frozen_post_stage2": part3_online["net_online_writeback_savings_vs_frozen_post_stage2"],
            "net_ppr_pool_on_savings_vs_compiled_cuetable": part3_online["net_ppr_pool_on_savings_vs_compiled_cuetable"],
            "net_ppr_pool_on_savings_vs_pool_off": part3_online["net_ppr_pool_on_savings_vs_pool_off"],
        },
    }
    (out_dir / "study_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    sha_lines: list[str] = []
    for fpath in sorted(p for p in out_dir.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt"):
        rel = fpath.relative_to(out_dir).as_posix()
        sha_lines.append(f"{_sha256_file(fpath)}  {rel}")
    (out_dir / "SHA256SUMS.txt").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
