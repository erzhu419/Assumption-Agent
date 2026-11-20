"""Execute Preregistered Hegel V17.7 (`M4.5` Audit Closure & `M5.0` Continual Lifecycle) Study:

1. **Part 1**: Full `Hegel_M44_Audit` 5-package counterfactual closure verification;
2. **Part 2**: Full 4-seed 11-arm continual lifecycle benchmark (`1,144` receipts) + all-seed Gaussian fork (`192` receipts)
   under unified `selected_action` & relational certificate guards;
3. **Part 3**: 24-step `M5.0` outcome-driven continual sublation stream (`5` arms = `120` receipts), full persistence of
   `online_writebacks/` (`W6_A/B`, `W7_A/B`), `post_m5_online_stream_forest_snapshot.json`, & `continual_session_checkpoint.json`,
   followed by **cross-process cold-reload continuation** (`S25..S27` in an isolated fresh Python subprocess AFTER
   deleting `TemporaryDirectory`)!

Writes static evidence bundle by default to:
  `docs/evidence/v17_7_m5_0_closure_and_continual_sublation_20261120/`
and supports `--out-dir <path>` for clean read-only external audit replays without touching any committed evidence or audit directory!
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
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
    serialize_location_independent_forest,
    verify_forest_snapshot_digests,
)
from hegel_machine.cognition.m5_0_closure_and_continual_sublation import (
    persist_m5_0_online_state_to_bundle,
    run_m4_4_audit_closure_verification_suite,
    run_m5_0_counterexample_sublation_and_persistent_stream,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVID_DIR = REPO_ROOT / "docs" / "evidence" / "v17_7_m5_0_closure_and_continual_sublation_20261120"

M5_0_BENCHMARK_SEEDS = (20_301_120, 20_311_120, 20_321_120, 20_331_120)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relocate_transient_summary_paths(obj: Any, tmp_lifecycle_dir: Path) -> Any:
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
        description="Run Hegel V17.7 (M4.5 Closure + M5.0 Continual Sublation & Cold Continuation) Study"
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
    with TemporaryDirectory(prefix="hegel-v17-7-run-") as td:
        tmp_work = Path(td)

        # Part 1: Full `Hegel_M44_Audit` 5-Package Closure Verification
        part1_audit = run_m4_4_audit_closure_verification_suite(tmp_work / "part1_m44_audit_closure")

        # Part 2: Full 4-Seed 11-Arm Continual Lifecycle Benchmark + All-Seed Gaussian Fork Receipts
        part2_study = run_m4_3_channel_bound_continual_study(
            tmp_work / "part2_continual_benchmark",
            seeds=M5_0_BENCHMARK_SEEDS,
            onset_pilot_noise_std=0.0020,
        )

        # Part 3: 24-Step `M5.0` Outcome-Driven Continual Sublation Stream (`5` Arms) + Persistent Online Bundle
        built_for_part3 = build_m4_3_continual_forest(tmp_work / "part3_base_lifecycle")
        part3_m5 = run_m5_0_counterexample_sublation_and_persistent_stream(
            tmp_work / "part3_m5_stream",
            built_for_part3,
        )

        lifecycle_dir = tmp_work / "part2_continual_benchmark" / "continual_lifecycle"
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
            snap_json = serialize_location_independent_forest(snaps[snap_key])
            (snapshots_dir / fname).write_text(json.dumps(snap_json, indent=2, sort_keys=True), encoding="utf-8")

        reval_cert_json = deepcopy(snaps["stage2_pi_s_reverified_certificate"])
        (snapshots_dir / "stage2_pi_s_reverified_certificate.json").write_text(
            json.dumps(reval_cert_json, indent=2, sort_keys=True), encoding="utf-8"
        )

        # Persist ALL online learned `W6_A/B`, `W7_A/B` bundles + `post_m5_online_stream_forest_snapshot.json` + checkpoint!
        primary_state = part3_m5.pop("primary_persisted_state")
        persisted_online_meta = persist_m5_0_online_state_to_bundle(primary_state, out_bundle_dir=out_dir)

        part2_receipts = part2_study.pop("execution_receipts")
        part2_fork_receipts = part2_study.pop("fork_execution_receipts")
        part2_clean = _relocate_transient_summary_paths(part2_study, lifecycle_dir)
        part3_receipts = part3_m5.pop("stream_receipts")

    # =========================================================================
    # TemporaryDirectory IS NOW COMPLETELY DESTROYED!
    # Verify ALL 4 snapshots + spawn a SEPARATE PYTHON SUBPROCESS to cold-resume `S25..S27`!
    # =========================================================================
    relocated_digest_checks: dict[str, Any] = {}
    for fname in (
        "true_stage0_forest_snapshot.json",
        "unmaintained_post_stage1_forest_snapshot.json",
        "post_stage2_forest_snapshot.json",
        "post_m5_online_stream_forest_snapshot.json",
    ):
        sp_p = out_dir / "relocatable_forest_snapshots" / fname
        disk_chk = verify_forest_snapshot_digests(json.loads(sp_p.read_text(encoding="utf-8")))
        mem_chk = verify_forest_snapshot_digests(load_location_independent_forest(sp_p, out_dir))
        assert disk_chk["semantic_digest_mismatch_count"] == 0, f"Disk digest mismatch in {fname}: {disk_chk}"
        assert disk_chk["history_semantic_digest_mismatch_count"] == 0, f"History mismatch in {fname}: {disk_chk}"
        assert disk_chk["certificate_digest_mismatch_count"] == 0, f"Cert digest mismatch in {fname}: {disk_chk}"
        assert disk_chk["edge_certificate_instance_mismatch_count"] == 0, f"Edge-cert mismatch in {fname}: {disk_chk}"
        assert len(disk_chk["missing_expected_instance_nodes"]) == 0, f"Missing instance digest in {fname}: {disk_chk}"
        assert mem_chk["semantic_digest_mismatch_count"] == 0
        relocated_digest_checks[fname] = {"disk_json": disk_chk, "reloaded_memory": mem_chk}

    reloaded_post_s2 = load_location_independent_forest(
        out_dir / "relocatable_forest_snapshots" / "post_stage2_forest_snapshot.json",
        out_dir,
    )
    relocated_r3_verify = run_interleaved_zero_fit_dispatcher_stream(reloaded_post_s2)
    assert relocated_r3_verify["all_routes_exact"] is True
    assert relocated_r3_verify["authorized_pass_rate"] == 1.0
    assert relocated_r3_verify["any_fit_called"] is False
    assert relocated_r3_verify["unauthorized_cleanly_blocked"] is True

    # Spawn a brand-new OS subprocess to verify true cross-process cold recovery & `S25..S27` continuation!
    with TemporaryDirectory(prefix="hegel-v17-7-cold-subproc-") as cold_td:
        subproc_out_p = Path(cold_td) / "cold_resume_result.json"
        subproc_code = (
            "import json, sys; from pathlib import Path; "
            "from hegel_machine.cognition.m5_0_closure_and_continual_sublation import "
            "resume_m5_0_continual_checkpoint_cold_continuation; "
            "res = resume_m5_0_continual_checkpoint_cold_continuation("
            "Path(sys.argv[1]), continuation_work_dir=Path(sys.argv[2])); "
            "Path(sys.argv[3]).write_text(json.dumps(res, indent=2, sort_keys=True), encoding='utf-8')"
        )
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONPATH"] = str((REPO_ROOT / "src").resolve())
        proc = subprocess.run(
            [sys.executable, "-c", subproc_code, str(out_dir), str(Path(cold_td) / "work"), str(subproc_out_p)],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert proc.returncode == 0
        cold_resume_report = json.loads(subproc_out_p.read_text(encoding="utf-8"))

    cold_cont_receipts = cold_resume_report.pop("continuation_receipts")
    assert cold_resume_report["all_continuation_strict_pass"] is True
    assert cold_resume_report["all_continuation_confirmed_via_recall"] is True
    assert cold_resume_report["any_source_rediscovery_triggered"] is False
    assert cold_resume_report["restored_cold_comprehensive_r3"]["distinct_domain_instances_verified"] is True

    part2_clean["relocated_post_stage2_cold_reload_and_digest_verified"] = {
        "all_routes_exact": relocated_r3_verify["all_routes_exact"],
        "authorized_pass_rate": relocated_r3_verify["authorized_pass_rate"],
        "max_authorized_abs_error": relocated_r3_verify["max_authorized_abs_error"],
        "any_fit_called": relocated_r3_verify["any_fit_called"],
        "unauthorized_cleanly_blocked": relocated_r3_verify["unauthorized_cleanly_blocked"],
        "snapshot_digest_verification": relocated_digest_checks,
    }
    part3_m5["persisted_online_artifacts"] = persisted_online_meta
    part3_m5["cross_process_cold_continuation_S29_to_S31"] = cold_resume_report

    (out_dir / "part1_m4_4_audit_closure_verification.json").write_text(
        json.dumps(part1_audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out_dir / "part2_m4_5_continual_benchmark_summary.json").write_text(
        json.dumps(part2_clean, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out_dir / "part2_m4_5_execution_receipts.jsonl").write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in part2_receipts) + "\n", encoding="utf-8"
    )
    (out_dir / "part2_m4_5_fork_noise_receipts.jsonl").write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in part2_fork_receipts) + "\n", encoding="utf-8"
    )
    (out_dir / "part3_m5_0_continual_sublation_summary.json").write_text(
        json.dumps(part3_m5, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out_dir / "part3_m5_0_stream_receipts.jsonl").write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in part3_receipts) + "\n", encoding="utf-8"
    )
    (out_dir / "part3_m5_0_cold_continuation_receipts.jsonl").write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in cold_cont_receipts) + "\n", encoding="utf-8"
    )

    for pyc_dir in list(out_dir.rglob("__pycache__")):
        shutil.rmtree(pyc_dir, ignore_errors=True)

    elapsed = round(time.perf_counter() - t0, 2)
    arms_p2 = part2_clean["stage3_arms"]
    manifest = {
        "version": "V17.7_M4.5_M5.0",
        "elapsed_wall_seconds": elapsed,
        "test_seeds": list(M5_0_BENCHMARK_SEEDS),
        "all_4_snapshots_semantic_and_relational_mismatches": sum(
            v["disk_json"]["semantic_digest_mismatch_count"]
            + v["disk_json"]["history_semantic_digest_mismatch_count"]
            + v["disk_json"]["certificate_digest_mismatch_count"]
            + v["disk_json"]["edge_certificate_instance_mismatch_count"]
            for v in relocated_digest_checks.values()
        ),
        "online_persisted_writeback_bundles": persisted_online_meta["online_writeback_node_dirs"],
        "cross_process_cold_resume_S29_S31_verified": cold_resume_report["all_continuation_strict_pass"],
        "hegel_part2_stage3_mean_composite": arms_p2["Hegel_Unified_Continual_Forest"]["mean_composite_cost"],
        "compiled_part2_stage3_mean_composite": arms_p2["Compiled_Relational_CueTable_SameForest"]["mean_composite_cost"],
        "no_maintenance_part2_stage3_mean_composite": arms_p2["Hegel_NoMaintenance_SameHistory"]["mean_composite_cost"],
        "part2_lifecycle_breakeven": part2_clean["lifecycle_breakeven"],
        "part3_m5_0_end_to_end_deltas": part3_m5["end_to_end_cost_deltas"],
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
