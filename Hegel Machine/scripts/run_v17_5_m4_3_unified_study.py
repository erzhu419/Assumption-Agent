"""Execute Preregistered Hegel V17.5 (M4.3) Study: Full `Hegel_M4_2_Audit` Counterfactual Closure,
Channel-Bound Multi-Scale Execution, Confirmed-Instance Identity, Relocatable Forest Snapshots,
and 11-Arm Continual Lifecycle Benchmark (`96` unseen in-scope + `8` OOS runs/arm on fresh seeds).

Writes full static evidence bundle by default to:
  `docs/evidence/v17_5_m4_3_channel_bound_lifecycle_20261120/`
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
    run_m4_2_audit_closure_verification_suite,
    run_m4_3_channel_bound_continual_study,
)
from hegel_machine.cognition.wormhole_forest import PhilosophicalForest


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVID_DIR = REPO_ROOT / "docs" / "evidence" / "v17_5_m4_3_channel_bound_lifecycle_20261120"

M4_3_FRESH_SEEDS = (20_301_120, 20_311_120, 20_321_120, 20_331_120)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relocate_paths_in_obj(obj: Any, tmp_lifecycle_dir: Path) -> Any:
    """Rewrite transient `/tmp/.../continual_lifecycle/...` paths into portable relative bundle paths."""
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
        return [_relocate_paths_in_obj(x, tmp_lifecycle_dir) for x in obj]
    if isinstance(obj, dict):
        return {k: _relocate_paths_in_obj(v, tmp_lifecycle_dir) for k, v in obj.items()}
    return obj


def serialize_relocatable_forest(forest: PhilosophicalForest, tmp_lifecycle_dir: Path) -> dict[str, Any]:
    raw = {
        "nodes": deepcopy(forest.nodes),
        "node_history": deepcopy(forest.node_history),
        "local_implication_edges": deepcopy(forest.local_implication_edges),
        "coarse_graining_edges": deepcopy(forest.coarse_graining_edges),
        "horizontal_wormhole_edges": deepcopy(forest.horizontal_wormhole_edges),
        "wormhole_history": deepcopy(forest.wormhole_history),
        "conflict_relations": deepcopy(forest.conflict_relations),
    }
    return _relocate_paths_in_obj(raw, tmp_lifecycle_dir)


def load_relocated_forest_from_bundle(snapshot_path: Path, bundle_root: Path) -> PhilosophicalForest:
    """Reload a persisted ``PhilosophicalForest`` JSON snapshot and resolve relative artifact paths."""
    data = json.loads(snapshot_path.read_text(encoding="utf-8"))

    def _resolve_rel(obj: Any) -> Any:
        if isinstance(obj, str) and obj.startswith("persisted_authored_workspaces/"):
            return str((bundle_root / obj).resolve())
        if isinstance(obj, list):
            return [_resolve_rel(x) for x in obj]
        if isinstance(obj, dict):
            return {k: _resolve_rel(v) for k, v in obj.items()}
        return obj

    resolved = _resolve_rel(data)
    f = PhilosophicalForest()
    f.nodes = resolved["nodes"]
    f.node_history = resolved["node_history"]
    f.local_implication_edges = resolved["local_implication_edges"]
    f.coarse_graining_edges = resolved["coarse_graining_edges"]
    f.horizontal_wormhole_edges = resolved["horizontal_wormhole_edges"]
    f.wormhole_history = resolved["wormhole_history"]
    f.conflict_relations = resolved["conflict_relations"]
    return f


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Hegel V17.5 (M4.3) Channel-Bound Continual Lifecycle Study")
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
    with TemporaryDirectory(prefix="hegel-v17-5-run-") as td:
        tmp_work = Path(td)
        part1_audit = run_m4_2_audit_closure_verification_suite(tmp_work / "part1_audit_closure")
        part2_study = run_m4_3_channel_bound_continual_study(
            tmp_work / "part2_m43_study",
            seeds=M4_3_FRESH_SEEDS,
            onset_pilot_noise_std=0.0020,
        )

        lifecycle_dir = tmp_work / "part2_m43_study" / "continual_lifecycle"
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
            shutil.copytree(s2_reval_src, persisted_dst / "stage2_lifecycle" / "stage2_pi_s_reverified", ignore=ignore_pyc)

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
            rel_json = serialize_relocatable_forest(snaps[snap_key], lifecycle_dir)
            (snapshots_dir / fname).write_text(json.dumps(rel_json, indent=2, sort_keys=True), encoding="utf-8")

        reval_cert_rel = _relocate_paths_in_obj(snaps["stage2_pi_s_reverified_certificate"], lifecycle_dir)
        (snapshots_dir / "stage2_pi_s_reverified_certificate.json").write_text(
            json.dumps(reval_cert_rel, indent=2, sort_keys=True), encoding="utf-8"
        )

        # Relocate paths inside `part2_study` summary before saving!
        part2_clean = _relocate_paths_in_obj(part2_study, lifecycle_dir)

    # Verify cold reload of `post_stage2_forest_snapshot.json` AFTER TemporaryDirectory deletion!
    reloaded_post_s2 = load_relocated_forest_from_bundle(
        out_dir / "relocatable_forest_snapshots" / "post_stage2_forest_snapshot.json",
        out_dir,
    )
    relocated_r3_verify = run_interleaved_zero_fit_dispatcher_stream(reloaded_post_s2)
    assert relocated_r3_verify["all_routes_exact"] is True
    assert relocated_r3_verify["any_fit_called"] is False
    assert relocated_r3_verify["unauthorized_cleanly_blocked"] is True

    receipts = part2_clean.pop("execution_receipts")
    part2_clean["relocated_post_stage2_cold_reload_r3_verified"] = {
        "all_routes_exact": relocated_r3_verify["all_routes_exact"],
        "any_fit_called": relocated_r3_verify["any_fit_called"],
        "unauthorized_cleanly_blocked": relocated_r3_verify["unauthorized_cleanly_blocked"],
    }

    (out_dir / "part1_m4_2_audit_closure_verification.json").write_text(
        json.dumps(part1_audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out_dir / "part2_m4_3_channel_bound_continual_summary.json").write_text(
        json.dumps(part2_clean, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out_dir / "part2_m4_3_execution_receipts.jsonl").write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in receipts) + "\n", encoding="utf-8"
    )

    # Remove any stray `__pycache__` inside `out_dir` just in case before computing SHA256SUMS.txt
    for pyc_dir in list(out_dir.rglob("__pycache__")):
        shutil.rmtree(pyc_dir, ignore_errors=True)

    elapsed = round(time.perf_counter() - t0, 2)
    arms = part2_clean["stage3_arms"]
    manifest = {
        "version": "V17.5_M4.3",
        "elapsed_wall_seconds": elapsed,
        "fresh_test_seeds": list(M4_3_FRESH_SEEDS),
        "onset_pilot_noise_std": part2_clean["onset_pilot_noise_std"],
        "relocated_snapshot_cold_r3_exact": relocated_r3_verify["all_routes_exact"],
        "max_prefix_confirm_overlap_across_all_arms": max(
            st["max_prefix_confirm_overlap"] for st in arms.values()
        ),
        "hegel_mean_composite_cost": arms["Hegel_Unified_Continual_Forest"]["mean_composite_cost"],
        "no_maintenance_mean_composite_cost": arms["Hegel_NoMaintenance_SameHistory"]["mean_composite_cost"],
        "compiled_cuetable_mean_composite_cost": arms["Compiled_Relational_CueTable_SameForest"]["mean_composite_cost"],
        "stratified_dev_mean_composite_cost": arms["Ordinary_AllHistory_StratifiedDevOrder"]["mean_composite_cost"],
        "best_pilot_rmse_mean_composite_cost": arms["Ordinary_AllHistory_BestPilotRMSE"]["mean_composite_cost"],
        "frozen_open_rediscovery_mean_composite_cost": arms["Frozen_After_Stage0_OpenRediscovery"]["mean_composite_cost"],
        "frozen_no_discovery_strict_pass_rate": arms["Frozen_After_Stage0_NoDiscovery"]["strict_pass_rate"],
        "observed_breakeven_vs_true_frozen_open_rediscovery": (
            part2_clean["lifecycle_breakeven"]["observed_breakeven_target_vs_true_frozen_open_rediscovery"]
        ),
        "observed_breakeven_vs_no_maintenance": (
            part2_clean["lifecycle_breakeven"]["observed_breakeven_target_vs_no_maintenance"]
        ),
    }
    (out_dir / "study_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    sha_lines: list[str] = []
    for fpath in sorted(p for p in out_dir.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt"):
        rel = fpath.relative_to(out_dir).as_posix()
        sha_lines.append(f"{_sha256_file(fpath)}  {rel}")
    (out_dir / "SHA256SUMS.txt").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
