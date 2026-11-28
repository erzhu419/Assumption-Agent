"""Offline finalization must not manufacture a paid run identity or elapsed time."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "docs/evidence/m9_9_dev_live_calibration_20261128"
SCRIPT = ROOT / "scripts/run_m9_9_autonomous_decomposition_experiment.py"


@pytest.mark.parametrize(("recorded", "recover_raw"), [(False, False), (True, False), (False, True)])
def test_finalize_cached_dev_bundle(tmp_path, recorded, recover_raw):
    for file in [
        DEV / "all_provider_call_receipts.json", DEV / "m9_9_factorized_ceiling_ablation.json",
        DEV / "m9_9_continual_summary.json", *DEV.glob("continual_streams/*/*/00_stream_summary.json"),
    ]:
        target = tmp_path / file.relative_to(DEV)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, target)
    receipts = json.loads((tmp_path / "all_provider_call_receipts.json").read_text())
    run_id, = {row["stage_subdir"].split("__")[0] for row in receipts}
    if recover_raw:
        (tmp_path / "all_provider_call_receipts.json").unlink()
        raw = tmp_path / "provider_raw_attempts_mirror"
        raw.mkdir()
        for mirror in (DEV / raw.name).glob("*.json"):
            shutil.copyfile(mirror, raw / mirror.name)
    observed = json.loads((DEV / "m9_9_continual_summary.json").read_text())
    manifest = tmp_path / "execution_manifest.json"
    if recorded:
        manifest.write_text(json.dumps({"execution_run_id": run_id, "model": "recorded-model"}))
        (tmp_path / "execution_completion.json").write_text(json.dumps({
            "execution_run_id": run_id, "execution_elapsed_wall_sec": 1234.5}))
    result = subprocess.run([sys.executable, str(SCRIPT), "--split", "dev", "--finalize-existing",
        "--out-dir", str(tmp_path), "--v7-bundle", str(tmp_path / "absent.json")],
        cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    d = json.loads((tmp_path / "m9_9_continual_summary.json").read_text())
    if recover_raw:
        rebuilt = json.loads((tmp_path / "all_provider_call_receipts.json").read_text())
        assert all("utc_start_epoch_sec" in row and "monotonic_start_sec" not in row for row in rebuilt)
        expected_roles = {r["call_uuid"]: (r["call_role"], r["episode_short_name"]) for r in receipts}
        assert {r["call_uuid"]: (r["call_role"], r["episode_short_name"]) for r in rebuilt} == expected_roles
        for arm, ledger in observed["arm_cost_ledgers"].items():
            assert d["arm_cost_ledgers"][arm]["scope1_in_stream_total"]["proxy_v2_u_010c_4o"] == ledger["scope1_in_stream_total"]["proxy_v2_u_010c_4o"]
    else:
        assert d["arm_cost_ledgers"] == observed["arm_cost_ledgers"]
    assert d["run_id"] == run_id and d["wall_sec"] == (1234.5 if recorded else None)
    assert d["model"] == ("recorded-model" if recorded else observed["model"])
    assert d["execution_provenance"]["source_tree_sha256_at_start"] is None
    assert d["finalizer_provenance"]["finalization_run_id"] != run_id
    assert d["finalizer_provenance"]["reused_ablation"]
    assert d["ablation_provenance"]["code_commit"] is None
    billed = {(r["arm"], r["episode_short_name"]) for r in receipts}
    for arm, totals in d["arm_outcome_totals"].items():
        assert totals["strict_passes"] == observed["arm_outcome_totals"][arm]["strict_passes"]
        assert totals["zero_turn_passed_closures"] == sum(
            ep["strict_pass"] and (arm, ep["short_name"]) not in billed
            for stream in d["streams"] if stream["arm"] == arm for ep in stream["episodes_meta"])
    assert manifest.exists() == recorded
    if recorded:
        assert json.loads(manifest.read_text()) == {"execution_run_id": run_id, "model": "recorded-model"}
        (tmp_path / "ablation_manifest.json").write_text(json.dumps({
            "source_run_id": run_id, "result_sha256": "0" * 64}))
        bad = subprocess.run(result.args, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                             capture_output=True, text=True)
        assert bad.returncode != 0 and "no longer matches its manifest" in bad.stderr
