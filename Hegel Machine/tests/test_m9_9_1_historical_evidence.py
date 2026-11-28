"""Regression checks for provenance of historical M9.9 audit projections."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def audit():
    path = Path(__file__).resolve().parents[1] / "scripts/run_m9_9_1_audit_closure_and_backverification.py"
    spec = importlib.util.spec_from_file_location("m99_audit_replay", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cancelled_provider_cost_keeps_unknowns_and_omits_response_text(audit, tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "OUT_DIR", tmp_path)
    stage = "m99_dev_20261127T175948Z__dev_alpha_m99_00__step04"
    source = tmp_path / "provider" / stage
    source.mkdir(parents=True)
    common = {"stage_subdir": stage, "raw_response_text": "MODEL TEXT FOR EXCLUSION"}
    paid = dict(common, call_uuid="paid", call_seq=1, attempt_count=1, status="completed", attempts=[
        {"attempt_index": 1, "usage_known": True, "prompt_tokens": 20, "cached_tokens": 13,
         "completion_tokens": 4, "response_sha256": "abc", "timed_out": False}])
    pending = dict(common, call_uuid="pending", call_seq=2, attempt_count=0, status="in_progress", attempts=[])
    (source / "attempts_001_paid.json").write_text(json.dumps(paid))
    (source / "attempts_002_pending.json").write_text(json.dumps(pending))
    (source / "resp_002_att0.json").touch()
    result = audit._cancelled_campaign_ledger(source.parent, set(), 100)
    assert result["known_physical_v2"] == 24
    assert result["total_rd_v2_formula"] == "124 + x + y"
    assert result["unclosed_calls"][0]["pending_response_bytes"] == 0
    with gzip.open(tmp_path / result["archive"], "rt") as sidecar:
        payload = sidecar.read()
    assert "MODEL TEXT FOR EXCLUSION" not in payload
    assert str(source) not in payload
    assert hashlib.sha256((source / "attempts_001_paid.json").read_bytes()).hexdigest() in payload
    assert audit._cancelled_campaign_ledger(None, set(), 100) == result
    with pytest.raises(ValueError, match="repeated"):
        audit._cancelled_campaign_ledger(None, {"paid"}, 100)


def test_bound_host_measurement_requires_receipt_and_both_artifact_identities(audit):
    assets = {"/holdout_task/public_batch.pkl": "public"}
    gate = {"candidate_bundle_sha256": "model", "public_setup_files_sha256": assets,
            "scientific_probe_receipt": None}
    assert not audit._bound_host_measurement(gate, "public")
    receipt = {"signed_by_host_probe_executor": True, "execution_status": "completed",
               "public_input_digest": "public", "candidate_bundle_sha256": "model",
               "public_setup_files_sha256": assets, "kernel_version": "v1",
               "measured_scientific_metrics": {"residual": 2.0}}
    gate["scientific_probe_receipt"] = receipt
    assert audit._bound_host_measurement(gate, "public")
    assert not audit._bound_host_measurement(gate, "different public")
    receipt["candidate_bundle_sha256"] = "another model"
    assert not audit._bound_host_measurement(gate, "public")


def test_original_withdrawal_foreign_keys_and_missing_negative_receipts(audit):
    summary = json.loads((audit.M99_CONF_DIR / "m9_9_continual_summary.json").read_bytes())
    rows = audit._historical_withdrawal_links(summary["streams"])
    assert len(rows) == 7 and all(row["matches"] for row in rows)
    assert not any(row["historical_signed_scientific_receipt"] for row in rows)



def test_historical_policy_hash_and_container_input_have_separate_provenance(audit):
    # This episode has no top-level setup receipt. The signed trial recorded the actual sandbox file.
    wid = "beta_m99_c02"
    root = audit.M99_CONF_DIR / "continual_streams" / wid
    episode = next(root.glob("Hegel*/*batch_07__*/00_episode_summary.json"))
    data = json.loads(episode.read_bytes())
    trial = data["pre_llm_trial_receipt"]["probe_result"]["scientific_probe_receipt"]
    expected = trial["public_setup_files_sha256"]["/holdout_task/public_batch.pkl"]
    assert audit._sealed_container_public_sha(wid, 7) == expected
    ablation = json.loads((audit.M99_CONF_DIR / "m9_9_factorized_ceiling_ablation.json").read_bytes())
    original = next(r for r in ablation["rows"] if r["world_id"] == wid and r["event"] == "E7_coupled_revisit"
                    and r["policy"] == "hegel_selective_cross_factor_graph")
    assert original["public_batch_pkl_sha256"] != expected
