from __future__ import annotations

from pathlib import Path

import pytest

from assumption_agent.benchmarks.preflight import build_preflight
from assumption_agent.benchmarks.prewarm import (
    DEVELOPMENT_PREWARM_VERSION,
    _selected_item_set_hash,
    validate_development_prewarm_receipt,
)
from assumption_agent.benchmarks.skilllearnbench import SkillLearnBenchAdapter
from assumption_agent.models import stable_hash
from assumption_agent.splits import SplitManifest


ROOT = Path(__file__).resolve().parents[1]
BENCH_ROOT = (
    ROOT
    / "reference"
    / "self_evo_continual_20260707"
    / "repos"
    / "SkillLearnBench"
)


def test_required_env_preflight_is_bound_to_selected_manifest_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GH_TOKEN", raising=False)
    adapter = SkillLearnBenchAdapter(BENCH_ROOT)
    full_ids = [item.id for item in adapter.discover()]
    eligible_ids = [item.id for item in adapter.credential_independent_items()]

    full = build_preflight(
        BENCH_ROOT,
        trial_provider_mode="codex_subscription",
        item_ids=full_ids,
    )
    eligible = build_preflight(
        BENCH_ROOT,
        trial_provider_mode="codex_subscription",
        item_ids=eligible_ids,
    )

    assert full["checks"]["benchmark_required_env"] == {
        "passed": False,
        "selected_item_count": 100,
        "required_env_names": ["GH_TOKEN"],
        "missing_env_names": ["GH_TOKEN"],
        "missing_env_item_counts": {"GH_TOKEN": 5},
        "unknown_item_count": 0,
        "secret_value_persisted": False,
    }
    assert "benchmark_required_env" in full["blockers"]
    assert eligible["checks"]["benchmark_required_env"]["passed"] is True
    assert eligible["checks"]["benchmark_required_env"]["required_env_names"] == []
    assert "benchmark_required_env" not in eligible["blockers"]


def test_development_prewarm_receipt_binds_every_train_validation_item() -> None:
    manifest = SplitManifest.read(
        ROOT
        / "manifests"
        / "skilllearnbench_instance_holdout_credential_independent_v1.json"
    )
    selected_ids = (*manifest.train_ids, *manifest.validation_ids)
    rows = [
        {
            "item_id_hash": stable_hash({"item_id": item_id}),
            "family_hash": stable_hash({"family": manifest.family_by_id[item_id]}),
            "attempt_count": 1,
            "passed": True,
            "prebuilt_image_key": stable_hash({"image": item_id}),
            "prebuilt_image_id": "sha256:" + stable_hash({"image_id": item_id}),
            "agent_runtime_key": "a" * 64,
            "agent_runtime_version": "codex-cli 0.144.1",
            "error_type": None,
            "error_message_hash": None,
        }
        for item_id in selected_ids
    ]
    receipt = {
        "prewarm_version": DEVELOPMENT_PREWARM_VERSION,
        "manifest_hash": manifest.manifest_hash,
        "split_names": ["train", "validation"],
        "selected_item_set_hash": _selected_item_set_hash(manifest),
        "selected_item_count": len(selected_ids),
        "completed_item_count": len(selected_ids),
        "passed_item_count": len(selected_ids),
        "failed_item_count": 0,
        "unique_image_count": len(selected_ids),
        "parallel_workers": 4,
        "maximum_attempts": 3,
        "passed": True,
        "items": rows,
        "test_content_accessed": False,
        "secret_value_persisted": False,
        "raw_content_persisted": False,
    }
    receipt["receipt_hash"] = stable_hash(receipt)

    assert validate_development_prewarm_receipt(
        receipt,
        manifest=manifest,
    ) == receipt["receipt_hash"]

    tampered = {**receipt, "failed_item_count": 1}
    tampered["receipt_hash"] = stable_hash(
        {key: value for key, value in tampered.items() if key != "receipt_hash"}
    )
    with pytest.raises(ValueError, match="failed_item_count"):
        validate_development_prewarm_receipt(tampered, manifest=manifest)
