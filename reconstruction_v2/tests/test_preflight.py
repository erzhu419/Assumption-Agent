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
from assumption_agent.benchmarks.docker_egress import DEPENDENCY_CACHE_POLICY_VERSION
from assumption_agent.benchmarks.offline_verifier import (
    OFFLINE_VERIFIER_POLICY_VERSION,
    offline_verifier_profile_for_family,
    offline_verifier_runtime_key,
)
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
    monkeypatch.setenv("ASSUMPTION_V2_API_BASE", "https://ruoli.dev")
    monkeypatch.setenv("ASSUMPTION_V2_API_KEY", "test-key")
    monkeypatch.setenv("ASSUMPTION_V2_API_ALLOWED_IPV4S", "45.78.76.197")
    monkeypatch.setenv("ASSUMPTION_V2_SKILLLEARN_CACHE_ONLY", "1")
    adapter = SkillLearnBenchAdapter(BENCH_ROOT)
    full_ids = [item.id for item in adapter.discover()]
    eligible_ids = [item.id for item in adapter.credential_independent_items()]

    full = build_preflight(
        BENCH_ROOT,
        trial_provider_mode="openai_compatible",
        item_ids=full_ids,
    )
    eligible = build_preflight(
        BENCH_ROOT,
        trial_provider_mode="openai_compatible",
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
    assert full["checks"]["offline_verifier_profile_coverage"][
        "missing_profile_item_count"
    ] == 13
    assert eligible["checks"]["offline_verifier_profile_coverage"][
        "missing_profile_item_count"
    ] == 8
    assert full["checks"]["offline_verifier_profile_coverage"][
        "activation_blocked_item_count"
    ] == 3
    assert eligible["checks"]["offline_verifier_profile_coverage"][
        "activation_blocked_item_count"
    ] == 3
    assert eligible["checks"]["offline_verifier_profile_coverage"][
        "activation_blocked_profile_count"
    ] == 1
    assert eligible["checks"]["offline_verifier_profile_coverage"][
        "activation_blocker_item_counts"
    ] == {"druid_maven_cache_incomplete": 3}
    assert full["checks"]["verifier_payload_completeness"][
        "missing_test_outputs_item_count"
    ] == 1
    assert eligible["checks"]["verifier_payload_completeness"][
        "missing_test_outputs_item_count"
    ] == 1
    assert "verifier_payload_completeness" in full["blockers"]
    assert "verifier_payload_completeness" in eligible["blockers"]

    poster_ids = [
        item.id for item in adapter.discover() if item.family == "anthropic-poster-design"
    ]
    poster = build_preflight(
        BENCH_ROOT,
        trial_provider_mode="openai_compatible",
        item_ids=poster_ids,
    )
    assert poster["checks"]["offline_verifier_profile_coverage"]["passed"] is True
    assert poster["checks"]["offline_verifier_profile_coverage"][
        "missing_profile_item_count"
    ] == 0
    assert poster["checks"]["verifier_payload_completeness"]["passed"] is True


def test_development_prewarm_receipt_binds_every_manifest_item() -> None:
    manifest = SplitManifest.read(
        ROOT
        / "manifests"
        / "skilllearnbench_instance_holdout_credential_independent_v1.json"
    )
    selected_ids = (
        *manifest.train_ids,
        *manifest.validation_ids,
        *manifest.test_ids,
    )
    rows = []
    for item_id in selected_ids:
        profile = offline_verifier_profile_for_family(
            manifest.family_by_id[item_id]
        )
        rows.append(
            {
                "item_id_hash": stable_hash({"item_id": item_id}),
                "family_hash": stable_hash(
                    {"family": manifest.family_by_id[item_id]}
                ),
                "attempt_count": 1,
                "passed": True,
                "prebuilt_image_key": stable_hash({"image": item_id}),
                "prebuilt_image_id": "sha256:"
                + stable_hash({"image_id": item_id}),
                "agent_runtime_key": "a" * 64,
                "agent_runtime_version": "codex-cli 0.144.1",
                "verifier_runtime_mode": (
                    "local_profile" if profile is not None else "native_image"
                ),
                "offline_verifier_profile_id": (
                    profile.profile_id if profile is not None else None
                ),
                "offline_verifier_profile_hash": (
                    profile.profile_hash if profile is not None else None
                ),
                "offline_verifier_runtime_key": (
                    offline_verifier_runtime_key(profile=profile)
                    if profile is not None
                    else None
                ),
                "verifier_runtime_network": "none",
                "error_type": None,
                "error_message_hash": None,
            }
        )
    local_profile_item_count = sum(
        row["verifier_runtime_mode"] == "local_profile" for row in rows
    )
    receipt = {
        "prewarm_version": DEVELOPMENT_PREWARM_VERSION,
        "manifest_hash": manifest.manifest_hash,
        "split_names": ["train", "validation", "test"],
        "selected_item_set_hash": _selected_item_set_hash(manifest),
        "selected_item_count": len(selected_ids),
        "completed_item_count": len(selected_ids),
        "passed_item_count": len(selected_ids),
        "failed_item_count": 0,
        "unique_image_count": len(selected_ids),
        "parallel_workers": 4,
        "maximum_attempts": 3,
        "dependency_cache_policy": DEPENDENCY_CACHE_POLICY_VERSION,
        "dependency_cache_only_enforced": True,
        "offline_verifier_policy": OFFLINE_VERIFIER_POLICY_VERSION,
        "offline_verifier_runtime_network": "none",
        "offline_verifier_runtime_network_fallback_allowed": False,
        "local_profile_item_count": local_profile_item_count,
        "native_image_verifier_item_count": len(rows) - local_profile_item_count,
        "unique_offline_verifier_profile_count": len(
            {
                row["offline_verifier_profile_hash"]
                for row in rows
                if row["offline_verifier_profile_hash"]
            }
        ),
        "unique_offline_verifier_runtime_count": len(
            {
                row["offline_verifier_runtime_key"]
                for row in rows
                if row["offline_verifier_runtime_key"]
            }
        ),
        "offline_verifier_profile_set_hash": stable_hash(
            sorted(
                {
                    row["offline_verifier_profile_hash"]
                    for row in rows
                    if row["offline_verifier_profile_hash"]
                }
            )
        ),
        "offline_verifier_runtime_set_hash": stable_hash(
            sorted(
                {
                    row["offline_verifier_runtime_key"]
                    for row in rows
                    if row["offline_verifier_runtime_key"]
                }
            )
        ),
        "online_build_attempted": False,
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

    tampered_rows = [dict(row) for row in rows]
    local_index = next(
        index
        for index, row in enumerate(tampered_rows)
        if row["verifier_runtime_mode"] == "local_profile"
    )
    tampered_rows[local_index]["offline_verifier_runtime_key"] = "b" * 64
    wrong_runtime = {**receipt, "items": tampered_rows}
    wrong_runtime["receipt_hash"] = stable_hash(
        {
            key: value
            for key, value in wrong_runtime.items()
            if key != "receipt_hash"
        }
    )
    with pytest.raises(ValueError, match="profile does not match family"):
        validate_development_prewarm_receipt(wrong_runtime, manifest=manifest)
