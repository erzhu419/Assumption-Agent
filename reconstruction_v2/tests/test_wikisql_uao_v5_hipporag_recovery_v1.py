from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from assumption_agent.benchmarks import (
    wikisql_uao_action_runtime_v1 as action_runtime,
)
from replication_runtime.wikisql_uao_v5_hipporag_recovery_v1 import runner


def _items() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ordinal in range(runner.ITEM_COUNT):
        rows.append(
            {
                "opaque_item_id": hashlib.sha256(
                    f"item-{ordinal:03d}".encode("ascii")
                ).hexdigest(),
                "physical_rows": [
                    [f"row-{row:02d}", ordinal + row] for row in range(11)
                ],
                "question": f"Which row belongs to item {ordinal}?",
                "table_header": ["name", "value"],
                "table_types": ["text", "real"],
            }
        )
    return rows


def _config(tmp_path: Path, view_path: Path) -> runner.RecoveryConfig:
    binding = runner.FileBinding(
        path=view_path,
        sha256=runner.file_sha256(view_path),
        mode=0o600,
    )
    return runner.RecoveryConfig(
        path=tmp_path / "control/recovery_config.json",
        recovery_root=tmp_path,
        unit_name="unused.service",
        shard_count=4,
        lane_assignments=("0", "0", "1", "1"),
        files={"source_view": binding},
        trees={},
        self_sha256="0" * 64,
    )


def test_four_shards_reassemble_one_full_view_action_pack(
    tmp_path: Path,
) -> None:
    view = action_runtime.build_view_pack(block="A_hold", items=_items())
    source = tmp_path / "source_view.json"
    source.write_bytes(runner.canonical_json_bytes(view))
    source.chmod(0o600)
    work = tmp_path / "work"
    work.mkdir(mode=0o700)
    (work / "hippo").mkdir(mode=0o700)
    config = _config(tmp_path, source)

    full_view, full_items, shard_paths = runner._make_shards(
        config=config,
        action_runtime=action_runtime,
        work=work,
    )

    assert len(shard_paths) == 4
    shard_rows: list[dict[str, object]] = []
    for path in shard_paths:
        shard = runner._read_json(path, field="test shard")
        items = action_runtime.decode_view_pack(
            shard,
            expected_block="A_hold",
            expected_count=18,
        )
        shard_rows.extend(
            {
                "opaque_item_id": item.item_id,
                "top5_row_ids": [0, 1, 2, 3, 4],
            }
            for item in items
        )

    action, receipt = runner._seal_hippo(
        config=config,
        action_runtime=action_runtime,
        full_view=full_view,
        full_items=full_items,
        shard_rows=shard_rows,
        shard_receipts={"shards": []},
        work=work,
    )

    decoded = action_runtime.decode_action_pack(
        action,
        expected_block="A_hold",
        expected_arm="HippoRAG",
        expected_action_view_pack_sha256=view["self_sha256"],
    )
    assert len(decoded) == runner.ITEM_COUNT
    assert receipt["patched_source_sha256"] == runner.PATCHED_SOURCE_SHA256
    assert receipt["parallel_shard_count"] == 4


def test_reuses_exact_previously_verified_short_model_aliases(
    tmp_path: Path,
) -> None:
    llm = tmp_path / "llm"
    embedding = tmp_path / "embedding"
    llm.mkdir(mode=0o700)
    embedding.mkdir(mode=0o700)
    hippo = tmp_path / "hippo"
    hippo.mkdir(mode=0o700)
    aliases = hippo / "model_aliases"
    aliases.mkdir(mode=0o700)
    (aliases / "smollm2").symlink_to(llm, target_is_directory=True)
    (aliases / "minilm").symlink_to(embedding, target_is_directory=True)
    (aliases / "stdout.log").write_bytes(b"")
    (aliases / "stderr.log").write_bytes(b"")
    body = {
        "derived_hipporag_component": (
            "Transformers_smollm2_Transformers_minilm"
        ),
        "derived_hipporag_component_utf8_bytes": 40,
        "schema": "wikisql_uao_short_model_alias_runtime_receipt_v1",
        "status": "short_model_aliases_bound_and_verified",
    }
    receipt = {**body, "self_sha256": runner.semantic_sha256(body)}
    receipt_path = hippo / "model_alias.safe.json"
    receipt_path.write_bytes(runner.canonical_json_bytes(receipt))
    receipt_path.chmod(0o600)
    config = runner.RecoveryConfig(
        path=tmp_path / "unused.json",
        recovery_root=tmp_path,
        unit_name="unused.service",
        shard_count=4,
        lane_assignments=("0", "0", "1", "1"),
        files={
            "original_model_alias_receipt": runner.FileBinding(
                path=receipt_path,
                sha256=runner.file_sha256(receipt_path),
                mode=0o600,
            )
        },
        trees={},
        self_sha256="0" * 64,
    )
    trees = {
        "hippo_llm_model_tree": SimpleNamespace(path=llm),
        "encoder_model_tree": SimpleNamespace(path=embedding),
    }
    original = SimpleNamespace(tree=lambda name: trees[name])

    root, checked = runner._verify_original_model_aliases(
        config=config,
        original_config=original,
    )

    assert root == aliases
    assert checked == receipt


def _write_self_hashed(path: Path, body: dict[str, object]) -> runner.FileBinding:
    value = {**body, "self_sha256": runner.semantic_sha256(body)}
    path.write_bytes(runner.canonical_json_bytes(value))
    path.chmod(0o600)
    return runner.FileBinding(
        path=path,
        sha256=runner.file_sha256(path),
        mode=0o600,
    )


def test_continuation_requires_preserved_zero_effect_preindex_evidence(
    tmp_path: Path,
) -> None:
    prior_config = tmp_path / "prior_config.json"
    prior_config.write_bytes(b"{}\n")
    prior_config.chmod(0o600)
    prior_config_binding = runner.FileBinding(
        path=prior_config,
        sha256=runner.file_sha256(prior_config),
        mode=0o600,
    )
    invocation = "a1a4ba82c5974d6b93d58bfc78c92225"
    attempt = _write_self_hashed(
        tmp_path / "attempt.json",
        {
            "config_file_sha256": prior_config_binding.sha256,
            "invocation_id_sha256": hashlib.sha256(
                invocation.encode("ascii")
            ).hexdigest(),
            "nrestarts": 0,
            "schema": "wikisql_uao_v5_hipporag_recovery_v1_attempt_v1",
            "status": "claimed_once",
        },
    )
    intent = _write_self_hashed(
        tmp_path / "intent.json",
        {
            "Agent_and_RAW_rerun_count": 0,
            "A_hold_label_open_count_before_three_arm_barrier": 0,
            "schema": "wikisql_uao_v5_hipporag_recovery_v1_intent_v1",
            "status": "missing_HippoRAG_only_intent_frozen",
        },
    )
    preserved = tmp_path / "preserved_work"
    preserved.mkdir(mode=0o700)
    hippo = preserved / "hippo"
    hippo.mkdir(mode=0o700)
    attempt_value = runner._read_json(attempt.path, field="test attempt")
    intent_value = runner._read_json(intent.path, field="test intent")
    evidence = _write_self_hashed(
        tmp_path / "evidence.json",
        {
            "API_or_online_evaluation_count": 0,
            "Agent_and_RAW_rerun_count": 0,
            "action_barrier_count": 0,
            "action_file_count": 0,
            "completed_HippoRAG_shard_count": 0,
            "effect_measurement_consumed": False,
            "index_directory_count": 0,
            "label_projection_count": 0,
            "original_attempt_file_sha256": attempt.sha256,
            "original_attempt_self_sha256": attempt_value["self_sha256"],
            "original_intent_file_sha256": intent.sha256,
            "original_intent_self_sha256": intent_value["self_sha256"],
            "preserved_work_path": str(preserved),
            "schema": (
                "wikisql_uao_v5_hipporag_recovery_v1_"
                "preindex_dstate_evidence_v1"
            ),
            "scorer_launch_count": 0,
            "service_invocation_id": invocation,
            "shard_launch_count": 4,
            "status": "PRESERVED_PREINDEX_INFRASTRUCTURE_INVALID_ZERO_EFFECT",
            "worker_receipt_count": 0,
        },
    )
    config = runner.RecoveryConfig(
        path=tmp_path / "unused.json",
        recovery_root=tmp_path,
        unit_name="unused.service",
        shard_count=4,
        lane_assignments=("0", "0", "1", "1"),
        files={
            "preindex_dstate_evidence": evidence,
            "prior_attempt": attempt,
            "prior_intent": intent,
            "prior_recovery_config": prior_config_binding,
        },
        trees={},
        self_sha256="0" * 64,
    )

    checked_attempt, checked_intent, checked_evidence = (
        runner._verify_continuation_evidence(config)
    )
    assert checked_attempt["self_sha256"] == attempt_value["self_sha256"]
    assert checked_intent["self_sha256"] == intent_value["self_sha256"]
    assert checked_evidence["effect_measurement_consumed"] is False

    (hippo / "A_hold.HippoRAG.actions.json").write_bytes(b"{}\n")
    with pytest.raises(runner.RecoveryError, match="gained effect outputs"):
        runner._verify_continuation_evidence(config)
