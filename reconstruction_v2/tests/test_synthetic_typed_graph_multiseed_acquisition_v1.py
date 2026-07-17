from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from assumption_agent.benchmarks import (
    synthetic_typed_graph_multiseed_acquisition_v1 as acquisition,
)


def _batch() -> bytes:
    return b"".join(bytes([index]) * acquisition.SEED_BYTES for index in range(8))


def _patch_preformal(monkeypatch: pytest.MonkeyPatch, root: Path) -> dict[str, str]:
    freeze = {"implementation_freeze_sha256": "f" * 64}
    monkeypatch.setattr(acquisition, "verify_frozen_design", lambda _root: {})
    monkeypatch.setattr(
        acquisition,
        "verify_implementation_freeze",
        lambda _root: (freeze, "a" * 40),
    )
    monkeypatch.setattr(acquisition, "_FORMAL_ENTRY_ACTIVE", True)
    return freeze


def _make_custody(
    monkeypatch: pytest.MonkeyPatch, root: Path, *, batch: bytes | None = None
) -> dict[str, object]:
    _patch_preformal(monkeypatch, root)
    raw = _batch() if batch is None else batch
    calls: list[int] = []

    def one_entropy_call(size: int) -> bytes:
        calls.append(size)
        assert (root / acquisition.SEED_MARKER_RELATIVE_PATH).is_file()
        return raw

    monkeypatch.setattr(acquisition.os, "urandom", one_entropy_call)
    custody = acquisition.create_seed_custody(project_root=root)
    assert calls == [256]
    return custody


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, object], mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(acquisition.canonical_bytes(payload) + b"\n")
    path.chmod(mode)


def test_design_is_self_hashed_and_scope_is_fixed() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / acquisition.DESIGN_RELATIVE_PATH
    design = _read(path)
    body = dict(design)
    declared = body.pop("design_sha256")
    assert declared == acquisition.DESIGN_SHA256
    assert acquisition.semantic_hash(body) == declared
    assert hashlib.sha256(path.read_bytes()).hexdigest() == acquisition.DESIGN_FILE_SHA256
    assert design["seed_contract"]["fixed_seed_count"] == 8
    assert design["cohort_contract"]["total_items"] == 512
    assert design["arm_contract"]["recipe_id"] == "R1_DEFINITION_1SWAP"
    assert design["analysis_contract"]["prohibited_interpretations_or_outputs"] == [
        "threshold",
        "p_value",
        "confidence_interval",
        "pass_fail",
        "promotion",
        "gate",
        "seed_level_population_inference",
    ]
    source = Path(acquisition.__file__).read_text(encoding="utf-8")
    assert "generate_all_blocks" not in source
    assert source.count("grammar.generate_block") == 1
    assert "private/cohort" not in source


def test_seed_custody_uses_one_marked_256_byte_call(tmp_path: Path, monkeypatch) -> None:
    custody = _make_custody(monkeypatch, tmp_path)
    raw = _batch()
    batch_path = tmp_path / acquisition.SEED_BATCH_RELATIVE_PATH
    assert batch_path.read_bytes() == raw
    assert stat.S_IMODE(batch_path.stat().st_mode) == acquisition.PRIVATE_MODE
    assert custody["seed_batch_commitment_sha256"] == hashlib.sha256(raw).hexdigest()
    assert custody["ordered_seed_commitments_sha256"] == [
        hashlib.sha256(raw[index * 32 : (index + 1) * 32]).hexdigest()
        for index in range(8)
    ]
    assert acquisition.load_seed_custody(
        tmp_path / acquisition.SEED_CUSTODY_RELATIVE_PATH
    ) == custody


@pytest.mark.parametrize("collision", ["duplicate", "original"])
def test_seed_collision_is_terminal_without_replacement(
    tmp_path: Path, monkeypatch, collision: str
) -> None:
    _patch_preformal(monkeypatch, tmp_path)
    raw = b"z" * 256 if collision == "duplicate" else _batch()
    calls: list[int] = []

    def entropy(size: int) -> bytes:
        calls.append(size)
        assert (tmp_path / acquisition.SEED_MARKER_RELATIVE_PATH).is_file()
        return raw

    monkeypatch.setattr(acquisition.os, "urandom", entropy)
    if collision == "original":
        monkeypatch.setattr(
            acquisition,
            "ORIGINAL_SEED_COMMITMENT_SHA256",
            hashlib.sha256(raw[:32]).hexdigest(),
        )
    with pytest.raises(acquisition.SyntheticMultiseedAcquisitionError):
        acquisition.create_seed_custody(project_root=tmp_path)
    assert calls == [256]
    assert not (tmp_path / acquisition.SEED_BATCH_RELATIVE_PATH).exists()
    assert not (tmp_path / acquisition.SEED_CUSTODY_RELATIVE_PATH).exists()
    failure = _read(tmp_path / acquisition.SEED_FAILURE_RELATIVE_PATH)
    assert failure["status"] == "terminal_seed_batch_invalid_no_replacement"
    assert failure["retry_replacement_or_smaller_N_authorized"] is False


def test_acquisition_generates_only_eight_A_hold_blocks_and_separates_labels(
    tmp_path: Path, monkeypatch
) -> None:
    custody = _make_custody(monkeypatch, tmp_path)
    original_generate = acquisition.grammar.generate_block
    calls: list[tuple[bytes, str]] = []

    def generate(seed: bytes, block: str):
        calls.append((seed, block))
        return original_generate(seed, block)

    monkeypatch.setattr(acquisition.grammar, "generate_block", generate)
    monkeypatch.setattr(
        acquisition,
        "_load_original_A_hold_commitments_after_marker",
        lambda root: (
            pytest.fail("acquisition marker was not durable before publication open")
            if not (root / acquisition.ACQUISITION_MARKER_RELATIVE_PATH).is_file()
            else frozenset()
        ),
    )

    def committed(root: Path, relative: Path, _field: str):
        assert relative == acquisition.SEED_CUSTODY_RELATIVE_PATH
        return _read(root / relative)

    monkeypatch.setattr(acquisition, "_load_committed_public_json", committed)
    receipt = acquisition.acquire_formal_cohort(project_root=tmp_path)
    assert receipt["status"] == (
        "formal_multiseed_A_hold_cohort_acquired_private_labels_separated"
    )
    assert len(calls) == 8
    assert [block for _seed, block in calls] == ["A_hold"] * 8
    assert [hashlib.sha256(seed).hexdigest() for seed, _block in calls] == custody[
        "ordered_seed_commitments_sha256"
    ]

    action_path = tmp_path / acquisition.ACTION_PACK_RELATIVE_PATH
    label_path = tmp_path / acquisition.LABEL_PACK_RELATIVE_PATH
    compiled_path = tmp_path / acquisition.COMPILED_COHORT_PACK_RELATIVE_PATH
    assert stat.S_IMODE(action_path.stat().st_mode) == acquisition.PRIVATE_MODE
    assert stat.S_IMODE(label_path.stat().st_mode) == acquisition.PRIVATE_MODE
    assert stat.S_IMODE(compiled_path.stat().st_mode) == acquisition.PRIVATE_MODE
    action = _read(action_path)
    labels = _read(label_path)
    compiled = _read(compiled_path)
    assert set(action) == {
        "schema",
        "version",
        "block",
        "seed_count",
        "item_count_per_seed",
        "total_item_count",
        "labels_included",
        "items",
        "pack_sha256",
    }
    assert set(labels) == {
        "schema",
        "version",
        "block",
        "seed_count",
        "item_count_per_seed",
        "total_item_count",
        "items",
        "pack_sha256",
    }
    assert len(action["items"]) == len(labels["items"]) == len(compiled["items"]) == 512
    assert compiled["schema"] == acquisition.COMPILED_COHORT_PACK_SCHEMA
    assert compiled["labels_included"] is True
    action_row = action["items"][0]
    label_row = labels["items"][0]
    assert set(action_row) == {
        "schema",
        "global_ordinal",
        "seed_index",
        "seed_ordinal",
        "question",
        "context",
        "nodes",
        "designated_edges",
        "action_item_sha256",
    }
    assert not {
        "gold_node_indices",
        "family_id",
        "family_role",
        "polarity",
        "edge_family",
        "item_commitment_sha256",
    }.intersection(action_row)
    assert set(label_row) == {
        "schema",
        "global_ordinal",
        "seed_index",
        "seed_ordinal",
        "action_item_sha256",
        "gold_node_indices",
        "family_id",
        "family_role",
        "polarity",
        "edge_family",
        "label_item_sha256",
    }
    assert label_row["action_item_sha256"] == action_row["action_item_sha256"]
    compiled_row = dict(compiled["items"][0])
    compiled_row_hash = compiled_row.pop("compiled_row_sha256")
    assert acquisition.semantic_hash(compiled_row) == compiled_row_hash
    assert compiled_row["item_commitment_sha256"]
    verified = acquisition._verify_compiled_cohort_pack(
        tmp_path,
        expected_file_hash=receipt["commitments"][
            "compiled_cohort_pack_file_sha256"
        ],
        expected_row_set_hash=receipt["commitments"][
            "compiled_row_commitment_set_sha256"
        ],
        expected_item_set_hash=receipt["generated_item_commitment_set_sha256"],
    )
    assert verified == compiled


def test_item_overlap_is_terminal_and_writes_no_pack(tmp_path: Path, monkeypatch) -> None:
    _make_custody(monkeypatch, tmp_path)
    old = "1" * 64
    call_index = 0

    def generate(_seed: bytes, block: str):
        nonlocal call_index
        assert block == "A_hold"
        seed_index = call_index
        call_index += 1
        return tuple(
            SimpleNamespace(
                item_commitment_sha256=(
                    old
                    if seed_index == 0 and ordinal == 0
                    else hashlib.sha256(f"{seed_index}:{ordinal}".encode()).hexdigest()
                )
            )
            for ordinal in range(64)
        )

    monkeypatch.setattr(acquisition.grammar, "generate_block", generate)
    monkeypatch.setattr(acquisition, "_validate_compiled_item", lambda _item, _i: None)
    monkeypatch.setattr(
        acquisition,
        "_load_original_A_hold_commitments_after_marker",
        lambda _root: frozenset({old}),
    )
    monkeypatch.setattr(
        acquisition,
        "_load_committed_public_json",
        lambda root, relative, _field: _read(root / relative),
    )
    with pytest.raises(acquisition.SyntheticMultiseedAcquisitionError):
        acquisition.acquire_formal_cohort(project_root=tmp_path)
    assert call_index == 8
    assert not (tmp_path / acquisition.ACTION_PACK_RELATIVE_PATH).exists()
    assert not (tmp_path / acquisition.LABEL_PACK_RELATIVE_PATH).exists()
    assert not (tmp_path / acquisition.COMPILED_COHORT_PACK_RELATIVE_PATH).exists()
    failure = _read(tmp_path / acquisition.ACQUISITION_RECEIPT_RELATIVE_PATH)
    assert failure["status"] == "terminal_multiseed_acquisition_invalid_no_replay"
    assert failure[
        "retry_replacement_smaller_N_or_overlap_repair_authorized"
    ] is False


def test_committed_infrastructure_terminal_is_publishable_without_a_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze = {"implementation_freeze_sha256": "f" * 64}
    acquisition_receipt = {
        "receipt_sha256": "c" * 64,
        "generated_item_commitment_set_sha256": "d" * 64,
    }
    acquisition_path = tmp_path / acquisition.ACQUISITION_RECEIPT_RELATIVE_PATH
    _write_json(acquisition_path, acquisition_receipt, acquisition.PUBLIC_MODE)
    acquisition_file_hash = hashlib.sha256(acquisition_path.read_bytes()).hexdigest()

    marker_body = {
        "schema": f"{acquisition.RUNNER_VERSION}_formal_attempt_marker",
        "version": acquisition.RUNNER_VERSION,
        "status": "sole_formal_replication_attempt_consumed",
        "actual_HEAD": "a" * 40,
        "design_sha256": acquisition.DESIGN_SHA256,
        "implementation_freeze_sha256": freeze["implementation_freeze_sha256"],
        "acquisition_receipt_sha256": acquisition_receipt["receipt_sha256"],
        "acquisition_receipt_file_sha256": acquisition_file_hash,
        "attempt_count": 1,
        "private_packs_opened_before_marker": False,
    }
    marker = {
        **marker_body,
        "marker_sha256": acquisition.semantic_hash(marker_body),
    }
    marker_path = tmp_path / acquisition.RUNNER_MARKER_RELATIVE_PATH
    _write_json(marker_path, marker, acquisition.PRIVATE_MODE)
    marker_file_hash = hashlib.sha256(marker_path.read_bytes()).hexdigest()

    result_body = {
        "schema": acquisition.RESULT_SCHEMA,
        "version": acquisition.VERSION,
        "status": acquisition.FAILURE_RESULT_STATUS,
        "invocation_HEAD": "a" * 40,
        "design_sha256": acquisition.DESIGN_SHA256,
        "design_file_sha256": acquisition.DESIGN_FILE_SHA256,
        "implementation_freeze_sha256": freeze["implementation_freeze_sha256"],
        "acquisition_receipt_sha256": acquisition_receipt["receipt_sha256"],
        "acquisition_receipt_file_sha256": acquisition_file_hash,
        "generated_item_commitment_set_sha256": acquisition_receipt[
            "generated_item_commitment_set_sha256"
        ],
        "formal_attempt_marker_sha256": marker["marker_sha256"],
        "formal_attempt_marker_file_sha256": marker_file_hash,
        "action_seal_file_sha256": None,
        "failure_class": "SyntheticTypedGraphMultiseedRunnerError",
        "retry_replacement_or_backup_attempt_authorized": False,
        "exception_message_seed_item_or_label_content_persisted_publicly": False,
        "result_must_be_committed_before_terminal_publication": True,
    }
    result = {
        **result_body,
        "receipt_sha256": acquisition.semantic_hash(result_body),
    }
    result_path = tmp_path / acquisition.RESULT_RELATIVE_PATH
    _write_json(result_path, result, acquisition.PUBLIC_MODE)
    monkeypatch.setattr(
        acquisition,
        "_load_committed_public_json",
        lambda _root, relative, _field: (
            result
            if relative == acquisition.RESULT_RELATIVE_PATH
            else pytest.fail("unexpected committed public read")
        ),
    )
    loaded, file_hash = acquisition._load_committed_terminal_result(
        tmp_path,
        freeze=freeze,
        acquisition=acquisition_receipt,
    )
    assert loaded == result
    assert file_hash == hashlib.sha256(result_path.read_bytes()).hexdigest()
