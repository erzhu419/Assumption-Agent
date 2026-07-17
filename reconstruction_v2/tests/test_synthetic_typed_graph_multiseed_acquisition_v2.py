from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from assumption_agent.benchmarks import (
    synthetic_typed_graph_multiseed_acquisition_v2 as acquisition,
)


def _batch() -> bytes:
    return b"".join(
        bytes([index + 32]) * acquisition.SEED_BYTES for index in range(8)
    )


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, object], mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(acquisition.canonical_bytes(payload) + b"\n")
    path.chmod(mode)


def _patch_preformal(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
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
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    *,
    batch: bytes | None = None,
) -> dict[str, object]:
    _patch_preformal(monkeypatch)
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


def _diagnostic() -> dict[str, object]:
    body: dict[str, object] = {
        "schema": acquisition.INTEGRATION_DIAGNOSTIC_SCHEMA,
        "version": acquisition.VERSION,
        "status": acquisition.INTEGRATION_SUCCESS_STATUS,
        "invocation_HEAD": "a" * 40,
        "design_sha256": acquisition.DESIGN_SHA256,
        "design_file_sha256": acquisition.DESIGN_FILE_SHA256,
        "bindings": [
            {
                "relative_path": relative,
                "file_sha256": str(index + 1) * 64,
                "git_blob_sha1": chr(ord("a") + index) * 40,
            }
            for index, relative in enumerate(acquisition.DIAGNOSTIC_BINDING_PATHS)
        ],
        "source_v1_publication": {
            "file_sha256": acquisition.V1_PUBLICATION_FILE_SHA256,
            "reproducibility_sha256": acquisition.V1_PUBLICATION_SHA256,
            "generated_item_commitment_set_sha256": (
                acquisition.V1_ITEM_COMMITMENT_SET_SHA256
            ),
            "projected_action_pack_sha256": "5" * 64,
            "projected_action_item_commitment_set_sha256": "6" * 64,
            "source_label_free_commitment_set_sha256": "b" * 64,
        },
        "chunk_schedule": {
            "chunk_count": 2,
            "texts_per_chunk": 8448,
            "total_text_count": 16896,
            "chunk_schedule_sha256": acquisition.CHUNK_SCHEDULE_SHA256,
        },
        "counts": {
            "item_count": 512,
            "action_work_unit_count": 1536,
            "submitted_action_work_unit_count": 1536,
            "terminal_action_work_unit_count": 1536,
            "official_retrieve_action_count": 512,
            "RAW_action_count": 512,
            "Agent_R1_action_count": 512,
        },
        "arms": ["RAW", "official_HippoRAG", "Agent_R1"],
        "official_concurrency_cap": 8,
        "local_concurrency_cap": 64,
        "observed_encoder_output_row_counts": [8448, 8448],
        "observed_encoder_input_row_counts": [8448, 8448],
        "official_peak_concurrency_count": 8,
        "local_peak_concurrency_count": 64,
        "runtime_binding_sha256": "7" * 64,
        "official_postflight_receipt_sha256": "c" * 64,
        "action_table_sha256": "8" * 64,
        "action_seal_sha256": "9" * 64,
        "action_seal_file_sha256": "a" * 64,
        "diagnostic_attempt_marker_sha256": "d" * 64,
        "diagnostic_attempt_marker_file_sha256": "e" * 64,
        "labels_opened": False,
        "scores_computed": False,
        "estimands_computed": False,
        "claims_made": False,
        "network_calls": 0,
        "retrieval_actions_model_outputs_or_scores_disclosed": False,
        "action_rows_or_ranked_indices_persisted": False,
        "action_identity_or_quality_used_for_decision": False,
        "diagnostic_is_non_claim": True,
        "fresh_formal_seed_authorized": True,
    }
    return {**body, "diagnostic_sha256": acquisition.semantic_hash(body)}


def test_design_and_module_are_fixed_to_v2_scope() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / acquisition.DESIGN_RELATIVE_PATH
    design = _read(path)
    body = dict(design)
    declared = body.pop("design_sha256")
    assert declared == acquisition.DESIGN_SHA256
    assert acquisition.semantic_hash(body) == declared
    assert hashlib.sha256(path.read_bytes()).hexdigest() == acquisition.DESIGN_FILE_SHA256
    assert design["seed_contract"]["fixed_seed_count"] == 8
    assert design["cohort_contract"]["generator_calls_total"] == 8
    assert design["cohort_contract"]["total_items"] == 512
    assert design["minilm_chunk_repair_contract"]["chunk_schedule_sha256"] == (
        acquisition.CHUNK_SCHEDULE_SHA256
    )
    source = Path(acquisition.__file__).read_text(encoding="utf-8")
    assert "generate_all_blocks" not in source
    assert source.count("grammar.generate_block") == 1
    assert "private/cohort" not in source


def test_only_exact_successful_non_scoring_diagnostic_authorizes_freeze() -> None:
    diagnostic = _diagnostic()
    assert acquisition._validate_integration_diagnostic_payload(diagnostic) == (
        diagnostic["diagnostic_sha256"]
    )
    for field, value in (
        ("labels_opened", True),
        ("scores_computed", True),
        ("fresh_formal_seed_authorized", False),
    ):
        drifted = dict(diagnostic)
        drifted.pop("diagnostic_sha256")
        drifted[field] = value
        drifted["diagnostic_sha256"] = acquisition.semantic_hash(drifted)
        with pytest.raises(acquisition.SyntheticMultiseedV2AcquisitionError):
            acquisition._validate_integration_diagnostic_payload(drifted)


def test_formal_freeze_requires_exact_diagnostic_code_test_tuples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = tuple(acquisition.DIAGNOSTIC_BINDING_PATHS)
    for index, relative in enumerate(paths):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"bound:{index}".encode())
    bindings = [
        {
            "relative_path": relative,
            "file_sha256": hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest(),
            "git_blob_sha1": acquisition._git_blob_sha1(
                (tmp_path / relative).read_bytes()
            ),
        }
        for relative in paths
    ]
    diagnostic = {"invocation_HEAD": "a" * 40, "status": acquisition.INTEGRATION_SUCCESS_STATUS, "bindings": bindings}
    monkeypatch.setattr(acquisition, "REQUIRED_FREEZE_PATHS", frozenset(paths))
    monkeypatch.setattr(acquisition, "verify_frozen_design", lambda _root: {})
    monkeypatch.setattr(acquisition, "_git_project_prefix", lambda _root: "")
    monkeypatch.setattr(
        acquisition,
        "load_committed_integration_diagnostic",
        lambda _root: (diagnostic, "d" * 64, "e" * 64),
    )
    monkeypatch.setattr(
        acquisition,
        "_committed_bytes",
        lambda root, relative: (root / relative).read_bytes(),
    )
    monkeypatch.setattr(
        acquisition,
        "_git",
        lambda _root, *arguments: (
            b"a" * 40 + b"\n"
            if arguments == ("rev-parse", "HEAD")
            else pytest.fail("unexpected Git call")
        ),
    )
    freeze = acquisition.create_implementation_freeze(tmp_path)
    assert [
        row for row in freeze["bindings"] if row["relative_path"] in paths
    ] == bindings

    second = tmp_path / "second"
    for index, relative in enumerate(paths):
        path = second / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"bound:{index}".encode())
    drifted = [dict(row) for row in bindings]
    drifted[0]["file_sha256"] = "0" * 64
    diagnostic["bindings"] = drifted
    with pytest.raises(acquisition.SyntheticMultiseedV2AcquisitionError):
        acquisition.create_implementation_freeze(second)
    assert not (second / acquisition.IMPLEMENTATION_FREEZE_RELATIVE_PATH).exists()


def test_seed_custody_uses_one_marked_256_byte_call(tmp_path: Path, monkeypatch) -> None:
    custody = _make_custody(monkeypatch, tmp_path)
    raw = _batch()
    batch_path = tmp_path / acquisition.SEED_BATCH_RELATIVE_PATH
    assert batch_path.read_bytes() == raw
    assert stat.S_IMODE(batch_path.stat().st_mode) == acquisition.PRIVATE_MODE
    assert custody["seed_batch_commitment_sha256"] == hashlib.sha256(raw).hexdigest()
    assert custody["fresh_seeds_disjoint_from_original_and_v1"] is True
    assert acquisition.load_seed_custody(
        tmp_path / acquisition.SEED_CUSTODY_RELATIVE_PATH
    ) == custody


@pytest.mark.parametrize("collision", ["duplicate", "original", "v1"])
def test_seed_collision_is_terminal_without_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, collision: str
) -> None:
    _patch_preformal(monkeypatch)
    raw = b"z" * 256 if collision == "duplicate" else _batch()
    first_commitment = hashlib.sha256(raw[:32]).hexdigest()
    calls: list[int] = []

    def entropy(size: int) -> bytes:
        calls.append(size)
        assert (tmp_path / acquisition.SEED_MARKER_RELATIVE_PATH).is_file()
        return raw

    monkeypatch.setattr(acquisition.os, "urandom", entropy)
    if collision == "v1":
        monkeypatch.setattr(
            acquisition,
            "V1_ORDERED_SEED_COMMITMENTS",
            (first_commitment, *("1" * 64 for _ in range(7))),
        )
    if collision == "original":
        monkeypatch.setattr(
            acquisition, "ORIGINAL_SEED_COMMITMENT_SHA256", first_commitment
        )
    with pytest.raises(acquisition.SyntheticMultiseedV2AcquisitionError):
        acquisition.create_seed_custody(project_root=tmp_path)
    assert calls == [256]
    assert not (tmp_path / acquisition.SEED_BATCH_RELATIVE_PATH).exists()
    assert not (tmp_path / acquisition.SEED_CUSTODY_RELATIVE_PATH).exists()
    failure = _read(tmp_path / acquisition.SEED_FAILURE_RELATIVE_PATH)
    assert failure["status"] == "terminal_v2_seed_batch_invalid_no_replacement"
    assert failure["retry_replacement_or_smaller_N_authorized"] is False


def test_acquisition_generates_eight_blocks_and_separates_all_private_packs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
        "_load_prior_item_commitments_after_marker",
        lambda root: (
            pytest.fail("prior publications opened before durable acquisition marker")
            if not (root / acquisition.ACQUISITION_MARKER_RELATIVE_PATH).is_file()
            else frozenset()
        ),
    )
    monkeypatch.setattr(
        acquisition,
        "_load_committed_public_json",
        lambda root, relative, _field: _read(root / relative),
    )
    receipt = acquisition.acquire_formal_cohort(project_root=tmp_path)
    assert receipt["status"] == (
        "formal_v2_multiseed_A_hold_cohort_acquired_private_labels_separated"
    )
    assert len(calls) == 8
    assert [block for _seed, block in calls] == ["A_hold"] * 8
    assert [hashlib.sha256(seed).hexdigest() for seed, _block in calls] == custody[
        "ordered_seed_commitments_sha256"
    ]
    for relative in (
        acquisition.ACTION_PACK_RELATIVE_PATH,
        acquisition.LABEL_PACK_RELATIVE_PATH,
        acquisition.COMPILED_COHORT_PACK_RELATIVE_PATH,
    ):
        assert stat.S_IMODE((tmp_path / relative).stat().st_mode) == (
            acquisition.PRIVATE_MODE
        )
    action = _read(tmp_path / acquisition.ACTION_PACK_RELATIVE_PATH)
    labels = _read(tmp_path / acquisition.LABEL_PACK_RELATIVE_PATH)
    compiled = _read(tmp_path / acquisition.COMPILED_COHORT_PACK_RELATIVE_PATH)
    assert len(action["items"]) == len(labels["items"]) == len(compiled["items"]) == 512
    action_row = action["items"][0]
    assert not {
        "gold_node_indices",
        "family_id",
        "family_role",
        "polarity",
        "edge_family",
        "item_commitment_sha256",
    }.intersection(action_row)
    assert labels["items"][0]["action_item_sha256"] == action_row[
        "action_item_sha256"
    ]
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


@pytest.mark.parametrize("prior_source", ["original", "v1"])
def test_item_overlap_with_either_prior_cohort_is_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prior_source: str,
) -> None:
    _make_custody(monkeypatch, tmp_path)
    old = hashlib.sha256(prior_source.encode()).hexdigest()
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
        "_load_prior_item_commitments_after_marker",
        lambda _root: frozenset({old}),
    )
    monkeypatch.setattr(
        acquisition,
        "_load_committed_public_json",
        lambda root, relative, _field: _read(root / relative),
    )
    with pytest.raises(acquisition.SyntheticMultiseedV2AcquisitionError):
        acquisition.acquire_formal_cohort(project_root=tmp_path)
    assert call_index == 8
    assert not (tmp_path / acquisition.ACTION_PACK_RELATIVE_PATH).exists()
    assert not (tmp_path / acquisition.LABEL_PACK_RELATIVE_PATH).exists()
    assert not (tmp_path / acquisition.COMPILED_COHORT_PACK_RELATIVE_PATH).exists()
    failure = _read(tmp_path / acquisition.ACQUISITION_RECEIPT_RELATIVE_PATH)
    assert failure["status"] == "terminal_v2_multiseed_acquisition_invalid_no_replay"


def test_terminal_publication_reads_stored_rows_without_regeneration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze = {"implementation_freeze_sha256": "f" * 64}
    receipt = {
        "receipt_sha256": "c" * 64,
        "generated_item_commitment_set_sha256": "d" * 64,
        "commitments": {
            "compiled_cohort_pack_file_sha256": "1" * 64,
            "compiled_row_commitment_set_sha256": "2" * 64,
        },
    }
    custody = {"seed_batch_commitment_sha256": hashlib.sha256(_batch()).hexdigest()}
    seeds = tuple(
        _batch()[index * 32 : (index + 1) * 32] for index in range(8)
    )
    stored_rows = [
        {
            "global_ordinal": index,
            "seed_index": index // 64,
            "seed_ordinal": index % 64,
            "item_commitment_sha256": hashlib.sha256(str(index).encode()).hexdigest(),
            "compiled_row_sha256": hashlib.sha256(f"row:{index}".encode()).hexdigest(),
        }
        for index in range(512)
    ]
    monkeypatch.setattr(acquisition, "_FORMAL_ENTRY_ACTIVE", True)
    monkeypatch.setattr(
        acquisition,
        "verify_implementation_freeze",
        lambda _root: (freeze, "a" * 40),
    )
    monkeypatch.setattr(
        acquisition,
        "load_committed_acquisition_receipt",
        lambda _root, **_kwargs: receipt,
    )
    monkeypatch.setattr(
        acquisition,
        "_load_committed_terminal_result",
        lambda _root, **_kwargs: ({"receipt_sha256": "e" * 64}, "b" * 64),
    )
    monkeypatch.setattr(
        acquisition,
        "_load_committed_public_json",
        lambda _root, relative, _field: (
            custody
            if relative == acquisition.SEED_CUSTODY_RELATIVE_PATH
            else pytest.fail("unexpected committed read")
        ),
    )
    monkeypatch.setattr(acquisition, "_read_seed_batch", lambda *_args: seeds)
    monkeypatch.setattr(
        acquisition,
        "_verify_compiled_cohort_pack",
        lambda *_args, **_kwargs: {"items": stored_rows},
    )
    monkeypatch.setattr(
        acquisition.grammar,
        "generate_block",
        lambda *_args: pytest.fail("publication must never regenerate the cohort"),
    )
    artifact = acquisition.publish_terminal(project_root=tmp_path)
    assert len(artifact["formal_seed_hexes"]) == 8
    assert len(artifact["items"]) == 512
    assert artifact["cohort_regenerated_during_publication"] is False
    assert artifact["retrieval_actions_model_outputs_or_scores_included"] is False
    assert all("compiled_row_sha256" not in row for row in artifact["items"])


def test_v1_publication_projection_is_exact_and_public_only() -> None:
    root = Path(__file__).resolve().parents[1]
    seeds, items = acquisition._load_v1_publication_projection(root)
    assert seeds == frozenset(acquisition.V1_ORDERED_SEED_COMMITMENTS)
    assert len(items) == 512
    assert acquisition.stable_hash(
        [
            row["item_commitment_sha256"]
            for row in _read(root / acquisition.V1_PUBLICATION_RELATIVE_PATH)["items"]
        ]
    ) == acquisition.V1_ITEM_COMMITMENT_SET_SHA256
