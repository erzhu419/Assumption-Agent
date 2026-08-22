from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import stat
import subprocess
import sys
import textwrap
import threading
from typing import Callable

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from hegel_machine import (
    phase2b_formal_wire_covert_step5_r01_durable_anchor_v1 as durable_anchor,
)
from hegel_machine import (
    phase2b_formal_wire_covert_step5_r01_external_witness_admission_contract_v1
    as admission,
)
from test_phase2b_formal_wire_covert_step5_r01_external_witness_admission_contract_v1 import (
    _PositiveCorpus,
    _positive_corpus,
    _signed_admission_object,
)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> _PositiveCorpus:
    return _positive_corpus(tmp_path_factory.mktemp("durable-anchor-corpus"))


def _private_spool(tmp_path: Path, name: str = "a-spool") -> Path:
    path = tmp_path / name
    path.mkdir(mode=0o700)
    path.chmod(0o700)
    return path


def _construct_arguments(
    corpus: _PositiveCorpus, spool: Path
) -> dict[str, object]:
    call = corpus.call
    return {
        "bootstrap_external_witness_policy_payload": corpus.payloads[
            "bootstrap_policy"
        ],
        "online_witness_policy_payload": corpus.payloads["online_policy"],
        "durable_anchor_policy_payload": corpus.payloads["durable_policy"],
        "out_of_band_expected_bootstrap_external_witness_policy_id_bytes": call[
            "out_of_band_expected_bootstrap_external_witness_policy_id_bytes"
        ],
        "out_of_band_expected_bootstrap_external_witness_policy_payload_sha256_digest": call[
            "out_of_band_expected_bootstrap_external_witness_policy_payload_sha256_digest"
        ],
        "out_of_band_expected_online_witness_policy_id_bytes": call[
            "out_of_band_expected_online_witness_policy_id_bytes"
        ],
        "out_of_band_expected_online_witness_policy_payload_sha256_digest": call[
            "out_of_band_expected_online_witness_policy_payload_sha256_digest"
        ],
        "out_of_band_expected_durable_anchor_policy_id_bytes": call[
            "out_of_band_expected_durable_anchor_policy_id_bytes"
        ],
        "out_of_band_expected_durable_anchor_policy_payload_sha256_digest": call[
            "out_of_band_expected_durable_anchor_policy_payload_sha256_digest"
        ],
        "durable_anchor_private_key": corpus.durable_private_key,
        "owner_private_spool_directory_path": str(spool),
    }


def _construct(
    corpus: _PositiveCorpus,
    spool: Path,
    **overrides: object,
) -> durable_anchor.LocalDurableAnchorActorV1:
    arguments = _construct_arguments(corpus, spool)
    arguments.update(overrides)
    return durable_anchor.construct_local_durable_anchor_actor_v1(**arguments)


def _append(
    actor: object,
    corpus: _PositiveCorpus,
    *,
    q1_request: bytes | None = None,
    q1_response: bytes | None = None,
    a_request: bytes | None = None,
) -> durable_anchor.IssuedLocalDurableAnchorAppendCandidateV1:
    return durable_anchor.compare_and_append_local_durable_anchor_v1(
        actor,
        corpus.payloads["q1_request"] if q1_request is None else q1_request,
        corpus.payloads["q1_response"] if q1_response is None else q1_response,
        corpus.payloads["a_request"] if a_request is None else a_request,
    )


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _invalid_signature_payload(
    corpus: _PositiveCorpus, name: str
) -> bytes:
    root = copy.deepcopy(dict(corpus.roots[name]))
    signature_field = "bootstrap_signature" if name == "online_policy" else "signature"
    signature = root[signature_field]
    original = signature["signature_hex"]
    signature["signature_hex"] = ("0" if original[0] != "0" else "1") + original[1:]
    return admission.canonical_external_witness_admission_json_bytes_v1(root)


def _alternate_request_triplet(
    corpus: _PositiveCorpus,
) -> tuple[bytes, bytes, bytes]:
    bootstrap = corpus.roots["bootstrap_policy"]
    online = corpus.roots["online_policy"]["body"]
    prepare_nonce = "d" * 64

    q1_body = copy.deepcopy(dict(corpus.roots["q1_request"]["body"]))
    q1_body["prepare_nonce"] = prepare_nonce
    q1_payload, q1 = _signed_admission_object(
        artifact_kind="q1_prepare_request",
        schema=admission.Q1_PREPARE_REQUEST_SCHEMA_V1,
        id_field="q1_request_id",
        body=q1_body,
        private_key=corpus.graph.witness_private_key,
        signer_key_id=bootstrap["witness_key_id"],
        purpose=admission.SIGN_Q1_PREPARE_REQUEST_PURPOSE_V1,
    )

    response_body = copy.deepcopy(dict(corpus.roots["q1_response"]["body"]))
    response_body.update(
        {
            "q1_request_id": q1["q1_request_id"],
            "q1_request_payload_sha256": _sha(q1_payload),
            "prepare_nonce": prepare_nonce,
        }
    )
    response_payload, response = _signed_admission_object(
        artifact_kind="q1_prepare_response",
        schema=admission.Q1_PREPARE_RESPONSE_SCHEMA_V1,
        id_field="q1_response_id",
        body=response_body,
        private_key=corpus.online_private_key,
        signer_key_id=online["online_key_id"],
        purpose=admission.SIGN_Q1_PREPARE_RESPONSE_PURPOSE_V1,
    )

    request_body = copy.deepcopy(dict(corpus.roots["a_request"]["body"]))
    request_body.update(
        {
            "q1_request_id": q1["q1_request_id"],
            "q1_request_payload_sha256": _sha(q1_payload),
            "q1_response_id": response["q1_response_id"],
            "q1_response_payload_sha256": _sha(response_payload),
            "prepare_nonce": prepare_nonce,
        }
    )
    request_payload, _ = _signed_admission_object(
        artifact_kind="a_compare_and_append_request",
        schema=admission.A_COMPARE_AND_APPEND_REQUEST_SCHEMA_V1,
        id_field="a_request_id",
        body=request_body,
        private_key=corpus.online_private_key,
        signer_key_id=online["online_key_id"],
        purpose=admission.SIGN_A_COMPARE_AND_APPEND_REQUEST_PURPOSE_V1,
    )
    return q1_payload, response_payload, request_payload


def _bomb(*args: object, **kwargs: object) -> object:
    raise AssertionError("post-import global or signing path was consulted")


def _fork_append_worker(
    connection: object,
    actor: object,
    q1_request: bytes,
    q1_response: bytes,
    a_request: bytes,
) -> None:
    try:
        result = durable_anchor.compare_and_append_local_durable_anchor_v1(
            actor, q1_request, q1_response, a_request
        )
        connection.send(
            (
                "ok",
                result.idempotent_replay,
                tuple(_sha(payload) for payload in result.payloads_in_contract_order),
            )
        )
    except BaseException as exc:  # pragma: no cover - asserted in the parent.
        connection.send(("error", type(exc).__name__, str(exc)))
    finally:
        connection.close()


def test_source_pin_and_exact_b480_first_append_differential(
    tmp_path: Path, corpus: _PositiveCorpus
) -> None:
    assert durable_anchor.ADMISSION_CONTRACT_SOURCE_SHA256_V1 == (
        "b480b0d4b286e1b0cfb43f88846bbbf635b753fffc64b5e89e9073264360e285"
    )
    spool = _private_spool(tmp_path)
    actor = _construct(corpus, spool)
    result = _append(actor, corpus)

    assert result.idempotent_replay is False
    assert result.payloads_in_contract_order == (
        corpus.payloads["a_checkpoint"],
        corpus.payloads["a_head"],
        corpus.payloads["a_receipt"],
    )
    assert result.a_durable_checkpoint_payload == corpus.payloads["a_checkpoint"]
    assert result.a_durable_head_payload == corpus.payloads["a_head"]
    assert (
        result.a_compare_and_append_receipt_payload
        == corpus.payloads["a_receipt"]
    )
    assert tuple(spool.iterdir())
    assert stat.S_IMODE(spool.stat().st_mode) == 0o700
    for name in result.spool_file_basenames_in_commit_order:
        metadata = (spool / name).stat(follow_symlinks=False)
        assert stat.S_ISREG(metadata.st_mode)
        assert stat.S_IMODE(metadata.st_mode) == 0o600
        assert metadata.st_nlink == 1

    verifier_call = dict(corpus.call)
    verifier_call.update(
        {
            "a_durable_checkpoint_payload": result.a_durable_checkpoint_payload,
            "a_durable_head_payload": result.a_durable_head_payload,
            "a_compare_and_append_receipt_payload": (
                result.a_compare_and_append_receipt_payload
            ),
        }
    )
    verified = admission.verify_external_witness_admission_transcript_candidate_v1(
        **verifier_call
    )
    assert verified.presented_q1_a_q2_chain_verified is True
    assert verified.authority_evidence is False


@pytest.mark.parametrize(
    "field",
    [
        "bootstrap_external_witness_policy_payload",
        "online_witness_policy_payload",
        "durable_anchor_policy_payload",
    ],
)
def test_all_three_policy_inputs_are_raw_only(
    tmp_path: Path, corpus: _PositiveCorpus, field: str
) -> None:
    spool = _private_spool(tmp_path)
    arguments = _construct_arguments(corpus, spool)
    arguments[field] = json.loads(arguments[field])
    with pytest.raises(durable_anchor.Phase2BStep5R01DurableAnchorErrorV1):
        durable_anchor.construct_local_durable_anchor_actor_v1(**arguments)
    assert list(spool.iterdir()) == []


@pytest.mark.parametrize(
    "field",
    [
        "out_of_band_expected_bootstrap_external_witness_policy_id_bytes",
        "out_of_band_expected_bootstrap_external_witness_policy_payload_sha256_digest",
        "out_of_band_expected_online_witness_policy_id_bytes",
        "out_of_band_expected_online_witness_policy_payload_sha256_digest",
        "out_of_band_expected_durable_anchor_policy_id_bytes",
        "out_of_band_expected_durable_anchor_policy_payload_sha256_digest",
    ],
)
def test_every_policy_oob_pin_fails_closed(
    tmp_path: Path, corpus: _PositiveCorpus, field: str
) -> None:
    spool = _private_spool(tmp_path)
    arguments = _construct_arguments(corpus, spool)
    original = arguments[field]
    assert type(original) is bytes
    if field.endswith("_id_bytes"):
        replacement = b"0" if original[-1:] != b"0" else b"1"
        arguments[field] = original[:-1] + replacement
    else:
        arguments[field] = bytes([original[0] ^ 1]) + original[1:]
    with pytest.raises(durable_anchor.Phase2BStep5R01DurableAnchorErrorV1):
        durable_anchor.construct_local_durable_anchor_actor_v1(**arguments)
    assert list(spool.iterdir()) == []


@pytest.mark.parametrize("name", ["online_policy", "durable_policy"])
def test_raw_policy_signatures_are_verified(
    tmp_path: Path, corpus: _PositiveCorpus, name: str
) -> None:
    spool = _private_spool(tmp_path)
    payload = _invalid_signature_payload(corpus, name)
    if name == "online_policy":
        overrides: dict[str, object] = {
            "online_witness_policy_payload": payload,
            "out_of_band_expected_online_witness_policy_payload_sha256_digest": hashlib.sha256(
                payload
            ).digest(),
        }
    else:
        overrides = {
            "durable_anchor_policy_payload": payload,
            "out_of_band_expected_durable_anchor_policy_payload_sha256_digest": hashlib.sha256(
                payload
            ).digest(),
        }
    with pytest.raises(durable_anchor.Phase2BStep5R01DurableAnchorErrorV1) as caught:
        _construct(corpus, spool, **overrides)
    assert caught.value.code == durable_anchor.FAIL_SIGNATURE_V1


@pytest.mark.parametrize("name", ["q1_request", "q1_response", "a_request"])
def test_raw_request_signatures_are_verified(
    tmp_path: Path, corpus: _PositiveCorpus, name: str
) -> None:
    actor = _construct(corpus, _private_spool(tmp_path))
    overrides = {name: _invalid_signature_payload(corpus, name)}
    with pytest.raises(durable_anchor.Phase2BStep5R01DurableAnchorErrorV1) as caught:
        _append(actor, corpus, **overrides)
    assert caught.value.code == durable_anchor.FAIL_SIGNATURE_V1


def test_matching_a_private_key_is_mandatory(
    tmp_path: Path, corpus: _PositiveCorpus
) -> None:
    with pytest.raises(durable_anchor.Phase2BStep5R01DurableAnchorErrorV1) as caught:
        _construct(
            corpus,
            _private_spool(tmp_path),
            durable_anchor_private_key=Ed25519PrivateKey.generate(),
        )
    assert caught.value.code == durable_anchor.FAIL_KEY_V1


def test_exact_retry_returns_original_bytes_without_resigning(
    tmp_path: Path,
    corpus: _PositiveCorpus,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spool = _private_spool(tmp_path)
    actor = _construct(corpus, spool)
    first = _append(actor, corpus)
    before = {
        path.name: (path.stat().st_dev, path.stat().st_ino, path.stat().st_mtime_ns)
        for path in spool.iterdir()
    }

    captured_globals = durable_anchor.compare_and_append_local_durable_anchor_v1.__globals__
    monkeypatch.setitem(captured_globals, "_compose_a_outputs", _bomb)
    replay = _append(actor, corpus)

    assert replay.idempotent_replay is True
    assert replay.payloads_in_contract_order == first.payloads_in_contract_order
    assert replay.artifact_ids == first.artifact_ids
    assert replay.payload_sha256s == first.payload_sha256s
    after = {
        path.name: (path.stat().st_dev, path.stat().st_ino, path.stat().st_mtime_ns)
        for path in spool.iterdir()
    }
    assert after == before


def test_reconstructed_actor_replays_complete_same_inode_slot_without_resigning(
    tmp_path: Path,
    corpus: _PositiveCorpus,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spool = _private_spool(tmp_path)
    first_actor = _construct(corpus, spool)
    first = _append(first_actor, corpus)
    before = {
        path.name: (path.stat().st_dev, path.stat().st_ino, path.stat().st_mtime_ns)
        for path in spool.iterdir()
    }
    reconstructed_actor = _construct(corpus, spool)
    captured_globals = durable_anchor.compare_and_append_local_durable_anchor_v1.__globals__
    monkeypatch.setitem(captured_globals, "_compose_a_outputs", _bomb)

    replay = _append(reconstructed_actor, corpus)

    assert replay.idempotent_replay is True
    assert replay.payloads_in_contract_order == first.payloads_in_contract_order
    assert {
        path.name: (path.stat().st_dev, path.stat().st_ino, path.stat().st_mtime_ns)
        for path in spool.iterdir()
    } == before


def test_same_slot_different_valid_request_conflicts_without_poisoning_retry(
    tmp_path: Path, corpus: _PositiveCorpus
) -> None:
    actor = _construct(corpus, _private_spool(tmp_path))
    committed = _append(actor, corpus)
    alternative = _alternate_request_triplet(corpus)
    with pytest.raises(durable_anchor.Phase2BStep5R01DurableAnchorErrorV1) as caught:
        _append(
            actor,
            corpus,
            q1_request=alternative[0],
            q1_response=alternative[1],
            a_request=alternative[2],
        )
    assert caught.value.code == durable_anchor.FAIL_CONFLICT_V1
    replay = _append(actor, corpus)
    assert replay.idempotent_replay is True
    assert replay.payloads_in_contract_order == committed.payloads_in_contract_order


def test_threads_serialize_one_first_append_and_exact_replays(
    tmp_path: Path, corpus: _PositiveCorpus
) -> None:
    actor = _construct(corpus, _private_spool(tmp_path))
    count = 8
    barrier = threading.Barrier(count)

    def run() -> tuple[bool, tuple[str, ...]]:
        barrier.wait()
        result = _append(actor, corpus)
        return result.idempotent_replay, tuple(
            _sha(payload) for payload in result.payloads_in_contract_order
        )

    with ThreadPoolExecutor(max_workers=count) as executor:
        outcomes = list(executor.map(lambda _: run(), range(count)))
    flags = [outcome[0] for outcome in outcomes]
    assert flags.count(False) == 1
    assert flags.count(True) == count - 1
    assert len({outcome[1] for outcome in outcomes}) == 1


def test_forked_processes_serialize_with_directory_flock(
    tmp_path: Path, corpus: _PositiveCorpus
) -> None:
    actor = _construct(corpus, _private_spool(tmp_path))
    context = multiprocessing.get_context("fork")
    receivers: list[object] = []
    processes: list[object] = []
    for _ in range(3):
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(
            target=_fork_append_worker,
            args=(
                sender,
                actor,
                corpus.payloads["q1_request"],
                corpus.payloads["q1_response"],
                corpus.payloads["a_request"],
            ),
        )
        process.start()
        sender.close()
        receivers.append(receiver)
        processes.append(process)
    messages = []
    for receiver in receivers:
        assert receiver.poll(30)
        messages.append(receiver.recv())
        receiver.close()
    for process in processes:
        process.join(30)
        assert process.exitcode == 0
    assert all(message[0] == "ok" for message in messages), messages
    flags = [message[1] for message in messages]
    assert flags.count(False) == 1
    assert flags.count(True) == 2
    assert len({message[2] for message in messages}) == 1


@pytest.mark.parametrize(
    "poison",
    ["intent", "partial", "gap", "extra", "symlink", "hardlink", "mode"],
)
def test_committed_namespace_poison_fails_closed(
    tmp_path: Path, corpus: _PositiveCorpus, poison: str
) -> None:
    spool = _private_spool(tmp_path)
    actor = _construct(corpus, spool)
    result = _append(actor, corpus)
    names = result.spool_file_basenames_in_commit_order
    intent, q1_request, q1_response, a_request, checkpoint, receipt, head = names

    if poison == "intent":
        (spool / intent).write_bytes(b"{}")
        (spool / intent).chmod(0o600)
    elif poison == "partial":
        for name in names[1:]:
            (spool / name).unlink()
    elif poison == "gap":
        (spool / receipt).unlink()
    elif poison == "extra":
        extra = spool / "unexpected-v1-record.json"
        extra.write_bytes(b"{}")
        extra.chmod(0o600)
    elif poison == "symlink":
        target = tmp_path / "outside-record.json"
        target.write_bytes((spool / checkpoint).read_bytes())
        target.chmod(0o600)
        (spool / checkpoint).unlink()
        (spool / checkpoint).symlink_to(target)
    elif poison == "hardlink":
        os.link(spool / checkpoint, tmp_path / "outside-hardlink.json")
    elif poison == "mode":
        (spool / checkpoint).chmod(0o640)
    else:  # pragma: no cover - parametrization is exact.
        raise AssertionError(poison)

    with pytest.raises(durable_anchor.Phase2BStep5R01DurableAnchorErrorV1):
        _append(actor, corpus)


def test_actor_freezes_construction_inode_against_path_replacement(
    tmp_path: Path, corpus: _PositiveCorpus
) -> None:
    spool = _private_spool(tmp_path)
    actor = _construct(corpus, spool)
    committed = _append(actor, corpus)
    construction_identity = (spool.stat().st_dev, spool.stat().st_ino)

    moved = tmp_path / "committed-spool-moved-aside"
    spool.rename(moved)
    spool.mkdir(mode=0o700)
    spool.chmod(0o700)
    assert (spool.stat().st_dev, spool.stat().st_ino) != construction_identity

    with pytest.raises(durable_anchor.Phase2BStep5R01DurableAnchorErrorV1) as caught:
        _append(actor, corpus)
    assert caught.value.code == durable_anchor.FAIL_FILESYSTEM_V1
    assert list(spool.iterdir()) == []
    assert {path.name for path in moved.iterdir()} == set(
        committed.spool_file_basenames_in_commit_order
    )


def test_fsync_readback_and_head_last_are_observed_mechanics(
    tmp_path: Path,
    corpus: _PositiveCorpus,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor = _construct(corpus, _private_spool(tmp_path))
    captured_globals = durable_anchor.compare_and_append_local_durable_anchor_v1.__globals__
    original_publish: Callable[..., object] = captured_globals[
        "_publish_staged_no_replace"
    ]
    original_read_exact: Callable[..., object] = captured_globals["_read_exact_record"]
    real_fsync = os.fsync
    publications: list[str] = []
    exact_readbacks: list[str] = []
    fsync_kinds: list[str] = []

    def publish(directory_fd: int, basename: str, payload: bytes) -> object:
        publications.append(basename)
        return original_publish(directory_fd, basename, payload)

    def read_exact(directory_fd: int, basename: str, expected: bytes) -> object:
        exact_readbacks.append(basename)
        return original_read_exact(directory_fd, basename, expected)

    def fsync(fd: int) -> None:
        fsync_kinds.append("directory" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file")
        real_fsync(fd)

    monkeypatch.setitem(captured_globals, "_publish_staged_no_replace", publish)
    monkeypatch.setitem(captured_globals, "_read_exact_record", read_exact)
    monkeypatch.setattr(os, "fsync", fsync)
    result = _append(actor, corpus)

    names = result.spool_file_basenames_in_commit_order
    assert tuple(publications) == names[1:]
    assert publications[-1].startswith("a-head-v1-")
    assert set(names).issubset(set(exact_readbacks))
    assert fsync_kinds.count("file") >= 7
    assert fsync_kinds.count("directory") >= 20
    receipt = json.loads(result.a_compare_and_append_receipt_payload)
    assert receipt["body"]["durable_before_return"] is True


def test_fsync_failure_leaves_a_permanently_fail_closed_partial_slot(
    tmp_path: Path,
    corpus: _PositiveCorpus,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spool = _private_spool(tmp_path)
    actor = _construct(corpus, spool)

    def fail_fsync(fd: int) -> None:
        raise OSError("injected fsync failure")

    with monkeypatch.context() as scoped:
        scoped.setattr(os, "fsync", fail_fsync)
        with pytest.raises(durable_anchor.Phase2BStep5R01DurableAnchorErrorV1) as caught:
            _append(actor, corpus)
        assert caught.value.code == durable_anchor.FAIL_FILESYSTEM_V1
    assert list(spool.iterdir())
    with pytest.raises(durable_anchor.Phase2BStep5R01DurableAnchorErrorV1):
        _append(actor, corpus)


def test_actor_and_result_private_lineage_reject_forge_copy_and_mutation(
    tmp_path: Path, corpus: _PositiveCorpus
) -> None:
    actor = _construct(corpus, _private_spool(tmp_path))
    result = _append(actor, corpus)
    assert not any("private_key" in name for name in dir(actor))
    assert not any("private_key" in name for name in dir(result))
    with pytest.raises(TypeError):
        durable_anchor.LocalDurableAnchorActorV1()
    with pytest.raises(TypeError):
        durable_anchor.IssuedLocalDurableAnchorAppendCandidateV1()

    forged_actor = object.__new__(durable_anchor.LocalDurableAnchorActorV1)
    for field in actor.__dataclass_fields__:
        object.__setattr__(forged_actor, field, getattr(actor, field))
    with pytest.raises(durable_anchor.Phase2BStep5R01DurableAnchorErrorV1):
        durable_anchor.require_constructed_local_durable_anchor_actor_v1(forged_actor)

    forged_result = object.__new__(
        durable_anchor.IssuedLocalDurableAnchorAppendCandidateV1
    )
    for field in result.__dataclass_fields__:
        object.__setattr__(forged_result, field, getattr(result, field))
    with pytest.raises(durable_anchor.Phase2BStep5R01DurableAnchorErrorV1):
        durable_anchor.require_issued_local_durable_anchor_append_candidate_v1(
            forged_result
        )

    for value, require in (
        (actor, durable_anchor.require_constructed_local_durable_anchor_actor_v1),
        (
            result,
            durable_anchor.require_issued_local_durable_anchor_append_candidate_v1,
        ),
    ):
        try:
            copied = copy.copy(value)
        except TypeError:
            copied = None
        if copied is not None and copied is not value:
            with pytest.raises(durable_anchor.Phase2BStep5R01DurableAnchorErrorV1):
                require(copied)

    with pytest.raises(TypeError):
        result.artifact_ids["authority_established"] = "true"
    object.__setattr__(actor, "authority_established", True)
    with pytest.raises(durable_anchor.Phase2BStep5R01DurableAnchorErrorV1):
        durable_anchor.require_constructed_local_durable_anchor_actor_v1(actor)
    object.__setattr__(result, "authority_established", True)
    with pytest.raises(durable_anchor.Phase2BStep5R01DurableAnchorErrorV1):
        durable_anchor.require_issued_local_durable_anchor_append_candidate_v1(result)


def test_public_entry_points_ignore_module_and_transitive_global_rebinding(
    tmp_path: Path,
    corpus: _PositiveCorpus,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "_canonical",
        "_decode",
        "_verify_ed25519",
        "_contract_content_id",
        "_contract_signature_preimage",
        "_contract_canonical",
        "_load_policy_closure",
        "_load_request_closure",
        "_compose_a_outputs",
        "_publish_new_slot",
    ):
        monkeypatch.setattr(durable_anchor, name, _bomb)
    for name in (
        "canonical_external_witness_admission_json_bytes_v1",
        "phase2b_step5_external_witness_admission_content_id_v1",
        "phase2b_step5_external_witness_admission_signature_preimage_v1",
    ):
        monkeypatch.setattr(admission, name, _bomb)

    actor = _construct(corpus, _private_spool(tmp_path))
    result = _append(actor, corpus)
    assert result.payloads_in_contract_order == (
        corpus.payloads["a_checkpoint"],
        corpus.payloads["a_head"],
        corpus.payloads["a_receipt"],
    )


def test_clean_import_blocks_producers_stores_runtime_and_services() -> None:
    source_root = str(Path(durable_anchor.__file__).resolve().parents[1])
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        blocked = {
            "hegel_machine.phase2b_formal_wire_covert_step5_logical_time_v1",
            "hegel_machine.phase2b_formal_wire_covert_step5_r01_current_cut_v1",
            "hegel_machine.phase2b_formal_wire_covert_step5_r01_external_witness_v1",
            "hegel_machine.phase2b_formal_wire_covert_step5_r01_online_witness_v1",
            "hegel_machine.phase2b_formal_wire_covert_step5_owner_status_v1",
            "hegel_machine.phase2b_formal_wire_covert_step5_runtime_v1",
            "hegel_machine.phase2b_formal_wire_covert_step5_service_v1",
            "hegel_machine.phase2b_formal_wire_covert_step5_store_v1",
            "hegel_machine.phase2b_formal_wire_covert_step5_store_v2",
        }

        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in blocked:
                    raise RuntimeError("blocked mutable dependency import: " + fullname)
                return None

        sys.meta_path.insert(0, Blocker())
        import hegel_machine.phase2b_formal_wire_covert_step5_r01_durable_anchor_v1
        assert blocked.isdisjoint(sys.modules)
        print("CLEAN_IMPORT_OK")
        """
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = source_root
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "CLEAN_IMPORT_OK"


def test_all_elevated_claims_remain_false_and_real_admission_is_not_implemented(
    tmp_path: Path, corpus: _PositiveCorpus
) -> None:
    actor = _construct(corpus, _private_spool(tmp_path))
    result = _append(actor, corpus)
    for field in (
        "key_custody_verified",
        "external_durability_verified",
        "host_rollback_resistance_verified",
        "non_equivocation_verified",
        "live_service_verified",
        "external_currentness_verified",
        "logical_time_ancestry_verified",
        "authority_established",
        "r01_satisfied",
        "readiness_satisfied",
        "execution_authorized",
    ):
        assert getattr(actor, field) is False
    for field in (
        "key_custody_verified",
        "external_durability_verified",
        "host_rollback_resistance_verified",
        "non_equivocation_verified",
        "live_service_verified",
        "external_currentness_verified",
        "global_anti_fork_verified",
        "logical_time_ancestry_verified",
        "authority_established",
        "r01_satisfied",
        "readiness_satisfied",
        "execution_authorized",
    ):
        assert getattr(result, field) is False
    assert "NOT_EXTERNAL_DURABILITY" in result.mechanics_claim_boundary
    assert "NOT_AUTHORITY" in result.mechanics_claim_boundary
    with pytest.raises(durable_anchor.Phase2BStep5R01DurableAnchorErrorV1) as caught:
        durable_anchor.verify_real_external_witness_admission_v1()
    assert caught.value.code == durable_anchor.FAIL_NOT_IMPLEMENTED_V1
