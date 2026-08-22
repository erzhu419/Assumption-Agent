from __future__ import annotations

import ast
import copy
from dataclasses import dataclass, fields, replace
import hashlib
import inspect
import json
import os
from pathlib import Path
import socket
import stat
import struct
import subprocess
import sys
import threading
from types import SimpleNamespace
from typing import Callable

import pytest

from hegel_machine import phase2b_formal_wire_covert_step5_authority_v2 as authority
from hegel_machine import phase2b_formal_wire_covert_step5_logical_time_replay_core_v1 as time_core
from hegel_machine import phase2b_formal_wire_covert_step5_owner_status_v1 as owner_status
from hegel_machine import phase2b_formal_wire_covert_step5_r01_current_cut_v1 as current_cut
from hegel_machine import phase2b_formal_wire_covert_step5_r01_evidence_contract_v1 as evidence
from hegel_machine.rfc8785_v1 import canonicalize_rfc8785_v1

# This producer fixture is used only to make real authority-v2, owner-ledger,
# and logical-time wire bytes.  The current-cut module never imports it.
import test_phase2b_formal_wire_covert_step5_r01_pure_replay_v1 as producer_fixture


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    PROJECT_ROOT
    / "src"
    / "hegel_machine"
    / "phase2b_formal_wire_covert_step5_r01_current_cut_v1.py"
)
REQUEST_ID_DOMAIN = b"HEGEL/PHASE2B/STEP5/R01/CURRENT_CUT_REQUEST/V1\0"


@dataclass(frozen=True)
class Corpus:
    root: Path
    graph: object
    active: object
    revoked: object
    append: owner_status.OwnerStatusServiceAppendV1


def _json(payload: bytes) -> dict[str, object]:
    value = json.loads(payload)
    assert type(value) is dict
    return value


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _address(payload: bytes) -> str:
    return "sha256:" + _sha(payload)


def _public_implementation(public_entry: Callable[..., object]) -> Callable[..., object]:
    assert public_entry.__closure__ is not None
    functions = tuple(
        cell.cell_contents
        for cell in public_entry.__closure__
        if inspect.isfunction(cell.cell_contents)
    )
    assert len(functions) == 1
    return functions[0]


def _private_namespace() -> dict[str, object]:
    request_impl = _public_implementation(
        current_cut.request_current_cut_over_seqpacket_v1
    )
    process_impl = _public_implementation(
        current_cut.process_one_current_cut_request_v1
    )
    build_impl = _public_implementation(current_cut.build_current_cut_request_v1)
    assert request_impl.__globals__ is process_impl.__globals__
    assert request_impl.__globals__ is build_impl.__globals__
    assert request_impl.__globals__ is not current_cut.__dict__
    return request_impl.__globals__


def _expect(code: str, operation: Callable[[], object]) -> None:
    with pytest.raises(current_cut.Phase2BStep5R01CurrentCutErrorV1) as caught:
        operation()
    assert caught.value.code == code


def _make_append(chain: object) -> owner_status.OwnerStatusServiceAppendV1:
    decision = _json(chain.decisions[0])  # type: ignore[attr-defined]
    event = _json(chain.events[0])  # type: ignore[attr-defined]
    checkpoint = _json(chain.owner_checkpoints[0])  # type: ignore[attr-defined]
    head = _json(chain.owner_heads[1])  # type: ignore[attr-defined]
    return owner_status.OwnerStatusServiceAppendV1(
        decision_id=decision["decision_id"],  # type: ignore[arg-type]
        event_id=event["event_id"],  # type: ignore[arg-type]
        checkpoint_id=checkpoint["checkpoint_id"],  # type: ignore[arg-type]
        head_id=head["head_id"],  # type: ignore[arg-type]
        complete_event_set_root=head["complete_event_set_root"],  # type: ignore[arg-type]
        signed_event_payload=chain.events[0],  # type: ignore[attr-defined]
        checkpoint_payload=chain.owner_checkpoints[0],  # type: ignore[attr-defined]
        head_payload=chain.owner_heads[1],  # type: ignore[attr-defined]
        projection_payload=chain.projection,  # type: ignore[attr-defined]
    )


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Corpus:
    root = tmp_path_factory.mktemp("r01-current-cut-real-wires")
    graph = producer_fixture._build_authority_and_five_runtime_graphs(salt=0)
    active_root = root / "active"
    revoked_root = root / "revoked"
    active_root.mkdir()
    revoked_root.mkdir()
    active = producer_fixture._build_owner_chain(active_root, graph, 1)
    revoked = producer_fixture._build_owner_chain(revoked_root, graph, 2)
    return Corpus(root, graph, active, revoked, _make_append(active))


def _authority_arguments(corpus: Corpus) -> dict[str, object]:
    graph = corpus.graph
    return {
        "stage_b_trust_lock_payload": graph.lock,
        "authority_profile_payload": graph.profile,
        "signer_registry_payload": graph.registry,
        "out_of_band_expected_lock_id_bytes": graph.oob_id,
        "out_of_band_expected_payload_sha256_digest": graph.oob_digest,
        "expected_attempt_context_id": corpus.active.attempt,
    }


def _context(corpus: Corpus) -> tuple[object, ...]:
    graph = corpus.graph
    loader = _private_namespace()["_load_authority_context"]
    assert callable(loader)
    return loader(
        graph.lock,
        graph.profile,
        graph.registry,
        out_of_band_expected_lock_id_bytes=graph.oob_id,
        out_of_band_expected_payload_sha256_digest=graph.oob_digest,
        expected_attempt_context_id=corpus.active.attempt,
    )


def _request_payload(corpus: Corpus, nonce: bytes = b"Q" * 32) -> bytes:
    graph = corpus.graph
    return current_cut.build_current_cut_request_v1(
        graph.lock,
        graph.profile,
        graph.registry,
        out_of_band_expected_lock_id_bytes=graph.oob_id,
        out_of_band_expected_payload_sha256_digest=graph.oob_digest,
        expected_attempt_context_id=corpus.active.attempt,
        cut_request_nonce_bytes=nonce,
    )


def _wire_material(corpus: Corpus) -> tuple[tuple[object, ...], bytes, dict[str, object], bytes, dict[str, object]]:
    namespace = _private_namespace()
    context = _context(corpus)
    request_payload = _request_payload(corpus)
    request = namespace["_load_request"](request_payload, context)  # type: ignore[operator]
    head_payload, projection_payload, head, projection, stable = namespace[
        "_owner_snapshot"
    ](corpus.append, context)  # type: ignore[operator]
    assert stable is False
    response_payload = namespace["_build_response"](  # type: ignore[operator]
        request,
        context,
        head_payload,
        projection_payload,
        head,
        projection,
    )
    response = namespace["_load_response"](  # type: ignore[operator]
        response_payload, request, context
    )
    return context, request_payload, request, response_payload, response


def _instrument_peer_credentials(
    monkeypatch: pytest.MonkeyPatch,
    client: socket.socket,
    server: socket.socket,
) -> None:
    namespace = _private_namespace()

    def peer(held: socket.socket) -> tuple[int, int, int]:
        if held.fileno() == client.fileno():
            return (31005, 41005, 42005)
        if held.fileno() == server.fileno():
            return (31001, 41001, 42001)
        raise AssertionError("unexpected test socket")

    monkeypatch.setitem(namespace, "_peer_credentials", peer)


def _roundtrip(
    corpus: Corpus, monkeypatch: pytest.MonkeyPatch
) -> current_cut.VerifiedR01CurrentCutMechanicsV1:
    graph = corpus.graph
    client, server = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    _instrument_peer_credentials(monkeypatch, client, server)
    server_failures: list[BaseException] = []

    def serve() -> None:
        try:
            current_cut.process_one_current_cut_request_v1(
                server,
                corpus.append,
                graph.lock,
                graph.profile,
                graph.registry,
                out_of_band_expected_lock_id_bytes=graph.oob_id,
                out_of_band_expected_payload_sha256_digest=graph.oob_digest,
                expected_attempt_context_id=corpus.active.attempt,
            )
        except BaseException as exc:  # pragma: no branch - asserted below
            server_failures.append(exc)

    thread = threading.Thread(target=serve, name="candidate-current-cut-test-server")
    thread.start()
    try:
        result = current_cut.request_current_cut_over_seqpacket_v1(
            client,
            graph.lock,
            graph.profile,
            graph.registry,
            corpus.active.time_requests,
            corpus.active.time_tokens,
            corpus.active.time_checkpoints,
            corpus.active.time_heads,
            out_of_band_expected_lock_id_bytes=graph.oob_id,
            out_of_band_expected_payload_sha256_digest=graph.oob_digest,
            expected_attempt_context_id=corpus.active.attempt,
            expected_logical_time_event_plan=corpus.active.event_plan,
            cut_request_nonce_bytes=b"Q" * 32,
        )
    finally:
        thread.join(timeout=10)
        client.close()
        server.close()
    assert not thread.is_alive()
    assert server_failures == []
    return result


def test_source_review_pins_import_surface_and_clean_compile() -> None:
    assert _sha((PROJECT_ROOT / "src/hegel_machine/phase2b_formal_wire_covert_step5_authority_v2.py").read_bytes()) == current_cut.AUTHORITY_V2_SOURCE_SHA256_V1
    assert _sha((PROJECT_ROOT / "src/hegel_machine/phase2b_formal_wire_covert_step5_owner_status_v1.py").read_bytes()) == current_cut.OWNER_STATUS_SOURCE_SHA256_V1
    assert _sha((PROJECT_ROOT / "src/hegel_machine/phase2b_formal_wire_covert_step5_logical_time_replay_core_v1.py").read_bytes()) == current_cut.LOGICAL_TIME_REPLAY_CORE_SOURCE_SHA256_V1
    assert _sha((PROJECT_ROOT / "src/hegel_machine/phase2b_formal_wire_covert_step5_r01_evidence_contract_v1.py").read_bytes()) == current_cut.R01_EVIDENCE_CONTRACT_SOURCE_SHA256_V1
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    imported = {
        module_name
        for node in ast.walk(tree)
        for module_name in (
            tuple(alias.name for alias in node.names)
            if isinstance(node, ast.Import)
            else ((node.module or ""),)
            if isinstance(node, ast.ImportFrom)
            else ()
        )
    }
    assert not any(
        marker in name
        for name in imported
        for marker in ("store_v", "runtime_v", "runtime_closure_v", "r01_pure_replay")
    )
    subprocess.run(
        [sys.executable, "-m", "py_compile", str(SOURCE)],
        check=True,
        env={**os.environ, "PYTHONPYCACHEPREFIX": "/tmp"},
    )


def test_public_request_nonce_domain_id_and_raw_authority_binding(corpus: Corpus) -> None:
    payload = _request_payload(corpus, b"N" * 32)
    body = _json(payload)
    assert frozenset(body) == current_cut.CURRENT_CUT_REQUEST_FIELDS_V1
    assert body["cut_request_nonce"] == (b"N" * 32).hex()
    subject = {key: value for key, value in body.items() if key != "request_id"}
    expected = "phase2b_step5_current_cut_request_v1_" + hashlib.sha256(
        REQUEST_ID_DOMAIN + canonicalize_rfc8785_v1(subject)
    ).hexdigest()
    assert body["request_id"] == expected
    assert body["status"] == evidence.CANDIDATE_STATUS_V1
    assert "NOT_CURRENT" in body["claim_level"]
    assert "NOT_AUTHORITY" in body["claim_level"]
    for nonce in (b"", b"x" * 31, b"x" * 33, b"\0" * 32, bytearray(b"x" * 32)):
        _expect(
            current_cut.FAIL_INPUT_V1,
            lambda nonce=nonce: _request_payload(corpus, nonce),  # type: ignore[arg-type]
        )


def test_request_loader_rejects_nonce_or_domain_readdressing(corpus: Corpus) -> None:
    namespace = _private_namespace()
    context = _context(corpus)
    valid = _json(_request_payload(corpus))
    changed_nonce = {**valid, "cut_request_nonce": "7" * 64}
    _expect(
        current_cut.FAIL_CONTENT_ID_V1,
        lambda: namespace["_load_request"](  # type: ignore[operator]
            canonicalize_rfc8785_v1(changed_nonce), context
        ),
    )
    wrong_domain = dict(valid)
    subject = {key: value for key, value in wrong_domain.items() if key != "request_id"}
    wrong_domain["request_id"] = "phase2b_step5_current_cut_request_v1_" + hashlib.sha256(
        b"EVIL/DOMAIN\0" + canonicalize_rfc8785_v1(subject)
    ).hexdigest()
    _expect(
        current_cut.FAIL_CONTENT_ID_V1,
        lambda: namespace["_load_request"](  # type: ignore[operator]
            canonicalize_rfc8785_v1(wrong_domain), context
        ),
    )


def test_exact_append_producer_response_bytes_and_no_elevated_claim(
    corpus: Corpus, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = corpus.graph
    request_payload = _request_payload(corpus)
    client, server = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    namespace = _private_namespace()
    monkeypatch.setitem(namespace, "_peer_credentials", lambda _held: (31001, 41001, 42001))
    try:
        assert client.send(request_payload) == len(request_payload)
        returned = current_cut.process_one_current_cut_request_v1(
            server,
            corpus.append,
            graph.lock,
            graph.profile,
            graph.registry,
            out_of_band_expected_lock_id_bytes=graph.oob_id,
            out_of_band_expected_payload_sha256_digest=graph.oob_digest,
            expected_attempt_context_id=corpus.active.attempt,
        )
        received = client.recv(8 * 1024 * 1024)
    finally:
        client.close()
        server.close()
    assert returned == received
    body = _json(returned)
    assert body["current_head_id"] == corpus.append.head_id
    assert body["current_head_payload_sha256"] == _address(corpus.append.head_payload)
    assert body["projection_payload_sha256"] == _address(corpus.append.projection_payload)
    assert body["complete_event_set_root"] == corpus.append.complete_event_set_root
    assert body["event_count"] == 1
    assert body["latest_effective_status"] == "ACTIVE"
    assert body["revocation_count"] == 0
    assert body["terminal"] is False
    assert body["status"] == evidence.CANDIDATE_STATUS_V1


def _open_active_owner_ledger(corpus: Corpus) -> owner_status.OwnerStatusLedgerV1:
    graph = corpus.graph
    governance = graph.role_rows[authority.ROLE_ORDER_V2.index("GOVERNANCE_ANCHOR")]
    logical = graph.role_rows[authority.ROLE_ORDER_V2.index("LOGICAL_TIME")]
    return owner_status.OwnerStatusLedgerV1(
        str(corpus.root / "active" / "owner"),
        authority_profile_id=graph.profile_id,
        signer_registry_id=graph.registry_id,
        governance_key_id=governance["key_id"],
        governance_key_epoch=1,
        attempt_context_id=corpus.active.attempt,
        governance_public_key=bytes.fromhex(governance["public_key_hex"]),
        logical_time_public_key=bytes.fromhex(logical["public_key_hex"]),
        owner_authorization_payload=corpus.active.owner,
        initialize_empty=False,
    )


def test_exact_ledger_head_projection_head_three_replays(corpus: Corpus) -> None:
    owner_snapshot = _private_namespace()["_owner_snapshot"]
    with _open_active_owner_ledger(corpus) as ledger:
        head_payload, projection_payload, head, projection, stable = owner_snapshot(  # type: ignore[operator]
            ledger, _context(corpus)
        )
    assert stable is True
    assert head_payload == corpus.append.head_payload
    assert projection_payload == corpus.append.projection_payload
    assert head.head_id == corpus.append.head_id
    assert projection.head_id == corpus.append.head_id


def test_three_replay_append_injection_and_projection_forgery_fail_closed(
    corpus: Corpus, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = _private_namespace()
    owner_snapshot = namespace["_owner_snapshot"]
    original_defaults = owner_snapshot.__defaults__  # type: ignore[attr-defined]
    assert original_defaults is not None and len(original_defaults) == 3
    original_head, original_projection, append_post_init = original_defaults
    calls = 0

    def injected_head(ledger: owner_status.OwnerStatusLedgerV1) -> bytes:
        nonlocal calls
        calls += 1
        payload = original_head(ledger)
        return payload if calls == 1 else payload + b" "

    monkeypatch.setattr(
        owner_snapshot,
        "__defaults__",
        (injected_head, original_projection, append_post_init),
    )
    with _open_active_owner_ledger(corpus) as ledger:
        _expect(
            current_cut.FAIL_OWNER_SNAPSHOT_V1,
            lambda: owner_snapshot(ledger, _context(corpus)),
        )

    forged = object.__new__(owner_status.OwnerStatusServiceAppendV1)
    for field in fields(owner_status.OwnerStatusServiceAppendV1):
        value = getattr(corpus.append, field.name)
        if field.name == "projection_payload":
            value = b"{}"
        object.__setattr__(forged, field.name, value)
    monkeypatch.setattr(owner_snapshot, "__defaults__", original_defaults)
    _expect(
        current_cut.FAIL_OWNER_SNAPSHOT_V1,
        lambda: owner_snapshot(forged, _context(corpus)),
    )


@pytest.mark.parametrize(
    "packet",
    [
        (b"x", [(socket.SOL_SOCKET, socket.SCM_RIGHTS, b"bad")], 0, None),
        (b"x", [], socket.MSG_TRUNC, None),
        (b"x", [], getattr(socket, "MSG_CTRUNC", socket.MSG_TRUNC), None),
        (b"x", [], 0, b"/tmp/untrusted-source"),
    ],
    ids=("ancillary", "trunc", "ctrunc", "source-address"),
)
def test_recvmsg_rejects_ancillary_truncation_and_source_address(packet: tuple[object, ...]) -> None:
    receiver = SimpleNamespace(recvmsg=lambda _maximum: packet)
    recv_packet = _private_namespace()["_recv_packet"]
    _expect(
        current_cut.FAIL_SOCKET_V1,
        lambda: recv_packet(receiver, 16, "redteam packet"),  # type: ignore[operator]
    )


def test_real_seqpacket_truncation_socket_identity_and_short_send() -> None:
    namespace = _private_namespace()
    recv_packet = namespace["_recv_packet"]
    socket_identity = namespace["_socket_identity"]
    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    try:
        left.send(b"x" * 18)
        _expect(
            current_cut.FAIL_SOCKET_V1,
            lambda: recv_packet(right, 16, "oversize packet"),  # type: ignore[operator]
        )
        identity = socket_identity(right)  # type: ignore[operator]
        observed = os.fstat(right.fileno())
        assert identity[:2] == (observed.st_dev, observed.st_ino)
        assert identity[3] == right.fileno()
        assert stat.S_ISSOCK(identity[2])
    finally:
        left.close()
        right.close()

    sender = SimpleNamespace(send=lambda payload: len(payload) - 1)
    _expect(
        current_cut.FAIL_SOCKET_V1,
        lambda: namespace["_send_packet"](sender, b"payload", "short"),  # type: ignore[operator]
    )


def test_socket_boundary_rejects_stream_and_unconnected_seqpacket() -> None:
    require_seqpacket = _private_namespace()["_require_seqpacket"]
    stream_left, stream_right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    unconnected = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    try:
        _expect(
            current_cut.FAIL_SOCKET_V1,
            lambda: require_seqpacket(stream_left),  # type: ignore[operator]
        )
        _expect(
            current_cut.FAIL_SOCKET_V1,
            lambda: require_seqpacket(unconnected),  # type: ignore[operator]
        )
    finally:
        stream_left.close()
        stream_right.close()
        unconnected.close()


def test_production_so_peercred_is_kernel_read_and_static_mismatch_rejected(
    corpus: Corpus,
) -> None:
    graph = corpus.graph
    request_payload = _request_payload(corpus)
    client, server = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    raw = server.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
    pid, uid, gid = struct.unpack("3i", raw)
    assert pid >= 1 and uid >= 0 and gid >= 0
    if (uid, gid) == (41001, 42001):
        pytest.skip("test process genuinely has the frozen runner credentials")
    try:
        client.send(request_payload)
        _expect(
            current_cut.FAIL_PEER_CREDENTIAL_V1,
            lambda: current_cut.process_one_current_cut_request_v1(
                server,
                corpus.append,
                graph.lock,
                graph.profile,
                graph.registry,
                out_of_band_expected_lock_id_bytes=graph.oob_id,
                out_of_band_expected_payload_sha256_digest=graph.oob_digest,
                expected_attempt_context_id=corpus.active.attempt,
            ),
        )
    finally:
        client.close()
        server.close()


@pytest.mark.parametrize("changed_read", [2, 3])
def test_peercred_before_after_receive_and_send_cannot_change(
    corpus: Corpus,
    monkeypatch: pytest.MonkeyPatch,
    changed_read: int,
) -> None:
    namespace = _private_namespace()
    graph = corpus.graph
    calls = 0

    def peer(_held: socket.socket) -> tuple[int, int, int]:
        nonlocal calls
        calls += 1
        return (
            (31001, 41001, 42001)
            if calls != changed_read
            else (39999, 41001, 42001)
        )

    monkeypatch.setitem(namespace, "_peer_credentials", peer)
    client, server = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    try:
        client.send(_request_payload(corpus))
        _expect(
            current_cut.FAIL_SOCKET_V1,
            lambda: current_cut.process_one_current_cut_request_v1(
                server,
                corpus.append,
                graph.lock,
                graph.profile,
                graph.registry,
                out_of_band_expected_lock_id_bytes=graph.oob_id,
                out_of_band_expected_payload_sha256_digest=graph.oob_digest,
                expected_attempt_context_id=corpus.active.attempt,
            ),
        )
    finally:
        client.close()
        server.close()


def test_held_socket_inode_identity_cannot_change(
    corpus: Corpus, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = _private_namespace()
    original_identity = namespace["_socket_identity"]
    calls = 0

    def identity(held: socket.socket) -> tuple[int, int, int, int]:
        nonlocal calls
        calls += 1
        value = original_identity(held)
        return value if calls == 1 else (value[0], value[1] + 1, value[2], value[3])

    monkeypatch.setitem(namespace, "_peer_credentials", lambda _held: (31001, 41001, 42001))
    monkeypatch.setitem(namespace, "_socket_identity", identity)
    graph = corpus.graph
    client, server = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    try:
        client.send(_request_payload(corpus))
        _expect(
            current_cut.FAIL_SOCKET_V1,
            lambda: current_cut.process_one_current_cut_request_v1(
                server,
                corpus.append,
                graph.lock,
                graph.profile,
                graph.registry,
                out_of_band_expected_lock_id_bytes=graph.oob_id,
                out_of_band_expected_payload_sha256_digest=graph.oob_digest,
                expected_attempt_context_id=corpus.active.attempt,
            ),
        )
    finally:
        client.close()
        server.close()


def _verify_time(
    corpus: Corpus,
    *,
    requests: object | None = None,
    tokens: object | None = None,
    checkpoints: object | None = None,
    heads: object | None = None,
    plan: object | None = None,
    oob_digest: object | None = None,
) -> tuple[dict[str, object], object]:
    context, _request_payload_bytes, _request, _response_payload, response = _wire_material(corpus)
    chain = corpus.active
    return _private_namespace()["_verify_pre_cut_time"](  # type: ignore[operator]
        context,
        response,
        corpus.graph.lock,
        corpus.graph.profile,
        corpus.graph.registry,
        chain.time_requests if requests is None else requests,
        chain.time_tokens if tokens is None else tokens,
        chain.time_checkpoints if checkpoints is None else checkpoints,
        chain.time_heads if heads is None else heads,
        out_of_band_expected_lock_id_bytes=corpus.graph.oob_id,
        out_of_band_expected_payload_sha256_digest=(
            corpus.graph.oob_digest if oob_digest is None else oob_digest
        ),
        expected_logical_time_event_plan=(
            chain.event_plan if plan is None else plan
        ),
    )


def test_raw_non_genesis_logical_time_lineage_and_exact_count(corpus: Corpus) -> None:
    binding, result = _verify_time(corpus)
    token = _json(corpus.active.time_tokens[-1])
    head = _json(corpus.active.time_heads[-1])
    assert result.verified_event_count == 1
    assert result.zero_event_genesis_only is False
    assert binding == {
        "ledger_id": result.ledger_id,
        "final_head_id": head["head_id"],
        "final_head_payload_sha256": _address(corpus.active.time_heads[-1]),
        "last_token_id": token["token_id"],
        "next_sequence": 1,
    }
    assert result.external_currentness_verified is False
    assert result.external_anti_fork_verified is False
    assert result.authority_evidence is False
    assert result.r01_satisfied is False
    assert result.readiness_evidence is False
    assert result.execution_authorized is False


def test_logical_time_rejects_genesis_two_event_bad_plan_raw_and_oob(
    corpus: Corpus,
) -> None:
    _expect(
        current_cut.FAIL_LOGICAL_TIME_V1,
        lambda: _verify_time(
            corpus,
            requests=(),
            tokens=(),
            checkpoints=(),
            heads=(corpus.active.time_heads[0],),
            plan=(),
        ),
    )
    _expect(
        current_cut.FAIL_LOGICAL_TIME_V1,
        lambda: _verify_time(
            corpus,
            requests=corpus.revoked.time_requests,
            tokens=corpus.revoked.time_tokens,
            checkpoints=corpus.revoked.time_checkpoints,
            heads=corpus.revoked.time_heads,
            plan=corpus.revoked.event_plan,
        ),
    )
    wrong_plan = (
        replace(corpus.active.event_plan[0], request_nonce="f" * 64),
    )
    _expect(
        current_cut.FAIL_LOGICAL_TIME_V1,
        lambda: _verify_time(corpus, plan=wrong_plan),
    )
    raw_request = _json(corpus.active.time_requests[0])
    raw_request["request_nonce"] = "e" * 64
    _expect(
        current_cut.FAIL_LOGICAL_TIME_V1,
        lambda: _verify_time(
            corpus,
            requests=(canonicalize_rfc8785_v1(raw_request),),
        ),
    )
    _expect(
        current_cut.FAIL_LOGICAL_TIME_V1,
        lambda: _verify_time(corpus, oob_digest=b"\0" * 32),
    )


def test_public_surface_has_no_decoded_bypass_or_uid_override(corpus: Corpus) -> None:
    for public in (
        current_cut.build_current_cut_request_v1,
        current_cut.process_one_current_cut_request_v1,
        current_cut.request_current_cut_over_seqpacket_v1,
    ):
        names = frozenset(inspect.signature(public).parameters)
        assert not names.intersection(
            {
                "decoded_authority",
                "authority_context",
                "owner_projection",
                "logical_time_replay_result",
                "expected_peer_uid",
                "expected_peer_gid",
                "peer_credentials",
            }
        )
    graph = corpus.graph
    with pytest.raises(TypeError):
        current_cut.build_current_cut_request_v1(
            graph.lock,
            graph.profile,
            graph.registry,
            out_of_band_expected_lock_id_bytes=graph.oob_id,
            out_of_band_expected_payload_sha256_digest=graph.oob_digest,
            expected_attempt_context_id=corpus.active.attempt,
            cut_request_nonce_bytes=b"Q" * 32,
            decoded_authority={},  # type: ignore[call-arg]
        )
    with pytest.raises(TypeError):
        current_cut.process_one_current_cut_request_v1(
            object(),
            corpus.append,
            graph.lock,
            graph.profile,
            graph.registry,
            out_of_band_expected_lock_id_bytes=graph.oob_id,
            out_of_band_expected_payload_sha256_digest=graph.oob_digest,
            expected_attempt_context_id=corpus.active.attempt,
            expected_peer_uid=41001,  # type: ignore[call-arg]
        )


def test_instrumented_candidate_roundtrip_result_is_private_and_all_elevated_claims_false(
    corpus: Corpus, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _roundtrip(corpus, monkeypatch)
    assert current_cut.require_verified_r01_current_cut_mechanics_v1(result) is result
    assert result.raw_authority_v2_graph_verified is True
    assert result.kernel_so_peercred_verified is True
    assert result.seqpacket_request_response_verified is True
    assert result.owner_projection_candidate_presented is True
    assert result.active_zero_revocation_candidate is True
    assert result.pre_cut_logical_time_supplied_chain_verified is True
    assert result.owner_time_event_count_cross_checked is True
    assert result.stable_during_three_replays is False
    assert result.exact_producer_append_projection_presented is False
    false_fields = (
        "owner_time_raw_decision_binding_verified",
        "response_time_currentness_verified",
        "admission_time_currentness_verified",
        "atomic_owner_time_fence_verified",
        "external_currentness_verified",
        "external_anti_fork_anchor_verified",
        "live_runtime_closure_verified",
        "live_distinct_service_uids_gids_verified",
        "live_distinct_control_domains_verified",
        "service_key_custody_verified",
        "cas_durability_verified",
        "owner_signature_authenticated",
        "authority_evidence",
        "r01_satisfied",
        "readiness_evidence",
        "execution_authorized",
    )
    assert all(getattr(result, name) is False for name in false_fields)
    statement = _json(result.statement_payload)
    assert statement["claim_level"] == evidence.CANDIDATE_CLAIM_LEVEL_V1
    assert statement["owner_status_binding"]["event_count"] == 1  # type: ignore[index]
    assert statement["pre_cut_logical_time_binding"]["next_sequence"] == 1  # type: ignore[index]


def test_result_forge_copy_and_mutation_are_rejected(
    corpus: Corpus, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _roundtrip(corpus, monkeypatch)
    result_type = type(result)
    with pytest.raises(TypeError):
        result_type()
    forged = object.__new__(result_type)
    for field in fields(result):
        object.__setattr__(forged, field.name, getattr(result, field.name))
    _expect(
        current_cut.FAIL_RESULT_V1,
        lambda: current_cut.require_verified_r01_current_cut_mechanics_v1(forged),
    )
    for copier in (copy.copy, copy.deepcopy):
        try:
            copied = copier(result)
        except TypeError:
            continue
        if copied is result:
            assert current_cut.require_verified_r01_current_cut_mechanics_v1(copied) is result
        else:
            _expect(
                current_cut.FAIL_RESULT_V1,
                lambda copied=copied: current_cut.require_verified_r01_current_cut_mechanics_v1(copied),
            )
    object.__setattr__(result, "authority_evidence", True)
    _expect(
        current_cut.FAIL_RESULT_V1,
        lambda: current_cut.require_verified_r01_current_cut_mechanics_v1(result),
    )
    object.__setattr__(result, "authority_evidence", False)
    object.__setattr__(result, "request_payload", result.request_payload + b" ")
    _expect(
        current_cut.FAIL_RESULT_V1,
        lambda: current_cut.require_verified_r01_current_cut_mechanics_v1(result),
    )


def test_definition_cut_survives_module_defaults_and_transitive_entrypoint_evil(
    corpus: Corpus, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_result_type = current_cut.VerifiedR01CurrentCutMechanicsV1

    def evil(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("EVIL rebound helper executed")

    for name in (
        "_fail",
        "_load_stage_b_lock",
        "_load_authority_profile",
        "_load_signer_registry",
        "_require_authority_graph",
        "_verify_time_replay",
        "_require_time_replay",
        "_evidence_canonical",
        "_evidence_content_id",
        "_payload_sha_hex",
    ):
        monkeypatch.setattr(current_cut, name, evil)
    monkeypatch.setattr(current_cut, "CURRENT_CUT_MECHANICS_RESULT_SCHEMA_V1", "EVIL")
    monkeypatch.setattr(current_cut, "CURRENT_CUT_MECHANICS_CLAIM_BOUNDARY_V1", "EVIL")
    monkeypatch.setattr(current_cut, "FAIL_RESULT_V1", "EVIL")
    monkeypatch.setattr(current_cut, "Phase2BStep5R01CurrentCutErrorV1", RuntimeError)

    monkeypatch.setattr(authority, "load_step5_stage_b_trust_lock_v2", evil)
    monkeypatch.setattr(authority, "load_step5_authority_profile_v2", evil)
    monkeypatch.setattr(authority, "load_step5_signer_registry_v2", evil)
    monkeypatch.setattr(authority, "require_locked_step5_authority_graph_v2", evil)
    monkeypatch.setattr(time_core, "verify_logical_time_replay_core_v1", evil)
    monkeypatch.setattr(time_core, "require_verified_logical_time_replay_core_v1", evil)
    monkeypatch.setattr(evidence, "canonical_r01_evidence_contract_json_bytes_v1", evil)
    monkeypatch.setattr(evidence, "phase2b_step5_r01_evidence_content_id_v1", evil)

    for function_name in ("_canonical", "_load_authority_context", "_build_response", "_verify_pre_cut_time"):
        function = getattr(current_cut, function_name)
        if function.__defaults__ is not None:
            monkeypatch.setattr(
                function,
                "__defaults__",
                tuple(evil for _ in function.__defaults__),
            )
        if function.__kwdefaults__ is not None:
            monkeypatch.setattr(
                function,
                "__kwdefaults__",
                {key: evil for key in function.__kwdefaults__},
            )

    result = _roundtrip(corpus, monkeypatch)
    assert type(result) is original_result_type
    assert result.schema_version == "hegel-machine-phase2b-formal-wire-covert-step5-current-cut-mechanics-result/1"
    assert result.authority_evidence is False
    assert result.r01_satisfied is False
    assert result.readiness_evidence is False
    assert result.execution_authorized is False


def test_real_admission_is_exact_not_implemented_even_after_global_rebinding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_error = current_cut.Phase2BStep5R01CurrentCutErrorV1
    monkeypatch.setattr(current_cut, "_fail", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(current_cut, "FAIL_NOT_IMPLEMENTED_V1", "EVIL")
    monkeypatch.setattr(current_cut, "Phase2BStep5R01CurrentCutErrorV1", RuntimeError)
    with pytest.raises(original_error) as caught:
        current_cut.verify_real_current_cut_admission_v1()
    assert caught.value.code == "FAIL_STEP5_R01_CURRENT_CUT_V1_NOT_IMPLEMENTED"


def test_clean_import_has_no_listener_filesystem_or_process_side_effect(tmp_path: Path) -> None:
    script = r'''
import json
import sys
events = []
def audit(event, args):
    if event.startswith("socket.") or event in {"subprocess.Popen", "os.system", "os.remove", "os.rename"}:
        events.append(event)
sys.addaudithook(audit)
import hegel_machine.phase2b_formal_wire_covert_step5_r01_current_cut_v1 as module
print(json.dumps({"events": events, "exports": module.__all__}, sort_keys=True))
'''
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
        env={
            **os.environ,
            "PYTHONPATH": str(PROJECT_ROOT / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    # -I deliberately ignores PYTHONPATH; inject the source path explicitly.
    if completed.returncode != 0:
        script = "import sys; sys.path.insert(0, " + repr(str(PROJECT_ROOT / "src")) + "); " + script
        completed = subprocess.run(
            [sys.executable, "-I", "-c", script],
            cwd=tmp_path,
            text=True,
            capture_output=True,
            check=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
    report = json.loads(completed.stdout)
    assert report["events"] == []
    assert "process_one_current_cut_request_v1" in report["exports"]
    assert tuple(tmp_path.iterdir()) == ()
