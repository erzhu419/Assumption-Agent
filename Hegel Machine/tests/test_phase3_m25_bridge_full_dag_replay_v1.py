from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hegel_machine.phase3_m25_bridge_full_dag_replay_v1 import (
    BridgeDagReplayError,
    FAIL_ACTOR_RECEIPT,
    FAIL_NODE_COUNT,
    FAIL_NODE_SET,
    FAIL_PACKAGE_AUTHORITY,
    FAIL_ROOT_BINDING,
    FAIL_SIGNATURE,
    ReplayNodeV1,
    ROLE_SPECS,
    build_bridge_actor_replay_receipt_v1,
    build_bridge_dag_replay_package_v1,
    make_openssl_ed25519_verifier_v1,
    replay_bridge_dag_package_v1,
    validate_bridge_actor_replay_receipt_v1,
)
from hegel_machine.phase3_m25_wire_v1 import (
    bridge_attestation_signature_preimage_v1,
    build_formal_object,
    candidate_content_root,
    encode_formal_object,
    git_sha1_commit_id,
    id_digest_v1,
)
from hegel_machine.strict_cbor_v1 import (
    canonical_cbor_decode,
    canonical_cbor_encode,
    content_hash,
    rfc6962_root,
)


def _dummy_preimage(spec, salt: int) -> bytes:
    value = (1, spec.tag, spec.schema_id, *((salt + index) for index in range(spec.field_count)))
    return canonical_cbor_encode(value)


def _root(spec, node: ReplayNodeV1) -> bytes:
    if spec.operation_id == 3:
        assert node.sealed_root is not None
        return node.sealed_root
    values = tuple(canonical_cbor_decode(item) for item in node.preimages)
    if spec.operation_id == 1:
        return content_hash(spec.domain.decode("ascii"), values[0])
    return rfc6962_root(values)


def _typed_rows(count: int, *, outside: bool) -> tuple[tuple[bytes, ...], tuple[bytes, ...]]:
    universe = []
    truth = []
    for index in range(count):
        if outside:
            nested = (1, 0x3401, b"hegel-odd-input/1", 5, tuple((index >> bit) & 1 for bit in range(5)))
            signature_id = 1
        else:
            nested = (1, 0x3402, b"hegel-sink-input/1", 0, 0, 0, 0)
            signature_id = 2
        universe.append(
            canonical_cbor_encode(
                (1, 0x3201, b"hegel-bounded-universe-row/1", index, signature_id, nested)
            )
        )
        truth.append(
            canonical_cbor_encode(
                (
                    1,
                    0x3202,
                    b"hegel-target-truth-row/1",
                    index,
                    content_hash("HEGEL/CANONICAL_INPUT/V1", nested),
                    index & 1 if outside else 1,
                )
            )
        )
    return tuple(universe), tuple(truth)


def _fixture(purpose: int = 3, *, authority: bool = False):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    timestamp = 1_750_000_000
    commit = git_sha1_commit_id(bytes.fromhex("11" * 20))
    private = Ed25519PrivateKey.from_private_bytes(hashlib.sha256(b"non-authority-replay-test-key").digest())
    public = private.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    key_fields = {
        "purpose_id": 1,
        "key_id": hashlib.sha256(public).digest()[:16],
        "public_key_32_bytes": public,
        "key_epoch": 0,
        "valid_from_unix_seconds": timestamp - 1,
        "valid_until_unix_seconds_or_null": None,
        "repository_commit_id": commit,
    }
    key_root = candidate_content_root("ActorKeyManifestV1", key_fields)

    by_id = {}
    for spec in ROLE_SPECS:
        if spec.operation_id == 3:
            by_id[spec.role_id] = ReplayNodeV1(
                spec.role_id,
                sealed_root=hashlib.sha256(f"synthetic-sealed-{spec.role_id}".encode()).digest(),
            )
        elif spec.role_id not in {10, 11, 12, 20, 21, 22, 23, 24}:
            by_id[spec.role_id] = ReplayNodeV1(
                spec.role_id,
                tuple(_dummy_preimage(spec, spec.role_id * 1000 + row) for row in range(spec.exact_count)),
            )
    odd_u, odd_t = _typed_rows(480, outside=True)
    sink_u, sink_t = _typed_rows(85, outside=False)
    by_id[21] = ReplayNodeV1(21, odd_u)
    by_id[22] = ReplayNodeV1(22, odd_t)
    by_id[23] = ReplayNodeV1(23, sink_u)
    by_id[24] = ReplayNodeV1(24, sink_t)
    # Pre-M3 the hidden ledger head is exactly its genesis record/root.
    by_id[18] = ReplayNodeV1(18, by_id[17].preimages)

    roots = {spec.role_id: _root(spec, by_id[spec.role_id]) for spec in ROLE_SPECS if spec.role_id in by_id}
    split_fields = {
        "split_contract_root": bytes.fromhex("21" * 32),
        "split_seed_commitment_manifest_root": bytes.fromhex("22" * 32),
        "seed_continuity_manifest_root": roots[14],
        "split_algorithm_id_digest": id_digest_v1("hegel-split-algorithm-hkdf-hmac-sha256-rank-v1"),
        "outside_target_discovery_root": roots[25],
        "outside_target_validation_root": roots[26],
        "outside_target_sealed_root": roots[27],
        "null_control_discovery_root": roots[28],
        "null_control_validation_root": roots[29],
        "null_control_sealed_root": roots[30],
        "hidden_access_ledger_genesis_root": roots[17],
        "hidden_access_ledger_head_root": roots[18],
        "split_instantiation_status_id": 2,
        "created_at_unix_seconds": timestamp,
        "repository_commit_id": commit,
    }
    by_id[12] = ReplayNodeV1(12, (encode_formal_object("SplitBindingManifestV1", split_fields),))
    roots[12] = _root(ROLE_SPECS[11], by_id[12])

    trust_fields = {
        "trust_genesis_id_16_bytes": bytes.fromhex("31" * 16),
        "purpose_key_entries": (
            (1, key_root),
            (2, bytes.fromhex("32" * 32)),
            (3, bytes.fromhex("33" * 32)),
            (4, bytes.fromhex("34" * 32)),
        ),
        "purpose_key_policy_root": bytes.fromhex("35" * 32),
        "created_at_unix_seconds": timestamp,
        "repository_commit_id": commit,
    }
    by_id[20] = ReplayNodeV1(20, (encode_formal_object("ActorTrustGenesisV1", trust_fields),))
    roots[20] = _root(ROLE_SPECS[19], by_id[20])

    def role_binding(role_id: int) -> bytes:
        outside = role_id == 1
        fields = {
            "role_id": role_id,
            "child_dsl_spec_root": roots[1],
            "child_freeze_root": roots[2],
            "operator_semantics_root": roots[5],
            "identifier_registry_root": roots[6],
            "canonical_ast_schema_root": roots[7],
            "canonical_cbor_profile_root": roots[8],
            "semantic_spec_diagnostic_id_digest": bytes.fromhex(("41" if outside else "42") * 32),
            "semantic_spec_formal_root": bytes.fromhex(("43" if outside else "44") * 32),
            "universe_diagnostic_id_digest": bytes.fromhex(("45" if outside else "46") * 32),
            "truth_diagnostic_id_digest": bytes.fromhex(("47" if outside else "48") * 32),
            "formal_universe_root": roots[21 if outside else 23],
            "formal_truth_root": roots[22 if outside else 24],
            "split_binding_manifest_root": roots[12],
            "custodian_binding_manifest_root": roots[13],
            "seed_continuity_manifest_root": roots[14],
            "parent_binding_manifest_root_or_null": None,
            "legacy_parent_payload_source_id_digest_or_null": bytes.fromhex(("49" if outside else "4a") * 32),
            "parent_manifest_absence_attestation_root_or_null": roots[16],
            "fallback_registry_root_or_null": None,
            "created_at_unix_seconds": timestamp,
            "repository_commit_id": commit,
        }
        return encode_formal_object("DslRoleBindingManifestV1", fields)

    by_id[10] = ReplayNodeV1(10, (role_binding(1),))
    by_id[11] = ReplayNodeV1(11, (role_binding(2),))
    roots[10] = _root(ROLE_SPECS[9], by_id[10])
    roots[11] = _root(ROLE_SPECS[10], by_id[11])

    candidate_fields = {spec.field_name: roots[spec.role_id] for spec in ROLE_SPECS}
    candidate_fields.update(
        {
            "run_id": bytes.fromhex("51" * 16),
            "canonical_program_budget": 50_000,
            "raw_operator_application_cap": 5_000_000,
            "records_per_chunk": 4096,
            "equivalence_mode_id": 1,
            "created_at_unix_seconds": timestamp,
            "repository_commit_id": commit,
        }
    )
    candidate_root = candidate_content_root("M3ExecutionCandidateV1", candidate_fields)
    bridge_fields = {
        "run_id": candidate_fields["run_id"],
        "diagnostic_formal_bridge_root": candidate_fields["diagnostic_formal_bridge_root"],
        "m3_execution_candidate_root": candidate_root,
        "child_dsl_spec_root": candidate_fields["child_dsl_spec_root"],
        "child_freeze_root": candidate_fields["child_freeze_root"],
        "actor_trust_genesis_root": candidate_fields["actor_trust_genesis_root"],
        "opaque_id_registry_snapshot_root": candidate_fields["opaque_id_registry_snapshot_root"],
    }
    bridge_root = candidate_content_root("BridgeReplayStatementV1", bridge_fields)
    signature = None if purpose == 1 else private.sign(bridge_attestation_signature_preimage_v1(bridge_root, 1, 0))
    package = build_bridge_dag_replay_package_v1(
        purpose_id=purpose,
        candidate_fields=candidate_fields,
        bridge_statement_fields=bridge_fields,
        nodes=tuple(by_id[index] for index in range(1, 38)),
        purpose1_actor_key_manifest_fields=key_fields,
        purpose1_bridge_signature=signature,
        authority=authority,
    )

    def verifier(key: bytes, sig: bytes, message: bytes) -> None:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        Ed25519PublicKey.from_public_bytes(key).verify(sig, message)

    return package, verifier


def _mutate(package: bytes, mutate) -> bytes:
    value = list(canonical_cbor_decode(package))
    mutate(value)
    return canonical_cbor_encode(tuple(value))


def _rust_replay_binary() -> Path:
    configured = os.environ.get("HEGEL_M25_BRIDGE_DAG_REPLAY_BINARY")
    if configured:
        return Path(configured)
    return (
        Path(__file__).resolve().parents[1]
        / "rust/m25_bridge_dag_replay/target/debug/hegel-m25-bridge-dag-replay"
    )


def test_unsigned_purpose1_replays_but_does_not_claim_split_membership() -> None:
    package, _ = _fixture(1)
    result = replay_bridge_dag_package_v1(package)
    assert result.eligible_to_sign_bridge_statement
    assert not result.purpose1_signature_verified
    assert not result.split_membership_recomputed
    assert not result.authoritative


def test_signed_purpose3_replays_with_injected_host_verifier() -> None:
    package, verifier = _fixture(3)
    result = replay_bridge_dag_package_v1(package, signature_verifier=verifier)
    assert result.purpose1_signature_verified
    assert result.purpose_id == 3


def test_signed_replay_fails_without_explicit_verifier() -> None:
    package, _ = _fixture(2)
    with pytest.raises(BridgeDagReplayError) as caught:
        replay_bridge_dag_package_v1(package)
    assert caught.value.code == FAIL_SIGNATURE


def test_openssl_verifier_uses_explicit_private_tmp_and_cleans(tmp_path: Path) -> None:
    private = Path("/tmp") / f"hegel-m25-openssl-test-{hashlib.sha256(str(tmp_path).encode()).hexdigest()[:16]}"
    private.mkdir(mode=0o700)
    private.chmod(0o700)
    try:
        package, _ = _fixture(3)
        verifier = make_openssl_ed25519_verifier_v1(private)
        assert replay_bridge_dag_package_v1(package, signature_verifier=verifier).purpose1_signature_verified
        assert list(private.iterdir()) == []
    finally:
        private.rmdir()


@pytest.mark.parametrize("role_id", [1, 5, 6, 21])
def test_preimage_substitution_wrong_dsl_operator_registry_or_universe_fails(role_id: int) -> None:
    package, verifier = _fixture(3)

    def attack(value):
        nodes = list(value[7])
        node = list(nodes[role_id - 1])
        rows = list(node[5])
        row = bytearray(rows[0])
        row[-1] ^= 1
        rows[0] = bytes(row)
        node[5] = tuple(rows)
        nodes[role_id - 1] = tuple(node)
        value[7] = tuple(nodes)

    with pytest.raises(BridgeDagReplayError):
        replay_bridge_dag_package_v1(_mutate(package, attack), signature_verifier=verifier)


def test_node_omission_fails() -> None:
    package, verifier = _fixture(3)
    attacked = _mutate(package, lambda value: value.__setitem__(7, tuple(value[7][:-1])))
    with pytest.raises(BridgeDagReplayError) as caught:
        replay_bridge_dag_package_v1(attacked, signature_verifier=verifier)
    assert caught.value.code == FAIL_NODE_SET


def test_cross_role_swap_fails() -> None:
    package, verifier = _fixture(3)

    def attack(value):
        nodes = list(value[7])
        left, right = list(nodes[9]), list(nodes[10])
        left[5], right[5] = right[5], left[5]
        nodes[9], nodes[10] = tuple(left), tuple(right)
        value[7] = tuple(nodes)

    with pytest.raises(BridgeDagReplayError) as caught:
        replay_bridge_dag_package_v1(_mutate(package, attack), signature_verifier=verifier)
    assert caught.value.code == FAIL_ROOT_BINDING


def test_candidate_root_splice_fails() -> None:
    package, verifier = _fixture(3)

    def attack(value):
        candidate = list(canonical_cbor_decode(value[5]))
        candidate[4] = candidate[5]
        value[5] = canonical_cbor_encode(tuple(candidate))

    with pytest.raises(BridgeDagReplayError) as caught:
        replay_bridge_dag_package_v1(_mutate(package, attack), signature_verifier=verifier)
    assert caught.value.code == FAIL_ROOT_BINDING


def test_signature_substitution_fails() -> None:
    package, verifier = _fixture(3)
    attacked = _mutate(package, lambda value: value.__setitem__(9, bytes(64)))
    with pytest.raises(BridgeDagReplayError) as caught:
        replay_bridge_dag_package_v1(attacked, signature_verifier=verifier)
    assert caught.value.code == FAIL_SIGNATURE


def test_authority_flag_is_fail_closed() -> None:
    package, verifier = _fixture(3)
    attacked = _mutate(package, lambda value: value.__setitem__(3, True))
    with pytest.raises(BridgeDagReplayError) as caught:
        replay_bridge_dag_package_v1(attacked, signature_verifier=verifier)
    assert caught.value.code == FAIL_PACKAGE_AUTHORITY


def _load_actor_worker(filename: str, module_name: str):
    project = Path(__file__).resolve().parents[1]
    tools = project / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    spec = importlib.util.spec_from_file_location(module_name, tools / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prepare_python_actor(worker, monkeypatch, tmp_path: Path, purpose: int) -> tuple[Path, Path, Path]:
    input_directory = tmp_path / "input"
    output_directory = tmp_path / "output"
    state_directory = tmp_path / "state"
    input_directory.mkdir()
    output_directory.mkdir()
    state_directory.mkdir(mode=0o700)
    private_key = state_directory / "ed25519-private.pem"
    completed = subprocess.run(
        [
            "/usr/bin/openssl",
            "genpkey",
            "-algorithm",
            "ED25519",
            "-out",
            str(private_key),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
        env={"LC_ALL": "C", "LANG": "C"},
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    private_key.chmod(0o600)
    monkeypatch.setattr(worker, "INPUT", input_directory)
    monkeypatch.setattr(worker, "OUTPUT", output_directory)
    monkeypatch.setattr(worker, "STATE", state_directory)
    monkeypatch.setattr(worker, "PRIVATE_KEY", private_key)
    monkeypatch.setattr(worker, "SIGNATURE", output_directory / "ed25519-signature.bin")
    monkeypatch.setattr(worker, "BRIDGE_DAG_PACKAGE", input_directory / "bridge-dag-package.cbor")
    monkeypatch.setattr(
        worker,
        "BRIDGE_REPLAY_RECEIPT",
        output_directory / "bridge-dag-replay-receipt.json",
    )
    nonce = hashlib.sha256(f"{tmp_path}:{purpose}".encode()).hexdigest()[:32]
    monkeypatch.setenv("HEGEL_OPERATION_NONCE", nonce)
    return input_directory, output_directory, private_key


def _assert_actor_signature_and_receipt(
    *,
    package: bytes,
    purpose: int,
    output_directory: Path,
    private_key: Path,
    verifier,
) -> None:
    from cryptography.hazmat.primitives import serialization

    replay = replay_bridge_dag_package_v1(
        package,
        allow_authoritative=True,
        signature_verifier=verifier,
    )
    signature = (output_directory / "ed25519-signature.bin").read_bytes()
    loaded = serialization.load_pem_private_key(private_key.read_bytes(), password=None)
    loaded.public_key().verify(
        signature,
        bridge_attestation_signature_preimage_v1(
            replay.bridge_statement_root, purpose, 0
        ),
    )
    receipt_path = output_directory / "bridge-dag-replay-receipt.json"
    raw = receipt_path.read_bytes()
    validated = validate_bridge_actor_replay_receipt_v1(
        raw,
        expected_result=replay,
        expected_implementation="python-full-dag-replay-v1",
        require_authoritative=True,
    )
    assert validated["purpose"] == purpose
    receipt = json.loads(raw)
    assert raw == (
        json.dumps(receipt, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")
    receipt_hash = receipt.pop("receipt_sha256")
    assert receipt_hash == hashlib.sha256(
        (
            json.dumps(receipt, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
            + "\n"
        ).encode("ascii")
    ).hexdigest()
    assert receipt["authoritative"] is True
    assert receipt["bridge_statement_root_hex"] == replay.bridge_statement_root.hex()
    assert receipt["candidate_root_hex"] == replay.candidate_root.hex()
    assert receipt["package_digest_hex"] == replay.package_digest.hex()
    assert receipt["purpose"] == purpose
    assert receipt["purpose1_signature_verified"] is (purpose != 1)
    assert receipt["split_membership_recomputed"] is False
    assert receipt["status"] == "PASS"


def test_purpose1_actor_replays_authoritative_full_dag_and_derives_preimage_internally(
    monkeypatch, tmp_path: Path
) -> None:
    worker = _load_actor_worker(
        "phase3_m25_formal_actor_worker_v1.py", "hegel_test_purpose1_actor"
    )
    input_directory, output_directory, private_key = _prepare_python_actor(
        worker, monkeypatch, tmp_path, 1
    )
    monkeypatch.setattr(worker, "require_profile", lambda purpose: None)
    package, verifier = _fixture(1, authority=True)
    (input_directory / "bridge-dag-package.cbor").write_bytes(package)

    worker.python_bridge_replay_sign(1)

    assert not (input_directory / "signing-preimage.bin").exists()
    assert not (input_directory / "bridge-statement.cbor").exists()
    assert not (input_directory / "expected-root.bin").exists()
    _assert_actor_signature_and_receipt(
        package=package,
        purpose=1,
        output_directory=output_directory,
        private_key=private_key,
        verifier=verifier,
    )


def test_purpose2_actor_verifies_purpose1_before_internal_derivation_and_signing(
    monkeypatch, tmp_path: Path
) -> None:
    worker = _load_actor_worker(
        "phase3_m25_python_bridge_actor_worker_v1.py", "hegel_test_purpose2_actor"
    )
    input_directory, output_directory, private_key = _prepare_python_actor(
        worker, monkeypatch, tmp_path, 2
    )
    monkeypatch.setattr(worker, "require_profile", lambda: None)
    package, verifier = _fixture(2, authority=True)
    (input_directory / "bridge-dag-package.cbor").write_bytes(package)

    worker.bridge_replay_sign()

    assert not (input_directory / "signing-preimage.bin").exists()
    assert not (input_directory / "bridge-statement.cbor").exists()
    assert not (input_directory / "expected-root.bin").exists()
    _assert_actor_signature_and_receipt(
        package=package,
        purpose=2,
        output_directory=output_directory,
        private_key=private_key,
        verifier=verifier,
    )


@pytest.mark.parametrize("attack", ["purpose1-signature", "wrong-purpose", "non-authoritative"])
def test_purpose2_actor_fails_before_signing_on_replay_or_authority_attack(
    attack: str, monkeypatch, tmp_path: Path
) -> None:
    worker = _load_actor_worker(
        "phase3_m25_python_bridge_actor_worker_v1.py",
        f"hegel_test_purpose2_actor_attack_{attack}",
    )
    input_directory, output_directory, _private_key = _prepare_python_actor(
        worker, monkeypatch, tmp_path, 2
    )
    monkeypatch.setattr(worker, "require_profile", lambda: None)
    if attack == "wrong-purpose":
        package, _ = _fixture(3, authority=True)
    else:
        package, _ = _fixture(2, authority=attack != "non-authoritative")
    if attack == "purpose1-signature":
        package = _mutate(package, lambda value: value.__setitem__(9, bytes(64)))
    (input_directory / "bridge-dag-package.cbor").write_bytes(package)

    with pytest.raises(Exception):
        worker.bridge_replay_sign()

    assert not (output_directory / "ed25519-signature.bin").exists()
    assert not (output_directory / "bridge-dag-replay-receipt.json").exists()


def test_rust_actor_contract_has_no_host_bridge_root_or_signing_preimage() -> None:
    project = Path(__file__).resolve().parents[1]
    source = (project / "tools/phase3_m25_formal_rust_actor_worker_v1.sh").read_text(
        encoding="utf-8"
    )
    assert "/input/bridge-dag-package.cbor" in source
    assert "/input/rust-bridge-dag-replay" in source
    assert "--authoritative-runtime" in source
    assert "--expected-purpose 3" in source
    assert '"purpose1_signature_verified":true' in source
    assert "/input/signing-preimage.bin" not in source
    assert "/input/expected-root.bin" not in source
    assert "/input/bridge-statement.cbor" not in source
    assert source.index("rust-live-probe") < source.index(
        "/input/rust-bridge-dag-replay \\\n"
    ) < source.index("/usr/bin/openssl pkeyutl -sign")


@pytest.mark.parametrize(
    "attack",
    ("extra-field", "self-hash", "purpose-signature-state", "noncanonical"),
)
def test_strict_actor_receipt_validator_rejects_tampering(attack: str) -> None:
    package, verifier = _fixture(2, authority=True)
    replay = replay_bridge_dag_package_v1(
        package,
        allow_authoritative=True,
        signature_verifier=verifier,
    )
    payload = build_bridge_actor_replay_receipt_v1(
        replay, implementation="python-full-dag-replay-v1"
    )
    value = json.loads(payload)
    if attack == "extra-field":
        value["unexpected"] = 1
    elif attack == "self-hash":
        value["receipt_sha256"] = "00" * 32
    elif attack == "purpose-signature-state":
        value["purpose1_signature_verified"] = False
    elif attack == "noncanonical":
        attacked = json.dumps(value, sort_keys=False, indent=2).encode("ascii")
        with pytest.raises(BridgeDagReplayError) as caught:
            validate_bridge_actor_replay_receipt_v1(attacked)
        assert caught.value.code == FAIL_ACTOR_RECEIPT
        return
    attacked = (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")
    with pytest.raises(BridgeDagReplayError) as caught:
        validate_bridge_actor_replay_receipt_v1(attacked)
    assert caught.value.code == FAIL_ACTOR_RECEIPT


def test_rust_replayer_agrees_when_prebuilt(tmp_path: Path) -> None:
    binary = _rust_replay_binary()
    if not binary.is_file():
        pytest.skip("independent Rust binary is built in the pinned offline Rust image")
    package, _ = _fixture(3)
    package_path = tmp_path / "package.cbor"
    package_path.write_bytes(package)
    private = Path("/tmp") / f"hegel-m25-rust-replay-{hashlib.sha256(str(tmp_path).encode()).hexdigest()[:16]}"
    private.mkdir(mode=0o700)
    private.chmod(0o700)
    try:
        completed = subprocess.run(
            [str(binary), str(package_path), str(private)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={"LC_ALL": "C", "LANG": "C"},
            check=False,
            timeout=60,
        )
        assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
        assert b'"status":"PASS"' in completed.stdout
        assert list(private.iterdir()) == []
    finally:
        private.rmdir()


def test_rust_replayer_independently_rejects_substitution_omission_and_signature(tmp_path: Path) -> None:
    binary = _rust_replay_binary()
    if not binary.is_file():
        pytest.skip("independent Rust binary is built in the pinned offline Rust image")
    package, _ = _fixture(3)

    def substitution(value):
        nodes = list(value[7])
        node = list(nodes[0])
        rows = list(node[5])
        row = bytearray(rows[0])
        row[-1] ^= 1
        rows[0] = bytes(row)
        node[5] = tuple(rows)
        nodes[0] = tuple(node)
        value[7] = tuple(nodes)

    attacked = (
        _mutate(package, substitution),
        _mutate(package, lambda value: value.__setitem__(7, tuple(value[7][:-1]))),
        _mutate(package, lambda value: value.__setitem__(9, bytes(64))),
    )
    private = Path("/tmp") / f"hegel-m25-rust-attacks-{hashlib.sha256(str(tmp_path).encode()).hexdigest()[:16]}"
    private.mkdir(mode=0o700)
    private.chmod(0o700)
    try:
        for index, payload in enumerate(attacked):
            path = tmp_path / f"attack-{index}.cbor"
            path.write_bytes(payload)
            completed = subprocess.run(
                [str(binary), str(path), str(private)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={"LC_ALL": "C", "LANG": "C"},
                check=False,
                timeout=60,
            )
            assert completed.returncode == 1
            assert b"FAIL_M25_BRIDGE_REPLAY_" in completed.stderr
            assert list(private.iterdir()) == []
    finally:
        private.rmdir()


def test_rust_authoritative_runtime_flag_derives_purpose3_preimage_and_strict_receipt(
    tmp_path: Path,
) -> None:
    binary = _rust_replay_binary()
    if not binary.is_file():
        pytest.skip("independent Rust binary is built in the pinned offline Rust image")
    package, verifier = _fixture(3, authority=True)
    expected = replay_bridge_dag_package_v1(
        package,
        allow_authoritative=True,
        signature_verifier=verifier,
    )
    package_path = tmp_path / "authoritative-package.cbor"
    package_path.write_bytes(package)
    private = Path("/tmp") / (
        "hegel-m25-rust-authoritative-"
        + hashlib.sha256(str(tmp_path).encode()).hexdigest()[:16]
    )
    private.mkdir(mode=0o700)
    private.chmod(0o700)
    preimage_path = private / "purpose3-preimage.bin"
    try:
        without_flag = subprocess.run(
            [str(binary), str(package_path), str(private)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={"LC_ALL": "C", "LANG": "C"},
            check=False,
            timeout=60,
        )
        assert without_flag.returncode == 1
        assert b"FAIL_M25_BRIDGE_REPLAY_AUTHORITY_GUARD" in without_flag.stderr
        completed = subprocess.run(
            [
                str(binary),
                "--authoritative-runtime",
                "--expected-purpose",
                "3",
                "--signature-preimage-out",
                str(preimage_path),
                str(package_path),
                str(private),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={"LC_ALL": "C", "LANG": "C"},
            check=False,
            timeout=60,
        )
        assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
        validated = validate_bridge_actor_replay_receipt_v1(
            completed.stdout,
            expected_result=expected,
            expected_implementation="rust-full-dag-replay-v1",
            require_authoritative=True,
        )
        assert validated["purpose"] == 3
        receipt = json.loads(completed.stdout)
        assert completed.stdout == (
            json.dumps(receipt, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
            + "\n"
        ).encode("ascii")
        digest = receipt.pop("receipt_sha256")
        assert digest == hashlib.sha256(
            (
                json.dumps(receipt, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
                + "\n"
            ).encode("ascii")
        ).hexdigest()
        assert receipt["authoritative"] is True
        assert receipt["bridge_statement_root_hex"] == expected.bridge_statement_root.hex()
        assert receipt["candidate_root_hex"] == expected.candidate_root.hex()
        assert receipt["package_digest_hex"] == expected.package_digest.hex()
        assert receipt["purpose"] == 3
        assert receipt["purpose1_signature_verified"] is True
        assert preimage_path.read_bytes() == bridge_attestation_signature_preimage_v1(
            expected.bridge_statement_root, 3, 0
        )
    finally:
        try:
            preimage_path.unlink()
        except FileNotFoundError:
            pass
        private.rmdir()
