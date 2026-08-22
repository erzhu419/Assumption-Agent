from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import fields
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import stat
from typing import Callable, Mapping

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import hegel_machine.phase2b_formal_wire_covert_step5_authority_v2 as authority
import hegel_machine.phase2b_formal_wire_covert_step5_r01_evidence_contract_v1 as contract
import hegel_machine.phase2b_formal_wire_covert_step5_r01_external_witness_v1 as witness
from test_phase2b_formal_wire_covert_step5_r01_evidence_contract_v1 import (
    _EvidenceGraph,
    _authority_signature,
    _build_evidence,
    _verify,
)


_EVIDENCE_SOURCE_SHA256 = (
    "574e07e7ce0f19075c29fa59cc94e4e988a34c301e9e09005a8aed805d69e96d"
)
_AUTHORITY_SOURCE_SHA256 = (
    "03d689327ca422f93e5a32db0a7f813fe84b1099f1bac1b31395c975d76d8e86"
)
_EXTERNAL_POLICY_DOMAIN = (
    b"HEGEL/PHASE2B/STEP5/EXTERNAL_WITNESS_POLICY/V1\0"
)
_MERKLE_LEAF_DOMAIN = (
    b"HEGEL/PHASE2B/STEP5/EXTERNAL_WITNESS/MERKLE_LEAF/V1\0"
)


def _private_spool(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.mkdir(mode=0o700)
    path.chmod(0o700)
    return path


def _call_arguments(graph: _EvidenceGraph, spool: Path) -> dict[str, object]:
    authority_graph = graph.authority
    profile_id = authority.phase2b_step5_content_id_v2(
        "authority_profile", authority_graph.profile_body
    )
    registry_id = json.loads(authority_graph.registry_payload)["id"]
    return {
        "stage_b_trust_lock_payload": authority_graph.lock_payload,
        "authority_profile_payload": authority_graph.profile_payload,
        "signer_registry_payload": authority_graph.registry_payload,
        "external_witness_policy_payload": graph.payloads["policy"],
        "signed_owner_status_anti_fork_anchor_payload": graph.payloads["anchor"],
        "out_of_band_expected_lock_id_bytes": authority_graph.lock_id.encode(
            "ascii"
        ),
        "out_of_band_expected_lock_payload_sha256_digest": hashlib.sha256(
            authority_graph.lock_payload
        ).digest(),
        "out_of_band_expected_external_witness_policy_id_bytes": graph.values[
            "policy"
        ]["external_witness_policy_id"].encode("ascii"),
        "out_of_band_expected_external_witness_policy_payload_sha256_digest": (
            hashlib.sha256(graph.payloads["policy"]).digest()
        ),
        "expected_stage_b_trust_lock_id_bytes": authority_graph.lock_id.encode(
            "ascii"
        ),
        "expected_authority_profile_id_bytes": profile_id.encode("ascii"),
        "expected_signer_registry_id_bytes": registry_id.encode("ascii"),
        "expected_attempt_context_id_bytes": graph.attempt_id.encode("ascii"),
        "external_witness_private_key": graph.witness_private_key,
        "owner_private_spool_directory_path": str(spool),
    }


def _issue(graph: _EvidenceGraph, spool: Path, **changes: object):
    arguments = _call_arguments(graph, spool)
    arguments.update(changes)
    return witness.issue_initial_external_owner_status_anchor_receipt_v1(
        **arguments
    )


def _assert_external_error(code: str, operation: Callable[[], object]) -> None:
    with pytest.raises(witness.Phase2BStep5R01ExternalWitnessErrorV1) as raised:
        operation()
    assert raised.value.code == code


def _verified_inputs(graph: _EvidenceGraph):
    arguments = _call_arguments(graph, Path("/unused"))
    lock, profile, registry = witness._load_raw_authority_graph(
        arguments["stage_b_trust_lock_payload"],
        arguments["authority_profile_payload"],
        arguments["signer_registry_payload"],
        out_of_band_expected_lock_id_bytes=arguments[
            "out_of_band_expected_lock_id_bytes"
        ],
        out_of_band_expected_lock_payload_sha256_digest=arguments[
            "out_of_band_expected_lock_payload_sha256_digest"
        ],
        expected_stage_b_trust_lock_id_bytes=arguments[
            "expected_stage_b_trust_lock_id_bytes"
        ],
        expected_authority_profile_id_bytes=arguments[
            "expected_authority_profile_id_bytes"
        ],
        expected_signer_registry_id_bytes=arguments[
            "expected_signer_registry_id_bytes"
        ],
    )
    policy = witness._load_external_policy(
        arguments["external_witness_policy_payload"],
        expected_id_bytes=arguments[
            "out_of_band_expected_external_witness_policy_id_bytes"
        ],
        expected_payload_digest=arguments[
            "out_of_band_expected_external_witness_policy_payload_sha256_digest"
        ],
        registry=registry,
    )
    anchor = witness._load_signed_anchor(
        arguments["signed_owner_status_anti_fork_anchor_payload"],
        lock=lock,
        profile=profile,
        registry=registry,
        expected_attempt_context_id=graph.attempt_id,
        external_policy_id=policy["external_witness_policy_id"],
    )
    return lock, profile, registry, policy, anchor


def _install_persisted_receipt(
    graph: _EvidenceGraph,
    spool: Path,
    mutate: Callable[[dict[str, object]], None],
    *,
    recompute_id: bool = True,
    resign: bool = True,
) -> bytes:
    lock, profile, registry, policy, anchor = _verified_inputs(graph)
    receipt = json.loads(graph.payloads["external_receipt"])
    receipt.pop("signature_hex")
    receipt.pop("receipt_id")
    mutate(receipt)
    if recompute_id:
        receipt["receipt_id"] = (
            contract.phase2b_step5_r01_evidence_content_id_v1(
                "external_owner_status_anchor_receipt", dict(receipt)
            )
        )
    else:
        receipt["receipt_id"] = graph.values["external_receipt"]["receipt_id"]
    if resign:
        receipt["signature_hex"] = graph.witness_private_key.sign(
            contract.phase2b_step5_external_anchor_receipt_signature_preimage_v1(
                receipt
            )
        ).hex()
    else:
        valid_signature = graph.values["external_receipt"]["signature_hex"]
        receipt["signature_hex"] = valid_signature[:-1] + (
            "0" if valid_signature[-1] != "0" else "1"
        )
    receipt_payload = contract.canonical_r01_evidence_contract_json_bytes_v1(
        receipt
    )
    (
        ledger_payload,
        ledger,
        checkpoint_payload,
        checkpoint,
        head_payload,
        head,
    ) = witness._compose_spool_artifacts(
        lock=lock,
        profile=profile,
        registry=registry,
        expected_attempt_context_id=graph.attempt_id,
        policy=policy,
        anchor_payload=graph.payloads["anchor"],
        anchor=anchor,
        receipt_payload=receipt_payload,
        receipt=receipt,
    )
    initialization_lock_payload, basenames = witness._compose_initialization_lock(
        lock=lock,
        profile=profile,
        registry=registry,
        expected_attempt_context_id=graph.attempt_id,
        policy_payload=graph.payloads["policy"],
        policy=policy,
        anchor_payload=graph.payloads["anchor"],
        anchor=anchor,
        receipt_payload=receipt_payload,
        receipt=receipt,
        ledger_payload=ledger_payload,
        ledger=ledger,
        checkpoint_payload=checkpoint_payload,
        checkpoint=checkpoint,
        head_payload=head_payload,
        head=head,
    )
    witness._spool_initial_leaf(
        str(spool),
        initialization_lock_payload=initialization_lock_payload,
        receipt_payload=receipt_payload,
        ledger_payload=ledger_payload,
        checkpoint_payload=checkpoint_payload,
        head_payload=head_payload,
        basenames=basenames,
    )
    return receipt_payload


def _policy_payload(policy: Mapping[str, object]) -> bytes:
    value = copy.deepcopy(dict(policy))
    value.pop("external_witness_policy_id", None)
    value["external_witness_policy_id"] = (
        "phase2b_step5_external_witness_policy_v1_"
        + hashlib.sha256(
            _EXTERNAL_POLICY_DOMAIN
            + contract.canonical_r01_evidence_contract_json_bytes_v1(value)
        ).hexdigest()
    )
    return contract.canonical_r01_evidence_contract_json_bytes_v1(value)


def _resigned_anchor(
    graph: _EvidenceGraph, mutate: Callable[[dict[str, object]], None]
) -> bytes:
    root = copy.deepcopy(dict(graph.values["anchor"]))
    body = root["body"]
    mutate(body)
    root["anchor_id"] = contract.phase2b_step5_r01_evidence_content_id_v1(
        "signed_owner_status_anti_fork_anchor", body
    )
    root["signature"] = _authority_signature(
        graph.authority,
        role="PURE_REPLAYER",
        purpose="SIGN_OWNER_STATUS_ANTI_FORK_ANCHOR",
        artifact_kind="owner_status_anti_fork_anchor",
        body=body,
    )
    return contract.canonical_r01_evidence_contract_json_bytes_v1(root)


def _flip_digest(value: bytes) -> bytes:
    return value[:-1] + bytes([value[-1] ^ 1])


def _different_raw_pin(value: bytes) -> bytes:
    if len(value) == 32:
        return _flip_digest(value)
    replacement = b"0" if value[-1:] != b"0" else b"1"
    return value[:-1] + replacement


def test_exact_upstream_sources_and_public_api_are_frozen() -> None:
    assert witness.EVIDENCE_CONTRACT_SOURCE_SHA256_V1 == _EVIDENCE_SOURCE_SHA256
    assert hashlib.sha256(Path(contract.__file__).read_bytes()).hexdigest() == (
        _EVIDENCE_SOURCE_SHA256
    )
    assert hashlib.sha256(Path(authority.__file__).read_bytes()).hexdigest() == (
        _AUTHORITY_SOURCE_SHA256
    )
    assert tuple(
        inspect.signature(
            witness.issue_initial_external_owner_status_anchor_receipt_v1
        ).parameters
    ) == (
        "stage_b_trust_lock_payload",
        "authority_profile_payload",
        "signer_registry_payload",
        "external_witness_policy_payload",
        "signed_owner_status_anti_fork_anchor_payload",
        "out_of_band_expected_lock_id_bytes",
        "out_of_band_expected_lock_payload_sha256_digest",
        "out_of_band_expected_external_witness_policy_id_bytes",
        "out_of_band_expected_external_witness_policy_payload_sha256_digest",
        "expected_stage_b_trust_lock_id_bytes",
        "expected_authority_profile_id_bytes",
        "expected_signer_registry_id_bytes",
        "expected_attempt_context_id_bytes",
        "external_witness_private_key",
        "owner_private_spool_directory_path",
    )


def test_real_authority_pure_and_external_key_match_evidence_bytes(
    tmp_path: Path,
) -> None:
    graph = _build_evidence()
    spool = _private_spool(tmp_path, "positive")
    result = _issue(graph, spool)

    assert result.receipt_payload == graph.payloads["external_receipt"]
    assert witness.require_issued_initial_external_witness_candidate_v1(result) is result
    assert _verify(graph).candidate_signature_mechanics_verified is True
    receipt = json.loads(result.receipt_payload)
    assert frozenset(receipt) == contract.EXTERNAL_OWNER_STATUS_ANCHOR_RECEIPT_FIELDS_V1
    assert receipt["receipt_id"] == (
        contract.phase2b_step5_r01_evidence_content_id_v1(
            "external_owner_status_anchor_receipt",
            {
                key: value
                for key, value in receipt.items()
                if key not in {"receipt_id", "signature_hex"}
            },
        )
    )
    expected_root = hashlib.sha256(
        _MERKLE_LEAF_DOMAIN + graph.payloads["anchor"]
    ).hexdigest()
    assert receipt["signed_tree_root_sha256"] == expected_root
    assert receipt["leaf_index"] == 0
    assert receipt["tree_size"] == 1
    assert receipt["inclusion_path_sha256_in_exact_order"] == []
    public_key = graph.witness_private_key.public_key()
    public_key.verify(
        bytes.fromhex(receipt["signature_hex"]),
        contract.phase2b_step5_external_anchor_receipt_signature_preimage_v1(
            receipt
        ),
    )

    names = result.spool_file_basenames_in_commit_order
    assert len(names) == 5
    assert names[0] == "external-witness-initialization-v1.lock"
    assert ".external-witness-head." in names[-1]
    assert tuple(path.name for path in spool.iterdir()) != ()
    assert {path.name for path in spool.iterdir()} == set(names)
    for name in names:
        metadata = (spool / name).stat()
        assert stat.S_ISREG(metadata.st_mode)
        assert stat.S_IMODE(metadata.st_mode) == 0o600
        assert metadata.st_nlink == 1


def test_exact_policy_anchor_receipt_and_local_head_graph(tmp_path: Path) -> None:
    graph = _build_evidence()
    spool = _private_spool(tmp_path, "exact-graph")
    result = _issue(graph, spool)
    names = result.spool_file_basenames_in_commit_order
    initialization = json.loads((spool / names[0]).read_bytes())
    receipt = json.loads((spool / names[1]).read_bytes())
    ledger = json.loads((spool / names[2]).read_bytes())
    checkpoint = json.loads((spool / names[3]).read_bytes())
    head = json.loads((spool / names[4]).read_bytes())
    policy = json.loads(graph.payloads["policy"])
    anchor = json.loads(graph.payloads["anchor"])

    assert frozenset(policy) == contract.EXTERNAL_WITNESS_POLICY_FIELDS_V1
    assert frozenset(anchor) == contract.SIGNED_OWNER_STATUS_ANTI_FORK_ANCHOR_FIELDS_V1
    assert frozenset(anchor["body"]) == (
        contract.SIGNED_OWNER_STATUS_ANTI_FORK_ANCHOR_BODY_FIELDS_V1
    )
    assert frozenset(receipt) == contract.EXTERNAL_OWNER_STATUS_ANCHOR_RECEIPT_FIELDS_V1
    assert frozenset(initialization) == witness._INITIALIZATION_LOCK_FIELDS
    assert frozenset(ledger) == witness._LEDGER_FIELDS
    assert frozenset(checkpoint) == witness._CHECKPOINT_FIELDS
    assert frozenset(head) == witness._HEAD_FIELDS

    policy_subject = {
        key: value
        for key, value in policy.items()
        if key != "external_witness_policy_id"
    }
    assert policy["external_witness_policy_id"] == (
        "phase2b_step5_external_witness_policy_v1_"
        + hashlib.sha256(
            _EXTERNAL_POLICY_DOMAIN
            + contract.canonical_r01_evidence_contract_json_bytes_v1(
                policy_subject
            )
        ).hexdigest()
    )
    assert anchor["anchor_id"] == (
        contract.phase2b_step5_r01_evidence_content_id_v1(
            "signed_owner_status_anti_fork_anchor", anchor["body"]
        )
    )
    pure_row = graph.authority.role_keys[
        authority.ROLE_ORDER_V2.index("PURE_REPLAYER")
    ]
    pure_preimage = authority.phase2b_step5_signature_preimage_v2(
        artifact_kind="owner_status_anti_fork_anchor",
        body=anchor["body"],
        role="PURE_REPLAYER",
        purpose="SIGN_OWNER_STATUS_ANTI_FORK_ANCHOR",
        signer_key_id=pure_row["key_id"],
        key_epoch=1,
    )
    public_key = graph.authority.private_keys["PURE_REPLAYER"].public_key()
    public_key.verify(bytes.fromhex(anchor["signature"]["signature_hex"]), pure_preimage)

    assert ledger["ledger_id"] == witness._content_id(
        witness._LEDGER_PREFIX,
        witness._LEDGER_DOMAIN,
        witness._without(ledger, "ledger_id"),
    )
    assert checkpoint["checkpoint_id"] == witness._content_id(
        witness._CHECKPOINT_PREFIX,
        witness._CHECKPOINT_DOMAIN,
        witness._without(checkpoint, "checkpoint_id"),
    )
    assert head["head_id"] == witness._content_id(
        witness._HEAD_PREFIX,
        witness._HEAD_DOMAIN,
        witness._without(head, "head_id"),
    )
    assert checkpoint["ledger_id"] == ledger["ledger_id"]
    assert head["ledger_id"] == ledger["ledger_id"]
    assert head["checkpoint_id"] == checkpoint["checkpoint_id"]
    assert head["checkpoint_payload_sha256"] == hashlib.sha256(
        (spool / names[3]).read_bytes()
    ).hexdigest()
    assert initialization["receipt_id"] == receipt["receipt_id"]
    assert initialization["ledger_id"] == ledger["ledger_id"]
    assert initialization["checkpoint_id"] == checkpoint["checkpoint_id"]
    assert initialization["head_id"] == head["head_id"]
    assert initialization["head_file_basename"] == names[4]
    assert checkpoint["previous_checkpoint_id"] is None
    assert head["previous_head_id"] is None


def test_receipt_id_and_signature_preimages_have_no_cycle(tmp_path: Path) -> None:
    graph = _build_evidence()
    result = _issue(graph, _private_spool(tmp_path, "preimage"))
    receipt = json.loads(result.receipt_payload)
    id_subject = {
        key: value
        for key, value in receipt.items()
        if key not in {"receipt_id", "signature_hex"}
    }
    assert receipt["receipt_id"] == (
        contract.phase2b_step5_r01_evidence_content_id_v1(
            "external_owner_status_anchor_receipt", id_subject
        )
    )
    preimage = (
        contract.phase2b_step5_external_anchor_receipt_signature_preimage_v1(
            receipt
        )
    )
    changed_signature = dict(receipt)
    changed_signature["signature_hex"] = "0" * 128
    assert (
        contract.phase2b_step5_external_anchor_receipt_signature_preimage_v1(
            changed_signature
        )
        == preimage
    )
    changed_id = dict(receipt)
    changed_id["receipt_id"] = changed_id["receipt_id"][:-1] + (
        "0" if changed_id["receipt_id"][-1] != "0" else "1"
    )
    assert (
        contract.phase2b_step5_external_anchor_receipt_signature_preimage_v1(
            changed_id
        )
        != preimage
    )


def test_response_loss_retry_is_byte_and_inode_idempotent(tmp_path: Path) -> None:
    graph = _build_evidence()
    spool = _private_spool(tmp_path, "retry")
    first = _issue(graph, spool)
    before = {
        name: (
            (spool / name).read_bytes(),
            (spool / name).stat().st_ino,
            (spool / name).stat().st_mtime_ns,
        )
        for name in first.spool_file_basenames_in_commit_order
    }
    second = _issue(graph, spool)
    after = {
        name: (
            (spool / name).read_bytes(),
            (spool / name).stat().st_ino,
            (spool / name).stat().st_mtime_ns,
        )
        for name in second.spool_file_basenames_in_commit_order
    }
    assert second.receipt_payload == first.receipt_payload
    assert second.artifact_ids == first.artifact_ids
    assert second.payload_sha256s == first.payload_sha256s
    assert second.spool_file_basenames_in_commit_order == (
        first.spool_file_basenames_in_commit_order
    )
    assert after == before


@pytest.mark.parametrize(
    "field,value",
    (
        ("witness_key_epoch", True),
        ("witness_sequence", True),
        ("leaf_index", False),
        ("tree_size", True),
    ),
)
def test_persisted_receipt_rejects_json_booleans_as_integers(
    tmp_path: Path, field: str, value: bool
) -> None:
    graph = _build_evidence()
    spool = _private_spool(tmp_path, field)
    payload = _install_persisted_receipt(
        graph, spool, lambda receipt: receipt.__setitem__(field, value)
    )
    assert json.loads(payload)[field] is value
    _assert_external_error(
        witness.FAIL_SCHEMA_V1, lambda: _issue(graph, spool)
    )


def test_non_bytes_policy_fails_with_stable_module_error(tmp_path: Path) -> None:
    graph = _build_evidence()
    spool = _private_spool(tmp_path, "nonbytes-policy")
    _assert_external_error(
        witness.FAIL_INPUT_V1,
        lambda: _issue(
            graph,
            spool,
            external_witness_policy_payload=graph.payloads["policy"].decode(
                "ascii"
            ),
        ),
    )


@pytest.mark.parametrize(
    "argument,code",
    (
        ("stage_b_trust_lock_payload", witness.FAIL_TRUST_ANCHOR_V1),
        ("authority_profile_payload", witness.FAIL_TRUST_ANCHOR_V1),
        ("signer_registry_payload", witness.FAIL_TRUST_ANCHOR_V1),
        ("external_witness_policy_payload", witness.FAIL_INPUT_V1),
        ("signed_owner_status_anti_fork_anchor_payload", witness.FAIL_INPUT_V1),
    ),
)
def test_all_public_raw_artifacts_reject_non_bytes_with_stable_errors(
    tmp_path: Path, argument: str, code: str
) -> None:
    graph = _build_evidence()
    spool = _private_spool(tmp_path, f"raw-{argument}")
    original = _call_arguments(graph, spool)[argument]
    assert type(original) is bytes
    _assert_external_error(
        code,
        lambda: _issue(
            graph,
            spool,
            **{argument: original.decode("ascii")},
        ),
    )


@pytest.mark.parametrize(
    "argument",
    (
        "out_of_band_expected_lock_id_bytes",
        "out_of_band_expected_lock_payload_sha256_digest",
        "out_of_band_expected_external_witness_policy_id_bytes",
        "out_of_band_expected_external_witness_policy_payload_sha256_digest",
        "expected_stage_b_trust_lock_id_bytes",
        "expected_authority_profile_id_bytes",
        "expected_signer_registry_id_bytes",
        "expected_attempt_context_id_bytes",
    ),
)
def test_every_raw_oob_or_expected_binding_is_fail_closed(
    tmp_path: Path, argument: str
) -> None:
    graph = _build_evidence()
    spool = _private_spool(tmp_path, argument)
    original = _call_arguments(graph, spool)[argument]
    assert type(original) is bytes
    replacement = _different_raw_pin(original)
    _assert_external_error(
        witness.FAIL_TRUST_ANCHOR_V1
        if argument != "expected_attempt_context_id_bytes"
        else witness.FAIL_BINDING_V1,
        lambda: _issue(graph, spool, **{argument: replacement}),
    )


@pytest.mark.parametrize(
    "field,replacement",
    (
        ("public_key_hex", "authority_public_key"),
        ("witness_key_id", "authority_key_id"),
        ("witness_control_domain_id", "authority_control_domain"),
        ("witness_namespace", "authority_role"),
    ),
)
def test_external_policy_must_be_disjoint_from_all_authority_roles(
    tmp_path: Path, field: str, replacement: str
) -> None:
    graph = _build_evidence()
    policy = copy.deepcopy(dict(graph.values["policy"]))
    role = graph.authority.role_keys[0]
    values = {
        "authority_public_key": role["public_key_hex"],
        "authority_key_id": role["key_id"],
        "authority_control_domain": role["control_domain"],
        "authority_role": authority.ROLE_ORDER_V2[0],
    }
    policy[field] = values[replacement]
    payload = _policy_payload(policy)
    policy_id = json.loads(payload)["external_witness_policy_id"]
    spool = _private_spool(tmp_path, field)
    _assert_external_error(
        witness.FAIL_SCHEMA_V1
        if field == "witness_control_domain_id"
        else witness.FAIL_TRUST_ANCHOR_V1,
        lambda: _issue(
            graph,
            spool,
            external_witness_policy_payload=payload,
            out_of_band_expected_external_witness_policy_id_bytes=policy_id.encode(
                "ascii"
            ),
            out_of_band_expected_external_witness_policy_payload_sha256_digest=(
                hashlib.sha256(payload).digest()
            ),
        ),
    )


def test_wrong_external_private_key_is_rejected(tmp_path: Path) -> None:
    graph = _build_evidence()
    spool = _private_spool(tmp_path, "wrong-key")
    wrong_key = Ed25519PrivateKey.from_private_bytes(b"z" * 32)
    _assert_external_error(
        witness.FAIL_KEY_V1,
        lambda: _issue(graph, spool, external_witness_private_key=wrong_key),
    )


@pytest.mark.parametrize("failure", ("purpose", "id", "body"))
def test_pure_signature_id_and_body_are_exact(
    tmp_path: Path, failure: str
) -> None:
    graph = _build_evidence()
    root = copy.deepcopy(dict(graph.values["anchor"]))
    if failure == "purpose":
        root["signature"]["purpose"] = "SIGN_OWNER_STATUS_CHECKPOINT"
    elif failure == "id":
        root["anchor_id"] = root["anchor_id"][:-1] + (
            "0" if root["anchor_id"][-1] != "0" else "1"
        )
    else:
        root["body"]["anchor_nonce"] = "unsigned-body-mutation"
    payload = contract.canonical_r01_evidence_contract_json_bytes_v1(root)
    spool = _private_spool(tmp_path, f"pure-{failure}")
    expected = {
        "purpose": witness.FAIL_SIGNATURE_V1,
        "id": witness.FAIL_CONTENT_ID_V1,
        "body": witness.FAIL_CONTENT_ID_V1,
    }[failure]
    _assert_external_error(
        expected,
        lambda: _issue(
            graph,
            spool,
            signed_owner_status_anti_fork_anchor_payload=payload,
        ),
    )


@pytest.mark.parametrize(
    "failure,mutate,recompute_id,resign,code",
    (
        (
            "root",
            lambda receipt: receipt.__setitem__(
                "signed_tree_root_sha256", "0" * 64
            ),
            True,
            True,
            witness.FAIL_BINDING_V1,
        ),
        (
            "predecessor",
            lambda receipt: receipt.__setitem__(
                "previous_external_receipt_id",
                "phase2b_step5_external_owner_status_anchor_receipt_v1_"
                + "1" * 64,
            ),
            True,
            True,
            witness.FAIL_BINDING_V1,
        ),
        (
            "id",
            lambda receipt: receipt.__setitem__("anchor_payload_sha256", "2" * 64),
            False,
            True,
            witness.FAIL_BINDING_V1,
        ),
        (
            "signature",
            lambda receipt: None,
            True,
            False,
            witness.FAIL_SIGNATURE_V1,
        ),
    ),
)
def test_persisted_receipt_root_predecessor_id_and_signature_fail_closed(
    tmp_path: Path,
    failure: str,
    mutate: Callable[[dict[str, object]], None],
    recompute_id: bool,
    resign: bool,
    code: str,
) -> None:
    graph = _build_evidence()
    spool = _private_spool(tmp_path, f"receipt-{failure}")
    _install_persisted_receipt(
        graph,
        spool,
        mutate,
        recompute_id=recompute_id,
        resign=resign,
    )
    _assert_external_error(code, lambda: _issue(graph, spool))


def test_crash_prefixes_poison_but_exact_complete_clone_replays(
    tmp_path: Path,
) -> None:
    graph = _build_evidence()
    source = _private_spool(tmp_path, "committed")
    result = _issue(graph, source)
    names = result.spool_file_basenames_in_commit_order
    for count in range(1, 6):
        destination = _private_spool(tmp_path, f"prefix-{count}")
        for name in names[:count]:
            shutil.copyfile(source / name, destination / name)
            (destination / name).chmod(0o600)
        if count < 5:
            _assert_external_error(
                witness.FAIL_FILESYSTEM_V1,
                lambda destination=destination: _issue(graph, destination),
            )
        else:
            replay = _issue(graph, destination)
            assert replay.receipt_payload == result.receipt_payload
            assert replay.host_rollback_resistance_verified is False


@pytest.mark.parametrize("attack", ("extra", "mode", "hardlink", "symlink"))
def test_extra_mode_link_and_symlink_spools_fail_closed(
    tmp_path: Path, attack: str
) -> None:
    graph = _build_evidence()
    spool = _private_spool(tmp_path, attack)
    if attack == "symlink":
        target = tmp_path / "outside"
        target.write_bytes(b"not-a-lock")
        os.symlink(target, spool / "external-witness-initialization-v1.lock")
    else:
        result = _issue(graph, spool)
        first = spool / result.spool_file_basenames_in_commit_order[0]
        if attack == "extra":
            (spool / "extra").write_bytes(b"x")
            (spool / "extra").chmod(0o600)
        elif attack == "mode":
            first.chmod(0o640)
        else:
            os.link(first, spool / "hardlink")
    _assert_external_error(
        witness.FAIL_FILESYSTEM_V1, lambda: _issue(graph, spool)
    )


def test_concurrent_initialization_is_exclusive_and_recovers_by_retry(
    tmp_path: Path,
) -> None:
    graph = _build_evidence()
    spool = _private_spool(tmp_path, "concurrent")

    def invoke():
        try:
            return _issue(graph, spool)
        except witness.Phase2BStep5R01ExternalWitnessErrorV1 as exc:
            assert exc.code == witness.FAIL_FILESYSTEM_V1
            return exc

    with ThreadPoolExecutor(max_workers=4) as executor:
        outcomes = list(executor.map(lambda _index: invoke(), range(4)))
    successes = [
        value
        for value in outcomes
        if isinstance(value, witness.IssuedInitialExternalWitnessCandidateV1)
    ]
    assert successes
    retry = _issue(graph, spool)
    assert retry.receipt_payload == graph.payloads["external_receipt"]
    assert len(tuple(spool.iterdir())) == 5


def test_same_policy_can_fork_two_sequence_one_local_spools(tmp_path: Path) -> None:
    graph = _build_evidence()
    first_spool = _private_spool(tmp_path, "fork-a")
    second_spool = _private_spool(tmp_path, "fork-b")
    first = _issue(graph, first_spool)
    second_anchor = _resigned_anchor(
        graph, lambda body: body.__setitem__("anchor_nonce", "fork-b")
    )
    second = _issue(
        graph,
        second_spool,
        signed_owner_status_anti_fork_anchor_payload=second_anchor,
    )
    first_receipt = json.loads(first.receipt_payload)
    second_receipt = json.loads(second.receipt_payload)
    assert first_receipt["external_witness_policy_id"] == second_receipt[
        "external_witness_policy_id"
    ]
    assert first_receipt["witness_sequence"] == second_receipt["witness_sequence"] == 1
    assert first_receipt["receipt_id"] != second_receipt["receipt_id"]
    assert first_receipt["signed_tree_root_sha256"] != second_receipt[
        "signed_tree_root_sha256"
    ]
    assert first.host_rollback_resistance_verified is False
    assert second.external_anti_fork_anchor_verified is False
    _assert_external_error(
        witness.FAIL_BINDING_V1,
        lambda: _issue(
            graph,
            first_spool,
            signed_owner_status_anti_fork_anchor_payload=second_anchor,
        ),
    )


def test_result_lineage_is_private_and_every_nonmechanical_claim_is_false(
    tmp_path: Path,
) -> None:
    graph = _build_evidence()
    result = _issue(graph, _private_spool(tmp_path, "claims"))
    assert result.raw_authority_signature_mechanics_verified is True
    assert result.receipt_signature_mechanics_verified is True
    assert result.local_fsync_readback_mechanics_verified is True
    false_fields = (
        "same_uid_isolation_verified",
        "independent_or_local_filesystem_verified",
        "host_rollback_resistance_verified",
        "key_custody_verified",
        "attempt_freshness_verified",
        "external_live_service_verified",
        "external_control_verified",
        "external_currentness_verified",
        "external_anti_fork_anchor_verified",
        "durability_established",
        "authority_established",
        "r01_satisfied",
        "readiness_satisfied",
        "execution_authorized",
    )
    assert all(getattr(result, name) is False for name in false_fields)
    assert set(false_fields) < {
        field.name
        for field in fields(witness.IssuedInitialExternalWitnessCandidateV1)
    }
    with pytest.raises(TypeError):
        witness.IssuedInitialExternalWitnessCandidateV1()
    forged = object.__new__(witness.IssuedInitialExternalWitnessCandidateV1)
    _assert_external_error(
        witness.FAIL_INPUT_V1,
        lambda: witness.require_issued_initial_external_witness_candidate_v1(
            forged
        ),
    )
    object.__setattr__(result, "execution_authorized", True)
    _assert_external_error(
        witness.FAIL_INPUT_V1,
        lambda: witness.require_issued_initial_external_witness_candidate_v1(
            result
        ),
    )


def test_public_entry_isolated_from_post_import_global_rebinding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = _build_evidence()
    monkeypatch.setattr(witness, "_MERKLE_LEAF_DOMAIN", b"rebound")
    monkeypatch.setattr(witness, "_load_raw_authority_graph", lambda *a, **k: ())
    monkeypatch.setattr(authority, "load_step5_stage_b_trust_lock_v2", None)
    monkeypatch.setattr(
        contract,
        "phase2b_step5_external_anchor_receipt_signature_preimage_v1",
        None,
    )
    result = _issue(graph, _private_spool(tmp_path, "captured"))
    assert result.receipt_payload == graph.payloads["external_receipt"]


def test_real_external_admission_remains_not_implemented() -> None:
    _assert_external_error(
        witness.FAIL_NOT_IMPLEMENTED_V1,
        lambda: witness.verify_real_external_witness_admission_v1(),
    )


def test_policy_epoch_boolean_and_schema_copy_drift_are_rejected(
    tmp_path: Path,
) -> None:
    graph = _build_evidence()
    for name, mutate in (
        (
            "bool-epoch",
            lambda policy: policy.__setitem__("witness_key_epoch", True),
        ),
        ("extra-field", lambda policy: policy.__setitem__("copied_field", False)),
    ):
        policy = copy.deepcopy(dict(graph.values["policy"]))
        mutate(policy)
        payload = _policy_payload(policy)
        policy_id = json.loads(payload)["external_witness_policy_id"]
        spool = _private_spool(tmp_path, name)
        _assert_external_error(
            witness.FAIL_SCHEMA_V1,
            lambda payload=payload, policy_id=policy_id, spool=spool: _issue(
                graph,
                spool,
                external_witness_policy_payload=payload,
                out_of_band_expected_external_witness_policy_id_bytes=(
                    policy_id.encode("ascii")
                ),
                out_of_band_expected_external_witness_policy_payload_sha256_digest=(
                    hashlib.sha256(payload).digest()
                ),
            ),
        )
