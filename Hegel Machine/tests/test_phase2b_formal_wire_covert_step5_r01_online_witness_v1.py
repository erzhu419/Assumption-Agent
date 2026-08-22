from __future__ import annotations

import copy
from dataclasses import dataclass, fields
import hashlib
import importlib
import inspect
import itertools
import json
import os
import pickle
from pathlib import Path
import subprocess
import sys
import textwrap
from types import FunctionType, MappingProxyType
from typing import Callable, Mapping

import pytest

from hegel_machine import (
    phase2b_formal_wire_covert_step5_r01_durable_anchor_v1 as durable_anchor,
)
from hegel_machine import (
    phase2b_formal_wire_covert_step5_r01_external_witness_admission_contract_v1
    as admission,
)
from hegel_machine import (
    phase2b_formal_wire_covert_step5_r01_online_witness_v1 as online_witness,
)
from test_phase2b_formal_wire_covert_step5_r01_external_witness_admission_contract_v1 import (
    _PositiveCorpus,
    _common_body,
    _positive_corpus,
    _sha,
    _signed_admission_object,
)


_CORPUS_SALTS = itertools.count(21)


class _EqAll:
    def __eq__(self, other: object) -> bool:
        return True


@dataclass(frozen=True)
class _ActorFixture:
    corpus: _PositiveCorpus
    actor: object
    observation: Mapping[str, object]
    root: Path


def _captured_method_dependency(method: Callable[..., object], name: str) -> object:
    closure = method.__closure__
    assert closure is not None
    captured = dict(zip(method.__code__.co_freevars, closure, strict=True))
    return captured[name].cell_contents


def _private_implementation_globals() -> dict[str, object]:
    implementation = _captured_method_dependency(
        online_witness.R01OnlineWitnessActorV1.begin_prepare, "begin_prepare"
    )
    assert type(implementation) is FunctionType
    return implementation.__globals__


def _construct_actor(corpus: _PositiveCorpus) -> object:
    call = corpus.call
    return online_witness.R01OnlineWitnessActorV1(
        call["stage_b_trust_lock_payload"],
        call["authority_profile_payload"],
        call["signer_registry_payload"],
        call["bootstrap_external_witness_policy_payload"],
        call["online_witness_policy_payload"],
        call["durable_anchor_policy_payload"],
        call["candidate_anchor_payload"],
        call["candidate_receipt_payload"],
        call["candidate_ledger_payload"],
        call["candidate_checkpoint_payload"],
        call["candidate_head_payload"],
        out_of_band_expected_lock_id_bytes=call[
            "out_of_band_expected_lock_id_bytes"
        ],
        out_of_band_expected_lock_payload_sha256_digest=call[
            "out_of_band_expected_lock_payload_sha256_digest"
        ],
        out_of_band_expected_bootstrap_external_witness_policy_id_bytes=call[
            "out_of_band_expected_bootstrap_external_witness_policy_id_bytes"
        ],
        out_of_band_expected_bootstrap_external_witness_policy_payload_sha256_digest=call[
            "out_of_band_expected_bootstrap_external_witness_policy_payload_sha256_digest"
        ],
        out_of_band_expected_online_witness_policy_id_bytes=call[
            "out_of_band_expected_online_witness_policy_id_bytes"
        ],
        out_of_band_expected_online_witness_policy_payload_sha256_digest=call[
            "out_of_band_expected_online_witness_policy_payload_sha256_digest"
        ],
        out_of_band_expected_durable_anchor_policy_id_bytes=call[
            "out_of_band_expected_durable_anchor_policy_id_bytes"
        ],
        out_of_band_expected_durable_anchor_policy_payload_sha256_digest=call[
            "out_of_band_expected_durable_anchor_policy_payload_sha256_digest"
        ],
        expected_attempt_context_id_bytes=call[
            "expected_attempt_context_id_bytes"
        ],
        online_witness_private_key=corpus.online_private_key,
    )


@pytest.fixture
def actor_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[[], _ActorFixture]:
    def make() -> _ActorFixture:
        salt = next(_CORPUS_SALTS)
        root = tmp_path / f"online-witness-{salt}"
        root.mkdir(mode=0o700)
        corpus = _positive_corpus(root, salt=salt)
        policy = corpus.roots["online_policy"]["body"]
        observation = {
            "pid": 7200 + salt,
            "uid": policy["service_uid"],
            "gid": policy["service_gid"],
            "process_start_ticks": 7300 + salt,
            "user_namespace_inode": policy["service_user_namespace_inode"],
        }
        private_globals = _private_implementation_globals()
        monkeypatch.setitem(
            private_globals, "_observe_process", lambda: dict(observation)
        )
        return _ActorFixture(
            corpus=corpus,
            actor=_construct_actor(corpus),
            observation=MappingProxyType(dict(observation)),
            root=root,
        )

    return make


def _common(corpus: _PositiveCorpus) -> dict[str, object]:
    return _common_body(
        corpus.graph,
        bootstrap_policy_id=corpus.roots["bootstrap_policy"][
            "external_witness_policy_id"
        ],
        online_policy_id=corpus.roots["online_policy"]["online_witness_policy_id"],
        durable_policy_id=corpus.roots["durable_policy"][
            "durable_anchor_policy_id"
        ],
    )


def _expected_runtime_binding(
    corpus: _PositiveCorpus, observation: Mapping[str, object]
) -> dict[str, object]:
    policy = corpus.roots["online_policy"]["body"]
    runtime_subject = {
        "schema_version": (
            "hegel-machine-phase2b-step5-online-witness-local-runtime-observation/1"
        ),
        "service_binding_id": policy["service_binding_id"],
        "runtime_closure_policy_id": policy["runtime_closure_policy_id"],
        "pid": observation["pid"],
        "uid": observation["uid"],
        "gid": observation["gid"],
        "process_start_ticks": observation["process_start_ticks"],
        "user_namespace_inode": observation["user_namespace_inode"],
        "control_domain_id": policy["witness_control_domain_id"],
    }
    runtime_bytes = admission.canonical_external_witness_admission_json_bytes_v1(
        runtime_subject
    )
    runtime_digest = hashlib.sha256(runtime_bytes).hexdigest()
    custody_subject = {
        "schema_version": (
            "hegel-machine-phase2b-step5-online-witness-local-key-observation/1"
        ),
        "custody_attestation_policy_id": policy[
            "custody_attestation_policy_id"
        ],
        "online_key_id": policy["online_key_id"],
        "online_public_key_hex": policy["online_public_key_hex"],
        "process_start_ticks": observation["process_start_ticks"],
    }
    custody_bytes = admission.canonical_external_witness_admission_json_bytes_v1(
        custody_subject
    )
    custody_digest = hashlib.sha256(custody_bytes).hexdigest()
    return {
        "service_binding_id": policy["service_binding_id"],
        "runtime_instance_id": (
            "phase2b_step5_online_witness_local_runtime_v1_" + runtime_digest
        ),
        "peer_pid": observation["pid"],
        "peer_uid": observation["uid"],
        "peer_gid": observation["gid"],
        "user_namespace_inode": observation["user_namespace_inode"],
        "process_start_ticks": observation["process_start_ticks"],
        "control_domain_id": policy["witness_control_domain_id"],
        "runtime_closure_policy_id": policy["runtime_closure_policy_id"],
        "runtime_closure_report_id": (
            "phase2b_step5_online_witness_local_runtime_report_v1_"
            + runtime_digest
        ),
        "runtime_closure_report_payload_sha256": runtime_digest,
        "custody_attestation_policy_id": policy[
            "custody_attestation_policy_id"
        ],
        "custody_attestation_id": (
            "phase2b_step5_online_witness_local_custody_observation_v1_"
            + custody_digest
        ),
        "custody_attestation_payload_sha256": custody_digest,
        "online_key_nonexportable": True,
    }


def _expected_prepare_payloads(
    fixture: _ActorFixture, prepared: object
) -> tuple[bytes, bytes, str, Mapping[str, object]]:
    corpus = fixture.corpus
    q1 = corpus.roots["q1_request"]
    q1_body = q1["body"]
    actual_q1_response = json.loads(prepared.q1_response_payload)
    confirmation_nonce = actual_q1_response["body"]["confirmation_nonce"]
    runtime_binding = _expected_runtime_binding(corpus, fixture.observation)
    online_policy = corpus.roots["online_policy"]["body"]
    common = _common(corpus)
    response_body = {
        **common,
        "schema_version": admission.Q1_PREPARE_RESPONSE_SCHEMA_V1,
        "q1_request_id": q1["q1_request_id"],
        "q1_request_payload_sha256": _sha(corpus.payloads["q1_request"]),
        "prepare_nonce": q1_body["prepare_nonce"],
        "confirmation_nonce": confirmation_nonce,
        "observed_current_mapping": copy.deepcopy(corpus.candidate_mapping),
        "prepared_candidate_mapping": copy.deepcopy(corpus.candidate_mapping),
        "runtime_custody_binding": copy.deepcopy(runtime_binding),
        "prepare_outcome": admission.PREPARE_OUTCOME_V1,
    }
    expected_q1_response, expected_q1_root = _signed_admission_object(
        artifact_kind="q1_prepare_response",
        schema=admission.Q1_PREPARE_RESPONSE_SCHEMA_V1,
        id_field="q1_response_id",
        body=response_body,
        private_key=corpus.online_private_key,
        signer_key_id=online_policy["online_key_id"],
        purpose=admission.SIGN_Q1_PREPARE_RESPONSE_PURPOSE_V1,
    )
    durable_policy = corpus.roots["durable_policy"]["body"]
    a_request_body = {
        **common,
        "schema_version": admission.A_COMPARE_AND_APPEND_REQUEST_SCHEMA_V1,
        "q1_request_id": q1["q1_request_id"],
        "q1_request_payload_sha256": _sha(corpus.payloads["q1_request"]),
        "q1_response_id": expected_q1_root["q1_response_id"],
        "q1_response_payload_sha256": _sha(expected_q1_response),
        "prepare_nonce": q1_body["prepare_nonce"],
        "confirmation_nonce": confirmation_nonce,
        "expected_current_mapping": copy.deepcopy(corpus.genesis_mapping),
        "proposed_current_mapping": copy.deepcopy(corpus.candidate_mapping),
        "expected_previous_durable_checkpoint_id": durable_policy[
            "oob_high_water_checkpoint_id"
        ],
        "expected_previous_durable_checkpoint_payload_sha256": durable_policy[
            "oob_high_water_checkpoint_payload_sha256"
        ],
        "expected_previous_durable_head_id": durable_policy[
            "oob_high_water_head_id"
        ],
        "expected_previous_durable_head_payload_sha256": durable_policy[
            "oob_high_water_head_payload_sha256"
        ],
        "expected_previous_durable_checkpoint_sequence": durable_policy[
            "oob_high_water_checkpoint_sequence"
        ],
        "expected_previous_durable_head_sequence": durable_policy[
            "oob_high_water_head_sequence"
        ],
        "expected_previous_durable_tree_root_sha256": durable_policy[
            "oob_high_water_tree_root_sha256"
        ],
        "compare_and_append_mode": (
            "ATOMIC_COMPARE_EXACT_OOB_HIGH_WATER_THEN_APPEND_ONCE"
        ),
    }
    expected_a_request, _ = _signed_admission_object(
        artifact_kind="a_compare_and_append_request",
        schema=admission.A_COMPARE_AND_APPEND_REQUEST_SCHEMA_V1,
        id_field="a_request_id",
        body=a_request_body,
        private_key=corpus.online_private_key,
        signer_key_id=online_policy["online_key_id"],
        purpose=admission.SIGN_A_COMPARE_AND_APPEND_REQUEST_PURPOSE_V1,
    )
    return expected_q1_response, expected_a_request, confirmation_nonce, runtime_binding


def _a_outputs(
    fixture: _ActorFixture, prepared: object
) -> object:
    corpus = fixture.corpus
    call = corpus.call
    spool = fixture.root / "durable-anchor-spool"
    spool.mkdir(mode=0o700)
    os.chmod(spool, 0o700)
    actor = durable_anchor.construct_local_durable_anchor_actor_v1(
        corpus.payloads["bootstrap_policy"],
        corpus.payloads["online_policy"],
        corpus.payloads["durable_policy"],
        out_of_band_expected_bootstrap_external_witness_policy_id_bytes=call[
            "out_of_band_expected_bootstrap_external_witness_policy_id_bytes"
        ],
        out_of_band_expected_bootstrap_external_witness_policy_payload_sha256_digest=call[
            "out_of_band_expected_bootstrap_external_witness_policy_payload_sha256_digest"
        ],
        out_of_band_expected_online_witness_policy_id_bytes=call[
            "out_of_band_expected_online_witness_policy_id_bytes"
        ],
        out_of_band_expected_online_witness_policy_payload_sha256_digest=call[
            "out_of_band_expected_online_witness_policy_payload_sha256_digest"
        ],
        out_of_band_expected_durable_anchor_policy_id_bytes=call[
            "out_of_band_expected_durable_anchor_policy_id_bytes"
        ],
        out_of_band_expected_durable_anchor_policy_payload_sha256_digest=call[
            "out_of_band_expected_durable_anchor_policy_payload_sha256_digest"
        ],
        durable_anchor_private_key=corpus.durable_private_key,
        owner_private_spool_directory_path=str(spool),
    )
    return durable_anchor.compare_and_append_local_durable_anchor_v1(
        actor,
        corpus.payloads["q1_request"],
        prepared.q1_response_payload,
        prepared.a_compare_and_append_request_payload,
    )


def _q2_request(
    fixture: _ActorFixture, prepared: object, anchored: object
) -> tuple[bytes, Mapping[str, object]]:
    corpus = fixture.corpus
    q1_response = json.loads(prepared.q1_response_payload)
    a_request = json.loads(prepared.a_compare_and_append_request_payload)
    checkpoint = json.loads(anchored.a_durable_checkpoint_payload)
    head = json.loads(anchored.a_durable_head_payload)
    receipt = json.loads(anchored.a_compare_and_append_receipt_payload)
    q1 = corpus.roots["q1_request"]
    body = {
        **_common(corpus),
        "schema_version": admission.Q2_CONFIRM_REQUEST_SCHEMA_V1,
        "q1_response_id": q1_response["q1_response_id"],
        "q1_response_payload_sha256": _sha(prepared.q1_response_payload),
        "a_request_id": a_request["a_request_id"],
        "a_request_payload_sha256": _sha(
            prepared.a_compare_and_append_request_payload
        ),
        "a_receipt_id": receipt["a_receipt_id"],
        "a_receipt_payload_sha256": _sha(
            anchored.a_compare_and_append_receipt_payload
        ),
        "a_checkpoint_id": checkpoint["a_checkpoint_id"],
        "a_checkpoint_payload_sha256": _sha(anchored.a_durable_checkpoint_payload),
        "a_head_id": head["a_head_id"],
        "a_head_payload_sha256": _sha(anchored.a_durable_head_payload),
        "prepare_nonce": q1["body"]["prepare_nonce"],
        "confirmation_nonce": q1_response["body"]["confirmation_nonce"],
        "expected_current_mapping": copy.deepcopy(corpus.candidate_mapping),
        "expected_cas_outcome": admission.CAS_OUTCOME_V1,
    }
    bootstrap = corpus.roots["bootstrap_policy"]
    return _signed_admission_object(
        artifact_kind="q2_confirm_request",
        schema=admission.Q2_CONFIRM_REQUEST_SCHEMA_V1,
        id_field="q2_request_id",
        body=body,
        private_key=corpus.graph.witness_private_key,
        signer_key_id=bootstrap["witness_key_id"],
        purpose=admission.SIGN_Q2_CONFIRM_REQUEST_PURPOSE_V1,
    )


def _expected_q2_response(
    fixture: _ActorFixture,
    q2_payload: bytes,
    q2_root: Mapping[str, object],
    anchored: object,
    runtime_binding: Mapping[str, object],
) -> bytes:
    corpus = fixture.corpus
    receipt = json.loads(anchored.a_compare_and_append_receipt_payload)
    online_policy = corpus.roots["online_policy"]["body"]
    body = {
        **_common(corpus),
        "schema_version": admission.Q2_CONFIRM_RESPONSE_SCHEMA_V1,
        "q2_request_id": q2_root["q2_request_id"],
        "q2_request_payload_sha256": _sha(q2_payload),
        "a_receipt_id": receipt["a_receipt_id"],
        "a_receipt_payload_sha256": _sha(
            anchored.a_compare_and_append_receipt_payload
        ),
        "prepare_nonce": q2_root["body"]["prepare_nonce"],
        "confirmation_nonce": q2_root["body"]["confirmation_nonce"],
        "observed_current_mapping": copy.deepcopy(corpus.candidate_mapping),
        "runtime_custody_binding": copy.deepcopy(runtime_binding),
        "observed_cas_outcome": admission.CAS_OUTCOME_V1,
        "confirm_outcome": admission.CONFIRM_OUTCOME_V1,
    }
    payload, _ = _signed_admission_object(
        artifact_kind="q2_confirm_response",
        schema=admission.Q2_CONFIRM_RESPONSE_SCHEMA_V1,
        id_field="q2_response_id",
        body=body,
        private_key=corpus.online_private_key,
        signer_key_id=online_policy["online_key_id"],
        purpose=admission.SIGN_Q2_CONFIRM_RESPONSE_PURPOSE_V1,
    )
    return payload


def _different_q1(corpus: _PositiveCorpus) -> bytes:
    body = copy.deepcopy(dict(corpus.roots["q1_request"]["body"]))
    body["prepare_nonce"] = "e" * 64
    bootstrap = corpus.roots["bootstrap_policy"]
    payload, _ = _signed_admission_object(
        artifact_kind="q1_prepare_request",
        schema=admission.Q1_PREPARE_REQUEST_SCHEMA_V1,
        id_field="q1_request_id",
        body=body,
        private_key=corpus.graph.witness_private_key,
        signer_key_id=bootstrap["witness_key_id"],
        purpose=admission.SIGN_Q1_PREPARE_REQUEST_PURPOSE_V1,
    )
    return payload


def _assert_all_elevated_false(result: object) -> None:
    elevated = {
        "distinct_live_uid_gid_verified",
        "live_transport_verified",
        "durability_verified",
        "non_equivocation_verified",
        "global_anti_fork_verified",
        "logical_time_ancestry_verified",
        "restart_recovery_verified",
        "durable_session_verified",
        "live_external_currentness_verified",
        "runtime_closure_verified",
        "key_custody_verified",
        "authority_evidence",
        "r01_satisfied",
        "readiness_evidence",
        "execution_authorized",
    }
    names = {field.name for field in fields(result)}
    for name in sorted(elevated & names):
        assert getattr(result, name) is False


def test_exact_b480_q1_a_q2_differential_with_actual_local_a_outputs(
    actor_factory: Callable[[], _ActorFixture],
) -> None:
    fixture = actor_factory()
    corpus = fixture.corpus
    prepared = fixture.actor.begin_prepare(corpus.payloads["q1_request"])
    assert online_witness.require_prepared_r01_online_witness_mechanics_v1(
        prepared
    ) is prepared
    expected_q1, expected_a, confirmation_nonce, runtime_binding = (
        _expected_prepare_payloads(fixture, prepared)
    )
    assert prepared.q1_response_payload == expected_q1
    assert prepared.a_compare_and_append_request_payload == expected_a
    assert confirmation_nonce != corpus.roots["q1_request"]["body"]["prepare_nonce"]
    assert confirmation_nonce != "0" * 64
    assert len(confirmation_nonce) == 64
    assert inspect.signature(online_witness.R01OnlineWitnessActorV1).parameters.get(
        "confirmation_nonce"
    ) is None

    response = json.loads(prepared.q1_response_payload)
    a_request = json.loads(prepared.a_compare_and_append_request_payload)
    for mapping in (
        response["body"]["observed_current_mapping"],
        response["body"]["prepared_candidate_mapping"],
        a_request["body"]["proposed_current_mapping"],
    ):
        assert mapping == corpus.candidate_mapping
        assert mapping["effective_state"] == "CURRENT_UNSUPERSEDED"
        assert mapping["superseded"] is False
        assert mapping["revoked"] is False
        assert mapping["map_leaf_index"] == 0
        assert mapping["map_tree_size"] == 1
        assert mapping["map_inclusion_path_sha256_in_exact_order"] == []
        assert mapping[
            "presented_to_live_consistency_path_sha256_in_exact_order"
        ] == []
    assert a_request["body"]["expected_current_mapping"] == corpus.genesis_mapping
    assert response["body"]["runtime_custody_binding"] == runtime_binding
    assert prepared.runtime_binding_self_report_only is True
    _assert_all_elevated_false(prepared)

    anchored = _a_outputs(fixture, prepared)
    assert durable_anchor.require_issued_local_durable_anchor_append_candidate_v1(
        anchored
    ) is anchored
    q2_payload, q2_root = _q2_request(fixture, prepared, anchored)
    completed = fixture.actor.confirm(
        anchored.a_durable_checkpoint_payload,
        anchored.a_durable_head_payload,
        anchored.a_compare_and_append_receipt_payload,
        q2_payload,
    )
    assert online_witness.require_completed_r01_online_witness_mechanics_v1(
        completed
    ) is completed
    assert completed.q2_response_payload == _expected_q2_response(
        fixture, q2_payload, q2_root, anchored, runtime_binding
    )
    q2_response = json.loads(completed.q2_response_payload)
    assert q2_response["body"]["observed_current_mapping"] == corpus.candidate_mapping
    assert q2_response["body"]["runtime_custody_binding"] == runtime_binding
    assert completed.runtime_binding_self_report_only is True
    assert completed.exact_a_triple_verified is True
    assert completed.exact_bootstrap_q2_request_verified is True
    _assert_all_elevated_false(completed)
    object.__setattr__(completed, "q2_response_id", _EqAll())
    with pytest.raises(online_witness.Phase2BStep5R01OnlineWitnessErrorV1) as mutated:
        online_witness.require_completed_r01_online_witness_mechanics_v1(completed)
    assert mutated.value.code == online_witness.FAIL_RESULT_V1


def test_exact_q1_retry_is_byte_identical_and_valid_cross_request_poisons_prepared_slot(
    actor_factory: Callable[[], _ActorFixture],
) -> None:
    fixture = actor_factory()
    q1 = fixture.corpus.payloads["q1_request"]
    first = fixture.actor.begin_prepare(q1)
    replay = fixture.actor.begin_prepare(q1)
    assert replay.replayed_existing_session is True
    assert replay.q1_response_payload == first.q1_response_payload
    assert (
        replay.a_compare_and_append_request_payload
        == first.a_compare_and_append_request_payload
    )
    with pytest.raises(online_witness.Phase2BStep5R01OnlineWitnessErrorV1) as conflict:
        fixture.actor.begin_prepare(_different_q1(fixture.corpus))
    assert conflict.value.code == online_witness.FAIL_FENCE_V1
    with pytest.raises(online_witness.Phase2BStep5R01OnlineWitnessErrorV1) as poisoned:
        fixture.actor.begin_prepare(q1)
    assert poisoned.value.code == online_witness.FAIL_POISONED_V1


def test_malformed_cross_request_cannot_dos_poison_prepared_slot(
    actor_factory: Callable[[], _ActorFixture],
) -> None:
    fixture = actor_factory()
    q1 = fixture.corpus.payloads["q1_request"]
    first = fixture.actor.begin_prepare(q1)
    with pytest.raises(online_witness.Phase2BStep5R01OnlineWitnessErrorV1):
        fixture.actor.begin_prepare(b"{}")
    replay = fixture.actor.begin_prepare(q1)
    assert replay.replayed_existing_session is True
    assert replay.q1_response_payload == first.q1_response_payload


def test_terminal_exact_retry_is_cached_and_nonexact_terminal_input_is_only_conflict(
    actor_factory: Callable[[], _ActorFixture], monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = actor_factory()
    prepared = fixture.actor.begin_prepare(fixture.corpus.payloads["q1_request"])
    anchored = _a_outputs(fixture, prepared)
    q2_payload, _ = _q2_request(fixture, prepared, anchored)
    first = fixture.actor.confirm(
        *anchored.payloads_in_contract_order, q2_payload
    )
    with pytest.raises(online_witness.Phase2BStep5R01OnlineWitnessErrorV1) as conflict:
        fixture.actor.confirm(*anchored.payloads_in_contract_order, q2_payload + b" ")
    assert conflict.value.code == online_witness.FAIL_FENCE_V1

    def bomb() -> object:
        raise AssertionError("terminal exact retry must not requery local process state")

    monkeypatch.setitem(_private_implementation_globals(), "_observe_process", bomb)
    replay = fixture.actor.confirm(*anchored.payloads_in_contract_order, q2_payload)
    assert replay.replayed_terminal_result is True
    assert replay.q2_response_payload == first.q2_response_payload
    _assert_all_elevated_false(replay)


def test_raw_policy_authority_oob_and_signature_rebinding_fail_closed(
    actor_factory: Callable[[], _ActorFixture],
) -> None:
    fixture = actor_factory()
    corpus = fixture.corpus
    call = dict(corpus.call)
    constructor_args = [
        call["stage_b_trust_lock_payload"],
        call["authority_profile_payload"],
        call["signer_registry_payload"],
        call["bootstrap_external_witness_policy_payload"],
        call["online_witness_policy_payload"],
        call["durable_anchor_policy_payload"],
        call["candidate_anchor_payload"],
        call["candidate_receipt_payload"],
        call["candidate_ledger_payload"],
        call["candidate_checkpoint_payload"],
        call["candidate_head_payload"],
    ]
    keywords = {
        name: call[name]
        for name in (
            "out_of_band_expected_lock_id_bytes",
            "out_of_band_expected_lock_payload_sha256_digest",
            "out_of_band_expected_bootstrap_external_witness_policy_id_bytes",
            "out_of_band_expected_bootstrap_external_witness_policy_payload_sha256_digest",
            "out_of_band_expected_online_witness_policy_id_bytes",
            "out_of_band_expected_online_witness_policy_payload_sha256_digest",
            "out_of_band_expected_durable_anchor_policy_id_bytes",
            "out_of_band_expected_durable_anchor_policy_payload_sha256_digest",
            "expected_attempt_context_id_bytes",
        )
    }
    keywords["online_witness_private_key"] = corpus.online_private_key
    for index in (0, 1, 2, 3, 4, 5):
        tampered = list(constructor_args)
        tampered[index] += b" "
        with pytest.raises(online_witness.Phase2BStep5R01OnlineWitnessErrorV1):
            online_witness.R01OnlineWitnessActorV1(*tampered, **keywords)
    bad_keywords = dict(keywords)
    bad_keywords[
        "out_of_band_expected_online_witness_policy_payload_sha256_digest"
    ] = b"\x01" * 32
    with pytest.raises(online_witness.Phase2BStep5R01OnlineWitnessErrorV1) as pin:
        online_witness.R01OnlineWitnessActorV1(*constructor_args, **bad_keywords)
    assert pin.value.code == online_witness.FAIL_TRUST_ANCHOR_V1

    body = copy.deepcopy(dict(corpus.roots["q1_request"]["body"]))
    wrong_domain, _ = _signed_admission_object(
        artifact_kind="q1_prepare_request",
        schema=admission.Q1_PREPARE_REQUEST_SCHEMA_V1,
        id_field="q1_request_id",
        body=body,
        private_key=corpus.graph.witness_private_key,
        signer_key_id=corpus.roots["bootstrap_policy"]["witness_key_id"],
        purpose=admission.SIGN_Q2_CONFIRM_REQUEST_PURPOSE_V1,
    )
    with pytest.raises(online_witness.Phase2BStep5R01OnlineWitnessErrorV1) as signature:
        fixture.actor.begin_prepare(wrong_domain)
    assert signature.value.code == online_witness.FAIL_SIGNATURE_V1


def test_actor_has_no_state_attributes_and_forge_subclass_copy_pickle_fail(
    actor_factory: Callable[[], _ActorFixture],
) -> None:
    fixture = actor_factory()
    actor = fixture.actor
    assert type(actor).__slots__ == ()
    assert not hasattr(actor, "__dict__")
    with pytest.raises(TypeError):
        vars(actor)
    for name in (
        "context",
        "_context",
        "state",
        "_state",
        "private_key",
        "online_witness_private_key",
        "candidate_mapping",
        "raw_payloads",
    ):
        assert not hasattr(actor, name)
    with pytest.raises(AttributeError):
        actor.private_key = fixture.corpus.online_private_key
    with pytest.raises(online_witness.Phase2BStep5R01OnlineWitnessErrorV1):
        type("ForgedOnlineWitnessSubclass", (type(actor),), {})
    forged = object.__new__(type(actor))
    with pytest.raises(online_witness.Phase2BStep5R01OnlineWitnessErrorV1):
        forged.begin_prepare(fixture.corpus.payloads["q1_request"])
    for operation in (
        lambda: copy.copy(actor),
        lambda: copy.deepcopy(actor),
        lambda: pickle.dumps(actor),
    ):
        with pytest.raises(online_witness.Phase2BStep5R01OnlineWitnessErrorV1):
            operation()


def test_result_private_lineage_is_identity_exact_and_eqall_safe(
    actor_factory: Callable[[], _ActorFixture],
) -> None:
    fixture = actor_factory()
    result = fixture.actor.begin_prepare(fixture.corpus.payloads["q1_request"])
    require = online_witness.require_prepared_r01_online_witness_mechanics_v1
    assert require(result) is result
    with pytest.raises(TypeError):
        online_witness.PreparedR01OnlineWitnessMechanicsV1()
    forged = object.__new__(online_witness.PreparedR01OnlineWitnessMechanicsV1)
    for field in fields(result):
        object.__setattr__(forged, field.name, getattr(result, field.name))
    with pytest.raises(online_witness.Phase2BStep5R01OnlineWitnessErrorV1):
        require(forged)
    for operation in (lambda: copy.copy(result), lambda: copy.deepcopy(result)):
        with pytest.raises(TypeError, match="privately issued"):
            operation()
    with pytest.raises((TypeError, online_witness.Phase2BStep5R01OnlineWitnessErrorV1)):
        pickle.loads(pickle.dumps(result))
    object.__setattr__(result, "q1_response_id", _EqAll())
    with pytest.raises(online_witness.Phase2BStep5R01OnlineWitnessErrorV1) as mutated:
        require(result)
    assert mutated.value.code == online_witness.FAIL_RESULT_V1


def test_definition_capture_survives_module_and_transitive_global_rebinding(
    actor_factory: Callable[[], _ActorFixture], monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = actor_factory()

    def bomb(*args: object, **kwargs: object) -> object:
        raise AssertionError("rebound module/transitive helper was consulted")

    for name in (
        "_load_context",
        "_begin_prepare",
        "_sign_protocol",
        "_admission_preimage",
        "_TOKEN_BYTES",
    ):
        monkeypatch.setattr(online_witness, name, bomb, raising=False)
    monkeypatch.setattr(
        admission,
        "phase2b_step5_external_witness_admission_signature_preimage_v1",
        bomb,
    )
    require = online_witness.require_prepared_r01_online_witness_mechanics_v1
    result = fixture.actor.begin_prepare(fixture.corpus.payloads["q1_request"])
    monkeypatch.setattr(
        online_witness, "PreparedR01OnlineWitnessMechanicsV1", _EqAll
    )
    monkeypatch.setattr(
        online_witness, "Phase2BStep5R01OnlineWitnessErrorV1", RuntimeError
    )
    assert require(result) is result
    assert result.q1_response_payload.startswith(b"{")


def test_restart_has_no_session_recovery(
    actor_factory: Callable[[], _ActorFixture], monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = actor_factory()
    prepared = fixture.actor.begin_prepare(fixture.corpus.payloads["q1_request"])
    anchored = _a_outputs(fixture, prepared)
    q2_payload, _ = _q2_request(fixture, prepared, anchored)
    fresh_module = importlib.reload(online_witness)
    policy = fixture.corpus.roots["online_policy"]["body"]
    observation = dict(fixture.observation)
    private_globals = _private_implementation_globals()
    monkeypatch.setitem(
        private_globals, "_observe_process", lambda: dict(observation)
    )
    assert observation["uid"] == policy["service_uid"]
    fresh_actor = _construct_actor(fixture.corpus)
    with pytest.raises(fresh_module.Phase2BStep5R01OnlineWitnessErrorV1) as missing:
        fresh_actor.confirm(*anchored.payloads_in_contract_order, q2_payload)
    assert missing.value.code == fresh_module.FAIL_FENCE_V1


def test_clean_import_surface_and_real_admission_stays_not_implemented() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src"
    script = textwrap.dedent(
        """
        import sys
        import hegel_machine.phase2b_formal_wire_covert_step5_r01_online_witness_v1 as witness
        forbidden = [
            name for name in sys.modules
            if any(token in name for token in (
                "qualification", "generator", "composer", "service_v2",
                "runtime_closure_service", "store_service",
            ))
        ]
        if forbidden:
            raise SystemExit("forbidden imports: " + repr(forbidden))
        try:
            witness.verify_real_external_witness_admission_v1()
        except witness.Phase2BStep5R01OnlineWitnessErrorV1 as exc:
            if exc.code != witness.FAIL_NOT_IMPLEMENTED_V1:
                raise
        else:
            raise SystemExit("real admission unexpectedly implemented")
        print("CLEAN_NOT_IMPLEMENTED")
        """
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(source_root)
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    # Isolated mode ignores PYTHONPATH, so pass the source root explicitly.
    if completed.returncode != 0 and "No module named 'hegel_machine'" in completed.stderr:
        isolated_script = f"import sys; sys.path.insert(0, {str(source_root)!r})\n" + script
        completed = subprocess.run(
            [sys.executable, "-I", "-c", isolated_script],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            check=False,
        )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "CLEAN_NOT_IMPLEMENTED"
