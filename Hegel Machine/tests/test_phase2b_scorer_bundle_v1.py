"""Synthetic-only tests for the bounded Phase-2B scorer bundle codec.

The fixtures below are unbacked mechanics graphs.  They do not authenticate an
actor, execute a recognizer, establish reveal chronology, perform actual
scoring/formal gate evaluation, estimate an effect, or constitute C1 evidence.
"""

from __future__ import annotations

import ast
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest

import hegel_machine.phase2b_actual_unsealed_960_replay_input_contract_v2 as replay_v2
import hegel_machine.phase2b_formal_unsealed_prediction_scoring_contract_v2 as answer_v2
import hegel_machine.phase2b_scorer_bundle_v1 as bundle_v1
import hegel_machine.phase2b_unsealed_960_available_gate_mechanics_v2 as gate_v2
import hegel_machine.phase2b_unsealed_960_prediction_scoring_mechanics_v2 as score_v2
import hegel_machine.phase2b_unsealed_prediction_evaluator_v2 as evaluator_v2
from hegel_machine.phase2b_protocol import salted_answer_commitment_sha256


SOURCE = Path(bundle_v1.__file__).read_text(encoding="utf-8")
INPUT_BYTES = b"I" * bundle_v1.ARCHIVE_HEADER_BYTES_V2
PREDICTION_BYTES = b"P" * bundle_v1.PREDICTION_ARCHIVE_HEADER_BYTES_V2
FREEZE_BYTES = b"{}"
INPUT_ID = "phase2b_recognizer_input_archive_v2_" + "1" * 64
PREDICTION_ID = "phase2b_recognizer_prediction_archive_v2_" + "2" * 64
PREDICTION_RECEIPT_ID = "phase2b_prediction_if_absent_receipt_v1_" + "3" * 64
REVEAL_TOKEN_ID = "phase2b_signed_reveal_token_v1_" + "4" * 64
CAMPAIGN_ID = "phase2b_actual_unsealed_960_campaign_v2_" + "5" * 64
ATTEMPT_ID = "phase2b_actual_unsealed_960_attempt_v2_" + "6" * 64
EXECUTION_REGISTRY_ID = (
    "phase2b_c1_owner_accepted_execution_registry_v1_" + "7" * 64
)
SCORER_IMAGE_DIGEST = "sha256:" + "8" * 64
SCORER_SOURCE_ROOT_SHA256 = "9" * 64
ANSWER_SALT = "synthetic-answer-opening-only-0123456789abcdef"
GATE_SALT = "synthetic-gate-opening-only-0123456789abcdef"
EXPECTED_FALSE_CLAIMS = (
    "actual_960_case_run_verified",
    "authority_requirement_satisfied",
    "execution_registry_verified",
    "campaign_authority_verified",
    "attempt_authority_verified",
    "organizational_independence_verified",
    "independent_human_actors_verified",
    "answer_authority_verified",
    "gate_input_authority_verified",
    "prediction_receipt_signature_verified",
    "reveal_token_signature_verified",
    "scorer_image_runtime_verified",
    "scorer_source_root_verified",
    "pre_reveal_commitment_timing_verified",
    "one_shot_policy_enforced",
    "durable_attempt_ledger_verified",
    "rollback_resistance_verified",
    "authoritative_time_verified",
    "input_archive_membership_verified",
    "source_registry_projection_verified",
    "single_live_allocation_verified",
    "secret_custodian_replay_verified",
    "execution_manifest_authority_verified",
    "partition_manifest_authority_verified",
    "recognizer_executed",
    "runtime_executed",
    "recognizer_capacity_evidence",
    "origin_authenticated",
    "formal_uuid_audit",
    "formal_covert_audit",
    "sealed_holdout_eligible",
    "scoring_performed",
    "prediction_scored",
    "actual_prediction_scoring_evidence",
    "formal_gate_evaluation_performed",
    "effect_evidence",
    "c1_exit_evidence",
)


def _replay_namespace() -> dict[str, object]:
    return runpy.run_path(
        str(Path(__file__).with_name(
            "test_phase2b_actual_unsealed_960_replay_input_contract_v2.py"
        ))
    )


def _build_graphs() -> SimpleNamespace:
    namespace = _replay_namespace()
    base = namespace["_synthetic_fixture_base"]()
    original_gate_rows = namespace["_gate_rows"](base)
    freeze = runpy.run_path(
        str(Path(__file__).with_name("test_phase2b_runner.py"))
    )["_freeze_manifest"]()
    old = base.answer
    answer = answer_v2.build_formal_unsealed_answer_manifest_v2(
        input_archive_id=INPUT_ID,
        input_archive_sha256=hashlib.sha256(INPUT_BYTES).hexdigest(),
        input_archive_version=old.input_archive_version,
        input_archive_policy_id=old.input_archive_policy_id,
        batch_id=old.batch_id,
        batch_policy_id=old.batch_policy_id,
        exact_freeze_id=old.exact_freeze_id,
        phase2b_protocol_id=old.phase2b_protocol_id,
        execution_freeze_manifest_id=freeze.manifest_id,
        ordered_archive_input_row_ids_root=old.ordered_archive_input_row_ids_root,
        main_row_ids_root=old.main_row_ids_root,
        semantic_conflict_row_ids_root=old.semantic_conflict_row_ids_root,
        partition_union_row_ids_root=old.partition_union_row_ids_root,
        main_answer_rows=tuple(
            answer_v2.FormalUnsealedAnswerRowV2(
                input_row_id=row.input_row_id,
                case_type=row.case_type,
                expected_decision=row.expected_decision,
                canonical_family_id=row.canonical_family_id,
                binding=row.binding,
                admissible_scale_ids=row.admissible_scale_ids,
            )
            for row in old.main_answer_rows
        ),
    )
    gate_rows = tuple(
        replay_v2.FormalUnsealedGateInputRowV2(
            input_row_id=new_answer.input_row_id,
            answer_row_id=new_answer.answer_row_id,
            case_type=old_gate.case_type,
            margin_stratum=old_gate.margin_stratum,
            canonical_family_id=old_gate.canonical_family_id,
            scale_slice_id=old_gate.scale_slice_id,
            latent_base_case_id=old_gate.latent_base_case_id,
        )
        for new_answer, old_gate in zip(
            answer.main_answer_rows,
            original_gate_rows,
            strict=True,
        )
    )
    gate = replay_v2.build_formal_unsealed_gate_input_manifest_v2(
        answer_manifest=answer,
        main_gate_input_rows=gate_rows,
    )
    partition = object.__new__(bundle_v1.UnsealedPredictionPartitionManifestV2)
    for name, value in (
        ("main_row_ids", base.main_row_ids),
        ("semantic_conflict_row_ids", base.semantic_conflict_row_ids),
        ("exact_freeze_id", answer.exact_freeze_id),
        ("ordered_archive_input_row_ids_root", answer.ordered_archive_input_row_ids_root),
        ("main_row_ids_root", answer.main_row_ids_root),
        ("semantic_conflict_row_ids_root", answer.semantic_conflict_row_ids_root),
        ("partition_union_row_ids_root", answer.partition_union_row_ids_root),
    ):
        object.__setattr__(partition, name, value)
    receipt = object.__new__(bundle_v1.StrictRecognizerStructuralReceiptV2)
    for name, value in (
        ("input_archive_id", INPUT_ID),
        ("input_archive_sha256", hashlib.sha256(INPUT_BYTES).hexdigest()),
        ("input_archive_version", answer.input_archive_version),
        ("input_archive_policy_id", answer.input_archive_policy_id),
        ("prediction_archive_id", PREDICTION_ID),
        ("prediction_archive_sha256", hashlib.sha256(PREDICTION_BYTES).hexdigest()),
        ("batch_id", answer.batch_id),
        ("batch_policy_id", answer.batch_policy_id),
        ("execution_freeze_manifest_id", answer.execution_freeze_manifest_id),
        ("protocol_id", answer.phase2b_protocol_id),
    ):
        object.__setattr__(receipt, name, value)
    decoded_input = object.__new__(bundle_v1.DecodedRecognizerInputArchiveV2)
    object.__setattr__(decoded_input, "archive_id", INPUT_ID)
    decoded_prediction = object.__new__(bundle_v1.DecodedRecognizerPredictionArchiveV2)
    object.__setattr__(decoded_prediction, "archive_id", PREDICTION_ID)
    return SimpleNamespace(
        freeze=freeze,
        decoded_input=decoded_input,
        decoded_prediction=decoded_prediction,
        answer=answer,
        gate=gate,
        partition=partition,
        receipt=receipt,
        main_row_ids=base.main_row_ids,
        semantic_conflict_row_ids=base.semantic_conflict_row_ids,
        answer_commitment=salted_answer_commitment_sha256(
            answer.answer_manifest_sha256,
            ANSWER_SALT,
        ),
        gate_commitment=replay_v2.salted_gate_input_commitment_sha256_v2(
            gate.gate_input_manifest_sha256,
            GATE_SALT,
        ),
    )


@pytest.fixture(scope="module")
def graphs() -> SimpleNamespace:
    return _build_graphs()


def _install_public_graph_stubs(
    monkeypatch: pytest.MonkeyPatch,
    graphs: SimpleNamespace,
) -> dict[str, int]:
    calls = {
        "freeze": 0,
        "input": 0,
        "prediction": 0,
        "strict": 0,
        "partition": 0,
        "structural": 0,
        "formal": 0,
        "replay": 0,
    }

    def freeze_decode(value: bytes) -> object:
        calls["freeze"] += 1
        assert value == FREEZE_BYTES
        return graphs.freeze

    def input_decode(value: bytes) -> object:
        calls["input"] += 1
        assert value == INPUT_BYTES
        return graphs.decoded_input

    def prediction_decode(value: bytes) -> object:
        calls["prediction"] += 1
        assert value == PREDICTION_BYTES
        return graphs.decoded_prediction

    def strict_verify(*, input_archive: bytes, prediction_archive: bytes) -> object:
        calls["strict"] += 1
        assert input_archive == INPUT_BYTES
        assert prediction_archive == PREDICTION_BYTES
        return graphs.receipt

    def partition_build(**kwargs: object) -> object:
        calls["partition"] += 1
        assert kwargs["main_row_ids"] == graphs.main_row_ids
        assert kwargs["semantic_conflict_row_ids"] == graphs.semantic_conflict_row_ids
        return graphs.partition

    def structural_evaluate(**kwargs: object) -> evaluator_v2.UnsealedPredictionStructuralEvaluationV2:
        calls["structural"] += 1
        return object.__new__(evaluator_v2.UnsealedPredictionStructuralEvaluationV2)

    def formal_validate(**kwargs: object) -> answer_v2.FormalUnsealedPredictionScoringContractValidationV2:
        calls["formal"] += 1
        return object.__new__(answer_v2.FormalUnsealedPredictionScoringContractValidationV2)

    original_replay = bundle_v1.validate_actual_unsealed_960_replay_input_contract_v2

    def replay_validate(**kwargs: object) -> object:
        calls["replay"] += 1
        return original_replay(**kwargs)

    monkeypatch.setattr(bundle_v1, "decode_execution_freeze_manifest_json_v2", freeze_decode)
    monkeypatch.setattr(bundle_v1, "decode_public_recognizer_input_archive_v2", input_decode)
    monkeypatch.setattr(bundle_v1, "decode_public_recognizer_prediction_archive_v2", prediction_decode)
    monkeypatch.setattr(bundle_v1, "verify_strict_recognizer_io_structure_v2", strict_verify)
    monkeypatch.setattr(bundle_v1, "build_unsealed_prediction_partition_manifest_v2", partition_build)
    monkeypatch.setattr(bundle_v1, "evaluate_unsealed_prediction_archive_structure_v2", structural_evaluate)
    monkeypatch.setattr(bundle_v1, "validate_formal_unsealed_prediction_scoring_contract_v2", formal_validate)
    monkeypatch.setattr(bundle_v1, "validate_actual_unsealed_960_replay_input_contract_v2", replay_validate)
    return calls


def _encode_kwargs(graphs: SimpleNamespace) -> dict[str, object]:
    return {
        "authority_requirement_id": (
            bundle_v1.PHASE2B_C1_OWNER_ACCEPTED_AUTHORITY_REQUIREMENT_ID_V1
        ),
        "actor_profile_content_id": (
            bundle_v1.PHASE2B_SCORER_ACTOR_PROFILE_CONTENT_ID_V1
        ),
        "actor_profile_canonical_sha256": (
            bundle_v1.PHASE2B_SCORER_ACTOR_PROFILE_CANONICAL_SHA256_V1
        ),
        "campaign_id": CAMPAIGN_ID,
        "attempt_id": ATTEMPT_ID,
        "execution_registry_id": EXECUTION_REGISTRY_ID,
        "scorer_image_digest": SCORER_IMAGE_DIGEST,
        "scorer_source_root_sha256": SCORER_SOURCE_ROOT_SHA256,
        "execution_freeze_bytes": FREEZE_BYTES,
        "input_archive": INPUT_BYTES,
        "prediction_archive": PREDICTION_BYTES,
        "main_row_ids": graphs.main_row_ids,
        "semantic_conflict_row_ids": graphs.semantic_conflict_row_ids,
        "answer_manifest": graphs.answer,
        "answer_commitment_salt": ANSWER_SALT,
        "salted_answer_commitment_sha256": graphs.answer_commitment,
        "gate_input_manifest": graphs.gate,
        "gate_input_commitment_salt": GATE_SALT,
        "salted_gate_input_commitment_sha256": graphs.gate_commitment,
        "prediction_if_absent_receipt_id": PREDICTION_RECEIPT_ID,
        "signed_reveal_token_id": REVEAL_TOKEN_ID,
    }


def _encode(graphs: SimpleNamespace) -> bytes:
    return bundle_v1.encode_phase2b_scorer_bundle_v1(**_encode_kwargs(graphs))


def _split(bundle: bytes) -> tuple[list[int | bytes], dict[str, object], bytes, bytes, bytes]:
    header = list(bundle_v1._HEADER_V1.unpack_from(bundle, 0))
    metadata_length, freeze_length, input_length, prediction_length = header[1:]
    cursor = bundle_v1._HEADER_V1.size
    metadata_bytes = bundle[cursor : cursor + metadata_length]
    cursor += metadata_length
    freeze = bundle[cursor : cursor + freeze_length]
    cursor += freeze_length
    raw_input = bundle[cursor : cursor + input_length]
    cursor += input_length
    prediction = bundle[cursor : cursor + prediction_length]
    return header, json.loads(metadata_bytes), freeze, raw_input, prediction


def _repack(header: list[int | bytes], metadata: dict[str, object], freeze: bytes, raw_input: bytes, prediction: bytes) -> bytes:
    metadata_bytes = json.dumps(
        metadata,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return bundle_v1._HEADER_V1.pack(
        header[0], len(metadata_bytes), len(freeze), len(raw_input), len(prediction)
    ) + metadata_bytes + freeze + raw_input + prediction


def _refresh_run_manifest_id(metadata: dict[str, object]) -> None:
    run = metadata["run_manifest"]
    ordered = {name: run[name] for name in bundle_v1._RUN_FIELDS_WITHOUT_ID_V1}
    run["run_manifest_id"] = bundle_v1._run_manifest_id_v1(ordered)


def test_exact_filenames_profile_source_and_private_surfaces() -> None:
    assert bundle_v1.PHASE2B_SCORER_BUNDLE_V1_FILENAME == "phase2b_scorer_bundle_v1.bin"
    assert bundle_v1.PHASE2B_SCORER_SCORE_MECHANICS_REPORT_V1_FILENAME == "score_mechanics_report_v1.json"
    assert bundle_v1.PHASE2B_SCORER_AVAILABLE_GATE_MECHANICS_REPORT_V1_FILENAME == "available_gate_mechanics_report_v1.json"
    assert bundle_v1.PHASE2B_SCORING_SOURCE_SHA256_V1 == "c3c12e9cd72a930b9ca8667aab22c4eb3675ceab6249b0b26ba940bf794e7f11"
    scoring_source = Path(bundle_v1.__file__).with_name(
        "phase2b_unsealed_960_prediction_scoring_mechanics_v2.py"
    ).read_bytes()
    assert hashlib.sha256(scoring_source).hexdigest() == (
        bundle_v1.PHASE2B_SCORING_SOURCE_SHA256_V1
    )
    assert bundle_v1.PHASE2B_SCORER_ACTOR_PROFILE_CONTENT_ID_V1 == (
        "phase2b_c1_container_actor_profile_v2_"
        "b2f081771f3de8d2a531b08b58109d95c51815904e9612e66f1bbfe69babf906"
    )
    assert bundle_v1.PHASE2B_SCORER_ACTOR_PROFILE_CANONICAL_SHA256_V1 == (
        "8ca1850f026dabb7a4b69add0f89b33a3fcfb789223b2fd95ef0379608adbeed"
    )
    assert bundle_v1.SCORE_REPORT_ID_PREFIX_V1 == "phase2b_score_mechanics_report_v1_"
    assert bundle_v1.SCORE_REPORT_DOMAIN_V1 == b"HEGEL/PHASE2B/SCORER/SCORE_REPORT/V1\x00"
    assert bundle_v1.GATE_REPORT_ID_PREFIX_V1 == (
        "phase2b_available_gate_mechanics_report_v1_"
    )
    assert bundle_v1.GATE_REPORT_DOMAIN_V1 == b"HEGEL/PHASE2B/SCORER/GATE_REPORT/V1\x00"
    assert len(bundle_v1.PHASE2B_SCORER_BUNDLE_V1_MAGIC) == 32
    assert tuple(item.name for item in fields(bundle_v1.DecodedPhase2BScorerBundleV1)) == (
        bundle_v1._DECODED_FIELDS_V1
    )
    assert "replay_input_contract" not in bundle_v1._DECODED_FIELDS_V1
    assert tuple(item.name for item in fields(bundle_v1.EvaluatedPhase2BScorerBundleV1)) == (
        bundle_v1._EVALUATED_FIELDS_V1
    )
    assert not {
        "canonicalize_score_mechanics_report_v1",
        "canonicalize_available_gate_mechanics_report_v1",
        "canonical_score_mechanics_report_bytes_v1",
        "canonical_available_gate_mechanics_report_bytes_v1",
    }.intersection(bundle_v1.__all__)
    with pytest.raises(TypeError):
        bundle_v1.DecodedPhase2BScorerBundleV1()
    with pytest.raises(TypeError):
        bundle_v1.EvaluatedPhase2BScorerBundleV1()


def test_deterministic_roundtrip_independent_bytes_and_ids(
    monkeypatch: pytest.MonkeyPatch,
    graphs: SimpleNamespace,
) -> None:
    calls = _install_public_graph_stubs(monkeypatch, graphs)
    first = _encode(graphs)
    second = _encode(graphs)
    assert first == second
    decoded = bundle_v1.decode_phase2b_scorer_bundle_v1(first)
    assert decoded.bundle_bytes == first
    expected_sha = hashlib.sha256(first).hexdigest()
    assert decoded.bundle_sha256 == expected_sha
    assert decoded.bundle_id == "phase2b_scorer_bundle_v1_" + expected_sha
    assert decoded.authority_requirement_id == (
        bundle_v1.PHASE2B_C1_OWNER_ACCEPTED_AUTHORITY_REQUIREMENT_ID_V1
    )
    assert decoded.actor_profile_content_id == bundle_v1.PHASE2B_SCORER_ACTOR_PROFILE_CONTENT_ID_V1
    assert decoded.actor_profile_canonical_sha256 == (
        bundle_v1.PHASE2B_SCORER_ACTOR_PROFILE_CANONICAL_SHA256_V1
    )
    assert decoded.campaign_id == CAMPAIGN_ID
    assert decoded.attempt_id == ATTEMPT_ID
    assert decoded.execution_registry_id == EXECUTION_REGISTRY_ID
    assert decoded.scorer_image_digest == SCORER_IMAGE_DIGEST
    assert decoded.scorer_source_root_sha256 == SCORER_SOURCE_ROOT_SHA256
    assert decoded.protocol_id == graphs.receipt.protocol_id
    assert decoded.exact_freeze_id == graphs.partition.exact_freeze_id
    assert decoded.execution_freeze_manifest_id == graphs.freeze.manifest_id
    assert decoded.scoring_source_sha256 == bundle_v1.PHASE2B_SCORING_SOURCE_SHA256_V1
    assert decoded.prediction_if_absent_receipt_id == PREDICTION_RECEIPT_ID
    assert decoded.signed_reveal_token_id == REVEAL_TOKEN_ID
    assert len(decoded.answer_manifest.main_answer_rows) == 720
    assert len(decoded.partition_manifest.main_row_ids) == 720
    assert len(graphs.semantic_conflict_row_ids) == 240
    header, metadata, _freeze, _raw_input, _prediction = _split(first)
    metadata_bytes = first[
        bundle_v1._HEADER_V1.size : bundle_v1._HEADER_V1.size + header[1]
    ]
    assert b"\n" not in metadata_bytes and b"\r" not in metadata_bytes
    assert bundle_v1._FALSE_CLAIMS_V1 == EXPECTED_FALSE_CLAIMS
    assert tuple(sorted(metadata["claims"])) == tuple(sorted(EXPECTED_FALSE_CLAIMS))
    assert all(type(value) is bool and value is False for value in metadata["claims"].values())
    assert all("answer_row_id" not in row for row in metadata["answer_rows"])
    assert all("answer_row_id" not in row for row in metadata["gate_rows"])
    assert all("gate_input_row_id" not in row for row in metadata["gate_rows"])
    run = metadata["run_manifest"]
    assert tuple(run[name] for name in bundle_v1._RUN_FIELDS_WITHOUT_ID_V1) == tuple(
        decoded.metadata["run_manifest"][name]
        for name in bundle_v1._RUN_FIELDS_WITHOUT_ID_V1
    )
    run_preimage = [
        [name, run[name]] for name in bundle_v1._RUN_FIELDS_WITHOUT_ID_V1
    ]
    expected_run_id = "phase2b_scorer_run_manifest_v1_" + hashlib.sha256(
        b"HEGEL/PHASE2B/SCORER/RUN_MANIFEST/V1\x00"
        + json.dumps(
            run_preimage,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()
    assert run["run_manifest_id"] == decoded.run_manifest_id == expected_run_id
    assert calls == {
        "freeze": 5,
        "input": 3,
        "prediction": 5,
        "strict": 5,
        "partition": 5,
        "structural": 3,
        "formal": 3,
        "replay": 0,
    }


@pytest.mark.parametrize(
    "mutation",
    (
        "magic",
        "metadata",
        "noncanonical",
        "section",
        "crossbind",
        "answer-derived-id",
        "gate-derived-ids",
        "unknown",
        "missing",
        "float",
        "truncated",
        "trailing",
    ),
)
def test_tamper_noncanonical_unknown_float_and_crossbinding_reject(
    monkeypatch: pytest.MonkeyPatch,
    graphs: SimpleNamespace,
    mutation: str,
) -> None:
    _install_public_graph_stubs(monkeypatch, graphs)
    bundle = _encode(graphs)
    header, metadata, freeze, raw_input, prediction = _split(bundle)
    if mutation == "magic":
        header[0] = b"X" * 32
        changed = _repack(header, metadata, freeze, raw_input, prediction)
    elif mutation == "metadata":
        raw = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("ascii")
        raw = raw.replace(b'"schema_version":', b'"schema_version":"x","schema_version":', 1)
        changed = bundle_v1._HEADER_V1.pack(header[0], len(raw), len(freeze), len(raw_input), len(prediction)) + raw + freeze + raw_input + prediction
    elif mutation == "noncanonical":
        raw = json.dumps(metadata, sort_keys=True, separators=(", ", ": ")).encode("ascii")
        changed = bundle_v1._HEADER_V1.pack(header[0], len(raw), len(freeze), len(raw_input), len(prediction)) + raw + freeze + raw_input + prediction
    elif mutation == "section":
        changed = bundle[:-1] + bytes((bundle[-1] ^ 1,))
    elif mutation == "crossbind":
        metadata["run_manifest"]["protocol_id"] = "phase2b_protocol_" + "9" * 64
        _refresh_run_manifest_id(metadata)
        changed = _repack(header, metadata, freeze, raw_input, prediction)
    elif mutation == "answer-derived-id":
        metadata["answer_rows"][0]["answer_row_id"] = (
            graphs.answer.main_answer_rows[0].answer_row_id
        )
        changed = _repack(header, metadata, freeze, raw_input, prediction)
    elif mutation == "gate-derived-ids":
        metadata["gate_rows"][0]["answer_row_id"] = (
            graphs.gate.main_gate_input_rows[0].answer_row_id
        )
        metadata["gate_rows"][0]["gate_input_row_id"] = (
            graphs.gate.main_gate_input_rows[0].gate_input_row_id
        )
        changed = _repack(header, metadata, freeze, raw_input, prediction)
    elif mutation == "unknown":
        metadata["unknown"] = False
        changed = _repack(header, metadata, freeze, raw_input, prediction)
    elif mutation == "missing":
        del metadata["claims"]
        changed = _repack(header, metadata, freeze, raw_input, prediction)
    elif mutation == "float":
        raw = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("ascii")
        raw = raw[:-1] + b',"float":1.5}'
        changed = bundle_v1._HEADER_V1.pack(header[0], len(raw), len(freeze), len(raw_input), len(prediction)) + raw + freeze + raw_input + prediction
    elif mutation == "truncated":
        changed = bundle[:-1]
    else:
        changed = bundle + b"\x00"
    with pytest.raises(bundle_v1.Phase2BScorerBundleErrorV1):
        bundle_v1.decode_phase2b_scorer_bundle_v1(changed)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        (
            "authority_requirement_id",
            "phase2b_c1_owner_accepted_container_authority_requirement_v1_"
            + "0" * 64,
        ),
        (
            "actor_profile_content_id",
            "phase2b_c1_container_actor_profile_v2_" + "0" * 64,
        ),
        ("actor_profile_canonical_sha256", "0" * 64),
        ("campaign_id", "phase2b_wrong_campaign_v2_" + "0" * 64),
        ("attempt_id", "phase2b_wrong_attempt_v2_" + "0" * 64),
        (
            "execution_registry_id",
            "phase2b_wrong_execution_registry_v1_" + "0" * 64,
        ),
        ("scorer_image_digest", "sha256:" + "A" * 64),
        ("scorer_source_root_sha256", "A" * 64),
        (
            "prediction_if_absent_receipt_id",
            "phase2b_wrong_receipt_v1_" + "0" * 64,
        ),
        (
            "signed_reveal_token_id",
            "phase2b_wrong_reveal_v1_" + "0" * 64,
        ),
        ("scoring_source_sha256", "0" * 64),
    ),
)
def test_run_manifest_static_identity_prefix_and_shape_tamper_reject(
    monkeypatch: pytest.MonkeyPatch,
    graphs: SimpleNamespace,
    field: str,
    replacement: str,
) -> None:
    _install_public_graph_stubs(monkeypatch, graphs)
    bundle = _encode(graphs)
    header, metadata, freeze, raw_input, prediction = _split(bundle)
    metadata["run_manifest"][field] = replacement
    _refresh_run_manifest_id(metadata)
    changed = _repack(header, metadata, freeze, raw_input, prediction)
    with pytest.raises(bundle_v1.Phase2BScorerBundleErrorV1):
        bundle_v1.decode_phase2b_scorer_bundle_v1(changed)


def test_wrong_types_and_freshness_reject_before_first_hash(
    monkeypatch: pytest.MonkeyPatch,
    graphs: SimpleNamespace,
) -> None:
    class HashReached(BaseException):
        pass

    monkeypatch.setattr(bundle_v1.hashlib, "sha256", lambda *args, **kwargs: (_ for _ in ()).throw(HashReached()))
    wrong_type = _encode_kwargs(graphs)
    wrong_type["execution_freeze_bytes"] = bytearray(FREEZE_BYTES)
    with pytest.raises(bundle_v1.Phase2BScorerBundleErrorV1):
        bundle_v1.encode_phase2b_scorer_bundle_v1(**wrong_type)
    wrong_prefix = _encode_kwargs(graphs)
    wrong_prefix["signed_reveal_token_id"] = PREDICTION_RECEIPT_ID
    with pytest.raises(bundle_v1.Phase2BScorerBundleErrorV1):
        bundle_v1.encode_phase2b_scorer_bundle_v1(**wrong_prefix)


def test_decode_evaluate_calls_replay_once_total_and_scorer_gate_once(
    monkeypatch: pytest.MonkeyPatch,
    graphs: SimpleNamespace,
) -> None:
    calls = _install_public_graph_stubs(monkeypatch, graphs)
    decoded = bundle_v1.decode_phase2b_scorer_bundle_v1(_encode(graphs))
    scoring = object.__new__(score_v2.Unsealed960PredictionScoringMechanicsV2)
    gate = object.__new__(gate_v2.Unsealed960AvailableGateMechanicsV2)
    runtime_calls = {"score": 0, "gate": 0}

    def score(**kwargs: object) -> object:
        runtime_calls["score"] += 1
        return scoring

    observed_replay: list[object] = []

    def gate_eval(**kwargs: object) -> object:
        runtime_calls["gate"] += 1
        observed_replay.append(kwargs["replay_input_contract"])
        return gate

    monkeypatch.setattr(bundle_v1, "score_unsealed_960_prediction_scoring_mechanics_v2", score)
    monkeypatch.setattr(bundle_v1, "evaluate_unsealed_960_available_gate_mechanics_v2", gate_eval)
    monkeypatch.setattr(bundle_v1, "_canonical_score_mechanics_report_v1", lambda value: b'{"score":false}\n')
    monkeypatch.setattr(bundle_v1, "_canonical_available_gate_mechanics_report_v1", lambda value: b'{"gate":false}\n')
    before = dict(calls)
    evaluated = bundle_v1.evaluate_phase2b_scorer_bundle_v1(decoded)
    assert runtime_calls == {"score": 1, "gate": 1}
    assert calls == {name: count + 1 for name, count in before.items()}
    assert before["replay"] == 0 and calls["replay"] == 1
    assert observed_replay == [evaluated.replay_input_contract]
    assert evaluated.score_report_bytes == b'{"score":false}\n'
    assert evaluated.available_gate_report_bytes == b'{"gate":false}\n'
    assert evaluated.score_report_id == (
        bundle_v1.SCORE_REPORT_ID_PREFIX_V1
        + hashlib.sha256(
            bundle_v1.SCORE_REPORT_DOMAIN_V1 + evaluated.score_report_bytes
        ).hexdigest()
    )
    assert evaluated.available_gate_report_id == (
        bundle_v1.GATE_REPORT_ID_PREFIX_V1
        + hashlib.sha256(
            bundle_v1.GATE_REPORT_DOMAIN_V1
            + evaluated.available_gate_report_bytes
        ).hexdigest()
    )


def test_evaluate_redecodes_and_ignores_forged_nested_dto_surface(
    monkeypatch: pytest.MonkeyPatch,
    graphs: SimpleNamespace,
) -> None:
    calls = _install_public_graph_stubs(monkeypatch, graphs)
    bundle = _encode(graphs)
    bundle_sha = hashlib.sha256(bundle).hexdigest()
    forged = object.__new__(bundle_v1.DecodedPhase2BScorerBundleV1)
    for name, value in (
        ("bundle_bytes", bundle),
        ("bundle_sha256", bundle_sha),
        ("bundle_id", "phase2b_scorer_bundle_v1_" + bundle_sha),
        ("answer_manifest", None),
        ("gate_input_manifest", None),
        ("prediction_archive_bytes", b"forged"),
    ):
        object.__setattr__(forged, name, value)
    scoring = object.__new__(score_v2.Unsealed960PredictionScoringMechanicsV2)
    gate = object.__new__(gate_v2.Unsealed960AvailableGateMechanicsV2)
    observed: dict[str, object] = {}

    def score(**kwargs: object) -> object:
        observed["answer"] = kwargs["answer_manifest"]
        observed["prediction"] = kwargs["prediction_archive"]
        return scoring

    def gate_eval(**kwargs: object) -> object:
        observed["gate_manifest"] = kwargs["gate_input_manifest"]
        return gate

    monkeypatch.setattr(
        bundle_v1,
        "score_unsealed_960_prediction_scoring_mechanics_v2",
        score,
    )
    monkeypatch.setattr(
        bundle_v1,
        "evaluate_unsealed_960_available_gate_mechanics_v2",
        gate_eval,
    )
    monkeypatch.setattr(
        bundle_v1,
        "_canonical_score_mechanics_report_v1",
        lambda value: b'{"score":false}\n',
    )
    monkeypatch.setattr(
        bundle_v1,
        "_canonical_available_gate_mechanics_report_v1",
        lambda value: b'{"gate":false}\n',
    )
    before = dict(calls)
    evaluated = bundle_v1.evaluate_phase2b_scorer_bundle_v1(forged)
    assert type(evaluated) is bundle_v1.EvaluatedPhase2BScorerBundleV1
    assert observed["prediction"] == PREDICTION_BYTES
    assert type(observed["answer"]) is answer_v2.FormalUnsealedAnswerManifestV2
    assert type(observed["gate_manifest"]) is replay_v2.FormalUnsealedGateInputManifestV2
    assert calls == {name: count + 1 for name, count in before.items()}

    wrong_identity = object.__new__(bundle_v1.DecodedPhase2BScorerBundleV1)
    object.__setattr__(wrong_identity, "bundle_bytes", bundle)
    object.__setattr__(wrong_identity, "bundle_sha256", bundle_sha)
    object.__setattr__(wrong_identity, "bundle_id", "phase2b_scorer_bundle_v1_" + "0" * 64)
    before = dict(calls)
    with pytest.raises(bundle_v1.Phase2BScorerBundleErrorV1):
        bundle_v1.evaluate_phase2b_scorer_bundle_v1(wrong_identity)
    assert calls == before


@pytest.mark.parametrize("exception", (TypeError("synthetic"), KeyError("synthetic")))
def test_decode_normalizes_ordinary_upstream_exceptions(
    monkeypatch: pytest.MonkeyPatch,
    graphs: SimpleNamespace,
    exception: Exception,
) -> None:
    _install_public_graph_stubs(monkeypatch, graphs)
    bundle = _encode(graphs)

    def reject(_payload: bytes) -> object:
        raise exception

    monkeypatch.setattr(bundle_v1, "decode_execution_freeze_manifest_json_v2", reject)
    with pytest.raises(bundle_v1.Phase2BScorerBundleErrorV1) as caught:
        bundle_v1.decode_phase2b_scorer_bundle_v1(bundle)
    assert caught.value.__cause__ is exception


def test_exact_real_mechanics_graph_reports_are_canonical_single_line_and_float_free() -> None:
    namespace = runpy.run_path(
        str(Path(__file__).with_name(
            "test_phase2b_unsealed_960_available_gate_mechanics_v2.py"
        ))
    )
    scoring, replay, manifest = namespace["supplied_graphs"].__wrapped__()
    available = gate_v2.evaluate_unsealed_960_available_gate_mechanics_v2(
        scoring_mechanics=scoring,
        replay_input_contract=replay,
        gate_input_manifest=manifest,
    )
    assert type(scoring) is score_v2.Unsealed960PredictionScoringMechanicsV2
    assert type(available) is gate_v2.Unsealed960AvailableGateMechanicsV2
    reports = (
        bundle_v1._canonical_score_mechanics_report_v1(scoring),
        bundle_v1._canonical_available_gate_mechanics_report_v1(available),
    )

    def reject_float(value: object) -> None:
        assert type(value) is not float
        if type(value) is dict:
            for key, item in value.items():
                assert type(key) is str
                reject_float(item)
        elif type(value) is list:
            for item in value:
                reject_float(item)

    for report in reports:
        assert type(report) is bytes
        assert report.count(b"\n") == 1 and b"\r" not in report
        assert report.startswith(b"{") and report.endswith(b"}\n")
        decoded = json.loads(report[:-1].decode("ascii"))
        assert decoded["actual_scoring_claimed"] is False
        assert decoded["formal_gate_evaluation_claimed"] is False
        reject_float(decoded)
        assert json.dumps(
            decoded,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii") + b"\n" == report


def test_ast_has_required_public_reconstruction_and_no_side_effect_imports() -> None:
    tree = ast.parse(SOURCE)
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    required = {
        "verify_strict_recognizer_io_structure_v2",
        "decode_public_recognizer_input_archive_v2",
        "decode_public_recognizer_prediction_archive_v2",
        "build_unsealed_prediction_partition_manifest_v2",
        "evaluate_unsealed_prediction_archive_structure_v2",
        "build_formal_unsealed_answer_manifest_v2",
        "validate_formal_unsealed_prediction_scoring_contract_v2",
        "build_formal_unsealed_gate_input_manifest_v2",
        "validate_actual_unsealed_960_replay_input_contract_v2",
        "score_unsealed_960_prediction_scoring_mechanics_v2",
        "evaluate_unsealed_960_available_gate_mechanics_v2",
    }
    assert required <= calls
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not imports.intersection({"os", "pathlib", "subprocess", "socket", "time", "random", "secrets"})
    decode_node = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "decode_phase2b_scorer_bundle_v1"
    )
    decode_calls = {
        node.func.id
        for node in ast.walk(decode_node)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_decode_phase2b_scorer_bundle_v1_impl" in decode_calls
    assert "validate_actual_unsealed_960_replay_input_contract_v2" not in decode_calls
    decode_impl_node = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_decode_phase2b_scorer_bundle_v1_impl"
    )
    decode_impl_calls = {
        node.func.id
        for node in ast.walk(decode_impl_node)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "validate_actual_unsealed_960_replay_input_contract_v2" not in decode_impl_calls
    assert "salted_gate_input_commitment_sha256_v2" in decode_impl_calls
    evaluate_node = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "evaluate_phase2b_scorer_bundle_v1"
    )
    evaluate_calls = {
        node.func.id
        for node in ast.walk(evaluate_node)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "decode_phase2b_scorer_bundle_v1" in evaluate_calls
    assert "validate_actual_unsealed_960_replay_input_contract_v2" in evaluate_calls
    assert "canonicalize_score_mechanics_report_v1" not in SOURCE
    assert "canonicalize_available_gate_mechanics_report_v1" not in SOURCE
    assert SOURCE.count("actual_960_case_run_verified") >= 1
    assert SOURCE.count("c1_exit_evidence") >= 1
