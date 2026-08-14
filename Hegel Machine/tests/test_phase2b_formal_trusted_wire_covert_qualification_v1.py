from __future__ import annotations

import ast
import copy
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import runpy

import pytest

import hegel_machine.phase2b_formal_trusted_wire_covert_qualification_v1 as q
from hegel_machine.phase2b_covert_audit_v1 import (
    AnswerabilityLabel,
    AuditorLabels,
    AuditorPermutationStrata,
    InvariantDecisionRow,
    InvarianceAuditBatch,
    InvarianceKind,
    RecognizerInvariantReceipt,
)
from hegel_machine.phase2b_trusted_wire_batch_v2 import TrustedWireKeySourcesV2


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/phase2b_formal_trusted_wire_covert_qualification_v1.json"
SOURCE = ROOT / "src/hegel_machine/phase2b_formal_trusted_wire_covert_qualification_v1.py"
DOC = ROOT / "docs/Hegel_Machine_Phase2B_Formal_Trusted_Wire_Covert_Qualification_Scaffold_v1.md"
EXPECTED_PROFILE_ID = (
    "phase2b_formal_trusted_wire_covert_qualification_profile_v1_"
    "de3f7edfacc8115c565b30a60bd285500cb0027cf4e7640749c8c1b954ae5c63"
)
EXPECTED_PROFILE_BODY_SHA256 = (
    "85e5697f42d36871862b88b1aa3ba68c6eee262404a8e779518bfdf4f1cab0f2"
)
EXPECTED_QUALIFICATION_RECEIPT_PAYLOAD_SHA256 = (
    "69980b341cf6b16608f0af848b624a94b0bc669e0d3e2936e2e9272cd75dea3e"
)
EXPECTED_QUALIFICATION_RECEIPT_ID = (
    "phase2b_formal_trusted_wire_covert_qualification_receipt_v1_"
    "bcd177752c2395b13fde824940870cb124c4e3ed9b2dbccec6550042a6e66c9c"
)


@pytest.fixture(scope="module")
def profile() -> q.Phase2BFormalTrustedWireCovertProfileV1:
    return q.load_phase2b_formal_trusted_wire_covert_qualification_profile_v1(
        CONFIG.read_bytes()
    )


@pytest.fixture(scope="module")
def authorities():
    namespace = runpy.run_path(
        str(ROOT / "tests/test_phase2b_exact_transform_semantics_v1.py")
    )
    return tuple(
        namespace[name]() for name in ("identity_authority", "unit_authority")
    )


def _keys() -> TrustedWireKeySourcesV2:
    return TrustedWireKeySourcesV2(
        bytes.fromhex(
            "a6a0e6f539332c38d883124893a457bcd3e91be544ad6f4e211f8539e17bcaea"
        ),
        bytes.fromhex(
            "e35b1f110b64e900804e47da1688d6fd0782975cb622e36496ffa9fdaa4bb439"
        ),
        bytes.fromhex(
            "d196ef9335af26147ffcb5700976f6ce2194f14b6362c8b70e548d316750a30b"
        ),
    )


def _run_id() -> bytes:
    return bytes.fromhex(
        "50e9ed3d8f4126d17566865ee40e9ede560c82ecdd87f5017327bc8f3172c494"
    )


def _metadata():
    labels = (
        AuditorLabels(
            "F01",
            "binding-a",
            "root",
            AnswerabilityLabel.ANSWERABLE,
            "joint-a",
            "main",
        ),
        AuditorLabels(
            "F02",
            "binding-b",
            "child",
            AnswerabilityLabel.ABSTAIN,
            "joint-b",
            "main",
        ),
    )
    strata = tuple(
        AuditorPermutationStrata(
            (item.case_type, item.scale),
            (item.case_type, item.family),
            (item.case_type, item.family, item.scale),
            (item.family, item.scale),
            (item.case_type,),
        )
        for item in labels
    )
    return labels, strata


def _invariance(*, changed: bool = False) -> InvarianceAuditBatch:
    decisions = (
        InvariantDecisionRow(
            0,
            "SELECT",
            "F01",
            (("lhs", "entity-a"),),
            ("root",),
            b"residual-0",
        ),
        InvariantDecisionRow(
            1,
            "ABSTAIN",
            "F02",
            (("lhs", "entity-b"),),
            ("child",),
            b"residual-1",
        ),
    )
    baseline = RecognizerInvariantReceipt(InvarianceKind.BASELINE, 0, decisions)
    renamings = tuple(
        RecognizerInvariantReceipt(
            InvarianceKind.GLOBAL_RENAMING,
            index,
            (
                decisions
                if not changed or index != 0
                else (
                    InvariantDecisionRow(
                        0,
                        "SELECT",
                        "F01",
                        (("lhs", "entity-a"),),
                        ("root",),
                        b"attacked",
                    ),
                    decisions[1],
                )
            ),
        )
        for index in range(32)
    )
    case_orders = tuple(
        RecognizerInvariantReceipt(
            InvarianceKind.CASE_ORDER,
            index,
            decisions[index % 2 :] + decisions[: index % 2],
        )
        for index in range(32)
    )
    observations = tuple(
        RecognizerInvariantReceipt(
            InvarianceKind.OBSERVATION_ORDER,
            index,
            decisions,
        )
        for index in range(16)
    )
    return InvarianceAuditBatch(baseline, renamings, case_orders, observations)


@pytest.fixture(scope="module")
def qualified_receipt(profile, authorities):
    labels, strata = _metadata()
    result = q.qualify_phase2b_formal_trusted_wire_covert_scaffold_v1(
        profile=profile,
        authorities=authorities,
        run_id=_run_id(),
        key_sources=_keys(),
        auditor_labels=labels,
        auditor_strata=strata,
        invariance=_invariance(),
    )
    assert type(result) is q.Phase2BFormalTrustedWireCovertQualificationReceiptV1
    return result


def _decoded() -> dict[str, object]:
    value = json.loads(CONFIG.read_bytes())
    assert type(value) is dict
    return value


def test_independent_profile_canonicalization_and_content_id_kat() -> None:
    value = _decoded()
    body = value["qualification_profile"]
    canonical_body = json.dumps(
        body,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    assert hashlib.sha256(canonical_body).hexdigest() == EXPECTED_PROFILE_BODY_SHA256
    independent = q.PROFILE_ID_PREFIX + hashlib.sha256(
        b"HEGEL/PHASE2B/FORMAL_TRUSTED_WIRE_COVERT_QUALIFICATION/PROFILE/V1\x00"
        + canonical_body
    ).hexdigest()
    assert independent == EXPECTED_PROFILE_ID == q.PROFILE_ID
    assert value["qualification_profile_id"] == EXPECTED_PROFILE_ID
    canonical_root = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii") + b"\n"
    assert CONFIG.read_bytes() == canonical_root


def test_profile_loader_is_closed_strict_and_immutable(profile) -> None:
    assert profile.profile_id == EXPECTED_PROFILE_ID
    with pytest.raises(TypeError):
        profile.profile["status"] = "ATTACKED"
    with pytest.raises(TypeError, match="privately issued"):
        q.Phase2BFormalTrustedWireCovertProfileV1()


@pytest.mark.parametrize(
    "payload,code",
    [
        (
            b'{"qualification_profile":{},"qualification_profile":{},"qualification_profile_id":"x"}\n',
            q.FAIL_JSON,
        ),
        (
            b'{"qualification_profile":1.0,"qualification_profile_id":"x"}\n',
            q.FAIL_JSON,
        ),
        (
            b'{"qualification_profile":{},"qualification_profile_id":"x","unknown":false}\n',
            q.FAIL_PROFILE,
        ),
        (CONFIG.read_bytes()[:-1], q.FAIL_CANONICAL),
    ],
    ids=("duplicate", "float", "unknown", "missing-lf"),
)
def test_profile_attack_matrix_rejects(payload: bytes, code: str) -> None:
    with pytest.raises(q.Phase2BFormalTrustedWireCovertQualificationErrorV1) as exc:
        q.load_phase2b_formal_trusted_wire_covert_qualification_profile_v1(payload)
    assert exc.value.code == code


def test_critical_transitive_source_manifest_matches_live_bytes_but_runtime_does_not_claim_rehash() -> None:
    body = _decoded()["qualification_profile"]
    manifest = body["source_identity_manifest_in_order"]
    assert len(manifest) == 14
    for item in manifest:
        assert hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() == item[
            "sha256"
        ]
    assert body["source_identity_manifest_semantics"].endswith(
        "NOT_LIVE_SOURCE_REHASH_EVIDENCE"
    )
    assert body["permanent_nonclaims"]["source_files_rehashed"] is False
    assert body["permanent_nonclaims"]["git_commit_live_verified"] is False


def test_gap_ledger_covers_every_redteam_p0_and_claims_remain_false() -> None:
    body = _decoded()["qualification_profile"]
    gaps = "|".join(body["remaining_formal_gaps_in_order"])
    for token in (
        "FULL_RFC8785",
        "1024_AUTHORITY",
        "CAPPED_AT_64_ROWS",
        "KEY_SOURCE_INDEPENDENCE",
        "TEN_NAMESPACE",
        "ALL_UUID_FIELD",
        "MISSINGNESS",
        "TRANSFORM_CATALOG",
        "FILENAME_AND_ARCHIVE_ORDER",
        "PROVENANCE_AND_ELIGIBILITY_MASK",
        "PAYLOAD_LENGTH_HEADER",
        "PADDING_AND_UNUSED_TRANSFORM_SLOT",
        "DERIVED_PRIVATE_ANSWER",
        "CALLER_SUPPLIED_RECEIPTS_NOT_ACTUAL_RECOGNIZER_RERUNS",
        "ID_MESSAGE_FRAMING",
        "STANDARD_ERROR_UNSUPPORTED",
        "BINARY64",
    ):
        assert token in gaps
    nonclaims = body["permanent_nonclaims"]
    assert all(type(value) is bool and value is False for value in nonclaims.values())
    assert nonclaims["formal_wire_builder_implemented"] is False
    assert nonclaims["formal_corpus_auditor_implemented"] is False
    assert nonclaims["exact_freeze_blocker_closed"] is False
    assert nonclaims["1024_authority_capacity_qualified"] is False


def test_real_public_mechanics_kat_issues_only_scaffold_receipt(qualified_receipt) -> None:
    receipt = qualified_receipt
    assert receipt.disposition is q.QualificationDispositionV1.SYNTHETIC_SCAFFOLD_QUALIFIED
    assert receipt.profile_id == EXPECTED_PROFILE_ID
    assert receipt.implementation_profile_bound is True
    assert receipt.loaded_public_mechanics_exercised is True
    assert receipt.accepted_jcs_subset_roundtrip_kat_qualified is True
    assert receipt.fixed_envelope_and_closed_wrapper_kat_qualified is True
    assert receipt.namespace_aware_field_manifest_parse_kat_qualified is True
    assert receipt.global_renaming_receipt_coverage_and_decision_equality_kat_qualified is True
    assert receipt.insufficient_metadata_predictivity_input_abstains_kat_qualified is True
    assert receipt.covert_mechanics_receipt.disposition.value == "ABSTAIN"
    assert receipt.covert_mechanics_receipt.results == ()
    assert receipt.metadata_predictivity_statistics_executed is False
    assert receipt.metadata_predictivity_gate_passed is False
    assert receipt.formal_wire_builder_implemented is False
    assert receipt.formal_corpus_auditor_implemented is False
    assert receipt.source_files_rehashed is False
    assert receipt.capacity_1024_authority_qualified is False
    assert receipt.actual_corpus_loaded is False
    assert receipt.formal_uuid_audit is False
    assert receipt.formal_covert_audit is False
    assert receipt.execution_authorized is False
    assert receipt.exact_freeze_blocker_closed is False
    assert q.verify_phase2b_formal_trusted_wire_covert_qualification_receipt_v1(
        receipt
    ) is receipt


def test_independent_canonical_json_domain_oracle_freezes_qualification_receipt_id(
    qualified_receipt,
) -> None:
    """Recompute the public receipt ID without the production ID helper."""

    receipt = qualified_receipt
    qualified_true_claim_names = (
        "implementation_profile_bound",
        "loaded_public_mechanics_exercised",
        "accepted_jcs_subset_roundtrip_kat_qualified",
        "fixed_envelope_and_closed_wrapper_kat_qualified",
        "purpose_separated_key_input_policy_kat_qualified",
        "post_shuffle_uuidv4_allocation_kat_qualified",
        "namespace_aware_field_manifest_parse_kat_qualified",
        "public_typed_replay_kat_qualified",
        "global_renaming_receipt_coverage_and_decision_equality_kat_qualified",
        "case_and_observation_order_receipt_coverage_kat_qualified",
        "insufficient_metadata_predictivity_input_abstains_kat_qualified",
        "atomic_success_or_empty_rejection_api_qualified",
    )
    permanent_nonclaim_field_map = (
        ("source_files_rehashed", "source_files_rehashed"),
        ("git_commit_live_verified", "git_commit_live_verified"),
        ("full_rfc8785_domain_qualified", "full_rfc8785_domain_qualified"),
        ("formal_wire_builder_implemented", "formal_wire_builder_implemented"),
        ("formal_corpus_auditor_implemented", "formal_corpus_auditor_implemented"),
        (
            "metadata_predictivity_statistics_executed",
            "metadata_predictivity_statistics_executed",
        ),
        ("metadata_predictivity_gate_passed", "metadata_predictivity_gate_passed"),
        ("key_source_independence_verified", "key_source_independence_verified"),
        (
            "namespace_closed_leaf_classification_qualified",
            "namespace_closed_leaf_classification_qualified",
        ),
        (
            "ten_namespace_feature_coverage_qualified",
            "ten_namespace_feature_coverage_qualified",
        ),
        (
            "all_uuid_field_predictivity_qualified",
            "all_uuid_field_predictivity_qualified",
        ),
        (
            "missingness_freeze_representation_aligned",
            "missingness_freeze_representation_aligned",
        ),
        (
            "per_case_transform_catalog_freeze_aligned",
            "per_case_transform_catalog_freeze_aligned",
        ),
        ("payload_length_channel_audited", "payload_length_channel_audited"),
        (
            "standard_error_unsupported_atomicity_qualified",
            "standard_error_unsupported_atomicity_qualified",
        ),
        (
            "binary64_cross_runtime_formal_replay_qualified",
            "binary64_cross_runtime_formal_replay_qualified",
        ),
        (
            "global_consistent_renaming_execution_verified",
            "global_consistent_renaming_execution_verified",
        ),
        ("1024_authority_capacity_qualified", "capacity_1024_authority_qualified"),
        ("recognizer_capacity_evidence", "recognizer_capacity_evidence"),
        ("actual_corpus_loaded", "actual_corpus_loaded"),
        ("actual_corpus_audited", "actual_corpus_audited"),
        ("origin_authenticated", "origin_authenticated"),
        ("formal_uuid_audit", "formal_uuid_audit"),
        ("formal_covert_audit", "formal_covert_audit"),
        (
            "formal_auditor_authority_verified",
            "formal_auditor_authority_verified",
        ),
        ("sealed_holdout_eligible", "sealed_holdout_eligible"),
        ("actual_execution_claimed", "actual_execution_claimed"),
        ("execution_authorized", "execution_authorized"),
        (
            "formal_holdout_generation_authorized",
            "formal_holdout_generation_authorized",
        ),
        ("exact_freeze_blocker_closed", "exact_freeze_blocker_closed"),
        ("c1_exit_evidence", "c1_exit_evidence"),
        ("docker_invoked", "docker_invoked"),
        ("filesystem_accessed", "filesystem_accessed"),
        ("network_accessed", "network_accessed"),
        ("keys_generated", "keys_generated"),
    )
    independent_payload = {
        "profile_id": receipt.profile_id,
        "source_authority_content_ids": receipt.source_authority_content_ids,
        "batch_id": receipt.batch_id,
        "replay_receipt_id": receipt.replay_receipt_id,
        "envelope_ids": receipt.envelope_ids,
        "namespace_audit_ids": receipt.namespace_audit_ids,
        "structural_receipt_id": receipt.structural_receipt_id,
        "covert_mechanics_receipt_id": receipt.covert_mechanics_receipt_id,
        "claim_level": receipt.claim_level,
        "qualified_true_claims": {
            name: getattr(receipt, name) for name in qualified_true_claim_names
        },
        "permanent_nonclaims": {
            json_name: getattr(receipt, public_field_name)
            for json_name, public_field_name in permanent_nonclaim_field_map
        },
    }
    independent_canonical = json.dumps(
        independent_payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    assert hashlib.sha256(independent_canonical).hexdigest() == (
        EXPECTED_QUALIFICATION_RECEIPT_PAYLOAD_SHA256
    )
    independent_domain = (
        b"HEGEL/PHASE2B/FORMAL_TRUSTED_WIRE_COVERT_QUALIFICATION/RECEIPT/V1\x00"
    )
    independent_prefix = (
        "phase2b_formal_trusted_wire_covert_qualification_receipt_v1_"
    )
    independent_id = independent_prefix + hashlib.sha256(
        independent_domain + independent_canonical
    ).hexdigest()
    assert independent_id == EXPECTED_QUALIFICATION_RECEIPT_ID
    assert receipt.qualification_receipt_id == EXPECTED_QUALIFICATION_RECEIPT_ID


@pytest.mark.parametrize("mutation", ("three-authorities", "run-id", "aliased-keys", "labels-list"))
def test_prebuild_attack_matrix_returns_atomic_empty_root_rejection(
    mutation,
    profile,
    authorities,
) -> None:
    labels, strata = _metadata()
    candidate_authorities = authorities
    run_id = _run_id()
    keys = _keys()
    candidate_labels = labels
    if mutation == "three-authorities":
        candidate_authorities = (*authorities, authorities[0])
    elif mutation == "run-id":
        run_id = b"X" * 32
    elif mutation == "aliased-keys":
        keys = object.__new__(TrustedWireKeySourcesV2)
        object.__setattr__(keys, "shuffle_ikm", b"K" * 32)
        object.__setattr__(keys, "id_ikm", b"K" * 32)
        object.__setattr__(keys, "padding_ikm", b"P" * 32)
    elif mutation == "labels-list":
        candidate_labels = list(labels)
    result = q.qualify_phase2b_formal_trusted_wire_covert_scaffold_v1(
        profile=profile,
        authorities=candidate_authorities,
        run_id=run_id,
        key_sources=keys,
        auditor_labels=candidate_labels,
        auditor_strata=strata,
        invariance=_invariance(),
    )
    assert type(result) is q.Phase2BFormalTrustedWireCovertQualificationRejectionV1
    assert result.disposition is q.QualificationDispositionV1.REJECTED
    assert result.batch_id is None
    assert result.replay_receipt_id is None
    assert result.envelope_ids == ()
    assert result.namespace_audit_ids == ()
    assert result.structural_receipt_id is None
    assert result.covert_mechanics_receipt_id is None
    assert result.qualification_receipt_id is None


def test_invariance_attack_after_public_build_returns_no_partial_roots(
    monkeypatch,
    profile,
    authorities,
    qualified_receipt,
) -> None:
    labels, strata = _metadata()
    monkeypatch.setattr(q, "build_trusted_wire_batch_v2", lambda **_: qualified_receipt.batch)
    monkeypatch.setattr(
        q,
        "replay_typed_trusted_wire_batch_v2",
        lambda **_: qualified_receipt.replay,
    )
    result = q.qualify_phase2b_formal_trusted_wire_covert_scaffold_v1(
        profile=profile,
        authorities=authorities,
        run_id=_run_id(),
        key_sources=_keys(),
        auditor_labels=labels,
        auditor_strata=strata,
        invariance=_invariance(changed=True),
    )
    assert type(result) is q.Phase2BFormalTrustedWireCovertQualificationRejectionV1
    assert result.reason == q.FAIL_COVERT_STRUCTURE
    assert result.batch_id is None
    assert result.envelope_ids == ()
    assert result.namespace_audit_ids == ()


def test_receipt_tamper_is_rejected(qualified_receipt) -> None:
    attacked = object.__new__(type(qualified_receipt))
    for item in fields(qualified_receipt):
        object.__setattr__(
            attacked,
            item.name,
            True
            if item.name == "formal_wire_builder_implemented"
            else getattr(qualified_receipt, item.name),
        )
    with pytest.raises(ValueError, match="permanent nonclaim"):
        q.verify_phase2b_formal_trusted_wire_covert_qualification_receipt_v1(
            attacked
        )


def test_entrypoint_has_no_filesystem_docker_network_or_key_generation_path() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    class_fields: dict[str, tuple[str, ...]] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in {
            "Phase2BFormalTrustedWireCovertQualificationRejectionV1",
            "Phase2BFormalTrustedWireCovertQualificationReceiptV1",
        }:
            names = tuple(
                item.target.id
                for item in node.body
                if isinstance(item, ast.AnnAssign)
                and isinstance(item.target, ast.Name)
            )
            assert len(names) == len(set(names))
            class_fields[node.name] = names
    assert class_fields[
        "Phase2BFormalTrustedWireCovertQualificationRejectionV1"
    ] == tuple(item.name for item in fields(
        q.Phase2BFormalTrustedWireCovertQualificationRejectionV1
    ))
    assert class_fields[
        "Phase2BFormalTrustedWireCovertQualificationReceiptV1"
    ] == tuple(item.name for item in fields(
        q.Phase2BFormalTrustedWireCovertQualificationReceiptV1
    ))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imported.isdisjoint(
        {"pathlib", "os", "subprocess", "socket", "secrets", "docker", "requests"}
    )
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert calls.isdisjoint({"open", "exec", "eval", "compile", "__import__"})
    assert DOC.exists()
