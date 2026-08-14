from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import pytest

from hegel_machine import phase2b_c1_container_actor_profile_v1 as v1
from hegel_machine import phase2b_c1_container_actor_profile_v2 as v2
from hegel_machine.phase2b_c1_container_actor_profile_v2 import (
    FAIL_PROFILE_CLAIM_BOUNDARY,
    FAIL_PROFILE_SCHEMA,
    PHASE2B_C1_CONTAINER_ACTOR_AUTHORITY_CLASS_V2,
    PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CANONICAL_SHA256,
    PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CONTENT_DOMAIN,
    PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CONTENT_ID,
    PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CONTENT_PREFIX,
    PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_ID,
    PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_SCHEMA,
    PROFILE_PATH,
    Phase2BC1ContainerActorProfileErrorV2,
    Phase2BC1ContainerActorProfileSnapshotV2,
    load_phase2b_c1_container_actor_profile_v2,
    validate_phase2b_c1_container_actor_profile_v2,
)


_V1_CONFIG_SHA256 = "22908060372947bdedee30dd4c2920daec4a998b3dc9e0f50f4f3bb5f2e586ad"
_SCORING_SOURCE_SHA256 = "c3c12e9cd72a930b9ca8667aab22c4eb3675ceab6249b0b26ba940bf794e7f11"
_ADDED_INPUTS = (
    "COMMITTED_BLINDED_INPUT_ARCHIVE",
    "PUBLIC_PARTITION_SPEC",
    "EXECUTION_FREEZE_MANIFEST",
    "SCORER_RUN_MANIFEST",
)
_LOGICAL_SECTIONS = (
    "EXECUTION_FREEZE",
    "RAW_INPUT",
    "COMMITTED_PREDICTION",
    "PUBLIC_PARTITION_720_240",
    "ANSWER_ROWS_720_AND_OPENING",
    "GATE_ROWS_720_AND_OPENING",
    "PREDICTION_IF_ABSENT_RECEIPT_ID",
    "SIGNED_REVEAL_TOKEN_ID",
    "SCORER_RUN_MANIFEST",
)


def _source_config() -> dict[str, object]:
    value = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    assert type(value) is dict
    return value


def _effective_profile() -> dict[str, object]:
    value = v1._deep_thaw(load_phase2b_c1_container_actor_profile_v2().profile)
    assert type(value) is dict
    return value


def test_load_returns_content_addressed_deeply_immutable_effective_profile() -> None:
    first = load_phase2b_c1_container_actor_profile_v2()
    second = load_phase2b_c1_container_actor_profile_v2()
    plain = v1._deep_thaw(first.profile)
    canonical = v1._canonical_profile_bytes(plain)
    expected_digest = hashlib.sha256(
        PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CONTENT_DOMAIN + canonical
    ).hexdigest()
    assert first is not second
    assert first.profile is not second.profile
    assert isinstance(first.profile, MappingProxyType)
    assert first.profile == second.profile
    assert first.schema_version == PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_SCHEMA
    assert first.profile_id == PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_ID
    assert first.authority_class == PHASE2B_C1_CONTAINER_ACTOR_AUTHORITY_CLASS_V2
    assert first.profile_content_id == (
        PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CONTENT_PREFIX + expected_digest
    )
    assert first.profile_content_id == PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CONTENT_ID
    assert first.canonical_profile_sha256 == hashlib.sha256(canonical).hexdigest()
    assert first.canonical_profile_sha256 == (
        PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CANONICAL_SHA256
    )
    assert first.source_profile_sha256 == hashlib.sha256(PROFILE_PATH.read_bytes()).hexdigest()
    assert first.bound_source_sha256_verified is True
    with pytest.raises(TypeError):
        first.profile["profile_id"] = "attacked"
    with pytest.raises(TypeError):
        first.profile["actor_roles"][5]["read_only_input_classes"] = ()


def test_snapshot_constructor_is_private() -> None:
    with pytest.raises(TypeError, match="privately issued"):
        Phase2BC1ContainerActorProfileSnapshotV2()


def test_v2_is_an_exact_v1_conservative_projection_except_declared_extension() -> None:
    effective = _effective_profile()
    projected = v2._project_effective_profile_to_v1(effective)
    expected_v1 = v1._deep_thaw(
        v1.load_phase2b_c1_container_actor_profile_v1().profile
    )
    assert projected == expected_v1
    v1.validate_phase2b_c1_container_actor_profile_v1(projected)

    assert effective["schema_version"] == PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_SCHEMA
    assert effective["profile_id"] == PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_ID
    assert effective["authority_class"] == (
        expected_v1["authority_class"]
    )
    assert effective["owner_accepted_authority_requirement"] == (
        expected_v1["owner_accepted_authority_requirement"]
    )
    assert effective["claim_policy"] == expected_v1["claim_policy"]


def test_supersession_is_explicit_and_binds_the_unchanged_v1_source() -> None:
    config = _source_config()
    supersession = config["supersession_policy"]
    assert supersession == {
        "supersedes_schema_version": v1.PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V1_SCHEMA,
        "supersedes_profile_id": v1.PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V1_ID,
        "supersedes_source_path": "config/phase2b_c1_container_actor_profile_v1.json",
        "supersedes_source_sha256": _V1_CONFIG_SHA256,
        "supersession_kind": (
            "STRICT_CONSERVATIVE_EXTENSION_SCORER_POST_REVEAL_REPLAY_INPUTS_ONLY"
        ),
        "owner_accepted_authority_requirement_reused_exactly": True,
        "v1_profile_remains_historical_record": True,
        "new_attempts_must_bind_v2_profile_content_id": True,
    }
    v1_path = v2.PROJECT_ROOT / supersession["supersedes_source_path"]
    assert hashlib.sha256(v1_path.read_bytes()).hexdigest() == _V1_CONFIG_SHA256


def test_scorer_post_reveal_dataflow_expansion_is_exact_and_role_local() -> None:
    v1_profile = v1.load_phase2b_c1_container_actor_profile_v1().profile
    v2_profile = load_phase2b_c1_container_actor_profile_v2().profile
    v1_roles = v1_profile["actor_roles"]
    v2_roles = v2_profile["actor_roles"]

    assert v2_roles[:5] == v1_roles[:5]
    assert v2_roles[5]["role"] == "SCORER_EVALUATOR"
    old_inputs = v1_roles[5]["read_only_input_classes"]
    new_inputs = v2_roles[5]["read_only_input_classes"]
    assert new_inputs == old_inputs + _ADDED_INPUTS
    assert {
        "PREDICTION_STORE_IF_ABSENT_RECEIPT",
        "SIGNED_REVEAL_TOKEN",
        "COMMITTED_PREDICTION_ARCHIVE",
        "REVEALED_ANSWER_MANIFEST",
        "REVEALED_GATE_INPUT_MANIFEST",
    } <= set(new_inputs)
    assert "PRE_REVEAL_ANSWER_ACCESS" in v2_roles[5]["forbidden_input_classes"]
    assert v2_profile["scorer_dataflow_extension"]["availability_stage"] == (
        "POST_REVEAL_ONLY"
    )


def test_single_bundle_namespace_and_exact_report_filenames_are_frozen() -> None:
    policy = load_phase2b_c1_container_actor_profile_v2().profile[
        "scorer_dataflow_extension"
    ]
    assert policy["single_bundle_namespace"] is True
    assert policy["nine_independent_input_files_allowed"] is False
    assert policy["canonical_input_bundle_filename"] == (
        "phase2b_scorer_bundle_v1.bin"
    )
    assert policy["logical_sections_in_order"] == _LOGICAL_SECTIONS
    assert policy["canonical_output_report_filenames_in_order"] == (
        "score_mechanics_report_v1.json",
        "available_gate_mechanics_report_v1.json",
    )
    assert policy["runtime_receipt_transport"] == "STDOUT_ONLY_NO_THIRD_FILE"


def test_current_non_authoritative_scoring_mechanics_source_is_exactly_bound() -> None:
    profile = load_phase2b_c1_container_actor_profile_v2().profile
    extension_binding = profile["scorer_dataflow_extension"][
        "scoring_mechanics_source_binding"
    ]
    control_binding = profile["base_control_bindings"][
        "unsealed_960_scoring_mechanics_source"
    ]
    source_path = v2.PROJECT_ROOT / extension_binding["path"]

    assert extension_binding["sha256"] == _SCORING_SOURCE_SHA256
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == _SCORING_SOURCE_SHA256
    assert control_binding["path"] == extension_binding["path"]
    assert control_binding["sha256"] == extension_binding["sha256"]
    assert extension_binding["claim_level"] == (
        "NON_AUTHORITATIVE_SCORING_MECHANICS_ONLY"
    )


def test_source_config_and_effective_profile_validation_are_fail_closed() -> None:
    source = _source_config()
    validated_source = v2._validate_source_config(source)
    assert isinstance(validated_source, MappingProxyType)

    effective = _effective_profile()
    validated_effective = validate_phase2b_c1_container_actor_profile_v2(effective)
    assert isinstance(validated_effective, MappingProxyType)

    attacked = _effective_profile()
    attacked["actor_roles"][5]["read_only_input_classes"].remove(
        "COMMITTED_BLINDED_INPUT_ARCHIVE"
    )
    with pytest.raises(Phase2BC1ContainerActorProfileErrorV2) as caught:
        validate_phase2b_c1_container_actor_profile_v2(attacked)
    assert caught.value.code == FAIL_PROFILE_SCHEMA


@pytest.mark.parametrize(
    "mutation,expected_code",
    [
        (
            lambda value: value["supersession_policy"].update(
                {"supersedes_source_sha256": "0" * 64}
            ),
            FAIL_PROFILE_SCHEMA,
        ),
        (
            lambda value: value["scorer_dataflow_extension"].update(
                {"canonical_input_bundle_filename": "wrong.bin"}
            ),
            FAIL_PROFILE_SCHEMA,
        ),
        (
            lambda value: value["scorer_dataflow_extension"].update(
                {"actual_scoring_claimed": True}
            ),
            FAIL_PROFILE_SCHEMA,
        ),
        (
            lambda value: value["scorer_dataflow_extension"][
                "logical_sections_in_order"
            ].reverse(),
            FAIL_PROFILE_SCHEMA,
        ),
    ],
    ids=("v1-binding", "bundle-filename", "actual-claim", "section-order"),
)
def test_source_config_drift_is_rejected(mutation, expected_code: str) -> None:
    value = _source_config()
    mutation(value)
    with pytest.raises(Phase2BC1ContainerActorProfileErrorV2) as caught:
        v2._validate_source_config(value)
    assert caught.value.code == expected_code


def test_every_actual_and_permanent_nonclaim_remains_false_without_live_operations() -> None:
    snapshot = load_phase2b_c1_container_actor_profile_v2()
    claims = snapshot.profile["claim_policy"]
    assert all(value is False for value in claims["actual_run_claims_required_false"].values())
    assert all(value is False for value in claims["permanent_nonclaims_required_false"].values())
    extension = snapshot.profile["scorer_dataflow_extension"]
    assert extension["profile_loader_executes_scorer"] is False
    assert extension["actual_scoring_claimed"] is False
    assert extension["formal_gate_evaluation_claimed"] is False
    assert extension["effect_evidence_claimed"] is False
    assert extension["c1_exit_evidence_claimed"] is False
    assert snapshot.docker_invoked is False
    assert snapshot.keys_generated is False
    assert snapshot.signatures_generated is False
    assert snapshot.recognizer_executed is False
    assert snapshot.scoring_performed is False
    assert snapshot.c1_exit_evidence is False


def test_only_the_three_new_v2_paths_are_in_this_change() -> None:
    expected = {
        Path("config/phase2b_c1_container_actor_profile_v2.json"),
        Path("src/hegel_machine/phase2b_c1_container_actor_profile_v2.py"),
        Path("tests/test_phase2b_c1_container_actor_profile_v2.py"),
    }
    assert all((v2.PROJECT_ROOT / path).is_file() for path in expected)
