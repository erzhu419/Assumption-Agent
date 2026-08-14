from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import pytest

from hegel_machine import phase2b_c1_container_actor_profile_v2 as actor_v2
from hegel_machine import phase2b_c1_logical_stage_authority_v2 as logical_v2
from hegel_machine import phase2b_c1_owner_actual_unsealed_960_authority_v1 as auth
from hegel_machine.phase2b_freeze_v1 import frozen_phase2b_exact_freeze
from hegel_machine.phase2b_protocol import frozen_phase2b_protocol


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/phase2b_c1_owner_actual_unsealed_960_authorization_v1.json"
SOURCE = ROOT / "src/hegel_machine/phase2b_c1_owner_actual_unsealed_960_authority_v1.py"
DOC = ROOT / "docs/Hegel_Machine_Phase2B_C1_Owner_Authorization_From_Zero_Actual_Unsealed_960_v1.md"

EXPECTED_BODY_SHA256 = "02b43e6bd2bc34032f9458da4553c45b144e5cdbefc9f6e893cf8851473e0a27"
EXPECTED_AUTHORIZATION_ID = (
    "phase2b_c1_owner_actual_unsealed_960_authorization_v1_"
    "dd2cd6a98b429fe01f172daf16ea70404c91356e40e9e42daed4cb5416a5c50d"
)


def _payload() -> bytes:
    return CONFIG.read_bytes()


def _decoded() -> dict[str, object]:
    value = json.loads(_payload())
    assert type(value) is dict
    return value


def test_independent_canonical_kat_and_content_id() -> None:
    value = _decoded()
    body = value["authorization_body"]
    independent_canonical = json.dumps(
        body,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    independent_domain = (
        b"HEGEL/PHASE2B/C1/OWNER_ACTUAL_UNSEALED_960_AUTHORIZATION/BODY/V1"
        + bytes([0])
    )
    assert hashlib.sha256(independent_canonical).hexdigest() == EXPECTED_BODY_SHA256
    independent_id = auth.PHASE2B_C1_OWNER_ACTUAL_UNSEALED_960_AUTHORIZATION_V1_PREFIX + hashlib.sha256(
        independent_domain + independent_canonical
    ).hexdigest()
    assert independent_id == EXPECTED_AUTHORIZATION_ID
    assert value["authorization_id"] == EXPECTED_AUTHORIZATION_ID
    assert auth.phase2b_c1_owner_actual_unsealed_960_authorization_id_v1(body) == (
        EXPECTED_AUTHORIZATION_ID
    )
    assert _payload() == json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def test_loader_returns_private_immutable_success_with_exact_claim_matrix() -> None:
    result = auth.load_phase2b_c1_owner_actual_unsealed_960_authorization_v1(
        _payload()
    )
    assert result.disposition is (
        auth.Phase2BC1OwnerActualUnsealed960AuthorizationDispositionV1.
        PROSPECTIVE_CONDITIONAL_AUTHORIZATION_VERIFIED_NOT_EXECUTION
    )
    assert result.reason is (
        auth.Phase2BC1OwnerActualUnsealed960AuthorizationReasonV1.
        EXACT_PROSPECTIVE_AUTHORIZATION_CONTENT_VERIFIED
    )
    assert result.authorization_id == EXPECTED_AUTHORIZATION_ID
    assert result.authorization_content_verified is True
    assert result.construction is True
    assert result.actual_unsealed_960_campaign_conditionally_authorized is True
    assert result.authorization_is_self_executing is False
    assert result.campaign_id is None
    assert result.attempt_id is None
    assert result.campaign_id_allocation_authorized is False
    assert result.actual_campaign_eligible is False
    assert result.execution_authorized is False
    assert result.formal_holdout_generation_authorized is False
    assert result.shadow_only is True
    assert result.legacy_logical_stage_fields_unchanged is True
    assert result.owner_signature_authenticated is False
    assert result.docker_invoked is False
    assert result.filesystem_accessed is False
    assert result.network_accessed is False
    assert result.keys_generated is False
    assert result.scoring_performed is False
    assert result.actual_execution_claimed is False
    assert result.c1_exit_evidence is False
    assert isinstance(result.policy, MappingProxyType)
    with pytest.raises(TypeError):
        result.policy["status"] = "ATTACKED"
    with pytest.raises(TypeError):
        result.policy["preflight_safety_boundary"]["execution_authorized"] = True


def test_result_constructor_is_private() -> None:
    with pytest.raises(TypeError, match="privately issued"):
        auth.Phase2BC1OwnerActualUnsealed960AuthorizationVerificationV1()


def test_scope_is_exactly_generator_three_baselines_machinery_and_one_future_campaign() -> None:
    policy = auth.load_phase2b_c1_owner_actual_unsealed_960_authorization_v1(
        _payload()
    ).policy
    assert policy is not None
    assert policy["authorized_scope_in_order"] == (
        "INDEPENDENT_HOLDOUT_GENERATOR_FROM_ZERO_CONSTRUCTION",
        "EMBEDDING_NEAREST_PROTOTYPE_BASELINE_REGISTRATION_CAMPAIGN",
        "FROZEN_LLM_SEMANTIC_ONLY_BASELINE_REGISTRATION_CAMPAIGN",
        "FLAT_LEARNED_TYPED_BASELINE_REGISTRATION_CAMPAIGN",
        "READINESS_AND_CAMPAIGN_MACHINERY_CONSTRUCTION",
        "ONE_FUTURE_FRESH_ACTUAL_UNSEALED_960_CAMPAIGN_AFTER_PREREQUISITES_AND_INDEPENDENT_LAUNCH_OVERLAY",
    )
    baselines = policy["baseline_registrations_in_order"]
    assert tuple(item["baseline_id"] for item in baselines) == (
        "embedding_nearest_prototype",
        "frozen_llm_semantic_only",
        "flat_learned_typed",
    )
    assert policy["prospective_authorization_claims"][
        "maximum_fresh_campaigns_authorized"
    ] == 1


def test_preflight_and_permanent_nonclaims_are_fail_closed() -> None:
    policy = auth.load_phase2b_c1_owner_actual_unsealed_960_authorization_v1(
        _payload()
    ).policy
    assert policy is not None
    preflight = policy["preflight_safety_boundary"]
    assert preflight["authorization_is_self_executing"] is False
    assert preflight["campaign_id"] is None
    assert preflight["attempt_id"] is None
    assert preflight["campaign_id_allocation_authorized"] is False
    assert preflight["actual_campaign_eligible"] is False
    assert preflight["execution_authorized"] is False
    assert preflight["formal_holdout_generation_authorized"] is False
    assert preflight["shadow_only"] is True
    assert preflight["prerequisites_verified"] is False
    assert preflight["independent_launch_overlay_verified"] is False
    assert all(
        value is False
        for value in policy[
            "authorization_artifact_permanent_nonclaims"
        ].values()
    )


def test_container_actor_threat_model_has_exact_types_and_polarity() -> None:
    policy = auth.load_phase2b_c1_owner_actual_unsealed_960_authorization_v1(
        _payload()
    ).policy
    assert policy is not None
    threat_model = policy["container_actor_threat_model"]
    assert dict(threat_model) == {
        "same_admin_controller": True,
        "owner_accepted_threat_model": True,
        "organizational_independence": False,
        "independent_human_actors": False,
        "technical_role_independence_profile_required": True,
        "technical_role_independence_live_verified": False,
        "remote_attestation": False,
        "hardware_key_nonexportability": False,
        "malicious_admin_resistance": False,
    }
    assert all(type(value) is bool for value in threat_model.values())


def test_old_profile_and_logical_stage_boundaries_have_not_drifted() -> None:
    bindings = _decoded()["authorization_body"]["exact_contract_bindings"]
    exact_freeze = frozen_phase2b_exact_freeze()
    protocol = frozen_phase2b_protocol()
    assert bindings["protocol_id"] == protocol.protocol_id
    assert bindings["exact_freeze_version"] == protocol.exact_freeze_version
    assert bindings["exact_freeze_version"] == exact_freeze.freeze_version
    assert bindings["exact_freeze_version"] == "hegel-freeze-p2b-p3-v1.0.1"
    assert bindings["exact_freeze_id"] == exact_freeze.freeze_id
    assert exact_freeze.formal_holdout_generation_authorized is False
    assert exact_freeze.shadow_only is True
    assert protocol.shadow_only is True
    assert protocol.active_promotion_enabled is False
    assert bindings["actor_profile_content_id"] == (
        actor_v2.PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CONTENT_ID
    )
    assert bindings["actor_profile_canonical_sha256"] == (
        actor_v2.PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CANONICAL_SHA256
    )
    assert bindings["logical_stage_wire_schema_id"] == (
        logical_v2.PHASE2B_C1_LOGICAL_STAGE_WIRE_SCHEMA_ID_V1
    )
    assert bindings["logical_stage_wire_policy_id"] == (
        logical_v2.PHASE2B_C1_LOGICAL_STAGE_WIRE_POLICY_ID_V2
    )
    assert bindings["stage_event_contract_manifest_id"] == (
        logical_v2.PHASE2B_C1_STAGE_EVENT_CONTRACT_MANIFEST_ID_V1
    )
    preflight = _decoded()["authorization_body"]["preflight_safety_boundary"]
    assert preflight["legacy_registry_policy_authorization_literal"] == (
        "NOT_AUTHORIZED_BY_FROZEN_ACTOR_PROFILE_V2_OR_EXECUTION_REGISTRY_V1"
    )
    assert preflight["legacy_actual_campaign_eligible"] is False
    assert preflight["legacy_actual_execution_claimed"] is False
    assert preflight["legacy_execution_authorized"] is False
    assert preflight["formal_holdout_generation_authorized"] is False
    assert preflight["shadow_only"] is True


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["authorization_body"].update({"unknown": False}),
        lambda value: value["authorization_body"]["preflight_safety_boundary"].update(
            {"execution_authorized": True}
        ),
        lambda value: value["authorization_body"][
            "prospective_authorization_claims"
        ].update({"construction": 1}),
        lambda value: value["authorization_body"]["container_actor_threat_model"].update(
            {"same_admin_controller": 1}
        ),
        lambda value: value["authorization_body"]["container_actor_threat_model"].update(
            {"technical_role_independence_live_verified": True}
        ),
        lambda value: value["authorization_body"][
            "baseline_registrations_in_order"
        ].append(
            copy.deepcopy(
                value["authorization_body"]["baseline_registrations_in_order"][0]
            )
        ),
        lambda value: value["authorization_body"]["baseline_registrations_in_order"].reverse(),
        lambda value: value.update(
            {
                "authorization_id": (
                    auth.PHASE2B_C1_OWNER_ACTUAL_UNSEALED_960_AUTHORIZATION_V1_PREFIX
                    + "0" * 64
                )
            }
        ),
    ],
    ids=(
        "unknown",
        "claim-tamper",
        "bool-type",
        "threat-type",
        "threat-polarity",
        "duplicate-baseline",
        "order",
        "id",
    ),
)
def test_decoded_unknown_type_duplicate_order_and_tamper_reject_atomically(mutation) -> None:
    value = _decoded()
    mutation(value)
    result = auth.verify_phase2b_c1_owner_actual_unsealed_960_authorization_v1(value)
    assert result.disposition is (
        auth.Phase2BC1OwnerActualUnsealed960AuthorizationDispositionV1.REJECTED
    )
    assert result.authorization_id is None
    assert result.policy is None
    assert result.authorization_content_verified is False
    assert result.construction is False
    assert result.actual_unsealed_960_campaign_conditionally_authorized is False
    assert result.actual_campaign_eligible is False
    assert result.execution_authorized is False
    assert result.formal_holdout_generation_authorized is False
    assert result.shadow_only is True


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"[]",
        b"{\"authorization_body\":null,\"authorization_body\":null,\"authorization_id\":null}",
        _payload().replace(b'\"authorization_epoch\":1', b'\"authorization_epoch\":1.0'),
        _payload().replace(b'\"authorization_epoch\":1', b'\"authorization_epoch\":NaN'),
        _payload() + b"\n",
        b"\xff",
    ],
    ids=("empty", "array", "duplicate-key", "float", "nan", "newline", "non-ascii"),
)
def test_byte_loader_rejects_noncanonical_and_invalid_inputs(payload: bytes) -> None:
    result = auth.load_phase2b_c1_owner_actual_unsealed_960_authorization_v1(payload)
    assert result.disposition is (
        auth.Phase2BC1OwnerActualUnsealed960AuthorizationDispositionV1.REJECTED
    )
    assert result.authorization_content_verified is False
    assert result.construction is False
    assert result.campaign_id is None
    assert result.attempt_id is None
    assert result.execution_authorized is False
    assert result.formal_holdout_generation_authorized is False
    assert result.shadow_only is True


def test_module_ast_has_no_side_effect_capability_or_forbidden_call_surface() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    forbidden_import_roots = {
        "asyncio",
        "docker",
        "http",
        "os",
        "pathlib",
        "requests",
        "shutil",
        "socket",
        "subprocess",
        "tempfile",
        "urllib",
    }
    forbidden_calls = {
        "__import__",
        "compile",
        "eval",
        "exec",
        "open",
        "system",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] not in forbidden_import_roots for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            assert node.module.split(".")[0] not in forbidden_import_roots
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in forbidden_calls


def test_public_surface_is_exact_and_contains_no_campaign_launcher() -> None:
    assert set(auth.__all__) == {
        "FAIL_AUTHORIZATION_CANONICAL",
        "FAIL_AUTHORIZATION_CLAIM_BOUNDARY",
        "FAIL_AUTHORIZATION_CONTENT_ID",
        "FAIL_AUTHORIZATION_INPUT",
        "FAIL_AUTHORIZATION_JSON",
        "FAIL_AUTHORIZATION_SCHEMA",
        "PHASE2B_C1_OWNER_ACTUAL_UNSEALED_960_AUTHORIZATION_V1_CLAIM_LEVEL",
        "PHASE2B_C1_OWNER_ACTUAL_UNSEALED_960_AUTHORIZATION_V1_DOCUMENT_ID",
        "PHASE2B_C1_OWNER_ACTUAL_UNSEALED_960_AUTHORIZATION_V1_DOMAIN",
        "PHASE2B_C1_OWNER_ACTUAL_UNSEALED_960_AUTHORIZATION_V1_ID",
        "PHASE2B_C1_OWNER_ACTUAL_UNSEALED_960_AUTHORIZATION_V1_PREFIX",
        "PHASE2B_C1_OWNER_ACTUAL_UNSEALED_960_AUTHORIZATION_V1_SCHEMA",
        "PHASE2B_C1_OWNER_ACTUAL_UNSEALED_960_AUTHORIZATION_V1_STATUS",
        "Phase2BC1OwnerActualUnsealed960AuthorizationDispositionV1",
        "Phase2BC1OwnerActualUnsealed960AuthorizationErrorV1",
        "Phase2BC1OwnerActualUnsealed960AuthorizationReasonV1",
        "Phase2BC1OwnerActualUnsealed960AuthorizationVerificationV1",
        "canonical_phase2b_c1_owner_actual_unsealed_960_authorization_body_bytes_v1",
        "load_phase2b_c1_owner_actual_unsealed_960_authorization_v1",
        "phase2b_c1_owner_actual_unsealed_960_authorization_id_v1",
        "verify_phase2b_c1_owner_actual_unsealed_960_authorization_v1",
    }
    assert not any(
        word in name.lower()
        for name in auth.__all__
        for word in ("launch", "execute", "docker", "score", "keygen", "allocate_campaign")
    )


def test_only_the_requested_four_new_files_are_declared_by_this_unit() -> None:
    expected = {CONFIG, SOURCE, DOC, Path(__file__)}
    assert all(path.is_file() for path in expected)
