from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
import tempfile

import pytest

import hegel_machine.phase2b_c1_live_docker_ceremony_v1 as live_v1
import hegel_machine.phase2b_c1_live_docker_ceremony_v2 as ceremony
from hegel_machine import phase2b_c1_logical_stage_authority_v2 as authority
from hegel_machine import phase2b_c1_six_role_wire_v1 as six

from test_phase2b_c1_live_docker_ceremony_v1 import (
    FakeDocker,
    InjectedBaseException,
    _install_fake,
    _request,
)


EXPECTED_OPERATIONS = (
    *((index, role, "QUALIFY") for index, role in enumerate(
        six.PHASE2B_C1_SIX_ROLE_WIRE_V1_ROLE_ORDER, 1
    )),
    *((index, role, "KEYGEN") for index, role in enumerate(
        six.PHASE2B_C1_SIX_ROLE_WIRE_V1_ROLE_ORDER, 1
    )),
    (1, six.Phase2BC1SixRoleV1.GOVERNANCE_ANCHOR, "GOVERNANCE_SIGN"),
    *((index, role, "LOGICAL_STAGE_AUTHORIZATION_SIGN")
      for index, role in enumerate(six.PHASE2B_C1_SIX_ROLE_WIRE_V1_ROLE_ORDER, 1)),
    (3, six.Phase2BC1SixRoleV1.STORE_RECEIPT, "STORE_IF_ABSENT_AND_SIGN"),
    (3, six.Phase2BC1SixRoleV1.STORE_RECEIPT,
     "LOGICAL_STAGE_AUTHORIZATION_STORE_IF_ABSENT_AND_SIGN"),
    (4, six.Phase2BC1SixRoleV1.TIME_ATTESTOR, "TIME_GENESIS_AND_SIGN"),
)
EXPECTED_PUBLIC_SURFACE = {
    "FAIL_AUTHORITY", "FAIL_OPERATION_BINDING",
    "PHASE2B_C1_LIVE_DOCKER_CEREMONY_V2_CLAIM_LEVEL",
    "PHASE2B_C1_LIVE_DOCKER_CEREMONY_V2_COMPLETE_REASON",
    "PHASE2B_C1_LIVE_DOCKER_CEREMONY_V2_VERSION",
    "Phase2BC1LiveDockerCeremonyReceiptV2",
    "execute_phase2b_c1_live_docker_ceremony_v2",
}


@pytest.fixture
def linux_tmp_path() -> Path:
    with tempfile.TemporaryDirectory(
        prefix="hegel-live-v2-test-", dir="/tmp"
    ) as directory:
        yield Path(directory)


class SecondStoreDriftDocker(FakeDocker):
    def __init__(self) -> None:
        super().__init__()
        self.store_calls = 0

    def __call__(self, command, **kwargs):
        args = tuple(command)[2:]
        if args[:3] == ("container", "exec", "--interactive") and args[-4] == "store-and-sign":
            self.store_calls += 1
            self.store_output_drift = self.store_calls == 2
        return super().__call__(command, **kwargs)


class BadAuthoritySignatureDocker(FakeDocker):
    def __init__(self) -> None:
        super().__init__()
        self.sign_calls = 0

    def __call__(self, command, **kwargs):
        result = super().__call__(command, **kwargs)
        args = tuple(command)[2:]
        if args[:3] == ("container", "exec", "--interactive") and args[-4] == "sign":
            self.sign_calls += 1
            if self.sign_calls == 7:
                value = live_v1._decode_json_v1(
                    result.stdout, code=live_v1.FAIL_SIGNATURE,
                    label="test bad authority signature",
                )
                value["signature_hex"] = "00" * 64
                return live_v1._BoundedProcessV1(
                    command=result.command, returncode=result.returncode,
                    stdout=(live_v1._canonical_json_bytes(value) + b"\n"),
                    stderr=result.stderr,
                    started_monotonic_ns=result.started_monotonic_ns,
                    finished_monotonic_ns=result.finished_monotonic_ns,
                )
        return result


def _receipt_graph(result: ceremony.Phase2BC1LiveDockerCeremonyReceiptV2):
    return (
        result.legacy_context, result.legacy_registry,
        result.legacy_store_receipt, result.legacy_logical_genesis_token,
        result.legacy_wire_verification, result.authority_context,
        result.authority_registry, result.authority_store_receipt,
        result.authority_verification, result.stored_v1_registry_sha256,
        result.stored_v2_registry_sha256, result.logical_genesis_record_sha256,
    )


def _receipt_id_for(
    result: ceremony.Phase2BC1LiveDockerCeremonyReceiptV2,
    runtime_bindings,
) -> str:
    return ceremony._receipt_id(
        request=result.request, graph=_receipt_graph(result),
        runtime_bindings=runtime_bindings,
        operation_bindings=result.operation_bindings,
        control_sha=result.docker_control_plane_binding_sha256,
        recognizer_inspect_sha=result.recognizer_image_inspect_sha256,
        scorer_inspect_sha=result.scorer_image_inspect_sha256,
    )


def test_success_runs_one_six_container_lifecycle_with_exact_twenty_two_operations(
    monkeypatch: pytest.MonkeyPatch, linux_tmp_path: Path,
) -> None:
    fake = FakeDocker()
    _install_fake(monkeypatch, linux_tmp_path, fake)
    result = ceremony.execute_phase2b_c1_live_docker_ceremony_v2(_request())

    assert tuple(
        (row.purpose_id, row.role, row.operation) for row in result.operation_bindings
    ) == EXPECTED_OPERATIONS
    assert len(result.operation_bindings) == 22
    assert result.legacy_wire_verification.disposition is (
        six.Phase2BC1SixRoleWireDispositionV1.CALLER_SUPPLIED_SIX_ROLE_WIRE_GRAPH_VERIFIED_NOT_AUTHORITY
    )
    assert result.authority_verification.disposition is (
        authority.Phase2BC1LogicalStageAuthorityDispositionV2.CALLER_SUPPLIED_OWNER_ACCEPTED_V2_AUTHORITY_GRAPH_VERIFIED_NOT_EXECUTION
    )
    assert result.authority_context.body.legacy_context_id == result.legacy_context.context_id
    assert (
        result.authority_context.body.legacy_execution_registry_id
        == result.legacy_registry.execution_registry_id
    )
    assert result.authority_registry.body.required_signer_roles == (
        six.PHASE2B_C1_SIX_ROLE_WIRE_V1_ROLE_ORDER
    )
    assert len(result.authority_registry.role_acceptance_signatures) == 6
    assert tuple(
        signature.signer_key_id
        for signature in result.authority_registry.role_acceptance_signatures
    ) == tuple(key.key_id for key in result.legacy_registry.body.role_keys)
    assert result.legacy_store_receipt.body.object_key != (
        result.authority_store_receipt.body.object_key
    )
    assert result.legacy_store_receipt.body.condition == "IF_ABSENT"
    assert result.authority_store_receipt.body.condition == "IF_ABSENT"
    assert result.legacy_store_receipt.body.new_generation == 1
    assert result.authority_store_receipt.body.new_generation == 1
    assert result.legacy_logical_genesis_token.body.version == (
        six.PHASE2B_C1_SIX_ROLE_WIRE_V1_VERSION
    )
    assert result.legacy_logical_genesis_token.body.stage is (
        six.Phase2BC1LogicalStageV1.PREREGISTERED
    )
    assert result.legacy_logical_genesis_token.body.logical_sequence == 0
    assert result.legacy_wire_verification.actual_execution_claimed is False
    assert result.legacy_wire_verification.execution_authorized is False
    assert result.authority_verification.actual_campaign_eligible is False
    assert result.authority_verification.actual_execution_claimed is False
    assert result.authority_verification.execution_authorized is False
    for name in ceremony._FALSE_RESULT_CLAIMS:
        assert getattr(result, name) is False
    assert fake.events.count("governance-sign") == 7
    assert fake.events.count("store-and-sign") == 2
    assert fake.events.count("time-and-sign") == 1
    assert fake.events.index("store-and-sign") > fake.events.index("governance-sign")
    assert fake.events.index("time-and-sign") > max(
        index for index, event in enumerate(fake.events) if event == "store-and-sign"
    )
    assert fake.containers == {}
    assert len(fake.volumes) == 11
    assert fake.events.count("container-rm") == 17
    assert "volume-rm" not in fake.events


def test_runtime_binding_is_fully_content_addressed_and_forgery_is_rejected(
    monkeypatch: pytest.MonkeyPatch, linux_tmp_path: Path,
) -> None:
    fake = FakeDocker()
    _install_fake(monkeypatch, linux_tmp_path, fake)
    result = ceremony.execute_phase2b_c1_live_docker_ceremony_v2(_request())
    first = result.role_runtime_bindings[0]
    assert tuple(
        name for name, _ in ceremony._runtime_binding_projection((first,))[0]
    ) == tuple(item.name for item in fields(type(first)))
    mutations = {
        "purpose_id": 2,
        "role": six.Phase2BC1SixRoleV1.CUSTODIAN,
        "image_digest": "sha256:" + "0" * 64,
        "container_name": "hegel-p2b-c1-" + "0" * 24 + "-p1",
        "container_id": "0" * 64,
        "purpose_private_volumes": (("/state", "hp2bc1v-" + "0" * 48),),
        "qualification_sha256": "0" * 64,
        "namespace_identities": tuple(
            (name, f"{name}:[999]") for name in ("pid", "mnt", "net", "ipc", "uts")
        ),
        "openssl_sha256": "0" * 64,
        "openssl_version": "OpenSSL mutation",
    }
    for name, mutation in mutations.items():
        changed = object.__new__(type(first))
        for field in fields(type(first)):
            object.__setattr__(
                changed, field.name,
                mutation if field.name == name else getattr(first, field.name),
            )
        candidate = (changed, *result.role_runtime_bindings[1:])
        assert _receipt_id_for(result, candidate) != result.ceremony_receipt_id
    mutated_first = replace(
        result.role_runtime_bindings[0], qualification_sha256="0" * 64
    )
    mutated = (mutated_first, *result.role_runtime_bindings[1:])
    assert _receipt_id_for(result, result.role_runtime_bindings) == (
        result.ceremony_receipt_id
    )
    assert _receipt_id_for(result, mutated) != result.ceremony_receipt_id

    forged = object.__new__(type(result))
    for field in fields(type(result)):
        object.__setattr__(forged, field.name, getattr(result, field.name))
    object.__setattr__(forged, "role_runtime_bindings", mutated)
    with pytest.raises(ValueError, match="content address drift"):
        forged.__post_init__()

    duplicated = object.__new__(type(result))
    for field in fields(type(result)):
        object.__setattr__(duplicated, field.name, getattr(result, field.name))
    object.__setattr__(
        duplicated, "role_runtime_bindings",
        (result.role_runtime_bindings[0], result.role_runtime_bindings[0],
         *result.role_runtime_bindings[2:]),
    )
    with pytest.raises(ValueError, match="identity/shape drift"):
        duplicated.__post_init__()


def test_second_independent_store_drift_fails_before_genesis_and_removes_all_state(
    monkeypatch: pytest.MonkeyPatch, linux_tmp_path: Path,
) -> None:
    fake = SecondStoreDriftDocker()
    _install_fake(monkeypatch, linux_tmp_path, fake)
    with pytest.raises(live_v1.Phase2BC1LiveDockerCeremonyErrorV1) as captured:
        ceremony.execute_phase2b_c1_live_docker_ceremony_v2(_request())
    assert captured.value.code == live_v1.FAIL_STORE
    assert fake.store_calls == 2
    assert "time-and-sign" not in fake.events
    assert fake.containers == {}
    assert fake.volumes == set()
    assert fake.events.count("volume-rm") == 11


def test_sixth_authority_signature_corruption_is_rejected_after_internal_reverification(
    monkeypatch: pytest.MonkeyPatch, linux_tmp_path: Path,
) -> None:
    fake = BadAuthoritySignatureDocker()
    _install_fake(monkeypatch, linux_tmp_path, fake)
    with pytest.raises(live_v1.Phase2BC1LiveDockerCeremonyErrorV1) as captured:
        ceremony.execute_phase2b_c1_live_docker_ceremony_v2(_request())
    assert captured.value.code == ceremony.FAIL_AUTHORITY
    assert fake.sign_calls == 7
    assert fake.events.count("store-and-sign") == 2
    assert fake.events.count("time-and-sign") == 1
    assert fake.containers == {}
    assert fake.volumes == set()


def test_public_executor_reuses_deadline_exhaustion_and_cleans_owned_side_effects(
    monkeypatch: pytest.MonkeyPatch, linux_tmp_path: Path,
) -> None:
    fake = FakeDocker()
    _install_fake(monkeypatch, linux_tmp_path, fake)

    def clock() -> float:
        if live_v1._CEREMONY_CLEANUP_PHASE_V1.get():
            return 200.0
        return 700.0 if "create-1" in fake.events else 100.0

    monkeypatch.setattr(live_v1.time, "monotonic", clock)

    def deadline_bounded_fake(command, **kwargs):
        live_v1._bounded_by_ceremony_deadline_v1(60)
        return fake(command, **kwargs)

    monkeypatch.setattr(
        live_v1, "_run_docker_command_v1", deadline_bounded_fake
    )
    with pytest.raises(live_v1.Phase2BC1LiveDockerCeremonyErrorV1) as captured:
        ceremony.execute_phase2b_c1_live_docker_ceremony_v2(_request())
    assert captured.value.code == live_v1.FAIL_DOCKER_TIMEOUT
    assert "create-1" in fake.events
    assert fake.containers == {}
    assert fake.volumes == set()


def test_public_executor_reconciles_create_after_side_effect_ambiguity(
    monkeypatch: pytest.MonkeyPatch, linux_tmp_path: Path,
) -> None:
    fake = FakeDocker(actor_create_fault_purpose=3)
    _install_fake(monkeypatch, linux_tmp_path, fake)
    with pytest.raises(live_v1.Phase2BC1LiveDockerCeremonyErrorV1) as captured:
        ceremony.execute_phase2b_c1_live_docker_ceremony_v2(_request())
    assert captured.value.code == live_v1.FAIL_DOCKER_TIMEOUT
    assert fake.actor_create_fault_raised is True
    assert fake.containers == {}
    assert fake.volumes == set()
    removed = [
        call[0][-1] for call in fake.calls
        if call[0][2:5] == ("container", "rm", "--force")
    ]
    assert removed
    assert all(len(item) == 64 for item in removed)


def test_public_executor_retention_proof_failure_removes_all_volumes(
    monkeypatch: pytest.MonkeyPatch, linux_tmp_path: Path,
) -> None:
    fake = FakeDocker()
    _install_fake(monkeypatch, linux_tmp_path, fake)

    def fail_retention(*args, **kwargs):
        raise InjectedBaseException("retention proof fault")

    monkeypatch.setattr(live_v1, "_prove_volumes_retained_v1", fail_retention)
    with pytest.raises(live_v1.Phase2BC1LiveDockerCeremonyErrorV1) as captured:
        ceremony.execute_phase2b_c1_live_docker_ceremony_v2(_request())
    assert captured.value.code == live_v1.FAIL_CLEANUP
    assert fake.containers == {}
    assert fake.volumes == set()
    assert fake.events.count("volume-rm") == 11


def test_public_executor_receipt_issuance_failure_removes_retained_volumes(
    monkeypatch: pytest.MonkeyPatch, linux_tmp_path: Path,
) -> None:
    fake = FakeDocker()
    _install_fake(monkeypatch, linux_tmp_path, fake)

    def fail_receipt(**kwargs):
        raise InjectedBaseException("V2 receipt issuance fault")

    monkeypatch.setattr(ceremony, "_issue_receipt", fail_receipt)
    with pytest.raises(InjectedBaseException):
        ceremony.execute_phase2b_c1_live_docker_ceremony_v2(_request())
    assert fake.containers == {}
    assert fake.volumes == set()
    assert fake.events.count("volume-rm") == 11


def test_v1_private_helpers_are_reused_without_copying_or_runtime_monkeypatching() -> None:
    assert ceremony.live_v1 is live_v1
    assert ceremony.live_v1._qualify_all_roles_v1 is live_v1._qualify_all_roles_v1
    assert ceremony.live_v1._generate_role_keys_v1 is live_v1._generate_role_keys_v1
    assert ceremony.live_v1._inspect_actor_v1 is live_v1._inspect_actor_v1
    assert ceremony.live_v1._cleanup_containers_best_effort_v1 is (
        live_v1._cleanup_containers_best_effort_v1
    )
    assert ceremony.live_v1._cleanup_volumes_best_effort_v1 is (
        live_v1._cleanup_volumes_best_effort_v1
    )
    assert "setattr(" not in Path(ceremony.__file__).read_text(encoding="utf-8")


def test_receipt_is_private_claims_are_closed_and_public_surface_is_literal() -> None:
    with pytest.raises(TypeError, match="privately issued"):
        ceremony.Phase2BC1LiveDockerCeremonyReceiptV2()
    assert set(ceremony.__all__) == EXPECTED_PUBLIC_SURFACE
    assert len(ceremony.__all__) == len(EXPECTED_PUBLIC_SURFACE)
    assert tuple(item.name for item in fields(
        ceremony.Phase2BC1LiveDockerCeremonyReceiptV2
    ))[-len(ceremony._FALSE_RESULT_CLAIMS):] == ceremony._FALSE_RESULT_CLAIMS
