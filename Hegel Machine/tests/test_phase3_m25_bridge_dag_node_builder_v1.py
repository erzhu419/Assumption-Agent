from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
from types import MappingProxyType

import pytest

from hegel_machine.phase3_m25_bridge_dag_node_builder_v1 import (
    BridgeDagNodeBuildError,
    BridgeDagNodeBuildInputsV1,
    BridgeDagPackageBuildInputsV1,
    FAIL_BASIS_BINDING,
    FAIL_CANDIDATE_BINDING,
    FAIL_CROSS_ROLE,
    FAIL_FIELD_SET,
    FAIL_PACKAGE_PREFLIGHT,
    FAIL_SIGNATURE_PHASE,
    M3ExecutionBindingContractFieldsV1,
    build_bridge_dag_nodes_v1,
    build_bridge_dag_replay_package_from_inputs_v1,
)
from hegel_machine.phase3_m25_bridge_full_dag_replay_v1 import (
    ROLE_SPECS,
    replay_bridge_dag_package_v1,
)
from hegel_machine.phase3_m25_formal_static_basis_v1 import (
    FormalStaticBasisV1,
    build_formal_static_basis_v1,
)
from hegel_machine.phase3_m25_rows_v1 import (
    generate_odd_role_rows_v1,
    generate_sink_role_rows_v1,
)
from hegel_machine.phase3_m25_wire_v1 import (
    AUDITED_PARENT_COMMIT_SHA1,
    candidate_content_root,
    git_sha1_commit_id,
    id_digest_v1,
)
from hegel_machine.strict_cbor_v1 import canonical_cbor_decode


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent

# Keep this fixture independent of the user's dirty worktree: the static basis
# consumes only exact Git blobs from this temporary, deterministic commit.
COMMITTED_BASIS_PATHS = (
    "Hegel Machine/config/phase3_container_actor_profile_v1.json",
    "Hegel Machine/config/phase3_internal_actor_seccomp_v1.json",
    "Hegel Machine/artifacts/phase3_dual_strict_capacity_replay_v1.json",
    "Hegel Machine/artifacts/phase3_shrink1_dual_capacity_replay_v1.json",
    "Hegel Machine/docs/Hegel_Machine_Phase3A_M25_Bit_Exact_Wire_Completion_Amendment.md",
    "Hegel Machine/docs/Hegel_Machine_Phase3A_M25_Exact_Wire_Errata_Resolution.md",
    "Hegel Machine/docs/Hegel_Machine_Phase3A_M25_Implementation_Closure_Addendum_v1.md",
    "Hegel Machine/docs/Hegel_Machine_Phase3A_M25_Formal_Static_Basis_Engineering_Freeze_v1.md",
    "Hegel Machine/docs/Hegel_Machine_Phase3_Shrink_Step1_Freeze_Decisions.md",
    "Hegel Machine/src/hegel_machine/phase3_dsl_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_dsl_core_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_shrink1_core_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m25_formal_static_basis_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m25_rows_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m25_wire_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_shrink1_registry_v1.py",
    "Hegel Machine/src/hegel_machine/strict_ast_shrink1_v1.py",
    "Hegel Machine/src/hegel_machine/strict_ast_v1.py",
    "Hegel Machine/src/hegel_machine/strict_cbor_v1.py",
    "Hegel Machine/rust/formal_bridge_m25/Cargo.lock",
    "Hegel Machine/rust/formal_bridge_m25/Cargo.toml",
    "Hegel Machine/rust/formal_bridge_m25/src/lib.rs",
    "Hegel Machine/rust/formal_bridge_m25/src/main.rs",
)


@dataclass(frozen=True, slots=True)
class SyntheticPublicFixture:
    node_inputs: BridgeDagNodeBuildInputsV1
    bridge_fields: MappingProxyType
    key_fields: MappingProxyType


def _copy_and_commit_static_basis(tmp_path: Path) -> FormalStaticBasisV1:
    repository = tmp_path / "repository"
    for relative in COMMITTED_BASIS_PATHS:
        source = REPOSITORY_ROOT / relative
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    python_binary = tmp_path / "synthetic-python-binary"
    rust_binary = tmp_path / "synthetic-rust-binary"
    python_binary.write_bytes(b"synthetic-public-python-binary\n")
    rust_binary.write_bytes(b"synthetic-public-rust-binary\n")
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(["git", "config", "user.email", "bridge-builder@example.invalid"], cwd=repository, check=True)
    subprocess.run(["git", "config", "user.name", "bridge-builder-test"], cwd=repository, check=True)
    subprocess.run(["git", "add", "--", "Hegel Machine"], cwd=repository, check=True)
    environment = dict(os.environ)
    environment.update(
        {
            "GIT_AUTHOR_DATE": "2026-08-02T00:00:00+00:00",
            "GIT_COMMITTER_DATE": "2026-08-02T00:00:00+00:00",
        }
    )
    subprocess.run(
        ["git", "commit", "-qm", "bridge DAG public fixture"],
        cwd=repository,
        env=environment,
        check=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    return build_formal_static_basis_v1(
        commit,
        repository_root=repository,
        python_binary_path=python_binary,
        rust_binary_path=rust_binary,
    )


def _qualified_synthetic_basis(basis: FormalStaticBasisV1) -> tuple[FormalStaticBasisV1, M3ExecutionBindingContractFieldsV1]:
    python_binding = dict(basis.objects["python_static_replay_implementation_binding"])
    rust_binding = dict(basis.objects["rust_static_replay_implementation_binding"])
    python_root = candidate_content_root("ImplementationBindingV1", python_binding)
    rust_root = candidate_content_root("ImplementationBindingV1", rust_binding)
    roots = dict(basis.roots)
    roots.update(
        {
            "python_implementation_binding_root": python_root,
            "rust_implementation_binding_root": rust_root,
        }
    )
    objects = dict(basis.objects)
    objects.update(
        {
            "python_m3_implementation_binding": MappingProxyType(python_binding),
            "rust_m3_implementation_binding": MappingProxyType(rust_binding),
        }
    )
    candidate_static = dict(basis.m3_candidate_static_fields)
    candidate_static.update(
        {
            "python_implementation_binding_root": python_root,
            "rust_implementation_binding_root": rust_root,
        }
    )
    qualified = replace(
        basis,
        objects=MappingProxyType(objects),
        roots=MappingProxyType(roots),
        m3_candidate_static_fields=MappingProxyType(candidate_static),
    )
    execution = M3ExecutionBindingContractFieldsV1(
        python_implementation_binding_fields=python_binding,
        rust_implementation_binding_fields=rust_binding,
        traversal_contract_fields=basis.objects["traversal_contract"],
        bucket_accounting_contract_fields=basis.objects["bucket_accounting_contract"],
        program_archive_contract_fields=basis.objects["program_archive_contract"],
        output_archive_contract_fields=basis.objects["output_archive_contract"],
        state_machine_contract_fields=basis.objects["state_machine_contract"],
    )
    return qualified, execution


def _public_fixture(basis: FormalStaticBasisV1) -> SyntheticPublicFixture:
    basis, m3_execution = _qualified_synthetic_basis(basis)
    timestamp = 1_750_000_000
    commit_wire = basis.m3_candidate_static_fields["repository_commit_id"]
    public_key = hashlib.sha256(b"synthetic-public-purpose-1-key").digest()
    key_id = hashlib.sha256(public_key).digest()[:16]
    key_fields = {
        "purpose_id": 1,
        "key_id": key_id,
        "public_key_32_bytes": public_key,
        "key_epoch": 0,
        "valid_from_unix_seconds": timestamp - 1,
        "valid_until_unix_seconds_or_null": None,
        "repository_commit_id": commit_wire,
    }
    key_root = candidate_content_root("ActorKeyManifestV1", key_fields)
    actor_trust_fields = {
        "trust_genesis_id_16_bytes": bytes.fromhex("31" * 16),
        "purpose_key_entries": (
            (1, key_root),
            (2, bytes.fromhex("32" * 32)),
            (3, bytes.fromhex("33" * 32)),
            (4, bytes.fromhex("34" * 32)),
        ),
        "purpose_key_policy_root": basis.roots["replacement_policy_root"],
        "created_at_unix_seconds": timestamp,
        "repository_commit_id": commit_wire,
    }
    actor_trust_root = candidate_content_root("ActorTrustGenesisV1", actor_trust_fields)

    parent_fields = {
        "parent_dsl_version_digest": id_digest_v1("hegel-old-dsl-v1.0.0"),
        "parent_freeze_version_digest": id_digest_v1("hegel-freeze-p2b-p3-v1.0.2"),
        "parent_repository_commit_id": git_sha1_commit_id(AUDITED_PARENT_COMMIT_SHA1),
        "audit_bundle_root": bytes.fromhex("41" * 32),
        "absence_reason_bitmask": 0b1111,
        "auditor_key_id": bytes.fromhex("42" * 16),
        "audited_at_unix_seconds": timestamp,
    }
    parent_root = candidate_content_root("ParentManifestAbsenceAttestationV2", parent_fields)

    ledger_fields = {
        "ledger_id": bytes.fromhex("51" * 16),
        "sequence_number": 0,
        "previous_record_root_or_null": None,
        "event_type_id": 1,
        "actor_key_id": key_id,
        "subject_manifest_root": bytes.fromhex("52" * 32),
        "revealed_artifact_root_or_null": None,
        "authorization_root_or_null": None,
        "created_at_unix_seconds": timestamp,
        "repository_commit_id": commit_wire,
    }
    ledger_root = candidate_content_root("HiddenAccessLedgerRecordV1", ledger_fields)

    continuity_fields = dict(basis.preseed_manifest_static_fields["SeedContinuityManifestV1"])
    continuity_fields.update(
        {
            "current_seed_commitment_manifest_root": bytes.fromhex("53" * 32),
            "parent_manifest_absence_attestation_root": parent_root,
            "hidden_access_ledger_genesis_root": ledger_root,
            "custodian_binding_core_root": bytes.fromhex("54" * 32),
            "instantiated_at_unix_seconds": timestamp,
        }
    )
    continuity_root = candidate_content_root("SeedContinuityManifestV1", continuity_fields)
    custodian_fields = {
        "custodian_key_id": key_id,
        "custodian_public_key_32_bytes": public_key,
        "custodian_key_epoch": 0,
        "responsibility_bitmask": 0b011111,
        "split_seed_commitment_manifest_root": bytes.fromhex("53" * 32),
        "hidden_access_ledger_genesis_root": ledger_root,
        "seed_continuity_manifest_root": continuity_root,
        "valid_from_unix_seconds": timestamp,
        "valid_until_unix_seconds_or_null": None,
        "replacement_policy_root": basis.roots["replacement_policy_root"],
        "repository_commit_id": commit_wire,
    }
    custodian_root = candidate_content_root("CustodianBindingManifestV1", custodian_fields)

    sealed = {
        name: hashlib.sha256(f"synthetic-public-{name}".encode("ascii")).digest()
        for name in (
            "outside_discovery_split_root",
            "outside_validation_split_root",
            "outside_sealed_split_root",
            "null_discovery_split_root",
            "null_validation_split_root",
            "null_sealed_split_root",
        )
    }
    split_fields = dict(basis.preseed_manifest_static_fields["SplitBindingManifestV1"])
    split_fields.update(
        {
            "split_seed_commitment_manifest_root": bytes.fromhex("53" * 32),
            "seed_continuity_manifest_root": continuity_root,
            "outside_target_discovery_root": sealed["outside_discovery_split_root"],
            "outside_target_validation_root": sealed["outside_validation_split_root"],
            "outside_target_sealed_root": sealed["outside_sealed_split_root"],
            "null_control_discovery_root": sealed["null_discovery_split_root"],
            "null_control_validation_root": sealed["null_validation_split_root"],
            "null_control_sealed_root": sealed["null_sealed_split_root"],
            "hidden_access_ledger_genesis_root": ledger_root,
            "hidden_access_ledger_head_root": ledger_root,
            "created_at_unix_seconds": timestamp,
        }
    )
    split_root = candidate_content_root("SplitBindingManifestV1", split_fields)

    def role_fields(role_id: int) -> dict[str, object]:
        name = (
            "DslRoleBindingManifestV1/OUTSIDE_TARGET"
            if role_id == 1
            else "DslRoleBindingManifestV1/IN_LANGUAGE_NULL"
        )
        fields = dict(basis.preseed_manifest_static_fields[name])
        fields.update(
            {
                "split_binding_manifest_root": split_root,
                "custodian_binding_manifest_root": custodian_root,
                "seed_continuity_manifest_root": continuity_root,
                "parent_manifest_absence_attestation_root_or_null": parent_root,
                "created_at_unix_seconds": timestamp,
            }
        )
        return fields

    outside_fields = role_fields(1)
    null_fields = role_fields(2)
    outside_root = candidate_content_root("DslRoleBindingManifestV1", outside_fields)
    null_root = candidate_content_root("DslRoleBindingManifestV1", null_fields)
    shrink_fields = dict(basis.preseed_manifest_static_fields["DslShrinkTransitionFormalV1"])
    shrink_fields.update(
        {
            "outside_target_binding_manifest_root": outside_root,
            "null_control_binding_manifest_root": null_root,
            "split_binding_manifest_root": split_root,
            "custodian_binding_manifest_root": custodian_root,
            "seed_continuity_manifest_root": continuity_root,
            "created_at_unix_seconds": timestamp,
        }
    )
    shrink_root = candidate_content_root("DslShrinkTransitionFormalV1", shrink_fields)

    external_fields = {
        "attestations": (
            (1, bytes.fromhex("61" * 32), bytes.fromhex("62" * 32)),
            (4, bytes.fromhex("63" * 32), bytes.fromhex("64" * 32)),
        )
    }
    external_root = candidate_content_root("AttestationBundleV1", external_fields)
    snapshot_fields = {
        "previous_snapshot_root_or_null": bytes.fromhex("71" * 32),
        "registry_tree_root": bytes.fromhex("72" * 32),
        "record_count": 2,
        "added_record_root": bytes.fromhex("73" * 32),
        "repository_commit_id": commit_wire,
    }
    snapshot_root = candidate_content_root("OpaqueIdRegistrySnapshotV1", snapshot_fields)

    candidate = dict(basis.m3_candidate_static_fields)
    candidate.update(
        {
            "run_id": bytes.fromhex("81" * 16),
            "shrink_transition_root": shrink_root,
            "outside_target_binding_manifest_root": outside_root,
            "null_control_binding_manifest_root": null_root,
            "split_binding_manifest_root": split_root,
            "custodian_binding_manifest_root": custodian_root,
            "seed_continuity_manifest_root": continuity_root,
            "custodian_attestation_bundle_root": external_root,
            "parent_absence_attestation_root": parent_root,
            "hidden_access_ledger_genesis_root": ledger_root,
            "hidden_access_ledger_head_root": ledger_root,
            "opaque_id_registry_snapshot_root": snapshot_root,
            "actor_trust_genesis_root": actor_trust_root,
            **sealed,
            "created_at_unix_seconds": timestamp,
        }
    )
    candidate_root = candidate_content_root("M3ExecutionCandidateV1", candidate)
    bridge = {
        "run_id": candidate["run_id"],
        "diagnostic_formal_bridge_root": candidate["diagnostic_formal_bridge_root"],
        "m3_execution_candidate_root": candidate_root,
        "child_dsl_spec_root": candidate["child_dsl_spec_root"],
        "child_freeze_root": candidate["child_freeze_root"],
        "actor_trust_genesis_root": actor_trust_root,
        "opaque_id_registry_snapshot_root": snapshot_root,
    }
    dynamic = {
        "shrink_transition_root": shrink_fields,
        "outside_target_binding_manifest_root": outside_fields,
        "null_control_binding_manifest_root": null_fields,
        "split_binding_manifest_root": split_fields,
        "custodian_binding_manifest_root": custodian_fields,
        "seed_continuity_manifest_root": continuity_fields,
        "hidden_access_ledger_genesis_root": ledger_fields,
        "hidden_access_ledger_head_root": dict(ledger_fields),
    }
    node_inputs = BridgeDagNodeBuildInputsV1(
        basis=basis,
        candidate_fields=MappingProxyType(candidate),
        dynamic_object_fields=MappingProxyType(
            {name: MappingProxyType(fields) for name, fields in dynamic.items()}
        ),
        external_attestation_bundle_fields=MappingProxyType(external_fields),
        parent_attestation_fields=MappingProxyType(parent_fields),
        final_opaque_snapshot_fields=MappingProxyType(snapshot_fields),
        actor_trust_fields=MappingProxyType(actor_trust_fields),
        outside_typed_rows=generate_odd_role_rows_v1(),
        null_typed_rows=generate_sink_role_rows_v1(),
        sealed_split_roots=MappingProxyType(sealed),
        m3_execution_fields=m3_execution,
    )
    return SyntheticPublicFixture(
        node_inputs=node_inputs,
        bridge_fields=MappingProxyType(bridge),
        key_fields=MappingProxyType(key_fields),
    )


@pytest.fixture(scope="module")
def public_fixture(tmp_path_factory: pytest.TempPathFactory) -> SyntheticPublicFixture:
    basis = _copy_and_commit_static_basis(tmp_path_factory.mktemp("bridge-dag-builder"))
    return _public_fixture(basis)


def _code(action, *args, **kwargs) -> str:
    with pytest.raises(BridgeDagNodeBuildError) as captured:
        action(*args, **kwargs)
    return captured.value.code


def test_builder_emits_exact_frozen_37_node_sequence(public_fixture: SyntheticPublicFixture) -> None:
    nodes = build_bridge_dag_nodes_v1(public_fixture.node_inputs)
    assert len(nodes) == 37
    assert tuple(node.role_id for node in nodes) == tuple(range(1, 38))
    for node, spec in zip(nodes, ROLE_SPECS, strict=True):
        if spec.operation_id == 3:
            assert node.preimages == ()
            assert type(node.sealed_root) is bytes and len(node.sealed_root) == 32
        else:
            assert len(node.preimages) == spec.exact_count
            assert node.sealed_root is None


def test_builder_supports_unsigned_purpose1_and_carries_existing_p1_signature_for_2_and_3(
    public_fixture: SyntheticPublicFixture,
) -> None:
    purpose1 = build_bridge_dag_replay_package_from_inputs_v1(
        BridgeDagPackageBuildInputsV1(
            node_inputs=public_fixture.node_inputs,
            purpose_id=1,
            bridge_statement_fields=public_fixture.bridge_fields,
            purpose1_actor_key_manifest_fields=public_fixture.key_fields,
            purpose1_bridge_signature=None,
        )
    )
    assert replay_bridge_dag_package_v1(purpose1).purpose_id == 1
    assert canonical_cbor_decode(purpose1)[9] is None

    synthetic_existing_signature = bytes.fromhex("91" * 64)
    for purpose_id in (2, 3):
        package = build_bridge_dag_replay_package_from_inputs_v1(
            BridgeDagPackageBuildInputsV1(
                node_inputs=public_fixture.node_inputs,
                purpose_id=purpose_id,
                bridge_statement_fields=public_fixture.bridge_fields,
                purpose1_actor_key_manifest_fields=public_fixture.key_fields,
                purpose1_bridge_signature=synthetic_existing_signature,
            )
        )
        decoded = canonical_cbor_decode(package)
        assert decoded[4] == purpose_id
        assert decoded[9] == synthetic_existing_signature


def test_builder_rejects_dynamic_role_omission(public_fixture: SyntheticPublicFixture) -> None:
    dynamic = dict(public_fixture.node_inputs.dynamic_object_fields)
    dynamic.pop("seed_continuity_manifest_root")
    attacked = replace(public_fixture.node_inputs, dynamic_object_fields=dynamic)
    assert _code(build_bridge_dag_nodes_v1, attacked) == FAIL_FIELD_SET


def test_builder_rejects_dynamic_preimage_substitution(public_fixture: SyntheticPublicFixture) -> None:
    dynamic = dict(public_fixture.node_inputs.dynamic_object_fields)
    ledger = dict(dynamic["hidden_access_ledger_genesis_root"])
    ledger["subject_manifest_root"] = bytes.fromhex("a1" * 32)
    dynamic["hidden_access_ledger_genesis_root"] = ledger
    attacked = replace(public_fixture.node_inputs, dynamic_object_fields=dynamic)
    assert _code(build_bridge_dag_nodes_v1, attacked) == FAIL_CANDIDATE_BINDING


def test_builder_rejects_outside_null_cross_role_splice(public_fixture: SyntheticPublicFixture) -> None:
    dynamic = dict(public_fixture.node_inputs.dynamic_object_fields)
    dynamic["outside_target_binding_manifest_root"], dynamic["null_control_binding_manifest_root"] = (
        dynamic["null_control_binding_manifest_root"],
        dynamic["outside_target_binding_manifest_root"],
    )
    candidate = dict(public_fixture.node_inputs.candidate_fields)
    candidate["outside_target_binding_manifest_root"], candidate["null_control_binding_manifest_root"] = (
        candidate["null_control_binding_manifest_root"],
        candidate["outside_target_binding_manifest_root"],
    )
    attacked = replace(
        public_fixture.node_inputs,
        dynamic_object_fields=dynamic,
        candidate_fields=candidate,
    )
    assert _code(build_bridge_dag_nodes_v1, attacked) == FAIL_CROSS_ROLE


def test_builder_rejects_typed_role_substitution(public_fixture: SyntheticPublicFixture) -> None:
    attacked = replace(
        public_fixture.node_inputs,
        outside_typed_rows=public_fixture.node_inputs.null_typed_rows,
    )
    assert _code(build_bridge_dag_nodes_v1, attacked) == FAIL_CROSS_ROLE


def test_builder_rejects_joint_candidate_and_contract_substitution_against_basis(
    public_fixture: SyntheticPublicFixture,
) -> None:
    old = public_fixture.node_inputs.m3_execution_fields
    traversal = dict(old.traversal_contract_fields)
    traversal["maximum_canonical_programs"] = 49_999
    m3 = replace(old, traversal_contract_fields=traversal)
    candidate = dict(public_fixture.node_inputs.candidate_fields)
    candidate["traversal_contract_root"] = candidate_content_root("TraversalContractV1", traversal)
    attacked = replace(
        public_fixture.node_inputs,
        m3_execution_fields=m3,
        candidate_fields=candidate,
    )
    assert _code(build_bridge_dag_nodes_v1, attacked) == FAIL_BASIS_BINDING


@pytest.mark.parametrize(
    ("purpose_id", "signature"),
    ((1, bytes(64)), (2, None), (3, b"short")),
)
def test_builder_enforces_signature_phase(
    public_fixture: SyntheticPublicFixture,
    purpose_id: int,
    signature: bytes | None,
) -> None:
    request = BridgeDagPackageBuildInputsV1(
        node_inputs=public_fixture.node_inputs,
        purpose_id=purpose_id,
        bridge_statement_fields=public_fixture.bridge_fields,
        purpose1_actor_key_manifest_fields=public_fixture.key_fields,
        purpose1_bridge_signature=signature,
    )
    assert _code(build_bridge_dag_replay_package_from_inputs_v1, request) == FAIL_SIGNATURE_PHASE


def test_package_preflight_rejects_purpose1_key_not_bound_by_actor_trust(
    public_fixture: SyntheticPublicFixture,
) -> None:
    key = dict(public_fixture.key_fields)
    public = hashlib.sha256(b"substituted-public-key").digest()
    key["public_key_32_bytes"] = public
    key["key_id"] = hashlib.sha256(public).digest()[:16]
    request = BridgeDagPackageBuildInputsV1(
        node_inputs=public_fixture.node_inputs,
        purpose_id=1,
        bridge_statement_fields=public_fixture.bridge_fields,
        purpose1_actor_key_manifest_fields=key,
        purpose1_bridge_signature=None,
    )
    assert _code(build_bridge_dag_replay_package_from_inputs_v1, request) == FAIL_PACKAGE_PREFLIGHT
