from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import pytest

from hegel_machine import phase2b_actual_unsealed_960_readiness_dag_v1 as readiness
from hegel_machine import phase2b_c1_owner_actual_unsealed_960_authority_v1 as owner
from hegel_machine.phase2b_freeze_v1 import frozen_phase2b_exact_freeze
from hegel_machine.phase2b_protocol import frozen_phase2b_protocol


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/phase2b_actual_unsealed_960_readiness_dag_v1.json"
SOURCE = ROOT / "src/hegel_machine/phase2b_actual_unsealed_960_readiness_dag_v1.py"
DOC = ROOT / "docs/Hegel_Machine_Phase2B_Actual_Unsealed_960_Readiness_DAG_Erratum_v1.md"
TEST = Path(__file__)
OWNER_CONFIG = ROOT / "config/phase2b_c1_owner_actual_unsealed_960_authorization_v1.json"
OWNER_SOURCE = ROOT / "src/hegel_machine/phase2b_c1_owner_actual_unsealed_960_authority_v1.py"
OWNER_DOC = ROOT / "docs/Hegel_Machine_Phase2B_C1_Owner_Authorization_From_Zero_Actual_Unsealed_960_v1.md"
FREEZE_SOURCE = ROOT / "src/hegel_machine/phase2b_freeze_v1.py"
PROTOCOL_SOURCE = ROOT / "src/hegel_machine/phase2b_protocol.py"

EXPECTED_BODY_SHA256 = "424e07f89f5b287201a6c96c54c50a4d0e7ad4a72ee1511b35d0305359d93ffb"
EXPECTED_DAG_ID = (
    "phase2b_actual_unsealed_960_readiness_dag_v1_"
    "775b86cb499db3a991686634335ec444f7e44c7caea92b1c2b948036b585ea84"
)
EXPECTED_NODE_IDS = (
    "R01_AUTHORIZATION_VERIFIED",
    "R02_EXECUTION_BRANCH_BINDINGS_REPLAYED",
    "R03_FORMAL_TRUSTED_WIRE_IMPLEMENTATION_QUALIFIED",
    "R04_FORMAL_COVERT_AUDITOR_IMPLEMENTATION_QUALIFIED",
    "R05_SYNTHETIC_DEV_WIRE_COVERT_QUALIFICATION_VERIFIED",
    "R06_GENERATOR_IMPLEMENTATION_QUALIFIED",
    "R07_BASELINE_REGISTRATIONS_FROZEN",
    "R08_STATISTICAL_VALIDATION_PASSED",
    "R09_SIX_ROLE_LIVE_DOCKER_QUALIFIED",
    "R10_DURABLE_CUSTODY_AND_LEDGER_VERIFIED",
    "R11_EXECUTION_FREEZE_COMMITMENTS_BOUND",
    "R12_INDEPENDENT_LAUNCH_OVERLAY_VERIFIED",
    "R13_FRESH_IDS_ALLOCATED_BY_VERIFIED_OVERLAY",
    "R14_ACTUAL_UNSEALED_960_COHORT_GENERATED",
    "R15_ACTUAL_CORPUS_COVERT_AUDIT_PASSED",
    "R16_INFERENCE_ADMISSION_UNLOCKED",
)


def _payload() -> bytes:
    return CONFIG.read_bytes()


def _decoded() -> dict[str, object]:
    value = json.loads(_payload())
    assert type(value) is dict
    return value


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _attacker_recompute_id(value: dict[str, object]) -> None:
    digest = hashlib.sha256(
        b"HEGEL/PHASE2B/ACTUAL_UNSEALED_960/READINESS_DAG/BODY/V1\x00"
        + _canonical(value["readiness_dag_body"])
    ).hexdigest()
    value["readiness_dag_id"] = readiness.PHASE2B_ACTUAL_UNSEALED_960_READINESS_DAG_V1_PREFIX + digest


def test_independent_canonical_kat_and_content_id() -> None:
    value = _decoded()
    body = value["readiness_dag_body"]
    independent_body = _canonical(body)
    assert hashlib.sha256(independent_body).hexdigest() == EXPECTED_BODY_SHA256
    independent_id = readiness.PHASE2B_ACTUAL_UNSEALED_960_READINESS_DAG_V1_PREFIX + hashlib.sha256(
        b"HEGEL/PHASE2B/ACTUAL_UNSEALED_960/READINESS_DAG/BODY/V1\x00"
        + independent_body
    ).hexdigest()
    assert independent_id == EXPECTED_DAG_ID
    assert value["readiness_dag_id"] == EXPECTED_DAG_ID
    assert readiness.phase2b_actual_unsealed_960_readiness_dag_id_v1(body) == (
        EXPECTED_DAG_ID
    )
    assert _payload() == _canonical(value)


def test_loader_returns_private_deeply_immutable_ordering_only_result() -> None:
    result = readiness.load_phase2b_actual_unsealed_960_readiness_dag_v1(_payload())
    assert result.disposition is (
        readiness.Phase2BActualUnsealed960ReadinessDispositionV1.
        READINESS_ORDERING_VERIFIED_NOT_GATE_EVIDENCE
    )
    assert result.reason is (
        readiness.Phase2BActualUnsealed960ReadinessReasonV1.
        EXACT_READINESS_ORDERING_ERRATUM_VERIFIED
    )
    assert result.readiness_dag_id == EXPECTED_DAG_ID
    assert result.readiness_ordering_verified is True
    assert result.ordering_erratum_verified is True
    assert result.node_count == 16
    assert result.satisfied_node_count == 0
    assert result.formal_trusted_wire_implementation_qualified is False
    assert result.formal_covert_channel_audit_passed is False
    assert result.synthetic_dev_qualification_executed is False
    assert result.formal_corpus_audit_executed is False
    assert result.actual_960_capacity_qualified is False
    assert result.generator_exists is False
    assert result.baseline_registrations_exist is False
    assert result.prerequisites_verified is False
    assert result.campaign_id is None
    assert result.attempt_id is None
    assert result.campaign_id_allocation_authorized is False
    assert result.actual_campaign_eligible is False
    assert result.execution_authorized is False
    assert result.formal_holdout_generation_authorized is False
    assert result.actual_holdout_generated is False
    assert result.inference_admission_unlocked is False
    assert result.inference_performed is False
    assert result.shadow_only is True
    assert result.docker_invoked is False
    assert result.verifier_filesystem_accessed is False
    assert result.network_accessed is False
    assert result.keys_generated is False
    assert result.scoring_performed is False
    assert result.actual_execution_claimed is False
    assert isinstance(result.policy, MappingProxyType)
    with pytest.raises(TypeError):
        result.policy["status"] = "ATTACKED"
    with pytest.raises(TypeError):
        result.policy["readiness_dag"]["nodes_in_canonical_topological_order"][0][
            "initial_satisfied"
        ] = True


def test_result_constructor_is_private() -> None:
    with pytest.raises(TypeError, match="privately issued"):
        readiness.Phase2BActualUnsealed960ReadinessVerificationV1()


def test_erratum_preserves_old_artifact_but_places_wire_before_statistical_work() -> None:
    old_body = json.loads(OWNER_CONFIG.read_bytes())["authorization_body"]
    old_order = old_body["future_execution_prerequisites_in_order"]
    assert old_order.index("VALIDATION_VERSION_PASSED") < old_order.index(
        "WIRE_BUILDER_AND_COVERT_CHANNEL_AUDIT_PASSED"
    )

    body = _decoded()["readiness_dag_body"]
    erratum = body["ordering_erratum"]
    assert erratum == {
        "authorization_v1_historical_artifact_immutable": True,
        "conditions_relaxed": False,
        "correction_kind": "STRICTER_ADDITIVE_LAUNCH_READINESS_ORDERING_COMPANION",
        "correction_reason": (
            "EXACT_FREEZE_PRIORITY_3_PRECEDES_PHASE2B_STATISTICAL_COMPLETION_PRIORITY_5"
        ),
        "old_list_order_must_not_be_used_for_launch_readiness": True,
        "old_list_problem": (
            "GENERATOR_BASELINE_AND_VALIDATION_PRECEDED_WIRE_COVERT_QUALIFICATION"
        ),
        "prerequisite_evidence_created": False,
        "scope_changed": False,
        "strict_ordering_constraints_in_order": [
            "FORMAL_WIRE_COVERT_QUALIFICATION_BEFORE_GENERATOR_BASELINES_AND_STATISTICAL_VALIDATION",
            "LAUNCH_OVERLAY_AUTHORIZATION_AND_VERIFICATION_BEFORE_FRESH_ID_ALLOCATION",
            "ACTUAL_CORPUS_COVERT_AUDIT_AFTER_GENERATION_AND_BEFORE_INFERENCE_ADMISSION",
        ],
        "supersedes_owner_authorization": False,
    }
    nodes = body["readiness_dag"]["nodes_in_canonical_topological_order"]
    node_ids = tuple(node["node_id"] for node in nodes)
    assert node_ids == EXPECTED_NODE_IDS
    wire_index = node_ids.index("R03_FORMAL_TRUSTED_WIRE_IMPLEMENTATION_QUALIFIED")
    covert_index = node_ids.index("R04_FORMAL_COVERT_AUDITOR_IMPLEMENTATION_QUALIFIED")
    integrated_index = node_ids.index(
        "R05_SYNTHETIC_DEV_WIRE_COVERT_QUALIFICATION_VERIFIED"
    )
    assert wire_index < covert_index < integrated_index
    for later in (
        "R06_GENERATOR_IMPLEMENTATION_QUALIFIED",
        "R07_BASELINE_REGISTRATIONS_FROZEN",
        "R08_STATISTICAL_VALIDATION_PASSED",
    ):
        assert integrated_index < node_ids.index(later)
    assert node_ids.index("R14_ACTUAL_UNSEALED_960_COHORT_GENERATED") < node_ids.index(
        "R15_ACTUAL_CORPUS_COVERT_AUDIT_PASSED"
    ) < node_ids.index("R16_INFERENCE_ADMISSION_UNLOCKED")
    assert node_ids.index("R12_INDEPENDENT_LAUNCH_OVERLAY_VERIFIED") < node_ids.index(
        "R13_FRESH_IDS_ALLOCATED_BY_VERIFIED_OVERLAY"
    )


def test_dag_dependencies_are_exact_topological_and_fail_closed() -> None:
    nodes = _decoded()["readiness_dag_body"]["readiness_dag"][
        "nodes_in_canonical_topological_order"
    ]
    expected_dependencies = (
        (),
        (EXPECTED_NODE_IDS[0],),
        (EXPECTED_NODE_IDS[1],),
        (EXPECTED_NODE_IDS[2],),
        (EXPECTED_NODE_IDS[3],),
        (EXPECTED_NODE_IDS[4],),
        (EXPECTED_NODE_IDS[4],),
        (EXPECTED_NODE_IDS[5], EXPECTED_NODE_IDS[6]),
        (EXPECTED_NODE_IDS[7],),
        (EXPECTED_NODE_IDS[8],),
        (EXPECTED_NODE_IDS[9],),
        (EXPECTED_NODE_IDS[10],),
        (EXPECTED_NODE_IDS[11],),
        (EXPECTED_NODE_IDS[12],),
        (EXPECTED_NODE_IDS[13],),
        (EXPECTED_NODE_IDS[14],),
    )
    assert tuple(tuple(node["dependencies"]) for node in nodes) == expected_dependencies
    assert all(node["initial_satisfied"] is False for node in nodes)
    dag = _decoded()["readiness_dag_body"]["readiness_dag"]
    assert type(dag["node_count"]) is int and dag["node_count"] == 16
    assert type(dag["satisfied_node_count"]) is int and dag["satisfied_node_count"] == 0
    assert dag["all_nodes_satisfied"] is False


def test_exact_freeze_protocol_and_owner_authorization_bindings() -> None:
    bindings = _decoded()["readiness_dag_body"]["exact_contract_bindings"]
    exact_freeze = frozen_phase2b_exact_freeze()
    protocol = frozen_phase2b_protocol()
    assert bindings == {
        "exact_freeze_id": exact_freeze.freeze_id,
        "exact_freeze_version": exact_freeze.freeze_version,
        "owner_authorization_id": owner.PHASE2B_C1_OWNER_ACTUAL_UNSEALED_960_AUTHORIZATION_V1_ID,
        "protocol_id": protocol.protocol_id,
    }
    assert bindings["exact_freeze_version"] == protocol.exact_freeze_version
    assert exact_freeze.formal_holdout_generation_authorized is False
    assert exact_freeze.shadow_only is True
    assert protocol.shadow_only is True
    assert protocol.active_promotion_enabled is False


def test_wire_covert_qualification_contract_is_exactly_synthetic_and_atomic() -> None:
    contract = _decoded()["readiness_dag_body"]["wire_covert_qualification_contract"]
    assert contract["canonical_json"] == "RFC-8785"
    assert type(contract["envelope_bytes"]) is int
    assert contract["envelope_bytes"] == 65_536
    assert contract["fixed_envelope_capacity_audit_required"] is True
    assert contract["independent_secret_keys"] == ["K_shuffle", "K_id", "K_padding"]
    assert contract["keys_separate_from"] == ["latent_generator_seed", "answer_seed"]
    assert contract["id_assignment_after_global_shuffle"] is True
    assert contract["id_algorithm"] == "HMAC_SHA256_TRUNCATE_128_SET_UUIDV4_BITS"
    assert contract["namespace_aware_audit_required"] is True
    assert contract["renaming_namespaces"] == [
        "bundle_id",
        "observation_id",
        "entity_id",
        "role_candidate_id",
        "quantity_id",
        "context_id",
        "task_id",
        "scale_id",
        "aggregate_map_id",
        "transform_id",
    ]
    assert contract["global_consistent_renamings"] == 32
    assert contract["global_case_order_permutations"] == 32
    assert contract["within_case_observation_order_permutations"] == 16
    assert contract["standard_error_enabled_by_default"] is False
    assert contract["standard_error_status"] == "STANDARD_ERROR_UNSUPPORTED"
    assert contract["standard_error_encounter_policy"] == (
        "ABORT_WHOLE_BATCH_NO_PARTIAL_OUTPUT"
    )
    assert contract["synthetic_dev_input_only"] is True
    assert contract["synthetic_dev_only"] is True
    assert contract["actual_corpus_input_allowed"] is False
    assert contract["actual_corpus_qualification_claimed"] is False
    assert contract["atomic_failure_requirements"] == [
        "ANY_SCHEMA_OR_FIELD_CLOSURE_FAILURE_ABORTS_THE_WHOLE_BATCH",
        "ANY_ENVELOPE_CAPACITY_OVERFLOW_ABORTS_THE_WHOLE_BATCH",
        "ANY_NAMESPACE_RENAMING_INVARIANCE_FAILURE_ABORTS_QUALIFICATION",
        "ANY_COVERT_GATE_FAILURE_ABORTS_QUALIFICATION",
        "ANY_SOURCE_OR_ANSWER_BINDING_FAILURE_ABORTS_THE_WHOLE_BATCH",
        "ANY_KEY_INDEPENDENCE_EVIDENCE_FAILURE_ABORTS_QUALIFICATION",
        "ANY_RFC8785_NONCANONICAL_DUPLICATE_UNICODE_OR_NUMBER_VECTOR_FAILURE_ABORTS_QUALIFICATION",
        "ANY_UNCLASSIFIED_OR_AMBIGUOUS_NAMESPACE_ABORTS_QUALIFICATION",
        "ANY_STATISTICAL_SUFFICIENCY_FAILURE_ABORTS_QUALIFICATION",
    ]
    assert contract["implementation_qualification_evidence_fields_in_order"] == [
        "source_commit_sha",
        "source_archive_root",
        "dependency_lock_sha256",
        "runtime_image_digest",
        "synthetic_fixture_root",
        "qualification_receipt_id",
    ]
    assert contract["existing_mechanics_nonqualifications"] == [
        "ACCEPTED_JCS_SUBSET_IS_NOT_COMPLETE_RFC8785_QUALIFICATION",
        "SYNTHETIC_COVERT_STATISTICS_CAP_64_IS_NOT_ACTUAL_960_QUALIFICATION",
        "NON_AUTHORITATIVE_MECHANICS_DO_NOT_SATISFY_R03_R04_OR_R05",
    ]
    assert contract["channel_only_gate"] == {
        "family_wise_alpha": "0.01_EXACT_DECIMAL",
        "label_permutations": 10_000,
        "maximum_balanced_accuracy_advantage": "0.05_EXACT_DECIMAL",
        "maximum_normalized_mutual_information": "0.02_EXACT_DECIMAL",
        "minimum_adjusted_p": "0.01_EXACT_DECIMAL",
        "multiple_testing": "HOLM_BONFERRONI",
    }


def test_safety_boundary_contains_no_execution_or_qualification_evidence() -> None:
    safety = _decoded()["readiness_dag_body"]["safety_boundary"]
    assert safety["campaign_id"] is None
    assert safety["attempt_id"] is None
    assert safety["shadow_only"] is True
    false_fields = set(safety) - {"campaign_id", "attempt_id", "shadow_only"}
    assert false_fields
    assert all(type(safety[field]) is bool and safety[field] is False for field in false_fields)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["readiness_dag_body"].update({"unknown": False}),
        lambda value: value["readiness_dag_body"]["exact_contract_bindings"].update(
            {"owner_authorization_id": "wrong"}
        ),
        lambda value: value["readiness_dag_body"]["ordering_erratum"].update(
            {"conditions_relaxed": True}
        ),
        lambda value: value["readiness_dag_body"]["readiness_dag"].update(
            {"node_count": True}
        ),
        lambda value: value["readiness_dag_body"]["readiness_dag"][
            "nodes_in_canonical_topological_order"
        ][2].update({"dependencies": []}),
        lambda value: value["readiness_dag_body"]["readiness_dag"][
            "nodes_in_canonical_topological_order"
        ][4].update({"initial_satisfied": True}),
        lambda value: value["readiness_dag_body"]["readiness_dag"][
            "nodes_in_canonical_topological_order"
        ].reverse(),
        lambda value: value["readiness_dag_body"]["safety_boundary"].update(
            {"execution_authorized": True}
        ),
        lambda value: value["readiness_dag_body"][
            "wire_covert_qualification_contract"
        ].update({"actual_corpus_input_allowed": True}),
        lambda value: value["readiness_dag_body"][
            "wire_covert_qualification_contract"
        ]["renaming_namespaces"].pop(),
    ],
    ids=(
        "unknown",
        "owner-binding",
        "relaxation",
        "bool-as-int",
        "dependency-bypass",
        "satisfied-node",
        "node-order",
        "execution-claim",
        "actual-corpus",
        "namespace-omission",
    ),
)
def test_mutation_rejects_atomically_even_with_attacker_recomputed_id(mutation) -> None:
    value = copy.deepcopy(_decoded())
    mutation(value)
    _attacker_recompute_id(value)
    result = readiness.verify_phase2b_actual_unsealed_960_readiness_dag_v1(value)
    assert result.disposition is (
        readiness.Phase2BActualUnsealed960ReadinessDispositionV1.REJECTED
    )
    assert result.readiness_dag_id is None
    assert result.policy is None
    assert result.readiness_ordering_verified is False
    assert result.ordering_erratum_verified is False
    assert result.node_count == 0
    assert result.satisfied_node_count == 0
    assert result.formal_trusted_wire_implementation_qualified is False
    assert result.formal_covert_channel_audit_passed is False
    assert result.prerequisites_verified is False
    assert result.campaign_id is None
    assert result.attempt_id is None
    assert result.execution_authorized is False
    assert result.formal_holdout_generation_authorized is False
    assert result.shadow_only is True


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"[]",
        b'{"readiness_dag_body":null,"readiness_dag_body":null,"readiness_dag_id":null}',
        _payload().replace(b'"node_count":16', b'"node_count":16.0'),
        _payload().replace(b'"node_count":16', b'"node_count":NaN'),
        b"{ }",
        _payload() + b"\n",
        b"\xff",
    ],
    ids=(
        "empty",
        "array",
        "duplicate-key",
        "float",
        "nan",
        "noncanonical-whitespace",
        "newline",
        "non-ascii",
    ),
)
def test_byte_loader_rejects_invalid_or_noncanonical_inputs(payload: bytes) -> None:
    result = readiness.load_phase2b_actual_unsealed_960_readiness_dag_v1(payload)
    assert result.disposition is (
        readiness.Phase2BActualUnsealed960ReadinessDispositionV1.REJECTED
    )
    assert result.readiness_ordering_verified is False
    assert result.readiness_dag_id is None
    assert result.policy is None
    assert result.campaign_id is None
    assert result.attempt_id is None
    assert result.actual_campaign_eligible is False
    assert result.execution_authorized is False
    assert result.shadow_only is True


def test_old_authorization_freeze_protocol_files_are_byte_unchanged() -> None:
    expected = {
        OWNER_CONFIG: "26ff84c3fe239158b1e8db24b5634ccfe8b3f3995d54bdc1f728c49a70f8fca1",
        OWNER_SOURCE: "4c96d50706e520f1c67b2d2c31fce9ed605feacbf4c1177ddf9a7a2a9c1d9a73",
        OWNER_DOC: "ba77f1c43796f09ecb850b00488a4e8a0a3a259f64f75d2cefa50c66298e3ce7",
        FREEZE_SOURCE: "39a289ef0c000c1c62eb3a7a5f97ddfae3e51eb7904757642b0c2d167067f543",
        PROTOCOL_SOURCE: "aeccff3558e82ebf261cbeb1c389e968ff22c8ae229b9ce201808accf76074ef",
    }
    assert {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in expected
    } == expected


def test_module_ast_has_no_side_effect_capability_or_forbidden_call_surface() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    forbidden_import_roots = {
        "asyncio",
        "docker",
        "http",
        "os",
        "pathlib",
        "requests",
        "secrets",
        "shutil",
        "socket",
        "subprocess",
        "tempfile",
        "urllib",
        "uuid",
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
            assert all(
                alias.name.split(".")[0] not in forbidden_import_roots
                for alias in node.names
            )
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            assert node.module.split(".")[0] not in forbidden_import_roots
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in forbidden_calls


def test_public_surface_is_exact_and_contains_no_runtime_operation() -> None:
    assert set(readiness.__all__) == {
        "EXACT_FREEZE_ID",
        "EXACT_FREEZE_VERSION",
        "FAIL_READINESS_CANONICAL",
        "FAIL_READINESS_CLAIM_BOUNDARY",
        "FAIL_READINESS_CONTENT_ID",
        "FAIL_READINESS_INPUT",
        "FAIL_READINESS_JSON",
        "FAIL_READINESS_SCHEMA",
        "OWNER_AUTHORIZATION_ID_V1",
        "PHASE2B_ACTUAL_UNSEALED_960_READINESS_DAG_V1_CLAIM_LEVEL",
        "PHASE2B_ACTUAL_UNSEALED_960_READINESS_DAG_V1_DOCUMENT_ID",
        "PHASE2B_ACTUAL_UNSEALED_960_READINESS_DAG_V1_DOMAIN",
        "PHASE2B_ACTUAL_UNSEALED_960_READINESS_DAG_V1_ID",
        "PHASE2B_ACTUAL_UNSEALED_960_READINESS_DAG_V1_PREFIX",
        "PHASE2B_ACTUAL_UNSEALED_960_READINESS_DAG_V1_SCHEMA",
        "PHASE2B_ACTUAL_UNSEALED_960_READINESS_DAG_V1_STATUS",
        "PROTOCOL_ID",
        "Phase2BActualUnsealed960ReadinessDagErrorV1",
        "Phase2BActualUnsealed960ReadinessDispositionV1",
        "Phase2BActualUnsealed960ReadinessReasonV1",
        "Phase2BActualUnsealed960ReadinessVerificationV1",
        "canonical_phase2b_actual_unsealed_960_readiness_dag_body_bytes_v1",
        "load_phase2b_actual_unsealed_960_readiness_dag_v1",
        "phase2b_actual_unsealed_960_readiness_dag_id_v1",
        "verify_phase2b_actual_unsealed_960_readiness_dag_v1",
    }
    forbidden_public_words = (
        "allocate",
        "docker",
        "execute",
        "generate",
        "launch",
        "network",
        "score",
    )
    assert not any(
        word in name.lower()
        for name in readiness.__all__
        for word in forbidden_public_words
    )


def test_unit_declares_only_the_four_new_files() -> None:
    assert {CONFIG, SOURCE, DOC, TEST} == {
        ROOT / "config/phase2b_actual_unsealed_960_readiness_dag_v1.json",
        ROOT / "src/hegel_machine/phase2b_actual_unsealed_960_readiness_dag_v1.py",
        ROOT / "docs/Hegel_Machine_Phase2B_Actual_Unsealed_960_Readiness_DAG_Erratum_v1.md",
        ROOT / "tests/test_phase2b_actual_unsealed_960_readiness_dag_v1.py",
    }
    assert all(path.is_file() for path in (CONFIG, SOURCE, DOC, TEST))
