from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

import hegel_machine.phase3_m3_shrink6_dual_diagnostic_v1 as dual
from hegel_machine.phase3_m3_bounded_enumerator_v1 import (
    CHUNK_BLOB_DOMAIN,
    program_mdl_length_q32,
)
from hegel_machine.phase3_m3_record_wire_v1 import build_m3_record_object_v1
from hegel_machine.phase3_m3_shrink6_diagnostic_profile_v1 import (
    BINDING_PROFILE_ID,
    CLAIM_LEVEL,
    PROFILE_ID,
    STRICT_QUALIFICATION_ARTIFACT_PATH,
    STRICT_QUALIFICATION_ARTIFACT_SHA256,
    STRICT_QUALIFICATION_DIAGNOSTIC_REPORT_HASH,
    STRICT_QUALIFICATION_EVIDENCE_COMMIT,
    STRICT_QUALIFICATION_SOURCE_COMMIT,
    STRICT_QUALIFICATION_STATUS,
    diagnostic_root_hex_v1,
)
from hegel_machine.strict_ast_shrink6_v1 import canonicalize_shrink6_source_ast
from hegel_machine.strict_cbor_v1 import (
    canonical_cbor_decode,
    canonical_cbor_encode,
    rfc6962_root,
)


def _framed(records: tuple[bytes, ...]) -> bytes:
    return b"".join(len(record).to_bytes(4, "big") + record for record in records)


def _program_record(index: int, context_id: int) -> tuple[object, ...]:
    ast = canonicalize_shrink6_source_ast(["context_flag", context_id])
    roots = tuple(bytes.fromhex(diagnostic_root_hex_v1()[name]) for name in dual._ROOT_NAMES[:3])
    return build_m3_record_object_v1(
        "CanonicalProgramRecordV2",
        {
            "program_index": index,
            "canonical_ast_cbor_bytes": ast.cbor_bytes,
            "canonical_ast_hash": ast.digest,
            "output_sort_id": 1,
            "ast_depth": ast.metrics.depth,
            "ast_node_count": ast.metrics.node_count,
            "distinct_bit_slot_count": len(ast.metrics.distinct_bit_slots),
            "program_mdl_length_q32": program_mdl_length_q32(ast),
            "child_dsl_spec_root": roots[0],
            "operator_semantics_root": roots[1],
            "identifier_registry_root": roots[2],
        },
    )


def _fixture_material(
    *, witness_context_id: int | None = 2
) -> tuple[dict[str, bytes], dict[str, object]]:
    complete = witness_context_id is None
    program_objects = (_program_record(0, 0), _program_record(1, 1))
    program_frames = tuple(canonical_cbor_encode(item) for item in program_objects)
    program_stream = _framed(program_frames)
    blob_hash = sha256(CHUNK_BLOB_DOMAIN + b"\x00" + program_stream).digest()
    chunk_object = build_m3_record_object_v1(
        "ProgramChunkManifestV2",
        {
            "chunk_index": 0,
            "first_program_index": 0,
            "last_program_index": 1,
            "record_count": 2,
            "canonical_program_record_subtree_root": rfc6962_root(
                list(program_objects)
            ),
            "compressed_program_blob_hash": blob_hash,
            "uncompressed_program_byte_length": len(program_stream),
        },
    )
    chunk_frames = (canonical_cbor_encode(chunk_object),)

    bucket_objects: list[tuple[object, ...]] = []
    for bucket_index, (sort_id, depth, nodes) in enumerate(
        (sort_id, depth, nodes)
        for sort_id in range(1, 6)
        for depth in range(4)
        for nodes in range(1, 7)
    ):
        first = bucket_index == 0
        bucket_objects.append(
            build_m3_record_object_v1(
                "BucketAccountingRecordV1",
                {
                    "bucket_index": bucket_index,
                    "output_sort_id": sort_id,
                    "ast_depth": depth,
                    "ast_node_count": nodes,
                    "raw_operator_applications": (2 if complete else 6) if first else 0,
                    "accepted_canonical_programs": 2 if first else 0,
                    "syntactic_duplicates": 0,
                    "type_rejections": 0,
                    "structural_limit_rejections": 0,
                    "rewrite_collapses": 0,
                    "first_program_index_or_null": 0 if first else None,
                    "last_program_index_or_null": 1 if first else None,
                },
            )
        )
    bucket_frames = tuple(canonical_cbor_encode(item) for item in bucket_objects)
    witness = (
        None
        if witness_context_id is None
        else canonicalize_shrink6_source_ast(["context_flag", witness_context_id])
    )
    roots = diagnostic_root_hex_v1()
    report: dict[str, object] = {
        "profile_id": PROFILE_ID,
        "binding_profile_id": BINDING_PROFILE_ID,
        "claim_level": CLAIM_LEVEL,
        "diagnostic_only": True,
        "authoritative_claim_allowed": False,
        "execution_state": "NOT_RUN",
        "formal_roots_generated": False,
        "formal_roots": None,
        "strict_qualification_source_commit": STRICT_QUALIFICATION_SOURCE_COMMIT,
        "strict_qualification_evidence_commit": (
            STRICT_QUALIFICATION_EVIDENCE_COMMIT
        ),
        "strict_qualification_artifact_path": STRICT_QUALIFICATION_ARTIFACT_PATH,
        "strict_qualification_artifact_sha256": (
            STRICT_QUALIFICATION_ARTIFACT_SHA256
        ),
        "strict_qualification_diagnostic_report_hash": (
            STRICT_QUALIFICATION_DIAGNOSTIC_REPORT_HASH
        ),
        "strict_qualification_status": STRICT_QUALIFICATION_STATUS,
        "parent_dsl_version": "hegel-old-dsl-v1.5.0",
        "parent_freeze_version": "hegel-freeze-p2b-p3-v1.5.0",
        "dsl_version": "hegel-old-dsl-v1.6.0",
        "freeze_version": "hegel-freeze-p2b-p3-v1.6.0",
        "human_amendment_id": "hegel-freeze-p2b-p3-v1.6.0-shrink-step6",
        "shrink_step_id": (
            "SHRINK_STEP_6_REDUCE_MAX_TOTAL_AST_DEPTH_4_TO_3"
        ),
        "canonicalizer_profile": "hegel-canonical-ast-v1",
        "mdl_code_table_id": "hegel-mdl-prefix-v1.0.0",
        "closure_status": "COMPLETE" if complete else "DSL_TOO_LARGE",
        "closure_status_id": 1 if complete else 2,
        "raw_operator_application_count": 2 if complete else 6,
        "canonical_program_count": 2,
        "closure_cardinality_or_null": 2 if complete else None,
        "frontier_exhausted": complete,
        "all_type_buckets_closed": complete,
        "raw_expansion_limit_hit": False,
        "wall_clock_abort_hit": False,
        "program_record_count": 2,
        "chunk_manifest_count": 1,
        "bucket_record_count": 120,
        "records_per_chunk": 2,
        "maximum_canonical_programs": 2,
        "maximum_raw_operator_applications": 10,
        "maximum_ast_depth": 3,
        "maximum_ast_node_count": 6,
        "maximum_top_level_clauses": 2,
        "and3_generator_attempts_allowed": False,
        "and3_raw_operator_application_count": 0,
        "formal_bucket_count": 120,
        "traversal_prefix_complete": True,
        "target_roles_evaluated": False,
        "split_material_accessed": False,
        "secrets_accessed": False,
        "aliases_excluded_before_count": [
            "greater_equal",
            "approx_equal:tolerance=0",
        ],
        "active_aggregate_map_ids": [0, 1, 5],
        "tombstoned_aggregate_map_ids": [2, 3, 4],
        "active_rational_parameter_ids": [1, 3, 5],
        "tombstoned_rational_parameter_ids": [0, 2, 4, 6],
        "reserved_rational_parameter_ids": [7],
        "active_source_binary_operator_ids": [1, 2, 3, 4, 5, 6],
        "active_formal_canonical_binary_operator_ids": [1, 2, 3, 5, 6],
        "source_alias_binary_operator_ids": [4],
        "tombstoned_binary_operator_ids": [0],
        "reserved_binary_operator_ids": [7],
        "operator_id_compaction_performed": False,
        "automatic_operator_migration_performed": False,
        **roots,
        "canonical_program_archive_root_or_null": dual._rfc6962_root_from_payloads(
            program_frames
        ).hex(),
        "program_chunk_manifest_root_or_null": dual._rfc6962_root_from_payloads(
            chunk_frames
        ).hex(),
        "bucket_accounting_root_or_null": dual._rfc6962_root_from_payloads(
            bucket_frames
        ).hex(),
        "first_out_of_budget_program_hash_or_null": (
            None if witness is None else witness.digest.hex()
        ),
        "first_out_of_budget_program_cbor_hex_or_null": (
            None if witness is None else witness.cbor_bytes.hex()
        ),
        "first_out_of_budget_program_ordinal_or_null": None if complete else 3,
    }
    return {
        "canonical_program_records": program_stream,
        "program_chunk_manifests": _framed(chunk_frames),
        "bucket_accounting_records": _framed(bucket_frames),
    }, report


def _write_output(
    directory: Path,
    streams: dict[str, bytes],
    report: dict[str, object],
    *,
    implementation: str,
) -> None:
    directory.mkdir()
    value = dict(report)
    value.update(
        {
            "schema_version": dual._EXPECTED_SCHEMAS[implementation],
            "implementation": implementation,
            "implementation_id": dual._EXPECTED_IMPLEMENTATIONS[implementation][0],
            "implementation_machine_id": dual._EXPECTED_IMPLEMENTATIONS[implementation][1],
        }
    )
    if implementation == "python":
        value.update(
            {
                "loaded_hegel_modules": list(dual._EXPECTED_PYTHON_MODULES),
                "target_free_isolation_verified": True,
                "target_or_split_modules_loaded": False,
            }
        )
    (directory / "report.json").write_text(
        json.dumps(value, sort_keys=True) + "\n", encoding="utf-8"
    )
    for name, payload in streams.items():
        (directory / f"{name}.cborframed").write_bytes(payload)


def _dual_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    witness_context_id: int | None = 2,
) -> tuple[Path, Path]:
    monkeypatch.setattr(dual, "CANONICAL_PROGRAM_BUDGET", 2)
    monkeypatch.setattr(dual, "RAW_OPERATOR_APPLICATION_CAP", 10)
    monkeypatch.setattr(dual, "RECORDS_PER_CHUNK", 2)
    expected_witness = (
        None
        if witness_context_id is None
        else canonicalize_shrink6_source_ast(["context_flag", 2])
    )
    monkeypatch.setattr(dual, "EXPECTED_CANONICAL_PROGRAM_COUNT", 2)
    monkeypatch.setattr(dual, "EXPECTED_FIRST_OUT_OF_BUDGET_ORDINAL", 3)
    monkeypatch.setattr(
        dual,
        "EXPECTED_FIRST_OUT_OF_BUDGET_PROGRAM_CBOR_HEX",
        None if expected_witness is None else expected_witness.cbor_bytes.hex(),
    )
    monkeypatch.setattr(
        dual,
        "EXPECTED_FIRST_OUT_OF_BUDGET_PROGRAM_HASH",
        None if expected_witness is None else "sha256:" + expected_witness.digest.hex(),
    )
    monkeypatch.setattr(dual, "EXPECTED_RAW_OPERATOR_APPLICATION_COUNT", 6)
    monkeypatch.setattr(
        dual, "EXPECTED_RESIDUAL_OUT_OF_BUDGET_CANONICAL_PROGRAMS", 4
    )
    monkeypatch.setattr(dual, "EXPECTED_WITNESS_BUCKET_INDEX", 0)
    monkeypatch.setattr(dual, "EXPECTED_WITNESS_OUTPUT_SORT_ID", 1)
    monkeypatch.setattr(dual, "EXPECTED_WITNESS_AST_DEPTH", 0)
    monkeypatch.setattr(dual, "EXPECTED_WITNESS_AST_NODE_COUNT", 1)
    monkeypatch.setattr(
        dual,
        "_assert_host_module_closure",
        lambda: tuple(sorted(dual._EXPECTED_HOST_PROJECT_MODULES)),
    )
    streams, report = _fixture_material(witness_context_id=witness_context_id)
    python = tmp_path / "python"
    rust = tmp_path / "rust"
    _write_output(python, streams, report, implementation="python")
    _write_output(rust, streams, report, implementation="rust")
    return python, rust


def test_public_validator_rejects_a_package_initializer_contaminated_host(
    tmp_path: Path,
) -> None:
    with pytest.raises(dual.Shrink6DualDiagnosticError) as caught:
        dual.validate_shrink6_dual_diagnostic_v1(
            tmp_path / "not-read-python", tmp_path / "not-read-rust"
        )
    assert caught.value.code == dual.FAIL_HOST_ISOLATION


def test_dual_host_replay_is_diagnostic_only_and_recomputes_every_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    python, rust = _dual_fixture(tmp_path, monkeypatch)

    receipt = dual.validate_shrink6_dual_diagnostic_v1(python, rust)

    assert receipt["claim_level"] == CLAIM_LEVEL
    assert receipt["qualification_level"] == "DIAGNOSTIC_ONLY"
    assert receipt["diagnostic_only"] is True
    assert receipt["authoritative_claim_allowed"] is False
    assert receipt["execution_state"] == "NOT_RUN"
    assert receipt["formal_roots_generated"] is False
    assert receipt["formal_roots"] is None
    assert receipt["formal_state_transition_allowed"] is False
    assert receipt["strict_qualification_source_commit"] == (
        STRICT_QUALIFICATION_SOURCE_COMMIT
    )
    assert receipt["strict_qualification_evidence_commit"] == (
        STRICT_QUALIFICATION_EVIDENCE_COMMIT
    )
    assert receipt["strict_qualification_artifact_path"] == (
        STRICT_QUALIFICATION_ARTIFACT_PATH
    )
    assert receipt["strict_qualification_artifact_sha256"] == (
        STRICT_QUALIFICATION_ARTIFACT_SHA256
    )
    assert receipt["strict_qualification_diagnostic_report_hash"] == (
        STRICT_QUALIFICATION_DIAGNOSTIC_REPORT_HASH
    )
    assert receipt["strict_qualification_status"] == STRICT_QUALIFICATION_STATUS
    assert receipt["maximum_ast_depth"] == 3
    assert receipt["maximum_ast_node_count"] == 6
    assert receipt["maximum_top_level_clauses"] == 2
    assert receipt["formal_bucket_count"] == 120
    assert receipt["and3_generator_attempts_allowed"] is False
    assert receipt["and3_raw_operator_application_count"] == 0
    assert receipt["dual_archive_bytes_equal"] is True
    assert receipt["host_strict_archive_replay_verified"] is True
    assert receipt["witness_adjacency_verified"] is True
    assert receipt["typed_language_boundary_independently_derived"] is True
    assert receipt["archive_prefix_exact"] is True
    assert receipt["witness_closed_bucket_rank_verified"] is True
    assert receipt["residual_out_of_budget_canonical_programs"] == 4
    assert receipt["binary_operator_registry_verified"] is True
    assert receipt["removed_binary_operator_absent_from_archive"] is True
    assert receipt["operator_id_compaction_performed"] is False
    assert receipt["automatic_operator_migration_performed"] is False
    assert receipt["independence_scope"] == (
        "INDEPENDENT_OF_ENDPOINT_REPORTED_WITNESS_NOT_A_THIRD_IMPLEMENTATION"
    )
    assert receipt["raw_operator_application_count_scope"] == (
        "THROUGH_FULLY_CLOSED_BOUNDARY_BUCKET"
    )
    assert receipt["first_out_of_budget_program_ordinal"] == 3
    assert receipt["witness_bucket_index"] == 0
    assert receipt["witness_output_sort_id"] == 1
    assert receipt["witness_ast_depth"] == 0
    assert receipt["witness_ast_node_count"] == 1
    assert receipt["prefix_preservation_expectation_id"] == (
        dual.PREFIX_PRESERVATION_EXPECTATION_ID
    )
    assert receipt["prefix_preservation_verified"] is True
    assert receipt["preregistered_shrink_order_total_steps"] == 6
    assert receipt["preregistered_shrink_order_consumed_through_step"] == 6
    assert receipt["next_preregistered_shrink_step_or_null"] is None
    assert receipt["terminal_route"] == (
        "HALT_NO_PREREGISTERED_SHRINK_REMAINING_NEEDS_NEW_NORMATIVE_DECISION"
    )
    assert receipt["budget_change_authorized"] is False
    assert receipt["additional_shrink_authorized"] is False
    assert receipt["new_dsl_version_authorized"] is False


def test_report_field_sets_are_frozen_at_75_and_78() -> None:
    assert len(dual._COMMON_REPORT_FIELDS) == 75
    assert len(dual._EXPECTED_REPORT_FIELDS["rust"]) == 75
    assert len(dual._EXPECTED_REPORT_FIELDS["python"]) == 78
    assert len(dual._EXPECTED_PYTHON_MODULES) == 23
    assert len(dual._EXPECTED_HOST_PROJECT_MODULES) == 24


def test_dual_host_replay_accepts_a_closed_complete_frontier_without_witness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    python, rust = _dual_fixture(
        tmp_path, monkeypatch, witness_context_id=None
    )
    program_keys = tuple(
        dual._program_key(
            canonicalize_shrink6_source_ast(["context_flag", context_id])
        )
        for context_id in range(2)
    )
    bucket_frames = dual._decode_framed_stream(
        (python / "bucket_accounting_records.cborframed").read_bytes(),
        expected_count=120,
        label="fixture buckets",
    )
    bucket_values = tuple(canonical_cbor_decode(frame) for frame in bucket_frames)
    monkeypatch.setattr(
        dual,
        "_derive_frozen_language_boundary",
        lambda: ("COMPLETE", program_keys, None, bucket_values, 2, 0),
    )

    receipt = dual.validate_shrink6_dual_diagnostic_v1(python, rust)

    assert receipt["closure_status"] == "COMPLETE"
    assert receipt["canonical_program_count"] == 2
    assert receipt["residual_out_of_budget_canonical_programs"] == 0
    assert receipt["witness_adjacency_verified"] is None
    assert receipt["first_out_of_budget_program_ordinal"] is None
    assert receipt["witness_bucket_index"] is None
    assert receipt["witness_output_sort_id"] is None
    assert receipt["witness_ast_depth"] is None
    assert receipt["witness_ast_node_count"] is None
    assert receipt["prefix_preservation_verified"] is None
    assert receipt["terminal_route"] == (
        "FORMAL_CHILD_ROOT_AND_IMPLEMENTATION_REQUALIFICATION_ELIGIBLE_NOT_STARTED"
    )
    assert receipt["first_out_of_budget_program_hash"] is None
    assert receipt["first_out_of_budget_program_cbor_hex"] is None
    assert receipt["raw_operator_application_count_scope"] == (
        "THROUGH_FULLY_CLOSED_FRONTIER"
    )


def test_dual_host_replay_rejects_any_cross_language_stream_difference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    python, rust = _dual_fixture(tmp_path, monkeypatch)
    path = rust / "canonical_program_records.cborframed"
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)

    with pytest.raises(dual.Shrink6DualDiagnosticError) as caught:
        dual.validate_shrink6_dual_diagnostic_v1(python, rust)
    assert caught.value.code == dual.FAIL_DUAL


def test_witness_must_have_the_independently_derived_typed_bucket_rank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    python, rust = _dual_fixture(
        tmp_path, monkeypatch, witness_context_id=3
    )

    with pytest.raises(dual.Shrink6DualDiagnosticError) as caught:
        dual.validate_shrink6_dual_diagnostic_v1(python, rust)
    assert caught.value.code == dual.FAIL_WITNESS
    assert "rank 50,001" in caught.value.detail


def test_preservation_mismatch_is_inconclusive_not_a_pass_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    python, rust = _dual_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(dual, "EXPECTED_RAW_OPERATOR_APPLICATION_COUNT", 7)

    with pytest.raises(dual.Shrink6DualDiagnosticError) as caught:
        dual.validate_shrink6_dual_diagnostic_v1(python, rust)
    assert caught.value.code == dual.INCONCLUSIVE_PRESERVATION_MISMATCH
    assert "raw_operator_application_count" in caught.value.detail


@pytest.mark.parametrize("implementation", ["python", "rust"])
def test_report_rejects_every_unknown_field_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    implementation: str,
) -> None:
    python, rust = _dual_fixture(tmp_path, monkeypatch)
    directory = python if implementation == "python" else rust
    path = directory / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["formal_m3_started"] = True
    path.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(dual.Shrink6DualDiagnosticError) as caught:
        dual.validate_shrink6_dual_diagnostic_v1(python, rust)
    assert caught.value.code == dual.FAIL_REPORT
    assert "unknown=['formal_m3_started']" in caught.value.detail


def test_python_report_rejects_false_target_free_isolation_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    python, rust = _dual_fixture(tmp_path, monkeypatch)
    path = python / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["target_or_split_modules_loaded"] = True
    path.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(dual.Shrink6DualDiagnosticError) as caught:
        dual.validate_shrink6_dual_diagnostic_v1(python, rust)
    assert caught.value.code == dual.FAIL_REPORT
    assert "target_or_split_modules_loaded" in caught.value.detail


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("strict_qualification_evidence_commit", "0" * 40),
        ("and3_generator_attempts_allowed", True),
        ("and3_raw_operator_application_count", 1),
        ("closure_status", "INCONCLUSIVE_BUDGET"),
    ],
)
def test_report_rejects_binding_structural_or_nonpublishing_status_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    python, rust = _dual_fixture(tmp_path, monkeypatch)
    path = python / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report[field] = value
    path.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(dual.Shrink6DualDiagnosticError) as caught:
        dual.validate_shrink6_dual_diagnostic_v1(python, rust)
    assert caught.value.code == dual.FAIL_REPORT


@pytest.mark.parametrize(
    "suffix",
    [b',"strict_qualification_evidence_commit":"duplicate"}', b',"x":NaN}'],
)
def test_report_rejects_duplicate_keys_and_nonfinite_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: bytes,
) -> None:
    python, rust = _dual_fixture(tmp_path, monkeypatch)
    path = python / "report.json"
    payload = path.read_bytes().rstrip()
    assert payload.endswith(b"}")
    path.write_bytes(payload[:-1] + suffix + b"\n")

    with pytest.raises(dual.Shrink6DualDiagnosticError) as caught:
        dual.validate_shrink6_dual_diagnostic_v1(python, rust)
    assert caught.value.code == dual.FAIL_REPORT


def test_framed_stream_rejects_zero_length_and_trailing_headers() -> None:
    with pytest.raises(dual.Shrink6DualDiagnosticError) as zero:
        dual._decode_framed_stream(b"\x00\x00\x00\x00", expected_count=1, label="x")
    assert zero.value.code == dual.FAIL_OUTPUT

    with pytest.raises(dual.Shrink6DualDiagnosticError) as trailing:
        dual._decode_framed_stream(b"\x00", expected_count=0, label="x")
    assert trailing.value.code == dual.FAIL_OUTPUT


def test_raw_rfc6962_replay_matches_object_encoder() -> None:
    values = ((1, b"a"), (2, b"b"), (3, b"c"))
    payloads = tuple(canonical_cbor_encode(value) for value in values)
    assert dual._rfc6962_root_from_payloads(payloads) == rfc6962_root(list(values))
