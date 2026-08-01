from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

import pytest

import hegel_machine.phase3_certificate_v1 as cert


def digest(digit: str) -> str:
    return f"sha256:{digit * 64}"


def bucket(*, accepted: int = 2, raw: int = 10) -> cert.BucketCount:
    return cert.BucketCount(
        output_sort="Bit",
        depth=0,
        node_count=0,
        raw_operator_applications=raw,
        accepted_canonical_programs=accepted,
        canonical_duplicates=1,
        type_rejections=2,
        limit_rejections=0,
    )


def receipt(language: str, *, matches: int = 0) -> cert.ExhaustionReceipt:
    del matches
    return cert.ExhaustionReceipt(
        implementation_id=f"{language}-independent-v1",
        dsl_spec_root=digest("1"),
        bucket_counts=(bucket(),),
        raw_operator_application_count=10,
        canonical_program_count=2,
        frontier_exhausted=True,
        program_archive_root=digest("2"),
        output_archive_root=digest("3"),
        exhaustion_receipt_root=digest("4" if language == "python" else "5"),
    )


def replay(
    language: cert.ReplayLanguage,
    *,
    match_hashes: tuple[str, ...] = (),
    status: cert.ReplayStatus = cert.ReplayStatus.COMPLETE,
) -> cert.ReplaySummary:
    source_digits = {
        cert.ReplayLanguage.PYTHON: ("6", "7", "8"),
        cert.ReplayLanguage.RUST: ("9", "a", "b"),
    }[language]
    return cert.ReplaySummary(
        language=language,
        status=status,
        receipt=receipt(language.value),
        operator_semantics_root=digest("c"),
        identifier_registry_root=digest("d"),
        canonicalizer_source_root=digest(source_digits[0]),
        enumerator_source_root=digest(source_digits[1]),
        evaluator_source_root=digest(source_digits[2]),
        bounded_universe_root=digest("e"),
        target_truth_table_root=digest("f"),
        chunk_manifest_root=digest("0"),
        match_program_hashes=match_hashes,
        undefined_target_row_count=0,
        raw_expansion_limit_hit=False,
        wall_clock_abort_hit=False,
        all_type_buckets_closed=True,
    )


def agreement() -> cert.ReplayAgreement:
    return cert.ReplayAgreement(
        python=replay(cert.ReplayLanguage.PYTHON),
        rust=replay(cert.ReplayLanguage.RUST),
    )


def claim() -> cert.OutsideFrozenClosureClaim:
    return cert.OutsideFrozenClosureClaim(
        dsl_version=cert.DSL_VERSION,
        bounded_universe_root=digest("e"),
        target_truth_table_root=digest("f"),
    )


def body() -> cert.OutsideCertificateBody:
    pair = agreement()
    return cert.OutsideCertificateBody(
        claim=claim(),
        dsl_spec_status=cert.DslSpecStatus.FROZEN,
        target_commitment_precedes_synthesis=True,
        replay_agreement=pair,
        covert_channel_audit_pass=True,
        key_epoch=1,
        issued_at="2030-01-01T00:00:00Z",
        python_replay_environment=cert.ReplayEnvironmentBinding(
            language=cert.ReplayLanguage.PYTHON,
            replay_implementation_id=pair.python.implementation_id,
            repository_commit_sha="a" * 40,
            container_image_digest=digest("6"),
        ),
        rust_replay_environment=cert.ReplayEnvironmentBinding(
            language=cert.ReplayLanguage.RUST,
            replay_implementation_id=pair.rust.implementation_id,
            repository_commit_sha="b" * 40,
            container_image_digest=digest("7"),
        ),
    )


def public_key_records(
    *,
    epoch: int,
    public_hexes: tuple[str, str, str] | None = None,
) -> tuple[cert.Ed25519PublicKeyRecord, ...]:
    values = public_hexes or ("01" * 32, "02" * 32, "03" * 32)
    return tuple(
        cert.Ed25519PublicKeyRecord(
            role=role,
            key_id=f"key-{index}",
            key_epoch=epoch,
            public_key_hex=values[index],
        )
        for index, role in enumerate(cert.FORMAL_CERTIFICATE_ROLES)
    )


def binding_mapping() -> dict[str, object]:
    root_fields = (
        "mdl_code_table_root",
        "dsl_spec_root",
        "identifier_registry_root",
        "discovery_partition_root",
        "validation_partition_root",
        "sealed_partition_root",
        "target_truth_table_root",
        "old_program_ast_hash",
        "new_symbol_definition_hash",
        "new_call_program_ast_hash",
        "old_prediction_vector_root",
        "new_prediction_vector_root",
        "validation_prediction_root",
        "sealed_prediction_root",
        "container_image_digest",
    )
    data: dict[str, object] = {
        "schema_version": cert.MDL_BINDINGS_SCHEMA,
        "fixed_point_precision_id": cert.FIXED_POINT_PRECISION_ID,
        "mdl_algorithm_id": "hegel-mdl-replay-v1",
        "repository_commit_sha": "b" * 40,
    }
    data.update({name: digest(format(index % 16, "x")) for index, name in enumerate(root_fields)})
    return data


def mdl_request_mapping() -> dict[str, object]:
    return {
        "schema_version": cert.MDL_REQUEST_SCHEMA,
        "bindings": binding_mapping(),
        "code_table_id": cert.MDL_CODE_TABLE_ID,
        "old_program_ast": {"op": "old", "children": []},
        "new_symbol_definition": {"class": "bounded_generic_reducer"},
        "new_call_program_ast": {"op": "new_symbol_call", "children": []},
        "discovery_target_labels": [0, 1, 0, 1],
        "old_discovery_predictions": [None, 1, 1, 1],
        "new_discovery_predictions": [0, 1, 0, 1],
    }


def test_canonical_cbor_fixed_vector_or_hard_disabled_without_backend():
    with pytest.raises(TypeError, match="floating point"):
        cert.FrozenCborMap.from_mapping({"bad": 0.5})

    # RFC 8949 deterministic map ordering: the shorter encoded key comes first.
    vector = {"aa": 1, "b": 2}
    expected = bytes.fromhex("a261620262616101")
    if cert.CANONICAL_CBOR_ENCODER_IMPLEMENTED:
        assert cert.canonical_cbor_bytes(vector) == expected
    else:
        with pytest.raises(cert.CapabilityUnavailable, match="canonical CBOR"):
            cert.canonical_cbor_bytes(vector)
        # Preserve the literal interoperability vector even in minimal installs.
        assert expected == b"\xa2\x61\x62\x02\x62\x61\x61\x01"


def test_frozen_cbor_map_is_recursive_sorted_and_strict():
    frozen = cert.FrozenCborMap.from_mapping({"z": [1, {"b": 2}], "a": b"x"})
    assert tuple(key for key, _ in frozen.entries) == ("a", "z")
    assert frozen.to_mapping() == {"a": b"x", "z": [1, {"b": 2}]}
    with pytest.raises(TypeError, match="map keys"):
        cert.FrozenCborMap.from_mapping({1: "not allowed"})


def test_rfc6962_empty_and_single_leaf_known_vectors():
    assert cert.rfc6962_merkle_root(()) == sha256(b"").digest()
    assert cert.rfc6962_leaf_hash(b"").hex() == (
        "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d"
    )
    assert cert.rfc6962_merkle_root((b"",)) == cert.rfc6962_leaf_hash(b"")


def test_rfc6962_three_leaf_golden_uses_largest_split_without_tail_duplication():
    leaves = (b"a", b"b", b"c")
    assert cert.rfc6962_leaf_hash(b"a").hex() == (
        "022a6979e6dab7aa5ae4c3e5e45f7e977112a7e63593820dbec1ec738a24f93c"
    )
    assert cert.rfc6962_merkle_root(leaves).hex() == (
        "36642e73c2540ab121e3a6bf9545b0a24982cd830eb13d3cd19de3ce6c021ec1"
    )
    duplicated_tail = cert.rfc6962_node_hash(
        cert.rfc6962_node_hash(
            cert.rfc6962_leaf_hash(b"a"),
            cert.rfc6962_leaf_hash(b"b"),
        ),
        cert.rfc6962_node_hash(
            cert.rfc6962_leaf_hash(b"c"),
            cert.rfc6962_leaf_hash(b"c"),
        ),
    )
    assert cert.rfc6962_merkle_root(leaves) != duplicated_tail


def test_program_record_schema_is_exact_and_accepts_frozen_zero_placeholders():
    record = {
        "schema_version": cert.PROGRAM_RECORD_SCHEMA,
        "program_index": 0,
        "canonical_ast": {},
        "canonical_ast_hash": digest("0"),
        "output_sort": "Bit",
        "depth": 0,
        "node_count": 0,
        "distinct_entity_slot_count": 0,
        "program_code_length_q32": 0,
        "undefined_row_bitmap_hash": digest("1"),
        "output_vector_hash": digest("2"),
        "extensional_class_hash": digest("3"),
        "first_extensional_representative_index": 0,
        "dsl_spec_root": digest("4"),
        "bounded_universe_root": digest("5"),
    }
    parsed = cert.ProgramRecord.from_mapping(record)
    assert parsed.to_mapping() == record
    with pytest.raises(ValueError, match="schema mismatch"):
        cert.ProgramRecord.from_mapping({**record, "caller_claim": "trust me"})
    with pytest.raises(TypeError, match="program_index"):
        cert.ProgramRecord.from_mapping({**record, "program_index": True})


def test_universe_and_target_rows_require_exact_alignment():
    universe = (
        cert.UniverseRow(
            universe_index=0,
            input_signature_id="parity-row",
            canonical_input=cert.FrozenCborMap.from_mapping({"x": [0, 1]}),
            canonical_input_hash=digest("1"),
        ),
    )
    target = (
        cert.TargetTruthRow(
            universe_index=0,
            canonical_input_hash=digest("1"),
            target_output=1,
        ),
    )
    cert.validate_universe_and_target_rows(universe, target)
    assert set(universe[0].to_mapping()) == {
        "universe_index",
        "input_signature_id",
        "canonical_input",
        "canonical_input_hash",
    }
    assert set(target[0].to_mapping()) == {
        "universe_index",
        "canonical_input_hash",
        "target_output",
    }
    with pytest.raises(ValueError, match="schema mismatch"):
        cert.UniverseRow.from_mapping({**universe[0].to_mapping(), "schema_version": "invented"})
    with pytest.raises(ValueError, match="does not bind"):
        cert.validate_universe_and_target_rows(
            universe,
            (replace(target[0], canonical_input_hash=digest("2")),),
        )


def test_chunk_geometry_is_exact_for_8193_records():
    chunks = tuple(
        cert.ChunkManifest(
            chunk_index=index,
            first_program_index=index * cert.RECORDS_PER_CHUNK,
            last_program_index=index * cert.RECORDS_PER_CHUNK + count - 1,
            record_count=count,
            record_merkle_root=digest("1"),
            compressed_blob_sha256=digest("2"),
            uncompressed_byte_length=0,
        )
        for index, count in enumerate((4096, 4096, 1))
    )
    cert.validate_chunk_manifests(chunks, canonical_program_count=8193)
    assert "schema_version" not in chunks[0].to_mapping()
    with pytest.raises(ValueError, match="exactly 4,096"):
        cert.validate_chunk_manifests(
            (replace(chunks[0], record_count=4095, last_program_index=4094),) + chunks[1:],
            canonical_program_count=8193,
        )


def test_exhaustion_receipt_recomputes_bucket_totals_and_has_no_fake_root_rule():
    valid = receipt("python")
    assert valid.canonical_program_count == 2
    with pytest.raises(ValueError, match="accepted counts"):
        replace(valid, canonical_program_count=3)
    assert "exhaustion_receipt_root_preimage_exclusion_rule_not_frozen" in (
        cert.SPECIFICATION_RESOLUTION_BLOCKERS
    )


def test_python_and_rust_replays_agree_only_when_outputs_and_sources_are_independent():
    pair = agreement()
    assert pair.agreement_failures() == ()
    assert pair.outside_condition_failures() == ()

    shared_source = replace(
        pair.rust,
        canonicalizer_source_root=pair.python.canonicalizer_source_root,
    )
    failures = cert.ReplayAgreement(pair.python, shared_source).agreement_failures()
    assert "shared_canonicalizer_source_root" in failures

    wrong_output = replace(
        pair.rust,
        receipt=replace(pair.rust.receipt, output_archive_root=digest("9")),
    )
    assert "replay_output_archive_root_mismatch" in cert.ReplayAgreement(
        pair.python,
        wrong_output,
    ).agreement_failures()

    wrong_chunk_manifest = replace(pair.rust, chunk_manifest_root=digest("1"))
    assert "replay_chunk_manifest_root_mismatch" in cert.ReplayAgreement(
        pair.python,
        wrong_chunk_manifest,
    ).agreement_failures()

    with pytest.raises(ValueError, match="chunk_manifest_root"):
        replace(pair.python, chunk_manifest_root="not-a-sha256-root")


def test_nonempty_match_set_and_incomplete_replay_block_outside_claim():
    match = (digest("a"),)
    pair = cert.ReplayAgreement(
        replay(cert.ReplayLanguage.PYTHON, match_hashes=match),
        replay(
            cert.ReplayLanguage.RUST,
            match_hashes=match,
            status=cert.ReplayStatus.INCONCLUSIVE_BUDGET,
        ),
    )
    failures = pair.outside_condition_failures()
    assert "python_match_set_not_empty" in failures
    assert "rust_match_set_not_empty" in failures
    assert "rust_replay_not_complete" in failures


def test_over_budget_receipts_remain_representable_but_cannot_certify_closure():
    oversized_bucket = replace(
        bucket(),
        accepted_canonical_programs=50_001,
        raw_operator_applications=5_000_001,
    )

    def oversized(value: cert.ReplaySummary) -> cert.ReplaySummary:
        return replace(
            value,
            receipt=replace(
                value.receipt,
                bucket_counts=(oversized_bucket,),
                canonical_program_count=50_001,
                raw_operator_application_count=5_000_001,
            ),
        )

    pair = cert.ReplayAgreement(
        oversized(replay(cert.ReplayLanguage.PYTHON)),
        oversized(replay(cert.ReplayLanguage.RUST)),
    )
    failures = pair.outside_condition_failures()
    assert "python_canonical_program_limit_exceeded" in failures
    assert "rust_canonical_program_limit_exceeded" in failures
    assert "python_raw_operator_application_limit_exceeded" in failures
    assert "rust_raw_operator_application_limit_exceeded" in failures


def test_only_exact_bounded_outside_frozen_closure_claim_is_constructible():
    value = claim()
    assert value.render() == (
        "OUTSIDE_FROZEN_CLOSURE("
        f"{cert.DSL_VERSION},{digest('e')},{digest('f')},"
        "equivalence=exact_extensional)"
    )
    with pytest.raises(ValueError, match="only authorized"):
        replace(value, claim_kind="OUTSIDE_LANGUAGE")
    with pytest.raises(ValueError, match="exact_extensional"):
        replace(value, equivalence="observational")


def test_certificate_body_binds_each_replay_to_its_own_environment_record():
    value = body()
    encoded = value.to_mapping()
    assert "repository_commit_sha" not in encoded
    assert "container_image_digest" not in encoded
    assert encoded["python_replay_environment"] != encoded["rust_replay_environment"]

    with pytest.raises(ValueError, match="wrong replay language"):
        replace(value, rust_replay_environment=value.python_replay_environment)

    mismatched_rust = replace(
        value.rust_replay_environment,
        replay_implementation_id="rust-independent-v2",
    )
    with pytest.raises(ValueError, match="does not bind its replay implementation"):
        replace(value, rust_replay_environment=mismatched_rust)


def test_ideal_machine_conditions_still_cannot_issue_without_real_capabilities():
    ideal_body = body()
    assert ideal_body.machine_condition_failures() == ()
    signatures = tuple(
        cert.DetachedSignature(
            role=role,
            key_id=f"key-{index}",
            key_epoch=1,
            signature_hex="00" * 64,
        )
        for index, role in enumerate(cert.FORMAL_CERTIFICATE_ROLES)
    )
    result = cert.verify_outside_certificate(
        cert.OutsideCertificateEnvelope(ideal_body, signatures),
        public_key_records(epoch=1),
        latest_key_epoch=1,
    )
    assert not result.issued
    assert result.claim is None
    assert "python_closure_replay_unimplemented" in result.failures
    assert "rust_closure_replay_unimplemented" in result.failures
    assert "latest_key_status_resolver_unimplemented" in result.failures


@pytest.mark.skipif(
    not cert.ED25519_VERIFIER_IMPLEMENTED,
    reason="cryptography Ed25519 backend is unavailable",
)
def test_real_ed25519_three_of_three_and_rotation_two_of_three_thresholds():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private_keys = tuple(Ed25519PrivateKey.generate() for _ in range(3))
    public_hexes = tuple(
        key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        .hex()
        for key in private_keys
    )
    keys = public_key_records(epoch=3, public_hexes=public_hexes)  # type: ignore[arg-type]
    signed_digest = sha256(b"certificate body").digest()
    signatures = tuple(
        cert.DetachedSignature(
            role=role,
            key_id=f"key-{index}",
            key_epoch=3,
            signature_hex=private_keys[index].sign(signed_digest).hex(),
        )
        for index, role in enumerate(cert.FORMAL_CERTIFICATE_ROLES)
    )
    formal = cert.verify_ed25519_digest(
        signed_digest,
        signatures,
        keys,
        threshold=3,
        required_roles=cert.FORMAL_CERTIFICATE_ROLES,
        key_epoch=3,
    )
    assert formal.passed
    assert set(formal.valid_roles) == set(cert.FORMAL_CERTIFICATE_ROLES)

    missing_role = cert.verify_ed25519_digest(
        signed_digest,
        signatures[:2],
        keys,
        threshold=3,
        required_roles=cert.FORMAL_CERTIFICATE_ROLES,
        key_epoch=3,
    )
    assert not missing_role.passed
    assert "signature_threshold_not_met" in missing_role.failures

    rotation = cert.verify_ed25519_digest(
        signed_digest,
        signatures[:2],
        keys,
        threshold=2,
        key_epoch=3,
    )
    assert rotation.passed


def test_key_epoch_and_revocation_records_are_role_and_history_strict():
    new_keys = public_key_records(epoch=2)
    manifest = cert.KeyEpochManifest(
        key_epoch=2,
        previous_key_epoch=1,
        new_public_keys=new_keys,
        effective_at="2030-01-01T00:00:00Z",
        reason="scheduled rotation",
        invalidate_certificates_before=None,
    )
    assert manifest.previous_key_epoch == 1
    with pytest.raises(ValueError, match="adjacent"):
        replace(manifest, previous_key_epoch=2)
    with pytest.raises(ValueError, match="exactly three"):
        replace(manifest, new_public_keys=new_keys[:2])

    revocation = cert.KeyRevocationManifest(
        key_epoch=2,
        revoked_key_ids=("key-1",),
        effective_at="2030-02-01T00:00:00Z",
        reason="device loss",
        invalidate_certificates_before=None,
        invalidate_certificates_after=None,
    )
    assert revocation.invalidate_certificates_before is None
    assert revocation.invalidate_certificates_after is None


def test_rotation_requires_complete_three_role_old_epoch_trust_store_before_cbor():
    keys = public_key_records(epoch=3)

    incomplete = cert.verify_rotation_or_revocation_signatures(
        {},
        (),
        keys[:2],
        old_key_epoch=3,
    )
    assert not incomplete.passed
    assert "old_epoch_trust_store_size_not_three" in incomplete.failures
    assert (
        "old_epoch_trust_store_missing_role:K_replay_rust" in incomplete.failures
    )

    duplicate_role_keys = (
        keys[0],
        replace(keys[1], role=cert.KeyRole.CUSTODIAN),
        keys[2],
    )
    duplicate_role = cert.verify_rotation_or_revocation_signatures(
        {},
        (),
        duplicate_role_keys,
        old_key_epoch=3,
    )
    assert not duplicate_role.passed
    assert "old_epoch_trust_store_duplicate_role" in duplicate_role.failures
    assert (
        "old_epoch_trust_store_missing_role:K_replay_python"
        in duplicate_role.failures
    )

    wrong_epoch = cert.verify_rotation_or_revocation_signatures(
        {},
        (),
        keys[:2] + (replace(keys[2], key_epoch=2),),
        old_key_epoch=3,
    )
    assert not wrong_epoch.passed
    assert "old_epoch_trust_store_key_epoch_mismatch:key-2" in wrong_epoch.failures

    duplicate_id = cert.verify_rotation_or_revocation_signatures(
        {},
        (),
        keys[:2] + (replace(keys[2], key_id=keys[1].key_id),),
        old_key_epoch=3,
    )
    assert not duplicate_id.passed
    assert "old_epoch_trust_store_duplicate_key_id" in duplicate_id.failures


def test_frozen_prefix_tables_are_prefix_free_and_exact():
    assert cert.prefix_code_is_prefix_free(tuple(cert.AST_SHAPE_PREFIXES.values()))
    assert cert.AST_SHAPE_PREFIXES["top_level_and_3"] == "111110"
    assert cert.LEAF_CLASS_CODES["new_symbol_call"] == "110"
    assert cert.BINARY_TOKEN_CODES["difference"] == "001"
    assert cert.TERNARY_TOKEN_CODES["approx_equal"] == "0"
    assert cert.RATIONAL_PARAMETER_CODES["-1/2"] == "010"
    assert cert.TOLERANCE_CODES["1/2"] == "10"


@pytest.mark.parametrize(
    ("index", "length"),
    ((1, 1), (2, 4), (3, 4), (4, 5), (8, 8), (16, 9)),
)
def test_elias_delta_lengths_match_frozen_formula(index: int, length: int):
    assert cert.elias_delta_bit_length(index) == length


def test_scope_aggregate_and_new_reducer_code_lengths_are_recomputed():
    assert cert.scope_extension_code_length_bits(0) == 1
    assert cert.scope_extension_code_length_bits(1) == 5
    assert cert.scope_extension_code_length_bits(2) == 8
    assert cert.aggregate_leaf_code_length_bits(0) == 12
    assert cert.new_reducer_fixed_code_length_bits(arity=2, clause_count=0) == (
        16 + cert.elias_delta_bit_length(2) + 8 + 4 + 1 + 3 + 4 + 1 + 256
    )


def test_q32_log_and_binary_enumerative_code_match_fixed_golden_integers():
    assert cert.ceil_log2_q32_integer(1) == 0
    assert cert.ceil_log2_q32_integer(2) == cert.Q32_SCALE
    assert cert.ceil_log2_q32_integer(8) == 3 * cert.Q32_SCALE
    assert cert.Q32_SCALE < cert.ceil_log2_q32_integer(3) < 2 * cert.Q32_SCALE
    assert cert.binary_enumerative_data_code_length_q32(0, 0) == 0
    # Precomputed with a high-precision reference, not another module helper.
    assert cert.binary_enumerative_data_code_length_q32(3, 1) == 15_397_296_698
    assert cert.binary_enumerative_data_code_length_q32(3, 2) == 15_397_296_698
    assert cert.binary_enumerative_data_code_length_q32(4, 2) == 21_074_934_634
    assert cert.binary_enumerative_data_code_length_q32(192, 96) == 839_547_347_179
    assert cert.binary_enumerative_data_code_length_q32(480, 240) == 2_079_322_295_041
    with pytest.raises(TypeError, match="integer"):
        cert.ceil_log2_q32_integer(2.0)  # type: ignore[arg-type]


def test_mdl_required_gain_uses_exact_q32_max_of_32_bits_and_five_percent():
    assert cert.mdl_required_gain_q32(100 * cert.Q32_SCALE) == 32 * cert.Q32_SCALE
    assert cert.mdl_required_gain_q32(1000 * cert.Q32_SCALE) == 50 * cert.Q32_SCALE


def test_mdl_request_ignores_caller_lengths_and_formal_scorer_stays_disabled():
    raw = mdl_request_mapping()
    raw.update(
        {
            "length": 0,
            "Fraction": "1/0",
            "L_old_program": 0,
            "L_train_given_old": 0,
            "L_new_symbol_definition": 0,
            "L_new_call_program": 0,
            "L_train_given_new": 0,
            "delta_L": 10**100,
            "required_delta_L": 0,
            "threshold_pass": True,
        }
    )
    request = cert.MdlReplayRequest.from_mapping(raw)
    assert set(request.ignored_caller_fields) == set(raw) & cert.MDL_IGNORED_CALLER_FIELDS
    result = cert.score_mdl_formally(request)
    assert result.status is cert.MdlScorerStatus.HARD_DISABLED
    assert not result.formal_gate_pass
    assert result.old_error_count == 2
    assert result.new_error_count == 0
    assert result.train_given_old_q32 == cert.binary_enumerative_data_code_length_q32(4, 2)
    assert result.train_given_new_q32 == cert.binary_enumerative_data_code_length_q32(4, 0)
    assert result.old_program_length_q32 is None
    assert result.new_symbol_definition_length_q32 is None
    assert result.new_call_program_length_q32 is None
    assert result.delta_l_q32 is None
    assert "formal_mdl_ast_scorer_unimplemented" in result.blockers
    assert "rust_mdl_replay_unimplemented" in result.blockers


def test_mdl_request_rejects_unrelated_unknown_fields_and_bad_precision():
    raw = mdl_request_mapping()
    raw["proof_looking_json"] = True
    with pytest.raises(ValueError, match="schema mismatch"):
        cert.MdlReplayRequest.from_mapping(raw)

    bindings = binding_mapping()
    bindings["fixed_point_precision_id"] = "binary-float"
    with pytest.raises(ValueError, match="unsigned-Q32"):
        cert.MdlCertificateBindings.from_mapping(bindings)


def test_unresolved_spec_items_are_machine_readable_and_formal_paths_fail_closed():
    blockers = set(cert.SPECIFICATION_RESOLUTION_BLOCKERS)
    assert "canonical_cbor_backend_not_declared_as_project_dependency" in blockers
    assert "program_output_blob_archive_record_and_root_schema_not_frozen" in blockers
    assert "final_certificate_envelope_and_timestamp_schema_not_frozen" in blockers
    assert "latest_key_status_manifest_discovery_and_trust_anchor_not_frozen" in blockers
    assert "cross_language_q32_log2_reference_algorithm_not_frozen" in blockers
    assert not cert.FORMAL_OUTSIDE_CERTIFICATE_ISSUANCE_IMPLEMENTED
    assert cert.outside_certificate_capability_failures()
    assert cert.formal_mdl_capability_failures()
