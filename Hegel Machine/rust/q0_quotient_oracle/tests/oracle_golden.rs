use hegel_q0_quotient_oracle::{
    hex_encode, probe_canonical_bytes, projection_manifest_root, run_micro_oracle,
    semantic_binding_root, DIRECT_STATE_ROOT_DOMAIN, EXPECTED_PROBE_CANONICAL_CBOR_HEX,
    EXPECTED_PROJECTION_MANIFEST_ROOT_HEX, EXPECTED_SEMANTIC_BINDING_ROOT_HEX,
    SINGLE_IMPLEMENTATION_PASS_STATUS, SYNTAX_STATE_ROOT_DOMAIN,
};
use sha2::{Digest, Sha256};
use std::sync::OnceLock;

fn endpoint() -> &'static hegel_q0_quotient_oracle::OracleEndpoint {
    static ENDPOINT: OnceLock<hegel_q0_quotient_oracle::OracleEndpoint> = OnceLock::new();
    ENDPOINT.get_or_init(|| run_micro_oracle().expect("Q0 Rust oracle must pass"))
}

fn hex_decode(value: &str) -> Vec<u8> {
    assert_eq!(value.len() % 2, 0, "hex must contain complete bytes");
    (0..value.len())
        .step_by(2)
        .map(|offset| u8::from_str_radix(&value[offset..offset + 2], 16).expect("lower hex"))
        .collect()
}

fn state_root_from_preimage(domain: &[u8], preimage: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update([0]);
    hasher.update(preimage);
    format!("sha256:{}", hex_encode(&hasher.finalize()))
}

#[test]
fn frozen_probe_projection_and_semantic_roots_match() {
    assert_eq!(
        hex_encode(&probe_canonical_bytes()),
        EXPECTED_PROBE_CANONICAL_CBOR_HEX
    );
    assert_eq!(
        hex_encode(&projection_manifest_root()),
        EXPECTED_PROJECTION_MANIFEST_ROOT_HEX
    );
    assert_eq!(
        hex_encode(&semantic_binding_root().expect("semantic binding")),
        EXPECTED_SEMANTIC_BINDING_ROOT_HEX
    );
}

#[test]
fn independent_syntax_and_direct_quotient_golden_vector_matches() {
    let result = endpoint();
    assert_eq!(result.terminal_status, SINGLE_IMPLEMENTATION_PASS_STATUS);
    assert_eq!(result.syntax_raw_operator_applications, 567);
    assert_eq!(result.quotient_raw_operator_applications, 545);
    assert_eq!(result.syntax_strict_admitted_applications, 567);
    assert_eq!(result.quotient_strict_admitted_applications, 545);
    assert_eq!(result.syntax_rewrite_collapses, 30);
    assert_eq!(result.quotient_rewrite_collapses, 30);
    assert_eq!(result.canonical_syntax_count, 537);
    assert_eq!(result.behavior_class_count, 69);
    assert_eq!(result.frontier_point_count, 122);
    assert_eq!(result.maximum_frontier_size, 4);
    assert_eq!(result.syntax_continuation_bank_point_count, 251);
    assert_eq!(result.quotient_continuation_bank_point_count, 251);
    assert_eq!(result.maximum_syntax_bank_points_per_class, 43);
    assert_eq!(result.maximum_quotient_bank_points_per_class, 43);

    assert_eq!(
        result.syntax_program_root,
        "sha256:bd1a59f816bd6648d0dd73b9a1622f2bb88bb9aeca1489a0d876fbc9dbf0c829"
    );
    assert_eq!(
        result.syntax_class_archive_root,
        "sha256:a2f0dacf4524fdb8725d29a2c3883a7ebd78fa686cb2030ac0d0608710176cf1"
    );
    assert_eq!(result.syntax_class_archive_root, result.direct_class_archive_root);
    assert_eq!(
        result.syntax_coverage_root,
        "sha256:6953f39dc97f17288850b524ca8b04dbb2f6ddd3d53eaf4cb8e4e6465bcd840c"
    );
    assert_eq!(
        result.direct_coverage_root,
        "sha256:a9a0b6fdc97c475323ccae31fba14a6df411307220efd8538c7971fe9c38c1fd"
    );
    assert_eq!(
        result.syntax_state_root,
        "sha256:7028819d133c4da6071c06a0bfca2d0b91622e106207d0b0f081148f41c0826a"
    );
    assert_eq!(
        result.direct_state_root,
        "sha256:d87ef33d9d7010ded284b55acfa71aab4d7d991e3d7703c30f1db2caf5893933"
    );
    assert_eq!(
        result.endpoint_state_root,
        "sha256:d33e54dd99e6cbe8aacc541fc0877af9657a553be58523670cce5c474006d4d2"
    );
    assert_eq!(
        format!("sha256:{}", hex_encode(&result.endpoint_root().unwrap())),
        result.endpoint_state_root
    );
}

#[test]
fn pass_is_target_blind_non_authoritative_and_fixed_point_complete() {
    let result = endpoint();
    assert!(result.work_queue_empty);
    assert!(result.zero_delta_full_round);
    assert!(result.all_typed_operator_frontier_tuples_covered);
    assert!(result.exhaustive_syntax_oracle_complete);
    assert!(result.syntax_direct_states_equal);
    assert!(result.resource_guards_ok);
    assert_eq!(result.final_class_delta, 0);
    assert_eq!(result.final_frontier_delta, 0);
    assert_eq!(result.final_bank_delta, 0);
    assert!(!result.target_truth_accessed);
    assert!(!result.split_accessed);
    assert!(!result.role_evaluation_performed);
    assert!(!result.formal_roots_generated);
    assert!(!result.authority_claimed);
    assert!(!result.terminal_status.contains("DUAL"));

    assert_eq!(
        result
            .direct_rounds
            .iter()
            .map(|round| round.round_index)
            .collect::<Vec<_>>(),
        vec![1, 2, 3]
    );

    let final_round = result.direct_rounds.last().expect("zero-delta round");
    assert_eq!(final_round.queued_application_count, 0);
    assert_eq!(final_round.new_canonical_program_count, 0);
    assert_eq!(final_round.new_behavior_class_count, 0);
    assert_eq!(final_round.frontier_mutation_count, 0);
    assert_eq!(final_round.cohort_bank_mutation_count, 0);
    assert!(!final_round.complete_state_changed);
}

#[test]
fn complete_saturation_state_preimages_replay_both_state_roots() {
    let result = endpoint();
    let syntax = hex_decode(&result.syntax_saturation_state_preimage_cbor_hex);
    let direct = hex_decode(&result.direct_saturation_state_preimage_cbor_hex);

    // 0x85 is the canonical fixed-length array header for the exact five-tuple
    // (program records, continuation bank, visible classes, coverage, fixed point).
    assert_eq!(syntax.first(), Some(&0x85));
    assert_eq!(direct.first(), Some(&0x85));
    assert_eq!(syntax.len(), 127_439);
    assert_eq!(direct.len(), 125_153);
    assert_ne!(syntax, direct, "path IDs/program sets/coverage are path-specific");
    assert_eq!(
        state_root_from_preimage(SYNTAX_STATE_ROOT_DOMAIN, &syntax),
        result.syntax_state_root
    );
    assert_eq!(
        state_root_from_preimage(DIRECT_STATE_ROOT_DOMAIN, &direct),
        result.direct_state_root
    );

    let diagnostic: serde_json::Value =
        serde_json::from_str(&result.canonical_json().expect("diagnostic JSON"))
            .expect("diagnostic object");
    assert!(
        result.canonical_json().expect("diagnostic JSON").len() as u64 + 1
            <= hegel_q0_quotient_oracle::MAX_OUTPUT_BYTES,
        "the output guard must include the one LF emitted by println!"
    );
    assert_eq!(
        diagnostic["syntax_saturation_state_preimage_cbor_hex"],
        result.syntax_saturation_state_preimage_cbor_hex
    );
    assert_eq!(
        diagnostic["direct_saturation_state_preimage_cbor_hex"],
        result.direct_saturation_state_preimage_cbor_hex
    );

    // Adding diagnostic preimages must not alter the frozen 43-field root.
    assert_eq!(
        result.endpoint_state_root,
        "sha256:d33e54dd99e6cbe8aacc541fc0877af9657a553be58523670cce5c474006d4d2"
    );
}

#[test]
fn source_has_no_target_split_or_role_oracle_dependency() {
    let source = include_str!("../src/lib.rs");
    for forbidden in [
        "phase3_dsl_v1::",
        "phase3_m25_rows_v1::",
        "static_basis::",
        "target_truth::",
        "split_assignment::",
        "role_matcher::",
    ] {
        assert!(
            !source.contains(forbidden),
            "forbidden dependency marker present: {forbidden}"
        );
    }
}
