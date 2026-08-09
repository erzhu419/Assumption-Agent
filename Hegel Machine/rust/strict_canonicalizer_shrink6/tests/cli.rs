use hegel_strict_canonicalizer::hex_encode;
use hegel_strict_canonicalizer_shrink5::canonicalize_shrink5_source_json;
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::process::{Command, Output};

fn run(arguments: &[&str]) -> (Output, Value) {
    let output = Command::new(env!("CARGO_BIN_EXE_hegel-strict-canonicalizer-shrink6"))
        .args(arguments)
        .output()
        .expect("shrink-6 CLI must start");
    let report: Value = serde_json::from_slice(&output.stdout)
        .expect("shrink-6 CLI must emit exactly one JSON report");
    let canonical = serde_json::to_string(&report).expect("report must reserialize");
    assert_eq!(
        String::from_utf8(output.stdout.clone()).expect("stdout must be UTF-8"),
        format!("{canonical}\n"),
        "default stdout must be exactly one canonical compact JSON line"
    );
    assert!(output.stderr.is_empty(), "successful dispatch must not use stderr");
    (output, report)
}

fn fields(report: &Value) -> BTreeSet<&str> {
    report
        .as_object()
        .expect("CLI report must be an object")
        .keys()
        .map(String::as_str)
        .collect()
}

fn depth_four_source() -> Value {
    json!([
        "sign",
        [
            "absolute",
            [
                "difference",
                ["bit_to_scalar", ["bit_at", 0]],
                ["scalar_const", -1, 1]
            ]
        ]
    ])
}

#[test]
fn source_cli_preserves_accept_and_reject_exit_contract() {
    let (accepted, report) = run(&[
        "--ast-json",
        r#"["absolute",["difference",["bit_to_scalar",["bit_at",0]],["scalar_const",1]]]"#,
    ]);
    assert!(accepted.status.success());
    assert_eq!(report["status"], "ACCEPTED");
    assert_eq!(report["depth"], 3);
    assert_eq!(report["node_count"], 5);
    assert_eq!(report["maximum_ast_depth"], 3);
    assert_eq!(report["maximum_ast_node_count"], 6);
    assert_eq!(report["maximum_top_level_clauses"], 2);
    assert_eq!(report["target_or_split_modules_loaded"], false);
    assert_eq!(
        fields(&report),
        BTreeSet::from([
            "ast_hash_domain",
            "ast_schema_id",
            "boundary",
            "canonical_ast_hash",
            "canonical_cbor_hex",
            "cbor_profile_id",
            "depth",
            "dsl_version",
            "freeze_version",
            "implementation",
            "maximum_ast_depth",
            "maximum_ast_node_count",
            "maximum_top_level_clauses",
            "node_count",
            "output_sort",
            "parent_dsl_version",
            "parent_freeze_version",
            "root_operator_id",
            "scalar_parameter_occurrence_count",
            "schema_version",
            "status",
            "target_or_split_modules_loaded",
        ])
    );

    let (rejected, report) = run(&[
        "--ast-json",
        r#"["sign",["absolute",["difference",["bit_to_scalar",["bit_at",0]],["scalar_const",-1,1]]]]"#,
    ]);
    assert_eq!(rejected.status.code(), Some(1));
    assert_eq!(report["status"], "REJECTED");
    assert_eq!(report["error_code"], "REJECT_STRUCTURAL_LIMIT");
    assert_eq!(report["maximum_ast_depth"], 3);
    assert_eq!(report["maximum_ast_node_count"], 6);
    assert_eq!(report["maximum_top_level_clauses"], 2);
    assert_eq!(report["target_or_split_modules_loaded"], false);
    assert_eq!(
        fields(&report),
        BTreeSet::from([
            "ast_hash_domain",
            "ast_schema_id",
            "boundary",
            "cbor_profile_id",
            "dsl_version",
            "error_code",
            "error_message",
            "freeze_version",
            "implementation",
            "maximum_ast_depth",
            "maximum_ast_node_count",
            "maximum_top_level_clauses",
            "parent_dsl_version",
            "parent_freeze_version",
            "schema_version",
            "status",
            "target_or_split_modules_loaded",
        ])
    );
}

#[test]
fn formal_cli_rejects_a_genuine_canonical_depth_four_parent() {
    let parent = canonicalize_shrink5_source_json(&depth_four_source()).unwrap();
    assert_eq!(parent.depth, 4);
    assert_eq!(parent.node_count, 6);
    let cbor_hex = hex_encode(&parent.canonical_cbor);
    let (rejected, report) = run(&["--decode-cbor-hex", &cbor_hex]);
    assert_eq!(rejected.status.code(), Some(1));
    assert_eq!(report["boundary"], "FORMAL_CBOR");
    assert_eq!(report["generic_cbor_parse"], true);
    assert_eq!(report["error_code"], "REJECT_STRUCTURAL_LIMIT");
    assert_eq!(report["maximum_ast_depth"], 3);
    assert_eq!(report["maximum_ast_node_count"], 6);
    assert_eq!(report["maximum_top_level_clauses"], 2);
    let mut expected = BTreeSet::from([
        "ast_hash_domain",
        "ast_schema_id",
        "boundary",
        "cbor_profile_id",
        "dsl_version",
        "error_code",
        "error_message",
        "freeze_version",
        "implementation",
        "maximum_ast_depth",
        "maximum_ast_node_count",
        "maximum_top_level_clauses",
        "parent_dsl_version",
        "parent_freeze_version",
        "schema_version",
        "status",
        "target_or_split_modules_loaded",
    ]);
    expected.insert("generic_cbor_parse");
    assert_eq!(fields(&report), expected);
}

#[test]
fn builtin_cli_reports_are_sealed_and_fail_closed() {
    let (golden_status, golden) = run(&["--golden-replay"]);
    assert!(golden_status.status.success());
    assert_eq!(golden["vector_count"], 25);
    assert_eq!(golden["passed_count"], 25);
    assert_eq!(golden["source_depth_limit_checks"], 3);
    assert_eq!(golden["formal_depth_limit_checks"], 3);
    assert_eq!(golden["maximum_ast_depth"], 3);
    assert_eq!(golden["maximum_ast_node_count"], 6);
    assert_eq!(golden["maximum_top_level_clauses"], 2);
    assert_eq!(
        golden["golden_vector_manifest_root"],
        "sha256:2690413926d15db52dbd5a502ebe3fdfb1dc74d5ee3c82b2ed868cd16ab34a42"
    );
    assert_eq!(
        golden["golden_outcome_root"],
        "sha256:e5fd0885f95669dc6d369d0d3274778425fabb7e8c6286a27237a1b2bc8d3960"
    );
    assert_eq!(golden["execution_state"], "NOT_RUN");
    assert!(golden["formal_roots"].is_null());
    assert_eq!(
        fields(&golden),
        BTreeSet::from([
            "active_formal_binary_operator_ids",
            "active_source_binary_operator_ids",
            "closure_executed",
            "dsl_version",
            "execution_state",
            "formal_depth_limit_checks",
            "formal_priority_checks",
            "formal_roots",
            "formal_roots_generated",
            "formal_surviving_identity_checks",
            "freeze_version",
            "golden_outcome_root",
            "golden_vector_manifest_root",
            "human_amendment_id",
            "implementation",
            "maximum_ast_depth",
            "maximum_ast_node_count",
            "maximum_top_level_clauses",
            "ordered_vector_ids",
            "parent_dsl_version",
            "parent_freeze_version",
            "passed_count",
            "removed_binary_operator_error",
            "reserved_binary_operator_ids",
            "schema_version",
            "shrink_step_id",
            "source_alias_binary_operator_ids",
            "source_depth_limit_checks",
            "source_normalization_before_limit_checks",
            "source_priority_checks",
            "surviving_identity_checks",
            "target_or_split_modules_loaded",
            "tombstoned_binary_operator_ids",
            "vector_count",
        ])
    );

    let (capacity_status, capacity) = run(&["--capacity-replay"]);
    assert!(capacity_status.status.success());
    assert_eq!(capacity["challenge_source_candidate_count"], 1_266);
    assert_eq!(capacity["challenge_parent_accepted_count"], 1_266);
    assert_eq!(capacity["challenge_parent_canonical_unique_count"], 1_249);
    assert_eq!(capacity["normalized_survivor_source_count"], 67);
    assert_eq!(capacity["normalized_survivor_unique_count"], 50);
    assert_eq!(capacity["inherited_survivor_source_count"], 175);
    assert_eq!(capacity["inherited_survivor_unique_count"], 175);
    assert_eq!(capacity["survivor_source_candidate_count"], 242);
    assert_eq!(capacity["survivor_accepted_count"], 242);
    assert_eq!(capacity["survivor_unique_count"], 225);
    assert_eq!(capacity["survivor_rejected_count"], 0);
    assert_eq!(capacity["parent_only_source_candidate_count"], 1_199);
    assert_eq!(capacity["parent_only_parent_accepted_count"], 1_199);
    assert_eq!(capacity["parent_only_unique_count"], 1_199);
    assert_eq!(capacity["parent_only_depth"], 4);
    assert_eq!(capacity["parent_only_node_count"], 6);
    assert_eq!(capacity["parent_only_source_child_rejected_count"], 1_199);
    assert_eq!(capacity["parent_only_formal_child_rejected_count"], 1_199);
    assert_eq!(
        capacity["challenge_source_lattice_commitment"],
        "sha256:a8cfb37278000933c2c51a2797e5bc0f4e7aad6970b37e178fc681f9358574d0"
    );
    assert_eq!(
        capacity["challenge_parent_canonical_set_commitment"],
        "sha256:8f125763d3098d087dd7e9eb484b93097295ebd765b6f079795e8009623fb13e"
    );
    assert_eq!(
        capacity["normalized_survivor_set_commitment"],
        "sha256:dcbb5562fc754fdef932188b189dbcdc0f7c500d3fc49651ee4dbb0f271afd29"
    );
    assert_eq!(
        capacity["inherited_survivor_set_commitment"],
        "sha256:477a5abe659a7a7e7d2d50b2a5bda61b0dae1019c44fe84950c4a05036258619"
    );
    assert_eq!(
        capacity["survivor_accepted_set_commitment"],
        "sha256:6787cd6c0782fda149e1ee93b37ca8d425f5ac78850c610e21cebf9da13a16d1"
    );
    assert_eq!(
        capacity["parent_only_set_commitment"],
        "sha256:d3eb2b2d9caf1eece5a709d8113540e4709d579cdfbe3194f1cf176c9100b20d"
    );
    assert_eq!(
        capacity["parent_only_source_rejection_outcome_commitment"],
        "sha256:9b0b766a4139db6297aea8b6032ad49147c1a26bf9b56291444a83681428cb0e"
    );
    assert_eq!(
        capacity["parent_only_formal_rejection_outcome_commitment"],
        "sha256:97d50c34f51683a2502157961acc79d3b4e108b28bdaa266cf3721ffda8b3a96"
    );
    assert_eq!(capacity["maximum_ast_depth"], 3);
    assert_eq!(capacity["maximum_ast_node_count"], 6);
    assert_eq!(capacity["maximum_top_level_clauses"], 2);
    assert_eq!(capacity["executed_closure_status"], "NOT_RUN");
    assert!(capacity["formal_roots"].is_null());
    assert_eq!(capacity["target_or_split_modules_loaded"], false);
    assert_eq!(
        fields(&capacity),
        BTreeSet::from([
            "canonical_program_budget",
            "challenge_parent_accepted_count",
            "challenge_parent_canonical_set_commitment",
            "challenge_parent_canonical_unique_count",
            "challenge_source_candidate_count",
            "challenge_source_family_counts",
            "challenge_source_lattice_commitment",
            "complete_closure_enumerated",
            "constant_atom_count",
            "dsl_version",
            "executed_closure_status",
            "first_out_of_budget_ordinal",
            "first_survivor_canonical_ast_hash",
            "first_survivor_canonical_cbor_hex",
            "formal_roots",
            "freeze_version",
            "generator_rule",
            "human_amendment_id",
            "implementation",
            "inherited_survivor_set_commitment",
            "inherited_survivor_source_count",
            "inherited_survivor_unique_count",
            "interpreted_as_complete_closure",
            "last_survivor_canonical_ast_hash",
            "last_survivor_canonical_cbor_hex",
            "maximum_ast_depth",
            "maximum_ast_node_count",
            "maximum_top_level_clauses",
            "mixed_atom_count",
            "normalized_survivor_set_commitment",
            "normalized_survivor_source_count",
            "normalized_survivor_source_family_counts",
            "normalized_survivor_unique_count",
            "parent_dsl_version",
            "parent_freeze_version",
            "parent_only_depth",
            "parent_only_formal_child_rejected_count",
            "parent_only_formal_child_rejection_counts",
            "parent_only_formal_rejection_outcome_commitment",
            "parent_only_node_count",
            "parent_only_parent_accepted_count",
            "parent_only_set_commitment",
            "parent_only_source_candidate_count",
            "parent_only_source_child_rejected_count",
            "parent_only_source_child_rejection_counts",
            "parent_only_source_family_counts",
            "parent_only_source_rejection_outcome_commitment",
            "parent_only_unique_count",
            "rational_aggregate_count",
            "removed_binary_operator_ids",
            "retained_difference_id",
            "schema_version",
            "shrink_step_id",
            "subset_status",
            "survivor_accepted_count",
            "survivor_accepted_set_commitment",
            "survivor_parent_identity_match_count",
            "survivor_rejected_count",
            "survivor_rejection_counts",
            "survivor_source_candidate_count",
            "survivor_unique_count",
            "target_or_split_modules_loaded",
        ])
    );
}
