use serde_json::Value;
use std::collections::BTreeSet;
use std::process::{Command, Output};

fn run(arguments: &[&str]) -> (Output, Value) {
    let output = Command::new(env!("CARGO_BIN_EXE_hegel-strict-canonicalizer-shrink5"))
        .args(arguments)
        .output()
        .expect("shrink-5 CLI must start");
    let report: Value = serde_json::from_slice(&output.stdout)
        .expect("shrink-5 CLI must emit exactly one JSON report");
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

#[test]
fn source_cli_preserves_accept_and_reject_exit_contract() {
    let (accepted, report) = run(&[
        "--ast-json",
        r#"["absolute",["difference",["bit_to_scalar",["bit_at",0]],["bit_to_scalar",["bit_at",1]]]]"#,
    ]);
    assert!(accepted.status.success());
    assert_eq!(report["status"], "ACCEPTED");
    assert_eq!(report["node_count"], 6);
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
        r#"["absolute",["difference",["bit_to_scalar",["bit_at",0]],["absolute",["bit_to_scalar",["bit_at",1]]]]]"#,
    ]);
    assert_eq!(rejected.status.code(), Some(1));
    assert_eq!(report["status"], "REJECTED");
    assert_eq!(report["error_code"], "REJECT_STRUCTURAL_LIMIT");
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
fn formal_cli_rejects_the_canonical_seven_node_parent() {
    let (rejected, report) = run(&[
        "--decode-cbor-hex",
        "82018301028402018301008300010083010283010083000101",
    ]);
    assert_eq!(rejected.status.code(), Some(1));
    assert_eq!(report["boundary"], "FORMAL_CBOR");
    assert_eq!(report["generic_cbor_parse"], true);
    assert_eq!(report["error_code"], "REJECT_STRUCTURAL_LIMIT");
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
    assert_eq!(golden["vector_count"], 22);
    assert_eq!(golden["passed_count"], 22);
    assert_eq!(golden["maximum_ast_node_count"], 6);
    assert_eq!(golden["maximum_top_level_clauses"], 2);
    assert_eq!(
        golden["golden_vector_manifest_root"],
        "sha256:156f7e20407437bb753b097a87932f469701d1de6d1d577b0fa1b7a98f47e52e"
    );
    assert_eq!(
        golden["golden_outcome_root"],
        "sha256:8f82178c0f33d5295601d2e112b0b6e25ef18d73e5fc35d8d601024c1f0ddf94"
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
            "formal_priority_checks",
            "formal_roots",
            "formal_roots_generated",
            "formal_structural_limit_checks",
            "formal_surviving_identity_checks",
            "freeze_version",
            "golden_outcome_root",
            "golden_vector_manifest_root",
            "human_amendment_id",
            "implementation",
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
            "source_normalization_before_limit_checks",
            "source_priority_checks",
            "source_structural_limit_checks",
            "surviving_identity_checks",
            "target_or_split_modules_loaded",
            "tombstoned_binary_operator_ids",
            "vector_count",
        ])
    );

    let (capacity_status, capacity) = run(&["--capacity-replay"]);
    assert!(capacity_status.status.success());
    assert_eq!(capacity["survivor_source_candidate_count"], 175);
    assert_eq!(capacity["survivor_accepted_count"], 175);
    assert_eq!(capacity["survivor_unique_count"], 175);
    assert_eq!(capacity["survivor_rejected_count"], 0);
    assert_eq!(capacity["parent_only_source_candidate_count"], 2_160);
    assert_eq!(capacity["parent_only_parent_accepted_count"], 2_160);
    assert_eq!(capacity["parent_only_node_count"], 7);
    assert_eq!(capacity["parent_only_source_child_rejected_count"], 2_160);
    assert_eq!(capacity["parent_only_formal_child_rejected_count"], 2_160);
    assert_eq!(
        capacity["survivor_accepted_set_commitment"],
        "sha256:f5ab7f079ad943d65a74881eb59c7bb46385e1c437ca8ab036bb071dfa3874ac"
    );
    assert_eq!(
        capacity["parent_only_set_commitment"],
        "sha256:7e0e8780149f03ce85723408f7e3eff2cd684e8938896125cf8e34be9ac70b5e"
    );
    assert_eq!(
        capacity["parent_only_source_rejection_outcome_commitment"],
        "sha256:8617b56bdfa347f11f2c68b6a41f0992652f1e23e6d651017b17eb50169a9f39"
    );
    assert_eq!(
        capacity["parent_only_formal_rejection_outcome_commitment"],
        "sha256:9a6b489ed90960008aebbecdbcf0bc5cf1595b7a8206d179bbe898540dabf617"
    );
    assert_eq!(capacity["maximum_ast_node_count"], 6);
    assert_eq!(capacity["maximum_top_level_clauses"], 2);
    assert_eq!(capacity["executed_closure_status"], "NOT_RUN");
    assert!(capacity["formal_roots"].is_null());
    assert_eq!(capacity["target_or_split_modules_loaded"], false);
    assert_eq!(
        fields(&capacity),
        BTreeSet::from([
            "canonical_program_budget",
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
            "interpreted_as_complete_closure",
            "last_survivor_canonical_ast_hash",
            "last_survivor_canonical_cbor_hex",
            "maximum_ast_node_count",
            "maximum_top_level_clauses",
            "mixed_atom_count",
            "parent_dsl_version",
            "parent_freeze_version",
            "parent_only_formal_child_rejected_count",
            "parent_only_formal_child_rejection_counts",
            "parent_only_formal_rejection_outcome_commitment",
            "parent_only_node_count",
            "parent_only_parent_accepted_count",
            "parent_only_set_commitment",
            "parent_only_source_candidate_count",
            "parent_only_source_child_rejected_count",
            "parent_only_source_child_rejection_counts",
            "parent_only_source_rejection_outcome_commitment",
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
