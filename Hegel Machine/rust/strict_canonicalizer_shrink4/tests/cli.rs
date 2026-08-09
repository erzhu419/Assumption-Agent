use serde_json::Value;
use std::process::{Command, Output};

fn run(arguments: &[&str]) -> (Output, Value) {
    let output = Command::new(env!("CARGO_BIN_EXE_hegel-strict-canonicalizer-shrink4"))
        .args(arguments)
        .output()
        .expect("shrink-4 CLI must start");
    let report: Value = serde_json::from_slice(&output.stdout)
        .expect("shrink-4 CLI must emit exactly one JSON report");
    (output, report)
}

#[test]
fn source_cli_preserves_accept_and_reject_exit_contract() {
    let (accepted, report) = run(&["--ast-json", r#"["scalar_const",1]"#]);
    assert!(accepted.status.success());
    assert_eq!(report["status"], "ACCEPTED");
    assert_eq!(report["maximum_top_level_clauses"], 2);
    assert_eq!(report["target_or_split_modules_loaded"], false);

    let (rejected, report) = run(&[
        "--ast-json",
        r#"["top_level_AND",["context_flag","c0"],["context_flag","c1"],["context_flag","c2"]]"#,
    ]);
    assert_eq!(rejected.status.code(), Some(1));
    assert_eq!(report["status"], "REJECTED");
    assert_eq!(report["error_code"], "REJECT_STRUCTURAL_LIMIT");
    assert_eq!(report["maximum_top_level_clauses"], 2);
    assert_eq!(report["target_or_split_modules_loaded"], false);
}

#[test]
fn formal_cli_matches_sealed_and3_disposition() {
    let (rejected, report) = run(&[
        "--decode-cbor-hex",
        "8201820483830004008300040183000402",
    ]);
    assert_eq!(rejected.status.code(), Some(1));
    assert_eq!(report["boundary"], "FORMAL_CBOR");
    assert_eq!(report["generic_cbor_parse"], true);
    assert_eq!(report["error_code"], "REJECT_STRUCTURAL_LIMIT");
    assert_eq!(report["maximum_top_level_clauses"], 2);
}

#[test]
fn builtin_cli_reports_are_sealed_and_fail_closed() {
    let (golden_status, golden) = run(&["--golden-replay"]);
    assert!(golden_status.status.success());
    assert_eq!(golden["vector_count"], 22);
    assert_eq!(golden["passed_count"], 22);
    assert_eq!(
        golden["golden_vector_manifest_root"],
        "sha256:f84035e632bf5a655a9ebd636a0cafe7ab1097c45be87d4db944a0012f52aa90"
    );
    assert_eq!(
        golden["golden_outcome_root"],
        "sha256:c19341f08ac5f5759c2cdcb3681a37d958de362b81d02c184f7e2413dca18d7c"
    );
    assert_eq!(golden["execution_state"], "NOT_RUN");
    assert!(golden["formal_roots"].is_null());

    let (capacity_status, capacity) = run(&["--capacity-replay"]);
    assert!(capacity_status.status.success());
    assert_eq!(capacity["normalized_and2_count"], 2_160);
    assert_eq!(capacity["executed_closure_status"], "NOT_RUN");
    assert!(capacity["formal_roots"].is_null());
    assert_eq!(capacity["target_or_split_modules_loaded"], false);
}
