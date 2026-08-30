use hegel_q1_full_capacity_preflight_engine::{
    canonical_json_line, run_capacity_preflight, run_capacity_preflight_with_scratch,
    run_partition_capacity_preflight, sidecar_payloads, CapacityLimits,
    LOCAL_PROTOTYPE_SUBSET_TRAVERSAL_CLOSED, PREFLIGHT_CAPACITY_GUARD_HIT, SIDECAR_NAMES,
};
use serde_json::Value;
use std::process::Command;

#[test]
fn bounded_node3_matches_frozen_python_and_rust_guidance() {
    let limits = CapacityLimits::bounded_for_test(3).unwrap();
    let odd = run_partition_capacity_preflight(1, limits).unwrap();
    assert_eq!(odd.terminal_status, LOCAL_PROTOTYPE_SUBSET_TRAVERSAL_CLOSED);
    assert_eq!(odd.behavior_class_count, 40);
    assert_eq!(odd.visible_frontier_point_count, 59);
    assert_eq!(odd.continuation_bank_point_count, 110);
    assert_eq!(odd.raw_operator_application_count, 1_048);
    assert_eq!(odd.peak_saturation_round_count, 4);
    assert_eq!(odd.depth_barriers[1].continuation_bank_mutation_count, 78);

    let sink = run_partition_capacity_preflight(2, limits).unwrap();
    assert_eq!(sink.terminal_status, LOCAL_PROTOTYPE_SUBSET_TRAVERSAL_CLOSED);
    assert_eq!(sink.behavior_class_count, 28);
    assert_eq!(sink.visible_frontier_point_count, 84);
    assert_eq!(sink.continuation_bank_point_count, 144);
    assert_eq!(sink.raw_operator_application_count, 1_101);
    assert_eq!(sink.peak_saturation_round_count, 4);
    assert_eq!(sink.depth_barriers[1].continuation_bank_mutation_count, 113);
    assert_eq!(
        odd.diagnostic_state_projection
            .as_ref()
            .unwrap()
            .combined_state_projection_sha256,
        "c19d3ccf996292490f0243627f380ded5a0aa743041ee9a50a163c9fc70adbb7"
    );
    assert_eq!(
        odd.diagnostic_state_projection
            .as_ref()
            .unwrap()
            .streams
            .iter()
            .map(|row| (
                row.stream_kind,
                row.record_count,
                row.payload_bytes,
                row.framed_bytes,
                row.counting_discard_sha256.as_str(),
            ))
            .collect::<Vec<_>>(),
        vec![
            ("PROGRAM", 110, 13_724, 14_604, "e78f0b56ecdce07fbff2435068e91d74fdf9984d6f638c75767dc95c16a05caf"),
            ("COHORT", 86, 15_904, 16_592, "4f62693738c7a10240454aa99919337870560a09034f4a5031a9f91ce631c0e0"),
            ("CLASS", 40, 133_784, 134_104, "4ae8ffd5dcf86bcb6f77bb1c3a6ba2b59108e0287e3a4a2b1152cacb5cf523d8"),
            ("COVERAGE", 846, 30_456, 37_224, "be0e67ea10a388afdb3126d47a2fc0d759707347a529da400f5faba5cb449082"),
        ]
    );
    assert_eq!(
        sink.diagnostic_state_projection
            .as_ref()
            .unwrap()
            .combined_state_projection_sha256,
        "0d9b8506d64051c8d89a25975c765630e30b53eeda1d6f5f32529933af246e44"
    );
    assert_eq!(
        sink.diagnostic_state_projection
            .as_ref()
            .unwrap()
            .streams
            .iter()
            .map(|row| (
                row.stream_kind,
                row.record_count,
                row.payload_bytes,
                row.framed_bytes,
                row.counting_discard_sha256.as_str(),
            ))
            .collect::<Vec<_>>(),
        vec![
            ("PROGRAM", 144, 18_038, 19_190, "8839a25d1045130fc8fe80009d208fec559d7277c595053db953b9a25ee81986"),
            ("COHORT", 112, 20_736, 21_632, "2824952e491112e2a4ff68bf89cadd88089039256a8695b24ed6568953395240"),
            ("CLASS", 28, 22_898, 23_122, "b7a4cc9370a9049ce00cc94807a6138540b8746c4bb56c07fe93b4867b85ee7f"),
            ("COVERAGE", 846, 30_456, 37_224, "4d4b7def075077179568181a8bc428ab6fd3fd3ac41715e2cf9c147877f3f975"),
        ]
    );

    let frozen = hegel_q1_archive_projection_oracle::golden_node3().unwrap();
    for (actual, expected) in [odd, sink].iter().zip(frozen.partitions.iter()) {
        let actual_projection = actual
            .formal_wire_projection
            .as_ref()
            .expect("closed bounded traversal must produce exact frozen-v1 projection");
        assert_eq!(
            actual_projection
                .streams
                .iter()
                .map(|row| (
                    row.stream_kind_id,
                    row.stream_kind.as_str(),
                    row.record_count,
                    row.archive_root.strip_prefix("sha256:").unwrap(),
                    row.framed_bytes,
                    row.chunk_count,
                    row.chunk_manifest_subtree_root
                        .strip_prefix("sha256:")
                        .unwrap(),
                    row.counting_discard_commitment
                        .strip_prefix("sha256:")
                        .unwrap(),
                    row.external_sort_projection_root
                        .strip_prefix("sha256:")
                        .unwrap(),
                ))
                .collect::<Vec<_>>(),
            expected
                .archives
                .streams
                .iter()
                .map(|row| (
                    row.stream_kind_id,
                    row.stream_kind_name.as_str(),
                    row.record_count,
                    row.archive_root.as_str(),
                    row.framed_stream_bytes,
                    row.chunk_count,
                    row.chunk_manifest_archive_root.as_str(),
                    row.projected_stream_commitment.as_str(),
                    row.external_sort.diagnostic_root.as_str(),
                ))
                .collect::<Vec<_>>(),
        );
        assert_eq!(
            actual_projection.projected_record_stream_bytes,
            expected
                .archives
                .streams
                .iter()
                .map(|row| row.framed_stream_bytes)
                .sum::<u64>(),
        );
        assert_eq!(
            actual_projection.projected_partition_payload_bytes,
            actual_projection.projected_record_stream_bytes
                + actual_projection.projected_chunk_manifest_stream_bytes
                + actual_projection.projected_fixed_point_frame_bytes
                + actual_projection.projected_signature_manifest_frame_bytes,
        );
        assert_eq!(
            actual_projection.projected_peak_formal_scratch_bytes,
            expected
                .archives
                .streams
                .iter()
                .map(|row| row.external_sort.charged_scratch_high_water_bytes)
                .max()
                .unwrap(),
        );
    }
}

#[test]
fn bounded_node4_matches_preregistered_nonformal_guidance() {
    let limits = CapacityLimits::bounded_for_test(4).unwrap();
    let odd = run_partition_capacity_preflight(1, limits).unwrap();
    assert_eq!(
        (
            odd.behavior_class_count,
            odd.visible_frontier_point_count,
            odd.continuation_bank_point_count,
            odd.raw_operator_application_count,
        ),
        (107, 154, 478, 1_844),
    );
    assert_eq!(odd.peak_visible_frontier_points_per_class, 13);
    assert_eq!(odd.maximum_frontier_points_per_class, 5);
    let sink = run_partition_capacity_preflight(2, limits).unwrap();
    assert_eq!(
        (
            sink.behavior_class_count,
            sink.visible_frontier_point_count,
            sink.continuation_bank_point_count,
            sink.raw_operator_application_count,
        ),
        (47, 138, 466, 2_108),
    );
}

#[test]
fn bounded_node3_exact_frozen_resource_frame_kat() {
    let report = run_capacity_preflight(
        CapacityLimits::bounded_for_test(3).unwrap(),
        &format!("sha256:{}", "a".repeat(64)),
        &"b".repeat(40),
    )
    .unwrap();
    let rows = report
        .partitions
        .iter()
        .map(|partition| {
            let projection = partition.formal_wire_projection.as_ref().unwrap();
            (
                projection.projected_record_stream_bytes,
                projection.projected_chunk_manifest_stream_bytes,
                projection.projected_fixed_point_frame_bytes,
                projection.projected_signature_manifest_frame_bytes,
                projection.projected_partition_payload_bytes,
                projection.projected_peak_formal_scratch_bytes,
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        rows,
        vec![
            (294_548, 897, 268, 699, 296_412, 167_936),
            (251_526, 895, 269, 696, 253_386, 167_936),
        ],
    );
    assert_eq!(
        (
            report.projected_closure_bundle_frame_bytes,
            report.projected_formal_archive_payload_bytes,
            report.projected_formal_total_output_bytes,
            report.projected_peak_formal_scratch_bytes,
        ),
        (Some(314), Some(550_112), Some(1_598_688), Some(167_936)),
    );
}

#[test]
fn report_keeps_q1_unstarted_and_projects_without_materializing_archives() {
    let report = run_capacity_preflight(
        CapacityLimits::bounded_for_test(3).unwrap(),
        &format!("sha256:{}", "a".repeat(64)),
        &"b".repeat(40),
    )
    .unwrap();
    assert_eq!(report.q1_state, "NOT_RUN");
    assert_eq!(report.q1_gate_count, 0);
    assert_eq!(report.q1_gate_mask, 0);
    assert!(report.q1_formal_roots.is_none());
    assert!(report.q1_output_slots.iter().all(Option::is_none));
    assert!(!report.q1_execution_started);
    assert_eq!(report.actual_preflight_terminal_live_scratch_bytes, 0);
    assert_eq!(
        report.projected_formal_total_output_bytes,
        report
            .projected_formal_archive_payload_bytes
            .map(|value| value + 1_048_576),
    );
    assert_eq!(
        report.projected_formal_archive_payload_bytes,
        Some(
            report
                .partitions
                .iter()
                .map(|partition| {
                    partition
                        .formal_wire_projection
                        .as_ref()
                        .unwrap()
                        .projected_partition_payload_bytes
                })
                .sum::<u64>()
                + report.projected_closure_bundle_frame_bytes.unwrap(),
        ),
    );
    assert!(report.partitions.iter().all(|partition| {
        partition
            .diagnostic_state_projection
            .as_ref()
            .is_some_and(|projection| {
                projection.diagnostic_only && !projection.formal_archive_materialized
            })
            && partition.formal_roots.is_none()
            && !partition.target_truth_accessed
    }));
}

#[test]
fn sidecar_registry_is_exact_and_manifest_binds_four_payloads() {
    let report = run_capacity_preflight(
        CapacityLimits::bounded_for_test(3).unwrap(),
        &format!("sha256:{}", "a".repeat(64)),
        &"b".repeat(40),
    )
    .unwrap();
    let payloads = sidecar_payloads(&report).unwrap();
    assert_eq!(
        payloads.keys().copied().collect::<std::collections::BTreeSet<_>>(),
        SIDECAR_NAMES.into_iter().collect(),
    );
    for payload in payloads.values() {
        assert!(payload.ends_with(b"\n"));
        assert_eq!(payload.iter().filter(|byte| **byte == b'\n').count(), 1);
        let _: Value = serde_json::from_slice(payload).unwrap();
    }
    let manifest: Value =
        serde_json::from_slice(payloads["endpoint-manifest-v1.json"].as_slice()).unwrap();
    assert_eq!(manifest["actor_id"], "RUST_ENDPOINT");
    assert_eq!(manifest["rows"].as_array().unwrap().len(), 4);
    assert_eq!(manifest["q1_state"], "NOT_RUN");
    let rows = manifest["rows"].as_array().unwrap();
    assert_eq!(
        rows.iter()
            .map(|row| row["filename"].as_str().unwrap())
            .collect::<Vec<_>>(),
        vec![
            "coverage-summary-v1.json",
            "diagnostic-summary-v1.json",
            "fixed-point-summary-v1.json",
            "projection-summary-v1.json",
        ],
    );
    for row in rows {
        let filename = row["filename"].as_str().unwrap();
        assert_eq!(
            row["byte_count"].as_u64().unwrap(),
            payloads[filename].len() as u64,
        );
    }
    let stdout = canonical_json_line(&report).unwrap();
    let declared_nonmanifest_bytes = rows
        .iter()
        .map(|row| row["byte_count"].as_u64().unwrap())
        .sum::<u64>();
    let exact_diagnostic_output_bytes = stdout.len() as u64
        + declared_nonmanifest_bytes
        + payloads["endpoint-manifest-v1.json"].len() as u64;
    assert_eq!(
        exact_diagnostic_output_bytes,
        stdout.len() as u64 + payloads.values().map(|payload| payload.len() as u64).sum::<u64>(),
    );
}

#[test]
fn first_raw_guard_rolls_back_the_unaccepted_event() {
    let mut limits = CapacityLimits::bounded_for_test(3).unwrap();
    limits.maximum_raw_operator_applications = 810;
    let result = run_partition_capacity_preflight(1, limits).unwrap();
    assert_eq!(result.terminal_status, PREFLIGHT_CAPACITY_GUARD_HIT);
    assert_eq!(result.resource_guard_id, Some(1));
    assert_eq!(result.resource_guard_name.as_deref(), Some("RAW_OPERATOR_APPLICATIONS"));
    assert_eq!(result.raw_operator_application_count, 810);
    assert_eq!(result.depth_barriers.len(), 1);
}

#[test]
fn aggregate_stops_after_the_first_registered_guard_prefix() {
    let mut limits = CapacityLimits::bounded_for_test(3).unwrap();
    limits.maximum_raw_operator_applications = 810;
    let report = run_capacity_preflight(
        limits,
        &format!("sha256:{}", "a".repeat(64)),
        &"b".repeat(40),
    )
    .unwrap();
    assert_eq!(report.terminal_status, PREFLIGHT_CAPACITY_GUARD_HIT);
    assert_eq!(report.actual_preflight_peak_scratch_bytes, 0);
    assert_eq!(report.actual_preflight_terminal_live_scratch_bytes, 0);
    assert!(report.projected_closure_bundle_frame_bytes.is_none());
    assert!(report.projected_formal_archive_payload_bytes.is_none());
    assert!(report.projected_formal_total_output_bytes.is_none());
    assert!(report.projected_peak_formal_scratch_bytes.is_none());
    assert_eq!(report.partitions.len(), 1);
    assert_eq!(report.partitions[0].input_signature_id, 1);
    assert_eq!(
        report.partitions[0].resource_guard_name.as_deref(),
        Some("RAW_OPERATOR_APPLICATIONS")
    );
    assert!(report.partitions[0].formal_wire_projection.is_none());
    assert!(report.partitions[0].diagnostic_state_projection.is_none());
    assert!(report.partitions[0].coverage.is_none());
    assert!(report.partitions[0].fixed_point.is_none());
    assert!(report.partitions[0].same_run_record_set_root.is_none());
}

#[test]
fn aggregate_second_signature_guard_clears_the_entire_prefix_material() {
    let mut limits = CapacityLimits::bounded_for_test(3).unwrap();
    limits.maximum_raw_operator_applications = 1_048;
    let report = run_capacity_preflight(
        limits,
        &format!("sha256:{}", "a".repeat(64)),
        &"b".repeat(40),
    )
    .unwrap();

    assert_eq!(report.terminal_status, PREFLIGHT_CAPACITY_GUARD_HIT);
    assert_eq!(report.actual_preflight_peak_scratch_bytes, 0);
    assert_eq!(report.actual_preflight_terminal_live_scratch_bytes, 0);
    assert!(report.projected_closure_bundle_frame_bytes.is_none());
    assert!(report.projected_formal_archive_payload_bytes.is_none());
    assert!(report.projected_formal_total_output_bytes.is_none());
    assert!(report.projected_peak_formal_scratch_bytes.is_none());
    assert_eq!(report.partitions.len(), 2);
    assert_eq!(
        report.partitions[0].terminal_status,
        LOCAL_PROTOTYPE_SUBSET_TRAVERSAL_CLOSED
    );
    assert_eq!(report.partitions[1].terminal_status, PREFLIGHT_CAPACITY_GUARD_HIT);
    assert_eq!(
        report.partitions[1].resource_guard_name.as_deref(),
        Some("RAW_OPERATOR_APPLICATIONS")
    );
    for partition in &report.partitions {
        assert!(partition.fixed_point.is_none());
        assert!(partition.coverage.is_none());
        assert!(partition.diagnostic_state_projection.is_none());
        assert!(partition.formal_wire_projection.is_none());
        assert!(partition.same_run_record_set_root.is_none());
    }

    let payloads = sidecar_payloads(&report).unwrap();
    for filename in [
        "projection-summary-v1.json",
        "coverage-summary-v1.json",
        "fixed-point-summary-v1.json",
    ] {
        let value: Value = serde_json::from_slice(payloads[filename].as_slice()).unwrap();
        for partition in value["partitions"].as_array().unwrap() {
            for field in [
                "diagnostic_state_projection",
                "formal_wire_projection",
                "coverage",
                "fixed_point",
                "same_run_record_set_root",
            ] {
                if partition.get(field).is_some() {
                    assert!(partition[field].is_null(), "{filename}:{field}");
                }
            }
        }
    }
}

#[test]
fn successful_projection_removes_its_transient_scratch_tree() {
    let parent = std::env::temp_dir().join(format!(
        "hegel-rust-full-preflight-test-{}",
        std::process::id()
    ));
    std::fs::create_dir(&parent).unwrap();
    let report = run_capacity_preflight_with_scratch(
        CapacityLimits::bounded_for_test(3).unwrap(),
        &format!("sha256:{}", "a".repeat(64)),
        &"b".repeat(40),
        &parent,
    )
    .unwrap();
    assert_eq!(report.partitions.len(), 2);
    assert_eq!(report.actual_preflight_terminal_live_scratch_bytes, 0);
    assert!(std::fs::read_dir(&parent).unwrap().next().is_none());
    std::fs::remove_dir(&parent).unwrap();
}

#[test]
fn production_cli_rejects_every_nonfull_or_ambiguous_invocation() {
    let binary = env!("CARGO_BIN_EXE_hegel-q1-full-capacity-preflight-engine");
    for arguments in [vec![], vec!["--action", "bounded-node3"], vec!["--node-limit", "3"]] {
        let output = Command::new(binary).args(arguments).output().unwrap();
        assert_eq!(output.status.code(), Some(1));
        assert!(output.stderr.is_empty());
        assert_eq!(output.stdout.iter().filter(|byte| **byte == b'\n').count(), 1);
        let value: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(value["error_code"], "Q1_FULL_CAPACITY_ACTION_NOT_ADMITTED");
        assert_eq!(value["q1_state"], "NOT_RUN");
        assert_eq!(value["q1_gate_count"], 0);
        assert!(value["q1_formal_roots"].is_null());
        assert_eq!(value["target_truth_accessed"], false);
    }

    let directory = std::env::temp_dir().join(format!(
        "hegel-rust-full-preflight-cli-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&directory).unwrap();
    let output = Command::new(binary)
        .args([
            "--action",
            "full-node6-capacity-preflight-v1",
            "--capsule-digest",
            &format!("sha256:{}", "a".repeat(64)),
            "--source-commit",
            &"b".repeat(40),
            "--output-dir",
            directory.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    let value: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["error_code"], "Q1_FULL_CAPACITY_ACTION_NOT_ADMITTED");

    let output = Command::new(binary)
        .args([
            "--action",
            "full-node6-capacity-preflight-v1",
            "--capsule-digest",
            &format!("sha256:{}", "a".repeat(64)),
            "--source-commit",
            &"b".repeat(40),
            "--output-dir",
            directory.to_str().unwrap(),
            "--scratch-dir",
            directory.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    let value: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["error_code"], "REJECT_PREFLIGHT_DIRECTORY_ALIAS");
    std::fs::remove_dir(directory).unwrap();
}
