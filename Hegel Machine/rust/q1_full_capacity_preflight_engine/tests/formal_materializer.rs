use hegel_q1_full_capacity_preflight_engine::{
    formal_materializer_json_line, run_formal_archive_materializer_with_scratch, CapacityLimits,
    FORMAL_MATERIALIZER_ACTION_ID, FORMAL_MATERIALIZER_ACTOR_ID,
    FORMAL_MATERIALIZER_IMPLEMENTATION_ID, FORMAL_MATERIALIZER_STATUS,
    FORMAL_OUTPUT_BYTES_LIMIT, FORMAL_SCRATCH_BYTES_LIMIT,
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

const FIXED_POINT_DOMAIN: &[u8] = b"HEGEL/Q1/FIXED_POINT_RECORD/V1";
const SIGNATURE_MANIFEST_DOMAIN: &[u8] = b"HEGEL/Q1/SIGNATURE_ARCHIVE_MANIFEST/V1";
const CLOSURE_BUNDLE_DOMAIN: &[u8] = b"HEGEL/Q1/CLOSURE_BUNDLE/V1";
const FRAMED_BLOB_DOMAIN: &[u8] = b"HEGEL/Q1/FRAMED_BLOB/V1";

fn sha256(parts: &[&[u8]]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update(part);
    }
    hasher.finalize().into()
}

fn hex(value: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(value.len() * 2);
    for byte in value {
        output.push(DIGITS[(byte >> 4) as usize] as char);
        output.push(DIGITS[(byte & 15) as usize] as char);
    }
    output
}

fn prefixed(value: &[u8]) -> String {
    format!("sha256:{}", hex(value))
}

fn content_root(domain: &[u8], encoded_object: &[u8]) -> String {
    prefixed(&sha256(&[domain, &[0], encoded_object]))
}

fn rfc6962(records: &[Vec<u8>]) -> [u8; 32] {
    match records.len() {
        0 => sha256(&[b""]),
        1 => sha256(&[&[0], &records[0]]),
        length => {
            let split = 1_usize << ((length - 1).ilog2());
            let left = rfc6962(&records[..split]);
            let right = rfc6962(&records[split..]);
            sha256(&[&[1], &left, &right])
        }
    }
}

fn cbor_head(major: u8, value: usize) -> Vec<u8> {
    let prefix = major << 5;
    match value {
        0..=23 => vec![prefix | value as u8],
        24..=0xff => vec![prefix | 24, value as u8],
        0x100..=0xffff => {
            let mut output = vec![prefix | 25];
            output.extend_from_slice(&(value as u16).to_be_bytes());
            output
        }
        0x1_0000..=0xffff_ffff => {
            let mut output = vec![prefix | 26];
            output.extend_from_slice(&(value as u32).to_be_bytes());
            output
        }
        _ => {
            let mut output = vec![prefix | 27];
            output.extend_from_slice(&(value as u64).to_be_bytes());
            output
        }
    }
}

fn framed_blob_root(blob: &[u8]) -> String {
    prefixed(&sha256(&[
        FRAMED_BLOB_DOMAIN,
        &[0],
        &[0x81],
        &cbor_head(2, blob.len()),
        blob,
    ]))
}

fn read_frames(path: &Path) -> Vec<Vec<u8>> {
    let bytes = fs::read(path).unwrap();
    let mut offset = 0_usize;
    let mut records = Vec::new();
    while offset < bytes.len() {
        assert!(offset + 4 <= bytes.len(), "truncated frame in {}", path.display());
        let length = u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        assert!(
            offset + length <= bytes.len(),
            "truncated payload in {}",
            path.display()
        );
        records.push(bytes[offset..offset + length].to_vec());
        offset += length;
    }
    assert_eq!(offset, bytes.len());
    records
}

fn names(path: &Path) -> BTreeSet<String> {
    fs::read_dir(path)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .collect()
}

fn fresh_roots(label: &str) -> (PathBuf, PathBuf, PathBuf) {
    let parent = std::env::temp_dir().join(format!(
        "hegel-rust-formal-materializer-{label}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let output = parent.join("output");
    let scratch = parent.join("scratch");
    fs::create_dir(&parent).unwrap();
    fs::create_dir(&output).unwrap();
    fs::create_dir(&scratch).unwrap();
    (parent, output, scratch)
}

#[test]
fn q1_formal_schedule_matches_the_frozen_odd_sink_wire() {
    let [odd, sink] = CapacityLimits::q1_formal_schedule();
    assert_eq!(odd.maximum_ast_depth, 3);
    assert_eq!(odd.maximum_ast_node_count, 6);
    assert_eq!(odd.maximum_raw_operator_applications, 32_768);
    assert_eq!(odd.maximum_behavior_classes, 1_024);
    assert_eq!(odd.maximum_visible_frontier_points, 4_096);
    assert_eq!(odd.maximum_visible_frontier_points_per_class, 32);
    assert_eq!(odd.maximum_continuation_bank_points, 8_192);
    assert_eq!(odd.maximum_continuation_bank_points_per_class, 4_096);
    assert_eq!(odd.maximum_work_queue_points, 16_384);
    assert_eq!(odd.maximum_saturation_rounds, 7);
    assert_eq!(odd.maximum_wall_time_seconds, 3_600);
    assert_eq!(sink.maximum_ast_depth, 3);
    assert_eq!(sink.maximum_ast_node_count, 6);
    assert_eq!(sink.maximum_raw_operator_applications, 32_768);
    assert_eq!(sink.maximum_behavior_classes, 256);
    assert_eq!(sink.maximum_visible_frontier_points, 1_024);
    assert_eq!(sink.maximum_visible_frontier_points_per_class, 256);
    assert_eq!(sink.maximum_continuation_bank_points, 8_192);
    assert_eq!(sink.maximum_continuation_bank_points_per_class, 4_096);
    assert_eq!(sink.maximum_work_queue_points, 16_384);
    assert_eq!(sink.maximum_saturation_rounds, 7);
    assert_eq!(sink.maximum_wall_time_seconds, 3_600);
}

#[test]
fn formal_api_stops_on_a_first_signature_guard_without_writing_outputs() {
    let (parent, output, scratch) = fresh_roots("guard");
    let mut odd = CapacityLimits::q1_formal_odd();
    odd.maximum_raw_operator_applications = 1;
    let error = run_formal_archive_materializer_with_scratch(
        [odd, CapacityLimits::q1_formal_sink()],
        &format!("sha256:{}", "a".repeat(64)),
        &"b".repeat(40),
        &format!("sha256:{}", "c".repeat(64)),
        &format!("sha256:{}", "d".repeat(64)),
        &output,
        &scratch,
    )
    .unwrap_err();
    assert_eq!(error.code, "FAIL_FORMAL_MATERIALIZER_TRAVERSAL");
    assert!(names(&output).is_empty());
    assert!(names(&scratch).is_empty());
    fs::remove_dir_all(parent).unwrap();
}

#[test]
fn bounded_node3_retained_archive_replays_exact_frozen_roots_and_lengths() {
    let (parent, output, scratch) = fresh_roots("kat");
    let capsule = format!("sha256:{}", "a".repeat(64));
    let source = "b".repeat(40);
    let genesis = format!("sha256:{}", "c".repeat(64));
    let prestart = format!("sha256:{}", "d".repeat(64));
    let report = run_formal_archive_materializer_with_scratch(
        [CapacityLimits::bounded_for_test(3).unwrap(); 2],
        &capsule,
        &source,
        &genesis,
        &prestart,
        &output,
        &scratch,
    )
    .unwrap();

    assert_eq!(report.status, FORMAL_MATERIALIZER_STATUS);
    assert_eq!(report.actor_id, FORMAL_MATERIALIZER_ACTOR_ID);
    assert_eq!(report.implementation_id, FORMAL_MATERIALIZER_IMPLEMENTATION_ID);
    assert_eq!(report.action_id, FORMAL_MATERIALIZER_ACTION_ID);
    assert_eq!(report.run_genesis_root, genesis);
    assert_eq!(report.prestart_admission_receipt_root, prestart);
    assert_eq!(
        (report.q1_state, report.q1_gate_count, report.q1_gate_total, report.q1_gate_mask),
        ("NOT_RUN", 14, 20, 0x003fff)
    );
    assert!(report.q1_output_slots.iter().all(Option::is_none));
    assert!(report.formal_archive_materialized);
    assert!(!report.diagnostic_only);
    assert!(!report.target_truth_accessed);
    assert!(!report.split_accessed);
    assert!(!report.role_evaluation_performed);
    assert_eq!(report.actual_formal_terminal_live_scratch_bytes, 0);
    assert_eq!(report.formal_output_bytes_limit, FORMAL_OUTPUT_BYTES_LIMIT);
    assert_eq!(FORMAL_SCRATCH_BYTES_LIMIT, 134_217_728);
    assert_eq!(report.formal_scratch_bytes_limit, FORMAL_SCRATCH_BYTES_LIMIT);
    assert_eq!(
        report.formal_capacity_limits_by_signature,
        [
            hegel_q1_full_capacity_preflight_engine::FormalCapacityLimitRow {
                input_signature_id: 1,
                maximum_ast_depth: 3,
                maximum_ast_node_count: 3,
                maximum_raw_operator_applications: 4_294_967_295,
                maximum_behavior_classes: 16_777_216,
                maximum_visible_frontier_points: 33_554_432,
                maximum_visible_frontier_points_per_class: 65_536,
                maximum_continuation_bank_points: 67_108_864,
                maximum_continuation_bank_points_per_class: 65_536,
                maximum_work_queue_points: 67_108_864,
                maximum_saturation_rounds: 16,
                maximum_wall_time_seconds: 172_800,
            },
            hegel_q1_full_capacity_preflight_engine::FormalCapacityLimitRow {
                input_signature_id: 2,
                maximum_ast_depth: 3,
                maximum_ast_node_count: 3,
                maximum_raw_operator_applications: 4_294_967_295,
                maximum_behavior_classes: 16_777_216,
                maximum_visible_frontier_points: 33_554_432,
                maximum_visible_frontier_points_per_class: 65_536,
                maximum_continuation_bank_points: 67_108_864,
                maximum_continuation_bank_points_per_class: 65_536,
                maximum_work_queue_points: 67_108_864,
                maximum_saturation_rounds: 16,
                maximum_wall_time_seconds: 172_800,
            },
        ]
    );
    assert!(names(&scratch).is_empty());
    assert_eq!(report.formal_archive_payload_bytes, 550_112);
    assert_eq!(
        report.closure_bundle_root,
        "sha256:c4e1f2d950f66f5530ae834d2769e6c65decf7f50842fc66f158d11fc8aa9511"
    );

    assert_eq!(
        names(&output),
        BTreeSet::from([
            "closure-bundle.framed".to_owned(),
            "endpoint-result.json".to_owned(),
            "signature-01".to_owned(),
            "signature-02".to_owned(),
        ])
    );
    let frozen = hegel_q1_archive_projection_oracle::golden_node3().unwrap();
    let expected_singleton_roots = [
        (
            "sha256:0730c1a71c6e6f30ee67a00637ca7704787456c890db18e6b0403068a18f17d4",
            "sha256:c05244f21720c60877bc27bc9dabb553c09cec194d951cf71d0435e64729e5dd",
        ),
        (
            "sha256:09139e258bdd9c1a59b15b561d0d3109e76c1b6770ea9bababc14a208726b29a",
            "sha256:04b48abf1c6e9db9223d12a39f40fa7517f918d99505b122165cfde6d2ed5fa3",
        ),
    ];
    let mut retained_payload_bytes = fs::metadata(output.join("closure-bundle.framed"))
        .unwrap()
        .len();
    for ((partition, expected), singleton_roots) in report
        .partitions
        .iter()
        .zip(frozen.partitions.iter())
        .zip(expected_singleton_roots)
    {
        let signature_dir = output.join(format!(
            "signature-{:02}",
            partition.input_signature_id
        ));
        assert_eq!(
            names(&signature_dir),
            BTreeSet::from([
                "chunk-manifests.framed".to_owned(),
                "fixed-point.framed".to_owned(),
                "signature-manifest.framed".to_owned(),
                "stream-01".to_owned(),
                "stream-02".to_owned(),
                "stream-03".to_owned(),
                "stream-04".to_owned(),
            ])
        );
        assert_eq!(partition.fixed_point_record_root, singleton_roots.0);
        assert_eq!(partition.signature_manifest_root, singleton_roots.1);
        assert_eq!(partition.streams.len(), 4);
        for (stream, expected_stream) in partition
            .streams
            .iter()
            .zip(expected.archives.streams.iter())
        {
            assert_eq!(
                stream.archive_root,
                format!("sha256:{}", expected_stream.archive_root)
            );
            assert_eq!(stream.record_count, expected_stream.record_count);
            assert_eq!(stream.framed_bytes, expected_stream.framed_stream_bytes);
            assert_eq!(stream.chunk_count, expected_stream.chunk_count);
            assert_eq!(
                stream.chunk_manifest_subtree_root,
                format!("sha256:{}", expected_stream.chunk_manifest_archive_root)
            );
            let stream_dir = signature_dir.join(format!("stream-{:02}", stream.stream_kind_id));
            assert_eq!(names(&stream_dir).len() as u64, stream.chunk_count);
            let mut record_payloads = Vec::new();
            let mut observed_stream_bytes = 0_u64;
            for binding in &stream.chunk_bindings {
                let chunk_path = stream_dir.join(&binding.relative_path);
                let blob = fs::read(&chunk_path).unwrap();
                assert_eq!(blob.len() as u64, binding.framed_blob_bytes);
                assert_eq!(framed_blob_root(&blob), binding.framed_blob_hash);
                let frames = read_frames(&chunk_path);
                assert_eq!(frames.len() as u64, binding.record_count);
                record_payloads.extend(frames);
                observed_stream_bytes += blob.len() as u64;
            }
            assert_eq!(observed_stream_bytes, stream.framed_bytes);
            assert_eq!(record_payloads.len() as u64, stream.record_count);
            assert_eq!(prefixed(&rfc6962(&record_payloads)), stream.archive_root);
            retained_payload_bytes += observed_stream_bytes;
        }

        let manifest_path = signature_dir.join("chunk-manifests.framed");
        let manifest_records = read_frames(&manifest_path);
        assert_eq!(manifest_records.len() as u64, partition.chunk_manifest_count);
        assert_eq!(
            prefixed(&rfc6962(&manifest_records)),
            partition.chunk_manifest_archive_root
        );
        let fixed_point_path = signature_dir.join("fixed-point.framed");
        let fixed_point = read_frames(&fixed_point_path);
        assert_eq!(fixed_point.len(), 1);
        assert_eq!(
            content_root(FIXED_POINT_DOMAIN, &fixed_point[0]),
            partition.fixed_point_record_root
        );
        let manifest_object_path = signature_dir.join("signature-manifest.framed");
        let signature_manifest = read_frames(&manifest_object_path);
        assert_eq!(signature_manifest.len(), 1);
        assert_eq!(
            content_root(SIGNATURE_MANIFEST_DOMAIN, &signature_manifest[0]),
            partition.signature_manifest_root
        );
        retained_payload_bytes += fs::metadata(manifest_path).unwrap().len();
        retained_payload_bytes += fs::metadata(fixed_point_path).unwrap().len();
        retained_payload_bytes += fs::metadata(manifest_object_path).unwrap().len();
    }
    let closure = read_frames(&output.join("closure-bundle.framed"));
    assert_eq!(closure.len(), 1);
    assert_eq!(
        content_root(CLOSURE_BUNDLE_DOMAIN, &closure[0]),
        report.closure_bundle_root
    );
    assert_eq!(retained_payload_bytes, report.formal_archive_payload_bytes);
    assert_eq!(
        fs::read(output.join("endpoint-result.json")).unwrap(),
        formal_materializer_json_line(&report).unwrap()
    );
    let endpoint: Value =
        serde_json::from_slice(&fs::read(output.join("endpoint-result.json")).unwrap()).unwrap();
    assert_eq!(
        endpoint
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>(),
        [
            "action_id",
            "actor_id",
            "actual_formal_peak_scratch_bytes",
            "actual_formal_terminal_live_scratch_bytes",
            "capsule_digest",
            "closure_bundle_root",
            "closure_semantics_version",
            "diagnostic_only",
            "dsl_version",
            "formal_archive_materialized",
            "formal_archive_payload_bytes",
            "formal_capacity_limits_by_signature",
            "formal_output_bytes_limit",
            "formal_scratch_bytes_limit",
            "full_v16_structural_limits_applied",
            "implementation_id",
            "maximum_ast_depth",
            "maximum_ast_node_count",
            "partitions",
            "prestart_admission_receipt_root",
            "q1_gate_count",
            "q1_gate_mask",
            "q1_gate_total",
            "q1_output_slots",
            "q1_state",
            "role_evaluation_performed",
            "run_genesis_root",
            "schema_version",
            "source_commit",
            "split_accessed",
            "status",
            "target_truth_accessed",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect(),
    );
    for partition in endpoint["partitions"].as_array().unwrap() {
        assert_eq!(
            partition
                .as_object()
                .unwrap()
                .keys()
                .cloned()
                .collect::<BTreeSet<_>>(),
            [
                "chunk_manifest_archive_root",
                "chunk_manifest_count",
                "fixed_point_record_root",
                "input_signature_id",
                "input_signature_name",
                "partition_external_sort_root",
                "partition_stream_commitment_root",
                "projected_partition_payload_bytes",
                "same_run_record_set_root",
                "saturation_state_root",
                "signature_manifest_root",
                "streams",
                "universe_root",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect(),
        );
        for stream in partition["streams"].as_array().unwrap() {
            assert_eq!(
                stream
                    .as_object()
                    .unwrap()
                    .keys()
                    .cloned()
                    .collect::<BTreeSet<_>>(),
                [
                    "archive_root",
                    "canonical_record_payload_bytes",
                    "chunk_count",
                    "chunk_manifest_subtree_root",
                    "counting_discard_commitment",
                    "external_sort_projection_root",
                    "framed_bytes",
                    "record_count",
                    "stream_kind",
                    "stream_kind_id",
                ]
                .into_iter()
                .map(str::to_owned)
                .collect(),
            );
        }
    }

    fs::remove_dir_all(&parent).unwrap();
}

#[test]
fn formal_api_rejects_missing_prestart_binding_before_traversal_or_write() {
    let (parent, output, scratch) = fresh_roots("admission");
    let error = run_formal_archive_materializer_with_scratch(
        [CapacityLimits::bounded_for_test(3).unwrap(); 2],
        &format!("sha256:{}", "a".repeat(64)),
        &"b".repeat(40),
        &format!("sha256:{}", "c".repeat(64)),
        "null",
        &output,
        &scratch,
    )
    .unwrap_err();
    assert_eq!(error.code, "REJECT_FORMAL_MATERIALIZER_ADMISSION");
    assert!(names(&output).is_empty());
    assert!(names(&scratch).is_empty());
    fs::remove_dir_all(&parent).unwrap();
}

#[test]
fn formal_cli_rejects_diagnostic_or_zero_gate_actions_without_running() {
    let binary = env!("CARGO_BIN_EXE_hegel-q1-formal-archive-materializer");
    for action in ["full-node6-capacity-preflight-v1", "bounded-node3-golden-v1"] {
        let (parent, output_dir, scratch_dir) = fresh_roots(action);
        let output = Command::new(binary)
            .args([
                "--action",
                action,
                "--capsule-digest",
                &format!("sha256:{}", "a".repeat(64)),
                "--source-commit",
                &"b".repeat(40),
                "--run-genesis-root",
                &format!("sha256:{}", "c".repeat(64)),
                "--prestart-admission-receipt-root",
                &format!("sha256:{}", "d".repeat(64)),
                "--output-dir",
                output_dir.to_str().unwrap(),
                "--scratch-dir",
                scratch_dir.to_str().unwrap(),
            ])
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(1));
        assert!(output.stderr.is_empty());
        let value: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(
            value["error_code"],
            "Q1_FORMAL_MATERIALIZER_ACTION_NOT_ADMITTED"
        );
        assert_eq!(value["formal_archive_materialized"], false);
        assert_eq!(value["target_truth_accessed"], false);
        assert!(names(&output_dir).is_empty());
        assert!(names(&scratch_dir).is_empty());
        fs::remove_dir_all(parent).unwrap();
    }
}
