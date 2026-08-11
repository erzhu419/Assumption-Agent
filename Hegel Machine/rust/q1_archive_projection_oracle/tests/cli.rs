use hegel_q1_archive_projection_oracle::{
    canonical_json_line, ACTOR_ACTION_ID, ACTOR_IMPLEMENTATION_ID, ACTOR_SCHEMA_VERSION,
    ACTOR_STATUS, OUTPUT_RELATIVE_PATHS, SOURCE_IDENTITY_ENV,
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
use std::os::unix::fs::{symlink, MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

static SCRATCH_COUNTER: AtomicU64 = AtomicU64::new(0);

struct ScratchTree(PathBuf);

impl ScratchTree {
    fn new(label: &str) -> Self {
        let serial = SCRATCH_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "hegel-q05b-rust-actor-{label}-{}-{serial}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        Self(fs::canonicalize(path).unwrap())
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for ScratchTree {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn isolated_binary(root: &Path) -> PathBuf {
    let target = root.join("hegel-q1-archive-projection-oracle");
    fs::copy(
        env!("CARGO_BIN_EXE_hegel-q1-archive-projection-oracle"),
        &target,
    )
    .unwrap();
    fs::set_permissions(&target, fs::Permissions::from_mode(0o555)).unwrap();
    target
}

fn run(binary: &Path, cwd: &Path, arguments: &[&str]) -> Output {
    let mut command = Command::new(binary);
    command.current_dir(cwd).env_clear().args(arguments);
    command.output().unwrap()
}

fn strict_json(stdout: &[u8]) -> Value {
    assert!(stdout.ends_with(b"\n"));
    assert_eq!(stdout.iter().filter(|byte| **byte == b'\n').count(), 1);
    let value: Value = serde_json::from_slice(stdout).unwrap();
    assert_eq!(stdout, canonical_json_line(&value).unwrap());
    value
}

fn assert_error(output: &Output, expected_code: &str) -> Value {
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stderr.is_empty());
    let value = strict_json(&output.stdout);
    assert_eq!(value["error_code"], expected_code);
    assert_eq!(value["q1_state"], "NOT_RUN");
    assert_eq!(value["q1_gate_count"], 0);
    assert_eq!(value["q1_gate_mask"], 0);
    assert!(value["q1_formal_roots"].is_null());
    assert_eq!(value["full_node6_executed"], false);
    assert_eq!(value["sidecar_set_complete"], false);
    value
}

fn sha256_hex(payload: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(payload);
    format!("{:x}", digest.finalize())
}

fn expected_runtime_identity(binary: &Path) -> String {
    let payload = fs::read(binary).unwrap();
    let mut digest = Sha256::new();
    digest.update(b"HEGEL/Q05B/RUST_RUNTIME_IDENTITY/V1\x00");
    digest.update((payload.len() as u64).to_be_bytes());
    digest.update(&payload);
    format!("{:x}", digest.finalize())
}

fn output_files(output: &Path) -> BTreeSet<String> {
    let mut files = BTreeSet::new();
    for child in fs::read_dir(output).unwrap() {
        let child = child.unwrap();
        let kind = child.file_type().unwrap();
        assert!(kind.is_dir());
        assert!(!kind.is_symlink());
        for file in fs::read_dir(child.path()).unwrap() {
            let file = file.unwrap();
            let kind = file.file_type().unwrap();
            assert!(kind.is_file());
            assert!(!kind.is_symlink());
            files.insert(
                file.path()
                    .strip_prefix(output)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/"),
            );
        }
    }
    files
}

#[test]
fn formal_actor_publishes_exact_read_only_sidecars_and_shared_envelope() {
    let scratch = ScratchTree::new("success");
    let binary = isolated_binary(scratch.path());
    let output_dir = scratch.path().join("output");
    fs::create_dir(&output_dir).unwrap();
    let output_path = output_dir.to_str().unwrap();
    let result = run(
        &binary,
        scratch.path(),
        &["--action", ACTOR_ACTION_ID, "--output-dir", output_path],
    );
    assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stdout));
    assert!(result.stderr.is_empty());
    assert!(result.stdout.len() < 4096);
    let envelope = strict_json(&result.stdout);
    let expected_fields = BTreeSet::from([
        "action_id",
        "actor_id",
        "file_count",
        "implementation_id",
        "neutral_manifest_length",
        "neutral_manifest_raw_sha256",
        "neutral_manifest_relative_path",
        "neutral_manifest_root",
        "q1_formal_roots",
        "q1_gate_count",
        "q1_gate_mask",
        "q1_output_slots",
        "q1_state",
        "runtime_identity_sha256",
        "schema_version",
        "sidecar_manifest_length",
        "sidecar_manifest_raw_sha256",
        "sidecar_manifest_relative_path",
        "sidecar_manifest_root",
        "source_identity_sha256",
        "status",
    ]);
    let actual_fields = envelope
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    assert_eq!(actual_fields, expected_fields);
    assert_eq!(envelope["actor_id"], "RUST_ENDPOINT");
    assert_eq!(envelope["implementation_id"], ACTOR_IMPLEMENTATION_ID);
    assert_eq!(envelope["schema_version"], ACTOR_SCHEMA_VERSION);
    assert_eq!(envelope["action_id"], ACTOR_ACTION_ID);
    assert_eq!(envelope["status"], ACTOR_STATUS);
    assert_eq!(envelope["file_count"], 5);
    assert_eq!(envelope["q1_state"], "NOT_RUN");
    assert_eq!(envelope["q1_gate_count"], 0);
    assert_eq!(envelope["q1_gate_mask"], 0);
    assert!(envelope["q1_formal_roots"].is_null());
    assert_eq!(envelope["q1_output_slots"], Value::Array(vec![Value::Null; 8]));
    assert_eq!(
        envelope["source_identity_sha256"],
        option_env!("HEGEL_Q05B_RUST_SOURCE_IDENTITY_SHA256")
            .unwrap_or_else(|| panic!("{SOURCE_IDENTITY_ENV} must be set for the build"))
    );
    assert_eq!(
        envelope["runtime_identity_sha256"],
        expected_runtime_identity(&binary)
    );
    assert_eq!(
        envelope["neutral_manifest_relative_path"],
        OUTPUT_RELATIVE_PATHS[4]
    );
    assert_eq!(
        envelope["sidecar_manifest_relative_path"],
        OUTPUT_RELATIVE_PATHS[3]
    );

    let expected_paths = OUTPUT_RELATIVE_PATHS
        .into_iter()
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
    assert_eq!(output_files(&output_dir), expected_paths);
    for relative in OUTPUT_RELATIVE_PATHS {
        let metadata = fs::symlink_metadata(output_dir.join(relative)).unwrap();
        assert!(metadata.is_file());
        assert!(!metadata.file_type().is_symlink());
        assert_eq!(metadata.mode() & 0o777, 0o444);
    }
    let neutral = fs::read(output_dir.join(OUTPUT_RELATIVE_PATHS[4])).unwrap();
    let sidecar = fs::read(output_dir.join(OUTPUT_RELATIVE_PATHS[3])).unwrap();
    assert_eq!(neutral.len(), 4_134);
    assert_eq!(sidecar.len(), 552);
    assert_eq!(
        sha256_hex(&neutral),
        "7fd529708a068e2fa1a8d17f5cc81a41420db944120f4f1591f73e1c67f4cc05"
    );
    assert_eq!(
        sha256_hex(&sidecar),
        "318b8fb9e9ba3ce881057742d59bf43314c89891cbc37e4824349ac3f72d4ba3"
    );
    assert_eq!(
        envelope["neutral_manifest_root"],
        "cbc22f6a9dc91589f77aa1564eb40d688c45ee3aa6af5a66d777ffe08a086b15"
    );
    assert_eq!(
        envelope["sidecar_manifest_root"],
        "1d68a6fe330f3bfe581ef37933f64d2258e1043079dae15c85607836d99ea59d"
    );
    assert!(!neutral.windows(64).any(|window| {
        window == envelope["source_identity_sha256"].as_str().unwrap().as_bytes()
            || window == envelope["runtime_identity_sha256"].as_str().unwrap().as_bytes()
    }));

    let duplicate = run(
        &binary,
        scratch.path(),
        &["--action", ACTOR_ACTION_ID, "--output-dir", output_path],
    );
    assert_error(&duplicate, "FAIL_Q1_PROJECTION_OUTPUT_DIR");
    assert_eq!(output_files(&output_dir), expected_paths);
}

#[test]
fn illegal_old_node6_duplicate_nan_and_output_directory_shapes_fail_closed() {
    let scratch = ScratchTree::new("reject");
    let binary = PathBuf::from(env!("CARGO_BIN_EXE_hegel-q1-archive-projection-oracle"));
    for arguments in [
        vec![],
        vec!["--golden-node3"],
        vec!["--action", "bounded-node6-golden-v1"],
        vec!["--action", "NaN"],
        vec![
            "--action",
            ACTOR_ACTION_ID,
            "--action",
            ACTOR_ACTION_ID,
            "--output-dir",
            "/tmp/never-used",
        ],
        vec!["--node6"],
    ] {
        let result = run(&binary, scratch.path(), &arguments);
        assert_error(&result, "Q1_PROJECTION_ACTION_NOT_ADMITTED");
    }

    let relative = run(
        &binary,
        scratch.path(),
        &[
            "--action",
            ACTOR_ACTION_ID,
            "--output-dir",
            "relative-output",
        ],
    );
    assert_error(&relative, "FAIL_Q1_PROJECTION_OUTPUT_DIR");

    let nonempty = scratch.path().join("nonempty");
    fs::create_dir(&nonempty).unwrap();
    fs::write(nonempty.join("untrusted"), b"x").unwrap();
    let occupied = run(
        &binary,
        scratch.path(),
        &[
            "--action",
            ACTOR_ACTION_ID,
            "--output-dir",
            nonempty.to_str().unwrap(),
        ],
    );
    assert_error(&occupied, "FAIL_Q1_PROJECTION_OUTPUT_DIR");

    let real = scratch.path().join("real-output");
    let link = scratch.path().join("linked-output");
    fs::create_dir(&real).unwrap();
    symlink(&real, &link).unwrap();
    let linked = run(
        &binary,
        scratch.path(),
        &[
            "--action",
            ACTOR_ACTION_ID,
            "--output-dir",
            link.to_str().unwrap(),
        ],
    );
    assert_error(&linked, "FAIL_Q1_PROJECTION_OUTPUT_DIR");
}
