use hegel_q1_full_capacity_preflight_engine::{
    canonical_json_line, run_capacity_preflight_with_scratch, sidecar_payloads, CapacityLimits,
    EngineError, ACTION_ID,
};
use serde_json::json;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn error_line(error: &EngineError) -> Vec<u8> {
    let value = json!({
        "active_transition_allowed": false,
        "error_code": error.code,
        "error_detail": error.detail,
        "formal_roots_generated": false,
        "q1_execution_started": false,
        "q1_formal_roots": null,
        "q1_gate_count": 0,
        "q1_gate_mask": 0,
        "q1_output_slots": [null, null, null, null, null, null, null, null],
        "q1_receipt": null,
        "q1_state": "NOT_RUN",
        "target_truth_accessed": false
    });
    let mut bytes = serde_json::to_vec(&value).expect("error envelope is serializable");
    bytes.push(b'\n');
    bytes
}

fn parse(arguments: &[String]) -> Result<(&str, &str, PathBuf, PathBuf), EngineError> {
    if arguments.len() != 10
        || arguments[0] != "--action"
        || arguments[1] != ACTION_ID
        || arguments[2] != "--capsule-digest"
        || arguments[4] != "--source-commit"
        || arguments[6] != "--output-dir"
        || arguments[8] != "--scratch-dir"
    {
        return Err(EngineError {
            code: "Q1_FULL_CAPACITY_ACTION_NOT_ADMITTED".to_owned(),
            detail: format!(
                "the only admitted invocation is --action {ACTION_ID} --capsule-digest sha256:HEX64 --source-commit HEX40 --output-dir ABSOLUTE_EMPTY_DIR --scratch-dir ABSOLUTE_EMPTY_DIR"
            ),
        });
    }
    Ok((
        &arguments[3],
        &arguments[5],
        PathBuf::from(&arguments[7]),
        PathBuf::from(&arguments[9]),
    ))
}

fn validate_empty_dir(
    path: &Path,
    label: &str,
    error_code: &str,
) -> Result<(u64, u64), EngineError> {
    let metadata = fs::symlink_metadata(path).map_err(|error| EngineError {
        code: error_code.to_owned(),
        detail: format!("cannot inspect {label} directory: {error}"),
    })?;
    let resolved = fs::canonicalize(path).map_err(|error| EngineError {
        code: error_code.to_owned(),
        detail: format!("cannot resolve {label} directory: {error}"),
    })?;
    if !path.is_absolute()
        || metadata.file_type().is_symlink()
        || !metadata.is_dir()
        || resolved != path
    {
        return Err(EngineError {
            code: error_code.to_owned(),
            detail: format!(
                "{label} directory must be one existing canonical nonsymlink absolute directory"
            ),
        });
    }
    if fs::read_dir(path)
        .map_err(|error| EngineError {
            code: error_code.to_owned(),
            detail: format!("cannot list {label} directory: {error}"),
        })?
        .next()
        .is_some()
    {
        return Err(EngineError {
            code: error_code.to_owned(),
            detail: format!("{label} directory must be empty"),
        });
    }
    Ok((metadata.dev(), metadata.ino()))
}

fn write_sidecars(
    path: &Path,
    report: &hegel_q1_full_capacity_preflight_engine::EngineDiagnostic,
) -> Result<(), EngineError> {
    for (filename, payload) in sidecar_payloads(report)? {
        let target = path.join(filename);
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&target)
            .map_err(|error| EngineError {
                code: "FAIL_PREFLIGHT_SIDECAR_WRITE".to_owned(),
                detail: format!("cannot create {filename}: {error}"),
            })?;
        file.write_all(&payload)
            .and_then(|_| file.sync_all())
            .map_err(|error| EngineError {
                code: "FAIL_PREFLIGHT_SIDECAR_WRITE".to_owned(),
                detail: format!("cannot persist {filename}: {error}"),
            })?;
    }
    Ok(())
}

fn run(arguments: &[String]) -> Result<Vec<u8>, EngineError> {
    let (capsule_digest, source_commit, output_dir, scratch_dir) = parse(arguments)?;
    let output_identity = validate_empty_dir(
        &output_dir,
        "output",
        "REJECT_PREFLIGHT_OUTPUT_DIR",
    )?;
    let scratch_identity = validate_empty_dir(
        &scratch_dir,
        "scratch",
        "REJECT_PREFLIGHT_SCRATCH_DIR",
    )?;
    if output_identity == scratch_identity {
        return Err(EngineError {
            code: "REJECT_PREFLIGHT_DIRECTORY_ALIAS".to_owned(),
            detail: "output and scratch directories must be distinct".to_owned(),
        });
    }
    let report = run_capacity_preflight_with_scratch(
        CapacityLimits::full_node6(),
        capsule_digest,
        source_commit,
        &scratch_dir,
    );
    let scratch_after = validate_empty_dir(
        &scratch_dir,
        "scratch",
        "FAIL_PREFLIGHT_SCRATCH_CLEANUP",
    )?;
    if scratch_after != scratch_identity {
        return Err(EngineError {
            code: "FAIL_PREFLIGHT_SCRATCH_CLEANUP".to_owned(),
            detail: "scratch directory identity changed during the diagnostic".to_owned(),
        });
    }
    let report = report?;
    let output_after = validate_empty_dir(
        &output_dir,
        "output",
        "FAIL_PREFLIGHT_OUTPUT_ISOLATION",
    )?;
    if output_after != output_identity {
        return Err(EngineError {
            code: "FAIL_PREFLIGHT_OUTPUT_ISOLATION".to_owned(),
            detail: "output directory identity changed before sidecar publication".to_owned(),
        });
    }
    write_sidecars(&output_dir, &report)?;
    canonical_json_line(&report)
}

fn main() -> ExitCode {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    let (bytes, code) = match run(&arguments) {
        Ok(bytes) => (bytes, ExitCode::SUCCESS),
        Err(error) => (error_line(&error), ExitCode::FAILURE),
    };
    let mut stdout = io::stdout().lock();
    if stdout.write_all(&bytes).and_then(|_| stdout.flush()).is_err() {
        return ExitCode::FAILURE;
    }
    code
}
