use hegel_q1_full_capacity_preflight_engine::{
    formal_materializer_json_line, run_formal_archive_materializer_with_scratch, CapacityLimits,
    EngineError, FORMAL_MATERIALIZER_ACTION_ID, FORMAL_MATERIALIZER_IMPLEMENTATION_ID,
};
use serde_json::json;
use std::fs;
use std::io::{self, Write};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn error_line(error: &EngineError) -> Vec<u8> {
    let value = json!({
        "action_id": FORMAL_MATERIALIZER_ACTION_ID,
        "diagnostic_only": false,
        "error_code": error.code,
        "error_detail": error.detail,
        "formal_archive_materialized": false,
        "implementation_id": FORMAL_MATERIALIZER_IMPLEMENTATION_ID,
        "role_evaluation_performed": false,
        "split_accessed": false,
        "target_truth_accessed": false
    });
    let mut bytes = serde_json::to_vec(&value).expect("error envelope is serializable");
    bytes.push(b'\n');
    bytes
}

fn parse(
    arguments: &[String],
) -> Result<(&str, &str, &str, &str, PathBuf, PathBuf), EngineError> {
    if arguments.len() != 14
        || arguments[0] != "--action"
        || arguments[1] != FORMAL_MATERIALIZER_ACTION_ID
        || arguments[2] != "--capsule-digest"
        || arguments[4] != "--source-commit"
        || arguments[6] != "--run-genesis-root"
        || arguments[8] != "--prestart-admission-receipt-root"
        || arguments[10] != "--output-dir"
        || arguments[12] != "--scratch-dir"
    {
        return Err(EngineError {
            code: "Q1_FORMAL_MATERIALIZER_ACTION_NOT_ADMITTED".to_owned(),
            detail: format!(
                "the only admitted invocation is --action {FORMAL_MATERIALIZER_ACTION_ID} --capsule-digest sha256:HEX64 --source-commit HEX40 --run-genesis-root sha256:HEX64 --prestart-admission-receipt-root sha256:HEX64 --output-dir ABSOLUTE_EMPTY_DIR --scratch-dir ABSOLUTE_EMPTY_DIR"
            ),
        });
    }
    Ok((
        &arguments[3],
        &arguments[5],
        &arguments[7],
        &arguments[9],
        PathBuf::from(&arguments[11]),
        PathBuf::from(&arguments[13]),
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

fn run(arguments: &[String]) -> Result<Vec<u8>, EngineError> {
    let (
        capsule_digest,
        source_commit,
        run_genesis_root,
        prestart_admission_receipt_root,
        output_dir,
        scratch_dir,
    ) = parse(arguments)?;
    let output_identity = validate_empty_dir(
        &output_dir,
        "output",
        "REJECT_FORMAL_MATERIALIZER_OUTPUT_DIR",
    )?;
    let scratch_identity = validate_empty_dir(
        &scratch_dir,
        "scratch",
        "REJECT_FORMAL_MATERIALIZER_SCRATCH_DIR",
    )?;
    if output_identity == scratch_identity {
        return Err(EngineError {
            code: "REJECT_FORMAL_MATERIALIZER_DIRECTORY_ALIAS".to_owned(),
            detail: "output and scratch directories must be distinct".to_owned(),
        });
    }
    let report = run_formal_archive_materializer_with_scratch(
        CapacityLimits::q1_formal_schedule(),
        capsule_digest,
        source_commit,
        run_genesis_root,
        prestart_admission_receipt_root,
        &output_dir,
        &scratch_dir,
    );
    let scratch_after = validate_empty_dir(
        &scratch_dir,
        "scratch",
        "FAIL_FORMAL_MATERIALIZER_SCRATCH_CLEANUP",
    )?;
    if scratch_after != scratch_identity {
        return Err(EngineError {
            code: "FAIL_FORMAL_MATERIALIZER_SCRATCH_CLEANUP".to_owned(),
            detail: "scratch directory identity changed during materialization".to_owned(),
        });
    }
    let report = report?;
    let output_after = fs::symlink_metadata(&output_dir).map_err(|error| EngineError {
        code: "FAIL_FORMAL_MATERIALIZER_OUTPUT_ISOLATION".to_owned(),
        detail: format!("cannot inspect output directory after materialization: {error}"),
    })?;
    if (output_after.dev(), output_after.ino()) != output_identity {
        return Err(EngineError {
            code: "FAIL_FORMAL_MATERIALIZER_OUTPUT_ISOLATION".to_owned(),
            detail: "output directory identity changed during materialization".to_owned(),
        });
    }
    formal_materializer_json_line(&report)
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
