use hegel_strict_canonicalizer::{
    canonicalize_source_json, decode_strict_canonical_ast, hex_decode, replay_capacity_subset,
    replay_golden_vectors_file, AST_SCHEMA_ID, CBOR_PROFILE_ID, REPLAY_SCHEMA_VERSION,
};
use serde_json::{json, Value};
use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

enum Mode {
    Vectors(PathBuf),
    AstJson(String),
    DecodeCborHex(String),
    CapacityReplay,
}

fn usage() -> &'static str {
    "Usage:\n  hegel-strict-canonicalizer [--vectors PATH] [--pretty]\n  hegel-strict-canonicalizer --ast-json JSON [--pretty]\n  hegel-strict-canonicalizer --decode-cbor-hex HEX [--pretty]\n  hegel-strict-canonicalizer --capacity-replay [--pretty]\n\nWith no mode argument, the shared strict_ast_cbor_v1.json fixture is used."
}

fn default_vectors_path() -> PathBuf {
    let from_cwd = PathBuf::from("golden_vectors/strict_ast_cbor_v1.json");
    if from_cwd.exists() {
        return from_cwd;
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../golden_vectors/strict_ast_cbor_v1.json")
}

fn parse_args() -> Result<(Mode, bool), String> {
    let mut mode = None;
    let mut pretty = false;
    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--vectors" => {
                let path = args
                    .next()
                    .ok_or_else(|| "--vectors requires a path".to_owned())?;
                if mode.is_some() {
                    return Err("choose exactly one input mode".to_owned());
                }
                mode = Some(Mode::Vectors(PathBuf::from(path)));
            }
            "--ast-json" => {
                let source = args
                    .next()
                    .ok_or_else(|| "--ast-json requires a JSON value".to_owned())?;
                if mode.is_some() {
                    return Err("choose exactly one input mode".to_owned());
                }
                mode = Some(Mode::AstJson(source));
            }
            "--decode-cbor-hex" => {
                let hex = args
                    .next()
                    .ok_or_else(|| "--decode-cbor-hex requires hex bytes".to_owned())?;
                if mode.is_some() {
                    return Err("choose exactly one input mode".to_owned());
                }
                mode = Some(Mode::DecodeCborHex(hex));
            }
            "--capacity-replay" => {
                if mode.is_some() {
                    return Err("choose exactly one input mode".to_owned());
                }
                mode = Some(Mode::CapacityReplay);
            }
            "--pretty" => pretty = true,
            "-h" | "--help" => return Err(usage().to_owned()),
            unknown => return Err(format!("unknown argument {unknown:?}\n\n{}", usage())),
        }
    }
    Ok((
        mode.unwrap_or_else(|| Mode::Vectors(default_vectors_path())),
        pretty,
    ))
}

fn program_json(program: &hegel_strict_canonicalizer::CanonicalProgram) -> Value {
    json!({
        "schema_version": REPLAY_SCHEMA_VERSION,
        "implementation": "rust",
        "cbor_profile_id": CBOR_PROFILE_ID,
        "ast_schema_id": AST_SCHEMA_ID,
        "status": "ACCEPTED",
        "canonical_cbor_hex": program.canonical_cbor_hex(),
        "canonical_ast_hash": program.canonical_ast_hash_id(),
        "program_hash": program.canonical_ast_hash_id(),
        "root_operator_id": program.root_operator_id,
        "output_sort": program.output_sort,
        "depth": program.depth,
        "node_count": program.node_count,
        "distinct_bit_slot_count": program.distinct_bit_slot_count,
        "aggregate_leaf_count": program.aggregate_leaf_count,
        "scalar_parameter_occurrence_count": program.scalar_parameter_occurrence_count,
    })
}

fn error_json(error: &hegel_strict_canonicalizer::CanonicalError) -> Value {
    json!({
        "schema_version": REPLAY_SCHEMA_VERSION,
        "implementation": "rust",
        "status": "REJECTED",
        "error_code": error.code,
        "error_message": error.message,
    })
}

fn render(value: &impl serde::Serialize, pretty: bool) -> Result<String, String> {
    if pretty {
        serde_json::to_string_pretty(value)
    } else {
        serde_json::to_string(value)
    }
    .map_err(|error| format!("failed to serialize JSON output: {error}"))
}

fn run() -> Result<(Value, u8, bool), String> {
    let (mode, pretty) = parse_args()?;
    match mode {
        Mode::Vectors(path) => {
            let summary = replay_golden_vectors_file(&path)?;
            let exit = if summary.all_expectations_match { 0 } else { 1 };
            let value = serde_json::to_value(summary)
                .map_err(|error| format!("failed to serialize replay summary: {error}"))?;
            Ok((value, exit, pretty))
        }
        Mode::AstJson(source) => {
            let source: Value = serde_json::from_str(&source)
                .map_err(|error| format!("--ast-json is not valid JSON: {error}"))?;
            match canonicalize_source_json(&source) {
                Ok(program) => Ok((program_json(&program), 0, pretty)),
                Err(error) => Ok((error_json(&error), 1, pretty)),
            }
        }
        Mode::DecodeCborHex(hex) => {
            let bytes = hex_decode(&hex).map_err(|error| error.to_string())?;
            match decode_strict_canonical_ast(&bytes) {
                Ok(program) => Ok((program_json(&program), 0, pretty)),
                Err(error) => Ok((error_json(&error), 1, pretty)),
            }
        }
        Mode::CapacityReplay => match replay_capacity_subset() {
            Ok(report) => {
                let value = serde_json::to_value(report).map_err(|error| {
                    format!("failed to serialize capacity replay report: {error}")
                })?;
                Ok((value, 0, pretty))
            }
            Err(error) => Ok((error_json(&error), 1, pretty)),
        },
    }
}

fn main() -> ExitCode {
    match run() {
        Ok((value, code, pretty)) => match render(&value, pretty) {
            Ok(output) => {
                println!("{output}");
                ExitCode::from(code)
            }
            Err(error) => {
                eprintln!("{error}");
                ExitCode::from(2)
            }
        },
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(2)
        }
    }
}
