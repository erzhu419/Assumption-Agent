use hegel_strict_canonicalizer::{hex_decode, validate_strict_cbor};
use hegel_strict_canonicalizer_shrink3::{
    canonicalize_shrink3_source_json, decode_shrink3_canonical_ast,
    replay_shrink3_capacity_subset, replay_shrink3_golden_vectors, sort_name, Shrink3Error,
    AST_HASH_DOMAIN, AST_SCHEMA_ID, CBOR_PROFILE_ID, DSL_VERSION, FREEZE_VERSION,
    PARENT_DSL_VERSION, PARENT_FREEZE_VERSION, REPLAY_SCHEMA_VERSION,
};
use serde_json::{json, Value};
use std::env;
use std::process::ExitCode;

enum Mode {
    AstJson(String),
    DecodeCborHex(String),
    CapacityReplay,
    GoldenReplay,
}

fn usage() -> &'static str {
    "Usage:\n  hegel-strict-canonicalizer-shrink3 --ast-json JSON [--pretty]\n  hegel-strict-canonicalizer-shrink3 --decode-cbor-hex HEX [--pretty]\n  hegel-strict-canonicalizer-shrink3 --capacity-replay [--pretty]\n  hegel-strict-canonicalizer-shrink3 --golden-replay [--pretty]"
}

fn parse_args() -> Result<(Mode, bool), String> {
    let mut mode = None;
    let mut pretty = false;
    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--ast-json" => {
                let source = args
                    .next()
                    .ok_or_else(|| "--ast-json requires JSON".to_owned())?;
                if mode.is_some() {
                    return Err("choose exactly one mode".to_owned());
                }
                mode = Some(Mode::AstJson(source));
            }
            "--decode-cbor-hex" => {
                let source = args
                    .next()
                    .ok_or_else(|| "--decode-cbor-hex requires HEX".to_owned())?;
                if mode.is_some() {
                    return Err("choose exactly one mode".to_owned());
                }
                mode = Some(Mode::DecodeCborHex(source));
            }
            "--golden-replay" => {
                if mode.is_some() {
                    return Err("choose exactly one mode".to_owned());
                }
                mode = Some(Mode::GoldenReplay);
            }
            "--capacity-replay" => {
                if mode.is_some() {
                    return Err("choose exactly one mode".to_owned());
                }
                mode = Some(Mode::CapacityReplay);
            }
            "--pretty" => pretty = true,
            "-h" | "--help" => return Err(usage().to_owned()),
            unknown => return Err(format!("unknown argument {unknown:?}\n\n{}", usage())),
        }
    }
    Ok((mode.ok_or_else(|| usage().to_owned())?, pretty))
}

fn program_json(program: &hegel_strict_canonicalizer::CanonicalProgram) -> Value {
    json!({
        "schema_version": REPLAY_SCHEMA_VERSION,
        "implementation": "rust",
        "parent_dsl_version": PARENT_DSL_VERSION,
        "parent_freeze_version": PARENT_FREEZE_VERSION,
        "dsl_version": DSL_VERSION,
        "freeze_version": FREEZE_VERSION,
        "cbor_profile_id": CBOR_PROFILE_ID,
        "ast_schema_id": AST_SCHEMA_ID,
        "ast_hash_domain": AST_HASH_DOMAIN,
        "status": "ACCEPTED",
        "canonical_cbor_hex": program.canonical_cbor_hex(),
        "canonical_ast_hash": program.canonical_ast_hash_id(),
        "root_operator_id": program.root_operator_id,
        "output_sort": sort_name(program.output_sort),
        "depth": program.depth,
        "node_count": program.node_count,
        "scalar_parameter_occurrence_count": program.scalar_parameter_occurrence_count,
    })
}

fn error_json(error: &Shrink3Error) -> Value {
    json!({
        "schema_version": REPLAY_SCHEMA_VERSION,
        "implementation": "rust",
        "parent_dsl_version": PARENT_DSL_VERSION,
        "parent_freeze_version": PARENT_FREEZE_VERSION,
        "dsl_version": DSL_VERSION,
        "freeze_version": FREEZE_VERSION,
        "cbor_profile_id": CBOR_PROFILE_ID,
        "ast_schema_id": AST_SCHEMA_ID,
        "ast_hash_domain": AST_HASH_DOMAIN,
        "status": "REJECTED",
        "error_code": error.code,
        "error_message": error.message,
    })
}

fn run() -> Result<(Value, u8, bool), String> {
    let (mode, pretty) = parse_args()?;
    match mode {
        Mode::AstJson(source) => {
            let value: Value = serde_json::from_str(&source)
                .map_err(|error| format!("invalid --ast-json: {error}"))?;
            match canonicalize_shrink3_source_json(&value) {
                Ok(program) => Ok((program_json(&program), 0, pretty)),
                Err(error) => Ok((error_json(&error), 1, pretty)),
            }
        }
        Mode::DecodeCborHex(source) => {
            let bytes = hex_decode(&source).map_err(|error| error.to_string())?;
            let generic_cbor_parse = validate_strict_cbor(&bytes).is_ok();
            match decode_shrink3_canonical_ast(&bytes) {
                Ok(program) => {
                    let mut report = program_json(&program);
                    report["generic_cbor_parse"] = json!(generic_cbor_parse);
                    Ok((report, 0, pretty))
                }
                Err(error) => {
                    let mut report = error_json(&error);
                    report["generic_cbor_parse"] = json!(generic_cbor_parse);
                    Ok((report, 1, pretty))
                }
            }
        }
        Mode::GoldenReplay => {
            let report = replay_shrink3_golden_vectors().map_err(|error| error.to_string())?;
            let value = serde_json::to_value(report)
                .map_err(|error| format!("failed to serialize report: {error}"))?;
            Ok((value, 0, pretty))
        }
        Mode::CapacityReplay => {
            let report = replay_shrink3_capacity_subset().map_err(|error| error.to_string())?;
            let value = serde_json::to_value(report)
                .map_err(|error| format!("failed to serialize report: {error}"))?;
            Ok((value, 0, pretty))
        }
    }
}

fn main() -> ExitCode {
    match run() {
        Ok((value, code, pretty)) => {
            let output = if pretty {
                serde_json::to_string_pretty(&value)
            } else {
                serde_json::to_string(&value)
            };
            match output {
                Ok(output) => {
                    println!("{output}");
                    ExitCode::from(code)
                }
                Err(error) => {
                    eprintln!("{error}");
                    ExitCode::from(2)
                }
            }
        }
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(2)
        }
    }
}
