use hegel_m3_closure_enumerator_shrink2::{
    enumerate_complete_diagnostic, implementation_binding_material, EnumerationError,
};
use serde_json::to_string_pretty;
use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

enum Mode {
    BindingMaterial,
    Enumerate {
        child_dsl_root: [u8; 32],
        operator_root: [u8; 32],
        identifier_root: [u8; 32],
        output_directory: PathBuf,
    },
}

fn usage() -> &'static str {
    "Usage:\n  hegel-m3-closure-enumerator-shrink2 --binding-material\n  \
     hegel-m3-closure-enumerator-shrink2 --enumerate-diagnostic \
     --child-dsl-spec-root HEX64 --operator-semantics-root HEX64 \
     --identifier-registry-root HEX64 --output-directory PATH"
}

fn parse_hex_root(value: &str, field: &str) -> Result<[u8; 32], String> {
    let value = value.strip_prefix("0x").unwrap_or(value);
    if value.len() != 64 {
        return Err(format!("{field} must be exactly 64 hexadecimal digits"));
    }
    let mut result = [0u8; 32];
    for (index, byte) in result.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)
            .map_err(|_| format!("{field} contains non-hexadecimal input"))?;
    }
    Ok(result)
}

fn parse_args() -> Result<Mode, String> {
    let mut args = env::args().skip(1);
    let Some(mode) = args.next() else {
        return Err(usage().to_owned());
    };
    if mode == "--binding-material" {
        if args.next().is_some() {
            return Err("--binding-material takes no additional arguments".to_owned());
        }
        return Ok(Mode::BindingMaterial);
    }
    if mode != "--enumerate-diagnostic" {
        return Err(usage().to_owned());
    }
    let mut child = None;
    let mut operator = None;
    let mut identifier = None;
    let mut output = None;
    while let Some(argument) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("{argument} requires a value"))?;
        match argument.as_str() {
            "--child-dsl-spec-root" => child = Some(parse_hex_root(&value, &argument)?),
            "--operator-semantics-root" => {
                operator = Some(parse_hex_root(&value, &argument)?)
            }
            "--identifier-registry-root" => {
                identifier = Some(parse_hex_root(&value, &argument)?)
            }
            "--output-directory" => output = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown argument {argument:?}\n\n{}", usage())),
        }
    }
    Ok(Mode::Enumerate {
        child_dsl_root: child.ok_or("missing --child-dsl-spec-root")?,
        operator_root: operator.ok_or("missing --operator-semantics-root")?,
        identifier_root: identifier.ok_or("missing --identifier-registry-root")?,
        output_directory: output.ok_or("missing --output-directory")?,
    })
}

fn run() -> Result<(), String> {
    match parse_args()? {
        Mode::BindingMaterial => {
            println!(
                "{}",
                to_string_pretty(&implementation_binding_material())
                    .map_err(|error| error.to_string())?
            );
        }
        Mode::Enumerate {
            child_dsl_root,
            operator_root,
            identifier_root,
            output_directory,
        } => {
            let artifacts =
                enumerate_complete_diagnostic(child_dsl_root, operator_root, identifier_root)
                    .map_err(format_enumeration_error)?;
            artifacts
                .write_to_directory(&output_directory)
                .map_err(format_enumeration_error)?;
            println!(
                "{}",
                to_string_pretty(&artifacts.report).map_err(|error| error.to_string())?
            );
        }
    }
    Ok(())
}

fn format_enumeration_error(error: EnumerationError) -> String {
    error.to_string()
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(2)
        }
    }
}
