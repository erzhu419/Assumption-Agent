use hegel_q0_quotient_oracle::run_micro_oracle;
use serde_json::json;

fn main() {
    match run_micro_oracle() {
        Ok(endpoint) => match endpoint.canonical_json() {
            Ok(json) => println!("{json}"),
            Err(error) => {
                println!(
                    "{}",
                    serde_json::to_string(&json!({
                        "schema_version": "hegel-q0-rust-micro-oracle-error/1",
                        "status": "FAIL_SEMANTICS_MISMATCH",
                        "error_code": error.code,
                        "detail": error.detail,
                        "guard_id": error.guard_id,
                        "authority_claimed": false,
                    }))
                    .expect("static error JSON must encode")
                );
                std::process::exit(1);
            }
        },
        Err(error) => {
            println!(
                "{}",
                serde_json::to_string(&json!({
                    "schema_version": "hegel-q0-rust-micro-oracle-error/1",
                    "status": error.code,
                    "error_code": error.code,
                    "detail": error.detail,
                    "guard_id": error.guard_id,
                    "authority_claimed": false,
                }))
                .expect("static error JSON must encode")
            );
            std::process::exit(1);
        }
    }
}
