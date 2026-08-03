use hegel_m25_bridge_dag_replay::{
    bridge_attestation_signature_preimage_v1, replay_file, FAIL_PACKAGE_AUTHORITY,
    FAIL_PURPOSE,
};
use sha2::{Digest, Sha256};
use std::fs::OpenOptions;
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

const RECEIPT_SCHEMA: &str = "hegel-phase3-m25-bridge-dag-actor-replay-receipt/1";
const SPLIT_CLAIM: &str = "SEALED_ROOT_COUNT_AND_PURPOSE1_BINDING_ONLY";

struct Options {
    package: PathBuf,
    private_temp: PathBuf,
    authoritative_runtime: bool,
    expected_purpose: Option<u64>,
    signing_preimage_out: Option<PathBuf>,
}

fn usage() -> ! {
    eprintln!(
        "usage: hegel-m25-bridge-dag-replay PACKAGE.cbor PRIVATE_TEMP_DIR\n\
         or: hegel-m25-bridge-dag-replay --authoritative-runtime \
         --expected-purpose PURPOSE --signature-preimage-out PATH \
         PACKAGE.cbor PRIVATE_TEMP_DIR"
    );
    std::process::exit(64)
}

fn parse_options() -> Options {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() == 2 {
        return Options {
            package: PathBuf::from(&args[0]),
            private_temp: PathBuf::from(&args[1]),
            authoritative_runtime: false,
            expected_purpose: None,
            signing_preimage_out: None,
        };
    }
    if args.len() != 7
        || args[0] != "--authoritative-runtime"
        || args[1] != "--expected-purpose"
        || args[3] != "--signature-preimage-out"
    {
        usage();
    }
    let expected_purpose = args[2].parse::<u64>().unwrap_or_else(|_| usage());
    if !(1..=3).contains(&expected_purpose) {
        usage();
    }
    Options {
        package: PathBuf::from(&args[5]),
        private_temp: PathBuf::from(&args[6]),
        authoritative_runtime: true,
        expected_purpose: Some(expected_purpose),
        signing_preimage_out: Some(PathBuf::from(&args[4])),
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn write_new(path: &Path, payload: &[u8]) -> std::io::Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)?;
    file.write_all(payload)?;
    file.sync_all()
}

fn receipt(
    authoritative: bool,
    bridge_root: &[u8; 32],
    candidate_root: &[u8; 32],
    package_digest: &[u8; 32],
    purpose: u64,
    purpose1_signature_verified: bool,
) -> String {
    let body = format!(
        "{{\"authoritative\":{authoritative},\"bridge_statement_root_hex\":\"{}\",\
         \"candidate_root_hex\":\"{}\",\"eligible_to_sign_bridge_statement\":true,\
         \"implementation\":\"rust-full-dag-replay-v1\",\"package_digest_hex\":\"{}\",\
         \"purpose\":{purpose},\"purpose1_signature_verified\":{purpose1_signature_verified},\
         \"schema\":\"{RECEIPT_SCHEMA}\",\"signing_key_epoch\":0,\
         \"split_claim\":\"{SPLIT_CLAIM}\",\"split_membership_recomputed\":false,\
         \"status\":\"PASS\"}}\n",
        hex(bridge_root),
        hex(candidate_root),
        hex(package_digest),
    );
    let digest = Sha256::digest(body.as_bytes());
    format!(
        "{{\"authoritative\":{authoritative},\"bridge_statement_root_hex\":\"{}\",\
         \"candidate_root_hex\":\"{}\",\"eligible_to_sign_bridge_statement\":true,\
         \"implementation\":\"rust-full-dag-replay-v1\",\"package_digest_hex\":\"{}\",\
         \"purpose\":{purpose},\"purpose1_signature_verified\":{purpose1_signature_verified},\
         \"receipt_sha256\":\"{}\",\"schema\":\"{RECEIPT_SCHEMA}\",\
         \"signing_key_epoch\":0,\"split_claim\":\"{SPLIT_CLAIM}\",\
         \"split_membership_recomputed\":false,\"status\":\"PASS\"}}\n",
        hex(bridge_root),
        hex(candidate_root),
        hex(package_digest),
        hex(&digest),
    )
}

fn main() {
    let options = parse_options();
    let result = match replay_file(
        &options.package,
        options.authoritative_runtime,
        &options.private_temp,
        Path::new("/usr/bin/openssl"),
    ) {
        Ok(result) => result,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1)
        }
    };
    if options.authoritative_runtime && !result.authoritative {
        eprintln!("{FAIL_PACKAGE_AUTHORITY}: formal actor requires an authoritative package");
        std::process::exit(1);
    }
    if options
        .expected_purpose
        .is_some_and(|purpose| purpose != result.purpose)
    {
        eprintln!("{FAIL_PURPOSE}: package purpose differs from actor purpose");
        std::process::exit(1);
    }
    if let Some(path) = options.signing_preimage_out {
        if path.parent() != Some(options.private_temp.as_path()) {
            eprintln!("{FAIL_PURPOSE}: signature preimage output must be inside private temp");
            std::process::exit(1);
        }
        let preimage = match bridge_attestation_signature_preimage_v1(
            &result.bridge_root,
            result.purpose,
            0,
        ) {
            Ok(value) => value,
            Err(error) => {
                eprintln!("{error}");
                std::process::exit(1)
            }
        };
        if let Err(error) = write_new(&path, &preimage) {
            eprintln!("{FAIL_PURPOSE}: cannot commit private signing preimage: {error}");
            std::process::exit(1)
        }
    }
    print!(
        "{}",
        receipt(
            result.authoritative,
            &result.bridge_root,
            &result.candidate_root,
            &result.package_digest,
            result.purpose,
            result.purpose1_signature_verified,
        )
    );
}
