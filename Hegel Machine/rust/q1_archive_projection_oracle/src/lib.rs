//! Independent target-blind Rust semantic oracle for the frozen Q1 node-three
//! golden projection.
//!
//! This crate intentionally depends only on the strict canonical AST boundary
//! and its shrink-6 admission layer.  It regenerates the two production input
//! universes, the complete 810-leaf v1.6 surface, exact typed behaviors, the
//! continuation bank and its multiplicity-aware Pareto view.  The output is a
//! diagnostic candidate: Q1 remains `NOT_RUN / 0/20 / null`.

use hegel_strict_canonicalizer::{BinaryOp, CanonicalProgram, Node, Sort, UnaryOp};
use hegel_strict_canonicalizer_shrink6::canonicalize_shrink6_source_node;
use serde::de::{DeserializeSeed, Error as DeError, MapAccess, SeqAccess, Visitor};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::io::Read;
use std::path::Path;

pub const SCHEMA_VERSION: &str = "hegel-phase3a-q05b-rust-node3-semantic-core/1";
pub const IMPLEMENTATION_ID: &str = "hegel-rust-q1-archive-projection-oracle-v1";
pub const DSL_VERSION: &str = "hegel-old-dsl-v1.6.0";
pub const DSL_FREEZE_VERSION: &str = "hegel-freeze-p2b-p3-v1.6.0";
pub const CLOSURE_SEMANTICS_VERSION: &str = "hegel-quotient-closure-v1.0.1";
pub const ARCHIVE_WIRE_VERSION: &str = "hegel-q1-archive-wire-v1.0.0";
pub const PROJECTION_FREEZE_VERSION: &str =
    "hegel-freeze-p3a-q05a-q1-projection-v1.0.0";
pub const QUALIFICATION_WIRE_VERSION: &str = "hegel-q05b-wire-qualification-v1.0.0";
pub const CLAIM: &str = "TARGET_BLIND_NODE3_CANDIDATE_ONLY";
pub const Q1_STATE: &str = "NOT_RUN";
#[cfg(test)]
pub const EXPECTED_ODD_UNIVERSE_ROOT: &str =
    "b7e6eed1174ceee1944bd540cbb0ace4c5c7fc0dffea79dd956a51f7a2410a05";
#[cfg(test)]
pub const EXPECTED_SINK_UNIVERSE_ROOT: &str =
    "1a46f9967ad48df6ec1d9be609de701413ce86171fb0f92495a982bef0f40ff5";
#[cfg(test)]
pub const EXPECTED_COVERAGE_REGISTRY_ROOT: &str =
    "4ff6ef274f4a1122f286a64a155350dab35c5ebedb198977895df97aa402d9c8";

pub const ACTOR_ACTION_ID: &str = "bounded-node3-golden-v1";
pub const ACTOR_ID: &str = "RUST_ENDPOINT";
pub const ACTOR_IMPLEMENTATION_ID: &str =
    "HEGEL_Q1_BOUNDED_NODE3_PROJECTION_RUST_V1";
pub const ACTOR_SCHEMA_VERSION: &str = "hegel-q05b-actor-envelope/1";
pub const ACTOR_STATUS: &str = "BOUNDED_NODE3_CANDIDATE_EMITTED_NOT_QUALIFIED";
pub const ACTOR_ERROR_SCHEMA_VERSION: &str =
    "hegel-phase3a-q05b-rust-projection-actor-error/1";
pub const SOURCE_IDENTITY_ENV: &str = "HEGEL_Q05B_RUST_SOURCE_IDENTITY_SHA256";
pub const OUTPUT_RELATIVE_PATHS: [&str; 5] = [
    "preimages/000-full-v16-leaf-manifest-v1.cbor",
    "preimages/001-odd-node3-partition-evidence-v1.cbor",
    "preimages/002-sink-node3-partition-evidence-v1.cbor",
    "neutral/q05b-node3-sidecar-manifest-v1.cbor",
    "neutral/q05b-node3-golden-manifest-v1.cbor",
];

/// Exact clean-Commit-A source closure hashed by the build supervisor.
///
/// Paths are relative to the `Hegel Machine/` project root and are hashed in
/// this frozen order as `u32be(path_len) || path || u64be(payload_len) ||
/// payload`.  Production runtime containers contain only the resulting binary
/// and an empty output directory; they never read this source closure.
pub const SOURCE_IDENTITY_RELATIVE_PATHS: &[&str] = &[
    "config/phase3_q05b_node3_dual_projection_qualification_v1.json",
    "config/phase3_q1_archive_projection_freeze_v1.json",
    "docs/Hegel_Machine_Phase3A_Q05a_Q1_Archive_Projection_Engineering_Freeze_v1.md",
    "docs/Hegel_Machine_Phase3A_Q05b_Node3_Dual_Projection_Qualification_Engineering_v1.md",
    "rust/q1_archive_projection_oracle/Cargo.lock",
    "rust/q1_archive_projection_oracle/Cargo.toml",
    "rust/q1_archive_projection_oracle/src/lib.rs",
    "rust/q1_archive_projection_oracle/src/main.rs",
    "rust/q1_archive_projection_oracle/tests/cli.rs",
    "rust/strict_canonicalizer/Cargo.toml",
    "rust/strict_canonicalizer/src/lib.rs",
    "rust/strict_canonicalizer_shrink1/Cargo.toml",
    "rust/strict_canonicalizer_shrink1/src/lib.rs",
    "rust/strict_canonicalizer_shrink2/Cargo.toml",
    "rust/strict_canonicalizer_shrink2/src/lib.rs",
    "rust/strict_canonicalizer_shrink3/Cargo.toml",
    "rust/strict_canonicalizer_shrink3/src/lib.rs",
    "rust/strict_canonicalizer_shrink4/Cargo.toml",
    "rust/strict_canonicalizer_shrink4/src/lib.rs",
    "rust/strict_canonicalizer_shrink5/Cargo.toml",
    "rust/strict_canonicalizer_shrink5/src/lib.rs",
    "rust/strict_canonicalizer_shrink6/Cargo.toml",
    "rust/strict_canonicalizer_shrink6/src/lib.rs",
    "src/hegel_machine/phase3_q05b_wire_qualification_contract_v1.py",
    "src/hegel_machine/phase3_q1_archive_projection_v1.py",
    "src/hegel_machine/phase3_q1_external_sort_profile_v1.py",
    "src/hegel_machine/phase3_q1_qualification_wire_v1.py",
];

const BEHAVIOR_TAG: u64 = 0x3701;
const SIGNATURE_TAG: u64 = 0x3702;
const BEHAVIOR_SCHEMA: &[u8] = b"hegel-q1-behavior-blob/1";
const SIGNATURE_SCHEMA: &[u8] = b"hegel-q1-construction-signature/1";
const APPLICATION_SCHEMA: &[u8] = b"hegel-q1-semantic-application-key/1";
const BEHAVIOR_ID_DOMAIN: &[u8] = b"HEGEL/Q1/BEHAVIOR_ID/V1";
const SIGNATURE_ID_DOMAIN: &[u8] = b"HEGEL/Q1/CONSTRUCTION_SIGNATURE_ID/V1";
const PROGRAM_ID_DOMAIN: &[u8] = b"HEGEL/Q1/PROGRAM_ID/V1";
const APPLICATION_ID_DOMAIN: &[u8] = b"HEGEL/Q1/APPLICATION_ID/V1";
const COHORT_ID_DOMAIN: &[u8] = b"HEGEL/Q1/COHORT_ID/V1";
const SNAPSHOT_RECORD_SET_DOMAIN: &[u8] = b"HEGEL/Q1/PREFLIGHT/SNAPSHOT_RECORD_SET/V1";
const PROGRAM_RECORD_TAG: u64 = 0x3703;
const COHORT_RECORD_TAG: u64 = 0x3704;
const CLASS_RECORD_TAG: u64 = 0x3705;
const COVERAGE_RECORD_TAG: u64 = 0x3706;
const PROGRAM_RECORD_SCHEMA: &[u8] = b"hegel-q1-representative-program/1";
const COHORT_RECORD_SCHEMA: &[u8] = b"hegel-q1-continuation-cohort/1";
const CLASS_RECORD_SCHEMA: &[u8] = b"hegel-q1-quotient-class/1";
const COVERAGE_RECORD_SCHEMA: &[u8] = b"hegel-q1-semantic-coverage/1";
const SNAPSHOT_RECORD_SET_SCHEMA: &[u8] = b"hegel-q1-snapshot-record-set/1";
const PROJECTED_STREAM_SCHEMA: &[u8] = b"hegel-q1-projected-record-stream/1";
const COUNTING_DISCARD_STREAM_SCHEMA: &[u8] =
    b"hegel-q05b-counting-discard-record-stream/1";
const PROJECTED_STREAM_DOMAIN: &[u8] = b"HEGEL/Q1/PREFLIGHT/PROJECTED_STREAM/V1";
const STREAM_DESCRIPTOR_SCHEMA: &[u8] = b"hegel-q1-stream-descriptor/1";
const CHUNK_MANIFEST_TAG: u64 = 0x3708;
const CHUNK_MANIFEST_SCHEMA: &[u8] = b"hegel-q1-archive-chunk-manifest/1";
const PROGRAM_RECORD_ID_DOMAIN: &[u8] = b"HEGEL/Q1/PROGRAM_RECORD_ID/V1";
const COHORT_RECORD_ID_DOMAIN: &[u8] = b"HEGEL/Q1/COHORT_RECORD_ID/V1";
const CLASS_RECORD_ID_DOMAIN: &[u8] = b"HEGEL/Q1/CLASS_RECORD_ID/V1";
const COVERAGE_RECORD_ID_DOMAIN: &[u8] = b"HEGEL/Q1/COVERAGE_RECORD_ID/V1";
const FRAMED_BLOB_DOMAIN: &[u8] = b"HEGEL/Q1/FRAMED_BLOB/V1";
const MAX_RECORDS_PER_CHUNK: usize = 4096;
const MAX_CHUNK_FRAMED_BYTES: usize = 16_777_216;
const EXTERNAL_SORT_RUN_PAYLOAD_LIMIT: usize = 268_435_456;
const EXTERNAL_SORT_MERGE_FAN_IN: usize = 16;
const EXTERNAL_SORT_HEADER_BYTES: usize = 68;
const SORTED_STREAM_DOMAIN: &[u8] = b"HEGEL/Q1/PREFLIGHT/SORTED_STREAM/V1";
const SCRATCH_LEDGER_DOMAIN: &[u8] = b"HEGEL/Q1/PREFLIGHT/SCRATCH_LEDGER/V1";
const EXTERNAL_SORT_PROJECTION_DOMAIN: &[u8] =
    b"HEGEL/Q1/PREFLIGHT/EXTERNAL_SORT_PROJECTION/V1";
const EXTERNAL_SORT_PROJECTION_SCHEMA: &[u8] = b"hegel-q1-external-sort-projection/1";
const EXTERNAL_SORT_TRACE_SCHEMA: &[u8] = b"hegel-q1-external-sort-trace/1";
const EXTERNAL_SORT_RUN_SCHEMA: &[u8] = b"hegel-q1-external-sort-run/1";
const SCRATCH_EVENT_SCHEMA: &[u8] = b"hegel-q1-scratch-event/1";
const Q05B_FULL_LEAF_ROW_TAG: u64 = 0x3a00;
const Q05B_FULL_LEAF_MANIFEST_TAG: u64 = 0x3a01;
const Q05B_PARTITION_EVIDENCE_TAG: u64 = 0x3a02;
const Q05B_SIDECAR_MANIFEST_TAG: u64 = 0x3a03;
const Q05B_NODE3_GOLDEN_MANIFEST_TAG: u64 = 0x3a04;
const Q05B_BOUNDED_NODE3_STATE_TAG: u64 = 0x3a07;
const Q05B_FULL_LEAF_ROW_SCHEMA: &[u8] = b"hegel-q05b-full-leaf-row/1";
const Q05B_FULL_LEAF_MANIFEST_SCHEMA: &[u8] = b"hegel-q05b-full-leaf-manifest/1";
const Q05B_PARTITION_EVIDENCE_SCHEMA: &[u8] =
    b"hegel-q05b-node3-partition-evidence/1";
const Q05B_SIDECAR_MANIFEST_SCHEMA: &[u8] = b"hegel-q05b-sidecar-manifest/1";
const Q05B_NODE3_GOLDEN_MANIFEST_SCHEMA: &[u8] =
    b"hegel-q05b-node3-golden-manifest/1";
const Q05B_BOUNDED_NODE3_STATE_SCHEMA: &[u8] = b"hegel-q05b-bounded-node3-state/1";
const Q05B_FULL_LEAF_CONTENT_DOMAIN: &[u8] =
    b"HEGEL/Q05B/FULL_V16_LEAF_MANIFEST_SIDECAR/V1";
const Q05B_PARTITION_EVIDENCE_DOMAIN: &[u8] =
    b"HEGEL/Q05B/NODE3/PARTITION_EVIDENCE/V1";
const Q05B_SIDECAR_MANIFEST_DOMAIN: &[u8] = b"HEGEL/Q05B/NODE3/SIDECAR_MANIFEST/V1";
const Q05B_NODE3_GOLDEN_MANIFEST_DOMAIN: &[u8] =
    b"HEGEL/Q05B/NODE3/GOLDEN_MANIFEST/V1";
const Q05B_BOUNDED_NODE3_STATE_DOMAIN: &[u8] = b"HEGEL/Q05B/NODE3/BOUNDED_STATE/V1";
const Q05B_TAG_REGISTRY_DOMAIN: &[u8] = b"HEGEL/Q05B/QUALIFICATION/TAG_REGISTRY/V1";
const Q1_SEMANTIC_BINDING_DOMAIN: &[u8] = b"HEGEL/Q1/SEMANTIC_BINDING/V1";
const Q1_PROJECTION_PROFILE_DOMAIN: &[u8] = b"HEGEL/Q1/ARCHIVE_PROJECTION_PROFILE/V1";
const Q05B_FULL_LEAF_PATH: &[u8] = b"preimages/000-full-v16-leaf-manifest-v1.cbor";
const Q05B_ODD_EVIDENCE_PATH: &[u8] = b"preimages/001-odd-node3-partition-evidence-v1.cbor";
const Q05B_SINK_EVIDENCE_PATH: &[u8] = b"preimages/002-sink-node3-partition-evidence-v1.cbor";
const Q05B_SIDECAR_PATH: &[u8] = b"neutral/q05b-node3-sidecar-manifest-v1.cbor";
const Q05B_NODE3_GOLDEN_PATH: &[u8] = b"neutral/q05b-node3-golden-manifest-v1.cbor";
const Q05B_OUTPUT_FILE_MODE: u64 = 0o444;
const Q1_SEMANTIC_BINDING_TAG: u64 = 0x3700;
const Q1_ARCHIVE_PROJECTION_PROFILE_TAG: u64 = 0x370b;
const Q1_SEMANTIC_BINDING_SCHEMA: &[u8] = b"hegel-q1-semantic-binding-manifest/1";
const Q1_ARCHIVE_PROJECTION_PROFILE_SCHEMA: &[u8] =
    b"hegel-q1-archive-projection-profile/1";
const Q1_PROJECTION_PROFILE_ID: &[u8] = b"hegel-q1-archive-projection-profile-v1";
const Q1_MDL_PROFILE_ID: &[u8] = b"hegel-mdl-prefix-v1.0.0";
const Q05B_SCOPE_ID: &[u8] = b"BOUNDED_NODE3_SOURCE_AND_WIRE_QUALIFICATION";
const NODE3_TERMINAL_STATUS: &[u8] = b"LOCAL_PROTOTYPE_SUBSET_TRAVERSAL_CLOSED";
const MAX_AST_DEPTH: u32 = 3;
const NODE3_AST_NODE_LIMIT: u32 = 3;
const MAX_AGGREGATE_LEAVES: u32 = 1;
const MAX_SCALAR_OCCURRENCES: u32 = 3;
const MAX_SCOPE_CLAUSES: u32 = 2;
const MAX_DISTINCT_BITS: u32 = 4;
const FROZEN_LEAF_COUNT: usize = 810;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OracleError {
    pub code: String,
    pub detail: String,
}

impl OracleError {
    pub fn new(code: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            detail: detail.into(),
        }
    }
}

impl fmt::Display for OracleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.code, self.detail)
    }
}

impl std::error::Error for OracleError {}

fn validate_lower_hex_sha256(value: &str, label: &str) -> Result<(), OracleError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(OracleError::new(
            "FAIL_Q1_PROJECTION_IDENTITY",
            format!("{label} must be 64 lowercase hexadecimal characters"),
        ));
    }
    Ok(())
}

pub fn embedded_source_identity_sha256() -> Result<&'static str, OracleError> {
    let value = option_env!("HEGEL_Q05B_RUST_SOURCE_IDENTITY_SHA256").ok_or_else(|| {
        OracleError::new(
            "FAIL_Q1_PROJECTION_SOURCE_IDENTITY",
            "build supervisor did not embed HEGEL_Q05B_RUST_SOURCE_IDENTITY_SHA256",
        )
    })?;
    validate_lower_hex_sha256(value, "embedded source identity")?;
    Ok(value)
}

pub fn source_identity_sha256_from_project_root(
    project_root: &Path,
) -> Result<String, OracleError> {
    if !project_root.is_absolute() {
        return Err(OracleError::new(
            "FAIL_Q1_PROJECTION_SOURCE_IDENTITY",
            "source snapshot project root must be absolute",
        ));
    }
    if !SOURCE_IDENTITY_RELATIVE_PATHS
        .windows(2)
        .all(|pair| pair[0] < pair[1])
    {
        return Err(OracleError::new(
            "FAIL_Q1_PROJECTION_SOURCE_IDENTITY",
            "source identity path registry is not strictly ordered",
        ));
    }
    let mut digest = Sha256::new();
    for relative in SOURCE_IDENTITY_RELATIVE_PATHS {
        let relative_path = Path::new(relative);
        if relative_path.is_absolute()
            || relative_path
                .components()
                .any(|part| matches!(part, std::path::Component::ParentDir))
        {
            return Err(OracleError::new(
                "FAIL_Q1_PROJECTION_SOURCE_IDENTITY",
                format!("unadmitted source identity path {relative}"),
            ));
        }
        let path = project_root.join(relative_path);
        let metadata = fs::symlink_metadata(&path).map_err(|error| {
            OracleError::new(
                "FAIL_Q1_PROJECTION_SOURCE_IDENTITY",
                format!("{relative}: {error}"),
            )
        })?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(OracleError::new(
                "FAIL_Q1_PROJECTION_SOURCE_IDENTITY",
                format!("{relative} must be one regular nonsymlink file"),
            ));
        }
        let payload = fs::read(&path).map_err(|error| {
            OracleError::new(
                "FAIL_Q1_PROJECTION_SOURCE_IDENTITY",
                format!("{relative}: {error}"),
            )
        })?;
        let path_bytes = relative.as_bytes();
        let path_length = u32::try_from(path_bytes.len()).map_err(|_| {
            OracleError::new(
                "FAIL_Q1_PROJECTION_SOURCE_IDENTITY",
                "source identity path length exceeds u32",
            )
        })?;
        let payload_length = u64::try_from(payload.len()).map_err(|_| {
            OracleError::new(
                "FAIL_Q1_PROJECTION_SOURCE_IDENTITY",
                "source identity payload length exceeds u64",
            )
        })?;
        digest.update(path_length.to_be_bytes());
        digest.update(path_bytes);
        digest.update(payload_length.to_be_bytes());
        digest.update(&payload);
    }
    Ok(hex_encode(&digest.finalize()))
}

pub fn runtime_identity_sha256() -> Result<String, OracleError> {
    let executable = fs::canonicalize("/proc/self/exe").map_err(|error| {
        OracleError::new(
            "FAIL_Q1_PROJECTION_RUNTIME_IDENTITY",
            format!("cannot resolve /proc/self/exe: {error}"),
        )
    })?;
    let metadata = fs::symlink_metadata(&executable).map_err(|error| {
        OracleError::new(
            "FAIL_Q1_PROJECTION_RUNTIME_IDENTITY",
            format!("cannot stat runtime executable: {error}"),
        )
    })?;
    if !metadata.is_file() {
        return Err(OracleError::new(
            "FAIL_Q1_PROJECTION_RUNTIME_IDENTITY",
            "runtime executable is not a regular file",
        ));
    }
    let mut source = fs::File::open(&executable).map_err(|error| {
        OracleError::new(
            "FAIL_Q1_PROJECTION_RUNTIME_IDENTITY",
            format!("cannot open runtime executable: {error}"),
        )
    })?;
    let mut digest = Sha256::new();
    digest.update(b"HEGEL/Q05B/RUST_RUNTIME_IDENTITY/V1\x00");
    digest.update(metadata.len().to_be_bytes());
    let mut observed = 0_u64;
    let mut block = [0_u8; 1024 * 1024];
    loop {
        let count = source.read(&mut block).map_err(|error| {
            OracleError::new(
                "FAIL_Q1_PROJECTION_RUNTIME_IDENTITY",
                format!("cannot read runtime executable: {error}"),
            )
        })?;
        if count == 0 {
            break;
        }
        observed = observed.checked_add(count as u64).ok_or_else(|| {
            OracleError::new(
                "FAIL_Q1_PROJECTION_RUNTIME_IDENTITY",
                "runtime executable length overflow",
            )
        })?;
        digest.update(&block[..count]);
    }
    if observed != metadata.len() {
        return Err(OracleError::new(
            "FAIL_Q1_PROJECTION_RUNTIME_IDENTITY",
            "runtime executable changed while hashing",
        ));
    }
    Ok(hex_encode(&digest.finalize()))
}

#[derive(Clone, Copy)]
struct StrictJsonSeed;

impl<'de> DeserializeSeed<'de> for StrictJsonSeed {
    type Value = serde_json::Value;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_any(StrictJsonVisitor)
    }
}

struct StrictJsonVisitor;

impl<'de> Visitor<'de> for StrictJsonVisitor {
    type Value = serde_json::Value;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("strict finite JSON")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(serde_json::Value::Bool(value))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(serde_json::Value::Number(value.into()))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(serde_json::Value::Number(value.into()))
    }

    fn visit_f64<E>(self, _value: f64) -> Result<Self::Value, E>
    where
        E: DeError,
    {
        Err(E::custom("floating-point and non-finite JSON numbers are forbidden"))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: DeError,
    {
        Ok(serde_json::Value::String(value.to_owned()))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(serde_json::Value::String(value))
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(serde_json::Value::Null)
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(serde_json::Value::Null)
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element_seed(StrictJsonSeed)? {
            values.push(value);
        }
        Ok(serde_json::Value::Array(values))
    }

    fn visit_map<A>(self, mut mapping: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut values = serde_json::Map::new();
        let mut keys = BTreeSet::new();
        while let Some(key) = mapping.next_key::<String>()? {
            if !keys.insert(key.clone()) {
                return Err(A::Error::custom(format!("duplicate JSON key {key:?}")));
            }
            values.insert(key, mapping.next_value_seed(StrictJsonSeed)?);
        }
        Ok(serde_json::Value::Object(values))
    }
}

pub fn parse_strict_config_json(payload: &[u8]) -> Result<serde_json::Value, OracleError> {
    let mut deserializer = serde_json::Deserializer::from_slice(payload);
    let value = StrictJsonSeed.deserialize(&mut deserializer).map_err(|error| {
        OracleError::new("FAIL_Q1_PROJECTION_CONFIG_WIRE", error.to_string())
    })?;
    deserializer.end().map_err(|error| {
        OracleError::new("FAIL_Q1_PROJECTION_CONFIG_WIRE", error.to_string())
    })?;
    if !value.is_object() {
        return Err(OracleError::new(
            "FAIL_Q1_PROJECTION_CONFIG_WIRE",
            "configuration must be one JSON object",
        ));
    }
    Ok(value)
}

fn json_path<'a>(
    value: &'a serde_json::Value,
    path: &[&str],
) -> Result<&'a serde_json::Value, OracleError> {
    let mut current = value;
    for name in path {
        current = current.get(*name).ok_or_else(|| {
            OracleError::new(
                "FAIL_Q1_PROJECTION_CONFIG_BINDING",
                format!("missing config field {}", path.join(".")),
            )
        })?;
    }
    Ok(current)
}

fn require_json_value(
    value: &serde_json::Value,
    path: &[&str],
    expected: serde_json::Value,
) -> Result<(), OracleError> {
    let actual = json_path(value, path)?;
    if actual != &expected {
        return Err(OracleError::new(
            "FAIL_Q1_PROJECTION_CONFIG_BINDING",
            format!("config field {} differs", path.join(".")),
        ));
    }
    Ok(())
}

fn commit_a_actual_preconditions_v1() -> serde_json::Value {
    serde_json::json!({
        "actual_entrypoint_implemented": true,
        "source_freeze_execution_status": "NOT_EXECUTED_AT_COMMIT_A",
        "execution_admission_policy": "CONDITIONAL_SINGLE_ATTEMPT_RUNTIME_ADMISSION",
        "implementation_blocked_predicate_ids": [],
        "pending_actual_evidence_predicate_ids": [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16, 17, 18, 19, 20
        ],
        "clean_full40_commit_required": true,
        "head_must_equal_requested_commit": true,
        "commit_a_config_blob_must_equal_runtime_config": true,
        "pinned_local_images_required": true,
        "sealed_source_cargo_runtime_required": true,
        "attempt_unique_docker_execution_authority_required": true,
        "initial_and_precreate_name_absence_required": true,
        "docker_cleanup_owned_cid_only_required": true,
        "foreign_or_unknown_docker_state_zero_mutation_required": true,
        "artifact_target_must_be_absent": true,
        "all_runtime_preconditions_required": true,
        "all_20_predicates_required_before_artifact": true,
        "toctou_revalidation_required": true,
        "atomic_noreplace_artifact_required": true
    })
}

fn require_commit_a_actual_authority_v1(
    primary: &serde_json::Value,
) -> Result<(), OracleError> {
    require_json_value(
        primary,
        &["engineering_status"],
        serde_json::json!("ACTUAL_IMPLEMENTED_CONDITIONALLY_ADMITTED_NOT_EXECUTED"),
    )?;
    require_json_value(
        primary,
        &["actual_preconditions"],
        commit_a_actual_preconditions_v1(),
    )?;
    Ok(())
}

pub fn validate_build_snapshot(project_root: &Path) -> Result<(), OracleError> {
    let primary_bytes = fs::read(
        project_root.join("config/phase3_q05b_node3_dual_projection_qualification_v1.json"),
    )
    .map_err(|error| {
        OracleError::new("FAIL_Q1_PROJECTION_CONFIG_WIRE", error.to_string())
    })?;
    let referenced_bytes = fs::read(
        project_root.join("config/phase3_q1_archive_projection_freeze_v1.json"),
    )
    .map_err(|error| {
        OracleError::new("FAIL_Q1_PROJECTION_CONFIG_WIRE", error.to_string())
    })?;
    let primary = parse_strict_config_json(&primary_bytes)?;
    let referenced = parse_strict_config_json(&referenced_bytes)?;

    require_json_value(
        &primary,
        &["schema_version"],
        serde_json::json!("hegel-phase3a-q05b-node3-dual-projection-qualification/1"),
    )?;
    require_json_value(
        &primary,
        &["registry_and_profile_roots", "qualification_tag_registry_root_hex"],
        serde_json::json!(hex_encode(&q05b_tag_registry_root())),
    )?;
    require_json_value(
        &primary,
        &[
            "registry_and_profile_roots",
            "qualification_predicate_registry_root_hex",
        ],
        serde_json::json!(hex_encode(&qualification_predicate_registry_root())),
    )?;
    require_json_value(
        &primary,
        &["registry_and_profile_roots", "qualification_wire_profile_root_hex"],
        serde_json::json!(hex_encode(&qualification_wire_profile_root())),
    )?;

    let (full_leaf, full_leaf_root) = full_leaf_neutral_object()?;
    let (odd_root, sink_root) = regenerated_universe_root_bytes();
    let semantic_root = q1_semantic_binding_root(full_leaf_root, odd_root, sink_root);
    let projection_root = q1_projection_profile_root(semantic_root);
    require_json_value(
        &primary,
        &["full_v16_leaf_manifest", "root_hex"],
        serde_json::json!(hex_encode(&full_leaf_root)),
    )?;
    require_json_value(
        &primary,
        &["full_v16_leaf_manifest", "sidecar_canonical_cbor_bytes"],
        serde_json::json!(encode(&full_leaf).len()),
    )?;
    require_json_value(
        &primary,
        &["q1_formal_input_roots", "q1_semantic_binding_root_hex"],
        serde_json::json!(hex_encode(&semantic_root)),
    )?;
    require_json_value(
        &primary,
        &["q1_formal_input_roots", "q1_projection_profile_root_hex"],
        serde_json::json!(hex_encode(&projection_root)),
    )?;
    require_json_value(
        &primary,
        &["authority"],
        serde_json::json!({
            "qualification_state": "ACTUAL_IMPLEMENTED_CONDITIONALLY_ADMITTED_NOT_EXECUTED",
            "qualification_predicate_count": 0,
            "qualification_predicate_mask": 0,
            "qualification_predicate_total": 20,
            "qualification_candidate_receipt": null,
            "qualification_final_receipt": null,
            "q1_state": "NOT_RUN",
            "q1_gate_count": 0,
            "q1_gate_mask": 0,
            "q1_gate_total": 20,
            "q1_formal_output_roots": [null, null, null, null, null, null, null, null],
            "q1_receipt": null,
            "q2_state": "NOT_RUN",
            "m3_formal_roots": null,
            "outside_certificate_issued": false,
            "active_transition_allowed": false
        }),
    )?;
    require_commit_a_actual_authority_v1(&primary)?;
    require_json_value(
        &primary,
        &["sidecar_layout", "output_file_count"],
        serde_json::json!(5),
    )?;
    require_json_value(
        &primary,
        &["sidecar_layout", "file_mode_octal"],
        serde_json::json!("0444"),
    )?;
    require_json_value(
        &primary,
        &["actor_stdout_envelope", "action_id"],
        serde_json::json!(ACTOR_ACTION_ID),
    )?;

    require_json_value(
        &referenced,
        &["schema_version"],
        serde_json::json!("hegel-phase3a-q05a-q1-archive-projection-freeze/1"),
    )?;
    require_json_value(
        &referenced,
        &["authority"],
        serde_json::json!({
            "q1_state": "NOT_RUN",
            "q1_execution_started": false,
            "q1_gate_count": 0,
            "q1_gate_mask": 0,
            "q1_gate_total": 20,
            "q1_formal_roots": null,
            "q1_receipt": null,
            "q2_state": "NOT_RUN",
            "role_evaluation_performed": false,
            "target_truth_accessed": false,
            "split_accessed": false,
            "m3_formal_roots": null,
            "outside_certificate_issued": false,
            "active_transition_allowed": false
        }),
    )?;
    require_json_value(
        &referenced,
        &["claim_boundary", "full_node6_capacity_preflight_allowed_now"],
        serde_json::json!(false),
    )?;

    let q05a_doc = fs::read_to_string(project_root.join(
        "docs/Hegel_Machine_Phase3A_Q05a_Q1_Archive_Projection_Engineering_Freeze_v1.md",
    ))
    .map_err(|error| OracleError::new("FAIL_Q1_PROJECTION_DOC_BINDING", error.to_string()))?;
    let q05b_doc = fs::read_to_string(project_root.join(
        "docs/Hegel_Machine_Phase3A_Q05b_Node3_Dual_Projection_Qualification_Engineering_v1.md",
    ))
    .map_err(|error| OracleError::new("FAIL_Q1_PROJECTION_DOC_BINDING", error.to_string()))?;
    for required in [
        "Q1 remains `NOT_RUN`",
        "node-six capacity preflight remains forbidden",
    ] {
        if !q05a_doc.contains(required) {
            return Err(OracleError::new(
                "FAIL_Q1_PROJECTION_DOC_BINDING",
                format!("Q0.5a document is missing {required:?}"),
            ));
        }
    }
    for required in [
        "bd85abed6feb4b4e9fd6102f43c5db3bbaf9733f0ec42ab5b5363e14a86d350e",
        "e3b3df3e81b7632c7c713ef5ec84913f990ad8232a25b851f20c46ac7416bfcb",
        "aa441cdc49ab60324483b9aa44e9fdfc324a6ad49a6bff50af6daa775209816d",
        "q1_state=NOT_RUN",
        "gate count/mask zero",
    ] {
        if !q05b_doc.contains(required) {
            return Err(OracleError::new(
                "FAIL_Q1_PROJECTION_DOC_BINDING",
                format!("Q0.5b document is missing {required:?}"),
            ));
        }
    }
    let wire_source = fs::read_to_string(project_root.join(
        "src/hegel_machine/phase3_q05b_wire_qualification_contract_v1.py",
    ))
    .map_err(|error| OracleError::new("FAIL_Q1_PROJECTION_WIRE_BINDING", error.to_string()))?;
    for required in [
        "HEGEL/Q05B/NODE3/SIDECAR_MANIFEST/V1",
        "bounded-node3-golden-v1",
    ] {
        if !wire_source.contains(required) {
            return Err(OracleError::new(
                "FAIL_Q1_PROJECTION_WIRE_BINDING",
                format!("shared wire source is missing {required:?}"),
            ));
        }
    }
    let projection_source = fs::read_to_string(
        project_root.join("src/hegel_machine/phase3_q1_archive_projection_v1.py"),
    )
    .map_err(|error| OracleError::new("FAIL_Q1_PROJECTION_WIRE_BINDING", error.to_string()))?;
    if !projection_source.contains("hegel-q05b-counting-discard-record-stream/1") {
        return Err(OracleError::new(
            "FAIL_Q1_PROJECTION_WIRE_BINDING",
            "shared projection source is missing the counting/discard schema",
        ));
    }
    Ok(())
}

fn push_json_string(output: &mut String, value: &str) {
    output.push('"');
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\u{08}' => output.push_str("\\b"),
            '\u{0c}' => output.push_str("\\f"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            character if (' '..='~').contains(&character) => output.push(character),
            character => {
                let scalar = character as u32;
                if scalar <= 0xffff {
                    output.push_str(&format!("\\u{scalar:04x}"));
                } else {
                    let adjusted = scalar - 0x1_0000;
                    let high = 0xd800 + (adjusted >> 10);
                    let low = 0xdc00 + (adjusted & 0x3ff);
                    output.push_str(&format!("\\u{high:04x}\\u{low:04x}"));
                }
            }
        }
    }
    output.push('"');
}

fn push_canonical_json(
    output: &mut String,
    value: &serde_json::Value,
) -> Result<(), OracleError> {
    match value {
        serde_json::Value::Null => output.push_str("null"),
        serde_json::Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        serde_json::Value::Number(value) if value.is_i64() || value.is_u64() => {
            output.push_str(&value.to_string())
        }
        serde_json::Value::Number(_) => {
            return Err(OracleError::new(
                "FAIL_Q1_PROJECTION_OUTPUT_WIRE",
                "floating-point JSON output is forbidden",
            ))
        }
        serde_json::Value::String(value) => push_json_string(output, value),
        serde_json::Value::Array(values) => {
            output.push('[');
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    output.push(',');
                }
                push_canonical_json(output, value)?;
            }
            output.push(']');
        }
        serde_json::Value::Object(values) => {
            output.push('{');
            let mut rows = values.iter().collect::<Vec<_>>();
            rows.sort_by(|left, right| left.0.cmp(right.0));
            for (index, (key, value)) in rows.into_iter().enumerate() {
                if index != 0 {
                    output.push(',');
                }
                push_json_string(output, key);
                output.push(':');
                push_canonical_json(output, value)?;
            }
            output.push('}');
        }
    }
    Ok(())
}

pub fn canonical_json_line(value: &serde_json::Value) -> Result<Vec<u8>, OracleError> {
    let mut output = String::new();
    push_canonical_json(&mut output, value)?;
    output.push('\n');
    Ok(output.into_bytes())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActorFile {
    pub relative_path: &'static str,
    pub payload: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActorEmission {
    pub files: Vec<ActorFile>,
    pub stdout: Vec<u8>,
}

pub fn bounded_node3_actor_emission(
    source_identity: &str,
    runtime_identity: &str,
) -> Result<ActorEmission, OracleError> {
    validate_lower_hex_sha256(source_identity, "source identity")?;
    validate_lower_hex_sha256(runtime_identity, "runtime identity")?;
    let bundle = golden_node3_neutral_bundle()?;
    let summary = bundle.summary()?;
    let files = vec![
        ActorFile {
            relative_path: OUTPUT_RELATIVE_PATHS[0],
            payload: bundle.sidecar.full_leaf_manifest.clone(),
        },
        ActorFile {
            relative_path: OUTPUT_RELATIVE_PATHS[1],
            payload: bundle.sidecar.odd_partition_evidence.clone(),
        },
        ActorFile {
            relative_path: OUTPUT_RELATIVE_PATHS[2],
            payload: bundle.sidecar.sink_partition_evidence.clone(),
        },
        ActorFile {
            relative_path: OUTPUT_RELATIVE_PATHS[3],
            payload: bundle.sidecar.sidecar_manifest.clone(),
        },
        ActorFile {
            relative_path: OUTPUT_RELATIVE_PATHS[4],
            payload: bundle.node3_golden_manifest.clone(),
        },
    ];
    if files
        .iter()
        .map(|file| file.relative_path)
        .ne(OUTPUT_RELATIVE_PATHS)
        || files.iter().any(|file| file.payload.is_empty())
    {
        return Err(OracleError::new(
            "FAIL_Q1_PROJECTION_SIDECAR_SET",
            "actor sidecar set differs from the frozen ordered registry",
        ));
    }
    let sidecar = &summary.sidecar.sidecar_manifest;
    let neutral = &summary.node3_golden_manifest;
    let envelope = serde_json::json!({
        "action_id": ACTOR_ACTION_ID,
        "actor_id": ACTOR_ID,
        "file_count": 5,
        "implementation_id": ACTOR_IMPLEMENTATION_ID,
        "neutral_manifest_length": neutral.canonical_byte_length,
        "neutral_manifest_raw_sha256": neutral.raw_sha256,
        "neutral_manifest_relative_path": OUTPUT_RELATIVE_PATHS[4],
        "neutral_manifest_root": neutral.content_root,
        "q1_formal_roots": null,
        "q1_gate_count": 0,
        "q1_gate_mask": 0,
        "q1_output_slots": [null, null, null, null, null, null, null, null],
        "q1_state": "NOT_RUN",
        "runtime_identity_sha256": runtime_identity,
        "schema_version": ACTOR_SCHEMA_VERSION,
        "sidecar_manifest_length": sidecar.canonical_byte_length,
        "sidecar_manifest_raw_sha256": sidecar.raw_sha256,
        "sidecar_manifest_relative_path": OUTPUT_RELATIVE_PATHS[3],
        "sidecar_manifest_root": sidecar.content_root,
        "source_identity_sha256": source_identity,
        "status": ACTOR_STATUS,
    });
    let stdout = canonical_json_line(&envelope)?;
    if stdout.len() >= 1024 * 1024 {
        return Err(OracleError::new(
            "INCONCLUSIVE_Q1_PROJECTION_OUTPUT_LIMIT",
            "actor stdout exceeds one MiB",
        ));
    }
    Ok(ActorEmission { files, stdout })
}

pub fn actor_error_json(error: &OracleError) -> Vec<u8> {
    let value = serde_json::json!({
        "active_transition_allowed": false,
        "authority_claimed": false,
        "detail": error.detail,
        "error_code": error.code,
        "formal_roots_generated": false,
        "full_node6_executed": false,
        "q1_formal_roots": null,
        "q1_gate_count": 0,
        "q1_gate_mask": 0,
        "q1_output_slots": [null, null, null, null, null, null, null, null],
        "q1_receipt": null,
        "q1_state": "NOT_RUN",
        "q2_state": "NOT_RUN",
        "schema_version": ACTOR_ERROR_SCHEMA_VERSION,
        "sidecar_set_complete": false,
    });
    canonical_json_line(&value).unwrap_or_else(|_| {
        b"{\"error_code\":\"FAIL_Q1_PROJECTION_ERROR_OUTPUT_LIMIT\"}\n".to_vec()
    })
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum Cbor {
    Unsigned(u64),
    Negative(u64),
    Bytes(Vec<u8>),
    Array(Vec<Cbor>),
    Bool(bool),
    Null,
}

fn uint(value: u64) -> Cbor {
    Cbor::Unsigned(value)
}

fn int(value: i64) -> Cbor {
    if value >= 0 {
        Cbor::Unsigned(value as u64)
    } else {
        Cbor::Negative((-1 - value) as u64)
    }
}

fn bytes(value: impl AsRef<[u8]>) -> Cbor {
    Cbor::Bytes(value.as_ref().to_vec())
}

fn array(values: impl IntoIterator<Item = Cbor>) -> Cbor {
    Cbor::Array(values.into_iter().collect())
}

fn encode_head(major: u8, value: u64, output: &mut Vec<u8>) {
    let prefix = major << 5;
    match value {
        0..=23 => output.push(prefix | value as u8),
        24..=0xff => {
            output.push(prefix | 24);
            output.push(value as u8);
        }
        0x100..=0xffff => {
            output.push(prefix | 25);
            output.extend_from_slice(&(value as u16).to_be_bytes());
        }
        0x1_0000..=0xffff_ffff => {
            output.push(prefix | 26);
            output.extend_from_slice(&(value as u32).to_be_bytes());
        }
        _ => {
            output.push(prefix | 27);
            output.extend_from_slice(&value.to_be_bytes());
        }
    }
}

fn encode_into(value: &Cbor, output: &mut Vec<u8>) {
    match value {
        Cbor::Unsigned(value) => encode_head(0, *value, output),
        Cbor::Negative(value) => encode_head(1, *value, output),
        Cbor::Bytes(value) => {
            encode_head(2, value.len() as u64, output);
            output.extend_from_slice(value);
        }
        Cbor::Array(values) => {
            encode_head(4, values.len() as u64, output);
            for child in values {
                encode_into(child, output);
            }
        }
        Cbor::Bool(false) => output.push(0xf4),
        Cbor::Bool(true) => output.push(0xf5),
        Cbor::Null => output.push(0xf6),
    }
}

fn encode(value: &Cbor) -> Vec<u8> {
    let mut output = Vec::new();
    encode_into(value, &mut output);
    output
}

fn decode_head(payload: &[u8], offset: &mut usize, additional: u8) -> Result<u64, OracleError> {
    let width = match additional {
        0..=23 => return Ok(u64::from(additional)),
        24 => 1,
        25 => 2,
        26 => 4,
        27 => 8,
        _ => {
            return Err(OracleError::new(
                "REJECT_Q1_CANONICAL_CBOR",
                "indefinite or reserved CBOR length",
            ))
        }
    };
    let end = offset.checked_add(width).ok_or_else(|| {
        OracleError::new("REJECT_Q1_CANONICAL_CBOR", "CBOR length overflow")
    })?;
    let slice = payload.get(*offset..end).ok_or_else(|| {
        OracleError::new("REJECT_Q1_CANONICAL_CBOR", "truncated CBOR argument")
    })?;
    *offset = end;
    let mut bytes = [0_u8; 8];
    bytes[8 - width..].copy_from_slice(slice);
    let value = u64::from_be_bytes(bytes);
    let minimum = match width {
        1 => 24,
        2 => 0x100,
        4 => 0x1_0000,
        8 => 0x1_0000_0000,
        _ => unreachable!(),
    };
    if value < minimum {
        return Err(OracleError::new(
            "REJECT_Q1_CANONICAL_CBOR",
            "non-minimal CBOR integer or length",
        ));
    }
    Ok(value)
}

fn decode_item(payload: &[u8], offset: &mut usize) -> Result<Cbor, OracleError> {
    let initial = *payload.get(*offset).ok_or_else(|| {
        OracleError::new("REJECT_Q1_CANONICAL_CBOR", "truncated CBOR item")
    })?;
    *offset += 1;
    let major = initial >> 5;
    let additional = initial & 31;
    match major {
        0 => Ok(Cbor::Unsigned(decode_head(payload, offset, additional)?)),
        1 => Ok(Cbor::Negative(decode_head(payload, offset, additional)?)),
        2 => {
            let length = usize::try_from(decode_head(payload, offset, additional)?).map_err(|_| {
                OracleError::new("REJECT_Q1_CANONICAL_CBOR", "byte string length exceeds usize")
            })?;
            let end = offset.checked_add(length).ok_or_else(|| {
                OracleError::new("REJECT_Q1_CANONICAL_CBOR", "byte string length overflow")
            })?;
            let value = payload.get(*offset..end).ok_or_else(|| {
                OracleError::new("REJECT_Q1_CANONICAL_CBOR", "truncated byte string")
            })?;
            *offset = end;
            Ok(Cbor::Bytes(value.to_vec()))
        }
        4 => {
            let length = usize::try_from(decode_head(payload, offset, additional)?).map_err(|_| {
                OracleError::new("REJECT_Q1_CANONICAL_CBOR", "array length exceeds usize")
            })?;
            let mut values = Vec::with_capacity(length);
            for _ in 0..length {
                values.push(decode_item(payload, offset)?);
            }
            Ok(Cbor::Array(values))
        }
        7 if additional == 20 => Ok(Cbor::Bool(false)),
        7 if additional == 21 => Ok(Cbor::Bool(true)),
        7 if additional == 22 => Ok(Cbor::Null),
        _ => Err(OracleError::new(
            "REJECT_Q1_CANONICAL_CBOR",
            "unsupported CBOR major type or simple value",
        )),
    }
}

fn decode_strict(payload: &[u8]) -> Result<Cbor, OracleError> {
    let mut offset = 0;
    let value = decode_item(payload, &mut offset)?;
    if offset != payload.len() || encode(&value) != payload {
        return Err(OracleError::new(
            "REJECT_Q1_CANONICAL_CBOR",
            "CBOR payload is trailing or noncanonical",
        ));
    }
    Ok(value)
}

fn sha256(parts: &[&[u8]]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update(part);
    }
    hasher.finalize().into()
}

fn content_hash(domain: &[u8], object: &Cbor) -> [u8; 32] {
    sha256(&[domain, &[0], &encode(object)])
}

fn register_preimage(
    registry: &mut BTreeMap<[u8; 32], Vec<u8>>,
    digest: [u8; 32],
    preimage: Vec<u8>,
    label: &str,
) -> Result<(), OracleError> {
    if let Some(previous) = registry.get(&digest) {
        if previous != &preimage {
            return Err(OracleError::new(
                "FAIL_SHA256_PREIMAGE_COLLISION",
                format!("{label} has different preimages"),
            ));
        }
        return Err(OracleError::new(
            "REJECT_Q1_RECORD_SET_DUPLICATE",
            format!("{label} repeats exactly"),
        ));
    }
    registry.insert(digest, preimage);
    Ok(())
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

pub fn hex_encode(value: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(value.len() * 2);
    for byte in value {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 15) as usize] as char);
    }
    output
}

fn odd_input_object(bits: &[u8]) -> Cbor {
    array([
        uint(1),
        uint(0x3401),
        bytes(b"hegel-odd-input/1"),
        uint(bits.len() as u64),
        array(bits.iter().map(|bit| uint(u64::from(*bit)))),
    ])
}

fn sink_input_object(values: [u8; 4]) -> Cbor {
    array(
        [uint(1), uint(0x3402), bytes(b"hegel-sink-input/1")]
            .into_iter()
            .chain(values.into_iter().map(|value| uint(u64::from(value)))),
    )
}

fn universe_row(index: usize, signature: u64, input: Cbor) -> Vec<u8> {
    encode(&array([
        uint(1),
        uint(0x3201),
        bytes(b"hegel-bounded-universe-row/1"),
        uint(index as u64),
        uint(signature),
        input,
    ]))
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Observation {
    Odd(Vec<u8>),
    Sink([Rational; 4]),
}

fn odd_universe() -> (Vec<Observation>, Vec<Vec<u8>>) {
    let mut observations = Vec::with_capacity(480);
    let mut rows = Vec::with_capacity(480);
    for size in 5_usize..=8 {
        for numeric in 0_u16..(1_u16 << size) {
            let bits = (0..size)
                .map(|offset| ((numeric >> (size - offset - 1)) & 1) as u8)
                .collect::<Vec<_>>();
            rows.push(universe_row(rows.len(), 1, odd_input_object(&bits)));
            observations.push(Observation::Odd(bits));
        }
    }
    (observations, rows)
}

fn sink_universe() -> (Vec<Observation>, Vec<Vec<u8>>) {
    let mut observations = Vec::with_capacity(85);
    let mut rows = Vec::with_capacity(85);
    for a in 0_i8..=4 {
        for b in 0_i8..=4 {
            for c in 0_i8..=4 {
                let d = a + b - c;
                if !(0..=4).contains(&d) {
                    continue;
                }
                let values = [a as u8, b as u8, c as u8, d as u8];
                rows.push(universe_row(rows.len(), 2, sink_input_object(values)));
                observations.push(Observation::Sink([
                    Rational::integer(a as i64),
                    Rational::integer(b as i64),
                    Rational::integer(c as i64),
                    Rational::integer(d as i64),
                ]));
            }
        }
    }
    (observations, rows)
}

pub fn regenerated_universe_roots() -> (String, String) {
    let (odd_root, sink_root) = regenerated_universe_root_bytes();
    (hex_encode(&odd_root), hex_encode(&sink_root))
}

fn regenerated_universe_root_bytes() -> ([u8; 32], [u8; 32]) {
    let (_, odd_rows) = odd_universe();
    let (_, sink_rows) = sink_universe();
    (rfc6962(&odd_rows), rfc6962(&sink_rows))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Rational {
    numerator: i64,
    denominator: i64,
}

impl Rational {
    fn new(numerator: i64, denominator: i64) -> Result<Self, OracleError> {
        if denominator == 0 {
            return Err(OracleError::new("FAIL_SEMANTICS_MISMATCH", "zero denominator"));
        }
        let (mut numerator, mut denominator) = (numerator, denominator);
        if denominator < 0 {
            numerator = numerator
                .checked_neg()
                .ok_or_else(|| OracleError::new("FAIL_SEMANTICS_MISMATCH", "rational overflow"))?;
            denominator = denominator
                .checked_neg()
                .ok_or_else(|| OracleError::new("FAIL_SEMANTICS_MISMATCH", "rational overflow"))?;
        }
        let divisor = gcd(numerator.unsigned_abs(), denominator as u64) as i64;
        Ok(Self {
            numerator: numerator / divisor,
            denominator: denominator / divisor,
        })
    }

    const fn integer(value: i64) -> Self {
        Self {
            numerator: value,
            denominator: 1,
        }
    }

    fn add(self, other: Self) -> Option<Self> {
        let numerator = (self.numerator as i128)
            .checked_mul(other.denominator as i128)?
            .checked_add((other.numerator as i128).checked_mul(self.denominator as i128)?)?;
        let denominator = (self.denominator as i128).checked_mul(other.denominator as i128)?;
        Self::new(i64::try_from(numerator).ok()?, i64::try_from(denominator).ok()?).ok()
    }

    fn difference(self, other: Self) -> Option<Self> {
        self.add(Self::new(other.numerator.checked_neg()?, other.denominator).ok()?)
    }

    fn absolute(self) -> Option<Self> {
        Self::new(self.numerator.checked_abs()?, self.denominator).ok()
    }

    fn compare(self, other: Self) -> Ordering {
        ((self.numerator as i128) * (other.denominator as i128))
            .cmp(&((other.numerator as i128) * (self.denominator as i128)))
    }

    fn in_grid(self) -> bool {
        self.numerator.abs() <= 64 && (1..=8).contains(&self.denominator)
    }
}

fn gcd(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        (left, right) = (right, left % right);
    }
    left.max(1)
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum RuntimeValue {
    Bottom,
    Bool(bool),
    Bit(u8),
    Sign(i8),
    BoundedInt(i8),
    Rational(Rational),
}

impl RuntimeValue {
    fn bottom(&self) -> bool {
        matches!(self, Self::Bottom)
    }
}

impl Observation {
    fn set_size(&self) -> i8 {
        match self {
            Self::Odd(bits) => bits.len() as i8,
            Self::Sink(_) => 4,
        }
    }

    fn bit_at(&self, index: u64) -> RuntimeValue {
        match self {
            Self::Odd(bits) => bits
                .get(index as usize)
                .copied()
                .map(RuntimeValue::Bit)
                .unwrap_or(RuntimeValue::Bottom),
            Self::Sink(_) => RuntimeValue::Bottom,
        }
    }

    fn aggregate(
        &self,
        map_id: u64,
        scope_id: u64,
        quantity_id: u64,
        extension: &[(u64, bool)],
    ) -> RuntimeValue {
        let Self::Sink(values) = self else {
            return RuntimeValue::Bottom;
        };
        if scope_id != 3 || quantity_id != 0 || !extension.is_empty() {
            return RuntimeValue::Bottom;
        }
        match map_id {
            0 => values
                .iter()
                .try_fold(Rational::integer(0), |sum, value| sum.add(*value))
                .map(bounded_rational)
                .unwrap_or(RuntimeValue::Bottom),
            1 => RuntimeValue::BoundedInt(
                values.iter().filter(|value| value.numerator != 0).count() as i8,
            ),
            5 => [1_i64, 1, -1, -1]
                .into_iter()
                .zip(values)
                .try_fold(Rational::integer(0), |sum, (orientation, value)| {
                    sum.add(Rational::new(value.numerator * orientation, value.denominator).ok()?)
                })
                .map(bounded_rational)
                .unwrap_or(RuntimeValue::Bottom),
            _ => RuntimeValue::Bottom,
        }
    }
}

fn bounded_rational(value: Rational) -> RuntimeValue {
    if value.in_grid() {
        RuntimeValue::Rational(value)
    } else {
        RuntimeValue::Bottom
    }
}

fn evaluate(node: &Node, observation: &Observation) -> Result<RuntimeValue, OracleError> {
    match node {
        Node::ScalarConst(1) => Ok(RuntimeValue::Rational(Rational::integer(-1))),
        Node::ScalarConst(3) => Ok(RuntimeValue::Rational(Rational::integer(0))),
        Node::ScalarConst(5) => Ok(RuntimeValue::Rational(Rational::integer(1))),
        Node::ScalarConst(id) => Err(OracleError::new(
            "FAIL_SEMANTICS_MISMATCH",
            format!("inactive rational parameter {id}"),
        )),
        Node::BitAt(index) if *index < 8 => Ok(observation.bit_at(*index)),
        Node::BitAt(index) => Err(OracleError::new(
            "FAIL_SEMANTICS_MISMATCH",
            format!("invalid bit index {index}"),
        )),
        Node::SetSize => Ok(RuntimeValue::BoundedInt(observation.set_size())),
        Node::Aggregate {
            map_id,
            scope_id,
            quantity_id,
            scope_extension,
        } if [0, 1, 5].contains(map_id) => Ok(observation.aggregate(
            *map_id,
            *scope_id,
            *quantity_id,
            scope_extension,
        )),
        Node::Aggregate { map_id, .. } => Err(OracleError::new(
            "FAIL_SEMANTICS_MISMATCH",
            format!("removed aggregate map {map_id}"),
        )),
        Node::ContextFlag(id) if *id < 4 => Ok(RuntimeValue::Bottom),
        Node::TaskFlag(id) if *id < 2 => Ok(RuntimeValue::Bottom),
        Node::ContextFlag(id) | Node::TaskFlag(id) => Err(OracleError::new(
            "FAIL_SEMANTICS_MISMATCH",
            format!("invalid flag {id}"),
        )),
        Node::NewSymbolCall(_) => Err(OracleError::new(
            "FAIL_SEMANTICS_MISMATCH",
            "new symbol reached old DSL evaluator",
        )),
        Node::Unary { op, child } => {
            let child = evaluate(child, observation)?;
            if child.bottom() {
                return Ok(RuntimeValue::Bottom);
            }
            match (op, child) {
                (UnaryOp::BitToScalar, RuntimeValue::Bit(value)) => {
                    Ok(RuntimeValue::Rational(Rational::integer(i64::from(value))))
                }
                (UnaryOp::IntToScalar, RuntimeValue::BoundedInt(value)) => {
                    Ok(RuntimeValue::Rational(Rational::integer(i64::from(value))))
                }
                (UnaryOp::Absolute, RuntimeValue::Rational(value)) => Ok(value
                    .absolute()
                    .map(bounded_rational)
                    .unwrap_or(RuntimeValue::Bottom)),
                (UnaryOp::Sign, RuntimeValue::Rational(value)) => {
                    Ok(RuntimeValue::Sign(value.numerator.signum() as i8))
                }
                _ => Err(OracleError::new(
                    "FAIL_SEMANTICS_MISMATCH",
                    "unary type mismatch",
                )),
            }
        }
        Node::Binary { op, left, right } => {
            let left = evaluate(left, observation)?;
            let right = evaluate(right, observation)?;
            if left.bottom() || right.bottom() {
                return Ok(RuntimeValue::Bottom);
            }
            match (op, left, right) {
                (
                    BinaryOp::Difference,
                    RuntimeValue::Rational(left),
                    RuntimeValue::Rational(right),
                ) => Ok(left
                    .difference(right)
                    .map(bounded_rational)
                    .unwrap_or(RuntimeValue::Bottom)),
                (
                    BinaryOp::EqualExact,
                    RuntimeValue::Rational(left),
                    RuntimeValue::Rational(right),
                ) => Ok(RuntimeValue::Bool(left == right)),
                (
                    BinaryOp::LessEqual,
                    RuntimeValue::Rational(left),
                    RuntimeValue::Rational(right),
                ) => Ok(RuntimeValue::Bool(left.compare(right) != Ordering::Greater)),
                (
                    BinaryOp::SameSign,
                    RuntimeValue::Sign(left),
                    RuntimeValue::Sign(right),
                ) => Ok(RuntimeValue::Bool(left == right)),
                (
                    BinaryOp::OppositeSign,
                    RuntimeValue::Sign(left),
                    RuntimeValue::Sign(right),
                ) => Ok(RuntimeValue::Bool(left != 0 && left == -right)),
                (BinaryOp::Add | BinaryOp::GreaterEqual, _, _) => Err(OracleError::new(
                    "FAIL_SEMANTICS_MISMATCH",
                    "noncanonical binary alias",
                )),
                _ => Err(OracleError::new(
                    "FAIL_SEMANTICS_MISMATCH",
                    "binary type mismatch",
                )),
            }
        }
        Node::ApproxEqual {
            left,
            right,
            tolerance_index,
        } => {
            let left = evaluate(left, observation)?;
            let right = evaluate(right, observation)?;
            if left.bottom() || right.bottom() {
                return Ok(RuntimeValue::Bottom);
            }
            let tolerance = match tolerance_index {
                1 => Rational::new(1, 4)?,
                2 => Rational::new(1, 2)?,
                _ => {
                    return Err(OracleError::new(
                        "FAIL_SEMANTICS_MISMATCH",
                        "removed tolerance",
                    ))
                }
            };
            match (left, right) {
                (RuntimeValue::Rational(left), RuntimeValue::Rational(right)) => {
                    let distance = left.difference(right).and_then(Rational::absolute).ok_or_else(
                        || OracleError::new("FAIL_SEMANTICS_MISMATCH", "rational overflow"),
                    )?;
                    Ok(RuntimeValue::Bool(
                        distance.compare(tolerance) != Ordering::Greater,
                    ))
                }
                _ => Err(OracleError::new(
                    "FAIL_SEMANTICS_MISMATCH",
                    "approx type mismatch",
                )),
            }
        }
        Node::And(children) if children.len() == 2 => {
            let mut result = true;
            for child in children {
                match evaluate(child, observation)? {
                    RuntimeValue::Bottom => return Ok(RuntimeValue::Bottom),
                    RuntimeValue::Bool(value) => result &= value,
                    _ => {
                        return Err(OracleError::new(
                            "FAIL_SEMANTICS_MISMATCH",
                            "AND2 type mismatch",
                        ))
                    }
                }
            }
            Ok(RuntimeValue::Bool(result))
        }
        Node::And(_) => Err(OracleError::new(
            "FAIL_SEMANTICS_MISMATCH",
            "non-AND2 node",
        )),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[repr(u64)]
pub enum OutputSortId {
    Bool = 1,
    Bit = 2,
    Sign = 3,
    BoundedInt = 4,
    RationalValue = 5,
}

impl OutputSortId {
    fn from_sort(sort: Sort) -> Self {
        match sort {
            Sort::Bool => Self::Bool,
            Sort::Bit => Self::Bit,
            Sort::Sign => Self::Sign,
            Sort::BoundedInt => Self::BoundedInt,
            Sort::RationalValue => Self::RationalValue,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Behavior {
    input_signature_id: u64,
    universe_root: [u8; 32],
    output_sort: OutputSortId,
    cells: Vec<RuntimeValue>,
}

impl Behavior {
    fn cell(&self, value: &RuntimeValue) -> Result<Cbor, OracleError> {
        if value.bottom() {
            return Ok(array([uint(0)]));
        }
        let payload = match (self.output_sort, value) {
            (OutputSortId::Bool, RuntimeValue::Bool(value)) => Cbor::Bool(*value),
            (OutputSortId::Bit, RuntimeValue::Bit(value)) if *value <= 1 => {
                uint(u64::from(*value))
            }
            (OutputSortId::Sign, RuntimeValue::Sign(value)) if (-1..=1).contains(value) => {
                int(i64::from(*value))
            }
            (OutputSortId::BoundedInt, RuntimeValue::BoundedInt(value))
                if (-8..=8).contains(value) =>
            {
                int(i64::from(*value))
            }
            (OutputSortId::RationalValue, RuntimeValue::Rational(value)) if value.in_grid() => {
                array([int(value.numerator), uint(value.denominator as u64)])
            }
            _ => {
                return Err(OracleError::new(
                    "FAIL_SEMANTICS_MISMATCH",
                    "typed behavior cell mismatch",
                ))
            }
        };
        Ok(array([uint(1), payload]))
    }

    fn object(&self) -> Result<Cbor, OracleError> {
        Ok(array([
            uint(1),
            uint(BEHAVIOR_TAG),
            bytes(BEHAVIOR_SCHEMA),
            uint(self.input_signature_id),
            bytes(self.universe_root),
            uint(self.output_sort as u64),
            uint(self.cells.len() as u64),
            array(
                self.cells
                    .iter()
                    .map(|cell| self.cell(cell))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
        ]))
    }

    fn canonical_bytes(&self) -> Result<Vec<u8>, OracleError> {
        Ok(encode(&self.object()?))
    }

    fn id(&self) -> Result<[u8; 32], OracleError> {
        Ok(content_hash(BEHAVIOR_ID_DOMAIN, &self.object()?))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u64)]
enum NormalizationProfile {
    General = 0,
    AbsoluteRoot = 1,
    ConstNegativeOne = 2,
    ConstZero = 3,
    ConstPositiveOne = 4,
    TopLevelAnd2 = 5,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ConstructionSignature {
    output_sort: OutputSortId,
    ast_depth: u32,
    ast_node_count: u32,
    scalar_occurrence_count: u32,
    aggregate_leaf_count: u32,
    distinct_bit_slot_bitmap: u8,
    scope_clause_count: u32,
    top_level_clause_count: u32,
    old_law_depth: u32,
    normalization_profile: NormalizationProfile,
    mdl_q32: u64,
}

impl ConstructionSignature {
    fn object(&self) -> Cbor {
        array([
            uint(1),
            uint(SIGNATURE_TAG),
            bytes(SIGNATURE_SCHEMA),
            uint(self.output_sort as u64),
            uint(u64::from(self.ast_depth)),
            uint(u64::from(self.ast_node_count)),
            uint(u64::from(self.scalar_occurrence_count)),
            uint(u64::from(self.aggregate_leaf_count)),
            uint(u64::from(self.distinct_bit_slot_bitmap)),
            uint(u64::from(self.scope_clause_count)),
            uint(u64::from(self.top_level_clause_count)),
            uint(u64::from(self.old_law_depth)),
            uint(self.normalization_profile as u64),
            uint(self.mdl_q32),
        ])
    }

    fn canonical_bytes(&self) -> Vec<u8> {
        encode(&self.object())
    }

    fn id(&self) -> [u8; 32] {
        content_hash(SIGNATURE_ID_DOMAIN, &self.object())
    }

    fn dominates(&self, other: &Self) -> bool {
        if self.output_sort != other.output_sort
            || self.normalization_profile != other.normalization_profile
        {
            return false;
        }
        let subset = (self.distinct_bit_slot_bitmap | other.distinct_bit_slot_bitmap)
            == other.distinct_bit_slot_bitmap;
        let left = [
            u64::from(self.ast_depth),
            u64::from(self.ast_node_count),
            u64::from(self.scalar_occurrence_count),
            u64::from(self.aggregate_leaf_count),
            u64::from(self.scope_clause_count),
            u64::from(self.top_level_clause_count),
            u64::from(self.old_law_depth),
            self.mdl_q32,
        ];
        let right = [
            u64::from(other.ast_depth),
            u64::from(other.ast_node_count),
            u64::from(other.scalar_occurrence_count),
            u64::from(other.aggregate_leaf_count),
            u64::from(other.scope_clause_count),
            u64::from(other.top_level_clause_count),
            u64::from(other.old_law_depth),
            other.mdl_q32,
        ];
        let no_worse = subset && left.iter().zip(right).all(|(left, right)| *left <= right);
        let strict = self.distinct_bit_slot_bitmap != other.distinct_bit_slot_bitmap
            || left.iter().zip(right).any(|(left, right)| *left < right);
        no_worse && strict
    }
}

fn elias_delta_length(value: u64) -> Result<u64, OracleError> {
    if value == 0 {
        return Err(OracleError::new(
            "FAIL_SEMANTICS_MISMATCH",
            "Elias-delta value is one-based",
        ));
    }
    let log_n = 63 - u64::from(value.leading_zeros());
    let log_log = 63 - u64::from((log_n + 1).leading_zeros());
    Ok(log_n + 2 * log_log + 1)
}

fn mdl_bits(node: &Node) -> Result<u64, OracleError> {
    match node {
        Node::ScalarConst(id) if [1, 3, 5].contains(id) => Ok(8),
        Node::BitAt(id) if *id < 8 => Ok(5 + elias_delta_length(id + 1)?),
        Node::SetSize => Ok(5),
        Node::Aggregate {
            map_id,
            scope_id,
            quantity_id,
            scope_extension,
        } if [0, 1, 5].contains(map_id)
            && *scope_id < 4
            && *quantity_id < 2
            && scope_extension.len() <= 2 =>
        {
            let clause_count_bits = if scope_extension.is_empty() { 1 } else { 2 };
            Ok(11 + clause_count_bits + 3 * scope_extension.len() as u64)
        }
        Node::ContextFlag(id) if *id < 4 => Ok(5 + elias_delta_length(id + 1)?),
        Node::TaskFlag(id) if *id < 2 => Ok(5 + elias_delta_length(id + 1)?),
        Node::Unary { child, .. } => Ok(4 + mdl_bits(child)?),
        Node::Binary { left, right, .. } => Ok(5 + mdl_bits(left)? + mdl_bits(right)?),
        Node::ApproxEqual {
            left,
            right,
            tolerance_index,
        } if (1..=2).contains(tolerance_index) => {
            Ok(6 + mdl_bits(left)? + mdl_bits(right)?)
        }
        Node::And(children) if children.len() == 2 => Ok(
            5 + children
                .iter()
                .map(mdl_bits)
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .sum::<u64>(),
        ),
        _ => Err(OracleError::new(
            "FAIL_SEMANTICS_MISMATCH",
            "AST has no frozen MDL code",
        )),
    }
}

#[derive(Debug, Default)]
struct ResourceMetrics {
    bitmask: u8,
    scope_clauses: u32,
}

fn collect_metrics(node: &Node, metrics: &mut ResourceMetrics) {
    match node {
        Node::BitAt(index) if *index < 8 => metrics.bitmask |= 1 << *index,
        Node::Aggregate {
            scope_extension, ..
        } => metrics.scope_clauses += scope_extension.len() as u32,
        Node::Unary { child, .. } => collect_metrics(child, metrics),
        Node::Binary { left, right, .. } | Node::ApproxEqual { left, right, .. } => {
            collect_metrics(left, metrics);
            collect_metrics(right, metrics);
        }
        Node::And(children) => {
            for child in children {
                collect_metrics(child, metrics);
            }
        }
        _ => {}
    }
}

fn normalization_profile(node: &Node) -> NormalizationProfile {
    match node {
        Node::Unary {
            op: UnaryOp::Absolute,
            ..
        } => NormalizationProfile::AbsoluteRoot,
        Node::ScalarConst(1) => NormalizationProfile::ConstNegativeOne,
        Node::ScalarConst(3) => NormalizationProfile::ConstZero,
        Node::ScalarConst(5) => NormalizationProfile::ConstPositiveOne,
        Node::And(children) if children.len() == 2 => NormalizationProfile::TopLevelAnd2,
        _ => NormalizationProfile::General,
    }
}

fn signature(program: &CanonicalProgram) -> Result<ConstructionSignature, OracleError> {
    let mut metrics = ResourceMetrics::default();
    collect_metrics(&program.canonical_node, &mut metrics);
    Ok(ConstructionSignature {
        output_sort: OutputSortId::from_sort(program.output_sort),
        ast_depth: program.depth,
        ast_node_count: program.node_count,
        scalar_occurrence_count: program.scalar_parameter_occurrence_count,
        aggregate_leaf_count: program.aggregate_leaf_count,
        distinct_bit_slot_bitmap: metrics.bitmask,
        scope_clause_count: metrics.scope_clauses,
        top_level_clause_count: match &program.canonical_node {
            Node::And(children) => children.len() as u32,
            _ => 0,
        },
        old_law_depth: 0,
        normalization_profile: normalization_profile(&program.canonical_node),
        mdl_q32: mdl_bits(&program.canonical_node)? << 32,
    })
}

#[derive(Debug, Clone)]
struct Program {
    canonical: CanonicalProgram,
    behavior: Behavior,
    signature: ConstructionSignature,
    program_id: [u8; 32],
}

fn program_identity_object(
    input_signature_id: u64,
    universe_root: [u8; 32],
    canonical: &CanonicalProgram,
    signature: &ConstructionSignature,
) -> Cbor {
    array([
        uint(input_signature_id),
        bytes(universe_root),
        bytes(&canonical.canonical_cbor),
        bytes(canonical.canonical_ast_hash),
        signature.object(),
    ])
}

impl Program {
    fn new(
        canonical: CanonicalProgram,
        input_signature_id: u64,
        universe_root: [u8; 32],
        observations: &[Observation],
    ) -> Result<Self, OracleError> {
        let signature = signature(&canonical)?;
        let behavior = Behavior {
            input_signature_id,
            universe_root,
            output_sort: signature.output_sort,
            cells: observations
                .iter()
                .map(|observation| evaluate(&canonical.canonical_node, observation))
                .collect::<Result<Vec<_>, _>>()?,
        };
        behavior.canonical_bytes()?;
        let program_preimage =
            program_identity_object(input_signature_id, universe_root, &canonical, &signature);
        let program_id = content_hash(PROGRAM_ID_DOMAIN, &program_preimage);
        Ok(Self {
            canonical,
            behavior,
            signature,
            program_id,
        })
    }
}

fn scope_extensions() -> Vec<Vec<(u64, bool)>> {
    let mut extensions = vec![Vec::new()];
    for context in 0_u64..4 {
        for expected in [false, true] {
            extensions.push(vec![(context, expected)]);
        }
    }
    for first in 0_u64..4 {
        for second in (first + 1)..4 {
            for first_expected in [false, true] {
                for second_expected in [false, true] {
                    extensions.push(vec![(first, first_expected), (second, second_expected)]);
                }
            }
        }
    }
    debug_assert_eq!(extensions.len(), 33);
    extensions
}

fn full_leaf_sources() -> Vec<Node> {
    let mut leaves = vec![
        Node::ScalarConst(1),
        Node::ScalarConst(3),
        Node::ScalarConst(5),
    ];
    leaves.extend((0_u64..8).map(Node::BitAt));
    leaves.push(Node::SetSize);
    let extensions = scope_extensions();
    for map_id in [0_u64, 1, 5] {
        for scope_id in 0_u64..4 {
            for quantity_id in 0_u64..2 {
                for scope_extension in &extensions {
                    leaves.push(Node::Aggregate {
                        map_id,
                        scope_id,
                        quantity_id,
                        scope_extension: scope_extension.clone(),
                    });
                }
            }
        }
    }
    leaves.extend((0_u64..4).map(Node::ContextFlag));
    leaves.extend((0_u64..2).map(Node::TaskFlag));
    debug_assert_eq!(leaves.len(), FROZEN_LEAF_COUNT);
    leaves
}

#[derive(Debug, Clone)]
struct FrozenLeaf {
    coverage_code: u16,
    canonical: CanonicalProgram,
}

fn frozen_leaf_manifest() -> Result<Vec<FrozenLeaf>, OracleError> {
    let mut leaves = full_leaf_sources()
        .into_iter()
        .map(|source| {
            canonicalize_shrink6_source_node(source)
                .map_err(|error| OracleError::new(error.code, error.message))
        })
        .collect::<Result<Vec<_>, _>>()?;
    leaves.sort_by(|left, right| {
        (OutputSortId::from_sort(left.output_sort) as u64)
            .cmp(&(OutputSortId::from_sort(right.output_sort) as u64))
            .then_with(|| left.root_operator_id.cmp(&right.root_operator_id))
            .then_with(|| left.canonical_cbor.cmp(&right.canonical_cbor))
    });
    leaves.dedup_by(|left, right| left.canonical_cbor == right.canonical_cbor);
    if leaves.len() != FROZEN_LEAF_COUNT {
        return Err(OracleError::new(
            "FAIL_SEMANTICS_MISMATCH",
            format!("full leaf manifest has {} rows", leaves.len()),
        ));
    }
    Ok(leaves
        .into_iter()
        .enumerate()
        .map(|(coverage_code, canonical)| FrozenLeaf {
            coverage_code: coverage_code as u16,
            canonical,
        })
        .collect())
}

fn admit(
    source: Node,
    input_signature_id: u64,
    universe_root: [u8; 32],
    observations: &[Observation],
) -> Result<Option<Program>, OracleError> {
    let canonical = match canonicalize_shrink6_source_node(source) {
        Ok(program) => program,
        Err(error) if error.code == "REJECT_STRUCTURAL_LIMIT" => return Ok(None),
        Err(error) => return Err(OracleError::new(error.code, error.message)),
    };
    if canonical.depth > MAX_AST_DEPTH
        || canonical.node_count > NODE3_AST_NODE_LIMIT
        || canonical.aggregate_leaf_count > MAX_AGGREGATE_LEAVES
        || canonical.scalar_parameter_occurrence_count > MAX_SCALAR_OCCURRENCES
    {
        return Ok(None);
    }
    let mut metrics = ResourceMetrics::default();
    collect_metrics(&canonical.canonical_node, &mut metrics);
    if metrics.scope_clauses > MAX_SCOPE_CLAUSES
        || metrics.bitmask.count_ones() > MAX_DISTINCT_BITS
        || matches!(&canonical.canonical_node, Node::And(children) if children.len() != 2)
    {
        return Ok(None);
    }
    Ok(Some(Program::new(
        canonical,
        input_signature_id,
        universe_root,
        observations,
    )?))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Operator {
    Unary(UnaryOp),
    Binary(BinaryOp),
    Approx(u64),
    And2,
}

impl Operator {
    fn coverage_code(self) -> u16 {
        match self {
            Self::Unary(UnaryOp::BitToScalar) => 0x1000,
            Self::Unary(UnaryOp::IntToScalar) => 0x1001,
            Self::Unary(UnaryOp::Absolute) => 0x1002,
            Self::Unary(UnaryOp::Sign) => 0x1003,
            Self::Binary(BinaryOp::Difference) => 0x2001,
            Self::Binary(BinaryOp::EqualExact) => 0x2002,
            Self::Binary(BinaryOp::LessEqual) => 0x2003,
            Self::Binary(BinaryOp::SameSign) => 0x2005,
            Self::Binary(BinaryOp::OppositeSign) => 0x2006,
            Self::Approx(1) => 0x3001,
            Self::Approx(2) => 0x3002,
            Self::And2 => 0x4002,
            Self::Binary(BinaryOp::Add | BinaryOp::GreaterEqual) | Self::Approx(_) => {
                unreachable!("removed operator")
            }
        }
    }

    fn parameters(self) -> Cbor {
        match self {
            Self::Approx(tolerance) => array([uint(tolerance)]),
            _ => array([]),
        }
    }

    fn child_sorts(self) -> &'static [OutputSortId] {
        match self {
            Self::Unary(UnaryOp::BitToScalar) => &[OutputSortId::Bit],
            Self::Unary(UnaryOp::IntToScalar) => &[OutputSortId::BoundedInt],
            Self::Unary(UnaryOp::Absolute | UnaryOp::Sign) => &[OutputSortId::RationalValue],
            Self::Binary(BinaryOp::Difference)
            | Self::Binary(BinaryOp::EqualExact)
            | Self::Binary(BinaryOp::LessEqual)
            | Self::Approx(_) => &[OutputSortId::RationalValue, OutputSortId::RationalValue],
            Self::Binary(BinaryOp::SameSign | BinaryOp::OppositeSign) => {
                &[OutputSortId::Sign, OutputSortId::Sign]
            }
            Self::And2 => &[OutputSortId::Bool, OutputSortId::Bool],
            Self::Binary(BinaryOp::Add | BinaryOp::GreaterEqual) => unreachable!(),
        }
    }

    fn commutative(self) -> bool {
        matches!(
            self,
            Self::Binary(BinaryOp::EqualExact)
                | Self::Binary(BinaryOp::SameSign)
                | Self::Binary(BinaryOp::OppositeSign)
                | Self::Approx(_)
                | Self::And2
        )
    }

    fn node(self, children: &[Program]) -> Node {
        match self {
            Self::Unary(op) => Node::Unary {
                op,
                child: Box::new(children[0].canonical.canonical_node.clone()),
            },
            Self::Binary(op) => Node::Binary {
                op,
                left: Box::new(children[0].canonical.canonical_node.clone()),
                right: Box::new(children[1].canonical.canonical_node.clone()),
            },
            Self::Approx(tolerance_index) => Node::ApproxEqual {
                left: Box::new(children[0].canonical.canonical_node.clone()),
                right: Box::new(children[1].canonical.canonical_node.clone()),
                tolerance_index,
            },
            Self::And2 => {
                Node::And(
                    children
                        .iter()
                        .map(|child| child.canonical.canonical_node.clone())
                        .collect(),
                )
            }
        }
    }
}

const OPERATORS: [Operator; 12] = [
    Operator::Unary(UnaryOp::BitToScalar),
    Operator::Unary(UnaryOp::IntToScalar),
    Operator::Unary(UnaryOp::Absolute),
    Operator::Unary(UnaryOp::Sign),
    Operator::Binary(BinaryOp::Difference),
    Operator::Binary(BinaryOp::EqualExact),
    Operator::Binary(BinaryOp::LessEqual),
    Operator::Binary(BinaryOp::SameSign),
    Operator::Binary(BinaryOp::OppositeSign),
    Operator::Approx(1),
    Operator::Approx(2),
    Operator::And2,
];

fn capacity(sort: OutputSortId) -> usize {
    match sort {
        OutputSortId::Bool | OutputSortId::RationalValue => 2,
        OutputSortId::Bit | OutputSortId::Sign | OutputSortId::BoundedInt => 1,
    }
}

fn resource_eligible(operator: Operator, children: &[Program], depth: u32) -> bool {
    if children.len() != operator.child_sorts().len()
        || children
            .iter()
            .zip(operator.child_sorts())
            .any(|(child, sort)| child.signature.output_sort != *sort)
        || children
            .iter()
            .any(|child| matches!(child.canonical.canonical_node, Node::And(_)))
    {
        return false;
    }
    let resulting_depth = 1 + children
        .iter()
        .map(|child| child.signature.ast_depth)
        .max()
        .unwrap_or(0);
    let nodes = 1 + children
        .iter()
        .map(|child| child.signature.ast_node_count)
        .sum::<u32>();
    let aggregates = children
        .iter()
        .map(|child| child.signature.aggregate_leaf_count)
        .sum::<u32>();
    let scalars = children
        .iter()
        .map(|child| child.signature.scalar_occurrence_count)
        .sum::<u32>();
    let clauses = children
        .iter()
        .map(|child| child.signature.scope_clause_count)
        .sum::<u32>();
    let bitmask = children.iter().fold(0_u8, |mask, child| {
        mask | child.signature.distinct_bit_slot_bitmap
    });
    resulting_depth == depth
        && resulting_depth <= MAX_AST_DEPTH
        && nodes <= NODE3_AST_NODE_LIMIT
        && aggregates <= MAX_AGGREGATE_LEAVES
        && scalars <= MAX_SCALAR_OCCURRENCES
        && clauses <= MAX_SCOPE_CLAUSES
        && bitmask.count_ones() <= MAX_DISTINCT_BITS
}

#[derive(Debug, Clone)]
struct Application {
    operator: Operator,
    children: Vec<Program>,
}

impl Application {
    fn source_node(&self) -> Node {
        self.operator.node(&self.children)
    }
}

fn child_order_key(program: &Program) -> ([u8; 32], &[u8]) {
    let node_bytes = &program.canonical.canonical_cbor[2..];
    (sha256(&[node_bytes]), node_bytes)
}

fn eligible_applications(programs: &[Program], depth: u32) -> Vec<Application> {
    let mut by_sort: BTreeMap<OutputSortId, Vec<Program>> = BTreeMap::new();
    for program in programs {
        by_sort
            .entry(program.signature.output_sort)
            .or_default()
            .push(program.clone());
    }
    for programs in by_sort.values_mut() {
        programs.sort_by(|left, right| {
            child_order_key(left)
                .cmp(&child_order_key(right))
                .then_with(|| left.canonical.canonical_cbor.cmp(&right.canonical.canonical_cbor))
        });
    }
    let mut applications = Vec::new();
    for operator in OPERATORS {
        let sorts = operator.child_sorts();
        if sorts.len() == 1 {
            for child in by_sort.get(&sorts[0]).into_iter().flatten() {
                let tuple = vec![child.clone()];
                if resource_eligible(operator, &tuple, depth) {
                    applications.push(Application {
                        operator,
                        children: tuple,
                    });
                }
            }
            continue;
        }
        let Some(lefts) = by_sort.get(&sorts[0]) else {
            continue;
        };
        let Some(rights) = by_sort.get(&sorts[1]) else {
            continue;
        };
        if operator.commutative() && sorts[0] == sorts[1] {
            let mut ordered_lefts;
            let (lefts, rights) = if matches!(operator, Operator::And2) {
                ordered_lefts = lefts.clone();
                ordered_lefts.sort_by_key(program_sort_key);
                (&ordered_lefts, &ordered_lefts)
            } else {
                (lefts, rights)
            };
            for left_index in 0..lefts.len() {
                let right_start = if matches!(operator, Operator::And2) {
                    left_index + 1
                } else {
                    left_index
                };
                for right_index in right_start..rights.len() {
                    let tuple = vec![lefts[left_index].clone(), rights[right_index].clone()];
                    if resource_eligible(operator, &tuple, depth) {
                        applications.push(Application {
                            operator,
                            children: tuple,
                        });
                    }
                }
            }
        } else {
            for left in lefts {
                for right in rights {
                    let tuple = vec![left.clone(), right.clone()];
                    if resource_eligible(operator, &tuple, depth) {
                        applications.push(Application {
                            operator,
                            children: tuple,
                        });
                    }
                }
            }
        }
    }
    applications
}

#[derive(Debug, Clone)]
struct QuotientClass {
    behavior_bytes: Vec<u8>,
    behavior: Behavior,
    cohorts: BTreeMap<Vec<u8>, Vec<Program>>,
}

impl QuotientClass {
    fn add(&mut self, program: Program) -> bool {
        let signature_bytes = program.signature.canonical_bytes();
        let cohort = self.cohorts.entry(signature_bytes).or_default();
        let before = cohort
            .iter()
            .map(|entry| entry.canonical.canonical_cbor.clone())
            .collect::<Vec<_>>();
        cohort.push(program);
        cohort.sort_by(|left, right| {
            left.canonical.canonical_cbor.cmp(&right.canonical.canonical_cbor)
        });
        cohort.dedup_by(|left, right| {
            left.canonical.canonical_cbor == right.canonical.canonical_cbor
        });
        cohort.truncate(capacity(cohort[0].signature.output_sort));
        before
            != cohort
                .iter()
                .map(|entry| entry.canonical.canonical_cbor.clone())
                .collect::<Vec<_>>()
    }

    fn visible_cohorts(&self) -> Vec<&Vec<Program>> {
        self.cohorts
            .values()
            .filter(|candidate| {
                !self.cohorts.values().any(|other| {
                    other[0].signature.dominates(&candidate[0].signature)
                        && other.len() >= candidate.len()
                })
            })
            .collect()
    }

    fn bank_count(&self) -> usize {
        self.cohorts.values().map(Vec::len).sum()
    }

    fn frontier_count(&self) -> usize {
        self.visible_cohorts().into_iter().map(Vec::len).sum()
    }
}

#[derive(Debug, Clone, Default)]
struct QuotientState {
    classes: BTreeMap<[u8; 32], QuotientClass>,
}

impl QuotientState {
    fn insert(&mut self, program: Program) -> Result<bool, OracleError> {
        let class_id = program.behavior.id()?;
        let behavior_bytes = program.behavior.canonical_bytes()?;
        if let Some(class) = self.classes.get_mut(&class_id) {
            if class.behavior_bytes != behavior_bytes {
                return Err(OracleError::new(
                    "FAIL_SHA256_PREIMAGE_COLLISION",
                    "different behavior preimages share one digest",
                ));
            }
            return Ok(class.add(program));
        }
        let mut class = QuotientClass {
            behavior_bytes,
            behavior: program.behavior.clone(),
            cohorts: BTreeMap::new(),
        };
        let changed = class.add(program);
        debug_assert!(changed);
        self.classes.insert(class_id, class);
        Ok(true)
    }

    fn continuation_programs(&self) -> Vec<Program> {
        let mut programs = BTreeMap::new();
        for program in self
            .classes
            .values()
            .flat_map(|class| class.cohorts.values())
            .flat_map(|cohort| cohort.iter())
        {
            programs.insert(program.canonical.canonical_cbor.clone(), program.clone());
        }
        programs.into_values().collect()
    }

    fn class_count(&self) -> usize {
        self.classes.len()
    }

    fn cohort_count(&self) -> usize {
        self.classes.values().map(|class| class.cohorts.len()).sum()
    }

    fn bank_count(&self) -> usize {
        self.classes.values().map(QuotientClass::bank_count).sum()
    }

    fn frontier_count(&self) -> usize {
        self.classes
            .values()
            .map(QuotientClass::frontier_count)
            .sum()
    }

    fn maximum_bank_per_class(&self) -> usize {
        self.classes
            .values()
            .map(QuotientClass::bank_count)
            .max()
            .unwrap_or(0)
    }

    fn maximum_frontier_per_class(&self) -> usize {
        self.classes
            .values()
            .map(QuotientClass::frontier_count)
            .max()
            .unwrap_or(0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct CoverageKey {
    depth: u8,
    code: u16,
}

#[derive(Debug, Clone, Default)]
struct CoverageAccumulator {
    eligible_keys: Vec<Cbor>,
    processed_keys: Vec<Cbor>,
    strict_admissions: Vec<([u8; 32], [u8; 32])>,
    canonical_asts: BTreeSet<Vec<u8>>,
    rewrite_collapses: u64,
}

impl CoverageAccumulator {
    fn add_eligible(&mut self, application: &Cbor) {
        self.eligible_keys.push(application.clone());
        self.processed_keys.push(application.clone());
    }

    fn add_admitted(
        &mut self,
        application_id: [u8; 32],
        program: &Program,
        rewritten: bool,
    ) {
        self.strict_admissions
            .push((application_id, program.canonical.canonical_ast_hash));
        self.canonical_asts
            .insert(program.canonical.canonical_cbor.clone());
        self.rewrite_collapses += u64::from(rewritten);
    }

    fn ordered_keys(values: &[Cbor]) -> Vec<Cbor> {
        let mut values = values.to_vec();
        values.sort_by_key(encode);
        values
    }

    fn root_keys(values: &[Cbor]) -> [u8; 32] {
        rfc6962(
            &Self::ordered_keys(values)
                .iter()
                .map(encode)
                .collect::<Vec<_>>(),
        )
    }

    fn strict_root(&self) -> [u8; 32] {
        rfc6962(&self.ordered_strict_rows().iter().map(encode).collect::<Vec<_>>())
    }

    fn ordered_strict_rows(&self) -> Vec<Cbor> {
        let mut rows = self
            .strict_admissions
            .iter()
            .map(|(application, ast_hash)| array([bytes(application), bytes(ast_hash)]))
            .collect::<Vec<_>>();
        rows.sort_by_key(encode);
        rows
    }

    fn sidecar_row(&self, record: &ArchiveRecord) -> Cbor {
        array([
            record.object.clone(),
            array(Self::ordered_keys(&self.eligible_keys)),
            array(Self::ordered_keys(&self.processed_keys)),
            array(self.ordered_strict_rows()),
        ])
    }

    fn evidence(&self, depth: u8, code: u16) -> CoverageEvidence {
        CoverageEvidence {
            construction_depth: depth,
            coverage_code: code,
            eligible_count: self.eligible_keys.len() as u64,
            eligible_root: hex_encode(&Self::root_keys(&self.eligible_keys)),
            processed_count: self.processed_keys.len() as u64,
            processed_root: hex_encode(&Self::root_keys(&self.processed_keys)),
            strict_admitted_count: self.strict_admissions.len() as u64,
            strict_admission_root: hex_encode(&self.strict_root()),
            unique_canonical_ast_count: self.canonical_asts.len() as u64,
            rewrite_collapse_count: self.rewrite_collapses,
        }
    }
}

fn empty_coverage() -> BTreeMap<CoverageKey, CoverageAccumulator> {
    let mut coverage = BTreeMap::new();
    for code in 0_u16..FROZEN_LEAF_COUNT as u16 {
        coverage.insert(
            CoverageKey { depth: 0, code },
            CoverageAccumulator::default(),
        );
    }
    for depth in 1_u8..=3 {
        for operator in OPERATORS {
            coverage.insert(
                CoverageKey {
                    depth,
                    code: operator.coverage_code(),
                },
                CoverageAccumulator::default(),
            );
        }
    }
    debug_assert_eq!(coverage.len(), 846);
    coverage
}

fn application_object(
    input_signature_id: u64,
    universe_root: [u8; 32],
    depth: u8,
    code: u16,
    parameters: Cbor,
    child_program_ids: impl IntoIterator<Item = [u8; 32]>,
) -> Cbor {
    array([
        uint(1),
        bytes(APPLICATION_SCHEMA),
        uint(input_signature_id),
        bytes(universe_root),
        uint(u64::from(depth)),
        uint(u64::from(code)),
        parameters,
        array(child_program_ids.into_iter().map(bytes)),
    ])
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct CoverageEvidence {
    pub construction_depth: u8,
    pub coverage_code: u16,
    pub eligible_count: u64,
    pub eligible_root: String,
    pub processed_count: u64,
    pub processed_root: String,
    pub strict_admitted_count: u64,
    pub strict_admission_root: String,
    pub unique_canonical_ast_count: u64,
    pub rewrite_collapse_count: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct DepthBarrierEvidence {
    pub depth: u8,
    pub eligible_raw_application_count: u64,
    pub strict_admitted_application_count: u64,
    pub rewrite_collapse_count: u64,
    pub behavior_class_count_after_barrier: u64,
    pub signature_cohort_count_after_barrier: u64,
    pub continuation_bank_point_count_after_barrier: u64,
    pub visible_frontier_point_count_after_barrier: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct PartitionSemanticResult {
    pub input_signature_id: u64,
    pub universe_root: String,
    pub universe_row_count: u64,
    pub raw_application_count: u64,
    pub strict_admitted_application_count: u64,
    pub rewrite_collapse_count: u64,
    pub behavior_class_count: u64,
    pub signature_cohort_count: u64,
    pub continuation_bank_point_count: u64,
    pub visible_frontier_point_count: u64,
    pub maximum_bank_points_per_class: u64,
    pub maximum_frontier_points_per_class: u64,
    pub coverage_record_count: u64,
    pub work_queue_empty: bool,
    pub zero_delta_full_boundary: bool,
    pub all_eligible_covered: bool,
    pub depth_barriers: Vec<DepthBarrierEvidence>,
    pub coverage: Vec<CoverageEvidence>,
    pub archives: ArchiveSummary,
}

fn run_partition(
    input_signature_id: u64,
    universe_root: [u8; 32],
    observations: Vec<Observation>,
) -> Result<PartitionSemanticResult, OracleError> {
    let mut state = QuotientState::default();
    let mut coverage = empty_coverage();
    let mut raw_count = 0_u64;
    let mut strict_count = 0_u64;
    let mut rewrite_count = 0_u64;
    let mut depth_barriers = Vec::new();

    for leaf in frozen_leaf_manifest()? {
        let key = CoverageKey {
            depth: 0,
            code: leaf.coverage_code,
        };
        let application = application_object(
            input_signature_id,
            universe_root,
            0,
            leaf.coverage_code,
            array([]),
            [],
        );
        let application_id = content_hash(APPLICATION_ID_DOMAIN, &application);
        coverage
            .get_mut(&key)
            .expect("leaf coverage exists")
            .add_eligible(&application);
        raw_count += 1;
        let source = leaf.canonical.canonical_node.clone();
        let program = admit(source.clone(), input_signature_id, universe_root, &observations)?
            .ok_or_else(|| {
                OracleError::new(
                    "FAIL_SEMANTICS_MISMATCH",
                    format!("frozen leaf {} was not admitted", leaf.coverage_code),
                )
            })?;
        strict_count += 1;
        coverage
            .get_mut(&key)
            .expect("leaf coverage exists")
            .add_admitted(application_id, &program, program.canonical.canonical_node != source);
        state.insert(program)?;
    }
    depth_barriers.push(DepthBarrierEvidence {
        depth: 0,
        eligible_raw_application_count: FROZEN_LEAF_COUNT as u64,
        strict_admitted_application_count: FROZEN_LEAF_COUNT as u64,
        rewrite_collapse_count: 0,
        behavior_class_count_after_barrier: state.class_count() as u64,
        signature_cohort_count_after_barrier: state.cohort_count() as u64,
        continuation_bank_point_count_after_barrier: state.bank_count() as u64,
        visible_frontier_point_count_after_barrier: state.frontier_count() as u64,
    });

    for depth in 1_u8..=3 {
        let snapshot = state.continuation_programs();
        let applications = eligible_applications(&snapshot, u32::from(depth));
        let barrier_raw = applications.len() as u64;
        let mut barrier_strict = 0_u64;
        let mut barrier_rewrite = 0_u64;
        for application in applications {
            let code = application.operator.coverage_code();
            let key = CoverageKey { depth, code };
            let application_object = application_object(
                input_signature_id,
                universe_root,
                depth,
                code,
                application.operator.parameters(),
                application.children.iter().map(|child| child.program_id),
            );
            let application_id = content_hash(APPLICATION_ID_DOMAIN, &application_object);
            coverage
                .get_mut(&key)
                .expect("operator coverage exists")
                .add_eligible(&application_object);
            raw_count += 1;
            let source = application.source_node();
            let Some(program) = admit(
                source.clone(),
                input_signature_id,
                universe_root,
                &observations,
            )? else {
                continue;
            };
            let rewritten = program.canonical.canonical_node != source;
            barrier_strict += 1;
            strict_count += 1;
            barrier_rewrite += u64::from(rewritten);
            rewrite_count += u64::from(rewritten);
            coverage
                .get_mut(&key)
                .expect("operator coverage exists")
                .add_admitted(application_id, &program, rewritten);
            state.insert(program)?;
        }
        depth_barriers.push(DepthBarrierEvidence {
            depth,
            eligible_raw_application_count: barrier_raw,
            strict_admitted_application_count: barrier_strict,
            rewrite_collapse_count: barrier_rewrite,
            behavior_class_count_after_barrier: state.class_count() as u64,
            signature_cohort_count_after_barrier: state.cohort_count() as u64,
            continuation_bank_point_count_after_barrier: state.bank_count() as u64,
            visible_frontier_point_count_after_barrier: state.frontier_count() as u64,
        });
    }

    // The separately recorded structural boundary is empty because a node-three
    // candidate cannot add a fourth node.  This is diagnostic fixed-point
    // evidence, not Q1 completion authority.
    let boundary_empty = eligible_applications(&state.continuation_programs(), 4).is_empty();
    let archives = build_archives(input_signature_id, universe_root, &state, &coverage)?;
    let evidence = coverage
        .iter()
        .map(|(key, value)| value.evidence(key.depth, key.code))
        .collect::<Vec<_>>();
    if evidence.len() != 846
        || evidence
            .iter()
            .any(|row| row.eligible_count != row.processed_count || row.eligible_root != row.processed_root)
    {
        return Err(OracleError::new(
            "FAIL_SEMANTICS_MISMATCH",
            "coverage registry is incomplete",
        ));
    }
    Ok(PartitionSemanticResult {
        input_signature_id,
        universe_root: hex_encode(&universe_root),
        universe_row_count: observations.len() as u64,
        raw_application_count: raw_count,
        strict_admitted_application_count: strict_count,
        rewrite_collapse_count: rewrite_count,
        behavior_class_count: state.class_count() as u64,
        signature_cohort_count: state.cohort_count() as u64,
        continuation_bank_point_count: state.bank_count() as u64,
        visible_frontier_point_count: state.frontier_count() as u64,
        maximum_bank_points_per_class: state.maximum_bank_per_class() as u64,
        maximum_frontier_points_per_class: state.maximum_frontier_per_class() as u64,
        coverage_record_count: evidence.len() as u64,
        work_queue_empty: boundary_empty,
        zero_delta_full_boundary: boundary_empty,
        all_eligible_covered: boundary_empty,
        depth_barriers,
        coverage: evidence,
        archives,
    })
}

#[derive(Debug, Clone)]
struct ArchiveRecord {
    key: Vec<u8>,
    object: Cbor,
    encoded: Vec<u8>,
}

impl ArchiveRecord {
    fn new(key: Vec<u8>, object: Cbor) -> Self {
        let encoded = encode(&object);
        Self {
            key,
            object,
            encoded,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[repr(u64)]
pub enum StreamKind {
    Program = 1,
    Cohort = 2,
    Class = 3,
    Coverage = 4,
}

impl StreamKind {
    fn record_id_domain(self) -> &'static [u8] {
        match self {
            Self::Program => PROGRAM_RECORD_ID_DOMAIN,
            Self::Cohort => COHORT_RECORD_ID_DOMAIN,
            Self::Class => CLASS_RECORD_ID_DOMAIN,
            Self::Coverage => COVERAGE_RECORD_ID_DOMAIN,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Program => "PROGRAM",
            Self::Cohort => "COHORT",
            Self::Class => "CLASS",
            Self::Coverage => "COVERAGE",
        }
    }
}

fn emit_frame(
    record: &[u8],
    mut emit: impl FnMut(&[u8]),
) -> Result<usize, OracleError> {
    let length = u32::try_from(record.len()).map_err(|_| {
        OracleError::new("REJECT_Q1_FRAME", "record length exceeds u32 framing")
    })?;
    let prefix = length.to_be_bytes();
    emit(&prefix);
    emit(record);
    Ok(4 + record.len())
}

fn cbor_bytes_head(length: usize) -> Vec<u8> {
    let mut output = Vec::new();
    encode_head(2, length as u64, &mut output);
    output
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CountingPathMutation {
    None,
    FlipFirstPayloadByte,
}

fn counting_update_frame_hash(
    hasher: &mut Sha256,
    record: &[u8],
    mutate_payload: bool,
) -> Result<usize, OracleError> {
    let length = u32::try_from(record.len()).map_err(|_| {
        OracleError::new(
            "REJECT_Q1_COUNTING_DISCARD",
            "counting frame length exceeds u32",
        )
    })?;
    // Deliberately independent from the materialized `emit_frame` path.
    hasher.update(length.to_be_bytes());
    if mutate_payload && !record.is_empty() {
        hasher.update([record[0] ^ 1]);
        hasher.update(&record[1..]);
    } else {
        hasher.update(record);
    }
    Ok(4 + record.len())
}

fn counting_framed_blob_hash<'a>(
    records: impl IntoIterator<Item = &'a [u8]>,
    total_length: usize,
    mutation: CountingPathMutation,
) -> Result<[u8; 32], OracleError> {
    let mut hasher = Sha256::new();
    hasher.update(FRAMED_BLOB_DOMAIN);
    hasher.update([0]);
    hasher.update([0x81]);
    hasher.update(cbor_bytes_head(total_length));
    let mut observed_length = 0_usize;
    for (index, record) in records.into_iter().enumerate() {
        observed_length = observed_length
            .checked_add(counting_update_frame_hash(
                &mut hasher,
                record,
                index == 0 && mutation == CountingPathMutation::FlipFirstPayloadByte,
            )?)
            .ok_or_else(|| {
                OracleError::new(
                    "REJECT_Q1_COUNTING_DISCARD",
                    "counting framed length overflow",
                )
            })?;
    }
    if observed_length != total_length {
        return Err(OracleError::new(
            "FAIL_Q1_COUNTING_ENCODER",
            "counting framed length differs from the independent chunk total",
        ));
    }
    Ok(hasher.finalize().into())
}

#[derive(Debug, Clone)]
struct ChunkEncoding {
    manifest: ArchiveRecord,
    framed_blob: Vec<u8>,
    framed_length: usize,
    materialized_sha256: [u8; 32],
    counting_sha256: [u8; 32],
}

fn encode_chunks_with_counting_mutation(
    input_signature_id: u64,
    universe_root: [u8; 32],
    stream_kind: StreamKind,
    records: &[ArchiveRecord],
    mutation: CountingPathMutation,
) -> Result<Vec<ChunkEncoding>, OracleError> {
    let mut ranges = Vec::new();
    let mut first = 0_usize;
    while first < records.len() {
        let mut end = first;
        let mut framed_bytes = 0_usize;
        while end < records.len() {
            let next = 4 + records[end].encoded.len();
            if next > MAX_CHUNK_FRAMED_BYTES {
                return Err(OracleError::new(
                    "INCONCLUSIVE_Q1_RECORD_TOO_LARGE",
                    "one record exceeds the frozen chunk limit",
                ));
            }
            if end > first
                && (end - first >= MAX_RECORDS_PER_CHUNK
                    || framed_bytes + next > MAX_CHUNK_FRAMED_BYTES)
            {
                break;
            }
            framed_bytes += next;
            end += 1;
        }
        ranges.push((first, end, framed_bytes));
        first = end;
    }

    let mut output = Vec::with_capacity(ranges.len());
    for (chunk_index, (first, end, framed_length)) in ranges.into_iter().enumerate() {
        let mut materialized_blob = Vec::with_capacity(framed_length);
        for record in &records[first..end] {
            emit_frame(&record.encoded, |value| materialized_blob.extend_from_slice(value))?;
        }
        if materialized_blob.len() != framed_length {
            return Err(OracleError::new(
                "FAIL_Q1_COUNTING_ENCODER",
                "materialized frame length differs",
            ));
        }
        let materialized_blob_hash = content_hash(
            FRAMED_BLOB_DOMAIN,
            &array([bytes(&materialized_blob)]),
        );
        let counting_blob_hash = counting_framed_blob_hash(
            records[first..end].iter().map(|record| record.encoded.as_slice()),
            framed_length,
            if chunk_index == 0 {
                mutation
            } else {
                CountingPathMutation::None
            },
        )?;
        if materialized_blob_hash != counting_blob_hash {
            return Err(OracleError::new(
                "FAIL_Q1_COUNTING_ENCODER",
                "counting/discard blob hash differs from materialized encoder",
            ));
        }
        let record_ids = records[first..end]
            .iter()
            .map(|record| content_hash(stream_kind.record_id_domain(), &record.object))
            .collect::<Vec<_>>();
        let manifest_object = array([
            uint(1),
            uint(CHUNK_MANIFEST_TAG),
            bytes(CHUNK_MANIFEST_SCHEMA),
            uint(input_signature_id),
            bytes(universe_root),
            uint(stream_kind as u64),
            uint(chunk_index as u64),
            uint(first as u64),
            uint((end - first) as u64),
            bytes(record_ids[0]),
            bytes(*record_ids.last().expect("nonempty chunk")),
            bytes(rfc6962(
                &records[first..end]
                    .iter()
                    .map(|record| record.encoded.clone())
                    .collect::<Vec<_>>(),
            )),
            bytes(materialized_blob_hash),
            uint(framed_length as u64),
        ]);
        let manifest = ArchiveRecord::new(
            [
                (stream_kind as u16).to_be_bytes().as_slice(),
                (chunk_index as u64).to_be_bytes().as_slice(),
            ]
            .concat(),
            manifest_object,
        );
        let materialized_sha256 = sha256(&[&materialized_blob]);
        output.push(ChunkEncoding {
            manifest,
            framed_blob: materialized_blob,
            framed_length,
            materialized_sha256,
            counting_sha256: {
                let mut hasher = Sha256::new();
                for record in &records[first..end] {
                    counting_update_frame_hash(
                        &mut hasher,
                        &record.encoded,
                        false,
                    )?;
                }
                hasher.finalize().into()
            },
        });
    }
    Ok(output)
}

fn encode_chunks(
    input_signature_id: u64,
    universe_root: [u8; 32],
    stream_kind: StreamKind,
    records: &[ArchiveRecord],
) -> Result<Vec<ChunkEncoding>, OracleError> {
    encode_chunks_with_counting_mutation(
        input_signature_id,
        universe_root,
        stream_kind,
        records,
        CountingPathMutation::None,
    )
}

fn external_row(key: &[u8], record: &[u8]) -> Result<Vec<u8>, OracleError> {
    if key.is_empty() || record.is_empty() {
        return Err(OracleError::new(
            "REJECT_Q1_SORT_ROW",
            "external-sort key and record must be nonempty",
        ));
    }
    let key_length = u32::try_from(key.len())
        .map_err(|_| OracleError::new("REJECT_Q1_SORT_ROW", "key exceeds u32"))?;
    let record_length = u32::try_from(record.len())
        .map_err(|_| OracleError::new("REJECT_Q1_SORT_ROW", "record exceeds u32"))?;
    let mut output = Vec::with_capacity(8 + key.len() + record.len());
    output.extend_from_slice(&key_length.to_be_bytes());
    output.extend_from_slice(key);
    output.extend_from_slice(&record_length.to_be_bytes());
    output.extend_from_slice(record);
    Ok(output)
}

#[derive(Debug, Clone)]
struct SortRun {
    level: u16,
    index: u32,
    rows: Vec<(Vec<u8>, Vec<u8>)>,
}

impl SortRun {
    fn payload(&self) -> Result<Vec<u8>, OracleError> {
        let mut output = Vec::new();
        for (key, record) in &self.rows {
            output.extend_from_slice(&external_row(key, record)?);
        }
        Ok(output)
    }

    fn file_id(&self) -> String {
        format!("level-{:04}-run-{:08}", self.level, self.index)
    }

    fn manifest(
        &self,
        input_signature_id: u64,
        stream_kind: StreamKind,
    ) -> Result<Cbor, OracleError> {
        let payload = self.payload()?;
        Ok(array([
            uint(1),
            bytes(EXTERNAL_SORT_RUN_SCHEMA),
            uint(input_signature_id),
            uint(stream_kind as u64),
            uint(u64::from(self.level)),
            uint(u64::from(self.index)),
            uint(self.rows.len() as u64),
            uint(payload.len() as u64),
            bytes(sha256(&[&payload])),
        ]))
    }

    fn file_size(&self) -> Result<usize, OracleError> {
        Ok(EXTERNAL_SORT_HEADER_BYTES + self.payload()?.len())
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ScratchEventEvidence {
    pub sequence: u64,
    pub action_id: u8,
    pub file_id: String,
    pub prior_size: u64,
    pub new_size: u64,
    pub live_logical_bytes_after: u64,
    pub live_charged_bytes_after: u64,
}

impl ScratchEventEvidence {
    fn object(&self) -> Cbor {
        array([
            uint(1),
            bytes(SCRATCH_EVENT_SCHEMA),
            uint(self.sequence),
            uint(u64::from(self.action_id)),
            bytes(self.file_id.as_bytes()),
            uint(self.prior_size),
            uint(self.new_size),
            uint(self.live_logical_bytes_after),
            uint(self.live_charged_bytes_after),
        ])
    }
}

#[derive(Debug, Default)]
struct ScratchLedger {
    live: BTreeMap<String, usize>,
    events: Vec<ScratchEventEvidence>,
    logical_high_water: usize,
    charged_high_water: usize,
}

fn charged_file_bytes(size: usize) -> usize {
    size.div_ceil(4096) * 4096 + 4096
}

impl ScratchLedger {
    fn append(&mut self, action: u8, file_id: String, prior: usize, new: usize) {
        let logical = self.live.values().sum::<usize>();
        let charged = self
            .live
            .values()
            .map(|size| charged_file_bytes(*size))
            .sum::<usize>();
        self.logical_high_water = self.logical_high_water.max(logical);
        self.charged_high_water = self.charged_high_water.max(charged);
        self.events.push(ScratchEventEvidence {
            sequence: self.events.len() as u64,
            action_id: action,
            file_id,
            prior_size: prior as u64,
            new_size: new as u64,
            live_logical_bytes_after: logical as u64,
            live_charged_bytes_after: charged as u64,
        });
    }

    fn allocate(&mut self, file_id: String) -> Result<(), OracleError> {
        if self.live.insert(file_id.clone(), EXTERNAL_SORT_HEADER_BYTES).is_some() {
            return Err(OracleError::new(
                "FAIL_Q1_SCRATCH_LEDGER",
                "file allocated twice",
            ));
        }
        self.append(1, file_id, 0, EXTERNAL_SORT_HEADER_BYTES);
        Ok(())
    }

    fn grow(&mut self, file_id: &str, final_size: usize) -> Result<(), OracleError> {
        let prior = *self.live.get(file_id).ok_or_else(|| {
            OracleError::new("FAIL_Q1_SCRATCH_LEDGER", "growth references absent file")
        })?;
        if final_size < prior {
            return Err(OracleError::new(
                "FAIL_Q1_SCRATCH_LEDGER",
                "file growth decreases size",
            ));
        }
        self.live.insert(file_id.to_owned(), final_size);
        self.append(2, file_id.to_owned(), prior, final_size);
        Ok(())
    }

    fn seal(&mut self, file_id: &str) -> Result<(), OracleError> {
        let size = *self.live.get(file_id).ok_or_else(|| {
            OracleError::new("FAIL_Q1_SCRATCH_LEDGER", "seal references absent file")
        })?;
        self.append(3, file_id.to_owned(), size, size);
        Ok(())
    }

    fn free(&mut self, file_id: &str) -> Result<(), OracleError> {
        let size = self.live.remove(file_id).ok_or_else(|| {
            OracleError::new("FAIL_Q1_SCRATCH_LEDGER", "free references absent file")
        })?;
        self.append(4, file_id.to_owned(), size, 0);
        Ok(())
    }
}

pub fn external_sort_merge_shape(initial: usize) -> Vec<usize> {
    if initial == 0 {
        return vec![0];
    }
    let mut shape = vec![initial];
    while *shape.last().expect("shape nonempty") > 1 {
        shape.push(shape.last().expect("shape nonempty").div_ceil(EXTERNAL_SORT_MERGE_FAN_IN));
    }
    shape
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ExternalSortSummary {
    pub record_count: u64,
    pub input_payload_bytes: u64,
    pub initial_run_count: u64,
    pub merge_level_count: u64,
    pub final_run_bytes: u64,
    pub logical_scratch_high_water_bytes: u64,
    pub charged_scratch_high_water_bytes: u64,
    pub sorted_stream_root: String,
    pub run_manifest_archive_root: String,
    pub scratch_event_ledger_root: String,
    pub scratch_event_count: u64,
    pub diagnostic_root: String,
    pub run_manifests_cbor_hex: Vec<String>,
    pub scratch_events: Vec<ScratchEventEvidence>,
    #[serde(skip)]
    projection_object: Cbor,
    #[serde(skip)]
    trace_object: Cbor,
}

fn external_sort_projection_with_limit(
    input_signature_id: u64,
    stream_kind: StreamKind,
    records: &[ArchiveRecord],
    payload_limit: usize,
) -> Result<ExternalSortSummary, OracleError> {
    let mut rows = records
        .iter()
        .rev()
        .map(|record| (record.key.clone(), record.encoded.clone()))
        .collect::<Vec<_>>();
    rows.sort();
    if rows.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Err(OracleError::new(
            "REJECT_Q1_SORT_INPUT",
            "duplicate external-sort key",
        ));
    }
    let input_payload_bytes = rows
        .iter()
        .map(|(key, record)| external_row(key, record).map(|value| value.len()))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .sum::<usize>();
    let mut initial = Vec::new();
    let mut pending = Vec::new();
    let mut pending_bytes = 0_usize;
    for row in &rows {
        let encoded_length = external_row(&row.0, &row.1)?.len();
        if encoded_length > payload_limit {
            return Err(OracleError::new(
                "INCONCLUSIVE_Q1_SORT_ROW_TOO_LARGE",
                "one external-sort row exceeds run limit",
            ));
        }
        if !pending.is_empty() && pending_bytes + encoded_length > payload_limit {
            initial.push(SortRun {
                level: 0,
                index: initial.len() as u32,
                rows: std::mem::take(&mut pending),
            });
            pending_bytes = 0;
        }
        pending.push(row.clone());
        pending_bytes += encoded_length;
    }
    if !pending.is_empty() {
        initial.push(SortRun {
            level: 0,
            index: initial.len() as u32,
            rows: pending,
        });
    }

    let mut ledger = ScratchLedger::default();
    let mut manifests = Vec::new();
    let append_run = |run: &SortRun,
                      ledger: &mut ScratchLedger,
                      manifests: &mut Vec<Cbor>|
     -> Result<(), OracleError> {
        let file_id = run.file_id();
        ledger.allocate(file_id.clone())?;
        ledger.grow(&file_id, run.file_size()?)?;
        ledger.seal(&file_id)?;
        manifests.push(run.manifest(input_signature_id, stream_kind)?);
        Ok(())
    };
    for run in &initial {
        append_run(run, &mut ledger, &mut manifests)?;
    }
    let initial_run_count = initial.len();
    let mut current = initial;
    let mut level = 0_u16;
    while current.len() > 1 {
        level += 1;
        let mut output = Vec::new();
        for group in current.chunks(EXTERNAL_SORT_MERGE_FAN_IN) {
            let mut merged_rows = group
                .iter()
                .flat_map(|run| run.rows.iter().cloned())
                .collect::<Vec<_>>();
            merged_rows.sort();
            let run = SortRun {
                level,
                index: output.len() as u32,
                rows: merged_rows,
            };
            append_run(&run, &mut ledger, &mut manifests)?;
            for child in group {
                ledger.free(&child.file_id())?;
            }
            output.push(run);
        }
        current = output;
    }
    let final_run_bytes = current
        .first()
        .map(SortRun::file_size)
        .transpose()?
        .unwrap_or(0);
    if let Some(run) = current.first() {
        ledger.free(&run.file_id())?;
    }
    if !ledger.live.is_empty() {
        return Err(OracleError::new(
            "FAIL_Q1_SCRATCH_LEDGER",
            "scratch ledger did not close",
        ));
    }
    let sorted_rows_object = array(
        rows.iter()
            .map(|(key, record)| array([bytes(key), bytes(record)])),
    );
    let sorted_stream_root = content_hash(SORTED_STREAM_DOMAIN, &sorted_rows_object);
    let run_manifest_archive_root = rfc6962(
        &manifests.iter().map(encode).collect::<Vec<_>>(),
    );
    let run_manifest_objects = array(manifests.iter().cloned());
    let scratch_objects = array(ledger.events.iter().map(ScratchEventEvidence::object));
    let scratch_event_ledger_root = content_hash(SCRATCH_LEDGER_DOMAIN, &scratch_objects);
    let projection_object = array([
        uint(1),
        bytes(EXTERNAL_SORT_PROJECTION_SCHEMA),
        uint(input_signature_id),
        uint(stream_kind as u64),
        uint(rows.len() as u64),
        uint(input_payload_bytes as u64),
        uint(initial_run_count as u64),
        uint(u64::from(level)),
        uint(final_run_bytes as u64),
        uint(ledger.logical_high_water as u64),
        uint(ledger.charged_high_water as u64),
        bytes(sorted_stream_root),
        bytes(run_manifest_archive_root),
        bytes(scratch_event_ledger_root),
        uint(ledger.events.len() as u64),
    ]);
    let trace_object = array([
        uint(1),
        bytes(EXTERNAL_SORT_TRACE_SCHEMA),
        projection_object.clone(),
        sorted_rows_object,
        run_manifest_objects,
        scratch_objects,
    ]);
    Ok(ExternalSortSummary {
        record_count: rows.len() as u64,
        input_payload_bytes: input_payload_bytes as u64,
        initial_run_count: initial_run_count as u64,
        merge_level_count: u64::from(level),
        final_run_bytes: final_run_bytes as u64,
        logical_scratch_high_water_bytes: ledger.logical_high_water as u64,
        charged_scratch_high_water_bytes: ledger.charged_high_water as u64,
        sorted_stream_root: hex_encode(&sorted_stream_root),
        run_manifest_archive_root: hex_encode(&run_manifest_archive_root),
        scratch_event_ledger_root: hex_encode(&scratch_event_ledger_root),
        scratch_event_count: ledger.events.len() as u64,
        diagnostic_root: hex_encode(&content_hash(
            EXTERNAL_SORT_PROJECTION_DOMAIN,
            &projection_object,
        )),
        run_manifests_cbor_hex: manifests.iter().map(|value| hex_encode(&encode(value))).collect(),
        scratch_events: ledger.events,
        projection_object,
        trace_object,
    })
}

fn external_sort_projection(
    input_signature_id: u64,
    stream_kind: StreamKind,
    records: &[ArchiveRecord],
) -> Result<ExternalSortSummary, OracleError> {
    external_sort_projection_with_limit(
        input_signature_id,
        stream_kind,
        records,
        EXTERNAL_SORT_RUN_PAYLOAD_LIMIT,
    )
}

fn validate_external_sort_projection(
    input_signature_id: u64,
    stream_kind: StreamKind,
    records: &[ArchiveRecord],
    supplied: &ExternalSortSummary,
) -> Result<(), OracleError> {
    let expected = external_sort_projection(input_signature_id, stream_kind, records)?;
    if &expected != supplied {
        return Err(OracleError::new(
            "REJECT_Q1_SORT_TRACE",
            "external-sort or scratch preimage differs from exact replay",
        ));
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct CountingDiscardStream {
    canonical_record_payload_bytes: usize,
    framed_stream_bytes: usize,
    descriptor_object: Cbor,
    chunk_objects: Cbor,
    external_sort: ExternalSortSummary,
    diagnostic_commitment: [u8; 32],
}

fn counting_discard_stream(
    input_signature_id: u64,
    universe_root: [u8; 32],
    stream_kind: StreamKind,
    records: &[ArchiveRecord],
) -> Result<CountingDiscardStream, OracleError> {
    if records.is_empty() {
        return Err(OracleError::new(
            "REJECT_Q1_COUNTING_DISCARD",
            "counting/discard stream must be nonempty",
        ));
    }
    let encoded = records.iter().map(|record| encode(&record.object)).collect::<Vec<_>>();
    if encoded
        .iter()
        .zip(records)
        .any(|(replayed, record)| replayed != &record.encoded)
    {
        return Err(OracleError::new(
            "FAIL_Q1_COUNTING_ENCODER",
            "counting/discard canonical record encoder differs",
        ));
    }
    let frame_lengths = encoded
        .iter()
        .map(|record| 4_usize.checked_add(record.len()))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| OracleError::new("REJECT_Q1_COUNTING_DISCARD", "frame length overflow"))?;
    if frame_lengths.iter().any(|length| *length > MAX_CHUNK_FRAMED_BYTES) {
        return Err(OracleError::new(
            "INCONCLUSIVE_Q1_RECORD_TOO_LARGE",
            "one counting/discard record exceeds the frozen chunk limit",
        ));
    }
    let mut ranges = Vec::new();
    let mut start = 0_usize;
    while start < records.len() {
        let mut end = start;
        let mut framed_bytes = 0_usize;
        while end < records.len() {
            let next = frame_lengths[end];
            if end > start
                && (end - start >= MAX_RECORDS_PER_CHUNK
                    || framed_bytes + next > MAX_CHUNK_FRAMED_BYTES)
            {
                break;
            }
            framed_bytes += next;
            end += 1;
        }
        ranges.push((start, end, framed_bytes));
        start = end;
    }
    let mut manifests = Vec::with_capacity(ranges.len());
    for (chunk_index, (start, end, framed_length)) in ranges.iter().copied().enumerate() {
        let blob_hash = counting_framed_blob_hash(
            encoded[start..end].iter().map(Vec::as_slice),
            framed_length,
            CountingPathMutation::None,
        )?;
        let record_ids = records[start..end]
            .iter()
            .map(|record| content_hash(stream_kind.record_id_domain(), &record.object))
            .collect::<Vec<_>>();
        manifests.push(array([
            uint(1),
            uint(CHUNK_MANIFEST_TAG),
            bytes(CHUNK_MANIFEST_SCHEMA),
            uint(input_signature_id),
            bytes(universe_root),
            uint(stream_kind as u64),
            uint(chunk_index as u64),
            uint(start as u64),
            uint((end - start) as u64),
            bytes(record_ids[0]),
            bytes(*record_ids.last().expect("nonempty counting chunk")),
            bytes(rfc6962(&encoded[start..end])),
            bytes(blob_hash),
            uint(framed_length as u64),
        ]));
    }
    let archive_root = rfc6962(&encoded);
    let chunk_root = rfc6962(&manifests.iter().map(encode).collect::<Vec<_>>());
    let framed_stream_bytes = ranges.iter().map(|row| row.2).sum::<usize>();
    let descriptor_object = array([
        uint(1),
        bytes(STREAM_DESCRIPTOR_SCHEMA),
        uint(stream_kind as u64),
        uint(records.len() as u64),
        bytes(archive_root),
        uint(framed_stream_bytes as u64),
        uint(manifests.len() as u64),
        bytes(chunk_root),
    ]);
    let chunk_objects = array(manifests);
    let external_sort = external_sort_projection(input_signature_id, stream_kind, records)?;
    let projected_preimage = array([
        uint(1),
        bytes(PROJECTED_STREAM_SCHEMA),
        uint(input_signature_id),
        bytes(universe_root),
        uint(stream_kind as u64),
        descriptor_object.clone(),
        chunk_objects.clone(),
        external_sort.projection_object.clone(),
    ]);
    Ok(CountingDiscardStream {
        canonical_record_payload_bytes: encoded.iter().map(Vec::len).sum(),
        framed_stream_bytes,
        descriptor_object,
        chunk_objects,
        external_sort,
        diagnostic_commitment: content_hash(PROJECTED_STREAM_DOMAIN, &projected_preimage),
    })
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct StreamEncodingSummary {
    pub stream_kind_id: u64,
    pub stream_kind_name: String,
    pub record_count: u64,
    pub archive_root: String,
    pub framed_stream_bytes: u64,
    pub chunk_count: u64,
    pub chunk_manifest_archive_root: String,
    pub projected_stream_commitment: String,
    pub materialized_stream_sha256: String,
    pub counting_discard_stream_sha256: String,
    pub materialized_counting_equal: bool,
    pub external_sort: ExternalSortSummary,
    #[serde(skip)]
    projected_object: Cbor,
    #[serde(skip)]
    counting_discard_object: Cbor,
    #[serde(skip)]
    framed_blobs: Vec<Vec<u8>>,
}

fn encode_stream(
    input_signature_id: u64,
    universe_root: [u8; 32],
    stream_kind: StreamKind,
    records: &[ArchiveRecord],
) -> Result<StreamEncodingSummary, OracleError> {
    if records.is_empty() {
        return Err(OracleError::new(
            "REJECT_Q1_STREAM",
            "node3 stream must be nonempty",
        ));
    }
    let chunks = encode_chunks(input_signature_id, universe_root, stream_kind, records)?;
    let framed_stream_bytes = chunks.iter().map(|chunk| chunk.framed_length).sum::<usize>();
    let (materialized_stream, materialized_stream_sha) = {
        let mut materialized = Vec::with_capacity(framed_stream_bytes);
        for record in records {
            emit_frame(&record.encoded, |value| materialized.extend_from_slice(value))?;
        }
        let digest = sha256(&[&materialized]);
        (materialized, digest)
    };
    let counting_stream_sha = {
        let mut hasher = Sha256::new();
        for record in records {
            let replayed = encode(&record.object);
            counting_update_frame_hash(&mut hasher, &replayed, false)?;
        }
        <[u8; 32]>::from(hasher.finalize())
    };
    if materialized_stream.len() != framed_stream_bytes
        || materialized_stream_sha != counting_stream_sha
        || chunks
            .iter()
            .any(|chunk| chunk.materialized_sha256 != chunk.counting_sha256)
    {
        return Err(OracleError::new(
            "FAIL_Q1_COUNTING_ENCODER",
            "materialized and counting streams differ",
        ));
    }
    let external_sort = external_sort_projection(input_signature_id, stream_kind, records)?;
    validate_external_sort_projection(
        input_signature_id,
        stream_kind,
        records,
        &external_sort,
    )?;
    let archive_root = archive_root(records);
    let chunk_manifest_archive_root = rfc6962(
        &chunks
            .iter()
            .map(|chunk| chunk.manifest.encoded.clone())
            .collect::<Vec<_>>(),
    );
    let descriptor_object = array([
        uint(1),
        bytes(STREAM_DESCRIPTOR_SCHEMA),
        uint(stream_kind as u64),
        uint(records.len() as u64),
        bytes(archive_root),
        uint(framed_stream_bytes as u64),
        uint(chunks.len() as u64),
        bytes(chunk_manifest_archive_root),
    ]);
    let chunk_objects = array(chunks.iter().map(|chunk| chunk.manifest.object.clone()));
    let projected_preimage = array([
        uint(1),
        bytes(PROJECTED_STREAM_SCHEMA),
        uint(input_signature_id),
        bytes(universe_root),
        uint(stream_kind as u64),
        descriptor_object.clone(),
        chunk_objects.clone(),
        external_sort.projection_object.clone(),
    ]);
    let projected_stream_commitment = content_hash(PROJECTED_STREAM_DOMAIN, &projected_preimage);
    let projected_object = match projected_preimage {
        Cbor::Array(mut fields) => {
            fields.push(bytes(projected_stream_commitment));
            Cbor::Array(fields)
        }
        _ => unreachable!(),
    };
    let counting_discard =
        counting_discard_stream(input_signature_id, universe_root, stream_kind, records)?;
    if counting_discard.framed_stream_bytes != framed_stream_bytes
        || counting_discard.descriptor_object != descriptor_object
        || counting_discard.chunk_objects != chunk_objects
        || counting_discard.external_sort != external_sort
        || counting_discard.diagnostic_commitment != projected_stream_commitment
    {
        return Err(OracleError::new(
            "REJECT_Q1_DUAL_ENCODER",
            "independent counting/discard projection differs from materialized projection",
        ));
    }
    let counting_discard_object = array([
        uint(1),
        bytes(COUNTING_DISCARD_STREAM_SCHEMA),
        uint(input_signature_id),
        bytes(universe_root),
        uint(stream_kind as u64),
        uint(records.len() as u64),
        uint(counting_discard.canonical_record_payload_bytes as u64),
        uint(counting_discard.framed_stream_bytes as u64),
        uint(chunks.len() as u64),
        counting_discard.descriptor_object,
        counting_discard.chunk_objects,
        counting_discard.external_sort.projection_object,
        bytes(counting_discard.diagnostic_commitment),
        uint(0),
        uint(0),
    ]);
    Ok(StreamEncodingSummary {
        stream_kind_id: stream_kind as u64,
        stream_kind_name: stream_kind.name().to_owned(),
        record_count: records.len() as u64,
        archive_root: hex_encode(&archive_root),
        framed_stream_bytes: framed_stream_bytes as u64,
        chunk_count: chunks.len() as u64,
        chunk_manifest_archive_root: hex_encode(&chunk_manifest_archive_root),
        projected_stream_commitment: hex_encode(&projected_stream_commitment),
        materialized_stream_sha256: hex_encode(&materialized_stream_sha),
        counting_discard_stream_sha256: hex_encode(&counting_stream_sha),
        materialized_counting_equal: true,
        external_sort,
        projected_object,
        counting_discard_object,
        framed_blobs: chunks.into_iter().map(|chunk| chunk.framed_blob).collect(),
    })
}

#[derive(Debug, Clone)]
struct ArchiveMaterial {
    programs: Vec<ArchiveRecord>,
    cohorts: Vec<ArchiveRecord>,
    classes: Vec<ArchiveRecord>,
    coverage: Vec<ArchiveRecord>,
}

fn program_sort_key(program: &Program) -> Vec<u8> {
    let mut key = Vec::with_capacity(6 + program.canonical.canonical_cbor.len());
    key.push(program.canonical.depth as u8);
    key.extend_from_slice(&(program.canonical.node_count as u16).to_be_bytes());
    key.push(program.signature.output_sort as u8);
    key.extend_from_slice(&(program.canonical.root_operator_id as u16).to_be_bytes());
    key.extend_from_slice(&program.canonical.canonical_cbor);
    key
}

fn program_record_object(
    input_signature_id: u64,
    universe_root: [u8; 32],
    program_index: usize,
    program: &Program,
) -> Result<Cbor, OracleError> {
    Ok(array([
        uint(1),
        uint(PROGRAM_RECORD_TAG),
        bytes(PROGRAM_RECORD_SCHEMA),
        uint(input_signature_id),
        bytes(universe_root),
        uint(program_index as u64),
        bytes(program.program_id),
        bytes(program.behavior.id()?),
        bytes(&program.canonical.canonical_cbor),
        bytes(program.canonical.canonical_ast_hash),
        program.signature.object(),
        bytes(program.signature.id()),
    ]))
}

fn cohort_id(
    input_signature_id: u64,
    universe_root: [u8; 32],
    class_id: [u8; 32],
    signature_id: [u8; 32],
) -> [u8; 32] {
    content_hash(
        COHORT_ID_DOMAIN,
        &array([
            uint(input_signature_id),
            bytes(universe_root),
            bytes(class_id),
            bytes(signature_id),
        ]),
    )
}

fn cohort_record_object(
    input_signature_id: u64,
    universe_root: [u8; 32],
    cohort_index: usize,
    class_id: [u8; 32],
    cohort: &[Program],
    visible: bool,
) -> Cbor {
    let signature = &cohort[0].signature;
    let signature_id = signature.id();
    array([
        uint(1),
        uint(COHORT_RECORD_TAG),
        bytes(COHORT_RECORD_SCHEMA),
        uint(input_signature_id),
        bytes(universe_root),
        uint(cohort_index as u64),
        bytes(cohort_id(
            input_signature_id,
            universe_root,
            class_id,
            signature_id,
        )),
        bytes(class_id),
        signature.object(),
        bytes(signature_id),
        uint(capacity(signature.output_sort) as u64),
        uint(cohort.len() as u64),
        array(cohort.iter().enumerate().map(|(rank, program)| {
            array([
                uint(rank as u64),
                bytes(program.program_id),
                bytes(program.canonical.canonical_ast_hash),
            ])
        })),
        Cbor::Bool(visible),
    ])
}

fn class_record_object(
    input_signature_id: u64,
    universe_root: [u8; 32],
    class_index: usize,
    class_id: [u8; 32],
    class: &QuotientClass,
    first_cohort_index: usize,
    cohort_records: &[ArchiveRecord],
) -> Result<Cbor, OracleError> {
    let visible_records = cohort_records
        .iter()
        .filter(|record| match &record.object {
            Cbor::Array(fields) => matches!(fields.last(), Some(Cbor::Bool(true))),
            _ => false,
        })
        .map(|record| record.encoded.clone())
        .collect::<Vec<_>>();
    let minimum_mdl = class
        .cohorts
        .values()
        .flat_map(|cohort| cohort.iter())
        .map(|program| program.signature.mdl_q32)
        .min()
        .ok_or_else(|| OracleError::new("FAIL_SEMANTICS_MISMATCH", "empty class bank"))?;
    Ok(array([
        uint(1),
        uint(CLASS_RECORD_TAG),
        bytes(CLASS_RECORD_SCHEMA),
        uint(input_signature_id),
        bytes(universe_root),
        uint(class_index as u64),
        class.behavior.object()?,
        bytes(class_id),
        uint(first_cohort_index as u64),
        uint(cohort_records.len() as u64),
        bytes(rfc6962(
            &cohort_records
                .iter()
                .map(|record| record.encoded.clone())
                .collect::<Vec<_>>(),
        )),
        uint(class.bank_count() as u64),
        uint(visible_records.len() as u64),
        uint(class.frontier_count() as u64),
        bytes(rfc6962(&visible_records)),
        uint(minimum_mdl),
    ]))
}

fn coverage_record_object(
    input_signature_id: u64,
    universe_root: [u8; 32],
    key: CoverageKey,
    accumulator: &CoverageAccumulator,
) -> Cbor {
    array([
        uint(1),
        uint(COVERAGE_RECORD_TAG),
        bytes(COVERAGE_RECORD_SCHEMA),
        uint(input_signature_id),
        bytes(universe_root),
        uint(u64::from(key.depth)),
        uint(u64::from(key.code)),
        uint(accumulator.eligible_keys.len() as u64),
        bytes(CoverageAccumulator::root_keys(&accumulator.eligible_keys)),
        uint(accumulator.processed_keys.len() as u64),
        bytes(CoverageAccumulator::root_keys(&accumulator.processed_keys)),
        uint(accumulator.strict_admissions.len() as u64),
        bytes(accumulator.strict_root()),
        uint(accumulator.canonical_asts.len() as u64),
        uint(accumulator.rewrite_collapses),
    ])
}

fn build_archive_material(
    input_signature_id: u64,
    universe_root: [u8; 32],
    state: &QuotientState,
    coverage: &BTreeMap<CoverageKey, CoverageAccumulator>,
) -> Result<ArchiveMaterial, OracleError> {
    let mut bank_programs = state.continuation_programs();
    bank_programs.sort_by_key(program_sort_key);
    let mut program_id_preimages = BTreeMap::new();
    for program in &bank_programs {
        let preimage = program_identity_object(
            input_signature_id,
            universe_root,
            &program.canonical,
            &program.signature,
        );
        register_preimage(
            &mut program_id_preimages,
            program.program_id,
            encode(&preimage),
            "program ID",
        )?;
    }
    let programs = bank_programs
        .iter()
        .enumerate()
        .map(|(index, program)| {
            Ok(ArchiveRecord::new(
                program_sort_key(program),
                program_record_object(input_signature_id, universe_root, index, program)?,
            ))
        })
        .collect::<Result<Vec<_>, OracleError>>()?;

    let mut cohort_rows = Vec::new();
    for (class_id, class) in &state.classes {
        let visible = class
            .visible_cohorts()
            .into_iter()
            .map(|cohort| cohort[0].signature.canonical_bytes())
            .collect::<BTreeSet<_>>();
        for cohort in class.cohorts.values() {
            let signature_id = cohort[0].signature.id();
            let signature_bytes = cohort[0].signature.canonical_bytes();
            cohort_rows.push((
                *class_id,
                signature_id,
                signature_bytes,
                cohort.clone(),
                visible.contains(&cohort[0].signature.canonical_bytes()),
            ));
        }
    }
    cohort_rows.sort_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then_with(|| left.1.cmp(&right.1))
            .then_with(|| left.2.cmp(&right.2))
    });
    let cohorts = cohort_rows
        .iter()
        .enumerate()
        .map(|(index, (class_id, signature_id, signature_bytes, cohort, visible))| {
            let mut key = Vec::new();
            key.extend_from_slice(class_id);
            key.extend_from_slice(signature_id);
            key.extend_from_slice(signature_bytes);
            ArchiveRecord::new(
                key,
                cohort_record_object(
                    input_signature_id,
                    universe_root,
                    index,
                    *class_id,
                    cohort,
                    *visible,
                ),
            )
        })
        .collect::<Vec<_>>();

    let mut classes = Vec::new();
    let mut cohort_cursor = 0_usize;
    for (class_index, (class_id, class)) in state.classes.iter().enumerate() {
        let cohort_count = class.cohorts.len();
        let cohort_slice = &cohorts[cohort_cursor..cohort_cursor + cohort_count];
        let mut key = class_id.to_vec();
        key.extend_from_slice(&class.behavior_bytes);
        classes.push(ArchiveRecord::new(
            key,
            class_record_object(
                input_signature_id,
                universe_root,
                class_index,
                *class_id,
                class,
                cohort_cursor,
                cohort_slice,
            )?,
        ));
        cohort_cursor += cohort_count;
    }
    debug_assert_eq!(cohort_cursor, cohorts.len());

    let coverage = coverage
        .iter()
        .map(|(key, accumulator)| {
            let key_bytes = vec![
                key.depth,
                (key.code >> 8) as u8,
                (key.code & 0xff) as u8,
            ];
            ArchiveRecord::new(
                key_bytes,
                coverage_record_object(input_signature_id, universe_root, *key, accumulator),
            )
        })
        .collect::<Vec<_>>();
    Ok(ArchiveMaterial {
        programs,
        cohorts,
        classes,
        coverage,
    })
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ArchiveSummary {
    pub snapshot_record_set_root: String,
    pub program_record_count: u64,
    pub program_archive_root: String,
    pub cohort_record_count: u64,
    pub cohort_archive_root: String,
    pub class_record_count: u64,
    pub class_archive_root: String,
    pub coverage_record_count: u64,
    pub coverage_archive_root: String,
    pub streams: Vec<StreamEncodingSummary>,
    #[serde(skip)]
    record_set_object: Cbor,
    #[serde(skip)]
    coverage_sidecar_rows: Vec<Cbor>,
}

fn archive_root(records: &[ArchiveRecord]) -> [u8; 32] {
    rfc6962(
        &records
            .iter()
            .map(|record| record.encoded.clone())
            .collect::<Vec<_>>(),
    )
}

fn build_archives(
    input_signature_id: u64,
    universe_root: [u8; 32],
    state: &QuotientState,
    coverage: &BTreeMap<CoverageKey, CoverageAccumulator>,
) -> Result<ArchiveSummary, OracleError> {
    let material = build_archive_material(input_signature_id, universe_root, state, coverage)?;
    let snapshot = array([
        uint(1),
        bytes(SNAPSHOT_RECORD_SET_SCHEMA),
        uint(input_signature_id),
        bytes(universe_root),
        array(material.programs.iter().map(|record| record.object.clone())),
        array(material.cohorts.iter().map(|record| record.object.clone())),
        array(material.classes.iter().map(|record| record.object.clone())),
    ]);
    let streams = [
        (StreamKind::Program, material.programs.as_slice()),
        (StreamKind::Cohort, material.cohorts.as_slice()),
        (StreamKind::Class, material.classes.as_slice()),
        (StreamKind::Coverage, material.coverage.as_slice()),
    ]
    .into_iter()
    .map(|(kind, records)| encode_stream(input_signature_id, universe_root, kind, records))
    .collect::<Result<Vec<_>, _>>()?;
    let coverage_sidecar_rows = coverage
        .values()
        .zip(material.coverage.iter())
        .map(|(accumulator, record)| accumulator.sidecar_row(record))
        .collect::<Vec<_>>();
    Ok(ArchiveSummary {
        snapshot_record_set_root: hex_encode(&content_hash(SNAPSHOT_RECORD_SET_DOMAIN, &snapshot)),
        program_record_count: material.programs.len() as u64,
        program_archive_root: hex_encode(&archive_root(&material.programs)),
        cohort_record_count: material.cohorts.len() as u64,
        cohort_archive_root: hex_encode(&archive_root(&material.cohorts)),
        class_record_count: material.classes.len() as u64,
        class_archive_root: hex_encode(&archive_root(&material.classes)),
        coverage_record_count: material.coverage.len() as u64,
        coverage_archive_root: hex_encode(&archive_root(&material.coverage)),
        streams,
        record_set_object: snapshot,
        coverage_sidecar_rows,
    })
}

pub fn coverage_registry_root() -> String {
    let mut rows = (0_u16..FROZEN_LEAF_COUNT as u16)
        .map(|code| encode(&array([uint(0), uint(u64::from(code))])))
        .collect::<Vec<_>>();
    for depth in 1_u64..=3 {
        rows.extend(OPERATORS.iter().map(|operator| {
            encode(&array([
                uint(depth),
                uint(u64::from(operator.coverage_code())),
            ]))
        }));
    }
    hex_encode(&rfc6962(&rows))
}

fn decode_hex_root(value: &str) -> Result<[u8; 32], OracleError> {
    if value.len() != 64 {
        return Err(OracleError::new("REJECT_Q05B_ROOT", "root hex length differs"));
    }
    let mut output = [0_u8; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        let digit = |value: u8| match value {
            b'0'..=b'9' => Some(value - b'0'),
            b'a'..=b'f' => Some(value - b'a' + 10),
            _ => None,
        };
        output[index] = (digit(pair[0]).ok_or_else(|| {
            OracleError::new("REJECT_Q05B_ROOT", "root hex is not lowercase")
        })? << 4)
            | digit(pair[1]).ok_or_else(|| {
                OracleError::new("REJECT_Q05B_ROOT", "root hex is not lowercase")
            })?;
    }
    Ok(output)
}

fn frozen_root(value: &str) -> [u8; 32] {
    decode_hex_root(value).expect("frozen lowercase 32-byte root")
}

fn q1_tag_registry_object() -> Cbor {
    let names: [&[u8]; 13] = [
        b"Q1_SEMANTIC_BINDING_MANIFEST",
        b"Q1_BEHAVIOR_BLOB",
        b"Q1_CONSTRUCTION_SIGNATURE",
        b"Q1_REPRESENTATIVE_PROGRAM_RECORD",
        b"Q1_CONTINUATION_COHORT_RECORD",
        b"Q1_QUOTIENT_CLASS_RECORD",
        b"Q1_SEMANTIC_COVERAGE_RECORD",
        b"Q1_FIXED_POINT_RECORD",
        b"Q1_ARCHIVE_CHUNK_MANIFEST",
        b"Q1_SIGNATURE_ARCHIVE_MANIFEST",
        b"Q1_CLOSURE_BUNDLE",
        b"Q1_ARCHIVE_PROJECTION_PROFILE",
        b"Q1_ARCHIVE_PROJECTION_RESULT",
    ];
    array(names.into_iter().enumerate().map(|(index, name)| {
        array([uint(0x3700 + index as u64), bytes(name)])
    }))
}

fn coverage_registry_object() -> Cbor {
    let mut rows = (0_u64..FROZEN_LEAF_COUNT as u64)
        .map(|code| array([uint(0), uint(code)]))
        .collect::<Vec<_>>();
    for depth in 1_u64..=3 {
        rows.extend(
            OPERATORS
                .iter()
                .map(|operator| array([uint(depth), uint(u64::from(operator.coverage_code()))])),
        );
    }
    array(rows)
}

fn resource_guard_registry_object() -> Cbor {
    let names: [&[u8]; 12] = [
        b"RAW_OPERATOR_APPLICATIONS",
        b"BEHAVIOR_CLASSES",
        b"VISIBLE_FRONTIER_TOTAL",
        b"VISIBLE_FRONTIER_PER_CLASS",
        b"CONTINUATION_BANK_TOTAL",
        b"CONTINUATION_BANK_PER_CLASS",
        b"WORK_QUEUE_POINTS",
        b"SATURATION_ROUNDS",
        b"OUTPUT_BYTES",
        b"SCRATCH_BYTES",
        b"RESIDENT_MEMORY",
        b"WALL_TIME",
    ];
    array(names.into_iter().enumerate().map(|(index, name)| {
        array([uint(index as u64 + 1), bytes(name)])
    }))
}

fn q1_output_slot_names() -> [&'static [u8]; 8] {
    [
        b"odd_signature_archive_manifest_root",
        b"odd_signature_saturation_state_root",
        b"sink_signature_archive_manifest_root",
        b"sink_signature_saturation_state_root",
        b"q1_closure_bundle_root",
        b"q1_dual_replay_agreement_root",
        b"q1_target_blind_access_ledger_root",
        b"q1_completion_receipt_root",
    ]
}

fn q1_authority_object() -> Cbor {
    let slots = array(
        q1_output_slot_names()
            .into_iter()
            .enumerate()
            .map(|(index, name)| array([uint(index as u64 + 1), bytes(name), Cbor::Null])),
    );
    array([
        uint(0),
        uint(0),
        uint(0),
        uint(20),
        uint(8),
        slots,
        Cbor::Null,
        uint(0),
        Cbor::Null,
        Cbor::Bool(false),
        Cbor::Null,
        Cbor::Bool(false),
        Cbor::Bool(false),
        Cbor::Bool(false),
        Cbor::Bool(false),
        Cbor::Bool(false),
    ])
}

fn q1_semantic_binding_object(
    full_leaf_manifest_root: [u8; 32],
    odd_universe_root: [u8; 32],
    sink_universe_root: [u8; 32],
) -> Cbor {
    let child_dsl = frozen_root("da5ed2db33a88a0912d5003999f787cc26ba18564876615773a82bb742d9f8ae");
    let operator = frozen_root("922e48ada22dfa8621a4d516e07ec9aa7dc8fc10c165d1cafc963575aed5ec03");
    let identifiers = frozen_root("64c9415f7759eec140e439030c5a5374851b9024d7d4849b52b995704ba76ff1");
    let ast = frozen_root("5de72fc51e27e5501561ffda6b05522f4941d1a13c4b324f5edcc15fa584a0bd");
    let cbor = frozen_root("ef0008912962de9da322eaeea6e421e1e58d16be152f968298774af0fd3249ab");
    let q0_receipt = frozen_root("ee198614e94cf425202f9c667836fc6ad61fda02c9439a689eb90012c5798ad2");
    let preregistration = frozen_root("2fbbba865abf0589c0d48ead9a170fae0b81f1cc1d440ddbc9c5d93909615f42");
    let post_shrink6 = frozen_root("1df8d3ff3ede2cbead98e7901a3e82b91c460ad1d5eb0d1af78938e7b2d23b95");
    array([
        uint(1),
        uint(Q1_SEMANTIC_BINDING_TAG),
        bytes(Q1_SEMANTIC_BINDING_SCHEMA),
        bytes(DSL_VERSION.as_bytes()),
        bytes(DSL_FREEZE_VERSION.as_bytes()),
        bytes(CLOSURE_SEMANTICS_VERSION.as_bytes()),
        bytes(child_dsl),
        bytes(operator),
        bytes(identifiers),
        bytes(ast),
        bytes(cbor),
        bytes(Q1_MDL_PROFILE_ID),
        bytes(q0_receipt),
        bytes(full_leaf_manifest_root),
        array([
            array([uint(1), bytes(odd_universe_root), uint(480)]),
            array([uint(2), bytes(sink_universe_root), uint(85)]),
        ]),
        bytes(preregistration),
        bytes(post_shrink6),
    ])
}

fn q1_semantic_binding_root(
    full_leaf_manifest_root: [u8; 32],
    odd_universe_root: [u8; 32],
    sink_universe_root: [u8; 32],
) -> [u8; 32] {
    content_hash(
        Q1_SEMANTIC_BINDING_DOMAIN,
        &q1_semantic_binding_object(
            full_leaf_manifest_root,
            odd_universe_root,
            sink_universe_root,
        ),
    )
}

fn q1_projection_profile_object(semantic_binding_root: [u8; 32]) -> Cbor {
    let coverage = coverage_registry_object();
    let coverage_root = match &coverage {
        Cbor::Array(rows) => rfc6962(&rows.iter().map(encode).collect::<Vec<_>>()),
        _ => unreachable!(),
    };
    array([
        uint(1),
        uint(Q1_ARCHIVE_PROJECTION_PROFILE_TAG),
        bytes(Q1_ARCHIVE_PROJECTION_PROFILE_SCHEMA),
        bytes(ARCHIVE_WIRE_VERSION.as_bytes()),
        bytes(PROJECTION_FREEZE_VERSION.as_bytes()),
        bytes(Q1_PROJECTION_PROFILE_ID),
        bytes(semantic_binding_root),
        q1_tag_registry_object(),
        coverage,
        bytes(coverage_root),
        array([uint(1), uint(2), uint(3), uint(4)]),
        uint(MAX_RECORDS_PER_CHUNK as u64),
        uint(MAX_CHUNK_FRAMED_BYTES as u64),
        uint(4),
        uint(0),
        bytes(b"FRAME_U32BE_LENGTH_PLUS_CANONICAL_CBOR"),
        bytes(b"CHUNK_CLOSE_BEFORE_NEXT_RECORD_EXCEEDS_RECORD_OR_FRAMED_BYTE_LIMIT"),
        array([
            bytes(b"PROGRAM_U8_DEPTH_U16_NODES_U8_SORT_U16_ROOT_OPERATOR_AST_CBOR"),
            bytes(b"COHORT_CLASS_ID_SIGNATURE_ID_SIGNATURE_CBOR"),
            bytes(b"CLASS_ID_BEHAVIOR_CBOR"),
            bytes(b"COVERAGE_U8_DEPTH_U16_COVERAGE_CODE"),
        ]),
        uint(1_048_576),
        uint(1_048_576),
        uint(64),
        uint(16_384),
        uint(EXTERNAL_SORT_RUN_PAYLOAD_LIMIT as u64),
        uint(EXTERNAL_SORT_MERGE_FAN_IN as u64),
        bytes(b"HGQ1RUN1"),
        uint(EXTERNAL_SORT_HEADER_BYTES as u64),
        uint(4),
        uint(4096),
        uint(4096),
        Cbor::Bool(true),
        array([uint(1), uint(2), uint(3), uint(4)]),
        array([uint(1), uint(2)]),
        bytes(b"STABLE_K_WAY_MERGE_CONTIGUOUS_RUN_INDEX_GROUPS"),
        bytes(b"SEAL_HASH_REOPEN_VERIFY_THEN_FREE_INPUT_GROUP"),
        bytes(b"NO_RANDOM_OR_TIME_COMPONENT_IN_RUN_FILE_NAME"),
        bytes(b"RUN_ROW_U32BE_KEY_LENGTH_KEY_U32BE_RECORD_LENGTH_CANONICAL_RECORD"),
        bytes(b"SCRATCH_CHARGE_CEIL_FILE_SIZE_TO_4096_PLUS_4096_PER_LIVE_FILE"),
        resource_guard_registry_object(),
        array(q1_output_slot_names().into_iter().map(bytes)),
        bytes(b"DEPTH_BARRIER_DIRECT_FULL_BANK"),
        bytes(b"RAW_AND_SEMANTIC_COVERAGE_EXACT_EQUAL_WORK_QUEUE_HIGH_WATER_MAX"),
        bytes(b"COUNTING_DISCARD_USES_IDENTICAL_ENCODER_AND_FIXED_ROOT_PLACEHOLDERS"),
    ])
}

fn q1_projection_profile_root(semantic_binding_root: [u8; 32]) -> [u8; 32] {
    content_hash(
        Q1_PROJECTION_PROFILE_DOMAIN,
        &q1_projection_profile_object(semantic_binding_root),
    )
}

fn q05b_tag_registry_object() -> Cbor {
    let rows: [(u64, &[u8]); 8] = [
        (0x3a00, b"Q05B_FULL_LEAF_MANIFEST_ROW"),
        (0x3a01, b"Q05B_FULL_LEAF_MANIFEST"),
        (0x3a02, b"Q05B_NODE3_PARTITION_EVIDENCE"),
        (0x3a03, b"Q05B_SIDECAR_MANIFEST"),
        (0x3a04, b"Q05B_NODE3_GOLDEN_MANIFEST"),
        (0x3a05, b"Q05B_QUALIFICATION_CANDIDATE_RECEIPT"),
        (0x3a06, b"Q05B_QUALIFICATION_RECEIPT"),
        (0x3a07, b"Q05B_BOUNDED_NODE3_STATE"),
    ];
    array(rows.into_iter().map(|(tag, name)| array([uint(tag), bytes(name)])))
}

fn q05b_tag_registry_root() -> [u8; 32] {
    content_hash(Q05B_TAG_REGISTRY_DOMAIN, &q05b_tag_registry_object())
}

fn qualification_predicate_registry_object() -> Cbor {
    let names: [&[u8]; 20] = [
        b"QUALIFICATION_TAG_NAMESPACE_SEPARATE_FROM_FORMAL_Q1",
        b"STRICT_CANONICAL_CBOR_DUAL_REPLAY",
        b"FULL_810_LEAF_MANIFEST_EXACT_REPLAY",
        b"Q0_AND_Q1_PREREG_SEMANTIC_SOURCE_ROOTS_BOUND",
        b"BOUNDED_NODE3_SCOPE_NO_FORMAL_FIXED_POINT_ALIAS",
        b"NEUTRAL_GOLDEN_MANIFEST_PYTHON_RUST_HOST_BYTE_EQUAL",
        b"SIDECAR_MANIFEST_PYTHON_RUST_HOST_BYTE_EQUAL",
        b"SIDECAR_RAW_SHA_LENGTH_CONTENT_ROOT_REPLAY",
        b"PYTHON_ACTOR_SOURCE_RUNTIME_IDENTITY_QUALIFIED",
        b"RUST_ACTOR_SOURCE_RUNTIME_IDENTITY_QUALIFIED",
        b"TRUSTED_HOST_READ_ONLY_REPLAY_QUALIFIED",
        b"STRICT_PARTITION_MANIFEST_BUNDLE_ASSEMBLER_REPLAY",
        b"CHUNK_FRAMING_BOUNDARY_AND_TAMPER_VECTORS_PASS",
        b"COUNTING_DISCARD_AND_MATERIALIZED_ENCODER_EQUAL",
        b"EXTERNAL_SORT_RUN_AND_MERGE_REPLAY_PASS",
        b"THREE_ACTOR_SCRATCH_LEDGER_REPLAY_PASS",
        b"OUTPUT_AND_METADATA_FORMULA_REPLAY_PASS",
        b"COLLISION_DUPLICATE_AND_TAMPER_FAIL_CLOSED",
        b"OFFLINE_SOURCE_RUNTIME_AND_FILESYSTEM_ISOLATION_PASS",
        b"CANDIDATE_RECEIPT_VALIDATED_WHILE_Q1_REMAINS_NOT_RUN",
    ];
    array(names.into_iter().enumerate().map(|(index, name)| {
        array([uint(index as u64 + 1), bytes(name)])
    }))
}

fn qualification_predicate_registry_root() -> [u8; 32] {
    content_hash(
        b"HEGEL/Q05B/QUALIFICATION/PREDICATE_REGISTRY/V1",
        &qualification_predicate_registry_object(),
    )
}

fn version_binding_rows_object() -> Cbor {
    let rows: [(u64, &[u8], &[u8]); 6] = [
        (1, b"dsl_version", DSL_VERSION.as_bytes()),
        (2, b"dsl_freeze_version", DSL_FREEZE_VERSION.as_bytes()),
        (3, b"closure_semantics_version", CLOSURE_SEMANTICS_VERSION.as_bytes()),
        (4, b"archive_wire_version", ARCHIVE_WIRE_VERSION.as_bytes()),
        (5, b"projection_freeze_version", PROJECTION_FREEZE_VERSION.as_bytes()),
        (6, b"qualification_wire_version", QUALIFICATION_WIRE_VERSION.as_bytes()),
    ];
    array(rows.into_iter().map(|(index, name, value)| {
        array([uint(index), bytes(name), bytes(value)])
    }))
}

fn semantic_source_roots_object() -> Cbor {
    array([
        bytes(frozen_root("da5ed2db33a88a0912d5003999f787cc26ba18564876615773a82bb742d9f8ae")),
        bytes(frozen_root("922e48ada22dfa8621a4d516e07ec9aa7dc8fc10c165d1cafc963575aed5ec03")),
        bytes(frozen_root("64c9415f7759eec140e439030c5a5374851b9024d7d4849b52b995704ba76ff1")),
        bytes(frozen_root("5de72fc51e27e5501561ffda6b05522f4941d1a13c4b324f5edcc15fa584a0bd")),
        bytes(frozen_root("ef0008912962de9da322eaeea6e421e1e58d16be152f968298774af0fd3249ab")),
        bytes(frozen_root("b7ec5e860a007469b8a1b3930f17c130f59a800d2a832dfd438d18a75538ff99")),
        bytes(frozen_root("ee198614e94cf425202f9c667836fc6ad61fda02c9439a689eb90012c5798ad2")),
    ])
}

fn qualification_wire_profile_object() -> Cbor {
    let schema_registry: [(u64, &[u8], u64); 8] = [
        (0x3a00, Q05B_FULL_LEAF_ROW_SCHEMA, 8),
        (0x3a01, Q05B_FULL_LEAF_MANIFEST_SCHEMA, 8),
        (0x3a02, Q05B_PARTITION_EVIDENCE_SCHEMA, 10),
        (0x3a03, Q05B_SIDECAR_MANIFEST_SCHEMA, 5),
        (0x3a04, Q05B_NODE3_GOLDEN_MANIFEST_SCHEMA, 21),
        (0x3a05, b"hegel-q05b-qualification-candidate-receipt/1", 25),
        (0x3a06, b"hegel-q05b-qualification-receipt/1", 12),
        (0x3a07, Q05B_BOUNDED_NODE3_STATE_SCHEMA, 26),
    ];
    let hash_domains: [(u64, &[u8], &[u8]); 12] = [
        (1, b"NODE3_PARTITION_EVIDENCE", Q05B_PARTITION_EVIDENCE_DOMAIN),
        (2, b"SIDECAR_MANIFEST", Q05B_SIDECAR_MANIFEST_DOMAIN),
        (3, b"NODE3_GOLDEN_MANIFEST", Q05B_NODE3_GOLDEN_MANIFEST_DOMAIN),
        (4, b"BOUNDED_NODE3_STATE", Q05B_BOUNDED_NODE3_STATE_DOMAIN),
        (5, b"PREDICATE_REGISTRY", b"HEGEL/Q05B/QUALIFICATION/PREDICATE_REGISTRY/V1"),
        (6, b"TAG_REGISTRY", Q05B_TAG_REGISTRY_DOMAIN),
        (7, b"PRE_RECEIPT_EVIDENCE", b"HEGEL/Q05B/QUALIFICATION/PRE_RECEIPT_EVIDENCE/V1"),
        (8, b"PRE_RECEIPT", b"HEGEL/Q05B/QUALIFICATION/PRE_RECEIPT/V1"),
        (9, b"CANDIDATE_RECEIPT", b"HEGEL/Q05B/QUALIFICATION/CANDIDATE_RECEIPT/V1"),
        (10, b"PREDICATE20_EVIDENCE", b"HEGEL/Q05B/QUALIFICATION/PREDICATE20_EVIDENCE/V1"),
        (11, b"FINAL_RECEIPT", b"HEGEL/Q05B/QUALIFICATION/FINAL_RECEIPT/V1"),
        (12, b"WIRE_PROFILE", b"HEGEL/Q05B/QUALIFICATION/WIRE_PROFILE/V1"),
    ];
    let failures: [&[u8]; 25] = [
        b"REJECT_Q05B_UINT",
        b"REJECT_Q05B_BYTES",
        b"REJECT_Q05B_ARRAY",
        b"REJECT_Q05B_CBOR",
        b"REJECT_Q05B_Q1_AUTHORITY",
        b"REJECT_Q05B_LEAF_ROW",
        b"REJECT_Q05B_LEAF_AST",
        b"REJECT_Q05B_LEAF_MANIFEST",
        b"REJECT_Q05B_LEAF_ORDER",
        b"REJECT_Q05B_NODE3_SCOPE",
        b"REJECT_Q05B_PARTITION",
        b"REJECT_Q05B_BOUNDED_STATE",
        b"REJECT_Q05B_SIDECAR_PATH",
        b"REJECT_Q05B_SIDECAR_KIND",
        b"REJECT_Q05B_SIDECAR",
        b"REJECT_Q05B_SIDECAR_REPLAY",
        b"REJECT_Q05B_GOLDEN",
        b"REJECT_Q05B_PREDICATE",
        b"REJECT_Q05B_CANDIDATE_RECEIPT",
        b"REJECT_Q05B_RECEIPT",
        b"REJECT_Q05B_RECEIPT_BINDING",
        b"REJECT_Q05B_ACTOR_STDOUT",
        b"FAIL_SHA256_PREIMAGE_COLLISION",
        b"INCONCLUSIVE_Q05B_OUTPUT_LIMIT",
        b"INCONCLUSIVE_Q05B_SCRATCH_LIMIT",
    ];
    let authority_fields: [&[u8]; 16] = [
        b"q1_state_id",
        b"q1_gate_count",
        b"q1_gate_mask",
        b"q1_gate_total",
        b"q1_output_slot_count",
        b"q1_output_slots",
        b"q1_receipt_or_null",
        b"q2_state_id",
        b"m3_formal_roots_or_null",
        b"formal_fixed_point_claimed",
        b"formal_fixed_point_tag_or_null",
        b"target_truth_accessed",
        b"split_accessed",
        b"role_evaluation_performed",
        b"outside_certificate_issued",
        b"active_transition_allowed",
    ];
    let actor_fields: [&[u8]; 21] = [
        b"action_id", b"actor_id", b"file_count", b"implementation_id",
        b"neutral_manifest_length", b"neutral_manifest_raw_sha256",
        b"neutral_manifest_relative_path", b"neutral_manifest_root",
        b"q1_formal_roots", b"q1_gate_count", b"q1_gate_mask", b"q1_output_slots",
        b"q1_state", b"runtime_identity_sha256", b"sidecar_manifest_length",
        b"sidecar_manifest_raw_sha256", b"sidecar_manifest_relative_path",
        b"sidecar_manifest_root", b"source_identity_sha256", b"schema_version", b"status",
    ];
    let output_paths: [&[u8]; 5] = [
        Q05B_FULL_LEAF_PATH,
        Q05B_ODD_EVIDENCE_PATH,
        Q05B_SINK_EVIDENCE_PATH,
        Q05B_SIDECAR_PATH,
        Q05B_NODE3_GOLDEN_PATH,
    ];
    array([
        uint(1),
        bytes(b"hegel-q05b-qualification-wire-profile/1"),
        bytes(QUALIFICATION_WIRE_VERSION.as_bytes()),
        bytes(q05b_tag_registry_root()),
        q05b_tag_registry_object(),
        array(schema_registry.into_iter().map(|(tag, schema, fields)| {
            array([uint(tag), bytes(schema), uint(fields)])
        })),
        array(hash_domains.into_iter().map(|(index, name, domain)| {
            array([uint(index), bytes(name), bytes(domain)])
        })),
        array(failures.into_iter().map(bytes)),
        array(output_paths.into_iter().map(|path| {
            array([bytes(path), uint(Q05B_OUTPUT_FILE_MODE)])
        })),
        array([uint(3), uint(3), uint(4)]),
        array([
            uint(16_777_207),
            uint(16_777_212),
            uint(16_777_216),
            uint(16_777_208),
            uint(16_777_217),
        ]),
        array(authority_fields.into_iter().map(bytes)),
        q1_authority_object(),
        bytes(qualification_predicate_registry_root()),
        qualification_predicate_registry_object(),
        array([
            bytes(EXTERNAL_SORT_TRACE_SCHEMA),
            uint(6),
            array([
                bytes(b"version"), bytes(b"schema_id"), bytes(b"projection_object"),
                bytes(b"ordered_rows"), bytes(b"run_manifests"), bytes(b"scratch_events"),
            ]),
        ]),
        array([
            bytes(COUNTING_DISCARD_STREAM_SCHEMA),
            uint(15),
            array([
                bytes(b"version"), bytes(b"schema_id"), bytes(b"input_signature_id"),
                bytes(b"universe_root"), bytes(b"stream_kind_id"), bytes(b"record_count"),
                bytes(b"canonical_record_payload_bytes"), bytes(b"framed_stream_bytes"),
                bytes(b"chunk_count"), bytes(b"descriptor_object"),
                bytes(b"chunk_manifest_objects"), bytes(b"external_sort_projection_object"),
                bytes(b"diagnostic_commitment"), bytes(b"retained_framed_blob_count"),
                bytes(b"retained_framed_blob_bytes"),
            ]),
            array([
                bytes(b"MATERIALIZED_FRAMED_BLOBS_STRICT_REPLAY_AND_REENCODE_EXACT"),
                bytes(b"RECORD_COUNT_EQUALS_MATERIALIZED_DESCRIPTOR_SLOT_3"),
                bytes(b"FRAMED_STREAM_BYTES_EQUALS_MATERIALIZED_DESCRIPTOR_SLOT_5"),
                bytes(b"CHUNK_COUNT_EQUALS_MATERIALIZED_DESCRIPTOR_SLOT_6"),
                bytes(b"DESCRIPTOR_OBJECT_EQUALS_MATERIALIZED_PROJECTED_SLOT_5"),
                bytes(b"CHUNK_MANIFEST_OBJECTS_EQUAL_MATERIALIZED_PROJECTED_SLOT_6"),
                bytes(b"EXTERNAL_SORT_PROJECTION_EQUALS_MATERIALIZED_PROJECTED_SLOT_7"),
                bytes(b"DIAGNOSTIC_COMMITMENT_EQUALS_MATERIALIZED_PROJECTED_SLOT_8"),
                bytes(b"COUNTING_SINK_REENCODES_ORDERED_FORMAL_RECORDS_INDEPENDENTLY"),
                bytes(b"RETAINED_FRAMED_BLOB_COUNT_AND_BYTES_EQUAL_ZERO_ZERO"),
            ]),
            Cbor::Bool(true),
        ]),
        array([
            bytes(b"hegel-q05b-actor-envelope/1"),
            bytes(b"bounded-node3-golden-v1"),
            bytes(b"BOUNDED_NODE3_CANDIDATE_EMITTED_NOT_QUALIFIED"),
            array([
                array([bytes(b"PYTHON_ENDPOINT"), bytes(b"HEGEL_Q1_BOUNDED_NODE3_PROJECTION_PYTHON_V1")]),
                array([bytes(b"RUST_ENDPOINT"), bytes(b"HEGEL_Q1_BOUNDED_NODE3_PROJECTION_RUST_V1")]),
                array([bytes(b"TRUSTED_HOST_REPLAY"), bytes(b"HEGEL_Q1_BOUNDED_NODE3_PROJECTION_HOST_REPLAY_V1")]),
            ]),
            array(actor_fields.into_iter().map(bytes)),
        ]),
    ])
}

fn qualification_wire_profile_root() -> [u8; 32] {
    content_hash(
        b"HEGEL/Q05B/QUALIFICATION/WIRE_PROFILE/V1",
        &qualification_wire_profile_object(),
    )
}

fn full_leaf_neutral_object() -> Result<(Cbor, [u8; 32]), OracleError> {
    let rows = frozen_leaf_manifest()?
        .into_iter()
        .map(|leaf| {
            array([
                uint(1),
                uint(Q05B_FULL_LEAF_ROW_TAG),
                bytes(Q05B_FULL_LEAF_ROW_SCHEMA),
                uint(u64::from(leaf.coverage_code)),
                uint(OutputSortId::from_sort(leaf.canonical.output_sort) as u64),
                uint(u64::from(leaf.canonical.root_operator_id)),
                bytes(&leaf.canonical.canonical_cbor),
                bytes(leaf.canonical.canonical_ast_hash),
            ])
        })
        .collect::<Vec<_>>();
    let manifest_root = rfc6962(&rows.iter().map(encode).collect::<Vec<_>>());
    Ok((
        array([
            uint(1),
            uint(Q05B_FULL_LEAF_MANIFEST_TAG),
            bytes(Q05B_FULL_LEAF_MANIFEST_SCHEMA),
            bytes(DSL_VERSION.as_bytes()),
            bytes(DSL_FREEZE_VERSION.as_bytes()),
            uint(FROZEN_LEAF_COUNT as u64),
            bytes(manifest_root),
            array(rows),
        ]),
        manifest_root,
    ))
}

fn partition_evidence_object(partition: &PartitionSemanticResult) -> Result<Cbor, OracleError> {
    let universe_root = decode_hex_root(&partition.universe_root)?;
    let stream_rows = partition
        .archives
        .streams
        .iter()
        .map(|stream| {
            array([
                uint(stream.stream_kind_id),
                stream.projected_object.clone(),
                array(stream.framed_blobs.iter().map(bytes)),
                stream.external_sort.trace_object.clone(),
                stream.counting_discard_object.clone(),
            ])
        })
        .collect::<Vec<_>>();
    if partition.archives.coverage_sidecar_rows.len() != 846 || stream_rows.len() != 4 {
        return Err(OracleError::new(
            "REJECT_Q05B_PARTITION",
            "neutral partition preimage cardinality differs",
        ));
    }
    Ok(array([
        uint(1),
        uint(Q05B_PARTITION_EVIDENCE_TAG),
        bytes(Q05B_PARTITION_EVIDENCE_SCHEMA),
        uint(partition.input_signature_id),
        bytes(universe_root),
        partition.archives.record_set_object.clone(),
        uint(846),
        array(partition.archives.coverage_sidecar_rows.iter().cloned()),
        uint(4),
        array(stream_rows),
    ]))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Q05BNeutralSidecarBundle {
    pub full_leaf_manifest: Vec<u8>,
    pub odd_partition_evidence: Vec<u8>,
    pub sink_partition_evidence: Vec<u8>,
    pub sidecar_manifest: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct NeutralArtifactSummary {
    pub relative_path: String,
    pub canonical_byte_length: u64,
    pub raw_sha256: String,
    pub content_root: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct Q05BNeutralSidecarSummary {
    pub full_leaf_manifest_rfc6962_root: String,
    pub preimages: Vec<NeutralArtifactSummary>,
    pub sidecar_manifest: NeutralArtifactSummary,
}

fn sidecar_file_row(
    index: usize,
    path: &[u8],
    content_kind_id: u64,
    domain: &[u8],
    object: &Cbor,
) -> Cbor {
    let payload = encode(object);
    array([
        uint(index as u64),
        bytes(path),
        uint(content_kind_id),
        uint(Q05B_OUTPUT_FILE_MODE),
        uint(payload.len() as u64),
        bytes(sha256(&[&payload])),
        bytes(domain),
        bytes(content_hash(domain, object)),
    ])
}

fn build_neutral_sidecar_bundle(
    partitions: &[PartitionSemanticResult],
) -> Result<Q05BNeutralSidecarBundle, OracleError> {
    if partitions.len() != 2
        || partitions[0].input_signature_id != 1
        || partitions[1].input_signature_id != 2
    {
        return Err(OracleError::new(
            "REJECT_Q05B_PARTITION",
            "odd and sink partitions are not exact and ordered",
        ));
    }
    let (full_leaf, _) = full_leaf_neutral_object()?;
    let odd = partition_evidence_object(&partitions[0])?;
    let sink = partition_evidence_object(&partitions[1])?;
    let sidecar = array([
        uint(1),
        uint(Q05B_SIDECAR_MANIFEST_TAG),
        bytes(Q05B_SIDECAR_MANIFEST_SCHEMA),
        uint(3),
        array([
            sidecar_file_row(
                0,
                Q05B_FULL_LEAF_PATH,
                1,
                Q05B_FULL_LEAF_CONTENT_DOMAIN,
                &full_leaf,
            ),
            sidecar_file_row(
                1,
                Q05B_ODD_EVIDENCE_PATH,
                2,
                Q05B_PARTITION_EVIDENCE_DOMAIN,
                &odd,
            ),
            sidecar_file_row(
                2,
                Q05B_SINK_EVIDENCE_PATH,
                3,
                Q05B_PARTITION_EVIDENCE_DOMAIN,
                &sink,
            ),
        ]),
    ]);
    let bundle = Q05BNeutralSidecarBundle {
        full_leaf_manifest: encode(&full_leaf),
        odd_partition_evidence: encode(&odd),
        sink_partition_evidence: encode(&sink),
        sidecar_manifest: encode(&sidecar),
    };
    validate_neutral_sidecar_shape(&bundle)?;
    Ok(bundle)
}

fn validate_neutral_sidecar_shape(bundle: &Q05BNeutralSidecarBundle) -> Result<(), OracleError> {
    let full_leaf = decode_strict(&bundle.full_leaf_manifest)?;
    let odd = decode_strict(&bundle.odd_partition_evidence)?;
    let sink = decode_strict(&bundle.sink_partition_evidence)?;
    let supplied_sidecar = decode_strict(&bundle.sidecar_manifest)?;
    let expected_sidecar = array([
        uint(1),
        uint(Q05B_SIDECAR_MANIFEST_TAG),
        bytes(Q05B_SIDECAR_MANIFEST_SCHEMA),
        uint(3),
        array([
            sidecar_file_row(
                0,
                Q05B_FULL_LEAF_PATH,
                1,
                Q05B_FULL_LEAF_CONTENT_DOMAIN,
                &full_leaf,
            ),
            sidecar_file_row(
                1,
                Q05B_ODD_EVIDENCE_PATH,
                2,
                Q05B_PARTITION_EVIDENCE_DOMAIN,
                &odd,
            ),
            sidecar_file_row(
                2,
                Q05B_SINK_EVIDENCE_PATH,
                3,
                Q05B_PARTITION_EVIDENCE_DOMAIN,
                &sink,
            ),
        ]),
    ]);
    if supplied_sidecar != expected_sidecar {
        return Err(OracleError::new(
            "REJECT_Q05B_SIDECAR_REPLAY",
            "sidecar bytes, lengths, SHA-256, domains, or content roots differ",
        ));
    }
    Ok(())
}

fn artifact_summary(path: &[u8], domain: &[u8], payload: &[u8]) -> Result<NeutralArtifactSummary, OracleError> {
    let object = decode_strict(payload)?;
    Ok(NeutralArtifactSummary {
        relative_path: String::from_utf8(path.to_vec()).expect("frozen ASCII path"),
        canonical_byte_length: payload.len() as u64,
        raw_sha256: hex_encode(&sha256(&[payload])),
        content_root: hex_encode(&content_hash(domain, &object)),
    })
}

impl Q05BNeutralSidecarBundle {
    pub fn summary(&self) -> Result<Q05BNeutralSidecarSummary, OracleError> {
        validate_neutral_sidecar_shape(self)?;
        let (_, full_leaf_root) = full_leaf_neutral_object()?;
        Ok(Q05BNeutralSidecarSummary {
            full_leaf_manifest_rfc6962_root: hex_encode(&full_leaf_root),
            preimages: vec![
                artifact_summary(
                    Q05B_FULL_LEAF_PATH,
                    Q05B_FULL_LEAF_CONTENT_DOMAIN,
                    &self.full_leaf_manifest,
                )?,
                artifact_summary(
                    Q05B_ODD_EVIDENCE_PATH,
                    Q05B_PARTITION_EVIDENCE_DOMAIN,
                    &self.odd_partition_evidence,
                )?,
                artifact_summary(
                    Q05B_SINK_EVIDENCE_PATH,
                    Q05B_PARTITION_EVIDENCE_DOMAIN,
                    &self.sink_partition_evidence,
                )?,
            ],
            sidecar_manifest: artifact_summary(
                Q05B_SIDECAR_PATH,
                Q05B_SIDECAR_MANIFEST_DOMAIN,
                &self.sidecar_manifest,
            )?,
        })
    }
}

fn bounded_node3_state_object(
    partition: &PartitionSemanticResult,
    partition_evidence: &Cbor,
) -> Result<Cbor, OracleError> {
    let primary_counts = array([
        uint(partition.raw_application_count),
        uint(partition.strict_admitted_application_count),
        uint(partition.rewrite_collapse_count),
        uint(partition.behavior_class_count),
        uint(partition.signature_cohort_count),
        uint(partition.continuation_bank_point_count),
        uint(partition.visible_frontier_point_count),
    ]);
    let peak_work_queue = partition
        .depth_barriers
        .iter()
        .map(|row| row.eligible_raw_application_count)
        .max()
        .unwrap_or(0);
    Ok(array([
        uint(1),
        uint(Q05B_BOUNDED_NODE3_STATE_TAG),
        bytes(Q05B_BOUNDED_NODE3_STATE_SCHEMA),
        bytes(Q05B_SCOPE_ID),
        uint(partition.input_signature_id),
        bytes(decode_hex_root(&partition.universe_root)?),
        uint(partition.universe_row_count),
        uint(3),
        uint(3),
        uint(4),
        bytes(NODE3_TERMINAL_STATUS),
        Cbor::Bool(true),
        Cbor::Bool(true),
        Cbor::Bool(true),
        Cbor::Bool(true),
        Cbor::Bool(true),
        primary_counts,
        uint(partition.maximum_bank_points_per_class),
        uint(partition.maximum_frontier_points_per_class),
        uint(peak_work_queue),
        uint(5),
        bytes(decode_hex_root(&partition.archives.coverage_archive_root)?),
        bytes(content_hash(Q05B_PARTITION_EVIDENCE_DOMAIN, partition_evidence)),
        Cbor::Bool(false),
        Cbor::Null,
        q1_authority_object(),
    ]))
}

fn partition_summary_object(
    partition: &PartitionSemanticResult,
    partition_evidence: &Cbor,
) -> Result<Cbor, OracleError> {
    let payload = encode(partition_evidence);
    let stream_summaries = partition
        .archives
        .streams
        .iter()
        .map(|stream| {
            array([
                uint(stream.stream_kind_id),
                uint(stream.record_count),
                bytes(decode_hex_root(&stream.archive_root).expect("stream archive root")),
                uint(stream.framed_stream_bytes),
                uint(stream.chunk_count),
                bytes(
                    decode_hex_root(&stream.chunk_manifest_archive_root)
                        .expect("chunk manifest root"),
                ),
                bytes(
                    decode_hex_root(&stream.projected_stream_commitment)
                        .expect("projected stream root"),
                ),
                bytes(content_hash(
                    EXTERNAL_SORT_PROJECTION_DOMAIN,
                    &stream.external_sort.projection_object,
                )),
                uint(stream.external_sort.charged_scratch_high_water_bytes),
            ])
        })
        .collect::<Vec<_>>();
    let peak_work_queue = partition
        .depth_barriers
        .iter()
        .map(|row| row.eligible_raw_application_count)
        .max()
        .unwrap_or(0);
    Ok(array([
        uint(partition.input_signature_id),
        bytes(decode_hex_root(&partition.universe_root)?),
        uint(partition.universe_row_count),
        uint(3),
        uint(3),
        uint(4),
        bytes(NODE3_TERMINAL_STATUS),
        uint(partition.raw_application_count),
        uint(partition.strict_admitted_application_count),
        uint(partition.rewrite_collapse_count),
        uint(partition.behavior_class_count),
        uint(partition.signature_cohort_count),
        uint(partition.continuation_bank_point_count),
        uint(partition.visible_frontier_point_count),
        uint(partition.maximum_bank_points_per_class),
        uint(partition.maximum_frontier_points_per_class),
        uint(peak_work_queue),
        bytes(decode_hex_root(&partition.archives.snapshot_record_set_root)?),
        uint(846),
        bytes(decode_hex_root(&partition.archives.coverage_archive_root)?),
        uint(4),
        array(stream_summaries),
        uint(payload.len() as u64),
        bytes(sha256(&[&payload])),
        bytes(content_hash(Q05B_PARTITION_EVIDENCE_DOMAIN, partition_evidence)),
    ]))
}

fn build_node3_golden_manifest(
    partitions: &[PartitionSemanticResult],
    sidecar: &Q05BNeutralSidecarBundle,
) -> Result<Vec<u8>, OracleError> {
    if partitions.len() != 2
        || partitions[0].input_signature_id != 1
        || partitions[0].universe_row_count != 480
        || partitions[1].input_signature_id != 2
        || partitions[1].universe_row_count != 85
    {
        return Err(OracleError::new(
            "REJECT_Q05B_GOLDEN",
            "exact ordered odd480 and sink85 partitions are required",
        ));
    }
    let (_, full_leaf_root) = full_leaf_neutral_object()?;
    let odd_root = decode_hex_root(&partitions[0].universe_root)?;
    let sink_root = decode_hex_root(&partitions[1].universe_root)?;
    let semantic_root = q1_semantic_binding_root(full_leaf_root, odd_root, sink_root);
    let projection_root = q1_projection_profile_root(semantic_root);
    let sidecar_object = decode_strict(&sidecar.sidecar_manifest)?;
    let sidecar_root = content_hash(Q05B_SIDECAR_MANIFEST_DOMAIN, &sidecar_object);
    let evidence = [
        decode_strict(&sidecar.odd_partition_evidence)?,
        decode_strict(&sidecar.sink_partition_evidence)?,
    ];
    let mut state_rows = Vec::new();
    let mut summaries = Vec::new();
    for (partition, evidence_object) in partitions.iter().zip(evidence.iter()) {
        let state = bounded_node3_state_object(partition, evidence_object)?;
        state_rows.push(array([
            uint(partition.input_signature_id),
            state.clone(),
            bytes(content_hash(Q05B_BOUNDED_NODE3_STATE_DOMAIN, &state)),
        ]));
        summaries.push(partition_summary_object(partition, evidence_object)?);
    }
    let manifest = array([
        uint(1),
        uint(Q05B_NODE3_GOLDEN_MANIFEST_TAG),
        bytes(Q05B_NODE3_GOLDEN_MANIFEST_SCHEMA),
        bytes(QUALIFICATION_WIRE_VERSION.as_bytes()),
        bytes(Q05B_SCOPE_ID),
        uint(3),
        uint(3),
        uint(4),
        version_binding_rows_object(),
        bytes(q05b_tag_registry_root()),
        bytes(qualification_wire_profile_root()),
        semantic_source_roots_object(),
        bytes(semantic_root),
        bytes(projection_root),
        bytes(full_leaf_root),
        bytes(sidecar_root),
        uint(2),
        array(state_rows),
        uint(2),
        array(summaries),
        q1_authority_object(),
    ]);
    Ok(encode(&manifest))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Q05BNeutralBundle {
    pub sidecar: Q05BNeutralSidecarBundle,
    pub node3_golden_manifest: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct Q05BNeutralBundleSummary {
    pub sidecar: Q05BNeutralSidecarSummary,
    pub q1_semantic_binding_root: String,
    pub q1_projection_profile_root: String,
    pub qualification_tag_registry_root: String,
    pub qualification_predicate_registry_root: String,
    pub qualification_wire_profile_root: String,
    pub bounded_node3_state_roots: Vec<String>,
    pub node3_golden_manifest: NeutralArtifactSummary,
}

impl Q05BNeutralBundle {
    pub fn summary(&self) -> Result<Q05BNeutralBundleSummary, OracleError> {
        let (_, leaf_root) = full_leaf_neutral_object()?;
        let (odd_root, sink_root) = regenerated_universe_root_bytes();
        let semantic_root = q1_semantic_binding_root(leaf_root, odd_root, sink_root);
        let projection_root = q1_projection_profile_root(semantic_root);
        let manifest = decode_strict(&self.node3_golden_manifest)?;
        let Cbor::Array(fields) = &manifest else {
            return Err(OracleError::new("REJECT_Q05B_GOLDEN", "manifest is not an array"));
        };
        if fields.len() != 21 {
            return Err(OracleError::new(
                "REJECT_Q05B_GOLDEN",
                "manifest field count differs",
            ));
        }
        let Cbor::Array(state_rows) = &fields[17] else {
            return Err(OracleError::new("REJECT_Q05B_GOLDEN", "state rows are not an array"));
        };
        let bounded_node3_state_roots = state_rows
            .iter()
            .map(|row| match row {
                Cbor::Array(values) if values.len() == 3 => match &values[2] {
                    Cbor::Bytes(value) if value.len() == 32 => Ok(hex_encode(value)),
                    _ => Err(OracleError::new("REJECT_Q05B_GOLDEN", "state root differs")),
                },
                _ => Err(OracleError::new("REJECT_Q05B_GOLDEN", "state row differs")),
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Q05BNeutralBundleSummary {
            sidecar: self.sidecar.summary()?,
            q1_semantic_binding_root: hex_encode(&semantic_root),
            q1_projection_profile_root: hex_encode(&projection_root),
            qualification_tag_registry_root: hex_encode(&q05b_tag_registry_root()),
            qualification_predicate_registry_root: hex_encode(
                &qualification_predicate_registry_root(),
            ),
            qualification_wire_profile_root: hex_encode(&qualification_wire_profile_root()),
            bounded_node3_state_roots,
            node3_golden_manifest: artifact_summary(
                Q05B_NODE3_GOLDEN_PATH,
                Q05B_NODE3_GOLDEN_MANIFEST_DOMAIN,
                &self.node3_golden_manifest,
            )?,
        })
    }
}

fn build_neutral_bundle(
    partitions: &[PartitionSemanticResult],
) -> Result<Q05BNeutralBundle, OracleError> {
    let sidecar = build_neutral_sidecar_bundle(partitions)?;
    let node3_golden_manifest = build_node3_golden_manifest(partitions, &sidecar)?;
    Ok(Q05BNeutralBundle {
        sidecar,
        node3_golden_manifest,
    })
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct DiagnosticAuthority {
    pub formal_roots: Option<Vec<String>>,
    pub q1_gate_count: u8,
    pub q1_gate_mask: u32,
    pub q1_gate_total: u8,
    pub q1_state: String,
    pub q2_state: String,
    pub role_evaluation_performed: bool,
    pub split_accessed: bool,
    pub target_truth_accessed: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct GoldenNode3Result {
    pub schema_version: String,
    pub implementation_id: String,
    pub claim: String,
    pub dsl_version: String,
    pub closure_semantics_version: String,
    pub frozen_leaf_count: u64,
    pub coverage_registry_root: String,
    pub partitions: Vec<PartitionSemanticResult>,
    pub neutral: Q05BNeutralBundleSummary,
    pub authority: DiagnosticAuthority,
}

fn compute_node3_partitions() -> Result<Vec<PartitionSemanticResult>, OracleError> {
    let (odd_observations, odd_rows) = odd_universe();
    let (sink_observations, sink_rows) = sink_universe();
    let odd_root = rfc6962(&odd_rows);
    let sink_root = rfc6962(&sink_rows);
    let odd = run_partition(1, odd_root, odd_observations)?;
    let sink = run_partition(2, sink_root, sink_observations)?;
    Ok(vec![odd, sink])
}

pub fn golden_node3_neutral_sidecar_bundle() -> Result<Q05BNeutralSidecarBundle, OracleError> {
    build_neutral_sidecar_bundle(&compute_node3_partitions()?)
}

pub fn golden_node3_neutral_bundle() -> Result<Q05BNeutralBundle, OracleError> {
    build_neutral_bundle(&compute_node3_partitions()?)
}

pub fn replay_golden_node3_neutral_sidecar(
    supplied: &Q05BNeutralSidecarBundle,
) -> Result<(), OracleError> {
    validate_neutral_sidecar_shape(supplied)?;
    let expected = golden_node3_neutral_sidecar_bundle()?;
    if supplied != &expected {
        return Err(OracleError::new(
            "REJECT_Q05B_SIDECAR_REPLAY",
            "neutral sidecar differs from independent Rust regeneration",
        ));
    }
    Ok(())
}

pub fn replay_golden_node3_neutral_bundle(
    supplied: &Q05BNeutralBundle,
) -> Result<(), OracleError> {
    validate_neutral_sidecar_shape(&supplied.sidecar)?;
    decode_strict(&supplied.node3_golden_manifest)?;
    let expected = golden_node3_neutral_bundle()?;
    if supplied != &expected {
        return Err(OracleError::new(
            "REJECT_Q05B_GOLDEN",
            "neutral golden manifest or sidecar differs from independent regeneration",
        ));
    }
    Ok(())
}

pub fn golden_node3() -> Result<GoldenNode3Result, OracleError> {
    let partitions = compute_node3_partitions()?;
    let neutral = build_neutral_bundle(&partitions)?.summary()?;
    Ok(GoldenNode3Result {
        schema_version: SCHEMA_VERSION.to_owned(),
        implementation_id: IMPLEMENTATION_ID.to_owned(),
        claim: CLAIM.to_owned(),
        dsl_version: DSL_VERSION.to_owned(),
        closure_semantics_version: CLOSURE_SEMANTICS_VERSION.to_owned(),
        frozen_leaf_count: FROZEN_LEAF_COUNT as u64,
        coverage_registry_root: coverage_registry_root(),
        partitions,
        neutral,
        authority: DiagnosticAuthority {
            formal_roots: None,
            q1_gate_count: 0,
            q1_gate_mask: 0,
            q1_gate_total: 20,
            q1_state: Q1_STATE.to_owned(),
            q2_state: "NOT_RUN".to_owned(),
            role_evaluation_performed: false,
            split_accessed: false,
            target_truth_accessed: false,
        },
    })
}

pub fn golden_node3_json() -> Result<String, OracleError> {
    serde_json::to_string(&golden_node3()?)
        .map_err(|error| OracleError::new("FAIL_DIAGNOSTIC_JSON", error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::OnceLock;

    fn project_root() -> &'static Path {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .expect("crate must remain under Hegel Machine/rust")
    }

    fn golden() -> &'static GoldenNode3Result {
        static RESULT: OnceLock<GoldenNode3Result> = OnceLock::new();
        RESULT.get_or_init(|| golden_node3().unwrap())
    }

    fn neutral() -> &'static Q05BNeutralSidecarBundle {
        static RESULT: OnceLock<Q05BNeutralSidecarBundle> = OnceLock::new();
        RESULT.get_or_init(|| golden_node3_neutral_sidecar_bundle().unwrap())
    }

    fn full_neutral() -> &'static Q05BNeutralBundle {
        static RESULT: OnceLock<Q05BNeutralBundle> = OnceLock::new();
        RESULT.get_or_init(|| golden_node3_neutral_bundle().unwrap())
    }

    #[test]
    fn build_snapshot_configs_docs_wire_and_embedded_source_identity_are_exact() {
        validate_build_snapshot(project_root()).unwrap();
        let recomputed = source_identity_sha256_from_project_root(project_root()).unwrap();
        assert_eq!(embedded_source_identity_sha256().unwrap(), recomputed);
        assert_eq!(recomputed.len(), 64);
    }

    #[test]
    fn strict_build_config_parser_rejects_duplicate_and_nonfinite_tokens() {
        let valid = fs::read(
            project_root()
                .join("config/phase3_q05b_node3_dual_projection_qualification_v1.json"),
        )
        .unwrap();
        parse_strict_config_json(&valid).unwrap();
        let text = String::from_utf8(valid).unwrap();
        let marker = "  \"freeze_id\":";
        let line = text
            .lines()
            .find(|line| line.starts_with(marker))
            .unwrap();
        let duplicate = text.replacen(line, &format!("{line}\n{line}"), 1);
        assert_eq!(
            parse_strict_config_json(duplicate.as_bytes())
                .unwrap_err()
                .code,
            "FAIL_Q1_PROJECTION_CONFIG_WIRE"
        );
        let nonfinite = text.replacen(
            "    \"q1_gate_count\": 0,",
            "    \"q1_gate_count\": NaN,",
            1,
        );
        assert_eq!(
            parse_strict_config_json(nonfinite.as_bytes())
                .unwrap_err()
                .code,
            "FAIL_Q1_PROJECTION_CONFIG_WIRE"
        );
    }

    #[test]
    fn commit_a_actual_authority_is_exact_and_type_strict() {
        let payload = fs::read(
            project_root()
                .join("config/phase3_q05b_node3_dual_projection_qualification_v1.json"),
        )
        .unwrap();
        let primary = parse_strict_config_json(&payload).unwrap();
        require_commit_a_actual_authority_v1(&primary).unwrap();

        let mut wrong_status = primary.clone();
        wrong_status["engineering_status"] =
            serde_json::json!("ACTUAL_IMPLEMENTED_CONDITIONALLY_ADMITTED_EXECUTED");
        let error = require_commit_a_actual_authority_v1(&wrong_status).unwrap_err();
        assert_eq!(error.code, "FAIL_Q1_PROJECTION_CONFIG_BINDING");

        let mut status_bool_alias = primary.clone();
        status_bool_alias["engineering_status"] = serde_json::json!(true);
        let error =
            require_commit_a_actual_authority_v1(&status_bool_alias).unwrap_err();
        assert_eq!(error.code, "FAIL_Q1_PROJECTION_CONFIG_BINDING");

        let mut missing_ownership_authority = primary.clone();
        missing_ownership_authority["actual_preconditions"]
            ["attempt_unique_docker_execution_authority_required"] = serde_json::json!(false);
        let error = require_commit_a_actual_authority_v1(&missing_ownership_authority)
            .unwrap_err();
        assert_eq!(error.code, "FAIL_Q1_PROJECTION_CONFIG_BINDING");

        let mut bool_alias = primary;
        bool_alias["actual_preconditions"]
            ["docker_cleanup_owned_cid_only_required"] = serde_json::json!(1);
        let error = require_commit_a_actual_authority_v1(&bool_alias).unwrap_err();
        assert_eq!(error.code, "FAIL_Q1_PROJECTION_CONFIG_BINDING");
    }

    #[test]
    fn canonical_actor_json_is_ascii_sorted_and_has_one_lf() {
        let value = serde_json::json!({"z": "中文", "a": [null, false, 3]});
        let encoded = canonical_json_line(&value).unwrap();
        assert_eq!(
            encoded,
            b"{\"a\":[null,false,3],\"z\":\"\\u4e2d\\u6587\"}\n"
        );
    }

    #[test]
    fn production_universe_roots_are_regenerated() {
        let (odd, sink) = regenerated_universe_roots();
        assert_eq!(odd, EXPECTED_ODD_UNIVERSE_ROOT);
        assert_eq!(sink, EXPECTED_SINK_UNIVERSE_ROOT);
    }

    #[test]
    fn full_leaf_manifest_is_independently_generated() {
        let leaves = frozen_leaf_manifest().unwrap();
        assert_eq!(leaves.len(), 810);
        assert_eq!(leaves.first().unwrap().coverage_code, 0);
        assert_eq!(leaves.last().unwrap().coverage_code, 809);
    }

    #[test]
    fn coverage_registry_has_exact_846_row_root() {
        assert_eq!(coverage_registry_root(), EXPECTED_COVERAGE_REGISTRY_ROOT);
    }

    #[test]
    fn node3_observations_match_the_frozen_candidate_milestone() {
        let result = golden();
        let odd = &result.partitions[0];
        let sink = &result.partitions[1];
        assert_eq!(
            (
                odd.raw_application_count,
                odd.behavior_class_count,
                odd.visible_frontier_point_count,
                odd.continuation_bank_point_count,
            ),
            (1048, 40, 59, 110)
        );
        assert_eq!(
            (
                sink.raw_application_count,
                sink.behavior_class_count,
                sink.visible_frontier_point_count,
                sink.continuation_bank_point_count,
            ),
            (1101, 28, 84, 144)
        );
        assert_eq!(odd.coverage.len(), 846);
        assert_eq!(sink.coverage.len(), 846);
        assert_eq!(result.authority.q1_state, "NOT_RUN");
        assert_eq!(result.authority.q1_gate_count, 0);
        assert!(result.authority.formal_roots.is_none());
    }

    #[test]
    fn candidate_record_sets_and_all_eight_stream_roots_match_cross_probe() {
        let result = golden();
        let odd = &result.partitions[0].archives;
        let sink = &result.partitions[1].archives;
        assert_eq!(
            odd.snapshot_record_set_root,
            "99f288b6ff434f2bc2d618f7518b276aab7d945a60c9b73d0094345970b6d791"
        );
        assert_eq!(
            sink.snapshot_record_set_root,
            "1a4c488a6f83eba515b0f67eb4b6ec6fcf2399103ac7a5ef09d2d7e391ac7feb"
        );
        assert_eq!(
            [
                odd.program_archive_root.as_str(),
                odd.cohort_archive_root.as_str(),
                odd.class_archive_root.as_str(),
                odd.coverage_archive_root.as_str(),
            ],
            [
                "c13a827f47949732a2cc252b0f81691e1c802d6ce5fe835826adfa378321c87b",
                "2883f3e3c5cbf5bfca4a5e0cee3d32de5ab03e2ef39c967c0eb7937d57c8515c",
                "76f0a88caebbb53ea1b386c3e281e7ab5b2036f201b959f6eda5165723417c30",
                "fe056f876706971d5bd15959325e4fdcc164a5ad66303d0a61bfbff88fae929c",
            ]
        );
        assert_eq!(
            [
                sink.program_archive_root.as_str(),
                sink.cohort_archive_root.as_str(),
                sink.class_archive_root.as_str(),
                sink.coverage_archive_root.as_str(),
            ],
            [
                "67a13e35ac7863f0348d2d1933685f27d09d45de5a2458b3c3185cd06bb078dd",
                "3cbd765f4ff301df0fec87eee61f4b69bf0fe5af3483bf8f2b894379ad047972",
                "7d35bad68c4768c6666e403d03d62f24c9aaec6414e1ab84513a05a7a661492d",
                "bdc7821d3a96087c1f0b97d7d6e0317e953d00dbf748d2e774eb3040ff0dc6ea",
            ]
        );
    }

    #[test]
    fn all_b3_projection_roots_and_trace_preimages_match_cross_probe() {
        let expected = [
            [
                (
                    "7771d4662573ac7f721f204dd0c0d26fb43e674d1f74c9df7f2484cf4eee9cb8",
                    "dcfdd0cc2c6662bf0ede92975ec5c45d1e2fe679ffe786c5fabf524976d1f565",
                    "0489c519c911180de18197ed0bac37207dbb1e9c4caf32e0aaddf5db7f702c4d",
                ),
                (
                    "6ccf958e86e1265911f2ccfd8ab4d98f9753e3c6042a490fffdadf27b833af44",
                    "86bca9fa0f3ef83f8a6a895080752ff1d5999f6e2811936f0b4d480b088aff7f",
                    "006e820758c5a27696b10fd5ec34fae5b56cd5cbf1dd60000eb43d8c01252fa1",
                ),
                (
                    "920f993599b92c9d05e48e2c71c838ad2a8c47cf8d0d041ce1d186cfd286fa50",
                    "c9dac32e2a5625e31b4d64a3487a0354e57f8dfa8cb65a460215e6aee68a98dd",
                    "ff3bf80d7b5d883cdbbf3babdd172e15a5f4f9536a2a12b26e33fa9861a9b710",
                ),
                (
                    "727797a029bdea79c7cb9ff5e85321520928056b630b25ed2e49215cb6b14a5a",
                    "8768eff30d94161db322730f9e936597c5de168cccfa06cf9694633ea43a460e",
                    "0558238849ab04470880a5e0c19b9bb8ccf1ea4d01bab54297d496dcbb2045fa",
                ),
            ],
            [
                (
                    "fd7d82fc8d3bb0f41c07a438a4a76292f6a1a8c9e84ab812008d48f3d233914a",
                    "ea40c0b19d2ac27abcc6a0974b561cb4a126ed306e50e2a5936ba536390216ce",
                    "fa711a730619d7153d8c3b7e6fca24e2b89d1fa19761c27a08628286ca3c300b",
                ),
                (
                    "fe1a6d4c7712fd70f137473a5d40c512b8deaf5c308fc29e03bb894c7f0892a7",
                    "352c7fedf23f2f9d89f63d9a1b874888608be1ee4a8cf9ca000e51d6ca954453",
                    "0824b01e3652694649fc0b93f96dec6465c9b9de6ad3d57bde2b130eefbdbeb8",
                ),
                (
                    "01b414e800c42803d92d48c11ec2c44b2509a1592bf532954cb11dd01a95515d",
                    "d52de0239fe1d1b3cadba59c32b7f7eb0fa76e9ece7e23c5f454102070e5537b",
                    "4a289ec2eeb355dea073856c232d9638b3f177ab163e2987a47aca1b943e1aaa",
                ),
                (
                    "9cb8360549c9eeadc4fdb56545ec2358f2db940d3c902c90156de5131d9060c5",
                    "f447fa48e630aca20e2c92ffae491987bfd8ede863c0fa55ec0800906184ea01",
                    "77da55805ee2d3a90a3ab5bc20b250a61ac3c89ef112c928e266998de8b2e88b",
                ),
            ],
        ];
        for (partition, expected_partition) in golden().partitions.iter().zip(expected) {
            for (stream, (chunk_root, sort_root, projected_root)) in
                partition.archives.streams.iter().zip(expected_partition)
            {
                assert_eq!(stream.chunk_manifest_archive_root, chunk_root);
                assert_eq!(stream.external_sort.diagnostic_root, sort_root);
                assert_eq!(stream.projected_stream_commitment, projected_root);
                let Cbor::Array(trace) = &stream.external_sort.trace_object else {
                    panic!("external-sort trace is not an array");
                };
                assert_eq!(trace.len(), 6);
                assert_eq!(trace[2], stream.external_sort.projection_object);
                let Cbor::Array(rows) = &trace[3] else {
                    panic!("ordered rows are not an array");
                };
                let Cbor::Array(manifests) = &trace[4] else {
                    panic!("run manifests are not an array");
                };
                let Cbor::Array(events) = &trace[5] else {
                    panic!("scratch events are not an array");
                };
                assert_eq!(rows.len() as u64, stream.record_count);
                assert_eq!(
                    manifests.len(),
                    external_sort_merge_shape(stream.external_sort.initial_run_count as usize)
                        .into_iter()
                        .sum::<usize>()
                );
                assert_eq!(events.len() as u64, stream.external_sort.scratch_event_count);
                let Cbor::Array(counting) = &stream.counting_discard_object else {
                    panic!("counting/discard evidence is not an array");
                };
                let Cbor::Array(projected) = &stream.projected_object else {
                    panic!("materialized projection is not an array");
                };
                assert_eq!(counting.len(), 15);
                assert_eq!(counting[0], uint(1));
                assert_eq!(counting[1], bytes(COUNTING_DISCARD_STREAM_SCHEMA));
                assert_eq!(counting[9], projected[5]);
                assert_eq!(counting[10], projected[6]);
                assert_eq!(counting[11], projected[7]);
                assert_eq!(counting[12], projected[8]);
                assert_eq!(&counting[13..], &[uint(0), uint(0)]);
            }
        }
    }

    #[test]
    fn neutral_sidecar_replays_complete_preimages_and_rejects_tampering() {
        replay_golden_node3_neutral_sidecar(neutral()).unwrap();
        let summary = neutral().summary().unwrap();
        assert_eq!(summary.preimages.len(), 3);
        assert!(summary.preimages.iter().all(|row| row.canonical_byte_length > 0));
        assert!(summary.sidecar_manifest.canonical_byte_length > 0);

        let mut tampered = neutral().clone();
        let index = tampered.odd_partition_evidence.len() - 1;
        tampered.odd_partition_evidence[index] ^= 1;
        assert!(replay_golden_node3_neutral_sidecar(&tampered).is_err());
    }

    #[test]
    fn neutral_golden_manifest_replays_closed_authority_and_formal_input_roots() {
        replay_golden_node3_neutral_bundle(full_neutral()).unwrap();
        let summary = full_neutral().summary().unwrap();
        let (_, leaf_root) = full_leaf_neutral_object().unwrap();
        let (odd_root, sink_root) = regenerated_universe_root_bytes();
        let semantic_root = q1_semantic_binding_root(leaf_root, odd_root, sink_root);
        let projection_root = q1_projection_profile_root(semantic_root);
        assert_eq!(
            hex_encode(&semantic_root),
            "e3b3df3e81b7632c7c713ef5ec84913f990ad8232a25b851f20c46ac7416bfcb"
        );
        assert_eq!(
            hex_encode(&projection_root),
            "aa441cdc49ab60324483b9aa44e9fdfc324a6ad49a6bff50af6daa775209816d"
        );
        assert_eq!(
            hex_encode(&q05b_tag_registry_root()),
            "7daf75e861dacd3f3bda5ba6a0f7952e82b0109009bf306b23ba5db346c51d10"
        );
        assert_eq!(
            hex_encode(&qualification_predicate_registry_root()),
            "2ef7f84f6c046fad7f75b034d7ebb30dfcbf8924dfd94ffbb98794e8bfeba614"
        );
        assert_eq!(
            summary.qualification_wire_profile_root,
            "bd85abed6feb4b4e9fd6102f43c5db3bbaf9733f0ec42ab5b5363e14a86d350e"
        );
        assert_eq!(
            summary.bounded_node3_state_roots,
            [
                "a7460841bcd36797fa9d5d9987fafe5b5efd91f96e4e49b73a78c6406a20db37",
                "1788df25b4cd6b8830db28d8622e2fe146f3a3c454404e5e7eafe51315acab8f",
            ]
        );
        assert_eq!(
            (
                summary.node3_golden_manifest.canonical_byte_length,
                summary.node3_golden_manifest.raw_sha256.as_str(),
                summary.node3_golden_manifest.content_root.as_str(),
            ),
            (
                4_134,
                "7fd529708a068e2fa1a8d17f5cc81a41420db944120f4f1591f73e1c67f4cc05",
                "cbc22f6a9dc91589f77aa1564eb40d688c45ee3aa6af5a66d777ffe08a086b15",
            )
        );
        let expected_preimages = [
            (
                70_244,
                "9b983a8b9486690785d23e27a64735c6c5d4379d9fdb0a2944cf19179256f21f",
                "88aa7e007810074c0a68323cceef73897c69d179c85cccd6f49c1b3092ed6f0f",
            ),
            (
                1_244_549,
                "0b2b41acce572e05cd2f201f78a5911782b1559ed31c68625eef984bbf4b39de",
                "99357fc3a5f48e8a63e6a87f4b182153c5cdae52bd911676f7b2ecc1058aa097",
            ),
            (
                1_078_063,
                "2d708648b948ac984a7632c06a71d88a6d03388ee00373c6abaf47ef8bff8756",
                "51d017cd9d7e452198d9d12c53e16728c1e220e56d47f43ce3954c4e92c9ef67",
            ),
        ];
        for (artifact, expected) in summary.sidecar.preimages.iter().zip(expected_preimages) {
            assert_eq!(
                (
                    artifact.canonical_byte_length,
                    artifact.raw_sha256.as_str(),
                    artifact.content_root.as_str(),
                ),
                expected
            );
        }
        assert_eq!(
            (
                summary.sidecar.sidecar_manifest.canonical_byte_length,
                summary.sidecar.sidecar_manifest.raw_sha256.as_str(),
                summary.sidecar.sidecar_manifest.content_root.as_str(),
            ),
            (
                552,
                "318b8fb9e9ba3ce881057742d59bf43314c89891cbc37e4824349ac3f72d4ba3",
                "1d68a6fe330f3bfe581ef37933f64d2258e1043079dae15c85607836d99ea59d",
            )
        );

        let mut tampered = full_neutral().clone();
        let index = tampered.node3_golden_manifest.len() - 1;
        tampered.node3_golden_manifest[index] ^= 1;
        assert!(replay_golden_node3_neutral_bundle(&tampered).is_err());
    }

    #[test]
    fn strict_cbor_sidecar_decoder_rejects_aliases_and_trailing_data() {
        assert_eq!(decode_strict(&[0x00]).unwrap(), uint(0));
        assert_eq!(decode_strict(&[0xf4]).unwrap(), Cbor::Bool(false));
        assert_eq!(decode_strict(&[0xf6]).unwrap(), Cbor::Null);
        assert!(decode_strict(&[0x18, 0x00]).is_err());
        assert!(decode_strict(&[0x00, 0x00]).is_err());
        assert!(decode_strict(&[0x9f, 0xff]).is_err());
    }

    #[test]
    fn typed_bottom_bool_bit_integer_and_rational_are_distinct() {
        let root = [0x5a; 32];
        let variants = [
            Behavior {
                input_signature_id: 1,
                universe_root: root,
                output_sort: OutputSortId::Bool,
                cells: vec![RuntimeValue::Bottom],
            },
            Behavior {
                input_signature_id: 1,
                universe_root: root,
                output_sort: OutputSortId::Bool,
                cells: vec![RuntimeValue::Bool(false)],
            },
            Behavior {
                input_signature_id: 1,
                universe_root: root,
                output_sort: OutputSortId::Bit,
                cells: vec![RuntimeValue::Bit(0)],
            },
            Behavior {
                input_signature_id: 1,
                universe_root: root,
                output_sort: OutputSortId::BoundedInt,
                cells: vec![RuntimeValue::BoundedInt(0)],
            },
            Behavior {
                input_signature_id: 1,
                universe_root: root,
                output_sort: OutputSortId::RationalValue,
                cells: vec![RuntimeValue::Rational(Rational::integer(0))],
            },
        ];
        let bytes = variants
            .iter()
            .map(Behavior::canonical_bytes)
            .collect::<Result<BTreeSet<_>, _>>()
            .unwrap();
        assert_eq!(bytes.len(), variants.len());
    }

    #[test]
    fn and2_uses_program_order_and_preserves_the_two_witness_counterexample() {
        let (observations, rows) = odd_universe();
        let root = rfc6962(&rows);
        let context = admit(Node::ContextFlag(0), 1, root, &observations)
            .unwrap()
            .unwrap();
        let task = admit(Node::TaskFlag(0), 1, root, &observations)
            .unwrap()
            .unwrap();
        let mut state = QuotientState::default();
        state.insert(context).unwrap();
        state.insert(task).unwrap();
        assert_eq!(state.class_count(), 1);
        assert_eq!(state.cohort_count(), 1);
        assert_eq!(state.bank_count(), 2);
        assert_eq!(state.frontier_count(), 2);
        let applications = eligible_applications(&state.continuation_programs(), 1);
        let and = applications
            .into_iter()
            .find(|application| matches!(application.operator, Operator::And2))
            .unwrap();
        assert!(program_sort_key(&and.children[0]) < program_sort_key(&and.children[1]));
        let canonical = canonicalize_shrink6_source_node(and.source_node()).unwrap();
        assert_eq!(
            hex_encode(&canonical.canonical_cbor),
            "82018204828300040083000500"
        );
    }

    fn synthetic_record(index: u32, encoded_length: usize) -> ArchiveRecord {
        ArchiveRecord {
            key: index.to_be_bytes().to_vec(),
            object: array([uint(u64::from(index))]),
            encoded: vec![(index & 0xff) as u8; encoded_length],
        }
    }

    #[test]
    fn chunk_record_and_byte_boundaries_are_exact() {
        let root = [0x11; 32];
        let rows_4096 = (0..4096)
            .map(|index| synthetic_record(index, 1))
            .collect::<Vec<_>>();
        assert_eq!(
            encode_chunks(1, root, StreamKind::Coverage, &rows_4096)
                .unwrap()
                .len(),
            1
        );
        let rows_4097 = (0..4097)
            .map(|index| synthetic_record(index, 1))
            .collect::<Vec<_>>();
        assert_eq!(
            encode_chunks(1, root, StreamKind::Coverage, &rows_4097)
                .unwrap()
                .len(),
            2
        );
        let exact = vec![synthetic_record(0, MAX_CHUNK_FRAMED_BYTES - 4)];
        let exact_chunks = encode_chunks(1, root, StreamKind::Program, &exact).unwrap();
        assert_eq!(exact_chunks.len(), 1);
        assert_eq!(exact_chunks[0].framed_length, MAX_CHUNK_FRAMED_BYTES);
        let plus_one = vec![
            synthetic_record(0, MAX_CHUNK_FRAMED_BYTES - 4),
            synthetic_record(1, 1),
        ];
        assert_eq!(
            encode_chunks(1, root, StreamKind::Program, &plus_one)
                .unwrap()
                .len(),
            2
        );
        let one_record_plus_one = vec![synthetic_record(0, MAX_CHUNK_FRAMED_BYTES - 3)];
        assert_eq!(
            encode_chunks(1, root, StreamKind::Program, &one_record_plus_one)
                .unwrap_err()
                .code,
            "INCONCLUSIVE_Q1_RECORD_TOO_LARGE"
        );
    }

    #[test]
    fn materialized_and_counting_discard_streams_are_identical() {
        for partition in &golden().partitions {
            for stream in &partition.archives.streams {
                assert!(stream.materialized_counting_equal);
                assert_eq!(
                    stream.materialized_stream_sha256,
                    stream.counting_discard_stream_sha256
                );
                assert_eq!(stream.chunk_count, 1);
                assert_eq!(stream.external_sort.initial_run_count, 1);
                assert_eq!(stream.external_sort.merge_level_count, 0);
                assert_eq!(
                    stream.external_sort.scratch_events.iter().map(|event| event.action_id).collect::<Vec<_>>(),
                    vec![1, 2, 3, 4]
                );
            }
        }
    }

    #[test]
    fn counting_only_frame_mutation_is_detected_by_the_dual_comparison() {
        let records = vec![synthetic_record(7, 32), synthetic_record(8, 17)];
        let error = encode_chunks_with_counting_mutation(
            1,
            [0x42; 32],
            StreamKind::Program,
            &records,
            CountingPathMutation::FlipFirstPayloadByte,
        )
        .unwrap_err();
        assert_eq!(error.code, "FAIL_Q1_COUNTING_ENCODER");
        assert!(error.detail.contains("counting/discard blob hash"));
    }

    #[test]
    fn external_sort_merge_boundaries_and_scratch_replay_are_closed() {
        assert_eq!(external_sort_merge_shape(1), vec![1]);
        assert_eq!(external_sort_merge_shape(16), vec![16, 1]);
        assert_eq!(external_sort_merge_shape(17), vec![17, 2, 1]);
        assert_eq!(external_sort_merge_shape(256), vec![256, 16, 1]);
        assert_eq!(external_sort_merge_shape(257), vec![257, 17, 2, 1]);
        let records = (0..17)
            .map(|index| synthetic_record(index, 8))
            .collect::<Vec<_>>();
        let one_row_limit = external_row(&records[0].key, &records[0].encoded)
            .unwrap()
            .len();
        let trace = external_sort_projection_with_limit(
            1,
            StreamKind::Program,
            &records,
            one_row_limit,
        )
        .unwrap();
        assert_eq!(trace.initial_run_count, 17);
        assert_eq!(trace.merge_level_count, 2);
        assert_eq!(trace.scratch_event_count, 80);
        assert_eq!(trace.scratch_events.last().unwrap().action_id, 4);
        assert_eq!(trace.scratch_events.last().unwrap().live_logical_bytes_after, 0);
        assert_eq!(trace.scratch_events.last().unwrap().live_charged_bytes_after, 0);
    }

    #[test]
    fn collisions_duplicate_sort_keys_and_tampering_fail_closed() {
        let mut registry = BTreeMap::new();
        let digest = [7; 32];
        register_preimage(&mut registry, digest, vec![1], "test").unwrap();
        let collision = register_preimage(&mut registry, digest, vec![2], "test").unwrap_err();
        assert_eq!(collision.code, "FAIL_SHA256_PREIMAGE_COLLISION");

        let duplicate_key = vec![synthetic_record(0, 1), synthetic_record(0, 2)];
        assert_eq!(
            external_sort_projection(1, StreamKind::Program, &duplicate_key)
                .unwrap_err()
                .code,
            "REJECT_Q1_SORT_INPUT"
        );

        let records = vec![synthetic_record(1, 4), synthetic_record(2, 4)];
        let valid = external_sort_projection(1, StreamKind::Program, &records).unwrap();
        validate_external_sort_projection(1, StreamKind::Program, &records, &valid).unwrap();
        let mut tampered = valid.clone();
        tampered.scratch_events[0].new_size += 1;
        assert_eq!(
            validate_external_sort_projection(1, StreamKind::Program, &records, &tampered)
                .unwrap_err()
                .code,
            "REJECT_Q1_SORT_TRACE"
        );

        let original_root = archive_root(&records);
        let mut altered = records.clone();
        altered[0].encoded[0] ^= 1;
        assert_ne!(archive_root(&altered), original_root);
    }
}
