//! Independent Phase-3A M2.5 formal-wire primitives.
//!
//! This module implements the frozen deterministic subset of the M2.5 wire.
//! It does not generate seeds or keys, sign objects, publish formal roots, or
//! advance the M3 state machine.  The errata/addendum schemas implemented here
//! produce non-authoritative candidate vectors only.

use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};
use std::fmt;

type HmacSha256 = Hmac<Sha256>;

pub const CBOR_PROFILE_ID: &str = "hegel-cbor-det-v1";
pub const HASH_ALGORITHM: &str = "SHA-256";
pub const SPLIT_HKDF_SALT: &[u8] = b"HEGEL/SPLIT/HKDF/SALT/V1";
pub const SPLIT_ROLE_INFO_PREFIX: &[u8] = b"HEGEL/SPLIT/ROLE/V1";
pub const SPLIT_RANK_PREFIX: &[u8] = b"HEGEL/SPLIT/RANK/V1";
pub const SPLIT_SEED_COMMITMENT_DOMAIN: &[u8] = b"HEGEL/SPLIT_MASTER_SEED_COMMITMENT/V1";
pub const ID_DIGEST_PREFIX: &[u8] = b"HEGEL/ID_DIGEST/V1\0";
pub const CANONICAL_INPUT_DOMAIN: &str = "HEGEL/CANONICAL_INPUT/V1";
pub const MACHINE_FREEZE_ID: &str = "hegel-freeze-p2b-p3-v1.1.2";
pub const ERRATA_VECTOR_SCHEMA: &str = "hegel-phase3-m25-exact-wire-errata-vectors/1";
pub const LEGACY_OUTSIDE_TARGET_SOURCE_ID: &str =
    "target_spec_b491c0a9719fb0279fe02798ede026e440c17a539965514145a7818b15387ac3";
pub const LEGACY_NULL_CONTROL_SOURCE_ID: &str =
    "sink_control_spec_7fd6f9a6e2b4c6eda0c7e1545ad42cb19666743ede8ed87f40d82c0ef46198a0";

pub const BRIDGE_ATTESTATION_SIGNATURE_DOMAIN: &[u8] = b"HEGEL/BRIDGE_ATTESTATION_SIGNATURE/V1";
pub const CUSTODIAN_SPLIT_SEED_COMMITMENT_SIGNATURE_DOMAIN: &str =
    "HEGEL/CUSTODIAN_SPLIT_SEED_COMMITMENT_SIGNATURE/V1";
pub const CUSTODIAN_BINDING_SIGNATURE_DOMAIN: &str = "HEGEL/CUSTODIAN_BINDING_SIGNATURE/V1";
pub const CUSTODIAN_SEED_CONTINUITY_SIGNATURE_DOMAIN: &str =
    "HEGEL/CUSTODIAN_SEED_CONTINUITY_SIGNATURE/V1";
pub const CUSTODIAN_LEDGER_GENESIS_SIGNATURE_DOMAIN: &str =
    "HEGEL/CUSTODIAN_LEDGER_GENESIS_SIGNATURE/V1";
pub const PARENT_ABSENCE_AUDITOR_SIGNATURE_DOMAIN: &str =
    "HEGEL/PARENT_ABSENCE_AUDITOR_SIGNATURE/V2";
pub const SIGNED_MANIFEST_ENVELOPE_DOMAIN: &str = "HEGEL/SIGNED_MANIFEST_ENVELOPE/V1";
pub const M3_DUAL_REPLAY_AGREEMENT_DOMAIN: &str = "HEGEL/M3_DUAL_REPLAY_AGREEMENT/V1";

pub const NORMATIVE_DOCUMENT_BUNDLE_TAG: u64 = 0x3018;
pub const CANONICAL_AST_PROFILE_TAG: u64 = 0x3019;
pub const CANONICAL_CBOR_PROFILE_TAG: u64 = 0x301a;
pub const PHASE2B_CONTRACT_TAG: u64 = 0x301b;
pub const MDL_CODE_TABLE_TAG: u64 = 0x301c;
pub const STATIC_ROLE_METADATA_TAG: u64 = 0x301d;
pub const HIDDEN_ARTIFACT_SCOPE_TAG: u64 = 0x301e;
pub const BRIDGE_REPLAY_STATEMENT_TAG: u64 = 0x310e;
pub const M3_EXECUTION_CANDIDATE_TAG: u64 = 0x310f;
pub const M3_EXECUTION_MANIFEST_V2_TAG: u64 = 0x3110;
pub const ACTOR_TRUST_GENESIS_TAG: u64 = 0x3111;
pub const OPAQUE_ID_REGISTRY_SNAPSHOT_TAG: u64 = 0x3112;
pub const PARENT_ABSENCE_AUDIT_BUNDLE_TAG: u64 = 0x3113;
pub const PARENT_ABSENCE_ATTESTATION_V2_TAG: u64 = 0x3114;
pub const OPAQUE_ID_REGISTRATION_INTENT_TAG: u64 = 0x3115;
pub const AUDITED_PATH_BLOB_RECORD_TAG: u64 = 0x3210;
pub const AUDITED_HISTORY_ROW_TAG: u64 = 0x3211;
pub const LEGACY_PARENT_SOURCE_ROW_TAG: u64 = 0x3212;
pub const REPOSITORY_PATH_ALIAS_RECORD_TAG: u64 = 0x3213;
pub const SOURCE_FILE_RECORD_TAG: u64 = 0x3215;
pub const DEPENDENCY_LOCK_RECORD_TAG: u64 = 0x3216;
pub const LEGAL_TRANSITION_ROW_TAG: u64 = 0x3217;
pub const OPAQUE_ID_REGISTRY_RECORD_TAG: u64 = 0x3218;
pub const M3_RUN_GENESIS_TAG: u64 = 0x3300;
pub const M3_RUN_STATE_RECORD_TAG: u64 = 0x3301;
pub const M3_DUAL_REPLAY_AGREEMENT_TAG: u64 = 0x3304;
pub const SIGNED_MANIFEST_ENVELOPE_TAG: u64 = 0x31ff;

pub const NORMATIVE_DOCUMENT_BUNDLE_SCHEMA_ID: &[u8] = b"hegel-normative-document-bundle/1";
pub const CANONICAL_AST_PROFILE_SCHEMA_ID: &[u8] = b"hegel-canonical-ast-profile/1";
pub const CANONICAL_CBOR_PROFILE_SCHEMA_ID: &[u8] = b"hegel-canonical-cbor-profile/1";
pub const PHASE2B_CONTRACT_SCHEMA_ID: &[u8] = b"hegel-phase2b-contract/1";
pub const MDL_CODE_TABLE_SCHEMA_ID: &[u8] = b"hegel-mdl-code-table/1";
pub const STATIC_ROLE_METADATA_SCHEMA_ID: &[u8] = b"hegel-static-role-metadata/1";
pub const HIDDEN_ARTIFACT_SCOPE_SCHEMA_ID: &[u8] = b"hegel-hidden-artifact-scope/1";
pub const BRIDGE_REPLAY_STATEMENT_SCHEMA_ID: &[u8] = b"hegel-bridge-replay-statement/1";
pub const M3_EXECUTION_CANDIDATE_SCHEMA_ID: &[u8] = b"hegel-m3-execution-candidate/1";
pub const M3_EXECUTION_MANIFEST_V2_SCHEMA_ID: &[u8] = b"hegel-m3-execution-manifest/2";
pub const ACTOR_TRUST_GENESIS_SCHEMA_ID: &[u8] = b"hegel-actor-trust-genesis/1";
pub const OPAQUE_ID_REGISTRY_SNAPSHOT_SCHEMA_ID: &[u8] = b"hegel-opaque-id-registry-snapshot/1";
pub const PARENT_ABSENCE_AUDIT_BUNDLE_SCHEMA_ID: &[u8] = b"hegel-parent-absence-audit-bundle/1";
pub const PARENT_ABSENCE_ATTESTATION_V2_SCHEMA_ID: &[u8] =
    b"hegel-parent-manifest-absence-attestation/2";
pub const OPAQUE_ID_REGISTRATION_INTENT_SCHEMA_ID: &[u8] = b"hegel-opaque-id-registration-intent/1";
pub const AUDITED_PATH_BLOB_RECORD_SCHEMA_ID: &[u8] = b"hegel-audited-path-blob-record/1";
pub const AUDITED_HISTORY_ROW_SCHEMA_ID: &[u8] = b"hegel-audited-history-row/1";
pub const LEGACY_PARENT_SOURCE_ROW_SCHEMA_ID: &[u8] = b"hegel-legacy-parent-source-row/1";
pub const REPOSITORY_PATH_ALIAS_RECORD_SCHEMA_ID: &[u8] = b"hegel-repository-path-alias-record/1";
pub const SOURCE_FILE_RECORD_SCHEMA_ID: &[u8] = b"hegel-source-file-record/1";
pub const DEPENDENCY_LOCK_RECORD_SCHEMA_ID: &[u8] = b"hegel-dependency-lock-record/1";
pub const LEGAL_TRANSITION_ROW_SCHEMA_ID: &[u8] = b"hegel-legal-transition-row/1";
pub const OPAQUE_ID_REGISTRY_RECORD_SCHEMA_ID: &[u8] = b"hegel-opaque-id-registry-record/1";
pub const M3_RUN_GENESIS_SCHEMA_ID: &[u8] = b"hegel-m3-run-genesis/1";
pub const M3_RUN_STATE_RECORD_SCHEMA_ID: &[u8] = b"hegel-m3-run-state-record/1";
pub const M3_DUAL_REPLAY_AGREEMENT_SCHEMA_ID: &[u8] = b"hegel-m3-dual-replay-agreement/1";
pub const SIGNED_MANIFEST_ENVELOPE_SCHEMA_ID: &[u8] = b"hegel-signed-manifest-envelope/1";

pub const ODD_INPUT_TAG: u64 = 0x3401;
pub const SINK_INPUT_TAG: u64 = 0x3402;
pub const UNIVERSE_ROW_TAG: u64 = 0x3201;
pub const TRUTH_ROW_TAG: u64 = 0x3202;
pub const ODD_INPUT_SIGNATURE_ID: u16 = 1;
pub const SINK_INPUT_SIGNATURE_ID: u16 = 2;

pub const ODD_INPUT_SCHEMA_ID: &[u8] = b"hegel-odd-input/1";
pub const SINK_INPUT_SCHEMA_ID: &[u8] = b"hegel-sink-input/1";
pub const UNIVERSE_ROW_SCHEMA_ID: &[u8] = b"hegel-bounded-universe-row/1";
pub const TRUTH_ROW_SCHEMA_ID: &[u8] = b"hegel-target-truth-row/1";

pub const REJECT_NONCANONICAL_CBOR: &str = "REJECT_NONCANONICAL_CBOR";
pub const REJECT_TRUNCATED_CBOR: &str = "REJECT_TRUNCATED_CBOR";
pub const REJECT_RESERVED_CBOR: &str = "REJECT_RESERVED_CBOR";
pub const REJECT_CBOR_TEXT: &str = "REJECT_CBOR_TEXT";
pub const REJECT_CBOR_MAP: &str = "REJECT_CBOR_MAP";
pub const REJECT_CBOR_TAG: &str = "REJECT_CBOR_TAG";
pub const REJECT_CBOR_FLOAT: &str = "REJECT_CBOR_FLOAT";
pub const REJECT_INDEFINITE_CBOR: &str = "REJECT_INDEFINITE_CBOR";
pub const REJECT_TRAILING_CBOR: &str = "REJECT_TRAILING_CBOR";
pub const REJECT_CBOR_NESTING: &str = "REJECT_CBOR_NESTING";
pub const REJECT_CBOR_UNDEFINED: &str = "REJECT_CBOR_UNDEFINED";
pub const REJECT_CBOR_SIMPLE: &str = "REJECT_CBOR_SIMPLE";
pub const REJECT_INVALID_LENGTH: &str = "REJECT_INVALID_LENGTH";
pub const REJECT_HKDF_LENGTH: &str = "REJECT_HKDF_LENGTH";
pub const REJECT_HASH_DOMAIN: &str = "REJECT_HASH_DOMAIN";
pub const REJECT_MACHINE_ID_NON_ASCII: &str = "REJECT_MACHINE_ID_NON_ASCII";
pub const REJECT_MACHINE_ID_SYNTAX: &str = "REJECT_MACHINE_ID_SYNTAX";
pub const REJECT_MACHINE_ID_LENGTH: &str = "REJECT_MACHINE_ID_LENGTH";
pub const REJECT_TYPED_INPUT_PREFIX: &str = "REJECT_TYPED_INPUT_PREFIX";
pub const REJECT_ODD_SET_SIZE: &str = "REJECT_ODD_SET_SIZE";
pub const REJECT_ODD_BIT_COUNT: &str = "REJECT_ODD_BIT_COUNT";
pub const REJECT_ODD_BIT_TYPE: &str = "REJECT_ODD_BIT_TYPE";
pub const REJECT_SINK_VALUE: &str = "REJECT_SINK_VALUE";
pub const REJECT_SINK_BALANCE: &str = "REJECT_SINK_BALANCE";
pub const REJECT_UNIVERSE_ROW_SCHEMA: &str = "REJECT_UNIVERSE_ROW_SCHEMA";
pub const REJECT_TRUTH_ROW_SCHEMA: &str = "REJECT_TRUTH_ROW_SCHEMA";
pub const FAIL_UNIVERSE_INDEX_DUPLICATE: &str = "FAIL_UNIVERSE_INDEX_DUPLICATE";
pub const FAIL_UNIVERSE_INDEX_GAP: &str = "FAIL_UNIVERSE_INDEX_GAP";
pub const FAIL_CANONICAL_INPUT_HASH_MISMATCH: &str = "FAIL_CANONICAL_INPUT_HASH_MISMATCH";
pub const FAIL_TARGET_OUTPUT_TYPE: &str = "FAIL_TARGET_OUTPUT_TYPE";
pub const FAIL_INPUT_SIGNATURE_MISMATCH: &str = "FAIL_INPUT_SIGNATURE_MISMATCH";
pub const FAIL_ROW_ORDERING: &str = "FAIL_ROW_ORDERING";
pub const FAIL_M25_NORMATIVE_GAP: &str = "FAIL_M25_NORMATIVE_GAP";
pub const REJECT_M25_OBJECT_PREFIX: &str = "REJECT_M25_OBJECT_PREFIX";
pub const REJECT_M25_FIELD_SET: &str = "REJECT_M25_FIELD_SET";
pub const REJECT_M25_FIELD_TYPE: &str = "REJECT_M25_FIELD_TYPE";
pub const REJECT_M25_FIELD_VALUE: &str = "REJECT_M25_FIELD_VALUE";
pub const REJECT_M25_SIGNATURE_COUNT: &str = "REJECT_M25_SIGNATURE_COUNT";
pub const REJECT_UNKNOWN_ENUM_VALUE: &str = "REJECT_UNKNOWN_ENUM_VALUE";
pub const FAIL_ACTOR_TRUST_PURPOSE_SET: &str = "FAIL_ACTOR_TRUST_PURPOSE_SET";
pub const FAIL_ACTOR_KEY_CROSS_PURPOSE_REUSE: &str = "FAIL_ACTOR_KEY_CROSS_PURPOSE_REUSE";
pub const FAIL_ACTOR_TRUST_KEY_ROOT_MISMATCH: &str = "FAIL_ACTOR_TRUST_KEY_ROOT_MISMATCH";
pub const FAIL_BRIDGE_ATTESTATION_PURPOSE_SET: &str = "FAIL_BRIDGE_ATTESTATION_PURPOSE_SET";
pub const FAIL_M3_OUTPUT_ROOT_PREPOPULATED: &str = "FAIL_M3_OUTPUT_ROOT_PREPOPULATED";
pub const FAIL_M3_LEDGER_HEAD_NOT_GENESIS: &str = "FAIL_M3_LEDGER_HEAD_NOT_GENESIS";
pub const FAIL_ILLEGAL_M3_STATE_TRANSITION: &str = "FAIL_ILLEGAL_M3_STATE_TRANSITION";
pub const FAIL_OPAQUE_ID_REGISTRY_SEQUENCE: &str = "FAIL_OPAQUE_ID_REGISTRY_SEQUENCE";
pub const FAIL_OPAQUE_ID_ALREADY_USED: &str = "FAIL_OPAQUE_ID_ALREADY_USED";
pub const FAIL_OPAQUE_ID_REGISTRY_SNAPSHOT: &str = "FAIL_OPAQUE_ID_REGISTRY_SNAPSHOT";
pub const REJECT_M25_RECORD_ORDER: &str = "REJECT_M25_RECORD_ORDER";
pub const FAIL_NULL_WITNESS_BINDING_MISMATCH: &str = "FAIL_NULL_WITNESS_BINDING_MISMATCH";
pub const FAIL_EXTERNAL_INPUT_ATTESTATION_COVERAGE: &str =
    "FAIL_EXTERNAL_INPUT_ATTESTATION_COVERAGE";

const MAX_CBOR_NESTING: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FormalWireError {
    pub code: &'static str,
    pub message: String,
}

impl FormalWireError {
    fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

impl fmt::Display for FormalWireError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for FormalWireError {}

/// Values admitted by the M2.5 deterministic-CBOR formal core.
///
/// `Negative(argument)` represents the mathematical integer `-1 - argument`,
/// matching CBOR major type 1 without losing the full `u64` argument range.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CborValue {
    Unsigned(u64),
    Negative(u64),
    Bytes(Vec<u8>),
    Array(Vec<CborValue>),
    Bool(bool),
    Null,
}

fn encode_major_value(major: u8, value: u64, output: &mut Vec<u8>) {
    debug_assert!(major <= 7);
    if value <= 23 {
        output.push((major << 5) | value as u8);
    } else if value <= u8::MAX as u64 {
        output.push((major << 5) | 24);
        output.push(value as u8);
    } else if value <= u16::MAX as u64 {
        output.push((major << 5) | 25);
        output.extend_from_slice(&(value as u16).to_be_bytes());
    } else if value <= u32::MAX as u64 {
        output.push((major << 5) | 26);
        output.extend_from_slice(&(value as u32).to_be_bytes());
    } else {
        output.push((major << 5) | 27);
        output.extend_from_slice(&value.to_be_bytes());
    }
}

fn encode_value(value: &CborValue, output: &mut Vec<u8>) -> Result<(), FormalWireError> {
    match value {
        CborValue::Unsigned(value) => encode_major_value(0, *value, output),
        CborValue::Negative(argument) => encode_major_value(1, *argument, output),
        CborValue::Bytes(bytes) => {
            let length = u64::try_from(bytes.len()).map_err(|_| {
                FormalWireError::new(REJECT_INVALID_LENGTH, "byte string length exceeds u64")
            })?;
            encode_major_value(2, length, output);
            output.extend_from_slice(bytes);
        }
        CborValue::Array(values) => {
            let length = u64::try_from(values.len()).map_err(|_| {
                FormalWireError::new(REJECT_INVALID_LENGTH, "array length exceeds u64")
            })?;
            encode_major_value(4, length, output);
            for child in values {
                encode_value(child, output)?;
            }
        }
        CborValue::Bool(false) => output.push(0xf4),
        CborValue::Bool(true) => output.push(0xf5),
        CborValue::Null => output.push(0xf6),
    }
    Ok(())
}

/// Encode one formal-core value using shortest deterministic CBOR.
pub fn encode_canonical_cbor(value: &CborValue) -> Result<Vec<u8>, FormalWireError> {
    let mut output = Vec::new();
    encode_value(value, &mut output)?;
    Ok(output)
}

fn read_exact<'a>(
    bytes: &'a [u8],
    cursor: &mut usize,
    length: usize,
) -> Result<&'a [u8], FormalWireError> {
    let end = cursor
        .checked_add(length)
        .ok_or_else(|| FormalWireError::new(REJECT_NONCANONICAL_CBOR, "CBOR length overflow"))?;
    if end > bytes.len() {
        return Err(FormalWireError::new(
            REJECT_TRUNCATED_CBOR,
            "truncated CBOR item",
        ));
    }
    let result = &bytes[*cursor..end];
    *cursor = end;
    Ok(result)
}

fn read_argument(additional: u8, bytes: &[u8], cursor: &mut usize) -> Result<u64, FormalWireError> {
    let (value, minimum) = match additional {
        0..=23 => return Ok(additional as u64),
        24 => (read_exact(bytes, cursor, 1)?[0] as u64, 24),
        25 => {
            let raw: [u8; 2] = read_exact(bytes, cursor, 2)?.try_into().expect("two bytes");
            (u16::from_be_bytes(raw) as u64, 0x100)
        }
        26 => {
            let raw: [u8; 4] = read_exact(bytes, cursor, 4)?
                .try_into()
                .expect("four bytes");
            (u32::from_be_bytes(raw) as u64, 0x1_0000)
        }
        27 => {
            let raw: [u8; 8] = read_exact(bytes, cursor, 8)?
                .try_into()
                .expect("eight bytes");
            (u64::from_be_bytes(raw), 0x1_0000_0000)
        }
        31 => {
            return Err(FormalWireError::new(
                REJECT_INDEFINITE_CBOR,
                "indefinite-length CBOR is forbidden",
            ))
        }
        _ => {
            return Err(FormalWireError::new(
                REJECT_RESERVED_CBOR,
                "reserved CBOR additional-information value",
            ))
        }
    };
    if value < minimum {
        return Err(FormalWireError::new(
            REJECT_NONCANONICAL_CBOR,
            "integer or length does not use its shortest encoding",
        ));
    }
    Ok(value)
}

fn parse_value(
    bytes: &[u8],
    cursor: &mut usize,
    recursion_depth: usize,
) -> Result<CborValue, FormalWireError> {
    if recursion_depth > MAX_CBOR_NESTING {
        return Err(FormalWireError::new(
            REJECT_CBOR_NESTING,
            "CBOR nesting exceeds the strict decoder limit",
        ));
    }
    let initial = read_exact(bytes, cursor, 1)?[0];
    let major = initial >> 5;
    let additional = initial & 0x1f;
    match major {
        0 => Ok(CborValue::Unsigned(read_argument(
            additional, bytes, cursor,
        )?)),
        1 => Ok(CborValue::Negative(read_argument(
            additional, bytes, cursor,
        )?)),
        2 => {
            let length = read_argument(additional, bytes, cursor)?;
            let length = usize::try_from(length).map_err(|_| {
                FormalWireError::new(
                    REJECT_NONCANONICAL_CBOR,
                    "byte-string length does not fit this implementation",
                )
            })?;
            Ok(CborValue::Bytes(
                read_exact(bytes, cursor, length)?.to_vec(),
            ))
        }
        3 => Err(FormalWireError::new(
            REJECT_CBOR_TEXT,
            "CBOR text strings are forbidden",
        )),
        4 => {
            let length = read_argument(additional, bytes, cursor)?;
            let length = usize::try_from(length).map_err(|_| {
                FormalWireError::new(
                    REJECT_NONCANONICAL_CBOR,
                    "array length does not fit this implementation",
                )
            })?;
            // Do not reject solely from the remaining byte count: a present
            // forbidden child must retain precedence over a later missing
            // child, matching the independent Python decoder. Cap the initial
            // allocation at the bytes that could possibly contain children so
            // an enormous declared length cannot force an enormous allocation.
            let mut values = Vec::with_capacity(length.min(bytes.len().saturating_sub(*cursor)));
            for _ in 0..length {
                values.push(parse_value(bytes, cursor, recursion_depth + 1)?);
            }
            Ok(CborValue::Array(values))
        }
        5 => Err(FormalWireError::new(
            REJECT_CBOR_MAP,
            "CBOR maps are forbidden",
        )),
        6 => Err(FormalWireError::new(
            REJECT_CBOR_TAG,
            "CBOR tags are forbidden",
        )),
        7 => match additional {
            20 => Ok(CborValue::Bool(false)),
            21 => Ok(CborValue::Bool(true)),
            22 => Ok(CborValue::Null),
            23 => Err(FormalWireError::new(
                REJECT_CBOR_UNDEFINED,
                "CBOR undefined is forbidden",
            )),
            25..=27 => Err(FormalWireError::new(
                REJECT_CBOR_FLOAT,
                "CBOR floating-point values are forbidden",
            )),
            31 => Err(FormalWireError::new(
                REJECT_INDEFINITE_CBOR,
                "CBOR break/indefinite encoding is forbidden",
            )),
            _ => Err(FormalWireError::new(
                REJECT_CBOR_SIMPLE,
                "only false, true, and null simple values are admitted",
            )),
        },
        _ => unreachable!("CBOR major type is three bits"),
    }
}

/// Decode one CBOR item and require exact deterministic re-encoding.
pub fn decode_strict_cbor(bytes: &[u8]) -> Result<CborValue, FormalWireError> {
    let mut cursor = 0;
    let value = parse_value(bytes, &mut cursor, 0)?;
    if cursor != bytes.len() {
        return Err(FormalWireError::new(
            REJECT_TRAILING_CBOR,
            "trailing bytes after the CBOR item",
        ));
    }
    let reencoded = encode_canonical_cbor(&value)?;
    if reencoded != bytes {
        return Err(FormalWireError::new(
            REJECT_NONCANONICAL_CBOR,
            "CBOR item differs from its exact deterministic re-encoding",
        ));
    }
    Ok(value)
}

/// Validate an existing CBOR item and return its identical canonical bytes.
pub fn validate_strict_cbor(bytes: &[u8]) -> Result<Vec<u8>, FormalWireError> {
    let value = decode_strict_cbor(bytes)?;
    encode_canonical_cbor(&value)
}

fn sha256(parts: &[&[u8]]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update(part);
    }
    hasher.finalize().into()
}

/// `SHA256(UTF8(domain) || 0x00 || CanonicalCBOR(object))`.
pub fn content_hash(domain: &str, object: &CborValue) -> Result<[u8; 32], FormalWireError> {
    validate_hash_domain(domain)?;
    let encoded = encode_canonical_cbor(object)?;
    Ok(sha256(&[domain.as_bytes(), &[0], &encoded]))
}

/// ContentHash for a pre-encoded object, after strict validation.
pub fn content_hash_cbor(domain: &str, canonical_cbor: &[u8]) -> Result<[u8; 32], FormalWireError> {
    validate_hash_domain(domain)?;
    validate_strict_cbor(canonical_cbor)?;
    Ok(sha256(&[domain.as_bytes(), &[0], canonical_cbor]))
}

fn validate_hash_domain(domain: &str) -> Result<(), FormalWireError> {
    if domain.is_empty() || domain.as_bytes().contains(&0) {
        return Err(FormalWireError::new(
            REJECT_HASH_DOMAIN,
            "ContentHash domain must be nonempty UTF-8 without NUL",
        ));
    }
    Ok(())
}

/// RFC 6962 leaf hash: `SHA256(0x00 || leaf_bytes)`.
pub fn rfc6962_leaf_hash(leaf: &[u8]) -> [u8; 32] {
    sha256(&[&[0], leaf])
}

/// RFC 6962 internal-node hash: `SHA256(0x01 || left || right)`.
pub fn rfc6962_node_hash(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    sha256(&[&[1], left, right])
}

fn largest_power_of_two_less_than(value: usize) -> usize {
    debug_assert!(value > 1);
    let exponent = (usize::BITS - 1 - (value - 1).leading_zeros()) as usize;
    1usize << exponent
}

/// RFC 6962 Merkle Tree Hash. The empty-tree root is `SHA256("")`.
pub fn rfc6962_root(leaves: &[Vec<u8>]) -> [u8; 32] {
    match leaves.len() {
        0 => sha256(&[b""]),
        1 => rfc6962_leaf_hash(&leaves[0]),
        count => {
            let split = largest_power_of_two_less_than(count);
            let left = rfc6962_root(&leaves[..split]);
            let right = rfc6962_root(&leaves[split..]);
            rfc6962_node_hash(&left, &right)
        }
    }
}

/// RFC 6962 root over records that are each strict canonical CBOR bytes.
pub fn rfc6962_canonical_record_root(
    canonical_records: &[Vec<u8>],
) -> Result<[u8; 32], FormalWireError> {
    for record in canonical_records {
        validate_strict_cbor(record)?;
    }
    Ok(rfc6962_root(canonical_records))
}

/// Exact bytes hashed by the v1.1.2 ``IdDigestV1`` profile.
pub fn id_digest_preimage_v1(machine_id: &str) -> Result<Vec<u8>, FormalWireError> {
    if !machine_id.is_ascii() {
        return Err(FormalWireError::new(
            REJECT_MACHINE_ID_NON_ASCII,
            "machine ID must contain ASCII only",
        ));
    }
    let bytes = machine_id.as_bytes();
    if bytes.len() > 256 {
        return Err(FormalWireError::new(
            REJECT_MACHINE_ID_LENGTH,
            "machine ID exceeds 256 ASCII bytes",
        ));
    }
    let allowed_tail =
        |byte: u8| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b':' | b'/' | b'-');
    if bytes.is_empty()
        || !bytes[0].is_ascii_alphanumeric()
        || !bytes[1..].iter().copied().all(allowed_tail)
    {
        return Err(FormalWireError::new(
            REJECT_MACHINE_ID_SYNTAX,
            "machine ID violates the frozen syntax",
        ));
    }
    let mut preimage = Vec::with_capacity(ID_DIGEST_PREFIX.len() + bytes.len());
    preimage.extend_from_slice(ID_DIGEST_PREFIX);
    preimage.extend_from_slice(bytes);
    Ok(preimage)
}

/// ``SHA-256(IdDigestV1 preimage)``.
pub fn id_digest_v1(machine_id: &str) -> Result<[u8; 32], FormalWireError> {
    let preimage = id_digest_preimage_v1(machine_id)?;
    Ok(sha256(&[&preimage]))
}

fn unsigned(value: u64) -> CborValue {
    CborValue::Unsigned(value)
}

fn bytes_value(value: &[u8]) -> CborValue {
    CborValue::Bytes(value.to_vec())
}

fn array(value: &CborValue) -> Result<&[CborValue], FormalWireError> {
    match value {
        CborValue::Array(values) => Ok(values),
        _ => Err(FormalWireError::new(
            REJECT_TYPED_INPUT_PREFIX,
            "typed input must be a CBOR array",
        )),
    }
}

fn exact_prefix(values: &[CborValue], tag: u64, schema_id: &[u8]) -> bool {
    values.len() >= 3
        && values[0] == unsigned(1)
        && values[1] == unsigned(tag)
        && values[2] == bytes_value(schema_id)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrataRootRule {
    ContentHash(&'static str),
    Rfc6962Records,
    NormativeGap(&'static str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ErrataSchema {
    pub name: &'static str,
    pub tag: u64,
    pub schema_id: &'static [u8],
    pub body_field_count: usize,
    pub root_rule: ErrataRootRule,
}

pub const ERRATA_SCHEMAS: &[ErrataSchema] = &[
    ErrataSchema {
        name: "NormativeDocumentBundleV1",
        tag: NORMATIVE_DOCUMENT_BUNDLE_TAG,
        schema_id: NORMATIVE_DOCUMENT_BUNDLE_SCHEMA_ID,
        body_field_count: 3,
        root_rule: ErrataRootRule::ContentHash("HEGEL/NORMATIVE_DOCUMENT_BUNDLE/V1"),
    },
    ErrataSchema {
        name: "CanonicalAstProfileSpecV1",
        tag: CANONICAL_AST_PROFILE_TAG,
        schema_id: CANONICAL_AST_PROFILE_SCHEMA_ID,
        body_field_count: 6,
        root_rule: ErrataRootRule::ContentHash("HEGEL/CANONICAL_AST_PROFILE/V1"),
    },
    ErrataSchema {
        name: "CanonicalCborProfileSpecV1",
        tag: CANONICAL_CBOR_PROFILE_TAG,
        schema_id: CANONICAL_CBOR_PROFILE_SCHEMA_ID,
        body_field_count: 6,
        root_rule: ErrataRootRule::ContentHash("HEGEL/CANONICAL_CBOR_PROFILE/V1"),
    },
    ErrataSchema {
        name: "Phase2BContractSpecV1",
        tag: PHASE2B_CONTRACT_TAG,
        schema_id: PHASE2B_CONTRACT_SCHEMA_ID,
        body_field_count: 6,
        root_rule: ErrataRootRule::ContentHash("HEGEL/PHASE2B_CONTRACT/V1"),
    },
    ErrataSchema {
        name: "MdlCodeTableSpecV1",
        tag: MDL_CODE_TABLE_TAG,
        schema_id: MDL_CODE_TABLE_SCHEMA_ID,
        body_field_count: 6,
        root_rule: ErrataRootRule::ContentHash("HEGEL/MDL_CODE_TABLE/V1"),
    },
    ErrataSchema {
        name: "StaticRoleMetadataV1",
        tag: STATIC_ROLE_METADATA_TAG,
        schema_id: STATIC_ROLE_METADATA_SCHEMA_ID,
        body_field_count: 6,
        root_rule: ErrataRootRule::ContentHash("HEGEL/STATIC_ROLE_METADATA/V1"),
    },
    ErrataSchema {
        name: "HiddenArtifactScopeV1",
        tag: HIDDEN_ARTIFACT_SCOPE_TAG,
        schema_id: HIDDEN_ARTIFACT_SCOPE_SCHEMA_ID,
        body_field_count: 6,
        root_rule: ErrataRootRule::ContentHash("HEGEL/HIDDEN_ARTIFACT_SCOPE/V1"),
    },
    ErrataSchema {
        name: "BridgeReplayStatementV1",
        tag: BRIDGE_REPLAY_STATEMENT_TAG,
        schema_id: BRIDGE_REPLAY_STATEMENT_SCHEMA_ID,
        body_field_count: 7,
        root_rule: ErrataRootRule::ContentHash("HEGEL/BRIDGE_REPLAY_STATEMENT/V1"),
    },
    ErrataSchema {
        name: "M3ExecutionCandidateV1",
        tag: M3_EXECUTION_CANDIDATE_TAG,
        schema_id: M3_EXECUTION_CANDIDATE_SCHEMA_ID,
        body_field_count: 44,
        root_rule: ErrataRootRule::ContentHash("HEGEL/M3_EXECUTION_CANDIDATE/V1"),
    },
    ErrataSchema {
        name: "M3ExecutionManifestV2",
        tag: M3_EXECUTION_MANIFEST_V2_TAG,
        schema_id: M3_EXECUTION_MANIFEST_V2_SCHEMA_ID,
        body_field_count: 8,
        root_rule: ErrataRootRule::ContentHash("HEGEL/M3_EXECUTION_MANIFEST/V2"),
    },
    ErrataSchema {
        name: "ActorTrustGenesisV1",
        tag: ACTOR_TRUST_GENESIS_TAG,
        schema_id: ACTOR_TRUST_GENESIS_SCHEMA_ID,
        body_field_count: 5,
        root_rule: ErrataRootRule::ContentHash("HEGEL/ACTOR_TRUST_GENESIS/V1"),
    },
    ErrataSchema {
        name: "OpaqueIdRegistrySnapshotV1",
        tag: OPAQUE_ID_REGISTRY_SNAPSHOT_TAG,
        schema_id: OPAQUE_ID_REGISTRY_SNAPSHOT_SCHEMA_ID,
        body_field_count: 5,
        root_rule: ErrataRootRule::ContentHash("HEGEL/OPAQUE_ID_REGISTRY_SNAPSHOT/V1"),
    },
    ErrataSchema {
        name: "ParentAbsenceAuditBundleV1",
        tag: PARENT_ABSENCE_AUDIT_BUNDLE_TAG,
        schema_id: PARENT_ABSENCE_AUDIT_BUNDLE_SCHEMA_ID,
        body_field_count: 7,
        root_rule: ErrataRootRule::ContentHash("HEGEL/PARENT_ABSENCE_AUDIT_BUNDLE/V1"),
    },
    ErrataSchema {
        name: "ParentManifestAbsenceAttestationV2",
        tag: PARENT_ABSENCE_ATTESTATION_V2_TAG,
        schema_id: PARENT_ABSENCE_ATTESTATION_V2_SCHEMA_ID,
        body_field_count: 7,
        root_rule: ErrataRootRule::ContentHash("HEGEL/PARENT_MANIFEST_ABSENCE_ATTESTATION/V2"),
    },
    ErrataSchema {
        name: "OpaqueIdRegistrationIntentV1",
        tag: OPAQUE_ID_REGISTRATION_INTENT_TAG,
        schema_id: OPAQUE_ID_REGISTRATION_INTENT_SCHEMA_ID,
        body_field_count: 5,
        root_rule: ErrataRootRule::ContentHash("HEGEL/OPAQUE_ID_REGISTRATION_INTENT/V1"),
    },
    ErrataSchema {
        name: "SignedManifestEnvelopeV1",
        tag: SIGNED_MANIFEST_ENVELOPE_TAG,
        schema_id: SIGNED_MANIFEST_ENVELOPE_SCHEMA_ID,
        body_field_count: 5,
        root_rule: ErrataRootRule::ContentHash(SIGNED_MANIFEST_ENVELOPE_DOMAIN),
    },
    ErrataSchema {
        name: "M3RunGenesisV1",
        tag: M3_RUN_GENESIS_TAG,
        schema_id: M3_RUN_GENESIS_SCHEMA_ID,
        body_field_count: 20,
        root_rule: ErrataRootRule::ContentHash("HEGEL/M3_RUN_GENESIS/V1"),
    },
    ErrataSchema {
        name: "M3RunStateRecordV1",
        tag: M3_RUN_STATE_RECORD_TAG,
        schema_id: M3_RUN_STATE_RECORD_SCHEMA_ID,
        body_field_count: 11,
        root_rule: ErrataRootRule::ContentHash("HEGEL/M3_RUN_STATE_RECORD/V1"),
    },
    ErrataSchema {
        name: "M3DualReplayAgreementV1",
        tag: M3_DUAL_REPLAY_AGREEMENT_TAG,
        schema_id: M3_DUAL_REPLAY_AGREEMENT_SCHEMA_ID,
        body_field_count: 16,
        root_rule: ErrataRootRule::ContentHash(M3_DUAL_REPLAY_AGREEMENT_DOMAIN),
    },
    ErrataSchema {
        name: "AuditedPathBlobRecordV1",
        tag: AUDITED_PATH_BLOB_RECORD_TAG,
        schema_id: AUDITED_PATH_BLOB_RECORD_SCHEMA_ID,
        body_field_count: 6,
        root_rule: ErrataRootRule::Rfc6962Records,
    },
    ErrataSchema {
        name: "AuditedHistoryRowV1",
        tag: AUDITED_HISTORY_ROW_TAG,
        schema_id: AUDITED_HISTORY_ROW_SCHEMA_ID,
        body_field_count: 4,
        root_rule: ErrataRootRule::Rfc6962Records,
    },
    ErrataSchema {
        name: "LegacyParentSourceRowV1",
        tag: LEGACY_PARENT_SOURCE_ROW_TAG,
        schema_id: LEGACY_PARENT_SOURCE_ROW_SCHEMA_ID,
        body_field_count: 5,
        root_rule: ErrataRootRule::Rfc6962Records,
    },
    ErrataSchema {
        name: "RepositoryPathAliasRecordV1",
        tag: REPOSITORY_PATH_ALIAS_RECORD_TAG,
        schema_id: REPOSITORY_PATH_ALIAS_RECORD_SCHEMA_ID,
        body_field_count: 3,
        root_rule: ErrataRootRule::Rfc6962Records,
    },
    ErrataSchema {
        name: "SourceFileRecordV1",
        tag: SOURCE_FILE_RECORD_TAG,
        schema_id: SOURCE_FILE_RECORD_SCHEMA_ID,
        body_field_count: 6,
        root_rule: ErrataRootRule::Rfc6962Records,
    },
    ErrataSchema {
        name: "DependencyLockRecordV1",
        tag: DEPENDENCY_LOCK_RECORD_TAG,
        schema_id: DEPENDENCY_LOCK_RECORD_SCHEMA_ID,
        body_field_count: 5,
        root_rule: ErrataRootRule::Rfc6962Records,
    },
    ErrataSchema {
        name: "LegalTransitionRowV1",
        tag: LEGAL_TRANSITION_ROW_TAG,
        schema_id: LEGAL_TRANSITION_ROW_SCHEMA_ID,
        body_field_count: 5,
        root_rule: ErrataRootRule::Rfc6962Records,
    },
    ErrataSchema {
        name: "OpaqueIdRegistryRecordV1",
        tag: OPAQUE_ID_REGISTRY_RECORD_TAG,
        schema_id: OPAQUE_ID_REGISTRY_RECORD_SCHEMA_ID,
        body_field_count: 6,
        root_rule: ErrataRootRule::Rfc6962Records,
    },
];

pub fn errata_schema(name: &str) -> Result<&'static ErrataSchema, FormalWireError> {
    ERRATA_SCHEMAS
        .iter()
        .find(|schema| schema.name == name)
        .ok_or_else(|| FormalWireError::new(REJECT_M25_OBJECT_PREFIX, "unknown errata schema"))
}

fn require_errata_array<'a>(
    schema: &ErrataSchema,
    value: &'a CborValue,
) -> Result<&'a [CborValue], FormalWireError> {
    if let ErrataRootRule::NormativeGap(detail) = schema.root_rule {
        return Err(FormalWireError::new(
            FAIL_M25_NORMATIVE_GAP,
            format!("{}: {detail}", schema.name),
        ));
    }
    let values = array(value)?;
    if !exact_prefix(values, schema.tag, schema.schema_id) {
        return Err(FormalWireError::new(
            REJECT_M25_OBJECT_PREFIX,
            format!("{} prefix mismatch", schema.name),
        ));
    }
    if values.len() != 3 + schema.body_field_count {
        return Err(FormalWireError::new(
            REJECT_M25_FIELD_SET,
            format!("{} array arity mismatch", schema.name),
        ));
    }
    Ok(values)
}

fn require_uint(value: &CborValue, field: &str) -> Result<u64, FormalWireError> {
    match value {
        CborValue::Unsigned(value) => Ok(*value),
        _ => Err(FormalWireError::new(
            REJECT_M25_FIELD_TYPE,
            format!("{field} must be an unsigned integer"),
        )),
    }
}

fn require_bytes<'a>(value: &'a CborValue, field: &str) -> Result<&'a [u8], FormalWireError> {
    match value {
        CborValue::Bytes(value) => Ok(value),
        _ => Err(FormalWireError::new(
            REJECT_M25_FIELD_TYPE,
            format!("{field} must be a byte string"),
        )),
    }
}

fn require_exact_bytes<'a>(
    value: &'a CborValue,
    length: usize,
    field: &str,
) -> Result<&'a [u8], FormalWireError> {
    let bytes = require_bytes(value, field)?;
    if bytes.len() != length {
        return Err(FormalWireError::new(
            REJECT_M25_FIELD_TYPE,
            format!("{field} must be exactly {length} bytes"),
        ));
    }
    Ok(bytes)
}

fn require_array<'a>(
    value: &'a CborValue,
    field: &str,
) -> Result<&'a [CborValue], FormalWireError> {
    match value {
        CborValue::Array(value) => Ok(value),
        _ => Err(FormalWireError::new(
            REJECT_M25_FIELD_TYPE,
            format!("{field} must be an array"),
        )),
    }
}

fn require_root(value: &CborValue, field: &str) -> Result<(), FormalWireError> {
    require_exact_bytes(value, 32, field).map(|_| ())
}

fn require_commit(value: &CborValue, field: &str) -> Result<(), FormalWireError> {
    let commit = require_array(value, field)?;
    if commit.len() != 2
        || !matches!(commit[0], CborValue::Unsigned(1))
        || require_exact_bytes(&commit[1], 20, field).is_err()
    {
        return Err(FormalWireError::new(
            REJECT_M25_FIELD_TYPE,
            format!("{field} must be [1, 20-byte Git SHA-1 digest]"),
        ));
    }
    Ok(())
}

fn require_commit_digest<'a>(
    value: &'a CborValue,
    field: &str,
) -> Result<&'a [u8], FormalWireError> {
    require_commit(value, field)?;
    let commit = require_array(value, field)?;
    require_exact_bytes(&commit[1], 20, field)
}

fn require_audited_parent_commit(value: &CborValue, field: &str) -> Result<(), FormalWireError> {
    if require_commit_digest(value, field)? != AUDITED_PARENT_COMMIT_SHA1 {
        return Err(FormalWireError::new(
            REJECT_M25_FIELD_VALUE,
            format!("{field} must bind the frozen audited parent commit"),
        ));
    }
    Ok(())
}

fn require_id16(value: &CborValue, field: &str) -> Result<(), FormalWireError> {
    let bytes = require_exact_bytes(value, 16, field)?;
    if bytes.iter().all(|byte| *byte == 0) {
        return Err(FormalWireError::new(
            REJECT_M25_FIELD_VALUE,
            format!("{field} may not be all zero"),
        ));
    }
    Ok(())
}

fn validate_static_role_metadata(values: &[CborValue]) -> Result<(), FormalWireError> {
    let input_signature_id = require_uint(&values[3], "input_signature_id")?;
    if !matches!(input_signature_id, 1 | 2) {
        return Err(FormalWireError::new(
            REJECT_UNKNOWN_ENUM_VALUE,
            "input_signature_id must be 1 or 2",
        ));
    }
    let role_ids = require_array(&values[4], "role_ids")?;
    let quantity_ids = require_array(&values[5], "quantity_ids")?;
    let scope_ids = require_array(&values[6], "scope_ids")?;
    let orientations = require_array(&values[7], "signed_orientations")?;
    for (field, entries) in [
        ("role_ids", role_ids),
        ("quantity_ids", quantity_ids),
        ("scope_ids", scope_ids),
    ] {
        for entry in entries {
            require_uint(entry, field)?;
        }
    }
    for orientation in orientations {
        if !matches!(orientation, CborValue::Unsigned(1) | CborValue::Negative(0)) {
            return Err(FormalWireError::new(
                REJECT_M25_FIELD_VALUE,
                "signed orientations must be exactly +1 or -1",
            ));
        }
    }
    if input_signature_id == 1 {
        if !role_ids.is_empty()
            || !quantity_ids.is_empty()
            || !scope_ids.is_empty()
            || !orientations.is_empty()
        {
            return Err(FormalWireError::new(
                REJECT_M25_FIELD_VALUE,
                "odd static-role metadata arrays must all be empty",
            ));
        }
    } else {
        let expected_roles = [0, 1, 2, 3].map(CborValue::Unsigned);
        let expected_quantity = [CborValue::Unsigned(0)];
        let expected_scope = [CborValue::Unsigned(3)];
        let expected_orientations = [
            CborValue::Unsigned(1),
            CborValue::Unsigned(1),
            CborValue::Negative(0),
            CborValue::Negative(0),
        ];
        if role_ids != expected_roles
            || quantity_ids != expected_quantity
            || scope_ids != expected_scope
            || orientations != expected_orientations
        {
            return Err(FormalWireError::new(
                REJECT_M25_FIELD_VALUE,
                "sink static-role metadata must equal [roles 0..3],[q0],[scope 3],[1,1,-1,-1]",
            ));
        }
    }
    require_root(&values[8], "metadata_rule_id_digest")
}

fn validate_normative_document_bundle(values: &[CborValue]) -> Result<(), FormalWireError> {
    require_root(&values[3], "bundle_id_digest")?;
    let entries = require_array(&values[4], "document_entries")?;
    if entries.len() != 3 {
        return Err(FormalWireError::new(
            REJECT_M25_FIELD_VALUE,
            "normative document bundle must have exactly three role entries",
        ));
    }
    for (index, entry) in entries.iter().enumerate() {
        let pair = require_array(entry, "document role entry")?;
        if pair.len() != 2 || require_uint(&pair[0], "document_role_id")? != (index + 1) as u64 {
            return Err(FormalWireError::new(
                REJECT_M25_FIELD_VALUE,
                "document role entries must be unique and ordered [1,2,3]",
            ));
        }
        require_root(&pair[1], "normative_document_root")?;
    }
    require_commit(&values[5], "repository_commit_id")
}

fn validate_source_section_profile(values: &[CborValue]) -> Result<(), FormalWireError> {
    require_root(&values[3], "specialized_id_digest")?;
    require_root(&values[4], "governing_normative_document_root")?;
    require_root(&values[5], "section_selector_id_digest")?;
    require_root(&values[6], "section_blob_sha256")?;
    require_uint(&values[7], "section_byte_length")?;
    require_commit(&values[8], "repository_commit_id")
}

fn validate_bridge_replay_statement(values: &[CborValue]) -> Result<(), FormalWireError> {
    require_id16(&values[3], "run_id_16_bytes")?;
    for (index, field) in [
        "diagnostic_formal_bridge_root",
        "m3_execution_candidate_root",
        "child_dsl_spec_root",
        "child_freeze_root",
        "actor_trust_genesis_root",
        "opaque_id_registry_snapshot_root",
    ]
    .iter()
    .enumerate()
    {
        require_root(&values[4 + index], field)?;
    }
    Ok(())
}

fn validate_execution_candidate(values: &[CborValue]) -> Result<(), FormalWireError> {
    require_id16(&values[3], "run_id_16_bytes")?;
    for value in &values[4..=33] {
        require_root(value, "M3ExecutionCandidateV1 input root")?;
    }
    for (index, field) in [
        (34, "canonical_program_budget"),
        (35, "raw_operator_application_cap"),
        (36, "records_per_chunk"),
    ] {
        if require_uint(&values[index], field)? == 0 {
            return Err(FormalWireError::new(
                REJECT_M25_FIELD_VALUE,
                format!("{field} must be positive"),
            ));
        }
    }
    if require_uint(&values[37], "equivalence_mode_id")? != 1 {
        return Err(FormalWireError::new(
            REJECT_UNKNOWN_ENUM_VALUE,
            "equivalence_mode_id must be EXACT_EXTENSIONAL=1",
        ));
    }
    for value in &values[38..=44] {
        require_root(value, "M3ExecutionCandidateV1 contract root")?;
    }
    if values[20] != values[21] {
        return Err(FormalWireError::new(
            FAIL_M3_LEDGER_HEAD_NOT_GENESIS,
            "pre-M3 execution candidate requires ledger head equal to genesis",
        ));
    }
    require_uint(&values[45], "created_at_unix_seconds")?;
    require_commit(&values[46], "repository_commit_id")
}

fn validate_execution_manifest_v2(values: &[CborValue]) -> Result<(), FormalWireError> {
    require_id16(&values[3], "run_id_16_bytes")?;
    for value in &values[4..=8] {
        require_root(value, "M3ExecutionManifestV2 bound root")?;
    }
    require_uint(&values[9], "created_at_unix_seconds")?;
    require_commit(&values[10], "repository_commit_id")
}

fn validate_actor_trust_genesis(values: &[CborValue]) -> Result<(), FormalWireError> {
    require_id16(&values[3], "trust_genesis_id_16_bytes")?;
    let entries = require_array(&values[4], "purpose_key_manifest_entries")?;
    let mut purposes = Vec::with_capacity(entries.len());
    for entry in entries {
        let pair = require_array(entry, "purpose_key_manifest_entry")?;
        if pair.len() != 2 {
            return Err(FormalWireError::new(
                REJECT_M25_FIELD_SET,
                "actor trust entry must contain purpose_id and manifest root",
            ));
        }
        purposes.push(require_uint(&pair[0], "purpose_id")?);
        require_exact_bytes(&pair[1], 32, "actor_key_manifest_root")?;
    }
    if purposes != [1, 2, 3, 4] {
        return Err(FormalWireError::new(
            REJECT_M25_FIELD_VALUE,
            "ActorTrustGenesisV1 purposes must be exactly [1,2,3,4]",
        ));
    }
    require_root(&values[5], "purpose_key_policy_root")?;
    require_uint(&values[6], "created_at_unix_seconds")?;
    require_commit(&values[7], "repository_commit_id")
}

fn validate_opaque_snapshot(values: &[CborValue]) -> Result<(), FormalWireError> {
    let is_genesis = matches!(values[3], CborValue::Null);
    if !is_genesis {
        require_root(&values[3], "previous_snapshot_root_or_null")?;
    }
    require_root(&values[4], "registry_tree_root")?;
    let record_count = require_uint(&values[5], "record_count")?;
    if record_count == 0 || (is_genesis && record_count != 1) || (!is_genesis && record_count < 2) {
        return Err(FormalWireError::new(
            REJECT_M25_FIELD_VALUE,
            "opaque-ID snapshot must be genesis/count 1 or non-genesis/count at least 2",
        ));
    }
    require_root(&values[6], "added_record_root")?;
    require_commit(&values[7], "repository_commit_id")
}

fn validate_parent_audit_bundle(values: &[CborValue]) -> Result<(), FormalWireError> {
    require_audited_parent_commit(&values[3], "audited_parent_repository_commit_id")?;
    for value in &values[4..=6] {
        require_root(value, "parent-audit tree root")?;
    }
    let path_count = require_uint(&values[7], "audited_path_count")?;
    let history_count = require_uint(&values[8], "audited_history_row_count")?;
    let legacy_count = require_uint(&values[9], "legacy_source_count")?;
    if path_count == 0 || history_count == 0 || legacy_count != 2 {
        return Err(FormalWireError::new(
            REJECT_M25_FIELD_VALUE,
            "audit bundle requires nonempty path/history trees and two legacy sources",
        ));
    }
    Ok(())
}

fn validate_parent_absence_attestation_v2(values: &[CborValue]) -> Result<(), FormalWireError> {
    require_root(&values[3], "parent_dsl_version_digest")?;
    require_root(&values[4], "parent_freeze_version_digest")?;
    require_audited_parent_commit(&values[5], "parent_repository_commit_id")?;
    require_root(&values[6], "audit_bundle_root")?;
    if require_uint(&values[7], "absence_reason_bitmask")? != 0b1111 {
        return Err(FormalWireError::new(
            REJECT_M25_FIELD_VALUE,
            "absence_reason_bitmask must equal 0b1111",
        ));
    }
    require_id16(&values[8], "auditor_key_id")?;
    require_uint(&values[9], "audited_at_unix_seconds")?;
    Ok(())
}

fn validate_opaque_registration_intent(values: &[CborValue]) -> Result<(), FormalWireError> {
    if !matches!(require_uint(&values[3], "opaque_id_kind_id")?, 1 | 2) {
        return Err(FormalWireError::new(
            REJECT_UNKNOWN_ENUM_VALUE,
            "OpaqueIdKindId must be RUN_ID=1 or LEDGER_ID=2",
        ));
    }
    require_id16(&values[4], "opaque_id_16_bytes")?;
    require_root(&values[5], "registration_context_root")?;
    require_uint(&values[6], "created_at_unix_seconds")?;
    require_commit(&values[7], "repository_commit_id")
}

fn validate_signature_record(value: &CborValue) -> Result<&[u8], FormalWireError> {
    let record = require_array(value, "SignatureRecordV1")?;
    if record.len() != 2 {
        return Err(FormalWireError::new(
            REJECT_M25_FIELD_SET,
            "SignatureRecordV1 must contain key_id and signature bytes",
        ));
    }
    let key_id = require_exact_bytes(&record[0], 16, "signature key_id")?;
    require_exact_bytes(&record[1], 64, "Ed25519 signature")?;
    Ok(key_id)
}

fn validate_signed_manifest_envelope(values: &[CborValue]) -> Result<(), FormalWireError> {
    require_uint(&values[3], "enclosed_object_tag")?;
    require_root(&values[4], "enclosed_manifest_root")?;
    require_uint(&values[5], "created_at_unix_seconds")?;
    require_uint(&values[6], "signer_key_epoch")?;
    let signatures = require_array(&values[7], "signatures")?;
    if signatures.len() != 1 {
        return Err(FormalWireError::new(
            REJECT_M25_SIGNATURE_COUNT,
            "SignedManifestEnvelopeV1 must contain exactly one signature",
        ));
    }
    let mut previous: Option<Vec<u8>> = None;
    for signature in signatures {
        let key_id = validate_signature_record(signature)?.to_vec();
        if previous.as_ref().is_some_and(|value| value >= &key_id) {
            return Err(FormalWireError::new(
                REJECT_M25_FIELD_VALUE,
                "signature key IDs must be unique and ascending",
            ));
        }
        previous = Some(key_id);
    }
    Ok(())
}

fn validate_m3_run_genesis(values: &[CborValue]) -> Result<(), FormalWireError> {
    require_id16(&values[3], "run_id")?;
    require_root(&values[4], "execution_manifest_root")?;
    if require_uint(&values[5], "initial_state_id")? != 0 {
        return Err(FormalWireError::new(
            REJECT_M25_FIELD_VALUE,
            "M3RunGenesisV1 initial state must be M3StateId.NOT_RUN=0",
        ));
    }
    if values[6..=20]
        .iter()
        .any(|value| !matches!(value, CborValue::Null))
    {
        return Err(FormalWireError::new(
            FAIL_M3_OUTPUT_ROOT_PREPOPULATED,
            "all fifteen M3 run-produced output slots must be null",
        ));
    }
    require_uint(&values[21], "created_at_unix_seconds")?;
    require_commit(&values[22], "repository_commit_id")
}

fn validate_m3_run_state(values: &[CborValue]) -> Result<(), FormalWireError> {
    require_id16(&values[3], "run_id")?;
    require_uint(&values[4], "transition_index")?;
    if !matches!(values[5], CborValue::Null) {
        require_root(&values[5], "previous_state_record_root_or_null")?;
    }
    for value in &values[6..=10] {
        require_uint(value, "M3 state/phase/reason ID")?;
    }
    require_root(&values[11], "execution_manifest_root")?;
    if !matches!(values[12], CborValue::Null) {
        require_root(&values[12], "triggering_receipt_root_or_null")?;
    }
    require_uint(&values[13], "recorded_at_unix_seconds")?;
    Ok(())
}

pub fn validate_m3_start_record(value: &CborValue) -> Result<(), FormalWireError> {
    validate_errata_object("M3RunStateRecordV1", value)?;
    let values = array(value)?;
    if require_uint(&values[4], "transition_index")? != 0
        || !matches!(values[5], CborValue::Null)
        || require_uint(&values[6], "from_state_id")? != 0
        || require_uint(&values[7], "from_phase_id")? != 0
        || require_uint(&values[8], "to_state_id")? != 1
        || require_uint(&values[9], "to_phase_id")? != 1
        || require_uint(&values[10], "transition_reason_id")? != 1
        || !matches!(values[12], CborValue::Null)
    {
        return Err(FormalWireError::new(
            FAIL_ILLEGAL_M3_STATE_TRANSITION,
            "M3 start must be index 0, NOT_RUN/NONE -> RUNNING/CANONICAL_ENUMERATION, reason ENTRY_GATES_24_OF_24, with null previous/trigger receipt",
        ));
    }
    Ok(())
}

fn validate_m3_dual_replay_agreement(values: &[CborValue]) -> Result<(), FormalWireError> {
    require_id16(&values[3], "run_id")?;
    require_root(&values[4], "execution_manifest_root")?;
    require_root(&values[5], "python_enumeration_receipt_root")?;
    require_root(&values[6], "rust_enumeration_receipt_root")?;
    require_uint(&values[7], "agreed_closure_status_id")?;
    require_array(&values[14], "role_agreement_entries")?;
    match &values[15] {
        CborValue::Bool(_) => {}
        _ => {
            return Err(FormalWireError::new(
                REJECT_M25_FIELD_TYPE,
                "enumeration_agreement must be a CBOR bool",
            ));
        }
    }
    require_uint(&values[16], "role_agreement_status_id")?;
    require_uint(&values[18], "created_at_unix_seconds")?;
    Ok(())
}

fn validate_audited_path_record(values: &[CborValue]) -> Result<(), FormalWireError> {
    require_root(&values[3], "repository_path_alias_id_digest")?;
    require_bytes(&values[4], "raw_repository_path_utf8_bytes")?;
    if require_uint(&values[5], "git_object_algorithm_id")? != 1 {
        return Err(FormalWireError::new(
            REJECT_M25_FIELD_VALUE,
            "Git object algorithm must be SHA1=1",
        ));
    }
    require_exact_bytes(&values[6], 20, "git_blob_digest")?;
    require_uint(&values[7], "file_mode")?;
    require_uint(&values[8], "byte_length")?;
    Ok(())
}

fn validate_audited_history_record(values: &[CborValue]) -> Result<(), FormalWireError> {
    require_uint(&values[3], "commit_generation")?;
    require_commit(&values[4], "repository_commit_id")?;
    let parents = require_array(&values[5], "ordered_parent_commit_ids")?;
    for parent in parents {
        require_commit(parent, "parent_repository_commit_id")?;
    }
    require_root(&values[6], "touched_path_set_root")
}

fn validate_legacy_source_record(values: &[CborValue]) -> Result<(), FormalWireError> {
    let role_id = require_uint(&values[3], "target_role_id")?;
    if !matches!(role_id, 1 | 2) {
        return Err(FormalWireError::new(
            REJECT_UNKNOWN_ENUM_VALUE,
            "TargetRoleId must be 1 or 2",
        ));
    }
    let (source_id, expected_namespace, diagnostic_suffix) = match role_id {
        1 => (
            LEGACY_OUTSIDE_TARGET_SOURCE_ID,
            1,
            "b491c0a9719fb0279fe02798ede026e440c17a539965514145a7818b15387ac3",
        ),
        2 => (
            LEGACY_NULL_CONTROL_SOURCE_ID,
            2,
            "7fd6f9a6e2b4c6eda0c7e1545ad42cb19666743ede8ed87f40d82c0ef46198a0",
        ),
        _ => unreachable!("role was validated above"),
    };
    let source_digest =
        require_exact_bytes(&values[4], 32, "legacy_parent_payload_source_id_digest")?;
    if source_digest != id_digest_v1(source_id)? {
        return Err(FormalWireError::new(
            REJECT_M25_FIELD_VALUE,
            "legacy source ID does not match its target role",
        ));
    }
    if require_uint(&values[5], "diagnostic_namespace_id")? != expected_namespace {
        return Err(FormalWireError::new(
            REJECT_M25_FIELD_VALUE,
            "legacy diagnostic namespace does not match its target role",
        ));
    }
    let expected_diagnostic = hex_decode(diagnostic_suffix)?;
    if require_exact_bytes(&values[6], 32, "diagnostic_digest")? != expected_diagnostic {
        return Err(FormalWireError::new(
            REJECT_M25_FIELD_VALUE,
            "legacy diagnostic digest must equal the source-ID hex suffix",
        ));
    }
    require_audited_parent_commit(&values[7], "source_repository_commit_id")
}

fn validate_repository_path_alias_record(values: &[CborValue]) -> Result<(), FormalWireError> {
    require_root(&values[3], "path_alias_id_digest")?;
    require_bytes(&values[4], "raw_repository_path_utf8_bytes")?;
    require_commit(&values[5], "repository_commit_id")
}

fn validate_source_file_record(values: &[CborValue]) -> Result<(), FormalWireError> {
    require_root(&values[3], "path_alias_id_digest")?;
    require_bytes(&values[4], "raw_path_bytes")?;
    if require_uint(&values[5], "git_blob_algorithm_id")? != 1 {
        return Err(FormalWireError::new(
            REJECT_M25_FIELD_VALUE,
            "Git blob algorithm must be SHA1=1",
        ));
    }
    require_exact_bytes(&values[6], 20, "git_blob_digest")?;
    require_uint(&values[7], "file_mode")?;
    require_uint(&values[8], "byte_length")?;
    Ok(())
}

fn validate_dependency_lock_record(values: &[CborValue]) -> Result<(), FormalWireError> {
    if !matches!(require_uint(&values[3], "ecosystem_id")?, 1..=3) {
        return Err(FormalWireError::new(
            REJECT_UNKNOWN_ENUM_VALUE,
            "DependencyEcosystemId must be PYTHON=1, RUST=2, or SYSTEM=3",
        ));
    }
    for value in &values[4..=7] {
        require_root(value, "dependency lock digest")?;
    }
    Ok(())
}

fn validate_legal_transition_record(values: &[CborValue]) -> Result<(), FormalWireError> {
    for value in &values[3..=6] {
        require_uint(value, "state/phase ID")?;
    }
    let reasons = require_array(&values[7], "allowed_reason_ids")?;
    let mut previous = None;
    for reason in reasons {
        let reason = require_uint(reason, "allowed_reason_id")?;
        if previous.is_some_and(|value| value >= reason) {
            return Err(FormalWireError::new(
                REJECT_M25_FIELD_VALUE,
                "allowed reason IDs must be unique and ascending",
            ));
        }
        previous = Some(reason);
    }
    Ok(())
}

fn validate_opaque_id_record(values: &[CborValue]) -> Result<(), FormalWireError> {
    require_uint(&values[3], "registry_sequence_number")?;
    if !matches!(require_uint(&values[4], "opaque_id_kind_id")?, 1 | 2) {
        return Err(FormalWireError::new(
            REJECT_UNKNOWN_ENUM_VALUE,
            "OpaqueIdKindId must be RUN_ID=1 or LEDGER_ID=2",
        ));
    }
    require_id16(&values[5], "opaque_id_16_bytes")?;
    require_root(&values[6], "first_seen_object_root")?;
    require_commit(&values[7], "first_seen_repository_commit_id")?;
    require_uint(&values[8], "created_at_unix_seconds")?;
    Ok(())
}

pub fn validate_errata_object(name: &str, value: &CborValue) -> Result<(), FormalWireError> {
    let schema = errata_schema(name)?;
    let values = require_errata_array(schema, value)?;
    match name {
        "NormativeDocumentBundleV1" => validate_normative_document_bundle(values),
        "CanonicalAstProfileSpecV1"
        | "CanonicalCborProfileSpecV1"
        | "Phase2BContractSpecV1"
        | "MdlCodeTableSpecV1"
        | "HiddenArtifactScopeV1" => validate_source_section_profile(values),
        "StaticRoleMetadataV1" => validate_static_role_metadata(values),
        "BridgeReplayStatementV1" => validate_bridge_replay_statement(values),
        "M3ExecutionCandidateV1" => validate_execution_candidate(values),
        "M3ExecutionManifestV2" => validate_execution_manifest_v2(values),
        "ActorTrustGenesisV1" => validate_actor_trust_genesis(values),
        "OpaqueIdRegistrySnapshotV1" => validate_opaque_snapshot(values),
        "ParentAbsenceAuditBundleV1" => validate_parent_audit_bundle(values),
        "ParentManifestAbsenceAttestationV2" => validate_parent_absence_attestation_v2(values),
        "OpaqueIdRegistrationIntentV1" => validate_opaque_registration_intent(values),
        "SignedManifestEnvelopeV1" => validate_signed_manifest_envelope(values),
        "M3RunGenesisV1" => validate_m3_run_genesis(values),
        "M3RunStateRecordV1" => validate_m3_run_state(values),
        "M3DualReplayAgreementV1" => validate_m3_dual_replay_agreement(values),
        "AuditedPathBlobRecordV1" => validate_audited_path_record(values),
        "AuditedHistoryRowV1" => validate_audited_history_record(values),
        "LegacyParentSourceRowV1" => validate_legacy_source_record(values),
        "RepositoryPathAliasRecordV1" => validate_repository_path_alias_record(values),
        "SourceFileRecordV1" => validate_source_file_record(values),
        "DependencyLockRecordV1" => validate_dependency_lock_record(values),
        "LegalTransitionRowV1" => validate_legal_transition_record(values),
        "OpaqueIdRegistryRecordV1" => validate_opaque_id_record(values),
        _ => Err(FormalWireError::new(
            FAIL_M25_NORMATIVE_GAP,
            format!("{name} has no executable exact schema"),
        )),
    }
}

pub fn errata_candidate_content_root(
    name: &str,
    value: &CborValue,
) -> Result<[u8; 32], FormalWireError> {
    let schema = errata_schema(name)?;
    validate_errata_object(name, value)?;
    match schema.root_rule {
        ErrataRootRule::ContentHash(domain) => content_hash(domain, value),
        ErrataRootRule::Rfc6962Records => Err(FormalWireError::new(
            REJECT_M25_FIELD_VALUE,
            "record schema requires errata_record_tree_root",
        )),
        ErrataRootRule::NormativeGap(detail) => {
            Err(FormalWireError::new(FAIL_M25_NORMATIVE_GAP, detail))
        }
    }
}

fn compare_errata_records(
    name: &str,
    left: &[CborValue],
    right: &[CborValue],
) -> Result<std::cmp::Ordering, FormalWireError> {
    use std::cmp::Ordering;

    fn compare_uint(left: &CborValue, right: &CborValue) -> Result<Ordering, FormalWireError> {
        Ok(require_uint(left, "ordering key")?.cmp(&require_uint(right, "ordering key")?))
    }

    fn compare_bytes(left: &CborValue, right: &CborValue) -> Result<Ordering, FormalWireError> {
        Ok(require_bytes(left, "ordering key")?.cmp(require_bytes(right, "ordering key")?))
    }

    fn compare_commit(left: &CborValue, right: &CborValue) -> Result<Ordering, FormalWireError> {
        require_commit(left, "repository_commit_id")?;
        require_commit(right, "repository_commit_id")?;
        let left = require_array(left, "repository_commit_id")?;
        let right = require_array(right, "repository_commit_id")?;
        compare_bytes(&left[1], &right[1])
    }

    let ordering = match name {
        "AuditedPathBlobRecordV1" => compare_bytes(&left[4], &right[4])?
            .then(compare_bytes(&left[3], &right[3])?)
            .then(compare_bytes(&left[6], &right[6])?),
        "AuditedHistoryRowV1" => {
            let generation = compare_uint(&left[3], &right[3])?;
            if generation == Ordering::Equal {
                compare_commit(&left[4], &right[4])?
            } else {
                generation
            }
        }
        "LegacyParentSourceRowV1" => compare_uint(&left[3], &right[3])?,
        "RepositoryPathAliasRecordV1" => {
            compare_bytes(&left[3], &right[3])?.then(compare_bytes(&left[4], &right[4])?)
        }
        "SourceFileRecordV1" => compare_bytes(&left[4], &right[4])?,
        "DependencyLockRecordV1" => compare_uint(&left[3], &right[3])?
            .then(compare_bytes(&left[4], &right[4])?)
            .then(compare_bytes(&left[5], &right[5])?),
        "LegalTransitionRowV1" => compare_uint(&left[3], &right[3])?
            .then(compare_uint(&left[4], &right[4])?)
            .then(compare_uint(&left[5], &right[5])?)
            .then(compare_uint(&left[6], &right[6])?),
        "OpaqueIdRegistryRecordV1" => compare_uint(&left[3], &right[3])?,
        _ => {
            return Err(FormalWireError::new(
                REJECT_M25_FIELD_VALUE,
                "object is not an errata record-tree schema",
            ))
        }
    };
    Ok(ordering)
}

pub fn errata_record_tree_root(
    name: &str,
    records: &[CborValue],
) -> Result<[u8; 32], FormalWireError> {
    let schema = errata_schema(name)?;
    if schema.root_rule != ErrataRootRule::Rfc6962Records {
        return Err(FormalWireError::new(
            REJECT_M25_FIELD_VALUE,
            "schema is not an RFC6962 record tree",
        ));
    }
    if records.is_empty() {
        return Err(FormalWireError::new(
            REJECT_M25_FIELD_VALUE,
            "errata record tree must be nonempty",
        ));
    }
    let mut arrays = Vec::with_capacity(records.len());
    for record in records {
        validate_errata_object(name, record)?;
        arrays.push(array(record)?);
    }
    for pair in arrays.windows(2) {
        if compare_errata_records(name, pair[0], pair[1])? != std::cmp::Ordering::Less {
            return Err(FormalWireError::new(
                REJECT_M25_RECORD_ORDER,
                format!("{name} records are not in unique frozen order"),
            ));
        }
    }

    if name == "LegacyParentSourceRowV1" {
        let roles = arrays
            .iter()
            .map(|record| require_uint(&record[3], "target_role_id"))
            .collect::<Result<Vec<_>, _>>()?;
        if roles != [1, 2] {
            return Err(FormalWireError::new(
                REJECT_M25_FIELD_VALUE,
                "legacy source tree must contain exactly roles [1,2]",
            ));
        }
    }
    if name == "RepositoryPathAliasRecordV1" {
        for pair in arrays.windows(2) {
            if pair[0][3] == pair[1][3] || pair[0][4] == pair[1][4] {
                return Err(FormalWireError::new(
                    REJECT_M25_RECORD_ORDER,
                    "path alias digest and raw path must both be unique",
                ));
            }
        }
    }
    if name == "OpaqueIdRegistryRecordV1" {
        for (index, record) in arrays.iter().enumerate() {
            if require_uint(&record[3], "registry_sequence_number")? != index as u64 {
                return Err(FormalWireError::new(
                    FAIL_OPAQUE_ID_REGISTRY_SEQUENCE,
                    "opaque-ID registry sequence must be contiguous from zero",
                ));
            }
        }
        for (index, record) in arrays.iter().enumerate() {
            let raw_id = require_exact_bytes(&record[5], 16, "opaque_id_16_bytes")?;
            if arrays[..index]
                .iter()
                .any(|previous| require_bytes(&previous[5], "opaque_id_16_bytes") == Ok(raw_id))
            {
                return Err(FormalWireError::new(
                    FAIL_OPAQUE_ID_ALREADY_USED,
                    "raw opaque IDs must be globally unique across kinds",
                ));
            }
        }
    }

    let encoded = records
        .iter()
        .map(encode_canonical_cbor)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rfc6962_root(&encoded))
}

pub fn validate_opaque_snapshot_append(
    snapshot: &CborValue,
    previous_snapshot_root: Option<&[u8; 32]>,
    previous_record_count: usize,
    records_through_snapshot: &[CborValue],
) -> Result<(), FormalWireError> {
    validate_errata_object("OpaqueIdRegistrySnapshotV1", snapshot)?;
    let values = array(snapshot)?;
    match (previous_snapshot_root, &values[3]) {
        (None, CborValue::Null) if previous_record_count == 0 => {}
        (Some(expected), CborValue::Bytes(actual))
            if previous_record_count > 0 && actual.as_slice() == expected => {}
        _ => {
            return Err(FormalWireError::new(
                FAIL_OPAQUE_ID_REGISTRY_SNAPSHOT,
                "snapshot previous root/count do not describe genesis or one append",
            ))
        }
    }
    let expected_count = previous_record_count
        .checked_add(1)
        .ok_or_else(|| FormalWireError::new(FAIL_OPAQUE_ID_REGISTRY_SNAPSHOT, "count overflow"))?;
    if records_through_snapshot.len() != expected_count
        || require_uint(&values[5], "record_count")? != expected_count as u64
    {
        return Err(FormalWireError::new(
            FAIL_OPAQUE_ID_REGISTRY_SNAPSHOT,
            "each opaque-ID snapshot must append exactly one record",
        ));
    }
    let tree_root = errata_record_tree_root("OpaqueIdRegistryRecordV1", records_through_snapshot)?;
    let added_root = errata_record_tree_root(
        "OpaqueIdRegistryRecordV1",
        &records_through_snapshot[expected_count - 1..expected_count],
    )?;
    if require_exact_bytes(&values[4], 32, "registry_tree_root")? != tree_root
        || require_exact_bytes(&values[6], 32, "added_record_root")? != added_root
    {
        return Err(FormalWireError::new(
            FAIL_OPAQUE_ID_REGISTRY_SNAPSHOT,
            "snapshot tree or singleton added-record root mismatch",
        ));
    }
    Ok(())
}

pub fn bridge_attestation_signature_preimage(
    bridge_replay_statement_root: &[u8; 32],
    signer_purpose_id: u16,
    signer_key_epoch: u64,
) -> Result<Vec<u8>, FormalWireError> {
    if !matches!(signer_purpose_id, 1..=3) {
        return Err(FormalWireError::new(
            REJECT_UNKNOWN_ENUM_VALUE,
            "bridge signer purpose must be 1, 2, or 3",
        ));
    }
    let mut output = Vec::with_capacity(BRIDGE_ATTESTATION_SIGNATURE_DOMAIN.len() + 1 + 32 + 2 + 8);
    output.extend_from_slice(BRIDGE_ATTESTATION_SIGNATURE_DOMAIN);
    output.push(0);
    output.extend_from_slice(bridge_replay_statement_root);
    output.extend_from_slice(&signer_purpose_id.to_be_bytes());
    output.extend_from_slice(&signer_key_epoch.to_be_bytes());
    Ok(output)
}

pub fn external_signature_preimage(
    domain: &str,
    enclosed_object_root: &[u8; 32],
    signer_purpose_id: u16,
    signer_key_epoch: u64,
) -> Result<Vec<u8>, FormalWireError> {
    validate_hash_domain(domain)?;
    if signer_purpose_id == 0 || signer_purpose_id > 5 {
        return Err(FormalWireError::new(
            REJECT_UNKNOWN_ENUM_VALUE,
            "external signer purpose must be in 1..=5",
        ));
    }
    let mut output = Vec::with_capacity(domain.len() + 1 + 32 + 2 + 8);
    output.extend_from_slice(domain.as_bytes());
    output.push(0);
    output.extend_from_slice(enclosed_object_root);
    output.extend_from_slice(&signer_purpose_id.to_be_bytes());
    output.extend_from_slice(&signer_key_epoch.to_be_bytes());
    Ok(output)
}

/// Exact owner-authorized external-signature preimage selected by object tag.
///
/// This is still a pure byte constructor: it does not load keys or sign.
pub fn external_signature_preimage_for_tag(
    enclosed_object_tag: u64,
    enclosed_object_root: &[u8; 32],
    signer_purpose_id: u16,
    signer_key_epoch: u64,
) -> Result<Vec<u8>, FormalWireError> {
    let (domain, required_purpose) = match enclosed_object_tag {
        0x310e => {
            return bridge_attestation_signature_preimage(
                enclosed_object_root,
                signer_purpose_id,
                signer_key_epoch,
            )
        }
        0x3103 => (CUSTODIAN_SPLIT_SEED_COMMITMENT_SIGNATURE_DOMAIN, 1),
        0x3105 => (CUSTODIAN_BINDING_SIGNATURE_DOMAIN, 1),
        0x3106 => (CUSTODIAN_SEED_CONTINUITY_SIGNATURE_DOMAIN, 1),
        0x3108 => (CUSTODIAN_LEDGER_GENESIS_SIGNATURE_DOMAIN, 1),
        0x3114 => (PARENT_ABSENCE_AUDITOR_SIGNATURE_DOMAIN, 4),
        _ => {
            return Err(FormalWireError::new(
                REJECT_M25_FIELD_VALUE,
                "object tag has no frozen external-signature domain",
            ))
        }
    };
    if signer_purpose_id != required_purpose {
        return Err(FormalWireError::new(
            REJECT_M25_FIELD_VALUE,
            format!("object tag requires signer purpose {required_purpose}"),
        ));
    }
    external_signature_preimage(
        domain,
        enclosed_object_root,
        signer_purpose_id,
        signer_key_epoch,
    )
}

pub fn validate_bridge_attester_purposes(purposes: &[u16]) -> Result<(), FormalWireError> {
    let mut normalized = purposes.to_vec();
    normalized.sort_unstable();
    if normalized != [1, 2, 3] {
        return Err(FormalWireError::new(
            FAIL_BRIDGE_ATTESTATION_PURPOSE_SET,
            "bridge attestation bundle purposes must be exactly [1,2,3]",
        ));
    }
    Ok(())
}

pub fn validate_actor_key_material(
    entries: &[(u16, [u8; 16], [u8; 32])],
) -> Result<(), FormalWireError> {
    let mut purposes = entries.iter().map(|entry| entry.0).collect::<Vec<_>>();
    purposes.sort_unstable();
    if purposes != [1, 2, 3, 4] {
        return Err(FormalWireError::new(
            FAIL_ACTOR_TRUST_PURPOSE_SET,
            "pre-M4 actor key material must contain purposes [1,2,3,4]",
        ));
    }
    for (index, (_, key_id, public_key)) in entries.iter().enumerate() {
        if *key_id == [0; 16] || *public_key == [0; 32] {
            return Err(FormalWireError::new(
                REJECT_M25_FIELD_VALUE,
                "actor key IDs and public keys may not be all zero",
            ));
        }
        if entries[..index]
            .iter()
            .any(|(_, previous_id, previous_key)| {
                previous_id == key_id || previous_key == public_key
            })
        {
            return Err(FormalWireError::new(
                FAIL_ACTOR_KEY_CROSS_PURPOSE_REUSE,
                "actor key IDs and raw public keys must both be pairwise distinct",
            ));
        }
    }
    Ok(())
}

/// Validate the trust-genesis purpose/root entries against dereferenced actor
/// key manifests.  Key IDs and public keys are checked separately by
/// [`validate_actor_key_material`].
pub fn validate_actor_trust_manifest_roots(
    trust_genesis: &CborValue,
    dereferenced_manifest_roots: &[(u16, [u8; 32])],
) -> Result<(), FormalWireError> {
    validate_errata_object("ActorTrustGenesisV1", trust_genesis)?;
    let mut supplied = dereferenced_manifest_roots.to_vec();
    supplied.sort_unstable_by_key(|entry| entry.0);
    if supplied.iter().map(|entry| entry.0).collect::<Vec<_>>() != [1, 2, 3, 4] {
        return Err(FormalWireError::new(
            FAIL_ACTOR_TRUST_PURPOSE_SET,
            "pre-M4 actor trust requires dereferenced purposes [1,2,3,4]",
        ));
    }

    let values = array(trust_genesis)?;
    let entries = require_array(&values[4], "purpose_key_manifest_entries")?;
    let mut bound = Vec::with_capacity(entries.len());
    for entry in entries {
        let pair = require_array(entry, "purpose_key_manifest_entry")?;
        let purpose = require_uint(&pair[0], "purpose_id")? as u16;
        let root: [u8; 32] = require_exact_bytes(&pair[1], 32, "actor_key_manifest_root")?
            .try_into()
            .expect("validated 32-byte root");
        bound.push((purpose, root));
    }
    if bound != supplied {
        return Err(FormalWireError::new(
            FAIL_ACTOR_TRUST_KEY_ROOT_MISMATCH,
            "trust genesis entries do not match dereferenced actor-key manifest roots",
        ));
    }
    Ok(())
}

pub fn validate_external_input_attestation_entries(
    entries: &[(u16, u64)],
) -> Result<(), FormalWireError> {
    let expected = [
        (1, 0x3103),
        (1, 0x3105),
        (1, 0x3106),
        (1, 0x3108),
        (4, 0x3114),
    ];
    let mut normalized = entries.to_vec();
    normalized.sort_unstable();
    if normalized != expected {
        return Err(FormalWireError::new(
            FAIL_EXTERNAL_INPUT_ATTESTATION_COVERAGE,
            "external-input attestation bundle must contain the exact four custodian and one auditor entries",
        ));
    }
    Ok(())
}

pub fn validate_null_witness_binding(
    target_role_id: u16,
    target_spec_witness: Option<&[u8; 32]>,
    target_bundle_witness: Option<&[u8; 32]>,
) -> Result<(), FormalWireError> {
    if !matches!(target_role_id, 1 | 2) {
        return Err(FormalWireError::new(
            REJECT_UNKNOWN_ENUM_VALUE,
            "TargetRoleId must be 1 or 2",
        ));
    }
    if target_role_id == 2
        && (target_spec_witness.is_none()
            || target_bundle_witness.is_none()
            || target_spec_witness != target_bundle_witness)
    {
        return Err(FormalWireError::new(
            FAIL_NULL_WITNESS_BINDING_MISMATCH,
            "null-role witness hashes must be non-null and byte-identical",
        ));
    }
    Ok(())
}

const ERRATA_VECTOR_ROOT_PREFIX: &[u8] = b"HEGEL/M25/ERRATA/VECTOR/ROOT/V1\0";
const ERRATA_VECTOR_ID_PREFIX: &[u8] = b"HEGEL/M25/ERRATA/VECTOR/ID/V1\0";
const ERRATA_VECTOR_KEY_PREFIX: &[u8] = b"HEGEL/M25/ERRATA/VECTOR/KEY/V1\0";
const ERRATA_VECTOR_SIG_A_PREFIX: &[u8] = b"HEGEL/M25/ERRATA/VECTOR/SIG/A/V1\0";
const ERRATA_VECTOR_SIG_B_PREFIX: &[u8] = b"HEGEL/M25/ERRATA/VECTOR/SIG/B/V1\0";
pub const ERRATA_VECTOR_BASE_TIMESTAMP: u64 = 1_704_067_200;
pub const AUDITED_PARENT_COMMIT_SHA1: [u8; 20] = [
    0xfb, 0x3a, 0x3e, 0xe4, 0x86, 0x5a, 0x14, 0x0c, 0x55, 0x88, 0x21, 0x01, 0x7d, 0xdd, 0x3e, 0x9a,
    0x6a, 0x99, 0xde, 0x48,
];
pub const ERRATA_VECTOR_GIT_SHA1: [u8; 20] = AUDITED_PARENT_COMMIT_SHA1;

fn validate_vector_label(label: &str) -> Result<(), FormalWireError> {
    if label.is_empty() || !label.is_ascii() {
        return Err(FormalWireError::new(
            REJECT_M25_FIELD_VALUE,
            "errata vector label must be nonempty ASCII",
        ));
    }
    Ok(())
}

pub fn errata_vector_root(label: &str) -> Result<[u8; 32], FormalWireError> {
    validate_vector_label(label)?;
    Ok(sha256(&[ERRATA_VECTOR_ROOT_PREFIX, label.as_bytes()]))
}

pub fn errata_vector_id(label: &str) -> Result<[u8; 16], FormalWireError> {
    validate_vector_label(label)?;
    let digest = sha256(&[ERRATA_VECTOR_ID_PREFIX, label.as_bytes()]);
    let mut output = [0_u8; 16];
    output.copy_from_slice(&digest[..16]);
    Ok(output)
}

pub fn errata_vector_key(label: &str) -> Result<[u8; 32], FormalWireError> {
    validate_vector_label(label)?;
    Ok(sha256(&[ERRATA_VECTOR_KEY_PREFIX, label.as_bytes()]))
}

pub fn errata_vector_signature(label: &str) -> Result<[u8; 64], FormalWireError> {
    validate_vector_label(label)?;
    let first = sha256(&[ERRATA_VECTOR_SIG_A_PREFIX, label.as_bytes()]);
    let second = sha256(&[ERRATA_VECTOR_SIG_B_PREFIX, label.as_bytes()]);
    let mut output = [0_u8; 64];
    output[..32].copy_from_slice(&first);
    output[32..].copy_from_slice(&second);
    Ok(output)
}

fn vector_git_commit() -> CborValue {
    CborValue::Array(vec![
        CborValue::Unsigned(1),
        CborValue::Bytes(ERRATA_VECTOR_GIT_SHA1.to_vec()),
    ])
}

fn errata_object_value(schema: &ErrataSchema, body: Vec<CborValue>) -> CborValue {
    let mut values = Vec::with_capacity(3 + body.len());
    values.push(CborValue::Unsigned(1));
    values.push(CborValue::Unsigned(schema.tag));
    values.push(CborValue::Bytes(schema.schema_id.to_vec()));
    values.extend(body);
    CborValue::Array(values)
}

fn root_value(label: &str) -> Result<CborValue, FormalWireError> {
    Ok(CborValue::Bytes(errata_vector_root(label)?.to_vec()))
}

fn id_value(label: &str) -> Result<CborValue, FormalWireError> {
    Ok(CborValue::Bytes(errata_vector_id(label)?.to_vec()))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrataObjectVector {
    pub name: String,
    pub schema_name: String,
    pub tag: u64,
    pub status: &'static str,
    pub bytes: Option<Vec<u8>>,
    pub candidate_root: Option<[u8; 32]>,
    pub error_code: Option<&'static str>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrataRecordTreeVector {
    pub name: String,
    pub schema_name: String,
    pub tag: u64,
    pub status: &'static str,
    pub record_count: usize,
    pub first_record_cbor: Option<Vec<u8>>,
    pub root: Option<[u8; 32]>,
    pub error_code: Option<&'static str>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrataGuardVector {
    pub vector_id: String,
    pub error_code: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrataVectorReport {
    pub machine_freeze_id: &'static str,
    pub vector_schema: &'static str,
    pub objects: Vec<ErrataObjectVector>,
    pub record_trees: Vec<ErrataRecordTreeVector>,
    pub guard_errors: Vec<ErrataGuardVector>,
}

const PASS_CANDIDATE: &str = "PASS_CANDIDATE_NON_AUTHORITATIVE";

fn positive_object_vector(
    vector_name: &str,
    schema_name: &str,
    value: CborValue,
) -> Result<ErrataObjectVector, FormalWireError> {
    let schema = errata_schema(schema_name)?;
    validate_errata_object(schema_name, &value)?;
    Ok(ErrataObjectVector {
        name: vector_name.to_owned(),
        schema_name: schema_name.to_owned(),
        tag: schema.tag,
        status: PASS_CANDIDATE,
        bytes: Some(encode_canonical_cbor(&value)?),
        candidate_root: Some(errata_candidate_content_root(schema_name, &value)?),
        error_code: None,
    })
}

fn positive_record_tree_vector(
    vector_name: &str,
    schema_name: &str,
    records: Vec<CborValue>,
) -> Result<ErrataRecordTreeVector, FormalWireError> {
    let schema = errata_schema(schema_name)?;
    let first_record_cbor = records.first().map(encode_canonical_cbor).transpose()?;
    Ok(ErrataRecordTreeVector {
        name: vector_name.to_owned(),
        schema_name: schema_name.to_owned(),
        tag: schema.tag,
        status: PASS_CANDIDATE,
        record_count: records.len(),
        first_record_cbor,
        root: Some(errata_record_tree_root(schema_name, &records)?),
        error_code: None,
    })
}

fn expected_guard_error<T>(
    vector_id: &str,
    expected_code: &'static str,
    result: Result<T, FormalWireError>,
) -> Result<ErrataGuardVector, FormalWireError> {
    match result {
        Err(error) if error.code == expected_code => Ok(ErrataGuardVector {
            vector_id: vector_id.to_owned(),
            error_code: error.code,
        }),
        Err(error) => Err(FormalWireError::new(
            REJECT_M25_FIELD_VALUE,
            format!(
                "guard {vector_id} returned {}, expected {expected_code}",
                error.code
            ),
        )),
        Ok(_) => Err(FormalWireError::new(
            REJECT_M25_FIELD_VALUE,
            format!("guard {vector_id} unexpectedly accepted its negative vector"),
        )),
    }
}

fn schema_value(name: &str, body: Vec<CborValue>) -> Result<CborValue, FormalWireError> {
    Ok(errata_object_value(errata_schema(name)?, body))
}

fn source_section_profile_value(
    schema_name: &str,
    id_label: &str,
    governing_root: [u8; 32],
    selector_label: &str,
    section_label: &str,
    section_length: u64,
) -> Result<CborValue, FormalWireError> {
    schema_value(
        schema_name,
        vec![
            root_value(id_label)?,
            CborValue::Bytes(governing_root.to_vec()),
            root_value(selector_label)?,
            root_value(section_label)?,
            CborValue::Unsigned(section_length),
            vector_git_commit(),
        ],
    )
}

pub fn generate_errata_vector_report_v1() -> Result<ErrataVectorReport, FormalWireError> {
    let mut objects = Vec::new();
    let mut record_trees = Vec::new();
    let mut guard_errors = Vec::new();

    let bundle_value = schema_value(
        "NormativeDocumentBundleV1",
        vec![
            root_value("normative_document_bundle_id")?,
            CborValue::Array(vec![
                CborValue::Array(vec![
                    CborValue::Unsigned(1),
                    root_value("base_amendment_document_root")?,
                ]),
                CborValue::Array(vec![
                    CborValue::Unsigned(2),
                    root_value("errata_resolution_document_root")?,
                ]),
                CborValue::Array(vec![
                    CborValue::Unsigned(3),
                    root_value("implementation_closure_addendum_document_root")?,
                ]),
            ]),
            vector_git_commit(),
        ],
    )?;
    let bundle_vector = positive_object_vector(
        "NormativeDocumentBundleV1",
        "NormativeDocumentBundleV1",
        bundle_value.clone(),
    )?;
    let bundle_root = bundle_vector
        .candidate_root
        .expect("positive object vector has a root");
    objects.push(bundle_vector);

    let profile_definitions = [
        (
            "CanonicalAstProfileSpecV1",
            "canonical_ast_profile_id",
            "canonical_ast_section_selector",
            "canonical_ast_section_blob",
            4096,
        ),
        (
            "CanonicalCborProfileSpecV1",
            "canonical_cbor_profile_id",
            "canonical_cbor_section_selector",
            "canonical_cbor_section_blob",
            2048,
        ),
        (
            "Phase2BContractSpecV1",
            "phase2b_contract_id",
            "phase2b_section_selector",
            "phase2b_section_blob",
            8192,
        ),
        (
            "MdlCodeTableSpecV1",
            "mdl_code_table_id",
            "mdl_section_selector",
            "mdl_section_blob",
            1024,
        ),
        (
            "HiddenArtifactScopeV1",
            "hidden_artifact_scope_policy_id",
            "hidden_scope_section_selector",
            "hidden_scope_section_blob",
            512,
        ),
    ];
    for (name, id_label, selector_label, section_label, section_length) in profile_definitions {
        objects.push(positive_object_vector(
            name,
            name,
            source_section_profile_value(
                name,
                id_label,
                bundle_root,
                selector_label,
                section_label,
                section_length,
            )?,
        )?);
    }

    let odd_static_value = schema_value(
        "StaticRoleMetadataV1",
        vec![
            CborValue::Unsigned(1),
            CborValue::Array(vec![]),
            CborValue::Array(vec![]),
            CborValue::Array(vec![]),
            CborValue::Array(vec![]),
            root_value("static_role_metadata_rule")?,
        ],
    )?;
    objects.push(positive_object_vector(
        "StaticRoleMetadataV1.odd",
        "StaticRoleMetadataV1",
        odd_static_value,
    )?);
    let sink_static_value = schema_value(
        "StaticRoleMetadataV1",
        vec![
            CborValue::Unsigned(2),
            CborValue::Array((0..4).map(CborValue::Unsigned).collect()),
            CborValue::Array(vec![CborValue::Unsigned(0)]),
            CborValue::Array(vec![CborValue::Unsigned(3)]),
            CborValue::Array(vec![
                CborValue::Unsigned(1),
                CborValue::Unsigned(1),
                CborValue::Negative(0),
                CborValue::Negative(0),
            ]),
            root_value("static_role_metadata_rule")?,
        ],
    )?;
    objects.push(positive_object_vector(
        "StaticRoleMetadataV1.sink",
        "StaticRoleMetadataV1",
        sink_static_value.clone(),
    )?);

    let run_id = errata_vector_id("run_id")?;
    let ledger_id = errata_vector_id("ledger_id")?;
    let run_intent_value = schema_value(
        "OpaqueIdRegistrationIntentV1",
        vec![
            CborValue::Unsigned(1),
            CborValue::Bytes(run_id.to_vec()),
            root_value("run_registration_context")?,
            CborValue::Unsigned(ERRATA_VECTOR_BASE_TIMESTAMP),
            vector_git_commit(),
        ],
    )?;
    let run_intent_vector = positive_object_vector(
        "OpaqueIdRegistrationIntentV1.run",
        "OpaqueIdRegistrationIntentV1",
        run_intent_value,
    )?;
    let run_intent_root = run_intent_vector
        .candidate_root
        .expect("positive object vector has a root");
    objects.push(run_intent_vector);

    let path_alias_a = id_digest_v1("repo-path:errata-vector-a")?;
    let path_alias_b = id_digest_v1("repo-path:errata-vector-b")?;
    let blob_a = &errata_vector_root("git_blob_a")?[..20];
    let blob_b = &errata_vector_root("git_blob_b")?[..20];
    let audited_path_records = vec![
        schema_value(
            "AuditedPathBlobRecordV1",
            vec![
                CborValue::Bytes(path_alias_a.to_vec()),
                CborValue::Bytes(b"Hegel Machine/a.md".to_vec()),
                CborValue::Unsigned(1),
                CborValue::Bytes(blob_a.to_vec()),
                CborValue::Unsigned(0o100644),
                CborValue::Unsigned(123),
            ],
        )?,
        schema_value(
            "AuditedPathBlobRecordV1",
            vec![
                CborValue::Bytes(path_alias_b.to_vec()),
                CborValue::Bytes(b"Hegel Machine/b.md".to_vec()),
                CborValue::Unsigned(1),
                CborValue::Bytes(blob_b.to_vec()),
                CborValue::Unsigned(0o100644),
                CborValue::Unsigned(456),
            ],
        )?,
    ];
    let audited_path_vector = positive_record_tree_vector(
        "AuditedPathBlobRecordV1",
        "AuditedPathBlobRecordV1",
        audited_path_records.clone(),
    )?;
    let audited_path_root = audited_path_vector.root.expect("positive tree has root");
    record_trees.push(audited_path_vector);

    let history_records = vec![
        schema_value(
            "AuditedHistoryRowV1",
            vec![
                CborValue::Unsigned(0),
                vector_git_commit(),
                CborValue::Array(vec![]),
                CborValue::Bytes(audited_path_root.to_vec()),
            ],
        )?,
        schema_value(
            "AuditedHistoryRowV1",
            vec![
                CborValue::Unsigned(1),
                vector_git_commit(),
                CborValue::Array(vec![vector_git_commit()]),
                CborValue::Bytes(audited_path_root.to_vec()),
            ],
        )?,
    ];
    let history_vector = positive_record_tree_vector(
        "AuditedHistoryRowV1",
        "AuditedHistoryRowV1",
        history_records,
    )?;
    let history_root = history_vector.root.expect("positive tree has root");
    record_trees.push(history_vector);

    let legacy_source_records = vec![
        schema_value(
            "LegacyParentSourceRowV1",
            vec![
                CborValue::Unsigned(1),
                CborValue::Bytes(id_digest_v1(LEGACY_OUTSIDE_TARGET_SOURCE_ID)?.to_vec()),
                CborValue::Unsigned(1),
                CborValue::Bytes(hex_decode(
                    "b491c0a9719fb0279fe02798ede026e440c17a539965514145a7818b15387ac3",
                )?),
                vector_git_commit(),
            ],
        )?,
        schema_value(
            "LegacyParentSourceRowV1",
            vec![
                CborValue::Unsigned(2),
                CborValue::Bytes(id_digest_v1(LEGACY_NULL_CONTROL_SOURCE_ID)?.to_vec()),
                CborValue::Unsigned(2),
                CborValue::Bytes(hex_decode(
                    "7fd6f9a6e2b4c6eda0c7e1545ad42cb19666743ede8ed87f40d82c0ef46198a0",
                )?),
                vector_git_commit(),
            ],
        )?,
    ];
    let legacy_source_vector = positive_record_tree_vector(
        "LegacyParentSourceRowV1",
        "LegacyParentSourceRowV1",
        legacy_source_records,
    )?;
    let legacy_source_root = legacy_source_vector.root.expect("positive tree has root");
    record_trees.push(legacy_source_vector);

    let mut path_alias_records = vec![
        schema_value(
            "RepositoryPathAliasRecordV1",
            vec![
                CborValue::Bytes(path_alias_a.to_vec()),
                CborValue::Bytes(b"Hegel Machine/a.md".to_vec()),
                vector_git_commit(),
            ],
        )?,
        schema_value(
            "RepositoryPathAliasRecordV1",
            vec![
                CborValue::Bytes(path_alias_b.to_vec()),
                CborValue::Bytes(b"Hegel Machine/b.md".to_vec()),
                vector_git_commit(),
            ],
        )?,
    ];
    if path_alias_a > path_alias_b {
        path_alias_records.reverse();
    }
    record_trees.push(positive_record_tree_vector(
        "RepositoryPathAliasRecordV1",
        "RepositoryPathAliasRecordV1",
        path_alias_records,
    )?);

    let source_file_records = vec![
        schema_value(
            "SourceFileRecordV1",
            vec![
                CborValue::Bytes(path_alias_a.to_vec()),
                CborValue::Bytes(b"Hegel Machine/a.md".to_vec()),
                CborValue::Unsigned(1),
                CborValue::Bytes(blob_a.to_vec()),
                CborValue::Unsigned(0o100644),
                CborValue::Unsigned(123),
            ],
        )?,
        schema_value(
            "SourceFileRecordV1",
            vec![
                CborValue::Bytes(path_alias_b.to_vec()),
                CborValue::Bytes(b"Hegel Machine/b.md".to_vec()),
                CborValue::Unsigned(1),
                CborValue::Bytes(blob_b.to_vec()),
                CborValue::Unsigned(0o100644),
                CborValue::Unsigned(456),
            ],
        )?,
    ];
    record_trees.push(positive_record_tree_vector(
        "SourceFileRecordV1",
        "SourceFileRecordV1",
        source_file_records,
    )?);

    let dependency_records = vec![
        schema_value(
            "DependencyLockRecordV1",
            vec![
                CborValue::Unsigned(1),
                root_value("dependency_package_python")?,
                root_value("dependency_version_python")?,
                root_value("dependency_source_python")?,
                root_value("dependency_lock_python")?,
            ],
        )?,
        schema_value(
            "DependencyLockRecordV1",
            vec![
                CborValue::Unsigned(2),
                root_value("dependency_package_rust")?,
                root_value("dependency_version_rust")?,
                root_value("dependency_source_rust")?,
                root_value("dependency_lock_rust")?,
            ],
        )?,
    ];
    record_trees.push(positive_record_tree_vector(
        "DependencyLockRecordV1",
        "DependencyLockRecordV1",
        dependency_records,
    )?);

    let legal_transition_records = vec![
        schema_value(
            "LegalTransitionRowV1",
            vec![
                CborValue::Unsigned(0),
                CborValue::Unsigned(0),
                CborValue::Unsigned(1),
                CborValue::Unsigned(1),
                CborValue::Array(vec![CborValue::Unsigned(1)]),
            ],
        )?,
        schema_value(
            "LegalTransitionRowV1",
            vec![
                CborValue::Unsigned(1),
                CborValue::Unsigned(1),
                CborValue::Unsigned(1),
                CborValue::Unsigned(2),
                CborValue::Array(vec![CborValue::Unsigned(8)]),
            ],
        )?,
    ];
    record_trees.push(positive_record_tree_vector(
        "LegalTransitionRowV1",
        "LegalTransitionRowV1",
        legal_transition_records,
    )?);

    let opaque_records = vec![
        schema_value(
            "OpaqueIdRegistryRecordV1",
            vec![
                CborValue::Unsigned(0),
                CborValue::Unsigned(1),
                CborValue::Bytes(run_id.to_vec()),
                CborValue::Bytes(run_intent_root.to_vec()),
                vector_git_commit(),
                CborValue::Unsigned(ERRATA_VECTOR_BASE_TIMESTAMP),
            ],
        )?,
        schema_value(
            "OpaqueIdRegistryRecordV1",
            vec![
                CborValue::Unsigned(1),
                CborValue::Unsigned(2),
                CborValue::Bytes(ledger_id.to_vec()),
                root_value("ledger_registration_intent_root")?,
                vector_git_commit(),
                CborValue::Unsigned(ERRATA_VECTOR_BASE_TIMESTAMP + 1),
            ],
        )?,
    ];
    let opaque_vector = positive_record_tree_vector(
        "OpaqueIdRegistryRecordV1",
        "OpaqueIdRegistryRecordV1",
        opaque_records.clone(),
    )?;
    let opaque_tree_root = opaque_vector.root.expect("positive tree has root");
    record_trees.push(opaque_vector);

    let audit_bundle_value = schema_value(
        "ParentAbsenceAuditBundleV1",
        vec![
            vector_git_commit(),
            CborValue::Bytes(audited_path_root.to_vec()),
            CborValue::Bytes(history_root.to_vec()),
            CborValue::Bytes(legacy_source_root.to_vec()),
            CborValue::Unsigned(2),
            CborValue::Unsigned(2),
            CborValue::Unsigned(2),
        ],
    )?;
    let audit_bundle_vector = positive_object_vector(
        "ParentAbsenceAuditBundleV1",
        "ParentAbsenceAuditBundleV1",
        audit_bundle_value,
    )?;
    let audit_bundle_root = audit_bundle_vector
        .candidate_root
        .expect("positive object vector has a root");
    objects.push(audit_bundle_vector);

    let parent_attestation_value = schema_value(
        "ParentManifestAbsenceAttestationV2",
        vec![
            root_value("parent_dsl_version_digest")?,
            root_value("parent_freeze_version_digest")?,
            vector_git_commit(),
            CborValue::Bytes(audit_bundle_root.to_vec()),
            CborValue::Unsigned(0b1111),
            id_value("parent_absence_auditor_key_id")?,
            CborValue::Unsigned(ERRATA_VECTOR_BASE_TIMESTAMP),
        ],
    )?;
    let parent_attestation_vector = positive_object_vector(
        "ParentManifestAbsenceAttestationV2",
        "ParentManifestAbsenceAttestationV2",
        parent_attestation_value.clone(),
    )?;
    let parent_attestation_root = parent_attestation_vector
        .candidate_root
        .expect("positive object vector has a root");
    objects.push(parent_attestation_vector);

    let actor_manifest_roots = (1..=4)
        .map(|purpose| {
            Ok((
                purpose as u16,
                errata_vector_root(&format!("actor_key_manifest_purpose_{purpose}"))?,
            ))
        })
        .collect::<Result<Vec<_>, FormalWireError>>()?;
    let actor_trust_value = schema_value(
        "ActorTrustGenesisV1",
        vec![
            id_value("actor_trust_genesis_id")?,
            CborValue::Array(
                actor_manifest_roots
                    .iter()
                    .map(|(purpose, root)| {
                        CborValue::Array(vec![
                            CborValue::Unsigned(u64::from(*purpose)),
                            CborValue::Bytes(root.to_vec()),
                        ])
                    })
                    .collect(),
            ),
            root_value("replacement_policy_root")?,
            CborValue::Unsigned(ERRATA_VECTOR_BASE_TIMESTAMP),
            vector_git_commit(),
        ],
    )?;
    let actor_trust_vector = positive_object_vector(
        "ActorTrustGenesisV1",
        "ActorTrustGenesisV1",
        actor_trust_value.clone(),
    )?;
    let actor_trust_root = actor_trust_vector
        .candidate_root
        .expect("positive object vector has a root");
    objects.push(actor_trust_vector);

    let singleton_opaque_root =
        errata_record_tree_root("OpaqueIdRegistryRecordV1", &opaque_records[..1])?;
    let opaque_snapshot_value = schema_value(
        "OpaqueIdRegistrySnapshotV1",
        vec![
            CborValue::Null,
            CborValue::Bytes(singleton_opaque_root.to_vec()),
            CborValue::Unsigned(1),
            CborValue::Bytes(singleton_opaque_root.to_vec()),
            vector_git_commit(),
        ],
    )?;
    let opaque_snapshot_vector = positive_object_vector(
        "OpaqueIdRegistrySnapshotV1.genesis",
        "OpaqueIdRegistrySnapshotV1",
        opaque_snapshot_value.clone(),
    )?;
    let opaque_snapshot_root = opaque_snapshot_vector
        .candidate_root
        .expect("positive object vector has a root");
    objects.push(opaque_snapshot_vector);
    validate_opaque_snapshot_append(&opaque_snapshot_value, None, 0, &opaque_records[..1])?;

    let envelope_value = schema_value(
        "SignedManifestEnvelopeV1",
        vec![
            CborValue::Unsigned(PARENT_ABSENCE_ATTESTATION_V2_TAG),
            CborValue::Bytes(parent_attestation_root.to_vec()),
            CborValue::Unsigned(ERRATA_VECTOR_BASE_TIMESTAMP),
            CborValue::Unsigned(0),
            CborValue::Array(vec![CborValue::Array(vec![
                id_value("parent_absence_auditor_key_id")?,
                CborValue::Bytes(errata_vector_signature("parent_absence_auditor")?.to_vec()),
            ])]),
        ],
    )?;
    objects.push(positive_object_vector(
        "SignedManifestEnvelopeV1.parent_absence",
        "SignedManifestEnvelopeV1",
        envelope_value,
    )?);

    let candidate_root_labels = [
        "child_dsl_spec_root",
        "child_freeze_root",
        "approval_manifest_root",
        "shrink_transition_root",
        "operator_semantics_root",
        "identifier_registry_root",
        "canonical_ast_schema_root",
        "canonical_cbor_profile_root",
        "diagnostic_formal_bridge_root",
        "outside_target_binding_manifest_root",
        "null_control_binding_manifest_root",
        "split_binding_manifest_root",
        "custodian_binding_manifest_root",
        "seed_continuity_manifest_root",
        "custodian_attestation_bundle_root",
        "parent_absence_attestation_root",
        "hidden_access_ledger_genesis_root",
        "hidden_access_ledger_head_root",
        "opaque_id_registry_snapshot_root",
        "actor_trust_genesis_root",
        "outside_target_universe_root",
        "outside_target_truth_root",
        "null_control_universe_root",
        "null_control_truth_root",
        "outside_discovery_split_root",
        "outside_validation_split_root",
        "outside_sealed_split_root",
        "null_discovery_split_root",
        "null_validation_split_root",
        "null_sealed_split_root",
    ];
    let mut candidate_body = vec![CborValue::Bytes(run_id.to_vec())];
    for label in candidate_root_labels {
        let value = match label {
            "canonical_ast_schema_root" => objects
                .iter()
                .find(|vector| vector.name == "CanonicalAstProfileSpecV1")
                .and_then(|vector| vector.candidate_root)
                .expect("profile vector root"),
            "canonical_cbor_profile_root" => objects
                .iter()
                .find(|vector| vector.name == "CanonicalCborProfileSpecV1")
                .and_then(|vector| vector.candidate_root)
                .expect("profile vector root"),
            "parent_absence_attestation_root" => parent_attestation_root,
            "hidden_access_ledger_head_root" => {
                errata_vector_root("hidden_access_ledger_genesis_root")?
            }
            "opaque_id_registry_snapshot_root" => opaque_snapshot_root,
            "actor_trust_genesis_root" => actor_trust_root,
            _ => errata_vector_root(label)?,
        };
        candidate_body.push(CborValue::Bytes(value.to_vec()));
    }
    candidate_body.extend([
        CborValue::Unsigned(50_000),
        CborValue::Unsigned(1_000_000),
        CborValue::Unsigned(4_096),
        CborValue::Unsigned(1),
    ]);
    for label in [
        "python_implementation_binding_root",
        "rust_implementation_binding_root",
        "traversal_contract_root",
        "bucket_accounting_contract_root",
        "program_archive_contract_root",
        "output_archive_contract_root",
        "state_machine_contract_root",
    ] {
        candidate_body.push(root_value(label)?);
    }
    candidate_body.extend([
        CborValue::Unsigned(ERRATA_VECTOR_BASE_TIMESTAMP),
        vector_git_commit(),
    ]);
    let execution_candidate_value = schema_value("M3ExecutionCandidateV1", candidate_body)?;
    let execution_candidate_vector = positive_object_vector(
        "M3ExecutionCandidateV1",
        "M3ExecutionCandidateV1",
        execution_candidate_value,
    )?;
    let execution_candidate_root = execution_candidate_vector
        .candidate_root
        .expect("positive object vector has a root");
    objects.push(execution_candidate_vector);

    let bridge_statement_value = schema_value(
        "BridgeReplayStatementV1",
        vec![
            CborValue::Bytes(run_id.to_vec()),
            root_value("diagnostic_formal_bridge_root")?,
            CborValue::Bytes(execution_candidate_root.to_vec()),
            root_value("child_dsl_spec_root")?,
            root_value("child_freeze_root")?,
            CborValue::Bytes(actor_trust_root.to_vec()),
            CborValue::Bytes(opaque_snapshot_root.to_vec()),
        ],
    )?;
    let bridge_statement_vector = positive_object_vector(
        "BridgeReplayStatementV1",
        "BridgeReplayStatementV1",
        bridge_statement_value,
    )?;
    let bridge_statement_root = bridge_statement_vector
        .candidate_root
        .expect("positive object vector has a root");
    objects.push(bridge_statement_vector);

    objects.push(ErrataObjectVector {
        name: "BridgeAttestationSignaturePreimageV1".to_owned(),
        schema_name: "raw-signature-preimage".to_owned(),
        tag: 0,
        status: PASS_CANDIDATE,
        bytes: Some(bridge_attestation_signature_preimage(
            &bridge_statement_root,
            2,
            7,
        )?),
        candidate_root: None,
        error_code: None,
    });

    let execution_manifest_value = schema_value(
        "M3ExecutionManifestV2",
        vec![
            CborValue::Bytes(run_id.to_vec()),
            CborValue::Bytes(execution_candidate_root.to_vec()),
            CborValue::Bytes(bridge_statement_root.to_vec()),
            root_value("bridge_attestation_bundle_root")?,
            CborValue::Bytes(actor_trust_root.to_vec()),
            CborValue::Bytes(opaque_snapshot_root.to_vec()),
            CborValue::Unsigned(ERRATA_VECTOR_BASE_TIMESTAMP),
            vector_git_commit(),
        ],
    )?;
    let execution_manifest_vector = positive_object_vector(
        "M3ExecutionManifestV2",
        "M3ExecutionManifestV2",
        execution_manifest_value,
    )?;
    let execution_manifest_root = execution_manifest_vector
        .candidate_root
        .expect("positive object vector has a root");
    objects.push(execution_manifest_vector);

    let mut genesis_body = vec![
        CborValue::Bytes(run_id.to_vec()),
        CborValue::Bytes(execution_manifest_root.to_vec()),
        CborValue::Unsigned(0),
    ];
    genesis_body.extend((0..15).map(|_| CborValue::Null));
    genesis_body.extend([
        CborValue::Unsigned(ERRATA_VECTOR_BASE_TIMESTAMP),
        vector_git_commit(),
    ]);
    let m3_genesis_value = schema_value("M3RunGenesisV1", genesis_body)?;
    objects.push(positive_object_vector(
        "M3RunGenesisV1",
        "M3RunGenesisV1",
        m3_genesis_value.clone(),
    )?);

    let m3_start_value = schema_value(
        "M3RunStateRecordV1",
        vec![
            CborValue::Bytes(run_id.to_vec()),
            CborValue::Unsigned(0),
            CborValue::Null,
            CborValue::Unsigned(0),
            CborValue::Unsigned(0),
            CborValue::Unsigned(1),
            CborValue::Unsigned(1),
            CborValue::Unsigned(1),
            CborValue::Bytes(execution_manifest_root.to_vec()),
            CborValue::Null,
            CborValue::Unsigned(ERRATA_VECTOR_BASE_TIMESTAMP),
        ],
    )?;
    validate_m3_start_record(&m3_start_value)?;
    objects.push(positive_object_vector(
        "M3RunStateRecordV1.synthetic_start_shape",
        "M3RunStateRecordV1",
        m3_start_value.clone(),
    )?);

    let dual_agreement_value = schema_value(
        "M3DualReplayAgreementV1",
        vec![
            CborValue::Bytes(run_id.to_vec()),
            CborValue::Bytes(execution_manifest_root.to_vec()),
            root_value("python_enumeration_receipt_root")?,
            root_value("rust_enumeration_receipt_root")?,
            CborValue::Unsigned(0),
            CborValue::Null,
            CborValue::Null,
            CborValue::Null,
            CborValue::Null,
            CborValue::Null,
            CborValue::Null,
            CborValue::Array(vec![]),
            CborValue::Bool(true),
            CborValue::Unsigned(0),
            CborValue::Null,
            CborValue::Unsigned(ERRATA_VECTOR_BASE_TIMESTAMP),
        ],
    )?;
    objects.push(positive_object_vector(
        "M3DualReplayAgreementV1",
        "M3DualReplayAgreementV1",
        dual_agreement_value,
    )?);

    let mut bad_document_bundle = bundle_value.clone();
    if let CborValue::Array(values) = &mut bad_document_bundle {
        if let CborValue::Array(entries) = &mut values[4] {
            entries.swap(0, 1);
        }
    }
    guard_errors.push(expected_guard_error(
        "document_roles_wrong_order",
        REJECT_M25_FIELD_VALUE,
        validate_errata_object("NormativeDocumentBundleV1", &bad_document_bundle),
    )?);

    let mut bad_static = sink_static_value;
    if let CborValue::Array(values) = &mut bad_static {
        values[7] = CborValue::Array(vec![CborValue::Unsigned(1); 4]);
    }
    guard_errors.push(expected_guard_error(
        "sink_static_orientation_mismatch",
        REJECT_M25_FIELD_VALUE,
        validate_errata_object("StaticRoleMetadataV1", &bad_static),
    )?);

    guard_errors.push(expected_guard_error(
        "actor_trust_missing_purpose",
        FAIL_ACTOR_TRUST_PURPOSE_SET,
        validate_actor_trust_manifest_roots(&actor_trust_value, &actor_manifest_roots[..3]),
    )?);

    let mut reused_actor_root = actor_trust_value.clone();
    if let CborValue::Array(values) = &mut reused_actor_root {
        if let CborValue::Array(entries) = &mut values[4] {
            let first_root = match &entries[0] {
                CborValue::Array(pair) => pair[1].clone(),
                _ => unreachable!("generated actor entry is an array"),
            };
            if let CborValue::Array(pair) = &mut entries[1] {
                pair[1] = first_root;
            }
        }
    }
    guard_errors.push(expected_guard_error(
        "actor_trust_reused_manifest_root",
        FAIL_ACTOR_TRUST_KEY_ROOT_MISMATCH,
        validate_actor_trust_manifest_roots(&reused_actor_root, &actor_manifest_roots),
    )?);
    let mut actor_key_material = (1_u16..=4)
        .map(|purpose| {
            Ok((
                purpose,
                errata_vector_id(&format!("actor_key_id_{purpose}"))?,
                errata_vector_key(&format!("actor_public_key_{purpose}"))?,
            ))
        })
        .collect::<Result<Vec<_>, FormalWireError>>()?;
    actor_key_material[1].2 = actor_key_material[0].2;
    guard_errors.push(expected_guard_error(
        "actor_public_key_reused_across_purposes",
        FAIL_ACTOR_KEY_CROSS_PURPOSE_REUSE,
        validate_actor_key_material(&actor_key_material),
    )?);

    let mut bad_attestation = parent_attestation_value;
    if let CborValue::Array(values) = &mut bad_attestation {
        values[7] = CborValue::Unsigned(7);
    }
    guard_errors.push(expected_guard_error(
        "parent_absence_bitmask_not_15",
        REJECT_M25_FIELD_VALUE,
        validate_errata_object("ParentManifestAbsenceAttestationV2", &bad_attestation),
    )?);

    let mut prepopulated_genesis = m3_genesis_value;
    if let CborValue::Array(values) = &mut prepopulated_genesis {
        values[6] = root_value("forbidden_prepopulated_output")?;
    }
    guard_errors.push(expected_guard_error(
        "m3_genesis_output_prepopulated",
        FAIL_M3_OUTPUT_ROOT_PREPOPULATED,
        validate_errata_object("M3RunGenesisV1", &prepopulated_genesis),
    )?);

    let mut bad_start = m3_start_value;
    if let CborValue::Array(values) = &mut bad_start {
        values[10] = CborValue::Unsigned(2);
    }
    guard_errors.push(expected_guard_error(
        "m3_start_wrong_reason",
        FAIL_ILLEGAL_M3_STATE_TRANSITION,
        validate_m3_start_record(&bad_start),
    )?);

    guard_errors.push(expected_guard_error(
        "bridge_attester_purpose_order",
        FAIL_BRIDGE_ATTESTATION_PURPOSE_SET,
        validate_bridge_attester_purposes(&[1, 2, 2]),
    )?);
    guard_errors.push(expected_guard_error(
        "external_attestation_missing_auditor",
        FAIL_EXTERNAL_INPUT_ATTESTATION_COVERAGE,
        validate_external_input_attestation_entries(&[
            (1, 0x3103),
            (1, 0x3105),
            (1, 0x3106),
            (1, 0x3108),
        ]),
    )?);
    let witness_a = errata_vector_root("null_witness_a")?;
    let witness_b = errata_vector_root("null_witness_b")?;
    guard_errors.push(expected_guard_error(
        "null_witness_mismatch",
        FAIL_NULL_WITNESS_BINDING_MISMATCH,
        validate_null_witness_binding(2, Some(&witness_a), Some(&witness_b)),
    )?);

    let mut wrong_order_paths = audited_path_records;
    wrong_order_paths.reverse();
    guard_errors.push(expected_guard_error(
        "audited_path_wrong_order",
        REJECT_M25_RECORD_ORDER,
        errata_record_tree_root("AuditedPathBlobRecordV1", &wrong_order_paths),
    )?);

    let mut sequence_gap = opaque_records.clone();
    if let CborValue::Array(values) = &mut sequence_gap[1] {
        values[3] = CborValue::Unsigned(2);
    }
    guard_errors.push(expected_guard_error(
        "opaque_registry_sequence_gap",
        FAIL_OPAQUE_ID_REGISTRY_SEQUENCE,
        errata_record_tree_root("OpaqueIdRegistryRecordV1", &sequence_gap),
    )?);

    let mut bad_snapshot = opaque_snapshot_value;
    if let CborValue::Array(values) = &mut bad_snapshot {
        values[4] = root_value("wrong_opaque_registry_tree_root")?;
    }
    guard_errors.push(expected_guard_error(
        "opaque_snapshot_tree_root_mismatch",
        FAIL_OPAQUE_ID_REGISTRY_SNAPSHOT,
        validate_opaque_snapshot_append(&bad_snapshot, None, 0, &opaque_records[..1]),
    )?);

    let mut duplicate_opaque_id = opaque_records;
    let first_id = match &duplicate_opaque_id[0] {
        CborValue::Array(values) => values[5].clone(),
        _ => unreachable!("generated opaque record is an array"),
    };
    if let CborValue::Array(values) = &mut duplicate_opaque_id[1] {
        values[5] = first_id;
    }
    guard_errors.push(expected_guard_error(
        "opaque_registry_raw_id_reuse_across_kinds",
        FAIL_OPAQUE_ID_ALREADY_USED,
        errata_record_tree_root("OpaqueIdRegistryRecordV1", &duplicate_opaque_id),
    )?);

    objects.sort_by(|left, right| left.name.cmp(&right.name));
    record_trees.sort_by(|left, right| left.name.cmp(&right.name));
    guard_errors.sort_by(|left, right| left.vector_id.cmp(&right.vector_id));

    if opaque_tree_root == [0; 32] {
        return Err(FormalWireError::new(
            REJECT_M25_FIELD_VALUE,
            "opaque tree root unexpectedly all zero",
        ));
    }
    Ok(ErrataVectorReport {
        machine_freeze_id: MACHINE_FREEZE_ID,
        vector_schema: ERRATA_VECTOR_SCHEMA,
        objects,
        record_trees,
        guard_errors,
    })
}

fn is_uint_bit(value: &CborValue) -> bool {
    matches!(value, CborValue::Unsigned(bit) if *bit <= 1)
}

/// Strictly validate one decoded ``OddInputV1`` object.
pub fn validate_odd_input_v1(value: &CborValue) -> Result<(), FormalWireError> {
    let values = array(value)?;
    if values.len() != 5 || !exact_prefix(values, ODD_INPUT_TAG, ODD_INPUT_SCHEMA_ID) {
        return Err(FormalWireError::new(
            REJECT_TYPED_INPUT_PREFIX,
            "OddInputV1 prefix or arity mismatch",
        ));
    }
    let set_size = match &values[3] {
        CborValue::Unsigned(value) if (5..=8).contains(value) => *value as usize,
        _ => {
            return Err(FormalWireError::new(
                REJECT_ODD_SET_SIZE,
                "odd set_size must be one of 5, 6, 7, 8",
            ))
        }
    };
    let bits = match &values[4] {
        CborValue::Array(bits) => bits,
        _ => {
            return Err(FormalWireError::new(
                REJECT_ODD_BIT_COUNT,
                "odd bits must be an array",
            ))
        }
    };
    if bits.len() != set_size {
        return Err(FormalWireError::new(
            REJECT_ODD_BIT_COUNT,
            "odd bit count must equal set_size",
        ));
    }
    if bits.iter().any(|bit| !is_uint_bit(bit)) {
        return Err(FormalWireError::new(
            REJECT_ODD_BIT_TYPE,
            "odd bits must be CBOR uint 0 or 1",
        ));
    }
    Ok(())
}

/// Construct one strict ``OddInputV1`` object.
pub fn odd_input_v1(set_size: u64, bits: &[u8]) -> Result<CborValue, FormalWireError> {
    let value = CborValue::Array(vec![
        unsigned(1),
        unsigned(ODD_INPUT_TAG),
        bytes_value(ODD_INPUT_SCHEMA_ID),
        unsigned(set_size),
        CborValue::Array(
            bits.iter()
                .copied()
                .map(|bit| unsigned(bit as u64))
                .collect(),
        ),
    ]);
    validate_odd_input_v1(&value)?;
    Ok(value)
}

/// Strictly validate one decoded ``SinkInputV1`` object.
pub fn validate_sink_input_v1(value: &CborValue) -> Result<(), FormalWireError> {
    let values = array(value)?;
    if values.len() != 7 || !exact_prefix(values, SINK_INPUT_TAG, SINK_INPUT_SCHEMA_ID) {
        return Err(FormalWireError::new(
            REJECT_TYPED_INPUT_PREFIX,
            "SinkInputV1 prefix or arity mismatch",
        ));
    }
    let mut fields = [0_i64; 4];
    for (target, value) in fields.iter_mut().zip(&values[3..]) {
        match value {
            CborValue::Unsigned(value @ 0..=4) => *target = *value as i64,
            _ => {
                return Err(FormalWireError::new(
                    REJECT_SINK_VALUE,
                    "sink a, b, c, d must be CBOR uint in [0, 4]",
                ))
            }
        }
    }
    if fields[3] != fields[0] + fields[1] - fields[2] {
        return Err(FormalWireError::new(
            REJECT_SINK_BALANCE,
            "sink input must satisfy d = a + b - c",
        ));
    }
    Ok(())
}

/// Construct one strict ``SinkInputV1`` object.
pub fn sink_input_v1(a: u64, b: u64, c: u64, d: u64) -> Result<CborValue, FormalWireError> {
    let value = CborValue::Array(vec![
        unsigned(1),
        unsigned(SINK_INPUT_TAG),
        bytes_value(SINK_INPUT_SCHEMA_ID),
        unsigned(a),
        unsigned(b),
        unsigned(c),
        unsigned(d),
    ]);
    validate_sink_input_v1(&value)?;
    Ok(value)
}

/// Validate a typed input and return its frozen ``InputSignatureId``.
pub fn typed_input_signature_id(value: &CborValue) -> Result<u16, FormalWireError> {
    let values = array(value)?;
    match values.get(1) {
        Some(CborValue::Unsigned(ODD_INPUT_TAG)) => {
            validate_odd_input_v1(value)?;
            Ok(ODD_INPUT_SIGNATURE_ID)
        }
        Some(CborValue::Unsigned(SINK_INPUT_TAG)) => {
            validate_sink_input_v1(value)?;
            Ok(SINK_INPUT_SIGNATURE_ID)
        }
        _ => Err(FormalWireError::new(
            REJECT_TYPED_INPUT_PREFIX,
            "unknown typed input tag",
        )),
    }
}

/// ContentHash for one already typed odd/sink input.
pub fn canonical_input_hash_v1(value: &CborValue) -> Result<[u8; 32], FormalWireError> {
    typed_input_signature_id(value)?;
    content_hash(CANONICAL_INPUT_DOMAIN, value)
}

/// Construct one typed bounded-universe row.
pub fn bounded_universe_row_v1(
    universe_index: u64,
    input_signature_id: u16,
    canonical_input: &CborValue,
) -> Result<CborValue, FormalWireError> {
    let actual_signature = typed_input_signature_id(canonical_input)?;
    if input_signature_id != actual_signature {
        return Err(FormalWireError::new(
            FAIL_INPUT_SIGNATURE_MISMATCH,
            "row InputSignatureId does not match the canonical input tag",
        ));
    }
    Ok(CborValue::Array(vec![
        unsigned(1),
        unsigned(UNIVERSE_ROW_TAG),
        bytes_value(UNIVERSE_ROW_SCHEMA_ID),
        unsigned(universe_index),
        unsigned(input_signature_id as u64),
        canonical_input.clone(),
    ]))
}

/// Construct one typed truth row; bool and integers outside ``0/1`` fail.
pub fn target_truth_row_v1(
    universe_index: u64,
    canonical_input_hash: &[u8],
    target_output: &CborValue,
) -> Result<CborValue, FormalWireError> {
    if canonical_input_hash.len() != 32 {
        return Err(FormalWireError::new(
            FAIL_CANONICAL_INPUT_HASH_MISMATCH,
            "canonical_input_hash must be exactly 32 bytes",
        ));
    }
    if !is_uint_bit(target_output) {
        return Err(FormalWireError::new(
            FAIL_TARGET_OUTPUT_TYPE,
            "target_output must be CBOR uint Bit 0 or 1",
        ));
    }
    Ok(CborValue::Array(vec![
        unsigned(1),
        unsigned(TRUTH_ROW_TAG),
        bytes_value(TRUTH_ROW_SCHEMA_ID),
        unsigned(universe_index),
        CborValue::Bytes(canonical_input_hash.to_vec()),
        target_output.clone(),
    ]))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypedRoleRows {
    pub role_name: &'static str,
    pub input_signature_id: u16,
    pub universe_rows: Vec<CborValue>,
    pub truth_rows: Vec<CborValue>,
}

fn parse_universe_row(row: &CborValue) -> Result<(u64, u16, &CborValue), FormalWireError> {
    let values = match row {
        CborValue::Array(values) => values,
        _ => {
            return Err(FormalWireError::new(
                REJECT_UNIVERSE_ROW_SCHEMA,
                "universe row must be an array",
            ))
        }
    };
    if values.len() != 6 || !exact_prefix(values, UNIVERSE_ROW_TAG, UNIVERSE_ROW_SCHEMA_ID) {
        return Err(FormalWireError::new(
            REJECT_UNIVERSE_ROW_SCHEMA,
            "universe row schema mismatch",
        ));
    }
    let index = match &values[3] {
        CborValue::Unsigned(index) => *index,
        _ => {
            return Err(FormalWireError::new(
                REJECT_UNIVERSE_ROW_SCHEMA,
                "universe index must be uint",
            ))
        }
    };
    let signature = match &values[4] {
        CborValue::Unsigned(value) => u16::try_from(*value).map_err(|_| {
            FormalWireError::new(
                FAIL_INPUT_SIGNATURE_MISMATCH,
                "InputSignatureId exceeds uint16",
            )
        })?,
        _ => {
            return Err(FormalWireError::new(
                FAIL_INPUT_SIGNATURE_MISMATCH,
                "InputSignatureId must be uint",
            ))
        }
    };
    let actual_signature = typed_input_signature_id(&values[5])?;
    if signature != actual_signature {
        return Err(FormalWireError::new(
            FAIL_INPUT_SIGNATURE_MISMATCH,
            "universe row signature mismatch",
        ));
    }
    Ok((index, signature, &values[5]))
}

fn parse_truth_row(row: &CborValue) -> Result<(u64, [u8; 32], u8), FormalWireError> {
    let values = match row {
        CborValue::Array(values) => values,
        _ => {
            return Err(FormalWireError::new(
                REJECT_TRUTH_ROW_SCHEMA,
                "truth row must be an array",
            ))
        }
    };
    if values.len() != 6 || !exact_prefix(values, TRUTH_ROW_TAG, TRUTH_ROW_SCHEMA_ID) {
        return Err(FormalWireError::new(
            REJECT_TRUTH_ROW_SCHEMA,
            "truth row schema mismatch",
        ));
    }
    let index = match &values[3] {
        CborValue::Unsigned(index) => *index,
        _ => {
            return Err(FormalWireError::new(
                REJECT_TRUTH_ROW_SCHEMA,
                "truth index must be uint",
            ))
        }
    };
    let input_hash: [u8; 32] = match &values[4] {
        CborValue::Bytes(bytes) => bytes.as_slice().try_into().map_err(|_| {
            FormalWireError::new(
                FAIL_CANONICAL_INPUT_HASH_MISMATCH,
                "truth-row canonical input hash must be 32 bytes",
            )
        })?,
        _ => {
            return Err(FormalWireError::new(
                FAIL_CANONICAL_INPUT_HASH_MISMATCH,
                "truth-row canonical input hash must be bytes",
            ))
        }
    };
    let output = match &values[5] {
        CborValue::Unsigned(output) if *output <= 1 => *output as u8,
        _ => {
            return Err(FormalWireError::new(
                FAIL_TARGET_OUTPUT_TYPE,
                "truth output must be CBOR uint Bit 0 or 1",
            ))
        }
    };
    Ok((index, input_hash, output))
}

fn validate_indices(indices: &[u64]) -> Result<(), FormalWireError> {
    let mut sorted = indices.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    if sorted.len() != indices.len() {
        return Err(FormalWireError::new(
            FAIL_UNIVERSE_INDEX_DUPLICATE,
            "universe indices must be unique",
        ));
    }
    if indices.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(FormalWireError::new(
            FAIL_ROW_ORDERING,
            "rows must be ordered by ascending universe_index",
        ));
    }
    if indices
        .iter()
        .enumerate()
        .any(|(expected, actual)| *actual != expected as u64)
    {
        return Err(FormalWireError::new(
            FAIL_UNIVERSE_INDEX_GAP,
            "universe indices must be contiguous from zero",
        ));
    }
    Ok(())
}

/// Validate index, signature, hash and Bit-output binding for one role.
pub fn validate_typed_role_rows(rows: &TypedRoleRows) -> Result<(), FormalWireError> {
    let universe = rows
        .universe_rows
        .iter()
        .map(parse_universe_row)
        .collect::<Result<Vec<_>, _>>()?;
    let truth = rows
        .truth_rows
        .iter()
        .map(parse_truth_row)
        .collect::<Result<Vec<_>, _>>()?;
    let universe_indices: Vec<u64> = universe.iter().map(|row| row.0).collect();
    let truth_indices: Vec<u64> = truth.iter().map(|row| row.0).collect();
    validate_indices(&universe_indices)?;
    validate_indices(&truth_indices)?;
    if universe_indices != truth_indices {
        return Err(FormalWireError::new(
            FAIL_UNIVERSE_INDEX_GAP,
            "universe and truth indices differ",
        ));
    }
    for ((_, signature, input), (_, input_hash, _)) in universe.iter().zip(&truth) {
        if *signature != rows.input_signature_id {
            return Err(FormalWireError::new(
                FAIL_INPUT_SIGNATURE_MISMATCH,
                "role InputSignatureId mismatch",
            ));
        }
        if canonical_input_hash_v1(input)? != *input_hash {
            return Err(FormalWireError::new(
                FAIL_CANONICAL_INPUT_HASH_MISMATCH,
                "truth row does not bind its canonical input",
            ));
        }
    }
    Ok(())
}

fn build_role_rows(
    role_name: &'static str,
    input_signature_id: u16,
    inputs_and_outputs: Vec<(CborValue, u8)>,
) -> Result<TypedRoleRows, FormalWireError> {
    let mut universe_rows = Vec::with_capacity(inputs_and_outputs.len());
    let mut truth_rows = Vec::with_capacity(inputs_and_outputs.len());
    for (index, (input, output)) in inputs_and_outputs.into_iter().enumerate() {
        let input_hash = canonical_input_hash_v1(&input)?;
        universe_rows.push(bounded_universe_row_v1(
            index as u64,
            input_signature_id,
            &input,
        )?);
        truth_rows.push(target_truth_row_v1(
            index as u64,
            &input_hash,
            &unsigned(output as u64),
        )?);
    }
    let rows = TypedRoleRows {
        role_name,
        input_signature_id,
        universe_rows,
        truth_rows,
    };
    validate_typed_role_rows(&rows)?;
    Ok(rows)
}

/// Independently generate all 480 odd-role rows in MSB-first numeric order.
pub fn generate_odd_role_rows_v1() -> Result<TypedRoleRows, FormalWireError> {
    let mut inputs = Vec::with_capacity(480);
    for set_size in 5_u64..=8 {
        for numeric_value in 0_u64..(1_u64 << set_size) {
            let bits: Vec<u8> = (0..set_size)
                .map(|offset| ((numeric_value >> (set_size - 1 - offset)) & 1) as u8)
                .collect();
            let output = (bits.iter().copied().map(u16::from).sum::<u16>() % 2) as u8;
            inputs.push((odd_input_v1(set_size, &bits)?, output));
        }
    }
    let rows = build_role_rows("odd", ODD_INPUT_SIGNATURE_ID, inputs)?;
    debug_assert_eq!(rows.universe_rows.len(), 480);
    Ok(rows)
}

/// Independently generate all 85 legal sink-role rows in lexicographic order.
pub fn generate_sink_role_rows_v1() -> Result<TypedRoleRows, FormalWireError> {
    let mut inputs = Vec::with_capacity(85);
    for a in 0_u64..=4 {
        for b in 0_u64..=4 {
            for c in 0_u64..=4 {
                for d in 0_u64..=4 {
                    if d as i64 == a as i64 + b as i64 - c as i64 {
                        inputs.push((sink_input_v1(a, b, c, d)?, 1));
                    }
                }
            }
        }
    }
    let rows = build_role_rows("sink", SINK_INPUT_SIGNATURE_ID, inputs)?;
    debug_assert_eq!(rows.universe_rows.len(), 85);
    Ok(rows)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypedRowSample {
    pub universe_index: u64,
    pub input_cbor: Vec<u8>,
    pub canonical_input_hash: [u8; 32],
    pub universe_row_cbor: Vec<u8>,
    pub universe_leaf_hash: [u8; 32],
    pub truth_row_cbor: Vec<u8>,
    pub truth_leaf_hash: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypedRoleReport {
    pub role_name: &'static str,
    pub input_signature_id: u16,
    pub row_count: usize,
    pub samples: Vec<TypedRowSample>,
    pub universe_two_row_root: [u8; 32],
    pub truth_two_row_root: [u8; 32],
    pub universe_root: [u8; 32],
    pub truth_root: [u8; 32],
}

fn encode_records(records: &[CborValue]) -> Result<Vec<Vec<u8>>, FormalWireError> {
    records.iter().map(encode_canonical_cbor).collect()
}

/// Produce the complete independent typed-row report used by the Rust CLI.
pub fn typed_role_report_v1(rows: &TypedRoleRows) -> Result<TypedRoleReport, FormalWireError> {
    validate_typed_role_rows(rows)?;
    let universe_encoded = encode_records(&rows.universe_rows)?;
    let truth_encoded = encode_records(&rows.truth_rows)?;
    let mut samples = Vec::with_capacity(2);
    for index in 0..2 {
        let (_, _, input) = parse_universe_row(&rows.universe_rows[index])?;
        let input_cbor = encode_canonical_cbor(input)?;
        samples.push(TypedRowSample {
            universe_index: index as u64,
            input_cbor,
            canonical_input_hash: canonical_input_hash_v1(input)?,
            universe_row_cbor: universe_encoded[index].clone(),
            universe_leaf_hash: rfc6962_leaf_hash(&universe_encoded[index]),
            truth_row_cbor: truth_encoded[index].clone(),
            truth_leaf_hash: rfc6962_leaf_hash(&truth_encoded[index]),
        });
    }
    Ok(TypedRoleReport {
        role_name: rows.role_name,
        input_signature_id: rows.input_signature_id,
        row_count: rows.universe_rows.len(),
        samples,
        universe_two_row_root: rfc6962_root(&universe_encoded[..2]),
        truth_two_row_root: rfc6962_root(&truth_encoded[..2]),
        universe_root: rfc6962_root(&universe_encoded),
        truth_root: rfc6962_root(&truth_encoded),
    })
}

pub fn generate_typed_role_report_v1(
    input_signature_id: u16,
) -> Result<TypedRoleReport, FormalWireError> {
    match input_signature_id {
        ODD_INPUT_SIGNATURE_ID => typed_role_report_v1(&generate_odd_role_rows_v1()?),
        SINK_INPUT_SIGNATURE_ID => typed_role_report_v1(&generate_sink_role_rows_v1()?),
        _ => Err(FormalWireError::new(
            FAIL_INPUT_SIGNATURE_MISMATCH,
            "typed-row report role must be InputSignatureId 1 or 2",
        )),
    }
}

fn hmac_sha256(key: &[u8], message_parts: &[&[u8]]) -> [u8; 32] {
    let mut mac = <HmacSha256 as Mac>::new_from_slice(key)
        .expect("HMAC-SHA256 accepts keys of every byte length");
    for part in message_parts {
        mac.update(part);
    }
    mac.finalize().into_bytes().into()
}

/// RFC 5869 HKDF-Extract-SHA256.
pub fn hkdf_extract_sha256(salt: &[u8], input_key_material: &[u8]) -> [u8; 32] {
    hmac_sha256(salt, &[input_key_material])
}

/// RFC 5869 HKDF-Expand-SHA256.
pub fn hkdf_expand_sha256(
    pseudorandom_key: &[u8],
    info: &[u8],
    output_length: usize,
) -> Result<Vec<u8>, FormalWireError> {
    if output_length > 255 * 32 {
        return Err(FormalWireError::new(
            REJECT_HKDF_LENGTH,
            "HKDF-SHA256 output exceeds 255 digest blocks",
        ));
    }
    let blocks = output_length.div_ceil(32);
    let mut output = Vec::with_capacity(output_length);
    let mut previous = Vec::new();
    for block_index in 1..=blocks {
        let counter = [block_index as u8];
        let block = hmac_sha256(pseudorandom_key, &[&previous, info, &counter]);
        output.extend_from_slice(&block);
        previous.clear();
        previous.extend_from_slice(&block);
    }
    output.truncate(output_length);
    Ok(output)
}

/// Derive the frozen 32-byte role key from a 32-byte split master seed.
pub fn derive_split_role_key(master_seed: &[u8; 32], role_id: u16) -> [u8; 32] {
    let prk = hkdf_extract_sha256(SPLIT_HKDF_SALT, master_seed);
    let role_bytes = role_id.to_be_bytes();
    let mut info = Vec::with_capacity(SPLIT_ROLE_INFO_PREFIX.len() + role_bytes.len());
    info.extend_from_slice(SPLIT_ROLE_INFO_PREFIX);
    info.extend_from_slice(&role_bytes);
    let expanded = hkdf_expand_sha256(&prk, &info, 32)
        .expect("32-byte output is always within the HKDF-SHA256 limit");
    expanded.try_into().expect("requested exactly 32 bytes")
}

/// Compute the frozen rank for one `(row, role_id, stratum_id)` tuple.
pub fn split_row_rank(
    role_key: &[u8; 32],
    role_id: u16,
    stratum_id: u16,
    canonical_input_hash: &[u8; 32],
) -> [u8; 32] {
    hmac_sha256(
        role_key,
        &[
            SPLIT_RANK_PREFIX,
            &role_id.to_be_bytes(),
            &stratum_id.to_be_bytes(),
            canonical_input_hash,
        ],
    )
}

/// Compute the public digest commitment to a split master seed.
pub fn split_seed_commitment(master_seed: &[u8; 32]) -> [u8; 32] {
    sha256(&[SPLIT_SEED_COMMITMENT_DOMAIN, &[0], master_seed])
}

pub fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

pub fn hex_decode(value: &str) -> Result<Vec<u8>, FormalWireError> {
    if value.len() & 1 != 0 {
        return Err(FormalWireError::new(
            REJECT_INVALID_LENGTH,
            "hexadecimal input has odd length",
        ));
    }
    let mut output = Vec::with_capacity(value.len() / 2);
    let bytes = value.as_bytes();
    for index in (0..bytes.len()).step_by(2) {
        let high = hex_nibble(bytes[index]).ok_or_else(|| {
            FormalWireError::new(REJECT_NONCANONICAL_CBOR, "invalid hexadecimal input")
        })?;
        let low = hex_nibble(bytes[index + 1]).ok_or_else(|| {
            FormalWireError::new(REJECT_NONCANONICAL_CBOR, "invalid hexadecimal input")
        })?;
        output.push((high << 4) | low);
    }
    Ok(output)
}

fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(hex: &str) -> Vec<u8> {
        hex_decode(hex).expect("valid test vector")
    }

    #[test]
    fn deterministic_cbor_known_vector_round_trips() {
        let value = CborValue::Array(vec![
            CborValue::Unsigned(1),
            CborValue::Unsigned(0x3103),
            CborValue::Bytes(b"schema/1".to_vec()),
            CborValue::Negative(0),
            CborValue::Bool(true),
            CborValue::Null,
        ]);
        let encoded = encode_canonical_cbor(&value).unwrap();
        assert_eq!(hex_encode(&encoded), "860119310348736368656d612f3120f5f6");
        assert_eq!(decode_strict_cbor(&encoded).unwrap(), value);
    }

    #[test]
    fn decoder_rejects_every_forbidden_cbor_class() {
        let cases = [
            ("", REJECT_TRUNCATED_CBOR),        // empty input
            ("42aa", REJECT_TRUNCATED_CBOR),    // truncated byte string
            ("1817", REJECT_NONCANONICAL_CBOR), // non-shortest 23
            ("6161", REJECT_CBOR_TEXT),         // text
            ("a0", REJECT_CBOR_MAP),            // map
            ("c000", REJECT_CBOR_TAG),          // tag
            ("f90000", REJECT_CBOR_FLOAT),      // half float
            ("9fff", REJECT_INDEFINITE_CBOR),   // indefinite array
            ("0000", REJECT_TRAILING_CBOR),     // trailing item
            ("f7", REJECT_CBOR_UNDEFINED),      // undefined
            ("1c", REJECT_RESERVED_CBOR),       // reserved additional info
            ("f818", REJECT_CBOR_SIMPLE),       // unapproved simple value
            ("82a0", REJECT_CBOR_MAP),          // first child precedes missing child
        ];
        for (encoded, expected_code) in cases {
            let error = decode_strict_cbor(&bytes(encoded)).unwrap_err();
            assert_eq!(error.code, expected_code, "vector {encoded}");
        }
    }

    #[test]
    fn decoder_rejects_nonshortest_container_lengths() {
        for encoded in ["5800", "9800"] {
            let error = decode_strict_cbor(&bytes(encoded)).unwrap_err();
            assert_eq!(error.code, REJECT_NONCANONICAL_CBOR);
        }
    }

    #[test]
    fn content_hash_has_stable_known_answer() {
        let object = CborValue::Array(vec![CborValue::Unsigned(1), CborValue::Bytes(vec![0])]);
        assert_eq!(
            hex_encode(&content_hash("HEGEL/EXAMPLE/V1", &object).unwrap()),
            "7a10e8d1625e1e2723d78c6142e7f6af1c9ed47b477b1f8251508a796edf9e7b"
        );
        for domain in ["", "HEGEL/INVALID\0DOMAIN"] {
            assert_eq!(
                content_hash(domain, &object).unwrap_err().code,
                REJECT_HASH_DOMAIN
            );
        }
    }

    #[test]
    fn rfc6962_known_answers_include_empty_and_unbalanced_tree() {
        assert_eq!(
            hex_encode(&rfc6962_root(&[])),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        let leaves = vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()];
        assert_eq!(
            hex_encode(&rfc6962_root(&leaves)),
            "36642e73c2540ab121e3a6bf9545b0a24982cd830eb13d3cd19de3ce6c021ec1"
        );
    }

    #[test]
    fn formal_record_root_rejects_noncanonical_leaf_bytes() {
        let records = vec![bytes("820100"), bytes("1817")];
        assert_eq!(
            rfc6962_canonical_record_root(&records).unwrap_err().code,
            REJECT_NONCANONICAL_CBOR
        );
        assert!(rfc6962_canonical_record_root(&records[..1]).is_ok());
    }

    #[test]
    fn hkdf_matches_rfc5869_sha256_case_1() {
        let ikm = vec![0x0b; 22];
        let salt = bytes("000102030405060708090a0b0c");
        let info = bytes("f0f1f2f3f4f5f6f7f8f9");
        let prk = hkdf_extract_sha256(&salt, &ikm);
        assert_eq!(
            hex_encode(&prk),
            "077709362c2e32df0ddc3f0dc47bba6390b6c73bb50f9c3122ec844ad7c2b3e5"
        );
        let okm = hkdf_expand_sha256(&prk, &info, 42).unwrap();
        assert_eq!(
            hex_encode(&okm),
            "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865"
        );
    }

    #[test]
    fn frozen_split_pure_functions_have_stable_known_answers() {
        let master_seed: [u8; 32] = (0u8..32).collect::<Vec<_>>().try_into().unwrap();
        let role_key = derive_split_role_key(&master_seed, 0x1234);
        assert_eq!(
            hex_encode(&role_key),
            "63bdfca4c160ef660732b34bb2ed31d74a796aa904eebb68236842b5f1433b79"
        );
        let input_hash = [0xa5; 32];
        assert_eq!(
            hex_encode(&split_row_rank(&role_key, 0x1234, 0x00ff, &input_hash)),
            "ad30d61d2fc75190038c8d11f2012b1e4df3d8b5edd5bec31f43b1fc4119c98b"
        );
        assert_eq!(
            hex_encode(&split_seed_commitment(&master_seed)),
            "3126668b3227a5e6ab711bcaa66f9d573a7e8bf8b1d1c6cabbb07a96ccf566ba"
        );
    }

    #[test]
    fn hkdf_rejects_too_long_output() {
        let error = hkdf_expand_sha256(&[0; 32], b"", 255 * 32 + 1).unwrap_err();
        assert_eq!(error.code, REJECT_HKDF_LENGTH);
    }

    #[test]
    fn id_digest_v1_matches_the_v112_golden_and_rejects_invalid_ids() {
        let machine_id = "hegel-old-dsl-v1.1.0";
        assert_eq!(
            hex_encode(&id_digest_preimage_v1(machine_id).unwrap()),
            "484547454c2f49445f4449474553542f563100686567656c2d6f6c642d64736c2d76312e312e30"
        );
        assert_eq!(
            hex_encode(&id_digest_v1(machine_id).unwrap()),
            "49022ed9fa53522e10dd60ce5da983a4ac0be2d7bc8c7737f6d5ae1dc88c4703"
        );
        assert_eq!(
            id_digest_v1("é").unwrap_err().code,
            REJECT_MACHINE_ID_NON_ASCII
        );
        assert_eq!(
            id_digest_v1(&"a".repeat(257)).unwrap_err().code,
            REJECT_MACHINE_ID_LENGTH
        );
        for invalid in ["", " leading", "bad?character"] {
            assert_eq!(
                id_digest_v1(invalid).unwrap_err().code,
                REJECT_MACHINE_ID_SYNTAX
            );
        }
    }

    #[test]
    fn typed_input_negative_codes_do_not_coerce_bool_or_invalid_uints() {
        assert_eq!(
            odd_input_v1(4, &[0; 4]).unwrap_err().code,
            REJECT_ODD_SET_SIZE
        );
        assert_eq!(
            odd_input_v1(5, &[0; 4]).unwrap_err().code,
            REJECT_ODD_BIT_COUNT
        );
        assert_eq!(
            odd_input_v1(5, &[0, 0, 0, 0, 2]).unwrap_err().code,
            REJECT_ODD_BIT_TYPE
        );
        let bool_bit = CborValue::Array(vec![
            unsigned(1),
            unsigned(ODD_INPUT_TAG),
            bytes_value(ODD_INPUT_SCHEMA_ID),
            unsigned(5),
            CborValue::Array(vec![
                unsigned(0),
                unsigned(0),
                unsigned(0),
                unsigned(0),
                CborValue::Bool(true),
            ]),
        ]);
        assert_eq!(
            validate_odd_input_v1(&bool_bit).unwrap_err().code,
            REJECT_ODD_BIT_TYPE
        );
        assert_eq!(
            sink_input_v1(0, 0, 0, 5).unwrap_err().code,
            REJECT_SINK_VALUE
        );
        assert_eq!(
            sink_input_v1(0, 1, 0, 0).unwrap_err().code,
            REJECT_SINK_BALANCE
        );
    }

    #[test]
    fn odd_and_sink_generators_match_all_v112_golden_roots() {
        let odd = typed_role_report_v1(&generate_odd_role_rows_v1().unwrap()).unwrap();
        assert_eq!(odd.row_count, 480);
        assert_eq!(
            hex_encode(&odd.universe_two_row_root),
            "a10e24853c11986ceec4a7167c8dca3a7587261dbc0fcd5df0dfc9f7604acf24"
        );
        assert_eq!(
            hex_encode(&odd.truth_two_row_root),
            "b6e2a6d9808cb9c0542a0bcf5cd4af398a419e275221733736045fe3de960fd6"
        );
        assert_eq!(
            hex_encode(&odd.universe_root),
            "b7e6eed1174ceee1944bd540cbb0ace4c5c7fc0dffea79dd956a51f7a2410a05"
        );
        assert_eq!(
            hex_encode(&odd.truth_root),
            "f5bbdc26bec62f9966e5ef31eaa800190ed52dedc73ee61545e0f9c122a1a506"
        );
        assert_eq!(
            hex_encode(&odd.samples[0].universe_leaf_hash),
            "82d372f5c01c3cb6acbc296e7499bf66c9d69fa96f01dc214a14399ea40300c3"
        );

        let sink = typed_role_report_v1(&generate_sink_role_rows_v1().unwrap()).unwrap();
        assert_eq!(sink.row_count, 85);
        assert_eq!(
            hex_encode(&sink.universe_two_row_root),
            "2d06e5870c0ea2a67468f814647f8b11b6cd60243ff4c399d7031d99c33a9b13"
        );
        assert_eq!(
            hex_encode(&sink.truth_two_row_root),
            "bac8bb909d6bf86b097c9a97e3656173cadcda3b1c6a8e7184fc5be256118c32"
        );
        assert_eq!(
            hex_encode(&sink.universe_root),
            "1a46f9967ad48df6ec1d9be609de701413ce86171fb0f92495a982bef0f40ff5"
        );
        assert_eq!(
            hex_encode(&sink.truth_root),
            "9c0f5d75ea3c31f6cb1ea9917346a7a3f480ae9ce0ac0cb3bb21aac9d3bd7808"
        );
        assert_eq!(
            hex_encode(&sink.samples[1].truth_leaf_hash),
            "188092325daf0ac152753b9518df3358f30cc112ac0249827202069708bf6591"
        );
    }

    #[test]
    fn typed_rows_reject_signature_output_hash_and_index_failures_exactly() {
        let odd_input = odd_input_v1(5, &[0; 5]).unwrap();
        assert_eq!(
            bounded_universe_row_v1(0, SINK_INPUT_SIGNATURE_ID, &odd_input)
                .unwrap_err()
                .code,
            FAIL_INPUT_SIGNATURE_MISMATCH
        );
        assert_eq!(
            target_truth_row_v1(0, &[0; 32], &CborValue::Bool(true))
                .unwrap_err()
                .code,
            FAIL_TARGET_OUTPUT_TYPE
        );
        assert_eq!(
            target_truth_row_v1(0, &[0; 32], &unsigned(2))
                .unwrap_err()
                .code,
            FAIL_TARGET_OUTPUT_TYPE
        );
        assert_eq!(
            target_truth_row_v1(0, &[0; 31], &unsigned(0))
                .unwrap_err()
                .code,
            FAIL_CANONICAL_INPUT_HASH_MISMATCH
        );

        let source = generate_odd_role_rows_v1().unwrap();
        let mut duplicate = TypedRoleRows {
            role_name: source.role_name,
            input_signature_id: source.input_signature_id,
            universe_rows: source.universe_rows[..3].to_vec(),
            truth_rows: source.truth_rows[..3].to_vec(),
        };
        if let CborValue::Array(fields) = &mut duplicate.universe_rows[1] {
            fields[3] = unsigned(0);
        }
        assert_eq!(
            validate_typed_role_rows(&duplicate).unwrap_err().code,
            FAIL_UNIVERSE_INDEX_DUPLICATE
        );

        let mut gap = TypedRoleRows {
            role_name: source.role_name,
            input_signature_id: source.input_signature_id,
            universe_rows: source.universe_rows[..3].to_vec(),
            truth_rows: source.truth_rows[..3].to_vec(),
        };
        if let CborValue::Array(fields) = &mut gap.universe_rows[1] {
            fields[3] = unsigned(2);
        }
        if let CborValue::Array(fields) = &mut gap.universe_rows[2] {
            fields[3] = unsigned(3);
        }
        assert_eq!(
            validate_typed_role_rows(&gap).unwrap_err().code,
            FAIL_UNIVERSE_INDEX_GAP
        );

        let mut wrong_order = TypedRoleRows {
            role_name: source.role_name,
            input_signature_id: source.input_signature_id,
            universe_rows: source.universe_rows[..3].to_vec(),
            truth_rows: source.truth_rows[..3].to_vec(),
        };
        wrong_order.universe_rows.swap(1, 2);
        assert_eq!(
            validate_typed_role_rows(&wrong_order).unwrap_err().code,
            FAIL_ROW_ORDERING
        );

        let mut wrong_hash = TypedRoleRows {
            role_name: source.role_name,
            input_signature_id: source.input_signature_id,
            universe_rows: source.universe_rows[..3].to_vec(),
            truth_rows: source.truth_rows[..3].to_vec(),
        };
        if let CborValue::Array(fields) = &mut wrong_hash.truth_rows[0] {
            fields[4] = CborValue::Bytes(vec![0xff; 32]);
        }
        assert_eq!(
            validate_typed_role_rows(&wrong_hash).unwrap_err().code,
            FAIL_CANONICAL_INPUT_HASH_MISMATCH
        );
    }

    #[test]
    fn errata_derivation_primitives_have_language_neutral_known_answers() {
        assert_eq!(
            hex_encode(&errata_vector_root("run_id").unwrap()),
            "9373e2614b1ee77547c9c3ebe41d9db6dfc99d170820eac49fa9f910cd996cc8"
        );
        assert_eq!(
            hex_encode(&errata_vector_id("run_id").unwrap()),
            "90423d805c9dec51c2f2972542beb83c"
        );
        assert_eq!(
            hex_encode(&errata_vector_key("actor_public_key_1").unwrap()),
            "92f3ba114b5074996a1250233dd595272049b76cdef8e3bd0b4bfdea0918cebb"
        );
        assert_eq!(
            hex_encode(&errata_vector_signature("parent_absence_auditor").unwrap()),
            concat!(
                "a537c2e69b03c6409038f149e7a44d623f3bc10819725dbe61671bf33fa1d331",
                "96a97d64e0d65d2e347cab3a5bc0728b05887085673df6ac0280b5c3db546401"
            )
        );
        assert_eq!(
            errata_vector_root("非ASCII").unwrap_err().code,
            REJECT_M25_FIELD_VALUE
        );
    }

    #[test]
    fn external_signature_tags_bind_domain_purpose_and_epoch() {
        let root = [0xa5; 32];
        let bridge = external_signature_preimage_for_tag(0x310e, &root, 2, 7).unwrap();
        let mut expected = BRIDGE_ATTESTATION_SIGNATURE_DOMAIN.to_vec();
        expected.push(0);
        expected.extend_from_slice(&root);
        expected.extend_from_slice(&2_u16.to_be_bytes());
        expected.extend_from_slice(&7_u64.to_be_bytes());
        assert_eq!(bridge, expected);

        let custodian = external_signature_preimage_for_tag(0x3103, &root, 1, 9).unwrap();
        assert!(custodian.starts_with(CUSTODIAN_SPLIT_SEED_COMMITMENT_SIGNATURE_DOMAIN.as_bytes()));
        assert!(external_signature_preimage_for_tag(0x3114, &root, 4, 11)
            .unwrap()
            .starts_with(PARENT_ABSENCE_AUDITOR_SIGNATURE_DOMAIN.as_bytes()));
        for (tag, purpose) in [(0x3103, 2), (0x3114, 1), (0xffff, 1)] {
            assert_eq!(
                external_signature_preimage_for_tag(tag, &root, purpose, 0)
                    .unwrap_err()
                    .code,
                REJECT_M25_FIELD_VALUE
            );
        }
    }

    #[test]
    fn errata_report_is_sorted_complete_and_non_authoritative() {
        let report = generate_errata_vector_report_v1().unwrap();
        assert_eq!(report.machine_freeze_id, "hegel-freeze-p2b-p3-v1.1.2");
        assert_eq!(
            report.vector_schema,
            "hegel-phase3-m25-exact-wire-errata-vectors/1"
        );
        assert_eq!(report.objects.len(), 21);
        assert_eq!(report.record_trees.len(), 8);
        assert_eq!(report.guard_errors.len(), 15);
        assert!(ERRATA_SCHEMAS
            .iter()
            .all(|schema| !matches!(schema.root_rule, ErrataRootRule::NormativeGap(_))));
        assert!(report
            .objects
            .windows(2)
            .all(|pair| pair[0].name < pair[1].name));
        assert!(report
            .record_trees
            .windows(2)
            .all(|pair| pair[0].name < pair[1].name));
        assert!(report
            .guard_errors
            .windows(2)
            .all(|pair| pair[0].vector_id < pair[1].vector_id));
        assert!(report
            .objects
            .iter()
            .all(|vector| { vector.status == PASS_CANDIDATE && vector.error_code.is_none() }));
        assert!(report
            .record_trees
            .iter()
            .all(|vector| { vector.status == PASS_CANDIDATE && vector.error_code.is_none() }));

        let expected_guards = [
            (
                "actor_public_key_reused_across_purposes",
                FAIL_ACTOR_KEY_CROSS_PURPOSE_REUSE,
            ),
            ("actor_trust_missing_purpose", FAIL_ACTOR_TRUST_PURPOSE_SET),
            (
                "actor_trust_reused_manifest_root",
                FAIL_ACTOR_TRUST_KEY_ROOT_MISMATCH,
            ),
            ("audited_path_wrong_order", REJECT_M25_RECORD_ORDER),
            (
                "bridge_attester_purpose_order",
                FAIL_BRIDGE_ATTESTATION_PURPOSE_SET,
            ),
            ("document_roles_wrong_order", REJECT_M25_FIELD_VALUE),
            (
                "external_attestation_missing_auditor",
                FAIL_EXTERNAL_INPUT_ATTESTATION_COVERAGE,
            ),
            (
                "m3_genesis_output_prepopulated",
                FAIL_M3_OUTPUT_ROOT_PREPOPULATED,
            ),
            ("m3_start_wrong_reason", FAIL_ILLEGAL_M3_STATE_TRANSITION),
            ("null_witness_mismatch", FAIL_NULL_WITNESS_BINDING_MISMATCH),
            (
                "opaque_registry_raw_id_reuse_across_kinds",
                FAIL_OPAQUE_ID_ALREADY_USED,
            ),
            (
                "opaque_registry_sequence_gap",
                FAIL_OPAQUE_ID_REGISTRY_SEQUENCE,
            ),
            (
                "opaque_snapshot_tree_root_mismatch",
                FAIL_OPAQUE_ID_REGISTRY_SNAPSHOT,
            ),
            ("parent_absence_bitmask_not_15", REJECT_M25_FIELD_VALUE),
            ("sink_static_orientation_mismatch", REJECT_M25_FIELD_VALUE),
        ];
        let actual_guards = report
            .guard_errors
            .iter()
            .map(|guard| (guard.vector_id.as_str(), guard.error_code))
            .collect::<Vec<_>>();
        assert_eq!(actual_guards, expected_guards);

        let candidate = report
            .objects
            .iter()
            .find(|vector| vector.name == "M3ExecutionCandidateV1")
            .unwrap();
        assert_eq!(
            hex_encode(&candidate.candidate_root.unwrap()),
            "586d5321f8f712e127a434baccf2aacb7de73acde4f2d02e4e9b53bbe8d45a58"
        );
        let candidate_value = decode_strict_cbor(candidate.bytes.as_ref().unwrap()).unwrap();
        let fields = array(&candidate_value).unwrap();
        assert_eq!(fields[20], fields[21]);

        let legacy = report
            .record_trees
            .iter()
            .find(|vector| vector.name == "LegacyParentSourceRowV1")
            .unwrap();
        assert_eq!(
            hex_encode(&legacy.root.unwrap()),
            "982a60f88ceee5a08f3f0ab4cb44002308ce4b288de334407e02fdc210bbf3c7"
        );
        let first_legacy = decode_strict_cbor(legacy.first_record_cbor.as_ref().unwrap()).unwrap();
        let fields = array(&first_legacy).unwrap();
        assert_eq!(
            require_exact_bytes(&fields[6], 32, "diagnostic_digest").unwrap(),
            hex_decode("b491c0a9719fb0279fe02798ede026e440c17a539965514145a7818b15387ac3").unwrap()
        );
    }

    #[test]
    fn production_errata_validators_reject_cross_field_semantic_drift() {
        let report = generate_errata_vector_report_v1().unwrap();
        let candidate = report
            .objects
            .iter()
            .find(|vector| vector.name == "M3ExecutionCandidateV1")
            .unwrap();
        let mut bad_candidate = decode_strict_cbor(candidate.bytes.as_ref().unwrap()).unwrap();
        if let CborValue::Array(fields) = &mut bad_candidate {
            fields[21] = CborValue::Bytes([0xff; 32].to_vec());
        }
        assert_eq!(
            validate_errata_object("M3ExecutionCandidateV1", &bad_candidate)
                .unwrap_err()
                .code,
            FAIL_M3_LEDGER_HEAD_NOT_GENESIS
        );

        let legacy = report
            .record_trees
            .iter()
            .find(|vector| vector.name == "LegacyParentSourceRowV1")
            .unwrap();
        let mut bad_legacy =
            decode_strict_cbor(legacy.first_record_cbor.as_ref().unwrap()).unwrap();
        if let CborValue::Array(fields) = &mut bad_legacy {
            fields[6] = CborValue::Bytes([0; 32].to_vec());
        }
        assert_eq!(
            validate_errata_object("LegacyParentSourceRowV1", &bad_legacy)
                .unwrap_err()
                .code,
            REJECT_M25_FIELD_VALUE
        );
    }
}
