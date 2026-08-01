//! Independent Phase-3A M2.5 formal-wire primitives.
//!
//! This module intentionally implements only frozen, schema-neutral primitives.
//! It does not generate seeds or keys and does not define formal object schemas
//! or state-machine enums.

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
}
