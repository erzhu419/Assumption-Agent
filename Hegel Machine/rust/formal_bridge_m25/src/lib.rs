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
    if value.len() % 2 != 0 {
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
}
