//! Target-free deterministic-CBOR and RFC 6962 primitives.
//!
//! Keeping these few wire primitives inside the enumerator prevents its
//! production dependency closure from including the larger target-aware
//! ceremony bridge crate.  The algorithms remain independently checked
//! against the Python wire implementation by the dual qualification receipt.

use sha2::{Digest, Sha256};
use std::fmt;

#[cfg(test)]
const MAX_CBOR_NESTING: usize = 64;

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FormalCoreError(pub &'static str);

impl fmt::Display for FormalCoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.0)
    }
}

impl std::error::Error for FormalCoreError {}

#[allow(dead_code)]
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
    if value <= 23 {
        output.push((major << 5) | value as u8);
    } else if value <= u8::MAX as u64 {
        output.extend_from_slice(&[(major << 5) | 24, value as u8]);
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

fn encode_value(value: &CborValue, output: &mut Vec<u8>) -> Result<(), FormalCoreError> {
    match value {
        CborValue::Unsigned(value) => encode_major_value(0, *value, output),
        CborValue::Negative(value) => encode_major_value(1, *value, output),
        CborValue::Bytes(value) => {
            let length = u64::try_from(value.len())
                .map_err(|_| FormalCoreError("byte string length exceeds u64"))?;
            encode_major_value(2, length, output);
            output.extend_from_slice(value);
        }
        CborValue::Array(values) => {
            let length = u64::try_from(values.len())
                .map_err(|_| FormalCoreError("array length exceeds u64"))?;
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

pub fn encode_canonical_cbor(value: &CborValue) -> Result<Vec<u8>, FormalCoreError> {
    let mut output = Vec::new();
    encode_value(value, &mut output)?;
    Ok(output)
}

#[cfg(test)]
fn read_exact<'a>(
    bytes: &'a [u8],
    cursor: &mut usize,
    length: usize,
) -> Result<&'a [u8], FormalCoreError> {
    let end = cursor
        .checked_add(length)
        .ok_or(FormalCoreError("CBOR length overflow"))?;
    if end > bytes.len() {
        return Err(FormalCoreError("truncated CBOR item"));
    }
    let result = &bytes[*cursor..end];
    *cursor = end;
    Ok(result)
}

#[cfg(test)]
fn read_argument(
    additional: u8,
    bytes: &[u8],
    cursor: &mut usize,
) -> Result<u64, FormalCoreError> {
    let (value, minimum) = match additional {
        0..=23 => return Ok(additional as u64),
        24 => (read_exact(bytes, cursor, 1)?[0] as u64, 24),
        25 => {
            let raw: [u8; 2] = read_exact(bytes, cursor, 2)?
                .try_into()
                .map_err(|_| FormalCoreError("truncated uint16"))?;
            (u16::from_be_bytes(raw) as u64, 0x100)
        }
        26 => {
            let raw: [u8; 4] = read_exact(bytes, cursor, 4)?
                .try_into()
                .map_err(|_| FormalCoreError("truncated uint32"))?;
            (u32::from_be_bytes(raw) as u64, 0x1_0000)
        }
        27 => {
            let raw: [u8; 8] = read_exact(bytes, cursor, 8)?
                .try_into()
                .map_err(|_| FormalCoreError("truncated uint64"))?;
            (u64::from_be_bytes(raw), 0x1_0000_0000)
        }
        31 => return Err(FormalCoreError("indefinite-length CBOR is forbidden")),
        _ => return Err(FormalCoreError("reserved CBOR additional value")),
    };
    if value < minimum {
        return Err(FormalCoreError("CBOR integer is not shortest form"));
    }
    Ok(value)
}

#[cfg(test)]
fn parse_value(
    bytes: &[u8],
    cursor: &mut usize,
    depth: usize,
) -> Result<CborValue, FormalCoreError> {
    if depth > MAX_CBOR_NESTING {
        return Err(FormalCoreError("CBOR nesting exceeds limit"));
    }
    let initial = read_exact(bytes, cursor, 1)?[0];
    let major = initial >> 5;
    let additional = initial & 0x1f;
    match major {
        0 => Ok(CborValue::Unsigned(read_argument(additional, bytes, cursor)?)),
        1 => Ok(CborValue::Negative(read_argument(additional, bytes, cursor)?)),
        2 => {
            let length = usize::try_from(read_argument(additional, bytes, cursor)?)
                .map_err(|_| FormalCoreError("byte string length exceeds usize"))?;
            Ok(CborValue::Bytes(read_exact(bytes, cursor, length)?.to_vec()))
        }
        3 => Err(FormalCoreError("CBOR text is forbidden")),
        4 => {
            let length = usize::try_from(read_argument(additional, bytes, cursor)?)
                .map_err(|_| FormalCoreError("array length exceeds usize"))?;
            let mut values = Vec::with_capacity(length.min(bytes.len().saturating_sub(*cursor)));
            for _ in 0..length {
                values.push(parse_value(bytes, cursor, depth + 1)?);
            }
            Ok(CborValue::Array(values))
        }
        5 => Err(FormalCoreError("CBOR maps are forbidden")),
        6 => Err(FormalCoreError("CBOR tags are forbidden")),
        7 => match additional {
            20 => Ok(CborValue::Bool(false)),
            21 => Ok(CborValue::Bool(true)),
            22 => Ok(CborValue::Null),
            25..=27 => Err(FormalCoreError("CBOR floats are forbidden")),
            31 => Err(FormalCoreError("indefinite CBOR is forbidden")),
            _ => Err(FormalCoreError("CBOR simple value is forbidden")),
        },
        _ => unreachable!("three-bit CBOR major type"),
    }
}

#[cfg(test)]
pub fn decode_strict_cbor(bytes: &[u8]) -> Result<CborValue, FormalCoreError> {
    let mut cursor = 0;
    let value = parse_value(bytes, &mut cursor, 0)?;
    if cursor != bytes.len() {
        return Err(FormalCoreError("trailing CBOR bytes"));
    }
    if encode_canonical_cbor(&value)? != bytes {
        return Err(FormalCoreError("CBOR differs from deterministic re-encoding"));
    }
    Ok(value)
}

fn digest(parts: &[&[u8]]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update(part);
    }
    hasher.finalize().into()
}

fn leaf_hash(leaf: &[u8]) -> [u8; 32] {
    digest(&[&[0], leaf])
}

fn node_hash(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    digest(&[&[1], left, right])
}

fn largest_power_of_two_less_than(value: usize) -> usize {
    let exponent = (usize::BITS - 1 - (value - 1).leading_zeros()) as usize;
    1usize << exponent
}

pub fn rfc6962_root(leaves: &[Vec<u8>]) -> [u8; 32] {
    match leaves.len() {
        0 => digest(&[b""]),
        1 => leaf_hash(&leaves[0]),
        count => {
            let split = largest_power_of_two_less_than(count);
            node_hash(
                &rfc6962_root(&leaves[..split]),
                &rfc6962_root(&leaves[split..]),
            )
        }
    }
}
