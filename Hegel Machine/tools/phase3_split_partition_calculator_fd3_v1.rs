//! Standalone crate-free v1.1.2 split-partition calculator.
//!
//! This is intentionally independent from the Hegel Machine Rust replay.  It
//! regenerates both typed universes, strict canonical CBOR, SHA-256/HMAC/HKDF,
//! quota allocation, SplitAssignmentRowV1 bytes, and RFC6962 roots using only
//! the Rust standard library.  Exactly 32 seed bytes plus EOF arrive on FIFO
//! FD 3.  One length-delimited public implementation-evidence CBOR frame is
//! written on FIFO FD 5.  No derived key, rank, or membership is emitted.

#![deny(unsafe_op_in_unsafe_fn)]

use std::cmp::Ordering as CmpOrdering;
use std::env;
use std::ffi::c_void;
use std::fs::File;
use std::io::{Read, Write};
use std::os::fd::FromRawFd;
use std::os::unix::fs::FileTypeExt;
use std::process::ExitCode;
use std::sync::atomic::{compiler_fence, Ordering};

const SEED_FD: i32 = 3;
const PUBLIC_RESPONSE_FD: i32 = 5;
const SEED_SIZE: usize = 32;
const FAILURE_EXIT_STATUS: u8 = 71;
const F_GETFD: i32 = 1;

const RESPONSE_SCHEMA_ID: &[u8] = b"hegel-phase3-split-calculator-fd3-response/2";
const SEED_COMMITMENT_PREFIX: &[u8] = b"HEGEL/SPLIT_MASTER_SEED_COMMITMENT/V1";
const HKDF_SALT: &[u8] = b"HEGEL/SPLIT/HKDF/SALT/V1";
const ROLE_INFO_PREFIX: &[u8] = b"HEGEL/SPLIT/ROLE/V1";
const RANK_PREFIX: &[u8] = b"HEGEL/SPLIT/RANK/V1";
const CANONICAL_INPUT_DOMAIN: &[u8] = b"HEGEL/CANONICAL_INPUT/V1";

const OUTSIDE_ROLE_ID: u16 = 1;
const NULL_CONTROL_ROLE_ID: u16 = 2;
const DISCOVERY_PARTITION_ID: u8 = 1;
const VALIDATION_PARTITION_ID: u8 = 2;
const SEALED_PARTITION_ID: u8 = 3;

const ODD_INPUT_TAG: u64 = 0x3401;
const SINK_INPUT_TAG: u64 = 0x3402;
const SPLIT_ASSIGNMENT_ROW_TAG: u64 = 0x3203;
const ODD_INPUT_SCHEMA_ID: &[u8] = b"hegel-odd-input/1";
const SINK_INPUT_SCHEMA_ID: &[u8] = b"hegel-sink-input/1";
const SPLIT_ASSIGNMENT_ROW_SCHEMA_ID: &[u8] = b"hegel-split-assignment-row/1";

unsafe extern "C" {
    fn fcntl(fd: i32, command: i32, ...) -> i32;
    fn mlock(address: *const c_void, length: usize) -> i32;
    fn munlock(address: *const c_void, length: usize) -> i32;
}

#[derive(Debug)]
struct CalculatorFailure;

#[derive(Clone, Copy)]
struct Quota {
    stratum: u16,
    universe: usize,
    discovery: usize,
    validation: usize,
    sealed: usize,
}

const ODD_QUOTAS: [Quota; 8] = [
    Quota { stratum: 1, universe: 16, discovery: 6, validation: 3, sealed: 7 },
    Quota { stratum: 2, universe: 16, discovery: 6, validation: 3, sealed: 7 },
    Quota { stratum: 3, universe: 32, discovery: 13, validation: 6, sealed: 13 },
    Quota { stratum: 4, universe: 32, discovery: 13, validation: 6, sealed: 13 },
    Quota { stratum: 5, universe: 64, discovery: 26, validation: 13, sealed: 25 },
    Quota { stratum: 6, universe: 64, discovery: 26, validation: 13, sealed: 25 },
    Quota { stratum: 7, universe: 128, discovery: 51, validation: 26, sealed: 51 },
    Quota { stratum: 8, universe: 128, discovery: 51, validation: 26, sealed: 51 },
];

const SINK_QUOTAS: [Quota; 5] = [
    Quota { stratum: 9, universe: 15, discovery: 7, validation: 4, sealed: 4 },
    Quota { stratum: 10, universe: 18, discovery: 8, validation: 4, sealed: 6 },
    Quota { stratum: 11, universe: 19, discovery: 9, validation: 4, sealed: 6 },
    Quota { stratum: 12, universe: 18, discovery: 8, validation: 4, sealed: 6 },
    Quota { stratum: 13, universe: 15, discovery: 7, validation: 4, sealed: 4 },
];

#[derive(Clone)]
struct InputRow {
    universe_index: usize,
    input_hash: [u8; 32],
    stratum: u16,
}

#[derive(Clone)]
struct RankedRow {
    universe_index: usize,
    input_hash: [u8; 32],
    stratum: u16,
    rank: [u8; 32],
}

#[derive(Clone)]
struct Assignment {
    role: u16,
    universe_index: usize,
    input_hash: [u8; 32],
    stratum: u16,
    partition: u8,
    rank: [u8; 32],
}

struct RoleEvidence {
    role: u16,
    counts: [usize; 3],
    roots: [[u8; 32]; 3],
}

fn take_fifo(fd: i32) -> Result<File, CalculatorFailure> {
    // SAFETY: F_GETFD takes an integer descriptor and dereferences no pointer.
    if unsafe { fcntl(fd, F_GETFD) } < 0 {
        return Err(CalculatorFailure);
    }
    // SAFETY: this single-threaded one-shot process takes each contract FD once.
    let file = unsafe { File::from_raw_fd(fd) };
    let metadata = file.metadata().map_err(|_| CalculatorFailure)?;
    if !metadata.file_type().is_fifo() {
        return Err(CalculatorFailure);
    }
    Ok(file)
}

fn try_mlock(secret: &mut [u8]) -> bool {
    !secret.is_empty() && unsafe { mlock(secret.as_ptr().cast(), secret.len()) == 0 }
}

fn try_munlock(secret: &mut [u8], locked: bool) {
    if locked && !secret.is_empty() {
        let _ = unsafe { munlock(secret.as_ptr().cast(), secret.len()) };
    }
}

fn zeroize(secret: &mut [u8]) {
    compiler_fence(Ordering::SeqCst);
    for byte in secret {
        // SAFETY: each byte is live and exclusively borrowed for the store.
        unsafe { std::ptr::write_volatile(byte, 0) };
    }
    compiler_fence(Ordering::SeqCst);
}

fn read_seed(seed: &mut [u8; SEED_SIZE]) -> Result<(), CalculatorFailure> {
    let mut input = take_fifo(SEED_FD)?;
    let mut offset = 0;
    while offset < seed.len() {
        let count = input.read(&mut seed[offset..]).map_err(|_| CalculatorFailure)?;
        if count == 0 {
            return Err(CalculatorFailure);
        }
        offset += count;
    }
    let mut extra = [0_u8; 1];
    let extra_result = input.read(&mut extra);
    zeroize(&mut extra);
    match extra_result {
        Ok(0) => Ok(()),
        _ => Err(CalculatorFailure),
    }
}

fn push_cbor_head(output: &mut Vec<u8>, major: u8, value: u64) {
    let prefix = major << 5;
    if value <= 23 {
        output.push(prefix | value as u8);
    } else if value <= u8::MAX as u64 {
        output.extend_from_slice(&[prefix | 24, value as u8]);
    } else if value <= u16::MAX as u64 {
        output.push(prefix | 25);
        output.extend_from_slice(&(value as u16).to_be_bytes());
    } else if value <= u32::MAX as u64 {
        output.push(prefix | 26);
        output.extend_from_slice(&(value as u32).to_be_bytes());
    } else {
        output.push(prefix | 27);
        output.extend_from_slice(&value.to_be_bytes());
    }
}

fn push_uint(output: &mut Vec<u8>, value: u64) {
    push_cbor_head(output, 0, value);
}

fn push_bytes(output: &mut Vec<u8>, value: &[u8]) {
    push_cbor_head(output, 2, value.len() as u64);
    output.extend_from_slice(value);
}

fn push_array(output: &mut Vec<u8>, length: usize) {
    push_cbor_head(output, 4, length as u64);
}

fn encode_odd_input(set_size: usize, numeric_value: usize) -> (Vec<u8>, u8) {
    let mut output = Vec::with_capacity(32);
    push_array(&mut output, 5);
    push_uint(&mut output, 1);
    push_uint(&mut output, ODD_INPUT_TAG);
    push_bytes(&mut output, ODD_INPUT_SCHEMA_ID);
    push_uint(&mut output, set_size as u64);
    push_array(&mut output, set_size);
    let mut parity = 0_u8;
    for offset in 0..set_size {
        let bit = ((numeric_value >> (set_size - 1 - offset)) & 1) as u8;
        parity ^= bit;
        push_uint(&mut output, bit as u64);
    }
    (output, parity)
}

fn encode_sink_input(a: u8, b: u8, c: u8, d: u8) -> Vec<u8> {
    let mut output = Vec::with_capacity(32);
    push_array(&mut output, 7);
    push_uint(&mut output, 1);
    push_uint(&mut output, SINK_INPUT_TAG);
    push_bytes(&mut output, SINK_INPUT_SCHEMA_ID);
    for value in [a, b, c, d] {
        push_uint(&mut output, value as u64);
    }
    output
}

fn sha256_parts(parts: &[&[u8]]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update(part);
    }
    hasher.finalize()
}

fn canonical_input_hash(input_cbor: &[u8]) -> [u8; 32] {
    sha256_parts(&[CANONICAL_INPUT_DOMAIN, &[0], input_cbor])
}

fn generate_odd_rows() -> Result<Vec<InputRow>, CalculatorFailure> {
    let mut rows = Vec::with_capacity(480);
    for set_size in 5_usize..=8 {
        for numeric_value in 0..(1_usize << set_size) {
            let (input_cbor, parity) = encode_odd_input(set_size, numeric_value);
            rows.push(InputRow {
                universe_index: rows.len(),
                input_hash: canonical_input_hash(&input_cbor),
                stratum: 1 + 2 * (set_size as u16 - 5) + parity as u16,
            });
        }
    }
    if rows.len() != 480 {
        return Err(CalculatorFailure);
    }
    Ok(rows)
}

fn generate_sink_rows() -> Result<Vec<InputRow>, CalculatorFailure> {
    let mut rows = Vec::with_capacity(85);
    for a in 0_u8..5 {
        for b in 0_u8..5 {
            for c in 0_u8..5 {
                for d in 0_u8..5 {
                    if d as i16 != a as i16 + b as i16 - c as i16 {
                        continue;
                    }
                    let input_cbor = encode_sink_input(a, b, c, d);
                    rows.push(InputRow {
                        universe_index: rows.len(),
                        input_hash: canonical_input_hash(&input_cbor),
                        stratum: 9 + d as u16,
                    });
                }
            }
        }
    }
    if rows.len() != 85 {
        return Err(CalculatorFailure);
    }
    Ok(rows)
}

fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; 32] {
    let mut key_block = [0_u8; 64];
    if key.len() > key_block.len() {
        key_block[..32].copy_from_slice(&sha256_parts(&[key]));
    } else {
        key_block[..key.len()].copy_from_slice(key);
    }
    let mut inner_pad = [0x36_u8; 64];
    let mut outer_pad = [0x5c_u8; 64];
    for index in 0..64 {
        inner_pad[index] ^= key_block[index];
        outer_pad[index] ^= key_block[index];
    }
    let inner = sha256_parts(&[&inner_pad, message]);
    let result = sha256_parts(&[&outer_pad, &inner]);
    zeroize(&mut key_block);
    zeroize(&mut inner_pad);
    zeroize(&mut outer_pad);
    result
}

fn derive_role_key(seed: &[u8; 32], role: u16) -> [u8; 32] {
    let mut prk = hmac_sha256(HKDF_SALT, seed);
    let role_bytes = role.to_be_bytes();
    let mut info = Vec::with_capacity(ROLE_INFO_PREFIX.len() + 3);
    info.extend_from_slice(ROLE_INFO_PREFIX);
    info.extend_from_slice(&role_bytes);
    info.push(1); // RFC 5869 block counter; exactly one block is needed.
    let result = hmac_sha256(&prk, &info);
    zeroize(&mut prk);
    result
}

fn split_rank(
    role_key: &[u8; 32],
    role: u16,
    stratum: u16,
    input_hash: &[u8; 32],
) -> [u8; 32] {
    let mut message = Vec::with_capacity(RANK_PREFIX.len() + 36);
    message.extend_from_slice(RANK_PREFIX);
    message.extend_from_slice(&role.to_be_bytes());
    message.extend_from_slice(&stratum.to_be_bytes());
    message.extend_from_slice(input_hash);
    hmac_sha256(role_key, &message)
}

fn encode_assignment(row: &Assignment) -> Vec<u8> {
    let mut output = Vec::with_capacity(100);
    push_array(&mut output, 9);
    push_uint(&mut output, 1);
    push_uint(&mut output, SPLIT_ASSIGNMENT_ROW_TAG);
    push_bytes(&mut output, SPLIT_ASSIGNMENT_ROW_SCHEMA_ID);
    push_uint(&mut output, row.role as u64);
    push_uint(&mut output, row.universe_index as u64);
    push_bytes(&mut output, &row.input_hash);
    push_uint(&mut output, row.stratum as u64);
    push_uint(&mut output, row.partition as u64);
    push_bytes(&mut output, &row.rank);
    output
}

fn rfc6962_root_from_hashes(hashes: &[[u8; 32]]) -> [u8; 32] {
    match hashes.len() {
        0 => sha256_parts(&[b""]),
        1 => hashes[0],
        count => {
            let split = 1_usize << ((usize::BITS - (count - 1).leading_zeros() - 1) as usize);
            let left = rfc6962_root_from_hashes(&hashes[..split]);
            let right = rfc6962_root_from_hashes(&hashes[split..]);
            sha256_parts(&[&[1], &left, &right])
        }
    }
}

fn rfc6962_root(records: &[Vec<u8>]) -> [u8; 32] {
    let leaves: Vec<[u8; 32]> = records
        .iter()
        .map(|record| sha256_parts(&[&[0], record]))
        .collect();
    rfc6962_root_from_hashes(&leaves)
}

fn build_role_evidence(
    seed: &[u8; 32],
    role: u16,
    rows: Vec<InputRow>,
    quotas: &[Quota],
) -> Result<RoleEvidence, CalculatorFailure> {
    let mut role_key = derive_role_key(seed, role);
    let mut assignments = Vec::with_capacity(rows.len());

    for quota in quotas {
        if quota.discovery + quota.validation + quota.sealed != quota.universe {
            zeroize(&mut role_key);
            return Err(CalculatorFailure);
        }
        let mut ranked: Vec<RankedRow> = rows
            .iter()
            .filter(|row| row.stratum == quota.stratum)
            .map(|row| RankedRow {
                universe_index: row.universe_index,
                input_hash: row.input_hash,
                stratum: row.stratum,
                rank: split_rank(&role_key, role, row.stratum, &row.input_hash),
            })
            .collect();
        if ranked.len() != quota.universe {
            zeroize(&mut role_key);
            return Err(CalculatorFailure);
        }
        ranked.sort_by(|left, right| match left.rank.cmp(&right.rank) {
            CmpOrdering::Equal => left.input_hash.cmp(&right.input_hash),
            other => other,
        });
        if ranked.windows(2).any(|pair| {
            pair[0].rank == pair[1].rank
                && pair[0].input_hash == pair[1].input_hash
                && pair[0].universe_index != pair[1].universe_index
        }) {
            zeroize(&mut role_key);
            return Err(CalculatorFailure);
        }
        for (position, ranked_row) in ranked.into_iter().enumerate() {
            let partition = if position < quota.discovery {
                DISCOVERY_PARTITION_ID
            } else if position < quota.discovery + quota.validation {
                VALIDATION_PARTITION_ID
            } else {
                SEALED_PARTITION_ID
            };
            assignments.push(Assignment {
                role,
                universe_index: ranked_row.universe_index,
                input_hash: ranked_row.input_hash,
                stratum: ranked_row.stratum,
                partition,
                rank: ranked_row.rank,
            });
        }
    }
    zeroize(&mut role_key);

    assignments.sort_by_key(|row| row.universe_index);
    if assignments.len() != rows.len()
        || assignments
            .iter()
            .enumerate()
            .any(|(index, row)| row.universe_index != index)
    {
        return Err(CalculatorFailure);
    }

    let mut counts = [0_usize; 3];
    let mut roots = [[0_u8; 32]; 3];
    for (offset, partition) in [
        DISCOVERY_PARTITION_ID,
        VALIDATION_PARTITION_ID,
        SEALED_PARTITION_ID,
    ]
    .iter()
    .enumerate()
    {
        let records: Vec<Vec<u8>> = assignments
            .iter()
            .filter(|row| row.partition == *partition)
            .map(encode_assignment)
            .collect();
        counts[offset] = records.len();
        roots[offset] = rfc6962_root(&records);
    }
    let expected = [
        quotas.iter().map(|quota| quota.discovery).sum(),
        quotas.iter().map(|quota| quota.validation).sum(),
        quotas.iter().map(|quota| quota.sealed).sum(),
    ];
    if counts != expected || counts.iter().sum::<usize>() != rows.len() {
        return Err(CalculatorFailure);
    }
    Ok(RoleEvidence { role, counts, roots })
}

fn encode_partition_evidence(
    output: &mut Vec<u8>,
    evidence: &RoleEvidence,
    offset: usize,
) {
    push_array(output, 4);
    push_uint(output, evidence.role as u64);
    push_uint(output, (offset + 1) as u64);
    push_uint(output, evidence.counts[offset] as u64);
    push_bytes(output, &evidence.roots[offset]);
}

fn public_payload(seed: &[u8; 32]) -> Result<Vec<u8>, CalculatorFailure> {
    let commitment = sha256_parts(&[SEED_COMMITMENT_PREFIX, &[0], seed]);
    let odd = build_role_evidence(seed, OUTSIDE_ROLE_ID, generate_odd_rows()?, &ODD_QUOTAS)?;
    let sink = build_role_evidence(
        seed,
        NULL_CONTROL_ROLE_ID,
        generate_sink_rows()?,
        &SINK_QUOTAS,
    )?;

    let mut output = Vec::with_capacity(320);
    push_array(&mut output, 4);
    push_uint(&mut output, 1);
    push_bytes(&mut output, RESPONSE_SCHEMA_ID);
    push_bytes(&mut output, &commitment);
    push_array(&mut output, 6);
    for offset in 0..3 {
        encode_partition_evidence(&mut output, &odd, offset);
    }
    for offset in 0..3 {
        encode_partition_evidence(&mut output, &sink, offset);
    }
    Ok(output)
}

fn write_response(payload: &[u8]) -> Result<(), CalculatorFailure> {
    let mut output = take_fifo(PUBLIC_RESPONSE_FD)?;
    let length = u64::try_from(payload.len()).map_err(|_| CalculatorFailure)?;
    output
        .write_all(&length.to_be_bytes())
        .and_then(|_| output.write_all(payload))
        .and_then(|_| output.flush())
        .map_err(|_| CalculatorFailure)
}

fn run() -> Result<(), CalculatorFailure> {
    if env::args_os().count() != 1 {
        return Err(CalculatorFailure);
    }
    let mut seed = Box::new([0_u8; SEED_SIZE]);
    let locked = try_mlock(seed.as_mut_slice());
    if let Err(error) = read_seed(&mut seed) {
        zeroize(seed.as_mut_slice());
        try_munlock(seed.as_mut_slice(), locked);
        return Err(error);
    }
    let payload_result = public_payload(&seed);
    zeroize(seed.as_mut_slice());
    try_munlock(seed.as_mut_slice(), locked);
    write_response(&payload_result?)
}

fn main() -> ExitCode {
    if run().is_ok() {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(FAILURE_EXIT_STATUS)
    }
}

struct Sha256 {
    state: [u32; 8],
    buffer: [u8; 64],
    buffer_len: usize,
    message_len: u64,
}

impl Sha256 {
    const INITIAL_STATE: [u32; 8] = [
        0x6a09_e667,
        0xbb67_ae85,
        0x3c6e_f372,
        0xa54f_f53a,
        0x510e_527f,
        0x9b05_688c,
        0x1f83_d9ab,
        0x5be0_cd19,
    ];

    const ROUND_CONSTANTS: [u32; 64] = [
        0x428a_2f98, 0x7137_4491, 0xb5c0_fbcf, 0xe9b5_dba5, 0x3956_c25b, 0x59f1_11f1,
        0x923f_82a4, 0xab1c_5ed5, 0xd807_aa98, 0x1283_5b01, 0x2431_85be, 0x550c_7dc3,
        0x72be_5d74, 0x80de_b1fe, 0x9bdc_06a7, 0xc19b_f174, 0xe49b_69c1, 0xefbe_4786,
        0x0fc1_9dc6, 0x240c_a1cc, 0x2de9_2c6f, 0x4a74_84aa, 0x5cb0_a9dc, 0x76f9_88da,
        0x983e_5152, 0xa831_c66d, 0xb003_27c8, 0xbf59_7fc7, 0xc6e0_0bf3, 0xd5a7_9147,
        0x06ca_6351, 0x1429_2967, 0x27b7_0a85, 0x2e1b_2138, 0x4d2c_6dfc, 0x5338_0d13,
        0x650a_7354, 0x766a_0abb, 0x81c2_c92e, 0x9272_2c85, 0xa2bf_e8a1, 0xa81a_664b,
        0xc24b_8b70, 0xc76c_51a3, 0xd192_e819, 0xd699_0624, 0xf40e_3585, 0x106a_a070,
        0x19a4_c116, 0x1e37_6c08, 0x2748_774c, 0x34b0_bcb5, 0x391c_0cb3, 0x4ed8_aa4a,
        0x5b9c_ca4f, 0x682e_6ff3, 0x748f_82ee, 0x78a5_636f, 0x84c8_7814, 0x8cc7_0208,
        0x90be_fffa, 0xa450_6ceb, 0xbef9_a3f7, 0xc671_78f2,
    ];

    fn new() -> Self {
        Self {
            state: Self::INITIAL_STATE,
            buffer: [0; 64],
            buffer_len: 0,
            message_len: 0,
        }
    }

    fn update(&mut self, mut input: &[u8]) {
        self.message_len = self
            .message_len
            .checked_add(input.len() as u64)
            .expect("SHA-256 input length overflow");
        if self.buffer_len != 0 {
            let count = (64 - self.buffer_len).min(input.len());
            self.buffer[self.buffer_len..self.buffer_len + count]
                .copy_from_slice(&input[..count]);
            self.buffer_len += count;
            input = &input[count..];
            if self.buffer_len == 64 {
                let block = self.buffer;
                self.compress(&block);
                self.buffer_len = 0;
            } else {
                return;
            }
        }
        while input.len() >= 64 {
            let (block, rest) = input.split_at(64);
            self.compress(block.try_into().expect("block length"));
            input = rest;
        }
        self.buffer[..input.len()].copy_from_slice(input);
        self.buffer_len = input.len();
    }

    fn finalize(mut self) -> [u8; 32] {
        let bit_len = self.message_len.checked_mul(8).expect("SHA-256 bit length");
        self.buffer[self.buffer_len] = 0x80;
        self.buffer_len += 1;
        if self.buffer_len > 56 {
            self.buffer[self.buffer_len..].fill(0);
            let block = self.buffer;
            self.compress(&block);
            self.buffer = [0; 64];
        } else {
            self.buffer[self.buffer_len..56].fill(0);
        }
        self.buffer[56..64].copy_from_slice(&bit_len.to_be_bytes());
        let block = self.buffer;
        self.compress(&block);
        let mut digest = [0_u8; 32];
        for (word, bytes) in self.state.iter().zip(digest.chunks_exact_mut(4)) {
            bytes.copy_from_slice(&word.to_be_bytes());
        }
        digest
    }

    fn compress(&mut self, block: &[u8; 64]) {
        let mut schedule = [0_u32; 64];
        for (index, bytes) in block.chunks_exact(4).take(16).enumerate() {
            schedule[index] = u32::from_be_bytes(bytes.try_into().expect("word length"));
        }
        for index in 16..64 {
            let s0 = schedule[index - 15].rotate_right(7)
                ^ schedule[index - 15].rotate_right(18)
                ^ (schedule[index - 15] >> 3);
            let s1 = schedule[index - 2].rotate_right(17)
                ^ schedule[index - 2].rotate_right(19)
                ^ (schedule[index - 2] >> 10);
            schedule[index] = schedule[index - 16]
                .wrapping_add(s0)
                .wrapping_add(schedule[index - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
        for index in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choice = (e & f) ^ ((!e) & g);
            let temporary1 = h
                .wrapping_add(s1)
                .wrapping_add(choice)
                .wrapping_add(Self::ROUND_CONSTANTS[index])
                .wrapping_add(schedule[index]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temporary2 = s0.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temporary1);
            d = c;
            c = b;
            b = a;
            a = temporary1.wrapping_add(temporary2);
        }
        self.state[0] = self.state[0].wrapping_add(a);
        self.state[1] = self.state[1].wrapping_add(b);
        self.state[2] = self.state[2].wrapping_add(c);
        self.state[3] = self.state[3].wrapping_add(d);
        self.state[4] = self.state[4].wrapping_add(e);
        self.state[5] = self.state[5].wrapping_add(f);
        self.state[6] = self.state[6].wrapping_add(g);
        self.state[7] = self.state[7].wrapping_add(h);
    }
}
