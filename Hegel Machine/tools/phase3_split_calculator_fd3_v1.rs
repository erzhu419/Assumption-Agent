//! One-shot, dependency-free Phase-3 split-seed commitment calculator.
//!
//! Secret input is accepted only from an inherited FIFO at raw file descriptor
//! 3.  Exactly 32 bytes followed by EOF are required.  One public response is
//! written to an inherited FIFO at file descriptor 5 as a uint64-be length and
//! a strict-canonical-CBOR payload.  There is no argv, environment, stdin, or
//! filesystem secret fallback.

#![deny(unsafe_op_in_unsafe_fn)]

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
const SEED_COMMITMENT_DOMAIN: &[u8] = b"HEGEL/SPLIT_MASTER_SEED_COMMITMENT/V1";
const RESPONSE_SCHEMA_ID: &[u8] = b"hegel-phase3-split-calculator-fd3-response/1";
const FAILURE_EXIT_STATUS: u8 = 70;
const F_GETFD: i32 = 1;

unsafe extern "C" {
    fn fcntl(fd: i32, command: i32, ...) -> i32;
    fn mlock(address: *const c_void, length: usize) -> i32;
    fn munlock(address: *const c_void, length: usize) -> i32;
}

#[derive(Debug)]
struct CalculatorFailure;

fn take_fifo(fd: i32) -> Result<File, CalculatorFailure> {
    // SAFETY: F_GETFD accepts any integer descriptor and does not dereference
    // memory.  A negative result rejects a missing descriptor before ownership.
    if unsafe { fcntl(fd, F_GETFD) } < 0 {
        return Err(CalculatorFailure);
    }
    // SAFETY: this one-shot program takes ownership of each contract FD once.
    // The immediately preceding F_GETFD check established that it is open;
    // this single-threaded process cannot race another closer here.
    let file = unsafe { File::from_raw_fd(fd) };
    let metadata = file.metadata().map_err(|_| CalculatorFailure)?;
    if !metadata.file_type().is_fifo() {
        return Err(CalculatorFailure);
    }
    Ok(file)
}

fn try_mlock(secret: &mut [u8]) -> bool {
    if secret.is_empty() {
        return false;
    }
    // SAFETY: the slice remains alive and immovable until the paired munlock.
    unsafe { mlock(secret.as_ptr().cast(), secret.len()) == 0 }
}

fn try_munlock(secret: &mut [u8], locked: bool) {
    if !locked || secret.is_empty() {
        return;
    }
    // SAFETY: this is the same live slice passed to the successful mlock call.
    // Cleanup is best effort, so an OS-level failure is deliberately ignored.
    let _ = unsafe { munlock(secret.as_ptr().cast(), secret.len()) };
}

fn read_seed(seed: &mut [u8; SEED_SIZE]) -> Result<(), CalculatorFailure> {
    let mut input = take_fifo(SEED_FD)?;
    let mut offset = 0;
    while offset < seed.len() {
        let count = input
            .read(&mut seed[offset..])
            .map_err(|_| CalculatorFailure)?;
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

fn push_cbor_byte_string(output: &mut Vec<u8>, value: &[u8]) -> Result<(), CalculatorFailure> {
    match value.len() {
        length @ 0..=23 => output.push(0x40 | length as u8),
        length @ 24..=255 => {
            output.push(0x58);
            output.push(length as u8);
        }
        _ => return Err(CalculatorFailure),
    }
    output.extend_from_slice(value);
    Ok(())
}

fn public_payload(commitment: &[u8; 32]) -> Result<Vec<u8>, CalculatorFailure> {
    let mut payload = Vec::with_capacity(82);
    payload.extend_from_slice(&[0x83, 0x01]); // array(3), shortest-form uint(1)
    push_cbor_byte_string(&mut payload, RESPONSE_SCHEMA_ID)?;
    push_cbor_byte_string(&mut payload, commitment)?;
    Ok(payload)
}

fn write_response(commitment: &[u8; 32]) -> Result<(), CalculatorFailure> {
    let mut output = take_fifo(PUBLIC_RESPONSE_FD)?;
    let payload = public_payload(commitment)?;
    let length = u64::try_from(payload.len()).map_err(|_| CalculatorFailure)?;
    let mut frame = Vec::with_capacity(8 + payload.len());
    frame.extend_from_slice(&length.to_be_bytes());
    frame.extend_from_slice(&payload);
    output
        .write_all(&frame)
        .and_then(|_| output.flush())
        .map_err(|_| CalculatorFailure)
}

fn zeroize(secret: &mut [u8]) {
    // Volatile stores plus compiler fences are a best-effort erasure promise;
    // they do not claim to erase copies held in CPU registers or SHA state.
    compiler_fence(Ordering::SeqCst);
    for byte in secret {
        // SAFETY: `byte` is a valid, exclusively borrowed byte for this loop.
        unsafe { std::ptr::write_volatile(byte, 0) };
    }
    compiler_fence(Ordering::SeqCst);
}

fn run() -> Result<(), CalculatorFailure> {
    // Inspect argv only to reject every user argument; never parse a secret.
    if env::args_os().count() != 1 {
        return Err(CalculatorFailure);
    }
    // Heap allocation keeps the address stable across the complete mlock to
    // zeroize to munlock lifetime, even if the owning Box itself is moved.
    let mut seed = Box::new([0_u8; SEED_SIZE]);
    let seed_locked = try_mlock(seed.as_mut_slice());
    if let Err(error) = read_seed(&mut seed) {
        zeroize(seed.as_mut_slice());
        try_munlock(seed.as_mut_slice(), seed_locked);
        return Err(error);
    }
    let mut hasher = Sha256::new();
    hasher.update(SEED_COMMITMENT_DOMAIN);
    hasher.update(&[0]);
    hasher.update(seed.as_slice());
    let commitment = hasher.finalize();
    zeroize(seed.as_mut_slice());
    try_munlock(seed.as_mut_slice(), seed_locked);
    write_response(&commitment)
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
                // The new part was fully absorbed by the pending block.  Do
                // not reset `buffer_len` merely because `input` is now empty.
                return;
            }
        }

        while input.len() >= 64 {
            let (block, rest) = input.split_at(64);
            self.compress(block.try_into().expect("block length is 64"));
            input = rest;
        }

        self.buffer[..input.len()].copy_from_slice(input);
        self.buffer_len = input.len();
    }

    fn finalize(mut self) -> [u8; 32] {
        let bit_len = self
            .message_len
            .checked_mul(8)
            .expect("SHA-256 bit length overflow");
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
        for (word, output) in self.state.iter().zip(digest.chunks_exact_mut(4)) {
            output.copy_from_slice(&word.to_be_bytes());
        }
        digest
    }

    fn compress(&mut self, block: &[u8; 64]) {
        let mut schedule = [0_u32; 64];
        for (index, chunk) in block.chunks_exact(4).take(16).enumerate() {
            schedule[index] = u32::from_be_bytes(chunk.try_into().expect("word length is 4"));
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
