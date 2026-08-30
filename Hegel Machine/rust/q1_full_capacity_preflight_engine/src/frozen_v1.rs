//! Exact frozen v1 archive-wire projection used by the full capacity endpoint.
//!
//! The quotient traversal lives in the parent module.  This module consumes
//! that same returned continuation bank, replays application identities only,
//! and emits the already frozen CBOR/framing/chunk/external-sort commitments.
//! It never evaluates a target, starts Q1, or materializes a formal archive.

use super::{
    canonicalize_shrink6_source_node, frozen_leaf_manifest, representative_capacity,
    visit_eligible_applications, CapacityLimits, ConstructionSignature, CoverageCounter,
    CoverageKey, EngineError, Operator, OutputSortId, Program, ProgramRef, QuotientState,
    RuntimeValue, TraversalFailure, FROZEN_LEAF_COUNT, FULL_MAXIMUM_AST_DEPTH,
    FULL_MAXIMUM_AST_NODE_COUNT, OPERATORS,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, ErrorKind, Read, Write};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

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
const PROGRAM_RECORD_TAG: u64 = 0x3703;
const COHORT_RECORD_TAG: u64 = 0x3704;
const CLASS_RECORD_TAG: u64 = 0x3705;
const COVERAGE_RECORD_TAG: u64 = 0x3706;
const FIXED_POINT_TAG: u64 = 0x3707;
const SIGNATURE_MANIFEST_TAG: u64 = 0x3709;
const CLOSURE_BUNDLE_TAG: u64 = 0x370a;
const ARCHIVE_PROJECTION_RESULT_TAG: u64 = 0x370c;
const PROGRAM_RECORD_SCHEMA: &[u8] = b"hegel-q1-representative-program/1";
const COHORT_RECORD_SCHEMA: &[u8] = b"hegel-q1-continuation-cohort/1";
const CLASS_RECORD_SCHEMA: &[u8] = b"hegel-q1-quotient-class/1";
const COVERAGE_RECORD_SCHEMA: &[u8] = b"hegel-q1-semantic-coverage/1";
const FIXED_POINT_SCHEMA: &[u8] = b"hegel-q1-fixed-point/1";
const SIGNATURE_MANIFEST_SCHEMA: &[u8] = b"hegel-q1-signature-archive-manifest/1";
const CLOSURE_BUNDLE_SCHEMA: &[u8] = b"hegel-q1-closure-bundle/1";
const SIGNATURE_SATURATION_STATE_SCHEMA: &[u8] =
    b"hegel-q1-signature-saturation-state/1";
const PROJECTION_PARTITION_ROW_SCHEMA: &[u8] =
    b"hegel-q1-archive-projection-partition-row/1";
const PARTITION_STREAM_COMMITMENT_SCHEMA: &[u8] =
    b"hegel-q1-partition-stream-commitment/1";
const PARTITION_EXTERNAL_SORT_SCHEMA: &[u8] =
    b"hegel-q1-partition-external-sort-projection/1";
const ARCHIVE_PROJECTION_RESULT_SCHEMA: &[u8] =
    b"hegel-q1-archive-projection-result/1";
const PROJECTED_STREAM_SCHEMA: &[u8] = b"hegel-q1-projected-record-stream/1";
const PROJECTED_STREAM_DOMAIN: &[u8] = b"HEGEL/Q1/PREFLIGHT/PROJECTED_STREAM/V1";
const STREAM_DESCRIPTOR_SCHEMA: &[u8] = b"hegel-q1-stream-descriptor/1";
const CHUNK_MANIFEST_TAG: u64 = 0x3708;
const CHUNK_MANIFEST_SCHEMA: &[u8] = b"hegel-q1-archive-chunk-manifest/1";
const PROGRAM_RECORD_ID_DOMAIN: &[u8] = b"HEGEL/Q1/PROGRAM_RECORD_ID/V1";
const COHORT_RECORD_ID_DOMAIN: &[u8] = b"HEGEL/Q1/COHORT_RECORD_ID/V1";
const CLASS_RECORD_ID_DOMAIN: &[u8] = b"HEGEL/Q1/CLASS_RECORD_ID/V1";
const COVERAGE_RECORD_ID_DOMAIN: &[u8] = b"HEGEL/Q1/COVERAGE_RECORD_ID/V1";
const FIXED_POINT_ROOT_DOMAIN: &[u8] = b"HEGEL/Q1/FIXED_POINT_RECORD/V1";
const SIGNATURE_MANIFEST_ROOT_DOMAIN: &[u8] =
    b"HEGEL/Q1/SIGNATURE_ARCHIVE_MANIFEST/V1";
const SIGNATURE_SATURATION_STATE_ROOT_DOMAIN: &[u8] =
    b"HEGEL/Q1/SIGNATURE_SATURATION_STATE/V1";
const CLOSURE_BUNDLE_ROOT_DOMAIN: &[u8] = b"HEGEL/Q1/CLOSURE_BUNDLE/V1";
const PARTITION_STREAM_COMMITMENT_DOMAIN: &[u8] =
    b"HEGEL/Q1/PREFLIGHT/PARTITION_STREAM_COMMITMENT/V1";
const PARTITION_EXTERNAL_SORT_ROOT_DOMAIN: &[u8] =
    b"HEGEL/Q1/PREFLIGHT/PARTITION_EXTERNAL_SORT/V1";
const PROJECTION_RESULT_ROOT_DOMAIN: &[u8] = b"HEGEL/Q1/PREFLIGHT/PROJECTION_RESULT/V1";
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
const EXTERNAL_SORT_RUN_SCHEMA: &[u8] = b"hegel-q1-external-sort-run/1";
const SCRATCH_EVENT_SCHEMA: &[u8] = b"hegel-q1-scratch-event/1";
const SAME_RUN_RECORD_SET_SCHEMA: &[u8] = b"hegel-phase3a-q1-full-node6-projection/1";
const SAME_RUN_RECORD_SET_DOMAIN: &[u8] = b"HEGEL/Q1/FULL_NODE6/SAME_RUN_RECORD_SET/V1";
const FROZEN_SEMANTIC_BINDING_ROOT: &str =
    "e3b3df3e81b7632c7c713ef5ec84913f990ad8232a25b851f20c46ac7416bfcb";
const FROZEN_PROJECTION_PROFILE_ROOT: &str =
    "aa441cdc49ab60324483b9aa44e9fdfc324a6ad49a6bff50af6daa775209816d";
const ENDPOINT_RUN_METADATA_RESERVATION_BYTES: u64 = 1_048_576;
const COVERAGE_SORT_BUFFER_BYTES: usize = 1_048_576;
static SCRATCH_NONCE: AtomicU64 = AtomicU64::new(0);

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

fn framed_payload_bytes(payload_bytes: usize) -> Result<u64, EngineError> {
    u32::try_from(payload_bytes).map_err(|_| {
        EngineError::new(
            "INCONCLUSIVE_FROZEN_V1_FRAME_OVERFLOW",
            "formal-wire frame exceeds u32 length",
        )
    })?;
    u64::try_from(payload_bytes)
        .ok()
        .and_then(|length| length.checked_add(4))
        .ok_or_else(|| {
            EngineError::new(
                "INCONCLUSIVE_FROZEN_V1_FRAME_OVERFLOW",
                "formal-wire framed length overflow",
            )
        })
}

fn framed_object_bytes(value: &Cbor) -> Result<u64, EngineError> {
    framed_payload_bytes(encode(value).len())
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

fn hex(value: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(value.len() * 2);
    for byte in value {
        output.push(DIGITS[(byte >> 4) as usize] as char);
        output.push(DIGITS[(byte & 15) as usize] as char);
    }
    output
}

fn prefixed(value: &[u8]) -> String {
    format!("sha256:{}", hex(value))
}

fn decode_root(value: &str) -> Result<[u8; 32], EngineError> {
    if value.len() != 64 {
        return Err(EngineError::new(
            "FAIL_FROZEN_V1_PROJECTION_ROOT",
            "universe root is not 32-byte lowercase hexadecimal",
        ));
    }
    let mut output = [0_u8; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        let digit = |byte: u8| match byte {
            b'0'..=b'9' => Some(byte - b'0'),
            b'a'..=b'f' => Some(byte - b'a' + 10),
            _ => None,
        };
        output[index] = (digit(pair[0]).ok_or_else(|| {
            EngineError::new("FAIL_FROZEN_V1_PROJECTION_ROOT", "universe root is not lowercase")
        })? << 4)
            | digit(pair[1]).ok_or_else(|| {
                EngineError::new("FAIL_FROZEN_V1_PROJECTION_ROOT", "universe root is not lowercase")
            })?;
    }
    Ok(output)
}

fn signature_object(signature: &ConstructionSignature) -> Cbor {
    array([
        uint(1),
        uint(SIGNATURE_TAG),
        bytes(SIGNATURE_SCHEMA),
        uint(signature.output_sort as u64),
        uint(u64::from(signature.ast_depth)),
        uint(u64::from(signature.ast_node_count)),
        uint(u64::from(signature.scalar_occurrence_count)),
        uint(u64::from(signature.aggregate_leaf_count)),
        uint(u64::from(signature.distinct_bit_slot_bitmap)),
        uint(u64::from(signature.scope_clause_count)),
        uint(u64::from(signature.top_level_clause_count)),
        uint(u64::from(signature.old_law_depth)),
        uint(signature.normalization_profile as u64),
        uint(signature.mdl_q32),
    ])
}

fn behavior_cell(sort: OutputSortId, value: &RuntimeValue) -> Result<Cbor, EngineError> {
    if matches!(value, RuntimeValue::Bottom) {
        return Ok(array([uint(0)]));
    }
    let payload = match (sort, value) {
        (OutputSortId::Bool, RuntimeValue::Bool(value)) => Cbor::Bool(*value),
        (OutputSortId::Bit, RuntimeValue::Bit(value)) if *value <= 1 => uint(u64::from(*value)),
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
            return Err(EngineError::new(
                "FAIL_FROZEN_V1_BEHAVIOR",
                "typed behavior cell differs from frozen v1",
            ))
        }
    };
    Ok(array([uint(1), payload]))
}

fn behavior_object(
    input_signature_id: u64,
    universe_root: [u8; 32],
    program: &Program,
) -> Result<Cbor, EngineError> {
    Ok(array([
        uint(1),
        uint(BEHAVIOR_TAG),
        bytes(BEHAVIOR_SCHEMA),
        uint(input_signature_id),
        bytes(universe_root),
        uint(program.signature.output_sort as u64),
        uint(program.behavior.cells.len() as u64),
        array(
            program
                .behavior
                .cells
                .iter()
                .map(|cell| behavior_cell(program.signature.output_sort, cell))
                .collect::<Result<Vec<_>, _>>()?,
        ),
    ]))
}

fn program_identity_object(
    input_signature_id: u64,
    universe_root: [u8; 32],
    program: &Program,
) -> Cbor {
    array([
        uint(input_signature_id),
        bytes(universe_root),
        bytes(&program.canonical.canonical_cbor),
        bytes(program.canonical.canonical_ast_hash),
        signature_object(&program.signature),
    ])
}

#[derive(Debug, Clone)]
struct FormalProgram {
    program: ProgramRef,
    behavior_object: Cbor,
    behavior_id: [u8; 32],
    signature_object: Cbor,
    signature_bytes: Vec<u8>,
    signature_id: [u8; 32],
    program_id: [u8; 32],
}

impl FormalProgram {
    fn new(
        input_signature_id: u64,
        universe_root: [u8; 32],
        program: ProgramRef,
    ) -> Result<Self, EngineError> {
        let behavior_object = behavior_object(input_signature_id, universe_root, &program)?;
        let signature_object = signature_object(&program.signature);
        Ok(Self {
            behavior_id: content_hash(BEHAVIOR_ID_DOMAIN, &behavior_object),
            signature_bytes: encode(&signature_object),
            signature_id: content_hash(SIGNATURE_ID_DOMAIN, &signature_object),
            program_id: content_hash(
                PROGRAM_ID_DOMAIN,
                &program_identity_object(input_signature_id, universe_root, &program),
            ),
            program,
            behavior_object,
            signature_object,
        })
    }
}

fn program_sort_key(program: &FormalProgram) -> Vec<u8> {
    let mut key = Vec::with_capacity(6 + program.program.canonical.canonical_cbor.len());
    key.push(program.program.canonical.depth as u8);
    key.extend_from_slice(&(program.program.canonical.node_count as u16).to_be_bytes());
    key.push(program.program.signature.output_sort as u8);
    key.extend_from_slice(&program.program.canonical.root_operator_id.to_be_bytes());
    key.extend_from_slice(&program.program.canonical.canonical_cbor);
    key
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

#[derive(Debug, Default)]
struct ActualScratchState {
    live: BTreeMap<PathBuf, u64>,
    peak_logical_bytes: u64,
}

#[derive(Debug, Clone, Default)]
pub(super) struct ActualScratchTracker {
    state: Rc<RefCell<ActualScratchState>>,
}

impl ActualScratchTracker {
    fn update_peak(state: &mut ActualScratchState) -> Result<(), EngineError> {
        let live = state.live.values().try_fold(0_u64, |total, bytes| {
            total.checked_add(*bytes).ok_or_else(|| {
                EngineError::new(
                    "INCONCLUSIVE_PREFLIGHT_ACTUAL_SCRATCH_BYTES",
                    "actual scratch logical byte total overflowed u64",
                )
            })
        })?;
        state.peak_logical_bytes = state.peak_logical_bytes.max(live);
        Ok(())
    }

    fn register_created(&self, path: &Path) -> Result<(), EngineError> {
        let metadata = fs::symlink_metadata(path).map_err(|error| {
            EngineError::new(
                "INCONCLUSIVE_PREFLIGHT_ACTUAL_SCRATCH_TRACKER",
                format!("cannot inspect new scratch file {}: {error}", path.display()),
            )
        })?;
        if !metadata.file_type().is_file() || metadata.len() != 0 {
            return Err(EngineError::new(
                "FAIL_PREFLIGHT_ACTUAL_SCRATCH_TRACKER",
                format!(
                    "new scratch path is not an empty regular file: {}",
                    path.display()
                ),
            ));
        }
        let mut state = self.state.borrow_mut();
        if state.live.insert(path.to_path_buf(), 0).is_some() {
            return Err(EngineError::new(
                "FAIL_PREFLIGHT_ACTUAL_SCRATCH_TRACKER",
                format!("scratch file was registered twice: {}", path.display()),
            ));
        }
        Self::update_peak(&mut state)
    }

    fn reconcile_file(&self, path: &Path) -> Result<(), EngineError> {
        let metadata = fs::symlink_metadata(path).map_err(|error| {
            EngineError::new(
                "INCONCLUSIVE_PREFLIGHT_ACTUAL_SCRATCH_TRACKER",
                format!("cannot inspect live scratch file {}: {error}", path.display()),
            )
        })?;
        if !metadata.file_type().is_file() {
            return Err(EngineError::new(
                "FAIL_PREFLIGHT_ACTUAL_SCRATCH_TRACKER",
                format!("live scratch path changed type: {}", path.display()),
            ));
        }
        let mut state = self.state.borrow_mut();
        let prior = state.live.get_mut(path).ok_or_else(|| {
            EngineError::new(
                "FAIL_PREFLIGHT_ACTUAL_SCRATCH_TRACKER",
                format!("scratch file is not registered: {}", path.display()),
            )
        })?;
        if metadata.len() < *prior {
            return Err(EngineError::new(
                "FAIL_PREFLIGHT_ACTUAL_SCRATCH_TRACKER",
                format!("live scratch file shrank: {}", path.display()),
            ));
        }
        *prior = metadata.len();
        Self::update_peak(&mut state)
    }

    fn reconcile_workspace(&self, root: &Path) -> Result<(), EngineError> {
        let mut actual = BTreeSet::new();
        for entry in fs::read_dir(root).map_err(|error| {
            EngineError::new(
                "INCONCLUSIVE_PREFLIGHT_ACTUAL_SCRATCH_TRACKER",
                format!("cannot list scratch workspace {}: {error}", root.display()),
            )
        })? {
            let entry = entry.map_err(|error| {
                EngineError::new(
                    "INCONCLUSIVE_PREFLIGHT_ACTUAL_SCRATCH_TRACKER",
                    format!("cannot inspect scratch workspace row: {error}"),
                )
            })?;
            let file_type = entry.file_type().map_err(|error| {
                EngineError::new(
                    "INCONCLUSIVE_PREFLIGHT_ACTUAL_SCRATCH_TRACKER",
                    format!("cannot inspect scratch workspace entry: {error}"),
                )
            })?;
            if !file_type.is_file() {
                return Err(EngineError::new(
                    "FAIL_PREFLIGHT_ACTUAL_SCRATCH_TRACKER",
                    format!(
                        "scratch workspace contains a non-regular entry: {}",
                        entry.path().display()
                    ),
                ));
            }
            actual.insert(entry.path());
        }
        let registered = self
            .state
            .borrow()
            .live
            .keys()
            .filter(|path| path.starts_with(root))
            .cloned()
            .collect::<BTreeSet<_>>();
        if actual != registered {
            return Err(EngineError::new(
                "FAIL_PREFLIGHT_ACTUAL_SCRATCH_TRACKER",
                "scratch workspace regular-file registry differs from disk",
            ));
        }
        for path in actual {
            self.reconcile_file(&path)?;
        }
        Ok(())
    }

    fn release_removed_workspace(&self, root: &Path) -> Result<(), EngineError> {
        match fs::symlink_metadata(root) {
            Ok(_) => {
                return Err(EngineError::new(
                    "FAIL_PREFLIGHT_ACTUAL_SCRATCH_TRACKER",
                    format!("scratch workspace remains present: {}", root.display()),
                ))
            }
            Err(error) if error.kind() == ErrorKind::NotFound => {}
            Err(error) => {
                return Err(EngineError::new(
                    "INCONCLUSIVE_PREFLIGHT_ACTUAL_SCRATCH_TRACKER",
                    format!(
                        "cannot prove scratch workspace removal {}: {error}",
                        root.display()
                    ),
                ))
            }
        }
        self.state
            .borrow_mut()
            .live
            .retain(|path, _bytes| !path.starts_with(root));
        Ok(())
    }

    pub(super) fn peak_logical_bytes(&self) -> u64 {
        self.state.borrow().peak_logical_bytes
    }

    pub(super) fn require_terminal_empty(&self) -> Result<u64, EngineError> {
        let state = self.state.borrow();
        let live = state.live.values().sum::<u64>();
        if live != 0 || !state.live.is_empty() {
            return Err(EngineError::new(
                "FAIL_PREFLIGHT_ACTUAL_SCRATCH_TRACKER",
                "actual scratch tracker did not close at diagnostic terminal",
            ));
        }
        Ok(live)
    }
}

struct ScratchWorkspace {
    path: PathBuf,
    tracker: ActualScratchTracker,
    cleanup_required: bool,
}

impl ScratchWorkspace {
    fn create(
        parent: Option<&Path>,
        input_signature_id: u64,
        tracker: ActualScratchTracker,
    ) -> Result<Self, EngineError> {
        let nonce = SCRATCH_NONCE.fetch_add(1, AtomicOrdering::Relaxed);
        let name = format!(
            ".hegel-frozen-v1-projection-{}-{}-{}",
            std::process::id(),
            input_signature_id,
            nonce,
        );
        let base = parent
            .map(Path::to_path_buf)
            .unwrap_or_else(std::env::temp_dir);
        let path = base.join(name);
        fs::create_dir(&path).map_err(|error| {
            EngineError::new(
                "FAIL_FROZEN_V1_SCRATCH_CREATE",
                format!("cannot create projection scratch {}: {error}", path.display()),
            )
        })?;
        Ok(Self {
            path,
            tracker,
            cleanup_required: true,
        })
    }

    fn cleanup(mut self) -> Result<(), EngineError> {
        self.tracker.reconcile_workspace(&self.path)?;
        fs::remove_dir_all(&self.path).map_err(|error| {
            EngineError::new(
                "FAIL_FROZEN_V1_SCRATCH_CLEANUP",
                format!(
                    "cannot remove projection scratch {}: {error}",
                    self.path.display()
                ),
            )
        })?;
        if self.path.exists() {
            return Err(EngineError::new(
                "FAIL_FROZEN_V1_SCRATCH_CLEANUP",
                format!("projection scratch {} remains present", self.path.display()),
            ));
        }
        self.tracker.release_removed_workspace(&self.path)?;
        self.cleanup_required = false;
        Ok(())
    }
}

impl Drop for ScratchWorkspace {
    fn drop(&mut self) {
        if self.cleanup_required {
            let _ = self.tracker.reconcile_workspace(&self.path);
            if fs::remove_dir_all(&self.path).is_ok() {
                let _ = self.tracker.release_removed_workspace(&self.path);
            }
        }
    }
}

#[derive(Default)]
struct MerkleFrontier {
    peaks: Vec<(u64, [u8; 32])>,
    count: u64,
}

impl MerkleFrontier {
    fn push(&mut self, record: &[u8]) {
        let mut size = 1_u64;
        let mut root = sha256(&[&[0], record]);
        while self.peaks.last().is_some_and(|(prior, _)| *prior == size) {
            let (_, left) = self.peaks.pop().expect("matching Merkle peak");
            root = sha256(&[&[1], &left, &root]);
            size *= 2;
        }
        self.peaks.push((size, root));
        self.count += 1;
    }

    fn root(&self) -> [u8; 32] {
        let mut peaks = self.peaks.iter().rev();
        let Some((_, first)) = peaks.next() else {
            return sha256(&[b""]);
        };
        let mut root = *first;
        for (_, left) in peaks {
            root = sha256(&[&[1], left, &root]);
        }
        root
    }
}

struct RunReader {
    reader: BufReader<File>,
}

impl RunReader {
    fn open(path: &Path) -> Result<Self, EngineError> {
        Ok(Self {
            reader: BufReader::new(File::open(path).map_err(|error| {
                EngineError::new(
                    "FAIL_FROZEN_V1_SCRATCH_READ",
                    format!("cannot open scratch run {}: {error}", path.display()),
                )
            })?),
        })
    }

    fn next(&mut self) -> Result<Option<Vec<u8>>, EngineError> {
        let mut length = [0_u8; 4];
        match self.reader.read(&mut length[..1]) {
            Ok(0) => return Ok(None),
            Ok(1) => {}
            Ok(_) => unreachable!(),
            Err(error) => {
                return Err(EngineError::new(
                    "FAIL_FROZEN_V1_SCRATCH_READ",
                    format!("cannot read scratch row prefix: {error}"),
                ))
            }
        }
        self.reader.read_exact(&mut length[1..]).map_err(|error| {
            EngineError::new(
                "FAIL_FROZEN_V1_SCRATCH_READ",
                format!("truncated scratch row prefix: {error}"),
            )
        })?;
        let length = u32::from_be_bytes(length) as usize;
        let mut row = vec![0_u8; length];
        self.reader.read_exact(&mut row).map_err(|error| {
            EngineError::new(
                "FAIL_FROZEN_V1_SCRATCH_READ",
                format!("truncated scratch row: {error}"),
            )
        })?;
        Ok(Some(row))
    }
}

struct SortedRows {
    root: [u8; 32],
    count: u64,
    distinct_count: u64,
}

struct SpillSorter {
    scratch: PathBuf,
    tracker: ActualScratchTracker,
    prefix: String,
    buffer: Vec<Vec<u8>>,
    buffer_bytes: usize,
    runs: Vec<PathBuf>,
    pushed: u64,
}

impl SpillSorter {
    fn new(scratch: &Path, tracker: ActualScratchTracker, prefix: String) -> Self {
        Self {
            scratch: scratch.to_path_buf(),
            tracker,
            prefix,
            buffer: Vec::new(),
            buffer_bytes: 0,
            runs: Vec::new(),
            pushed: 0,
        }
    }

    fn push(&mut self, row: Vec<u8>) -> Result<(), EngineError> {
        if !self.buffer.is_empty()
            && self.buffer_bytes.saturating_add(row.len()) > COVERAGE_SORT_BUFFER_BYTES
        {
            self.flush()?;
        }
        self.buffer_bytes = self.buffer_bytes.checked_add(row.len()).ok_or_else(|| {
            EngineError::new(
                "INCONCLUSIVE_FROZEN_V1_COVERAGE_BYTES",
                "coverage sort buffer length overflow",
            )
        })?;
        self.buffer.push(row);
        self.pushed += 1;
        Ok(())
    }

    fn flush(&mut self) -> Result<(), EngineError> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        self.buffer.sort();
        let path = self
            .scratch
            .join(format!("{}-{:08}.run", self.prefix, self.runs.len()));
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|error| {
                EngineError::new(
                    "FAIL_FROZEN_V1_SCRATCH_WRITE",
                    format!("cannot create scratch run {}: {error}", path.display()),
                )
            })?;
        self.tracker.register_created(&path)?;
        let mut writer = BufWriter::new(file);
        for row in &self.buffer {
            let length = u32::try_from(row.len()).map_err(|_| {
                EngineError::new(
                    "INCONCLUSIVE_FROZEN_V1_COVERAGE_ROW",
                    "coverage replay row exceeds u32",
                )
            })?;
            writer
                .write_all(&length.to_be_bytes())
                .and_then(|_| writer.write_all(row))
                .map_err(|error| {
                    EngineError::new(
                        "FAIL_FROZEN_V1_SCRATCH_WRITE",
                        format!("cannot write scratch run {}: {error}", path.display()),
                    )
                })?;
        }
        writer.flush().map_err(|error| {
            EngineError::new(
                "FAIL_FROZEN_V1_SCRATCH_WRITE",
                format!("cannot flush scratch run {}: {error}", path.display()),
            )
        })?;
        self.tracker.reconcile_file(&path)?;
        self.runs.push(path);
        self.buffer.clear();
        self.buffer_bytes = 0;
        Ok(())
    }

    fn finish(mut self) -> Result<SortedRows, EngineError> {
        // Every nonempty coverage stream is actual projection scratch work,
        // including streams that never crossed the bounded-buffer threshold.
        // Flush the final buffer before choosing the empty-input fast path so
        // the shared tracker observes the same real run that the merge reads.
        self.flush()?;
        if self.runs.is_empty() {
            let frontier = MerkleFrontier::default();
            return Ok(SortedRows {
                root: frontier.root(),
                count: frontier.count,
                distinct_count: 0,
            });
        }
        let mut readers = self
            .runs
            .iter()
            .map(|path| RunReader::open(path))
            .collect::<Result<Vec<_>, _>>()?;
        let mut heap = BinaryHeap::<Reverse<(Vec<u8>, usize)>>::new();
        for (index, reader) in readers.iter_mut().enumerate() {
            if let Some(row) = reader.next()? {
                heap.push(Reverse((row, index)));
            }
        }
        let mut frontier = MerkleFrontier::default();
        let mut previous: Option<Vec<u8>> = None;
        let mut distinct_count = 0_u64;
        while let Some(Reverse((row, index))) = heap.pop() {
            frontier.push(&row);
            if previous.as_ref() != Some(&row) {
                distinct_count += 1;
                previous = Some(row.clone());
            }
            if let Some(next) = readers[index].next()? {
                heap.push(Reverse((next, index)));
            }
        }
        if frontier.count != self.pushed {
            return Err(EngineError::new(
                "FAIL_FROZEN_V1_COVERAGE_SORT",
                "coverage merge lost or duplicated a row",
            ));
        }
        Ok(SortedRows {
            root: frontier.root(),
            count: frontier.count,
            distinct_count,
        })
    }
}

struct CoverageMaterial {
    eligible: SpillSorter,
    strict: SpillSorter,
    canonical_asts: SpillSorter,
    rewrites: u64,
}

impl CoverageMaterial {
    fn new(scratch: &Path, tracker: ActualScratchTracker, key: CoverageKey) -> Self {
        let prefix = format!("coverage-{:01}-{:04x}", key.depth, key.code);
        Self {
            eligible: SpillSorter::new(
                scratch,
                tracker.clone(),
                format!("{prefix}-eligible"),
            ),
            strict: SpillSorter::new(scratch, tracker.clone(), format!("{prefix}-strict")),
            canonical_asts: SpillSorter::new(scratch, tracker, format!("{prefix}-ast")),
            rewrites: 0,
        }
    }
}

struct CoverageSummary {
    eligible_count: u64,
    eligible_root: [u8; 32],
    strict_count: u64,
    strict_root: [u8; 32],
    unique_canonical_ast_count: u64,
    rewrites: u64,
}

fn empty_coverage_material(
    scratch: &Path,
    tracker: &ActualScratchTracker,
) -> BTreeMap<CoverageKey, CoverageMaterial> {
    let mut rows = BTreeMap::new();
    for code in 0_u16..FROZEN_LEAF_COUNT as u16 {
        let key = CoverageKey { depth: 0, code };
        rows.insert(key, CoverageMaterial::new(scratch, tracker.clone(), key));
    }
    for depth in 1_u8..=3 {
        for operator in OPERATORS {
            let key = CoverageKey {
                depth,
                code: operator.coverage_code(),
            };
            rows.insert(key, CoverageMaterial::new(scratch, tracker.clone(), key));
        }
    }
    rows
}

fn replay_coverage(
    input_signature_id: u64,
    universe_root: [u8; 32],
    limits: CapacityLimits,
    bank: &[FormalProgram],
    measured: &BTreeMap<CoverageKey, CoverageCounter>,
    scratch: &Path,
    tracker: &ActualScratchTracker,
) -> Result<BTreeMap<CoverageKey, CoverageSummary>, EngineError> {
    let mut coverage = empty_coverage_material(scratch, tracker);
    for (coverage_code, canonical) in frozen_leaf_manifest()?.into_iter().enumerate() {
        let application = application_object(
            input_signature_id,
            universe_root,
            0,
            coverage_code as u16,
            array([]),
            [],
        );
        let application_id = content_hash(APPLICATION_ID_DOMAIN, &application);
        let row = coverage
            .get_mut(&CoverageKey {
                depth: 0,
                code: coverage_code as u16,
            })
            .expect("frozen leaf coverage row");
        row.eligible.push(encode(&application))?;
        row.strict.push(encode(&array([
            bytes(application_id),
            bytes(canonical.canonical_ast_hash),
        ])))?;
        row.canonical_asts.push(canonical.canonical_cbor)?;
    }

    let program_ids = bank
        .iter()
        .map(|row| (row.program.canonical.canonical_cbor.clone(), row.program_id))
        .collect::<BTreeMap<_, _>>();
    let programs = bank.iter().map(|row| row.program.clone()).collect::<Vec<_>>();
    for depth in 1..=limits.maximum_ast_depth {
        visit_eligible_applications(
            &programs,
            depth,
            limits.maximum_ast_node_count,
            |operator, left, right| {
                let code = operator.coverage_code();
                let child_ids = right
                    .into_iter()
                    .chain(std::iter::once(left))
                    .map(|child| {
                        program_ids
                            .get(&child.canonical.canonical_cbor)
                            .copied()
                            .ok_or_else(|| {
                                TraversalFailure::Engine(EngineError::new(
                                    "FAIL_FROZEN_V1_APPLICATION_ID",
                                    "coverage child is absent from the final continuation bank",
                                ))
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                // The traversal callback is left-first; the iterator above is
                // right-first when a right child exists, so restore wire order.
                let child_ids = if right.is_some() {
                    vec![child_ids[1], child_ids[0]]
                } else {
                    child_ids
                };
                let parameters = match operator {
                    Operator::Approx(tolerance) => array([uint(tolerance)]),
                    _ => array([]),
                };
                let application = application_object(
                    input_signature_id,
                    universe_root,
                    depth as u8,
                    code,
                    parameters,
                    child_ids,
                );
                let application_id = content_hash(APPLICATION_ID_DOMAIN, &application);
                let row = coverage
                    .get_mut(&CoverageKey {
                        depth: depth as u8,
                        code,
                    })
                    .expect("frozen operator coverage row");
                row.eligible.push(encode(&application)).map_err(TraversalFailure::Engine)?;
                let source = operator.node(left, right.map(AsRef::as_ref));
                let canonical = match canonicalize_shrink6_source_node(source.clone()) {
                    Ok(value) => value,
                    Err(error) if error.code == "REJECT_STRUCTURAL_LIMIT" => return Ok(()),
                    Err(error) => {
                        return Err(TraversalFailure::Engine(EngineError::new(
                            error.code,
                            error.message,
                        )))
                    }
                };
                if canonical.depth > limits.maximum_ast_depth
                    || canonical.node_count > limits.maximum_ast_node_count
                    || canonical.aggregate_leaf_count > super::MAX_AGGREGATE_LEAVES
                    || canonical.scalar_parameter_occurrence_count > super::MAX_SCALAR_OCCURRENCES
                {
                    return Ok(());
                }
                let rewritten = canonical.canonical_node != source;
                row.strict
                    .push(encode(&array([
                        bytes(application_id),
                        bytes(canonical.canonical_ast_hash),
                    ])))
                    .map_err(TraversalFailure::Engine)?;
                row.canonical_asts
                    .push(canonical.canonical_cbor)
                    .map_err(TraversalFailure::Engine)?;
                row.rewrites += u64::from(rewritten);
                Ok(())
            },
        )
        .map_err(|error| match error {
            TraversalFailure::Engine(error) => error,
            TraversalFailure::Guard(_) => EngineError::new(
                "FAIL_FROZEN_V1_COVERAGE_REPLAY",
                "projection-only replay unexpectedly reached a capacity guard",
            ),
        })?;
    }

    for (key, expected) in measured {
        let actual = coverage.get(key).expect("complete frozen coverage registry");
        if actual.eligible.pushed != expected.eligible
            || actual.strict.pushed != expected.strict_admitted
            || actual.rewrites != expected.rewrite_collapses
        {
            return Err(EngineError::new(
                "FAIL_FROZEN_V1_COVERAGE_REPLAY",
                format!(
                    "coverage ({},{}) replay differs: measured {}/{}/{}, replayed {}/{}/{}",
                    key.depth,
                    key.code,
                    expected.eligible,
                    expected.strict_admitted,
                    expected.rewrite_collapses,
                    actual.eligible.pushed,
                    actual.strict.pushed,
                    actual.rewrites,
                ),
            ));
        }
    }
    coverage
        .into_iter()
        .map(|(key, row)| {
            let CoverageMaterial {
                eligible,
                strict,
                canonical_asts,
                rewrites,
            } = row;
            let eligible = eligible.finish()?;
            let strict = strict.finish()?;
            let canonical_asts = canonical_asts.finish()?;
            Ok((
                key,
                CoverageSummary {
                    eligible_count: eligible.count,
                    eligible_root: eligible.root,
                    strict_count: strict.count,
                    strict_root: strict.root,
                    unique_canonical_ast_count: canonical_asts.distinct_count,
                    rewrites,
                },
            ))
        })
        .collect()
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
        Self { key, object, encoded }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u64)]
enum StreamKind {
    Program = 1,
    Cohort = 2,
    Class = 3,
    Coverage = 4,
}

impl StreamKind {
    fn name(self) -> &'static str {
        match self {
            Self::Program => "PROGRAM",
            Self::Cohort => "COHORT",
            Self::Class => "CLASS",
            Self::Coverage => "COVERAGE",
        }
    }

    fn record_id_domain(self) -> &'static [u8] {
        match self {
            Self::Program => PROGRAM_RECORD_ID_DOMAIN,
            Self::Cohort => COHORT_RECORD_ID_DOMAIN,
            Self::Class => CLASS_RECORD_ID_DOMAIN,
            Self::Coverage => COVERAGE_RECORD_ID_DOMAIN,
        }
    }
}

#[derive(Debug, Clone)]
struct FormalClass {
    behavior_id: [u8; 32],
    behavior_object: Cbor,
    behavior_bytes: Vec<u8>,
    cohorts: BTreeMap<Vec<u8>, Vec<FormalProgram>>,
}

impl FormalClass {
    fn visible_signature_bytes(&self) -> BTreeSet<Vec<u8>> {
        self.cohorts
            .iter()
            .filter(|(_candidate_key, candidate)| {
                !self.cohorts.values().any(|other| {
                    other[0]
                        .program
                        .signature
                        .dominates(&candidate[0].program.signature)
                        && other.len() >= candidate.len()
                })
            })
            .map(|(key, _)| key.clone())
            .collect()
    }
}

struct ArchiveMaterial {
    programs: Vec<ArchiveRecord>,
    cohorts: Vec<ArchiveRecord>,
    classes: Vec<ArchiveRecord>,
    coverage: Vec<ArchiveRecord>,
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

fn build_archive_material(
    input_signature_id: u64,
    universe_root: [u8; 32],
    bank: Vec<FormalProgram>,
    coverage: &BTreeMap<CoverageKey, CoverageSummary>,
) -> Result<ArchiveMaterial, EngineError> {
    let mut bank_programs = bank;
    bank_programs.sort_by_key(program_sort_key);
    let programs = bank_programs
        .iter()
        .enumerate()
        .map(|(index, program)| {
            ArchiveRecord::new(
                program_sort_key(program),
                array([
                    uint(1),
                    uint(PROGRAM_RECORD_TAG),
                    bytes(PROGRAM_RECORD_SCHEMA),
                    uint(input_signature_id),
                    bytes(universe_root),
                    uint(index as u64),
                    bytes(program.program_id),
                    bytes(program.behavior_id),
                    bytes(&program.program.canonical.canonical_cbor),
                    bytes(program.program.canonical.canonical_ast_hash),
                    program.signature_object.clone(),
                    bytes(program.signature_id),
                ]),
            )
        })
        .collect::<Vec<_>>();

    let mut classes = BTreeMap::<[u8; 32], FormalClass>::new();
    for program in bank_programs {
        let class = classes.entry(program.behavior_id).or_insert_with(|| FormalClass {
            behavior_id: program.behavior_id,
            behavior_bytes: encode(&program.behavior_object),
            behavior_object: program.behavior_object.clone(),
            cohorts: BTreeMap::new(),
        });
        if class.behavior_bytes != encode(&program.behavior_object) {
            return Err(EngineError::new(
                "FAIL_FROZEN_V1_BEHAVIOR",
                "different behavior objects share one frozen v1 identity",
            ));
        }
        class
            .cohorts
            .entry(program.signature_bytes.clone())
            .or_default()
            .push(program);
    }
    for class in classes.values_mut() {
        for cohort in class.cohorts.values_mut() {
            cohort.sort_by(|left, right| {
                left.program
                    .canonical
                    .canonical_cbor
                    .cmp(&right.program.canonical.canonical_cbor)
            });
        }
    }

    let mut cohort_rows = Vec::new();
    for (class_id, class) in &classes {
        let visible = class.visible_signature_bytes();
        for cohort in class.cohorts.values() {
            cohort_rows.push((
                *class_id,
                cohort[0].signature_id,
                cohort[0].signature_bytes.clone(),
                cohort.clone(),
                visible.contains(&cohort[0].signature_bytes),
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
            let signature = &cohort[0].program.signature;
            ArchiveRecord::new(
                key,
                array([
                    uint(1),
                    uint(COHORT_RECORD_TAG),
                    bytes(COHORT_RECORD_SCHEMA),
                    uint(input_signature_id),
                    bytes(universe_root),
                    uint(index as u64),
                    bytes(cohort_id(
                        input_signature_id,
                        universe_root,
                        *class_id,
                        *signature_id,
                    )),
                    bytes(class_id),
                    cohort[0].signature_object.clone(),
                    bytes(signature_id),
                    uint(representative_capacity(signature.output_sort) as u64),
                    uint(cohort.len() as u64),
                    array(cohort.iter().enumerate().map(|(rank, program)| {
                        array([
                            uint(rank as u64),
                            bytes(program.program_id),
                            bytes(program.program.canonical.canonical_ast_hash),
                        ])
                    })),
                    Cbor::Bool(*visible),
                ]),
            )
        })
        .collect::<Vec<_>>();

    let mut class_records = Vec::new();
    let mut cohort_cursor = 0_usize;
    for (class_index, (class_id, class)) in classes.iter().enumerate() {
        debug_assert_eq!(*class_id, class.behavior_id);
        let cohort_count = class.cohorts.len();
        let cohort_slice = &cohorts[cohort_cursor..cohort_cursor + cohort_count];
        let visible_records = cohort_slice
            .iter()
            .filter(|record| match &record.object {
                Cbor::Array(fields) => matches!(fields.last(), Some(Cbor::Bool(true))),
                _ => false,
            })
            .map(|record| record.encoded.clone())
            .collect::<Vec<_>>();
        let bank_count = class.cohorts.values().map(Vec::len).sum::<usize>();
        let frontier_count = class
            .cohorts
            .iter()
            .filter(|(key, _)| class.visible_signature_bytes().contains(*key))
            .map(|(_, cohort)| cohort.len())
            .sum::<usize>();
        let minimum_mdl = class
            .cohorts
            .values()
            .flat_map(|cohort| cohort.iter())
            .map(|program| program.program.signature.mdl_q32)
            .min()
            .ok_or_else(|| EngineError::new("FAIL_FROZEN_V1_CLASS", "empty class bank"))?;
        let object = array([
            uint(1),
            uint(CLASS_RECORD_TAG),
            bytes(CLASS_RECORD_SCHEMA),
            uint(input_signature_id),
            bytes(universe_root),
            uint(class_index as u64),
            class.behavior_object.clone(),
            bytes(class_id),
            uint(cohort_cursor as u64),
            uint(cohort_count as u64),
            bytes(rfc6962(
                &cohort_slice
                    .iter()
                    .map(|record| record.encoded.clone())
                    .collect::<Vec<_>>(),
            )),
            uint(bank_count as u64),
            uint(visible_records.len() as u64),
            uint(frontier_count as u64),
            bytes(rfc6962(&visible_records)),
            uint(minimum_mdl),
        ]);
        let mut key = class_id.to_vec();
        key.extend_from_slice(&class.behavior_bytes);
        class_records.push(ArchiveRecord::new(key, object));
        cohort_cursor += cohort_count;
    }

    let coverage_records = coverage
        .iter()
        .map(|(key, row)| {
            ArchiveRecord::new(
                vec![key.depth, (key.code >> 8) as u8, key.code as u8],
                array([
                    uint(1),
                    uint(COVERAGE_RECORD_TAG),
                    bytes(COVERAGE_RECORD_SCHEMA),
                    uint(input_signature_id),
                    bytes(universe_root),
                    uint(u64::from(key.depth)),
                    uint(u64::from(key.code)),
                    uint(row.eligible_count),
                    bytes(row.eligible_root),
                    uint(row.eligible_count),
                    bytes(row.eligible_root),
                    uint(row.strict_count),
                    bytes(row.strict_root),
                    uint(row.unique_canonical_ast_count),
                    uint(row.rewrites),
                ]),
            )
        })
        .collect::<Vec<_>>();
    Ok(ArchiveMaterial {
        programs,
        cohorts,
        classes: class_records,
        coverage: coverage_records,
    })
}

fn cbor_bytes_head(length: usize) -> Vec<u8> {
    let mut output = Vec::new();
    encode_head(2, length as u64, &mut output);
    output
}

fn counting_update_frame_hash(
    hasher: &mut Sha256,
    record: &[u8],
) -> Result<usize, EngineError> {
    let length = u32::try_from(record.len()).map_err(|_| {
        EngineError::new(
            "INCONCLUSIVE_FROZEN_V1_RECORD_TOO_LARGE",
            "record exceeds the frozen u32 frame length",
        )
    })?;
    hasher.update(length.to_be_bytes());
    hasher.update(record);
    Ok(4 + record.len())
}

fn counting_framed_blob_hash<'a>(
    records: impl IntoIterator<Item = &'a [u8]>,
    total_length: usize,
) -> Result<[u8; 32], EngineError> {
    let mut hasher = Sha256::new();
    hasher.update(FRAMED_BLOB_DOMAIN);
    hasher.update([0]);
    hasher.update([0x81]);
    hasher.update(cbor_bytes_head(total_length));
    let mut observed = 0_usize;
    for record in records {
        observed = observed
            .checked_add(counting_update_frame_hash(&mut hasher, record)?)
            .ok_or_else(|| {
                EngineError::new(
                    "INCONCLUSIVE_FROZEN_V1_FRAME_OVERFLOW",
                    "framed chunk length overflows usize",
                )
            })?;
    }
    if observed != total_length {
        return Err(EngineError::new(
            "FAIL_FROZEN_V1_COUNTING_ENCODER",
            "counted chunk length differs from the independent total",
        ));
    }
    Ok(hasher.finalize().into())
}

fn external_row(key: &[u8], record: &[u8]) -> Result<Vec<u8>, EngineError> {
    if key.is_empty() || record.is_empty() {
        return Err(EngineError::new(
            "FAIL_FROZEN_V1_SORT_ROW",
            "external-sort key and record must be nonempty",
        ));
    }
    let key_length = u32::try_from(key.len()).map_err(|_| {
        EngineError::new("INCONCLUSIVE_FROZEN_V1_SORT_ROW", "sort key exceeds u32")
    })?;
    let record_length = u32::try_from(record.len()).map_err(|_| {
        EngineError::new("INCONCLUSIVE_FROZEN_V1_SORT_ROW", "sort record exceeds u32")
    })?;
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
    fn file_id(&self) -> String {
        format!("level-{:04}-run-{:08}", self.level, self.index)
    }

    fn payload(&self) -> Result<Vec<u8>, EngineError> {
        let mut output = Vec::new();
        for (key, record) in &self.rows {
            output.extend_from_slice(&external_row(key, record)?);
        }
        Ok(output)
    }

    fn file_size(&self) -> Result<usize, EngineError> {
        Ok(EXTERNAL_SORT_HEADER_BYTES + self.payload()?.len())
    }

    fn manifest(
        &self,
        input_signature_id: u64,
        stream_kind: StreamKind,
    ) -> Result<Cbor, EngineError> {
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
}

#[derive(Debug, Clone)]
struct ScratchEvent {
    sequence: u64,
    action_id: u8,
    file_id: String,
    prior_size: u64,
    new_size: u64,
    live_logical_bytes_after: u64,
    live_charged_bytes_after: u64,
}

impl ScratchEvent {
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

fn charged_file_bytes(size: usize) -> usize {
    size.div_ceil(4096) * 4096 + 4096
}

#[derive(Default)]
struct ScratchLedger {
    live: BTreeMap<String, usize>,
    events: Vec<ScratchEvent>,
    logical_high_water: usize,
    charged_high_water: usize,
}

impl ScratchLedger {
    fn append(&mut self, action_id: u8, file_id: String, prior: usize, new: usize) {
        let logical = self.live.values().sum::<usize>();
        let charged = self
            .live
            .values()
            .map(|size| charged_file_bytes(*size))
            .sum::<usize>();
        self.logical_high_water = self.logical_high_water.max(logical);
        self.charged_high_water = self.charged_high_water.max(charged);
        self.events.push(ScratchEvent {
            sequence: self.events.len() as u64,
            action_id,
            file_id,
            prior_size: prior as u64,
            new_size: new as u64,
            live_logical_bytes_after: logical as u64,
            live_charged_bytes_after: charged as u64,
        });
    }

    fn allocate(&mut self, file_id: String) -> Result<(), EngineError> {
        if self
            .live
            .insert(file_id.clone(), EXTERNAL_SORT_HEADER_BYTES)
            .is_some()
        {
            return Err(EngineError::new(
                "FAIL_FROZEN_V1_SCRATCH_LEDGER",
                "scratch file was allocated twice",
            ));
        }
        self.append(1, file_id, 0, EXTERNAL_SORT_HEADER_BYTES);
        Ok(())
    }

    fn grow(&mut self, file_id: &str, size: usize) -> Result<(), EngineError> {
        let prior = *self.live.get(file_id).ok_or_else(|| {
            EngineError::new(
                "FAIL_FROZEN_V1_SCRATCH_LEDGER",
                "scratch growth references an absent file",
            )
        })?;
        if size < prior {
            return Err(EngineError::new(
                "FAIL_FROZEN_V1_SCRATCH_LEDGER",
                "scratch file growth decreased size",
            ));
        }
        self.live.insert(file_id.to_owned(), size);
        self.append(2, file_id.to_owned(), prior, size);
        Ok(())
    }

    fn seal(&mut self, file_id: &str) -> Result<(), EngineError> {
        let size = *self.live.get(file_id).ok_or_else(|| {
            EngineError::new(
                "FAIL_FROZEN_V1_SCRATCH_LEDGER",
                "scratch seal references an absent file",
            )
        })?;
        self.append(3, file_id.to_owned(), size, size);
        Ok(())
    }

    fn free(&mut self, file_id: &str) -> Result<(), EngineError> {
        let size = self.live.remove(file_id).ok_or_else(|| {
            EngineError::new(
                "FAIL_FROZEN_V1_SCRATCH_LEDGER",
                "scratch free references an absent file",
            )
        })?;
        self.append(4, file_id.to_owned(), size, 0);
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct ExternalSortProjection {
    charged_scratch_high_water_bytes: u64,
    diagnostic_root: [u8; 32],
    object: Cbor,
}

fn external_sort_projection(
    input_signature_id: u64,
    stream_kind: StreamKind,
    records: &[ArchiveRecord],
) -> Result<ExternalSortProjection, EngineError> {
    let mut rows = records
        .iter()
        .rev()
        .map(|record| (record.key.clone(), record.encoded.clone()))
        .collect::<Vec<_>>();
    rows.sort();
    if rows.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Err(EngineError::new(
            "FAIL_FROZEN_V1_SORT_INPUT",
            "external-sort input repeats a key",
        ));
    }
    let input_payload_bytes = rows
        .iter()
        .map(|(key, record)| external_row(key, record).map(|row| row.len()))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .sum::<usize>();
    let mut initial = Vec::new();
    let mut pending = Vec::new();
    let mut pending_bytes = 0_usize;
    for row in &rows {
        let encoded_length = external_row(&row.0, &row.1)?.len();
        if encoded_length > EXTERNAL_SORT_RUN_PAYLOAD_LIMIT {
            return Err(EngineError::new(
                "INCONCLUSIVE_FROZEN_V1_SORT_ROW_TOO_LARGE",
                "one external-sort row exceeds the frozen run limit",
            ));
        }
        if !pending.is_empty()
            && pending_bytes + encoded_length > EXTERNAL_SORT_RUN_PAYLOAD_LIMIT
        {
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
     -> Result<(), EngineError> {
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
        return Err(EngineError::new(
            "FAIL_FROZEN_V1_SCRATCH_LEDGER",
            "scratch ledger did not close",
        ));
    }
    let sorted_rows_object = array(
        rows.iter()
            .map(|(key, record)| array([bytes(key), bytes(record)])),
    );
    let sorted_stream_root = content_hash(SORTED_STREAM_DOMAIN, &sorted_rows_object);
    let run_manifest_archive_root = rfc6962(&manifests.iter().map(encode).collect::<Vec<_>>());
    let scratch_objects = array(ledger.events.iter().map(ScratchEvent::object));
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
    Ok(ExternalSortProjection {
        charged_scratch_high_water_bytes: ledger.charged_high_water as u64,
        diagnostic_root: content_hash(EXTERNAL_SORT_PROJECTION_DOMAIN, &projection_object),
        object: projection_object,
    })
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct FrozenFormalStreamProjection {
    pub archive_root: String,
    pub canonical_record_payload_bytes: u64,
    pub chunk_count: u64,
    pub chunk_manifest_subtree_root: String,
    pub counting_discard_commitment: String,
    pub external_sort_projection_root: String,
    pub framed_bytes: u64,
    pub record_count: u64,
    pub retained_framed_blob_bytes: u64,
    pub retained_framed_blob_count: u64,
    pub stream_kind: String,
    pub stream_kind_id: u64,
    #[serde(skip)]
    charged_scratch_high_water_bytes: u64,
    #[serde(skip)]
    archive_root_bytes: [u8; 32],
    #[serde(skip)]
    counting_discard_commitment_bytes: [u8; 32],
    #[serde(skip)]
    descriptor_object: Cbor,
    #[serde(skip)]
    external_sort_projection_root_bytes: [u8; 32],
    #[serde(skip)]
    chunk_manifest_records: Vec<Vec<u8>>,
}

fn encode_counting_discard_stream(
    input_signature_id: u64,
    universe_root: [u8; 32],
    stream_kind: StreamKind,
    records: &[ArchiveRecord],
) -> Result<FrozenFormalStreamProjection, EngineError> {
    if records.is_empty() {
        return Err(EngineError::new(
            "FAIL_FROZEN_V1_EMPTY_STREAM",
            "frozen formal-wire stream is empty",
        ));
    }
    let frame_lengths = records
        .iter()
        .map(|record| 4_usize.checked_add(record.encoded.len()))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
            EngineError::new(
                "INCONCLUSIVE_FROZEN_V1_FRAME_OVERFLOW",
                "formal-wire frame length overflow",
            )
        })?;
    if frame_lengths
        .iter()
        .any(|length| *length > MAX_CHUNK_FRAMED_BYTES)
    {
        return Err(EngineError::new(
            "INCONCLUSIVE_FROZEN_V1_RECORD_TOO_LARGE",
            "one record exceeds the frozen chunk limit",
        ));
    }
    let mut ranges = Vec::new();
    let mut first = 0_usize;
    while first < records.len() {
        let mut end = first;
        let mut framed_bytes = 0_usize;
        while end < records.len() {
            let next = frame_lengths[end];
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

    let mut manifests = Vec::with_capacity(ranges.len());
    for (chunk_index, (first, end, framed_length)) in ranges.iter().copied().enumerate() {
        let blob_hash = counting_framed_blob_hash(
            records[first..end]
                .iter()
                .map(|record| record.encoded.as_slice()),
            framed_length,
        )?;
        let record_ids = records[first..end]
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
            uint(first as u64),
            uint((end - first) as u64),
            bytes(record_ids[0]),
            bytes(*record_ids.last().expect("nonempty frozen chunk")),
            bytes(rfc6962(
                &records[first..end]
                    .iter()
                    .map(|record| record.encoded.clone())
                    .collect::<Vec<_>>(),
            )),
            bytes(blob_hash),
            uint(framed_length as u64),
        ]));
    }
    let archive_root = rfc6962(
        &records
            .iter()
            .map(|record| record.encoded.clone())
            .collect::<Vec<_>>(),
    );
    let chunk_manifest_records = manifests.iter().map(encode).collect::<Vec<_>>();
    let chunk_root = rfc6962(&chunk_manifest_records);
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
    let external_sort = external_sort_projection(input_signature_id, stream_kind, records)?;
    let projected_preimage = array([
        uint(1),
        bytes(PROJECTED_STREAM_SCHEMA),
        uint(input_signature_id),
        bytes(universe_root),
        uint(stream_kind as u64),
        descriptor_object.clone(),
        array(manifests),
        external_sort.object.clone(),
    ]);
    let commitment = content_hash(PROJECTED_STREAM_DOMAIN, &projected_preimage);
    Ok(FrozenFormalStreamProjection {
        archive_root: prefixed(&archive_root),
        canonical_record_payload_bytes: records
            .iter()
            .map(|record| record.encoded.len() as u64)
            .sum(),
        chunk_count: ranges.len() as u64,
        chunk_manifest_subtree_root: prefixed(&chunk_root),
        counting_discard_commitment: prefixed(&commitment),
        external_sort_projection_root: prefixed(&external_sort.diagnostic_root),
        framed_bytes: framed_stream_bytes as u64,
        record_count: records.len() as u64,
        retained_framed_blob_bytes: 0,
        retained_framed_blob_count: 0,
        stream_kind: stream_kind.name().to_owned(),
        stream_kind_id: stream_kind as u64,
        charged_scratch_high_water_bytes: external_sort.charged_scratch_high_water_bytes,
        archive_root_bytes: archive_root,
        counting_discard_commitment_bytes: commitment,
        descriptor_object,
        external_sort_projection_root_bytes: external_sort.diagnostic_root,
        chunk_manifest_records,
    })
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct FrozenFormalWireProjection {
    pub diagnostic_only: bool,
    pub formal_archive_materialized: bool,
    pub projected_chunk_manifest_stream_bytes: u64,
    pub projected_fixed_point_frame_bytes: u64,
    pub projected_partition_payload_bytes: u64,
    pub projected_peak_formal_scratch_bytes: u64,
    pub projected_record_stream_bytes: u64,
    pub projected_signature_manifest_frame_bytes: u64,
    pub projection_kind: &'static str,
    pub schema_version: &'static str,
    pub streams: Vec<FrozenFormalStreamProjection>,
    #[serde(skip)]
    pub(super) input_signature_id: u64,
    #[serde(skip)]
    pub(super) universe_root_bytes: [u8; 32],
    #[serde(skip)]
    pub(super) signature_manifest_root: [u8; 32],
    #[serde(skip)]
    pub(super) saturation_state_root: [u8; 32],
    #[serde(skip)]
    pub(super) fixed_point_record_root: [u8; 32],
    #[serde(skip)]
    pub(super) partition_stream_commitment_root: [u8; 32],
    #[serde(skip)]
    pub(super) partition_external_sort_root: [u8; 32],
    #[serde(skip)]
    projection_partition_row: Cbor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct FrozenEndpointFormalResourceProjection {
    pub projected_closure_bundle_frame_bytes: u64,
    pub projected_formal_archive_payload_bytes: u64,
    pub projected_formal_total_output_bytes: u64,
    pub projected_peak_formal_scratch_bytes: u64,
    pub closure_bundle_root: [u8; 32],
    pub archive_projection_result_diagnostic_root: [u8; 32],
}

pub(super) fn endpoint_formal_resource_projection(
    partitions: &[&FrozenFormalWireProjection],
) -> Result<FrozenEndpointFormalResourceProjection, EngineError> {
    if partitions.len() != 2
        || partitions[0].input_signature_id != 1
        || partitions[1].input_signature_id != 2
    {
        return Err(EngineError::new(
            "FAIL_FROZEN_V1_ENDPOINT_PROJECTION",
            "exact ordered Odd/Sink formal partition projections are required",
        ));
    }
    let semantic_binding_root = decode_root(FROZEN_SEMANTIC_BINDING_ROOT)?;
    let projection_profile_root = decode_root(FROZEN_PROJECTION_PROFILE_ROOT)?;
    let closure_bundle = array([
        uint(1),
        uint(CLOSURE_BUNDLE_TAG),
        bytes(CLOSURE_BUNDLE_SCHEMA),
        bytes(semantic_binding_root),
        bytes(projection_profile_root),
        uint(2),
        array(partitions.iter().map(|partition| {
            array([
                uint(partition.input_signature_id),
                bytes(partition.universe_root_bytes),
                bytes(partition.signature_manifest_root),
                bytes(partition.saturation_state_root),
            ])
        })),
    ]);
    let projected_closure_bundle_frame_bytes = framed_object_bytes(&closure_bundle)?;
    let projected_formal_archive_payload_bytes = partitions
        .iter()
        .try_fold(projected_closure_bundle_frame_bytes, |total, partition| {
            total
                .checked_add(partition.projected_partition_payload_bytes)
                .ok_or_else(|| {
                    EngineError::new(
                        "INCONCLUSIVE_FROZEN_V1_FRAME_OVERFLOW",
                        "endpoint formal archive payload length overflow",
                    )
                })
        })?;
    let projected_formal_total_output_bytes = projected_formal_archive_payload_bytes
        .checked_add(ENDPOINT_RUN_METADATA_RESERVATION_BYTES)
        .ok_or_else(|| {
            EngineError::new(
                "INCONCLUSIVE_FROZEN_V1_FRAME_OVERFLOW",
                "endpoint formal total output length overflow",
            )
        })?;
    let projected_peak_formal_scratch_bytes = partitions
        .iter()
        .map(|partition| partition.projected_peak_formal_scratch_bytes)
        .max()
        .unwrap_or(0);
    let archive_projection_result = array([
        uint(1),
        uint(ARCHIVE_PROJECTION_RESULT_TAG),
        bytes(ARCHIVE_PROJECTION_RESULT_SCHEMA),
        bytes(projection_profile_root),
        bytes(semantic_binding_root),
        array(
            partitions
                .iter()
                .map(|partition| partition.projection_partition_row.clone()),
        ),
        uint(projected_closure_bundle_frame_bytes),
        uint(projected_formal_archive_payload_bytes),
        uint(projected_formal_total_output_bytes),
        uint(projected_peak_formal_scratch_bytes),
        uint(ENDPOINT_RUN_METADATA_RESERVATION_BYTES),
        uint(0),
        uint(0),
        uint(0),
        uint(0),
        Cbor::Null,
        array(std::iter::repeat(Cbor::Null).take(8)),
        uint(0),
        Cbor::Bool(false),
        Cbor::Null,
        Cbor::Bool(false),
    ]);
    Ok(FrozenEndpointFormalResourceProjection {
        projected_closure_bundle_frame_bytes,
        projected_formal_archive_payload_bytes,
        projected_formal_total_output_bytes,
        projected_peak_formal_scratch_bytes,
        closure_bundle_root: content_hash(CLOSURE_BUNDLE_ROOT_DOMAIN, &closure_bundle),
        archive_projection_result_diagnostic_root: content_hash(
            PROJECTION_RESULT_ROOT_DOMAIN,
            &archive_projection_result,
        ),
    })
}

#[derive(Debug, Clone)]
pub(super) struct FrozenProjectionResult {
    pub same_run_record_set_root: String,
    pub formal_wire_projection: FrozenFormalWireProjection,
}

pub(super) fn project_same_run(
    input_signature_id: u64,
    universe_root_hex: &str,
    limits: CapacityLimits,
    state: &QuotientState,
    measured_coverage: &BTreeMap<CoverageKey, CoverageCounter>,
    peak_work_queue_points: u64,
    scratch_parent: Option<&Path>,
    scratch_tracker: &ActualScratchTracker,
) -> Result<FrozenProjectionResult, EngineError> {
    let universe_root = decode_root(universe_root_hex)?;
    let workspace = ScratchWorkspace::create(
        scratch_parent,
        input_signature_id,
        scratch_tracker.clone(),
    )?;
    let bank = state
        .continuation_programs()?
        .into_iter()
        .map(|program| FormalProgram::new(input_signature_id, universe_root, program))
        .collect::<Result<Vec<_>, _>>()?;
    let coverage = replay_coverage(
        input_signature_id,
        universe_root,
        limits,
        &bank,
        measured_coverage,
        &workspace.path,
        scratch_tracker,
    )?;
    let material = build_archive_material(input_signature_id, universe_root, bank, &coverage)?;
    let same_run_object = array([
        uint(1),
        bytes(SAME_RUN_RECORD_SET_SCHEMA),
        uint(input_signature_id),
        bytes(universe_root),
        array(material.programs.iter().map(|record| record.object.clone())),
        array(material.cohorts.iter().map(|record| record.object.clone())),
        array(material.classes.iter().map(|record| record.object.clone())),
        array(material.coverage.iter().map(|record| record.object.clone())),
    ]);
    let streams = [
        (StreamKind::Program, material.programs.as_slice()),
        (StreamKind::Cohort, material.cohorts.as_slice()),
        (StreamKind::Class, material.classes.as_slice()),
        (StreamKind::Coverage, material.coverage.as_slice()),
    ]
    .into_iter()
    .map(|(kind, records)| {
        encode_counting_discard_stream(input_signature_id, universe_root, kind, records)
    })
    .collect::<Result<Vec<_>, _>>()?;
    let semantic_binding_root = decode_root(FROZEN_SEMANTIC_BINDING_ROOT)?;
    let projection_profile_root = decode_root(FROZEN_PROJECTION_PROFILE_ROOT)?;
    let counts = state.counts();
    if material.programs.len() as u64 != counts.bank
        || material.cohorts.len() as u64 != counts.cohorts
        || material.classes.len() as u64 != counts.classes
        || material.coverage.len() != 846
        || counts.classes == 0
        || counts.classes > counts.cohorts
        || counts.cohorts > counts.bank
        || counts.bank > 2 * counts.cohorts
        || counts.frontier < counts.classes
        || counts.frontier > counts.bank
        || counts.maximum_frontier_per_class == 0
        || counts.maximum_frontier_per_class > counts.maximum_bank_per_class
        || counts.maximum_bank_per_class > counts.bank
    {
        return Err(EngineError::new(
            "FAIL_FROZEN_V1_FORMAL_ASSEMBLY",
            "same-run quotient cardinalities do not satisfy the frozen formal DAG",
        ));
    }
    let raw_application_count = measured_coverage
        .values()
        .map(|row| row.eligible)
        .sum::<u64>();
    let strict_admitted_count = measured_coverage
        .values()
        .map(|row| row.strict_admitted)
        .sum::<u64>();
    let rewrite_collapse_count = measured_coverage
        .values()
        .map(|row| row.rewrite_collapses)
        .sum::<u64>();
    let structural_boundary_depth = FULL_MAXIMUM_AST_DEPTH + 1;
    let fixed_point = array([
        uint(1),
        uint(FIXED_POINT_TAG),
        bytes(FIXED_POINT_SCHEMA),
        uint(input_signature_id),
        bytes(universe_root),
        bytes(projection_profile_root),
        uint(raw_application_count),
        uint(strict_admitted_count),
        uint(rewrite_collapse_count),
        uint(u64::from(FULL_MAXIMUM_AST_DEPTH)),
        uint(u64::from(FULL_MAXIMUM_AST_NODE_COUNT)),
        uint(u64::from(structural_boundary_depth)),
        Cbor::Bool(true),
        Cbor::Bool(true),
        Cbor::Bool(true),
        uint(0),
        uint(0),
        uint(0),
        uint(0),
        uint(material.programs.len() as u64),
        uint(counts.classes),
        uint(counts.cohorts),
        uint(counts.bank),
        uint(counts.frontier),
        uint(counts.maximum_bank_per_class),
        uint(counts.maximum_frontier_per_class),
        bytes(streams[0].archive_root_bytes),
        bytes(streams[1].archive_root_bytes),
        bytes(streams[2].archive_root_bytes),
        bytes(streams[3].archive_root_bytes),
    ]);
    let fixed_point_record_root = content_hash(FIXED_POINT_ROOT_DOMAIN, &fixed_point);
    let projected_fixed_point_frame_bytes = framed_object_bytes(&fixed_point)?;
    let saturation_state = array([
        uint(1),
        bytes(SIGNATURE_SATURATION_STATE_SCHEMA),
        uint(input_signature_id),
        bytes(universe_root),
        bytes(semantic_binding_root),
        bytes(projection_profile_root),
        bytes(streams[0].archive_root_bytes),
        bytes(streams[1].archive_root_bytes),
        bytes(streams[2].archive_root_bytes),
        bytes(streams[3].archive_root_bytes),
        bytes(fixed_point_record_root),
    ]);
    let saturation_state_root = content_hash(
        SIGNATURE_SATURATION_STATE_ROOT_DOMAIN,
        &saturation_state,
    );
    let chunk_manifest_records = streams
        .iter()
        .flat_map(|stream| stream.chunk_manifest_records.iter().cloned())
        .collect::<Vec<_>>();
    let projected_chunk_manifest_stream_bytes = chunk_manifest_records.iter().try_fold(
        0_u64,
        |total, record| {
            total
                .checked_add(framed_payload_bytes(record.len())?)
                .ok_or_else(|| {
                    EngineError::new(
                        "INCONCLUSIVE_FROZEN_V1_FRAME_OVERFLOW",
                        "chunk-manifest stream framed length overflow",
                    )
                })
        },
    )?;
    let chunk_manifest_archive_root = rfc6962(&chunk_manifest_records);
    let chunk_manifest_count = streams.iter().map(|stream| stream.chunk_count).sum::<u64>();
    let universe_row_count = match input_signature_id {
        1 => 480,
        2 => 85,
        _ => {
            return Err(EngineError::new(
                "FAIL_FROZEN_V1_ENDPOINT_PROJECTION",
                "input signature is outside the frozen Odd/Sink registry",
            ))
        }
    };
    let signature_manifest = array([
        uint(1),
        uint(SIGNATURE_MANIFEST_TAG),
        bytes(SIGNATURE_MANIFEST_SCHEMA),
        uint(input_signature_id),
        bytes(universe_root),
        uint(universe_row_count),
        bytes(semantic_binding_root),
        bytes(projection_profile_root),
        uint(MAX_RECORDS_PER_CHUNK as u64),
        uint(MAX_CHUNK_FRAMED_BYTES as u64),
        array(streams.iter().map(|stream| stream.descriptor_object.clone())),
        bytes(fixed_point_record_root),
        bytes(saturation_state_root),
        uint(chunk_manifest_count),
        bytes(chunk_manifest_archive_root),
    ]);
    let signature_manifest_root = content_hash(
        SIGNATURE_MANIFEST_ROOT_DOMAIN,
        &signature_manifest,
    );
    let projected_signature_manifest_frame_bytes = framed_object_bytes(&signature_manifest)?;
    let projected_record_stream_bytes = streams.iter().map(|row| row.framed_bytes).sum::<u64>();
    let projected_partition_payload_bytes = [
        projected_record_stream_bytes,
        projected_chunk_manifest_stream_bytes,
        projected_fixed_point_frame_bytes,
        projected_signature_manifest_frame_bytes,
    ]
    .into_iter()
    .try_fold(0_u64, |total, value| {
        total.checked_add(value).ok_or_else(|| {
            EngineError::new(
                "INCONCLUSIVE_FROZEN_V1_FRAME_OVERFLOW",
                "formal partition payload length overflow",
            )
        })
    })?;
    let projected_peak_formal_scratch_bytes = streams
        .iter()
        .map(|row| row.charged_scratch_high_water_bytes)
        .max()
        .unwrap_or(0);
    let partition_stream_commitment = array([
        uint(1),
        bytes(PARTITION_STREAM_COMMITMENT_SCHEMA),
        uint(input_signature_id),
        bytes(universe_root),
        uint(raw_application_count),
        uint(counts.classes),
        uint(counts.cohorts),
        uint(counts.bank),
        uint(counts.frontier),
        uint(counts.maximum_bank_per_class),
        uint(counts.maximum_frontier_per_class),
        uint(material.programs.len() as u64),
        uint(material.coverage.len() as u64),
        uint(projected_record_stream_bytes),
        uint(projected_chunk_manifest_stream_bytes),
        array(
            streams
                .iter()
                .map(|stream| bytes(stream.counting_discard_commitment_bytes)),
        ),
    ]);
    let partition_stream_commitment_root = content_hash(
        PARTITION_STREAM_COMMITMENT_DOMAIN,
        &partition_stream_commitment,
    );
    let partition_external_sort = array([
        uint(1),
        bytes(PARTITION_EXTERNAL_SORT_SCHEMA),
        uint(input_signature_id),
        bytes(universe_root),
        array(
            streams
                .iter()
                .map(|stream| bytes(stream.external_sort_projection_root_bytes)),
        ),
        uint(projected_peak_formal_scratch_bytes),
    ]);
    let partition_external_sort_root = content_hash(
        PARTITION_EXTERNAL_SORT_ROOT_DOMAIN,
        &partition_external_sort,
    );
    let projection_partition_row = array([
        uint(1),
        bytes(PROJECTION_PARTITION_ROW_SCHEMA),
        uint(input_signature_id),
        bytes(universe_root),
        uint(raw_application_count),
        uint(counts.classes),
        uint(counts.cohorts),
        uint(counts.bank),
        uint(counts.frontier),
        uint(counts.maximum_bank_per_class),
        uint(counts.maximum_frontier_per_class),
        uint(peak_work_queue_points),
        uint(material.programs.len() as u64),
        uint(material.coverage.len() as u64),
        uint(projected_record_stream_bytes),
        uint(projected_chunk_manifest_stream_bytes),
        uint(projected_fixed_point_frame_bytes),
        uint(projected_signature_manifest_frame_bytes),
        uint(projected_partition_payload_bytes),
        uint(projected_peak_formal_scratch_bytes),
        array(
            streams
                .iter()
                .map(|stream| bytes(stream.counting_discard_commitment_bytes)),
        ),
        bytes(partition_stream_commitment_root),
        array(
            streams
                .iter()
                .map(|stream| bytes(stream.external_sort_projection_root_bytes)),
        ),
        bytes(partition_external_sort_root),
    ]);
    let result = FrozenProjectionResult {
        same_run_record_set_root: prefixed(&content_hash(
            SAME_RUN_RECORD_SET_DOMAIN,
            &same_run_object,
        )),
        formal_wire_projection: FrozenFormalWireProjection {
            diagnostic_only: true,
            formal_archive_materialized: false,
            projected_chunk_manifest_stream_bytes,
            projected_fixed_point_frame_bytes,
            projected_partition_payload_bytes,
            projected_peak_formal_scratch_bytes,
            projected_record_stream_bytes,
            projected_signature_manifest_frame_bytes,
            projection_kind: "FROZEN_FORMAL_WIRE_COUNTING_DISCARD_NOT_MATERIALIZED",
            schema_version: "hegel-phase3a-q1-full-node6-formal-wire-projection/1",
            streams,
            input_signature_id,
            universe_root_bytes: universe_root,
            signature_manifest_root,
            saturation_state_root,
            fixed_point_record_root,
            partition_stream_commitment_root,
            partition_external_sort_root,
            projection_partition_row,
        },
    };
    workspace.cleanup()?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_formal_assembler_node3_root_kat() {
        let report = crate::run_capacity_preflight(
            CapacityLimits::bounded_for_test(3).unwrap(),
            &format!("sha256:{}", "a".repeat(64)),
            &"b".repeat(40),
        )
        .unwrap();
        let partitions = report
            .partitions
            .iter()
            .map(|partition| partition.formal_wire_projection.as_ref().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            partitions
                .iter()
                .map(|partition| (
                    hex(&partition.fixed_point_record_root),
                    hex(&partition.saturation_state_root),
                    hex(&partition.signature_manifest_root),
                    hex(&partition.partition_stream_commitment_root),
                    hex(&partition.partition_external_sort_root),
                ))
                .collect::<Vec<_>>(),
            vec![
                (
                    "0730c1a71c6e6f30ee67a00637ca7704787456c890db18e6b0403068a18f17d4".to_owned(),
                    "1395ffbd845769567674f6b517a1915575606ae3377b3cff2d9933e2383fec4f".to_owned(),
                    "c05244f21720c60877bc27bc9dabb553c09cec194d951cf71d0435e64729e5dd".to_owned(),
                    "27dcd6efdeea1986f3513dea0fc4e46d47d7024864cf1bde2b981fd04c52cd11".to_owned(),
                    "a50027c7ae466cbd6dfb726babdddb1b1280a42fffb88054d1c85046eee12b98".to_owned(),
                ),
                (
                    "09139e258bdd9c1a59b15b561d0d3109e76c1b6770ea9bababc14a208726b29a".to_owned(),
                    "114057d41a0510b688f603d5f3575037beceee601f65f694865278ee72e8942c".to_owned(),
                    "04b48abf1c6e9db9223d12a39f40fa7517f918d99505b122165cfde6d2ed5fa3".to_owned(),
                    "e8b9065a4b256aa22c3f76ccc6bacd3ea299178508cfe5c37ca471baa2e7998d".to_owned(),
                    "10bec0234d380988bdab2880749e7ad6d4aef8fbe35c80f5173ee1d5c1c88004".to_owned(),
                ),
            ],
        );
        let endpoint = endpoint_formal_resource_projection(&partitions).unwrap();
        assert_eq!(
            (
                hex(&endpoint.closure_bundle_root),
                hex(&endpoint.archive_projection_result_diagnostic_root),
            ),
            (
                "c4e1f2d950f66f5530ae834d2769e6c65decf7f50842fc66f158d11fc8aa9511".to_owned(),
                "b29b3ea0c2e0d5b608394458f74cb29c37e813b3a36d05e39975ba12652a7d00".to_owned(),
            ),
        );
    }

    #[test]
    fn spill_merge_preserves_sorted_rfc6962_and_distinct_count() {
        let tracker = ActualScratchTracker::default();
        let workspace = ScratchWorkspace::create(None, 99, tracker.clone()).unwrap();
        let mut sorter = SpillSorter::new(
            &workspace.path,
            tracker.clone(),
            "spill-kat".to_owned(),
        );
        let mut expected = Vec::new();
        for index in (0_u32..24_000).rev() {
            let identity = index % 12_000;
            let mut row = Vec::with_capacity(64);
            row.extend_from_slice(&identity.to_be_bytes());
            row.extend_from_slice(&sha256(&[&identity.to_be_bytes()]));
            row.extend_from_slice(&[0x5a; 28]);
            sorter.push(row.clone()).unwrap();
            expected.push(row);
        }
        assert!(!sorter.runs.is_empty(), "fixture must cross the spill boundary");
        expected.sort();
        let actual = sorter.finish().unwrap();
        assert_eq!(actual.count, expected.len() as u64);
        assert_eq!(actual.root, rfc6962(&expected));
        assert_eq!(
            actual.distinct_count,
            expected.windows(2).filter(|pair| pair[0] != pair[1]).count() as u64 + 1,
        );
        workspace.cleanup().unwrap();
        assert!(tracker.peak_logical_bytes() > 0);
        assert_eq!(tracker.require_terminal_empty().unwrap(), 0);
    }

    #[test]
    fn final_below_threshold_buffer_uses_tracked_scratch_and_preserves_result() {
        let tracker = ActualScratchTracker::default();
        let workspace = ScratchWorkspace::create(None, 101, tracker.clone()).unwrap();
        let workspace_path = workspace.path.clone();
        let mut sorter = SpillSorter::new(
            &workspace.path,
            tracker.clone(),
            "final-buffer-kat".to_owned(),
        );
        let mut expected = vec![
            b"zeta".to_vec(),
            b"alpha".to_vec(),
            b"alpha".to_vec(),
            b"middle".to_vec(),
        ];
        for row in &expected {
            sorter.push(row.clone()).unwrap();
        }
        assert!(sorter.runs.is_empty(), "fixture must remain below the spill boundary");

        expected.sort();
        let actual = sorter.finish().unwrap();
        assert_eq!(actual.count, expected.len() as u64);
        assert_eq!(actual.root, rfc6962(&expected));
        assert_eq!(actual.distinct_count, 3);
        assert!(tracker.peak_logical_bytes() > 0);
        assert!(std::fs::read_dir(&workspace_path).unwrap().next().is_some());

        workspace.cleanup().unwrap();
        assert!(!workspace_path.exists());
        assert_eq!(tracker.require_terminal_empty().unwrap(), 0);
    }

    #[test]
    fn actual_scratch_tracker_counts_all_live_regular_files_and_closes() {
        let tracker = ActualScratchTracker::default();
        let workspace = ScratchWorkspace::create(None, 100, tracker.clone()).unwrap();
        for (name, bytes) in [("first.run", 11_usize), ("second.run", 17_usize)] {
            let path = workspace.path.join(name);
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .unwrap();
            tracker.register_created(&path).unwrap();
            file.write_all(&vec![0x5a; bytes]).unwrap();
            file.flush().unwrap();
            tracker.reconcile_file(&path).unwrap();
        }
        assert_eq!(tracker.peak_logical_bytes(), 28);
        workspace.cleanup().unwrap();
        assert_eq!(tracker.peak_logical_bytes(), 28);
        assert_eq!(tracker.require_terminal_empty().unwrap(), 0);
    }
}
