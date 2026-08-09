//! Independent Rust shrink-step-5 complete-closure diagnostic.
//!
//! This crate is deliberately separate from the Python generator and from the
//! historical 25,872-source shrink-1 capacity subset.  It enumerates the full
//! typed canonical source-token surface of `hegel-old-dsl-v1.5.0`, applies the
//! independently verified shrink-5 strict Rust canonicalizer, and emits exact
//! program/chunk/bucket archive material. It never evaluates a target role and
//! has no authority to start or advance formal M3.

mod formal_core;

use formal_core::{encode_canonical_cbor, rfc6962_root, CborValue};
use hegel_strict_canonicalizer::{BinaryOp, Node, Sort, UnaryOp};
use hegel_strict_canonicalizer_shrink5::{
    canonicalize_shrink5_source_node, Shrink5Error,
    MAXIMUM_TOP_LEVEL_CLAUSES as MAX_TOP_LEVEL_CLAUSES,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::fs;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

pub const IMPLEMENTATION_ID: u64 = 2; // ImplementationId.RUST_INDEPENDENT
pub const IMPLEMENTATION_MACHINE_ID: &str =
    "hegel-rust-m3-shrink5-complete-closure-diagnostic-v1";
pub const PROFILE_ID: &str = "hegel-m3-shrink5-dual-diagnostic-profile-v1";
pub const CLAIM_LEVEL: &str = "NON_FORMAL_DUAL_CHILD_DIAGNOSTIC";
pub const BINDING_PROFILE_ID: &str = "NON_FORMAL_SYNTHETIC_CHILD_BINDINGS_V1";
pub const DSL_VERSION: &str = "hegel-old-dsl-v1.5.0";
pub const FREEZE_VERSION: &str = "hegel-freeze-p2b-p3-v1.5.0";
pub const PARENT_DSL_VERSION: &str = "hegel-old-dsl-v1.4.0";
pub const PARENT_FREEZE_VERSION: &str = "hegel-freeze-p2b-p3-v1.4.0";
pub const HUMAN_AMENDMENT_ID: &str = "hegel-freeze-p2b-p3-v1.5.0-shrink-step5";
pub const SHRINK_STEP_ID: &str = "SHRINK_STEP_5_REDUCE_MAX_TOTAL_NODE_COUNT_7_TO_6";
pub const STRICT_QUALIFICATION_SOURCE_COMMIT: &str =
    "320b0a3458901090cb738023a4398220fb1d9277";
pub const STRICT_QUALIFICATION_EVIDENCE_COMMIT: &str =
    "01b66cd8effeab258797998f594b250188d823da";
pub const STRICT_QUALIFICATION_ARTIFACT_PATH: &str =
    "Hegel Machine/artifacts/phase3_m3_runtime/phase3_shrink5_sealed_dual_strict_qualification_v1.json";
pub const STRICT_QUALIFICATION_ARTIFACT_SHA256: &str =
    "75761fc536d96d5d0bc91c5c0ba30dbc7c9ee21aac8d3f1dc5c96f6aca919b76";
pub const STRICT_QUALIFICATION_DIAGNOSTIC_REPORT_HASH: &str =
    "sha256:5ee04b21477fd9f09271272fd6ecbf876b885b7831b37a868343a93996a187db";
pub const STRICT_QUALIFICATION_STATUS: &str =
    "SEALED_DUAL_STRICT_OUTCOME_REPLAY_PASS";
pub const CANONICALIZER_PROFILE: &str = "hegel-canonical-ast-v1";
pub const MDL_CODE_TABLE_ID: &str = "hegel-mdl-prefix-v1.0.0";
pub const PROGRAM_RECORD_SCHEMA_ID: &[u8] = b"hegel-canonical-program-record/2";
pub const CHUNK_MANIFEST_SCHEMA_ID: &[u8] = b"hegel-program-chunk-manifest/2";
pub const BUCKET_RECORD_SCHEMA_ID: &[u8] = b"hegel-bucket-accounting-record/1";
pub const PROGRAM_RECORD_TAG: u64 = 0x3207;
pub const CHUNK_MANIFEST_TAG: u64 = 0x3209;
pub const BUCKET_RECORD_TAG: u64 = 0x320c;
pub const MAX_CANONICAL_PROGRAMS: usize = 50_000;
pub const MAX_RAW_OPERATOR_APPLICATIONS: u64 = 5_000_000;
pub const RECORDS_PER_CHUNK: usize = 4096;
pub const MAX_DEPTH: u32 = 4;
pub const MAX_NODE_COUNT: u32 = 6;
pub const FORMAL_BUCKET_COUNT: usize = 5 * 5 * 6;
pub const ACTIVE_RATIONAL_PARAMETER_IDS: [u64; 3] = [1, 3, 5];
pub const TOMBSTONED_RATIONAL_PARAMETER_IDS: [u64; 4] = [0, 2, 4, 6];
pub const ACTIVE_BINARY_OPERATOR_IDS_SOURCE: [u64; 6] = [1, 2, 3, 4, 5, 6];
pub const ACTIVE_BINARY_OPERATOR_IDS_FORMAL: [u64; 5] = [1, 2, 3, 5, 6];
pub const TOMBSTONED_BINARY_OPERATOR_IDS: [u64; 1] = [0];
pub const RESERVED_BINARY_OPERATOR_IDS: [u64; 1] = [7];
pub const REJECT_REMOVED_BINARY_OPERATOR: &str = "REJECT_REMOVED_BINARY_OPERATOR";
pub const FROZEN_CHILD_DSL_SPEC_ROOT: [u8; 32] = [
    0x33, 0x40, 0xb3, 0x27, 0x8c, 0xaf, 0x56, 0x2b, 0x56, 0x0c, 0xc3, 0x0c, 0xd1, 0x4d,
    0x3c, 0xd5, 0xf1, 0xd6, 0x28, 0xe2, 0x22, 0xb4, 0x3d, 0x29, 0xd9, 0xd1, 0xe4, 0x1b,
    0x37, 0x9f, 0x56, 0x75,
];
pub const FROZEN_OPERATOR_SEMANTICS_ROOT: [u8; 32] = [
    0x5d, 0x27, 0x00, 0x88, 0x4a, 0xe7, 0x12, 0x5b, 0x97, 0x12, 0xa2, 0xbd, 0x06, 0xaa,
    0x92, 0x9f, 0xea, 0xf2, 0xfa, 0xd1, 0xd4, 0xbf, 0xcd, 0x4f, 0xa5, 0x95, 0x3c, 0x15,
    0x7a, 0x72, 0x0e, 0xe1,
];
pub const FROZEN_IDENTIFIER_REGISTRY_ROOT: [u8; 32] = [
    0x1b, 0x0c, 0x14, 0x11, 0x26, 0xb2, 0x78, 0x77, 0x80, 0x09, 0xd3, 0xeb, 0xbb, 0xf4,
    0x9f, 0x5d, 0xe2, 0x31, 0xad, 0x01, 0x66, 0xa8, 0x8a, 0x8a, 0x9c, 0xaf, 0x36, 0x7b,
    0x35, 0xbf, 0xf8, 0xef,
];
pub const CANONICAL_AST_SCHEMA_ROOT: [u8; 32] = [
    0x82, 0x8f, 0xdc, 0xc9, 0xf1, 0x6e, 0xbd, 0x59, 0x07, 0x02, 0xff, 0x42, 0x97, 0xca,
    0xc6, 0xf6, 0xff, 0xa1, 0x9b, 0x01, 0x29, 0x9e, 0xa7, 0xa9, 0x37, 0x53, 0xa4, 0xfc,
    0xed, 0x09, 0x61, 0xc5,
];
pub const CANONICAL_CBOR_PROFILE_ROOT: [u8; 32] = [
    0x0c, 0xcb, 0xd7, 0x40, 0xc0, 0xb1, 0xf6, 0xa3, 0x9f, 0xb8, 0x15, 0x1e, 0xa5, 0x6e,
    0x11, 0x45, 0x61, 0x09, 0x3e, 0xe4, 0xfc, 0xcb, 0x22, 0x8b, 0xf8, 0x3a, 0x92, 0x94,
    0xe0, 0xba, 0xe7, 0x83,
];

pub const FAIL_ENUMERATOR_INTERNAL: &str = "FAIL_M3_RUST_ENUMERATOR_INTERNAL";
pub const FAIL_ENUMERATION_BINDING: &str = "FAIL_ENUMERATION_BINDING";
pub const FAIL_LATE_CANONICAL_PROGRAM: &str = "FAIL_LATE_CANONICAL_PROGRAM";
pub const FAIL_TYPED_GENERATOR_REJECTED: &str = "FAIL_TYPED_GENERATOR_REJECTED";
pub const FAIL_PROGRAM_MDL_LENGTH_UNAVAILABLE: &str =
    "FAIL_PROGRAM_MDL_LENGTH_UNAVAILABLE";
pub const INCONCLUSIVE_BUDGET: &str = "INCONCLUSIVE_BUDGET";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumerationError {
    pub code: &'static str,
    pub message: String,
}

impl EnumerationError {
    fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

impl fmt::Display for EnumerationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for EnumerationError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct BucketKey {
    output_sort_id: u64,
    depth: u32,
    node_count: u32,
}

impl BucketKey {
    fn traversal_key(self) -> (u32, u32, u64) {
        (self.depth, self.node_count, self.output_sort_id)
    }

    fn formal_index(self) -> u64 {
        debug_assert!((1..=5).contains(&self.output_sort_id));
        debug_assert!(self.depth <= MAX_DEPTH);
        debug_assert!((1..=MAX_NODE_COUNT).contains(&self.node_count));
        (self.output_sort_id - 1) * u64::from((MAX_DEPTH + 1) * MAX_NODE_COUNT)
            + u64::from(self.depth) * u64::from(MAX_NODE_COUNT)
            + u64::from(self.node_count - 1)
    }
}

#[derive(Debug, Clone)]
struct ProgramEntry {
    node: Node,
    canonical_cbor: Vec<u8>,
    canonical_ast_hash: [u8; 32],
    output_sort_id: u64,
    depth: u32,
    node_count: u32,
    root_operator_id: u16,
    distinct_bit_slot_count: usize,
    mdl_bit_length: u64,
    child_order_hash: [u8; 32],
}

impl ProgramEntry {
    fn node_cbor(&self) -> &[u8] {
        // The frozen AST envelope is exactly [1, RootNode].  Both array(2)
        // and uint(1) use one-byte shortest CBOR encodings.
        debug_assert!(self.canonical_cbor.starts_with(&[0x82, 0x01]));
        &self.canonical_cbor[2..]
    }

    fn commutative_key(&self) -> ([u8; 32], &[u8]) {
        (self.child_order_hash, self.node_cbor())
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct BucketCounters {
    pub bucket_index: u64,
    pub output_sort_id: u64,
    pub ast_depth: u32,
    pub ast_node_count: u32,
    pub raw_operator_applications: u64,
    pub accepted_canonical_programs: u64,
    pub syntactic_duplicates: u64,
    pub type_rejections: u64,
    pub structural_limit_rejections: u64,
    pub rewrite_collapses: u64,
    pub first_program_index_or_null: Option<u64>,
    pub last_program_index_or_null: Option<u64>,
}

impl BucketCounters {
    fn new(key: BucketKey) -> Self {
        Self {
            bucket_index: key.formal_index(),
            output_sort_id: key.output_sort_id,
            ast_depth: key.depth,
            ast_node_count: key.node_count,
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct EnumerationReport {
    pub schema_version: &'static str,
    pub profile_id: &'static str,
    pub claim_level: &'static str,
    pub binding_profile_id: &'static str,
    pub diagnostic_only: bool,
    pub authoritative_claim_allowed: bool,
    pub execution_state: &'static str,
    pub formal_roots_generated: bool,
    pub formal_roots: Option<String>,
    pub implementation: &'static str,
    pub implementation_id: u64,
    pub implementation_machine_id: &'static str,
    pub dsl_version: &'static str,
    pub freeze_version: &'static str,
    pub parent_dsl_version: &'static str,
    pub parent_freeze_version: &'static str,
    pub human_amendment_id: &'static str,
    pub shrink_step_id: &'static str,
    pub strict_qualification_source_commit: &'static str,
    pub strict_qualification_evidence_commit: &'static str,
    pub strict_qualification_artifact_path: &'static str,
    pub strict_qualification_artifact_sha256: &'static str,
    pub strict_qualification_diagnostic_report_hash: &'static str,
    pub strict_qualification_status: &'static str,
    pub maximum_top_level_clauses: usize,
    pub and3_generator_attempts_allowed: bool,
    pub and3_raw_operator_application_count: u64,
    pub canonicalizer_profile: &'static str,
    pub mdl_code_table_id: &'static str,
    pub closure_status: &'static str,
    pub closure_status_id: u64,
    pub raw_operator_application_count: u64,
    pub canonical_program_count: usize,
    pub closure_cardinality_or_null: Option<usize>,
    pub frontier_exhausted: bool,
    pub all_type_buckets_closed: bool,
    pub raw_expansion_limit_hit: bool,
    pub wall_clock_abort_hit: bool,
    pub canonical_program_archive_root_or_null: Option<String>,
    pub program_chunk_manifest_root_or_null: Option<String>,
    pub bucket_accounting_root_or_null: Option<String>,
    pub first_out_of_budget_program_hash_or_null: Option<String>,
    pub first_out_of_budget_program_cbor_hex_or_null: Option<String>,
    pub first_out_of_budget_program_ordinal_or_null: Option<u64>,
    pub program_record_count: usize,
    pub chunk_manifest_count: usize,
    pub bucket_record_count: usize,
    pub records_per_chunk: usize,
    pub maximum_canonical_programs: usize,
    pub maximum_raw_operator_applications: u64,
    pub maximum_ast_depth: u32,
    pub maximum_ast_node_count: u32,
    pub formal_bucket_count: usize,
    pub traversal_prefix_complete: bool,
    pub target_roles_evaluated: bool,
    pub split_material_accessed: bool,
    pub secrets_accessed: bool,
    pub aliases_excluded_before_count: [&'static str; 2],
    pub active_aggregate_map_ids: [u64; 3],
    pub tombstoned_aggregate_map_ids: [u64; 3],
    pub active_rational_parameter_ids: [u64; 3],
    pub tombstoned_rational_parameter_ids: [u64; 4],
    pub reserved_rational_parameter_ids: [u64; 1],
    pub active_source_binary_operator_ids: [u64; 6],
    pub active_formal_canonical_binary_operator_ids: [u64; 5],
    pub source_alias_binary_operator_ids: [u64; 1],
    pub tombstoned_binary_operator_ids: [u64; 1],
    pub reserved_binary_operator_ids: [u64; 1],
    pub operator_id_compaction_performed: bool,
    pub automatic_operator_migration_performed: bool,
    pub child_dsl_spec_root: String,
    pub operator_semantics_root: String,
    pub identifier_registry_root: String,
    pub canonical_ast_schema_root: String,
    pub canonical_cbor_profile_root: String,
}

#[derive(Debug)]
pub struct EnumerationArtifacts {
    pub report: EnumerationReport,
    pub canonical_program_records: Vec<Vec<u8>>,
    pub program_chunk_manifests: Vec<Vec<u8>>,
    pub bucket_accounting_records: Vec<Vec<u8>>,
}

impl EnumerationArtifacts {
    /// Write length-framed public archive material.  This is an explicit CLI
    /// output operation and is never used implicitly by the library.
    pub fn write_to_directory(&self, directory: &Path) -> Result<(), EnumerationError> {
        fs::create_dir(directory).map_err(|error| {
            EnumerationError::new(
                FAIL_ENUMERATOR_INTERNAL,
                format!(
                    "failed to exclusively create output directory {}: {error}",
                    directory.display()
                ),
            )
        })?;
        write_framed(
            &directory.join("canonical_program_records.cborframed"),
            &self.canonical_program_records,
        )?;
        write_framed(
            &directory.join("program_chunk_manifests.cborframed"),
            &self.program_chunk_manifests,
        )?;
        write_framed(
            &directory.join("bucket_accounting_records.cborframed"),
            &self.bucket_accounting_records,
        )?;
        let report = canonical_json_line(&self.report)?;
        write_new(&directory.join("report.json"), &report)?;
        Ok(())
    }
}

/// Encode one JSON value as the frozen diagnostic wire: recursively
/// lexicographic object keys, compact separators, ASCII-safe escapes and one
/// trailing LF. `serde_json::Map` is backed by `BTreeMap` because this crate
/// deliberately does not enable the `preserve_order` feature.
pub fn canonical_json_line<T: Serialize>(value: &T) -> Result<Vec<u8>, EnumerationError> {
    let canonical_value = serde_json::to_value(value).map_err(|error| {
        EnumerationError::new(
            FAIL_ENUMERATOR_INTERNAL,
            format!("failed to materialize canonical report JSON: {error}"),
        )
    })?;
    let utf8_wire = serde_json::to_vec(&canonical_value).map_err(|error| {
        EnumerationError::new(
            FAIL_ENUMERATOR_INTERNAL,
            format!("failed to encode canonical report JSON: {error}"),
        )
    })?;
    let utf8_text = std::str::from_utf8(&utf8_wire).map_err(|error| {
        EnumerationError::new(
            FAIL_ENUMERATOR_INTERNAL,
            format!("canonical report JSON was not UTF-8: {error}"),
        )
    })?;
    let mut wire = Vec::with_capacity(utf8_wire.len() + 1);
    for character in utf8_text.chars() {
        let scalar = character as u32;
        if scalar <= 0x7e {
            wire.push(scalar as u8);
        } else if scalar <= 0xffff {
            push_json_u16_escape(&mut wire, scalar as u16);
        } else {
            let supplementary = scalar - 0x1_0000;
            push_json_u16_escape(&mut wire, 0xd800 | ((supplementary >> 10) as u16));
            push_json_u16_escape(&mut wire, 0xdc00 | ((supplementary & 0x3ff) as u16));
        }
    }
    wire.push(b'\n');
    Ok(wire)
}

fn push_json_u16_escape(output: &mut Vec<u8>, value: u16) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    output.extend_from_slice(b"\\u");
    output.push(HEX[((value >> 12) & 0x0f) as usize]);
    output.push(HEX[((value >> 8) & 0x0f) as usize]);
    output.push(HEX[((value >> 4) & 0x0f) as usize]);
    output.push(HEX[(value & 0x0f) as usize]);
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<(), EnumerationError> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| {
            EnumerationError::new(
                FAIL_ENUMERATOR_INTERNAL,
                format!("refused to overwrite {}: {error}", path.display()),
            )
        })?;
    file.write_all(bytes).map_err(|error| {
        EnumerationError::new(
            FAIL_ENUMERATOR_INTERNAL,
            format!("failed to write {}: {error}", path.display()),
        )
    })
}

fn write_framed(path: &Path, records: &[Vec<u8>]) -> Result<(), EnumerationError> {
    let mut output = Vec::new();
    for record in records {
        let length = u32::try_from(record.len()).map_err(|_| {
            EnumerationError::new(FAIL_ENUMERATOR_INTERNAL, "formal record exceeds uint32")
        })?;
        output.extend_from_slice(&length.to_be_bytes());
        output.extend_from_slice(record);
    }
    write_new(path, &output)
}

#[derive(Debug, Clone, Copy)]
struct EnumeratorLimits {
    canonical_budget: usize,
    raw_budget: u64,
    max_depth: u32,
    max_node_count: u32,
}

fn boundary_status(
    complete: bool,
    limits: EnumeratorLimits,
) -> (&'static str, u64) {
    if complete {
        ("COMPLETE", 1)
    } else if limits.canonical_budget == MAX_CANONICAL_PROGRAMS
        && limits.raw_budget == MAX_RAW_OPERATOR_APPLICATIONS
    {
        ("DSL_TOO_LARGE", 2)
    } else {
        // Reduced budgets are structural tests only. They cannot borrow the
        // frozen rank-50,001 terminal label or status ID.
        ("DIAGNOSTIC_PREFIX_BUDGET_EXCEEDED", 0)
    }
}

impl EnumeratorLimits {
    const DIAGNOSTIC: Self = Self {
        canonical_budget: MAX_CANONICAL_PROGRAMS,
        raw_budget: MAX_RAW_OPERATOR_APPLICATIONS,
        max_depth: MAX_DEPTH,
        max_node_count: MAX_NODE_COUNT,
    };
}

struct Enumerator {
    roots: [[u8; 32]; 3],
    limits: EnumeratorLimits,
    pools: BTreeMap<BucketKey, Vec<Arc<ProgramEntry>>>,
    known_cbor: HashSet<Vec<u8>>,
    ordered_programs: Vec<Arc<ProgramEntry>>,
    counters: BTreeMap<BucketKey, BucketCounters>,
    raw_total: u64,
}

impl Enumerator {
    fn new(roots: [[u8; 32]; 3], limits: EnumeratorLimits) -> Self {
        let mut counters = BTreeMap::new();
        for output_sort_id in 1..=5 {
            for depth in 0..=MAX_DEPTH {
                for node_count in 1..=MAX_NODE_COUNT {
                    let key = BucketKey {
                        output_sort_id,
                        depth,
                        node_count,
                    };
                    counters.insert(key, BucketCounters::new(key));
                }
            }
        }
        Self {
            roots,
            limits,
            pools: BTreeMap::new(),
            known_cbor: HashSet::new(),
            ordered_programs: Vec::new(),
            counters,
            raw_total: 0,
        }
    }

    fn enumerate(mut self) -> Result<EnumerationArtifacts, EnumerationError> {
        let mut threshold_reached = false;
        'traversal: for depth in 0..=self.limits.max_depth {
            for node_count in 1..=self.limits.max_node_count {
                for output_sort_id in 1..=5 {
                    let key = BucketKey {
                        output_sort_id,
                        depth,
                        node_count,
                    };
                    let local = self.generate_bucket(key)?;
                    let remaining_through_witness = self
                        .limits
                        .canonical_budget
                        .saturating_add(1)
                        .saturating_sub(self.ordered_programs.len());
                    if local.len() >= remaining_through_witness {
                        let accepted_here = self
                            .limits
                            .canonical_budget
                            .saturating_sub(self.ordered_programs.len());
                        self.record_accepted_range(key, accepted_here)?;
                        self.ordered_programs.extend(
                            local
                                .into_values()
                                .take(remaining_through_witness)
                                .map(Arc::new),
                        );
                        threshold_reached = true;
                        break 'traversal;
                    }
                    self.record_accepted_range(key, local.len())?;
                    let mut pool = Vec::with_capacity(local.len());
                    for entry in local.into_values() {
                        self.known_cbor.insert(entry.canonical_cbor.clone());
                        let entry = Arc::new(entry);
                        self.ordered_programs.push(Arc::clone(&entry));
                        pool.push(entry);
                    }
                    if !pool.is_empty() {
                        self.pools.insert(key, pool);
                    }
                }
            }
        }
        if threshold_reached {
            self.finish_closed_decision(false)
        } else {
            self.finish_closed_decision(true)
        }
    }

    fn record_accepted_range(
        &mut self,
        key: BucketKey,
        count: usize,
    ) -> Result<(), EnumerationError> {
        if count == 0 {
            return Ok(());
        }
        let first = u64::try_from(self.ordered_programs.len()).map_err(|_| {
            EnumerationError::new(FAIL_ENUMERATOR_INTERNAL, "program index exceeds u64")
        })?;
        let count_u64 = u64::try_from(count).map_err(|_| {
            EnumerationError::new(FAIL_ENUMERATOR_INTERNAL, "bucket count exceeds u64")
        })?;
        let counter = self.counters.get_mut(&key).expect("formal bucket exists");
        counter.accepted_canonical_programs = count_u64;
        counter.first_program_index_or_null = Some(first);
        counter.last_program_index_or_null = Some(first + count_u64 - 1);
        Ok(())
    }

    fn begin_raw(&mut self, key: BucketKey) -> Result<(), EnumerationError> {
        if self.raw_total >= self.limits.raw_budget {
            return Err(EnumerationError::new(
                INCONCLUSIVE_BUDGET,
                format!(
                    "raw operator application cap {} reached before a \
                     prefix-complete witness bucket",
                    self.limits.raw_budget
                ),
            ));
        }
        self.raw_total += 1;
        self.counters
            .get_mut(&key)
            .expect("formal bucket exists")
            .raw_operator_applications += 1;
        Ok(())
    }

    fn generate_bucket(
        &mut self,
        key: BucketKey,
    ) -> Result<BTreeMap<(u16, Vec<u8>), ProgramEntry>, EnumerationError> {
        let mut local = BTreeMap::new();
        if key.depth == 0 && key.node_count == 1 {
            self.generate_leaves(key, &mut local)?;
        }
        if key.depth >= 1 && key.node_count >= 2 {
            self.generate_unary(key, &mut local)?;
        }
        if key.depth >= 1 && key.node_count >= 3 {
            self.generate_binary_and_ternary(key, &mut local)?;
            if key.output_sort_id == 1 {
                self.generate_and(key, &mut local)?;
            }
        }
        Ok(local)
    }

    fn generate_leaves(
        &mut self,
        key: BucketKey,
        local: &mut BTreeMap<(u16, Vec<u8>), ProgramEntry>,
    ) -> Result<(), EnumerationError> {
        let mut sources = Vec::new();
        match key.output_sort_id {
            1 => {
                sources.extend((0..4).map(Node::ContextFlag));
                sources.extend((0..2).map(Node::TaskFlag));
            }
            2 => sources.extend((0..8).map(Node::BitAt)),
            3 => {}
            4 => {
                sources.push(Node::SetSize);
                for scope_id in 0..4 {
                    for quantity_id in 0..2 {
                        for scope_extension in all_scope_extensions() {
                            sources.push(Node::Aggregate {
                                map_id: 1,
                                scope_id,
                                quantity_id,
                                scope_extension,
                            });
                        }
                    }
                }
            }
            5 => {
                sources.extend(ACTIVE_RATIONAL_PARAMETER_IDS.map(Node::ScalarConst));
                for map_id in [0, 5] {
                    for scope_id in 0..4 {
                        for quantity_id in 0..2 {
                            for scope_extension in all_scope_extensions() {
                                sources.push(Node::Aggregate {
                                    map_id,
                                    scope_id,
                                    quantity_id,
                                    scope_extension,
                                });
                            }
                        }
                    }
                }
            }
            _ => unreachable!("formal primitive sort ID"),
        }
        for source in sources {
            self.try_source(key, source, local)?;
        }
        Ok(())
    }

    fn generate_unary(
        &mut self,
        key: BucketKey,
        local: &mut BTreeMap<(u16, Vec<u8>), ProgramEntry>,
    ) -> Result<(), EnumerationError> {
        let child_key = |sort_id| BucketKey {
            output_sort_id: sort_id,
            depth: key.depth - 1,
            node_count: key.node_count - 1,
        };
        let specifications: &[(u64, UnaryOp)] = match key.output_sort_id {
            3 => &[(5, UnaryOp::Sign)],
            5 => &[
                (2, UnaryOp::BitToScalar),
                (4, UnaryOp::IntToScalar),
                (5, UnaryOp::Absolute),
            ],
            _ => &[],
        };
        for (child_sort, op) in specifications {
            let children = self
                .pools
                .get(&child_key(*child_sort))
                .cloned()
                .unwrap_or_default();
            for child in children {
                self.try_source(
                    key,
                    Node::Unary {
                        op: *op,
                        child: Box::new(child.node.clone()),
                    },
                    local,
                )?;
            }
        }
        Ok(())
    }

    fn generate_binary_and_ternary(
        &mut self,
        key: BucketKey,
        local: &mut BTreeMap<(u16, Vec<u8>), ProgramEntry>,
    ) -> Result<(), EnumerationError> {
        match key.output_sort_id {
            1 => {
                self.generate_pair_operator(
                    key,
                    5,
                    PairOperator::Binary(BinaryOp::EqualExact),
                    true,
                    local,
                )?;
                self.generate_pair_operator(
                    key,
                    5,
                    PairOperator::Binary(BinaryOp::LessEqual),
                    false,
                    local,
                )?;
                // greater_equal and approx_equal tolerance=0 are frozen
                // normalize-before-count aliases and are intentionally absent.
                self.generate_pair_operator(key, 5, PairOperator::ApproxEqual(1), true, local)?;
                self.generate_pair_operator(key, 5, PairOperator::ApproxEqual(2), true, local)?;
                self.generate_pair_operator(
                    key,
                    3,
                    PairOperator::Binary(BinaryOp::SameSign),
                    true,
                    local,
                )?;
                self.generate_pair_operator(
                    key,
                    3,
                    PairOperator::Binary(BinaryOp::OppositeSign),
                    true,
                    local,
                )?;
            }
            5 => {
                self.generate_pair_operator(
                    key,
                    5,
                    PairOperator::Binary(BinaryOp::Difference),
                    false,
                    local,
                )?;
            }
            _ => {}
        }
        Ok(())
    }

    fn child_bucket_keys(&self, sort_id: u64, target: BucketKey) -> Vec<BucketKey> {
        self.pools
            .keys()
            .copied()
            .filter(|candidate| {
                candidate.output_sort_id == sort_id
                    && candidate.depth < target.depth
                    && candidate.node_count < target.node_count
            })
            .collect()
    }

    fn generate_pair_operator(
        &mut self,
        key: BucketKey,
        child_sort_id: u64,
        operator: PairOperator,
        commutative: bool,
        local: &mut BTreeMap<(u16, Vec<u8>), ProgramEntry>,
    ) -> Result<(), EnumerationError> {
        let keys = self.child_bucket_keys(child_sort_id, key);
        for left_bucket_index in 0..keys.len() {
            let right_start = if commutative { left_bucket_index } else { 0 };
            for right_bucket_index in right_start..keys.len() {
                let left_key = keys[left_bucket_index];
                let right_key = keys[right_bucket_index];
                if left_key.node_count + right_key.node_count + 1 != key.node_count
                    || left_key.depth.max(right_key.depth) + 1 != key.depth
                {
                    continue;
                }
                let left_pool = self.pools.get(&left_key).cloned().unwrap_or_default();
                let right_pool = self.pools.get(&right_key).cloned().unwrap_or_default();
                if commutative && left_bucket_index == right_bucket_index {
                    for left_index in 0..left_pool.len() {
                        for right_index in left_index..right_pool.len() {
                            self.apply_pair(
                                key,
                                &left_pool[left_index],
                                &right_pool[right_index],
                                operator,
                                true,
                                local,
                            )?;
                        }
                    }
                } else {
                    for left in &left_pool {
                        for right in &right_pool {
                            self.apply_pair(
                                key,
                                left,
                                right,
                                operator,
                                commutative,
                                local,
                            )?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn apply_pair(
        &mut self,
        key: BucketKey,
        left: &Arc<ProgramEntry>,
        right: &Arc<ProgramEntry>,
        operator: PairOperator,
        commutative: bool,
        local: &mut BTreeMap<(u16, Vec<u8>), ProgramEntry>,
    ) -> Result<(), EnumerationError> {
        let (left, right) = if commutative && left.commutative_key() > right.commutative_key() {
            (right, left)
        } else {
            (left, right)
        };
        let source = match operator {
            PairOperator::Binary(op) => Node::Binary {
                op,
                left: Box::new(left.node.clone()),
                right: Box::new(right.node.clone()),
            },
            PairOperator::ApproxEqual(tolerance_index) => Node::ApproxEqual {
                left: Box::new(left.node.clone()),
                right: Box::new(right.node.clone()),
                tolerance_index,
            },
        };
        self.try_source(key, source, local)
    }

    fn generate_and(
        &mut self,
        key: BucketKey,
        local: &mut BTreeMap<(u16, Vec<u8>), ProgramEntry>,
    ) -> Result<(), EnumerationError> {
        let keys = self
            .child_bucket_keys(1, key)
            .into_iter()
            .filter(|bucket| {
                self.pools
                    .get(bucket)
                    .is_some_and(|pool| pool.iter().any(|entry| entry.root_operator_id != 0x0400))
            })
            .collect::<Vec<_>>();
        for first_bucket in 0..keys.len() {
            for second_bucket in first_bucket..keys.len() {
                let first_key = keys[first_bucket];
                let second_key = keys[second_bucket];
                if first_key.node_count + second_key.node_count + 1 == key.node_count
                    && first_key.depth.max(second_key.depth) + 1 == key.depth
                {
                    let first = non_and_pool(self.pools.get(&first_key));
                    let second = non_and_pool(self.pools.get(&second_key));
                    if first_bucket == second_bucket {
                        for left in 0..first.len() {
                            for right in left + 1..second.len() {
                                self.apply_and(key, &[&first[left], &second[right]], local)?;
                            }
                        }
                    } else {
                        for left in &first {
                            for right in &second {
                                self.apply_and(key, &[left, right], local)?;
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn apply_and(
        &mut self,
        key: BucketKey,
        children: &[&Arc<ProgramEntry>],
        local: &mut BTreeMap<(u16, Vec<u8>), ProgramEntry>,
    ) -> Result<(), EnumerationError> {
        if children.len() != MAX_TOP_LEVEL_CLAUSES {
            return Err(EnumerationError::new(
                FAIL_ENUMERATOR_INTERNAL,
                "shrink-5 conjunction generator constructed a non-AND2 source",
            ));
        }
        let mut children = children.iter().copied().collect::<Vec<_>>();
        children.sort_by(|left, right| left.node_cbor().cmp(right.node_cbor()));
        if children
            .windows(2)
            .any(|pair| pair[0].canonical_cbor == pair[1].canonical_cbor)
        {
            return Err(EnumerationError::new(
                FAIL_ENUMERATOR_INTERNAL,
                "AND generator admitted duplicate atoms",
            ));
        }
        self.try_source(
            key,
            Node::And(children.into_iter().map(|entry| entry.node.clone()).collect()),
            local,
        )
    }

    fn try_source(
        &mut self,
        source_key: BucketKey,
        source: Node,
        local: &mut BTreeMap<(u16, Vec<u8>), ProgramEntry>,
    ) -> Result<(), EnumerationError> {
        self.begin_raw(source_key)?;
        let original = source.clone();
        let program = match canonicalize_shrink5_source_node(source) {
            Ok(program) => program,
            Err(error) if error.code == "REJECT_STRUCTURAL_LIMIT" => {
                self.counters
                    .get_mut(&source_key)
                    .expect("formal bucket exists")
                    .structural_limit_rejections += 1;
                return Ok(());
            }
            Err(error) => return Err(self.unexpected_canonicalizer_error(error)),
        };
        if original != program.canonical_node {
            self.counters
                .get_mut(&source_key)
                .expect("formal bucket exists")
                .rewrite_collapses += 1;
            // A source that changes under a frozen pre-count rewrite is not a
            // syntactically canonical program attempt.  Its normal form is
            // reached independently through the canonical source surface, so
            // rewrite and duplicate accounting are disjoint.
            return Ok(());
        }
        let output_sort_id = sort_id(program.output_sort);
        let canonical_key = BucketKey {
            output_sort_id,
            depth: program.depth,
            node_count: program.node_count,
        };
        if canonical_key.traversal_key() > source_key.traversal_key() {
            return Err(EnumerationError::new(
                FAIL_ENUMERATOR_INTERNAL,
                "canonicalization increased the traversal bucket",
            ));
        }
        let local_key = (program.root_operator_id, program.canonical_cbor.clone());
        let duplicate = self.known_cbor.contains(&program.canonical_cbor)
            || local.contains_key(&local_key);
        if duplicate {
            self.counters
                .get_mut(&source_key)
                .expect("formal bucket exists")
                .syntactic_duplicates += 1;
            return Ok(());
        }
        if canonical_key != source_key {
            return Err(EnumerationError::new(
                FAIL_LATE_CANONICAL_PROGRAM,
                format!(
                    "source bucket {:?} first discovered canonical bucket {:?}",
                    source_key.traversal_key(),
                    canonical_key.traversal_key()
                ),
            ));
        }
        let mdl_bit_length = mdl_bit_length(&program.canonical_node)?;
        let node_cbor = program.canonical_cbor.get(2..).ok_or_else(|| {
            EnumerationError::new(FAIL_ENUMERATOR_INTERNAL, "canonical AST envelope truncated")
        })?;
        if !program.canonical_cbor.starts_with(&[0x82, 0x01]) {
            return Err(EnumerationError::new(
                FAIL_ENUMERATOR_INTERNAL,
                "canonical AST envelope differs from [1, RootNode]",
            ));
        }
        let child_order_hash = Sha256::digest(node_cbor).into();
        let entry = ProgramEntry {
            node: program.canonical_node,
            canonical_cbor: program.canonical_cbor,
            canonical_ast_hash: program.canonical_ast_hash,
            output_sort_id,
            depth: program.depth,
            node_count: program.node_count,
            root_operator_id: program.root_operator_id,
            distinct_bit_slot_count: program.distinct_bit_slot_count,
            mdl_bit_length,
            child_order_hash,
        };
        local.insert(local_key, entry);
        Ok(())
    }

    fn unexpected_canonicalizer_error(&mut self, error: Shrink5Error) -> EnumerationError {
        if error.code == "REJECT_TYPE_MISMATCH" || error.code == "REJECT_IMPLICIT_COERCION" {
            EnumerationError::new(
                FAIL_TYPED_GENERATOR_REJECTED,
                format!("typed generator produced an illegal tuple: {error}"),
            )
        } else if error.code == "REJECT_REMOVED_AGGREGATE_MAP" {
            EnumerationError::new(
                FAIL_ENUMERATOR_INTERNAL,
                "enumerator produced a tombstoned aggregate map",
            )
        } else if error.code == "REJECT_REMOVED_RATIONAL_PARAMETER" {
            EnumerationError::new(
                FAIL_ENUMERATOR_INTERNAL,
                "enumerator produced a tombstoned rational parameter",
            )
        } else if error.code == REJECT_REMOVED_BINARY_OPERATOR {
            EnumerationError::new(
                FAIL_ENUMERATOR_INTERNAL,
                "enumerator produced tombstoned BinaryOperatorId 0/add",
            )
        } else {
            EnumerationError::new(
                FAIL_ENUMERATOR_INTERNAL,
                format!("strict canonicalizer rejected generated source: {error}"),
            )
        }
    }

    fn finish_closed_decision(
        self,
        complete: bool,
    ) -> Result<EnumerationArtifacts, EnumerationError> {
        let witness = if complete {
            None
        } else {
            self.ordered_programs.get(self.limits.canonical_budget).cloned()
        };
        if !complete && witness.is_none() {
            return Err(EnumerationError::new(
                FAIL_ENUMERATOR_INTERNAL,
                "DSL_TOO_LARGE path lacks program 50,001 witness",
            ));
        }
        let witness_ordinal = if witness.is_some() {
            Some(
                u64::try_from(self.limits.canonical_budget + 1).map_err(|_| {
                    EnumerationError::new(
                        FAIL_ENUMERATOR_INTERNAL,
                        "program ordinal exceeds u64",
                    )
                })?,
            )
        } else {
            None
        };
        let record_count = if complete {
            self.ordered_programs.len()
        } else {
            self.limits.canonical_budget
        };
        let program_records = self
            .ordered_programs
            .iter()
            .take(record_count)
            .enumerate()
            .map(|(index, entry)| encode_program_record(index, entry, &self.roots))
            .collect::<Result<Vec<_>, _>>()?;
        let program_archive_root = rfc6962_root(&program_records);
        let chunk_manifests = encode_chunk_manifests(&program_records)?;
        let chunk_root = rfc6962_root(&chunk_manifests);
        let bucket_records = encode_bucket_records(&self.counters)?;
        let bucket_root = rfc6962_root(&bucket_records);
        let accepted_sum: u64 = self
            .counters
            .values()
            .map(|counter| counter.accepted_canonical_programs)
            .sum();
        if accepted_sum != record_count as u64 {
            return Err(EnumerationError::new(
                FAIL_ENUMERATOR_INTERNAL,
                format!("bucket accepted sum {accepted_sum} != record count {record_count}"),
            ));
        }
        let raw_sum: u64 = self
            .counters
            .values()
            .map(|counter| counter.raw_operator_applications)
            .sum();
        if raw_sum != self.raw_total {
            return Err(EnumerationError::new(
                FAIL_ENUMERATOR_INTERNAL,
                format!("bucket raw sum {raw_sum} != receipt raw count {}", self.raw_total),
            ));
        }
        let (closure_status, closure_status_id) =
            boundary_status(complete, self.limits);
        let report = EnumerationReport {
            schema_version: "hegel-m3-shrink5-rust-closure-enumerator-report/1",
            profile_id: PROFILE_ID,
            claim_level: CLAIM_LEVEL,
            binding_profile_id: BINDING_PROFILE_ID,
            diagnostic_only: true,
            authoritative_claim_allowed: false,
            execution_state: "NOT_RUN",
            formal_roots_generated: false,
            formal_roots: None,
            implementation: "rust",
            implementation_id: IMPLEMENTATION_ID,
            implementation_machine_id: IMPLEMENTATION_MACHINE_ID,
            dsl_version: DSL_VERSION,
            freeze_version: FREEZE_VERSION,
            parent_dsl_version: PARENT_DSL_VERSION,
            parent_freeze_version: PARENT_FREEZE_VERSION,
            human_amendment_id: HUMAN_AMENDMENT_ID,
            shrink_step_id: SHRINK_STEP_ID,
            strict_qualification_source_commit: STRICT_QUALIFICATION_SOURCE_COMMIT,
            strict_qualification_evidence_commit: STRICT_QUALIFICATION_EVIDENCE_COMMIT,
            strict_qualification_artifact_path: STRICT_QUALIFICATION_ARTIFACT_PATH,
            strict_qualification_artifact_sha256: STRICT_QUALIFICATION_ARTIFACT_SHA256,
            strict_qualification_diagnostic_report_hash:
                STRICT_QUALIFICATION_DIAGNOSTIC_REPORT_HASH,
            strict_qualification_status: STRICT_QUALIFICATION_STATUS,
            maximum_top_level_clauses: MAX_TOP_LEVEL_CLAUSES,
            and3_generator_attempts_allowed: false,
            and3_raw_operator_application_count: 0,
            canonicalizer_profile: CANONICALIZER_PROFILE,
            mdl_code_table_id: MDL_CODE_TABLE_ID,
            closure_status,
            closure_status_id,
            raw_operator_application_count: self.raw_total,
            canonical_program_count: record_count,
            closure_cardinality_or_null: complete.then_some(record_count),
            frontier_exhausted: complete,
            all_type_buckets_closed: complete,
            raw_expansion_limit_hit: false,
            wall_clock_abort_hit: false,
            canonical_program_archive_root_or_null: Some(hex_digest(program_archive_root)),
            program_chunk_manifest_root_or_null: Some(hex_digest(chunk_root)),
            bucket_accounting_root_or_null: Some(hex_digest(bucket_root)),
            first_out_of_budget_program_hash_or_null: witness
                .as_ref()
                .map(|entry| hex_digest(entry.canonical_ast_hash)),
            first_out_of_budget_program_cbor_hex_or_null: witness
                .as_ref()
                .map(|entry| hex_encode(&entry.canonical_cbor)),
            first_out_of_budget_program_ordinal_or_null: witness_ordinal,
            program_record_count: program_records.len(),
            chunk_manifest_count: chunk_manifests.len(),
            bucket_record_count: bucket_records.len(),
            records_per_chunk: RECORDS_PER_CHUNK,
            maximum_canonical_programs: self.limits.canonical_budget,
            maximum_raw_operator_applications: self.limits.raw_budget,
            maximum_ast_depth: self.limits.max_depth,
            maximum_ast_node_count: self.limits.max_node_count,
            formal_bucket_count: FORMAL_BUCKET_COUNT,
            traversal_prefix_complete: true,
            target_roles_evaluated: false,
            split_material_accessed: false,
            secrets_accessed: false,
            aliases_excluded_before_count: ["greater_equal", "approx_equal:tolerance=0"],
            active_aggregate_map_ids: [0, 1, 5],
            tombstoned_aggregate_map_ids: [2, 3, 4],
            active_rational_parameter_ids: ACTIVE_RATIONAL_PARAMETER_IDS,
            tombstoned_rational_parameter_ids: TOMBSTONED_RATIONAL_PARAMETER_IDS,
            reserved_rational_parameter_ids: [7],
            active_source_binary_operator_ids: ACTIVE_BINARY_OPERATOR_IDS_SOURCE,
            active_formal_canonical_binary_operator_ids: ACTIVE_BINARY_OPERATOR_IDS_FORMAL,
            source_alias_binary_operator_ids: [4],
            tombstoned_binary_operator_ids: TOMBSTONED_BINARY_OPERATOR_IDS,
            reserved_binary_operator_ids: RESERVED_BINARY_OPERATOR_IDS,
            operator_id_compaction_performed: false,
            automatic_operator_migration_performed: false,
            child_dsl_spec_root: hex_digest(self.roots[0]),
            operator_semantics_root: hex_digest(self.roots[1]),
            identifier_registry_root: hex_digest(self.roots[2]),
            canonical_ast_schema_root: hex_digest(CANONICAL_AST_SCHEMA_ROOT),
            canonical_cbor_profile_root: hex_digest(CANONICAL_CBOR_PROFILE_ROOT),
        };
        Ok(EnumerationArtifacts {
            report,
            canonical_program_records: program_records,
            program_chunk_manifests: chunk_manifests,
            bucket_accounting_records: bucket_records,
        })
    }

}

#[derive(Debug, Clone, Copy)]
enum PairOperator {
    Binary(BinaryOp),
    ApproxEqual(u64),
}

fn non_and_pool(pool: Option<&Vec<Arc<ProgramEntry>>>) -> Vec<Arc<ProgramEntry>> {
    pool.into_iter()
        .flatten()
        .filter(|entry| entry.root_operator_id != 0x0400)
        .cloned()
        .collect()
}

fn all_scope_extensions() -> Vec<Vec<(u64, bool)>> {
    let mut result = vec![Vec::new()];
    for context_id in 0..4 {
        for expected in [false, true] {
            result.push(vec![(context_id, expected)]);
        }
    }
    for first in 0..4 {
        for second in first + 1..4 {
            for first_expected in [false, true] {
                for second_expected in [false, true] {
                    result.push(vec![
                        (first, first_expected),
                        (second, second_expected),
                    ]);
                }
            }
        }
    }
    debug_assert_eq!(result.len(), 33);
    result
}

fn sort_id(sort: Sort) -> u64 {
    match sort {
        Sort::Bool => 1,
        Sort::Bit => 2,
        Sort::Sign => 3,
        Sort::BoundedInt => 4,
        Sort::RationalValue => 5,
    }
}

/// Exact frozen program-code length in integer bits.  Q32 conversion is
/// `bits * 2^32`, so no floating-point operation is involved.
pub fn mdl_bit_length(node: &Node) -> Result<u64, EnumerationError> {
    match node {
        Node::ScalarConst(index) if ACTIVE_RATIONAL_PARAMETER_IDS.contains(index) => {
            Ok(2 + 3 + 3)
        }
        Node::BitAt(index) if *index < 8 => Ok(2 + 3 + elias_delta_length(index + 1)?),
        Node::SetSize => Ok(2 + 3),
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
            Ok(2 + 3 + 3 + 2 + 1 + clause_count_bits + 3 * scope_extension.len() as u64)
        }
        Node::ContextFlag(index) if *index < 4 => {
            Ok(2 + 3 + elias_delta_length(index + 1)?)
        }
        Node::TaskFlag(index) if *index < 2 => Ok(2 + 3 + elias_delta_length(index + 1)?),
        Node::NewSymbolCall(_) => Err(EnumerationError::new(
            FAIL_PROGRAM_MDL_LENGTH_UNAVAILABLE,
            "new_symbol_call is outside the frozen old DSL",
        )),
        Node::Unary { child, .. } => Ok(2 + 2 + mdl_bit_length(child)?),
        Node::Binary { left, right, .. } => {
            Ok(2 + 3 + mdl_bit_length(left)? + mdl_bit_length(right)?)
        }
        Node::ApproxEqual {
            left,
            right,
            tolerance_index,
        } if (1..=2).contains(tolerance_index) => {
            Ok(3 + 1 + mdl_bit_length(left)? + mdl_bit_length(right)? + 2)
        }
        Node::And(children) if children.len() == MAX_TOP_LEVEL_CLAUSES => {
            let child_bits = children
                .iter()
                .map(mdl_bit_length)
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .sum::<u64>();
            Ok(5 + child_bits)
        }
        _ => Err(EnumerationError::new(
            FAIL_PROGRAM_MDL_LENGTH_UNAVAILABLE,
            "canonical node is outside hegel-mdl-prefix-v1.0.0",
        )),
    }
}

fn elias_delta_length(value: u64) -> Result<u64, EnumerationError> {
    if value == 0 {
        return Err(EnumerationError::new(
            FAIL_PROGRAM_MDL_LENGTH_UNAVAILABLE,
            "Elias-delta identifier index must be 1-based",
        ));
    }
    let log_n = 63 - u64::from(value.leading_zeros());
    let log_log = 63 - u64::from((log_n + 1).leading_zeros());
    Ok(log_n + 2 * log_log + 1)
}

fn encode_program_record(
    index: usize,
    entry: &ProgramEntry,
    roots: &[[u8; 32]; 3],
) -> Result<Vec<u8>, EnumerationError> {
    let index = u64::try_from(index).map_err(|_| {
        EnumerationError::new(FAIL_ENUMERATOR_INTERNAL, "program index exceeds u64")
    })?;
    let mdl_q32 = entry.mdl_bit_length.checked_shl(32).ok_or_else(|| {
        EnumerationError::new(
            FAIL_PROGRAM_MDL_LENGTH_UNAVAILABLE,
            "program MDL Q32 length overflow",
        )
    })?;
    encode_formal(CborValue::Array(vec![
        CborValue::Unsigned(1),
        CborValue::Unsigned(PROGRAM_RECORD_TAG),
        CborValue::Bytes(PROGRAM_RECORD_SCHEMA_ID.to_vec()),
        CborValue::Unsigned(index),
        CborValue::Bytes(entry.canonical_cbor.clone()),
        CborValue::Bytes(entry.canonical_ast_hash.to_vec()),
        CborValue::Unsigned(entry.output_sort_id),
        CborValue::Unsigned(u64::from(entry.depth)),
        CborValue::Unsigned(u64::from(entry.node_count)),
        CborValue::Unsigned(entry.distinct_bit_slot_count as u64),
        CborValue::Unsigned(mdl_q32),
        CborValue::Bytes(roots[0].to_vec()),
        CborValue::Bytes(roots[1].to_vec()),
        CborValue::Bytes(roots[2].to_vec()),
    ]))
}

fn encode_chunk_manifests(records: &[Vec<u8>]) -> Result<Vec<Vec<u8>>, EnumerationError> {
    let mut manifests = Vec::new();
    for (chunk_index, chunk) in records.chunks(RECORDS_PER_CHUNK).enumerate() {
        let first = chunk_index * RECORDS_PER_CHUNK;
        let last = first + chunk.len() - 1;
        let subtree_root = rfc6962_root(chunk);
        let mut blob = Vec::new();
        for record in chunk {
            let length = u32::try_from(record.len()).map_err(|_| {
                EnumerationError::new(FAIL_ENUMERATOR_INTERNAL, "program record exceeds uint32")
            })?;
            blob.extend_from_slice(&length.to_be_bytes());
            blob.extend_from_slice(record);
        }
        let blob_hash = domain_hash(b"HEGEL/CHUNK_BLOB/V1", &blob);
        manifests.push(encode_formal(CborValue::Array(vec![
            CborValue::Unsigned(1),
            CborValue::Unsigned(CHUNK_MANIFEST_TAG),
            CborValue::Bytes(CHUNK_MANIFEST_SCHEMA_ID.to_vec()),
            CborValue::Unsigned(chunk_index as u64),
            CborValue::Unsigned(first as u64),
            CborValue::Unsigned(last as u64),
            CborValue::Unsigned(chunk.len() as u64),
            CborValue::Bytes(subtree_root.to_vec()),
            CborValue::Bytes(blob_hash.to_vec()),
            CborValue::Unsigned(blob.len() as u64),
        ]))?);
    }
    Ok(manifests)
}

fn encode_bucket_records(
    counters: &BTreeMap<BucketKey, BucketCounters>,
) -> Result<Vec<Vec<u8>>, EnumerationError> {
    let by_index = counters
        .values()
        .map(|counter| (counter.bucket_index, counter))
        .collect::<BTreeMap<_, _>>();
    if by_index.len() != FORMAL_BUCKET_COUNT {
        return Err(EnumerationError::new(
            FAIL_ENUMERATOR_INTERNAL,
            "formal zero-bucket grid is not 150 rows",
        ));
    }
    by_index
        .into_values()
        .map(|counter| {
            encode_formal(CborValue::Array(vec![
                CborValue::Unsigned(1),
                CborValue::Unsigned(BUCKET_RECORD_TAG),
                CborValue::Bytes(BUCKET_RECORD_SCHEMA_ID.to_vec()),
                CborValue::Unsigned(counter.bucket_index),
                CborValue::Unsigned(counter.output_sort_id),
                CborValue::Unsigned(u64::from(counter.ast_depth)),
                CborValue::Unsigned(u64::from(counter.ast_node_count)),
                CborValue::Unsigned(counter.raw_operator_applications),
                CborValue::Unsigned(counter.accepted_canonical_programs),
                CborValue::Unsigned(counter.syntactic_duplicates),
                CborValue::Unsigned(counter.type_rejections),
                CborValue::Unsigned(counter.structural_limit_rejections),
                CborValue::Unsigned(counter.rewrite_collapses),
                option_uint(counter.first_program_index_or_null),
                option_uint(counter.last_program_index_or_null),
            ]))
        })
        .collect()
}

fn option_uint(value: Option<u64>) -> CborValue {
    value.map(CborValue::Unsigned).unwrap_or(CborValue::Null)
}

fn encode_formal(value: CborValue) -> Result<Vec<u8>, EnumerationError> {
    encode_canonical_cbor(&value).map_err(|error| {
        EnumerationError::new(
            FAIL_ENUMERATOR_INTERNAL,
            format!("formal CBOR encoding failed: {error}"),
        )
    })
}

fn domain_hash(domain: &[u8], payload: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update([0]);
    hasher.update(payload);
    hasher.finalize().into()
}

fn hex_digest(digest: [u8; 32]) -> String {
    hex_encode(&digest)
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        result.push(HEX[(byte >> 4) as usize] as char);
        result.push(HEX[(byte & 0x0f) as usize] as char);
    }
    result
}

/// Run the frozen target-free complete-closure diagnostic. The three roots are
/// embedded verbatim in every CanonicalProgramRecordV2.
pub fn enumerate_complete_diagnostic(
    child_dsl_spec_root: [u8; 32],
    operator_semantics_root: [u8; 32],
    identifier_registry_root: [u8; 32],
) -> Result<EnumerationArtifacts, EnumerationError> {
    if child_dsl_spec_root != FROZEN_CHILD_DSL_SPEC_ROOT
        || operator_semantics_root != FROZEN_OPERATOR_SEMANTICS_ROOT
        || identifier_registry_root != FROZEN_IDENTIFIER_REGISTRY_ROOT
    {
        return Err(EnumerationError::new(
            FAIL_ENUMERATION_BINDING,
            "roots differ from NON_FORMAL_SYNTHETIC_CHILD_BINDINGS_V1",
        ));
    }
    Enumerator::new(
        [
            child_dsl_spec_root,
            operator_semantics_root,
            identifier_registry_root,
        ],
        EnumeratorLimits::DIAGNOSTIC,
    )
    .enumerate()
}

/// Deterministic source-binding material.  The final ImplementationBindingV1
/// additionally binds Git source/dependency roots, binary digest, OCI
/// environment and exact basis commit outside this target-independent engine.
#[derive(Debug, Serialize)]
pub struct ImplementationBindingMaterial {
    pub implementation_id: u64,
    pub implementation_machine_id: &'static str,
    pub profile_id: &'static str,
    pub claim_level: &'static str,
    pub binding_profile_id: &'static str,
    pub entrypoint: &'static str,
    pub dsl_version: &'static str,
    pub freeze_version: &'static str,
    pub parent_dsl_version: &'static str,
    pub parent_freeze_version: &'static str,
    pub human_amendment_id: &'static str,
    pub shrink_step_id: &'static str,
    pub strict_qualification_source_commit: &'static str,
    pub strict_qualification_evidence_commit: &'static str,
    pub strict_qualification_artifact_path: &'static str,
    pub strict_qualification_artifact_sha256: &'static str,
    pub strict_qualification_diagnostic_report_hash: &'static str,
    pub strict_qualification_status: &'static str,
    pub maximum_ast_node_count: u32,
    pub maximum_top_level_clauses: usize,
    pub and3_generator_attempts_allowed: bool,
    pub and3_raw_operator_application_count: u64,
    pub canonicalizer_profile: &'static str,
    pub mdl_code_table_id: &'static str,
    pub source_surface_rule: &'static str,
    pub traversal_order: [&'static str; 5],
    pub bucket_record_order: [&'static str; 3],
    pub child_dsl_spec_root: String,
    pub operator_semantics_root: String,
    pub identifier_registry_root: String,
    pub canonical_ast_schema_root: String,
    pub canonical_cbor_profile_root: String,
    pub target_independent: bool,
    pub role_evaluation_supported: bool,
}

pub fn implementation_binding_material() -> ImplementationBindingMaterial {
    ImplementationBindingMaterial {
        implementation_id: IMPLEMENTATION_ID,
        implementation_machine_id: IMPLEMENTATION_MACHINE_ID,
        profile_id: PROFILE_ID,
        claim_level: CLAIM_LEVEL,
        binding_profile_id: BINDING_PROFILE_ID,
        entrypoint: "hegel-m3-closure-enumerator-shrink5 --enumerate-diagnostic",
        dsl_version: DSL_VERSION,
        freeze_version: FREEZE_VERSION,
        parent_dsl_version: PARENT_DSL_VERSION,
        parent_freeze_version: PARENT_FREEZE_VERSION,
        human_amendment_id: HUMAN_AMENDMENT_ID,
        shrink_step_id: SHRINK_STEP_ID,
        strict_qualification_source_commit: STRICT_QUALIFICATION_SOURCE_COMMIT,
        strict_qualification_evidence_commit: STRICT_QUALIFICATION_EVIDENCE_COMMIT,
        strict_qualification_artifact_path: STRICT_QUALIFICATION_ARTIFACT_PATH,
        strict_qualification_artifact_sha256: STRICT_QUALIFICATION_ARTIFACT_SHA256,
        strict_qualification_diagnostic_report_hash:
            STRICT_QUALIFICATION_DIAGNOSTIC_REPORT_HASH,
        strict_qualification_status: STRICT_QUALIFICATION_STATUS,
        maximum_ast_node_count: MAX_NODE_COUNT,
        maximum_top_level_clauses: MAX_TOP_LEVEL_CLAUSES,
        and3_generator_attempts_allowed: false,
        and3_raw_operator_application_count: 0,
        canonicalizer_profile: CANONICALIZER_PROFILE,
        mdl_code_table_id: MDL_CODE_TABLE_ID,
        source_surface_rule: "canonical-typed-source-token-surface-v1",
        traversal_order: [
            "ast_depth",
            "ast_node_count",
            "output_sort_id",
            "root_operator_id",
            "canonical_ast_cbor_bytes",
        ],
        bucket_record_order: ["output_sort_id", "ast_depth", "ast_node_count"],
        child_dsl_spec_root: hex_digest(FROZEN_CHILD_DSL_SPEC_ROOT),
        operator_semantics_root: hex_digest(FROZEN_OPERATOR_SEMANTICS_ROOT),
        identifier_registry_root: hex_digest(FROZEN_IDENTIFIER_REGISTRY_ROOT),
        canonical_ast_schema_root: hex_digest(CANONICAL_AST_SCHEMA_ROOT),
        canonical_cbor_profile_root: hex_digest(CANONICAL_CBOR_PROFILE_ROOT),
        target_independent: true,
        role_evaluation_supported: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formal_core::decode_strict_cbor;
    use hegel_strict_canonicalizer_shrink5::decode_shrink5_canonical_ast;
    use std::collections::BTreeSet;

    fn diagnostic_roots() -> [[u8; 32]; 3] {
        [[0x11; 32], [0x22; 32], [0x33; 32]]
    }

    #[test]
    fn scope_extension_surface_is_exactly_33_and_canonical() {
        let extensions = all_scope_extensions();
        assert_eq!(extensions.len(), 33);
        assert_eq!(extensions.iter().collect::<BTreeSet<_>>().len(), 33);
        assert!(extensions.iter().all(|extension| extension.len() <= 2));
        assert!(extensions.iter().all(|extension| extension
            .windows(2)
            .all(|pair| pair[0].0 < pair[1].0)));
    }

    #[test]
    fn exact_leaf_surface_has_810_programs_and_no_tombstones() {
        let mut enumerator = Enumerator::new(
            diagnostic_roots(),
            EnumeratorLimits {
                canonical_budget: 10_000,
                raw_budget: 20_000,
                max_depth: 0,
                max_node_count: 1,
            },
        );
        let mut count = 0;
        for sort in 1..=5 {
            let key = BucketKey {
                output_sort_id: sort,
                depth: 0,
                node_count: 1,
            };
            let programs = enumerator.generate_bucket(key).unwrap();
            for program in programs.values() {
                let decoded = decode_shrink5_canonical_ast(&program.canonical_cbor).unwrap();
                if let Node::Aggregate { map_id, .. } = decoded.canonical_node {
                    assert!([0, 1, 5].contains(&map_id));
                }
                if let Node::ScalarConst(parameter_id) = decoded.canonical_node {
                    assert!(ACTIVE_RATIONAL_PARAMETER_IDS.contains(&parameter_id));
                    assert!(!TOMBSTONED_RATIONAL_PARAMETER_IDS.contains(&parameter_id));
                }
            }
            count += programs.len();
        }
        assert_eq!(count, 810);
        assert_eq!(enumerator.raw_total, 810);
    }

    #[test]
    fn binary_generator_omits_add_and_keeps_ordered_difference_pairs() {
        let mut enumerator = Enumerator::new(
            diagnostic_roots(),
            EnumeratorLimits {
                canonical_budget: 100,
                raw_budget: 100,
                max_depth: 2,
                max_node_count: 5,
            },
        );
        let child_key = BucketKey {
            output_sort_id: 5,
            depth: 1,
            node_count: 2,
        };
        let mut child_local = BTreeMap::new();
        for slot in [0, 1] {
            enumerator
                .try_source(
                    child_key,
                    Node::Unary {
                        op: UnaryOp::BitToScalar,
                        child: Box::new(Node::BitAt(slot)),
                    },
                    &mut child_local,
                )
                .unwrap();
        }
        let child_pool = child_local
            .into_values()
            .map(|entry| {
                enumerator.known_cbor.insert(entry.canonical_cbor.clone());
                Arc::new(entry)
            })
            .collect::<Vec<_>>();
        assert_eq!(child_pool.len(), 2);
        enumerator.pools.insert(child_key, child_pool);

        let target_key = BucketKey {
            output_sort_id: 5,
            depth: 2,
            node_count: 5,
        };
        let generated = enumerator.generate_bucket(target_key).unwrap();
        assert_eq!(
            enumerator.counters[&target_key].raw_operator_applications,
            4,
            "two ordered children must produce exactly 2x2 difference attempts"
        );
        assert!(!generated.is_empty());
        assert!(generated
            .values()
            .all(|program| program.root_operator_id == 0x0201));

        let removed = canonicalize_shrink5_source_node(Node::Binary {
            op: BinaryOp::Add,
            left: Box::new(Node::Unary {
                op: UnaryOp::BitToScalar,
                child: Box::new(Node::BitAt(0)),
            }),
            right: Box::new(Node::ScalarConst(1)),
        })
        .unwrap_err();
        assert_eq!(removed.code, REJECT_REMOVED_BINARY_OPERATOR);
    }

    #[test]
    fn conjunction_generator_constructs_only_and2_and_never_attempts_and3() {
        let mut enumerator = Enumerator::new(
            diagnostic_roots(),
            EnumeratorLimits {
                canonical_budget: 100,
                raw_budget: 100,
                max_depth: 1,
                max_node_count: 4,
            },
        );
        let child_key = BucketKey {
            output_sort_id: 1,
            depth: 0,
            node_count: 1,
        };
        let children = enumerator
            .generate_bucket(child_key)
            .unwrap()
            .into_values()
            .map(Arc::new)
            .collect::<Vec<_>>();
        enumerator.pools.insert(child_key, children);

        let and2_key = BucketKey {
            output_sort_id: 1,
            depth: 1,
            node_count: 3,
        };
        let and2 = enumerator.generate_bucket(and2_key).unwrap();
        assert!(!and2.is_empty());
        assert!(and2.values().all(|program| {
            matches!(&program.node, Node::And(atoms) if atoms.len() == MAX_TOP_LEVEL_CLAUSES)
        }));
        let raw_after_and2 = enumerator.raw_total;

        let former_and3_key = BucketKey {
            output_sort_id: 1,
            depth: 1,
            node_count: 4,
        };
        let former_and3 = enumerator.generate_bucket(former_and3_key).unwrap();
        assert!(former_and3.is_empty());
        assert_eq!(enumerator.raw_total, raw_after_and2);
        assert_eq!(
            enumerator.counters[&former_and3_key].raw_operator_applications,
            0
        );
    }

    #[test]
    fn dsl_too_large_requires_both_exact_frozen_budgets() {
        assert_eq!(
            boundary_status(false, EnumeratorLimits::DIAGNOSTIC),
            ("DSL_TOO_LARGE", 2)
        );
        assert_eq!(
            boundary_status(
                false,
                EnumeratorLimits {
                    canonical_budget: MAX_CANONICAL_PROGRAMS,
                    raw_budget: MAX_RAW_OPERATOR_APPLICATIONS - 1,
                    max_depth: MAX_DEPTH,
                    max_node_count: MAX_NODE_COUNT,
                },
            ),
            ("DIAGNOSTIC_PREFIX_BUDGET_EXCEEDED", 0)
        );
        assert_eq!(
            boundary_status(
                false,
                EnumeratorLimits {
                    canonical_budget: MAX_CANONICAL_PROGRAMS - 1,
                    raw_budget: MAX_RAW_OPERATOR_APPLICATIONS,
                    max_depth: MAX_DEPTH,
                    max_node_count: MAX_NODE_COUNT,
                },
            ),
            ("DIAGNOSTIC_PREFIX_BUDGET_EXCEEDED", 0)
        );
    }

    #[test]
    fn mdl_integer_lengths_match_frozen_code_table() {
        assert_eq!(mdl_bit_length(&Node::ScalarConst(1)).unwrap(), 8);
        assert_eq!(
            mdl_bit_length(&Node::ScalarConst(0)).unwrap_err().code,
            FAIL_PROGRAM_MDL_LENGTH_UNAVAILABLE
        );
        assert_eq!(mdl_bit_length(&Node::BitAt(0)).unwrap(), 6);
        assert_eq!(mdl_bit_length(&Node::BitAt(1)).unwrap(), 9);
        assert_eq!(mdl_bit_length(&Node::SetSize).unwrap(), 5);
        assert_eq!(
            mdl_bit_length(&Node::Aggregate {
                map_id: 5,
                scope_id: 3,
                quantity_id: 1,
                scope_extension: vec![(0, false), (3, true)],
            })
            .unwrap(),
            19
        );
        assert_eq!(
            mdl_bit_length(&Node::Unary {
                op: UnaryOp::BitToScalar,
                child: Box::new(Node::BitAt(0)),
            })
            .unwrap(),
            10
        );
    }

    #[test]
    fn diagnostic_small_budget_is_prefix_complete_and_wire_records_validate() {
        let artifacts = Enumerator::new(
            diagnostic_roots(),
            EnumeratorLimits {
                canonical_budget: 10,
                raw_budget: 20_000,
                max_depth: MAX_DEPTH,
                max_node_count: MAX_NODE_COUNT,
            },
        )
        .enumerate()
        .unwrap();
        assert_eq!(
            artifacts.report.closure_status,
            "DIAGNOSTIC_PREFIX_BUDGET_EXCEEDED"
        );
        assert_eq!(artifacts.report.closure_status_id, 0);
        assert_eq!(artifacts.report.canonical_program_count, 10);
        assert_eq!(
            artifacts.report.first_out_of_budget_program_ordinal_or_null,
            Some(11)
        );
        assert_eq!(artifacts.report.claim_level, CLAIM_LEVEL);
        assert!(artifacts.report.diagnostic_only);
        assert!(!artifacts.report.authoritative_claim_allowed);
        assert_eq!(artifacts.report.execution_state, "NOT_RUN");
        assert!(!artifacts.report.formal_roots_generated);
        assert!(artifacts.report.formal_roots.is_none());
        assert_eq!(artifacts.report.parent_dsl_version, PARENT_DSL_VERSION);
        assert_eq!(artifacts.report.parent_freeze_version, PARENT_FREEZE_VERSION);
        assert_eq!(artifacts.report.human_amendment_id, HUMAN_AMENDMENT_ID);
        assert_eq!(artifacts.report.shrink_step_id, SHRINK_STEP_ID);
        assert_eq!(
            artifacts.report.strict_qualification_source_commit,
            STRICT_QUALIFICATION_SOURCE_COMMIT
        );
        assert_eq!(
            artifacts.report.strict_qualification_evidence_commit,
            STRICT_QUALIFICATION_EVIDENCE_COMMIT
        );
        assert_eq!(
            artifacts.report.strict_qualification_status,
            STRICT_QUALIFICATION_STATUS
        );
        assert_eq!(artifacts.report.maximum_top_level_clauses, 2);
        assert!(!artifacts.report.and3_generator_attempts_allowed);
        assert_eq!(artifacts.report.and3_raw_operator_application_count, 0);
        assert_eq!(
            serde_json::to_value(&artifacts.report)
                .unwrap()
                .as_object()
                .unwrap()
                .len(),
            75
        );
        assert_eq!(artifacts.report.reserved_rational_parameter_ids, [7]);
        assert_eq!(
            artifacts.report.active_source_binary_operator_ids,
            [1, 2, 3, 4, 5, 6]
        );
        assert_eq!(
            artifacts
                .report
                .active_formal_canonical_binary_operator_ids,
            [1, 2, 3, 5, 6]
        );
        assert_eq!(artifacts.report.source_alias_binary_operator_ids, [4]);
        assert_eq!(artifacts.report.tombstoned_binary_operator_ids, [0]);
        assert_eq!(artifacts.report.reserved_binary_operator_ids, [7]);
        assert!(!artifacts.report.operator_id_compaction_performed);
        assert!(!artifacts.report.automatic_operator_migration_performed);
        assert_eq!(artifacts.canonical_program_records.len(), 10);
        assert_eq!(artifacts.program_chunk_manifests.len(), 1);
        assert_eq!(artifacts.bucket_accounting_records.len(), 150);
        assert!(artifacts
            .report
            .first_out_of_budget_program_hash_or_null
            .is_some());
        for record in artifacts
            .canonical_program_records
            .iter()
            .chain(artifacts.program_chunk_manifests.iter())
            .chain(artifacts.bucket_accounting_records.iter())
        {
            decode_strict_cbor(record).unwrap();
        }
    }

    #[test]
    fn reduced_budget_prefix_is_self_consistent_without_frozen_observed_values() {
        let artifacts = Enumerator::new(
            [
                FROZEN_CHILD_DSL_SPEC_ROOT,
                FROZEN_OPERATOR_SEMANTICS_ROOT,
                FROZEN_IDENTIFIER_REGISTRY_ROOT,
            ],
            EnumeratorLimits {
                canonical_budget: 100,
                raw_budget: 100_000,
                max_depth: MAX_DEPTH,
                max_node_count: MAX_NODE_COUNT,
            },
        )
        .enumerate()
        .unwrap();
        assert_eq!(artifacts.report.canonical_program_count, 100);
        assert_eq!(
            artifacts.report.first_out_of_budget_program_ordinal_or_null,
            Some(101)
        );
        assert_eq!(
            artifacts.report.canonical_program_archive_root_or_null,
            Some(hex_digest(rfc6962_root(&artifacts.canonical_program_records)))
        );
        assert_eq!(
            artifacts.report.program_chunk_manifest_root_or_null,
            Some(hex_digest(rfc6962_root(&artifacts.program_chunk_manifests)))
        );
        assert_eq!(
            artifacts.report.bucket_accounting_root_or_null,
            Some(hex_digest(rfc6962_root(&artifacts.bucket_accounting_records)))
        );
        assert_eq!(artifacts.report.and3_raw_operator_application_count, 0);
    }

    #[test]
    fn public_diagnostic_rejects_unregistered_binding_roots_before_enumeration() {
        let error = enumerate_complete_diagnostic([0x11; 32], [0x22; 32], [0x33; 32])
            .unwrap_err();
        assert_eq!(error.code, FAIL_ENUMERATION_BINDING);
    }

    #[test]
    fn binding_material_uses_domain_separated_shrink5_roots() {
        let material = implementation_binding_material();
        assert_eq!(
            material.child_dsl_spec_root,
            "3340b3278caf562b560cc30cd14d3cd5f1d628e222b43d29d9d1e41b379f5675"
        );
        assert_eq!(
            material.operator_semantics_root,
            "5d2700884ae7125b9712a2bd06aa929feaf2fad1d4bfcd4fa5953c157a720ee1"
        );
        assert_eq!(
            material.identifier_registry_root,
            "1b0c141126b278778009d3ebbbf49f5de231ad0166a88a8a9caf367b35bff8ef"
        );
        assert_eq!(
            material.canonical_ast_schema_root,
            "828fdcc9f16ebd590702ff4297cac6f6ffa19b01299ea7a93753a4fced0961c5"
        );
        assert_eq!(
            material.canonical_cbor_profile_root,
            "0ccbd740c0b1f6a39fb8151ea56e114561093ee4fccb228bf83a9294e0bae783"
        );
        assert_eq!(material.dsl_version, DSL_VERSION);
        assert_eq!(material.parent_dsl_version, PARENT_DSL_VERSION);
        assert_eq!(material.maximum_ast_node_count, 6);
        assert_eq!(material.maximum_top_level_clauses, 2);
        assert_eq!(
            material.strict_qualification_evidence_commit,
            STRICT_QUALIFICATION_EVIDENCE_COMMIT
        );
        assert_eq!(material.strict_qualification_status, STRICT_QUALIFICATION_STATUS);
        assert!(material.target_independent);
        assert!(!material.role_evaluation_supported);
    }

    #[test]
    fn raw_cap_is_a_fail_closed_error_and_produces_no_artifacts() {
        let error = Enumerator::new(
            diagnostic_roots(),
            EnumeratorLimits {
                canonical_budget: 10,
                raw_budget: 1,
                max_depth: MAX_DEPTH,
                max_node_count: MAX_NODE_COUNT,
            },
        )
        .enumerate()
        .unwrap_err();
        assert_eq!(error.code, INCONCLUSIVE_BUDGET);
    }

    #[test]
    fn bucket_indices_are_output_sort_depth_node_order() {
        let first = BucketKey {
            output_sort_id: 1,
            depth: 0,
            node_count: 1,
        };
        let next_sort = BucketKey {
            output_sort_id: 2,
            depth: 0,
            node_count: 1,
        };
        assert_eq!(first.formal_index(), 0);
        assert_eq!(
            BucketKey {
                output_sort_id: 1,
                depth: 4,
                node_count: 6,
            }
            .formal_index(),
            29
        );
        assert_eq!(next_sort.formal_index(), 30);
    }

    #[test]
    fn formal_bucket_lattice_is_exactly_150_rows_without_node_seven() {
        let enumerator = Enumerator::new(diagnostic_roots(), EnumeratorLimits::DIAGNOSTIC);
        assert_eq!(enumerator.counters.len(), FORMAL_BUCKET_COUNT);
        assert!(enumerator
            .counters
            .keys()
            .all(|key| (1..=MAX_NODE_COUNT).contains(&key.node_count)));
        assert!(enumerator
            .counters
            .keys()
            .all(|key| key.node_count != 7));
        assert_eq!(
            enumerator
                .counters
                .values()
                .map(|counter| counter.bucket_index)
                .collect::<BTreeSet<_>>(),
            (0..FORMAL_BUCKET_COUNT as u64).collect::<BTreeSet<_>>()
        );
    }

    #[test]
    fn canonical_json_wire_is_recursive_sorted_compact_ascii_and_lf_terminated() {
        let value = serde_json::json!({
            "z": 1,
            "emoji": "\u{1f600}",
            "del": "\u{007f}",
            "a": {"z": "\u{4e2d}", "a": true},
        });
        assert_eq!(
            canonical_json_line(&value).unwrap(),
            b"{\"a\":{\"a\":true,\"z\":\"\\u4e2d\"},\"del\":\"\\u007f\",\"emoji\":\"\\ud83d\\ude00\",\"z\":1}\n"
        );
    }

    #[test]
    fn output_directory_is_exclusive_and_never_overwrites() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let artifacts = Enumerator::new(
            diagnostic_roots(),
            EnumeratorLimits {
                canonical_budget: 10,
                raw_budget: 20_000,
                max_depth: MAX_DEPTH,
                max_node_count: MAX_NODE_COUNT,
            },
        )
        .enumerate()
        .unwrap();
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "hegel-shrink5-exclusive-output-{}-{nonce}",
            std::process::id()
        ));
        artifacts.write_to_directory(&directory).unwrap();
        let report_path = directory.join("report.json");
        let before = fs::read(&report_path).unwrap();
        assert_eq!(before, canonical_json_line(&artifacts.report).unwrap());
        let error = artifacts.write_to_directory(&directory).unwrap_err();
        assert_eq!(error.code, FAIL_ENUMERATOR_INTERNAL);
        assert_eq!(fs::read(&report_path).unwrap(), before);
        fs::remove_dir_all(&directory).unwrap();
    }
}
