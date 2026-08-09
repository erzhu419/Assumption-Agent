//! Independent Rust strict admission profile for `hegel-old-dsl-v1.5.0`.
//!
//! Shrink step 5 changes one normalized structural bound:
//! `max_total_node_count` decreases from seven to six.  The machine report
//! field for the same bound is `maximum_ast_node_count`.  The shrink-4 limit
//! of two top-level clauses and every syntax, typing, registry, tombstone,
//! normalization, strict-CBOR, and rejection-priority rule are inherited.
//!
//! The ordering is deliberate: the complete shrink-4 strict path runs first.
//! Only an otherwise accepted normalized/canonical parent program reaches the
//! new node-count gate.  Every survivor therefore retains byte/hash identity.

use hegel_strict_canonicalizer::{
    encode_strict_cbor_json, BinaryOp, CanonicalProgram, Node, Sort,
    REJECT_MALFORMED_SOURCE_AST, REJECT_NONCANONICAL_AST, REJECT_STRUCTURAL_LIMIT,
    REJECT_TYPE_MISMATCH,
};
use hegel_strict_canonicalizer_shrink2::{
    REJECT_REMOVED_AGGREGATE_MAP, REJECT_REMOVED_RATIONAL_PARAMETER,
};
use hegel_strict_canonicalizer_shrink3::{
    ACTIVE_BINARY_OPERATOR_IDS_FORMAL, ACTIVE_BINARY_OPERATOR_IDS_SOURCE,
    ACTIVE_RATIONAL_PARAMETER_IDS, REJECT_REMOVED_BINARY_OPERATOR,
    RESERVED_BINARY_OPERATOR_IDS, TOMBSTONED_BINARY_OPERATOR_IDS,
};
use hegel_strict_canonicalizer_shrink4::{
    canonicalize_shrink4_source_json, canonicalize_shrink4_source_node,
    decode_shrink4_canonical_ast, Shrink4Error,
};
use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub const DSL_VERSION: &str = "hegel-old-dsl-v1.5.0";
pub const FREEZE_VERSION: &str = "hegel-freeze-p2b-p3-v1.5.0";
pub const PARENT_DSL_VERSION: &str = "hegel-old-dsl-v1.4.0";
pub const PARENT_FREEZE_VERSION: &str = "hegel-freeze-p2b-p3-v1.4.0";
pub const HUMAN_AMENDMENT_ID: &str = "hegel-freeze-p2b-p3-v1.5.0-shrink-step5";
pub const SHRINK_STEP_ID: &str = "SHRINK_STEP_5_REDUCE_MAX_TOTAL_NODE_COUNT_7_TO_6";
pub const AST_SCHEMA_ID: &str = "hegel-canonical-ast-v1";
pub const CBOR_PROFILE_ID: &str = "hegel-cbor-det-v1";
pub const AST_HASH_DOMAIN: &str = "HEGEL/AST/V1";
pub const REPLAY_SCHEMA_VERSION: &str = "hegel-strict-canonicalizer-shrink5-replay/1";
pub const GOLDEN_SCHEMA_VERSION: &str = "hegel-strict-canonicalizer-shrink5-golden/1";
pub const CAPACITY_SCHEMA_VERSION: &str = "hegel-strict-capacity-replay-shrink5/1";
pub const PARENT_MAXIMUM_AST_NODE_COUNT: u32 = 7;
pub const MAXIMUM_AST_NODE_COUNT: u32 = 6;
pub const MAXIMUM_TOP_LEVEL_CLAUSES: usize = 2;
pub const REJECT_INTERNAL_SHRINK5_REPLAY: &str = "REJECT_INTERNAL_SHRINK5_REPLAY";

const GOLDEN_MANIFEST_DOMAIN: &[u8] = b"HEGEL/SHRINK5/STRICT_GOLDEN_MANIFEST/V1";
const GOLDEN_OUTCOME_DOMAIN: &[u8] = b"HEGEL/SHRINK5/STRICT_GOLDEN_OUTCOME/V1";
const SURVIVOR_CAPACITY_SET_DOMAIN: &[u8] =
    b"HEGEL/SHRINK5/STRICT_SURVIVOR_CAPACITY_SET/V1";
const PARENT_ONLY_NODE7_SET_DOMAIN: &[u8] =
    b"HEGEL/SHRINK5/STRICT_PARENT_ONLY_NODE7_SET/V1";
const PARENT_ONLY_SOURCE_REJECTION_DOMAIN: &[u8] =
    b"HEGEL/SHRINK5/STRICT_PARENT_ONLY_NODE7_SOURCE_REJECTION/V1";
const PARENT_ONLY_FORMAL_REJECTION_DOMAIN: &[u8] =
    b"HEGEL/SHRINK5/STRICT_PARENT_ONLY_NODE7_FORMAL_REJECTION/V1";
const ACCEPT_PARENT_IDENTITY: &str = "ACCEPT_PARENT_IDENTITY";

// These constants are replaced with the values independently replayed from
// the exact vector and capacity manifests below.  Keeping them in the strict
// implementation turns accidental vector drift into fail-closed rejection.
pub const EXPECTED_GOLDEN_MANIFEST_ROOT: &str =
    "sha256:156f7e20407437bb753b097a87932f469701d1de6d1d577b0fa1b7a98f47e52e";
pub const EXPECTED_GOLDEN_OUTCOME_ROOT: &str =
    "sha256:8f82178c0f33d5295601d2e112b0b6e25ef18d73e5fc35d8d601024c1f0ddf94";
pub const EXPECTED_SURVIVOR_CAPACITY_SET_COMMITMENT: &str =
    "sha256:f5ab7f079ad943d65a74881eb59c7bb46385e1c437ca8ab036bb071dfa3874ac";
pub const EXPECTED_PARENT_ONLY_NODE7_SET_COMMITMENT: &str =
    "sha256:7e0e8780149f03ce85723408f7e3eff2cd684e8938896125cf8e34be9ac70b5e";
pub const EXPECTED_PARENT_ONLY_SOURCE_REJECTION_COMMITMENT: &str =
    "sha256:8617b56bdfa347f11f2c68b6a41f0992652f1e23e6d651017b17eb50169a9f39";
pub const EXPECTED_PARENT_ONLY_FORMAL_REJECTION_COMMITMENT: &str =
    "sha256:9a6b489ed90960008aebbecdbcf0bc5cf1595b7a8206d179bbe898540dabf617";
pub const EXPECTED_FIRST_SURVIVOR_CANONICAL_CBOR_HEX: &str =
    "82018402028300000183000001";
pub const EXPECTED_FIRST_SURVIVOR_CANONICAL_AST_HASH: &str =
    "sha256:d917dbe2b23af1af68d789536914f959adb10dc4b1fde3db970ea04adc8d51f0";
pub const EXPECTED_LAST_SURVIVOR_CANONICAL_CBOR_HEX: &str =
    "820186000305030180";
pub const EXPECTED_LAST_SURVIVOR_CANONICAL_AST_HASH: &str =
    "sha256:e35153f2bdd1a6e25d629ed3ab9afb178bb45ecd163efba4960a2a69db40ce2c";

pub const ORDERED_GOLDEN_VECTOR_IDS: [&str; 22] = [
    "S01", "S02", "S03", "N01", "N02", "L01", "L02", "P01", "P02", "P03", "P04",
    "P05", "F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08", "F09", "F10",
];

pub const EXPECTED_PARENT_BOUNDARY_SOURCE_COUNT: usize = 2_160;
pub const CAPACITY_GENERATOR_RULE: &str = concat!(
    "replay the exact 175 inherited target-free atom survivors and require ",
    "identical parent/child canonical bytes, hashes, and MDL lengths; also ",
    "replay the exact 2160-source shrink-4 AND2 capacity set, require every ",
    "parent canonical AST to contain exactly seven nodes, and require ",
    "REJECT_STRUCTURAL_LIMIT at both shrink-5 source and formal boundaries"
);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shrink5Error {
    pub code: String,
    pub message: String,
}

impl Shrink5Error {
    fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }
}

impl From<Shrink4Error> for Shrink5Error {
    fn from(error: Shrink4Error) -> Self {
        Self::new(error.code, error.message)
    }
}

impl From<hegel_strict_canonicalizer::CanonicalError> for Shrink5Error {
    fn from(error: hegel_strict_canonicalizer::CanonicalError) -> Self {
        Self::new(error.code, error.message)
    }
}

impl fmt::Display for Shrink5Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for Shrink5Error {}

fn enforce_shrink5_node_limit(
    program: CanonicalProgram,
) -> Result<CanonicalProgram, Shrink5Error> {
    if program.node_count > MAXIMUM_AST_NODE_COUNT {
        return Err(Shrink5Error::new(
            REJECT_STRUCTURAL_LIMIT,
            format!(
                "AST has {} nodes; maximum is {MAXIMUM_AST_NODE_COUNT}",
                program.node_count
            ),
        ));
    }
    Ok(program)
}

/// Run the complete shrink-4 source path before applying the sole step-5 gate.
pub fn canonicalize_shrink5_source_json(
    value: &Value,
) -> Result<CanonicalProgram, Shrink5Error> {
    enforce_shrink5_node_limit(canonicalize_shrink4_source_json(value)?)
}

/// Apply the same ordered gate to an already parsed source node.
pub fn canonicalize_shrink5_source_node(
    source: Node,
) -> Result<CanonicalProgram, Shrink5Error> {
    enforce_shrink5_node_limit(canonicalize_shrink4_source_node(source)?)
}

/// Decode exact formal CBOR through shrink-4 before measuring the node bound.
pub fn decode_shrink5_canonical_ast(
    bytes: &[u8],
) -> Result<CanonicalProgram, Shrink5Error> {
    enforce_shrink5_node_limit(decode_shrink4_canonical_ast(bytes)?)
}

pub fn sort_name(sort: Sort) -> &'static str {
    hegel_strict_canonicalizer_shrink4::sort_name(sort)
}

fn same_program_identity(left: &CanonicalProgram, right: &CanonicalProgram) -> bool {
    left.canonical_cbor == right.canonical_cbor
        && left.canonical_ast_hash == right.canonical_ast_hash
        && left.output_sort == right.output_sort
        && left.root_operator_id == right.root_operator_id
        && left.node_count == right.node_count
        && left.depth == right.depth
        && left.scalar_parameter_occurrence_count == right.scalar_parameter_occurrence_count
}

fn failure(message: impl Into<String>) -> Shrink5Error {
    Shrink5Error::new(REJECT_INTERNAL_SHRINK5_REPLAY, message)
}

fn framed_hash(domain: &[u8], rows: &[Vec<Vec<u8>>]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update([0]);
    for row in rows {
        for field in row {
            hasher.update((field.len() as u64).to_be_bytes());
            hasher.update(field);
        }
    }
    format!("sha256:{:x}", hasher.finalize())
}

fn accepted_outcome(program: &CanonicalProgram) -> Vec<u8> {
    let mut value = b"ACCEPT\x00".to_vec();
    value.extend_from_slice(&(program.canonical_cbor.len() as u64).to_be_bytes());
    value.extend_from_slice(&program.canonical_cbor);
    value.extend_from_slice(&program.canonical_ast_hash);
    value
}

fn rejected_outcome(code: &str) -> Vec<u8> {
    let mut value = b"REJECT\x00".to_vec();
    value.extend_from_slice(code.as_bytes());
    value
}

#[derive(Debug, Clone)]
enum GoldenInput {
    Source(Value),
    Formal(Value),
}

#[derive(Debug, Clone)]
struct GoldenVector {
    vector_id: &'static str,
    category: &'static str,
    input: GoldenInput,
    expected: &'static str,
}

fn source_vector(
    vector_id: &'static str,
    category: &'static str,
    value: Value,
    expected: &'static str,
) -> GoldenVector {
    GoldenVector {
        vector_id,
        category,
        input: GoldenInput::Source(value),
        expected,
    }
}

fn formal_vector(
    vector_id: &'static str,
    category: &'static str,
    value: Value,
    expected: &'static str,
) -> GoldenVector {
    GoldenVector {
        vector_id,
        category,
        input: GoldenInput::Formal(value),
        expected,
    }
}

fn atom(index: u64) -> Value {
    let name = match index {
        0 => "c0",
        1 => "c1",
        2 => "c2",
        3 => "c3",
        _ => return json!(["context_flag", index]),
    };
    json!(["context_flag", name])
}

fn formal_atom(index: u64) -> Value {
    json!([0, 4, index])
}

#[cfg(test)]
fn source_node6() -> Value {
    json!([
        "absolute",
        [
            "difference",
            ["bit_to_scalar", ["bit_at", 0]],
            ["bit_to_scalar", ["bit_at", 1]]
        ]
    ])
}

#[cfg(test)]
fn source_node7() -> Value {
    json!([
        "absolute",
        [
            "difference",
            ["bit_to_scalar", ["bit_at", 0]],
            ["absolute", ["bit_to_scalar", ["bit_at", 1]]]
        ]
    ])
}

#[cfg(test)]
fn formal_node7() -> Value {
    json!([
        1,
        [
            1,
            2,
            [
                2,
                1,
                [1, 0, [0, 1, 0]],
                [1, 2, [1, 0, [0, 1, 1]]]
            ]
        ]
    ])
}

fn golden_vectors() -> Vec<GoldenVector> {
    vec![
        source_vector(
            "S01",
            "surviving_identity_checks",
            json!(["scalar_const", 1]),
            ACCEPT_PARENT_IDENTITY,
        ),
        source_vector(
            "S02",
            "surviving_identity_checks",
            json!(["difference", ["scalar_const", 1], ["scalar_const", 5]]),
            ACCEPT_PARENT_IDENTITY,
        ),
        source_vector(
            "S03",
            "surviving_identity_checks",
            json!([
                "top_level_AND",
                atom(0),
                [
                    "less_equal",
                    ["scalar_const", 1],
                    [
                        "absolute",
                        ["aggregate", "sum_v1", "scope_all_observed_v1", "q0", []]
                    ]
                ]
            ]),
            ACCEPT_PARENT_IDENTITY,
        ),
        source_vector(
            "N01",
            "source_normalization_before_limit_checks",
            json!([
                "top_level_AND",
                atom(0),
                [
                    "less_equal",
                    ["difference", ["scalar_const", 1], ["scalar_const", 3]],
                    [
                        "absolute",
                        ["aggregate", "sum_v1", "scope_all_observed_v1", "q0", []]
                    ]
                ]
            ]),
            ACCEPT_PARENT_IDENTITY,
        ),
        source_vector(
            "N02",
            "source_normalization_before_limit_checks",
            json!([
                "top_level_AND",
                atom(0),
                [
                    "top_level_AND",
                    atom(0),
                    [
                        "less_equal",
                        ["scalar_const", 1],
                        [
                            "absolute",
                            [
                                "aggregate",
                                "sum_v1",
                                "scope_all_observed_v1",
                                "q0",
                                []
                            ]
                        ]
                    ]
                ]
            ]),
            ACCEPT_PARENT_IDENTITY,
        ),
        source_vector(
            "L01",
            "source_structural_limit_checks",
            json!([
                "top_level_AND",
                ["less_equal", ["scalar_const", 1], ["scalar_const", 3]],
                [
                    "less_equal",
                    ["scalar_const", 5],
                    ["aggregate", "sum_v1", "scope_all_observed_v1", "q0", []]
                ]
            ]),
            REJECT_STRUCTURAL_LIMIT,
        ),
        source_vector(
            "L02",
            "source_structural_limit_checks",
            json!([
                "top_level_AND",
                atom(0),
                [
                    "same_sign",
                    ["sign", ["scalar_const", 1]],
                    [
                        "sign",
                        ["aggregate", "sum_v1", "scope_all_observed_v1", "q0", []]
                    ]
                ]
            ]),
            REJECT_STRUCTURAL_LIMIT,
        ),
        source_vector(
            "P01",
            "source_priority_checks",
            json!([
                "top_level_AND",
                atom(0),
                [
                    "equal_exact",
                    ["aggregate", "mean_v1", "scope_all_observed_v1", "q0", []],
                    ["scalar_const", 1]
                ]
            ]),
            REJECT_REMOVED_AGGREGATE_MAP,
        ),
        source_vector(
            "P02",
            "source_priority_checks",
            json!([
                "top_level_AND",
                atom(0),
                [
                    "equal_exact",
                    ["scalar_const", -2, 1],
                    ["scalar_const", 1]
                ]
            ]),
            REJECT_REMOVED_RATIONAL_PARAMETER,
        ),
        source_vector(
            "P03",
            "source_priority_checks",
            json!([
                "top_level_AND",
                atom(0),
                [
                    "equal_exact",
                    ["add", ["scalar_const", 1], ["scalar_const", 5]],
                    ["scalar_const", 3]
                ]
            ]),
            REJECT_REMOVED_BINARY_OPERATOR,
        ),
        source_vector(
            "P04",
            "source_priority_checks",
            json!(["top_level_AND", atom(0), ["scalar_const", 1]]),
            REJECT_TYPE_MISMATCH,
        ),
        source_vector(
            "P05",
            "source_priority_checks",
            json!([
                "top_level_AND",
                atom(0),
                ["add", ["scalar_const", 1]]
            ]),
            REJECT_MALFORMED_SOURCE_AST,
        ),
        formal_vector(
            "F01",
            "formal_surviving_identity_checks",
            json!([
                1,
                [
                    4,
                    [
                        formal_atom(0),
                        [
                            2,
                            3,
                            [0, 0, 1],
                            [1, 2, [0, 3, 0, 0, 0, []]]
                        ]
                    ]
                ]
            ]),
            ACCEPT_PARENT_IDENTITY,
        ),
        formal_vector(
            "F02",
            "formal_structural_limit_checks",
            json!([
                1,
                [
                    4,
                    [
                        [2, 3, [0, 0, 1], [0, 0, 3]],
                        [2, 3, [0, 0, 5], [0, 3, 0, 0, 0, []]]
                    ]
                ]
            ]),
            REJECT_STRUCTURAL_LIMIT,
        ),
        formal_vector(
            "F03",
            "formal_priority_checks",
            json!([
                1,
                [
                    4,
                    [
                        [
                            2,
                            3,
                            [0, 0, 1],
                            [1, 2, [0, 3, 0, 0, 0, []]]
                        ],
                        formal_atom(0)
                    ]
                ]
            ]),
            REJECT_NONCANONICAL_AST,
        ),
        formal_vector(
            "F04",
            "formal_priority_checks",
            json!([
                1,
                [
                    4,
                    [
                        formal_atom(0),
                        [2, 2, [0, 3, 2, 0, 0, []], [0, 0, 1]]
                    ]
                ]
            ]),
            REJECT_REMOVED_AGGREGATE_MAP,
        ),
        formal_vector(
            "F05",
            "formal_priority_checks",
            json!([
                1,
                [
                    4,
                    [formal_atom(0), [2, 2, [0, 0, 0], [0, 0, 1]]]
                ]
            ]),
            REJECT_REMOVED_RATIONAL_PARAMETER,
        ),
        formal_vector(
            "F06",
            "formal_priority_checks",
            json!([
                1,
                [
                    4,
                    [
                        formal_atom(0),
                        [
                            2,
                            2,
                            [2, 0, [0, 0, 1], [0, 0, 5]],
                            [0, 0, 3]
                        ]
                    ]
                ]
            ]),
            REJECT_REMOVED_BINARY_OPERATOR,
        ),
        formal_vector(
            "F07",
            "formal_priority_checks",
            json!([1, [4, [formal_atom(0), [2, 4, [0, 0, 1], [0, 0, 5]]]]]),
            REJECT_NONCANONICAL_AST,
        ),
        formal_vector(
            "F08",
            "formal_priority_checks",
            json!([1, [4, [formal_atom(0), [2, 7, [0, 0, 1], [0, 0, 5]]]]]),
            REJECT_NONCANONICAL_AST,
        ),
        formal_vector(
            "F09",
            "formal_priority_checks",
            json!([1, [4, [formal_atom(0), [0, 0, 1]]]]),
            REJECT_TYPE_MISMATCH,
        ),
        formal_vector(
            "F10",
            "formal_priority_checks",
            json!([
                1,
                [
                    4,
                    [formal_atom(0), formal_atom(1), formal_atom(2), formal_atom(3)]
                ]
            ]),
            REJECT_NONCANONICAL_AST,
        ),
    ]
}

fn golden_input_wire(vector: &GoldenVector) -> Result<Vec<u8>, Shrink5Error> {
    match &vector.input {
        GoldenInput::Source(value) => serde_json::to_vec(value)
            .map_err(|error| failure(format!("golden source JSON encoding failed: {error}"))),
        GoldenInput::Formal(value) => Ok(encode_strict_cbor_json(value)?),
    }
}

fn golden_manifest_root(vectors: &[GoldenVector]) -> Result<String, Shrink5Error> {
    let rows = vectors
        .iter()
        .map(|vector| {
            Ok(vec![
                vector.vector_id.as_bytes().to_vec(),
                vector.category.as_bytes().to_vec(),
                match vector.input {
                    GoldenInput::Source(_) => b"SOURCE_JSON".to_vec(),
                    GoldenInput::Formal(_) => b"FORMAL_CBOR".to_vec(),
                },
                golden_input_wire(vector)?,
                vector.expected.as_bytes().to_vec(),
            ])
        })
        .collect::<Result<Vec<_>, Shrink5Error>>()?;
    Ok(framed_hash(GOLDEN_MANIFEST_DOMAIN, &rows))
}

fn increment_category(
    category: &str,
    surviving_identity_checks: &mut usize,
    source_normalization_before_limit_checks: &mut usize,
    source_structural_limit_checks: &mut usize,
    source_priority_checks: &mut usize,
    formal_surviving_identity_checks: &mut usize,
    formal_structural_limit_checks: &mut usize,
    formal_priority_checks: &mut usize,
) -> Result<(), Shrink5Error> {
    match category {
        "surviving_identity_checks" => *surviving_identity_checks += 1,
        "source_normalization_before_limit_checks" => {
            *source_normalization_before_limit_checks += 1
        }
        "source_structural_limit_checks" => *source_structural_limit_checks += 1,
        "source_priority_checks" => *source_priority_checks += 1,
        "formal_surviving_identity_checks" => *formal_surviving_identity_checks += 1,
        "formal_structural_limit_checks" => *formal_structural_limit_checks += 1,
        "formal_priority_checks" => *formal_priority_checks += 1,
        other => return Err(failure(format!("unknown golden category {other}"))),
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize)]
pub struct Shrink5GoldenReplayReport {
    pub schema_version: &'static str,
    pub implementation: &'static str,
    pub parent_dsl_version: &'static str,
    pub parent_freeze_version: &'static str,
    pub dsl_version: &'static str,
    pub freeze_version: &'static str,
    pub human_amendment_id: &'static str,
    pub shrink_step_id: &'static str,
    pub active_source_binary_operator_ids: [u64; 6],
    pub active_formal_binary_operator_ids: [u64; 5],
    pub source_alias_binary_operator_ids: [u64; 1],
    pub tombstoned_binary_operator_ids: [u64; 1],
    pub reserved_binary_operator_ids: [u64; 1],
    pub removed_binary_operator_error: &'static str,
    pub maximum_ast_node_count: u32,
    pub maximum_top_level_clauses: usize,
    pub vector_count: usize,
    pub passed_count: usize,
    pub surviving_identity_checks: usize,
    pub source_normalization_before_limit_checks: usize,
    pub source_structural_limit_checks: usize,
    pub source_priority_checks: usize,
    pub formal_surviving_identity_checks: usize,
    pub formal_structural_limit_checks: usize,
    pub formal_priority_checks: usize,
    pub golden_vector_manifest_root: String,
    pub golden_outcome_root: String,
    pub ordered_vector_ids: [&'static str; 22],
    pub execution_state: &'static str,
    pub closure_executed: bool,
    pub formal_roots_generated: bool,
    pub formal_roots: Option<String>,
    pub target_or_split_modules_loaded: bool,
}

/// Replay the exact 22-vector target-free shrink-5 strict manifest.
pub fn replay_shrink5_golden_vectors() -> Result<Shrink5GoldenReplayReport, Shrink5Error> {
    let vectors = golden_vectors();
    let ids = vectors
        .iter()
        .map(|vector| vector.vector_id)
        .collect::<Vec<_>>();
    if ids.as_slice() != ORDERED_GOLDEN_VECTOR_IDS {
        return Err(failure("golden vector ID/order drift"));
    }
    let manifest_root = golden_manifest_root(&vectors)?;
    let mut outcome_rows = Vec::with_capacity(vectors.len());
    let mut surviving_identity_checks = 0usize;
    let mut source_normalization_before_limit_checks = 0usize;
    let mut source_structural_limit_checks = 0usize;
    let mut source_priority_checks = 0usize;
    let mut formal_surviving_identity_checks = 0usize;
    let mut formal_structural_limit_checks = 0usize;
    let mut formal_priority_checks = 0usize;

    for vector in &vectors {
        let outcome = match &vector.input {
            GoldenInput::Source(value) if vector.expected == ACCEPT_PARENT_IDENTITY => {
                let parent = canonicalize_shrink4_source_json(value)?;
                let child = canonicalize_shrink5_source_json(value)?;
                if !same_program_identity(&parent, &child) {
                    return Err(failure(format!(
                        "{}: surviving source identity changed",
                        vector.vector_id
                    )));
                }
                accepted_outcome(&child)
            }
            GoldenInput::Formal(value) if vector.expected == ACCEPT_PARENT_IDENTITY => {
                let bytes = encode_strict_cbor_json(value)?;
                let parent = decode_shrink4_canonical_ast(&bytes)?;
                let child = decode_shrink5_canonical_ast(&bytes)?;
                if !same_program_identity(&parent, &child) {
                    return Err(failure(format!(
                        "{}: surviving formal identity changed",
                        vector.vector_id
                    )));
                }
                accepted_outcome(&child)
            }
            GoldenInput::Source(value) => {
                if vector.category == "source_structural_limit_checks" {
                    let parent = canonicalize_shrink4_source_json(value)?;
                    if parent.node_count != PARENT_MAXIMUM_AST_NODE_COUNT {
                        return Err(failure(format!(
                            "{}: source limit vector is not an accepted seven-node parent",
                            vector.vector_id
                        )));
                    }
                }
                match canonicalize_shrink5_source_json(value) {
                    Ok(_) => {
                        return Err(failure(format!(
                            "{}: source unexpectedly accepted",
                            vector.vector_id
                        )))
                    }
                    Err(error) if error.code == vector.expected => rejected_outcome(&error.code),
                    Err(error) => {
                        return Err(failure(format!(
                            "{}: expected {}, got {}",
                            vector.vector_id, vector.expected, error.code
                        )))
                    }
                }
            }
            GoldenInput::Formal(value) => {
                let bytes = encode_strict_cbor_json(value)?;
                if vector.category == "formal_structural_limit_checks" {
                    let parent = decode_shrink4_canonical_ast(&bytes)?;
                    if parent.node_count != PARENT_MAXIMUM_AST_NODE_COUNT {
                        return Err(failure(format!(
                            "{}: formal limit vector is not an accepted seven-node parent",
                            vector.vector_id
                        )));
                    }
                }
                match decode_shrink5_canonical_ast(&bytes) {
                    Ok(_) => {
                        return Err(failure(format!(
                            "{}: formal input unexpectedly accepted",
                            vector.vector_id
                        )))
                    }
                    Err(error) if error.code == vector.expected => rejected_outcome(&error.code),
                    Err(error) => {
                        return Err(failure(format!(
                            "{}: expected {}, got {}",
                            vector.vector_id, vector.expected, error.code
                        )))
                    }
                }
            }
        };
        outcome_rows.push(vec![vector.vector_id.as_bytes().to_vec(), outcome]);
        increment_category(
            vector.category,
            &mut surviving_identity_checks,
            &mut source_normalization_before_limit_checks,
            &mut source_structural_limit_checks,
            &mut source_priority_checks,
            &mut formal_surviving_identity_checks,
            &mut formal_structural_limit_checks,
            &mut formal_priority_checks,
        )?;
    }

    let outcome_root = framed_hash(GOLDEN_OUTCOME_DOMAIN, &outcome_rows);
    if manifest_root != EXPECTED_GOLDEN_MANIFEST_ROOT
        || outcome_root != EXPECTED_GOLDEN_OUTCOME_ROOT
    {
        return Err(failure(format!(
            "golden root drift: manifest={manifest_root}, outcome={outcome_root}"
        )));
    }

    Ok(Shrink5GoldenReplayReport {
        schema_version: GOLDEN_SCHEMA_VERSION,
        implementation: "rust",
        parent_dsl_version: PARENT_DSL_VERSION,
        parent_freeze_version: PARENT_FREEZE_VERSION,
        dsl_version: DSL_VERSION,
        freeze_version: FREEZE_VERSION,
        human_amendment_id: HUMAN_AMENDMENT_ID,
        shrink_step_id: SHRINK_STEP_ID,
        active_source_binary_operator_ids: ACTIVE_BINARY_OPERATOR_IDS_SOURCE,
        active_formal_binary_operator_ids: ACTIVE_BINARY_OPERATOR_IDS_FORMAL,
        source_alias_binary_operator_ids: [4],
        tombstoned_binary_operator_ids: TOMBSTONED_BINARY_OPERATOR_IDS,
        reserved_binary_operator_ids: RESERVED_BINARY_OPERATOR_IDS,
        removed_binary_operator_error: REJECT_REMOVED_BINARY_OPERATOR,
        maximum_ast_node_count: MAXIMUM_AST_NODE_COUNT,
        maximum_top_level_clauses: MAXIMUM_TOP_LEVEL_CLAUSES,
        vector_count: vectors.len(),
        passed_count: vectors.len(),
        surviving_identity_checks,
        source_normalization_before_limit_checks,
        source_structural_limit_checks,
        source_priority_checks,
        formal_surviving_identity_checks,
        formal_structural_limit_checks,
        formal_priority_checks,
        golden_vector_manifest_root: manifest_root,
        golden_outcome_root: outcome_root,
        ordered_vector_ids: ORDERED_GOLDEN_VECTOR_IDS,
        execution_state: "NOT_RUN",
        closure_executed: false,
        formal_roots_generated: false,
        formal_roots: None,
        target_or_split_modules_loaded: false,
    })
}

fn capacity_constant_atoms() -> Vec<Node> {
    let constants = ACTIVE_RATIONAL_PARAMETER_IDS
        .iter()
        .copied()
        .map(Node::ScalarConst)
        .collect::<Vec<_>>();
    let mut atoms = Vec::with_capacity(15);
    for left in 0..constants.len() {
        for right in left..constants.len() {
            atoms.push(Node::Binary {
                op: BinaryOp::EqualExact,
                left: Box::new(constants[left].clone()),
                right: Box::new(constants[right].clone()),
            });
        }
    }
    for left in &constants {
        for right in &constants {
            atoms.push(Node::Binary {
                op: BinaryOp::LessEqual,
                left: Box::new(left.clone()),
                right: Box::new(right.clone()),
            });
        }
    }
    atoms
}

fn capacity_rational_aggregates() -> Vec<Node> {
    let mut aggregates = Vec::with_capacity(16);
    for map_id in [0_u64, 5] {
        for scope_id in 0..4_u64 {
            for quantity_id in 0..2_u64 {
                aggregates.push(Node::Aggregate {
                    map_id,
                    scope_id,
                    quantity_id,
                    scope_extension: Vec::new(),
                });
            }
        }
    }
    aggregates
}

fn capacity_mixed_atoms() -> Vec<Node> {
    let constants = ACTIVE_RATIONAL_PARAMETER_IDS
        .iter()
        .copied()
        .map(Node::ScalarConst)
        .collect::<Vec<_>>();
    let aggregates = capacity_rational_aggregates();
    let mut atoms = Vec::with_capacity(144);
    for constant in &constants {
        for aggregate in &aggregates {
            atoms.push(Node::Binary {
                op: BinaryOp::EqualExact,
                left: Box::new(constant.clone()),
                right: Box::new(aggregate.clone()),
            });
        }
    }
    for constant in &constants {
        for aggregate in &aggregates {
            atoms.push(Node::Binary {
                op: BinaryOp::LessEqual,
                left: Box::new(constant.clone()),
                right: Box::new(aggregate.clone()),
            });
            atoms.push(Node::Binary {
                op: BinaryOp::LessEqual,
                left: Box::new(aggregate.clone()),
                right: Box::new(constant.clone()),
            });
        }
    }
    atoms
}

fn capacity_set_commitment(domain: &[u8], sorted_cbor: &BTreeSet<Vec<u8>>) -> String {
    let rows = sorted_cbor
        .iter()
        .map(|bytes| vec![bytes.clone()])
        .collect::<Vec<_>>();
    framed_hash(domain, &rows)
}

fn capacity_rejection_commitment(
    domain: &[u8],
    sorted_cbor: &BTreeSet<Vec<u8>>,
) -> String {
    let rows = sorted_cbor
        .iter()
        .map(|bytes| {
            vec![
                bytes.clone(),
                REJECT_STRUCTURAL_LIMIT.as_bytes().to_vec(),
            ]
        })
        .collect::<Vec<_>>();
    framed_hash(domain, &rows)
}

#[derive(Debug, Clone, Serialize)]
pub struct Shrink5CapacityReplayReport {
    pub canonical_program_budget: usize,
    pub complete_closure_enumerated: bool,
    pub constant_atom_count: usize,
    pub dsl_version: &'static str,
    pub executed_closure_status: &'static str,
    pub first_out_of_budget_ordinal: Option<usize>,
    pub first_survivor_canonical_ast_hash: String,
    pub first_survivor_canonical_cbor_hex: String,
    pub formal_roots: Option<String>,
    pub freeze_version: &'static str,
    pub generator_rule: &'static str,
    pub human_amendment_id: &'static str,
    pub implementation: &'static str,
    pub interpreted_as_complete_closure: bool,
    pub last_survivor_canonical_ast_hash: String,
    pub last_survivor_canonical_cbor_hex: String,
    pub maximum_ast_node_count: u32,
    pub maximum_top_level_clauses: usize,
    pub mixed_atom_count: usize,
    pub parent_dsl_version: &'static str,
    pub parent_freeze_version: &'static str,
    pub parent_only_formal_child_rejected_count: usize,
    pub parent_only_formal_child_rejection_counts: BTreeMap<String, usize>,
    pub parent_only_formal_rejection_outcome_commitment: String,
    pub parent_only_node_count: u32,
    pub parent_only_parent_accepted_count: usize,
    pub parent_only_set_commitment: String,
    pub parent_only_source_candidate_count: usize,
    pub parent_only_source_child_rejected_count: usize,
    pub parent_only_source_child_rejection_counts: BTreeMap<String, usize>,
    pub parent_only_source_rejection_outcome_commitment: String,
    pub rational_aggregate_count: usize,
    pub removed_binary_operator_ids: [u64; 1],
    pub retained_difference_id: u64,
    pub schema_version: &'static str,
    pub shrink_step_id: &'static str,
    pub subset_status: &'static str,
    pub survivor_accepted_count: usize,
    pub survivor_accepted_set_commitment: String,
    pub survivor_parent_identity_match_count: usize,
    pub survivor_rejected_count: usize,
    pub survivor_rejection_counts: BTreeMap<String, usize>,
    pub survivor_source_candidate_count: usize,
    pub survivor_unique_count: usize,
    pub target_or_split_modules_loaded: bool,
}

/// Replay the exact 175-program survivor set and inherited 2,160-program
/// parent-only boundary.  This constructive control is never closure.
pub fn replay_shrink5_capacity_subset(
) -> Result<Shrink5CapacityReplayReport, Shrink5Error> {
    let constant_atoms = capacity_constant_atoms();
    let rational_aggregates = capacity_rational_aggregates();
    let mixed_atoms = capacity_mixed_atoms();
    if constant_atoms.len() != 15 || rational_aggregates.len() != 16 || mixed_atoms.len() != 144 {
        return Err(failure("shrink-5 capacity component count drift"));
    }

    let survivor_sources = constant_atoms
        .iter()
        .chain(rational_aggregates.iter())
        .chain(mixed_atoms.iter())
        .cloned()
        .collect::<Vec<_>>();
    let mut survivor_accepted_count = 0usize;
    let mut survivor_parent_identity_match_count = 0usize;
    let mut survivor_rejection_counts = BTreeMap::new();
    let mut survivor_set = BTreeSet::new();
    for (index, source) in survivor_sources.iter().cloned().enumerate() {
        let parent_source = canonicalize_shrink4_source_node(source.clone())?;
        match canonicalize_shrink5_source_node(source) {
            Ok(child_source) => {
                let parent_formal =
                    decode_shrink4_canonical_ast(&parent_source.canonical_cbor)?;
                let child_formal =
                    decode_shrink5_canonical_ast(&parent_source.canonical_cbor)?;
                if !same_program_identity(&parent_source, &child_source)
                    || !same_program_identity(&parent_source, &parent_formal)
                    || !same_program_identity(&parent_source, &child_formal)
                {
                    return Err(failure(format!(
                        "survivor source/formal identity changed at ordinal {}",
                        index + 1
                    )));
                }
                survivor_accepted_count += 1;
                survivor_parent_identity_match_count += 1;
                survivor_set.insert(child_source.canonical_cbor);
            }
            Err(error) => {
                *survivor_rejection_counts.entry(error.code).or_insert(0) += 1;
            }
        }
    }

    let survivor_rejected_count: usize = survivor_rejection_counts.values().sum();
    let survivor_commitment =
        capacity_set_commitment(SURVIVOR_CAPACITY_SET_DOMAIN, &survivor_set);
    let first_survivor = survivor_set
        .iter()
        .next()
        .ok_or_else(|| failure("survivor set is empty"))?;
    let last_survivor = survivor_set
        .iter()
        .next_back()
        .ok_or_else(|| failure("survivor set is empty"))?;
    let first_survivor_program = decode_shrink5_canonical_ast(first_survivor)?;
    let last_survivor_program = decode_shrink5_canonical_ast(last_survivor)?;
    let first_survivor_hex = hegel_strict_canonicalizer::hex_encode(first_survivor);
    let last_survivor_hex = hegel_strict_canonicalizer::hex_encode(last_survivor);

    let mut parent_only_source_candidate_count = 0usize;
    let mut parent_only_parent_accepted_count = 0usize;
    let mut parent_only_source_child_rejection_counts = BTreeMap::new();
    let mut parent_only_formal_child_rejection_counts = BTreeMap::new();
    let mut parent_only_set = BTreeSet::new();
    for constant_atom in &constant_atoms {
        for mixed_atom in &mixed_atoms {
            parent_only_source_candidate_count += 1;
            let source = Node::And(vec![constant_atom.clone(), mixed_atom.clone()]);
            let parent = canonicalize_shrink4_source_node(source.clone())?;
            parent_only_parent_accepted_count += 1;
            if parent.node_count != PARENT_MAXIMUM_AST_NODE_COUNT {
                return Err(failure(format!(
                    "parent-only node count differs at source ordinal {parent_only_source_candidate_count}: {}",
                    parent.node_count
                )));
            }
            let parent_formal = decode_shrink4_canonical_ast(&parent.canonical_cbor)?;
            if !same_program_identity(&parent, &parent_formal) {
                return Err(failure(format!(
                    "parent-only formal identity differs at source ordinal {parent_only_source_candidate_count}"
                )));
            }
            parent_only_set.insert(parent.canonical_cbor.clone());
            match canonicalize_shrink5_source_node(source) {
                Ok(_) => {
                    return Err(failure(format!(
                        "parent-only source unexpectedly accepted at ordinal {parent_only_source_candidate_count}"
                    )))
                }
                Err(error) => {
                    *parent_only_source_child_rejection_counts
                        .entry(error.code)
                        .or_insert(0) += 1
                }
            }
            match decode_shrink5_canonical_ast(&parent.canonical_cbor) {
                Ok(_) => {
                    return Err(failure(format!(
                        "parent-only formal input unexpectedly accepted at ordinal {parent_only_source_candidate_count}"
                    )))
                }
                Err(error) => {
                    *parent_only_formal_child_rejection_counts
                        .entry(error.code)
                        .or_insert(0) += 1
                }
            }
        }
    }

    let parent_only_source_child_rejected_count: usize =
        parent_only_source_child_rejection_counts.values().sum();
    let parent_only_formal_child_rejected_count: usize =
        parent_only_formal_child_rejection_counts.values().sum();
    let parent_only_set_commitment =
        capacity_set_commitment(PARENT_ONLY_NODE7_SET_DOMAIN, &parent_only_set);
    let parent_only_source_rejection_outcome_commitment = capacity_rejection_commitment(
        PARENT_ONLY_SOURCE_REJECTION_DOMAIN,
        &parent_only_set,
    );
    let parent_only_formal_rejection_outcome_commitment = capacity_rejection_commitment(
        PARENT_ONLY_FORMAL_REJECTION_DOMAIN,
        &parent_only_set,
    );

    if survivor_sources.len() != 175
        || survivor_accepted_count != 175
        || survivor_parent_identity_match_count != 175
        || survivor_set.len() != 175
        || survivor_rejected_count != 0
        || !survivor_rejection_counts.is_empty()
        || survivor_commitment != EXPECTED_SURVIVOR_CAPACITY_SET_COMMITMENT
        || first_survivor_hex != EXPECTED_FIRST_SURVIVOR_CANONICAL_CBOR_HEX
        || first_survivor_program.canonical_ast_hash_id()
            != EXPECTED_FIRST_SURVIVOR_CANONICAL_AST_HASH
        || last_survivor_hex != EXPECTED_LAST_SURVIVOR_CANONICAL_CBOR_HEX
        || last_survivor_program.canonical_ast_hash_id()
            != EXPECTED_LAST_SURVIVOR_CANONICAL_AST_HASH
        || parent_only_source_candidate_count != EXPECTED_PARENT_BOUNDARY_SOURCE_COUNT
        || parent_only_parent_accepted_count != EXPECTED_PARENT_BOUNDARY_SOURCE_COUNT
        || parent_only_set.len() != EXPECTED_PARENT_BOUNDARY_SOURCE_COUNT
        || parent_only_source_child_rejected_count != EXPECTED_PARENT_BOUNDARY_SOURCE_COUNT
        || parent_only_formal_child_rejected_count != EXPECTED_PARENT_BOUNDARY_SOURCE_COUNT
        || parent_only_source_child_rejection_counts.len() != 1
        || parent_only_formal_child_rejection_counts.len() != 1
        || parent_only_source_child_rejection_counts.get(REJECT_STRUCTURAL_LIMIT)
            != Some(&EXPECTED_PARENT_BOUNDARY_SOURCE_COUNT)
        || parent_only_formal_child_rejection_counts.get(REJECT_STRUCTURAL_LIMIT)
            != Some(&EXPECTED_PARENT_BOUNDARY_SOURCE_COUNT)
        || parent_only_set_commitment != EXPECTED_PARENT_ONLY_NODE7_SET_COMMITMENT
        || parent_only_source_rejection_outcome_commitment
            != EXPECTED_PARENT_ONLY_SOURCE_REJECTION_COMMITMENT
        || parent_only_formal_rejection_outcome_commitment
            != EXPECTED_PARENT_ONLY_FORMAL_REJECTION_COMMITMENT
    {
        return Err(failure(format!(
            "capacity invariant failure: survivor_source={}, survivor_accepted={survivor_accepted_count}, survivor_unique={}, survivor_rejected={survivor_rejected_count}, survivor_commitment={survivor_commitment}, first={first_survivor_hex}, last={last_survivor_hex}, parent_source={parent_only_source_candidate_count}, parent_accepted={parent_only_parent_accepted_count}, parent_unique={}, source_rejected={parent_only_source_child_rejected_count}, formal_rejected={parent_only_formal_child_rejected_count}, parent_commitment={parent_only_set_commitment}, source_rejection_commitment={parent_only_source_rejection_outcome_commitment}, formal_rejection_commitment={parent_only_formal_rejection_outcome_commitment}",
            survivor_sources.len(),
            survivor_set.len(),
            parent_only_set.len(),
        )));
    }

    Ok(Shrink5CapacityReplayReport {
        canonical_program_budget: 50_000,
        complete_closure_enumerated: false,
        constant_atom_count: constant_atoms.len(),
        dsl_version: DSL_VERSION,
        executed_closure_status: "NOT_RUN",
        first_out_of_budget_ordinal: None,
        first_survivor_canonical_ast_hash: first_survivor_program.canonical_ast_hash_id(),
        first_survivor_canonical_cbor_hex: first_survivor_hex,
        formal_roots: None,
        freeze_version: FREEZE_VERSION,
        generator_rule: CAPACITY_GENERATOR_RULE,
        human_amendment_id: HUMAN_AMENDMENT_ID,
        implementation: "rust",
        interpreted_as_complete_closure: false,
        last_survivor_canonical_ast_hash: last_survivor_program.canonical_ast_hash_id(),
        last_survivor_canonical_cbor_hex: last_survivor_hex,
        maximum_ast_node_count: MAXIMUM_AST_NODE_COUNT,
        maximum_top_level_clauses: MAXIMUM_TOP_LEVEL_CLAUSES,
        mixed_atom_count: mixed_atoms.len(),
        parent_dsl_version: PARENT_DSL_VERSION,
        parent_freeze_version: PARENT_FREEZE_VERSION,
        parent_only_formal_child_rejected_count,
        parent_only_formal_child_rejection_counts,
        parent_only_formal_rejection_outcome_commitment,
        parent_only_node_count: PARENT_MAXIMUM_AST_NODE_COUNT,
        parent_only_parent_accepted_count,
        parent_only_set_commitment,
        parent_only_source_candidate_count,
        parent_only_source_child_rejected_count,
        parent_only_source_child_rejection_counts,
        parent_only_source_rejection_outcome_commitment,
        rational_aggregate_count: rational_aggregates.len(),
        removed_binary_operator_ids: TOMBSTONED_BINARY_OPERATOR_IDS,
        retained_difference_id: 1,
        schema_version: CAPACITY_SCHEMA_VERSION,
        shrink_step_id: SHRINK_STEP_ID,
        subset_status: "FULL_175_SURVIVOR_AND_2160_PARENT_NODE7_BOUNDARY_SETS_ONLY_NOT_COMPLETE",
        survivor_accepted_count,
        survivor_accepted_set_commitment: survivor_commitment,
        survivor_parent_identity_match_count,
        survivor_rejected_count,
        survivor_rejection_counts,
        survivor_source_candidate_count: survivor_sources.len(),
        survivor_unique_count: survivor_set.len(),
        target_or_split_modules_loaded: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn six_node_parent_survives_with_exact_identity() {
        let parent = canonicalize_shrink4_source_json(&source_node6()).unwrap();
        assert_eq!(parent.node_count, 6);
        let child = canonicalize_shrink5_source_json(&source_node6()).unwrap();
        assert!(same_program_identity(&parent, &child));
    }

    #[test]
    fn source_and_formal_seven_node_programs_reject() {
        let parent_source = canonicalize_shrink4_source_json(&source_node7()).unwrap();
        assert_eq!(parent_source.node_count, 7);
        assert_eq!(
            canonicalize_shrink5_source_json(&source_node7())
                .unwrap_err()
                .code,
            REJECT_STRUCTURAL_LIMIT
        );
        let bytes = encode_strict_cbor_json(&formal_node7()).unwrap();
        assert_eq!(decode_shrink4_canonical_ast(&bytes).unwrap().node_count, 7);
        assert_eq!(
            decode_shrink5_canonical_ast(&bytes).unwrap_err().code,
            REJECT_STRUCTURAL_LIMIT
        );
    }

    #[test]
    fn normalization_precedes_the_new_node_limit() {
        let source = json!(["top_level_AND", atom(0), atom(0), atom(1)]);
        let parent = canonicalize_shrink4_source_json(&source).unwrap();
        let child = canonicalize_shrink5_source_json(&source).unwrap();
        assert_eq!(child.node_count, 3);
        assert!(same_program_identity(&parent, &child));
    }

    #[test]
    fn inherited_source_errors_precede_the_new_node_limit() {
        for (source, expected) in [
            (
                json!(["top_level_AND", atom(0), atom(1), atom(2)]),
                REJECT_STRUCTURAL_LIMIT,
            ),
            (
                json!([
                    "top_level_AND",
                    atom(0),
                    [
                        "equal_exact",
                        ["add", ["scalar_const", 1], ["scalar_const", 5]],
                        ["absolute", ["bit_to_scalar", ["bit_at", 0]]]
                    ]
                ]),
                REJECT_REMOVED_BINARY_OPERATOR,
            ),
        ] {
            assert_eq!(
                canonicalize_shrink5_source_json(&source).unwrap_err().code,
                expected
            );
        }
    }

    #[test]
    fn formal_noncanonical_and_tombstone_priorities_survive() {
        let noncanonical = encode_strict_cbor_json(&json!([
            1,
            [4, [formal_atom(2), formal_atom(1), formal_atom(0)]]
        ]))
        .unwrap();
        assert_eq!(
            decode_shrink5_canonical_ast(&noncanonical).unwrap_err().code,
            REJECT_NONCANONICAL_AST
        );
        let removed = encode_strict_cbor_json(&json!([
            1,
            [
                2,
                2,
                [2, 0, [0, 0, 1], [0, 0, 5]],
                [1, 2, [1, 0, [0, 1, 0]]]
            ]
        ]))
        .unwrap();
        assert_eq!(
            decode_shrink5_canonical_ast(&removed).unwrap_err().code,
            REJECT_REMOVED_BINARY_OPERATOR
        );
    }

    #[test]
    fn golden_replay_covers_exact_twenty_two_vector_layout() {
        let report = replay_shrink5_golden_vectors().unwrap();
        assert_eq!(report.vector_count, 22);
        assert_eq!(report.passed_count, 22);
        assert_eq!(report.surviving_identity_checks, 3);
        assert_eq!(report.source_normalization_before_limit_checks, 2);
        assert_eq!(report.source_structural_limit_checks, 2);
        assert_eq!(report.source_priority_checks, 5);
        assert_eq!(report.formal_surviving_identity_checks, 1);
        assert_eq!(report.formal_structural_limit_checks, 1);
        assert_eq!(report.formal_priority_checks, 8);
        assert_eq!(report.maximum_ast_node_count, 6);
        assert_eq!(report.maximum_top_level_clauses, 2);
        assert_eq!(report.ordered_vector_ids, ORDERED_GOLDEN_VECTOR_IDS);
    }

    #[test]
    fn survivor_and_parent_only_capacity_sets_replay_but_are_not_closure() {
        let report = replay_shrink5_capacity_subset().unwrap();
        assert_eq!(report.survivor_source_candidate_count, 175);
        assert_eq!(report.survivor_accepted_count, 175);
        assert_eq!(report.survivor_unique_count, 175);
        assert_eq!(report.survivor_rejected_count, 0);
        assert_eq!(report.parent_only_source_candidate_count, 2_160);
        assert_eq!(report.parent_only_parent_accepted_count, 2_160);
        assert_eq!(report.parent_only_node_count, 7);
        assert_eq!(report.parent_only_source_child_rejected_count, 2_160);
        assert_eq!(report.parent_only_formal_child_rejected_count, 2_160);
        assert_eq!(
            report
                .parent_only_source_child_rejection_counts
                .get(REJECT_STRUCTURAL_LIMIT),
            Some(&2_160)
        );
        assert_eq!(
            report
                .parent_only_formal_child_rejection_counts
                .get(REJECT_STRUCTURAL_LIMIT),
            Some(&2_160)
        );
        assert_eq!(report.maximum_ast_node_count, 6);
        assert_eq!(report.maximum_top_level_clauses, 2);
        assert!(!report.complete_closure_enumerated);
        assert!(!report.interpreted_as_complete_closure);
    }
}
