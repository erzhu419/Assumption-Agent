//! Independent Rust strict admission profile for `hegel-old-dsl-v1.6.0`.
//!
//! Shrink step 6 changes one normalized structural bound:
//! `max_total_ast_depth` decreases from four to three.  The shrink-5 limit of
//! six total AST nodes, the shrink-4 limit of two top-level clauses, and every
//! syntax, typing, registry, tombstone, normalization, strict-CBOR, and
//! rejection-priority rule are inherited.
//!
//! The ordering is deliberate: the complete shrink-5 strict path runs first.
//! Only an otherwise accepted normalized/canonical parent program reaches the
//! new depth gate.  Every survivor therefore retains byte/hash/MDL identity.

use hegel_strict_canonicalizer::{
    encode_strict_cbor_json, BinaryOp, CanonicalProgram, Node, Sort, UnaryOp,
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
use hegel_strict_canonicalizer_shrink5::{
    canonicalize_shrink5_source_json, canonicalize_shrink5_source_node,
    decode_shrink5_canonical_ast, Shrink5Error,
};
use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub const DSL_VERSION: &str = "hegel-old-dsl-v1.6.0";
pub const FREEZE_VERSION: &str = "hegel-freeze-p2b-p3-v1.6.0";
pub const PARENT_DSL_VERSION: &str = "hegel-old-dsl-v1.5.0";
pub const PARENT_FREEZE_VERSION: &str = "hegel-freeze-p2b-p3-v1.5.0";
pub const HUMAN_AMENDMENT_ID: &str = "hegel-freeze-p2b-p3-v1.6.0-shrink-step6";
pub const SHRINK_STEP_ID: &str = "SHRINK_STEP_6_REDUCE_MAX_TOTAL_AST_DEPTH_4_TO_3";
pub const AST_SCHEMA_ID: &str = "hegel-canonical-ast-v1";
pub const CBOR_PROFILE_ID: &str = "hegel-cbor-det-v1";
pub const AST_HASH_DOMAIN: &str = "HEGEL/AST/V1";
pub const MDL_CODE_TABLE_ID: &str = "hegel-mdl-prefix-v1.0.0";
pub const REPLAY_SCHEMA_VERSION: &str = "hegel-strict-canonicalizer-shrink6-replay/1";
pub const GOLDEN_SCHEMA_VERSION: &str = "hegel-strict-canonicalizer-shrink6-golden/1";
pub const CAPACITY_SCHEMA_VERSION: &str = "hegel-strict-capacity-replay-shrink6/1";
pub const PARENT_MAXIMUM_AST_DEPTH: u32 = 4;
pub const MAXIMUM_AST_DEPTH: u32 = 3;
pub const MAXIMUM_AST_NODE_COUNT: u32 = 6;
pub const MAXIMUM_TOP_LEVEL_CLAUSES: usize = 2;
pub const REJECT_INTERNAL_SHRINK6_REPLAY: &str = "REJECT_INTERNAL_SHRINK6_REPLAY";

const GOLDEN_MANIFEST_DOMAIN: &[u8] = b"HEGEL/SHRINK6/STRICT_GOLDEN_MANIFEST/V1";
const GOLDEN_OUTCOME_DOMAIN: &[u8] = b"HEGEL/SHRINK6/STRICT_GOLDEN_OUTCOME/V1";
const CHALLENGE_SOURCE_LATTICE_DOMAIN: &[u8] =
    b"HEGEL/SHRINK6/STRICT_DEPTH4_CHALLENGE_SOURCE_LATTICE/V1";
const INHERITED_SURVIVOR_SET_DOMAIN: &[u8] =
    b"HEGEL/SHRINK6/STRICT_INHERITED_SURVIVOR_SET/V1";
const NORMALIZED_SURVIVOR_SET_DOMAIN: &[u8] =
    b"HEGEL/SHRINK6/STRICT_NORMALIZED_SURVIVOR_SET/V1";
const FULL_SURVIVOR_SET_DOMAIN: &[u8] = b"HEGEL/SHRINK6/STRICT_FULL_SURVIVOR_SET/V1";
const PARENT_CANONICAL_SET_DOMAIN: &[u8] =
    b"HEGEL/SHRINK6/STRICT_CHALLENGE_PARENT_CANONICAL_SET/V1";
const PARENT_ONLY_DEPTH4_SET_DOMAIN: &[u8] =
    b"HEGEL/SHRINK6/STRICT_PARENT_ONLY_DEPTH4_NODE6_SET/V1";
const PARENT_ONLY_SOURCE_REJECTION_DOMAIN: &[u8] =
    b"HEGEL/SHRINK6/STRICT_PARENT_ONLY_DEPTH4_SOURCE_REJECTION/V1";
const PARENT_ONLY_FORMAL_REJECTION_DOMAIN: &[u8] =
    b"HEGEL/SHRINK6/STRICT_PARENT_ONLY_DEPTH4_FORMAL_REJECTION/V1";
const ACCEPT_PARENT_IDENTITY: &str = "ACCEPT_PARENT_IDENTITY";

// These constants are replaced with the values independently replayed from
// the exact vector and capacity manifests below.  Keeping them in the strict
// implementation turns accidental vector drift into fail-closed rejection.
pub const EXPECTED_GOLDEN_MANIFEST_ROOT: &str =
    "sha256:2690413926d15db52dbd5a502ebe3fdfb1dc74d5ee3c82b2ed868cd16ab34a42";
pub const EXPECTED_GOLDEN_OUTCOME_ROOT: &str =
    "sha256:e5fd0885f95669dc6d369d0d3274778425fabb7e8c6286a27237a1b2bc8d3960";
pub const EXPECTED_SURVIVOR_CAPACITY_SET_COMMITMENT: &str =
    "sha256:6787cd6c0782fda149e1ee93b37ca8d425f5ac78850c610e21cebf9da13a16d1";
pub const EXPECTED_CHALLENGE_SOURCE_LATTICE_COMMITMENT: &str =
    "sha256:a8cfb37278000933c2c51a2797e5bc0f4e7aad6970b37e178fc681f9358574d0";
pub const EXPECTED_CHALLENGE_PARENT_CANONICAL_SET_COMMITMENT: &str =
    "sha256:8f125763d3098d087dd7e9eb484b93097295ebd765b6f079795e8009623fb13e";
pub const EXPECTED_NORMALIZED_SURVIVOR_SET_COMMITMENT: &str =
    "sha256:dcbb5562fc754fdef932188b189dbcdc0f7c500d3fc49651ee4dbb0f271afd29";
pub const EXPECTED_INHERITED_SURVIVOR_SET_COMMITMENT: &str =
    "sha256:477a5abe659a7a7e7d2d50b2a5bda61b0dae1019c44fe84950c4a05036258619";
pub const EXPECTED_PARENT_ONLY_DEPTH4_SET_COMMITMENT: &str =
    "sha256:d3eb2b2d9caf1eece5a709d8113540e4709d579cdfbe3194f1cf176c9100b20d";
pub const EXPECTED_PARENT_ONLY_SOURCE_REJECTION_COMMITMENT: &str =
    "sha256:9b0b766a4139db6297aea8b6032ad49147c1a26bf9b56291444a83681428cb0e";
pub const EXPECTED_PARENT_ONLY_FORMAL_REJECTION_COMMITMENT: &str =
    "sha256:97d50c34f51683a2502157961acc79d3b4e108b28bdaa266cf3721ffda8b3a96";
pub const EXPECTED_FIRST_SURVIVOR_CANONICAL_CBOR_HEX: &str =
    "820183010283010083000100";
pub const EXPECTED_FIRST_SURVIVOR_CANONICAL_AST_HASH: &str =
    "sha256:0f319bb95ea24abc9b4c62d03274a20cefe5dbb92fcfffbce0f0e9449aab04a6";
pub const EXPECTED_LAST_SURVIVOR_CANONICAL_CBOR_HEX: &str =
    "820186000305030180";
pub const EXPECTED_LAST_SURVIVOR_CANONICAL_AST_HASH: &str =
    "sha256:e35153f2bdd1a6e25d629ed3ab9afb178bb45ecd163efba4960a2a69db40ce2c";

pub const ORDERED_GOLDEN_VECTOR_IDS: [&str; 25] = [
    "S01", "S02", "S03", "N01", "N02", "L01", "L02", "L03", "P01", "P02", "P03",
    "P04", "P05", "F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08", "F09",
    "F10", "F11", "F12",
];

pub const EXPECTED_CHALLENGE_SOURCE_COUNT: usize = 1_266;
pub const EXPECTED_PARENT_ONLY_SOURCE_COUNT: usize = 1_199;
pub const CAPACITY_GENERATOR_RULE: &str = concat!(
    "family order A,B_abs,B_sign; operand outer, R inner, direction 0 then 1; ",
    "R is active constants -1,0,1 followed by the exact inherited 16 rational ",
    "aggregate leaves in map/scope/quantity order; A-U1 is bit_to_scalar(bit_at ",
    "0..7), int_to_scalar(set_size), int_to_scalar(count_nonzero) in ",
    "scope/quantity order, then absolute of the inherited rational aggregates; ",
    "B-Q is absolute of the first 17 A-U1 non-rational-aggregate forms; rows ",
    "with aggregate-bearing operand and aggregate R are excluded; no source ",
    "deduplication; this is FROZEN_DEPTH4_CHALLENGE_LATTICE_ONLY_NOT_COMPLETE"
);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shrink6Error {
    pub code: String,
    pub message: String,
}

impl Shrink6Error {
    fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }
}

impl From<Shrink5Error> for Shrink6Error {
    fn from(error: Shrink5Error) -> Self {
        Self::new(error.code, error.message)
    }
}

impl From<hegel_strict_canonicalizer::CanonicalError> for Shrink6Error {
    fn from(error: hegel_strict_canonicalizer::CanonicalError) -> Self {
        Self::new(error.code, error.message)
    }
}

impl fmt::Display for Shrink6Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for Shrink6Error {}

fn enforce_shrink6_depth_limit(
    program: CanonicalProgram,
) -> Result<CanonicalProgram, Shrink6Error> {
    if program.depth > MAXIMUM_AST_DEPTH {
        return Err(Shrink6Error::new(
            REJECT_STRUCTURAL_LIMIT,
            format!(
                "AST has depth {}; maximum is {MAXIMUM_AST_DEPTH}",
                program.depth
            ),
        ));
    }
    Ok(program)
}

/// Run the complete shrink-5 source path before applying the sole step-6 gate.
pub fn canonicalize_shrink6_source_json(
    value: &Value,
) -> Result<CanonicalProgram, Shrink6Error> {
    enforce_shrink6_depth_limit(canonicalize_shrink5_source_json(value)?)
}

/// Apply the same ordered gate to an already parsed source node.
pub fn canonicalize_shrink6_source_node(
    source: Node,
) -> Result<CanonicalProgram, Shrink6Error> {
    enforce_shrink6_depth_limit(canonicalize_shrink5_source_node(source)?)
}

/// Decode exact formal CBOR through shrink-5 before measuring the depth bound.
pub fn decode_shrink6_canonical_ast(
    bytes: &[u8],
) -> Result<CanonicalProgram, Shrink6Error> {
    enforce_shrink6_depth_limit(decode_shrink5_canonical_ast(bytes)?)
}

pub fn sort_name(sort: Sort) -> &'static str {
    hegel_strict_canonicalizer_shrink5::sort_name(sort)
}

fn same_program_identity(left: &CanonicalProgram, right: &CanonicalProgram) -> bool {
    left.canonical_node == right.canonical_node
        && left.canonical_cbor == right.canonical_cbor
        && left.canonical_ast_hash == right.canonical_ast_hash
        && left.output_sort == right.output_sort
        && left.root_operator_id == right.root_operator_id
        && left.node_count == right.node_count
        && left.depth == right.depth
        && left.distinct_bit_slot_count == right.distinct_bit_slot_count
        && left.aggregate_leaf_count == right.aggregate_leaf_count
        && left.scalar_parameter_occurrence_count == right.scalar_parameter_occurrence_count
        && mdl_bit_length(&left.canonical_node).is_some()
        && mdl_bit_length(&left.canonical_node) == mdl_bit_length(&right.canonical_node)
}

fn elias_delta_length(value: u64) -> Option<u64> {
    if value == 0 {
        return None;
    }
    let log_n = 63 - u64::from(value.leading_zeros());
    let log_log = 63 - u64::from((log_n + 1).leading_zeros());
    Some(log_n + 2 * log_log + 1)
}

/// Exact integer-bit length under the inherited `hegel-mdl-prefix-v1.0.0`
/// table.  Q32 is this value shifted left by 32, with no floating point.
fn mdl_bit_length(node: &Node) -> Option<u64> {
    match node {
        Node::ScalarConst(index) if ACTIVE_RATIONAL_PARAMETER_IDS.contains(index) => Some(8),
        Node::BitAt(index) if *index < 8 => Some(5 + elias_delta_length(index + 1)?),
        Node::SetSize => Some(5),
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
            Some(11 + clause_count_bits + 3 * scope_extension.len() as u64)
        }
        Node::ContextFlag(index) if *index < 4 => Some(5 + elias_delta_length(index + 1)?),
        Node::TaskFlag(index) if *index < 2 => Some(5 + elias_delta_length(index + 1)?),
        Node::Unary { child, .. } => Some(4 + mdl_bit_length(child)?),
        Node::Binary { left, right, .. } => {
            Some(5 + mdl_bit_length(left)? + mdl_bit_length(right)?)
        }
        Node::ApproxEqual {
            left,
            right,
            tolerance_index,
        } if (1..=2).contains(tolerance_index) => {
            Some(6 + mdl_bit_length(left)? + mdl_bit_length(right)?)
        }
        Node::And(children) if children.len() == MAXIMUM_TOP_LEVEL_CLAUSES => Some(
            5 + children.iter().map(mdl_bit_length).collect::<Option<Vec<_>>>()?.into_iter().sum::<u64>(),
        ),
        Node::NewSymbolCall(_)
        | Node::ScalarConst(_)
        | Node::BitAt(_)
        | Node::ContextFlag(_)
        | Node::TaskFlag(_)
        | Node::Aggregate { .. }
        | Node::ApproxEqual { .. }
        | Node::And(_) => None,
    }
}

fn failure(message: impl Into<String>) -> Shrink6Error {
    Shrink6Error::new(REJECT_INTERNAL_SHRINK6_REPLAY, message)
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
fn source_depth3_survivor() -> Value {
    json!([
        "absolute",
        [
            "difference",
            ["bit_to_scalar", ["bit_at", 0]],
            ["scalar_const", 1]
        ]
    ])
}

#[cfg(test)]
fn source_depth4_parent_only() -> Value {
    json!([
        "sign",
        [
            "absolute",
            [
                "difference",
                ["bit_to_scalar", ["bit_at", 0]],
                ["scalar_const", -1, 1]
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
            "N02",
            "source_normalization_before_limit_checks",
            json!([
                "sign",
                [
                    "absolute",
                    [
                        "difference",
                        ["bit_to_scalar", ["bit_at", 0]],
                        ["scalar_const", 0, 1]
                    ]
                ]
            ]),
            ACCEPT_PARENT_IDENTITY,
        ),
        source_vector(
            "L01",
            "source_depth_limit_checks",
            json!([
                "sign",
                [
                    "absolute",
                    [
                        "difference",
                        ["bit_to_scalar", ["bit_at", 0]],
                        ["scalar_const", -1, 1]
                    ]
                ]
            ]),
            REJECT_STRUCTURAL_LIMIT,
        ),
        source_vector(
            "L02",
            "source_depth_limit_checks",
            json!([
                "absolute",
                [
                    "difference",
                    ["absolute", ["bit_to_scalar", ["bit_at", 0]]],
                    ["scalar_const", -1, 1]
                ]
            ]),
            REJECT_STRUCTURAL_LIMIT,
        ),
        source_vector(
            "L03",
            "source_depth_limit_checks",
            json!([
                "sign",
                [
                    "difference",
                    ["absolute", ["bit_to_scalar", ["bit_at", 0]]],
                    ["scalar_const", -1, 1]
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
            "formal_depth_limit_checks",
            json!([
                1,
                [
                    1,
                    3,
                    [
                        1,
                        2,
                        [
                            2,
                            1,
                            [1, 0, [0, 1, 0]],
                            [0, 0, 1]
                        ]
                    ]
                ]
            ]),
            REJECT_STRUCTURAL_LIMIT,
        ),
        formal_vector(
            "F03",
            "formal_depth_limit_checks",
            json!([
                1,
                [
                    1,
                    2,
                    [
                        2,
                        1,
                        [1, 2, [1, 0, [0, 1, 0]]],
                        [0, 0, 1]
                    ]
                ]
            ]),
            REJECT_STRUCTURAL_LIMIT,
        ),
        formal_vector(
            "F04",
            "formal_depth_limit_checks",
            json!([
                1,
                [
                    1,
                    3,
                    [
                        2,
                        1,
                        [1, 2, [1, 0, [0, 1, 0]]],
                        [0, 0, 1]
                    ]
                ]
            ]),
            REJECT_STRUCTURAL_LIMIT,
        ),
        formal_vector(
            "F05",
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
            "F06",
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
            "F07",
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
            "F08",
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
            "F09",
            "formal_priority_checks",
            json!([1, [4, [formal_atom(0), [2, 4, [0, 0, 1], [0, 0, 5]]]]]),
            REJECT_NONCANONICAL_AST,
        ),
        formal_vector(
            "F10",
            "formal_priority_checks",
            json!([1, [4, [formal_atom(0), [2, 7, [0, 0, 1], [0, 0, 5]]]]]),
            REJECT_NONCANONICAL_AST,
        ),
        formal_vector(
            "F11",
            "formal_priority_checks",
            json!([1, [4, [formal_atom(0), [0, 0, 1]]]]),
            REJECT_TYPE_MISMATCH,
        ),
        formal_vector(
            "F12",
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

fn golden_input_wire(vector: &GoldenVector) -> Result<Vec<u8>, Shrink6Error> {
    match &vector.input {
        GoldenInput::Source(value) => serde_json::to_vec(value)
            .map_err(|error| failure(format!("golden source JSON encoding failed: {error}"))),
        GoldenInput::Formal(value) => Ok(encode_strict_cbor_json(value)?),
    }
}

fn golden_manifest_root(vectors: &[GoldenVector]) -> Result<String, Shrink6Error> {
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
        .collect::<Result<Vec<_>, Shrink6Error>>()?;
    Ok(framed_hash(GOLDEN_MANIFEST_DOMAIN, &rows))
}

fn increment_category(
    category: &str,
    surviving_identity_checks: &mut usize,
    source_normalization_before_limit_checks: &mut usize,
    source_depth_limit_checks: &mut usize,
    source_priority_checks: &mut usize,
    formal_surviving_identity_checks: &mut usize,
    formal_depth_limit_checks: &mut usize,
    formal_priority_checks: &mut usize,
) -> Result<(), Shrink6Error> {
    match category {
        "surviving_identity_checks" => *surviving_identity_checks += 1,
        "source_normalization_before_limit_checks" => {
            *source_normalization_before_limit_checks += 1
        }
        "source_depth_limit_checks" => *source_depth_limit_checks += 1,
        "source_priority_checks" => *source_priority_checks += 1,
        "formal_surviving_identity_checks" => *formal_surviving_identity_checks += 1,
        "formal_depth_limit_checks" => *formal_depth_limit_checks += 1,
        "formal_priority_checks" => *formal_priority_checks += 1,
        other => return Err(failure(format!("unknown golden category {other}"))),
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize)]
pub struct Shrink6GoldenReplayReport {
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
    pub maximum_ast_depth: u32,
    pub maximum_ast_node_count: u32,
    pub maximum_top_level_clauses: usize,
    pub vector_count: usize,
    pub passed_count: usize,
    pub surviving_identity_checks: usize,
    pub source_normalization_before_limit_checks: usize,
    pub source_depth_limit_checks: usize,
    pub source_priority_checks: usize,
    pub formal_surviving_identity_checks: usize,
    pub formal_depth_limit_checks: usize,
    pub formal_priority_checks: usize,
    pub golden_vector_manifest_root: String,
    pub golden_outcome_root: String,
    pub ordered_vector_ids: [&'static str; 25],
    pub execution_state: &'static str,
    pub closure_executed: bool,
    pub formal_roots_generated: bool,
    pub formal_roots: Option<String>,
    pub target_or_split_modules_loaded: bool,
}

/// Replay the exact 25-vector target-free shrink-6 strict manifest.
pub fn replay_shrink6_golden_vectors() -> Result<Shrink6GoldenReplayReport, Shrink6Error> {
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
    let mut source_depth_limit_checks = 0usize;
    let mut source_priority_checks = 0usize;
    let mut formal_surviving_identity_checks = 0usize;
    let mut formal_depth_limit_checks = 0usize;
    let mut formal_priority_checks = 0usize;

    for vector in &vectors {
        let outcome = match &vector.input {
            GoldenInput::Source(value) if vector.expected == ACCEPT_PARENT_IDENTITY => {
                let parent = canonicalize_shrink5_source_json(value)?;
                let child = canonicalize_shrink6_source_json(value)?;
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
                let parent = decode_shrink5_canonical_ast(&bytes)?;
                let child = decode_shrink6_canonical_ast(&bytes)?;
                if !same_program_identity(&parent, &child) {
                    return Err(failure(format!(
                        "{}: surviving formal identity changed",
                        vector.vector_id
                    )));
                }
                accepted_outcome(&child)
            }
            GoldenInput::Source(value) => {
                if vector.category == "source_depth_limit_checks" {
                    let parent = canonicalize_shrink5_source_json(value)?;
                    if parent.depth != PARENT_MAXIMUM_AST_DEPTH
                        || parent.node_count > MAXIMUM_AST_NODE_COUNT
                    {
                        return Err(failure(format!(
                            "{}: source limit vector is not an accepted depth-four parent",
                            vector.vector_id
                        )));
                    }
                }
                match canonicalize_shrink6_source_json(value) {
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
                if vector.category == "formal_depth_limit_checks" {
                    let parent = decode_shrink5_canonical_ast(&bytes)?;
                    if parent.depth != PARENT_MAXIMUM_AST_DEPTH
                        || parent.node_count > MAXIMUM_AST_NODE_COUNT
                    {
                        return Err(failure(format!(
                            "{}: formal limit vector is not an accepted depth-four parent",
                            vector.vector_id
                        )));
                    }
                }
                match decode_shrink6_canonical_ast(&bytes) {
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
            &mut source_depth_limit_checks,
            &mut source_priority_checks,
            &mut formal_surviving_identity_checks,
            &mut formal_depth_limit_checks,
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

    Ok(Shrink6GoldenReplayReport {
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
        maximum_ast_depth: MAXIMUM_AST_DEPTH,
        maximum_ast_node_count: MAXIMUM_AST_NODE_COUNT,
        maximum_top_level_clauses: MAXIMUM_TOP_LEVEL_CLAUSES,
        vector_count: vectors.len(),
        passed_count: vectors.len(),
        surviving_identity_checks,
        source_normalization_before_limit_checks,
        source_depth_limit_checks,
        source_priority_checks,
        formal_surviving_identity_checks,
        formal_depth_limit_checks,
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

fn unary(op: UnaryOp, child: Node) -> Node {
    Node::Unary {
        op,
        child: Box::new(child),
    }
}

fn difference(left: Node, right: Node) -> Node {
    Node::Binary {
        op: BinaryOp::Difference,
        left: Box::new(left),
        right: Box::new(right),
    }
}

/// Frozen scalar leaves used as the right/left comparison pool.
///
/// Ordering is normative: active rational parameters first, followed by the
/// inherited rational aggregate order (map, scope, quantity).
fn challenge_r_pool() -> Vec<Node> {
    ACTIVE_RATIONAL_PARAMETER_IDS
        .iter()
        .copied()
        .map(Node::ScalarConst)
        .chain(capacity_rational_aggregates())
        .collect()
}

fn count_nonzero_leaf(scope_id: u64, quantity_id: u64) -> Node {
    Node::Aggregate {
        map_id: 1,
        scope_id,
        quantity_id,
        scope_extension: Vec::new(),
    }
}

/// Unary-one-step rational pool used by challenge family A.
fn challenge_a_u1_pool() -> Vec<Node> {
    let mut values = Vec::with_capacity(33);
    for bit_index in 0..8_u64 {
        values.push(unary(UnaryOp::BitToScalar, Node::BitAt(bit_index)));
    }
    values.push(unary(UnaryOp::IntToScalar, Node::SetSize));
    for scope_id in 0..4_u64 {
        for quantity_id in 0..2_u64 {
            values.push(unary(
                UnaryOp::IntToScalar,
                count_nonzero_leaf(scope_id, quantity_id),
            ));
        }
    }
    for aggregate in capacity_rational_aggregates() {
        values.push(unary(UnaryOp::Absolute, aggregate));
    }
    values
}

/// Absolute-wrapped rational pool used by challenge families B-abs/B-sign.
fn challenge_b_q_pool() -> Vec<Node> {
    let mut values = Vec::with_capacity(17);
    for bit_index in 0..8_u64 {
        values.push(unary(
            UnaryOp::Absolute,
            unary(UnaryOp::BitToScalar, Node::BitAt(bit_index)),
        ));
    }
    values.push(unary(
        UnaryOp::Absolute,
        unary(UnaryOp::IntToScalar, Node::SetSize),
    ));
    for scope_id in 0..4_u64 {
        for quantity_id in 0..2_u64 {
            values.push(unary(
                UnaryOp::Absolute,
                unary(
                    UnaryOp::IntToScalar,
                    count_nonzero_leaf(scope_id, quantity_id),
                ),
            ));
        }
    }
    values
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ChallengeFamily {
    A,
    BAbsolute,
    BSign,
}

impl ChallengeFamily {
    fn id(self) -> &'static str {
        match self {
            Self::A => "A",
            Self::BAbsolute => "B_abs",
            Self::BSign => "B_sign",
        }
    }
}

#[derive(Debug, Clone)]
struct ChallengeSource {
    family: ChallengeFamily,
    source: Node,
}

fn challenge_source(family: ChallengeFamily, left: Node, right: Node) -> Node {
    let delta = difference(left, right);
    match family {
        ChallengeFamily::A => unary(
            UnaryOp::Sign,
            unary(UnaryOp::Absolute, delta),
        ),
        ChallengeFamily::BAbsolute => unary(UnaryOp::Absolute, delta),
        ChallengeFamily::BSign => unary(UnaryOp::Sign, delta),
    }
}

fn aggregate_bearing(node: &Node) -> bool {
    match node {
        Node::Aggregate { .. } => true,
        Node::Unary { child, .. } => aggregate_bearing(child),
        Node::Binary { left, right, .. } | Node::ApproxEqual { left, right, .. } => {
            aggregate_bearing(left) || aggregate_bearing(right)
        }
        Node::And(children) => children.iter().any(aggregate_bearing),
        Node::ScalarConst(_)
        | Node::BitAt(_)
        | Node::SetSize
        | Node::ContextFlag(_)
        | Node::TaskFlag(_)
        | Node::NewSymbolCall(_) => false,
    }
}

/// Frozen ordered depth-four challenge source lattice.
///
/// Source instances are intentionally not deduplicated.  Family order is A,
/// B-absolute, B-sign; candidate is outer, R is inner, direction is innermost
/// with 0=(candidate,R) and 1=(R,candidate).
fn depth4_challenge_sources() -> Vec<ChallengeSource> {
    let r_pool = challenge_r_pool();
    let a_pool = challenge_a_u1_pool();
    let b_pool = challenge_b_q_pool();
    let mut rows = Vec::with_capacity(1_266);
    for (family, candidates) in [
        (ChallengeFamily::A, &a_pool),
        (ChallengeFamily::BAbsolute, &b_pool),
        (ChallengeFamily::BSign, &b_pool),
    ] {
        for candidate in candidates {
            for r_value in &r_pool {
                if aggregate_bearing(candidate) && aggregate_bearing(r_value) {
                    continue;
                }
                rows.push(ChallengeSource {
                    family,
                    source: challenge_source(family, candidate.clone(), r_value.clone()),
                });
                rows.push(ChallengeSource {
                    family,
                    source: challenge_source(family, r_value.clone(), candidate.clone()),
                });
            }
        }
    }
    debug_assert_eq!(rows.len(), 1_266);
    rows
}

fn challenge_source_json(node: &Node) -> Result<Value, Shrink6Error> {
    let value = match node {
        Node::ScalarConst(index) => match index {
            1 => json!(["scalar_const", -1, 1]),
            3 => json!(["scalar_const", 0, 1]),
            5 => json!(["scalar_const", 1, 1]),
            _ => return Err(failure("challenge source uses an inactive scalar parameter")),
        },
        Node::BitAt(index) => json!(["bit_at", index]),
        Node::SetSize => json!(["set_size"]),
        Node::Aggregate {
            map_id,
            scope_id,
            quantity_id,
            scope_extension,
        } => {
            if !scope_extension.is_empty() {
                return Err(failure("challenge aggregate extension must be empty"));
            }
            let map = match map_id {
                0 => "sum_v1",
                1 => "count_nonzero_v1",
                5 => "signed_balance_v1",
                _ => return Err(failure("challenge source uses an inactive aggregate map")),
            };
            let scope = match scope_id {
                0 => "scope_all_observed_v1",
                1 => "scope_primary_only_v1",
                2 => "scope_boundary_only_v1",
                3 => "control_volume_all_observed_v1",
                _ => return Err(failure("challenge source uses an unknown scope")),
            };
            let quantity = match quantity_id {
                0 => "q0",
                1 => "q1",
                _ => return Err(failure("challenge source uses an unknown quantity")),
            };
            json!(["aggregate", map, scope, quantity, []])
        }
        Node::ContextFlag(index) if *index < 4 => json!(["context_flag", format!("c{index}")]),
        Node::TaskFlag(index) if *index < 2 => json!(["task_flag", format!("t{index}")]),
        Node::Unary { op, child } => {
            let name = match op {
                UnaryOp::BitToScalar => "bit_to_scalar",
                UnaryOp::IntToScalar => "int_to_scalar",
                UnaryOp::Absolute => "absolute",
                UnaryOp::Sign => "sign",
            };
            json!([name, challenge_source_json(child)?])
        }
        Node::Binary { op, left, right } => {
            let name = match op {
                BinaryOp::Add => "add",
                BinaryOp::Difference => "difference",
                BinaryOp::EqualExact => "equal_exact",
                BinaryOp::LessEqual => "less_equal",
                BinaryOp::GreaterEqual => "greater_equal",
                BinaryOp::SameSign => "same_sign",
                BinaryOp::OppositeSign => "opposite_sign",
            };
            json!([name, challenge_source_json(left)?, challenge_source_json(right)?])
        }
        Node::ApproxEqual {
            left,
            right,
            tolerance_index,
        } => json!([
            "approx_equal",
            challenge_source_json(left)?,
            challenge_source_json(right)?,
            tolerance_index,
        ]),
        Node::And(children) => {
            let mut values = vec![Value::String("top_level_AND".to_owned())];
            for child in children {
                values.push(challenge_source_json(child)?);
            }
            Value::Array(values)
        }
        Node::NewSymbolCall(_)
        | Node::ContextFlag(_)
        | Node::TaskFlag(_) => {
            return Err(failure("challenge source is outside the frozen old DSL"))
        }
    };
    Ok(value)
}

fn challenge_source_lattice_commitment(
    rows: &[ChallengeSource],
) -> Result<String, Shrink6Error> {
    let framed_rows = rows
        .iter()
        .map(|row| {
            let source_wire = serde_json::to_vec(&challenge_source_json(&row.source)?)
                .map_err(|error| failure(format!("challenge source JSON encoding failed: {error}")))?;
            Ok(vec![row.family.id().as_bytes().to_vec(), source_wire])
        })
        .collect::<Result<Vec<_>, Shrink6Error>>()?;
    Ok(framed_hash(CHALLENGE_SOURCE_LATTICE_DOMAIN, &framed_rows))
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
pub struct Shrink6CapacityReplayReport {
    pub canonical_program_budget: usize,
    pub challenge_parent_accepted_count: usize,
    pub challenge_parent_canonical_set_commitment: String,
    pub challenge_parent_canonical_unique_count: usize,
    pub challenge_source_candidate_count: usize,
    pub challenge_source_family_counts: BTreeMap<String, usize>,
    pub challenge_source_lattice_commitment: String,
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
    pub inherited_survivor_set_commitment: String,
    pub inherited_survivor_source_count: usize,
    pub inherited_survivor_unique_count: usize,
    pub interpreted_as_complete_closure: bool,
    pub last_survivor_canonical_ast_hash: String,
    pub last_survivor_canonical_cbor_hex: String,
    pub maximum_ast_depth: u32,
    pub maximum_ast_node_count: u32,
    pub maximum_top_level_clauses: usize,
    pub mixed_atom_count: usize,
    pub normalized_survivor_set_commitment: String,
    pub normalized_survivor_source_count: usize,
    pub normalized_survivor_source_family_counts: BTreeMap<String, usize>,
    pub normalized_survivor_unique_count: usize,
    pub parent_dsl_version: &'static str,
    pub parent_freeze_version: &'static str,
    pub parent_only_depth: u32,
    pub parent_only_formal_child_rejected_count: usize,
    pub parent_only_formal_child_rejection_counts: BTreeMap<String, usize>,
    pub parent_only_formal_rejection_outcome_commitment: String,
    pub parent_only_node_count: u32,
    pub parent_only_parent_accepted_count: usize,
    pub parent_only_set_commitment: String,
    pub parent_only_source_candidate_count: usize,
    pub parent_only_source_child_rejected_count: usize,
    pub parent_only_source_child_rejection_counts: BTreeMap<String, usize>,
    pub parent_only_source_family_counts: BTreeMap<String, usize>,
    pub parent_only_source_rejection_outcome_commitment: String,
    pub parent_only_unique_count: usize,
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

fn increment_family(counts: &mut BTreeMap<String, usize>, family: ChallengeFamily) {
    *counts.entry(family.id().to_owned()).or_insert(0) += 1;
}

/// Replay the frozen finite depth-four challenge lattice.  It is neither the
/// full depth-four grammar nor closure enumeration.
pub fn replay_shrink6_capacity_subset(
) -> Result<Shrink6CapacityReplayReport, Shrink6Error> {
    let constant_atoms = capacity_constant_atoms();
    let rational_aggregates = capacity_rational_aggregates();
    let mixed_atoms = capacity_mixed_atoms();
    if constant_atoms.len() != 15 || rational_aggregates.len() != 16 || mixed_atoms.len() != 144 {
        return Err(failure("shrink-6 capacity component count drift"));
    }

    let inherited_sources = constant_atoms
        .iter()
        .chain(rational_aggregates.iter())
        .chain(mixed_atoms.iter())
        .cloned()
        .collect::<Vec<_>>();
    if inherited_sources.len() != 175 {
        return Err(failure("inherited survivor source count drift"));
    }

    let challenge_rows = depth4_challenge_sources();
    if challenge_rows.len() != EXPECTED_CHALLENGE_SOURCE_COUNT {
        return Err(failure("challenge source count drift"));
    }
    let source_lattice_commitment = challenge_source_lattice_commitment(&challenge_rows)?;
    let mut challenge_source_family_counts = BTreeMap::new();
    let mut normalized_survivor_source_family_counts = BTreeMap::new();
    let mut parent_only_source_family_counts = BTreeMap::new();
    let mut challenge_parent_set = BTreeSet::new();
    let mut normalized_rows = Vec::new();
    let mut parent_only_rows = Vec::new();
    for (index, row) in challenge_rows.iter().enumerate() {
        increment_family(&mut challenge_source_family_counts, row.family);
        let parent = canonicalize_shrink5_source_node(row.source.clone())?;
        challenge_parent_set.insert(parent.canonical_cbor.clone());
        if parent.depth <= MAXIMUM_AST_DEPTH {
            increment_family(&mut normalized_survivor_source_family_counts, row.family);
            normalized_rows.push((row, parent));
        } else {
            if parent.depth != PARENT_MAXIMUM_AST_DEPTH
                || parent.node_count != MAXIMUM_AST_NODE_COUNT
            {
                return Err(failure(format!(
                    "challenge parent is not depth-four node-six at ordinal {}",
                    index + 1
                )));
            }
            increment_family(&mut parent_only_source_family_counts, row.family);
            parent_only_rows.push((row, parent));
        }
    }

    let mut inherited_set = BTreeSet::new();
    let mut survivor_set = BTreeSet::new();
    let mut survivor_parent_identity_match_count = 0usize;
    for (index, source) in inherited_sources.iter().cloned().enumerate() {
        let parent = canonicalize_shrink5_source_node(source.clone())?;
        let child = canonicalize_shrink6_source_node(source)?;
        let parent_formal = decode_shrink5_canonical_ast(&parent.canonical_cbor)?;
        let child_formal = decode_shrink6_canonical_ast(&parent.canonical_cbor)?;
        if !same_program_identity(&parent, &child)
            || !same_program_identity(&parent, &parent_formal)
            || !same_program_identity(&parent, &child_formal)
        {
            return Err(failure(format!(
                "inherited survivor identity or MDL changed at ordinal {}",
                index + 1
            )));
        }
        survivor_parent_identity_match_count += 1;
        inherited_set.insert(child.canonical_cbor.clone());
        survivor_set.insert(child.canonical_cbor);
    }

    let mut normalized_set = BTreeSet::new();
    for (index, (row, parent)) in normalized_rows.iter().enumerate() {
        let child = canonicalize_shrink6_source_node(row.source.clone())?;
        let parent_formal = decode_shrink5_canonical_ast(&parent.canonical_cbor)?;
        let child_formal = decode_shrink6_canonical_ast(&parent.canonical_cbor)?;
        if !same_program_identity(parent, &child)
            || !same_program_identity(parent, &parent_formal)
            || !same_program_identity(parent, &child_formal)
        {
            return Err(failure(format!(
                "normalized survivor identity or MDL changed at ordinal {}",
                index + 1
            )));
        }
        survivor_parent_identity_match_count += 1;
        normalized_set.insert(child.canonical_cbor.clone());
        survivor_set.insert(child.canonical_cbor);
    }

    let mut parent_only_source_child_rejection_counts = BTreeMap::new();
    let mut parent_only_formal_child_rejection_counts = BTreeMap::new();
    let mut parent_only_set = BTreeSet::new();
    for (index, (row, parent)) in parent_only_rows.iter().enumerate() {
        let parent_formal = decode_shrink5_canonical_ast(&parent.canonical_cbor)?;
        if !same_program_identity(parent, &parent_formal) {
            return Err(failure(format!(
                "parent-only formal identity differs at ordinal {}",
                index + 1
            )));
        }
        parent_only_set.insert(parent.canonical_cbor.clone());
        match canonicalize_shrink6_source_node(row.source.clone()) {
            Ok(_) => return Err(failure("parent-only source unexpectedly accepted")),
            Err(error) => {
                *parent_only_source_child_rejection_counts
                    .entry(error.code)
                    .or_insert(0) += 1
            }
        }
        match decode_shrink6_canonical_ast(&parent.canonical_cbor) {
            Ok(_) => return Err(failure("parent-only formal input unexpectedly accepted")),
            Err(error) => {
                *parent_only_formal_child_rejection_counts
                    .entry(error.code)
                    .or_insert(0) += 1
            }
        }
    }

    let survivor_source_candidate_count = inherited_sources.len() + normalized_rows.len();
    let survivor_accepted_count = survivor_source_candidate_count;
    let survivor_rejected_count = 0usize;
    let survivor_rejection_counts = BTreeMap::new();
    let parent_only_source_child_rejected_count: usize =
        parent_only_source_child_rejection_counts.values().sum();
    let parent_only_formal_child_rejected_count: usize =
        parent_only_formal_child_rejection_counts.values().sum();
    let challenge_parent_commitment =
        capacity_set_commitment(PARENT_CANONICAL_SET_DOMAIN, &challenge_parent_set);
    let inherited_commitment =
        capacity_set_commitment(INHERITED_SURVIVOR_SET_DOMAIN, &inherited_set);
    let normalized_commitment =
        capacity_set_commitment(NORMALIZED_SURVIVOR_SET_DOMAIN, &normalized_set);
    let survivor_commitment =
        capacity_set_commitment(FULL_SURVIVOR_SET_DOMAIN, &survivor_set);
    let parent_only_set_commitment =
        capacity_set_commitment(PARENT_ONLY_DEPTH4_SET_DOMAIN, &parent_only_set);
    let parent_only_source_rejection_outcome_commitment = capacity_rejection_commitment(
        PARENT_ONLY_SOURCE_REJECTION_DOMAIN,
        &parent_only_set,
    );
    let parent_only_formal_rejection_outcome_commitment = capacity_rejection_commitment(
        PARENT_ONLY_FORMAL_REJECTION_DOMAIN,
        &parent_only_set,
    );

    let first_survivor = survivor_set
        .iter()
        .next()
        .ok_or_else(|| failure("survivor set is empty"))?;
    let last_survivor = survivor_set
        .iter()
        .next_back()
        .ok_or_else(|| failure("survivor set is empty"))?;
    let first_survivor_program = decode_shrink6_canonical_ast(first_survivor)?;
    let last_survivor_program = decode_shrink6_canonical_ast(last_survivor)?;
    let first_survivor_hex = hegel_strict_canonicalizer::hex_encode(first_survivor);
    let last_survivor_hex = hegel_strict_canonicalizer::hex_encode(last_survivor);

    let expected_challenge_families = BTreeMap::from([
        ("A".to_owned(), 486),
        ("B_abs".to_owned(), 390),
        ("B_sign".to_owned(), 390),
    ]);
    let expected_normalized_families = BTreeMap::from([
        ("A".to_owned(), 33),
        ("B_abs".to_owned(), 17),
        ("B_sign".to_owned(), 17),
    ]);
    let expected_parent_only_families = BTreeMap::from([
        ("A".to_owned(), 453),
        ("B_abs".to_owned(), 373),
        ("B_sign".to_owned(), 373),
    ]);
    if source_lattice_commitment != EXPECTED_CHALLENGE_SOURCE_LATTICE_COMMITMENT
        || challenge_source_family_counts != expected_challenge_families
        || challenge_parent_set.len() != 1_249
        || challenge_parent_commitment != EXPECTED_CHALLENGE_PARENT_CANONICAL_SET_COMMITMENT
        || normalized_rows.len() != 67
        || normalized_set.len() != 50
        || normalized_survivor_source_family_counts != expected_normalized_families
        || normalized_commitment != EXPECTED_NORMALIZED_SURVIVOR_SET_COMMITMENT
        || inherited_set.len() != 175
        || inherited_commitment != EXPECTED_INHERITED_SURVIVOR_SET_COMMITMENT
        || survivor_source_candidate_count != 242
        || survivor_accepted_count != 242
        || survivor_parent_identity_match_count != 242
        || survivor_set.len() != 225
        || survivor_rejected_count != 0
        || !survivor_rejection_counts.is_empty()
        || survivor_commitment != EXPECTED_SURVIVOR_CAPACITY_SET_COMMITMENT
        || first_survivor_hex != EXPECTED_FIRST_SURVIVOR_CANONICAL_CBOR_HEX
        || first_survivor_program.canonical_ast_hash_id()
            != EXPECTED_FIRST_SURVIVOR_CANONICAL_AST_HASH
        || last_survivor_hex != EXPECTED_LAST_SURVIVOR_CANONICAL_CBOR_HEX
        || last_survivor_program.canonical_ast_hash_id()
            != EXPECTED_LAST_SURVIVOR_CANONICAL_AST_HASH
        || parent_only_rows.len() != EXPECTED_PARENT_ONLY_SOURCE_COUNT
        || parent_only_set.len() != EXPECTED_PARENT_ONLY_SOURCE_COUNT
        || parent_only_source_family_counts != expected_parent_only_families
        || parent_only_source_child_rejected_count != EXPECTED_PARENT_ONLY_SOURCE_COUNT
        || parent_only_formal_child_rejected_count != EXPECTED_PARENT_ONLY_SOURCE_COUNT
        || parent_only_source_child_rejection_counts.len() != 1
        || parent_only_formal_child_rejection_counts.len() != 1
        || parent_only_source_child_rejection_counts.get(REJECT_STRUCTURAL_LIMIT)
            != Some(&EXPECTED_PARENT_ONLY_SOURCE_COUNT)
        || parent_only_formal_child_rejection_counts.get(REJECT_STRUCTURAL_LIMIT)
            != Some(&EXPECTED_PARENT_ONLY_SOURCE_COUNT)
        || parent_only_set_commitment != EXPECTED_PARENT_ONLY_DEPTH4_SET_COMMITMENT
        || parent_only_source_rejection_outcome_commitment
            != EXPECTED_PARENT_ONLY_SOURCE_REJECTION_COMMITMENT
        || parent_only_formal_rejection_outcome_commitment
            != EXPECTED_PARENT_ONLY_FORMAL_REJECTION_COMMITMENT
    {
        return Err(failure(format!(
            "capacity invariant failure: challenge={}, source_lattice={source_lattice_commitment}, parent_unique={}, normalized={}/{}, inherited_unique={}, survivor={}/{}, parent_only={}/{}, source_rejected={parent_only_source_child_rejected_count}, formal_rejected={parent_only_formal_child_rejected_count}",
            challenge_rows.len(),
            challenge_parent_set.len(),
            normalized_rows.len(),
            normalized_set.len(),
            inherited_set.len(),
            survivor_source_candidate_count,
            survivor_set.len(),
            parent_only_rows.len(),
            parent_only_set.len(),
        )));
    }

    Ok(Shrink6CapacityReplayReport {
        canonical_program_budget: 50_000,
        challenge_parent_accepted_count: challenge_rows.len(),
        challenge_parent_canonical_set_commitment: challenge_parent_commitment,
        challenge_parent_canonical_unique_count: challenge_parent_set.len(),
        challenge_source_candidate_count: challenge_rows.len(),
        challenge_source_family_counts,
        challenge_source_lattice_commitment: source_lattice_commitment,
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
        inherited_survivor_set_commitment: inherited_commitment,
        inherited_survivor_source_count: inherited_sources.len(),
        inherited_survivor_unique_count: inherited_set.len(),
        interpreted_as_complete_closure: false,
        last_survivor_canonical_ast_hash: last_survivor_program.canonical_ast_hash_id(),
        last_survivor_canonical_cbor_hex: last_survivor_hex,
        maximum_ast_depth: MAXIMUM_AST_DEPTH,
        maximum_ast_node_count: MAXIMUM_AST_NODE_COUNT,
        maximum_top_level_clauses: MAXIMUM_TOP_LEVEL_CLAUSES,
        mixed_atom_count: mixed_atoms.len(),
        normalized_survivor_set_commitment: normalized_commitment,
        normalized_survivor_source_count: normalized_rows.len(),
        normalized_survivor_source_family_counts,
        normalized_survivor_unique_count: normalized_set.len(),
        parent_dsl_version: PARENT_DSL_VERSION,
        parent_freeze_version: PARENT_FREEZE_VERSION,
        parent_only_depth: PARENT_MAXIMUM_AST_DEPTH,
        parent_only_formal_child_rejected_count,
        parent_only_formal_child_rejection_counts,
        parent_only_formal_rejection_outcome_commitment,
        parent_only_node_count: MAXIMUM_AST_NODE_COUNT,
        parent_only_parent_accepted_count: parent_only_rows.len(),
        parent_only_set_commitment,
        parent_only_source_candidate_count: parent_only_rows.len(),
        parent_only_source_child_rejected_count,
        parent_only_source_child_rejection_counts,
        parent_only_source_family_counts,
        parent_only_source_rejection_outcome_commitment,
        parent_only_unique_count: parent_only_set.len(),
        rational_aggregate_count: rational_aggregates.len(),
        removed_binary_operator_ids: TOMBSTONED_BINARY_OPERATOR_IDS,
        retained_difference_id: 1,
        schema_version: CAPACITY_SCHEMA_VERSION,
        shrink_step_id: SHRINK_STEP_ID,
        subset_status: "FROZEN_DEPTH4_CHALLENGE_LATTICE_ONLY_NOT_COMPLETE",
        survivor_accepted_count,
        survivor_accepted_set_commitment: survivor_commitment,
        survivor_parent_identity_match_count,
        survivor_rejected_count,
        survivor_rejection_counts,
        survivor_source_candidate_count,
        survivor_unique_count: survivor_set.len(),
        target_or_split_modules_loaded: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_depth4_challenge_lattice_matches_declared_partition() {
        let rows = depth4_challenge_sources();
        assert_eq!(rows.len(), 1_266);
        let mut family_sources = BTreeMap::<&str, usize>::new();
        let mut family_parent_only = BTreeMap::<&str, usize>::new();
        let mut normalized_survivor_sources = 0usize;
        let mut normalized_survivors = BTreeSet::new();
        let mut parent_only = BTreeSet::new();
        let mut all_parent = BTreeSet::new();
        for row in rows {
            *family_sources.entry(row.family.id()).or_insert(0) += 1;
            let parent = canonicalize_shrink5_source_node(row.source).unwrap();
            all_parent.insert(parent.canonical_cbor.clone());
            if parent.depth == PARENT_MAXIMUM_AST_DEPTH {
                assert_eq!(parent.node_count, MAXIMUM_AST_NODE_COUNT);
                *family_parent_only.entry(row.family.id()).or_insert(0) += 1;
                parent_only.insert(parent.canonical_cbor);
            } else {
                assert!(parent.depth <= MAXIMUM_AST_DEPTH);
                normalized_survivor_sources += 1;
                normalized_survivors.insert(parent.canonical_cbor);
            }
        }
        assert_eq!(family_sources, BTreeMap::from([
            ("A", 486), ("B_abs", 390), ("B_sign", 390),
        ]));
        assert_eq!(family_parent_only, BTreeMap::from([
            ("A", 453), ("B_abs", 373), ("B_sign", 373),
        ]));
        assert_eq!(normalized_survivor_sources, 67);
        assert_eq!(normalized_survivors.len(), 50);
        assert_eq!(parent_only.len(), 1_199);
        assert_eq!(all_parent.len(), 1_249);
    }

    #[test]
    fn depth_three_parent_survives_with_exact_identity_and_mdl() {
        let parent = canonicalize_shrink5_source_json(&source_depth3_survivor()).unwrap();
        assert_eq!(parent.depth, MAXIMUM_AST_DEPTH);
        let child = canonicalize_shrink6_source_json(&source_depth3_survivor()).unwrap();
        assert!(same_program_identity(&parent, &child));
    }

    #[test]
    fn genuine_depth_four_source_and_formal_programs_reject() {
        let source = source_depth4_parent_only();
        let parent_source = canonicalize_shrink5_source_json(&source).unwrap();
        assert_eq!(parent_source.depth, PARENT_MAXIMUM_AST_DEPTH);
        assert_eq!(parent_source.node_count, MAXIMUM_AST_NODE_COUNT);
        assert_eq!(
            canonicalize_shrink6_source_json(&source).unwrap_err().code,
            REJECT_STRUCTURAL_LIMIT
        );
        let bytes = parent_source.canonical_cbor;
        let parent_formal = decode_shrink5_canonical_ast(&bytes).unwrap();
        assert_eq!(parent_formal.depth, PARENT_MAXIMUM_AST_DEPTH);
        assert_eq!(parent_formal.node_count, MAXIMUM_AST_NODE_COUNT);
        assert_eq!(
            decode_shrink6_canonical_ast(&bytes).unwrap_err().code,
            REJECT_STRUCTURAL_LIMIT
        );
    }

    #[test]
    fn normalization_precedes_the_new_depth_limit() {
        let source = json!([
            "sign",
            [
                "absolute",
                [
                    "difference",
                    ["bit_to_scalar", ["bit_at", 0]],
                    ["scalar_const", 0, 1]
                ]
            ]
        ]);
        let parent = canonicalize_shrink5_source_json(&source).unwrap();
        let child = canonicalize_shrink6_source_json(&source).unwrap();
        assert!(child.depth <= MAXIMUM_AST_DEPTH);
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
                canonicalize_shrink6_source_json(&source).unwrap_err().code,
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
            decode_shrink6_canonical_ast(&noncanonical).unwrap_err().code,
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
            decode_shrink6_canonical_ast(&removed).unwrap_err().code,
            REJECT_REMOVED_BINARY_OPERATOR
        );
    }

    #[test]
    fn golden_replay_covers_exact_twenty_five_vector_layout() {
        let report = replay_shrink6_golden_vectors().unwrap();
        assert_eq!(report.vector_count, 25);
        assert_eq!(report.passed_count, 25);
        assert_eq!(report.surviving_identity_checks, 3);
        assert_eq!(report.source_normalization_before_limit_checks, 2);
        assert_eq!(report.source_depth_limit_checks, 3);
        assert_eq!(report.source_priority_checks, 5);
        assert_eq!(report.formal_surviving_identity_checks, 1);
        assert_eq!(report.formal_depth_limit_checks, 3);
        assert_eq!(report.formal_priority_checks, 8);
        assert_eq!(report.maximum_ast_depth, 3);
        assert_eq!(report.maximum_ast_node_count, 6);
        assert_eq!(report.maximum_top_level_clauses, 2);
        assert_eq!(report.ordered_vector_ids, ORDERED_GOLDEN_VECTOR_IDS);
    }

    #[test]
    fn survivor_and_parent_only_capacity_sets_replay_but_are_not_closure() {
        let report = replay_shrink6_capacity_subset().unwrap();
        assert_eq!(report.challenge_source_candidate_count, 1_266);
        assert_eq!(report.challenge_parent_accepted_count, 1_266);
        assert_eq!(report.challenge_parent_canonical_unique_count, 1_249);
        assert_eq!(
            report.challenge_source_family_counts,
            BTreeMap::from([
                ("A".to_owned(), 486),
                ("B_abs".to_owned(), 390),
                ("B_sign".to_owned(), 390),
            ])
        );
        assert_eq!(report.normalized_survivor_source_count, 67);
        assert_eq!(report.normalized_survivor_unique_count, 50);
        assert_eq!(report.inherited_survivor_source_count, 175);
        assert_eq!(report.inherited_survivor_unique_count, 175);
        assert_eq!(report.survivor_source_candidate_count, 242);
        assert_eq!(report.survivor_accepted_count, 242);
        assert_eq!(report.survivor_parent_identity_match_count, 242);
        assert_eq!(report.survivor_unique_count, 225);
        assert_eq!(report.survivor_rejected_count, 0);
        assert_eq!(report.parent_only_source_candidate_count, 1_199);
        assert_eq!(report.parent_only_parent_accepted_count, 1_199);
        assert_eq!(report.parent_only_unique_count, 1_199);
        assert_eq!(report.parent_only_depth, 4);
        assert_eq!(report.parent_only_node_count, 6);
        assert_eq!(report.parent_only_source_child_rejected_count, 1_199);
        assert_eq!(report.parent_only_formal_child_rejected_count, 1_199);
        assert_eq!(
            report
                .parent_only_source_child_rejection_counts
                .get(REJECT_STRUCTURAL_LIMIT),
            Some(&1_199)
        );
        assert_eq!(
            report
                .parent_only_formal_child_rejection_counts
                .get(REJECT_STRUCTURAL_LIMIT),
            Some(&1_199)
        );
        assert_eq!(
            report.challenge_source_lattice_commitment,
            EXPECTED_CHALLENGE_SOURCE_LATTICE_COMMITMENT
        );
        assert_eq!(
            report.challenge_parent_canonical_set_commitment,
            EXPECTED_CHALLENGE_PARENT_CANONICAL_SET_COMMITMENT
        );
        assert_eq!(
            report.normalized_survivor_set_commitment,
            EXPECTED_NORMALIZED_SURVIVOR_SET_COMMITMENT
        );
        assert_eq!(
            report.inherited_survivor_set_commitment,
            EXPECTED_INHERITED_SURVIVOR_SET_COMMITMENT
        );
        assert_eq!(
            report.survivor_accepted_set_commitment,
            EXPECTED_SURVIVOR_CAPACITY_SET_COMMITMENT
        );
        assert_eq!(
            report.parent_only_set_commitment,
            EXPECTED_PARENT_ONLY_DEPTH4_SET_COMMITMENT
        );
        assert_eq!(
            report.parent_only_source_rejection_outcome_commitment,
            EXPECTED_PARENT_ONLY_SOURCE_REJECTION_COMMITMENT
        );
        assert_eq!(
            report.parent_only_formal_rejection_outcome_commitment,
            EXPECTED_PARENT_ONLY_FORMAL_REJECTION_COMMITMENT
        );
        assert_eq!(report.maximum_ast_depth, 3);
        assert_eq!(report.maximum_ast_node_count, 6);
        assert_eq!(report.maximum_top_level_clauses, 2);
        assert!(!report.complete_closure_enumerated);
        assert!(!report.interpreted_as_complete_closure);
    }
}
