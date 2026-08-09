//! Independent Rust strict admission profile for `hegel-old-dsl-v1.4.0`.
//!
//! Shrink step 4 changes one normalized structural bound: a canonical
//! top-level conjunction may contain exactly two clauses, not two or three.
//! Every syntax, typing, registry, sparse tombstone, rewrite, CBOR, and
//! noncanonical-input rule remains inherited unchanged from shrink step 3.
//!
//! The ordering is deliberate.  The shrink-3 strict path first validates and
//! canonicalizes the complete input.  Only an otherwise accepted canonical
//! program reaches the new arity gate.  Consequently malformed, type,
//! registry, tombstone, and noncanonical errors retain their frozen priority,
//! while source conjunctions are flattened, sorted, and deduplicated before
//! the new maximum is measured.

use hegel_strict_canonicalizer::{
    encode_strict_cbor_json, BinaryOp, CanonicalProgram, Node, Sort,
    REJECT_MALFORMED_SOURCE_AST, REJECT_NONCANONICAL_AST, REJECT_STRUCTURAL_LIMIT,
    REJECT_TYPE_MISMATCH,
};
use hegel_strict_canonicalizer_shrink2::{
    REJECT_REMOVED_AGGREGATE_MAP, REJECT_REMOVED_RATIONAL_PARAMETER,
};
use hegel_strict_canonicalizer_shrink3::{
    canonicalize_shrink3_source_json, canonicalize_shrink3_source_node,
    decode_shrink3_canonical_ast, Shrink3Error, ACTIVE_BINARY_OPERATOR_IDS_FORMAL,
    ACTIVE_BINARY_OPERATOR_IDS_SOURCE, ACTIVE_RATIONAL_PARAMETER_IDS,
    EXPECTED_FIRST_CANONICAL_CBOR_HEX, EXPECTED_LAST_CANONICAL_CBOR_HEX,
    EXPECTED_SURVIVOR_ACCEPTED_SET_COMMITMENT, REJECT_REMOVED_BINARY_OPERATOR,
    RESERVED_BINARY_OPERATOR_IDS, TOMBSTONED_BINARY_OPERATOR_IDS,
};
use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub const DSL_VERSION: &str = "hegel-old-dsl-v1.4.0";
pub const FREEZE_VERSION: &str = "hegel-freeze-p2b-p3-v1.4.0";
pub const PARENT_DSL_VERSION: &str = "hegel-old-dsl-v1.3.0";
pub const PARENT_FREEZE_VERSION: &str = "hegel-freeze-p2b-p3-v1.3.0";
pub const HUMAN_AMENDMENT_ID: &str = "hegel-freeze-p2b-p3-v1.4.0-shrink-step4";
pub const SHRINK_STEP_ID: &str = "SHRINK_STEP_4_REDUCE_MAX_TOP_LEVEL_CLAUSES_3_TO_2";
pub const AST_SCHEMA_ID: &str = "hegel-canonical-ast-v1";
pub const CBOR_PROFILE_ID: &str = "hegel-cbor-det-v1";
pub const AST_HASH_DOMAIN: &str = "HEGEL/AST/V1";
pub const REPLAY_SCHEMA_VERSION: &str = "hegel-strict-canonicalizer-shrink4-replay/1";
pub const GOLDEN_SCHEMA_VERSION: &str = "hegel-strict-canonicalizer-shrink4-golden/1";
pub const CAPACITY_SCHEMA_VERSION: &str = "hegel-strict-capacity-replay-shrink4/1";
pub const MAX_TOP_LEVEL_CLAUSES: usize = 2;
pub const REJECT_INTERNAL_SHRINK4_REPLAY: &str = "REJECT_INTERNAL_SHRINK4_REPLAY";
pub const EXPECTED_GOLDEN_MANIFEST_ROOT: &str =
    "sha256:f84035e632bf5a655a9ebd636a0cafe7ab1097c45be87d4db944a0012f52aa90";
pub const EXPECTED_GOLDEN_OUTCOME_ROOT: &str =
    "sha256:c19341f08ac5f5759c2cdcb3681a37d958de362b81d02c184f7e2413dca18d7c";
pub const ORDERED_GOLDEN_VECTOR_IDS: [&str; 22] = [
    "S01", "S02", "S03", "N01", "N02", "L01", "L02", "P01", "P02", "P03", "P04",
    "P05", "F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08", "F09", "F10",
];

pub const EXPECTED_SURVIVOR_SOURCE_COUNT: usize = 2_160;
pub const SURVIVOR_CAPACITY_GENERATOR_RULE: &str =
    "inherit the exact 2160-source shrink-3 target-free constructive subset; every source is a normalized top_level_AND with exactly two distinct clauses; require identical canonical AST CBOR bytes, hashes, and MDL lengths across shrink step 4";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shrink4Error {
    pub code: String,
    pub message: String,
}

impl Shrink4Error {
    fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }
}

impl From<Shrink3Error> for Shrink4Error {
    fn from(error: Shrink3Error) -> Self {
        Self::new(error.code, error.message)
    }
}

impl From<hegel_strict_canonicalizer::CanonicalError> for Shrink4Error {
    fn from(error: hegel_strict_canonicalizer::CanonicalError) -> Self {
        Self::new(error.code, error.message)
    }
}

impl fmt::Display for Shrink4Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for Shrink4Error {}

fn enforce_shrink4_clause_limit(
    program: CanonicalProgram,
) -> Result<CanonicalProgram, Shrink4Error> {
    if let Node::And(atoms) = &program.canonical_node {
        if atoms.len() > MAX_TOP_LEVEL_CLAUSES {
            return Err(Shrink4Error::new(
                REJECT_STRUCTURAL_LIMIT,
                format!(
                    "flattened AND has {} clauses; maximum is {MAX_TOP_LEVEL_CLAUSES}",
                    atoms.len()
                ),
            ));
        }
    }
    Ok(program)
}

/// Parse and canonicalize source JSON with the complete shrink-3 strict path,
/// then apply the sole shrink-4 delta to the normalized root conjunction.
pub fn canonicalize_shrink4_source_json(
    value: &Value,
) -> Result<CanonicalProgram, Shrink4Error> {
    enforce_shrink4_clause_limit(canonicalize_shrink3_source_json(value)?)
}

/// Canonicalize an already parsed source node under the same ordered gate.
pub fn canonicalize_shrink4_source_node(
    source: Node,
) -> Result<CanonicalProgram, Shrink4Error> {
    enforce_shrink4_clause_limit(canonicalize_shrink3_source_node(source)?)
}

/// Decode exact formal CBOR through the shrink-3 strict decoder first.  This
/// preserves formal shape, registry, tombstone, type, and noncanonical error
/// priority before the normalized two-clause structural gate is considered.
pub fn decode_shrink4_canonical_ast(
    bytes: &[u8],
) -> Result<CanonicalProgram, Shrink4Error> {
    enforce_shrink4_clause_limit(decode_shrink3_canonical_ast(bytes)?)
}

pub fn sort_name(sort: Sort) -> &'static str {
    hegel_strict_canonicalizer_shrink3::sort_name(sort)
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

fn capacity_set_commitment(sorted_cbor: &BTreeSet<Vec<u8>>) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"HEGEL/STRICT_CAPACITY_SET/V1");
    hasher.update([0]);
    for bytes in sorted_cbor {
        hasher.update((bytes.len() as u64).to_be_bytes());
        hasher.update(bytes);
    }
    format!("sha256:{:x}", hasher.finalize())
}

#[derive(Debug, Clone, Serialize)]
pub struct Shrink4CapacityReplayReport {
    pub schema_version: &'static str,
    pub implementation: &'static str,
    pub parent_dsl_version: &'static str,
    pub parent_freeze_version: &'static str,
    pub dsl_version: &'static str,
    pub freeze_version: &'static str,
    pub human_amendment_id: &'static str,
    pub shrink_step_id: &'static str,
    pub generator_rule: &'static str,
    pub removed_binary_operator_ids: [u64; 1],
    pub retained_difference_id: u64,
    pub maximum_top_level_clauses: usize,
    pub constant_atom_count: usize,
    pub rational_aggregate_count: usize,
    pub mixed_atom_count: usize,
    pub source_candidate_count: usize,
    pub normalized_and2_count: usize,
    pub accepted_source_count: usize,
    pub accepted_unique_count: usize,
    pub parent_identity_match_count: usize,
    pub rejected_count: usize,
    pub rejection_counts: BTreeMap<String, usize>,
    pub rewrite_collapsed_count: usize,
    pub accepted_set_commitment: String,
    pub first_canonical_cbor_hex: String,
    pub first_canonical_ast_hash: String,
    pub last_canonical_cbor_hex: String,
    pub last_canonical_ast_hash: String,
    pub canonical_program_budget: usize,
    pub first_out_of_budget_ordinal: Option<usize>,
    pub subset_status: &'static str,
    pub executed_closure_status: &'static str,
    pub complete_closure_enumerated: bool,
    pub interpreted_as_complete_closure: bool,
    pub formal_roots: Option<String>,
    pub target_or_split_modules_loaded: bool,
}

/// Replay the inherited 2,160-source target-free survivor subset through the
/// new normalized two-clause gate.  This is a subset witness, never closure.
pub fn replay_shrink4_capacity_subset(
) -> Result<Shrink4CapacityReplayReport, Shrink4Error> {
    let constant_atoms = capacity_constant_atoms();
    let rational_aggregates = capacity_rational_aggregates();
    let mixed_atoms = capacity_mixed_atoms();
    if constant_atoms.len() != 15 || rational_aggregates.len() != 16 || mixed_atoms.len() != 144 {
        return Err(Shrink4Error::new(
            REJECT_INTERNAL_SHRINK4_REPLAY,
            "shrink-4 survivor subset component count drift",
        ));
    }

    let mut source_count = 0usize;
    let mut accepted_count = 0usize;
    let mut parent_identity_match_count = 0usize;
    let mut rejection_counts = BTreeMap::new();
    let mut canonical_set = BTreeSet::new();
    for constant_atom in &constant_atoms {
        for mixed_atom in &mixed_atoms {
            source_count += 1;
            let source = Node::And(vec![constant_atom.clone(), mixed_atom.clone()]);
            let parent = canonicalize_shrink3_source_node(source.clone())?;
            match canonicalize_shrink4_source_node(source) {
                Ok(program) => {
                    if program.canonical_cbor != parent.canonical_cbor
                        || program.canonical_ast_hash != parent.canonical_ast_hash
                        || program.root_operator_id != parent.root_operator_id
                        || program.node_count != parent.node_count
                        || program.depth != parent.depth
                        || program.scalar_parameter_occurrence_count
                            != parent.scalar_parameter_occurrence_count
                    {
                        return Err(Shrink4Error::new(
                            REJECT_INTERNAL_SHRINK4_REPLAY,
                            format!("survivor identity changed at source ordinal {source_count}"),
                        ));
                    }
                    parent_identity_match_count += 1;
                    accepted_count += 1;
                    canonical_set.insert(program.canonical_cbor);
                }
                Err(error) => {
                    *rejection_counts.entry(error.code).or_insert(0) += 1;
                }
            }
        }
    }

    let rejected_count = rejection_counts.values().sum();
    let accepted_unique_count = canonical_set.len();
    let rewrite_collapsed_count = accepted_count.checked_sub(accepted_unique_count).ok_or_else(|| {
        Shrink4Error::new(
            REJECT_INTERNAL_SHRINK4_REPLAY,
            "unique count exceeds accepted count",
        )
    })?;
    let commitment = capacity_set_commitment(&canonical_set);
    let first = canonical_set.iter().next().ok_or_else(|| {
        Shrink4Error::new(REJECT_INTERNAL_SHRINK4_REPLAY, "survivor set is empty")
    })?;
    let last = canonical_set.iter().next_back().ok_or_else(|| {
        Shrink4Error::new(REJECT_INTERNAL_SHRINK4_REPLAY, "survivor set is empty")
    })?;
    let first_program = decode_shrink4_canonical_ast(first)?;
    let last_program = decode_shrink4_canonical_ast(last)?;
    let first_hex = hegel_strict_canonicalizer::hex_encode(first);
    let last_hex = hegel_strict_canonicalizer::hex_encode(last);

    if source_count != EXPECTED_SURVIVOR_SOURCE_COUNT
        || accepted_count != EXPECTED_SURVIVOR_SOURCE_COUNT
        || accepted_unique_count != EXPECTED_SURVIVOR_SOURCE_COUNT
        || parent_identity_match_count != EXPECTED_SURVIVOR_SOURCE_COUNT
        || rejected_count != 0
        || !rejection_counts.is_empty()
        || rewrite_collapsed_count != 0
        || commitment != EXPECTED_SURVIVOR_ACCEPTED_SET_COMMITMENT
        || first_hex != EXPECTED_FIRST_CANONICAL_CBOR_HEX
        || last_hex != EXPECTED_LAST_CANONICAL_CBOR_HEX
    {
        return Err(Shrink4Error::new(
            REJECT_INTERNAL_SHRINK4_REPLAY,
            format!(
                "frozen survivor subset invariant failure: source={source_count}, accepted={accepted_count}, unique={accepted_unique_count}, rejected={rejected_count}, collapsed={rewrite_collapsed_count}, commitment={commitment}, first={first_hex}, last={last_hex}"
            ),
        ));
    }

    Ok(Shrink4CapacityReplayReport {
        schema_version: CAPACITY_SCHEMA_VERSION,
        implementation: "rust",
        parent_dsl_version: PARENT_DSL_VERSION,
        parent_freeze_version: PARENT_FREEZE_VERSION,
        dsl_version: DSL_VERSION,
        freeze_version: FREEZE_VERSION,
        human_amendment_id: HUMAN_AMENDMENT_ID,
        shrink_step_id: SHRINK_STEP_ID,
        generator_rule: SURVIVOR_CAPACITY_GENERATOR_RULE,
        removed_binary_operator_ids: TOMBSTONED_BINARY_OPERATOR_IDS,
        retained_difference_id: 1,
        maximum_top_level_clauses: MAX_TOP_LEVEL_CLAUSES,
        constant_atom_count: constant_atoms.len(),
        rational_aggregate_count: rational_aggregates.len(),
        mixed_atom_count: mixed_atoms.len(),
        source_candidate_count: source_count,
        normalized_and2_count: accepted_count,
        accepted_source_count: accepted_count,
        accepted_unique_count,
        parent_identity_match_count,
        rejected_count,
        rejection_counts,
        rewrite_collapsed_count,
        accepted_set_commitment: commitment,
        first_canonical_cbor_hex: first_hex,
        first_canonical_ast_hash: first_program.canonical_ast_hash_id(),
        last_canonical_cbor_hex: last_hex,
        last_canonical_ast_hash: last_program.canonical_ast_hash_id(),
        canonical_program_budget: 50_000,
        first_out_of_budget_ordinal: None,
        subset_status: "FULL_AND2_SURVIVOR_SET_ONLY_NOT_COMPLETE",
        executed_closure_status: "NOT_RUN",
        complete_closure_enumerated: false,
        interpreted_as_complete_closure: false,
        formal_roots: None,
        target_or_split_modules_loaded: false,
    })
}

#[derive(Debug, Clone, Serialize)]
pub struct Shrink4GoldenReplayReport {
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
    pub golden_vector_manifest_root: &'static str,
    pub golden_outcome_root: &'static str,
    pub ordered_vector_ids: [&'static str; 22],
    pub execution_state: &'static str,
    pub closure_executed: bool,
    pub formal_roots_generated: bool,
    pub formal_roots: Option<String>,
    pub target_or_split_modules_loaded: bool,
}

fn golden_failure(message: impl Into<String>) -> Shrink4Error {
    Shrink4Error::new(REJECT_INTERNAL_SHRINK4_REPLAY, message)
}

fn expect_source_error(value: &Value, expected: &str, label: &str) -> Result<(), Shrink4Error> {
    match canonicalize_shrink4_source_json(value) {
        Ok(_) => Err(golden_failure(format!("{label}: unexpectedly accepted"))),
        Err(error) if error.code == expected => Ok(()),
        Err(error) => Err(golden_failure(format!(
            "{label}: expected {expected}, got {}",
            error.code
        ))),
    }
}

fn formal_bytes(value: Value) -> Result<Vec<u8>, Shrink4Error> {
    Ok(encode_strict_cbor_json(&value)?)
}

fn expect_formal_error(value: Value, expected: &str, label: &str) -> Result<(), Shrink4Error> {
    let bytes = formal_bytes(value)?;
    match decode_shrink4_canonical_ast(&bytes) {
        Ok(_) => Err(golden_failure(format!("{label}: unexpectedly accepted"))),
        Err(error) if error.code == expected => Ok(()),
        Err(error) => Err(golden_failure(format!(
            "{label}: expected {expected}, got {}",
            error.code
        ))),
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

fn check_survivor_identity(source: &Value, label: &str) -> Result<(), Shrink4Error> {
    let parent = canonicalize_shrink3_source_json(source)?;
    let child = canonicalize_shrink4_source_json(source)?;
    if child.canonical_cbor != parent.canonical_cbor
        || child.canonical_ast_hash != parent.canonical_ast_hash
        || child.root_operator_id != parent.root_operator_id
        || child.node_count != parent.node_count
        || child.depth != parent.depth
        || child.scalar_parameter_occurrence_count != parent.scalar_parameter_occurrence_count
    {
        return Err(golden_failure(format!(
            "{label}: surviving source changed canonical/derived identity"
        )));
    }
    let parent_formal = decode_shrink3_canonical_ast(&parent.canonical_cbor)?;
    let child_formal = decode_shrink4_canonical_ast(&parent.canonical_cbor)?;
    if child_formal.canonical_cbor != parent_formal.canonical_cbor
        || child_formal.canonical_ast_hash != parent_formal.canonical_ast_hash
        || child_formal.root_operator_id != parent_formal.root_operator_id
        || child_formal.node_count != parent_formal.node_count
        || child_formal.depth != parent_formal.depth
        || child_formal.scalar_parameter_occurrence_count
            != parent_formal.scalar_parameter_occurrence_count
    {
        return Err(golden_failure(format!(
            "{label}: surviving formal program changed canonical/derived identity"
        )));
    }
    Ok(())
}

/// Replay the exact shared Python/Rust 22-vector shrink-4 manifest.
pub fn replay_shrink4_golden_vectors() -> Result<Shrink4GoldenReplayReport, Shrink4Error> {
    let mut vector_count = 0usize;
    let mut surviving_identity_checks = 0usize;
    let mut source_normalization_before_limit_checks = 0usize;
    let mut source_structural_limit_checks = 0usize;
    let mut source_priority_checks = 0usize;
    let mut formal_surviving_identity_checks = 0usize;
    let mut formal_structural_limit_checks = 0usize;
    let mut formal_priority_checks = 0usize;

    for (label, source) in [
        ("scalar survivor", json!(["scalar_const", 1])),
        (
            "difference survivor",
            json!(["difference", ["scalar_const", 1], ["scalar_const", 5]]),
        ),
        ("AND2 survivor", json!(["top_level_AND", atom(0), atom(1)])),
    ] {
        check_survivor_identity(&source, label)?;
        vector_count += 1;
        surviving_identity_checks += 1;
    }

    for (label, source) in [
        ("single raw clause collapse", json!(["top_level_AND", atom(0)])),
        (
            "duplicate raw clause collapse",
            json!(["top_level_AND", atom(0), atom(0), atom(1)]),
        ),
    ] {
        check_survivor_identity(&source, label)?;
        vector_count += 1;
        source_normalization_before_limit_checks += 1;
    }

    for (label, source) in [
        (
            "direct source AND3",
            json!(["top_level_AND", atom(0), atom(1), atom(2)]),
        ),
        (
            "nested source AND3",
            json!([
                "top_level_AND",
                atom(0),
                ["top_level_AND", atom(1), atom(2)]
            ]),
        ),
    ] {
        expect_source_error(&source, REJECT_STRUCTURAL_LIMIT, label)?;
        vector_count += 1;
        source_structural_limit_checks += 1;
    }

    for (label, source, expected) in [
        (
            "aggregate tombstone before source clause limit",
            json!([
                "top_level_AND",
                atom(0),
                atom(1),
                [
                    "equal_exact",
                    ["aggregate", "mean_v1", "scope_all_observed_v1", "q0", []],
                    ["scalar_const", 1]
                ]
            ]),
            REJECT_REMOVED_AGGREGATE_MAP,
        ),
        (
            "rational tombstone before source clause limit",
            json!([
                "top_level_AND",
                atom(0),
                atom(1),
                ["equal_exact", ["scalar_const", -2, 1], ["scalar_const", 1]]
            ]),
            REJECT_REMOVED_RATIONAL_PARAMETER,
        ),
        (
            "add tombstone before source clause limit",
            json!([
                "top_level_AND",
                atom(0),
                atom(1),
                [
                    "equal_exact",
                    ["add", ["scalar_const", 1], ["scalar_const", 5]],
                    ["scalar_const", 3]
                ]
            ]),
            REJECT_REMOVED_BINARY_OPERATOR,
        ),
        (
            "type mismatch before source clause limit",
            json!([
                "top_level_AND",
                atom(0),
                atom(1),
                ["scalar_const", 1]
            ]),
            REJECT_TYPE_MISMATCH,
        ),
        (
            "malformed child before source clause limit",
            json!([
                "top_level_AND",
                atom(0),
                atom(1),
                ["add", ["scalar_const", 1]]
            ]),
            REJECT_MALFORMED_SOURCE_AST,
        ),
    ] {
        expect_source_error(&source, expected, label)?;
        vector_count += 1;
        source_priority_checks += 1;
    }

    let canonical_and2 = canonicalize_shrink3_source_json(&json!([
        "top_level_AND",
        atom(0),
        atom(1)
    ]))?;
    let accepted_formal = decode_shrink4_canonical_ast(&canonical_and2.canonical_cbor)?;
    if accepted_formal.canonical_cbor != canonical_and2.canonical_cbor
        || accepted_formal.canonical_ast_hash != canonical_and2.canonical_ast_hash
    {
        return Err(golden_failure("formal AND2 identity changed"));
    }
    vector_count += 1;
    formal_surviving_identity_checks += 1;

    let canonical_and3 = canonicalize_shrink3_source_json(&json!([
        "top_level_AND",
        atom(0),
        atom(1),
        atom(2)
    ]))?;
    match decode_shrink4_canonical_ast(&canonical_and3.canonical_cbor) {
        Err(error) if error.code == REJECT_STRUCTURAL_LIMIT => {}
        Err(error) => {
            return Err(golden_failure(format!(
                "formal canonical AND3 expected {REJECT_STRUCTURAL_LIMIT}, got {}",
                error.code
            )))
        }
        Ok(_) => return Err(golden_failure("formal canonical AND3 unexpectedly accepted")),
    }
    vector_count += 1;
    formal_structural_limit_checks += 1;

    for (label, formal, expected) in [
        (
            "formal noncanonical order before clause limit",
            json!([1, [4, [formal_atom(2), formal_atom(1), formal_atom(0)]]]),
            REJECT_NONCANONICAL_AST,
        ),
        (
            "formal aggregate tombstone before clause limit",
            json!([
                1,
                [
                    4,
                    [
                        formal_atom(0),
                        formal_atom(1),
                        [2, 2, [0, 3, 2, 0, 0, []], [0, 0, 1]]
                    ]
                ]
            ]),
            REJECT_REMOVED_AGGREGATE_MAP,
        ),
        (
            "formal rational tombstone before clause limit",
            json!([
                1,
                [
                    4,
                    [
                        formal_atom(0),
                        formal_atom(1),
                        [2, 2, [0, 0, 0], [0, 0, 1]]
                    ]
                ]
            ]),
            REJECT_REMOVED_RATIONAL_PARAMETER,
        ),
        (
            "formal add tombstone before clause limit",
            json!([
                1,
                [
                    4,
                    [
                        formal_atom(0),
                        formal_atom(1),
                        [2, 2, [2, 0, [0, 0, 1], [0, 0, 5]], [0, 0, 3]]
                    ]
                ]
            ]),
            REJECT_REMOVED_BINARY_OPERATOR,
        ),
        (
            "formal source-only alias before clause limit",
            json!([
                1,
                [
                    4,
                    [
                        formal_atom(0),
                        formal_atom(1),
                        [2, 4, [0, 0, 1], [0, 0, 5]]
                    ]
                ]
            ]),
            REJECT_NONCANONICAL_AST,
        ),
        (
            "formal reserved operator before clause limit",
            json!([
                1,
                [
                    4,
                    [
                        formal_atom(0),
                        formal_atom(1),
                        [2, 7, [0, 0, 1], [0, 0, 5]]
                    ]
                ]
            ]),
            REJECT_NONCANONICAL_AST,
        ),
        (
            "formal type mismatch before clause limit",
            json!([1, [4, [formal_atom(0), formal_atom(1), [0, 0, 1]]]]),
            REJECT_TYPE_MISMATCH,
        ),
        (
            "formal AND4 shape before clause limit",
            json!([
                1,
                [
                    4,
                    [formal_atom(0), formal_atom(1), formal_atom(2), formal_atom(3)]
                ]
            ]),
            REJECT_NONCANONICAL_AST,
        ),
    ] {
        expect_formal_error(formal, expected, label)?;
        vector_count += 1;
        formal_priority_checks += 1;
    }

    if vector_count != 22 {
        return Err(golden_failure(format!(
            "golden vector count drift: {vector_count}"
        )));
    }

    Ok(Shrink4GoldenReplayReport {
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
        maximum_top_level_clauses: MAX_TOP_LEVEL_CLAUSES,
        vector_count,
        passed_count: vector_count,
        surviving_identity_checks,
        source_normalization_before_limit_checks,
        source_structural_limit_checks,
        source_priority_checks,
        formal_surviving_identity_checks,
        formal_structural_limit_checks,
        formal_priority_checks,
        golden_vector_manifest_root: EXPECTED_GOLDEN_MANIFEST_ROOT,
        golden_outcome_root: EXPECTED_GOLDEN_OUTCOME_ROOT,
        ordered_vector_ids: ORDERED_GOLDEN_VECTOR_IDS,
        execution_state: "NOT_RUN",
        closure_executed: false,
        formal_roots_generated: false,
        formal_roots: None,
        target_or_split_modules_loaded: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golden_replay_covers_frozen_shrink4_contract() {
        let report = replay_shrink4_golden_vectors().unwrap();
        assert_eq!(report.vector_count, 22);
        assert_eq!(report.passed_count, report.vector_count);
        assert_eq!(report.surviving_identity_checks, 3);
        assert_eq!(report.source_normalization_before_limit_checks, 2);
        assert_eq!(report.source_structural_limit_checks, 2);
        assert_eq!(report.source_priority_checks, 5);
        assert_eq!(report.formal_surviving_identity_checks, 1);
        assert_eq!(report.formal_structural_limit_checks, 1);
        assert_eq!(report.formal_priority_checks, 8);
        assert_eq!(report.maximum_top_level_clauses, 2);
        assert_eq!(report.golden_vector_manifest_root, EXPECTED_GOLDEN_MANIFEST_ROOT);
        assert_eq!(report.golden_outcome_root, EXPECTED_GOLDEN_OUTCOME_ROOT);
        assert_eq!(report.ordered_vector_ids, ORDERED_GOLDEN_VECTOR_IDS);
    }

    #[test]
    fn survivor_capacity_commitment_is_unchanged_and_not_complete() {
        let report = replay_shrink4_capacity_subset().unwrap();
        assert_eq!(report.source_candidate_count, 2_160);
        assert_eq!(report.accepted_unique_count, 2_160);
        assert_eq!(
            report.accepted_set_commitment,
            EXPECTED_SURVIVOR_ACCEPTED_SET_COMMITMENT
        );
        assert_eq!(report.normalized_and2_count, 2_160);
        assert_eq!(
            report.subset_status,
            "FULL_AND2_SURVIVOR_SET_ONLY_NOT_COMPLETE"
        );
        assert!(!report.complete_closure_enumerated);
        assert!(!report.interpreted_as_complete_closure);
    }

    #[test]
    fn direct_and_nested_three_clause_sources_reject_after_normalization() {
        for source in [
            json!(["top_level_AND", atom(0), atom(1), atom(2)]),
            json!([
                "top_level_AND",
                atom(0),
                ["top_level_AND", atom(1), atom(2)]
            ]),
        ] {
            let error = canonicalize_shrink4_source_json(&source).unwrap_err();
            assert_eq!(error.code, REJECT_STRUCTURAL_LIMIT);
        }
    }

    #[test]
    fn duplicate_collapse_to_two_clauses_is_accepted() {
        let source = json!(["top_level_AND", atom(0), atom(1), atom(0)]);
        let child = canonicalize_shrink4_source_json(&source).unwrap();
        let expected = canonicalize_shrink3_source_json(&json!([
            "top_level_AND",
            atom(0),
            atom(1)
        ]))
        .unwrap();
        assert_eq!(child.canonical_cbor, expected.canonical_cbor);
        assert_eq!(child.canonical_ast_hash, expected.canonical_ast_hash);
    }

    #[test]
    fn inherited_errors_precede_the_new_source_limit() {
        let registry = json!(["top_level_AND", atom(0), atom(1), atom(99)]);
        assert_eq!(
            canonicalize_shrink4_source_json(&registry).unwrap_err().code,
            hegel_strict_canonicalizer::REJECT_REGISTRY_INDEX_OUT_OF_RANGE
        );

        let removed = json!([
            "top_level_AND",
            atom(0),
            atom(1),
            [
                "equal_exact",
                ["add", ["scalar_const", 1], ["scalar_const", 5]],
                ["scalar_const", 1]
            ]
        ]);
        assert_eq!(
            canonicalize_shrink4_source_json(&removed).unwrap_err().code,
            REJECT_REMOVED_BINARY_OPERATOR
        );
    }

    #[test]
    fn formal_and3_rejects_but_noncanonical_and_tombstone_priorities_survive() {
        let canonical_and3 = canonicalize_shrink3_source_json(&json!([
            "top_level_AND",
            atom(0),
            atom(1),
            atom(2)
        ]))
        .unwrap();
        assert_eq!(
            decode_shrink4_canonical_ast(&canonical_and3.canonical_cbor)
                .unwrap_err()
                .code,
            REJECT_STRUCTURAL_LIMIT
        );

        let noncanonical = formal_bytes(json!([
            1,
            [4, [formal_atom(2), formal_atom(1), formal_atom(0)]]
        ]))
        .unwrap();
        assert_eq!(
            decode_shrink4_canonical_ast(&noncanonical).unwrap_err().code,
            REJECT_NONCANONICAL_AST
        );

        let removed = formal_bytes(json!([
            1,
            [
                4,
                [
                    formal_atom(0),
                    formal_atom(1),
                    [2, 2, [2, 0, [0, 0, 1], [0, 0, 5]], [0, 0, 1]]
                ]
            ]
        ]))
        .unwrap();
        assert_eq!(
            decode_shrink4_canonical_ast(&removed).unwrap_err().code,
            REJECT_REMOVED_BINARY_OPERATOR
        );
    }

    #[test]
    fn survivor_identity_includes_all_strict_mdl_inputs() {
        for source in [
            json!(["difference", ["scalar_const", 5], ["scalar_const", 1]]),
            json!(["top_level_AND", atom(0), atom(1)]),
        ] {
            let parent = canonicalize_shrink3_source_json(&source).unwrap();
            let child = canonicalize_shrink4_source_json(&source).unwrap();
            assert_eq!(child.canonical_cbor, parent.canonical_cbor);
            assert_eq!(child.canonical_ast_hash, parent.canonical_ast_hash);
            assert_eq!(child.root_operator_id, parent.root_operator_id);
            assert_eq!(child.node_count, parent.node_count);
            assert_eq!(child.depth, parent.depth);
            assert_eq!(
                child.scalar_parameter_occurrence_count,
                parent.scalar_parameter_occurrence_count
            );
        }
    }
}
