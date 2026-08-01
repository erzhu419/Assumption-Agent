//! Independent Rust admission profile for `hegel-old-dsl-v1.1.0`.
//!
//! The parent crate remains byte-for-byte unchanged. This sibling checks the
//! sparse AggregateMapId/v1 tombstones before type checking/normalization and
//! then delegates unchanged AST/CBOR mechanics to the independently verified
//! parent Rust implementation.

use hegel_strict_canonicalizer::{
    canonicalize_source_node, parse_source_ast, validate_strict_cbor, BinaryOp, CanonicalError,
    CanonicalProgram, Node, Sort,
};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt;

pub const DSL_VERSION: &str = "hegel-old-dsl-v1.1.0";
pub const FREEZE_VERSION: &str = "hegel-freeze-p2b-p3-v1.1.0";
pub const PARENT_DSL_VERSION: &str = "hegel-old-dsl-v1.0.0";
pub const AST_SCHEMA_ID: &str = "hegel-canonical-ast-v1";
pub const CBOR_PROFILE_ID: &str = "hegel-cbor-det-v1";
pub const AST_HASH_DOMAIN: &str = "HEGEL/AST/V1";
pub const REPLAY_SCHEMA_VERSION: &str = "hegel-strict-canonicalizer-shrink1-replay/1";
pub const REJECT_REMOVED_AGGREGATE_MAP: &str = "REJECT_REMOVED_AGGREGATE_MAP";
pub const REJECT_INTERNAL_SHRINK1_REPLAY: &str = "REJECT_INTERNAL_SHRINK1_REPLAY";
pub const EXPECTED_SHRINK1_SOURCE_COUNT: usize = 25_872;
pub const CANONICAL_PROGRAM_BUDGET: usize = 50_000;
pub const EXPECTED_SHRINK1_ACCEPTED_SET_COMMITMENT: &str =
    "sha256:653fcb9428684cfed11c3f2345ac95ed98ded6e31564c9eeabf97c57ee71a7e9";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shrink1Error {
    pub code: String,
    pub message: String,
}

impl Shrink1Error {
    fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }
}

impl From<CanonicalError> for Shrink1Error {
    fn from(error: CanonicalError) -> Self {
        Self::new(error.code, error.message)
    }
}

impl fmt::Display for Shrink1Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for Shrink1Error {}

fn reject_removed_nodes(node: &Node) -> Result<(), Shrink1Error> {
    match node {
        Node::Aggregate { map_id: 2..=4, .. } => Err(Shrink1Error::new(
            REJECT_REMOVED_AGGREGATE_MAP,
            "AggregateMapId 2, 3, or 4 is tombstoned in hegel-old-dsl-v1.1.0",
        )),
        Node::Unary { child, .. } => reject_removed_nodes(child),
        Node::Binary { left, right, .. } => {
            reject_removed_nodes(left)?;
            reject_removed_nodes(right)
        }
        Node::ApproxEqual { left, right, .. } => {
            reject_removed_nodes(left)?;
            reject_removed_nodes(right)
        }
        Node::And(children) => children.iter().try_for_each(reject_removed_nodes),
        _ => Ok(()),
    }
}

fn reject_source_tombstones(value: &Value) -> Result<(), Shrink1Error> {
    let Some(items) = value.as_array() else {
        return Ok(());
    };
    let Some(name) = items.first().and_then(Value::as_str) else {
        return Ok(());
    };
    match name {
        "aggregate" if items.len() == 5 => {
            let removed_name = matches!(items[1].as_str(), Some("mean_v1" | "min_v1" | "max_v1"));
            let removed_id = matches!(items[1].as_u64(), Some(2..=4));
            if removed_name || removed_id {
                return Err(Shrink1Error::new(
                    REJECT_REMOVED_AGGREGATE_MAP,
                    "aggregate map is tombstoned in hegel-old-dsl-v1.1.0",
                ));
            }
        }
        "bit_to_scalar" | "int_to_scalar" | "absolute" | "sign" if items.len() == 2 => {
            reject_source_tombstones(&items[1])?;
        }
        "add" | "difference" | "equal_exact" | "less_equal" | "greater_equal" | "same_sign"
        | "opposite_sign"
            if items.len() == 3 =>
        {
            reject_source_tombstones(&items[1])?;
            reject_source_tombstones(&items[2])?;
        }
        "approx_equal" if items.len() == 4 || items.len() == 5 => {
            reject_source_tombstones(&items[1])?;
            reject_source_tombstones(&items[2])?;
        }
        "top_level_AND" if items.len() >= 2 => {
            let nested_children = (items.len() == 2)
                .then(|| items[1].as_array())
                .flatten()
                .filter(|children| {
                    !children.is_empty()
                        && children.iter().all(|child| {
                            child
                                .as_array()
                                .is_some_and(|child_items| !child_items.is_empty())
                        })
                });
            if let Some(children) = nested_children {
                for child in children {
                    reject_source_tombstones(child)?;
                }
            } else {
                for child in &items[1..] {
                    reject_source_tombstones(child)?;
                }
            }
        }
        _ => {}
    }
    Ok(())
}

/// Parse source JSON, reject tombstones, then run unchanged strict mechanics.
pub fn canonicalize_shrink1_source_json(value: &Value) -> Result<CanonicalProgram, Shrink1Error> {
    reject_source_tombstones(value)?;
    let source = parse_source_ast(value)?;
    canonicalize_shrink1_source_node(source)
}

/// Child-profile entry point used by the independent Rust subset generator.
pub fn canonicalize_shrink1_source_node(source: Node) -> Result<CanonicalProgram, Shrink1Error> {
    reject_removed_nodes(&source)?;
    canonicalize_source_node(source).map_err(Into::into)
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ProbeValue {
    Unsigned(u64),
    Negative,
    Bytes,
    Array(Vec<ProbeValue>),
    Bool,
    Null,
}

fn read_argument(additional: u8, bytes: &[u8], cursor: &mut usize) -> Result<u64, Shrink1Error> {
    let width = match additional {
        0..=23 => return Ok(u64::from(additional)),
        24 => 1,
        25 => 2,
        26 => 4,
        27 => 8,
        _ => {
            return Err(Shrink1Error::new(
                REJECT_INTERNAL_SHRINK1_REPLAY,
                "validated CBOR carried an unsupported argument width",
            ))
        }
    };
    let end = cursor
        .checked_add(width)
        .ok_or_else(|| Shrink1Error::new(REJECT_INTERNAL_SHRINK1_REPLAY, "CBOR cursor overflow"))?;
    let payload = bytes.get(*cursor..end).ok_or_else(|| {
        Shrink1Error::new(REJECT_INTERNAL_SHRINK1_REPLAY, "truncated validated CBOR")
    })?;
    *cursor = end;
    let mut value = 0u64;
    for byte in payload {
        value = (value << 8) | u64::from(*byte);
    }
    Ok(value)
}

fn parse_probe(bytes: &[u8], cursor: &mut usize) -> Result<ProbeValue, Shrink1Error> {
    let initial = *bytes.get(*cursor).ok_or_else(|| {
        Shrink1Error::new(
            REJECT_INTERNAL_SHRINK1_REPLAY,
            "missing validated CBOR item",
        )
    })?;
    *cursor += 1;
    let major = initial >> 5;
    let additional = initial & 0x1f;
    match major {
        0 => Ok(ProbeValue::Unsigned(read_argument(
            additional, bytes, cursor,
        )?)),
        1 => {
            read_argument(additional, bytes, cursor)?;
            Ok(ProbeValue::Negative)
        }
        2 => {
            let length =
                usize::try_from(read_argument(additional, bytes, cursor)?).map_err(|_| {
                    Shrink1Error::new(REJECT_INTERNAL_SHRINK1_REPLAY, "byte length overflow")
                })?;
            let end = cursor.checked_add(length).ok_or_else(|| {
                Shrink1Error::new(REJECT_INTERNAL_SHRINK1_REPLAY, "byte cursor overflow")
            })?;
            if bytes.get(*cursor..end).is_none() {
                return Err(Shrink1Error::new(
                    REJECT_INTERNAL_SHRINK1_REPLAY,
                    "truncated validated byte string",
                ));
            }
            *cursor = end;
            Ok(ProbeValue::Bytes)
        }
        4 => {
            let length =
                usize::try_from(read_argument(additional, bytes, cursor)?).map_err(|_| {
                    Shrink1Error::new(REJECT_INTERNAL_SHRINK1_REPLAY, "array length overflow")
                })?;
            let mut values = Vec::with_capacity(length);
            for _ in 0..length {
                values.push(parse_probe(bytes, cursor)?);
            }
            Ok(ProbeValue::Array(values))
        }
        7 if additional == 20 || additional == 21 => Ok(ProbeValue::Bool),
        7 if additional == 22 => Ok(ProbeValue::Null),
        _ => Err(Shrink1Error::new(
            REJECT_INTERNAL_SHRINK1_REPLAY,
            "validated CBOR carried an unsupported value",
        )),
    }
}

fn reject_probe_node_tombstones(value: &ProbeValue) -> Result<(), Shrink1Error> {
    let ProbeValue::Array(items) = value else {
        return Ok(());
    };
    match items.first() {
        Some(ProbeValue::Unsigned(0)) if items.len() == 6 => {
            if items.get(1) == Some(&ProbeValue::Unsigned(3))
                && matches!(items.get(2), Some(ProbeValue::Unsigned(2..=4)))
            {
                return Err(Shrink1Error::new(
                    REJECT_REMOVED_AGGREGATE_MAP,
                    "formal AggregateMapId 2, 3, or 4 is tombstoned in hegel-old-dsl-v1.1.0",
                ));
            }
        }
        Some(ProbeValue::Unsigned(1)) if items.len() == 3 => {
            reject_probe_node_tombstones(&items[2])?;
        }
        Some(ProbeValue::Unsigned(2)) if items.len() == 4 => {
            reject_probe_node_tombstones(&items[2])?;
            reject_probe_node_tombstones(&items[3])?;
        }
        Some(ProbeValue::Unsigned(3)) if items.len() == 5 => {
            reject_probe_node_tombstones(&items[2])?;
            reject_probe_node_tombstones(&items[3])?;
        }
        Some(ProbeValue::Unsigned(4)) if items.len() == 2 => {
            if let Some(ProbeValue::Array(children)) = items.get(1) {
                for child in children {
                    reject_probe_node_tombstones(child)?;
                }
            }
        }
        _ => {}
    }
    Ok(())
}

fn reject_probe_tombstones(value: &ProbeValue) -> Result<(), Shrink1Error> {
    let ProbeValue::Array(envelope) = value else {
        return Ok(());
    };
    if envelope.len() != 2 || envelope.first() != Some(&ProbeValue::Unsigned(1)) {
        return Ok(());
    }
    reject_probe_node_tombstones(&envelope[1])
}

/// Validate generic deterministic CBOR, reject tombstones, then decode AST.
pub fn decode_shrink1_canonical_ast(bytes: &[u8]) -> Result<CanonicalProgram, Shrink1Error> {
    validate_strict_cbor(bytes)?;
    let mut cursor = 0usize;
    let probe = parse_probe(bytes, &mut cursor)?;
    if cursor != bytes.len() {
        return Err(Shrink1Error::new(
            REJECT_INTERNAL_SHRINK1_REPLAY,
            "validated CBOR probe left trailing bytes",
        ));
    }
    reject_probe_tombstones(&probe)?;
    hegel_strict_canonicalizer::decode_strict_canonical_ast(bytes).map_err(Into::into)
}

fn capacity_constant_atoms() -> Vec<Node> {
    let constants = (0..7_u64).map(Node::ScalarConst).collect::<Vec<_>>();
    let mut atoms = Vec::with_capacity(77);
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
    let constants = (0..7_u64).map(Node::ScalarConst).collect::<Vec<_>>();
    let aggregates = capacity_rational_aggregates();
    let mut atoms = Vec::with_capacity(336);
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
pub struct Shrink1CapacityReplayReport {
    pub schema_version: &'static str,
    pub implementation: &'static str,
    pub dsl_version: &'static str,
    pub freeze_version: &'static str,
    pub source_candidate_count: usize,
    pub accepted_source_count: usize,
    pub accepted_unique_count: usize,
    pub rejected_count: usize,
    pub rejection_counts: std::collections::BTreeMap<String, usize>,
    pub rewrite_collapsed_count: usize,
    pub accepted_set_commitment: String,
    pub canonical_program_budget: usize,
    pub first_out_of_budget_ordinal: Option<usize>,
    pub first_out_of_budget_cbor_hex: Option<String>,
    pub first_out_of_budget_ast_hash: Option<String>,
    pub subset_status: &'static str,
    pub executed_closure_status: &'static str,
    pub complete_closure_enumerated: bool,
    pub interpreted_as_complete_closure: bool,
}

/// Independently construct and replay the frozen 25,872-source child subset.
pub fn replay_shrink1_capacity_subset() -> Result<Shrink1CapacityReplayReport, Shrink1Error> {
    let constant_atoms = capacity_constant_atoms();
    let mixed_atoms = capacity_mixed_atoms();
    if constant_atoms.len() != 77 || mixed_atoms.len() != 336 {
        return Err(Shrink1Error::new(
            REJECT_INTERNAL_SHRINK1_REPLAY,
            "shrink-1 generator component count drift",
        ));
    }
    let mut source_count = 0usize;
    let mut accepted_count = 0usize;
    let mut rejection_counts = std::collections::BTreeMap::new();
    let mut canonical_set = BTreeSet::new();
    for constant_atom in &constant_atoms {
        for mixed_atom in &mixed_atoms {
            source_count += 1;
            let source = Node::And(vec![constant_atom.clone(), mixed_atom.clone()]);
            match canonicalize_shrink1_source_node(source) {
                Ok(program) => {
                    accepted_count += 1;
                    canonical_set.insert(program.canonical_cbor);
                }
                Err(error) => {
                    *rejection_counts.entry(error.code).or_insert(0) += 1;
                }
            }
        }
    }
    if source_count != EXPECTED_SHRINK1_SOURCE_COUNT {
        return Err(Shrink1Error::new(
            REJECT_INTERNAL_SHRINK1_REPLAY,
            format!(
                "shrink-1 generator emitted {source_count}; expected {EXPECTED_SHRINK1_SOURCE_COUNT}"
            ),
        ));
    }
    let rejected_count = rejection_counts.values().sum();
    let accepted_unique_count = canonical_set.len();
    let rewrite_collapsed_count = accepted_count
        .checked_sub(accepted_unique_count)
        .ok_or_else(|| {
            Shrink1Error::new(
                REJECT_INTERNAL_SHRINK1_REPLAY,
                "unique count exceeds accepted count",
            )
        })?;
    let out_of_budget = canonical_set.iter().nth(CANONICAL_PROGRAM_BUDGET);
    let accepted_set_commitment = capacity_set_commitment(&canonical_set);
    let subset_invariants_hold = source_count == EXPECTED_SHRINK1_SOURCE_COUNT
        && accepted_count == EXPECTED_SHRINK1_SOURCE_COUNT
        && accepted_unique_count == EXPECTED_SHRINK1_SOURCE_COUNT
        && rejected_count == 0
        && rejection_counts.is_empty()
        && rewrite_collapsed_count == 0
        && accepted_set_commitment == EXPECTED_SHRINK1_ACCEPTED_SET_COMMITMENT
        && out_of_budget.is_none();
    if !subset_invariants_hold {
        return Err(Shrink1Error::new(
            REJECT_INTERNAL_SHRINK1_REPLAY,
            format!(
                "frozen subset invariant failure: source={source_count}, accepted={accepted_count}, \
                 unique={accepted_unique_count}, rejected={rejected_count}, \
                 collapsed={rewrite_collapsed_count}, commitment={accepted_set_commitment}, \
                 witness_present={}",
                out_of_budget.is_some()
            ),
        ));
    }
    Ok(Shrink1CapacityReplayReport {
        schema_version: "hegel-strict-capacity-replay-shrink1/1",
        implementation: "rust",
        dsl_version: DSL_VERSION,
        freeze_version: FREEZE_VERSION,
        source_candidate_count: source_count,
        accepted_source_count: accepted_count,
        accepted_unique_count,
        rejected_count,
        rejection_counts,
        rewrite_collapsed_count,
        accepted_set_commitment,
        canonical_program_budget: CANONICAL_PROGRAM_BUDGET,
        first_out_of_budget_ordinal: out_of_budget.map(|_| CANONICAL_PROGRAM_BUDGET + 1),
        first_out_of_budget_cbor_hex: out_of_budget
            .map(|bytes| hegel_strict_canonicalizer::hex_encode(bytes)),
        first_out_of_budget_ast_hash: out_of_budget.map(|bytes| {
            let program = decode_shrink1_canonical_ast(bytes)
                .expect("generated canonical program must decode");
            program.canonical_ast_hash_id()
        }),
        subset_status: "VERIFIED_WITHIN_BUDGET",
        executed_closure_status: "NOT_RUN",
        complete_closure_enumerated: false,
        interpreted_as_complete_closure: false,
    })
}

pub fn sort_name(sort: Sort) -> &'static str {
    match sort {
        Sort::Bool => "Bool",
        Sort::Bit => "Bit",
        Sort::Sign => "Sign",
        Sort::BoundedInt => "BoundedInt",
        Sort::RationalValue => "RationalValue",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hegel_strict_canonicalizer::{
        canonicalize_source_json, decode_strict_canonical_ast, encode_strict_cbor_json,
    };
    use serde_json::json;

    #[test]
    fn surviving_signed_balance_keeps_parent_identity() {
        let source = json!([
            "aggregate",
            "signed_balance_v1",
            "scope_all_observed_v1",
            "q0",
            []
        ]);
        let parent = canonicalize_source_json(&source).unwrap();
        let child = canonicalize_shrink1_source_json(&source).unwrap();
        assert_eq!(parent.canonical_cbor, child.canonical_cbor);
        assert_eq!(parent.canonical_ast_hash, child.canonical_ast_hash);
        assert_eq!(
            child.canonical_node,
            Node::Aggregate {
                map_id: 5,
                scope_id: 0,
                quantity_id: 0,
                scope_extension: Vec::new(),
            }
        );
    }

    #[test]
    fn source_tombstones_are_rejected_before_canonicalization() {
        for map in [
            json!("mean_v1"),
            json!("min_v1"),
            json!("max_v1"),
            json!(2),
            json!(3),
            json!(4),
        ] {
            let source = json!(["aggregate", map, "scope_all_observed_v1", "q0", []]);
            let error = canonicalize_shrink1_source_json(&source).unwrap_err();
            assert_eq!(error.code, REJECT_REMOVED_AGGREGATE_MAP);
        }
    }

    #[test]
    fn nested_and_atom_list_checks_the_first_real_ast_child() {
        let source = json!([
            "top_level_AND",
            [
                ["aggregate", "mean_v1", "scope_all_observed_v1", "q0", []],
                ["equal_exact", ["scalar_const", 0], ["scalar_const", 0]]
            ]
        ]);
        let error = canonicalize_shrink1_source_json(&source).unwrap_err();
        assert_eq!(error.code, REJECT_REMOVED_AGGREGATE_MAP);
    }

    #[test]
    fn source_precheck_does_not_scan_non_ast_payloads() {
        let cases = [
            json!([
                "scalar_const",
                0,
                ["aggregate", "mean_v1", "scope_all_observed_v1", "q0", []]
            ]),
            json!([
                "unknown_outer",
                ["aggregate", "mean_v1", "scope_all_observed_v1", "q0", []]
            ]),
            json!([
                "top_level_AND",
                [[], ["aggregate", 2, "scope_all_observed_v1", "q0", []]]
            ]),
        ];
        for source in cases {
            let parent = canonicalize_source_json(&source).unwrap_err();
            let child = canonicalize_shrink1_source_json(&source).unwrap_err();
            assert_eq!(child.code, parent.code);
            assert_ne!(child.code, REJECT_REMOVED_AGGREGATE_MAP);
        }
    }

    #[test]
    fn formal_tombstone_is_generic_cbor_but_not_child_ast() {
        let source = json!(["aggregate", "mean_v1", "scope_all_observed_v1", "q0", []]);
        let parent = canonicalize_source_json(&source).unwrap();
        validate_strict_cbor(&parent.canonical_cbor).unwrap();
        decode_strict_canonical_ast(&parent.canonical_cbor).unwrap();
        let error = decode_shrink1_canonical_ast(&parent.canonical_cbor).unwrap_err();
        assert_eq!(error.code, REJECT_REMOVED_AGGREGATE_MAP);
    }

    #[test]
    fn formal_precheck_requires_the_v1_ast_envelope() {
        for formal_value in [json!([2, [0, 3, 2, 0, 0, []]]), json!([0, 3, 2, 0, 0, []])] {
            let bytes = encode_strict_cbor_json(&formal_value).unwrap();
            let parent = decode_strict_canonical_ast(&bytes).unwrap_err();
            let child = decode_shrink1_canonical_ast(&bytes).unwrap_err();
            assert_eq!(child.code, parent.code);
            assert_ne!(child.code, REJECT_REMOVED_AGGREGATE_MAP);
        }
    }

    #[test]
    fn real_formal_child_tombstone_precedes_noncanonical_and_arity() {
        let bytes = encode_strict_cbor_json(&json!([1, [4, [[0, 3, 2, 0, 0, []]]]])).unwrap();
        let parent = decode_strict_canonical_ast(&bytes).unwrap_err();
        assert_ne!(parent.code, REJECT_REMOVED_AGGREGATE_MAP);
        let child = decode_shrink1_canonical_ast(&bytes).unwrap_err();
        assert_eq!(child.code, REJECT_REMOVED_AGGREGATE_MAP);
    }

    #[test]
    fn shrink1_subset_is_not_complete_closure() {
        let report = replay_shrink1_capacity_subset().unwrap();
        assert_eq!(report.source_candidate_count, EXPECTED_SHRINK1_SOURCE_COUNT);
        assert_eq!(report.accepted_source_count, EXPECTED_SHRINK1_SOURCE_COUNT);
        assert_eq!(report.accepted_unique_count, EXPECTED_SHRINK1_SOURCE_COUNT);
        assert_eq!(report.rejected_count, 0);
        assert!(report.rejection_counts.is_empty());
        assert_eq!(report.rewrite_collapsed_count, 0);
        assert_eq!(
            report.accepted_set_commitment,
            EXPECTED_SHRINK1_ACCEPTED_SET_COMMITMENT
        );
        assert_eq!(report.subset_status, "VERIFIED_WITHIN_BUDGET");
        assert!(report.first_out_of_budget_ordinal.is_none());
        assert!(report.first_out_of_budget_cbor_hex.is_none());
        assert!(report.first_out_of_budget_ast_hash.is_none());
        assert!(!report.complete_closure_enumerated);
        assert_eq!(report.executed_closure_status, "NOT_RUN");
    }
}
