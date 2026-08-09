//! Independent Rust strict admission profile for `hegel-old-dsl-v1.2.0`.
//!
//! Shrink step 2 keeps the v1 numeric AST and deterministic-CBOR wire. It
//! inherits the sparse AggregateMap registry from shrink step 1 and makes
//! RationalParameterId/v1 sparse without renumbering: 1, 3, and 5 remain
//! active; 0, 2, 4, and 6 become permanent tombstones; 7 remains outside the
//! allocated registry. Constant folding is allowed only when the result is an
//! active parameter. This is intentionally not equivalent to filtering the
//! output of the parent normalizer: a valid child expression such as
//! `add(1, 1)` must retain its operator AST rather than first becoming the
//! tombstoned parameter 2.

use hegel_strict_canonicalizer::{
    canonicalize_source_node as canonicalize_parent_source_node, encode_strict_cbor_json,
    type_check, validate_strict_cbor, BinaryOp, CanonicalError, CanonicalProgram, Node, Sort,
    UnaryOp, REJECT_CBOR_FLOAT, REJECT_CBOR_MAP, REJECT_CBOR_NESTING, REJECT_CBOR_TAG,
    REJECT_CBOR_TEXT, REJECT_DUPLICATE_SCOPE_CONTEXT, REJECT_IMPLICIT_COERCION,
    REJECT_INDEFINITE_CBOR, REJECT_INTERNAL_CANONICALIZATION, REJECT_MALFORMED_SOURCE_AST,
    REJECT_NEW_SYMBOL_IN_OLD_DSL, REJECT_NONCANONICAL_AST, REJECT_NONCANONICAL_CBOR,
    REJECT_NONCANONICAL_SCOPE_ALIAS, REJECT_REGISTRY_INDEX_OUT_OF_RANGE,
    REJECT_STRUCTURAL_LIMIT, REJECT_TRAILING_CBOR, REJECT_TYPE_MISMATCH,
    REJECT_UNKNOWN_EXPRESSION,
};
pub use hegel_strict_canonicalizer_shrink1::REJECT_REMOVED_AGGREGATE_MAP;
use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub const DSL_VERSION: &str = "hegel-old-dsl-v1.2.0";
pub const FREEZE_VERSION: &str = "hegel-freeze-p2b-p3-v1.2.0";
pub const PARENT_DSL_VERSION: &str = "hegel-old-dsl-v1.1.0";
pub const PARENT_FREEZE_VERSION: &str = "hegel-freeze-p2b-p3-v1.1.2";
pub const HUMAN_AMENDMENT_ID: &str = "hegel-freeze-p2b-p3-v1.2.0-shrink-step2";
pub const SHRINK_STEP_ID: &str =
    "SHRINK_STEP_2_REDUCE_RATIONAL_PARAMETER_TO_NEG1_ZERO_POS1";
pub const AST_SCHEMA_ID: &str = "hegel-canonical-ast-v1";
pub const CBOR_PROFILE_ID: &str = "hegel-cbor-det-v1";
pub const AST_HASH_DOMAIN: &str = "HEGEL/AST/V1";
pub const REPLAY_SCHEMA_VERSION: &str = "hegel-strict-canonicalizer-shrink2-replay/1";
pub const GOLDEN_SCHEMA_VERSION: &str = "hegel-strict-canonicalizer-shrink2-golden/1";
pub const RATIONAL_PARAMETER_REGISTRY_NAMESPACE: &str = "RationalParameterId/v1";
pub const REJECT_REMOVED_RATIONAL_PARAMETER: &str = "REJECT_REMOVED_RATIONAL_PARAMETER";
pub const REJECT_UNKNOWN_AST_SCHEMA: &str = "REJECT_UNKNOWN_AST_SCHEMA";
pub const REJECT_INTERNAL_SHRINK2_REPLAY: &str = "REJECT_INTERNAL_SHRINK2_REPLAY";
pub const REJECT_EMPTY_CONJUNCTION: &str = "REJECT_EMPTY_CONJUNCTION";
pub const REJECT_TRUNCATED_CBOR: &str = "REJECT_TRUNCATED_CBOR";
pub const REJECT_RESERVED_CBOR: &str = "REJECT_RESERVED_CBOR";
pub const REJECT_CBOR_UNDEFINED: &str = "REJECT_CBOR_UNDEFINED";
pub const REJECT_CBOR_SIMPLE: &str = "REJECT_CBOR_SIMPLE";
pub const EXPECTED_SHRINK2_SOURCE_COUNT: usize = 2_160;
pub const CANONICAL_PROGRAM_BUDGET: usize = 50_000;
pub const EXPECTED_SHRINK2_ACCEPTED_SET_COMMITMENT: &str =
    "sha256:9045e4ebe6416dcbf699e7972f25468aef45c0f0aec0e58806061b7ce64d790e";
pub const EXPECTED_SHRINK2_FIRST_CANONICAL_CBOR_HEX: &str =
    "820182048284020283000001830000018402028300000186000300000180";
pub const EXPECTED_SHRINK2_LAST_CANONICAL_CBOR_HEX: &str =
    "820182048284020383000005830000058402038600030503018083000005";
pub const SHRINK2_CAPACITY_GENERATOR_RULE: &str =
    "15 active-constant comparison atoms x 144 active-constant/aggregate comparison atoms -> canonical top_level_AND Cartesian product; RationalParameterId/v1 active IDs are 1,3,5; rational AggregateMapId/v1 active IDs are 0,5; expected source count=2160";

pub const ACTIVE_RATIONAL_PARAMETER_IDS: [u64; 3] = [1, 3, 5];
pub const TOMBSTONED_RATIONAL_PARAMETER_IDS: [u64; 4] = [0, 2, 4, 6];
pub const RESERVED_RATIONAL_PARAMETER_IDS: [u64; 1] = [7];
pub const ACTIVE_AGGREGATE_MAP_IDS: [u64; 3] = [0, 1, 5];
pub const TOMBSTONED_AGGREGATE_MAP_IDS: [u64; 3] = [2, 3, 4];

const MAX_TOTAL_AST_DEPTH: u32 = 4;
const MAX_TOTAL_NODE_COUNT: u32 = 7;
const MAX_TOP_LEVEL_CLAUSES: usize = 3;
const MAX_DISTINCT_BIT_SLOTS: usize = 4;
const MAX_AGGREGATE_LEAVES: u32 = 1;
const MAX_FITTED_SCALAR_PARAMETER_OCCURRENCES: u32 = 3;
const MAX_CBOR_NESTING: usize = 64;
const ZERO_PARAMETER_INDEX: u64 = 3;
const RATIONAL_PARAMETERS: [(i64, i64); 7] =
    [(-2, 1), (-1, 1), (-1, 2), (0, 1), (1, 2), (1, 1), (2, 1)];
const TOLERANCES: [(i64, i64); 3] = [(0, 1), (1, 4), (1, 2)];
const AGGREGATE_MAP_NAMES: [&str; 6] = [
    "sum_v1",
    "count_nonzero_v1",
    "mean_v1",
    "min_v1",
    "max_v1",
    "signed_balance_v1",
];
const SCOPE_NAMES: [&str; 4] = [
    "scope_all_observed_v1",
    "scope_primary_only_v1",
    "scope_boundary_only_v1",
    "control_volume_all_observed_v1",
];
const QUANTITY_NAMES: [&str; 2] = ["q0", "q1"];
const CONTEXT_NAMES: [&str; 4] = ["c0", "c1", "c2", "c3"];
const TASK_NAMES: [&str; 2] = ["t0", "t1"];
const DEPRECATED_SCOPE_ALIAS: &str = "control_volume_primary_only_v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shrink2Error {
    pub code: String,
    pub message: String,
}

impl Shrink2Error {
    fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }
}

impl From<CanonicalError> for Shrink2Error {
    fn from(error: CanonicalError) -> Self {
        Self::new(error.code, error.message)
    }
}

impl fmt::Display for Shrink2Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for Shrink2Error {}

fn is_active_parameter(index: u64) -> bool {
    ACTIVE_RATIONAL_PARAMETER_IDS.contains(&index)
}

fn is_tombstoned_parameter(index: u64) -> bool {
    TOMBSTONED_RATIONAL_PARAMETER_IDS.contains(&index)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ExactJsonInteger {
    sign: i8,
    digits: String,
}

fn exact_json_integer(value: &Value) -> Option<ExactJsonInteger> {
    if !value.is_number() {
        return None;
    }
    let text = value.to_string();
    let (negative, digits) = text
        .strip_prefix('-')
        .map_or((false, text.as_str()), |digits| (true, digits));
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let normalized = digits.trim_start_matches('0');
    if normalized.is_empty() {
        return Some(ExactJsonInteger {
            sign: 0,
            digits: "0".to_owned(),
        });
    }
    Some(ExactJsonInteger {
        sign: if negative { -1 } else { 1 },
        digits: normalized.to_owned(),
    })
}

fn multiply_decimal_digits(digits: &str, factor: u8) -> String {
    if factor == 0 || digits == "0" {
        return "0".to_owned();
    }
    let mut carry = 0u16;
    let mut reversed = Vec::with_capacity(digits.len() + 1);
    for byte in digits.bytes().rev() {
        let product = u16::from(byte - b'0') * u16::from(factor) + carry;
        reversed.push((product % 10) as u8 + b'0');
        carry = product / 10;
    }
    while carry != 0 {
        reversed.push((carry % 10) as u8 + b'0');
        carry /= 10;
    }
    reversed.reverse();
    String::from_utf8(reversed).expect("decimal multiplication emitted ASCII")
}

fn signed_small_product(value: &ExactJsonInteger, factor: i64) -> (i8, String) {
    if value.sign == 0 || factor == 0 {
        return (0, "0".to_owned());
    }
    let sign = value.sign * if factor < 0 { -1 } else { 1 };
    let magnitude = u8::try_from(factor.unsigned_abs())
        .expect("frozen rational-grid factors fit in u8");
    (sign, multiply_decimal_digits(&value.digits, magnitude))
}

enum RationalBoundary {
    NonInteger,
    Index(u64),
    OutOfRange,
}

fn rational_grid_boundary(
    numerator: &Value,
    denominator: &Value,
    grid: &[(i64, i64)],
) -> RationalBoundary {
    let (Some(numerator), Some(denominator)) = (
        exact_json_integer(numerator),
        exact_json_integer(denominator),
    ) else {
        return RationalBoundary::NonInteger;
    };
    if denominator.sign <= 0 {
        return RationalBoundary::OutOfRange;
    }
    for (index, (candidate_numerator, candidate_denominator)) in grid.iter().enumerate() {
        let left = signed_small_product(&numerator, *candidate_denominator);
        let right = signed_small_product(&denominator, *candidate_numerator);
        if left == right {
            return RationalBoundary::Index(index as u64);
        }
    }
    RationalBoundary::OutOfRange
}

fn source_array<'a>(value: &'a Value, context: &str) -> Result<&'a [Value], Shrink2Error> {
    value.as_array().map(Vec::as_slice).ok_or_else(|| {
        Shrink2Error::new(
            REJECT_MALFORMED_SOURCE_AST,
            format!("{context} must be an array"),
        )
    })
}

fn source_bounded_uint(
    value: &Value,
    upper_exclusive: u64,
    context: &str,
) -> Result<u64, Shrink2Error> {
    let Some(integer) = exact_json_integer(value) else {
        return Err(Shrink2Error::new(
            REJECT_REGISTRY_INDEX_OUT_OF_RANGE,
            format!("{context} must be an exact JSON uint"),
        ));
    };
    if integer.sign < 0 {
        return Err(Shrink2Error::new(
            REJECT_REGISTRY_INDEX_OUT_OF_RANGE,
            format!("{context} is negative"),
        ));
    }
    let index = integer.digits.parse::<u64>().map_err(|_| {
        Shrink2Error::new(
            REJECT_REGISTRY_INDEX_OUT_OF_RANGE,
            format!("{context} exceeds the frozen registry width"),
        )
    })?;
    if index >= upper_exclusive {
        return Err(Shrink2Error::new(
            REJECT_REGISTRY_INDEX_OUT_OF_RANGE,
            format!("{context} is outside 0..{upper_exclusive}"),
        ));
    }
    Ok(index)
}

fn source_registry_index(
    value: &Value,
    names: &[&str],
    context: &str,
) -> Result<u64, Shrink2Error> {
    if value.is_number() {
        return source_bounded_uint(value, names.len() as u64, context);
    }
    value
        .as_str()
        .and_then(|name| names.iter().position(|candidate| *candidate == name))
        .map(|index| index as u64)
        .ok_or_else(|| {
            Shrink2Error::new(
                REJECT_REGISTRY_INDEX_OUT_OF_RANGE,
                format!("unknown frozen {context}"),
            )
        })
}

fn source_rational_index(
    parts: &[Value],
    grid: &[(i64, i64)],
    context: &str,
) -> Result<u64, Shrink2Error> {
    match parts {
        [index] => source_bounded_uint(index, grid.len() as u64, context),
        [numerator, denominator] => match rational_grid_boundary(numerator, denominator, grid) {
            RationalBoundary::NonInteger => Err(Shrink2Error::new(
                REJECT_MALFORMED_SOURCE_AST,
                format!("{context} rational pair must contain exact JSON integers"),
            )),
            RationalBoundary::Index(index) => Ok(index),
            RationalBoundary::OutOfRange => Err(Shrink2Error::new(
                REJECT_REGISTRY_INDEX_OUT_OF_RANGE,
                format!("{context} rational pair is outside its frozen grid"),
            )),
        },
        _ => Err(Shrink2Error::new(
            REJECT_MALFORMED_SOURCE_AST,
            format!("{context} requires an index or numerator/denominator pair"),
        )),
    }
}

#[derive(Debug)]
struct TypedSourceNode {
    node: Node,
    sort: Sort,
}

fn source_type_error(operator: &str, actual: &[Sort], expected: &[Sort]) -> Shrink2Error {
    let implicit_bit_coercion = matches!(
        operator,
        "add" | "difference" | "equal_exact" | "less_equal" | "greater_equal"
    ) && actual.contains(&Sort::Bit);
    if implicit_bit_coercion {
        Shrink2Error::new(
            REJECT_IMPLICIT_COERCION,
            format!("{operator} received Bit; explicit bit_to_scalar is required"),
        )
    } else {
        Shrink2Error::new(
            REJECT_TYPE_MISMATCH,
            format!("{operator} expects {expected:?}, received {actual:?}"),
        )
    }
}

fn require_source_sorts(
    operator: &str,
    actual: &[Sort],
    expected: &[Sort],
) -> Result<(), Shrink2Error> {
    if actual == expected {
        Ok(())
    } else {
        Err(source_type_error(operator, actual, expected))
    }
}

/// Parse the frozen source vocabulary left-to-right without transforming an
/// unvisited sibling. This preserves the Python failure-priority contract even
/// for arbitrary-width JSON integers.
fn parse_typed_shrink2_source(value: &Value) -> Result<TypedSourceNode, Shrink2Error> {
    let items = source_array(value, "source AST node")?;
    let Some(name) = items.first().and_then(Value::as_str) else {
        return Err(Shrink2Error::new(
            REJECT_MALFORMED_SOURCE_AST,
            "source AST node needs a text expression name",
        ));
    };
    let args = &items[1..];
    match name {
        "scalar_const" => Ok(TypedSourceNode {
            node: Node::ScalarConst(source_rational_index(
                args,
                &RATIONAL_PARAMETERS,
                "rational parameter",
            )?),
            sort: Sort::RationalValue,
        }),
        "bit_at" => {
            if args.len() != 1 {
                return Err(Shrink2Error::new(
                    REJECT_MALFORMED_SOURCE_AST,
                    "bit_at requires exactly one entity-slot argument",
                ));
            }
            Ok(TypedSourceNode {
                node: Node::BitAt(source_bounded_uint(&args[0], 8, "entity slot")?),
                sort: Sort::Bit,
            })
        }
        "set_size" => {
            if !args.is_empty() {
                return Err(Shrink2Error::new(
                    REJECT_MALFORMED_SOURCE_AST,
                    "set_size takes no arguments",
                ));
            }
            Ok(TypedSourceNode {
                node: Node::SetSize,
                sort: Sort::BoundedInt,
            })
        }
        "aggregate" => {
            if args.len() != 4 {
                return Err(Shrink2Error::new(
                    REJECT_MALFORMED_SOURCE_AST,
                    "aggregate requires map, scope, quantity, and extension",
                ));
            }
            if args[1].as_str() == Some(DEPRECATED_SCOPE_ALIAS) {
                return Err(Shrink2Error::new(
                    REJECT_NONCANONICAL_SCOPE_ALIAS,
                    "deprecated scope alias is migration-only",
                ));
            }
            let map_id = source_registry_index(&args[0], &AGGREGATE_MAP_NAMES, "aggregate map")?;
            let scope_id = source_registry_index(&args[1], &SCOPE_NAMES, "scope")?;
            let quantity_id = source_registry_index(&args[2], &QUANTITY_NAMES, "quantity")?;
            let raw_clauses = source_array(&args[3], "scope extension")?;
            if raw_clauses.len() > 2 {
                return Err(Shrink2Error::new(
                    REJECT_STRUCTURAL_LIMIT,
                    "scope extension exceeds two clauses",
                ));
            }
            let mut scope_extension = Vec::with_capacity(raw_clauses.len());
            for raw_clause in raw_clauses {
                let clause = source_array(raw_clause, "scope clause")?;
                if clause.len() != 2 || !clause[1].is_boolean() {
                    return Err(Shrink2Error::new(
                        REJECT_MALFORMED_SOURCE_AST,
                        "scope clause must be [context, bool]",
                    ));
                }
                scope_extension.push((
                    source_registry_index(&clause[0], &CONTEXT_NAMES, "context")?,
                    clause[1].as_bool().expect("checked boolean"),
                ));
            }
            let mut seen_contexts = BTreeSet::new();
            if scope_extension
                .iter()
                .any(|(context_id, _)| !seen_contexts.insert(*context_id))
            {
                return Err(Shrink2Error::new(
                    REJECT_DUPLICATE_SCOPE_CONTEXT,
                    "scope extension contains a duplicate context id",
                ));
            }
            scope_extension.sort_unstable();
            Ok(TypedSourceNode {
                node: Node::Aggregate {
                    map_id,
                    scope_id,
                    quantity_id,
                    scope_extension,
                },
                sort: if map_id == 1 {
                    Sort::BoundedInt
                } else {
                    Sort::RationalValue
                },
            })
        }
        "context_flag" | "task_flag" => {
            if args.len() != 1 {
                return Err(Shrink2Error::new(
                    REJECT_MALFORMED_SOURCE_AST,
                    format!("{name} requires exactly one registry argument"),
                ));
            }
            let (node, sort) = if name == "context_flag" {
                (
                    Node::ContextFlag(source_registry_index(
                        &args[0],
                        &CONTEXT_NAMES,
                        "context",
                    )?),
                    Sort::Bool,
                )
            } else {
                (
                    Node::TaskFlag(source_registry_index(&args[0], &TASK_NAMES, "task")?),
                    Sort::Bool,
                )
            };
            Ok(TypedSourceNode { node, sort })
        }
        "new_symbol_call" => Err(Shrink2Error::new(
            REJECT_NEW_SYMBOL_IN_OLD_DSL,
            "new symbols are Phase-3B only",
        )),
        "bit_to_scalar" | "int_to_scalar" | "absolute" | "sign" => {
            if args.len() != 1 {
                return Err(Shrink2Error::new(
                    REJECT_MALFORMED_SOURCE_AST,
                    format!("{name} requires exactly one child"),
                ));
            }
            let child = parse_typed_shrink2_source(&args[0])?;
            let (op, expected, output) = match name {
                "bit_to_scalar" => (UnaryOp::BitToScalar, Sort::Bit, Sort::RationalValue),
                "int_to_scalar" => (
                    UnaryOp::IntToScalar,
                    Sort::BoundedInt,
                    Sort::RationalValue,
                ),
                "absolute" => (
                    UnaryOp::Absolute,
                    Sort::RationalValue,
                    Sort::RationalValue,
                ),
                "sign" => (UnaryOp::Sign, Sort::RationalValue, Sort::Sign),
                _ => unreachable!(),
            };
            require_source_sorts(name, &[child.sort], &[expected])?;
            Ok(TypedSourceNode {
                node: Node::Unary {
                    op,
                    child: Box::new(child.node),
                },
                sort: output,
            })
        }
        "add" | "difference" | "equal_exact" | "less_equal" | "greater_equal"
        | "same_sign" | "opposite_sign" => {
            if args.len() != 2 {
                return Err(Shrink2Error::new(
                    REJECT_MALFORMED_SOURCE_AST,
                    format!("{name} requires exactly two children"),
                ));
            }
            let left = parse_typed_shrink2_source(&args[0])?;
            let right = parse_typed_shrink2_source(&args[1])?;
            let (op, expected, output) = match name {
                "add" => (
                    BinaryOp::Add,
                    Sort::RationalValue,
                    Sort::RationalValue,
                ),
                "difference" => (
                    BinaryOp::Difference,
                    Sort::RationalValue,
                    Sort::RationalValue,
                ),
                "equal_exact" => (BinaryOp::EqualExact, Sort::RationalValue, Sort::Bool),
                "less_equal" => (BinaryOp::LessEqual, Sort::RationalValue, Sort::Bool),
                "greater_equal" => (
                    BinaryOp::GreaterEqual,
                    Sort::RationalValue,
                    Sort::Bool,
                ),
                "same_sign" => (BinaryOp::SameSign, Sort::Sign, Sort::Bool),
                "opposite_sign" => (BinaryOp::OppositeSign, Sort::Sign, Sort::Bool),
                _ => unreachable!(),
            };
            require_source_sorts(name, &[left.sort, right.sort], &[expected, expected])?;
            Ok(TypedSourceNode {
                node: Node::Binary {
                    op,
                    left: Box::new(left.node),
                    right: Box::new(right.node),
                },
                sort: output,
            })
        }
        "approx_equal" => {
            if !matches!(args.len(), 3 | 4) {
                return Err(Shrink2Error::new(
                    REJECT_MALFORMED_SOURCE_AST,
                    "approx_equal requires two children plus a tolerance",
                ));
            }
            let left = parse_typed_shrink2_source(&args[0])?;
            let right = parse_typed_shrink2_source(&args[1])?;
            require_source_sorts(
                name,
                &[left.sort, right.sort],
                &[Sort::RationalValue, Sort::RationalValue],
            )?;
            let tolerance_index = if args.len() == 3 {
                if exact_json_integer(&args[2]).is_none() {
                    return Err(Shrink2Error::new(
                        REJECT_MALFORMED_SOURCE_AST,
                        "tolerance rational must use separate numerator/denominator fields",
                    ));
                }
                source_bounded_uint(&args[2], TOLERANCES.len() as u64, "tolerance")?
            } else {
                source_rational_index(&args[2..], &TOLERANCES, "tolerance")?
            };
            Ok(TypedSourceNode {
                node: Node::ApproxEqual {
                    left: Box::new(left.node),
                    right: Box::new(right.node),
                    tolerance_index,
                },
                sort: Sort::Bool,
            })
        }
        "top_level_AND" => {
            let raw_children: &[Value] = if args.len() == 1 {
                args[0]
                    .as_array()
                    .filter(|possible| {
                        !possible.is_empty()
                            && possible.iter().all(|child| {
                                child
                                    .as_array()
                                    .is_some_and(|child_items| !child_items.is_empty())
                            })
                    })
                    .map(Vec::as_slice)
                    .unwrap_or(args)
            } else {
                args
            };
            if raw_children.is_empty() {
                return Err(Shrink2Error::new(
                    REJECT_EMPTY_CONJUNCTION,
                    "AND0 has no canonical true node",
                ));
            }
            let mut nodes = Vec::with_capacity(raw_children.len());
            let mut sorts = Vec::with_capacity(raw_children.len());
            for raw_child in raw_children {
                let child = parse_typed_shrink2_source(raw_child)?;
                sorts.push(child.sort);
                nodes.push(child.node);
            }
            let expected = vec![Sort::Bool; sorts.len()];
            require_source_sorts(name, &sorts, &expected)?;
            Ok(TypedSourceNode {
                node: Node::And(nodes),
                sort: Sort::Bool,
            })
        }
        _ => Err(Shrink2Error::new(
            REJECT_UNKNOWN_EXPRESSION,
            format!("unknown old-DSL expression {name:?}"),
        )),
    }
}

fn reject_removed_aggregate_nodes(node: &Node) -> Result<(), Shrink2Error> {
    match node {
        Node::Aggregate { map_id: 2..=4, .. } => Err(Shrink2Error::new(
            REJECT_REMOVED_AGGREGATE_MAP,
            "AggregateMapId 2, 3, or 4 is tombstoned in hegel-old-dsl-v1.1.0 and descendants",
        )),
        Node::Unary { child, .. } => reject_removed_aggregate_nodes(child),
        Node::Binary { left, right, .. } | Node::ApproxEqual { left, right, .. } => {
            reject_removed_aggregate_nodes(left)?;
            reject_removed_aggregate_nodes(right)
        }
        Node::And(children) => children.iter().try_for_each(reject_removed_aggregate_nodes),
        _ => Ok(()),
    }
}

fn reject_removed_parameter_nodes(node: &Node) -> Result<(), Shrink2Error> {
    match node {
        Node::ScalarConst(index) if is_tombstoned_parameter(*index) => Err(Shrink2Error::new(
            REJECT_REMOVED_RATIONAL_PARAMETER,
            format!(
                "RationalParameterId {index} is tombstoned in hegel-old-dsl-v1.2.0"
            ),
        )),
        Node::Unary { child, .. } => reject_removed_parameter_nodes(child),
        Node::Binary { left, right, .. } | Node::ApproxEqual { left, right, .. } => {
            reject_removed_parameter_nodes(left)?;
            reject_removed_parameter_nodes(right)
        }
        Node::And(children) => children.iter().try_for_each(reject_removed_parameter_nodes),
        _ => Ok(()),
    }
}

/// Parse the v1 named-list source vocabulary under shrink-2 sparse admission.
pub fn parse_shrink2_source_ast(value: &Value) -> Result<Node, Shrink2Error> {
    let node = parse_typed_shrink2_source(value)?.node;
    canonicalize_parent_source_node(node.clone())?;
    reject_removed_aggregate_nodes(&node)?;
    reject_removed_parameter_nodes(&node)?;
    Ok(node)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Rational {
    numerator: i64,
    denominator: i64,
}

impl Rational {
    fn new(numerator: i64, denominator: i64) -> Option<Self> {
        if denominator == 0 {
            return None;
        }
        let mut numerator = numerator;
        let mut denominator = denominator;
        if denominator < 0 {
            numerator = numerator.checked_neg()?;
            denominator = denominator.checked_neg()?;
        }
        let divisor = gcd_i64(numerator, denominator);
        Some(Self {
            numerator: numerator / divisor,
            denominator: denominator / divisor,
        })
    }

    fn add(&self, other: &Self) -> Option<Self> {
        let left = self.numerator.checked_mul(other.denominator)?;
        let right = other.numerator.checked_mul(self.denominator)?;
        Self::new(
            left.checked_add(right)?,
            self.denominator.checked_mul(other.denominator)?,
        )
    }

    fn difference(&self, other: &Self) -> Option<Self> {
        let left = self.numerator.checked_mul(other.denominator)?;
        let right = other.numerator.checked_mul(self.denominator)?;
        Self::new(
            left.checked_sub(right)?,
            self.denominator.checked_mul(other.denominator)?,
        )
    }

    fn absolute(&self) -> Option<Self> {
        Self::new(self.numerator.checked_abs()?, self.denominator)
    }
}

fn gcd_i64(left: i64, right: i64) -> i64 {
    let mut left = left.unsigned_abs();
    let mut right = right.unsigned_abs();
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    if left == 0 { 1 } else { left as i64 }
}

fn rational_parameter(index: u64) -> Option<Rational> {
    let (numerator, denominator) = *RATIONAL_PARAMETERS.get(index as usize)?;
    Rational::new(numerator, denominator)
}

fn active_rational_parameter_index(value: &Rational) -> Option<u64> {
    RATIONAL_PARAMETERS
        .iter()
        .position(|pair| *pair == (value.numerator, value.denominator))
        .map(|index| index as u64)
        .filter(|index| is_active_parameter(*index))
}

fn unary_op_id(op: UnaryOp) -> u64 {
    match op {
        UnaryOp::BitToScalar => 0,
        UnaryOp::IntToScalar => 1,
        UnaryOp::Absolute => 2,
        UnaryOp::Sign => 3,
    }
}

fn binary_op_id(op: BinaryOp) -> u64 {
    match op {
        BinaryOp::Add => 0,
        BinaryOp::Difference => 1,
        BinaryOp::EqualExact => 2,
        BinaryOp::LessEqual => 3,
        BinaryOp::GreaterEqual => 4,
        BinaryOp::SameSign => 5,
        BinaryOp::OppositeSign => 6,
    }
}

fn binary_op_is_commutative(op: BinaryOp) -> bool {
    matches!(
        op,
        BinaryOp::Add | BinaryOp::EqualExact | BinaryOp::SameSign | BinaryOp::OppositeSign
    )
}

fn node_formal_json(node: &Node) -> Value {
    match node {
        Node::ScalarConst(index) => json!([0, 0, index]),
        Node::BitAt(index) => json!([0, 1, index]),
        Node::SetSize => json!([0, 2]),
        Node::Aggregate {
            map_id,
            scope_id,
            quantity_id,
            scope_extension,
        } => json!([
            0,
            3,
            map_id,
            scope_id,
            quantity_id,
            scope_extension
                .iter()
                .map(|(context_id, expected)| json!([context_id, expected]))
                .collect::<Vec<_>>()
        ]),
        Node::ContextFlag(index) => json!([0, 4, index]),
        Node::TaskFlag(index) => json!([0, 5, index]),
        Node::NewSymbolCall(index) => json!([0, 6, index]),
        Node::Unary { op, child } => json!([1, unary_op_id(*op), node_formal_json(child)]),
        Node::Binary { op, left, right } => json!([
            2,
            binary_op_id(*op),
            node_formal_json(left),
            node_formal_json(right)
        ]),
        Node::ApproxEqual {
            left,
            right,
            tolerance_index,
        } => json!([
            3,
            0,
            node_formal_json(left),
            node_formal_json(right),
            tolerance_index
        ]),
        Node::And(atoms) => json!([
            4,
            atoms.iter().map(node_formal_json).collect::<Vec<_>>()
        ]),
    }
}

fn node_cbor(node: &Node) -> Vec<u8> {
    encode_strict_cbor_json(&node_formal_json(node))
        .expect("numeric shrink-2 AST node is always encodable under deterministic CBOR")
}

fn encode_ast_envelope(node: &Node) -> Vec<u8> {
    encode_strict_cbor_json(&json!([1, node_formal_json(node)]))
        .expect("numeric shrink-2 AST envelope is always encodable under deterministic CBOR")
}

fn canonical_child_key(node: &Node) -> ([u8; 32], Vec<u8>) {
    let bytes = node_cbor(node);
    let digest: [u8; 32] = Sha256::digest(&bytes).into();
    (digest, bytes)
}

fn order_commutative_pair(left: Node, right: Node) -> (Node, Node) {
    if canonical_child_key(&left) <= canonical_child_key(&right) {
        (left, right)
    } else {
        (right, left)
    }
}

fn scalar_const_value(node: &Node) -> Option<Rational> {
    match node {
        Node::ScalarConst(index) if is_active_parameter(*index) => rational_parameter(*index),
        _ => None,
    }
}

fn is_zero_const(node: &Node) -> bool {
    matches!(node, Node::ScalarConst(ZERO_PARAMETER_INDEX))
}

fn collect_add_operands(node: Node, output: &mut Vec<Node>) {
    match node {
        Node::Binary {
            op: BinaryOp::Add,
            left,
            right,
        } => {
            collect_add_operands(*left, output);
            collect_add_operands(*right, output);
        }
        other => output.push(other),
    }
}

fn build_right_associated_add(mut operands: Vec<Node>) -> Node {
    debug_assert!(operands.len() >= 2);
    let mut result = operands.pop().expect("at least two add operands");
    while let Some(left) = operands.pop() {
        result = Node::Binary {
            op: BinaryOp::Add,
            left: Box::new(left),
            right: Box::new(result),
        };
    }
    result
}

/// Apply exactly the parent rewrite order except that a constant result is
/// materialized only if RationalParameterId/v1 marks that result ACTIVE.
fn normalize_once(node: Node) -> Result<Node, Shrink2Error> {
    match node {
        Node::ScalarConst(_)
        | Node::BitAt(_)
        | Node::SetSize
        | Node::ContextFlag(_)
        | Node::TaskFlag(_)
        | Node::NewSymbolCall(_) => Ok(node),
        Node::Aggregate {
            map_id,
            scope_id,
            quantity_id,
            mut scope_extension,
        } => {
            scope_extension.sort_unstable_by_key(|clause| clause.0);
            if scope_extension
                .windows(2)
                .any(|pair| pair[0].0 == pair[1].0)
            {
                return Err(Shrink2Error::new(
                    REJECT_DUPLICATE_SCOPE_CONTEXT,
                    "scope extension contains a duplicate context id",
                ));
            }
            Ok(Node::Aggregate {
                map_id,
                scope_id,
                quantity_id,
                scope_extension,
            })
        }
        Node::Unary { op, child } => {
            let child = normalize_once(*child)?;
            match op {
                UnaryOp::Absolute => {
                    if let Node::Unary {
                        op: UnaryOp::Absolute,
                        child: grandchild,
                    } = child
                    {
                        return Ok(Node::Unary {
                            op: UnaryOp::Absolute,
                            child: grandchild,
                        });
                    }
                    if let Some(index) = scalar_const_value(&child)
                        .and_then(|value| value.absolute())
                        .as_ref()
                        .and_then(active_rational_parameter_index)
                    {
                        return Ok(Node::ScalarConst(index));
                    }
                    Ok(Node::Unary {
                        op,
                        child: Box::new(child),
                    })
                }
                _ => Ok(Node::Unary {
                    op,
                    child: Box::new(child),
                }),
            }
        }
        Node::Binary { op, left, right } => {
            let left = normalize_once(*left)?;
            let right = normalize_once(*right)?;
            match op {
                BinaryOp::GreaterEqual => Ok(Node::Binary {
                    op: BinaryOp::LessEqual,
                    left: Box::new(right),
                    right: Box::new(left),
                }),
                BinaryOp::Difference => {
                    if is_zero_const(&right) {
                        return Ok(left);
                    }
                    if left == right {
                        return Ok(Node::ScalarConst(ZERO_PARAMETER_INDEX));
                    }
                    if let (Some(left_value), Some(right_value)) =
                        (scalar_const_value(&left), scalar_const_value(&right))
                    {
                        if let Some(index) = left_value
                            .difference(&right_value)
                            .as_ref()
                            .and_then(active_rational_parameter_index)
                        {
                            return Ok(Node::ScalarConst(index));
                        }
                    }
                    Ok(Node::Binary {
                        op,
                        left: Box::new(left),
                        right: Box::new(right),
                    })
                }
                BinaryOp::Add => {
                    if let (Some(left_value), Some(right_value)) =
                        (scalar_const_value(&left), scalar_const_value(&right))
                    {
                        if let Some(index) = left_value
                            .add(&right_value)
                            .as_ref()
                            .and_then(active_rational_parameter_index)
                        {
                            return Ok(Node::ScalarConst(index));
                        }
                    }
                    let mut operands = Vec::new();
                    collect_add_operands(left, &mut operands);
                    collect_add_operands(right, &mut operands);
                    operands.retain(|operand| !is_zero_const(operand));
                    if operands.is_empty() {
                        return Ok(Node::ScalarConst(ZERO_PARAMETER_INDEX));
                    }
                    if operands.len() == 1 {
                        return Ok(operands.pop().expect("one add operand"));
                    }
                    operands.sort_by_key(canonical_child_key);
                    Ok(build_right_associated_add(operands))
                }
                _ if binary_op_is_commutative(op) => {
                    let (left, right) = order_commutative_pair(left, right);
                    Ok(Node::Binary {
                        op,
                        left: Box::new(left),
                        right: Box::new(right),
                    })
                }
                _ => Ok(Node::Binary {
                    op,
                    left: Box::new(left),
                    right: Box::new(right),
                }),
            }
        }
        Node::ApproxEqual {
            left,
            right,
            tolerance_index,
        } => {
            let left = normalize_once(*left)?;
            let right = normalize_once(*right)?;
            let (left, right) = order_commutative_pair(left, right);
            if tolerance_index == 0 {
                Ok(Node::Binary {
                    op: BinaryOp::EqualExact,
                    left: Box::new(left),
                    right: Box::new(right),
                })
            } else {
                Ok(Node::ApproxEqual {
                    left: Box::new(left),
                    right: Box::new(right),
                    tolerance_index,
                })
            }
        }
        Node::And(atoms) => {
            let mut flattened = Vec::new();
            for atom in atoms {
                match normalize_once(atom)? {
                    Node::And(nested) => flattened.extend(nested),
                    normalized => flattened.push(normalized),
                }
            }
            flattened.sort_by_key(node_cbor);
            flattened.dedup_by(|left, right| node_cbor(left) == node_cbor(right));
            if flattened.is_empty() {
                return Err(Shrink2Error::new(
                    REJECT_MALFORMED_SOURCE_AST,
                    "AND normalization cannot produce an empty conjunction",
                ));
            }
            if flattened.len() == 1 {
                return Ok(flattened.pop().expect("one AND atom"));
            }
            if flattened.len() > MAX_TOP_LEVEL_CLAUSES {
                return Err(Shrink2Error::new(
                    REJECT_STRUCTURAL_LIMIT,
                    format!(
                        "flattened AND has {} clauses; maximum is {MAX_TOP_LEVEL_CLAUSES}",
                        flattened.len()
                    ),
                ));
            }
            Ok(Node::And(flattened))
        }
    }
}

fn normalize_to_fixed_point(mut node: Node) -> Result<Node, Shrink2Error> {
    for _ in 0..64 {
        let next = normalize_once(node.clone())?;
        if next == node {
            return Ok(node);
        }
        node = next;
    }
    Err(Shrink2Error::new(
        REJECT_INTERNAL_CANONICALIZATION,
        "shrink-2 rewrite system did not reach a fixed point within 64 passes",
    ))
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AstStats {
    node_count: u32,
    depth: u32,
    bit_slots: BTreeSet<u64>,
    aggregate_leaves: u32,
    scalar_parameter_occurrences: u32,
}

fn merge_stats(root_nodes: u32, children: &[AstStats]) -> AstStats {
    let mut bit_slots = BTreeSet::new();
    let mut node_count = root_nodes;
    let mut aggregate_leaves = 0;
    let mut scalar_parameter_occurrences = 0;
    let mut maximum_child_depth = 0;
    for child in children {
        node_count += child.node_count;
        aggregate_leaves += child.aggregate_leaves;
        scalar_parameter_occurrences += child.scalar_parameter_occurrences;
        maximum_child_depth = maximum_child_depth.max(child.depth);
        bit_slots.extend(child.bit_slots.iter().copied());
    }
    AstStats {
        node_count,
        depth: if children.is_empty() {
            0
        } else {
            1 + maximum_child_depth
        },
        bit_slots,
        aggregate_leaves,
        scalar_parameter_occurrences,
    }
}

fn ast_stats(node: &Node) -> AstStats {
    match node {
        Node::ScalarConst(_) => {
            let mut stats = merge_stats(1, &[]);
            stats.scalar_parameter_occurrences = 1;
            stats
        }
        Node::SetSize | Node::ContextFlag(_) | Node::TaskFlag(_) | Node::NewSymbolCall(_) => {
            merge_stats(1, &[])
        }
        Node::BitAt(index) => {
            let mut stats = merge_stats(1, &[]);
            stats.bit_slots.insert(*index);
            stats
        }
        Node::Aggregate { .. } => {
            let mut stats = merge_stats(1, &[]);
            stats.aggregate_leaves = 1;
            stats
        }
        Node::Unary { child, .. } => merge_stats(1, &[ast_stats(child)]),
        Node::Binary { left, right, .. } | Node::ApproxEqual { left, right, .. } => {
            merge_stats(1, &[ast_stats(left), ast_stats(right)])
        }
        Node::And(atoms) => {
            let children = atoms.iter().map(ast_stats).collect::<Vec<_>>();
            merge_stats(1, &children)
        }
    }
}

fn validate_structural_limits(node: &Node) -> Result<AstStats, Shrink2Error> {
    if let Node::And(atoms) = node {
        if !(2..=MAX_TOP_LEVEL_CLAUSES).contains(&atoms.len()) {
            return Err(Shrink2Error::new(
                REJECT_STRUCTURAL_LIMIT,
                "canonical AND must contain exactly two or three atoms",
            ));
        }
    }
    let stats = ast_stats(node);
    if stats.node_count > MAX_TOTAL_NODE_COUNT {
        return Err(Shrink2Error::new(
            REJECT_STRUCTURAL_LIMIT,
            format!(
                "AST has {} nodes; maximum is {MAX_TOTAL_NODE_COUNT}",
                stats.node_count
            ),
        ));
    }
    if stats.depth > MAX_TOTAL_AST_DEPTH {
        return Err(Shrink2Error::new(
            REJECT_STRUCTURAL_LIMIT,
            format!(
                "AST depth is {}; maximum is {MAX_TOTAL_AST_DEPTH}",
                stats.depth
            ),
        ));
    }
    if stats.bit_slots.len() > MAX_DISTINCT_BIT_SLOTS {
        return Err(Shrink2Error::new(
            REJECT_STRUCTURAL_LIMIT,
            format!(
                "AST uses {} distinct bit slots; maximum is {MAX_DISTINCT_BIT_SLOTS}",
                stats.bit_slots.len()
            ),
        ));
    }
    if stats.aggregate_leaves > MAX_AGGREGATE_LEAVES {
        return Err(Shrink2Error::new(
            REJECT_STRUCTURAL_LIMIT,
            format!(
                "AST uses {} aggregate leaves; maximum is {MAX_AGGREGATE_LEAVES}",
                stats.aggregate_leaves
            ),
        ));
    }
    if stats.scalar_parameter_occurrences > MAX_FITTED_SCALAR_PARAMETER_OCCURRENCES {
        return Err(Shrink2Error::new(
            REJECT_STRUCTURAL_LIMIT,
            format!(
                "AST uses {} fitted scalar-parameter occurrences; maximum is {MAX_FITTED_SCALAR_PARAMETER_OCCURRENCES}",
                stats.scalar_parameter_occurrences
            ),
        ));
    }
    Ok(stats)
}

fn root_operator_id(node: &Node) -> u16 {
    match node {
        Node::ScalarConst(_) => 0x0000,
        Node::BitAt(_) => 0x0001,
        Node::SetSize => 0x0002,
        Node::Aggregate { .. } => 0x0003,
        Node::ContextFlag(_) => 0x0004,
        Node::TaskFlag(_) => 0x0005,
        Node::NewSymbolCall(_) => 0x0006,
        Node::Unary { op, .. } => 0x0100 + unary_op_id(*op) as u16,
        Node::Binary { op, .. } => 0x0200 + binary_op_id(*op) as u16,
        Node::ApproxEqual { .. } => 0x0300,
        Node::And(_) => 0x0400,
    }
}

fn finish_program(canonical_node: Node) -> Result<CanonicalProgram, Shrink2Error> {
    reject_removed_aggregate_nodes(&canonical_node)?;
    reject_removed_parameter_nodes(&canonical_node)?;
    let output_sort = type_check(&canonical_node)?;
    let stats = validate_structural_limits(&canonical_node)?;
    let canonical_cbor = encode_ast_envelope(&canonical_node);
    let mut hasher = Sha256::new();
    hasher.update(AST_HASH_DOMAIN.as_bytes());
    hasher.update([0]);
    hasher.update(&canonical_cbor);
    let canonical_ast_hash: [u8; 32] = hasher.finalize().into();
    Ok(CanonicalProgram {
        root_operator_id: root_operator_id(&canonical_node),
        node_count: stats.node_count,
        depth: stats.depth,
        distinct_bit_slot_count: stats.bit_slots.len(),
        aggregate_leaf_count: stats.aggregate_leaves,
        scalar_parameter_occurrence_count: stats.scalar_parameter_occurrences,
        canonical_node,
        canonical_cbor,
        canonical_ast_hash,
        output_sort,
    })
}

/// Canonicalize one already parsed source node under sparse shrink-2 admission.
pub fn canonicalize_shrink2_source_node(source: Node) -> Result<CanonicalProgram, Shrink2Error> {
    canonicalize_parent_source_node(source.clone())?;
    reject_removed_aggregate_nodes(&source)?;
    reject_removed_parameter_nodes(&source)?;
    // As in the parent, type checking intentionally precedes every rewrite.
    type_check(&source)?;
    let canonical = normalize_to_fixed_point(source)?;
    finish_program(canonical)
}

/// Parse and canonicalize one source JSON AST under sparse shrink-2 admission.
pub fn canonicalize_shrink2_source_json(value: &Value) -> Result<CanonicalProgram, Shrink2Error> {
    canonicalize_shrink2_source_node(parse_shrink2_source_ast(value)?)
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ProbeValue {
    Unsigned(u64),
    Negative,
    Bytes,
    Array(Vec<ProbeValue>),
    Bool(bool),
    Null,
}

fn read_argument(additional: u8, bytes: &[u8], cursor: &mut usize) -> Result<u64, Shrink2Error> {
    let (width, minimum) = match additional {
        0..=23 => return Ok(u64::from(additional)),
        24 => (1, 24),
        25 => (2, 0x100),
        26 => (4, 0x1_0000),
        27 => (8, 0x1_0000_0000),
        31 => {
            return Err(Shrink2Error::new(
                REJECT_INDEFINITE_CBOR,
                "indefinite-length CBOR is forbidden",
            ))
        }
        _ => {
            return Err(Shrink2Error::new(
                REJECT_RESERVED_CBOR,
                "reserved CBOR additional-information value",
            ))
        }
    };
    let end = cursor
        .checked_add(width)
        .ok_or_else(|| Shrink2Error::new(REJECT_NONCANONICAL_CBOR, "CBOR cursor overflow"))?;
    let payload = bytes.get(*cursor..end).ok_or_else(|| {
        Shrink2Error::new(REJECT_TRUNCATED_CBOR, "truncated CBOR argument")
    })?;
    *cursor = end;
    let mut value = 0u64;
    for byte in payload {
        value = (value << 8) | u64::from(*byte);
    }
    if value < minimum {
        return Err(Shrink2Error::new(
            REJECT_NONCANONICAL_CBOR,
            "CBOR argument does not use its shortest encoding",
        ));
    }
    Ok(value)
}

fn parse_probe(
    bytes: &[u8],
    cursor: &mut usize,
    depth: usize,
) -> Result<ProbeValue, Shrink2Error> {
    if depth > MAX_CBOR_NESTING {
        return Err(Shrink2Error::new(
            REJECT_CBOR_NESTING,
            "CBOR exceeded the strict nesting limit",
        ));
    }
    let initial = *bytes.get(*cursor).ok_or_else(|| {
        Shrink2Error::new(REJECT_TRUNCATED_CBOR, "expected a CBOR item")
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
            let length = usize::try_from(read_argument(additional, bytes, cursor)?).map_err(|_| {
                Shrink2Error::new(REJECT_NONCANONICAL_CBOR, "byte length overflow")
            })?;
            let end = cursor.checked_add(length).ok_or_else(|| {
                Shrink2Error::new(REJECT_NONCANONICAL_CBOR, "byte cursor overflow")
            })?;
            if bytes.get(*cursor..end).is_none() {
                return Err(Shrink2Error::new(
                    REJECT_TRUNCATED_CBOR,
                    "truncated CBOR byte string",
                ));
            }
            *cursor = end;
            Ok(ProbeValue::Bytes)
        }
        3 => Err(Shrink2Error::new(
            REJECT_CBOR_TEXT,
            "CBOR text strings are forbidden",
        )),
        4 => {
            let length = usize::try_from(read_argument(additional, bytes, cursor)?).map_err(|_| {
                Shrink2Error::new(REJECT_NONCANONICAL_CBOR, "array length overflow")
            })?;
            let mut values = Vec::with_capacity(length.min(bytes.len().saturating_sub(*cursor)));
            for _ in 0..length {
                values.push(parse_probe(bytes, cursor, depth + 1)?);
            }
            Ok(ProbeValue::Array(values))
        }
        5 => Err(Shrink2Error::new(
            REJECT_CBOR_MAP,
            "CBOR maps are forbidden",
        )),
        6 => Err(Shrink2Error::new(
            REJECT_CBOR_TAG,
            "CBOR tags are forbidden",
        )),
        7 if additional == 20 => Ok(ProbeValue::Bool(false)),
        7 if additional == 21 => Ok(ProbeValue::Bool(true)),
        7 if additional == 22 => Ok(ProbeValue::Null),
        7 if additional == 23 => Err(Shrink2Error::new(
            REJECT_CBOR_UNDEFINED,
            "CBOR undefined is forbidden",
        )),
        7 if matches!(additional, 25..=27) => Err(Shrink2Error::new(
            REJECT_CBOR_FLOAT,
            "CBOR floating point is forbidden",
        )),
        7 if additional == 31 => Err(Shrink2Error::new(
            REJECT_INDEFINITE_CBOR,
            "CBOR break is forbidden",
        )),
        7 => Err(Shrink2Error::new(
            REJECT_CBOR_SIMPLE,
            "unapproved CBOR simple value",
        )),
        _ => unreachable!("CBOR major type is three bits"),
    }
}

fn probe_array<'a>(value: &'a ProbeValue, context: &str) -> Result<&'a [ProbeValue], Shrink2Error> {
    match value {
        ProbeValue::Array(values) => Ok(values),
        _ => Err(Shrink2Error::new(
            REJECT_NONCANONICAL_AST,
            format!("{context} must be an array"),
        )),
    }
}

fn probe_registry_uint(value: &ProbeValue, context: &str) -> Result<u64, Shrink2Error> {
    match value {
        ProbeValue::Unsigned(value) => Ok(*value),
        _ => Err(Shrink2Error::new(
            REJECT_REGISTRY_INDEX_OUT_OF_RANGE,
            format!("{context} is outside its unsigned registry domain"),
        )),
    }
}

fn probe_bounded_registry_uint(
    value: &ProbeValue,
    upper_exclusive: u64,
    context: &str,
) -> Result<u64, Shrink2Error> {
    let index = probe_registry_uint(value, context)?;
    if index >= upper_exclusive {
        return Err(Shrink2Error::new(
            REJECT_REGISTRY_INDEX_OUT_OF_RANGE,
            format!("{context} is outside 0..{upper_exclusive}"),
        ));
    }
    Ok(index)
}

fn probe_expression_tag(value: &ProbeValue, context: &str) -> Result<u64, Shrink2Error> {
    match value {
        ProbeValue::Unsigned(value) => Ok(*value),
        ProbeValue::Negative => Err(Shrink2Error::new(
            REJECT_UNKNOWN_EXPRESSION,
            format!("{context} is an unknown negative expression ID"),
        )),
        _ => Err(Shrink2Error::new(
            REJECT_NONCANONICAL_AST,
            format!("{context} must be an integer"),
        )),
    }
}

fn probe_bool(value: &ProbeValue, context: &str) -> Result<bool, Shrink2Error> {
    match value {
        ProbeValue::Bool(value) => Ok(*value),
        _ => Err(Shrink2Error::new(
            REJECT_NONCANONICAL_AST,
            format!("{context} must be a boolean"),
        )),
    }
}

fn expect_probe_length(
    values: &[ProbeValue],
    expected: usize,
    context: &str,
) -> Result<(), Shrink2Error> {
    if values.len() != expected {
        return Err(Shrink2Error::new(
            REJECT_NONCANONICAL_AST,
            format!(
                "{context} has array length {}; expected {expected}",
                values.len()
            ),
        ));
    }
    Ok(())
}

fn formal_type_checked(node: Node) -> Result<Node, Shrink2Error> {
    if let Err(error) = type_check(&node) {
        if error.code == REJECT_IMPLICIT_COERCION {
            return Err(Shrink2Error::new(
                REJECT_TYPE_MISMATCH,
                "formal canonical AST has a child-sort mismatch",
            ));
        }
        return Err(error.into());
    }
    Ok(node)
}

fn node_from_probe(value: &ProbeValue) -> Result<Node, Shrink2Error> {
    let values = probe_array(value, "AST node")?;
    if values.is_empty() {
        return Err(Shrink2Error::new(
            REJECT_NONCANONICAL_AST,
            "AST node array must not be empty",
        ));
    }
    match probe_expression_tag(&values[0], "AST node tag")? {
        0 => {
            if values.len() < 2 {
                return Err(Shrink2Error::new(
                    REJECT_NONCANONICAL_AST,
                    "leaf node is missing its leaf id",
                ));
            }
            match probe_expression_tag(&values[1], "leaf id")? {
                0 => {
                    expect_probe_length(values, 3, "scalar_const")?;
                    Ok(Node::ScalarConst(probe_bounded_registry_uint(
                        &values[2],
                        RATIONAL_PARAMETERS.len() as u64,
                        "rational parameter index",
                    )?))
                }
                1 => {
                    expect_probe_length(values, 3, "bit_at")?;
                    Ok(Node::BitAt(probe_bounded_registry_uint(
                        &values[2],
                        8,
                        "entity slot index",
                    )?))
                }
                2 => {
                    expect_probe_length(values, 2, "set_size")?;
                    Ok(Node::SetSize)
                }
                3 => {
                    expect_probe_length(values, 6, "aggregate")?;
                    let map_id = probe_bounded_registry_uint(
                        &values[2],
                        AGGREGATE_MAP_NAMES.len() as u64,
                        "aggregate map id",
                    )?;
                    let scope_id = probe_bounded_registry_uint(
                        &values[3],
                        SCOPE_NAMES.len() as u64,
                        "scope id",
                    )?;
                    let quantity_id = probe_bounded_registry_uint(
                        &values[4],
                        QUANTITY_NAMES.len() as u64,
                        "quantity id",
                    )?;
                    let clauses = probe_array(&values[5], "scope extension")?;
                    let mut scope_extension = Vec::with_capacity(clauses.len());
                    for clause in clauses {
                        let clause = probe_array(clause, "scope clause")?;
                        expect_probe_length(clause, 2, "scope clause")?;
                        scope_extension.push((
                            probe_bounded_registry_uint(
                                &clause[0],
                                CONTEXT_NAMES.len() as u64,
                                "scope context id",
                            )?,
                            probe_bool(&clause[1], "scope expected bool")?,
                        ));
                    }
                    let mut ordered_scope_extension = scope_extension.clone();
                    ordered_scope_extension.sort_unstable();
                    if scope_extension.len() > 2
                        || scope_extension != ordered_scope_extension
                        || scope_extension
                            .windows(2)
                            .any(|pair| pair[0].0 == pair[1].0)
                    {
                        return Err(Shrink2Error::new(
                            REJECT_NONCANONICAL_AST,
                            "scope clauses must be unique, sorted, and contain at most two rows",
                        ));
                    }
                    Ok(Node::Aggregate {
                        map_id,
                        scope_id,
                        quantity_id,
                        scope_extension,
                    })
                }
                4 => {
                    expect_probe_length(values, 3, "context_flag")?;
                    Ok(Node::ContextFlag(probe_bounded_registry_uint(
                        &values[2],
                        CONTEXT_NAMES.len() as u64,
                        "context id",
                    )?))
                }
                5 => {
                    expect_probe_length(values, 3, "task_flag")?;
                    Ok(Node::TaskFlag(probe_bounded_registry_uint(
                        &values[2],
                        TASK_NAMES.len() as u64,
                        "task id",
                    )?))
                }
                6 => Err(Shrink2Error::new(
                    REJECT_NEW_SYMBOL_IN_OLD_DSL,
                    "new symbols are Phase-3B only",
                )),
                leaf_id => Err(Shrink2Error::new(
                    REJECT_UNKNOWN_EXPRESSION,
                    format!("unknown/reserved leaf id {leaf_id}"),
                )),
            }
        }
        1 => {
            expect_probe_length(values, 3, "unary node")?;
            let op = match probe_bounded_registry_uint(&values[1], 4, "unary operator id")? {
                0 => UnaryOp::BitToScalar,
                1 => UnaryOp::IntToScalar,
                2 => UnaryOp::Absolute,
                3 => UnaryOp::Sign,
                _ => unreachable!("bounded unary operator ID"),
            };
            formal_type_checked(Node::Unary {
                op,
                child: Box::new(node_from_probe(&values[2])?),
            })
        }
        2 => {
            expect_probe_length(values, 4, "binary node")?;
            let op = match probe_bounded_registry_uint(&values[1], 8, "binary operator id")? {
                0 => BinaryOp::Add,
                1 => BinaryOp::Difference,
                2 => BinaryOp::EqualExact,
                3 => BinaryOp::LessEqual,
                5 => BinaryOp::SameSign,
                6 => BinaryOp::OppositeSign,
                4 | 7 => {
                    return Err(Shrink2Error::new(
                        REJECT_NONCANONICAL_AST,
                        "source-only or reserved binary operator ID is not canonical",
                    ))
                }
                _ => unreachable!("bounded binary operator ID"),
            };
            let left = node_from_probe(&values[2])?;
            let right = node_from_probe(&values[3])?;
            formal_type_checked(Node::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
            })
        }
        3 => {
            expect_probe_length(values, 5, "ternary node")?;
            probe_bounded_registry_uint(&values[1], 1, "ternary operator id")?;
            let left = node_from_probe(&values[2])?;
            let right = node_from_probe(&values[3])?;
            if type_check(&left)? != Sort::RationalValue
                || type_check(&right)? != Sort::RationalValue
            {
                return Err(Shrink2Error::new(
                    REJECT_TYPE_MISMATCH,
                    "approx_equal expects two RationalValue children",
                ));
            }
            let tolerance_index = probe_bounded_registry_uint(
                &values[4],
                TOLERANCES.len() as u64,
                "tolerance index",
            )?;
            Ok(Node::ApproxEqual {
                left: Box::new(left),
                right: Box::new(right),
                tolerance_index,
            })
        }
        4 => {
            expect_probe_length(values, 2, "conjunction node")?;
            let raw_atoms = probe_array(&values[1], "conjunction atom list")?;
            if !matches!(raw_atoms.len(), 2 | 3) {
                return Err(Shrink2Error::new(
                    REJECT_NONCANONICAL_AST,
                    "canonical AND must contain exactly two or three atoms",
                ));
            }
            let atoms = raw_atoms
                .iter()
                .map(node_from_probe)
                .collect::<Result<Vec<_>, _>>()?;
            formal_type_checked(Node::And(atoms))
        }
        tag => Err(Shrink2Error::new(
            REJECT_UNKNOWN_EXPRESSION,
            format!("unknown/reserved AST node tag {tag}"),
        )),
    }
}

/// Validate and decode exact child-canonical AST CBOR. Generic deterministic
/// CBOR validation remains delegated to the independently verified parent.
pub fn decode_shrink2_canonical_ast(bytes: &[u8]) -> Result<CanonicalProgram, Shrink2Error> {
    let mut cursor = 0usize;
    let probe = parse_probe(bytes, &mut cursor, 0)?;
    if cursor != bytes.len() {
        return Err(Shrink2Error::new(
            REJECT_TRAILING_CBOR,
            "trailing bytes after the first CBOR item",
        ));
    }
    validate_strict_cbor(bytes)?;
    let envelope = probe_array(&probe, "CanonicalAstV1 envelope")?;
    expect_probe_length(envelope, 2, "CanonicalAstV1 envelope")?;
    if envelope[0] != ProbeValue::Unsigned(1) {
        return Err(Shrink2Error::new(
            REJECT_UNKNOWN_AST_SCHEMA,
            "unknown canonical AST schema version",
        ));
    }
    let source_node = node_from_probe(&envelope[1])?;
    reject_removed_aggregate_nodes(&source_node)?;
    reject_removed_parameter_nodes(&source_node)?;
    let normalized = normalize_to_fixed_point(source_node)?;
    let reencoded = encode_ast_envelope(&normalized);
    if reencoded != bytes {
        return Err(Shrink2Error::new(
            REJECT_NONCANONICAL_AST,
            "AST bytes are schema-readable but not in shrink-2 canonical normal form",
        ));
    }
    finish_program(normalized)
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
pub struct Shrink2CapacityReplayReport {
    pub schema_version: &'static str,
    pub implementation: &'static str,
    pub parent_dsl_version: &'static str,
    pub parent_freeze_version: &'static str,
    pub dsl_version: &'static str,
    pub freeze_version: &'static str,
    pub human_amendment_id: &'static str,
    pub shrink_step_id: &'static str,
    pub generator_rule: &'static str,
    pub active_rational_parameter_ids: [u64; 3],
    pub rational_aggregate_map_ids: [u64; 2],
    pub constant_atom_count: usize,
    pub rational_aggregate_count: usize,
    pub mixed_atom_count: usize,
    pub source_candidate_count: usize,
    pub accepted_source_count: usize,
    pub accepted_unique_count: usize,
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
}

/// Independently construct and replay the frozen shrink-2 2,160-source
/// capacity subset. Success is deliberately not a complete-closure claim.
pub fn replay_shrink2_capacity_subset() -> Result<Shrink2CapacityReplayReport, Shrink2Error> {
    let constant_atoms = capacity_constant_atoms();
    let rational_aggregates = capacity_rational_aggregates();
    let mixed_atoms = capacity_mixed_atoms();
    if constant_atoms.len() != 15
        || rational_aggregates.len() != 16
        || mixed_atoms.len() != 144
    {
        return Err(Shrink2Error::new(
            REJECT_INTERNAL_SHRINK2_REPLAY,
            "shrink-2 generator component count drift",
        ));
    }

    let mut source_count = 0usize;
    let mut accepted_count = 0usize;
    let mut rejection_counts = BTreeMap::new();
    let mut canonical_set = BTreeSet::new();
    for constant_atom in &constant_atoms {
        for mixed_atom in &mixed_atoms {
            source_count += 1;
            let source = Node::And(vec![constant_atom.clone(), mixed_atom.clone()]);
            match canonicalize_shrink2_source_node(source) {
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
    if source_count != EXPECTED_SHRINK2_SOURCE_COUNT {
        return Err(Shrink2Error::new(
            REJECT_INTERNAL_SHRINK2_REPLAY,
            format!(
                "shrink-2 generator emitted {source_count}; expected {EXPECTED_SHRINK2_SOURCE_COUNT}"
            ),
        ));
    }

    let rejected_count = rejection_counts.values().sum();
    let accepted_unique_count = canonical_set.len();
    let rewrite_collapsed_count = accepted_count
        .checked_sub(accepted_unique_count)
        .ok_or_else(|| {
            Shrink2Error::new(
                REJECT_INTERNAL_SHRINK2_REPLAY,
                "unique count exceeds accepted count",
            )
        })?;
    let accepted_set_commitment = capacity_set_commitment(&canonical_set);
    let first = canonical_set.iter().next().ok_or_else(|| {
        Shrink2Error::new(
            REJECT_INTERNAL_SHRINK2_REPLAY,
            "shrink-2 accepted set is unexpectedly empty",
        )
    })?;
    let last = canonical_set.iter().next_back().ok_or_else(|| {
        Shrink2Error::new(
            REJECT_INTERNAL_SHRINK2_REPLAY,
            "shrink-2 accepted set is unexpectedly empty",
        )
    })?;
    let first_program = decode_shrink2_canonical_ast(first)?;
    let last_program = decode_shrink2_canonical_ast(last)?;
    let first_hex = hegel_strict_canonicalizer::hex_encode(first);
    let last_hex = hegel_strict_canonicalizer::hex_encode(last);
    let out_of_budget = canonical_set.iter().nth(CANONICAL_PROGRAM_BUDGET);

    let subset_invariants_hold = source_count == EXPECTED_SHRINK2_SOURCE_COUNT
        && accepted_count == EXPECTED_SHRINK2_SOURCE_COUNT
        && accepted_unique_count == EXPECTED_SHRINK2_SOURCE_COUNT
        && rejected_count == 0
        && rejection_counts.is_empty()
        && rewrite_collapsed_count == 0
        && accepted_set_commitment == EXPECTED_SHRINK2_ACCEPTED_SET_COMMITMENT
        && first_hex == EXPECTED_SHRINK2_FIRST_CANONICAL_CBOR_HEX
        && last_hex == EXPECTED_SHRINK2_LAST_CANONICAL_CBOR_HEX
        && out_of_budget.is_none();
    if !subset_invariants_hold {
        return Err(Shrink2Error::new(
            REJECT_INTERNAL_SHRINK2_REPLAY,
            format!(
                "frozen shrink-2 subset invariant failure: source={source_count}, \
                 accepted={accepted_count}, unique={accepted_unique_count}, \
                 rejected={rejected_count}, collapsed={rewrite_collapsed_count}, \
                 commitment={accepted_set_commitment}, first={first_hex}, last={last_hex}, \
                 witness_present={}",
                out_of_budget.is_some()
            ),
        ));
    }

    Ok(Shrink2CapacityReplayReport {
        schema_version: "hegel-strict-capacity-replay-shrink2/1",
        implementation: "rust",
        parent_dsl_version: PARENT_DSL_VERSION,
        parent_freeze_version: PARENT_FREEZE_VERSION,
        dsl_version: DSL_VERSION,
        freeze_version: FREEZE_VERSION,
        human_amendment_id: HUMAN_AMENDMENT_ID,
        shrink_step_id: SHRINK_STEP_ID,
        generator_rule: SHRINK2_CAPACITY_GENERATOR_RULE,
        active_rational_parameter_ids: ACTIVE_RATIONAL_PARAMETER_IDS,
        rational_aggregate_map_ids: [0, 5],
        constant_atom_count: constant_atoms.len(),
        rational_aggregate_count: rational_aggregates.len(),
        mixed_atom_count: mixed_atoms.len(),
        source_candidate_count: source_count,
        accepted_source_count: accepted_count,
        accepted_unique_count,
        rejected_count,
        rejection_counts,
        rewrite_collapsed_count,
        accepted_set_commitment,
        first_canonical_cbor_hex: first_hex,
        first_canonical_ast_hash: first_program.canonical_ast_hash_id(),
        last_canonical_cbor_hex: last_hex,
        last_canonical_ast_hash: last_program.canonical_ast_hash_id(),
        canonical_program_budget: CANONICAL_PROGRAM_BUDGET,
        first_out_of_budget_ordinal: out_of_budget.map(|_| CANONICAL_PROGRAM_BUDGET + 1),
        subset_status: "SUBSET_ONLY_NOT_COMPLETE",
        executed_closure_status: "NOT_RUN",
        complete_closure_enumerated: false,
        interpreted_as_complete_closure: false,
        formal_roots: None,
    })
}

#[derive(Debug, Clone, Serialize)]
pub struct Shrink2GoldenReplayReport {
    pub schema_version: &'static str,
    pub implementation: &'static str,
    pub parent_dsl_version: &'static str,
    pub parent_freeze_version: &'static str,
    pub dsl_version: &'static str,
    pub freeze_version: &'static str,
    pub human_amendment_id: &'static str,
    pub shrink_step_id: &'static str,
    pub rational_parameter_registry_namespace: &'static str,
    pub active_rational_parameter_ids: [u64; 3],
    pub tombstoned_rational_parameter_ids: [u64; 4],
    pub reserved_rational_parameter_ids: [u64; 1],
    pub vector_count: usize,
    pub passed_count: usize,
    pub surviving_identity_checks: usize,
    pub operator_preservation_checks: usize,
    pub source_rejection_checks: usize,
    pub source_boundary_checks: usize,
    pub source_wide_integer_checks: usize,
    pub source_malformed_checks: usize,
    pub tombstone_priority_checks: usize,
    pub formal_rejection_checks: usize,
    pub formal_failure_code_checks: usize,
    pub execution_state: &'static str,
    pub closure_executed: bool,
    pub formal_roots_generated: bool,
    pub formal_roots: Option<String>,
}

fn golden_failure(message: impl Into<String>) -> Shrink2Error {
    Shrink2Error::new(REJECT_INTERNAL_SHRINK2_REPLAY, message)
}

/// Replay deterministic source/formal vectors for sparse admission and
/// child-aware constant folding. This is mechanics qualification, not closure.
pub fn replay_shrink2_golden_vectors() -> Result<Shrink2GoldenReplayReport, Shrink2Error> {
    let mut vector_count = 0usize;
    let mut surviving_identity_checks = 0usize;
    let mut operator_preservation_checks = 0usize;
    let mut source_rejection_checks = 0usize;
    let mut source_boundary_checks = 0usize;
    let mut source_wide_integer_checks = 0usize;
    let mut source_malformed_checks = 0usize;
    let mut tombstone_priority_checks = 0usize;
    let mut formal_rejection_checks = 0usize;
    let mut formal_failure_code_checks = 0usize;

    for index in ACTIVE_RATIONAL_PARAMETER_IDS {
        vector_count += 1;
        let source = json!(["scalar_const", index]);
        let parent = hegel_strict_canonicalizer_shrink1::canonicalize_shrink1_source_json(&source)
            .map_err(|error| golden_failure(format!("parent rejected active ID {index}: {error}")))?;
        let child = canonicalize_shrink2_source_json(&source)?;
        if child.canonical_cbor != parent.canonical_cbor
            || child.canonical_ast_hash != parent.canonical_ast_hash
        {
            return Err(golden_failure(format!(
                "active RationalParameterId {index} changed surviving identity"
            )));
        }
        surviving_identity_checks += 1;
    }

    let surviving_sources = [
        json!(["add", ["scalar_const", 1], ["scalar_const", 5]]),
        json!(["absolute", ["scalar_const", 1]]),
        json!([
            "aggregate",
            "signed_balance_v1",
            "scope_all_observed_v1",
            "q0",
            []
        ]),
        json!([
            "less_equal",
            ["scalar_const", 1],
            ["aggregate", "sum_v1", "scope_primary_only_v1", "q1", []]
        ]),
    ];
    for source in surviving_sources {
        vector_count += 1;
        let parent = hegel_strict_canonicalizer_shrink1::canonicalize_shrink1_source_json(&source)
            .map_err(|error| golden_failure(format!("parent surviving vector failed: {error}")))?;
        let child = canonicalize_shrink2_source_json(&source)?;
        if child.canonical_cbor != parent.canonical_cbor
            || child.canonical_ast_hash != parent.canonical_ast_hash
        {
            return Err(golden_failure("surviving AST bytes/hash changed"));
        }
        surviving_identity_checks += 1;
    }

    for index in TOMBSTONED_RATIONAL_PARAMETER_IDS {
        vector_count += 1;
        let error = canonicalize_shrink2_source_json(&json!(["scalar_const", index]))
            .expect_err("tombstoned source parameter must reject");
        if error.code != REJECT_REMOVED_RATIONAL_PARAMETER {
            return Err(golden_failure(format!(
                "source tombstone {index} returned {}",
                error.code
            )));
        }
        source_rejection_checks += 1;
    }

    for map_id in TOMBSTONED_AGGREGATE_MAP_IDS {
        vector_count += 1;
        let source = json!(["aggregate", map_id, 0, 0, []]);
        let error = canonicalize_shrink2_source_json(&source)
            .expect_err("inherited aggregate tombstone must reject");
        if error.code != REJECT_REMOVED_AGGREGATE_MAP {
            return Err(golden_failure(format!(
                "aggregate tombstone {map_id} returned {}",
                error.code
            )));
        }
        source_rejection_checks += 1;
    }

    vector_count += 1;
    let reserved = canonicalize_shrink2_source_json(&json!(["scalar_const", 7]))
        .expect_err("reserved ID 7 must be out of range");
    if reserved.code != REJECT_REGISTRY_INDEX_OUT_OF_RANGE {
        return Err(golden_failure(format!(
            "reserved ID 7 returned {}",
            reserved.code
        )));
    }
    source_rejection_checks += 1;

    let source_boundary_cases = [
        json!(["scalar_const", -1]),
        json!(["scalar_const", -1, -1]),
        json!(["scalar_const", -2, -1]),
        json!(["bit_at", -1]),
        json!(["context_flag", -1]),
        json!(["task_flag", -1]),
        json!(["scalar_const", "bad-index"]),
        json!(["bit_at", true]),
        json!(["aggregate", false, "scope_all_observed_v1", "q0", []]),
        json!(["context_flag", []]),
        json!(["aggregate", -1, "scope_all_observed_v1", "q0", []]),
        json!(["aggregate", "sum_v1", -1, "q0", []]),
        json!(["aggregate", "sum_v1", "scope_all_observed_v1", -1, []]),
        json!([
            "approx_equal",
            ["scalar_const", 1],
            ["scalar_const", 5],
            -1
        ]),
        json!([
            "approx_equal",
            ["scalar_const", 1],
            ["scalar_const", 5],
            -1,
            -4
        ]),
    ];
    for source in source_boundary_cases {
        vector_count += 1;
        let error = canonicalize_shrink2_source_json(&source)
            .expect_err("source numeric boundary vector must reject");
        if error.code != REJECT_REGISTRY_INDEX_OUT_OF_RANGE {
            return Err(golden_failure(format!(
                "source numeric boundary returned {} for {source}",
                error.code
            )));
        }
        source_rejection_checks += 1;
        source_boundary_checks += 1;
    }

    vector_count += 1;
    let malformed_tolerance = json!([
        "approx_equal",
        ["scalar_const", 1],
        ["scalar_const", 5],
        "not-an-index"
    ]);
    let malformed_error = canonicalize_shrink2_source_json(&malformed_tolerance)
        .expect_err("non-integer tolerance shorthand must reject");
    if malformed_error.code != REJECT_MALFORMED_SOURCE_AST {
        return Err(golden_failure(format!(
            "malformed tolerance shorthand returned {}",
            malformed_error.code
        )));
    }
    source_rejection_checks += 1;
    source_malformed_checks += 1;

    let wide_active: Value = serde_json::from_str(
        r#"["scalar_const",10000000000000000000000000000000000000000,10000000000000000000000000000000000000000]"#,
    )
    .map_err(|error| golden_failure(format!("wide active JSON failed: {error}")))?;
    vector_count += 1;
    let wide_active_program = canonicalize_shrink2_source_json(&wide_active)?;
    if wide_active_program.canonical_node != Node::ScalarConst(5) {
        return Err(golden_failure(
            "arbitrary-width rational alias did not resolve to active ID 5",
        ));
    }
    source_wide_integer_checks += 1;

    for source_text in [
        r#"["scalar_const",10000000000000000000000000000000000000000,1]"#,
        r#"["scalar_const",10000000000000000000000000000000000000000]"#,
    ] {
        vector_count += 1;
        let source: Value = serde_json::from_str(source_text)
            .map_err(|error| golden_failure(format!("wide rejection JSON failed: {error}")))?;
        let error = canonicalize_shrink2_source_json(&source)
            .expect_err("wide numeric boundary vector must reject");
        if error.code != REJECT_REGISTRY_INDEX_OUT_OF_RANGE {
            return Err(golden_failure(format!(
                "wide numeric boundary returned {}",
                error.code
            )));
        }
        source_rejection_checks += 1;
        source_wide_integer_checks += 1;
    }

    vector_count += 1;
    let mixed_tombstones = json!([
        "less_equal",
        ["scalar_const", 0],
        [
            "aggregate",
            "mean_v1",
            "scope_all_observed_v1",
            "q0",
            []
        ]
    ]);
    let mixed_error = canonicalize_shrink2_source_json(&mixed_tombstones)
        .expect_err("mixed tombstones must reject");
    if mixed_error.code != REJECT_REMOVED_AGGREGATE_MAP {
        return Err(golden_failure(format!(
            "mixed tombstone priority returned {}",
            mixed_error.code
        )));
    }
    source_rejection_checks += 1;
    tombstone_priority_checks += 1;

    let preserved_operators = [
        (
            json!(["add", ["scalar_const", 5], ["scalar_const", 5]]),
            BinaryOp::Add,
            "82018402008300000583000005",
        ),
        (
            json!(["difference", ["scalar_const", 1], ["scalar_const", 5]]),
            BinaryOp::Difference,
            "82018402018300000183000005",
        ),
        (
            json!(["add", ["scalar_const", 1], ["scalar_const", 1]]),
            BinaryOp::Add,
            "82018402008300000183000001",
        ),
        (
            json!(["difference", ["scalar_const", 5], ["scalar_const", 1]]),
            BinaryOp::Difference,
            "82018402018300000583000001",
        ),
    ];
    for (source, expected_op, expected_hex) in preserved_operators {
        vector_count += 1;
        let program = canonicalize_shrink2_source_json(&source)?;
        if !matches!(&program.canonical_node, Node::Binary { op, .. } if *op == expected_op)
            || hegel_strict_canonicalizer::hex_encode(&program.canonical_cbor) != expected_hex
        {
            return Err(golden_failure(format!(
                "inactive-result fold did not preserve {expected_op:?} operator identity"
            )));
        }
        let decoded = decode_shrink2_canonical_ast(&program.canonical_cbor)?;
        if decoded != program {
            return Err(golden_failure(
                "child-specific operator AST failed formal round trip",
            ));
        }
        operator_preservation_checks += 1;
    }

    for (source, expected_index) in [
        (
            json!(["add", ["scalar_const", 1], ["scalar_const", 5]]),
            3,
        ),
        (
            json!(["difference", ["scalar_const", 5], ["scalar_const", 5]]),
            3,
        ),
        (json!(["absolute", ["scalar_const", 1]]), 5),
    ] {
        vector_count += 1;
        let program = canonicalize_shrink2_source_json(&source)?;
        if program.canonical_node != Node::ScalarConst(expected_index) {
            return Err(golden_failure(format!(
                "active-result fold did not produce RationalParameterId {expected_index}"
            )));
        }
        operator_preservation_checks += 1;
    }

    for index in TOMBSTONED_RATIONAL_PARAMETER_IDS {
        vector_count += 1;
        let bytes = encode_strict_cbor_json(&json!([1, [0, 0, index]]))?;
        let error = decode_shrink2_canonical_ast(&bytes)
            .expect_err("formal tombstoned parameter must reject");
        if error.code != REJECT_REMOVED_RATIONAL_PARAMETER {
            return Err(golden_failure(format!(
                "formal tombstone {index} returned {}",
                error.code
            )));
        }
        formal_rejection_checks += 1;
    }

    vector_count += 1;
    let formal_reserved = encode_strict_cbor_json(&json!([1, [0, 0, 7]]))?;
    let error = decode_shrink2_canonical_ast(&formal_reserved)
        .expect_err("formal reserved ID 7 must reject");
    if error.code != REJECT_REGISTRY_INDEX_OUT_OF_RANGE {
        return Err(golden_failure(format!(
            "formal reserved ID 7 returned {}",
            error.code
        )));
    }
    formal_rejection_checks += 1;

    let formal_failure_cases = [
        (json!([2, [0, 0, 0]]), REJECT_UNKNOWN_AST_SCHEMA),
        (json!([1, [-1]]), REJECT_UNKNOWN_EXPRESSION),
        (json!([1, [0, -1]]), REJECT_UNKNOWN_EXPRESSION),
        (
            json!([1, [0, 0, -1]]),
            REJECT_REGISTRY_INDEX_OUT_OF_RANGE,
        ),
        (
            json!([1, [1, 4, [0, 0, 3]]]),
            REJECT_REGISTRY_INDEX_OUT_OF_RANGE,
        ),
        (
            json!([1, [2, 7, [0, 0, 3], [0, 0, 3]]]),
            REJECT_NONCANONICAL_AST,
        ),
        (
            json!([1, [3, 1, [0, 0, 3], [0, 0, 3], 0]]),
            REJECT_REGISTRY_INDEX_OUT_OF_RANGE,
        ),
        (
            json!([1, [4, [[0, 4, 0], [0, 4, 1], [0, 4, 2], [0, 4, 3]]]]),
            REJECT_NONCANONICAL_AST,
        ),
        (
            json!([1, [0, 3, 0, 0, 0, [[0, false], [0, true]]]]),
            REJECT_NONCANONICAL_AST,
        ),
        (
            json!([1, [0, 3, 0, 0, 0, [[0, false], [1, false], [2, false]]]]),
            REJECT_NONCANONICAL_AST,
        ),
        (
            json!([1, [0, 3, 99, 0, 0, [[0]]]]),
            REJECT_REGISTRY_INDEX_OUT_OF_RANGE,
        ),
        (
            json!([1, [2, 2, [0, 0, 0], [0, 0, 3, 99]]]),
            REJECT_NONCANONICAL_AST,
        ),
    ];
    for (formal, expected_code) in formal_failure_cases {
        vector_count += 1;
        let bytes = encode_strict_cbor_json(&formal)?;
        let error = decode_shrink2_canonical_ast(&bytes)
            .expect_err("formal failure-code vector must reject");
        if error.code != expected_code {
            return Err(golden_failure(format!(
                "formal failure-code vector returned {} instead of {expected_code}: {formal}",
                error.code
            )));
        }
        formal_rejection_checks += 1;
        formal_failure_code_checks += 1;
    }

    Ok(Shrink2GoldenReplayReport {
        schema_version: GOLDEN_SCHEMA_VERSION,
        implementation: "rust",
        parent_dsl_version: PARENT_DSL_VERSION,
        parent_freeze_version: PARENT_FREEZE_VERSION,
        dsl_version: DSL_VERSION,
        freeze_version: FREEZE_VERSION,
        human_amendment_id: HUMAN_AMENDMENT_ID,
        shrink_step_id: SHRINK_STEP_ID,
        rational_parameter_registry_namespace: RATIONAL_PARAMETER_REGISTRY_NAMESPACE,
        active_rational_parameter_ids: ACTIVE_RATIONAL_PARAMETER_IDS,
        tombstoned_rational_parameter_ids: TOMBSTONED_RATIONAL_PARAMETER_IDS,
        reserved_rational_parameter_ids: RESERVED_RATIONAL_PARAMETER_IDS,
        vector_count,
        passed_count: vector_count,
        surviving_identity_checks,
        operator_preservation_checks,
        source_rejection_checks,
        source_boundary_checks,
        source_wide_integer_checks,
        source_malformed_checks,
        tombstone_priority_checks,
        formal_rejection_checks,
        formal_failure_code_checks,
        execution_state: "NOT_RUN",
        closure_executed: false,
        formal_roots_generated: false,
        formal_roots: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use hegel_strict_canonicalizer::{
        canonicalize_source_json, decode_strict_canonical_ast, encode_strict_cbor_json,
    };

    fn source_constant(index: u64) -> Value {
        json!(["scalar_const", index])
    }

    #[test]
    fn registry_disposition_is_sparse_and_never_compacted() {
        assert_eq!(HUMAN_AMENDMENT_ID, "hegel-freeze-p2b-p3-v1.2.0-shrink-step2");
        assert_eq!(
            SHRINK_STEP_ID,
            "SHRINK_STEP_2_REDUCE_RATIONAL_PARAMETER_TO_NEG1_ZERO_POS1"
        );
        assert_eq!(ACTIVE_RATIONAL_PARAMETER_IDS, [1, 3, 5]);
        assert_eq!(TOMBSTONED_RATIONAL_PARAMETER_IDS, [0, 2, 4, 6]);
        assert_eq!(RESERVED_RATIONAL_PARAMETER_IDS, [7]);
        assert_eq!(ACTIVE_AGGREGATE_MAP_IDS, [0, 1, 5]);
        assert_eq!(TOMBSTONED_AGGREGATE_MAP_IDS, [2, 3, 4]);
    }

    #[test]
    fn source_ids_and_equivalent_rational_pairs_share_removed_error() {
        let cases = [
            json!(["scalar_const", 0]),
            json!(["scalar_const", -2, 1]),
            json!(["scalar_const", -4, 2]),
            json!(["scalar_const", 2]),
            json!(["scalar_const", -1, 2]),
            json!(["scalar_const", -2, 4]),
            json!(["scalar_const", 4]),
            json!(["scalar_const", 1, 2]),
            json!(["scalar_const", 6]),
            json!(["scalar_const", 2, 1]),
        ];
        for source in cases {
            let error = canonicalize_shrink2_source_json(&source).unwrap_err();
            assert_eq!(error.code, REJECT_REMOVED_RATIONAL_PARAMETER);
        }
        let active = canonicalize_shrink2_source_json(&json!(["scalar_const", -2, 2])).unwrap();
        assert_eq!(active.canonical_node, Node::ScalarConst(1));
    }

    #[test]
    fn source_numeric_domain_matches_positive_denominator_contract() {
        let range_cases = [
            json!(["scalar_const", -1]),
            json!(["scalar_const", -1, -1]),
            json!(["scalar_const", -2, -1]),
            json!(["bit_at", -1]),
            json!(["context_flag", -1]),
            json!(["task_flag", -1]),
            json!(["scalar_const", "bad-index"]),
            json!(["bit_at", true]),
            json!(["aggregate", false, "scope_all_observed_v1", "q0", []]),
            json!(["context_flag", []]),
            json!([
                "aggregate",
                -1,
                "scope_all_observed_v1",
                "q0",
                []
            ]),
            json!([
                "aggregate",
                "sum_v1",
                -1,
                "q0",
                []
            ]),
            json!([
                "aggregate",
                "sum_v1",
                "scope_all_observed_v1",
                -1,
                []
            ]),
            json!([
                "approx_equal",
                ["scalar_const", 1],
                ["scalar_const", 5],
                -1
            ]),
            json!([
                "approx_equal",
                ["scalar_const", 1],
                ["scalar_const", 5],
                -1,
                -4
            ]),
        ];
        for source in range_cases {
            let error = canonicalize_shrink2_source_json(&source).unwrap_err();
            assert_eq!(error.code, REJECT_REGISTRY_INDEX_OUT_OF_RANGE, "{source}");
        }
        let malformed_tolerance = json!([
            "approx_equal",
            ["scalar_const", 1],
            ["scalar_const", 5],
            "not-an-index"
        ]);
        assert_eq!(
            canonicalize_shrink2_source_json(&malformed_tolerance)
                .unwrap_err()
                .code,
            REJECT_MALFORMED_SOURCE_AST
        );
        let active = canonicalize_shrink2_source_json(&json!(["scalar_const", -2, 2])).unwrap();
        assert_eq!(active.canonical_node, Node::ScalarConst(1));

        let wide_active: Value = serde_json::from_str(
            r#"["scalar_const",10000000000000000000000000000000000000000,10000000000000000000000000000000000000000]"#,
        )
        .unwrap();
        assert_eq!(
            canonicalize_shrink2_source_json(&wide_active)
                .unwrap()
                .canonical_node,
            Node::ScalarConst(5)
        );
        for source_text in [
            r#"["scalar_const",10000000000000000000000000000000000000000,1]"#,
            r#"["scalar_const",10000000000000000000000000000000000000000]"#,
        ] {
            let source: Value = serde_json::from_str(source_text).unwrap();
            let error = canonicalize_shrink2_source_json(&source).unwrap_err();
            assert_eq!(error.code, REJECT_REGISTRY_INDEX_OUT_OF_RANGE);
        }
    }

    #[test]
    fn aggregate_tombstone_has_global_priority_after_valid_source_parse() {
        let source = json!([
            "less_equal",
            ["scalar_const", 0],
            [
                "aggregate",
                "mean_v1",
                "scope_all_observed_v1",
                "q0",
                []
            ]
        ]);
        let error = canonicalize_shrink2_source_json(&source).unwrap_err();
        assert_eq!(error.code, REJECT_REMOVED_AGGREGATE_MAP);
    }

    #[test]
    fn reserved_parameter_id_is_out_of_range_not_a_tombstone() {
        let source_error = canonicalize_shrink2_source_json(&source_constant(7)).unwrap_err();
        assert_eq!(source_error.code, REJECT_REGISTRY_INDEX_OUT_OF_RANGE);

        let formal = encode_strict_cbor_json(&json!([1, [0, 0, 7]])).unwrap();
        let formal_error = decode_shrink2_canonical_ast(&formal).unwrap_err();
        assert_eq!(formal_error.code, REJECT_REGISTRY_INDEX_OUT_OF_RANGE);
    }

    #[test]
    fn source_precheck_does_not_scan_malformed_non_ast_payloads() {
        let cases = [
            json!(["scalar_const", 0, 1, ["scalar_const", 6]]),
            json!(["unknown_outer", ["scalar_const", 0]]),
            json!(["top_level_AND", [[], ["scalar_const", 0]]]),
        ];
        for source in cases {
            let parent = canonicalize_source_json(&source).unwrap_err();
            let child = canonicalize_shrink2_source_json(&source).unwrap_err();
            assert_eq!(child.code, parent.code);
            assert_ne!(child.code, REJECT_REMOVED_RATIONAL_PARAMETER);
        }
    }

    #[test]
    fn inherited_aggregate_tombstones_reject_at_both_boundaries() {
        for map_id in TOMBSTONED_AGGREGATE_MAP_IDS {
            let source = json!(["aggregate", map_id, 0, 0, []]);
            let source_error = canonicalize_shrink2_source_json(&source).unwrap_err();
            assert_eq!(source_error.code, REJECT_REMOVED_AGGREGATE_MAP);

            let parent = canonicalize_source_json(&source).unwrap();
            let formal_error = decode_shrink2_canonical_ast(&parent.canonical_cbor).unwrap_err();
            assert_eq!(formal_error.code, REJECT_REMOVED_AGGREGATE_MAP);
        }
    }

    #[test]
    fn inactive_add_fold_results_preserve_operator_ast() {
        for (index, expected_hex) in [
            (1, "82018402008300000183000001"),
            (5, "82018402008300000583000005"),
        ] {
            let source = json!(["add", ["scalar_const", index], ["scalar_const", index]]);
            let child = canonicalize_shrink2_source_json(&source).unwrap();
            assert!(matches!(
                child.canonical_node,
                Node::Binary {
                    op: BinaryOp::Add,
                    ..
                }
            ));
            assert_eq!(
                hegel_strict_canonicalizer::hex_encode(&child.canonical_cbor),
                expected_hex
            );
            assert_eq!(decode_shrink2_canonical_ast(&child.canonical_cbor).unwrap(), child);
            assert_eq!(
                decode_strict_canonical_ast(&child.canonical_cbor)
                    .unwrap_err()
                    .code,
                REJECT_NONCANONICAL_AST
            );
        }
    }

    #[test]
    fn inactive_difference_fold_results_preserve_operator_ast() {
        for (left, right, expected_hex) in [
            (1, 5, "82018402018300000183000005"),
            (5, 1, "82018402018300000583000001"),
        ] {
            let source = json!([
                "difference",
                ["scalar_const", left],
                ["scalar_const", right]
            ]);
            let child = canonicalize_shrink2_source_json(&source).unwrap();
            assert!(matches!(
                child.canonical_node,
                Node::Binary {
                    op: BinaryOp::Difference,
                    ..
                }
            ));
            assert_eq!(
                hegel_strict_canonicalizer::hex_encode(&child.canonical_cbor),
                expected_hex
            );
            assert_eq!(decode_shrink2_canonical_ast(&child.canonical_cbor).unwrap(), child);
        }
    }

    #[test]
    fn active_fold_results_still_reach_parent_normal_form() {
        let cases = [
            (
                json!(["add", ["scalar_const", 1], ["scalar_const", 3]]),
                1,
            ),
            (
                json!(["add", ["scalar_const", 1], ["scalar_const", 5]]),
                3,
            ),
            (
                json!(["add", ["scalar_const", 3], ["scalar_const", 5]]),
                5,
            ),
            (
                json!(["difference", ["scalar_const", 3], ["scalar_const", 1]]),
                5,
            ),
            (
                json!(["difference", ["scalar_const", 3], ["scalar_const", 5]]),
                1,
            ),
            (
                json!(["difference", ["scalar_const", 5], ["scalar_const", 5]]),
                3,
            ),
            (json!(["absolute", ["scalar_const", 1]]), 5),
        ];
        for (source, expected_index) in cases {
            let parent = hegel_strict_canonicalizer_shrink1::canonicalize_shrink1_source_json(
                &source,
            )
            .unwrap();
            let child = canonicalize_shrink2_source_json(&source).unwrap();
            assert_eq!(child.canonical_node, Node::ScalarConst(expected_index));
            assert_eq!(child.canonical_cbor, parent.canonical_cbor);
            assert_eq!(child.canonical_ast_hash, parent.canonical_ast_hash);
        }
    }

    #[test]
    fn surviving_nonconstant_ast_identity_is_byte_and_hash_stable() {
        let sources = [
            json!([
                "aggregate",
                "signed_balance_v1",
                "scope_all_observed_v1",
                "q0",
                []
            ]),
            json!([
                "less_equal",
                ["scalar_const", 1],
                ["aggregate", "sum_v1", "scope_primary_only_v1", "q1", []]
            ]),
            json!([
                "equal_exact",
                ["absolute", ["scalar_const", 1]],
                ["scalar_const", 5]
            ]),
            json!([
                "top_level_AND",
                [
                    ["context_flag", "c0"],
                    ["task_flag", "t1"]
                ]
            ]),
        ];
        for source in sources {
            let parent = hegel_strict_canonicalizer_shrink1::canonicalize_shrink1_source_json(
                &source,
            )
            .unwrap();
            let child = canonicalize_shrink2_source_json(&source).unwrap();
            assert_eq!(child.canonical_cbor, parent.canonical_cbor);
            assert_eq!(child.canonical_ast_hash, parent.canonical_ast_hash);
            assert_eq!(decode_shrink2_canonical_ast(&parent.canonical_cbor).unwrap(), child);
        }
    }

    #[test]
    fn formal_parameter_tombstones_have_one_uniform_error() {
        for index in TOMBSTONED_RATIONAL_PARAMETER_IDS {
            let bytes = encode_strict_cbor_json(&json!([1, [0, 0, index]])).unwrap();
            validate_strict_cbor(&bytes).unwrap();
            decode_strict_canonical_ast(&bytes).unwrap();
            let error = decode_shrink2_canonical_ast(&bytes).unwrap_err();
            assert_eq!(error.code, REJECT_REMOVED_RATIONAL_PARAMETER);
        }
    }

    #[test]
    fn formal_tombstone_precheck_requires_a_real_v1_ast_envelope() {
        let schema = encode_strict_cbor_json(&json!([2, [0, 0, 0]])).unwrap();
        let schema_error = decode_shrink2_canonical_ast(&schema).unwrap_err();
        assert_eq!(schema_error.code, REJECT_UNKNOWN_AST_SCHEMA);

        let malformed = encode_strict_cbor_json(&json!([0, 0, 0])).unwrap();
        let malformed_error = decode_shrink2_canonical_ast(&malformed).unwrap_err();
        assert_eq!(malformed_error.code, REJECT_NONCANONICAL_AST);
        assert_ne!(
            malformed_error.code,
            REJECT_REMOVED_RATIONAL_PARAMETER
        );
    }

    #[test]
    fn formal_failure_code_and_tombstone_priority_matrix_is_exact() {
        let cases = [
            (json!([1, [-1]]), REJECT_UNKNOWN_EXPRESSION),
            (json!([1, [0, -1]]), REJECT_UNKNOWN_EXPRESSION),
            (
                json!([1, [0, 0, -1]]),
                REJECT_REGISTRY_INDEX_OUT_OF_RANGE,
            ),
            (
                json!([1, [1, 4, [0, 0, 3]]]),
                REJECT_REGISTRY_INDEX_OUT_OF_RANGE,
            ),
            (
                json!([1, [2, 7, [0, 0, 3], [0, 0, 3]]]),
                REJECT_NONCANONICAL_AST,
            ),
            (
                json!([1, [3, 1, [0, 0, 3], [0, 0, 3], 0]]),
                REJECT_REGISTRY_INDEX_OUT_OF_RANGE,
            ),
            (
                json!([1, [4, [[0, 4, 0], [0, 4, 1], [0, 4, 2], [0, 4, 3]]]]),
                REJECT_NONCANONICAL_AST,
            ),
            (
                json!([1, [0, 3, 0, 0, 0, [[0, false], [0, true]]]]),
                REJECT_NONCANONICAL_AST,
            ),
            (
                json!([1, [0, 3, 0, 0, 0, [[0, false], [1, false], [2, false]]]]),
                REJECT_NONCANONICAL_AST,
            ),
            (
                json!([1, [0, 3, 99, 0, 0, [[0]]]]),
                REJECT_REGISTRY_INDEX_OUT_OF_RANGE,
            ),
            (
                json!([1, [2, 2, [0, 0, 0], [0, 0, 3, 99]]]),
                REJECT_NONCANONICAL_AST,
            ),
        ];
        for (formal, expected) in cases {
            let bytes = encode_strict_cbor_json(&formal).unwrap();
            let error = decode_shrink2_canonical_ast(&bytes).unwrap_err();
            assert_eq!(error.code, expected, "{formal}");
        }
    }

    #[test]
    fn structural_limits_apply_after_operator_preserving_normalization() {
        let source = json!([
            "add",
            ["add", ["scalar_const", 5], ["scalar_const", 5]],
            ["add", ["scalar_const", 5], ["scalar_const", 5]]
        ]);
        let error = canonicalize_shrink2_source_json(&source).unwrap_err();
        assert_eq!(error.code, REJECT_STRUCTURAL_LIMIT);
    }

    #[test]
    fn golden_replay_is_deterministic_and_non_authoritative() {
        let first = replay_shrink2_golden_vectors().unwrap();
        let second = replay_shrink2_golden_vectors().unwrap();
        assert_eq!(first.vector_count, 59);
        assert_eq!(first.passed_count, first.vector_count);
        assert_eq!(first.surviving_identity_checks, 7);
        assert_eq!(first.operator_preservation_checks, 7);
        assert_eq!(first.source_rejection_checks, 27);
        assert_eq!(first.source_boundary_checks, 15);
        assert_eq!(first.source_wide_integer_checks, 3);
        assert_eq!(first.source_malformed_checks, 1);
        assert_eq!(first.tombstone_priority_checks, 1);
        assert_eq!(first.formal_rejection_checks, 17);
        assert_eq!(first.formal_failure_code_checks, 12);
        assert_eq!(first.execution_state, "NOT_RUN");
        assert_eq!(first.formal_roots, None);
        assert_eq!(
            serde_json::to_value(&first).unwrap(),
            serde_json::to_value(&second).unwrap()
        );
        assert!(!first.closure_executed);
        assert!(!first.formal_roots_generated);
    }

    #[test]
    fn constructive_capacity_subset_matches_python_commitment_but_is_not_closure() {
        let report = replay_shrink2_capacity_subset().unwrap();
        assert_eq!(report.constant_atom_count, 15);
        assert_eq!(report.rational_aggregate_count, 16);
        assert_eq!(report.mixed_atom_count, 144);
        assert_eq!(report.source_candidate_count, EXPECTED_SHRINK2_SOURCE_COUNT);
        assert_eq!(report.accepted_source_count, EXPECTED_SHRINK2_SOURCE_COUNT);
        assert_eq!(report.accepted_unique_count, EXPECTED_SHRINK2_SOURCE_COUNT);
        assert_eq!(report.rejected_count, 0);
        assert!(report.rejection_counts.is_empty());
        assert_eq!(report.rewrite_collapsed_count, 0);
        assert_eq!(
            report.accepted_set_commitment,
            EXPECTED_SHRINK2_ACCEPTED_SET_COMMITMENT
        );
        assert_eq!(
            report.first_canonical_cbor_hex,
            EXPECTED_SHRINK2_FIRST_CANONICAL_CBOR_HEX
        );
        assert_eq!(
            report.last_canonical_cbor_hex,
            EXPECTED_SHRINK2_LAST_CANONICAL_CBOR_HEX
        );
        assert_eq!(report.subset_status, "SUBSET_ONLY_NOT_COMPLETE");
        assert_eq!(report.executed_closure_status, "NOT_RUN");
        assert_eq!(report.formal_roots, None);
        assert!(!report.complete_closure_enumerated);
        assert!(!report.interpreted_as_complete_closure);
        assert!(report.first_out_of_budget_ordinal.is_none());
    }
}
