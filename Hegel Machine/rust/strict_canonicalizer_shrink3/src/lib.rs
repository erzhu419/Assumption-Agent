//! Independent Rust strict admission profile for `hegel-old-dsl-v1.3.0`.
//!
//! Shrink step 3 is a sparse admission change over shrink step 2.  It keeps
//! every `BinaryOperatorId/v1` code point fixed, tombstones ID 0 (`add`), and
//! retains ID 1 (`difference`) byte-for-byte.  Source and formal inputs still
//! parse the removed operator so that a removed identity is never confused
//! with an unknown identity or silently migrated to a different program.

use hegel_strict_canonicalizer::{
    encode_strict_cbor_json, type_check, validate_strict_cbor, BinaryOp, CanonicalError,
    CanonicalProgram, Node, Sort, UnaryOp, REJECT_CBOR_FLOAT, REJECT_CBOR_MAP,
    REJECT_CBOR_NESTING, REJECT_CBOR_TAG, REJECT_CBOR_TEXT,
    REJECT_DUPLICATE_SCOPE_CONTEXT, REJECT_IMPLICIT_COERCION, REJECT_INDEFINITE_CBOR,
    REJECT_MALFORMED_SOURCE_AST, REJECT_NEW_SYMBOL_IN_OLD_DSL, REJECT_NONCANONICAL_AST,
    REJECT_NONCANONICAL_CBOR, REJECT_NONCANONICAL_SCOPE_ALIAS,
    REJECT_REGISTRY_INDEX_OUT_OF_RANGE, REJECT_STRUCTURAL_LIMIT, REJECT_TRAILING_CBOR,
    REJECT_TYPE_MISMATCH, REJECT_UNKNOWN_EXPRESSION,
};
use hegel_strict_canonicalizer_shrink2::{
    canonicalize_shrink2_source_node, decode_shrink2_canonical_ast, Shrink2Error,
    REJECT_CBOR_SIMPLE, REJECT_CBOR_UNDEFINED, REJECT_EMPTY_CONJUNCTION,
    REJECT_REMOVED_AGGREGATE_MAP, REJECT_REMOVED_RATIONAL_PARAMETER,
    REJECT_RESERVED_CBOR, REJECT_TRUNCATED_CBOR, REJECT_UNKNOWN_AST_SCHEMA,
};
use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub const DSL_VERSION: &str = "hegel-old-dsl-v1.3.0";
pub const FREEZE_VERSION: &str = "hegel-freeze-p2b-p3-v1.3.0";
pub const PARENT_DSL_VERSION: &str = "hegel-old-dsl-v1.2.0";
pub const PARENT_FREEZE_VERSION: &str = "hegel-freeze-p2b-p3-v1.2.0";
pub const HUMAN_AMENDMENT_ID: &str = "hegel-freeze-p2b-p3-v1.3.0-shrink-step3";
pub const SHRINK_STEP_ID: &str = "SHRINK_STEP_3_REMOVE_ADD_RETAIN_DIFFERENCE";
pub const AST_SCHEMA_ID: &str = "hegel-canonical-ast-v1";
pub const CBOR_PROFILE_ID: &str = "hegel-cbor-det-v1";
pub const AST_HASH_DOMAIN: &str = "HEGEL/AST/V1";
pub const REPLAY_SCHEMA_VERSION: &str = "hegel-strict-canonicalizer-shrink3-replay/1";
pub const GOLDEN_SCHEMA_VERSION: &str = "hegel-strict-canonicalizer-shrink3-golden/1";
pub const CAPACITY_SCHEMA_VERSION: &str = "hegel-strict-capacity-replay-shrink3/1";
pub const BINARY_OPERATOR_REGISTRY_NAMESPACE: &str = "BinaryOperatorId/v1";
pub const REJECT_REMOVED_BINARY_OPERATOR: &str = "REJECT_REMOVED_BINARY_OPERATOR";
pub const REJECT_INTERNAL_SHRINK3_REPLAY: &str = "REJECT_INTERNAL_SHRINK3_REPLAY";

pub const ACTIVE_BINARY_OPERATOR_IDS_SOURCE: [u64; 6] = [1, 2, 3, 4, 5, 6];
pub const ACTIVE_BINARY_OPERATOR_IDS_FORMAL: [u64; 5] = [1, 2, 3, 5, 6];
pub const TOMBSTONED_BINARY_OPERATOR_IDS: [u64; 1] = [0];
pub const RESERVED_BINARY_OPERATOR_IDS: [u64; 1] = [7];
pub const ACTIVE_RATIONAL_PARAMETER_IDS: [u64; 3] = [1, 3, 5];
pub const TOMBSTONED_RATIONAL_PARAMETER_IDS: [u64; 4] = [0, 2, 4, 6];
pub const ACTIVE_AGGREGATE_MAP_IDS: [u64; 3] = [0, 1, 5];
pub const TOMBSTONED_AGGREGATE_MAP_IDS: [u64; 3] = [2, 3, 4];

pub const EXPECTED_SURVIVOR_SOURCE_COUNT: usize = 2_160;
pub const EXPECTED_SURVIVOR_ACCEPTED_SET_COMMITMENT: &str =
    "sha256:9045e4ebe6416dcbf699e7972f25468aef45c0f0aec0e58806061b7ce64d790e";
pub const EXPECTED_FIRST_CANONICAL_CBOR_HEX: &str =
    "820182048284020283000001830000018402028300000186000300000180";
pub const EXPECTED_LAST_CANONICAL_CBOR_HEX: &str =
    "820182048284020383000005830000058402038600030503018083000005";
pub const SURVIVOR_CAPACITY_GENERATOR_RULE: &str =
    "inherit the exact 2160-source shrink-2 target-free constructive subset; the subset contains no BinaryOperatorId 0/add node; require every source to retain identical canonical AST CBOR bytes and hash under shrink step 3";
pub const EXPECTED_FORMAL_SHAPE_COMPACT_JSON_SHA256: [&str; 6] = [
    "7279a81c8dc148fbb8ed92f39c73d4c98b8987ce32496342440b2e2da99176f3",
    "e85ccca2361541114cb6f51f8f567b5753c9448882c46959552412717af819c0",
    "05fb0c914d5a8dc122b24fa066c233bcfc7d5026a72e8003249fb14983789e21",
    "a9e14474f7a336ff4499615a89d783da8594e13a381636fc96307aee13d78120",
    "bf9a5953670e8295a145b9dec7cf93182fe0da26be99170bddc4715b50872d71",
    "aefe5b355a9ea9caa4262ec3595109a02206d1c6e9c2be073400cc3fd0998d33",
];

const MAX_CBOR_NESTING: usize = 64;
const RATIONAL_PARAMETER_COUNT: u64 = 7;
const AGGREGATE_MAP_COUNT: u64 = 6;
const SCOPE_COUNT: u64 = 4;
const QUANTITY_COUNT: u64 = 2;
const CONTEXT_COUNT: u64 = 4;
const TASK_COUNT: u64 = 2;
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
pub struct Shrink3Error {
    pub code: String,
    pub message: String,
}

impl Shrink3Error {
    fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }
}

impl From<CanonicalError> for Shrink3Error {
    fn from(error: CanonicalError) -> Self {
        Self::new(error.code, error.message)
    }
}

impl From<Shrink2Error> for Shrink3Error {
    fn from(error: Shrink2Error) -> Self {
        Self::new(error.code, error.message)
    }
}

impl fmt::Display for Shrink3Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for Shrink3Error {}

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
    let magnitude =
        u8::try_from(factor.unsigned_abs()).expect("frozen rational-grid factors fit in u8");
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
    let (Some(numerator), Some(denominator)) =
        (exact_json_integer(numerator), exact_json_integer(denominator))
    else {
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

fn source_array<'a>(value: &'a Value, context: &str) -> Result<&'a [Value], Shrink3Error> {
    value.as_array().map(Vec::as_slice).ok_or_else(|| {
        Shrink3Error::new(
            REJECT_MALFORMED_SOURCE_AST,
            format!("{context} must be an array"),
        )
    })
}

fn source_bounded_uint(
    value: &Value,
    upper_exclusive: u64,
    context: &str,
) -> Result<u64, Shrink3Error> {
    let Some(integer) = exact_json_integer(value) else {
        return Err(Shrink3Error::new(
            REJECT_REGISTRY_INDEX_OUT_OF_RANGE,
            format!("{context} must be an exact JSON uint"),
        ));
    };
    if integer.sign < 0 {
        return Err(Shrink3Error::new(
            REJECT_REGISTRY_INDEX_OUT_OF_RANGE,
            format!("{context} is negative"),
        ));
    }
    let index = integer.digits.parse::<u64>().map_err(|_| {
        Shrink3Error::new(
            REJECT_REGISTRY_INDEX_OUT_OF_RANGE,
            format!("{context} exceeds the frozen registry width"),
        )
    })?;
    if index >= upper_exclusive {
        return Err(Shrink3Error::new(
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
) -> Result<u64, Shrink3Error> {
    if value.is_number() {
        return source_bounded_uint(value, names.len() as u64, context);
    }
    value
        .as_str()
        .and_then(|name| names.iter().position(|candidate| *candidate == name))
        .map(|index| index as u64)
        .ok_or_else(|| {
            Shrink3Error::new(
                REJECT_REGISTRY_INDEX_OUT_OF_RANGE,
                format!("unknown frozen {context}"),
            )
        })
}

fn source_rational_index(
    parts: &[Value],
    grid: &[(i64, i64)],
    context: &str,
) -> Result<u64, Shrink3Error> {
    match parts {
        [index] => source_bounded_uint(index, grid.len() as u64, context),
        [numerator, denominator] => match rational_grid_boundary(numerator, denominator, grid) {
            RationalBoundary::NonInteger => Err(Shrink3Error::new(
                REJECT_MALFORMED_SOURCE_AST,
                format!("{context} rational pair must contain exact JSON integers"),
            )),
            RationalBoundary::Index(index) => Ok(index),
            RationalBoundary::OutOfRange => Err(Shrink3Error::new(
                REJECT_REGISTRY_INDEX_OUT_OF_RANGE,
                format!("{context} rational pair is outside its frozen grid"),
            )),
        },
        _ => Err(Shrink3Error::new(
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

fn source_type_error(operator: &str, actual: &[Sort], expected: &[Sort]) -> Shrink3Error {
    let implicit_bit_coercion = matches!(
        operator,
        "add" | "difference" | "equal_exact" | "less_equal" | "greater_equal"
    ) && actual.contains(&Sort::Bit);
    if implicit_bit_coercion {
        Shrink3Error::new(
            REJECT_IMPLICIT_COERCION,
            format!("{operator} received Bit; explicit bit_to_scalar is required"),
        )
    } else {
        Shrink3Error::new(
            REJECT_TYPE_MISMATCH,
            format!("{operator} expects {expected:?}, received {actual:?}"),
        )
    }
}

fn require_source_sorts(
    operator: &str,
    actual: &[Sort],
    expected: &[Sort],
) -> Result<(), Shrink3Error> {
    if actual == expected {
        Ok(())
    } else {
        Err(source_type_error(operator, actual, expected))
    }
}

/// Parse the frozen source vocabulary left-to-right without normalization or
/// whole-AST limit checks.  This is intentionally independent of shrink-2's
/// public parser because that parser canonicalizes before returning.
fn parse_typed_shrink3_source(value: &Value) -> Result<TypedSourceNode, Shrink3Error> {
    let items = source_array(value, "source AST node")?;
    let Some(name) = items.first().and_then(Value::as_str) else {
        return Err(Shrink3Error::new(
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
                return Err(Shrink3Error::new(
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
                return Err(Shrink3Error::new(
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
                return Err(Shrink3Error::new(
                    REJECT_MALFORMED_SOURCE_AST,
                    "aggregate requires map, scope, quantity, and extension",
                ));
            }
            if args[1].as_str() == Some(DEPRECATED_SCOPE_ALIAS) {
                return Err(Shrink3Error::new(
                    REJECT_NONCANONICAL_SCOPE_ALIAS,
                    "deprecated scope alias is migration-only",
                ));
            }
            let map_id = source_registry_index(&args[0], &AGGREGATE_MAP_NAMES, "aggregate map")?;
            let scope_id = source_registry_index(&args[1], &SCOPE_NAMES, "scope")?;
            let quantity_id = source_registry_index(&args[2], &QUANTITY_NAMES, "quantity")?;
            let raw_clauses = source_array(&args[3], "scope extension")?;
            if raw_clauses.len() > 2 {
                return Err(Shrink3Error::new(
                    REJECT_STRUCTURAL_LIMIT,
                    "scope extension exceeds two clauses",
                ));
            }
            let mut scope_extension = Vec::with_capacity(raw_clauses.len());
            for raw_clause in raw_clauses {
                let clause = source_array(raw_clause, "scope clause")?;
                if clause.len() != 2 || !clause[1].is_boolean() {
                    return Err(Shrink3Error::new(
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
                return Err(Shrink3Error::new(
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
                return Err(Shrink3Error::new(
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
        "new_symbol_call" => Err(Shrink3Error::new(
            REJECT_NEW_SYMBOL_IN_OLD_DSL,
            "new symbols are Phase-3B only",
        )),
        "bit_to_scalar" | "int_to_scalar" | "absolute" | "sign" => {
            if args.len() != 1 {
                return Err(Shrink3Error::new(
                    REJECT_MALFORMED_SOURCE_AST,
                    format!("{name} requires exactly one child"),
                ));
            }
            let child = parse_typed_shrink3_source(&args[0])?;
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
                return Err(Shrink3Error::new(
                    REJECT_MALFORMED_SOURCE_AST,
                    format!("{name} requires exactly two children"),
                ));
            }
            let left = parse_typed_shrink3_source(&args[0])?;
            let right = parse_typed_shrink3_source(&args[1])?;
            let (op, expected, output) = match name {
                "add" => (BinaryOp::Add, Sort::RationalValue, Sort::RationalValue),
                "difference" => (
                    BinaryOp::Difference,
                    Sort::RationalValue,
                    Sort::RationalValue,
                ),
                "equal_exact" => (BinaryOp::EqualExact, Sort::RationalValue, Sort::Bool),
                "less_equal" => (BinaryOp::LessEqual, Sort::RationalValue, Sort::Bool),
                "greater_equal" => (BinaryOp::GreaterEqual, Sort::RationalValue, Sort::Bool),
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
                return Err(Shrink3Error::new(
                    REJECT_MALFORMED_SOURCE_AST,
                    "approx_equal requires two children plus a tolerance",
                ));
            }
            let left = parse_typed_shrink3_source(&args[0])?;
            let right = parse_typed_shrink3_source(&args[1])?;
            require_source_sorts(
                name,
                &[left.sort, right.sort],
                &[Sort::RationalValue, Sort::RationalValue],
            )?;
            let tolerance_index = if args.len() == 3 {
                if exact_json_integer(&args[2]).is_none() {
                    return Err(Shrink3Error::new(
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
                return Err(Shrink3Error::new(
                    REJECT_EMPTY_CONJUNCTION,
                    "AND0 has no canonical true node",
                ));
            }
            let mut nodes = Vec::with_capacity(raw_children.len());
            let mut sorts = Vec::with_capacity(raw_children.len());
            for raw_child in raw_children {
                let child = parse_typed_shrink3_source(raw_child)?;
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
        _ => Err(Shrink3Error::new(
            REJECT_UNKNOWN_EXPRESSION,
            format!("unknown old-DSL expression {name:?}"),
        )),
    }
}

fn reject_removed_aggregate_nodes(node: &Node) -> Result<(), Shrink3Error> {
    match node {
        Node::Aggregate { map_id: 2..=4, .. } => Err(Shrink3Error::new(
            REJECT_REMOVED_AGGREGATE_MAP,
            "AggregateMapId 2, 3, or 4 remains tombstoned in hegel-old-dsl-v1.3.0",
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

fn reject_removed_parameter_nodes(node: &Node) -> Result<(), Shrink3Error> {
    match node {
        Node::ScalarConst(index) if TOMBSTONED_RATIONAL_PARAMETER_IDS.contains(index) => {
            Err(Shrink3Error::new(
                REJECT_REMOVED_RATIONAL_PARAMETER,
                format!(
                    "RationalParameterId {index} remains tombstoned in {DSL_VERSION}"
                ),
            ))
        }
        Node::Unary { child, .. } => reject_removed_parameter_nodes(child),
        Node::Binary { left, right, .. } | Node::ApproxEqual { left, right, .. } => {
            reject_removed_parameter_nodes(left)?;
            reject_removed_parameter_nodes(right)
        }
        Node::And(children) => children.iter().try_for_each(reject_removed_parameter_nodes),
        _ => Ok(()),
    }
}

fn reject_removed_add_nodes(node: &Node) -> Result<(), Shrink3Error> {
    match node {
        Node::Binary {
            op: BinaryOp::Add,
            ..
        } => Err(Shrink3Error::new(
            REJECT_REMOVED_BINARY_OPERATOR,
            format!(
                "BinaryOperatorId 0 (add) is tombstoned in {DSL_VERSION}; no migration is permitted"
            ),
        )),
        Node::Unary { child, .. } => reject_removed_add_nodes(child),
        Node::Binary { left, right, .. } | Node::ApproxEqual { left, right, .. } => {
            reject_removed_add_nodes(left)?;
            reject_removed_add_nodes(right)
        }
        Node::And(children) => children.iter().try_for_each(reject_removed_add_nodes),
        _ => Ok(()),
    }
}

/// Parse and canonicalize one source JSON AST under shrink-3 sparse admission.
///
/// The independent typed parser establishes source syntax, registry bounds,
/// and left-to-right type priority without normalization.  The three sparse
/// tombstone passes then run before shrink-2 normalization or whole-AST limit
/// checks can erase or mask a foldable, nested, or oversized `add`.
pub fn canonicalize_shrink3_source_json(value: &Value) -> Result<CanonicalProgram, Shrink3Error> {
    let source = parse_typed_shrink3_source(value)?.node;
    reject_removed_aggregate_nodes(&source)?;
    reject_removed_parameter_nodes(&source)?;
    reject_removed_add_nodes(&source)?;
    Ok(canonicalize_shrink2_source_node(source)?)
}

/// Canonicalize an already parsed source node while retaining shrink-2 error
/// precedence.  This entry point is used by the target-free capacity replay.
pub fn canonicalize_shrink3_source_node(source: Node) -> Result<CanonicalProgram, Shrink3Error> {
    type_check(&source)?;
    reject_removed_aggregate_nodes(&source)?;
    reject_removed_parameter_nodes(&source)?;
    reject_removed_add_nodes(&source)?;
    Ok(canonicalize_shrink2_source_node(source)?)
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

fn read_argument(additional: u8, bytes: &[u8], cursor: &mut usize) -> Result<u64, Shrink3Error> {
    let (width, minimum) = match additional {
        0..=23 => return Ok(u64::from(additional)),
        24 => (1, 24),
        25 => (2, 0x100),
        26 => (4, 0x1_0000),
        27 => (8, 0x1_0000_0000),
        31 => {
            return Err(Shrink3Error::new(
                REJECT_INDEFINITE_CBOR,
                "indefinite-length CBOR is forbidden",
            ))
        }
        _ => {
            return Err(Shrink3Error::new(
                REJECT_RESERVED_CBOR,
                "reserved CBOR additional-information value",
            ))
        }
    };
    let end = cursor
        .checked_add(width)
        .ok_or_else(|| Shrink3Error::new(REJECT_NONCANONICAL_CBOR, "CBOR cursor overflow"))?;
    let payload = bytes
        .get(*cursor..end)
        .ok_or_else(|| Shrink3Error::new(REJECT_TRUNCATED_CBOR, "truncated CBOR argument"))?;
    *cursor = end;
    let mut value = 0u64;
    for byte in payload {
        value = (value << 8) | u64::from(*byte);
    }
    if value < minimum {
        return Err(Shrink3Error::new(
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
) -> Result<ProbeValue, Shrink3Error> {
    if depth > MAX_CBOR_NESTING {
        return Err(Shrink3Error::new(
            REJECT_CBOR_NESTING,
            "CBOR exceeded the strict nesting limit",
        ));
    }
    let initial = *bytes
        .get(*cursor)
        .ok_or_else(|| Shrink3Error::new(REJECT_TRUNCATED_CBOR, "expected a CBOR item"))?;
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
            let length = usize::try_from(read_argument(additional, bytes, cursor)?)
                .map_err(|_| Shrink3Error::new(REJECT_NONCANONICAL_CBOR, "byte length overflow"))?;
            let end = cursor
                .checked_add(length)
                .ok_or_else(|| Shrink3Error::new(REJECT_NONCANONICAL_CBOR, "byte cursor overflow"))?;
            if bytes.get(*cursor..end).is_none() {
                return Err(Shrink3Error::new(
                    REJECT_TRUNCATED_CBOR,
                    "truncated CBOR byte string",
                ));
            }
            *cursor = end;
            Ok(ProbeValue::Bytes)
        }
        3 => Err(Shrink3Error::new(
            REJECT_CBOR_TEXT,
            "CBOR text strings are forbidden",
        )),
        4 => {
            let length = usize::try_from(read_argument(additional, bytes, cursor)?)
                .map_err(|_| Shrink3Error::new(REJECT_NONCANONICAL_CBOR, "array length overflow"))?;
            let mut values = Vec::with_capacity(length.min(bytes.len().saturating_sub(*cursor)));
            for _ in 0..length {
                values.push(parse_probe(bytes, cursor, depth + 1)?);
            }
            Ok(ProbeValue::Array(values))
        }
        5 => Err(Shrink3Error::new(REJECT_CBOR_MAP, "CBOR maps are forbidden")),
        6 => Err(Shrink3Error::new(REJECT_CBOR_TAG, "CBOR tags are forbidden")),
        7 if additional == 20 => Ok(ProbeValue::Bool(false)),
        7 if additional == 21 => Ok(ProbeValue::Bool(true)),
        7 if additional == 22 => Ok(ProbeValue::Null),
        7 if additional == 23 => Err(Shrink3Error::new(
            REJECT_CBOR_UNDEFINED,
            "CBOR undefined is forbidden",
        )),
        7 if matches!(additional, 25..=27) => Err(Shrink3Error::new(
            REJECT_CBOR_FLOAT,
            "CBOR floating point is forbidden",
        )),
        7 if additional == 31 => Err(Shrink3Error::new(
            REJECT_INDEFINITE_CBOR,
            "CBOR break is forbidden",
        )),
        7 => Err(Shrink3Error::new(
            REJECT_CBOR_SIMPLE,
            "unapproved CBOR simple value",
        )),
        _ => unreachable!("CBOR major type is three bits"),
    }
}

fn probe_array<'a>(value: &'a ProbeValue, context: &str) -> Result<&'a [ProbeValue], Shrink3Error> {
    match value {
        ProbeValue::Array(values) => Ok(values),
        _ => Err(Shrink3Error::new(
            REJECT_NONCANONICAL_AST,
            format!("{context} must be an array"),
        )),
    }
}

fn probe_registry_uint(value: &ProbeValue, context: &str) -> Result<u64, Shrink3Error> {
    match value {
        ProbeValue::Unsigned(value) => Ok(*value),
        _ => Err(Shrink3Error::new(
            REJECT_REGISTRY_INDEX_OUT_OF_RANGE,
            format!("{context} is outside its unsigned registry domain"),
        )),
    }
}

fn probe_bounded_registry_uint(
    value: &ProbeValue,
    upper_exclusive: u64,
    context: &str,
) -> Result<u64, Shrink3Error> {
    let index = probe_registry_uint(value, context)?;
    if index >= upper_exclusive {
        return Err(Shrink3Error::new(
            REJECT_REGISTRY_INDEX_OUT_OF_RANGE,
            format!("{context} is outside 0..{upper_exclusive}"),
        ));
    }
    Ok(index)
}

fn probe_expression_tag(value: &ProbeValue, context: &str) -> Result<u64, Shrink3Error> {
    match value {
        ProbeValue::Unsigned(value) => Ok(*value),
        ProbeValue::Negative => Err(Shrink3Error::new(
            REJECT_UNKNOWN_EXPRESSION,
            format!("{context} is an unknown negative expression ID"),
        )),
        _ => Err(Shrink3Error::new(
            REJECT_NONCANONICAL_AST,
            format!("{context} must be an integer"),
        )),
    }
}

fn probe_bool(value: &ProbeValue, context: &str) -> Result<bool, Shrink3Error> {
    match value {
        ProbeValue::Bool(value) => Ok(*value),
        _ => Err(Shrink3Error::new(
            REJECT_NONCANONICAL_AST,
            format!("{context} must be a boolean"),
        )),
    }
}

fn expect_probe_length(
    values: &[ProbeValue],
    expected: usize,
    context: &str,
) -> Result<(), Shrink3Error> {
    if values.len() != expected {
        return Err(Shrink3Error::new(
            REJECT_NONCANONICAL_AST,
            format!(
                "{context} has array length {}; expected {expected}",
                values.len()
            ),
        ));
    }
    Ok(())
}

fn formal_type_checked(node: Node) -> Result<Node, Shrink3Error> {
    if let Err(error) = type_check(&node) {
        if error.code == REJECT_IMPLICIT_COERCION {
            return Err(Shrink3Error::new(
                REJECT_TYPE_MISMATCH,
                "formal canonical AST has a child-sort mismatch",
            ));
        }
        return Err(error.into());
    }
    Ok(node)
}

/// Parse the formal numeric AST into the shared semantic node without making
/// a normalization decision.  Canonical-form-only defects are deliberately
/// left to the shrink-2 decoder after the three ordered tombstone scans.
fn node_from_probe(value: &ProbeValue) -> Result<Node, Shrink3Error> {
    let values = probe_array(value, "AST node")?;
    if values.is_empty() {
        return Err(Shrink3Error::new(
            REJECT_NONCANONICAL_AST,
            "AST node array must not be empty",
        ));
    }
    match probe_expression_tag(&values[0], "AST node tag")? {
        0 => {
            if values.len() < 2 {
                return Err(Shrink3Error::new(
                    REJECT_NONCANONICAL_AST,
                    "leaf node is missing its leaf id",
                ));
            }
            match probe_expression_tag(&values[1], "leaf id")? {
                0 => {
                    expect_probe_length(values, 3, "scalar_const")?;
                    Ok(Node::ScalarConst(probe_bounded_registry_uint(
                        &values[2],
                        RATIONAL_PARAMETER_COUNT,
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
                        AGGREGATE_MAP_COUNT,
                        "aggregate map id",
                    )?;
                    let scope_id = probe_bounded_registry_uint(
                        &values[3],
                        SCOPE_COUNT,
                        "scope id",
                    )?;
                    let quantity_id = probe_bounded_registry_uint(
                        &values[4],
                        QUANTITY_COUNT,
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
                                CONTEXT_COUNT,
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
                        return Err(Shrink3Error::new(
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
                        CONTEXT_COUNT,
                        "context id",
                    )?))
                }
                5 => {
                    expect_probe_length(values, 3, "task_flag")?;
                    Ok(Node::TaskFlag(probe_bounded_registry_uint(
                        &values[2],
                        TASK_COUNT,
                        "task id",
                    )?))
                }
                6 => Err(Shrink3Error::new(
                    REJECT_NEW_SYMBOL_IN_OLD_DSL,
                    "new symbols are Phase-3B only",
                )),
                leaf_id => Err(Shrink3Error::new(
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
                // ID 4 remains a source-only alias.  Represent it long enough
                // to apply inherited/add tombstone priority; shrink-2 will
                // subsequently emit its frozen noncanonical result.
                4 => BinaryOp::GreaterEqual,
                5 => BinaryOp::SameSign,
                6 => BinaryOp::OppositeSign,
                7 => {
                    return Err(Shrink3Error::new(
                        REJECT_NONCANONICAL_AST,
                        "reserved binary operator ID is not canonical",
                    ))
                }
                _ => unreachable!("bounded binary operator ID"),
            };
            formal_type_checked(Node::Binary {
                op,
                left: Box::new(node_from_probe(&values[2])?),
                right: Box::new(node_from_probe(&values[3])?),
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
                return Err(Shrink3Error::new(
                    REJECT_TYPE_MISMATCH,
                    "approx_equal expects two RationalValue children",
                ));
            }
            let tolerance_index =
                probe_bounded_registry_uint(&values[4], 3, "tolerance index")?;
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
                return Err(Shrink3Error::new(
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
        tag => Err(Shrink3Error::new(
            REJECT_UNKNOWN_EXPRESSION,
            format!("unknown/reserved AST node tag {tag}"),
        )),
    }
}

/// Validate and decode exact shrink-3 canonical AST bytes.
///
/// This function owns an independent structural/type path.  It never calls
/// the shrink-2 canonical decoder before inspecting ID 0, so a valid-shaped
/// foldable or nested `add` cannot be downgraded to `REJECT_NONCANONICAL_AST`.
pub fn decode_shrink3_canonical_ast(bytes: &[u8]) -> Result<CanonicalProgram, Shrink3Error> {
    let mut cursor = 0usize;
    let probe = parse_probe(bytes, &mut cursor, 0)?;
    if cursor != bytes.len() {
        return Err(Shrink3Error::new(
            REJECT_TRAILING_CBOR,
            "trailing bytes after the first CBOR item",
        ));
    }
    validate_strict_cbor(bytes)?;
    let envelope = probe_array(&probe, "CanonicalAstV1 envelope")?;
    expect_probe_length(envelope, 2, "CanonicalAstV1 envelope")?;
    if envelope[0] != ProbeValue::Unsigned(1) {
        return Err(Shrink3Error::new(
            REJECT_UNKNOWN_AST_SCHEMA,
            "unknown canonical AST schema version",
        ));
    }
    let source_node = formal_type_checked(node_from_probe(&envelope[1])?)?;

    // These are distinct full-tree passes.  Their order is part of the
    // shrink-3 acceptance contract and does not depend on AST traversal order.
    reject_removed_aggregate_nodes(&source_node)?;
    reject_removed_parameter_nodes(&source_node)?;
    reject_removed_add_nodes(&source_node)?;

    Ok(decode_shrink2_canonical_ast(bytes)?)
}

pub fn sort_name(sort: Sort) -> &'static str {
    hegel_strict_canonicalizer_shrink2::sort_name(sort)
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
pub struct Shrink3CapacityReplayReport {
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
    pub constant_atom_count: usize,
    pub rational_aggregate_count: usize,
    pub mixed_atom_count: usize,
    pub source_candidate_count: usize,
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

/// Replay the inherited add-free 2,160-source target-free subset through the
/// new admission layer.  Passing this routine never means complete closure.
pub fn replay_shrink3_capacity_subset() -> Result<Shrink3CapacityReplayReport, Shrink3Error> {
    let constant_atoms = capacity_constant_atoms();
    let rational_aggregates = capacity_rational_aggregates();
    let mixed_atoms = capacity_mixed_atoms();
    if constant_atoms.len() != 15 || rational_aggregates.len() != 16 || mixed_atoms.len() != 144 {
        return Err(Shrink3Error::new(
            REJECT_INTERNAL_SHRINK3_REPLAY,
            "shrink-3 survivor subset component count drift",
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
            let parent = canonicalize_shrink2_source_node(source.clone())?;
            match canonicalize_shrink3_source_node(source) {
                Ok(program) => {
                    if program.canonical_cbor != parent.canonical_cbor
                        || program.canonical_ast_hash != parent.canonical_ast_hash
                    {
                        return Err(Shrink3Error::new(
                            REJECT_INTERNAL_SHRINK3_REPLAY,
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
        Shrink3Error::new(
            REJECT_INTERNAL_SHRINK3_REPLAY,
            "unique count exceeds accepted count",
        )
    })?;
    let commitment = capacity_set_commitment(&canonical_set);
    let first = canonical_set.iter().next().ok_or_else(|| {
        Shrink3Error::new(REJECT_INTERNAL_SHRINK3_REPLAY, "survivor set is empty")
    })?;
    let last = canonical_set.iter().next_back().ok_or_else(|| {
        Shrink3Error::new(REJECT_INTERNAL_SHRINK3_REPLAY, "survivor set is empty")
    })?;
    let first_program = decode_shrink3_canonical_ast(first)?;
    let last_program = decode_shrink3_canonical_ast(last)?;
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
        return Err(Shrink3Error::new(
            REJECT_INTERNAL_SHRINK3_REPLAY,
            format!(
                "frozen survivor subset invariant failure: source={source_count}, accepted={accepted_count}, unique={accepted_unique_count}, rejected={rejected_count}, collapsed={rewrite_collapsed_count}, commitment={commitment}, first={first_hex}, last={last_hex}"
            ),
        ));
    }

    Ok(Shrink3CapacityReplayReport {
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
        constant_atom_count: constant_atoms.len(),
        rational_aggregate_count: rational_aggregates.len(),
        mixed_atom_count: mixed_atoms.len(),
        source_candidate_count: source_count,
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
        subset_status: "SURVIVOR_SUBSET_ONLY_NOT_COMPLETE",
        executed_closure_status: "NOT_RUN",
        complete_closure_enumerated: false,
        interpreted_as_complete_closure: false,
        formal_roots: None,
        target_or_split_modules_loaded: false,
    })
}

#[derive(Debug, Clone, Serialize)]
pub struct Shrink3GoldenReplayReport {
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
    pub vector_count: usize,
    pub passed_count: usize,
    pub surviving_identity_checks: usize,
    pub source_add_rejection_checks: usize,
    pub source_priority_checks: usize,
    pub formal_add_rejection_checks: usize,
    pub formal_priority_checks: usize,
    pub formal_shape_priority_checks: usize,
    pub formal_alias_or_reserved_checks: usize,
    pub execution_state: &'static str,
    pub closure_executed: bool,
    pub formal_roots_generated: bool,
    pub formal_roots: Option<String>,
    pub target_or_split_modules_loaded: bool,
}

fn golden_failure(message: impl Into<String>) -> Shrink3Error {
    Shrink3Error::new(REJECT_INTERNAL_SHRINK3_REPLAY, message)
}

fn expect_source_error(value: &Value, expected: &str, label: &str) -> Result<(), Shrink3Error> {
    match canonicalize_shrink3_source_json(value) {
        Ok(_) => Err(golden_failure(format!("{label}: unexpectedly accepted"))),
        Err(error) if error.code == expected => Ok(()),
        Err(error) => Err(golden_failure(format!(
            "{label}: expected {expected}, got {}",
            error.code
        ))),
    }
}

fn formal_bytes(value: Value) -> Result<Vec<u8>, Shrink3Error> {
    Ok(encode_strict_cbor_json(&value)?)
}

fn expect_formal_error(value: Value, expected: &str, label: &str) -> Result<(), Shrink3Error> {
    let bytes = formal_bytes(value)?;
    match decode_shrink3_canonical_ast(&bytes) {
        Ok(_) => Err(golden_failure(format!("{label}: unexpectedly accepted"))),
        Err(error) if error.code == expected => Ok(()),
        Err(error) => Err(golden_failure(format!(
            "{label}: expected {expected}, got {}",
            error.code
        ))),
    }
}

fn check_survivor_identity(source: &Value, label: &str) -> Result<(), Shrink3Error> {
    let parent = hegel_strict_canonicalizer_shrink2::canonicalize_shrink2_source_json(source)?;
    let child = canonicalize_shrink3_source_json(source)?;
    if child.canonical_cbor != parent.canonical_cbor
        || child.canonical_ast_hash != parent.canonical_ast_hash
    {
        return Err(golden_failure(format!(
            "{label}: surviving source changed canonical bytes/hash"
        )));
    }
    let parent_formal = decode_shrink2_canonical_ast(&parent.canonical_cbor)?;
    let child_formal = decode_shrink3_canonical_ast(&parent.canonical_cbor)?;
    if child_formal.canonical_cbor != parent_formal.canonical_cbor
        || child_formal.canonical_ast_hash != parent_formal.canonical_ast_hash
    {
        return Err(golden_failure(format!(
            "{label}: surviving formal program changed canonical bytes/hash"
        )));
    }
    Ok(())
}

fn nonconstant_binary(name: &str) -> Value {
    json!([
        name,
        ["bit_to_scalar", ["bit_at", 0]],
        ["scalar_const", 1, 1]
    ])
}

fn formal_shape_priority_cases() -> [Value; 6] {
    [
        json!([1, [4, []]]),
        json!([1, [4, [[2, 0, [0, 0, 1], [0, 0, 5]]]]]),
        json!([
            1,
            [
                4,
                [
                    [0, 4, 0],
                    [0, 4, 1],
                    [0, 4, 2],
                    [2, 0, [0, 0, 1], [0, 0, 5]]
                ]
            ]
        ]),
        json!([
            1,
            [
                2,
                0,
                [
                    0,
                    3,
                    2,
                    0,
                    0,
                    [[0, false], [1, false], [2, false]]
                ],
                [0, 0, 0]
            ]
        ]),
        json!([
            1,
            [
                2,
                0,
                [0, 3, 2, 0, 0, [[1, false], [0, false]]],
                [0, 0, 0]
            ]
        ]),
        json!([
            1,
            [
                2,
                0,
                [0, 3, 2, 0, 0, [[0, false], [0, true]]],
                [0, 0, 0]
            ]
        ]),
    ]
}

/// Replay the exact shared Python/Rust 36-vector shrink-3 profile.
pub fn replay_shrink3_golden_vectors() -> Result<Shrink3GoldenReplayReport, Shrink3Error> {
    let mut vector_count = 0usize;
    let mut surviving_identity_checks = 0usize;
    let mut source_add_rejection_checks = 0usize;
    let mut source_priority_checks = 0usize;
    let mut formal_add_rejection_checks = 0usize;
    let mut formal_priority_checks = 0usize;
    let mut formal_shape_priority_checks = 0usize;
    let mut formal_alias_or_reserved_checks = 0usize;

    let surviving_sources = [
        json!(["scalar_const", 1]),
        json!(["scalar_const", 3]),
        json!(["scalar_const", 5]),
        nonconstant_binary("difference"),
        json!(["difference", ["scalar_const", 1], ["scalar_const", 5]]),
        json!([
            "greater_equal",
            ["bit_to_scalar", ["bit_at", 0]],
            ["scalar_const", 1]
        ]),
        json!([
            "aggregate",
            "signed_balance_v1",
            "scope_all_observed_v1",
            "q0",
            []
        ]),
        json!([
            "equal_exact",
            ["bit_to_scalar", ["bit_at", 0]],
            ["scalar_const", 1]
        ]),
    ];
    for (index, source) in surviving_sources.iter().enumerate() {
        check_survivor_identity(source, &format!("survivor vector {index}"))?;
        vector_count += 1;
        surviving_identity_checks += 1;
    }

    let source_add_cases = [
        nonconstant_binary("add"),
        json!(["add", ["scalar_const", 1], ["scalar_const", 5]]),
        json!(["add", ["scalar_const", 5], ["scalar_const", 5]]),
        json!([
            "difference",
            nonconstant_binary("add"),
            ["scalar_const", 3]
        ]),
    ];
    for (index, source) in source_add_cases.iter().enumerate() {
        expect_source_error(
            source,
            REJECT_REMOVED_BINARY_OPERATOR,
            &format!("source add vector {index}"),
        )?;
        vector_count += 1;
        source_add_rejection_checks += 1;
    }

    let source_priority_cases = [
        (
            json!(["add", ["scalar_const", 1]]),
            REJECT_MALFORMED_SOURCE_AST,
        ),
        (
            json!(["add", ["bit_at", 0], ["scalar_const", 1]]),
            REJECT_IMPLICIT_COERCION,
        ),
        (
            json!([
                "add",
                [
                    "aggregate",
                    "mean_v1",
                    "scope_all_observed_v1",
                    "q0",
                    []
                ],
                ["scalar_const", -2, 1]
            ]),
            REJECT_REMOVED_AGGREGATE_MAP,
        ),
        (
            json!(["add", ["scalar_const", -2, 1], ["scalar_const", 1]]),
            REJECT_REMOVED_RATIONAL_PARAMETER,
        ),
        (
            json!(["unknown_outer", nonconstant_binary("add")]),
            REJECT_UNKNOWN_EXPRESSION,
        ),
        (
            json!([7, ["scalar_const", 1], ["scalar_const", 5]]),
            REJECT_MALFORMED_SOURCE_AST,
        ),
    ];
    for (index, (source, expected)) in source_priority_cases.iter().enumerate() {
        expect_source_error(
            source,
            expected,
            &format!("source priority vector {index}"),
        )?;
        vector_count += 1;
        source_priority_checks += 1;
    }

    let parent_add_source = nonconstant_binary("add");
    let parent_add =
        hegel_strict_canonicalizer_shrink2::canonicalize_shrink2_source_json(&parent_add_source)?;
    match decode_shrink3_canonical_ast(&parent_add.canonical_cbor) {
        Err(error) if error.code == REJECT_REMOVED_BINARY_OPERATOR => {}
        Err(error) => {
            return Err(golden_failure(format!(
                "formal canonical add: expected {REJECT_REMOVED_BINARY_OPERATOR}, got {}",
                error.code
            )))
        }
        Ok(_) => return Err(golden_failure("formal canonical add unexpectedly accepted")),
    }
    vector_count += 1;
    formal_add_rejection_checks += 1;

    let remaining_formal_add_cases = [
        json!([1, [2, 0, [0, 0, 1], [0, 0, 5]]]),
        json!([
            1,
            [
                2,
                1,
                [2, 0, [1, 0, [0, 1, 0]], [0, 0, 5]],
                [0, 0, 3]
            ]
        ]),
    ];
    for (index, formal) in remaining_formal_add_cases.into_iter().enumerate() {
        expect_formal_error(
            formal,
            REJECT_REMOVED_BINARY_OPERATOR,
            &format!("formal add vector {}", index + 1),
        )?;
        vector_count += 1;
        formal_add_rejection_checks += 1;
    }

    let formal_priority_cases = [
        (
            json!([1, [2, 0, [0, 0, 1]]]),
            REJECT_NONCANONICAL_AST,
        ),
        (
            json!([1, [2, 0, [0, 1, 0], [0, 0, 1]]]),
            REJECT_TYPE_MISMATCH,
        ),
        (
            json!([1, [2, 0, [0, 3, 2, 0, 0, []], [0, 0, 0]]]),
            REJECT_REMOVED_AGGREGATE_MAP,
        ),
        (
            json!([1, [2, 0, [0, 0, 0], [0, 0, 1]]]),
            REJECT_REMOVED_RATIONAL_PARAMETER,
        ),
        (
            json!([
                1,
                [
                    2,
                    4,
                    [2, 0, [1, 0, [0, 1, 0]], [0, 0, 1]],
                    [0, 0, 1]
                ]
            ]),
            REJECT_REMOVED_BINARY_OPERATOR,
        ),
        (
            json!([1, [0, 0, 1, [2, 0, [0, 0, 1], [0, 0, 5]]]]),
            REJECT_NONCANONICAL_AST,
        ),
    ];
    for (index, (formal, expected)) in formal_priority_cases.into_iter().enumerate() {
        expect_formal_error(
            formal,
            expected,
            &format!("formal priority vector {index}"),
        )?;
        vector_count += 1;
        formal_priority_checks += 1;
    }

    for (index, formal) in formal_shape_priority_cases().into_iter().enumerate() {
        expect_formal_error(
            formal,
            REJECT_NONCANONICAL_AST,
            &format!("formal shape priority vector {index}"),
        )?;
        vector_count += 1;
        formal_shape_priority_checks += 1;
    }

    for (label, operator_id, expected) in [
        ("formal source-only ID4", 4, REJECT_NONCANONICAL_AST),
        ("formal reserved ID7", 7, REJECT_NONCANONICAL_AST),
        (
            "formal out-of-range BinaryOperatorId8",
            8,
            REJECT_REGISTRY_INDEX_OUT_OF_RANGE,
        ),
    ] {
        expect_formal_error(
            json!([1, [2, operator_id, [0, 0, 1], [0, 0, 5]]]),
            expected,
            label,
        )?;
        vector_count += 1;
        formal_alias_or_reserved_checks += 1;
    }

    if vector_count != 36 {
        return Err(golden_failure(format!(
            "golden vector count drift: {vector_count}"
        )));
    }

    Ok(Shrink3GoldenReplayReport {
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
        vector_count,
        passed_count: vector_count,
        surviving_identity_checks,
        source_add_rejection_checks,
        source_priority_checks,
        formal_add_rejection_checks,
        formal_priority_checks,
        formal_shape_priority_checks,
        formal_alias_or_reserved_checks,
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
    fn golden_replay_covers_frozen_shrink3_contract() {
        let report = replay_shrink3_golden_vectors().unwrap();
        assert_eq!(report.vector_count, 36);
        assert_eq!(report.passed_count, report.vector_count);
        assert_eq!(report.surviving_identity_checks, 8);
        assert_eq!(report.source_add_rejection_checks, 4);
        assert_eq!(report.source_priority_checks, 6);
        assert_eq!(report.formal_add_rejection_checks, 3);
        assert_eq!(report.formal_priority_checks, 6);
        assert_eq!(report.formal_shape_priority_checks, 6);
        assert_eq!(report.formal_alias_or_reserved_checks, 3);
    }

    #[test]
    fn survivor_capacity_commitment_is_unchanged_and_not_complete() {
        let report = replay_shrink3_capacity_subset().unwrap();
        assert_eq!(report.source_candidate_count, 2_160);
        assert_eq!(report.accepted_unique_count, 2_160);
        assert_eq!(
            report.accepted_set_commitment,
            EXPECTED_SURVIVOR_ACCEPTED_SET_COMMITMENT
        );
        assert_eq!(report.subset_status, "SURVIVOR_SUBSET_ONLY_NOT_COMPLETE");
        assert!(!report.complete_closure_enumerated);
        assert!(!report.interpreted_as_complete_closure);
    }

    #[test]
    fn foldable_formal_add_is_removed_before_noncanonical() {
        let bytes = formal_bytes(json!([1, [2, 0, [0, 0, 1], [0, 0, 5]]])).unwrap();
        let error = decode_shrink3_canonical_ast(&bytes).unwrap_err();
        assert_eq!(error.code, REJECT_REMOVED_BINARY_OPERATOR);
    }

    #[test]
    fn difference_survivor_has_exact_parent_identity() {
        let source = json!(["difference", ["scalar_const", 5], ["scalar_const", 1]]);
        let parent =
            hegel_strict_canonicalizer_shrink2::canonicalize_shrink2_source_json(&source).unwrap();
        let child = canonicalize_shrink3_source_json(&source).unwrap();
        assert_eq!(child.root_operator_id, 0x0201);
        assert_eq!(child.canonical_cbor, parent.canonical_cbor);
        assert_eq!(child.canonical_ast_hash, parent.canonical_ast_hash);
    }

    #[test]
    fn oversized_source_add_is_removed_before_whole_ast_limits() {
        let source = json!([
            "absolute",
            [
                "absolute",
                [
                    "absolute",
                    [
                        "absolute",
                        [
                            "add",
                            ["bit_to_scalar", ["bit_at", 0]],
                            ["scalar_const", 5]
                        ]
                    ]
                ]
            ]
        ]);
        let error = canonicalize_shrink3_source_json(&source).unwrap_err();
        assert_eq!(error.code, REJECT_REMOVED_BINARY_OPERATOR);
    }

    #[test]
    fn arbitrary_width_registry_error_precedes_source_add() {
        let source: Value = serde_json::from_str(
            r#"["add",["scalar_const",184467440737095516160],["scalar_const",1]]"#,
        )
        .unwrap();
        let error = canonicalize_shrink3_source_json(&source).unwrap_err();
        assert_eq!(error.code, REJECT_REGISTRY_INDEX_OUT_OF_RANGE);
    }

    #[test]
    fn arbitrary_width_rational_tombstone_precedes_source_add() {
        let source: Value = serde_json::from_str(
            r#"["add",["scalar_const",-200000000000000000000,100000000000000000000],["scalar_const",1]]"#,
        )
        .unwrap();
        let error = canonicalize_shrink3_source_json(&source).unwrap_err();
        assert_eq!(error.code, REJECT_REMOVED_RATIONAL_PARAMETER);
    }

    #[test]
    fn formal_shape_priority_vectors_are_noncanonical_before_hidden_tombstones() {
        for (index, formal) in formal_shape_priority_cases().into_iter().enumerate() {
            let compact = serde_json::to_vec(&formal).unwrap();
            let compact_hash = format!("{:x}", Sha256::digest(&compact));
            assert_eq!(
                compact_hash,
                EXPECTED_FORMAL_SHAPE_COMPACT_JSON_SHA256[index]
            );
            let bytes = formal_bytes(formal).unwrap();
            let error = decode_shrink3_canonical_ast(&bytes).unwrap_err();
            assert_eq!(error.code, REJECT_NONCANONICAL_AST);
        }
    }
}
