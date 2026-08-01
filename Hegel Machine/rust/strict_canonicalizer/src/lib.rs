//! Independent Rust implementation of the Hegel Machine v1.0.2 strict
//! canonical AST and deterministic-CBOR acceptance gate.
//!
//! The normative identity of a program is the exact CBOR encoding of
//! `[1, RootNode]`. JSON is accepted only as a source/golden-vector transport.

use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt;
use std::fs;
use std::path::Path;

pub const CBOR_PROFILE_ID: &str = "hegel-cbor-det-v1";
pub const AST_SCHEMA_ID: &str = "hegel-canonical-ast-v1";
pub const REPLAY_SCHEMA_VERSION: &str = "hegel-strict-canonicalizer-replay/1";

pub const REJECT_MALFORMED_SOURCE_AST: &str = "REJECT_MALFORMED_SOURCE_AST";
pub const REJECT_UNKNOWN_EXPRESSION: &str = "REJECT_UNKNOWN_EXPRESSION";
pub const REJECT_REGISTRY_INDEX_OUT_OF_RANGE: &str = "REJECT_REGISTRY_INDEX_OUT_OF_RANGE";
pub const REJECT_TYPE_MISMATCH: &str = "REJECT_TYPE_MISMATCH";
pub const REJECT_IMPLICIT_COERCION: &str = "REJECT_IMPLICIT_COERCION";
pub const REJECT_NONCANONICAL_SCOPE_ALIAS: &str = "REJECT_NONCANONICAL_SCOPE_ALIAS";
pub const REJECT_DUPLICATE_SCOPE_CONTEXT: &str = "REJECT_DUPLICATE_SCOPE_CONTEXT";
pub const REJECT_NEW_SYMBOL_IN_OLD_DSL: &str = "REJECT_NEW_SYMBOL_IN_OLD_DSL";
pub const REJECT_STRUCTURAL_LIMIT: &str = "REJECT_STRUCTURAL_LIMIT";
pub const REJECT_NONCANONICAL_CBOR: &str = "REJECT_NONCANONICAL_CBOR";
pub const REJECT_CBOR_FLOAT: &str = "REJECT_CBOR_FLOAT";
pub const REJECT_CBOR_TEXT: &str = "REJECT_CBOR_TEXT";
pub const REJECT_CBOR_MAP: &str = "REJECT_CBOR_MAP";
pub const REJECT_CBOR_TAG: &str = "REJECT_CBOR_TAG";
pub const REJECT_INDEFINITE_CBOR: &str = "REJECT_INDEFINITE_CBOR";
pub const REJECT_CBOR_NESTING: &str = "REJECT_CBOR_NESTING";
pub const REJECT_TRAILING_CBOR: &str = "REJECT_TRAILING_CBOR";
pub const REJECT_NONCANONICAL_AST: &str = "REJECT_NONCANONICAL_AST";
pub const REJECT_INTERNAL_CANONICALIZATION: &str = "REJECT_INTERNAL_CANONICALIZATION";

const MAX_TOTAL_AST_DEPTH: u32 = 4;
const MAX_TOTAL_NODE_COUNT: u32 = 7;
const MAX_TOP_LEVEL_CLAUSES: usize = 3;
const MAX_DISTINCT_BIT_SLOTS: usize = 4;
const MAX_AGGREGATE_LEAVES: u32 = 1;
const MAX_SCOPE_CLAUSES: usize = 2;
const MAX_FITTED_SCALAR_PARAMETER_OCCURRENCES: u32 = 3;
const MAX_CBOR_NESTING: usize = 64;

const RATIONAL_PARAMETERS: [(i64, i64); 7] =
    [(-2, 1), (-1, 1), (-1, 2), (0, 1), (1, 2), (1, 1), (2, 1)];
const TOLERANCES: [(i64, i64); 3] = [(0, 1), (1, 4), (1, 2)];
const AGGREGATE_MAP_IDS: [&str; 6] = [
    "sum_v1",
    "count_nonzero_v1",
    "mean_v1",
    "min_v1",
    "max_v1",
    "signed_balance_v1",
];
const SCOPE_IDS: [&str; 4] = [
    "scope_all_observed_v1",
    "scope_primary_only_v1",
    "scope_boundary_only_v1",
    "control_volume_all_observed_v1",
];
const QUANTITY_IDS: [&str; 2] = ["q0", "q1"];
const CONTEXT_IDS: [&str; 4] = ["c0", "c1", "c2", "c3"];
const TASK_IDS: [&str; 2] = ["t0", "t1"];
const DEPRECATED_SCOPE_ALIAS: &str = "control_volume_primary_only_v1";
const ZERO_PARAMETER_INDEX: u64 = 3;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalError {
    pub code: &'static str,
    pub message: String,
}

impl CanonicalError {
    fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

impl fmt::Display for CanonicalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for CanonicalError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Sort {
    Bool,
    Bit,
    Sign,
    BoundedInt,
    RationalValue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    BitToScalar,
    IntToScalar,
    Absolute,
    Sign,
}

impl UnaryOp {
    fn id(self) -> u64 {
        match self {
            Self::BitToScalar => 0,
            Self::IntToScalar => 1,
            Self::Absolute => 2,
            Self::Sign => 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Difference,
    EqualExact,
    LessEqual,
    GreaterEqual,
    SameSign,
    OppositeSign,
}

impl BinaryOp {
    fn id(self) -> u64 {
        match self {
            Self::Add => 0,
            Self::Difference => 1,
            Self::EqualExact => 2,
            Self::LessEqual => 3,
            Self::GreaterEqual => 4,
            Self::SameSign => 5,
            Self::OppositeSign => 6,
        }
    }

    fn is_commutative(self) -> bool {
        matches!(
            self,
            Self::Add | Self::EqualExact | Self::SameSign | Self::OppositeSign
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Node {
    ScalarConst(u64),
    BitAt(u64),
    SetSize,
    Aggregate {
        map_id: u64,
        scope_id: u64,
        quantity_id: u64,
        scope_extension: Vec<(u64, bool)>,
    },
    ContextFlag(u64),
    TaskFlag(u64),
    NewSymbolCall(u64),
    Unary {
        op: UnaryOp,
        child: Box<Node>,
    },
    Binary {
        op: BinaryOp,
        left: Box<Node>,
        right: Box<Node>,
    },
    ApproxEqual {
        left: Box<Node>,
        right: Box<Node>,
        tolerance_index: u64,
    },
    And(Vec<Node>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Rational {
    numerator: i64,
    denominator: i64,
}

impl Rational {
    fn new(numerator: i64, denominator: i64) -> Result<Self, CanonicalError> {
        if denominator == 0 {
            return Err(CanonicalError::new(
                REJECT_MALFORMED_SOURCE_AST,
                "rational denominator must not be zero",
            ));
        }
        let mut numerator = numerator;
        let mut denominator = denominator;
        if denominator < 0 {
            numerator = numerator.checked_neg().ok_or_else(|| {
                CanonicalError::new(
                    REJECT_MALFORMED_SOURCE_AST,
                    "rational numerator overflow while normalizing sign",
                )
            })?;
            denominator = denominator.checked_neg().ok_or_else(|| {
                CanonicalError::new(
                    REJECT_MALFORMED_SOURCE_AST,
                    "rational denominator overflow while normalizing sign",
                )
            })?;
        }
        let divisor = gcd_i64(numerator, denominator);
        Ok(Self {
            numerator: numerator / divisor,
            denominator: denominator / divisor,
        })
    }

    fn add(&self, other: &Self) -> Option<Self> {
        let left = self.numerator.checked_mul(other.denominator)?;
        let right = other.numerator.checked_mul(self.denominator)?;
        let numerator = left.checked_add(right)?;
        let denominator = self.denominator.checked_mul(other.denominator)?;
        Self::new(numerator, denominator).ok()
    }

    fn difference(&self, other: &Self) -> Option<Self> {
        let left = self.numerator.checked_mul(other.denominator)?;
        let right = other.numerator.checked_mul(self.denominator)?;
        let numerator = left.checked_sub(right)?;
        let denominator = self.denominator.checked_mul(other.denominator)?;
        Self::new(numerator, denominator).ok()
    }

    fn absolute(&self) -> Option<Self> {
        Self::new(self.numerator.checked_abs()?, self.denominator).ok()
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
    if left == 0 {
        1
    } else {
        left as i64
    }
}

fn rational_parameter(index: u64) -> Result<Rational, CanonicalError> {
    let (numerator, denominator) = RATIONAL_PARAMETERS
        .get(index as usize)
        .copied()
        .ok_or_else(|| {
            CanonicalError::new(
                REJECT_REGISTRY_INDEX_OUT_OF_RANGE,
                format!("rational parameter index {index} is outside 0..7"),
            )
        })?;
    Rational::new(numerator, denominator)
}

fn rational_parameter_index(value: &Rational) -> Option<u64> {
    RATIONAL_PARAMETERS
        .iter()
        .position(|pair| *pair == (value.numerator, value.denominator))
        .map(|index| index as u64)
}

fn tolerance_index(value: &Rational) -> Option<u64> {
    TOLERANCES
        .iter()
        .position(|pair| *pair == (value.numerator, value.denominator))
        .map(|index| index as u64)
}

fn json_array<'a>(value: &'a Value, context: &str) -> Result<&'a [Value], CanonicalError> {
    value.as_array().map(Vec::as_slice).ok_or_else(|| {
        CanonicalError::new(
            REJECT_MALFORMED_SOURCE_AST,
            format!("{context} must be a JSON array"),
        )
    })
}

fn json_name<'a>(value: &'a Value, context: &str) -> Result<&'a str, CanonicalError> {
    value.as_str().ok_or_else(|| {
        CanonicalError::new(
            REJECT_MALFORMED_SOURCE_AST,
            format!("{context} must be a JSON string"),
        )
    })
}

fn json_i64(value: &Value, context: &str) -> Result<i64, CanonicalError> {
    value.as_i64().ok_or_else(|| {
        CanonicalError::new(
            REJECT_MALFORMED_SOURCE_AST,
            format!("{context} must be an exact JSON integer"),
        )
    })
}

fn json_u64(value: &Value, context: &str) -> Result<u64, CanonicalError> {
    value.as_u64().ok_or_else(|| {
        CanonicalError::new(
            REJECT_MALFORMED_SOURCE_AST,
            format!("{context} must be a nonnegative JSON integer"),
        )
    })
}

fn registry_index(value: &Value, names: &[&str], context: &str) -> Result<u64, CanonicalError> {
    if let Some(index) = value.as_u64() {
        if index < names.len() as u64 {
            return Ok(index);
        }
        return Err(CanonicalError::new(
            REJECT_REGISTRY_INDEX_OUT_OF_RANGE,
            format!("{context} index {index} is outside 0..{}", names.len()),
        ));
    }
    let name = json_name(value, context)?;
    names
        .iter()
        .position(|candidate| *candidate == name)
        .map(|index| index as u64)
        .ok_or_else(|| {
            CanonicalError::new(
                REJECT_REGISTRY_INDEX_OUT_OF_RANGE,
                format!("unknown frozen {context} value {name:?}"),
            )
        })
}

fn parse_rational_index(
    values: &[Value],
    start: usize,
    tolerance: bool,
) -> Result<u64, CanonicalError> {
    if values.len() == start + 1 {
        let index = json_u64(&values[start], "parameter index")?;
        let bound = if tolerance {
            TOLERANCES.len()
        } else {
            RATIONAL_PARAMETERS.len()
        };
        if index >= bound as u64 {
            return Err(CanonicalError::new(
                REJECT_REGISTRY_INDEX_OUT_OF_RANGE,
                format!("parameter index {index} is outside 0..{bound}"),
            ));
        }
        return Ok(index);
    }
    if values.len() == start + 2 {
        let rational = Rational::new(
            json_i64(&values[start], "rational numerator")?,
            json_i64(&values[start + 1], "rational denominator")?,
        )?;
        let index = if tolerance {
            tolerance_index(&rational)
        } else {
            rational_parameter_index(&rational)
        };
        return index.ok_or_else(|| {
            CanonicalError::new(
                REJECT_REGISTRY_INDEX_OUT_OF_RANGE,
                format!(
                    "rational {}/{} is outside the frozen {} grid",
                    rational.numerator,
                    rational.denominator,
                    if tolerance { "tolerance" } else { "parameter" }
                ),
            )
        });
    }
    Err(CanonicalError::new(
        REJECT_MALFORMED_SOURCE_AST,
        "a rational parameter needs an index or numerator/denominator pair",
    ))
}

fn parse_scope_extension(value: &Value) -> Result<Vec<(u64, bool)>, CanonicalError> {
    let clauses = json_array(value, "scope extension")?;
    if clauses.len() > MAX_SCOPE_CLAUSES {
        return Err(CanonicalError::new(
            REJECT_STRUCTURAL_LIMIT,
            format!(
                "scope extension has {} clauses; maximum is {MAX_SCOPE_CLAUSES}",
                clauses.len()
            ),
        ));
    }
    let mut result = Vec::with_capacity(clauses.len());
    for clause in clauses {
        let clause = json_array(clause, "scope clause")?;
        if clause.len() != 2 {
            return Err(CanonicalError::new(
                REJECT_MALFORMED_SOURCE_AST,
                "scope clause must be [context_id, expected_bool]",
            ));
        }
        let context_id = registry_index(&clause[0], &CONTEXT_IDS, "context id")?;
        let expected = clause[1].as_bool().ok_or_else(|| {
            CanonicalError::new(
                REJECT_MALFORMED_SOURCE_AST,
                "scope-clause expectation must be a JSON boolean",
            )
        })?;
        result.push((context_id, expected));
    }
    result.sort_unstable_by_key(|clause| clause.0);
    if result.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Err(CanonicalError::new(
            REJECT_DUPLICATE_SCOPE_CONTEXT,
            "scope extension contains a duplicate context id",
        ));
    }
    Ok(result)
}

fn parse_bit_slot(value: &Value) -> Result<u64, CanonicalError> {
    if let Some(index) = value.as_u64() {
        if index < 8 {
            return Ok(index);
        }
    }
    Err(CanonicalError::new(
        REJECT_REGISTRY_INDEX_OUT_OF_RANGE,
        "bit_at entity slot must be a numeric index in 0..8",
    ))
}

fn looks_like_source_node(value: &Value) -> bool {
    value
        .as_array()
        .and_then(|items| items.first())
        .and_then(Value::as_str)
        .is_some()
}

/// Parse the frozen named-list source vocabulary. This is not a formal JSON
/// identity; accepted values are immediately converted to the numeric AST.
pub fn parse_source_ast(value: &Value) -> Result<Node, CanonicalError> {
    let items = json_array(value, "source AST node")?;
    if items.is_empty() {
        return Err(CanonicalError::new(
            REJECT_MALFORMED_SOURCE_AST,
            "source AST node must not be empty",
        ));
    }
    let name = json_name(&items[0], "expression name")?;
    match name {
        "scalar_const" => Ok(Node::ScalarConst(parse_rational_index(items, 1, false)?)),
        "bit_at" => {
            if items.len() != 2 {
                return Err(CanonicalError::new(
                    REJECT_MALFORMED_SOURCE_AST,
                    "bit_at requires exactly one entity-slot argument",
                ));
            }
            Ok(Node::BitAt(parse_bit_slot(&items[1])?))
        }
        "set_size" => {
            if items.len() != 1 {
                return Err(CanonicalError::new(
                    REJECT_MALFORMED_SOURCE_AST,
                    "set_size takes no arguments",
                ));
            }
            Ok(Node::SetSize)
        }
        "aggregate" => {
            if items.len() != 5 {
                return Err(CanonicalError::new(
                    REJECT_MALFORMED_SOURCE_AST,
                    "aggregate requires map, scope, quantity, and a scope-extension array",
                ));
            }
            if items[2].as_str() == Some(DEPRECATED_SCOPE_ALIAS) {
                return Err(CanonicalError::new(
                    REJECT_NONCANONICAL_SCOPE_ALIAS,
                    format!("deprecated scope alias {DEPRECATED_SCOPE_ALIAS:?} is migration-only"),
                ));
            }
            let scope_extension = parse_scope_extension(&items[4])?;
            Ok(Node::Aggregate {
                map_id: registry_index(&items[1], &AGGREGATE_MAP_IDS, "aggregate map id")?,
                scope_id: registry_index(&items[2], &SCOPE_IDS, "scope id")?,
                quantity_id: registry_index(&items[3], &QUANTITY_IDS, "quantity id")?,
                scope_extension,
            })
        }
        "context_flag" => {
            if items.len() != 2 {
                return Err(CanonicalError::new(
                    REJECT_MALFORMED_SOURCE_AST,
                    "context_flag requires one context id",
                ));
            }
            Ok(Node::ContextFlag(registry_index(
                &items[1],
                &CONTEXT_IDS,
                "context id",
            )?))
        }
        "task_flag" => {
            if items.len() != 2 {
                return Err(CanonicalError::new(
                    REJECT_MALFORMED_SOURCE_AST,
                    "task_flag requires one task id",
                ));
            }
            Ok(Node::TaskFlag(registry_index(
                &items[1], &TASK_IDS, "task id",
            )?))
        }
        "new_symbol_call" => Err(CanonicalError::new(
            REJECT_NEW_SYMBOL_IN_OLD_DSL,
            "new_symbol_call is forbidden in the old DSL canonicalizer",
        )),
        "bit_to_scalar" | "int_to_scalar" | "absolute" | "sign" => {
            if items.len() != 2 {
                return Err(CanonicalError::new(
                    REJECT_MALFORMED_SOURCE_AST,
                    format!("{name} requires exactly one child"),
                ));
            }
            let op = match name {
                "bit_to_scalar" => UnaryOp::BitToScalar,
                "int_to_scalar" => UnaryOp::IntToScalar,
                "absolute" => UnaryOp::Absolute,
                "sign" => UnaryOp::Sign,
                _ => unreachable!(),
            };
            Ok(Node::Unary {
                op,
                child: Box::new(parse_source_ast(&items[1])?),
            })
        }
        "add" | "difference" | "equal_exact" | "less_equal" | "greater_equal" | "same_sign"
        | "opposite_sign" => {
            if items.len() != 3 {
                return Err(CanonicalError::new(
                    REJECT_MALFORMED_SOURCE_AST,
                    format!("{name} requires exactly two children"),
                ));
            }
            let op = match name {
                "add" => BinaryOp::Add,
                "difference" => BinaryOp::Difference,
                "equal_exact" => BinaryOp::EqualExact,
                "less_equal" => BinaryOp::LessEqual,
                "greater_equal" => BinaryOp::GreaterEqual,
                "same_sign" => BinaryOp::SameSign,
                "opposite_sign" => BinaryOp::OppositeSign,
                _ => unreachable!(),
            };
            Ok(Node::Binary {
                op,
                left: Box::new(parse_source_ast(&items[1])?),
                right: Box::new(parse_source_ast(&items[2])?),
            })
        }
        "approx_equal" => {
            if items.len() != 4 && items.len() != 5 {
                return Err(CanonicalError::new(
                    REJECT_MALFORMED_SOURCE_AST,
                    "approx_equal requires two children plus tolerance index or rational pair",
                ));
            }
            let tolerance_index = parse_rational_index(items, 3, true)?;
            Ok(Node::ApproxEqual {
                left: Box::new(parse_source_ast(&items[1])?),
                right: Box::new(parse_source_ast(&items[2])?),
                tolerance_index,
            })
        }
        "top_level_AND" => {
            if items.len() < 2 {
                return Err(CanonicalError::new(
                    REJECT_MALFORMED_SOURCE_AST,
                    "top_level_AND requires at least one atom",
                ));
            }
            let atom_values: Vec<&Value> = if items.len() == 2 && !looks_like_source_node(&items[1])
            {
                json_array(&items[1], "top_level_AND atom list")?
                    .iter()
                    .collect()
            } else {
                items[1..].iter().collect()
            };
            if atom_values.is_empty() {
                return Err(CanonicalError::new(
                    REJECT_MALFORMED_SOURCE_AST,
                    "top_level_AND must not have zero atoms",
                ));
            }
            let atoms = atom_values
                .into_iter()
                .map(parse_source_ast)
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Node::And(atoms))
        }
        _ => Err(CanonicalError::new(
            REJECT_UNKNOWN_EXPRESSION,
            format!("unknown old-DSL expression {name:?}"),
        )),
    }
}

fn rational_expected(actual: Sort, context: &str) -> Result<(), CanonicalError> {
    if actual == Sort::RationalValue {
        Ok(())
    } else if actual == Sort::Bit {
        Err(CanonicalError::new(
            REJECT_IMPLICIT_COERCION,
            format!("{context} requires explicit bit_to_scalar(Bit)"),
        ))
    } else {
        Err(CanonicalError::new(
            REJECT_TYPE_MISMATCH,
            format!("{context} requires RationalValue, got {actual:?}"),
        ))
    }
}

/// Recompute the exact old-DSL type of a source or canonical node.
pub fn type_check(node: &Node) -> Result<Sort, CanonicalError> {
    match node {
        Node::ScalarConst(index) => {
            rational_parameter(*index)?;
            Ok(Sort::RationalValue)
        }
        Node::BitAt(index) => {
            if *index >= 8 {
                return Err(CanonicalError::new(
                    REJECT_REGISTRY_INDEX_OUT_OF_RANGE,
                    format!("bit slot {index} is outside 0..8"),
                ));
            }
            Ok(Sort::Bit)
        }
        Node::SetSize => Ok(Sort::BoundedInt),
        Node::Aggregate {
            map_id,
            scope_id,
            quantity_id,
            scope_extension,
        } => {
            if *map_id >= AGGREGATE_MAP_IDS.len() as u64
                || *scope_id >= SCOPE_IDS.len() as u64
                || *quantity_id >= QUANTITY_IDS.len() as u64
            {
                return Err(CanonicalError::new(
                    REJECT_REGISTRY_INDEX_OUT_OF_RANGE,
                    "aggregate contains an out-of-range registry index",
                ));
            }
            if scope_extension.len() > MAX_SCOPE_CLAUSES {
                return Err(CanonicalError::new(
                    REJECT_STRUCTURAL_LIMIT,
                    "aggregate scope extension exceeds two clauses",
                ));
            }
            let mut previous = None;
            for (context_id, _) in scope_extension {
                if *context_id >= CONTEXT_IDS.len() as u64 {
                    return Err(CanonicalError::new(
                        REJECT_REGISTRY_INDEX_OUT_OF_RANGE,
                        "scope extension contains an out-of-range context id",
                    ));
                }
                if previous == Some(*context_id) {
                    return Err(CanonicalError::new(
                        REJECT_DUPLICATE_SCOPE_CONTEXT,
                        "scope extension contains a duplicate context id",
                    ));
                }
                previous = Some(*context_id);
            }
            Ok(if *map_id == 1 {
                Sort::BoundedInt
            } else {
                Sort::RationalValue
            })
        }
        Node::ContextFlag(index) => {
            if *index >= CONTEXT_IDS.len() as u64 {
                return Err(CanonicalError::new(
                    REJECT_REGISTRY_INDEX_OUT_OF_RANGE,
                    "context flag id is out of range",
                ));
            }
            Ok(Sort::Bool)
        }
        Node::TaskFlag(index) => {
            if *index >= TASK_IDS.len() as u64 {
                return Err(CanonicalError::new(
                    REJECT_REGISTRY_INDEX_OUT_OF_RANGE,
                    "task flag id is out of range",
                ));
            }
            Ok(Sort::Bool)
        }
        Node::NewSymbolCall(_) => Err(CanonicalError::new(
            REJECT_NEW_SYMBOL_IN_OLD_DSL,
            "new_symbol_call is forbidden in the old DSL canonicalizer",
        )),
        Node::Unary { op, child } => {
            let child_sort = type_check(child)?;
            match op {
                UnaryOp::BitToScalar if child_sort == Sort::Bit => Ok(Sort::RationalValue),
                UnaryOp::IntToScalar if child_sort == Sort::BoundedInt => Ok(Sort::RationalValue),
                UnaryOp::Absolute => {
                    rational_expected(child_sort, "absolute")?;
                    Ok(Sort::RationalValue)
                }
                UnaryOp::Sign => {
                    rational_expected(child_sort, "sign")?;
                    Ok(Sort::Sign)
                }
                _ => Err(CanonicalError::new(
                    REJECT_TYPE_MISMATCH,
                    format!("{op:?} cannot consume {child_sort:?}"),
                )),
            }
        }
        Node::Binary { op, left, right } => {
            let left_sort = type_check(left)?;
            let right_sort = type_check(right)?;
            match op {
                BinaryOp::Add | BinaryOp::Difference => {
                    rational_expected(left_sort, &format!("{op:?} left child"))?;
                    rational_expected(right_sort, &format!("{op:?} right child"))?;
                    Ok(Sort::RationalValue)
                }
                BinaryOp::EqualExact | BinaryOp::LessEqual | BinaryOp::GreaterEqual => {
                    rational_expected(left_sort, &format!("{op:?} left child"))?;
                    rational_expected(right_sort, &format!("{op:?} right child"))?;
                    Ok(Sort::Bool)
                }
                BinaryOp::SameSign | BinaryOp::OppositeSign => {
                    if left_sort != Sort::Sign || right_sort != Sort::Sign {
                        return Err(CanonicalError::new(
                            REJECT_TYPE_MISMATCH,
                            format!(
                                "{op:?} requires two Sign children, got {left_sort:?} and {right_sort:?}"
                            ),
                        ));
                    }
                    Ok(Sort::Bool)
                }
            }
        }
        Node::ApproxEqual {
            left,
            right,
            tolerance_index,
        } => {
            rational_expected(type_check(left)?, "approx_equal left child")?;
            rational_expected(type_check(right)?, "approx_equal right child")?;
            if *tolerance_index >= TOLERANCES.len() as u64 {
                return Err(CanonicalError::new(
                    REJECT_REGISTRY_INDEX_OUT_OF_RANGE,
                    "approx_equal tolerance index is outside 0..3",
                ));
            }
            Ok(Sort::Bool)
        }
        Node::And(atoms) => {
            if atoms.is_empty() {
                return Err(CanonicalError::new(
                    REJECT_MALFORMED_SOURCE_AST,
                    "top_level_AND must not be empty",
                ));
            }
            for atom in atoms {
                let atom_sort = type_check(atom)?;
                if atom_sort != Sort::Bool {
                    return Err(CanonicalError::new(
                        REJECT_TYPE_MISMATCH,
                        format!("top_level_AND atom must be Bool, got {atom_sort:?}"),
                    ));
                }
            }
            Ok(Sort::Bool)
        }
    }
}

fn scalar_const_value(node: &Node) -> Option<Rational> {
    match node {
        Node::ScalarConst(index) => rational_parameter(*index).ok(),
        _ => None,
    }
}

fn is_zero_const(node: &Node) -> bool {
    matches!(node, Node::ScalarConst(index) if *index == ZERO_PARAMETER_INDEX)
}

fn node_cbor(node: &Node) -> Vec<u8> {
    let mut output = Vec::new();
    encode_node(node, &mut output);
    output
}

fn canonical_child_key(node: &Node) -> ([u8; 32], Vec<u8>) {
    let bytes = node_cbor(node);
    let digest: [u8; 32] = Sha256::digest(&bytes).into();
    (digest, bytes)
}

fn canonical_child_less_or_equal(left: &Node, right: &Node) -> bool {
    canonical_child_key(left) <= canonical_child_key(right)
}

fn order_commutative_pair(left: Node, right: Node) -> (Node, Node) {
    if canonical_child_less_or_equal(&left, &right) {
        (left, right)
    } else {
        (right, left)
    }
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

fn normalize_once(node: Node) -> Result<Node, CanonicalError> {
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
                return Err(CanonicalError::new(
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
                    if let Some(value) = scalar_const_value(&child) {
                        if let Some(index) =
                            value.absolute().as_ref().and_then(rational_parameter_index)
                        {
                            return Ok(Node::ScalarConst(index));
                        }
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
                            .and_then(rational_parameter_index)
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
                            .and_then(rational_parameter_index)
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
                _ if op.is_commutative() => {
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
                return Err(CanonicalError::new(
                    REJECT_MALFORMED_SOURCE_AST,
                    "AND normalization cannot produce an empty conjunction",
                ));
            }
            if flattened.len() == 1 {
                return Ok(flattened.pop().expect("one AND atom"));
            }
            if flattened.len() > MAX_TOP_LEVEL_CLAUSES {
                return Err(CanonicalError::new(
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

fn normalize_to_fixed_point(mut node: Node) -> Result<Node, CanonicalError> {
    for _ in 0..64 {
        let next = normalize_once(node.clone())?;
        if next == node {
            return Ok(node);
        }
        node = next;
    }
    Err(CanonicalError::new(
        REJECT_INTERNAL_CANONICALIZATION,
        "frozen rewrite system did not reach a fixed point within 64 passes",
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
            let child_stats = atoms.iter().map(ast_stats).collect::<Vec<_>>();
            merge_stats(1, &child_stats)
        }
    }
}

fn validate_structural_limits(node: &Node) -> Result<AstStats, CanonicalError> {
    if let Node::And(atoms) = node {
        if !(2..=MAX_TOP_LEVEL_CLAUSES).contains(&atoms.len()) {
            return Err(CanonicalError::new(
                REJECT_STRUCTURAL_LIMIT,
                "canonical AND must contain exactly two or three atoms",
            ));
        }
    }
    let stats = ast_stats(node);
    if stats.node_count > MAX_TOTAL_NODE_COUNT {
        return Err(CanonicalError::new(
            REJECT_STRUCTURAL_LIMIT,
            format!(
                "AST has {} nodes; maximum is {MAX_TOTAL_NODE_COUNT}",
                stats.node_count
            ),
        ));
    }
    if stats.depth > MAX_TOTAL_AST_DEPTH {
        return Err(CanonicalError::new(
            REJECT_STRUCTURAL_LIMIT,
            format!(
                "AST depth is {}; maximum is {MAX_TOTAL_AST_DEPTH}",
                stats.depth
            ),
        ));
    }
    if stats.bit_slots.len() > MAX_DISTINCT_BIT_SLOTS {
        return Err(CanonicalError::new(
            REJECT_STRUCTURAL_LIMIT,
            format!(
                "AST uses {} distinct bit slots; maximum is {MAX_DISTINCT_BIT_SLOTS}",
                stats.bit_slots.len()
            ),
        ));
    }
    if stats.aggregate_leaves > MAX_AGGREGATE_LEAVES {
        return Err(CanonicalError::new(
            REJECT_STRUCTURAL_LIMIT,
            format!(
                "AST uses {} aggregate leaves; maximum is {MAX_AGGREGATE_LEAVES}",
                stats.aggregate_leaves
            ),
        ));
    }
    if stats.scalar_parameter_occurrences > MAX_FITTED_SCALAR_PARAMETER_OCCURRENCES {
        return Err(CanonicalError::new(
            REJECT_STRUCTURAL_LIMIT,
            format!(
                "AST uses {} fitted scalar-parameter occurrences; maximum is {MAX_FITTED_SCALAR_PARAMETER_OCCURRENCES}",
                stats.scalar_parameter_occurrences
            ),
        ));
    }
    Ok(stats)
}

fn encode_major_value(major: u8, value: u64, output: &mut Vec<u8>) {
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

fn encode_uint(value: u64, output: &mut Vec<u8>) {
    encode_major_value(0, value, output);
}

fn encode_array_len(length: usize, output: &mut Vec<u8>) {
    encode_major_value(4, length as u64, output);
}

fn encode_bool(value: bool, output: &mut Vec<u8>) {
    output.push(if value { 0xf5 } else { 0xf4 });
}

fn encode_node(node: &Node, output: &mut Vec<u8>) {
    match node {
        Node::ScalarConst(index) => {
            encode_array_len(3, output);
            encode_uint(0, output);
            encode_uint(0, output);
            encode_uint(*index, output);
        }
        Node::BitAt(index) => {
            encode_array_len(3, output);
            encode_uint(0, output);
            encode_uint(1, output);
            encode_uint(*index, output);
        }
        Node::SetSize => {
            encode_array_len(2, output);
            encode_uint(0, output);
            encode_uint(2, output);
        }
        Node::Aggregate {
            map_id,
            scope_id,
            quantity_id,
            scope_extension,
        } => {
            encode_array_len(6, output);
            encode_uint(0, output);
            encode_uint(3, output);
            encode_uint(*map_id, output);
            encode_uint(*scope_id, output);
            encode_uint(*quantity_id, output);
            encode_array_len(scope_extension.len(), output);
            for (context_id, expected) in scope_extension {
                encode_array_len(2, output);
                encode_uint(*context_id, output);
                encode_bool(*expected, output);
            }
        }
        Node::ContextFlag(index) => {
            encode_array_len(3, output);
            encode_uint(0, output);
            encode_uint(4, output);
            encode_uint(*index, output);
        }
        Node::TaskFlag(index) => {
            encode_array_len(3, output);
            encode_uint(0, output);
            encode_uint(5, output);
            encode_uint(*index, output);
        }
        Node::NewSymbolCall(index) => {
            encode_array_len(3, output);
            encode_uint(0, output);
            encode_uint(6, output);
            encode_uint(*index, output);
        }
        Node::Unary { op, child } => {
            encode_array_len(3, output);
            encode_uint(1, output);
            encode_uint(op.id(), output);
            encode_node(child, output);
        }
        Node::Binary { op, left, right } => {
            encode_array_len(4, output);
            encode_uint(2, output);
            encode_uint(op.id(), output);
            encode_node(left, output);
            encode_node(right, output);
        }
        Node::ApproxEqual {
            left,
            right,
            tolerance_index,
        } => {
            encode_array_len(5, output);
            encode_uint(3, output);
            encode_uint(0, output);
            encode_node(left, output);
            encode_node(right, output);
            encode_uint(*tolerance_index, output);
        }
        Node::And(atoms) => {
            encode_array_len(2, output);
            encode_uint(4, output);
            encode_array_len(atoms.len(), output);
            for atom in atoms {
                encode_node(atom, output);
            }
        }
    }
}

fn encode_ast_envelope(node: &Node) -> Vec<u8> {
    let mut output = Vec::new();
    encode_array_len(2, &mut output);
    encode_uint(1, &mut output);
    encode_node(node, &mut output);
    output
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
        Node::Unary { op, .. } => 0x0100 + op.id() as u16,
        Node::Binary { op, .. } => 0x0200 + op.id() as u16,
        Node::ApproxEqual { .. } => 0x0300,
        Node::And(_) => 0x0400,
    }
}

fn content_hash(domain: &[u8], canonical_cbor: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update([0]);
    hasher.update(canonical_cbor);
    hasher.finalize().into()
}

pub fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

pub fn hex_decode(input: &str) -> Result<Vec<u8>, CanonicalError> {
    let input = input.strip_prefix("0x").unwrap_or(input);
    if input.len() % 2 != 0 {
        return Err(CanonicalError::new(
            REJECT_MALFORMED_SOURCE_AST,
            "hex input must contain an even number of digits",
        ));
    }
    let mut output = Vec::with_capacity(input.len() / 2);
    let bytes = input.as_bytes();
    for index in (0..bytes.len()).step_by(2) {
        let high = hex_nibble(bytes[index]).ok_or_else(|| {
            CanonicalError::new(
                REJECT_MALFORMED_SOURCE_AST,
                format!("invalid hex digit at offset {index}"),
            )
        })?;
        let low = hex_nibble(bytes[index + 1]).ok_or_else(|| {
            CanonicalError::new(
                REJECT_MALFORMED_SOURCE_AST,
                format!("invalid hex digit at offset {}", index + 1),
            )
        })?;
        output.push((high << 4) | low);
    }
    Ok(output)
}

fn hex_nibble(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalProgram {
    pub canonical_node: Node,
    pub canonical_cbor: Vec<u8>,
    pub canonical_ast_hash: [u8; 32],
    pub output_sort: Sort,
    pub root_operator_id: u16,
    pub node_count: u32,
    pub depth: u32,
    pub distinct_bit_slot_count: usize,
    pub aggregate_leaf_count: u32,
    pub scalar_parameter_occurrence_count: u32,
}

impl CanonicalProgram {
    pub fn canonical_cbor_hex(&self) -> String {
        hex_encode(&self.canonical_cbor)
    }

    pub fn canonical_ast_hash_hex(&self) -> String {
        hex_encode(&self.canonical_ast_hash)
    }

    pub fn canonical_ast_hash_id(&self) -> String {
        format!("sha256:{}", self.canonical_ast_hash_hex())
    }
}

fn finish_program(canonical_node: Node) -> Result<CanonicalProgram, CanonicalError> {
    let output_sort = type_check(&canonical_node)?;
    let stats = validate_structural_limits(&canonical_node)?;
    let canonical_cbor = encode_ast_envelope(&canonical_node);
    let canonical_ast_hash = content_hash(b"HEGEL/AST/V1", &canonical_cbor);
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

/// Parse, type-check, normalize, and structurally validate one source JSON AST.
pub fn canonicalize_source_json(value: &Value) -> Result<CanonicalProgram, CanonicalError> {
    let source = parse_source_ast(value)?;
    canonicalize_source_node(source)
}

/// Canonicalize one already parsed source node. This entry point is used by
/// the independent in-crate 64,680 witness generator so the capacity replay
/// does not round-trip through JSON.
pub fn canonicalize_source_node(source: Node) -> Result<CanonicalProgram, CanonicalError> {
    // Type checking intentionally precedes rewrites. In particular,
    // difference(Bit, Bit) must not collapse through difference(x, x).
    type_check(&source)?;
    let canonical = normalize_to_fixed_point(source)?;
    finish_program(canonical)
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CborValue {
    Unsigned(u64),
    Negative(u64), // stores n for the CBOR mathematical value -1-n
    Bytes(Vec<u8>),
    Array(Vec<CborValue>),
    Bool(bool),
    Null,
}

fn encode_cbor_value(value: &CborValue, output: &mut Vec<u8>) {
    match value {
        CborValue::Unsigned(value) => encode_major_value(0, *value, output),
        CborValue::Negative(argument) => encode_major_value(1, *argument, output),
        CborValue::Bytes(bytes) => {
            encode_major_value(2, bytes.len() as u64, output);
            output.extend_from_slice(bytes);
        }
        CborValue::Array(values) => {
            encode_major_value(4, values.len() as u64, output);
            for value in values {
                encode_cbor_value(value, output);
            }
        }
        CborValue::Bool(false) => output.push(0xf4),
        CborValue::Bool(true) => output.push(0xf5),
        CborValue::Null => output.push(0xf6),
    }
}

fn read_exact<'a>(
    bytes: &'a [u8],
    cursor: &mut usize,
    length: usize,
) -> Result<&'a [u8], CanonicalError> {
    let end = cursor
        .checked_add(length)
        .ok_or_else(|| CanonicalError::new(REJECT_NONCANONICAL_CBOR, "CBOR length overflow"))?;
    if end > bytes.len() {
        return Err(CanonicalError::new(
            REJECT_NONCANONICAL_CBOR,
            "truncated CBOR item",
        ));
    }
    let result = &bytes[*cursor..end];
    *cursor = end;
    Ok(result)
}

fn read_argument(additional: u8, bytes: &[u8], cursor: &mut usize) -> Result<u64, CanonicalError> {
    let (value, minimum) = match additional {
        0..=23 => return Ok(additional as u64),
        24 => (read_exact(bytes, cursor, 1)?[0] as u64, 24),
        25 => {
            let raw: [u8; 2] = read_exact(bytes, cursor, 2)?.try_into().expect("two bytes");
            (u16::from_be_bytes(raw) as u64, 0x100)
        }
        26 => {
            let raw: [u8; 4] = read_exact(bytes, cursor, 4)?
                .try_into()
                .expect("four bytes");
            (u32::from_be_bytes(raw) as u64, 0x1_0000)
        }
        27 => {
            let raw: [u8; 8] = read_exact(bytes, cursor, 8)?
                .try_into()
                .expect("eight bytes");
            (u64::from_be_bytes(raw), 0x1_0000_0000)
        }
        31 => {
            return Err(CanonicalError::new(
                REJECT_INDEFINITE_CBOR,
                "indefinite-length CBOR is forbidden",
            ))
        }
        _ => {
            return Err(CanonicalError::new(
                REJECT_NONCANONICAL_CBOR,
                "reserved CBOR additional-information value",
            ))
        }
    };
    if value < minimum {
        return Err(CanonicalError::new(
            REJECT_NONCANONICAL_CBOR,
            "CBOR integer or length does not use the shortest encoding",
        ));
    }
    Ok(value)
}

fn parse_cbor_value(
    bytes: &[u8],
    cursor: &mut usize,
    recursion_depth: usize,
) -> Result<CborValue, CanonicalError> {
    if recursion_depth > MAX_CBOR_NESTING {
        return Err(CanonicalError::new(
            REJECT_CBOR_NESTING,
            "CBOR nesting exceeds the strict decoder limit",
        ));
    }
    let initial = *read_exact(bytes, cursor, 1)?
        .first()
        .expect("one initial byte");
    let major = initial >> 5;
    let additional = initial & 0x1f;
    match major {
        0 => Ok(CborValue::Unsigned(read_argument(
            additional, bytes, cursor,
        )?)),
        1 => Ok(CborValue::Negative(read_argument(
            additional, bytes, cursor,
        )?)),
        2 => {
            let length = read_argument(additional, bytes, cursor)?;
            let length = usize::try_from(length).map_err(|_| {
                CanonicalError::new(
                    REJECT_NONCANONICAL_CBOR,
                    "byte-string length does not fit this implementation",
                )
            })?;
            Ok(CborValue::Bytes(
                read_exact(bytes, cursor, length)?.to_vec(),
            ))
        }
        3 => Err(CanonicalError::new(
            REJECT_CBOR_TEXT,
            "CBOR text strings are forbidden",
        )),
        4 => {
            let length = read_argument(additional, bytes, cursor)?;
            let length = usize::try_from(length).map_err(|_| {
                CanonicalError::new(
                    REJECT_NONCANONICAL_CBOR,
                    "array length does not fit this implementation",
                )
            })?;
            // Every array element consumes at least one byte. This bound also
            // prevents a malicious declared length from allocating excessively.
            if length > bytes.len().saturating_sub(*cursor) {
                return Err(CanonicalError::new(
                    REJECT_NONCANONICAL_CBOR,
                    "CBOR array length exceeds remaining input",
                ));
            }
            let mut values = Vec::with_capacity(length);
            for _ in 0..length {
                values.push(parse_cbor_value(bytes, cursor, recursion_depth + 1)?);
            }
            Ok(CborValue::Array(values))
        }
        5 => Err(CanonicalError::new(
            REJECT_CBOR_MAP,
            "CBOR maps are forbidden",
        )),
        6 => Err(CanonicalError::new(
            REJECT_CBOR_TAG,
            "CBOR tags are forbidden",
        )),
        7 => match additional {
            20 => Ok(CborValue::Bool(false)),
            21 => Ok(CborValue::Bool(true)),
            22 => Ok(CborValue::Null),
            25..=27 => Err(CanonicalError::new(
                REJECT_CBOR_FLOAT,
                "CBOR floating-point values are forbidden",
            )),
            31 => Err(CanonicalError::new(
                REJECT_INDEFINITE_CBOR,
                "CBOR break/indefinite encoding is forbidden",
            )),
            _ => Err(CanonicalError::new(
                REJECT_NONCANONICAL_CBOR,
                "only false, true, and null CBOR simple values are allowed",
            )),
        },
        _ => unreachable!("CBOR major type is three bits"),
    }
}

fn decode_cbor_value(bytes: &[u8]) -> Result<CborValue, CanonicalError> {
    let mut cursor = 0;
    let value = parse_cbor_value(bytes, &mut cursor, 0)?;
    if cursor != bytes.len() {
        return Err(CanonicalError::new(
            REJECT_TRAILING_CBOR,
            "trailing bytes after the CBOR item",
        ));
    }
    let mut reencoded = Vec::new();
    encode_cbor_value(&value, &mut reencoded);
    if reencoded != bytes {
        return Err(CanonicalError::new(
            REJECT_NONCANONICAL_CBOR,
            "CBOR item differs from its exact deterministic re-encoding",
        ));
    }
    Ok(value)
}

/// Validate one item against the no-map/no-text/no-float/no-tag deterministic
/// CBOR subset. The returned bytes are the exact deterministic re-encoding.
pub fn validate_strict_cbor(bytes: &[u8]) -> Result<Vec<u8>, CanonicalError> {
    let value = decode_cbor_value(bytes)?;
    let mut reencoded = Vec::new();
    encode_cbor_value(&value, &mut reencoded);
    Ok(reencoded)
}

fn cbor_from_json(value: &Value) -> Result<CborValue, CanonicalError> {
    match value {
        Value::Null => Ok(CborValue::Null),
        Value::Bool(value) => Ok(CborValue::Bool(*value)),
        Value::Number(number) => {
            if let Some(value) = number.as_u64() {
                Ok(CborValue::Unsigned(value))
            } else if let Some(value) = number.as_i64() {
                if value >= 0 {
                    Ok(CborValue::Unsigned(value as u64))
                } else {
                    let argument = (-1_i128 - value as i128) as u64;
                    Ok(CborValue::Negative(argument))
                }
            } else {
                Err(CanonicalError::new(
                    REJECT_NONCANONICAL_CBOR,
                    "JSON floating-point values cannot enter strict CBOR",
                ))
            }
        }
        Value::Array(values) => Ok(CborValue::Array(
            values
                .iter()
                .map(cbor_from_json)
                .collect::<Result<Vec<_>, _>>()?,
        )),
        Value::String(_) => Err(CanonicalError::new(
            REJECT_NONCANONICAL_CBOR,
            "JSON strings cannot enter the formal CBOR core; use byte_string_hex",
        )),
        Value::Object(_) => Err(CanonicalError::new(
            REJECT_NONCANONICAL_CBOR,
            "JSON objects/maps cannot enter the formal CBOR core",
        )),
    }
}

/// Encode an allowed JSON scalar/array value into the project-minimal CBOR
/// profile. Byte strings use [`encode_strict_cbor_byte_string`].
pub fn encode_strict_cbor_json(value: &Value) -> Result<Vec<u8>, CanonicalError> {
    let value = cbor_from_json(value)?;
    let mut output = Vec::new();
    encode_cbor_value(&value, &mut output);
    Ok(output)
}

pub fn encode_strict_cbor_byte_string(bytes: &[u8]) -> Vec<u8> {
    let mut output = Vec::new();
    encode_cbor_value(&CborValue::Bytes(bytes.to_vec()), &mut output);
    output
}

fn cbor_array<'a>(value: &'a CborValue, context: &str) -> Result<&'a [CborValue], CanonicalError> {
    match value {
        CborValue::Array(values) => Ok(values),
        _ => Err(CanonicalError::new(
            REJECT_NONCANONICAL_AST,
            format!("{context} must be a CBOR array"),
        )),
    }
}

fn cbor_uint(value: &CborValue, context: &str) -> Result<u64, CanonicalError> {
    match value {
        CborValue::Unsigned(value) => Ok(*value),
        _ => Err(CanonicalError::new(
            REJECT_NONCANONICAL_AST,
            format!("{context} must be a CBOR unsigned integer"),
        )),
    }
}

fn cbor_bool(value: &CborValue, context: &str) -> Result<bool, CanonicalError> {
    match value {
        CborValue::Bool(value) => Ok(*value),
        _ => Err(CanonicalError::new(
            REJECT_NONCANONICAL_AST,
            format!("{context} must be a CBOR boolean"),
        )),
    }
}

fn expect_array_length(
    values: &[CborValue],
    expected: usize,
    context: &str,
) -> Result<(), CanonicalError> {
    if values.len() != expected {
        return Err(CanonicalError::new(
            REJECT_NONCANONICAL_AST,
            format!(
                "{context} has array length {}; expected {expected}",
                values.len()
            ),
        ));
    }
    Ok(())
}

fn node_from_cbor(value: &CborValue) -> Result<Node, CanonicalError> {
    let values = cbor_array(value, "AST node")?;
    if values.is_empty() {
        return Err(CanonicalError::new(
            REJECT_NONCANONICAL_AST,
            "AST node array must not be empty",
        ));
    }
    match cbor_uint(&values[0], "AST node tag")? {
        0 => {
            if values.len() < 2 {
                return Err(CanonicalError::new(
                    REJECT_NONCANONICAL_AST,
                    "leaf node is missing its leaf id",
                ));
            }
            match cbor_uint(&values[1], "leaf id")? {
                0 => {
                    expect_array_length(values, 3, "scalar_const")?;
                    Ok(Node::ScalarConst(cbor_uint(
                        &values[2],
                        "rational parameter index",
                    )?))
                }
                1 => {
                    expect_array_length(values, 3, "bit_at")?;
                    Ok(Node::BitAt(cbor_uint(&values[2], "entity slot index")?))
                }
                2 => {
                    expect_array_length(values, 2, "set_size")?;
                    Ok(Node::SetSize)
                }
                3 => {
                    expect_array_length(values, 6, "aggregate")?;
                    let clauses = cbor_array(&values[5], "scope extension")?;
                    let mut scope_extension = Vec::with_capacity(clauses.len());
                    for clause in clauses {
                        let clause = cbor_array(clause, "scope clause")?;
                        expect_array_length(clause, 2, "scope clause")?;
                        scope_extension.push((
                            cbor_uint(&clause[0], "scope context id")?,
                            cbor_bool(&clause[1], "scope expected bool")?,
                        ));
                    }
                    Ok(Node::Aggregate {
                        map_id: cbor_uint(&values[2], "aggregate map id")?,
                        scope_id: cbor_uint(&values[3], "scope id")?,
                        quantity_id: cbor_uint(&values[4], "quantity id")?,
                        scope_extension,
                    })
                }
                4 => {
                    expect_array_length(values, 3, "context_flag")?;
                    Ok(Node::ContextFlag(cbor_uint(&values[2], "context id")?))
                }
                5 => {
                    expect_array_length(values, 3, "task_flag")?;
                    Ok(Node::TaskFlag(cbor_uint(&values[2], "task id")?))
                }
                6 => {
                    expect_array_length(values, 3, "new_symbol_call")?;
                    Ok(Node::NewSymbolCall(cbor_uint(
                        &values[2],
                        "new-symbol registry index",
                    )?))
                }
                leaf_id => Err(CanonicalError::new(
                    REJECT_UNKNOWN_EXPRESSION,
                    format!("unknown/reserved leaf id {leaf_id}"),
                )),
            }
        }
        1 => {
            expect_array_length(values, 3, "unary node")?;
            let op = match cbor_uint(&values[1], "unary operator id")? {
                0 => UnaryOp::BitToScalar,
                1 => UnaryOp::IntToScalar,
                2 => UnaryOp::Absolute,
                3 => UnaryOp::Sign,
                op => {
                    return Err(CanonicalError::new(
                        REJECT_UNKNOWN_EXPRESSION,
                        format!("unknown unary operator id {op}"),
                    ))
                }
            };
            Ok(Node::Unary {
                op,
                child: Box::new(node_from_cbor(&values[2])?),
            })
        }
        2 => {
            expect_array_length(values, 4, "binary node")?;
            let op = match cbor_uint(&values[1], "binary operator id")? {
                0 => BinaryOp::Add,
                1 => BinaryOp::Difference,
                2 => BinaryOp::EqualExact,
                3 => BinaryOp::LessEqual,
                4 => BinaryOp::GreaterEqual,
                5 => BinaryOp::SameSign,
                6 => BinaryOp::OppositeSign,
                op => {
                    return Err(CanonicalError::new(
                        REJECT_UNKNOWN_EXPRESSION,
                        format!("unknown/reserved binary operator id {op}"),
                    ))
                }
            };
            Ok(Node::Binary {
                op,
                left: Box::new(node_from_cbor(&values[2])?),
                right: Box::new(node_from_cbor(&values[3])?),
            })
        }
        3 => {
            expect_array_length(values, 5, "ternary node")?;
            let operator_id = cbor_uint(&values[1], "ternary operator id")?;
            if operator_id != 0 {
                return Err(CanonicalError::new(
                    REJECT_UNKNOWN_EXPRESSION,
                    format!("unknown ternary operator id {operator_id}"),
                ));
            }
            Ok(Node::ApproxEqual {
                left: Box::new(node_from_cbor(&values[2])?),
                right: Box::new(node_from_cbor(&values[3])?),
                tolerance_index: cbor_uint(&values[4], "tolerance index")?,
            })
        }
        4 => {
            expect_array_length(values, 2, "conjunction node")?;
            let atoms = cbor_array(&values[1], "conjunction atom list")?
                .iter()
                .map(node_from_cbor)
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Node::And(atoms))
        }
        tag => Err(CanonicalError::new(
            REJECT_UNKNOWN_EXPRESSION,
            format!("unknown/reserved AST node tag {tag}"),
        )),
    }
}

/// Decode and validate exact strict canonical AST CBOR. Schema-valid but
/// rewriteable encodings are rejected rather than silently normalized.
pub fn decode_strict_canonical_ast(bytes: &[u8]) -> Result<CanonicalProgram, CanonicalError> {
    let value = decode_cbor_value(bytes)?;
    let envelope = cbor_array(&value, "CanonicalAstV1 envelope")?;
    expect_array_length(envelope, 2, "CanonicalAstV1 envelope")?;
    if cbor_uint(&envelope[0], "CanonicalAstV1 schema version")? != 1 {
        return Err(CanonicalError::new(
            REJECT_NONCANONICAL_AST,
            "unknown canonical AST schema version",
        ));
    }
    let source_node = node_from_cbor(&envelope[1])?;
    type_check(&source_node)?;
    let normalized = normalize_to_fixed_point(source_node)?;
    let reencoded = encode_ast_envelope(&normalized);
    if reencoded != bytes {
        return Err(CanonicalError::new(
            REJECT_NONCANONICAL_AST,
            "AST bytes are schema-readable but not in frozen canonical normal form",
        ));
    }
    finish_program(normalized)
}

fn sha256_parts(parts: &[&[u8]]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update(part);
    }
    hasher.finalize().into()
}

/// RFC6962 Merkle tree root over already canonical-CBOR records.
pub fn rfc6962_root(records: &[Vec<u8>]) -> [u8; 32] {
    if records.is_empty() {
        return Sha256::digest([]).into();
    }
    let leaf_hashes = records
        .iter()
        .map(|record| sha256_parts(&[&[0], record]))
        .collect::<Vec<_>>();
    rfc6962_hash_tree(&leaf_hashes)
}

fn rfc6962_hash_tree(hashes: &[[u8; 32]]) -> [u8; 32] {
    if hashes.len() == 1 {
        return hashes[0];
    }
    let mut split = 1usize;
    while split
        .checked_mul(2)
        .is_some_and(|candidate| candidate < hashes.len())
    {
        split *= 2;
    }
    let left = rfc6962_hash_tree(&hashes[..split]);
    let right = rfc6962_hash_tree(&hashes[split..]);
    sha256_parts(&[&[1], &left, &right])
}

/// Golden-vector record rule `schema1_index`: record `i` is canonical CBOR
/// `[1, i]` and records are ordered by ascending index.
pub fn rfc6962_schema1_index_root(leaf_count: usize) -> [u8; 32] {
    let records = (0..leaf_count)
        .map(|index| {
            let mut record = Vec::new();
            encode_array_len(2, &mut record);
            encode_uint(1, &mut record);
            encode_uint(index as u64, &mut record);
            record
        })
        .collect::<Vec<_>>();
    rfc6962_root(&records)
}

#[derive(Debug, Clone, Serialize)]
pub struct CapacityReplayReport {
    pub schema_version: &'static str,
    pub freeze_version: &'static str,
    pub implementation: &'static str,
    pub cbor_profile_id: &'static str,
    pub ast_schema_id: &'static str,
    pub generator_rule: &'static str,
    pub source_candidate_count: usize,
    pub source_count: usize,
    pub accepted_source_count: usize,
    pub accepted_total_count: usize,
    pub accepted_unique_count: usize,
    pub rejected_count: usize,
    pub type_rejected_count: usize,
    pub limit_rejected_count: usize,
    pub other_rejected_count: usize,
    pub rewrite_collapsed_count: usize,
    pub canonical_program_budget: usize,
    pub first_out_of_budget_ordinal: Option<usize>,
    pub first_out_of_budget_ast_hash: Option<String>,
    pub first_out_of_budget_cbor_hex: Option<String>,
    pub first_accepted_out_of_budget_ordinal: Option<usize>,
    pub first_accepted_out_of_budget_hash: Option<String>,
    pub first_accepted_out_of_budget_cbor_hex: Option<String>,
    pub accepted_set_commitment: String,
    pub accepted_set_commitment_hex: String,
    pub accepted_set_commitment_domain: &'static str,
    pub accepted_set_commitment_framing: &'static str,
    pub canonical_set_order: &'static str,
    pub executed_closure_status: &'static str,
    pub dual_replay_equal: Option<bool>,
    pub dsl_too_large_claim_allowed: bool,
    pub claim_boundary: &'static str,
}

fn capacity_constant_atoms() -> Vec<Node> {
    let constants = (0..RATIONAL_PARAMETERS.len() as u64)
        .map(Node::ScalarConst)
        .collect::<Vec<_>>();
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
    debug_assert_eq!(atoms.len(), 77);
    atoms
}

fn capacity_rational_aggregates() -> Vec<Node> {
    // map id 1 (count_nonzero_v1) returns BoundedInt and is deliberately
    // excluded from this RationalValue witness subset.
    let rational_map_ids = [0_u64, 2, 3, 4, 5];
    let mut aggregates = Vec::with_capacity(40);
    for map_id in rational_map_ids {
        for scope_id in 0..SCOPE_IDS.len() as u64 {
            for quantity_id in 0..QUANTITY_IDS.len() as u64 {
                aggregates.push(Node::Aggregate {
                    map_id,
                    scope_id,
                    quantity_id,
                    scope_extension: Vec::new(),
                });
            }
        }
    }
    debug_assert_eq!(aggregates.len(), 40);
    aggregates
}

fn capacity_mixed_atoms() -> Vec<Node> {
    let constants = (0..RATIONAL_PARAMETERS.len() as u64)
        .map(Node::ScalarConst)
        .collect::<Vec<_>>();
    let aggregates = capacity_rational_aggregates();
    let mut atoms = Vec::with_capacity(840);
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
    debug_assert_eq!(atoms.len(), 840);
    atoms
}

fn capacity_set_commitment(sorted_cbor: &BTreeSet<Vec<u8>>) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"HEGEL/STRICT_CAPACITY_SET/V1");
    hasher.update([0]);
    for bytes in sorted_cbor {
        hasher.update((bytes.len() as u64).to_be_bytes());
        hasher.update(bytes);
    }
    hasher.finalize().into()
}

/// Independently construct and strict-canonicalize the exact 64,680 witness
/// subset preregistered for the Phase-3 M2 capacity replay.
pub fn replay_capacity_subset() -> Result<CapacityReplayReport, CanonicalError> {
    const EXPECTED_SOURCE_COUNT: usize = 64_680;
    const PROGRAM_BUDGET: usize = 50_000;
    let constant_atoms = capacity_constant_atoms();
    let mixed_atoms = capacity_mixed_atoms();
    if constant_atoms.len() != 77 || mixed_atoms.len() != 840 {
        return Err(CanonicalError::new(
            REJECT_INTERNAL_CANONICALIZATION,
            "capacity witness component count drift",
        ));
    }

    let mut source_count = 0usize;
    let mut accepted_total_count = 0usize;
    let mut type_rejected_count = 0usize;
    let mut limit_rejected_count = 0usize;
    let mut other_rejected_count = 0usize;
    let mut canonical_set = BTreeSet::new();
    for constant_atom in &constant_atoms {
        for mixed_atom in &mixed_atoms {
            source_count += 1;
            let source = Node::And(vec![constant_atom.clone(), mixed_atom.clone()]);
            match canonicalize_source_node(source) {
                Ok(program) => {
                    accepted_total_count += 1;
                    canonical_set.insert(program.canonical_cbor);
                }
                Err(error)
                    if matches!(error.code, REJECT_TYPE_MISMATCH | REJECT_IMPLICIT_COERCION) =>
                {
                    type_rejected_count += 1;
                }
                Err(error) if error.code == REJECT_STRUCTURAL_LIMIT => {
                    limit_rejected_count += 1;
                }
                Err(_) => other_rejected_count += 1,
            }
        }
    }
    if source_count != EXPECTED_SOURCE_COUNT {
        return Err(CanonicalError::new(
            REJECT_INTERNAL_CANONICALIZATION,
            format!(
                "capacity generator emitted {source_count} sources; expected {EXPECTED_SOURCE_COUNT}"
            ),
        ));
    }
    let rejected_count = type_rejected_count + limit_rejected_count + other_rejected_count;
    let accepted_unique_count = canonical_set.len();
    let rewrite_collapsed_count = accepted_total_count
        .checked_sub(accepted_unique_count)
        .ok_or_else(|| {
            CanonicalError::new(
                REJECT_INTERNAL_CANONICALIZATION,
                "capacity unique count exceeds accepted total",
            )
        })?;
    let first_out_of_budget = canonical_set.iter().nth(PROGRAM_BUDGET);
    let first_accepted_out_of_budget_hash = first_out_of_budget.map(|bytes| {
        format!(
            "sha256:{}",
            hex_encode(&content_hash(b"HEGEL/AST/V1", bytes))
        )
    });
    let first_accepted_out_of_budget_cbor_hex = first_out_of_budget.map(|bytes| hex_encode(bytes));
    let commitment = capacity_set_commitment(&canonical_set);
    let commitment_hex = hex_encode(&commitment);

    Ok(CapacityReplayReport {
        schema_version: "hegel-strict-capacity-replay/1",
        freeze_version: "hegel-freeze-p2b-p3-v1.0.2",
        implementation: "rust",
        cbor_profile_id: CBOR_PROFILE_ID,
        ast_schema_id: AST_SCHEMA_ID,
        generator_rule: "77 constant atoms x 840 one-aggregate atoms -> AND2 Cartesian product",
        source_candidate_count: source_count,
        source_count,
        accepted_source_count: accepted_total_count,
        accepted_total_count,
        accepted_unique_count,
        rejected_count,
        type_rejected_count,
        limit_rejected_count,
        other_rejected_count,
        rewrite_collapsed_count,
        canonical_program_budget: PROGRAM_BUDGET,
        first_out_of_budget_ordinal: first_out_of_budget.map(|_| PROGRAM_BUDGET + 1),
        first_out_of_budget_ast_hash: first_accepted_out_of_budget_hash.clone(),
        first_out_of_budget_cbor_hex: first_accepted_out_of_budget_cbor_hex.clone(),
        first_accepted_out_of_budget_ordinal: first_out_of_budget.map(|_| PROGRAM_BUDGET + 1),
        first_accepted_out_of_budget_hash,
        first_accepted_out_of_budget_cbor_hex,
        accepted_set_commitment: format!("sha256:{commitment_hex}"),
        accepted_set_commitment_hex: commitment_hex,
        accepted_set_commitment_domain: "HEGEL/STRICT_CAPACITY_SET/V1",
        accepted_set_commitment_framing: "domain_utf8 || 0x00 || concat(u64_be(cbor_len) || canonical_ast_cbor)",
        canonical_set_order: "canonical_ast_cbor_bytes_lexicographic_ascending",
        executed_closure_status: "NOT_RUN",
        dual_replay_equal: None,
        dsl_too_large_claim_allowed: false,
        claim_boundary: "Independent Rust strict-capacity replay only. A dual-replay comparison is still required before any DSL_TOO_LARGE transition; this is not full closure or an adequacy certificate.",
    })
}

#[derive(Debug, Clone, Serialize)]
pub struct ReplayVectorResult {
    pub group: String,
    pub name: String,
    pub status: String,
    pub expectation_match: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub mismatches: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub canonical_cbor_hex: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub canonical_ast_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub program_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub root_operator_id: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_sort: Option<Sort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub depth: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub distinct_bit_slot_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aggregate_leaf_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scalar_parameter_occurrence_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub root: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub root_hex: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub leaf_count: Option<usize>,
}

impl ReplayVectorResult {
    fn base(group: &str, name: String, status: &str) -> Self {
        Self {
            group: group.to_owned(),
            name,
            status: status.to_owned(),
            expectation_match: true,
            mismatches: Vec::new(),
            canonical_cbor_hex: None,
            canonical_ast_hash: None,
            program_hash: None,
            root_operator_id: None,
            output_sort: None,
            depth: None,
            node_count: None,
            distinct_bit_slot_count: None,
            aggregate_leaf_count: None,
            scalar_parameter_occurrence_count: None,
            error_code: None,
            error_message: None,
            root: None,
            root_hex: None,
            leaf_count: None,
        }
    }

    fn mismatch(&mut self, message: impl Into<String>) {
        self.expectation_match = false;
        self.mismatches.push(message.into());
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ReplaySummary {
    pub schema_version: &'static str,
    pub implementation: &'static str,
    pub freeze_version: &'static str,
    pub cbor_profile_id: &'static str,
    pub ast_schema_id: &'static str,
    pub source_path: String,
    pub vector_count: usize,
    pub passed_count: usize,
    pub failed_count: usize,
    pub accepted_result_count: usize,
    pub rejected_result_count: usize,
    pub all_expectations_match: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub metadata_errors: Vec<String>,
    pub results: Vec<ReplayVectorResult>,
}

fn vector_name(value: &Value, group: &str, index: usize) -> String {
    value
        .get("name")
        .or_else(|| value.get("id"))
        .and_then(Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| format!("{group}_{index}"))
}

fn normalize_expected_hex(value: &str) -> String {
    value
        .strip_prefix("sha256:")
        .or_else(|| value.strip_prefix("0x"))
        .unwrap_or(value)
        .to_ascii_lowercase()
}

fn compare_hex(
    result: &mut ReplayVectorResult,
    label: &str,
    actual: &str,
    expected: Option<&Value>,
) {
    let Some(expected) = expected else {
        result.mismatch(format!("missing required expectation {label}"));
        return;
    };
    let Some(expected) = expected.as_str() else {
        result.mismatch(format!("expectation {label} must be a string"));
        return;
    };
    if normalize_expected_hex(actual) != normalize_expected_hex(expected) {
        result.mismatch(format!(
            "{label} mismatch: expected {expected}, observed {actual}"
        ));
    }
}

fn compare_u64(
    result: &mut ReplayVectorResult,
    label: &str,
    actual: u64,
    expected: Option<&Value>,
) {
    let Some(expected) = expected else {
        result.mismatch(format!("missing required expectation {label}"));
        return;
    };
    let parsed = expected.as_u64().or_else(|| {
        expected.as_str().and_then(|value| {
            value
                .strip_prefix("0x")
                .and_then(|hex| u64::from_str_radix(hex, 16).ok())
                .or_else(|| value.parse::<u64>().ok())
        })
    });
    match parsed {
        Some(expected) if expected == actual => {}
        Some(expected) => result.mismatch(format!(
            "{label} mismatch: expected {expected}, observed {actual}"
        )),
        None => result.mismatch(format!("expectation {label} must be an unsigned integer")),
    }
}

fn compare_string(
    result: &mut ReplayVectorResult,
    label: &str,
    actual: &str,
    expected: Option<&Value>,
) {
    let Some(expected) = expected else {
        result.mismatch(format!("missing required expectation {label}"));
        return;
    };
    match expected.as_str() {
        Some(expected) if expected == actual => {}
        Some(expected) => result.mismatch(format!(
            "{label} mismatch: expected {expected}, observed {actual}"
        )),
        None => result.mismatch(format!("expectation {label} must be a string")),
    }
}

fn rejected_result(group: &str, name: String, error: CanonicalError) -> ReplayVectorResult {
    let mut result = ReplayVectorResult::base(group, name, "REJECTED");
    result.error_code = Some(error.code.to_owned());
    result.error_message = Some(error.message);
    result
}

fn accepted_program_result(
    group: &str,
    name: String,
    program: &CanonicalProgram,
) -> ReplayVectorResult {
    let mut result = ReplayVectorResult::base(group, name, "ACCEPTED");
    let hash = program.canonical_ast_hash_id();
    result.canonical_cbor_hex = Some(program.canonical_cbor_hex());
    result.canonical_ast_hash = Some(hash.clone());
    result.program_hash = Some(hash);
    result.root_operator_id = Some(program.root_operator_id);
    result.output_sort = Some(program.output_sort);
    result.depth = Some(program.depth);
    result.node_count = Some(program.node_count);
    result.distinct_bit_slot_count = Some(program.distinct_bit_slot_count);
    result.aggregate_leaf_count = Some(program.aggregate_leaf_count);
    result.scalar_parameter_occurrence_count = Some(program.scalar_parameter_occurrence_count);
    result
}

fn object_array<'a>(root: &'a Value, field: &str) -> Result<&'a [Value], String> {
    match root.get(field) {
        None => Ok(&[]),
        Some(Value::Array(values)) => Ok(values),
        Some(_) => Err(format!("fixture field {field:?} must be an array")),
    }
}

fn run_cbor_encode_vector(value: &Value, index: usize) -> ReplayVectorResult {
    let group = "cbor_encode_vectors";
    let name = vector_name(value, group, index);
    let encoded = match (value.get("value"), value.get("byte_string_hex")) {
        (Some(input), None) => encode_strict_cbor_json(input),
        (None, Some(Value::String(hex))) => {
            hex_decode(hex).map(|bytes| encode_strict_cbor_byte_string(&bytes))
        }
        (None, Some(_)) => Err(CanonicalError::new(
            REJECT_MALFORMED_SOURCE_AST,
            "byte_string_hex must be a string",
        )),
        (Some(_), Some(_)) => Err(CanonicalError::new(
            REJECT_MALFORMED_SOURCE_AST,
            "CBOR encode vector must not provide both value and byte_string_hex",
        )),
        (None, None) => Err(CanonicalError::new(
            REJECT_MALFORMED_SOURCE_AST,
            "CBOR encode vector needs value or byte_string_hex",
        )),
    };
    match encoded {
        Ok(bytes) => {
            let mut result = ReplayVectorResult::base(group, name, "ACCEPTED");
            let actual = hex_encode(&bytes);
            result.canonical_cbor_hex = Some(actual.clone());
            compare_hex(
                &mut result,
                "expected_cbor_hex",
                &actual,
                value.get("expected_cbor_hex"),
            );
            result
        }
        Err(error) => {
            let mut result = rejected_result(group, name, error);
            result.mismatch("CBOR encode vector unexpectedly rejected");
            result
        }
    }
}

fn run_cbor_reject_vector(value: &Value, index: usize) -> ReplayVectorResult {
    let group = "cbor_reject_vectors";
    let name = vector_name(value, group, index);
    let expected_code = value.get("error_code");
    let decoded = value
        .get("encoded_hex")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            CanonicalError::new(
                REJECT_MALFORMED_SOURCE_AST,
                "CBOR reject vector needs string encoded_hex",
            )
        })
        .and_then(hex_decode)
        .and_then(|bytes| validate_strict_cbor(&bytes));
    match decoded {
        Ok(bytes) => {
            let mut result = ReplayVectorResult::base(group, name, "ACCEPTED");
            result.canonical_cbor_hex = Some(hex_encode(&bytes));
            result.mismatch("CBOR reject vector was unexpectedly accepted");
            result
        }
        Err(error) => {
            let actual_code = error.code;
            let mut result = rejected_result(group, name, error);
            compare_string(&mut result, "error_code", actual_code, expected_code);
            result
        }
    }
}

fn run_ast_accept_vector(value: &Value, index: usize) -> ReplayVectorResult {
    let group = "ast_accept_vectors";
    let name = vector_name(value, group, index);
    let program = value
        .get("source_ast")
        .ok_or_else(|| {
            CanonicalError::new(
                REJECT_MALFORMED_SOURCE_AST,
                "AST accept vector needs source_ast",
            )
        })
        .and_then(canonicalize_source_json);
    match program {
        Ok(program) => {
            let mut result = accepted_program_result(group, name, &program);
            compare_hex(
                &mut result,
                "canonical_cbor_hex",
                &program.canonical_cbor_hex(),
                value.get("canonical_cbor_hex"),
            );
            compare_hex(
                &mut result,
                "canonical_ast_hash",
                &program.canonical_ast_hash_id(),
                value.get("canonical_ast_hash"),
            );
            compare_u64(
                &mut result,
                "root_operator_id",
                program.root_operator_id as u64,
                value.get("root_operator_id"),
            );
            compare_string(
                &mut result,
                "output_sort",
                &format!("{:?}", program.output_sort),
                value.get("output_sort"),
            );
            compare_u64(
                &mut result,
                "depth",
                program.depth as u64,
                value.get("depth"),
            );
            compare_u64(
                &mut result,
                "node_count",
                program.node_count as u64,
                value.get("node_count"),
            );
            result
        }
        Err(error) => {
            let mut result = rejected_result(group, name, error);
            result.mismatch("AST accept vector unexpectedly rejected");
            result
        }
    }
}

fn run_ast_reject_vector(value: &Value, index: usize) -> ReplayVectorResult {
    let group = "ast_reject_vectors";
    let name = vector_name(value, group, index);
    let expected_code = value.get("error_code");
    let program = value
        .get("source_ast")
        .ok_or_else(|| {
            CanonicalError::new(
                REJECT_MALFORMED_SOURCE_AST,
                "AST reject vector needs source_ast",
            )
        })
        .and_then(canonicalize_source_json);
    match program {
        Ok(program) => {
            let mut result = accepted_program_result(group, name, &program);
            result.mismatch("AST reject vector was unexpectedly accepted");
            result
        }
        Err(error) => {
            let actual_code = error.code;
            let mut result = rejected_result(group, name, error);
            compare_string(&mut result, "error_code", actual_code, expected_code);
            result
        }
    }
}

fn run_rfc6962_vector(value: &Value, index: usize) -> ReplayVectorResult {
    let group = "rfc6962_vectors";
    let name = vector_name(value, group, index);
    let mut result = ReplayVectorResult::base(group, name, "ACCEPTED");
    let leaf_count = match value.get("leaf_count").and_then(Value::as_u64) {
        Some(value) => match usize::try_from(value) {
            Ok(value) => value,
            Err(_) => {
                result.mismatch("leaf_count does not fit this implementation");
                return result;
            }
        },
        None => {
            result.mismatch("RFC6962 vector needs integer leaf_count");
            return result;
        }
    };
    if value.get("record_rule").and_then(Value::as_str) != Some("schema1_index") {
        result.mismatch("unsupported RFC6962 record_rule; expected schema1_index");
        return result;
    }
    let root = rfc6962_schema1_index_root(leaf_count);
    let root_hex = hex_encode(&root);
    result.leaf_count = Some(leaf_count);
    result.root = Some(format!("sha256:{root_hex}"));
    result.root_hex = Some(root_hex.clone());
    compare_hex(
        &mut result,
        "expected_root",
        &root_hex,
        value.get("expected_root"),
    );
    result
}

fn run_generic_ast_vectors(
    root: &Value,
    results: &mut Vec<ReplayVectorResult>,
) -> Result<(), String> {
    let vectors = if let Some(vectors) = root.as_array() {
        Some(vectors.as_slice())
    } else {
        root.get("vectors")
            .or_else(|| root.get("cases"))
            .or_else(|| root.get("golden_vectors"))
            .and_then(Value::as_array)
            .map(Vec::as_slice)
    };
    let Some(vectors) = vectors else {
        return Ok(());
    };
    for (index, value) in vectors.iter().enumerate() {
        let group = "vectors";
        let name = vector_name(value, group, index);
        let source = value
            .get("source_ast")
            .or_else(|| value.get("ast"))
            .or_else(|| value.get("input"))
            .unwrap_or(value);
        match canonicalize_source_json(source) {
            Ok(program) => results.push(accepted_program_result(group, name, &program)),
            Err(error) => results.push(rejected_result(group, name, error)),
        }
    }
    Ok(())
}

/// Replay the fixed shared golden-vector schema.
pub fn replay_golden_vectors(
    root: &Value,
    source_path: impl Into<String>,
) -> Result<ReplaySummary, String> {
    let mut metadata_errors = Vec::new();
    if let Some(observed) = root.get("freeze_version").and_then(Value::as_str) {
        if observed != "hegel-freeze-p2b-p3-v1.0.2" {
            metadata_errors.push(format!(
                "freeze_version mismatch: expected hegel-freeze-p2b-p3-v1.0.2, observed {observed}"
            ));
        }
    }
    if let Some(observed) = root.get("cbor_profile_id").and_then(Value::as_str) {
        if observed != CBOR_PROFILE_ID {
            metadata_errors.push(format!(
                "cbor_profile_id mismatch: expected {CBOR_PROFILE_ID}, observed {observed}"
            ));
        }
    }
    if let Some(observed) = root.get("ast_schema_id").and_then(Value::as_str) {
        if observed != AST_SCHEMA_ID {
            metadata_errors.push(format!(
                "ast_schema_id mismatch: expected {AST_SCHEMA_ID}, observed {observed}"
            ));
        }
    }

    let mut results = Vec::new();
    if root.is_object() {
        for (index, vector) in object_array(root, "cbor_encode_vectors")?
            .iter()
            .enumerate()
        {
            results.push(run_cbor_encode_vector(vector, index));
        }
        for (index, vector) in object_array(root, "cbor_reject_vectors")?
            .iter()
            .enumerate()
        {
            results.push(run_cbor_reject_vector(vector, index));
        }
        for (index, vector) in object_array(root, "ast_accept_vectors")?.iter().enumerate() {
            results.push(run_ast_accept_vector(vector, index));
        }
        for (index, vector) in object_array(root, "ast_reject_vectors")?.iter().enumerate() {
            results.push(run_ast_reject_vector(vector, index));
        }
        for (index, vector) in object_array(root, "rfc6962_vectors")?.iter().enumerate() {
            results.push(run_rfc6962_vector(vector, index));
        }
    }
    run_generic_ast_vectors(root, &mut results)?;

    if results.is_empty() {
        return Err("golden-vector fixture contains no recognized vector arrays".to_owned());
    }
    let passed_count = results
        .iter()
        .filter(|result| result.expectation_match)
        .count();
    let accepted_result_count = results
        .iter()
        .filter(|result| result.status == "ACCEPTED")
        .count();
    let rejected_result_count = results.len() - accepted_result_count;
    let failed_count = results.len() - passed_count + metadata_errors.len();
    Ok(ReplaySummary {
        schema_version: REPLAY_SCHEMA_VERSION,
        implementation: "rust",
        freeze_version: "hegel-freeze-p2b-p3-v1.0.2",
        cbor_profile_id: CBOR_PROFILE_ID,
        ast_schema_id: AST_SCHEMA_ID,
        source_path: source_path.into(),
        vector_count: results.len(),
        passed_count,
        failed_count,
        accepted_result_count,
        rejected_result_count,
        all_expectations_match: failed_count == 0,
        metadata_errors,
        results,
    })
}

pub fn replay_golden_vectors_file(path: &Path) -> Result<ReplaySummary, String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    let value: Value = serde_json::from_str(&text)
        .map_err(|error| format!("failed to parse {} as JSON: {error}", path.display()))?;
    replay_golden_vectors(&value, path.display().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn canonical(source: Value) -> CanonicalProgram {
        canonicalize_source_json(&source).expect("source AST should canonicalize")
    }

    #[test]
    fn strict_cbor_uses_shortest_integer_and_length_encodings() {
        let cases = [
            (json!(0), "00"),
            (json!(23), "17"),
            (json!(24), "1818"),
            (json!(255), "18ff"),
            (json!(256), "190100"),
            (json!(-1), "20"),
            (json!(-24), "37"),
            (json!(-25), "3818"),
            (json!(-256), "38ff"),
            (json!(-257), "390100"),
        ];
        for (value, expected) in cases {
            assert_eq!(
                hex_encode(&encode_strict_cbor_json(&value).unwrap()),
                expected
            );
        }
        assert_eq!(
            hex_encode(&encode_strict_cbor_byte_string(&[0; 23])),
            format!("57{}", "00".repeat(23))
        );
        assert_eq!(
            hex_encode(&encode_strict_cbor_byte_string(&[0; 24])),
            format!("5818{}", "00".repeat(24))
        );
    }

    #[test]
    fn strict_cbor_rejects_forbidden_and_noncanonical_encodings() {
        let cases = [
            ("f90000", REJECT_CBOR_FLOAT),
            ("6161", REJECT_CBOR_TEXT),
            ("a0", REJECT_CBOR_MAP),
            ("c000", REJECT_CBOR_TAG),
            ("9fff", REJECT_INDEFINITE_CBOR),
            ("0000", REJECT_TRAILING_CBOR),
            ("1800", REJECT_NONCANONICAL_CBOR),
        ];
        for (hex, expected_code) in cases {
            let error = validate_strict_cbor(&hex_decode(hex).unwrap()).unwrap_err();
            assert_eq!(error.code, expected_code, "case {hex}");
        }
    }

    #[test]
    fn strict_cbor_decoder_enforces_the_shared_nesting_limit() {
        let mut accepted = vec![0x81; MAX_CBOR_NESTING];
        accepted.push(0x00);
        validate_strict_cbor(&accepted).expect("64 nested arrays remain within the limit");

        let mut rejected = vec![0x81; MAX_CBOR_NESTING + 1];
        rejected.push(0x00);
        assert_eq!(
            validate_strict_cbor(&rejected).unwrap_err().code,
            REJECT_CBOR_NESTING
        );
    }

    #[test]
    fn explicit_bit_coercion_is_accepted_and_implicit_coercion_rejected() {
        let explicit = canonical(json!([
            "absolute",
            [
                "difference",
                ["bit_to_scalar", ["bit_at", 0]],
                ["bit_to_scalar", ["bit_at", 1]]
            ]
        ]));
        assert_eq!(explicit.output_sort, Sort::RationalValue);
        assert_eq!(explicit.depth, 3);
        assert_eq!(explicit.node_count, 6);
        assert_eq!(explicit.root_operator_id, 0x0102);

        let error = canonicalize_source_json(&json!(["difference", ["bit_at", 0], ["bit_at", 1]]))
            .unwrap_err();
        assert_eq!(error.code, REJECT_IMPLICIT_COERCION);
    }

    #[test]
    fn greater_equal_approx_zero_and_commutative_order_share_frozen_forms() {
        let greater = canonical(json!([
            "greater_equal",
            ["scalar_const", -1, 1],
            ["scalar_const", 1, 1]
        ]));
        assert_eq!(greater.canonical_cbor_hex(), "82018402038300000583000001");

        let approx_zero = canonical(json!([
            "approx_equal",
            ["scalar_const", -1, 1],
            ["scalar_const", 1, 1],
            0
        ]));
        let equal_reversed = canonical(json!([
            "equal_exact",
            ["scalar_const", 1, 1],
            ["scalar_const", -1, 1]
        ]));
        assert_eq!(approx_zero.canonical_cbor, equal_reversed.canonical_cbor);
        assert_eq!(approx_zero.root_operator_id, 0x0202);
    }

    #[test]
    fn add_is_flattened_sorted_and_right_associated() {
        let program = canonical(json!([
            "add",
            [
                "add",
                ["bit_to_scalar", ["bit_at", 0]],
                ["scalar_const", 1, 2]
            ],
            ["bit_to_scalar", ["bit_at", 1]]
        ]));
        assert_eq!(
            program.canonical_cbor_hex(),
            "8201840200830100830001018402008300000483010083000100"
        );
        assert_eq!(program.node_count, 7);
        assert_eq!(program.depth, 3);

        let zero_removed = canonical(json!([
            "add",
            ["bit_to_scalar", ["bit_at", 0]],
            ["scalar_const", 0, 1]
        ]));
        let child = canonical(json!(["bit_to_scalar", ["bit_at", 0]]));
        assert_eq!(zero_removed.canonical_cbor, child.canonical_cbor);
    }

    #[test]
    fn local_constant_and_absolute_rewrites_reach_a_fixed_point() {
        let difference = canonical(json!([
            "difference",
            ["scalar_const", 1, 1],
            ["scalar_const", 1, 1]
        ]));
        assert_eq!(
            difference.canonical_node,
            Node::ScalarConst(ZERO_PARAMETER_INDEX)
        );

        let nested_absolute = canonical(json!(["absolute", ["absolute", ["scalar_const", -1, 1]]]));
        assert_eq!(nested_absolute.canonical_node, Node::ScalarConst(5));

        // Source rational spelling is normalized onto the frozen registry.
        let reduced = canonical(json!(["scalar_const", 2, 2]));
        assert_eq!(reduced.canonical_node, Node::ScalarConst(5));
    }

    #[test]
    fn and_scope_and_inline_tolerance_accounting_are_exact() {
        let duplicate = canonical(json!([
            "top_level_AND",
            [
                "equal_exact",
                ["scalar_const", -1, 1],
                ["scalar_const", 1, 1]
            ],
            [
                "equal_exact",
                ["scalar_const", 1, 1],
                ["scalar_const", -1, 1]
            ]
        ]));
        assert_eq!(duplicate.root_operator_id, 0x0202);
        assert_eq!(duplicate.node_count, 3);

        let aggregate = canonical(json!([
            "aggregate",
            "sum_v1",
            "scope_primary_only_v1",
            "q0",
            [["c1", false], ["c0", true]]
        ]));
        assert_eq!(aggregate.node_count, 1);
        assert_eq!(aggregate.depth, 0);
        assert_eq!(aggregate.aggregate_leaf_count, 1);
        assert_eq!(
            aggregate.canonical_cbor_hex(),
            "8201860003000100828200f58201f4"
        );

        let approximate = canonical(json!([
            "approx_equal",
            ["scalar_const", -1, 1],
            ["scalar_const", 1, 1],
            1
        ]));
        assert_eq!(approximate.node_count, 3);
        assert_eq!(approximate.depth, 1);
    }

    #[test]
    fn aliases_new_symbols_and_structural_overflow_fail_closed() {
        let text_slot = canonicalize_source_json(&json!(["bit_at", "e0"])).unwrap_err();
        assert_eq!(text_slot.code, REJECT_REGISTRY_INDEX_OUT_OF_RANGE);

        let omitted_extension = canonicalize_source_json(&json!([
            "aggregate",
            "sum_v1",
            "scope_primary_only_v1",
            "q0"
        ]))
        .unwrap_err();
        assert_eq!(omitted_extension.code, REJECT_MALFORMED_SOURCE_AST);

        let alias = canonicalize_source_json(&json!([
            "aggregate",
            "sum_v1",
            "control_volume_primary_only_v1",
            "q0",
            []
        ]))
        .unwrap_err();
        assert_eq!(alias.code, REJECT_NONCANONICAL_SCOPE_ALIAS);

        let new_symbol = canonicalize_source_json(&json!(["new_symbol_call", 0])).unwrap_err();
        assert_eq!(new_symbol.code, REJECT_NEW_SYMBOL_IN_OLD_DSL);

        let clauses = canonicalize_source_json(&json!([
            "top_level_AND",
            ["context_flag", "c0"],
            ["context_flag", "c1"],
            ["context_flag", "c2"],
            ["context_flag", "c3"]
        ]))
        .unwrap_err();
        assert_eq!(clauses.code, REJECT_STRUCTURAL_LIMIT);

        let two_aggregates = canonicalize_source_json(&json!([
            "add",
            ["aggregate", "sum_v1", "scope_all_observed_v1", "q0", []],
            ["aggregate", "mean_v1", "scope_all_observed_v1", "q0", []]
        ]))
        .unwrap_err();
        assert_eq!(two_aggregates.code, REJECT_STRUCTURAL_LIMIT);

        let four_parameters = canonicalize_source_json(&json!([
            "top_level_AND",
            [
                "equal_exact",
                ["scalar_const", -2, 1],
                ["scalar_const", -1, 1]
            ],
            ["less_equal", ["scalar_const", 1, 2], ["scalar_const", 1, 1]]
        ]))
        .unwrap_err();
        assert_eq!(four_parameters.code, REJECT_STRUCTURAL_LIMIT);
    }

    #[test]
    fn strict_ast_decoder_requires_exact_canonical_reencode() {
        let canonical_program = canonical(json!([
            "equal_exact",
            ["scalar_const", 1, 1],
            ["scalar_const", -1, 1]
        ]));
        let decoded = decode_strict_canonical_ast(&canonical_program.canonical_cbor).unwrap();
        assert_eq!(
            decoded.canonical_ast_hash,
            canonical_program.canonical_ast_hash
        );

        let noncanonical = Node::Binary {
            op: BinaryOp::EqualExact,
            left: Box::new(Node::ScalarConst(5)),
            right: Box::new(Node::ScalarConst(1)),
        };
        let noncanonical_bytes = encode_ast_envelope(&noncanonical);
        assert_ne!(noncanonical_bytes, canonical_program.canonical_cbor);
        assert_eq!(
            decode_strict_canonical_ast(&noncanonical_bytes)
                .unwrap_err()
                .code,
            REJECT_NONCANONICAL_AST
        );

        let unsorted_scope = Node::Aggregate {
            map_id: 0,
            scope_id: 1,
            quantity_id: 0,
            scope_extension: vec![(1, false), (0, true)],
        };
        assert_eq!(
            decode_strict_canonical_ast(&encode_ast_envelope(&unsorted_scope))
                .unwrap_err()
                .code,
            REJECT_NONCANONICAL_AST
        );
    }

    #[test]
    fn rfc6962_uses_largest_power_of_two_split_without_duplicate_last() {
        let expected = [
            (
                0,
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            ),
            (
                1,
                "104a497cc8d5ab0a833783fe2a9fb3ce5ef54dc8197c29863f30e3d35a620521",
            ),
            (
                3,
                "dc40076051b08b7998cda3f81d7eb49a930ecad1f4b04ec9cdceec7f0c071b09",
            ),
            (
                5,
                "3d58d6cd64ceb3e5cdee86613b3879480115573ad34a25b822143c373c2f12cf",
            ),
        ];
        for (leaf_count, root) in expected {
            assert_eq!(hex_encode(&rfc6962_schema1_index_root(leaf_count)), root);
        }
    }

    #[test]
    fn shared_cross_language_golden_fixture_replays_without_mismatch() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../golden_vectors/strict_ast_cbor_v1.json"
        ))
        .unwrap();
        let summary = replay_golden_vectors(&fixture, "embedded-shared-fixture").unwrap();
        assert_eq!(summary.vector_count, 48);
        assert_eq!(summary.failed_count, 0);
        assert!(summary.all_expectations_match);
    }

    #[test]
    fn capacity_replay_matches_the_frozen_dual_replay_commitments() {
        let report = replay_capacity_subset().unwrap();
        assert_eq!(report.source_count, 64_680);
        assert_eq!(report.accepted_total_count, 64_680);
        assert_eq!(report.accepted_unique_count, 64_680);
        assert_eq!(report.rejected_count, 0);
        assert_eq!(report.rewrite_collapsed_count, 0);
        assert_eq!(
            report.accepted_set_commitment,
            "sha256:c1a02a66a8d6d8f75204cb3daf03ab0b01c2b3b8e486d0ab3d481ee3be43c930"
        );
        assert_eq!(report.first_accepted_out_of_budget_ordinal, Some(50_001));
        assert_eq!(
            report.first_accepted_out_of_budget_hash.as_deref(),
            Some("sha256:7c7f786c2cc57d31506b3c61d162d175c7f69a2878a089c72c9d053694cba948")
        );
        assert_eq!(report.executed_closure_status, "NOT_RUN");
        assert!(!report.dsl_too_large_claim_allowed);
    }
}
