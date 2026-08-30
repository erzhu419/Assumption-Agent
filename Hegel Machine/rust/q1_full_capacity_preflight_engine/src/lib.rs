//! Target-blind Rust capacity diagnostic for the full v1.6 quotient surface.
//!
//! This is deliberately not a Q1 execution engine.  It does not accept target,
//! split, or role inputs and cannot create Q1 roots, receipts, gate rows, or
//! output slots.  The only full-run success status is the preregistered
//! `PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY` observation.

mod frozen_v1;

pub use frozen_v1::{
    FrozenFormalChunkBinding, FrozenFormalMaterializedPartition,
    FrozenFormalMaterializedStream, FrozenFormalStreamProjection, FrozenFormalWireProjection,
};

use hegel_strict_canonicalizer::{BinaryOp, CanonicalProgram, Node, Sort, UnaryOp};
use hegel_strict_canonicalizer_shrink6::canonicalize_shrink6_source_node;
use serde::Serialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub const SCHEMA_VERSION: &str = "hegel-phase3a-q1-full-node6-capacity-diagnostic/1";
pub const ENDPOINT_SCHEMA_VERSION: &str =
    "hegel-phase3a-q1-rust-full-node6-capacity-endpoint/1";
pub const IMPLEMENTATION_ID: &str = "hegel-rust-q1-full-capacity-preflight-engine-v1";
pub const ACTOR_ID: &str = "RUST_ENDPOINT";
pub const ACTION_ID: &str = "full-node6-capacity-preflight-v1";
pub const FORMAL_MATERIALIZER_SCHEMA_VERSION: &str =
    "hegel-phase3a-q1-full-node6-formal-archive-materializer/1";
pub const FORMAL_MATERIALIZER_ACTOR_ID: &str = "rust";
pub const FORMAL_MATERIALIZER_IMPLEMENTATION_ID: &str =
    "hegel-rust-q1-formal-archive-materializer-v1";
pub const FORMAL_MATERIALIZER_ACTION_ID: &str =
    "materialize-full-node6-formal-archive-v1";
pub const FORMAL_MATERIALIZER_STATUS: &str =
    "FULL_NODE6_FORMAL_ARCHIVE_MATERIALIZED_Q1_NOT_RUN";
pub const FORMAL_OUTPUT_BYTES_LIMIT: u64 = 16 * 1024 * 1024;
pub const FORMAL_SCRATCH_BYTES_LIMIT: u64 = 4 * 1024 * 1024;
pub const DSL_VERSION: &str = "hegel-old-dsl-v1.6.0";
pub const CLOSURE_SEMANTICS_VERSION: &str = "hegel-quotient-closure-v1.0.1";
pub const PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY: &str =
    "PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY";
pub const PREFLIGHT_CAPACITY_GUARD_HIT: &str = "PREFLIGHT_CAPACITY_GUARD_HIT";
pub const LOCAL_PROTOTYPE_SUBSET_TRAVERSAL_CLOSED: &str =
    "LOCAL_PROTOTYPE_SUBSET_TRAVERSAL_CLOSED";
pub const Q1_STATE: &str = "NOT_RUN";
pub const FROZEN_LEAF_COUNT: usize = 810;
pub const FULL_MAXIMUM_AST_DEPTH: u32 = 3;
pub const FULL_MAXIMUM_AST_NODE_COUNT: u32 = 6;
pub const ODD_UNIVERSE_ROOT: &str =
    "b7e6eed1174ceee1944bd540cbb0ace4c5c7fc0dffea79dd956a51f7a2410a05";
pub const SINK_UNIVERSE_ROOT: &str =
    "1a46f9967ad48df6ec1d9be609de701413ce86171fb0f92495a982bef0f40ff5";
pub const SIDECAR_NAMES: [&str; 5] = [
    "diagnostic-summary-v1.json",
    "projection-summary-v1.json",
    "coverage-summary-v1.json",
    "fixed-point-summary-v1.json",
    "endpoint-manifest-v1.json",
];

const MAX_AGGREGATE_LEAVES: u32 = 1;
const MAX_SCALAR_OCCURRENCES: u32 = 3;
const MAX_SCOPE_CLAUSES: u32 = 2;
const MAX_DISTINCT_BITS: u32 = 4;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineError {
    pub code: String,
    pub detail: String,
}

impl EngineError {
    fn new(code: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            detail: detail.into(),
        }
    }
}

impl fmt::Display for EngineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.code, self.detail)
    }
}

impl std::error::Error for EngineError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CapacityLimits {
    pub maximum_ast_depth: u32,
    pub maximum_ast_node_count: u32,
    pub maximum_raw_operator_applications: u64,
    pub maximum_behavior_classes: u64,
    pub maximum_visible_frontier_points: u64,
    pub maximum_visible_frontier_points_per_class: u64,
    pub maximum_continuation_bank_points: u64,
    pub maximum_continuation_bank_points_per_class: u64,
    pub maximum_work_queue_points: u64,
    pub maximum_saturation_rounds: u64,
    pub maximum_wall_time_seconds: u64,
}

impl CapacityLimits {
    pub const fn full_node6() -> Self {
        Self {
            maximum_ast_depth: 3,
            maximum_ast_node_count: 6,
            maximum_raw_operator_applications: 4_294_967_295,
            maximum_behavior_classes: 16_777_216,
            maximum_visible_frontier_points: 33_554_432,
            maximum_visible_frontier_points_per_class: 65_536,
            maximum_continuation_bank_points: 67_108_864,
            maximum_continuation_bank_points_per_class: 65_536,
            maximum_work_queue_points: 67_108_864,
            maximum_saturation_rounds: 16,
            maximum_wall_time_seconds: 172_800,
        }
    }

    pub const fn q1_formal_odd() -> Self {
        Self {
            maximum_ast_depth: 3,
            maximum_ast_node_count: 6,
            maximum_raw_operator_applications: 32_768,
            maximum_behavior_classes: 1_024,
            maximum_visible_frontier_points: 4_096,
            maximum_visible_frontier_points_per_class: 32,
            maximum_continuation_bank_points: 8_192,
            maximum_continuation_bank_points_per_class: 4_096,
            maximum_work_queue_points: 16_384,
            maximum_saturation_rounds: 7,
            maximum_wall_time_seconds: 3_600,
        }
    }

    pub const fn q1_formal_sink() -> Self {
        Self {
            maximum_ast_depth: 3,
            maximum_ast_node_count: 6,
            maximum_raw_operator_applications: 32_768,
            maximum_behavior_classes: 256,
            maximum_visible_frontier_points: 1_024,
            maximum_visible_frontier_points_per_class: 256,
            maximum_continuation_bank_points: 8_192,
            maximum_continuation_bank_points_per_class: 4_096,
            maximum_work_queue_points: 16_384,
            maximum_saturation_rounds: 7,
            maximum_wall_time_seconds: 3_600,
        }
    }

    pub const fn q1_formal_schedule() -> [Self; 2] {
        [Self::q1_formal_odd(), Self::q1_formal_sink()]
    }

    pub fn bounded_for_test(maximum_ast_node_count: u32) -> Result<Self, EngineError> {
        if !(1..=6).contains(&maximum_ast_node_count) {
            return Err(EngineError::new(
                "REJECT_PREFLIGHT_LIMITS",
                "maximum_ast_node_count must be in 1..=6",
            ));
        }
        Ok(Self {
            maximum_ast_node_count,
            ..Self::full_node6()
        })
    }

    fn validate(self) -> Result<Self, EngineError> {
        if self.maximum_ast_depth > FULL_MAXIMUM_AST_DEPTH
            || !(1..=FULL_MAXIMUM_AST_NODE_COUNT).contains(&self.maximum_ast_node_count)
            || self.maximum_raw_operator_applications == 0
            || self.maximum_behavior_classes == 0
            || self.maximum_visible_frontier_points == 0
            || self.maximum_visible_frontier_points_per_class == 0
            || self.maximum_continuation_bank_points == 0
            || self.maximum_continuation_bank_points_per_class == 0
            || self.maximum_work_queue_points == 0
            || self.maximum_saturation_rounds == 0
            || self.maximum_wall_time_seconds == 0
        {
            return Err(EngineError::new(
                "REJECT_PREFLIGHT_LIMITS",
                "capacity limits are outside the frozen diagnostic envelope",
            ));
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GuardHit {
    id: u8,
    name: &'static str,
}

impl GuardHit {
    const fn raw() -> Self {
        Self {
            id: 1,
            name: "RAW_OPERATOR_APPLICATIONS",
        }
    }

    const fn classes() -> Self {
        Self {
            id: 2,
            name: "BEHAVIOR_CLASSES",
        }
    }

    const fn frontier_total() -> Self {
        Self {
            id: 3,
            name: "VISIBLE_FRONTIER_TOTAL",
        }
    }

    const fn frontier_per_class() -> Self {
        Self {
            id: 4,
            name: "VISIBLE_FRONTIER_PER_CLASS",
        }
    }

    const fn bank_total() -> Self {
        Self {
            id: 5,
            name: "CONTINUATION_BANK_TOTAL",
        }
    }

    const fn bank_per_class() -> Self {
        Self {
            id: 6,
            name: "CONTINUATION_BANK_PER_CLASS",
        }
    }

    const fn work_queue() -> Self {
        Self {
            id: 7,
            name: "WORK_QUEUE_POINTS",
        }
    }

    const fn rounds() -> Self {
        Self {
            id: 8,
            name: "SATURATION_ROUNDS",
        }
    }

    const fn wall() -> Self {
        Self {
            id: 12,
            name: "WALL_TIME",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Rational {
    numerator: i64,
    denominator: i64,
}

impl Rational {
    fn new(numerator: i64, denominator: i64) -> Result<Self, EngineError> {
        if denominator == 0 {
            return Err(EngineError::new("FAIL_SEMANTICS_MISMATCH", "zero denominator"));
        }
        let (mut numerator, mut denominator) = (numerator, denominator);
        if denominator < 0 {
            numerator = numerator.checked_neg().ok_or_else(|| {
                EngineError::new("FAIL_SEMANTICS_MISMATCH", "rational overflow")
            })?;
            denominator = denominator.checked_neg().ok_or_else(|| {
                EngineError::new("FAIL_SEMANTICS_MISMATCH", "rational overflow")
            })?;
        }
        let divisor = gcd(numerator.unsigned_abs(), denominator as u64) as i64;
        Ok(Self {
            numerator: numerator / divisor,
            denominator: denominator / divisor,
        })
    }

    const fn integer(value: i64) -> Self {
        Self {
            numerator: value,
            denominator: 1,
        }
    }

    fn add(self, other: Self) -> Option<Self> {
        let numerator = (self.numerator as i128)
            .checked_mul(other.denominator as i128)?
            .checked_add((other.numerator as i128).checked_mul(self.denominator as i128)?)?;
        let denominator = (self.denominator as i128).checked_mul(other.denominator as i128)?;
        Self::new(i64::try_from(numerator).ok()?, i64::try_from(denominator).ok()?).ok()
    }

    fn difference(self, other: Self) -> Option<Self> {
        self.add(Self::new(other.numerator.checked_neg()?, other.denominator).ok()?)
    }

    fn absolute(self) -> Option<Self> {
        Self::new(self.numerator.checked_abs()?, self.denominator).ok()
    }

    fn compare(self, other: Self) -> Ordering {
        ((self.numerator as i128) * (other.denominator as i128))
            .cmp(&((other.numerator as i128) * (self.denominator as i128)))
    }

    fn in_grid(self) -> bool {
        self.numerator.abs() <= 64 && (1..=8).contains(&self.denominator)
    }
}

fn gcd(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        (left, right) = (right, left % right);
    }
    left.max(1)
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum RuntimeValue {
    Bottom,
    Bool(bool),
    Bit(u8),
    Sign(i8),
    BoundedInt(i8),
    Rational(Rational),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Observation {
    Odd(Vec<u8>),
    Sink([Rational; 4]),
}

fn odd_universe() -> Vec<Observation> {
    let mut observations = Vec::with_capacity(480);
    for size in 5_usize..=8 {
        for numeric in 0_u16..(1_u16 << size) {
            observations.push(Observation::Odd(
                (0..size)
                    .map(|offset| ((numeric >> (size - offset - 1)) & 1) as u8)
                    .collect(),
            ));
        }
    }
    observations
}

fn sink_universe() -> Vec<Observation> {
    let mut observations = Vec::with_capacity(85);
    for a in 0_i8..=4 {
        for b in 0_i8..=4 {
            for c in 0_i8..=4 {
                let d = a + b - c;
                if (0..=4).contains(&d) {
                    observations.push(Observation::Sink([
                        Rational::integer(a as i64),
                        Rational::integer(b as i64),
                        Rational::integer(c as i64),
                        Rational::integer(d as i64),
                    ]));
                }
            }
        }
    }
    observations
}

fn universe(input_signature_id: u64) -> Result<(Vec<Observation>, &'static str), EngineError> {
    match input_signature_id {
        1 => Ok((odd_universe(), ODD_UNIVERSE_ROOT)),
        2 => Ok((sink_universe(), SINK_UNIVERSE_ROOT)),
        _ => Err(EngineError::new(
            "REJECT_PREFLIGHT_INPUT_SIGNATURE",
            "input_signature_id must be 1 or 2",
        )),
    }
}

fn bounded_rational(value: Rational) -> RuntimeValue {
    if value.in_grid() {
        RuntimeValue::Rational(value)
    } else {
        RuntimeValue::Bottom
    }
}

impl Observation {
    fn set_size(&self) -> i8 {
        match self {
            Self::Odd(bits) => bits.len() as i8,
            Self::Sink(_) => 4,
        }
    }

    fn bit_at(&self, index: u64) -> RuntimeValue {
        match self {
            Self::Odd(bits) => bits
                .get(index as usize)
                .copied()
                .map(RuntimeValue::Bit)
                .unwrap_or(RuntimeValue::Bottom),
            Self::Sink(_) => RuntimeValue::Bottom,
        }
    }

    fn aggregate(
        &self,
        map_id: u64,
        scope_id: u64,
        quantity_id: u64,
        extension: &[(u64, bool)],
    ) -> RuntimeValue {
        let Self::Sink(values) = self else {
            return RuntimeValue::Bottom;
        };
        if scope_id != 3 || quantity_id != 0 || !extension.is_empty() {
            return RuntimeValue::Bottom;
        }
        match map_id {
            0 => values
                .iter()
                .try_fold(Rational::integer(0), |sum, value| sum.add(*value))
                .map(bounded_rational)
                .unwrap_or(RuntimeValue::Bottom),
            1 => RuntimeValue::BoundedInt(
                values.iter().filter(|value| value.numerator != 0).count() as i8,
            ),
            5 => [1_i64, 1, -1, -1]
                .into_iter()
                .zip(values)
                .try_fold(Rational::integer(0), |sum, (orientation, value)| {
                    sum.add(Rational::new(value.numerator * orientation, value.denominator).ok()?)
                })
                .map(bounded_rational)
                .unwrap_or(RuntimeValue::Bottom),
            _ => RuntimeValue::Bottom,
        }
    }
}

fn evaluate_leaf(node: &Node, observation: &Observation) -> Result<RuntimeValue, EngineError> {
    match node {
        Node::ScalarConst(1) => Ok(RuntimeValue::Rational(Rational::integer(-1))),
        Node::ScalarConst(3) => Ok(RuntimeValue::Rational(Rational::integer(0))),
        Node::ScalarConst(5) => Ok(RuntimeValue::Rational(Rational::integer(1))),
        Node::BitAt(index) if *index < 8 => Ok(observation.bit_at(*index)),
        Node::SetSize => Ok(RuntimeValue::BoundedInt(observation.set_size())),
        Node::Aggregate {
            map_id,
            scope_id,
            quantity_id,
            scope_extension,
        } if [0, 1, 5].contains(map_id) => Ok(observation.aggregate(
            *map_id,
            *scope_id,
            *quantity_id,
            scope_extension,
        )),
        Node::ContextFlag(id) if *id < 4 => Ok(RuntimeValue::Bottom),
        Node::TaskFlag(id) if *id < 2 => Ok(RuntimeValue::Bottom),
        _ => Err(EngineError::new(
            "FAIL_SEMANTICS_MISMATCH",
            "non-leaf or inactive leaf reached leaf evaluator",
        )),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[repr(u8)]
pub enum OutputSortId {
    Bool = 1,
    Bit = 2,
    Sign = 3,
    BoundedInt = 4,
    RationalValue = 5,
}

impl OutputSortId {
    fn from_sort(sort: Sort) -> Self {
        match sort {
            Sort::Bool => Self::Bool,
            Sort::Bit => Self::Bit,
            Sort::Sign => Self::Sign,
            Sort::BoundedInt => Self::BoundedInt,
            Sort::RationalValue => Self::RationalValue,
        }
    }
}

#[derive(Debug, Clone)]
struct Behavior {
    cells: Arc<Vec<RuntimeValue>>,
    key: Arc<Vec<u8>>,
    digest: [u8; 32],
}

fn append_runtime_value(output: &mut Vec<u8>, value: &RuntimeValue) {
    match value {
        RuntimeValue::Bottom => output.push(0),
        RuntimeValue::Bool(value) => output.extend([1, u8::from(*value)]),
        RuntimeValue::Bit(value) => output.extend([2, *value]),
        RuntimeValue::Sign(value) => output.extend([3, (*value + 1) as u8]),
        RuntimeValue::BoundedInt(value) => output.extend([4, (*value + 8) as u8]),
        RuntimeValue::Rational(value) => {
            output.push(5);
            output.extend_from_slice(&value.numerator.to_be_bytes());
            output.extend_from_slice(&value.denominator.to_be_bytes());
        }
    }
}

fn sha256_bytes(parts: &[&[u8]]) -> [u8; 32] {
    let mut digest = Sha256::new();
    for part in parts {
        digest.update(part);
    }
    digest.finalize().into()
}

impl Behavior {
    fn new(input_signature_id: u64, sort: OutputSortId, cells: Arc<Vec<RuntimeValue>>) -> Self {
        let mut key = Vec::with_capacity(16 + cells.len() * 2);
        key.extend_from_slice(b"HEGEL/Q1/BEHAVIOR/DIAGNOSTIC/V1\0");
        key.extend_from_slice(&input_signature_id.to_be_bytes());
        key.push(sort as u8);
        key.extend_from_slice(&(cells.len() as u64).to_be_bytes());
        for cell in cells.iter() {
            append_runtime_value(&mut key, cell);
        }
        let digest = sha256_bytes(&[&key]);
        Self {
            cells,
            key: Arc::new(key),
            digest,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
enum NormalizationProfile {
    General = 0,
    AbsoluteRoot = 1,
    ConstNegativeOne = 2,
    ConstZero = 3,
    ConstPositiveOne = 4,
    TopLevelAnd2 = 5,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ConstructionSignature {
    output_sort: OutputSortId,
    ast_depth: u32,
    ast_node_count: u32,
    scalar_occurrence_count: u32,
    aggregate_leaf_count: u32,
    distinct_bit_slot_bitmap: u8,
    scope_clause_count: u32,
    top_level_clause_count: u32,
    old_law_depth: u32,
    normalization_profile: NormalizationProfile,
    mdl_q32: u64,
}

impl ConstructionSignature {
    fn bytes(&self) -> Vec<u8> {
        let mut output = Vec::with_capacity(64);
        output.extend_from_slice(b"HEGEL/Q1/CONSTRUCTION_SIGNATURE/DIAGNOSTIC/V1\0");
        output.push(self.output_sort as u8);
        for value in [
            self.ast_depth,
            self.ast_node_count,
            self.scalar_occurrence_count,
            self.aggregate_leaf_count,
            u32::from(self.distinct_bit_slot_bitmap),
            self.scope_clause_count,
            self.top_level_clause_count,
            self.old_law_depth,
        ] {
            output.extend_from_slice(&value.to_be_bytes());
        }
        output.push(self.normalization_profile as u8);
        output.extend_from_slice(&self.mdl_q32.to_be_bytes());
        output
    }

    fn dominates(&self, other: &Self) -> bool {
        if self.output_sort != other.output_sort
            || self.normalization_profile != other.normalization_profile
        {
            return false;
        }
        let subset = (self.distinct_bit_slot_bitmap | other.distinct_bit_slot_bitmap)
            == other.distinct_bit_slot_bitmap;
        let left = [
            u64::from(self.ast_depth),
            u64::from(self.ast_node_count),
            u64::from(self.scalar_occurrence_count),
            u64::from(self.aggregate_leaf_count),
            u64::from(self.scope_clause_count),
            u64::from(self.top_level_clause_count),
            u64::from(self.old_law_depth),
            self.mdl_q32,
        ];
        let right = [
            u64::from(other.ast_depth),
            u64::from(other.ast_node_count),
            u64::from(other.scalar_occurrence_count),
            u64::from(other.aggregate_leaf_count),
            u64::from(other.scope_clause_count),
            u64::from(other.top_level_clause_count),
            u64::from(other.old_law_depth),
            other.mdl_q32,
        ];
        let no_worse = subset && left.iter().zip(right).all(|(left, right)| *left <= right);
        let strict = self.distinct_bit_slot_bitmap != other.distinct_bit_slot_bitmap
            || left.iter().zip(right).any(|(left, right)| *left < right);
        no_worse && strict
    }
}

fn elias_delta_length(value: u64) -> Result<u64, EngineError> {
    if value == 0 {
        return Err(EngineError::new(
            "FAIL_SEMANTICS_MISMATCH",
            "Elias-delta value is one-based",
        ));
    }
    let log_n = 63 - u64::from(value.leading_zeros());
    let log_log = 63 - u64::from((log_n + 1).leading_zeros());
    Ok(log_n + 2 * log_log + 1)
}

fn mdl_bits(node: &Node) -> Result<u64, EngineError> {
    match node {
        Node::ScalarConst(id) if [1, 3, 5].contains(id) => Ok(8),
        Node::BitAt(id) if *id < 8 => Ok(5 + elias_delta_length(id + 1)?),
        Node::SetSize => Ok(5),
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
            Ok(11 + clause_count_bits + 3 * scope_extension.len() as u64)
        }
        Node::ContextFlag(id) if *id < 4 => Ok(5 + elias_delta_length(id + 1)?),
        Node::TaskFlag(id) if *id < 2 => Ok(5 + elias_delta_length(id + 1)?),
        Node::Unary { child, .. } => Ok(4 + mdl_bits(child)?),
        Node::Binary { left, right, .. } => Ok(5 + mdl_bits(left)? + mdl_bits(right)?),
        Node::ApproxEqual {
            left,
            right,
            tolerance_index,
        } if (1..=2).contains(tolerance_index) => {
            Ok(6 + mdl_bits(left)? + mdl_bits(right)?)
        }
        Node::And(children) if children.len() == 2 => Ok(5
            + children
                .iter()
                .map(mdl_bits)
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .sum::<u64>()),
        _ => Err(EngineError::new(
            "FAIL_SEMANTICS_MISMATCH",
            "AST has no frozen MDL code",
        )),
    }
}

#[derive(Debug, Default)]
struct ResourceMetrics {
    bitmask: u8,
    scope_clauses: u32,
}

fn collect_metrics(node: &Node, metrics: &mut ResourceMetrics) {
    match node {
        Node::BitAt(index) if *index < 8 => metrics.bitmask |= 1 << *index,
        Node::Aggregate {
            scope_extension, ..
        } => metrics.scope_clauses += scope_extension.len() as u32,
        Node::Unary { child, .. } => collect_metrics(child, metrics),
        Node::Binary { left, right, .. } | Node::ApproxEqual { left, right, .. } => {
            collect_metrics(left, metrics);
            collect_metrics(right, metrics);
        }
        Node::And(children) => {
            for child in children {
                collect_metrics(child, metrics);
            }
        }
        _ => {}
    }
}

fn normalization_profile(node: &Node) -> NormalizationProfile {
    match node {
        Node::Unary {
            op: UnaryOp::Absolute,
            ..
        } => NormalizationProfile::AbsoluteRoot,
        Node::ScalarConst(1) => NormalizationProfile::ConstNegativeOne,
        Node::ScalarConst(3) => NormalizationProfile::ConstZero,
        Node::ScalarConst(5) => NormalizationProfile::ConstPositiveOne,
        Node::And(children) if children.len() == 2 => NormalizationProfile::TopLevelAnd2,
        _ => NormalizationProfile::General,
    }
}

fn construction_signature(
    program: &CanonicalProgram,
) -> Result<ConstructionSignature, EngineError> {
    let mut metrics = ResourceMetrics::default();
    collect_metrics(&program.canonical_node, &mut metrics);
    Ok(ConstructionSignature {
        output_sort: OutputSortId::from_sort(program.output_sort),
        ast_depth: program.depth,
        ast_node_count: program.node_count,
        scalar_occurrence_count: program.scalar_parameter_occurrence_count,
        aggregate_leaf_count: program.aggregate_leaf_count,
        distinct_bit_slot_bitmap: metrics.bitmask,
        scope_clause_count: metrics.scope_clauses,
        top_level_clause_count: match &program.canonical_node {
            Node::And(children) => children.len() as u32,
            _ => 0,
        },
        old_law_depth: 0,
        normalization_profile: normalization_profile(&program.canonical_node),
        mdl_q32: mdl_bits(&program.canonical_node)? << 32,
    })
}

#[derive(Debug)]
struct Program {
    canonical: CanonicalProgram,
    behavior: Behavior,
    signature: ConstructionSignature,
}

type ProgramRef = Arc<Program>;

impl Program {
    fn from_leaf(
        canonical: CanonicalProgram,
        input_signature_id: u64,
        observations: &[Observation],
    ) -> Result<ProgramRef, EngineError> {
        let signature = construction_signature(&canonical)?;
        let cells = observations
            .iter()
            .map(|observation| evaluate_leaf(&canonical.canonical_node, observation))
            .collect::<Result<Vec<_>, _>>()?;
        let behavior = Behavior::new(input_signature_id, signature.output_sort, Arc::new(cells));
        Ok(Arc::new(Self {
            canonical,
            behavior,
            signature,
        }))
    }

    fn from_application(
        canonical: CanonicalProgram,
        input_signature_id: u64,
        cells: Arc<Vec<RuntimeValue>>,
    ) -> Result<ProgramRef, EngineError> {
        let signature = construction_signature(&canonical)?;
        let behavior = Behavior::new(input_signature_id, signature.output_sort, cells);
        Ok(Arc::new(Self {
            canonical,
            behavior,
            signature,
        }))
    }
}

fn scope_extensions() -> Vec<Vec<(u64, bool)>> {
    let mut extensions = vec![Vec::new()];
    for context in 0_u64..4 {
        for expected in [false, true] {
            extensions.push(vec![(context, expected)]);
        }
    }
    for first in 0_u64..4 {
        for second in (first + 1)..4 {
            for first_expected in [false, true] {
                for second_expected in [false, true] {
                    extensions.push(vec![(first, first_expected), (second, second_expected)]);
                }
            }
        }
    }
    debug_assert_eq!(extensions.len(), 33);
    extensions
}

fn full_leaf_sources() -> Vec<Node> {
    let mut leaves = vec![
        Node::ScalarConst(1),
        Node::ScalarConst(3),
        Node::ScalarConst(5),
    ];
    leaves.extend((0_u64..8).map(Node::BitAt));
    leaves.push(Node::SetSize);
    let extensions = scope_extensions();
    for map_id in [0_u64, 1, 5] {
        for scope_id in 0_u64..4 {
            for quantity_id in 0_u64..2 {
                for scope_extension in &extensions {
                    leaves.push(Node::Aggregate {
                        map_id,
                        scope_id,
                        quantity_id,
                        scope_extension: scope_extension.clone(),
                    });
                }
            }
        }
    }
    leaves.extend((0_u64..4).map(Node::ContextFlag));
    leaves.extend((0_u64..2).map(Node::TaskFlag));
    debug_assert_eq!(leaves.len(), FROZEN_LEAF_COUNT);
    leaves
}

fn frozen_leaf_manifest() -> Result<Vec<CanonicalProgram>, EngineError> {
    let mut leaves = full_leaf_sources()
        .into_iter()
        .map(|source| {
            canonicalize_shrink6_source_node(source)
                .map_err(|error| EngineError::new(error.code, error.message))
        })
        .collect::<Result<Vec<_>, _>>()?;
    leaves.sort_by(|left, right| {
        (OutputSortId::from_sort(left.output_sort) as u8)
            .cmp(&(OutputSortId::from_sort(right.output_sort) as u8))
            .then_with(|| left.root_operator_id.cmp(&right.root_operator_id))
            .then_with(|| left.canonical_cbor.cmp(&right.canonical_cbor))
    });
    leaves.dedup_by(|left, right| left.canonical_cbor == right.canonical_cbor);
    if leaves.len() != FROZEN_LEAF_COUNT {
        return Err(EngineError::new(
            "FAIL_SEMANTICS_MISMATCH",
            format!("full leaf manifest has {} rows", leaves.len()),
        ));
    }
    Ok(leaves)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Operator {
    Unary(UnaryOp),
    Binary(BinaryOp),
    Approx(u64),
    And2,
}

impl Operator {
    fn coverage_code(self) -> u16 {
        match self {
            Self::Unary(UnaryOp::BitToScalar) => 0x1000,
            Self::Unary(UnaryOp::IntToScalar) => 0x1001,
            Self::Unary(UnaryOp::Absolute) => 0x1002,
            Self::Unary(UnaryOp::Sign) => 0x1003,
            Self::Binary(BinaryOp::Difference) => 0x2001,
            Self::Binary(BinaryOp::EqualExact) => 0x2002,
            Self::Binary(BinaryOp::LessEqual) => 0x2003,
            Self::Binary(BinaryOp::SameSign) => 0x2005,
            Self::Binary(BinaryOp::OppositeSign) => 0x2006,
            Self::Approx(1) => 0x3001,
            Self::Approx(2) => 0x3002,
            Self::And2 => 0x4002,
            Self::Binary(BinaryOp::Add | BinaryOp::GreaterEqual) | Self::Approx(_) => {
                unreachable!("removed operator")
            }
        }
    }

    fn child_sorts(self) -> &'static [OutputSortId] {
        match self {
            Self::Unary(UnaryOp::BitToScalar) => &[OutputSortId::Bit],
            Self::Unary(UnaryOp::IntToScalar) => &[OutputSortId::BoundedInt],
            Self::Unary(UnaryOp::Absolute | UnaryOp::Sign) => &[OutputSortId::RationalValue],
            Self::Binary(BinaryOp::Difference)
            | Self::Binary(BinaryOp::EqualExact)
            | Self::Binary(BinaryOp::LessEqual)
            | Self::Approx(_) => &[OutputSortId::RationalValue, OutputSortId::RationalValue],
            Self::Binary(BinaryOp::SameSign | BinaryOp::OppositeSign) => {
                &[OutputSortId::Sign, OutputSortId::Sign]
            }
            Self::And2 => &[OutputSortId::Bool, OutputSortId::Bool],
            Self::Binary(BinaryOp::Add | BinaryOp::GreaterEqual) => unreachable!(),
        }
    }

    fn node(self, left: &Program, right: Option<&Program>) -> Node {
        match self {
            Self::Unary(op) => Node::Unary {
                op,
                child: Box::new(left.canonical.canonical_node.clone()),
            },
            Self::Binary(op) => Node::Binary {
                op,
                left: Box::new(left.canonical.canonical_node.clone()),
                right: Box::new(right.expect("binary child").canonical.canonical_node.clone()),
            },
            Self::Approx(tolerance_index) => Node::ApproxEqual {
                left: Box::new(left.canonical.canonical_node.clone()),
                right: Box::new(right.expect("approx child").canonical.canonical_node.clone()),
                tolerance_index,
            },
            Self::And2 => Node::And(vec![
                left.canonical.canonical_node.clone(),
                right.expect("AND child").canonical.canonical_node.clone(),
            ]),
        }
    }
}

const OPERATORS: [Operator; 12] = [
    Operator::Unary(UnaryOp::BitToScalar),
    Operator::Unary(UnaryOp::IntToScalar),
    Operator::Unary(UnaryOp::Absolute),
    Operator::Unary(UnaryOp::Sign),
    Operator::Binary(BinaryOp::Difference),
    Operator::Binary(BinaryOp::EqualExact),
    Operator::Binary(BinaryOp::LessEqual),
    Operator::Binary(BinaryOp::SameSign),
    Operator::Binary(BinaryOp::OppositeSign),
    Operator::Approx(1),
    Operator::Approx(2),
    Operator::And2,
];

fn representative_capacity(sort: OutputSortId) -> usize {
    match sort {
        OutputSortId::Bool | OutputSortId::RationalValue => 2,
        OutputSortId::Bit | OutputSortId::Sign | OutputSortId::BoundedInt => 1,
    }
}

fn resource_eligible(
    operator: Operator,
    left: &Program,
    right: Option<&Program>,
    depth: u32,
    maximum_ast_node_count: u32,
) -> bool {
    let children = right.into_iter().chain(std::iter::once(left));
    let children = children.collect::<Vec<_>>();
    if children.len() != operator.child_sorts().len()
        || (operator.child_sorts().len() == 1
            && left.signature.output_sort != operator.child_sorts()[0])
        || (operator.child_sorts().len() == 2
            && (left.signature.output_sort != operator.child_sorts()[0]
                || right.expect("binary child").signature.output_sort
                    != operator.child_sorts()[1]))
        || children
            .iter()
            .any(|child| matches!(child.canonical.canonical_node, Node::And(_)))
    {
        return false;
    }
    let resulting_depth = 1 + children
        .iter()
        .map(|child| child.signature.ast_depth)
        .max()
        .unwrap_or(0);
    let nodes = 1 + children
        .iter()
        .map(|child| child.signature.ast_node_count)
        .sum::<u32>();
    let aggregates = children
        .iter()
        .map(|child| child.signature.aggregate_leaf_count)
        .sum::<u32>();
    let scalars = children
        .iter()
        .map(|child| child.signature.scalar_occurrence_count)
        .sum::<u32>();
    let clauses = children
        .iter()
        .map(|child| child.signature.scope_clause_count)
        .sum::<u32>();
    let bitmask = children.iter().fold(0_u8, |mask, child| {
        mask | child.signature.distinct_bit_slot_bitmap
    });
    resulting_depth == depth
        && resulting_depth <= FULL_MAXIMUM_AST_DEPTH
        && nodes <= maximum_ast_node_count
        && aggregates <= MAX_AGGREGATE_LEAVES
        && scalars <= MAX_SCALAR_OCCURRENCES
        && clauses <= MAX_SCOPE_CLAUSES
        && bitmask.count_ones() <= MAX_DISTINCT_BITS
}

fn child_order_key(program: &Program) -> ([u8; 32], &[u8]) {
    let node_bytes = &program.canonical.canonical_cbor[2..];
    (sha256_bytes(&[node_bytes]), node_bytes)
}

fn program_sort_key(program: &Program) -> Vec<u8> {
    let mut key = Vec::with_capacity(6 + program.canonical.canonical_cbor.len());
    key.push(program.canonical.depth as u8);
    key.extend_from_slice(&(program.canonical.node_count as u16).to_be_bytes());
    key.push(program.signature.output_sort as u8);
    key.extend_from_slice(&program.canonical.root_operator_id.to_be_bytes());
    key.extend_from_slice(&program.canonical.canonical_cbor);
    key
}

#[derive(Debug)]
enum TraversalFailure {
    Guard(GuardHit),
    Engine(EngineError),
}

impl From<EngineError> for TraversalFailure {
    fn from(error: EngineError) -> Self {
        Self::Engine(error)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ApplicationIdentity {
    depth: u32,
    operator_code: u16,
    left_canonical_cbor: Vec<u8>,
    right_canonical_cbor: Option<Vec<u8>>,
}

fn application_identity(
    depth: u32,
    operator: Operator,
    left: &Program,
    right: Option<&Program>,
) -> ApplicationIdentity {
    ApplicationIdentity {
        depth,
        operator_code: operator.coverage_code(),
        left_canonical_cbor: left.canonical.canonical_cbor.clone(),
        right_canonical_cbor: right.map(|program| program.canonical.canonical_cbor.clone()),
    }
}

fn visit_eligible_applications<F>(
    programs: &[ProgramRef],
    depth: u32,
    maximum_ast_node_count: u32,
    mut visit: F,
) -> Result<(), TraversalFailure>
where
    F: FnMut(Operator, &ProgramRef, Option<&ProgramRef>) -> Result<(), TraversalFailure>,
{
    let mut by_sort: BTreeMap<OutputSortId, Vec<ProgramRef>> = BTreeMap::new();
    for program in programs {
        by_sort
            .entry(program.signature.output_sort)
            .or_default()
            .push(program.clone());
    }
    for group in by_sort.values_mut() {
        group.sort_by_key(|program| program_sort_key(program));
    }

    for operator in [
        Operator::Unary(UnaryOp::BitToScalar),
        Operator::Unary(UnaryOp::IntToScalar),
        Operator::Unary(UnaryOp::Absolute),
        Operator::Unary(UnaryOp::Sign),
    ] {
        let input_sort = operator.child_sorts()[0];
        for child in by_sort.get(&input_sort).into_iter().flatten() {
            if resource_eligible(operator, child, None, depth, maximum_ast_node_count) {
                visit(operator, child, None)?;
            }
        }
    }

    if let Some(rational) = by_sort.get(&OutputSortId::RationalValue) {
        for left in rational {
            for right in rational {
                if resource_eligible(
                    Operator::Binary(BinaryOp::Difference),
                    left,
                    Some(right),
                    depth,
                    maximum_ast_node_count,
                ) {
                    visit(Operator::Binary(BinaryOp::Difference), left, Some(right))?;
                    visit(Operator::Binary(BinaryOp::LessEqual), left, Some(right))?;
                }
            }
        }

        let mut commutative_rational = rational.clone();
        commutative_rational.sort_by(|left, right| {
            child_order_key(left).cmp(&child_order_key(right))
        });
        for left_index in 0..commutative_rational.len() {
            for right_index in left_index..commutative_rational.len() {
                let left = &commutative_rational[left_index];
                let right = &commutative_rational[right_index];
                if resource_eligible(
                    Operator::Binary(BinaryOp::EqualExact),
                    left,
                    Some(right),
                    depth,
                    maximum_ast_node_count,
                ) {
                    visit(Operator::Binary(BinaryOp::EqualExact), left, Some(right))?;
                    visit(Operator::Approx(1), left, Some(right))?;
                    visit(Operator::Approx(2), left, Some(right))?;
                }
            }
        }
    }

    if let Some(signs) = by_sort.get(&OutputSortId::Sign) {
        let mut commutative_signs = signs.clone();
        commutative_signs.sort_by(|left, right| {
            child_order_key(left).cmp(&child_order_key(right))
        });
        for left_index in 0..commutative_signs.len() {
            for right_index in left_index..commutative_signs.len() {
                let left = &commutative_signs[left_index];
                let right = &commutative_signs[right_index];
                if resource_eligible(
                    Operator::Binary(BinaryOp::SameSign),
                    left,
                    Some(right),
                    depth,
                    maximum_ast_node_count,
                ) {
                    visit(Operator::Binary(BinaryOp::SameSign), left, Some(right))?;
                    visit(
                        Operator::Binary(BinaryOp::OppositeSign),
                        left,
                        Some(right),
                    )?;
                }
            }
        }
    }

    if let Some(booleans) = by_sort.get(&OutputSortId::Bool) {
        let mut bool_atoms = booleans
            .iter()
            .filter(|program| !matches!(program.canonical.canonical_node, Node::And(_)))
            .cloned()
            .collect::<Vec<_>>();
        bool_atoms.sort_by(|left, right| {
            left.canonical.canonical_cbor[2..].cmp(&right.canonical.canonical_cbor[2..])
        });
        for left_index in 0..bool_atoms.len() {
            for right_index in (left_index + 1)..bool_atoms.len() {
                let left = &bool_atoms[left_index];
                let right = &bool_atoms[right_index];
                if resource_eligible(
                    Operator::And2,
                    left,
                    Some(right),
                    depth,
                    maximum_ast_node_count,
                ) {
                    visit(Operator::And2, left, Some(right))?;
                }
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct VectorCacheKey {
    operator_code: u16,
    left_behavior: [u8; 32],
    right_behavior: Option<[u8; 32]>,
}

fn apply_unary(operator: UnaryOp, value: &RuntimeValue) -> Result<RuntimeValue, EngineError> {
    if matches!(value, RuntimeValue::Bottom) {
        return Ok(RuntimeValue::Bottom);
    }
    match (operator, value) {
        (UnaryOp::BitToScalar, RuntimeValue::Bit(value)) => {
            Ok(RuntimeValue::Rational(Rational::integer(i64::from(*value))))
        }
        (UnaryOp::IntToScalar, RuntimeValue::BoundedInt(value)) => {
            Ok(RuntimeValue::Rational(Rational::integer(i64::from(*value))))
        }
        (UnaryOp::Absolute, RuntimeValue::Rational(value)) => Ok(value
            .absolute()
            .map(bounded_rational)
            .unwrap_or(RuntimeValue::Bottom)),
        (UnaryOp::Sign, RuntimeValue::Rational(value)) => {
            Ok(RuntimeValue::Sign(value.numerator.signum() as i8))
        }
        _ => Err(EngineError::new(
            "FAIL_SEMANTICS_MISMATCH",
            "unary type mismatch",
        )),
    }
}

fn apply_binary(
    operator: Operator,
    left: &RuntimeValue,
    right: &RuntimeValue,
) -> Result<RuntimeValue, EngineError> {
    if matches!(left, RuntimeValue::Bottom) || matches!(right, RuntimeValue::Bottom) {
        return Ok(RuntimeValue::Bottom);
    }
    match (operator, left, right) {
        (
            Operator::Binary(BinaryOp::Difference),
            RuntimeValue::Rational(left),
            RuntimeValue::Rational(right),
        ) => Ok(left
            .difference(*right)
            .map(bounded_rational)
            .unwrap_or(RuntimeValue::Bottom)),
        (
            Operator::Binary(BinaryOp::EqualExact),
            RuntimeValue::Rational(left),
            RuntimeValue::Rational(right),
        ) => Ok(RuntimeValue::Bool(left == right)),
        (
            Operator::Binary(BinaryOp::LessEqual),
            RuntimeValue::Rational(left),
            RuntimeValue::Rational(right),
        ) => Ok(RuntimeValue::Bool(left.compare(*right) != Ordering::Greater)),
        (
            Operator::Binary(BinaryOp::SameSign),
            RuntimeValue::Sign(left),
            RuntimeValue::Sign(right),
        ) => Ok(RuntimeValue::Bool(left == right)),
        (
            Operator::Binary(BinaryOp::OppositeSign),
            RuntimeValue::Sign(left),
            RuntimeValue::Sign(right),
        ) => Ok(RuntimeValue::Bool(*left != 0 && *left == -*right)),
        (Operator::Approx(tolerance), RuntimeValue::Rational(left), RuntimeValue::Rational(right)) => {
            let tolerance = match tolerance {
                1 => Rational::new(1, 4)?,
                2 => Rational::new(1, 2)?,
                _ => unreachable!("removed tolerance"),
            };
            let distance = left
                .difference(*right)
                .and_then(Rational::absolute)
                .ok_or_else(|| EngineError::new("FAIL_SEMANTICS_MISMATCH", "rational overflow"))?;
            Ok(RuntimeValue::Bool(
                distance.compare(tolerance) != Ordering::Greater,
            ))
        }
        (Operator::And2, RuntimeValue::Bool(left), RuntimeValue::Bool(right)) => {
            Ok(RuntimeValue::Bool(*left && *right))
        }
        _ => Err(EngineError::new(
            "FAIL_SEMANTICS_MISMATCH",
            "binary type mismatch",
        )),
    }
}

fn apply_operator_vector(
    operator: Operator,
    left: &Behavior,
    right: Option<&Behavior>,
) -> Result<Arc<Vec<RuntimeValue>>, EngineError> {
    let cells = match right {
        None => left
            .cells
            .iter()
            .map(|value| match operator {
                Operator::Unary(unary) => apply_unary(unary, value),
                _ => unreachable!("unary arity"),
            })
            .collect::<Result<Vec<_>, _>>()?,
        Some(right) => {
            if left.cells.len() != right.cells.len() {
                return Err(EngineError::new(
                    "FAIL_SEMANTICS_MISMATCH",
                    "child behavior lengths differ",
                ));
            }
            left.cells
                .iter()
                .zip(right.cells.iter())
                .map(|(left, right)| apply_binary(operator, left, right))
                .collect::<Result<Vec<_>, _>>()?
        }
    };
    Ok(Arc::new(cells))
}

fn evaluate_canonical_node(
    node: &Node,
    observation: &Observation,
) -> Result<RuntimeValue, EngineError> {
    match node {
        Node::ScalarConst(_)
        | Node::BitAt(_)
        | Node::SetSize
        | Node::Aggregate { .. }
        | Node::ContextFlag(_)
        | Node::TaskFlag(_)
        | Node::NewSymbolCall(_) => evaluate_leaf(node, observation),
        Node::Unary { op, child } => {
            let child = evaluate_canonical_node(child, observation)?;
            apply_unary(*op, &child)
        }
        Node::Binary { op, left, right } => {
            let left = evaluate_canonical_node(left, observation)?;
            let right = evaluate_canonical_node(right, observation)?;
            apply_binary(Operator::Binary(*op), &left, &right)
        }
        Node::ApproxEqual {
            left,
            right,
            tolerance_index,
        } => {
            let left = evaluate_canonical_node(left, observation)?;
            let right = evaluate_canonical_node(right, observation)?;
            apply_binary(Operator::Approx(*tolerance_index), &left, &right)
        }
        Node::And(children) if children.len() == 2 => {
            let left = evaluate_canonical_node(&children[0], observation)?;
            let right = evaluate_canonical_node(&children[1], observation)?;
            apply_binary(Operator::And2, &left, &right)
        }
        Node::And(_) => Err(EngineError::new(
            "FAIL_SEMANTICS_MISMATCH",
            "canonical AND must contain exactly two children",
        )),
    }
}

fn evaluate_canonical_vector(
    canonical: &CanonicalProgram,
    observations: &[Observation],
) -> Result<Arc<Vec<RuntimeValue>>, EngineError> {
    let cells = observations
        .iter()
        .map(|observation| evaluate_canonical_node(&canonical.canonical_node, observation))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Arc::new(cells))
}

fn application_behavior_vector(
    canonical: &CanonicalProgram,
    rewritten: bool,
    raw_cells: Arc<Vec<RuntimeValue>>,
    observations: &[Observation],
) -> Result<Arc<Vec<RuntimeValue>>, EngineError> {
    if rewritten {
        evaluate_canonical_vector(canonical, observations)
    } else {
        Ok(raw_cells)
    }
}

#[derive(Debug, Clone)]
struct QuotientClass {
    behavior_key: Arc<Vec<u8>>,
    behavior_digest: [u8; 32],
    cohorts: BTreeMap<ConstructionSignature, Vec<ProgramRef>>,
}

impl QuotientClass {
    fn add(&mut self, program: ProgramRef) -> (bool, bool) {
        let signature = program.signature.clone();
        let cohort_created = !self.cohorts.contains_key(&signature);
        let cohort = self.cohorts.entry(signature).or_default();
        let before = cohort
            .iter()
            .map(|entry| entry.canonical.canonical_cbor.clone())
            .collect::<Vec<_>>();
        cohort.push(program);
        cohort.sort_by(|left, right| left.canonical.canonical_cbor.cmp(&right.canonical.canonical_cbor));
        cohort.dedup_by(|left, right| left.canonical.canonical_cbor == right.canonical.canonical_cbor);
        cohort.truncate(representative_capacity(cohort[0].signature.output_sort));
        let after = cohort
            .iter()
            .map(|entry| entry.canonical.canonical_cbor.clone())
            .collect::<Vec<_>>();
        (cohort_created, before != after)
    }

    fn visible_cohort_count(&self) -> usize {
        self.cohorts
            .iter()
            .filter(|(candidate_signature, candidate)| {
                !self.cohorts.iter().any(|(other_signature, other)| {
                    other_signature.dominates(candidate_signature)
                        && other.len() >= candidate.len()
                })
            })
            .map(|(_, cohort)| cohort.len())
            .sum()
    }

    fn bank_count(&self) -> usize {
        self.cohorts.values().map(Vec::len).sum()
    }
}

#[derive(Debug, Clone, Default)]
struct QuotientState {
    classes: BTreeMap<Vec<u8>, QuotientClass>,
}

#[derive(Debug, Clone, Copy, Default)]
struct StateCounts {
    classes: u64,
    cohorts: u64,
    bank: u64,
    frontier: u64,
    maximum_bank_per_class: u64,
    maximum_frontier_per_class: u64,
}

impl QuotientState {
    fn counts(&self) -> StateCounts {
        StateCounts {
            classes: self.classes.len() as u64,
            cohorts: self
                .classes
                .values()
                .map(|class| class.cohorts.len() as u64)
                .sum(),
            bank: self
                .classes
                .values()
                .map(|class| class.bank_count() as u64)
                .sum(),
            frontier: self
                .classes
                .values()
                .map(|class| class.visible_cohort_count() as u64)
                .sum(),
            maximum_bank_per_class: self
                .classes
                .values()
                .map(|class| class.bank_count() as u64)
                .max()
                .unwrap_or(0),
            maximum_frontier_per_class: self
                .classes
                .values()
                .map(|class| class.visible_cohort_count() as u64)
                .max()
                .unwrap_or(0),
        }
    }

    fn continuation_programs(&self) -> Result<Vec<ProgramRef>, EngineError> {
        let mut programs = BTreeMap::<Vec<u8>, ProgramRef>::new();
        for program in self
            .classes
            .values()
            .flat_map(|class| class.cohorts.values())
            .flat_map(|cohort| cohort.iter())
        {
            if let Some(prior) = programs.get(&program.canonical.canonical_cbor) {
                if prior.behavior.key != program.behavior.key {
                    let counts = self.counts();
                    let canonical_cbor_digest =
                        sha256_bytes(&[&program.canonical.canonical_cbor]);
                    return Err(EngineError::new(
                        "FAIL_Q1_PREFLIGHT_BEHAVIOR",
                        format!(
                            concat!(
                                "one canonical AST has two behaviors: ",
                                "canonical_cbor_sha256=sha256:{} depth={} ",
                                "prior_behavior_sha256=sha256:{} ",
                                "current_behavior_sha256=sha256:{} ",
                                "state_counts=classes:{},cohorts:{},bank:{},frontier:{}"
                            ),
                            hex(&canonical_cbor_digest),
                            program.canonical.depth,
                            hex(&prior.behavior.digest),
                            hex(&program.behavior.digest),
                            counts.classes,
                            counts.cohorts,
                            counts.bank,
                            counts.frontier,
                        ),
                    ));
                }
            }
            programs.insert(program.canonical.canonical_cbor.clone(), program.clone());
        }
        Ok(programs.into_values().collect())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct CoverageKey {
    depth: u8,
    code: u16,
}

#[derive(Debug, Clone, Default)]
struct CoverageCounter {
    eligible: u64,
    strict_admitted: u64,
    rewrite_collapses: u64,
}

fn empty_coverage() -> BTreeMap<CoverageKey, CoverageCounter> {
    let mut coverage = BTreeMap::new();
    for code in 0_u16..FROZEN_LEAF_COUNT as u16 {
        coverage.insert(CoverageKey { depth: 0, code }, CoverageCounter::default());
    }
    for depth in 1_u8..=3 {
        for operator in OPERATORS {
            coverage.insert(
                CoverageKey {
                    depth,
                    code: operator.coverage_code(),
                },
                CoverageCounter::default(),
            );
        }
    }
    debug_assert_eq!(coverage.len(), 846);
    coverage
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct DepthBarrierDiagnostic {
    pub depth: u32,
    pub barrier_kind: &'static str,
    pub eligible_raw_application_count: u64,
    pub strict_admitted_application_count: u64,
    pub rewrite_collapse_count: u64,
    pub new_behavior_class_count: u64,
    pub new_signature_cohort_count: u64,
    pub continuation_bank_mutation_count: u64,
    pub behavior_class_count_after_barrier: u64,
    pub signature_cohort_count_after_barrier: u64,
    pub continuation_bank_point_count_after_barrier: u64,
    pub visible_frontier_point_count_after_barrier: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ProjectionStreamDiagnostic {
    pub stream_kind: &'static str,
    pub record_count: u64,
    pub payload_bytes: u64,
    pub framed_bytes: u64,
    pub counting_discard_sha256: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct DiagnosticStateProjection {
    pub schema_version: &'static str,
    pub projection_kind: &'static str,
    pub diagnostic_only: bool,
    pub formal_archive_materialized: bool,
    pub projected_diagnostic_state_bytes: u64,
    pub projected_diagnostic_scratch_bytes: u64,
    pub streams: Vec<ProjectionStreamDiagnostic>,
    pub combined_state_projection_sha256: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct FixedPointDiagnostic {
    pub work_queue_empty: bool,
    pub zero_delta_structural_boundary: bool,
    pub all_eligible_operator_and_bank_tuple_expansions_covered: bool,
    pub final_new_behavior_class_delta: u64,
    pub final_visible_frontier_delta: u64,
    pub final_continuation_bank_delta: u64,
    pub no_resource_guard_hit: bool,
    pub saturated_diagnostic_only: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct CoverageDiagnostic {
    pub coverage_record_count: u64,
    pub eligible_application_count: u64,
    pub processed_application_count: u64,
    pub strict_admitted_application_count: u64,
    pub rewrite_collapse_count: u64,
    pub zero_eligible_record_count: u64,
    pub all_eligible_covered: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct PartitionDiagnostic {
    pub input_signature_id: u64,
    pub input_signature_name: &'static str,
    pub universe_root: String,
    pub universe_row_count: u64,
    pub limits: CapacityLimits,
    pub full_v16_structural_limits_applied: bool,
    pub terminal_status: String,
    pub resource_guard_id: Option<u8>,
    pub resource_guard_name: Option<String>,
    pub traversal_closed: bool,
    pub frozen_leaf_count: u64,
    pub raw_operator_application_count: u64,
    pub strict_admitted_application_count: u64,
    pub rewrite_collapse_count: u64,
    pub behavior_class_count: u64,
    pub signature_cohort_count: u64,
    pub continuation_bank_point_count: u64,
    pub visible_frontier_point_count: u64,
    pub maximum_bank_points_per_class: u64,
    pub maximum_frontier_points_per_class: u64,
    pub peak_raw_operator_application_count: u64,
    pub peak_behavior_class_count: u64,
    pub peak_visible_frontier_point_count: u64,
    pub peak_visible_frontier_points_per_class: u64,
    pub peak_continuation_bank_point_count: u64,
    pub peak_continuation_bank_points_per_class: u64,
    pub peak_work_queue_points: u64,
    pub peak_saturation_round_count: u64,
    pub vector_cache_entry_count: u64,
    pub vector_cache_hit_count: u64,
    pub vector_cache_miss_count: u64,
    pub depth_barriers: Vec<DepthBarrierDiagnostic>,
    pub fixed_point: Option<FixedPointDiagnostic>,
    pub coverage: Option<CoverageDiagnostic>,
    pub diagnostic_state_projection: Option<DiagnosticStateProjection>,
    pub same_run_record_set_root: Option<String>,
    pub formal_wire_projection: Option<FrozenFormalWireProjection>,
    pub elapsed_monotonic_ns: u64,
    pub diagnostic_only: bool,
    pub formal_roots_generated: bool,
    pub formal_roots: Option<String>,
    pub target_truth_accessed: bool,
    pub split_accessed: bool,
    pub role_evaluation_performed: bool,
    pub complete_claim_allowed: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct EngineDiagnostic {
    pub schema_version: &'static str,
    pub endpoint_schema_version: &'static str,
    pub actor_id: &'static str,
    pub implementation_id: &'static str,
    pub action_id: &'static str,
    pub capsule_digest: String,
    pub source_commit: String,
    pub dsl_version: &'static str,
    pub closure_semantics_version: &'static str,
    pub terminal_status: String,
    pub actual_preflight_peak_scratch_bytes: u64,
    pub actual_preflight_terminal_live_scratch_bytes: u64,
    pub projected_closure_bundle_frame_bytes: Option<u64>,
    pub projected_formal_archive_payload_bytes: Option<u64>,
    pub projected_formal_total_output_bytes: Option<u64>,
    pub projected_peak_formal_scratch_bytes: Option<u64>,
    pub partitions: Vec<PartitionDiagnostic>,
    pub diagnostic_only: bool,
    pub q1_state: &'static str,
    pub q1_gate_count: u8,
    pub q1_gate_mask: u32,
    pub q1_formal_roots: Option<String>,
    pub q1_receipt: Option<String>,
    pub q1_output_slots: [Option<String>; 8],
    pub q1_execution_started: bool,
    pub q2_state: &'static str,
    pub target_truth_accessed: bool,
    pub split_accessed: bool,
    pub role_evaluation_performed: bool,
    pub active_transition_allowed: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct FormalArchiveMaterializerResult {
    pub schema_version: &'static str,
    pub status: &'static str,
    pub actor_id: &'static str,
    pub implementation_id: &'static str,
    pub action_id: &'static str,
    pub capsule_digest: String,
    pub source_commit: String,
    pub run_genesis_root: String,
    pub prestart_admission_receipt_root: String,
    pub q1_state: &'static str,
    pub q1_gate_count: u8,
    pub q1_gate_total: u8,
    pub q1_gate_mask: u32,
    pub q1_output_slots: [Option<String>; 8],
    pub dsl_version: &'static str,
    pub closure_semantics_version: &'static str,
    pub maximum_ast_depth: u32,
    pub maximum_ast_node_count: u32,
    pub full_v16_structural_limits_applied: bool,
    pub formal_archive_materialized: bool,
    pub diagnostic_only: bool,
    pub formal_capacity_limits_by_signature: [FormalCapacityLimitRow; 2],
    pub formal_output_bytes_limit: u64,
    pub formal_scratch_bytes_limit: u64,
    pub closure_bundle_root: String,
    pub formal_archive_payload_bytes: u64,
    pub actual_formal_peak_scratch_bytes: u64,
    pub actual_formal_terminal_live_scratch_bytes: u64,
    pub partitions: Vec<FrozenFormalMaterializedPartition>,
    pub target_truth_accessed: bool,
    pub split_accessed: bool,
    pub role_evaluation_performed: bool,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub struct FormalCapacityLimitRow {
    pub input_signature_id: u64,
    pub maximum_ast_depth: u32,
    pub maximum_ast_node_count: u32,
    pub maximum_raw_operator_applications: u64,
    pub maximum_behavior_classes: u64,
    pub maximum_visible_frontier_points: u64,
    pub maximum_visible_frontier_points_per_class: u64,
    pub maximum_continuation_bank_points: u64,
    pub maximum_continuation_bank_points_per_class: u64,
    pub maximum_work_queue_points: u64,
    pub maximum_saturation_rounds: u64,
    pub maximum_wall_time_seconds: u64,
}

impl FormalCapacityLimitRow {
    fn from_limits(input_signature_id: u64, limits: CapacityLimits) -> Self {
        Self {
            input_signature_id,
            maximum_ast_depth: limits.maximum_ast_depth,
            maximum_ast_node_count: limits.maximum_ast_node_count,
            maximum_raw_operator_applications: limits.maximum_raw_operator_applications,
            maximum_behavior_classes: limits.maximum_behavior_classes,
            maximum_visible_frontier_points: limits.maximum_visible_frontier_points,
            maximum_visible_frontier_points_per_class: limits
                .maximum_visible_frontier_points_per_class,
            maximum_continuation_bank_points: limits.maximum_continuation_bank_points,
            maximum_continuation_bank_points_per_class: limits
                .maximum_continuation_bank_points_per_class,
            maximum_work_queue_points: limits.maximum_work_queue_points,
            maximum_saturation_rounds: limits.maximum_saturation_rounds,
            maximum_wall_time_seconds: limits.maximum_wall_time_seconds,
        }
    }
}

#[derive(Debug)]
struct Runner {
    input_signature_id: u64,
    universe_root: &'static str,
    observations: Vec<Observation>,
    limits: CapacityLimits,
    started: Instant,
    state: QuotientState,
    vector_cache: HashMap<VectorCacheKey, Arc<Vec<RuntimeValue>>>,
    vector_cache_hits: u64,
    vector_cache_misses: u64,
    raw_count: u64,
    current_barrier_raw_count: u64,
    strict_count: u64,
    rewrite_count: u64,
    bank_mutations: u64,
    peak: StateCounts,
    peak_work_queue_points: u64,
    peak_saturation_round_count: u64,
    saturation_round_count: u64,
    barriers: Vec<DepthBarrierDiagnostic>,
    coverage: BTreeMap<CoverageKey, CoverageCounter>,
}

impl Runner {
    fn new(input_signature_id: u64, limits: CapacityLimits) -> Result<Self, EngineError> {
        let (observations, universe_root) = universe(input_signature_id)?;
        Ok(Self {
            input_signature_id,
            universe_root,
            observations,
            limits: limits.validate()?,
            started: Instant::now(),
            state: QuotientState::default(),
            vector_cache: HashMap::new(),
            vector_cache_hits: 0,
            vector_cache_misses: 0,
            raw_count: 0,
            current_barrier_raw_count: 0,
            strict_count: 0,
            rewrite_count: 0,
            bank_mutations: 0,
            peak: StateCounts::default(),
            peak_work_queue_points: 0,
            peak_saturation_round_count: 0,
            saturation_round_count: 0,
            barriers: Vec::new(),
            coverage: empty_coverage(),
        })
    }

    fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }

    fn wall_guard(&self) -> Result<(), TraversalFailure> {
        if self.elapsed().as_secs() > self.limits.maximum_wall_time_seconds {
            return Err(TraversalFailure::Guard(GuardHit::wall()));
        }
        Ok(())
    }

    fn guard_next_saturation_round(&self) -> Result<(), TraversalFailure> {
        if self.saturation_round_count + 1 > self.limits.maximum_saturation_rounds {
            return Err(TraversalFailure::Guard(GuardHit::rounds()));
        }
        self.wall_guard()
    }

    fn complete_saturation_round(&mut self) {
        self.saturation_round_count += 1;
        self.peak_saturation_round_count = self
            .peak_saturation_round_count
            .max(self.saturation_round_count);
    }

    fn consume_raw(&mut self) -> Result<(), TraversalFailure> {
        if self.raw_count + 1 > self.limits.maximum_raw_operator_applications {
            return Err(TraversalFailure::Guard(GuardHit::raw()));
        }
        if self.current_barrier_raw_count + 1 > self.limits.maximum_work_queue_points {
            return Err(TraversalFailure::Guard(GuardHit::work_queue()));
        }
        self.wall_guard()?;
        self.raw_count += 1;
        self.current_barrier_raw_count += 1;
        self.peak_work_queue_points = self
            .peak_work_queue_points
            .max(self.current_barrier_raw_count)
            .max(self.vector_cache.len() as u64);
        Ok(())
    }

    fn guard_counts(&self, counts: StateCounts) -> Result<(), GuardHit> {
        if counts.classes > self.limits.maximum_behavior_classes {
            return Err(GuardHit::classes());
        }
        if counts.frontier > self.limits.maximum_visible_frontier_points {
            return Err(GuardHit::frontier_total());
        }
        if counts.maximum_frontier_per_class
            > self.limits.maximum_visible_frontier_points_per_class
        {
            return Err(GuardHit::frontier_per_class());
        }
        if counts.bank > self.limits.maximum_continuation_bank_points {
            return Err(GuardHit::bank_total());
        }
        if counts.maximum_bank_per_class
            > self.limits.maximum_continuation_bank_points_per_class
        {
            return Err(GuardHit::bank_per_class());
        }
        Ok(())
    }

    fn record_peak(&mut self, counts: StateCounts) {
        self.peak.classes = self.peak.classes.max(counts.classes);
        self.peak.cohorts = self.peak.cohorts.max(counts.cohorts);
        self.peak.bank = self.peak.bank.max(counts.bank);
        self.peak.frontier = self.peak.frontier.max(counts.frontier);
        self.peak.maximum_bank_per_class = self
            .peak
            .maximum_bank_per_class
            .max(counts.maximum_bank_per_class);
        self.peak.maximum_frontier_per_class = self
            .peak
            .maximum_frontier_per_class
            .max(counts.maximum_frontier_per_class);
    }

    fn insert(
        &mut self,
        program: ProgramRef,
        current_depth: u32,
    ) -> Result<(u64, u64, u64, u64), TraversalFailure> {
        let key = (*program.behavior.key).clone();
        let prior = self.state.classes.get(&key).cloned();
        let class_created = prior.is_none();
        let class = self.state.classes.entry(key.clone()).or_insert_with(|| QuotientClass {
            behavior_key: program.behavior.key.clone(),
            behavior_digest: program.behavior.digest,
            cohorts: BTreeMap::new(),
        });
        if class.behavior_key.as_slice() != program.behavior.key.as_slice()
            || class.behavior_digest != program.behavior.digest
        {
            return Err(TraversalFailure::Engine(EngineError::new(
                "FAIL_Q1_PREFLIGHT_BEHAVIOR",
                "behavior identity differs within a class",
            )));
        }
        let (cohort_created, bank_mutated) = class.add(program.clone());
        let late_lower_depth_bank_mutation =
            bank_mutated && program.canonical.depth < current_depth;
        let counts = self.state.counts();
        if let Err(guard) = self.guard_counts(counts) {
            match prior {
                Some(prior) => {
                    self.state.classes.insert(key, prior);
                }
                None => {
                    self.state.classes.remove(&key);
                }
            }
            return Err(TraversalFailure::Guard(guard));
        }
        self.record_peak(counts);
        Ok((
            u64::from(class_created),
            u64::from(cohort_created),
            u64::from(bank_mutated),
            u64::from(late_lower_depth_bank_mutation),
        ))
    }

    fn candidate_vector(
        &mut self,
        operator: Operator,
        left: &Program,
        right: Option<&Program>,
    ) -> Result<Arc<Vec<RuntimeValue>>, TraversalFailure> {
        let key = VectorCacheKey {
            operator_code: operator.coverage_code(),
            left_behavior: left.behavior.digest,
            right_behavior: right.map(|program| program.behavior.digest),
        };
        if let Some(value) = self.vector_cache.get(&key) {
            self.vector_cache_hits += 1;
            return Ok(value.clone());
        }
        if self.vector_cache.len() as u64 + 1 > self.limits.maximum_work_queue_points {
            return Err(TraversalFailure::Guard(GuardHit::work_queue()));
        }
        let value = apply_operator_vector(operator, &left.behavior, right.map(|p| &p.behavior))?;
        self.vector_cache.insert(key, value.clone());
        self.vector_cache_misses += 1;
        self.peak_work_queue_points = self
            .peak_work_queue_points
            .max(self.vector_cache.len() as u64)
            .max(self.current_barrier_raw_count);
        Ok(value)
    }

    fn seed(&mut self) -> Result<(), TraversalFailure> {
        self.guard_next_saturation_round()?;
        self.current_barrier_raw_count = 0;
        let before = self.state.counts();
        let raw_before = self.raw_count;
        let strict_before = self.strict_count;
        let mutation_before = self.bank_mutations;
        for (coverage_code, canonical) in frozen_leaf_manifest()?.into_iter().enumerate() {
            let checkpoint = (
                self.raw_count,
                self.current_barrier_raw_count,
                self.strict_count,
                self.rewrite_count,
            );
            if let Err(error) = self.consume_raw() {
                (
                    self.raw_count,
                    self.current_barrier_raw_count,
                    self.strict_count,
                    self.rewrite_count,
                ) = checkpoint;
                return Err(error);
            }
            let program = Program::from_leaf(
                canonical,
                self.input_signature_id,
                &self.observations,
            )?;
            self.strict_count += 1;
            match self.insert(program, 0) {
                Ok((_, _, mutation, _)) => {
                    self.bank_mutations += mutation;
                    let coverage = self
                        .coverage
                        .get_mut(&CoverageKey {
                            depth: 0,
                            code: coverage_code as u16,
                        })
                        .expect("leaf coverage row");
                    coverage.eligible += 1;
                    coverage.strict_admitted += 1;
                }
                Err(error @ TraversalFailure::Guard(_)) => {
                    (
                        self.raw_count,
                        self.current_barrier_raw_count,
                        self.strict_count,
                        self.rewrite_count,
                    ) = checkpoint;
                    return Err(error);
                }
                Err(error) => return Err(error),
            }
        }
        let after = self.state.counts();
        self.barriers.push(DepthBarrierDiagnostic {
            depth: 0,
            barrier_kind: "LEAF_SEED",
            eligible_raw_application_count: self.raw_count - raw_before,
            strict_admitted_application_count: self.strict_count - strict_before,
            rewrite_collapse_count: 0,
            new_behavior_class_count: after.classes - before.classes,
            new_signature_cohort_count: after.cohorts - before.cohorts,
            continuation_bank_mutation_count: self.bank_mutations - mutation_before,
            behavior_class_count_after_barrier: after.classes,
            signature_cohort_count_after_barrier: after.cohorts,
            continuation_bank_point_count_after_barrier: after.bank,
            visible_frontier_point_count_after_barrier: after.frontier,
        });
        self.complete_saturation_round();
        Ok(())
    }

    fn expand_depth(&mut self, depth: u32) -> Result<(), TraversalFailure> {
        self.current_barrier_raw_count = 0;
        let raw_before = self.raw_count;
        let strict_before = self.strict_count;
        let rewrite_before = self.rewrite_count;
        let mutation_before = self.bank_mutations;
        let mut new_classes = 0_u64;
        let mut new_cohorts = 0_u64;
        let mut processed_applications = BTreeSet::<ApplicationIdentity>::new();

        loop {
            let mut snapshot = self.state.continuation_programs()?;
            if snapshot
                .iter()
                .any(|program| program.canonical.depth > depth)
            {
                return Err(TraversalFailure::Engine(EngineError::new(
                    "FAIL_Q1_PREFLIGHT_AST_IDENTITY",
                    "depth barrier snapshot contains a future-depth representative",
                )));
            }
            snapshot.retain(|program| program.canonical.depth < depth);

            let mut unseen_application_count = 0_u64;
            let mut late_lower_depth_bank_mutation_count = 0_u64;
            visit_eligible_applications(
                &snapshot,
                depth,
                self.limits.maximum_ast_node_count,
                |operator, left, right| {
                    let identity = application_identity(
                        depth,
                        operator,
                        left,
                        right.map(AsRef::as_ref),
                    );
                    if processed_applications.contains(&identity) {
                        return Ok(());
                    }
                    if unseen_application_count == 0 {
                        self.guard_next_saturation_round()?;
                    }

                    let vector_key = VectorCacheKey {
                        operator_code: operator.coverage_code(),
                        left_behavior: left.behavior.digest,
                        right_behavior: right.map(|program| program.behavior.digest),
                    };
                    let vector_was_cached = self.vector_cache.contains_key(&vector_key);
                    let checkpoint = (
                        self.raw_count,
                        self.current_barrier_raw_count,
                        self.strict_count,
                        self.rewrite_count,
                        self.vector_cache_hits,
                        self.vector_cache_misses,
                    );
                    self.consume_raw()?;
                    let source = operator.node(left, right.map(AsRef::as_ref));
                    let canonical = match canonicalize_shrink6_source_node(source.clone()) {
                        Ok(program) => program,
                        Err(error) if error.code == "REJECT_STRUCTURAL_LIMIT" => {
                            let coverage = self
                                .coverage
                                .get_mut(&CoverageKey {
                                    depth: depth as u8,
                                    code: operator.coverage_code(),
                                })
                                .expect("operator coverage row");
                            coverage.eligible += 1;
                            processed_applications.insert(identity);
                            unseen_application_count += 1;
                            return Ok(());
                        }
                        Err(error) => {
                            return Err(TraversalFailure::Engine(EngineError::new(
                                error.code,
                                error.message,
                            )))
                        }
                    };
                    if canonical.depth > self.limits.maximum_ast_depth
                        || canonical.node_count > self.limits.maximum_ast_node_count
                        || canonical.aggregate_leaf_count > MAX_AGGREGATE_LEAVES
                        || canonical.scalar_parameter_occurrence_count > MAX_SCALAR_OCCURRENCES
                    {
                        let coverage = self
                            .coverage
                            .get_mut(&CoverageKey {
                                depth: depth as u8,
                                code: operator.coverage_code(),
                            })
                            .expect("operator coverage row");
                        coverage.eligible += 1;
                        processed_applications.insert(identity);
                        unseen_application_count += 1;
                        return Ok(());
                    }
                    let rewritten = canonical.canonical_node != source;
                    self.strict_count += 1;
                    self.rewrite_count += u64::from(rewritten);
                    let raw_cells = self.candidate_vector(
                        operator,
                        left,
                        right.map(AsRef::as_ref),
                    )?;
                    let cells = application_behavior_vector(
                        &canonical,
                        rewritten,
                        raw_cells,
                        &self.observations,
                    )?;
                    let program = Program::from_application(
                        canonical,
                        self.input_signature_id,
                        cells,
                    )?;
                    match self.insert(program, depth) {
                        Ok((class_delta, cohort_delta, mutation, late_mutation)) => {
                            new_classes += class_delta;
                            new_cohorts += cohort_delta;
                            self.bank_mutations += mutation;
                            late_lower_depth_bank_mutation_count += late_mutation;
                            let coverage = self
                                .coverage
                                .get_mut(&CoverageKey {
                                    depth: depth as u8,
                                    code: operator.coverage_code(),
                                })
                                .expect("operator coverage row");
                            coverage.eligible += 1;
                            coverage.strict_admitted += 1;
                            coverage.rewrite_collapses += u64::from(rewritten);
                            processed_applications.insert(identity);
                            unseen_application_count += 1;
                            Ok(())
                        }
                        Err(error @ TraversalFailure::Guard(_)) => {
                            if !vector_was_cached {
                                self.vector_cache.remove(&vector_key);
                            }
                            (
                                self.raw_count,
                                self.current_barrier_raw_count,
                                self.strict_count,
                                self.rewrite_count,
                                self.vector_cache_hits,
                                self.vector_cache_misses,
                            ) = checkpoint;
                            Err(error)
                        }
                        Err(error) => Err(error),
                    }
                },
            )?;
            if unseen_application_count > 0 {
                self.complete_saturation_round();
            }
            self.wall_guard()?;
            if unseen_application_count == 0
                && late_lower_depth_bank_mutation_count == 0
            {
                break;
            }
        }

        let after = self.state.counts();
        self.barriers.push(DepthBarrierDiagnostic {
            depth,
            barrier_kind: "CONSTRUCTION_DEPTH",
            eligible_raw_application_count: self.raw_count - raw_before,
            strict_admitted_application_count: self.strict_count - strict_before,
            rewrite_collapse_count: self.rewrite_count - rewrite_before,
            new_behavior_class_count: new_classes,
            new_signature_cohort_count: new_cohorts,
            continuation_bank_mutation_count: self.bank_mutations - mutation_before,
            behavior_class_count_after_barrier: after.classes,
            signature_cohort_count_after_barrier: after.cohorts,
            continuation_bank_point_count_after_barrier: after.bank,
            visible_frontier_point_count_after_barrier: after.frontier,
        });
        Ok(())
    }

    fn close_structural_boundary(&mut self) -> Result<(), TraversalFailure> {
        self.guard_next_saturation_round()?;
        let counts = self.state.counts();
        self.barriers.push(DepthBarrierDiagnostic {
            depth: self.limits.maximum_ast_depth + 1,
            barrier_kind: "STRUCTURAL_BOUNDARY",
            eligible_raw_application_count: 0,
            strict_admitted_application_count: 0,
            rewrite_collapse_count: 0,
            new_behavior_class_count: 0,
            new_signature_cohort_count: 0,
            continuation_bank_mutation_count: 0,
            behavior_class_count_after_barrier: counts.classes,
            signature_cohort_count_after_barrier: counts.cohorts,
            continuation_bank_point_count_after_barrier: counts.bank,
            visible_frontier_point_count_after_barrier: counts.frontier,
        });
        self.complete_saturation_round();
        Ok(())
    }
}

struct ProjectionAccumulator {
    kind: &'static str,
    digest: Sha256,
    records: u64,
    payload_bytes: u64,
    framed_bytes: u64,
}

impl ProjectionAccumulator {
    fn new(kind: &'static str) -> Self {
        let mut digest = Sha256::new();
        digest.update(b"HEGEL/Q1/CAPACITY/PREFLIGHT/COUNTING_DISCARD/V1\0");
        digest.update(kind.as_bytes());
        Self {
            kind,
            digest,
            records: 0,
            payload_bytes: 0,
            framed_bytes: 0,
        }
    }

    fn push(&mut self, record: &[u8]) {
        let length = record.len() as u64;
        self.digest.update(length.to_be_bytes());
        self.digest.update(record);
        self.records += 1;
        self.payload_bytes += length;
        self.framed_bytes += 8 + length;
    }

    fn finish(self) -> ProjectionStreamDiagnostic {
        ProjectionStreamDiagnostic {
            stream_kind: self.kind,
            record_count: self.records,
            payload_bytes: self.payload_bytes,
            framed_bytes: self.framed_bytes,
            counting_discard_sha256: hex(&self.digest.finalize()),
        }
    }
}

fn append_blob(output: &mut Vec<u8>, value: &[u8]) {
    output.extend_from_slice(&(value.len() as u64).to_be_bytes());
    output.extend_from_slice(value);
}

fn diagnostic_state_projection(
    state: &QuotientState,
    coverage: &BTreeMap<CoverageKey, CoverageCounter>,
) -> DiagnosticStateProjection {
    let mut programs = ProjectionAccumulator::new("PROGRAM");
    let mut cohorts = ProjectionAccumulator::new("COHORT");
    let mut classes = ProjectionAccumulator::new("CLASS");
    let mut coverage_stream = ProjectionAccumulator::new("COVERAGE");

    for class in state.classes.values() {
        let mut class_record = Vec::new();
        class_record.extend_from_slice(b"CLASS\0");
        append_blob(&mut class_record, class.behavior_key.as_slice());
        class_record.extend_from_slice(&(class.cohorts.len() as u64).to_be_bytes());
        classes.push(&class_record);

        for (signature, cohort) in &class.cohorts {
            let signature_bytes = signature.bytes();
            let visible = !class.cohorts.iter().any(|(other_signature, other)| {
                other_signature.dominates(signature) && other.len() >= cohort.len()
            });
            let mut cohort_record = Vec::new();
            cohort_record.extend_from_slice(b"COHORT\0");
            cohort_record.extend_from_slice(&class.behavior_digest);
            append_blob(&mut cohort_record, &signature_bytes);
            cohort_record.push(u8::from(visible));
            cohort_record.extend_from_slice(&(cohort.len() as u64).to_be_bytes());
            for program in cohort {
                cohort_record.extend_from_slice(&program.canonical.canonical_ast_hash);

                let mut program_record = Vec::new();
                program_record.extend_from_slice(b"PROGRAM\0");
                program_record.extend_from_slice(&class.behavior_digest);
                let signature_digest = sha256_bytes(&[&signature_bytes]);
                program_record.extend_from_slice(&signature_digest);
                append_blob(&mut program_record, &program.canonical.canonical_cbor);
                program_record.extend_from_slice(&program.canonical.canonical_ast_hash);
                programs.push(&program_record);
            }
            cohorts.push(&cohort_record);
        }
    }

    for (key, value) in coverage {
        let mut record = Vec::with_capacity(32);
        record.extend_from_slice(b"COVERAGE\0");
        record.push(key.depth);
        record.extend_from_slice(&key.code.to_be_bytes());
        record.extend_from_slice(&value.eligible.to_be_bytes());
        record.extend_from_slice(&value.strict_admitted.to_be_bytes());
        record.extend_from_slice(&value.rewrite_collapses.to_be_bytes());
        coverage_stream.push(&record);
    }

    let streams = vec![
        programs.finish(),
        cohorts.finish(),
        classes.finish(),
        coverage_stream.finish(),
    ];
    let mut combined = Sha256::new();
    combined.update(b"HEGEL/Q1/CAPACITY/PREFLIGHT/STATE_PROJECTION/V1\0");
    for stream in &streams {
        combined.update(stream.stream_kind.as_bytes());
        combined.update(stream.record_count.to_be_bytes());
        combined.update(stream.framed_bytes.to_be_bytes());
        combined.update(stream.counting_discard_sha256.as_bytes());
    }
    DiagnosticStateProjection {
        schema_version: "hegel-phase3a-q1-capacity-counting-discard-projection/1",
        projection_kind: "DIAGNOSTIC_STATE_AGREEMENT_KAT_NOT_FORMAL_ARCHIVE",
        diagnostic_only: true,
        formal_archive_materialized: false,
        projected_diagnostic_state_bytes: streams.iter().map(|row| row.framed_bytes).sum(),
        projected_diagnostic_scratch_bytes: 0,
        streams,
        combined_state_projection_sha256: hex(&combined.finalize()),
    }
}

fn hex(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(ALPHABET[(byte >> 4) as usize] as char);
        output.push(ALPHABET[(byte & 15) as usize] as char);
    }
    output
}

fn nanos(duration: Duration) -> u64 {
    duration.as_nanos().min(u128::from(u64::MAX)) as u64
}

fn finalize_partition(
    runner: Runner,
    guard: Option<GuardHit>,
    traversal_closed: bool,
    projection_allowed: bool,
    scratch_parent: Option<&Path>,
    scratch_tracker: &frozen_v1::ActualScratchTracker,
) -> Result<PartitionDiagnostic, EngineError> {
    let counts = runner.state.counts();
    let elapsed_monotonic_ns = nanos(runner.elapsed());
    let full = runner.limits.maximum_ast_depth == FULL_MAXIMUM_AST_DEPTH
        && runner.limits.maximum_ast_node_count == FULL_MAXIMUM_AST_NODE_COUNT;
    let terminal_status = match guard {
        Some(_) => PREFLIGHT_CAPACITY_GUARD_HIT,
        None if full => PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY,
        None => LOCAL_PROTOTYPE_SUBSET_TRAVERSAL_CLOSED,
    };
    let projection_complete = projection_allowed && guard.is_none() && traversal_closed;
    let projection = projection_complete
        .then(|| diagnostic_state_projection(&runner.state, &runner.coverage));
    let frozen_projection = if projection_complete {
        Some(frozen_v1::project_same_run(
            runner.input_signature_id,
            runner.universe_root,
            runner.limits,
            &runner.state,
            &runner.coverage,
            runner.peak_work_queue_points,
            scratch_parent,
            scratch_tracker,
        )?)
    } else {
        None
    };
    let eligible_application_count = runner
        .coverage
        .values()
        .map(|row| row.eligible)
        .sum::<u64>();
    let strict_admitted_application_count = runner
        .coverage
        .values()
        .map(|row| row.strict_admitted)
        .sum::<u64>();
    let coverage_rewrite_count = runner
        .coverage
        .values()
        .map(|row| row.rewrite_collapses)
        .sum::<u64>();
    let zero_eligible_record_count = runner
        .coverage
        .values()
        .filter(|row| row.eligible == 0)
        .count() as u64;
    let final_barrier = runner.barriers.last();
    let structural_boundary = final_barrier
        .filter(|row| row.barrier_kind == "STRUCTURAL_BOUNDARY")
        .cloned();
    let zero_delta_structural_boundary = structural_boundary.as_ref().is_some_and(|row| {
        row.eligible_raw_application_count == 0
            && row.new_behavior_class_count == 0
            && row.continuation_bank_mutation_count == 0
    });
    let fixed_point = FixedPointDiagnostic {
        work_queue_empty: traversal_closed,
        zero_delta_structural_boundary,
        all_eligible_operator_and_bank_tuple_expansions_covered: traversal_closed,
        final_new_behavior_class_delta: structural_boundary
            .as_ref()
            .map_or(0, |row| row.new_behavior_class_count),
        final_visible_frontier_delta: 0,
        final_continuation_bank_delta: structural_boundary
            .as_ref()
            .map_or(0, |row| row.continuation_bank_mutation_count),
        no_resource_guard_hit: guard.is_none(),
        saturated_diagnostic_only: traversal_closed && guard.is_none() && full,
    };
    let coverage = CoverageDiagnostic {
        coverage_record_count: runner.coverage.len() as u64,
        eligible_application_count,
        processed_application_count: eligible_application_count,
        strict_admitted_application_count,
        rewrite_collapse_count: coverage_rewrite_count,
        zero_eligible_record_count,
        all_eligible_covered: traversal_closed,
    };
    Ok(PartitionDiagnostic {
        input_signature_id: runner.input_signature_id,
        input_signature_name: if runner.input_signature_id == 1 {
            "OddInputV1"
        } else {
            "SinkInputV1"
        },
        universe_root: format!("sha256:{}", runner.universe_root),
        universe_row_count: runner.observations.len() as u64,
        limits: runner.limits,
        full_v16_structural_limits_applied: full,
        terminal_status: terminal_status.to_owned(),
        resource_guard_id: guard.map(|value| value.id),
        resource_guard_name: guard.map(|value| value.name.to_owned()),
        traversal_closed,
        frozen_leaf_count: FROZEN_LEAF_COUNT as u64,
        raw_operator_application_count: runner.raw_count,
        strict_admitted_application_count: runner.strict_count,
        rewrite_collapse_count: runner.rewrite_count,
        behavior_class_count: counts.classes,
        signature_cohort_count: counts.cohorts,
        continuation_bank_point_count: counts.bank,
        visible_frontier_point_count: counts.frontier,
        maximum_bank_points_per_class: counts.maximum_bank_per_class,
        maximum_frontier_points_per_class: counts.maximum_frontier_per_class,
        peak_raw_operator_application_count: runner.raw_count,
        peak_behavior_class_count: runner.peak.classes,
        peak_visible_frontier_point_count: runner.peak.frontier,
        peak_visible_frontier_points_per_class: runner.peak.maximum_frontier_per_class,
        peak_continuation_bank_point_count: runner.peak.bank,
        peak_continuation_bank_points_per_class: runner.peak.maximum_bank_per_class,
        peak_work_queue_points: runner.peak_work_queue_points,
        peak_saturation_round_count: runner.peak_saturation_round_count,
        vector_cache_entry_count: runner.vector_cache.len() as u64,
        vector_cache_hit_count: runner.vector_cache_hits,
        vector_cache_miss_count: runner.vector_cache_misses,
        depth_barriers: runner.barriers,
        fixed_point: projection_complete.then_some(fixed_point),
        coverage: projection_complete.then_some(coverage),
        diagnostic_state_projection: projection,
        same_run_record_set_root: frozen_projection
            .as_ref()
            .map(|value| value.same_run_record_set_root.clone()),
        formal_wire_projection: frozen_projection
            .map(|value| value.formal_wire_projection),
        elapsed_monotonic_ns,
        diagnostic_only: true,
        formal_roots_generated: false,
        formal_roots: None,
        target_truth_accessed: false,
        split_accessed: false,
        role_evaluation_performed: false,
        complete_claim_allowed: false,
    })
}

pub fn run_partition_capacity_preflight(
    input_signature_id: u64,
    limits: CapacityLimits,
) -> Result<PartitionDiagnostic, EngineError> {
    let scratch_tracker = frozen_v1::ActualScratchTracker::default();
    let partition = run_partition_capacity_preflight_inner(
        input_signature_id,
        limits,
        None,
        &scratch_tracker,
    )?;
    scratch_tracker.require_terminal_empty()?;
    Ok(partition)
}


fn traverse_partition(
    input_signature_id: u64,
    limits: CapacityLimits,
) -> Result<(Runner, Option<GuardHit>, bool), EngineError> {
    let mut runner = Runner::new(input_signature_id, limits)?;
    let traversal = (|| -> Result<(), TraversalFailure> {
        runner.seed()?;
        for depth in 1..=runner.limits.maximum_ast_depth {
            runner.expand_depth(depth)?;
        }
        runner.close_structural_boundary()?;
        Ok(())
    })();
    match traversal {
        Ok(()) => Ok((runner, None, true)),
        Err(TraversalFailure::Guard(guard)) => Ok((runner, Some(guard), false)),
        Err(TraversalFailure::Engine(error)) => Err(error),
    }
}


fn run_partition_capacity_preflight_inner(
    input_signature_id: u64,
    limits: CapacityLimits,
    scratch_parent: Option<&Path>,
    scratch_tracker: &frozen_v1::ActualScratchTracker,
) -> Result<PartitionDiagnostic, EngineError> {
    let (runner, guard, traversal_closed) = traverse_partition(input_signature_id, limits)?;
    finalize_partition(
        runner,
        guard,
        traversal_closed,
        true,
        scratch_parent,
        scratch_tracker,
    )
}

fn validate_identity(capsule_digest: &str, source_commit: &str) -> Result<(), EngineError> {
    let capsule_hex = capsule_digest.strip_prefix("sha256:").ok_or_else(|| {
        EngineError::new(
            "REJECT_PREFLIGHT_IDENTITY",
            "capsule digest must use sha256:<64 lowercase hex>",
        )
    })?;
    if capsule_hex.len() != 64
        || !capsule_hex
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        || source_commit.len() != 40
        || !source_commit
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(EngineError::new(
            "REJECT_PREFLIGHT_IDENTITY",
            "capsule digest or source commit is not exact lowercase hexadecimal identity",
        ));
    }
    Ok(())
}

fn validate_prefixed_sha256_root(value: &str, label: &str) -> Result<(), EngineError> {
    let hex = value.strip_prefix("sha256:").ok_or_else(|| {
        EngineError::new(
            "REJECT_FORMAL_MATERIALIZER_ADMISSION",
            format!("{label} must use sha256:<64 lowercase hex>"),
        )
    })?;
    if hex.len() != 64
        || !hex
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(EngineError::new(
            "REJECT_FORMAL_MATERIALIZER_ADMISSION",
            format!("{label} must use sha256:<64 lowercase hex>"),
        ));
    }
    Ok(())
}

pub fn run_capacity_preflight(
    limits: CapacityLimits,
    capsule_digest: &str,
    source_commit: &str,
) -> Result<EngineDiagnostic, EngineError> {
    run_capacity_preflight_inner(limits, capsule_digest, source_commit, None)
}

pub fn run_capacity_preflight_with_scratch(
    limits: CapacityLimits,
    capsule_digest: &str,
    source_commit: &str,
    scratch_parent: &Path,
) -> Result<EngineDiagnostic, EngineError> {
    run_capacity_preflight_inner(
        limits,
        capsule_digest,
        source_commit,
        Some(scratch_parent),
    )
}

fn run_capacity_preflight_inner(
    limits: CapacityLimits,
    capsule_digest: &str,
    source_commit: &str,
    scratch_parent: Option<&Path>,
) -> Result<EngineDiagnostic, EngineError> {
    validate_identity(capsule_digest, source_commit)?;
    let limits = limits.validate()?;
    let scratch_tracker = frozen_v1::ActualScratchTracker::default();
    let mut traversals = Vec::with_capacity(2);
    for input_signature_id in [1_u64, 2] {
        let traversal = traverse_partition(input_signature_id, limits)?;
        let guard_hit = traversal.1.is_some();
        traversals.push(traversal);
        if guard_hit {
            break;
        }
    }
    let aggregate_guard = traversals.iter().any(|row| row.1.is_some());
    let mut partitions = traversals
        .into_iter()
        .map(|(runner, guard, traversal_closed)| {
            finalize_partition(
                runner,
                guard,
                traversal_closed,
                !aggregate_guard,
                scratch_parent,
                &scratch_tracker,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let statuses = partitions
        .iter()
        .map(|partition| partition.terminal_status.as_str())
        .collect::<BTreeSet<_>>();
    let terminal_status = if partitions.len() == 2
        && statuses == BTreeSet::from([PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY])
    {
        PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY
    } else if partitions.len() == 2
        && statuses == BTreeSet::from([LOCAL_PROTOTYPE_SUBSET_TRAVERSAL_CLOSED])
    {
        LOCAL_PROTOTYPE_SUBSET_TRAVERSAL_CLOSED
    } else {
        PREFLIGHT_CAPACITY_GUARD_HIT
    };
    if terminal_status == PREFLIGHT_CAPACITY_GUARD_HIT {
        for partition in &mut partitions {
            partition.fixed_point = None;
            partition.coverage = None;
            partition.diagnostic_state_projection = None;
            partition.same_run_record_set_root = None;
            partition.formal_wire_projection = None;
        }
    }
    let formal_resource_projection = if terminal_status == PREFLIGHT_CAPACITY_GUARD_HIT {
        None
    } else {
        let formal_partitions = partitions
            .iter()
            .map(|partition| {
                partition.formal_wire_projection.as_ref().ok_or_else(|| {
                    EngineError::new(
                        "FAIL_FROZEN_V1_ENDPOINT_PROJECTION",
                        "closed aggregate is missing a formal partition projection",
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Some(frozen_v1::endpoint_formal_resource_projection(
            &formal_partitions,
        )?)
    };
    let terminal_live_scratch_bytes = scratch_tracker.require_terminal_empty()?;
    Ok(EngineDiagnostic {
        schema_version: SCHEMA_VERSION,
        endpoint_schema_version: ENDPOINT_SCHEMA_VERSION,
        actor_id: ACTOR_ID,
        implementation_id: IMPLEMENTATION_ID,
        action_id: ACTION_ID,
        capsule_digest: capsule_digest.to_owned(),
        source_commit: source_commit.to_owned(),
        dsl_version: DSL_VERSION,
        closure_semantics_version: CLOSURE_SEMANTICS_VERSION,
        terminal_status: terminal_status.to_owned(),
        actual_preflight_peak_scratch_bytes: scratch_tracker.peak_logical_bytes(),
        actual_preflight_terminal_live_scratch_bytes: terminal_live_scratch_bytes,
        projected_closure_bundle_frame_bytes: formal_resource_projection
            .map(|projection| projection.projected_closure_bundle_frame_bytes),
        projected_formal_archive_payload_bytes: formal_resource_projection
            .map(|projection| projection.projected_formal_archive_payload_bytes),
        projected_formal_total_output_bytes: formal_resource_projection
            .map(|projection| projection.projected_formal_total_output_bytes),
        projected_peak_formal_scratch_bytes: formal_resource_projection
            .map(|projection| projection.projected_peak_formal_scratch_bytes),
        partitions,
        diagnostic_only: true,
        q1_state: Q1_STATE,
        q1_gate_count: 0,
        q1_gate_mask: 0,
        q1_formal_roots: None,
        q1_receipt: None,
        q1_output_slots: [None, None, None, None, None, None, None, None],
        q1_execution_started: false,
        q2_state: "NOT_RUN",
        target_truth_accessed: false,
        split_accessed: false,
        role_evaluation_performed: false,
        active_transition_allowed: false,
    })
}

fn require_empty_materializer_directory(path: &Path, label: &str) -> Result<(), EngineError> {
    if !path.is_absolute() || !path.is_dir() {
        return Err(EngineError::new(
            "REJECT_FORMAL_MATERIALIZER_DIRECTORY",
            format!("{label} must be an existing absolute directory"),
        ));
    }
    if std::fs::read_dir(path)
        .map_err(|error| {
            EngineError::new(
                "REJECT_FORMAL_MATERIALIZER_DIRECTORY",
                format!("cannot list {label}: {error}"),
            )
        })?
        .next()
        .is_some()
    {
        return Err(EngineError::new(
            "REJECT_FORMAL_MATERIALIZER_DIRECTORY",
            format!("{label} must be empty"),
        ));
    }
    Ok(())
}

fn persist_formal_materializer_result(
    output_root: &Path,
    payload: &[u8],
) -> Result<(), EngineError> {
    let target = output_root.join("endpoint-result.json");
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&target)
        .map_err(|error| {
            EngineError::new(
                "FAIL_FORMAL_MATERIALIZER_WRITE",
                format!("cannot create {}: {error}", target.display()),
            )
        })?;
    file.write_all(payload)
        .and_then(|_| file.sync_all())
        .map_err(|error| {
            EngineError::new(
                "FAIL_FORMAL_MATERIALIZER_WRITE",
                format!("cannot persist {}: {error}", target.display()),
            )
        })?;
    File::open(output_root)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| {
            EngineError::new(
                "FAIL_FORMAL_MATERIALIZER_WRITE",
                format!("cannot sync {}: {error}", output_root.display()),
            )
        })
}

pub fn run_formal_archive_materializer_with_scratch(
    limits_by_signature: [CapacityLimits; 2],
    capsule_digest: &str,
    source_commit: &str,
    run_genesis_root: &str,
    prestart_admission_receipt_root: &str,
    output_root: &Path,
    scratch_root: &Path,
) -> Result<FormalArchiveMaterializerResult, EngineError> {
    validate_identity(capsule_digest, source_commit)?;
    validate_prefixed_sha256_root(run_genesis_root, "run genesis root")?;
    validate_prefixed_sha256_root(
        prestart_admission_receipt_root,
        "prestart admission receipt root",
    )?;
    let limits_by_signature = [
        limits_by_signature[0].validate()?,
        limits_by_signature[1].validate()?,
    ];
    require_empty_materializer_directory(output_root, "formal materializer output root")?;
    require_empty_materializer_directory(scratch_root, "formal materializer scratch root")?;
    let resolved_output = std::fs::canonicalize(output_root).map_err(|error| {
        EngineError::new(
            "REJECT_FORMAL_MATERIALIZER_DIRECTORY",
            format!("cannot resolve formal materializer output root: {error}"),
        )
    })?;
    let resolved_scratch = std::fs::canonicalize(scratch_root).map_err(|error| {
        EngineError::new(
            "REJECT_FORMAL_MATERIALIZER_DIRECTORY",
            format!("cannot resolve formal materializer scratch root: {error}"),
        )
    })?;
    if resolved_output == resolved_scratch {
        return Err(EngineError::new(
            "REJECT_FORMAL_MATERIALIZER_DIRECTORY_ALIAS",
            "formal materializer output and scratch roots must be distinct",
        ));
    }

    let mut traversals = Vec::with_capacity(2);
    for (index, input_signature_id) in [1_u64, 2].into_iter().enumerate() {
        let traversal = traverse_partition(input_signature_id, limits_by_signature[index])?;
        if traversal.1.is_some() || !traversal.2 {
            return Err(EngineError::new(
                "FAIL_FORMAL_MATERIALIZER_TRAVERSAL",
                "the exact ordered Odd/Sink traversal hit its formal resource guard",
            ));
        }
        traversals.push(traversal);
    }

    let scratch_tracker = frozen_v1::ActualScratchTracker::default();
    let mut projections = Vec::with_capacity(2);
    let mut partitions = Vec::with_capacity(2);
    for (runner, _, _) in traversals {
        let projected = frozen_v1::materialize_same_run(
            runner.input_signature_id,
            runner.universe_root,
            runner.limits,
            &runner.state,
            &runner.coverage,
            runner.peak_work_queue_points,
            Some(scratch_root),
            &scratch_tracker,
            output_root,
        )?;
        partitions.push(projected.formal_materialization.ok_or_else(|| {
            EngineError::new(
                "FAIL_FORMAL_MATERIALIZER_OUTPUT",
                "retained formal partition metadata is absent",
            )
        })?);
        projections.push(projected.formal_wire_projection);
    }
    let projection_refs = projections.iter().collect::<Vec<_>>();
    let endpoint = frozen_v1::materialize_endpoint_closure(&projection_refs, output_root)?;
    let terminal_live_scratch_bytes = scratch_tracker.require_terminal_empty()?;
    let peak_scratch_bytes = scratch_tracker.peak_logical_bytes();
    if peak_scratch_bytes > FORMAL_SCRATCH_BYTES_LIMIT {
        return Err(EngineError::new(
            "FAIL_FORMAL_MATERIALIZER_SCRATCH_GUARD",
            "formal projection scratch exceeded the frozen 4 MiB guard",
        ));
    }
    let report = FormalArchiveMaterializerResult {
        schema_version: FORMAL_MATERIALIZER_SCHEMA_VERSION,
        status: FORMAL_MATERIALIZER_STATUS,
        actor_id: FORMAL_MATERIALIZER_ACTOR_ID,
        implementation_id: FORMAL_MATERIALIZER_IMPLEMENTATION_ID,
        action_id: FORMAL_MATERIALIZER_ACTION_ID,
        capsule_digest: capsule_digest.to_owned(),
        source_commit: source_commit.to_owned(),
        run_genesis_root: run_genesis_root.to_owned(),
        prestart_admission_receipt_root: prestart_admission_receipt_root.to_owned(),
        q1_state: Q1_STATE,
        q1_gate_count: 14,
        q1_gate_total: 20,
        q1_gate_mask: 0x003fff,
        q1_output_slots: [None, None, None, None, None, None, None, None],
        dsl_version: DSL_VERSION,
        closure_semantics_version: CLOSURE_SEMANTICS_VERSION,
        maximum_ast_depth: FULL_MAXIMUM_AST_DEPTH,
        maximum_ast_node_count: FULL_MAXIMUM_AST_NODE_COUNT,
        full_v16_structural_limits_applied: limits_by_signature.iter().all(|limits| {
            limits.maximum_ast_depth == FULL_MAXIMUM_AST_DEPTH
                && limits.maximum_ast_node_count == FULL_MAXIMUM_AST_NODE_COUNT
        }),
        formal_archive_materialized: true,
        diagnostic_only: false,
        formal_capacity_limits_by_signature: [
            FormalCapacityLimitRow::from_limits(1, limits_by_signature[0]),
            FormalCapacityLimitRow::from_limits(2, limits_by_signature[1]),
        ],
        formal_output_bytes_limit: FORMAL_OUTPUT_BYTES_LIMIT,
        formal_scratch_bytes_limit: FORMAL_SCRATCH_BYTES_LIMIT,
        closure_bundle_root: endpoint.closure_bundle_root,
        formal_archive_payload_bytes: endpoint.formal_archive_payload_bytes,
        actual_formal_peak_scratch_bytes: peak_scratch_bytes,
        actual_formal_terminal_live_scratch_bytes: terminal_live_scratch_bytes,
        partitions,
        target_truth_accessed: false,
        split_accessed: false,
        role_evaluation_performed: false,
    };
    let report_payload = formal_materializer_json_line(&report)?;
    let total_output_bytes = report
        .formal_archive_payload_bytes
        .checked_add(report_payload.len() as u64)
        .ok_or_else(|| {
            EngineError::new(
                "FAIL_FORMAL_MATERIALIZER_OUTPUT_GUARD",
                "formal output byte count overflowed",
            )
        })?;
    if total_output_bytes > FORMAL_OUTPUT_BYTES_LIMIT {
        return Err(EngineError::new(
            "FAIL_FORMAL_MATERIALIZER_OUTPUT_GUARD",
            "formal archive plus endpoint result exceeded the frozen 16 MiB guard",
        ));
    }
    persist_formal_materializer_result(output_root, &report_payload)?;
    Ok(report)
}

pub fn formal_materializer_json_line(
    report: &FormalArchiveMaterializerResult,
) -> Result<Vec<u8>, EngineError> {
    let value = serde_json::to_value(report).map_err(|error| {
        EngineError::new(
            "FAIL_FORMAL_MATERIALIZER_JSON",
            format!("cannot form formal materializer result: {error}"),
        )
    })?;
    canonical_value_line(&value)
}

pub fn canonical_json_line(report: &EngineDiagnostic) -> Result<Vec<u8>, EngineError> {
    let value = serde_json::to_value(report).map_err(|error| {
        EngineError::new("FAIL_PREFLIGHT_JSON", format!("cannot form report: {error}"))
    })?;
    canonical_value_line(&value)
}

fn canonical_value_line(value: &serde_json::Value) -> Result<Vec<u8>, EngineError> {
    let mut bytes = serde_json::to_vec(value).map_err(|error| {
        EngineError::new("FAIL_PREFLIGHT_JSON", format!("cannot encode report: {error}"))
    })?;
    bytes.push(b'\n');
    Ok(bytes)
}

pub fn sidecar_payloads(
    report: &EngineDiagnostic,
) -> Result<BTreeMap<&'static str, Vec<u8>>, EngineError> {
    let common = |schema_version: &'static str, partitions: serde_json::Value| {
        json!({
            "actor_id": report.actor_id,
            "capsule_digest": report.capsule_digest,
            "diagnostic_only": true,
            "implementation_id": report.implementation_id,
            "partitions": partitions,
            "q1_formal_roots": null,
            "q1_gate_count": 0,
            "q1_gate_mask": 0,
            "q1_state": "NOT_RUN",
            "schema_version": schema_version,
            "source_commit": report.source_commit
        })
    };
    let projection_partitions = report
        .partitions
        .iter()
        .map(|partition| {
            json!({
                "diagnostic_state_projection": partition.diagnostic_state_projection,
                "formal_wire_projection": partition.formal_wire_projection,
                "input_signature_id": partition.input_signature_id,
                "input_signature_name": partition.input_signature_name,
                "universe_root": partition.universe_root
            })
        })
        .collect::<Vec<_>>();
    let coverage_partitions = report
        .partitions
        .iter()
        .map(|partition| {
            json!({
                "coverage": partition.coverage,
                "input_signature_id": partition.input_signature_id,
                "input_signature_name": partition.input_signature_name,
                "universe_root": partition.universe_root
            })
        })
        .collect::<Vec<_>>();
    let fixed_point_partitions = report
        .partitions
        .iter()
        .map(|partition| {
            json!({
                "fixed_point": partition.fixed_point,
                "input_signature_id": partition.input_signature_id,
                "input_signature_name": partition.input_signature_name,
                "same_run_record_set_root": partition.same_run_record_set_root,
                "universe_root": partition.universe_root
            })
        })
        .collect::<Vec<_>>();
    let mut payloads = BTreeMap::new();
    payloads.insert("diagnostic-summary-v1.json", canonical_json_line(report)?);
    payloads.insert(
        "projection-summary-v1.json",
        canonical_value_line(&common(
            "hegel-phase3a-q1-full-node6-projection-summary/1",
            json!(projection_partitions),
        ))?,
    );
    payloads.insert(
        "coverage-summary-v1.json",
        canonical_value_line(&common(
            "hegel-phase3a-q1-full-node6-coverage-summary/1",
            json!(coverage_partitions),
        ))?,
    );
    payloads.insert(
        "fixed-point-summary-v1.json",
        canonical_value_line(&common(
            "hegel-phase3a-q1-full-node6-fixed-point-summary/1",
            json!(fixed_point_partitions),
        ))?,
    );

    let rows = payloads
        .iter()
        .map(|(filename, payload)| {
            json!({
                "byte_count": payload.len(),
                "filename": filename,
                "sha256": hex(&sha256_bytes(&[payload]))
            })
        })
        .collect::<Vec<_>>();
    let manifest = json!({
        "actor_id": report.actor_id,
        "capsule_digest": report.capsule_digest,
        "diagnostic_only": true,
        "implementation_id": report.implementation_id,
        "q1_formal_roots": null,
        "q1_gate_count": 0,
        "q1_gate_mask": 0,
        "q1_state": "NOT_RUN",
        "rows": rows,
        "schema_version": "hegel-phase3a-q1-full-node6-endpoint-sidecar-manifest/1",
        "source_commit": report.source_commit
    });
    payloads.insert(
        "endpoint-manifest-v1.json",
        canonical_value_line(&manifest)?,
    );
    Ok(payloads)
}

#[cfg(test)]
mod fixed_point_regression_tests {
    use super::*;

    fn test_program(node: Node, behavior: &Behavior) -> ProgramRef {
        let canonical = canonicalize_shrink6_source_node(node).unwrap();
        let signature = construction_signature(&canonical).unwrap();
        Arc::new(Program {
            canonical,
            behavior: behavior.clone(),
            signature,
        })
    }

    fn schedule_row(
        operator: Operator,
        left: &ProgramRef,
        right: Option<&ProgramRef>,
    ) -> (u16, Vec<u8>, Option<Vec<u8>>) {
        (
            operator.coverage_code(),
            left.canonical.canonical_cbor.clone(),
            right.map(|program| program.canonical.canonical_cbor.clone()),
        )
    }

    fn eligible_identities(runner: &Runner, depth: u32) -> BTreeSet<ApplicationIdentity> {
        let snapshot = runner
            .state
            .continuation_programs()
            .unwrap()
            .into_iter()
            .filter(|program| program.canonical.depth < depth)
            .collect::<Vec<_>>();
        let mut identities = BTreeSet::new();
        visit_eligible_applications(
            &snapshot,
            depth,
            runner.limits.maximum_ast_node_count,
            |operator, left, right| {
                identities.insert(application_identity(
                    depth,
                    operator,
                    left,
                    right.map(AsRef::as_ref),
                ));
                Ok(())
            },
        )
        .unwrap();
        identities
    }

    #[test]
    fn application_identity_keeps_operator_and_child_tuple_distinctions() {
        let behavior = Behavior::new(
            1,
            OutputSortId::Bool,
            Arc::new(vec![RuntimeValue::Bool(false)]),
        );
        let left = test_program(Node::ContextFlag(0), &behavior);
        let right = test_program(Node::TaskFlag(0), &behavior);

        let equal = application_identity(
            3,
            Operator::Binary(BinaryOp::EqualExact),
            &left,
            Some(&right),
        );
        let less_equal = application_identity(
            3,
            Operator::Binary(BinaryOp::LessEqual),
            &left,
            Some(&right),
        );
        let reversed = application_identity(
            3,
            Operator::Binary(BinaryOp::EqualExact),
            &right,
            Some(&left),
        );

        assert_ne!(equal, less_equal);
        assert_ne!(equal, reversed);
        assert_eq!(
            equal,
            application_identity(
                3,
                Operator::Binary(BinaryOp::EqualExact),
                &left,
                Some(&right),
            )
        );
    }

    #[test]
    fn rational_visitor_matches_python_frozen_pair_schedule() {
        let behavior = Behavior::new(
            1,
            OutputSortId::RationalValue,
            Arc::new(vec![RuntimeValue::Rational(Rational::integer(0))]),
        );
        let negative_one = test_program(Node::ScalarConst(1), &behavior);
        let forward = test_program(
            Node::Binary {
                op: BinaryOp::Difference,
                left: Box::new(Node::ScalarConst(1)),
                right: Box::new(Node::ScalarConst(5)),
            },
            &behavior,
        );
        let reverse = test_program(
            Node::Binary {
                op: BinaryOp::Difference,
                left: Box::new(Node::ScalarConst(5)),
                right: Box::new(Node::ScalarConst(1)),
            },
            &behavior,
        );
        let programs = vec![reverse.clone(), negative_one.clone(), forward.clone()];
        let mut actual = Vec::new();
        visit_eligible_applications(&programs, 2, 6, |operator, left, right| {
            actual.push(schedule_row(operator, left, right));
            Ok(())
        })
        .unwrap();

        let mut global = programs.clone();
        global.sort_by_key(|program| program_sort_key(program));
        let mut commutative = programs;
        commutative.sort_by(|left, right| {
            child_order_key(left).cmp(&child_order_key(right))
        });
        assert_eq!(
            global
                .iter()
                .map(|program| program.canonical.canonical_cbor.as_slice())
                .collect::<Vec<_>>(),
            vec![
                negative_one.canonical.canonical_cbor.as_slice(),
                forward.canonical.canonical_cbor.as_slice(),
                reverse.canonical.canonical_cbor.as_slice(),
            ]
        );
        assert_eq!(
            commutative
                .iter()
                .map(|program| program.canonical.canonical_cbor.as_slice())
                .collect::<Vec<_>>(),
            vec![
                negative_one.canonical.canonical_cbor.as_slice(),
                reverse.canonical.canonical_cbor.as_slice(),
                forward.canonical.canonical_cbor.as_slice(),
            ]
        );

        let mut expected = vec![
            schedule_row(Operator::Unary(UnaryOp::Absolute), &forward, None),
            schedule_row(Operator::Unary(UnaryOp::Absolute), &reverse, None),
            schedule_row(Operator::Unary(UnaryOp::Sign), &forward, None),
            schedule_row(Operator::Unary(UnaryOp::Sign), &reverse, None),
        ];
        for (left, right) in [
            (&negative_one, &forward),
            (&negative_one, &reverse),
            (&forward, &negative_one),
            (&reverse, &negative_one),
        ] {
            expected.push(schedule_row(
                Operator::Binary(BinaryOp::Difference),
                left,
                Some(right),
            ));
            expected.push(schedule_row(
                Operator::Binary(BinaryOp::LessEqual),
                left,
                Some(right),
            ));
        }
        for (left, right) in [
            (&negative_one, &reverse),
            (&negative_one, &forward),
        ] {
            for operator in [
                Operator::Binary(BinaryOp::EqualExact),
                Operator::Approx(1),
                Operator::Approx(2),
            ] {
                expected.push(schedule_row(operator, left, Some(right)));
            }
        }

        assert_eq!(actual.len(), 18);
        assert_eq!(actual, expected);
    }

    #[test]
    fn late_lower_depth_bank_mutation_exposes_new_next_round_application() {
        let behavior = Behavior::new(
            1,
            OutputSortId::Bool,
            Arc::new(vec![RuntimeValue::Bool(false)]),
        );
        let first = test_program(Node::ContextFlag(0), &behavior);
        let later = test_program(Node::TaskFlag(0), &behavior);
        assert_eq!(first.signature, later.signature);

        let mut runner = Runner::new(1, CapacityLimits::bounded_for_test(3).unwrap()).unwrap();
        assert_eq!(runner.insert(first, 0).unwrap().3, 0);
        let before = eligible_identities(&runner, 1);
        let late = runner.insert(later, 1).unwrap();
        let after = eligible_identities(&runner, 1);

        assert_eq!(late.2, 1);
        assert_eq!(late.3, 1);
        assert!(before.is_empty());
        assert_eq!(after.len(), 1);
        assert!(after.is_disjoint(&before));
        assert_eq!(after.iter().next().unwrap().operator_code, 0x4002);
        assert_eq!(runner.state.continuation_programs().unwrap().len(), 2);
    }

    #[test]
    fn rewritten_partial_difference_uses_canonical_zero_behavior() {
        let observations = odd_universe();
        let bit = Program::from_leaf(
            canonicalize_shrink6_source_node(Node::BitAt(7)).unwrap(),
            1,
            &observations,
        )
        .unwrap();
        let scalar_source = Node::Unary {
            op: UnaryOp::BitToScalar,
            child: Box::new(bit.canonical.canonical_node.clone()),
        };
        let scalar_canonical =
            canonicalize_shrink6_source_node(scalar_source.clone()).unwrap();
        assert_eq!(scalar_canonical.canonical_node, scalar_source);
        let scalar_raw_cells = apply_operator_vector(
            Operator::Unary(UnaryOp::BitToScalar),
            &bit.behavior,
            None,
        )
        .unwrap();
        let scalar_cells = application_behavior_vector(
            &scalar_canonical,
            false,
            scalar_raw_cells.clone(),
            &observations,
        )
        .unwrap();
        assert!(Arc::ptr_eq(&scalar_raw_cells, &scalar_cells));
        let scalar = Program::from_application(scalar_canonical, 1, scalar_cells).unwrap();
        assert!(scalar
            .behavior
            .cells
            .iter()
            .any(|value| matches!(value, RuntimeValue::Bottom)));
        assert!(scalar
            .behavior
            .cells
            .iter()
            .any(|value| matches!(value, RuntimeValue::Rational(_))));

        let source = Operator::Binary(BinaryOp::Difference).node(&scalar, Some(&scalar));
        let canonical = canonicalize_shrink6_source_node(source.clone()).unwrap();
        assert_eq!(canonical.canonical_node, Node::ScalarConst(3));
        assert_ne!(canonical.canonical_node, source);

        let raw_cells = apply_operator_vector(
            Operator::Binary(BinaryOp::Difference),
            &scalar.behavior,
            Some(&scalar.behavior),
        )
        .unwrap();
        assert!(raw_cells
            .iter()
            .any(|value| matches!(value, RuntimeValue::Bottom)));
        assert!(raw_cells.iter().any(|value| {
            matches!(value, RuntimeValue::Rational(rational) if *rational == Rational::integer(0))
        }));

        let canonical_cells = application_behavior_vector(
            &canonical,
            true,
            raw_cells.clone(),
            &observations,
        )
        .unwrap();
        assert_ne!(raw_cells.as_slice(), canonical_cells.as_slice());
        assert!(canonical_cells.iter().all(|value| {
            matches!(value, RuntimeValue::Rational(rational) if *rational == Rational::integer(0))
        }));

        let canonical_cbor = canonical.canonical_cbor.clone();
        let raw_alias =
            Program::from_application(canonical.clone(), 1, raw_cells.clone()).unwrap();
        let zero = Program::from_leaf(
            canonicalize_shrink6_source_node(Node::ScalarConst(3)).unwrap(),
            1,
            &observations,
        )
        .unwrap();

        let mut alias_runner =
            Runner::new(1, CapacityLimits::bounded_for_test(6).unwrap()).unwrap();
        assert_eq!(alias_runner.insert(zero.clone(), 0).unwrap(), (1, 1, 1, 0));
        assert_eq!(alias_runner.insert(raw_alias, 2).unwrap(), (1, 1, 1, 1));
        let alias_error = alias_runner.state.continuation_programs().unwrap_err();
        assert_eq!(alias_error.code, "FAIL_Q1_PREFLIGHT_BEHAVIOR");
        for expected in [
            "canonical_cbor_sha256=sha256:",
            "depth=0",
            "prior_behavior_sha256=sha256:",
            "current_behavior_sha256=sha256:",
            "state_counts=classes:2,cohorts:2,bank:2,frontier:2",
        ] {
            assert!(
                alias_error.detail.contains(expected),
                "missing {expected:?} from {:?}",
                alias_error.detail.as_str()
            );
        }

        let rewritten = Program::from_application(canonical, 1, canonical_cells).unwrap();
        assert_eq!(
            rewritten.behavior.key.as_slice(),
            zero.behavior.key.as_slice()
        );

        let mut runner = Runner::new(1, CapacityLimits::bounded_for_test(6).unwrap()).unwrap();
        assert_eq!(runner.insert(zero, 0).unwrap(), (1, 1, 1, 0));
        assert_eq!(runner.insert(rewritten, 2).unwrap(), (0, 0, 0, 0));
        let continuations = runner.state.continuation_programs().unwrap();
        assert_eq!(
            continuations
                .iter()
                .filter(|program| {
                    program.canonical.canonical_cbor.as_slice() == canonical_cbor.as_slice()
                })
                .count(),
            1
        );
    }
}
