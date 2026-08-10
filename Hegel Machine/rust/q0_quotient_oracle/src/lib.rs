//! Independent Rust oracle for the frozen Phase-3A-Q0 micro projection.
//!
//! The existing shrink-6 canonicalizer is used only as the admission boundary.
//! Typed observations, exact evaluation, behavior encoding, MDL-aware Pareto
//! frontiers, exhaustive syntax enumeration, and quotient saturation are
//! implemented in this crate.  No production answer rows, split assignments,
//! role matcher, or target-conditioned input is accepted.

use hegel_strict_canonicalizer::{BinaryOp, CanonicalProgram, Node, Sort, UnaryOp};
use hegel_strict_canonicalizer_shrink6::canonicalize_shrink6_source_node;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub const SCHEMA_VERSION: &str = "hegel-q0-rust-micro-oracle/1";
pub const IMPLEMENTATION_ID: &str = "hegel-rust-q0-quotient-oracle-v1";
pub const DSL_VERSION: &str = "hegel-old-dsl-v1.6.0";
pub const DSL_FREEZE_VERSION: &str = "hegel-freeze-p2b-p3-v1.6.0";
pub const CLOSURE_SEMANTICS_VERSION: &str = "hegel-quotient-closure-v1.0.1";
pub const Q0_FREEZE_VERSION: &str = "hegel-freeze-p3a-q0-v1.0.1";
pub const PROJECTION_ID: &str = "hegel-q0-micro-projection-v1";
pub const Q0_QUALIFICATION_ID: &str =
    "hegel-phase3a-q0-exact-quotient-qualification-v1";
pub const NORMATIVE_DOCUMENT_SHA256_HEX: &str =
    "1df8d3ff3ede2cbead98e7901a3e82b91c460ad1d5eb0d1af78938e7b2d23b95";
pub const ADAPTER_SCHEMA_ID: &str = "hegel-phase3-q0-input-adapter/1";

pub const PROBE_INPUT_SIGNATURE_ID: u64 = 0x7001;
pub const PROBE_INPUT_TAG: u64 = 0x3606;
pub const PROBE_INPUT_SCHEMA_ID: &[u8] = b"hegel-q0-probe-input/1";
pub const PROBE_UNIVERSE_ROOT_DOMAIN: &[u8] = b"HEGEL/Q0/PROBE_UNIVERSE_ROOT/V1";
pub const EXPECTED_PROBE_CANONICAL_CBOR_HEX: &str = concat!(
    "860119360656686567656c2d71302d70726f62652d696e7075742f311970010484",
    "8301193401850119340151686567656c2d6f64642d696e7075742f310585000100",
    "01008301193401850119340151686567656c2d6f64642d696e7075742f31088801",
    "000100010001008302193402870119340252686567656c2d73696e6b2d696e7075",
    "742f31000000008302193402870119340252686567656c2d73696e6b2d696e7075",
    "742f3104010203"
);
pub const EXPECTED_PROBE_UNIVERSE_ROOT_HEX: &str =
    "2c960bcc229175afe6d5e106a34410216669bfe66b14d5c85103762c596f4192";

pub const BEHAVIOR_BLOB_TAG: u64 = 0x3601;
pub const CONSTRUCTION_SIGNATURE_TAG: u64 = 0x3602;
pub const FRONTIER_ENTRY_TAG: u64 = 0x3603;
pub const QUOTIENT_CLASS_TAG: u64 = 0x3604;
/// Reserved for the host-only dual receipt; this single endpoint never emits it.
pub const HOST_ONLY_SATURATION_RECEIPT_TAG: u64 = 0x3605;
pub const BEHAVIOR_BLOB_SCHEMA_ID: &[u8] = b"hegel-q0-behavior-blob/1";
pub const CONSTRUCTION_SIGNATURE_SCHEMA_ID: &[u8] =
    b"hegel-q0-construction-signature/1";
pub const FRONTIER_ENTRY_SCHEMA_ID: &[u8] = b"hegel-q0-frontier-entry/1";
pub const QUOTIENT_CLASS_SCHEMA_ID: &[u8] = b"hegel-q0-quotient-class/1";
pub const PROGRAM_RECORD_SCHEMA_ID: &[u8] = b"hegel-q0-syntax-program-record/1";
pub const BEHAVIOR_ID_DOMAIN: &[u8] = b"HEGEL/Q0/BEHAVIOR_ID/V1";
pub const FRONTIER_ENTRY_ID_DOMAIN: &[u8] = b"HEGEL/Q0/FRONTIER_ENTRY_ID/V1";
pub const SYNTAX_STATE_ROOT_DOMAIN: &[u8] = b"HEGEL/Q0/SYNTAX_STATE/V1";
pub const DIRECT_STATE_ROOT_DOMAIN: &[u8] = b"HEGEL/Q0/DIRECT_QUOTIENT_STATE/V1";
pub const FIXED_POINT_STATE_SCHEMA_ID: &[u8] = b"hegel-q0-fixed-point-state/1";
pub const SYNTAX_PATH_ID: &[u8] = b"hegel-q0-exhaustive-syntax-path/1";
pub const DIRECT_PATH_ID: &[u8] = b"hegel-q0-direct-quotient-path/1";
pub const ENDPOINT_STATE_SCHEMA_ID: &[u8] = b"hegel-q0-oracle-endpoint-state/1";
pub const ENDPOINT_STATE_ROOT_DOMAIN: &[u8] = b"HEGEL/Q0/ORACLE_ENDPOINT_STATE/V1";
pub const PROJECTION_MANIFEST_SCHEMA_ID: &[u8] = b"hegel-q0-projection-manifest/1";
pub const PROJECTION_MANIFEST_ROOT_DOMAIN: &[u8] = b"HEGEL/Q0/PROJECTION_MANIFEST/V1";
pub const SEMANTIC_BINDING_SCHEMA_ID: &[u8] = b"hegel-q0-semantic-binding/1";
pub const SEMANTIC_BINDING_ROOT_DOMAIN: &[u8] = b"HEGEL/Q0/SEMANTIC_BINDING/V1";
pub const EXPECTED_PROJECTION_MANIFEST_ROOT_HEX: &str =
    "2f39aa248f1305eeaf20a724f6d690cf2b13003f86620d09d2753815831f7ad1";
pub const EXPECTED_SEMANTIC_BINDING_ROOT_HEX: &str =
    "b7ec5e860a007469b8a1b3930f17c130f59a800d2a832dfd438d18a75538ff99";
pub const Q0_CHILD_DSL_SPEC_ROOT_HEX: &str =
    "da5ed2db33a88a0912d5003999f787cc26ba18564876615773a82bb742d9f8ae";
pub const Q0_OPERATOR_SEMANTICS_ROOT_HEX: &str =
    "922e48ada22dfa8621a4d516e07ec9aa7dc8fc10c165d1cafc963575aed5ec03";
pub const Q0_IDENTIFIER_REGISTRY_ROOT_HEX: &str =
    "64c9415f7759eec140e439030c5a5374851b9024d7d4849b52b995704ba76ff1";
pub const Q0_CANONICAL_AST_SCHEMA_ROOT_HEX: &str =
    "5de72fc51e27e5501561ffda6b05522f4941d1a13c4b324f5edcc15fa584a0bd";
pub const Q0_CANONICAL_CBOR_PROFILE_ROOT_HEX: &str =
    "ef0008912962de9da322eaeea6e421e1e58d16be152f968298774af0fd3249ab";

pub const MAX_AST_DEPTH: u32 = 2;
pub const MAX_AST_NODE_COUNT: u32 = 4;
pub const MAX_TOP_LEVEL_CLAUSES: usize = 2;
pub const MAX_AGGREGATE_LEAVES: u32 = 1;
pub const MAX_RAW_OPERATOR_APPLICATIONS: u64 = 5_000;
pub const MAX_CANONICAL_SYNTAX: usize = 2_000;
pub const MAX_BEHAVIOR_CLASSES: usize = 2_000;
pub const MAX_FRONTIER_POINTS: usize = 2_000;
pub const MAX_FRONTIER_POINTS_PER_CLASS: usize = 64;
pub const MAX_SATURATION_ROUNDS: u64 = 4;
pub const FROZEN_LEAF_COUNT: usize = 15;
pub const MAX_OUTPUT_BYTES: u64 = 64 * 1024 * 1024;
pub const MAX_WALL_TIME_SECONDS: u64 = 300;
pub const MAX_MEMORY_BYTES: u64 = 512 * 1024 * 1024;

pub const SINGLE_IMPLEMENTATION_PASS_STATUS: &str =
    "SINGLE_IMPLEMENTATION_EXHAUSTIVE_ORACLE_EQUIVALENCE_PASS";
pub const INCONCLUSIVE_RESOURCE_LIMIT: &str = "INCONCLUSIVE_RESOURCE_LIMIT";
pub const FAIL_SEMANTICS_MISMATCH: &str = "FAIL_SEMANTICS_MISMATCH";
pub const FAIL_SHA256_PREIMAGE_COLLISION: &str = "FAIL_SHA256_PREIMAGE_COLLISION";
pub const REJECT_Q0_PROJECTION: &str = "REJECT_Q0_PROJECTION";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[repr(u64)]
pub enum ResourceGuardId {
    RawOperatorApplications = 1,
    CanonicalSyntaxPrograms = 2,
    BehaviorClasses = 3,
    TotalFrontierPoints = 4,
    FrontierPointsPerClass = 5,
    SaturationRounds = 6,
    WallTime = 7,
    ResidentMemory = 8,
    OutputBytes = 9,
    TotalContinuationBankPoints = 10,
    ContinuationBankPointsPerClass = 11,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OracleError {
    pub code: String,
    pub detail: String,
    pub guard_id: Option<u64>,
}

impl OracleError {
    fn new(code: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            detail: detail.into(),
            guard_id: None,
        }
    }

    fn resource(guard: ResourceGuardId, detail: impl Into<String>) -> Self {
        Self {
            code: INCONCLUSIVE_RESOURCE_LIMIT.to_owned(),
            detail: detail.into(),
            guard_id: Some(guard as u64),
        }
    }
}

impl fmt::Display for OracleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.code, self.detail)
    }
}

impl std::error::Error for OracleError {}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum CborValue {
    Unsigned(u64),
    Negative(u64),
    Bytes(Vec<u8>),
    Array(Vec<CborValue>),
    Bool(bool),
}

fn cbor_uint(value: u64) -> CborValue {
    CborValue::Unsigned(value)
}

fn cbor_int(value: i64) -> CborValue {
    if value >= 0 {
        CborValue::Unsigned(value as u64)
    } else {
        CborValue::Negative((-1 - value) as u64)
    }
}

fn encode_head(major: u8, argument: u64, output: &mut Vec<u8>) {
    let prefix = major << 5;
    match argument {
        0..=23 => output.push(prefix | argument as u8),
        24..=0xff => {
            output.push(prefix | 24);
            output.push(argument as u8);
        }
        0x100..=0xffff => {
            output.push(prefix | 25);
            output.extend_from_slice(&(argument as u16).to_be_bytes());
        }
        0x1_0000..=0xffff_ffff => {
            output.push(prefix | 26);
            output.extend_from_slice(&(argument as u32).to_be_bytes());
        }
        _ => {
            output.push(prefix | 27);
            output.extend_from_slice(&argument.to_be_bytes());
        }
    }
}

fn encode_cbor_into(value: &CborValue, output: &mut Vec<u8>) {
    match value {
        CborValue::Unsigned(value) => encode_head(0, *value, output),
        CborValue::Negative(argument) => encode_head(1, *argument, output),
        CborValue::Bytes(bytes) => {
            encode_head(2, bytes.len() as u64, output);
            output.extend_from_slice(bytes);
        }
        CborValue::Array(values) => {
            encode_head(4, values.len() as u64, output);
            for child in values {
                encode_cbor_into(child, output);
            }
        }
        CborValue::Bool(false) => output.push(0xf4),
        CborValue::Bool(true) => output.push(0xf5),
    }
}

fn encode_cbor(value: &CborValue) -> Vec<u8> {
    let mut output = Vec::new();
    encode_cbor_into(value, &mut output);
    output
}

fn sha256(parts: &[&[u8]]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update(part);
    }
    hasher.finalize().into()
}

fn content_hash(domain: &[u8], object: &CborValue) -> [u8; 32] {
    sha256(&[domain, &[0], &encode_cbor(object)])
}

pub fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        result.push(HEX[(byte >> 4) as usize] as char);
        result.push(HEX[(byte & 0x0f) as usize] as char);
    }
    result
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Rational {
    numerator: i64,
    denominator: i64,
}

impl Rational {
    fn new(numerator: i64, denominator: i64) -> Result<Self, OracleError> {
        if denominator == 0 {
            return Err(OracleError::new(
                FAIL_SEMANTICS_MISMATCH,
                "zero rational denominator",
            ));
        }
        let mut numerator = numerator;
        let mut denominator = denominator;
        if denominator < 0 {
            numerator = numerator.checked_neg().ok_or_else(|| {
                OracleError::new(FAIL_SEMANTICS_MISMATCH, "rational sign overflow")
            })?;
            denominator = denominator.checked_neg().ok_or_else(|| {
                OracleError::new(FAIL_SEMANTICS_MISMATCH, "rational sign overflow")
            })?;
        }
        let divisor = gcd(numerator.unsigned_abs(), denominator as u64) as i64;
        Ok(Self {
            numerator: numerator / divisor,
            denominator: denominator / divisor,
        })
    }

    fn integer(value: i64) -> Self {
        Self {
            numerator: value,
            denominator: 1,
        }
    }

    fn checked_add(self, other: Self) -> Option<Self> {
        let numerator = (self.numerator as i128)
            .checked_mul(other.denominator as i128)?
            .checked_add((other.numerator as i128).checked_mul(self.denominator as i128)?)?;
        let denominator = (self.denominator as i128).checked_mul(other.denominator as i128)?;
        let numerator = i64::try_from(numerator).ok()?;
        let denominator = i64::try_from(denominator).ok()?;
        Self::new(numerator, denominator).ok()
    }

    fn checked_difference(self, other: Self) -> Option<Self> {
        self.checked_add(Self {
            numerator: other.numerator.checked_neg()?,
            denominator: other.denominator,
        })
    }

    fn absolute(self) -> Option<Self> {
        Some(Self {
            numerator: self.numerator.checked_abs()?,
            denominator: self.denominator,
        })
    }

    fn in_value_grid(self) -> bool {
        self.numerator.abs() <= 64 && (1..=8).contains(&self.denominator)
    }

    fn compare(self, other: Self) -> Ordering {
        ((self.numerator as i128) * (other.denominator as i128))
            .cmp(&((other.numerator as i128) * (self.denominator as i128)))
    }
}

fn gcd(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left.max(1)
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum RuntimeValue {
    Bottom,
    Bool(bool),
    Bit(u8),
    Sign(i8),
    BoundedInt(i8),
    Rational(Rational),
}

impl RuntimeValue {
    fn is_bottom(&self) -> bool {
        matches!(self, Self::Bottom)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ObservationEnvironment {
    Odd { bits: Vec<u8> },
    Sink { values: [Rational; 4] },
}

impl ObservationEnvironment {
    fn set_size(&self) -> i8 {
        match self {
            Self::Odd { bits } => bits.len() as i8,
            Self::Sink { .. } => 4,
        }
    }

    fn bit_at(&self, index: u64) -> RuntimeValue {
        match self {
            Self::Odd { bits } => bits
                .get(index as usize)
                .copied()
                .map(RuntimeValue::Bit)
                .unwrap_or(RuntimeValue::Bottom),
            Self::Sink { .. } => RuntimeValue::Bottom,
        }
    }

    fn aggregate(
        &self,
        map_id: u64,
        scope_id: u64,
        quantity_id: u64,
        extension: &[(u64, bool)],
    ) -> RuntimeValue {
        let values = match self {
            Self::Odd { .. } => return RuntimeValue::Bottom,
            Self::Sink { values } => values,
        };
        if scope_id != 3 || quantity_id != 0 || !extension.is_empty() {
            return RuntimeValue::Bottom;
        }
        match map_id {
            0 => {
                let mut result = Rational::integer(0);
                for value in values {
                    result = match result.checked_add(*value) {
                        Some(value) => value,
                        None => return RuntimeValue::Bottom,
                    };
                }
                bounded_rational(result)
            }
            1 => RuntimeValue::BoundedInt(
                values
                    .iter()
                    .filter(|value| value.numerator != 0)
                    .count() as i8,
            ),
            5 => {
                let orientations = [1_i64, 1, -1, -1];
                let mut result = Rational::integer(0);
                for (orientation, value) in orientations.into_iter().zip(values) {
                    let oriented = Rational::new(
                        value.numerator * orientation,
                        value.denominator,
                    )
                    .expect("small frozen orientation cannot fail");
                    result = match result.checked_add(oriented) {
                        Some(value) => value,
                        None => return RuntimeValue::Bottom,
                    };
                }
                bounded_rational(result)
            }
            _ => RuntimeValue::Bottom,
        }
    }
}

fn bounded_rational(value: Rational) -> RuntimeValue {
    if value.in_value_grid() {
        RuntimeValue::Rational(value)
    } else {
        RuntimeValue::Bottom
    }
}

fn odd_input_object(set_size: u64, bits: &[u8]) -> CborValue {
    debug_assert_eq!(set_size as usize, bits.len());
    CborValue::Array(vec![
        cbor_uint(1),
        cbor_uint(0x3401),
        CborValue::Bytes(b"hegel-odd-input/1".to_vec()),
        cbor_uint(set_size),
        CborValue::Array(bits.iter().map(|bit| cbor_uint(u64::from(*bit))).collect()),
    ])
}

fn sink_input_object(values: [u64; 4]) -> CborValue {
    let mut fields = vec![
        cbor_uint(1),
        cbor_uint(0x3402),
        CborValue::Bytes(b"hegel-sink-input/1".to_vec()),
    ];
    fields.extend(values.into_iter().map(cbor_uint));
    CborValue::Array(fields)
}

fn frozen_probe_rows() -> Vec<(u64, u64, CborValue, ObservationEnvironment)> {
    vec![
        (
            1,
            0x3401,
            odd_input_object(5, &[0, 1, 0, 1, 0]),
            ObservationEnvironment::Odd {
                bits: vec![0, 1, 0, 1, 0],
            },
        ),
        (
            1,
            0x3401,
            odd_input_object(8, &[1, 0, 1, 0, 1, 0, 1, 0]),
            ObservationEnvironment::Odd {
                bits: vec![1, 0, 1, 0, 1, 0, 1, 0],
            },
        ),
        (
            2,
            0x3402,
            sink_input_object([0, 0, 0, 0]),
            ObservationEnvironment::Sink {
                values: [Rational::integer(0); 4],
            },
        ),
        (
            2,
            0x3402,
            sink_input_object([4, 1, 2, 3]),
            ObservationEnvironment::Sink {
                values: [
                    Rational::integer(4),
                    Rational::integer(1),
                    Rational::integer(2),
                    Rational::integer(3),
                ],
            },
        ),
    ]
}

fn probe_object() -> CborValue {
    let rows = frozen_probe_rows()
        .into_iter()
        .map(|(signature_id, tag, object, _)| {
            CborValue::Array(vec![
                cbor_uint(signature_id),
                cbor_uint(tag),
                object,
            ])
        })
        .collect();
    CborValue::Array(vec![
        cbor_uint(1),
        cbor_uint(PROBE_INPUT_TAG),
        CborValue::Bytes(PROBE_INPUT_SCHEMA_ID.to_vec()),
        cbor_uint(PROBE_INPUT_SIGNATURE_ID),
        cbor_uint(4),
        CborValue::Array(rows),
    ])
}

pub fn probe_canonical_bytes() -> Vec<u8> {
    encode_cbor(&probe_object())
}

pub fn probe_universe_root() -> [u8; 32] {
    content_hash(PROBE_UNIVERSE_ROOT_DOMAIN, &probe_object())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[repr(u64)]
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct BehaviorBlob {
    output_sort: OutputSortId,
    cells: Vec<RuntimeValue>,
}

impl BehaviorBlob {
    fn cell_cbor(&self, value: &RuntimeValue) -> Result<CborValue, OracleError> {
        if value.is_bottom() {
            return Ok(CborValue::Array(vec![cbor_uint(0)]));
        }
        let payload = match (self.output_sort, value) {
            (OutputSortId::Bool, RuntimeValue::Bool(value)) => CborValue::Bool(*value),
            (OutputSortId::Bit, RuntimeValue::Bit(value)) if *value <= 1 => {
                cbor_uint(u64::from(*value))
            }
            (OutputSortId::Sign, RuntimeValue::Sign(value)) if (-1..=1).contains(value) => {
                cbor_int(i64::from(*value))
            }
            (OutputSortId::BoundedInt, RuntimeValue::BoundedInt(value))
                if (-8..=8).contains(value) =>
            {
                cbor_int(i64::from(*value))
            }
            (OutputSortId::RationalValue, RuntimeValue::Rational(value)) => {
                CborValue::Array(vec![
                    cbor_int(value.numerator),
                    cbor_uint(value.denominator as u64),
                ])
            }
            _ => {
                return Err(OracleError::new(
                    FAIL_SEMANTICS_MISMATCH,
                    "runtime value does not match the strict output sort",
                ))
            }
        };
        Ok(CborValue::Array(vec![cbor_uint(1), payload]))
    }

    fn canonical_object(&self) -> Result<CborValue, OracleError> {
        let cells = self
            .cells
            .iter()
            .map(|cell| self.cell_cbor(cell))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(CborValue::Array(vec![
            cbor_uint(1),
            cbor_uint(BEHAVIOR_BLOB_TAG),
            CborValue::Bytes(BEHAVIOR_BLOB_SCHEMA_ID.to_vec()),
            cbor_uint(PROBE_INPUT_SIGNATURE_ID),
            CborValue::Bytes(probe_universe_root().to_vec()),
            cbor_uint(self.output_sort as u64),
            cbor_uint(cells.len() as u64),
            CborValue::Array(cells),
        ]))
    }

    fn canonical_bytes(&self) -> Result<Vec<u8>, OracleError> {
        Ok(encode_cbor(&self.canonical_object()?))
    }

    fn behavior_id(&self) -> Result<[u8; 32], OracleError> {
        Ok(content_hash(BEHAVIOR_ID_DOMAIN, &self.canonical_object()?))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u64)]
pub enum NormalizationProfileId {
    General = 0,
    AbsoluteRoot = 1,
    ConstNegativeOne = 2,
    ConstZero = 3,
    ConstPositiveOne = 4,
    TopLevelAnd2 = 5,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Signature {
    output_sort: OutputSortId,
    ast_depth: u32,
    ast_node_count: u32,
    scalar_parameter_occurrence_count: u32,
    aggregate_leaf_count: u32,
    distinct_bit_slot_bitmap: u8,
    scope_clause_count: u32,
    top_level_clause_count: u32,
    old_law_composition_depth: u32,
    normalization_profile: NormalizationProfileId,
    mdl_length_q32: u64,
}

impl Signature {
    fn canonical_object(&self) -> CborValue {
        CborValue::Array(vec![
            cbor_uint(1),
            cbor_uint(CONSTRUCTION_SIGNATURE_TAG),
            CborValue::Bytes(CONSTRUCTION_SIGNATURE_SCHEMA_ID.to_vec()),
            cbor_uint(self.output_sort as u64),
            cbor_uint(u64::from(self.ast_depth)),
            cbor_uint(u64::from(self.ast_node_count)),
            cbor_uint(u64::from(self.scalar_parameter_occurrence_count)),
            cbor_uint(u64::from(self.aggregate_leaf_count)),
            cbor_uint(u64::from(self.distinct_bit_slot_bitmap)),
            cbor_uint(u64::from(self.scope_clause_count)),
            cbor_uint(u64::from(self.top_level_clause_count)),
            cbor_uint(u64::from(self.old_law_composition_depth)),
            cbor_uint(self.normalization_profile as u64),
            cbor_uint(self.mdl_length_q32),
        ])
    }

    fn canonical_bytes(&self) -> Vec<u8> {
        encode_cbor(&self.canonical_object())
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
            u64::from(self.scalar_parameter_occurrence_count),
            u64::from(self.aggregate_leaf_count),
            u64::from(self.scope_clause_count),
            u64::from(self.top_level_clause_count),
            u64::from(self.old_law_composition_depth),
            self.mdl_length_q32,
        ];
        let right = [
            u64::from(other.ast_depth),
            u64::from(other.ast_node_count),
            u64::from(other.scalar_parameter_occurrence_count),
            u64::from(other.aggregate_leaf_count),
            u64::from(other.scope_clause_count),
            u64::from(other.top_level_clause_count),
            u64::from(other.old_law_composition_depth),
            other.mdl_length_q32,
        ];
        let no_worse = subset && left.iter().zip(right.iter()).all(|(a, b)| a <= b);
        let strict = self.distinct_bit_slot_bitmap != other.distinct_bit_slot_bitmap
            || left.iter().zip(right.iter()).any(|(a, b)| a < b);
        no_worse && strict
    }
}

#[derive(Debug, Clone)]
struct Program {
    canonical: CanonicalProgram,
    behavior: BehaviorBlob,
    signature: Signature,
}

impl Program {
    fn frontier_object(&self, normalization_witness_rank: u64) -> CborValue {
        CborValue::Array(vec![
            cbor_uint(1),
            cbor_uint(FRONTIER_ENTRY_TAG),
            CborValue::Bytes(FRONTIER_ENTRY_SCHEMA_ID.to_vec()),
            self.signature.canonical_object(),
            cbor_uint(normalization_witness_rank),
            CborValue::Bytes(self.canonical.canonical_cbor.clone()),
            CborValue::Bytes(self.canonical.canonical_ast_hash.to_vec()),
        ])
    }

}

fn strict_bottom_unary(child: RuntimeValue) -> Option<RuntimeValue> {
    if child.is_bottom() {
        Some(RuntimeValue::Bottom)
    } else {
        None
    }
}

fn strict_bottom_binary(left: &RuntimeValue, right: &RuntimeValue) -> Option<RuntimeValue> {
    if left.is_bottom() || right.is_bottom() {
        Some(RuntimeValue::Bottom)
    } else {
        None
    }
}

fn evaluate_node(
    node: &Node,
    environment: &ObservationEnvironment,
) -> Result<RuntimeValue, OracleError> {
    match node {
        Node::ScalarConst(index) => match index {
            1 => Ok(RuntimeValue::Rational(Rational::integer(-1))),
            3 => Ok(RuntimeValue::Rational(Rational::integer(0))),
            5 => Ok(RuntimeValue::Rational(Rational::integer(1))),
            _ => Err(OracleError::new(
                FAIL_SEMANTICS_MISMATCH,
                format!("non-active rational parameter {index} reached evaluation"),
            )),
        },
        Node::BitAt(index) if *index < 8 => Ok(environment.bit_at(*index)),
        Node::BitAt(index) => Err(OracleError::new(
            FAIL_SEMANTICS_MISMATCH,
            format!("invalid bit slot {index} reached evaluation"),
        )),
        Node::SetSize => Ok(RuntimeValue::BoundedInt(environment.set_size())),
        Node::Aggregate {
            map_id,
            scope_id,
            quantity_id,
            scope_extension,
        } if [0, 1, 5].contains(map_id) => Ok(environment.aggregate(
            *map_id,
            *scope_id,
            *quantity_id,
            scope_extension,
        )),
        Node::Aggregate { map_id, .. } => Err(OracleError::new(
            FAIL_SEMANTICS_MISMATCH,
            format!("non-active aggregate map {map_id} reached evaluation"),
        )),
        Node::ContextFlag(index) if *index < 4 => Ok(RuntimeValue::Bottom),
        Node::TaskFlag(index) if *index < 2 => Ok(RuntimeValue::Bottom),
        Node::ContextFlag(index) | Node::TaskFlag(index) => Err(OracleError::new(
            FAIL_SEMANTICS_MISMATCH,
            format!("invalid flag ID {index} reached evaluation"),
        )),
        Node::NewSymbolCall(_) => Err(OracleError::new(
            FAIL_SEMANTICS_MISMATCH,
            "new symbol reached old-DSL evaluator",
        )),
        Node::Unary { op, child } => {
            let child = evaluate_node(child, environment)?;
            if let Some(bottom) = strict_bottom_unary(child.clone()) {
                return Ok(bottom);
            }
            match (op, child) {
                (UnaryOp::BitToScalar, RuntimeValue::Bit(value)) => {
                    Ok(RuntimeValue::Rational(Rational::integer(i64::from(value))))
                }
                (UnaryOp::IntToScalar, RuntimeValue::BoundedInt(value)) => {
                    Ok(RuntimeValue::Rational(Rational::integer(i64::from(value))))
                }
                (UnaryOp::Absolute, RuntimeValue::Rational(value)) => Ok(value
                    .absolute()
                    .map(bounded_rational)
                    .unwrap_or(RuntimeValue::Bottom)),
                (UnaryOp::Sign, RuntimeValue::Rational(value)) => {
                    Ok(RuntimeValue::Sign(value.numerator.signum() as i8))
                }
                _ => Err(OracleError::new(
                    FAIL_SEMANTICS_MISMATCH,
                    "strict unary operator received a value of the wrong sort",
                )),
            }
        }
        Node::Binary { op, left, right } => {
            let left = evaluate_node(left, environment)?;
            let right = evaluate_node(right, environment)?;
            if let Some(bottom) = strict_bottom_binary(&left, &right) {
                return Ok(bottom);
            }
            match (op, left, right) {
                (
                    BinaryOp::Difference,
                    RuntimeValue::Rational(left),
                    RuntimeValue::Rational(right),
                ) => Ok(left
                    .checked_difference(right)
                    .map(bounded_rational)
                    .unwrap_or(RuntimeValue::Bottom)),
                (
                    BinaryOp::EqualExact,
                    RuntimeValue::Rational(left),
                    RuntimeValue::Rational(right),
                ) => Ok(RuntimeValue::Bool(left == right)),
                (
                    BinaryOp::LessEqual,
                    RuntimeValue::Rational(left),
                    RuntimeValue::Rational(right),
                ) => Ok(RuntimeValue::Bool(left.compare(right) != Ordering::Greater)),
                (
                    BinaryOp::SameSign,
                    RuntimeValue::Sign(left),
                    RuntimeValue::Sign(right),
                ) => Ok(RuntimeValue::Bool(left == right)),
                (
                    BinaryOp::OppositeSign,
                    RuntimeValue::Sign(left),
                    RuntimeValue::Sign(right),
                ) => Ok(RuntimeValue::Bool(left != 0 && left == -right)),
                (BinaryOp::Add | BinaryOp::GreaterEqual, _, _) => Err(OracleError::new(
                    FAIL_SEMANTICS_MISMATCH,
                    "non-canonical binary alias reached evaluation",
                )),
                _ => Err(OracleError::new(
                    FAIL_SEMANTICS_MISMATCH,
                    "strict binary operator received values of the wrong sort",
                )),
            }
        }
        Node::ApproxEqual {
            left,
            right,
            tolerance_index,
        } => {
            let left = evaluate_node(left, environment)?;
            let right = evaluate_node(right, environment)?;
            if let Some(bottom) = strict_bottom_binary(&left, &right) {
                return Ok(bottom);
            }
            let tolerance = match tolerance_index {
                1 => Rational::new(1, 4)?,
                2 => Rational::new(1, 2)?,
                _ => {
                    return Err(OracleError::new(
                        FAIL_SEMANTICS_MISMATCH,
                        "non-surviving tolerance reached evaluation",
                    ))
                }
            };
            match (left, right) {
                (RuntimeValue::Rational(left), RuntimeValue::Rational(right)) => {
                    let distance = left
                        .checked_difference(right)
                        .and_then(Rational::absolute)
                        .ok_or_else(|| {
                            OracleError::new(
                                FAIL_SEMANTICS_MISMATCH,
                                "approximate-equality arithmetic overflow",
                            )
                        })?;
                    Ok(RuntimeValue::Bool(
                        distance.compare(tolerance) != Ordering::Greater,
                    ))
                }
                _ => Err(OracleError::new(
                    FAIL_SEMANTICS_MISMATCH,
                    "approximate equality received values of the wrong sort",
                )),
            }
        }
        Node::And(children) if children.len() == 2 => {
            let mut result = true;
            for child in children {
                match evaluate_node(child, environment)? {
                    RuntimeValue::Bottom => return Ok(RuntimeValue::Bottom),
                    RuntimeValue::Bool(value) => result &= value,
                    _ => {
                        return Err(OracleError::new(
                            FAIL_SEMANTICS_MISMATCH,
                            "AND2 received a non-Bool child",
                        ))
                    }
                }
            }
            Ok(RuntimeValue::Bool(result))
        }
        Node::And(_) => Err(OracleError::new(
            FAIL_SEMANTICS_MISMATCH,
            "non-AND2 canonical node reached the micro evaluator",
        )),
    }
}

fn evaluate_behavior(program: &CanonicalProgram) -> Result<BehaviorBlob, OracleError> {
    let cells = frozen_probe_rows()
        .into_iter()
        .map(|(_, _, _, environment)| evaluate_node(&program.canonical_node, &environment))
        .collect::<Result<Vec<_>, _>>()?;
    let behavior = BehaviorBlob {
        output_sort: OutputSortId::from_sort(program.output_sort),
        cells,
    };
    behavior.canonical_bytes()?;
    Ok(behavior)
}

fn elias_delta_length(value: u64) -> Result<u64, OracleError> {
    if value == 0 {
        return Err(OracleError::new(
            FAIL_SEMANTICS_MISMATCH,
            "Elias-delta ID must be one-based",
        ));
    }
    let log_n = 63 - u64::from(value.leading_zeros());
    let log_log = 63 - u64::from((log_n + 1).leading_zeros());
    Ok(log_n + 2 * log_log + 1)
}

fn mdl_bit_length(node: &Node) -> Result<u64, OracleError> {
    match node {
        Node::ScalarConst(index) if [1, 3, 5].contains(index) => Ok(8),
        Node::BitAt(index) if *index < 8 => Ok(5 + elias_delta_length(index + 1)?),
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
        Node::ContextFlag(index) if *index < 4 => Ok(5 + elias_delta_length(index + 1)?),
        Node::TaskFlag(index) if *index < 2 => Ok(5 + elias_delta_length(index + 1)?),
        Node::Unary { child, .. } => Ok(4 + mdl_bit_length(child)?),
        Node::Binary { left, right, .. } => {
            Ok(5 + mdl_bit_length(left)? + mdl_bit_length(right)?)
        }
        Node::ApproxEqual {
            left,
            right,
            tolerance_index,
        } if (1..=2).contains(tolerance_index) => {
            Ok(6 + mdl_bit_length(left)? + mdl_bit_length(right)?)
        }
        Node::And(children) if children.len() == 2 => Ok(
            5 + children
                .iter()
                .map(mdl_bit_length)
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .sum::<u64>(),
        ),
        _ => Err(OracleError::new(
            FAIL_SEMANTICS_MISMATCH,
            "canonical program has no frozen MDL code",
        )),
    }
}

#[derive(Debug, Default)]
struct ResourceMetrics {
    bit_slot_bitmap: u8,
    scope_clause_count: u32,
}

fn collect_resource_metrics(node: &Node, metrics: &mut ResourceMetrics) {
    match node {
        Node::BitAt(index) if *index < 8 => metrics.bit_slot_bitmap |= 1 << *index,
        Node::Aggregate {
            scope_extension, ..
        } => metrics.scope_clause_count += scope_extension.len() as u32,
        Node::Unary { child, .. } => collect_resource_metrics(child, metrics),
        Node::Binary { left, right, .. } | Node::ApproxEqual { left, right, .. } => {
            collect_resource_metrics(left, metrics);
            collect_resource_metrics(right, metrics);
        }
        Node::And(children) => {
            for child in children {
                collect_resource_metrics(child, metrics);
            }
        }
        _ => {}
    }
}

fn normalization_profile(node: &Node) -> NormalizationProfileId {
    match node {
        Node::Unary {
            op: UnaryOp::Absolute,
            ..
        } => NormalizationProfileId::AbsoluteRoot,
        Node::ScalarConst(1) => NormalizationProfileId::ConstNegativeOne,
        Node::ScalarConst(3) => NormalizationProfileId::ConstZero,
        Node::ScalarConst(5) => NormalizationProfileId::ConstPositiveOne,
        Node::And(children) if children.len() == 2 => NormalizationProfileId::TopLevelAnd2,
        _ => NormalizationProfileId::General,
    }
}

fn signature_from_program(program: &CanonicalProgram) -> Result<Signature, OracleError> {
    let mut resources = ResourceMetrics::default();
    collect_resource_metrics(&program.canonical_node, &mut resources);
    let mdl_bits = mdl_bit_length(&program.canonical_node)?;
    let mdl_length_q32 = mdl_bits.checked_shl(32).ok_or_else(|| {
        OracleError::new(FAIL_SEMANTICS_MISMATCH, "MDL Q32 overflow")
    })?;
    Ok(Signature {
        output_sort: OutputSortId::from_sort(program.output_sort),
        ast_depth: program.depth,
        ast_node_count: program.node_count,
        scalar_parameter_occurrence_count: program.scalar_parameter_occurrence_count,
        aggregate_leaf_count: program.aggregate_leaf_count,
        distinct_bit_slot_bitmap: resources.bit_slot_bitmap,
        scope_clause_count: resources.scope_clause_count,
        top_level_clause_count: match &program.canonical_node {
            Node::And(children) => children.len() as u32,
            _ => 0,
        },
        old_law_composition_depth: 0,
        normalization_profile: normalization_profile(&program.canonical_node),
        mdl_length_q32,
    })
}

fn micro_admit(node: Node) -> Result<Option<Program>, OracleError> {
    let canonical = match canonicalize_shrink6_source_node(node) {
        Ok(program) => program,
        Err(error) => {
            return match error.code.as_str() {
                "REJECT_STRUCTURAL_LIMIT" => Ok(None),
                _ => Err(OracleError::new(error.code, error.message)),
            }
        }
    };
    if !canonical_node_in_projection(&canonical.canonical_node, true) {
        return Err(OracleError::new(
            REJECT_Q0_PROJECTION,
            "strict canonical AST contains a leaf/operator outside the frozen Q0 projection manifest",
        ));
    }
    if canonical.depth > MAX_AST_DEPTH
        || canonical.node_count > MAX_AST_NODE_COUNT
        || canonical.aggregate_leaf_count > MAX_AGGREGATE_LEAVES
    {
        return Ok(None);
    }
    if matches!(&canonical.canonical_node, Node::And(children) if children.len() != MAX_TOP_LEVEL_CLAUSES)
    {
        return Ok(None);
    }
    let behavior = evaluate_behavior(&canonical)?;
    let signature = signature_from_program(&canonical)?;
    Ok(Some(Program {
        canonical,
        behavior,
        signature,
    }))
}

fn frozen_leaf_nodes() -> Vec<(u16, Node)> {
    vec![
        (0x0000, Node::ScalarConst(1)),
        (0x0001, Node::ScalarConst(3)),
        (0x0002, Node::ScalarConst(5)),
        (0x0003, Node::BitAt(0)),
        (0x0004, Node::BitAt(1)),
        (0x0005, Node::SetSize),
        (
            0x0006,
            Node::Aggregate {
                map_id: 0,
                scope_id: 3,
                quantity_id: 0,
                scope_extension: vec![],
            },
        ),
        (
            0x0007,
            Node::Aggregate {
                map_id: 1,
                scope_id: 3,
                quantity_id: 0,
                scope_extension: vec![],
            },
        ),
        (
            0x0008,
            Node::Aggregate {
                map_id: 5,
                scope_id: 3,
                quantity_id: 0,
                scope_extension: vec![],
            },
        ),
        (
            0x0009,
            Node::Aggregate {
                map_id: 0,
                scope_id: 0,
                quantity_id: 0,
                scope_extension: vec![],
            },
        ),
        (
            0x000a,
            Node::Aggregate {
                map_id: 0,
                scope_id: 3,
                quantity_id: 1,
                scope_extension: vec![],
            },
        ),
        (
            0x000b,
            Node::Aggregate {
                map_id: 0,
                scope_id: 3,
                quantity_id: 0,
                scope_extension: vec![(0, true)],
            },
        ),
        (
            0x000c,
            Node::Aggregate {
                map_id: 1,
                scope_id: 1,
                quantity_id: 0,
                scope_extension: vec![],
            },
        ),
        (0x000d, Node::ContextFlag(0)),
        (0x000e, Node::TaskFlag(0)),
    ]
}

fn binary_operator_id(operator: BinaryOp) -> u64 {
    match operator {
        BinaryOp::Add => 0,
        BinaryOp::Difference => 1,
        BinaryOp::EqualExact => 2,
        BinaryOp::LessEqual => 3,
        BinaryOp::GreaterEqual => 4,
        BinaryOp::SameSign => 5,
        BinaryOp::OppositeSign => 6,
    }
}

fn unary_operator_id(operator: UnaryOp) -> u64 {
    match operator {
        UnaryOp::BitToScalar => 0,
        UnaryOp::IntToScalar => 1,
        UnaryOp::Absolute => 2,
        UnaryOp::Sign => 3,
    }
}

fn node_cbor_object(node: &Node) -> CborValue {
    match node {
        Node::ScalarConst(index) => CborValue::Array(vec![
            cbor_uint(0),
            cbor_uint(0),
            cbor_uint(*index),
        ]),
        Node::BitAt(index) => CborValue::Array(vec![
            cbor_uint(0),
            cbor_uint(1),
            cbor_uint(*index),
        ]),
        Node::SetSize => CborValue::Array(vec![cbor_uint(0), cbor_uint(2)]),
        Node::Aggregate {
            map_id,
            scope_id,
            quantity_id,
            scope_extension,
        } => CborValue::Array(vec![
            cbor_uint(0),
            cbor_uint(3),
            cbor_uint(*map_id),
            cbor_uint(*scope_id),
            cbor_uint(*quantity_id),
            CborValue::Array(
                scope_extension
                    .iter()
                    .map(|(context_id, expected)| {
                        CborValue::Array(vec![
                            cbor_uint(*context_id),
                            CborValue::Bool(*expected),
                        ])
                    })
                    .collect(),
            ),
        ]),
        Node::ContextFlag(index) => CborValue::Array(vec![
            cbor_uint(0),
            cbor_uint(4),
            cbor_uint(*index),
        ]),
        Node::TaskFlag(index) => CborValue::Array(vec![
            cbor_uint(0),
            cbor_uint(5),
            cbor_uint(*index),
        ]),
        Node::NewSymbolCall(index) => CborValue::Array(vec![
            cbor_uint(0),
            cbor_uint(6),
            cbor_uint(*index),
        ]),
        Node::Unary { op, child } => CborValue::Array(vec![
            cbor_uint(1),
            cbor_uint(unary_operator_id(*op)),
            node_cbor_object(child),
        ]),
        Node::Binary { op, left, right } => CborValue::Array(vec![
            cbor_uint(2),
            cbor_uint(binary_operator_id(*op)),
            node_cbor_object(left),
            node_cbor_object(right),
        ]),
        Node::ApproxEqual {
            left,
            right,
            tolerance_index,
        } => CborValue::Array(vec![
            cbor_uint(3),
            cbor_uint(0),
            node_cbor_object(left),
            node_cbor_object(right),
            cbor_uint(*tolerance_index),
        ]),
        Node::And(children) => CborValue::Array(vec![
            cbor_uint(4),
            CborValue::Array(children.iter().map(node_cbor_object).collect()),
        ]),
    }
}

fn canonical_node_in_projection(node: &Node, top_level: bool) -> bool {
    if frozen_leaf_nodes().iter().any(|(_, leaf)| leaf == node) {
        return true;
    }
    match node {
        Node::Unary { op, child } => {
            matches!(
                op,
                UnaryOp::BitToScalar
                    | UnaryOp::IntToScalar
                    | UnaryOp::Absolute
                    | UnaryOp::Sign
            ) && canonical_node_in_projection(child, false)
                && !matches!(child.as_ref(), Node::And(_))
        }
        Node::Binary { op, left, right } => {
            matches!(
                op,
                BinaryOp::Difference
                    | BinaryOp::EqualExact
                    | BinaryOp::LessEqual
                    | BinaryOp::SameSign
                    | BinaryOp::OppositeSign
            ) && !matches!(left.as_ref(), Node::And(_))
                && !matches!(right.as_ref(), Node::And(_))
                && canonical_node_in_projection(left, false)
                && canonical_node_in_projection(right, false)
        }
        Node::ApproxEqual {
            left,
            right,
            tolerance_index,
        } => {
            matches!(tolerance_index, 1 | 2)
                && !matches!(left.as_ref(), Node::And(_))
                && !matches!(right.as_ref(), Node::And(_))
                && canonical_node_in_projection(left, false)
                && canonical_node_in_projection(right, false)
        }
        Node::And(children) => {
            top_level
                && children.len() == 2
                && children.iter().all(|child| {
                    !matches!(child, Node::And(_)) && canonical_node_in_projection(child, false)
                })
        }
        _ => false,
    }
}

fn coverage_codes() -> Vec<u64> {
    (0_u64..15)
        .chain(0x1000_u64..=0x1003)
        .chain([0x2001, 0x2002, 0x2003, 0x2005, 0x2006])
        .chain([0x3001, 0x3002, 0x4002])
        .collect()
}

fn projection_manifest_object() -> CborValue {
    let leaves = frozen_leaf_nodes()
        .into_iter()
        .map(|(_, node)| node_cbor_object(&node))
        .collect();
    let capacities = [
        (OutputSortId::Bool, 2_u64),
        (OutputSortId::Bit, 1),
        (OutputSortId::Sign, 1),
        (OutputSortId::BoundedInt, 1),
        (OutputSortId::RationalValue, 2),
    ]
    .into_iter()
    .map(|(sort, capacity)| {
        CborValue::Array(vec![cbor_uint(sort as u64), cbor_uint(capacity)])
    })
    .collect();
    let guard_registry = [
        (1_u64, b"RAW_OPERATOR_APPLICATIONS".as_slice()),
        (2, b"CANONICAL_SYNTAX_PROGRAMS".as_slice()),
        (3, b"BEHAVIOR_CLASSES".as_slice()),
        (4, b"TOTAL_FRONTIER_POINTS".as_slice()),
        (5, b"FRONTIER_POINTS_PER_CLASS".as_slice()),
        (6, b"SATURATION_ROUNDS".as_slice()),
        (7, b"WALL_TIME".as_slice()),
        (8, b"RESIDENT_MEMORY".as_slice()),
        (9, b"OUTPUT_BYTES".as_slice()),
        (10, b"TOTAL_CONTINUATION_BANK_POINTS".as_slice()),
        (11, b"CONTINUATION_BANK_POINTS_PER_CLASS".as_slice()),
    ]
    .into_iter()
    .map(|(id, name)| {
        CborValue::Array(vec![cbor_uint(id), CborValue::Bytes(name.to_vec())])
    })
    .collect();
    CborValue::Array(vec![
        cbor_uint(1),
        CborValue::Bytes(PROJECTION_MANIFEST_SCHEMA_ID.to_vec()),
        CborValue::Bytes(PROJECTION_ID.as_bytes().to_vec()),
        CborValue::Array(leaves),
        CborValue::Array((0_u64..=3).map(cbor_uint).collect()),
        CborValue::Array([1_u64, 2, 3, 5, 6].into_iter().map(cbor_uint).collect()),
        CborValue::Array([1_u64, 2].into_iter().map(cbor_uint).collect()),
        cbor_uint(2),
        CborValue::Array(
            [2_u64, 4, 2, 1, 3, 2, 4]
                .into_iter()
                .map(cbor_uint)
                .collect(),
        ),
        CborValue::Array(
            [
                MAX_RAW_OPERATOR_APPLICATIONS,
                MAX_CANONICAL_SYNTAX as u64,
                MAX_BEHAVIOR_CLASSES as u64,
                MAX_FRONTIER_POINTS as u64,
                MAX_FRONTIER_POINTS_PER_CLASS as u64,
                MAX_FRONTIER_POINTS as u64,
                MAX_FRONTIER_POINTS_PER_CLASS as u64,
                MAX_SATURATION_ROUNDS,
                MAX_OUTPUT_BYTES,
                MAX_WALL_TIME_SECONDS,
                MAX_MEMORY_BYTES,
            ]
            .into_iter()
            .map(cbor_uint)
            .collect(),
        ),
        CborValue::Array(capacities),
        CborValue::Bytes(b"LEX_MIN_REAL_AST_UP_TO_SORT_CAPACITY".to_vec()),
        CborValue::Bytes(
            b"EXPAND_EACH_BANK_REP_ONCE_REGARDLESS_OF_VISIBLE_DOMINANCE".to_vec(),
        ),
        CborValue::Bytes(b"PUBLIC_CLASS_ARCHIVE_VISIBLE_FRONTIER_ONLY".to_vec()),
        CborValue::Array(guard_registry),
        CborValue::Array(coverage_codes().into_iter().map(cbor_uint).collect()),
        cbor_uint(6),
    ])
}

pub fn projection_manifest_root() -> [u8; 32] {
    content_hash(PROJECTION_MANIFEST_ROOT_DOMAIN, &projection_manifest_object())
}

fn semantic_binding_object() -> Result<CborValue, OracleError> {
    Ok(CborValue::Array(vec![
        cbor_uint(1),
        CborValue::Bytes(SEMANTIC_BINDING_SCHEMA_ID.to_vec()),
        CborValue::Bytes(DSL_VERSION.as_bytes().to_vec()),
        CborValue::Bytes(DSL_FREEZE_VERSION.as_bytes().to_vec()),
        CborValue::Bytes(CLOSURE_SEMANTICS_VERSION.as_bytes().to_vec()),
        CborValue::Bytes(Q0_FREEZE_VERSION.as_bytes().to_vec()),
        CborValue::Bytes(Q0_QUALIFICATION_ID.as_bytes().to_vec()),
        CborValue::Bytes(parse_hex32(NORMATIVE_DOCUMENT_SHA256_HEX)?.to_vec()),
        CborValue::Bytes(projection_manifest_root().to_vec()),
        CborValue::Bytes(parse_hex32(Q0_CHILD_DSL_SPEC_ROOT_HEX)?.to_vec()),
        CborValue::Bytes(parse_hex32(Q0_OPERATOR_SEMANTICS_ROOT_HEX)?.to_vec()),
        CborValue::Bytes(parse_hex32(Q0_IDENTIFIER_REGISTRY_ROOT_HEX)?.to_vec()),
        CborValue::Bytes(parse_hex32(Q0_CANONICAL_AST_SCHEMA_ROOT_HEX)?.to_vec()),
        CborValue::Bytes(parse_hex32(Q0_CANONICAL_CBOR_PROFILE_ROOT_HEX)?.to_vec()),
        CborValue::Bytes(ADAPTER_SCHEMA_ID.as_bytes().to_vec()),
        CborValue::Bytes(PROJECTION_ID.as_bytes().to_vec()),
        CborValue::Bytes(probe_universe_root().to_vec()),
    ]))
}

pub fn semantic_binding_root() -> Result<[u8; 32], OracleError> {
    Ok(content_hash(
        SEMANTIC_BINDING_ROOT_DOMAIN,
        &semantic_binding_object()?,
    ))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OperatorToken {
    Unary(UnaryOp),
    Binary(BinaryOp),
    ApproxEqual(u64),
    And2,
}

impl OperatorToken {
    fn code(self) -> u16 {
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
            Self::ApproxEqual(1) => 0x3001,
            Self::ApproxEqual(2) => 0x3002,
            Self::And2 => 0x4002,
            Self::Binary(BinaryOp::Add | BinaryOp::GreaterEqual)
            | Self::ApproxEqual(_)=> unreachable!("non-canonical operator token"),
        }
    }

    fn child_sorts(self) -> &'static [OutputSortId] {
        match self {
            Self::Unary(UnaryOp::BitToScalar) => &[OutputSortId::Bit],
            Self::Unary(UnaryOp::IntToScalar) => &[OutputSortId::BoundedInt],
            Self::Unary(UnaryOp::Absolute | UnaryOp::Sign) => {
                &[OutputSortId::RationalValue]
            }
            Self::Binary(BinaryOp::Difference)
            | Self::Binary(BinaryOp::EqualExact)
            | Self::Binary(BinaryOp::LessEqual)
            | Self::ApproxEqual(_) => {
                &[OutputSortId::RationalValue, OutputSortId::RationalValue]
            }
            Self::Binary(BinaryOp::SameSign | BinaryOp::OppositeSign) => {
                &[OutputSortId::Sign, OutputSortId::Sign]
            }
            Self::And2 => &[OutputSortId::Bool, OutputSortId::Bool],
            Self::Binary(BinaryOp::Add | BinaryOp::GreaterEqual) => {
                unreachable!("non-canonical operator token")
            }
        }
    }

    fn commutative(self) -> bool {
        matches!(
            self,
            Self::Binary(BinaryOp::EqualExact)
                | Self::Binary(BinaryOp::SameSign)
                | Self::Binary(BinaryOp::OppositeSign)
                | Self::ApproxEqual(_)
                | Self::And2
        )
    }

    fn build_node(self, children: &[Program]) -> Node {
        match self {
            Self::Unary(op) => Node::Unary {
                op,
                child: Box::new(children[0].canonical.canonical_node.clone()),
            },
            Self::Binary(op) => Node::Binary {
                op,
                left: Box::new(children[0].canonical.canonical_node.clone()),
                right: Box::new(children[1].canonical.canonical_node.clone()),
            },
            Self::ApproxEqual(tolerance_index) => Node::ApproxEqual {
                left: Box::new(children[0].canonical.canonical_node.clone()),
                right: Box::new(children[1].canonical.canonical_node.clone()),
                tolerance_index,
            },
            Self::And2 => {
                let mut ordered = children.iter().collect::<Vec<_>>();
                ordered.sort_by(|left, right| {
                    left.canonical.canonical_cbor[2..]
                        .cmp(&right.canonical.canonical_cbor[2..])
                });
                Node::And(
                    ordered
                        .into_iter()
                        .map(|child| child.canonical.canonical_node.clone())
                        .collect(),
                )
            }
        }
    }
}

const OPERATORS: [OperatorToken; 12] = [
    OperatorToken::Unary(UnaryOp::BitToScalar),
    OperatorToken::Unary(UnaryOp::IntToScalar),
    OperatorToken::Unary(UnaryOp::Absolute),
    OperatorToken::Unary(UnaryOp::Sign),
    OperatorToken::Binary(BinaryOp::Difference),
    OperatorToken::Binary(BinaryOp::EqualExact),
    OperatorToken::Binary(BinaryOp::LessEqual),
    OperatorToken::Binary(BinaryOp::SameSign),
    OperatorToken::Binary(BinaryOp::OppositeSign),
    OperatorToken::ApproxEqual(1),
    OperatorToken::ApproxEqual(2),
    OperatorToken::And2,
];

fn child_order_key(program: &Program) -> ([u8; 32], &[u8]) {
    debug_assert!(program.canonical.canonical_cbor.starts_with(&[0x82, 0x01]));
    let node_cbor = &program.canonical.canonical_cbor[2..];
    (sha256(&[node_cbor]), node_cbor)
}

fn no_nested_and(program: &Program) -> bool {
    !matches!(program.canonical.canonical_node, Node::And(_))
}

fn resource_eligible(operator: OperatorToken, children: &[Program]) -> bool {
    if children.len() != operator.child_sorts().len()
        || children
            .iter()
            .zip(operator.child_sorts())
            .any(|(child, expected)| child.signature.output_sort != *expected)
    {
        return false;
    }
    // AND is a top-level constructor; a previously formed conjunction is not
    // an eligible child of any micro-projection operator.
    if children.iter().any(|child| !no_nested_and(child)) {
        return false;
    }
    let depth = 1 + children
        .iter()
        .map(|child| child.signature.ast_depth)
        .max()
        .unwrap_or(0);
    let nodes = 1 + children
        .iter()
        .map(|child| child.signature.ast_node_count)
        .sum::<u32>();
    let aggregate_leaves = children
        .iter()
        .map(|child| child.signature.aggregate_leaf_count)
        .sum::<u32>();
    let scalar_parameters = children
        .iter()
        .map(|child| child.signature.scalar_parameter_occurrence_count)
        .sum::<u32>();
    let scope_clauses = children
        .iter()
        .map(|child| child.signature.scope_clause_count)
        .sum::<u32>();
    let bitmask = children.iter().fold(0_u8, |mask, child| {
        mask | child.signature.distinct_bit_slot_bitmap
    });
    depth <= MAX_AST_DEPTH
        && nodes <= MAX_AST_NODE_COUNT
        && aggregate_leaves <= MAX_AGGREGATE_LEAVES
        && scalar_parameters <= 3
        && scope_clauses <= 2
        && bitmask.count_ones() <= 4
}

#[derive(Debug, Clone)]
struct Application {
    operator: OperatorToken,
    children: Vec<Program>,
    key: Vec<u8>,
}

impl Application {
    fn node(&self) -> Node {
        self.operator.build_node(&self.children)
    }
}

fn application_key(operator: OperatorToken, children: &[Program]) -> Vec<u8> {
    let mut key = Vec::new();
    key.extend_from_slice(&operator.code().to_be_bytes());
    key.push(children.len() as u8);
    for child in children {
        let bytes = &child.canonical.canonical_cbor;
        key.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
        key.extend_from_slice(bytes);
    }
    key
}

fn eligible_applications(programs: &[Program]) -> Vec<Application> {
    let mut by_sort: BTreeMap<OutputSortId, Vec<Program>> = BTreeMap::new();
    for program in programs {
        by_sort
            .entry(program.signature.output_sort)
            .or_default()
            .push(program.clone());
    }
    for values in by_sort.values_mut() {
        values.sort_by(|left, right| {
            child_order_key(left)
                .cmp(&child_order_key(right))
                .then_with(|| left.canonical.canonical_cbor.cmp(&right.canonical.canonical_cbor))
        });
    }

    let mut applications = Vec::new();
    for operator in OPERATORS {
        let sorts = operator.child_sorts();
        if sorts.len() == 1 {
            if let Some(children) = by_sort.get(&sorts[0]) {
                for child in children {
                    let tuple = vec![child.clone()];
                    if resource_eligible(operator, &tuple) {
                        applications.push(Application {
                            operator,
                            key: application_key(operator, &tuple),
                            children: tuple,
                        });
                    }
                }
            }
            continue;
        }
        let Some(lefts) = by_sort.get(&sorts[0]) else {
            continue;
        };
        let Some(rights) = by_sort.get(&sorts[1]) else {
            continue;
        };
        if operator.commutative() && sorts[0] == sorts[1] {
            for left_index in 0..lefts.len() {
                let first_right = if matches!(operator, OperatorToken::And2) {
                    left_index + 1
                } else {
                    left_index
                };
                for right_index in first_right..rights.len() {
                    let tuple = vec![lefts[left_index].clone(), rights[right_index].clone()];
                    if resource_eligible(operator, &tuple) {
                        applications.push(Application {
                            operator,
                            key: application_key(operator, &tuple),
                            children: tuple,
                        });
                    }
                }
            }
        } else {
            for left in lefts {
                for right in rights {
                    let tuple = vec![left.clone(), right.clone()];
                    if resource_eligible(operator, &tuple) {
                        applications.push(Application {
                            operator,
                            key: application_key(operator, &tuple),
                            children: tuple,
                        });
                    }
                }
            }
        }
    }
    applications.sort_by(|left, right| left.key.cmp(&right.key));
    applications
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct CoverageRow {
    pub operator_code: u16,
    pub eligible_raw: u64,
    pub strict_admitted: u64,
    pub rewrite_collapses: u64,
    pub canonical_duplicates: u64,
    pub new_canonical: u64,
}

impl CoverageRow {
    fn canonical_object(&self) -> CborValue {
        CborValue::Array(vec![
            cbor_uint(u64::from(self.operator_code)),
            cbor_uint(self.eligible_raw),
            cbor_uint(self.strict_admitted),
            cbor_uint(self.rewrite_collapses),
            cbor_uint(self.canonical_duplicates),
            cbor_uint(self.new_canonical),
        ])
    }
}

fn coverage_root(coverage: &BTreeMap<u16, CoverageRow>) -> [u8; 32] {
    let records = coverage
        .values()
        .map(|row| encode_cbor(&row.canonical_object()))
        .collect::<Vec<_>>();
    rfc6962_root(&records)
}

fn rfc6962_root(records: &[Vec<u8>]) -> [u8; 32] {
    match records.len() {
        0 => sha256(&[b""]),
        1 => sha256(&[&[0], &records[0]]),
        length => {
            let split = 1_usize << ((length - 1).ilog2());
            let left = rfc6962_root(&records[..split]);
            let right = rfc6962_root(&records[split..]);
            sha256(&[&[1], &left, &right])
        }
    }
}

fn coverage_row_mut(
    coverage: &mut BTreeMap<u16, CoverageRow>,
    operator_code: u16,
) -> &mut CoverageRow {
    coverage.entry(operator_code).or_insert_with(|| CoverageRow {
        operator_code,
        ..CoverageRow::default()
    })
}

fn empty_coverage_registry() -> BTreeMap<u16, CoverageRow> {
    let mut registry = BTreeMap::new();
    for code in (0_u16..FROZEN_LEAF_COUNT as u16).chain(
        OPERATORS
            .iter()
            .map(|operator| operator.code()),
    ) {
        registry.insert(
            code,
            CoverageRow {
                operator_code: code,
                ..CoverageRow::default()
            },
        );
    }
    debug_assert_eq!(registry.len(), 27);
    registry
}

#[derive(Debug)]
struct SyntaxOracleResult {
    programs: BTreeMap<Vec<u8>, Program>,
    coverage: BTreeMap<u16, CoverageRow>,
    raw_applications: u64,
    saturation_rounds: u64,
    zero_delta_full_round: bool,
}

fn insert_syntax_program_atomically(
    programs: &mut BTreeMap<Vec<u8>, Program>,
    program: Program,
) -> Result<bool, OracleError> {
    if programs.contains_key(&program.canonical.canonical_cbor) {
        return Ok(false);
    }
    if programs.len() >= MAX_CANONICAL_SYNTAX {
        return Err(OracleError::resource(
            ResourceGuardId::CanonicalSyntaxPrograms,
            "syntax oracle canonical-program guard would be exceeded",
        ));
    }
    programs.insert(program.canonical.canonical_cbor.clone(), program);
    Ok(true)
}

fn exhaustive_syntax_oracle() -> Result<SyntaxOracleResult, OracleError> {
    let mut programs = BTreeMap::new();
    let mut coverage = empty_coverage_registry();
    for (leaf_code, leaf) in frozen_leaf_nodes() {
        let row = coverage_row_mut(&mut coverage, leaf_code);
        row.eligible_raw += 1;
        let program = micro_admit(leaf.clone())?.ok_or_else(|| {
            OracleError::new(
                FAIL_SEMANTICS_MISMATCH,
                format!("frozen leaf {leaf_code:#06x} failed micro admission"),
            )
        })?;
        row.strict_admitted += 1;
        if program.canonical.canonical_node != leaf {
            row.rewrite_collapses += 1;
        }
        if programs
            .insert(program.canonical.canonical_cbor.clone(), program)
            .is_some()
        {
            row.canonical_duplicates += 1;
        } else {
            row.new_canonical += 1;
        }
    }
    if programs.len() != FROZEN_LEAF_COUNT {
        return Err(OracleError::new(
            FAIL_SEMANTICS_MISMATCH,
            "the fifteen frozen leaves did not retain distinct canonical identities",
        ));
    }

    let mut covered = BTreeSet::new();
    let mut raw_applications = FROZEN_LEAF_COUNT as u64;
    let mut saturation_rounds = 0_u64;
    loop {
        saturation_rounds += 1;
        if saturation_rounds > MAX_SATURATION_ROUNDS {
            return Err(OracleError::resource(
                ResourceGuardId::SaturationRounds,
                "syntax oracle exceeded maximum saturation rounds",
            ));
        }
        let snapshot = programs.values().cloned().collect::<Vec<_>>();
        let mut added = 0_usize;
        for application in eligible_applications(&snapshot) {
            if !covered.insert(application.key.clone()) {
                continue;
            }
            if raw_applications >= MAX_RAW_OPERATOR_APPLICATIONS {
                return Err(OracleError::resource(
                    ResourceGuardId::RawOperatorApplications,
                    "syntax oracle raw-application guard would be exceeded",
                ));
            }
            raw_applications += 1;
            let raw_node = application.node();
            let row = coverage_row_mut(&mut coverage, application.operator.code());
            row.eligible_raw += 1;
            let Some(program) = micro_admit(raw_node.clone())? else {
                continue;
            };
            row.strict_admitted += 1;
            if program.canonical.canonical_node != raw_node {
                row.rewrite_collapses += 1;
            }
            if insert_syntax_program_atomically(&mut programs, program)? {
                row.new_canonical += 1;
                added += 1;
            } else {
                row.canonical_duplicates += 1;
            }
        }
        if added == 0 {
            let final_snapshot = programs.values().cloned().collect::<Vec<_>>();
            let all_covered = eligible_applications(&final_snapshot)
                .iter()
                .all(|application| covered.contains(&application.key));
            if !all_covered {
                return Err(OracleError::new(
                    FAIL_SEMANTICS_MISMATCH,
                    "syntax zero-delta round left an eligible tuple uncovered",
                ));
            }
            break;
        }
    }
    Ok(SyntaxOracleResult {
        programs,
        coverage,
        raw_applications,
        saturation_rounds,
        zero_delta_full_round: true,
    })
}

fn deterministic_frontier(programs: &[Program]) -> Vec<Program> {
    let mut exact: BTreeMap<Vec<u8>, Vec<Program>> = BTreeMap::new();
    for program in programs {
        let key = program.signature.canonical_bytes();
        exact.entry(key).or_default().push(program.clone());
    }
    let cohorts = exact
        .into_values()
        .map(|mut cohort| {
            cohort.sort_by(|left, right| {
                left.canonical.canonical_cbor.cmp(&right.canonical.canonical_cbor)
            });
            cohort.dedup_by(|left, right| {
                left.canonical.canonical_cbor == right.canonical.canonical_cbor
            });
            let cap = match cohort[0].signature.output_sort {
                OutputSortId::Bool | OutputSortId::RationalValue => 2,
                OutputSortId::Bit | OutputSortId::Sign | OutputSortId::BoundedInt => 1,
            };
            cohort.truncate(cap);
            cohort
        })
        .collect::<Vec<_>>();
    let retained_cohorts = cohorts
        .iter()
        .filter(|candidate| {
            !cohorts.iter().any(|other| {
                other[0].signature.dominates(&candidate[0].signature)
                    && other.len() >= candidate.len()
            })
        })
        .cloned()
        .collect::<Vec<_>>();
    let mut retained = retained_cohorts
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    retained.sort_by(|left, right| {
        left.signature
            .canonical_bytes()
            .cmp(&right.signature.canonical_bytes())
            .then_with(|| left.canonical.canonical_cbor.cmp(&right.canonical.canonical_cbor))
    });
    retained
}

fn normalization_witness_capacity(sort: OutputSortId) -> usize {
    match sort {
        OutputSortId::Bool | OutputSortId::RationalValue => 2,
        OutputSortId::Bit | OutputSortId::Sign | OutputSortId::BoundedInt => 1,
    }
}

fn add_to_cohort_bank(bank: &mut BTreeMap<Vec<u8>, Vec<Program>>, program: Program) -> bool {
    let signature_key = program.signature.canonical_bytes();
    let before = bank
        .get(&signature_key)
        .map(|cohort| {
            cohort
                .iter()
                .map(|entry| entry.canonical.canonical_cbor.clone())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let cohort = bank
        .entry(signature_key)
        .or_default();
    cohort.push(program);
    cohort.sort_by(|left, right| {
        left.canonical
            .canonical_cbor
            .cmp(&right.canonical.canonical_cbor)
    });
    cohort.dedup_by(|left, right| {
        left.canonical.canonical_cbor == right.canonical.canonical_cbor
    });
    let capacity = normalization_witness_capacity(cohort[0].signature.output_sort);
    cohort.truncate(capacity);
    before
        != cohort
            .iter()
            .map(|entry| entry.canonical.canonical_cbor.clone())
            .collect::<Vec<_>>()
}

fn cohort_bank_from_programs(programs: &[Program]) -> BTreeMap<Vec<u8>, Vec<Program>> {
    let mut bank = BTreeMap::new();
    for program in programs {
        let _ = add_to_cohort_bank(&mut bank, program.clone());
    }
    bank
}

fn frontier_from_cohort_bank(bank: &BTreeMap<Vec<u8>, Vec<Program>>) -> Vec<Program> {
    deterministic_frontier(
        &bank
            .values()
            .flat_map(|cohort| cohort.iter().cloned())
            .collect::<Vec<_>>(),
    )
}

fn ranked_frontier(frontier: &[Program]) -> Vec<(&Program, u64)> {
    let mut prior_signature: Option<Vec<u8>> = None;
    let mut rank = 0_u64;
    frontier
        .iter()
        .map(|program| {
            let signature = program.signature.canonical_bytes();
            if prior_signature.as_ref() == Some(&signature) {
                rank += 1;
            } else {
                rank = 0;
                prior_signature = Some(signature);
            }
            (program, rank)
        })
        .collect()
}

#[derive(Debug, Clone)]
struct QuotientClass {
    behavior_id: [u8; 32],
    behavior_bytes: Vec<u8>,
    behavior: BehaviorBlob,
    frontier: Vec<Program>,
    cohort_bank: BTreeMap<Vec<u8>, Vec<Program>>,
    minimum_mdl_q32: u64,
}

impl QuotientClass {
    fn canonical_object(&self, class_index: u64) -> Result<CborValue, OracleError> {
        Ok(CborValue::Array(vec![
            cbor_uint(1),
            cbor_uint(QUOTIENT_CLASS_TAG),
            CborValue::Bytes(QUOTIENT_CLASS_SCHEMA_ID.to_vec()),
            cbor_uint(class_index),
            self.behavior.canonical_object()?,
            CborValue::Bytes(self.behavior_id.to_vec()),
            cbor_uint(self.frontier.len() as u64),
            CborValue::Array(
                ranked_frontier(&self.frontier)
                    .into_iter()
                    .map(|(program, rank)| program.frontier_object(rank))
                    .collect(),
            ),
            cbor_uint(self.minimum_mdl_q32),
        ]))
    }
}

#[derive(Debug, Clone, Default)]
struct QuotientState {
    classes: BTreeMap<[u8; 32], QuotientClass>,
}

impl QuotientState {
    fn class_count(&self) -> usize {
        self.classes.len()
    }

    fn frontier_count(&self) -> usize {
        self.classes
            .values()
            .map(|class| class.frontier.len())
            .sum()
    }

    fn maximum_frontier_size(&self) -> usize {
        self.classes
            .values()
            .map(|class| class.frontier.len())
            .max()
            .unwrap_or(0)
    }

    fn continuation_bank_count(&self) -> usize {
        self.classes
            .values()
            .map(|class| {
                class
                    .cohort_bank
                    .values()
                    .map(Vec::len)
                    .sum::<usize>()
            })
            .sum()
    }

    fn maximum_bank_size(&self) -> usize {
        self.classes
            .values()
            .map(|class| class.cohort_bank.values().map(Vec::len).sum::<usize>())
            .max()
            .unwrap_or(0)
    }

    fn continuation_programs(&self) -> Vec<Program> {
        let mut by_cbor = BTreeMap::new();
        for program in self
            .classes
            .values()
            .flat_map(|class| class.cohort_bank.values())
            .flat_map(|cohort| cohort.iter())
        {
            by_cbor.insert(
                program.canonical.canonical_cbor.clone(),
                program.clone(),
            );
        }
        by_cbor.into_values().collect()
    }

    fn ordered_classes(&self) -> Vec<&QuotientClass> {
        let mut classes = self.classes.values().collect::<Vec<_>>();
        classes.sort_by(|left, right| {
            left.behavior_id
                .cmp(&right.behavior_id)
                .then_with(|| left.behavior_bytes.cmp(&right.behavior_bytes))
        });
        classes
    }

    fn class_records(&self) -> Result<Vec<Vec<u8>>, OracleError> {
        self.ordered_classes()
            .into_iter()
            .enumerate()
            .map(|(index, class)| {
                Ok(encode_cbor(&class.canonical_object(index as u64)?))
            })
            .collect()
    }

    fn class_archive_root(&self) -> Result<[u8; 32], OracleError> {
        Ok(rfc6962_root(&self.class_records()?))
    }

    fn visible_class_objects(&self) -> Result<Vec<CborValue>, OracleError> {
        self.ordered_classes()
            .into_iter()
            .enumerate()
            .map(|(index, class)| class.canonical_object(index as u64))
            .collect()
    }

    fn continuation_bank_object(&self) -> CborValue {
        let mut rows = Vec::new();
        for class in self.classes.values() {
            for cohort in class.cohort_bank.values() {
                let signature = cohort[0].signature.canonical_object();
                let signature_bytes = encode_cbor(&signature);
                let entries = cohort
                    .iter()
                    .enumerate()
                    .map(|(rank, program)| {
                        CborValue::Array(vec![
                            cbor_uint(rank as u64),
                            CborValue::Bytes(program.canonical.canonical_cbor.clone()),
                            CborValue::Bytes(program.canonical.canonical_ast_hash.to_vec()),
                        ])
                    })
                    .collect::<Vec<_>>();
                rows.push((
                    class.behavior_id,
                    class.behavior_bytes.clone(),
                    signature_bytes,
                    CborValue::Array(vec![
                        CborValue::Bytes(class.behavior_id.to_vec()),
                        CborValue::Bytes(class.behavior_bytes.clone()),
                        signature,
                        CborValue::Array(entries),
                    ]),
                ));
            }
        }
        rows.sort_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then_with(|| left.1.cmp(&right.1))
                .then_with(|| left.2.cmp(&right.2))
        });
        CborValue::Array(rows.into_iter().map(|row| row.3).collect())
    }

    fn saturation_state_object(
        &self,
        programs: &BTreeMap<Vec<u8>, Program>,
        coverage: &BTreeMap<u16, CoverageRow>,
        metadata: &FixedPointMetadata,
    ) -> Result<CborValue, OracleError> {
        Ok(CborValue::Array(vec![
            CborValue::Array(program_records(programs)),
            self.continuation_bank_object(),
            CborValue::Array(self.visible_class_objects()?),
            CborValue::Array(
                coverage
                    .values()
                    .map(CoverageRow::canonical_object)
                    .collect(),
            ),
            metadata.canonical_object(),
        ]))
    }

    fn saturation_state_root(
        &self,
        programs: &BTreeMap<Vec<u8>, Program>,
        coverage: &BTreeMap<u16, CoverageRow>,
        metadata: &FixedPointMetadata,
        domain: &[u8],
    ) -> Result<[u8; 32], OracleError> {
        Ok(content_hash(
            domain,
            &self.saturation_state_object(programs, coverage, metadata)?,
        ))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FixedPointMetadata {
    path_id: &'static [u8],
    saturation_round_count: u64,
    work_queue_empty: bool,
    zero_delta_full_round: bool,
    all_eligible_tuples_covered: bool,
    final_new_program_delta: u64,
    final_class_delta: u64,
    final_frontier_delta: u64,
    final_bank_delta: u64,
}

impl FixedPointMetadata {
    fn canonical_object(&self) -> CborValue {
        CborValue::Array(vec![
            cbor_uint(1),
            CborValue::Bytes(FIXED_POINT_STATE_SCHEMA_ID.to_vec()),
            CborValue::Bytes(self.path_id.to_vec()),
            cbor_uint(self.saturation_round_count),
            CborValue::Bool(self.work_queue_empty),
            CborValue::Bool(self.zero_delta_full_round),
            CborValue::Bool(self.all_eligible_tuples_covered),
            cbor_uint(self.final_new_program_delta),
            cbor_uint(self.final_class_delta),
            cbor_uint(self.final_frontier_delta),
            cbor_uint(self.final_bank_delta),
        ])
    }

    fn pass_guards(&self) -> bool {
        self.work_queue_empty
            && self.zero_delta_full_round
            && self.all_eligible_tuples_covered
            && self.final_new_program_delta == 0
            && self.final_class_delta == 0
            && self.final_frontier_delta == 0
            && self.final_bank_delta == 0
    }
}

fn ensure_quotient_capacity(
    class_count: usize,
    total_frontier_points: usize,
    candidate_frontier_points: usize,
    total_bank_points: usize,
    candidate_bank_points: usize,
) -> Result<(), OracleError> {
    if class_count > MAX_BEHAVIOR_CLASSES {
        return Err(OracleError::resource(
            ResourceGuardId::BehaviorClasses,
            "behavior-class guard would be exceeded",
        ));
    }
    if total_frontier_points > MAX_FRONTIER_POINTS {
        return Err(OracleError::resource(
            ResourceGuardId::TotalFrontierPoints,
            "total-frontier guard would be exceeded",
        ));
    }
    if candidate_frontier_points > MAX_FRONTIER_POINTS_PER_CLASS {
        return Err(OracleError::resource(
            ResourceGuardId::FrontierPointsPerClass,
            "per-class frontier guard would be exceeded",
        ));
    }
    if total_bank_points > MAX_FRONTIER_POINTS {
        return Err(OracleError::resource(
            ResourceGuardId::TotalContinuationBankPoints,
            "total continuation-bank guard would be exceeded",
        ));
    }
    if candidate_bank_points > MAX_FRONTIER_POINTS_PER_CLASS {
        return Err(OracleError::resource(
            ResourceGuardId::ContinuationBankPointsPerClass,
            "per-class continuation-bank guard would be exceeded",
        ));
    }
    Ok(())
}

fn syntax_quotient_state(
    programs: &BTreeMap<Vec<u8>, Program>,
) -> Result<QuotientState, OracleError> {
    let mut grouped: BTreeMap<Vec<u8>, Vec<Program>> = BTreeMap::new();
    for program in programs.values() {
        grouped
            .entry(program.behavior.canonical_bytes()?)
            .or_default()
            .push(program.clone());
    }
    let mut state = QuotientState::default();
    for (behavior_bytes, group) in grouped {
        let behavior = group[0].behavior.clone();
        let behavior_id = behavior.behavior_id()?;
        if let Some(prior) = state.classes.get(&behavior_id) {
            if prior.behavior_bytes != behavior_bytes {
                return Err(OracleError::new(
                    FAIL_SHA256_PREIMAGE_COLLISION,
                    "distinct behavior preimages share one SHA-256 ID",
                ));
            }
        }
        let cohort_bank = cohort_bank_from_programs(&group);
        let frontier = frontier_from_cohort_bank(&cohort_bank);
        let minimum_mdl_q32 = frontier
            .iter()
            .map(|program| program.signature.mdl_length_q32)
            .min()
            .expect("behavior frontier is nonempty");
        let bank_points = cohort_bank.values().map(Vec::len).sum::<usize>();
        ensure_quotient_capacity(
            state.class_count() + 1,
            state.frontier_count() + frontier.len(),
            frontier.len(),
            state.continuation_bank_count() + bank_points,
            bank_points,
        )?;
        state.classes.insert(
            behavior_id,
            QuotientClass {
                behavior_id,
                behavior_bytes,
                behavior,
                frontier,
                cohort_bank,
                minimum_mdl_q32,
            },
        );
    }
    Ok(state)
}

#[derive(Debug, Clone, Copy, Default)]
struct StateDelta {
    class_delta: u64,
    frontier_delta: u64,
    bank_delta: u64,
}

fn insert_direct_program(
    state: &mut QuotientState,
    program: Program,
) -> Result<StateDelta, OracleError> {
    let behavior_id = program.behavior.behavior_id()?;
    let behavior_bytes = program.behavior.canonical_bytes()?;
    if let Some(prior) = state.classes.get(&behavior_id).cloned() {
        if prior.behavior_bytes != behavior_bytes {
            return Err(OracleError::new(
                FAIL_SHA256_PREIMAGE_COLLISION,
                "distinct behavior preimages share one SHA-256 ID",
            ));
        }
        let prior_bank_points = prior.cohort_bank.values().map(Vec::len).sum::<usize>();
        let mut candidate = prior.clone();
        let bank_changed = add_to_cohort_bank(&mut candidate.cohort_bank, program);
        let new_frontier = frontier_from_cohort_bank(&candidate.cohort_bank);
        let changed = ranked_frontier(&prior.frontier)
            .into_iter()
            .map(|(entry, rank)| encode_cbor(&entry.frontier_object(rank)))
            .collect::<Vec<_>>()
            != ranked_frontier(&new_frontier)
                .into_iter()
                .map(|(entry, rank)| encode_cbor(&entry.frontier_object(rank)))
                .collect::<Vec<_>>();
        candidate.frontier = new_frontier;
        candidate.minimum_mdl_q32 = candidate
            .frontier
            .iter()
            .map(|entry| entry.signature.mdl_length_q32)
            .min()
            .expect("behavior frontier is nonempty");
        let candidate_bank_points = candidate
            .cohort_bank
            .values()
            .map(Vec::len)
            .sum::<usize>();
        ensure_quotient_capacity(
            state.class_count(),
            state.frontier_count() - prior.frontier.len() + candidate.frontier.len(),
            candidate.frontier.len(),
            state.continuation_bank_count() - prior_bank_points + candidate_bank_points,
            candidate_bank_points,
        )?;
        state.classes.insert(behavior_id, candidate);
        return Ok(StateDelta {
            class_delta: 0,
            frontier_delta: u64::from(changed),
            bank_delta: u64::from(bank_changed),
        });
    }
    let minimum_mdl_q32 = program.signature.mdl_length_q32;
    let cohort_bank = cohort_bank_from_programs(std::slice::from_ref(&program));
    ensure_quotient_capacity(
        state.class_count() + 1,
        state.frontier_count() + 1,
        1,
        state.continuation_bank_count() + 1,
        1,
    )?;
    state.classes.insert(
        behavior_id,
        QuotientClass {
            behavior_id,
            behavior_bytes,
            behavior: program.behavior.clone(),
            frontier: vec![program],
            cohort_bank,
            minimum_mdl_q32,
        },
    );
    Ok(StateDelta {
        class_delta: 1,
        frontier_delta: 1,
        bank_delta: 1,
    })
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct RoundDelta {
    pub round_index: u64,
    pub queued_application_count: u64,
    pub new_canonical_program_count: u64,
    pub new_behavior_class_count: u64,
    pub frontier_mutation_count: u64,
    pub cohort_bank_mutation_count: u64,
    pub complete_state_changed: bool,
}

#[derive(Debug)]
struct DirectQuotientResult {
    state: QuotientState,
    programs: BTreeMap<Vec<u8>, Program>,
    coverage: BTreeMap<u16, CoverageRow>,
    raw_applications: u64,
    saturation_rounds: u64,
    work_queue_empty: bool,
    zero_delta_full_round: bool,
    all_eligible_tuples_covered: bool,
    rounds: Vec<RoundDelta>,
    final_program_delta: u64,
    final_class_delta: u64,
    final_frontier_delta: u64,
    final_bank_delta: u64,
}

fn direct_quotient_saturation() -> Result<DirectQuotientResult, OracleError> {
    let mut state = QuotientState::default();
    let mut coverage = empty_coverage_registry();
    let mut seen_programs = BTreeMap::new();
    for (leaf_code, leaf) in frozen_leaf_nodes() {
        let row = coverage_row_mut(&mut coverage, leaf_code);
        row.eligible_raw += 1;
        let program = micro_admit(leaf.clone())?.ok_or_else(|| {
            OracleError::new(
                FAIL_SEMANTICS_MISMATCH,
                format!("frozen leaf {leaf_code:#06x} failed direct admission"),
            )
        })?;
        row.strict_admitted += 1;
        if program.canonical.canonical_node != leaf {
            row.rewrite_collapses += 1;
        }
        if seen_programs
            .insert(program.canonical.canonical_cbor.clone(), program.clone())
            .is_none()
        {
            row.new_canonical += 1;
        } else {
            row.canonical_duplicates += 1;
        }
        insert_direct_program(&mut state, program)?;
    }

    let mut covered = BTreeSet::new();
    let mut raw_applications = FROZEN_LEAF_COUNT as u64;
    let mut saturation_rounds = 0_u64;
    let mut rounds = Vec::new();
    loop {
        saturation_rounds += 1;
        if saturation_rounds > MAX_SATURATION_ROUNDS {
            return Err(OracleError::resource(
                ResourceGuardId::SaturationRounds,
                "direct quotient exceeded maximum saturation rounds",
            ));
        }
        let snapshot = state.continuation_programs();
        let mut round_delta = StateDelta::default();
        let applications = eligible_applications(&snapshot);
        let mut queued_application_count = 0_u64;
        let mut new_canonical_program_count = 0_u64;
        for application in applications {
            if !covered.insert(application.key.clone()) {
                continue;
            }
            queued_application_count += 1;
            if raw_applications >= MAX_RAW_OPERATOR_APPLICATIONS {
                return Err(OracleError::resource(
                    ResourceGuardId::RawOperatorApplications,
                    "direct quotient raw-application guard would be exceeded",
                ));
            }
            raw_applications += 1;
            let raw_node = application.node();
            let row = coverage_row_mut(&mut coverage, application.operator.code());
            row.eligible_raw += 1;
            let Some(program) = micro_admit(raw_node.clone())? else {
                continue;
            };
            row.strict_admitted += 1;
            if program.canonical.canonical_node != raw_node {
                row.rewrite_collapses += 1;
            }
            if seen_programs
                .insert(program.canonical.canonical_cbor.clone(), program.clone())
                .is_none()
            {
                row.new_canonical += 1;
                new_canonical_program_count += 1;
            } else {
                row.canonical_duplicates += 1;
            }
            let delta = insert_direct_program(&mut state, program)?;
            round_delta.class_delta += delta.class_delta;
            round_delta.frontier_delta += delta.frontier_delta;
            round_delta.bank_delta += delta.bank_delta;
        }
        let complete_state_changed = new_canonical_program_count != 0
            || round_delta.class_delta != 0
            || round_delta.frontier_delta != 0
            || round_delta.bank_delta != 0;
        rounds.push(RoundDelta {
            round_index: saturation_rounds,
            queued_application_count,
            new_canonical_program_count,
            new_behavior_class_count: round_delta.class_delta,
            frontier_mutation_count: round_delta.frontier_delta,
            cohort_bank_mutation_count: round_delta.bank_delta,
            complete_state_changed,
        });
        if !complete_state_changed {
            break;
        }
    }
    let final_snapshot = state.continuation_programs();
    let all_eligible_tuples_covered = eligible_applications(&final_snapshot)
        .iter()
        .all(|application| covered.contains(&application.key));
    let work_queue_empty = all_eligible_tuples_covered;
    let final_round = *rounds.last().ok_or_else(|| {
        OracleError::new(FAIL_SEMANTICS_MISMATCH, "direct saturation has no rounds")
    })?;
    let zero_delta_full_round = !final_round.complete_state_changed
        && final_round.queued_application_count == 0
        && final_round.new_canonical_program_count == 0
        && final_round.new_behavior_class_count == 0
        && final_round.frontier_mutation_count == 0
        && final_round.cohort_bank_mutation_count == 0;
    if !work_queue_empty || !zero_delta_full_round {
        return Err(OracleError::new(
            FAIL_SEMANTICS_MISMATCH,
            "direct quotient stopped without complete frontier-tuple closure",
        ));
    }
    Ok(DirectQuotientResult {
        state,
        programs: seen_programs,
        coverage,
        raw_applications,
        saturation_rounds,
        work_queue_empty,
        zero_delta_full_round,
        all_eligible_tuples_covered,
        rounds,
        final_program_delta: final_round.new_canonical_program_count,
        final_class_delta: final_round.new_behavior_class_count,
        final_frontier_delta: final_round.frontier_mutation_count,
        final_bank_delta: final_round.cohort_bank_mutation_count,
    })
}

fn ordered_programs(programs: &BTreeMap<Vec<u8>, Program>) -> Vec<&Program> {
    let mut ordered = programs.values().collect::<Vec<_>>();
    ordered.sort_by(|left, right| {
        left.canonical
            .depth
            .cmp(&right.canonical.depth)
            .then_with(|| left.canonical.node_count.cmp(&right.canonical.node_count))
            .then_with(|| {
                (left.signature.output_sort as u64).cmp(&(right.signature.output_sort as u64))
            })
            .then_with(|| {
                left.canonical
                    .root_operator_id
                    .cmp(&right.canonical.root_operator_id)
            })
            .then_with(|| {
                left.canonical
                    .canonical_cbor
                    .cmp(&right.canonical.canonical_cbor)
            })
    });
    ordered
}

fn program_record(program: &Program, index: u64) -> CborValue {
    CborValue::Array(vec![
        cbor_uint(1),
        CborValue::Bytes(PROGRAM_RECORD_SCHEMA_ID.to_vec()),
        cbor_uint(index),
        CborValue::Bytes(program.canonical.canonical_cbor.clone()),
        CborValue::Bytes(program.canonical.canonical_ast_hash.to_vec()),
        cbor_uint(program.signature.output_sort as u64),
        cbor_uint(program.signature.mdl_length_q32),
    ])
}

fn program_records(programs: &BTreeMap<Vec<u8>, Program>) -> Vec<CborValue> {
    ordered_programs(programs)
        .into_iter()
        .enumerate()
        .map(|(index, program)| program_record(program, index as u64))
        .collect()
}

fn syntax_program_root(programs: &BTreeMap<Vec<u8>, Program>) -> [u8; 32] {
    rfc6962_root(
        &program_records(programs)
            .iter()
            .map(encode_cbor)
            .collect::<Vec<_>>(),
    )
}

#[derive(Debug, Clone, Serialize)]
pub struct OracleEndpoint {
    pub schema_version: &'static str,
    pub implementation_id: &'static str,
    pub terminal_status: &'static str,
    pub dsl_version: &'static str,
    pub dsl_freeze_version: &'static str,
    pub closure_semantics_version: &'static str,
    pub q0_freeze_version: &'static str,
    pub projection_id: &'static str,
    pub probe_input_signature_id: u64,
    pub probe_canonical_cbor_hex: String,
    pub probe_universe_root: String,
    pub frozen_leaf_count: usize,
    pub canonical_syntax_count: usize,
    pub syntax_raw_operator_applications: u64,
    pub quotient_raw_operator_applications: u64,
    pub syntax_strict_admitted_applications: u64,
    pub quotient_strict_admitted_applications: u64,
    pub syntax_rewrite_collapses: u64,
    pub quotient_rewrite_collapses: u64,
    pub behavior_class_count: usize,
    pub frontier_point_count: usize,
    pub maximum_frontier_size: usize,
    pub syntax_continuation_bank_point_count: usize,
    pub quotient_continuation_bank_point_count: usize,
    pub maximum_syntax_bank_points_per_class: usize,
    pub maximum_quotient_bank_points_per_class: usize,
    pub syntax_saturation_rounds: u64,
    pub direct_saturation_rounds: u64,
    pub work_queue_empty: bool,
    pub zero_delta_full_round: bool,
    pub all_typed_operator_frontier_tuples_covered: bool,
    pub exhaustive_syntax_oracle_complete: bool,
    pub syntax_direct_states_equal: bool,
    pub final_class_delta: u64,
    pub final_frontier_delta: u64,
    pub final_bank_delta: u64,
    pub projection_manifest_root: String,
    pub semantic_binding_root: String,
    pub syntax_program_root: String,
    pub syntax_class_archive_root: String,
    pub direct_class_archive_root: String,
    pub syntax_state_root: String,
    pub direct_state_root: String,
    /// Diagnostic-only complete five-tuple preimage.  This field is excluded
    /// from the frozen 43-field endpoint state and therefore cannot change its
    /// formal root.
    pub syntax_saturation_state_preimage_cbor_hex: String,
    /// Diagnostic-only complete five-tuple preimage for the direct path.
    pub direct_saturation_state_preimage_cbor_hex: String,
    pub syntax_coverage_root: String,
    pub direct_coverage_root: String,
    pub syntax_coverage: Vec<CoverageRow>,
    pub direct_coverage: Vec<CoverageRow>,
    pub direct_rounds: Vec<RoundDelta>,
    pub rust_source_root: String,
    pub endpoint_state_root: String,
    pub resource_guards_ok: bool,
    pub target_truth_accessed: bool,
    pub split_accessed: bool,
    pub role_evaluation_performed: bool,
    pub formal_roots_generated: bool,
    pub authority_claimed: bool,
}

impl OracleEndpoint {
    pub fn canonical_json(&self) -> Result<String, OracleError> {
        let value = serde_json::to_value(self).map_err(|error| {
            OracleError::new(
                FAIL_SEMANTICS_MISMATCH,
                format!("deterministic JSON encoding failed: {error}"),
            )
        })?;
        let object = match value {
            serde_json::Value::Object(object) => object,
            _ => {
                return Err(OracleError::new(
                    FAIL_SEMANTICS_MISMATCH,
                    "endpoint JSON must be an object",
                ))
            }
        };
        let ordered = object.into_iter().collect::<BTreeMap<_, _>>();
        serde_json::to_string(&ordered).map_err(|error| {
            OracleError::new(
                FAIL_SEMANTICS_MISMATCH,
                format!("canonical JSON encoding failed: {error}"),
            )
        })
    }

    pub fn endpoint_root(&self) -> Result<[u8; 32], OracleError> {
        Ok(content_hash(
            ENDPOINT_STATE_ROOT_DOMAIN,
            &self.canonical_state_object()?,
        ))
    }

    fn canonical_state_object(&self) -> Result<CborValue, OracleError> {
        Ok(CborValue::Array(vec![
            cbor_uint(1),
            CborValue::Bytes(ENDPOINT_STATE_SCHEMA_ID.to_vec()),
            CborValue::Bytes(Q0_FREEZE_VERSION.as_bytes().to_vec()),
            CborValue::Bytes(DSL_VERSION.as_bytes().to_vec()),
            CborValue::Bytes(CLOSURE_SEMANTICS_VERSION.as_bytes().to_vec()),
            CborValue::Bytes(PROJECTION_ID.as_bytes().to_vec()),
            CborValue::Bytes(parse_root_id(&self.probe_universe_root)?.to_vec()),
            CborValue::Bytes(parse_root_id(&self.projection_manifest_root)?.to_vec()),
            CborValue::Bytes(parse_root_id(&self.semantic_binding_root)?.to_vec()),
            CborValue::Bytes(self.terminal_status.as_bytes().to_vec()),
            cbor_uint(self.syntax_raw_operator_applications),
            cbor_uint(self.quotient_raw_operator_applications),
            cbor_uint(self.syntax_strict_admitted_applications),
            cbor_uint(self.quotient_strict_admitted_applications),
            cbor_uint(self.syntax_rewrite_collapses),
            cbor_uint(self.quotient_rewrite_collapses),
            cbor_uint(self.canonical_syntax_count as u64),
            cbor_uint(self.behavior_class_count as u64),
            cbor_uint(self.frontier_point_count as u64),
            cbor_uint(self.maximum_frontier_size as u64),
            cbor_uint(self.syntax_continuation_bank_point_count as u64),
            cbor_uint(self.quotient_continuation_bank_point_count as u64),
            cbor_uint(self.maximum_syntax_bank_points_per_class as u64),
            cbor_uint(self.maximum_quotient_bank_points_per_class as u64),
            cbor_uint(self.direct_saturation_rounds),
            CborValue::Bool(self.work_queue_empty),
            CborValue::Bool(self.zero_delta_full_round),
            cbor_uint(self.final_class_delta),
            cbor_uint(self.final_frontier_delta),
            cbor_uint(self.final_bank_delta),
            CborValue::Bytes(parse_root_id(&self.syntax_program_root)?.to_vec()),
            CborValue::Bytes(parse_root_id(&self.syntax_class_archive_root)?.to_vec()),
            CborValue::Bytes(parse_root_id(&self.direct_class_archive_root)?.to_vec()),
            CborValue::Bytes(parse_root_id(&self.syntax_coverage_root)?.to_vec()),
            CborValue::Bytes(parse_root_id(&self.direct_coverage_root)?.to_vec()),
            CborValue::Bytes(parse_root_id(&self.syntax_state_root)?.to_vec()),
            CborValue::Bytes(parse_root_id(&self.direct_state_root)?.to_vec()),
            CborValue::Bool(self.resource_guards_ok),
            CborValue::Bool(self.target_truth_accessed),
            CborValue::Bool(self.split_accessed),
            CborValue::Bool(self.role_evaluation_performed),
            CborValue::Bool(self.formal_roots_generated),
            CborValue::Bool(self.authority_claimed),
        ]))
    }
}

fn parse_root_id(value: &str) -> Result<[u8; 32], OracleError> {
    let hex = value.strip_prefix("sha256:").ok_or_else(|| {
        OracleError::new(FAIL_SEMANTICS_MISMATCH, "root ID lacks sha256 prefix")
    })?;
    parse_hex32(hex)
}

fn parse_hex32(hex: &str) -> Result<[u8; 32], OracleError> {
    if hex.len() != 64 {
        return Err(OracleError::new(
            FAIL_SEMANTICS_MISMATCH,
            "root ID does not contain 32 bytes",
        ));
    }
    let mut output = [0_u8; 32];
    for (index, slot) in output.iter_mut().enumerate() {
        let offset = index * 2;
        *slot = u8::from_str_radix(&hex[offset..offset + 2], 16).map_err(|_| {
            OracleError::new(FAIL_SEMANTICS_MISMATCH, "root ID contains invalid hex")
        })?;
    }
    Ok(output)
}

fn root_id(root: [u8; 32]) -> String {
    format!("sha256:{}", hex_encode(&root))
}

fn current_resident_memory_bytes() -> Result<u64, OracleError> {
    let status = std::fs::read_to_string("/proc/self/status").map_err(|error| {
        OracleError::resource(
            ResourceGuardId::ResidentMemory,
            format!("cannot read current resident memory: {error}"),
        )
    })?;
    let line = status
        .lines()
        .find(|line| line.starts_with("VmRSS:"))
        .ok_or_else(|| {
            OracleError::resource(
                ResourceGuardId::ResidentMemory,
                "/proc/self/status has no VmRSS field",
            )
        })?;
    let kibibytes = line
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| {
            OracleError::resource(
                ResourceGuardId::ResidentMemory,
                "VmRSS has no numeric value",
            )
        })?
        .parse::<u64>()
        .map_err(|_| {
            OracleError::resource(
                ResourceGuardId::ResidentMemory,
                "VmRSS is not an unsigned integer",
            )
        })?;
    kibibytes
        .checked_mul(1024)
        .ok_or_else(|| {
            OracleError::resource(
                ResourceGuardId::ResidentMemory,
                "VmRSS byte count overflow",
            )
        })
}

pub fn run_micro_oracle() -> Result<OracleEndpoint, OracleError> {
    let started = std::time::Instant::now();
    let probe_bytes = probe_canonical_bytes();
    let probe_root = probe_universe_root();
    if hex_encode(&probe_bytes) != EXPECTED_PROBE_CANONICAL_CBOR_HEX
        || hex_encode(&probe_root) != EXPECTED_PROBE_UNIVERSE_ROOT_HEX
    {
        return Err(OracleError::new(
            FAIL_SEMANTICS_MISMATCH,
            "probe input CBOR/root differs from the frozen cross-language vector",
        ));
    }
    let projection_root = projection_manifest_root();
    let semantic_root = semantic_binding_root()?;
    if hex_encode(&projection_root) != EXPECTED_PROJECTION_MANIFEST_ROOT_HEX
        || hex_encode(&semantic_root) != EXPECTED_SEMANTIC_BINDING_ROOT_HEX
    {
        return Err(OracleError::new(
            FAIL_SEMANTICS_MISMATCH,
            "independently reconstructed projection/semantic root differs from the frozen golden vector",
        ));
    }

    let syntax = exhaustive_syntax_oracle()?;
    let syntax_state = syntax_quotient_state(&syntax.programs)?;
    let direct = direct_quotient_saturation()?;

    let syntax_class_root = syntax_state.class_archive_root()?;
    let direct_class_root = direct.state.class_archive_root()?;
    let syntax_metadata = FixedPointMetadata {
        path_id: SYNTAX_PATH_ID,
        saturation_round_count: syntax.saturation_rounds,
        work_queue_empty: syntax.zero_delta_full_round,
        zero_delta_full_round: syntax.zero_delta_full_round,
        all_eligible_tuples_covered: syntax.zero_delta_full_round,
        final_new_program_delta: 0,
        final_class_delta: 0,
        final_frontier_delta: 0,
        final_bank_delta: 0,
    };
    let direct_metadata = FixedPointMetadata {
        path_id: DIRECT_PATH_ID,
        saturation_round_count: direct.saturation_rounds,
        work_queue_empty: direct.work_queue_empty,
        zero_delta_full_round: direct.zero_delta_full_round,
        all_eligible_tuples_covered: direct.all_eligible_tuples_covered,
        final_new_program_delta: direct.final_program_delta,
        final_class_delta: direct.final_class_delta,
        final_frontier_delta: direct.final_frontier_delta,
        final_bank_delta: direct.final_bank_delta,
    };
    if !syntax_metadata.pass_guards() || !direct_metadata.pass_guards() {
        return Err(OracleError::new(
            FAIL_SEMANTICS_MISMATCH,
            "terminal fixed-point metadata does not satisfy every PASS guard",
        ));
    }
    let syntax_state_root = syntax_state.saturation_state_root(
        &syntax.programs,
        &syntax.coverage,
        &syntax_metadata,
        SYNTAX_STATE_ROOT_DOMAIN,
    )?;
    let direct_state_root = direct.state.saturation_state_root(
        &direct.programs,
        &direct.coverage,
        &direct_metadata,
        DIRECT_STATE_ROOT_DOMAIN,
    )?;
    let syntax_state_preimage = encode_cbor(&syntax_state.saturation_state_object(
        &syntax.programs,
        &syntax.coverage,
        &syntax_metadata,
    )?);
    let direct_state_preimage = encode_cbor(&direct.state.saturation_state_object(
        &direct.programs,
        &direct.coverage,
        &direct_metadata,
    )?);
    let states_equal = syntax_state.class_records()? == direct.state.class_records()?
        && syntax_class_root == direct_class_root
        && syntax_state.continuation_bank_object() == direct.state.continuation_bank_object();
    if !states_equal {
        let mut missing = Vec::new();
        for (behavior_id, syntax_class) in &syntax_state.classes {
            let direct_entries = direct
                .state
                .classes
                .get(behavior_id)
                .map(|class| {
                    ranked_frontier(&class.frontier)
                        .into_iter()
                        .map(|(entry, rank)| encode_cbor(&entry.frontier_object(rank)))
                        .collect::<BTreeSet<_>>()
                })
                .unwrap_or_default();
            for (entry, rank) in ranked_frontier(&syntax_class.frontier) {
                if !direct_entries.contains(&encode_cbor(&entry.frontier_object(rank))) {
                    missing.push(format!(
                        "behavior={} ast={} signature={}",
                        hex_encode(behavior_id),
                        hex_encode(&entry.canonical.canonical_cbor),
                        hex_encode(&entry.signature.canonical_bytes()),
                    ));
                }
            }
        }
        return Err(OracleError::new(
            FAIL_SEMANTICS_MISMATCH,
            format!(
                "exhaustive syntax quotient differs from direct quotient saturation: syntax classes/frontiers={}/{}, direct={}/{}, syntax class/state roots={}/{}, direct={}/{}, syntax-only={:?}",
                syntax_state.class_count(),
                syntax_state.frontier_count(),
                direct.state.class_count(),
                direct.state.frontier_count(),
                root_id(syntax_class_root),
                root_id(syntax_state_root),
                root_id(direct_class_root),
                root_id(direct_state_root),
                missing,
            ),
        ));
    }
    if syntax.coverage.len() != 27 || direct.coverage.len() != 27 {
        return Err(OracleError::new(
            FAIL_SEMANTICS_MISMATCH,
            "coverage registry does not contain exactly 27 frozen operator codes",
        ));
    }
    if syntax.raw_applications > MAX_RAW_OPERATOR_APPLICATIONS
        || direct.raw_applications > MAX_RAW_OPERATOR_APPLICATIONS
    {
        return Err(OracleError::resource(
            ResourceGuardId::RawOperatorApplications,
            "raw-application postcondition exceeded",
        ));
    }
    if syntax.programs.len() > MAX_CANONICAL_SYNTAX
        || direct.programs.len() > MAX_CANONICAL_SYNTAX
    {
        return Err(OracleError::resource(
            ResourceGuardId::CanonicalSyntaxPrograms,
            "canonical-program postcondition exceeded",
        ));
    }
    if syntax_state.class_count() > MAX_BEHAVIOR_CLASSES
        || direct.state.class_count() > MAX_BEHAVIOR_CLASSES
    {
        return Err(OracleError::resource(
            ResourceGuardId::BehaviorClasses,
            "behavior-class postcondition exceeded",
        ));
    }
    if syntax_state.frontier_count() > MAX_FRONTIER_POINTS
        || direct.state.frontier_count() > MAX_FRONTIER_POINTS
    {
        return Err(OracleError::resource(
            ResourceGuardId::TotalFrontierPoints,
            "total-frontier postcondition exceeded",
        ));
    }
    if syntax_state.maximum_frontier_size() > MAX_FRONTIER_POINTS_PER_CLASS
        || direct.state.maximum_frontier_size() > MAX_FRONTIER_POINTS_PER_CLASS
    {
        return Err(OracleError::resource(
            ResourceGuardId::FrontierPointsPerClass,
            "per-class frontier postcondition exceeded",
        ));
    }
    if syntax_state.continuation_bank_count() > MAX_FRONTIER_POINTS
        || direct.state.continuation_bank_count() > MAX_FRONTIER_POINTS
    {
        return Err(OracleError::resource(
            ResourceGuardId::TotalContinuationBankPoints,
            "total continuation-bank postcondition exceeded",
        ));
    }
    if syntax_state.maximum_bank_size() > MAX_FRONTIER_POINTS_PER_CLASS
        || direct.state.maximum_bank_size() > MAX_FRONTIER_POINTS_PER_CLASS
    {
        return Err(OracleError::resource(
            ResourceGuardId::ContinuationBankPointsPerClass,
            "per-class continuation-bank postcondition exceeded",
        ));
    }
    if syntax.saturation_rounds > MAX_SATURATION_ROUNDS
        || direct.saturation_rounds > MAX_SATURATION_ROUNDS
    {
        return Err(OracleError::resource(
            ResourceGuardId::SaturationRounds,
            "saturation-round postcondition exceeded",
        ));
    }
    let resource_guards_ok = true;
    let syntax_coverage_root = coverage_root(&syntax.coverage);
    let direct_coverage_root = coverage_root(&direct.coverage);
    let syntax_strict_admitted = syntax
        .coverage
        .values()
        .map(|row| row.strict_admitted)
        .sum();
    let direct_strict_admitted = direct
        .coverage
        .values()
        .map(|row| row.strict_admitted)
        .sum();
    let syntax_rewrite_collapses = syntax
        .coverage
        .values()
        .map(|row| row.rewrite_collapses)
        .sum();
    let direct_rewrite_collapses = direct
        .coverage
        .values()
        .map(|row| row.rewrite_collapses)
        .sum();
    let rust_source_root = sha256(&[include_bytes!("lib.rs")]);
    let mut endpoint = OracleEndpoint {
        schema_version: SCHEMA_VERSION,
        implementation_id: IMPLEMENTATION_ID,
        terminal_status: SINGLE_IMPLEMENTATION_PASS_STATUS,
        dsl_version: DSL_VERSION,
        dsl_freeze_version: DSL_FREEZE_VERSION,
        closure_semantics_version: CLOSURE_SEMANTICS_VERSION,
        q0_freeze_version: Q0_FREEZE_VERSION,
        projection_id: PROJECTION_ID,
        probe_input_signature_id: PROBE_INPUT_SIGNATURE_ID,
        probe_canonical_cbor_hex: hex_encode(&probe_bytes),
        probe_universe_root: root_id(probe_root),
        frozen_leaf_count: FROZEN_LEAF_COUNT,
        canonical_syntax_count: syntax.programs.len(),
        syntax_raw_operator_applications: syntax.raw_applications,
        quotient_raw_operator_applications: direct.raw_applications,
        syntax_strict_admitted_applications: syntax_strict_admitted,
        quotient_strict_admitted_applications: direct_strict_admitted,
        syntax_rewrite_collapses,
        quotient_rewrite_collapses: direct_rewrite_collapses,
        behavior_class_count: syntax_state.class_count(),
        frontier_point_count: syntax_state.frontier_count(),
        maximum_frontier_size: syntax_state.maximum_frontier_size(),
        syntax_continuation_bank_point_count: syntax_state.continuation_bank_count(),
        quotient_continuation_bank_point_count: direct.state.continuation_bank_count(),
        maximum_syntax_bank_points_per_class: syntax_state.maximum_bank_size(),
        maximum_quotient_bank_points_per_class: direct.state.maximum_bank_size(),
        syntax_saturation_rounds: syntax.saturation_rounds,
        direct_saturation_rounds: direct.saturation_rounds,
        work_queue_empty: direct.work_queue_empty,
        zero_delta_full_round: direct.zero_delta_full_round,
        all_typed_operator_frontier_tuples_covered: direct.all_eligible_tuples_covered,
        exhaustive_syntax_oracle_complete: syntax.zero_delta_full_round,
        syntax_direct_states_equal: states_equal,
        final_class_delta: direct.final_class_delta,
        final_frontier_delta: direct.final_frontier_delta,
        final_bank_delta: direct.final_bank_delta,
        projection_manifest_root: root_id(projection_root),
        semantic_binding_root: root_id(semantic_root),
        syntax_program_root: root_id(syntax_program_root(&syntax.programs)),
        syntax_class_archive_root: root_id(syntax_class_root),
        direct_class_archive_root: root_id(direct_class_root),
        syntax_state_root: root_id(syntax_state_root),
        direct_state_root: root_id(direct_state_root),
        syntax_saturation_state_preimage_cbor_hex: hex_encode(&syntax_state_preimage),
        direct_saturation_state_preimage_cbor_hex: hex_encode(&direct_state_preimage),
        syntax_coverage_root: root_id(syntax_coverage_root),
        direct_coverage_root: root_id(direct_coverage_root),
        syntax_coverage: syntax.coverage.into_values().collect(),
        direct_coverage: direct.coverage.into_values().collect(),
        direct_rounds: direct.rounds,
        rust_source_root: root_id(rust_source_root),
        endpoint_state_root: String::new(),
        resource_guards_ok,
        target_truth_accessed: false,
        split_accessed: false,
        role_evaluation_performed: false,
        formal_roots_generated: false,
        authority_claimed: false,
    };
    endpoint.endpoint_state_root = root_id(endpoint.endpoint_root()?);
    let canonical_output_bytes = encode_cbor(&endpoint.canonical_state_object()?).len() as u64;
    // `main` emits the canonical JSON with `println!`, so the output guard
    // covers the exact stdout byte count, including its single LF terminator.
    let diagnostic_output_bytes = (endpoint.canonical_json()?.len() as u64)
        .checked_add(1)
        .ok_or_else(|| {
            OracleError::resource(
                ResourceGuardId::OutputBytes,
                "endpoint diagnostic stdout byte count overflowed",
            )
        })?;
    if canonical_output_bytes > MAX_OUTPUT_BYTES || diagnostic_output_bytes > MAX_OUTPUT_BYTES {
        return Err(OracleError::resource(
            ResourceGuardId::OutputBytes,
            "endpoint canonical or diagnostic output-byte guard exceeded",
        ));
    }
    if started.elapsed().as_secs_f64() > MAX_WALL_TIME_SECONDS as f64 {
        return Err(OracleError::resource(
            ResourceGuardId::WallTime,
            "Q0 wall-time guard exceeded",
        ));
    }
    if current_resident_memory_bytes()? > MAX_MEMORY_BYTES {
        return Err(OracleError::resource(
            ResourceGuardId::ResidentMemory,
            "Q0 resident-memory guard exceeded",
        ));
    }
    Ok(endpoint)
}

#[cfg(test)]
mod internal_tests {
    use super::*;

    fn fake_id(index: usize) -> [u8; 32] {
        let mut value = [0_u8; 32];
        value[..8].copy_from_slice(&(index as u64).to_be_bytes());
        value[31] = 0xa5;
        value
    }

    fn scalar_program() -> Program {
        micro_admit(Node::ScalarConst(1))
            .expect("strict admission")
            .expect("Q0 leaf admission")
    }

    fn singleton_class(program: &Program) -> QuotientClass {
        let behavior_id = program.behavior.behavior_id().unwrap();
        QuotientClass {
            behavior_id,
            behavior_bytes: program.behavior.canonical_bytes().unwrap(),
            behavior: program.behavior.clone(),
            frontier: vec![program.clone()],
            cohort_bank: cohort_bank_from_programs(std::slice::from_ref(program)),
            minimum_mdl_q32: program.signature.mdl_length_q32,
        }
    }

    #[test]
    fn manifest_recursion_rejects_unregistered_leaf() {
        let error = micro_admit(Node::BitAt(2)).expect_err("bit2 is outside the Q0 manifest");
        assert_eq!(error.code, REJECT_Q0_PROJECTION);

        let error = micro_admit(Node::Aggregate {
            map_id: 0,
            scope_id: 2,
            quantity_id: 0,
            scope_extension: vec![],
        })
        .expect_err("boundary-scope aggregate is outside the Q0 leaf manifest");
        assert_eq!(error.code, REJECT_Q0_PROJECTION);
    }

    #[test]
    fn multiplicity_reservoir_preserves_the_distinct_child_counterexample() {
        let syntax = exhaustive_syntax_oracle().expect("syntax enumeration");
        let state = syntax_quotient_state(&syntax.programs).expect("syntax quotient");
        let counterexample = parse_hex32(
            "b2ce28e4828243bf35fe7e9b204da66807d32522d63c838eb715c8e14d1a7f27",
        )
        .expect("static behavior ID");
        let class = state
            .classes
            .get(&counterexample)
            .expect("counterexample behavior class");
        assert_eq!(class.frontier.len(), 4);
        assert!(class.cohort_bank.values().any(|cohort| cohort.len() == 2));

        let frontier_cbors = class
            .frontier
            .iter()
            .map(|program| hex_encode(&program.canonical.canonical_cbor))
            .collect::<BTreeSet<_>>();
        assert!(frontier_cbors.contains(
            "82018402038300000583010186000301030080"
        ));
        assert!(frontier_cbors.contains(
            "82018402038300000583010286000300030080"
        ));
        assert!(syntax.programs.values().any(|program| {
            hex_encode(&program.canonical.canonical_cbor)
                == "82018204828300040083000500"
        }));
    }

    #[test]
    fn direct_bank_expansion_reaches_the_global_syntax_frontier() {
        let syntax = exhaustive_syntax_oracle().expect("syntax enumeration");
        let syntax_state = syntax_quotient_state(&syntax.programs).expect("syntax quotient");
        let direct = direct_quotient_saturation().expect("direct quotient saturation");
        assert_eq!(syntax_state.class_records().unwrap(), direct.state.class_records().unwrap());
        assert_eq!(syntax_state.frontier_count(), 122);
        assert_eq!(syntax_state.continuation_bank_count(), 251);
        assert_eq!(direct.raw_applications, 545);
        assert_eq!(direct.rounds.last().unwrap().cohort_bank_mutation_count, 0);
        assert!(direct.rounds.last().unwrap().queued_application_count == 0);
    }

    #[test]
    fn resource_guard_preflights_are_atomic_and_carry_exact_ids() {
        let program = scalar_program();

        let mut syntax_programs = BTreeMap::new();
        for index in 0..MAX_CANONICAL_SYNTAX {
            syntax_programs.insert(index.to_be_bytes().to_vec(), program.clone());
        }
        let syntax_keys_before = syntax_programs.keys().cloned().collect::<Vec<_>>();
        let error = insert_syntax_program_atomically(&mut syntax_programs, program.clone())
            .expect_err("canonical syntax guard");
        assert_eq!(error.guard_id, Some(ResourceGuardId::CanonicalSyntaxPrograms as u64));
        assert_eq!(syntax_keys_before, syntax_programs.keys().cloned().collect::<Vec<_>>());

        let actual_id = program.behavior.behavior_id().unwrap();
        let template = singleton_class(&program);

        let mut class_limited = QuotientState::default();
        let mut index = 0_usize;
        while class_limited.class_count() < MAX_BEHAVIOR_CLASSES {
            let key = fake_id(index);
            index += 1;
            if key == actual_id {
                continue;
            }
            let mut class = template.clone();
            class.behavior_id = key;
            class_limited.classes.insert(key, class);
        }
        let class_keys_before = class_limited.classes.keys().copied().collect::<Vec<_>>();
        let error = insert_direct_program(&mut class_limited, program.clone())
            .expect_err("behavior class guard");
        assert_eq!(error.guard_id, Some(ResourceGuardId::BehaviorClasses as u64));
        assert_eq!(class_keys_before, class_limited.classes.keys().copied().collect::<Vec<_>>());

        let mut frontier_limited = QuotientState::default();
        let mut oversized = template.clone();
        oversized.behavior_id = fake_id(90_000);
        oversized.frontier = vec![program.clone(); MAX_FRONTIER_POINTS];
        frontier_limited
            .classes
            .insert(oversized.behavior_id, oversized);
        let frontier_before = frontier_limited.frontier_count();
        let error = insert_direct_program(&mut frontier_limited, program.clone())
            .expect_err("total frontier guard");
        assert_eq!(error.guard_id, Some(ResourceGuardId::TotalFrontierPoints as u64));
        assert_eq!(frontier_before, frontier_limited.frontier_count());
        assert!(!frontier_limited.classes.contains_key(&actual_id));

        let mut bank_limited = QuotientState::default();
        let mut bank_class = template.clone();
        bank_class.behavior_id = fake_id(90_001);
        bank_class.cohort_bank.clear();
        bank_class.cohort_bank.insert(
            b"synthetic-total-bank".to_vec(),
            vec![program.clone(); MAX_FRONTIER_POINTS],
        );
        bank_limited.classes.insert(bank_class.behavior_id, bank_class);
        let bank_before = bank_limited.continuation_bank_count();
        let error = insert_direct_program(&mut bank_limited, program.clone())
            .expect_err("total bank guard");
        assert_eq!(
            error.guard_id,
            Some(ResourceGuardId::TotalContinuationBankPoints as u64)
        );
        assert_eq!(bank_before, bank_limited.continuation_bank_count());
        assert!(!bank_limited.classes.contains_key(&actual_id));

        let mut per_class_bank = QuotientState::default();
        let mut bank_class = template.clone();
        bank_class.cohort_bank.clear();
        for index in 0..MAX_FRONTIER_POINTS_PER_CLASS {
            bank_class
                .cohort_bank
                .insert(format!("synthetic-{index:03}").into_bytes(), vec![program.clone()]);
        }
        per_class_bank.classes.insert(actual_id, bank_class);
        let mut new_signature = program.clone();
        new_signature.signature.distinct_bit_slot_bitmap = 0xff;
        let records_before = per_class_bank.class_records().unwrap();
        let bank_object_before = per_class_bank.continuation_bank_object();
        let error = insert_direct_program(&mut per_class_bank, new_signature)
            .expect_err("per-class bank guard");
        assert_eq!(
            error.guard_id,
            Some(ResourceGuardId::ContinuationBankPointsPerClass as u64)
        );
        assert_eq!(records_before, per_class_bank.class_records().unwrap());
        assert_eq!(bank_object_before, per_class_bank.continuation_bank_object());

        let error = ensure_quotient_capacity(1, 65, 65, 1, 1)
            .expect_err("per-class frontier guard");
        assert_eq!(
            error.guard_id,
            Some(ResourceGuardId::FrontierPointsPerClass as u64)
        );
    }
}
