"""Source-free set-marginal core for the frozen HiTab DMC-1 mechanism.

This module is deliberately smaller than a benchmark runtime.  It has no
filesystem, dataset, split, model-loader, network, baseline, family, score, or
online-evaluator entrypoint.  Its only pre-label input is an already computed
item-local view:

* question facets;
* a canonical ``U:0 .. U:n-1`` ordering of evidence units;
* source-native unit types and typed edges; and
* integer-quantized cross-encoder and MiniLM tensors.

The label boundary is explicit.  :func:`build_and_seal_aform_registry` first
constructs and hashes every A_form state/action/feature row.  Only a sealed
registry can then be paired with a DNF proof through
:func:`label_sealed_registry`.  Consequently proof requirements cannot change
the state search, action slate, feature vector, or pre-label archive hash.

E0 rewrites the complete five-unit set from the empty state using the frozen
V0 marginal.  E1 is a unique lambda-one, no-intercept ridge fitted to exact
``60 * delta utility`` targets after centering both features and targets
within each state.  The ten coefficients are deterministically quantized
before E1 performs a fresh five-step ``beta dot phi`` argmax.  E1 has no
threshold, fallback, baseline-retention, or recipe branch.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import hmac
import json
import math
import re
import unicodedata
from typing import Mapping, Sequence


STUDY_ID = "HITAB_P1_DMC1_HIERARCHICAL_SET_EVALUATOR_V1"
VERSION = "hitab_p1_dmc1_core_v1"
TOP_K = 5
MIN_UNITS = 10
QUANT_SCALE = 1_000_000
TARGET_SCALE = 60
RIDGE_LAMBDA = Fraction(1, 1)
COEFFICIENT_SCALE = 1_000_000_000
A_FORM_V0_STATE_CAP = 8
A_FORM_HMAC_STATE_CAP = 8
MAX_FACETS = 32
MAX_UNITS = 256
MAX_SIGN_FLIP_PAIRS = 256

TOP_V0 = "TOP_V0"
HMAC_EXPLORATION = "HMAC_EXPLORATION"
STATE_CLASSES = (TOP_V0, HMAC_EXPLORATION)
SOURCE_NATIVE_EDGE_TYPE = "FORWARD_SHARED_AXIS_OR_HEADER"

FEATURE_NAMES = (
    "candidate_ce_max",
    "candidate_ce_mean",
    "candidate_minilm_max",
    "candidate_minilm_mean",
    "ce_residual_facet_coverage_gain",
    "minilm_residual_facet_coverage_gain",
    "source_native_type_novelty",
    "typed_incoming_from_selected_count",
    "typed_outgoing_to_selected_count",
    "pairwise_minilm_nonredundancy_gain",
)

# V0 is intentionally fixed and label-free.  The two directed edge features
# have equal weights so the accumulated set value is independent of insertion
# order.
V0_WEIGHTS = (40, 10, 30, 8, 1, 1, 1, 1, 1, 1)

# Exact mapping validation makes these absences executable, rather than merely
# documentary.  Family is an external result grouping and baseline outputs are
# evaluated only after all DMC-1 actions have been sealed.
PRELABEL_VIEW_FIELDS = frozenset(
    {
        "corpus_commitment",
        "question_facets",
        "unit_keys",
        "unit_types",
        "typed_edges",
        "ce_facet_unit",
        "minilm_facet_unit",
        "minilm_unit_unit",
    }
)
FORBIDDEN_PRELABEL_FIELDS = frozenset(
    {
        "source",
        "source_id",
        "split",
        "split_id",
        "family",
        "family_id",
        "item",
        "item_id",
        "gold",
        "qrel",
        "proof",
        "raw",
        "raw_rank",
        "raw_output",
        "hippo",
        "hipporag",
        "hipporag_rank",
        "hipporag_output",
        "recipe",
        "recipe_id",
    }
)

_UNIT_KEY = re.compile(r"U:(0|[1-9][0-9]*)\Z")
_TYPE_TOKEN = re.compile(r"[A-Z][A-Z0-9_]{0,63}\Z")
_WHITESPACE = re.compile(r"\s+", flags=re.UNICODE)


class HitabDmc1CoreError(ValueError):
    """Fail-closed error for DMC-1 schema, seal, or exact-metric drift."""


def canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    """Encode strict, stable ASCII JSON suitable for hashes and HMACs."""

    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise HitabDmc1CoreError("value is not canonical JSON") from exc
    return payload + (b"\n" if newline else b"")


def stable_hash(value: object) -> str:
    """Return the canonical SHA-256 hex digest of ``value``."""

    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _canonical_text(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise HitabDmc1CoreError(f"{field} must be text")
    normalized = unicodedata.normalize("NFKC", value)
    normalized = _WHITESPACE.sub(" ", normalized).strip()
    if not normalized or "\x00" in normalized or len(normalized) > 4_000:
        raise HitabDmc1CoreError(f"{field} is empty or invalid")
    return normalized


def _canonical_type(value: object, *, field: str) -> str:
    normalized = _canonical_text(value, field=field).upper().replace("-", "_")
    if _TYPE_TOKEN.fullmatch(normalized) is None:
        raise HitabDmc1CoreError(f"{field} is not a canonical type token")
    return normalized


def _strict_quantized(value: object, *, field: str) -> int:
    if type(value) is not int or not -QUANT_SCALE <= value <= QUANT_SCALE:
        raise HitabDmc1CoreError(
            f"{field} must be an integer in [-{QUANT_SCALE},{QUANT_SCALE}]"
        )
    return int(value)


def _strict_ordinal(value: object, *, unit_count: int, field: str) -> int:
    if type(value) is not int or not 0 <= value < unit_count:
        raise HitabDmc1CoreError(f"{field} is not an in-view unit ordinal")
    return int(value)


def _strict_selected(
    value: Sequence[int], *, unit_count: int, allow_full: bool = True
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise HitabDmc1CoreError("selected state must be an ordinal array")
    selected = tuple(
        _strict_ordinal(row, unit_count=unit_count, field="selected")
        for row in value
    )
    maximum = TOP_K if allow_full else TOP_K - 1
    if len(selected) > maximum:
        raise HitabDmc1CoreError("selected state exceeds the frozen depth")
    if selected != tuple(sorted(set(selected))):
        raise HitabDmc1CoreError(
            "selected state must be unique and canonically sorted"
        )
    return selected


@dataclass(frozen=True, order=True)
class TypedEdge:
    """One directed source-native typed relation between local unit ordinals."""

    source_ordinal: int
    target_ordinal: int
    edge_type: str

    def payload(self) -> dict[str, object]:
        return {
            "edge_type": self.edge_type,
            "source_ordinal": self.source_ordinal,
            "target_ordinal": self.target_ordinal,
        }


def _validated_edge(value: object, *, unit_count: int) -> TypedEdge:
    if isinstance(value, TypedEdge):
        source = value.source_ordinal
        target = value.target_ordinal
        edge_type = value.edge_type
    elif isinstance(value, Mapping):
        if set(value) != {"source_ordinal", "target_ordinal", "edge_type"}:
            raise HitabDmc1CoreError("typed edge schema drifted")
        source = value["source_ordinal"]
        target = value["target_ordinal"]
        edge_type = value["edge_type"]
    else:
        raise HitabDmc1CoreError("typed edge is not an object")
    checked_source = _strict_ordinal(
        source, unit_count=unit_count, field="edge source"
    )
    checked_target = _strict_ordinal(
        target, unit_count=unit_count, field="edge target"
    )
    if checked_source == checked_target:
        raise HitabDmc1CoreError("typed self-edges are forbidden")
    checked_type = _canonical_type(edge_type, field="edge_type")
    if checked_type != SOURCE_NATIVE_EDGE_TYPE:
        raise HitabDmc1CoreError("typed edge type is outside contract")
    if checked_source >= checked_target:
        raise HitabDmc1CoreError(
            "typed edge must follow canonical corpus order"
        )
    return TypedEdge(
        checked_source,
        checked_target,
        checked_type,
    )


def _strict_matrix(
    value: object,
    *,
    rows: int,
    columns: int,
    field: str,
) -> tuple[tuple[int, ...], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise HitabDmc1CoreError(f"{field} must be a matrix")
    if len(value) != rows:
        raise HitabDmc1CoreError(f"{field} row count drifted")
    checked_rows: list[tuple[int, ...]] = []
    for row_index, raw_row in enumerate(value):
        if isinstance(raw_row, (str, bytes)) or not isinstance(
            raw_row, Sequence
        ):
            raise HitabDmc1CoreError(f"{field}[{row_index}] must be an array")
        if len(raw_row) != columns:
            raise HitabDmc1CoreError(f"{field}[{row_index}] width drifted")
        checked_rows.append(
            tuple(
                _strict_quantized(
                    coordinate,
                    field=f"{field}[{row_index}][{column_index}]",
                )
                for column_index, coordinate in enumerate(raw_row)
            )
        )
    return tuple(checked_rows)


@dataclass(frozen=True)
class PrecomputedView:
    """The exact source-free, label-free DMC-1 input contract."""

    corpus_commitment: str
    question_facets: tuple[str, ...]
    unit_keys: tuple[str, ...]
    unit_types: tuple[str, ...]
    typed_edges: tuple[TypedEdge, ...]
    ce_facet_unit: tuple[tuple[int, ...], ...]
    minilm_facet_unit: tuple[tuple[int, ...], ...]
    minilm_unit_unit: tuple[tuple[int, ...], ...]
    _edge_pairs: frozenset[tuple[int, int]] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.corpus_commitment, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.corpus_commitment) is None
        ):
            raise HitabDmc1CoreError("corpus commitment is invalid")
        facets = tuple(
            _canonical_text(row, field=f"question_facets[{index}]")
            for index, row in enumerate(self.question_facets)
        )
        if not 1 <= len(facets) <= MAX_FACETS:
            raise HitabDmc1CoreError("question facet count is outside contract")
        if len({row.casefold() for row in facets}) != len(facets):
            raise HitabDmc1CoreError("question facets contain duplicates")

        if not MIN_UNITS <= len(self.unit_keys) <= MAX_UNITS:
            raise HitabDmc1CoreError("unit count is outside contract")
        expected_keys = tuple(f"U:{index}" for index in range(len(self.unit_keys)))
        if tuple(self.unit_keys) != expected_keys:
            raise HitabDmc1CoreError(
                "unit keys must be the complete canonical local ordering"
            )
        unit_count = len(expected_keys)
        if len(self.unit_types) != unit_count:
            raise HitabDmc1CoreError("unit type count drifted")
        unit_types = tuple(
            _canonical_type(value, field=f"unit_types[{index}]")
            for index, value in enumerate(self.unit_types)
        )

        edges = tuple(
            _validated_edge(value, unit_count=unit_count)
            for value in self.typed_edges
        )
        if edges != tuple(sorted(set(edges))):
            raise HitabDmc1CoreError(
                "typed edges must be unique and canonically sorted"
            )

        ce = _strict_matrix(
            self.ce_facet_unit,
            rows=len(facets),
            columns=unit_count,
            field="ce_facet_unit",
        )
        facet_minilm = _strict_matrix(
            self.minilm_facet_unit,
            rows=len(facets),
            columns=unit_count,
            field="minilm_facet_unit",
        )
        unit_minilm = _strict_matrix(
            self.minilm_unit_unit,
            rows=unit_count,
            columns=unit_count,
            field="minilm_unit_unit",
        )
        for left in range(unit_count):
            if unit_minilm[left][left] != QUANT_SCALE:
                raise HitabDmc1CoreError(
                    "MiniLM unit tensor diagonal must equal QUANT_SCALE"
                )
            for right in range(left + 1, unit_count):
                if unit_minilm[left][right] != unit_minilm[right][left]:
                    raise HitabDmc1CoreError(
                        "MiniLM unit tensor must be symmetric"
                    )

        object.__setattr__(self, "question_facets", facets)
        object.__setattr__(self, "unit_keys", expected_keys)
        object.__setattr__(self, "unit_types", unit_types)
        object.__setattr__(self, "typed_edges", edges)
        object.__setattr__(self, "ce_facet_unit", ce)
        object.__setattr__(self, "minilm_facet_unit", facet_minilm)
        object.__setattr__(self, "minilm_unit_unit", unit_minilm)
        object.__setattr__(
            self,
            "_edge_pairs",
            frozenset(
                (edge.source_ordinal, edge.target_ordinal) for edge in edges
            ),
        )

    @property
    def unit_count(self) -> int:
        return len(self.unit_keys)

    def payload(self) -> dict[str, object]:
        return {
            "ce_facet_unit": [list(row) for row in self.ce_facet_unit],
            "corpus_commitment": self.corpus_commitment,
            "minilm_facet_unit": [
                list(row) for row in self.minilm_facet_unit
            ],
            "minilm_unit_unit": [list(row) for row in self.minilm_unit_unit],
            "question_facets": list(self.question_facets),
            "typed_edges": [edge.payload() for edge in self.typed_edges],
            "unit_keys": list(self.unit_keys),
            "unit_types": list(self.unit_types),
        }

    @property
    def sha256(self) -> str:
        return stable_hash(self.payload())


def view_from_mapping(value: object) -> PrecomputedView:
    """Validate the exact view schema, rejecting family/baseline/label extras."""

    if not isinstance(value, Mapping):
        raise HitabDmc1CoreError("precomputed view must be an object")
    supplied = set(value)
    if supplied != PRELABEL_VIEW_FIELDS:
        forbidden = sorted(
            str(field) for field in supplied & FORBIDDEN_PRELABEL_FIELDS
        )
        missing = sorted(PRELABEL_VIEW_FIELDS - supplied)
        extra = sorted(supplied - PRELABEL_VIEW_FIELDS)
        raise HitabDmc1CoreError(
            "precomputed view schema drifted; "
            f"missing={missing}, extra={extra}, forbidden={forbidden}"
        )
    try:
        return PrecomputedView(
            corpus_commitment=value["corpus_commitment"],  # type: ignore[arg-type]
            question_facets=tuple(value["question_facets"]),  # type: ignore[arg-type]
            unit_keys=tuple(value["unit_keys"]),  # type: ignore[arg-type]
            unit_types=tuple(value["unit_types"]),  # type: ignore[arg-type]
            typed_edges=tuple(value["typed_edges"]),  # type: ignore[arg-type]
            ce_facet_unit=tuple(  # type: ignore[arg-type]
                tuple(row) for row in value["ce_facet_unit"]  # type: ignore[union-attr]
            ),
            minilm_facet_unit=tuple(  # type: ignore[arg-type]
                tuple(row) for row in value["minilm_facet_unit"]  # type: ignore[union-attr]
            ),
            minilm_unit_unit=tuple(  # type: ignore[arg-type]
                tuple(row) for row in value["minilm_unit_unit"]  # type: ignore[union-attr]
            ),
        )
    except (KeyError, TypeError) as exc:
        raise HitabDmc1CoreError("precomputed view payload is malformed") from exc


@dataclass(frozen=True)
class FeatureVector:
    """The fixed ten-dimensional state-action vector ``phi(S,e)``."""

    values: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.values) != len(FEATURE_NAMES):
            raise HitabDmc1CoreError("feature width drifted")
        bound = 64 * QUANT_SCALE
        if any(type(value) is not int or not -bound <= value <= bound for value in self.values):
            raise HitabDmc1CoreError("feature coordinate drifted")

    def payload(self) -> list[int]:
        return list(self.values)


def _integer_mean(values: Sequence[int]) -> int:
    if not values:
        raise HitabDmc1CoreError("cannot average an empty coordinate array")
    # Floor division is exact, platform-independent integer quantization.
    return sum(values) // len(values)


def action_features(
    view: PrecomputedView,
    selected: Sequence[int],
    candidate_ordinal: int,
) -> FeatureVector:
    """Compute the only frozen E0/E1 state-action representation."""

    state = _strict_selected(
        selected, unit_count=view.unit_count, allow_full=False
    )
    candidate = _strict_ordinal(
        candidate_ordinal,
        unit_count=view.unit_count,
        field="candidate_ordinal",
    )
    if candidate in state:
        raise HitabDmc1CoreError("candidate is already selected")

    candidate_ce = tuple(row[candidate] for row in view.ce_facet_unit)
    candidate_minilm = tuple(
        row[candidate] for row in view.minilm_facet_unit
    )

    ce_residual = 0
    minilm_residual = 0
    for facet_index in range(len(view.question_facets)):
        old_ce = max(
            (0, *(view.ce_facet_unit[facet_index][row] for row in state))
        )
        old_minilm = max(
            (
                0,
                *(
                    view.minilm_facet_unit[facet_index][row]
                    for row in state
                ),
            )
        )
        ce_residual += max(0, candidate_ce[facet_index] - old_ce)
        minilm_residual += max(
            0, candidate_minilm[facet_index] - old_minilm
        )

    selected_types = {view.unit_types[row] for row in state}
    type_novelty = (
        QUANT_SCALE
        if view.unit_types[candidate] not in selected_types
        else 0
    )
    incoming = (
        sum((row, candidate) in view._edge_pairs for row in state)
        * QUANT_SCALE
    )
    outgoing = (
        sum((candidate, row) in view._edge_pairs for row in state)
        * QUANT_SCALE
    )
    nonredundancy = sum(
        QUANT_SCALE - view.minilm_unit_unit[candidate][row]
        for row in state
    )

    return FeatureVector(
        (
            max(candidate_ce),
            _integer_mean(candidate_ce),
            max(candidate_minilm),
            _integer_mean(candidate_minilm),
            ce_residual,
            minilm_residual,
            type_novelty,
            incoming,
            outgoing,
            nonredundancy,
        )
    )


def v0_marginal(features: FeatureVector) -> int:
    """Return the frozen label-free V0 marginal for one action."""

    return sum(
        weight * coordinate
        for weight, coordinate in zip(V0_WEIGHTS, features.values)
    )


def v0_state_value(view: PrecomputedView, selected: Sequence[int]) -> int:
    """Canonical insertion-order-independent V0 value of a selected set."""

    state = _strict_selected(selected, unit_count=view.unit_count)
    prefix: tuple[int, ...] = ()
    total = 0
    for candidate in state:
        total += v0_marginal(action_features(view, prefix, candidate))
        prefix = tuple(sorted((*prefix, candidate)))
    return total


def select_e0(view: PrecomputedView) -> tuple[int, ...]:
    """Rewrite a complete K=5 set from empty with the fixed V0 marginal."""

    selected_set: tuple[int, ...] = ()
    ordered_output: list[int] = []
    while len(ordered_output) < TOP_K:
        rows = []
        for candidate in range(view.unit_count):
            if candidate in selected_set:
                continue
            phi = action_features(view, selected_set, candidate)
            rows.append((v0_marginal(phi), candidate))
        _score, chosen = min(rows, key=lambda row: (-row[0], row[1]))
        ordered_output.append(chosen)
        selected_set = tuple(sorted((*selected_set, chosen)))
    return tuple(ordered_output)


@dataclass(frozen=True)
class SealedAction:
    candidate_ordinal: int
    phi: FeatureVector

    def payload(self) -> dict[str, object]:
        return {
            "candidate_ordinal": self.candidate_ordinal,
            "phi": self.phi.payload(),
        }


@dataclass(frozen=True)
class SealedState:
    depth: int
    selected_ordinals: tuple[int, ...]
    state_class: str
    state_sha256: str
    v0_value: int
    actions: tuple[SealedAction, ...]

    def payload(self) -> dict[str, object]:
        return {
            "actions": [row.payload() for row in self.actions],
            "depth": self.depth,
            "selected_ordinals": list(self.selected_ordinals),
            "state_class": self.state_class,
            "state_sha256": self.state_sha256,
            "v0_value": self.v0_value,
        }


@dataclass(frozen=True)
class SealedAFormRegistry:
    """Immutable, qrel-free A_form state/action registry."""

    corpus_commitment: str
    view_sha256: str
    unit_count: int
    exploration_key_commitment: str
    states: tuple[SealedState, ...]
    seal_sha256: str


def _state_identity(depth: int, selected: tuple[int, ...]) -> str:
    return stable_hash(
        {
            "depth": depth,
            "schema": f"{VERSION}_state_identity_v1",
            "selected_ordinals": list(selected),
        }
    )


def _exploration_digest(
    key: bytes,
    *,
    view_sha256: str,
    depth: int,
    selected: tuple[int, ...],
) -> str:
    body = {
        "depth": depth,
        "schema": f"{VERSION}_hmac_exploration_v1",
        "selected_ordinals": list(selected),
        "view_sha256": view_sha256,
    }
    return hmac.new(key, canonical_bytes(body), hashlib.sha256).hexdigest()


def _registry_body(
    *,
    corpus_commitment: str,
    view_sha256: str,
    unit_count: int,
    exploration_key_commitment: str,
    states: Sequence[SealedState],
) -> dict[str, object]:
    return {
        "a_form_hmac_state_cap": A_FORM_HMAC_STATE_CAP,
        "a_form_v0_state_cap": A_FORM_V0_STATE_CAP,
        "corpus_commitment": corpus_commitment,
        "exploration_key_commitment": exploration_key_commitment,
        "feature_names": list(FEATURE_NAMES),
        "ridge_lambda": {
            "denominator": RIDGE_LAMBDA.denominator,
            "numerator": RIDGE_LAMBDA.numerator,
        },
        "schema": f"{VERSION}_sealed_a_form_registry_v1",
        "study_id": STUDY_ID,
        "target": "60_times_exact_DNF_set_utility_marginal",
        "target_scale": TARGET_SCALE,
        "top_k": TOP_K,
        "unit_count": unit_count,
        "v0_weights": list(V0_WEIGHTS),
        "view_sha256": view_sha256,
        "states": [row.payload() for row in states],
    }


def registry_payload(registry: SealedAFormRegistry) -> dict[str, object]:
    """Serialize the safe pre-label archive with its canonical self-hash."""

    _validate_registry(registry)
    body = _registry_body(
        corpus_commitment=registry.corpus_commitment,
        view_sha256=registry.view_sha256,
        unit_count=registry.unit_count,
        exploration_key_commitment=registry.exploration_key_commitment,
        states=registry.states,
    )
    body["self_sha256"] = registry.seal_sha256
    return body


def _validate_registry(registry: SealedAFormRegistry) -> None:
    if (
        not isinstance(registry.corpus_commitment, str)
        or re.fullmatch(r"[0-9a-f]{64}", registry.corpus_commitment) is None
        or not isinstance(registry.view_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", registry.view_sha256) is None
        or not isinstance(registry.exploration_key_commitment, str)
        or re.fullmatch(
            r"[0-9a-f]{64}", registry.exploration_key_commitment
        )
        is None
        or type(registry.unit_count) is not int
        or not MIN_UNITS <= registry.unit_count <= MAX_UNITS
    ):
        raise HitabDmc1CoreError("sealed registry header drifted")
    expected_order = sorted(
        registry.states, key=lambda state: (state.depth, state.selected_ordinals)
    )
    if list(registry.states) != expected_order or not registry.states:
        raise HitabDmc1CoreError("sealed state order drifted")
    if {state.depth for state in registry.states} != set(range(TOP_K)):
        raise HitabDmc1CoreError("sealed registry depth coverage drifted")
    for depth in range(TOP_K):
        depth_states = [
            state for state in registry.states if state.depth == depth
        ]
        if (
            not 1
            <= sum(state.state_class == TOP_V0 for state in depth_states)
            <= A_FORM_V0_STATE_CAP
            or sum(
                state.state_class == HMAC_EXPLORATION
                for state in depth_states
            )
            > A_FORM_HMAC_STATE_CAP
        ):
            raise HitabDmc1CoreError("sealed state-class capacity drifted")
    depth_zero = [state for state in registry.states if state.depth == 0]
    if (
        len(depth_zero) != 1
        or depth_zero[0].selected_ordinals
        or depth_zero[0].state_class != TOP_V0
    ):
        raise HitabDmc1CoreError("sealed empty-state contract drifted")
    seen_states: set[tuple[int, tuple[int, ...]]] = set()
    for state in registry.states:
        selected = _strict_selected(
            state.selected_ordinals,
            unit_count=registry.unit_count,
            allow_full=False,
        )
        if (
            type(state.depth) is not int
            or state.depth != len(selected)
            or not 0 <= state.depth < TOP_K
            or state.state_class not in STATE_CLASSES
            or type(state.v0_value) is not int
            or state.state_sha256
            != _state_identity(state.depth, selected)
        ):
            raise HitabDmc1CoreError("sealed state header drifted")
        identity = (state.depth, selected)
        if identity in seen_states:
            raise HitabDmc1CoreError("sealed registry duplicates a state")
        seen_states.add(identity)
        expected_candidates = tuple(
            ordinal
            for ordinal in range(registry.unit_count)
            if ordinal not in selected
        )
        actual_candidates = tuple(
            action.candidate_ordinal for action in state.actions
        )
        if actual_candidates != expected_candidates:
            raise HitabDmc1CoreError(
                "sealed state does not contain every remaining action"
            )
        if any(not isinstance(action.phi, FeatureVector) for action in state.actions):
            raise HitabDmc1CoreError("sealed action feature drifted")
    body = _registry_body(
        corpus_commitment=registry.corpus_commitment,
        view_sha256=registry.view_sha256,
        unit_count=registry.unit_count,
        exploration_key_commitment=registry.exploration_key_commitment,
        states=registry.states,
    )
    if registry.seal_sha256 != stable_hash(body):
        raise HitabDmc1CoreError("sealed registry self-hash drifted")


def build_and_seal_aform_registry(
    view: PrecomputedView, *, exploration_key: bytes
) -> SealedAFormRegistry:
    """Build top-8 V0 plus 8 HMAC exploration states at every depth.

    At a depth with fewer available unique states, all available states are
    retained.  HMAC states are drawn without overlap from the states not
    already retained by V0.  The key itself never enters the archive.
    """

    if not isinstance(exploration_key, bytes) or len(exploration_key) < 16:
        raise HitabDmc1CoreError(
            "A_form exploration key must contain at least 16 bytes"
        )
    key_commitment = hashlib.sha256(exploration_key).hexdigest()
    frontier: dict[tuple[int, ...], str] = {(): TOP_V0}
    sealed_states: list[SealedState] = []
    feature_cache: dict[tuple[tuple[int, ...], int], FeatureVector] = {}
    state_value_cache: dict[tuple[int, ...], int] = {(): 0}

    def cached_features(
        selected: tuple[int, ...], candidate: int
    ) -> FeatureVector:
        identity = (selected, candidate)
        if identity not in feature_cache:
            feature_cache[identity] = action_features(
                view, selected, candidate
            )
        return feature_cache[identity]

    def cached_state_value(selected: tuple[int, ...]) -> int:
        if selected not in state_value_cache:
            prefix = selected[:-1]
            candidate = selected[-1]
            state_value_cache[selected] = cached_state_value(
                prefix
            ) + v0_marginal(cached_features(prefix, candidate))
        return state_value_cache[selected]

    for depth in range(TOP_K):
        for selected in sorted(frontier):
            actions = tuple(
                SealedAction(
                    candidate,
                    cached_features(selected, candidate),
                )
                for candidate in range(view.unit_count)
                if candidate not in selected
            )
            sealed_states.append(
                SealedState(
                    depth=depth,
                    selected_ordinals=selected,
                    state_class=frontier[selected],
                    state_sha256=_state_identity(depth, selected),
                    v0_value=cached_state_value(selected),
                    actions=actions,
                )
            )

        if depth == TOP_K - 1:
            break
        candidates: set[tuple[int, ...]] = set()
        for selected in frontier:
            for candidate in range(view.unit_count):
                if candidate not in selected:
                    candidates.add(tuple(sorted((*selected, candidate))))

        v0_order = sorted(
            candidates,
            key=lambda selected: (
                -cached_state_value(selected),
                selected,
            ),
        )
        top = v0_order[:A_FORM_V0_STATE_CAP]
        top_set = set(top)
        exploration_pool = [row for row in candidates if row not in top_set]
        exploration = sorted(
            exploration_pool,
            key=lambda selected: (
                _exploration_digest(
                    exploration_key,
                    view_sha256=view.sha256,
                    depth=depth + 1,
                    selected=selected,
                ),
                selected,
            ),
        )[:A_FORM_HMAC_STATE_CAP]
        frontier = {
            **{selected: TOP_V0 for selected in top},
            **{selected: HMAC_EXPLORATION for selected in exploration},
        }

    ordered_states = tuple(
        sorted(
            sealed_states,
            key=lambda state: (state.depth, state.selected_ordinals),
        )
    )
    body = _registry_body(
        corpus_commitment=view.corpus_commitment,
        view_sha256=view.sha256,
        unit_count=view.unit_count,
        exploration_key_commitment=key_commitment,
        states=ordered_states,
    )
    registry = SealedAFormRegistry(
        corpus_commitment=view.corpus_commitment,
        view_sha256=view.sha256,
        unit_count=view.unit_count,
        exploration_key_commitment=key_commitment,
        states=ordered_states,
        seal_sha256=stable_hash(body),
    )
    _validate_registry(registry)
    return registry


@dataclass(frozen=True)
class ProofDNF:
    """Post-seal DNF qrel: alternatives -> buckets -> accepted ordinals."""

    alternatives: tuple[tuple[tuple[int, ...], ...], ...]
    corpus_commitment: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.corpus_commitment, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.corpus_commitment) is None
        ):
            raise HitabDmc1CoreError("DNF corpus commitment is invalid")
        if not self.alternatives:
            raise HitabDmc1CoreError("DNF must contain an alternative")
        canonical_alternatives: list[tuple[tuple[int, ...], ...]] = []
        for alternative_index, alternative in enumerate(self.alternatives):
            if not 1 <= len(alternative) <= TOP_K:
                raise HitabDmc1CoreError(
                    "each DNF alternative must contain one to five buckets"
                )
            buckets: list[tuple[int, ...]] = []
            for bucket_index, raw_bucket in enumerate(alternative):
                if isinstance(raw_bucket, (str, bytes)) or not isinstance(
                    raw_bucket, Sequence
                ):
                    raise HitabDmc1CoreError("DNF bucket must be an array")
                if not raw_bucket:
                    raise HitabDmc1CoreError("DNF bucket cannot be empty")
                if any(type(row) is not int or row < 0 for row in raw_bucket):
                    raise HitabDmc1CoreError(
                        "DNF bucket ordinals must be nonnegative integers"
                    )
                bucket = tuple(sorted(set(raw_bucket)))
                if len(bucket) != len(raw_bucket):
                    raise HitabDmc1CoreError(
                        f"DNF bucket {alternative_index}:{bucket_index} duplicates units"
                    )
                buckets.append(bucket)
            canonical = tuple(sorted(buckets))
            if len(set(canonical)) != len(canonical):
                raise HitabDmc1CoreError(
                    "DNF alternative duplicates a requirement bucket"
                )
            canonical_alternatives.append(canonical)
        canonical_dnf = tuple(sorted(canonical_alternatives))
        if len(set(canonical_dnf)) != len(canonical_dnf):
            raise HitabDmc1CoreError("DNF duplicates an alternative")
        object.__setattr__(self, "alternatives", canonical_dnf)

    @property
    def ordinal_mapping_commitment(self) -> str:
        return stable_hash(
            [
                [list(bucket) for bucket in alternative]
                for alternative in self.alternatives
            ]
        )

    def payload(self) -> dict[str, object]:
        return {
            "alternatives": [
                [list(bucket) for bucket in alternative]
                for alternative in self.alternatives
            ],
            "corpus_commitment": self.corpus_commitment,
            "ordinal_mapping_commitment": self.ordinal_mapping_commitment,
        }


def set_utility(
    selected_ordinals: Sequence[int],
    proof_dnf: ProofDNF,
    *,
    unit_count: int | None = None,
) -> Fraction:
    """Return ``max_proof(covered_fraction + complete)`` exactly.

    Alternatives are scored independently.  In particular, coverage from two
    incomplete alternatives is never unioned into an invented proof.
    """

    if isinstance(selected_ordinals, (str, bytes)) or not isinstance(
        selected_ordinals, Sequence
    ):
        raise HitabDmc1CoreError("utility state must be an ordinal array")
    if unit_count is None:
        if not proof_dnf.alternatives:
            raise HitabDmc1CoreError("utility DNF is empty")
        maximum = max(
            ordinal
            for alternative in proof_dnf.alternatives
            for bucket in alternative
            for ordinal in bucket
        )
        unit_count = max(maximum + 1, TOP_K)
    state = _strict_selected(
        selected_ordinals, unit_count=unit_count, allow_full=True
    )
    selected = set(state)
    best = Fraction(0, 1)
    for alternative in proof_dnf.alternatives:
        covered = sum(bool(selected & set(bucket)) for bucket in alternative)
        complete = int(covered == len(alternative))
        score = Fraction(covered, len(alternative)) + complete
        best = max(best, score)
    return best


@dataclass(frozen=True)
class LabelledAction:
    candidate_ordinal: int
    phi: FeatureVector
    target_y: int

    def __post_init__(self) -> None:
        if (
            type(self.candidate_ordinal) is not int
            or self.candidate_ordinal < 0
            or not isinstance(self.phi, FeatureVector)
            or type(self.target_y) is not int
            or not 0 <= self.target_y <= 2 * TARGET_SCALE
        ):
            raise HitabDmc1CoreError("labelled action contract drifted")


@dataclass(frozen=True)
class LabelledState:
    depth: int
    selected_ordinals: tuple[int, ...]
    state_sha256: str
    actions: tuple[LabelledAction, ...]

    def __post_init__(self) -> None:
        if (
            type(self.depth) is not int
            or self.depth != len(self.selected_ordinals)
            or not 0 <= self.depth < TOP_K
            or self.selected_ordinals
            != tuple(sorted(set(self.selected_ordinals)))
            or self.state_sha256
            != _state_identity(self.depth, self.selected_ordinals)
            or not self.actions
        ):
            raise HitabDmc1CoreError("labelled state contract drifted")
        candidates = tuple(row.candidate_ordinal for row in self.actions)
        if candidates != tuple(sorted(set(candidates))):
            raise HitabDmc1CoreError("labelled action order drifted")
        if set(candidates) & set(self.selected_ordinals):
            raise HitabDmc1CoreError("labelled action repeats a selected unit")


@dataclass(frozen=True)
class LabelledRegistry:
    seal_sha256: str
    corpus_commitment: str
    ordinal_mapping_commitment: str
    labelled_states: tuple[LabelledState, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.seal_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.seal_sha256) is None
            or not isinstance(self.corpus_commitment, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.corpus_commitment) is None
            or not isinstance(self.ordinal_mapping_commitment, str)
            or re.fullmatch(
                r"[0-9a-f]{64}", self.ordinal_mapping_commitment
            )
            is None
            or not self.labelled_states
            or tuple(self.labelled_states)
            != tuple(
                sorted(
                    self.labelled_states,
                    key=lambda row: (row.depth, row.selected_ordinals),
                )
            )
        ):
            raise HitabDmc1CoreError("labelled registry contract drifted")


def label_sealed_registry(
    registry: SealedAFormRegistry, proof_dnf: ProofDNF
) -> LabelledRegistry:
    """Attach exact post-seal targets without recomputing any feature."""

    _validate_registry(registry)
    if proof_dnf.corpus_commitment != registry.corpus_commitment:
        raise HitabDmc1CoreError(
            "DNF and sealed registry corpus commitments differ"
        )
    for alternative in proof_dnf.alternatives:
        for bucket in alternative:
            for ordinal in bucket:
                _strict_ordinal(
                    ordinal,
                    unit_count=registry.unit_count,
                    field="DNF qrel ordinal",
                )

    labelled_states: list[LabelledState] = []
    for state in registry.states:
        base_utility = set_utility(
            state.selected_ordinals,
            proof_dnf,
            unit_count=registry.unit_count,
        )
        actions: list[LabelledAction] = []
        for action in state.actions:
            rewritten = tuple(
                sorted((*state.selected_ordinals, action.candidate_ordinal))
            )
            delta = (
                set_utility(
                    rewritten,
                    proof_dnf,
                    unit_count=registry.unit_count,
                )
                - base_utility
            )
            scaled = delta * TARGET_SCALE
            if scaled.denominator != 1:
                raise HitabDmc1CoreError(
                    "DNF marginal is outside the exact scale-60 contract"
                )
            actions.append(
                LabelledAction(
                    candidate_ordinal=action.candidate_ordinal,
                    phi=action.phi,
                    target_y=scaled.numerator,
                )
            )
        labelled_states.append(
            LabelledState(
                depth=state.depth,
                selected_ordinals=state.selected_ordinals,
                state_sha256=state.state_sha256,
                actions=tuple(actions),
            )
        )
    return LabelledRegistry(
        seal_sha256=registry.seal_sha256,
        corpus_commitment=registry.corpus_commitment,
        ordinal_mapping_commitment=proof_dnf.ordinal_mapping_commitment,
        labelled_states=tuple(labelled_states),
    )


def _round_fraction_half_away(value: Fraction, scale: int) -> int:
    scaled_numerator = value.numerator * scale
    sign = -1 if scaled_numerator < 0 else 1
    quotient, remainder = divmod(abs(scaled_numerator), value.denominator)
    if 2 * remainder >= value.denominator:
        quotient += 1
    return sign * quotient


def _solve_fraction_system(
    matrix: Sequence[Sequence[Fraction]], rhs: Sequence[Fraction]
) -> tuple[Fraction, ...]:
    width = len(rhs)
    if len(matrix) != width or any(len(row) != width for row in matrix):
        raise HitabDmc1CoreError("ridge system shape drifted")
    augmented = [
        [Fraction(value) for value in matrix[row]] + [Fraction(rhs[row])]
        for row in range(width)
    ]
    for column in range(width):
        pivot = next(
            (
                row
                for row in range(column, width)
                if augmented[row][column] != 0
            ),
            None,
        )
        if pivot is None:
            raise HitabDmc1CoreError("lambda-one ridge is unexpectedly singular")
        if pivot != column:
            augmented[column], augmented[pivot] = (
                augmented[pivot],
                augmented[column],
            )
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(width):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [
                    left - factor * right
                    for left, right in zip(
                        augmented[row], augmented[column]
                    )
                ]
    return tuple(augmented[row][-1] for row in range(width))


@dataclass(frozen=True)
class E1Model:
    """The frozen quantized no-intercept DMC-1 challenger."""

    coefficient_q: tuple[int, ...]
    training_registry_count: int
    training_state_count: int
    training_row_count: int
    training_corpus_set_commitment: str
    training_qrel_mapping_set_commitment: str
    training_corpus_qrel_binding_set_commitment: str

    def __post_init__(self) -> None:
        if (
            len(self.coefficient_q) != len(FEATURE_NAMES)
            or any(type(value) is not int for value in self.coefficient_q)
            or type(self.training_registry_count) is not int
            or self.training_registry_count <= 0
            or type(self.training_state_count) is not int
            or self.training_state_count <= 0
            or type(self.training_row_count) is not int
            or self.training_row_count <= self.training_state_count
            or not isinstance(
                self.training_corpus_set_commitment, str
            )
            or re.fullmatch(
                r"[0-9a-f]{64}", self.training_corpus_set_commitment
            )
            is None
            or not isinstance(
                self.training_qrel_mapping_set_commitment, str
            )
            or re.fullmatch(
                r"[0-9a-f]{64}",
                self.training_qrel_mapping_set_commitment,
            )
            is None
            or not isinstance(
                self.training_corpus_qrel_binding_set_commitment, str
            )
            or re.fullmatch(
                r"[0-9a-f]{64}",
                self.training_corpus_qrel_binding_set_commitment,
            )
            is None
        ):
            raise HitabDmc1CoreError("E1 model contract drifted")


def fit_e1(labelled_registries: Sequence[LabelledRegistry]) -> E1Model:
    """Fit exact within-state-centered lambda-one ridge, then quantize once."""

    if not labelled_registries:
        raise HitabDmc1CoreError("A_form labelled registries are empty")
    ordered = sorted(labelled_registries, key=lambda row: row.seal_sha256)
    if len({row.seal_sha256 for row in ordered}) != len(ordered):
        raise HitabDmc1CoreError("A_form repeats a sealed registry")
    if len({row.corpus_commitment for row in ordered}) != len(ordered):
        raise HitabDmc1CoreError("A_form repeats a corpus commitment")

    width = len(FEATURE_NAMES)
    gram = [
        [
            RIDGE_LAMBDA if left == right else Fraction(0, 1)
            for right in range(width)
        ]
        for left in range(width)
    ]
    rhs = [Fraction(0, 1) for _ in range(width)]
    state_count = 0
    row_count = 0

    for registry in ordered:
        if (
            re.fullmatch(r"[0-9a-f]{64}", registry.seal_sha256) is None
            or not registry.labelled_states
        ):
            raise HitabDmc1CoreError("labelled registry header drifted")
        states = sorted(
            registry.labelled_states,
            key=lambda row: (row.depth, row.selected_ordinals),
        )
        if list(registry.labelled_states) != states:
            raise HitabDmc1CoreError("labelled state order drifted")
        for state in states:
            if state.depth != len(state.selected_ordinals) or not state.actions:
                raise HitabDmc1CoreError("labelled state shape drifted")
            candidates = tuple(row.candidate_ordinal for row in state.actions)
            if candidates != tuple(sorted(candidates)) or len(set(candidates)) != len(
                candidates
            ):
                raise HitabDmc1CoreError("labelled action order drifted")
            n = len(state.actions)
            feature_sums = [
                sum(action.phi.values[column] for action in state.actions)
                for column in range(width)
            ]
            target_sum = sum(action.target_y for action in state.actions)
            feature_target_sums = [
                sum(
                    action.phi.values[column] * action.target_y
                    for action in state.actions
                )
                for column in range(width)
            ]
            feature_cross = [
                [
                    sum(
                        action.phi.values[left]
                        * action.phi.values[right]
                        for action in state.actions
                    )
                    for right in range(width)
                ]
                for left in range(width)
            ]
            for left in range(width):
                rhs[left] += (
                    Fraction(
                        feature_target_sums[left], 1
                    )
                    - Fraction(feature_sums[left] * target_sum, n)
                ) / QUANT_SCALE
                for right in range(width):
                    gram[left][right] += (
                        Fraction(feature_cross[left][right], 1)
                        - Fraction(
                            feature_sums[left] * feature_sums[right], n
                        )
                    ) / (QUANT_SCALE * QUANT_SCALE)
            state_count += 1
            row_count += n

    exact_coefficients = _solve_fraction_system(gram, rhs)
    quantized = tuple(
        _round_fraction_half_away(value, COEFFICIENT_SCALE)
        for value in exact_coefficients
    )
    return E1Model(
        coefficient_q=quantized,
        training_registry_count=len(ordered),
        training_state_count=state_count,
        training_row_count=row_count,
        training_corpus_set_commitment=stable_hash(
            sorted(row.corpus_commitment for row in ordered)
        ),
        training_qrel_mapping_set_commitment=stable_hash(
            sorted(
                {
                    row.ordinal_mapping_commitment
                    for row in ordered
                }
            )
        ),
        training_corpus_qrel_binding_set_commitment=stable_hash(
            sorted(
                [
                    row.corpus_commitment,
                    row.ordinal_mapping_commitment,
                ]
                for row in ordered
            )
        ),
    )


def model_payload(model: E1Model) -> dict[str, object]:
    """Serialize a canonical, self-hashed E1 model receipt."""

    body: dict[str, object] = {
        "coefficient_q": list(model.coefficient_q),
        "coefficient_scale": COEFFICIENT_SCALE,
        "feature_names": list(FEATURE_NAMES),
        "fit": "within_state_centered_ridge",
        "intercept": False,
        "ridge_lambda": {
            "denominator": RIDGE_LAMBDA.denominator,
            "numerator": RIDGE_LAMBDA.numerator,
        },
        "schema": f"{VERSION}_E1_model_v1",
        "study_id": STUDY_ID,
        "target": "60_times_exact_DNF_set_utility_marginal",
        "training_registry_count": model.training_registry_count,
        "training_row_count": model.training_row_count,
        "training_state_count": model.training_state_count,
        "training_corpus_set_commitment": (
            model.training_corpus_set_commitment
        ),
        "training_qrel_mapping_set_commitment": (
            model.training_qrel_mapping_set_commitment
        ),
        "training_corpus_qrel_binding_set_commitment": (
            model.training_corpus_qrel_binding_set_commitment
        ),
    }
    body["self_sha256"] = stable_hash(body)
    return body


def select_e1(view: PrecomputedView, model: E1Model) -> tuple[int, ...]:
    """Rewrite all five units by a pure ``beta dot phi`` argmax."""

    # Revalidation also rejects forged dataclass instances.
    E1Model(
        coefficient_q=tuple(model.coefficient_q),
        training_registry_count=model.training_registry_count,
        training_state_count=model.training_state_count,
        training_row_count=model.training_row_count,
        training_corpus_set_commitment=(
            model.training_corpus_set_commitment
        ),
        training_qrel_mapping_set_commitment=(
            model.training_qrel_mapping_set_commitment
        ),
        training_corpus_qrel_binding_set_commitment=(
            model.training_corpus_qrel_binding_set_commitment
        ),
    )
    selected_set: tuple[int, ...] = ()
    ordered_output: list[int] = []
    for _depth in range(TOP_K):
        scored: list[tuple[int, int]] = []
        for candidate in range(view.unit_count):
            if candidate in selected_set:
                continue
            phi = action_features(view, selected_set, candidate)
            score = sum(
                coefficient * coordinate
                for coefficient, coordinate in zip(
                    model.coefficient_q, phi.values
                )
            )
            scored.append((score, candidate))
        _score, chosen = min(scored, key=lambda row: (-row[0], row[1]))
        ordered_output.append(chosen)
        selected_set = tuple(sorted((*selected_set, chosen)))
    if len(set(ordered_output)) != TOP_K:
        raise HitabDmc1CoreError("E1 did not produce five unique units")
    return tuple(ordered_output)


@dataclass(frozen=True)
class ExactPairedComparison:
    net_utility: Fraction
    positive_count: int
    negative_count: int
    tie_count: int
    reference_tail: Fraction


def exact_sign_flip(deltas: Sequence[Fraction]) -> Fraction:
    """Exact one-sided magnitude-preserving paired random-sign tail."""

    if not deltas or len(deltas) > MAX_SIGN_FLIP_PAIRS:
        raise HitabDmc1CoreError("paired sign-flip count is outside contract")
    checked: list[Fraction] = []
    for value in deltas:
        if not isinstance(value, Fraction):
            raise HitabDmc1CoreError("paired deltas must be exact Fractions")
        lifted = value * TARGET_SCALE
        if lifted.denominator != 1 or abs(value) > 2:
            raise HitabDmc1CoreError(
                "paired delta is outside the scale-60 DNF utility lattice"
            )
        if value:
            checked.append(value)
    if not checked:
        return Fraction(1, 1)
    integer_deltas = [
        (value * TARGET_SCALE).numerator for value in checked
    ]
    observed = sum(integer_deltas)
    distribution: Counter[int] = Counter({0: 1})
    for magnitude in map(abs, integer_deltas):
        next_distribution: Counter[int] = Counter()
        for total, count in distribution.items():
            next_distribution[total + magnitude] += count
            next_distribution[total - magnitude] += count
        distribution = next_distribution
    favorable = sum(
        count for total, count in distribution.items() if total >= observed
    )
    return Fraction(favorable, 2 ** len(integer_deltas))


def compare_paired(
    candidate: Sequence[Fraction], baseline: Sequence[Fraction]
) -> ExactPairedComparison:
    """Compare paired exact DNF utilities without any family grouping."""

    if len(candidate) != len(baseline) or not candidate:
        raise HitabDmc1CoreError("paired comparison shape drifted")
    deltas = tuple(left - right for left, right in zip(candidate, baseline))
    return ExactPairedComparison(
        net_utility=sum(deltas, Fraction(0, 1)),
        positive_count=sum(value > 0 for value in deltas),
        negative_count=sum(value < 0 for value in deltas),
        tie_count=sum(value == 0 for value in deltas),
        reference_tail=exact_sign_flip(deltas),
    )
