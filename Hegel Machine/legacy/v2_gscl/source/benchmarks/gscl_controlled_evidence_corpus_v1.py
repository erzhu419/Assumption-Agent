"""Blind atomic-fact corpus for the iterative, non-scoring GSCL harness.

This is public synthetic implementation evidence, not a scored study.  Its
unit of construction is a *paired case*, not 25 independent examples:

* ten primary roots (two per structural family);
* ten counterfactual negatives, each derived from one primary by a frozen
  pure operator; and
* five missingness controls, each paired with a family primary.

Every runtime item is a collection of law-agnostic atomic records.  There are
no free-text notes, law-specific narrative surrogates, law ids, canonical
roles, residual names, or expected outcomes in the raw pack.  The four views
test only serialization and field-aware alias invariance; they are not
natural-language paraphrases.  Runtime extraction receives only
``source_bytes`` and ``media_type``.  Pair identities, operators, law labels,
and expected outcomes remain in this benchmark-side gold pack.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import hashlib
import inspect
import json
import re
from typing import Any, Callable, Mapping, Sequence

from assumption_agent.generalized_structural_correspondence_v1 import (
    ExactRational,
    strict_content_hash,
)
CORPUS_VERSION = "gscl.controlled.atomic_facts.corpus.v3"
JSONL_MEDIA_TYPE = "application/vnd.gscl-neutral-records+jsonl"
PROSE_MEDIA_TYPE = "text/vnd.gscl-neutral-records"
ROOT_COUNT = 25
PRIMARY_ROOT_COUNT = 10
PAIRED_NEGATIVE_ROOT_COUNT = 10
PAIRED_CONTROL_ROOT_COUNT = 5
VIEWS_PER_ROOT = 4
VIEW_COUNT = ROOT_COUNT * VIEWS_PER_ROOT

T05 = "gscl.v1.t05_pair_interaction"
T09 = "gscl.v1.t09_path_composition"
T14 = "gscl.v1.t14_finite_equivariance"
T15 = "gscl.v1.t15_closed_balance"
T17 = "gscl.v1.t17_monotone_order"

OPERATOR_VERSION = "gscl.atomic_counterfactual_operators.v2"

_KIND_ALIAS = {
    "assertion": "statement",
    "edge": "link",
    "map": "mapping",
    "node": "entity",
    "observation": "reading",
    "subset_outcome": "subset_result",
    "transfer": "movement",
}
_KEY_ALIAS = {
    "amount": "magnitude",
    "boundary": "scope_ref",
    "dimension": "axis",
    "direction": "orientation",
    "endpoint": "endpoint_role",
    "fold": "replicate",
    "members": "participants",
    "mode": "mapping_mode",
    "order": "position",
    "ordinal": "index",
    "owner": "subject",
    "path_kind": "route_kind",
    "permutation": "reindex",
    "phase": "timepoint",
    "predicate": "claim",
    "relation": "relation_type",
    "rows": "pairs",
    "signs": "orientations",
    "sort": "category",
    "source": "from",
    "stage": "lane",
    "subset": "participants",
    "target": "to",
    "unit": "scale",
    "utility": "score",
    "value": "reading",
    "values": "items",
}
_VALUE_ALIAS = {
    "after": "later",
    "association": "group",
    "before": "earlier",
    "bundle": "record_bundle",
    "complementary": "positive_joint",
    "directed": "directed_link",
    "entity": "generic_entity",
    "inflow": "incoming",
    "outflow": "outgoing",
    "property": "value_claim",
    "reference": "comparison",
    "scalar": "scalar_claim",
    "set": "set_claim",
    "sink": "removed",
    "source": "created",
    "step_a": "initial",
    "step_b": "terminal",
    "unavailable": "withheld",
}
_REVERSE_ALIAS_LIST_KEYS = frozenset(
    {"members", "rows", "subset", "values"}
)

_FORBIDDEN_RAW_FRAGMENTS = (
    "gscl.v1.",
    "finite_equivariance",
    "monotone_order",
    "closed_balance",
    "path_composition",
    "pair_interaction",
    '"law_id"',
    '"root_id"',
    '"pair_id"',
    '"operator_id"',
    '"gold"',
    '"expected_disposition"',
    '"expected_decision"',
    "role:",
    "input_before",
    "input_after",
    "transformation",
    "output_before",
    "output_after",
    "lower_state",
    "upper_state",
    "order_relation",
    "lower_value",
    "upper_value",
    "system_boundary",
    "storage_before",
    "storage_after",
    "flow_ledger",
    "source_state",
    "target_state",
    "composed_path",
    "direct_path",
    "component_a",
    "component_b",
    "component_c",
    "utility_ledger",
    "equivariance_constraint",
    "monotone_constraint",
    "balance_constraint",
    "path_constraint",
    "interaction_constraint",
    "comparable_output_pairs",
    "input_action",
    "output_action",
    "outputs_before",
    "outputs_after",
    "declared_direction",
    "boundary_declaration",
    "quantity_ledger",
    "finite_domain",
    "first_map",
    "second_map",
    "direct_map",
    '"components"',
    "designated_pair",
    "held_fold_utilities",
    "interaction_expectation",
    "residual",
    "hard_negative",
    "counterfactual",
    '"satisfied"',
    '"violated"',
    '"inconclusive"',
    '"not_applicable"',
)


@dataclass(frozen=True)
class CounterfactualOperatorReceipt:
    """Gold-side proof that a paired transform touched exactly its target."""

    operator_id: str
    operator_version: str
    operator_implementation_sha256: str
    frozen_target_spec_hash: str
    input_records_sha256: str
    output_records_sha256: str
    allowed_diff_paths: tuple[str, ...]
    observed_diff_paths: tuple[str, ...]
    non_target_fields_unchanged: bool

    @property
    def receipt_hash(self) -> str:
        return strict_content_hash(
            {
                "operator_id": self.operator_id,
                "operator_version": self.operator_version,
                "operator_implementation_sha256": (
                    self.operator_implementation_sha256
                ),
                "frozen_target_spec_hash": self.frozen_target_spec_hash,
                "input_records_sha256": self.input_records_sha256,
                "output_records_sha256": self.output_records_sha256,
                "allowed_diff_paths": list(self.allowed_diff_paths),
                "observed_diff_paths": list(self.observed_diff_paths),
                "non_target_fields_unchanged": (
                    self.non_target_fields_unchanged
                ),
            }
        )


@dataclass(frozen=True)
class ControlledRoot:
    root_id: str
    law_id: str
    root_kind: str
    records: tuple[Mapping[str, Any], ...]
    expected_disposition: str
    expected_decision: str
    role_targets: Mapping[str, str]
    observable_expectations: Mapping[str, Mapping[str, Any]]
    quantity_expectations: Mapping[str, Mapping[str, Any]]
    relation_expectations: Mapping[str, Mapping[str, Any]]
    hard_negative_operator_id: str | None = None
    pair_id: str | None = None
    paired_primary_root_id: str | None = None
    pair_role: str = "primary"
    operator_receipt: CounterfactualOperatorReceipt | None = None

    @property
    def raw_records_sha256(self) -> str:
        return strict_content_hash(list(self.records))

    @property
    def gold_commitment(self) -> str:
        return strict_content_hash(
            {
                "root_id": self.root_id,
                "law_id": self.law_id,
                "root_kind": self.root_kind,
                "expected_disposition": self.expected_disposition,
                "expected_decision": self.expected_decision,
                "role_targets": dict(sorted(self.role_targets.items())),
                "observable_expectations": dict(
                    sorted(self.observable_expectations.items())
                ),
                "quantity_expectations": dict(
                    sorted(self.quantity_expectations.items())
                ),
                "relation_expectations": dict(
                    sorted(self.relation_expectations.items())
                ),
                "hard_negative_operator_id": (
                    self.hard_negative_operator_id
                ),
                "pair_id": self.pair_id,
                "paired_primary_root_id": (
                    self.paired_primary_root_id
                ),
                "pair_role": self.pair_role,
                "operator_receipt_hash": (
                    None
                    if self.operator_receipt is None
                    else self.operator_receipt.receipt_hash
                ),
            }
        )


@dataclass(frozen=True)
class ControlledView:
    item_id: str
    media_type: str
    source_bytes: bytes
    source_sha256: str
    record_spans: Mapping[str, tuple[int, int]]

    @property
    def view_id(self) -> str:
        """Compatibility alias; both names are the same opaque identity."""

        return self.item_id

    @property
    def raw_commitment(self) -> str:
        return strict_content_hash(
            {
                "item_id": self.item_id,
                "media_type": self.media_type,
                "source_sha256": self.source_sha256,
                "record_spans": {
                    key: list(value)
                    for key, value in sorted(self.record_spans.items())
                },
            }
        )


@dataclass(frozen=True)
class ControlledGoldLink:
    """Private benchmark-side linkage, never a predictor envelope field."""

    item_id: str
    root_id: str
    view_kind: str

    @property
    def linkage_commitment(self) -> str:
        return strict_content_hash(
            {
                "item_id": self.item_id,
                "root_id": self.root_id,
                "view_kind": self.view_kind,
            }
        )


def _r(numerator: int, denominator: int = 1) -> dict[str, int]:
    return ExactRational(numerator, denominator).safe_payload()


def _record(kind: str, record_id: str, **attrs: Any) -> dict[str, Any]:
    return {"kind": kind, "id": record_id, "attrs": attrs}


def _map_rows(rows: Mapping[str, str]) -> list[dict[str, str]]:
    return [
        {"source": source, "target": target}
        for source, target in sorted(rows.items())
    ]


def _observable(
    *,
    value_type: str,
    payload: Any,
    status: str = "inferred",
    dimension: str | None = None,
    unit: str | None = None,
) -> dict[str, Any]:
    return {
        "value_type": value_type,
        "payload": payload,
        "status": status,
        "dimension": dimension,
        "unit": unit,
    }


def _quantity(
    owner: str,
    value: int,
    *,
    dimension: str = "Scalar",
    unit: str = "unitless",
) -> dict[str, Any]:
    return {
        "owner": owner,
        "value": _r(value),
        "dimension": dimension,
        "unit": unit,
    }


def _decision(disposition: str) -> str:
    if disposition == "satisfied":
        return "accepted"
    if disposition in {"inconclusive", "not_applicable"}:
        return "abstain"
    return "rejected"


def _shared_decoy_map() -> dict[str, Any]:
    """A disconnected generic primitive reused across four families."""

    return _record(
        "map",
        "map.decoy",
        stage="reference",
        mode="finite",
        rows=_map_rows({"local:probe_00": "local:probe_00"}),
    )


def _shared_rare_kind_decoys() -> tuple[dict[str, Any], ...]:
    """Disconnected primitives prevent a fact kind from naming a family."""

    return (
        _record(
            "node",
            "node.boundary.decoy",
            sort="entity",
            boundary=False,
        ),
        _record(
            "node",
            "node.phase.decoy",
            sort="entity",
            phase="probe",
        ),
        _record(
            "node",
            "node.ordinal.decoy",
            sort="entity",
            ordinal=99,
        ),
        _record(
            "edge",
            "edge.decoy",
            relation="directed",
            source="node.alpha",
            target="node.alpha",
            order=99,
        ),
        _record(
            "observation",
            "observation.decoy",
            owner="node.alpha",
            phase="probe",
            dimension="Probe",
            unit="unitless",
            value=_r(0),
        ),
        _record(
            "observation",
            "observation.missing.decoy",
            owner="node.alpha",
            phase="probe",
            dimension="Probe",
            unit="unitless",
        ),
        _record(
            "map",
            "map.signed.decoy",
            stage="step_b",
            mode="signed_permutation",
            permutation=[0],
            signs=[1],
        ),
        _record(
            "transfer",
            "transfer.decoy",
            boundary="node.alpha",
            direction="inflow",
            amount=_r(0),
            dimension="Probe",
            unit="unitless",
        ),
        _record(
            "subset_outcome",
            "subset.decoy",
            fold=99,
            subset=[],
            utility=_r(0),
            dimension="Probe",
            unit="unitless",
        ),
        _record(
            "assertion",
            "assertion.scalar.decoy",
            predicate="scalar",
            target="edge.decoy",
            value=1,
        ),
        _record(
            "assertion",
            "assertion.set.decoy",
            predicate="set",
            values=["local:probe_00"],
        ),
        _record(
            "assertion",
            "assertion.property.decoy",
            predicate="property",
            value="neutral",
        ),
    )


def _equivariance_root(
    *,
    root_id: str,
    root_kind: str,
    magnitude: int,
    signs: Sequence[int],
    after_value: int,
    after_available: bool = True,
    expected_disposition: str,
) -> ControlledRoot:
    before_id = "node.alpha"
    after_id = "node.beta"
    edge_id = "edge.alpha"
    before_observation = "observation.alpha"
    after_observation = "observation.beta"
    records = [
        _record("node", before_id, sort="entity", phase="before"),
        _record("node", after_id, sort="entity", phase="after"),
        _record(
            "edge",
            edge_id,
            relation="directed",
            source=before_id,
            target=after_id,
            order=0,
        ),
        _record(
            "observation",
            before_observation,
            owner=before_id,
            phase="before",
            dimension="Scalar",
            unit="unitless",
            value=_r(magnitude),
        ),
        _record(
            "map",
            "map.alpha",
            stage="step_a",
            mode="finite",
            rows=_map_rows(
                {before_id: after_id, after_id: before_id}
            ),
        ),
        _record(
            "map",
            "map.beta",
            stage="step_b",
            mode="signed_permutation",
            permutation=[0],
            signs=list(signs),
        ),
        _shared_decoy_map(),
        *_shared_rare_kind_decoys(),
    ]
    records.append(
        _record(
            "observation",
            after_observation,
            owner=after_id,
            phase="after",
            dimension="Scalar",
            unit="unitless",
            **({"value": _r(after_value)} if after_available else {}),
        )
    )
    bundle_members = [
        str(record["id"])
        for record in records
        if not str(record["id"]).endswith(".decoy")
        and record["kind"] != "subset_outcome"
    ]
    records.extend(
        (
            _record(
                "assertion",
                "assertion.bundle",
                predicate="bundle",
                members=bundle_members,
            ),
            _record(
                "assertion",
                "assertion.association",
                predicate="association",
                members=[before_id, after_id],
            ),
        )
    )
    roles = {
        "input_before": before_id,
        "input_after": after_id,
        "transformation": edge_id,
        "output_before": before_observation,
        "output_after": after_observation,
        "equivariance_constraint": "constraint.derived",
    }
    quantities = {
        before_observation: _quantity(before_id, magnitude),
    }
    if after_available:
        quantities[after_observation] = _quantity(
            after_id, after_value
        )
    return ControlledRoot(
        root_id=root_id,
        law_id=T14,
        root_kind=root_kind,
        records=tuple(records),
        expected_disposition=expected_disposition,
        expected_decision=_decision(expected_disposition),
        role_targets=roles,
        observable_expectations={
            "input_action": _observable(
                value_type="finite_map",
                payload={
                    "rows": [
                        {
                            "source": "role:input_after",
                            "target": "role:input_before",
                        },
                        {
                            "source": "role:input_before",
                            "target": "role:input_after",
                        },
                    ]
                },
            ),
            "output_action": _observable(
                value_type="signed_permutation",
                payload={
                    "permutation": [0],
                    "signs": list(signs),
                },
            ),
            "outputs_before": _observable(
                value_type="exact_vector",
                payload={"values": [_r(magnitude)]},
                dimension="Scalar",
                unit="unitless",
            ),
            "outputs_after": _observable(
                value_type="exact_vector",
                payload=(
                    {"values": [_r(after_value)]}
                    if after_available
                    else None
                ),
                status="inferred" if after_available else "unknown",
                dimension="Scalar",
                unit="unitless",
            ),
        },
        quantity_expectations=quantities,
        relation_expectations={
            edge_id: {
                "type": "Transformation",
                "source": before_id,
                "target": after_id,
                "order": 0,
            }
        },
    )


def _monotone_root(
    *,
    root_id: str,
    root_kind: str,
    lower_value: int,
    upper_value: int,
    direction: int | None,
    expected_disposition: str,
) -> ControlledRoot:
    lower_id = "node.alpha"
    upper_id = "node.beta"
    edge_id = "edge.alpha"
    lower_observation = "observation.alpha"
    upper_observation = "observation.beta"
    records = [
        _record("node", lower_id, sort="entity", phase="before"),
        _record("node", upper_id, sort="entity", phase="after"),
        _record(
            "edge",
            edge_id,
            relation="directed",
            source=lower_id,
            target=upper_id,
            order=0,
        ),
        _record(
            "observation",
            lower_observation,
            owner=lower_id,
            phase="before",
            dimension="Scalar",
            unit="unitless",
            value=_r(lower_value),
        ),
        _record(
            "observation",
            upper_observation,
            owner=upper_id,
            phase="after",
            dimension="Scalar",
            unit="unitless",
            value=_r(upper_value),
        ),
        _shared_decoy_map(),
        *_shared_rare_kind_decoys(),
    ]
    if direction is not None:
        records.append(
            _record(
                "assertion",
                "assertion.gamma",
                predicate="scalar",
                target=edge_id,
                value=direction,
            )
        )
    bundle_members = [
        str(record["id"])
        for record in records
        if not str(record["id"]).endswith(".decoy")
        and record["kind"] != "subset_outcome"
    ]
    records.extend(
        (
            _record(
                "assertion",
                "assertion.bundle",
                predicate="bundle",
                members=bundle_members,
            ),
            _record(
                "assertion",
                "assertion.association",
                predicate="association",
                members=[lower_id, upper_id],
            ),
        )
    )
    return ControlledRoot(
        root_id=root_id,
        law_id=T17,
        root_kind=root_kind,
        records=tuple(records),
        expected_disposition=expected_disposition,
        expected_decision=_decision(expected_disposition),
        role_targets={
            "lower_state": lower_id,
            "upper_state": upper_id,
            "order_relation": edge_id,
            "lower_value": lower_observation,
            "upper_value": upper_observation,
            "monotone_constraint": "constraint.derived",
        },
        observable_expectations={
            "comparable_output_pairs": _observable(
                value_type="comparable_pairs",
                payload={
                    "pairs": [
                        {
                            "lower": _r(lower_value),
                            "upper": _r(upper_value),
                        }
                    ]
                },
                dimension="Scalar",
                unit="unitless",
            ),
            "declared_direction": _observable(
                value_type="direction",
                payload=(
                    {"direction": direction}
                    if direction is not None
                    else None
                ),
                status="inferred" if direction is not None else "unknown",
            ),
        },
        quantity_expectations={
            lower_observation: _quantity(lower_id, lower_value),
            upper_observation: _quantity(upper_id, upper_value),
        },
        relation_expectations={
            edge_id: {
                "type": "PartialOrder",
                "source": lower_id,
                "target": upper_id,
                "order": 0,
            }
        },
    )


def _balance_root(
    *,
    root_id: str,
    root_kind: str,
    before: int,
    after: int,
    transfers: Sequence[tuple[str, int]],
    complete: bool | None,
    transfer_ids: Sequence[str] | None = None,
    expected_disposition: str,
) -> ControlledRoot:
    boundary_id = "node.alpha"
    ledger_id = "node.beta"
    before_observation = "observation.alpha"
    after_observation = "observation.beta"
    ids = tuple(
        transfer_ids
        or tuple(
            f"transfer.{index:02d}"
            for index in range(len(transfers))
        )
    )
    if len(ids) != len(transfers) or len(set(ids)) != len(ids):
        raise ValueError("transfer ids must uniquely cover transfers")
    records = [
        _record(
            "node",
            boundary_id,
            sort="entity",
            boundary=True,
        ),
        _record("node", ledger_id, sort="entity"),
        _record(
            "observation",
            before_observation,
            owner=boundary_id,
            phase="before",
            dimension="Mass",
            unit="kg",
            value=_r(before),
        ),
        _record(
            "observation",
            after_observation,
            owner=boundary_id,
            phase="after",
            dimension="Mass",
            unit="kg",
            value=_r(after),
        ),
    ]
    records.extend(
        _record(
            "transfer",
            transfer_id,
            boundary=boundary_id,
            direction=direction,
            amount=_r(amount),
            dimension="Mass",
            unit="kg",
        )
        for transfer_id, (direction, amount) in zip(ids, transfers)
    )
    if complete is not None:
        records.append(
            _record(
                "assertion",
                "assertion.gamma",
                predicate="scalar",
                target=boundary_id,
                value=complete,
            )
        )
    bundle_members = [
        str(record["id"])
        for record in records
        if not str(record["id"]).endswith(".decoy")
    ]
    records.extend(
        (
            _shared_decoy_map(),
            *_shared_rare_kind_decoys(),
            _record(
                "assertion",
                "assertion.bundle",
                predicate="bundle",
                members=bundle_members,
            ),
            _record(
                "assertion",
                "assertion.association",
                predicate="association",
                members=[boundary_id, ledger_id],
            ),
        )
    )
    grouped: dict[str, list[dict[str, int]]] = {
        "inflow": [],
        "outflow": [],
        "source": [],
        "sink": [],
    }
    for direction, amount in transfers:
        grouped[direction].append(_r(amount))
    for values in grouped.values():
        values.sort(
            key=lambda value: (
                value["numerator"] / value["denominator"],
                value["numerator"],
                value["denominator"],
            )
        )
    return ControlledRoot(
        root_id=root_id,
        law_id=T15,
        root_kind=root_kind,
        records=tuple(records),
        expected_disposition=expected_disposition,
        expected_decision=_decision(expected_disposition),
        role_targets={
            "system_boundary": boundary_id,
            "storage_before": before_observation,
            "storage_after": after_observation,
            "flow_ledger": ledger_id,
            "balance_constraint": "constraint.derived",
        },
        observable_expectations={
            "boundary_declaration": _observable(
                value_type="boundary_declaration",
                payload=(
                    {
                        "boundary_id": "role:system_boundary",
                        "complete": complete,
                    }
                    if complete is not None
                    else None
                ),
                status="inferred" if complete is not None else "unknown",
            ),
            "quantity_ledger": _observable(
                value_type="quantity_ledger",
                payload={
                    "storage_before": _r(before),
                    "storage_after": _r(after),
                    "inflows": grouped["inflow"],
                    "outflows": grouped["outflow"],
                    "sources": grouped["source"],
                    "sinks": grouped["sink"],
                },
                dimension="Mass",
                unit="kg",
            ),
        },
        quantity_expectations={
            before_observation: _quantity(
                boundary_id,
                before,
                dimension="Mass",
                unit="kg",
            ),
            after_observation: _quantity(
                boundary_id,
                after,
                dimension="Mass",
                unit="kg",
            ),
        },
        relation_expectations={},
    )


def _path_root(
    *,
    root_id: str,
    root_kind: str,
    domain_size: int,
    first_override: Mapping[str, str] | None = None,
    second_override: Mapping[str, str] | None = None,
    direct_override: Mapping[str, str] | None = None,
    direct_available: bool = True,
    expected_disposition: str,
) -> ControlledRoot:
    if domain_size < 2:
        raise ValueError("domain size must be at least two")
    source_id = "node.alpha"
    target_id = "node.beta"
    chained_id = "node.gamma"
    direct_id = "node.delta"
    domain = [source_id] + [
        f"local:source_{index:02d}"
        for index in range(1, domain_size)
    ]
    first = {source_id: "local:middle_00"}
    second = {"local:middle_00": target_id}
    direct = {source_id: target_id}
    for index in range(1, domain_size):
        first[f"local:source_{index:02d}"] = (
            f"local:middle_{index:02d}"
        )
        second[f"local:middle_{index:02d}"] = (
            f"local:target_{index:02d}"
        )
        direct[f"local:source_{index:02d}"] = (
            f"local:target_{index:02d}"
        )
    first.update(first_override or {})
    second.update(second_override or {})
    direct.update(direct_override or {})
    records = [
        _record(
            "node",
            source_id,
            sort="entity",
        ),
        _record(
            "node",
            target_id,
            sort="entity",
        ),
        _record(
            "node",
            chained_id,
            sort="entity",
            ordinal=0,
        ),
        _record(
            "node",
            direct_id,
            sort="entity",
            ordinal=1,
        ),
        _record(
            "assertion",
            "assertion.gamma",
            predicate="set",
            values=domain,
        ),
        _record(
            "map",
            "map.alpha",
            stage="step_a",
            mode="finite",
            rows=_map_rows(first),
        ),
        _record(
            "map",
            "map.beta",
            stage="step_b",
            mode="finite",
            rows=_map_rows(second),
        ),
    ]
    if direct_available:
        records.append(
            _record(
                "map",
                "map.gamma",
                stage="reference",
                mode="finite",
                rows=_map_rows(direct),
            )
        )
    bundle_members = [
        str(record["id"]) for record in records
    ]
    records.extend(
        (
            _shared_decoy_map(),
            *_shared_rare_kind_decoys(),
            _record(
                "assertion",
                "assertion.bundle",
                predicate="bundle",
                members=bundle_members,
            ),
            _record(
                "assertion",
                "assertion.association",
                predicate="association",
                members=[source_id, target_id],
            ),
        )
    )

    def ref(value: str) -> str:
        if value == source_id:
            return "role:source_state"
        if value == target_id:
            return "role:target_state"
        return value

    def map_payload(rows: Mapping[str, str]) -> dict[str, Any]:
        return {
            "rows": sorted(
                (
                    {"source": ref(source), "target": ref(target)}
                    for source, target in rows.items()
                ),
                key=lambda row: (row["source"], row["target"]),
            )
        }

    return ControlledRoot(
        root_id=root_id,
        law_id=T09,
        root_kind=root_kind,
        records=tuple(records),
        expected_disposition=expected_disposition,
        expected_decision=_decision(expected_disposition),
        role_targets={
            "source_state": source_id,
            "target_state": target_id,
            "composed_path": chained_id,
            "direct_path": direct_id,
            "path_constraint": "constraint.derived",
        },
        observable_expectations={
            "finite_domain": _observable(
                value_type="finite_domain",
                payload={
                    "values": sorted(ref(value) for value in domain)
                },
            ),
            "first_map": _observable(
                value_type="finite_map",
                payload=map_payload(first),
            ),
            "second_map": _observable(
                value_type="finite_map",
                payload=map_payload(second),
            ),
            "direct_map": _observable(
                value_type="finite_map",
                payload=(
                    map_payload(direct) if direct_available else None
                ),
                status="inferred" if direct_available else "unknown",
            ),
        },
        quantity_expectations={},
        relation_expectations={},
    )


def _subset_utility(
    components: Sequence[str],
    subset: Sequence[str],
    pair: Sequence[str],
    *,
    pair_coefficient: int,
    third_order_coefficient: int,
) -> int:
    component_tuple = tuple(components)
    result = sum(
        index + 1
        for index, component in enumerate(component_tuple)
        if component in subset
    )
    if set(pair) <= set(subset):
        result += pair_coefficient
    if set(component_tuple) <= set(subset):
        result += third_order_coefficient
    return result


def _interaction_root(
    *,
    root_id: str,
    root_kind: str,
    pair_coefficient: int,
    third_order_coefficient: int,
    fold_count: int,
    folds_complete: bool,
    expected_disposition: str,
) -> ControlledRoot:
    components = ("node.alpha", "node.beta", "node.gamma")
    ledger_id = "node.delta"
    pair = components[:2]
    records = [
        *(
            _record(
                "node",
                component,
                sort="entity",
                ordinal=index,
            )
            for index, component in enumerate(components)
        ),
        _record("node", ledger_id, sort="entity"),
        _record(
            "assertion",
            "assertion.gamma",
            predicate="set",
            values=list(pair),
        ),
        _record(
            "assertion",
            "assertion.delta",
            predicate="property",
            value="positive_joint",
        ),
        _shared_decoy_map(),
        *_shared_rare_kind_decoys(),
    ]
    normalized_folds: list[dict[str, Any]] = []
    for fold in range(fold_count):
        normalized_rows = []
        for mask in range(1 << len(components)):
            subset = [
                component
                for index, component in enumerate(components)
                if mask & (1 << index)
            ]
            if not folds_complete and mask == (1 << len(components)) - 1:
                continue
            utility = _subset_utility(
                components,
                subset,
                pair,
                pair_coefficient=pair_coefficient,
                third_order_coefficient=third_order_coefficient,
            )
            records.append(
                _record(
                    "subset_outcome",
                    f"subset.{fold:02d}.{mask:02d}",
                    fold=fold,
                    subset=subset,
                    utility=_r(utility),
                    dimension="Utility",
                    unit="unitless",
                )
            )
            reverse = {
                components[0]: "role:component_a",
                components[1]: "role:component_b",
                components[2]: "role:component_c",
            }
            normalized_rows.append(
                {
                    "subset": sorted(reverse[value] for value in subset),
                    "utility": _r(utility),
                }
            )
        normalized_rows.sort(
            key=lambda row: (
                len(row["subset"]),
                tuple(row["subset"]),
            )
        )
        normalized_folds.append({"rows": normalized_rows})
    bundle_members = [
        str(record["id"])
        for record in records
        if not str(record["id"]).endswith(".decoy")
        and record["kind"] != "subset_outcome"
    ]
    records.extend(
        (
            _record(
                "assertion",
                "assertion.bundle",
                predicate="bundle",
                members=bundle_members,
            ),
            _record(
                "assertion",
                "assertion.association",
                predicate="association",
                members=list(components),
            ),
        )
    )
    normalized_folds.sort(key=strict_content_hash)
    return ControlledRoot(
        root_id=root_id,
        law_id=T05,
        root_kind=root_kind,
        records=tuple(records),
        expected_disposition=expected_disposition,
        expected_decision=_decision(expected_disposition),
        role_targets={
            "component_a": components[0],
            "component_b": components[1],
            "component_c": components[2],
            "utility_ledger": ledger_id,
            "interaction_constraint": "constraint.derived",
        },
        observable_expectations={
            "components": _observable(
                value_type="component_set",
                payload={
                    "values": [
                        "role:component_a",
                        "role:component_b",
                        "role:component_c",
                    ]
                },
            ),
            "designated_pair": _observable(
                value_type="designated_pair",
                payload={
                    "values": [
                        "role:component_a",
                        "role:component_b",
                    ]
                },
            ),
            "held_fold_utilities": _observable(
                value_type="subset_utility_folds",
                payload=(
                    {"folds": normalized_folds}
                    if folds_complete
                    else None
                ),
                status="inferred" if folds_complete else "unknown",
                dimension="Utility",
                unit="unitless",
            ),
            "interaction_expectation": _observable(
                value_type="interaction_expectation",
                payload={"value": "complementary"},
            ),
        },
        quantity_expectations={},
        relation_expectations={},
    )


def _records_by_id(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result = {
        str(record["id"]): copy.deepcopy(dict(record))
        for record in records
    }
    if len(result) != len(records):
        raise AssertionError("operator input record ids are not unique")
    return result


def _ordered_records(
    original: Sequence[Mapping[str, Any]],
    rows: Mapping[str, Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    original_ids = [str(record["id"]) for record in original]
    result = [
        copy.deepcopy(rows[record_id])
        for record_id in original_ids
        if record_id in rows
    ]
    result.extend(
        copy.deepcopy(rows[record_id])
        for record_id in sorted(set(rows) - set(original_ids))
    )
    return tuple(result)


def _diff_values(left: Any, right: Any, path: str) -> list[str]:
    if type(left) is not type(right):
        return [path]
    if isinstance(left, dict):
        paths: list[str] = []
        for key in sorted(set(left) | set(right)):
            child = f"{path}/{key}"
            if key not in left or key not in right:
                paths.append(child)
            else:
                paths.extend(_diff_values(left[key], right[key], child))
        return paths
    if isinstance(left, list):
        if len(left) != len(right):
            return [path]
        paths = []
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            paths.extend(
                _diff_values(
                    left_item, right_item, f"{path}/{index}"
                )
            )
        return paths
    return [] if left == right else [path]


def atomic_record_diff_paths(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    left_rows = {str(row["id"]): row for row in left}
    right_rows = {str(row["id"]): row for row in right}
    paths: list[str] = []
    for record_id in sorted(set(left_rows) | set(right_rows)):
        root = f"/records/{record_id}"
        if record_id not in left_rows or record_id not in right_rows:
            paths.append(root)
        else:
            paths.extend(
                _diff_values(
                    left_rows[record_id],
                    right_rows[record_id],
                    root,
                )
            )
    return tuple(sorted(set(paths)))


Operator = Callable[
    [Sequence[Mapping[str, Any]]],
    tuple[tuple[Mapping[str, Any], ...], tuple[str, ...]],
]


def _mutate_attr(
    records: Sequence[Mapping[str, Any]],
    record_id: str,
    attr: str,
    value: Any,
) -> tuple[Mapping[str, Any], ...]:
    rows = _records_by_id(records)
    rows[record_id]["attrs"][attr] = copy.deepcopy(value)
    return _ordered_records(records, rows)


def _op_t14_coordinate_sign(
    records: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Mapping[str, Any], ...], tuple[str, ...]]:
    output = _mutate_attr(records, "map.beta", "signs", [1])
    return output, ("/records/map.beta/attrs/signs/0",)


def _op_t14_post_value_sign(
    records: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Mapping[str, Any], ...], tuple[str, ...]]:
    rows = _records_by_id(records)
    value = rows["observation.beta"]["attrs"]["value"]
    value["numerator"] = -value["numerator"]
    return _ordered_records(records, rows), (
        "/records/observation.beta/attrs/value/numerator",
    )


def _op_t17_direction_sign(
    records: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Mapping[str, Any], ...], tuple[str, ...]]:
    rows = _records_by_id(records)
    attrs = rows["assertion.gamma"]["attrs"]
    attrs["value"] = -attrs["value"]
    return _ordered_records(records, rows), (
        "/records/assertion.gamma/attrs/value",
    )


def _op_t17_upper_value(
    records: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Mapping[str, Any], ...], tuple[str, ...]]:
    output = _mutate_attr(
        records, "observation.beta", "value", _r(5)
    )
    return output, (
        "/records/observation.beta/attrs/value/numerator",
    )


def _op_t15_delete_inflow(
    records: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Mapping[str, Any], ...], tuple[str, ...]]:
    rows = _records_by_id(records)
    del rows["transfer.00"]
    rows["assertion.bundle"]["attrs"]["members"].remove("transfer.00")
    return _ordered_records(records, rows), (
        "/records/assertion.bundle/attrs/members",
        "/records/transfer.00",
    )


def _op_t15_invert_transfer(
    records: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Mapping[str, Any], ...], tuple[str, ...]]:
    output = _mutate_attr(
        records, "transfer.00", "direction", "outflow"
    )
    return output, (
        "/records/transfer.00/attrs/direction",
    )


def _replace_map_target(
    records: Sequence[Mapping[str, Any]],
    *,
    record_id: str,
    source: str,
    target: str,
) -> tuple[tuple[Mapping[str, Any], ...], tuple[str, ...]]:
    rows = _records_by_id(records)
    map_rows = rows[record_id]["attrs"]["rows"]
    matches = [
        index
        for index, row in enumerate(map_rows)
        if row["source"] == source
    ]
    if len(matches) != 1:
        raise AssertionError("operator map source is not unique")
    index = matches[0]
    map_rows[index]["target"] = target
    return _ordered_records(records, rows), (
        f"/records/{record_id}/attrs/rows/{index}/target",
    )


def _op_t09_second_target(
    records: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Mapping[str, Any], ...], tuple[str, ...]]:
    return _replace_map_target(
        records,
        record_id="map.beta",
        source="local:middle_01",
        target="local:alternate_01",
    )


def _op_t09_direct_target(
    records: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Mapping[str, Any], ...], tuple[str, ...]]:
    return _replace_map_target(
        records,
        record_id="map.gamma",
        source="local:source_01",
        target="local:alternate_01",
    )


def _op_t05_pair_sign(
    records: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Mapping[str, Any], ...], tuple[str, ...]]:
    rows = _records_by_id(records)
    changed: list[str] = []
    for record_id, row in sorted(rows.items()):
        if row["kind"] != "subset_outcome":
            continue
        subset = set(row["attrs"]["subset"])
        if {"node.alpha", "node.beta"} <= subset:
            row["attrs"]["utility"]["numerator"] -= 2
            changed.append(
                f"/records/{record_id}/attrs/utility/numerator"
            )
    return _ordered_records(records, rows), tuple(changed)


def _op_t05_triple_term(
    records: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Mapping[str, Any], ...], tuple[str, ...]]:
    rows = _records_by_id(records)
    changed: list[str] = []
    for record_id, row in sorted(rows.items()):
        if (
            row["kind"] == "subset_outcome"
            and set(row["attrs"]["subset"])
            == {"node.alpha", "node.beta", "node.gamma"}
        ):
            row["attrs"]["utility"]["numerator"] += 1
            changed.append(
                f"/records/{record_id}/attrs/utility/numerator"
            )
    return _ordered_records(records, rows), tuple(changed)


def _op_t14_missing_post(
    records: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Mapping[str, Any], ...], tuple[str, ...]]:
    rows = _records_by_id(records)
    del rows["observation.beta"]["attrs"]["value"]
    return _ordered_records(records, rows), (
        "/records/observation.beta/attrs/value",
    )


def _op_t17_missing_direction(
    records: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Mapping[str, Any], ...], tuple[str, ...]]:
    rows = _records_by_id(records)
    del rows["assertion.gamma"]
    rows["assertion.bundle"]["attrs"]["members"].remove(
        "assertion.gamma"
    )
    return _ordered_records(records, rows), (
        "/records/assertion.bundle/attrs/members",
        "/records/assertion.gamma",
    )


def _op_t15_incomplete_scope(
    records: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Mapping[str, Any], ...], tuple[str, ...]]:
    rows = _records_by_id(records)
    del rows["assertion.gamma"]
    rows["assertion.bundle"]["attrs"]["members"].remove(
        "assertion.gamma"
    )
    return _ordered_records(records, rows), (
        "/records/assertion.bundle/attrs/members",
        "/records/assertion.gamma",
    )


def _op_t09_missing_direct(
    records: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Mapping[str, Any], ...], tuple[str, ...]]:
    rows = _records_by_id(records)
    del rows["map.gamma"]
    rows["assertion.bundle"]["attrs"]["members"].remove("map.gamma")
    return _ordered_records(records, rows), (
        "/records/assertion.bundle/attrs/members",
        "/records/map.gamma",
    )


def _op_t05_missing_subsets(
    records: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Mapping[str, Any], ...], tuple[str, ...]]:
    rows = _records_by_id(records)
    deleted = []
    for record_id, row in tuple(sorted(rows.items())):
        if (
            row["kind"] == "subset_outcome"
            and set(row["attrs"]["subset"])
            == {"node.alpha", "node.beta", "node.gamma"}
        ):
            deleted.append(record_id)
            del rows[record_id]
    allowed = [f"/records/{record_id}" for record_id in deleted]
    return _ordered_records(records, rows), tuple(sorted(allowed))


_OPERATORS: Mapping[str, Operator] = {
    "t14.coordinate_sign_identity_substitution": (
        _op_t14_coordinate_sign
    ),
    "t14.post_observation_sign_inversion": _op_t14_post_value_sign,
    "t17.declared_direction_sign_inversion": _op_t17_direction_sign,
    "t17.upper_observation_order_violation": _op_t17_upper_value,
    "t15.first_inflow_deletion": _op_t15_delete_inflow,
    "t15.transfer_direction_inversion": _op_t15_invert_transfer,
    "t09.second_map_target_substitution": _op_t09_second_target,
    "t09.direct_map_target_substitution": _op_t09_direct_target,
    "t05.designated_pair_interaction_sign_inversion": (
        _op_t05_pair_sign
    ),
    "t05.triple_subset_term_injection": _op_t05_triple_term,
    "t14.post_observation_missingness": _op_t14_missing_post,
    "t17.direction_missingness": _op_t17_missing_direction,
    "t15.scope_coverage_missingness": _op_t15_incomplete_scope,
    "t09.direct_map_missingness": _op_t09_missing_direct,
    "t05.subset_outcome_missingness": _op_t05_missing_subsets,
}
FROZEN_OPERATOR_IDS = tuple(sorted(_OPERATORS))
_OPERATOR_TARGET_SPECS: Mapping[str, tuple[str, ...]] = {
    "t05.designated_pair_interaction_sign_inversion": (
        "/records/subset.00.03/attrs/utility/numerator",
        "/records/subset.00.07/attrs/utility/numerator",
        "/records/subset.01.03/attrs/utility/numerator",
        "/records/subset.01.07/attrs/utility/numerator",
    ),
    "t05.subset_outcome_missingness": (
        "/records/subset.00.07",
        "/records/subset.01.07",
    ),
    "t05.triple_subset_term_injection": (
        "/records/subset.00.07/attrs/utility/numerator",
        "/records/subset.01.07/attrs/utility/numerator",
        "/records/subset.02.07/attrs/utility/numerator",
    ),
    "t09.direct_map_missingness": (
        "/records/assertion.bundle/attrs/members",
        "/records/map.gamma",
    ),
    "t09.direct_map_target_substitution": (
        "/records/map.gamma/attrs/rows/0/target",
    ),
    "t09.second_map_target_substitution": (
        "/records/map.beta/attrs/rows/1/target",
    ),
    "t14.coordinate_sign_identity_substitution": (
        "/records/map.beta/attrs/signs/0",
    ),
    "t14.post_observation_missingness": (
        "/records/observation.beta/attrs/value",
    ),
    "t14.post_observation_sign_inversion": (
        "/records/observation.beta/attrs/value/numerator",
    ),
    "t15.first_inflow_deletion": (
        "/records/assertion.bundle/attrs/members",
        "/records/transfer.00",
    ),
    "t15.scope_coverage_missingness": (
        "/records/assertion.bundle/attrs/members",
        "/records/assertion.gamma",
    ),
    "t15.transfer_direction_inversion": (
        "/records/transfer.00/attrs/direction",
    ),
    "t17.declared_direction_sign_inversion": (
        "/records/assertion.gamma/attrs/value",
    ),
    "t17.direction_missingness": (
        "/records/assertion.bundle/attrs/members",
        "/records/assertion.gamma",
    ),
    "t17.upper_observation_order_violation": (
        "/records/observation.beta/attrs/value/numerator",
    ),
}
_OPERATOR_IMPLEMENTATION_SHA256: Mapping[str, str] = {
    "t05.designated_pair_interaction_sign_inversion": "e49397450a0ff3b4759a1104b71dcf48e8c9f5881b5d9dcac6a6864d353d2965",
    "t05.subset_outcome_missingness": "90b5df36c4a767b6b7f60a77bab6b1031be6f1b89051eed3ce6c03266aa778f1",
    "t05.triple_subset_term_injection": "1a933f145f22e2c79344cb0b064c8139fecbc1995585c459db140a78d67b07e0",
    "t09.direct_map_missingness": "b334cf107ded0d4f162e4ebf5bff889f49ffae67ac86d1dae0006dd47d6aaebc",
    "t09.direct_map_target_substitution": "c305368d9b6c6d0e2f8bee4b22d3a0c1490abd92fc1932df7815edf4fcf43a1c",
    "t09.second_map_target_substitution": "fa54ced35f7b271095fd7be3120493ca375f8e0c1440404b1dd22df9b818f3ca",
    "t14.coordinate_sign_identity_substitution": "481e017e69e9305bb34f83ed8b67be5d3bd55fde22134f5a98f9656350f8085c",
    "t14.post_observation_missingness": "933f92352a4606901bbf44dff371ab8f2550b00231f582a951daad5a4394b948",
    "t14.post_observation_sign_inversion": "77732db9d2018457755fa320846a6e32ef5d714c42a9aac89131fa3f2b94086a",
    "t15.first_inflow_deletion": "468f03eb5628c0eace0305e12a7c326880c75c26e6034e3f9b524b2580faf07b",
    "t15.scope_coverage_missingness": "df174859d31c18cda9d65b736e13332f0cd714b556c051c116475e0f0633e85b",
    "t15.transfer_direction_inversion": "aaa00c9eb1a1b33a46868b5af22a3528e7e666b1a9f329431f8ac43e41028c4b",
    "t17.declared_direction_sign_inversion": "03ff5db8c28e0218e28b53b8ca72b9a9b4b41f1e21037d4a14153451e452ad52",
    "t17.direction_missingness": "9fd79b405bb058a5e4ba76d55aad1b73e2ea68ccf953397eea12eb4b551bd26a",
    "t17.upper_observation_order_violation": "3a7325f3a73b505a0ba857fc41e6cbc55f1076a26cfa58dfadb4ed8fa3c28ad3",
}


def _operator_implementation_sha256(operator_id: str) -> str:
    source = inspect.getsource(_OPERATORS[operator_id]).encode("utf-8")
    return hashlib.sha256(source).hexdigest()


OPERATOR_CONTRACT_HASH = strict_content_hash(
    {
        "operator_version": OPERATOR_VERSION,
        "operator_ids": list(FROZEN_OPERATOR_IDS),
        "target_specs": {
            key: list(value)
            for key, value in sorted(_OPERATOR_TARGET_SPECS.items())
        },
        "implementation_sha256": dict(
            sorted(_OPERATOR_IMPLEMENTATION_SHA256.items())
        ),
        "diff_semantics": "record_id_addressed_exact_leaf_or_collection_v1",
        "input_mutation": False,
        "output_order": "preserve_input_then_sorted_additions",
    }
)


def apply_frozen_operator(
    operator_id: str,
    records: Sequence[Mapping[str, Any]],
) -> tuple[
    tuple[Mapping[str, Any], ...], CounterfactualOperatorReceipt
]:
    """Apply one audited pure transform and prove its exact diff."""

    if operator_id not in _OPERATORS:
        raise KeyError(f"unknown frozen operator: {operator_id}")
    input_copy = copy.deepcopy(tuple(records))
    input_hash = strict_content_hash(list(input_copy))
    actual_implementation_hash = _operator_implementation_sha256(
        operator_id
    )
    if (
        actual_implementation_hash
        != _OPERATOR_IMPLEMENTATION_SHA256[operator_id]
    ):
        raise AssertionError(
            f"operator implementation hash drifted: {operator_id}"
        )
    output, _operator_reported_paths = _OPERATORS[operator_id](input_copy)
    if strict_content_hash(list(records)) != input_hash:
        raise AssertionError("operator mutated caller input")
    observed = atomic_record_diff_paths(records, output)
    normalized_allowed = tuple(
        sorted(_OPERATOR_TARGET_SPECS[operator_id])
    )
    unchanged = observed == normalized_allowed
    if not unchanged:
        raise AssertionError(
            "operator diff escaped frozen target: "
            f"{operator_id}: expected={normalized_allowed}, "
            f"observed={observed}"
        )
    receipt = CounterfactualOperatorReceipt(
        operator_id=operator_id,
        operator_version=OPERATOR_VERSION,
        operator_implementation_sha256=actual_implementation_hash,
        frozen_target_spec_hash=strict_content_hash(
            list(normalized_allowed)
        ),
        input_records_sha256=input_hash,
        output_records_sha256=strict_content_hash(list(output)),
        allowed_diff_paths=normalized_allowed,
        observed_diff_paths=observed,
        non_target_fields_unchanged=True,
    )
    return output, receipt


def _paired_root(
    primary: ControlledRoot,
    template: ControlledRoot,
    *,
    pair_id: str,
    pair_role: str,
    operator_id: str,
) -> ControlledRoot:
    output, receipt = apply_frozen_operator(
        operator_id, primary.records
    )
    output_by_id = {
        str(record["id"]): record for record in output
    }
    template_by_id = {
        str(record["id"]): record for record in template.records
    }
    if strict_content_hash(output_by_id) != strict_content_hash(
        template_by_id
    ):
        raise AssertionError(
            f"operator/template mismatch for {pair_id}"
        )
    return replace(
        template,
        records=output,
        hard_negative_operator_id=(
            operator_id
            if pair_role == "counterfactual_negative"
            else None
        ),
        pair_id=pair_id,
        paired_primary_root_id=primary.root_id,
        pair_role=pair_role,
        operator_receipt=receipt,
    )


def build_controlled_roots() -> tuple[ControlledRoot, ...]:
    """Build ten primaries, ten paired negatives, and five controls."""

    p14a = _equivariance_root(
        root_id="root.t14.primary_a",
        root_kind="primary_a",
        magnitude=2,
        signs=(-1,),
        after_value=-2,
        expected_disposition="satisfied",
    )
    p14b = _equivariance_root(
        root_id="root.t14.primary_b",
        root_kind="primary_b",
        magnitude=7,
        signs=(-1,),
        after_value=-7,
        expected_disposition="satisfied",
    )
    n14a = _paired_root(
        p14a,
        _equivariance_root(
            root_id="root.t14.negative_a",
            root_kind="hard_negative_a",
            magnitude=2,
            signs=(1,),
            after_value=-2,
            expected_disposition="violated",
        ),
        pair_id="pair.t14.negative_a",
        pair_role="counterfactual_negative",
        operator_id="t14.coordinate_sign_identity_substitution",
    )
    n14b = _paired_root(
        p14b,
        _equivariance_root(
            root_id="root.t14.negative_b",
            root_kind="hard_negative_b",
            magnitude=7,
            signs=(-1,),
            after_value=7,
            expected_disposition="violated",
        ),
        pair_id="pair.t14.negative_b",
        pair_role="counterfactual_negative",
        operator_id="t14.post_observation_sign_inversion",
    )
    c14 = _paired_root(
        p14a,
        _equivariance_root(
            root_id="root.t14.control",
            root_kind="missingness_control",
            magnitude=2,
            signs=(-1,),
            after_value=-2,
            after_available=False,
            expected_disposition="inconclusive",
        ),
        pair_id="pair.t14.control",
        pair_role="missingness_control",
        operator_id="t14.post_observation_missingness",
    )

    p17a = _monotone_root(
        root_id="root.t17.primary_a",
        root_kind="primary_a",
        lower_value=2,
        upper_value=5,
        direction=1,
        expected_disposition="satisfied",
    )
    p17b = _monotone_root(
        root_id="root.t17.primary_b",
        root_kind="primary_b",
        lower_value=7,
        upper_value=11,
        direction=1,
        expected_disposition="satisfied",
    )
    n17a = _paired_root(
        p17a,
        _monotone_root(
            root_id="root.t17.negative_a",
            root_kind="hard_negative_a",
            lower_value=2,
            upper_value=5,
            direction=-1,
            expected_disposition="violated",
        ),
        pair_id="pair.t17.negative_a",
        pair_role="counterfactual_negative",
        operator_id="t17.declared_direction_sign_inversion",
    )
    n17b = _paired_root(
        p17b,
        _monotone_root(
            root_id="root.t17.negative_b",
            root_kind="hard_negative_b",
            lower_value=7,
            upper_value=5,
            direction=1,
            expected_disposition="violated",
        ),
        pair_id="pair.t17.negative_b",
        pair_role="counterfactual_negative",
        operator_id="t17.upper_observation_order_violation",
    )
    c17 = _paired_root(
        p17a,
        _monotone_root(
            root_id="root.t17.control",
            root_kind="missingness_control",
            lower_value=2,
            upper_value=5,
            direction=None,
            expected_disposition="inconclusive",
        ),
        pair_id="pair.t17.control",
        pair_role="missingness_control",
        operator_id="t17.direction_missingness",
    )

    p15a = _balance_root(
        root_id="root.t15.primary_a",
        root_kind="primary_a",
        before=10,
        after=13,
        transfers=(("inflow", 5), ("outflow", 2)),
        complete=True,
        expected_disposition="satisfied",
    )
    p15b = _balance_root(
        root_id="root.t15.primary_b",
        root_kind="primary_b",
        before=20,
        after=26,
        transfers=(("inflow", 9), ("outflow", 3)),
        complete=True,
        expected_disposition="satisfied",
    )
    n15a = _paired_root(
        p15a,
        _balance_root(
            root_id="root.t15.negative_a",
            root_kind="hard_negative_a",
            before=10,
            after=13,
            transfers=(("outflow", 2),),
            transfer_ids=("transfer.01",),
            complete=True,
            expected_disposition="violated",
        ),
        pair_id="pair.t15.negative_a",
        pair_role="counterfactual_negative",
        operator_id="t15.first_inflow_deletion",
    )
    n15b = _paired_root(
        p15b,
        _balance_root(
            root_id="root.t15.negative_b",
            root_kind="hard_negative_b",
            before=20,
            after=26,
            transfers=(("outflow", 9), ("outflow", 3)),
            complete=True,
            expected_disposition="violated",
        ),
        pair_id="pair.t15.negative_b",
        pair_role="counterfactual_negative",
        operator_id="t15.transfer_direction_inversion",
    )
    c15 = _paired_root(
        p15a,
        _balance_root(
            root_id="root.t15.control",
            root_kind="missingness_control",
            before=10,
            after=13,
            transfers=(("inflow", 5), ("outflow", 2)),
            complete=None,
            expected_disposition="inconclusive",
        ),
        pair_id="pair.t15.control",
        pair_role="missingness_control",
        operator_id="t15.scope_coverage_missingness",
    )

    p09a = _path_root(
        root_id="root.t09.primary_a",
        root_kind="primary_a",
        domain_size=2,
        expected_disposition="satisfied",
    )
    p09b = _path_root(
        root_id="root.t09.primary_b",
        root_kind="primary_b",
        domain_size=3,
        expected_disposition="satisfied",
    )
    n09a = _paired_root(
        p09a,
        _path_root(
            root_id="root.t09.negative_a",
            root_kind="hard_negative_a",
            domain_size=2,
            second_override={
                "local:middle_01": "local:alternate_01"
            },
            expected_disposition="violated",
        ),
        pair_id="pair.t09.negative_a",
        pair_role="counterfactual_negative",
        operator_id="t09.second_map_target_substitution",
    )
    n09b = _paired_root(
        p09b,
        _path_root(
            root_id="root.t09.negative_b",
            root_kind="hard_negative_b",
            domain_size=3,
            direct_override={
                "local:source_01": "local:alternate_01"
            },
            expected_disposition="violated",
        ),
        pair_id="pair.t09.negative_b",
        pair_role="counterfactual_negative",
        operator_id="t09.direct_map_target_substitution",
    )
    c09 = _paired_root(
        p09a,
        _path_root(
            root_id="root.t09.control",
            root_kind="missingness_control",
            domain_size=2,
            direct_available=False,
            expected_disposition="inconclusive",
        ),
        pair_id="pair.t09.control",
        pair_role="missingness_control",
        operator_id="t09.direct_map_missingness",
    )

    p05a = _interaction_root(
        root_id="root.t05.primary_a",
        root_kind="primary_a",
        pair_coefficient=1,
        third_order_coefficient=0,
        fold_count=2,
        folds_complete=True,
        expected_disposition="satisfied",
    )
    p05b = _interaction_root(
        root_id="root.t05.primary_b",
        root_kind="primary_b",
        pair_coefficient=2,
        third_order_coefficient=0,
        fold_count=3,
        folds_complete=True,
        expected_disposition="satisfied",
    )
    n05a = _paired_root(
        p05a,
        _interaction_root(
            root_id="root.t05.negative_a",
            root_kind="hard_negative_a",
            pair_coefficient=-1,
            third_order_coefficient=0,
            fold_count=2,
            folds_complete=True,
            expected_disposition="violated",
        ),
        pair_id="pair.t05.negative_a",
        pair_role="counterfactual_negative",
        operator_id=(
            "t05.designated_pair_interaction_sign_inversion"
        ),
    )
    n05b = _paired_root(
        p05b,
        _interaction_root(
            root_id="root.t05.negative_b",
            root_kind="hard_negative_b",
            pair_coefficient=2,
            third_order_coefficient=1,
            fold_count=3,
            folds_complete=True,
            expected_disposition="violated",
        ),
        pair_id="pair.t05.negative_b",
        pair_role="counterfactual_negative",
        operator_id="t05.triple_subset_term_injection",
    )
    c05 = _paired_root(
        p05a,
        _interaction_root(
            root_id="root.t05.control",
            root_kind="missingness_control",
            pair_coefficient=1,
            third_order_coefficient=0,
            fold_count=2,
            folds_complete=False,
            expected_disposition="inconclusive",
        ),
        pair_id="pair.t05.control",
        pair_role="missingness_control",
        operator_id="t05.subset_outcome_missingness",
    )

    roots = (
        p14a,
        p14b,
        n14a,
        n14b,
        c14,
        p17a,
        p17b,
        n17a,
        n17b,
        c17,
        p15a,
        p15b,
        n15a,
        n15b,
        c15,
        p09a,
        p09b,
        n09a,
        n09b,
        c09,
        p05a,
        p05b,
        n05a,
        n05b,
        c05,
    )
    if len(roots) != ROOT_COUNT:
        raise AssertionError("paired case count drifted")
    if len({root.root_id for root in roots}) != ROOT_COUNT:
        raise AssertionError("controlled root ids are not unique")
    if (
        sum(root.pair_role == "primary" for root in roots)
        != PRIMARY_ROOT_COUNT
        or sum(
            root.pair_role == "counterfactual_negative"
            for root in roots
        )
        != PAIRED_NEGATIVE_ROOT_COUNT
        or sum(
            root.pair_role == "missingness_control"
            for root in roots
        )
        != PAIRED_CONTROL_ROOT_COUNT
    ):
        raise AssertionError("paired case composition drifted")
    return roots


def _alias_value(value: Any) -> Any:
    if isinstance(value, str):
        return _VALUE_ALIAS.get(value, value)
    if isinstance(value, list):
        return [_alias_value(item) for item in value]
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            aliased_key = _KEY_ALIAS.get(key, key)
            aliased_value = _alias_value(item)
            if (
                key in _REVERSE_ALIAS_LIST_KEYS
                and isinstance(aliased_value, list)
            ):
                aliased_value = list(reversed(aliased_value))
            result[aliased_key] = aliased_value
        return result
    return value


def _render_record(
    record: Mapping[str, Any], view_kind: str
) -> bytes:
    kind = str(record["kind"])
    record_id = str(record["id"])
    attrs = record["attrs"]
    if view_kind == "json_canonical":
        value = {"kind": kind, "id": record_id, "attrs": attrs}
        text = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    elif view_kind == "json_alias":
        value = {
            "record": _KIND_ALIAS[kind],
            "name": record_id,
            "fields": _alias_value(attrs),
        }
        text = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=False,
            separators=(",", ":"),
        )
    elif view_kind == "line_canonical":
        attrs_text = json.dumps(
            attrs,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        text = f"Fact {kind} {record_id} carries {attrs_text}."
    elif view_kind == "line_alias":
        attrs_text = json.dumps(
            _alias_value(attrs),
            ensure_ascii=True,
            sort_keys=False,
            separators=(",", ":"),
        )
        text = (
            f"Record {_KIND_ALIAS[kind]} named {record_id} "
            f"has {attrs_text}."
        )
    else:
        raise KeyError(f"unknown view kind: {view_kind}")
    return text.encode("utf-8")


def _render_view(
    root: ControlledRoot, view_kind: str
) -> tuple[ControlledView, ControlledGoldLink]:
    records = list(root.records)
    if view_kind.endswith("_alias"):
        records.reverse()
    lines = [_render_record(record, view_kind) for record in records]
    source_bytes = b"\n".join(lines) + b"\n"
    spans: dict[str, tuple[int, int]] = {}
    cursor = 0
    for record, line in zip(records, lines):
        spans[str(record["id"])] = (cursor, cursor + len(line))
        cursor += len(line) + 1
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    media_type = (
        JSONL_MEDIA_TYPE
        if view_kind.startswith("json_")
        else PROSE_MEDIA_TYPE
    )
    item_id = "item." + hashlib.sha256(
        media_type.encode("utf-8") + b"\0" + source_bytes
    ).hexdigest()
    view = ControlledView(
        item_id=item_id,
        media_type=media_type,
        source_bytes=source_bytes,
        source_sha256=source_sha256,
        record_spans=spans,
    )
    return view, ControlledGoldLink(
        item_id=item_id,
        root_id=root.root_id,
        view_kind=view_kind,
    )


def render_controlled_views(
    roots: Sequence[ControlledRoot] | None = None,
) -> tuple[ControlledView, ...]:
    actual_roots = tuple(roots or build_controlled_roots())
    bundles = tuple(
        _render_view(root, view_kind)
        for root in actual_roots
        for view_kind in (
            "json_canonical",
            "line_canonical",
            "json_alias",
            "line_alias",
        )
    )
    views = tuple(view for view, _link in bundles)
    expected = len(actual_roots) * VIEWS_PER_ROOT
    if len(views) != expected:
        raise AssertionError("controlled view count drifted")
    if len({view.view_id for view in views}) != expected:
        raise AssertionError("controlled view ids are not unique")
    return views


def controlled_view_gold_linkage(
    roots: Sequence[ControlledRoot] | None = None,
) -> tuple[ControlledGoldLink, ...]:
    """Return the private item-to-gold map outside the raw envelope."""

    actual_roots = tuple(roots or build_controlled_roots())
    links = tuple(
        _render_view(root, view_kind)[1]
        for root in actual_roots
        for view_kind in (
            "json_canonical",
            "line_canonical",
            "json_alias",
            "line_alias",
        )
    )
    if len({row.item_id for row in links}) != len(links):
        raise AssertionError("controlled gold item ids are not unique")
    return links


def raw_pack_contract(
    views: Sequence[ControlledView],
) -> dict[str, Any]:
    rows = []
    for view in sorted(views, key=lambda row: row.item_id):
        actual_sha256 = hashlib.sha256(view.source_bytes).hexdigest()
        expected_item_id = "item." + hashlib.sha256(
            view.media_type.encode("utf-8")
            + b"\0"
            + view.source_bytes
        ).hexdigest()
        if actual_sha256 != view.source_sha256:
            raise ValueError(
                f"declared source hash mismatch: {view.view_id}"
            )
        if view.item_id != expected_item_id:
            raise ValueError("opaque item id mismatch")
        spans = {
            key: list(value)
            for key, value in sorted(view.record_spans.items())
        }
        if any(
            not 0 <= start < end <= len(view.source_bytes)
            for start, end in view.record_spans.values()
        ):
            raise ValueError(
                f"record span outside source: {view.item_id}"
            )
        rows.append(
            {
                "item_id": view.item_id,
                "media_type": view.media_type,
                "source_sha256": actual_sha256,
                "source_size": len(view.source_bytes),
                "record_count": len(view.record_spans),
                "record_spans_commitment": strict_content_hash(spans),
            }
        )
    return {
        "corpus_version": CORPUS_VERSION,
        "case_count": ROOT_COUNT,
        "case_count_semantics": "paired_not_independent",
        "primary_root_count": PRIMARY_ROOT_COUNT,
        "paired_negative_count": PAIRED_NEGATIVE_ROOT_COUNT,
        "paired_missingness_control_count": PAIRED_CONTROL_ROOT_COUNT,
        "view_count": len(rows),
        "views_per_case": VIEWS_PER_ROOT,
        "view_semantics": (
            "serialization_and_field_alias_invariance_only"
        ),
        "natural_language_paraphrase_claimed": False,
        "rows": rows,
        "raw_pack_hash": strict_content_hash(rows),
    }


def gold_pack_contract(
    roots: Sequence[ControlledRoot],
) -> dict[str, Any]:
    view_links = controlled_view_gold_linkage(roots)
    view_link_commitments = [
        row.linkage_commitment
        for row in sorted(view_links, key=lambda item: item.item_id)
    ]
    commitments = [
        root.gold_commitment
        for root in sorted(roots, key=lambda row: row.root_id)
    ]
    pairs = [
        {
            "pair_id": root.pair_id,
            "pair_role": root.pair_role,
            "paired_primary_root_id": root.paired_primary_root_id,
            "paired_root_id": root.root_id,
            "operator_receipt_hash": (
                None
                if root.operator_receipt is None
                else root.operator_receipt.receipt_hash
            ),
        }
        for root in sorted(roots, key=lambda row: row.root_id)
        if root.pair_role != "primary"
    ]
    return {
        "corpus_version": CORPUS_VERSION,
        "case_count": len(commitments),
        "case_count_semantics": "paired_not_independent",
        "primary_root_count": sum(
            root.pair_role == "primary" for root in roots
        ),
        "paired_negative_count": sum(
            root.pair_role == "counterfactual_negative"
            for root in roots
        ),
        "paired_missingness_control_count": sum(
            root.pair_role == "missingness_control"
            for root in roots
        ),
        "operator_contract_hash": OPERATOR_CONTRACT_HASH,
        "private_view_linkage_count": len(view_links),
        "private_view_linkage_commitments": view_link_commitments,
        "root_gold_commitments": commitments,
        "pair_rows": pairs,
        "gold_pack_hash": strict_content_hash(
            {
                "commitments": commitments,
                "pairs": pairs,
                "operator_contract_hash": OPERATOR_CONTRACT_HASH,
                "private_view_linkage_commitments": (
                    view_link_commitments
                ),
            }
        ),
    }


_LINE_RECORD = re.compile(
    r"^(?:Fact|Record) (?P<kind>[a-z_]+) "
    r"(?:named )?(?P<record_id>[a-z][a-z0-9_.-]+) "
    r"(?:carries|has) (?P<attrs>\{.*\})\.$"
)


def _walk_strings(value: Any) -> Sequence[str]:
    result: list[str] = []
    if isinstance(value, str):
        result.append(value)
    elif isinstance(value, list):
        for item in value:
            result.extend(_walk_strings(item))
    elif isinstance(value, dict):
        for key, item in value.items():
            result.append(str(key))
            result.extend(_walk_strings(item))
    return result


def _decoded_view_strings(view: ControlledView) -> tuple[str, ...]:
    envelope = {
        "item_id": view.item_id,
        "media_type": view.media_type,
        "source_sha256": view.source_sha256,
        "source_size": len(view.source_bytes),
        "record_spans_commitment": strict_content_hash(
            {
                key: list(value)
                for key, value in sorted(view.record_spans.items())
            }
        ),
    }
    strings: list[str] = list(_walk_strings(envelope))
    for raw_line in view.source_bytes.splitlines():
        text = raw_line.decode("utf-8", errors="strict")
        strings.append(text)
        if view.media_type == JSONL_MEDIA_TYPE:
            payload = json.loads(text)
            strings.extend(_walk_strings(payload))
            continue
        match = _LINE_RECORD.fullmatch(text)
        if match is None:
            raise ValueError("controlled line record is malformed")
        strings.extend(
            (match.group("kind"), match.group("record_id"))
        )
        strings.extend(
            _walk_strings(json.loads(match.group("attrs")))
        )
    return tuple(strings)


def validate_no_runtime_answer_leak(
    views: Sequence[ControlledView],
) -> tuple[str, ...]:
    issues: list[str] = []
    for view in views:
        try:
            strings = _decoded_view_strings(view)
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
            issues.append(
                f"raw_leak_scan_parse_failed.{view.view_id}"
            )
            continue
        for fragment in _FORBIDDEN_RAW_FRAGMENTS:
            needle = fragment.lower()
            if any(needle in value.lower() for value in strings):
                issues.append(
                    f"raw_answer_fragment_leak.{fragment}"
                )
    return tuple(sorted(set(issues)))


def validate_pair_operator_receipts(
    roots: Sequence[ControlledRoot],
) -> tuple[str, ...]:
    issues: list[str] = []
    roots_by_id = {root.root_id: root for root in roots}
    for root in roots:
        if root.pair_role == "primary":
            if (
                root.operator_receipt is not None
                or root.paired_primary_root_id is not None
                or root.pair_id is not None
            ):
                issues.append(
                    f"primary_has_pair_metadata.{root.root_id}"
                )
            continue
        primary = roots_by_id.get(root.paired_primary_root_id or "")
        receipt = root.operator_receipt
        if primary is None or receipt is None or root.pair_id is None:
            issues.append(f"paired_metadata_missing.{root.root_id}")
            continue
        if (
            receipt.input_records_sha256
            != primary.raw_records_sha256
            or receipt.output_records_sha256
            != root.raw_records_sha256
            or receipt.operator_id not in _OPERATOR_TARGET_SPECS
            or receipt.operator_implementation_sha256
            != _OPERATOR_IMPLEMENTATION_SHA256.get(
                receipt.operator_id
            )
            or receipt.operator_implementation_sha256
            != _operator_implementation_sha256(receipt.operator_id)
            or receipt.frozen_target_spec_hash
            != strict_content_hash(
                list(_OPERATOR_TARGET_SPECS[receipt.operator_id])
            )
            or receipt.allowed_diff_paths
            != _OPERATOR_TARGET_SPECS[receipt.operator_id]
            or receipt.allowed_diff_paths
            != receipt.observed_diff_paths
            or not receipt.non_target_fields_unchanged
        ):
            issues.append(f"operator_receipt_invalid.{root.root_id}")
        if (
            root.pair_role == "counterfactual_negative"
            and root.hard_negative_operator_id
            != receipt.operator_id
        ):
            issues.append(
                f"negative_operator_binding_invalid.{root.root_id}"
            )
        if (
            root.pair_role == "missingness_control"
            and root.hard_negative_operator_id is not None
        ):
            issues.append(
                f"control_mislabeled_negative.{root.root_id}"
            )
    return tuple(sorted(set(issues)))


_TOKEN = re.compile(r"[a-z][a-z0-9_]{2,}")


def atomic_fact_tokens(root: ControlledRoot) -> frozenset[str]:
    """Compatibility diagnostic over facts; not a narrative-overlap metric."""

    text = json.dumps(
        list(root.records),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).lower()
    return frozenset(_TOKEN.findall(text))


def narrative_tokens(root: ControlledRoot) -> frozenset[str]:
    """Deprecated compatibility alias; the corpus contains no narratives."""

    return atomic_fact_tokens(root)


def jaccard(
    left: Sequence[str] | set[str] | frozenset[str],
    right: Sequence[str] | set[str] | frozenset[str],
) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    return 1.0 if not union else len(left_set & right_set) / len(union)


__all__ = [
    "CORPUS_VERSION",
    "CounterfactualOperatorReceipt",
    "ControlledGoldLink",
    "ControlledRoot",
    "ControlledView",
    "FROZEN_OPERATOR_IDS",
    "OPERATOR_CONTRACT_HASH",
    "PAIRED_CONTROL_ROOT_COUNT",
    "PAIRED_NEGATIVE_ROOT_COUNT",
    "PRIMARY_ROOT_COUNT",
    "ROOT_COUNT",
    "T05",
    "T09",
    "T14",
    "T15",
    "T17",
    "VIEW_COUNT",
    "VIEWS_PER_ROOT",
    "apply_frozen_operator",
    "atomic_fact_tokens",
    "atomic_record_diff_paths",
    "build_controlled_roots",
    "controlled_view_gold_linkage",
    "gold_pack_contract",
    "jaccard",
    "narrative_tokens",
    "raw_pack_contract",
    "render_controlled_views",
    "validate_no_runtime_answer_leak",
    "validate_pair_operator_receipts",
]
