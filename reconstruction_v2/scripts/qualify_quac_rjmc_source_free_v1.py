#!/usr/bin/env python3
"""Non-formal, source-free development qualification for RJMC-V1.

Every graph and label in this file is hand authored.  The script has no source
path, network, model-asset, API, or output-file argument and cannot create a
formal result.  It exercises pair complementarity, redundancy, retention, and
null-shift topologies before any QuAC payload is acquired.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Iterable


PROJECT_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_PACKAGE_ROOT))

from assumption_agent.benchmarks import quac_rjmc_evaluator_v1 as core  # noqa: E402


VERSION = "qualify_quac_rjmc_source_free_v1"
TOPOLOGIES = (
    "pair_complement",
    "redundancy_trap",
    "retention_trap",
    "null_shift",
)


class SourceFreeQualificationError(RuntimeError):
    """The hand-authored RJMC development qualification failed closed."""


def _edge(
    left: int,
    right: int,
    relation: str,
    strength: float,
) -> core.TypedEdge:
    return core.TypedEdge(
        left=f"u{min(left, right):02d}",
        right=f"u{max(left, right):02d}",
        relation=relation,
        strength=strength,
    )


def _unit(
    ordinal: int,
    *,
    dense: float,
    anchor: float,
    recency: float,
    proximity: float,
    facets: tuple[int, int, int, int],
) -> core.EvidenceUnit:
    return core.EvidenceUnit(
        unit_id=f"u{ordinal:02d}",
        node_features=(dense, anchor, recency, proximity),
        dialogue_facets=facets,
    )


def _utility(state: core.SetState, required: tuple[str, str]) -> int:
    hits = sum(unit_id in state.unit_ids for unit_id in required)
    return hits + 2 * int(hits == 2)


def _synthetic_graph(
    topology: str,
    *,
    perturbation: float,
    structural_variant: str = "base",
) -> tuple[core.RelationalGraph, tuple[str, str]]:
    if topology not in TOPOLOGIES:
        raise SourceFreeQualificationError("unknown synthetic topology")
    if structural_variant not in ("base", "extra_distractor"):
        raise SourceFreeQualificationError("unknown synthetic structural variant")
    # The small perturbation changes numeric presentation without changing the
    # causal topology.  It is fixed before each A_form/A_hold/M construction.
    offset = perturbation
    raw_units = [
        _unit(
            0,
            dense=1.00 - offset,
            anchor=0.35,
            recency=0.90,
            proximity=0.90,
            facets=(1, 0, 0, 0),
        ),
        _unit(
            1,
            dense=0.96 - offset,
            anchor=0.35,
            recency=0.85,
            proximity=0.90,
            facets=(0, 1, 0, 0),
        ),
        _unit(
            2,
            dense=0.70 + offset,
            anchor=0.55,
            recency=0.70,
            proximity=0.80,
            facets=(0, 0, 1, 0),
        ),
        _unit(
            3,
            dense=0.66 + offset,
            anchor=0.55,
            recency=0.65,
            proximity=0.80,
            facets=(0, 0, 0, 1),
        ),
        _unit(
            4,
            dense=0.62 + offset,
            anchor=0.50,
            recency=0.60,
            proximity=0.75,
            facets=(1, 0, 0, 0),
        ),
    ]
    edges = [
        _edge(0, 1, "adjacent_window", 0.65),
        _edge(1, 2, "adjacent_window", 0.55),
        _edge(2, 3, "adjacent_window", 0.55),
        _edge(3, 4, "adjacent_window", 0.45),
    ]

    if topology == "pair_complement":
        candidates = [
            _unit(
                5,
                dense=0.30 + offset,
                anchor=0.58,
                recency=0.55,
                proximity=0.72,
                facets=(1, 1, 0, 0),
            ),
            _unit(
                6,
                dense=0.28 + offset,
                anchor=0.58,
                recency=0.52,
                proximity=0.72,
                facets=(0, 0, 1, 1),
            ),
            _unit(
                7,
                dense=0.24 + offset,
                anchor=0.98,
                recency=0.50,
                proximity=0.68,
                facets=(1, 1, 1, 1),
            ),
        ]
        edges.extend(
            (
                _edge(5, 6, "entity_chain", 1.00),
                _edge(5, 1, "entity_chain", 0.72),
                _edge(6, 3, "entity_chain", 0.72),
                _edge(7, 2, "same_section", 1.00),
                _edge(7, 3, "same_section", 1.00),
            )
        )
        required = ("u05", "u06")
    elif topology == "redundancy_trap":
        candidates = [
            _unit(
                5,
                dense=0.52 + offset,
                anchor=0.60,
                recency=0.58,
                proximity=0.74,
                facets=(0, 1, 1, 0),
            ),
            _unit(
                6,
                dense=0.22 + offset,
                anchor=1.00,
                recency=0.48,
                proximity=0.68,
                facets=(1, 1, 1, 1),
            ),
            _unit(
                7,
                dense=0.20 + offset,
                anchor=1.00,
                recency=0.46,
                proximity=0.68,
                facets=(1, 1, 1, 1),
            ),
        ]
        edges.extend(
            (
                _edge(0, 5, "entity_chain", 1.00),
                _edge(5, 2, "entity_chain", 0.75),
                _edge(6, 7, "same_section", 1.00),
                _edge(6, 2, "same_section", 1.00),
                _edge(7, 3, "same_section", 1.00),
            )
        )
        required = ("u00", "u05")
    elif topology == "retention_trap":
        candidates = [
            _unit(
                ordinal,
                dense=0.16 + offset + (7 - ordinal) * 0.01,
                anchor=1.00,
                recency=0.45,
                proximity=0.65,
                facets=(1, 1, 1, 1),
            )
            for ordinal in (5, 6, 7)
        ]
        edges.extend(
            (
                _edge(5, 6, "same_section", 1.00),
                _edge(6, 7, "same_section", 1.00),
                _edge(5, 2, "same_section", 0.95),
                _edge(6, 3, "same_section", 0.95),
                _edge(7, 4, "same_section", 0.95),
            )
        )
        required = ("u00", "u01")
    else:
        candidates = [
            _unit(
                ordinal,
                dense=0.12 + offset,
                anchor=0.92,
                recency=0.42,
                proximity=0.62,
                facets=facets,
            )
            for ordinal, facets in (
                (5, (1, 1, 1, 0)),
                (6, (1, 1, 0, 1)),
                (7, (0, 1, 1, 1)),
            )
        ]
        edges.extend(
            (
                _edge(5, 6, "same_section", 0.95),
                _edge(6, 7, "same_section", 0.95),
                _edge(5, 3, "same_section", 0.90),
                _edge(7, 4, "same_section", 0.90),
            )
        )
        required = ("u00", "u01")

    if structural_variant == "extra_distractor":
        candidates.append(
            _unit(
                8,
                dense=0.10 + offset,
                anchor=0.99,
                recency=0.39,
                proximity=0.59,
                facets=(1, 1, 1, 1),
            )
        )
        edges.extend(
            (
                _edge(2, 8, "same_section", 0.88),
                _edge(7, 8, "same_section", 0.97),
            )
        )

    graph = core.RelationalGraph(
        units=tuple(sorted((*raw_units, *candidates), key=lambda row: row.unit_id)),
        edges=tuple(
            sorted(
                edges,
                key=lambda row: (
                    row.left,
                    row.right,
                    core.RELATION_TYPES.index(row.relation),
                ),
            )
        ),
    )
    return graph, required


def make_item(
    *,
    item_id: str,
    topology: str,
    component: int,
    perturbation: float,
    structural_variant: str = "base",
) -> tuple[core.ListwiseTrainingItem, tuple[str, str], str]:
    graph, required = _synthetic_graph(
        topology,
        perturbation=perturbation,
        structural_variant=structural_variant,
    )
    raw = tuple(f"u{ordinal:02d}" for ordinal in range(5))
    states = core.enumerate_complete_states(graph, raw_top5=raw)
    item = core.ListwiseTrainingItem(
        item_id=item_id,
        component=component,
        graph=graph,
        raw_top5=raw,
        utility=tuple(_utility(state, required) for state in states),
    )
    return item, required, topology


def build_a_form() -> tuple[core.ListwiseTrainingItem, ...]:
    rows = []
    for component in range(core.COMPONENT_COUNT):
        for topology_index, topology in enumerate(TOPOLOGIES):
            item, _required, _topology = make_item(
                item_id=f"aform-c{component}-{topology}",
                topology=topology,
                component=component,
                perturbation=0.006 * component + 0.002 * topology_index,
            )
            rows.append(item)
    return tuple(rows)


def build_measurement(
    name: str,
    *,
    phase: float,
    structural_variant: str = "base",
) -> tuple[tuple[core.ListwiseTrainingItem, tuple[str, str], str], ...]:
    rows = []
    for replicate in range(2):
        for topology_index, topology in enumerate(TOPOLOGIES):
            rows.append(
                make_item(
                    item_id=f"{name}-r{replicate}-{topology}",
                    topology=topology,
                    component=(replicate + topology_index) % core.COMPONENT_COUNT,
                    perturbation=phase
                    + 0.004 * replicate
                    + 0.001 * topology_index,
                    structural_variant=structural_variant,
                )
            )
    return tuple(rows)


def _evaluate(
    ensemble: core.JackknifeMinimaxComparator,
    rows: Iterable[
        tuple[core.ListwiseTrainingItem, tuple[str, str], str]
    ],
) -> dict[str, object]:
    deltas: dict[str, list[int]] = defaultdict(list)
    raw_harm: dict[str, list[int]] = defaultdict(list)
    selected_required: dict[str, list[bool]] = defaultdict(list)
    e0_total = 0
    e1_total = 0
    for item, required, topology in rows:
        states = item.states
        e0_index = core.select_e0_proof_coverage(
            item.graph, raw_top5=item.raw_top5
        )
        e1_index, _scores = ensemble.select(
            item.graph, raw_top5=item.raw_top5
        )
        e0_utility = item.utility[e0_index]
        e1_utility = item.utility[e1_index]
        e0_total += e0_utility
        e1_total += e1_utility
        deltas[topology].append(e1_utility - e0_utility)
        raw_harm[topology].append(e1_utility - item.utility[0])
        selected_required[topology].append(
            all(unit_id in states[e1_index].unit_ids for unit_id in required)
        )
    return {
        "item_count": sum(len(values) for values in deltas.values()),
        "E0_total_utility": e0_total,
        "E1_total_utility": e1_total,
        "E1_minus_E0": e1_total - e0_total,
        "topology_delta": {
            topology: sum(deltas[topology]) for topology in TOPOLOGIES
        },
        "topology_raw_harm": {
            topology: min(raw_harm[topology]) for topology in TOPOLOGIES
        },
        "topology_required_complete": {
            topology: all(selected_required[topology]) for topology in TOPOLOGIES
        },
    }


def _block_passes(block: dict[str, object]) -> bool:
    topology_delta = block["topology_delta"]
    topology_raw_harm = block["topology_raw_harm"]
    topology_complete = block["topology_required_complete"]
    return bool(
        block["E1_minus_E0"] > 0
        and all(topology_delta[name] > 0 for name in TOPOLOGIES)
        and topology_raw_harm["retention_trap"] >= 0
        and topology_raw_harm["null_shift"] >= 0
        and topology_complete["pair_complement"]
    )


def qualify() -> dict[str, object]:
    a_form = build_a_form()
    config = core.FitConfig()
    first = core.fit_component_jackknife(a_form, config=config)
    second = core.fit_component_jackknife(a_form, config=config)
    first_parameter_hash = core.model_parameter_sha256(first)
    second_parameter_hash = core.model_parameter_sha256(second)
    first_behavior_hash = core.behavior_sha256(first, a_form)
    second_behavior_hash = core.behavior_sha256(second, a_form)
    if (
        first_parameter_hash != second_parameter_hash
        or first_behavior_hash != second_behavior_hash
    ):
        raise SourceFreeQualificationError("same-host deterministic replay drifted")

    probe, _required, _topology = make_item(
        item_id="identity-probe",
        topology="pair_complement",
        component=0,
        perturbation=0.033,
    )
    states = probe.states
    if len(states) != core.complete_state_count(3):
        raise SourceFreeQualificationError("complete state count drifted")
    head = first.heads[0]
    left = states[-1].unit_ids
    raw = probe.raw_top5
    forward = core.compare_sets(head, probe.graph, left=left, right=raw)
    reverse = core.compare_sets(head, probe.graph, left=raw, right=left)
    permuted = core.compare_sets(
        head,
        probe.graph,
        left=tuple(reversed(left)),
        right=tuple(reversed(raw)),
    )
    raw_zero = core.compare_sets(head, probe.graph, left=raw, right=raw)
    if (
        forward != -reverse
        or forward != permuted
        or raw_zero != 0.0
    ):
        raise SourceFreeQualificationError(
            "antisymmetry, permutation invariance, or RAW zero drifted"
        )

    a_hold_rows = build_measurement("ahold", phase=0.041)
    a_hold = _evaluate(first, a_hold_rows)
    a_hold["promotion_passed"] = _block_passes(a_hold)
    if not a_hold["promotion_passed"]:
        raise SourceFreeQualificationError(
            "synthetic A_hold did not promote; M remained unopened"
        )

    # Construction itself occurs only after the preceding promotion branch.
    # M has a fourth, structurally new distractor and two new typed edges.
    m_rows = build_measurement(
        "msearch",
        phase=0.067,
        structural_variant="extra_distractor",
    )
    m_search = _evaluate(first, m_rows)
    m_search["structural_variant"] = "extra_distractor_and_two_new_edges"
    if not _block_passes(m_search):
        raise SourceFreeQualificationError(
            "independent synthetic M mechanism requirements failed"
        )

    body = {
        "schema": f"{VERSION}_development_receipt",
        "version": VERSION,
        "status": "passed_nonformal_source_free_development_qualification",
        "formal_result": False,
        "architecture_decision_self_sha256": (
            core.ARCHITECTURE_DECISION_SELF_SHA256
        ),
        "evaluator_version": core.VERSION,
        "fixture_provenance": "hand_authored_source_free_synthetic_only",
        "fixture_topologies": list(TOPOLOGIES),
        "complete_state_count_for_three_candidates": len(states),
        "component_jackknife_head_count": len(first.heads),
        "antisymmetric": True,
        "permutation_invariant": True,
        "RAW_structural_zero": True,
        "same_process_repeat_exact": True,
        "parameter_sha256": first_parameter_hash,
        "behavior_sha256": first_behavior_hash,
        "A_hold": a_hold,
        "M_search": m_search,
        "qualification_weights_disposition": "discarded_at_process_exit",
        "QuAC_source_payload_access_count": 0,
        "prior_private_source_access_count": 0,
        "online_or_API_evaluation_count": 0,
    }
    return {**body, "receipt_self_sha256": core.stable_hash(body)}


def main() -> int:
    receipt = qualify()
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
