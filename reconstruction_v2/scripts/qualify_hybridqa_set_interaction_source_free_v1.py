#!/usr/bin/env python3
"""Source-free exact-runtime qualification for the HybridQA set evaluator."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


PROJECT_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_PACKAGE_ROOT))

from assumption_agent.benchmarks import (  # noqa: E402
    hybridqa_marginal_replacement_meta_development_v1 as base,
)
from assumption_agent.benchmarks import (  # noqa: E402
    hybridqa_query_anchored_operator_v1 as operator,
)
from assumption_agent.benchmarks import (  # noqa: E402
    hybridqa_set_interaction_meta_development_v1 as subject,
)


VERSION = "qualify_hybridqa_set_interaction_source_free_v1"


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        + b"\n"
    )


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)[:-1]).hexdigest()


def self_hashed(body: dict[str, object], field: str) -> dict[str, object]:
    return {**body, field: stable_hash(body)}


def write_exclusive(path: Path, value: object) -> str:
    raw = canonical_bytes(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        written = 0
        while written < len(raw):
            written += os.write(descriptor, raw[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(raw).hexdigest()


def synthetic_item() -> tuple[subject.DiagnosticItem, operator.TypedCorpusGraph]:
    units: list[operator.AtomicUnit] = []
    for index in range(operator.CORPUS_UNIT_COUNT):
        if index <= 5:
            units.append(
                operator.AtomicUnit(
                    index,
                    "table_row",
                    "shared",
                    index,
                    ("target",) if index == 5 else (),
                )
            )
        elif index == 6:
            units.append(
                operator.AtomicUnit(
                    index,
                    "linked_passage",
                    "shared",
                    None,
                    ("target",),
                )
            )
        else:
            units.append(
                operator.AtomicUnit(
                    index,
                    "table_row",
                    f"table_{index}",
                    0,
                    (),
                )
            )
    graph = operator.build_typed_graph(units)
    facets = (
        operator.make_query_facet(0, "entity", "synthetic entity"),
        operator.make_query_facet(1, "relation_clause", "synthetic relation"),
    )
    dense = [0] * operator.CORPUS_UNIT_COUNT
    for ordinal, score in enumerate(
        (1_000_000, 900_000, 800_000, 700_000, 600_000, 500_000, 400_000)
    ):
        dense[ordinal] = score
    coverage = [[0] * operator.CORPUS_UNIT_COUNT for _ in facets]
    coverage[0][:7] = [
        101_003,
        202_007,
        303_011,
        404_017,
        505_019,
        923_457,
        246_802,
    ]
    coverage[1][:7] = [
        111_013,
        222_023,
        333_031,
        444_049,
        555_061,
        135_791,
        876_543,
    ]
    anchors = [[0] * operator.CORPUS_UNIT_COUNT for _ in facets]
    anchors[0][4] = 800_000
    tensor = operator.make_query_semantic_tensor(
        query_sha256="1" * 64,
        facets=facets,
        semantic_coverage_ints=coverage,
        direct_anchor_strength_ints=anchors,
        dense_relevance_ints=dense,
    )
    order = tuple(
        sorted(
            range(operator.CORPUS_UNIT_COUNT),
            key=lambda ordinal: (-tensor.dense_relevance_ints[ordinal], ordinal),
        )
    )
    ranks = [0] * operator.CORPUS_UNIT_COUNT
    for rank, ordinal in enumerate(order):
        ranks[ordinal] = rank
    reachability = operator._query_anchored_reachability(graph, tensor)
    candidates = tuple(
        ordinal
        for ordinal, record in enumerate(reachability)
        if ordinal not in set(order[:5])
        and record.path_length is not None
        and record.path_length <= subject.MAX_PATH_LENGTH
    )
    if candidates != (5, 6):
        raise RuntimeError("synthetic candidate universe drifted")
    return (
        subject.DiagnosticItem(
            block="A_form",
            family="DUAL_TABLE_PASSAGE",
            commitment="2" * 64,
            gold=(5, 6),
            tensor=tensor,
            raw_top5=tuple(order[:5]),
            raw_rank=tuple(ranks),
            reachability=reachability,
            candidates=candidates,
        ),
        graph,
    )


def qualify(
    *,
    asset_manifest: Path,
    model_root: Path,
) -> dict[str, object]:
    numeric_first = subject.compute_set_energy_numeric_canary()
    numeric_second = subject.compute_set_energy_numeric_canary()
    if numeric_first != numeric_second:
        raise RuntimeError("numeric canary is not repeat exact")
    expected_numeric = subject.EXPECTED_SET_ENERGY_NUMERIC_CANARY_SHA256
    if expected_numeric and (
        numeric_first["float64_payload_sha256"] != expected_numeric
    ):
        raise RuntimeError("frozen numeric canary mismatch")

    item, graph = synthetic_item()
    batches = tuple(subject.iter_state_batches(item=item, graph=graph))
    outputs = tuple(
        tuple(sorted(int(value) for value in output))
        for batch in batches
        for output in batch.outputs
    )
    if (
        len(subject.FEATURE_ORDER) != 48
        or subject.complete_state_count(2) != 21
        or len(outputs) != 20
        or len(set(outputs)) != len(outputs)
    ):
        raise RuntimeError("complete set enumeration drifted")
    context = subject._build_feature_context(item=item, graph=graph)
    left_features = subject._state_features(
        item=item,
        context=context,
        slots=(0, 1),
        candidate_local=np.asarray([[5, 6]], dtype=np.int64),
        outputs_local=np.asarray([[5, 6, 2, 3, 4]], dtype=np.int64),
    )
    right_features = subject._state_features(
        item=item,
        context=context,
        slots=(0, 1),
        candidate_local=np.asarray([[6, 5]], dtype=np.int64),
        outputs_local=np.asarray([[6, 5, 2, 3, 4]], dtype=np.int64),
    )
    if not np.array_equal(left_features, right_features):
        raise RuntimeError("set feature assignment invariance drifted")
    try:
        tuple(
            subject.iter_state_batches(
                item=replace(item, candidates=item.candidates[:-1]),
                graph=graph,
            )
        )
    except subject.HybridQaSetInteractionError:
        pass
    else:
        raise RuntimeError("candidate pruning was not rejected")
    statistics = subject.item_sufficient_statistics(item=item, graph=graph)
    learned, oracle = subject.select_set_and_oracle(
        item=item,
        graph=graph,
        weights=np.zeros(len(subject.FEATURE_ORDER), dtype=np.float64),
    )
    if (
        statistics.state_count != 21
        or learned != subject.PolicyOutcome(item.raw_top5, 0)
        or oracle.replacements != 2
        or not {5, 6}.issubset(oracle.output)
    ):
        raise RuntimeError("fit or global selection boundary drifted")

    encoder = base.open_gpu_encoder(
        asset_manifest_path=asset_manifest,
        model_root=model_root,
    )
    encoder_receipt = base.portable_encoder_receipt_sha256(encoder)
    return self_hashed(
        {
            "schema": f"{VERSION}_receipt",
            "version": VERSION,
            "status": (
                "verified_source_free_runtime"
                if expected_numeric
                else "observed_before_numeric_freeze"
            ),
            "architecture_decision_self_sha256": (
                subject.ARCHITECTURE_DECISION_SHA256
            ),
            "diagnostic_version": subject.VERSION,
            "feature_count": len(subject.FEATURE_ORDER),
            "synthetic_complete_state_count": statistics.state_count,
            "candidate_pruning_rejected": True,
            "candidate_assignment_bitwise_invariant": True,
            "numeric_canary_repeat_exact": True,
            "numeric_canary_expected_sha256": expected_numeric or None,
            "numeric_canary": numeric_first,
            "base_encoder_receipt_sha256": encoder_receipt,
            "base_encoder_canary_receipt_sha256": base.stable_hash(
                dict(encoder.canary_receipt)
            ),
            "source_pack_access_count": 0,
            "online_or_API_evaluation_count": 0,
        },
        "receipt_self_sha256",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    receipt = qualify(
        asset_manifest=arguments.asset_manifest,
        model_root=arguments.model_root,
    )
    write_exclusive(arguments.output, receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "numeric_canary": receipt["numeric_canary"],
                "receipt_self_sha256": receipt["receipt_self_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
