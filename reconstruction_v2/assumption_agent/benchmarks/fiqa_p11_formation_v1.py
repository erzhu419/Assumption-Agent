"""Retrospective, non-claim FiQA TRAIN formation of the frozen P11 ranker."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from reconstruction_v2.assumption_agent.benchmarks import (
    bright_bridge_expansion_core_v1 as bridge,
)
from reconstruction_v2.assumption_agent.benchmarks import p11_raw_ce_rrf_v1 as p11


SCHEMA = "fiqa_p11_formation_result_v1"
RESULT_RELATIVE = Path("manifests/fiqa_p11_formation_result_v1.json")
INTENTS_RELATIVE = Path(
    "artifacts/fiqa_bridge_expansion_train_runtime_v1/action.intents.json"
)
CROSS_RELATIVE = Path(
    "artifacts/fiqa_bridge_expansion_train_runtime_v1/cross_encoder.output.json"
)
ACTIONS_RELATIVE = Path(
    "artifacts/fiqa_bridge_expansion_train_runtime_v2/three_arm.actions.json"
)
LABELS_RELATIVE = Path(
    "artifacts/fiqa_bridge_expansion_train_integration_v2/train_integration.labels.jsonl"
)
CORPUS_RELATIVE = Path(
    "artifacts/fiqa_bridge_expansion_train_integration_v2/source_members/corpus.filtered.jsonl"
)

ARTIFACT_SHA256 = {
    INTENTS_RELATIVE.as_posix(): "3b70a9b353801ec34716f715dff63e47018f8fafb45972d1637855fa1b9f713b",
    CROSS_RELATIVE.as_posix(): "c9d5b6f907812cab0b19908cfbe52e3110eeb1b3f49d3910d48cd95839af0101",
    ACTIONS_RELATIVE.as_posix(): "4a2591c84fcd03270c711a60292b52b0175527c0a3d9d628b120c2f17b992acb",
    LABELS_RELATIVE.as_posix(): "51bb5b29d69fb55d71714f8a3970b3f5930048c86fc5176d18ecfb0f048a506a",
    CORPUS_RELATIVE.as_posix(): "f535f42980d3527dec75e10391d57937c6d0cd96b2792c9bcd0b4c3a6e864600",
}


class FiqaP11FormationError(RuntimeError):
    """The retrospective TRAIN formation failed closed."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()


def _write_result(path: Path, value: Mapping[str, Any]) -> None:
    raw = (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _read_json(path: Path, name: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FiqaP11FormationError(f"{name} is unavailable")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FiqaP11FormationError(f"{name} is invalid") from exc
    if not isinstance(value, Mapping):
        raise FiqaP11FormationError(f"{name} is not an object")
    return value


def _verify_artifacts(base: Path) -> None:
    for relative, expected in ARTIFACT_SHA256.items():
        path = base / relative
        if file_sha256(path) != expected:
            raise FiqaP11FormationError(f"formation artifact drifted: {relative}")


def _rrf(
    rankings: Sequence[Sequence[int]],
    pool: Sequence[int],
    weights: Sequence[int],
) -> tuple[int, ...]:
    if len(rankings) != len(weights):
        raise FiqaP11FormationError("formation RRF shape drifted")
    totals = {row: Fraction(0, 1) for row in pool}
    for ranking, weight in zip(rankings, weights):
        for rank, row in enumerate(ranking, start=1):
            if row in totals:
                totals[row] += Fraction(weight, 60 + rank)
    return tuple(sorted(pool, key=lambda row: (-totals[row], row))[:10])


def _paired(left: Sequence[int], right: Sequence[int]) -> dict[str, int]:
    deltas = [int(a) - int(b) for a, b in zip(left, right)]
    return {
        "gain": sum(value > 0 for value in deltas),
        "harm": sum(value < 0 for value in deltas),
        "net_integer_ndcg": sum(deltas),
        "tie": sum(value == 0 for value in deltas),
    }


def _candidate_rows(
    *,
    pool: Sequence[int],
    base_pool: Sequence[int],
    raw: Sequence[int],
    relation_scores: Sequence[int],
    mechanism_scores: Sequence[int],
) -> Mapping[str, tuple[int, ...]]:
    relation_by_row = {row: int(relation_scores[index]) for index, row in enumerate(pool)}
    mechanism_by_row = {row: int(mechanism_scores[index]) for index, row in enumerate(pool)}
    relation = tuple(sorted(pool, key=lambda row: (-relation_by_row[row], row)))
    mechanism = tuple(sorted(pool, key=lambda row: (-mechanism_by_row[row], row)))
    combined = tuple(
        sorted(
            pool,
            key=lambda row: (-(relation_by_row[row] + mechanism_by_row[row]), row),
        )
    )
    base = set(base_pool)
    relation_base = tuple(row for row in relation if row in base)
    return {
        "CE_RELATION_BASE": relation_base[:10],
        "CE_SUM_FULL": combined[:10],
        "RRF_RAW1_CE_RELATION1": _rrf((raw, relation), pool, (1, 1)),
        "RRF_RAW1_CE_SUM1": _rrf((raw, combined), pool, (1, 1)),
        "RRF_RAW1_CE_SUM2": p11.rank_p11(
            expanded_pool=pool,
            raw_top10=raw,
            cross_encoder_relation_scores=relation_scores,
            cross_encoder_mechanism_scores=mechanism_scores,
        ),
        "RRF_RAW2_CE_SUM1": _rrf((raw, combined), pool, (2, 1)),
        "RRF_RAW3_CE_SUM1": _rrf((raw, combined), pool, (3, 1)),
    }


def run_formation(project_root: Path) -> dict[str, Any]:
    base = project_root.resolve(strict=True) / "reconstruction_v2"
    result_path = base / RESULT_RELATIVE
    if result_path.exists() or result_path.is_symlink():
        raise FiqaP11FormationError("P11 formation result already exists")
    _verify_artifacts(base)
    intents = _read_json(base / INTENTS_RELATIVE, "TRAIN intents").get("items")
    cross = _read_json(base / CROSS_RELATIVE, "TRAIN cross output").get("items")
    actions = _read_json(base / ACTIONS_RELATIVE, "TRAIN actions").get("items")
    if not all(isinstance(value, list) and len(value) == 12 for value in (intents, cross, actions)):
        raise FiqaP11FormationError("TRAIN item vectors drifted")
    corpus_ids: list[str] = []
    with (base / CORPUS_RELATIVE).open(encoding="ascii") as handle:
        for line in handle:
            value = json.loads(line)
            corpus_ids.append(str(value["_id"]))
    id_to_row = {identifier: row for row, identifier in enumerate(corpus_ids)}
    labels: dict[str, tuple[int, ...]] = {}
    with (base / LABELS_RELATIVE).open(encoding="ascii") as handle:
        for line in handle:
            value = json.loads(line)
            labels[value["item_key"]] = tuple(
                id_to_row[str(identifier)] for identifier in value["gold_document_ids"]
            )
    recipe_scores: dict[str, list[int]] = {"RAW": [], "HippoRAG": [], "P10": []}
    recipe_rows: dict[str, list[tuple[int, ...]]] = {}
    item_diagnostics: list[dict[str, Any]] = []
    for ordinal, (intent, cross_row, action) in enumerate(zip(intents, cross, actions)):
        if not all(row.get("ordinal") == ordinal for row in (intent, cross_row, action)):
            raise FiqaP11FormationError("TRAIN ordinal drifted")
        key = action["item_key"]
        gold = labels[key]
        raw = tuple(action["RAW_rows"])
        hippo = tuple(action["HippoRAG"]["top_rows"])
        original_p10 = tuple(action["P10_rows"])
        recipe_scores["RAW"].append(bridge.integer_ndcg_at_10(raw, gold))
        recipe_scores["HippoRAG"].append(bridge.integer_ndcg_at_10(hippo, gold))
        recipe_scores["P10"].append(bridge.integer_ndcg_at_10(original_p10, gold))
        candidates = _candidate_rows(
            pool=intent["expanded_pool"],
            base_pool=intent["base_pool"],
            raw=raw,
            relation_scores=cross_row["relation_scores_quantized"],
            mechanism_scores=cross_row["mechanism_scores_quantized"],
        )
        for name, rows in candidates.items():
            recipe_rows.setdefault(name, []).append(rows)
            recipe_scores.setdefault(name, []).append(
                bridge.integer_ndcg_at_10(rows, gold)
            )
        p11_rows = candidates["RRF_RAW1_CE_SUM2"]
        item_diagnostics.append(
            {
                "item_key_sha256": hashlib.sha256(key.encode("ascii")).hexdigest(),
                "ordinal": ordinal,
                "P11_outside_base_top10_count": len(
                    set(p11_rows) - set(intent["base_pool"])
                ),
                "P11_score": recipe_scores["RRF_RAW1_CE_SUM2"][-1],
                "RAW_score": recipe_scores["RAW"][-1],
            }
        )
    candidates_report: dict[str, Any] = {}
    for name in sorted(recipe_scores):
        scores = recipe_scores[name]
        delta = [a - b for a, b in zip(scores, recipe_scores["RAW"])]
        candidates_report[name] = {
            "mean_ndcg_at_10": math.fsum(scores) / len(scores) / 1_000_000_000,
            "minimum_leave_one_out_net_vs_RAW": sum(delta) - max(delta),
            "sum_integer_ndcg": sum(scores),
            "vs_HippoRAG": _paired(scores, recipe_scores["HippoRAG"]),
            "vs_RAW": _paired(scores, recipe_scores["RAW"]),
        }
    eligible = [
        name
        for name in recipe_rows
        if candidates_report[name]["vs_RAW"]["harm"] == 0
    ]
    selected = max(
        eligible,
        key=lambda name: (
            candidates_report[name]["sum_integer_ndcg"],
            candidates_report[name]["minimum_leave_one_out_net_vs_RAW"],
            name,
        ),
    )
    if selected != "RRF_RAW1_CE_SUM2":
        raise FiqaP11FormationError("frozen P11 selection drifted")
    result: dict[str, Any] = {
        "artifact_bindings": ARTIFACT_SHA256,
        "candidate_reports": candidates_report,
        "claim_boundary": {
            "DEV_or_TEST_access_count": 0,
            "formation_is_retrospective_consumed_TRAIN_only": True,
            "prospective_performance_claim": False,
        },
        "item_diagnostic_set_sha256": stable_hash(item_diagnostics),
        "recorded_date": "2026-07-20",
        "schema": SCHEMA,
        "selected_candidate": {
            "candidate_name": p11.CANDIDATE_NAME,
            "cross_encoder_weight": p11.CROSS_ENCODER_WEIGHT,
            "formation_recipe_id": selected,
            "raw_weight": p11.RAW_WEIGHT,
            "rrf_k": p11.RRF_K,
            "selection_rule": "among_zero_harm_vs_RAW_recipes_maximize_sum_integer_ndcg_then_minimum_leave_one_out_net_then_name",
        },
        "status": "formed_nonclaim_P11_ready_for_freeze_before_new_source_payload_access",
    }
    result["self_sha256"] = stable_hash(result)
    _write_result(result_path, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    arguments = parser.parse_args(argv)
    value = run_formation(arguments.project_root)
    print(json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
