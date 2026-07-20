"""Recovery-only FiQA TRAIN runtime with a true late-label boundary.

The interrupted v1 run produced a complete, hash-bound label-free P10 plan
and expanded-pool cross-encoder output but no HippoRAG terminal output or
action seal.  This non-claim TRAIN version reconstructs and verifies that
plan, runs all HippoRAG units from clean directories, seals the three-arm
actions, and only then opens the TRAIN label pack.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import subprocess
import threading
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from reconstruction_v2.assumption_agent.benchmarks import (
    fiqa_bridge_expansion_train_runtime_v1 as v1,
)


SCHEMA = "fiqa_bridge_expansion_train_runtime_result_v2"
ATTEMPT_SCHEMA = "fiqa_bridge_expansion_train_runtime_attempt_v2"
ACTION_SCHEMA = "fiqa_bridge_expansion_train_runtime_actions_v2"
FREEZE_SCHEMA = "fiqa_bridge_expansion_train_runtime_implementation_freeze_v2"

FAILURE_RELATIVE = Path("manifests/fiqa_bridge_expansion_train_runtime_failure_v1.json")
FAILURE_FILE_SHA256 = "3956bd1d1302bd7249676dfd64cf3c47879d6ccb3a5fa86d09a74d8a14383522"
FAILURE_SELF_SHA256 = "d7b322eff80798982de4b4ad4e5e18e155216280ba9bfffe212437a84f0f24dc"
FREEZE_RELATIVE = Path(
    "manifests/fiqa_bridge_expansion_train_runtime_implementation_freeze_v2.json"
)
RESULT_RELATIVE = Path("manifests/fiqa_bridge_expansion_train_runtime_result_v2.json")
RUN_ROOT_RELATIVE = Path("artifacts/fiqa_bridge_expansion_train_runtime_v2")
V1_ROOT_RELATIVE = Path("artifacts/fiqa_bridge_expansion_train_runtime_v1")
IMPLEMENTATION_RELATIVE = Path(
    "assumption_agent/benchmarks/fiqa_bridge_expansion_train_runtime_v2.py"
)
TEST_RELATIVE = Path("tests/test_fiqa_bridge_expansion_train_runtime_v2.py")


class FiqaTrainRuntimeV2Error(RuntimeError):
    """The frozen recovery-only TRAIN runtime failed closed."""


class OneShotRefusal(FiqaTrainRuntimeV2Error):
    """The v2 recovery attempt or result path is already consumed."""


@dataclass(frozen=True)
class ReconstructedPlan:
    plans: tuple[v1.ExpandedPlan, ...]
    ids: tuple[str, ...]
    contents: tuple[str, ...]
    corpus_embeddings: np.ndarray
    cross_output: Mapping[str, Any]
    intents: Mapping[str, Any]


def _read_json(path: Path, name: str) -> Mapping[str, Any]:
    return v1._read_json(path, name)


def _load_failure(base: Path) -> Mapping[str, Any]:
    path = base / FAILURE_RELATIVE
    if (
        not path.is_file()
        or path.is_symlink()
        or v1.integration_v1.file_sha256(path) != FAILURE_FILE_SHA256
    ):
        raise FiqaTrainRuntimeV2Error("v1 failure receipt file binding drifted")
    value = _read_json(path, "v1 TRAIN runtime failure receipt")
    try:
        v1._verify_self(value, "self_sha256", FAILURE_SELF_SHA256)
    except v1.FiqaTrainRuntimeError as exc:
        raise FiqaTrainRuntimeV2Error(str(exc)) from exc
    return value


def _verify_freeze(base: Path) -> Mapping[str, Any]:
    value = _read_json(base / FREEZE_RELATIVE, "v2 TRAIN runtime freeze")
    if value.get("schema") != FREEZE_SCHEMA:
        raise FiqaTrainRuntimeV2Error("v2 TRAIN runtime freeze schema drifted")
    declared = value.get("self_sha256")
    if not isinstance(declared, str):
        raise FiqaTrainRuntimeV2Error("v2 TRAIN runtime freeze identity is absent")
    try:
        v1._verify_self(value, "self_sha256", declared)
    except v1.FiqaTrainRuntimeError as exc:
        raise FiqaTrainRuntimeV2Error(str(exc)) from exc
    rows = value.get("implementation_bindings")
    if not isinstance(rows, list):
        raise FiqaTrainRuntimeV2Error("v2 TRAIN runtime freeze bindings drifted")
    observed = {
        row.get("relative_path"): row.get("sha256")
        for row in rows
        if isinstance(row, Mapping)
    }
    expected_paths = {IMPLEMENTATION_RELATIVE.as_posix(), TEST_RELATIVE.as_posix()}
    if set(observed) != expected_paths:
        raise FiqaTrainRuntimeV2Error("v2 TRAIN runtime freeze file set drifted")
    for relative, expected in observed.items():
        if (
            not isinstance(relative, str)
            or not isinstance(expected, str)
            or v1.integration_v1.file_sha256(base / relative) != expected
        ):
            raise FiqaTrainRuntimeV2Error("v2 TRAIN runtime implementation drifted")
    if value.get("failure_v1_self_sha256") != FAILURE_SELF_SHA256:
        raise FiqaTrainRuntimeV2Error("v2 TRAIN runtime failure binding drifted")
    return value


def _verify_label_free_artifacts(
    base: Path,
    failure: Mapping[str, Any],
) -> Mapping[str, Path]:
    rows = failure.get("label_free_artifact_bindings")
    if not isinstance(rows, list) or len(rows) != 9:
        raise FiqaTrainRuntimeV2Error("v1 label-free artifact registry drifted")
    output: dict[str, Path] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise FiqaTrainRuntimeV2Error("v1 artifact row drifted")
        relative = row.get("relative_path")
        digest = row.get("sha256")
        size = row.get("size_bytes")
        if (
            not isinstance(relative, str)
            or not isinstance(digest, str)
            or isinstance(size, bool)
            or not isinstance(size, int)
        ):
            raise FiqaTrainRuntimeV2Error("v1 artifact binding value drifted")
        path = base / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != size
            or v1.integration_v1.file_sha256(path) != digest
        ):
            raise FiqaTrainRuntimeV2Error("v1 label-free artifact file drifted")
        output[path.name] = path
    if len(output) != 9:
        raise FiqaTrainRuntimeV2Error("v1 label-free artifact names drifted")
    return output


def load_train_views(
    base: Path,
    integration: Mapping[str, Any],
) -> tuple[v1.ViewItem, ...]:
    binding = integration.get("TRAIN_diagnostic_pack")
    if not isinstance(binding, Mapping) or binding.get("item_count") != v1.ITEM_COUNT:
        raise FiqaTrainRuntimeV2Error("TRAIN view binding drifted")
    view_path = base / v1.TRAIN_SOURCE_ROOT_RELATIVE / "train_integration.view.jsonl"
    if (
        view_path.stat().st_size != binding.get("view_file_size_bytes")
        or v1.integration_v1.file_sha256(view_path) != binding.get("view_file_sha256")
    ):
        raise FiqaTrainRuntimeV2Error("TRAIN view file drifted")
    rows = v1._parse_canonical_jsonl(view_path, "TRAIN view pack")
    if len(rows) != v1.ITEM_COUNT:
        raise FiqaTrainRuntimeV2Error("TRAIN view row count drifted")
    items: list[v1.ViewItem] = []
    for ordinal, row in enumerate(rows):
        if set(row) != {
            "excluded_document_ids",
            "family",
            "item_key",
            "query",
            "source_query_id",
        } or row.get("family") != "FIQA":
            raise FiqaTrainRuntimeV2Error("TRAIN view row shape drifted")
        excluded_raw = row.get("excluded_document_ids")
        if not isinstance(excluded_raw, list):
            raise FiqaTrainRuntimeV2Error("TRAIN view exclusions drifted")
        items.append(
            v1.ViewItem(
                ordinal=ordinal,
                item_key=v1.integration_v1._required_text(row.get("item_key"), "item key"),
                query=v1.integration_v1._required_text(row.get("query"), "query"),
                excluded_ids=tuple(
                    v1.integration_v1._required_text(value, "excluded document ID")
                    for value in excluded_raw
                ),
            )
        )
    if len({item.item_key for item in items}) != len(items):
        raise FiqaTrainRuntimeV2Error("TRAIN view item keys are duplicated")
    return tuple(items)


def load_train_labels_after_seal(
    *,
    base: Path,
    integration: Mapping[str, Any],
    items: Sequence[v1.ViewItem],
    action_path: Path,
    expected_action_file_sha256: str,
) -> Mapping[str, tuple[str, ...]]:
    if (
        not action_path.is_file()
        or action_path.is_symlink()
        or v1.integration_v1.file_sha256(action_path) != expected_action_file_sha256
    ):
        raise FiqaTrainRuntimeV2Error("action seal is absent or drifted before labels")
    binding = integration.get("TRAIN_diagnostic_pack")
    if not isinstance(binding, Mapping):
        raise FiqaTrainRuntimeV2Error("TRAIN label binding drifted")
    label_path = base / v1.TRAIN_SOURCE_ROOT_RELATIVE / "train_integration.labels.jsonl"
    if (
        label_path.stat().st_size != binding.get("label_file_size_bytes")
        or v1.integration_v1.file_sha256(label_path) != binding.get("label_file_sha256")
    ):
        raise FiqaTrainRuntimeV2Error("TRAIN label file drifted")
    rows = v1._parse_canonical_jsonl(label_path, "TRAIN label pack")
    if len(rows) != len(items):
        raise FiqaTrainRuntimeV2Error("TRAIN label row count drifted")
    labels: dict[str, tuple[str, ...]] = {}
    for item, row in zip(items, rows):
        if (
            set(row) != {"family", "gold_document_ids", "item_key"}
            or row.get("family") != "FIQA"
            or row.get("item_key") != item.item_key
        ):
            raise FiqaTrainRuntimeV2Error("TRAIN label identity drifted")
        gold_raw = row.get("gold_document_ids")
        if not isinstance(gold_raw, list) or not gold_raw:
            raise FiqaTrainRuntimeV2Error("TRAIN gold list drifted")
        gold = tuple(
            v1.integration_v1._required_text(value, "gold document ID")
            for value in gold_raw
        )
        if len(set(gold)) != len(gold):
            raise FiqaTrainRuntimeV2Error("TRAIN gold IDs are duplicated")
        labels[item.item_key] = gold
    return labels


def reconstruct_label_free_plan(
    *,
    base: Path,
    integration: Mapping[str, Any],
    items: Sequence[v1.ViewItem],
    artifacts: Mapping[str, Path],
) -> ReconstructedPlan:
    ids, contents = v1.load_filtered_corpus(base, integration)
    ids_pack = _read_json(artifacts["corpus.ids.json"], "v1 corpus ID pack")
    declared = ids_pack.get("pack_sha256")
    if not isinstance(declared, str):
        raise FiqaTrainRuntimeV2Error("v1 corpus ID pack identity is absent")
    try:
        v1.integration_v1.verify_self_hash(ids_pack, "pack_sha256", declared)
    except v1.integration_v1.FiqaTrainIntegrationError as exc:
        raise FiqaTrainRuntimeV2Error(str(exc)) from exc
    if tuple(ids_pack.get("document_ids", ())) != ids:
        raise FiqaTrainRuntimeV2Error("v1 corpus ID pack drifted")
    try:
        corpus_embeddings = np.asarray(
            np.load(artifacts["corpus.embeddings.npy"], allow_pickle=False),
            dtype=np.float32,
        )
        query_embeddings = np.asarray(
            np.load(artifacts["typed_query.embeddings.npy"], allow_pickle=False),
            dtype=np.float32,
        )
        bridge_embeddings = np.asarray(
            np.load(artifacts["bridge_query.embeddings.npy"], allow_pickle=False),
            dtype=np.float32,
        )
    except Exception as exc:
        raise FiqaTrainRuntimeV2Error("v1 embedding artifact is invalid") from exc
    if (
        corpus_embeddings.shape != (len(ids), 384)
        or query_embeddings.shape != (len(items) * 5, 384)
        or bridge_embeddings.ndim != 2
        or bridge_embeddings.shape[1] != 384
        or not all(
            np.isfinite(value).all()
            for value in (corpus_embeddings, query_embeddings, bridge_embeddings)
        )
    ):
        raise FiqaTrainRuntimeV2Error("v1 embedding shape drifted")

    qwen_input = v1.qwen_contract.parse_input(artifacts["qwen.input.json"].read_bytes())
    if tuple(row.query for row in qwen_input) != tuple(item.query for item in items):
        raise FiqaTrainRuntimeV2Error("v1 Qwen input identity drifted")
    qwen_output = v1.qwen_contract.parse_output(artifacts["qwen.output.json"].read_bytes())
    qwen_rows = qwen_output.get("items")
    if (
        not isinstance(qwen_rows, list)
        or len(qwen_rows) != len(items)
        or not all(row.get("generation_valid") is True for row in qwen_rows)
    ):
        raise FiqaTrainRuntimeV2Error("v1 Qwen output drifted")

    local_plans: list[v1.LocalPlan] = []
    for item, row in zip(items, qwen_rows):
        start = item.ordinal * 5
        scores = [
            v1.quantized_scores(corpus_embeddings, query_embeddings[index])
            for index in range(start, start + 5)
        ]
        local_plans.append(
            v1.build_local_plan(
                item=item,
                document_ids=ids,
                document_contents=contents,
                query_score_vectors=scores,
                expansions=row["expansions"],
            )
        )
    plans: list[v1.ExpandedPlan] = []
    offset = 0
    for local in local_plans:
        count = len(local.bridge_queries)
        bridge_scores = [
            v1.quantized_scores(corpus_embeddings, bridge_embeddings[index])
            for index in range(offset, offset + count)
        ]
        plans.append(v1.expand_plan(local, bridge_scores))
        offset += count
    if offset != len(bridge_embeddings):
        raise FiqaTrainRuntimeV2Error("v1 bridge embedding accounting drifted")

    intents = _read_json(artifacts["action.intents.json"], "v1 action intents")
    declared_intents = intents.get("pack_sha256")
    if not isinstance(declared_intents, str):
        raise FiqaTrainRuntimeV2Error("v1 intent identity is absent")
    try:
        v1.integration_v1.verify_self_hash(intents, "pack_sha256", declared_intents)
    except v1.integration_v1.FiqaTrainIntegrationError as exc:
        raise FiqaTrainRuntimeV2Error(str(exc)) from exc
    intent_rows = intents.get("items")
    if not isinstance(intent_rows, list) or len(intent_rows) != len(plans):
        raise FiqaTrainRuntimeV2Error("v1 intent rows drifted")
    for plan, row in zip(plans, intent_rows):
        if (
            not isinstance(row, Mapping)
            or row.get("ordinal") != plan.local.item.ordinal
            or row.get("item_key") != plan.local.item.item_key
            or tuple(row.get("base_pool", ())) != plan.local.base_pool
            or tuple(row.get("expanded_pool", ())) != plan.expanded.expanded_pool
        ):
            raise FiqaTrainRuntimeV2Error("reconstructed P10 plan differs from v1 intent")

    rebuilt_cross = v1.build_cross_input(plans, contents)
    if (
        v1.cross_contract.canonical_json_bytes(rebuilt_cross)
        != artifacts["cross_encoder.input.json"].read_bytes()
    ):
        raise FiqaTrainRuntimeV2Error("reconstructed cross-encoder input differs from v1")
    cross_output = v1.cross_contract.parse_output(
        artifacts["cross_encoder.output.json"].read_bytes()
    )
    return ReconstructedPlan(
        plans=tuple(plans),
        ids=ids,
        contents=contents,
        corpus_embeddings=corpus_embeddings,
        cross_output=cross_output,
        intents=intents,
    )


def _git_head(project_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _verify_replayed_hipporag_inputs(
    *,
    previous_root: Path,
    new_roots: Sequence[Path],
) -> None:
    previous_paths = tuple(
        sorted(previous_root.glob("item_*/input.json"))
    )
    new_paths = tuple(path / "input.json" for path in new_roots)
    if len(previous_paths) != v1.ITEM_COUNT or len(new_paths) != v1.ITEM_COUNT:
        raise FiqaTrainRuntimeV2Error("HippoRAG replay input count drifted")
    for previous, new in zip(previous_paths, new_paths):
        if (
            not previous.is_file()
            or previous.is_symlink()
            or not new.is_file()
            or new.is_symlink()
            or previous.read_bytes() != new.read_bytes()
        ):
            raise FiqaTrainRuntimeV2Error(
                "v2 HippoRAG input differs from reconstructed v1 intent"
            )


def run_formal(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    base = project_root / "reconstruction_v2"
    result_path = base / RESULT_RELATIVE
    if result_path.exists():
        raise OneShotRefusal("v2 TRAIN runtime result already exists")
    preconditions = v1._load_preconditions(project_root)
    failure = _load_failure(base)
    freeze = _verify_freeze(base)
    artifacts = _verify_label_free_artifacts(base, failure)
    integration = preconditions["integration"]
    items = load_train_views(base, integration)
    reconstructed = reconstruct_label_free_plan(
        base=base,
        integration=integration,
        items=items,
        artifacts=artifacts,
    )

    root = base / RUN_ROOT_RELATIVE
    try:
        root.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise OneShotRefusal("v2 TRAIN runtime root already exists") from exc
    marker = {
        "failure_v1_self_sha256": FAILURE_SELF_SHA256,
        "implementation_freeze_self_sha256": freeze["self_sha256"],
        "schema": ATTEMPT_SCHEMA,
        "v1_intent_pack_sha256": reconstructed.intents["pack_sha256"],
    }
    marker_path = root / "attempt.marker"
    v1.bright_runtime._write_json(marker_path, marker)
    hippo_roots = v1._prepare_hipporag_inputs(
        root=root,
        plans=reconstructed.plans,
        contents=reconstructed.contents,
    )
    _verify_replayed_hipporag_inputs(
        previous_root=base / V1_ROOT_RELATIVE / "hipporag",
        new_roots=hippo_roots,
    )

    semaphore = threading.Semaphore(v1.HIPPORAG_CONCURRENCY)
    counter = v1.bright_runtime._ConcurrencyCounter()
    futures: dict[Future[Any], int] = {}
    completed: dict[int, Mapping[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=v1.HIPPORAG_CONCURRENCY) as executor:
        for index, (plan, item_root) in enumerate(zip(reconstructed.plans, hippo_roots)):
            future = executor.submit(
                v1.bright_runtime._run_hipporag_item,
                project_root=base,
                item_root=item_root,
                candidate_rows=plan.local.base_pool,
                semaphore=semaphore,
                counter=counter,
            )
            futures[future] = index
        for future in as_completed(futures):
            completed[futures[future]] = future.result()
    if (
        counter.current != 0
        or counter.peak > v1.HIPPORAG_CONCURRENCY
        or set(completed) != set(range(v1.ITEM_COUNT))
    ):
        raise FiqaTrainRuntimeV2Error("v2 HippoRAG completion drifted")

    action_rows: list[dict[str, Any]] = []
    for plan, cross_row in zip(reconstructed.plans, reconstructed.cross_output["items"]):
        if cross_row["document_count"] != len(plan.expanded.expanded_pool):
            raise FiqaTrainRuntimeV2Error("v1 cross-encoder output row drifted")
        p10 = v1.bridge.rank_p10(
            expanded=plan.expanded,
            original_scores=plan.local.original_scores,
            relation_scores=plan.local.relation_scores,
            mechanism_scores=plan.local.mechanism_scores,
            cross_encoder_relation_scores=cross_row["relation_scores_quantized"],
            cross_encoder_mechanism_scores=cross_row["mechanism_scores_quantized"],
        )
        hippo = dict(completed[plan.local.item.ordinal])
        action_rows.append(
            {
                "bridge_anchor_count": len(plan.local.anchors),
                "bridge_query_count": len(plan.local.bridge_queries),
                "candidate_expansion": dict(
                    v1.bridge.candidate_expansion_diagnostics(
                        base_pool=plan.local.base_pool,
                        expanded_pool=plan.expanded.expanded_pool,
                        p10_rows=p10.rows,
                    )
                ),
                "HippoRAG": {
                    **hippo,
                    "document_ids": [reconstructed.ids[row] for row in hippo["top_rows"]],
                },
                "item_key": plan.local.item.item_key,
                "ordinal": plan.local.item.ordinal,
                "P10_document_ids": [reconstructed.ids[row] for row in p10.rows],
                "P10_rows": list(p10.rows),
                "RAW_document_ids": [
                    reconstructed.ids[row] for row in plan.local.raw_rows
                ],
                "RAW_rows": list(plan.local.raw_rows),
            }
        )
    actions = v1.integration_v1.self_hashed(
        {
            "active_Agent": "P10_TYPED_BRIDGE_EXPAND_CE_RRF",
            "item_count": v1.ITEM_COUNT,
            "items": action_rows,
            "schema": ACTION_SCHEMA,
            "v1_intent_pack_sha256": reconstructed.intents["pack_sha256"],
        },
        "pack_sha256",
    )
    action_path = root / "three_arm.actions.json"
    v1.bright_runtime._write_json(action_path, actions)
    action_file_sha256 = v1.integration_v1.file_sha256(action_path)

    labels = load_train_labels_after_seal(
        base=base,
        integration=integration,
        items=items,
        action_path=action_path,
        expected_action_file_sha256=action_file_sha256,
    )
    id_to_row = {
        identifier: index for index, identifier in enumerate(reconstructed.ids)
    }
    arm_scores: dict[str, list[int]] = {"P10": [], "RAW": [], "HippoRAG": []}
    diagnostics: list[Mapping[str, int]] = []
    for plan, action in zip(reconstructed.plans, action_rows):
        gold_ids = labels[plan.local.item.item_key]
        if not set(gold_ids) <= set(id_to_row):
            raise FiqaTrainRuntimeV2Error("TRAIN gold references filtered corpus absence")
        arm_scores["P10"].append(
            v1.core.integer_ndcg_at_10(action["P10_document_ids"], gold_ids)
        )
        arm_scores["RAW"].append(
            v1.core.integer_ndcg_at_10(action["RAW_document_ids"], gold_ids)
        )
        arm_scores["HippoRAG"].append(
            v1.core.integer_ndcg_at_10(action["HippoRAG"]["document_ids"], gold_ids)
        )
        diagnostics.append(
            v1.bridge.candidate_expansion_diagnostics(
                base_pool=plan.local.base_pool,
                expanded_pool=plan.expanded.expanded_pool,
                p10_rows=action["P10_rows"],
                gold_rows=[id_to_row[value] for value in gold_ids],
            )
        )
    aggregates = {
        arm: {
            "mean_ndcg_at_10": sum(values) / (v1.ITEM_COUNT * v1.core.UTILITY_SCALE),
            "sum_integer_ndcg": sum(values),
        }
        for arm, values in arm_scores.items()
    }
    result = v1.integration_v1.self_hashed(
        {
            "aggregates": aggregates,
            "candidate_expansion_aggregates": {
                key: sum(int(row[key]) for row in diagnostics)
                for key in diagnostics[0]
            },
            "claim_boundary": {
                "claim_eligible": False,
                "DEV_qrel_member_open_count": 0,
                "external_network_call_count": 0,
                "labels_opened_after_all_action_seal": True,
                "online_evaluator_call_count": 0,
                "TEST_qrel_member_open_count": 0,
            },
            "execution": {
                "cross_encoder_call_count": 0,
                "HippoRAG_peak_process_concurrency": counter.peak,
                "HippoRAG_terminal_count": len(completed),
                "MiniLM_call_count": 0,
                "qwen_call_count": 0,
                "reused_v1_label_free_cross_encoder_item_count": v1.ITEM_COUNT,
            },
            "formal_binding": {
                "action_file_sha256": action_file_sha256,
                "action_pack_sha256": actions["pack_sha256"],
                "attempt_marker_sha256": v1.integration_v1.file_sha256(marker_path),
                "failure_v1_self_sha256": FAILURE_SELF_SHA256,
                "formal_implementation_commit": _git_head(project_root),
                "implementation_freeze_self_sha256": freeze["self_sha256"],
                "integration_result_self_sha256": v1.INTEGRATION_RESULT_SELF_SHA256,
                "v1_intent_pack_sha256": reconstructed.intents["pack_sha256"],
            },
            "item_count": v1.ITEM_COUNT,
            "paired_descriptives": {
                "P10_minus_HippoRAG": v1._paired(
                    arm_scores["P10"], arm_scores["HippoRAG"]
                ),
                "P10_minus_RAW": v1._paired(arm_scores["P10"], arm_scores["RAW"]),
            },
            "schema": SCHEMA,
            "status": "TRAIN_end_to_end_runtime_v2_complete_nonclaim_true_late_label_DEV_and_TEST_unopened",
        },
        "result_sha256",
    )
    v1.bright_runtime._write_json(result_path, result, mode=0o644)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = run_formal(arguments.project_root)
    print(
        v1.integration_v1.canonical_json(
            {
                "result_sha256": result["result_sha256"],
                "schema": SCHEMA,
                "status": result["status"],
            }
        ).decode("ascii")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
