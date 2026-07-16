"""One-shot formal orchestration for the frozen QASC evaluator study.

The runner keeps source views, labels, NLI workers, and descriptive controls
behind distinct boundaries.  In particular, a QASC retrieval view contains no
identity commitment and no commitment derived from its label envelope.  The
label file is not even stat'ed until every primary action in the stage has
reached a terminal and the local NLI runtime has passed postflight.

This module owns orchestration only.  QASC acquisition semantics live in
``qasc_evaluator_direct_action_acquisition_v1`` and recipe semantics live in
``qasc_counterfactual_chain_margin_v1``; neither is reimplemented here.
"""

from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures
from dataclasses import asdict, dataclass, is_dataclass
from fractions import Fraction
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
from typing import Any, Callable, Mapping, Sequence

from ..models import stable_hash
from .hotpot_family_out_runner_v1 import _probe_bubblewrap, verify_capability_receipt
from .musique_formal_runtime_binding_v2 import (
    PreparedFormalRuntimeV2,
    prepare_formal_runtime_v2,
)
from .musique_recursive_study_blocks_v1 import load_study_frozen_program
from .musique_typed_retriever_formation_v1 import (
    RetrievalParagraph,
    TypedRetrievalProgram,
    retrieve as typed_retrieve,
)
from replication_runtime.qasc_nli_v1 import (
    NLIWorkerPool,
    run_canary as run_nli_canary,
    verify_runtime_asset,
)


VERSION = "qasc_evaluator_direct_action_coevolution_v1"
DESIGN_SCHEMA = "qasc_evaluator_direct_action_coevolution_design_v1"
DIAGNOSTIC_SCHEMA = f"{VERSION}_infrastructure_diagnostic"
FORMATION_FREEZE_SCHEMA = f"{VERSION}_formation_pre_run_freeze"
FORMATION_RECEIPT_SCHEMA = f"{VERSION}_formation_receipt"
A_FREEZE_SCHEMA = f"{VERSION}_A_hold_pre_run_freeze"
A_REPORT_SCHEMA = f"{VERSION}_A_hold_aggregate_report"
M_FREEZE_SCHEMA = f"{VERSION}_M_search_pre_run_freeze"
M_REPORT_SCHEMA = f"{VERSION}_M_search_aggregate_report"
CONSUMPTION_SCHEMA = f"{VERSION}_authorization_consumption"
FAILURE_SCHEMA = f"{VERSION}_terminal_failure"

DESIGN_RELATIVE = "manifests/qasc_evaluator_direct_action_coevolution_design_v1.json"
DESIGN_FILE_SHA256 = "fdd1bd1d088cee851a20015227d1f3dea1d086bcaf5c0f435f1bf52e943ab003"
DESIGN_SHA256 = "7c52b7e43d02ffa986683c49ca61863c3f36985b97a1a4677a40b6cddef8c150"
DESIGN_COMMIT = "ac95a656b7bd1c4c0078f3d8f54a8f5579209aff"
NLI_RUNTIME_COMMIT = "a248dc8ea3345a036a27a1d4aca652dfbb6cee55"

DIAGNOSTIC_RELATIVE = "manifests/qasc_evaluator_direct_action_infrastructure_diagnostic_v1.json"
FORMATION_FREEZE_RELATIVE = "manifests/qasc_evaluator_direct_action_formation_pre_run_freeze_v1.json"
FORMATION_RECEIPT_RELATIVE = "manifests/qasc_evaluator_direct_action_formation_receipt_v1.json"
A_FREEZE_RELATIVE = "manifests/qasc_evaluator_direct_action_a_hold_pre_run_freeze_v1.json"
A_REPORT_RELATIVE = "manifests/qasc_evaluator_direct_action_a_hold_aggregate_report_v1.json"
M_FREEZE_RELATIVE = "manifests/qasc_evaluator_direct_action_m_search_pre_run_freeze_v1.json"
M_REPORT_RELATIVE = "manifests/qasc_evaluator_direct_action_m_search_aggregate_report_v1.json"

PRIVATE_ROOT_RELATIVE = "artifacts/qasc_evaluator_direct_action_coevolution_v1"
FORMATION_ROOT_RELATIVE = f"{PRIVATE_ROOT_RELATIVE}/formation_formal_root_v1"
A_ROOT_RELATIVE = f"{PRIVATE_ROOT_RELATIVE}/a_hold_formal_root_v1"
M_ROOT_RELATIVE = f"{PRIVATE_ROOT_RELATIVE}/m_search_formal_root_v1"
FORMATION_EVIDENCE_RELATIVE = f"{FORMATION_ROOT_RELATIVE}/formation.private.evidence.json"
A_EVIDENCE_RELATIVE = f"{A_ROOT_RELATIVE}/a_hold.private.evidence.json"
M_EVIDENCE_RELATIVE = f"{M_ROOT_RELATIVE}/m_search.private.evidence.json"

P_PROGRAM_RELATIVE = "manifests/musique_recursive_study_f1_formation_v1/frozen_program.json"
P_FORMATION_RECEIPT_RELATIVE = "manifests/musique_recursive_study_f1_formation_v1/formation.receipt.json"
P_PROGRAM_SHA256 = "0e9fea159e2dbcb302575f97954be8461c9921a91e11ef9b64a80ecab9640785"
P_PROGRAM_FILE_SHA256 = "3ea4362281fa6d86eec41506e7f017dd8794f8d09aecbac04fd2ce6309dda8a6"

BLOCK_COUNTS = {"A_form": 64, "F_search": 64, "A_hold": 64, "M_search": 64}
FORMATION_BLOCKS = ("A_form", "F_search")
DOCUMENT_COUNT = 32
CHOICE_COUNT = 8
TOP_K = 5
RECIPE_COUNT = 16
FOLD_COUNT = 4
NLI_WORKERS = 8
TORCH_THREADS_PER_WORKER = 4
OFFICIAL_CONCURRENCY_CAP = 24
PROMOTION_ALPHA = Fraction(1, 10)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_CLEAN_MODULE_CLI_ACTIVE = False


class QASCCoevolutionError(RuntimeError):
    """Raised when a frozen runner boundary or terminal invariant drifts."""


@dataclass(frozen=True)
class ControlOutcome:
    control_id: str
    status: str
    rankings: tuple[tuple[int, ...] | None, ...]
    failure_type_hashes: tuple[str | None, ...]

    def public_summary(self) -> dict[str, Any]:
        available = sum(row is not None for row in self.rankings)
        failures = len(self.rankings) - available
        return {
            "control_id": self.control_id,
            "status": self.status,
            "item_count": len(self.rankings),
            "available_item_count": available,
            "failed_item_count": failures,
            "ranking_vector_sha256": stable_hash(
                [None if row is None else list(row) for row in self.rankings]
            ),
            "failure_type_set_sha256": stable_hash(
                sorted({row for row in self.failure_type_hashes if row is not None})
            ),
            "descriptive_only": True,
            "affects_primary_or_epoch": False,
        }


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: str | Path) -> str:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise QASCCoevolutionError("required file is unavailable")
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise QASCCoevolutionError(f"{field} must be lowercase SHA-256")
    return value


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise QASCCoevolutionError("value is not canonical JSON") from exc


def _read_json(path: str | Path, field: str) -> tuple[dict[str, Any], bytes]:
    candidate = Path(path).absolute()
    if candidate.is_symlink() or not candidate.is_file() or candidate.stat().st_size > 64 * 1024 * 1024:
        raise QASCCoevolutionError(f"{field} is unavailable or oversized")
    raw = candidate.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QASCCoevolutionError(f"{field} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise QASCCoevolutionError(f"{field} must be an object")
    return value, raw


def _atomic_write_exclusive(path: Path, raw: bytes, *, mode: int) -> None:
    if path.exists() or path.is_symlink() or not path.parent.is_dir():
        raise QASCCoevolutionError("exclusive output target is not fresh")
    temporary = path.parent / f".{path.name}.{os.urandom(12).hex()}.tmp"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        finally:
            temporary.unlink(missing_ok=True)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_json_exclusive(path: Path, value: Mapping[str, Any], *, public: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if public:
        _assert_public_safe(value)
    raw = json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    _atomic_write_exclusive(path, raw, mode=0o644 if public else 0o600)


def _prepare_output_parent(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise QASCCoevolutionError("formal output already exists; replay is forbidden")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    canary = path.parent / f".{path.name}.persistence-{os.urandom(8).hex()}"
    try:
        _atomic_write_exclusive(canary, b"persistence-canary\n", mode=0o600)
        canary.unlink()
        descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        canary.unlink(missing_ok=True)


def _assert_public_safe(value: object) -> None:
    forbidden_keys = {
        "answerKey",
        "choices",
        "documents",
        "fact1_document_id",
        "fact2_document_id",
        "formatted_question",
        "gold_document_ids",
        "identity_commitment_sha256",
        "item_rows",
        "label_rows",
        "raw_ranking",
        "view_rows",
        "view_sha256",
    }

    def inspect_value(node: object) -> None:
        if isinstance(node, Mapping):
            overlap = forbidden_keys.intersection(node)
            if overlap:
                raise QASCCoevolutionError(
                    f"public payload contains private field: {sorted(overlap)[0]}"
                )
            for child in node.values():
                inspect_value(child)
        elif isinstance(node, (list, tuple)):
            for child in node:
                inspect_value(child)
        elif isinstance(node, str) and (
            node.startswith("/home/")
            or node.startswith("/tmp/")
            or "\\wsl.localhost" in node
        ):
            raise QASCCoevolutionError("public payload contains a host path")

    inspect_value(value)


def _self_hashed(schema: str, body: Mapping[str, Any], field: str) -> dict[str, Any]:
    value = {"schema": schema, **dict(body)}
    value[field] = stable_hash(value)
    return value


def _verify_self_hash(
    value: Mapping[str, Any], *, schema: str, field: str
) -> str:
    body = dict(value)
    declared = _require_sha256(body.pop(field, None), field)
    if value.get("schema") != schema or stable_hash(body) != declared:
        raise QASCCoevolutionError(f"{schema} self-hash drifted")
    return declared


def _load_design(project: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = project / DESIGN_RELATIVE
    design, raw = _read_json(path, "QASC coevolution design")
    body = dict(design)
    declared = _require_sha256(body.pop("design_sha256", None), "design hash")
    runtime = design.get("runtime_and_preflight")
    formation = design.get("formation_contract")
    promotion = design.get("promotion_and_measurement_contract")
    if (
        _sha256_bytes(raw) != DESIGN_FILE_SHA256
        or declared != DESIGN_SHA256
        or stable_hash(body) != declared
        or design.get("schema") != DESIGN_SCHEMA
        or not isinstance(runtime, Mapping)
        or runtime.get("formal_default_NLI_workers") != NLI_WORKERS
        or runtime.get("formal_default_torch_threads_per_worker")
        != TORCH_THREADS_PER_WORKER
        or not isinstance(formation, Mapping)
        or formation.get("candidate_recipe_count") != RECIPE_COUNT
        or formation.get("fold_count") != FOLD_COUNT
        or not isinstance(promotion, Mapping)
        or promotion.get("alpha_numerator") != 1
        or promotion.get("alpha_denominator") != 10
        or promotion.get("labels_open_only_after_all_primary_action_terminals_in_the_stage")
        is not True
    ):
        raise QASCCoevolutionError("QASC coevolution design drifted")
    return design, {
        "relative_path": DESIGN_RELATIVE,
        "file_sha256": DESIGN_FILE_SHA256,
        "design_sha256": DESIGN_SHA256,
        "introducing_commit": DESIGN_COMMIT,
    }


def exact_magnitude_preserving_sign_flip(deltas: Sequence[int]) -> dict[str, Any]:
    """Compute the frozen one-sided exact test without Monte Carlo."""

    if not deltas or any(type(value) is not int for value in deltas):
        raise QASCCoevolutionError("paired U deltas are malformed")
    observed = sum(deltas)
    magnitudes = [abs(value) for value in deltas if value]
    distribution: Counter[int] = Counter({0: 1})
    for magnitude in magnitudes:
        updated: Counter[int] = Counter()
        for subtotal, count in distribution.items():
            updated[subtotal + magnitude] += count
            updated[subtotal - magnitude] += count
        distribution = updated
    p_value = Fraction(
        sum(count for subtotal, count in distribution.items() if subtotal >= observed),
        1 << len(magnitudes),
    )
    positive = observed > 0
    exact = p_value <= PROMOTION_ALPHA
    return {
        "test": "one_sided_exact_magnitude_preserving_sign_flip_v1",
        "observed_net_U": observed,
        "nonzero_pair_count": len(magnitudes),
        "p_value_numerator": p_value.numerator,
        "p_value_denominator": p_value.denominator,
        "p_value": float(p_value),
        "alpha_numerator": 1,
        "alpha_denominator": 10,
        "positive_observed_net": positive,
        "exact_p_at_or_below_alpha": exact,
        "promoted": positive and exact,
        "sole_promotion_criterion": True,
    }


def item_utility(top5: Sequence[int], gold_document_ids: Sequence[int]) -> tuple[int, int, int]:
    if (
        len(top5) != TOP_K
        or len(set(top5)) != TOP_K
        or any(type(value) is not int or not 0 <= value < DOCUMENT_COUNT for value in top5)
        or len(gold_document_ids) != 2
        or len(set(gold_document_ids)) != 2
        or any(type(value) is not int or not 0 <= value < DOCUMENT_COUNT for value in gold_document_ids)
    ):
        raise QASCCoevolutionError("top5 or gold document IDs are malformed")
    hits = len(set(top5).intersection(gold_document_ids))
    complete = int(hits == 2)
    return hits, complete, hits + complete


def aggregate_rankings(
    *, arm_id: str, rankings: Sequence[Sequence[int]], gold_rows: Sequence[Sequence[int]]
) -> dict[str, Any]:
    if not rankings or len(rankings) != len(gold_rows):
        raise QASCCoevolutionError("arm aggregate length drifted")
    metrics = [item_utility(ranking, gold) for ranking, gold in zip(rankings, gold_rows)]
    hit_vector = [row[0] for row in metrics]
    complete_vector = [row[1] for row in metrics]
    utility_vector = [row[2] for row in metrics]
    return {
        "arm_id": arm_id,
        "item_count": len(metrics),
        "support_hit_count": sum(hit_vector),
        "support_total": 2 * len(metrics),
        "complete_item_count": sum(complete_vector),
        "total_U": sum(utility_vector),
        "support_hit_vector_sha256": stable_hash(hit_vector),
        "complete_vector_sha256": stable_hash(complete_vector),
        "U_vector_sha256": stable_hash(utility_vector),
    }


def paired_utility_comparison(
    *,
    left_arm_id: str,
    right_arm_id: str,
    left_rankings: Sequence[Sequence[int]],
    right_rankings: Sequence[Sequence[int]],
    gold_rows: Sequence[Sequence[int]],
    confirmatory: bool,
) -> dict[str, Any]:
    if len(left_rankings) != len(right_rankings) or len(left_rankings) != len(gold_rows):
        raise QASCCoevolutionError("paired comparison length drifted")
    deltas = [
        item_utility(left, gold)[2] - item_utility(right, gold)[2]
        for left, right, gold in zip(left_rankings, right_rankings, gold_rows)
    ]
    test = exact_magnitude_preserving_sign_flip(deltas)
    if not confirmatory:
        test = {
            **test,
            "promoted": False,
            "sole_promotion_criterion": False,
            "descriptive_positive_and_p_at_or_below_alpha": (
                test["positive_observed_net"] and test["exact_p_at_or_below_alpha"]
            ),
            "descriptive_only": True,
        }
    return {
        "left_arm_id": left_arm_id,
        "right_arm_id": right_arm_id,
        "net_U": sum(deltas),
        "gain_item_count": sum(value > 0 for value in deltas),
        "harm_item_count": sum(value < 0 for value in deltas),
        "tie_item_count": sum(value == 0 for value in deltas),
        "paired_delta_vector_sha256": stable_hash(deltas),
        "paired_test": test,
        "confirmatory": confirmatory,
    }


def evaluator_epoch_transition(
    *, a_decision_sha256: str, promoted: bool
) -> dict[str, Any]:
    anchor = _require_sha256(a_decision_sha256, "A decision hash")
    incumbent_epoch = f"qasc_eval_epoch_0_{stable_hash({'anchor': anchor, 'evaluator': 'support_maximin_v1'})[:12]}"
    challenger_epoch = f"qasc_eval_epoch_1_{stable_hash({'anchor': anchor, 'evaluator': 'counterfactual_chain_margin_v1'})[:12]}"
    body = {
        "promoted": promoted,
        "previous_epoch_id": incumbent_epoch,
        "next_epoch_id": challenger_epoch if promoted else incumbent_epoch,
        "next_epoch_index": 1 if promoted else 0,
        "next_evaluator_id": (
            "counterfactual_chain_margin_v1" if promoted else "support_maximin_v1"
        ),
        "dependent_evaluator_scores_invalidated": promoted,
        "independent_source_actions_retained": True,
        "M_search_open_authorized": promoted,
        "M_search_can_rollback_epoch": False,
    }
    return {**body, "transition_sha256": stable_hash(body)}


def _object_payload(value: object) -> dict[str, Any]:
    if is_dataclass(value):
        payload = asdict(value)
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        raise QASCCoevolutionError("expected a dataclass or mapping payload")
    # The acquisition/recipe dataclasses deliberately use tuples, while their
    # committed private representation is JSON.  Normalize once through the
    # exact canonical JSON boundary so all downstream shape checks see the
    # same list/object types as the persisted view.
    normalized = json.loads(_canonical_bytes(payload))
    if not isinstance(normalized, dict):
        raise QASCCoevolutionError("object payload did not normalize to a mapping")
    return normalized


def _acquisition_module() -> Any:
    try:
        return importlib.import_module(
            ".qasc_evaluator_direct_action_acquisition_v1", package=__package__
        )
    except (ImportError, AttributeError) as exc:
        raise QASCCoevolutionError("QASC acquisition module is unavailable") from exc


def _recipe_module() -> Any:
    try:
        return importlib.import_module(
            ".qasc_counterfactual_chain_margin_v1", package=__package__
        )
    except (ImportError, AttributeError) as exc:
        raise QASCCoevolutionError("QASC recipe module is unavailable") from exc


def _load_retained_p(project: Path) -> TypedRetrievalProgram:
    if _sha256_file(project / P_PROGRAM_RELATIVE) != P_PROGRAM_FILE_SHA256:
        raise QASCCoevolutionError("retained P file identity drifted")
    try:
        program, _receipt, envelope = load_study_frozen_program(
            frozen_program_path=project / P_PROGRAM_RELATIVE,
            formation_receipt_path=project / P_FORMATION_RECEIPT_RELATIVE,
            verify_live=True,
            implementation_root=project,
        )
    except Exception as exc:
        raise QASCCoevolutionError("retained P custody drifted") from exc
    if program.program_hash != P_PROGRAM_SHA256 or envelope.get("program_hash") != P_PROGRAM_SHA256:
        raise QASCCoevolutionError("retained P identity drifted")
    return program


def _view_payload(view: object) -> dict[str, Any]:
    payload = _object_payload(view)
    if "identity_commitment_sha256" in payload or "label_envelope_sha256" in payload:
        raise QASCCoevolutionError("gold-free view contains a brute-forceable label proxy")
    return payload


def _view_item_key(view: object) -> str:
    return stable_hash(_view_payload(view))


def _retained_p_ranking(program: TypedRetrievalProgram, view: object) -> tuple[int, ...]:
    payload = _view_payload(view)
    documents = payload.get("documents")
    question = payload.get("formatted_question")
    choices = payload.get("choices")
    if not isinstance(documents, list) or len(documents) != DOCUMENT_COUNT:
        raise QASCCoevolutionError("retained P view has wrong document shape")
    acquisition = _acquisition_module()
    query = acquisition.canonical_query(
        question,
        [(row["label"], row["text"]) for row in choices],
    )
    corpus = tuple(
        RetrievalParagraph(idx=row["doc_id"], title="", text=row["text"])
        for row in documents
    )
    try:
        ranking = tuple(typed_retrieve(program, query, corpus))
    except Exception as exc:
        raise QASCCoevolutionError("retained P retrieval failed") from exc
    item_utility(ranking, (0, 1))
    return ranking


def _official_inputs(view: object) -> tuple[str, tuple[dict[str, Any], ...]]:
    payload = _view_payload(view)
    acquisition = _acquisition_module()
    choices = payload["choices"]
    query = acquisition.canonical_query(
        payload["formatted_question"],
        [(row["label"], row["text"]) for row in choices],
    )
    paragraphs = tuple(
        {
            "idx": row["doc_id"],
            "title": f"doc-{row['doc_id']}",
            "paragraph_text": row["text"],
        }
        for row in payload["documents"]
    )
    if len(paragraphs) != DOCUMENT_COUNT:
        raise QASCCoevolutionError("official control view has wrong document shape")
    return query, paragraphs


def _run_failure_isolated_control(
    *,
    control_id: str,
    views: Sequence[object],
    function: Callable[[int, object], tuple[int, ...]],
    maximum_workers: int,
) -> ControlOutcome:
    rankings: list[tuple[int, ...] | None] = [None] * len(views)
    failures: list[str | None] = [None] * len(views)

    def one(index: int, view: object) -> tuple[int, tuple[int, ...] | None, str | None]:
        try:
            ranking = tuple(function(index, view))
            item_utility(ranking, (0, 1))
            return index, ranking, None
        except Exception as exc:
            return index, None, stable_hash({"error_type": type(exc).__name__})

    with concurrent.futures.ThreadPoolExecutor(max_workers=maximum_workers) as executor:
        futures = tuple(executor.submit(one, index, view) for index, view in enumerate(views))
        for future in futures:
            index, ranking, failure = future.result()
            rankings[index] = ranking
            failures[index] = failure
    available = sum(row is not None for row in rankings)
    return ControlOutcome(
        control_id=control_id,
        status="available" if available == len(views) else "unavailable_or_partial",
        rankings=tuple(rankings),
        failure_type_hashes=tuple(failures),
    )


def _score_recipe_views_two_waves(
    *,
    views: Sequence[object],
    recipe_ids: Sequence[str] | None,
    pool: NLIWorkerPool,
) -> tuple[dict[str, tuple[object, ...]], dict[str, Any]]:
    """Execute exactly two global NLI waves and join before returning actions."""

    if not views:
        raise QASCCoevolutionError("recipe view set is empty")
    recipe = _recipe_module()
    view_keys = tuple(_view_item_key(view) for view in views)
    if len(set(view_keys)) != len(view_keys):
        raise QASCCoevolutionError("recipe views do not have unique local hashes")
    first_plans = tuple(
        recipe.build_first_stage_plan(view, recipe_ids) for view in views
    )
    if any(plan.view_sha256 != key for plan, key in zip(first_plans, view_keys)):
        raise QASCCoevolutionError("first-stage plan view binding drifted")
    first_batches = [
        (key, [pair.as_tuple() for pair in plan.pairs])
        for key, plan in zip(view_keys, first_plans)
    ]
    first_scores = pool.score_items(first_batches)
    if set(first_scores) != set(view_keys):
        raise QASCCoevolutionError("first NLI wave terminal set drifted")
    second_plans = tuple(
        recipe.build_second_stage_plan(
            view,
            first_plan.recipe_ids,
            first_scores[key],
            first_plan,
        )
        for view, key, first_plan in zip(views, view_keys, first_plans)
    )
    second_batches = [
        (key, [pair.as_tuple() for pair in plan.pairs])
        for key, plan in zip(view_keys, second_plans)
    ]
    second_scores = pool.score_items(second_batches)
    if set(second_scores) != set(view_keys):
        raise QASCCoevolutionError("second NLI wave terminal set drifted")
    actions: dict[str, tuple[object, ...]] = {}
    for view, key, first_plan, second_plan in zip(
        views, view_keys, first_plans, second_plans
    ):
        terminal = tuple(
            recipe.consume_stage_scores(
                view,
                first_plan,
                first_scores[key],
                second_plan,
                second_scores[key],
                first_plan.recipe_ids,
            )
        )
        if len(terminal) != len(first_plan.recipe_ids):
            raise QASCCoevolutionError("recipe action terminal count drifted")
        action_ids = tuple(getattr(action, "recipe_id", None) for action in terminal)
        if action_ids != tuple(first_plan.recipe_ids):
            raise QASCCoevolutionError("recipe action order drifted")
        if any(getattr(action, "view_sha256", None) != key for action in terminal):
            raise QASCCoevolutionError("recipe action view binding drifted")
        actions[key] = terminal
    action_count = sum(len(rows) for rows in actions.values())
    return actions, {
        "view_count": len(views),
        "recipe_count_per_view": len(first_plans[0].recipe_ids),
        "first_wave_actual_NLI_pair_count": sum(len(plan.pairs) for plan in first_plans),
        "first_wave_conceptual_request_count": sum(
            plan.conceptual_request_count for plan in first_plans
        ),
        "first_wave_item_terminal_count": len(first_scores),
        "second_wave_actual_NLI_pair_count": sum(len(plan.pairs) for plan in second_plans),
        "second_wave_conceptual_request_count": sum(
            plan.conceptual_request_count for plan in second_plans
        ),
        "second_wave_item_terminal_count": len(second_scores),
        "recipe_action_terminal_count": action_count,
        "all_first_wave_items_submitted_before_first_wave_join": True,
        "second_wave_built_only_after_complete_first_wave_join": True,
        "all_second_wave_items_submitted_before_second_wave_join": True,
        "labels_loaded_or_scored": False,
        "two_score_waves_exact": True,
    }


def _synthetic_view_mapping() -> dict[str, Any]:
    choices = [
        {"label": chr(ord("A") + index), "text": f"synthetic option {index}"}
        for index in range(CHOICE_COUNT)
    ]
    documents = [
        {
            "doc_id": index,
            "text": (
                f"Synthetic document {index} states that mineral sample {index} "
                f"has deterministic property {index % 7}."
            ),
            "bm25_score_int": DOCUMENT_COUNT - index,
        }
        for index in range(DOCUMENT_COUNT)
    ]
    return {
        "schema": "qasc_evaluator_direct_action_acquisition_v1_private_view",
        "block": "A_form",
        "source_member": "TRAIN",
        "formatted_question": "Which synthetic option follows from the supplied facts?",
        "choices": choices,
        "documents": documents,
        "raw_ranking": [0, 1, 2, 3, 4],
    }


def _prepare_official_runtime(
    *,
    project: Path,
    capability_receipt_path: str | Path,
    runtime_python: str | Path,
    local_llm_model: str | Path,
    local_embedding_model: str | Path,
    base_binding_receipt_path: str | Path,
    attestation_receipt_path: str | Path,
) -> tuple[dict[str, Any], PreparedFormalRuntimeV2]:
    try:
        capability, raw = verify_capability_receipt(capability_receipt_path)
        live = _probe_bubblewrap()
        if (
            live.get("probe_returncode") != 0
            or live.get("bwrap_file_sha256") != capability.get("bwrap_file_sha256")
            or live.get("probe_contract_sha256")
            != capability.get("probe_contract_sha256")
        ):
            raise QASCCoevolutionError("fresh bubblewrap probe drifted")
        prepared = prepare_formal_runtime_v2(
            project_root=project,
            attestation_receipt_path=Path(attestation_receipt_path).absolute(),
            base_binding_receipt_path=Path(base_binding_receipt_path).absolute(),
            runtime_python=Path(runtime_python).absolute(),
            local_llm_model=Path(local_llm_model).resolve(strict=True),
            local_embedding_model=Path(local_embedding_model).resolve(strict=True),
        )
    except Exception as exc:
        raise QASCCoevolutionError("official control runtime preflight failed") from exc
    binding = {
        "capability_file_sha256": _sha256_bytes(raw),
        "capability_receipt_sha256": capability["receipt_sha256"],
        "fresh_bubblewrap_probe_passed": True,
        "formal_runtime_binding": prepared.safe_binding,
    }
    _assert_public_safe(binding)
    return binding, prepared


def run_infrastructure_diagnostic(
    *,
    project_root: str | Path,
    nli_model_path: str | Path,
    capability_receipt_path: str | Path,
    runtime_python: str | Path,
    local_llm_model: str | Path,
    local_embedding_model: str | Path,
    base_binding_receipt_path: str | Path,
    attestation_receipt_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Run the row-free exact-shape pre-marker infrastructure diagnostic."""

    project = Path(project_root).resolve(strict=True)
    expected_output = (project / DIAGNOSTIC_RELATIVE).absolute()
    if Path(output_path).absolute() != expected_output:
        raise QASCCoevolutionError("diagnostic output must use its canonical path")
    _prepare_output_parent(expected_output)
    _design, design_binding = _load_design(project)
    nli_binding = verify_runtime_asset(project, nli_model_path)
    canary = run_nli_canary(project, nli_model_path, workers=NLI_WORKERS)
    recipe = _recipe_module()
    synthetic_mapping = _synthetic_view_mapping()
    view = recipe.load_retrieval_view(synthetic_mapping)
    if view.view_sha256 != stable_hash(synthetic_mapping):
        raise QASCCoevolutionError("synthetic recipe view hash drifted")
    with NLIWorkerPool(
        nli_model_path,
        workers=NLI_WORKERS,
        project_root=project,
    ) as pool:
        actions, recipe_execution = _score_recipe_views_two_waves(
            views=(view,), recipe_ids=None, pool=pool
        )
    nli_postflight = verify_runtime_asset(project, nli_model_path)
    if nli_postflight != nli_binding:
        raise QASCCoevolutionError("NLI runtime postflight drifted")
    terminal_actions = actions[view.view_sha256]
    if len(terminal_actions) != RECIPE_COUNT or any(
        len(getattr(action, "ordered_top5", ())) != TOP_K
        for action in terminal_actions
    ):
        raise QASCCoevolutionError("synthetic 16-recipe terminal shape drifted")

    raw_outcome = _run_failure_isolated_control(
        control_id="canonical_RAW",
        views=(view,),
        function=lambda _index, row: tuple(_view_payload(row)["raw_ranking"]),
        maximum_workers=1,
    )
    try:
        retained = _load_retained_p(project)
        p_outcome = _run_failure_isolated_control(
            control_id="retained_P",
            views=(view,),
            function=lambda _index, row: _retained_p_ranking(retained, row),
            maximum_workers=1,
        )
    except Exception as exc:
        p_outcome = ControlOutcome(
            control_id="retained_P",
            status="unavailable_or_partial",
            rankings=(None,),
            failure_type_hashes=(
                stable_hash({"error_type": type(exc).__name__}),
            ),
        )
    official_binding: dict[str, Any]
    official_postflight: dict[str, Any] | None = None
    try:
        official_binding, prepared = _prepare_official_runtime(
            project=project,
            capability_receipt_path=capability_receipt_path,
            runtime_python=runtime_python,
            local_llm_model=local_llm_model,
            local_embedding_model=local_embedding_model,
            base_binding_receipt_path=base_binding_receipt_path,
            attestation_receipt_path=attestation_receipt_path,
        )
        diagnostic_parent = project / PRIVATE_ROOT_RELATIVE / "diagnostic_official_work"
        diagnostic_parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        official_views = (view,) * OFFICIAL_CONCURRENCY_CAP
        official_outcome = _run_failure_isolated_control(
            control_id="official_HippoRAG_item_local_32",
            views=official_views,
            function=lambda index, row: prepared.retrieve(
                question=_official_inputs(row)[0],
                paragraphs=_official_inputs(row)[1],
                work_root=diagnostic_parent / f"item-{index:02d}",
            ),
            maximum_workers=OFFICIAL_CONCURRENCY_CAP,
        )
        try:
            official_postflight = prepared.fresh_reverify()
        except Exception as exc:
            official_outcome = ControlOutcome(
                control_id=official_outcome.control_id,
                status="unavailable_or_partial",
                rankings=tuple(None for _ in official_outcome.rankings),
                failure_type_hashes=tuple(
                    stable_hash({"error_type": type(exc).__name__})
                    for _ in official_outcome.rankings
                ),
            )
    except Exception as exc:
        official_binding = {
            "status": "preflight_unavailable",
            "error_type_sha256": stable_hash({"error_type": type(exc).__name__}),
        }
        official_outcome = ControlOutcome(
            control_id="official_HippoRAG_item_local_32",
            status="unavailable_or_partial",
            rankings=(None,) * OFFICIAL_CONCURRENCY_CAP,
            failure_type_hashes=(
                stable_hash({"error_type": type(exc).__name__}),
            )
            * OFFICIAL_CONCURRENCY_CAP,
        )

    action_hashes = [getattr(action, "action_sha256") for action in terminal_actions]
    body = {
        "status": "passed_row_free_QASC_infrastructure_diagnostic",
        "design_binding": design_binding,
        "implementation_binding": _current_implementation_binding(project),
        "nli_runtime_commit": NLI_RUNTIME_COMMIT,
        "nli_runtime_binding_sha256": stable_hash(nli_binding),
        "nli_canary": canary,
        "synthetic_recipe": {
            **recipe_execution,
            "document_count": DOCUMENT_COUNT,
            "choice_count": CHOICE_COUNT,
            "recipe_action_set_sha256": stable_hash(action_hashes),
            "all_ordered_top5_valid": True,
        },
        "controls": {
            "canonical_RAW": raw_outcome.public_summary(),
            "retained_P": p_outcome.public_summary(),
            "official_HippoRAG_item_local_32": official_outcome.public_summary(),
            "official_control_failure_invalidates_primary": False,
        },
        "official_runtime": {
            "preflight_binding": official_binding,
            "postflight_passed": official_postflight is not None,
            "postflight_binding_sha256": (
                None if official_postflight is None else stable_hash(official_postflight)
            ),
        },
        "formal_QA_rows_read": 0,
        "labels_opened": 0,
        "online_evaluator_calls": 0,
        "network_calls": 0,
        "raw_content_persisted": False,
    }
    report = _self_hashed(DIAGNOSTIC_SCHEMA, body, "diagnostic_sha256")
    _write_json_exclusive(expected_output, report, public=True)
    return report


def _git(project: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(project), *arguments],
        check=False,
        capture_output=True,
        timeout=60,
    )
    if completed.returncode != 0:
        raise QASCCoevolutionError(f"git command failed: {arguments[0]}")
    return completed.stdout


def _head_file_binding(project: Path, path: Path, field: str) -> dict[str, Any]:
    repository = Path(_git(project, "rev-parse", "--show-toplevel").decode().strip())
    actual = path.resolve(strict=True)
    try:
        relative = actual.relative_to(repository).as_posix()
    except ValueError as exc:
        raise QASCCoevolutionError(f"{field} is outside the repository") from exc
    raw = actual.read_bytes()
    if raw != _git(repository, "show", f"HEAD:{relative}"):
        raise QASCCoevolutionError(f"{field} is not the clean HEAD blob")
    if _git(repository, "status", "--porcelain", "--", relative):
        raise QASCCoevolutionError(f"{field} is dirty")
    return {
        "relative_path": actual.relative_to(project).as_posix(),
        "file_sha256": _sha256_bytes(raw),
        "clean_tracked_HEAD_blob": True,
    }


def _current_implementation_binding(project: Path) -> dict[str, Any]:
    acquisition = _acquisition_module()
    try:
        binding = acquisition.implementation_binding(project)
    except Exception as exc:
        raise QASCCoevolutionError("QASC implementation closure is not clean HEAD") from exc
    files = binding.get("files")
    if not isinstance(files, list) or binding.get("set_sha256") != stable_hash(files):
        raise QASCCoevolutionError("QASC implementation binding drifted")
    expected = {
        "assumption_agent/benchmarks/qasc_evaluator_direct_action_acquisition_v1.py",
        "assumption_agent/benchmarks/qasc_counterfactual_chain_margin_v1.py",
        "assumption_agent/benchmarks/qasc_evaluator_direct_action_coevolution_v1.py",
        "replication_runtime/qasc_nli_v1/binding.py",
        "replication_runtime/qasc_nli_v1/contract.py",
        "replication_runtime/qasc_nli_v1/worker.py",
    }
    if not expected.issubset({row.get("path") for row in files if isinstance(row, Mapping)}):
        raise QASCCoevolutionError("QASC implementation closure is incomplete")
    return dict(binding)


def _load_diagnostic(
    project: Path, path: str | Path, *, require_committed: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected = (project / DIAGNOSTIC_RELATIVE).absolute()
    if Path(path).absolute() != expected:
        raise QASCCoevolutionError("infrastructure diagnostic path is noncanonical")
    report, raw = _read_json(expected, "infrastructure diagnostic")
    declared = _verify_self_hash(
        report, schema=DIAGNOSTIC_SCHEMA, field="diagnostic_sha256"
    )
    synthetic = report.get("synthetic_recipe")
    canary = report.get("nli_canary")
    if (
        report.get("status")
        != "passed_row_free_QASC_infrastructure_diagnostic"
        or report.get("design_binding", {}).get("design_sha256") != DESIGN_SHA256
        or report.get("implementation_binding") != _current_implementation_binding(project)
        or report.get("formal_QA_rows_read") != 0
        or report.get("labels_opened") != 0
        or not isinstance(synthetic, Mapping)
        or synthetic.get("document_count") != DOCUMENT_COUNT
        or synthetic.get("choice_count") != CHOICE_COUNT
        or synthetic.get("recipe_action_terminal_count") != RECIPE_COUNT
        or synthetic.get("two_score_waves_exact") is not True
        or not isinstance(canary, Mapping)
        or canary.get("worker_count") != NLI_WORKERS
        or canary.get("torch_threads_per_worker") != TORCH_THREADS_PER_WORKER
        or canary.get("status")
        != "passed_exact_shape_8_worker_repeat_equality_and_capacity"
        or report.get("raw_content_persisted") is not False
    ):
        raise QASCCoevolutionError("infrastructure diagnostic contract drifted")
    binding = {
        "relative_path": DIAGNOSTIC_RELATIVE,
        "file_sha256": _sha256_bytes(raw),
        "diagnostic_sha256": declared,
    }
    if require_committed:
        custody = _head_file_binding(project, expected, "infrastructure diagnostic")
        if custody["file_sha256"] != binding["file_sha256"]:
            raise QASCCoevolutionError("infrastructure diagnostic custody drifted")
        binding.update({"clean_tracked_HEAD_blob": True})
    _assert_public_safe(report)
    return report, binding


def _load_acquisition_live(
    *, project: Path, receipt_path: str | Path, selection_secret_path: str | Path
) -> tuple[dict[str, Any], bytes, dict[str, object]]:
    acquisition = _acquisition_module()
    try:
        receipt, rows = acquisition.load_acquisition_binding_live(
            project=project,
            receipt_path=Path(receipt_path),
            selection_secret_path=Path(selection_secret_path),
        )
    except Exception as exc:
        raise QASCCoevolutionError("live QASC acquisition binding drifted") from exc
    canonical = project / acquisition.ACQUISITION_RELATIVE
    raw = canonical.read_bytes()
    commitments = {row.block: row for row in rows}
    if tuple(commitments) != tuple(BLOCK_COUNTS) or any(
        getattr(commitments[block], "count", None) != count
        for block, count in BLOCK_COUNTS.items()
    ):
        raise QASCCoevolutionError("QASC acquisition block closure drifted")
    return dict(receipt), raw, commitments


def _source_binding(
    receipt: Mapping[str, Any], raw: bytes, commitment: object
) -> dict[str, Any]:
    row = _object_payload(commitment)
    return {
        "acquisition_sha256": _require_sha256(
            receipt.get("acquisition_sha256"), "acquisition hash"
        ),
        "acquisition_file_sha256": _sha256_bytes(raw),
        "private_pack_sha256": _require_sha256(
            receipt.get("commitments", {}).get("private_pack_sha256"),
            "private pack hash",
        ),
        "block_id_sha256": stable_hash({"block": row["block"]}),
        "source_member_sha256": stable_hash({"source_member": row["source_member"]}),
        "item_count": row["count"],
        "view_file_sha256": _require_sha256(row["view_file_sha256"], "view file"),
        "label_file_sha256": _require_sha256(row["label_file_sha256"], "label file"),
        "view_commitment_set_sha256": _require_sha256(
            row["view_commitment_set_sha256"], "view commitment set"
        ),
        "label_commitment_set_sha256": _require_sha256(
            row["label_commitment_set_sha256"], "label commitment set"
        ),
        "joined_commitment_set_sha256": _require_sha256(
            row["joined_commitment_set_sha256"], "joined commitment set"
        ),
        "private_paths_persisted_publicly": False,
    }


def _selection_secret(
    *, project: Path, supplied: str | Path, receipt: Mapping[str, Any]
) -> bytes:
    acquisition = _acquisition_module()
    path = Path(supplied)
    if not path.is_absolute():
        path = project / path
    try:
        secret = acquisition.load_selection_secret(
            project=project,
            selection_secret_path=path.absolute(),
        )
    except Exception as exc:
        raise QASCCoevolutionError("selection secret path or encoding drifted") from exc
    if _sha256_bytes(secret) != receipt.get("selection", {}).get(
        "selection_secret_commitment_sha256"
    ):
        raise QASCCoevolutionError("selection secret identity drifted")
    return secret


def _canonical_public_output(
    project: Path, supplied: str | Path, relative: str, field: str
) -> Path:
    expected = (project / relative).absolute()
    candidate = Path(supplied)
    if not candidate.is_absolute():
        candidate = project / candidate
    candidate = candidate.absolute()
    if candidate != expected or candidate.exists() or candidate.is_symlink():
        raise QASCCoevolutionError(f"{field} must use its fresh canonical path")
    return expected


def _canonical_stage_root(project: Path, supplied: str | Path, relative: str) -> Path:
    expected = (project / relative).absolute()
    candidate = Path(supplied)
    if not candidate.is_absolute():
        candidate = project / candidate
    candidate = candidate.absolute()
    if candidate != expected or candidate.is_symlink():
        raise QASCCoevolutionError("formal execution root path drifted")
    ignored = subprocess.run(
        ["git", "-C", str(project), "check-ignore", "-q", "--", str(candidate)],
        check=False,
        capture_output=True,
        timeout=30,
    )
    if ignored.returncode != 0:
        raise QASCCoevolutionError("formal execution root must be git ignored")
    return expected


def _root_sha256(path: Path) -> str:
    return stable_hash({"absolute_execution_root": str(path)})


def _recipe_registry_binding() -> dict[str, Any]:
    recipe = _recipe_module()
    registry = tuple(recipe.recipe_registry())
    rows = [_object_payload(row) for row in registry]
    if len(rows) != RECIPE_COUNT or len({row["recipe_id"] for row in rows}) != RECIPE_COUNT:
        raise QASCCoevolutionError("recipe registry drifted")
    return {
        "candidate_recipe_count": RECIPE_COUNT,
        "recipe_ids": [row["recipe_id"] for row in rows],
        "recipe_registry_sha256": stable_hash(rows),
    }


def build_formation_freeze(
    *,
    project_root: str | Path,
    diagnostic_path: str | Path,
    acquisition_receipt_path: str | Path,
    selection_secret_path: str | Path,
    nli_model_path: str | Path,
    execution_root: str | Path,
    authorization_hash: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Freeze formation without reading a single private view or label byte."""

    project = Path(project_root).resolve(strict=True)
    output = _canonical_public_output(
        project, output_path, FORMATION_FREEZE_RELATIVE, "formation freeze"
    )
    root = _canonical_stage_root(project, execution_root, FORMATION_ROOT_RELATIVE)
    if root.exists():
        raise QASCCoevolutionError("formation execution root already exists")
    _design, design_binding = _load_design(project)
    _diagnostic, diagnostic_binding = _load_diagnostic(
        project, diagnostic_path, require_committed=True
    )
    receipt, raw, commitments = _load_acquisition_live(
        project=project,
        receipt_path=acquisition_receipt_path,
        selection_secret_path=selection_secret_path,
    )
    _selection_secret(
        project=project, supplied=selection_secret_path, receipt=receipt
    )
    nli_binding = verify_runtime_asset(project, nli_model_path)
    body = {
        "decision": "authorize_A_form_and_F_search_interleaved_two_wave_formation_once",
        "design_binding": design_binding,
        "diagnostic_binding": diagnostic_binding,
        "implementation_binding": _current_implementation_binding(project),
        "nli_runtime_binding_sha256": stable_hash(nli_binding),
        "recipe_registry_binding": _recipe_registry_binding(),
        "source_bindings": {
            block: _source_binding(receipt, raw, commitments[block])
            for block in FORMATION_BLOCKS
        },
        "execution_contract": {
            "block_order": list(FORMATION_BLOCKS),
            "item_count_per_block": 64,
            "interleaving": "A_form_ordinal_then_F_search_same_ordinal",
            "NLI_worker_count": NLI_WORKERS,
            "torch_threads_per_worker": TORCH_THREADS_PER_WORKER,
            "score_wave_count": 2,
            "labels_open_after_all_action_terminals_and_NLI_postflight": True,
            "selection_after_both_blocks_scored": True,
            "same_behavior_terminates_before_A_hold": True,
            "retries": 0,
            "replays": 0,
            "resamples": 0,
        },
        "authorization_hash": _require_sha256(
            authorization_hash, "formation authorization"
        ),
        "execution_root_sha256": _root_sha256(root),
        "ordering": {
            "A_form_view_rows_read": 0,
            "F_search_view_rows_read": 0,
            "formation_label_rows_read": 0,
            "A_hold_rows_read": 0,
            "M_search_rows_read": 0,
            "freeze_complete_before_private_view_open": True,
        },
        "raw_content_persisted": False,
    }
    freeze = _self_hashed(
        FORMATION_FREEZE_SCHEMA, body, "freeze_sha256"
    )
    _write_json_exclusive(output, freeze, public=True)
    return freeze


def _load_committed_freeze(
    *, project: Path, path: str | Path, relative: str, schema: str
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    expected = (project / relative).absolute()
    if Path(path).absolute() != expected:
        raise QASCCoevolutionError("pre-run freeze path is noncanonical")
    freeze, raw = _read_json(expected, "pre-run freeze")
    _verify_self_hash(freeze, schema=schema, field="freeze_sha256")
    if (
        freeze.get("design_binding", {}).get("design_sha256") != DESIGN_SHA256
        or freeze.get("implementation_binding") != _current_implementation_binding(project)
        or freeze.get("raw_content_persisted") is not False
    ):
        raise QASCCoevolutionError("pre-run freeze closure drifted")
    custody = _head_file_binding(project, expected, "pre-run freeze")
    if custody["file_sha256"] != _sha256_bytes(raw):
        raise QASCCoevolutionError("pre-run freeze custody drifted")
    _assert_public_safe(freeze)
    return freeze, raw, custody


def _load_block_views(
    *, project: Path, block: str, commitment: object
) -> tuple[object, ...]:
    acquisition = _acquisition_module()
    recipe = _recipe_module()
    path = project / acquisition.PRIVATE_PACK_ROOT_RELATIVE / f"{block}.views.jsonl"
    try:
        rows = acquisition.load_private_views(
            view_path=path,
            commitment=commitment,
            expected_block=block,
        )
        views = tuple(recipe.load_retrieval_view(row) for row in rows)
    except Exception as exc:
        raise QASCCoevolutionError(f"{block} gold-free view load failed") from exc
    if len(views) != BLOCK_COUNTS[block] or len(
        {_view_item_key(view) for view in views}
    ) != BLOCK_COUNTS[block]:
        raise QASCCoevolutionError(f"{block} gold-free view closure drifted")
    return views


def _load_block_labels_after_terminals(
    *,
    project: Path,
    block: str,
    commitment: object,
    views: Sequence[object],
) -> tuple[object, ...]:
    acquisition = _acquisition_module()
    recipe = _recipe_module()
    label_path = project / acquisition.PRIVATE_PACK_ROOT_RELATIVE / f"{block}.labels.jsonl"
    try:
        raw_labels = acquisition.load_private_labels(
            label_path=label_path,
            commitment=commitment,
            expected_block=block,
        )
        joined = acquisition.join_private_block(
            views=[_view_payload(view) for view in views],
            labels=raw_labels,
            commitment=commitment,
            expected_block=block,
        )
        by_view = {stable_hash(view_row): label_row for view_row, label_row in joined}
        labels = tuple(
            recipe.load_label_envelope(by_view[_view_item_key(view)]) for view in views
        )
        for view, label in zip(views, labels):
            recipe.validate_view_label_binding(view, label)
    except Exception as exc:
        raise QASCCoevolutionError(f"{block} authorized label join failed") from exc
    if len(labels) != len(views):
        raise QASCCoevolutionError(f"{block} label closure drifted")
    return labels


def _interleave_formation_views(
    a_views: Sequence[object], f_views: Sequence[object]
) -> tuple[object, ...]:
    if len(a_views) != 64 or len(f_views) != 64:
        raise QASCCoevolutionError("formation block shape drifted")
    return tuple(
        view
        for ordinal in range(64)
        for view in (a_views[ordinal], f_views[ordinal])
    )


def _score_formation_block(
    *,
    views: Sequence[object],
    labels: Sequence[object],
    actions: Mapping[str, Sequence[object]],
    selection_secret: bytes,
    block: str,
) -> tuple[object, dict[str, tuple[object, ...]], dict[str, int]]:
    recipe = _recipe_module()
    registry_ids = tuple(row.recipe_id for row in recipe.recipe_registry())
    evidence: dict[str, list[object]] = {recipe_id: [] for recipe_id in registry_ids}
    identities: list[str] = []
    for view, label in zip(views, labels):
        view_actions = tuple(actions[_view_item_key(view)])
        if tuple(action.recipe_id for action in view_actions) != registry_ids:
            raise QASCCoevolutionError("formation recipe terminal order drifted")
        identities.append(label.identity_commitment_sha256)
        for action in view_actions:
            evidence[action.recipe_id].append(
                recipe.score_recipe_action(view, action, label)
            )
    fold_map = recipe.assign_hmac_folds(
        identities,
        selection_secret,
        block=block,
    )
    selection = recipe.select_formation_recipes(
        {key: tuple(rows) for key, rows in evidence.items()}, fold_map
    )
    return selection, {key: tuple(rows) for key, rows in evidence.items()}, dict(fold_map)


def _formation_selection_public(selection: object) -> dict[str, Any]:
    payload = _object_payload(selection)
    required = {
        "incumbent_recipe_id",
        "challenger_recipe_id",
        "incumbent_key",
        "challenger_key",
        "same_behavior",
    }
    if set(payload) != required:
        raise QASCCoevolutionError("formation selection payload drifted")
    return payload


def execute_formation(
    *,
    project_root: str | Path,
    pre_run_freeze_path: str | Path,
    diagnostic_path: str | Path,
    acquisition_receipt_path: str | Path,
    selection_secret_path: str | Path,
    nli_model_path: str | Path,
    execution_root: str | Path,
    public_receipt_path: str | Path,
) -> dict[str, Any]:
    """Consume formation once, run both blocks, then and only then open labels."""

    if _CLEAN_MODULE_CLI_ACTIVE is not True:
        raise QASCCoevolutionError("formal formation is available only through clean CLI")
    project = Path(project_root).resolve(strict=True)
    freeze, freeze_raw, _freeze_custody = _load_committed_freeze(
        project=project,
        path=pre_run_freeze_path,
        relative=FORMATION_FREEZE_RELATIVE,
        schema=FORMATION_FREEZE_SCHEMA,
    )
    output = _canonical_public_output(
        project, public_receipt_path, FORMATION_RECEIPT_RELATIVE, "formation receipt"
    )
    root = _canonical_stage_root(project, execution_root, FORMATION_ROOT_RELATIVE)
    if freeze.get("execution_root_sha256") != _root_sha256(root) or root.exists():
        raise QASCCoevolutionError("formation execution root binding drifted")
    _diagnostic, diagnostic_binding = _load_diagnostic(
        project, diagnostic_path, require_committed=True
    )
    receipt, acquisition_raw, commitments = _load_acquisition_live(
        project=project,
        receipt_path=acquisition_receipt_path,
        selection_secret_path=selection_secret_path,
    )
    secret = _selection_secret(
        project=project, supplied=selection_secret_path, receipt=receipt
    )
    expected_sources = {
        block: _source_binding(receipt, acquisition_raw, commitments[block])
        for block in FORMATION_BLOCKS
    }
    nli_binding = verify_runtime_asset(project, nli_model_path)
    if (
        freeze.get("diagnostic_binding") != diagnostic_binding
        or freeze.get("source_bindings") != expected_sources
        or freeze.get("nli_runtime_binding_sha256") != stable_hash(nli_binding)
        or freeze.get("recipe_registry_binding") != _recipe_registry_binding()
    ):
        raise QASCCoevolutionError("formation frozen inputs drifted")
    _consume_authorization(
        root=root,
        stage="formation",
        freeze_sha256=freeze["freeze_sha256"],
        authorization_hash=freeze["authorization_hash"],
    )
    stage = "open_gold_free_formation_views"
    try:
        a_views = _load_block_views(
            project=project, block="A_form", commitment=commitments["A_form"]
        )
        f_views = _load_block_views(
            project=project, block="F_search", commitment=commitments["F_search"]
        )
        interleaved = _interleave_formation_views(a_views, f_views)
        stage = "two_global_NLI_score_waves"
        with NLIWorkerPool(
            nli_model_path,
            workers=NLI_WORKERS,
            project_root=project,
        ) as pool:
            actions, execution = _score_recipe_views_two_waves(
                views=interleaved,
                recipe_ids=None,
                pool=pool,
            )
        stage = "NLI_runtime_postflight_before_labels"
        if verify_runtime_asset(project, nli_model_path) != nli_binding:
            raise QASCCoevolutionError("formation NLI runtime postflight drifted")
        stage = "open_labels_after_all_formation_action_terminals"
        a_labels = _load_block_labels_after_terminals(
            project=project,
            block="A_form",
            commitment=commitments["A_form"],
            views=a_views,
        )
        f_labels = _load_block_labels_after_terminals(
            project=project,
            block="F_search",
            commitment=commitments["F_search"],
            views=f_views,
        )
        stage = "offline_formation_scoring_and_selection"
        a_selection, a_evidence, a_folds = _score_formation_block(
            views=a_views,
            labels=a_labels,
            actions=actions,
            selection_secret=secret,
            block="A_form",
        )
        f_selection, f_evidence, f_folds = _score_formation_block(
            views=f_views,
            labels=f_labels,
            actions=actions,
            selection_secret=secret,
            block="F_search",
        )
        a_public = _formation_selection_public(a_selection)
        f_public = _formation_selection_public(f_selection)
        same_behavior = bool(a_public["same_behavior"] or f_public["same_behavior"])
        evidence_body = {
            "schema": f"{VERSION}_formation_private_evidence",
            "freeze_sha256": freeze["freeze_sha256"],
            "blocks": {
                "A_form": {
                    "fold_by_identity": a_folds,
                    "evidence_by_recipe": {
                        recipe_id: [_object_payload(row) for row in rows]
                        for recipe_id, rows in a_evidence.items()
                    },
                },
                "F_search": {
                    "fold_by_identity": f_folds,
                    "evidence_by_recipe": {
                        recipe_id: [_object_payload(row) for row in rows]
                        for recipe_id, rows in f_evidence.items()
                    },
                },
            },
            "raw_view_or_label_content_persisted": False,
        }
        evidence = {
            **evidence_body,
            "evidence_sha256": stable_hash(evidence_body),
        }
        evidence_path = project / FORMATION_EVIDENCE_RELATIVE
        _write_json_exclusive(evidence_path, evidence, public=False)
        body = {
            "status": (
                "terminal_formation_unidentifiable_same_behavior"
                if same_behavior
                else "formed_distinct_incumbent_and_challenger_actions"
            ),
            "valid": True,
            "freeze_sha256": freeze["freeze_sha256"],
            "freeze_file_sha256": _sha256_bytes(freeze_raw),
            "design_binding": freeze["design_binding"],
            "diagnostic_binding": freeze["diagnostic_binding"],
            "implementation_binding": freeze["implementation_binding"],
            "recipe_registry_binding": freeze["recipe_registry_binding"],
            "source_bindings": freeze["source_bindings"],
            "selections": {"A_form": a_public, "F_search": f_public},
            "formation_identifiability": {
                "A_form_same_behavior": a_public["same_behavior"],
                "F_search_same_behavior": f_public["same_behavior"],
                "identifiable": not same_behavior,
                "A_hold_authorized": not same_behavior,
                "runner_up_or_objective_change_authorized": False,
            },
            "private_evidence_binding": {
                "file_sha256": _sha256_file(evidence_path),
                "evidence_sha256": evidence["evidence_sha256"],
                "item_level_evidence_persisted_publicly": False,
            },
            "execution": {
                **execution,
                "block_item_counts": {"A_form": 64, "F_search": 64},
                "interleaved_A_form_F_search": True,
                "NLI_worker_count": NLI_WORKERS,
                "torch_threads_per_worker": TORCH_THREADS_PER_WORKER,
                "NLI_postflight_before_label_open": True,
                "all_action_terminals_joined_before_label_open": True,
                "label_rows_opened_after_terminals": 128,
                "retries": 0,
                "replays": 0,
                "resamples": 0,
                "network_calls": 0,
                "online_evaluator_calls": 0,
            },
            "raw_content_persisted": False,
        }
        public = _self_hashed(
            FORMATION_RECEIPT_SCHEMA, body, "receipt_sha256"
        )
        _write_json_exclusive(output, public, public=True)
        return public
    except Exception as exc:
        failure_body = {
            "schema": FAILURE_SCHEMA,
            "stage": "formation",
            "freeze_sha256": freeze["freeze_sha256"],
            "failure_stage": stage,
            "error_type_sha256": stable_hash({"error_type": type(exc).__name__}),
            "authorization_consumed": True,
            "replay_authorized": False,
            "raw_content_persisted": False,
        }
        try:
            _write_json_exclusive(
                root / "formation.failure.json",
                {**failure_body, "failure_sha256": stable_hash(failure_body)},
                public=False,
            )
        except Exception:
            pass
        raise QASCCoevolutionError("formal formation failed and cohort is burned") from exc


def _scored_item_from_payload(payload: Mapping[str, Any]) -> object:
    recipe = _recipe_module()
    normalized = dict(payload)
    normalized["ordered_top5"] = tuple(normalized.get("ordered_top5", ()))
    try:
        return recipe.ScoredRecipeItem(**normalized)
    except TypeError as exc:
        raise QASCCoevolutionError("private scored recipe evidence is malformed") from exc


def _verify_formation_execution(execution: object) -> None:
    if not isinstance(execution, Mapping):
        raise QASCCoevolutionError("formation execution receipt is missing")
    required = {
        "view_count": sum(BLOCK_COUNTS[block] for block in FORMATION_BLOCKS),
        "recipe_count_per_view": RECIPE_COUNT,
        "first_wave_item_terminal_count": 128,
        "second_wave_item_terminal_count": 128,
        "recipe_action_terminal_count": 128 * RECIPE_COUNT,
        "all_first_wave_items_submitted_before_first_wave_join": True,
        "second_wave_built_only_after_complete_first_wave_join": True,
        "all_second_wave_items_submitted_before_second_wave_join": True,
        "labels_loaded_or_scored": False,
        "two_score_waves_exact": True,
        "block_item_counts": {"A_form": 64, "F_search": 64},
        "interleaved_A_form_F_search": True,
        "NLI_worker_count": NLI_WORKERS,
        "torch_threads_per_worker": TORCH_THREADS_PER_WORKER,
        "NLI_postflight_before_label_open": True,
        "all_action_terminals_joined_before_label_open": True,
        "label_rows_opened_after_terminals": 128,
        "retries": 0,
        "replays": 0,
        "resamples": 0,
        "network_calls": 0,
        "online_evaluator_calls": 0,
    }
    if any(execution.get(key) != expected for key, expected in required.items()):
        raise QASCCoevolutionError("formation execution invariant drifted")
    conceptual = {
        "first_wave_conceptual_request_count": 524288,
        "second_wave_conceptual_request_count": 2031616,
    }
    if any(execution.get(field) != count for field, count in conceptual.items()):
        raise QASCCoevolutionError("formation conceptual equal-compute drifted")
    for actual_field, conceptual_field in (
        (
            "first_wave_actual_NLI_pair_count",
            "first_wave_conceptual_request_count",
        ),
        (
            "second_wave_actual_NLI_pair_count",
            "second_wave_conceptual_request_count",
        ),
    ):
        actual = execution.get(actual_field)
        if type(actual) is not int or not 0 < actual <= execution[conceptual_field]:
            raise QASCCoevolutionError("formation actual NLI work count drifted")


def reverify_formation_receipt(
    *,
    project_root: str | Path,
    pre_run_freeze_path: str | Path,
    public_receipt_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recompute both selections from ignored item evidence."""

    project = Path(project_root).resolve(strict=True)
    freeze, freeze_raw, _custody = _load_committed_freeze(
        project=project,
        path=pre_run_freeze_path,
        relative=FORMATION_FREEZE_RELATIVE,
        schema=FORMATION_FREEZE_SCHEMA,
    )
    expected = (project / FORMATION_RECEIPT_RELATIVE).absolute()
    if Path(public_receipt_path).absolute() != expected:
        raise QASCCoevolutionError("formation receipt path is noncanonical")
    receipt, raw = _read_json(expected, "formation receipt")
    declared = _verify_self_hash(
        receipt, schema=FORMATION_RECEIPT_SCHEMA, field="receipt_sha256"
    )
    if (
        receipt.get("valid") is not True
        or receipt.get("freeze_sha256") != freeze["freeze_sha256"]
        or receipt.get("freeze_file_sha256") != _sha256_bytes(freeze_raw)
        or receipt.get("design_binding") != freeze["design_binding"]
        or receipt.get("diagnostic_binding") != freeze["diagnostic_binding"]
        or receipt.get("implementation_binding") != freeze["implementation_binding"]
        or receipt.get("recipe_registry_binding") != freeze["recipe_registry_binding"]
        or receipt.get("source_bindings") != freeze["source_bindings"]
        or receipt.get("raw_content_persisted") is not False
    ):
        raise QASCCoevolutionError("formation public receipt binding drifted")
    evidence_path = project / FORMATION_EVIDENCE_RELATIVE
    evidence, evidence_raw = _read_json(evidence_path, "formation private evidence")
    evidence_body = dict(evidence)
    evidence_hash = _require_sha256(
        evidence_body.pop("evidence_sha256", None), "formation evidence hash"
    )
    if (
        stable_hash(evidence_body) != evidence_hash
        or evidence.get("freeze_sha256") != freeze["freeze_sha256"]
        or receipt.get("private_evidence_binding")
        != {
            "file_sha256": _sha256_bytes(evidence_raw),
            "evidence_sha256": evidence_hash,
            "item_level_evidence_persisted_publicly": False,
        }
    ):
        raise QASCCoevolutionError("formation private evidence binding drifted")
    recipe = _recipe_module()
    recomputed: dict[str, dict[str, Any]] = {}
    for block in FORMATION_BLOCKS:
        private = evidence.get("blocks", {}).get(block)
        if not isinstance(private, Mapping):
            raise QASCCoevolutionError("formation private block evidence is missing")
        raw_by_recipe = private.get("evidence_by_recipe")
        folds = private.get("fold_by_identity")
        if not isinstance(raw_by_recipe, Mapping) or not isinstance(folds, Mapping):
            raise QASCCoevolutionError("formation private selection inputs are malformed")
        by_recipe = {
            recipe_id: tuple(_scored_item_from_payload(row) for row in rows)
            for recipe_id, rows in raw_by_recipe.items()
        }
        selection = recipe.select_formation_recipes(by_recipe, folds)
        recomputed[block] = _formation_selection_public(selection)
    if receipt.get("selections") != recomputed:
        raise QASCCoevolutionError("formation selections do not match private evidence")
    _verify_formation_execution(receipt.get("execution"))
    same = bool(
        recomputed["A_form"]["same_behavior"]
        or recomputed["F_search"]["same_behavior"]
    )
    identifiability = receipt.get("formation_identifiability")
    if (
        not isinstance(identifiability, Mapping)
        or identifiability.get("identifiable") is not (not same)
        or identifiability.get("A_hold_authorized") is not (not same)
        or identifiability.get("runner_up_or_objective_change_authorized") is not False
    ):
        raise QASCCoevolutionError("formation identifiability disposition drifted")
    custody = _head_file_binding(project, expected, "formation receipt")
    if custody["file_sha256"] != _sha256_bytes(raw):
        raise QASCCoevolutionError("formation receipt custody drifted")
    _assert_public_safe(receipt)
    return receipt, {
        "relative_path": FORMATION_RECEIPT_RELATIVE,
        "file_sha256": _sha256_bytes(raw),
        "receipt_sha256": declared,
        "private_evidence_file_sha256": _sha256_bytes(evidence_raw),
        "private_evidence_sha256": evidence_hash,
        "clean_tracked_HEAD_blob": True,
    }


def _selected_action_binding(formation: Mapping[str, Any]) -> dict[str, Any]:
    selections = formation.get("selections")
    if not isinstance(selections, Mapping):
        raise QASCCoevolutionError("formation selections are unavailable")
    result: dict[str, Any] = {}
    for block in FORMATION_BLOCKS:
        row = selections.get(block)
        if not isinstance(row, Mapping):
            raise QASCCoevolutionError("formation selection is malformed")
        incumbent = row.get("incumbent_recipe_id")
        challenger = row.get("challenger_recipe_id")
        if (
            not isinstance(incumbent, str)
            or not isinstance(challenger, str)
            or incumbent == challenger
        ):
            raise QASCCoevolutionError("formation selected recipe pair drifted")
        result[block] = {
            "incumbent_recipe_id": incumbent,
            "challenger_recipe_id": challenger,
            "incumbent_key_sha256": stable_hash(row.get("incumbent_key")),
            "challenger_key_sha256": stable_hash(row.get("challenger_key")),
            "same_behavior": row.get("same_behavior"),
        }
    return result


def build_a_hold_freeze(
    *,
    project_root: str | Path,
    diagnostic_path: str | Path,
    formation_freeze_path: str | Path,
    formation_receipt_path: str | Path,
    acquisition_receipt_path: str | Path,
    selection_secret_path: str | Path,
    nli_model_path: str | Path,
    capability_receipt_path: str | Path,
    runtime_python: str | Path,
    local_llm_model: str | Path,
    local_embedding_model: str | Path,
    base_binding_receipt_path: str | Path,
    attestation_receipt_path: str | Path,
    execution_root: str | Path,
    authorization_hash: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Freeze A_hold from committed acquisition and identifiable formation only."""

    project = Path(project_root).resolve(strict=True)
    output = _canonical_public_output(project, output_path, A_FREEZE_RELATIVE, "A_hold freeze")
    root = _canonical_stage_root(project, execution_root, A_ROOT_RELATIVE)
    if root.exists():
        raise QASCCoevolutionError("A_hold execution root already exists")
    _design, design_binding = _load_design(project)
    _diagnostic, diagnostic_binding = _load_diagnostic(
        project, diagnostic_path, require_committed=True
    )
    formation, formation_binding = reverify_formation_receipt(
        project_root=project,
        pre_run_freeze_path=formation_freeze_path,
        public_receipt_path=formation_receipt_path,
    )
    identifiability = formation.get("formation_identifiability")
    if (
        not isinstance(identifiability, Mapping)
        or identifiability.get("identifiable") is not True
        or identifiability.get("A_hold_authorized") is not True
    ):
        raise QASCCoevolutionError("formation is unidentifiable; A_hold must remain unopened")
    actions = _selected_action_binding(formation)
    if actions["A_form"]["same_behavior"] is not False or actions["F_search"]["same_behavior"] is not False:
        raise QASCCoevolutionError("formation selected coincident actions")
    receipt, raw, commitments = _load_acquisition_live(
        project=project,
        receipt_path=acquisition_receipt_path,
        selection_secret_path=selection_secret_path,
    )
    _selection_secret(project=project, supplied=selection_secret_path, receipt=receipt)
    nli_binding = verify_runtime_asset(project, nli_model_path)
    try:
        retained = _load_retained_p(project)
        retained_binding = {
            "status": "available",
            "program_sha256": retained.program_hash,
            "program_file_sha256": P_PROGRAM_FILE_SHA256,
            "role": "descriptive_control_only",
        }
    except Exception as exc:
        retained_binding = {
            "status": "preflight_unavailable",
            "error_type_sha256": stable_hash({"error_type": type(exc).__name__}),
            "role": "descriptive_control_only",
        }
    try:
        official_binding, _prepared = _prepare_official_runtime(
            project=project,
            capability_receipt_path=capability_receipt_path,
            runtime_python=runtime_python,
            local_llm_model=local_llm_model,
            local_embedding_model=local_embedding_model,
            base_binding_receipt_path=base_binding_receipt_path,
            attestation_receipt_path=attestation_receipt_path,
        )
        official_binding = {"status": "available", **official_binding}
    except Exception as exc:
        official_binding = {
            "status": "preflight_unavailable",
            "error_type_sha256": stable_hash({"error_type": type(exc).__name__}),
            "role": "descriptive_control_only",
        }
    body = {
        "decision": "authorize_one_shot_A_hold_after_identifiable_formation",
        "design_binding": design_binding,
        "diagnostic_binding": diagnostic_binding,
        "implementation_binding": _current_implementation_binding(project),
        "formation_binding": formation_binding,
        "source_binding": _source_binding(receipt, raw, commitments["A_hold"]),
        "nli_runtime_binding_sha256": stable_hash(nli_binding),
        "official_runtime_binding": official_binding,
        "retained_P_binding": retained_binding,
        "A_action_binding": actions["A_form"],
        "prospective_M_action_binding": {
            **actions["F_search"],
            "frozen_before_A_hold_open": True,
            "challenger_maps_to_active_only_if_A_promotes": True,
        },
        "execution_contract": {
            "item_count": 64,
            "primary_recipe_count": 2,
            "NLI_score_wave_count": 2,
            "NLI_worker_count": NLI_WORKERS,
            "controls": ["canonical_RAW", "retained_P", "official_HippoRAG_item_local_32"],
            "official_concurrency_cap": OFFICIAL_CONCURRENCY_CAP,
            "controls_descriptive_and_failure_isolated": True,
            "all_primary_action_terminals_and_NLI_postflight_before_labels": True,
            "promotion_test": "one_sided_exact_magnitude_preserving_sign_flip_v1",
            "alpha_numerator": 1,
            "alpha_denominator": 10,
            "retries": 0,
            "replays": 0,
            "resamples": 0,
        },
        "authorization_hash": _require_sha256(authorization_hash, "A authorization"),
        "execution_root_sha256": _root_sha256(root),
        "ordering": {
            "A_hold_view_rows_read": 0,
            "A_hold_label_rows_read": 0,
            "M_search_view_or_label_rows_read": 0,
            "freeze_complete_before_A_hold_open": True,
        },
        "raw_content_persisted": False,
    }
    freeze = _self_hashed(A_FREEZE_SCHEMA, body, "freeze_sha256")
    _write_json_exclusive(output, freeze, public=True)
    return freeze


def _execute_controls(
    *,
    views: Sequence[object],
    retained_p: TypedRetrievalProgram | None,
    prepared: PreparedFormalRuntimeV2 | None,
    root: Path,
) -> tuple[dict[str, ControlOutcome], dict[str, Any]]:
    """Run all controls concurrently; convert every failure to descriptive evidence."""

    def raw_task() -> ControlOutcome:
        return _run_failure_isolated_control(
            control_id="canonical_RAW",
            views=views,
            function=lambda _index, view: tuple(_view_payload(view)["raw_ranking"]),
            maximum_workers=len(views),
        )

    def p_task() -> ControlOutcome:
        if retained_p is None:
            raise QASCCoevolutionError("retained P is unavailable")
        return _run_failure_isolated_control(
            control_id="retained_P",
            views=views,
            function=lambda _index, view: _retained_p_ranking(retained_p, view),
            maximum_workers=len(views),
        )

    def official_task() -> ControlOutcome:
        if prepared is None:
            raise QASCCoevolutionError("official HippoRAG runtime is unavailable")
        (root / "official_control_work").mkdir(mode=0o700)
        return _run_failure_isolated_control(
            control_id="official_HippoRAG_item_local_32",
            views=views,
            function=lambda index, view: prepared.retrieve(
                question=_official_inputs(view)[0],
                paragraphs=_official_inputs(view)[1],
                work_root=root / "official_control_work" / f"item-{index:03d}",
            ),
            maximum_workers=min(OFFICIAL_CONCURRENCY_CAP, len(views)),
        )

    tasks = {"canonical_RAW": raw_task, "retained_P": p_task, "official_HippoRAG_item_local_32": official_task}
    outcomes: dict[str, ControlOutcome] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {name: executor.submit(function) for name, function in tasks.items()}
        for name, future in futures.items():
            try:
                outcomes[name] = future.result()
            except Exception as exc:
                outcomes[name] = ControlOutcome(
                    control_id=name,
                    status="unavailable_or_partial",
                    rankings=(None,) * len(views),
                    failure_type_hashes=(
                        stable_hash({"error_type": type(exc).__name__}),
                    )
                    * len(views),
                )
    try:
        if prepared is None:
            raise QASCCoevolutionError("official HippoRAG runtime is unavailable")
        postflight = prepared.fresh_reverify()
        official_postflight = {
            "passed": True,
            "binding_sha256": stable_hash(postflight),
        }
    except Exception as exc:
        old = outcomes["official_HippoRAG_item_local_32"]
        outcomes["official_HippoRAG_item_local_32"] = ControlOutcome(
            control_id=old.control_id,
            status="unavailable_or_partial",
            rankings=(None,) * len(views),
            failure_type_hashes=(
                stable_hash({"error_type": type(exc).__name__}),
            )
            * len(views),
        )
        official_postflight = {
            "passed": False,
            "failure_type_sha256": stable_hash({"error_type": type(exc).__name__}),
        }
    return outcomes, official_postflight


def _failed_control_bundle(
    *, item_count: int, error: BaseException
) -> tuple[dict[str, ControlOutcome], dict[str, Any]]:
    failure = stable_hash({"error_type": type(error).__name__})
    outcomes = {
        control_id: ControlOutcome(
            control_id=control_id,
            status="unavailable_or_partial",
            rankings=(None,) * item_count,
            failure_type_hashes=(failure,) * item_count,
        )
        for control_id in (
            "canonical_RAW",
            "retained_P",
            "official_HippoRAG_item_local_32",
        )
    }
    return outcomes, {
        "passed": False,
        "failure_type_sha256": failure,
        "control_orchestration_failure_isolated": True,
    }


def _join_controls_failure_isolated(
    future: concurrent.futures.Future[tuple[dict[str, ControlOutcome], dict[str, Any]]],
    *,
    item_count: int,
) -> tuple[dict[str, ControlOutcome], dict[str, Any]]:
    try:
        outcomes, postflight = future.result()
        expected = {
            "canonical_RAW",
            "retained_P",
            "official_HippoRAG_item_local_32",
        }
        if set(outcomes) != expected:
            raise QASCCoevolutionError("control terminal set drifted")
        normalized = {
            control_id: _control_from_private_payload(
                _control_private_payload(outcomes[control_id]),
                expected_id=control_id,
                item_count=item_count,
            )
            for control_id in sorted(expected)
        }
        normalized_postflight = _validate_official_postflight(postflight)
        official = normalized["official_HippoRAG_item_local_32"]
        if normalized_postflight["passed"] is False and (
            official.status != "unavailable_or_partial"
            or any(row is not None for row in official.rankings)
        ):
            raise QASCCoevolutionError("failed official postflight retained rankings")
        return normalized, normalized_postflight
    except Exception as exc:
        return _failed_control_bundle(item_count=item_count, error=exc)


def _score_selected_actions(
    *,
    views: Sequence[object],
    labels: Sequence[object],
    actions: Mapping[str, Sequence[object]],
    incumbent_recipe_id: str,
    challenger_recipe_id: str,
) -> tuple[
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
    tuple[dict[str, Any], ...],
]:
    recipe = _recipe_module()
    incumbent_rankings: list[tuple[int, ...]] = []
    challenger_rankings: list[tuple[int, ...]] = []
    private_rows: list[dict[str, Any]] = []
    for view, label in zip(views, labels):
        action_map = {
            action.recipe_id: action for action in actions[_view_item_key(view)]
        }
        if set(action_map) != {incumbent_recipe_id, challenger_recipe_id}:
            raise QASCCoevolutionError("selected measurement action set drifted")
        incumbent = recipe.score_recipe_action(
            view, action_map[incumbent_recipe_id], label
        )
        challenger = recipe.score_recipe_action(
            view, action_map[challenger_recipe_id], label
        )
        incumbent_rankings.append(tuple(incumbent.ordered_top5))
        challenger_rankings.append(tuple(challenger.ordered_top5))
        private_rows.append(
            {
                "identity_commitment_sha256": label.identity_commitment_sha256,
                "view_sha256": label.view_sha256,
                "gold_document_ids": list(label.gold_document_ids),
                "incumbent": _object_payload(incumbent),
                "challenger": _object_payload(challenger),
            }
        )
    return tuple(incumbent_rankings), tuple(challenger_rankings), tuple(private_rows)


def _control_metrics(
    *,
    outcomes: Mapping[str, ControlOutcome],
    challenger_rankings: Sequence[Sequence[int]],
    gold_rows: Sequence[Sequence[int]],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for control_id, outcome in outcomes.items():
        row: dict[str, Any] = {"availability": outcome.public_summary()}
        if all(ranking is not None for ranking in outcome.rankings):
            control_rankings = tuple(
                ranking for ranking in outcome.rankings if ranking is not None
            )
            row["arm_metrics"] = aggregate_rankings(
                arm_id=control_id,
                rankings=control_rankings,
                gold_rows=gold_rows,
            )
            row["challenger_minus_control"] = paired_utility_comparison(
                left_arm_id="challenger",
                right_arm_id=control_id,
                left_rankings=challenger_rankings,
                right_rankings=control_rankings,
                gold_rows=gold_rows,
                confirmatory=False,
            )
        else:
            row["arm_metrics"] = None
            row["challenger_minus_control"] = None
        metrics[control_id] = row
    return metrics


def _consume_authorization(
    *, root: Path, stage: str, freeze_sha256: str, authorization_hash: str
) -> Path:
    if root.exists() or root.is_symlink():
        raise QASCCoevolutionError("formal execution root exists; replay forbidden")
    _prepare_output_parent(root)
    os.mkdir(root, 0o700)
    body = {
        "schema": CONSUMPTION_SCHEMA,
        "stage": stage,
        "freeze_sha256": _require_sha256(freeze_sha256, "freeze hash"),
        "authorization_hash": _require_sha256(authorization_hash, "authorization hash"),
        "replay_authorized": False,
        "raw_content_persisted": False,
    }
    marker = _self_hashed(CONSUMPTION_SCHEMA, body, "consumption_sha256")
    path = root / "authorization.consumed.json"
    _write_json_exclusive(path, marker, public=False)
    return path


def _control_private_payload(outcome: ControlOutcome) -> dict[str, Any]:
    return {
        "control_id": outcome.control_id,
        "status": outcome.status,
        "rankings": [
            None if ranking is None else list(ranking)
            for ranking in outcome.rankings
        ],
        "failure_type_hashes": list(outcome.failure_type_hashes),
    }


def _control_from_private_payload(
    payload: Mapping[str, Any], *, expected_id: str, item_count: int
) -> ControlOutcome:
    if payload.get("control_id") != expected_id:
        raise QASCCoevolutionError("private control ID drifted")
    raw_rankings = payload.get("rankings")
    raw_failures = payload.get("failure_type_hashes")
    status = payload.get("status")
    if (
        status not in {"available", "unavailable_or_partial"}
        or not isinstance(raw_rankings, list)
        or not isinstance(raw_failures, list)
        or len(raw_rankings) != item_count
        or len(raw_failures) != item_count
    ):
        raise QASCCoevolutionError("private control payload shape drifted")
    rankings: list[tuple[int, ...] | None] = []
    failures: list[str | None] = []
    for ranking, failure in zip(raw_rankings, raw_failures):
        if ranking is None:
            if failure is None:
                raise QASCCoevolutionError("failed control row lacks failure receipt")
            failures.append(_require_sha256(failure, "control failure type"))
            rankings.append(None)
        else:
            if failure is not None or not isinstance(ranking, list):
                raise QASCCoevolutionError("available control row has failure receipt")
            normalized = tuple(ranking)
            item_utility(normalized, (0, 1))
            rankings.append(normalized)
            failures.append(None)
    expected_status = (
        "available" if all(row is not None for row in rankings)
        else "unavailable_or_partial"
    )
    if status != expected_status:
        raise QASCCoevolutionError("private control availability drifted")
    return ControlOutcome(
        control_id=expected_id,
        status=status,
        rankings=tuple(rankings),
        failure_type_hashes=tuple(failures),
    )


def _measurement_decision(
    *, stage: str, freeze_sha256: str, comparison: Mapping[str, Any]
) -> dict[str, Any]:
    if stage not in {"A_hold", "M_search"}:
        raise QASCCoevolutionError("measurement decision stage drifted")
    test = comparison.get("paired_test")
    if not isinstance(test, Mapping) or comparison.get("confirmatory") is not True:
        raise QASCCoevolutionError("primary paired test is unavailable")
    passed = bool(test.get("promoted"))
    body = {
        "schema": f"{VERSION}_{stage}_decision",
        "stage": stage,
        "freeze_sha256": _require_sha256(freeze_sha256, "freeze hash"),
        "primary_comparison_sha256": stable_hash(comparison),
        "observed_net_U": test.get("observed_net_U"),
        "p_value_numerator": test.get("p_value_numerator"),
        "p_value_denominator": test.get("p_value_denominator"),
        "positive_and_exact_p_at_or_below_alpha": passed,
        "sole_criterion": True,
    }
    field = "A_decision_sha256" if stage == "A_hold" else "M_decision_sha256"
    return {**body, field: stable_hash(body)}


def _measurement_private_evidence(
    *,
    schema: str,
    freeze_sha256: str,
    block: str,
    primary_rows: Sequence[Mapping[str, Any]],
    controls: Mapping[str, ControlOutcome],
    official_postflight: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "schema": schema,
        "freeze_sha256": freeze_sha256,
        "block": block,
        "primary_rows": [dict(row) for row in primary_rows],
        "controls": {
            control_id: _control_private_payload(outcome)
            for control_id, outcome in sorted(controls.items())
        },
        "official_postflight": dict(official_postflight),
        "raw_view_or_label_content_persisted": False,
    }
    return {**body, "evidence_sha256": stable_hash(body)}


def _validate_official_postflight(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or type(value.get("passed")) is not bool:
        raise QASCCoevolutionError("official postflight evidence is malformed")
    normalized = dict(value)
    if normalized["passed"]:
        if set(normalized) != {"passed", "binding_sha256"}:
            raise QASCCoevolutionError("successful official postflight shape drifted")
        _require_sha256(normalized.get("binding_sha256"), "official postflight")
    else:
        allowed = {
            "passed",
            "failure_type_sha256",
            "control_orchestration_failure_isolated",
        }
        if not set(normalized).issubset(allowed) or set(normalized) < {
            "passed",
            "failure_type_sha256",
        }:
            raise QASCCoevolutionError("failed official postflight shape drifted")
        _require_sha256(
            normalized.get("failure_type_sha256"), "official postflight failure"
        )
        if "control_orchestration_failure_isolated" in normalized and normalized[
            "control_orchestration_failure_isolated"
        ] is not True:
            raise QASCCoevolutionError("control orchestration receipt drifted")
    return normalized


def _load_measurement_evidence(
    *,
    path: Path,
    field: str,
    schema: str,
    freeze_sha256: str,
    block: str,
    expected_incumbent_recipe_id: str,
    expected_challenger_recipe_id: str,
) -> tuple[
    dict[str, Any],
    bytes,
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, int], ...],
    dict[str, ControlOutcome],
    dict[str, Any],
]:
    evidence, raw = _read_json(path, field)
    body = dict(evidence)
    declared = _require_sha256(body.pop("evidence_sha256", None), "evidence hash")
    rows = evidence.get("primary_rows")
    controls_payload = evidence.get("controls")
    official_postflight = _validate_official_postflight(
        evidence.get("official_postflight")
    )
    if (
        evidence.get("schema") != schema
        or stable_hash(body) != declared
        or evidence.get("freeze_sha256") != freeze_sha256
        or evidence.get("block") != block
        or evidence.get("raw_view_or_label_content_persisted") is not False
        or not isinstance(rows, list)
        or len(rows) != BLOCK_COUNTS[block]
        or not isinstance(controls_payload, Mapping)
        or set(controls_payload)
        != {"canonical_RAW", "retained_P", "official_HippoRAG_item_local_32"}
    ):
        raise QASCCoevolutionError(f"{field} closure drifted")
    incumbents: list[tuple[int, ...]] = []
    challengers: list[tuple[int, ...]] = []
    gold_rows: list[tuple[int, int]] = []
    identities: set[str] = set()
    view_hashes: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise QASCCoevolutionError("private primary row is malformed")
        identity = _require_sha256(
            row.get("identity_commitment_sha256"), "private identity"
        )
        view_hash = _require_sha256(row.get("view_sha256"), "private view hash")
        gold = row.get("gold_document_ids")
        incumbent = row.get("incumbent")
        challenger = row.get("challenger")
        if (
            identity in identities
            or view_hash in view_hashes
            or not isinstance(gold, list)
            or not isinstance(incumbent, Mapping)
            or not isinstance(challenger, Mapping)
        ):
            raise QASCCoevolutionError("private primary row binding drifted")
        identities.add(identity)
        view_hashes.add(view_hash)
        normalized_gold = tuple(gold)
        incumbent_ranking = tuple(incumbent.get("ordered_top5", ()))
        challenger_ranking = tuple(challenger.get("ordered_top5", ()))
        incumbent_metrics = item_utility(incumbent_ranking, normalized_gold)
        challenger_metrics = item_utility(challenger_ranking, normalized_gold)
        for scored, metrics, expected_recipe_id in (
            (incumbent, incumbent_metrics, expected_incumbent_recipe_id),
            (challenger, challenger_metrics, expected_challenger_recipe_id),
        ):
            if (
                scored.get("identity_commitment_sha256") != identity
                or scored.get("view_sha256") != view_hash
                or scored.get("recipe_id") != expected_recipe_id
                or scored.get("invalid") is not False
                or scored.get("support_hits_at_5") != metrics[0]
                or scored.get("complete") is not bool(metrics[1])
                or scored.get("U") != metrics[2]
            ):
                raise QASCCoevolutionError("private scored item metrics drifted")
            _require_sha256(scored.get("action_sha256"), "action hash")
            try:
                _recipe_module()._validate_scored_item(  # noqa: SLF001
                    _scored_item_from_payload(scored), expected_recipe_id
                )
            except Exception as exc:
                raise QASCCoevolutionError(
                    "private scored item full contract drifted"
                ) from exc
        gold_rows.append(normalized_gold)  # type: ignore[arg-type]
        incumbents.append(incumbent_ranking)
        challengers.append(challenger_ranking)
    controls = {
        control_id: _control_from_private_payload(
            payload,
            expected_id=control_id,
            item_count=BLOCK_COUNTS[block],
        )
        for control_id, payload in controls_payload.items()
        if isinstance(payload, Mapping)
    }
    if len(controls) != 3:
        raise QASCCoevolutionError("private control evidence is malformed")
    official_outcome = controls["official_HippoRAG_item_local_32"]
    if official_postflight["passed"] is False and (
        official_outcome.status != "unavailable_or_partial"
        or any(row is not None for row in official_outcome.rankings)
    ):
        raise QASCCoevolutionError(
            "failed official postflight did not invalidate its descriptive rows"
        )
    return (
        evidence,
        raw,
        tuple(incumbents),
        tuple(challengers),
        tuple(gold_rows),
        controls,
        official_postflight,
    )


def _verify_measurement_execution(
    *,
    execution: object,
    block: str,
    official_postflight: Mapping[str, Any],
) -> None:
    if not isinstance(execution, Mapping):
        raise QASCCoevolutionError("measurement execution receipt is missing")
    required = {
        "view_count": BLOCK_COUNTS[block],
        "recipe_count_per_view": 2,
        "first_wave_item_terminal_count": BLOCK_COUNTS[block],
        "second_wave_item_terminal_count": BLOCK_COUNTS[block],
        "recipe_action_terminal_count": 2 * BLOCK_COUNTS[block],
        "all_first_wave_items_submitted_before_first_wave_join": True,
        "second_wave_built_only_after_complete_first_wave_join": True,
        "all_second_wave_items_submitted_before_second_wave_join": True,
        "labels_loaded_or_scored": False,
        "two_score_waves_exact": True,
        "block": block,
        "item_count": BLOCK_COUNTS[block],
        "NLI_worker_count": NLI_WORKERS,
        "torch_threads_per_worker": TORCH_THREADS_PER_WORKER,
        "NLI_postflight_before_label_open": True,
        "all_primary_action_terminals_before_label_open": True,
        "control_actions_joined_before_label_open": True,
        "label_rows_opened_after_terminals": BLOCK_COUNTS[block],
        "controls_failure_isolated": True,
        "network_calls": 0,
        "online_evaluator_calls": 0,
        "retries": 0,
        "replays": 0,
        "resamples": 0,
    }
    if any(execution.get(key) != expected for key, expected in required.items()):
        raise QASCCoevolutionError("measurement execution invariant drifted")
    if execution.get("official_postflight") != dict(official_postflight):
        raise QASCCoevolutionError("public/private official postflight drifted")
    conceptual = {
        "first_wave_conceptual_request_count": 32768,
        "second_wave_conceptual_request_count": 126976,
    }
    if any(execution.get(field) != count for field, count in conceptual.items()):
        raise QASCCoevolutionError("measurement conceptual equal-compute drifted")
    for actual_field, conceptual_field in (
        (
            "first_wave_actual_NLI_pair_count",
            "first_wave_conceptual_request_count",
        ),
        (
            "second_wave_actual_NLI_pair_count",
            "second_wave_conceptual_request_count",
        ),
    ):
        actual = execution.get(actual_field)
        if type(actual) is not int or not 0 < actual <= execution[conceptual_field]:
            raise QASCCoevolutionError("measurement actual NLI work count drifted")
    if block == "A_hold" and execution.get("M_search_view_or_label_rows_opened") != 0:
        raise QASCCoevolutionError("A_hold opened M_search")


def _verify_frozen_control_evidence(
    *,
    freeze: Mapping[str, Any],
    outcomes: Mapping[str, ControlOutcome],
    official_postflight: Mapping[str, Any],
) -> None:
    retained_binding = freeze.get("retained_P_binding")
    official_binding = freeze.get("official_runtime_binding")
    if not isinstance(retained_binding, Mapping) or not isinstance(
        official_binding, Mapping
    ):
        raise QASCCoevolutionError("frozen control bindings are unavailable")
    retained = outcomes["retained_P"]
    official = outcomes["official_HippoRAG_item_local_32"]
    if retained_binding.get("status") == "preflight_unavailable" and any(
        ranking is not None for ranking in retained.rankings
    ):
        raise QASCCoevolutionError("frozen-unavailable retained P was recovered")
    if official_binding.get("status") == "preflight_unavailable":
        if official_postflight.get("passed") is not False or any(
            ranking is not None for ranking in official.rankings
        ):
            raise QASCCoevolutionError("frozen-unavailable official control was recovered")
    elif official_binding.get("status") == "available":
        if official_postflight.get("passed") is True and official_postflight.get(
            "binding_sha256"
        ) != stable_hash(official_binding.get("formal_runtime_binding")):
            raise QASCCoevolutionError("official postflight frozen binding drifted")
    else:
        raise QASCCoevolutionError("frozen official control status drifted")


def _load_descriptive_runtime_for_execution(
    *,
    project: Path,
    frozen_retained_binding: Mapping[str, Any],
    frozen_official_binding: Mapping[str, Any],
    capability_receipt_path: str | Path,
    runtime_python: str | Path,
    local_llm_model: str | Path,
    local_embedding_model: str | Path,
    base_binding_receipt_path: str | Path,
    attestation_receipt_path: str | Path,
) -> tuple[TypedRetrievalProgram | None, PreparedFormalRuntimeV2 | None]:
    retained: TypedRetrievalProgram | None = None
    prepared: PreparedFormalRuntimeV2 | None = None
    retained_status = frozen_retained_binding.get("status")
    official_status = frozen_official_binding.get("status")
    if retained_status not in {"available", "preflight_unavailable"} or official_status not in {
        "available",
        "preflight_unavailable",
    }:
        raise QASCCoevolutionError("frozen descriptive-control status drifted")
    # A preflight-unavailable control is terminally unavailable for this frozen
    # stage.  It is not retried or opportunistically recovered during execution.
    if retained_status == "available":
        try:
            candidate = _load_retained_p(project)
            current = {
                "status": "available",
                "program_sha256": candidate.program_hash,
                "program_file_sha256": P_PROGRAM_FILE_SHA256,
                "role": "descriptive_control_only",
            }
            if current != frozen_retained_binding:
                raise QASCCoevolutionError("retained P frozen binding drifted")
            retained = candidate
        except Exception:
            retained = None
    if official_status == "available":
        try:
            current_official, candidate_prepared = _prepare_official_runtime(
                project=project,
                capability_receipt_path=capability_receipt_path,
                runtime_python=runtime_python,
                local_llm_model=local_llm_model,
                local_embedding_model=local_embedding_model,
                base_binding_receipt_path=base_binding_receipt_path,
                attestation_receipt_path=attestation_receipt_path,
            )
            current = {"status": "available", **current_official}
            if current != frozen_official_binding:
                raise QASCCoevolutionError("official runtime frozen binding drifted")
            prepared = candidate_prepared
        except Exception:
            prepared = None
    return retained, prepared


def execute_a_hold(
    *,
    project_root: str | Path,
    pre_run_freeze_path: str | Path,
    diagnostic_path: str | Path,
    formation_freeze_path: str | Path,
    formation_receipt_path: str | Path,
    acquisition_receipt_path: str | Path,
    selection_secret_path: str | Path,
    nli_model_path: str | Path,
    capability_receipt_path: str | Path,
    runtime_python: str | Path,
    local_llm_model: str | Path,
    local_embedding_model: str | Path,
    base_binding_receipt_path: str | Path,
    attestation_receipt_path: str | Path,
    execution_root: str | Path,
    public_report_path: str | Path,
) -> dict[str, Any]:
    """Consume A_hold once and make the sole evaluator-promotion decision."""

    if _CLEAN_MODULE_CLI_ACTIVE is not True:
        raise QASCCoevolutionError("formal A_hold is available only through clean CLI")
    project = Path(project_root).resolve(strict=True)
    freeze, freeze_raw, _custody = _load_committed_freeze(
        project=project,
        path=pre_run_freeze_path,
        relative=A_FREEZE_RELATIVE,
        schema=A_FREEZE_SCHEMA,
    )
    output = _canonical_public_output(
        project, public_report_path, A_REPORT_RELATIVE, "A_hold report"
    )
    root = _canonical_stage_root(project, execution_root, A_ROOT_RELATIVE)
    if freeze.get("execution_root_sha256") != _root_sha256(root) or root.exists():
        raise QASCCoevolutionError("A_hold execution root binding drifted")
    _diagnostic, diagnostic_binding = _load_diagnostic(
        project, diagnostic_path, require_committed=True
    )
    formation, formation_binding = reverify_formation_receipt(
        project_root=project,
        pre_run_freeze_path=formation_freeze_path,
        public_receipt_path=formation_receipt_path,
    )
    current_actions = _selected_action_binding(formation)
    receipt, acquisition_raw, commitments = _load_acquisition_live(
        project=project,
        receipt_path=acquisition_receipt_path,
        selection_secret_path=selection_secret_path,
    )
    _selection_secret(project=project, supplied=selection_secret_path, receipt=receipt)
    source = _source_binding(receipt, acquisition_raw, commitments["A_hold"])
    nli_binding = verify_runtime_asset(project, nli_model_path)
    if (
        freeze.get("diagnostic_binding") != diagnostic_binding
        or freeze.get("formation_binding") != formation_binding
        or freeze.get("source_binding") != source
        or freeze.get("A_action_binding") != current_actions["A_form"]
        or freeze.get("prospective_M_action_binding", {}).get("incumbent_recipe_id")
        != current_actions["F_search"]["incumbent_recipe_id"]
        or freeze.get("prospective_M_action_binding", {}).get("challenger_recipe_id")
        != current_actions["F_search"]["challenger_recipe_id"]
        or freeze.get("nli_runtime_binding_sha256") != stable_hash(nli_binding)
    ):
        raise QASCCoevolutionError("A_hold frozen inputs drifted")
    retained, prepared = _load_descriptive_runtime_for_execution(
        project=project,
        frozen_retained_binding=freeze["retained_P_binding"],
        frozen_official_binding=freeze["official_runtime_binding"],
        capability_receipt_path=capability_receipt_path,
        runtime_python=runtime_python,
        local_llm_model=local_llm_model,
        local_embedding_model=local_embedding_model,
        base_binding_receipt_path=base_binding_receipt_path,
        attestation_receipt_path=attestation_receipt_path,
    )
    _consume_authorization(
        root=root,
        stage="A_hold",
        freeze_sha256=freeze["freeze_sha256"],
        authorization_hash=freeze["authorization_hash"],
    )
    stage = "open_gold_free_A_hold_views"
    try:
        views = _load_block_views(
            project=project, block="A_hold", commitment=commitments["A_hold"]
        )
        action_binding = freeze["A_action_binding"]
        recipe_ids = (
            action_binding["incumbent_recipe_id"],
            action_binding["challenger_recipe_id"],
        )
        stage = "parallel_primary_and_descriptive_action_execution"
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            controls_future = executor.submit(
                _execute_controls,
                views=views,
                retained_p=retained,
                prepared=prepared,
                root=root,
            )
            with NLIWorkerPool(
                nli_model_path,
                workers=NLI_WORKERS,
                project_root=project,
            ) as pool:
                actions, execution = _score_recipe_views_two_waves(
                    views=views,
                    recipe_ids=recipe_ids,
                    pool=pool,
                )
            stage = "NLI_runtime_postflight_before_A_labels"
            if verify_runtime_asset(project, nli_model_path) != nli_binding:
                raise QASCCoevolutionError("A_hold NLI runtime postflight drifted")
            outcomes, official_postflight = _join_controls_failure_isolated(
                controls_future,
                item_count=BLOCK_COUNTS["A_hold"],
            )
        stage = "open_A_labels_after_primary_terminals_and_postflight"
        labels = _load_block_labels_after_terminals(
            project=project,
            block="A_hold",
            commitment=commitments["A_hold"],
            views=views,
        )
        incumbent_rankings, challenger_rankings, private_rows = _score_selected_actions(
            views=views,
            labels=labels,
            actions=actions,
            incumbent_recipe_id=recipe_ids[0],
            challenger_recipe_id=recipe_ids[1],
        )
        gold_rows = tuple(tuple(label.gold_document_ids) for label in labels)
        incumbent_metrics = aggregate_rankings(
            arm_id="A_incumbent",
            rankings=incumbent_rankings,
            gold_rows=gold_rows,
        )
        challenger_metrics = aggregate_rankings(
            arm_id="A_challenger",
            rankings=challenger_rankings,
            gold_rows=gold_rows,
        )
        comparison = paired_utility_comparison(
            left_arm_id="A_challenger",
            right_arm_id="A_incumbent",
            left_rankings=challenger_rankings,
            right_rankings=incumbent_rankings,
            gold_rows=gold_rows,
            confirmatory=True,
        )
        decision = _measurement_decision(
            stage="A_hold",
            freeze_sha256=freeze["freeze_sha256"],
            comparison=comparison,
        )
        transition = evaluator_epoch_transition(
            a_decision_sha256=decision["A_decision_sha256"],
            promoted=decision["positive_and_exact_p_at_or_below_alpha"],
        )
        control_metrics = _control_metrics(
            outcomes=outcomes,
            challenger_rankings=challenger_rankings,
            gold_rows=gold_rows,
        )
        evidence = _measurement_private_evidence(
            schema=f"{VERSION}_A_hold_private_evidence",
            freeze_sha256=freeze["freeze_sha256"],
            block="A_hold",
            primary_rows=private_rows,
            controls=outcomes,
            official_postflight=official_postflight,
        )
        evidence_path = project / A_EVIDENCE_RELATIVE
        _write_json_exclusive(evidence_path, evidence, public=False)
        body = {
            "status": (
                "promoted_challenger_evaluator"
                if transition["promoted"]
                else "retained_incumbent_evaluator"
            ),
            "valid": True,
            "freeze_sha256": freeze["freeze_sha256"],
            "freeze_file_sha256": _sha256_bytes(freeze_raw),
            "design_binding": freeze["design_binding"],
            "diagnostic_binding": freeze["diagnostic_binding"],
            "implementation_binding": freeze["implementation_binding"],
            "formation_binding": freeze["formation_binding"],
            "source_binding": freeze["source_binding"],
            "A_action_binding": freeze["A_action_binding"],
            "prospective_M_action_binding": freeze["prospective_M_action_binding"],
            "primary": {
                "incumbent": incumbent_metrics,
                "challenger": challenger_metrics,
                "challenger_minus_incumbent": comparison,
            },
            "controls": control_metrics,
            "A_decision": decision,
            "evaluator_epoch_transition": transition,
            "M_search_disposition": {
                "opened_during_A_hold": False,
                "authorized_after_A_hold": transition["M_search_open_authorized"],
                "must_remain_unopened_if_not_promoted": True,
            },
            "private_evidence_binding": {
                "file_sha256": _sha256_file(evidence_path),
                "evidence_sha256": evidence["evidence_sha256"],
                "item_level_evidence_persisted_publicly": False,
            },
            "execution": {
                **execution,
                "block": "A_hold",
                "item_count": BLOCK_COUNTS["A_hold"],
                "NLI_worker_count": NLI_WORKERS,
                "torch_threads_per_worker": TORCH_THREADS_PER_WORKER,
                "NLI_postflight_before_label_open": True,
                "all_primary_action_terminals_before_label_open": True,
                "control_actions_joined_before_label_open": True,
                "label_rows_opened_after_terminals": BLOCK_COUNTS["A_hold"],
                "M_search_view_or_label_rows_opened": 0,
                "official_postflight": official_postflight,
                "controls_failure_isolated": True,
                "network_calls": 0,
                "online_evaluator_calls": 0,
                "retries": 0,
                "replays": 0,
                "resamples": 0,
            },
            "raw_content_persisted": False,
        }
        report = _self_hashed(A_REPORT_SCHEMA, body, "report_sha256")
        _write_json_exclusive(output, report, public=True)
        return report
    except Exception as exc:
        failure_body = {
            "schema": FAILURE_SCHEMA,
            "stage": "A_hold",
            "freeze_sha256": freeze["freeze_sha256"],
            "failure_stage": stage,
            "error_type_sha256": stable_hash({"error_type": type(exc).__name__}),
            "authorization_consumed": True,
            "replay_authorized": False,
            "raw_content_persisted": False,
        }
        try:
            _write_json_exclusive(
                root / "A_hold.failure.json",
                {**failure_body, "failure_sha256": stable_hash(failure_body)},
                public=False,
            )
        except Exception:
            pass
        raise QASCCoevolutionError("formal A_hold failed and cohort is burned") from exc


def reverify_a_hold_report(
    *,
    project_root: str | Path,
    pre_run_freeze_path: str | Path,
    public_report_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Exactly recompute the A decision from ignored private evidence."""

    project = Path(project_root).resolve(strict=True)
    freeze, freeze_raw, _custody = _load_committed_freeze(
        project=project,
        path=pre_run_freeze_path,
        relative=A_FREEZE_RELATIVE,
        schema=A_FREEZE_SCHEMA,
    )
    expected = (project / A_REPORT_RELATIVE).absolute()
    if Path(public_report_path).absolute() != expected:
        raise QASCCoevolutionError("A_hold report path is noncanonical")
    report, raw = _read_json(expected, "A_hold aggregate report")
    declared = _verify_self_hash(report, schema=A_REPORT_SCHEMA, field="report_sha256")
    if (
        report.get("valid") is not True
        or report.get("freeze_sha256") != freeze["freeze_sha256"]
        or report.get("freeze_file_sha256") != _sha256_bytes(freeze_raw)
        or report.get("design_binding") != freeze["design_binding"]
        or report.get("diagnostic_binding") != freeze["diagnostic_binding"]
        or report.get("implementation_binding") != freeze["implementation_binding"]
        or report.get("formation_binding") != freeze["formation_binding"]
        or report.get("source_binding") != freeze["source_binding"]
        or report.get("A_action_binding") != freeze["A_action_binding"]
        or report.get("prospective_M_action_binding")
        != freeze["prospective_M_action_binding"]
        or report.get("raw_content_persisted") is not False
    ):
        raise QASCCoevolutionError("A_hold public report binding drifted")
    evidence_path = project / A_EVIDENCE_RELATIVE
    (
        evidence,
        evidence_raw,
        incumbents,
        challengers,
        gold_rows,
        outcomes,
        official_postflight,
    ) = (
        _load_measurement_evidence(
            path=evidence_path,
            field="A_hold private evidence",
            schema=f"{VERSION}_A_hold_private_evidence",
            freeze_sha256=freeze["freeze_sha256"],
            block="A_hold",
            expected_incumbent_recipe_id=freeze["A_action_binding"][
                "incumbent_recipe_id"
            ],
            expected_challenger_recipe_id=freeze["A_action_binding"][
                "challenger_recipe_id"
            ],
        )
    )
    evidence_binding = {
        "file_sha256": _sha256_bytes(evidence_raw),
        "evidence_sha256": evidence["evidence_sha256"],
        "item_level_evidence_persisted_publicly": False,
    }
    if report.get("private_evidence_binding") != evidence_binding:
        raise QASCCoevolutionError("A_hold private evidence binding drifted")
    _verify_measurement_execution(
        execution=report.get("execution"),
        block="A_hold",
        official_postflight=official_postflight,
    )
    _verify_frozen_control_evidence(
        freeze=freeze,
        outcomes=outcomes,
        official_postflight=official_postflight,
    )
    primary = {
        "incumbent": aggregate_rankings(
            arm_id="A_incumbent", rankings=incumbents, gold_rows=gold_rows
        ),
        "challenger": aggregate_rankings(
            arm_id="A_challenger", rankings=challengers, gold_rows=gold_rows
        ),
    }
    comparison = paired_utility_comparison(
        left_arm_id="A_challenger",
        right_arm_id="A_incumbent",
        left_rankings=challengers,
        right_rankings=incumbents,
        gold_rows=gold_rows,
        confirmatory=True,
    )
    primary["challenger_minus_incumbent"] = comparison
    decision = _measurement_decision(
        stage="A_hold", freeze_sha256=freeze["freeze_sha256"], comparison=comparison
    )
    transition = evaluator_epoch_transition(
        a_decision_sha256=decision["A_decision_sha256"],
        promoted=decision["positive_and_exact_p_at_or_below_alpha"],
    )
    controls = _control_metrics(
        outcomes=outcomes,
        challenger_rankings=challengers,
        gold_rows=gold_rows,
    )
    if (
        report.get("primary") != primary
        or report.get("controls") != controls
        or report.get("A_decision") != decision
        or report.get("evaluator_epoch_transition") != transition
        or report.get("M_search_disposition")
        != {
            "opened_during_A_hold": False,
            "authorized_after_A_hold": transition["M_search_open_authorized"],
            "must_remain_unopened_if_not_promoted": True,
        }
    ):
        raise QASCCoevolutionError("A_hold aggregate recomputation drifted")
    expected_status = (
        "promoted_challenger_evaluator"
        if transition["promoted"]
        else "retained_incumbent_evaluator"
    )
    if report.get("status") != expected_status:
        raise QASCCoevolutionError("A_hold status drifted")
    custody = _head_file_binding(project, expected, "A_hold aggregate report")
    if custody["file_sha256"] != _sha256_bytes(raw):
        raise QASCCoevolutionError("A_hold report custody drifted")
    _assert_public_safe(report)
    return report, {
        "relative_path": A_REPORT_RELATIVE,
        "file_sha256": _sha256_bytes(raw),
        "report_sha256": declared,
        "A_decision_sha256": decision["A_decision_sha256"],
        "transition_sha256": transition["transition_sha256"],
        "private_evidence_file_sha256": _sha256_bytes(evidence_raw),
        "private_evidence_sha256": evidence["evidence_sha256"],
        "clean_tracked_HEAD_blob": True,
    }


def build_m_search_freeze(
    *,
    project_root: str | Path,
    diagnostic_path: str | Path,
    formation_freeze_path: str | Path,
    formation_receipt_path: str | Path,
    a_hold_freeze_path: str | Path,
    a_hold_report_path: str | Path,
    acquisition_receipt_path: str | Path,
    selection_secret_path: str | Path,
    nli_model_path: str | Path,
    capability_receipt_path: str | Path,
    runtime_python: str | Path,
    local_llm_model: str | Path,
    local_embedding_model: str | Path,
    base_binding_receipt_path: str | Path,
    attestation_receipt_path: str | Path,
    execution_root: str | Path,
    authorization_hash: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Conditionally freeze M_search without opening its private view or label."""

    project = Path(project_root).resolve(strict=True)
    output = _canonical_public_output(
        project, output_path, M_FREEZE_RELATIVE, "M_search freeze"
    )
    root = _canonical_stage_root(project, execution_root, M_ROOT_RELATIVE)
    if root.exists():
        raise QASCCoevolutionError("M_search execution root already exists")
    _design, design_binding = _load_design(project)
    _diagnostic, diagnostic_binding = _load_diagnostic(
        project, diagnostic_path, require_committed=True
    )
    a_report, a_binding = reverify_a_hold_report(
        project_root=project,
        pre_run_freeze_path=a_hold_freeze_path,
        public_report_path=a_hold_report_path,
    )
    transition = a_report.get("evaluator_epoch_transition")
    if (
        not isinstance(transition, Mapping)
        or transition.get("promoted") is not True
        or transition.get("M_search_open_authorized") is not True
        or a_report.get("M_search_disposition", {}).get("opened_during_A_hold")
        is not False
    ):
        raise QASCCoevolutionError(
            "A_hold did not promote; M_search must remain unopened"
        )
    formation, formation_binding = reverify_formation_receipt(
        project_root=project,
        pre_run_freeze_path=formation_freeze_path,
        public_receipt_path=formation_receipt_path,
    )
    actions = _selected_action_binding(formation)
    if a_report.get("prospective_M_action_binding", {}).get(
        "incumbent_recipe_id"
    ) != actions["F_search"]["incumbent_recipe_id"] or a_report.get(
        "prospective_M_action_binding", {}
    ).get("challenger_recipe_id") != actions["F_search"]["challenger_recipe_id"]:
        raise QASCCoevolutionError("prospective M action binding drifted")
    receipt, raw, commitments = _load_acquisition_live(
        project=project,
        receipt_path=acquisition_receipt_path,
        selection_secret_path=selection_secret_path,
    )
    _selection_secret(project=project, supplied=selection_secret_path, receipt=receipt)
    nli_binding = verify_runtime_asset(project, nli_model_path)
    try:
        retained = _load_retained_p(project)
        retained_binding = {
            "status": "available",
            "program_sha256": retained.program_hash,
            "program_file_sha256": P_PROGRAM_FILE_SHA256,
            "role": "descriptive_control_only",
        }
    except Exception as exc:
        retained_binding = {
            "status": "preflight_unavailable",
            "error_type_sha256": stable_hash({"error_type": type(exc).__name__}),
            "role": "descriptive_control_only",
        }
    try:
        official_binding, _prepared = _prepare_official_runtime(
            project=project,
            capability_receipt_path=capability_receipt_path,
            runtime_python=runtime_python,
            local_llm_model=local_llm_model,
            local_embedding_model=local_embedding_model,
            base_binding_receipt_path=base_binding_receipt_path,
            attestation_receipt_path=attestation_receipt_path,
        )
        official_binding = {"status": "available", **official_binding}
    except Exception as exc:
        official_binding = {
            "status": "preflight_unavailable",
            "error_type_sha256": stable_hash({"error_type": type(exc).__name__}),
            "role": "descriptive_control_only",
        }
    body = {
        "decision": "authorize_one_shot_M_search_only_after_committed_A_promotion",
        "design_binding": design_binding,
        "diagnostic_binding": diagnostic_binding,
        "implementation_binding": _current_implementation_binding(project),
        "formation_binding": formation_binding,
        "A_report_binding": a_binding,
        "source_binding": _source_binding(receipt, raw, commitments["M_search"]),
        "nli_runtime_binding_sha256": stable_hash(nli_binding),
        "official_runtime_binding": official_binding,
        "retained_P_binding": retained_binding,
        "M_action_binding": {
            **actions["F_search"],
            "incumbent_role": "F_incumbent_epoch_comparator",
            "challenger_role": "F_challenger_active_after_A_promotion",
        },
        "evaluator_epoch_binding": dict(transition),
        "execution_contract": {
            "item_count": 64,
            "primary_recipe_count": 2,
            "NLI_score_wave_count": 2,
            "NLI_worker_count": NLI_WORKERS,
            "controls": [
                "canonical_RAW",
                "retained_P",
                "official_HippoRAG_item_local_32",
            ],
            "official_concurrency_cap": OFFICIAL_CONCURRENCY_CAP,
            "controls_descriptive_and_failure_isolated": True,
            "all_primary_action_terminals_and_NLI_postflight_before_labels": True,
            "L5_test": "one_sided_exact_magnitude_preserving_sign_flip_v1",
            "alpha_numerator": 1,
            "alpha_denominator": 10,
            "M_result_can_rollback_epoch": False,
            "retries": 0,
            "replays": 0,
            "resamples": 0,
        },
        "authorization_hash": _require_sha256(
            authorization_hash, "M authorization"
        ),
        "execution_root_sha256": _root_sha256(root),
        "ordering": {
            "A_promotion_reverified_before_M_source_binding": True,
            "M_search_view_rows_read": 0,
            "M_search_label_rows_read": 0,
            "freeze_complete_before_M_search_open": True,
        },
        "raw_content_persisted": False,
    }
    freeze = _self_hashed(M_FREEZE_SCHEMA, body, "freeze_sha256")
    _write_json_exclusive(output, freeze, public=True)
    return freeze


def execute_m_search(
    *,
    project_root: str | Path,
    pre_run_freeze_path: str | Path,
    diagnostic_path: str | Path,
    formation_freeze_path: str | Path,
    formation_receipt_path: str | Path,
    a_hold_freeze_path: str | Path,
    a_hold_report_path: str | Path,
    acquisition_receipt_path: str | Path,
    selection_secret_path: str | Path,
    nli_model_path: str | Path,
    capability_receipt_path: str | Path,
    runtime_python: str | Path,
    local_llm_model: str | Path,
    local_embedding_model: str | Path,
    base_binding_receipt_path: str | Path,
    attestation_receipt_path: str | Path,
    execution_root: str | Path,
    public_report_path: str | Path,
) -> dict[str, Any]:
    """Consume authorized M_search once; its result can never roll back the epoch."""

    if _CLEAN_MODULE_CLI_ACTIVE is not True:
        raise QASCCoevolutionError("formal M_search is available only through clean CLI")
    project = Path(project_root).resolve(strict=True)
    freeze, freeze_raw, _custody = _load_committed_freeze(
        project=project,
        path=pre_run_freeze_path,
        relative=M_FREEZE_RELATIVE,
        schema=M_FREEZE_SCHEMA,
    )
    output = _canonical_public_output(
        project, public_report_path, M_REPORT_RELATIVE, "M_search report"
    )
    root = _canonical_stage_root(project, execution_root, M_ROOT_RELATIVE)
    if freeze.get("execution_root_sha256") != _root_sha256(root) or root.exists():
        raise QASCCoevolutionError("M_search execution root binding drifted")
    _diagnostic, diagnostic_binding = _load_diagnostic(
        project, diagnostic_path, require_committed=True
    )
    # This promotion check deliberately occurs before acquisition is asked for
    # even its public M block commitment.  No M private path is constructed here.
    a_report, a_binding = reverify_a_hold_report(
        project_root=project,
        pre_run_freeze_path=a_hold_freeze_path,
        public_report_path=a_hold_report_path,
    )
    transition = a_report.get("evaluator_epoch_transition")
    if (
        not isinstance(transition, Mapping)
        or transition.get("promoted") is not True
        or transition.get("M_search_open_authorized") is not True
    ):
        raise QASCCoevolutionError("M_search authorization is absent")
    formation, formation_binding = reverify_formation_receipt(
        project_root=project,
        pre_run_freeze_path=formation_freeze_path,
        public_receipt_path=formation_receipt_path,
    )
    current_f = _selected_action_binding(formation)["F_search"]
    receipt, acquisition_raw, commitments = _load_acquisition_live(
        project=project,
        receipt_path=acquisition_receipt_path,
        selection_secret_path=selection_secret_path,
    )
    _selection_secret(project=project, supplied=selection_secret_path, receipt=receipt)
    source = _source_binding(receipt, acquisition_raw, commitments["M_search"])
    nli_binding = verify_runtime_asset(project, nli_model_path)
    frozen_action = freeze.get("M_action_binding", {})
    if (
        freeze.get("diagnostic_binding") != diagnostic_binding
        or freeze.get("A_report_binding") != a_binding
        or freeze.get("formation_binding") != formation_binding
        or freeze.get("source_binding") != source
        or freeze.get("evaluator_epoch_binding") != transition
        or frozen_action.get("incumbent_recipe_id")
        != current_f["incumbent_recipe_id"]
        or frozen_action.get("challenger_recipe_id")
        != current_f["challenger_recipe_id"]
        or freeze.get("nli_runtime_binding_sha256") != stable_hash(nli_binding)
    ):
        raise QASCCoevolutionError("M_search frozen inputs drifted")
    retained, prepared = _load_descriptive_runtime_for_execution(
        project=project,
        frozen_retained_binding=freeze["retained_P_binding"],
        frozen_official_binding=freeze["official_runtime_binding"],
        capability_receipt_path=capability_receipt_path,
        runtime_python=runtime_python,
        local_llm_model=local_llm_model,
        local_embedding_model=local_embedding_model,
        base_binding_receipt_path=base_binding_receipt_path,
        attestation_receipt_path=attestation_receipt_path,
    )
    _consume_authorization(
        root=root,
        stage="M_search",
        freeze_sha256=freeze["freeze_sha256"],
        authorization_hash=freeze["authorization_hash"],
    )
    stage = "open_gold_free_M_search_views"
    try:
        views = _load_block_views(
            project=project, block="M_search", commitment=commitments["M_search"]
        )
        recipe_ids = (
            frozen_action["incumbent_recipe_id"],
            frozen_action["challenger_recipe_id"],
        )
        stage = "parallel_M_primary_and_descriptive_action_execution"
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            controls_future = executor.submit(
                _execute_controls,
                views=views,
                retained_p=retained,
                prepared=prepared,
                root=root,
            )
            with NLIWorkerPool(
                nli_model_path,
                workers=NLI_WORKERS,
                project_root=project,
            ) as pool:
                actions, execution = _score_recipe_views_two_waves(
                    views=views,
                    recipe_ids=recipe_ids,
                    pool=pool,
                )
            stage = "NLI_runtime_postflight_before_M_labels"
            if verify_runtime_asset(project, nli_model_path) != nli_binding:
                raise QASCCoevolutionError("M_search NLI runtime postflight drifted")
            outcomes, official_postflight = _join_controls_failure_isolated(
                controls_future,
                item_count=BLOCK_COUNTS["M_search"],
            )
        stage = "open_M_labels_after_primary_terminals_and_postflight"
        labels = _load_block_labels_after_terminals(
            project=project,
            block="M_search",
            commitment=commitments["M_search"],
            views=views,
        )
        incumbent_rankings, challenger_rankings, private_rows = _score_selected_actions(
            views=views,
            labels=labels,
            actions=actions,
            incumbent_recipe_id=recipe_ids[0],
            challenger_recipe_id=recipe_ids[1],
        )
        gold_rows = tuple(tuple(label.gold_document_ids) for label in labels)
        incumbent_metrics = aggregate_rankings(
            arm_id="F_incumbent_epoch_comparator",
            rankings=incumbent_rankings,
            gold_rows=gold_rows,
        )
        challenger_metrics = aggregate_rankings(
            arm_id="F_challenger_active",
            rankings=challenger_rankings,
            gold_rows=gold_rows,
        )
        comparison = paired_utility_comparison(
            left_arm_id="F_challenger_active",
            right_arm_id="F_incumbent_epoch_comparator",
            left_rankings=challenger_rankings,
            right_rankings=incumbent_rankings,
            gold_rows=gold_rows,
            confirmatory=True,
        )
        decision = _measurement_decision(
            stage="M_search",
            freeze_sha256=freeze["freeze_sha256"],
            comparison=comparison,
        )
        control_metrics = _control_metrics(
            outcomes=outcomes,
            challenger_rankings=challenger_rankings,
            gold_rows=gold_rows,
        )
        epoch_disposition = {
            "epoch_before_M": transition["next_epoch_id"],
            "epoch_after_M": transition["next_epoch_id"],
            "epoch_index_before_M": transition["next_epoch_index"],
            "epoch_index_after_M": transition["next_epoch_index"],
            "evaluator_before_M": transition["next_evaluator_id"],
            "evaluator_after_M": transition["next_evaluator_id"],
            "M_result_changed_or_rolled_back_epoch": False,
            "L5_improved_later_search": decision[
                "positive_and_exact_p_at_or_below_alpha"
            ],
        }
        evidence = _measurement_private_evidence(
            schema=f"{VERSION}_M_search_private_evidence",
            freeze_sha256=freeze["freeze_sha256"],
            block="M_search",
            primary_rows=private_rows,
            controls=outcomes,
            official_postflight=official_postflight,
        )
        evidence_path = project / M_EVIDENCE_RELATIVE
        _write_json_exclusive(evidence_path, evidence, public=False)
        body = {
            "status": (
                "L5_later_search_improved"
                if epoch_disposition["L5_improved_later_search"]
                else "evaluator_promoted_but_later_search_not_improved"
            ),
            "valid": True,
            "freeze_sha256": freeze["freeze_sha256"],
            "freeze_file_sha256": _sha256_bytes(freeze_raw),
            "design_binding": freeze["design_binding"],
            "diagnostic_binding": freeze["diagnostic_binding"],
            "implementation_binding": freeze["implementation_binding"],
            "formation_binding": freeze["formation_binding"],
            "A_report_binding": freeze["A_report_binding"],
            "source_binding": freeze["source_binding"],
            "M_action_binding": freeze["M_action_binding"],
            "evaluator_epoch_binding": freeze["evaluator_epoch_binding"],
            "primary": {
                "incumbent": incumbent_metrics,
                "challenger_active": challenger_metrics,
                "challenger_active_minus_incumbent": comparison,
            },
            "controls": control_metrics,
            "M_decision": decision,
            "evaluator_epoch_disposition": epoch_disposition,
            "private_evidence_binding": {
                "file_sha256": _sha256_file(evidence_path),
                "evidence_sha256": evidence["evidence_sha256"],
                "item_level_evidence_persisted_publicly": False,
            },
            "execution": {
                **execution,
                "block": "M_search",
                "item_count": BLOCK_COUNTS["M_search"],
                "NLI_worker_count": NLI_WORKERS,
                "torch_threads_per_worker": TORCH_THREADS_PER_WORKER,
                "NLI_postflight_before_label_open": True,
                "all_primary_action_terminals_before_label_open": True,
                "control_actions_joined_before_label_open": True,
                "label_rows_opened_after_terminals": BLOCK_COUNTS["M_search"],
                "official_postflight": official_postflight,
                "controls_failure_isolated": True,
                "network_calls": 0,
                "online_evaluator_calls": 0,
                "retries": 0,
                "replays": 0,
                "resamples": 0,
            },
            "raw_content_persisted": False,
        }
        report = _self_hashed(M_REPORT_SCHEMA, body, "report_sha256")
        _write_json_exclusive(output, report, public=True)
        return report
    except Exception as exc:
        failure_body = {
            "schema": FAILURE_SCHEMA,
            "stage": "M_search",
            "freeze_sha256": freeze["freeze_sha256"],
            "failure_stage": stage,
            "error_type_sha256": stable_hash({"error_type": type(exc).__name__}),
            "authorization_consumed": True,
            "replay_authorized": False,
            "raw_content_persisted": False,
        }
        try:
            _write_json_exclusive(
                root / "M_search.failure.json",
                {**failure_body, "failure_sha256": stable_hash(failure_body)},
                public=False,
            )
        except Exception:
            pass
        raise QASCCoevolutionError("formal M_search failed and cohort is burned") from exc


def reverify_m_search_report(
    *,
    project_root: str | Path,
    pre_run_freeze_path: str | Path,
    public_report_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Exactly recompute M's L5 result and prove the epoch did not change."""

    project = Path(project_root).resolve(strict=True)
    freeze, freeze_raw, _custody = _load_committed_freeze(
        project=project,
        path=pre_run_freeze_path,
        relative=M_FREEZE_RELATIVE,
        schema=M_FREEZE_SCHEMA,
    )
    expected = (project / M_REPORT_RELATIVE).absolute()
    if Path(public_report_path).absolute() != expected:
        raise QASCCoevolutionError("M_search report path is noncanonical")
    report, raw = _read_json(expected, "M_search aggregate report")
    declared = _verify_self_hash(report, schema=M_REPORT_SCHEMA, field="report_sha256")
    for field in (
        "design_binding",
        "diagnostic_binding",
        "implementation_binding",
        "formation_binding",
        "A_report_binding",
        "source_binding",
        "M_action_binding",
        "evaluator_epoch_binding",
    ):
        if report.get(field) != freeze.get(field):
            raise QASCCoevolutionError(f"M_search {field} drifted")
    if (
        report.get("valid") is not True
        or report.get("freeze_sha256") != freeze["freeze_sha256"]
        or report.get("freeze_file_sha256") != _sha256_bytes(freeze_raw)
        or report.get("raw_content_persisted") is not False
    ):
        raise QASCCoevolutionError("M_search public report binding drifted")
    evidence_path = project / M_EVIDENCE_RELATIVE
    (
        evidence,
        evidence_raw,
        incumbents,
        challengers,
        gold_rows,
        outcomes,
        official_postflight,
    ) = (
        _load_measurement_evidence(
            path=evidence_path,
            field="M_search private evidence",
            schema=f"{VERSION}_M_search_private_evidence",
            freeze_sha256=freeze["freeze_sha256"],
            block="M_search",
            expected_incumbent_recipe_id=freeze["M_action_binding"][
                "incumbent_recipe_id"
            ],
            expected_challenger_recipe_id=freeze["M_action_binding"][
                "challenger_recipe_id"
            ],
        )
    )
    if report.get("private_evidence_binding") != {
        "file_sha256": _sha256_bytes(evidence_raw),
        "evidence_sha256": evidence["evidence_sha256"],
        "item_level_evidence_persisted_publicly": False,
    }:
        raise QASCCoevolutionError("M_search private evidence binding drifted")
    _verify_measurement_execution(
        execution=report.get("execution"),
        block="M_search",
        official_postflight=official_postflight,
    )
    _verify_frozen_control_evidence(
        freeze=freeze,
        outcomes=outcomes,
        official_postflight=official_postflight,
    )
    primary = {
        "incumbent": aggregate_rankings(
            arm_id="F_incumbent_epoch_comparator",
            rankings=incumbents,
            gold_rows=gold_rows,
        ),
        "challenger_active": aggregate_rankings(
            arm_id="F_challenger_active",
            rankings=challengers,
            gold_rows=gold_rows,
        ),
    }
    comparison = paired_utility_comparison(
        left_arm_id="F_challenger_active",
        right_arm_id="F_incumbent_epoch_comparator",
        left_rankings=challengers,
        right_rankings=incumbents,
        gold_rows=gold_rows,
        confirmatory=True,
    )
    primary["challenger_active_minus_incumbent"] = comparison
    decision = _measurement_decision(
        stage="M_search", freeze_sha256=freeze["freeze_sha256"], comparison=comparison
    )
    transition = freeze["evaluator_epoch_binding"]
    epoch = {
        "epoch_before_M": transition["next_epoch_id"],
        "epoch_after_M": transition["next_epoch_id"],
        "epoch_index_before_M": transition["next_epoch_index"],
        "epoch_index_after_M": transition["next_epoch_index"],
        "evaluator_before_M": transition["next_evaluator_id"],
        "evaluator_after_M": transition["next_evaluator_id"],
        "M_result_changed_or_rolled_back_epoch": False,
        "L5_improved_later_search": decision[
            "positive_and_exact_p_at_or_below_alpha"
        ],
    }
    controls = _control_metrics(
        outcomes=outcomes,
        challenger_rankings=challengers,
        gold_rows=gold_rows,
    )
    if (
        report.get("primary") != primary
        or report.get("controls") != controls
        or report.get("M_decision") != decision
        or report.get("evaluator_epoch_disposition") != epoch
    ):
        raise QASCCoevolutionError("M_search aggregate recomputation drifted")
    expected_status = (
        "L5_later_search_improved"
        if epoch["L5_improved_later_search"]
        else "evaluator_promoted_but_later_search_not_improved"
    )
    if report.get("status") != expected_status:
        raise QASCCoevolutionError("M_search status drifted")
    custody = _head_file_binding(project, expected, "M_search aggregate report")
    if custody["file_sha256"] != _sha256_bytes(raw):
        raise QASCCoevolutionError("M_search report custody drifted")
    _assert_public_safe(report)
    return report, {
        "relative_path": M_REPORT_RELATIVE,
        "file_sha256": _sha256_bytes(raw),
        "report_sha256": declared,
        "M_decision_sha256": decision["M_decision_sha256"],
        "private_evidence_file_sha256": _sha256_bytes(evidence_raw),
        "private_evidence_sha256": evidence["evidence_sha256"],
        "clean_tracked_HEAD_blob": True,
    }


def formal_signatures_have_no_injection_surface() -> bool:
    forbidden = {
        "actions",
        "evidence",
        "items",
        "labels",
        "program",
        "programs",
        "rankings",
        "result",
        "results",
        "retriever",
        "scorer",
        "views",
    }
    formal = tuple(
        function
        for name, function in globals().items()
        if callable(function)
        and name
        in {
            "build_formation_freeze",
            "execute_formation",
            "build_a_hold_freeze",
            "execute_a_hold",
            "build_m_search_freeze",
            "execute_m_search",
        }
    )
    if len(formal) != 6:
        return False
    return not forbidden.intersection(
        set().union(*(set(inspect.signature(function).parameters) for function in formal))
    )


def _add_project(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", required=True)


def _add_diagnostic(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--diagnostic-path", required=True)


def _add_acquisition(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--acquisition-receipt-path", required=True)
    parser.add_argument("--selection-secret-path", required=True)


def _add_nli(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--nli-model-path", required=True)


def _add_official(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--capability-receipt-path", required=True)
    parser.add_argument("--runtime-python", required=True)
    parser.add_argument("--local-llm-model", required=True)
    parser.add_argument("--local-embedding-model", required=True)
    parser.add_argument("--base-binding-receipt-path", required=True)
    parser.add_argument("--attestation-receipt-path", required=True)


def _add_formation_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--formation-freeze-path", required=True)
    parser.add_argument("--formation-receipt-path", required=True)


def _add_a_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--a-hold-freeze-path", required=True)
    parser.add_argument("--a-hold-report-path", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen one-shot QASC evaluator co-evolution runner"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    diagnostic = commands.add_parser("diagnose")
    _add_project(diagnostic)
    _add_nli(diagnostic)
    _add_official(diagnostic)
    diagnostic.add_argument("--output-path", required=True)

    freeze_formation = commands.add_parser("freeze-formation")
    _add_project(freeze_formation)
    _add_diagnostic(freeze_formation)
    _add_acquisition(freeze_formation)
    _add_nli(freeze_formation)
    freeze_formation.add_argument("--execution-root", required=True)
    freeze_formation.add_argument("--authorization-hash", required=True)
    freeze_formation.add_argument("--output-path", required=True)

    run_formation = commands.add_parser("run-formation")
    _add_project(run_formation)
    _add_diagnostic(run_formation)
    _add_acquisition(run_formation)
    _add_nli(run_formation)
    run_formation.add_argument("--pre-run-freeze-path", required=True)
    run_formation.add_argument("--execution-root", required=True)
    run_formation.add_argument("--public-receipt-path", required=True)

    verify_formation = commands.add_parser("verify-formation")
    _add_project(verify_formation)
    verify_formation.add_argument("--pre-run-freeze-path", required=True)
    verify_formation.add_argument("--public-receipt-path", required=True)

    freeze_a = commands.add_parser("freeze-a")
    _add_project(freeze_a)
    _add_diagnostic(freeze_a)
    _add_formation_inputs(freeze_a)
    _add_acquisition(freeze_a)
    _add_nli(freeze_a)
    _add_official(freeze_a)
    freeze_a.add_argument("--execution-root", required=True)
    freeze_a.add_argument("--authorization-hash", required=True)
    freeze_a.add_argument("--output-path", required=True)

    run_a = commands.add_parser("run-a")
    _add_project(run_a)
    _add_diagnostic(run_a)
    _add_formation_inputs(run_a)
    _add_acquisition(run_a)
    _add_nli(run_a)
    _add_official(run_a)
    run_a.add_argument("--pre-run-freeze-path", required=True)
    run_a.add_argument("--execution-root", required=True)
    run_a.add_argument("--public-report-path", required=True)

    verify_a = commands.add_parser("verify-a")
    _add_project(verify_a)
    verify_a.add_argument("--pre-run-freeze-path", required=True)
    verify_a.add_argument("--public-report-path", required=True)

    freeze_m = commands.add_parser("freeze-m")
    _add_project(freeze_m)
    _add_diagnostic(freeze_m)
    _add_formation_inputs(freeze_m)
    _add_a_inputs(freeze_m)
    _add_acquisition(freeze_m)
    _add_nli(freeze_m)
    _add_official(freeze_m)
    freeze_m.add_argument("--execution-root", required=True)
    freeze_m.add_argument("--authorization-hash", required=True)
    freeze_m.add_argument("--output-path", required=True)

    run_m = commands.add_parser("run-m")
    _add_project(run_m)
    _add_diagnostic(run_m)
    _add_formation_inputs(run_m)
    _add_a_inputs(run_m)
    _add_acquisition(run_m)
    _add_nli(run_m)
    _add_official(run_m)
    run_m.add_argument("--pre-run-freeze-path", required=True)
    run_m.add_argument("--execution-root", required=True)
    run_m.add_argument("--public-report-path", required=True)

    verify_m = commands.add_parser("verify-m")
    _add_project(verify_m)
    verify_m.add_argument("--pre-run-freeze-path", required=True)
    verify_m.add_argument("--public-report-path", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    global _CLEAN_MODULE_CLI_ACTIVE
    arguments = _parser().parse_args(argv)
    command = arguments.command
    values = vars(arguments)
    del values["command"]
    if command in {"run-formation", "run-a", "run-m"}:
        _CLEAN_MODULE_CLI_ACTIVE = True
    try:
        if command == "diagnose":
            result: object = run_infrastructure_diagnostic(**values)
        elif command == "freeze-formation":
            result = build_formation_freeze(**values)
        elif command == "run-formation":
            result = execute_formation(**values)
        elif command == "verify-formation":
            result = reverify_formation_receipt(**values)[1]
        elif command == "freeze-a":
            result = build_a_hold_freeze(**values)
        elif command == "run-a":
            result = execute_a_hold(**values)
        elif command == "verify-a":
            result = reverify_a_hold_report(**values)[1]
        elif command == "freeze-m":
            result = build_m_search_freeze(**values)
        elif command == "run-m":
            result = execute_m_search(**values)
        elif command == "verify-m":
            result = reverify_m_search_report(**values)[1]
        else:  # pragma: no cover - argparse makes this unreachable.
            raise QASCCoevolutionError("unknown command")
    finally:
        _CLEAN_MODULE_CLI_ACTIVE = False
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
