"""One-shot, source-blind formal lifecycle for AVeriTeC P1.

The controller consumes only the selector's private label-free block views.
For each block it seals every retrieval/action result before the corresponding
qrel pack is opened.  ``F_search`` has no qrel path and is descriptive only.
``M_search`` is neither opened nor materialized unless the independently
sealed ``A_hold`` comparison promotes the frozen E1 evaluator.

The controller owns no benchmark source reader, API, network evaluator,
provider switch, retry, replay, resampling, candidate mutation, or gate
surface.  Production model processes are supplied through two narrow
executors and are physically capped at one MiniLM lane plus one official
HippoRAG lane.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Protocol, Sequence

from assumption_agent.benchmarks import averitec_p1_acquisition_v1 as acquisition
from assumption_agent.benchmarks import averitec_p1_coordinate_worker_v1 as coordinate
from assumption_agent.benchmarks import averitec_p1_typed_core_v1 as core


VERSION = "averitec_p1_formal_controller_v1"
STUDY_ID = core.STUDY_ID
STAGE_ARCHIVE_SCHEMA = f"{VERSION}_private_stage_action_archive_v1"
F_TRACE_SCHEMA = f"{VERSION}_private_F_search_trace_v1"
A_HOLD_SCORE_SCHEMA = f"{VERSION}_private_A_hold_score_v1"
M_SEARCH_SCORE_SCHEMA = f"{VERSION}_private_M_search_score_v1"
FINAL_SCHEMA = f"{VERSION}_safe_terminal_v1"
FAILURE_SCHEMA = f"{VERSION}_safe_failure_terminal_v1"

FORMAL_BLOCK_QUERY_COUNTS = {
    acquisition.A_FORM: 108,
    acquisition.F_SEARCH: 36,
    acquisition.A_HOLD: 36,
    acquisition.M_SEARCH: 36,
}
FORMAL_FAMILY_COUNTS = {
    acquisition.A_FORM: {family: 36 for family in acquisition.FAMILIES},
    acquisition.A_HOLD: {family: 12 for family in acquisition.FAMILIES},
    acquisition.M_SEARCH: {family: 12 for family in acquisition.FAMILIES},
}
MAX_PHYSICAL_MODEL_LANES = 2
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class AveritecP1FormalControllerError(RuntimeError):
    """A frozen formal lifecycle, custody, or result invariant failed."""


@dataclass(frozen=True)
class HippoResult:
    indices: tuple[tuple[int, ...], ...]
    receipt_sha256: str
    build_receipt_sha256: str

    def __post_init__(self) -> None:
        for field in (self.receipt_sha256, self.build_receipt_sha256):
            if not isinstance(field, str) or _HEX64.fullmatch(field) is None:
                raise AveritecP1FormalControllerError(
                    "official HippoRAG receipt identity drifted"
                )


class CoordinateExecutor(Protocol):
    def __call__(
        self,
        *,
        block: str,
        private_input: Mapping[str, object],
    ) -> Mapping[str, object]: ...


class HippoExecutor(Protocol):
    def __call__(
        self,
        *,
        block: str,
        articles: Sequence[Mapping[str, object]],
        queries: Sequence[tuple[str, str]],
    ) -> HippoResult: ...


def canonical_bytes(value: object, *, newline: bool = True) -> bytes:
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise AveritecP1FormalControllerError(
            "formal value is not canonical JSON"
        ) from exc
    return raw + (b"\n" if newline else b"")


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value, newline=False)).hexdigest()


def self_hashed(body: Mapping[str, object]) -> dict[str, object]:
    value = dict(body)
    if "self_sha256" in value:
        raise AveritecP1FormalControllerError("self hash already present")
    value["self_sha256"] = stable_hash(value)
    return value


def _verify_self(value: Mapping[str, object], field: str = "self_sha256") -> str:
    body = dict(value)
    claimed = body.pop(field, None)
    if (
        not isinstance(claimed, str)
        or _HEX64.fullmatch(claimed) is None
        or stable_hash(body) != claimed
    ):
        raise AveritecP1FormalControllerError("private payload self hash drifted")
    return claimed


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _make_private_directory(path: Path) -> None:
    path.mkdir(parents=True, mode=0o700, exist_ok=False)
    if path.is_symlink() or stat.S_IMODE(path.stat().st_mode) != 0o700:
        raise AveritecP1FormalControllerError("private directory mode drifted")
    _fsync_directory(path.parent)


def _read_canonical(path: Path, *, mode: int) -> dict[str, object]:
    try:
        before = path.lstat()
    except OSError as exc:
        raise AveritecP1FormalControllerError(
            "private artifact is unavailable"
        ) from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != mode
    ):
        raise AveritecP1FormalControllerError(
            "private artifact metadata drifted"
        )
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AveritecP1FormalControllerError(
            "private artifact cannot be parsed"
        ) from exc
    after = path.lstat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise AveritecP1FormalControllerError(
            "private artifact changed while read"
        )
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise AveritecP1FormalControllerError(
            "private artifact is not canonical JSON"
        )
    return value


def _seal(path: Path, value: Mapping[str, object]) -> str:
    """Atomically publish one immutable canonical artifact."""

    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise AveritecP1FormalControllerError("sealed artifact already exists")
    raw = canonical_bytes(value)
    building = path.parent / f".{path.name}.building"
    if building.exists() or building.is_symlink():
        raise AveritecP1FormalControllerError(
            "sealed artifact build path already exists"
        )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(building, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(building, 0o400)
        os.link(building, path, follow_symlinks=False)
        _fsync_directory(path.parent)
    finally:
        if building.exists():
            building.unlink()
    if (
        path.is_symlink()
        or not path.is_file()
        or stat.S_IMODE(path.stat().st_mode) != 0o400
        or path.read_bytes() != raw
    ):
        raise AveritecP1FormalControllerError("sealed artifact verification failed")
    return hashlib.sha256(raw).hexdigest()


def _fraction(value: Fraction) -> dict[str, int]:
    return {"denominator": value.denominator, "numerator": value.numerator}


def _comparison(value: core.ExactComparison) -> dict[str, object]:
    return {
        "negative_count": value.negative_count,
        "net_utility": _fraction(value.net_utility),
        "positive_count": value.positive_count,
        "reference_tail": _fraction(value.reference_tail),
        "tie_count": value.tie_count,
    }


def _serialized_document(row: Mapping[str, object]) -> str:
    title = row.get("title")
    body = row.get("body")
    if (
        not isinstance(title, str)
        or not title.strip()
        or "\x00" in title
        or not isinstance(body, str)
        or not body.strip()
        or "\x00" in body
    ):
        raise AveritecP1FormalControllerError("action document text drifted")
    return title + "\n\n" + body


def _validate_view(
    value: Mapping[str, object],
    *,
    block: str,
    expected_query_count: int,
) -> tuple[
    tuple[Mapping[str, object], ...],
    tuple[Mapping[str, object], ...],
    str,
]:
    if set(value) != {
        "block",
        "corpus",
        "queries",
        "schema",
        "self_sha256",
        "study_id",
    }:
        raise AveritecP1FormalControllerError("action view envelope drifted")
    view_hash = _verify_self(value)
    if (
        value.get("block") != block
        or value.get("study_id") != STUDY_ID
        or value.get("schema")
        != f"{acquisition.VERSION}_label_free_action_view_v1"
    ):
        raise AveritecP1FormalControllerError("action view binding drifted")
    corpus = value.get("corpus")
    queries = value.get("queries")
    if (
        not isinstance(corpus, list)
        or not 5 <= len(corpus) <= coordinate.MAX_DOCUMENT_COUNT
        or not isinstance(queries, list)
        or len(queries) != expected_query_count
    ):
        raise AveritecP1FormalControllerError("action view cardinality drifted")
    document_ids: set[str] = set()
    document_texts: set[str] = set()
    normalized_corpus: list[Mapping[str, object]] = []
    for ordinal, row in enumerate(corpus):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"body", "document_id", "ordinal", "title"}
            or row.get("ordinal") != ordinal
        ):
            raise AveritecP1FormalControllerError("action document shape drifted")
        document_id = row.get("document_id")
        if (
            not isinstance(document_id, str)
            or _HEX64.fullmatch(document_id) is None
            or document_id in document_ids
        ):
            raise AveritecP1FormalControllerError("action document ID drifted")
        text = _serialized_document(row)
        if text in document_texts:
            raise AveritecP1FormalControllerError(
                "serialized action documents are duplicated"
            )
        document_ids.add(document_id)
        document_texts.add(text)
        normalized_corpus.append(dict(row))
    item_ids: set[str] = set()
    normalized_queries: list[Mapping[str, object]] = []
    for ordinal, row in enumerate(queries):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"item_id", "ordinal", "text"}
            or row.get("ordinal") != ordinal
        ):
            raise AveritecP1FormalControllerError("action query shape drifted")
        item_id = row.get("item_id")
        text = row.get("text")
        if (
            not isinstance(item_id, str)
            or _HEX64.fullmatch(item_id) is None
            or item_id in item_ids
            or not isinstance(text, str)
            or not text.strip()
            or "\x00" in text
            or len(text) > 4_000
        ):
            raise AveritecP1FormalControllerError("action query value drifted")
        item_ids.add(item_id)
        normalized_queries.append(dict(row))
    return tuple(normalized_corpus), tuple(normalized_queries), view_hash


def _validate_qrels(
    value: Mapping[str, object],
    *,
    block: str,
    queries: Sequence[Mapping[str, object]],
    corpus_count: int,
) -> tuple[dict[str, tuple[str, tuple[int, ...]]], str]:
    if set(value) != {
        "block",
        "rows",
        "schema",
        "self_sha256",
        "study_id",
    }:
        raise AveritecP1FormalControllerError("qrel pack envelope drifted")
    qrel_hash = _verify_self(value)
    if (
        block == acquisition.F_SEARCH
        or value.get("block") != block
        or value.get("study_id") != STUDY_ID
        or value.get("schema") != f"{acquisition.VERSION}_late_qrel_pack_v1"
    ):
        raise AveritecP1FormalControllerError("qrel pack binding drifted")
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != len(queries):
        raise AveritecP1FormalControllerError("qrel pack cardinality drifted")
    expected_items = [row["item_id"] for row in queries]
    normalized: dict[str, tuple[str, tuple[int, ...]]] = {}
    family_counts: Counter[str] = Counter()
    for expected_item, row in zip(expected_items, rows):
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {"family", "item_id", "qrel_document_ordinals"}
            or row.get("item_id") != expected_item
            or row.get("family") not in acquisition.FAMILIES
        ):
            raise AveritecP1FormalControllerError("qrel row shape drifted")
        raw_ordinals = row.get("qrel_document_ordinals")
        if (
            not isinstance(raw_ordinals, list)
            or len(raw_ordinals) < 2
            or len(set(raw_ordinals)) != len(raw_ordinals)
            or any(
                type(ordinal) is not int
                or not 0 <= ordinal < corpus_count
                for ordinal in raw_ordinals
            )
        ):
            raise AveritecP1FormalControllerError("qrel ordinals drifted")
        family = str(row["family"])
        normalized[str(expected_item)] = (
            family,
            tuple(int(ordinal) for ordinal in raw_ordinals),
        )
        family_counts[family] += 1
    if dict(family_counts) != FORMAL_FAMILY_COUNTS[block]:
        raise AveritecP1FormalControllerError("qrel family quota drifted")
    return normalized, qrel_hash


def _validate_hippo_result(
    result: HippoResult,
    *,
    query_count: int,
    corpus_count: int,
) -> HippoResult:
    if not isinstance(result, HippoResult) or len(result.indices) != query_count:
        raise AveritecP1FormalControllerError(
            "official HippoRAG result count drifted"
        )
    for row in result.indices:
        if (
            len(row) != core.TOP_K
            or len(set(row)) != core.TOP_K
            or any(
                type(ordinal) is not int
                or not 0 <= ordinal < corpus_count
                for ordinal in row
            )
        ):
            raise AveritecP1FormalControllerError(
                "official HippoRAG top5 drifted"
            )
    return result


@dataclass(frozen=True)
class _Stage:
    block: str
    view: Mapping[str, object]
    corpus: tuple[Mapping[str, object], ...]
    queries: tuple[Mapping[str, object], ...]
    document_texts: tuple[str, ...]
    actions: tuple[Mapping[str, object], ...]
    archive: Mapping[str, object]
    archive_file_sha256: str


class FormalController:
    def __init__(
        self,
        *,
        acquisition_root: Path,
        work_root: Path,
        execution_binding_sha256: str,
        coordinate_executor: CoordinateExecutor,
        hippo_executor: HippoExecutor,
    ) -> None:
        if (
            not isinstance(execution_binding_sha256, str)
            or _HEX64.fullmatch(execution_binding_sha256) is None
        ):
            raise AveritecP1FormalControllerError(
                "execution binding SHA-256 drifted"
            )
        self.acquisition_root = acquisition_root.absolute()
        self.work_root = work_root.absolute()
        self.execution_binding_sha256 = execution_binding_sha256
        self.coordinate_executor = coordinate_executor
        self.hippo_executor = hippo_executor
        if self.work_root.exists() or self.work_root.is_symlink():
            raise AveritecP1FormalControllerError("formal work root is not fresh")

    def _view(self, block: str) -> dict[str, object]:
        return _read_canonical(
            self.acquisition_root / f"{block}.view.json", mode=0o600
        )

    def _qrels(
        self,
        *,
        block: str,
        queries: Sequence[Mapping[str, object]],
        corpus_count: int,
    ) -> tuple[dict[str, tuple[str, tuple[int, ...]]], str]:
        if block == acquisition.F_SEARCH:
            raise AveritecP1FormalControllerError(
                "F_search has no qrel-opening path"
            )
        payload = _read_canonical(
            self.acquisition_root / f"{block}.qrels.json", mode=0o600
        )
        return _validate_qrels(
            payload,
            block=block,
            queries=queries,
            corpus_count=corpus_count,
        )

    def _coordinate_input(
        self,
        *,
        corpus: Sequence[Mapping[str, object]],
        queries: Sequence[Mapping[str, object]],
    ) -> dict[str, object]:
        return coordinate.private_input_payload(
            documents=[_serialized_document(row) for row in corpus],
            queries=[
                (str(row["item_id"]), str(row["text"])) for row in queries
            ],
        )

    def _run_stage_models(
        self,
        *,
        block: str,
        corpus: Sequence[Mapping[str, object]],
        queries: Sequence[Mapping[str, object]],
        include_hippo: bool,
    ) -> tuple[dict[str, object], HippoResult | None]:
        private_input = self._coordinate_input(corpus=corpus, queries=queries)
        articles = [
            {
                "body": row["body"],
                "idx": ordinal,
                "title": row["title"],
            }
            for ordinal, row in enumerate(corpus)
        ]
        query_rows = [
            (str(row["item_id"]), str(row["text"])) for row in queries
        ]
        if include_hippo:
            with ThreadPoolExecutor(
                max_workers=MAX_PHYSICAL_MODEL_LANES,
                thread_name_prefix="averitec-two-gpu-lanes",
            ) as pool:
                coordinate_future = pool.submit(
                    self.coordinate_executor,
                    block=block,
                    private_input=private_input,
                )
                hippo_future = pool.submit(
                    self.hippo_executor,
                    block=block,
                    articles=articles,
                    queries=query_rows,
                )
                completed, pending = wait(
                    (coordinate_future, hippo_future),
                    return_when=FIRST_EXCEPTION,
                )
                failed = next(
                    (
                        future
                        for future in completed
                        if not future.cancelled()
                        and future.exception() is not None
                    ),
                    None,
                )
                if failed is not None:
                    for executor in (
                        self.coordinate_executor,
                        self.hippo_executor,
                    ):
                        cancel = getattr(executor, "cancel_all", None)
                        if callable(cancel):
                            cancel()
                    for future in pending:
                        future.cancel()
                    # Consume both futures before leaving the pool so no model
                    # child can survive an eager-lane failure.
                    wait((coordinate_future, hippo_future))
                    failed.result()
                coordinate_raw = coordinate_future.result()
                hippo_raw = hippo_future.result()
        else:
            coordinate_raw = self.coordinate_executor(
                block=block, private_input=private_input
            )
            hippo_raw = None
        coordinate_output = coordinate.validate_output(
            coordinate_raw, expected_input=private_input
        )
        hippo_output = (
            None
            if hippo_raw is None
            else _validate_hippo_result(
                hippo_raw,
                query_count=len(queries),
                corpus_count=len(corpus),
            )
        )
        return coordinate_output, hippo_output

    def _materialize_stage(
        self,
        *,
        block: str,
        model: core.E1Model | None,
        include_hippo: bool,
    ) -> _Stage:
        view = self._view(block)
        corpus, queries, view_hash = _validate_view(
            view,
            block=block,
            expected_query_count=FORMAL_BLOCK_QUERY_COUNTS[block],
        )
        document_texts = tuple(_serialized_document(row) for row in corpus)
        coordinate_output, hippo = self._run_stage_models(
            block=block,
            corpus=corpus,
            queries=queries,
            include_hippo=include_hippo,
        )
        coordinate_rows = coordinate_output["rows"]
        if not isinstance(coordinate_rows, list):
            raise AveritecP1FormalControllerError(
                "validated coordinate rows disappeared"
            )
        items: list[Mapping[str, object]] = []
        for ordinal, (query, coordinate_row) in enumerate(
            zip(queries, coordinate_rows)
        ):
            if (
                not isinstance(coordinate_row, Mapping)
                or coordinate_row.get("item_id") != query["item_id"]
                or not isinstance(coordinate_row.get("variant_scores"), Mapping)
            ):
                raise AveritecP1FormalControllerError(
                    "coordinate/action item binding drifted"
                )
            slate = core.materialize_recipe_actions(
                document_texts=document_texts,
                variant_scores=coordinate_row["variant_scores"],  # type: ignore[arg-type]
            )
            e0 = core.select_e0(slate)
            e1 = (
                None
                if model is None
                else core.select_e1(
                    model=model,
                    actions=slate,
                    document_texts=document_texts,
                )
            )
            items.append(
                {
                    "E0_recipe_id": e0,
                    "E1_recipe_id": e1,
                    "HippoRAG_top5_document_ordinals": (
                        None if hippo is None else list(hippo.indices[ordinal])
                    ),
                    "RAW_top5_document_ordinals": list(
                        slate[core.R0_DIRECT_DENSE].top5_document_ordinals
                    ),
                    "item_id": query["item_id"],
                    "recipe_actions": {
                        recipe_id: core.action_payload(slate[recipe_id])
                        for recipe_id in core.RECIPE_IDS
                    },
                }
            )
        archive = self_hashed(
            {
                "block": block,
                "coordinate_output_self_sha256": coordinate_output["self_sha256"],
                "execution_binding_sha256": self.execution_binding_sha256,
                "hipporag_build_receipt_sha256": (
                    None if hippo is None else hippo.build_receipt_sha256
                ),
                "hipporag_retrieval_receipt_sha256": (
                    None if hippo is None else hippo.receipt_sha256
                ),
                "items": items,
                "label_or_qrel_read_count_before_seal": 0,
                "online_evaluator_call_count": 0,
                "schema": STAGE_ARCHIVE_SCHEMA,
                "study_id": STUDY_ID,
                "view_self_sha256": view_hash,
            }
        )
        archive_path = self.work_root / "stages" / block / "actions.private.json"
        archive_file_sha256 = _seal(archive_path, archive)
        reread = _read_canonical(archive_path, mode=0o400)
        if reread != archive or _verify_self(reread) != archive["self_sha256"]:
            raise AveritecP1FormalControllerError(
                "sealed action archive revalidation failed"
            )
        return _Stage(
            block=block,
            view=view,
            corpus=corpus,
            queries=queries,
            document_texts=document_texts,
            actions=tuple(items),
            archive=archive,
            archive_file_sha256=archive_file_sha256,
        )

    @staticmethod
    def _slate(item: Mapping[str, object]) -> dict[str, core.RecipeAction]:
        payload = item.get("recipe_actions")
        if not isinstance(payload, Mapping) or tuple(payload) != core.RECIPE_IDS:
            raise AveritecP1FormalControllerError(
                "sealed recipe action slate drifted"
            )
        return {
            recipe_id: core.action_from_payload(payload[recipe_id])
            for recipe_id in core.RECIPE_IDS
        }

    def _fit_e1(self, stage: _Stage) -> tuple[core.E1Model, str]:
        qrels, qrel_hash = self._qrels(
            block=acquisition.A_FORM,
            queries=stage.queries,
            corpus_count=len(stage.corpus),
        )
        slates: list[core.AFormSlate] = []
        for item in stage.actions:
            item_id = str(item["item_id"])
            _family, qrel_ordinals = qrels[item_id]
            action_map = self._slate(item)
            rows = tuple(
                core.AFormAction(
                    recipe_id=recipe_id,
                    features=core.compute_action_features(
                        action=action_map[recipe_id],
                        document_texts=stage.document_texts,
                    ),
                    utility=core.utility(
                        top5_document_ordinals=action_map[
                            recipe_id
                        ].top5_document_ordinals,
                        qrel_document_ordinals=qrel_ordinals,
                    ),
                )
                for recipe_id in core.RECIPE_IDS
            )
            slates.append(core.AFormSlate(rows))
        model = core.fit_e1(slates)
        frozen_model = core.model_payload(model)
        payload = self_hashed(
            {
                "A_form_action_archive_self_sha256": stage.archive[
                    "self_sha256"
                ],
                "A_form_qrel_pack_self_sha256": qrel_hash,
                "model": frozen_model,
                "post_action_seal_qrel_open_count": 1,
                "schema": f"{VERSION}_private_E1_model_binding_v1",
                "study_id": STUDY_ID,
            }
        )
        model_path = self.work_root / "evaluator" / "E1.private.json"
        model_file_sha256 = _seal(model_path, payload)
        restored = core.model_from_payload(frozen_model)
        if restored != model:
            raise AveritecP1FormalControllerError("sealed E1 model drifted")
        return model, model_file_sha256

    def _f_trace(self, stage: _Stage) -> tuple[dict[str, object], str]:
        e0_counts: Counter[str] = Counter()
        e1_counts: Counter[str] = Counter()
        changed = 0
        for item in stage.actions:
            e0 = str(item["E0_recipe_id"])
            e1 = str(item["E1_recipe_id"])
            e0_counts[e0] += 1
            e1_counts[e1] += 1
            changed += e0 != e1
        trace = self_hashed(
            {
                "F_search_action_archive_self_sha256": stage.archive[
                    "self_sha256"
                ],
                "E0_recipe_count": dict(sorted(e0_counts.items())),
                "E1_recipe_count": dict(sorted(e1_counts.items())),
                "changed_selection_count": changed,
                "decision_or_gate_count": 0,
                "qrel_pack_exists": False,
                "qrel_open_count": 0,
                "schema": F_TRACE_SCHEMA,
                "status": "completed_descriptive_non_gating_trace",
                "study_id": STUDY_ID,
            }
        )
        file_sha = _seal(
            self.work_root / "scores" / "F_search.trace.private.json", trace
        )
        return trace, file_sha

    @staticmethod
    def _utilities(
        *,
        stage: _Stage,
        qrels: Mapping[str, tuple[str, tuple[int, ...]]],
    ) -> tuple[
        list[str],
        list[Fraction],
        list[Fraction],
        list[Fraction],
        list[Fraction],
    ]:
        families: list[str] = []
        e1_values: list[Fraction] = []
        e0_values: list[Fraction] = []
        raw_values: list[Fraction] = []
        hippo_values: list[Fraction] = []
        for item in stage.actions:
            item_id = str(item["item_id"])
            family, qrel_ordinals = qrels[item_id]
            action_map = FormalController._slate(item)
            e0_id = item.get("E0_recipe_id")
            e1_id = item.get("E1_recipe_id")
            if e0_id not in core.RECIPE_IDS or e1_id not in core.RECIPE_IDS:
                raise AveritecP1FormalControllerError(
                    "sealed evaluator selection drifted"
                )
            raw_top5 = item.get("RAW_top5_document_ordinals")
            hippo_top5 = item.get("HippoRAG_top5_document_ordinals")
            if not isinstance(raw_top5, list):
                raise AveritecP1FormalControllerError("RAW top5 disappeared")
            families.append(family)
            e0_values.append(
                core.utility(
                    top5_document_ordinals=action_map[
                        str(e0_id)
                    ].top5_document_ordinals,
                    qrel_document_ordinals=qrel_ordinals,
                )
            )
            e1_values.append(
                core.utility(
                    top5_document_ordinals=action_map[
                        str(e1_id)
                    ].top5_document_ordinals,
                    qrel_document_ordinals=qrel_ordinals,
                )
            )
            raw_values.append(
                core.utility(
                    top5_document_ordinals=raw_top5,
                    qrel_document_ordinals=qrel_ordinals,
                )
            )
            if hippo_top5 is not None:
                if not isinstance(hippo_top5, list):
                    raise AveritecP1FormalControllerError(
                        "HippoRAG top5 disappeared"
                    )
                hippo_values.append(
                    core.utility(
                        top5_document_ordinals=hippo_top5,
                        qrel_document_ordinals=qrel_ordinals,
                    )
                )
        return families, e1_values, e0_values, raw_values, hippo_values

    @staticmethod
    def _family_comparisons(
        *,
        families: Sequence[str],
        candidate: Sequence[Fraction],
        baseline: Sequence[Fraction],
    ) -> dict[str, core.ExactComparison]:
        result: dict[str, core.ExactComparison] = {}
        for family in acquisition.FAMILIES:
            indexes = [
                index
                for index, observed in enumerate(families)
                if observed == family
            ]
            result[family] = core.compare(
                [candidate[index] for index in indexes],
                [baseline[index] for index in indexes],
            )
        return result

    def _score_a_hold(
        self, stage: _Stage
    ) -> tuple[bool, bool, Mapping[str, object], str]:
        qrels, qrel_hash = self._qrels(
            block=acquisition.A_HOLD,
            queries=stage.queries,
            corpus_count=len(stage.corpus),
        )
        families, e1, e0, raw, hippo = self._utilities(
            stage=stage, qrels=qrels
        )
        if len(hippo) != len(e1):
            raise AveritecP1FormalControllerError(
                "A_hold official HippoRAG arm is incomplete"
            )
        promotion_comparison = core.compare(e1, e0)
        raw_comparison = core.compare(e1, raw)
        hippo_comparison = core.compare(e1, hippo)
        raw_family = self._family_comparisons(
            families=families, candidate=e1, baseline=raw
        )
        hippo_family = self._family_comparisons(
            families=families, candidate=e1, baseline=hippo
        )
        promoted = (
            promotion_comparison.net_utility > 0
            and promotion_comparison.reference_tail <= core.PROMOTION_ALPHA
        )
        reality_passed = (
            raw_comparison.net_utility > 0
            and raw_comparison.reference_tail <= core.PROMOTION_ALPHA
            and hippo_comparison.net_utility > 0
            and hippo_comparison.reference_tail <= core.PROMOTION_ALPHA
            and all(row.net_utility > 0 for row in raw_family.values())
            and all(row.net_utility > 0 for row in hippo_family.values())
        )
        private_rows = []
        for family, e1_value, e0_value, raw_value, hippo_value in zip(
            families, e1, e0, raw, hippo
        ):
            private_rows.append(
                {
                    "E0_utility": _fraction(e0_value),
                    "E1_utility": _fraction(e1_value),
                    "HippoRAG_utility": _fraction(hippo_value),
                    "RAW_utility": _fraction(raw_value),
                    "family": family,
                }
            )
        score = self_hashed(
            {
                "A_hold_action_archive_self_sha256": stage.archive[
                    "self_sha256"
                ],
                "A_hold_qrel_pack_self_sha256": qrel_hash,
                "E1_minus_E0": _comparison(promotion_comparison),
                "E1_minus_HippoRAG": _comparison(hippo_comparison),
                "E1_minus_HippoRAG_by_family": {
                    family: _comparison(hippo_family[family])
                    for family in acquisition.FAMILIES
                },
                "E1_minus_RAW": _comparison(raw_comparison),
                "E1_minus_RAW_by_family": {
                    family: _comparison(raw_family[family])
                    for family in acquisition.FAMILIES
                },
                "complete_case_item_count": len(e1),
                "evaluator_promoted": promoted,
                "item_rows": private_rows,
                "online_evaluator_call_count": 0,
                "post_action_seal_qrel_open_count": 1,
                "reality_three_family_double_baseline_passed": reality_passed,
                "schema": A_HOLD_SCORE_SCHEMA,
                "study_id": STUDY_ID,
            }
        )
        score_file_sha = _seal(
            self.work_root / "scores" / "A_hold.score.private.json", score
        )
        return promoted, reality_passed, score, score_file_sha

    def _score_m_search(
        self,
        stage: _Stage,
        *,
        promotion_self_sha256: str,
    ) -> tuple[bool, Mapping[str, object], str]:
        qrels, qrel_hash = self._qrels(
            block=acquisition.M_SEARCH,
            queries=stage.queries,
            corpus_count=len(stage.corpus),
        )
        families, e1, e0, _raw, hippo = self._utilities(
            stage=stage, qrels=qrels
        )
        if hippo:
            raise AveritecP1FormalControllerError(
                "M_search unexpectedly materialized HippoRAG"
            )
        comparison = core.compare(e1, e0)
        family_rows = self._family_comparisons(
            families=families, candidate=e1, baseline=e0
        )
        passed = (
            comparison.net_utility > 0
            and comparison.reference_tail <= core.PROMOTION_ALPHA
        )
        score = self_hashed(
            {
                "A_hold_promotion_score_self_sha256": promotion_self_sha256,
                "E1_minus_E0": _comparison(comparison),
                "E1_minus_E0_by_family_descriptive": {
                    family: _comparison(family_rows[family])
                    for family in acquisition.FAMILIES
                },
                "L5_passed": passed,
                "M_search_action_archive_self_sha256": stage.archive[
                    "self_sha256"
                ],
                "M_search_qrel_pack_self_sha256": qrel_hash,
                "complete_case_item_count": len(e1),
                "online_evaluator_call_count": 0,
                "post_action_seal_qrel_open_count": 1,
                "schema": M_SEARCH_SCORE_SCHEMA,
                "study_id": STUDY_ID,
            }
        )
        score_file_sha = _seal(
            self.work_root / "scores" / "M_search.score.private.json", score
        )
        return passed, score, score_file_sha

    def _terminal(
        self,
        *,
        status: str,
        stages: Mapping[str, _Stage],
        evidence_file_sha256s: Mapping[str, str],
        f_trace: Mapping[str, object] | None,
        promotion: bool | None,
        reality: bool | None,
        l5: bool | None,
        a_hold_score: Mapping[str, object] | None,
        m_score: Mapping[str, object] | None,
    ) -> dict[str, object]:
        terminal = self_hashed(
            {
                "A_hold_evaluator_promoted": promotion,
                "A_hold_reality_three_family_double_baseline_passed": reality,
                "F_search_changed_selection_count": (
                    None
                    if f_trace is None
                    else f_trace["changed_selection_count"]
                ),
                "F_search_decision_or_gate_count": 0,
                "F_search_qrel_open_count": 0,
                "M_search_L5_passed": l5,
                "aggregate_comparisons": {
                    "A_hold": (
                        None
                        if a_hold_score is None
                        else {
                            "E1_minus_E0": a_hold_score["E1_minus_E0"],
                            "E1_minus_HippoRAG": a_hold_score[
                                "E1_minus_HippoRAG"
                            ],
                            "E1_minus_HippoRAG_by_family": a_hold_score[
                                "E1_minus_HippoRAG_by_family"
                            ],
                            "E1_minus_RAW": a_hold_score["E1_minus_RAW"],
                            "E1_minus_RAW_by_family": a_hold_score[
                                "E1_minus_RAW_by_family"
                            ],
                        }
                    ),
                    "M_search": (
                        None
                        if m_score is None
                        else {
                            "E1_minus_E0": m_score["E1_minus_E0"],
                            "E1_minus_E0_by_family_descriptive": m_score[
                                "E1_minus_E0_by_family_descriptive"
                            ],
                        }
                    ),
                },
                "evidence_file_sha256s": dict(evidence_file_sha256s),
                "execution_binding_sha256": self.execution_binding_sha256,
                "individual_item_or_text_value_published": False,
                "late_qrel_open_count": (
                    1
                    + (1 if a_hold_score is not None else 0)
                    + (1 if m_score is not None else 0)
                ),
                "max_physical_model_lanes": MAX_PHYSICAL_MODEL_LANES,
                "online_or_API_evaluator_call_count": 0,
                "private_action_qrel_or_score_archive_retained_remote": True,
                "retry_replay_resample_gate_change_or_provider_switch_count": 0,
                "schema": FINAL_SCHEMA,
                "stage_action_archive_file_sha256s": {
                    block: stage.archive_file_sha256
                    for block, stage in stages.items()
                },
                "status": status,
                "study_id": STUDY_ID,
            }
        )
        _seal(self.work_root / "formal_terminal.json", terminal)
        return terminal

    def _failure_terminal(self, *, stage: str, exc: BaseException) -> None:
        path = self.work_root / "formal_terminal.json"
        if path.exists() or path.is_symlink():
            return
        failure = self_hashed(
            {
                "exception_message_sha256": hashlib.sha256(
                    str(exc).encode("utf-8")
                ).hexdigest(),
                "exception_type_sha256": hashlib.sha256(
                    type(exc).__qualname__.encode("utf-8")
                ).hexdigest(),
                "execution_binding_sha256": self.execution_binding_sha256,
                "formal_retry_authorized": False,
                "online_evaluator_fallback_authorized": False,
                "schema": FAILURE_SCHEMA,
                "stage": stage,
                "status": "implementation_or_infrastructure_invalid",
                "study_id": STUDY_ID,
            }
        )
        _seal(path, failure)

    def run(self) -> dict[str, object]:
        _make_private_directory(self.work_root)
        stages: dict[str, _Stage] = {}
        evidence: dict[str, str] = {}
        stage_name = "A_form_action"
        try:
            a_form = self._materialize_stage(
                block=acquisition.A_FORM,
                model=None,
                include_hippo=False,
            )
            stages[acquisition.A_FORM] = a_form
            stage_name = "A_form_qrel_and_E1_fit"
            model, model_sha = self._fit_e1(a_form)
            evidence["E1_model"] = model_sha

            stage_name = "F_search_action"
            f_search = self._materialize_stage(
                block=acquisition.F_SEARCH,
                model=model,
                include_hippo=False,
            )
            stages[acquisition.F_SEARCH] = f_search
            f_trace, f_trace_sha = self._f_trace(f_search)
            evidence["F_search_trace"] = f_trace_sha

            stage_name = "A_hold_two_gpu_action"
            a_hold = self._materialize_stage(
                block=acquisition.A_HOLD,
                model=model,
                include_hippo=True,
            )
            stages[acquisition.A_HOLD] = a_hold
            stage_name = "A_hold_late_qrel_score"
            promoted, reality, a_hold_score, a_hold_score_sha = (
                self._score_a_hold(a_hold)
            )
            evidence["A_hold_score"] = a_hold_score_sha
            if not promoted:
                return self._terminal(
                    status="terminal_A_hold_E1_not_promoted",
                    stages=stages,
                    evidence_file_sha256s=evidence,
                    f_trace=f_trace,
                    promotion=False,
                    reality=reality,
                    l5=None,
                    a_hold_score=a_hold_score,
                    m_score=None,
                )

            stage_name = "M_search_action_after_promotion"
            m_search = self._materialize_stage(
                block=acquisition.M_SEARCH,
                model=model,
                include_hippo=False,
            )
            stages[acquisition.M_SEARCH] = m_search
            stage_name = "M_search_late_qrel_score"
            l5, m_score, m_score_sha = self._score_m_search(
                m_search,
                promotion_self_sha256=str(a_hold_score["self_sha256"]),
            )
            evidence["M_search_score"] = m_score_sha
            return self._terminal(
                status="formal_lifecycle_complete",
                stages=stages,
                evidence_file_sha256s=evidence,
                f_trace=f_trace,
                promotion=True,
                reality=reality,
                l5=l5,
                a_hold_score=a_hold_score,
                m_score=m_score,
            )
        except BaseException as exc:
            self._failure_terminal(stage=stage_name, exc=exc)
            raise


__all__ = [
    "VERSION",
    "STUDY_ID",
    "FORMAL_BLOCK_QUERY_COUNTS",
    "FORMAL_FAMILY_COUNTS",
    "MAX_PHYSICAL_MODEL_LANES",
    "AveritecP1FormalControllerError",
    "HippoResult",
    "CoordinateExecutor",
    "HippoExecutor",
    "FormalController",
    "canonical_bytes",
    "stable_hash",
]
