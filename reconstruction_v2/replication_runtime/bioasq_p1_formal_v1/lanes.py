"""Source-free action lanes for the one-shot BioASQ P1 formal runtime.

The coordinate lane sends one exact 176-question initial batch, followed only
when requested by one exact 48-question ``M_search`` batch, to the frozen
BioASQ GPU1 adapter.  The adapter intentionally receives question text rather
than work IDs, so its ``query_ordinal`` output is mapped back to the input
item at that ordinal inside this private runtime.

The official-HippoRAG lane reuses the unchanged generic DSTC9 adapter and
contract.  It starts one asynchronous GPU0 build over the same 2,900-passage
corpus, then permits exactly two 48-question retrieval calls.  Call order is
the only stage discriminator because BioASQ ``FormalItemView`` deliberately
has no block field: the first retrieval is ``A_hold`` and the optional second
retrieval is ``M_search``.

Both lanes are one-shot.  An attempted worker call is never retried, even
after failure, and all content-bearing adapter/retrieval evidence is retained
only in mode-0400 files below private mode-0700 lane roots.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
import hmac
import os
from pathlib import Path
import stat
import threading
from typing import Callable, Mapping, Sequence

from assumption_agent.benchmarks import (
    bioasq_p1_formal_controller_v1 as ctl,
)
from assumption_agent.benchmarks import bioasq_p1_typed_core_v1 as core
from replication_runtime.bioasq_coordinate_scorer_v1 import (
    adapter as coordinate_adapter,
)
from replication_runtime.bioasq_coordinate_scorer_v1 import (
    contract as coordinate_contract,
)
from replication_runtime.dstc9_official_hipporag_v1 import (
    adapter as hippo_adapter,
)
from replication_runtime.dstc9_official_hipporag_v1 import (
    contract as hippo_contract,
)

from .contract import (
    BioasqP1FormalRuntimeError,
    canonical_bytes,
    fresh_private_directory,
    required_sha256,
    with_self_hash,
)


FORMAL_RUNTIME_VERSION = "bioasq_p1_formal_runtime_v1"
STUDY_ID = core.STUDY_ID
INITIAL_COORDINATE_QUERY_COUNT = sum(
    ctl.BLOCK_COUNTS[name] for name in ctl.INITIAL_BLOCKS
)
M_SEARCH_QUERY_COUNT = ctl.BLOCK_COUNTS["M_search"]
HIPPO_QUERY_COUNT = ctl.BLOCK_COUNTS["A_hold"]
HIPPO_STAGES = ("A_hold", "M_search")

CoordinateRun = Callable[..., Mapping[str, object]]
HippoBuild = Callable[..., Mapping[str, object]]
HippoRetrieve = Callable[..., hippo_contract.RetrievalBatch]


def _exclusive_bytes(path: Path, raw: bytes, *, mode: int = 0o400) -> None:
    """Write one private artifact with O_EXCL and durable final permissions."""

    if (
        not isinstance(path, Path)
        or not path.is_absolute()
        or path.parent.is_symlink()
        or not path.parent.is_dir()
    ):
        raise BioasqP1FormalRuntimeError(
            "private lane archive path is unsafe"
        )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, mode)
        os.fchmod(descriptor, mode)
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short private archive write")
            view = view[written:]
        os.fsync(descriptor)
    except OSError as exc:
        raise BioasqP1FormalRuntimeError(
            f"exclusive private lane archive failed: {path.name}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    metadata = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != mode
    ):
        raise BioasqP1FormalRuntimeError(
            "private lane archive permissions drifted"
        )


def _ensure_adapter_stage_root(path: Path) -> Path:
    """Accept an adapter-created private root or create it for a mock worker."""

    if not path.exists():
        try:
            path.mkdir(mode=0o700, parents=False, exist_ok=False)
        except OSError as exc:
            raise BioasqP1FormalRuntimeError(
                "adapter stage root could not be retained privately"
            ) from exc
    metadata = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise BioasqP1FormalRuntimeError(
            "adapter stage root permissions drifted"
        )
    return path


def _question_sha256(item: ctl.FormalItemView) -> str:
    return hashlib.sha256(item.question_text.encode("utf-8")).hexdigest()


class CoordinateScorerLane:
    """Exactly one initial-176 and at most one conditional-M48 GPU1 call."""

    def __init__(
        self,
        *,
        runtime_python: Path,
        project_root: Path,
        minilm_asset_manifest: Path,
        minilm_model_root: Path,
        cross_encoder_model_root: Path,
        expected_model_binding_sha256: str,
        lane_root: Path,
        timeout_seconds: int,
        run_callable: CoordinateRun = (
            coordinate_adapter.run_bioasq_coordinate_scorer_v1
        ),
    ) -> None:
        self._runtime_python = runtime_python
        self._project_root = project_root
        self._minilm_asset_manifest = minilm_asset_manifest
        self._minilm_model_root = minilm_model_root
        self._cross_encoder_model_root = cross_encoder_model_root
        self._expected_model_binding_sha256 = required_sha256(
            expected_model_binding_sha256,
            "expected coordinate model binding",
        )
        self._lane_root = fresh_private_directory(
            lane_root, "coordinate lane root"
        )
        if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 14_400:
            raise BioasqP1FormalRuntimeError(
                "coordinate timeout is outside the frozen integer bound"
            )
        if not callable(run_callable):
            raise BioasqP1FormalRuntimeError(
                "coordinate adapter callable is unavailable"
            )
        self._timeout_seconds = timeout_seconds
        self._run = run_callable
        self._corpus_view_sha256: str | None = None
        self._initial_work_ids: tuple[str, ...] | None = None
        self._m_work_ids: tuple[str, ...] | None = None
        self._initial_cache: dict[str, ctl.CoordinateScoreRow] | None = None
        self._m_cache: dict[str, ctl.CoordinateScoreRow] | None = None
        self._m_attempted = False
        self._worker_call_count = 0
        self._private_output_commitments: dict[str, str] = {}
        self._private_receipt_commitments: dict[str, str] = {}
        self._lock = threading.Lock()

    @property
    def worker_call_count(self) -> int:
        return self._worker_call_count

    @property
    def private_output_commitments(self) -> Mapping[str, str]:
        return dict(self._private_output_commitments)

    @property
    def private_receipt_commitments(self) -> Mapping[str, str]:
        return dict(self._private_receipt_commitments)

    @staticmethod
    def _validate_public_batch(
        corpus: ctl.CorpusView,
        items: Sequence[ctl.FormalItemView],
        *,
        expected_count: int,
    ) -> tuple[ctl.FormalItemView, ...]:
        if not isinstance(corpus, ctl.CorpusView):
            raise BioasqP1FormalRuntimeError(
                "coordinate corpus type drifted"
            )
        checked = tuple(items)
        if (
            len(checked) != expected_count
            or any(not isinstance(item, ctl.FormalItemView) for item in checked)
            or len({item.work_id for item in checked}) != expected_count
        ):
            raise BioasqP1FormalRuntimeError(
                f"coordinate batch must contain exactly {expected_count} "
                "unique public items"
            )
        return checked

    def _execute(
        self,
        corpus: ctl.CorpusView,
        items: Sequence[ctl.FormalItemView],
        *,
        stage_name: str,
    ) -> dict[str, ctl.CoordinateScoreRow]:
        """Run one adapter call and bind each ordinal row to its input item."""

        checked = tuple(items)
        passages = tuple(
            core.passage_public_payload(row) for row in corpus.passages
        )
        queries = tuple({"text": item.question_text} for item in checked)
        try:
            input_value = coordinate_contract.input_payload(
                passages=passages,
                queries=queries,
            )
            scorer_input = coordinate_contract.validate_input(input_value)
        except Exception as exc:
            raise BioasqP1FormalRuntimeError(
                "coordinate source-free input validation failed"
            ) from exc
        if (
            scorer_input.passage_projection_sha256
            != corpus.projection_sha256
            or len(scorer_input.queries) != len(checked)
            or tuple(row.text for row in scorer_input.queries)
            != tuple(item.question_text for item in checked)
        ):
            raise BioasqP1FormalRuntimeError(
                "coordinate input corpus/query binding drifted"
            )

        stage_root = self._lane_root / stage_name
        self._worker_call_count += 1
        try:
            output = self._run(
                input_value=input_value,
                runtime_python=self._runtime_python,
                project_root=self._project_root,
                minilm_asset_manifest=self._minilm_asset_manifest,
                minilm_model_root=self._minilm_model_root,
                cross_encoder_model_root=self._cross_encoder_model_root,
                work_root=stage_root,
                timeout_seconds=self._timeout_seconds,
            )
        except Exception as exc:
            raise BioasqP1FormalRuntimeError(
                "coordinate adapter failed without retry"
            ) from exc

        receipt = output.get("receipt") if isinstance(output, Mapping) else None
        if not isinstance(receipt, Mapping):
            raise BioasqP1FormalRuntimeError(
                "coordinate adapter receipt is unavailable"
            )
        try:
            model_binding_sha256 = required_sha256(
                receipt.get("model_binding_sha256"),
                "coordinate model binding",
            )
            if (
                model_binding_sha256
                != self._expected_model_binding_sha256
            ):
                raise BioasqP1FormalRuntimeError(
                    "formal coordinate model differs from the canary"
                )
            validated = coordinate_contract.validate_output(
                output,
                expected_input=scorer_input,
                expected_model_binding_sha256=(
                    self._expected_model_binding_sha256
                ),
            )
            receipt_sha256 = required_sha256(
                validated["receipt"].get("receipt_sha256"),
                "coordinate receipt",
            )
            output_self_sha256 = required_sha256(
                validated.get("self_sha256"),
                "coordinate output",
            )
        except Exception as exc:
            if isinstance(exc, BioasqP1FormalRuntimeError):
                raise
            raise BioasqP1FormalRuntimeError(
                "coordinate adapter output/receipt validation failed"
            ) from exc
        if (
            validated.get("input_self_sha256") != scorer_input.self_sha256
            or validated.get("query_count") != len(checked)
            or validated.get("corpus_count") != ctl.CORPUS_SIZE
            or validated["receipt"].get("passage_projection_sha256")
            != corpus.projection_sha256
            or validated["receipt"].get("retry_count") != 0
        ):
            raise BioasqP1FormalRuntimeError(
                "coordinate adapter receipt input/corpus binding drifted"
            )

        raw_rows = validated.get("rows")
        if not isinstance(raw_rows, list) or len(raw_rows) != len(checked):
            raise BioasqP1FormalRuntimeError(
                "coordinate adapter output coverage drifted"
            )
        by_work: dict[str, ctl.CoordinateScoreRow] = {}
        seen_ordinals: set[int] = set()
        for raw in raw_rows:
            if not isinstance(raw, Mapping):
                raise BioasqP1FormalRuntimeError(
                    "coordinate adapter row drifted"
                )
            ordinal = raw.get("query_ordinal")
            vectors = raw.get("vectors")
            if (
                type(ordinal) is not int
                or not 0 <= ordinal < len(checked)
                or ordinal in seen_ordinals
                or not isinstance(vectors, Mapping)
            ):
                raise BioasqP1FormalRuntimeError(
                    "coordinate query ordinal binding drifted"
                )
            seen_ordinals.add(ordinal)
            item = checked[ordinal]
            by_work[item.work_id] = ctl.CoordinateScoreRow.create(
                item=item,
                corpus=corpus,
                score_vectors=vectors,
            )
        if seen_ordinals != set(range(len(checked))):
            raise BioasqP1FormalRuntimeError(
                "coordinate query ordinal coverage drifted"
            )

        private_root = _ensure_adapter_stage_root(stage_root)
        private_path = private_root / "adapter_output.private.json"
        _exclusive_bytes(
            private_path,
            coordinate_contract.canonical_bytes(validated),
            mode=0o400,
        )
        if not hmac.compare_digest(
            hashlib.sha256(
                coordinate_contract.canonical_bytes(validated)
            ).hexdigest(),
            hashlib.sha256(private_path.read_bytes()).hexdigest(),
        ):
            raise BioasqP1FormalRuntimeError(
                "coordinate private output archive drifted"
            )
        self._private_output_commitments[stage_name] = output_self_sha256
        self._private_receipt_commitments[stage_name] = receipt_sha256
        return by_work

    def score(
        self,
        corpus: ctl.CorpusView,
        items: Sequence[ctl.FormalItemView],
    ) -> Sequence[ctl.CoordinateScoreRow]:
        """Serve the controller's exact initial batch and optional M batch."""

        with self._lock:
            if self._corpus_view_sha256 is None:
                checked = self._validate_public_batch(
                    corpus,
                    items,
                    expected_count=INITIAL_COORDINATE_QUERY_COUNT,
                )
                self._corpus_view_sha256 = corpus.view_sha256
                self._initial_work_ids = tuple(
                    item.work_id for item in checked
                )
                self._initial_cache = self._execute(
                    corpus,
                    checked,
                    stage_name="initial_176",
                )
                return tuple(
                    self._initial_cache[item.work_id] for item in checked
                )

            if self._corpus_view_sha256 != corpus.view_sha256:
                raise BioasqP1FormalRuntimeError(
                    "coordinate corpus changed after the first batch"
                )
            checked_items = tuple(items)
            work_ids = tuple(item.work_id for item in checked_items)
            if work_ids == self._initial_work_ids:
                if self._initial_cache is None:
                    raise BioasqP1FormalRuntimeError(
                        "initial coordinate attempt failed and cannot retry"
                    )
                return tuple(
                    self._initial_cache[item.work_id]
                    for item in checked_items
                )
            if work_ids == self._m_work_ids:
                if self._m_cache is None:
                    raise BioasqP1FormalRuntimeError(
                        "M_search coordinate attempt failed and cannot retry"
                    )
                return tuple(
                    self._m_cache[item.work_id] for item in checked_items
                )
            checked = self._validate_public_batch(
                corpus,
                checked_items,
                expected_count=M_SEARCH_QUERY_COUNT,
            )
            if (
                self._m_attempted
                or self._initial_work_ids is None
                or set(work_ids).intersection(self._initial_work_ids)
            ):
                raise BioasqP1FormalRuntimeError(
                    "coordinate batch lifecycle drifted"
                )
            self._m_attempted = True
            self._m_work_ids = work_ids
            self._m_cache = self._execute(
                corpus,
                checked,
                stage_name="M_search_48",
            )
            return tuple(self._m_cache[item.work_id] for item in checked)


# The short name is convenient in configs and preserves the task vocabulary.
CoordinateLane = CoordinateScorerLane


class OfficialHippoLane:
    """One asynchronous global build and ordered A_hold/M retrievals."""

    def __init__(
        self,
        *,
        runtime_python: Path,
        worker_project_root: Path,
        current_hardware_binding_path: Path,
        local_llm_model: Path,
        local_embedding_model: Path,
        runtime_fingerprint_path: Path,
        lane_root: Path,
        build_timeout_seconds: int,
        retrieve_timeout_seconds: int,
        build_callable: HippoBuild = (
            hippo_adapter.build_dstc9_official_hipporag_global_index_v1
        ),
        retrieve_callable: HippoRetrieve = (
            hippo_adapter.retrieve_dstc9_official_hipporag_global_index_v1
        ),
    ) -> None:
        for value, name in (
            (build_timeout_seconds, "HippoRAG build timeout"),
            (retrieve_timeout_seconds, "HippoRAG retrieve timeout"),
        ):
            if type(value) is not int or not 1 <= value <= 14_400:
                raise BioasqP1FormalRuntimeError(
                    f"{name} is outside the frozen integer bound"
                )
        if not callable(build_callable) or not callable(retrieve_callable):
            raise BioasqP1FormalRuntimeError(
                "HippoRAG adapter callable is unavailable"
            )
        self._runtime_python = runtime_python
        self._worker_project_root = worker_project_root
        self._current_hardware_binding_path = current_hardware_binding_path
        self._local_llm_model = local_llm_model
        self._local_embedding_model = local_embedding_model
        self._runtime_fingerprint_path = runtime_fingerprint_path
        self._lane_root = fresh_private_directory(
            lane_root, "HippoRAG lane root"
        )
        self._stage_root = self._lane_root / "global_build"
        self._build_timeout_seconds = build_timeout_seconds
        self._retrieve_timeout_seconds = retrieve_timeout_seconds
        self._build = build_callable
        self._retrieve = retrieve_callable
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="bioasq-hippo-build",
        )
        self._build_future: Future[Mapping[str, object]] | None = None
        self._build_receipt: Mapping[str, object] | None = None
        self._corpus_view_sha256: str | None = None
        self._corpus_input: hippo_contract.CorpusInput | None = None
        self._retrieval_attempt_count = 0
        self._retrieved_work_ids: set[str] = set()
        self._private_retrieval_commitments: dict[str, str] = {}
        self._lock = threading.Lock()
        self._closed = False

    @property
    def build_call_count(self) -> int:
        return int(self._build_future is not None)

    @property
    def retrieve_call_count(self) -> int:
        return self._retrieval_attempt_count

    @property
    def private_retrieval_commitments(self) -> Mapping[str, str]:
        return dict(self._private_retrieval_commitments)

    @staticmethod
    def _make_corpus_input(
        corpus: ctl.CorpusView,
    ) -> tuple[dict[str, object], hippo_contract.CorpusInput]:
        if not isinstance(corpus, ctl.CorpusView):
            raise BioasqP1FormalRuntimeError(
                "HippoRAG build corpus type drifted"
            )
        try:
            payload = hippo_contract.make_corpus_input(
                study_id=STUDY_ID,
                units=tuple(
                    {
                        "ordinal": passage.ordinal,
                        "text": core.serialize_passage(passage),
                    }
                    for passage in corpus.passages
                ),
            )
            validated = hippo_contract.validate_corpus_input(payload)
        except Exception as exc:
            raise BioasqP1FormalRuntimeError(
                "HippoRAG corpus input validation failed"
            ) from exc
        if (
            len(validated.units) != ctl.CORPUS_SIZE
            or tuple(unit.ordinal for unit in validated.units)
            != tuple(range(ctl.CORPUS_SIZE))
            or tuple(unit.text for unit in validated.units)
            != tuple(core.serialize_passage(row) for row in corpus.passages)
        ):
            raise BioasqP1FormalRuntimeError(
                "HippoRAG corpus projection drifted"
            )
        return payload, validated

    def _build_once(
        self, corpus_input: Mapping[str, object]
    ) -> Mapping[str, object]:
        try:
            receipt = self._build(
                corpus_input=corpus_input,
                runtime_python=self._runtime_python,
                worker_project_root=self._worker_project_root,
                current_hardware_binding_path=(
                    self._current_hardware_binding_path
                ),
                local_llm_model=self._local_llm_model,
                local_embedding_model=self._local_embedding_model,
                runtime_fingerprint_path=self._runtime_fingerprint_path,
                stage_root=self._stage_root,
                timeout_seconds=self._build_timeout_seconds,
            )
        except Exception as exc:
            raise BioasqP1FormalRuntimeError(
                "HippoRAG build failed without retry"
            ) from exc
        if not isinstance(receipt, Mapping):
            raise BioasqP1FormalRuntimeError(
                "HippoRAG build receipt is unavailable"
            )
        self._build_receipt = dict(receipt)
        return self._build_receipt

    def start_build(self, corpus: ctl.CorpusView) -> None:
        """Start the only index build at the earliest public-corpus boundary."""

        corpus_payload, validated = self._make_corpus_input(corpus)
        with self._lock:
            if self._closed or self._build_future is not None:
                raise BioasqP1FormalRuntimeError(
                    "HippoRAG global build lifecycle drifted"
                )
            self._corpus_view_sha256 = corpus.view_sha256
            self._corpus_input = validated
            self._build_future = self._executor.submit(
                self._build_once,
                corpus_payload,
            )

    @staticmethod
    def _validate_items(
        items: Sequence[ctl.FormalItemView],
    ) -> tuple[ctl.FormalItemView, ...]:
        checked = tuple(items)
        if (
            len(checked) != HIPPO_QUERY_COUNT
            or any(not isinstance(item, ctl.FormalItemView) for item in checked)
            or len({item.work_id for item in checked}) != HIPPO_QUERY_COUNT
        ):
            raise BioasqP1FormalRuntimeError(
                "HippoRAG retrieval requires exactly 48 unique public items"
            )
        return checked

    def retrieve(
        self,
        corpus: ctl.CorpusView,
        items: Sequence[ctl.FormalItemView],
    ) -> Sequence[ctl.HippoResult]:
        """Retrieve A_hold first and, if promoted, M_search second."""

        checked = self._validate_items(items)
        if not isinstance(corpus, ctl.CorpusView):
            raise BioasqP1FormalRuntimeError(
                "HippoRAG retrieve corpus type drifted"
            )
        work_ids = {item.work_id for item in checked}
        with self._lock:
            if (
                self._closed
                or self._build_future is None
                or self._corpus_view_sha256 != corpus.view_sha256
                or self._retrieval_attempt_count >= len(HIPPO_STAGES)
                or bool(work_ids.intersection(self._retrieved_work_ids))
            ):
                raise BioasqP1FormalRuntimeError(
                    "HippoRAG retrieve lifecycle drifted"
                )
            stage = HIPPO_STAGES[self._retrieval_attempt_count]
            self._retrieval_attempt_count += 1
            self._retrieved_work_ids.update(work_ids)
            future = self._build_future

        # Waiting outside the lock lets the GPU0 build overlap the GPU1 lane.
        future.result()
        try:
            query_payload = hippo_contract.make_query_input(
                study_id=STUDY_ID,
                queries=tuple(
                    {
                        "ordinal": ordinal,
                        "query_text": item.question_text,
                        "work_id": item.work_id,
                    }
                    for ordinal, item in enumerate(checked)
                ),
            )
            query_input = hippo_contract.validate_query_input(
                query_payload,
                expected_study_id=STUDY_ID,
            )
        except Exception as exc:
            raise BioasqP1FormalRuntimeError(
                "HippoRAG query input validation failed"
            ) from exc
        if (
            tuple(row.work_id for row in query_input.queries)
            != tuple(item.work_id for item in checked)
            or tuple(row.query_text for row in query_input.queries)
            != tuple(item.question_text for item in checked)
        ):
            raise BioasqP1FormalRuntimeError(
                "HippoRAG exact query projection drifted"
            )
        try:
            batch = self._retrieve(
                query_input=query_payload,
                runtime_python=self._runtime_python,
                worker_project_root=self._worker_project_root,
                current_hardware_binding_path=(
                    self._current_hardware_binding_path
                ),
                local_llm_model=self._local_llm_model,
                local_embedding_model=self._local_embedding_model,
                runtime_fingerprint_path=self._runtime_fingerprint_path,
                stage_root=self._stage_root,
                work_root=self._lane_root / f"retrieve_{stage}",
                timeout_seconds=self._retrieve_timeout_seconds,
            )
        except Exception as exc:
            raise BioasqP1FormalRuntimeError(
                f"HippoRAG {stage} retrieve failed without retry"
            ) from exc
        if (
            not isinstance(batch, hippo_contract.RetrievalBatch)
            or len(batch.ordinals) != len(checked)
            or any(
                len(row) != core.TOP_K
                or len(set(row)) != core.TOP_K
                or any(
                    type(value) is not int
                    or not 0 <= value < ctl.CORPUS_SIZE
                    for value in row
                )
                for row in batch.ordinals
            )
            or not isinstance(batch.receipt, Mapping)
        ):
            raise BioasqP1FormalRuntimeError(
                "HippoRAG retrieval result coverage drifted"
            )
        receipt_sha256 = required_sha256(
            batch.receipt.get("receipt_sha256"),
            "HippoRAG retrieval receipt",
        )
        # The real unchanged adapter supplies these bindings.  Synthetic unit
        # seams may provide only the receipt hash, so validate every binding
        # that is present without weakening the production contract.
        expected_receipt_fields = {
            "corpus_count": ctl.CORPUS_SIZE,
            "corpus_input_self_sha256": (
                self._corpus_input.self_sha256
                if self._corpus_input is not None
                else None
            ),
            "query_count": len(checked),
            "query_input_self_sha256": query_input.self_sha256,
            "retry_count": 0,
            "study_id": STUDY_ID,
        }
        if any(
            field in batch.receipt and batch.receipt.get(field) != expected
            for field, expected in expected_receipt_fields.items()
        ):
            raise BioasqP1FormalRuntimeError(
                "HippoRAG retrieval receipt corpus/query binding drifted"
            )

        private_evidence = with_self_hash(
            {
                "block": stage,
                "build_once": True,
                "corpus_projection_sha256": corpus.projection_sha256,
                "query_input_self_sha256": query_input.self_sha256,
                "receipt": dict(batch.receipt),
                "retrieved_ordinals": [
                    {
                        "top5_ordinals": list(ordinals),
                        "work_id": item.work_id,
                    }
                    for item, ordinals in zip(checked, batch.ordinals)
                ],
                "schema": (
                    f"{FORMAL_RUNTIME_VERSION}_"
                    "private_hipporag_retrieval_evidence_v1"
                ),
                "study_id": STUDY_ID,
            }
        )
        private_path = self._lane_root / f"{stage}.retrieval.private.json"
        _exclusive_bytes(
            private_path,
            canonical_bytes(private_evidence),
            mode=0o400,
        )
        self._private_retrieval_commitments[stage] = str(
            private_evidence["self_sha256"]
        )
        return tuple(
            ctl.HippoResult(
                work_id=item.work_id,
                normalized_query_sha256=_question_sha256(item),
                corpus_projection_sha256=corpus.projection_sha256,
                top5_ordinals=tuple(ordinals),
                receipt_sha256=receipt_sha256,
            )
            for item, ordinals in zip(checked, batch.ordinals)
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=False)


__all__ = [
    "CoordinateLane",
    "CoordinateScorerLane",
    "HIPPO_QUERY_COUNT",
    "HIPPO_STAGES",
    "INITIAL_COORDINATE_QUERY_COUNT",
    "M_SEARCH_QUERY_COUNT",
    "OfficialHippoLane",
]
