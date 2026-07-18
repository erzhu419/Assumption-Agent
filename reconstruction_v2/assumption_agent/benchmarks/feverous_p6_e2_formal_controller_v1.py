"""One-shot controller for the frozen FEVEROUS P6/E2 lifecycle.

The controller is deliberately capability ordered.  Before its exclusive
marker it may only verify the committed implementation/acquisition and perform
the model-free runtime preflight.  After the marker it opens claim-only views,
persists content-free action archives, and opens each label pack only after the
corresponding archive and seal have been durably verified.  ``M_search`` has no
code path before an explicit A_hold promotion authorization.

No function in this module retries, resamples, or calls an online evaluator.
Formal execution is intentionally non-injectable; the injectable core exists
only for offline synthetic lifecycle tests.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Protocol

from assumption_agent.benchmarks import feverous_e2_evaluator_v1 as evaluator
from assumption_agent.benchmarks import feverous_offline_semantic_tensor_v1 as semantic
from assumption_agent.benchmarks import feverous_p6_e2_formal_acquisition_v2 as formal_acquisition
from assumption_agent.benchmarks import feverous_p6_e2_formal_runner_v1 as runner
from assumption_agent.benchmarks import feverous_p6_e2_implementation_freeze_v1 as implementation_freeze
from assumption_agent.benchmarks import feverous_local_runtime_v1 as local_runtime
from replication_runtime.feverous_official_hipporag_v1.contract import RetrievalBatch


VERSION = "feverous_p6_e2_formal_controller_v1"
MARKER_SCHEMA = f"{VERSION}_one_shot_marker"
ARCHIVE_SCHEMA = f"{VERSION}_label_free_archive"
SEAL_SCHEMA = f"{VERSION}_label_free_archive_seal"
FAILURE_SCHEMA = f"{VERSION}_terminal_failure"
RESULT_SCHEMA = f"{VERSION}_terminal_result"

FORMAL_OUTPUT_ROOT_RELATIVE = Path(
    "artifacts/feverous_p6_e2_formal_v2/controller"
)
BLOCK_ORDER = ("A_form", "F_search", "A_hold", "M_search")
FORMATION_BLOCKS = ("A_form", "F_search")
ANCHOR_BLOCKS = ("A_hold", "M_search")
BLOCK_COUNTS = dict(runner.BLOCK_COUNTS)
FORMATION_QUERY_COUNT = BLOCK_COUNTS["A_form"] + BLOCK_COUNTS["F_search"]


class FeverousFormalControllerError(RuntimeError):
    """A capability order, one-shot, archive, or freeze invariant drifted."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise FeverousFormalControllerError("value is not canonical JSON") from exc


def stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FeverousFormalControllerError(f"{field} is not a lowercase SHA-256")
    return value


def _self_hashed(
    body: Mapping[str, Any], *, field: str
) -> dict[str, Any]:
    if field in body:
        raise FeverousFormalControllerError("self-hash field already exists")
    return {**dict(body), field: stable_hash(body)}


def verify_self_hash(payload: Mapping[str, Any], *, field: str) -> str:
    if not isinstance(payload, Mapping):
        raise FeverousFormalControllerError("self-hashed payload is not an object")
    body = dict(payload)
    declared = _require_sha256(body.pop(field, None), field)
    if stable_hash(body) != declared:
        raise FeverousFormalControllerError(f"{field} self-hash drifted")
    return declared


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise FeverousFormalControllerError("artifact cannot be hashed") from exc
    return digest.hexdigest()


def _canonical_project(project: str | Path) -> Path:
    try:
        root = Path(project).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise FeverousFormalControllerError("project root is unavailable") from exc
    if not root.is_dir() or root.is_symlink():
        raise FeverousFormalControllerError("project root is unsafe")
    return root


def _assert_under_project(*, project: Path, path: Path) -> None:
    try:
        relative = path.absolute().relative_to(project)
    except ValueError as exc:
        raise FeverousFormalControllerError("controller path escaped project") from exc
    cursor = project
    for component in relative.parts[:-1]:
        cursor = cursor / component
        try:
            metadata = cursor.lstat()
        except FileNotFoundError:
            return
        except OSError as exc:
            raise FeverousFormalControllerError(
                "controller path ancestor is unavailable"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise FeverousFormalControllerError("controller path ancestor is unsafe")


def _ensure_private_directory(*, project: Path, path: Path) -> None:
    _assert_under_project(project=project, path=path)
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as exc:
        raise FeverousFormalControllerError("private directory creation failed") from exc
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise FeverousFormalControllerError("private directory is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise FeverousFormalControllerError("private directory is unsafe")


def write_json_exclusive(path: Path, payload: Mapping[str, Any], *, mode: int) -> str:
    """Durably write canonical JSON once, with no final-component symlink."""

    if mode not in {0o600, 0o644}:
        raise FeverousFormalControllerError("artifact mode is invalid")
    raw = _canonical_bytes(payload) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, mode)
    except OSError as exc:
        raise FeverousFormalControllerError(
            "exclusive artifact already exists or is unsafe"
        ) from exc
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        metadata = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode):
            raise FeverousFormalControllerError("exclusive artifact type drifted")
        parent_descriptor = os.open(
            path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    return hashlib.sha256(raw).hexdigest()


def _load_canonical_json(path: Path) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise FeverousFormalControllerError("controller artifact is unavailable")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FeverousFormalControllerError("controller artifact is invalid") from exc
    if not isinstance(value, dict) or raw != _canonical_bytes(value) + b"\n":
        raise FeverousFormalControllerError("controller artifact is noncanonical")
    return value, hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class LifecycleOutputPaths:
    root_relative: Path = FORMAL_OUTPUT_ROOT_RELATIVE

    @property
    def marker_relative(self) -> Path:
        return self.root_relative / "lifecycle.one_shot_marker.json"

    @property
    def failure_relative(self) -> Path:
        return self.root_relative / "lifecycle.terminal_failure.json"

    @property
    def result_relative(self) -> Path:
        return self.root_relative / "lifecycle.terminal_result.json"

    def archive_relative(self, block: str) -> Path:
        _require_block(block)
        return self.root_relative / f"{block}.label_free.archive.json"

    def seal_relative(self, block: str) -> Path:
        _require_block(block)
        return self.root_relative / f"{block}.label_free.seal.json"

    def receipt_relative(self, name: str) -> Path:
        if name not in {
            "A_form_E2_fit",
            "F_search_policy_freeze",
            "A_hold_score",
            "A_hold_promotion",
            "M_search_score",
            "runtime_postflight",
        }:
            raise FeverousFormalControllerError("receipt output identity is invalid")
        return self.root_relative / f"{name}.json"


FORMAL_OUTPUT_PATHS = LifecycleOutputPaths()


def _require_block(block: str) -> str:
    if block not in BLOCK_ORDER:
        raise FeverousFormalControllerError("block identity is invalid")
    return block


@dataclass(frozen=True)
class PrerequisiteBinding:
    implementation_freeze_sha256: str
    acquisition_receipt_sha256: str
    runtime_preflight_sha256: str
    implementation_git_commit: str

    def validate(self) -> "PrerequisiteBinding":
        _require_sha256(self.implementation_freeze_sha256, "implementation freeze")
        _require_sha256(self.acquisition_receipt_sha256, "acquisition receipt")
        _require_sha256(self.runtime_preflight_sha256, "runtime preflight")
        if (
            not isinstance(self.implementation_git_commit, str)
            or len(self.implementation_git_commit) != 40
            or any(
                character not in "0123456789abcdef"
                for character in self.implementation_git_commit
            )
        ):
            raise FeverousFormalControllerError("implementation commit is invalid")
        return self


@dataclass(frozen=True)
class LifecycleArtifact:
    kind: str
    block: str | None
    path: Path
    receipt_sha256: str
    file_sha256: str
    payload: Mapping[str, Any]

    def validate(self, *, kind: str, block: str | None) -> "LifecycleArtifact":
        if self.kind != kind or self.block != block:
            raise FeverousFormalControllerError("artifact identity drifted")
        _require_sha256(self.receipt_sha256, f"{kind} receipt")
        _require_sha256(self.file_sha256, f"{kind} file")
        if not isinstance(self.payload, Mapping):
            raise FeverousFormalControllerError("artifact payload drifted")
        return self


@dataclass(frozen=True)
class PreparedExecution:
    semantic_corpus: semantic.PreparedSemanticCorpus
    hippo_build_receipt: Mapping[str, Any]


@dataclass(frozen=True)
class LabelFreeBlockStage:
    block: str
    local: runner.BlockExecution
    hippo_top5: tuple[tuple[int, ...], ...]
    hippo_receipt: Mapping[str, Any]
    execution_receipt: Mapping[str, Any]


@dataclass(frozen=True)
class FormationStages:
    A_form: LabelFreeBlockStage
    F_search: LabelFreeBlockStage
    receipt: Mapping[str, Any]


class AcquisitionBoundary(Protocol):
    def verify_prerequisites(self, *, project: Path) -> PrerequisiteBinding: ...

    def assert_stable(
        self, *, project: Path, prerequisites: PrerequisiteBinding
    ) -> None: ...

    def preflight_outputs(
        self,
        *,
        project: Path,
        runtime_config: object,
        output_paths: LifecycleOutputPaths,
    ) -> None: ...

    def load_corpus_view(self, *, project: Path) -> Mapping[str, Any]: ...

    def load_block_view(
        self, *, project: Path, expected_block: str
    ) -> Mapping[str, Any]: ...

    def load_block_labels(
        self, *, project: Path, expected_block: str
    ) -> Mapping[str, Any]: ...

    def load_private_secret(self, *, project: Path) -> bytes: ...


class RuntimeBoundary(Protocol):
    def preflight(self, runtime_config: object) -> Mapping[str, Any]: ...

    def open(self, runtime_config: object) -> AbstractContextManager[object]: ...

    def postflight(self, runtime: object) -> Mapping[str, Any]: ...


class FormalCore(Protocol):
    def prepare(
        self, *, corpus_view: Mapping[str, Any], runtime: object
    ) -> object: ...

    def execute_formation(
        self,
        *,
        A_form_view: Mapping[str, Any],
        F_search_view: Mapping[str, Any],
        prepared: object,
        runtime: object,
    ) -> object: ...

    def formation_blocks(self, formation: object) -> Mapping[str, object]: ...

    def execute_anchor(
        self,
        *,
        block: str,
        view: Mapping[str, Any],
        prepared: object,
        runtime: object,
    ) -> object: ...

    def archive_payload(self, stage: object, *, block: str) -> Mapping[str, Any]: ...

    def fit_e2(
        self,
        *,
        A_form_stage: object,
        labels: Mapping[str, Any],
        fold_secret: bytes,
    ) -> Mapping[str, Any]: ...

    def freeze_f_policies(
        self,
        *,
        F_search_stage: object,
        A_form_stage: object,
        fit_receipt: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def score_anchor(
        self,
        *,
        stage: object,
        labels: Mapping[str, Any],
        policy_receipt: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


def formation_query_schedule(
    A_form_claims: Sequence[str], F_search_claims: Sequence[str]
) -> tuple[tuple[str, ...], tuple[tuple[str, int], ...]]:
    """Return the one frozen interleaved 96+48 official-query schedule."""

    if (
        len(A_form_claims) != BLOCK_COUNTS["A_form"]
        or len(F_search_claims) != BLOCK_COUNTS["F_search"]
        or any(
            not isinstance(claim, str) or not claim or "\x00" in claim
            for claim in (*A_form_claims, *F_search_claims)
        )
    ):
        raise FeverousFormalControllerError("formation query shape drifted")
    by_block = {
        "A_form": tuple(A_form_claims),
        "F_search": tuple(F_search_claims),
    }
    schedule = tuple(
        (block, ordinal)
        for ordinal in range(BLOCK_COUNTS["A_form"])
        for block in FORMATION_BLOCKS
        if ordinal < len(by_block[block])
    )
    queries = tuple(by_block[block][ordinal] for block, ordinal in schedule)
    if len(schedule) != FORMATION_QUERY_COUNT:
        raise FeverousFormalControllerError("formation interleaving drifted")
    return queries, schedule


def split_formation_hippo_result(
    batch: RetrievalBatch,
    *,
    schedule: Sequence[tuple[str, int]],
) -> Mapping[str, tuple[tuple[int, ...], ...]]:
    """Split one 144-query official return without changing its row order."""

    if not isinstance(batch, RetrievalBatch) or len(batch.indices) != FORMATION_QUERY_COUNT:
        raise FeverousFormalControllerError("combined Hippo result count drifted")
    if len(schedule) != FORMATION_QUERY_COUNT:
        raise FeverousFormalControllerError("combined Hippo schedule count drifted")
    rows: dict[str, list[tuple[int, ...] | None]] = {
        block: [None] * BLOCK_COUNTS[block] for block in FORMATION_BLOCKS
    }
    for position, coordinate in enumerate(schedule):
        if (
            not isinstance(coordinate, tuple)
            or len(coordinate) != 2
            or coordinate[0] not in FORMATION_BLOCKS
            or type(coordinate[1]) is not int
            or not 0 <= coordinate[1] < BLOCK_COUNTS[coordinate[0]]
            or rows[coordinate[0]][coordinate[1]] is not None
        ):
            raise FeverousFormalControllerError("combined Hippo schedule drifted")
        rows[coordinate[0]][coordinate[1]] = tuple(batch.indices[position])
    if any(value is None for block_rows in rows.values() for value in block_rows):
        raise FeverousFormalControllerError("combined Hippo projection is incomplete")
    return {
        block: tuple(value for value in block_rows if value is not None)
        for block, block_rows in rows.items()
    }


def _hippo_receipt_sha256(receipt: Mapping[str, Any]) -> str:
    value = receipt.get("receipt_sha256")
    if isinstance(value, str) and len(value) == 64:
        return _require_sha256(value, "official Hippo receipt")
    # Synthetic test doubles use a content-free receipt with no official schema.
    return stable_hash(dict(receipt))


@dataclass(frozen=True)
class DefaultFormalCore:
    """Thin adapter over the committed local runner/evaluator primitives."""

    def _runtime_components(self, runtime: object) -> tuple[Any, Any, Any, Any]:
        minilm = getattr(runtime, "minilm", None)
        ner = getattr(runtime, "ner", None)
        nli = getattr(runtime, "nli", None)
        hippo = getattr(runtime, "hippo", None)
        if any(component is None for component in (minilm, ner, nli, hippo)):
            raise FeverousFormalControllerError("semantic runtime is incomplete")
        return minilm, ner, nli, hippo

    def prepare(
        self, *, corpus_view: Mapping[str, Any], runtime: object
    ) -> PreparedExecution:
        minilm, ner, _nli, hippo = self._runtime_components(runtime)
        units = runner.corpus_view_to_semantic_units(corpus_view)
        # The two corpus-wide consumers are independent and share exact text.
        with ThreadPoolExecutor(max_workers=2) as pool:
            semantic_future = pool.submit(
                semantic.prepare_semantic_corpus,
                corpus_units=units,
                minilm_backend=minilm,
                ner_backend=ner,
            )
            hippo_future = pool.submit(
                hippo.build,
                tuple(
                    {"idx": unit.corpus_ordinal, "text": unit.linearized_text}
                    for unit in units
                ),
            )
            prepared = semantic_future.result()
            hippo_receipt = hippo_future.result()
        if not isinstance(hippo_receipt, Mapping):
            raise FeverousFormalControllerError("Hippo build receipt drifted")
        return PreparedExecution(prepared, dict(hippo_receipt))

    def execute_formation(
        self,
        *,
        A_form_view: Mapping[str, Any],
        F_search_view: Mapping[str, Any],
        prepared: object,
        runtime: object,
    ) -> FormationStages:
        if not isinstance(prepared, PreparedExecution):
            raise FeverousFormalControllerError("prepared execution type drifted")
        minilm, ner, nli, hippo = self._runtime_components(runtime)
        a_claims = runner.claims_from_block_view(A_form_view, block="A_form")
        f_claims = runner.claims_from_block_view(F_search_view, block="F_search")
        queries, schedule = formation_query_schedule(a_claims, f_claims)
        # Both physical jobs are submitted before either is joined: one local
        # 144-item pool and one public 144-query official retrieval invocation.
        with ThreadPoolExecutor(max_workers=2) as pool:
            local_future = pool.submit(
                runner.execute_formation_blocks,
                A_form_claims=a_claims,
                F_search_claims=f_claims,
                prepared_corpus=prepared.semantic_corpus,
                minilm_backend=minilm,
                ner_backend=ner,
                nli_backend=nli,
                worker_count=runner.LOCAL_ITEM_WORKERS,
            )
            hippo_future = pool.submit(
                hippo.retrieve, block="A_form", queries=queries
            )
            local = local_future.result()
            hippo_batch = hippo_future.result()
        if not isinstance(local, runner.FormationExecution):
            raise FeverousFormalControllerError("combined local formation drifted")
        if not isinstance(hippo_batch, RetrievalBatch):
            raise FeverousFormalControllerError("combined Hippo return type drifted")
        projected = split_formation_hippo_result(hippo_batch, schedule=schedule)
        hippo_sha = _hippo_receipt_sha256(hippo_batch.receipt)
        preparation_receipt = getattr(prepared.semantic_corpus, "receipt", None)
        if not isinstance(preparation_receipt, Mapping):
            raise FeverousFormalControllerError(
                "semantic preparation receipt is unavailable"
            )
        preparation_sha = _require_sha256(
            preparation_receipt.get("preparation_receipt_sha256"),
            "semantic preparation receipt",
        )
        hippo_build_sha = _hippo_receipt_sha256(prepared.hippo_build_receipt)
        schedule_payload = [[block, ordinal] for block, ordinal in schedule]
        formation_body = {
            "schema": f"{VERSION}_formation_parallel_receipt",
            "version": VERSION,
            "query_count": len(queries),
            "local_combined_receipt_sha256": local.receipt[
                "formation_execution_receipt_sha256"
            ],
            "official_combined_receipt_sha256": hippo_sha,
            "semantic_preparation_receipt": dict(preparation_receipt),
            "semantic_preparation_receipt_sha256": preparation_sha,
            "official_hipporag_build_receipt": dict(
                prepared.hippo_build_receipt
            ),
            "official_hipporag_build_receipt_sha256": hippo_build_sha,
            "interleaved_schedule_sha256": stable_hash(schedule_payload),
            "single_local_pool_maximum_workers": runner.LOCAL_ITEM_WORKERS,
            "single_official_gateway_retrieve_call": True,
            "both_physical_jobs_submitted_before_join": True,
            "labels_family_gold_or_utility_accessed": False,
            "online_evaluator_calls": 0,
        }
        formation_receipt = _self_hashed(
            formation_body, field="formation_parallel_receipt_sha256"
        )
        stages: dict[str, LabelFreeBlockStage] = {}
        for block, local_block in (
            ("A_form", local.A_form),
            ("F_search", local.F_search),
        ):
            positions = [
                position
                for position, coordinate in enumerate(schedule)
                if coordinate[0] == block
            ]
            projection_body = {
                "schema": f"{VERSION}_official_formation_projection_receipt",
                "version": VERSION,
                "block": block,
                "item_count": BLOCK_COUNTS[block],
                "combined_official_receipt": dict(hippo_batch.receipt),
                "combined_official_receipt_sha256": hippo_sha,
                "interleaved_schedule_sha256": stable_hash(schedule_payload),
                "projection_positions": positions,
                "projected_top5_sha256": stable_hash(
                    [list(row) for row in projected[block]]
                ),
                "single_official_gateway_retrieve_call": True,
                "query_or_corpus_text_persisted": False,
            }
            projection = _self_hashed(
                projection_body, field="projection_receipt_sha256"
            )
            stages[block] = LabelFreeBlockStage(
                block=block,
                local=local_block,
                hippo_top5=projected[block],
                hippo_receipt=projection,
                execution_receipt=formation_receipt,
            )
        return FormationStages(
            A_form=stages["A_form"],
            F_search=stages["F_search"],
            receipt=formation_receipt,
        )

    def formation_blocks(self, formation: object) -> Mapping[str, object]:
        if not isinstance(formation, FormationStages):
            raise FeverousFormalControllerError("formation stage type drifted")
        return {"A_form": formation.A_form, "F_search": formation.F_search}

    def execute_anchor(
        self,
        *,
        block: str,
        view: Mapping[str, Any],
        prepared: object,
        runtime: object,
    ) -> LabelFreeBlockStage:
        if block not in ANCHOR_BLOCKS or not isinstance(prepared, PreparedExecution):
            raise FeverousFormalControllerError("anchor execution identity drifted")
        minilm, ner, nli, hippo = self._runtime_components(runtime)
        claims = runner.claims_from_block_view(view, block=block)
        with ThreadPoolExecutor(max_workers=2) as pool:
            local_future = pool.submit(
                runner.execute_local_block,
                block=block,
                claims=claims,
                prepared_corpus=prepared.semantic_corpus,
                minilm_backend=minilm,
                ner_backend=ner,
                nli_backend=nli,
                worker_count=runner.LOCAL_ITEM_WORKERS,
            )
            hippo_future = pool.submit(hippo.retrieve, block=block, queries=claims)
            local = local_future.result()
            hippo_batch = hippo_future.result()
        if not isinstance(local, runner.BlockExecution) or not isinstance(
            hippo_batch, RetrievalBatch
        ):
            raise FeverousFormalControllerError("anchor arm return type drifted")
        hippo_sha = _hippo_receipt_sha256(hippo_batch.receipt)
        body = {
            "schema": f"{VERSION}_anchor_parallel_receipt",
            "version": VERSION,
            "block": block,
            "item_count": len(claims),
            "local_block_receipt_sha256": local.receipt["block_receipt_sha256"],
            "official_retrieval_receipt": dict(hippo_batch.receipt),
            "official_retrieval_receipt_sha256": hippo_sha,
            "single_official_gateway_retrieve_call": True,
            "logical_RAW_Hippo_Agent_work_units": 3 * len(claims),
            "both_physical_jobs_submitted_before_join": True,
            "labels_family_gold_or_utility_accessed": False,
            "online_evaluator_calls": 0,
        }
        receipt = _self_hashed(body, field="anchor_parallel_receipt_sha256")
        return LabelFreeBlockStage(
            block=block,
            local=local,
            hippo_top5=tuple(hippo_batch.indices),
            hippo_receipt={
                "official_receipt": dict(hippo_batch.receipt),
                "official_receipt_sha256": hippo_sha,
                "single_official_gateway_retrieve_call": True,
            },
            execution_receipt=receipt,
        )

    def archive_payload(self, stage: object, *, block: str) -> Mapping[str, Any]:
        if (
            not isinstance(stage, LabelFreeBlockStage)
            or stage.block != block
            or stage.local.block != block
            or len(stage.local.items) != BLOCK_COUNTS[block]
            or len(stage.hippo_top5) != BLOCK_COUNTS[block]
        ):
            raise FeverousFormalControllerError("label-free stage shape drifted")
        item_payloads = [item.public_payload() for item in stage.local.items]
        if any(
            len(item.get("complete_action_trace_receipts", []))
            != len(runner.RECIPE_IDS)
            or len(item.get("feature_traces", [])) != len(runner.RECIPE_IDS)
            for item in item_payloads
        ):
            raise FeverousFormalControllerError("complete trace archive drifted")
        return {
            "block": block,
            "item_count": len(item_payloads),
            "local_block_receipt": dict(stage.local.receipt),
            "feature_receipt": dict(stage.local.feature_receipt),
            "execution_receipt": dict(stage.execution_receipt),
            "official_hipporag_receipt": dict(stage.hippo_receipt),
            "official_hipporag_top5": [list(row) for row in stage.hippo_top5],
            "items": item_payloads,
            "complete_content_free_action_traces_persisted": True,
            "complete_content_free_feature_traces_persisted": True,
            "raw_claim_corpus_gold_label_family_or_verdict_persisted": False,
        }

    def fit_e2(
        self,
        *,
        A_form_stage: object,
        labels: Mapping[str, Any],
        fold_secret: bytes,
    ) -> Mapping[str, Any]:
        if not isinstance(A_form_stage, LabelFreeBlockStage):
            raise FeverousFormalControllerError("A_form stage type drifted")
        utilities = runner.a_form_utility_matrix(
            block=A_form_stage.local, labels=labels
        )
        _model, receipt = evaluator.fit_e2_a_form(
            traces=A_form_stage.local.recipe_traces,
            utilities=utilities,
            fold_hmac_secret=fold_secret,
            feature_receipt=A_form_stage.local.feature_receipt,
        )
        evaluator.verify_fit_receipt(
            receipt,
            traces=A_form_stage.local.recipe_traces,
            utilities=utilities,
            fold_hmac_secret=fold_secret,
            feature_receipt=A_form_stage.local.feature_receipt,
        )
        return receipt

    def freeze_f_policies(
        self,
        *,
        F_search_stage: object,
        A_form_stage: object,
        fit_receipt: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if not isinstance(F_search_stage, LabelFreeBlockStage) or not isinstance(
            A_form_stage, LabelFreeBlockStage
        ):
            raise FeverousFormalControllerError("formation stage type drifted")
        receipt = evaluator.freeze_f_policies(
            traces=F_search_stage.local.recipe_traces,
            feature_receipt=F_search_stage.local.feature_receipt,
            fit_receipt=fit_receipt,
            expected_a_form_feature_receipt_sha256=str(
                A_form_stage.local.feature_receipt["feature_receipt_sha256"]
            ),
            expected_fit_receipt_sha256=str(fit_receipt["fit_receipt_sha256"]),
        )
        evaluator.verify_policy_receipt(
            receipt,
            traces=F_search_stage.local.recipe_traces,
            feature_receipt=F_search_stage.local.feature_receipt,
            fit_receipt=fit_receipt,
            expected_a_form_feature_receipt_sha256=str(
                A_form_stage.local.feature_receipt["feature_receipt_sha256"]
            ),
            expected_fit_receipt_sha256=str(fit_receipt["fit_receipt_sha256"]),
        )
        return receipt

    def score_anchor(
        self,
        *,
        stage: object,
        labels: Mapping[str, Any],
        policy_receipt: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if not isinstance(stage, LabelFreeBlockStage):
            raise FeverousFormalControllerError("anchor stage type drifted")
        e0_recipe = policy_receipt.get("E0_selected_recipe_id")
        e2_recipe = policy_receipt.get("E2_selected_recipe_id")
        identifiable = policy_receipt.get(
            "A_hold_evaluator_comparison_identifiable"
        )
        if (
            e0_recipe not in runner.RECIPE_IDS
            or e2_recipe not in runner.RECIPE_IDS
            or type(identifiable) is not bool
        ):
            raise FeverousFormalControllerError(
                "verified F policy fields drifted before anchor scoring"
            )
        return runner.score_anchor_block(
            block=stage.local,
            labels=labels,
            hippo_top5=stage.hippo_top5,
            e0_recipe_id=str(e0_recipe),
            e2_recipe_id=str(e2_recipe),
            evaluator_comparison_identifiable=identifiable,
        )


@dataclass(frozen=True)
class ModuleAcquisitionBoundary:
    """The only production object allowed to open formal private packs."""

    def verify_prerequisites(self, *, project: Path) -> PrerequisiteBinding:
        freeze = implementation_freeze.verify_committed_implementation_freeze(project)
        # Envelope verification checks public commitments and private metadata
        # only.  In particular it never hashes or decodes future M/view/label
        # bytes; requested-role loaders perform that work only after authority.
        acquisition = formal_acquisition.verify_acquisition_envelope(project)
        freeze_sha = _require_sha256(
            freeze.get("implementation_freeze_sha256"), "implementation freeze"
        )
        acquisition_sha = _require_sha256(
            acquisition.get("acquisition_receipt_sha256"), "acquisition receipt"
        )
        if (
            acquisition.get("implementation_freeze_sha256") != freeze_sha
            or acquisition.get(
                "identity_full_compile_equivalence_qualification_sha256"
            )
            != freeze.get("identity_compiler_qualification_sha256")
            or acquisition.get("source_epoch_rollover_sha256")
            != freeze.get("source_epoch_rollover_sha256")
            or acquisition.get("train_loader_qualification_sha256")
            != freeze.get("train_loader_qualification_sha256")
        ):
            raise FeverousFormalControllerError(
                "acquisition is outside the implementation freeze"
            )
        return PrerequisiteBinding(
            implementation_freeze_sha256=freeze_sha,
            acquisition_receipt_sha256=acquisition_sha,
            runtime_preflight_sha256=_require_sha256(
                freeze.get("runtime_preflight_sha256"), "runtime preflight"
            ),
            implementation_git_commit=str(freeze.get("implementation_git_commit")),
        ).validate()

    def assert_stable(
        self, *, project: Path, prerequisites: PrerequisiteBinding
    ) -> None:
        observed = self.verify_prerequisites(project=project)
        if observed != prerequisites:
            raise FeverousFormalControllerError("formal prerequisite drifted")

    def preflight_outputs(
        self,
        *,
        project: Path,
        runtime_config: object,
        output_paths: LifecycleOutputPaths,
    ) -> None:
        if not isinstance(runtime_config, local_runtime.FormalRuntimeConfig):
            raise FeverousFormalControllerError("formal runtime config drifted")
        controller_root = project / output_paths.root_relative
        owned_runtime_roots = (
            runtime_config.hippo_stage_root,
            runtime_config.hippo_work_root,
            runtime_config.ner_pycache_root,
        )
        for path in (controller_root, *owned_runtime_roots):
            _assert_under_project(project=project, path=path)
            if os.path.lexists(path):
                raise FeverousFormalControllerError(
                    "formal controller or runtime output already exists"
                )

    def load_corpus_view(self, *, project: Path) -> Mapping[str, Any]:
        return formal_acquisition.load_corpus_view(project)

    def load_block_view(
        self, *, project: Path, expected_block: str
    ) -> Mapping[str, Any]:
        return formal_acquisition.load_block_view(project, block=expected_block)

    def load_block_labels(
        self, *, project: Path, expected_block: str
    ) -> Mapping[str, Any]:
        if expected_block not in {"A_form", "A_hold", "M_search"}:
            raise FeverousFormalControllerError("label capability is invalid")
        return formal_acquisition.load_block_labels(project, block=expected_block)

    def load_private_secret(self, *, project: Path) -> bytes:
        return formal_acquisition.load_private_secret(project)


@dataclass(frozen=True)
class DefaultRuntimeBoundary:
    factory: local_runtime.FeverousLocalRuntimeFactory = (
        local_runtime.DEFAULT_RUNTIME_FACTORY
    )

    def preflight(self, runtime_config: object) -> Mapping[str, Any]:
        if not isinstance(runtime_config, local_runtime.FormalRuntimeConfig):
            raise FeverousFormalControllerError("formal runtime config drifted")
        return self.factory.preflight(runtime_config)

    def open(self, runtime_config: object) -> AbstractContextManager[object]:
        if not isinstance(runtime_config, local_runtime.FormalRuntimeConfig):
            raise FeverousFormalControllerError("formal runtime config drifted")
        return self.factory.create_semantic_runtime_context(runtime_config)

    def postflight(self, runtime: object) -> Mapping[str, Any]:
        if not isinstance(runtime, local_runtime.SemanticRuntimeBundle):
            raise FeverousFormalControllerError("runtime postflight type drifted")
        receipt = runtime.receipt()
        if (
            receipt.get("schema") != local_runtime.BUNDLE_SCHEMA
            or receipt.get("version") != local_runtime.VERSION
            or receipt.get("network_calls") != 0
            or receipt.get("ner_process_count") != local_runtime.NER_PROCESS_COUNT
            or receipt.get("nli_worker_count")
            != local_runtime.NLI_WORKER_COUNT
            or not isinstance(receipt.get("minilm_binding"), Mapping)
            or not isinstance(receipt.get("ner_binding"), Mapping)
            or not isinstance(receipt.get("nli_binding"), Mapping)
        ):
            raise FeverousFormalControllerError("runtime postflight receipt drifted")
        return receipt


def _assert_label_free_archive_payload(payload: Mapping[str, Any]) -> None:
    """Reject direct private fields while permitting explicit negative receipts."""

    forbidden_exact_keys = {
        "claim",
        "claim_text",
        "corpus",
        "corpus_view",
        "documents",
        "gold",
        "gold_unit_indices",
        "label",
        "labels",
        "family",
        "linearized_text",
        "query",
        "queries",
        "target_text",
        "text",
        "units",
        "verdict",
        "evidence",
    }

    def walk(value: object) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if not isinstance(key, str):
                    raise FeverousFormalControllerError(
                        "archive contains a non-string key"
                    )
                if key in forbidden_exact_keys:
                    raise FeverousFormalControllerError(
                        "archive contains a raw private field"
                    )
                walk(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                walk(child)

    walk(payload)


def _verify_archive_envelope(
    payload: Mapping[str, Any], *, expected_block: str | None = None
) -> str:
    expected_keys = {
        "schema",
        "version",
        "block",
        "marker_sha256",
        "stage",
        "raw_claim_corpus_gold_or_labels_persisted",
        "online_evaluator_calls",
        "archive_sha256",
    }
    stage = payload.get("stage")
    if (
        set(payload) != expected_keys
        or payload.get("schema") != ARCHIVE_SCHEMA
        or payload.get("version") != VERSION
        or payload.get("block") not in BLOCK_ORDER
        or (
            expected_block is not None
            and payload.get("block") != expected_block
        )
        or not isinstance(stage, Mapping)
        or payload.get("raw_claim_corpus_gold_or_labels_persisted") is not False
        or payload.get("online_evaluator_calls") != 0
    ):
        raise FeverousFormalControllerError("archive envelope drifted")
    _require_sha256(payload.get("marker_sha256"), "archive marker")
    _assert_label_free_archive_payload(stage)
    return verify_self_hash(payload, field="archive_sha256")


def _verify_seal_envelope(
    payload: Mapping[str, Any], *, expected_block: str
) -> str:
    expected_keys = {
        "schema",
        "version",
        "block",
        "archive_relative_path",
        "archive_sha256",
        "archive_file_sha256",
        "labels_opened_before_seal",
        "seal_replay_or_replacement_authorized",
        "seal_sha256",
    }
    if (
        set(payload) != expected_keys
        or payload.get("schema") != SEAL_SCHEMA
        or payload.get("version") != VERSION
        or payload.get("block") != expected_block
        or payload.get("labels_opened_before_seal") is not False
        or payload.get("seal_replay_or_replacement_authorized") is not False
    ):
        raise FeverousFormalControllerError("archive seal envelope drifted")
    _require_sha256(payload.get("archive_sha256"), "sealed archive")
    _require_sha256(payload.get("archive_file_sha256"), "sealed archive file")
    return verify_self_hash(payload, field="seal_sha256")


def persist_label_free_archive(
    *,
    project: Path,
    output_paths: LifecycleOutputPaths,
    block: str,
    marker_sha256: str,
    stage_payload: Mapping[str, Any],
) -> LifecycleArtifact:
    _require_block(block)
    _assert_label_free_archive_payload(stage_payload)
    body = {
        "schema": ARCHIVE_SCHEMA,
        "version": VERSION,
        "block": block,
        "marker_sha256": _require_sha256(marker_sha256, "marker"),
        "stage": dict(stage_payload),
        "raw_claim_corpus_gold_or_labels_persisted": False,
        "online_evaluator_calls": 0,
    }
    archive = _self_hashed(body, field="archive_sha256")
    path = project / output_paths.archive_relative(block)
    file_sha = write_json_exclusive(path, archive, mode=0o600)
    return LifecycleArtifact(
        kind="label_free_archive",
        block=block,
        path=path,
        receipt_sha256=str(archive["archive_sha256"]),
        file_sha256=file_sha,
        payload=archive,
    )


def seal_label_free_archive(
    *,
    project: Path,
    output_paths: LifecycleOutputPaths,
    archive: LifecycleArtifact,
) -> LifecycleArtifact:
    archive.validate(kind="label_free_archive", block=archive.block)
    if archive.block is None:
        raise FeverousFormalControllerError("archive block is absent")
    loaded, file_sha = _load_canonical_json(archive.path)
    declared = _verify_archive_envelope(
        loaded, expected_block=archive.block
    )
    if (
        declared != archive.receipt_sha256
        or file_sha != archive.file_sha256
        or loaded != dict(archive.payload)
    ):
        raise FeverousFormalControllerError("archive changed before seal")
    body = {
        "schema": SEAL_SCHEMA,
        "version": VERSION,
        "block": archive.block,
        "archive_relative_path": output_paths.archive_relative(
            archive.block
        ).as_posix(),
        "archive_sha256": declared,
        "archive_file_sha256": file_sha,
        "labels_opened_before_seal": False,
        "seal_replay_or_replacement_authorized": False,
    }
    seal = _self_hashed(body, field="seal_sha256")
    path = project / output_paths.seal_relative(archive.block)
    seal_file_sha = write_json_exclusive(path, seal, mode=0o644)
    return LifecycleArtifact(
        kind="label_free_seal",
        block=archive.block,
        path=path,
        receipt_sha256=str(seal["seal_sha256"]),
        file_sha256=seal_file_sha,
        payload=seal,
    )


def verify_archive_and_seal(
    *, archive: LifecycleArtifact, seal: LifecycleArtifact
) -> None:
    archive.validate(kind="label_free_archive", block=archive.block)
    seal.validate(kind="label_free_seal", block=archive.block)
    loaded_archive, archive_file_sha = _load_canonical_json(archive.path)
    loaded_seal, seal_file_sha = _load_canonical_json(seal.path)
    if archive.block is None:
        raise FeverousFormalControllerError("archive block is absent")
    archive_sha = _verify_archive_envelope(
        loaded_archive, expected_block=archive.block
    )
    seal_sha = _verify_seal_envelope(
        loaded_seal, expected_block=archive.block
    )
    if (
        archive_sha != archive.receipt_sha256
        or archive_file_sha != archive.file_sha256
        or loaded_archive != dict(archive.payload)
        or seal_sha != seal.receipt_sha256
        or seal_file_sha != seal.file_sha256
        or loaded_seal != dict(seal.payload)
        or loaded_seal.get("archive_sha256") != archive_sha
        or loaded_seal.get("archive_file_sha256") != archive_file_sha
    ):
        raise FeverousFormalControllerError("archive/seal binding drifted")


def _persist_receipt(
    *,
    project: Path,
    output_paths: LifecycleOutputPaths,
    name: str,
    kind: str,
    block: str,
    marker_sha256: str,
    receipt: Mapping[str, Any],
) -> LifecycleArtifact:
    body = {
        "schema": f"{VERSION}_{kind}_artifact",
        "version": VERSION,
        "kind": kind,
        "block": block,
        "marker_sha256": _require_sha256(marker_sha256, "marker"),
        "receipt": dict(receipt),
        "raw_claim_corpus_gold_or_label_rows_persisted": False,
        "online_evaluator_calls": 0,
    }
    payload = _self_hashed(body, field="artifact_sha256")
    path = project / output_paths.receipt_relative(name)
    file_sha = write_json_exclusive(path, payload, mode=0o600)
    return LifecycleArtifact(
        kind=kind,
        block=block,
        path=path,
        receipt_sha256=str(payload["artifact_sha256"]),
        file_sha256=file_sha,
        payload=payload,
    )


def _verify_receipt_artifact(artifact: LifecycleArtifact) -> None:
    artifact.validate(kind=artifact.kind, block=artifact.block)
    loaded, file_sha = _load_canonical_json(artifact.path)
    if (
        set(loaded)
        != {
            "schema",
            "version",
            "kind",
            "block",
            "marker_sha256",
            "receipt",
            "raw_claim_corpus_gold_or_label_rows_persisted",
            "online_evaluator_calls",
            "artifact_sha256",
        }
        or loaded.get("schema") != f"{VERSION}_{artifact.kind}_artifact"
        or loaded.get("version") != VERSION
        or not isinstance(loaded.get("receipt"), Mapping)
        or loaded.get("raw_claim_corpus_gold_or_label_rows_persisted") is not False
        or loaded.get("online_evaluator_calls") != 0
    ):
        raise FeverousFormalControllerError("receipt artifact envelope drifted")
    _require_sha256(loaded.get("marker_sha256"), "artifact marker")
    declared = verify_self_hash(loaded, field="artifact_sha256")
    if (
        declared != artifact.receipt_sha256
        or file_sha != artifact.file_sha256
        or loaded != dict(artifact.payload)
        or loaded.get("kind") != artifact.kind
        or loaded.get("block") != artifact.block
    ):
        raise FeverousFormalControllerError("receipt artifact binding drifted")


def _one_shot_marker(
    *, prerequisites: PrerequisiteBinding, runtime_preflight_sha256: str
) -> dict[str, Any]:
    prerequisites.validate()
    if runtime_preflight_sha256 != prerequisites.runtime_preflight_sha256:
        raise FeverousFormalControllerError("runtime preflight is outside the freeze")
    return _self_hashed(
        {
            "schema": MARKER_SCHEMA,
            "version": VERSION,
            "phase": "A_form_F_search_A_hold_then_promotion_gated_M_search",
            "implementation_freeze_sha256": (
                prerequisites.implementation_freeze_sha256
            ),
            "acquisition_receipt_sha256": prerequisites.acquisition_receipt_sha256,
            "runtime_preflight_sha256": runtime_preflight_sha256,
            "implementation_git_commit": prerequisites.implementation_git_commit,
            "retry_replay_resample_or_replacement_authorized": False,
            "online_evaluator_calls": 0,
        },
        field="marker_sha256",
    )


def _verify_marker_file(
    *, path: Path, expected: Mapping[str, Any], expected_file_sha256: str
) -> None:
    loaded, file_sha = _load_canonical_json(path)
    expected_keys = {
        "schema",
        "version",
        "phase",
        "implementation_freeze_sha256",
        "acquisition_receipt_sha256",
        "runtime_preflight_sha256",
        "implementation_git_commit",
        "retry_replay_resample_or_replacement_authorized",
        "online_evaluator_calls",
        "marker_sha256",
    }
    if (
        set(loaded) != expected_keys
        or loaded.get("schema") != MARKER_SCHEMA
        or loaded.get("version") != VERSION
        or loaded.get("retry_replay_resample_or_replacement_authorized")
        is not False
        or loaded.get("online_evaluator_calls") != 0
        or verify_self_hash(loaded, field="marker_sha256")
        != expected.get("marker_sha256")
        or loaded != dict(expected)
        or file_sha != _require_sha256(
            expected_file_sha256, "marker file"
        )
    ):
        raise FeverousFormalControllerError("one-shot marker binding drifted")


def _write_terminal_failure(
    *, path: Path, marker_sha256: str, failure_stage: str, exc: BaseException
) -> None:
    type_name = f"{type(exc).__module__}.{type(exc).__qualname__}"
    message = str(exc).encode("utf-8", errors="replace")
    payload = _self_hashed(
        {
            "schema": FAILURE_SCHEMA,
            "version": VERSION,
            "status": "terminal_cohort_burned_no_retry_or_resample",
            "marker_sha256": _require_sha256(marker_sha256, "marker"),
            "failure_stage": failure_stage,
            "exception_type_sha256": hashlib.sha256(
                type_name.encode("utf-8")
            ).hexdigest(),
            "exception_message_sha256": hashlib.sha256(message).hexdigest(),
            "exception_type_or_message_plaintext_persisted": False,
            "claim_corpus_gold_label_or_outcome_content_persisted": False,
            "retry_replay_resample_or_replacement_authorized": False,
        },
        field="failure_sha256",
    )
    try:
        write_json_exclusive(path, payload, mode=0o644)
    except BaseException:
        pass


def _promotion_authorization(
    *,
    marker_sha256: str,
    policy_artifact: LifecycleArtifact,
    score_artifact: LifecycleArtifact,
    score_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    if score_receipt.get("evaluator_promoted") is not True:
        raise FeverousFormalControllerError("M_search authorization lacks promotion")
    body = {
        "schema": f"{VERSION}_promotion_authorization",
        "version": VERSION,
        "block": "A_hold",
        "marker_sha256": _require_sha256(marker_sha256, "marker"),
        "policy_artifact_sha256": policy_artifact.receipt_sha256,
        "A_hold_score_artifact_sha256": score_artifact.receipt_sha256,
        "A_hold_score_receipt_sha256": _require_sha256(
            score_receipt.get("score_receipt_sha256"), "A_hold score"
        ),
        "M_search_view_and_execution_authorized": True,
        "invalidated_evaluator_dependent_epoch": "E0",
        "retained_source_actions_RAW_Hippo_and_independent_receipts": True,
        "epoch_rollback_from_M_authorized": False,
    }
    return _self_hashed(body, field="promotion_authorization_sha256")


def _verify_anchor_score_receipt(
    receipt: Mapping[str, Any], *, block: str
) -> str:
    if block not in ANCHOR_BLOCKS or not isinstance(receipt, Mapping):
        raise FeverousFormalControllerError("anchor score identity drifted")
    body = dict(receipt)
    declared = _require_sha256(
        body.pop("score_receipt_sha256", None), f"{block} score receipt"
    )
    if stable_hash(body) != declared or receipt.get("block") != block:
        raise FeverousFormalControllerError("anchor score self-hash drifted")
    if block == "A_hold":
        if (
            type(receipt.get("A_hold_real_domain_primary_passed")) is not bool
            or type(receipt.get("evaluator_promoted")) is not bool
            or receipt.get("M_L5_passed") is not None
        ):
            raise FeverousFormalControllerError("A_hold score flags drifted")
    elif (
        receipt.get("A_hold_real_domain_primary_passed") is not None
        or receipt.get("evaluator_promoted") is not None
        or type(receipt.get("M_L5_passed")) is not bool
    ):
        raise FeverousFormalControllerError("M_search score flags drifted")
    return declared


def _verify_policy_selection_fields(receipt: Mapping[str, Any]) -> None:
    if not isinstance(receipt, Mapping):
        raise FeverousFormalControllerError("F policy receipt is not an object")
    _require_sha256(receipt.get("policy_receipt_sha256"), "F policy receipt")
    if (
        receipt.get("E0_selected_recipe_id") not in runner.RECIPE_IDS
        or receipt.get("E2_selected_recipe_id") not in runner.RECIPE_IDS
        or type(receipt.get("A_hold_evaluator_comparison_identifiable")) is not bool
    ):
        raise FeverousFormalControllerError("F policy selection fields drifted")


def _run_lifecycle_core(
    *,
    project: Path,
    runtime_config: object,
    acquisition_boundary: AcquisitionBoundary,
    runtime_boundary: RuntimeBoundary,
    core: FormalCore,
    output_paths: LifecycleOutputPaths,
) -> dict[str, Any]:
    root = _canonical_project(project)
    # These are the only operations authorized before the one-shot marker.
    prerequisites = acquisition_boundary.verify_prerequisites(project=root).validate()
    runtime_preflight = runtime_boundary.preflight(runtime_config)
    if not isinstance(runtime_preflight, Mapping):
        raise FeverousFormalControllerError("runtime preflight returned no receipt")
    runtime_preflight_sha = stable_hash(dict(runtime_preflight))
    if runtime_preflight_sha != prerequisites.runtime_preflight_sha256:
        raise FeverousFormalControllerError("runtime preflight binding drifted")
    acquisition_boundary.preflight_outputs(
        project=root,
        runtime_config=runtime_config,
        output_paths=output_paths,
    )
    acquisition_boundary.assert_stable(project=root, prerequisites=prerequisites)

    output_root = root / output_paths.root_relative
    _ensure_private_directory(project=root, path=output_root)
    marker_payload = _one_shot_marker(
        prerequisites=prerequisites,
        runtime_preflight_sha256=runtime_preflight_sha,
    )
    marker_path = root / output_paths.marker_relative
    marker_existed = os.path.lexists(marker_path)
    marker_sha = str(marker_payload["marker_sha256"])
    try:
        marker_file_sha = write_json_exclusive(
            marker_path, marker_payload, mode=0o600
        )
        _verify_marker_file(
            path=marker_path,
            expected=marker_payload,
            expected_file_sha256=marker_file_sha,
        )
    except BaseException as exc:
        if not marker_existed and os.path.lexists(marker_path):
            _write_terminal_failure(
                path=root / output_paths.failure_relative,
                marker_sha256=marker_sha,
                failure_stage="exclusive_marker",
                exc=exc,
            )
        raise

    failure_stage = "runtime_initialization"
    artifacts: dict[str, LifecycleArtifact] = {}
    terminal_status = ""
    terminal_body: dict[str, Any] = {}
    executed_blocks: list[str] = []

    def record(name: str, artifact: LifecycleArtifact) -> None:
        if name in artifacts:
            raise FeverousFormalControllerError("artifact name was reused")
        artifacts[name] = artifact

    def archive_and_seal(block: str, stage: object) -> tuple[LifecycleArtifact, LifecycleArtifact]:
        nonlocal failure_stage
        failure_stage = f"{block}_content_free_archive"
        stage_payload = core.archive_payload(stage, block=block)
        archive = persist_label_free_archive(
            project=root,
            output_paths=output_paths,
            block=block,
            marker_sha256=marker_sha,
            stage_payload=stage_payload,
        )
        record(f"{block}_archive", archive)
        failure_stage = f"{block}_archive_seal"
        seal = seal_label_free_archive(
            project=root, output_paths=output_paths, archive=archive
        )
        record(f"{block}_seal", seal)
        verify_archive_and_seal(archive=archive, seal=seal)
        return archive, seal

    try:
        if isinstance(runtime_config, local_runtime.FormalRuntimeConfig):
            _ensure_private_directory(
                project=root, path=runtime_config.hippo_work_root
            )
        with runtime_boundary.open(runtime_config) as runtime:
            failure_stage = "corpus_view_and_shared_runtime_preparation"
            corpus_view = acquisition_boundary.load_corpus_view(project=root)
            prepared = core.prepare(corpus_view=corpus_view, runtime=runtime)
            acquisition_boundary.assert_stable(
                project=root, prerequisites=prerequisites
            )

            # Formation claim views are the only block capabilities opened here.
            failure_stage = "A_form_F_search_claim_view_open"
            a_form_view = acquisition_boundary.load_block_view(
                project=root, expected_block="A_form"
            )
            f_search_view = acquisition_boundary.load_block_view(
                project=root, expected_block="F_search"
            )
            failure_stage = "A_form_F_search_combined_label_free_execution"
            formation = core.execute_formation(
                A_form_view=a_form_view,
                F_search_view=f_search_view,
                prepared=prepared,
                runtime=runtime,
            )
            blocks = core.formation_blocks(formation)
            if set(blocks) != set(FORMATION_BLOCKS):
                raise FeverousFormalControllerError("formation block set drifted")
            executed_blocks.extend(FORMATION_BLOCKS)

            # Both formation archives and both seals must exist before A_form
            # labels or the fold secret are opened.
            formation_artifacts: dict[
                str, tuple[LifecycleArtifact, LifecycleArtifact]
            ] = {}
            for block in FORMATION_BLOCKS:
                formation_artifacts[block] = archive_and_seal(block, blocks[block])
            for archive, seal in formation_artifacts.values():
                verify_archive_and_seal(archive=archive, seal=seal)

            failure_stage = "A_form_late_label_open_and_E2_fit"
            acquisition_boundary.assert_stable(
                project=root, prerequisites=prerequisites
            )
            a_form_labels = acquisition_boundary.load_block_labels(
                project=root, expected_block="A_form"
            )
            fold_secret = acquisition_boundary.load_private_secret(project=root)
            fit_receipt = core.fit_e2(
                A_form_stage=blocks["A_form"],
                labels=a_form_labels,
                fold_secret=fold_secret,
            )
            fit_artifact = _persist_receipt(
                project=root,
                output_paths=output_paths,
                name="A_form_E2_fit",
                kind="E2_fit_freeze",
                block="A_form",
                marker_sha256=marker_sha,
                receipt=fit_receipt,
            )
            record("A_form_E2_fit", fit_artifact)
            _verify_receipt_artifact(fit_artifact)

            failure_stage = "F_search_label_free_policy_freeze"
            policy_receipt = core.freeze_f_policies(
                F_search_stage=blocks["F_search"],
                A_form_stage=blocks["A_form"],
                fit_receipt=fit_receipt,
            )
            _verify_policy_selection_fields(policy_receipt)
            policy_artifact = _persist_receipt(
                project=root,
                output_paths=output_paths,
                name="F_search_policy_freeze",
                kind="F_policy_freeze",
                block="F_search",
                marker_sha256=marker_sha,
                receipt=policy_receipt,
            )
            record("F_search_policy_freeze", policy_artifact)
            _verify_receipt_artifact(policy_artifact)

            # A_hold primary is unconditional, including a same-recipe or
            # behavior-identical evaluator comparison.
            failure_stage = "A_hold_claim_view_and_label_free_execution"
            acquisition_boundary.assert_stable(
                project=root, prerequisites=prerequisites
            )
            a_hold_view = acquisition_boundary.load_block_view(
                project=root, expected_block="A_hold"
            )
            a_hold_stage = core.execute_anchor(
                block="A_hold",
                view=a_hold_view,
                prepared=prepared,
                runtime=runtime,
            )
            executed_blocks.append("A_hold")
            a_hold_archive, a_hold_seal = archive_and_seal(
                "A_hold", a_hold_stage
            )
            verify_archive_and_seal(
                archive=a_hold_archive, seal=a_hold_seal
            )
            failure_stage = "A_hold_late_label_open_primary_and_promotion"
            acquisition_boundary.assert_stable(
                project=root, prerequisites=prerequisites
            )
            a_hold_labels = acquisition_boundary.load_block_labels(
                project=root, expected_block="A_hold"
            )
            a_hold_score = core.score_anchor(
                stage=a_hold_stage,
                labels=a_hold_labels,
                policy_receipt=policy_receipt,
            )
            _verify_anchor_score_receipt(a_hold_score, block="A_hold")
            a_hold_score_artifact = _persist_receipt(
                project=root,
                output_paths=output_paths,
                name="A_hold_score",
                kind="A_hold_score",
                block="A_hold",
                marker_sha256=marker_sha,
                receipt=a_hold_score,
            )
            record("A_hold_score", a_hold_score_artifact)
            _verify_receipt_artifact(a_hold_score_artifact)

            if (
                a_hold_score.get("evaluator_promoted") is True
                and policy_receipt.get(
                    "A_hold_evaluator_comparison_identifiable"
                )
                is not True
            ):
                raise FeverousFormalControllerError(
                    "unidentifiable evaluator comparison cannot promote"
                )

            if a_hold_score.get("evaluator_promoted") is not True:
                terminal_status = "valid_A_hold_nonpromotion_M_search_unopened"
                terminal_body = {
                    "A_hold_real_domain_primary_passed": a_hold_score.get(
                        "A_hold_real_domain_primary_passed"
                    ),
                    "evaluator_promoted": False,
                    "evaluator_comparison_identifiable": policy_receipt.get(
                        "A_hold_evaluator_comparison_identifiable"
                    ),
                    "A_hold_score_receipt": dict(a_hold_score),
                    "M_search_view_opened": False,
                    "M_search_labels_opened": False,
                    "M_search_executed": False,
                }
            else:
                failure_stage = "A_hold_promotion_authorization"
                promotion_receipt = _promotion_authorization(
                    marker_sha256=marker_sha,
                    policy_artifact=policy_artifact,
                    score_artifact=a_hold_score_artifact,
                    score_receipt=a_hold_score,
                )
                promotion_artifact = _persist_receipt(
                    project=root,
                    output_paths=output_paths,
                    name="A_hold_promotion",
                    kind="promotion_authorization",
                    block="A_hold",
                    marker_sha256=marker_sha,
                    receipt=promotion_receipt,
                )
                record("A_hold_promotion", promotion_artifact)
                _verify_receipt_artifact(promotion_artifact)

                # This is the sole M capability boundary in the module.
                failure_stage = "M_search_promoted_view_and_label_free_execution"
                acquisition_boundary.assert_stable(
                    project=root, prerequisites=prerequisites
                )
                m_view = acquisition_boundary.load_block_view(
                    project=root, expected_block="M_search"
                )
                m_stage = core.execute_anchor(
                    block="M_search",
                    view=m_view,
                    prepared=prepared,
                    runtime=runtime,
                )
                executed_blocks.append("M_search")
                m_archive, m_seal = archive_and_seal("M_search", m_stage)
                verify_archive_and_seal(archive=m_archive, seal=m_seal)
                failure_stage = "M_search_late_label_open_and_L5"
                acquisition_boundary.assert_stable(
                    project=root, prerequisites=prerequisites
                )
                m_labels = acquisition_boundary.load_block_labels(
                    project=root, expected_block="M_search"
                )
                m_score = core.score_anchor(
                    stage=m_stage,
                    labels=m_labels,
                    policy_receipt=policy_receipt,
                )
                _verify_anchor_score_receipt(m_score, block="M_search")
                m_score_artifact = _persist_receipt(
                    project=root,
                    output_paths=output_paths,
                    name="M_search_score",
                    kind="M_search_score",
                    block="M_search",
                    marker_sha256=marker_sha,
                    receipt=m_score,
                )
                record("M_search_score", m_score_artifact)
                _verify_receipt_artifact(m_score_artifact)
                terminal_status = "formal_M_search_complete"
                terminal_body = {
                    "A_hold_real_domain_primary_passed": a_hold_score.get(
                        "A_hold_real_domain_primary_passed"
                    ),
                    "evaluator_promoted": True,
                    "A_hold_score_receipt": dict(a_hold_score),
                    "M_search_score_receipt": dict(m_score),
                    "M_L5_passed": m_score.get("M_L5_passed"),
                    "M_search_view_opened": True,
                    "M_search_labels_opened": True,
                    "M_search_executed": True,
                }

            # Capture the live, content-free backend receipt before context
            # shutdown makes it unavailable.  Counts bind the exact terminal
            # branch and prevent a partial stage set from looking complete.
            failure_stage = "runtime_postflight_and_terminal_counts"
            runtime_bundle_receipt = runtime_boundary.postflight(runtime)
            if not isinstance(runtime_bundle_receipt, Mapping):
                raise FeverousFormalControllerError(
                    "runtime postflight returned no receipt"
                )
            expected_blocks = (
                ["A_form", "F_search", "A_hold", "M_search"]
                if terminal_body.get("M_search_executed") is True
                else ["A_form", "F_search", "A_hold"]
            )
            if executed_blocks != expected_blocks:
                raise FeverousFormalControllerError(
                    "terminal executed-block vector drifted"
                )
            postflight_body = {
                "schema": f"{VERSION}_runtime_postflight_receipt",
                "version": VERSION,
                "runtime_bundle_receipt": dict(runtime_bundle_receipt),
                "runtime_bundle_receipt_sha256": stable_hash(
                    dict(runtime_bundle_receipt)
                ),
                "terminal_blocks": list(executed_blocks),
                "terminal_item_counts": {
                    block: BLOCK_COUNTS[block] for block in executed_blocks
                },
                "terminal_item_count": sum(
                    BLOCK_COUNTS[block] for block in executed_blocks
                ),
                "terminal_arm_counts": {
                    arm: sum(BLOCK_COUNTS[block] for block in executed_blocks)
                    for arm in ("RAW", "official_HippoRAG", "Agent")
                },
                "logical_RAW_Hippo_Agent_terminal_count": 3
                * sum(BLOCK_COUNTS[block] for block in executed_blocks),
                "complete_agent_recipe_trace_count": len(runner.RECIPE_IDS)
                * sum(BLOCK_COUNTS[block] for block in executed_blocks),
                "label_free_archive_count": len(executed_blocks),
                "label_free_seal_count": len(executed_blocks),
                "M_search_executed": "M_search" in executed_blocks,
                "external_network_calls": 0,
                "online_evaluator_calls": 0,
            }
            runtime_postflight = _self_hashed(
                postflight_body, field="runtime_postflight_receipt_sha256"
            )
            runtime_postflight_artifact = _persist_receipt(
                project=root,
                output_paths=output_paths,
                name="runtime_postflight",
                kind="runtime_postflight",
                block="M_search" if "M_search" in executed_blocks else "A_hold",
                marker_sha256=marker_sha,
                receipt=runtime_postflight,
            )
            record("runtime_postflight", runtime_postflight_artifact)
            _verify_receipt_artifact(runtime_postflight_artifact)
            terminal_body.update(
                {
                    "runtime_postflight_receipt_sha256": runtime_postflight[
                        "runtime_postflight_receipt_sha256"
                    ],
                    "terminal_blocks": list(executed_blocks),
                    "terminal_item_counts": {
                        block: BLOCK_COUNTS[block] for block in executed_blocks
                    },
                }
            )

        failure_stage = "terminal_result"
        acquisition_boundary.assert_stable(
            project=root, prerequisites=prerequisites
        )
        _verify_marker_file(
            path=marker_path,
            expected=marker_payload,
            expected_file_sha256=marker_file_sha,
        )
        if not terminal_status:
            raise FeverousFormalControllerError("terminal status was not formed")
        result_body = {
            "schema": RESULT_SCHEMA,
            "version": VERSION,
            "status": terminal_status,
            "marker_sha256": marker_sha,
            "marker_file_sha256": marker_file_sha,
            "implementation_freeze_sha256": (
                prerequisites.implementation_freeze_sha256
            ),
            "acquisition_receipt_sha256": prerequisites.acquisition_receipt_sha256,
            "runtime_preflight_sha256": runtime_preflight_sha,
            "artifact_receipt_sha256s": {
                name: artifact.receipt_sha256
                for name, artifact in sorted(artifacts.items())
            },
            "artifact_file_sha256s": {
                name: artifact.file_sha256
                for name, artifact in sorted(artifacts.items())
            },
            **terminal_body,
            "retry_replay_resample_or_replacement_authorized": False,
            "same_source_epoch_rollback_authorized": False,
            "development_or_test_source_accessed": False,
            "external_network_calls": 0,
            "online_evaluator_calls": 0,
        }
        result = _self_hashed(result_body, field="result_sha256")
        write_json_exclusive(
            root / output_paths.result_relative, result, mode=0o644
        )
        return result
    except BaseException as exc:
        _write_terminal_failure(
            path=root / output_paths.failure_relative,
            marker_sha256=marker_sha,
            failure_stage=failure_stage,
            exc=exc,
        )
        raise


def run_formal_lifecycle(
    runtime_config: local_runtime.FormalRuntimeConfig,
) -> dict[str, Any]:
    """Run the sole production lifecycle; never use this for diagnostics."""

    if not isinstance(runtime_config, local_runtime.FormalRuntimeConfig):
        raise FeverousFormalControllerError("formal runtime config type drifted")
    project = _canonical_project(runtime_config.project)
    if runtime_config != local_runtime.default_formal_runtime_config(project):
        raise FeverousFormalControllerError("formal runtime config is not canonical")
    return _run_lifecycle_core(
        project=project,
        runtime_config=runtime_config,
        acquisition_boundary=ModuleAcquisitionBoundary(),
        runtime_boundary=DefaultRuntimeBoundary(),
        core=DefaultFormalCore(),
        output_paths=FORMAL_OUTPUT_PATHS,
    )


__all__ = [
    "ARCHIVE_SCHEMA",
    "BLOCK_COUNTS",
    "DefaultFormalCore",
    "DefaultRuntimeBoundary",
    "FAILURE_SCHEMA",
    "FORMAL_OUTPUT_PATHS",
    "FORMAL_OUTPUT_ROOT_RELATIVE",
    "FORMATION_QUERY_COUNT",
    "FeverousFormalControllerError",
    "FormationStages",
    "LabelFreeBlockStage",
    "LifecycleArtifact",
    "LifecycleOutputPaths",
    "MARKER_SCHEMA",
    "ModuleAcquisitionBoundary",
    "PrerequisiteBinding",
    "PreparedExecution",
    "RESULT_SCHEMA",
    "SEAL_SCHEMA",
    "VERSION",
    "formation_query_schedule",
    "persist_label_free_archive",
    "run_formal_lifecycle",
    "seal_label_free_archive",
    "split_formation_hippo_result",
    "stable_hash",
    "verify_archive_and_seal",
    "verify_self_hash",
    "write_json_exclusive",
]
