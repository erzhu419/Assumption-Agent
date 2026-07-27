"""One-shot, offline-only integration runtime for formal DSTC9 P1.

The trusted outer process compiles the pinned source exactly once and then
hands only typed public projections to the frozen controller and action
workers.  Qrels have a distinct loader and are opened only after the
controller supplies a durable action archive.  ``M_search`` has an additional
promotion-authorization boundary covering both its public block and qrels.

The two expensive public lanes are deliberately overlapped.  Loading the
public corpus starts one official HippoRAG global build on physical GPU0.
Once all three initial public blocks have been loaded, their 176 histories are
submitted as one coordinate-scoring batch on physical GPU1.  Controller calls
for A_form, F_search, and A_hold are served from that private cache.  If
promotion succeeds, the 48 M_search histories form one separate coordinate
batch while the unchanged HippoRAG index serves its second and final query
batch.

No action-side object receives a source path, private manifest, family, qrel,
label, split, or evaluator value.  There is no API or network evaluator and
there is no retry path.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import stat
import threading
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from assumption_agent.benchmarks import dstc9_p1_formal_controller_v1 as ctl
from assumption_agent.benchmarks import dstc9_p1_formal_source_v1 as source
from assumption_agent.benchmarks import dstc9_p1_typed_core_v1 as core
from replication_runtime.dstc9_coordinate_scorer_v1 import adapter as coord_adapter
from replication_runtime.dstc9_coordinate_scorer_v1 import contract as coord_contract
from replication_runtime.dstc9_official_hipporag_v1 import adapter as hippo_adapter
from replication_runtime.dstc9_official_hipporag_v1 import contract as hippo_contract
from replication_runtime.dstc9_official_hipporag_v1 import runtime_binding
from replication_runtime.qasper_minilm_portable_v2.binding import (
    PORTABLE_CANARY_SCHEMA,
    PortableOfflineMiniLMEncoder,
)
from replication_runtime.qasper_minilm_v1.binding import (
    EMBEDDING_DIMENSION,
    quantized_cosine_similarity,
)


FORMAL_RUNTIME_VERSION = "dstc9_p1_formal_runtime_v1"
FORMAL_CONFIG_SCHEMA = f"{FORMAL_RUNTIME_VERSION}_config_v1"
CANARY_CONFIG_SCHEMA = "dstc9_p1_source_free_canary_config_v2"
FORMAL_OUTER_TERMINAL_SCHEMA = f"{FORMAL_RUNTIME_VERSION}_safe_terminal_v1"
FORMAL_FAILURE_SCHEMA = f"{FORMAL_RUNTIME_VERSION}_safe_failure_terminal_v1"
CANARY_SCHEMA = "dstc9_p1_source_free_infrastructure_canary_receipt_v2"
STUDY_ID = ctl.STUDY_ID

OUTER_ATTEMPT_MARKER = "outer_formal_attempt.marker.json"
OUTER_TERMINAL = "formal_terminal.json"
CANARY_RECEIPT_FILENAME = "canary.receipt.json"
CURRENT_HARDWARE_BINDING_FILENAME = "current_hardware.binding.json"
CANARY_ATTEMPT_MARKER_FILENAME = "canary_attempt.marker.json"
SOURCE_STAGE_DIRECTORY = "source_stage"
CONTROLLER_STAGE_DIRECTORY = "controller_stage"
ACTION_RUNTIME_DIRECTORY = "action_runtime"

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_WORK_ID = re.compile(r"dstc9-work-v1-[0-9a-f]{64}\Z")

# These texts are source-independent and fixed before any formal source is
# opened.  They are public semantic prototypes, not keyword gates.
PREDICTOR_PROTOTYPES = (
    (
        "Travel accommodation: a hotel, room, lodging, reservation, "
        "facilities, check-in, or a place to stay."
    ),
    (
        "Dining: a restaurant, food, cuisine, menu, meal, reservation, "
        "or a place to eat."
    ),
    (
        "Local ground transportation: a taxi, cab, pickup, destination, "
        "fare, or a booked car journey."
    ),
    (
        "Rail transportation: a train, railway station, departure, arrival, "
        "ticket, schedule, or route."
    ),
)

PREDICTOR_COMMITMENT_PAYLOAD = {
    "bucket_order": list(core.PREDICTED_BUCKETS),
    "embedding_runtime": (
        "qasper_minilm_portable_v2.PortableOfflineMiniLMEncoder"
    ),
    "frozen_model_and_execution": (
        "unchanged_qasper_minilm_v1_asset_tree_CPU_float32_offline"
    ),
    "input_contract": (
        "typed_public_dialogue_history_serialized_model_query_only"
    ),
    "model_input_exclusions": [
        "corpus",
        "domain",
        "evaluator",
        "family",
        "label",
        "qrel",
        "score",
        "source",
        "split",
    ],
    "portable_startup_acceptance": {
        "acceptance_basis": (
            "public_synthetic_shape_dtype_finite_normalized_noncollapsed_"
            "repeat_elementwise_and_byte_exact_only"
        ),
        "expected_output_hash_or_allowlist_is_acceptance_oracle": False,
        "observed_output_hashes_are_normative": False,
        "repeat_count": 2,
        "schema": PORTABLE_CANARY_SCHEMA,
    },
    "prototype_texts": list(PREDICTOR_PROTOTYPES),
    "quantization": "qasper_minilm_v1_quantized_cosine_similarity",
    "selection": "largest_quantized_cosine_then_smallest_bucket_v1",
    "version": (
        f"{FORMAL_RUNTIME_VERSION}_"
        "public_prototype_predictor_portable_infrastructure_v2"
    ),
}
PREDICTOR_COMMITMENT = ctl.stable_hash(
    PREDICTOR_COMMITMENT_PAYLOAD
)


class Dstc9P1FormalRuntimeError(RuntimeError):
    """The one-shot outer lifecycle or a source-free action lane failed."""


def _canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise Dstc9P1FormalRuntimeError(
            "runtime value is not canonical JSON"
        ) from exc
    return raw + (b"\n" if newline else b"")


def _stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _with_self_hash(body: Mapping[str, object]) -> dict[str, object]:
    if "self_sha256" in body:
        raise Dstc9P1FormalRuntimeError("self hash supplied twice")
    value = dict(body)
    value["self_sha256"] = _stable_hash(value)
    return value


def _verify_self_hash(value: Mapping[str, object], field: str) -> str:
    body = dict(value)
    claimed = _required_sha256(body.pop("self_sha256", None), field)
    if not hmac.compare_digest(claimed, _stable_hash(body)):
        raise Dstc9P1FormalRuntimeError(f"{field} self hash drifted")
    return claimed


def _required_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise Dstc9P1FormalRuntimeError(f"{field} is not a SHA-256")
    return value


def _required_absolute(path: Path, field: str) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise Dstc9P1FormalRuntimeError(f"{field} must be absolute")
    return path


def _assert_no_symlink_components(path: Path, field: str) -> None:
    absolute = path.absolute()
    for component in (*reversed(absolute.parents), absolute):
        if component.is_symlink():
            raise Dstc9P1FormalRuntimeError(
                f"{field} contains a symlink component"
            )


def _exclusive_json(
    path: Path,
    value: Mapping[str, object],
    *,
    mode: int,
) -> None:
    raw = _canonical_bytes(value, newline=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, mode)
        try:
            os.fchmod(descriptor, mode)
            view = memoryview(raw)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short write")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise Dstc9P1FormalRuntimeError(
            f"exclusive runtime artifact failed: {path.name}"
        ) from exc


def _fresh_private_directory(path: Path, field: str) -> Path:
    _required_absolute(path, field)
    _assert_no_symlink_components(path.parent, f"{field} parent")
    try:
        path.mkdir(mode=0o700, parents=False, exist_ok=False)
    except OSError as exc:
        raise Dstc9P1FormalRuntimeError(
            f"{field} is not a fresh private directory"
        ) from exc
    if stat.S_IMODE(path.stat().st_mode) != 0o700:
        raise Dstc9P1FormalRuntimeError(f"{field} mode drifted")
    return path


@dataclass(frozen=True, slots=True)
class FormalRuntimeConfig:
    """All absolute source/runtime paths bound by the execution freeze."""

    formal_root: Path
    p0_receipt_path: Path
    private_eligibility_manifest_path: Path
    bundle_path: Path
    execution_binding_sha256: str
    coordinate_runtime_python: Path
    coordinate_project_root: Path
    minilm_asset_manifest: Path
    minilm_model_root: Path
    cross_encoder_model_root: Path
    hippo_runtime_python: Path
    hippo_worker_project_root: Path
    hippo_llm_model_root: Path
    hippo_embedding_model_root: Path
    hippo_runtime_fingerprint_path: Path
    current_hardware_binding_path: Path
    current_hardware_binding_file_sha256: str
    current_hardware_binding_self_sha256: str
    source_free_canary_receipt_path: Path
    source_free_canary_receipt_file_sha256: str
    source_free_canary_receipt_self_sha256: str
    coordinate_timeout_seconds: int = 14_400
    hippo_build_timeout_seconds: int = 7_200
    hippo_retrieve_timeout_seconds: int = 3_600

    def __post_init__(self) -> None:
        for name in (
            "formal_root",
            "p0_receipt_path",
            "private_eligibility_manifest_path",
            "bundle_path",
            "coordinate_runtime_python",
            "coordinate_project_root",
            "minilm_asset_manifest",
            "minilm_model_root",
            "cross_encoder_model_root",
            "hippo_runtime_python",
            "hippo_worker_project_root",
            "hippo_llm_model_root",
            "hippo_embedding_model_root",
            "hippo_runtime_fingerprint_path",
            "current_hardware_binding_path",
            "source_free_canary_receipt_path",
        ):
            _required_absolute(getattr(self, name), name)
        _required_sha256(
            self.execution_binding_sha256, "execution binding"
        )
        for name in (
            "current_hardware_binding_file_sha256",
            "current_hardware_binding_self_sha256",
            "source_free_canary_receipt_file_sha256",
            "source_free_canary_receipt_self_sha256",
        ):
            _required_sha256(getattr(self, name), name)
        for name, maximum in (
            ("coordinate_timeout_seconds", 14_400),
            ("hippo_build_timeout_seconds", 14_400),
            ("hippo_retrieve_timeout_seconds", 14_400),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise Dstc9P1FormalRuntimeError(
                    f"{name} is outside the frozen integer bound"
                )

    @classmethod
    def from_payload(cls, value: object) -> "FormalRuntimeConfig":
        path_fields = (
            "formal_root",
            "p0_receipt_path",
            "private_eligibility_manifest_path",
            "bundle_path",
            "coordinate_runtime_python",
            "coordinate_project_root",
            "minilm_asset_manifest",
            "minilm_model_root",
            "cross_encoder_model_root",
            "hippo_runtime_python",
            "hippo_worker_project_root",
            "hippo_llm_model_root",
            "hippo_embedding_model_root",
            "hippo_runtime_fingerprint_path",
            "current_hardware_binding_path",
            "source_free_canary_receipt_path",
        )
        scalar_fields = (
            "execution_binding_sha256",
            "current_hardware_binding_file_sha256",
            "current_hardware_binding_self_sha256",
            "source_free_canary_receipt_file_sha256",
            "source_free_canary_receipt_self_sha256",
            "coordinate_timeout_seconds",
            "hippo_build_timeout_seconds",
            "hippo_retrieve_timeout_seconds",
        )
        expected = {
            "schema",
            "self_sha256",
            *path_fields,
            *scalar_fields,
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise Dstc9P1FormalRuntimeError(
                "formal runtime config schema drifted"
            )
        if value.get("schema") != FORMAL_CONFIG_SCHEMA:
            raise Dstc9P1FormalRuntimeError(
                "formal runtime config identity drifted"
            )
        _verify_self_hash(value, "formal runtime config")
        return cls(
            **{name: Path(str(value[name])) for name in path_fields},
            **{name: value[name] for name in scalar_fields},
        )


@dataclass(frozen=True, slots=True)
class CanaryRuntimeConfig:
    """Source-free canary config: intentionally has no formal source channel."""

    canary_root: Path
    current_hardware_binding_path: Path
    hardware_capture_id: str
    canary_binding_sha256: str
    coordinate_runtime_python: Path
    coordinate_project_root: Path
    minilm_asset_manifest: Path
    minilm_model_root: Path
    cross_encoder_model_root: Path
    hippo_runtime_python: Path
    hippo_worker_project_root: Path
    hippo_llm_model_root: Path
    hippo_embedding_model_root: Path
    hippo_runtime_fingerprint_path: Path
    coordinate_timeout_seconds: int = 14_400
    hippo_build_timeout_seconds: int = 7_200
    hippo_retrieve_timeout_seconds: int = 3_600

    def __post_init__(self) -> None:
        for name in (
            "canary_root",
            "current_hardware_binding_path",
            "coordinate_runtime_python",
            "coordinate_project_root",
            "minilm_asset_manifest",
            "minilm_model_root",
            "cross_encoder_model_root",
            "hippo_runtime_python",
            "hippo_worker_project_root",
            "hippo_llm_model_root",
            "hippo_embedding_model_root",
            "hippo_runtime_fingerprint_path",
        ):
            _required_absolute(getattr(self, name), name)
        _required_sha256(self.canary_binding_sha256, "canary binding")
        if (
            not isinstance(self.hardware_capture_id, str)
            or re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._:+-]{0,255}",
                self.hardware_capture_id,
            )
            is None
        ):
            raise Dstc9P1FormalRuntimeError(
                "hardware capture ID drifted"
            )
        for name in (
            "coordinate_timeout_seconds",
            "hippo_build_timeout_seconds",
            "hippo_retrieve_timeout_seconds",
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= 14_400:
                raise Dstc9P1FormalRuntimeError(
                    f"{name} is outside the frozen integer bound"
                )

    @classmethod
    def from_payload(cls, value: object) -> "CanaryRuntimeConfig":
        path_fields = (
            "canary_root",
            "current_hardware_binding_path",
            "coordinate_runtime_python",
            "coordinate_project_root",
            "minilm_asset_manifest",
            "minilm_model_root",
            "cross_encoder_model_root",
            "hippo_runtime_python",
            "hippo_worker_project_root",
            "hippo_llm_model_root",
            "hippo_embedding_model_root",
            "hippo_runtime_fingerprint_path",
        )
        scalar_fields = (
            "hardware_capture_id",
            "canary_binding_sha256",
            "coordinate_timeout_seconds",
            "hippo_build_timeout_seconds",
            "hippo_retrieve_timeout_seconds",
        )
        if (
            not isinstance(value, Mapping)
            or set(value)
            != {"schema", "self_sha256", *path_fields, *scalar_fields}
            or value.get("schema") != CANARY_CONFIG_SCHEMA
        ):
            raise Dstc9P1FormalRuntimeError(
                "source-free canary config schema drifted"
            )
        _verify_self_hash(value, "source-free canary config")
        return cls(
            **{name: Path(str(value[name])) for name in path_fields},
            **{name: value[name] for name in scalar_fields},
        )


def _load_config(
    path: Path,
) -> FormalRuntimeConfig | CanaryRuntimeConfig:
    _required_absolute(path, "config path")
    _assert_no_symlink_components(path, "config path")
    if not path.is_file():
        raise Dstc9P1FormalRuntimeError("config path is unavailable")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Dstc9P1FormalRuntimeError("config is invalid JSON") from exc
    if raw not in {_canonical_bytes(value), _canonical_bytes(value, newline=True)}:
        raise Dstc9P1FormalRuntimeError("config is not canonical JSON")
    if isinstance(value, Mapping) and value.get("schema") == FORMAL_CONFIG_SCHEMA:
        return FormalRuntimeConfig.from_payload(value)
    if isinstance(value, Mapping) and value.get("schema") == CANARY_CONFIG_SCHEMA:
        return CanaryRuntimeConfig.from_payload(value)
    raise Dstc9P1FormalRuntimeError("runtime config identity drifted")


class PublicPrototypeBucketPredictor:
    """Frozen query-only semantic bucket predictor."""

    commitment = PREDICTOR_COMMITMENT

    def __init__(self, encoder: object) -> None:
        encode = getattr(encoder, "encode", None)
        if not callable(encode):
            raise Dstc9P1FormalRuntimeError(
                "prototype predictor encoder is unavailable"
            )
        self._encoder = encoder
        self._prototype_matrix = self._matrix(
            encode(PREDICTOR_PROTOTYPES),
            len(PREDICTOR_PROTOTYPES),
            "prototype",
        )

    @classmethod
    def from_paths(
        cls,
        *,
        asset_manifest_path: Path,
        model_root: Path,
    ) -> "PublicPrototypeBucketPredictor":
        return cls(
            PortableOfflineMiniLMEncoder(
                asset_manifest_path=asset_manifest_path,
                model_root=model_root,
            )
        )

    @staticmethod
    def _matrix(value: object, rows: int, field: str) -> np.ndarray:
        try:
            matrix = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise Dstc9P1FormalRuntimeError(
                f"{field} embedding matrix is invalid"
            ) from exc
        if (
            matrix.shape != (rows, EMBEDDING_DIMENSION)
            or not np.isfinite(matrix).all()
        ):
            raise Dstc9P1FormalRuntimeError(
                f"{field} embedding matrix drifted"
            )
        norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
        if not np.allclose(norms, 1.0, rtol=0.0, atol=2e-6):
            raise Dstc9P1FormalRuntimeError(
                f"{field} embeddings are not normalized"
            )
        return matrix

    def predict(
        self,
        items: Sequence[ctl.FormalItemView],
    ) -> Sequence[ctl.BucketPrediction]:
        if (
            isinstance(items, (str, bytes))
            or not isinstance(items, Sequence)
            or not items
            or any(not isinstance(item, ctl.FormalItemView) for item in items)
        ):
            raise Dstc9P1FormalRuntimeError(
                "prototype predictor input drifted"
            )
        queries = tuple(core.serialize_model_query(item.history) for item in items)
        query_matrix = self._matrix(
            self._encoder.encode(queries),
            len(items),
            "query",
        )
        output: list[ctl.BucketPrediction] = []
        for item, query in zip(items, query_matrix):
            similarities = tuple(
                quantized_cosine_similarity(query, prototype)
                for prototype in self._prototype_matrix
            )
            bucket = min(
                core.PREDICTED_BUCKETS,
                key=lambda index: (-similarities[index], index),
            )
            output.append(
                ctl.BucketPrediction.create(
                    item=item,
                    predicted_bucket=bucket,
                    predictor_commitment=self.commitment,
                )
            )
        return tuple(output)


CoordinateRun = Callable[..., Mapping[str, object]]


class CoordinateScorerLane:
    """One initial 176-query GPU1 batch plus, conditionally, one M batch."""

    def __init__(
        self,
        *,
        runtime_python: Path,
        project_root: Path,
        minilm_asset_manifest: Path,
        minilm_model_root: Path,
        cross_encoder_model_root: Path,
        lane_root: Path,
        timeout_seconds: int,
        run_callable: CoordinateRun = (
            coord_adapter.run_dstc9_coordinate_scorer_v1
        ),
    ) -> None:
        self._runtime_python = runtime_python
        self._project_root = project_root
        self._minilm_asset_manifest = minilm_asset_manifest
        self._minilm_model_root = minilm_model_root
        self._cross_encoder_model_root = cross_encoder_model_root
        self._lane_root = _fresh_private_directory(
            lane_root, "coordinate lane root"
        )
        self._timeout_seconds = timeout_seconds
        self._run = run_callable
        self._corpus_sha256: str | None = None
        self._initial_cache: dict[str, ctl.CoordinateScoreRow] | None = None
        self._initial_work_ids: frozenset[str] | None = None
        self._m_cache: dict[str, ctl.CoordinateScoreRow] | None = None
        self._call_count = 0
        self._lock = threading.Lock()

    @property
    def worker_call_count(self) -> int:
        return self._call_count

    def _execute(
        self,
        corpus: ctl.CorpusView,
        items: Sequence[ctl.FormalItemView],
        *,
        stage_name: str,
    ) -> dict[str, ctl.CoordinateScoreRow]:
        checked = tuple(items)
        if not 1 <= len(checked) <= coord_contract.MAX_QUERY_COUNT:
            raise Dstc9P1FormalRuntimeError(
                "coordinate batch exceeds the frozen 1..256 bound"
            )
        payload = coord_contract.input_payload(
            snippets=tuple(
                core.snippet_public_payload(row) for row in corpus.snippets
            ),
            histories=tuple(
                {
                    "turns": [
                        core.turn_public_payload(turn)
                        for turn in item.history
                    ],
                    "work_id": item.work_id,
                }
                for item in checked
            ),
        )
        output = self._run(
            input_value=payload,
            runtime_python=self._runtime_python,
            project_root=self._project_root,
            minilm_asset_manifest=self._minilm_asset_manifest,
            minilm_model_root=self._minilm_model_root,
            cross_encoder_model_root=self._cross_encoder_model_root,
            work_root=self._lane_root / stage_name,
            timeout_seconds=self._timeout_seconds,
        )
        self._call_count += 1
        raw_rows = output.get("rows")
        if (
            not isinstance(raw_rows, list)
            or len(raw_rows) != len(checked)
        ):
            raise Dstc9P1FormalRuntimeError(
                "coordinate adapter output coverage drifted"
            )
        item_by_work = {item.work_id: item for item in checked}
        result: dict[str, ctl.CoordinateScoreRow] = {}
        for raw in raw_rows:
            if not isinstance(raw, Mapping):
                raise Dstc9P1FormalRuntimeError(
                    "coordinate adapter row drifted"
                )
            work_id = raw.get("work_id")
            vectors = raw.get("vectors")
            if (
                not isinstance(work_id, str)
                or work_id not in item_by_work
                or work_id in result
                or not isinstance(vectors, Mapping)
            ):
                raise Dstc9P1FormalRuntimeError(
                    "coordinate adapter work binding drifted"
                )
            result[work_id] = ctl.CoordinateScoreRow.create(
                item=item_by_work[work_id],
                corpus=corpus,
                score_vectors=vectors,
            )
        if set(result) != set(item_by_work):
            raise Dstc9P1FormalRuntimeError(
                "coordinate adapter coverage drifted"
            )
        return result

    def prime_initial(
        self,
        corpus: ctl.CorpusView,
        blocks: Mapping[str, ctl.BlockView],
    ) -> None:
        if tuple(blocks) != ctl.INITIAL_BLOCKS:
            raise Dstc9P1FormalRuntimeError(
                "initial coordinate block order drifted"
            )
        items = tuple(
            item
            for block_name in ctl.INITIAL_BLOCKS
            for item in blocks[block_name].items
        )
        expected = sum(ctl.BLOCK_COUNTS[name] for name in ctl.INITIAL_BLOCKS)
        if (
            len(items) != expected
            or len({item.work_id for item in items}) != expected
        ):
            raise Dstc9P1FormalRuntimeError(
                "initial coordinate batch coverage drifted"
            )
        with self._lock:
            if self._initial_cache is not None:
                raise Dstc9P1FormalRuntimeError(
                    "initial coordinate batch was primed twice"
                )
            self._corpus_sha256 = corpus.view_sha256
            self._initial_cache = self._execute(
                corpus, items, stage_name="initial_176"
            )
            self._initial_work_ids = frozenset(self._initial_cache)

    def score(
        self,
        corpus: ctl.CorpusView,
        items: Sequence[ctl.FormalItemView],
    ) -> Sequence[ctl.CoordinateScoreRow]:
        checked = tuple(items)
        if (
            not checked
            or self._corpus_sha256 != corpus.view_sha256
            or any(not isinstance(item, ctl.FormalItemView) for item in checked)
        ):
            raise Dstc9P1FormalRuntimeError(
                "coordinate scorer corpus/item binding drifted"
            )
        work_ids = frozenset(item.work_id for item in checked)
        if len(work_ids) != len(checked):
            raise Dstc9P1FormalRuntimeError(
                "coordinate scorer work IDs are not unique"
            )
        with self._lock:
            if (
                self._initial_cache is not None
                and self._initial_work_ids is not None
                and work_ids <= self._initial_work_ids
            ):
                return tuple(self._initial_cache[item.work_id] for item in checked)
            if {item.block for item in checked} != {"M_search"}:
                raise Dstc9P1FormalRuntimeError(
                    "coordinate scorer received an unregistered batch"
                )
            if self._m_cache is None:
                if len(checked) != ctl.BLOCK_COUNTS["M_search"]:
                    raise Dstc9P1FormalRuntimeError(
                        "M_search coordinate coverage drifted"
                    )
                self._m_cache = self._execute(
                    corpus, checked, stage_name="M_search_48"
                )
            if set(self._m_cache) != work_ids:
                raise Dstc9P1FormalRuntimeError(
                    "M_search coordinate batch changed after formation"
                )
            return tuple(self._m_cache[item.work_id] for item in checked)


HippoBuild = Callable[..., Mapping[str, object]]
HippoRetrieve = Callable[..., hippo_contract.RetrievalBatch]


class OfficialHippoLane:
    """Build one public global index asynchronously and reopen it twice."""

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
        self._runtime_python = runtime_python
        self._worker_project_root = worker_project_root
        self._current_hardware_binding_path = (
            current_hardware_binding_path
        )
        self._local_llm_model = local_llm_model
        self._local_embedding_model = local_embedding_model
        self._runtime_fingerprint_path = runtime_fingerprint_path
        self._lane_root = _fresh_private_directory(
            lane_root, "HippoRAG lane root"
        )
        self._stage_root = self._lane_root / "global_build"
        self._build_timeout_seconds = build_timeout_seconds
        self._retrieve_timeout_seconds = retrieve_timeout_seconds
        self._build = build_callable
        self._retrieve = retrieve_callable
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="dstc9-hippo-build"
        )
        self._build_future: Future[Mapping[str, object]] | None = None
        self._corpus_view_sha256: str | None = None
        self._retrieve_blocks: set[str] = set()
        self._lock = threading.Lock()
        self._closed = False
        self._private_retrieval_commitments: dict[str, str] = {}

    @property
    def build_call_count(self) -> int:
        return int(self._build_future is not None)

    @property
    def retrieve_call_count(self) -> int:
        return len(self._retrieve_blocks)

    @property
    def private_retrieval_commitments(self) -> Mapping[str, str]:
        return dict(self._private_retrieval_commitments)

    @staticmethod
    def _corpus_input(corpus: ctl.CorpusView) -> dict[str, object]:
        return hippo_contract.make_corpus_input(
            study_id=STUDY_ID,
            units=tuple(
                {
                    "ordinal": snippet.ordinal,
                    "text": core.serialize_passage(snippet),
                }
                for snippet in corpus.snippets
            ),
        )

    def start_build(self, corpus: ctl.CorpusView) -> None:
        if not isinstance(corpus, ctl.CorpusView):
            raise Dstc9P1FormalRuntimeError(
                "HippoRAG build corpus drifted"
            )
        with self._lock:
            if self._closed or self._build_future is not None:
                raise Dstc9P1FormalRuntimeError(
                    "HippoRAG global build was started twice"
                )
            corpus_input = self._corpus_input(corpus)
            self._corpus_view_sha256 = corpus.view_sha256
            self._build_future = self._executor.submit(
                self._build,
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

    def retrieve(
        self,
        corpus: ctl.CorpusView,
        items: Sequence[ctl.FormalItemView],
    ) -> Sequence[ctl.HippoResult]:
        checked = tuple(items)
        blocks = {item.block for item in checked}
        if (
            not checked
            or len(blocks) != 1
            or next(iter(blocks)) not in ctl.SCORING_BLOCKS
            or self._corpus_view_sha256 != corpus.view_sha256
        ):
            raise Dstc9P1FormalRuntimeError(
                "HippoRAG retrieve batch drifted"
            )
        block = next(iter(blocks))
        with self._lock:
            if (
                self._build_future is None
                or block in self._retrieve_blocks
                or self._closed
            ):
                raise Dstc9P1FormalRuntimeError(
                    "HippoRAG retrieve lifecycle drifted"
                )
            self._retrieve_blocks.add(block)
            future = self._build_future
        # Waiting here preserves the causal build-once boundary while allowing
        # the build to overlap the initial coordinate worker.
        future.result()
        query_input = hippo_contract.make_query_input(
            study_id=STUDY_ID,
            queries=tuple(
                {
                    "ordinal": ordinal,
                    "query_text": core.serialize_model_query(item.history),
                    "work_id": item.work_id,
                }
                for ordinal, item in enumerate(checked)
            ),
        )
        batch = self._retrieve(
            query_input=query_input,
            runtime_python=self._runtime_python,
            worker_project_root=self._worker_project_root,
            current_hardware_binding_path=(
                self._current_hardware_binding_path
            ),
            local_llm_model=self._local_llm_model,
            local_embedding_model=self._local_embedding_model,
            runtime_fingerprint_path=self._runtime_fingerprint_path,
            stage_root=self._stage_root,
            work_root=self._lane_root / f"retrieve_{block}",
            timeout_seconds=self._retrieve_timeout_seconds,
        )
        if (
            not isinstance(batch, hippo_contract.RetrievalBatch)
            or len(batch.ordinals) != len(checked)
        ):
            raise Dstc9P1FormalRuntimeError(
                "HippoRAG result coverage drifted"
            )
        receipt_sha = _required_sha256(
            batch.receipt.get("receipt_sha256"), "HippoRAG receipt"
        )
        private_evidence = _with_self_hash(
            {
                "block": block,
                "build_once": True,
                "corpus_projection_sha256": (
                    corpus.projection_sha256
                ),
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
        _exclusive_json(
            self._lane_root / f"{block}.retrieval.private.json",
            private_evidence,
            mode=0o400,
        )
        self._private_retrieval_commitments[block] = str(
            private_evidence["self_sha256"]
        )
        return tuple(
            ctl.HippoResult(
                work_id=item.work_id,
                block=item.block,
                normalized_query_sha256=core.normalized_query_sha256(
                    item.history
                ),
                corpus_projection_sha256=corpus.projection_sha256,
                top5_ordinals=tuple(ordinals),
                receipt_sha256=receipt_sha,
            )
            for item, ordinals in zip(checked, batch.ordinals)
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=False)


class SealedSourceAcquisitionBoundary:
    """Exact artifact loader with separate public, qrel, and M channels."""

    def __init__(
        self,
        *,
        outputs: source.FormalOutputPaths,
        selection_receipt: Mapping[str, object],
        controller_root: Path,
        coordinate_lane: CoordinateScorerLane,
        hippo_lane: OfficialHippoLane,
    ) -> None:
        if not isinstance(outputs, source.FormalOutputPaths):
            raise Dstc9P1FormalRuntimeError(
                "formal source output registry drifted"
            )
        if not isinstance(selection_receipt, Mapping):
            raise Dstc9P1FormalRuntimeError(
                "selection receipt is unavailable"
            )
        self._receipt = dict(selection_receipt)
        _verify_self_hash(self._receipt, "selection receipt")
        try:
            receipt_metadata = outputs.safe_selection_receipt.lstat()
            receipt_raw = outputs.safe_selection_receipt.read_bytes()
            persisted_receipt = json.loads(receipt_raw.decode("ascii"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Dstc9P1FormalRuntimeError(
                "sealed selection receipt is unavailable"
            ) from exc
        if (
            outputs.safe_selection_receipt.is_symlink()
            or not stat.S_ISREG(receipt_metadata.st_mode)
            or receipt_metadata.st_nlink != 1
            or stat.S_IMODE(receipt_metadata.st_mode) != 0o600
            or receipt_raw
            != source.canonical_bytes(persisted_receipt, newline=True)
            or persisted_receipt != self._receipt
        ):
            raise Dstc9P1FormalRuntimeError(
                "sealed selection receipt differs from compiler return"
            )
        if (
            self._receipt.get("schema") != source.SELECTION_RECEIPT_SCHEMA
            or self._receipt.get("status") != "selected_and_sealed"
            or self._receipt.get("study_id") != STUDY_ID
        ):
            raise Dstc9P1FormalRuntimeError(
                "selection receipt identity drifted"
            )
        self._outputs = outputs
        self._controller_root = controller_root.absolute()
        self._coordinate_lane = coordinate_lane
        self._hippo_lane = hippo_lane
        self._claim: ctl.AcquisitionClaim | None = None
        self._corpus: ctl.CorpusView | None = None
        self._blocks: dict[str, ctl.BlockView] = {}
        self._qrel_opened: set[str] = set()
        self._m_authorized = False
        self.public_open_count = {
            "corpus": 0,
            **{block: 0 for block in source.BLOCKS},
        }
        self.qrel_open_count = {
            block: 0 for block in source.QREL_BLOCKS
        }

        try:
            artifacts = self._receipt["artifact_binding"]
            self._public_bindings = {
                "corpus": artifacts["public_corpus"],
                **dict(artifacts["public_blocks"]),
            }
            self._qrel_bindings = dict(artifacts["private_qrels"])
        except (KeyError, TypeError) as exc:
            raise Dstc9P1FormalRuntimeError(
                "selection receipt artifact registry drifted"
            ) from exc

        self._source_commitment = _stable_hash(
            {
                "p0_binding": self._receipt.get("p0_binding"),
                "source_access": self._receipt.get("source_access"),
            }
        )
        self._corpus_commitment = _required_sha256(
            self._public_bindings["corpus"].get("self_sha256"),
            "corpus selection commitment",
        )
        self._disjointness_commitment = _stable_hash(
            {
                "disjointness_aggregate": self._receipt.get(
                    "disjointness_aggregate"
                ),
                "quota": self._receipt.get("quota"),
                "selection": self._receipt.get("selection"),
            }
        )

    def _load_bound(
        self,
        path: Path,
        binding: Mapping[str, object],
        *,
        mode: int,
        field: str,
    ) -> dict[str, object]:
        _required_absolute(path, field)
        _assert_no_symlink_components(path, field)
        try:
            metadata = path.lstat()
            raw = path.read_bytes()
            value = json.loads(raw.decode("ascii"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Dstc9P1FormalRuntimeError(
                f"{field} is unavailable or invalid"
            ) from exc
        if (
            path.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != mode
            or metadata.st_size != binding.get("size_bytes")
            or hashlib.sha256(raw).hexdigest()
            != binding.get("file_sha256")
            or raw != _canonical_bytes(value, newline=True)
            or not isinstance(value, dict)
            or value.get("self_sha256") != binding.get("self_sha256")
        ):
            raise Dstc9P1FormalRuntimeError(f"{field} binding drifted")
        _verify_self_hash(value, field)
        return value

    def claim_formal_attempt(
        self, formal_marker_sha256: str
    ) -> ctl.AcquisitionClaim:
        _required_sha256(formal_marker_sha256, "formal marker")
        if self._claim is not None:
            raise Dstc9P1FormalRuntimeError(
                "formal acquisition was claimed twice"
            )
        self._claim = ctl.AcquisitionClaim.create(
            source_identity_commitment=self._source_commitment,
            corpus_selection_commitment=self._corpus_commitment,
            block_disjointness_commitment=self._disjointness_commitment,
            query_only_predictor_commitment=PREDICTOR_COMMITMENT,
        )
        return self._claim

    def load_public_corpus(
        self, claim: ctl.AcquisitionClaim
    ) -> ctl.CorpusView:
        if (
            self._claim is None
            or claim != self._claim
            or self._corpus is not None
        ):
            raise Dstc9P1FormalRuntimeError(
                "public corpus lifecycle drifted"
            )
        value = self._load_bound(
            self._outputs.public_corpus,
            self._public_bindings["corpus"],
            mode=0o600,
            field="public corpus",
        )
        if (
            value.get("schema") != source.PUBLIC_CORPUS_SCHEMA
            or value.get("study_id") != STUDY_ID
            or value.get("version") != source.VERSION
            or not isinstance(value.get("snippets"), list)
        ):
            raise Dstc9P1FormalRuntimeError(
                "public corpus schema drifted"
            )
        try:
            snippets = tuple(
                core.snippet_from_public_fields(row)
                for row in value["snippets"]
            )
        except (TypeError, core.Dstc9P1TypedCoreError) as exc:
            raise Dstc9P1FormalRuntimeError(
                "public corpus typed projection drifted"
            ) from exc
        if (
            len(snippets) != ctl.CORPUS_SIZE
            or self._public_bindings["corpus"].get("row_count")
            != ctl.CORPUS_SIZE
        ):
            raise Dstc9P1FormalRuntimeError(
                "public corpus exact row count drifted"
            )
        self._corpus = ctl.CorpusView.create(snippets)
        if (
            self._corpus.projection_sha256
            != core.stable_hash(value["snippets"])
        ):
            raise Dstc9P1FormalRuntimeError(
                "public corpus projection binding drifted"
            )
        self.public_open_count["corpus"] += 1
        # This is the earliest point at which the public corpus exists.
        self._hippo_lane.start_build(self._corpus)
        return self._corpus

    def _valid_m_authorization(
        self, value: Mapping[str, object] | None
    ) -> bool:
        if not isinstance(value, Mapping):
            return False
        expected_keys = {
            "A_hold_E1_minus_E0",
            "block_disjointness_commitment",
            "comparison_net_strictly_positive",
            "one_sided_exact_magnitude_preserving_tail_at_most_one_tenth",
            "schema",
            "self_sha256",
            "status",
            "study_id",
        }
        try:
            comparison = value.get("A_hold_E1_minus_E0")
            if not isinstance(comparison, Mapping):
                return False
            net = comparison.get("net_utility")
            tail = comparison.get(
                "one_sided_exact_magnitude_preserving_tail"
            )
            if (
                type(net) is not int
                or net <= 0
                or not isinstance(tail, Mapping)
                or type(tail.get("numerator")) is not int
                or type(tail.get("denominator")) is not int
                or tail["denominator"] <= 0
                or tail["numerator"] < 0
                or Fraction(
                    tail["numerator"], tail["denominator"]
                )
                > ctl.ALPHA
            ):
                return False
            authorization_path = (
                self._controller_root
                / ctl.PROMOTION_AUTHORIZATION_FILENAME
            )
            if (
                authorization_path.is_symlink()
                or not authorization_path.is_file()
                or authorization_path.stat().st_nlink != 1
                or stat.S_IMODE(authorization_path.stat().st_mode) != 0o400
                or authorization_path.read_bytes()
                != ctl.canonical_bytes(value)
            ):
                return False
            return (
                set(value) == expected_keys
                and value.get("schema")
                == (
                    f"{ctl.VERSION}_"
                    "M_search_materialization_authorization_v1"
                )
                and value.get("study_id") == STUDY_ID
                and value.get("status") == "A_hold_E1_promoted"
                and value.get("comparison_net_strictly_positive") is True
                and value.get(
                    "one_sided_exact_magnitude_preserving_tail_at_most_one_tenth"
                )
                is True
                and value.get("block_disjointness_commitment")
                == self._disjointness_commitment
                and _verify_self_hash(value, "M_search authorization")
                == value.get("self_sha256")
            )
        except Dstc9P1FormalRuntimeError:
            return False

    def load_label_free_block(
        self,
        block: str,
        authorization: Mapping[str, object] | None = None,
    ) -> ctl.BlockView:
        if self._claim is None or self._corpus is None:
            raise Dstc9P1FormalRuntimeError(
                "public block requested before corpus"
            )
        if block not in source.BLOCKS or block in self._blocks:
            raise Dstc9P1FormalRuntimeError(
                "public block lifecycle drifted"
            )
        if block == "M_search":
            if not self._valid_m_authorization(authorization):
                raise Dstc9P1FormalRuntimeError(
                    "M_search requires valid promotion authorization"
                )
            self._m_authorized = True
        elif authorization is not None:
            raise Dstc9P1FormalRuntimeError(
                "initial public block cannot receive authorization"
            )
        value = self._load_bound(
            self._outputs.public_blocks()[block],
            self._public_bindings[block],
            mode=0o400 if block == "M_search" else 0o600,
            field=f"{block} public block",
        )
        if (
            value.get("schema") != source.PUBLIC_BLOCK_SCHEMA
            or value.get("study_id") != STUDY_ID
            or value.get("version") != source.VERSION
            or value.get("block_id") != block
            or not isinstance(value.get("items"), list)
        ):
            raise Dstc9P1FormalRuntimeError(
                f"{block} public block schema drifted"
            )
        items: list[ctl.FormalItemView] = []
        try:
            for raw in value["items"]:
                if not isinstance(raw, Mapping) or set(raw) != source.PUBLIC_ITEM_KEYS:
                    raise Dstc9P1FormalRuntimeError(
                        f"{block} public item schema drifted"
                    )
                history = tuple(
                    core.turn_from_public_fields(turn)
                    for turn in raw["history"]
                )
                item = ctl.FormalItemView(
                    work_id=str(raw["work_id"]),
                    block=block,
                    history=history,
                )
                if (
                    raw["normalized_query_sha256"]
                    != core.normalized_query_sha256(history)
                ):
                    raise Dstc9P1FormalRuntimeError(
                        f"{block} query binding drifted"
                    )
                items.append(item)
        except (
            KeyError,
            TypeError,
            core.Dstc9P1TypedCoreError,
            ctl.Dstc9P1FormalControllerError,
        ) as exc:
            if isinstance(exc, Dstc9P1FormalRuntimeError):
                raise
            raise Dstc9P1FormalRuntimeError(
                f"{block} typed item projection drifted"
            ) from exc
        if (
            len(items) != ctl.BLOCK_COUNTS[block]
            or self._public_bindings[block].get("row_count")
            != ctl.BLOCK_COUNTS[block]
        ):
            raise Dstc9P1FormalRuntimeError(
                f"{block} exact public row count drifted"
            )
        view = ctl.BlockView.create(block, items)
        self._blocks[block] = view
        self.public_open_count[block] += 1
        if block == "A_hold":
            if tuple(self._blocks) != ctl.INITIAL_BLOCKS:
                raise Dstc9P1FormalRuntimeError(
                    "initial public block load order drifted"
                )
            self._coordinate_lane.prime_initial(
                self._corpus,
                {name: self._blocks[name] for name in ctl.INITIAL_BLOCKS},
            )
        return view

    def release_qrels_after_action_seal(
        self,
        block: str,
        custody_path: Path,
        sealed_action_archive: Mapping[str, object],
    ) -> ctl.QrelPack:
        if (
            block not in source.QREL_BLOCKS
            or block in self._qrel_opened
            or block not in self._blocks
            or self._corpus is None
        ):
            raise Dstc9P1FormalRuntimeError(
                "late-qrel lifecycle drifted"
            )
        if block == "M_search" and not self._m_authorized:
            raise Dstc9P1FormalRuntimeError(
                "M_search qrels require promotion authorization"
            )
        expected_name = f"{block}.actions.private.json"
        expected_path = self._controller_root / expected_name
        if (
            not isinstance(custody_path, Path)
            or custody_path.absolute() != expected_path
            or custody_path.is_symlink()
            or not custody_path.is_file()
            or stat.S_IMODE(custody_path.stat().st_mode) != 0o400
            or not isinstance(sealed_action_archive, Mapping)
        ):
            raise Dstc9P1FormalRuntimeError(
                "qrel release requires the exact sealed action archive"
            )
        archive_value = dict(sealed_action_archive)
        try:
            archive_self = _verify_self_hash(
                archive_value, f"{block} action archive"
            )
        except Dstc9P1FormalRuntimeError:
            raise
        if (
            archive_value.get("block") != block
            or archive_value.get("study_id") != STUDY_ID
            or custody_path.read_bytes() != ctl.canonical_bytes(archive_value)
        ):
            raise Dstc9P1FormalRuntimeError(
                "sealed action archive binding drifted"
            )
        value = self._load_bound(
            self._outputs.private_qrels()[block],
            self._qrel_bindings[block],
            mode=0o400,
            field=f"{block} private qrels",
        )
        if (
            value.get("schema") != source.PRIVATE_QREL_SCHEMA
            or value.get("study_id") != STUDY_ID
            or value.get("version") != source.VERSION
            or value.get("block_id") != block
            or not isinstance(value.get("qrels"), list)
        ):
            raise Dstc9P1FormalRuntimeError(
                f"{block} private qrel schema drifted"
            )
        rows: list[ctl.QrelRow] = []
        try:
            for raw in value["qrels"]:
                if (
                    not isinstance(raw, Mapping)
                    or set(raw) != source.PRIVATE_QREL_ROW_KEYS
                ):
                    raise Dstc9P1FormalRuntimeError(
                        f"{block} private qrel row drifted"
                    )
                rows.append(
                    ctl.QrelRow(
                        work_id=str(raw["work_id"]),
                        family=str(raw["family"]),
                        gold_ordinal=raw["gold_ordinal"],
                        corpus_projection_sha256=(
                            self._corpus.projection_sha256
                        ),
                    )
                )
        except (
            KeyError,
            TypeError,
            ctl.Dstc9P1FormalControllerError,
        ) as exc:
            if isinstance(exc, Dstc9P1FormalRuntimeError):
                raise
            raise Dstc9P1FormalRuntimeError(
                f"{block} private qrel typed projection drifted"
            ) from exc
        if (
            len(rows) != ctl.BLOCK_COUNTS[block]
            or self._qrel_bindings[block].get("row_count")
            != ctl.BLOCK_COUNTS[block]
        ):
            raise Dstc9P1FormalRuntimeError(
                f"{block} exact private qrel row count drifted"
            )
        pack = ctl.QrelPack.create(
            block=block,
            action_archive_sha256=archive_self,
            rows=rows,
        )
        self._qrel_opened.add(block)
        self.qrel_open_count[block] += 1
        return pack


def _source_output_paths(source_root: Path) -> source.FormalOutputPaths:
    return source.FormalOutputPaths(
        public_corpus=source_root / "public_corpus.json",
        public_a_form=source_root / "A_form.public.json",
        public_f_search=source_root / "F_search.public.json",
        public_a_hold=source_root / "A_hold.public.json",
        public_m_search=source_root / "M_search.public.private.json",
        private_a_form_qrels=source_root / "A_form.qrels.private.json",
        private_a_hold_qrels=source_root / "A_hold.qrels.private.json",
        private_m_search_qrels=source_root / "M_search.qrels.private.json",
        safe_selection_receipt=source_root / "selection.receipt.json",
    )


def _load_exact_safe_receipt(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_self_sha256: str,
    field: str,
) -> dict[str, object]:
    _required_absolute(path, field)
    _assert_no_symlink_components(path, field)
    try:
        metadata = path.lstat()
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Dstc9P1FormalRuntimeError(f"{field} is unavailable") from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or hashlib.sha256(raw).hexdigest() != expected_file_sha256
        or raw != _canonical_bytes(value, newline=True)
        or not isinstance(value, dict)
        or value.get("self_sha256") != expected_self_sha256
        or _verify_self_hash(value, field) != expected_self_sha256
    ):
        raise Dstc9P1FormalRuntimeError(f"{field} binding drifted")
    return value


def _verify_preformal_hardware_and_canary(
    config: FormalRuntimeConfig,
) -> dict[str, object]:
    hardware_value = _load_exact_safe_receipt(
        config.current_hardware_binding_path,
        expected_file_sha256=(
            config.current_hardware_binding_file_sha256
        ),
        expected_self_sha256=(
            config.current_hardware_binding_self_sha256
        ),
        field="current hardware binding",
    )
    try:
        live = runtime_binding.verify_current_study_hardware_binding(
            path=config.current_hardware_binding_path,
            worker_project_root=config.hippo_worker_project_root,
            expected_study_id=STUDY_ID,
        )
    except runtime_binding.Dstc9P17RuntimeBindingError as exc:
        raise Dstc9P1FormalRuntimeError(
            "current hardware no longer matches the pre-canary binding"
        ) from exc
    if (
        live.get("receipt_file_sha256")
        != config.current_hardware_binding_file_sha256
        or live.get("receipt_self_sha256")
        != config.current_hardware_binding_self_sha256
        or hardware_value.get("study_id") != STUDY_ID
    ):
        raise Dstc9P1FormalRuntimeError(
            "live current-hardware receipt identity drifted"
        )
    canary = _load_exact_safe_receipt(
        config.source_free_canary_receipt_path,
        expected_file_sha256=(
            config.source_free_canary_receipt_file_sha256
        ),
        expected_self_sha256=(
            config.source_free_canary_receipt_self_sha256
        ),
        field="source-free canary receipt",
    )
    if (
        canary.get("schema") != CANARY_SCHEMA
        or canary.get("status")
        != "passed_source_free_two_lane_canary_once"
        or canary.get("study_id") != STUDY_ID
        or canary.get("formal_source_access_count") != 0
        or canary.get("current_hardware_binding_file_sha256")
        != config.current_hardware_binding_file_sha256
        or canary.get("current_hardware_binding_self_sha256")
        != config.current_hardware_binding_self_sha256
        or canary.get("predictor_count") != 1
        or canary.get("predictor_commitment_sha256")
        != PREDICTOR_COMMITMENT
        or not isinstance(
            canary.get("predictor_result_commitment_sha256"), str
        )
        or _HEX64.fullmatch(
            str(canary.get("predictor_result_commitment_sha256"))
        )
        is None
        or canary.get("predictor_result_commitment_sha256")
        == "0" * 64
    ):
        raise Dstc9P1FormalRuntimeError(
            "source-free canary receipt binding drifted"
        )
    return {
        "canary_receipt_self_sha256": (
            config.source_free_canary_receipt_self_sha256
        ),
        "hardware_binding_self_sha256": (
            config.current_hardware_binding_self_sha256
        ),
    }


def _write_outer_failure(
    root: Path,
    *,
    config: FormalRuntimeConfig,
    stage: str,
    exc: BaseException,
) -> Mapping[str, object]:
    body = _with_self_hash(
        {
            "aggregate_only_public_terminal": True,
            "execution_binding_sha256": config.execution_binding_sha256,
            "current_hardware_binding_self_sha256": (
                config.current_hardware_binding_self_sha256
            ),
            "failure_exception_message_sha256": hashlib.sha256(
                str(exc).encode("utf-8", errors="replace")
            ).hexdigest(),
            "failure_exception_type_sha256": hashlib.sha256(
                type(exc).__name__.encode("ascii", errors="replace")
            ).hexdigest(),
            "failure_stage": stage,
            "item_query_document_qrel_action_or_per_item_score_values_published": False,
            "online_or_API_evaluator_calls": 0,
            "retry_replay_resample_model_provider_candidate_parser_family_quota_or_gate_change_count": 0,
            "schema": FORMAL_FAILURE_SCHEMA,
            "status": "terminal_formal_failure_no_retry",
            "study_id": STUDY_ID,
        }
    )
    try:
        _exclusive_json(root / OUTER_TERMINAL, body, mode=0o600)
    except Exception:
        pass
    return body


def run_formal_study_once(
    config: FormalRuntimeConfig,
    *,
    compile_callable: Callable[..., Mapping[str, object]] = (
        source.compile_formal_source
    ),
    controller_callable: Callable[..., Mapping[str, object]] = (
        ctl.run_formal_controller
    ),
    predictor_factory: Callable[..., PublicPrototypeBucketPredictor] = (
        PublicPrototypeBucketPredictor.from_paths
    ),
    coordinate_factory: Callable[..., CoordinateScorerLane] = (
        CoordinateScorerLane
    ),
    hippo_factory: Callable[..., OfficialHippoLane] = OfficialHippoLane,
    preformal_verifier: Callable[
        [FormalRuntimeConfig], Mapping[str, object]
    ] = _verify_preformal_hardware_and_canary,
) -> Mapping[str, object]:
    """Compile the formal source and execute the controller exactly once."""

    if not isinstance(config, FormalRuntimeConfig):
        raise Dstc9P1FormalRuntimeError("formal config type drifted")
    root = _fresh_private_directory(config.formal_root, "formal root")
    marker = _with_self_hash(
        {
            "execution_binding_sha256": config.execution_binding_sha256,
            "retry_count": 0,
            "schema": f"{FORMAL_RUNTIME_VERSION}_outer_attempt_marker_v1",
            "study_id": STUDY_ID,
        }
    )
    _exclusive_json(root / OUTER_ATTEMPT_MARKER, marker, mode=0o400)
    source_root = _fresh_private_directory(
        root / SOURCE_STAGE_DIRECTORY, "source stage root"
    )
    controller_root = root / CONTROLLER_STAGE_DIRECTORY
    action_root = _fresh_private_directory(
        root / ACTION_RUNTIME_DIRECTORY, "action runtime root"
    )
    outputs = _source_output_paths(source_root)
    stage = "verify_live_hardware_and_source_free_canary_before_source"
    hippo_lane: OfficialHippoLane | None = None
    try:
        preformal_binding = preformal_verifier(config)
        stage = "compile_formal_source_once"
        selection_receipt = compile_callable(
            p0_receipt_path=config.p0_receipt_path,
            private_eligibility_manifest_path=(
                config.private_eligibility_manifest_path
            ),
            bundle_path=config.bundle_path,
            outputs=outputs,
            contract=source.DEFAULT_CONTRACT,
        )
        stage = "initialize_source_free_action_runtime"
        predictor = predictor_factory(
            asset_manifest_path=config.minilm_asset_manifest,
            model_root=config.minilm_model_root,
        )
        if predictor.commitment != PREDICTOR_COMMITMENT:
            raise Dstc9P1FormalRuntimeError(
                "predictor commitment drifted"
            )
        coordinate_lane = coordinate_factory(
            runtime_python=config.coordinate_runtime_python,
            project_root=config.coordinate_project_root,
            minilm_asset_manifest=config.minilm_asset_manifest,
            minilm_model_root=config.minilm_model_root,
            cross_encoder_model_root=config.cross_encoder_model_root,
            lane_root=action_root / "coordinate",
            timeout_seconds=config.coordinate_timeout_seconds,
        )
        hippo_lane = hippo_factory(
            runtime_python=config.hippo_runtime_python,
            worker_project_root=config.hippo_worker_project_root,
            current_hardware_binding_path=(
                config.current_hardware_binding_path
            ),
            local_llm_model=config.hippo_llm_model_root,
            local_embedding_model=config.hippo_embedding_model_root,
            runtime_fingerprint_path=config.hippo_runtime_fingerprint_path,
            lane_root=action_root / "official_hipporag",
            build_timeout_seconds=config.hippo_build_timeout_seconds,
            retrieve_timeout_seconds=config.hippo_retrieve_timeout_seconds,
        )
        acquisition = SealedSourceAcquisitionBoundary(
            outputs=outputs,
            selection_receipt=selection_receipt,
            controller_root=controller_root,
            coordinate_lane=coordinate_lane,
            hippo_lane=hippo_lane,
        )
        stage = "run_frozen_formal_controller_once"
        controller_terminal = controller_callable(
            work_root=controller_root,
            execution_binding_sha256=config.execution_binding_sha256,
            acquisition=acquisition,
            predictor=predictor,
            coordinate_scorer=coordinate_lane,
            hippo_runner=hippo_lane,
        )
        stage = "seal_safe_outer_terminal"
        terminal = _with_self_hash(
            {
                "aggregate_only_public_terminal": True,
                "controller_terminal": dict(controller_terminal),
                "controller_terminal_self_sha256": _required_sha256(
                    controller_terminal.get("self_sha256"),
                    "controller terminal",
                ),
                "execution_binding_sha256": (
                    config.execution_binding_sha256
                ),
                "current_hardware_binding_file_sha256": (
                    config.current_hardware_binding_file_sha256
                ),
                "current_hardware_binding_self_sha256": (
                    config.current_hardware_binding_self_sha256
                ),
                "formal_source_access_count": 1,
                "hipporag_global_build_count": (
                    hippo_lane.build_call_count
                ),
                "hipporag_retrieve_count": (
                    hippo_lane.retrieve_call_count
                ),
                "private_hipporag_retrieval_archive_commitments": dict(
                    hippo_lane.private_retrieval_commitments
                ),
                "item_query_document_qrel_action_or_per_item_score_values_published": False,
                "online_or_API_evaluator_calls": 0,
                "private_coordinate_worker_count": (
                    coordinate_lane.worker_call_count
                ),
                "retry_replay_resample_model_provider_candidate_parser_family_quota_or_gate_change_count": 0,
                "schema": FORMAL_OUTER_TERMINAL_SCHEMA,
                "selection_receipt_self_sha256": _required_sha256(
                    selection_receipt.get("self_sha256"),
                    "selection receipt",
                ),
                "source_free_canary_receipt_self_sha256": (
                    preformal_binding["canary_receipt_self_sha256"]
                ),
                "status": "terminal_complete",
                "study_id": STUDY_ID,
            }
        )
        _exclusive_json(root / OUTER_TERMINAL, terminal, mode=0o600)
        return terminal
    except BaseException as exc:
        _write_outer_failure(root, config=config, stage=stage, exc=exc)
        if isinstance(exc, Dstc9P1FormalRuntimeError):
            raise
        raise Dstc9P1FormalRuntimeError(
            "formal integration failed closed"
        ) from exc
    finally:
        if hippo_lane is not None:
            hippo_lane.close()


def _synthetic_corpus_and_item() -> tuple[
    ctl.CorpusView, ctl.FormalItemView
]:
    five = (
        ("Harbor Hotel", "Hotel information", "Rooms and check-in."),
        ("Garden Restaurant", "Restaurant information", "Menu and booking."),
        ("City Taxi", "Taxi information", "Pickup and destination."),
        ("Central Train", "Train information", "Station and schedule."),
        (None, "General travel information", "Local visitor assistance."),
    )
    snippets = tuple(
        core.KnowledgeSnippet(
            ordinal=ordinal,
            entity_name=five[ordinal % 5][0],
            title=five[ordinal % 5][1],
            body=five[ordinal % 5][2],
        )
        for ordinal in range(ctl.CORPUS_SIZE)
    )
    corpus = ctl.CorpusView.create(snippets)
    item = ctl.FormalItemView(
        work_id=(
            "dstc9-work-v1-"
            + hashlib.sha256(b"source-free-canary-item").hexdigest()
        ),
        block="A_hold",
        history=(
            core.DialogueTurn(
                speaker="U",
                text="I need public travel information.",
            ),
        ),
    )
    return corpus, item


def run_source_free_canary_once(
    config: CanaryRuntimeConfig,
    *,
    predictor_factory: Callable[..., PublicPrototypeBucketPredictor] = (
        PublicPrototypeBucketPredictor.from_paths
    ),
    coordinate_factory: Callable[..., CoordinateScorerLane] = (
        CoordinateScorerLane
    ),
    hippo_factory: Callable[..., OfficialHippoLane] = OfficialHippoLane,
    hardware_capture_callable: Callable[..., Mapping[str, object]] = (
        runtime_binding.capture_current_study_hardware_binding
    ),
    hardware_verify_callable: Callable[..., Mapping[str, object]] = (
        runtime_binding.verify_current_study_hardware_binding
    ),
) -> Mapping[str, object]:
    """Run one source-free two-GPU canary; never inspect formal source paths."""

    if not isinstance(config, CanaryRuntimeConfig):
        raise Dstc9P1FormalRuntimeError("canary config type drifted")
    root = _fresh_private_directory(config.canary_root, "canary root")
    if (
        config.current_hardware_binding_path
        != root / CURRENT_HARDWARE_BINDING_FILENAME
    ):
        raise Dstc9P1FormalRuntimeError(
            "current hardware binding must be the exact canary-root artifact"
        )
    canary_marker = _with_self_hash(
        {
            "canary_binding_sha256": config.canary_binding_sha256,
            "formal_source_capability_present": False,
            "hardware_capture_id": config.hardware_capture_id,
            "retry_count": 0,
            "schema": (
                "dstc9_p1_source_free_canary_attempt_marker_v2"
            ),
            "study_id": STUDY_ID,
        }
    )
    _exclusive_json(
        root / CANARY_ATTEMPT_MARKER_FILENAME,
        canary_marker,
        mode=0o400,
    )
    stage = "capture_current_hardware_immediately_before_canary"
    try:
        hardware_value = hardware_capture_callable(
            study_id=STUDY_ID,
            capture_id=config.hardware_capture_id,
        )
        _exclusive_json(
            config.current_hardware_binding_path,
            hardware_value,
            mode=0o600,
        )
        hardware_raw = config.current_hardware_binding_path.read_bytes()
        hardware_file_sha256 = hashlib.sha256(hardware_raw).hexdigest()
        hardware_self_sha256 = _required_sha256(
            hardware_value.get("self_sha256"),
            "current hardware binding",
        )
        live_hardware = hardware_verify_callable(
            path=config.current_hardware_binding_path,
            worker_project_root=config.hippo_worker_project_root,
            expected_study_id=STUDY_ID,
        )
        if (
            live_hardware.get("receipt_file_sha256")
            != hardware_file_sha256
            or live_hardware.get("receipt_self_sha256")
            != hardware_self_sha256
        ):
            raise Dstc9P1FormalRuntimeError(
                "new current hardware binding did not verify live"
            )
    except BaseException as exc:
        failure = _with_self_hash(
            {
                "aggregate_only_public_receipt": True,
                "canary_binding_sha256": config.canary_binding_sha256,
                "canary_attempt_marker_self_sha256": (
                    canary_marker["self_sha256"]
                ),
                "failure_exception_message_sha256": hashlib.sha256(
                    str(exc).encode("utf-8", errors="replace")
                ).hexdigest(),
                "failure_stage": stage,
                "formal_source_access_count": 0,
                "online_or_API_evaluator_calls": 0,
                "retry_count": 0,
                "schema": f"{CANARY_SCHEMA}_failure_v2",
                "status": "failed_source_free_canary_no_retry",
                "study_id": STUDY_ID,
            }
        )
        try:
            _exclusive_json(
                root / CANARY_RECEIPT_FILENAME,
                failure,
                mode=0o600,
            )
        except Exception:
            pass
        if isinstance(exc, Dstc9P1FormalRuntimeError):
            raise
        raise Dstc9P1FormalRuntimeError(
            "source-free hardware capture failed closed"
        ) from exc
    stage = "initialize_source_free_canary_lanes"
    hippo_lane: OfficialHippoLane | None = None
    try:
        action_root = _fresh_private_directory(
            root / ACTION_RUNTIME_DIRECTORY, "canary action root"
        )
        coordinate_lane = coordinate_factory(
            runtime_python=config.coordinate_runtime_python,
            project_root=config.coordinate_project_root,
            minilm_asset_manifest=config.minilm_asset_manifest,
            minilm_model_root=config.minilm_model_root,
            cross_encoder_model_root=config.cross_encoder_model_root,
            lane_root=action_root / "coordinate",
            timeout_seconds=config.coordinate_timeout_seconds,
        )
        hippo_lane = hippo_factory(
            runtime_python=config.hippo_runtime_python,
            worker_project_root=config.hippo_worker_project_root,
            current_hardware_binding_path=(
                config.current_hardware_binding_path
            ),
            local_llm_model=config.hippo_llm_model_root,
            local_embedding_model=config.hippo_embedding_model_root,
            runtime_fingerprint_path=(
                config.hippo_runtime_fingerprint_path
            ),
            lane_root=action_root / "official_hipporag",
            build_timeout_seconds=config.hippo_build_timeout_seconds,
            retrieve_timeout_seconds=(
                config.hippo_retrieve_timeout_seconds
            ),
        )
        corpus, item = _synthetic_corpus_and_item()
        predictor = predictor_factory(
            asset_manifest_path=config.minilm_asset_manifest,
            model_root=config.minilm_model_root,
        )
        if (
            getattr(predictor, "commitment", None)
            != PREDICTOR_COMMITMENT
        ):
            raise Dstc9P1FormalRuntimeError(
                "source-free canary predictor commitment drifted"
            )
        predictions = predictor.predict((item,))
        if (
            not isinstance(predictions, Sequence)
            or len(predictions) != 1
            or not isinstance(predictions[0], ctl.BucketPrediction)
            or predictions[0].work_id != item.work_id
            or predictions[0].block != item.block
            or predictions[0].normalized_query_sha256
            != core.normalized_query_sha256(item.history)
            or predictions[0].predictor_commitment
            != PREDICTOR_COMMITMENT
            or predictions[0].predicted_bucket
            not in core.PREDICTED_BUCKETS
        ):
            raise Dstc9P1FormalRuntimeError(
                "source-free canary predictor coverage drifted"
            )
        predictor_result_commitment = _stable_hash(
            [predictions[0].audit_payload()]
        )
    except BaseException as exc:
        failure = _with_self_hash(
            {
                "aggregate_only_public_receipt": True,
                "canary_attempt_marker_self_sha256": (
                    canary_marker["self_sha256"]
                ),
                "canary_binding_sha256": config.canary_binding_sha256,
                "current_hardware_binding_file_sha256": (
                    hardware_file_sha256
                ),
                "current_hardware_binding_self_sha256": (
                    hardware_self_sha256
                ),
                "failure_exception_message_sha256": hashlib.sha256(
                    str(exc).encode("utf-8", errors="replace")
                ).hexdigest(),
                "failure_stage": stage,
                "formal_source_access_count": 0,
                "online_or_API_evaluator_calls": 0,
                "retry_count": 0,
                "schema": f"{CANARY_SCHEMA}_failure_v2",
                "status": "failed_source_free_canary_no_retry",
                "study_id": STUDY_ID,
            }
        )
        try:
            _exclusive_json(
                root / CANARY_RECEIPT_FILENAME,
                failure,
                mode=0o600,
            )
        except Exception:
            pass
        if hippo_lane is not None:
            hippo_lane.close()
        if isinstance(exc, Dstc9P1FormalRuntimeError):
            raise
        raise Dstc9P1FormalRuntimeError(
            "source-free canary initialization failed closed"
        ) from exc

    assert hippo_lane is not None
    stage = "source_free_two_lane_canary"
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            coordinate_future = pool.submit(
                coordinate_lane._execute,
                corpus,
                (item,),
                stage_name="synthetic_one_query",
            )

            def hippo_build_and_retrieve() -> Sequence[ctl.HippoResult]:
                hippo_lane.start_build(corpus)
                return hippo_lane.retrieve(corpus, (item,))

            hippo_future = pool.submit(hippo_build_and_retrieve)
            coordinate_rows = coordinate_future.result()
            hippo_rows = hippo_future.result()
        if (
            set(coordinate_rows) != {item.work_id}
            or len(hippo_rows) != 1
            or hippo_rows[0].work_id != item.work_id
        ):
            raise Dstc9P1FormalRuntimeError(
                "source-free canary coverage drifted"
            )
        receipt = _with_self_hash(
            {
                "aggregate_only_public_receipt": True,
                "coordinate_gpu": 1,
                "coordinate_query_count": 1,
                "coordinate_worker_count": coordinate_lane.worker_call_count,
                "canary_binding_sha256": (
                    config.canary_binding_sha256
                ),
                "canary_attempt_marker_self_sha256": (
                    canary_marker["self_sha256"]
                ),
                "current_hardware_binding_file_sha256": (
                    hardware_file_sha256
                ),
                "current_hardware_binding_self_sha256": (
                    hardware_self_sha256
                ),
                "formal_source_access_count": 0,
                "hipporag_build_count": hippo_lane.build_call_count,
                "hipporag_gpu": 0,
                "hipporag_query_count": 1,
                "hipporag_retrieve_count": hippo_lane.retrieve_call_count,
                "private_hipporag_retrieval_archive_commitments": dict(
                    getattr(
                        hippo_lane,
                        "private_retrieval_commitments",
                        {},
                    )
                ),
                "online_or_API_evaluator_calls": 0,
                "predictor_commitment_sha256": PREDICTOR_COMMITMENT,
                "predictor_count": 1,
                "predictor_result_commitment_sha256": (
                    predictor_result_commitment
                ),
                "retry_count": 0,
                "schema": CANARY_SCHEMA,
                "status": "passed_source_free_two_lane_canary_once",
                "study_id": STUDY_ID,
                "synthetic_corpus_count": ctl.CORPUS_SIZE,
                "synthetic_unique_serialized_text_count": 5,
            }
        )
        _exclusive_json(
            root / CANARY_RECEIPT_FILENAME, receipt, mode=0o600
        )
        return receipt
    except BaseException as exc:
        failure = _with_self_hash(
            {
                "aggregate_only_public_receipt": True,
                "canary_binding_sha256": (
                    config.canary_binding_sha256
                ),
                "canary_attempt_marker_self_sha256": (
                    canary_marker["self_sha256"]
                ),
                "current_hardware_binding_file_sha256": (
                    hardware_file_sha256
                ),
                "current_hardware_binding_self_sha256": (
                    hardware_self_sha256
                ),
                "failure_exception_message_sha256": hashlib.sha256(
                    str(exc).encode("utf-8", errors="replace")
                ).hexdigest(),
                "failure_stage": stage,
                "formal_source_access_count": 0,
                "online_or_API_evaluator_calls": 0,
                "retry_count": 0,
                "schema": f"{CANARY_SCHEMA}_failure_v2",
                "status": "failed_source_free_canary_no_retry",
                "study_id": STUDY_ID,
            }
        )
        try:
            _exclusive_json(
                root / CANARY_RECEIPT_FILENAME, failure, mode=0o600
            )
        except Exception:
            pass
        if isinstance(exc, Dstc9P1FormalRuntimeError):
            raise
        raise Dstc9P1FormalRuntimeError(
            "source-free canary failed closed"
        ) from exc
    finally:
        hippo_lane.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--formal", action="store_true")
    mode.add_argument("--source-free-canary", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = _load_config(args.config)
    if args.formal:
        if not isinstance(config, FormalRuntimeConfig):
            raise Dstc9P1FormalRuntimeError(
                "formal mode requires the formal config schema"
            )
        value = run_formal_study_once(config)
    else:
        if not isinstance(config, CanaryRuntimeConfig):
            raise Dstc9P1FormalRuntimeError(
                "canary mode requires the source-free config schema"
            )
        value = run_source_free_canary_once(config)
    print(
        _canonical_bytes(
            {
                "schema": value["schema"],
                "self_sha256": value["self_sha256"],
                "status": value["status"],
            }
        ).decode("ascii")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANARY_SCHEMA",
    "CANARY_CONFIG_SCHEMA",
    "CanaryRuntimeConfig",
    "FORMAL_CONFIG_SCHEMA",
    "FORMAL_RUNTIME_VERSION",
    "PREDICTOR_COMMITMENT",
    "PREDICTOR_COMMITMENT_PAYLOAD",
    "PREDICTOR_PROTOTYPES",
    "CoordinateScorerLane",
    "Dstc9P1FormalRuntimeError",
    "FormalRuntimeConfig",
    "OfficialHippoLane",
    "PublicPrototypeBucketPredictor",
    "SealedSourceAcquisitionBoundary",
    "main",
    "run_formal_study_once",
    "run_source_free_canary_once",
]
