"""One-shot offline formal lifecycle for the frozen EBM-NLP P1 study.

The controller owns scientific stage order and label release.  Runtime-heavy
operations cross two narrow injected boundaries: a local MiniLM embedding
executor and a bounded official-HippoRAG batch executor.  The controller has
no network, API, retry, reserve, resampling, or online-evaluation path.

Private action archives may contain PMIDs and rankings, but remain mode 0600
under the remote work root.  The public terminal contains only counts,
content hashes, aggregate exact-Fraction comparisons, and lifecycle status.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import stat
from typing import Any, Callable, Mapping, Protocol, Sequence

from replication_runtime.ebmnlp_p1_official_v1 import contract as hippo

from . import ebmnlp_p1_source_qualification_v1 as source
from . import ebmnlp_p1_typed_pico_core_v1 as core


VERSION = "ebmnlp_p1_formal_controller_v1"
STUDY_ID = core.STUDY_ID
ROLE_TO_SOURCE = {
    core.PARTICIPANT: "participants",
    core.INTERVENTION: "interventions",
    core.OUTCOME: "outcomes",
}
SAFE_TERMINAL_SCHEMA = f"{VERSION}_safe_terminal_v1"
ACTION_ARCHIVE_SCHEMA = f"{VERSION}_private_action_archive_v1"
EXPECTED_MINILM_TREE_SHA256 = (
    "1514beb65d2d3a2824a93f133a300cc60d5b437ccd6ea1e622eb4cd9881dcfdb"
)
EXPECTED_HIPPORAG_SOURCE_TREE_SHA256 = (
    "a644ab2811db2739db3cfbdc051561e2cfdf2ed87286f8ebd00a5971d189cdd5"
)
EXPECTED_HIPPORAG_LLM_TREE_SHA256 = (
    "d626d755c99c006761d5e069aa85a73fe8b011c6c0f5d0323a6f8de85246bcb5"
)


class EbmNlpP1FormalControllerError(RuntimeError):
    """A frozen lifecycle, private archive, or aggregate invariant failed."""


class EmbeddingExecutor(Protocol):
    """Frozen local MiniLM boundary; returned rows must be L2 normalized."""

    def __call__(
        self, texts: Sequence[str]
    ) -> Sequence[Sequence[float]]: ...

    def safe_runtime_receipt(self) -> Mapping[str, object]: ...


class HippoBatchExecutor(Protocol):
    """Frozen two-lane official-core boundary for complete abstract batches."""

    def __call__(
        self, payloads: Sequence[Mapping[str, object]]
    ) -> Mapping[str, Mapping[str, object]]: ...

    def safe_runtime_receipt(self) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class FormalExecutionBinding:
    """Pre-source hashes proving the exact frozen offline execution envelope."""

    implementation_freeze_sha256: str
    runtime_fingerprint_sha256: str
    source_free_canary_sha256: str
    execution_config_sha256: str
    execution_freeze_sha256: str
    live_execution_attestation_sha256: str
    source_archive_sha256: str
    minilm_tree_sha256: str
    hipporag_source_tree_sha256: str
    hipporag_llm_tree_sha256: str
    remote_host: str = "jtl311linux"
    gpu_assignment: tuple[str, str] = ("0", "1")
    maximum_hipporag_processes: int = 2
    external_network_call_count: int = 0
    online_or_api_evaluator_call_count: int = 0

    def payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RawAbstract:
    pmid: str
    windows: tuple[core.EvidenceWindow, ...]
    embeddings: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class PreparedAbstract:
    pmid: str
    abstract_work_id: str
    windows: tuple[core.EvidenceWindow, ...]
    slates: Mapping[str, core.RecipeSlate]
    role_work_ids: Mapping[str, str]


@dataclass(frozen=True)
class PreparedStage:
    block: str
    items: tuple[PreparedAbstract, ...]
    hippo_outputs: Mapping[str, Mapping[str, object]]


@dataclass(frozen=True)
class StageScores:
    arm_role_utilities: Mapping[
        str, tuple[Mapping[str, Fraction | None], ...]
    ]
    arm_role_coverages: Mapping[
        str, tuple[Mapping[str, Fraction | None], ...]
    ]
    arm_role_completes: Mapping[
        str, tuple[Mapping[str, int | None], ...]
    ]
    recipe_selection_counts: Mapping[str, Mapping[str, int]]
    typed_outside_raw: Mapping[str, Mapping[str, int]]
    zero_positive_role_count: int


@dataclass(frozen=True)
class SealedActionArchive:
    block: str
    path: Path
    file_sha256: str
    archive_sha256: str
    payload: Mapping[str, object]


def _canonical_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise EbmNlpP1FormalControllerError(
            "private value is not canonical JSON"
        ) from exc


def _stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _validate_formal_execution_binding(
    binding: FormalExecutionBinding,
) -> str:
    if not isinstance(binding, FormalExecutionBinding):
        raise EbmNlpP1FormalControllerError(
            "formal execution binding is absent"
        )
    payload = binding.payload()
    hash_fields = (
        "implementation_freeze_sha256",
        "runtime_fingerprint_sha256",
        "source_free_canary_sha256",
        "execution_config_sha256",
        "execution_freeze_sha256",
        "live_execution_attestation_sha256",
        "source_archive_sha256",
        "minilm_tree_sha256",
        "hipporag_source_tree_sha256",
        "hipporag_llm_tree_sha256",
    )
    if any(
        not isinstance(payload[field], str)
        or len(str(payload[field])) != 64
        or any(
            character not in "0123456789abcdef"
            for character in str(payload[field])
        )
        for field in hash_fields
    ):
        raise EbmNlpP1FormalControllerError(
            "formal execution binding hash drifted"
        )
    if (
        binding.source_archive_sha256
        != source.FORMAL_CONTRACT.archive_sha256
        or binding.minilm_tree_sha256
        != EXPECTED_MINILM_TREE_SHA256
        or binding.hipporag_source_tree_sha256
        != EXPECTED_HIPPORAG_SOURCE_TREE_SHA256
        or binding.hipporag_llm_tree_sha256
        != EXPECTED_HIPPORAG_LLM_TREE_SHA256
        or binding.remote_host != "jtl311linux"
        or binding.gpu_assignment != ("0", "1")
        or binding.maximum_hipporag_processes != 2
        or binding.external_network_call_count != 0
        or binding.online_or_api_evaluator_call_count != 0
    ):
        raise EbmNlpP1FormalControllerError(
            "formal execution binding policy drifted"
        )
    return _stable_hash(payload)


def _ensure_private_directory(path: Path, *, fresh: bool = False) -> None:
    if fresh:
        try:
            path.mkdir(mode=0o700)
        except OSError as exc:
            raise EbmNlpP1FormalControllerError(
                "one-shot work root is already consumed or unavailable"
            ) from exc
    else:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        info = path.lstat()
    except OSError as exc:
        raise EbmNlpP1FormalControllerError(
            "private directory is unavailable"
        ) from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise EbmNlpP1FormalControllerError("private directory is unsafe")
    os.chmod(path, 0o700)


def _write_once(path: Path, value: Mapping[str, object]) -> str:
    _ensure_private_directory(path.parent)
    raw = _canonical_bytes(value)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except OSError as exc:
        raise EbmNlpP1FormalControllerError(
            f"{path.name} is already consumed or unavailable"
        ) from exc
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(raw).hexdigest()


def _read_verified_private_archive(
    path: Path, *, expected_file_sha256: str
) -> tuple[dict[str, object], str]:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise EbmNlpP1FormalControllerError(
            "sealed action archive is unavailable"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_size > 256 * 1024 * 1024
        ):
            raise EbmNlpP1FormalControllerError(
                "sealed action archive metadata drifted"
            )
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                raise EbmNlpP1FormalControllerError(
                    "sealed action archive was truncated"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    if hashlib.sha256(raw).hexdigest() != expected_file_sha256:
        raise EbmNlpP1FormalControllerError(
            "sealed action archive file hash drifted"
        )
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EbmNlpP1FormalControllerError(
            "sealed action archive is not canonical JSON"
        ) from exc
    if (
        not isinstance(value, dict)
        or _canonical_bytes(value) != raw
    ):
        raise EbmNlpP1FormalControllerError(
            "sealed action archive canonical form drifted"
        )
    body = dict(value)
    claimed = body.pop("archive_sha256", None)
    if (
        not isinstance(claimed, str)
        or _stable_hash(body) != claimed
    ):
        raise EbmNlpP1FormalControllerError(
            "sealed action archive self hash drifted"
        )
    return value, claimed


def _read_private_tokens(
    acquisition: source.AcquisitionResult, pmid: str
) -> tuple[str, ...]:
    path = acquisition.private_root / "documents" / f"{pmid}.tokens"
    try:
        info = path.lstat()
        raw = path.read_bytes()
    except OSError as exc:
        raise EbmNlpP1FormalControllerError(
            "selected private token document is unavailable"
        ) from exc
    record = acquisition.documents[pmid]
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) & 0o077
        or hashlib.sha256(raw).hexdigest() != record.tokens_sha256
    ):
        raise EbmNlpP1FormalControllerError(
            "selected private token document identity drifted"
        )
    return source.parse_document_tokens(
        raw,
        maximum_tokens=acquisition.contract.maximum_tokens_per_document,
    )


def _validate_embeddings(
    rows: Sequence[Sequence[float]], *, expected_count: int
) -> tuple[tuple[float, ...], ...]:
    converted: list[tuple[float, ...]] = []
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise EbmNlpP1FormalControllerError(
            "embedding population is not a row sequence"
        )
    for row in rows:
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise EbmNlpP1FormalControllerError(
                "embedding row is not a numeric sequence"
            )
        values: list[float] = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise EbmNlpP1FormalControllerError(
                    "embedding coordinate is not a finite real"
                )
            converted_value = float(value)
            if not math.isfinite(converted_value):
                raise EbmNlpP1FormalControllerError(
                    "embedding coordinate is not a finite real"
                )
            values.append(converted_value)
        converted.append(tuple(values))
    values = tuple(converted)
    if len(values) != expected_count or not values:
        raise EbmNlpP1FormalControllerError(
            "embedding row count drifted"
        )
    width = len(values[0])
    if width <= 0:
        raise EbmNlpP1FormalControllerError(
            "embedding width is empty"
        )
    for row in values:
        if (
            len(row) != width
            or any(not math.isfinite(value) for value in row)
        ):
            raise EbmNlpP1FormalControllerError(
                "embedding shape or finite contract drifted"
            )
        norm = math.sqrt(math.fsum(value * value for value in row))
        if not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise EbmNlpP1FormalControllerError(
                "embedding row is not L2 normalized"
            )
    return values


def _executor_runtime_receipt(
    executor: object,
    *,
    kind: str,
    formal_scope: bool,
    expected_hippo_workers: int | None = None,
) -> tuple[str, Mapping[str, object]]:
    method = getattr(executor, "safe_runtime_receipt", None)
    if not callable(method):
        raise EbmNlpP1FormalControllerError(
            f"{kind} safe runtime receipt is unavailable"
        )
    receipt = method()
    if not isinstance(receipt, Mapping):
        raise EbmNlpP1FormalControllerError(
            f"{kind} safe runtime receipt drifted"
        )
    canonical = dict(receipt)
    if (
        canonical.get("external_network_call_count") != 0
        or canonical.get("online_or_api_evaluator_call_count") != 0
        or canonical.get("retry_or_replay_count") != 0
    ):
        raise EbmNlpP1FormalControllerError(
            f"{kind} offline or one-shot runtime policy drifted"
        )
    if kind == "embedding":
        if (
            canonical.get("schema")
            != "ebmnlp_p1_local_minilm_embedder_v1_safe_runtime_receipt"
            or not isinstance(canonical.get("call_count"), int)
            or canonical.get("call_count", 0) <= 0
            or not isinstance(canonical.get("encoded_text_count"), int)
            or canonical.get("encoded_text_count", 0) <= 0
        ):
            raise EbmNlpP1FormalControllerError(
                "embedding runtime receipt shape drifted"
            )
        if formal_scope and (
            canonical.get("model_tree_sha256")
            != EXPECTED_MINILM_TREE_SHA256
            or canonical.get("device") != "cuda:0"
            or canonical.get("dtype") != "float32"
            or canonical.get("embedding_dimension") != 384
        ):
            raise EbmNlpP1FormalControllerError(
                "formal embedding runtime identity drifted"
            )
    elif kind == "hipporag":
        if (
            canonical.get("schema")
            != "ebmnlp_p1_official_hipporag_batch_v1_safe_runtime_receipt"
            or type(expected_hippo_workers) is not int
            or canonical.get("worker_attempt_count")
            != expected_hippo_workers
            or canonical.get("worker_completed_count")
            != expected_hippo_workers
            or canonical.get("index_destroyed_count")
            != expected_hippo_workers
            or canonical.get("attempted_network_syscall_count")
            != canonical.get("denied_network_syscall_count")
        ):
            raise EbmNlpP1FormalControllerError(
                "HippoRAG runtime audit receipt drifted"
            )
        if formal_scope and (
            canonical.get("gpu_assignment") != ["0", "1"]
            or canonical.get("maximum_process_count") != 2
            or not isinstance(
                canonical.get("observed_process_peak"), int
            )
            or not 1
            <= canonical.get("observed_process_peak", 0)
            <= 2
        ):
            raise EbmNlpP1FormalControllerError(
                "formal HippoRAG runtime concurrency drifted"
            )
    else:
        raise EbmNlpP1FormalControllerError(
            "unknown executor receipt kind"
        )
    return _stable_hash(canonical), canonical


def _raw_abstracts(
    acquisition: source.AcquisitionResult,
    *,
    block: str,
    embedder: EmbeddingExecutor,
) -> tuple[RawAbstract, ...]:
    pmids = acquisition.assignment.pmids(block)
    windows_by_pmid: list[tuple[core.EvidenceWindow, ...]] = []
    texts: list[str] = []
    for pmid in pmids:
        windows = core.build_evidence_windows(
            _read_private_tokens(acquisition, pmid)
        )
        windows_by_pmid.append(windows)
        texts.extend(window.text for window in windows)
    embeddings = _validate_embeddings(
        embedder(tuple(texts)), expected_count=len(texts)
    )
    cursor = 0
    items: list[RawAbstract] = []
    for pmid, windows in zip(pmids, windows_by_pmid):
        end = cursor + len(windows)
        items.append(
            RawAbstract(
                pmid=pmid,
                windows=windows,
                embeddings=embeddings[cursor:end],
            )
        )
        cursor = end
    if cursor != len(embeddings):
        raise EbmNlpP1FormalControllerError(
            "embedding population slicing drifted"
        )
    return tuple(items)


def _window_binary_labels(
    windows: Sequence[core.EvidenceWindow],
    token_labels: Sequence[int],
) -> tuple[int, ...]:
    return tuple(
        int(any(token_labels[position] != 0 for position in range(window.start, window.end)))
        for window in windows
    )


def _plain_numeric_state(value: object, *, label: str) -> object:
    """Convert numpy/sklearn numeric state to strict canonical JSON values."""

    if hasattr(value, "tolist"):
        try:
            return _plain_numeric_state(
                getattr(value, "tolist")(), label=label
            )
        except Exception as exc:
            raise EbmNlpP1FormalControllerError(
                f"{label} could not be serialized"
            ) from exc
    if isinstance(value, (list, tuple)):
        return [
            _plain_numeric_state(item, label=label) for item in value
        ]
    if isinstance(value, bool):
        raise EbmNlpP1FormalControllerError(
            f"{label} contains a Boolean coordinate"
        )
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise EbmNlpP1FormalControllerError(
                f"{label} contains a non-finite coordinate"
            )
        return number
    raise EbmNlpP1FormalControllerError(
        f"{label} contains a nonnumeric coordinate"
    )


def _probe_state_payload(
    probes: core.FrozenRoleProbes,
) -> dict[str, object]:
    """Serialize the actual fitted probe coefficients, not only fit counts."""

    if not isinstance(probes, core.FrozenRoleProbes):
        raise EbmNlpP1FormalControllerError(
            "G_form probe registry drifted"
        )
    rows: list[dict[str, object]] = []
    frozen_parameters = {
        "C": 1,
        "class_weight": "balanced",
        "fit_intercept": True,
        "max_iter": 1000,
        "penalty": "l2",
        "random_state": 0,
        "solver": "liblinear",
        "tol": 1e-6,
    }
    for role, model in zip(core.ROLE_ORDER, probes.models):
        if isinstance(model, core.ConstantProbabilityProbe):
            probability = model.probability
            rows.append(
                {
                    "kind": "constant_empirical_class_probability",
                    "probability": {
                        "denominator": probability.denominator,
                        "numerator": probability.numerator,
                    },
                    "role": role,
                }
            )
            continue
        if not isinstance(model, core.SklearnProbabilityProbe):
            raise EbmNlpP1FormalControllerError(
                "G_form probe implementation drifted"
            )
        estimator = model.estimator
        getter = getattr(estimator, "get_params", None)
        if not callable(getter):
            raise EbmNlpP1FormalControllerError(
                "G_form sklearn probe parameters are unavailable"
            )
        parameters = getter(deep=False)
        if (
            not isinstance(parameters, Mapping)
            or any(
                parameters.get(key) != expected
                for key, expected in frozen_parameters.items()
            )
        ):
            raise EbmNlpP1FormalControllerError(
                "G_form sklearn probe hyperparameters drifted"
            )
        classes = _plain_numeric_state(
            getattr(estimator, "classes_", None),
            label=f"{role} classes",
        )
        coefficients = _plain_numeric_state(
            getattr(estimator, "coef_", None),
            label=f"{role} coefficients",
        )
        intercept = _plain_numeric_state(
            getattr(estimator, "intercept_", None),
            label=f"{role} intercept",
        )
        iterations = _plain_numeric_state(
            getattr(estimator, "n_iter_", None),
            label=f"{role} iterations",
        )
        if (
            classes != [0, 1]
            or model.positive_column != 1
            or not isinstance(coefficients, list)
            or len(coefficients) != 1
            or not isinstance(coefficients[0], list)
            or not coefficients[0]
            or not isinstance(intercept, list)
            or len(intercept) != 1
            or not isinstance(iterations, list)
            or len(iterations) != 1
        ):
            raise EbmNlpP1FormalControllerError(
                "G_form sklearn probe fitted state shape drifted"
            )
        rows.append(
            {
                "classes": classes,
                "coefficient": coefficients,
                "estimator_class": (
                    f"{type(estimator).__module__}."
                    f"{type(estimator).__qualname__}"
                ),
                "frozen_parameters": dict(frozen_parameters),
                "intercept": intercept,
                "iteration_count": iterations,
                "kind": "sklearn_binary_logistic_regression",
                "positive_column": model.positive_column,
                "role": role,
            }
        )
    return {
        "schema": f"{VERSION}_private_G_form_probe_state_v1",
        "study_id": STUDY_ID,
        "role_order": list(core.ROLE_ORDER),
        "models": rows,
        "encoder_trainable_parameter_count": 0,
    }


def _assert_probe_state(
    probes: core.FrozenRoleProbes, expected_sha256: str
) -> None:
    if _stable_hash(_probe_state_payload(probes)) != expected_sha256:
        raise EbmNlpP1FormalControllerError(
            "G_form probe state drifted after sealing"
        )


def fit_G_form_probes(
    acquisition: source.AcquisitionResult,
    *,
    embedder: EmbeddingExecutor,
) -> tuple[core.FrozenRoleProbes, str]:
    raw_items = _raw_abstracts(
        acquisition, block="G_form", embedder=embedder
    )
    labels = source.open_labels_for_stage(
        acquisition,
        stage="G_form",
        authorization=source.LabelOpenAuthorization(
            stage="G_form",
            source_sha256=acquisition.inventory.archive_sha256,
            assignment_sha256=acquisition.assignment.assignment_sha256,
            prerequisites_sealed=True,
        ),
    )
    embeddings: list[tuple[float, ...]] = []
    by_role: dict[str, list[int]] = {
        role: [] for role in core.ROLE_ORDER
    }
    for item in raw_items:
        embeddings.extend(item.embeddings)
        for role in core.ROLE_ORDER:
            by_role[role].extend(
                _window_binary_labels(
                    item.windows,
                    labels[item.pmid][ROLE_TO_SOURCE[role]],
                )
            )
    probes = core.fit_independent_role_probes(
        embeddings,
        {role: tuple(values) for role, values in by_role.items()},
    )
    return probes, _stable_hash(_probe_state_payload(probes))


def _opaque_id(
    acquisition: source.AcquisitionResult,
    block: str,
    ordinal: int,
    role: str | None,
) -> str:
    return hashlib.sha256(
        _canonical_bytes(
            {
                "assignment_sha256": acquisition.assignment.assignment_sha256,
                "block": block,
                "ordinal": ordinal,
                "role": role,
                "study_id": STUDY_ID,
            }
        )
    ).hexdigest()


def _query_embeddings(
    embedder: EmbeddingExecutor,
) -> Mapping[str, tuple[float, ...]]:
    rows = _validate_embeddings(
        embedder(tuple(core.ROLE_QUERIES[role] for role in core.ROLE_ORDER)),
        expected_count=len(core.ROLE_ORDER),
    )
    return dict(zip(core.ROLE_ORDER, rows))


def _prepare_stage_items(
    acquisition: source.AcquisitionResult,
    *,
    block: str,
    probes: core.FrozenRoleProbes,
    embedder: EmbeddingExecutor,
    query_embeddings: Mapping[str, Sequence[float]],
) -> tuple[PreparedAbstract, ...]:
    raw_items = _raw_abstracts(
        acquisition, block=block, embedder=embedder
    )
    prepared: list[PreparedAbstract] = []
    for ordinal, raw in enumerate(raw_items):
        probabilities = probes.score_quantized(raw.embeddings)
        quantized_embeddings = tuple(
            tuple(core.quantize_half_even(value) for value in row)
            for row in raw.embeddings
        )
        slates: dict[str, core.RecipeSlate] = {}
        role_work_ids: dict[str, str] = {}
        for role in core.ROLE_ORDER:
            query = tuple(float(value) for value in query_embeddings[role])
            if len(query) != len(raw.embeddings[0]):
                raise EbmNlpP1FormalControllerError(
                    "query and window embedding widths differ"
                )
            query_cosines = tuple(
                core.quantize_half_even(
                    min(1.0, max(0.0, (math.fsum(a * b for a, b in zip(row, query)) + 1.0) / 2.0)),
                    unit_interval=True,
                )
                for row in raw.embeddings
            )
            slates[role] = core.build_recipe_slate(
                windows=raw.windows,
                target_role=role,
                role_probabilities=probabilities,
                query_cosines=query_cosines,
                embeddings=quantized_embeddings,
            )
            role_work_ids[role] = _opaque_id(
                acquisition, block, ordinal, role
            )
        prepared.append(
            PreparedAbstract(
                pmid=raw.pmid,
                abstract_work_id=_opaque_id(
                    acquisition, block, ordinal, None
                ),
                windows=raw.windows,
                slates=slates,
                role_work_ids=role_work_ids,
            )
        )
    return tuple(prepared)


def _hippo_payload(item: PreparedAbstract) -> dict[str, object]:
    return hippo.input_payload(
        abstract_work_id=item.abstract_work_id,
        documents=[
            {
                "ordinal": window.ordinal,
                "text": window.text,
                "window_id": window.window_id,
            }
            for window in item.windows
        ],
        queries=[
            {
                "ordinal": ordinal,
                "role": role,
                "text": core.ROLE_QUERIES[role],
                "work_id": item.role_work_ids[role],
            }
            for ordinal, role in enumerate(core.ROLE_ORDER)
        ],
    )


def _validated_hippo_outputs(
    items: Sequence[PreparedAbstract],
    launcher: HippoBatchExecutor,
) -> Mapping[str, Mapping[str, object]]:
    payloads = tuple(_hippo_payload(item) for item in items)
    raw_outputs = launcher(payloads)
    if set(raw_outputs) != {
        item.abstract_work_id for item in items
    }:
        raise EbmNlpP1FormalControllerError(
            "HippoRAG batch output identity set drifted"
        )
    validated: dict[str, Mapping[str, object]] = {}
    for item, payload in zip(items, payloads):
        raw = _canonical_bytes(dict(raw_outputs[item.abstract_work_id]))
        try:
            output = hippo.parse_output(raw)
        except Exception as exc:
            raise EbmNlpP1FormalControllerError(
                "HippoRAG output contract drifted"
            ) from exc
        if (
            output["abstract_work_id"] != item.abstract_work_id
            or output["corpus_sha256"] != payload["corpus_sha256"]
            or output["document_count"] != len(item.windows)
        ):
            raise EbmNlpP1FormalControllerError(
                "HippoRAG output binding drifted"
            )
        rows = output.get("rows")
        if not isinstance(rows, list) or len(rows) != len(core.ROLE_ORDER):
            raise EbmNlpP1FormalControllerError(
                "HippoRAG role output population drifted"
            )
        for ordinal, role in enumerate(core.ROLE_ORDER):
            row = rows[ordinal]
            if (
                not isinstance(row, Mapping)
                or row.get("query_ordinal") != ordinal
                or row.get("role") != role
                or row.get("work_id") != item.role_work_ids[role]
            ):
                raise EbmNlpP1FormalControllerError(
                    "HippoRAG role output binding drifted before seal"
                )
        validated[item.abstract_work_id] = output
    return validated


def _action_payload(action: core.RecipeAction) -> dict[str, object]:
    return {
        "recipe_id": action.recipe_id,
        "window_ordinals": list(action.window_ordinals),
        "window_ids": list(action.window_ids),
        "behavior_sha256": action.behavior_sha256,
    }


def _seal_candidate_archive(
    path: Path,
    *,
    block: str,
    items: Sequence[PreparedAbstract],
    e1: core.E1DeepSetsModel | None,
    hippo_outputs: Mapping[str, Mapping[str, object]] | None,
) -> SealedActionArchive:
    rows: list[dict[str, object]] = []
    for item in items:
        role_rows: list[dict[str, object]] = []
        for role in core.ROLE_ORDER:
            slate = item.slates[role]
            row: dict[str, object] = {
                "role": role,
                "work_id": item.role_work_ids[role],
                "candidate_actions": [
                    _action_payload(action) for action in slate.actions
                ],
                "E0": _action_payload(core.select_e0(slate)),
                "RAW": _action_payload(core.raw_probe_ranking(slate)),
            }
            if e1 is not None:
                row["E1"] = _action_payload(core.select_e1(e1, slate))
            if hippo_outputs is not None:
                output = hippo_outputs[item.abstract_work_id]
                hippo_row = output["rows"][core.ROLE_ORDER.index(role)]
                row["HippoRAG"] = {
                    "rank_window_ordinals": list(
                        hippo_row["rank_window_ordinals"]
                    ),
                    "work_id": hippo_row["work_id"],
                }
            role_rows.append(row)
        rows.append(
            {
                "abstract_work_id": item.abstract_work_id,
                "pmid": item.pmid,
                "roles": role_rows,
            }
        )
    body: dict[str, object] = {
        "schema": ACTION_ARCHIVE_SCHEMA,
        "study_id": STUDY_ID,
        "block": block,
        "abstract_count": len(items),
        "roles_per_abstract": len(core.ROLE_ORDER),
        "items": rows,
    }
    body["archive_sha256"] = _stable_hash(body)
    file_sha256 = _write_once(path, body)
    observed, archive_sha256 = _read_verified_private_archive(
        path, expected_file_sha256=file_sha256
    )
    if observed != body:
        raise EbmNlpP1FormalControllerError(
            "sealed action archive read-back differs from submission"
        )
    return SealedActionArchive(
        block=block,
        path=path,
        file_sha256=file_sha256,
        archive_sha256=archive_sha256,
        payload=observed,
    )


def _sealed_role_row(
    archive: SealedActionArchive,
    *,
    items: Sequence[PreparedAbstract],
    item_ordinal: int,
    role: str,
) -> Mapping[str, object]:
    if (
        archive.block != archive.payload.get("block")
        or archive.payload.get("schema") != ACTION_ARCHIVE_SCHEMA
        or archive.payload.get("study_id") != STUDY_ID
        or archive.payload.get("abstract_count") != len(items)
        or archive.payload.get("roles_per_abstract")
        != len(core.ROLE_ORDER)
    ):
        raise EbmNlpP1FormalControllerError(
            "sealed action archive envelope drifted"
        )
    raw_items = archive.payload.get("items")
    if not isinstance(raw_items, list) or len(raw_items) != len(items):
        raise EbmNlpP1FormalControllerError(
            "sealed action archive item population drifted"
        )
    if not 0 <= item_ordinal < len(items):
        raise EbmNlpP1FormalControllerError(
            "sealed action archive item ordinal drifted"
        )
    item = items[item_ordinal]
    raw_item = raw_items[item_ordinal]
    if (
        not isinstance(raw_item, Mapping)
        or raw_item.get("abstract_work_id") != item.abstract_work_id
        or raw_item.get("pmid") != item.pmid
    ):
        raise EbmNlpP1FormalControllerError(
            "sealed action archive item binding drifted"
        )
    role_rows = raw_item.get("roles")
    role_ordinal = core.ROLE_ORDER.index(role)
    if (
        not isinstance(role_rows, list)
        or len(role_rows) != len(core.ROLE_ORDER)
        or not isinstance(role_rows[role_ordinal], Mapping)
    ):
        raise EbmNlpP1FormalControllerError(
            "sealed action archive role population drifted"
        )
    row = role_rows[role_ordinal]
    if (
        row.get("role") != role
        or row.get("work_id") != item.role_work_ids[role]
        or row.get("candidate_actions")
        != [
            _action_payload(action)
            for action in item.slates[role].actions
        ]
    ):
        raise EbmNlpP1FormalControllerError(
            "sealed action archive role binding drifted"
        )
    return row


def _sealed_ranking(
    row: Mapping[str, object],
    *,
    arm: str,
    window_count: int,
) -> tuple[int, ...]:
    raw_action = row.get(arm)
    ranking_key = (
        "rank_window_ordinals"
        if arm == "HippoRAG"
        else "window_ordinals"
    )
    if not isinstance(raw_action, Mapping):
        raise EbmNlpP1FormalControllerError(
            f"sealed {arm} action is unavailable"
        )
    raw_ranking = raw_action.get(ranking_key)
    if (
        not isinstance(raw_ranking, list)
        or not raw_ranking
        or any(type(value) is not int for value in raw_ranking)
        or len(set(raw_ranking)) != len(raw_ranking)
        or any(
            value < 0 or value >= window_count
            for value in raw_ranking
        )
    ):
        raise EbmNlpP1FormalControllerError(
            f"sealed {arm} ranking drifted"
        )
    expected_length = (
        window_count
        if arm == "HippoRAG"
        else min(core.TOP_K, window_count)
    )
    if len(raw_ranking) != expected_length:
        raise EbmNlpP1FormalControllerError(
            f"sealed {arm} ranking length drifted"
        )
    return tuple(raw_ranking[: core.TOP_K])


def _open_stage_labels(
    acquisition: source.AcquisitionResult,
    *,
    block: str,
    promotion: bool = False,
) -> Mapping[str, Mapping[str, tuple[int, ...]]]:
    return source.open_labels_for_stage(
        acquisition,
        stage=block,
        authorization=source.LabelOpenAuthorization(
            stage=block,
            source_sha256=acquisition.inventory.archive_sha256,
            assignment_sha256=acquisition.assignment.assignment_sha256,
            prerequisites_sealed=True,
            promotion_authorized=promotion,
        ),
    )


def fit_A_form_evaluator(
    items: Sequence[PreparedAbstract],
    labels: Mapping[str, Mapping[str, tuple[int, ...]]],
    *,
    sealed_archive: SealedActionArchive,
) -> tuple[core.E1DeepSetsModel, int]:
    training_slates: list[core.RecipeSlate] = []
    standardization_slates: list[core.RecipeSlate] = []
    utility_slates: list[tuple[Fraction, ...]] = []
    undefined = 0
    for item_ordinal, item in enumerate(items):
        for role in core.ROLE_ORDER:
            slate = item.slates[role]
            standardization_slates.append(slate)
            sealed_row = _sealed_role_row(
                sealed_archive,
                items=items,
                item_ordinal=item_ordinal,
                role=role,
            )
            positives = tuple(
                index
                for index, value in enumerate(
                    labels[item.pmid][ROLE_TO_SOURCE[role]]
                )
                if value != 0
            )
            if not positives:
                undefined += 1
                continue
            raw_candidates = sealed_row.get("candidate_actions")
            if (
                not isinstance(raw_candidates, list)
                or len(raw_candidates) != len(core.RECIPE_IDS)
            ):
                raise EbmNlpP1FormalControllerError(
                    "sealed A_form candidate population drifted"
                )
            utilities: list[Fraction] = []
            for recipe_id, raw_action in zip(
                core.RECIPE_IDS, raw_candidates
            ):
                if (
                    not isinstance(raw_action, Mapping)
                    or raw_action.get("recipe_id") != recipe_id
                ):
                    raise EbmNlpP1FormalControllerError(
                        "sealed A_form recipe order drifted"
                    )
                ranking = _sealed_ranking(
                    {"candidate": raw_action},
                    arm="candidate",
                    window_count=len(item.windows),
                )
                score = core.score_ranked_token_coverage(
                    windows=item.windows,
                    ranking=ranking,
                    positive_token_positions=positives,
                )
                assert score.primary_utility is not None
                utilities.append(score.primary_utility)
            training_slates.append(slate)
            utility_slates.append(tuple(utilities))
    if not training_slates:
        raise EbmNlpP1FormalControllerError(
            "A_form has no defined role utility for E1 fit"
        )
    return (
        core.fit_e1_deepsets(
            training_slates,
            utility_slates,
            standardization_slates=standardization_slates,
        ),
        undefined,
    )


def score_stage(
    items: Sequence[PreparedAbstract],
    labels: Mapping[str, Mapping[str, tuple[int, ...]]],
    *,
    sealed_archive: SealedActionArchive,
) -> StageScores:
    arms = ("E0", "E1", "RAW", "HippoRAG")
    utility_rows: dict[str, list[Mapping[str, Fraction | None]]] = {
        arm: [] for arm in arms
    }
    coverage_rows: dict[str, list[Mapping[str, Fraction | None]]] = {
        arm: [] for arm in arms
    }
    complete_rows: dict[str, list[Mapping[str, int | None]]] = {
        arm: [] for arm in arms
    }
    recipe_counts = {
        arm: {recipe_id: 0 for recipe_id in core.RECIPE_IDS}
        for arm in ("E0", "E1", "RAW")
    }
    outside_raw = {
        arm: {
            "selected_window_count": 0,
            "incremental_positive_token_count": 0,
            "defined_positive_token_count": 0,
            "defined_role_query_count": 0,
        }
        for arm in ("E0", "E1")
    }
    zero_positive = 0
    for item_ordinal, item in enumerate(items):
        per_arm_utility: dict[str, dict[str, Fraction | None]] = {
            arm: {} for arm in arms
        }
        per_arm_coverage: dict[str, dict[str, Fraction | None]] = {
            arm: {} for arm in arms
        }
        per_arm_complete: dict[str, dict[str, int | None]] = {
            arm: {} for arm in arms
        }
        for role in core.ROLE_ORDER:
            sealed_row = _sealed_role_row(
                sealed_archive,
                items=items,
                item_ordinal=item_ordinal,
                role=role,
            )
            positives = tuple(
                index
                for index, value in enumerate(
                    labels[item.pmid][ROLE_TO_SOURCE[role]]
                )
                if value != 0
            )
            rankings = {
                arm: _sealed_ranking(
                    sealed_row,
                    arm=arm,
                    window_count=len(item.windows),
                )
                for arm in arms
            }
            for arm in ("E0", "E1", "RAW"):
                action = sealed_row.get(arm)
                recipe_id = (
                    action.get("recipe_id")
                    if isinstance(action, Mapping)
                    else None
                )
                if recipe_id not in core.RECIPE_IDS:
                    raise EbmNlpP1FormalControllerError(
                        f"sealed {arm} recipe identity drifted"
                    )
                recipe_counts[arm][str(recipe_id)] += 1
            for arm, ranking in rankings.items():
                score = core.score_ranked_token_coverage(
                    windows=item.windows,
                    ranking=ranking,
                    positive_token_positions=positives,
                )
                per_arm_utility[arm][role] = score.primary_utility
                per_arm_coverage[arm][role] = (
                    score.undiscounted_coverage_at_5
                )
                per_arm_complete[arm][role] = score.complete_at_5
            if not positives:
                zero_positive += 1
            else:
                raw_covered = {
                    position
                    for ordinal in rankings["RAW"]
                    for position in positives
                    if (
                        item.windows[ordinal].start
                        <= position
                        < item.windows[ordinal].end
                    )
                }
                raw_set = set(rankings["RAW"])
                for arm in ("E0", "E1"):
                    outside = tuple(
                        ordinal
                        for ordinal in rankings[arm]
                        if ordinal not in raw_set
                    )
                    outside_covered = {
                        position
                        for ordinal in outside
                        for position in positives
                        if (
                            item.windows[ordinal].start
                            <= position
                            < item.windows[ordinal].end
                        )
                    }
                    outside_raw[arm]["selected_window_count"] += len(
                        outside
                    )
                    outside_raw[arm][
                        "incremental_positive_token_count"
                    ] += len(outside_covered - raw_covered)
                    outside_raw[arm][
                        "defined_positive_token_count"
                    ] += len(positives)
                    outside_raw[arm][
                        "defined_role_query_count"
                    ] += 1
        for arm in arms:
            utility_rows[arm].append(dict(per_arm_utility[arm]))
            coverage_rows[arm].append(dict(per_arm_coverage[arm]))
            complete_rows[arm].append(dict(per_arm_complete[arm]))
    return StageScores(
        arm_role_utilities={
            arm: tuple(values)
            for arm, values in utility_rows.items()
        },
        arm_role_coverages={
            arm: tuple(values)
            for arm, values in coverage_rows.items()
        },
        arm_role_completes={
            arm: tuple(values)
            for arm, values in complete_rows.items()
        },
        recipe_selection_counts={
            arm: dict(values) for arm, values in recipe_counts.items()
        },
        typed_outside_raw={
            arm: dict(values) for arm, values in outside_raw.items()
        },
        zero_positive_role_count=zero_positive,
    )


def _fraction(value: Fraction | None) -> Mapping[str, int] | None:
    if value is None:
        return None
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
    }


def _comparison_payload(
    comparison: core.PairedAbstractComparison,
) -> dict[str, object]:
    return {
        "mean_delta": _fraction(comparison.mean_delta),
        "zero_defined_abstract_count": (
            comparison.zero_defined_abstract_count
        ),
        "gain_count": comparison.sign_test.gains,
        "harm_count": comparison.sign_test.harms,
        "tie_count": comparison.sign_test.ties,
        "nonzero_count": comparison.sign_test.nonzero_count,
        "one_sided_exact_sign_p": _fraction(
            comparison.sign_test.one_sided_p
        ),
        "family_deltas": {
            role: _fraction(comparison.family_deltas[role])
            for role in core.ROLE_ORDER
        },
    }


def _secondary_payload(scores: StageScores) -> dict[str, object]:
    coverage: dict[str, object] = {}
    complete: dict[str, object] = {}
    for arm in ("E0", "E1", "RAW", "HippoRAG"):
        coverage[arm] = {
            role: _fraction(value)
            for role, value in core.family_aggregate(
                scores.arm_role_coverages[arm]
            ).items()
        }
        role_complete: dict[str, Mapping[str, int] | None] = {}
        for role in core.ROLE_ORDER:
            values = [
                row[role]
                for row in scores.arm_role_completes[arm]
                if row[role] is not None
            ]
            mean = (
                None
                if not values
                else Fraction(
                    sum(int(value) for value in values),
                    len(values),
                )
            )
            role_complete[role] = _fraction(mean)
        complete[arm] = role_complete
    outside: dict[str, object] = {}
    for arm in ("E0", "E1"):
        row = scores.typed_outside_raw[arm]
        denominator = row["defined_positive_token_count"]
        outside[arm] = {
            **dict(row),
            "incremental_positive_token_coverage": _fraction(
                None
                if denominator == 0
                else Fraction(
                    row["incremental_positive_token_count"],
                    denominator,
                )
            ),
        }
    return {
        "undiscounted_coverage_at_5_family_means": coverage,
        "complete_at_5_family_rates": complete,
        "recipe_selection_counts": {
            arm: dict(values)
            for arm, values in scores.recipe_selection_counts.items()
        },
        "typed_selected_outside_RAW_top5": outside,
    }


def _comparison_success(
    comparison: core.PairedAbstractComparison,
    *,
    require_all_families: bool,
) -> bool:
    if (
        comparison.mean_delta <= 0
        or comparison.sign_test.one_sided_p > Fraction(1, 10)
    ):
        return False
    if require_all_families:
        return all(
            comparison.family_deltas[role] is not None
            and comparison.family_deltas[role] > 0
            for role in core.ROLE_ORDER
        )
    return True


def _compare(scores: StageScores, left: str, right: str) -> core.PairedAbstractComparison:
    return core.compare_abstract_arms(
        scores.arm_role_utilities[left],
        scores.arm_role_utilities[right],
    )


def _model_payload(model: core.E1DeepSetsModel) -> dict[str, object]:
    return {
        "schema": f"{VERSION}_private_E1_model_v1",
        "study_id": STUDY_ID,
        "model": asdict(model),
    }


def _attach_runtime_receipts(
    safe: dict[str, object],
    *,
    embedder: EmbeddingExecutor,
    hippo_launcher: HippoBatchExecutor,
    formal_scope: bool,
    expected_hippo_workers: int,
) -> None:
    embedding_hash, embedding = _executor_runtime_receipt(
        embedder,
        kind="embedding",
        formal_scope=formal_scope,
    )
    hippo_hash, hippo_receipt = _executor_runtime_receipt(
        hippo_launcher,
        kind="hipporag",
        formal_scope=formal_scope,
        expected_hippo_workers=expected_hippo_workers,
    )
    safe.update(
        {
            "embedding_safe_runtime_receipt_sha256": embedding_hash,
            "embedding_call_count": embedding["call_count"],
            "embedding_encoded_text_count": (
                embedding["encoded_text_count"]
            ),
            "HippoRAG_safe_runtime_receipt_sha256": hippo_hash,
            "HippoRAG_worker_attempt_count": (
                hippo_receipt["worker_attempt_count"]
            ),
            "HippoRAG_worker_completed_count": (
                hippo_receipt["worker_completed_count"]
            ),
            "HippoRAG_attempted_network_syscall_count": (
                hippo_receipt["attempted_network_syscall_count"]
            ),
            "HippoRAG_denied_network_syscall_count": (
                hippo_receipt["denied_network_syscall_count"]
            ),
            "HippoRAG_external_network_call_count": 0,
        }
    )


def _run_formal_study_once(
    *,
    archive_path: Path,
    work_root: Path,
    contract: source.QualificationContract,
    embedder: EmbeddingExecutor,
    hippo_launcher: HippoBatchExecutor,
    execution_binding_sha256: str,
    execution_scope: str,
) -> dict[str, object]:
    formal_scope = execution_scope == "formal_exact_EBM_NLP_source_epoch"
    _ensure_private_directory(work_root, fresh=True)
    _write_once(
        work_root / "formal.attempt_consumed.json",
        {
            "schema": f"{VERSION}_attempt_consumed_v1",
            "execution_binding_sha256": execution_binding_sha256,
            "execution_scope": execution_scope,
            "study_id": STUDY_ID,
            "status": "consumed_before_source_open_no_retry",
        },
    )
    acquisition = source.acquire_once(
        archive_path=archive_path,
        private_root=work_root / "source_private",
        contract=contract,
    )
    query_embeddings = _query_embeddings(embedder)
    probes, probe_freeze_hash = fit_G_form_probes(
        acquisition, embedder=embedder
    )
    probe_state = _probe_state_payload(probes)
    if _stable_hash(probe_state) != probe_freeze_hash:
        raise EbmNlpP1FormalControllerError(
            "G_form probe state changed before sealing"
        )
    probe_state_file_hash = _write_once(
        work_root / "G_form.probes.private.json",
        probe_state,
    )

    a_form = _prepare_stage_items(
        acquisition,
        block="A_form",
        probes=probes,
        embedder=embedder,
        query_embeddings=query_embeddings,
    )
    a_form_archive = _seal_candidate_archive(
        work_root / "A_form.actions.private.json",
        block="A_form",
        items=a_form,
        e1=None,
        hippo_outputs=None,
    )
    a_form_labels = _open_stage_labels(
        acquisition, block="A_form"
    )
    e1, a_form_undefined = fit_A_form_evaluator(
        a_form,
        a_form_labels,
        sealed_archive=a_form_archive,
    )
    e1_model_file_hash = _write_once(
        work_root / "E1.model.private.json",
        _model_payload(e1),
    )

    _assert_probe_state(probes, probe_freeze_hash)
    f_search = _prepare_stage_items(
        acquisition,
        block="F_search",
        probes=probes,
        embedder=embedder,
        query_embeddings=query_embeddings,
    )
    f_archive = _seal_candidate_archive(
        work_root / "F_search.actions.private.json",
        block="F_search",
        items=f_search,
        e1=e1,
        hippo_outputs=None,
    )
    f_equal = 0
    for item_ordinal, item in enumerate(f_search):
        for role in core.ROLE_ORDER:
            row = _sealed_role_row(
                f_archive,
                items=f_search,
                item_ordinal=item_ordinal,
                role=role,
            )
            e0 = row.get("E0")
            e1_row = row.get("E1")
            if not isinstance(e0, Mapping) or not isinstance(
                e1_row, Mapping
            ):
                raise EbmNlpP1FormalControllerError(
                    "sealed F_search evaluator action is unavailable"
                )
            f_equal += int(
                e0.get("behavior_sha256")
                == e1_row.get("behavior_sha256")
            )

    _assert_probe_state(probes, probe_freeze_hash)
    a_hold = _prepare_stage_items(
        acquisition,
        block="A_hold",
        probes=probes,
        embedder=embedder,
        query_embeddings=query_embeddings,
    )
    a_hold_hippo = _validated_hippo_outputs(
        a_hold, hippo_launcher
    )
    a_hold_archive = _seal_candidate_archive(
        work_root / "A_hold.actions.private.json",
        block="A_hold",
        items=a_hold,
        e1=e1,
        hippo_outputs=a_hold_hippo,
    )
    a_hold_labels = _open_stage_labels(
        acquisition, block="A_hold"
    )
    a_hold_scores = score_stage(
        a_hold,
        a_hold_labels,
        sealed_archive=a_hold_archive,
    )
    promotion_comparison = _compare(a_hold_scores, "E1", "E0")
    raw_comparison = _compare(a_hold_scores, "E1", "RAW")
    hippo_comparison = _compare(
        a_hold_scores, "E1", "HippoRAG"
    )
    promotion = _comparison_success(
        promotion_comparison, require_all_families=False
    )
    a_hold_reality = _comparison_success(
        raw_comparison, require_all_families=True
    ) and _comparison_success(
        hippo_comparison, require_all_families=True
    )

    safe: dict[str, object] = {
        "schema": SAFE_TERMINAL_SCHEMA,
        "study_id": STUDY_ID,
        "execution_binding_sha256": execution_binding_sha256,
        "execution_scope": execution_scope,
        "status": (
            "A_hold_complete_promotion_authorized"
            if promotion
            else (
                "complete_valid_nonpromotion_"
                "M_action_model_view_and_gold_unopened"
            )
        ),
        "source_archive_sha256": acquisition.inventory.archive_sha256,
        "source_assignment_sha256": (
            acquisition.assignment.assignment_sha256
        ),
        "block_counts": {
            block: len(acquisition.assignment.pmids(block))
            for block in source.BLOCK_ORDER
        },
        "probe_freeze_sha256": probe_freeze_hash,
        "G_form_probe_state_file_sha256": probe_state_file_hash,
        "A_form_action_archive_file_sha256": (
            a_form_archive.file_sha256
        ),
        "A_form_undefined_role_count": a_form_undefined,
        "E1_model_file_sha256": e1_model_file_hash,
        "F_search_action_archive_file_sha256": f_archive.file_sha256,
        "F_search_E0_E1_equal_behavior_count": f_equal,
        "F_search_total_role_query_count": len(f_search)
        * len(core.ROLE_ORDER),
        "A_hold_action_archive_file_sha256": (
            a_hold_archive.file_sha256
        ),
        "A_hold_zero_positive_role_count": (
            a_hold_scores.zero_positive_role_count
        ),
        "A_hold_promotion": promotion,
        "A_hold_reality_primary": a_hold_reality,
        "A_hold_E1_minus_E0": _comparison_payload(
            promotion_comparison
        ),
        "A_hold_E1_minus_RAW": _comparison_payload(raw_comparison),
        "A_hold_E1_minus_HippoRAG": _comparison_payload(
            hippo_comparison
        ),
        "A_hold_secondary": _secondary_payload(a_hold_scores),
        "M_search_action_or_model_view_opened": False,
        "M_search_gold_opened": False,
        "M_search_documents_present_only_in_private_acquisition": True,
        "online_or_API_evaluator_call_count": 0,
        "formal_retry_replay_resample_provider_model_secret_or_gate_change_count": 0,
        "raw_PMID_token_text_label_or_action_output_count": 0,
    }
    if not promotion:
        _assert_probe_state(probes, probe_freeze_hash)
        _attach_runtime_receipts(
            safe,
            embedder=embedder,
            hippo_launcher=hippo_launcher,
            formal_scope=formal_scope,
            expected_hippo_workers=len(a_hold),
        )
        safe["joint_total_goal"] = False
        safe["terminal_sha256"] = _stable_hash(safe)
        return safe

    _assert_probe_state(probes, probe_freeze_hash)
    m_search = _prepare_stage_items(
        acquisition,
        block="M_search",
        probes=probes,
        embedder=embedder,
        query_embeddings=query_embeddings,
    )
    m_hippo = _validated_hippo_outputs(m_search, hippo_launcher)
    m_archive = _seal_candidate_archive(
        work_root / "M_search.actions.private.json",
        block="M_search",
        items=m_search,
        e1=e1,
        hippo_outputs=m_hippo,
    )
    m_labels = _open_stage_labels(
        acquisition, block="M_search", promotion=True
    )
    m_scores = score_stage(
        m_search,
        m_labels,
        sealed_archive=m_archive,
    )
    m_l5 = _compare(m_scores, "E1", "E0")
    m_raw = _compare(m_scores, "E1", "RAW")
    m_hippo_comparison = _compare(m_scores, "E1", "HippoRAG")
    l5_success = _comparison_success(
        m_l5, require_all_families=False
    )
    m_reality = _comparison_success(
        m_raw, require_all_families=True
    ) and _comparison_success(
        m_hippo_comparison, require_all_families=True
    )
    joint = a_hold_reality and promotion and l5_success and m_reality
    safe.update(
        {
            "status": "complete_promotion_and_M_evaluated",
            "M_search_action_or_model_view_opened": True,
            "M_search_gold_opened": True,
            "M_search_action_archive_file_sha256": (
                m_archive.file_sha256
            ),
            "M_search_zero_positive_role_count": (
                m_scores.zero_positive_role_count
            ),
            "M_search_L5_success": l5_success,
            "M_search_reality_replication": m_reality,
            "M_search_E1_minus_E0": _comparison_payload(m_l5),
            "M_search_E1_minus_RAW": _comparison_payload(m_raw),
            "M_search_E1_minus_HippoRAG": _comparison_payload(
                m_hippo_comparison
            ),
            "M_search_secondary": _secondary_payload(m_scores),
            "joint_total_goal": joint,
        }
    )
    _assert_probe_state(probes, probe_freeze_hash)
    _attach_runtime_receipts(
        safe,
        embedder=embedder,
        hippo_launcher=hippo_launcher,
        formal_scope=formal_scope,
        expected_hippo_workers=len(a_hold) + len(m_search),
    )
    safe["terminal_sha256"] = _stable_hash(safe)
    return safe


def _run_study_with_terminal(
    *,
    archive_path: Path,
    work_root: Path,
    contract: source.QualificationContract,
    embedder: EmbeddingExecutor,
    hippo_launcher: HippoBatchExecutor,
    execution_binding_sha256: str,
    execution_scope: str,
) -> dict[str, object]:
    """Execute one source epoch and always attempt one safe terminal."""

    work_root = Path(work_root)
    try:
        terminal = _run_formal_study_once(
            archive_path=Path(archive_path),
            work_root=work_root,
            contract=contract,
            embedder=embedder,
            hippo_launcher=hippo_launcher,
            execution_binding_sha256=execution_binding_sha256,
            execution_scope=execution_scope,
        )
    except BaseException as exc:
        if not work_root.exists():
            raise
        terminal = {
            "schema": SAFE_TERMINAL_SCHEMA,
            "study_id": STUDY_ID,
            "execution_binding_sha256": execution_binding_sha256,
            "execution_scope": execution_scope,
            "status": "terminal_implementation_or_runtime_invalid",
            "efficacy": "unknown",
            "primary_evaluated": False,
            "replay_permitted": False,
            "error_type": type(exc).__name__,
            "error_message_sha256": hashlib.sha256(
                str(exc).encode("utf-8", errors="replace")
            ).hexdigest(),
            "online_or_API_evaluator_call_count": 0,
            "raw_PMID_token_text_label_or_action_output_count": 0,
        }
        terminal["terminal_sha256"] = _stable_hash(terminal)
    _write_once(work_root / "formal_terminal.json", terminal)
    return terminal


def run_formal_study(
    *,
    archive_path: Path,
    work_root: Path,
    contract: source.QualificationContract,
    embedder: EmbeddingExecutor,
    hippo_launcher: HippoBatchExecutor,
    execution_binding: FormalExecutionBinding,
) -> dict[str, object]:
    """Execute the unique official EBM-NLP source epoch.

    Synthetic contract tests use the private harness above.  This public
    formal surface rejects every alternate source contract and binds the
    pre-source implementation, runtime fingerprint, canary, assets, and
    execution configuration into the terminal.
    """

    if contract != source.FORMAL_CONTRACT:
        raise EbmNlpP1FormalControllerError(
            "public formal entrypoint requires the exact frozen source contract"
        )
    binding_sha256 = _validate_formal_execution_binding(
        execution_binding
    )
    return _run_study_with_terminal(
        archive_path=archive_path,
        work_root=work_root,
        contract=contract,
        embedder=embedder,
        hippo_launcher=hippo_launcher,
        execution_binding_sha256=binding_sha256,
        execution_scope="formal_exact_EBM_NLP_source_epoch",
    )


__all__ = [
    "EmbeddingExecutor",
    "EbmNlpP1FormalControllerError",
    "FormalExecutionBinding",
    "HippoBatchExecutor",
    "PreparedAbstract",
    "PreparedStage",
    "RawAbstract",
    "SAFE_TERMINAL_SCHEMA",
    "SealedActionArchive",
    "StageScores",
    "fit_A_form_evaluator",
    "fit_G_form_probes",
    "run_formal_study",
    "score_stage",
]
