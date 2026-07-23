"""One-shot, offline-scored lifecycle controller for formal BIRCO P1.

The controller is deliberately source blind.  It reads only the selector's
label-free action packs.  Numeric qrels can enter through the injected
``QrelOpener`` *after* a complete stage action archive has been atomically
linked into place, fsynced, made read-only, and re-verified.  ``F_search`` has
no qrel-opening code path.

Three independent executor pools are used where the frozen study needs them:
Agent semantic calls, RAW semantic calls, and official HippoRAG.  Agent and RAW
share only a total API semaphore; they never share a worker pool.  Planner
terminals are consumed before their matrix calls are constructed.  RAW and
HippoRAG are nevertheless bulk-submitted while planners are in flight.

Every external action has a durable, exclusive attempt claim.  Recovery may
reuse only a claim plus a complete, canonical, independently revalidated
terminal.  A claim without such a terminal is terminally consumed and is
never retried, replayed, resampled, or switched to another provider.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import threading
from typing import Any, Callable, Mapping, Protocol, Sequence

from replication_runtime.birco_gpt54_semantic_v1 import contract as semantic
from replication_runtime.birco_official_hipporag_v1 import contract as hippo

from . import birco_p1_action_integration_v1 as integration
from . import birco_p1_private_selection_v1 as selection
from . import birco_p1_typed_constraint_e4_core_v1 as core


VERSION = "birco_p1_formal_controller_v1"
STUDY_ID = selection.STUDY_ID
IMPLEMENTATION_FREEZE_SCHEMA = "birco_p1_implementation_freeze_v1"
EXECUTION_FREEZE_SCHEMA = f"{VERSION}_execution_freeze_v1"
STAGE_ARCHIVE_SCHEMA = f"{VERSION}_stage_action_archive_v1"
ATTEMPT_CLAIM_SCHEMA = f"{VERSION}_external_attempt_claim_v1"
HIPPO_TERMINAL_SCHEMA = f"{VERSION}_hipporag_terminal_v1"
FAILURE_SCHEMA = f"{VERSION}_terminal_failure_v1"
FINAL_SCHEMA = f"{VERSION}_final_receipt_v1"

BLOCK_ORDER = selection.BLOCK_ORDER
BLOCK_ITEM_COUNT = len(selection.FAMILIES) * selection.PER_FAMILY_QUOTA
BLOCK_ARMS: Mapping[str, tuple[str, ...]] = {
    "A_form": ("Agent",),
    "F_search": ("Agent",),
    "A_hold": ("Agent", "RAW", "HippoRAG"),
    "M_search": ("Agent",),
}

# These are hard safety ceilings, not targets.  The execution freeze binds the
# lower host-specific values actually used by a formal run.
MAX_AGENT_API_WORKERS = 64
MAX_RAW_API_WORKERS = 64
MAX_TOTAL_API_CALLS = 64
MAX_HIPPORAG_WORKERS = 4

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_RELATIVE = re.compile(r"[A-Za-z0-9._/-]+\Z")


class BircoP1FormalControllerError(RuntimeError):
    """A frozen formal lifecycle, custody, or consumed-attempt invariant failed."""


class SemanticExecutor(Protocol):
    """One already-frozen semantic transport, called exactly once per claim."""

    def __call__(
        self, *, mode: str, payload: Mapping[str, object]
    ) -> Mapping[str, object]: ...


class HippoExecutor(Protocol):
    """One already-frozen official-core adapter invocation."""

    def __call__(
        self,
        *,
        payload: Mapping[str, object],
        runtime_policy: Mapping[str, object],
    ) -> Mapping[str, object]: ...


class QrelOpener(Protocol):
    """Trusted late-release boundary; it must perform only local offline I/O."""

    def __call__(
        self,
        *,
        block: str,
        action_archive_sha256s: Sequence[str],
        promotion_sha256: str | None,
    ) -> Mapping[str, object]: ...


def _canonical_bytes(value: object, *, newline: bool = True) -> bytes:
    try:
        raw = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise BircoP1FormalControllerError("value is not canonical JSON") from exc
    return raw + (b"\n" if newline else b"")


def stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value, newline=False)).hexdigest()


def _self_hashed(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in body:
        raise BircoP1FormalControllerError("self-hash field was supplied twice")
    frozen = dict(body)
    frozen[field] = stable_hash(frozen)
    return frozen


def _verify_self(value: Mapping[str, Any], field: str) -> str:
    body = dict(value)
    claimed = body.pop(field, None)
    if not isinstance(claimed, str) or _SHA256.fullmatch(claimed) is None:
        raise BircoP1FormalControllerError(f"{field} is not SHA-256")
    if stable_hash(body) != claimed:
        raise BircoP1FormalControllerError(f"{field} self hash drifted")
    return claimed


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise BircoP1FormalControllerError(f"{field} is not SHA-256")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _safe_relative(value: object, field: str) -> Path:
    if not isinstance(value, str) or _SAFE_RELATIVE.fullmatch(value) is None:
        raise BircoP1FormalControllerError(f"{field} is not a safe relative path")
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise BircoP1FormalControllerError(f"{field} escaped the project root")
    return path


def _bound_project_path(project: Path, relative: Path, field: str) -> Path:
    candidate = project / relative
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(project)
    except (OSError, ValueError) as exc:
        raise BircoP1FormalControllerError(f"{field} escaped the project root") from exc
    if resolved != candidate.absolute():
        raise BircoP1FormalControllerError(f"{field} traverses a symlink")
    return candidate


def _regular_file(path: Path, field: str, *, mode: int | None = None) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as exc:
        raise BircoP1FormalControllerError(f"{field} is unavailable") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise BircoP1FormalControllerError(f"{field} is not a regular file")
    if mode is not None and stat.S_IMODE(info.st_mode) != mode:
        raise BircoP1FormalControllerError(f"{field} mode drifted")
    return info


def _read_canonical(path: Path, field: str, *, mode: int | None = None) -> dict[str, Any]:
    before = _regular_file(path, field, mode=mode)
    raw = path.read_bytes()
    after = _regular_file(path, field, mode=mode)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise BircoP1FormalControllerError(f"{field} changed while read")
    try:
        value = json.loads(raw.decode("ascii"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise BircoP1FormalControllerError(f"{field} is invalid JSON") from exc
    if not isinstance(value, dict) or raw != _canonical_bytes(value):
        raise BircoP1FormalControllerError(f"{field} is not canonical")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_private_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise BircoP1FormalControllerError("controller directory is unsafe")
    os.chmod(path, 0o700)


def _atomic_write_once(
    path: Path,
    value: Mapping[str, Any],
    *,
    final_mode: int = 0o600,
) -> str:
    """Atomically publish one canonical file without an overwrite primitive."""

    _ensure_private_directory(path.parent)
    if path.exists() or path.is_symlink():
        raise BircoP1FormalControllerError(f"{path.name} already exists")
    raw = _canonical_bytes(value)
    temporary = path.parent / (
        f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
    )
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, path, follow_symlinks=False)
        os.chmod(path, final_mode, follow_symlinks=False)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass
    return hashlib.sha256(raw).hexdigest()


def _seal_read_only(path: Path, value: Mapping[str, Any], hash_field: str) -> str:
    semantic_sha = _verify_self(value, hash_field)
    _atomic_write_once(path, value, final_mode=0o400)
    reread = _read_canonical(path, path.name, mode=0o400)
    if reread != dict(value) or _verify_self(reread, hash_field) != semantic_sha:
        raise BircoP1FormalControllerError("immutable seal postflight drifted")
    return semantic_sha


@dataclass(frozen=True)
class ConcurrencyPolicy:
    agent_api_workers: int = 64
    raw_api_workers: int = 64
    total_api_call_cap: int = 64
    hipporag_workers: int = 4

    def __post_init__(self) -> None:
        bounds = (
            (self.agent_api_workers, MAX_AGENT_API_WORKERS, "Agent API workers"),
            (self.raw_api_workers, MAX_RAW_API_WORKERS, "RAW API workers"),
            (self.total_api_call_cap, MAX_TOTAL_API_CALLS, "total API cap"),
            (self.hipporag_workers, MAX_HIPPORAG_WORKERS, "HippoRAG workers"),
        )
        for value, maximum, field in bounds:
            if type(value) is not int or not 1 <= value <= maximum:
                raise BircoP1FormalControllerError(f"{field} is outside its frozen cap")

    def payload(self) -> dict[str, int]:
        return {
            "agent_api_workers": self.agent_api_workers,
            "raw_api_workers": self.raw_api_workers,
            "total_api_call_cap": self.total_api_call_cap,
            "hipporag_workers": self.hipporag_workers,
        }


@dataclass(frozen=True)
class PrerunBindings:
    execution_freeze_self_sha256: str
    execution_freeze_file_sha256: str
    implementation_freeze_self_sha256: str
    selection_receipt_self_sha256: str
    provider_identity: Mapping[str, object]
    hipporag_runtime_policy: Mapping[str, object]
    concurrency: ConcurrencyPolicy
    action_packs: Mapping[str, Mapping[str, Any]]
    action_pack_sha256s: Mapping[str, str]


def _verify_binding_file(
    project: Path,
    binding: object,
    *,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(binding, Mapping) or set(binding) != {
        "relative_path",
        "file_sha256",
        "semantic_sha256",
    }:
        raise BircoP1FormalControllerError(f"{label} binding shape drifted")
    relative = _safe_relative(binding.get("relative_path"), f"{label} path")
    path = _bound_project_path(project, relative, label)
    expected_file = _sha256(binding.get("file_sha256"), f"{label} file hash")
    if _file_sha256(path) != expected_file:
        raise BircoP1FormalControllerError(f"{label} file hash drifted")
    return path, _read_canonical(path, label)


def verify_prerun_freezes(
    *,
    project_root: Path,
    execution_freeze_path: Path,
    expected_execution_freeze_file_sha256: str,
    expected_execution_freeze_self_sha256: str,
) -> PrerunBindings:
    """Verify all three trust layers before any formal action claim is created."""

    project = Path(project_root).resolve(strict=True)
    supplied_freeze_path = Path(execution_freeze_path)
    _regular_file(supplied_freeze_path, "execution freeze")
    freeze_path = supplied_freeze_path.resolve(strict=True)
    if freeze_path != supplied_freeze_path.absolute():
        raise BircoP1FormalControllerError("execution freeze traverses a symlink")
    try:
        freeze_path.relative_to(project)
    except ValueError as exc:
        raise BircoP1FormalControllerError("execution freeze is outside project") from exc
    expected_file = _sha256(
        expected_execution_freeze_file_sha256, "expected execution-freeze file hash"
    )
    expected_self = _sha256(
        expected_execution_freeze_self_sha256, "expected execution-freeze self hash"
    )
    if _file_sha256(freeze_path) != expected_file:
        raise BircoP1FormalControllerError("execution freeze file hash drifted")
    execution = _read_canonical(freeze_path, "execution freeze")
    observed_self = _verify_self(execution, "self_sha256")
    if observed_self != expected_self:
        raise BircoP1FormalControllerError("execution freeze authority drifted")
    expected_keys = {
        "schema",
        "version",
        "study_id",
        "implementation_freeze_binding",
        "selection_receipt_binding",
        "action_pack_bindings",
        "provider_identity",
        "hipporag_runtime_policy",
        "concurrency_policy",
        "stage_order",
        "arms_by_block",
        "offline_only_scoring",
        "online_evaluator_call_count",
        "retry_replay_resample_or_provider_switch_count",
        "official_hipporag_commit",
        "self_sha256",
    }
    if (
        set(execution) != expected_keys
        or execution.get("schema") != EXECUTION_FREEZE_SCHEMA
        or execution.get("version") != VERSION
        or execution.get("study_id") != STUDY_ID
        or execution.get("stage_order") != list(BLOCK_ORDER)
        or execution.get("arms_by_block")
        != {block: list(BLOCK_ARMS[block]) for block in BLOCK_ORDER}
        or execution.get("offline_only_scoring") is not True
        or execution.get("online_evaluator_call_count") != 0
        or execution.get("retry_replay_resample_or_provider_switch_count") != 0
        or execution.get("official_hipporag_commit") != hippo.OFFICIAL_HIPPORAG_COMMIT
    ):
        raise BircoP1FormalControllerError("execution freeze policy drifted")

    policy_value = execution.get("concurrency_policy")
    if not isinstance(policy_value, Mapping) or set(policy_value) != {
        "agent_api_workers",
        "raw_api_workers",
        "total_api_call_cap",
        "hipporag_workers",
    }:
        raise BircoP1FormalControllerError("execution concurrency policy drifted")
    concurrency = ConcurrencyPolicy(**dict(policy_value))  # type: ignore[arg-type]

    provider = execution.get("provider_identity")
    if not isinstance(provider, Mapping):
        raise BircoP1FormalControllerError("provider identity is absent")
    try:
        integration._validate_provider(provider)
    except Exception as exc:
        raise BircoP1FormalControllerError("provider identity drifted") from exc
    provider_identity = dict(provider)

    hippo_policy = execution.get("hipporag_runtime_policy")
    expected_hippo_policy_fields = {
        "model_alias_cwd_relative",
        "llm_model_alias",
        "embedding_model_alias",
        "aliases_are_single_relative_components",
        "subprocess_cwd_is_model_alias_cwd",
        "absolute_model_path_argument_count",
        "logical_slot_count",
        "gpu_assignment",
        "maximum_processes_per_gpu",
        "cpu_threads_per_process",
    }
    if not isinstance(hippo_policy, Mapping) or set(hippo_policy) != expected_hippo_policy_fields:
        raise BircoP1FormalControllerError("HippoRAG runtime policy drifted")
    alias_cwd = _safe_relative(
        hippo_policy.get("model_alias_cwd_relative"), "HippoRAG model-alias cwd"
    ).as_posix()
    if (
        alias_cwd in {".", ""}
        or hippo_policy.get("llm_model_alias") != "smollm2"
        or hippo_policy.get("embedding_model_alias") != "minilm"
        or any(
            "/" in str(hippo_policy.get(field))
            or "\\" in str(hippo_policy.get(field))
            or str(hippo_policy.get(field)) in {".", ".."}
            for field in ("llm_model_alias", "embedding_model_alias")
        )
        or hippo_policy.get("aliases_are_single_relative_components") is not True
        or hippo_policy.get("subprocess_cwd_is_model_alias_cwd") is not True
        or hippo_policy.get("absolute_model_path_argument_count") != 0
        or hippo_policy.get("logical_slot_count") != 4
        or hippo_policy.get("gpu_assignment") != ["0", "1", "0", "1"]
        or hippo_policy.get("maximum_processes_per_gpu") != 2
        or hippo_policy.get("cpu_threads_per_process") != 2
    ):
        raise BircoP1FormalControllerError("HippoRAG short-alias policy drifted")
    hipporag_runtime_policy = dict(hippo_policy)

    implementation_path, implementation = _verify_binding_file(
        project,
        execution.get("implementation_freeze_binding"),
        label="implementation freeze",
    )
    implementation_binding = execution["implementation_freeze_binding"]
    implementation_self = _verify_self(implementation, "self_sha256")
    if (
        implementation_binding.get("semantic_sha256") != implementation_self
        or implementation.get("schema") != IMPLEMENTATION_FREEZE_SCHEMA
        or implementation.get("study_id") != STUDY_ID
    ):
        raise BircoP1FormalControllerError("implementation freeze identity drifted")
    rows = implementation.get("implementation_bindings")
    if not isinstance(rows, list) or not rows:
        raise BircoP1FormalControllerError("implementation binding registry is absent")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"relative_path", "sha256"}:
            raise BircoP1FormalControllerError("implementation binding row drifted")
        relative = _safe_relative(row.get("relative_path"), "implementation path")
        text = relative.as_posix()
        if text in seen:
            raise BircoP1FormalControllerError("implementation binding duplicated")
        seen.add(text)
        member_path = _bound_project_path(project, relative, "implementation member")
        _regular_file(member_path, "implementation member")
        if _file_sha256(member_path) != _sha256(
            row.get("sha256"), "implementation member hash"
        ):
            raise BircoP1FormalControllerError("implementation member drifted")
    if implementation_path == freeze_path:
        raise BircoP1FormalControllerError("implementation and execution freezes alias")

    _selection_path, receipt = _verify_binding_file(
        project,
        execution.get("selection_receipt_binding"),
        label="selection receipt",
    )
    selection_binding = execution["selection_receipt_binding"]
    try:
        selection_self = selection.verify_self_hash(receipt, "acquisition_sha256")
    except Exception as exc:
        raise BircoP1FormalControllerError("selection receipt self hash drifted") from exc
    if (
        selection_binding.get("semantic_sha256") != selection_self
        or receipt.get("schema") != f"{selection.VERSION}_public_receipt_v1"
        or receipt.get("study_id") != STUDY_ID
        or receipt.get("status")
        != "private_query_disjoint_four_block_selection_complete"
    ):
        raise BircoP1FormalControllerError("selection receipt identity drifted")

    action_bindings = execution.get("action_pack_bindings")
    if not isinstance(action_bindings, Mapping) or set(action_bindings) != set(BLOCK_ORDER):
        raise BircoP1FormalControllerError("action-pack binding registry drifted")
    packs: dict[str, Mapping[str, Any]] = {}
    pack_hashes: dict[str, str] = {}
    selection_bindings = receipt.get("private_pack_bindings")
    if not isinstance(selection_bindings, Mapping):
        raise BircoP1FormalControllerError("selection private bindings are absent")
    for block in BLOCK_ORDER:
        _path, pack = _verify_binding_file(
            project, action_bindings[block], label=f"{block} action pack"
        )
        try:
            semantic_sha = selection._validate_action_pack(pack, block=block)
        except Exception as exc:
            raise BircoP1FormalControllerError(f"{block} action pack drifted") from exc
        binding = action_bindings[block]
        source_block = selection_bindings.get(block)
        source_action = source_block.get("action") if isinstance(source_block, Mapping) else None
        if (
            binding.get("semantic_sha256") != semantic_sha
            or not isinstance(source_action, Mapping)
            or source_action.get("semantic_sha256") != semantic_sha
            or source_action.get("file_sha256") != binding.get("file_sha256")
        ):
            raise BircoP1FormalControllerError(f"{block} action-pack custody drifted")
        packs[block] = pack
        pack_hashes[block] = semantic_sha

    return PrerunBindings(
        execution_freeze_self_sha256=observed_self,
        execution_freeze_file_sha256=expected_file,
        implementation_freeze_self_sha256=implementation_self,
        selection_receipt_self_sha256=selection_self,
        provider_identity=provider_identity,
        hipporag_runtime_policy=hipporag_runtime_policy,
        concurrency=concurrency,
        action_packs=packs,
        action_pack_sha256s=pack_hashes,
    )


class _LiveCounter:
    def __init__(self) -> None:
        self.current = 0
        self.peak = 0
        self._lock = threading.Lock()

    def enter(self) -> None:
        with self._lock:
            self.current += 1
            self.peak = max(self.peak, self.current)

    def leave(self) -> None:
        with self._lock:
            self.current -= 1


def _exception_hash(exc: BaseException) -> str:
    return hashlib.sha256(
        f"{type(exc).__module__}.{type(exc).__qualname__}".encode("utf-8")
    ).hexdigest()


def _ranking_payload(ranking: core.RecipeRanking) -> list[int]:
    return list(core.validate_full_permutation(
        ranking.candidate_ordinals, len(ranking.candidate_ordinals)
    ))


def _family_for_ordinal(ordinal: int) -> str:
    if type(ordinal) is not int or not 0 <= ordinal < BLOCK_ITEM_COUNT:
        raise BircoP1FormalControllerError("block ordinal is invalid")
    return selection.FAMILIES[ordinal // selection.PER_FAMILY_QUOTA]


def _model_payload(model: core.E4Model) -> dict[str, object]:
    return {
        "population_mean": list(model.population_mean),
        "population_std": list(model.population_std),
        "coefficients": list(model.coefficients),
        "laplace_covariance": [list(row) for row in model.laplace_covariance],
        "solver": model.solver,
        "iterations": model.iterations,
        "converged": model.converged,
        "objective": model.objective,
    }


def _model_from_payload(value: object) -> core.E4Model:
    if not isinstance(value, Mapping) or set(value) != {
        "population_mean",
        "population_std",
        "coefficients",
        "laplace_covariance",
        "solver",
        "iterations",
        "converged",
        "objective",
    }:
        raise BircoP1FormalControllerError("E4 model payload drifted")
    try:
        return core.E4Model(
            population_mean=tuple(value["population_mean"]),  # type: ignore[arg-type]
            population_std=tuple(value["population_std"]),  # type: ignore[arg-type]
            coefficients=tuple(value["coefficients"]),  # type: ignore[arg-type]
            laplace_covariance=tuple(
                tuple(row) for row in value["laplace_covariance"]  # type: ignore[union-attr]
            ),
            solver=value["solver"],  # type: ignore[arg-type]
            iterations=value["iterations"],  # type: ignore[arg-type]
            converged=value["converged"],  # type: ignore[arg-type]
            objective=value["objective"],  # type: ignore[arg-type]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise BircoP1FormalControllerError("E4 model payload is invalid") from exc


class FormalController:
    """Injected, resumable-at-terminal, one-shot BIRCO P1 controller."""

    def __init__(
        self,
        *,
        project_root: Path,
        control_root: Path,
        execution_freeze_path: Path,
        expected_execution_freeze_file_sha256: str,
        expected_execution_freeze_self_sha256: str,
        agent_executor: SemanticExecutor,
        raw_executor: SemanticExecutor,
        hipporag_executor: HippoExecutor,
        qrel_opener: QrelOpener,
    ) -> None:
        self.project_root = Path(project_root)
        self.control_root = Path(control_root)
        self.execution_freeze_path = Path(execution_freeze_path)
        self.expected_execution_freeze_file_sha256 = expected_execution_freeze_file_sha256
        self.expected_execution_freeze_self_sha256 = expected_execution_freeze_self_sha256
        self.agent_executor = agent_executor
        self.raw_executor = raw_executor
        self.hipporag_executor = hipporag_executor
        self.qrel_opener = qrel_opener
        self._bindings: PrerunBindings | None = None
        self._api_semaphore: threading.BoundedSemaphore | None = None
        self._api_counter = _LiveCounter()
        self._agent_counter = _LiveCounter()
        self._raw_counter = _LiveCounter()
        self._hippo_counter = _LiveCounter()
        self._hippo_gpu_counters = {"0": _LiveCounter(), "1": _LiveCounter()}

    @property
    def bindings(self) -> PrerunBindings:
        if self._bindings is None:
            raise BircoP1FormalControllerError("prerun freezes were not verified")
        return self._bindings

    def _verify_before_execution(self) -> PrerunBindings:
        verified = verify_prerun_freezes(
            project_root=self.project_root,
            execution_freeze_path=self.execution_freeze_path,
            expected_execution_freeze_file_sha256=self.expected_execution_freeze_file_sha256,
            expected_execution_freeze_self_sha256=self.expected_execution_freeze_self_sha256,
        )
        if self._bindings is not None and self._bindings != verified:
            raise BircoP1FormalControllerError("prerun bindings changed during lifecycle")
        self._bindings = verified
        self._api_semaphore = threading.BoundedSemaphore(
            verified.concurrency.total_api_call_cap
        )
        _ensure_private_directory(self.control_root)
        return verified

    def _stage_root(self, block: str) -> Path:
        if block not in BLOCK_ORDER:
            raise BircoP1FormalControllerError("stage block is invalid")
        return self.control_root / "stages" / block

    def _attempt_paths(
        self, block: str, ordinal: int, action_name: str
    ) -> tuple[Path, Path, Path]:
        root = self._stage_root(block) / "attempts" / f"{ordinal:02d}"
        return (
            root / f"{action_name}.claim.json",
            root / f"{action_name}.terminal.json",
            root / f"{action_name}.failure.json",
        )

    def _validate_provider_binding(self, terminal: Mapping[str, object]) -> None:
        if terminal.get("provider") != self.bindings.provider_identity:
            raise BircoP1FormalControllerError(
                "semantic terminal switched provider/model/key commitment"
            )

    def _semantic_validator(
        self, mode: str, expected_input: Mapping[str, object]
    ) -> Callable[[Mapping[str, object]], None]:
        def validate(terminal: Mapping[str, object]) -> None:
            try:
                integration._validate_semantic_terminal(
                    terminal, mode=mode, expected_input=expected_input
                )
            except Exception as exc:
                raise BircoP1FormalControllerError(
                    f"{mode} semantic terminal drifted"
                ) from exc
            self._validate_provider_binding(terminal)

        return validate

    def _consume_once(
        self,
        *,
        block: str,
        ordinal: int,
        action_name: str,
        role: str,
        input_sha256: str,
        call: Callable[[], Mapping[str, object]],
        validate: Callable[[Mapping[str, object]], None],
        counter: _LiveCounter,
        semaphore: threading.BoundedSemaphore | None,
    ) -> Mapping[str, object]:
        claim_path, terminal_path, failure_path = self._attempt_paths(
            block, ordinal, action_name
        )
        if failure_path.exists() or failure_path.is_symlink():
            raise BircoP1FormalControllerError(
                f"{action_name} has a terminal failure receipt"
            )
        if terminal_path.exists() or terminal_path.is_symlink():
            claim = _read_canonical(claim_path, f"{action_name} attempt claim", mode=0o400)
            terminal = _read_canonical(
                terminal_path, f"{action_name} consumed terminal", mode=0o400
            )
            if (
                claim.get("schema") != ATTEMPT_CLAIM_SCHEMA
                or claim.get("block") != block
                or claim.get("block_ordinal") != ordinal
                or claim.get("action_name") != action_name
                or claim.get("role") != role
                or claim.get("input_sha256") != input_sha256
                or claim.get("attempt_count") != 1
                or claim.get("retry_replay_resample_or_provider_switch_count") != 0
                or _verify_self(claim, "claim_sha256") != claim.get("claim_sha256")
            ):
                raise BircoP1FormalControllerError("consumed attempt claim drifted")
            validate(terminal)
            return terminal
        if claim_path.exists() or claim_path.is_symlink():
            raise BircoP1FormalControllerError(
                f"{action_name} attempt was consumed without a reusable terminal"
            )
        claim = _self_hashed(
            {
                "schema": ATTEMPT_CLAIM_SCHEMA,
                "version": VERSION,
                "study_id": STUDY_ID,
                "block": block,
                "block_ordinal": ordinal,
                "action_name": action_name,
                "role": role,
                "input_sha256": input_sha256,
                "attempt_count": 1,
                "retry_replay_resample_or_provider_switch_count": 0,
            },
            "claim_sha256",
        )
        _atomic_write_once(claim_path, claim, final_mode=0o400)
        try:
            if semaphore is not None:
                semaphore.acquire()
            counter.enter()
            try:
                result = call()
            finally:
                counter.leave()
                if semaphore is not None:
                    semaphore.release()
            if not isinstance(result, Mapping):
                raise BircoP1FormalControllerError("executor terminal is not a mapping")
            terminal = dict(result)
            validate(terminal)
            _atomic_write_once(terminal_path, terminal, final_mode=0o400)
            reread = _read_canonical(
                terminal_path, f"{action_name} consumed terminal", mode=0o400
            )
            validate(reread)
            return reread
        except BaseException as exc:
            if not failure_path.exists():
                failure = _self_hashed(
                    {
                        "schema": FAILURE_SCHEMA,
                        "version": VERSION,
                        "study_id": STUDY_ID,
                        "failure_scope": "external_attempt",
                        "block": block,
                        "block_ordinal": ordinal,
                        "action_name": action_name,
                        "role": role,
                        "claim_sha256": claim["claim_sha256"],
                        "exception_type_sha256": _exception_hash(exc),
                        "retry_authorized": False,
                    },
                    "failure_sha256",
                )
                _atomic_write_once(failure_path, failure, final_mode=0o400)
            raise

    def _semantic_call(
        self,
        *,
        block: str,
        ordinal: int,
        action_name: str,
        role: str,
        mode: str,
        payload: Mapping[str, object],
        executor: SemanticExecutor,
    ) -> Mapping[str, object]:
        counter = self._agent_counter if role == "Agent" else self._raw_counter
        return self._consume_once(
            block=block,
            ordinal=ordinal,
            action_name=action_name,
            role=role,
            input_sha256=semantic.semantic_hash(payload),
            call=lambda: executor(mode=mode, payload=payload),
            validate=self._semantic_validator(mode, payload),
            counter=counter,
            semaphore=self._api_semaphore,
        )

    def _hippo_call(
        self,
        *,
        block: str,
        ordinal: int,
        payload: Mapping[str, object],
    ) -> Mapping[str, object]:
        input_hash = hashlib.sha256(
            hippo.canonical_json_bytes(payload, newline=False)
        ).hexdigest()
        base_policy = self.bindings.hipporag_runtime_policy
        assignment = base_policy.get("gpu_assignment")
        if not isinstance(assignment, list) or len(assignment) != 4:
            raise BircoP1FormalControllerError("HippoRAG GPU assignment drifted")
        logical_slot = ordinal % 4
        visible_gpu = assignment[logical_slot]
        if visible_gpu not in self._hippo_gpu_counters:
            raise BircoP1FormalControllerError("HippoRAG visible GPU drifted")
        call_policy = {
            **dict(base_policy),
            "logical_slot_ordinal": logical_slot,
            "visible_gpu": visible_gpu,
        }

        def invoke() -> Mapping[str, object]:
            gpu_counter = self._hippo_gpu_counters[str(visible_gpu)]
            gpu_counter.enter()
            try:
                result = self.hipporag_executor(
                    payload=payload,
                    runtime_policy=call_policy,
                )
            finally:
                gpu_counter.leave()
            if not isinstance(result, Mapping) or set(result) != {
                "output",
                "runtime_receipt",
            }:
                raise BircoP1FormalControllerError(
                    "HippoRAG executor result shape drifted"
                )
            output = result.get("output")
            runtime_receipt = result.get("runtime_receipt")
            if not isinstance(output, Mapping):
                raise BircoP1FormalControllerError("HippoRAG output is absent")
            checked = self._validate_hippo_output(output, payload)
            checked_runtime = self._validate_hippo_runtime_receipt(
                runtime_receipt,
                logical_slot=logical_slot,
                visible_gpu=str(visible_gpu),
            )
            return _self_hashed(
                {
                    "schema": HIPPO_TERMINAL_SCHEMA,
                    "version": VERSION,
                    "study_id": STUDY_ID,
                    "input_sha256": input_hash,
                    "attempt_count": 1,
                    "retry_replay_or_resample_count": 0,
                    "output": checked,
                    "runtime_receipt": checked_runtime,
                },
                "self_sha256",
            )

        def validate(terminal: Mapping[str, object]) -> None:
            if (
                set(terminal)
                != {
                    "schema",
                    "version",
                    "study_id",
                    "input_sha256",
                    "attempt_count",
                    "retry_replay_or_resample_count",
                    "output",
                    "runtime_receipt",
                    "self_sha256",
                }
                or terminal.get("schema") != HIPPO_TERMINAL_SCHEMA
                or terminal.get("version") != VERSION
                or terminal.get("study_id") != STUDY_ID
                or terminal.get("input_sha256") != input_hash
                or terminal.get("attempt_count") != 1
                or terminal.get("retry_replay_or_resample_count") != 0
            ):
                raise BircoP1FormalControllerError("HippoRAG terminal drifted")
            _verify_self(terminal, "self_sha256")
            output = terminal.get("output")
            if not isinstance(output, Mapping):
                raise BircoP1FormalControllerError("HippoRAG output is absent")
            self._validate_hippo_output(output, payload)
            self._validate_hippo_runtime_receipt(
                terminal.get("runtime_receipt"),
                logical_slot=logical_slot,
                visible_gpu=str(visible_gpu),
            )

        return self._consume_once(
            block=block,
            ordinal=ordinal,
            action_name="hipporag",
            role="HippoRAG",
            input_sha256=input_hash,
            call=invoke,
            validate=validate,
            counter=self._hippo_counter,
            semaphore=None,
        )

    def _validate_hippo_runtime_receipt(
        self,
        value: object,
        *,
        logical_slot: int,
        visible_gpu: str,
    ) -> dict[str, object]:
        expected_fields = {
            "model_alias_cwd_relative",
            "subprocess_cwd_relative",
            "llm_model_argument",
            "embedding_model_argument",
            "model_arguments_are_single_relative_components",
            "absolute_model_path_argument_count",
            "logical_slot_ordinal",
            "visible_gpu",
            "configured_cpu_threads",
            "external_network_call_count",
        }
        if not isinstance(value, Mapping) or set(value) != expected_fields:
            raise BircoP1FormalControllerError("HippoRAG runtime receipt drifted")
        policy = self.bindings.hipporag_runtime_policy
        if (
            value.get("model_alias_cwd_relative")
            != policy.get("model_alias_cwd_relative")
            or value.get("subprocess_cwd_relative")
            != policy.get("model_alias_cwd_relative")
            or value.get("llm_model_argument") != "smollm2"
            or value.get("embedding_model_argument") != "minilm"
            or value.get("model_arguments_are_single_relative_components") is not True
            or value.get("absolute_model_path_argument_count") != 0
            or value.get("logical_slot_ordinal") != logical_slot
            or value.get("visible_gpu") != visible_gpu
            or value.get("configured_cpu_threads") != 2
            or value.get("external_network_call_count") != 0
        ):
            raise BircoP1FormalControllerError(
                "HippoRAG executor did not use the frozen short-alias cwd"
            )
        return dict(value)

    @staticmethod
    def _validate_hippo_output(
        value: Mapping[str, object], expected_input: Mapping[str, object]
    ) -> dict[str, Any]:
        if not isinstance(value, Mapping) or set(value) != hippo.OUTPUT_KEYS:
            raise BircoP1FormalControllerError("HippoRAG output shape drifted")
        try:
            checked = hippo.output_payload(
                work_id=value.get("work_id"),
                common_projection_sha256=value.get("common_projection_sha256"),
                candidate_count=value.get("candidate_count"),
                rank_ordinals=value.get("rank_ordinals", ()),  # type: ignore[arg-type]
                graph_nodes=value.get("graph_node_count"),
                graph_edges=value.get("graph_edge_count"),
            )
        except Exception as exc:
            raise BircoP1FormalControllerError("HippoRAG output is invalid") from exc
        if (
            checked != dict(value)
            or checked["work_id"] != expected_input.get("work_id")
            or checked["candidate_count"] != len(expected_input.get("documents", ()))
            or checked["common_projection_sha256"]
            != expected_input.get("common_projection_sha256")
        ):
            raise BircoP1FormalControllerError("HippoRAG output binding drifted")
        return checked

    @staticmethod
    def _raw_permutation(
        prepared: integration.CanonicalActionInputs,
        terminals: Sequence[Mapping[str, object]],
    ) -> tuple[int, ...]:
        if len(terminals) != len(prepared.raw_inputs):
            raise BircoP1FormalControllerError("RAW terminal batch count drifted")
        by_batch: dict[int, Mapping[str, object]] = {}
        for terminal in terminals:
            batch = terminal.get("batch_ordinal")
            if type(batch) is not int or batch in by_batch:
                raise BircoP1FormalControllerError("RAW batch ordinal drifted")
            by_batch[batch] = terminal
        scores: dict[int, int] = {}
        for batch, expected in enumerate(prepared.raw_inputs):
            terminal = by_batch.get(batch)
            if terminal is None:
                raise BircoP1FormalControllerError("RAW batch is missing")
            action = integration._validate_semantic_terminal(
                terminal, mode="raw", expected_input=expected
            )
            if set(action) != {"scores"}:
                raise BircoP1FormalControllerError("RAW action shape drifted")
            rows = action.get("scores")
            expected_ordinals = set(prepared.batch_candidate_ordinals[batch])
            if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
                raise BircoP1FormalControllerError("RAW scores are absent")
            for row in rows:
                if not isinstance(row, Mapping) or set(row) != {"ordinal", "score"}:
                    raise BircoP1FormalControllerError("RAW score row drifted")
                ordinal = row.get("ordinal")
                score = row.get("score")
                if (
                    type(ordinal) is not int
                    or ordinal not in expected_ordinals
                    or ordinal in scores
                    or type(score) is not int
                    or not 0 <= score <= 100
                ):
                    raise BircoP1FormalControllerError("RAW score value drifted")
                scores[ordinal] = score
        if set(scores) != set(range(prepared.action_item.candidate_count)):
            raise BircoP1FormalControllerError("RAW full-pool coverage drifted")
        ranking = tuple(sorted(scores, key=lambda ordinal: (-scores[ordinal], ordinal)))
        return core.validate_full_permutation(ranking, len(scores))

    def _load_stage_archive(self, block: str) -> dict[str, Any] | None:
        path = self._stage_root(block) / "action_archive.json"
        if not path.exists() and not path.is_symlink():
            return None
        value = _read_canonical(path, f"{block} action archive", mode=0o400)
        if (
            set(value)
            != {
                "schema",
                "version",
                "study_id",
                "status",
                "block",
                "arms",
                "item_count",
                "action_pack_sha256",
                "execution_freeze_sha256",
                "configured_concurrency",
                "observed_concurrency",
                "online_evaluator_call_count",
                "retry_replay_resample_or_provider_switch_count",
                "numeric_qrel_value_opened_before_archive",
                "items",
                "archive_sha256",
            }
            or value.get("schema") != STAGE_ARCHIVE_SCHEMA
            or value.get("version") != VERSION
            or value.get("study_id") != STUDY_ID
            or value.get("status")
            != "complete_action_outputs_immutable_before_qrel_open"
            or value.get("block") != block
            or value.get("arms") != list(BLOCK_ARMS[block])
            or value.get("action_pack_sha256")
            != self.bindings.action_pack_sha256s[block]
            or value.get("execution_freeze_sha256")
            != self.bindings.execution_freeze_self_sha256
            or value.get("item_count") != BLOCK_ITEM_COUNT
            or value.get("configured_concurrency")
            != self.bindings.concurrency.payload()
            or value.get("online_evaluator_call_count") != 0
            or value.get("retry_replay_resample_or_provider_switch_count") != 0
            or value.get("numeric_qrel_value_opened_before_archive") is not False
            or not isinstance(value.get("items"), list)
            or len(value["items"]) != BLOCK_ITEM_COUNT
        ):
            raise BircoP1FormalControllerError(f"{block} action archive drifted")
        observed = value.get("observed_concurrency")
        if (
            not isinstance(observed, Mapping)
            or set(observed)
            != {
                "agent_api_peak",
                "raw_api_peak",
                "total_api_peak",
                "hipporag_peak",
                "hipporag_per_gpu_peak",
            }
            or any(
                type(observed.get(field)) is not int or observed[field] < 0
                for field in (
                    "agent_api_peak",
                    "raw_api_peak",
                    "total_api_peak",
                    "hipporag_peak",
                )
            )
            or observed["agent_api_peak"]
            > self.bindings.concurrency.agent_api_workers
            or observed["raw_api_peak"] > self.bindings.concurrency.raw_api_workers
            or observed["total_api_peak"]
            > self.bindings.concurrency.total_api_call_cap
            or observed["hipporag_peak"]
            > self.bindings.concurrency.hipporag_workers
            or not isinstance(observed.get("hipporag_per_gpu_peak"), Mapping)
            or set(observed["hipporag_per_gpu_peak"]) != {"0", "1"}
            or any(
                type(row) is not int
                or not 0
                <= row
                <= int(
                    self.bindings.hipporag_runtime_policy[
                        "maximum_processes_per_gpu"
                    ]
                )
                for row in observed["hipporag_per_gpu_peak"].values()
            )
        ):
            raise BircoP1FormalControllerError(
                f"{block} concurrency attestation drifted"
            )
        _verify_self(value, "archive_sha256")
        return value

    def _verify_archived_action_terminals(
        self,
        *,
        block: str,
        archive: Mapping[str, Any],
        e4_model: core.E4Model | None,
    ) -> None:
        """Revalidate every consumed terminal without invoking an executor."""

        pack_items = self.bindings.action_packs[block].get("items")
        archived_items = archive.get("items")
        if not isinstance(pack_items, list) or not isinstance(archived_items, list):
            raise BircoP1FormalControllerError("archive recovery inputs are absent")
        row_fields = {
            "block_ordinal",
            "work_id",
            "candidate_count",
            "common_projection_sha256",
            "plan_terminal_sha256",
            "matrix_terminal_sha256s",
            "recipe_rankings",
            "action_features",
            "E0_recipe_id",
            "E0_ranking",
            "E4_selection",
            "E4_ranking",
            "RAW_ranking",
            "RAW_terminal_sha256s",
            "HippoRAG_ranking",
            "HippoRAG_terminal_sha256",
            "all_present_arms_common_projection_sha256_equal",
        }
        for ordinal, (pack_item, archived) in enumerate(
            zip(pack_items, archived_items)
        ):
            if not isinstance(archived, Mapping) or set(archived) != row_fields:
                raise BircoP1FormalControllerError("archived action row drifted")
            prepared = integration.prepare_canonical_action_inputs(pack_item)
            if (
                archived.get("block_ordinal") != ordinal
                or archived.get("work_id") != prepared.action_item.work_id
                or archived.get("candidate_count")
                != prepared.action_item.candidate_count
                or archived.get("common_projection_sha256")
                != prepared.pool_common_projection_sha256
                or archived.get("all_present_arms_common_projection_sha256_equal")
                is not True
            ):
                raise BircoP1FormalControllerError("archive action binding drifted")
            plan_terminal = self._semantic_call(
                block=block,
                ordinal=ordinal,
                action_name="plan",
                role="Agent",
                mode="plan",
                payload=prepared.planner_input,
                executor=lambda **_kwargs: (_ for _ in ()).throw(
                    BircoP1FormalControllerError("planner recovery attempted execution")
                ),
            )
            if archived.get("plan_terminal_sha256") != plan_terminal.get(
                "self_sha256"
            ):
                raise BircoP1FormalControllerError("archived planner hash drifted")
            matrix_stage = integration.build_canonical_matrix_inputs(
                prepared, plan_terminal
            )
            matrix_terminals = tuple(
                self._semantic_call(
                    block=block,
                    ordinal=ordinal,
                    action_name=f"matrix.{batch:03d}",
                    role="Agent",
                    mode="matrix",
                    payload=payload,
                    executor=lambda **_kwargs: (_ for _ in ()).throw(
                        BircoP1FormalControllerError(
                            "matrix recovery attempted execution"
                        )
                    ),
                )
                for batch, payload in enumerate(matrix_stage.matrix_inputs)
            )
            if archived.get("matrix_terminal_sha256s") != [
                row["self_sha256"] for row in matrix_terminals
            ]:
                raise BircoP1FormalControllerError("archived matrix hashes drifted")
            matrix = integration.merge_matrix_terminals(
                matrix_stage, matrix_terminals
            )
            rankings = core.build_recipe_rankings(matrix_stage.core_plan, matrix)
            features = {
                recipe: core.compute_action_features(
                    matrix_stage.core_plan, matrix, rankings[recipe]
                )
                for recipe in core.RECIPE_IDS
            }
            expected_rankings = {
                recipe: _ranking_payload(rankings[recipe])
                for recipe in core.RECIPE_IDS
            }
            expected_features = {
                recipe: list(features[recipe]) for recipe in core.RECIPE_IDS
            }
            e0 = core.select_e0_recipe(matrix_stage.core_plan)
            if (
                archived.get("recipe_rankings") != expected_rankings
                or archived.get("action_features") != expected_features
                or archived.get("E0_recipe_id") != e0
                or archived.get("E0_ranking") != expected_rankings[e0]
            ):
                raise BircoP1FormalControllerError("archived Agent action drifted")
            if e4_model is None:
                if archived.get("E4_selection") is not None or archived.get(
                    "E4_ranking"
                ) is not None:
                    raise BircoP1FormalControllerError("unexpected archived E4 action")
            else:
                e4 = core.select_e4_recipe(e4_model, features, e0_recipe_id=e0)
                if (
                    archived.get("E4_selection") != e4.payload()
                    or archived.get("E4_ranking")
                    != expected_rankings[e4.selected_recipe_id]
                ):
                    raise BircoP1FormalControllerError("archived E4 action drifted")

            if "RAW" in BLOCK_ARMS[block]:
                raw_terminals = tuple(
                    self._semantic_call(
                        block=block,
                        ordinal=ordinal,
                        action_name=f"raw.{batch:03d}",
                        role="RAW",
                        mode="raw",
                        payload=payload,
                        executor=lambda **_kwargs: (_ for _ in ()).throw(
                            BircoP1FormalControllerError(
                                "RAW recovery attempted execution"
                            )
                        ),
                    )
                    for batch, payload in enumerate(prepared.raw_inputs)
                )
                if (
                    archived.get("RAW_terminal_sha256s")
                    != [row["self_sha256"] for row in raw_terminals]
                    or archived.get("RAW_ranking")
                    != list(self._raw_permutation(prepared, raw_terminals))
                ):
                    raise BircoP1FormalControllerError("archived RAW action drifted")
            elif archived.get("RAW_ranking") is not None or archived.get(
                "RAW_terminal_sha256s"
            ) != []:
                raise BircoP1FormalControllerError("unexpected archived RAW action")

            if "HippoRAG" in BLOCK_ARMS[block]:
                nested = pack_item.get("hipporag_input")
                if not isinstance(nested, Mapping):
                    raise BircoP1FormalControllerError("HippoRAG recovery input absent")
                hippo_terminal = self._hippo_call(
                    block=block,
                    ordinal=ordinal,
                    payload=nested,
                )
                output = hippo_terminal["output"]
                if (
                    archived.get("HippoRAG_terminal_sha256")
                    != hippo_terminal.get("self_sha256")
                    or archived.get("HippoRAG_ranking")
                    != output.get("rank_ordinals")
                ):
                    raise BircoP1FormalControllerError(
                        "archived HippoRAG action drifted"
                    )
            elif archived.get("HippoRAG_ranking") is not None or archived.get(
                "HippoRAG_terminal_sha256"
            ) is not None:
                raise BircoP1FormalControllerError(
                    "unexpected archived HippoRAG action"
                )

    def _materialize_stage(
        self, *, block: str, e4_model: core.E4Model | None
    ) -> dict[str, Any]:
        stage_failure = self._stage_root(block) / "stage.failure.json"
        if stage_failure.exists() or stage_failure.is_symlink():
            raise BircoP1FormalControllerError(
                f"{block} already has a terminal stage-failure receipt"
            )
        existing = self._load_stage_archive(block)
        if existing is not None:
            self._verify_archived_action_terminals(
                block=block, archive=existing, e4_model=e4_model
            )
            return existing
        pack = self.bindings.action_packs[block]
        items = pack.get("items")
        if not isinstance(items, list) or len(items) != BLOCK_ITEM_COUNT:
            raise BircoP1FormalControllerError("action pack items drifted")
        prepared = tuple(integration.prepare_canonical_action_inputs(item) for item in items)
        if tuple(row.action_item.block_ordinal for row in prepared) != tuple(
            range(BLOCK_ITEM_COUNT)
        ):
            raise BircoP1FormalControllerError("action pack ordinal order drifted")
        if (block == "A_form") != (e4_model is None):
            raise BircoP1FormalControllerError("E4 model stage availability drifted")

        before = (
            self._agent_counter.peak,
            self._raw_counter.peak,
            self._hippo_counter.peak,
            self._api_counter.peak,
        )
        # The total counter surrounds the same shared semaphore as both API
        # arms.  A small wrapper keeps arm and total observations independent.
        original_agent = self.agent_executor
        original_raw = self.raw_executor

        def agent_exec(*, mode: str, payload: Mapping[str, object]) -> Mapping[str, object]:
            self._api_counter.enter()
            try:
                return original_agent(mode=mode, payload=payload)
            finally:
                self._api_counter.leave()

        def raw_exec(*, mode: str, payload: Mapping[str, object]) -> Mapping[str, object]:
            self._api_counter.enter()
            try:
                return original_raw(mode=mode, payload=payload)
            finally:
                self._api_counter.leave()

        plan_futures: dict[Future[Mapping[str, object]], int] = {}
        matrix_futures: dict[Future[Mapping[str, object]], tuple[int, int]] = {}
        raw_futures: dict[Future[Mapping[str, object]], tuple[int, int]] = {}
        hippo_futures: dict[Future[Mapping[str, object]], int] = {}
        plans: dict[int, Mapping[str, object]] = {}
        matrix_stages: dict[int, integration.CanonicalMatrixInputs] = {}
        matrices: dict[int, dict[int, Mapping[str, object]]] = {
            ordinal: {} for ordinal in range(BLOCK_ITEM_COUNT)
        }
        raws: dict[int, dict[int, Mapping[str, object]]] = {
            ordinal: {} for ordinal in range(BLOCK_ITEM_COUNT)
        }
        hippos: dict[int, Mapping[str, object]] = {}
        policy = self.bindings.concurrency
        with ThreadPoolExecutor(
            max_workers=policy.agent_api_workers,
            thread_name_prefix=f"birco-{block}-agent",
        ) as agent_pool, ThreadPoolExecutor(
            max_workers=policy.raw_api_workers,
            thread_name_prefix=f"birco-{block}-raw",
        ) as raw_pool, ThreadPoolExecutor(
            max_workers=policy.hipporag_workers,
            thread_name_prefix=f"birco-{block}-hippo",
        ) as hippo_pool:
            # Eager independent plan, RAW, and Hippo submission precedes joins.
            for ordinal, row in enumerate(prepared):
                plan_futures[
                    agent_pool.submit(
                        self._semantic_call,
                        block=block,
                        ordinal=ordinal,
                        action_name="plan",
                        role="Agent",
                        mode="plan",
                        payload=row.planner_input,
                        executor=agent_exec,
                    )
                ] = ordinal
                if "RAW" in BLOCK_ARMS[block]:
                    for batch, payload in enumerate(row.raw_inputs):
                        raw_futures[
                            raw_pool.submit(
                                self._semantic_call,
                                block=block,
                                ordinal=ordinal,
                                action_name=f"raw.{batch:03d}",
                                role="RAW",
                                mode="raw",
                                payload=payload,
                                executor=raw_exec,
                            )
                        ] = (ordinal, batch)
                if "HippoRAG" in BLOCK_ARMS[block]:
                    hippo_payload = row.action_item
                    nested = items[ordinal].get("hipporag_input")
                    if not isinstance(nested, Mapping):
                        raise BircoP1FormalControllerError("HippoRAG input is absent")
                    hippo_futures[
                        hippo_pool.submit(
                            self._hippo_call,
                            block=block,
                            ordinal=ordinal,
                            payload=dict(nested),
                        )
                    ] = ordinal

            # Each matrix payload is created only from its consumed plan.
            for future in as_completed(plan_futures):
                ordinal = plan_futures[future]
                terminal = future.result()
                plans[ordinal] = terminal
                stage = integration.build_canonical_matrix_inputs(
                    prepared[ordinal], terminal
                )
                matrix_stages[ordinal] = stage
                for batch, payload in enumerate(stage.matrix_inputs):
                    matrix_futures[
                        agent_pool.submit(
                            self._semantic_call,
                            block=block,
                            ordinal=ordinal,
                            action_name=f"matrix.{batch:03d}",
                            role="Agent",
                            mode="matrix",
                            payload=payload,
                            executor=agent_exec,
                        )
                    ] = (ordinal, batch)
            for future in as_completed(matrix_futures):
                ordinal, batch = matrix_futures[future]
                matrices[ordinal][batch] = future.result()
            for future in as_completed(raw_futures):
                ordinal, batch = raw_futures[future]
                raws[ordinal][batch] = future.result()
            for future in as_completed(hippo_futures):
                hippos[hippo_futures[future]] = future.result()

        rows: list[dict[str, Any]] = []
        for ordinal, row in enumerate(prepared):
            stage = matrix_stages[ordinal]
            matrix_terminals = tuple(
                matrices[ordinal][batch] for batch in range(len(stage.matrix_inputs))
            )
            matrix = integration.merge_matrix_terminals(stage, matrix_terminals)
            ranking_map = core.build_recipe_rankings(stage.core_plan, matrix)
            features = {
                recipe: core.compute_action_features(
                    stage.core_plan, matrix, ranking_map[recipe]
                )
                for recipe in core.RECIPE_IDS
            }
            e0 = core.select_e0_recipe(stage.core_plan)
            item_row: dict[str, Any] = {
                "block_ordinal": ordinal,
                "work_id": row.action_item.work_id,
                "candidate_count": row.action_item.candidate_count,
                "common_projection_sha256": row.pool_common_projection_sha256,
                "plan_terminal_sha256": plans[ordinal]["self_sha256"],
                "matrix_terminal_sha256s": [
                    terminal["self_sha256"] for terminal in matrix_terminals
                ],
                "recipe_rankings": {
                    recipe: _ranking_payload(ranking_map[recipe])
                    for recipe in core.RECIPE_IDS
                },
                "action_features": {
                    recipe: list(features[recipe]) for recipe in core.RECIPE_IDS
                },
                "E0_recipe_id": e0,
                "E0_ranking": _ranking_payload(ranking_map[e0]),
                "E4_selection": None,
                "E4_ranking": None,
                "RAW_ranking": None,
                "RAW_terminal_sha256s": [],
                "HippoRAG_ranking": None,
                "HippoRAG_terminal_sha256": None,
                "all_present_arms_common_projection_sha256_equal": True,
            }
            if e4_model is not None:
                e4 = core.select_e4_recipe(e4_model, features, e0_recipe_id=e0)
                item_row["E4_selection"] = e4.payload()
                item_row["E4_ranking"] = _ranking_payload(
                    ranking_map[e4.selected_recipe_id]
                )
            if "RAW" in BLOCK_ARMS[block]:
                raw_terminals = tuple(
                    raws[ordinal][batch] for batch in range(len(row.raw_inputs))
                )
                for terminal in raw_terminals:
                    self._validate_provider_binding(terminal)
                    if terminal.get("pool_common_projection_sha256") != row.pool_common_projection_sha256:
                        raise BircoP1FormalControllerError("RAW common projection drifted")
                item_row["RAW_ranking"] = list(
                    self._raw_permutation(row, raw_terminals)
                )
                item_row["RAW_terminal_sha256s"] = [
                    terminal["self_sha256"] for terminal in raw_terminals
                ]
            if "HippoRAG" in BLOCK_ARMS[block]:
                terminal = hippos[ordinal]
                output = terminal["output"]
                if output["common_projection_sha256"] != row.pool_common_projection_sha256:
                    raise BircoP1FormalControllerError("HippoRAG common projection drifted")
                item_row["HippoRAG_ranking"] = list(output["rank_ordinals"])
                item_row["HippoRAG_terminal_sha256"] = terminal["self_sha256"]
            rows.append(item_row)

        observed = {
            "agent_api_peak": self._agent_counter.peak,
            "raw_api_peak": self._raw_counter.peak if "RAW" in BLOCK_ARMS[block] else 0,
            "total_api_peak": self._api_counter.peak,
            "hipporag_peak": self._hippo_counter.peak if "HippoRAG" in BLOCK_ARMS[block] else 0,
            "hipporag_per_gpu_peak": {
                gpu: counter.peak if "HippoRAG" in BLOCK_ARMS[block] else 0
                for gpu, counter in sorted(self._hippo_gpu_counters.items())
            },
        }
        if (
            observed["agent_api_peak"] > policy.agent_api_workers
            or observed["raw_api_peak"] > policy.raw_api_workers
            or observed["total_api_peak"] > policy.total_api_call_cap
            or observed["hipporag_peak"] > policy.hipporag_workers
            or any(
                value
                > int(self.bindings.hipporag_runtime_policy["maximum_processes_per_gpu"])
                for value in observed["hipporag_per_gpu_peak"].values()
            )
        ):
            raise BircoP1FormalControllerError("observed concurrency exceeded freeze")
        archive = _self_hashed(
            {
                "schema": STAGE_ARCHIVE_SCHEMA,
                "version": VERSION,
                "study_id": STUDY_ID,
                "status": "complete_action_outputs_immutable_before_qrel_open",
                "block": block,
                "arms": list(BLOCK_ARMS[block]),
                "item_count": len(rows),
                "action_pack_sha256": self.bindings.action_pack_sha256s[block],
                "execution_freeze_sha256": self.bindings.execution_freeze_self_sha256,
                "configured_concurrency": policy.payload(),
                "observed_concurrency": observed,
                "online_evaluator_call_count": 0,
                "retry_replay_resample_or_provider_switch_count": 0,
                "numeric_qrel_value_opened_before_archive": False,
                "items": rows,
            },
            "archive_sha256",
        )
        _seal_read_only(
            self._stage_root(block) / "action_archive.json",
            archive,
            "archive_sha256",
        )
        return archive

    def _qrels_after_archive(
        self,
        *,
        block: str,
        archive: Mapping[str, Any],
        promotion_sha256: str | None = None,
    ) -> dict[str, Any]:
        if block == "F_search":
            raise BircoP1FormalControllerError("F_search qrels are permanently sealed")
        archive_sha = _verify_self(archive, "archive_sha256")
        path = self._stage_root(block) / "action_archive.json"
        reread = _read_canonical(path, f"{block} action archive", mode=0o400)
        if reread != dict(archive) or _verify_self(reread, "archive_sha256") != archive_sha:
            raise BircoP1FormalControllerError("action archive is not immutable")
        qrels = self.qrel_opener(
            block=block,
            action_archive_sha256s=(archive_sha,),
            promotion_sha256=promotion_sha256,
        )
        if not isinstance(qrels, Mapping):
            raise BircoP1FormalControllerError("qrel opener returned no pack")
        qrel_pack = dict(qrels)
        try:
            selection._validate_qrel_pack(
                qrel_pack,
                block=block,
                expected_action_pack_sha256=self.bindings.action_pack_sha256s[block],
            )
        except Exception as exc:
            raise BircoP1FormalControllerError("opened qrel pack drifted") from exc
        return qrel_pack

    @staticmethod
    def _qrel_rows(
        archive: Mapping[str, Any], qrel_pack: Mapping[str, Any]
    ) -> tuple[tuple[str, Mapping[int, float]], ...]:
        action_items = archive.get("items")
        qrel_items = qrel_pack.get("items")
        if not isinstance(action_items, list) or not isinstance(qrel_items, list):
            raise BircoP1FormalControllerError("score inputs are absent")
        if len(action_items) != len(qrel_items):
            raise BircoP1FormalControllerError("qrel/action item counts differ")
        result: list[tuple[str, Mapping[int, float]]] = []
        for ordinal, (action, labels) in enumerate(zip(action_items, qrel_items)):
            if not isinstance(action, Mapping) or not isinstance(labels, Mapping):
                raise BircoP1FormalControllerError("qrel/action row drifted")
            family = labels.get("family")
            values = labels.get("qrel_values")
            if (
                labels.get("block_ordinal") != ordinal
                or action.get("block_ordinal") != ordinal
                or labels.get("work_id") != action.get("work_id")
                or family != _family_for_ordinal(ordinal)
                or not isinstance(values, list)
                or len(values) != action.get("candidate_count")
            ):
                raise BircoP1FormalControllerError("qrel/action binding drifted")
            relevance: dict[int, float] = {}
            for candidate, row in enumerate(values):
                if (
                    not isinstance(row, Mapping)
                    or row.get("candidate_ordinal") != candidate
                    or isinstance(row.get("value"), bool)
                    or not isinstance(row.get("value"), (int, float))
                    or not math.isfinite(float(row["value"]))
                ):
                    raise BircoP1FormalControllerError("qrel value drifted")
                relevance[candidate] = float(row["value"])
            result.append((str(family), relevance))
        return tuple(result)

    def _load_or_fit_e4(self, archive: Mapping[str, Any]) -> tuple[core.E4Model, str]:
        path = self.control_root / "A_form_e4_model.json"
        if path.exists() or path.is_symlink():
            receipt = _read_canonical(path, "A_form E4 model", mode=0o400)
            if (
                set(receipt)
                != {
                    "schema",
                    "version",
                    "study_id",
                    "status",
                    "A_form_action_archive_sha256",
                    "A_form_qrel_pack_sha256",
                    "training_slate_count",
                    "model",
                    "online_evaluator_call_count",
                    "model_receipt_sha256",
                }
                or receipt.get("schema") != f"{VERSION}_A_form_e4_model_v1"
                or receipt.get("version") != VERSION
                or receipt.get("study_id") != STUDY_ID
                or receipt.get("status")
                != "single_E4_fit_after_immutable_A_form_actions"
                or receipt.get("A_form_action_archive_sha256")
                != archive.get("archive_sha256")
                or _SHA256.fullmatch(str(receipt.get("A_form_qrel_pack_sha256")))
                is None
                or receipt.get("training_slate_count") != BLOCK_ITEM_COUNT
                or receipt.get("online_evaluator_call_count") != 0
            ):
                raise BircoP1FormalControllerError("A_form E4 model receipt drifted")
            receipt_sha = _verify_self(receipt, "model_receipt_sha256")
            return _model_from_payload(receipt.get("model")), receipt_sha
        qrels = self._qrels_after_archive(block="A_form", archive=archive)
        label_rows = self._qrel_rows(archive, qrels)
        slates: list[core.E4TrainingSlate] = []
        for action, (_family, relevance) in zip(archive["items"], label_rows):
            features = action["action_features"]
            rankings = action["recipe_rankings"]
            utilities = {
                recipe: core.score_full_permutation(
                    rankings[recipe], relevance
                ).integer_utility
                for recipe in core.RECIPE_IDS
            }
            slates.append(core.make_e4_training_slate(features, utilities))
        model = core.fit_e4_listwise_softmax(tuple(slates))
        receipt = _self_hashed(
            {
                "schema": f"{VERSION}_A_form_e4_model_v1",
                "version": VERSION,
                "study_id": STUDY_ID,
                "status": "single_E4_fit_after_immutable_A_form_actions",
                "A_form_action_archive_sha256": archive["archive_sha256"],
                "A_form_qrel_pack_sha256": qrels["qrel_pack_sha256"],
                "training_slate_count": len(slates),
                "model": _model_payload(model),
                "online_evaluator_call_count": 0,
            },
            "model_receipt_sha256",
        )
        receipt_sha = _seal_read_only(path, receipt, "model_receipt_sha256")
        return model, receipt_sha

    def _f_identifiability(
        self, archive: Mapping[str, Any], model_receipt_sha256: str
    ) -> tuple[core.FIdentifiabilityResult, str]:
        path = self.control_root / "F_search_identifiability.json"
        if path.exists() or path.is_symlink():
            receipt = _read_canonical(path, "F_search identifiability", mode=0o400)
            if (
                set(receipt)
                != {
                    "schema",
                    "version",
                    "study_id",
                    "status",
                    "F_search_action_archive_sha256",
                    "A_form_model_receipt_sha256",
                    "item_count",
                    "differing_ranking_count",
                    "differing_family_count",
                    "passed",
                    "F_search_qrel_open_count",
                    "receipt_sha256",
                }
                or receipt.get("schema")
                != f"{VERSION}_F_search_identifiability_v1"
                or receipt.get("version") != VERSION
                or receipt.get("study_id") != STUDY_ID
                or receipt.get("status")
                != "label_free_permutation_identifiability_complete"
                or receipt.get("F_search_action_archive_sha256")
                != archive.get("archive_sha256")
                or receipt.get("A_form_model_receipt_sha256")
                != model_receipt_sha256
                or receipt.get("item_count") != BLOCK_ITEM_COUNT
                or type(receipt.get("differing_ranking_count")) is not int
                or type(receipt.get("differing_family_count")) is not int
                or type(receipt.get("passed")) is not bool
                or receipt.get("F_search_qrel_open_count") != 0
            ):
                raise BircoP1FormalControllerError(
                    "F_search identifiability receipt drifted"
                )
            receipt_sha = _verify_self(receipt, "receipt_sha256")
            result = core.FIdentifiabilityResult(
                item_count=receipt["item_count"],
                differing_ranking_count=receipt["differing_ranking_count"],
                differing_family_count=receipt["differing_family_count"],
                passed=receipt["passed"],
            )
            return result, receipt_sha
        items = archive["items"]
        result = core.assess_f_identifiability(
            [row["E4_ranking"] for row in items],
            [row["E0_ranking"] for row in items],
            [_family_for_ordinal(index) for index in range(len(items))],
        )
        receipt = _self_hashed(
            {
                "schema": f"{VERSION}_F_search_identifiability_v1",
                "version": VERSION,
                "study_id": STUDY_ID,
                "status": "label_free_permutation_identifiability_complete",
                "F_search_action_archive_sha256": archive["archive_sha256"],
                "A_form_model_receipt_sha256": model_receipt_sha256,
                "item_count": result.item_count,
                "differing_ranking_count": result.differing_ranking_count,
                "differing_family_count": result.differing_family_count,
                "passed": result.passed,
                "F_search_qrel_open_count": 0,
            },
            "receipt_sha256",
        )
        receipt_sha = _seal_read_only(path, receipt, "receipt_sha256")
        return result, receipt_sha

    @staticmethod
    def _paired_payload(value: core.PairedUtilitySummary) -> dict[str, object]:
        return value.payload()

    @staticmethod
    def _paired_from_payload(value: object) -> core.PairedUtilitySummary:
        expected = {
            "item_count",
            "total_integer_delta",
            "gains",
            "harms",
            "ties",
            "ties_excluded_from_tail",
            "descriptive_binomial_tail",
            "descriptive_reference_only",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise BircoP1FormalControllerError("paired utility payload drifted")
        tail = value.get("descriptive_binomial_tail")
        if (
            not isinstance(tail, Mapping)
            or set(tail) != {"numerator", "denominator"}
            or type(tail.get("numerator")) is not int
            or type(tail.get("denominator")) is not int
            or tail["denominator"] <= 0
            or value.get("ties_excluded_from_tail") is not True
            or value.get("descriptive_reference_only") is not True
        ):
            raise BircoP1FormalControllerError("paired utility tail drifted")
        try:
            result = core.PairedUtilitySummary(
                item_count=value["item_count"],  # type: ignore[arg-type]
                total_integer_delta=value["total_integer_delta"],  # type: ignore[arg-type]
                gains=value["gains"],  # type: ignore[arg-type]
                harms=value["harms"],  # type: ignore[arg-type]
                ties=value["ties"],  # type: ignore[arg-type]
                descriptive_reference_tail=Fraction(
                    tail["numerator"], tail["denominator"]
                ),
            )
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            raise BircoP1FormalControllerError(
                "paired utility payload is invalid"
            ) from exc
        if result.payload() != dict(value):
            raise BircoP1FormalControllerError("paired utility payload is noncanonical")
        return result

    def _score_a_hold(
        self,
        archive: Mapping[str, Any],
        *,
        f_passed: bool,
        f_receipt_sha256: str,
    ) -> tuple[core.RealityPrimaryDecision, core.E4PromotionDecision, str]:
        path = self.control_root / "A_hold_score_and_promotion.json"
        if path.exists() or path.is_symlink():
            receipt = _read_canonical(path, "A_hold score and promotion", mode=0o400)
            if (
                set(receipt)
                != {
                    "schema",
                    "version",
                    "study_id",
                    "status",
                    "A_hold_action_archive_sha256",
                    "A_hold_qrel_pack_sha256",
                    "F_search_identifiability_receipt_sha256",
                    "reality_primary",
                    "E4_promotion",
                    "online_evaluator_call_count",
                    "promotion_receipt_sha256",
                }
                or receipt.get("schema")
                != f"{VERSION}_A_hold_score_and_promotion_v1"
                or receipt.get("version") != VERSION
                or receipt.get("study_id") != STUDY_ID
                or receipt.get("status")
                != "offline_A_hold_reality_and_single_challenger_promotion_complete"
                or receipt.get("A_hold_action_archive_sha256")
                != archive.get("archive_sha256")
                or _SHA256.fullmatch(str(receipt.get("A_hold_qrel_pack_sha256")))
                is None
                or receipt.get("F_search_identifiability_receipt_sha256")
                != f_receipt_sha256
                or receipt.get("online_evaluator_call_count") != 0
            ):
                raise BircoP1FormalControllerError(
                    "A_hold score and promotion receipt drifted"
                )
            receipt_sha = _verify_self(receipt, "promotion_receipt_sha256")
            reality_value = receipt.get("reality_primary")
            promotion_value = receipt.get("E4_promotion")
            if (
                not isinstance(reality_value, Mapping)
                or set(reality_value)
                != {
                    "agent_minus_RAW",
                    "agent_minus_HippoRAG",
                    "RAW_family_integer_deltas",
                    "HippoRAG_family_integer_deltas",
                    "passed",
                }
                or not isinstance(promotion_value, Mapping)
            ):
                raise BircoP1FormalControllerError("A_hold decision payload drifted")
            reality = core.RealityPrimaryDecision(
                agent_minus_raw=self._paired_from_payload(
                    reality_value["agent_minus_RAW"]
                ),
                agent_minus_hipporag=self._paired_from_payload(
                    reality_value["agent_minus_HippoRAG"]
                ),
                raw_family_integer_deltas=tuple(
                    reality_value["RAW_family_integer_deltas"]
                ),  # type: ignore[arg-type]
                hipporag_family_integer_deltas=tuple(
                    reality_value["HippoRAG_family_integer_deltas"]
                ),  # type: ignore[arg-type]
                passed=reality_value["passed"],  # type: ignore[arg-type]
            )
            comparison = self._paired_from_payload(promotion_value.get("comparison"))
            promotion = core.E4PromotionDecision(
                comparison=comparison,
                f_identifiability_passed=promotion_value.get(
                    "F_identifiability_passed"
                ),  # type: ignore[arg-type]
                promoted=promotion_value.get("promoted"),  # type: ignore[arg-type]
            )
            if (
                promotion.payload() != dict(promotion_value)
                or promotion.f_identifiability_passed is not f_passed
                or promotion.comparison.item_count != BLOCK_ITEM_COUNT
                or promotion.promoted
                is not (
                    f_passed
                    and promotion.comparison.total_integer_delta > 0
                    and promotion.comparison.tail_at_most_alpha
                )
                or list(reality.raw_family_integer_deltas)
                != reality_value["RAW_family_integer_deltas"]
                or list(reality.hipporag_family_integer_deltas)
                != reality_value["HippoRAG_family_integer_deltas"]
                or len(reality.raw_family_integer_deltas) != 3
                or len(reality.hipporag_family_integer_deltas) != 3
                or any(
                    type(row) is not int
                    for row in (
                        *reality.raw_family_integer_deltas,
                        *reality.hipporag_family_integer_deltas,
                    )
                )
                or reality.agent_minus_raw.item_count != BLOCK_ITEM_COUNT
                or reality.agent_minus_hipporag.item_count != BLOCK_ITEM_COUNT
                or reality.passed
                is not (
                    reality.agent_minus_raw.total_integer_delta > 0
                    and reality.agent_minus_hipporag.total_integer_delta > 0
                    and all(
                        value > 0
                        for value in (
                            *reality.raw_family_integer_deltas,
                            *reality.hipporag_family_integer_deltas,
                        )
                    )
                    and reality.agent_minus_raw.tail_at_most_alpha
                    and reality.agent_minus_hipporag.tail_at_most_alpha
                )
                or type(reality.passed) is not bool
            ):
                raise BircoP1FormalControllerError("A_hold decision binding drifted")
            return reality, promotion, receipt_sha
        qrels = self._qrels_after_archive(block="A_hold", archive=archive)
        labels = self._qrel_rows(archive, qrels)
        utilities: dict[str, list[int]] = {
            name: [] for name in ("E0", "E4", "RAW", "HippoRAG")
        }
        families: list[str] = []
        for action, (family, relevance) in zip(archive["items"], labels):
            families.append(family)
            for name, field in (
                ("E0", "E0_ranking"),
                ("E4", "E4_ranking"),
                ("RAW", "RAW_ranking"),
                ("HippoRAG", "HippoRAG_ranking"),
            ):
                ranking = action.get(field)
                if not isinstance(ranking, list):
                    raise BircoP1FormalControllerError(f"A_hold {name} ranking absent")
                utilities[name].append(
                    core.score_full_permutation(ranking, relevance).integer_utility
                )
        reality = core.decide_a_hold_reality_primary(
            utilities["E0"], utilities["RAW"], utilities["HippoRAG"], families
        )
        promotion = core.decide_a_hold_e4_promotion(
            utilities["E4"], utilities["E0"], f_identifiability_passed=f_passed
        )
        receipt = _self_hashed(
            {
                "schema": f"{VERSION}_A_hold_score_and_promotion_v1",
                "version": VERSION,
                "study_id": STUDY_ID,
                "status": "offline_A_hold_reality_and_single_challenger_promotion_complete",
                "A_hold_action_archive_sha256": archive["archive_sha256"],
                "A_hold_qrel_pack_sha256": qrels["qrel_pack_sha256"],
                "F_search_identifiability_receipt_sha256": f_receipt_sha256,
                "reality_primary": {
                    "agent_minus_RAW": self._paired_payload(reality.agent_minus_raw),
                    "agent_minus_HippoRAG": self._paired_payload(
                        reality.agent_minus_hipporag
                    ),
                    "RAW_family_integer_deltas": list(
                        reality.raw_family_integer_deltas
                    ),
                    "HippoRAG_family_integer_deltas": list(
                        reality.hipporag_family_integer_deltas
                    ),
                    "passed": reality.passed,
                },
                "E4_promotion": promotion.payload(),
                "online_evaluator_call_count": 0,
            },
            "promotion_receipt_sha256",
        )
        receipt_sha = _seal_read_only(path, receipt, "promotion_receipt_sha256")
        return reality, promotion, receipt_sha

    def _score_m_search(
        self, archive: Mapping[str, Any], *, promotion_sha256: str
    ) -> tuple[core.MSearchDecision, str]:
        path = self.control_root / "M_search_score.json"
        if path.exists() or path.is_symlink():
            receipt = _read_canonical(path, "M_search score", mode=0o400)
            if (
                set(receipt)
                != {
                    "schema",
                    "version",
                    "study_id",
                    "status",
                    "M_search_action_archive_sha256",
                    "M_search_qrel_pack_sha256",
                    "A_hold_promotion_receipt_sha256",
                    "comparison",
                    "family_integer_deltas",
                    "passed",
                    "online_evaluator_call_count",
                    "score_receipt_sha256",
                }
                or receipt.get("schema") != f"{VERSION}_M_search_score_v1"
                or receipt.get("version") != VERSION
                or receipt.get("study_id") != STUDY_ID
                or receipt.get("status") != "offline_untouched_M_search_complete"
                or receipt.get("M_search_action_archive_sha256")
                != archive.get("archive_sha256")
                or _SHA256.fullmatch(str(receipt.get("M_search_qrel_pack_sha256")))
                is None
                or receipt.get("A_hold_promotion_receipt_sha256")
                != promotion_sha256
                or receipt.get("online_evaluator_call_count") != 0
                or type(receipt.get("passed")) is not bool
            ):
                raise BircoP1FormalControllerError("M_search score receipt drifted")
            receipt_sha = _verify_self(receipt, "score_receipt_sha256")
            decision = core.MSearchDecision(
                comparison=self._paired_from_payload(receipt.get("comparison")),
                family_integer_deltas=tuple(
                    receipt.get("family_integer_deltas", ())
                ),  # type: ignore[arg-type]
                passed=receipt.get("passed"),  # type: ignore[arg-type]
            )
            if list(decision.family_integer_deltas) != receipt.get(
                "family_integer_deltas"
            ) or (
                decision.comparison.item_count != BLOCK_ITEM_COUNT
                or len(decision.family_integer_deltas) != 3
                or any(type(row) is not int for row in decision.family_integer_deltas)
                or decision.passed
                is not (
                    decision.comparison.total_integer_delta > 0
                    and all(value >= 0 for value in decision.family_integer_deltas)
                    and sum(value > 0 for value in decision.family_integer_deltas)
                    >= 2
                    and decision.comparison.tail_at_most_alpha
                )
            ):
                raise BircoP1FormalControllerError("M_search decision binding drifted")
            return decision, receipt_sha
        qrels = self._qrels_after_archive(
            block="M_search",
            archive=archive,
            promotion_sha256=promotion_sha256,
        )
        labels = self._qrel_rows(archive, qrels)
        e4: list[int] = []
        e0: list[int] = []
        families: list[str] = []
        for action, (family, relevance) in zip(archive["items"], labels):
            families.append(family)
            e4.append(
                core.score_full_permutation(
                    action["E4_ranking"], relevance
                ).integer_utility
            )
            e0.append(
                core.score_full_permutation(
                    action["E0_ranking"], relevance
                ).integer_utility
            )
        decision = core.decide_m_search_e4_improvement(e4, e0, families)
        receipt = _self_hashed(
            {
                "schema": f"{VERSION}_M_search_score_v1",
                "version": VERSION,
                "study_id": STUDY_ID,
                "status": "offline_untouched_M_search_complete",
                "M_search_action_archive_sha256": archive["archive_sha256"],
                "M_search_qrel_pack_sha256": qrels["qrel_pack_sha256"],
                "A_hold_promotion_receipt_sha256": promotion_sha256,
                "comparison": decision.comparison.payload(),
                "family_integer_deltas": list(decision.family_integer_deltas),
                "passed": decision.passed,
                "online_evaluator_call_count": 0,
            },
            "score_receipt_sha256",
        )
        receipt_sha = _seal_read_only(path, receipt, "score_receipt_sha256")
        return decision, receipt_sha

    def _final(
        self,
        *,
        status: str,
        stage_archive_sha256s: Mapping[str, str],
        f_identifiability_passed: bool | None,
        reality_primary_passed: bool | None,
        e4_promoted: bool | None,
        m_search_passed: bool | None,
        evidence_sha256s: Mapping[str, str],
    ) -> dict[str, Any]:
        path = self.control_root / "formal_terminal.json"
        receipt = _self_hashed(
            {
                "schema": FINAL_SCHEMA,
                "version": VERSION,
                "study_id": STUDY_ID,
                "status": status,
                "execution_freeze_sha256": self.bindings.execution_freeze_self_sha256,
                "stage_archive_sha256s": dict(stage_archive_sha256s),
                "evidence_sha256s": dict(evidence_sha256s),
                "F_search_identifiability_passed": f_identifiability_passed,
                "A_hold_reality_primary_passed": reality_primary_passed,
                "A_hold_E4_promoted": e4_promoted,
                "M_search_E4_improvement_passed": m_search_passed,
                "F_search_qrel_open_count": 0,
                "online_evaluator_call_count": 0,
                "retry_replay_resample_or_provider_switch_count": 0,
            },
            "final_receipt_sha256",
        )
        _seal_read_only(path, receipt, "final_receipt_sha256")
        return receipt

    def _write_stage_failure(self, block: str, exc: BaseException) -> None:
        path = self._stage_root(block) / "stage.failure.json"
        if path.exists() or path.is_symlink():
            return
        failure = _self_hashed(
            {
                "schema": FAILURE_SCHEMA,
                "version": VERSION,
                "study_id": STUDY_ID,
                "failure_scope": "stage",
                "block": block,
                "exception_type_sha256": _exception_hash(exc),
                "formal_retry_authorized": False,
                "online_evaluator_fallback_authorized": False,
            },
            "failure_sha256",
        )
        _atomic_write_once(path, failure, final_mode=0o400)

    def run(self) -> dict[str, Any]:
        """Run the frozen lifecycle, returning only a terminal aggregate receipt."""

        if (self.control_root / "formal_terminal.json").exists():
            raise BircoP1FormalControllerError("formal lifecycle already terminated")
        self._verify_before_execution()
        archives: dict[str, str] = {}
        evidence: dict[str, str] = {}
        current = "A_form"
        try:
            a_form = self._materialize_stage(block="A_form", e4_model=None)
            archives["A_form"] = a_form["archive_sha256"]
            model, model_sha = self._load_or_fit_e4(a_form)
            evidence["A_form_E4_model"] = model_sha

            current = "F_search"
            f_search = self._materialize_stage(block="F_search", e4_model=model)
            archives["F_search"] = f_search["archive_sha256"]
            f_result, f_sha = self._f_identifiability(f_search, model_sha)
            evidence["F_search_identifiability"] = f_sha
            if not f_result.passed:
                return self._final(
                    status="terminal_F_search_label_free_unidentifiable",
                    stage_archive_sha256s=archives,
                    f_identifiability_passed=False,
                    reality_primary_passed=None,
                    e4_promoted=None,
                    m_search_passed=None,
                    evidence_sha256s=evidence,
                )

            current = "A_hold"
            a_hold = self._materialize_stage(block="A_hold", e4_model=model)
            archives["A_hold"] = a_hold["archive_sha256"]
            reality, promotion, promotion_sha = self._score_a_hold(
                a_hold,
                f_passed=True,
                f_receipt_sha256=f_sha,
            )
            evidence["A_hold_score_and_promotion"] = promotion_sha
            if not promotion.promoted:
                return self._final(
                    status="terminal_A_hold_E4_not_promoted",
                    stage_archive_sha256s=archives,
                    f_identifiability_passed=True,
                    reality_primary_passed=reality.passed,
                    e4_promoted=False,
                    m_search_passed=None,
                    evidence_sha256s=evidence,
                )

            current = "M_search"
            m_search = self._materialize_stage(block="M_search", e4_model=model)
            archives["M_search"] = m_search["archive_sha256"]
            m_decision, m_sha = self._score_m_search(
                m_search, promotion_sha256=promotion_sha
            )
            evidence["M_search_score"] = m_sha
            return self._final(
                status="formal_lifecycle_complete",
                stage_archive_sha256s=archives,
                f_identifiability_passed=True,
                reality_primary_passed=reality.passed,
                e4_promoted=True,
                m_search_passed=m_decision.passed,
                evidence_sha256s=evidence,
            )
        except BaseException as exc:
            self._write_stage_failure(current, exc)
            raise


__all__ = [
    "VERSION",
    "STUDY_ID",
    "IMPLEMENTATION_FREEZE_SCHEMA",
    "EXECUTION_FREEZE_SCHEMA",
    "STAGE_ARCHIVE_SCHEMA",
    "ATTEMPT_CLAIM_SCHEMA",
    "HIPPO_TERMINAL_SCHEMA",
    "FAILURE_SCHEMA",
    "FINAL_SCHEMA",
    "BLOCK_ORDER",
    "BLOCK_ARMS",
    "MAX_AGENT_API_WORKERS",
    "MAX_RAW_API_WORKERS",
    "MAX_TOTAL_API_CALLS",
    "MAX_HIPPORAG_WORKERS",
    "BircoP1FormalControllerError",
    "SemanticExecutor",
    "HippoExecutor",
    "QrelOpener",
    "ConcurrencyPolicy",
    "PrerunBindings",
    "stable_hash",
    "verify_prerun_freezes",
    "FormalController",
]
