"""One-attempt top-level orchestration for the frozen QuAC P1 study.

The runner owns lifecycle ordering and durable custody, not source decoding,
model implementation, or scientific policy.  Its formal input is an already
strict-decoded train/dev pair.  It creates one attempt marker and one 32-byte
secret, then executes exactly:

``A_form label-free runtime -> late labels -> one fit -> A_hold four-arm
barrier -> late labels -> offline score -> conditional M_search``.

``M_search`` is never projected into runtime material unless the frozen
A_hold promotion proof has produced and consumed one authenticated broker
capability.  Every stage is injectable for mock-only tests.  The default
bridges call the frozen acquisition, runtime, action-adapter, official
HippoRAG, RJMC evaluator, and scientific-controller modules.

This module performs no source-file, network, API, or online-evaluator access.
The only I/O it owns is exclusive private/safe archive creation under a fresh
attempt root.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence

from assumption_agent.benchmarks import quac_p1_action_adapter_v1 as action
from assumption_agent.benchmarks import (
    quac_p1_formal_acquisition_v1 as acquisition,
)
from assumption_agent.benchmarks import (
    quac_p1_formal_controller_v1 as controller,
)
from assumption_agent.benchmarks import quac_p1_runtime_v1 as runtime
from assumption_agent.benchmarks import quac_rjmc_evaluator_v1 as evaluator
from replication_runtime.quac_p1_official_v1 import contract as official_contract


VERSION = "quac_p1_formal_runner_v1"
STUDY_ID = acquisition.STUDY_ID
EFFECT_DESIGN_SELF_SHA256 = (
    "def417300b3c25f127517eef1cdd61760757762f08cc5a9b9877b261036dace2"
)

ATTEMPT_SCHEMA = f"{VERSION}_private_attempt_v1"
SAFE_TERMINAL_SCHEMA = f"{VERSION}_safe_terminal_v1"
SAFE_FAILURE_SCHEMA = f"{VERSION}_safe_failure_v1"
FORMAL_CONFIG_SCHEMA = f"{VERSION}_formal_config_v1"
OUTER_ATTEMPT_SCHEMA = f"{VERSION}_outer_attempt_v1"
OUTER_TERMINAL_SCHEMA = f"{VERSION}_outer_safe_terminal_v1"
OUTER_FAILURE_SCHEMA = f"{VERSION}_outer_safe_failure_v1"

ATTEMPT_FILENAME = "formal_attempt.private.json"
SECRET_FILENAME = "selection_secret.private.bin"
TERMINAL_FILENAME = "formal_terminal.safe.json"
FAILURE_FILENAME = "formal_failure.safe.json"
OUTER_ATTEMPT_FILENAME = "outer_attempt.private.json"
OUTER_TERMINAL_FILENAME = "outer_terminal.safe.json"
OUTER_FAILURE_FILENAME = "outer_failure.safe.json"

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_PINNED_SOURCE_SHA256 = {
    "train": "ff5cca5a2e4b4d1cb5b5ced68b9fce88394ef6d93117426d6d4baafbcc05c56a",
    "dev": "09e622916280ba04c9352acb1bc5bbe80f11a2598f6f34e934c51d9e6570f378",
}
_PINNED_SOURCE_SIZE = {
    "train": 68_114_819,
    "dev": 8_929_167,
}
_CUSTODY_INCIDENT_SELF_SHA256 = (
    "ca219ae09314064f1126549f8092f56bf10f7e96f12131bcd26c04cf2d416494"
)


class QuacP1FormalRunnerError(RuntimeError):
    """The one-shot lifecycle, injected boundary, or archive drifted."""


def canonical_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
            + b"\n"
        )
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise QuacP1FormalRunnerError(
            "runner value is not canonical JSON"
        ) from exc


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value).rstrip(b"\n")).hexdigest()


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze_json(child)
                for key, child in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(child) for child in value)
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise QuacP1FormalRunnerError(
        "scientific payload contains a non-JSON value"
    )


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _thaw_json(child)
            for key, child in value.items()
        }
    if isinstance(value, tuple):
        return [_thaw_json(child) for child in value]
    return value


def _require_sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise QuacP1FormalRunnerError(
            f"{field_name} must be a lowercase SHA256"
        )
    return value


def _self_hashed(body: Mapping[str, object]) -> dict[str, object]:
    if "self_sha256" in body:
        raise QuacP1FormalRunnerError(
            "self-hash field already exists"
        )
    result = dict(body)
    result["self_sha256"] = stable_hash(result)
    return result


def _private_directory(path: Path, *, fresh: bool) -> None:
    try:
        path.mkdir(parents=True, mode=0o700, exist_ok=not fresh)
        os.chmod(path, 0o700)
        metadata = path.lstat()
    except OSError as exc:
        raise QuacP1FormalRunnerError(
            "private directory cannot be created"
        ) from exc
    if (
        path.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise QuacP1FormalRunnerError(
            "private directory identity drifted"
        )


def _write_once(path: Path, raw: bytes, *, mode: int) -> str:
    if mode not in (0o400, 0o600):
        raise QuacP1FormalRunnerError("archive mode is invalid")
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    try:
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, mode)
        metadata = path.lstat()
        observed = path.read_bytes()
    except OSError as exc:
        raise QuacP1FormalRunnerError(
            "archive cannot be created exactly once"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != mode
        or observed != raw
    ):
        raise QuacP1FormalRunnerError(
            "archive verification failed"
        )
    return hashlib.sha256(raw).hexdigest()


def _write_json_once(
    path: Path,
    value: Mapping[str, object],
    *,
    mode: int = 0o400,
) -> str:
    return _write_once(path, canonical_bytes(value), mode=mode)


def _reject_duplicate_pairs(
    pairs: Sequence[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise QuacP1FormalRunnerError(
                "JSON contains a duplicate object key"
            )
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise QuacP1FormalRunnerError(
        f"JSON contains a forbidden numeric constant: {value}"
    )


def _read_direct_bytes_once(path: Path, *, field_name: str) -> bytes:
    if not isinstance(path, Path) or not path.is_absolute():
        raise QuacP1FormalRunnerError(
            f"{field_name} path must be absolute"
        )
    if not hasattr(os, "O_NOFOLLOW"):
        raise QuacP1FormalRunnerError(
            f"{field_name} requires O_NOFOLLOW"
        )
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
        )
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        pathname = path.lstat()
    except OSError as exc:
        raise QuacP1FormalRunnerError(
            f"{field_name} direct read failed"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        or not stat.S_ISREG(pathname.st_mode)
        or pathname.st_nlink != 1
        or (
            pathname.st_dev,
            pathname.st_ino,
            pathname.st_size,
            pathname.st_mtime_ns,
        )
        != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
    ):
        raise QuacP1FormalRunnerError(
            f"{field_name} direct-file identity drifted"
        )
    return b"".join(chunks)


def _strict_json_bytes(
    raw: bytes,
    *,
    field_name: str,
    require_canonical: bool,
) -> Mapping[str, object]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise QuacP1FormalRunnerError(
            f"{field_name} is not strict JSON"
        ) from exc
    if not isinstance(value, Mapping):
        raise QuacP1FormalRunnerError(
            f"{field_name} root is not an object"
        )
    if require_canonical and raw not in {
        canonical_bytes(value),
        canonical_bytes(value).rstrip(b"\n"),
    }:
        raise QuacP1FormalRunnerError(
            f"{field_name} is not canonical JSON"
        )
    return value


def _exact_object(
    value: object,
    *,
    keys: frozenset[str],
    field_name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise QuacP1FormalRunnerError(
            f"{field_name} key set drifted"
        )
    return value


def _absolute_path(value: object, field_name: str) -> Path:
    if not isinstance(value, str):
        raise QuacP1FormalRunnerError(
            f"{field_name} must be an absolute path"
        )
    path = Path(value)
    if not path.is_absolute():
        raise QuacP1FormalRunnerError(
            f"{field_name} must be an absolute path"
        )
    return path


@dataclass(frozen=True, slots=True)
class FrozenFileConfig:
    path: Path
    file_sha256: str
    self_sha256: str | None = None

    @classmethod
    def from_payload(
        cls,
        value: object,
        *,
        field_name: str,
        with_self_hash: bool,
    ) -> "FrozenFileConfig":
        keys = {"file_sha256", "path"}
        if with_self_hash:
            keys.add("self_sha256")
        checked = _exact_object(
            value,
            keys=frozenset(keys),
            field_name=field_name,
        )
        return cls(
            path=_absolute_path(checked["path"], field_name),
            file_sha256=_require_sha256(
                checked["file_sha256"],
                f"{field_name} file SHA256",
            ),
            self_sha256=(
                _require_sha256(
                    checked["self_sha256"],
                    f"{field_name} self SHA256",
                )
                if with_self_hash
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class FormalSourceMember:
    role: str
    path: Path
    file_sha256: str
    size_bytes: int

    @classmethod
    def from_payload(
        cls,
        role: str,
        value: object,
    ) -> "FormalSourceMember":
        checked = _exact_object(
            value,
            keys=frozenset(
                {"file_sha256", "path", "size_bytes"}
            ),
            field_name=f"{role} source",
        )
        size = checked["size_bytes"]
        if (
            role not in _PINNED_SOURCE_SHA256
            or checked["file_sha256"]
            != _PINNED_SOURCE_SHA256[role]
            or size != _PINNED_SOURCE_SIZE[role]
        ):
            raise QuacP1FormalRunnerError(
                f"{role} source binding is not the frozen official member"
            )
        return cls(
            role=role,
            path=_absolute_path(
                checked["path"],
                f"{role} source",
            ),
            file_sha256=str(checked["file_sha256"]),
            size_bytes=int(size),
        )


def _tree_binding_from_payload(
    value: object,
    *,
    field_name: str,
) -> runtime.FrozenTreeBinding:
    checked = _exact_object(
        value,
        keys=frozenset(
            {"file_count", "path", "total_bytes", "tree_sha256"}
        ),
        field_name=field_name,
    )
    return runtime.FrozenTreeBinding(
        path=str(_absolute_path(checked["path"], field_name)),
        tree_sha256=_require_sha256(
            checked["tree_sha256"],
            f"{field_name} tree SHA256",
        ),
        file_count=checked["file_count"],  # type: ignore[arg-type]
        total_bytes=checked["total_bytes"],  # type: ignore[arg-type]
    )


def _executable_binding_from_payload(
    value: object,
    *,
    field_name: str,
) -> runtime.FrozenExecutableBinding:
    checked = _exact_object(
        value,
        keys=frozenset(
            {"path", "realpath", "sha256", "size_bytes"}
        ),
        field_name=field_name,
    )
    return runtime.FrozenExecutableBinding(
        path=str(_absolute_path(checked["path"], field_name)),
        realpath=str(
            _absolute_path(
                checked["realpath"],
                f"{field_name} realpath",
            )
        ),
        sha256=_require_sha256(
            checked["sha256"],
            f"{field_name} SHA256",
        ),
        size_bytes=checked["size_bytes"],  # type: ignore[arg-type]
    )


def _python_binding_from_payload(
    value: object,
    *,
    field_name: str,
) -> runtime.PythonRuntimeBinding:
    checked = _exact_object(
        value,
        keys=frozenset(
            {"executable", "identity_sha256", "import_tree"}
        ),
        field_name=field_name,
    )
    return runtime.PythonRuntimeBinding(
        executable=_executable_binding_from_payload(
            checked["executable"],
            field_name=f"{field_name} executable",
        ),
        import_tree=_tree_binding_from_payload(
            checked["import_tree"],
            field_name=f"{field_name} import tree",
        ),
        identity_sha256=_require_sha256(
            checked["identity_sha256"],
            f"{field_name} identity",
        ),
    )


def _runtime_bindings_from_payload(
    value: object,
) -> runtime.RuntimeBindings:
    checked = _exact_object(
        value,
        keys=frozenset(
            {
                "gpu0_python",
                "gpu1_base_import_tree",
                "gpu1_python",
                "gpu1_overlay_import_tree",
                "hipporag_source",
                "llm_alias",
                "llm_asset",
                "minilm_alias",
                "minilm_asset",
            }
        ),
        field_name="runtime bindings",
    )
    if not isinstance(checked["minilm_alias"], str) or not isinstance(
        checked["llm_alias"],
        str,
    ):
        raise QuacP1FormalRunnerError(
            "runtime model aliases must be text"
        )
    return runtime.RuntimeBindings(
        gpu0_python=_python_binding_from_payload(
            checked["gpu0_python"],
            field_name="GPU0 Python",
        ),
        gpu1_python=_python_binding_from_payload(
            checked["gpu1_python"],
            field_name="GPU1 Python",
        ),
        gpu1_base_import_tree=_tree_binding_from_payload(
            checked["gpu1_base_import_tree"],
            field_name="GPU1 base import tree",
        ),
        gpu1_overlay_import_tree=_tree_binding_from_payload(
            checked["gpu1_overlay_import_tree"],
            field_name="GPU1 overlay import tree",
        ),
        minilm_asset=_tree_binding_from_payload(
            checked["minilm_asset"],
            field_name="MiniLM asset",
        ),
        llm_asset=_tree_binding_from_payload(
            checked["llm_asset"],
            field_name="LLM asset",
        ),
        hipporag_source=_tree_binding_from_payload(
            checked["hipporag_source"],
            field_name="HippoRAG source",
        ),
        minilm_alias=checked["minilm_alias"],
        llm_alias=checked["llm_alias"],
    )


@dataclass(frozen=True, slots=True)
class FormalProductionConfig:
    self_sha256: str
    formal_root: Path
    effect_design: FrozenFileConfig
    custody_incident: FrozenFileConfig
    implementation_files: Mapping[str, FrozenFileConfig]
    service_unit_name: str
    service_unit: FrozenFileConfig
    train: FormalSourceMember
    dev: FormalSourceMember
    runtime_bindings: runtime.RuntimeBindings

    @classmethod
    def from_payload(
        cls,
        value: object,
    ) -> "FormalProductionConfig":
        checked = _exact_object(
            value,
            keys=frozenset(
                {
                    "custody_incident_file",
                    "effect_design_file",
                    "formal_root",
                    "hash_only_custody_counts",
                    "implementation_files",
                    "runtime_bindings",
                    "schema",
                    "self_sha256",
                    "service_unit",
                    "source_inputs",
                    "study_id",
                }
            ),
            field_name="formal config",
        )
        if (
            checked["schema"] != FORMAL_CONFIG_SCHEMA
            or checked["study_id"] != STUDY_ID
        ):
            raise QuacP1FormalRunnerError(
                "formal config identity drifted"
            )
        self_sha = _require_sha256(
            checked["self_sha256"],
            "formal config self SHA256",
        )
        body = dict(checked)
        body.pop("self_sha256")
        if stable_hash(body) != self_sha:
            raise QuacP1FormalRunnerError(
                "formal config self hash drifted"
            )
        counts = _exact_object(
            checked["hash_only_custody_counts"],
            keys=frozenset(
                {
                    "postqualification_hash_only_member_read_count",
                    "postqualification_hash_only_operation_count",
                    "semantic_decode_before_formal_count",
                }
            ),
            field_name="hash-only custody counts",
        )
        if dict(counts) != {
            "postqualification_hash_only_member_read_count": 2,
            "postqualification_hash_only_operation_count": 1,
            "semantic_decode_before_formal_count": 0,
        }:
            raise QuacP1FormalRunnerError(
                "hash-only custody disclosure drifted"
            )
        implementations = checked["implementation_files"]
        required_names = {
            "acquisition",
            "action_adapter",
            "controller",
            "evaluator",
            "official_contract",
            "runner",
            "runtime",
        }
        if (
            not isinstance(implementations, Mapping)
            or set(implementations) != required_names
        ):
            raise QuacP1FormalRunnerError(
                "implementation binding registry drifted"
            )
        service = _exact_object(
            checked["service_unit"],
            keys=frozenset(
                {"file_sha256", "path", "unit_name"}
            ),
            field_name="service unit",
        )
        if (
            not isinstance(service["unit_name"], str)
            or not service["unit_name"].endswith(".service")
        ):
            raise QuacP1FormalRunnerError(
                "service unit name drifted"
            )
        sources = _exact_object(
            checked["source_inputs"],
            keys=frozenset({"dev", "train"}),
            field_name="source inputs",
        )
        effect = FrozenFileConfig.from_payload(
            checked["effect_design_file"],
            field_name="effect design",
            with_self_hash=True,
        )
        incident = FrozenFileConfig.from_payload(
            checked["custody_incident_file"],
            field_name="custody incident",
            with_self_hash=True,
        )
        if (
            effect.self_sha256 != EFFECT_DESIGN_SELF_SHA256
            or incident.self_sha256
            != _CUSTODY_INCIDENT_SELF_SHA256
        ):
            raise QuacP1FormalRunnerError(
                "effect or custody incident binding drifted"
            )
        return cls(
            self_sha256=self_sha,
            formal_root=_absolute_path(
                checked["formal_root"],
                "formal root",
            ),
            effect_design=effect,
            custody_incident=incident,
            implementation_files=MappingProxyType(
                {
                    name: FrozenFileConfig.from_payload(
                        implementations[name],
                        field_name=f"{name} implementation",
                        with_self_hash=False,
                    )
                    for name in sorted(required_names)
                }
            ),
            service_unit_name=str(service["unit_name"]),
            service_unit=FrozenFileConfig(
                path=_absolute_path(
                    service["path"],
                    "service unit",
                ),
                file_sha256=_require_sha256(
                    service["file_sha256"],
                    "service unit SHA256",
                ),
            ),
            train=FormalSourceMember.from_payload(
                "train",
                sources["train"],
            ),
            dev=FormalSourceMember.from_payload(
                "dev",
                sources["dev"],
            ),
            runtime_bindings=_runtime_bindings_from_payload(
                checked["runtime_bindings"]
            ),
        )

    @classmethod
    def from_path(cls, path: Path) -> "FormalProductionConfig":
        raw = _read_direct_bytes_once(
            path,
            field_name="formal config",
        )
        return cls.from_payload(
            _strict_json_bytes(
                raw,
                field_name="formal config",
                require_canonical=True,
            )
        )


def runtime_block_from_broker(
    broker: acquisition.TrustedAcquisitionBroker,
    block: str,
) -> runtime.RuntimeBlock:
    """Bridge the broker's separated label-free material into the runtime."""

    if not isinstance(broker, acquisition.TrustedAcquisitionBroker):
        raise QuacP1FormalRunnerError(
            "runtime block bridge requires the trusted broker"
        )
    material = broker.runtime_material(block)
    documents = tuple(
        action.BlockDocument(
            unit_id=row.unit_id,
            context_id=row.context_id,
            title=row.title,
            section_title=row.section_title,
            context_window_ordinal=row.context_window_ordinal,
            text=row.text,
        )
        for row in material.documents
    )
    queries = tuple(
        runtime.RuntimeQuery(
            query_id=row.query_id,
            question_turns=tuple(
                action.QuestionTurn(question)
                for question in row.question_turns
            ),
        )
        for row in material.queries
    )
    return runtime.RuntimeBlock(
        block_id=material.block_id,
        documents=documents,
        queries=queries,
    )


class BlockExecutorProtocol(Protocol):
    """One injected call to the frozen label-free runtime."""

    def __call__(
        self,
        *,
        block_name: str,
        block: runtime.RuntimeBlock,
        work_root: Path,
        official_required: bool,
    ) -> runtime.BlockRuntimeResult: ...


@dataclass
class BoundRuntimeExecutor:
    """Production bridge to ``runtime.run_block`` and its official lane."""

    bindings: runtime.RuntimeBindings
    verified_bindings: runtime.VerifiedRuntimeBindings
    encoder: runtime.MiniLMEncoderProtocol
    official_lane: runtime.OfficialLaneProtocol
    action_adapter: runtime.ActionAdapterProtocol | None = None

    def __call__(
        self,
        *,
        block_name: str,
        block: runtime.RuntimeBlock,
        work_root: Path,
        official_required: bool,
    ) -> runtime.BlockRuntimeResult:
        if block_name not in acquisition.BLOCK_ORDER:
            raise QuacP1FormalRunnerError(
                "runtime executor block drifted"
            )
        if official_required is not (
            block_name in runtime.OFFICIAL_BLOCK_ROLES
        ):
            raise QuacP1FormalRunnerError(
                "runtime executor official role drifted"
            )
        return runtime.run_block(
            block_role=block_name,
            block=block,
            work_root=work_root,
            bindings=self.bindings,
            verified_bindings=self.verified_bindings,
            encoder=self.encoder,
            official_lane=(
                self.official_lane if official_required else None
            ),
            action_adapter=self.action_adapter,
        )


@dataclass(frozen=True)
class FittedEvaluator:
    model: object = field(repr=False)
    parameter_sha256: str
    parameter_archive: bytes = field(repr=False)

    def __post_init__(self) -> None:
        _require_sha256(
            self.parameter_sha256,
            "model parameter SHA256",
        )
        if (
            type(self.parameter_archive) is not bytes
            or not self.parameter_archive
            or hashlib.sha256(self.parameter_archive).hexdigest()
            != self.parameter_sha256
        ):
            raise QuacP1FormalRunnerError(
                "model parameter archive does not match its commitment"
            )


def _model_parameter_archive(model: object) -> bytes:
    """Serialize exactly the bytes hashed by the frozen evaluator."""

    if not isinstance(model, evaluator.JackknifeMinimaxComparator):
        raise QuacP1FormalRunnerError(
            "formal model archive requires the frozen RJMC type"
        )
    payload = bytearray()
    for name, parameter in sorted(model.named_parameters()):
        encoded = name.encode("ascii")
        values = (
            parameter.detach()
            .cpu()
            .to(dtype=evaluator.torch.float64)
            .contiguous()
            .numpy()
            .astype("<f8", copy=False)
        )
        payload.extend(len(encoded).to_bytes(4, "big"))
        payload.extend(encoded)
        payload.extend(tuple(values.shape).__repr__().encode("ascii"))
        payload.extend(values.tobytes(order="C"))
    raw = bytes(payload)
    if (
        not raw
        or hashlib.sha256(raw).hexdigest()
        != evaluator.model_parameter_sha256(model)
    ):
        raise QuacP1FormalRunnerError(
            "formal model parameter archive drifted"
        )
    return raw


def _assert_fitted_unchanged(fitted: FittedEvaluator) -> None:
    if isinstance(
        fitted.model,
        evaluator.JackknifeMinimaxComparator,
    ):
        current = evaluator.model_parameter_sha256(fitted.model)
    else:
        current = hashlib.sha256(fitted.parameter_archive).hexdigest()
    if current != fitted.parameter_sha256:
        raise QuacP1FormalRunnerError(
            "fitted evaluator mutated after its model seal"
        )


@dataclass(frozen=True)
class MeasurementActions:
    block: str
    native: object = field(repr=False)
    corpus_unit_ids: tuple[str, ...]
    payload: Mapping[str, object]
    action_sha256: str

    def __post_init__(self) -> None:
        if self.block not in {"A_hold", "M_search"}:
            raise QuacP1FormalRunnerError(
                "measurement action block drifted"
            )
        corpus = tuple(self.corpus_unit_ids)
        if (
            len(corpus) < evaluator.TOP_K
            or corpus != tuple(sorted(corpus))
            or len(set(corpus)) != len(corpus)
            or any(_HEX64.fullmatch(row) is None for row in corpus)
        ):
            raise QuacP1FormalRunnerError(
                "measurement corpus registry drifted"
            )
        object.__setattr__(self, "corpus_unit_ids", corpus)
        if type(self.native) is not controller.SealedStageActions:
            raise QuacP1FormalRunnerError(
                "measurement native action type is not exact"
            )
        payload = _thaw_json(_freeze_json(self.payload))
        if (
            not isinstance(payload, dict)
            or self.native.block != self.block
            or self.native.payload() != payload
            or self.native.corpus_unit_ids_sha256
            != controller.stable_hash(list(corpus))
        ):
            raise QuacP1FormalRunnerError(
                "measurement native action/payload binding drifted"
            )
        _require_sha256(self.action_sha256, "action SHA256")
        if stable_hash(payload) != self.action_sha256:
            raise QuacP1FormalRunnerError(
                "measurement action commitment drifted"
            )
        object.__setattr__(self, "payload", _freeze_json(payload))


@dataclass(frozen=True)
class MeasurementScore:
    block: str
    native: object = field(repr=False)
    safe_payload: Mapping[str, object]
    e1_minus_e0: int
    p_numerator: int
    p_denominator: int
    promoted: bool

    def __post_init__(self) -> None:
        if self.block not in {"A_hold", "M_search"}:
            raise QuacP1FormalRunnerError(
                "measurement score block drifted"
            )
        if type(self.native) is not controller.StageScore:
            raise QuacP1FormalRunnerError(
                "measurement native score type is not exact"
            )
        safe_payload = _thaw_json(_freeze_json(self.safe_payload))
        if (
            not isinstance(safe_payload, dict)
            or self.native.block != self.block
            or self.native.safe_payload() != safe_payload
        ):
            raise QuacP1FormalRunnerError(
                "measurement native score/payload binding drifted"
            )
        if (
            type(self.e1_minus_e0) is not int
            or type(self.p_numerator) is not int
            or type(self.p_denominator) is not int
            or self.p_numerator < 0
            or self.p_denominator <= 0
            or not isinstance(self.promoted, bool)
        ):
            raise QuacP1FormalRunnerError(
                "measurement exact statistic drifted"
            )
        frozen_decision = bool(
            self.e1_minus_e0 > 0
            and self.p_numerator * 10 <= self.p_denominator
        )
        if self.promoted != frozen_decision:
            raise QuacP1FormalRunnerError(
                "measurement promotion decision drifted"
            )
        comparison = self.native.comparison("E0")
        native_decision = (
            self.native.promotion
            if self.block == "A_hold"
            else self.native.l5
        )
        if (
            self.e1_minus_e0 != comparison.net
            or self.p_numerator != comparison.exact.numerator
            or self.p_denominator != comparison.exact.denominator
            or self.promoted is not native_decision
        ):
            raise QuacP1FormalRunnerError(
                "measurement statistic detached from native score"
            )
        object.__setattr__(
            self,
            "safe_payload",
            _freeze_json(safe_payload),
        )


class ScientificOpsProtocol(Protocol):
    def fit_a_form(
        self,
        items: Sequence[controller.LabelFreeGraphItem],
        labels: Sequence[controller.LateLabelRow],
        *,
        block_corpus_unit_ids: Sequence[str],
    ) -> FittedEvaluator: ...

    def select_measurement(
        self,
        *,
        block: str,
        items: Sequence[controller.LabelFreeGraphItem],
        fitted: FittedEvaluator,
        hipporag_top5: Mapping[str, Sequence[str]],
        block_corpus_unit_ids: Sequence[str],
    ) -> MeasurementActions: ...

    def score_measurement(
        self,
        actions: MeasurementActions,
        labels: Sequence[controller.LateLabelRow],
    ) -> MeasurementScore: ...

    def safe_terminal(
        self,
        *,
        a_hold: MeasurementScore,
        m_search: MeasurementScore | None,
        fitted: FittedEvaluator,
        action_commitments: Mapping[str, str],
        runtime_commitments: Mapping[str, str],
        m_materialization_count_before_promotion: int,
    ) -> Mapping[str, object]: ...


class FrozenScientificOps:
    """Exact adapter around the preregistered scientific controller."""

    def fit_a_form(
        self,
        items: Sequence[controller.LabelFreeGraphItem],
        labels: Sequence[controller.LateLabelRow],
        *,
        block_corpus_unit_ids: Sequence[str],
    ) -> FittedEvaluator:
        model = controller.fit_a_form_once(
            items,
            labels,
            block_corpus_unit_ids=block_corpus_unit_ids,
        )
        archive = _model_parameter_archive(model)
        return FittedEvaluator(
            model=model,
            parameter_sha256=evaluator.model_parameter_sha256(model),
            parameter_archive=archive,
        )

    def select_measurement(
        self,
        *,
        block: str,
        items: Sequence[controller.LabelFreeGraphItem],
        fitted: FittedEvaluator,
        hipporag_top5: Mapping[str, Sequence[str]],
        block_corpus_unit_ids: Sequence[str],
    ) -> MeasurementActions:
        if not isinstance(
            fitted.model,
            evaluator.JackknifeMinimaxComparator,
        ):
            raise QuacP1FormalRunnerError(
                "fitted RJMC model type drifted"
            )
        native = controller.select_measurement_actions(
            block=block,
            items=items,
            model=fitted.model,
            hipporag_top5=hipporag_top5,
            block_corpus_unit_ids=block_corpus_unit_ids,
        )
        payload = native.payload()
        return MeasurementActions(
            block=block,
            native=native,
            corpus_unit_ids=tuple(block_corpus_unit_ids),
            payload=payload,
            action_sha256=stable_hash(payload),
        )

    def score_measurement(
        self,
        actions: MeasurementActions,
        labels: Sequence[controller.LateLabelRow],
    ) -> MeasurementScore:
        if not isinstance(actions.native, controller.SealedStageActions):
            raise QuacP1FormalRunnerError(
                "native sealed actions type drifted"
            )
        native = controller.score_sealed_stage(
            actions.native,
            labels,
            block_corpus_unit_ids=actions.corpus_unit_ids,
        )
        comparison = native.comparison("E0")
        return MeasurementScore(
            block=native.block,
            native=native,
            safe_payload=native.safe_payload(),
            e1_minus_e0=comparison.net,
            p_numerator=comparison.exact.numerator,
            p_denominator=comparison.exact.denominator,
            promoted=native.promotion if native.block == "A_hold" else native.l5,
        )

    def safe_terminal(
        self,
        *,
        a_hold: MeasurementScore,
        m_search: MeasurementScore | None,
        fitted: FittedEvaluator,
        action_commitments: Mapping[str, str],
        runtime_commitments: Mapping[str, str],
        m_materialization_count_before_promotion: int,
    ) -> Mapping[str, object]:
        if not isinstance(a_hold.native, controller.StageScore) or (
            m_search is not None
            and not isinstance(m_search.native, controller.StageScore)
        ):
            raise QuacP1FormalRunnerError(
                "native score type drifted"
            )
        return controller.safe_terminal(
            a_hold=a_hold.native,
            m_search=(
                None
                if m_search is None
                else m_search.native
            ),
            model_parameter_sha256=fitted.parameter_sha256,
            action_commitments=action_commitments,
            runtime_commitments=runtime_commitments,
            M_materialization_count_before_promotion=(
                m_materialization_count_before_promotion
            ),
        )


def _runtime_commitment(
    result: runtime.BlockRuntimeResult,
    *,
    block_name: str,
    block: runtime.RuntimeBlock,
    official_required: bool,
) -> tuple[str, str, str]:
    if not isinstance(result, runtime.BlockRuntimeResult):
        raise QuacP1FormalRunnerError(
            "block executor result type drifted"
        )
    receipt = result.safe_receipt
    expected_keys = {
        "API_or_online_evaluation_call_count",
        "action_count",
        "action_pack_file_sha256",
        "asset_binding_sha256",
        "attempt_count",
        "attempt_file_sha256",
        "binding_verification_token_sha256",
        "block_input_file_sha256",
        "block_role",
        "corpus_count",
        "index_cleanup",
        "label_family_qrel_or_answer_input_count",
        "logical_action_query_count",
        "max_concurrent_physical_model_lanes",
        "minilm_encode_call_count",
        "minilm_receipt_file_sha256",
        "official_full_rankings_sha256",
        "official_index_call_count",
        "official_output_file_sha256",
        "official_required",
        "official_retrieve_call_count",
        "parallel_submission_barrier_passed",
        "query_count",
        "retry_replay_resample_or_fallback_count",
        "schema",
        "self_sha256",
        "status",
        "unique_embedding_count",
    }
    index_cleanup = (
        receipt.get("index_cleanup")
        if isinstance(receipt, Mapping)
        else None
    )
    if (
        not isinstance(receipt, Mapping)
        or set(receipt) != expected_keys
        or receipt.get("schema") != runtime.SAFE_RESULT_SCHEMA
        or receipt.get("status") != "passed_label_free_block_runtime"
        or receipt.get("attempt_count") != 1
        or receipt.get("block_role") != block_name
        or receipt.get("official_required") is not official_required
        or receipt.get("API_or_online_evaluation_call_count") != 0
        or receipt.get("retry_replay_resample_or_fallback_count") != 0
        or receipt.get("label_family_qrel_or_answer_input_count") != 0
        or receipt.get("action_count") != len(result.actions)
        or receipt.get("logical_action_query_count")
        != len(block.queries)
        or receipt.get("query_count") != len(block.queries)
        or receipt.get("corpus_count") != len(block.documents)
        or receipt.get("minilm_encode_call_count") != 1
        or receipt.get("max_concurrent_physical_model_lanes")
        != (2 if official_required else 1)
        or receipt.get("official_index_call_count")
        != (1 if official_required else 0)
        or receipt.get("official_retrieve_call_count")
        != (1 if official_required else 0)
        or receipt.get("parallel_submission_barrier_passed")
        != (True if official_required else None)
        or not isinstance(receipt.get("unique_embedding_count"), int)
        or receipt["unique_embedding_count"] <= 0
        or not isinstance(index_cleanup, Mapping)
        or index_cleanup.get("cleanup_verified") is not True
    ):
        raise QuacP1FormalRunnerError(
            "safe runtime receipt drifted"
        )
    for key in (
        "action_pack_file_sha256",
        "asset_binding_sha256",
        "attempt_file_sha256",
        "binding_verification_token_sha256",
        "block_input_file_sha256",
        "minilm_receipt_file_sha256",
    ):
        _require_sha256(receipt.get(key), f"runtime {key}")
    if official_required:
        for key in (
            "official_full_rankings_sha256",
            "official_output_file_sha256",
        ):
            _require_sha256(receipt.get(key), f"runtime {key}")
        if (
            set(index_cleanup)
            != {
                "cleanup_verified",
                "file_count",
                "total_bytes",
                "tree_sha256",
            }
            or type(index_cleanup.get("file_count")) is not int
            or index_cleanup["file_count"] < 1
            or type(index_cleanup.get("total_bytes")) is not int
            or index_cleanup["total_bytes"] < 0
        ):
            raise QuacP1FormalRunnerError(
                "official runtime cleanup receipt drifted"
            )
        _require_sha256(
            index_cleanup.get("tree_sha256"),
            "official runtime index tree",
        )
    elif (
        dict(index_cleanup)
        != {
            "cleanup_verified": True,
            "file_count": 0,
            "total_bytes": 0,
            "tree_sha256": None,
        }
        or receipt.get("official_full_rankings_sha256") is not None
        or receipt.get("official_output_file_sha256") is not None
    ):
        raise QuacP1FormalRunnerError(
            "A_form runtime official lane residue drifted"
        )
    self_sha = _require_sha256(
        receipt.get("self_sha256"),
        "runtime receipt self SHA256",
    )
    body = dict(receipt)
    body.pop("self_sha256")
    if runtime.stable_hash(body) != self_sha:
        raise QuacP1FormalRunnerError(
            "runtime receipt self hash drifted"
        )
    return (
        self_sha,
        str(receipt["binding_verification_token_sha256"]),
        str(receipt["asset_binding_sha256"]),
    )


def _runtime_action_payload(
    *,
    block: runtime.RuntimeBlock,
    result: runtime.BlockRuntimeResult,
) -> dict[str, object]:
    if set(result.actions) != {row.query_id for row in block.queries}:
        raise QuacP1FormalRunnerError(
            "runtime block/action registries drifted"
        )
    rows = []
    for query in block.queries:
        action_result = result.actions[query.query_id]
        if not isinstance(action_result, action.ActionAdapterResult):
            raise QuacP1FormalRunnerError(
                "runtime action graph type drifted"
            )
        payload = action.canonical_action_payload(action_result)
        rows.append(
            {
                "action": payload,
                "action_sha256": runtime.stable_hash(payload),
                "query_id": query.query_id,
            }
        )
    return {
        "block_id": block.block_id,
        "rows": rows,
        "schema": runtime.ACTION_PACK_SCHEMA,
    }


def _graph_items(
    result: runtime.BlockRuntimeResult,
    *,
    folds: Mapping[str, int] | None,
) -> tuple[controller.LabelFreeGraphItem, ...]:
    if not isinstance(result.actions, Mapping) or not result.actions:
        raise QuacP1FormalRunnerError(
            "runtime action registry is empty"
        )
    expected_ids = set(result.actions)
    if folds is not None and set(folds) != expected_ids:
        raise QuacP1FormalRunnerError(
            "A_form fold and action registries drifted"
        )
    rows = []
    for item_id in sorted(result.actions):
        action_result = result.actions[item_id]
        if not isinstance(action_result, action.ActionAdapterResult):
            raise QuacP1FormalRunnerError(
                "runtime action graph type drifted"
            )
        rows.append(
            controller.LabelFreeGraphItem(
                item_id=item_id,
                fold=(0 if folds is None else folds[item_id]),
                graph=action_result.graph,
                raw_top5=action_result.raw_top5,
            )
        )
    return tuple(rows)


def _controller_labels(
    pack: acquisition.LabelPack,
) -> tuple[controller.LateLabelRow, ...]:
    if not isinstance(pack, acquisition.LabelPack):
        raise QuacP1FormalRunnerError(
            "late-label pack type drifted"
        )
    rows = tuple(
        sorted(
            (
                controller.LateLabelRow(
                    item_id=row.work_id.removeprefix(
                        "quac-work-v1-"
                    ),
                    family=row.family,
                    previous_qrel=(
                        row.previous_turn_orig_answer
                    ),
                    current_qrel=row.current_turn_orig_answer,
                )
                for row in pack.rows
            ),
            key=lambda row: row.item_id,
        )
    )
    return rows


def _open_labels_after_seal(
    *,
    broker: acquisition.TrustedAcquisitionBroker,
    block: str,
    action_path: Path,
    expected_action_payload: Mapping[str, object],
    private_root: Path,
) -> tuple[controller.LateLabelRow, ...]:
    capability = broker.register_durable_action_barrier(
        block=block,
        action_path=action_path,
        expected_payload=expected_action_payload,
    )
    _write_json_once(
        private_root / f"{block}.label_capability.private.json",
        capability.payload(),
    )
    pack = broker.open_late_labels(capability)
    _write_once(
        private_root / f"{block}.labels.private.json",
        pack.canonical_bytes(),
        mode=0o400,
    )
    return _controller_labels(pack)


def _require_m_reservation_state(
    broker: acquisition.TrustedAcquisitionBroker,
    *,
    materialization_count: int,
    materialized_path_count: int,
) -> Mapping[str, object]:
    receipt = broker.m_reservation_receipt()
    if (
        receipt.get("materialization_count")
        != materialization_count
        or receipt.get("materialized_path_count")
        != materialized_path_count
        or "rows" in receipt
    ):
        raise QuacP1FormalRunnerError(
            "live M_search reservation state drifted"
        )
    return receipt


@dataclass(frozen=True)
class FormalRunResult:
    terminal: Mapping[str, object]
    terminal_path: Path


def _verify_frozen_file(
    binding: FrozenFileConfig,
    *,
    field_name: str,
    require_self_sha256: bool,
) -> Mapping[str, object] | None:
    raw = _read_direct_bytes_once(
        binding.path,
        field_name=field_name,
    )
    if hashlib.sha256(raw).hexdigest() != binding.file_sha256:
        raise QuacP1FormalRunnerError(
            f"{field_name} file SHA256 drifted"
        )
    if not require_self_sha256:
        return None
    value = _strict_json_bytes(
        raw,
        field_name=field_name,
        require_canonical=False,
    )
    if (
        value.get("self_sha256") != binding.self_sha256
        or stable_hash(
            {
                key: child
                for key, child in value.items()
                if key != "self_sha256"
            }
        )
        != binding.self_sha256
    ):
        raise QuacP1FormalRunnerError(
            f"{field_name} semantic self hash drifted"
        )
    return value


def _live_implementation_paths() -> Mapping[str, Path]:
    return {
        "acquisition": Path(acquisition.__file__).resolve(),
        "action_adapter": Path(action.__file__).resolve(),
        "controller": Path(controller.__file__).resolve(),
        "evaluator": Path(evaluator.__file__).resolve(),
        "official_contract": Path(official_contract.__file__).resolve(),
        "runner": Path(__file__).resolve(),
        "runtime": Path(runtime.__file__).resolve(),
    }


def _default_service_state_reader(
    unit_name: str,
) -> Mapping[str, object]:
    try:
        completed = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                unit_name,
                "--property=ActiveState",
                "--property=LoadState",
                "--property=NRestarts",
                "--property=Restart",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise QuacP1FormalRunnerError(
            "formal service state could not be verified"
        ) from exc
    result: dict[str, object] = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if not separator or key in result:
            raise QuacP1FormalRunnerError(
                "formal service state output drifted"
            )
        result[key] = value
    return result


def _verify_production_freeze(
    config: FormalProductionConfig,
    *,
    service_state_reader: Callable[
        [str], Mapping[str, object]
    ],
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    effect = _verify_frozen_file(
        config.effect_design,
        field_name="effect design",
        require_self_sha256=True,
    )
    incident = _verify_frozen_file(
        config.custody_incident,
        field_name="custody incident",
        require_self_sha256=True,
    )
    assert effect is not None and incident is not None
    incident_details = incident.get("incident")
    incident_counts = incident.get("activity_counts_at_incident")
    if (
        not isinstance(incident_details, Mapping)
        or not isinstance(incident_counts, Mapping)
        or effect.get("self_sha256") != EFFECT_DESIGN_SELF_SHA256
        or incident.get("self_sha256")
        != _CUSTODY_INCIDENT_SELF_SHA256
        or incident_details.get(
            "postqualification_hash_only_operation_count"
        )
        != 1
        or incident_details.get(
            "postqualification_hash_only_member_read_count"
        )
        != 2
        or incident_counts.get("formal_source_decode")
        != 0
    ):
        raise QuacP1FormalRunnerError(
            "effect design or custody incident semantics drifted"
        )
    live_paths = _live_implementation_paths()
    implementation_hashes: dict[str, str] = {}
    for name, binding in config.implementation_files.items():
        try:
            configured = binding.path.resolve(strict=True)
        except OSError as exc:
            raise QuacP1FormalRunnerError(
                f"{name} implementation path is unavailable"
            ) from exc
        if configured != live_paths[name]:
            raise QuacP1FormalRunnerError(
                f"{name} implementation is not the live imported file"
            )
        _verify_frozen_file(
            binding,
            field_name=f"{name} implementation",
            require_self_sha256=False,
        )
        implementation_hashes[name] = binding.file_sha256
    service_raw = _read_direct_bytes_once(
        config.service_unit.path,
        field_name="formal service unit",
    )
    if (
        hashlib.sha256(service_raw).hexdigest()
        != config.service_unit.file_sha256
    ):
        raise QuacP1FormalRunnerError(
            "formal service unit file SHA256 drifted"
        )
    try:
        service_text = service_raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise QuacP1FormalRunnerError(
            "formal service unit is not UTF-8"
        ) from exc
    if "Restart=no" not in {
        line.strip() for line in service_text.splitlines()
    }:
        raise QuacP1FormalRunnerError(
            "formal service unit must freeze Restart=no"
        )
    service_state = service_state_reader(config.service_unit_name)
    if (
        set(service_state)
        != {"ActiveState", "LoadState", "NRestarts", "Restart"}
        or service_state.get("LoadState") != "loaded"
        or service_state.get("ActiveState")
        not in {"active", "activating"}
        or str(service_state.get("NRestarts")) != "0"
        or service_state.get("Restart") != "no"
    ):
        raise QuacP1FormalRunnerError(
            "formal service Restart/NRestarts state drifted"
        )
    return (
        MappingProxyType(dict(sorted(implementation_hashes.items()))),
        MappingProxyType(dict(service_state)),
    )


def _read_formal_source_member_once(
    member: FormalSourceMember,
    *,
    access_counts: dict[str, int],
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    if access_counts.get(member.role) != 0:
        raise QuacP1FormalRunnerError(
            f"{member.role} formal source read replay is forbidden"
        )
    access_counts[member.role] = 1
    raw = _read_direct_bytes_once(
        member.path,
        field_name=f"{member.role} formal source",
    )
    observed_sha256 = hashlib.sha256(raw).hexdigest()
    if (
        len(raw) != member.size_bytes
        or observed_sha256 != member.file_sha256
    ):
        raise QuacP1FormalRunnerError(
            f"{member.role} formal source byte binding drifted"
        )
    value = _strict_json_bytes(
        raw,
        field_name=f"{member.role} formal source",
        require_canonical=False,
    )
    del raw
    return value, {
        "file_sha256": observed_sha256,
        "read_count": 1,
        "role": member.role,
        "size_bytes": member.size_bytes,
    }


def _outer_failure(
    *,
    root: Path,
    stage: str,
    config_self_sha256: str,
    source_access_counts: Mapping[str, int],
    exc: BaseException,
) -> None:
    try:
        _write_json_once(
            root / "private" / "outer_failure.private.json",
            {
                "exception_message": str(exc),
                "exception_type": type(exc).__qualname__,
                "schema": f"{VERSION}_outer_private_failure_v1",
                "stage": stage,
            },
        )
    except QuacP1FormalRunnerError:
        pass
    try:
        _write_json_once(
            root / OUTER_FAILURE_FILENAME,
            _self_hashed(
                {
                    "API_or_online_evaluation_call_count": 0,
                    "config_self_sha256": config_self_sha256,
                    "failure_code": "formal_outer_stage_failed_closed",
                    "postqualification_hash_only_member_read_count": 2,
                    "postqualification_hash_only_operation_count": 1,
                    "private_item_query_document_qrel_action_score_or_exception_present_in_this_safe_receipt": False,
                    "retry_replay_resample_repair_or_fallback_authorized": False,
                    "schema": OUTER_FAILURE_SCHEMA,
                    "semantic_decode_before_formal_count": 0,
                    "source_access_counts": dict(
                        sorted(source_access_counts.items())
                    ),
                    "stage": stage,
                    "status": "implementation_or_infrastructure_invalid",
                    "study_id": STUDY_ID,
                }
            ),
        )
    except QuacP1FormalRunnerError:
        pass


def _safe_failure(
    *,
    work_root: Path,
    attempt_self_sha256: str | None,
    stage: str,
    exc: BaseException,
    secret_generation_count: int,
    block_execution_counts: Mapping[str, int],
    m_materialization_count_before_promotion: int,
) -> None:
    path = work_root / FAILURE_FILENAME
    if path.exists() or path.is_symlink():
        return
    try:
        _write_json_once(
            work_root
            / "private"
            / "formal_failure.private.json",
            {
                "exception_message": str(exc),
                "exception_type": type(exc).__qualname__,
                "schema": f"{VERSION}_private_failure_v1",
                "stage": stage,
            },
        )
    except QuacP1FormalRunnerError:
        pass
    receipt = _self_hashed(
        {
            "API_or_online_evaluation_call_count": 0,
            "attempt_self_sha256": attempt_self_sha256,
            "block_execution_counts": dict(
                sorted(block_execution_counts.items())
            ),
            "effect_design_self_sha256": EFFECT_DESIGN_SELF_SHA256,
            "failure_code": "stage_failed_closed",
            "m_materialization_count_before_promotion": (
                m_materialization_count_before_promotion
            ),
            "private_item_query_document_qrel_action_or_score_present_in_this_safe_receipt": False,
            "retry_replay_resample_repair_or_fallback_authorized": False,
            "schema": SAFE_FAILURE_SCHEMA,
            "secret_generation_count": secret_generation_count,
            "stage": stage,
            "status": "implementation_or_infrastructure_invalid",
            "study_id": STUDY_ID,
        }
    )
    try:
        _write_json_once(path, receipt)
    except QuacP1FormalRunnerError:
        return


def run_formal_once(
    *,
    train_obj: object,
    dev_obj: object,
    work_root: Path,
    block_executor: BlockExecutorProtocol,
    scientific_ops: ScientificOpsProtocol | None = None,
    acquisition_factory: Callable[
        [object, object, bytes],
        acquisition.TrustedAcquisitionBroker,
    ] = acquisition.acquire_decoded_sources_once,
    secret_factory: Callable[[int], bytes] = os.urandom,
    block_factory: Callable[
        [acquisition.TrustedAcquisitionBroker, str],
        runtime.RuntimeBlock,
    ] = runtime_block_from_broker,
    source_release_callback: Callable[[], None] | None = None,
) -> FormalRunResult:
    """Consume one complete formal attempt; failures are terminal and re-raised."""

    if not isinstance(work_root, Path) or not work_root.is_absolute():
        raise QuacP1FormalRunnerError(
            "formal work root must be an absolute Path"
        )
    if work_root.exists() or work_root.is_symlink():
        raise QuacP1FormalRunnerError(
            "formal work root is not fresh; retry is forbidden"
        )
    ops = scientific_ops or FrozenScientificOps()
    _private_directory(work_root, fresh=True)
    private_root = work_root / "private"
    safe_root = work_root / "safe"
    stage_root = work_root / "stages"
    _private_directory(private_root, fresh=True)
    _private_directory(safe_root, fresh=True)
    _private_directory(stage_root, fresh=True)

    stage = "claim_outer_attempt"
    attempt_self_sha256: str | None = None
    secret_generation_count = 0
    block_counts = {block: 0 for block in acquisition.BLOCK_ORDER}
    m_count_before_promotion = 0
    try:
        attempt = _self_hashed(
            {
                "API_or_online_evaluation_authorized": False,
                "acquisition_version": acquisition.VERSION,
                "action_adapter_version": action.VERSION,
                "controller_version": controller.VERSION,
                "effect_design_self_sha256": EFFECT_DESIGN_SELF_SHA256,
                "official_contract_version": official_contract.VERSION,
                "one_32_byte_secret_authorized": True,
                "one_attempt_no_retry_replay_resample_or_repair": True,
                "runtime_version": runtime.VERSION,
                "schema": ATTEMPT_SCHEMA,
                "study_id": STUDY_ID,
            }
        )
        attempt_self_sha256 = str(attempt["self_sha256"])
        _write_json_once(work_root / ATTEMPT_FILENAME, attempt)
        stage = "create_one_selection_secret"
        secret_generation_count += 1
        if secret_generation_count != 1:
            raise QuacP1FormalRunnerError(
                "selection secret generation count drifted"
            )
        secret = secret_factory(acquisition.HMAC_SECRET_BYTES)
        if (
            type(secret) is not bytes
            or len(secret) != acquisition.HMAC_SECRET_BYTES
        ):
            raise QuacP1FormalRunnerError(
                "secret factory did not return exactly 32 bytes"
            )
        _write_once(
            work_root / SECRET_FILENAME,
            secret,
            mode=0o600,
        )

        stage = "joint_acquisition_and_opaque_M_reservation"
        broker = acquisition_factory(train_obj, dev_obj, secret)
        if source_release_callback is not None:
            source_release_callback()
            source_release_callback = None
        del train_obj, dev_obj
        gc.collect()
        if not isinstance(
            broker,
            acquisition.TrustedAcquisitionBroker,
        ):
            raise QuacP1FormalRunnerError(
                "acquisition factory did not return the trusted broker"
            )
        selection_receipt = broker.safe_selection_receipt()
        _write_json_once(
            safe_root / "selection.safe.json",
            selection_receipt,
        )
        reservation = _require_m_reservation_state(
            broker,
            materialization_count=0,
            materialized_path_count=0,
        )
        _write_json_once(
            safe_root / "M_search.reservation.safe.json",
            reservation,
        )

        runtime_commitments: dict[str, str] = {}
        action_commitments: dict[str, str] = {}

        stage = "A_form_label_free_runtime"
        a_form_block = block_factory(broker, "A_form")
        block_counts["A_form"] += 1
        a_form_runtime = block_executor(
            block_name="A_form",
            block=a_form_block,
            work_root=stage_root / "A_form",
            official_required=False,
        )
        (
            runtime_commitments["A_form_runtime"],
            runtime_binding_token,
            runtime_asset_binding,
        ) = _runtime_commitment(
            a_form_runtime,
            block_name="A_form",
            block=a_form_block,
            official_required=False,
        )
        a_form_items = _graph_items(
            a_form_runtime,
            folds=broker.a_form_folds(),
        )

        stage = "A_form_late_labels_after_action_seal"
        a_form_action_payload = _runtime_action_payload(
            block=a_form_block,
            result=a_form_runtime,
        )
        action_commitments["A_form_label_free_actions"] = stable_hash(
            a_form_action_payload
        )
        a_form_labels = _open_labels_after_seal(
            broker=broker,
            block="A_form",
            action_path=(
                stage_root
                / "A_form"
                / "private"
                / "actions.private.json"
            ),
            expected_action_payload=a_form_action_payload,
            private_root=private_root,
        )

        stage = "A_form_fit_once"
        fitted = ops.fit_a_form(
            a_form_items,
            a_form_labels,
            block_corpus_unit_ids=tuple(
                row.unit_id for row in a_form_block.documents
            ),
        )
        if type(fitted) is not FittedEvaluator:
            raise QuacP1FormalRunnerError(
                "scientific fit result type drifted"
            )
        _assert_fitted_unchanged(fitted)
        _write_once(
            private_root / "A_form.model_parameters.private.bin",
            fitted.parameter_archive,
            mode=0o400,
        )
        model_seal = broker.issue_a_form_model_seal(
            model_parameter_sha256=fitted.parameter_sha256,
        )
        model_seal_path = (
            private_root / "A_form.model_seal.private.json"
        )
        _write_json_once(
            model_seal_path,
            model_seal.payload(),
        )
        broker.register_durable_a_form_model_seal(
            seal=model_seal,
            seal_path=model_seal_path,
        )

        stage = "A_hold_label_free_three_arm_runtime"
        a_hold_block = block_factory(broker, "A_hold")
        block_counts["A_hold"] += 1
        a_hold_runtime = block_executor(
            block_name="A_hold",
            block=a_hold_block,
            work_root=stage_root / "A_hold",
            official_required=True,
        )
        (
            runtime_commitments["A_hold_runtime"],
            a_hold_binding_token,
            a_hold_asset_binding,
        ) = _runtime_commitment(
            a_hold_runtime,
            block_name="A_hold",
            block=a_hold_block,
            official_required=True,
        )
        if (
            a_hold_binding_token != runtime_binding_token
            or a_hold_asset_binding != runtime_asset_binding
        ):
            raise QuacP1FormalRunnerError(
                "A_hold runtime escaped the preverified binding epoch"
            )
        if a_hold_runtime.official_top5 is None:
            raise QuacP1FormalRunnerError(
                "A_hold official HippoRAG result is absent"
            )
        a_hold_items = _graph_items(a_hold_runtime, folds=None)
        a_hold_actions = ops.select_measurement(
            block="A_hold",
            items=a_hold_items,
            fitted=fitted,
            hipporag_top5=a_hold_runtime.official_top5,
            block_corpus_unit_ids=tuple(
                row.unit_id for row in a_hold_block.documents
            ),
        )
        if type(a_hold_actions) is not MeasurementActions:
            raise QuacP1FormalRunnerError(
                "A_hold measurement action type drifted"
            )
        _assert_fitted_unchanged(fitted)
        _write_json_once(
            private_root / "A_hold.measurement_actions.private.json",
            _thaw_json(a_hold_actions.payload),
        )
        action_commitments["A_hold_four_arm_actions"] = (
            a_hold_actions.action_sha256
        )

        stage = "A_hold_late_labels_after_four_arm_seal"
        a_hold_labels = _open_labels_after_seal(
            broker=broker,
            block="A_hold",
            action_path=(
                private_root
                / "A_hold.measurement_actions.private.json"
            ),
            expected_action_payload=_thaw_json(
                a_hold_actions.payload
            ),
            private_root=private_root,
        )
        stage = "A_hold_offline_score_and_promotion"
        a_hold_score = ops.score_measurement(
            a_hold_actions,
            a_hold_labels,
        )
        if (
            type(a_hold_score) is not MeasurementScore
            or a_hold_score.block != "A_hold"
        ):
            raise QuacP1FormalRunnerError(
                "A_hold score type drifted"
            )
        _assert_fitted_unchanged(fitted)
        a_hold_score_file_sha = _write_json_once(
            safe_root / "A_hold.score.safe.json",
            _thaw_json(a_hold_score.safe_payload),
        )

        m_score: MeasurementScore | None = None
        _require_m_reservation_state(
            broker,
            materialization_count=0,
            materialized_path_count=0,
        )
        if a_hold_score.promoted:
            stage = "issue_and_consume_one_M_capability"
            _write_json_once(
                safe_root / "A_hold.promotion.safe.json",
                {
                    "a_hold_score_receipt_sha256": (
                        a_hold_score_file_sha
                    ),
                    "aggregate_e1_minus_e0": (
                        a_hold_score.e1_minus_e0
                    ),
                    "p_denominator": a_hold_score.p_denominator,
                    "p_numerator": a_hold_score.p_numerator,
                    "promoted": True,
                    "schema": (
                        f"{VERSION}_safe_promotion_receipt_v1"
                    ),
                    "selection_commitment": (
                        broker.selection_commitment
                    ),
                    "study_id": STUDY_ID,
                },
            )
            m_capability = (
                broker.authorize_m_search_from_stage_score(
                    stage_score=a_hold_score.native,
                    score_receipt_path=(
                        safe_root / "A_hold.score.safe.json"
                    ),
                )
            )
            _write_json_once(
                private_root / "M_search.capability.private.json",
                m_capability.payload(),
            )
            broker.materialize_m_search(m_capability)

            stage = "M_search_label_free_three_arm_runtime"
            m_block = block_factory(broker, "M_search")
            m_registry = (
                broker.m_search_materialized_registry_payload()
            )
            m_registry_path = (
                private_root
                / "M_search.materialized_registry.private.json"
            )
            _write_json_once(m_registry_path, m_registry)
            broker.register_durable_m_search_materialized_registry(
                registry_path=m_registry_path,
                expected_payload=m_registry,
            )
            _require_m_reservation_state(
                broker,
                materialization_count=1,
                materialized_path_count=1,
            )
            block_counts["M_search"] += 1
            m_runtime = block_executor(
                block_name="M_search",
                block=m_block,
                work_root=stage_root / "M_search",
                official_required=True,
            )
            (
                runtime_commitments["M_search_runtime"],
                m_binding_token,
                m_asset_binding,
            ) = _runtime_commitment(
                m_runtime,
                block_name="M_search",
                block=m_block,
                official_required=True,
            )
            if (
                m_binding_token != runtime_binding_token
                or m_asset_binding != runtime_asset_binding
            ):
                raise QuacP1FormalRunnerError(
                    "M_search runtime escaped the preverified binding epoch"
                )
            if m_runtime.official_top5 is None:
                raise QuacP1FormalRunnerError(
                    "M_search official HippoRAG result is absent"
                )
            m_items = _graph_items(m_runtime, folds=None)
            m_actions = ops.select_measurement(
                block="M_search",
                items=m_items,
                fitted=fitted,
                hipporag_top5=m_runtime.official_top5,
                block_corpus_unit_ids=tuple(
                    row.unit_id for row in m_block.documents
                ),
            )
            if type(m_actions) is not MeasurementActions:
                raise QuacP1FormalRunnerError(
                    "M_search measurement action type drifted"
                )
            _assert_fitted_unchanged(fitted)
            _write_json_once(
                private_root
                / "M_search.measurement_actions.private.json",
                _thaw_json(m_actions.payload),
            )
            action_commitments["M_search_four_arm_actions"] = (
                m_actions.action_sha256
            )

            stage = "M_search_late_labels_after_four_arm_seal"
            m_labels = _open_labels_after_seal(
                broker=broker,
                block="M_search",
                action_path=(
                    private_root
                    / "M_search.measurement_actions.private.json"
                ),
                expected_action_payload=_thaw_json(
                    m_actions.payload
                ),
                private_root=private_root,
            )
            stage = "M_search_offline_L5_score"
            m_score = ops.score_measurement(m_actions, m_labels)
            if (
                type(m_score) is not MeasurementScore
                or m_score.block != "M_search"
            ):
                raise QuacP1FormalRunnerError(
                    "M_search score type drifted"
                )
            _assert_fitted_unchanged(fitted)
            _write_json_once(
                safe_root / "M_search.score.safe.json",
                _thaw_json(m_score.safe_payload),
            )

        stage = "write_single_safe_terminal"
        final_m_state = _require_m_reservation_state(
            broker,
            materialization_count=(
                1 if a_hold_score.native.promotion else 0
            ),
            materialized_path_count=(
                1 if a_hold_score.native.promotion else 0
            ),
        )
        scientific_terminal = ops.safe_terminal(
            a_hold=a_hold_score,
            m_search=m_score,
            fitted=fitted,
            action_commitments=action_commitments,
            runtime_commitments=runtime_commitments,
            m_materialization_count_before_promotion=(
                m_count_before_promotion
            ),
        )
        if not isinstance(scientific_terminal, Mapping):
            raise QuacP1FormalRunnerError(
                "scientific terminal type drifted"
            )
        expected_scientific_terminal = controller.safe_terminal(
            a_hold=a_hold_score.native,
            m_search=(
                None if m_score is None else m_score.native
            ),
            model_parameter_sha256=fitted.parameter_sha256,
            action_commitments=action_commitments,
            runtime_commitments=runtime_commitments,
            M_materialization_count_before_promotion=(
                m_count_before_promotion
            ),
        )
        if dict(scientific_terminal) != expected_scientific_terminal:
            raise QuacP1FormalRunnerError(
                "scientific terminal is not the exact controller terminal"
            )
        scientific_terminal = expected_scientific_terminal
        inner_status = scientific_terminal.get("status")
        expected_outer_status = (
            "VALID_COMPLETE_PROMOTED_M_MEASURED"
            if a_hold_score.native.promotion
            else "VALID_NONPROMOTION_M_UNOPENED"
        )
        if inner_status != expected_outer_status:
            raise QuacP1FormalRunnerError(
                "scientific terminal branch status drifted"
            )
        terminal_body = {
            "API_or_online_evaluation_call_count": 0,
            "attempt_self_sha256": attempt_self_sha256,
            "block_execution_counts": dict(sorted(block_counts.items())),
            "effect_design_self_sha256": EFFECT_DESIGN_SELF_SHA256,
            "inner_scientific_terminal": dict(scientific_terminal),
            "m_materialization_count_before_promotion": 0,
            "m_materialization_count_final": final_m_state[
                "materialization_count"
            ],
            "m_materialized_path_count_final": final_m_state[
                "materialized_path_count"
            ],
            "private_item_query_document_qrel_action_or_score_present_in_this_safe_receipt": False,
            "retry_replay_resample_repair_or_fallback_count": 0,
            "schema": SAFE_TERMINAL_SCHEMA,
            "secret_generation_count": secret_generation_count,
            "status": inner_status,
            "study_id": STUDY_ID,
        }
        terminal = _self_hashed(terminal_body)
        terminal_path = work_root / TERMINAL_FILENAME
        _write_json_once(terminal_path, terminal)
        return FormalRunResult(
            terminal=terminal,
            terminal_path=terminal_path,
        )
    except BaseException as exc:
        _safe_failure(
            work_root=work_root,
            attempt_self_sha256=attempt_self_sha256,
            stage=stage,
            exc=exc,
            secret_generation_count=secret_generation_count,
            block_execution_counts=block_counts,
            m_materialization_count_before_promotion=(
                m_count_before_promotion
            ),
        )
        raise


def run_production_formal_once(
    config: FormalProductionConfig,
    *,
    service_state_reader: Callable[
        [str], Mapping[str, object]
    ] = _default_service_state_reader,
    runtime_verifier: Callable[..., runtime.VerifiedRuntimeBindings] = (
        runtime.verify_runtime_bindings_once
    ),
    encoder_factory: Callable[
        [runtime.RuntimeBindings], runtime.MiniLMEncoderProtocol
    ] = lambda bindings: runtime.LocalMiniLMGpu0Encoder(
        Path(bindings.minilm_asset.path)
    ),
    official_lane_factory: Callable[
        [], runtime.OfficialLaneProtocol
    ] = runtime.LocalOfficialGpu1Lane,
    core_runner: Callable[..., FormalRunResult] = run_formal_once,
) -> Mapping[str, object]:
    """Run the noninjectable-by-CLI production custody and core lifecycle."""

    raise QuacP1FormalRunnerError(
        "production formal execution is available only through "
        "replication_runtime.quac_p1_formal_v1.runner"
    )

    if type(config) is not FormalProductionConfig:
        raise QuacP1FormalRunnerError(
            "production formal config type is not exact"
        )
    root = config.formal_root
    if root.exists() or root.is_symlink():
        raise QuacP1FormalRunnerError(
            "production formal root is not fresh; retry is forbidden"
        )
    _private_directory(root, fresh=True)
    _private_directory(root / "private", fresh=True)
    _private_directory(root / "safe", fresh=True)
    stage = "claim_global_outer_attempt"
    source_access_counts = {"dev": 0, "train": 0}
    source_holder: list[Mapping[str, object]] = []
    try:
        marker = _self_hashed(
            {
                "API_or_online_evaluation_authorized": False,
                "config_self_sha256": config.self_sha256,
                "custody_incident_self_sha256": (
                    _CUSTODY_INCIDENT_SELF_SHA256
                ),
                "effect_design_self_sha256": (
                    EFFECT_DESIGN_SELF_SHA256
                ),
                "formal_CLI_injection_authorized": False,
                "one_global_attempt_no_restart_retry_or_replay": True,
                "postqualification_hash_only_member_read_count": 2,
                "postqualification_hash_only_operation_count": 1,
                "schema": OUTER_ATTEMPT_SCHEMA,
                "semantic_decode_before_formal_count": 0,
                "service_restart_policy": "no",
                "study_id": STUDY_ID,
            }
        )
        _write_json_once(
            root / OUTER_ATTEMPT_FILENAME,
            marker,
        )

        stage = "verify_live_freeze_before_source"
        implementation_hashes, service_state = (
            _verify_production_freeze(
                config,
                service_state_reader=service_state_reader,
            )
        )

        stage = "verify_runtime_bindings_once_before_source"
        verified = runtime_verifier(
            config.runtime_bindings,
            source_access_count=sum(source_access_counts.values()),
        )
        if type(verified) is not runtime.VerifiedRuntimeBindings:
            raise QuacP1FormalRunnerError(
                "runtime verifier did not return the exact token type"
            )
        verified_receipt = verified.require(
            config.runtime_bindings
        )
        if (
            verified_receipt.get("full_tree_verification_count") != 1
            or verified_receipt.get(
                "source_access_count_at_verification"
            )
            != 0
        ):
            raise QuacP1FormalRunnerError(
                "pre-source runtime verification receipt drifted"
            )
        verified_receipt_file_sha256 = _write_once(
            root / "private" / "runtime_bindings.private.json",
            verified.canonical_receipt,
            mode=0o400,
        )
        preflight = _self_hashed(
            {
                "binding_verification_token_sha256": (
                    verified.token_sha256
                ),
                "custody_incident_self_sha256": (
                    _CUSTODY_INCIDENT_SELF_SHA256
                ),
                "effect_design_self_sha256": (
                    EFFECT_DESIGN_SELF_SHA256
                ),
                "full_tree_verification_count": 1,
                "implementation_registry_sha256": stable_hash(
                    dict(implementation_hashes)
                ),
                "runtime_binding_receipt_file_sha256": (
                    verified_receipt_file_sha256
                ),
                "schema": f"{VERSION}_safe_preflight_v1",
                "service_NRestarts": 0,
                "service_restart_policy": "no",
                "source_access_count_at_verification": 0,
            }
        )
        _write_json_once(
            root / "safe" / "preflight.safe.json",
            preflight,
        )

        stage = "read_each_formal_source_member_once"
        train_obj, train_receipt = _read_formal_source_member_once(
            config.train,
            access_counts=source_access_counts,
        )
        dev_obj, dev_receipt = _read_formal_source_member_once(
            config.dev,
            access_counts=source_access_counts,
        )
        if source_access_counts != {"dev": 1, "train": 1}:
            raise QuacP1FormalRunnerError(
                "formal source member access counts drifted"
            )
        _write_json_once(
            root / "safe" / "source_access.safe.json",
            _self_hashed(
                {
                    "members": [train_receipt, dev_receipt],
                    "postqualification_hash_only_member_read_count": 2,
                    "postqualification_hash_only_operation_count": 1,
                    "schema": f"{VERSION}_safe_source_access_v1",
                    "semantic_decode_before_formal_count": 0,
                    "source_formal_decode_count": 2,
                    "source_member_read_count": 2,
                }
            ),
        )

        stage = "construct_fixed_production_executor"
        encoder = encoder_factory(config.runtime_bindings)
        official_lane = official_lane_factory()
        executor = BoundRuntimeExecutor(
            bindings=config.runtime_bindings,
            verified_bindings=verified,
            encoder=encoder,
            official_lane=official_lane,
        )
        source_holder.extend((train_obj, dev_obj))
        del train_obj, dev_obj

        def release_sources() -> None:
            source_holder.clear()
            gc.collect()

        stage = "run_frozen_formal_core_once"
        core_result = core_runner(
            train_obj=source_holder[0],
            dev_obj=source_holder[1],
            work_root=root / "core",
            block_executor=executor,
            scientific_ops=FrozenScientificOps(),
            source_release_callback=release_sources,
        )
        if type(core_result) is not FormalRunResult:
            raise QuacP1FormalRunnerError(
                "formal core result type is not exact"
            )
        release_sources()

        stage = "verify_service_no_restart_postflight"
        post_service = service_state_reader(
            config.service_unit_name
        )
        if (
            dict(post_service) != dict(service_state)
            or str(post_service.get("NRestarts")) != "0"
            or post_service.get("Restart") != "no"
        ):
            raise QuacP1FormalRunnerError(
                "formal service restarted during the attempt"
            )
        core_terminal_raw = _read_direct_bytes_once(
            core_result.terminal_path,
            field_name="core safe terminal",
        )
        parsed_core_terminal = _strict_json_bytes(
            core_terminal_raw,
            field_name="core safe terminal",
            require_canonical=True,
        )
        if dict(parsed_core_terminal) != dict(core_result.terminal):
            raise QuacP1FormalRunnerError(
                "core result detached from its durable terminal"
            )
        outer = _self_hashed(
            {
                "API_or_online_evaluation_call_count": 0,
                "binding_verification_token_sha256": (
                    verified.token_sha256
                ),
                "config_self_sha256": config.self_sha256,
                "core_terminal_file_sha256": hashlib.sha256(
                    core_terminal_raw
                ).hexdigest(),
                "core_terminal_self_sha256": (
                    parsed_core_terminal["self_sha256"]
                ),
                "custody_incident_self_sha256": (
                    _CUSTODY_INCIDENT_SELF_SHA256
                ),
                "effect_design_self_sha256": (
                    EFFECT_DESIGN_SELF_SHA256
                ),
                "full_tree_verification_count": 1,
                "postqualification_hash_only_member_read_count": 2,
                "postqualification_hash_only_operation_count": 1,
                "private_item_query_document_qrel_action_score_or_source_present_in_this_safe_receipt": False,
                "retry_replay_resample_repair_or_fallback_count": 0,
                "schema": OUTER_TERMINAL_SCHEMA,
                "semantic_decode_before_formal_count": 0,
                "service_NRestarts": 0,
                "source_access_counts": dict(
                    sorted(source_access_counts.items())
                ),
                "status": parsed_core_terminal["status"],
                "study_id": STUDY_ID,
            }
        )
        _write_json_once(
            root / OUTER_TERMINAL_FILENAME,
            outer,
        )
        return outer
    except BaseException as exc:
        source_holder.clear()
        gc.collect()
        _outer_failure(
            root=root,
            stage=stage,
            config_self_sha256=config.self_sha256,
            source_access_counts=source_access_counts,
            exc=exc,
        )
        raise


def load_formal_production_config(
    path: Path,
) -> FormalProductionConfig:
    return FormalProductionConfig.from_path(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the one-shot frozen QuAC P1 formal lifecycle."
    )
    parser.add_argument("--config", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    raise QuacP1FormalRunnerError(
        "this module is a scientific core without a CLI; use "
        "replication_runtime.quac_p1_formal_v1.runner"
    )


__all__ = [
    "ATTEMPT_FILENAME",
    "ATTEMPT_SCHEMA",
    "BoundRuntimeExecutor",
    "EFFECT_DESIGN_SELF_SHA256",
    "FAILURE_FILENAME",
    "FittedEvaluator",
    "FormalRunResult",
    "FrozenScientificOps",
    "MeasurementActions",
    "MeasurementScore",
    "QuacP1FormalRunnerError",
    "SAFE_FAILURE_SCHEMA",
    "SAFE_TERMINAL_SCHEMA",
    "SECRET_FILENAME",
    "ScientificOpsProtocol",
    "TERMINAL_FILENAME",
    "VERSION",
    "canonical_bytes",
    "run_formal_once",
    "runtime_block_from_broker",
    "stable_hash",
]
