"""One-shot production entrypoint for the frozen TAT-QA P19 study.

This module wires the already-frozen runtime, custody adapters, and lifecycle
controller.  It intentionally has no TAT-QA source, label, answer, family, or
provider-API argument.  Runtime workers receive only the allowlisted,
environment-cleared capabilities implemented by :mod:`formal_runtime`.

The entrypoint is terminal: after it starts, either the controller disposition
or a content-free bootstrap failure disposition is written exactly once as a
canonical, durable, locally verifiable JSON record.  There is no retry path.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Callable, Mapping, Sequence

from assumption_agent.benchmarks import tatqa_p19_acquisition_v1 as acquisition
from assumption_agent.benchmarks import tatqa_p19_formal_adapters_v1 as adapters
from assumption_agent.benchmarks import tatqa_p19_formal_controller_v1 as controller
from replication_runtime.tatqa_p19_v1 import formal_runtime


VERSION = "tatqa_p19_formal_study_v1"
FINAL_DISPOSITION_FILENAME = "formal.disposition.json"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_FINAL_KEYS = frozenset(
    {
        "api_or_online_evaluator_calls",
        "controller_disposition",
        "controller_disposition_sha256",
        "external_network_calls",
        "final_disposition_sha256",
        "offline_artifacts",
        "replay_retry_resample_provider_switch",
        "runtime_fingerprint_self_sha256",
        "schema",
        "status",
        "systemd_network_preflight",
        "version",
    }
)


class TatqaP19FormalStudyError(RuntimeError):
    """The formal entry or its durable terminal record failed closed."""


class TatqaP19FormalStudyAlreadyConsumed(TatqaP19FormalStudyError):
    """The requested one-shot control root already exists."""


@dataclass(frozen=True)
class FormalStudyConfig:
    """Explicit, label-free production paths.

    There is deliberately no source/acquisition-private path beyond the
    project root understood solely by :class:`TrustedAcquisitionAdapter`.
    """

    project_root: Path
    control_root: Path
    work_root: Path
    typed_runtime_python: Path
    hippo_runtime_python: Path
    qwen_model: Path
    minilm_asset_manifest: Path
    minilm_model: Path
    hippo_llm_model: Path
    hippo_embedding_model: Path
    hipporag_source: Path
    hippo_attestation: Path
    runtime_fingerprint: Path
    production_canary: Path

    def runtime_paths(self) -> formal_runtime.RuntimePaths:
        return formal_runtime.RuntimePaths(
            project_root=Path(self.project_root),
            typed_runtime_python=Path(self.typed_runtime_python),
            hippo_runtime_python=Path(self.hippo_runtime_python),
            qwen_model=Path(self.qwen_model),
            minilm_asset_manifest=Path(self.minilm_asset_manifest),
            minilm_model=Path(self.minilm_model),
            hippo_llm_model=Path(self.hippo_llm_model),
            hippo_embedding_model=Path(self.hippo_embedding_model),
            hipporag_source=Path(self.hipporag_source),
            hippo_attestation=Path(self.hippo_attestation),
            fingerprint_manifest=Path(self.runtime_fingerprint),
            work_root=Path(self.work_root),
        )

    @property
    def result_path(self) -> Path:
        return Path(self.control_root) / FINAL_DISPOSITION_FILENAME


@dataclass(frozen=True)
class FormalStudyDependencies:
    """Narrow injection seam used by non-model orchestration tests."""

    verify_implementation_freeze: Callable[..., Mapping[str, Any]]
    verify_runtime_fingerprint: Callable[[formal_runtime.RuntimePaths], Mapping[str, Any]]
    systemd_network_preflight: Callable[[], Mapping[str, object]]
    minilm_encoder_factory: Callable[[formal_runtime.RuntimePaths], object]
    typed_plan_runner_factory: Callable[[formal_runtime.RuntimePaths], object]
    hippo_runner_factory: Callable[[formal_runtime.RuntimePaths], object]
    runtime_adapter_factory: Callable[..., object]
    acquisition_adapter_factory: Callable[..., object]
    controller_factory: Callable[..., object]


DEFAULT_DEPENDENCIES = FormalStudyDependencies(
    verify_implementation_freeze=acquisition.verify_implementation_freeze,
    verify_runtime_fingerprint=formal_runtime.verify_runtime_fingerprint,
    systemd_network_preflight=formal_runtime.systemd_network_preflight,
    minilm_encoder_factory=formal_runtime.BoundMiniLMEncoder,
    typed_plan_runner_factory=formal_runtime.SystemdTypedPlanBatchRunner,
    hippo_runner_factory=formal_runtime.SystemdHippoByteRunner,
    runtime_adapter_factory=adapters.ProductionRuntimeAdapter,
    acquisition_adapter_factory=adapters.TrustedAcquisitionAdapter,
    controller_factory=controller.TatqaP19FormalController,
)


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
        raise TatqaP19FormalStudyError("terminal value is not canonical JSON") from exc


def _semantic_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value).rstrip(b"\n")).hexdigest()


def _failure_type_sha256(exc: BaseException) -> str:
    return hashlib.sha256(
        f"{type(exc).__module__}.{type(exc).__qualname__}".encode("utf-8")
    ).hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise ValueError("nonfinite JSON constant")


def _read_regular(path: Path, *, field: str) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise TatqaP19FormalStudyError(f"{field} is unavailable") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise TatqaP19FormalStudyError(f"{field} is not a regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as handle:
            opened = os.fstat(handle.fileno())
            if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
                raise TatqaP19FormalStudyError(f"{field} changed during open")
            raw = handle.read()
    except OSError as exc:
        raise TatqaP19FormalStudyError(f"{field} cannot be read safely") from exc
    if len(raw) != before.st_size:
        raise TatqaP19FormalStudyError(f"{field} changed during read")
    return raw


def _strict_json(raw: bytes, *, field: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise TatqaP19FormalStudyError(f"{field} is not strict JSON") from exc
    if not isinstance(value, dict) or _canonical_bytes(value) != raw:
        raise TatqaP19FormalStudyError(f"{field} is not canonical JSON")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_safe_absent_root(path: Path, *, field: str) -> Path:
    absolute = path.expanduser().absolute()
    if absolute == Path(absolute.anchor):
        raise TatqaP19FormalStudyError(f"{field} cannot be a filesystem root")
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink():
            raise TatqaP19FormalStudyError(f"{field} contains a symlink")
    if absolute.exists():
        raise TatqaP19FormalStudyAlreadyConsumed(f"{field} already exists")
    return absolute


def _mkdir_exclusive(path: Path) -> None:
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.mkdir(mode=0o700)
        os.chmod(path, 0o700)
        _fsync_directory(path.parent)
    except OSError as exc:
        raise TatqaP19FormalStudyError("one-shot directory claim failed") from exc


def _write_terminal_exclusive(path: Path, value: Mapping[str, Any]) -> str:
    raw = _canonical_bytes(dict(value))
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(path.parent)
    except OSError as exc:
        raise TatqaP19FormalStudyError("exclusive terminal write failed") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if _read_regular(path, field="terminal disposition") != raw:
        raise TatqaP19FormalStudyError("terminal disposition reopen drifted")
    return hashlib.sha256(raw).hexdigest()


def load_final_disposition(path: str | Path) -> dict[str, Any]:
    """Read and verify the exact durable terminal record offline."""

    result_path = Path(path)
    raw = _read_regular(result_path, field="terminal disposition")
    value = _strict_json(raw, field="terminal disposition")
    claimed = value.get("final_disposition_sha256")
    if not isinstance(claimed, str) or _SHA256.fullmatch(claimed) is None:
        raise TatqaP19FormalStudyError("terminal self hash is absent")
    body = dict(value)
    del body["final_disposition_sha256"]
    controller_payload = value.get("controller_disposition")
    # The physical file hash is returned by :func:`run_formal_study`; it cannot
    # recursively live inside the bytes it hashes.  The file itself carries a
    # semantic self hash and is byte-for-byte canonical.
    if (
        set(value) != _FINAL_KEYS
        or value.get("schema") != f"{VERSION}_durable_final_disposition_v1"
        or value.get("version") != VERSION
        or _semantic_hash(body) != claimed
        or not isinstance(controller_payload, Mapping)
        or _semantic_hash(controller_payload)
        != value.get("controller_disposition_sha256")
        or value.get("status") != controller_payload.get("status")
        or value.get("external_network_calls") != 0
        or value.get("api_or_online_evaluator_calls") != 0
        or value.get("replay_retry_resample_provider_switch") != 0
    ):
        raise TatqaP19FormalStudyError("terminal disposition binding drifted")
    return value


def _disposition_artifacts(
    disposition: controller.FormalDisposition,
) -> dict[str, object]:
    fields = (
        ("runtime_preflight", disposition.preflight),
        ("A_form_fit", disposition.a_form_fit),
        ("A_form_archive", disposition.a_form_archive),
        ("F_search_archive", disposition.f_search_archive),
        ("policy_freeze", disposition.policy_freeze),
        ("A_hold_archive", disposition.a_hold_archive),
        ("A_hold_score", disposition.a_hold_score),
        ("epoch_authorization", disposition.epoch_authorization),
        ("M_search_archive", disposition.m_search_archive),
        ("M_search_score", disposition.m_search_score),
    )
    result = {
        name: None if value is None else value.payload()
        for name, value in fields
    }
    result["E1_model"] = (
        None
        if disposition.a_form_fit is None
        else disposition.a_form_fit.model.payload()
    )
    return result


def _invalid_disposition(
    exc: BaseException, *, stage: str
) -> controller.FormalDisposition:
    return controller.FormalDisposition(
        status="implementation_or_runtime_invalid",
        primary_evaluated=False,
        primary_value=None,
        efficacy="unknown",
        a_hold_promoted=False,
        epoch_transition_count=0,
        m_view_released=False,
        m_labels_released=False,
        replay_authorized=False,
        failure_stage=stage,
        failure_type_sha256=_failure_type_sha256(exc),
    )


def _terminal_envelope(
    *,
    disposition: controller.FormalDisposition,
    runtime_fingerprint_self_sha256: str | None,
    network_preflight: Mapping[str, object] | None,
) -> dict[str, Any]:
    disposition_payload = disposition.payload()
    body: dict[str, Any] = {
        "api_or_online_evaluator_calls": 0,
        "controller_disposition": disposition_payload,
        "controller_disposition_sha256": disposition.disposition_sha256,
        "external_network_calls": 0,
        "offline_artifacts": _disposition_artifacts(disposition),
        "replay_retry_resample_provider_switch": 0,
        "runtime_fingerprint_self_sha256": runtime_fingerprint_self_sha256,
        "schema": f"{VERSION}_durable_final_disposition_v1",
        "status": disposition.status,
        "systemd_network_preflight": (
            None if network_preflight is None else dict(network_preflight)
        ),
        "version": VERSION,
    }
    return {**body, "final_disposition_sha256": _semantic_hash(body)}


def _persist_terminal(
    control_root: Path,
    envelope: Mapping[str, Any],
    *,
    root_existed_for_this_attempt: bool,
) -> tuple[Path, str, dict[str, Any]]:
    if not root_existed_for_this_attempt:
        _mkdir_exclusive(control_root)
    elif (
        control_root.is_symlink()
        or not control_root.is_dir()
        or stat.S_IMODE(control_root.stat().st_mode) != 0o700
    ):
        raise TatqaP19FormalStudyError("claimed control root drifted")
    path = control_root / FINAL_DISPOSITION_FILENAME
    file_sha = _write_terminal_exclusive(path, envelope)
    reopened = load_final_disposition(path)
    if reopened != dict(envelope):
        raise TatqaP19FormalStudyError("terminal disposition semantic reopen drifted")
    return path, file_sha, reopened


def run_formal_study(
    config: FormalStudyConfig,
    *,
    dependencies: FormalStudyDependencies = DEFAULT_DEPENDENCIES,
) -> dict[str, Any]:
    """Run one formal lifecycle and durably persist its unique disposition.

    Bootstrap faults are converted into the same terminal invalid semantics as
    controller/runtime faults.  Only an already-consumed control root or a
    failure to persist the sole terminal record escapes as an exception.
    """

    if not isinstance(config, FormalStudyConfig):
        raise TatqaP19FormalStudyError("formal study configuration drifted")
    control_root = _ensure_safe_absent_root(Path(config.control_root), field="control root")
    work_root = _ensure_safe_absent_root(Path(config.work_root), field="work root")
    if (
        control_root == work_root
        or control_root in work_root.parents
        or work_root in control_root.parents
    ):
        raise TatqaP19FormalStudyError("control and runtime work roots overlap")
    runtime_paths = config.runtime_paths()
    if runtime_paths.work_root.expanduser().absolute() != work_root:
        raise TatqaP19FormalStudyError("runtime work-root binding drifted")

    fingerprint_sha: str | None = None
    network_receipt: Mapping[str, object] | None = None
    stage = "implementation_freeze"
    control_claimed = False
    runtime: object | None = None
    try:
        dependencies.verify_implementation_freeze(
            Path(config.project_root),
            runtime_fingerprint_path=Path(config.runtime_fingerprint),
            production_canary_path=Path(config.production_canary),
        )
        stage = "runtime_fingerprint"
        fingerprint = dict(dependencies.verify_runtime_fingerprint(runtime_paths))
        fingerprint_sha_value = fingerprint.get("self_sha256")
        if (
            not isinstance(fingerprint_sha_value, str)
            or _SHA256.fullmatch(fingerprint_sha_value) is None
        ):
            raise TatqaP19FormalStudyError("runtime fingerprint self hash drifted")
        fingerprint_sha = fingerprint_sha_value

        stage = "systemd_network_preflight"
        network_receipt = dict(dependencies.systemd_network_preflight())
        if (
            network_receipt.get("returncode") != 0
            or network_receipt.get("network_properties")
            != list(formal_runtime.SYSTEMD_NETWORK_PROPERTIES)
        ):
            raise TatqaP19FormalStudyError("systemd network preflight drifted")

        stage = "runtime_work_root_claim"
        _mkdir_exclusive(work_root)

        stage = "bound_minilm_initialization"
        minilm = dependencies.minilm_encoder_factory(runtime_paths)
        stage = "offline_worker_runner_initialization"
        typed_runner = dependencies.typed_plan_runner_factory(runtime_paths)
        hippo_runner = dependencies.hippo_runner_factory(runtime_paths)
        stage = "production_runtime_adapter_initialization"
        runtime = dependencies.runtime_adapter_factory(
            control_root=control_root,
            receipt_paths=adapters.RuntimeReceiptPaths(
                runtime_fingerprint=Path(config.runtime_fingerprint),
                production_canary=Path(config.production_canary),
            ),
            typed_plan_runner=typed_runner,
            minilm_encoder=minilm,
            hippo_runner=hippo_runner,
        )
        stage = "trusted_acquisition_adapter_initialization"
        custody = dependencies.acquisition_adapter_factory(
            project_root=Path(config.project_root),
            runtime=runtime,
            control_root=control_root,
        )
        stage = "formal_controller_initialization"
        lifecycle = dependencies.controller_factory(
            acquisition=custody,
            runtime=runtime,
        )
        stage = "formal_controller_run"
        disposition = lifecycle.run()
        if not isinstance(disposition, controller.FormalDisposition):
            raise TatqaP19FormalStudyError("controller disposition type drifted")
        control_claimed = control_root.is_dir()
    except Exception as exc:
        if runtime is not None:
            abort = getattr(runtime, "abort_all_inference", None)
            verify_closed = getattr(runtime, "verify_all_inference_closed", None)
            if callable(abort) and callable(verify_closed):
                abort_exc: Exception | None = None
                try:
                    abort()
                except Exception as cleanup_exc:
                    abort_exc = cleanup_exc
                try:
                    verify_closed()
                except Exception as closure_exc:
                    # Never seal a terminal record while actual worker
                    # closure remains unproved.  The already-claimed root is
                    # consumed, so this cannot become a hidden retry path.
                    raise TatqaP19FormalStudyError(
                        "formal inference closure could not be proved"
                    ) from closure_exc
                if abort_exc is not None:
                    exc = abort_exc
                    stage = "terminal_inference_abort"
        control_claimed = control_root.is_dir()
        disposition = _invalid_disposition(exc, stage=stage)

    envelope = _terminal_envelope(
        disposition=disposition,
        runtime_fingerprint_self_sha256=fingerprint_sha,
        network_preflight=network_receipt,
    )
    result_path, file_sha, reopened = _persist_terminal(
        control_root,
        envelope,
        root_existed_for_this_attempt=control_claimed,
    )
    return {
        "disposition": reopened,
        "file_sha256": file_sha,
        "path": str(result_path),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen, offline-only one-shot TAT-QA P19 study"
    )
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--control-root", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--typed-runtime-python", required=True, type=Path)
    parser.add_argument("--hippo-runtime-python", required=True, type=Path)
    parser.add_argument("--qwen-model", required=True, type=Path)
    parser.add_argument("--minilm-asset-manifest", required=True, type=Path)
    parser.add_argument("--minilm-model", required=True, type=Path)
    parser.add_argument("--hippo-llm-model", required=True, type=Path)
    parser.add_argument("--hippo-embedding-model", required=True, type=Path)
    parser.add_argument("--hipporag-source", required=True, type=Path)
    parser.add_argument("--hippo-attestation", required=True, type=Path)
    parser.add_argument("--runtime-fingerprint", required=True, type=Path)
    parser.add_argument("--production-canary", required=True, type=Path)
    return parser


def _main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = FormalStudyConfig(
        project_root=args.project_root,
        control_root=args.control_root,
        work_root=args.work_root,
        typed_runtime_python=args.typed_runtime_python,
        hippo_runtime_python=args.hippo_runtime_python,
        qwen_model=args.qwen_model,
        minilm_asset_manifest=args.minilm_asset_manifest,
        minilm_model=args.minilm_model,
        hippo_llm_model=args.hippo_llm_model,
        hippo_embedding_model=args.hippo_embedding_model,
        hipporag_source=args.hipporag_source,
        hippo_attestation=args.hippo_attestation,
        runtime_fingerprint=args.runtime_fingerprint,
        production_canary=args.production_canary,
    )
    result = run_formal_study(config)
    summary = {
        "file_sha256": result["file_sha256"],
        "path": result["path"],
        "status": result["disposition"]["status"],
    }
    print(_canonical_bytes(summary).decode("ascii"), end="")
    return 0


def main() -> int:
    return _main()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_DEPENDENCIES",
    "FINAL_DISPOSITION_FILENAME",
    "FormalStudyConfig",
    "FormalStudyDependencies",
    "TatqaP19FormalStudyAlreadyConsumed",
    "TatqaP19FormalStudyError",
    "VERSION",
    "load_final_disposition",
    "main",
    "run_formal_study",
]
