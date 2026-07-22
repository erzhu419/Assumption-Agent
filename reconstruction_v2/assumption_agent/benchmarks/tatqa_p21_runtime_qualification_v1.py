"""One-shot, pre-source production qualification for TAT-QA P21.

This entrypoint fingerprints the exact remote runtime, proves user-systemd
network isolation, executes the fixed public Qwen/MiniLM P0-vs-P1 canary twice,
and executes one public item-local official-HippoRAG canary.  It has no source
loader and refuses to start if any formal TAT-QA source or acquisition state
already exists.  A claimed qualification root is terminal on every failure;
there is no retry path hidden in this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

from assumption_agent.benchmarks import tatqa_p21_acquisition_v1 as acquisition
from assumption_agent.benchmarks import tatqa_p21_implementation_freeze_v1 as freeze
from assumption_agent.benchmarks import tatqa_p21_public_canary_v1 as canary
from replication_runtime.tatqa_p21_v1 import formal_runtime


VERSION = "tatqa_p21_runtime_qualification_v1"
DEFAULT_ROOT_RELATIVE = Path("artifacts/tatqa_p21_runtime_qualification_v1")
DEFAULT_FINGERPRINT_RELATIVE = Path(
    "manifests/tatqa_p21_composite_runtime_fingerprint_v1.json"
)
DEFAULT_CANARY_RELATIVE = Path(
    "manifests/tatqa_p21_public_synthetic_production_canary_v1.json"
)
MARKER_FILENAME = "qualification.one_shot_marker.json"
FAILURE_FILENAME = "qualification.terminal_failure.json"

class TatqaP21RuntimeQualificationError(RuntimeError):
    """The source-free remote qualification failed closed."""


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
        raise TatqaP21RuntimeQualificationError(
            "qualification receipt is not canonical JSON"
        ) from exc


def _semantic_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value).rstrip(b"\n")).hexdigest()


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> str:
    raw = _canonical_bytes(value)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise TatqaP21RuntimeQualificationError(
            "exclusive qualification receipt already exists"
        ) from exc
    if path.read_bytes() != raw or stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise TatqaP21RuntimeQualificationError(
            "qualification receipt reopen or mode drifted"
        )
    return hashlib.sha256(raw).hexdigest()


def runtime_inventory(
    *,
    typed_runtime_python: Path,
    hippo_runtime_python: Path,
    qwen_model: Path,
    minilm_manifest: Path,
    hippo_attestation: Path,
) -> dict[str, object]:
    """Return the same split-interpreter inventory formal entry rechecks."""

    try:
        return formal_runtime.runtime_inventory_snapshot(
            typed_runtime_python=typed_runtime_python,
            hippo_runtime_python=hippo_runtime_python,
            qwen_model=qwen_model,
            minilm_manifest=minilm_manifest,
            hippo_attestation=hippo_attestation,
        )
    except Exception as exc:
        raise TatqaP21RuntimeQualificationError(
            "split offline runtime inventory cannot be bound"
        ) from exc


def _terminal_failure(root: Path, stage: str, exc: BaseException) -> None:
    body = {
        "schema": f"{VERSION}_terminal_failure_v1",
        "status": "terminal_no_retry_requalification_or_formal_source_open",
        "failure_stage": stage,
        "failure_type_sha256": hashlib.sha256(
            f"{type(exc).__module__}.{type(exc).__qualname__}".encode("utf-8")
        ).hexdigest(),
        "formal_source_opened": False,
        "external_network_calls_other_than_none": 0,
        "api_or_online_evaluator_calls": 0,
    }
    receipt = {**body, "self_sha256": _semantic_hash(body)}
    try:
        _write_exclusive(root / FAILURE_FILENAME, receipt)
    except BaseException:
        pass


def run_runtime_qualification(
    *,
    project_root: str | Path,
    qualification_root: str | Path,
    typed_runtime_python: str | Path,
    hippo_runtime_python: str | Path,
    qwen_model: str | Path,
    minilm_asset_manifest: str | Path,
    minilm_model: str | Path,
    hippo_llm_model: str | Path,
    hippo_embedding_model: str | Path,
    hipporag_source: str | Path,
    hippo_attestation: str | Path,
    runtime_implementation_commit: str,
    fingerprint_output: str | Path,
    canary_output: str | Path,
) -> dict[str, object]:
    """Consume the sole source-free P21 production qualification attempt."""

    project = Path(project_root).resolve(strict=True)
    root = Path(qualification_root).expanduser().absolute()
    if root.exists() or root.is_symlink():
        raise TatqaP21RuntimeQualificationError(
            "runtime qualification root is already consumed"
        )
    forbidden = (
        project / acquisition.SOURCE_RECEIPT_RELATIVE,
        project / acquisition.SOURCE_ROOT_RELATIVE,
        project / acquisition.ACQUISITION_ROOT_RELATIVE,
    )
    if any(path.exists() or path.is_symlink() for path in forbidden):
        raise TatqaP21RuntimeQualificationError(
            "formal source/acquisition state predates runtime qualification"
        )
    fingerprint_path = Path(fingerprint_output).expanduser().absolute()
    canary_path = Path(canary_output).expanduser().absolute()
    if any(path.exists() or path.is_symlink() for path in (fingerprint_path, canary_path)):
        raise TatqaP21RuntimeQualificationError(
            "runtime qualification output is already consumed"
        )
    root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    root.mkdir(mode=0o700)
    marker_body = {
        "schema": f"{VERSION}_one_shot_marker_v1",
        "status": "started_before_model_or_formal_source_open",
        "study_design_self_sha256": acquisition.DESIGN_SELF_SHA256,
        "runtime_implementation_commit": runtime_implementation_commit,
        "formal_source_opened": False,
        "retry_requalification": 0,
    }
    _write_exclusive(
        root / MARKER_FILENAME,
        {**marker_body, "marker_sha256": _semantic_hash(marker_body)},
    )

    typed_runtime_python_path = Path(typed_runtime_python).expanduser().absolute()
    hippo_runtime_python_path = Path(hippo_runtime_python).expanduser().absolute()
    qwen = Path(qwen_model).expanduser().absolute()
    minilm_manifest = Path(minilm_asset_manifest).expanduser().absolute()
    minilm = Path(minilm_model).expanduser().absolute()
    hippo_llm = Path(hippo_llm_model).expanduser().absolute()
    hippo_embedding = Path(hippo_embedding_model).expanduser().absolute()
    hippo_source = Path(hipporag_source).expanduser().absolute()
    hippo_attestation_path = Path(hippo_attestation).expanduser().absolute()
    work_root = root / "work"
    stage = "safe_user_systemd_launch_envelope"
    typed_runner: formal_runtime.SystemdTypedPlanBatchRunner | None = None
    hippo_runner: formal_runtime.SystemdHippoByteRunner | None = None
    try:
        entry_launcher_phase = formal_runtime.user_systemd_launcher_phase_receipt(
            phase="entry"
        )
        stage = "runtime_inventory"
        inventory = runtime_inventory(
            typed_runtime_python=typed_runtime_python_path,
            hippo_runtime_python=hippo_runtime_python_path,
            qwen_model=qwen,
            minilm_manifest=minilm_manifest,
            hippo_attestation=hippo_attestation_path,
        )
        stage = "post_runtime_inventory_launch_envelope"
        post_inventory_launcher_phase = (
            formal_runtime.user_systemd_launcher_phase_receipt(
                phase="post_runtime_inventory"
            )
        )
        launcher_capability = (
            formal_runtime.user_systemd_launcher_capability_receipt(
                entry=entry_launcher_phase,
                post_runtime_inventory=post_inventory_launcher_phase,
            )
        )
        stage = "systemd_network_preflight"
        network = formal_runtime.systemd_network_preflight()
        paths = formal_runtime.RuntimePaths(
            project_root=project,
            typed_runtime_python=typed_runtime_python_path,
            hippo_runtime_python=hippo_runtime_python_path,
            qwen_model=qwen,
            minilm_asset_manifest=minilm_manifest,
            minilm_model=minilm,
            hippo_llm_model=hippo_llm,
            hippo_embedding_model=hippo_embedding,
            hipporag_source=hippo_source,
            hippo_attestation=hippo_attestation_path,
            fingerprint_manifest=fingerprint_path,
            work_root=work_root,
        )
        stage = "runtime_fingerprint"
        fingerprint = freeze.build_runtime_fingerprint(
            output_path=fingerprint_path,
            asset_roots={
                "Qwen": qwen,
                "MiniLM": minilm,
                "HippoRAG_LLM": hippo_llm,
                "HippoRAG_embedding": hippo_embedding,
                "HippoRAG_source": hippo_source,
            },
            runtime_inventory=inventory,
            safe_user_systemd_launch_envelope=launcher_capability,
            systemd_network_preflight=network,
            runtime_implementation_commit=runtime_implementation_commit,
        )
        stage = "runtime_fingerprint_reverification"
        formal_runtime.verify_runtime_fingerprint(paths)
        work_root.mkdir(mode=0o700)
        stage = "bound_minilm_initialization"
        encoder = formal_runtime.BoundMiniLMEncoder(paths)
        stage = "post_minilm_launch_envelope"
        post_minilm_launcher_phase = (
            formal_runtime.user_systemd_launcher_phase_receipt(
                phase="post_minilm"
            )
        )
        stage = "public_production_canary"
        typed_runner = formal_runtime.SystemdTypedPlanBatchRunner(paths)
        hippo_runner = formal_runtime.SystemdHippoByteRunner(paths)
        if (
            type(encoder) is not formal_runtime.BoundMiniLMEncoder
            or type(typed_runner) is not formal_runtime.SystemdTypedPlanBatchRunner
            or type(hippo_runner) is not formal_runtime.SystemdHippoByteRunner
        ):
            raise TatqaP21RuntimeQualificationError(
                "production qualification capability class drifted"
            )
        production_canary = canary.run_public_production_canary(
            runtime_fingerprint_path=fingerprint_path,
            output_path=canary_path,
            typed_plan_runner=typed_runner,
            encoder=encoder,
            hippo_runner=hippo_runner,
            post_minilm_launcher_phase_receipt=post_minilm_launcher_phase,
        )
        if (
            production_canary.get("hippo_canary_ran") is not True
            or production_canary.get("P1_retains_ordered_P0_top3") is not True
            or not isinstance(
                production_canary.get("P1_outside_P0_unit_count"), int
            )
            or production_canary["P1_outside_P0_unit_count"] < 1
            or production_canary.get("typed_plan_worker_receipt_source")
            != "capability_receipt_snapshot"
            or production_canary.get("minilm_worker_receipt_source")
            != "capability_receipt_snapshot"
            or production_canary.get("hippo_worker_receipt_source")
            != "capability_receipt_snapshot"
        ):
            raise TatqaP21RuntimeQualificationError(
                "production canary lacked real capability receipts or typed behavior"
            )
        acquisition.validate_production_canary_capability_receipts(
            production_canary,
            runtime_fingerprint=fingerprint,
        )
        stage = "worker_closure_verification"
        typed_runner.abort_all_workers()
        hippo_runner.abort_all_workers()
        typed_runner.verify_all_workers_closed()
        hippo_runner.verify_all_workers_closed()
        terminal_body = {
            "schema": f"{VERSION}_terminal_success_v1",
            "status": "qualified_before_formal_source_open",
            "runtime_fingerprint_self_sha256": fingerprint["self_sha256"],
            "safe_user_systemd_launch_envelope_self_sha256": (
                launcher_capability["self_sha256"]
            ),
            "safe_user_systemd_launch_phase_self_sha256s": {
                "entry": entry_launcher_phase["self_sha256"],
                "post_runtime_inventory": post_inventory_launcher_phase[
                    "self_sha256"
                ],
                "post_minilm": post_minilm_launcher_phase["self_sha256"],
            },
            "production_canary_self_sha256": production_canary["self_sha256"],
            "formal_source_opened": False,
            "retry_requalification": 0,
        }
        terminal = {**terminal_body, "self_sha256": _semantic_hash(terminal_body)}
        _write_exclusive(root / "qualification.terminal_success.json", terminal)
        return terminal
    except BaseException as exc:
        closure_error: BaseException | None = None
        for runner in (typed_runner, hippo_runner):
            if runner is None:
                continue
            try:
                runner.abort_all_workers()
                runner.verify_all_workers_closed()
            except BaseException as candidate:
                if closure_error is None:
                    closure_error = candidate
        if closure_error is not None:
            stage = "terminal_worker_closure_unproved"
            exc = closure_error
        _terminal_failure(root, stage, exc)
        raise TatqaP21RuntimeQualificationError(
            "runtime qualification failed terminally"
        ) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--qualification-root", required=True, type=Path)
    parser.add_argument("--typed-runtime-python", required=True, type=Path)
    parser.add_argument("--hippo-runtime-python", required=True, type=Path)
    parser.add_argument("--qwen-model", required=True, type=Path)
    parser.add_argument("--minilm-asset-manifest", required=True, type=Path)
    parser.add_argument("--minilm-model", required=True, type=Path)
    parser.add_argument("--hippo-llm-model", required=True, type=Path)
    parser.add_argument("--hippo-embedding-model", required=True, type=Path)
    parser.add_argument("--hipporag-source", required=True, type=Path)
    parser.add_argument("--hippo-attestation", required=True, type=Path)
    parser.add_argument("--runtime-implementation-commit", required=True)
    parser.add_argument("--fingerprint-output", required=True, type=Path)
    parser.add_argument("--canary-output", required=True, type=Path)
    return parser


def _main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_runtime_qualification(
        project_root=args.project_root,
        qualification_root=args.qualification_root,
        typed_runtime_python=args.typed_runtime_python,
        hippo_runtime_python=args.hippo_runtime_python,
        qwen_model=args.qwen_model,
        minilm_asset_manifest=args.minilm_asset_manifest,
        minilm_model=args.minilm_model,
        hippo_llm_model=args.hippo_llm_model,
        hippo_embedding_model=args.hippo_embedding_model,
        hipporag_source=args.hipporag_source,
        hippo_attestation=args.hippo_attestation,
        runtime_implementation_commit=args.runtime_implementation_commit,
        fingerprint_output=args.fingerprint_output,
        canary_output=args.canary_output,
    )
    print(_canonical_bytes(result).decode("ascii"), end="")
    return 0


def main() -> int:
    return _main()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CANARY_RELATIVE",
    "DEFAULT_FINGERPRINT_RELATIVE",
    "DEFAULT_ROOT_RELATIVE",
    "TatqaP21RuntimeQualificationError",
    "VERSION",
    "main",
    "run_runtime_qualification",
    "runtime_inventory",
]
