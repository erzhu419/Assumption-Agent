"""One-shot offline entrypoint for the frozen BioASQ P1 formal study.

The source-free coordinate canary and the formal run deliberately share only
this small command-line entrypoint.  The formal source compiler, acquisition
boundary, controller, and action lanes are imported lazily after the complete
preflight has passed, so the canary process has no source path or source
reader.

The formal path creates one exclusive attempt root, authenticates the frozen
coordinate and official-HippoRAG evidence before source access, compiles the
source exactly once, and then delegates the late-qrel lifecycle to the frozen
controller.  It never retries a worker, reopens a completed attempt, or calls
an online/API evaluator.
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
from typing import Mapping, Sequence

from . import contract


VERSION = "bioasq_p1_formal_runtime_v1"
STUDY_ID = contract.STUDY_ID
FORMAL_CONFIG_SCHEMA = f"{VERSION}_formal_config_v1"
FORMAL_ATTEMPT_SCHEMA = f"{VERSION}_outer_attempt_v1"
FORMAL_TERMINAL_SCHEMA = f"{VERSION}_safe_terminal_v1"
FORMAL_FAILURE_SCHEMA = f"{VERSION}_safe_failure_v1"

FORMAL_ATTEMPT_FILENAME = "runtime_attempt.marker.json"
PREFLIGHT_RECEIPT_FILENAME = "preflight.safe.json"
FORMAL_TERMINAL_FILENAME = "runtime_terminal.safe.json"
FORMAL_FAILURE_FILENAME = "runtime_failure.safe.json"

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def _required_absolute(value: object, field: str) -> Path:
    if not isinstance(value, str):
        raise contract.BioasqP1FormalRuntimeError(
            f"{field} must be an absolute path string"
        )
    path = Path(value)
    if not path.is_absolute():
        raise contract.BioasqP1FormalRuntimeError(
            f"{field} must be absolute"
        )
    return path


def _required_int(
    value: object,
    field: str,
    *,
    minimum: int = 1,
    maximum: int = 14_400,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise contract.BioasqP1FormalRuntimeError(
            f"{field} is outside the frozen integer bound"
        )
    return value


def _required_sha256(value: object, field: str) -> str:
    return contract.required_sha256(value, field)


def _strict_object(
    value: object,
    *,
    keys: set[str],
    field: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise contract.BioasqP1FormalRuntimeError(
            f"{field} schema drifted"
        )
    return value


def _reject_duplicate_keys(
    pairs: Sequence[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise contract.BioasqP1FormalRuntimeError(
                "formal config contains a duplicate key"
            )
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise contract.BioasqP1FormalRuntimeError(
        f"formal config contains forbidden constant: {value}"
    )


def _read_canonical_config(path: Path) -> Mapping[str, object]:
    if not path.is_absolute():
        raise contract.BioasqP1FormalRuntimeError(
            "formal config path must be absolute"
        )
    contract.assert_no_symlink_components(path, "formal config")
    try:
        metadata = path.lstat()
        if (
            path.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
        ):
            raise contract.BioasqP1FormalRuntimeError(
                "formal config is not a direct file"
            )
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise contract.BioasqP1FormalRuntimeError(
            "formal config is invalid JSON"
        ) from exc
    if (
        not isinstance(value, Mapping)
        or raw
        not in {
            contract.canonical_bytes(value),
            contract.canonical_bytes(value, newline=True),
        }
    ):
        raise contract.BioasqP1FormalRuntimeError(
            "formal config is not canonical JSON"
        )
    return value


def _file_sha256(path: Path, field: str) -> str:
    contract.assert_no_symlink_components(path, field)
    try:
        metadata = path.lstat()
        if (
            path.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
        ):
            raise contract.BioasqP1FormalRuntimeError(
                f"{field} is not a direct file"
            )
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError as exc:
        raise contract.BioasqP1FormalRuntimeError(
            f"{field} could not be hashed"
        ) from exc


@dataclass(frozen=True, slots=True)
class FormalSourceInputs:
    source_path: Path
    p0_receipt_path: Path
    p0_private_manifest_path: Path

    @classmethod
    def from_payload(cls, value: object) -> "FormalSourceInputs":
        checked = _strict_object(
            value,
            keys={
                "p0_private_manifest_path",
                "p0_receipt_path",
                "source_path",
            },
            field="formal source inputs",
        )
        return cls(
            source_path=_required_absolute(
                checked["source_path"], "formal source path"
            ),
            p0_receipt_path=_required_absolute(
                checked["p0_receipt_path"], "P0 receipt path"
            ),
            p0_private_manifest_path=_required_absolute(
                checked["p0_private_manifest_path"],
                "P0 private manifest path",
            ),
        )


@dataclass(frozen=True, slots=True)
class CoordinateLaneInputs:
    runtime_python: Path
    project_root: Path
    minilm_asset_manifest: Path
    minilm_model_root: Path
    cross_encoder_model_root: Path
    timeout_seconds: int

    @classmethod
    def from_payload(cls, value: object) -> "CoordinateLaneInputs":
        checked = _strict_object(
            value,
            keys={
                "cross_encoder_model_root",
                "minilm_asset_manifest",
                "minilm_model_root",
                "project_root",
                "runtime_python",
                "timeout_seconds",
            },
            field="coordinate lane inputs",
        )
        return cls(
            runtime_python=_required_absolute(
                checked["runtime_python"], "coordinate runtime Python"
            ),
            project_root=_required_absolute(
                checked["project_root"], "coordinate project root"
            ),
            minilm_asset_manifest=_required_absolute(
                checked["minilm_asset_manifest"], "MiniLM asset manifest"
            ),
            minilm_model_root=_required_absolute(
                checked["minilm_model_root"], "MiniLM model root"
            ),
            cross_encoder_model_root=_required_absolute(
                checked["cross_encoder_model_root"],
                "cross-encoder model root",
            ),
            timeout_seconds=_required_int(
                checked["timeout_seconds"], "coordinate timeout"
            ),
        )


@dataclass(frozen=True, slots=True)
class HippoLaneInputs:
    runtime_python: Path
    worker_project_root: Path
    current_hardware_binding_path: Path
    local_llm_model: Path
    local_embedding_model: Path
    runtime_fingerprint_path: Path
    build_timeout_seconds: int
    retrieve_timeout_seconds: int

    @classmethod
    def from_payload(cls, value: object) -> "HippoLaneInputs":
        checked = _strict_object(
            value,
            keys={
                "build_timeout_seconds",
                "current_hardware_binding_path",
                "local_embedding_model",
                "local_llm_model",
                "retrieve_timeout_seconds",
                "runtime_fingerprint_path",
                "runtime_python",
                "worker_project_root",
            },
            field="HippoRAG lane inputs",
        )
        return cls(
            runtime_python=_required_absolute(
                checked["runtime_python"], "HippoRAG runtime Python"
            ),
            worker_project_root=_required_absolute(
                checked["worker_project_root"],
                "HippoRAG worker project root",
            ),
            current_hardware_binding_path=_required_absolute(
                checked["current_hardware_binding_path"],
                "current hardware binding",
            ),
            local_llm_model=_required_absolute(
                checked["local_llm_model"], "HippoRAG local LLM"
            ),
            local_embedding_model=_required_absolute(
                checked["local_embedding_model"],
                "HippoRAG local embedding model",
            ),
            runtime_fingerprint_path=_required_absolute(
                checked["runtime_fingerprint_path"],
                "HippoRAG runtime fingerprint",
            ),
            build_timeout_seconds=_required_int(
                checked["build_timeout_seconds"],
                "HippoRAG build timeout",
            ),
            retrieve_timeout_seconds=_required_int(
                checked["retrieve_timeout_seconds"],
                "HippoRAG retrieve timeout",
            ),
        )


@dataclass(frozen=True, slots=True)
class FormalRuntimeConfig:
    execution_binding_sha256: str
    formal_root: Path
    preflight_config_path: Path
    preflight_config_file_sha256: str
    preflight_config_self_sha256: str
    source: FormalSourceInputs
    coordinate: CoordinateLaneInputs
    hippo: HippoLaneInputs

    @classmethod
    def from_payload(cls, value: object) -> "FormalRuntimeConfig":
        checked = _strict_object(
            value,
            keys={
                "coordinate_lane",
                "execution_binding_sha256",
                "formal_root",
                "hippo_lane",
                "preflight_config_file_sha256",
                "preflight_config_path",
                "preflight_config_self_sha256",
                "schema",
                "self_sha256",
                "source_inputs",
            },
            field="formal runtime config",
        )
        if checked.get("schema") != FORMAL_CONFIG_SCHEMA:
            raise contract.BioasqP1FormalRuntimeError(
                "formal runtime config identity drifted"
            )
        contract.verify_self_hash(checked, "formal runtime config")
        result = cls(
            execution_binding_sha256=_required_sha256(
                checked["execution_binding_sha256"],
                "execution binding",
            ),
            formal_root=_required_absolute(
                checked["formal_root"], "formal root"
            ),
            preflight_config_path=_required_absolute(
                checked["preflight_config_path"],
                "preflight config path",
            ),
            preflight_config_file_sha256=_required_sha256(
                checked["preflight_config_file_sha256"],
                "preflight config file",
            ),
            preflight_config_self_sha256=_required_sha256(
                checked["preflight_config_self_sha256"],
                "preflight config self",
            ),
            source=FormalSourceInputs.from_payload(
                checked["source_inputs"]
            ),
            coordinate=CoordinateLaneInputs.from_payload(
                checked["coordinate_lane"]
            ),
            hippo=HippoLaneInputs.from_payload(checked["hippo_lane"]),
        )
        result._validate_cross_bindings()
        return result

    @classmethod
    def from_path(cls, path: Path) -> "FormalRuntimeConfig":
        return cls.from_payload(_read_canonical_config(path))

    def _validate_cross_bindings(self) -> None:
        if (
            self.formal_root == self.source.source_path
            or self.formal_root in self.source.source_path.parents
            or self.source.source_path in self.formal_root.parents
        ):
            raise contract.BioasqP1FormalRuntimeError(
                "formal output and source custody paths overlap"
            )
        if (
            self.coordinate.project_root
            == self.hippo.worker_project_root
        ):
            # The same deployed source tree is expected for both adapters.
            pass
        if (
            self.hippo.current_hardware_binding_path
            == self.preflight_config_path
        ):
            raise contract.BioasqP1FormalRuntimeError(
                "preflight config and hardware receipt overlap"
            )


def load_formal_runtime_config(path: Path) -> FormalRuntimeConfig:
    return FormalRuntimeConfig.from_path(path)


def _source_output_paths(
    root: Path,
) -> tuple[Path, Mapping[str, Path]]:
    """Return the private source-output root and its fixed path registry."""

    output_root = root / "formal_source"
    values = {
        "private_selection_secret": (
            output_root / "selection_secret.private.bin"
        ),
        "public_corpus": output_root / "corpus.public.json",
        "public_a_form": output_root / "A_form.public.json",
        "public_f_search": output_root / "F_search.public.json",
        "public_a_hold": output_root / "A_hold.public.json",
        "public_m_search": output_root / "M_search.public.json",
        "private_a_form_qrels": (
            output_root / "A_form.qrels.private.json"
        ),
        "private_a_hold_qrels": (
            output_root / "A_hold.qrels.private.json"
        ),
        "private_m_search_qrels": (
            output_root / "M_search.qrels.private.json"
        ),
        "safe_selection_receipt": output_root / "selection.safe.json",
    }
    return output_root, values


def _failure_hash(value: BaseException) -> tuple[str, str]:
    return (
        hashlib.sha256(
            type(value).__name__.encode("ascii", errors="replace")
        ).hexdigest(),
        hashlib.sha256(
            str(value).encode("utf-8", errors="replace")
        ).hexdigest(),
    )


def _safe_failure(
    *,
    root: Path,
    attempt_self_sha256: str,
    execution_binding_sha256: str,
    stage: str,
    source_compiler_invocation_count: int,
    exc: BaseException,
) -> None:
    exception_type_sha256, exception_message_sha256 = _failure_hash(exc)
    value = contract.with_self_hash(
        {
            "aggregate_only_public_receipt": True,
            "attempt_self_sha256": attempt_self_sha256,
            "execution_binding_sha256": execution_binding_sha256,
            "failure_exception_message_sha256": exception_message_sha256,
            "failure_exception_type_sha256": exception_type_sha256,
            "failure_stage": stage,
            "online_or_API_evaluator_calls": 0,
            "retry_count": 0,
            "schema": FORMAL_FAILURE_SCHEMA,
            "source_compiler_invocation_count": (
                source_compiler_invocation_count
            ),
            "status": "failed_closed_no_retry",
            "study_id": STUDY_ID,
        }
    )
    try:
        contract.exclusive_json(
            root / FORMAL_FAILURE_FILENAME,
            value,
            mode=0o600,
        )
    except BaseException:
        pass


def run_formal_once(
    config: FormalRuntimeConfig,
) -> Mapping[str, object]:
    """Execute one preflight-bound, offline formal attempt."""

    if not isinstance(config, FormalRuntimeConfig):
        raise contract.BioasqP1FormalRuntimeError(
            "formal runtime config type drifted"
        )
    root = contract.fresh_private_directory(
        config.formal_root, "formal runtime root"
    )
    marker = contract.with_self_hash(
        {
            "execution_binding_sha256": (
                config.execution_binding_sha256
            ),
            "online_or_API_evaluator_capability_present": False,
            "retry_count": 0,
            "schema": FORMAL_ATTEMPT_SCHEMA,
            "study_id": STUDY_ID,
        }
    )
    contract.exclusive_json(
        root / FORMAL_ATTEMPT_FILENAME,
        marker,
        mode=0o400,
    )
    stage = "preflight_config_identity"
    source_compiler_invocation_count = 0
    hippo_lane: object | None = None
    try:
        if (
            _file_sha256(
                config.preflight_config_path,
                "formal preflight config",
            )
            != config.preflight_config_file_sha256
        ):
            raise contract.BioasqP1FormalRuntimeError(
                "formal preflight config file hash drifted"
            )
        preflight_config = contract.load_runtime_config(
            config.preflight_config_path
        )
        if not isinstance(
            preflight_config, contract.FormalPreflightConfig
        ):
            raise contract.BioasqP1FormalRuntimeError(
                "formal preflight config type drifted"
            )
        if (
            preflight_config.execution_binding_sha256
            != config.execution_binding_sha256
            or contract.verify_self_hash(
                _read_canonical_config(
                    config.preflight_config_path
                ),
                "formal preflight config",
            )
            != config.preflight_config_self_sha256
            or preflight_config.coordinate_project_root
            != config.coordinate.project_root
            or preflight_config.hippo_worker_project_root
            != config.hippo.worker_project_root
            or preflight_config.bioasq_hardware_binding_path
            != config.hippo.current_hardware_binding_path
            or preflight_config.hippo_runtime_python
            != config.hippo.runtime_python
            or preflight_config.hippo_local_llm_model
            != config.hippo.local_llm_model
            or preflight_config.hippo_local_embedding_model
            != config.hippo.local_embedding_model
            or preflight_config.hippo_runtime_fingerprint_path
            != config.hippo.runtime_fingerprint_path
        ):
            raise contract.BioasqP1FormalRuntimeError(
                "formal/preflight configuration binding drifted"
            )

        stage = "offline_preflight_before_source_access"
        preflight_receipt = contract.verify_formal_preflight(
            preflight_config
        )
        contract.exclusive_json(
            root / PREFLIGHT_RECEIPT_FILENAME,
            preflight_receipt,
            mode=0o600,
        )

        # Formal-only code is unreachable until all source-free evidence,
        # backend bytes, and live hardware have authenticated successfully.
        from assumption_agent.benchmarks import (
            bioasq_p1_formal_controller_v1 as controller,
        )
        from assumption_agent.benchmarks import (
            bioasq_p1_formal_source_v2 as formal_source,
        )
        from .acquisition import SealedSourceAcquisitionBoundary
        from .lanes import CoordinateScorerLane, OfficialHippoLane

        stage = "create_fresh_formal_source_output_root"
        output_root, output_values = _source_output_paths(root)
        contract.fresh_private_directory(
            output_root, "formal source output root"
        )
        outputs = formal_source.FormalOutputPaths(**output_values)

        stage = "compile_formal_source_once"
        source_compiler_invocation_count += 1
        selection_receipt = formal_source.compile_formal_source(
            p0_receipt_path=config.source.p0_receipt_path,
            private_eligibility_manifest_path=(
                config.source.p0_private_manifest_path
            ),
            source_path=config.source.source_path,
            outputs=outputs,
        )

        stage = "construct_frozen_action_lanes"
        hippo_lane = OfficialHippoLane(
            runtime_python=config.hippo.runtime_python,
            worker_project_root=config.hippo.worker_project_root,
            current_hardware_binding_path=(
                config.hippo.current_hardware_binding_path
            ),
            local_llm_model=config.hippo.local_llm_model,
            local_embedding_model=config.hippo.local_embedding_model,
            runtime_fingerprint_path=(
                config.hippo.runtime_fingerprint_path
            ),
            lane_root=root / "official_hipporag_lane",
            build_timeout_seconds=(
                config.hippo.build_timeout_seconds
            ),
            retrieve_timeout_seconds=(
                config.hippo.retrieve_timeout_seconds
            ),
        )
        coordinate_lane = CoordinateScorerLane(
            runtime_python=config.coordinate.runtime_python,
            project_root=config.coordinate.project_root,
            minilm_asset_manifest=(
                config.coordinate.minilm_asset_manifest
            ),
            minilm_model_root=config.coordinate.minilm_model_root,
            cross_encoder_model_root=(
                config.coordinate.cross_encoder_model_root
            ),
            expected_model_binding_sha256=_required_sha256(
                preflight_receipt.get(
                    "coordinate_model_binding_sha256"
                ),
                "preflight coordinate model binding",
            ),
            lane_root=root / "coordinate_lane",
            timeout_seconds=config.coordinate.timeout_seconds,
        )
        controller_root = root / "formal_study"
        acquisition = SealedSourceAcquisitionBoundary(
            outputs=outputs,
            selection_receipt=selection_receipt,
            controller_root=controller_root,
            hippo_lane=hippo_lane,
        )

        stage = "run_frozen_formal_controller"
        controller_terminal = controller.run_formal_controller(
            work_root=controller_root,
            execution_binding_sha256=(
                config.execution_binding_sha256
            ),
            acquisition=acquisition,
            coordinate_scorer=coordinate_lane,
            hippo_runner=hippo_lane,
        )
        if not isinstance(controller_terminal, Mapping):
            raise contract.BioasqP1FormalRuntimeError(
                "formal controller terminal is unavailable"
            )
        controller_terminal_self = contract.verify_self_hash(
            controller_terminal, "formal controller terminal"
        )

        stage = "seal_outer_safe_terminal"
        terminal = contract.with_self_hash(
            {
                "aggregate_only_public_receipt": True,
                "attempt_self_sha256": marker["self_sha256"],
                "controller_status": controller_terminal.get("status"),
                "controller_terminal_self_sha256": (
                    controller_terminal_self
                ),
                "coordinate_worker_call_count": (
                    coordinate_lane.worker_call_count
                ),
                "execution_binding_sha256": (
                    config.execution_binding_sha256
                ),
                "formal_source_access_count": 1,
                "hipporag_build_call_count": hippo_lane.build_call_count,
                "hipporag_retrieve_call_count": (
                    hippo_lane.retrieve_call_count
                ),
                "item_query_document_qrel_action_or_per_item_score_values_published": (
                    False
                ),
                "online_or_API_evaluator_calls": 0,
                "preflight_receipt_self_sha256": (
                    preflight_receipt["self_sha256"]
                ),
                "retry_count": 0,
                "schema": FORMAL_TERMINAL_SCHEMA,
                "selection_receipt_self_sha256": (
                    selection_receipt["self_sha256"]
                ),
                "source_compiler_invocation_count": (
                    source_compiler_invocation_count
                ),
                "status": "terminal_complete",
                "study_id": STUDY_ID,
            }
        )
        contract.exclusive_json(
            root / FORMAL_TERMINAL_FILENAME,
            terminal,
            mode=0o600,
        )
        return terminal
    except BaseException as exc:
        _safe_failure(
            root=root,
            attempt_self_sha256=str(marker["self_sha256"]),
            execution_binding_sha256=(
                config.execution_binding_sha256
            ),
            stage=stage,
            source_compiler_invocation_count=(
                source_compiler_invocation_count
            ),
            exc=exc,
        )
        if isinstance(exc, contract.BioasqP1FormalRuntimeError):
            raise
        raise contract.BioasqP1FormalRuntimeError(
            "formal runtime failed closed"
        ) from exc
    finally:
        if hippo_lane is not None:
            close = getattr(hippo_lane, "close", None)
            if callable(close):
                close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one frozen offline BioASQ P1 lifecycle stage."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--source-free-coordinate-canary",
        action="store_true",
    )
    mode.add_argument("--formal", action="store_true")
    parser.add_argument("--config", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.config.is_absolute():
        raise contract.BioasqP1FormalRuntimeError(
            "runtime config path must be absolute"
        )
    if args.source_free_coordinate_canary:
        canary_config = contract.load_runtime_config(args.config)
        if not isinstance(
            canary_config, contract.CoordinateCanaryConfig
        ):
            raise contract.BioasqP1FormalRuntimeError(
                "coordinate canary config type drifted"
            )
        result = contract.run_source_free_coordinate_canary_once(
            canary_config
        )
    else:
        result = run_formal_once(
            load_formal_runtime_config(args.config)
        )
    os.write(1, contract.canonical_bytes(result, newline=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FORMAL_ATTEMPT_FILENAME",
    "FORMAL_CONFIG_SCHEMA",
    "FORMAL_FAILURE_FILENAME",
    "FORMAL_TERMINAL_FILENAME",
    "CoordinateLaneInputs",
    "FormalRuntimeConfig",
    "FormalSourceInputs",
    "HippoLaneInputs",
    "load_formal_runtime_config",
    "main",
    "run_formal_once",
]
