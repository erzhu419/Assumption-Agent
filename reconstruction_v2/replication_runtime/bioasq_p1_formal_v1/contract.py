"""Frozen infrastructure contract for the one-shot BioASQ P1 formal study.

This module has two deliberately narrow responsibilities:

* run one source-free, coordinate-only canary over a fixed synthetic
  2,900-passage corpus and one synthetic question; and
* authenticate that canary together with the already-successful DSTC9 v5
  official-HippoRAG canary, its copied hardware binding, and the unchanged
  generic HippoRAG backend before any BioASQ formal source is opened.

The coordinate worker's score vectors stay inside its private work root.
Only aggregate counters and cryptographic commitments are copied into the
safe canary receipt.  The official-HippoRAG canary is never rerun.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import stat
from typing import Any

from assumption_agent.benchmarks import bioasq_p1_typed_core_v1 as core
from replication_runtime.bioasq_coordinate_scorer_v1 import (
    adapter as coordinate_adapter,
)
from replication_runtime.bioasq_coordinate_scorer_v1 import (
    contract as coordinate_contract,
)
from replication_runtime.dstc9_official_hipporag_v1 import runtime_binding


VERSION = "bioasq_p1_formal_runtime_contract_v1"
STUDY_ID = core.STUDY_ID
LEGACY_HIPPO_STUDY_ID = (
    "DSTC9_P1_HIERARCHICAL_KNOWLEDGE_EVALUATOR_L5_V1"
)

COORDINATE_CANARY_CONFIG_SCHEMA = (
    f"{VERSION}_coordinate_canary_config_v1"
)
FORMAL_PREFLIGHT_CONFIG_SCHEMA = f"{VERSION}_formal_preflight_config_v1"
COORDINATE_CANARY_SCHEMA = (
    "bioasq_p1_source_free_coordinate_canary_receipt_v1"
)
COORDINATE_CANARY_FAILURE_SCHEMA = (
    "bioasq_p1_source_free_coordinate_canary_failure_v1"
)
FORMAL_PREFLIGHT_RECEIPT_SCHEMA = (
    "bioasq_p1_offline_formal_preflight_receipt_v1"
)

CANARY_ATTEMPT_FILENAME = "coordinate_canary_attempt.marker.json"
CANARY_RECEIPT_FILENAME = "coordinate_canary.safe.json"
CURRENT_HARDWARE_FILENAME = "current_hardware.safe.json"

SYNTHETIC_CORPUS_COUNT = 2_900
SYNTHETIC_QUERY_COUNT = 1
SYNTHETIC_UNIQUE_PASSAGE_TEXT_COUNT = 5

# A previous source-free DSTC9 v5 run already qualified this exact generic
# official-HippoRAG backend and exact 311linux hardware.  These identities
# are evidence, not configurable alternatives.
LEGACY_HIPPO_CANARY_FILE_SHA256 = (
    "a18412b30d72f8530073d5d1401481ee"
    "4460c1acaaa0a20fcf44348dd3c98ac0"
)
LEGACY_HIPPO_CANARY_SELF_SHA256 = (
    "135232740adab0e2478ea476d277cf669"
    "e35f9a5bbf52e7f1e966bafead9708f"
)
LEGACY_HARDWARE_FILE_SHA256 = (
    "9fcd26da6391c0619c96dad2d3dceea9"
    "212d5bb316332927e287dc71588e69d4"
)
LEGACY_HARDWARE_SELF_SHA256 = (
    "18bcb732132b763a17d946f2e58d8527"
    "b7a298c8620ed97b2976bdcfc7557655"
)

GENERIC_HIPPO_BACKEND_SHA256 = {
    "adapter.py": (
        "83c5fa7bf63aba51ffc21c4c5dfb507a2"
        "2f6086c16d87617d8ef27e9816a5586"
    ),
    "contract.py": (
        "b02b3f8e547568110c75e09757a9b961"
        "110f60d430a976f8731224136c681a43"
    ),
    "runtime_binding.py": (
        "f2224e6f3c15ac1f7fbaa9d79faaac3"
        "6a80becd52f376e2819d9cc9c5285593f"
    ),
    "worker.py": (
        "bebe5053f0cd5513ffe5ccefd9fb9c3d"
        "767fbe48202563b8675fd4bd351a47e4"
    ),
}

COORDINATE_BACKEND_SHA256 = {
    "assumption_agent/benchmarks/bioasq_p1_typed_core_v1.py": (
        "6bfd386431b977043f43eac0984a67b68"
        "8fad9def276d37902b2fb3c4cff9342"
    ),
    "replication_runtime/bioasq_coordinate_scorer_v1/adapter.py": (
        "f514d0088210ab1b829a8bd3c436e110"
        "ae2bf0f1f54e72cb9306081b50ef9347"
    ),
    "replication_runtime/bioasq_coordinate_scorer_v1/contract.py": (
        "9f96e27d2f5b5c933d74b75afa04c5a4"
        "f7e2be4823b258faa0f0511be401c457"
    ),
    "replication_runtime/bioasq_coordinate_scorer_v1/worker.py": (
        "98cac3b8a005b8bc1d5aaa600faefa14c"
        "e338e212c3bb7d194d15484b1dba316"
    ),
}

_GENERIC_HIPPO_RELATIVE_ROOT = Path(
    "replication_runtime/dstc9_official_hipporag_v1"
)
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+-]{0,255}\Z")


class BioasqP1FormalRuntimeError(RuntimeError):
    """A source-free canary or preformal infrastructure check failed closed."""


def canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    """Encode strict deterministic ASCII JSON."""

    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise BioasqP1FormalRuntimeError(
            "runtime value is not canonical JSON"
        ) from exc
    return raw + (b"\n" if newline else b"")


def stable_hash(value: object) -> str:
    """Hash the canonical JSON encoding of ``value``."""

    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def with_self_hash(body: Mapping[str, object]) -> dict[str, object]:
    """Return a copy of ``body`` with one canonical self commitment."""

    if not isinstance(body, Mapping) or "self_sha256" in body:
        raise BioasqP1FormalRuntimeError("self hash was supplied twice")
    result = dict(body)
    result["self_sha256"] = stable_hash(result)
    return result


def required_sha256(value: object, field: str) -> str:
    """Return a validated lowercase SHA-256 digest."""

    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise BioasqP1FormalRuntimeError(f"{field} is not a SHA-256 digest")
    return value


def verify_self_hash(value: Mapping[str, object], field: str) -> str:
    """Verify and return a canonical object's self commitment."""

    if not isinstance(value, Mapping):
        raise BioasqP1FormalRuntimeError(f"{field} is not an object")
    body = dict(value)
    claimed = required_sha256(
        body.pop("self_sha256", None), f"{field} self hash"
    )
    if not hmac.compare_digest(claimed, stable_hash(body)):
        raise BioasqP1FormalRuntimeError(f"{field} self hash drifted")
    return claimed


def _required_absolute(path: Path, field: str) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise BioasqP1FormalRuntimeError(f"{field} must be absolute")
    return path


def assert_no_symlink_components(path: Path, field: str) -> None:
    """Reject a path if any existing component is a symbolic link."""

    absolute = _required_absolute(path, field).absolute()
    for component in (*reversed(absolute.parents), absolute):
        if component.is_symlink():
            raise BioasqP1FormalRuntimeError(
                f"{field} contains a symbolic-link component"
            )


def _direct_file(path: Path, field: str) -> Path:
    absolute = _required_absolute(path, field).absolute()
    assert_no_symlink_components(absolute, field)
    try:
        metadata = absolute.lstat()
    except OSError as exc:
        raise BioasqP1FormalRuntimeError(f"{field} is unavailable") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or absolute.is_symlink()
    ):
        raise BioasqP1FormalRuntimeError(f"{field} is not a direct file")
    return absolute


def _direct_directory(path: Path, field: str) -> Path:
    absolute = _required_absolute(path, field).absolute()
    assert_no_symlink_components(absolute, field)
    try:
        metadata = absolute.lstat()
    except OSError as exc:
        raise BioasqP1FormalRuntimeError(f"{field} is unavailable") from exc
    if not stat.S_ISDIR(metadata.st_mode) or absolute.is_symlink():
        raise BioasqP1FormalRuntimeError(f"{field} is not a direct directory")
    return absolute


def _sha256_file(path: Path, field: str) -> str:
    direct = _direct_file(path, field)
    digest = hashlib.sha256()
    try:
        with direct.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise BioasqP1FormalRuntimeError(f"{field} could not be hashed") from exc
    return digest.hexdigest()


def exclusive_json(
    path: Path,
    value: Mapping[str, object],
    *,
    mode: int,
) -> None:
    """Persist one canonical artifact with O_EXCL and a fixed mode."""

    absolute = _required_absolute(path, "exclusive JSON path").absolute()
    if mode not in {0o400, 0o600}:
        raise BioasqP1FormalRuntimeError("exclusive JSON mode is invalid")
    assert_no_symlink_components(absolute.parent, "exclusive JSON parent")
    raw = canonical_bytes(value, newline=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(absolute, flags, mode)
        os.fchmod(descriptor, mode)
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write")
            view = view[written:]
        os.fsync(descriptor)
    except OSError as exc:
        raise BioasqP1FormalRuntimeError(
            f"exclusive artifact failed: {absolute.name}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    metadata = absolute.lstat()
    if (
        absolute.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != mode
        or metadata.st_size != len(raw)
    ):
        raise BioasqP1FormalRuntimeError(
            f"exclusive artifact metadata drifted: {absolute.name}"
        )


def fresh_private_directory(path: Path, field: str) -> Path:
    """Create one new mode-0700 directory without following symlinks."""

    absolute = _required_absolute(path, field).absolute()
    assert_no_symlink_components(absolute.parent, f"{field} parent")
    try:
        absolute.mkdir(mode=0o700, parents=False, exist_ok=False)
        os.chmod(absolute, 0o700)
    except OSError as exc:
        raise BioasqP1FormalRuntimeError(
            f"{field} is not a fresh private directory"
        ) from exc
    metadata = absolute.lstat()
    if (
        absolute.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise BioasqP1FormalRuntimeError(f"{field} mode drifted")
    return absolute


def _strict_config_value(path: Path) -> Mapping[str, object]:
    direct = _direct_file(path, "runtime config")
    try:
        raw = direct.read_bytes()
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        BioasqP1FormalRuntimeError,
    ) as exc:
        if isinstance(exc, BioasqP1FormalRuntimeError):
            raise
        raise BioasqP1FormalRuntimeError("runtime config is invalid JSON") from exc
    if (
        not isinstance(value, Mapping)
        or raw
        not in {
            canonical_bytes(value),
            canonical_bytes(value, newline=True),
        }
    ):
        raise BioasqP1FormalRuntimeError(
            "runtime config is not canonical JSON"
        )
    return value


def _reject_duplicate_keys(
    pairs: Sequence[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise BioasqP1FormalRuntimeError(
                "runtime config contains a duplicate key"
            )
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise BioasqP1FormalRuntimeError(
        f"runtime config contains forbidden constant: {value}"
    )


@dataclass(frozen=True, slots=True)
class CoordinateCanaryConfig:
    """All source-free inputs for one coordinate-only infrastructure canary."""

    canary_root: Path
    canary_binding_sha256: str
    runtime_python: Path
    project_root: Path
    minilm_asset_manifest: Path
    minilm_model_root: Path
    cross_encoder_model_root: Path
    timeout_seconds: int = 14_400

    def __post_init__(self) -> None:
        for name in (
            "canary_root",
            "runtime_python",
            "project_root",
            "minilm_asset_manifest",
            "minilm_model_root",
            "cross_encoder_model_root",
        ):
            _required_absolute(getattr(self, name), name)
        required_sha256(self.canary_binding_sha256, "canary binding")
        if (
            type(self.timeout_seconds) is not int
            or not 1 <= self.timeout_seconds <= 14_400
        ):
            raise BioasqP1FormalRuntimeError(
                "coordinate timeout is outside the frozen bound"
            )

    @classmethod
    def from_payload(cls, value: object) -> "CoordinateCanaryConfig":
        path_fields = (
            "canary_root",
            "runtime_python",
            "project_root",
            "minilm_asset_manifest",
            "minilm_model_root",
            "cross_encoder_model_root",
        )
        scalar_fields = ("canary_binding_sha256", "timeout_seconds")
        expected = {
            "schema",
            "self_sha256",
            *path_fields,
            *scalar_fields,
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != expected
            or value.get("schema") != COORDINATE_CANARY_CONFIG_SCHEMA
        ):
            raise BioasqP1FormalRuntimeError(
                "coordinate canary config schema drifted"
            )
        verify_self_hash(value, "coordinate canary config")
        return cls(
            **{name: Path(str(value[name])) for name in path_fields},
            **{name: value[name] for name in scalar_fields},
        )


@dataclass(frozen=True, slots=True)
class FormalPreflightConfig:
    """Frozen evidence paths checked before formal source access."""

    execution_binding_sha256: str
    coordinate_canary_binding_sha256: str
    coordinate_canary_receipt_path: Path
    coordinate_canary_receipt_file_sha256: str
    coordinate_canary_receipt_self_sha256: str
    bioasq_hardware_binding_path: Path
    bioasq_hardware_binding_file_sha256: str
    bioasq_hardware_binding_self_sha256: str
    legacy_hippo_canary_receipt_path: Path
    legacy_hardware_binding_path: Path
    coordinate_project_root: Path
    hippo_worker_project_root: Path
    hippo_runtime_python: Path
    hippo_local_llm_model: Path
    hippo_local_embedding_model: Path
    hippo_runtime_fingerprint_path: Path

    def __post_init__(self) -> None:
        for name in (
            "coordinate_canary_receipt_path",
            "bioasq_hardware_binding_path",
            "legacy_hippo_canary_receipt_path",
            "legacy_hardware_binding_path",
            "coordinate_project_root",
            "hippo_worker_project_root",
            "hippo_runtime_python",
            "hippo_local_llm_model",
            "hippo_local_embedding_model",
            "hippo_runtime_fingerprint_path",
        ):
            _required_absolute(getattr(self, name), name)
        for name in (
            "execution_binding_sha256",
            "coordinate_canary_binding_sha256",
            "coordinate_canary_receipt_file_sha256",
            "coordinate_canary_receipt_self_sha256",
            "bioasq_hardware_binding_file_sha256",
            "bioasq_hardware_binding_self_sha256",
        ):
            required_sha256(getattr(self, name), name)

    @classmethod
    def from_payload(cls, value: object) -> "FormalPreflightConfig":
        path_fields = (
            "coordinate_canary_receipt_path",
            "bioasq_hardware_binding_path",
            "legacy_hippo_canary_receipt_path",
            "legacy_hardware_binding_path",
            "coordinate_project_root",
            "hippo_worker_project_root",
            "hippo_runtime_python",
            "hippo_local_llm_model",
            "hippo_local_embedding_model",
            "hippo_runtime_fingerprint_path",
        )
        scalar_fields = (
            "execution_binding_sha256",
            "coordinate_canary_binding_sha256",
            "coordinate_canary_receipt_file_sha256",
            "coordinate_canary_receipt_self_sha256",
            "bioasq_hardware_binding_file_sha256",
            "bioasq_hardware_binding_self_sha256",
        )
        expected = {
            "schema",
            "self_sha256",
            *path_fields,
            *scalar_fields,
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != expected
            or value.get("schema") != FORMAL_PREFLIGHT_CONFIG_SCHEMA
        ):
            raise BioasqP1FormalRuntimeError(
                "formal preflight config schema drifted"
            )
        verify_self_hash(value, "formal preflight config")
        return cls(
            **{name: Path(str(value[name])) for name in path_fields},
            **{name: value[name] for name in scalar_fields},
        )


def load_runtime_config(
    path: Path,
) -> CoordinateCanaryConfig | FormalPreflightConfig:
    """Load exactly one of the two canonical config schemas."""

    value = _strict_config_value(path)
    if value.get("schema") == COORDINATE_CANARY_CONFIG_SCHEMA:
        return CoordinateCanaryConfig.from_payload(value)
    if value.get("schema") == FORMAL_PREFLIGHT_CONFIG_SCHEMA:
        return FormalPreflightConfig.from_payload(value)
    raise BioasqP1FormalRuntimeError("runtime config identity drifted")


def _synthetic_coordinate_input() -> dict[str, object]:
    passage_texts = (
        "Aspirin irreversibly inhibits platelet cyclooxygenase activity.",
        "Platelets participate in primary hemostasis and thrombus formation.",
        "A placebo contains no pharmacologically active study treatment.",
        "Genomic sequencing can identify inherited nucleotide variants.",
        "Randomized trials compare prospectively assigned interventions.",
    )
    passages = tuple(
        core.passage_public_payload(
            core.Passage(
                ordinal=ordinal,
                text=passage_texts[ordinal % len(passage_texts)],
            )
        )
        for ordinal in range(SYNTHETIC_CORPUS_COUNT)
    )
    return coordinate_contract.input_payload(
        passages=passages,
        queries=({"text": "Does aspirin inhibit platelet activity?"},),
    )


def _validate_coordinate_worker_receipt(
    output: object,
    *,
    input_self_sha256: str,
) -> Mapping[str, object]:
    if not isinstance(output, Mapping):
        raise BioasqP1FormalRuntimeError(
            "coordinate canary output is unavailable"
        )
    rows = output.get("rows")
    receipt = output.get("receipt")
    if (
        output.get("schema") != coordinate_contract.OUTPUT_SCHEMA
        or output.get("study_id") != STUDY_ID
        or output.get("query_count") != SYNTHETIC_QUERY_COUNT
        or output.get("corpus_count") != SYNTHETIC_CORPUS_COUNT
        or output.get("input_self_sha256") != input_self_sha256
        or not isinstance(rows, list)
        or len(rows) != SYNTHETIC_QUERY_COUNT
        or not isinstance(receipt, Mapping)
    ):
        raise BioasqP1FormalRuntimeError(
            "coordinate canary output identity drifted"
        )
    expected = {
        "cross_encoder_call_count": 2,
        "cross_encoder_model_load_count": 1,
        "cross_encoder_pair_count": (
            2 * SYNTHETIC_QUERY_COUNT * SYNTHETIC_CORPUS_COUNT
        ),
        "cuda_visible_devices": "1",
        "dynamic_resize_count": 0,
        "minilm_constructor_canary_encode_call_count": 2,
        "minilm_formal_batch_encode_call_count": 1,
        "minilm_model_load_count": 1,
        "minilm_passage_count": SYNTHETIC_CORPUS_COUNT,
        "minilm_query_variant_count": (
            len(coordinate_contract.DENSE_SCORE_NAMES)
            * SYNTHETIC_QUERY_COUNT
        ),
        "minilm_text_count": (
            SYNTHETIC_CORPUS_COUNT
            + len(coordinate_contract.DENSE_SCORE_NAMES)
            * SYNTHETIC_QUERY_COUNT
        ),
        "minilm_total_encode_call_count": 3,
        "network_access": "denied",
        "physical_gpu": 1,
        "query_count": SYNTHETIC_QUERY_COUNT,
        "retry_count": 0,
        "status": "passed_private_coordinate_scoring_once",
        "study_id": STUDY_ID,
        "typed_core_sha256": coordinate_contract.TYPED_CORE_SHA256,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise BioasqP1FormalRuntimeError(
            "coordinate canary worker counters drifted"
        )
    for name in (
        "model_binding_sha256",
        "receipt_sha256",
        "score_bundle_sha256",
    ):
        required_sha256(receipt.get(name), f"coordinate worker {name}")
    required_sha256(output.get("self_sha256"), "coordinate output self hash")
    # The vectors are intentionally inspected only for structural coverage.
    # No vector or per-passage value is copied into the safe receipt.
    row = rows[0]
    if (
        not isinstance(row, Mapping)
        or row.get("query_ordinal") != 0
        or not isinstance(row.get("vectors"), Mapping)
        or set(row["vectors"]) != set(coordinate_contract.SCORE_NAMES)
    ):
        raise BioasqP1FormalRuntimeError(
            "coordinate canary private vector registry drifted"
        )
    return receipt


CoordinateRun = Callable[..., Mapping[str, object]]
HardwareCapture = Callable[..., Mapping[str, object]]


def _validate_new_hardware_binding(
    value: object,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise BioasqP1FormalRuntimeError(
            "BioASQ current-hardware binding is unavailable"
        )
    supplied = dict(value)
    hardware = supplied.get("hardware")
    boundary = supplied.get("source_free_boundary")
    if (
        set(supplied)
        != {
            "capture_id",
            "hardware",
            "schema",
            "self_sha256",
            "source_free_boundary",
            "status",
            "study_id",
        }
        or supplied.get("schema")
        != runtime_binding.CURRENT_HARDWARE_SCHEMA
        or supplied.get("status")
        != runtime_binding.CURRENT_HARDWARE_STATUS
        or supplied.get("study_id") != STUDY_ID
        or not isinstance(hardware, Mapping)
        or not isinstance(boundary, Mapping)
        or boundary
        != {
            "capture_scope": (
                "hardware_only_no_model_source_or_evaluator_action_v1"
            ),
            "external_network_call_count": 0,
            "formal_source_open_count": 0,
            "old_P17_driver_or_kernel_used_as_requirement": False,
        }
    ):
        raise BioasqP1FormalRuntimeError(
            "BioASQ current-hardware binding drifted"
        )
    verify_self_hash(supplied, "BioASQ current-hardware binding")
    return supplied


def run_source_free_coordinate_canary_once(
    config: CoordinateCanaryConfig,
    *,
    run_callable: CoordinateRun = (
        coordinate_adapter.run_bioasq_coordinate_scorer_v1
    ),
    hardware_capture_callable: HardwareCapture = (
        runtime_binding.capture_current_study_hardware_binding
    ),
) -> Mapping[str, object]:
    """Run one GPU1 coordinate canary with no source or evaluator channel."""

    if not isinstance(config, CoordinateCanaryConfig):
        raise BioasqP1FormalRuntimeError(
            "coordinate canary config type drifted"
        )
    root = fresh_private_directory(config.canary_root, "canary root")
    marker = with_self_hash(
        {
            "canary_binding_sha256": config.canary_binding_sha256,
            "formal_source_capability_present": False,
            "online_or_API_evaluator_capability_present": False,
            "retry_count": 0,
            "schema": f"{VERSION}_coordinate_canary_attempt_v1",
            "study_id": STUDY_ID,
        }
    )
    exclusive_json(
        root / CANARY_ATTEMPT_FILENAME,
        marker,
        mode=0o400,
    )
    stage = "construct_fixed_source_free_input"
    try:
        stage = "capture_BioASQ_current_hardware_once"
        try:
            captured = hardware_capture_callable(
                study_id=STUDY_ID,
                capture_id=(
                    "bioasq-p1-coordinate-canary-"
                    f"{config.canary_binding_sha256[:16]}"
                ),
            )
        except Exception as exc:
            raise BioasqP1FormalRuntimeError(
                "BioASQ source-free hardware capture failed"
            ) from exc
        hardware_binding = _validate_new_hardware_binding(captured)
        hardware_path = root / CURRENT_HARDWARE_FILENAME
        exclusive_json(hardware_path, hardware_binding, mode=0o600)
        hardware_raw = canonical_bytes(hardware_binding, newline=True)
        hardware_file_sha256 = hashlib.sha256(
            hardware_raw
        ).hexdigest()
        hardware_self_sha256 = required_sha256(
            hardware_binding.get("self_sha256"),
            "BioASQ current-hardware binding",
        )

        stage = "construct_fixed_source_free_input"
        payload = _synthetic_coordinate_input()
        input_self_sha256 = required_sha256(
            payload.get("self_sha256"), "coordinate canary input"
        )
        stage = "run_coordinate_worker_once"
        output = run_callable(
            input_value=payload,
            runtime_python=config.runtime_python,
            project_root=config.project_root,
            minilm_asset_manifest=config.minilm_asset_manifest,
            minilm_model_root=config.minilm_model_root,
            cross_encoder_model_root=config.cross_encoder_model_root,
            work_root=root / "private_coordinate_worker",
            timeout_seconds=config.timeout_seconds,
        )
        worker_receipt = _validate_coordinate_worker_receipt(
            output,
            input_self_sha256=input_self_sha256,
        )
        receipt = with_self_hash(
            {
                "aggregate_only_public_receipt": True,
                "canary_attempt_self_sha256": marker["self_sha256"],
                "canary_binding_sha256": config.canary_binding_sha256,
                "coordinate_gpu": 1,
                "coordinate_worker_count": 1,
                "current_hardware_binding_file_sha256": (
                    hardware_file_sha256
                ),
                "current_hardware_binding_self_sha256": (
                    hardware_self_sha256
                ),
                "current_hardware_binding_study_id": STUDY_ID,
                "current_hardware_capture_count": 1,
                "formal_action_count": 0,
                "formal_evaluator_count": 0,
                "formal_score_count": 0,
                "formal_source_access_count": 0,
                "minilm_constructor_canary_encode_call_count": 2,
                "minilm_formal_batch_encode_call_count": 1,
                "minilm_total_encode_call_count": 3,
                "model_binding_sha256": (
                    worker_receipt["model_binding_sha256"]
                ),
                "online_or_API_evaluator_calls": 0,
                "private_output_self_sha256": output["self_sha256"],
                "private_score_bundle_sha256": (
                    worker_receipt["score_bundle_sha256"]
                ),
                "private_vector_values_published": False,
                "query_count": SYNTHETIC_QUERY_COUNT,
                "retry_count": 0,
                "schema": COORDINATE_CANARY_SCHEMA,
                "status": "passed_source_free_coordinate_canary_once",
                "study_id": STUDY_ID,
                "synthetic_corpus_count": SYNTHETIC_CORPUS_COUNT,
                "synthetic_unique_passage_text_count": (
                    SYNTHETIC_UNIQUE_PASSAGE_TEXT_COUNT
                ),
                "worker_receipt_sha256": (
                    worker_receipt["receipt_sha256"]
                ),
            }
        )
        exclusive_json(
            root / CANARY_RECEIPT_FILENAME,
            receipt,
            mode=0o600,
        )
        return receipt
    except BaseException as exc:
        failure = with_self_hash(
            {
                "aggregate_only_public_receipt": True,
                "canary_attempt_self_sha256": marker["self_sha256"],
                "canary_binding_sha256": config.canary_binding_sha256,
                "failure_exception_message_sha256": hashlib.sha256(
                    str(exc).encode("utf-8", errors="replace")
                ).hexdigest(),
                "failure_exception_type_sha256": hashlib.sha256(
                    type(exc).__name__.encode(
                        "ascii", errors="replace"
                    )
                ).hexdigest(),
                "failure_stage": stage,
                "formal_action_count": 0,
                "formal_evaluator_count": 0,
                "formal_score_count": 0,
                "formal_source_access_count": 0,
                "online_or_API_evaluator_calls": 0,
                "retry_count": 0,
                "schema": COORDINATE_CANARY_FAILURE_SCHEMA,
                "status": "failed_source_free_coordinate_canary_no_retry",
                "study_id": STUDY_ID,
            }
        )
        try:
            exclusive_json(
                root / CANARY_RECEIPT_FILENAME,
                failure,
                mode=0o600,
            )
        except BaseException:
            pass
        if isinstance(exc, BioasqP1FormalRuntimeError):
            raise
        raise BioasqP1FormalRuntimeError(
            "source-free coordinate canary failed closed"
        ) from exc


def _load_exact_receipt(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_self_sha256: str,
    field: str,
) -> dict[str, object]:
    direct = _direct_file(path, field)
    metadata = direct.lstat()
    try:
        raw = direct.read_bytes()
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        BioasqP1FormalRuntimeError,
    ) as exc:
        if isinstance(exc, BioasqP1FormalRuntimeError):
            raise
        raise BioasqP1FormalRuntimeError(f"{field} is unavailable") from exc
    if (
        stat.S_IMODE(metadata.st_mode) != 0o600
        or raw != canonical_bytes(value, newline=True)
        or hashlib.sha256(raw).hexdigest()
        != required_sha256(expected_file_sha256, f"{field} file hash")
        or not isinstance(value, dict)
        or value.get("self_sha256")
        != required_sha256(expected_self_sha256, f"{field} self hash")
        or verify_self_hash(value, field) != expected_self_sha256
    ):
        raise BioasqP1FormalRuntimeError(f"{field} binding drifted")
    return value


def verify_generic_hippo_backend(
    worker_project_root: Path,
) -> Mapping[str, str]:
    """Verify the four unchanged generic backend files byte-for-byte."""

    project = _direct_directory(
        worker_project_root, "HippoRAG worker project root"
    )
    backend_root = project / _GENERIC_HIPPO_RELATIVE_ROOT
    observed: dict[str, str] = {}
    for filename, expected in sorted(GENERIC_HIPPO_BACKEND_SHA256.items()):
        digest = _sha256_file(
            backend_root / filename,
            f"generic HippoRAG backend {filename}",
        )
        if not hmac.compare_digest(digest, expected):
            raise BioasqP1FormalRuntimeError(
                f"generic HippoRAG backend drifted: {filename}"
            )
        observed[filename] = digest
    return observed


def verify_coordinate_backend(
    coordinate_project_root: Path,
) -> Mapping[str, str]:
    """Verify the typed core and coordinate worker stack byte-for-byte."""

    project = _direct_directory(
        coordinate_project_root, "coordinate project root"
    )
    observed: dict[str, str] = {}
    for relative, expected in sorted(COORDINATE_BACKEND_SHA256.items()):
        digest = _sha256_file(
            project / relative,
            f"coordinate backend {relative}",
        )
        if not hmac.compare_digest(digest, expected):
            raise BioasqP1FormalRuntimeError(
                f"coordinate backend drifted: {relative}"
            )
        observed[relative] = digest
    return observed


def _validate_coordinate_canary_receipt(
    value: Mapping[str, object],
    *,
    binding_sha256: str,
    hardware_file_sha256: str,
    hardware_self_sha256: str,
) -> None:
    exact = {
        "aggregate_only_public_receipt": True,
        "canary_binding_sha256": binding_sha256,
        "coordinate_gpu": 1,
        "coordinate_worker_count": 1,
        "current_hardware_binding_file_sha256": (
            hardware_file_sha256
        ),
        "current_hardware_binding_self_sha256": (
            hardware_self_sha256
        ),
        "current_hardware_binding_study_id": STUDY_ID,
        "current_hardware_capture_count": 1,
        "formal_action_count": 0,
        "formal_evaluator_count": 0,
        "formal_score_count": 0,
        "formal_source_access_count": 0,
        "minilm_constructor_canary_encode_call_count": 2,
        "minilm_formal_batch_encode_call_count": 1,
        "minilm_total_encode_call_count": 3,
        "online_or_API_evaluator_calls": 0,
        "private_vector_values_published": False,
        "query_count": 1,
        "retry_count": 0,
        "schema": COORDINATE_CANARY_SCHEMA,
        "status": "passed_source_free_coordinate_canary_once",
        "study_id": STUDY_ID,
        "synthetic_corpus_count": SYNTHETIC_CORPUS_COUNT,
        "synthetic_unique_passage_text_count": (
            SYNTHETIC_UNIQUE_PASSAGE_TEXT_COUNT
        ),
    }
    if any(value.get(key) != expected for key, expected in exact.items()):
        raise BioasqP1FormalRuntimeError(
            "coordinate canary receipt counters drifted"
        )
    for name in (
        "canary_attempt_self_sha256",
        "model_binding_sha256",
        "private_output_self_sha256",
        "private_score_bundle_sha256",
        "worker_receipt_sha256",
    ):
        required_sha256(value.get(name), f"coordinate canary {name}")


HardwareVerify = Callable[..., Mapping[str, object]]
ClosureVerify = Callable[..., Mapping[str, object]]


def verify_formal_preflight(
    config: FormalPreflightConfig,
    *,
    hardware_verify_callable: HardwareVerify = (
        runtime_binding.verify_current_study_hardware_binding
    ),
    closure_verify_callable: ClosureVerify = (
        runtime_binding.verify_p17_reused_closure_binding
    ),
) -> Mapping[str, object]:
    """Authenticate frozen infrastructure before formal source access."""

    if not isinstance(config, FormalPreflightConfig):
        raise BioasqP1FormalRuntimeError(
            "formal preflight config type drifted"
        )
    # Code and source-free evidence are checked before the source compiler can
    # be constructed by the outer runner.
    coordinate_backend = verify_coordinate_backend(
        config.coordinate_project_root
    )
    hippo_backend = verify_generic_hippo_backend(
        config.hippo_worker_project_root
    )
    coordinate_canary = _load_exact_receipt(
        config.coordinate_canary_receipt_path,
        expected_file_sha256=(
            config.coordinate_canary_receipt_file_sha256
        ),
        expected_self_sha256=(
            config.coordinate_canary_receipt_self_sha256
        ),
        field="BioASQ coordinate canary receipt",
    )
    _validate_coordinate_canary_receipt(
        coordinate_canary,
        binding_sha256=config.coordinate_canary_binding_sha256,
        hardware_file_sha256=(
            config.bioasq_hardware_binding_file_sha256
        ),
        hardware_self_sha256=(
            config.bioasq_hardware_binding_self_sha256
        ),
    )
    legacy_canary = _load_exact_receipt(
        config.legacy_hippo_canary_receipt_path,
        expected_file_sha256=LEGACY_HIPPO_CANARY_FILE_SHA256,
        expected_self_sha256=LEGACY_HIPPO_CANARY_SELF_SHA256,
        field="legacy official-HippoRAG canary receipt",
    )
    if (
        legacy_canary.get("schema")
        != "dstc9_p1_source_free_infrastructure_canary_receipt_v2"
        or legacy_canary.get("status")
        != "passed_source_free_two_lane_canary_once"
        or legacy_canary.get("study_id") != LEGACY_HIPPO_STUDY_ID
        or legacy_canary.get("formal_source_access_count") != 0
        or legacy_canary.get("online_or_API_evaluator_calls") != 0
        or legacy_canary.get("retry_count") != 0
        or legacy_canary.get("hipporag_build_count") != 1
        or legacy_canary.get("hipporag_retrieve_count") != 1
        or legacy_canary.get("current_hardware_binding_file_sha256")
        != LEGACY_HARDWARE_FILE_SHA256
        or legacy_canary.get("current_hardware_binding_self_sha256")
        != LEGACY_HARDWARE_SELF_SHA256
    ):
        raise BioasqP1FormalRuntimeError(
            "legacy official-HippoRAG canary semantics drifted"
        )
    legacy_hardware = _load_exact_receipt(
        config.legacy_hardware_binding_path,
        expected_file_sha256=LEGACY_HARDWARE_FILE_SHA256,
        expected_self_sha256=LEGACY_HARDWARE_SELF_SHA256,
        field="copied legacy hardware binding",
    )
    if (
        legacy_hardware.get("schema")
        != runtime_binding.CURRENT_HARDWARE_SCHEMA
        or legacy_hardware.get("study_id") != LEGACY_HIPPO_STUDY_ID
        or legacy_hardware.get("status")
        != runtime_binding.CURRENT_HARDWARE_STATUS
    ):
        raise BioasqP1FormalRuntimeError(
            "copied legacy hardware receipt identity drifted"
        )
    bioasq_hardware = _load_exact_receipt(
        config.bioasq_hardware_binding_path,
        expected_file_sha256=(
            config.bioasq_hardware_binding_file_sha256
        ),
        expected_self_sha256=(
            config.bioasq_hardware_binding_self_sha256
        ),
        field="BioASQ current-hardware binding",
    )
    if (
        bioasq_hardware.get("schema")
        != runtime_binding.CURRENT_HARDWARE_SCHEMA
        or bioasq_hardware.get("study_id") != STUDY_ID
        or bioasq_hardware.get("status")
        != runtime_binding.CURRENT_HARDWARE_STATUS
        or bioasq_hardware.get("hardware")
        != legacy_hardware.get("hardware")
    ):
        raise BioasqP1FormalRuntimeError(
            "BioASQ/legacy hardware identity differs"
        )
    try:
        live = hardware_verify_callable(
            path=config.bioasq_hardware_binding_path,
            worker_project_root=config.hippo_worker_project_root,
            expected_study_id=STUDY_ID,
        )
    except Exception as exc:
        raise BioasqP1FormalRuntimeError(
            "live 311linux hardware differs from the BioASQ binding"
        ) from exc
    if (
        not isinstance(live, Mapping)
        or live.get("receipt_file_sha256")
        != config.bioasq_hardware_binding_file_sha256
        or live.get("receipt_self_sha256")
        != config.bioasq_hardware_binding_self_sha256
        or live.get("hardware") != bioasq_hardware.get("hardware")
        or live.get("study_id") != STUDY_ID
    ):
        raise BioasqP1FormalRuntimeError(
            "live 311linux hardware verification drifted"
        )
    try:
        closure = closure_verify_callable(
            expected_study_id=STUDY_ID,
            worker_project_root=config.hippo_worker_project_root,
            current_hardware_binding_path=(
                config.bioasq_hardware_binding_path
            ),
            runtime_fingerprint_path=(
                config.hippo_runtime_fingerprint_path
            ),
            runtime_python=config.hippo_runtime_python,
            local_llm_model=config.hippo_local_llm_model,
            local_embedding_model=config.hippo_local_embedding_model,
        )
    except Exception as exc:
        raise BioasqP1FormalRuntimeError(
            "BioASQ official-HippoRAG closure verification failed"
        ) from exc
    if (
        not isinstance(closure, Mapping)
        or closure.get("schema") != runtime_binding.SCHEMA
        or closure.get("status")
        != (
            "verified_P17_reused_dependency_closure_with_"
            "separate_current_hardware_binding"
        )
        or closure.get("current_hardware_binding") != live
    ):
        raise BioasqP1FormalRuntimeError(
            "BioASQ official-HippoRAG closure binding drifted"
        )
    closure_self_sha256 = verify_self_hash(
        closure, "BioASQ official-HippoRAG closure"
    )
    receipt = with_self_hash(
        {
            "aggregate_only_public_receipt": True,
            "coordinate_canary_receipt_self_sha256": (
                config.coordinate_canary_receipt_self_sha256
            ),
            "coordinate_model_binding_sha256": (
                coordinate_canary["model_binding_sha256"]
            ),
            "coordinate_source_free_canary_reused": True,
            "bioasq_hardware_binding_self_sha256": (
                config.bioasq_hardware_binding_self_sha256
            ),
            "bioasq_hardware_matches_legacy_canary_hardware": True,
            "execution_binding_sha256": config.execution_binding_sha256,
            "formal_source_access_count": 0,
            "coordinate_backend_file_sha256": dict(coordinate_backend),
            "generic_hippo_backend_file_sha256": dict(hippo_backend),
            "generic_backend_reused_despite_legacy_benchmark_label": True,
            "hipporag_closure_self_sha256": closure_self_sha256,
            "legacy_hardware_binding_self_sha256": (
                LEGACY_HARDWARE_SELF_SHA256
            ),
            "legacy_hippo_canary_receipt_self_sha256": (
                LEGACY_HIPPO_CANARY_SELF_SHA256
            ),
            "legacy_hippo_canary_rerun_count": 0,
            "live_hardware_matches_qualified_binding": True,
            "online_or_API_evaluator_calls": 0,
            "retry_count": 0,
            "schema": FORMAL_PREFLIGHT_RECEIPT_SCHEMA,
            "status": "passed_offline_preformal_infrastructure_binding",
            "study_id": STUDY_ID,
        }
    )
    return receipt


__all__ = [
    "CANARY_ATTEMPT_FILENAME",
    "CANARY_RECEIPT_FILENAME",
    "COORDINATE_CANARY_CONFIG_SCHEMA",
    "COORDINATE_CANARY_FAILURE_SCHEMA",
    "COORDINATE_CANARY_SCHEMA",
    "COORDINATE_BACKEND_SHA256",
    "CURRENT_HARDWARE_FILENAME",
    "FORMAL_PREFLIGHT_CONFIG_SCHEMA",
    "FORMAL_PREFLIGHT_RECEIPT_SCHEMA",
    "GENERIC_HIPPO_BACKEND_SHA256",
    "LEGACY_HARDWARE_FILE_SHA256",
    "LEGACY_HARDWARE_SELF_SHA256",
    "LEGACY_HIPPO_CANARY_FILE_SHA256",
    "LEGACY_HIPPO_CANARY_SELF_SHA256",
    "LEGACY_HIPPO_STUDY_ID",
    "STUDY_ID",
    "VERSION",
    "BioasqP1FormalRuntimeError",
    "CoordinateCanaryConfig",
    "FormalPreflightConfig",
    "assert_no_symlink_components",
    "canonical_bytes",
    "exclusive_json",
    "fresh_private_directory",
    "load_runtime_config",
    "required_sha256",
    "run_source_free_coordinate_canary_once",
    "stable_hash",
    "verify_formal_preflight",
    "verify_coordinate_backend",
    "verify_generic_hippo_backend",
    "verify_self_hash",
    "with_self_hash",
]
