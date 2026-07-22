"""Public, label-free production canary for the frozen TAT-QA P19 runtime.

The fixture in this module is synthetic and fixed in source.  It contains no
TAT-QA identifier, answer, family, mapping, or gold evidence.  A qualifying
run crosses the same byte-oriented Qwen boundary, strict typed-plan parser,
exact MiniLM encoder, semantic compiler, and P0/P1 action algebra used by the
formal study.  The complete path is run twice and must be byte/exactly
reproducible; no retry or totalizer around the runtime is provided.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from assumption_agent.benchmarks import tatqa_p19_acquisition_v1 as acquisition
from assumption_agent.benchmarks import tatqa_p19_label_free_runtime_v1 as features
from assumption_agent.benchmarks import tatqa_p19_typed_evaluator_core_v1 as core
from replication_runtime.tatqa_p19_v1 import hipporag_contract
from replication_runtime.tatqa_p19_v1 import typed_plan_contract


VERSION = "tatqa_p19_public_canary_v1"
SCHEMA = "tatqa_p19_public_synthetic_production_canary_v1"
RUNTIME_FINGERPRINT_SCHEMA = "tatqa_p19_composite_runtime_fingerprint_v1"
RUNTIME_SUBFINGERPRINT_SCHEMAS = {
    "typed_plan_minilm_runtime_python": (
        "tatqa_p19_typed_minilm_runtime_python_subfingerprint_v1"
    ),
    "hipporag_runtime_python": (
        "tatqa_p19_hipporag_runtime_python_subfingerprint_v1"
    ),
}
RUNTIME_SUBFINGERPRINT_HASHES_FIELD = (
    "runtime_python_subfingerprint_self_sha256s"
)
REPEAT_COUNT = 2
FILESYSTEM_ISOLATION = (
    "systemd_InaccessiblePaths_official_source_and_acquisition_v1"
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class TatqaP19PublicCanaryError(RuntimeError):
    """The public production path or its immutable receipt drifted."""


class TypedPlanByteRunner(Protocol):
    def __call__(self, block: str, canonical_input: bytes) -> bytes: ...


class MiniLMEncoder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class HippoByteRunner(Protocol):
    def __call__(
        self, block: str, item_commitment_sha256: str, canonical_input: bytes
    ) -> bytes: ...


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
        raise TatqaP19PublicCanaryError("value is not canonical JSON") from exc
    return raw + (b"\n" if newline else b"")


def _stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value, newline=False)).hexdigest()


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise TatqaP19PublicCanaryError(f"{field} is not a lowercase SHA-256")
    return value


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _strict_json_object(raw: bytes, *, field: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"nonfinite constant {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise TatqaP19PublicCanaryError(f"{field} is not strict JSON") from exc
    if not isinstance(value, dict) or _canonical_bytes(value) != raw:
        raise TatqaP19PublicCanaryError(f"{field} is not canonical JSON")
    return value


def _verify_self_hash(value: Mapping[str, Any], *, field: str = "self_sha256") -> str:
    claimed = _require_sha256(value.get(field), field)
    body = dict(value)
    del body[field]
    if _stable_hash(body) != claimed:
        raise TatqaP19PublicCanaryError(f"{field} binding drifted")
    return claimed


def _runtime_subfingerprint_self_hashes(
    fingerprint: Mapping[str, Any],
) -> dict[str, str]:
    inventory = fingerprint.get("runtime_inventory")
    if not isinstance(inventory, Mapping):
        raise TatqaP19PublicCanaryError(
            "runtime fingerprint inventory is unavailable"
        )
    nested = inventory.get("runtime_python_subfingerprints")
    if not isinstance(nested, Mapping) or set(nested) != set(
        RUNTIME_SUBFINGERPRINT_SCHEMAS
    ):
        raise TatqaP19PublicCanaryError(
            "runtime Python subfingerprint registry drifted"
        )
    result: dict[str, str] = {}
    for key, schema in RUNTIME_SUBFINGERPRINT_SCHEMAS.items():
        value = nested.get(key)
        if not isinstance(value, Mapping) or value.get("schema") != schema:
            raise TatqaP19PublicCanaryError(
                f"{key} subfingerprint schema drifted"
            )
        result[key] = _verify_self_hash(
            value, field="self_sha256"
        )
    return result


def _read_runtime_fingerprint(
    path: Path,
) -> tuple[dict[str, Any], str, str, dict[str, str]]:
    try:
        if path.is_symlink() or not path.is_file():
            raise OSError("not a regular file")
        raw = path.read_bytes()
    except OSError as exc:
        raise TatqaP19PublicCanaryError("runtime fingerprint is unavailable") from exc
    value = _strict_json_object(raw, field="runtime fingerprint")
    self_sha = _verify_self_hash(value)
    if (
        value.get("schema") != RUNTIME_FINGERPRINT_SCHEMA
        or value.get("status") != "verified_before_formal_source_open"
        or value.get("study_design_self_sha256") != acquisition.DESIGN_SELF_SHA256
        or value.get("formal_source_opened") is not False
        or value.get("external_network_calls") != 0
        or value.get("api_or_online_evaluator_calls") != 0
    ):
        raise TatqaP19PublicCanaryError("runtime fingerprint is not canary-safe")
    subfingerprint_hashes = _runtime_subfingerprint_self_hashes(value)
    return (
        value,
        self_sha,
        hashlib.sha256(raw).hexdigest(),
        subfingerprint_hashes,
    )


def public_fixture_payload() -> dict[str, object]:
    """Return the complete fixed public fixture without a source identity."""

    return {
        "fixture": "PUBLIC_SYNTHETIC_NORTHWIND_RENEWABLE_SHARE_V1",
        "question": (
            "By how many percentage points did Northwind's renewable electricity "
            "share increase from 2022 to 2024, and which note explains the change?"
        ),
        "units": [
            {
                "unit_id": "T:0",
                "text": "TABLE_HEADER|year|renewable electricity share|capacity note",
            },
            {
                "unit_id": "T:1",
                "text": "TABLE_ROW_1|2022|40 percent|baseline portfolio",
            },
            {
                "unit_id": "T:2",
                "text": "TABLE_ROW_2|2023|48 percent|solar build-out",
            },
            {
                "unit_id": "T:3",
                "text": "TABLE_ROW_3|2024|55 percent|solar build-out completed",
            },
            {
                "unit_id": "P:1",
                "text": "Northwind reports renewable electricity share each year.",
            },
            {
                "unit_id": "P:2",
                "text": "The increase was explained by newly commissioned solar capacity.",
            },
            {
                "unit_id": "P:3",
                "text": "A separate note discusses debt maturity and currency exposure.",
            },
            {
                "unit_id": "P:4",
                "text": "Percentages use electricity consumed during the calendar year.",
            },
        ],
    }


def public_runtime_item() -> features.LabelFreeRuntimeItem:
    payload = public_fixture_payload()
    item_id = _stable_hash(payload)
    return features.LabelFreeRuntimeItem(
        item_id=item_id,
        question=str(payload["question"]),
        units=tuple(
            features.RuntimeUnit(str(row["unit_id"]), str(row["text"]))
            for row in payload["units"]  # type: ignore[index,union-attr]
        ),
    )


def _matrix_sha256(value: object) -> str:
    matrix = np.asarray(value)
    body = {
        "dtype": matrix.dtype.str,
        "shape": list(matrix.shape),
        "bytes_sha256": hashlib.sha256(
            np.ascontiguousarray(matrix).tobytes(order="C")
        ).hexdigest(),
    }
    return _stable_hash(body)


def _json_copy(value: object) -> object:
    try:
        return json.loads(_canonical_bytes(value, newline=False).decode("ascii"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:  # pragma: no cover
        raise TatqaP19PublicCanaryError("capability receipt is not JSON-safe") from exc


def _receipt_snapshot(value: object, *, role: str) -> object | None:
    if role == "typed_plan":
        observed = getattr(value, "receipts", None)
        if not isinstance(observed, Mapping) or not observed:
            return None
        return {
            "schema": "tatqa_p19_typed_plan_capability_receipt_snapshot_v1",
            "capability_class": "SystemdTypedPlanBatchRunner",
            "receipts": [
                _json_copy(observed[key]) for key in sorted(observed)
            ],
        }
    if role == "minilm":
        runtime = getattr(value, "runtime_receipt", None)
        startup = getattr(value, "canary_receipt", None)
        if not isinstance(runtime, Mapping) or not isinstance(startup, Mapping):
            return None
        sanitized_runtime = {
            key: child
            for key, child in runtime.items()
            if key not in {"asset_manifest_path", "model_root"}
        }
        return {
            "schema": "tatqa_p19_minilm_capability_receipt_snapshot_v1",
            "execution": {
                "capability_class": "BoundMiniLMEncoder",
                "device": "cpu",
                "dtype": "float32",
                "in_process": True,
                "torch_threads": 1,
            },
            "omitted_absolute_path_fields": ["asset_manifest_path", "model_root"],
            "runtime_receipt": _json_copy(sanitized_runtime),
            "startup_canary_receipt": _json_copy(startup),
        }
    if role == "hipporag":
        observed = getattr(value, "receipts", None)
        if not isinstance(observed, list) or not observed:
            return None
        return {
            "schema": "tatqa_p19_hipporag_capability_receipt_snapshot_v1",
            "capability_class": "SystemdHippoByteRunner",
            "receipts": [_json_copy(row) for row in observed],
        }
    raise TatqaP19PublicCanaryError("capability receipt role drifted")


def _bound_receipt_hash(
    *,
    explicit: object | None,
    capability: object,
    fallback: Mapping[str, object],
    role: str,
) -> tuple[str, str, object]:
    if explicit is not None:
        receipt, source = explicit, "explicit_formal_receipt"
    else:
        observed = _receipt_snapshot(capability, role=role)
        if observed is not None:
            receipt, source = observed, "capability_receipt_snapshot"
        else:
            receipt, source = dict(fallback), "canary_transport_fallback"
    copied = _json_copy(receipt)
    return _stable_hash(copied), source, copied


@dataclass(frozen=True)
class _Repeat:
    output_raw_sha256: str
    embedding_sha256: str
    tensor_sha256: str
    p0_action_sha256: str
    p1_action_sha256: str
    p0_behavior_sha256: str
    p1_behavior_sha256: str
    p0_units: tuple[str, ...]
    p1_units: tuple[str, ...]
    generation_valid: bool


def _run_repeat(
    *,
    index: int,
    item: features.LabelFreeRuntimeItem,
    canonical_input: bytes,
    typed_plan_runner: TypedPlanByteRunner,
    encoder: MiniLMEncoder,
) -> _Repeat:
    try:
        raw_output = typed_plan_runner(
            f"PUBLIC_CANARY_REPEAT_{index + 1}", canonical_input
        )
    except Exception as exc:
        raise TatqaP19PublicCanaryError("typed-plan byte runner failed") from exc
    if not isinstance(raw_output, bytes):
        raise TatqaP19PublicCanaryError("typed-plan runner did not return bytes")
    try:
        output = typed_plan_contract.parse_output(raw_output)
    except Exception as exc:
        raise TatqaP19PublicCanaryError("strict typed-plan output parse failed") from exc
    if len(output["items"]) != 1:
        raise TatqaP19PublicCanaryError("typed-plan canary cardinality drifted")
    row = output["items"][0]
    if row["ordinal"] != 0 or row["generation_valid"] is not True:
        raise TatqaP19PublicCanaryError("Qwen did not emit one valid typed plan")
    plan = core.validate_typed_plan(row["plan"])
    texts = features.embedding_texts(item, plan)
    try:
        matrix = encoder.encode(texts)
    except Exception as exc:
        raise TatqaP19PublicCanaryError("exact MiniLM canary encoding failed") from exc
    embedding_sha = _matrix_sha256(matrix)
    try:
        compiled = features.compile_from_embeddings(item, plan, matrix)
        p0, p1 = core.build_action_pair(
            compiled.plan,
            compiled.units,
            redundancy_features=compiled.redundancy_features,
        )
    except Exception as exc:
        raise TatqaP19PublicCanaryError("public typed-action canary failed") from exc
    if p1.selected_unit_ids[:3] != p0.selected_unit_ids[:3]:
        raise TatqaP19PublicCanaryError("P1 did not retain P0 top three")
    if len(set(p1.selected_unit_ids) - set(p0.selected_unit_ids)) < 1:
        raise TatqaP19PublicCanaryError("P1 introduced no typed residual unit")
    return _Repeat(
        output_raw_sha256=hashlib.sha256(raw_output).hexdigest(),
        embedding_sha256=embedding_sha,
        tensor_sha256=compiled.tensor_sha256,
        p0_action_sha256=p0.action_sha256,
        p1_action_sha256=p1.action_sha256,
        p0_behavior_sha256=p0.behavior_sha256,
        p1_behavior_sha256=p1.behavior_sha256,
        p0_units=p0.selected_unit_ids,
        p1_units=p1.selected_unit_ids,
        generation_valid=True,
    )


def _run_optional_hippo(
    *, item: features.LabelFreeRuntimeItem, runner: HippoByteRunner
) -> dict[str, object]:
    units = [
        {"ordinal": ordinal, "text": row.text, "unit_id": row.unit_id}
        for ordinal, row in enumerate(item.units)
    ]
    payload = hipporag_contract.input_payload(query=item.question, units=units)
    raw_input = hipporag_contract.canonical_json_bytes(payload)
    try:
        raw_output = runner("PUBLIC_CANARY_HIPPO", item.item_id, raw_input)
    except Exception as exc:
        raise TatqaP19PublicCanaryError("optional Hippo canary failed") from exc
    if not isinstance(raw_output, bytes):
        raise TatqaP19PublicCanaryError("Hippo canary did not return bytes")
    try:
        output = hipporag_contract.parse_output(raw_output)
    except Exception as exc:
        raise TatqaP19PublicCanaryError("Hippo canary output parse failed") from exc
    if (
        output["input_sha256"] != payload["input_sha256"]
        or output["unit_count"] != len(item.units)
        or not set(output["top_unit_ids"]).issubset(
            {row.unit_id for row in item.units}
        )
    ):
        raise TatqaP19PublicCanaryError("Hippo canary corpus binding drifted")
    return {
        "ran": True,
        "input_file_sha256": hashlib.sha256(raw_input).hexdigest(),
        "input_semantic_sha256": payload["input_sha256"],
        "output_file_sha256": hashlib.sha256(raw_output).hexdigest(),
        "top5_behavior_sha256": core.canonical_behavior_hash(
            output["top_unit_ids"]
        ),
    }


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> str:
    raw = _canonical_bytes(value)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except OSError as exc:
        raise TatqaP19PublicCanaryError("canary receipt path is already consumed") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    if path.read_bytes() != raw:
        raise TatqaP19PublicCanaryError("canary receipt reopen verification failed")
    return hashlib.sha256(raw).hexdigest()


def run_public_production_canary(
    *,
    runtime_fingerprint_path: str | Path,
    output_path: str | Path,
    typed_plan_runner: TypedPlanByteRunner,
    encoder: MiniLMEncoder,
    hippo_runner: HippoByteRunner | None = None,
    typed_plan_worker_receipt: object | None = None,
    minilm_worker_receipt: object | None = None,
    hippo_worker_receipt: object | None = None,
) -> dict[str, Any]:
    """Run the complete public production path exactly twice and seal it."""

    (
        _fingerprint,
        fingerprint_self,
        fingerprint_file,
        subfingerprint_hashes,
    ) = _read_runtime_fingerprint(Path(runtime_fingerprint_path))
    item = public_runtime_item()
    projected = typed_plan_contract.project_item(item, 0)
    canonical_input = typed_plan_contract.canonical_json_bytes(
        typed_plan_contract.input_payload((projected,))
    )
    repeats = tuple(
        _run_repeat(
            index=index,
            item=item,
            canonical_input=canonical_input,
            typed_plan_runner=typed_plan_runner,
            encoder=encoder,
        )
        for index in range(REPEAT_COUNT)
    )
    if repeats[0] != repeats[1]:
        raise TatqaP19PublicCanaryError("public production path is not exact on repeat")
    result = repeats[0]
    if result.p0_behavior_sha256 == result.p1_behavior_sha256:
        raise TatqaP19PublicCanaryError("public P0 and P1 behaviors are identical")

    typed_receipt_sha, typed_receipt_source, typed_receipt_snapshot = _bound_receipt_hash(
        explicit=typed_plan_worker_receipt,
        capability=typed_plan_runner,
        fallback={
            "input_sha256": hashlib.sha256(canonical_input).hexdigest(),
            "output_sha256": result.output_raw_sha256,
            "repeat_count": REPEAT_COUNT,
            "role": "typed_plan_byte_runner",
        },
        role="typed_plan",
    )
    minilm_receipt_sha, minilm_receipt_source, minilm_receipt_snapshot = _bound_receipt_hash(
        explicit=minilm_worker_receipt,
        capability=encoder,
        fallback={
            "embedding_sha256": result.embedding_sha256,
            "repeat_count": REPEAT_COUNT,
            "role": "exact_minilm_encoder",
        },
        role="minilm",
    )
    if hippo_runner is None:
        hippo = {
            "ran": False,
            "input_file_sha256": None,
            "input_semantic_sha256": None,
            "output_file_sha256": None,
            "top5_behavior_sha256": None,
        }
        hippo_receipt_sha = _stable_hash(
            {"role": "official_hippo_canary", "status": "not_run_optional"}
        )
        hippo_receipt_source = "explicit_optional_absence_receipt"
        hippo_receipt_snapshot: object = {
            "role": "official_hippo_canary",
            "status": "not_run_optional",
        }
        if hippo_worker_receipt is not None:
            raise TatqaP19PublicCanaryError(
                "Hippo receipt was supplied without a Hippo canary"
            )
    else:
        hippo = _run_optional_hippo(item=item, runner=hippo_runner)
        hippo_receipt_sha, hippo_receipt_source, hippo_receipt_snapshot = _bound_receipt_hash(
            explicit=hippo_worker_receipt,
            capability=hippo_runner,
            fallback={**hippo, "role": "official_hippo_byte_runner"},
            role="hipporag",
        )

    outside = len(set(result.p1_units) - set(result.p0_units))
    body: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "qualified_before_formal_source_open",
        "qualified": True,
        "study_design_self_sha256": acquisition.DESIGN_SELF_SHA256,
        "runtime_fingerprint_self_sha256": fingerprint_self,
        "runtime_fingerprint_file_sha256": fingerprint_file,
        RUNTIME_SUBFINGERPRINT_HASHES_FIELD: subfingerprint_hashes,
        "public_synthetic_fixture": public_fixture_payload(),
        "public_synthetic_fixture_sha256": item.item_id,
        "repeat_count": REPEAT_COUNT,
        "typed_plan_input_file_sha256": hashlib.sha256(canonical_input).hexdigest(),
        "typed_plan_output_file_sha256": result.output_raw_sha256,
        "typed_plan_output_exact_repeat": True,
        "embedding_matrix_sha256": result.embedding_sha256,
        "embedding_matrix_exact_repeat": True,
        "compiled_tensor_sha256": result.tensor_sha256,
        "compiled_tensor_exact_repeat": True,
        "public_synthetic_p0_action_sha256": result.p0_action_sha256,
        "public_synthetic_p1_action_sha256": result.p1_action_sha256,
        "public_synthetic_p0_behavior_sha256": result.p0_behavior_sha256,
        "public_synthetic_p1_behavior_sha256": result.p1_behavior_sha256,
        "public_synthetic_p0_top5": list(result.p0_units),
        "public_synthetic_p1_top5": list(result.p1_units),
        "public_synthetic_distinct_rankings": result.p0_units != result.p1_units,
        "P1_retains_ordered_P0_top3": result.p1_units[:3] == result.p0_units[:3],
        "P1_outside_P0_unit_count": outside,
        "typed_plan_worker_receipt_sha256": typed_receipt_sha,
        "typed_plan_worker_receipt_source": typed_receipt_source,
        "typed_plan_worker_receipt_snapshot": typed_receipt_snapshot,
        "minilm_worker_receipt_sha256": minilm_receipt_sha,
        "minilm_worker_receipt_source": minilm_receipt_source,
        "minilm_worker_receipt_snapshot": minilm_receipt_snapshot,
        "hippo_canary_ran": hippo["ran"],
        "hippo_canary_input_file_sha256": hippo["input_file_sha256"],
        "hippo_canary_input_semantic_sha256": hippo["input_semantic_sha256"],
        "hippo_canary_output_file_sha256": hippo["output_file_sha256"],
        "hippo_canary_top5_behavior_sha256": hippo["top5_behavior_sha256"],
        "hippo_worker_receipt_sha256": hippo_receipt_sha,
        "hippo_worker_receipt_source": hippo_receipt_source,
        "hippo_worker_receipt_snapshot": hippo_receipt_snapshot,
        "filesystem_isolation": FILESYSTEM_ISOLATION,
        "formal_source_opened": False,
        "source_identifiers_answers_families_mappings_or_labels_present": False,
        "external_network_calls": 0,
        "api_or_online_evaluator_calls": 0,
        "retry_replay_resample_provider_switch": 0,
    }
    receipt = {**body, "self_sha256": _stable_hash(body)}
    _write_exclusive(Path(output_path), receipt)
    return receipt


__all__ = [
    "HippoByteRunner",
    "FILESYSTEM_ISOLATION",
    "MiniLMEncoder",
    "REPEAT_COUNT",
    "SCHEMA",
    "TatqaP19PublicCanaryError",
    "TypedPlanByteRunner",
    "VERSION",
    "public_fixture_payload",
    "public_runtime_item",
    "run_public_production_canary",
]
