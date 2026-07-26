"""One-shot formal controller for MAUD extraction P1.

The controller is intentionally split into three trust domains:

* acquisition is always a separate ``python -S -B -m`` subprocess;
* label-free action formation reads only acquisition action views and local
  model/official-worker outputs; and
* a late scorer may open the matching sealed gold pack only after the complete
  action archive has been durably written.

This module never imports the MAUD source or acquisition implementation.  The
acquisition module name is an opaque subprocess target.  F_search has no
gold-opening function.  TEST acquisition is launched only after a canonical,
self-hashed A_hold receipt says both ``promoted`` and
``M_search_authorized`` are exactly true.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any, Callable, Mapping, Protocol, Sequence

from replication_runtime.maud_extraction_p1_official_v1 import (
    worker as official_worker,
)

from . import maud_extraction_p1_coordinate_worker_v1 as coordinate_worker
from . import maud_extraction_p1_runtime_v1 as runtime
from . import maud_extraction_p1_typed_core_v1 as core


VERSION = "maud_extraction_p1_formal_controller_v1"
STUDY_ID = "MAUD_EXTRACTION_P1_TYPED_EVALUATOR_V1"
STUDY_DESIGN_SELF_SHA256 = (
    "bcbcac216e3345b50a9cb3841e366c14e1971a9bf84cd6fd7814ade3ac9eb832"
)
SOURCE_CUSTODY_SELF_SHA256 = (
    "27a60c19dd1c0d3b0632f073ed7fe50286bb88b8f89cd30747eff3717c41f1fe"
)
PRE_SOURCE_CLARIFICATION_SELF_SHA256 = (
    "e8eb673249e9f00f267d93b65491a09693da2ed02135b2d87d978acaf9fae591"
)

ACQUISITION_MODULE = (
    "assumption_agent.benchmarks.maud_extraction_p1_acquisition_v1"
)
CONTROLLER_MODULE = (
    "assumption_agent.benchmarks.maud_extraction_p1_formal_controller_v1"
)
ACTION_VIEW_SCHEMA = "maud_extraction_p1_source_v1_label_free_action_view"
GOLD_PACK_SCHEMA = "maud_extraction_p1_source_v1_private_gold_pack"

BLOCK_A_FORM = "A_form"
BLOCK_F_SEARCH = "F_search"
BLOCK_A_HOLD = "A_hold"
BLOCK_M_SEARCH = "M_search"
BLOCKS = (BLOCK_A_FORM, BLOCK_F_SEARCH, BLOCK_A_HOLD, BLOCK_M_SEARCH)
GOLD_BLOCKS = (BLOCK_A_FORM, BLOCK_A_HOLD, BLOCK_M_SEARCH)
OFFICIAL_BLOCKS = (BLOCK_A_HOLD,)

ROLE_AGENT_E0 = "Agent_E0"
ROLE_AGENT_E1 = "Agent_E1"
ROLE_RAW = "RAW"
ROLE_HIPPORAG = "official_HippoRAG"

MODEL_SCHEMA = f"{VERSION}_e1_model_private_v1"
ACTION_ARCHIVE_SCHEMA = f"{VERSION}_action_archive_private_v1"
ACTION_RECEIPT_SCHEMA = f"{VERSION}_action_receipt_v1"
SCORE_ARCHIVE_SCHEMA = f"{VERSION}_score_archive_private_v1"
SCORE_RECEIPT_SCHEMA = f"{VERSION}_score_receipt_v1"
PROMOTION_SCHEMA = f"{VERSION}_A_hold_promotion_receipt_v1"
FULL_CANARY_SCHEMA = f"{VERSION}_full_source_free_canary_v1"
TERMINAL_SCHEMA = f"{VERSION}_safe_terminal_v1"
CONFIG_SCHEMA = f"{VERSION}_execution_config_v1"

COORDINATE_TIMEOUT_SECONDS = 7_200
ACQUISITION_TIMEOUT_SECONDS = 7_200
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class MaudExtractionP1FormalControllerError(RuntimeError):
    """A frozen stage, binding, ordering rule, or privacy boundary drifted."""


class CoordinateBatchLauncher(Protocol):
    def __call__(
        self,
        *,
        private_input: Mapping[str, object],
        stage_root: Path,
        runtime_paths: runtime.RuntimePaths,
    ) -> "CoordinateBatchResult": ...


class HippoBatchLauncher(Protocol):
    def __call__(
        self,
        jobs: Sequence[runtime.ContractLaunchJob],
        *,
        runtime_paths: runtime.RuntimePaths,
    ) -> Sequence[runtime.WorkerRun]: ...


def canonical_json_bytes(
    value: object, *, newline: bool = True
) -> bytes:
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise MaudExtractionP1FormalControllerError(
            "controller value is not canonical JSON"
        ) from exc
    return raw + (b"\n" if newline else b"")


def semantic_sha256(value: object) -> str:
    return hashlib.sha256(
        canonical_json_bytes(value, newline=False)
    ).hexdigest()


def self_hashed(body: Mapping[str, object]) -> dict[str, object]:
    if "self_sha256" in body:
        raise MaudExtractionP1FormalControllerError(
            "self-hashed body already contains its digest"
        )
    result = dict(body)
    result["self_sha256"] = semantic_sha256(result)
    return result


def verify_self_hash(value: Mapping[str, object]) -> str:
    body = dict(value)
    declared = body.pop("self_sha256", None)
    if (
        not isinstance(declared, str)
        or _HEX64.fullmatch(declared) is None
        or semantic_sha256(body) != declared
    ):
        raise MaudExtractionP1FormalControllerError(
            "canonical self-hash drifted"
        )
    return declared


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _absolute_path(value: object, field: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise MaudExtractionP1FormalControllerError(
            f"{field} path is invalid"
        )
    raw = os.fspath(value)
    if not raw or "\x00" in raw or not os.path.isabs(raw):
        raise MaudExtractionP1FormalControllerError(
            f"{field} path must be absolute"
        )
    return Path(os.path.abspath(raw))


def _ensure_private_directory(path: Path, *, fresh: bool) -> None:
    if fresh:
        if os.path.lexists(path):
            raise MaudExtractionP1FormalControllerError(
                "one-shot private directory is already consumed"
            )
        if path.parent.is_symlink() or not path.parent.is_dir():
            raise MaudExtractionP1FormalControllerError(
                "private directory parent is unsafe"
            )
        os.mkdir(path, 0o700)
    if path.is_symlink() or not path.is_dir():
        raise MaudExtractionP1FormalControllerError(
            "private directory is unavailable"
        )
    if stat.S_IMODE(path.stat().st_mode) != 0o700:
        raise MaudExtractionP1FormalControllerError(
            "private directory mode drifted"
        )


def _write_private_bytes_once(path: Path, raw: bytes) -> dict[str, object]:
    if not isinstance(raw, bytes):
        raise MaudExtractionP1FormalControllerError(
            "private payload must be bytes"
        )
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(descriptor, "wb") as handle:
        os.fchmod(handle.fileno(), 0o600)
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
        metadata = os.fstat(handle.fileno())
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise MaudExtractionP1FormalControllerError(
                "private file mode drifted"
            )
    return {
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "semantic_sha256": hashlib.sha256(raw.rstrip(b"\n")).hexdigest(),
        "size_bytes": len(raw),
        "mode_octal": "0600",
    }


def write_private_json_once(
    path: Path, value: Mapping[str, object]
) -> dict[str, object]:
    return _write_private_bytes_once(path, canonical_json_bytes(value))


def _private_file_metadata(path: Path) -> os.stat_result:
    if path.is_symlink() or not path.is_file():
        raise MaudExtractionP1FormalControllerError(
            "private file is unavailable"
        )
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise MaudExtractionP1FormalControllerError(
            "private file mode or type drifted"
        )
    return metadata


def read_canonical_private(path: Path) -> dict[str, object]:
    _private_file_metadata(path)
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MaudExtractionP1FormalControllerError(
            "private file is invalid JSON"
        ) from exc
    if (
        not isinstance(value, Mapping)
        or raw
        not in {
            canonical_json_bytes(value, newline=True),
            canonical_json_bytes(value, newline=False),
        }
    ):
        raise MaudExtractionP1FormalControllerError(
            "private file is not canonical JSON"
        )
    return dict(value)


def _verify_file_binding(
    path: Path,
    binding: Mapping[str, object],
    *,
    semantic_field: str = "semantic_sha256",
    embedded_self_hash_field: str | None = None,
) -> None:
    if (
        not isinstance(binding, Mapping)
        or binding.get("mode_octal") != "0600"
        or binding.get("file_sha256") != _sha256_file(path)
    ):
        raise MaudExtractionP1FormalControllerError(
            "private archive file binding drifted"
        )
    value = read_canonical_private(path)
    semantic = binding.get(semantic_field)
    if embedded_self_hash_field is None:
        observed_semantic = semantic_sha256(value)
    else:
        body = dict(value)
        embedded = body.pop(embedded_self_hash_field, None)
        if embedded != semantic:
            raise MaudExtractionP1FormalControllerError(
                "private archive embedded hash drifted"
            )
        observed_semantic = semantic_sha256(body)
    if (
        semantic is not None
        and (
            not isinstance(semantic, str)
            or _HEX64.fullmatch(semantic) is None
            or observed_semantic != semantic
        )
    ):
        raise MaudExtractionP1FormalControllerError(
            "private archive semantic binding drifted"
        )


def _fraction_payload(value: Fraction) -> dict[str, int]:
    if not isinstance(value, Fraction):
        raise MaudExtractionP1FormalControllerError(
            "aggregate is not exact"
        )
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
    }


def _fraction_from_payload(value: object) -> Fraction:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"denominator", "numerator"}
        or type(value.get("numerator")) is not int
        or type(value.get("denominator")) is not int
        or int(value["denominator"]) <= 0
    ):
        raise MaudExtractionP1FormalControllerError(
            "exact aggregate payload drifted"
        )
    return Fraction(int(value["numerator"]), int(value["denominator"]))


@dataclass(frozen=True)
class ArchiveBinding:
    file_sha256: str
    semantic_sha256: str
    size_bytes: int | None = None
    mode_octal: str = "0600"

    @classmethod
    def from_mapping(cls, value: object) -> "ArchiveBinding":
        if not isinstance(value, Mapping):
            raise MaudExtractionP1FormalControllerError(
                "archive binding is not an object"
            )
        file_sha256 = value.get("file_sha256")
        semantic = value.get("semantic_sha256")
        size = value.get("size_bytes")
        if (
            not isinstance(file_sha256, str)
            or _HEX64.fullmatch(file_sha256) is None
            or not isinstance(semantic, str)
            or _HEX64.fullmatch(semantic) is None
            or value.get("mode_octal") != "0600"
            or (
                size is not None
                and (
                    type(size) is not int
                    or int(size) < 0
                )
            )
        ):
            raise MaudExtractionP1FormalControllerError(
                "archive binding drifted"
            )
        return cls(
            file_sha256=file_sha256,
            semantic_sha256=semantic,
            size_bytes=None if size is None else int(size),
        )

    def as_mapping(self) -> dict[str, object]:
        result: dict[str, object] = {
            "file_sha256": self.file_sha256,
            "semantic_sha256": self.semantic_sha256,
            "mode_octal": self.mode_octal,
        }
        if self.size_bytes is not None:
            result["size_bytes"] = self.size_bytes
        return result


@dataclass(frozen=True)
class LabelFreeItem:
    work_id: str
    question: str
    deal_point_type: str
    family: str


@dataclass(frozen=True)
class LabelFreeContract:
    contract_work_id: str
    context_sha256: str
    context: str
    passages: tuple[core.Passage, ...]
    serialized_passages: tuple[str, ...]
    items: tuple[LabelFreeItem, ...]


@dataclass(frozen=True)
class CoordinateBatchResult:
    minilm_output: Mapping[str, object]
    cross_encoder_output: Mapping[str, object]


@dataclass(frozen=True)
class StageResult:
    block: str
    archive_path: Path
    archive_file_sha256: str
    archive_semantic_sha256: str
    archive_self_sha256: str
    receipt_path: Path
    receipt_self_sha256: str
    contract_count: int
    item_count: int


@dataclass(frozen=True)
class ModelResult:
    model_path: Path
    model_file_sha256: str
    model_self_sha256: str
    model_sha256: str
    receipt_path: Path


@dataclass(frozen=True)
class ScoreResult:
    block: str
    score_archive_path: Path
    score_archive_file_sha256: str
    receipt_path: Path
    receipt: Mapping[str, object]


def _validate_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise MaudExtractionP1FormalControllerError(
            f"{field} is not SHA-256"
        )
    return value


def _validate_acquisition_receipt(
    value: Mapping[str, object], *, phase: str
) -> str:
    digest = verify_self_hash(value)
    expected_schema = (
        "maud_extraction_p1_acquisition_v1_initial_receipt_v1"
        if phase == "initial"
        else "maud_extraction_p1_acquisition_v1_test_receipt_v1"
    )
    expected_status = (
        "trusted_initial_parse_complete_and_process_must_exit"
        if phase == "initial"
        else (
            "promotion_authorized_TEST_parse_complete_and_process_must_exit"
        )
    )
    if (
        value.get("schema") != expected_schema
        or value.get("study_id") != STUDY_ID
        or value.get("status") != expected_status
        or not isinstance(value.get("private_archives"), Mapping)
        or value.get("retry_replay_resample_or_secret_rotation_count") != 0
    ):
        raise MaudExtractionP1FormalControllerError(
            "acquisition receipt drifted"
        )
    return digest


def load_acquisition_archive(
    acquisition_root: Path, *, block: str, role: str
) -> tuple[Path, ArchiveBinding, str]:
    if block not in BLOCKS or role not in {"action", "gold"}:
        raise MaudExtractionP1FormalControllerError(
            "acquisition archive request drifted"
        )
    if block == BLOCK_F_SEARCH and role == "gold":
        raise MaudExtractionP1FormalControllerError(
            "F_search gold is structurally forbidden"
        )
    phase = "test" if block == BLOCK_M_SEARCH else "initial"
    receipt_path = acquisition_root / (
        "test_parse.receipt.json"
        if phase == "test"
        else "initial.receipt.json"
    )
    receipt = read_canonical_private(receipt_path)
    receipt_sha256 = _validate_acquisition_receipt(
        receipt, phase=phase
    )
    archives = receipt["private_archives"]
    assert isinstance(archives, Mapping)
    key = f"{block}_{role}"
    binding = ArchiveBinding.from_mapping(archives.get(key))
    path = acquisition_root / (
        f"{block}.action.private.json"
        if role == "action"
        else f"{block}.gold.sealed.private.json"
    )
    _verify_file_binding(
        path,
        binding.as_mapping(),
        embedded_self_hash_field=(
            "gold_pack_sha256" if role == "gold" else None
        ),
    )
    return path, binding, receipt_sha256


def _load_action_view(
    path: Path,
    *,
    block: str,
    binding: ArchiveBinding,
) -> tuple[LabelFreeContract, ...]:
    _verify_file_binding(path, binding.as_mapping())
    value = read_canonical_private(path)
    expected_split = {
        BLOCK_A_FORM: "TRAIN",
        BLOCK_F_SEARCH: "TRAIN",
        BLOCK_A_HOLD: "DEV",
        BLOCK_M_SEARCH: "TEST",
    }[block]
    expected_keys = {
        "answerability_gold_text_offset_or_span_included",
        "block",
        "contract_count",
        "contracts",
        "item_count",
        "schema",
        "split",
        "study_id",
    }
    contracts_raw = value.get("contracts")
    if (
        set(value) != expected_keys
        or value.get("schema") != ACTION_VIEW_SCHEMA
        or value.get("study_id") != STUDY_ID
        or value.get("split") != expected_split
        or value.get("block") != block
        or value.get(
            "answerability_gold_text_offset_or_span_included"
        )
        is not False
        or isinstance(contracts_raw, (str, bytes))
        or not isinstance(contracts_raw, Sequence)
        or not contracts_raw
        or value.get("contract_count") != len(contracts_raw)
    ):
        raise MaudExtractionP1FormalControllerError(
            "label-free action view drifted"
        )
    contracts: list[LabelFreeContract] = []
    contract_ids: set[str] = set()
    work_ids: set[str] = set()
    item_count = 0
    for raw_contract in contracts_raw:
        if (
            not isinstance(raw_contract, Mapping)
            or set(raw_contract)
            != {
                "context",
                "context_sha256",
                "contract_work_id",
                "items",
            }
        ):
            raise MaudExtractionP1FormalControllerError(
                "label-free contract shape drifted"
            )
        contract_id = _validate_sha256(
            raw_contract.get("contract_work_id"), "contract work ID"
        )
        context_sha256 = _validate_sha256(
            raw_contract.get("context_sha256"), "context SHA-256"
        )
        context = raw_contract.get("context")
        raw_items = raw_contract.get("items")
        if (
            contract_id in contract_ids
            or not isinstance(context, str)
            or not context
            or hashlib.sha256(context.encode("utf-8")).hexdigest()
            != context_sha256
            or isinstance(raw_items, (str, bytes))
            or not isinstance(raw_items, Sequence)
            or len(raw_items) != 22
        ):
            raise MaudExtractionP1FormalControllerError(
                "label-free contract values drifted"
            )
        contract_ids.add(contract_id)
        passages = core.build_passages(context)
        if len(passages) < core.TOP_K:
            raise MaudExtractionP1FormalControllerError(
                "contract cannot form a top-five passage action"
            )
        serialized = tuple(
            passage.serialized_bytes().decode("ascii")
            for passage in passages
        )
        items: list[LabelFreeItem] = []
        deal_point_types: set[str] = set()
        for raw_item in raw_items:
            if (
                not isinstance(raw_item, Mapping)
                or set(raw_item)
                != {
                    "deal_point_type",
                    "family",
                    "question",
                    "work_id",
                }
            ):
                raise MaudExtractionP1FormalControllerError(
                    "label-free item shape drifted"
                )
            work_id = _validate_sha256(
                raw_item.get("work_id"), "work ID"
            )
            question = raw_item.get("question")
            deal_point_type = raw_item.get("deal_point_type")
            family = raw_item.get("family")
            if (
                work_id in work_ids
                or not isinstance(question, str)
                or not question.strip()
                or not isinstance(deal_point_type, str)
                or not deal_point_type.strip()
                or deal_point_type in deal_point_types
                or family not in core.QUERY_FAMILIES
            ):
                raise MaudExtractionP1FormalControllerError(
                    "label-free item values drifted"
                )
            work_ids.add(work_id)
            deal_point_types.add(deal_point_type)
            items.append(
                LabelFreeItem(
                    work_id=work_id,
                    question=question,
                    deal_point_type=deal_point_type,
                    family=str(family),
                )
            )
        item_count += len(items)
        contracts.append(
            LabelFreeContract(
                contract_work_id=contract_id,
                context_sha256=context_sha256,
                context=context,
                passages=passages,
                serialized_passages=serialized,
                items=tuple(items),
            )
        )
    if value.get("item_count") != item_count:
        raise MaudExtractionP1FormalControllerError(
            "label-free action view item count drifted"
        )
    return tuple(contracts)


def _coordinate_private_input(
    contracts: Sequence[LabelFreeContract],
) -> dict[str, object]:
    payload = coordinate_worker.private_input_payload(
        [
            {
                "contract_id": contract.contract_work_id,
                "passages": [
                    {
                        "ordinal": passage.ordinal,
                        "start": passage.start,
                        "end": passage.end,
                        # This exact serialized string is also supplied to the
                        # official HippoRAG payload below.
                        "text": contract.serialized_passages[
                            passage.ordinal
                        ],
                    }
                    for passage in contract.passages
                ],
                "queries": [
                    {
                        "work_id": item.work_id,
                        "family": item.family,
                        "question": item.question,
                    }
                    for item in contract.items
                ],
            }
            for contract in contracts
        ]
    )
    return payload


def _coordinate_environment(
    paths: runtime.RuntimePaths,
    *,
    physical_gpu: str,
    private_root: Path,
) -> dict[str, str]:
    if physical_gpu not in runtime.PHYSICAL_GPU_IDS:
        raise MaudExtractionP1FormalControllerError(
            "coordinate GPU lane drifted"
        )
    private_directories = {
        "HOME": private_root / "home",
        "HF_HOME": private_root / "hf_home",
        "TRANSFORMERS_CACHE": private_root / "transformers_cache",
        "TMPDIR": private_root / "tmp",
    }
    for path in set(private_directories.values()):
        _ensure_private_directory(path, fresh=True)
    environment = {
        "PATH": f"{Path(paths.typed_python).parent}:/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "CUDA_VISIBLE_DEVICES": physical_gpu,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": paths.typed_pythonpath(),
        "HOME": str(private_directories["HOME"]),
        "HF_HOME": str(private_directories["HF_HOME"]),
        "TRANSFORMERS_CACHE": str(
            private_directories["TRANSFORMERS_CACHE"]
        ),
        "TMPDIR": str(private_directories["TMPDIR"]),
        "TEMP": str(private_directories["TMPDIR"]),
        "TMP": str(private_directories["TMPDIR"]),
        **runtime.CPU_THREAD_ENV,
    }
    fixed_nonsecret_exceptions = {
        "HF_HUB_DISABLE_TELEMETRY",
        "TOKENIZERS_PARALLELISM",
    }
    forbidden_fragments = ("API", "TOKEN", "SECRET", "KEY", "PROXY")
    if any(
        any(fragment in key.upper() for fragment in forbidden_fragments)
        and key not in fixed_nonsecret_exceptions
        for key in environment
    ):
        raise MaudExtractionP1FormalControllerError(
            "credential-like environment key escaped the clean allowlist"
        )
    return environment


def _run_logged_subprocess(
    *,
    command: Sequence[str],
    cwd: Path,
    environment: Mapping[str, str],
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: int,
    runner: Callable[..., object],
) -> dict[str, object]:
    if not command or any(not isinstance(row, str) or not row for row in command):
        raise MaudExtractionP1FormalControllerError(
            "subprocess command drifted"
        )
    stdout_descriptor = os.open(
        stdout_path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    stderr_descriptor = os.open(
        stderr_path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(stdout_descriptor, "wb") as stdout_handle, (
            os.fdopen(stderr_descriptor, "wb")
        ) as stderr_handle:
            os.fchmod(stdout_handle.fileno(), 0o600)
            os.fchmod(stderr_handle.fileno(), 0o600)
            completed = runner(
                list(command),
                check=False,
                cwd=cwd,
                env=dict(environment),
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                timeout=timeout_seconds,
            )
            stdout_handle.flush()
            stderr_handle.flush()
            os.fsync(stdout_handle.fileno())
            os.fsync(stderr_handle.fileno())
    except BaseException:
        raise
    returncode = getattr(completed, "returncode", None)
    if type(returncode) is not int or returncode != 0:
        raise MaudExtractionP1FormalControllerError(
            "one-shot subprocess failed; retry is forbidden"
        )
    return {
        "returncode": returncode,
        "stdout_sha256": _sha256_file(stdout_path),
        "stdout_bytes": stdout_path.stat().st_size,
        "stderr_sha256": _sha256_file(stderr_path),
        "stderr_bytes": stderr_path.stat().st_size,
        "mode_octal": "0600",
    }


def _create_model_alias(
    cwd: Path, *, alias: str, target: str
) -> None:
    _ensure_private_directory(cwd, fresh=True)
    target_path = Path(target)
    if target_path.is_symlink() or not target_path.is_dir():
        raise MaudExtractionP1FormalControllerError(
            "coordinate model target drifted"
        )
    link = cwd / alias
    os.symlink(str(target_path), link)
    if (
        not link.is_symlink()
        or os.readlink(link) != str(target_path)
        or not os.path.samefile(link.resolve(strict=True), target_path)
    ):
        raise MaudExtractionP1FormalControllerError(
            "coordinate model alias drifted"
        )


def _validate_coordinate_output(
    value: Mapping[str, object],
    *,
    role: str,
    private_input: Mapping[str, object],
    expected_model_sha256: str,
) -> dict[str, object]:
    body = dict(value)
    declared = body.pop("self_sha256", None)
    if (
        not isinstance(declared, str)
        or _HEX64.fullmatch(declared) is None
        or coordinate_worker.semantic_sha256(body) != declared
        or value.get("schema")
        != f"{coordinate_worker.VERSION}_private_output_v1"
        or value.get("study_id") != STUDY_ID
        or value.get("role") != role
        or value.get("input_sha256")
        != coordinate_worker.semantic_sha256(private_input)
        or value.get("model_tree_sha256") != expected_model_sha256
        or value.get("retry_replay_resample_count") != 0
        or value.get("dynamic_batch_resize_count") != 0
        or value.get("network_or_API_call_count") != 0
    ):
        raise MaudExtractionP1FormalControllerError(
            "coordinate output binding drifted"
        )
    # Reuse the worker's strict public output constructor as a validator.
    checked = coordinate_worker.coordinate_output(
        role=role,
        rows=value.get("rows", ()),  # type: ignore[arg-type]
        input_sha256=str(value.get("input_sha256")),
        model_tree_sha256=str(value.get("model_tree_sha256")),
        contract_pairwise=value.get(
            "contract_pairwise", ()
        ),  # type: ignore[arg-type]
    )
    if checked != dict(value):
        raise MaudExtractionP1FormalControllerError(
            "coordinate output canonical projection drifted"
        )
    return checked


def run_coordinate_workers(
    *,
    private_input: Mapping[str, object],
    stage_root: Path,
    runtime_paths: runtime.RuntimePaths,
    runner: Callable[..., object] = subprocess.run,
) -> CoordinateBatchResult:
    """Run MiniLM on physical GPU0 and CE on GPU1, submit before joining."""

    coordinate_worker.validate_private_input(private_input)
    input_path = stage_root / "coordinate.input.private.json"
    write_private_json_once(input_path, dict(private_input))
    roles = (
        (
            coordinate_worker.ROLE_MINILM,
            "minilm",
            runtime_paths.minilm_model_root,
            str(runtime.EXPECTED_MINILM_TREE["tree_sha256"]),
            "0",
        ),
        (
            coordinate_worker.ROLE_CROSS_ENCODER,
            "cross_encoder",
            runtime_paths.cross_encoder_model_root,
            str(runtime.EXPECTED_CROSS_ENCODER_TREE["tree_sha256"]),
            "1",
        ),
    )

    def launch(
        role: str,
        alias: str,
        target: str,
        tree_sha256: str,
        gpu: str,
    ) -> dict[str, object]:
        role_root = stage_root / f"coordinate_{role.lower()}"
        _create_model_alias(role_root, alias=alias, target=target)
        output_path = role_root / "output.private.json"
        command = (
            runtime_paths.typed_python,
            "-S",
            "-B",
            "-m",
            (
                "assumption_agent.benchmarks."
                "maud_extraction_p1_coordinate_worker_v1"
            ),
            "--role",
            role,
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--model",
            alias,
            "--model-tree-sha256",
            tree_sha256,
        )
        if target in command:
            raise MaudExtractionP1FormalControllerError(
                "absolute model path escaped into coordinate argv"
            )
        _run_logged_subprocess(
            command=command,
            cwd=role_root,
            environment=_coordinate_environment(
                runtime_paths,
                physical_gpu=gpu,
                private_root=role_root,
            ),
            stdout_path=role_root / "stdout.private.bin",
            stderr_path=role_root / "stderr.private.bin",
            timeout_seconds=COORDINATE_TIMEOUT_SECONDS,
            runner=runner,
        )
        return read_canonical_private(output_path)

    futures: dict[Future[dict[str, object]], str] = {}
    outputs: dict[str, dict[str, object]] = {}
    with ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="maud-p1-coordinates"
    ) as executor:
        for role, alias, target, tree_sha256, gpu in roles:
            future = executor.submit(
                launch, role, alias, target, tree_sha256, gpu
            )
            futures[future] = role
        for future in as_completed(futures):
            outputs[futures[future]] = future.result()
    if set(outputs) != set(coordinate_worker.ROLES):
        raise MaudExtractionP1FormalControllerError(
            "coordinate worker pair is incomplete"
        )
    minilm = _validate_coordinate_output(
        outputs[coordinate_worker.ROLE_MINILM],
        role=coordinate_worker.ROLE_MINILM,
        private_input=private_input,
        expected_model_sha256=str(
            runtime.EXPECTED_MINILM_TREE["tree_sha256"]
        ),
    )
    cross_encoder = _validate_coordinate_output(
        outputs[coordinate_worker.ROLE_CROSS_ENCODER],
        role=coordinate_worker.ROLE_CROSS_ENCODER,
        private_input=private_input,
        expected_model_sha256=str(
            runtime.EXPECTED_CROSS_ENCODER_TREE["tree_sha256"]
        ),
    )
    return CoordinateBatchResult(
        minilm_output=minilm,
        cross_encoder_output=cross_encoder,
    )


@dataclass(frozen=True)
class SyntheticCanaryFixture:
    passages: tuple[core.Passage, ...]
    families: tuple[str, ...]
    coordinate_input: Mapping[str, object]
    official_payload: Mapping[str, object]


def synthetic_canary_fixture() -> SyntheticCanaryFixture:
    """Build one corpus/query fixture shared by both coordinate and official."""

    sections = []
    for index in range(1, 15):
        sections.append(
            (
                f'Section {index}. "Aurora Event {index}" means a public '
                "synthetic change. The Aurora Event shall satisfy a "
                "synthetic covenant and closing condition. Except as "
                "provided in the preceding public clause, a termination "
                "remedy and matching right apply. This is public filler "
                "text used only to exercise deterministic overlapping "
                "passage construction. "
            )
            * 2
        )
    context = "\n".join(sections)
    passages = core.build_passages(context)
    if len(passages) < core.TOP_K:
        raise MaudExtractionP1FormalControllerError(
            "public full canary passage construction drifted"
        )
    contract_id = hashlib.sha256(
        b"maud-p1-public-full-canary-contract-v1"
    ).hexdigest()
    families = tuple(
        core.QUERY_FAMILIES[index % len(core.QUERY_FAMILIES)]
        for index in range(22)
    )
    queries = tuple(
        {
            "ordinal": index,
            "work_id": hashlib.sha256(
                f"maud-p1-public-full-canary-query-{index:02d}".encode(
                    "ascii"
                )
            ).hexdigest(),
            "text": (
                "Which public synthetic passages concern "
                f"clause {index % len(passages) + 1}?"
            ),
        }
        for index in range(22)
    )
    serialized = tuple(
        passage.serialized_bytes().decode("ascii")
        for passage in passages
    )
    coordinate_input = coordinate_worker.private_input_payload(
        [
            {
                "contract_id": contract_id,
                "passages": [
                    {
                        "ordinal": passage.ordinal,
                        "start": passage.start,
                        "end": passage.end,
                        "text": serialized[passage.ordinal],
                    }
                    for passage in passages
                ],
                "queries": [
                    {
                        "work_id": query["work_id"],
                        "family": families[index],
                        "question": query["text"],
                    }
                    for index, query in enumerate(queries)
                ],
            }
        ]
    )
    official_payload = official_worker.input_payload(
        contract_work_id=contract_id,
        documents=[
            {"ordinal": index, "text": text}
            for index, text in enumerate(serialized)
        ],
        queries=list(queries),
    )
    coordinate_contract = coordinate_input["contracts"][0]  # type: ignore[index]
    if (
        [row["text"] for row in coordinate_contract["passages"]]  # type: ignore[index]
        != [row["text"] for row in official_payload["documents"]]  # type: ignore[index]
        or [
            (row["work_id"], row["question"])
            for row in coordinate_contract["queries"]  # type: ignore[index]
        ]
        != [
            (row["work_id"], row["text"])
            for row in official_payload["queries"]  # type: ignore[index]
        ]
    ):
        raise MaudExtractionP1FormalControllerError(
            "public canary same-corpus/query binding drifted"
        )
    return SyntheticCanaryFixture(
        passages=passages,
        families=families,
        coordinate_input=coordinate_input,
        official_payload=official_payload,
    )


def synthetic_coordinate_input() -> dict[str, object]:
    """Compatibility projection of the single full public fixture."""

    return dict(synthetic_canary_fixture().coordinate_input)


def run_full_source_free_canary(
    *,
    runtime_paths: runtime.RuntimePaths,
    runtime_fingerprint_sha256: str,
    canary_root: Path,
    coordinate_launcher: CoordinateBatchLauncher = run_coordinate_workers,
    official_launcher: Callable[..., runtime.WorkerRun] = (
        runtime.production_contract_launcher
    ),
) -> dict[str, object]:
    """Exercise both coordinate roles concurrently and official Hippo once."""

    fingerprint = _validate_sha256(
        runtime_fingerprint_sha256, "runtime fingerprint"
    )
    _ensure_private_directory(canary_root, fresh=True)
    coordinate_root = canary_root / "coordinate"
    _ensure_private_directory(coordinate_root, fresh=True)
    fixture = synthetic_canary_fixture()
    coordinate_input = fixture.coordinate_input
    coordinate_result = coordinate_launcher(
        private_input=coordinate_input,
        stage_root=coordinate_root,
        runtime_paths=runtime_paths,
    )
    if not isinstance(coordinate_result, CoordinateBatchResult):
        raise MaudExtractionP1FormalControllerError(
            "public coordinate canary result drifted"
        )
    minilm = _validate_coordinate_output(
        coordinate_result.minilm_output,
        role=coordinate_worker.ROLE_MINILM,
        private_input=coordinate_input,
        expected_model_sha256=str(
            runtime.EXPECTED_MINILM_TREE["tree_sha256"]
        ),
    )
    cross_encoder = _validate_coordinate_output(
        coordinate_result.cross_encoder_output,
        role=coordinate_worker.ROLE_CROSS_ENCODER,
        private_input=coordinate_input,
        expected_model_sha256=str(
            runtime.EXPECTED_CROSS_ENCODER_TREE["tree_sha256"]
        ),
    )
    official_run = official_launcher(
        payload=fixture.official_payload,
        runtime_paths=runtime_paths,
        scratch_root=canary_root / "official_scratch",
        private_custody_root=canary_root / "official_custody",
        physical_gpu="0",
        timeout_seconds=runtime.WORKER_TIMEOUT_SECONDS,
    )
    if not isinstance(official_run, runtime.WorkerRun):
        raise MaudExtractionP1FormalControllerError(
            "public official canary result drifted"
        )
    official_output = official_worker.parse_output(
        official_worker.canonical_json_bytes(official_run.output)
    )
    if (
        official_output.get("contract_work_id")
        != fixture.official_payload["contract_work_id"]
        or official_output.get("corpus_sha256")
        != fixture.official_payload["corpus_sha256"]
    ):
        raise MaudExtractionP1FormalControllerError(
            "public official canary output binding drifted"
        )
    minilm_rows = {
        str(row["work_id"]): tuple(row["scores"])  # type: ignore[index]
        for row in minilm["rows"]  # type: ignore[index]
    }
    cross_rows = {
        str(row["work_id"]): tuple(row["scores"])  # type: ignore[index]
        for row in cross_encoder["rows"]  # type: ignore[index]
    }
    pairwise_rows = minilm["contract_pairwise"]  # type: ignore[index]
    pairwise = tuple(
        tuple(int(score) for score in row)
        for row in pairwise_rows[0]["pairwise_scores"]  # type: ignore[index]
    )
    typed_edges = core.build_typed_edges(fixture.passages)
    query_rows = fixture.official_payload["queries"]
    e0_behavior_hashes: list[str] = []
    for index, query in enumerate(query_rows):  # type: ignore[arg-type]
        work_id = str(query["work_id"])
        table = core.build_coordinate_table_from_quantized(
            query=str(query["text"]),
            passages=fixture.passages,
            cross_encoder_sigmoid=cross_rows[work_id],
            minilm_unit_interval=minilm_rows[work_id],
            pairwise_minilm_unit_interval=pairwise,
        )
        slate = core.build_recipe_slate(
            query=str(query["text"]),
            passages=fixture.passages,
            coordinates=table,
            edges=typed_edges,
        )
        if tuple(action.recipe_id for action in slate.actions) != core.RECIPE_IDS:
            raise MaudExtractionP1FormalControllerError(
                "public canary nine-recipe registry drifted"
            )
        selected = core.select_e0(slate)
        e0_behavior_hashes.append(
            slate.actions[selected.registry_ordinal].behavior_sha256
        )
    body = {
        "schema": FULL_CANARY_SCHEMA,
        "study_id": STUDY_ID,
        "study_design_self_sha256": STUDY_DESIGN_SELF_SHA256,
        "pre_source_clarification_self_sha256": (
            PRE_SOURCE_CLARIFICATION_SELF_SHA256
        ),
        "status": (
            "passed_source_free_coordinate_pair_and_official_hipporag"
        ),
        "runtime_fingerprint_sha256": fingerprint,
        "coordinate_fixture_sha256": coordinate_worker.semantic_sha256(
            coordinate_input
        ),
        "minilm_output_self_sha256": minilm["self_sha256"],
        "cross_encoder_output_self_sha256": cross_encoder["self_sha256"],
        "official_output_sha256": runtime.semantic_sha256(
            official_output
        ),
        "official_safe_terminal_sha256": runtime.semantic_sha256(
            official_run.safe_terminal
        ),
        "typed_recipe_registry_sha256": semantic_sha256(
            list(core.RECIPE_IDS)
        ),
        "E0_behavior_set_sha256": semantic_sha256(
            e0_behavior_hashes
        ),
        "shape": {
            "coordinate_contract_count": 1,
            "coordinate_query_count": 22,
            "coordinate_worker_count": 2,
            "official_contract_count": 1,
            "typed_recipe_count_per_query": len(core.RECIPE_IDS),
            "E0_selection_count": 22,
        },
        "execution": {
            "coordinate_workers_bulk_submitted_before_join": True,
            "cross_encoder_physical_gpu": "1",
            "minilm_physical_gpu": "0",
            "official_physical_gpu": "0",
            "retry_replay_resample_count": 0,
            "api_or_online_evaluator_call_count": 0,
            "formal_source_action_or_score_count": 0,
        },
    }
    return self_hashed(body)


def _coordinate_indexes(
    result: CoordinateBatchResult,
    *,
    private_input: Mapping[str, object],
) -> tuple[
    dict[str, tuple[int, ...]],
    dict[str, tuple[int, ...]],
    dict[str, tuple[tuple[int, ...], ...]],
]:
    minilm = _validate_coordinate_output(
        result.minilm_output,
        role=coordinate_worker.ROLE_MINILM,
        private_input=private_input,
        expected_model_sha256=str(
            runtime.EXPECTED_MINILM_TREE["tree_sha256"]
        ),
    )
    cross_encoder = _validate_coordinate_output(
        result.cross_encoder_output,
        role=coordinate_worker.ROLE_CROSS_ENCODER,
        private_input=private_input,
        expected_model_sha256=str(
            runtime.EXPECTED_CROSS_ENCODER_TREE["tree_sha256"]
        ),
    )

    def rows(value: Mapping[str, object]) -> dict[str, tuple[int, ...]]:
        result_rows: dict[str, tuple[int, ...]] = {}
        raw_rows = value.get("rows")
        assert isinstance(raw_rows, Sequence)
        for row in raw_rows:
            assert isinstance(row, Mapping)
            result_rows[str(row["work_id"])] = tuple(
                int(score) for score in row["scores"]  # type: ignore[index]
            )
        return result_rows

    pairwise: dict[str, tuple[tuple[int, ...], ...]] = {}
    raw_pairwise = minilm.get("contract_pairwise")
    assert isinstance(raw_pairwise, Sequence)
    for row in raw_pairwise:
        assert isinstance(row, Mapping)
        pairwise[str(row["contract_id"])] = tuple(
            tuple(int(score) for score in matrix_row)
            for matrix_row in row["pairwise_scores"]  # type: ignore[index]
        )
    return rows(minilm), rows(cross_encoder), pairwise


def _official_payload(
    contract: LabelFreeContract,
) -> dict[str, object]:
    return official_worker.input_payload(
        contract_work_id=contract.contract_work_id,
        documents=[
            {
                "ordinal": passage.ordinal,
                "text": contract.serialized_passages[passage.ordinal],
            }
            for passage in contract.passages
        ],
        queries=[
            {
                "ordinal": ordinal,
                "work_id": item.work_id,
                "text": item.question,
            }
            for ordinal, item in enumerate(contract.items)
        ],
    )


def _run_official_stage(
    contracts: Sequence[LabelFreeContract],
    *,
    stage_root: Path,
    runtime_paths: runtime.RuntimePaths,
    batch_launcher: HippoBatchLauncher,
) -> dict[str, tuple[int, ...]]:
    payloads = tuple(_official_payload(contract) for contract in contracts)
    jobs = tuple(
        runtime.ContractLaunchJob(
            payload=payload,
            scratch_root=str(
                (stage_root / f"hippo_scratch_{index:06d}").absolute()
            ),
            private_custody_root=str(
                (stage_root / f"hippo_custody_{index:06d}").absolute()
            ),
            physical_gpu=runtime.PHYSICAL_GPU_IDS[
                index % len(runtime.PHYSICAL_GPU_IDS)
            ],
        )
        for index, payload in enumerate(payloads)
    )
    launched = tuple(
        batch_launcher(jobs, runtime_paths=runtime_paths)
    )
    if len(launched) != len(payloads) or any(
        not isinstance(row, runtime.WorkerRun) for row in launched
    ):
        raise MaudExtractionP1FormalControllerError(
            "official HippoRAG batch result drifted"
        )
    top5: dict[str, tuple[int, ...]] = {}
    for payload, result in zip(payloads, launched):
        output = official_worker.parse_output(
            official_worker.canonical_json_bytes(result.output)
        )
        contract_id, corpus_hash, documents, queries = (
            official_worker.validate_input(payload)
        )
        rows = output.get("rows")
        if (
            output.get("contract_work_id") != contract_id
            or output.get("corpus_sha256") != corpus_hash
            or output.get("passage_count") != len(documents)
            or not isinstance(rows, Sequence)
            or len(rows) != len(queries)
        ):
            raise MaudExtractionP1FormalControllerError(
                "official HippoRAG output binding drifted"
            )
        for query, row in zip(queries, rows):
            if (
                not isinstance(row, Mapping)
                or row.get("work_id") != query.work_id
                or row.get("query_ordinal") != query.ordinal
            ):
                raise MaudExtractionP1FormalControllerError(
                    "official HippoRAG query binding drifted"
                )
            values = row.get("top5_passage_ordinals")
            if (
                not isinstance(values, Sequence)
                or isinstance(values, (str, bytes))
            ):
                raise MaudExtractionP1FormalControllerError(
                    "official HippoRAG top-five drifted"
                )
            top5[query.work_id] = tuple(int(value) for value in values)
    return top5


def _raw_top5(
    coordinates: core.CoordinateTable,
    passages: Sequence[core.Passage],
) -> tuple[int, ...]:
    return tuple(
        sorted(
            range(len(passages)),
            key=lambda ordinal: (
                -coordinates.cross_encoder[ordinal],
                passages[ordinal].start,
                passages[ordinal].end,
                ordinal,
            ),
        )[: core.TOP_K]
    )


def _edge_payload(edge: core.TypedEdge) -> dict[str, object]:
    return {
        "source_ordinal": edge.source_ordinal,
        "target_ordinal": edge.target_ordinal,
        "edge_family": edge.edge_family,
    }


def _action_payload(
    action: core.RecipeAction, features: core.ActionFeatures
) -> dict[str, object]:
    return {
        "recipe_id": action.recipe_id,
        "passage_ordinals": list(action.passage_ordinals),
        "accepted_edges": [
            _edge_payload(edge) for edge in action.accepted_edges
        ],
        "behavior_sha256": action.behavior_sha256,
        "features": list(features.values),
    }


def _selection_payload(
    selection: core.EvaluatorSelection,
    slate: core.RecipeSlate,
) -> dict[str, object]:
    action = slate.actions[selection.registry_ordinal]
    if isinstance(selection.score, Fraction):
        score: object = {
            "kind": "Fraction",
            "value": _fraction_payload(selection.score),
        }
    else:
        score = {"kind": "float64", "hex": float(selection.score).hex()}
    return {
        "evaluator_id": selection.evaluator_id,
        "recipe_id": selection.recipe_id,
        "registry_ordinal": selection.registry_ordinal,
        "score": score,
        "passage_ordinals": list(action.passage_ordinals),
        "behavior_sha256": action.behavior_sha256,
    }


def _passage_payload(passage: core.Passage) -> dict[str, object]:
    return {
        "ordinal": passage.ordinal,
        "context_sha256": passage.context_sha256,
        "start": passage.start,
        "end": passage.end,
        "text": passage.text,
        "exact_substring_sha256": passage.exact_substring_sha256,
        "serialized_sha256": hashlib.sha256(
            passage.serialized_bytes()
        ).hexdigest(),
    }


def _model_payload(model: core.E1RidgeModel) -> dict[str, object]:
    return {
        "identifier": "E1_AFORM_CENTERED_RIDGE_L2_1",
        "means_hex": [value.hex() for value in model.means],
        "population_standard_deviations_hex": [
            value.hex()
            for value in model.population_standard_deviations
        ],
        "weights_hex": [value.hex() for value in model.weights],
        "zero_variance_columns": list(model.zero_variance_columns),
        "training_row_count": model.training_row_count,
        "model_sha256": model.model_sha256,
    }


def _model_from_payload(value: object) -> core.E1RidgeModel:
    if (
        not isinstance(value, Mapping)
        or set(value)
        != {
            "identifier",
            "means_hex",
            "model_sha256",
            "population_standard_deviations_hex",
            "training_row_count",
            "weights_hex",
            "zero_variance_columns",
        }
        or value.get("identifier") != "E1_AFORM_CENTERED_RIDGE_L2_1"
    ):
        raise MaudExtractionP1FormalControllerError(
            "frozen E1 model payload drifted"
        )
    try:
        return core.E1RidgeModel(
            means=tuple(float.fromhex(row) for row in value["means_hex"]),  # type: ignore[index]
            population_standard_deviations=tuple(
                float.fromhex(row)
                for row in value[
                    "population_standard_deviations_hex"
                ]  # type: ignore[index]
            ),
            weights=tuple(
                float.fromhex(row)
                for row in value["weights_hex"]  # type: ignore[index]
            ),
            zero_variance_columns=tuple(
                value["zero_variance_columns"]  # type: ignore[arg-type]
            ),
            training_row_count=value["training_row_count"],  # type: ignore[arg-type]
            model_sha256=str(value["model_sha256"]),
        )
    except (TypeError, ValueError) as exc:
        raise MaudExtractionP1FormalControllerError(
            "frozen E1 model could not be reconstructed"
        ) from exc


def write_e1_model(
    path: Path,
    *,
    model: core.E1RidgeModel,
    a_form_action_archive_file_sha256: str,
    a_form_gold_file_sha256: str,
) -> tuple[dict[str, object], dict[str, object]]:
    body = {
        "schema": MODEL_SCHEMA,
        "study_id": STUDY_ID,
        "study_design_self_sha256": STUDY_DESIGN_SELF_SHA256,
        "a_form_action_archive_file_sha256": _validate_sha256(
            a_form_action_archive_file_sha256,
            "A_form action archive",
        ),
        "a_form_gold_file_sha256": _validate_sha256(
            a_form_gold_file_sha256, "A_form gold"
        ),
        "model": _model_payload(model),
        "fit_block": BLOCK_A_FORM,
        "refit_count": 0,
        "online_evaluator_or_API_call_count": 0,
    }
    envelope = self_hashed(body)
    binding = write_private_json_once(path, envelope)
    return envelope, binding


def load_e1_model(
    path: Path,
) -> tuple[core.E1RidgeModel, dict[str, object], str]:
    value = read_canonical_private(path)
    self_sha256 = verify_self_hash(value)
    if (
        set(value)
        != {
            "a_form_action_archive_file_sha256",
            "a_form_gold_file_sha256",
            "fit_block",
            "model",
            "online_evaluator_or_API_call_count",
            "refit_count",
            "schema",
            "self_sha256",
            "study_design_self_sha256",
            "study_id",
        }
        or value.get("schema") != MODEL_SCHEMA
        or value.get("study_id") != STUDY_ID
        or value.get("study_design_self_sha256")
        != STUDY_DESIGN_SELF_SHA256
        or value.get("fit_block") != BLOCK_A_FORM
        or value.get("refit_count") != 0
        or value.get("online_evaluator_or_API_call_count") != 0
    ):
        raise MaudExtractionP1FormalControllerError(
            "frozen E1 model envelope drifted"
        )
    return _model_from_payload(value["model"]), value, self_sha256


def run_action_stage(
    *,
    block: str,
    action_view_path: Path,
    action_view_binding: ArchiveBinding,
    acquisition_receipt_self_sha256: str,
    stage_root: Path,
    runtime_paths: runtime.RuntimePaths,
    e1_model_path: Path | None = None,
    coordinate_launcher: CoordinateBatchLauncher = run_coordinate_workers,
    hippo_batch_launcher: HippoBatchLauncher = runtime.run_contract_batch,
) -> StageResult:
    """Form and seal every label-free action; this API has no gold argument."""

    if block not in BLOCKS:
        raise MaudExtractionP1FormalControllerError(
            "unknown formal block"
        )
    _ensure_private_directory(stage_root, fresh=True)
    contracts = _load_action_view(
        action_view_path, block=block, binding=action_view_binding
    )
    private_input = _coordinate_private_input(contracts)
    coordinate_result = coordinate_launcher(
        private_input=private_input,
        stage_root=stage_root,
        runtime_paths=runtime_paths,
    )
    if not isinstance(coordinate_result, CoordinateBatchResult):
        raise MaudExtractionP1FormalControllerError(
            "coordinate launcher result drifted"
        )
    minilm_rows, cross_encoder_rows, pairwise_rows = (
        _coordinate_indexes(
            coordinate_result, private_input=private_input
        )
    )
    expected_work = {
        item.work_id for contract in contracts for item in contract.items
    }
    expected_contracts = {
        contract.contract_work_id for contract in contracts
    }
    if (
        set(minilm_rows) != expected_work
        or set(cross_encoder_rows) != expected_work
        or set(pairwise_rows) != expected_contracts
    ):
        raise MaudExtractionP1FormalControllerError(
            "coordinate output work registry drifted"
        )

    official_top5: dict[str, tuple[int, ...]] = {}
    if block in OFFICIAL_BLOCKS:
        official_top5 = _run_official_stage(
            contracts,
            stage_root=stage_root,
            runtime_paths=runtime_paths,
            batch_launcher=hippo_batch_launcher,
        )
        if set(official_top5) != expected_work:
            raise MaudExtractionP1FormalControllerError(
                "official HippoRAG work registry drifted"
            )
    model: core.E1RidgeModel | None = None
    model_self_sha256: str | None = None
    if block != BLOCK_A_FORM:
        if e1_model_path is None:
            raise MaudExtractionP1FormalControllerError(
                "post-A_form action stage requires frozen E1"
            )
        model, _envelope, model_self_sha256 = load_e1_model(
            e1_model_path
        )
    elif e1_model_path is not None:
        raise MaudExtractionP1FormalControllerError(
            "A_form cannot consume a preexisting evaluator"
        )

    archived_contracts: list[dict[str, object]] = []
    e0_e1_equal = 0
    for contract in contracts:
        edges = core.build_typed_edges(contract.passages)
        archived_items: list[dict[str, object]] = []
        for item in contract.items:
            coordinates = core.build_coordinate_table_from_quantized(
                query=item.question,
                passages=contract.passages,
                cross_encoder_sigmoid=cross_encoder_rows[item.work_id],
                minilm_unit_interval=minilm_rows[item.work_id],
                pairwise_minilm_unit_interval=pairwise_rows[
                    contract.contract_work_id
                ],
            )
            slate = core.build_recipe_slate(
                query=item.question,
                passages=contract.passages,
                coordinates=coordinates,
                edges=edges,
            )
            e0 = core.select_e0(slate)
            e1 = (
                None
                if model is None
                else core.select_e1(model, slate, item.family)
            )
            if e1 is not None and (
                slate.actions[e0.registry_ordinal].behavior_sha256
                == slate.actions[e1.registry_ordinal].behavior_sha256
            ):
                e0_e1_equal += 1
            archived_items.append(
                {
                    "work_id": item.work_id,
                    "family": item.family,
                    "recipe_slate": [
                        _action_payload(action, features)
                        for action, features in zip(
                            slate.actions, slate.features
                        )
                    ],
                    "E0": _selection_payload(e0, slate),
                    "E1": (
                        None
                        if e1 is None
                        else _selection_payload(e1, slate)
                    ),
                    "RAW": {
                        "passage_ordinals": list(
                            _raw_top5(coordinates, contract.passages)
                        )
                    },
                    "official_HippoRAG": (
                        None
                        if block not in OFFICIAL_BLOCKS
                        else {
                            "passage_ordinals": list(
                                official_top5[item.work_id]
                            )
                        }
                    ),
                }
            )
        archived_contracts.append(
            {
                "contract_work_id": contract.contract_work_id,
                "context_sha256": contract.context_sha256,
                "passages": [
                    _passage_payload(passage)
                    for passage in contract.passages
                ],
                "typed_edges": [
                    _edge_payload(edge) for edge in edges
                ],
                "items": archived_items,
            }
        )
    archive_body = {
        "schema": ACTION_ARCHIVE_SCHEMA,
        "study_id": STUDY_ID,
        "study_design_self_sha256": STUDY_DESIGN_SELF_SHA256,
        "block": block,
        "status": "all_label_free_actions_complete_and_sealed_before_gold",
        "acquisition_receipt_self_sha256": _validate_sha256(
            acquisition_receipt_self_sha256,
            "acquisition receipt",
        ),
        "action_view_file_sha256": action_view_binding.file_sha256,
        "action_view_semantic_sha256": (
            action_view_binding.semantic_sha256
        ),
        "coordinate_input_semantic_sha256": (
            coordinate_worker.semantic_sha256(private_input)
        ),
        "evaluator_model_self_sha256": model_self_sha256,
        "contract_count": len(contracts),
        "item_count": sum(len(row.items) for row in contracts),
        "contracts": archived_contracts,
        "F_search_behavior_identity_only": block == BLOCK_F_SEARCH,
        "gold_open_count_before_archive": 0,
        "retry_replay_resample_count": 0,
        "online_evaluator_API_or_fine_tune_count": 0,
    }
    archive = self_hashed(archive_body)
    archive_path = stage_root / "action.archive.private.json"
    archive_binding = write_private_json_once(archive_path, archive)
    verified = read_canonical_private(archive_path)
    archive_self_sha256 = verify_self_hash(verified)
    if verified != archive:
        raise MaudExtractionP1FormalControllerError(
            "sealed action archive readback drifted"
        )
    receipt_body = {
        "schema": ACTION_RECEIPT_SCHEMA,
        "study_id": STUDY_ID,
        "study_design_self_sha256": STUDY_DESIGN_SELF_SHA256,
        "block": block,
        "status": "action_archive_sealed",
        "action_archive_file_sha256": archive_binding["file_sha256"],
        "action_archive_semantic_sha256": (
            archive_binding["semantic_sha256"]
        ),
        "action_archive_self_sha256": archive_self_sha256,
        "contract_count": archive_body["contract_count"],
        "item_count": archive_body["item_count"],
        "evaluator_model_self_sha256": model_self_sha256,
        "F_search_E0_E1_equal_behavior_count": (
            e0_e1_equal if block == BLOCK_F_SEARCH else None
        ),
        "gold_open_count": 0,
        "raw_contract_item_context_question_or_action_included": False,
    }
    receipt = self_hashed(receipt_body)
    receipt_path = stage_root / "action.receipt.json"
    write_private_json_once(receipt_path, receipt)
    return StageResult(
        block=block,
        archive_path=archive_path,
        archive_file_sha256=str(
            archive_binding["file_sha256"]
        ),
        archive_semantic_sha256=str(
            archive_binding["semantic_sha256"]
        ),
        archive_self_sha256=archive_self_sha256,
        receipt_path=receipt_path,
        receipt_self_sha256=str(receipt["self_sha256"]),
        contract_count=len(contracts),
        item_count=sum(len(row.items) for row in contracts),
    )


@dataclass(frozen=True)
class ArchivedItem:
    work_id: str
    family: str
    slate: core.RecipeSlate
    e0_registry_ordinal: int
    e1_registry_ordinal: int | None
    raw_top5: tuple[int, ...]
    hippo_top5: tuple[int, ...] | None


@dataclass(frozen=True)
class ArchivedContract:
    contract_work_id: str
    context_sha256: str
    passages: tuple[core.Passage, ...]
    items: tuple[ArchivedItem, ...]


@dataclass(frozen=True)
class LoadedActionArchive:
    block: str
    self_sha256: str
    file_sha256: str
    semantic_sha256: str
    acquisition_receipt_self_sha256: str
    evaluator_model_self_sha256: str | None
    contracts: tuple[ArchivedContract, ...]


def _top5(value: object, *, passage_count: int) -> tuple[int, ...]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != core.TOP_K
        or len(set(value)) != core.TOP_K
        or any(
            type(row) is not int or not 0 <= row < passage_count
            for row in value
        )
    ):
        raise MaudExtractionP1FormalControllerError(
            "archived top-five drifted"
        )
    return tuple(int(row) for row in value)


def _typed_edge_from_payload(
    value: object, *, passage_count: int
) -> core.TypedEdge:
    if (
        not isinstance(value, Mapping)
        or set(value)
        != {"edge_family", "source_ordinal", "target_ordinal"}
    ):
        raise MaudExtractionP1FormalControllerError(
            "archived typed edge drifted"
        )
    edge = core.TypedEdge(
        source_ordinal=value.get("source_ordinal"),  # type: ignore[arg-type]
        target_ordinal=value.get("target_ordinal"),  # type: ignore[arg-type]
        edge_family=str(value.get("edge_family")),
    )
    if (
        edge.source_ordinal >= passage_count
        or edge.target_ordinal >= passage_count
    ):
        raise MaudExtractionP1FormalControllerError(
            "archived typed edge escaped the corpus"
        )
    return edge


def _slate_from_payload(
    value: object, *, passage_count: int
) -> core.RecipeSlate:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != len(core.RECIPE_IDS)
    ):
        raise MaudExtractionP1FormalControllerError(
            "archived recipe slate drifted"
        )
    actions: list[core.RecipeAction] = []
    features: list[core.ActionFeatures] = []
    for expected_recipe, row in zip(core.RECIPE_IDS, value):
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {
                "accepted_edges",
                "behavior_sha256",
                "features",
                "passage_ordinals",
                "recipe_id",
            }
            or row.get("recipe_id") != expected_recipe
        ):
            raise MaudExtractionP1FormalControllerError(
                "archived recipe row drifted"
            )
        accepted = row.get("accepted_edges")
        raw_features = row.get("features")
        if (
            isinstance(accepted, (str, bytes))
            or not isinstance(accepted, Sequence)
            or isinstance(raw_features, (str, bytes))
            or not isinstance(raw_features, Sequence)
        ):
            raise MaudExtractionP1FormalControllerError(
                "archived recipe evidence drifted"
            )
        actions.append(
            core.RecipeAction(
                recipe_id=expected_recipe,
                passage_ordinals=_top5(
                    row.get("passage_ordinals"),
                    passage_count=passage_count,
                ),
                accepted_edges=tuple(
                    _typed_edge_from_payload(
                        edge, passage_count=passage_count
                    )
                    for edge in accepted
                ),
                behavior_sha256=str(row.get("behavior_sha256")),
            )
        )
        features.append(
            core.ActionFeatures(tuple(raw_features))  # type: ignore[arg-type]
        )
    return core.RecipeSlate(
        actions=tuple(actions), features=tuple(features)
    )


def _validate_selection_payload(
    value: object,
    *,
    expected: core.EvaluatorSelection,
    slate: core.RecipeSlate,
) -> int:
    if (
        not isinstance(value, Mapping)
        or set(value)
        != {
            "behavior_sha256",
            "evaluator_id",
            "passage_ordinals",
            "recipe_id",
            "registry_ordinal",
            "score",
        }
        or value.get("evaluator_id") != expected.evaluator_id
        or value.get("recipe_id") != expected.recipe_id
        or value.get("registry_ordinal") != expected.registry_ordinal
    ):
        raise MaudExtractionP1FormalControllerError(
            "archived evaluator selection drifted"
        )
    action = slate.actions[expected.registry_ordinal]
    if (
        _top5(
            value.get("passage_ordinals"),
            passage_count=max(
                ordinal
                for row in slate.actions
                for ordinal in row.passage_ordinals
            )
            + 1,
        )
        != action.passage_ordinals
        or value.get("behavior_sha256") != action.behavior_sha256
    ):
        raise MaudExtractionP1FormalControllerError(
            "archived evaluator action binding drifted"
        )
    raw_score = value.get("score")
    if isinstance(expected.score, Fraction):
        if (
            not isinstance(raw_score, Mapping)
            or set(raw_score) != {"kind", "value"}
            or raw_score.get("kind") != "Fraction"
            or _fraction_from_payload(raw_score.get("value"))
            != expected.score
        ):
            raise MaudExtractionP1FormalControllerError(
                "archived exact evaluator score drifted"
            )
    elif (
        not isinstance(raw_score, Mapping)
        or set(raw_score) != {"hex", "kind"}
        or raw_score.get("kind") != "float64"
        or raw_score.get("hex") != float(expected.score).hex()
    ):
        raise MaudExtractionP1FormalControllerError(
            "archived ridge evaluator score drifted"
        )
    return expected.registry_ordinal


def load_action_archive(
    path: Path,
    *,
    expected_block: str,
    e1_model: core.E1RidgeModel | None,
    expected_model_self_sha256: str | None,
) -> LoadedActionArchive:
    value = read_canonical_private(path)
    file_sha256 = _sha256_file(path)
    semantic = semantic_sha256(value)
    self_sha256 = verify_self_hash(value)
    raw_contracts = value.get("contracts")
    expected_keys = {
        "F_search_behavior_identity_only",
        "acquisition_receipt_self_sha256",
        "action_view_file_sha256",
        "action_view_semantic_sha256",
        "block",
        "contract_count",
        "contracts",
        "coordinate_input_semantic_sha256",
        "evaluator_model_self_sha256",
        "gold_open_count_before_archive",
        "item_count",
        "online_evaluator_API_or_fine_tune_count",
        "retry_replay_resample_count",
        "schema",
        "self_sha256",
        "status",
        "study_design_self_sha256",
        "study_id",
    }
    if (
        set(value) != expected_keys
        or value.get("schema") != ACTION_ARCHIVE_SCHEMA
        or value.get("study_id") != STUDY_ID
        or value.get("study_design_self_sha256")
        != STUDY_DESIGN_SELF_SHA256
        or value.get("block") != expected_block
        or value.get("status")
        != "all_label_free_actions_complete_and_sealed_before_gold"
        or value.get("gold_open_count_before_archive") != 0
        or value.get("retry_replay_resample_count") != 0
        or value.get("online_evaluator_API_or_fine_tune_count") != 0
        or value.get("F_search_behavior_identity_only")
        is not (expected_block == BLOCK_F_SEARCH)
        or value.get("evaluator_model_self_sha256")
        != expected_model_self_sha256
        or isinstance(raw_contracts, (str, bytes))
        or not isinstance(raw_contracts, Sequence)
        or not raw_contracts
        or value.get("contract_count") != len(raw_contracts)
    ):
        raise MaudExtractionP1FormalControllerError(
            "sealed action archive envelope drifted"
        )
    if (expected_block == BLOCK_A_FORM) != (e1_model is None):
        raise MaudExtractionP1FormalControllerError(
            "action archive evaluator stage drifted"
        )
    contracts: list[ArchivedContract] = []
    item_count = 0
    seen_contracts: set[str] = set()
    seen_work: set[str] = set()
    for raw_contract in raw_contracts:
        if (
            not isinstance(raw_contract, Mapping)
            or set(raw_contract)
            != {
                "context_sha256",
                "contract_work_id",
                "items",
                "passages",
                "typed_edges",
            }
        ):
            raise MaudExtractionP1FormalControllerError(
                "archived contract shape drifted"
            )
        contract_id = _validate_sha256(
            raw_contract.get("contract_work_id"),
            "archived contract work ID",
        )
        context_sha256 = _validate_sha256(
            raw_contract.get("context_sha256"),
            "archived context SHA-256",
        )
        raw_passages = raw_contract.get("passages")
        raw_edges = raw_contract.get("typed_edges")
        raw_items = raw_contract.get("items")
        if (
            contract_id in seen_contracts
            or isinstance(raw_passages, (str, bytes))
            or not isinstance(raw_passages, Sequence)
            or len(raw_passages) < core.TOP_K
            or isinstance(raw_edges, (str, bytes))
            or not isinstance(raw_edges, Sequence)
            or isinstance(raw_items, (str, bytes))
            or not isinstance(raw_items, Sequence)
            or len(raw_items) != 22
        ):
            raise MaudExtractionP1FormalControllerError(
                "archived contract values drifted"
            )
        seen_contracts.add(contract_id)
        passages: list[core.Passage] = []
        for ordinal, row in enumerate(raw_passages):
            if (
                not isinstance(row, Mapping)
                or set(row)
                != {
                    "context_sha256",
                    "end",
                    "exact_substring_sha256",
                    "ordinal",
                    "serialized_sha256",
                    "start",
                    "text",
                }
                or row.get("ordinal") != ordinal
                or row.get("context_sha256") != context_sha256
            ):
                raise MaudExtractionP1FormalControllerError(
                    "archived passage shape drifted"
                )
            passage = core.Passage(
                ordinal=ordinal,
                context_sha256=context_sha256,
                start=row.get("start"),  # type: ignore[arg-type]
                end=row.get("end"),  # type: ignore[arg-type]
                text=row.get("text"),  # type: ignore[arg-type]
                exact_substring_sha256=str(
                    row.get("exact_substring_sha256")
                ),
            )
            if (
                hashlib.sha256(passage.serialized_bytes()).hexdigest()
                != row.get("serialized_sha256")
            ):
                raise MaudExtractionP1FormalControllerError(
                    "archived serialized passage drifted"
                )
            passages.append(passage)
        typed_edges = tuple(
            _typed_edge_from_payload(
                edge, passage_count=len(passages)
            )
            for edge in raw_edges
        )
        if typed_edges != core.build_typed_edges(tuple(passages)):
            raise MaudExtractionP1FormalControllerError(
                "archived typed edge registry drifted"
            )
        items: list[ArchivedItem] = []
        for raw_item in raw_items:
            if (
                not isinstance(raw_item, Mapping)
                or set(raw_item)
                != {
                    "E0",
                    "E1",
                    "RAW",
                    "family",
                    "official_HippoRAG",
                    "recipe_slate",
                    "work_id",
                }
            ):
                raise MaudExtractionP1FormalControllerError(
                    "archived item shape drifted"
                )
            work_id = _validate_sha256(
                raw_item.get("work_id"), "archived work ID"
            )
            family = raw_item.get("family")
            if work_id in seen_work or family not in core.QUERY_FAMILIES:
                raise MaudExtractionP1FormalControllerError(
                    "archived item values drifted"
                )
            seen_work.add(work_id)
            slate = _slate_from_payload(
                raw_item.get("recipe_slate"),
                passage_count=len(passages),
            )
            e0 = core.select_e0(slate)
            e0_ordinal = _validate_selection_payload(
                raw_item.get("E0"), expected=e0, slate=slate
            )
            raw_e1 = raw_item.get("E1")
            if e1_model is None:
                if raw_e1 is not None:
                    raise MaudExtractionP1FormalControllerError(
                        "A_form unexpectedly contains E1"
                    )
                e1_ordinal = None
            else:
                e1 = core.select_e1(e1_model, slate, str(family))
                e1_ordinal = _validate_selection_payload(
                    raw_e1, expected=e1, slate=slate
                )
            raw_raw = raw_item.get("RAW")
            raw_hippo = raw_item.get("official_HippoRAG")
            if (
                not isinstance(raw_raw, Mapping)
                or set(raw_raw) != {"passage_ordinals"}
            ):
                raise MaudExtractionP1FormalControllerError(
                    "archived RAW action drifted"
                )
            raw_top5 = _top5(
                raw_raw.get("passage_ordinals"),
                passage_count=len(passages),
            )
            if expected_block in OFFICIAL_BLOCKS:
                if (
                    not isinstance(raw_hippo, Mapping)
                    or set(raw_hippo) != {"passage_ordinals"}
                ):
                    raise MaudExtractionP1FormalControllerError(
                        "archived HippoRAG action drifted"
                    )
                hippo_top5 = _top5(
                    raw_hippo.get("passage_ordinals"),
                    passage_count=len(passages),
                )
            else:
                if raw_hippo is not None:
                    raise MaudExtractionP1FormalControllerError(
                        "unexpected official action outside A_hold"
                    )
                hippo_top5 = None
            items.append(
                ArchivedItem(
                    work_id=work_id,
                    family=str(family),
                    slate=slate,
                    e0_registry_ordinal=e0_ordinal,
                    e1_registry_ordinal=e1_ordinal,
                    raw_top5=raw_top5,
                    hippo_top5=hippo_top5,
                )
            )
        item_count += len(items)
        contracts.append(
            ArchivedContract(
                contract_work_id=contract_id,
                context_sha256=context_sha256,
                passages=tuple(passages),
                items=tuple(items),
            )
        )
    if value.get("item_count") != item_count:
        raise MaudExtractionP1FormalControllerError(
            "archived item count drifted"
        )
    return LoadedActionArchive(
        block=expected_block,
        self_sha256=self_sha256,
        file_sha256=file_sha256,
        semantic_sha256=semantic,
        acquisition_receipt_self_sha256=_validate_sha256(
            value.get("acquisition_receipt_self_sha256"),
            "archived acquisition receipt",
        ),
        evaluator_model_self_sha256=expected_model_self_sha256,
        contracts=tuple(contracts),
    )


def _gold_authorization(
    path: Path,
    *,
    block: str,
    archive: LoadedActionArchive,
    gold_binding: ArchiveBinding,
    evaluator_model_self_sha256: str | None,
) -> dict[str, object]:
    if block not in GOLD_BLOCKS:
        raise MaudExtractionP1FormalControllerError(
            "block has no gold authorization path"
        )
    value = self_hashed(
        {
            "schema": f"{VERSION}_gold_open_authorization_v1",
            "study_id": STUDY_ID,
            "study_design_self_sha256": STUDY_DESIGN_SELF_SHA256,
            "block": block,
            "action_archive_file_sha256": archive.file_sha256,
            "action_archive_semantic_sha256": archive.semantic_sha256,
            "action_archive_self_sha256": archive.self_sha256,
            "gold_file_sha256": gold_binding.file_sha256,
            "gold_semantic_sha256": gold_binding.semantic_sha256,
            "evaluator_model_self_sha256": (
                evaluator_model_self_sha256
            ),
            "status": "action_archive_sealed_before_gold_open",
            "gold_open_count_before_authorization": 0,
        }
    )
    write_private_json_once(path, value)
    return value


def _reconstruct_context(
    passages: Sequence[core.Passage],
) -> str:
    if not passages or passages[0].start != 0:
        raise MaudExtractionP1FormalControllerError(
            "archived passages do not begin at context zero"
        )
    length = max(row.end for row in passages)
    characters: list[str | None] = [None] * length
    for passage in passages:
        for offset, character in enumerate(
            passage.text, start=passage.start
        ):
            previous = characters[offset]
            if previous is not None and previous != character:
                raise MaudExtractionP1FormalControllerError(
                    "overlapping archived passage text disagrees"
                )
            characters[offset] = character
    if any(row is None for row in characters):
        raise MaudExtractionP1FormalControllerError(
            "archived passage union has a context gap"
        )
    context = "".join(str(row) for row in characters)
    if (
        hashlib.sha256(context.encode("utf-8")).hexdigest()
        != passages[0].context_sha256
    ):
        raise MaudExtractionP1FormalControllerError(
            "archived context reconstruction drifted"
        )
    return context


def _load_gold(
    path: Path,
    *,
    block: str,
    binding: ArchiveBinding,
    archive: LoadedActionArchive,
) -> dict[str, tuple[core.CharacterInterval, ...]]:
    _verify_file_binding(
        path,
        binding.as_mapping(),
        embedded_self_hash_field="gold_pack_sha256",
    )
    value = read_canonical_private(path)
    raw_contracts = value.get("contracts")
    expected_split = {
        BLOCK_A_FORM: "TRAIN",
        BLOCK_A_HOLD: "DEV",
        BLOCK_M_SEARCH: "TEST",
    }[block]
    if (
        set(value)
        != {
            "block",
            "contract_count",
            "contracts",
            "gold_pack_sha256",
            "item_count",
            "schema",
            "split",
            "study_id",
        }
        or value.get("schema") != GOLD_PACK_SCHEMA
        or value.get("study_id") != STUDY_ID
        or value.get("block") != block
        or value.get("split") != expected_split
        or isinstance(raw_contracts, (str, bytes))
        or not isinstance(raw_contracts, Sequence)
        or len(raw_contracts) != len(archive.contracts)
        or value.get("contract_count") != len(archive.contracts)
    ):
        raise MaudExtractionP1FormalControllerError(
            "late gold pack envelope drifted"
        )
    result: dict[str, tuple[core.CharacterInterval, ...]] = {}
    item_count = 0
    for archived_contract, raw_contract in zip(
        archive.contracts, raw_contracts
    ):
        if (
            not isinstance(raw_contract, Mapping)
            or set(raw_contract) != {"contract_work_id", "items"}
            or raw_contract.get("contract_work_id")
            != archived_contract.contract_work_id
            or not isinstance(raw_contract.get("items"), Sequence)
        ):
            raise MaudExtractionP1FormalControllerError(
                "late gold contract binding drifted"
            )
        context = _reconstruct_context(archived_contract.passages)
        raw_items = raw_contract["items"]
        assert isinstance(raw_items, Sequence)
        if len(raw_items) != len(archived_contract.items):
            raise MaudExtractionP1FormalControllerError(
                "late gold item count drifted"
            )
        for archived_item, raw_item in zip(
            archived_contract.items, raw_items
        ):
            if (
                not isinstance(raw_item, Mapping)
                or set(raw_item)
                != {"merged_intervals", "spans", "work_id"}
                or raw_item.get("work_id") != archived_item.work_id
                or not isinstance(raw_item.get("spans"), Sequence)
                or not isinstance(
                    raw_item.get("merged_intervals"), Sequence
                )
            ):
                raise MaudExtractionP1FormalControllerError(
                    "late gold item binding drifted"
                )
            answers: list[core.GoldAnswer] = []
            for span in raw_item["spans"]:  # type: ignore[index]
                if (
                    not isinstance(span, Mapping)
                    or set(span) != {"end", "start", "text"}
                    or type(span.get("start")) is not int
                    or type(span.get("end")) is not int
                    or not isinstance(span.get("text"), str)
                    or span["end"]
                    != span["start"] + len(span["text"])
                ):
                    raise MaudExtractionP1FormalControllerError(
                        "late gold span drifted"
                    )
                answers.append(
                    core.GoldAnswer(
                        answer_start=int(span["start"]),
                        text=str(span["text"]),
                    )
                )
            validated = core.validate_gold_intervals(context, answers)
            declared = tuple(
                core.CharacterInterval(
                    start=row[0], end=row[1]  # type: ignore[index]
                )
                for row in raw_item["merged_intervals"]  # type: ignore[index]
            )
            if declared != validated:
                raise MaudExtractionP1FormalControllerError(
                    "late gold merged interval drifted"
                )
            result[archived_item.work_id] = validated
            item_count += 1
    if value.get("item_count") != item_count:
        raise MaudExtractionP1FormalControllerError(
            "late gold total item count drifted"
        )
    return result


def _utility(
    passages: Sequence[core.Passage],
    top5: Sequence[int],
    gold: Sequence[core.CharacterInterval],
) -> int | None:
    return core.score_evidence_coverage(
        passages=passages,
        selected_ordinals=top5,
        merged_gold_intervals=gold,
    ).primary_utility


def _coverage_score(
    passages: Sequence[core.Passage],
    top5: Sequence[int],
    gold: Sequence[core.CharacterInterval],
) -> core.CoverageScore:
    return core.score_evidence_coverage(
        passages=passages,
        selected_ordinals=top5,
        merged_gold_intervals=gold,
    )


def _coverage_payload(value: core.CoverageScore) -> dict[str, object]:
    return {
        "answerable": value.answerable,
        "primary_utility": value.primary_utility,
        "complete_at_5": value.complete_at_5,
        "coverage_at_least_half": value.coverage_at_least_half,
        "rank_discounted_incremental_utility": (
            value.rank_discounted_incremental_utility
        ),
        "merged_gold_length": value.merged_gold_length,
    }


def _score_archive_and_receipt(
    *,
    block: str,
    output_root: Path,
    archive: LoadedActionArchive,
    gold_binding: ArchiveBinding,
    detail: Mapping[str, object],
    safe_aggregates: Mapping[str, object],
) -> ScoreResult:
    score_archive = self_hashed(
        {
            "schema": SCORE_ARCHIVE_SCHEMA,
            "study_id": STUDY_ID,
            "study_design_self_sha256": STUDY_DESIGN_SELF_SHA256,
            "block": block,
            "action_archive_file_sha256": archive.file_sha256,
            "action_archive_semantic_sha256": archive.semantic_sha256,
            "gold_file_sha256": gold_binding.file_sha256,
            "gold_semantic_sha256": gold_binding.semantic_sha256,
            "private_item_scores": dict(detail),
            "online_evaluator_API_or_fine_tune_count": 0,
            "retry_replay_resample_count": 0,
        }
    )
    score_path = output_root / "score.archive.private.json"
    score_binding = write_private_json_once(
        score_path, score_archive
    )
    receipt = self_hashed(
        {
            "schema": SCORE_RECEIPT_SCHEMA,
            "study_id": STUDY_ID,
            "study_design_self_sha256": STUDY_DESIGN_SELF_SHA256,
            "block": block,
            "status": "offline_scoring_complete",
            "action_archive_file_sha256": archive.file_sha256,
            "gold_file_sha256": gold_binding.file_sha256,
            "score_archive_file_sha256": score_binding[
                "file_sha256"
            ],
            "score_archive_self_sha256": score_archive[
                "self_sha256"
            ],
            "safe_aggregates": dict(safe_aggregates),
            "raw_contract_item_action_gold_or_score_included": False,
            "online_evaluator_API_or_fine_tune_count": 0,
        }
    )
    receipt_path = output_root / "score.receipt.json"
    write_private_json_once(receipt_path, receipt)
    return ScoreResult(
        block=block,
        score_archive_path=score_path,
        score_archive_file_sha256=str(
            score_binding["file_sha256"]
        ),
        receipt_path=receipt_path,
        receipt=receipt,
    )


def score_a_form(
    *,
    action_archive_path: Path,
    gold_path: Path,
    gold_binding: ArchiveBinding,
    output_root: Path,
) -> tuple[ModelResult, ScoreResult]:
    """Open A_form gold after sealing and fit the sole frozen E1 once."""

    _ensure_private_directory(output_root, fresh=True)
    archive = load_action_archive(
        action_archive_path,
        expected_block=BLOCK_A_FORM,
        e1_model=None,
        expected_model_self_sha256=None,
    )
    _gold_authorization(
        output_root / "gold.open.authorization.private.json",
        block=BLOCK_A_FORM,
        archive=archive,
        gold_binding=gold_binding,
        evaluator_model_self_sha256=None,
    )
    gold = _load_gold(
        gold_path,
        block=BLOCK_A_FORM,
        binding=gold_binding,
        archive=archive,
    )
    training: list[core.AFormSlate] = []
    detail_contracts: list[dict[str, object]] = []
    unanswerable_count = 0
    for contract in archive.contracts:
        detail_items: list[dict[str, object]] = []
        for item in contract.items:
            intervals = gold[item.work_id]
            utilities = tuple(
                _utility(
                    contract.passages,
                    action.passage_ordinals,
                    intervals,
                )
                for action in item.slate.actions
            )
            if not intervals:
                if any(value is not None for value in utilities):
                    raise MaudExtractionP1FormalControllerError(
                        "unanswerable A_form utility became defined"
                    )
                unanswerable_count += 1
                training_utilities = tuple(
                    0 for _ in core.RECIPE_IDS
                )
            else:
                if any(value is None for value in utilities):
                    raise MaudExtractionP1FormalControllerError(
                        "answerable A_form utility became undefined"
                    )
                training_utilities = tuple(
                    int(value) for value in utilities
                )
            # Training-only handling is frozen: unanswerable items retain all
            # nine rows with utility zero, hence zero delta from E0.  They are
            # still absent from retrieval/contract/family primary metrics.
            training.append(
                core.AFormSlate(
                    family=item.family,
                    slate=item.slate,
                    recipe_utilities=training_utilities,
                )
            )
            detail_items.append(
                {
                    "work_id": item.work_id,
                    "family": item.family,
                    "recipe_utilities": list(training_utilities),
                    "primary_retrieval_utility_defined": bool(
                        intervals
                    ),
                }
            )
        detail_contracts.append(
            {
                "contract_work_id": contract.contract_work_id,
                "items": detail_items,
            }
        )
    model = core.fit_e1_ridge(tuple(training))
    model_path = output_root / "E1.model.private.json"
    model_envelope, model_binding = write_e1_model(
        model_path,
        model=model,
        a_form_action_archive_file_sha256=archive.file_sha256,
        a_form_gold_file_sha256=gold_binding.file_sha256,
    )
    score_result = _score_archive_and_receipt(
        block=BLOCK_A_FORM,
        output_root=output_root,
        archive=archive,
        gold_binding=gold_binding,
        detail={
            "contracts": detail_contracts,
            "training_item_count": len(training),
            "answerable_training_item_count": (
                len(training) - unanswerable_count
            ),
            "unanswerable_item_count": unanswerable_count,
        },
        safe_aggregates={
            "contract_count": len(archive.contracts),
            "training_item_count": len(training),
            "answerable_training_item_count": (
                len(training) - unanswerable_count
            ),
            "unanswerable_item_count": unanswerable_count,
            "training_row_count": model.training_row_count,
            "model_sha256": model.model_sha256,
            "refit_count": 0,
        },
    )
    model_result = ModelResult(
        model_path=model_path,
        model_file_sha256=str(model_binding["file_sha256"]),
        model_self_sha256=str(model_envelope["self_sha256"]),
        model_sha256=model.model_sha256,
        receipt_path=score_result.receipt_path,
    )
    return model_result, score_result


def _comparison_payload(
    comparison: core.ContractClusterComparison,
) -> dict[str, object]:
    return {
        "left_arm": comparison.left_arm,
        "right_arm": comparison.right_arm,
        "contract_count": len(comparison.paired_contract_deltas),
        "nonzero_contract_count": (
            comparison.sign_flip.nonzero_contract_count
        ),
        "zero_answerable_contract_count": (
            comparison.zero_answerable_contract_count
        ),
        "equal_weight_contract_mean_delta": _fraction_payload(
            comparison.equal_weight_contract_mean_delta
        ),
        "observed_contract_net": _fraction_payload(
            comparison.sign_flip.observed_net
        ),
        "exact_sign_flip_reference_tail": _fraction_payload(
            comparison.sign_flip.reference_tail
        ),
        "family_deltas": {
            family: _fraction_payload(value)
            for family, value in comparison.family_deltas.items()
        },
    }


def _real_legal_success(
    comparison: core.ContractClusterComparison,
) -> bool:
    return (
        comparison.equal_weight_contract_mean_delta > 0
        and comparison.sign_flip.reference_tail <= core.PROMOTION_ALPHA
        and set(comparison.family_deltas) == set(core.QUERY_FAMILIES)
        and all(
            comparison.family_deltas[family] > 0
            for family in core.QUERY_FAMILIES
        )
    )


def _score_cluster_block(
    *,
    archive: LoadedActionArchive,
    gold: Mapping[str, tuple[core.CharacterInterval, ...]],
    include_baselines: bool,
) -> tuple[
    tuple[core.ContractCluster, ...],
    list[dict[str, object]],
    dict[str, object],
]:
    clusters: list[core.ContractCluster] = []
    detail_contracts: list[dict[str, object]] = []
    arms = [ROLE_AGENT_E0, ROLE_AGENT_E1]
    if include_baselines:
        arms.extend((ROLE_RAW, ROLE_HIPPORAG))
    secondary: dict[str, dict[str, int]] = {
        arm: {
            "answerable_item_count": 0,
            "complete_at_5_count": 0,
            "coverage_at_least_half_count": 0,
            "rank_discounted_incremental_utility_sum": 0,
        }
        for arm in arms
    }
    for contract in archive.contracts:
        cluster_items: list[core.ClusterItem] = []
        detail_items: list[dict[str, object]] = []
        for item in contract.items:
            if item.e1_registry_ordinal is None:
                raise MaudExtractionP1FormalControllerError(
                    "post-A_form archive lacks E1 selection"
                )
            intervals = gold[item.work_id]
            scores: dict[str, core.CoverageScore] = {
                ROLE_AGENT_E0: _coverage_score(
                    contract.passages,
                    item.slate.actions[
                        item.e0_registry_ordinal
                    ].passage_ordinals,
                    intervals,
                ),
                ROLE_AGENT_E1: _coverage_score(
                    contract.passages,
                    item.slate.actions[
                        item.e1_registry_ordinal
                    ].passage_ordinals,
                    intervals,
                ),
            }
            if include_baselines:
                if item.hippo_top5 is None:
                    raise MaudExtractionP1FormalControllerError(
                        "A_hold archive lacks official HippoRAG"
                    )
                scores[ROLE_RAW] = _coverage_score(
                    contract.passages, item.raw_top5, intervals
                )
                scores[ROLE_HIPPORAG] = _coverage_score(
                    contract.passages, item.hippo_top5, intervals
                )
            utilities = {
                arm: score.primary_utility
                for arm, score in scores.items()
            }
            for arm, score in scores.items():
                if not score.answerable:
                    continue
                row = secondary[arm]
                row["answerable_item_count"] += 1
                row["complete_at_5_count"] += int(
                    score.complete_at_5
                )
                row["coverage_at_least_half_count"] += int(
                    score.coverage_at_least_half
                )
                row[
                    "rank_discounted_incremental_utility_sum"
                ] += int(
                    score.rank_discounted_incremental_utility
                )
            cluster_items.append(
                core.ClusterItem(
                    family=item.family, arm_utilities=utilities
                )
            )
            detail_items.append(
                {
                    "work_id": item.work_id,
                    "family": item.family,
                    "arm_utilities": utilities,
                    "arm_scores": {
                        arm: _coverage_payload(score)
                        for arm, score in scores.items()
                    },
                }
            )
        clusters.append(core.ContractCluster(tuple(cluster_items)))
        detail_contracts.append(
            {
                "contract_work_id": contract.contract_work_id,
                "items": detail_items,
            }
        )
    safe_secondary: dict[str, object] = {}
    for arm, row in secondary.items():
        count = row["answerable_item_count"]
        safe_secondary[arm] = {
            "answerable_item_count": count,
            "complete_at_5_count": row["complete_at_5_count"],
            "coverage_at_least_half_count": row[
                "coverage_at_least_half_count"
            ],
            "mean_rank_discounted_incremental_coverage": (
                None
                if count == 0
                else _fraction_payload(
                    Fraction(
                        row[
                            "rank_discounted_incremental_utility_sum"
                        ],
                        count * core.INTEGER_SCALE,
                    )
                )
            ),
        }
    return tuple(clusters), detail_contracts, safe_secondary


def score_a_hold(
    *,
    action_archive_path: Path,
    gold_path: Path,
    gold_binding: ArchiveBinding,
    e1_model_path: Path,
    initial_acquisition_receipt_self_sha256: str,
    source_custody_self_sha256: str,
    output_root: Path,
) -> tuple[dict[str, object], ScoreResult]:
    """Score A_hold and create the only TEST-opening capability."""

    _ensure_private_directory(output_root, fresh=True)
    model, _model_envelope, model_self_sha256 = load_e1_model(
        e1_model_path
    )
    archive = load_action_archive(
        action_archive_path,
        expected_block=BLOCK_A_HOLD,
        e1_model=model,
        expected_model_self_sha256=model_self_sha256,
    )
    if (
        archive.acquisition_receipt_self_sha256
        != _validate_sha256(
            initial_acquisition_receipt_self_sha256,
            "initial acquisition receipt",
        )
    ):
        raise MaudExtractionP1FormalControllerError(
            "A_hold initial acquisition binding drifted"
        )
    if (
        _validate_sha256(
            source_custody_self_sha256, "source custody"
        )
        != SOURCE_CUSTODY_SELF_SHA256
    ):
        raise MaudExtractionP1FormalControllerError(
            "source custody commitment drifted"
        )
    _gold_authorization(
        output_root / "gold.open.authorization.private.json",
        block=BLOCK_A_HOLD,
        archive=archive,
        gold_binding=gold_binding,
        evaluator_model_self_sha256=model_self_sha256,
    )
    gold = _load_gold(
        gold_path,
        block=BLOCK_A_HOLD,
        binding=gold_binding,
        archive=archive,
    )
    clusters, detail, secondary = _score_cluster_block(
        archive=archive, gold=gold, include_baselines=True
    )
    promotion_comparison = core.compare_contract_clusters(
        clusters,
        left_arm=ROLE_AGENT_E1,
        right_arm=ROLE_AGENT_E0,
    )
    raw_comparison = core.compare_contract_clusters(
        clusters,
        left_arm=ROLE_AGENT_E1,
        right_arm=ROLE_RAW,
    )
    hippo_comparison = core.compare_contract_clusters(
        clusters,
        left_arm=ROLE_AGENT_E1,
        right_arm=ROLE_HIPPORAG,
    )
    promoted = (
        promotion_comparison.equal_weight_contract_mean_delta > 0
        and promotion_comparison.sign_flip.reference_tail
        <= core.PROMOTION_ALPHA
    )
    reality_passed = _real_legal_success(
        raw_comparison
    ) and _real_legal_success(hippo_comparison)
    score_result = _score_archive_and_receipt(
        block=BLOCK_A_HOLD,
        output_root=output_root,
        archive=archive,
        gold_binding=gold_binding,
        detail={"contracts": detail},
        safe_aggregates={
            "E1_minus_E0": _comparison_payload(
                promotion_comparison
            ),
            "E1_minus_RAW": _comparison_payload(raw_comparison),
            "E1_minus_official_HippoRAG": _comparison_payload(
                hippo_comparison
            ),
            "secondary_metrics_non_gate": secondary,
            "promoted": promoted,
            "reality_primary_passed": reality_passed,
        },
    )
    promotion_body = {
        "schema": PROMOTION_SCHEMA,
        "study_id": STUDY_ID,
        "study_design_self_sha256": STUDY_DESIGN_SELF_SHA256,
        "source_custody_self_sha256": SOURCE_CUSTODY_SELF_SHA256,
        "initial_acquisition_receipt_self_sha256": (
            initial_acquisition_receipt_self_sha256
        ),
        "A_hold_action_archive_file_sha256": archive.file_sha256,
        "A_hold_action_archive_semantic_sha256": archive.semantic_sha256,
        "A_hold_gold_file_sha256": gold_binding.file_sha256,
        "A_hold_gold_semantic_sha256": gold_binding.semantic_sha256,
        "incumbent_evaluator_id": "E0_FIXED_GENERAL_COVERAGE",
        "challenger_evaluator_id": "E1_AFORM_CENTERED_RIDGE_L2_1",
        "challenger_model_sha256": model.model_sha256,
        "challenger_model_self_sha256": model_self_sha256,
        # Exact-set capability projection; acquisition independently
        # recomputes promoted from this complete safe comparison.
        "E1_minus_E0_comparison": {
            "contract_count": len(
                promotion_comparison.paired_contract_deltas
            ),
            "nonzero_contract_count": (
                promotion_comparison.sign_flip.nonzero_contract_count
            ),
            "net": _fraction_payload(
                promotion_comparison.equal_weight_contract_mean_delta
            ),
            "exact_sign_flip_reference_tail": _fraction_payload(
                promotion_comparison.sign_flip.reference_tail
            ),
        },
        "promoted": promoted,
        "M_search_authorized": promoted,
        "promotion_rule": (
            "net_strictly_positive_and_complete_contract_sign_flip_"
            "reference_tail_at_most_1_over_10"
        ),
        "retry_replay_resample_refit_or_gate_change_count": 0,
        "online_evaluator_API_or_fine_tune_count": 0,
    }
    promotion_receipt = self_hashed(promotion_body)
    promotion_path = output_root / "A_hold.promotion.receipt.json"
    write_private_json_once(promotion_path, promotion_receipt)
    return promotion_receipt, score_result


def validate_promotion_receipt(
    value: Mapping[str, object],
    *,
    expected_initial_receipt_self_sha256: str,
    expected_action_archive_file_sha256: str,
    expected_action_archive_semantic_sha256: str,
    expected_gold_file_sha256: str,
    expected_gold_semantic_sha256: str,
    expected_model_sha256: str,
    expected_model_self_sha256: str,
    require_promoted: bool,
) -> str:
    expected_keys = {
        "A_hold_action_archive_file_sha256",
        "A_hold_action_archive_semantic_sha256",
        "A_hold_gold_file_sha256",
        "A_hold_gold_semantic_sha256",
        "E1_minus_E0_comparison",
        "M_search_authorized",
        "challenger_evaluator_id",
        "challenger_model_self_sha256",
        "challenger_model_sha256",
        "incumbent_evaluator_id",
        "initial_acquisition_receipt_self_sha256",
        "online_evaluator_API_or_fine_tune_count",
        "promoted",
        "promotion_rule",
        "retry_replay_resample_refit_or_gate_change_count",
        "schema",
        "self_sha256",
        "source_custody_self_sha256",
        "study_design_self_sha256",
        "study_id",
    }
    digest = verify_self_hash(value)
    comparison = value.get("E1_minus_E0_comparison")
    if (
        set(value) != expected_keys
        or value.get("schema") != PROMOTION_SCHEMA
        or value.get("study_id") != STUDY_ID
        or value.get("study_design_self_sha256")
        != STUDY_DESIGN_SELF_SHA256
        or value.get("source_custody_self_sha256")
        != SOURCE_CUSTODY_SELF_SHA256
        or value.get("initial_acquisition_receipt_self_sha256")
        != expected_initial_receipt_self_sha256
        or value.get("A_hold_action_archive_file_sha256")
        != expected_action_archive_file_sha256
        or value.get("A_hold_action_archive_semantic_sha256")
        != expected_action_archive_semantic_sha256
        or value.get("A_hold_gold_file_sha256")
        != expected_gold_file_sha256
        or value.get("A_hold_gold_semantic_sha256")
        != expected_gold_semantic_sha256
        or value.get("incumbent_evaluator_id")
        != "E0_FIXED_GENERAL_COVERAGE"
        or value.get("challenger_evaluator_id")
        != "E1_AFORM_CENTERED_RIDGE_L2_1"
        or value.get("challenger_model_sha256")
        != expected_model_sha256
        or value.get("challenger_model_self_sha256")
        != expected_model_self_sha256
        or value.get(
            "retry_replay_resample_refit_or_gate_change_count"
        )
        != 0
        or value.get("online_evaluator_API_or_fine_tune_count") != 0
        or not isinstance(comparison, Mapping)
        or set(comparison)
        != {
            "contract_count",
            "exact_sign_flip_reference_tail",
            "net",
            "nonzero_contract_count",
        }
        or type(comparison.get("contract_count")) is not int
        or int(comparison["contract_count"]) <= 0
        or type(comparison.get("nonzero_contract_count")) is not int
        or not 0
        <= int(comparison["nonzero_contract_count"])
        <= int(comparison["contract_count"])
    ):
        raise MaudExtractionP1FormalControllerError(
            "A_hold promotion receipt exact schema drifted"
        )
    net = _fraction_from_payload(comparison["net"])
    tail = _fraction_from_payload(
        comparison["exact_sign_flip_reference_tail"]
    )
    derived = net > 0 and tail <= core.PROMOTION_ALPHA
    if (
        value.get("promoted") is not derived
        or value.get("M_search_authorized") is not derived
        or (require_promoted and not derived)
    ):
        raise MaudExtractionP1FormalControllerError(
            "A_hold promotion boolean is not derived from the frozen rule"
        )
    return digest


def score_m_search(
    *,
    action_archive_path: Path,
    gold_path: Path,
    gold_binding: ArchiveBinding,
    e1_model_path: Path,
    promotion_receipt: Mapping[str, object],
    output_root: Path,
) -> ScoreResult:
    """Score untouched M once with frozen E1 versus E0; never refit."""

    _ensure_private_directory(output_root, fresh=True)
    model, model_envelope, model_self_sha256 = load_e1_model(
        e1_model_path
    )
    archive = load_action_archive(
        action_archive_path,
        expected_block=BLOCK_M_SEARCH,
        e1_model=model,
        expected_model_self_sha256=model_self_sha256,
    )
    validate_promotion_receipt(
        promotion_receipt,
        expected_initial_receipt_self_sha256=str(
            promotion_receipt[
                "initial_acquisition_receipt_self_sha256"
            ]
        ),
        expected_action_archive_file_sha256=str(
            promotion_receipt[
                "A_hold_action_archive_file_sha256"
            ]
        ),
        expected_action_archive_semantic_sha256=str(
            promotion_receipt[
                "A_hold_action_archive_semantic_sha256"
            ]
        ),
        expected_gold_file_sha256=str(
            promotion_receipt["A_hold_gold_file_sha256"]
        ),
        expected_gold_semantic_sha256=str(
            promotion_receipt["A_hold_gold_semantic_sha256"]
        ),
        expected_model_sha256=model.model_sha256,
        expected_model_self_sha256=model_self_sha256,
        require_promoted=True,
    )
    _gold_authorization(
        output_root / "gold.open.authorization.private.json",
        block=BLOCK_M_SEARCH,
        archive=archive,
        gold_binding=gold_binding,
        evaluator_model_self_sha256=model_self_sha256,
    )
    gold = _load_gold(
        gold_path,
        block=BLOCK_M_SEARCH,
        binding=gold_binding,
        archive=archive,
    )
    clusters, detail, secondary = _score_cluster_block(
        archive=archive, gold=gold, include_baselines=False
    )
    comparison = core.compare_contract_clusters(
        clusters,
        left_arm=ROLE_AGENT_E1,
        right_arm=ROLE_AGENT_E0,
    )
    improved = (
        comparison.equal_weight_contract_mean_delta > 0
        and comparison.sign_flip.reference_tail
        <= core.PROMOTION_ALPHA
    )
    return _score_archive_and_receipt(
        block=BLOCK_M_SEARCH,
        output_root=output_root,
        archive=archive,
        gold_binding=gold_binding,
        detail={
            "contracts": detail,
            "frozen_model_self_sha256": model_self_sha256,
            "frozen_model_refit_count": model_envelope["refit_count"],
        },
        safe_aggregates={
            "E1_minus_E0": _comparison_payload(comparison),
            "secondary_metrics_non_gate": secondary,
            "L5_search_improved": improved,
            "model_sha256": model.model_sha256,
            "model_self_sha256": model_self_sha256,
            "refit_count": 0,
        },
    )


def _acquisition_environment(
    paths: runtime.RuntimePaths, *, private_root: Path
) -> dict[str, str]:
    private_directories = {
        "HOME": private_root / "home",
        "TMPDIR": private_root / "tmp",
    }
    for path in private_directories.values():
        _ensure_private_directory(path, fresh=True)
    return {
        "PATH": f"{Path(paths.typed_python).parent}:/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONPATH": paths.typed_pythonpath(),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "HOME": str(private_directories["HOME"]),
        "TMPDIR": str(private_directories["TMPDIR"]),
        "TMP": str(private_directories["TMPDIR"]),
        "TEMP": str(private_directories["TMPDIR"]),
    }


def initial_acquisition_command(
    *,
    paths: runtime.RuntimePaths,
    train_path: Path,
    dev_path: Path,
    secret_path: Path,
    download_receipt_path: Path,
    acquisition_root: Path,
) -> tuple[str, ...]:
    return (
        paths.typed_python,
        "-S",
        "-B",
        "-m",
        ACQUISITION_MODULE,
        "initial",
        "--train",
        str(_absolute_path(train_path, "TRAIN source")),
        "--dev",
        str(_absolute_path(dev_path, "DEV source")),
        "--secret",
        str(_absolute_path(secret_path, "selection secret")),
        "--download-receipt",
        str(
            _absolute_path(
                download_receipt_path, "download receipt"
            )
        ),
        "--output-root",
        str(_absolute_path(acquisition_root, "acquisition root")),
    )


def test_acquisition_command(
    *,
    paths: runtime.RuntimePaths,
    test_path: Path,
    secret_path: Path,
    download_receipt_path: Path,
    promotion_receipt_path: Path,
    acquisition_root: Path,
) -> tuple[str, ...]:
    return (
        paths.typed_python,
        "-S",
        "-B",
        "-m",
        ACQUISITION_MODULE,
        "test",
        "--test",
        str(_absolute_path(test_path, "TEST source")),
        "--secret",
        str(_absolute_path(secret_path, "selection secret")),
        "--download-receipt",
        str(
            _absolute_path(
                download_receipt_path, "download receipt"
            )
        ),
        "--promotion-receipt",
        str(
            _absolute_path(
                promotion_receipt_path, "promotion receipt"
            )
        ),
        "--output-root",
        str(_absolute_path(acquisition_root, "acquisition root")),
    )


def run_initial_acquisition_process(
    *,
    paths: runtime.RuntimePaths,
    train_path: Path,
    dev_path: Path,
    secret_path: Path,
    download_receipt_path: Path,
    acquisition_root: Path,
    process_root: Path,
    runner: Callable[..., object] = subprocess.run,
) -> tuple[dict[str, object], str]:
    """Launch trusted TRAIN/DEV parsing in one separate process exactly once."""

    if acquisition_root.exists() or acquisition_root.is_symlink():
        raise MaudExtractionP1FormalControllerError(
            "initial acquisition root is already consumed"
        )
    _ensure_private_directory(process_root, fresh=True)
    command = initial_acquisition_command(
        paths=paths,
        train_path=train_path,
        dev_path=dev_path,
        secret_path=secret_path,
        download_receipt_path=download_receipt_path,
        acquisition_root=acquisition_root,
    )
    _run_logged_subprocess(
        command=command,
        cwd=Path(paths.deployed_project_root),
        environment=_acquisition_environment(
            paths, private_root=process_root
        ),
        stdout_path=process_root / "stdout.private.bin",
        stderr_path=process_root / "stderr.private.bin",
        timeout_seconds=ACQUISITION_TIMEOUT_SECONDS,
        runner=runner,
    )
    receipt = read_canonical_private(
        acquisition_root / "initial.receipt.json"
    )
    return receipt, _validate_acquisition_receipt(
        receipt, phase="initial"
    )


def run_test_acquisition_process(
    *,
    paths: runtime.RuntimePaths,
    test_path: Path,
    secret_path: Path,
    download_receipt_path: Path,
    acquisition_root: Path,
    promotion_receipt_path: Path,
    process_root: Path,
    expected_initial_receipt_self_sha256: str,
    expected_action_archive_file_sha256: str,
    expected_action_archive_semantic_sha256: str,
    expected_gold_file_sha256: str,
    expected_gold_semantic_sha256: str,
    expected_model_sha256: str,
    expected_model_self_sha256: str,
    runner: Callable[..., object] = subprocess.run,
) -> tuple[dict[str, object], str]:
    """Unlock TEST only after independently revalidating the exact receipt."""

    if not acquisition_root.is_dir() or acquisition_root.is_symlink():
        raise MaudExtractionP1FormalControllerError(
            "initial acquisition root is unavailable"
        )
    promotion = read_canonical_private(promotion_receipt_path)
    promotion_sha256 = validate_promotion_receipt(
        promotion,
        expected_initial_receipt_self_sha256=(
            expected_initial_receipt_self_sha256
        ),
        expected_action_archive_file_sha256=(
            expected_action_archive_file_sha256
        ),
        expected_action_archive_semantic_sha256=(
            expected_action_archive_semantic_sha256
        ),
        expected_gold_file_sha256=expected_gold_file_sha256,
        expected_gold_semantic_sha256=expected_gold_semantic_sha256,
        expected_model_sha256=expected_model_sha256,
        expected_model_self_sha256=expected_model_self_sha256,
        require_promoted=True,
    )
    _ensure_private_directory(process_root, fresh=True)
    command = test_acquisition_command(
        paths=paths,
        test_path=test_path,
        secret_path=secret_path,
        download_receipt_path=download_receipt_path,
        promotion_receipt_path=promotion_receipt_path,
        acquisition_root=acquisition_root,
    )
    _run_logged_subprocess(
        command=command,
        cwd=Path(paths.deployed_project_root),
        environment=_acquisition_environment(
            paths, private_root=process_root
        ),
        stdout_path=process_root / "stdout.private.bin",
        stderr_path=process_root / "stderr.private.bin",
        timeout_seconds=ACQUISITION_TIMEOUT_SECONDS,
        runner=runner,
    )
    receipt = read_canonical_private(
        acquisition_root / "test_parse.receipt.json"
    )
    receipt_sha256 = _validate_acquisition_receipt(
        receipt, phase="test"
    )
    if receipt.get("a_hold_promotion_receipt_sha256") != promotion_sha256:
        raise MaudExtractionP1FormalControllerError(
            "TEST acquisition escaped promotion binding"
        )
    return receipt, receipt_sha256


def _validate_runtime_fingerprint(
    path: Path,
    paths: runtime.RuntimePaths,
    *,
    expected_self_sha256: str,
) -> str:
    expected = _validate_sha256(
        expected_self_sha256,
        "execution-freeze runtime fingerprint self hash",
    )
    value = read_canonical_private(path)
    digest = verify_self_hash(value)
    if (
        digest != expected
        or value.get("schema") != runtime.RUNTIME_FINGERPRINT_SCHEMA
        or value.get("study_id") != STUDY_ID
        or value.get("study_design_self_sha256")
        != STUDY_DESIGN_SELF_SHA256
        or value.get("pre_source_clarification_self_sha256")
        != PRE_SOURCE_CLARIFICATION_SELF_SHA256
        or value.get("status")
        != "verified_source_free_post_reboot_runtime_fingerprint"
        or value.get("path_commitments") != paths.path_commitments()
    ):
        raise MaudExtractionP1FormalControllerError(
            "source-free runtime fingerprint drifted"
        )
    return digest


def _validate_full_source_free_canary_receipt(
    path: Path,
    *,
    expected_self_sha256: str,
    expected_runtime_fingerprint_self_sha256: str,
) -> dict[str, object]:
    """Read the one pre-formal canary receipt pinned by execution freeze."""

    expected = _validate_sha256(
        expected_self_sha256,
        "execution-freeze full canary self hash",
    )
    expected_fingerprint = _validate_sha256(
        expected_runtime_fingerprint_self_sha256,
        "execution-freeze runtime fingerprint self hash",
    )
    value = read_canonical_private(path)
    digest = verify_self_hash(value)
    hash_fields = (
        "coordinate_fixture_sha256",
        "minilm_output_self_sha256",
        "cross_encoder_output_self_sha256",
        "official_output_sha256",
        "official_safe_terminal_sha256",
        "typed_recipe_registry_sha256",
        "E0_behavior_set_sha256",
    )
    expected_shape = {
        "coordinate_contract_count": 1,
        "coordinate_query_count": 22,
        "coordinate_worker_count": 2,
        "official_contract_count": 1,
        "typed_recipe_count_per_query": len(core.RECIPE_IDS),
        "E0_selection_count": 22,
    }
    expected_execution = {
        "coordinate_workers_bulk_submitted_before_join": True,
        "cross_encoder_physical_gpu": "1",
        "minilm_physical_gpu": "0",
        "official_physical_gpu": "0",
        "retry_replay_resample_count": 0,
        "api_or_online_evaluator_call_count": 0,
        "formal_source_action_or_score_count": 0,
    }
    if (
        digest != expected
        or value.get("schema") != FULL_CANARY_SCHEMA
        or value.get("study_id") != STUDY_ID
        or value.get("study_design_self_sha256")
        != STUDY_DESIGN_SELF_SHA256
        or value.get("pre_source_clarification_self_sha256")
        != PRE_SOURCE_CLARIFICATION_SELF_SHA256
        or value.get("status")
        != "passed_source_free_coordinate_pair_and_official_hipporag"
        or value.get("runtime_fingerprint_sha256")
        != expected_fingerprint
        or value.get("shape") != expected_shape
        or value.get("execution") != expected_execution
        or any(
            not isinstance(value.get(field), str)
            or _HEX64.fullmatch(str(value[field])) is None
            for field in hash_fields
        )
    ):
        raise MaudExtractionP1FormalControllerError(
            "source-free full canary escaped execution-freeze binding"
        )
    return value


@dataclass(frozen=True)
class FormalExecutionConfig:
    train_path: Path
    dev_path: Path
    test_path: Path
    secret_path: Path
    download_receipt_path: Path
    acquisition_root: Path
    control_root: Path
    runtime_fingerprint_path: Path
    full_canary_receipt_path: Path
    expected_runtime_fingerprint_self_sha256: str
    expected_full_canary_self_sha256: str
    runtime_paths: runtime.RuntimePaths

    def __post_init__(self) -> None:
        for field in (
            "train_path",
            "dev_path",
            "test_path",
            "secret_path",
            "download_receipt_path",
            "acquisition_root",
            "control_root",
            "runtime_fingerprint_path",
            "full_canary_receipt_path",
        ):
            object.__setattr__(
                self,
                field,
                _absolute_path(getattr(self, field), field),
            )
        if not isinstance(self.runtime_paths, runtime.RuntimePaths):
            raise MaudExtractionP1FormalControllerError(
                "formal runtime paths drifted"
            )
        object.__setattr__(
            self,
            "expected_runtime_fingerprint_self_sha256",
            _validate_sha256(
                self.expected_runtime_fingerprint_self_sha256,
                "execution-freeze runtime fingerprint self hash",
            ),
        )
        object.__setattr__(
            self,
            "expected_full_canary_self_sha256",
            _validate_sha256(
                self.expected_full_canary_self_sha256,
                "execution-freeze full canary self hash",
            ),
        )


def _config_from_private_file(path: Path) -> FormalExecutionConfig:
    value = read_canonical_private(path)
    verify_self_hash(value)
    expected_keys = {
        "acquisition_root",
        "control_root",
        "dev_path",
        "download_receipt_path",
        "expected_full_canary_self_sha256",
        "expected_runtime_fingerprint_self_sha256",
        "full_canary_receipt_path",
        "runtime_fingerprint_path",
        "runtime_paths",
        "schema",
        "secret_path",
        "self_sha256",
        "study_design_self_sha256",
        "study_id",
        "test_path",
        "train_path",
    }
    raw_runtime = value.get("runtime_paths")
    if (
        set(value) != expected_keys
        or value.get("schema") != CONFIG_SCHEMA
        or value.get("study_id") != STUDY_ID
        or value.get("study_design_self_sha256")
        != STUDY_DESIGN_SELF_SHA256
        or not isinstance(raw_runtime, Mapping)
        or set(raw_runtime)
        != set(runtime.RuntimePaths.__dataclass_fields__)
    ):
        raise MaudExtractionP1FormalControllerError(
            "formal execution config drifted"
        )
    return FormalExecutionConfig(
        train_path=Path(str(value["train_path"])),
        dev_path=Path(str(value["dev_path"])),
        test_path=Path(str(value["test_path"])),
        secret_path=Path(str(value["secret_path"])),
        download_receipt_path=Path(
            str(value["download_receipt_path"])
        ),
        acquisition_root=Path(str(value["acquisition_root"])),
        control_root=Path(str(value["control_root"])),
        runtime_fingerprint_path=Path(
            str(value["runtime_fingerprint_path"])
        ),
        full_canary_receipt_path=Path(
            str(value["full_canary_receipt_path"])
        ),
        expected_runtime_fingerprint_self_sha256=str(
            value["expected_runtime_fingerprint_self_sha256"]
        ),
        expected_full_canary_self_sha256=str(
            value["expected_full_canary_self_sha256"]
        ),
        runtime_paths=runtime.RuntimePaths(
            **{key: str(raw_runtime[key]) for key in raw_runtime}
        ),
    )


def _run_formal_study_once(
    config: FormalExecutionConfig,
    *,
    process_runner: Callable[..., object] = subprocess.run,
    coordinate_launcher: CoordinateBatchLauncher = run_coordinate_workers,
    hippo_batch_launcher: HippoBatchLauncher = runtime.run_contract_batch,
) -> dict[str, object]:
    """Execute the frozen lifecycle once; every negative result is terminal."""

    if not isinstance(config, FormalExecutionConfig):
        raise MaudExtractionP1FormalControllerError(
            "formal execution config is invalid"
        )
    fingerprint_sha256 = _validate_runtime_fingerprint(
        config.runtime_fingerprint_path,
        config.runtime_paths,
        expected_self_sha256=(
            config.expected_runtime_fingerprint_self_sha256
        ),
    )
    canary = _validate_full_source_free_canary_receipt(
        config.full_canary_receipt_path,
        expected_self_sha256=config.expected_full_canary_self_sha256,
        expected_runtime_fingerprint_self_sha256=fingerprint_sha256,
    )
    _ensure_private_directory(config.control_root, fresh=True)
    initial_receipt, initial_receipt_sha256 = (
        run_initial_acquisition_process(
            paths=config.runtime_paths,
            train_path=config.train_path,
            dev_path=config.dev_path,
            secret_path=config.secret_path,
            download_receipt_path=config.download_receipt_path,
            acquisition_root=config.acquisition_root,
            process_root=config.control_root
            / "initial_acquisition_process",
            runner=process_runner,
        )
    )

    a_form_action_path, a_form_action_binding, _ = (
        load_acquisition_archive(
            config.acquisition_root,
            block=BLOCK_A_FORM,
            role="action",
        )
    )
    a_form_stage = run_action_stage(
        block=BLOCK_A_FORM,
        action_view_path=a_form_action_path,
        action_view_binding=a_form_action_binding,
        acquisition_receipt_self_sha256=initial_receipt_sha256,
        stage_root=config.control_root / "A_form_action",
        runtime_paths=config.runtime_paths,
        coordinate_launcher=coordinate_launcher,
        hippo_batch_launcher=hippo_batch_launcher,
    )
    a_form_gold_path, a_form_gold_binding, _ = (
        load_acquisition_archive(
            config.acquisition_root,
            block=BLOCK_A_FORM,
            role="gold",
        )
    )
    model_result, _a_form_score = score_a_form(
        action_archive_path=a_form_stage.archive_path,
        gold_path=a_form_gold_path,
        gold_binding=a_form_gold_binding,
        output_root=config.control_root / "A_form_score",
    )

    f_action_path, f_action_binding, _ = load_acquisition_archive(
        config.acquisition_root,
        block=BLOCK_F_SEARCH,
        role="action",
    )
    f_stage = run_action_stage(
        block=BLOCK_F_SEARCH,
        action_view_path=f_action_path,
        action_view_binding=f_action_binding,
        acquisition_receipt_self_sha256=initial_receipt_sha256,
        stage_root=config.control_root / "F_search_action",
        runtime_paths=config.runtime_paths,
        e1_model_path=model_result.model_path,
        coordinate_launcher=coordinate_launcher,
        hippo_batch_launcher=hippo_batch_launcher,
    )

    hold_action_path, hold_action_binding, _ = (
        load_acquisition_archive(
            config.acquisition_root,
            block=BLOCK_A_HOLD,
            role="action",
        )
    )
    hold_stage = run_action_stage(
        block=BLOCK_A_HOLD,
        action_view_path=hold_action_path,
        action_view_binding=hold_action_binding,
        acquisition_receipt_self_sha256=initial_receipt_sha256,
        stage_root=config.control_root / "A_hold_action",
        runtime_paths=config.runtime_paths,
        e1_model_path=model_result.model_path,
        coordinate_launcher=coordinate_launcher,
        hippo_batch_launcher=hippo_batch_launcher,
    )
    hold_gold_path, hold_gold_binding, _ = load_acquisition_archive(
        config.acquisition_root,
        block=BLOCK_A_HOLD,
        role="gold",
    )
    promotion, hold_score = score_a_hold(
        action_archive_path=hold_stage.archive_path,
        gold_path=hold_gold_path,
        gold_binding=hold_gold_binding,
        e1_model_path=model_result.model_path,
        initial_acquisition_receipt_self_sha256=(
            initial_receipt_sha256
        ),
        source_custody_self_sha256=SOURCE_CUSTODY_SELF_SHA256,
        output_root=config.control_root / "A_hold_score",
    )
    promotion_path = (
        config.control_root
        / "A_hold_score"
        / "A_hold.promotion.receipt.json"
    )
    if promotion["promoted"] is not True:
        terminal = self_hashed(
            {
                "schema": TERMINAL_SCHEMA,
                "study_id": STUDY_ID,
                "study_design_self_sha256": (
                    STUDY_DESIGN_SELF_SHA256
                ),
                "pre_source_clarification_self_sha256": (
                    PRE_SOURCE_CLARIFICATION_SELF_SHA256
                ),
                "status": "valid_A_hold_nonpromotion_TEST_remains_closed",
                "runtime_fingerprint_sha256": fingerprint_sha256,
                "full_canary_self_sha256": canary["self_sha256"],
                "initial_acquisition_receipt_self_sha256": (
                    initial_receipt_sha256
                ),
                "A_form_model_sha256": model_result.model_sha256,
                "F_search_action_archive_self_sha256": (
                    f_stage.archive_self_sha256
                ),
                "A_hold_promotion_receipt_self_sha256": (
                    promotion["self_sha256"]
                ),
                "A_hold_score_receipt_self_sha256": (
                    hold_score.receipt["self_sha256"]
                ),
                "promoted": False,
                "M_search_authorized": False,
                "TEST_source_opened_or_parsed": False,
                "retry_replay_resample_refit_or_gate_change_count": 0,
                "raw_contract_item_action_gold_or_score_included": False,
            }
        )
        write_private_json_once(
            config.control_root / "formal.terminal.json", terminal
        )
        return terminal

    run_test_acquisition_process(
        paths=config.runtime_paths,
        test_path=config.test_path,
        secret_path=config.secret_path,
        download_receipt_path=config.download_receipt_path,
        acquisition_root=config.acquisition_root,
        promotion_receipt_path=promotion_path,
        process_root=config.control_root / "test_acquisition_process",
        expected_initial_receipt_self_sha256=initial_receipt_sha256,
        expected_action_archive_file_sha256=hold_stage.archive_file_sha256,
        expected_action_archive_semantic_sha256=(
            hold_stage.archive_semantic_sha256
        ),
        expected_gold_file_sha256=hold_gold_binding.file_sha256,
        expected_gold_semantic_sha256=(
            hold_gold_binding.semantic_sha256
        ),
        expected_model_sha256=model_result.model_sha256,
        expected_model_self_sha256=model_result.model_self_sha256,
        runner=process_runner,
    )
    m_action_path, m_action_binding, test_receipt_sha256 = (
        load_acquisition_archive(
            config.acquisition_root,
            block=BLOCK_M_SEARCH,
            role="action",
        )
    )
    m_stage = run_action_stage(
        block=BLOCK_M_SEARCH,
        action_view_path=m_action_path,
        action_view_binding=m_action_binding,
        acquisition_receipt_self_sha256=test_receipt_sha256,
        stage_root=config.control_root / "M_search_action",
        runtime_paths=config.runtime_paths,
        e1_model_path=model_result.model_path,
        coordinate_launcher=coordinate_launcher,
        hippo_batch_launcher=hippo_batch_launcher,
    )
    m_gold_path, m_gold_binding, _ = load_acquisition_archive(
        config.acquisition_root,
        block=BLOCK_M_SEARCH,
        role="gold",
    )
    m_score = score_m_search(
        action_archive_path=m_stage.archive_path,
        gold_path=m_gold_path,
        gold_binding=m_gold_binding,
        e1_model_path=model_result.model_path,
        promotion_receipt=promotion,
        output_root=config.control_root / "M_search_score",
    )
    terminal = self_hashed(
        {
            "schema": TERMINAL_SCHEMA,
            "study_id": STUDY_ID,
            "study_design_self_sha256": STUDY_DESIGN_SELF_SHA256,
            "pre_source_clarification_self_sha256": (
                PRE_SOURCE_CLARIFICATION_SELF_SHA256
            ),
            "status": "formal_study_complete",
            "runtime_fingerprint_sha256": fingerprint_sha256,
            "full_canary_self_sha256": canary["self_sha256"],
            "initial_acquisition_receipt_self_sha256": (
                initial_receipt_sha256
            ),
            "A_form_model_sha256": model_result.model_sha256,
            "F_search_action_archive_self_sha256": (
                f_stage.archive_self_sha256
            ),
            "A_hold_promotion_receipt_self_sha256": (
                promotion["self_sha256"]
            ),
            "A_hold_score_receipt_self_sha256": (
                hold_score.receipt["self_sha256"]
            ),
            "M_search_action_archive_self_sha256": (
                m_stage.archive_self_sha256
            ),
            "M_search_score_receipt_self_sha256": (
                m_score.receipt["self_sha256"]
            ),
            "promoted": True,
            "M_search_authorized": True,
            "TEST_source_opened_or_parsed": True,
            "retry_replay_resample_refit_or_gate_change_count": 0,
            "raw_contract_item_action_gold_or_score_included": False,
        }
    )
    write_private_json_once(
        config.control_root / "formal.terminal.json", terminal
    )
    return terminal


def _formal_failure_phase(control_root: Path) -> str:
    """Infer only a coarse safe phase from private artifact names."""

    milestones = (
        ("M_search_score", "M_search_scoring"),
        ("M_search_action", "M_search_action"),
        ("test_acquisition_process", "TEST_acquisition"),
        ("A_hold_score", "A_hold_scoring"),
        ("A_hold_action", "A_hold_action"),
        ("F_search_action", "F_search_action"),
        ("A_form_score", "A_form_scoring"),
        ("A_form_action", "A_form_action"),
        ("initial_acquisition_process", "initial_acquisition"),
    )
    for name, phase in milestones:
        if os.path.lexists(control_root / name):
            return phase
    return "formal_control_initialized"


def _write_formal_failure_terminal_once(
    config: object,
    error: Exception,
) -> None:
    """Best-effort custody receipt that cannot mask the original failure."""

    if not isinstance(config, FormalExecutionConfig):
        return
    root = config.control_root
    terminal_path = root / "formal.terminal.json"
    failure_path = root / "formal.failure.terminal.json"
    if (
        not root.is_dir()
        or root.is_symlink()
        or os.path.lexists(terminal_path)
        or os.path.lexists(failure_path)
    ):
        return
    safe_phase = _formal_failure_phase(root)
    receipt = self_hashed(
        {
            "schema": f"{VERSION}_safe_failure_terminal_v1",
            "study_id": STUDY_ID,
            "study_design_self_sha256": STUDY_DESIGN_SELF_SHA256,
            "pre_source_clarification_self_sha256": (
                PRE_SOURCE_CLARIFICATION_SELF_SHA256
            ),
            "status": "terminal_failure_no_retry_replay_or_resample",
            "error_type": type(error).__name__,
            "safe_phase": safe_phase,
            "safe_phase_sha256": hashlib.sha256(
                safe_phase.encode("ascii")
            ).hexdigest(),
            "expected_runtime_fingerprint_self_sha256": (
                config.expected_runtime_fingerprint_self_sha256
            ),
            "expected_full_canary_self_sha256": (
                config.expected_full_canary_self_sha256
            ),
            "retry_replay_resample_refit_or_gate_change_count": 0,
            "online_evaluator_API_or_fine_tune_count": 0,
            "raw_contract_item_action_gold_score_or_error_message_included": (
                False
            ),
        }
    )
    try:
        write_private_json_once(failure_path, receipt)
    except Exception:
        return


def run_formal_study(
    config: FormalExecutionConfig,
    *,
    process_runner: Callable[..., object] = subprocess.run,
    coordinate_launcher: CoordinateBatchLauncher = run_coordinate_workers,
    hippo_batch_launcher: HippoBatchLauncher = runtime.run_contract_batch,
) -> dict[str, object]:
    """Run once and preserve a content-free terminal on in-run failure."""

    try:
        return _run_formal_study_once(
            config,
            process_runner=process_runner,
            coordinate_launcher=coordinate_launcher,
            hippo_batch_launcher=hippo_batch_launcher,
        )
    except Exception as exc:
        _write_formal_failure_terminal_once(config, exc)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    terminal = run_formal_study(
        _config_from_private_file(arguments.config)
    )
    print(
        json.dumps(
            terminal,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACQUISITION_MODULE",
    "ACTION_ARCHIVE_SCHEMA",
    "ArchiveBinding",
    "BLOCK_A_FORM",
    "BLOCK_A_HOLD",
    "BLOCK_F_SEARCH",
    "BLOCK_M_SEARCH",
    "CONFIG_SCHEMA",
    "CoordinateBatchResult",
    "FormalExecutionConfig",
    "FULL_CANARY_SCHEMA",
    "MaudExtractionP1FormalControllerError",
    "ModelResult",
    "PROMOTION_SCHEMA",
    "PRE_SOURCE_CLARIFICATION_SELF_SHA256",
    "SOURCE_CUSTODY_SELF_SHA256",
    "STUDY_DESIGN_SELF_SHA256",
    "STUDY_ID",
    "ScoreResult",
    "StageResult",
    "VERSION",
    "canonical_json_bytes",
    "initial_acquisition_command",
    "load_acquisition_archive",
    "load_action_archive",
    "load_e1_model",
    "main",
    "read_canonical_private",
    "run_action_stage",
    "run_coordinate_workers",
    "run_formal_study",
    "run_full_source_free_canary",
    "run_initial_acquisition_process",
    "run_test_acquisition_process",
    "score_a_form",
    "score_a_hold",
    "score_m_search",
    "self_hashed",
    "semantic_sha256",
    "synthetic_canary_fixture",
    "synthetic_coordinate_input",
    "test_acquisition_command",
    "validate_promotion_receipt",
    "verify_self_hash",
    "write_e1_model",
    "write_private_json_once",
]
