#!/usr/bin/env python3
"""Frozen, one-shot controller for the QuAC aggregate source qualifier.

Preflight validates the complete implementation, source metadata, interpreter,
environment, unit symlink, and historical bindings without opening either
QuAC payload.  Formal mode repeats that validation, writes and fsyncs an
immutable attempt receipt, and only then permits the qualifier to read the two
fixed source files.

The CLI exposes only one fixed freeze plus mutually exclusive preflight/formal
modes.  Bootstrap failures before an attempt write a fixed safe STOP.  Once an
attempt exists, all controlled failures write an aggregate-only result and
terminal; an existing attempt or terminal can never be replayed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import stat
import sys
from types import ModuleType
from typing import Any, Mapping, Sequence


VERSION = "run_quac_p1_source_qualification_v1"
FREEZE_SCHEMA = "quac_p1_source_qualification_freeze_v1"
FREEZE_FILENAME = "quac_p1_source_qualification_freeze_v1.json"
STUDY_ID = "QUAC_P1_RJMC_DIALOGUE_EVIDENCE_L5_V1"

STUDY_ROOT = Path("/home/erzhu419/quac_rjmc_20260728")
FORMAL_ROOT = STUDY_ROOT / "source_qualification_v1"
PROJECT_ROOT = FORMAL_ROOT / "reconstruction_v2"
SOURCE_ROOT = STUDY_ROOT / "official_source_v1"
TRAIN_SOURCE_PATH = SOURCE_ROOT / "train_v0.2.json"
DEV_SOURCE_PATH = SOURCE_ROOT / "val_v0.2.json"
WORK_ROOT = FORMAL_ROOT / "work"
FREEZE_PATH = PROJECT_ROOT / "manifests" / FREEZE_FILENAME
UNIT_NAME = "quac-p1-source-qualification-v1.service"
UNIT_SOURCE_PATH = (
    PROJECT_ROOT / "manifests/quac_p1_source_qualification_unit_v1.service"
)
INSTALLED_UNIT_PATH = (
    Path("/home/erzhu419/.config/systemd/user") / UNIT_NAME
)
FROZEN_PYTHON = Path("/usr/bin/python3")

ATTEMPT_NAME = "attempt.json"
RESULT_NAME = "source_qualification.result.safe.json"
TERMINAL_NAME = "source_qualification.terminal.safe.json"
CORE_RELATIVE_PATH = (
    "assumption_agent/benchmarks/quac_p1_source_qualification_v1.py"
)
RUNNER_RELATIVE_PATH = "scripts/run_quac_p1_source_qualification_v1.py"
UNIT_RELATIVE_PATH = (
    "manifests/quac_p1_source_qualification_unit_v1.service"
)
CORE_TEST_RELATIVE_PATH = "tests/test_quac_p1_source_qualification_v1.py"
RUNNER_TEST_RELATIVE_PATH = (
    "tests/test_run_quac_p1_source_qualification_v1.py"
)
ARCHITECTURE_RELATIVE_PATH = (
    "manifests/red_queen_poststop_rjmc_architecture_decision_v1.json"
)
CUSTODY_RELATIVE_PATH = "manifests/quac_p1_source_custody_v1.json"
SOURCE_FREE_RESULT_RELATIVE_PATH = (
    "manifests/quac_rjmc_source_free_qualification_result_v1.json"
)
SOURCE_FREE_FREEZE_RELATIVE_PATH = (
    "manifests/quac_rjmc_source_free_qualification_freeze_v1.json"
)
REQUIRED_FILE_RELATIVE_PATHS = (
    CORE_RELATIVE_PATH,
    RUNNER_RELATIVE_PATH,
    UNIT_RELATIVE_PATH,
    CORE_TEST_RELATIVE_PATH,
    RUNNER_TEST_RELATIVE_PATH,
    ARCHITECTURE_RELATIVE_PATH,
    CUSTODY_RELATIVE_PATH,
    SOURCE_FREE_RESULT_RELATIVE_PATH,
    SOURCE_FREE_FREEZE_RELATIVE_PATH,
)

ARCHITECTURE_DECISION_SELF_SHA256 = (
    "9efb416359c1efc315846523a67382b0b942a8a827976cece72175085fe79462"
)
SOURCE_CUSTODY_SELF_SHA256 = (
    "d098b6e7a14e0e7d77f6b59869a4e913a210e4d30bf8bb72f97addd89bba3c30"
)
SOURCE_FREE_RESULT_SELF_SHA256 = (
    "225ace75a6ea07372827670fa7872709de45f2fa4c8fbe2b60286adfac07a450"
)
SOURCE_FREE_RESULT_FILE_SHA256 = (
    "996de9122d96c66a569e7f05198d4d03645425484866686fb08bbe531d3c1eaf"
)
SOURCE_FREE_FREEZE_SELF_SHA256 = (
    "139e12bb8f7f21cf72f4c61da12a9ebde5bd217c53724ac5969526b9b03ca30b"
)

TRAIN_EXPECTED_SIZE_BYTES = 68_114_819
TRAIN_EXPECTED_SHA256 = (
    "ff5cca5a2e4b4d1cb5b5ced68b9fce88394ef6d93117426d6d4baafbcc05c56a"
)
DEV_EXPECTED_SIZE_BYTES = 8_929_167
DEV_EXPECTED_SHA256 = (
    "09e622916280ba04c9352acb1bc5bbe80f11a2598f6f34e934c51d9e6570f378"
)
SOURCE_MODE_OCTAL = "0600"

PASS_STATUS = "PASS_QUAC_SCHEMA_TOPOLOGY_AND_FAMILY_CAPACITY"
CAPACITY_STOP_STATUS = "STOP_QUAC_FAMILY_CAPACITY"
SOURCE_STOP_STATUS = "STOP_QUAC_SOURCE_IDENTITY_OR_SCHEMA"
INFRASTRUCTURE_STOP_STATUS = (
    "INFRASTRUCTURE_INVALID_QUAC_SOURCE_QUALIFICATION"
)
BOOTSTRAP_STOP_STATUS = (
    "STOP_QUAC_SOURCE_QUALIFICATION_BOOTSTRAP_INVALID"
)
PREFLIGHT_PASS_STATUS = "PASS_QUAC_SOURCE_QUALIFICATION_PREFLIGHT"

FAMILY_ORDER = ("FOLLOW", "MAYBE_FOLLOW", "DONT_FOLLOW")
PARTITION_ORDER = ("A_form", "A_hold", "M_search")
FORMAL_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "HOME": str(FORMAL_ROOT),
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
}
FORMAL_QUOTAS = {
    "A_form": 64,
    "A_hold": 32,
    "M_search": 32,
}

_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class QuacP1OneShotError(RuntimeError):
    """The frozen source-qualification controller failed closed."""


class OneShotConsumed(QuacP1OneShotError):
    """The formal work root already contains one-shot evidence."""


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
        raise QuacP1OneShotError("receipt is not canonical JSON") from exc


def _stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)[:-1]).hexdigest()


def _self_hashed(
    body: Mapping[str, Any],
    field: str = "self_sha256",
) -> dict[str, Any]:
    result = dict(body)
    result[field] = _stable_hash(result)
    return result


def _duplicate_rejecting_object(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise QuacP1OneShotError("JSON contains duplicate object keys")
        result[key] = value
    return result


def _reject_nonfinite(_value: str) -> None:
    raise QuacP1OneShotError("JSON contains a non-finite number")


def _decode_json(raw: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            raw.decode("ascii", errors="strict"),
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=_reject_nonfinite,
        )
    except QuacP1OneShotError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise QuacP1OneShotError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, Mapping):
        raise QuacP1OneShotError(f"{label} must be an object")
    return value


def _read_regular_bytes(path: Path, *, label: str) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise QuacP1OneShotError(f"{label} is unavailable") from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
    ):
        raise QuacP1OneShotError(f"{label} is not a private regular file")
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
        ) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ):
            raise QuacP1OneShotError(f"{label} changed during open")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise QuacP1OneShotError(f"{label} could not be read") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) != (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ):
        raise QuacP1OneShotError(f"{label} changed while being read")
    return b"".join(chunks)


def _file_sha256(path: Path, *, label: str) -> str:
    return hashlib.sha256(_read_regular_bytes(path, label=label)).hexdigest()


def _manifest_self(path: Path, *, expected: str, label: str) -> None:
    value = _decode_json(_read_regular_bytes(path, label=label), label=label)
    if value.get("self_sha256") != expected:
        raise QuacP1OneShotError(f"{label} semantic binding drifted")
    body = dict(value)
    body.pop("self_sha256", None)
    if _stable_hash(body) != expected:
        raise QuacP1OneShotError(f"{label} self hash drifted")


def _actual_python_identity(python_path: Path) -> dict[str, Any]:
    try:
        realpath = python_path.resolve(strict=True)
        metadata = realpath.stat()
    except OSError as exc:
        raise QuacP1OneShotError("frozen interpreter is unavailable") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise QuacP1OneShotError("frozen interpreter target is not regular")
    return {
        "launcher": str(python_path),
        "realpath": str(realpath),
        "version": platform.python_version(),
        "size_bytes": metadata.st_size,
        "mode_octal": f"{stat.S_IMODE(metadata.st_mode):04o}",
        "sha256": _file_sha256(realpath, label="frozen interpreter"),
    }


def _formal_source_contract(
    train_path: Path,
    dev_path: Path,
) -> dict[str, dict[str, Any]]:
    return {
        "train": {
            "path": str(train_path),
            "size_bytes": TRAIN_EXPECTED_SIZE_BYTES,
            "sha256": TRAIN_EXPECTED_SHA256,
            "mode_octal": SOURCE_MODE_OCTAL,
        },
        "dev": {
            "path": str(dev_path),
            "size_bytes": DEV_EXPECTED_SIZE_BYTES,
            "sha256": DEV_EXPECTED_SHA256,
            "mode_octal": SOURCE_MODE_OCTAL,
        },
    }


def _validate_source_metadata(
    source_contract: Mapping[str, Mapping[str, Any]],
) -> None:
    """Validate metadata only; preflight must never hash source payloads."""

    for split in ("train", "dev"):
        row = source_contract[split]
        path = Path(row["path"])
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise QuacP1OneShotError(
                "frozen source metadata is unavailable"
            ) from exc
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_uid != os.getuid()
            or metadata.st_size != row["size_bytes"]
            or f"{stat.S_IMODE(metadata.st_mode):04o}" != row["mode_octal"]
        ):
            raise QuacP1OneShotError("frozen source metadata drifted")


def _expected_freeze_keys() -> set[str]:
    return {
        "schema",
        "version",
        "study_id",
        "formal_attempt_limit",
        "formal_root",
        "project_root",
        "source_root",
        "work_root",
        "implementation_commit",
        "architecture_decision_self_sha256",
        "source_custody_self_sha256",
        "source_free_qualification_result_self_sha256",
        "source_free_qualification_result_file_sha256",
        "source_free_qualification_freeze_self_sha256",
        "python_identity",
        "environment",
        "source_contract",
        "unit_name",
        "unit_source_path",
        "unit_installed_path",
        "required_file_sha256s",
        "source_payload_access_count_before_qualification",
        "online_or_API_evaluation_count_before_qualification",
        "retry_replay_resample_or_repair_count_before_qualification",
        "self_sha256",
    }


def load_and_validate_freeze(
    freeze_path: Path,
    *,
    expected_formal_root: Path,
    expected_python: Path,
    expected_environment: Mapping[str, str],
    expected_source_contract: Mapping[str, Mapping[str, Any]],
    expected_installed_unit_path: Path,
    enforce_invocation_path: bool,
) -> dict[str, Any]:
    project_root = expected_formal_root / "reconstruction_v2"
    source_root = expected_formal_root.parent / "official_source_v1"
    work_root = expected_formal_root / "work"
    expected_freeze_path = project_root / "manifests" / FREEZE_FILENAME
    if enforce_invocation_path and freeze_path != expected_freeze_path:
        raise QuacP1OneShotError("formal freeze path drifted")

    raw = _read_regular_bytes(freeze_path, label="formal freeze")
    freeze_file_sha256 = hashlib.sha256(raw).hexdigest()
    value = dict(_decode_json(raw, label="formal freeze"))
    if set(value) != _expected_freeze_keys():
        raise QuacP1OneShotError("formal freeze field set drifted")
    self_sha = value.get("self_sha256")
    if not isinstance(self_sha, str) or _HEX64.fullmatch(self_sha) is None:
        raise QuacP1OneShotError("formal freeze self hash is invalid")
    body = dict(value)
    body.pop("self_sha256")
    if _stable_hash(body) != self_sha:
        raise QuacP1OneShotError("formal freeze self hash drifted")

    if (
        value["schema"] != FREEZE_SCHEMA
        or value["version"] != "v1"
        or value["study_id"] != STUDY_ID
        or value["formal_attempt_limit"] != 1
        or Path(value["formal_root"]) != expected_formal_root
        or Path(value["project_root"]) != project_root
        or Path(value["source_root"]) != source_root
        or Path(value["work_root"]) != work_root
        or not isinstance(value["implementation_commit"], str)
        or _HEX40.fullmatch(value["implementation_commit"]) is None
        or value["architecture_decision_self_sha256"]
        != ARCHITECTURE_DECISION_SELF_SHA256
        or value["source_custody_self_sha256"]
        != SOURCE_CUSTODY_SELF_SHA256
        or value["source_free_qualification_result_self_sha256"]
        != SOURCE_FREE_RESULT_SELF_SHA256
        or value["source_free_qualification_result_file_sha256"]
        != SOURCE_FREE_RESULT_FILE_SHA256
        or value["source_free_qualification_freeze_self_sha256"]
        != SOURCE_FREE_FREEZE_SELF_SHA256
        or value["source_payload_access_count_before_qualification"] != 0
        or value["online_or_API_evaluation_count_before_qualification"] != 0
        or value[
            "retry_replay_resample_or_repair_count_before_qualification"
        ]
        != 0
    ):
        raise QuacP1OneShotError("formal freeze semantic binding drifted")

    if value["python_identity"] != _actual_python_identity(expected_python):
        raise QuacP1OneShotError("frozen interpreter identity drifted")
    if Path(sys.executable) != expected_python:
        raise QuacP1OneShotError("running interpreter launcher drifted")
    if value["environment"] != dict(expected_environment):
        raise QuacP1OneShotError("frozen environment binding drifted")
    if dict(os.environ) != dict(expected_environment):
        raise QuacP1OneShotError("running environment drifted")
    if value["source_contract"] != expected_source_contract:
        raise QuacP1OneShotError("frozen source contract drifted")

    if (
        value["unit_name"] != UNIT_NAME
        or Path(value["unit_source_path"])
        != project_root / UNIT_RELATIVE_PATH
        or Path(value["unit_installed_path"]) != expected_installed_unit_path
    ):
        raise QuacP1OneShotError("formal unit binding drifted")
    try:
        installed_metadata = expected_installed_unit_path.lstat()
        installed_target = expected_installed_unit_path.resolve(strict=True)
    except OSError as exc:
        raise QuacP1OneShotError("installed formal unit is unavailable") from exc
    if (
        not stat.S_ISLNK(installed_metadata.st_mode)
        or installed_target != project_root / UNIT_RELATIVE_PATH
    ):
        raise QuacP1OneShotError("installed formal unit symlink drifted")

    required_hashes = value["required_file_sha256s"]
    if (
        not isinstance(required_hashes, Mapping)
        or set(required_hashes) != set(REQUIRED_FILE_RELATIVE_PATHS)
    ):
        raise QuacP1OneShotError("required frozen file set drifted")
    for relative in REQUIRED_FILE_RELATIVE_PATHS:
        expected_hash = required_hashes[relative]
        if (
            not isinstance(expected_hash, str)
            or _HEX64.fullmatch(expected_hash) is None
            or _file_sha256(
                project_root / relative,
                label=f"required file {relative}",
            )
            != expected_hash
        ):
            raise QuacP1OneShotError("required frozen file hash drifted")

    _manifest_self(
        project_root / ARCHITECTURE_RELATIVE_PATH,
        expected=ARCHITECTURE_DECISION_SELF_SHA256,
        label="architecture decision",
    )
    _manifest_self(
        project_root / CUSTODY_RELATIVE_PATH,
        expected=SOURCE_CUSTODY_SELF_SHA256,
        label="source custody",
    )
    _manifest_self(
        project_root / SOURCE_FREE_RESULT_RELATIVE_PATH,
        expected=SOURCE_FREE_RESULT_SELF_SHA256,
        label="source-free qualification result",
    )
    if (
        required_hashes[SOURCE_FREE_RESULT_RELATIVE_PATH]
        != SOURCE_FREE_RESULT_FILE_SHA256
    ):
        raise QuacP1OneShotError(
            "source-free qualification result file binding drifted"
        )
    _manifest_self(
        project_root / SOURCE_FREE_FREEZE_RELATIVE_PATH,
        expected=SOURCE_FREE_FREEZE_SELF_SHA256,
        label="source-free qualification freeze",
    )
    _validate_source_metadata(expected_source_contract)

    value["_freeze_file_sha256"] = freeze_file_sha256
    value["_freeze_path"] = freeze_path
    value["_project_root_path"] = project_root
    value["_work_root_path"] = work_root
    value["_train_path"] = Path(expected_source_contract["train"]["path"])
    value["_dev_path"] = Path(expected_source_contract["dev"]["path"])
    return value


def _load_core(path: Path, *, expected_sha256: str) -> ModuleType:
    if _file_sha256(path, label="frozen qualifier core") != expected_sha256:
        raise QuacP1OneShotError("frozen qualifier core hash drifted")
    spec = importlib.util.spec_from_file_location(
        "_quac_p1_source_qualification_v1_frozen",
        path,
    )
    if spec is None or spec.loader is None:
        raise QuacP1OneShotError("frozen qualifier core cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise QuacP1OneShotError("frozen qualifier core import failed") from exc
    if _file_sha256(path, label="frozen qualifier core") != expected_sha256:
        raise QuacP1OneShotError("frozen qualifier core changed during import")
    return module


def _contract_payload(
    contract: object,
    *,
    train_path: Path,
    dev_path: Path,
    mode_octal: str,
) -> dict[str, dict[str, Any]]:
    try:
        return {
            "train": {
                "path": str(train_path),
                "size_bytes": contract.train.size_bytes,
                "sha256": contract.train.sha256,
                "mode_octal": mode_octal,
            },
            "dev": {
                "path": str(dev_path),
                "size_bytes": contract.dev.size_bytes,
                "sha256": contract.dev.sha256,
                "mode_octal": mode_octal,
            },
        }
    except AttributeError as exc:
        raise QuacP1OneShotError("qualifier source contract is invalid") from exc


def _validate_safe_qualification(
    value: object,
    *,
    quotas: Mapping[str, int],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise QuacP1OneShotError("qualifier safe aggregate is not an object")
    expected_keys = {
        "schema",
        "status",
        "passed",
        "source_identity_pass",
        "required_schema_subset_pass",
        "train_dev_component_overlap_count",
        "global_component_count",
        "source_aggregates",
        "activity_counts",
        "capacity_flow",
    }
    if set(value) != expected_keys:
        raise QuacP1OneShotError("qualifier safe aggregate fields drifted")
    if (
        value["schema"] != "quac_p1_source_qualification_v1"
        or value["status"] not in {PASS_STATUS, CAPACITY_STOP_STATUS}
        or type(value["passed"]) is not bool
        or value["passed"] != (value["status"] == PASS_STATUS)
        or value["source_identity_pass"] != {"train": True, "dev": True}
        or value["required_schema_subset_pass"] is not True
    ):
        raise QuacP1OneShotError("qualifier safe aggregate semantics drifted")
    for field in (
        "train_dev_component_overlap_count",
        "global_component_count",
    ):
        if type(value[field]) is not int or value[field] < 0:
            raise QuacP1OneShotError("qualifier aggregate count drifted")

    aggregates = value["source_aggregates"]
    aggregate_keys = {
        "component_count",
        "eligible_component_count",
        "eligible_item_count",
        "nonfirst_turn_count",
        "role_ineligibility_reason_counts",
        "family_eligible_component_counts",
        "family_eligible_item_counts",
    }
    if not isinstance(aggregates, Mapping) or set(aggregates) != {"train", "dev"}:
        raise QuacP1OneShotError("source aggregate split set drifted")
    for split in ("train", "dev"):
        row = aggregates[split]
        if not isinstance(row, Mapping) or set(row) != aggregate_keys:
            raise QuacP1OneShotError("source aggregate fields drifted")
        for field in (
            "component_count",
            "eligible_component_count",
            "eligible_item_count",
            "nonfirst_turn_count",
        ):
            if type(row[field]) is not int or row[field] < 0:
                raise QuacP1OneShotError("source aggregate count drifted")
        for field in (
            "family_eligible_component_counts",
            "family_eligible_item_counts",
        ):
            counts = row[field]
            if not isinstance(counts, Mapping) or tuple(counts) != FAMILY_ORDER:
                raise QuacP1OneShotError("source family aggregate order drifted")
            if any(type(counts[name]) is not int or counts[name] < 0 for name in FAMILY_ORDER):
                raise QuacP1OneShotError("source family aggregate count drifted")
        role_reasons = row["role_ineligibility_reason_counts"]
        expected_reasons = (
            "previous_CANNOTANSWER",
            "current_CANNOTANSWER",
            "previous_NOT_CONTAINED_IN_FROZEN_WINDOW",
            "current_NOT_CONTAINED_IN_FROZEN_WINDOW",
        )
        if (
            not isinstance(role_reasons, Mapping)
            or tuple(role_reasons) != expected_reasons
            or any(
                type(role_reasons[name]) is not int
                or role_reasons[name] < 0
                for name in expected_reasons
            )
        ):
            raise QuacP1OneShotError(
                "source role-ineligibility aggregate drifted"
            )
        if (
            sum(
                row["family_eligible_item_counts"][family]
                for family in FAMILY_ORDER
            )
            != row["eligible_item_count"]
            or row["eligible_item_count"] > row["nonfirst_turn_count"]
        ):
            raise QuacP1OneShotError(
                "source item aggregate arithmetic drifted"
            )

    expected_activity = {
        "selection": 0,
        "model": 0,
        "action": 0,
        "score": 0,
        "online_or_API_evaluation": 0,
    }
    if value["activity_counts"] != expected_activity:
        raise QuacP1OneShotError("source activity boundary drifted")

    capacity = value["capacity_flow"]
    capacity_keys = {
        "component_global_capacity",
        "required_flow",
        "achieved_flow",
        "aggregate_slack",
        "slot_flow",
        "slot_slack",
        "all_nine_slots_saturated",
        "assignment_witness_output_count",
    }
    if not isinstance(capacity, Mapping) or set(capacity) != capacity_keys:
        raise QuacP1OneShotError("capacity aggregate fields drifted")
    if (
        set(quotas) != set(PARTITION_ORDER)
        or any(
            type(quotas[block]) is not int or quotas[block] <= 0
            for block in PARTITION_ORDER
        )
    ):
        raise QuacP1OneShotError("frozen capacity quotas drifted")
    expected_required_flow = sum(
        quotas[block] * len(FAMILY_ORDER)
        for block in PARTITION_ORDER
    )
    if (
        capacity["component_global_capacity"] != 1
        or type(capacity["required_flow"]) is not int
        or type(capacity["achieved_flow"]) is not int
        or type(capacity["aggregate_slack"]) is not int
        or capacity["required_flow"] < 0
        or capacity["achieved_flow"] < 0
        or capacity["aggregate_slack"] < 0
        or capacity["required_flow"] - capacity["achieved_flow"]
        != capacity["aggregate_slack"]
        or capacity["required_flow"] != expected_required_flow
        or capacity["all_nine_slots_saturated"] is not value["passed"]
        or capacity["assignment_witness_output_count"] != 0
    ):
        raise QuacP1OneShotError("capacity aggregate semantics drifted")
    for field in ("slot_flow", "slot_slack"):
        slots = capacity[field]
        if not isinstance(slots, Mapping) or tuple(slots) != PARTITION_ORDER:
            raise QuacP1OneShotError("capacity partition order drifted")
        for block in PARTITION_ORDER:
            counts = slots[block]
            if not isinstance(counts, Mapping) or tuple(counts) != FAMILY_ORDER:
                raise QuacP1OneShotError("capacity family order drifted")
            if any(type(counts[name]) is not int or counts[name] < 0 for name in FAMILY_ORDER):
                raise QuacP1OneShotError("capacity slot count drifted")
    achieved_from_slots = 0
    slack_from_slots = 0
    all_slots_saturated = True
    for block in PARTITION_ORDER:
        for family in FAMILY_ORDER:
            slot_flow = capacity["slot_flow"][block][family]
            slot_slack = capacity["slot_slack"][block][family]
            if slot_flow + slot_slack != quotas[block]:
                raise QuacP1OneShotError(
                    "capacity slot quota arithmetic drifted"
                )
            achieved_from_slots += slot_flow
            slack_from_slots += slot_slack
            all_slots_saturated = all_slots_saturated and slot_slack == 0
    if (
        achieved_from_slots != capacity["achieved_flow"]
        or slack_from_slots != capacity["aggregate_slack"]
        or all_slots_saturated != capacity["all_nine_slots_saturated"]
    ):
        raise QuacP1OneShotError("capacity slot totals drifted")

    # Rebuild, rather than forwarding, the complete aggregate whitelist.
    safe_aggregates = {
        split: {
            "component_count": aggregates[split]["component_count"],
            "eligible_component_count": aggregates[split][
                "eligible_component_count"
            ],
            "eligible_item_count": aggregates[split]["eligible_item_count"],
            "nonfirst_turn_count": aggregates[split][
                "nonfirst_turn_count"
            ],
            "role_ineligibility_reason_counts": {
                reason: aggregates[split][
                    "role_ineligibility_reason_counts"
                ][reason]
                for reason in (
                    "previous_CANNOTANSWER",
                    "current_CANNOTANSWER",
                    "previous_NOT_CONTAINED_IN_FROZEN_WINDOW",
                    "current_NOT_CONTAINED_IN_FROZEN_WINDOW",
                )
            },
            "family_eligible_component_counts": {
                family: aggregates[split][
                    "family_eligible_component_counts"
                ][family]
                for family in FAMILY_ORDER
            },
            "family_eligible_item_counts": {
                family: aggregates[split][
                    "family_eligible_item_counts"
                ][family]
                for family in FAMILY_ORDER
            },
        }
        for split in ("train", "dev")
    }
    safe_capacity = {
        "component_global_capacity": 1,
        "required_flow": capacity["required_flow"],
        "achieved_flow": capacity["achieved_flow"],
        "aggregate_slack": capacity["aggregate_slack"],
        "slot_flow": {
            block: {
                family: capacity["slot_flow"][block][family]
                for family in FAMILY_ORDER
            }
            for block in PARTITION_ORDER
        },
        "slot_slack": {
            block: {
                family: capacity["slot_slack"][block][family]
                for family in FAMILY_ORDER
            }
            for block in PARTITION_ORDER
        },
        "all_nine_slots_saturated": capacity[
            "all_nine_slots_saturated"
        ],
        "assignment_witness_output_count": 0,
    }
    return {
        "schema": "quac_p1_source_qualification_v1",
        "status": value["status"],
        "passed": value["passed"],
        "source_identity_pass": {"train": True, "dev": True},
        "required_schema_subset_pass": True,
        "train_dev_component_overlap_count": value[
            "train_dev_component_overlap_count"
        ],
        "global_component_count": value["global_component_count"],
        "source_aggregates": safe_aggregates,
        "activity_counts": expected_activity,
        "capacity_flow": safe_capacity,
    }


def _assert_pristine_work_root(path: Path) -> None:
    if not path.exists():
        return
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise OneShotConsumed("formal work root is unsafe or consumed")
    if any(path.iterdir()):
        raise OneShotConsumed("formal work root is already consumed")


def _prepare_pristine_work_root(path: Path) -> None:
    _assert_pristine_work_root(path)
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    if stat.S_IMODE(path.stat().st_mode) != 0o700:
        raise QuacP1OneShotError("formal work root mode drifted")


def _exclusive_write_json(path: Path, value: Mapping[str, Any]) -> str:
    """Publish a complete receipt atomically without replacing any path."""

    raw = _canonical_bytes(value)
    temporary_path = path.parent / f".{path.name}.tmp.{os.getpid()}"
    descriptor = -1
    try:
        descriptor = os.open(
            temporary_path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        os.fchmod(descriptor, 0o600)
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise QuacP1OneShotError("formal receipt write failed")
            view = view[written:]
        os.fsync(descriptor)
    except OSError as exc:
        raise OneShotConsumed("formal receipt path is already consumed") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    try:
        # Creating the final hard link is atomic and fails if any file or
        # symlink already occupies the immutable destination.
        os.link(
            temporary_path,
            path,
            src_dir_fd=None,
            dst_dir_fd=None,
            follow_symlinks=False,
        )
    except OSError as exc:
        try:
            temporary_path.unlink()
        except OSError:
            pass
        raise OneShotConsumed(
            "formal receipt path is already consumed"
        ) from exc
    try:
        temporary_path.unlink()
    except OSError as exc:
        raise QuacP1OneShotError(
            "formal receipt temporary link cleanup failed"
        ) from exc
    directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise QuacP1OneShotError("formal receipt mode drifted")
    return hashlib.sha256(raw).hexdigest()


def _attempt_body(freeze: Mapping[str, Any]) -> dict[str, Any]:
    return _self_hashed(
        {
            "schema": f"{VERSION}_attempt",
            "version": "v1",
            "study_id": STUDY_ID,
            "status": "formal_attempt_consumed_once",
            "attempt_ordinal": 1,
            "freeze_path": str(freeze["_freeze_path"]),
            "freeze_file_sha256": freeze["_freeze_file_sha256"],
            "freeze_self_sha256": freeze["self_sha256"],
            "implementation_commit": freeze["implementation_commit"],
            "architecture_decision_self_sha256": (
                ARCHITECTURE_DECISION_SELF_SHA256
            ),
            "source_custody_self_sha256": SOURCE_CUSTODY_SELF_SHA256,
            "source_free_qualification_result_self_sha256": (
                SOURCE_FREE_RESULT_SELF_SHA256
            ),
            "source_free_qualification_result_file_sha256": (
                SOURCE_FREE_RESULT_FILE_SHA256
            ),
            "source_free_qualification_freeze_self_sha256": (
                SOURCE_FREE_FREEZE_SELF_SHA256
            ),
            "source_payload_access_count_before_attempt": 0,
            "online_or_API_evaluation_count": 0,
            "retry_replay_resample_or_repair_count": 0,
        }
    )


def _fixed_failure_payload(
    *,
    status: str,
    freeze: Mapping[str, Any],
    attempt: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "quac_p1_source_qualification_v1",
        "status": status,
        "passed": False,
        "formal_result": True,
        "study_id": STUDY_ID,
        "freeze_file_sha256": freeze["_freeze_file_sha256"],
        "freeze_self_sha256": freeze["self_sha256"],
        "attempt_self_sha256": attempt["self_sha256"],
        "source_identity_pass": False,
        "required_schema_subset_pass": False,
        "train_dev_component_overlap_count": 0,
        "global_component_count": 0,
        "source_aggregates": {},
        "capacity_flow": {
            "required_flow": 0,
            "achieved_flow": 0,
            "aggregate_slack": 0,
            "assignment_witness_output_count": 0,
        },
        "activity_counts": {
            "selection": 0,
            "model": 0,
            "action": 0,
            "score": 0,
            "online_or_API_evaluation": 0,
        },
        "source_qualification_invocation_count": 1,
        "online_or_API_evaluation_count": 0,
        "retry_replay_resample_or_repair_count": 0,
    }


def _write_result_and_terminal(
    *,
    work_root: Path,
    result_body: Mapping[str, Any],
    freeze: Mapping[str, Any],
    attempt: Mapping[str, Any] | None,
    bootstrap: bool = False,
) -> dict[str, Any]:
    result = _self_hashed(result_body)
    result_file_sha256 = _exclusive_write_json(
        work_root / RESULT_NAME,
        result,
    )
    terminal = _self_hashed(
        {
            "schema": (
                f"{VERSION}_bootstrap_terminal"
                if bootstrap
                else "quac_p1_source_qualification_terminal_v1"
            ),
            "version": "v1",
            "study_id": STUDY_ID,
            "status": result["status"],
            "passed": result["passed"],
            "formal_complete": True,
            "bootstrap": bootstrap,
            "freeze_path": str(freeze.get("_freeze_path", FREEZE_PATH)),
            "freeze_file_sha256": freeze.get("_freeze_file_sha256"),
            "freeze_self_sha256": freeze.get("self_sha256"),
            "attempt_self_sha256": (
                None if attempt is None else attempt["self_sha256"]
            ),
            "result_file_sha256": result_file_sha256,
            "result_self_sha256": result["self_sha256"],
            "online_or_API_evaluation_count": 0,
            "retry_replay_resample_or_repair_count": 0,
        }
    )
    _exclusive_write_json(work_root / TERMINAL_NAME, terminal)
    return terminal


def _validate_core_contract(
    core: ModuleType,
    *,
    freeze: Mapping[str, Any],
    contract_override: object | None,
) -> object:
    contract = (
        core.FORMAL_CONTRACT if contract_override is None else contract_override
    )
    payload = _contract_payload(
        contract,
        train_path=freeze["_train_path"],
        dev_path=freeze["_dev_path"],
        mode_octal=SOURCE_MODE_OCTAL,
    )
    if payload != freeze["source_contract"]:
        raise QuacP1OneShotError("qualifier core source contract drifted")
    return contract


def _load_validated_execution(
    freeze_path: Path,
    *,
    expected_formal_root: Path,
    expected_python: Path,
    expected_environment: Mapping[str, str],
    expected_source_contract: Mapping[str, Mapping[str, Any]],
    expected_installed_unit_path: Path,
    enforce_invocation_path: bool,
    contract_override: object | None,
) -> tuple[dict[str, Any], ModuleType, object]:
    freeze = load_and_validate_freeze(
        freeze_path,
        expected_formal_root=expected_formal_root,
        expected_python=expected_python,
        expected_environment=expected_environment,
        expected_source_contract=expected_source_contract,
        expected_installed_unit_path=expected_installed_unit_path,
        enforce_invocation_path=enforce_invocation_path,
    )
    core = _load_core(
        freeze["_project_root_path"] / CORE_RELATIVE_PATH,
        expected_sha256=freeze["required_file_sha256s"][
            CORE_RELATIVE_PATH
        ],
    )
    contract = _validate_core_contract(
        core,
        freeze=freeze,
        contract_override=contract_override,
    )
    return freeze, core, contract


def run_preflight(
    freeze_path: Path,
    *,
    expected_formal_root: Path = FORMAL_ROOT,
    expected_python: Path = FROZEN_PYTHON,
    expected_environment: Mapping[str, str] = FORMAL_ENVIRONMENT,
    expected_source_contract: Mapping[str, Mapping[str, Any]] | None = None,
    expected_installed_unit_path: Path = INSTALLED_UNIT_PATH,
    enforce_invocation_path: bool = True,
    contract_override: object | None = None,
) -> dict[str, Any]:
    source_contract = (
        _formal_source_contract(TRAIN_SOURCE_PATH, DEV_SOURCE_PATH)
        if expected_source_contract is None
        else expected_source_contract
    )
    freeze, _core, _contract = _load_validated_execution(
        freeze_path,
        expected_formal_root=expected_formal_root,
        expected_python=expected_python,
        expected_environment=expected_environment,
        expected_source_contract=source_contract,
        expected_installed_unit_path=expected_installed_unit_path,
        enforce_invocation_path=enforce_invocation_path,
        contract_override=contract_override,
    )
    _assert_pristine_work_root(freeze["_work_root_path"])
    return _self_hashed(
        {
            "schema": f"{VERSION}_preflight_receipt",
            "version": "v1",
            "study_id": STUDY_ID,
            "status": PREFLIGHT_PASS_STATUS,
            "formal_attempt_created": False,
            "freeze_file_sha256": freeze["_freeze_file_sha256"],
            "freeze_self_sha256": freeze["self_sha256"],
            "source_payload_access_count": 0,
            "online_or_API_evaluation_count": 0,
            "retry_replay_resample_or_repair_count": 0,
        }
    )


def run_once(
    freeze_path: Path,
    *,
    expected_formal_root: Path = FORMAL_ROOT,
    expected_python: Path = FROZEN_PYTHON,
    expected_environment: Mapping[str, str] = FORMAL_ENVIRONMENT,
    expected_source_contract: Mapping[str, Mapping[str, Any]] | None = None,
    expected_installed_unit_path: Path = INSTALLED_UNIT_PATH,
    enforce_invocation_path: bool = True,
    contract_override: object | None = None,
) -> dict[str, Any]:
    """Validate, consume the attempt, read the source once, and terminate."""

    source_contract = (
        _formal_source_contract(TRAIN_SOURCE_PATH, DEV_SOURCE_PATH)
        if expected_source_contract is None
        else expected_source_contract
    )
    freeze, core, contract = _load_validated_execution(
        freeze_path,
        expected_formal_root=expected_formal_root,
        expected_python=expected_python,
        expected_environment=expected_environment,
        expected_source_contract=source_contract,
        expected_installed_unit_path=expected_installed_unit_path,
        enforce_invocation_path=enforce_invocation_path,
        contract_override=contract_override,
    )
    work_root = freeze["_work_root_path"]
    _prepare_pristine_work_root(work_root)
    attempt = _attempt_body(freeze)
    _exclusive_write_json(work_root / ATTEMPT_NAME, attempt)

    try:
        qualification = core.qualify_source_files(
            freeze["_train_path"],
            freeze["_dev_path"],
            contract=contract,
        )
        safe = _validate_safe_qualification(
            qualification,
            quotas=contract.quotas,
        )
        result_body = {
            "schema": safe["schema"],
            "status": safe["status"],
            "passed": safe["passed"],
            "source_identity_pass": safe["source_identity_pass"],
            "required_schema_subset_pass": safe[
                "required_schema_subset_pass"
            ],
            "train_dev_component_overlap_count": safe[
                "train_dev_component_overlap_count"
            ],
            "global_component_count": safe["global_component_count"],
            "source_aggregates": safe["source_aggregates"],
            "activity_counts": safe["activity_counts"],
            "capacity_flow": safe["capacity_flow"],
            "formal_result": True,
            "study_id": STUDY_ID,
            "freeze_file_sha256": freeze["_freeze_file_sha256"],
            "freeze_self_sha256": freeze["self_sha256"],
            "attempt_self_sha256": attempt["self_sha256"],
            "source_qualification_invocation_count": 1,
            "online_or_API_evaluation_count": 0,
            "retry_replay_resample_or_repair_count": 0,
        }
    except core.QuacP1SourceQualificationError:
        result_body = _fixed_failure_payload(
            status=SOURCE_STOP_STATUS,
            freeze=freeze,
            attempt=attempt,
        )
    except Exception:
        result_body = _fixed_failure_payload(
            status=INFRASTRUCTURE_STOP_STATUS,
            freeze=freeze,
            attempt=attempt,
        )
    return _write_result_and_terminal(
        work_root=work_root,
        result_body=result_body,
        freeze=freeze,
        attempt=attempt,
    )


def _write_bootstrap_stop(
    formal_root: Path,
    *,
    freeze_path: Path,
) -> dict[str, Any]:
    work_root = formal_root / "work"
    _prepare_pristine_work_root(work_root)
    bootstrap_freeze = {
        "_freeze_path": freeze_path,
        "_freeze_file_sha256": None,
        "self_sha256": None,
    }
    result_body = {
        "schema": "quac_p1_source_qualification_v1",
        "version": "v1",
        "study_id": STUDY_ID,
        "status": BOOTSTRAP_STOP_STATUS,
        "passed": False,
        "formal_result": True,
        "attempt_created": False,
        "source_payload_access_count": 0,
        "activity_counts": {
            "selection": 0,
            "model": 0,
            "action": 0,
            "score": 0,
            "online_or_API_evaluation": 0,
        },
        "online_or_API_evaluation_count": 0,
        "retry_replay_resample_or_repair_count": 0,
        "next_action": "close_QuAC_and_RJMC_without_source_replacement",
    }
    return _write_result_and_terminal(
        work_root=work_root,
        result_body=result_body,
        freeze=bootstrap_freeze,
        attempt=None,
        bootstrap=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen offline QuAC aggregate qualifier."
    )
    parser.add_argument("--freeze", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--formal", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    freeze_path = Path(arguments.freeze)
    if freeze_path != FREEZE_PATH:
        if arguments.formal:
            try:
                _write_bootstrap_stop(
                    FORMAL_ROOT,
                    freeze_path=freeze_path,
                )
                return 0
            except QuacP1OneShotError:
                pass
        return 2
    if arguments.preflight:
        try:
            receipt = run_preflight(freeze_path)
        except QuacP1OneShotError:
            return 2
        sys.stdout.buffer.write(_canonical_bytes(receipt))
        sys.stdout.buffer.flush()
        return 0

    try:
        terminal = run_once(freeze_path)
    except OneShotConsumed:
        return 3
    except QuacP1OneShotError:
        attempt_path = WORK_ROOT / ATTEMPT_NAME
        if not attempt_path.exists():
            try:
                _write_bootstrap_stop(
                    FORMAL_ROOT,
                    freeze_path=freeze_path,
                )
                return 0
            except QuacP1OneShotError:
                pass
        return 2
    # A completed scientific STOP or infrastructure-invalid terminal is a
    # successful one-shot service completion, not a systemd execution error.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
