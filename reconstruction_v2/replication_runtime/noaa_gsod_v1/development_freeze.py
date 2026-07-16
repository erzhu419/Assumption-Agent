from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from .contract import (
    ORACLE_IDS,
    STUDY_ID,
    TASK_CONTRACT,
    NoaaGsodError,
    canonical_json_bytes,
    payload_hash,
    sha256_file,
    verify_self_hash,
    with_self_hash,
)
from .development_schemas import (
    BATCH_POLICY_FIELDS,
    CALL_LEDGER_FIELDS,
    CONTENT_BOUNDARY_FIELDS,
    CONTROLLER_PLAN_FIELDS,
    DEVELOPMENT_ITEM_FIELDS,
    DEVELOPMENT_SCHEMA_SET_HASH,
    FREEZE_STATE_FIELDS,
    OPERATOR_BINDING_FIELDS,
    POST_JOIN_VERIFICATION_FIELDS,
    PROVIDER_POLICY_FIELDS,
    PUBLIC_BINDING_HASH_FIELDS,
    PUBLIC_FREEZE_FIELDS,
    SCHEDULE_FIELDS,
    SHARED_CONTEXT_FIELDS,
    SOURCE_VIEW_BINDING_FIELDS,
    WORKER_PLAN_FIELDS,
    WORK_UNIT_COUNTS_FIELDS,
    WORK_UNIT_FIELDS,
)
from .development_implementation import build_development_implementation_set
from .development_source import (
    PRIVATE_INDEX_NAME,
    verify_development_source_bundle,
    verify_development_source_index,
    verify_public_development_source_receipt,
)
from .pack import read_json, write_json
from .train_export import verify_train_preparation_receipt
from .typed_relational import (
    FORMATION_VERSION,
    OPERATOR_VERSION,
    load_formation_receipt,
    load_frozen_program,
)


PRE_RUN_FREEZE_VERSION = "noaa_gsod_formal_development_pre_run_freeze_v2"
WORKER_PLAN_VERSION = "noaa_gsod_formal_development_worker_plan_v2"
CONTROLLER_PLAN_VERSION = "noaa_gsod_formal_development_controller_plan_v2"
SOURCE_VIEW_TREE_VERSION = "noaa_gsod_development_source_view_tree_v1"
ENDPOINT_IDENTITY_VERSION = "sha256_canonical_origin_payload_v1"
REGISTERED_PROVIDER_ORIGIN = "https://ruoli.dev"
DEVELOPMENT_ITEM_COUNT = 6
ARM_IDS = ("raw_model", "agent_typed_model", "operator_only_local")
WORK_UNIT_COUNT = DEVELOPMENT_ITEM_COUNT * len(ARM_IDS)
MODEL_WORK_UNIT_COUNT = DEVELOPMENT_ITEM_COUNT * 2
LOCAL_WORK_UNIT_COUNT = DEVELOPMENT_ITEM_COUNT
MODEL_ID = "gpt-5.4-mini"
MODEL_REQUEST_BODY_BYTE_BUDGET = 64 * 1024
MODEL_OUTPUT_TOKEN_BUDGET = 256
ANONYMOUS_COLUMNS = ("STATION", "DATE", "PRCP")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CHANNEL_ID = re.compile(r"^[a-z0-9_.-]{1,64}$")

_BATCH_POLICY = {
    "attempts_per_work_unit": 1,
    "fallback_condition": "complete_plus_canary_unavailable",
    "fallback_scope": "entire_model_batch_pre_submission",
    "maximum_model_concurrency": MODEL_WORK_UNIT_COUNT,
    "mid_batch_provider_switch": False,
    "model_batch_count": 1,
    "model_batch_size": MODEL_WORK_UNIT_COUNT,
    "replays": 0,
    "resamples": 0,
    "retries": 0,
}
_WORK_UNIT_COUNTS = {
    "agent_typed_model": DEVELOPMENT_ITEM_COUNT,
    "model_total": MODEL_WORK_UNIT_COUNT,
    "operator_only_local": DEVELOPMENT_ITEM_COUNT,
    "raw_model": DEVELOPMENT_ITEM_COUNT,
    "total": WORK_UNIT_COUNT,
}
_POST_JOIN_VERIFICATION = {
    "all_work_units_must_join_before_release": True,
    "expected_join_count": WORK_UNIT_COUNT,
    "gold_or_oracle_material_in_worker_plan": False,
    "offline_oracle_ids": list(ORACLE_IDS),
    "offline_oracle_release_phase": "after_all_generation_and_operator_join",
    "online_judge_calls": 0,
    "required_offline_oracle_calls": DEVELOPMENT_ITEM_COUNT * len(ORACLE_IDS),
}
_CALL_LEDGER = {field: 0 for field in sorted(CALL_LEDGER_FIELDS)}
_CONTENT_BOUNDARY = {
    "development_gold_persisted_publicly": False,
    "development_raw_input_persisted_publicly": False,
    "development_station_identity_persisted_publicly": False,
    "model_answer_persisted_publicly": False,
    "private_controller_plan_persisted_publicly": False,
    "sealed_mapping_persisted_publicly": False,
    "source_view_private_index_persisted_publicly": False,
    "task_content_persisted_publicly": False,
    "trace_persisted_publicly": False,
}
_FREEZE_STATE = {
    "development_input_accessed": True,
    "development_input_staged": True,
    "generation_joined_count": 0,
    "generation_started": False,
    "gold_released": False,
    "launch_authorized": False,
    "model_request_hashes_precommitted": False,
    "operator_joined_count": 0,
    "scored": False,
    "sealed_runtime_accessed": False,
    "staged_item_count": DEVELOPMENT_ITEM_COUNT,
    "status": "pre_run_frozen_not_launched",
}
_SCHEDULE = {
    "agent_typed_model_units": DEVELOPMENT_ITEM_COUNT,
    "attempts_per_unit": 1,
    "development_items": DEVELOPMENT_ITEM_COUNT,
    "maximum_model_concurrency": MODEL_WORK_UNIT_COUNT,
    "max_output_tokens": MODEL_OUTPUT_TOKEN_BUDGET,
    "model_request_body_byte_budget": MODEL_REQUEST_BODY_BYTE_BUDGET,
    "model_units": MODEL_WORK_UNIT_COUNT,
    "operator_only_local_units": LOCAL_WORK_UNIT_COUNT,
    "raw_model_units": DEVELOPMENT_ITEM_COUNT,
    "replays": 0,
    "resamples": 0,
    "retries": 0,
    "total_work_units": WORK_UNIT_COUNT,
}


def _canonical_origin(value: str) -> str:
    raw = str(value or "").strip()
    try:
        parsed = urlsplit(raw)
        port = parsed.port
    except ValueError as exc:
        raise NoaaGsodError("provider endpoint origin is malformed") from exc
    if (
        parsed.scheme.casefold() != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        raise NoaaGsodError("provider endpoint must have one canonical HTTPS origin")
    host = parsed.hostname.encode("idna").decode("ascii").casefold()
    if ":" in host:
        host = f"[{host}]"
    authority = host if port in (None, 443) else f"{host}:{port}"
    canonical = f"https://{authority}"
    if canonical != REGISTERED_PROVIDER_ORIGIN:
        raise NoaaGsodError("provider endpoint is not the registered Ruoli origin")
    return canonical


def endpoint_identity_hash(origin: str) -> str:
    return payload_hash(
        {
            "canonical_origin": _canonical_origin(origin),
            "endpoint_identity_version": ENDPOINT_IDENTITY_VERSION,
        }
    )


@dataclass(frozen=True)
class ProviderIdentity:
    plus_channel_id: str
    plus_endpoint_origin: str
    pro_channel_id: str
    pro_endpoint_origin: str

    def validate(self) -> None:
        plus = str(self.plus_channel_id).strip().casefold()
        pro = str(self.pro_channel_id).strip().casefold()
        if _CHANNEL_ID.fullmatch(plus) is None or "plus" not in plus:
            raise NoaaGsodError("primary provider channel is not a valid Plus channel")
        if _CHANNEL_ID.fullmatch(pro) is None or "pro" not in pro:
            raise NoaaGsodError("fallback provider channel is not a valid Pro channel")
        if plus == pro:
            raise NoaaGsodError("Plus and Pro provider channels are not distinct")
        _canonical_origin(self.plus_endpoint_origin)
        _canonical_origin(self.pro_endpoint_origin)
        serialized = canonical_json_bytes(
            {"plus_channel_id": plus, "pro_channel_id": pro}
        ).decode("utf-8")
        if any(token in serialized for token in ("sk-", "api_key", "secret")):
            raise NoaaGsodError("provider channel identity contains a credential marker")

    @property
    def plus_endpoint_identity_hash(self) -> str:
        return endpoint_identity_hash(self.plus_endpoint_origin)

    @property
    def pro_endpoint_identity_hash(self) -> str:
        return endpoint_identity_hash(self.pro_endpoint_origin)

    def private_policy(self) -> dict[str, Any]:
        self.validate()
        return {
            "endpoint_identity_version": ENDPOINT_IDENTITY_VERSION,
            "primary": {
                "channel_id": self.plus_channel_id.strip().casefold(),
                "endpoint_identity_hash": self.plus_endpoint_identity_hash,
                "model": MODEL_ID,
            },
            "fallback": {
                "channel_id": self.pro_channel_id.strip().casefold(),
                "endpoint_identity_hash": self.pro_endpoint_identity_hash,
                "model": MODEL_ID,
            },
            "canary_uses_development_input": False,
            "fallback_condition": "complete_plus_canary_unavailable",
            "fallback_scope": "entire_12_model_unit_batch_before_submission",
            "partial_plus_batch_allows_fallback": False,
            "mid_batch_provider_switch": False,
            "secret_hmac_precommit_phase": (
                "runner_launch_precommit_before_any_model_submission"
            ),
        }

    @property
    def identity_hash(self) -> str:
        return payload_hash(self.private_policy())


def _require_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise NoaaGsodError(f"{label} is not a SHA-256 hash")
    return value


def _is_exact_mapping(
    value: object,
    expected: Mapping[str, Any],
    fields: frozenset[str],
) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == fields
        and canonical_json_bytes(value) == canonical_json_bytes(expected)
    )


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _require_ignored_when_inside_repository(path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    try:
        path.relative_to(repository)
    except ValueError:
        return
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", str(path)],
        cwd=repository,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        raise NoaaGsodError("development root is not git-ignored")


def _write_json_no_clobber(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise NoaaGsodError("public freeze already exists; no-clobber required")
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.tmp-",
            dir=path.parent,
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(payload) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError as exc:
            raise NoaaGsodError(
                "public freeze already exists; no-clobber required"
            ) from exc
        temporary_path.unlink()
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _validate_anonymous_csv(path: Path, expected_token: str) -> int:
    row_count = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ANONYMOUS_COLUMNS:
            raise NoaaGsodError("development input is not minimal")
        for row in reader:
            if None in row or row.get("STATION") != expected_token:
                raise NoaaGsodError("development input is not anonymous")
            row_count += 1
    if row_count == 0:
        raise NoaaGsodError("development input is empty")
    return row_count


def _stage_anonymous_input(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    expected_token: str,
) -> None:
    if source.is_symlink() or not source.is_file():
        raise NoaaGsodError("development source-view input is not a regular file")
    if sha256_file(source) != expected_sha256:
        raise NoaaGsodError("development source-view input hash drifted")
    _validate_anonymous_csv(source, expected_token)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        shutil.copyfile(source, temporary)
        if sha256_file(temporary) != expected_sha256:
            raise NoaaGsodError("staged development input differs from source-view")
        _validate_anonymous_csv(temporary, expected_token)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _staged_input_set_hash(items: Sequence[Mapping[str, Any]]) -> str:
    return payload_hash(
        [
            {
                "anonymous_item_id": item["anonymous_item_id"],
                "input_sha256": item["input_sha256"],
            }
            for item in items
        ]
    )


def _source_view_tree_hash(index: Mapping[str, Any]) -> str:
    return payload_hash(
        {
            "files": [
                {
                    "input_relative_path": item["input_relative_path"],
                    "input_sha256": item["input_sha256"],
                }
                for item in index["items"]
            ],
            "input_set_hash": index["input_set_hash"],
            "private_index_hash": index["private_index_hash"],
            "tree_version": SOURCE_VIEW_TREE_VERSION,
        }
    )


def _formation_binding(
    formation_receipt_path: Path,
    frozen_program_path: Path,
    train_receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], str, str]:
    receipt = dict(load_formation_receipt(formation_receipt_path))
    program = load_frozen_program(
        frozen_program_path,
        receipt_path=formation_receipt_path,
    )
    envelope = read_json(frozen_program_path)
    if (
        receipt.get("formation_version") != FORMATION_VERSION
        or receipt.get("study_id") != STUDY_ID
        or receipt.get("status") != "formed_unique_exact_crossfit"
        or receipt.get("raw_content_persisted") is not False
    ):
        raise NoaaGsodError("candidate formation is not the frozen TRAIN-only result")
    offline = receipt.get("offline_contract")
    claims = receipt.get("claim_boundary")
    source = receipt.get("source_receipt")
    selection = receipt.get("selection_receipt")
    if (
        not isinstance(offline, dict)
        or offline.get("partition") != "train"
        or offline.get("development_or_sealed_accessed") is not False
        or any(
            offline.get(field) != 0
            for field in ("model_calls", "network_calls", "online_judge_calls")
        )
        or not isinstance(claims, dict)
        or claims.get("train_only_formation") is not True
        or claims.get("performance_claim") is not False
        or claims.get("development_run_authorized") is not False
        or claims.get("sealed_run_authorized") is not False
        or not isinstance(source, dict)
        or source.get("train_view_hash") != train_receipt.get("train_view_hash")
        or source.get("task_contract_hash") != payload_hash(TASK_CONTRACT)
        or not isinstance(selection, dict)
        or selection.get("selected_program_hash") != program.program_hash
        or envelope.get("operator_version") != OPERATOR_VERSION
        or envelope.get("program_hash") != program.program_hash
        or envelope.get("formation_receipt_hash") != receipt.get("receipt_hash")
        or envelope.get("raw_content_persisted") is not False
    ):
        raise NoaaGsodError("candidate program and formation binding mismatch")
    return receipt, program.program_hash, payload_hash(envelope)


def prepare_development_pre_run_freeze(
    *,
    development_source_view_root: str | Path,
    development_source_index_path: str | Path,
    development_source_receipt_path: str | Path,
    acquisition_receipt_path: str | Path,
    development_root: str | Path,
    public_freeze_path: str | Path,
    train_preparation_receipt_path: str | Path,
    formation_receipt_path: str | Path,
    frozen_program_path: str | Path,
    provider_identity: ProviderIdentity,
) -> dict[str, Any]:
    provider_identity.validate()
    implementation_set = build_development_implementation_set()
    source_root_request = Path(development_source_view_root)
    if source_root_request.is_symlink():
        raise NoaaGsodError("development source-view root is a symbolic link")
    source_root = source_root_request.resolve(strict=True)
    source_index_file = Path(development_source_index_path).resolve(strict=True)
    expected_index_file = (source_root / PRIVATE_INDEX_NAME).resolve(strict=True)
    if source_index_file != expected_index_file:
        raise NoaaGsodError("development source index is not the source-view index")
    source_receipt_file = Path(development_source_receipt_path).resolve(strict=True)
    acquisition_receipt_file = Path(acquisition_receipt_path).resolve(strict=True)
    train_receipt_file = Path(train_preparation_receipt_path).resolve(strict=True)
    formation_file = Path(formation_receipt_path).resolve(strict=True)
    program_file = Path(frozen_program_path).resolve(strict=True)
    output_request = Path(development_root)
    public_request = Path(public_freeze_path)
    if output_request.is_symlink():
        raise NoaaGsodError("development root must not be a symbolic link")
    if public_request.exists() or public_request.is_symlink():
        raise NoaaGsodError("public freeze already exists; no-clobber required")
    output_root = output_request.resolve()
    public_file = public_request.resolve()
    if _paths_overlap(output_root, source_root):
        raise NoaaGsodError("development root overlaps gold-free source-view root")
    if _paths_overlap(public_file, source_root) or _paths_overlap(
        public_file, output_root
    ):
        raise NoaaGsodError("public freeze path overlaps a source or development root")
    _require_ignored_when_inside_repository(output_root)
    if output_root.exists():
        raise NoaaGsodError("development root already exists; unique root required")

    source_receipt = verify_public_development_source_receipt(
        read_json(source_receipt_file)
    )
    verify_development_source_bundle(
        source_receipt,
        source_view_root=source_root,
        acquisition_receipt_path=acquisition_receipt_file,
    )
    source_index = verify_development_source_index(
        read_json(source_index_file),
        source_view_root=source_root,
    )
    if source_index.get("study_id") != STUDY_ID:
        raise NoaaGsodError("development source-view study identity mismatch")
    source_tree_hash = _source_view_tree_hash(source_index)

    train_receipt = verify_train_preparation_receipt(read_json(train_receipt_file))
    if (
        train_receipt.get("study_id") != STUDY_ID
        or train_receipt.get("task_contract_hash") != payload_hash(TASK_CONTRACT)
        or train_receipt.get("source_private_pack_hash")
        != source_receipt["binding_hashes"]["source_private_pack_hash"]
    ):
        raise NoaaGsodError("TRAIN and development source receipts do not share acquisition")
    formation, program_id, program_envelope_hash = _formation_binding(
        formation_file,
        program_file,
        train_receipt,
    )

    output_root.mkdir(parents=True, exist_ok=False)
    try:
        staged_items: list[dict[str, Any]] = []
        for ordinal, source_item in enumerate(source_index["items"]):
            anonymous_item_id = f"development_item_{ordinal:02d}"
            anonymous_token = f"DEVELOPMENT_STATION_{ordinal:02d}"
            source_relative = str(source_item["input_relative_path"])
            source_path = (source_root / source_relative).resolve(strict=True)
            if source_root not in source_path.parents:
                raise NoaaGsodError("development source input escapes source-view root")
            staged_relative = f"generation_inputs/{anonymous_item_id}.csv"
            destination = output_root / staged_relative
            expected_sha256 = _require_sha(
                source_item.get("input_sha256"),
                "development source input",
            )
            _stage_anonymous_input(
                source_path,
                destination,
                expected_sha256=expected_sha256,
                expected_token=anonymous_token,
            )
            staged_items.append(
                {
                    "anonymous_item_id": anonymous_item_id,
                    "anonymized_station_token": anonymous_token,
                    "input_columns": list(ANONYMOUS_COLUMNS),
                    "input_relative_path": staged_relative,
                    "input_sha256": sha256_file(destination),
                    "ordinal": ordinal,
                }
            )
        staged_input_set_hash = _staged_input_set_hash(staged_items)
        if staged_input_set_hash != source_index["input_set_hash"]:
            raise NoaaGsodError("staged development input set differs from source-view")
        verify_development_source_bundle(
            source_receipt,
            source_view_root=source_root,
        )
        if _source_view_tree_hash(read_json(source_index_file)) != source_tree_hash:
            raise NoaaGsodError("development source-view tree changed during staging")

        operator_root = output_root / "operator"
        operator_root.mkdir()
        staged_program = operator_root / "frozen_program.json"
        shutil.copyfile(program_file, staged_program)
        shared_context = {
            "max_output_tokens": MODEL_OUTPUT_TOKEN_BUDGET,
            "model_request_body_byte_budget": MODEL_REQUEST_BODY_BYTE_BUDGET,
            "task_contract": TASK_CONTRACT,
            "task_contract_hash": payload_hash(TASK_CONTRACT),
        }
        shared_context_hash = payload_hash(shared_context)
        work_units: list[dict[str, Any]] = []
        for item in staged_items:
            common = {
                "anonymous_item_id": item["anonymous_item_id"],
                "attempts": 1,
                "input_sha256": item["input_sha256"],
                "model": MODEL_ID,
                # Exact effective request hashes belong to the runner launch
                # precommit receipt; this pre-run plan intentionally has none.
                "model_request_hash": None,
                "shared_context_hash": shared_context_hash,
            }
            work_units.extend(
                [
                    {
                        **common,
                        "arm": "raw_model",
                        "execution_kind": "provider_model",
                        "model_response_contract": "canonical_task_json_only",
                        "post_model_local_operator": False,
                        "program_id": None,
                        "work_unit_id": f"{item['anonymous_item_id']}:raw_model",
                    },
                    {
                        **common,
                        "arm": "agent_typed_model",
                        "execution_kind": "provider_model_then_local_operator",
                        "model_response_contract": "opaque_frozen_program_id_only",
                        "post_model_local_operator": True,
                        "program_id": program_id,
                        "work_unit_id": (
                            f"{item['anonymous_item_id']}:agent_typed_model"
                        ),
                    },
                    {
                        **common,
                        "arm": "operator_only_local",
                        "execution_kind": "local_operator",
                        "model": None,
                        "model_response_contract": None,
                        "post_model_local_operator": True,
                        "program_id": program_id,
                        "work_unit_id": (
                            f"{item['anonymous_item_id']}:operator_only_local"
                        ),
                    },
                ]
            )
        worker_body: dict[str, Any] = {
            "batch_policy": dict(_BATCH_POLICY),
            "development_item_count": DEVELOPMENT_ITEM_COUNT,
            "items": staged_items,
            "operator_binding": {
                "frozen_program_file_sha256": sha256_file(staged_program),
                "frozen_program_relative_path": "operator/frozen_program.json",
                "operator_version": OPERATOR_VERSION,
                "program_envelope_hash": program_envelope_hash,
                "program_id": program_id,
            },
            "shared_context": shared_context,
            "study_id": STUDY_ID,
            "work_unit_counts": dict(_WORK_UNIT_COUNTS),
            "work_units": work_units,
            "worker_plan_version": WORKER_PLAN_VERSION,
        }
        worker_plan = with_self_hash(worker_body, "worker_plan_hash")
        verify_worker_plan(worker_plan, development_root=output_root)
        write_json(output_root / "worker_plan.json", worker_plan)

        source_view_binding = {
            "development_source_index_file_sha256": sha256_file(
                source_index_file
            ),
            "development_source_index_hash": source_index["private_index_hash"],
            "development_source_receipt_file_sha256": sha256_file(
                source_receipt_file
            ),
            "development_source_receipt_hash": source_receipt[
                "development_source_receipt_hash"
            ],
            "source_view_input_set_hash": source_index["input_set_hash"],
            "source_view_tree_hash": source_tree_hash,
            "staged_input_set_hash": staged_input_set_hash,
        }
        controller_body: dict[str, Any] = {
            "controller_plan_version": CONTROLLER_PLAN_VERSION,
            "development_root": str(output_root),
            "development_root_commitment": payload_hash(str(output_root)),
            "generation_worker_plan_hash": worker_plan["worker_plan_hash"],
            "post_join_verification": dict(_POST_JOIN_VERIFICATION),
            "source_view_binding": source_view_binding,
            "study_id": STUDY_ID,
        }
        controller_plan = with_self_hash(controller_body, "controller_plan_hash")
        verify_controller_plan(controller_plan, worker_plan=worker_plan)
        write_json(output_root / "controller_plan.private.json", controller_plan)

        public_body: dict[str, Any] = {
            "binding_hashes": {
                "acquisition_receipt_file_sha256": sha256_file(
                    acquisition_receipt_file
                ),
                "acquisition_receipt_hash": source_receipt["binding_hashes"][
                    "acquisition_receipt_hash"
                ],
                "candidate_formation_receipt_file_sha256": sha256_file(
                    formation_file
                ),
                "candidate_formation_receipt_hash": formation["receipt_hash"],
                "candidate_program_file_sha256": sha256_file(program_file),
                "candidate_program_id": program_id,
                "controller_plan_hash": controller_plan["controller_plan_hash"],
                "development_root_commitment": controller_plan[
                    "development_root_commitment"
                ],
                "development_schema_set_hash": DEVELOPMENT_SCHEMA_SET_HASH,
                "development_source_index_file_sha256": source_view_binding[
                    "development_source_index_file_sha256"
                ],
                "development_source_index_hash": source_view_binding[
                    "development_source_index_hash"
                ],
                "development_source_input_set_hash": source_view_binding[
                    "source_view_input_set_hash"
                ],
                "development_source_receipt_file_sha256": source_view_binding[
                    "development_source_receipt_file_sha256"
                ],
                "development_source_receipt_hash": source_view_binding[
                    "development_source_receipt_hash"
                ],
                "development_source_tree_hash": source_tree_hash,
                "implementation_set_hash": implementation_set[
                    "implementation_set_hash"
                ],
                "provider_identity_hash": provider_identity.identity_hash,
                "staged_input_set_hash": staged_input_set_hash,
                "task_contract_hash": payload_hash(TASK_CONTRACT),
                "train_preparation_receipt_file_sha256": sha256_file(
                    train_receipt_file
                ),
                "train_preparation_receipt_hash": train_receipt[
                    "preparation_receipt_hash"
                ],
                "worker_plan_hash": worker_plan["worker_plan_hash"],
            },
            "call_ledger_at_freeze": dict(_CALL_LEDGER),
            "content_boundary": dict(_CONTENT_BOUNDARY),
            "freeze_state": dict(_FREEZE_STATE),
            "performance_gate_added": False,
            "pre_run_freeze_version": PRE_RUN_FREEZE_VERSION,
            "provider_policy": {
                "endpoint_identity_version": ENDPOINT_IDENTITY_VERSION,
                "fallback_condition": "complete_plus_canary_unavailable",
                "fallback_scope": (
                    "entire_12_model_unit_batch_before_submission"
                ),
                "mid_batch_switch": False,
                "model": MODEL_ID,
                "primary_tier": "plus",
                "secondary_tier": "pro",
                "secret_hmac_precommit_phase": (
                    "runner_launch_precommit_before_any_model_submission"
                ),
            },
            "schedule": dict(_SCHEDULE),
            "study_id": STUDY_ID,
        }
        public_freeze = with_self_hash(public_body, "pre_run_freeze_hash")
        verify_public_pre_run_freeze(public_freeze)
        _write_json_no_clobber(public_file, public_freeze)
    except Exception:
        shutil.rmtree(output_root, ignore_errors=True)
        raise

    return {
        "development_item_count": DEVELOPMENT_ITEM_COUNT,
        "development_root_commitment": controller_plan[
            "development_root_commitment"
        ],
        "development_source_tree_hash": source_tree_hash,
        "implementation_set_hash": implementation_set["implementation_set_hash"],
        "model_work_unit_count": MODEL_WORK_UNIT_COUNT,
        "pre_run_freeze_hash": public_freeze["pre_run_freeze_hash"],
        "staged_input_set_hash": staged_input_set_hash,
        "total_work_unit_count": WORK_UNIT_COUNT,
        "worker_plan_hash": worker_plan["worker_plan_hash"],
    }


def verify_worker_plan(
    payload: Mapping[str, Any],
    *,
    development_root: str | Path,
) -> dict[str, Any]:
    plan = dict(payload)
    verify_self_hash(plan, "worker_plan_hash")
    if set(plan) != WORKER_PLAN_FIELDS:
        raise NoaaGsodError("development worker plan schema mismatch")
    if (
        plan.get("worker_plan_version") != WORKER_PLAN_VERSION
        or plan.get("study_id") != STUDY_ID
        or plan.get("development_item_count") != DEVELOPMENT_ITEM_COUNT
    ):
        raise NoaaGsodError("development worker plan identity mismatch")
    root = Path(development_root).resolve(strict=True)
    items = plan.get("items")
    if not isinstance(items, list) or len(items) != DEVELOPMENT_ITEM_COUNT:
        raise NoaaGsodError("development worker item count mismatch")
    item_hashes: dict[str, str] = {}
    for ordinal, item in enumerate(items):
        if not isinstance(item, dict) or set(item) != DEVELOPMENT_ITEM_FIELDS:
            raise NoaaGsodError("development item schema mismatch")
        item_id = f"development_item_{ordinal:02d}"
        token = f"DEVELOPMENT_STATION_{ordinal:02d}"
        relative = f"generation_inputs/{item_id}.csv"
        if (
            type(item.get("ordinal")) is not int
            or item.get("ordinal") != ordinal
            or item.get("anonymous_item_id") != item_id
            or item.get("anonymized_station_token") != token
            or not isinstance(item.get("input_columns"), list)
            or tuple(item["input_columns"]) != ANONYMOUS_COLUMNS
            or item.get("input_relative_path") != relative
        ):
            raise NoaaGsodError("development anonymous item identity mismatch")
        input_sha256 = _require_sha(item.get("input_sha256"), "development input")
        path = (root / relative).resolve(strict=True)
        if root not in path.parents or path.is_symlink() or not path.is_file():
            raise NoaaGsodError("development anonymous input binding mismatch")
        if sha256_file(path) != input_sha256:
            raise NoaaGsodError("development anonymous input hash mismatch")
        _validate_anonymous_csv(path, token)
        item_hashes[item_id] = input_sha256

    shared = plan.get("shared_context")
    expected_shared = {
        "max_output_tokens": MODEL_OUTPUT_TOKEN_BUDGET,
        "model_request_body_byte_budget": MODEL_REQUEST_BODY_BYTE_BUDGET,
        "task_contract": TASK_CONTRACT,
        "task_contract_hash": payload_hash(TASK_CONTRACT),
    }
    if not _is_exact_mapping(shared, expected_shared, SHARED_CONTEXT_FIELDS):
        raise NoaaGsodError("development shared context mismatch")
    shared_hash = payload_hash(shared)

    operator = plan.get("operator_binding")
    if not isinstance(operator, dict) or set(operator) != OPERATOR_BINDING_FIELDS:
        raise NoaaGsodError("development operator binding schema mismatch")
    if (
        operator.get("frozen_program_relative_path")
        != "operator/frozen_program.json"
        or operator.get("operator_version") != OPERATOR_VERSION
    ):
        raise NoaaGsodError("development operator binding identity mismatch")
    program_id = _require_sha(operator.get("program_id"), "program id")
    program_envelope_hash = _require_sha(
        operator.get("program_envelope_hash"),
        "program envelope",
    )
    program_file_sha256 = _require_sha(
        operator.get("frozen_program_file_sha256"),
        "frozen program file",
    )
    program_path = (root / "operator/frozen_program.json").resolve(strict=True)
    if root not in program_path.parents or program_path.is_symlink():
        raise NoaaGsodError("frozen program escapes development root")
    envelope = read_json(program_path)
    if (
        sha256_file(program_path) != program_file_sha256
        or payload_hash(envelope) != program_envelope_hash
        or set(envelope)
        != {
            "formation_receipt_hash",
            "operator_version",
            "program",
            "program_hash",
            "raw_content_persisted",
        }
        or envelope.get("operator_version") != OPERATOR_VERSION
        or envelope.get("program_hash") != program_id
        or envelope.get("raw_content_persisted") is not False
    ):
        raise NoaaGsodError("frozen program file binding mismatch")

    counts = plan.get("work_unit_counts")
    if not _is_exact_mapping(
        counts,
        _WORK_UNIT_COUNTS,
        WORK_UNIT_COUNTS_FIELDS,
    ):
        raise NoaaGsodError("development work unit counts mismatch")
    units = plan.get("work_units")
    if not isinstance(units, list) or len(units) != WORK_UNIT_COUNT:
        raise NoaaGsodError("development work unit count mismatch")
    expected_pairs = {
        (item_id, arm) for item_id in item_hashes for arm in ARM_IDS
    }
    observed_pairs: set[tuple[str, str]] = set()
    observed_ids: set[str] = set()
    for unit in units:
        if not isinstance(unit, dict) or set(unit) != WORK_UNIT_FIELDS:
            raise NoaaGsodError("development work unit schema mismatch")
        item_id = str(unit.get("anonymous_item_id"))
        arm = str(unit.get("arm"))
        pair = (item_id, arm)
        work_unit_id = str(unit.get("work_unit_id"))
        if (
            pair not in expected_pairs
            or pair in observed_pairs
            or work_unit_id in observed_ids
            or work_unit_id != f"{item_id}:{arm}"
        ):
            raise NoaaGsodError("development work unit Cartesian identity mismatch")
        observed_pairs.add(pair)
        observed_ids.add(work_unit_id)
        if (
            type(unit.get("attempts")) is not int
            or unit.get("attempts") != 1
            or unit.get("input_sha256") != item_hashes[item_id]
            or unit.get("shared_context_hash") != shared_hash
            or unit.get("model_request_hash") is not None
        ):
            raise NoaaGsodError("development work unit fairness binding mismatch")
        if arm == "raw_model":
            if (
                unit.get("execution_kind") != "provider_model"
                or unit.get("model") != MODEL_ID
                or unit.get("model_response_contract")
                != "canonical_task_json_only"
                or unit.get("program_id") is not None
                or unit.get("post_model_local_operator") is not False
            ):
                raise NoaaGsodError("RAW work unit contract mismatch")
        elif arm == "agent_typed_model":
            if (
                unit.get("execution_kind")
                != "provider_model_then_local_operator"
                or unit.get("model") != MODEL_ID
                or unit.get("model_response_contract")
                != "opaque_frozen_program_id_only"
                or unit.get("program_id") != program_id
                or unit.get("post_model_local_operator") is not True
            ):
                raise NoaaGsodError("agent+typed work unit contract mismatch")
        elif (
            unit.get("execution_kind") != "local_operator"
            or unit.get("model") is not None
            or unit.get("model_response_contract") is not None
            or unit.get("program_id") != program_id
            or unit.get("post_model_local_operator") is not True
        ):
            raise NoaaGsodError("operator-only work unit contract mismatch")
    if observed_pairs != expected_pairs or len(observed_ids) != WORK_UNIT_COUNT:
        raise NoaaGsodError("development work unit Cartesian set mismatch")
    batch = plan.get("batch_policy")
    if not _is_exact_mapping(batch, _BATCH_POLICY, BATCH_POLICY_FIELDS):
        raise NoaaGsodError("development batch policy mismatch")
    return plan


def verify_controller_plan(
    payload: Mapping[str, Any],
    *,
    worker_plan: Mapping[str, Any],
) -> dict[str, Any]:
    plan = dict(payload)
    verify_self_hash(plan, "controller_plan_hash")
    if set(plan) != CONTROLLER_PLAN_FIELDS:
        raise NoaaGsodError("development controller plan schema mismatch")
    if (
        plan.get("controller_plan_version") != CONTROLLER_PLAN_VERSION
        or plan.get("study_id") != STUDY_ID
        or plan.get("generation_worker_plan_hash")
        != worker_plan.get("worker_plan_hash")
        or not isinstance(plan.get("development_root"), str)
        or plan.get("development_root_commitment")
        != payload_hash(str(plan.get("development_root")))
    ):
        raise NoaaGsodError("development controller binding mismatch")
    source_binding = plan.get("source_view_binding")
    if (
        not isinstance(source_binding, dict)
        or set(source_binding) != SOURCE_VIEW_BINDING_FIELDS
    ):
        raise NoaaGsodError("development source-view binding schema mismatch")
    for field, value in source_binding.items():
        _require_sha(value, field)
    staged_hash = _staged_input_set_hash(worker_plan["items"])
    if (
        source_binding.get("source_view_input_set_hash") != staged_hash
        or source_binding.get("staged_input_set_hash") != staged_hash
    ):
        raise NoaaGsodError("development source-view and staged input sets differ")
    verification = plan.get("post_join_verification")
    if not _is_exact_mapping(
        verification,
        _POST_JOIN_VERIFICATION,
        POST_JOIN_VERIFICATION_FIELDS,
    ):
        raise NoaaGsodError("development post-join verifier policy mismatch")
    return plan


def verify_public_pre_run_freeze(payload: Mapping[str, Any]) -> dict[str, Any]:
    freeze = dict(payload)
    verify_self_hash(freeze, "pre_run_freeze_hash")
    if set(freeze) != PUBLIC_FREEZE_FIELDS:
        raise NoaaGsodError("public development freeze schema mismatch")
    if (
        freeze.get("pre_run_freeze_version") != PRE_RUN_FREEZE_VERSION
        or freeze.get("study_id") != STUDY_ID
        or freeze.get("performance_gate_added") is not False
    ):
        raise NoaaGsodError("public development freeze identity mismatch")
    bindings = freeze.get("binding_hashes")
    if not isinstance(bindings, dict) or set(bindings) != PUBLIC_BINDING_HASH_FIELDS:
        raise NoaaGsodError("public development binding schema mismatch")
    for field, value in bindings.items():
        _require_sha(value, field)
    if (
        bindings.get("development_schema_set_hash")
        != DEVELOPMENT_SCHEMA_SET_HASH
        or bindings.get("task_contract_hash") != payload_hash(TASK_CONTRACT)
        or bindings.get("development_source_input_set_hash")
        != bindings.get("staged_input_set_hash")
    ):
        raise NoaaGsodError("public development fixed binding mismatch")
    ledger = freeze.get("call_ledger_at_freeze")
    if not _is_exact_mapping(ledger, _CALL_LEDGER, CALL_LEDGER_FIELDS):
        raise NoaaGsodError("public freeze call ledger mismatch")
    boundary = freeze.get("content_boundary")
    if not _is_exact_mapping(
        boundary,
        _CONTENT_BOUNDARY,
        CONTENT_BOUNDARY_FIELDS,
    ):
        raise NoaaGsodError("public development content boundary mismatch")
    state = freeze.get("freeze_state")
    if not _is_exact_mapping(state, _FREEZE_STATE, FREEZE_STATE_FIELDS):
        raise NoaaGsodError("public development freeze state mismatch")
    policy = freeze.get("provider_policy")
    expected_policy = {
        "endpoint_identity_version": ENDPOINT_IDENTITY_VERSION,
        "fallback_condition": "complete_plus_canary_unavailable",
        "fallback_scope": "entire_12_model_unit_batch_before_submission",
        "mid_batch_switch": False,
        "model": MODEL_ID,
        "primary_tier": "plus",
        "secondary_tier": "pro",
        "secret_hmac_precommit_phase": (
            "runner_launch_precommit_before_any_model_submission"
        ),
    }
    if not _is_exact_mapping(policy, expected_policy, PROVIDER_POLICY_FIELDS):
        raise NoaaGsodError("public development provider policy mismatch")
    schedule = freeze.get("schedule")
    if not _is_exact_mapping(schedule, _SCHEDULE, SCHEDULE_FIELDS):
        raise NoaaGsodError("public development schedule mismatch")
    return freeze


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage and freeze, but do not launch, NOAA formal development."
    )
    parser.add_argument("--development-source-view-root", required=True)
    parser.add_argument("--development-source-index", required=True)
    parser.add_argument("--development-source-receipt", required=True)
    parser.add_argument("--acquisition-receipt", required=True)
    parser.add_argument("--development-root", required=True)
    parser.add_argument("--public-freeze", required=True)
    parser.add_argument("--train-preparation-receipt", required=True)
    parser.add_argument("--formation-receipt", required=True)
    parser.add_argument("--frozen-program", required=True)
    parser.add_argument("--plus-channel-id", required=True)
    parser.add_argument("--plus-endpoint-origin", required=True)
    parser.add_argument("--pro-channel-id", required=True)
    parser.add_argument("--pro-endpoint-origin", required=True)
    arguments = parser.parse_args(argv)
    summary = prepare_development_pre_run_freeze(
        development_source_view_root=arguments.development_source_view_root,
        development_source_index_path=arguments.development_source_index,
        development_source_receipt_path=arguments.development_source_receipt,
        acquisition_receipt_path=arguments.acquisition_receipt,
        development_root=arguments.development_root,
        public_freeze_path=arguments.public_freeze,
        train_preparation_receipt_path=arguments.train_preparation_receipt,
        formation_receipt_path=arguments.formation_receipt,
        frozen_program_path=arguments.frozen_program,
        provider_identity=ProviderIdentity(
            plus_channel_id=arguments.plus_channel_id,
            plus_endpoint_origin=arguments.plus_endpoint_origin,
            pro_channel_id=arguments.pro_channel_id,
            pro_endpoint_origin=arguments.pro_endpoint_origin,
        ),
    )
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
