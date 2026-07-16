"""One-shot RAW/P/official-HippoRAG HotpotQA family-out measurement."""

from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading
from typing import Any, Mapping, Sequence

from ..models import stable_hash
from . import hotpot_family_out_acquisition_v1 as acquisition
from .musique_formal_runtime_binding_v2 import (
    PreparedFormalRuntimeV2,
    prepare_formal_runtime_v2,
)
from .musique_official_core_comparison_v1 import _assert_git_ignored_private_path
from .musique_recursive_study_blocks_v1 import load_study_frozen_program
from .musique_typed_retriever_formation_v1 import (
    RetrievalParagraph,
    TypedRetrievalProgram,
    retrieve as typed_retrieve,
)


VERSION = "hotpot_family_out_runner_v1"
CAPABILITY_SCHEMA = f"{acquisition.VERSION}_bubblewrap_capability"
FREEZE_SCHEMA = f"{VERSION}_pre_run_freeze"
REPORT_SCHEMA = f"{VERSION}_aggregate_report"
FAILURE_SCHEMA = f"{VERSION}_failure"
ARM_IDS = ("canonical_RAW", "frozen_P", "official_HippoRAG")
WORK_UNIT_COUNT = len(ARM_IDS) * acquisition.SAMPLE_COUNT
MAXIMUM_CONCURRENCY = WORK_UNIT_COUNT
CONSUMPTION_FILENAME = "family_out.authorization.consumed.json"
REPORT_FILENAME = "family_out.aggregate.report.json"
FAILURE_FILENAME = "family_out.failure.json"
IMPLEMENTATION_SCHEMA = f"{VERSION}_implementation"
IMPLEMENTATION_RELATIVE_FILES = acquisition.IMPLEMENTATION_RELATIVE_FILES
BWRAP_PATH = Path("/usr/bin/bwrap")
BWRAP_SHA256 = acquisition.BWRAP_SHA256
BWRAP_VERSION = acquisition.BWRAP_VERSION
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_CLEAN_MODULE_CLI_ACTIVE = False


class HotpotFamilyOutRunnerError(RuntimeError):
    """The frozen family-out execution failed closed."""


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise HotpotFamilyOutRunnerError("required file is unavailable")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise HotpotFamilyOutRunnerError(
            f"{field_name} must be lowercase SHA-256"
        )
    return value


def _absolute_no_symlink(path: str | Path, field_name: str) -> Path:
    candidate = Path(os.path.abspath(os.fspath(path)))
    for component in (*reversed(candidate.parents), candidate):
        if component.is_symlink():
            raise HotpotFamilyOutRunnerError(
                f"{field_name} contains a symlink component"
            )
    return candidate


def _new_root(path: str | Path, project: Path) -> Path:
    root = _absolute_no_symlink(path, "family-out execution root")
    if not root.parent.is_dir():
        raise HotpotFamilyOutRunnerError("execution root parent is unavailable")
    try:
        _assert_git_ignored_private_path(project=project, path=root, require_file=False)
    except Exception as exc:
        raise HotpotFamilyOutRunnerError(
            "execution root must be git-ignored"
        ) from exc
    return root


def root_binding_hash(path: str | Path, project: Path) -> str:
    return stable_hash({"absolute_execution_root": str(_new_root(path, project))})


def _write_json_exclusive(path: Path, payload: Mapping[str, Any], mode: int = 0o600) -> None:
    raw = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _read_json(path: str | Path, field_name: str) -> tuple[dict[str, Any], bytes]:
    candidate = _absolute_no_symlink(path, field_name)
    if not candidate.is_file():
        raise HotpotFamilyOutRunnerError(f"{field_name} is unavailable")
    raw = candidate.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HotpotFamilyOutRunnerError(f"{field_name} is invalid") from exc
    if not isinstance(payload, dict):
        raise HotpotFamilyOutRunnerError(f"{field_name} must be one object")
    return payload, raw


def _assert_public_safe(payload: Mapping[str, Any]) -> None:
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    forbidden = (
        '"corpus"',
        '"item_id"',
        '"paragraph_text"',
        '"private_pack_path"',
        '"question"',
        '"support_indices"',
    )
    if any(value in raw for value in forbidden):
        raise HotpotFamilyOutRunnerError("public artifact contains private content")


def _probe_contract(writable_root: str) -> tuple[str, ...]:
    return (
        str(BWRAP_PATH),
        "--unshare-net",
        "--die-with-parent",
        "--new-session",
        "--ro-bind",
        "/",
        "/",
        "--dev",
        "/dev",
        "--bind",
        writable_root,
        writable_root,
        "/bin/true",
    )


def _probe_contract_hash() -> str:
    return stable_hash(
        {"argv_without_binary": list(acquisition.BWRAP_PROBE_TEMPLATE_ARGS)}
    )


def _probe_bubblewrap() -> dict[str, Any]:
    if _sha256_file(BWRAP_PATH) != BWRAP_SHA256:
        raise HotpotFamilyOutRunnerError("bubblewrap binary drifted")
    version = subprocess.run(
        [str(BWRAP_PATH), "--version"], check=False, capture_output=True
    )
    version_text = version.stdout.decode("utf-8", errors="replace").strip()
    with tempfile.TemporaryDirectory(prefix="hotpot-bwrap-probe-", dir="/tmp") as root:
        completed = subprocess.run(
            list(_probe_contract(root)), check=False, capture_output=True
        )
    result = {
        "bwrap_file_sha256": BWRAP_SHA256,
        "bwrap_version": version_text,
        "probe_contract_sha256": _probe_contract_hash(),
        "probe_returncode": completed.returncode,
        "probe_stdout_sha256": _sha256_bytes(completed.stdout),
        "probe_stderr_sha256": _sha256_bytes(completed.stderr),
    }
    if version.returncode != 0 or version_text != BWRAP_VERSION:
        raise HotpotFamilyOutRunnerError("bubblewrap version drifted")
    if completed.returncode != 0:
        raise HotpotFamilyOutRunnerError(
            "bubblewrap capability unavailable; "
            f"stderr_sha256={result['probe_stderr_sha256']}"
        )
    return result


def build_capability_receipt(output_path: Path) -> dict[str, Any]:
    probe = _probe_bubblewrap()
    body: dict[str, Any] = {
        "schema": CAPABILITY_SCHEMA,
        "decision": "bubblewrap_network_namespace_available",
        **probe,
        "execution_context_requirement": (
            "formal_command_must_have_same_network_namespace_capability"
        ),
        "benchmark_rows_read": 0,
        "model_calls": 0,
        "network_connectivity_attempted": False,
        "performance_gate": False,
    }
    receipt = {**body, "receipt_sha256": stable_hash(body)}
    _assert_public_safe(receipt)
    _write_json_exclusive(output_path, receipt, mode=0o644)
    return receipt


def verify_capability_receipt(path: str | Path) -> tuple[dict[str, Any], bytes]:
    payload, raw = _read_json(path, "bubblewrap capability receipt")
    body = dict(payload)
    declared = _require_sha256(body.pop("receipt_sha256", None), "capability hash")
    expected_keys = {
        "benchmark_rows_read",
        "bwrap_file_sha256",
        "bwrap_version",
        "decision",
        "execution_context_requirement",
        "model_calls",
        "network_connectivity_attempted",
        "performance_gate",
        "probe_contract_sha256",
        "probe_returncode",
        "probe_stderr_sha256",
        "probe_stdout_sha256",
        "receipt_sha256",
        "schema",
    }
    if (
        set(payload) != expected_keys
        or payload.get("schema") != CAPABILITY_SCHEMA
        or payload.get("decision") != "bubblewrap_network_namespace_available"
        or stable_hash(body) != declared
        or payload.get("bwrap_file_sha256") != BWRAP_SHA256
        or payload.get("bwrap_version") != BWRAP_VERSION
        or payload.get("probe_contract_sha256")
        != _probe_contract_hash()
        or payload.get("probe_returncode") != 0
        or payload.get("benchmark_rows_read") != 0
        or payload.get("model_calls") != 0
        or payload.get("network_connectivity_attempted") is not False
        or payload.get("performance_gate") is not False
    ):
        raise HotpotFamilyOutRunnerError("capability receipt drifted")
    return payload, raw


def current_implementation_binding(project: Path) -> dict[str, Any]:
    value = acquisition.implementation_binding(project)
    return {
        "schema": IMPLEMENTATION_SCHEMA,
        "files": value["files"],
        "set_sha256": value["set_sha256"],
    }


def _load_acquisition(path: str | Path) -> tuple[dict[str, Any], bytes]:
    payload, raw = _read_json(path, "family-out acquisition receipt")
    body = dict(payload)
    declared = _require_sha256(body.pop("acquisition_sha256", None), "acquisition hash")
    source = payload.get("source")
    counts = payload.get("counts")
    commitments = payload.get("commitments")
    safety = payload.get("safety")
    if (
        set(payload)
        != {
            "acquisition_sha256",
            "acquisition_runtime",
            "commitments",
            "counts",
            "decision",
            "preregistration_sha256",
            "preregistration_custody",
            "prospective_ordering",
            "safety",
            "schema",
            "source",
        }
        or payload.get("schema") != acquisition.ACQUISITION_SCHEMA
        or payload.get("decision")
        != "fresh_family_out_pack_formed_measurement_not_authorized"
        or stable_hash(body) != declared
        or not isinstance(source, Mapping)
        or dict(source)
        != {
            "file_sha256": acquisition.SOURCE_SHA256,
            "file_size": acquisition.SOURCE_SIZE,
            "hf_repository_commit": acquisition.HF_REPOSITORY_COMMIT,
            "original_CMU_JSON_equivalence_claim": False,
            "row_count": acquisition.SOURCE_ROW_COUNT,
        }
        or not isinstance(counts, Mapping)
        or set(counts)
        != {
            "eligible_unique_id_rows",
            "selected_rows",
            "source_rows",
            "structurally_valid_rows",
        }
        or counts.get("source_rows") != acquisition.SOURCE_ROW_COUNT
        or counts.get("selected_rows") != acquisition.SAMPLE_COUNT
        or any(
            type(counts.get(key)) is not int
            for key in ("eligible_unique_id_rows", "structurally_valid_rows")
        )
        or not (
            acquisition.SAMPLE_COUNT
            <= counts.get("eligible_unique_id_rows", -1)
            <= counts.get("structurally_valid_rows", -1)
            <= acquisition.SOURCE_ROW_COUNT
        )
        or safety
        != {
            "measurement_executed": False,
            "model_calls": 0,
            "online_evaluator_calls": 0,
            "scores_computed": 0,
        }
        or payload.get("acquisition_runtime")
        != acquisition.acquisition_runtime_binding()
    ):
        raise HotpotFamilyOutRunnerError("acquisition receipt drifted")
    if not isinstance(commitments, Mapping):
        raise HotpotFamilyOutRunnerError("acquisition commitments are unavailable")
    if set(commitments) != {
        "item_commitment_set_sha256",
        "item_ids_persisted_publicly",
        "private_pack_file_sha256",
        "private_paths_persisted_publicly",
        "selection_secret_commitment_sha256",
    } or commitments.get("item_ids_persisted_publicly") is not False or commitments.get(
        "private_paths_persisted_publicly"
    ) is not False:
        raise HotpotFamilyOutRunnerError("acquisition custody contract drifted")
    _require_sha256(payload.get("preregistration_sha256"), "preregistration hash")
    custody = payload.get("preregistration_custody")
    ordering = payload.get("prospective_ordering")
    if (
        not isinstance(custody, Mapping)
        or set(custody)
        != {
            "preregistration_file_sha256",
            "preregistration_head_blob_sha256",
            "repository_commit",
        }
        or custody.get("preregistration_file_sha256")
        != custody.get("preregistration_head_blob_sha256")
        or not isinstance(custody.get("repository_commit"), str)
        or acquisition._GIT_COMMIT_RE.fullmatch(custody["repository_commit"]) is None
        or not isinstance(ordering, Mapping)
        or set(ordering)
        != {
            "acquisition_consumed_before_source_row_open",
            "acquisition_consumption_file_sha256",
            "acquisition_consumption_sha256",
            "preregistration_committed_before_source_row_open",
            "retry_replay_resample_authorized",
            "source_rows_opened_before_consumption",
        }
        or ordering.get("preregistration_committed_before_source_row_open") is not True
        or ordering.get("acquisition_consumed_before_source_row_open") is not True
        or ordering.get("source_rows_opened_before_consumption") != 0
        or ordering.get("retry_replay_resample_authorized") is not False
    ):
        raise HotpotFamilyOutRunnerError("prospective acquisition ordering drifted")
    for key in (
        "preregistration_file_sha256",
        "preregistration_head_blob_sha256",
    ):
        _require_sha256(custody.get(key), key)
    for key in (
        "acquisition_consumption_file_sha256",
        "acquisition_consumption_sha256",
    ):
        _require_sha256(ordering.get(key), key)
    for key in (
        "private_pack_file_sha256",
        "item_commitment_set_sha256",
        "selection_secret_commitment_sha256",
    ):
        _require_sha256(commitments.get(key), key)
    _assert_public_safe(payload)
    return payload, raw


def _p_program(
    *,
    project: Path,
    formation_receipt_path: str | Path,
    frozen_program_path: str | Path,
) -> tuple[TypedRetrievalProgram, dict[str, str]]:
    formation_path = _absolute_no_symlink(
        formation_receipt_path, "P formation receipt"
    )
    program_path = _absolute_no_symlink(frozen_program_path, "P frozen program")
    program, receipt, envelope = load_study_frozen_program(
        frozen_program_path=program_path,
        formation_receipt_path=formation_path,
        verify_live=True,
        implementation_root=project,
    )
    expected = stable_hash({"block": "F1"})
    if receipt.get("formation_block_id_hash") != expected:
        raise HotpotFamilyOutRunnerError("P was not formed on exact F1")
    binding = {
        "formation_receipt_file_sha256": _sha256_file(formation_path),
        "formation_receipt_hash": receipt["receipt_hash"],
        "frozen_program_file_sha256": _sha256_file(program_path),
        "frozen_program_envelope_hash": envelope["envelope_hash"],
        "program_hash": program.program_hash,
        "formed_on_block_id_hash": expected,
    }
    return program, binding


@dataclass(frozen=True)
class RuntimePaths:
    runtime_python: Path = field(repr=False)
    local_llm_model: Path = field(repr=False)
    local_embedding_model: Path = field(repr=False)
    base_binding_receipt_path: Path = field(repr=False)
    attestation_receipt_path: Path = field(repr=False)


def _runtime_paths(
    *,
    runtime_python: str | Path,
    local_llm_model: str | Path,
    local_embedding_model: str | Path,
    base_binding_receipt_path: str | Path,
    attestation_receipt_path: str | Path,
) -> RuntimePaths:
    value = RuntimePaths(
        runtime_python=Path(runtime_python).absolute(),
        local_llm_model=Path(local_llm_model).resolve(strict=True),
        local_embedding_model=Path(local_embedding_model).resolve(strict=True),
        base_binding_receipt_path=_absolute_no_symlink(
            base_binding_receipt_path, "base binding receipt"
        ),
        attestation_receipt_path=_absolute_no_symlink(
            attestation_receipt_path, "runtime attestation receipt"
        ),
    )
    if not value.runtime_python.is_file():
        raise HotpotFamilyOutRunnerError("runtime Python is unavailable")
    return value


def _prepare(project: Path, runtime: RuntimePaths) -> PreparedFormalRuntimeV2:
    return prepare_formal_runtime_v2(
        project_root=project,
        attestation_receipt_path=runtime.attestation_receipt_path,
        base_binding_receipt_path=runtime.base_binding_receipt_path,
        runtime_python=runtime.runtime_python,
        local_llm_model=runtime.local_llm_model,
        local_embedding_model=runtime.local_embedding_model,
    )


def _runtime_binding(prepared: PreparedFormalRuntimeV2, runtime: RuntimePaths) -> dict[str, Any]:
    safe = prepared.safe_binding
    return {
        "prepared_safe_binding": safe,
        "base_binding_file_sha256": _sha256_file(runtime.base_binding_receipt_path),
        "attestation_file_sha256": _sha256_file(runtime.attestation_receipt_path),
        "fresh_preflight_before_authorization": True,
        "fresh_postflight_before_scoring": True,
    }


def build_pre_run_freeze(
    *,
    project_root: str | Path,
    acquisition_receipt_path: str | Path,
    p_formation_receipt_path: str | Path,
    p_frozen_program_path: str | Path,
    capability_receipt_path: str | Path,
    runtime_python: str | Path,
    local_llm_model: str | Path,
    local_embedding_model: str | Path,
    base_binding_receipt_path: str | Path,
    attestation_receipt_path: str | Path,
    execution_root: str | Path,
    authorization_hash: str,
    output_path: str | Path,
) -> dict[str, Any]:
    project = Path(project_root).resolve(strict=True)
    acquisition_receipt, acquisition_raw = _load_acquisition(
        acquisition_receipt_path
    )
    _program, p_binding = _p_program(
        project=project,
        formation_receipt_path=p_formation_receipt_path,
        frozen_program_path=p_frozen_program_path,
    )
    capability, capability_raw = verify_capability_receipt(capability_receipt_path)
    runtime = _runtime_paths(
        runtime_python=runtime_python,
        local_llm_model=local_llm_model,
        local_embedding_model=local_embedding_model,
        base_binding_receipt_path=base_binding_receipt_path,
        attestation_receipt_path=attestation_receipt_path,
    )
    prepared = _prepare(project, runtime)
    commitments = acquisition_receipt["commitments"]
    body: dict[str, Any] = {
        "schema": FREEZE_SCHEMA,
        "decision": "authorize_exact_Hotpot_family_out_retrieval_once",
        "implementation": current_implementation_binding(project),
        "authorization_hash": _require_sha256(
            authorization_hash, "execution authorization"
        ),
        "execution_root_hash": root_binding_hash(execution_root, project),
        "source_binding": {
            "acquisition_file_sha256": _sha256_bytes(acquisition_raw),
            "acquisition_sha256": acquisition_receipt["acquisition_sha256"],
            "private_pack_file_sha256": commitments["private_pack_file_sha256"],
            "item_commitment_set_sha256": commitments[
                "item_commitment_set_sha256"
            ],
            "item_count": acquisition.SAMPLE_COUNT,
            "source_file_sha256": acquisition.SOURCE_SHA256,
            "hf_repository_commit": acquisition.HF_REPOSITORY_COMMIT,
        },
        "p_program_binding": {
            **p_binding,
            "adaptation_or_reformation_on_HotpotQA": False,
        },
        "capability_binding": {
            "file_sha256": _sha256_bytes(capability_raw),
            "receipt_sha256": capability["receipt_sha256"],
            "bwrap_file_sha256": capability["bwrap_file_sha256"],
            "probe_contract_sha256": capability["probe_contract_sha256"],
            "fresh_probe_required_before_authorization": True,
        },
        "runtime_binding": _runtime_binding(prepared, runtime),
        "execution_contract": {
            "arms": list(ARM_IDS),
            "item_count": acquisition.SAMPLE_COUNT,
            "top_k": acquisition.TOP_K,
            "work_unit_count": WORK_UNIT_COUNT,
            "maximum_concurrency": MAXIMUM_CONCURRENCY,
            "formal_entry": "clean_python_module_cli_v1",
            "all_terminals_joined_before_support_scoring": True,
            "fresh_runtime_postflight_before_support_scoring": True,
            "generator_calls": 0,
            "external_network_calls": 0,
            "online_evaluator_calls": 0,
            "retries": 0,
            "replays": 0,
            "resamples": 0,
        },
        "ordering": {
            "private_pack_rows_read_while_freezing": 0,
            "support_labels_read_while_freezing": 0,
            "freeze_complete_before_private_pack_open": True,
        },
        "raw_content_persisted": False,
    }
    freeze = {**body, "freeze_hash": stable_hash(body)}
    _assert_public_safe(freeze)
    output = _absolute_no_symlink(output_path, "family-out freeze output")
    if output.exists():
        raise HotpotFamilyOutRunnerError("freeze output already exists")
    _write_json_exclusive(output, freeze, mode=0o644)
    return freeze


def _load_freeze(path: str | Path) -> tuple[dict[str, Any], str]:
    payload, raw = _read_json(path, "family-out pre-run freeze")
    body = dict(payload)
    declared = _require_sha256(body.pop("freeze_hash", None), "freeze hash")
    expected_contract = {
        "arms": list(ARM_IDS),
        "item_count": acquisition.SAMPLE_COUNT,
        "top_k": acquisition.TOP_K,
        "work_unit_count": WORK_UNIT_COUNT,
        "maximum_concurrency": MAXIMUM_CONCURRENCY,
        "formal_entry": "clean_python_module_cli_v1",
        "all_terminals_joined_before_support_scoring": True,
        "fresh_runtime_postflight_before_support_scoring": True,
        "generator_calls": 0,
        "external_network_calls": 0,
        "online_evaluator_calls": 0,
        "retries": 0,
        "replays": 0,
        "resamples": 0,
    }
    expected_ordering = {
        "private_pack_rows_read_while_freezing": 0,
        "support_labels_read_while_freezing": 0,
        "freeze_complete_before_private_pack_open": True,
    }
    if (
        set(payload)
        != {
            "authorization_hash",
            "capability_binding",
            "decision",
            "execution_contract",
            "execution_root_hash",
            "freeze_hash",
            "implementation",
            "ordering",
            "p_program_binding",
            "raw_content_persisted",
            "runtime_binding",
            "schema",
            "source_binding",
        }
        or payload.get("schema") != FREEZE_SCHEMA
        or payload.get("decision")
        != "authorize_exact_Hotpot_family_out_retrieval_once"
        or stable_hash(body) != declared
        or payload.get("raw_content_persisted") is not False
        or payload.get("execution_contract") != expected_contract
        or payload.get("ordering") != expected_ordering
    ):
        raise HotpotFamilyOutRunnerError("pre-run freeze drifted")
    _require_sha256(payload.get("authorization_hash"), "execution authorization")
    _require_sha256(payload.get("execution_root_hash"), "execution root hash")
    _assert_public_safe(payload)
    return payload, _sha256_bytes(raw)


@dataclass(frozen=True)
class HotpotItem:
    item_id: str = field(repr=False)
    question: str = field(repr=False)
    corpus: tuple[RetrievalParagraph, ...] = field(repr=False)
    support_indices: tuple[int, ...] = field(repr=False)
    item_commitment_sha256: str

    @property
    def item_id_hash(self) -> str:
        return stable_hash({"item_id": self.item_id})

    def retrieval_view(self) -> "HotpotRetrievalItem":
        return HotpotRetrievalItem(
            question=self.question,
            corpus=self.corpus,
        )


@dataclass(frozen=True)
class HotpotRetrievalItem:
    question: str = field(repr=False)
    corpus: tuple[RetrievalParagraph, ...] = field(repr=False)

    def hipporag_paragraphs(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "idx": paragraph.idx,
                "title": paragraph.title,
                "paragraph_text": paragraph.text,
            }
            for paragraph in self.corpus
        )


def _load_private_pack(
    *,
    project: Path,
    path: str | Path,
    expected_file_sha256: str,
    expected_item_set_sha256: str,
) -> tuple[HotpotItem, ...]:
    private = _assert_git_ignored_private_path(
        project=project, path=Path(path), require_file=True
    )
    if _sha256_file(private) != expected_file_sha256:
        raise HotpotFamilyOutRunnerError("private pack file hash drifted")
    items: list[HotpotItem] = []
    commitments: list[str] = []
    for line in private.read_bytes().splitlines():
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HotpotFamilyOutRunnerError("private pack row is invalid") from exc
        if not isinstance(row, dict) or set(row) != {
            "corpus",
            "item_id",
            "question",
            "schema",
            "source_row_sha256",
            "support_indices",
        }:
            raise HotpotFamilyOutRunnerError("private pack row schema drifted")
        if acquisition._canonical_bytes(row) != line:
            raise HotpotFamilyOutRunnerError("private pack row is not canonical")
        if row.get("schema") != acquisition.PRIVATE_ROW_SCHEMA:
            raise HotpotFamilyOutRunnerError("private pack version drifted")
        corpus_raw = row.get("corpus")
        supports_raw = row.get("support_indices")
        if not isinstance(corpus_raw, list) or not isinstance(supports_raw, list):
            raise HotpotFamilyOutRunnerError("private pack labels are malformed")
        corpus: list[RetrievalParagraph] = []
        observed_supports: list[int] = []
        for index, paragraph in enumerate(corpus_raw):
            if (
                not isinstance(paragraph, Mapping)
                or set(paragraph) != {"idx", "is_supporting", "text", "title"}
                or paragraph.get("idx") != index
                or type(paragraph.get("is_supporting")) is not bool
                or not isinstance(paragraph.get("title"), str)
                or not isinstance(paragraph.get("text"), str)
            ):
                raise HotpotFamilyOutRunnerError("private corpus drifted")
            corpus.append(
                RetrievalParagraph(
                    idx=index,
                    title=str(paragraph["title"]),
                    text=str(paragraph["text"]),
                )
            )
            if paragraph["is_supporting"]:
                observed_supports.append(index)
        if (
            len(corpus) < acquisition.TOP_K
            or supports_raw != observed_supports
            or len(observed_supports) != 2
            or not isinstance(row.get("item_id"), str)
            or not isinstance(row.get("question"), str)
        ):
            raise HotpotFamilyOutRunnerError("private support contract drifted")
        commitment = stable_hash(row)
        commitments.append(commitment)
        items.append(
            HotpotItem(
                item_id=row["item_id"],
                question=row["question"],
                corpus=tuple(corpus),
                support_indices=tuple(observed_supports),
                item_commitment_sha256=commitment,
            )
        )
    if (
        len(items) != acquisition.SAMPLE_COUNT
        or len({item.item_id_hash for item in items}) != len(items)
        or stable_hash(commitments) != expected_item_set_sha256
    ):
        raise HotpotFamilyOutRunnerError("private pack commitment set drifted")
    return tuple(items)


def _raw(item: HotpotRetrievalItem) -> tuple[int, ...]:
    return tuple(paragraph.idx for paragraph in item.corpus[: acquisition.TOP_K])


def _p(program: TypedRetrievalProgram, item: HotpotRetrievalItem) -> tuple[int, ...]:
    return typed_retrieve(program, item.question, item.corpus)


def _official(
    runtime: PreparedFormalRuntimeV2,
    item: HotpotRetrievalItem,
    work_root: Path,
) -> tuple[int, ...]:
    return runtime.retrieve(
        question=item.question,
        paragraphs=item.hipporag_paragraphs(),
        work_root=work_root,
    )


def _validate_ranking(value: Sequence[int], item: HotpotRetrievalItem) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise HotpotFamilyOutRunnerError("retrieval output is not an index sequence")
    ranking = tuple(value)
    if (
        len(ranking) != acquisition.TOP_K
        or len(set(ranking)) != acquisition.TOP_K
        or any(type(index) is not int or not 0 <= index < len(item.corpus) for index in ranking)
    ):
        raise HotpotFamilyOutRunnerError("retrieval output violates top-five contract")
    return ranking


def _score(
    items: Sequence[HotpotItem], rankings: Mapping[tuple[int, str], tuple[int, ...]]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for ordinal, item in enumerate(items):
        supports = frozenset(item.support_indices)
        arm_rows = {}
        for arm in ARM_IDS:
            ranking = rankings[(ordinal, arm)]
            arm_rows[arm] = {
                "support_hits": len(supports.intersection(ranking)),
                "ranking_hash": stable_hash({"retrieved_indices": list(ranking)}),
            }
        rows.append(
            {
                "item_id_hash": item.item_id_hash,
                "support_total": len(supports),
                "arms": arm_rows,
            }
        )
    support_total = sum(row["support_total"] for row in rows)
    metrics = {}
    for arm in ARM_IDS:
        hits = sum(row["arms"][arm]["support_hits"] for row in rows)
        metrics[arm] = {
            "support_hit_count": hits,
            "support_total": support_total,
            "support_recall_at_5": float(Fraction(hits, support_total)),
            "items_with_any_support_hit": sum(row["arms"][arm]["support_hits"] > 0 for row in rows),
            "ranking_score_closure_hash": stable_hash(
                [
                    {
                        "item_id_hash": row["item_id_hash"],
                        "support_total": row["support_total"],
                        **row["arms"][arm],
                    }
                    for row in rows
                ]
            ),
        }
    def paired(left: str, right: str) -> dict[str, Any]:
        deltas = [
            row["arms"][left]["support_hits"]
            - row["arms"][right]["support_hits"]
            for row in rows
        ]
        net = sum(deltas)
        return {
            "left_arm": left,
            "right_arm": right,
            "net_support_hit_count": net,
            "support_recall_delta": float(Fraction(net, support_total)),
            "gain_item_count": sum(value > 0 for value in deltas),
            "harm_item_count": sum(value < 0 for value in deltas),
            "tie_item_count": sum(value == 0 for value in deltas),
        }
    return {
        "primary_metric": "offline_micro_support_recall_at_5",
        "arm_metrics": metrics,
        "paired_P_minus_RAW": paired("frozen_P", "canonical_RAW"),
        "paired_P_minus_official_HippoRAG": paired(
            "frozen_P", "official_HippoRAG"
        ),
        "new_family": "HotpotQA_distractor_validation_conversion",
        "P_adapted_or_reformed_on_new_family": False,
        "original_CMU_JSON_equivalence_claim": False,
        "scored_item_closure_hash": stable_hash(rows),
        "raw_content_persisted": False,
    }


def execute_formal(
    *,
    project_root: str | Path,
    pre_run_freeze_path: str | Path,
    acquisition_receipt_path: str | Path,
    private_pack_path: str | Path,
    p_formation_receipt_path: str | Path,
    p_frozen_program_path: str | Path,
    capability_receipt_path: str | Path,
    runtime_python: str | Path,
    local_llm_model: str | Path,
    local_embedding_model: str | Path,
    base_binding_receipt_path: str | Path,
    attestation_receipt_path: str | Path,
    execution_root: str | Path,
) -> dict[str, Any]:
    if _CLEAN_MODULE_CLI_ACTIVE is not True:
        raise HotpotFamilyOutRunnerError(
            "formal family-out execution is available only through clean module CLI"
        )
    project = Path(project_root).resolve(strict=True)
    freeze, freeze_file_hash = _load_freeze(pre_run_freeze_path)
    if freeze.get("implementation") != current_implementation_binding(project):
        raise HotpotFamilyOutRunnerError("live family-out implementation drifted")
    root = _new_root(execution_root, project)
    if freeze.get("execution_root_hash") != root_binding_hash(root, project):
        raise HotpotFamilyOutRunnerError("execution root binding drifted")
    if root.exists():
        raise HotpotFamilyOutRunnerError("fresh root exists; replay is forbidden")
    acquisition_receipt, acquisition_raw = _load_acquisition(acquisition_receipt_path)
    commitments = acquisition_receipt["commitments"]
    expected_source = {
        "acquisition_file_sha256": _sha256_bytes(acquisition_raw),
        "acquisition_sha256": acquisition_receipt["acquisition_sha256"],
        "private_pack_file_sha256": commitments["private_pack_file_sha256"],
        "item_commitment_set_sha256": commitments["item_commitment_set_sha256"],
        "item_count": acquisition.SAMPLE_COUNT,
        "source_file_sha256": acquisition.SOURCE_SHA256,
        "hf_repository_commit": acquisition.HF_REPOSITORY_COMMIT,
    }
    if freeze.get("source_binding") != expected_source:
        raise HotpotFamilyOutRunnerError("family-out source binding drifted")
    program, p_binding = _p_program(
        project=project,
        formation_receipt_path=p_formation_receipt_path,
        frozen_program_path=p_frozen_program_path,
    )
    if freeze.get("p_program_binding") != {
        **p_binding,
        "adaptation_or_reformation_on_HotpotQA": False,
    }:
        raise HotpotFamilyOutRunnerError("frozen P binding drifted")
    capability, capability_raw = verify_capability_receipt(capability_receipt_path)
    expected_capability = {
        "file_sha256": _sha256_bytes(capability_raw),
        "receipt_sha256": capability["receipt_sha256"],
        "bwrap_file_sha256": capability["bwrap_file_sha256"],
        "probe_contract_sha256": capability["probe_contract_sha256"],
        "fresh_probe_required_before_authorization": True,
    }
    if freeze.get("capability_binding") != expected_capability:
        raise HotpotFamilyOutRunnerError("capability binding drifted")
    live_probe = _probe_bubblewrap()
    if (
        live_probe["bwrap_file_sha256"] != capability["bwrap_file_sha256"]
        or live_probe["probe_contract_sha256"]
        != capability["probe_contract_sha256"]
        or live_probe["probe_returncode"] != 0
    ):
        raise HotpotFamilyOutRunnerError(
            "fresh bubblewrap preflight differs from frozen capability"
        )
    runtime = _runtime_paths(
        runtime_python=runtime_python,
        local_llm_model=local_llm_model,
        local_embedding_model=local_embedding_model,
        base_binding_receipt_path=base_binding_receipt_path,
        attestation_receipt_path=attestation_receipt_path,
    )
    prepared = _prepare(project, runtime)
    safe_runtime = prepared.safe_binding
    if freeze.get("runtime_binding") != _runtime_binding(prepared, runtime):
        raise HotpotFamilyOutRunnerError("runtime binding drifted")
    try:
        os.mkdir(root, 0o700)
    except FileExistsError as exc:
        raise HotpotFamilyOutRunnerError("fresh root exists; replay is forbidden") from exc
    stage = "authorization_consumption"
    attempted = 0
    completed = 0
    counter_lock = threading.Lock()
    start_barrier = threading.Barrier(WORK_UNIT_COUNT)
    try:
        consumption_body = {
            "schema": f"{VERSION}_authorization_consumption",
            "authorization_hash": freeze["authorization_hash"],
            "freeze_hash": freeze["freeze_hash"],
            "freeze_file_sha256": freeze_file_hash,
            "execution_root_hash": freeze["execution_root_hash"],
            "replay_authorized": False,
            "raw_content_persisted": False,
        }
        consumption = {
            **consumption_body,
            "consumption_hash": stable_hash(consumption_body),
        }
        _write_json_exclusive(root / CONSUMPTION_FILENAME, consumption)
        stage = "exact_private_pack_open_after_freeze"
        items = _load_private_pack(
            project=project,
            path=private_pack_path,
            expected_file_sha256=commitments["private_pack_file_sha256"],
            expected_item_set_sha256=commitments["item_commitment_set_sha256"],
        )
        work_units = tuple(
            (ordinal, arm, item.retrieval_view())
            for ordinal, item in enumerate(items)
            for arm in ARM_IDS
        )
        stage = "retrieval_execution"

        def run_one(unit):
            nonlocal attempted, completed
            ordinal, arm, item = unit
            with counter_lock:
                attempted += 1
            try:
                start_barrier.wait(timeout=60)
            except threading.BrokenBarrierError as exc:
                raise HotpotFamilyOutRunnerError(
                    "36-way retrieval start barrier did not close"
                ) from exc
            if arm == "canonical_RAW":
                value = _raw(item)
            elif arm == "frozen_P":
                value = _p(program, item)
            elif arm == "official_HippoRAG":
                value = _official(
                    prepared,
                    item,
                    root / f"official_item_{ordinal:02d}",
                )
            else:  # pragma: no cover
                raise HotpotFamilyOutRunnerError("unknown family-out arm")
            ranking = _validate_ranking(value, item)
            with counter_lock:
                completed += 1
            return (ordinal, arm), ranking

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=MAXIMUM_CONCURRENCY,
            thread_name_prefix="hotpot-family-out",
        ) as executor:
            futures = [executor.submit(run_one, unit) for unit in work_units]
            terminal_rows = [future.result() for future in futures]
        if attempted != WORK_UNIT_COUNT or completed != WORK_UNIT_COUNT:
            raise HotpotFamilyOutRunnerError("retrieval terminal closure is incomplete")
        rankings = dict(terminal_rows)
        if len(rankings) != WORK_UNIT_COUNT:
            raise HotpotFamilyOutRunnerError("retrieval keys are not one-to-one")
        stage = "fresh_runtime_postflight_before_scoring"
        postflight = prepared.fresh_reverify()
        if postflight != safe_runtime:
            raise HotpotFamilyOutRunnerError("runtime postflight drifted")
        stage = "offline_support_scoring_after_join"
        measurement = _score(items, rankings)
        ranking_receipts = [
            {
                "ordinal_hash": stable_hash({"ordinal": ordinal}),
                "arm_id": arm,
                "ranking_hash": stable_hash({"retrieved_indices": list(ranking)}),
            }
            for (ordinal, arm), ranking in sorted(rankings.items())
        ]
        report_body: dict[str, Any] = {
            "schema": REPORT_SCHEMA,
            "valid": True,
            "freeze_hash": freeze["freeze_hash"],
            "freeze_file_sha256": freeze_file_hash,
            "measurement": measurement,
            "execution": {
                "arm_ids": list(ARM_IDS),
                "item_count": len(items),
                "work_unit_count": WORK_UNIT_COUNT,
                "retrieval_call_count": attempted,
                "retrieval_terminal_count": completed,
                "configured_maximum_concurrency": MAXIMUM_CONCURRENCY,
                "observed_start_barrier_party_count": start_barrier.parties,
                "all_work_units_released_from_single_start_barrier": True,
                "all_terminals_joined_before_support_scoring": True,
                "ranking_receipt_set_sha256": stable_hash(ranking_receipts),
                "generator_calls": 0,
                "external_network_calls": 0,
                "online_evaluator_calls": 0,
                "retries": 0,
                "replays": 0,
                "resamples": 0,
            },
            "runtime": {
                "capability_receipt_sha256": capability["receipt_sha256"],
                "fresh_bubblewrap_preflight_before_authorization": True,
                "attestation_receipt_sha256": safe_runtime[
                    "attestation_receipt_sha256"
                ],
                "official_arm_terminal_count": acquisition.SAMPLE_COUNT,
                "postflight_fresh_filesystem_attestation": True,
                "postflight_binding_sha256": postflight["binding_sha256"],
            },
            "sealed_or_test_content_accessed": False,
            "raw_content_persisted": False,
        }
        report = {**report_body, "report_hash": stable_hash(report_body)}
        _assert_public_safe(report)
        stage = "aggregate_report_persistence"
        report_path = root / REPORT_FILENAME
        _write_json_exclusive(report_path, report)
        persisted, _persisted_raw = _read_json(
            report_path, "persisted family-out aggregate report"
        )
        persisted_body = dict(persisted)
        persisted_hash = persisted_body.pop("report_hash", None)
        if persisted != report or stable_hash(persisted_body) != persisted_hash:
            raise HotpotFamilyOutRunnerError("persisted family-out report drifted")
        return persisted
    except Exception as exc:
        failure_body = {
            "schema": FAILURE_SCHEMA,
            "valid": False,
            "freeze_hash": freeze["freeze_hash"],
            "failure_stage": stage,
            "error_type_sha256": stable_hash({"error_type": type(exc).__name__}),
            "authorization_consumed": (root / CONSUMPTION_FILENAME).is_file(),
            "retrieval_work_unit_count": WORK_UNIT_COUNT,
            "retrieval_attempt_count": attempted,
            "retrieval_terminal_count": completed,
            "retries": 0,
            "replays": 0,
            "resamples": 0,
            "replay_authorized": False,
            "raw_content_persisted": False,
        }
        failure = {**failure_body, "failure_hash": stable_hash(failure_body)}
        try:
            _write_json_exclusive(root / FAILURE_FILENAME, failure)
        except Exception:
            pass
        raise HotpotFamilyOutRunnerError(
            "formal family-out run failed and cannot be replayed"
        ) from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    capability = sub.add_parser("capability")
    freeze = sub.add_parser("freeze")
    execute = sub.add_parser("execute")
    capability.add_argument("--output", type=Path, required=True)
    for command in (freeze, execute):
        command.add_argument("--project-root", type=Path, required=True)
        command.add_argument("--acquisition-receipt", type=Path, required=True)
        command.add_argument("--p-formation-receipt", type=Path, required=True)
        command.add_argument("--p-frozen-program", type=Path, required=True)
        command.add_argument("--capability-receipt", type=Path, required=True)
        command.add_argument("--runtime-python", type=Path, required=True)
        command.add_argument("--local-llm-model", type=Path, required=True)
        command.add_argument("--local-embedding-model", type=Path, required=True)
        command.add_argument("--base-binding-receipt", type=Path, required=True)
        command.add_argument("--attestation-receipt", type=Path, required=True)
        command.add_argument("--execution-root", type=Path, required=True)
    freeze.add_argument("--authorization-hash", required=True)
    freeze.add_argument("--output", type=Path, required=True)
    execute.add_argument("--pre-run-freeze", type=Path, required=True)
    execute.add_argument("--private-pack", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "capability":
        build_capability_receipt(arguments.output)
        return 0
    common = {
        "project_root": arguments.project_root,
        "acquisition_receipt_path": arguments.acquisition_receipt,
        "p_formation_receipt_path": arguments.p_formation_receipt,
        "p_frozen_program_path": arguments.p_frozen_program,
        "capability_receipt_path": arguments.capability_receipt,
        "runtime_python": arguments.runtime_python,
        "local_llm_model": arguments.local_llm_model,
        "local_embedding_model": arguments.local_embedding_model,
        "base_binding_receipt_path": arguments.base_binding_receipt,
        "attestation_receipt_path": arguments.attestation_receipt,
        "execution_root": arguments.execution_root,
    }
    if arguments.command == "freeze":
        build_pre_run_freeze(
            **common,
            authorization_hash=arguments.authorization_hash,
            output_path=arguments.output,
        )
        return 0
    global _CLEAN_MODULE_CLI_ACTIVE
    _CLEAN_MODULE_CLI_ACTIVE = True
    try:
        execute_formal(
            **common,
            pre_run_freeze_path=arguments.pre_run_freeze,
            private_pack_path=arguments.private_pack,
        )
    finally:
        _CLEAN_MODULE_CLI_ACTIVE = False
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
