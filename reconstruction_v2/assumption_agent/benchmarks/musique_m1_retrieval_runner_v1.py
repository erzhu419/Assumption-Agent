"""One-shot generation-one retrieval-only measurement on exact MuSiQue M1.

The pre-run freeze is built without opening M1.  Formal entry verifies that
self-hashed freeze, the F1-formed typed program, and the prospective v2
official-HippoRAG filesystem attestation before atomically consuming a fresh
execution root.  Only then is the exact M1 file opened.

All ``3 * n`` canonical-RAW, frozen-P, and official-HippoRAG work units are
submitted to one maximum-width executor.  Retrieval callables receive a
gold-free item view, and support scoring begins only after every terminal has
joined.  The terminal report contains only
aggregate counts and hashes, and applies the predeclared disposition ``promote
P iff its net support-hit count versus canonical RAW is strictly positive``.
There is no generator, network judge, online evaluator, retry, replay, or
resample surface.  Formal entry has no injectable result or operator argument;
tests may monkeypatch the module's private execution functions.
"""

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
import threading
from typing import Any, Mapping, Sequence

from ..models import stable_hash
from .musique_recursive_study_blocks_v1 import (
    RetrievalStudyItem,
    StudyItem,
    load_measurement_block_after_freeze,
    load_study_acquisition_binding,
    load_study_frozen_program,
)
from .musique_formal_runtime_binding_v2 import (
    PreparedFormalRuntimeV2,
    prepare_formal_runtime_v2,
)
from .musique_typed_retriever_formation_v1 import (
    TOP_K,
    TypedRetrievalProgram,
    retrieve as typed_retrieve,
)
RUNNER_VERSION = "musique_generation_one_m1_retrieval_only_runner_v1"
FREEZE_SCHEMA = "musique_generation_one_m1_pre_run_freeze_v1"
REPORT_SCHEMA = "musique_generation_one_m1_aggregate_report_v1"
FAILURE_SCHEMA = "musique_generation_one_m1_failure_v1"
ARM_IDS = ("canonical_RAW", "frozen_P", "official_HippoRAG")
M1_ITEM_COUNT = 12
WORK_UNIT_COUNT = len(ARM_IDS) * M1_ITEM_COUNT
MAXIMUM_CONCURRENCY = WORK_UNIT_COUNT
PROMOTION_POLICY = (
    "promote_frozen_P_iff_total_support_hits_P_minus_canonical_RAW_is_strictly_positive_v1"
)
FREEZE_DECISION = "authorize_exact_M1_retrieval_only_once_after_full_pre_run_freeze"
CONSUMPTION_FILENAME = "m1.execution.authorization.consumed.json"
REPORT_FILENAME = "m1.aggregate.report.json"
FAILURE_FILENAME = "m1.failure.json"
IMPLEMENTATION_SCHEMA = "musique_generation_one_m1_implementation_v1"
_CLEAN_MODULE_CLI_ACTIVE = False
IMPLEMENTATION_RELATIVE_FILES = (
    "assumption_agent/benchmarks/musique_m1_retrieval_runner_v1.py",
    "assumption_agent/benchmarks/musique_formal_runtime_binding_v2.py",
    "assumption_agent/benchmarks/musique_recursive_study_blocks_v1.py",
    "assumption_agent/benchmarks/musique_typed_retriever_formation_v1.py",
    "assumption_agent/models.py",
    "replication_runtime/musique_official_hipporag_v1/adapter_v2.py",
    "replication_runtime/musique_official_hipporag_v1/contract.py",
    "replication_runtime/musique_official_hipporag_v1/runtime_attestation_v2.py",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class MuSiQueM1RunnerError(RuntimeError):
    """The prospective M1 retrieval measurement failed closed."""


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise MuSiQueM1RunnerError("required frozen file is unavailable")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise MuSiQueM1RunnerError(
            f"{field_name} must be a lowercase SHA-256 digest"
        )
    return value


def _absolute_no_symlink(path: str | Path, field_name: str) -> Path:
    candidate = Path(os.path.abspath(os.fspath(path)))
    for component in (*reversed(candidate.parents), candidate):
        if component.is_symlink():
            raise MuSiQueM1RunnerError(
                f"{field_name} contains a symlink component"
            )
    return candidate


def _canonical_new_root(path: str | Path) -> Path:
    candidate = _absolute_no_symlink(path, "M1 execution root")
    try:
        parent = candidate.parent.resolve(strict=True)
    except OSError as exc:
        raise MuSiQueM1RunnerError(
            "M1 execution root parent is unavailable"
        ) from exc
    if not parent.is_dir():
        raise MuSiQueM1RunnerError(
            "M1 execution root parent is not a directory"
        )
    return parent / candidate.name


def _root_binding_hash(path: str | Path) -> str:
    return stable_hash(
        {"absolute_execution_root": str(_canonical_new_root(path))}
    )


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    raw = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        indent=2,
    ).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _read_json_object(path: Path, field_name: str) -> tuple[dict[str, Any], bytes]:
    path = _absolute_no_symlink(path, field_name)
    if not path.is_file():
        raise MuSiQueM1RunnerError(f"{field_name} is unavailable")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MuSiQueM1RunnerError(f"{field_name} is invalid") from exc
    if not isinstance(payload, dict):
        raise MuSiQueM1RunnerError(f"{field_name} must contain one object")
    return payload, raw


def current_m1_implementation_binding(
    project_root: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve(strict=True)
    rows: list[dict[str, str]] = []
    for relative in IMPLEMENTATION_RELATIVE_FILES:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise MuSiQueM1RunnerError(
                f"M1 implementation file is missing: {relative}"
            )
        rows.append({"path": relative, "sha256": _sha256_file(path)})
    return {
        "schema": IMPLEMENTATION_SCHEMA,
        "files": rows,
        "set_sha256": stable_hash(rows),
    }


def _validate_implementation_binding(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "files",
        "set_sha256",
    }:
        raise MuSiQueM1RunnerError("M1 implementation binding is malformed")
    if value.get("schema") != IMPLEMENTATION_SCHEMA:
        raise MuSiQueM1RunnerError("M1 implementation schema drifted")
    files = value.get("files")
    if not isinstance(files, list) or len(files) != len(
        IMPLEMENTATION_RELATIVE_FILES
    ):
        raise MuSiQueM1RunnerError("M1 implementation set drifted")
    rows = []
    for expected, row in zip(IMPLEMENTATION_RELATIVE_FILES, files):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"path", "sha256"}
            or row.get("path") != expected
        ):
            raise MuSiQueM1RunnerError("M1 implementation row drifted")
        rows.append(
            {
                "path": expected,
                "sha256": _require_sha256(
                    row.get("sha256"), "M1 implementation file hash"
                ),
            }
        )
    set_hash = _require_sha256(
        value.get("set_sha256"), "M1 implementation set hash"
    )
    if stable_hash(rows) != set_hash:
        raise MuSiQueM1RunnerError("M1 implementation hash drifted")
    return {"schema": IMPLEMENTATION_SCHEMA, "files": rows, "set_sha256": set_hash}


@dataclass(frozen=True)
class OfficialRuntimePaths:
    runtime_python: Path = field(repr=False)
    local_llm_model: Path = field(repr=False)
    local_embedding_model: Path = field(repr=False)
    base_binding_receipt_path: Path = field(repr=False)
    attestation_receipt_path: Path = field(repr=False)

    def verify(self) -> None:
        if not self.runtime_python.absolute().is_file():
            raise MuSiQueM1RunnerError("runtime Python is unavailable")
        for path in (self.local_llm_model, self.local_embedding_model):
            if not path.resolve(strict=True).is_dir():
                raise MuSiQueM1RunnerError("local runtime asset is unavailable")
        for path in (
            self.base_binding_receipt_path,
            self.attestation_receipt_path,
        ):
            if path.is_symlink() or not path.is_file():
                raise MuSiQueM1RunnerError(
                    "official runtime receipt is unavailable"
                )


def _runtime_paths(
    *,
    runtime_python: str | Path,
    local_llm_model: str | Path,
    local_embedding_model: str | Path,
    base_binding_receipt_path: str | Path,
    attestation_receipt_path: str | Path,
) -> OfficialRuntimePaths:
    value = OfficialRuntimePaths(
        runtime_python=Path(runtime_python).absolute(),
        local_llm_model=Path(local_llm_model).resolve(strict=True),
        local_embedding_model=Path(local_embedding_model).resolve(strict=True),
        base_binding_receipt_path=_absolute_no_symlink(
            base_binding_receipt_path, "base binding receipt"
        ),
        attestation_receipt_path=_absolute_no_symlink(
            attestation_receipt_path, "v2 attestation receipt"
        ),
    )
    value.verify()
    return value


def _prepare_runtime(
    *,
    project: Path,
    runtime: OfficialRuntimePaths,
) -> PreparedFormalRuntimeV2:
    return prepare_formal_runtime_v2(
        project_root=project,
        attestation_receipt_path=runtime.attestation_receipt_path,
        base_binding_receipt_path=runtime.base_binding_receipt_path,
        runtime_python=runtime.runtime_python,
        local_llm_model=runtime.local_llm_model,
        local_embedding_model=runtime.local_embedding_model,
    )


def build_m1_pre_run_freeze(
    *,
    project_root: str | Path,
    acquisition_receipt_path: str | Path,
    formation_receipt_path: str | Path,
    frozen_program_path: str | Path,
    runtime_python: str | Path,
    local_llm_model: str | Path,
    local_embedding_model: str | Path,
    base_binding_receipt_path: str | Path,
    attestation_receipt_path: str | Path,
    execution_root: str | Path,
    authorization_hash: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Freeze M1 without accepting or reading an M1 path."""

    project = Path(project_root).resolve(strict=True)
    acquisition_path = _absolute_no_symlink(
        acquisition_receipt_path, "study acquisition receipt"
    )
    formation_path = _absolute_no_symlink(
        formation_receipt_path, "F1 formation receipt"
    )
    program_path = _absolute_no_symlink(
        frozen_program_path, "F1 frozen program"
    )
    binding = load_study_acquisition_binding(acquisition_path)
    m1 = binding.commitment_for("M1")
    program, receipt, envelope = load_study_frozen_program(
        frozen_program_path=program_path,
        formation_receipt_path=formation_path,
        verify_live=True,
        implementation_root=project,
    )
    if receipt.get("formation_block_id_hash") != stable_hash({"block": "F1"}):
        raise MuSiQueM1RunnerError("M1 P program was not formed on exact F1")
    runtime = _runtime_paths(
        runtime_python=runtime_python,
        local_llm_model=local_llm_model,
        local_embedding_model=local_embedding_model,
        base_binding_receipt_path=base_binding_receipt_path,
        attestation_receipt_path=attestation_receipt_path,
    )
    prepared_runtime = _prepare_runtime(project=project, runtime=runtime)
    safe_runtime_binding = prepared_runtime.safe_binding
    if safe_runtime_binding.get("formal_entry_executable_identity_probe_calls") != 0:
        raise MuSiQueM1RunnerError(
            "prospective v2 attestation attempted an executable probe"
        )
    body: dict[str, Any] = {
        "schema": FREEZE_SCHEMA,
        "decision": FREEZE_DECISION,
        "implementation": current_m1_implementation_binding(project),
        "authorization_hash": _require_sha256(
            authorization_hash, "M1 execution authorization"
        ),
        "execution_root_hash": _root_binding_hash(execution_root),
        "source_binding": {
            "acquisition_sha256": binding.acquisition_sha256,
            "acquisition_file_sha256": binding.acquisition_file_sha256,
            "private_pack_sha256": binding.private_pack_sha256,
            "measurement_block_id_hash": stable_hash({"block": "M1"}),
            "measurement_block_file_sha256": m1.file_sha256,
            "measurement_item_commitment_set_sha256": (
                m1.item_commitment_set_sha256
            ),
            "measurement_item_count": m1.count,
        },
        "p_operator_binding": {
            "formation_receipt_file_sha256": _sha256_file(formation_path),
            "formation_receipt_hash": receipt["receipt_hash"],
            "frozen_program_file_sha256": _sha256_file(program_path),
            "frozen_program_envelope_hash": envelope["envelope_hash"],
            "program_hash": program.program_hash,
            "formed_on_block_id_hash": receipt["formation_block_id_hash"],
        },
        "official_runtime_binding": {
            "prepared_safe_binding": safe_runtime_binding,
            "base_binding_file_sha256": _sha256_file(
                runtime.base_binding_receipt_path
            ),
            "attestation_file_sha256": _sha256_file(
                runtime.attestation_receipt_path
            ),
            "pre_run_fresh_filesystem_attestation": True,
            "formal_entry_uses_filesystem_attestation_only": True,
        },
        "execution_contract": {
            "arms": list(ARM_IDS),
            "item_count": M1_ITEM_COUNT,
            "top_k": TOP_K,
            "work_unit_count": WORK_UNIT_COUNT,
            "maximum_concurrency": MAXIMUM_CONCURRENCY,
            "formal_entry": "clean_python_module_cli_v1",
            "all_work_units_submitted_before_support_scoring": True,
            "promotion_policy": PROMOTION_POLICY,
            "generator_calls": 0,
            "external_network_calls": 0,
            "online_evaluator_calls": 0,
            "retries": 0,
            "replays": 0,
            "resamples": 0,
        },
        "ordering": {
            "measurement_block_rows_read_while_freezing": 0,
            "measurement_support_labels_read_while_freezing": 0,
            "pre_run_freeze_complete_before_measurement_open": True,
        },
        "raw_content_persisted": False,
    }
    freeze = {**body, "freeze_hash": stable_hash(body)}
    output = _absolute_no_symlink(output_path, "M1 pre-run freeze output")
    if output.exists():
        raise MuSiQueM1RunnerError("M1 pre-run freeze output already exists")
    _write_json_exclusive(output, freeze)
    return freeze


def _load_freeze(path: str | Path) -> tuple[dict[str, Any], str]:
    payload, raw = _read_json_object(Path(path), "M1 pre-run freeze")
    body = dict(payload)
    declared = _require_sha256(body.pop("freeze_hash", None), "M1 freeze hash")
    expected_keys = {
        "authorization_hash",
        "decision",
        "execution_contract",
        "execution_root_hash",
        "freeze_hash",
        "implementation",
        "official_runtime_binding",
        "ordering",
        "p_operator_binding",
        "raw_content_persisted",
        "schema",
        "source_binding",
    }
    if (
        set(payload) != expected_keys
        or payload.get("schema") != FREEZE_SCHEMA
        or payload.get("decision") != FREEZE_DECISION
        or stable_hash(body) != declared
        or payload.get("raw_content_persisted") is not False
    ):
        raise MuSiQueM1RunnerError("M1 pre-run freeze drifted")
    _require_sha256(payload.get("authorization_hash"), "M1 authorization hash")
    _require_sha256(payload.get("execution_root_hash"), "M1 root binding hash")
    contract = payload.get("execution_contract")
    ordering = payload.get("ordering")
    if (
        not isinstance(contract, Mapping)
        or not isinstance(ordering, Mapping)
        or contract.get("arms") != list(ARM_IDS)
        or contract.get("item_count") != M1_ITEM_COUNT
        or contract.get("work_unit_count") != WORK_UNIT_COUNT
        or contract.get("maximum_concurrency") != MAXIMUM_CONCURRENCY
        or contract.get("formal_entry") != "clean_python_module_cli_v1"
        or contract.get("promotion_policy") != PROMOTION_POLICY
        or any(contract.get(field) != 0 for field in (
            "generator_calls",
            "external_network_calls",
            "online_evaluator_calls",
            "retries",
            "replays",
            "resamples",
        ))
        or ordering.get("measurement_block_rows_read_while_freezing") != 0
        or ordering.get("measurement_support_labels_read_while_freezing") != 0
        or ordering.get("pre_run_freeze_complete_before_measurement_open")
        is not True
    ):
        raise MuSiQueM1RunnerError("M1 execution contract drifted")
    _validate_implementation_binding(payload.get("implementation"))
    return payload, _sha256_bytes(raw)


def _canonical_raw_retrieve(item: RetrievalStudyItem) -> tuple[int, ...]:
    return tuple(paragraph.idx for paragraph in item.corpus[:TOP_K])


def _typed_program_retrieve(
    program: TypedRetrievalProgram, item: RetrievalStudyItem
) -> tuple[int, ...]:
    return typed_retrieve(program, item.question, item.corpus)


def _official_retrieve(
    item: RetrievalStudyItem,
    runtime: PreparedFormalRuntimeV2,
    work_root: Path,
) -> tuple[int, ...]:
    return runtime.retrieve(
        question=item.question,
        paragraphs=item.hipporag_paragraphs(),
        work_root=work_root,
    )


@dataclass(frozen=True)
class _WorkUnit:
    ordinal: int
    arm_id: str
    item: RetrievalStudyItem = field(repr=False)

    @property
    def key(self) -> tuple[int, str]:
        return self.ordinal, self.arm_id


def _validate_ranking(
    value: Sequence[int], *, item: RetrievalStudyItem
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise MuSiQueM1RunnerError("retrieval output is not an index sequence")
    try:
        ranking = tuple(value)
    except TypeError as exc:
        raise MuSiQueM1RunnerError(
            "retrieval output is not an index sequence"
        ) from exc
    if (
        len(ranking) != TOP_K
        or len(set(ranking)) != TOP_K
        or any(
            type(index) is not int or not 0 <= index < len(item.corpus)
            for index in ranking
        )
    ):
        raise MuSiQueM1RunnerError(
            "retrieval output violates exact top-five idx contract"
        )
    return ranking


def _score_after_join(
    *,
    items: Sequence[StudyItem],
    rankings: Mapping[tuple[int, str], tuple[int, ...]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for ordinal, item in enumerate(items):
        supports = frozenset(item.support_indices)
        arm_rows: dict[str, dict[str, Any]] = {}
        for arm_id in ARM_IDS:
            ranking = rankings[(ordinal, arm_id)]
            arm_rows[arm_id] = {
                "support_hits": len(supports.intersection(ranking)),
                "ranking_hash": stable_hash(
                    {"retrieved_indices": list(ranking)}
                ),
            }
        rows.append(
            {
                "item_id_hash": item.item_id_hash,
                "support_total": len(supports),
                "arms": arm_rows,
            }
        )
    support_total = sum(row["support_total"] for row in rows)
    arm_metrics: dict[str, dict[str, Any]] = {}
    for arm_id in ARM_IDS:
        hits = sum(row["arms"][arm_id]["support_hits"] for row in rows)
        arm_metrics[arm_id] = {
            "arm_id": arm_id,
            "item_count": len(rows),
            "support_hit_count": hits,
            "support_total": support_total,
            "support_recall_at_5": float(Fraction(hits, support_total)),
            "items_with_any_support_hit": sum(
                row["arms"][arm_id]["support_hits"] > 0 for row in rows
            ),
            "ranking_score_closure_hash": stable_hash(
                [
                    {
                        "item_id_hash": row["item_id_hash"],
                        "support_total": row["support_total"],
                        **row["arms"][arm_id],
                    }
                    for row in rows
                ]
            ),
        }
    p_hits = arm_metrics["frozen_P"]["support_hit_count"]
    raw_hits = arm_metrics["canonical_RAW"]["support_hit_count"]
    net = p_hits - raw_hits
    gain = harm = tie = 0
    for row in rows:
        p_value = row["arms"]["frozen_P"]["support_hits"]
        raw_value = row["arms"]["canonical_RAW"]["support_hits"]
        gain += int(p_value > raw_value)
        harm += int(p_value < raw_value)
        tie += int(p_value == raw_value)
    promoted = net > 0
    return {
        "primary_metric": "official_support_recall_at_5",
        "arm_metrics": arm_metrics,
        "paired_P_minus_RAW": {
            "net_support_hit_count": net,
            "support_recall_delta": float(Fraction(net, support_total)),
            "gain_item_count": gain,
            "harm_item_count": harm,
            "tie_item_count": tie,
        },
        "promotion_disposition": {
            "policy": PROMOTION_POLICY,
            "positive_net_support": promoted,
            "disposition": (
                "promote_P_to_retained_generation_one"
                if promoted
                else "do_not_promote_P"
            ),
            "archive_mutated_by_runner": False,
        },
        "score_closure_hash": stable_hash(rows),
        "item_level_rows_persisted": False,
        "raw_content_persisted": False,
    }


def execute_m1_retrieval_formal(
    *,
    project_root: str | Path,
    pre_run_freeze_path: str | Path,
    m1_block_path: str | Path,
    acquisition_receipt_path: str | Path,
    formation_receipt_path: str | Path,
    frozen_program_path: str | Path,
    runtime_python: str | Path,
    local_llm_model: str | Path,
    local_embedding_model: str | Path,
    base_binding_receipt_path: str | Path,
    attestation_receipt_path: str | Path,
    execution_root: str | Path,
) -> dict[str, Any]:
    """Consume the sole frozen M1 authorization; no result injection exists."""

    if _CLEAN_MODULE_CLI_ACTIVE is not True:
        raise MuSiQueM1RunnerError(
            "formal M1 execution is available only through the clean module CLI"
        )
    project = Path(project_root).resolve(strict=True)
    freeze, freeze_file_hash = _load_freeze(pre_run_freeze_path)
    if freeze["implementation"] != current_m1_implementation_binding(project):
        raise MuSiQueM1RunnerError("live M1 implementation drifted")
    root = _canonical_new_root(execution_root)
    if freeze.get("execution_root_hash") != _root_binding_hash(root):
        raise MuSiQueM1RunnerError("M1 execution root binding drifted")
    if root.exists() or root.is_symlink():
        raise MuSiQueM1RunnerError(
            "fresh M1 execution root already exists; replay is forbidden"
        )
    acquisition_path = _absolute_no_symlink(
        acquisition_receipt_path, "study acquisition receipt"
    )
    formation_path = _absolute_no_symlink(
        formation_receipt_path, "F1 formation receipt"
    )
    program_path = _absolute_no_symlink(
        frozen_program_path, "F1 frozen program"
    )
    binding = load_study_acquisition_binding(acquisition_path)
    source = freeze.get("source_binding")
    if not isinstance(source, Mapping):
        raise MuSiQueM1RunnerError("M1 source binding is malformed")
    m1_commitment = binding.commitment_for("M1")
    if (
        source.get("acquisition_sha256") != binding.acquisition_sha256
        or source.get("acquisition_file_sha256")
        != binding.acquisition_file_sha256
        or source.get("private_pack_sha256") != binding.private_pack_sha256
        or source.get("measurement_block_id_hash")
        != stable_hash({"block": "M1"})
        or source.get("measurement_block_file_sha256")
        != m1_commitment.file_sha256
        or source.get("measurement_item_commitment_set_sha256")
        != m1_commitment.item_commitment_set_sha256
        or source.get("measurement_item_count") != M1_ITEM_COUNT
    ):
        raise MuSiQueM1RunnerError("M1 source binding drifted")
    program, receipt, envelope = load_study_frozen_program(
        frozen_program_path=program_path,
        formation_receipt_path=formation_path,
        verify_live=True,
        implementation_root=project,
    )
    p_binding = freeze.get("p_operator_binding")
    if not isinstance(p_binding, Mapping) or p_binding != {
        "formation_receipt_file_sha256": _sha256_file(formation_path),
        "formation_receipt_hash": receipt["receipt_hash"],
        "frozen_program_file_sha256": _sha256_file(program_path),
        "frozen_program_envelope_hash": envelope["envelope_hash"],
        "program_hash": program.program_hash,
        "formed_on_block_id_hash": stable_hash({"block": "F1"}),
    }:
        raise MuSiQueM1RunnerError("M1 P-operator binding drifted")
    runtime = _runtime_paths(
        runtime_python=runtime_python,
        local_llm_model=local_llm_model,
        local_embedding_model=local_embedding_model,
        base_binding_receipt_path=base_binding_receipt_path,
        attestation_receipt_path=attestation_receipt_path,
    )
    prepared_runtime = _prepare_runtime(project=project, runtime=runtime)
    safe_runtime_binding = prepared_runtime.safe_binding
    official_binding = freeze.get("official_runtime_binding")
    expected_official = {
        "prepared_safe_binding": safe_runtime_binding,
        "base_binding_file_sha256": _sha256_file(
            runtime.base_binding_receipt_path
        ),
        "attestation_file_sha256": _sha256_file(
            runtime.attestation_receipt_path
        ),
        "pre_run_fresh_filesystem_attestation": True,
        "formal_entry_uses_filesystem_attestation_only": True,
    }
    if (
        safe_runtime_binding.get("formal_entry_executable_identity_probe_calls")
        != 0
        or official_binding != expected_official
    ):
        raise MuSiQueM1RunnerError("M1 official runtime binding drifted")

    try:
        os.mkdir(root, 0o700)
    except FileExistsError as exc:
        raise MuSiQueM1RunnerError(
            "fresh M1 execution root already exists; replay is forbidden"
        ) from exc
    stage = "authorization_consumption"
    attempted = 0
    completed = 0
    counter_lock = threading.Lock()
    try:
        consumption_body = {
            "schema": f"{RUNNER_VERSION}_authorization_consumption",
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

        stage = "exact_M1_open_after_freeze"
        block = load_measurement_block_after_freeze(
            block_path=m1_block_path,
            acquisition_receipt_path=acquisition_path,
            measurement_freeze_path=pre_run_freeze_path,
            expected_block="M1",
        )
        if len(block.items) != M1_ITEM_COUNT:
            raise MuSiQueM1RunnerError("exact M1 item count drifted")
        work_units = tuple(
            _WorkUnit(
                ordinal=ordinal,
                arm_id=arm_id,
                item=item.retrieval_view(),
            )
            for ordinal, item in enumerate(block.items)
            for arm_id in ARM_IDS
        )
        if len(work_units) != WORK_UNIT_COUNT:
            raise MuSiQueM1RunnerError("M1 work-unit grid drifted")

        stage = "retrieval_execution"

        def run_one(work: _WorkUnit) -> tuple[tuple[int, str], tuple[int, ...]]:
            nonlocal attempted, completed
            with counter_lock:
                attempted += 1
            if work.arm_id == "canonical_RAW":
                value = _canonical_raw_retrieve(work.item)
            elif work.arm_id == "frozen_P":
                value = _typed_program_retrieve(program, work.item)
            elif work.arm_id == "official_HippoRAG":
                value = _official_retrieve(
                    work.item,
                    prepared_runtime,
                    root / f"official_item_{work.ordinal:02d}",
                )
            else:  # pragma: no cover - constant grid
                raise MuSiQueM1RunnerError("unknown frozen M1 arm")
            ranking = _validate_ranking(value, item=work.item)
            with counter_lock:
                completed += 1
            return work.key, ranking

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=MAXIMUM_CONCURRENCY,
            thread_name_prefix="musique-m1",
        ) as executor:
            futures = [executor.submit(run_one, work) for work in work_units]
            terminal_rows = [future.result() for future in futures]
        if attempted != WORK_UNIT_COUNT or completed != WORK_UNIT_COUNT:
            raise MuSiQueM1RunnerError("M1 terminal closure is incomplete")
        rankings = dict(terminal_rows)
        if len(rankings) != WORK_UNIT_COUNT:
            raise MuSiQueM1RunnerError("M1 terminal keys are not one-to-one")

        stage = "fresh_runtime_postflight_before_scoring"
        postflight_binding = prepared_runtime.fresh_reverify()
        if postflight_binding != safe_runtime_binding:
            raise MuSiQueM1RunnerError(
                "M1 runtime drifted between preflight and postflight"
            )

        stage = "offline_support_scoring_after_join"
        measurement = _score_after_join(items=block.items, rankings=rankings)
        ranking_receipts = [
            {
                "ordinal_hash": stable_hash({"ordinal": ordinal}),
                "arm_id": arm_id,
                "ranking_hash": stable_hash(
                    {"retrieved_indices": list(ranking)}
                ),
            }
            for (ordinal, arm_id), ranking in sorted(rankings.items())
        ]
        report_body: dict[str, Any] = {
            "schema": REPORT_SCHEMA,
            "valid": True,
            "freeze_hash": freeze["freeze_hash"],
            "freeze_file_sha256": freeze_file_hash,
            "measurement_block_id_hash": stable_hash({"block": "M1"}),
            "measurement_block_file_sha256": block.file_sha256,
            "measurement": measurement,
            "execution": {
                "arm_ids": list(ARM_IDS),
                "item_count": len(block.items),
                "work_unit_count": WORK_UNIT_COUNT,
                "retrieval_call_count": attempted,
                "retrieval_terminal_count": completed,
                "configured_maximum_concurrency": MAXIMUM_CONCURRENCY,
                "all_terminals_joined_before_support_scoring": True,
                "ranking_receipt_set_hash": stable_hash(ranking_receipts),
                "generator_calls": 0,
                "external_network_calls": 0,
                "online_evaluator_calls": 0,
                "retries": 0,
                "replays": 0,
                "resamples": 0,
            },
            "runtime": {
                "attestation_receipt_sha256": safe_runtime_binding[
                    "attestation_receipt_sha256"
                ],
                "formal_entry_executable_identity_probe_calls": 0,
                "official_arm_terminal_count": sum(
                    arm_id == "official_HippoRAG"
                    for _ordinal, arm_id in rankings
                ),
                "worker_process_count_inferred_from_arm_count": False,
                "postflight_fresh_filesystem_attestation": True,
                "postflight_binding_sha256": postflight_binding[
                    "binding_sha256"
                ],
            },
            "sealed_or_test_content_accessed": False,
            "raw_content_persisted": False,
        }
        report = {**report_body, "report_hash": stable_hash(report_body)}
        stage = "aggregate_report_persistence"
        _write_json_exclusive(root / REPORT_FILENAME, report)
        persisted = json.loads((root / REPORT_FILENAME).read_text("utf-8"))
        persisted_hash = persisted.pop("report_hash", None)
        if persisted_hash != stable_hash(persisted):
            raise MuSiQueM1RunnerError("persisted M1 report hash drifted")
        return report
    except Exception as exc:
        failure_body = {
            "schema": FAILURE_SCHEMA,
            "valid": False,
            "freeze_hash": freeze["freeze_hash"],
            "failure_stage": stage,
            "error_type_hash": stable_hash(
                {"error_type": type(exc).__name__}
            ),
            "authorization_consumed": (
                root / CONSUMPTION_FILENAME
            ).is_file(),
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
        raise MuSiQueM1RunnerError(
            "formal M1 run failed and cannot be replayed"
        ) from exc


__all__ = [
    "MuSiQueM1RunnerError",
    "build_m1_pre_run_freeze",
    "execute_m1_retrieval_formal",
]


def main(argv: Sequence[str] | None = None) -> int:
    """Clean module-CLI entry; no callable or result injection is accepted."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    execute = subparsers.add_parser("execute")
    for command in (freeze, execute):
        command.add_argument("--project-root", type=Path, required=True)
        command.add_argument("--acquisition-receipt", type=Path, required=True)
        command.add_argument("--formation-receipt", type=Path, required=True)
        command.add_argument("--frozen-program", type=Path, required=True)
        command.add_argument("--runtime-python", type=Path, required=True)
        command.add_argument("--local-llm-model", type=Path, required=True)
        command.add_argument("--local-embedding-model", type=Path, required=True)
        command.add_argument("--base-binding-receipt", type=Path, required=True)
        command.add_argument("--attestation-receipt", type=Path, required=True)
        command.add_argument("--execution-root", type=Path, required=True)
    freeze.add_argument("--authorization-hash", required=True)
    freeze.add_argument("--output", type=Path, required=True)
    execute.add_argument("--pre-run-freeze", type=Path, required=True)
    execute.add_argument("--m1-block", type=Path, required=True)
    arguments = parser.parse_args(argv)
    common = {
        "project_root": arguments.project_root,
        "acquisition_receipt_path": arguments.acquisition_receipt,
        "formation_receipt_path": arguments.formation_receipt,
        "frozen_program_path": arguments.frozen_program,
        "runtime_python": arguments.runtime_python,
        "local_llm_model": arguments.local_llm_model,
        "local_embedding_model": arguments.local_embedding_model,
        "base_binding_receipt_path": arguments.base_binding_receipt,
        "attestation_receipt_path": arguments.attestation_receipt,
        "execution_root": arguments.execution_root,
    }
    if arguments.command == "freeze":
        build_m1_pre_run_freeze(
            **common,
            authorization_hash=arguments.authorization_hash,
            output_path=arguments.output,
        )
        return 0
    global _CLEAN_MODULE_CLI_ACTIVE
    _CLEAN_MODULE_CLI_ACTIVE = True
    try:
        execute_m1_retrieval_formal(
            **common,
            pre_run_freeze_path=arguments.pre_run_freeze,
            m1_block_path=arguments.m1_block,
        )
    finally:
        _CLEAN_MODULE_CLI_ACTIVE = False
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
