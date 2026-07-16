"""Exact A_form/F3 wiring for offline evaluator co-evolution.

Both entry points open one exact formation block through the recursive-study
custody parser, enumerate the complete fixed typed-retrieval DSL, execute every
program against every item with gold-free operator inputs, join all retrieval
terminals, and only then compute official support evidence.  Item-level
evidence is written to a private, git-ignored cache.  Public receipts contain
only aggregate counts, commitments, and hashes.

``form_a_form_stage`` selects the evaluator challenger using A_form only.
``form_f3_stage`` verifies the exact A_form cache/receipt pair and freezes the
incumbent/challenger program choices on F3 before A_hold or M3 can be opened.
There is no model, generator, network, online judge, retry, replay, result
injection, or caller-supplied program surface.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import threading
from typing import Any, Mapping, Sequence

from ..models import stable_hash
from .musique_evaluator_coevolution_v1 import (
    EvaluatorFormationError,
    ItemRetrievalEvidence,
    ProgramRetrievalEvidence,
    form_evaluator_challenger,
    freeze_prospective_search_formation,
)
from .musique_recursive_study_blocks_v1 import (
    LoadedStudyBlock,
    RetrievalStudyItem,
    load_formation_block,
)
from .musique_typed_retriever_formation_v1 import (
    DSL_VERSION,
    MAX_CANDIDATES,
    OPERATOR_VERSION,
    TOKENIZER_VERSION,
    TOP_K,
    TypedRetrievalProgram,
    enumerate_programs,
    retrieve as typed_retrieve,
)


VERSION = "musique_evaluator_stage_formation_v1"
EVIDENCE_CACHE_SCHEMA = f"{VERSION}_private_evidence_cache"
A_FORM_PUBLIC_SCHEMA = f"{VERSION}_A_form_public_receipt"
F3_PUBLIC_SCHEMA = f"{VERSION}_F3_public_receipt"
IMPLEMENTATION_SCHEMA = f"{VERSION}_implementation"
IMPLEMENTATION_RELATIVE_FILES = (
    "assumption_agent/benchmarks/musique_evaluator_stage_formation_v1.py",
    "assumption_agent/benchmarks/musique_evaluator_coevolution_v1.py",
    "assumption_agent/benchmarks/musique_recursive_study_blocks_v1.py",
    "assumption_agent/benchmarks/musique_typed_retriever_formation_v1.py",
    "assumption_agent/models.py",
)
FORMATION_STAGES = ("A_form", "F3")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class MuSiQueEvaluatorStageFormationError(RuntimeError):
    """An evaluator formation stage failed its exact offline contract."""


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise MuSiQueEvaluatorStageFormationError(
            "required evaluator formation file is unavailable"
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise MuSiQueEvaluatorStageFormationError(
            f"{field_name} must be a lowercase SHA-256 digest"
        )
    return value


def _absolute_no_symlink(path: str | Path, field_name: str) -> Path:
    candidate = Path(os.path.abspath(os.fspath(path)))
    for component in (*reversed(candidate.parents), candidate):
        if component.is_symlink():
            raise MuSiQueEvaluatorStageFormationError(
                f"{field_name} contains a symlink component"
            )
    return candidate


def _read_json_object(path: str | Path, field_name: str) -> tuple[dict[str, Any], bytes]:
    candidate = _absolute_no_symlink(path, field_name)
    if not candidate.is_file():
        raise MuSiQueEvaluatorStageFormationError(f"{field_name} is unavailable")
    try:
        raw = candidate.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MuSiQueEvaluatorStageFormationError(
            f"{field_name} is invalid"
        ) from exc
    if not isinstance(value, dict):
        raise MuSiQueEvaluatorStageFormationError(
            f"{field_name} must contain one object"
        )
    return value, raw


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


def _containing_git_repository(path: Path) -> Path | None:
    anchor = path if path.is_dir() else path.parent
    while not anchor.exists():
        if anchor.parent == anchor:
            return None
        anchor = anchor.parent
    completed = subprocess.run(
        ["git", "-C", str(anchor), "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        return None
    return Path(completed.stdout.strip()).resolve(strict=True)


def _require_private_cache_boundary(path: Path) -> None:
    """Require an in-repository private cache target to be ignored/untracked."""

    repository = _containing_git_repository(path)
    if repository is None:
        return
    try:
        relative = path.resolve(strict=False).relative_to(repository).as_posix()
    except ValueError as exc:
        raise MuSiQueEvaluatorStageFormationError(
            "private evidence cache repository containment is ambiguous"
        ) from exc
    ignored = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "check-ignore",
            "--no-index",
            "-q",
            "--",
            relative,
        ],
        check=False,
        capture_output=True,
        timeout=30,
    )
    if ignored.returncode != 0:
        raise MuSiQueEvaluatorStageFormationError(
            "private evidence cache must be git-ignored"
        )
    tracked = subprocess.run(
        ["git", "-C", str(repository), "ls-files", "--", relative],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if tracked.returncode != 0 or tracked.stdout.strip():
        raise MuSiQueEvaluatorStageFormationError(
            "private evidence cache must be untracked"
        )


def _assert_public_safe(payload: Mapping[str, Any]) -> None:
    serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    forbidden = (
        '"answers"',
        '"corpus"',
        '"item_id"',
        '"items"',
        '"normalized_answers"',
        '"paragraph_text"',
        '"private_evidence_path"',
        '"private_root"',
        '"question"',
        '"source_row_sha256"',
        '"support_indices"',
    )
    if any(token in serialized for token in forbidden):
        raise MuSiQueEvaluatorStageFormationError(
            "public evaluator receipt contains private or item-level content"
        )


def current_evaluator_stage_implementation_binding(
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    root = (
        Path(__file__).resolve(strict=True).parents[2]
        if project_root is None
        else Path(project_root).resolve(strict=True)
    )
    rows: list[dict[str, str]] = []
    for relative in IMPLEMENTATION_RELATIVE_FILES:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise MuSiQueEvaluatorStageFormationError(
                f"evaluator formation implementation is missing: {relative}"
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
        raise MuSiQueEvaluatorStageFormationError(
            "evaluator formation implementation binding is malformed"
        )
    if value.get("schema") != IMPLEMENTATION_SCHEMA:
        raise MuSiQueEvaluatorStageFormationError(
            "evaluator formation implementation schema drifted"
        )
    files = value.get("files")
    if not isinstance(files, list) or len(files) != len(IMPLEMENTATION_RELATIVE_FILES):
        raise MuSiQueEvaluatorStageFormationError(
            "evaluator formation implementation set drifted"
        )
    rows: list[dict[str, str]] = []
    for expected, row in zip(IMPLEMENTATION_RELATIVE_FILES, files):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"path", "sha256"}
            or row.get("path") != expected
        ):
            raise MuSiQueEvaluatorStageFormationError(
                "evaluator formation implementation row drifted"
            )
        rows.append(
            {
                "path": expected,
                "sha256": _require_sha256(
                    row.get("sha256"), "implementation file hash"
                ),
            }
        )
    set_hash = _require_sha256(value.get("set_sha256"), "implementation set hash")
    if stable_hash(rows) != set_hash:
        raise MuSiQueEvaluatorStageFormationError(
            "evaluator formation implementation set hash drifted"
        )
    return {"schema": IMPLEMENTATION_SCHEMA, "files": rows, "set_sha256": set_hash}


def fixed_typed_programs() -> tuple[TypedRetrievalProgram, ...]:
    """Return the complete typed DSL and fail closed on any identity drift."""

    programs = tuple(enumerate_programs())
    if (
        not programs
        or len(programs) > MAX_CANDIDATES
        or len({program.program_hash for program in programs}) != len(programs)
        or any(program.type_issues() for program in programs)
    ):
        raise MuSiQueEvaluatorStageFormationError(
            "fixed typed candidate grammar drifted"
        )
    return programs


def candidate_set_binding(
    programs: Sequence[TypedRetrievalProgram] | None = None,
) -> dict[str, Any]:
    candidates = fixed_typed_programs() if programs is None else tuple(programs)
    if tuple(program.program_hash for program in candidates) != tuple(
        program.program_hash for program in fixed_typed_programs()
    ) or any(program.to_dict() != fixed.to_dict() for program, fixed in zip(
        candidates, fixed_typed_programs()
    )):
        raise MuSiQueEvaluatorStageFormationError(
            "caller candidate set differs from fixed typed DSL"
        )
    payloads = [program.to_dict() for program in candidates]
    hashes = [program.program_hash for program in candidates]
    return {
        "dsl_version": DSL_VERSION,
        "operator_version": OPERATOR_VERSION,
        "tokenizer_version": TOKENIZER_VERSION,
        "top_k": TOP_K,
        "candidate_count": len(candidates),
        "candidate_budget": MAX_CANDIDATES,
        "program_order_sha256": stable_hash(hashes),
        "program_payload_set_sha256": stable_hash(payloads),
        "all_candidates_type_valid": True,
    }


def evidence_set_sha256(
    evidences: Sequence[ProgramRetrievalEvidence],
) -> str:
    rows = tuple(evidence.validate() for evidence in evidences)
    if not rows:
        raise MuSiQueEvaluatorStageFormationError("evidence set is empty")
    return stable_hash(
        [
            {
                "program_sha256": program.program_sha256,
                "program_length": program.program_length,
                "items": [asdict(item) for item in program.items],
            }
            for program in rows
        ]
    )


def _validate_ranking(
    value: Sequence[int], item: RetrievalStudyItem
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError("retrieval output is not an index sequence")
    ranking = tuple(value)
    if (
        len(ranking) != TOP_K
        or len(set(ranking)) != TOP_K
        or any(
            type(index) is not int or not 0 <= index < len(item.corpus)
            for index in ranking
        )
    ):
        raise ValueError("retrieval output violates exact top-five contract")
    return ranking


def build_program_retrieval_evidence(
    block: LoadedStudyBlock,
) -> tuple[tuple[ProgramRetrievalEvidence, ...], dict[str, Any]]:
    """Execute the fixed program/item grid, then score only after full join."""

    if block.block not in (*FORMATION_STAGES, "A_hold", "M3"):
        raise MuSiQueEvaluatorStageFormationError(
            "evaluator evidence is limited to A_form, F3, A_hold, or M3"
        )
    programs = fixed_typed_programs()
    item_count = len(block.items)
    work_unit_count = len(programs) * item_count
    if item_count < 4 or work_unit_count <= 0:
        raise MuSiQueEvaluatorStageFormationError(
            "evaluator evidence grid is incomplete"
        )
    attempted = 0
    completed = 0
    lock = threading.Lock()

    def run_one(
        program_ordinal: int,
        item_ordinal: int,
    ) -> tuple[tuple[int, int], tuple[int, ...] | None, bool]:
        nonlocal attempted, completed
        program = programs[program_ordinal]
        item = block.items[item_ordinal].retrieval_view()
        with lock:
            attempted += 1
        try:
            ranking = _validate_ranking(
                typed_retrieve(program, item.question, item.corpus), item
            )
            invalid = False
        except (TypeError, ValueError, ArithmeticError):
            ranking = None
            invalid = True
        with lock:
            completed += 1
        return (program_ordinal, item_ordinal), ranking, invalid

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=work_unit_count,
        thread_name_prefix=f"musique-{block.block.lower()}-evidence",
    ) as executor:
        futures = [
            executor.submit(run_one, program_ordinal, item_ordinal)
            for program_ordinal in range(len(programs))
            for item_ordinal in range(item_count)
        ]
        terminal_rows = [future.result() for future in futures]
    if attempted != work_unit_count or completed != work_unit_count:
        raise MuSiQueEvaluatorStageFormationError(
            "evaluator retrieval terminal closure is incomplete"
        )
    terminals = {
        key: (ranking, invalid) for key, ranking, invalid in terminal_rows
    }
    if len(terminals) != work_unit_count:
        raise MuSiQueEvaluatorStageFormationError(
            "evaluator retrieval terminal keys are not one-to-one"
        )

    # Official support labels are consulted only after every gold-free
    # retrieval terminal above has joined.
    evidences: list[ProgramRetrievalEvidence] = []
    invalid_count = 0
    for program_ordinal, program in enumerate(programs):
        item_rows: list[ItemRetrievalEvidence] = []
        for item_ordinal, item in enumerate(block.items):
            ranking, invalid = terminals[(program_ordinal, item_ordinal)]
            invalid_count += int(invalid)
            hits = (
                0
                if ranking is None
                else len(frozenset(ranking).intersection(item.support_indices))
            )
            retrieval_hash = stable_hash(
                {
                    "program_sha256": program.program_hash,
                    "item_commitment_sha256": item.item_commitment_sha256,
                    "retrieved_indices": None if ranking is None else list(ranking),
                    "invalid": invalid,
                }
            )
            item_rows.append(
                ItemRetrievalEvidence(
                    item_commitment_sha256=item.item_commitment_sha256,
                    support_hits=hits,
                    support_total=len(item.support_indices),
                    invalid=invalid,
                    retrieval_sha256=retrieval_hash,
                ).validate()
            )
        evidences.append(
            ProgramRetrievalEvidence(
                program_sha256=program.program_hash,
                program_length=program.program_length,
                items=tuple(item_rows),
            ).validate()
        )
    evidence_tuple = tuple(evidences)
    execution = {
        "candidate_count": len(programs),
        "item_count": item_count,
        "work_unit_count": work_unit_count,
        "retrieval_call_count": attempted,
        "retrieval_terminal_count": completed,
        "configured_maximum_concurrency": work_unit_count,
        "all_work_units_submitted_before_support_scoring": True,
        "all_terminals_joined_before_support_scoring": True,
        "invalid_terminal_count": invalid_count,
        "generator_calls": 0,
        "network_calls": 0,
        "online_evaluator_calls": 0,
        "retries": 0,
        "replays": 0,
        "resamples": 0,
    }
    return evidence_tuple, execution


def _serialize_evidence(
    evidences: Sequence[ProgramRetrievalEvidence],
) -> list[dict[str, Any]]:
    return [
        {
            "program_sha256": row.program_sha256,
            "program_length": row.program_length,
            "items": [asdict(item) for item in row.items],
        }
        for row in evidences
    ]


def _deserialize_evidence(value: object) -> tuple[ProgramRetrievalEvidence, ...]:
    if not isinstance(value, list):
        raise MuSiQueEvaluatorStageFormationError(
            "private evaluator evidence list is malformed"
        )
    rows: list[ProgramRetrievalEvidence] = []
    try:
        for program in value:
            if not isinstance(program, Mapping) or set(program) != {
                "items",
                "program_length",
                "program_sha256",
            }:
                raise MuSiQueEvaluatorStageFormationError(
                    "private evaluator program evidence is malformed"
                )
            items = program.get("items")
            if not isinstance(items, list):
                raise MuSiQueEvaluatorStageFormationError(
                    "private evaluator item evidence is malformed"
                )
            rows.append(
                ProgramRetrievalEvidence(
                    program_sha256=str(program["program_sha256"]),
                    program_length=program["program_length"],
                    items=tuple(
                        ItemRetrievalEvidence(**dict(item)).validate()
                        for item in items
                        if isinstance(item, Mapping)
                    ),
                ).validate()
            )
    except (KeyError, TypeError, EvaluatorFormationError) as exc:
        raise MuSiQueEvaluatorStageFormationError(
            "private evaluator evidence payload drifted"
        ) from exc
    if len(rows) != len(value):
        raise MuSiQueEvaluatorStageFormationError(
            "private evaluator evidence row was discarded"
        )
    return tuple(rows)


def _cache_payload(
    *,
    stage: str,
    block: LoadedStudyBlock,
    evidences: Sequence[ProgramRetrievalEvidence],
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema": EVIDENCE_CACHE_SCHEMA,
        "stage": stage,
        "source_binding": block.safe_payload(),
        "candidate_set_binding": candidate_set_binding(),
        "evidence_set_sha256": evidence_set_sha256(evidences),
        "program_evidence": _serialize_evidence(evidences),
        "execution": dict(execution),
        "raw_question_answer_or_corpus_persisted": False,
    }
    return {**body, "cache_sha256": stable_hash(body)}


def load_private_evidence_cache(
    path: str | Path,
    *,
    expected_stage: str,
) -> tuple[tuple[ProgramRetrievalEvidence, ...], dict[str, Any], str]:
    payload, raw = _read_json_object(path, "private evaluator evidence cache")
    body = dict(payload)
    declared = _require_sha256(body.pop("cache_sha256", None), "evidence cache hash")
    expected_keys = {
        "cache_sha256",
        "candidate_set_binding",
        "evidence_set_sha256",
        "execution",
        "program_evidence",
        "raw_question_answer_or_corpus_persisted",
        "schema",
        "source_binding",
        "stage",
    }
    if (
        set(payload) != expected_keys
        or payload.get("schema") != EVIDENCE_CACHE_SCHEMA
        or payload.get("stage") != expected_stage
        or stable_hash(body) != declared
        or payload.get("raw_question_answer_or_corpus_persisted") is not False
        or payload.get("candidate_set_binding") != candidate_set_binding()
    ):
        raise MuSiQueEvaluatorStageFormationError(
            "private evaluator evidence cache drifted"
        )
    evidences = _deserialize_evidence(payload.get("program_evidence"))
    programs = fixed_typed_programs()
    if (
        tuple(row.program_sha256 for row in evidences)
        != tuple(program.program_hash for program in programs)
        or tuple(row.program_length for row in evidences)
        != tuple(program.program_length for program in programs)
        or payload.get("evidence_set_sha256") != evidence_set_sha256(evidences)
    ):
        raise MuSiQueEvaluatorStageFormationError(
            "private evaluator evidence candidate identity drifted"
        )
    source = payload.get("source_binding")
    execution = payload.get("execution")
    if (
        not isinstance(source, Mapping)
        or source.get("block_id_hash") != stable_hash({"block": expected_stage})
        or not isinstance(execution, Mapping)
        or execution.get("candidate_count") != len(programs)
        or execution.get("item_count") != len(evidences[0].items)
        or execution.get("work_unit_count") != len(programs) * len(evidences[0].items)
        or execution.get("retrieval_call_count") != execution.get("work_unit_count")
        or execution.get("retrieval_terminal_count") != execution.get("work_unit_count")
        or execution.get("all_terminals_joined_before_support_scoring") is not True
        or any(execution.get(field) != 0 for field in (
            "generator_calls",
            "network_calls",
            "online_evaluator_calls",
            "retries",
            "replays",
            "resamples",
        ))
    ):
        raise MuSiQueEvaluatorStageFormationError(
            "private evaluator evidence execution contract drifted"
        )
    return evidences, payload, _sha256_bytes(raw)


def _public_receipt(
    *,
    schema: str,
    stage: str,
    implementation: Mapping[str, Any],
    cache: Mapping[str, Any],
    cache_file_sha256: str,
    core_receipt: Mapping[str, Any],
    upstream: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    execution = cache["execution"]
    body: dict[str, Any] = {
        "schema": schema,
        "status": "formed_offline_on_exact_committed_block",
        "stage": stage,
        "implementation": dict(implementation),
        "source_binding": dict(cache["source_binding"]),
        "candidate_set_binding": dict(cache["candidate_set_binding"]),
        "evidence_binding": {
            "evidence_cache_file_sha256": cache_file_sha256,
            "evidence_cache_sha256": cache["cache_sha256"],
            "evidence_set_sha256": cache["evidence_set_sha256"],
            "item_level_evidence_persisted_publicly": False,
            "private_paths_persisted_publicly": False,
        },
        "core_receipt": dict(core_receipt),
        "execution": {
            "candidate_count": execution["candidate_count"],
            "item_count": execution["item_count"],
            "work_unit_count": execution["work_unit_count"],
            "retrieval_call_count": execution["retrieval_call_count"],
            "retrieval_terminal_count": execution["retrieval_terminal_count"],
            "configured_maximum_concurrency": execution[
                "configured_maximum_concurrency"
            ],
            "all_terminals_joined_before_support_scoring": True,
            "invalid_terminal_count": execution["invalid_terminal_count"],
        },
        "upstream_binding": None if upstream is None else dict(upstream),
        "offline_contract": {
            "model_calls": 0,
            "generator_calls": 0,
            "network_calls": 0,
            "online_evaluator_calls": 0,
            "measurement_blocks_accessed": 0,
            "retries": 0,
            "replays": 0,
            "resamples": 0,
        },
        "raw_content_persisted": False,
    }
    receipt = {**body, "receipt_sha256": stable_hash(body)}
    _assert_public_safe(receipt)
    return receipt


def _validate_public_shell(
    payload: Mapping[str, Any],
    *,
    stage: str,
    cache: Mapping[str, Any],
    cache_file_sha256: str,
    verify_live: bool,
    project_root: str | Path | None,
) -> None:
    body = dict(payload)
    declared = _require_sha256(body.pop("receipt_sha256", None), "public receipt hash")
    schema = A_FORM_PUBLIC_SCHEMA if stage == "A_form" else F3_PUBLIC_SCHEMA
    expected_keys = {
        "candidate_set_binding",
        "core_receipt",
        "evidence_binding",
        "execution",
        "implementation",
        "offline_contract",
        "raw_content_persisted",
        "receipt_sha256",
        "schema",
        "source_binding",
        "stage",
        "status",
        "upstream_binding",
    }
    if (
        set(payload) != expected_keys
        or payload.get("schema") != schema
        or payload.get("stage") != stage
        or payload.get("status") != "formed_offline_on_exact_committed_block"
        or stable_hash(body) != declared
        or payload.get("raw_content_persisted") is not False
        or payload.get("source_binding") != cache.get("source_binding")
        or payload.get("candidate_set_binding") != cache.get("candidate_set_binding")
    ):
        raise MuSiQueEvaluatorStageFormationError(
            f"{stage} public formation receipt drifted"
        )
    evidence_binding = payload.get("evidence_binding")
    offline = payload.get("offline_contract")
    if (
        not isinstance(evidence_binding, Mapping)
        or evidence_binding.get("evidence_cache_file_sha256") != cache_file_sha256
        or evidence_binding.get("evidence_cache_sha256") != cache.get("cache_sha256")
        or evidence_binding.get("evidence_set_sha256")
        != cache.get("evidence_set_sha256")
        or evidence_binding.get("item_level_evidence_persisted_publicly") is not False
        or evidence_binding.get("private_paths_persisted_publicly") is not False
        or not isinstance(offline, Mapping)
        or offline.get("measurement_blocks_accessed") != 0
        or any(offline.get(field) != 0 for field in (
            "model_calls",
            "generator_calls",
            "network_calls",
            "online_evaluator_calls",
            "retries",
            "replays",
            "resamples",
        ))
    ):
        raise MuSiQueEvaluatorStageFormationError(
            f"{stage} public evidence or offline binding drifted"
        )
    implementation = _validate_implementation_binding(payload.get("implementation"))
    if verify_live and implementation != current_evaluator_stage_implementation_binding(
        project_root
    ):
        raise MuSiQueEvaluatorStageFormationError(
            "live evaluator formation implementation drifted"
        )
    _assert_public_safe(payload)


def load_a_form_bundle(
    *,
    private_evidence_path: str | Path,
    public_receipt_path: str | Path,
    verify_live: bool = True,
    project_root: str | Path | None = None,
) -> tuple[tuple[ProgramRetrievalEvidence, ...], dict[str, Any], dict[str, Any]]:
    evidence, cache, cache_file_hash = load_private_evidence_cache(
        private_evidence_path, expected_stage="A_form"
    )
    public, _raw = _read_json_object(public_receipt_path, "A_form public receipt")
    _validate_public_shell(
        public,
        stage="A_form",
        cache=cache,
        cache_file_sha256=cache_file_hash,
        verify_live=verify_live,
        project_root=project_root,
    )
    expected_core = form_evaluator_challenger(evidence)
    if public.get("core_receipt") != expected_core or public.get("upstream_binding") is not None:
        raise MuSiQueEvaluatorStageFormationError(
            "A_form core evaluator receipt drifted"
        )
    return evidence, cache, public


def load_f3_bundle(
    *,
    private_evidence_path: str | Path,
    public_receipt_path: str | Path,
    a_form_private_evidence_path: str | Path,
    a_form_public_receipt_path: str | Path,
    verify_live: bool = True,
    project_root: str | Path | None = None,
) -> tuple[tuple[ProgramRetrievalEvidence, ...], dict[str, Any], dict[str, Any]]:
    a_evidence, a_cache, a_public = load_a_form_bundle(
        private_evidence_path=a_form_private_evidence_path,
        public_receipt_path=a_form_public_receipt_path,
        verify_live=verify_live,
        project_root=project_root,
    )
    evidence, cache, cache_file_hash = load_private_evidence_cache(
        private_evidence_path, expected_stage="F3"
    )
    public, _raw = _read_json_object(public_receipt_path, "F3 public receipt")
    _validate_public_shell(
        public,
        stage="F3",
        cache=cache,
        cache_file_sha256=cache_file_hash,
        verify_live=verify_live,
        project_root=project_root,
    )
    a_public_path = _absolute_no_symlink(
        a_form_public_receipt_path, "A_form public receipt"
    )
    a_cache_path = _absolute_no_symlink(
        a_form_private_evidence_path, "A_form private evidence cache"
    )
    upstream = {
        "a_form_public_receipt_sha256": a_public["receipt_sha256"],
        "a_form_public_file_sha256": _sha256_file(a_public_path),
        "a_form_evidence_cache_sha256": a_cache["cache_sha256"],
        "a_form_evidence_file_sha256": _sha256_file(a_cache_path),
        "a_form_evidence_set_sha256": a_cache["evidence_set_sha256"],
    }
    expected_core = freeze_prospective_search_formation(
        formation_evidence=evidence,
        evaluator_formation_evidence=a_evidence,
        evaluator_formation_receipt=a_public["core_receipt"],
    )
    if (
        public.get("core_receipt") != expected_core
        or public.get("upstream_binding") != upstream
        or cache["source_binding"].get("acquisition_sha256")
        != a_cache["source_binding"].get("acquisition_sha256")
        or cache["source_binding"].get("private_pack_sha256")
        != a_cache["source_binding"].get("private_pack_sha256")
    ):
        raise MuSiQueEvaluatorStageFormationError(
            "F3 prospective formation or A_form binding drifted"
        )
    return evidence, cache, public


def _prepare_output_paths(
    private_evidence_path: str | Path,
    public_receipt_path: str | Path,
) -> tuple[Path, Path]:
    private_path = _absolute_no_symlink(
        private_evidence_path, "private evaluator evidence output"
    )
    public_path = _absolute_no_symlink(
        public_receipt_path, "public evaluator receipt output"
    )
    if private_path == public_path or private_path.exists() or public_path.exists():
        raise MuSiQueEvaluatorStageFormationError(
            "evaluator formation outputs must be distinct fresh files"
        )
    _require_private_cache_boundary(private_path)
    return private_path, public_path


def form_a_form_stage(
    *,
    block_path: str | Path,
    acquisition_receipt_path: str | Path,
    private_evidence_output_path: str | Path,
    public_receipt_output_path: str | Path,
) -> dict[str, Any]:
    """Form the evaluator challenger from exact A_form only."""

    private_path, public_path = _prepare_output_paths(
        private_evidence_output_path, public_receipt_output_path
    )
    implementation = current_evaluator_stage_implementation_binding()
    block = load_formation_block(
        block_path=block_path,
        acquisition_receipt_path=acquisition_receipt_path,
        expected_block="A_form",
    )
    evidences, execution = build_program_retrieval_evidence(block)
    cache = _cache_payload(
        stage="A_form",
        block=block,
        evidences=evidences,
        execution=execution,
    )
    _write_json_exclusive(private_path, cache)
    cache_file_hash = _sha256_file(private_path)
    core = form_evaluator_challenger(evidences)
    receipt = _public_receipt(
        schema=A_FORM_PUBLIC_SCHEMA,
        stage="A_form",
        implementation=implementation,
        cache=cache,
        cache_file_sha256=cache_file_hash,
        core_receipt=core,
    )
    _write_json_exclusive(public_path, receipt)
    return receipt


def form_f3_stage(
    *,
    block_path: str | Path,
    acquisition_receipt_path: str | Path,
    a_form_private_evidence_path: str | Path,
    a_form_public_receipt_path: str | Path,
    private_evidence_output_path: str | Path,
    public_receipt_output_path: str | Path,
) -> dict[str, Any]:
    """Freeze incumbent/challenger candidate selections from exact F3."""

    private_path, public_path = _prepare_output_paths(
        private_evidence_output_path, public_receipt_output_path
    )
    implementation = current_evaluator_stage_implementation_binding()
    a_evidence, a_cache, a_public = load_a_form_bundle(
        private_evidence_path=a_form_private_evidence_path,
        public_receipt_path=a_form_public_receipt_path,
        verify_live=True,
    )
    block = load_formation_block(
        block_path=block_path,
        acquisition_receipt_path=acquisition_receipt_path,
        expected_block="F3",
    )
    if (
        block.acquisition.acquisition_sha256
        != a_cache["source_binding"].get("acquisition_sha256")
        or block.acquisition.private_pack_sha256
        != a_cache["source_binding"].get("private_pack_sha256")
    ):
        raise MuSiQueEvaluatorStageFormationError(
            "F3 and A_form do not belong to the same acquired pack"
        )
    evidences, execution = build_program_retrieval_evidence(block)
    cache = _cache_payload(
        stage="F3",
        block=block,
        evidences=evidences,
        execution=execution,
    )
    _write_json_exclusive(private_path, cache)
    core = freeze_prospective_search_formation(
        formation_evidence=evidences,
        evaluator_formation_evidence=a_evidence,
        evaluator_formation_receipt=a_public["core_receipt"],
    )
    a_public_path = _absolute_no_symlink(
        a_form_public_receipt_path, "A_form public receipt"
    )
    a_private_path = _absolute_no_symlink(
        a_form_private_evidence_path, "A_form private evidence cache"
    )
    upstream = {
        "a_form_public_receipt_sha256": a_public["receipt_sha256"],
        "a_form_public_file_sha256": _sha256_file(a_public_path),
        "a_form_evidence_cache_sha256": a_cache["cache_sha256"],
        "a_form_evidence_file_sha256": _sha256_file(a_private_path),
        "a_form_evidence_set_sha256": a_cache["evidence_set_sha256"],
    }
    receipt = _public_receipt(
        schema=F3_PUBLIC_SCHEMA,
        stage="F3",
        implementation=implementation,
        cache=cache,
        cache_file_sha256=_sha256_file(private_path),
        core_receipt=core,
        upstream=upstream,
    )
    _write_json_exclusive(public_path, receipt)
    return receipt


__all__ = [
    "MuSiQueEvaluatorStageFormationError",
    "build_program_retrieval_evidence",
    "candidate_set_binding",
    "current_evaluator_stage_implementation_binding",
    "evidence_set_sha256",
    "fixed_typed_programs",
    "form_a_form_stage",
    "form_f3_stage",
    "load_a_form_bundle",
    "load_f3_bundle",
    "load_private_evidence_cache",
]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    a_form = subparsers.add_parser("a-form")
    f3 = subparsers.add_parser("f3")
    for command in (a_form, f3):
        command.add_argument("--block", type=Path, required=True)
        command.add_argument("--acquisition-receipt", type=Path, required=True)
        command.add_argument("--private-evidence-output", type=Path, required=True)
        command.add_argument("--public-receipt-output", type=Path, required=True)
    f3.add_argument("--a-form-private-evidence", type=Path, required=True)
    f3.add_argument("--a-form-public-receipt", type=Path, required=True)
    arguments = parser.parse_args(argv)
    common = {
        "block_path": arguments.block,
        "acquisition_receipt_path": arguments.acquisition_receipt,
        "private_evidence_output_path": arguments.private_evidence_output,
        "public_receipt_output_path": arguments.public_receipt_output,
    }
    if arguments.command == "a-form":
        form_a_form_stage(**common)
    else:
        form_f3_stage(
            **common,
            a_form_private_evidence_path=arguments.a_form_private_evidence,
            a_form_public_receipt_path=arguments.a_form_public_receipt,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
