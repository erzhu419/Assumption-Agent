"""Fresh offline 2Wiki zero-shot evaluator transfer.

The retrievers in this module are not selected on 2Wiki.  Retained P and both
two-Q actions are materialised by exact hash from the clean typed-program
registry and the two committed public MuSiQue formation receipts.  A_hold is
the only promotion measurement.  Its 48 items execute as two eager 192-party
waves (24 items x eight physical components); M_search remains unopened unless
the exact paired promotion rule succeeds.

Formal execution is deliberately available only through this module's CLI.
No study data is read at import time or by either freeze builder.
"""

from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures
from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import threading
from typing import Any, Callable, Mapping, Sequence

from ..models import stable_hash
from .hotpot_family_out_acquisition_v1 import committed_public_file_receipt
from .hotpot_family_out_runner_v1 import _probe_bubblewrap, verify_capability_receipt
from .musique_formal_runtime_binding_v2 import PreparedFormalRuntimeV2, prepare_formal_runtime_v2
from .musique_typed_retriever_formation_v1 import (
    RetrievalParagraph,
    TypedRetrievalProgram,
    enumerate_programs,
    retrieve as typed_retrieve,
    verify_live_implementation as verify_typed_implementation,
)


VERSION = "twowiki_evaluator_zero_shot_transfer_v1"
DESIGN_SCHEMA = VERSION.replace("_v1", "_design_v1")
FREEZE_A_SCHEMA = f"{VERSION}_A_hold_pre_run_freeze"
REPORT_A_SCHEMA = f"{VERSION}_A_hold_aggregate_report"
FREEZE_M_SCHEMA = f"{VERSION}_M_search_pre_run_freeze"
REPORT_M_SCHEMA = f"{VERSION}_M_search_aggregate_report"
CONSUMPTION_SCHEMA = f"{VERSION}_authorization_consumption"
FAILURE_SCHEMA = f"{VERSION}_failure"

DESIGN_RELATIVE = "manifests/twowiki_evaluator_zero_shot_transfer_design_v1.json"
A_FORM_RELATIVE = "manifests/musique_evaluator_portfolio_a_form_receipt_v1.json"
F_SEARCH_RELATIVE = "manifests/musique_evaluator_portfolio_f_search_receipt_v1.json"
P_PROGRAM_RELATIVE = "manifests/musique_recursive_study_f1_formation_v1/frozen_program.json"
A_HOLD_BLOCK_RELATIVE = (
    "artifacts/twowiki_evaluator_zero_shot_transfer_v1/private_pack/A_hold.jsonl"
)
M_SEARCH_BLOCK_RELATIVE = (
    "artifacts/twowiki_evaluator_zero_shot_transfer_v1/private_pack/M_search.jsonl"
)
A_FREEZE_RELATIVE = (
    "manifests/twowiki_evaluator_zero_shot_transfer_a_hold_pre_run_freeze_v1.json"
)
A_REPORT_RELATIVE = (
    "manifests/twowiki_evaluator_zero_shot_transfer_a_hold_aggregate_report_v1.json"
)
M_FREEZE_RELATIVE = (
    "manifests/twowiki_evaluator_zero_shot_transfer_m_search_pre_run_freeze_v1.json"
)
M_REPORT_RELATIVE = (
    "manifests/twowiki_evaluator_zero_shot_transfer_m_search_aggregate_report_v1.json"
)
A_EXECUTION_ROOT_RELATIVE = (
    "artifacts/twowiki_evaluator_zero_shot_transfer_v1/a_hold_formal_root_v1"
)
M_EXECUTION_ROOT_RELATIVE = (
    "artifacts/twowiki_evaluator_zero_shot_transfer_v1/m_search_formal_root_v1"
)
CAPACITY_RECEIPT_RELATIVE = (
    "manifests/hotpot_recursive_official_capacity24_diagnostic_v1.json"
)
CAPACITY_RECEIPT_FILE_SHA256 = (
    "884b6d496b03a586d0ed8bbba3b2bf17f73b54edc60e5a255bae24b998e5217a"
)
CAPACITY_RECEIPT_SHA256 = (
    "b463c5ea7374def703d67a7584d2c29c474fc1b2d8a644bd0d139fce7ba24e2a"
)

DESIGN_FILE_SHA256 = "1a5ab0d806324c721ff7ddc48ac7b22de94abadf12e2887b182a1af76db755ba"
DESIGN_SEMANTIC_SHA256 = "903cf6dee77dedab34894330b1ae54b3893d6a2648392fb0cdd6f7569c354754"
TOP_K = 5
A_HOLD_ITEM_COUNT = 48
M_SEARCH_ITEM_COUNT = 24
WAVE_ITEM_COUNT = 24
COMPONENT_COUNT = 8
WAVE_PARTY_COUNT = WAVE_ITEM_COUNT * COMPONENT_COUNT
PROMOTION_ALPHA = Fraction(1, 10)
OFFICIAL_CONCURRENCY_CAP = 24

A_COMPONENT_IDS = (
    "canonical_RAW",
    "incumbent_P",
    "incumbent_Q1",
    "incumbent_Q2",
    "challenger_P",
    "challenger_Q1",
    "challenger_Q2",
    "official_HippoRAG_core_item_local",
)
M_COMPONENT_IDS = (
    "canonical_RAW",
    "incumbent_P",
    "incumbent_Q1",
    "incumbent_Q2",
    "active_P",
    "active_Q1",
    "active_Q2",
    "official_HippoRAG_core_item_local",
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_CLEAN_MODULE_CLI_ACTIVE = False


class TwoWikiZeroShotTransferError(RuntimeError):
    """A frozen binding, custody boundary, or one-shot execution drifted."""


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: str | Path) -> str:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise TwoWikiZeroShotTransferError("required file unavailable")
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise TwoWikiZeroShotTransferError(f"{field} must be lowercase sha256")
    return value


def _read_json(path: str | Path, field: str) -> tuple[dict[str, Any], bytes]:
    candidate = Path(path).absolute()
    if candidate.is_symlink() or not candidate.is_file():
        raise TwoWikiZeroShotTransferError(f"{field} unavailable")
    try:
        raw = candidate.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TwoWikiZeroShotTransferError(f"{field} invalid") from exc
    if not isinstance(value, dict):
        raise TwoWikiZeroShotTransferError(f"{field} must be an object")
    return value, raw


def _atomic_write_exclusive(destination: Path, raw: bytes, *, mode: int) -> None:
    temporary = destination.parent / f".{destination.name}.{os.urandom(12).hex()}.tmp"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination, follow_symlinks=False)
        finally:
            temporary.unlink(missing_ok=True)
        directory = os.open(
            destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_json_exclusive(
    path: str | Path, value: Mapping[str, Any], *, mode: int = 0o600
) -> None:
    destination = Path(path).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw = json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2).encode() + b"\n"
    _atomic_write_exclusive(destination, raw, mode=mode)


def _prepare_output(path: str | Path) -> None:
    destination = Path(path).absolute()
    if destination.exists() or destination.is_symlink():
        raise TwoWikiZeroShotTransferError("formal output already exists")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if destination.parent.is_symlink() or not destination.parent.is_dir():
        raise TwoWikiZeroShotTransferError("formal output parent unavailable")
    canary = destination.parent / f".{destination.name}.canary-{os.urandom(8).hex()}"
    try:
        _atomic_write_exclusive(canary, b"atomic-persistence-canary\n", mode=0o600)
        canary.unlink()
        directory = os.open(
            destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        canary.unlink(missing_ok=True)


def _canonical_unwritten_public_path(
    *, project: Path, supplied: str | Path, relative: str, field: str
) -> Path:
    expected = (project / relative).absolute()
    candidate = Path(supplied)
    if not candidate.is_absolute():
        candidate = project / candidate
    candidate = candidate.absolute()
    if candidate != expected or candidate.exists() or candidate.is_symlink():
        raise TwoWikiZeroShotTransferError(
            f"{field} must use its fixed unwritten canonical path"
        )
    return expected


def _canonical_committed_public_path(
    *, project: Path, supplied: str | Path, relative: str, field: str
) -> tuple[Path, str]:
    expected = project / relative
    candidate = Path(supplied)
    if not candidate.is_absolute():
        candidate = project / candidate
    try:
        actual = candidate.resolve(strict=True)
        canonical = expected.resolve(strict=True)
    except FileNotFoundError as exc:
        raise TwoWikiZeroShotTransferError(f"canonical {field} unavailable") from exc
    if actual != canonical or candidate.is_symlink():
        raise TwoWikiZeroShotTransferError(f"{field} must use its fixed canonical path")
    try:
        custody = committed_public_file_receipt(project=project, path=canonical)
    except Exception as exc:
        raise TwoWikiZeroShotTransferError(
            f"{field} must be the clean tracked HEAD blob"
        ) from exc
    file_sha = custody.get("preregistration_file_sha256")
    head_sha = custody.get("preregistration_head_blob_sha256")
    if (
        not isinstance(file_sha, str)
        or file_sha != head_sha
        or file_sha != _sha256_file(canonical)
    ):
        raise TwoWikiZeroShotTransferError(f"{field} committed custody drifted")
    return canonical, file_sha


def _canonical_execution_root(
    *, project: Path, supplied: str | Path, stage: str
) -> Path:
    relative = (
        A_EXECUTION_ROOT_RELATIVE if stage == "A_hold" else M_EXECUTION_ROOT_RELATIVE
    )
    expected = (project / relative).absolute()
    candidate = Path(supplied)
    if not candidate.is_absolute():
        candidate = project / candidate
    if candidate.absolute() != expected:
        raise TwoWikiZeroShotTransferError(
            f"{stage} execution root must use its fixed canonical path"
        )
    return _new_root(expected)


def _assert_public_safe(value: Mapping[str, Any]) -> None:
    serialized = json.dumps(value, ensure_ascii=True, sort_keys=True)
    forbidden = (
        '"answers"', '"canonical_row_sha256"', '"corpus"', '"item_id"',
        '"normalized_answers"', '"paragraph_text"', '"question"',
        '"source_row_sha256"', '"support_indices"', '"private_root"',
    )
    if any(token in serialized for token in forbidden):
        raise TwoWikiZeroShotTransferError("public artifact contains private content")

    def walk(node: object) -> None:
        if isinstance(node, Mapping):
            for key, child in node.items():
                if key in {"path", "private_path", "private_locator"} and isinstance(child, str):
                    if Path(child).is_absolute():
                        raise TwoWikiZeroShotTransferError(
                            "public artifact contains an absolute private locator"
                        )
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(value)


def _require_private_boundary(path: str | Path) -> None:
    destination = Path(path).absolute()
    probe = subprocess.run(
        ["git", "-C", str(destination.parent), "rev-parse", "--show-toplevel"],
        check=False, capture_output=True, text=True, timeout=30,
    )
    if probe.returncode != 0:
        return
    repository = Path(probe.stdout.strip()).resolve(strict=True)
    try:
        relative = destination.resolve(strict=False).relative_to(repository).as_posix()
    except ValueError as exc:
        raise TwoWikiZeroShotTransferError("private boundary ambiguous") from exc
    ignored = subprocess.run(
        ["git", "-C", str(repository), "check-ignore", "--no-index", "-q", "--", relative],
        check=False, capture_output=True, timeout=30,
    )
    tracked = subprocess.run(
        ["git", "-C", str(repository), "ls-files", "--error-unmatch", "--", relative],
        check=False, capture_output=True, timeout=30,
    )
    if ignored.returncode != 0 or tracked.returncode == 0:
        raise TwoWikiZeroShotTransferError("execution root must be ignored and untracked")


def _new_root(path: str | Path) -> Path:
    candidate = Path(path).absolute()
    if candidate.is_symlink() or not candidate.parent.resolve(strict=True).is_dir():
        raise TwoWikiZeroShotTransferError("execution-root parent unavailable")
    return candidate


def _root_hash(path: str | Path) -> str:
    return stable_hash({"absolute_execution_root": str(_new_root(path))})


def _acquisition_module() -> Any:
    try:
        return importlib.import_module(
            ".twowiki_evaluator_zero_shot_transfer_acquisition_v1", package=__package__
        )
    except (ImportError, AttributeError) as exc:
        raise TwoWikiZeroShotTransferError("acquisition implementation unavailable") from exc


def _load_design(project: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    design, raw = _read_json(project / DESIGN_RELATIVE, "zero-shot design")
    body = dict(design)
    declared = _require_sha256(body.pop("design_sha256", None), "design")
    if (
        _sha256_bytes(raw) != DESIGN_FILE_SHA256
        or declared != DESIGN_SEMANTIC_SHA256
        or stable_hash(body) != declared
        or design.get("schema") != DESIGN_SCHEMA
        or design.get("status")
        != "fixed_zero_shot_transfer_before_private_selection_or_any_retrieval_score"
    ):
        raise TwoWikiZeroShotTransferError("zero-shot design drifted")
    return design, {
        "relative_path": DESIGN_RELATIVE,
        "file_sha256": DESIGN_FILE_SHA256,
        "design_sha256": declared,
    }


def current_implementation_binding(project_root: str | Path) -> dict[str, Any]:
    """Bind the clean tracked runner plus the acquisition implementation closure."""

    project = Path(project_root).resolve(strict=True)
    relative = "assumption_agent/benchmarks/twowiki_evaluator_zero_shot_transfer_v1.py"
    path = project / relative
    if (
        path.is_symlink()
        or not path.is_file()
        or Path(__file__).resolve(strict=True) != path.resolve(strict=True)
    ):
        raise TwoWikiZeroShotTransferError("runner implementation unavailable")
    tracked = subprocess.run(
        ["git", "-C", str(project), "ls-files", "--error-unmatch", "--", relative],
        check=False, capture_output=True, timeout=30,
    )
    clean = subprocess.run(
        ["git", "-C", str(project), "diff", "--quiet", "HEAD", "--", relative],
        check=False, capture_output=True, timeout=30,
    )
    if tracked.returncode != 0 or clean.returncode != 0:
        raise TwoWikiZeroShotTransferError("runner is not the clean tracked HEAD file")
    module = _acquisition_module()
    acquisition_relative = (
        "assumption_agent/benchmarks/"
        "twowiki_evaluator_zero_shot_transfer_acquisition_v1.py"
    )
    acquisition_loaded = Path(str(getattr(module, "__file__", "")))
    typed_loaded = Path(str(inspect.getsourcefile(typed_retrieve) or ""))
    typed_expected = (
        project
        / "assumption_agent/benchmarks/musique_typed_retriever_formation_v1.py"
    )
    if (
        not acquisition_loaded.is_file()
        or acquisition_loaded.resolve(strict=True)
        != (project / acquisition_relative).resolve(strict=True)
        or not typed_loaded.is_file()
        or typed_loaded.resolve(strict=True) != typed_expected.resolve(strict=True)
    ):
        raise TwoWikiZeroShotTransferError("loaded implementation module path drifted")
    try:
        acquisition = module.implementation_binding(project)
    except Exception as exc:
        raise TwoWikiZeroShotTransferError("acquisition closure unavailable") from exc
    files = [{"path": relative, "sha256": _sha256_file(path)}]
    body = {"schema": f"{VERSION}_implementation", "files": files, "acquisition": acquisition}
    return {**body, "set_sha256": stable_hash(body)}


def _load_self_hashed_receipt(path: Path, *, expected_file: str, expected_semantic: str) -> dict[str, Any]:
    value, raw = _read_json(path, "public formation receipt")
    body = dict(value)
    declared = _require_sha256(body.pop("receipt_sha256", None), "formation receipt")
    if _sha256_bytes(raw) != expected_file or declared != expected_semantic or stable_hash(body) != declared:
        raise TwoWikiZeroShotTransferError("public formation receipt drifted")
    _assert_public_safe(value)
    return value


@dataclass(frozen=True)
class FixedActions:
    retained_p: TypedRetrievalProgram = field(repr=False)
    a_incumbent: tuple[TypedRetrievalProgram, TypedRetrievalProgram] = field(repr=False)
    a_challenger: tuple[TypedRetrievalProgram, TypedRetrievalProgram] = field(repr=False)
    f_incumbent: tuple[TypedRetrievalProgram, TypedRetrievalProgram] = field(repr=False)
    f_challenger: tuple[TypedRetrievalProgram, TypedRetrievalProgram] = field(repr=False)
    public_binding: dict[str, Any]


def _program_registry() -> dict[str, TypedRetrievalProgram]:
    programs = tuple(enumerate_programs())
    registry = {row.program_hash: row for row in programs}
    if len(programs) != 84 or len(registry) != 84 or any(row.type_issues() for row in programs):
        raise TwoWikiZeroShotTransferError("fixed typed registry drifted")
    return registry


def _load_fixed_actions(project: Path) -> FixedActions:
    design, _ = _load_design(project)
    fixed = design["fixed_actions"]
    a_meta = fixed["A_form_public_receipt"]
    f_meta = fixed["F_search_public_receipt"]
    a = _load_self_hashed_receipt(
        project / A_FORM_RELATIVE,
        expected_file=a_meta["file_sha256"], expected_semantic=a_meta["semantic_sha256"],
    )
    f = _load_self_hashed_receipt(
        project / F_SEARCH_RELATIVE,
        expected_file=f_meta["file_sha256"], expected_semantic=f_meta["semantic_sha256"],
    )
    if (
        a_meta.get("relative_path") != A_FORM_RELATIVE
        or f_meta.get("relative_path") != F_SEARCH_RELATIVE
        or a["formation_core"].get("formation_sha256") != a_meta["formation_sha256"]
        or f["formation_core"].get("formation_sha256") != f_meta["formation_sha256"]
    ):
        raise TwoWikiZeroShotTransferError("public action lineage drifted")
    p_meta = fixed["retained_P"]
    p_path = project / P_PROGRAM_RELATIVE
    p_json, p_raw = _read_json(p_path, "retained P")
    if (
        p_meta.get("relative_path") != P_PROGRAM_RELATIVE
        or _sha256_bytes(p_raw) != p_meta.get("file_sha256")
        or p_json.get("program_hash") != p_meta.get("program_sha256")
    ):
        raise TwoWikiZeroShotTransferError("retained P lineage drifted")
    try:
        p_program = TypedRetrievalProgram.from_dict(p_json["program"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TwoWikiZeroShotTransferError("retained P program malformed") from exc
    if p_program.type_issues() or p_program.program_hash != p_meta["program_sha256"]:
        raise TwoWikiZeroShotTransferError("retained P program drifted")
    try:
        verify_typed_implementation(
            p_json["typed_dsl_implementation"], project_root=project
        )
    except Exception as exc:
        raise TwoWikiZeroShotTransferError(
            "live typed retrieval implementation drifted"
        ) from exc
    registry = _program_registry()

    def pair(receipt: Mapping[str, Any], role: str, exact: Mapping[str, Any]) -> tuple[TypedRetrievalProgram, TypedRetrievalProgram]:
        selected = receipt["formation_core"][role]
        hashes = selected.get("program_sha256s")
        if (
            selected.get("action_sha256") != exact.get("action_sha256")
            or hashes != exact.get("Q_program_sha256s")
            or not isinstance(hashes, list) or len(hashes) != 2 or hashes[0] == hashes[1]
            or any(value not in registry for value in hashes)
            or p_program.program_hash in hashes
        ):
            raise TwoWikiZeroShotTransferError("exact public action drifted")
        return registry[hashes[0]], registry[hashes[1]]

    a_inc = pair(a, "incumbent", fixed["A_incumbent"])
    a_chal = pair(a, "challenger", fixed["A_challenger"])
    f_inc = pair(f, "incumbent", fixed["F_incumbent"])
    f_chal = pair(f, "challenger", fixed["F_challenger_if_promoted_active"])
    binding = {
        "retained_P_program_sha256": p_program.program_hash,
        "A_incumbent": fixed["A_incumbent"],
        "A_challenger": fixed["A_challenger"],
        "F_incumbent": fixed["F_incumbent"],
        "F_challenger_if_promoted_active": fixed["F_challenger_if_promoted_active"],
        "A_form_receipt": a_meta,
        "F_search_receipt": f_meta,
        "program_materialization": "clean_fixed_registry_lookup_by_public_program_sha256",
        "private_MuSiQue_formation_cache_opened": False,
    }
    return FixedActions(p_program, a_inc, a_chal, f_inc, f_chal, binding)


@dataclass(frozen=True)
class RetrievalItem:
    question: str = field(repr=False)
    corpus: tuple[RetrievalParagraph, ...] = field(repr=False)
    item_commitment_sha256: str

    def hipporag_paragraphs(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {"idx": row.idx, "title": row.title, "paragraph_text": row.text}
            for row in self.corpus
        )


@dataclass(frozen=True)
class StudyItem:
    view: RetrievalItem = field(repr=False)
    support_indices: tuple[int, ...] = field(repr=False)


def _row_to_item(row: Mapping[str, Any], *, expected_block: str) -> StudyItem:
    expected_keys = {
        "schema", "block", "source_member", "question_type", "item_id", "question",
        "corpus", "answers", "normalized_answers", "support_indices",
        "source_row_sha256", "normalized_question_sha256",
        "canonical_question_plus_ordered_context_sha256", "canonical_row_sha256",
    }
    corpus_raw = row.get("corpus")
    support_raw = row.get("support_indices")
    question = row.get("question")
    item_id = row.get("item_id")
    answers = row.get("answers")
    normalized_answers = row.get("normalized_answers")
    if (
        set(row) != expected_keys or row.get("block") != expected_block
        or row.get("source_member") != ("train.json" if expected_block == "A_hold" else "dev.json")
        or row.get("question_type") not in {"bridge_comparison", "comparison", "compositional", "inference"}
        or row.get("schema")
        != "twowiki_evaluator_zero_shot_transfer_acquisition_v1_private_row"
        or not isinstance(item_id, str) or not item_id
        or not isinstance(question, str) or not question.strip()
        or not isinstance(answers, list) or not answers
        or any(not isinstance(value, str) or not value for value in answers)
        or not isinstance(normalized_answers, list) or not normalized_answers
        or any(not isinstance(value, str) or not value for value in normalized_answers)
        or not isinstance(corpus_raw, list) or len(corpus_raw) != 10
        or not isinstance(support_raw, list) or not support_raw
        or any(type(value) is not int for value in support_raw)
        or len(set(support_raw)) != len(support_raw)
    ):
        raise TwoWikiZeroShotTransferError("private 2Wiki row schema drifted")
    for field_name in (
        "source_row_sha256", "normalized_question_sha256",
        "canonical_question_plus_ordered_context_sha256", "canonical_row_sha256",
    ):
        _require_sha256(row.get(field_name), field_name)
    corpus: list[RetrievalParagraph] = []
    titles: set[str] = set()
    for ordinal, paragraph in enumerate(corpus_raw):
        if not isinstance(paragraph, Mapping) or set(paragraph) != {
            "paragraph_idx", "paragraph_title", "paragraph_text"
        }:
            raise TwoWikiZeroShotTransferError("2Wiki paragraph schema drifted")
        idx = paragraph.get("paragraph_idx")
        title = paragraph.get("paragraph_title")
        text = paragraph.get("paragraph_text")
        if (
            type(idx) is not int or idx != ordinal or not isinstance(title, str)
            or not title.strip() or title in titles or not isinstance(text, str) or not text.strip()
        ):
            raise TwoWikiZeroShotTransferError("2Wiki paragraph value drifted")
        titles.add(title)
        corpus.append(RetrievalParagraph(idx=idx, title=title, text=text))
    if any(not 0 <= value < len(corpus) for value in support_raw):
        raise TwoWikiZeroShotTransferError("2Wiki support index out of range")
    return StudyItem(
        view=RetrievalItem(
            question=question,
            corpus=tuple(corpus),
            item_commitment_sha256=stable_hash(row),
        ),
        support_indices=tuple(support_raw),
    )


def _ranking(program: TypedRetrievalProgram, item: RetrievalItem) -> tuple[int, ...]:
    try:
        ranking = tuple(typed_retrieve(program, item.question, item.corpus))
    except (TypeError, ValueError) as exc:
        raise TwoWikiZeroShotTransferError("typed retrieval failed") from exc
    if (
        len(ranking) != TOP_K or len(set(ranking)) != TOP_K
        or any(type(value) is not int or not 0 <= value < len(item.corpus) for value in ranking)
    ):
        raise TwoWikiZeroShotTransferError("typed ranking drifted")
    return ranking


def _fuse(*rankings: Sequence[int]) -> tuple[int, ...]:
    if len(rankings) != 3:
        raise TwoWikiZeroShotTransferError("portfolio width drifted")
    score: Counter[int] = Counter()
    for ranking in rankings:
        for position, idx in enumerate(ranking):
            score[idx] += Fraction(1, 60 + position + 1)
    return tuple(sorted(score, key=lambda idx: (-score[idx], idx))[:TOP_K])


def exact_paired_sign_flip(deltas: Sequence[int]) -> dict[str, Any]:
    if not deltas or any(type(value) is not int for value in deltas):
        raise TwoWikiZeroShotTransferError("paired deltas malformed")
    observed = sum(deltas)
    distribution: Counter[int] = Counter({0: 1})
    magnitudes = [abs(value) for value in deltas if value]
    for magnitude in magnitudes:
        updated: Counter[int] = Counter()
        for subtotal, count in distribution.items():
            updated[subtotal + magnitude] += count
            updated[subtotal - magnitude] += count
        distribution = updated
    p_value = Fraction(
        sum(count for subtotal, count in distribution.items() if subtotal >= observed),
        1 << len(magnitudes),
    )
    positive = observed > 0
    exact = p_value <= PROMOTION_ALPHA
    return {
        "test": "one_sided_exact_magnitude_preserving_paired_sign_flip_v1",
        "observed_net_support_hits": observed,
        "nonzero_pair_count": len(magnitudes),
        "p_value_numerator": p_value.numerator,
        "p_value_denominator": p_value.denominator,
        "p_value": float(p_value),
        "alpha_numerator": 1,
        "alpha_denominator": 10,
        "positive_observed_net": positive,
        "exact_p_at_or_below_alpha": exact,
        "promoted": positive and exact,
        "sole_promotion_criterion": True,
    }


def _aggregate(arm: str, items: Sequence[StudyItem], rankings: Sequence[Sequence[int]]) -> dict[str, Any]:
    if not items or len(items) != len(rankings):
        raise TwoWikiZeroShotTransferError("aggregate length drifted")
    hits = total = complete = coverage = 0
    item_hits: list[int] = []
    for item, ranking in zip(items, rankings):
        supports = frozenset(item.support_indices)
        value = len(supports.intersection(ranking))
        item_hits.append(value)
        hits += value
        total += len(supports)
        complete += int(value == len(supports))
        coverage += int(value > 0)
    ratio = Fraction(hits, total)
    return {
        "arm_id": arm, "item_count": len(items), "support_hit_count": hits,
        "support_total": total, "support_recall_at_5_numerator": ratio.numerator,
        "support_recall_at_5_denominator": ratio.denominator,
        "complete_item_count": complete, "coverage_item_count": coverage,
        "item_hit_vector_sha256": stable_hash(item_hits),
    }


def _paired(left: str, right: str, items: Sequence[StudyItem], arms: Mapping[str, Sequence[Sequence[int]]]) -> dict[str, Any]:
    deltas = [
        len(frozenset(item.support_indices).intersection(arms[left][ordinal]))
        - len(frozenset(item.support_indices).intersection(arms[right][ordinal]))
        for ordinal, item in enumerate(items)
    ]
    return {
        "left_arm_id": left, "right_arm_id": right,
        "net_support_hit_count": sum(deltas),
        "gain_item_count": sum(value > 0 for value in deltas),
        "harm_item_count": sum(value < 0 for value in deltas),
        "tie_item_count": sum(value == 0 for value in deltas),
        "paired_delta_vector_sha256": stable_hash(deltas),
        "paired_test": exact_paired_sign_flip(deltas),
    }


def _descriptive_paired(
    left: str,
    right: str,
    items: Sequence[StudyItem],
    arms: Mapping[str, Sequence[Sequence[int]]],
) -> dict[str, Any]:
    comparison = _paired(left, right, items, arms)
    test = dict(comparison["paired_test"])
    criterion_met = bool(test.pop("promoted"))
    test["sole_promotion_criterion"] = False
    test["descriptive_only"] = True
    test["affects_L5_or_epoch"] = False
    test["positive_and_p_at_or_below_alpha"] = criterion_met
    comparison["paired_test"] = test
    comparison["descriptive_only"] = True
    comparison["affects_L5_or_epoch"] = False
    return comparison


def _prepare_runtime(
    *, project: Path, runtime_python: str | Path, local_llm_model: str | Path,
    local_embedding_model: str | Path, base_binding_receipt_path: str | Path,
    attestation_receipt_path: str | Path,
) -> PreparedFormalRuntimeV2:
    return prepare_formal_runtime_v2(
        project_root=project,
        attestation_receipt_path=Path(attestation_receipt_path).absolute(),
        base_binding_receipt_path=Path(base_binding_receipt_path).absolute(),
        runtime_python=Path(runtime_python).absolute(),
        local_llm_model=Path(local_llm_model).resolve(strict=True),
        local_embedding_model=Path(local_embedding_model).resolve(strict=True),
    )


def _capacity_receipt_binding(project: Path) -> dict[str, Any]:
    path, custody_sha = _canonical_committed_public_path(
        project=project,
        supplied=project / CAPACITY_RECEIPT_RELATIVE,
        relative=CAPACITY_RECEIPT_RELATIVE,
        field="official 24-worker capacity receipt",
    )
    receipt, raw = _read_json(path, "official 24-worker capacity receipt")
    body = dict(receipt)
    declared = _require_sha256(
        body.pop("receipt_sha256", None), "capacity receipt"
    )
    contract = receipt.get("diagnostic_contract")
    if (
        custody_sha != CAPACITY_RECEIPT_FILE_SHA256
        or _sha256_bytes(raw) != CAPACITY_RECEIPT_FILE_SHA256
        or declared != CAPACITY_RECEIPT_SHA256
        or stable_hash(body) != declared
        or receipt.get("schema")
        != "hotpot_recursive_official_capacity24_diagnostic_v1"
        or receipt.get("status") != "passed"
        or receipt.get("runtime_binding_sha256")
        != "04498dcbf70084959b9bc9cc1e0cfb451c50015614fc880ff21d0a59b0853df6"
        or not isinstance(contract, Mapping)
        or contract.get("official_worker_count") != OFFICIAL_CONCURRENCY_CAP
        or contract.get("terminal_count") != OFFICIAL_CONCURRENCY_CAP
        or contract.get("single_start_barrier") is not True
        or contract.get("fresh_runtime_postflight_passed") is not True
        or contract.get("private_or_scored_data_accessed") is not False
    ):
        raise TwoWikiZeroShotTransferError("official capacity receipt drifted")
    return {
        "relative_path": CAPACITY_RECEIPT_RELATIVE,
        "file_sha256": CAPACITY_RECEIPT_FILE_SHA256,
        "receipt_sha256": CAPACITY_RECEIPT_SHA256,
        "official_worker_count": OFFICIAL_CONCURRENCY_CAP,
        "clean_tracked_HEAD_blob": True,
    }


def _runtime_bundle(
    *, project: Path, capability_receipt_path: str | Path, runtime_python: str | Path,
    local_llm_model: str | Path, local_embedding_model: str | Path,
    base_binding_receipt_path: str | Path, attestation_receipt_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], PreparedFormalRuntimeV2]:
    capability, raw = verify_capability_receipt(capability_receipt_path)
    prepared = _prepare_runtime(
        project=project, runtime_python=runtime_python, local_llm_model=local_llm_model,
        local_embedding_model=local_embedding_model,
        base_binding_receipt_path=base_binding_receipt_path,
        attestation_receipt_path=attestation_receipt_path,
    )
    binding = {
        "file_sha256": _sha256_bytes(raw),
        "receipt_sha256": capability["receipt_sha256"],
        "bwrap_file_sha256": capability["bwrap_file_sha256"],
        "probe_contract_sha256": capability["probe_contract_sha256"],
        "fresh_probe_required_before_authorization": True,
        "capacity_receipt": _capacity_receipt_binding(project),
    }
    return capability, binding, prepared


def _fresh_probe(capability: Mapping[str, Any]) -> None:
    live = _probe_bubblewrap()
    if (
        live.get("probe_returncode") != 0
        or live.get("bwrap_file_sha256") != capability.get("bwrap_file_sha256")
        or live.get("probe_contract_sha256") != capability.get("probe_contract_sha256")
    ):
        raise TwoWikiZeroShotTransferError("fresh bwrap preflight drifted")


def _load_acquisition(
    *, project: Path, receipt_path: str | Path, selection_secret_path: str | Path
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    module = _acquisition_module()
    try:
        receipt, rows = module.load_acquisition_binding_live(
            project=project, path=receipt_path, selection_secret_path=selection_secret_path
        )
    except Exception as exc:
        raise TwoWikiZeroShotTransferError("canonical acquisition binding drifted") from exc
    canonical_receipt = project / module.ACQUISITION_RELATIVE
    raw = canonical_receipt.read_bytes()
    commitments = {row.block: row for row in rows}
    if (
        tuple(commitments) != ("A_hold", "M_search")
        or commitments["A_hold"].count != A_HOLD_ITEM_COUNT
        or commitments["M_search"].count != M_SEARCH_ITEM_COUNT
    ):
        raise TwoWikiZeroShotTransferError("acquisition block contract drifted")
    _assert_public_safe(receipt)
    return dict(receipt), raw, commitments


def _source_binding(receipt: Mapping[str, Any], raw: bytes, commitment: Any) -> dict[str, Any]:
    acquisition_sha = receipt.get("acquisition_sha256") or receipt.get("receipt_sha256")
    pack = receipt.get("private_pack_sha256")
    if pack is None and isinstance(receipt.get("commitments"), Mapping):
        pack = receipt["commitments"].get("private_pack_sha256")
    return {
        "acquisition_sha256": _require_sha256(acquisition_sha, "acquisition"),
        "acquisition_file_sha256": _sha256_bytes(raw),
        "private_pack_sha256": _require_sha256(pack, "private pack"),
        "measurement_block_id_hash": stable_hash({"block": commitment.block}),
        "measurement_source_member_hash": stable_hash({"source_member": commitment.source_member}),
        "measurement_block_file_sha256": _require_sha256(commitment.file_sha256, "block file"),
        "measurement_item_commitment_set_sha256": _require_sha256(
            commitment.item_commitment_set_sha256, "block item commitments"
        ),
        "measurement_item_count": commitment.count,
        "question_type_counts": dict(commitment.question_type_counts),
    }


def _execution_contract(stage: str) -> dict[str, Any]:
    if stage == "A_hold":
        return {
            "physical_component_ids": list(A_COMPONENT_IDS), "item_count": 48,
            "physical_work_unit_count": 384, "item_wave_count": 2,
            "items_per_wave": 24, "party_count_per_wave": 192,
            "maximum_concurrency": 192, "official_worker_concurrency_cap": 24,
            "all_wave_terminals_and_runtime_postflight_before_support_scoring": True,
            "promotion_comparison": "challenger_portfolio_minus_incumbent_portfolio",
            "promotion_requires_positive_net_and_exact_p_le_0_10": True,
            "promotion_validity_requires_all_eight_components_and_official_runtime_postflight": True,
            "control_scores_enter_promotion_statistic": False,
            "external_network_calls": 0, "online_evaluator_calls": 0,
            "retries": 0, "replays": 0, "resamples": 0,
        }
    return {
        "physical_component_ids": list(M_COMPONENT_IDS), "item_count": 24,
        "physical_work_unit_count": 192, "item_wave_count": 1,
        "items_per_wave": 24, "party_count_per_wave": 192,
        "maximum_concurrency": 192, "official_worker_concurrency_cap": 24,
        "all_terminals_and_runtime_postflight_before_support_scoring": True,
        "primary_comparison": "active_portfolio_minus_incumbent_portfolio",
        "L5_requires_positive_net_and_exact_p_le_0_10": True,
        "M_search_cannot_change_or_rollback_epoch": True,
        "external_network_calls": 0, "online_evaluator_calls": 0,
        "retries": 0, "replays": 0, "resamples": 0,
    }


def _submit_eager_then_join(
    *, executor: concurrent.futures.Executor,
    function: Callable[[Any], Any], work_units: Sequence[Any],
) -> tuple[Any, ...]:
    """Submit the entire barrier cohort before awaiting a single terminal."""

    futures = [executor.submit(function, unit) for unit in work_units]
    return tuple(future.result() for future in futures)


def _load_private_items(project: Path, block: str, commitment: Any) -> tuple[StudyItem, ...]:
    path = project / (A_HOLD_BLOCK_RELATIVE if block == "A_hold" else M_SEARCH_BLOCK_RELATIVE)
    module = _acquisition_module()
    try:
        rows = module.load_private_block(path, commitment=commitment, expected_block=block)
    except Exception as exc:
        raise TwoWikiZeroShotTransferError("private block validation failed") from exc
    items = tuple(_row_to_item(row, expected_block=block) for row in rows)
    if len(items) != commitment.count or len({row.view.item_commitment_sha256 for row in items}) != len(items):
        raise TwoWikiZeroShotTransferError("private block item closure drifted")
    return items


def _build_freeze_common(
    *, stage: str, project_root: str | Path, acquisition_receipt_path: str | Path,
    selection_secret_path: str | Path, capability_receipt_path: str | Path,
    runtime_python: str | Path, local_llm_model: str | Path,
    local_embedding_model: str | Path, base_binding_receipt_path: str | Path,
    attestation_receipt_path: str | Path, execution_root: str | Path,
) -> tuple[Path, dict[str, Any], FixedActions, dict[str, Any], dict[str, Any], PreparedFormalRuntimeV2]:
    project = Path(project_root).resolve(strict=True)
    design, design_binding = _load_design(project)
    actions = _load_fixed_actions(project)
    receipt, raw, commitments = _load_acquisition(
        project=project, receipt_path=acquisition_receipt_path,
        selection_secret_path=selection_secret_path,
    )
    capability, capability_binding, prepared = _runtime_bundle(
        project=project, capability_receipt_path=capability_receipt_path,
        runtime_python=runtime_python, local_llm_model=local_llm_model,
        local_embedding_model=local_embedding_model,
        base_binding_receipt_path=base_binding_receipt_path,
        attestation_receipt_path=attestation_receipt_path,
    )
    runtime_contract = design.get("runtime")
    if (
        not isinstance(runtime_contract, Mapping)
        or runtime_contract.get("runtime_binding_sha256")
        != prepared.safe_binding.get("binding_sha256")
        or runtime_contract.get("official_worker_concurrency_cap")
        != OFFICIAL_CONCURRENCY_CAP
        or runtime_contract.get("synthetic_24_worker_capacity_receipt_sha256")
        != CAPACITY_RECEIPT_SHA256
        or runtime_contract.get("external_network_calls") != 0
        or runtime_contract.get("online_evaluator_calls") != 0
    ):
        raise TwoWikiZeroShotTransferError("design-fixed official runtime drifted")
    common = {
        "implementation": current_implementation_binding(project),
        "design_binding": design_binding,
        "source_binding": _source_binding(receipt, raw, commitments[stage]),
        "fixed_action_binding": actions.public_binding,
        "capability_binding": capability_binding,
        "runtime_binding": prepared.safe_binding,
        "execution_contract": _execution_contract(stage),
        "execution_root_sha256": _root_hash(execution_root),
    }
    del design
    return project, common, actions, capability, receipt, prepared


def build_a_hold_pre_run_freeze(
    *, project_root: str | Path, acquisition_receipt_path: str | Path,
    selection_secret_path: str | Path, capability_receipt_path: str | Path,
    runtime_python: str | Path, local_llm_model: str | Path,
    local_embedding_model: str | Path, base_binding_receipt_path: str | Path,
    attestation_receipt_path: str | Path, execution_root: str | Path,
    authorization_hash: str, output_path: str | Path,
) -> dict[str, Any]:
    project = Path(project_root).resolve(strict=True)
    canonical_output = _canonical_unwritten_public_path(
        project=project,
        supplied=output_path,
        relative=A_FREEZE_RELATIVE,
        field="A_hold pre-run freeze",
    )
    canonical_root = _canonical_execution_root(
        project=project, supplied=execution_root, stage="A_hold"
    )
    _project, common, _actions, _capability, _receipt, _prepared = _build_freeze_common(
        stage="A_hold", project_root=project_root,
        acquisition_receipt_path=acquisition_receipt_path,
        selection_secret_path=selection_secret_path,
        capability_receipt_path=capability_receipt_path, runtime_python=runtime_python,
        local_llm_model=local_llm_model, local_embedding_model=local_embedding_model,
        base_binding_receipt_path=base_binding_receipt_path,
        attestation_receipt_path=attestation_receipt_path,
        execution_root=canonical_root,
    )
    body = {
        "schema": FREEZE_A_SCHEMA,
        "decision": "authorize_exact_fresh_2Wiki_A_hold_once",
        **common,
        "authorization_hash": _require_sha256(authorization_hash, "A_hold authorization"),
        "ordering": {
            "A_hold_rows_read_while_freezing": 0, "A_hold_labels_read_while_freezing": 0,
            "M_search_rows_read_while_freezing": 0,
            "fixed_actions_loaded_without_private_MuSiQue_caches": True,
            "freeze_complete_before_A_hold_open": True,
        },
        "raw_content_persisted": False,
    }
    freeze = {**body, "freeze_sha256": stable_hash(body)}
    _assert_public_safe(freeze)
    _write_json_exclusive(canonical_output, freeze, mode=0o644)
    return freeze


def _load_freeze(path: str | Path, *, schema: str, project: Path, stage: str) -> tuple[dict[str, Any], str]:
    relative = A_FREEZE_RELATIVE if stage == "A_hold" else M_FREEZE_RELATIVE
    canonical, custody_sha = _canonical_committed_public_path(
        project=project, supplied=path, relative=relative, field=f"{stage} freeze"
    )
    freeze, raw = _read_json(canonical, f"{stage} freeze")
    body = dict(freeze)
    declared = _require_sha256(body.pop("freeze_sha256", None), f"{stage} freeze")
    if (
        freeze.get("schema") != schema or stable_hash(body) != declared
        or _sha256_bytes(raw) != custody_sha
        or freeze.get("implementation") != current_implementation_binding(project)
        or freeze.get("execution_contract") != _execution_contract(stage)
        or freeze.get("raw_content_persisted") is not False
    ):
        raise TwoWikiZeroShotTransferError(f"{stage} freeze drifted")
    _assert_public_safe(freeze)
    return freeze, _sha256_bytes(raw)


def _verify_runtime_inputs(
    *, freeze: Mapping[str, Any], project: Path, capability_receipt_path: str | Path,
    runtime_python: str | Path, local_llm_model: str | Path,
    local_embedding_model: str | Path, base_binding_receipt_path: str | Path,
    attestation_receipt_path: str | Path,
) -> tuple[dict[str, Any], PreparedFormalRuntimeV2]:
    capability, binding, prepared = _runtime_bundle(
        project=project, capability_receipt_path=capability_receipt_path,
        runtime_python=runtime_python, local_llm_model=local_llm_model,
        local_embedding_model=local_embedding_model,
        base_binding_receipt_path=base_binding_receipt_path,
        attestation_receipt_path=attestation_receipt_path,
    )
    if freeze.get("capability_binding") != binding or freeze.get("runtime_binding") != prepared.safe_binding:
        raise TwoWikiZeroShotTransferError("formal runtime binding drifted")
    return capability, prepared


def _execute_components(
    *, root: Path, items: Sequence[StudyItem], component_ids: Sequence[str],
    programs: Mapping[str, TypedRetrievalProgram], prepared: PreparedFormalRuntimeV2,
    wave_count: int, progress: dict[str, Any] | None = None,
) -> tuple[dict[tuple[int, str], tuple[int, ...]], dict[str, Any]]:
    attempted = completed = official_attempted = official_completed = 0
    counters_lock = threading.Lock()
    official_slots = threading.BoundedSemaphore(OFFICIAL_CONCURRENCY_CAP)
    direct: dict[tuple[int, str], tuple[int, ...]] = {}
    barrier_parties: list[int] = []
    progress = {} if progress is None else progress
    progress.update({
        "retrieval_attempt_count": 0, "retrieval_terminal_count": 0,
        "official_attempt_count": 0, "official_terminal_count": 0,
        "observed_barrier_party_counts": barrier_parties,
    })
    for wave in range(wave_count):
        start = wave * WAVE_ITEM_COUNT
        wave_items = items[start : start + WAVE_ITEM_COUNT]
        if len(wave_items) != WAVE_ITEM_COUNT:
            raise TwoWikiZeroShotTransferError("wave item count drifted")
        barrier = threading.Barrier(WAVE_PARTY_COUNT)
        barrier_parties.append(barrier.parties)
        units = tuple(
            (start + local_ordinal, component, item.view)
            for local_ordinal, item in enumerate(wave_items)
            for component in component_ids
        )

        def run_one(unit: tuple[int, str, RetrievalItem]) -> tuple[tuple[int, str], tuple[int, ...]]:
            nonlocal attempted, completed, official_attempted, official_completed
            ordinal, component, item = unit
            with counters_lock:
                attempted += 1
                progress["retrieval_attempt_count"] = attempted
                if component == "official_HippoRAG_core_item_local":
                    official_attempted += 1
                    progress["official_attempt_count"] = official_attempted
            try:
                barrier.wait(timeout=180)
            except threading.BrokenBarrierError as exc:
                raise TwoWikiZeroShotTransferError("eager start barrier did not close") from exc
            if component == "canonical_RAW":
                ranking = tuple(row.idx for row in item.corpus[:TOP_K])
            elif component == "official_HippoRAG_core_item_local":
                with official_slots:
                    ranking = tuple(prepared.retrieve(
                        question=item.question,
                        paragraphs=item.hipporag_paragraphs(),
                        work_root=root / f"official_item_{ordinal:02d}",
                    ))
                if (
                    len(ranking) != TOP_K or len(set(ranking)) != TOP_K
                    or any(type(value) is not int or not 0 <= value < len(item.corpus) for value in ranking)
                ):
                    raise TwoWikiZeroShotTransferError("official ranking drifted")
                with counters_lock:
                    official_completed += 1
                    progress["official_terminal_count"] = official_completed
            else:
                ranking = _ranking(programs[component], item)
            with counters_lock:
                completed += 1
                progress["retrieval_terminal_count"] = completed
            return (ordinal, component), ranking

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=WAVE_PARTY_COUNT,
            thread_name_prefix=f"twowiki-zero-shot-wave-{wave}",
        ) as executor:
            terminals = _submit_eager_then_join(
                executor=executor, function=run_one, work_units=units,
            )
        if len(terminals) != WAVE_PARTY_COUNT:
            raise TwoWikiZeroShotTransferError("wave terminal closure incomplete")
        for key, ranking in terminals:
            if key in direct:
                raise TwoWikiZeroShotTransferError("duplicate terminal key")
            direct[key] = ranking
    expected = len(items) * len(component_ids)
    if attempted != expected or completed != expected or len(direct) != expected:
        raise TwoWikiZeroShotTransferError("terminal closure incomplete")
    if official_attempted != len(items) or official_completed != len(items):
        raise TwoWikiZeroShotTransferError("official terminal closure incomplete")
    result = {
        "retrieval_attempt_count": attempted, "retrieval_terminal_count": completed,
        "official_attempt_count": official_attempted,
        "official_terminal_count": official_completed,
        "observed_barrier_party_counts": barrier_parties,
        "all_work_units_eagerly_submitted_before_join": True,
    }
    progress.update(result)
    return direct, result


def _anchor_arms(items: Sequence[StudyItem], direct: Mapping[tuple[int, str], tuple[int, ...]]) -> dict[str, list[tuple[int, ...]]]:
    arms = {name: [] for name in (
        "canonical_RAW", "retained_P", "incumbent_portfolio",
        "challenger_portfolio", "official_HippoRAG_core_item_local",
    )}
    for ordinal in range(len(items)):
        if direct[(ordinal, "incumbent_P")] != direct[(ordinal, "challenger_P")]:
            raise TwoWikiZeroShotTransferError(
                "duplicated retained-P physical calls disagreed on A_hold"
            )
        arms["canonical_RAW"].append(direct[(ordinal, "canonical_RAW")])
        arms["retained_P"].append(direct[(ordinal, "incumbent_P")])
        arms["incumbent_portfolio"].append(_fuse(
            direct[(ordinal, "incumbent_P")], direct[(ordinal, "incumbent_Q1")],
            direct[(ordinal, "incumbent_Q2")],
        ))
        arms["challenger_portfolio"].append(_fuse(
            direct[(ordinal, "challenger_P")], direct[(ordinal, "challenger_Q1")],
            direct[(ordinal, "challenger_Q2")],
        ))
        arms["official_HippoRAG_core_item_local"].append(
            direct[(ordinal, "official_HippoRAG_core_item_local")]
        )
    return arms


def _search_arms(items: Sequence[StudyItem], direct: Mapping[tuple[int, str], tuple[int, ...]]) -> dict[str, list[tuple[int, ...]]]:
    arms = {name: [] for name in (
        "canonical_RAW", "retained_P", "incumbent_portfolio",
        "active_portfolio", "official_HippoRAG_core_item_local",
    )}
    for ordinal in range(len(items)):
        if direct[(ordinal, "incumbent_P")] != direct[(ordinal, "active_P")]:
            raise TwoWikiZeroShotTransferError(
                "duplicated retained-P physical calls disagreed on M_search"
            )
        arms["canonical_RAW"].append(direct[(ordinal, "canonical_RAW")])
        arms["retained_P"].append(direct[(ordinal, "active_P")])
        arms["incumbent_portfolio"].append(_fuse(
            direct[(ordinal, "incumbent_P")], direct[(ordinal, "incumbent_Q1")],
            direct[(ordinal, "incumbent_Q2")],
        ))
        arms["active_portfolio"].append(_fuse(
            direct[(ordinal, "active_P")], direct[(ordinal, "active_Q1")],
            direct[(ordinal, "active_Q2")],
        ))
        arms["official_HippoRAG_core_item_local"].append(
            direct[(ordinal, "official_HippoRAG_core_item_local")]
        )
    return arms


def _report_binding(report: Mapping[str, Any], raw: bytes) -> dict[str, Any]:
    return {
        "report_sha256": report["report_sha256"],
        "report_file_sha256": _sha256_bytes(raw),
        "freeze_sha256": report["freeze_sha256"],
    }


def _validate_metric_envelope(
    metrics: object, *, expected_arms: set[str], item_count: int
) -> None:
    if not isinstance(metrics, Mapping) or set(metrics) != expected_arms:
        raise TwoWikiZeroShotTransferError("report arm metric set drifted")
    totals: set[int] = set()
    for arm, value in metrics.items():
        if not isinstance(value, Mapping) or value.get("arm_id") != arm:
            raise TwoWikiZeroShotTransferError("report arm metric malformed")
        hits = value.get("support_hit_count")
        total = value.get("support_total")
        if (
            value.get("item_count") != item_count
            or type(hits) is not int or type(total) is not int
            or not 0 <= hits <= total or total <= 0
            or not isinstance(value.get("item_hit_vector_sha256"), str)
        ):
            raise TwoWikiZeroShotTransferError("report arm metric value drifted")
        totals.add(total)
    if len(totals) != 1:
        raise TwoWikiZeroShotTransferError("report support totals differ across arms")


def _validate_paired(value: object, *, left: str, right: str) -> bool:
    if not isinstance(value, Mapping):
        raise TwoWikiZeroShotTransferError("paired comparison malformed")
    test = value.get("paired_test")
    net = value.get("net_support_hit_count")
    if (
        value.get("left_arm_id") != left or value.get("right_arm_id") != right
        or type(net) is not int or not isinstance(test, Mapping)
        or test.get("test") != "one_sided_exact_magnitude_preserving_paired_sign_flip_v1"
        or test.get("observed_net_support_hits") != net
        or test.get("alpha_numerator") != 1 or test.get("alpha_denominator") != 10
        or test.get("sole_promotion_criterion") is not True
    ):
        raise TwoWikiZeroShotTransferError("paired comparison contract drifted")
    numerator = test.get("p_value_numerator")
    denominator = test.get("p_value_denominator")
    if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
        raise TwoWikiZeroShotTransferError("paired exact probability malformed")
    probability = Fraction(numerator, denominator)
    positive = net > 0
    exact = probability <= PROMOTION_ALPHA
    promoted = positive and exact
    if (
        test.get("positive_observed_net") is not positive
        or test.get("exact_p_at_or_below_alpha") is not exact
        or test.get("promoted") is not promoted
    ):
        raise TwoWikiZeroShotTransferError("paired decision was not mechanically derived")
    return promoted


def _validate_descriptive_paired(
    value: object, *, left: str, right: str
) -> None:
    if not isinstance(value, Mapping):
        raise TwoWikiZeroShotTransferError("descriptive paired comparison malformed")
    test = value.get("paired_test")
    net = value.get("net_support_hit_count")
    if (
        value.get("left_arm_id") != left
        or value.get("right_arm_id") != right
        or value.get("descriptive_only") is not True
        or value.get("affects_L5_or_epoch") is not False
        or type(net) is not int
        or not isinstance(test, Mapping)
        or test.get("test")
        != "one_sided_exact_magnitude_preserving_paired_sign_flip_v1"
        or test.get("observed_net_support_hits") != net
        or test.get("sole_promotion_criterion") is not False
        or test.get("descriptive_only") is not True
        or test.get("affects_L5_or_epoch") is not False
        or "promoted" in test
    ):
        raise TwoWikiZeroShotTransferError("descriptive comparison contract drifted")
    numerator = test.get("p_value_numerator")
    denominator = test.get("p_value_denominator")
    if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
        raise TwoWikiZeroShotTransferError("descriptive exact probability malformed")
    criterion = net > 0 and Fraction(numerator, denominator) <= PROMOTION_ALPHA
    if test.get("positive_and_p_at_or_below_alpha") is not criterion:
        raise TwoWikiZeroShotTransferError("descriptive criterion drifted")


def _recompute_private_execution_evidence(
    *,
    project: Path,
    stage: str,
    freeze: Mapping[str, Any],
    freeze_file_sha: str,
    report: Mapping[str, Any],
    public_report_raw: bytes,
) -> dict[str, Any]:
    """Rebuild every reported score from the fixed ignored execution root."""

    if stage == "A_hold":
        item_count = A_HOLD_ITEM_COUNT
        component_ids = A_COMPONENT_IDS
        evidence_name = "a_hold.private.evidence.json"
        report_name = "a_hold.aggregate.report.json"
        consumption_name = "a_hold.authorization.consumed.json"
        failure_name = "a_hold.failure.json"
    elif stage == "M_search":
        item_count = M_SEARCH_ITEM_COUNT
        component_ids = M_COMPONENT_IDS
        evidence_name = "m_search.private.evidence.json"
        report_name = "m_search.aggregate.report.json"
        consumption_name = "m_search.authorization.consumed.json"
        failure_name = "m_search.failure.json"
    else:
        raise TwoWikiZeroShotTransferError("unknown evidence stage")
    root = _canonical_execution_root(
        project=project,
        supplied=project
        / (
            A_EXECUTION_ROOT_RELATIVE
            if stage == "A_hold"
            else M_EXECUTION_ROOT_RELATIVE
        ),
        stage=stage,
    )
    if (
        root.is_symlink()
        or not root.is_dir()
        or freeze.get("execution_root_sha256") != _root_hash(root)
        or (root / failure_name).exists()
    ):
        raise TwoWikiZeroShotTransferError(f"{stage} private execution root drifted")
    _require_private_boundary(root)
    private_report_path = root / report_name
    private_report, private_report_raw = _read_json(
        private_report_path, f"{stage} private aggregate report"
    )
    if private_report != report or private_report_raw != public_report_raw:
        raise TwoWikiZeroShotTransferError(
            f"{stage} public report is not byte-identical to the execution report"
        )
    consumption, _consumption_raw = _read_json(
        root / consumption_name, f"{stage} authorization consumption"
    )
    consumption_body = dict(consumption)
    consumption_sha = _require_sha256(
        consumption_body.pop("consumption_sha256", None),
        f"{stage} consumption",
    )
    if (
        stable_hash(consumption_body) != consumption_sha
        or consumption.get("schema") != CONSUMPTION_SCHEMA
        or consumption.get("stage") != stage
        or consumption.get("authorization_hash") != freeze.get("authorization_hash")
        or consumption.get("freeze_sha256") != freeze.get("freeze_sha256")
        or consumption.get("freeze_file_sha256") != freeze_file_sha
        or consumption.get("execution_root_sha256")
        != freeze.get("execution_root_sha256")
        or consumption.get("fresh_bwrap_probe_completed_before_marker") is not True
        or consumption.get("replay_authorized") is not False
        or consumption.get("raw_content_persisted") is not False
    ):
        raise TwoWikiZeroShotTransferError(f"{stage} consumption evidence drifted")

    evidence_path = root / evidence_name
    evidence, evidence_raw = _read_json(evidence_path, f"{stage} private evidence")
    evidence_body = dict(evidence)
    evidence_sha = _require_sha256(
        evidence_body.pop("evidence_sha256", None), f"{stage} evidence"
    )
    expected_evidence_schema = f"{VERSION}_{stage}_private_evidence"
    binding = report.get("private_evidence_binding")
    rows = evidence.get("item_rows")
    if (
        set(evidence)
        != {
            "schema",
            "freeze_sha256",
            "item_rows",
            "raw_question_or_corpus_persisted",
            "evidence_sha256",
        }
        or evidence.get("schema") != expected_evidence_schema
        or evidence.get("freeze_sha256") != freeze.get("freeze_sha256")
        or evidence.get("raw_question_or_corpus_persisted") is not False
        or stable_hash(evidence_body) != evidence_sha
        or not isinstance(binding, Mapping)
        or binding
        != {
            "file_sha256": _sha256_bytes(evidence_raw),
            "evidence_sha256": evidence_sha,
            "private_path_persisted_publicly": False,
            "item_level_evidence_persisted_publicly": False,
        }
        or not isinstance(rows, list)
        or len(rows) != item_count
    ):
        raise TwoWikiZeroShotTransferError(f"{stage} private evidence binding drifted")

    direct: dict[tuple[int, str], tuple[int, ...]] = {}
    item_commitments: list[str] = []
    evidence_supports: list[tuple[int, ...]] = []
    for ordinal, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {
            "item_commitment_sha256",
            "support_indices",
            "component_rankings",
        }:
            raise TwoWikiZeroShotTransferError(f"{stage} private item evidence malformed")
        item_commitment = _require_sha256(
            row.get("item_commitment_sha256"), "private item commitment"
        )
        support = row.get("support_indices")
        rankings = row.get("component_rankings")
        if (
            not isinstance(support, list)
            or not support
            or any(type(value) is not int or not 0 <= value < 10 for value in support)
            or len(set(support)) != len(support)
            or not isinstance(rankings, Mapping)
            or set(rankings) != set(component_ids)
        ):
            raise TwoWikiZeroShotTransferError(f"{stage} support/ranking evidence malformed")
        for component in component_ids:
            ranking = rankings.get(component)
            if (
                not isinstance(ranking, list)
                or len(ranking) != TOP_K
                or len(set(ranking)) != TOP_K
                or any(type(value) is not int or not 0 <= value < 10 for value in ranking)
            ):
                raise TwoWikiZeroShotTransferError(
                    f"{stage} component ranking evidence malformed"
                )
            direct[(ordinal, component)] = tuple(ranking)
        if direct[(ordinal, "canonical_RAW")] != tuple(range(TOP_K)):
            raise TwoWikiZeroShotTransferError(f"{stage} canonical RAW evidence drifted")
        item_commitments.append(item_commitment)
        evidence_supports.append(tuple(support))
    source_binding = report.get("source_binding")
    if (
        len(set(item_commitments)) != item_count
        or not isinstance(source_binding, Mapping)
        or source_binding.get("measurement_item_commitment_set_sha256")
        != stable_hash(item_commitments)
    ):
        raise TwoWikiZeroShotTransferError(f"{stage} item commitment closure drifted")

    module = _acquisition_module()
    source_member = "train.json" if stage == "A_hold" else "dev.json"
    type_counts = source_binding.get("question_type_counts")
    if not isinstance(type_counts, Mapping):
        raise TwoWikiZeroShotTransferError(f"{stage} source type closure drifted")
    commitment = module.BlockCommitment(
        block=stage,
        source_member=source_member,
        question_type_counts=dict(type_counts),
        count=item_count,
        file_sha256=_require_sha256(
            source_binding.get("measurement_block_file_sha256"),
            f"{stage} block file",
        ),
        item_commitment_set_sha256=_require_sha256(
            source_binding.get("measurement_item_commitment_set_sha256"),
            f"{stage} item set",
        ),
    )
    canonical_items = _load_private_items(project, stage, commitment)
    actions = _load_fixed_actions(project)
    if freeze.get("fixed_action_binding") != actions.public_binding:
        raise TwoWikiZeroShotTransferError(f"{stage} fixed action binding drifted")
    if stage == "A_hold":
        programs = {
            "incumbent_P": actions.retained_p,
            "incumbent_Q1": actions.a_incumbent[0],
            "incumbent_Q2": actions.a_incumbent[1],
            "challenger_P": actions.retained_p,
            "challenger_Q1": actions.a_challenger[0],
            "challenger_Q2": actions.a_challenger[1],
        }
    else:
        programs = {
            "incumbent_P": actions.retained_p,
            "incumbent_Q1": actions.f_incumbent[0],
            "incumbent_Q2": actions.f_incumbent[1],
            "active_P": actions.retained_p,
            "active_Q1": actions.f_challenger[0],
            "active_Q2": actions.f_challenger[1],
        }
    for ordinal, item in enumerate(canonical_items):
        if (
            item.view.item_commitment_sha256 != item_commitments[ordinal]
            or item.support_indices != evidence_supports[ordinal]
        ):
            raise TwoWikiZeroShotTransferError(
                f"{stage} evidence does not match the canonical private block"
            )
        for component in component_ids:
            if component == "official_HippoRAG_core_item_local":
                continue
            expected_ranking = (
                tuple(range(TOP_K))
                if component == "canonical_RAW"
                else _ranking(programs[component], item.view)
            )
            if direct[(ordinal, component)] != expected_ranking:
                raise TwoWikiZeroShotTransferError(
                    f"{stage} deterministic ranking evidence drifted"
                )
    ranking_receipts = [
        {
            "ordinal_sha256": stable_hash({"ordinal": ordinal}),
            "component_id": component,
            "ranking_sha256": stable_hash({"retrieved_indices": list(ranking)}),
        }
        for (ordinal, component), ranking in sorted(direct.items())
    ]
    execution = report.get("execution")
    if (
        not isinstance(execution, Mapping)
        or execution.get("ranking_receipt_set_sha256")
        != stable_hash(ranking_receipts)
    ):
        raise TwoWikiZeroShotTransferError(f"{stage} ranking receipt closure drifted")

    frozen_items = canonical_items
    arms = (
        _anchor_arms(frozen_items, direct)
        if stage == "A_hold"
        else _search_arms(frozen_items, direct)
    )
    metrics = {
        name: _aggregate(name, frozen_items, rankings)
        for name, rankings in arms.items()
    }
    if stage == "A_hold":
        return {
            "arm_metrics": metrics,
            "primary": _paired(
                "challenger_portfolio", "incumbent_portfolio", frozen_items, arms
            ),
        }
    return {
        "arm_metrics": metrics,
        "primary": _paired(
            "active_portfolio", "incumbent_portfolio", frozen_items, arms
        ),
        "secondary_raw": _descriptive_paired(
            "active_portfolio", "canonical_RAW", frozen_items, arms
        ),
        "secondary_p": _descriptive_paired(
            "active_portfolio", "retained_P", frozen_items, arms
        ),
        "secondary_official": _descriptive_paired(
            "active_portfolio",
            "official_HippoRAG_core_item_local",
            frozen_items,
            arms,
        ),
    }


def reverify_a_hold_public_report(
    *, project_root: str | Path, pre_run_freeze_path: str | Path,
    report_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    project = Path(project_root).resolve(strict=True)
    freeze, freeze_file_sha = _load_freeze(
        pre_run_freeze_path, schema=FREEZE_A_SCHEMA, project=project, stage="A_hold"
    )
    canonical_report, report_custody_sha = _canonical_committed_public_path(
        project=project,
        supplied=report_path,
        relative=A_REPORT_RELATIVE,
        field="A_hold public report",
    )
    report, raw = _read_json(canonical_report, "A_hold public report")
    body = dict(report)
    declared = _require_sha256(body.pop("report_sha256", None), "A_hold report")
    paired = report.get("challenger_minus_incumbent")
    transition = report.get("evaluator_epoch_transition")
    if (
        report.get("schema") != REPORT_A_SCHEMA or report.get("valid") is not True
        or _sha256_bytes(raw) != report_custody_sha
        or stable_hash(body) != declared or report.get("freeze_sha256") != freeze["freeze_sha256"]
        or report.get("freeze_file_sha256") != freeze_file_sha
        or report.get("source_binding") != freeze["source_binding"]
        or report.get("design_binding") != freeze["design_binding"]
        or report.get("fixed_action_binding") != freeze["fixed_action_binding"]
        or report.get("execution", {}).get("physical_work_unit_count") != 384
        or not isinstance(paired, Mapping) or not isinstance(transition, Mapping)
        or transition.get("promoted") != paired.get("paired_test", {}).get("promoted")
        or report.get("M_search_authorized") != transition.get("promoted")
        or report.get("raw_content_persisted") is not False
    ):
        raise TwoWikiZeroShotTransferError("A_hold public report drifted")
    _validate_metric_envelope(
        report.get("arm_metrics"),
        expected_arms={
            "canonical_RAW", "retained_P", "incumbent_portfolio",
            "challenger_portfolio", "official_HippoRAG_core_item_local",
        },
        item_count=A_HOLD_ITEM_COUNT,
    )
    promoted = _validate_paired(
        paired, left="challenger_portfolio", right="incumbent_portfolio"
    )
    recomputed = _recompute_private_execution_evidence(
        project=project,
        stage="A_hold",
        freeze=freeze,
        freeze_file_sha=freeze_file_sha,
        report=report,
        public_report_raw=raw,
    )
    if (
        report.get("arm_metrics") != recomputed["arm_metrics"]
        or paired != recomputed["primary"]
    ):
        raise TwoWikiZeroShotTransferError(
            "A_hold public scores do not match private execution evidence"
        )
    transition_body = dict(transition)
    transition_sha = _require_sha256(
        transition_body.pop("transition_sha256", None), "A_hold transition"
    )
    fixed = freeze["fixed_action_binding"]
    expected_transition = {
        "promoted": promoted,
        "incumbent_action_sha256": fixed["A_incumbent"]["action_sha256"],
        "active_action_sha256": fixed[
            "A_challenger" if promoted else "A_incumbent"
        ]["action_sha256"],
        "selective_evaluator_dependent_score_invalidation": promoted,
        "independent_source_records_retained": True,
        "M_search_may_open": promoted,
    }
    execution = report.get("execution")
    runtime = report.get("runtime")
    controls = report.get("control_validity")
    if (
        transition_body != expected_transition
        or transition_sha != stable_hash(expected_transition)
        or not isinstance(execution, Mapping)
        or execution.get("item_count") != 48
        or execution.get("physical_work_unit_count") != 384
        or execution.get("physical_component_ids") != list(A_COMPONENT_IDS)
        or execution.get("item_wave_count") != 2
        or execution.get("configured_maximum_concurrency") != 192
        or execution.get("official_worker_concurrency_cap") != 24
        or execution.get("retrieval_attempt_count") != 384
        or execution.get("retrieval_terminal_count") != 384
        or execution.get("official_attempt_count") != 48
        or execution.get("official_terminal_count") != 48
        or execution.get("observed_barrier_party_counts") != [192, 192]
        or execution.get("all_work_units_eagerly_submitted_before_join") is not True
        or execution.get("both_waves_joined_before_postflight_and_support_scoring") is not True
        or execution.get("external_network_calls") != 0
        or execution.get("online_evaluator_calls") != 0
        or any(execution.get(field) != 0 for field in ("retries", "replays", "resamples"))
        or not isinstance(runtime, Mapping)
        or runtime.get("fresh_bwrap_probe_before_authorization_consumption") is not True
        or runtime.get("postflight_fresh_filesystem_attestation") is not True
        or runtime.get("capability_receipt_sha256")
        != freeze.get("capability_binding", {}).get("receipt_sha256")
        or runtime.get("postflight_binding_sha256")
        != freeze.get("runtime_binding", {}).get("binding_sha256")
        or controls != {
            "RAW_retained_P_and_official_are_execution_gating": True,
            "control_scores_are_statistically_non_gating": True,
            "duplicated_retained_P_rankings_strictly_equal": True,
            "retained_P_control_source_component": "incumbent_P",
        }
        or report.get("M_search_opened") is not False
        or report.get("M_search_authorized") is not promoted
    ):
        raise TwoWikiZeroShotTransferError("A_hold terminal or transition evidence drifted")
    _assert_public_safe(report)
    return report, _report_binding(report, raw)


def execute_a_hold_formal(
    *, project_root: str | Path, pre_run_freeze_path: str | Path,
    acquisition_receipt_path: str | Path, selection_secret_path: str | Path,
    capability_receipt_path: str | Path, runtime_python: str | Path,
    local_llm_model: str | Path, local_embedding_model: str | Path,
    base_binding_receipt_path: str | Path, attestation_receipt_path: str | Path,
    execution_root: str | Path,
) -> dict[str, Any]:
    if _CLEAN_MODULE_CLI_ACTIVE is not True:
        raise TwoWikiZeroShotTransferError("formal A_hold is available only through clean CLI")
    project = Path(project_root).resolve(strict=True)
    freeze, freeze_file_sha = _load_freeze(
        pre_run_freeze_path, schema=FREEZE_A_SCHEMA, project=project, stage="A_hold"
    )
    root = _canonical_execution_root(
        project=project, supplied=execution_root, stage="A_hold"
    )
    if freeze.get("execution_root_sha256") != _root_hash(root) or root.exists():
        raise TwoWikiZeroShotTransferError("fresh A_hold root drifted; replay forbidden")
    design, design_binding = _load_design(project)
    actions = _load_fixed_actions(project)
    receipt, raw, commitments = _load_acquisition(
        project=project, receipt_path=acquisition_receipt_path,
        selection_secret_path=selection_secret_path,
    )
    expected = {
        "design_binding": design_binding,
        "source_binding": _source_binding(receipt, raw, commitments["A_hold"]),
        "fixed_action_binding": actions.public_binding,
    }
    if any(freeze.get(key) != value for key, value in expected.items()):
        raise TwoWikiZeroShotTransferError("A_hold frozen source or action drifted")
    capability, prepared = _verify_runtime_inputs(
        freeze=freeze, project=project, capability_receipt_path=capability_receipt_path,
        runtime_python=runtime_python, local_llm_model=local_llm_model,
        local_embedding_model=local_embedding_model,
        base_binding_receipt_path=base_binding_receipt_path,
        attestation_receipt_path=attestation_receipt_path,
    )
    del design
    # The elevated sandbox proof is deliberately before root creation and marker consumption.
    _fresh_probe(capability)
    _require_private_boundary(root)
    _prepare_output(root)
    os.mkdir(root, 0o700)
    stage = "authorization_consumption"
    execution_counts: dict[str, Any] = {
        "retrieval_attempt_count": 0, "retrieval_terminal_count": 0,
        "official_attempt_count": 0, "official_terminal_count": 0,
    }
    try:
        consumption_body = {
            "schema": CONSUMPTION_SCHEMA, "stage": "A_hold",
            "authorization_hash": freeze["authorization_hash"],
            "freeze_sha256": freeze["freeze_sha256"],
            "freeze_file_sha256": freeze_file_sha,
            "execution_root_sha256": freeze["execution_root_sha256"],
            "fresh_bwrap_probe_completed_before_marker": True,
            "replay_authorized": False, "raw_content_persisted": False,
        }
        _write_json_exclusive(root / "a_hold.authorization.consumed.json", {
            **consumption_body, "consumption_sha256": stable_hash(consumption_body),
        })
        stage = "exact_A_hold_open_after_consumption"
        items = _load_private_items(project, "A_hold", commitments["A_hold"])
        programs = {
            "incumbent_P": actions.retained_p,
            "incumbent_Q1": actions.a_incumbent[0], "incumbent_Q2": actions.a_incumbent[1],
            "challenger_P": actions.retained_p,
            "challenger_Q1": actions.a_challenger[0], "challenger_Q2": actions.a_challenger[1],
        }
        stage = "two_eager_192_party_waves"
        direct, execution_counts = _execute_components(
            root=root, items=items, component_ids=A_COMPONENT_IDS,
            programs=programs, prepared=prepared, wave_count=2,
            progress=execution_counts,
        )
        stage = "fresh_runtime_postflight_before_support_scoring"
        postflight = prepared.fresh_reverify()
        if postflight != prepared.safe_binding:
            raise TwoWikiZeroShotTransferError("A_hold runtime postflight drifted")
        stage = "offline_support_scoring_after_all_terminals_and_postflight"
        arms = _anchor_arms(items, direct)
        metrics = {name: _aggregate(name, items, rankings) for name, rankings in arms.items()}
        paired = _paired("challenger_portfolio", "incumbent_portfolio", items, arms)
        promoted = paired["paired_test"]["promoted"]
        transition_body = {
            "promoted": promoted,
            "incumbent_action_sha256": actions.public_binding["A_incumbent"]["action_sha256"],
            "active_action_sha256": actions.public_binding[
                "A_challenger" if promoted else "A_incumbent"
            ]["action_sha256"],
            "selective_evaluator_dependent_score_invalidation": promoted,
            "independent_source_records_retained": True,
            "M_search_may_open": promoted,
        }
        transition = {**transition_body, "transition_sha256": stable_hash(transition_body)}
        ranking_receipts = [
            {"ordinal_sha256": stable_hash({"ordinal": ordinal}), "component_id": component,
             "ranking_sha256": stable_hash({"retrieved_indices": list(ranking)})}
            for (ordinal, component), ranking in sorted(direct.items())
        ]
        private_body = {
            "schema": f"{VERSION}_A_hold_private_evidence",
            "freeze_sha256": freeze["freeze_sha256"],
            "item_rows": [
                {"item_commitment_sha256": item.view.item_commitment_sha256,
                 "support_indices": list(item.support_indices),
                 "component_rankings": {
                     component: list(direct[(ordinal, component)]) for component in A_COMPONENT_IDS
                 }}
                for ordinal, item in enumerate(items)
            ],
            "raw_question_or_corpus_persisted": False,
        }
        private = {**private_body, "evidence_sha256": stable_hash(private_body)}
        private_path = root / "a_hold.private.evidence.json"
        _write_json_exclusive(private_path, private)
        report_body = {
            "schema": REPORT_A_SCHEMA, "valid": True,
            "freeze_sha256": freeze["freeze_sha256"], "freeze_file_sha256": freeze_file_sha,
            "source_binding": freeze["source_binding"],
            "design_binding": freeze["design_binding"],
            "fixed_action_binding": freeze["fixed_action_binding"],
            "arm_metrics": metrics,
            "challenger_minus_incumbent": paired,
            "control_validity": {
                "RAW_retained_P_and_official_are_execution_gating": True,
                "control_scores_are_statistically_non_gating": True,
                "duplicated_retained_P_rankings_strictly_equal": True,
                "retained_P_control_source_component": "incumbent_P",
            },
            "evaluator_epoch_transition": transition,
            "private_evidence_binding": {
                "file_sha256": _sha256_file(private_path),
                "evidence_sha256": private["evidence_sha256"],
                "private_path_persisted_publicly": False,
                "item_level_evidence_persisted_publicly": False,
            },
            "execution": {
                "physical_component_ids": list(A_COMPONENT_IDS), "item_count": len(items),
                "physical_work_unit_count": 384, "item_wave_count": 2,
                "configured_maximum_concurrency": 192,
                "official_worker_concurrency_cap": 24, **execution_counts,
                "both_waves_joined_before_postflight_and_support_scoring": True,
                "ranking_receipt_set_sha256": stable_hash(ranking_receipts),
                "external_network_calls": 0, "online_evaluator_calls": 0,
                "retries": 0, "replays": 0, "resamples": 0,
            },
            "runtime": {
                "capability_receipt_sha256": capability["receipt_sha256"],
                "fresh_bwrap_probe_before_authorization_consumption": True,
                "postflight_fresh_filesystem_attestation": True,
                "postflight_binding_sha256": postflight["binding_sha256"],
            },
            "M_search_opened": False, "M_search_authorized": promoted,
            "raw_content_persisted": False,
        }
        report = {**report_body, "report_sha256": stable_hash(report_body)}
        _assert_public_safe(report)
        _write_json_exclusive(root / "a_hold.aggregate.report.json", report)
        return report
    except Exception as exc:
        failure_body = {
            "schema": FAILURE_SCHEMA, "stage": "A_hold", "valid": False,
            "freeze_sha256": freeze["freeze_sha256"], "failure_stage": stage,
            "error_type_sha256": stable_hash({"error_type": type(exc).__name__}),
            "authorization_consumed": (root / "a_hold.authorization.consumed.json").is_file(),
            "physical_work_unit_count": 384, **execution_counts,
            "retries": 0, "replays": 0, "resamples": 0,
            "replay_authorized": False, "M_search_opened": False,
            "raw_content_persisted": False,
        }
        try:
            _write_json_exclusive(root / "a_hold.failure.json", {
                **failure_body, "failure_sha256": stable_hash(failure_body),
            })
        except Exception:
            pass
        raise TwoWikiZeroShotTransferError("formal A_hold failed and cannot be replayed") from exc


def build_m_search_pre_run_freeze(
    *, project_root: str | Path, acquisition_receipt_path: str | Path,
    selection_secret_path: str | Path, a_hold_pre_run_freeze_path: str | Path,
    a_hold_report_path: str | Path, capability_receipt_path: str | Path,
    runtime_python: str | Path, local_llm_model: str | Path,
    local_embedding_model: str | Path, base_binding_receipt_path: str | Path,
    attestation_receipt_path: str | Path, execution_root: str | Path,
    authorization_hash: str, output_path: str | Path,
) -> dict[str, Any]:
    project = Path(project_root).resolve(strict=True)
    anchor, anchor_binding = reverify_a_hold_public_report(
        project_root=project, pre_run_freeze_path=a_hold_pre_run_freeze_path,
        report_path=a_hold_report_path,
    )
    if anchor["evaluator_epoch_transition"].get("promoted") is not True:
        raise TwoWikiZeroShotTransferError("A_hold did not promote; M_search must remain unopened")
    canonical_output = _canonical_unwritten_public_path(
        project=project,
        supplied=output_path,
        relative=M_FREEZE_RELATIVE,
        field="M_search pre-run freeze",
    )
    canonical_root = _canonical_execution_root(
        project=project, supplied=execution_root, stage="M_search"
    )
    _project, common, actions, _capability, _receipt, _prepared = _build_freeze_common(
        stage="M_search", project_root=project,
        acquisition_receipt_path=acquisition_receipt_path,
        selection_secret_path=selection_secret_path,
        capability_receipt_path=capability_receipt_path, runtime_python=runtime_python,
        local_llm_model=local_llm_model, local_embedding_model=local_embedding_model,
        base_binding_receipt_path=base_binding_receipt_path,
        attestation_receipt_path=attestation_receipt_path,
        execution_root=canonical_root,
    )
    body = {
        "schema": FREEZE_M_SCHEMA,
        "decision": "authorize_exact_promoted_2Wiki_M_search_once",
        **common,
        "A_hold_binding": anchor_binding,
        "evaluator_epoch_transition": anchor["evaluator_epoch_transition"],
        "search_action_binding": {
            "incumbent": actions.public_binding["F_incumbent"],
            "active": actions.public_binding["F_challenger_if_promoted_active"],
            "conditional_role_mapping": {
                "promoted_A_challenger_maps_to": "F_challenger_if_promoted_active",
                "A_incumbent_maps_to": "F_incumbent",
            },
            "frozen_before_A_hold_open": True,
        },
        "authorization_hash": _require_sha256(authorization_hash, "M_search authorization"),
        "ordering": {
            "M_search_rows_read_while_freezing": 0, "M_search_labels_read_while_freezing": 0,
            "promoted_A_hold_reverified_before_freeze": True,
            "freeze_complete_before_M_search_open": True,
        },
        "raw_content_persisted": False,
    }
    freeze = {**body, "freeze_sha256": stable_hash(body)}
    _assert_public_safe(freeze)
    _write_json_exclusive(canonical_output, freeze, mode=0o644)
    return freeze


def execute_m_search_formal(
    *, project_root: str | Path, pre_run_freeze_path: str | Path,
    acquisition_receipt_path: str | Path, selection_secret_path: str | Path,
    a_hold_pre_run_freeze_path: str | Path, a_hold_report_path: str | Path,
    capability_receipt_path: str | Path, runtime_python: str | Path,
    local_llm_model: str | Path, local_embedding_model: str | Path,
    base_binding_receipt_path: str | Path, attestation_receipt_path: str | Path,
    execution_root: str | Path,
) -> dict[str, Any]:
    if _CLEAN_MODULE_CLI_ACTIVE is not True:
        raise TwoWikiZeroShotTransferError("formal M_search is available only through clean CLI")
    project = Path(project_root).resolve(strict=True)
    freeze, freeze_file_sha = _load_freeze(
        pre_run_freeze_path, schema=FREEZE_M_SCHEMA, project=project, stage="M_search"
    )
    root = _canonical_execution_root(
        project=project, supplied=execution_root, stage="M_search"
    )
    if freeze.get("execution_root_sha256") != _root_hash(root) or root.exists():
        raise TwoWikiZeroShotTransferError("fresh M_search root drifted; replay forbidden")
    anchor, anchor_binding = reverify_a_hold_public_report(
        project_root=project, pre_run_freeze_path=a_hold_pre_run_freeze_path,
        report_path=a_hold_report_path,
    )
    if (
        anchor["evaluator_epoch_transition"].get("promoted") is not True
        or freeze.get("A_hold_binding") != anchor_binding
        or freeze.get("evaluator_epoch_transition") != anchor["evaluator_epoch_transition"]
    ):
        raise TwoWikiZeroShotTransferError("promoted A_hold binding drifted")
    _design, design_binding = _load_design(project)
    actions = _load_fixed_actions(project)
    receipt, raw, commitments = _load_acquisition(
        project=project, receipt_path=acquisition_receipt_path,
        selection_secret_path=selection_secret_path,
    )
    if (
        freeze.get("design_binding") != design_binding
        or freeze.get("source_binding") != _source_binding(receipt, raw, commitments["M_search"])
        or freeze.get("fixed_action_binding") != actions.public_binding
        or freeze.get("search_action_binding") != {
            "incumbent": actions.public_binding["F_incumbent"],
            "active": actions.public_binding["F_challenger_if_promoted_active"],
            "conditional_role_mapping": {
                "promoted_A_challenger_maps_to": "F_challenger_if_promoted_active",
                "A_incumbent_maps_to": "F_incumbent",
            },
            "frozen_before_A_hold_open": True,
        }
    ):
        raise TwoWikiZeroShotTransferError("M_search frozen source or action drifted")
    capability, prepared = _verify_runtime_inputs(
        freeze=freeze, project=project, capability_receipt_path=capability_receipt_path,
        runtime_python=runtime_python, local_llm_model=local_llm_model,
        local_embedding_model=local_embedding_model,
        base_binding_receipt_path=base_binding_receipt_path,
        attestation_receipt_path=attestation_receipt_path,
    )
    _fresh_probe(capability)
    _require_private_boundary(root)
    _prepare_output(root)
    os.mkdir(root, 0o700)
    stage = "authorization_consumption"
    execution_counts: dict[str, Any] = {
        "retrieval_attempt_count": 0, "retrieval_terminal_count": 0,
        "official_attempt_count": 0, "official_terminal_count": 0,
    }
    try:
        consumption_body = {
            "schema": CONSUMPTION_SCHEMA, "stage": "M_search",
            "authorization_hash": freeze["authorization_hash"],
            "freeze_sha256": freeze["freeze_sha256"],
            "freeze_file_sha256": freeze_file_sha,
            "execution_root_sha256": freeze["execution_root_sha256"],
            "fresh_bwrap_probe_completed_before_marker": True,
            "replay_authorized": False, "raw_content_persisted": False,
        }
        _write_json_exclusive(root / "m_search.authorization.consumed.json", {
            **consumption_body, "consumption_sha256": stable_hash(consumption_body),
        })
        stage = "exact_M_search_open_after_consumption"
        items = _load_private_items(project, "M_search", commitments["M_search"])
        programs = {
            "incumbent_P": actions.retained_p,
            "incumbent_Q1": actions.f_incumbent[0], "incumbent_Q2": actions.f_incumbent[1],
            "active_P": actions.retained_p,
            "active_Q1": actions.f_challenger[0], "active_Q2": actions.f_challenger[1],
        }
        stage = "one_eager_192_party_wave"
        direct, execution_counts = _execute_components(
            root=root, items=items, component_ids=M_COMPONENT_IDS,
            programs=programs, prepared=prepared, wave_count=1,
            progress=execution_counts,
        )
        stage = "fresh_runtime_postflight_before_support_scoring"
        postflight = prepared.fresh_reverify()
        if postflight != prepared.safe_binding:
            raise TwoWikiZeroShotTransferError("M_search runtime postflight drifted")
        stage = "offline_support_scoring_after_all_terminals_and_postflight"
        arms = _search_arms(items, direct)
        metrics = {name: _aggregate(name, items, rankings) for name, rankings in arms.items()}
        primary = _paired("active_portfolio", "incumbent_portfolio", items, arms)
        secondary_raw = _descriptive_paired(
            "active_portfolio", "canonical_RAW", items, arms
        )
        secondary_p = _descriptive_paired(
            "active_portfolio", "retained_P", items, arms
        )
        secondary_official = _descriptive_paired(
            "active_portfolio", "official_HippoRAG_core_item_local", items, arms
        )
        l5 = primary["net_support_hit_count"] > 0 and primary["paired_test"]["exact_p_at_or_below_alpha"]
        ranking_receipts = [
            {"ordinal_sha256": stable_hash({"ordinal": ordinal}), "component_id": component,
             "ranking_sha256": stable_hash({"retrieved_indices": list(ranking)})}
            for (ordinal, component), ranking in sorted(direct.items())
        ]
        private_body = {
            "schema": f"{VERSION}_M_search_private_evidence",
            "freeze_sha256": freeze["freeze_sha256"],
            "item_rows": [
                {"item_commitment_sha256": item.view.item_commitment_sha256,
                 "support_indices": list(item.support_indices),
                 "component_rankings": {
                     component: list(direct[(ordinal, component)]) for component in M_COMPONENT_IDS
                 }}
                for ordinal, item in enumerate(items)
            ],
            "raw_question_or_corpus_persisted": False,
        }
        private = {**private_body, "evidence_sha256": stable_hash(private_body)}
        private_path = root / "m_search.private.evidence.json"
        _write_json_exclusive(private_path, private)
        report_body = {
            "schema": REPORT_M_SCHEMA, "valid": True,
            "freeze_sha256": freeze["freeze_sha256"], "freeze_file_sha256": freeze_file_sha,
            "source_binding": freeze["source_binding"], "design_binding": freeze["design_binding"],
            "fixed_action_binding": freeze["fixed_action_binding"],
            "A_hold_binding": freeze["A_hold_binding"],
            "evaluator_epoch_transition": freeze["evaluator_epoch_transition"],
            "search_action_binding": freeze["search_action_binding"],
            "arm_metrics": metrics,
            "primary_active_minus_incumbent": primary,
            "secondary_active_minus_canonical_RAW": secondary_raw,
            "secondary_active_minus_retained_P": secondary_p,
            "secondary_active_minus_official_HippoRAG_core_item_local": secondary_official,
            "control_validity": {
                "RAW_retained_P_and_official_are_execution_gating": True,
                "control_scores_are_statistically_non_gating": True,
                "duplicated_retained_P_rankings_strictly_equal": True,
                "retained_P_control_source_component": "active_P",
            },
            "L5_disposition": {
                "positive_net": primary["net_support_hit_count"] > 0,
                "exact_sign_flip_p_le_0_10": primary["paired_test"]["exact_p_at_or_below_alpha"],
                "L5_achieved": l5, "M_search_used_for_epoch_transition": False,
                "followup_same_source_attempt_authorized": False,
            },
            "private_evidence_binding": {
                "file_sha256": _sha256_file(private_path),
                "evidence_sha256": private["evidence_sha256"],
                "private_path_persisted_publicly": False,
                "item_level_evidence_persisted_publicly": False,
            },
            "execution": {
                "physical_component_ids": list(M_COMPONENT_IDS), "item_count": len(items),
                "physical_work_unit_count": 192, "item_wave_count": 1,
                "configured_maximum_concurrency": 192,
                "official_worker_concurrency_cap": 24, **execution_counts,
                "all_terminals_joined_before_postflight_and_support_scoring": True,
                "ranking_receipt_set_sha256": stable_hash(ranking_receipts),
                "external_network_calls": 0, "online_evaluator_calls": 0,
                "retries": 0, "replays": 0, "resamples": 0,
            },
            "runtime": {
                "capability_receipt_sha256": capability["receipt_sha256"],
                "fresh_bwrap_probe_before_authorization_consumption": True,
                "postflight_fresh_filesystem_attestation": True,
                "postflight_binding_sha256": postflight["binding_sha256"],
            },
            "evaluator_epoch_unchanged_by_M_search": True,
            "raw_content_persisted": False,
        }
        report = {**report_body, "report_sha256": stable_hash(report_body)}
        _assert_public_safe(report)
        _write_json_exclusive(root / "m_search.aggregate.report.json", report)
        return report
    except Exception as exc:
        failure_body = {
            "schema": FAILURE_SCHEMA, "stage": "M_search", "valid": False,
            "freeze_sha256": freeze["freeze_sha256"], "failure_stage": stage,
            "error_type_sha256": stable_hash({"error_type": type(exc).__name__}),
            "authorization_consumed": (root / "m_search.authorization.consumed.json").is_file(),
            "physical_work_unit_count": 192, **execution_counts,
            "retries": 0, "replays": 0, "resamples": 0,
            "replay_authorized": False, "raw_content_persisted": False,
            "evaluator_epoch_unchanged_by_M_search": True,
            "followup_same_source_attempt_authorized": False,
        }
        try:
            _write_json_exclusive(root / "m_search.failure.json", {
                **failure_body, "failure_sha256": stable_hash(failure_body),
            })
        except Exception:
            pass
        raise TwoWikiZeroShotTransferError("formal M_search failed and cannot be replayed") from exc


def reverify_m_search_public_report(
    *, project_root: str | Path, pre_run_freeze_path: str | Path,
    report_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Independently validate a mechanically copied public M_search report."""

    project = Path(project_root).resolve(strict=True)
    freeze, freeze_file_sha = _load_freeze(
        pre_run_freeze_path, schema=FREEZE_M_SCHEMA, project=project, stage="M_search"
    )
    canonical_report, report_custody_sha = _canonical_committed_public_path(
        project=project,
        supplied=report_path,
        relative=M_REPORT_RELATIVE,
        field="M_search public report",
    )
    report, raw = _read_json(canonical_report, "M_search public report")
    body = dict(report)
    declared = _require_sha256(body.pop("report_sha256", None), "M_search report")
    if (
        report.get("schema") != REPORT_M_SCHEMA or report.get("valid") is not True
        or _sha256_bytes(raw) != report_custody_sha
        or stable_hash(body) != declared
        or report.get("freeze_sha256") != freeze["freeze_sha256"]
        or report.get("freeze_file_sha256") != freeze_file_sha
        or report.get("source_binding") != freeze["source_binding"]
        or report.get("design_binding") != freeze["design_binding"]
        or report.get("fixed_action_binding") != freeze["fixed_action_binding"]
        or report.get("A_hold_binding") != freeze["A_hold_binding"]
        or report.get("evaluator_epoch_transition") != freeze["evaluator_epoch_transition"]
        or report.get("search_action_binding") != freeze["search_action_binding"]
        or report.get("raw_content_persisted") is not False
    ):
        raise TwoWikiZeroShotTransferError("M_search public report drifted")
    _validate_metric_envelope(
        report.get("arm_metrics"),
        expected_arms={
            "canonical_RAW", "retained_P", "incumbent_portfolio",
            "active_portfolio", "official_HippoRAG_core_item_local",
        },
        item_count=M_SEARCH_ITEM_COUNT,
    )
    primary = report.get("primary_active_minus_incumbent")
    _validate_paired(primary, left="active_portfolio", right="incumbent_portfolio")
    _validate_descriptive_paired(
        report.get("secondary_active_minus_canonical_RAW"),
        left="active_portfolio", right="canonical_RAW",
    )
    _validate_descriptive_paired(
        report.get("secondary_active_minus_retained_P"),
        left="active_portfolio", right="retained_P",
    )
    _validate_descriptive_paired(
        report.get("secondary_active_minus_official_HippoRAG_core_item_local"),
        left="active_portfolio", right="official_HippoRAG_core_item_local",
    )
    recomputed = _recompute_private_execution_evidence(
        project=project,
        stage="M_search",
        freeze=freeze,
        freeze_file_sha=freeze_file_sha,
        report=report,
        public_report_raw=raw,
    )
    if (
        report.get("arm_metrics") != recomputed["arm_metrics"]
        or primary != recomputed["primary"]
        or report.get("secondary_active_minus_canonical_RAW")
        != recomputed["secondary_raw"]
        or report.get("secondary_active_minus_retained_P")
        != recomputed["secondary_p"]
        or report.get("secondary_active_minus_official_HippoRAG_core_item_local")
        != recomputed["secondary_official"]
    ):
        raise TwoWikiZeroShotTransferError(
            "M_search public scores do not match private execution evidence"
        )
    disposition = report.get("L5_disposition")
    assert isinstance(primary, Mapping)
    positive = primary.get("net_support_hit_count", 0) > 0
    exact = primary.get("paired_test", {}).get("exact_p_at_or_below_alpha") is True
    execution = report.get("execution")
    runtime = report.get("runtime")
    controls = report.get("control_validity")
    if (
        not isinstance(disposition, Mapping)
        or disposition.get("positive_net") is not positive
        or disposition.get("exact_sign_flip_p_le_0_10") is not exact
        or disposition.get("L5_achieved") is not (positive and exact)
        or disposition.get("M_search_used_for_epoch_transition") is not False
        or disposition.get("followup_same_source_attempt_authorized") is not False
        or not isinstance(execution, Mapping)
        or execution.get("item_count") != 24
        or execution.get("physical_work_unit_count") != 192
        or execution.get("physical_component_ids") != list(M_COMPONENT_IDS)
        or execution.get("item_wave_count") != 1
        or execution.get("configured_maximum_concurrency") != 192
        or execution.get("official_worker_concurrency_cap") != 24
        or execution.get("retrieval_attempt_count") != 192
        or execution.get("retrieval_terminal_count") != 192
        or execution.get("official_attempt_count") != 24
        or execution.get("official_terminal_count") != 24
        or execution.get("observed_barrier_party_counts") != [192]
        or execution.get("all_work_units_eagerly_submitted_before_join") is not True
        or execution.get("all_terminals_joined_before_postflight_and_support_scoring") is not True
        or execution.get("external_network_calls") != 0
        or execution.get("online_evaluator_calls") != 0
        or any(execution.get(field) != 0 for field in ("retries", "replays", "resamples"))
        or not isinstance(runtime, Mapping)
        or runtime.get("fresh_bwrap_probe_before_authorization_consumption") is not True
        or runtime.get("postflight_fresh_filesystem_attestation") is not True
        or runtime.get("capability_receipt_sha256")
        != freeze.get("capability_binding", {}).get("receipt_sha256")
        or runtime.get("postflight_binding_sha256")
        != freeze.get("runtime_binding", {}).get("binding_sha256")
        or controls != {
            "RAW_retained_P_and_official_are_execution_gating": True,
            "control_scores_are_statistically_non_gating": True,
            "duplicated_retained_P_rankings_strictly_equal": True,
            "retained_P_control_source_component": "active_P",
        }
        or report.get("evaluator_epoch_unchanged_by_M_search") is not True
    ):
        raise TwoWikiZeroShotTransferError("M_search terminal or L5 evidence drifted")
    _assert_public_safe(report)
    return report, _report_binding(report, raw)


def formal_signatures_have_no_injection_surface() -> bool:
    forbidden = {
        "program", "programs", "retriever", "retrievers", "callable", "result",
        "results", "evidence", "items", "rankings", "support_indices",
        "a_hold_block_path", "m_search_block_path", "a_form_private_cache_path",
        "f_search_private_cache_path",
    }
    functions = (
        build_a_hold_pre_run_freeze, execute_a_hold_formal,
        build_m_search_pre_run_freeze, execute_m_search_formal,
    )
    return not forbidden.intersection(
        set().union(*(set(inspect.signature(function).parameters) for function in functions))
    )


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--acquisition-receipt", type=Path, required=True)
    parser.add_argument("--selection-secret", type=Path, required=True)
    parser.add_argument("--capability-receipt", type=Path, required=True)
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--local-llm-model", type=Path, required=True)
    parser.add_argument("--local-embedding-model", type=Path, required=True)
    parser.add_argument("--base-binding-receipt", type=Path, required=True)
    parser.add_argument("--attestation-receipt", type=Path, required=True)


def _common_args(arguments: argparse.Namespace) -> dict[str, Any]:
    return {
        "project_root": arguments.project_root,
        "acquisition_receipt_path": arguments.acquisition_receipt,
        "selection_secret_path": arguments.selection_secret,
        "capability_receipt_path": arguments.capability_receipt,
        "runtime_python": arguments.runtime_python,
        "local_llm_model": arguments.local_llm_model,
        "local_embedding_model": arguments.local_embedding_model,
        "base_binding_receipt_path": arguments.base_binding_receipt,
        "attestation_receipt_path": arguments.attestation_receipt,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze-a-hold", "run-a-hold", "freeze-m-search", "run-m-search"):
        command = commands.add_parser(name)
        _add_common(command)
        command.add_argument("--execution-root", type=Path, required=True)
        if name.startswith("freeze-"):
            command.add_argument("--authorization-hash", required=True)
            command.add_argument("--output", type=Path, required=True)
        else:
            command.add_argument("--pre-run-freeze", type=Path, required=True)
        if name in {"freeze-m-search", "run-m-search"}:
            command.add_argument("--a-hold-freeze", type=Path, required=True)
            command.add_argument("--a-hold-report", type=Path, required=True)
    arguments = parser.parse_args(argv)
    common = _common_args(arguments)
    if arguments.command == "freeze-a-hold":
        build_a_hold_pre_run_freeze(
            **common, execution_root=arguments.execution_root,
            authorization_hash=arguments.authorization_hash, output_path=arguments.output,
        )
        return 0
    global _CLEAN_MODULE_CLI_ACTIVE
    if arguments.command == "run-a-hold":
        _CLEAN_MODULE_CLI_ACTIVE = True
        try:
            execute_a_hold_formal(
                **common, pre_run_freeze_path=arguments.pre_run_freeze,
                execution_root=arguments.execution_root,
            )
        finally:
            _CLEAN_MODULE_CLI_ACTIVE = False
        return 0
    anchor = {
        "a_hold_pre_run_freeze_path": arguments.a_hold_freeze,
        "a_hold_report_path": arguments.a_hold_report,
    }
    if arguments.command == "freeze-m-search":
        build_m_search_pre_run_freeze(
            **common, **anchor, execution_root=arguments.execution_root,
            authorization_hash=arguments.authorization_hash, output_path=arguments.output,
        )
        return 0
    _CLEAN_MODULE_CLI_ACTIVE = True
    try:
        execute_m_search_formal(
            **common, **anchor, pre_run_freeze_path=arguments.pre_run_freeze,
            execution_root=arguments.execution_root,
        )
    finally:
        _CLEAN_MODULE_CLI_ACTIVE = False
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
