"""One final offline MuSiQue evaluator transition with equal-compute portfolios.

The evaluator grammar and objectives in this module are inherited verbatim from
the prospectively frozen Hotpot portfolio design: both policies choose two
typed lexical retrievers and every compared arm is ``RRF(P, Q1, Q2)[:5]``.
Only the MuSiQue row parser, variable support-count accounting, retained-P
lineage adapter, and official HippoRAG runtime adapter are domain specific.

Formation completes two independent 2,040-terminal environment grids before
support scoring.  A_hold executes six arm-qualified calls for each of 48 items
(288 terminals).  Only an exact one-sided magnitude-preserving sign-flip
promotion at alpha 0.10 can authorize M_search, whose eight components execute
for 24 items (192 terminals).  Formal freezes do not accept measurement paths;
formal execution is available only through this module's clean CLI.

This module never reads study data at import time.
"""

from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures
from dataclasses import asdict, dataclass, field
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
from typing import Any, Mapping, Sequence

from ..archive import EvaluatorEpoch, EvaluatorSpec, PolicyArchive
from ..models import stable_hash
from . import hotpot_evaluator_portfolio_coevolution_v1 as frozen_core
from . import musique_recursive_study_blocks_v1 as old_blocks
from .hotpot_family_out_runner_v1 import _probe_bubblewrap, verify_capability_receipt
from .musique_formal_runtime_binding_v2 import (
    PreparedFormalRuntimeV2,
    prepare_formal_runtime_v2,
)
from .musique_typed_retriever_formation_v1 import (
    RetrievalParagraph,
    TypedRetrievalProgram,
    enumerate_programs,
    retrieve as typed_retrieve,
)


VERSION = "musique_evaluator_portfolio_coevolution_v1"
IMPLEMENTATION_SCHEMA = f"{VERSION}_implementation"
PRIVATE_CACHE_SCHEMA = f"{VERSION}_private_formation_cache"
FORMATION_RECEIPT_SCHEMA = f"{VERSION}_formation_receipt"
ANCHOR_FREEZE_SCHEMA = f"{VERSION}_A_hold_pre_run_freeze"
ANCHOR_REPORT_SCHEMA = f"{VERSION}_A_hold_aggregate_report"
SEARCH_FREEZE_SCHEMA = f"{VERSION}_M_search_pre_run_freeze"
SEARCH_REPORT_SCHEMA = f"{VERSION}_M_search_aggregate_report"
CONSUMPTION_SCHEMA = f"{VERSION}_authorization_consumption"
FAILURE_SCHEMA = f"{VERSION}_failure"

TOP_K = 5
FOLD_COUNT = 4
FORMATION_ENV_COUNT = 2
FORMATION_ENV_ITEM_COUNT = 24
FORMATION_ITEM_COUNT = 48
A_HOLD_ITEM_COUNT = 48
M_SEARCH_ITEM_COUNT = 24
CANDIDATE_COUNT = 84
PROMOTION_ALPHA = Fraction(1, 10)

A_FORM_ENVIRONMENTS = ("A_form_0", "A_form_1")
F_SEARCH_ENVIRONMENTS = ("F_search_0", "F_search_1")
FORMATION_BLOCKS = A_FORM_ENVIRONMENTS + F_SEARCH_ENVIRONMENTS
ANCHOR_BLOCK = "A_hold"
SEARCH_BLOCK = "M_search"

CAPABILITY_FAMILIES = frozen_core.CAPABILITY_FAMILIES
INCUMBENT_POLICY_ID = frozen_core.INCUMBENT_POLICY_ID
CHALLENGER_POLICY_ID = frozen_core.CHALLENGER_POLICY_ID

ANCHOR_COMPONENT_IDS = (
    "incumbent_P",
    "incumbent_Q1",
    "incumbent_Q2",
    "challenger_P",
    "challenger_Q1",
    "challenger_Q2",
)
SEARCH_COMPONENT_IDS = (
    "canonical_RAW",
    "incumbent_P",
    "incumbent_Q1",
    "incumbent_Q2",
    "active_P",
    "active_Q1",
    "active_Q2",
    "official_HippoRAG",
)
FORMATION_ENV_WORK_UNIT_COUNT = (CANDIDATE_COUNT + 1) * FORMATION_ENV_ITEM_COUNT
FORMATION_WORK_UNIT_COUNT = FORMATION_ENV_COUNT * FORMATION_ENV_WORK_UNIT_COUNT
FORMATION_MAXIMUM_CONCURRENCY = FORMATION_ENV_WORK_UNIT_COUNT
ANCHOR_WORK_UNIT_COUNT = len(ANCHOR_COMPONENT_IDS) * A_HOLD_ITEM_COUNT
ANCHOR_MAXIMUM_CONCURRENCY = ANCHOR_WORK_UNIT_COUNT
SEARCH_WORK_UNIT_COUNT = len(SEARCH_COMPONENT_IDS) * M_SEARCH_ITEM_COUNT
SEARCH_MAXIMUM_CONCURRENCY = SEARCH_WORK_UNIT_COUNT

A_FORM_CONSUMPTION_RELATIVE = (
    "artifacts/musique_evaluator_portfolio_v1/a_form.authorization.consumed.json"
)
F_SEARCH_CONSUMPTION_RELATIVE = (
    "artifacts/musique_evaluator_portfolio_v1/f_search.authorization.consumed.json"
)
DESIGN_RELATIVE = "manifests/musique_evaluator_portfolio_design_v1.json"
# The authoritative transitive file set is the acquisition module's committed
# clean-HEAD closure.  Keeping a second local list here would allow the two
# custody definitions to drift.

ANCHOR_CONSUMPTION_FILENAME = "a_hold.authorization.consumed.json"
ANCHOR_PRIVATE_EVIDENCE_FILENAME = "a_hold.private.evidence.json"
ANCHOR_REPORT_FILENAME = "a_hold.aggregate.report.json"
ANCHOR_FAILURE_FILENAME = "a_hold.failure.json"
SEARCH_CONSUMPTION_FILENAME = "m_search.authorization.consumed.json"
SEARCH_REPORT_FILENAME = "m_search.aggregate.report.json"
SEARCH_FAILURE_FILENAME = "m_search.failure.json"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_CLEAN_MODULE_CLI_ACTIVE = False


class MuSiQueEvaluatorPortfolioError(RuntimeError):
    """The portfolio design, custody, lineage, or formal execution drifted."""


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise MuSiQueEvaluatorPortfolioError("required file is unavailable")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise MuSiQueEvaluatorPortfolioError(f"{field_name} must be lowercase sha256")
    return value


def _read_json(path: str | Path, field_name: str) -> tuple[dict[str, Any], bytes]:
    candidate = Path(path).absolute()
    if candidate.is_symlink() or not candidate.is_file():
        raise MuSiQueEvaluatorPortfolioError(f"{field_name} is unavailable")
    try:
        raw = candidate.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MuSiQueEvaluatorPortfolioError(f"{field_name} is invalid") from exc
    if not isinstance(value, dict):
        raise MuSiQueEvaluatorPortfolioError(f"{field_name} must be an object")
    return value, raw


def _atomic_write_exclusive(destination: Path, raw: bytes, *, mode: int) -> None:
    temporary = destination.parent / (
        f".{destination.name}.{os.urandom(12).hex()}.tmp"
    )
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
            destination.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_json_exclusive(
    path: str | Path, payload: Mapping[str, Any], *, mode: int = 0o600
) -> None:
    destination = Path(path).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, indent=2
    ).encode("utf-8") + b"\n"
    _atomic_write_exclusive(destination, raw, mode=mode)


def _prepare_output_parent(path: str | Path) -> None:
    """Prove exclusive persistence before any one-shot marker is consumed."""

    destination = Path(path).absolute()
    if destination.exists() or destination.is_symlink():
        raise MuSiQueEvaluatorPortfolioError("formal output already exists")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if destination.parent.is_symlink() or not destination.parent.is_dir():
        raise MuSiQueEvaluatorPortfolioError("formal output parent is unavailable")
    canary = destination.parent / (
        f".{destination.name}.persistence-canary-{os.urandom(8).hex()}"
    )
    try:
        _atomic_write_exclusive(
            canary, b"portfolio-persistence-canary\n", mode=0o600
        )
        canary.unlink()
        directory = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        canary.unlink(missing_ok=True)


def _require_private_boundary(path: str | Path) -> None:
    """Require an in-repository private artifact to be ignored and untracked."""

    destination = Path(path).absolute()
    anchor = destination.parent
    while not anchor.exists() and anchor.parent != anchor:
        anchor = anchor.parent
    probe = subprocess.run(
        ["git", "-C", str(anchor), "rev-parse", "--show-toplevel"],
        check=False, capture_output=True, text=True, timeout=30,
    )
    if probe.returncode != 0:
        return
    repository = Path(probe.stdout.strip()).resolve(strict=True)
    try:
        relative = destination.resolve(strict=False).relative_to(repository).as_posix()
    except ValueError as exc:
        raise MuSiQueEvaluatorPortfolioError("private boundary is ambiguous") from exc
    ignored = subprocess.run(
        ["git", "-C", str(repository), "check-ignore", "--no-index", "-q", "--", relative],
        check=False, capture_output=True, timeout=30,
    )
    tracked = subprocess.run(
        ["git", "-C", str(repository), "ls-files", "--", relative],
        check=False, capture_output=True, text=True, timeout=30,
    )
    if ignored.returncode != 0 or tracked.returncode != 0 or tracked.stdout.strip():
        raise MuSiQueEvaluatorPortfolioError(
            "private artifact must be git-ignored and untracked"
        )


def _assert_public_safe(value: Mapping[str, Any]) -> None:
    serialized = json.dumps(value, ensure_ascii=True, sort_keys=True)
    forbidden = (
        '"answers"',
        '"corpus"',
        '"item_id"',
        '"normalized_answers"',
        '"paragraph_text"',
        '"question"',
        '"source_row_sha256"',
        '"support_indices"',
        '"private_root"',
    )
    if any(token in serialized for token in forbidden):
        raise MuSiQueEvaluatorPortfolioError("public artifact contains private content")

    def inspect_locators(node: object) -> None:
        if isinstance(node, Mapping):
            for key, child in node.items():
                if (
                    key in {"path", "private_path", "private_locator"}
                    and isinstance(child, str)
                    and Path(child).is_absolute()
                ):
                    raise MuSiQueEvaluatorPortfolioError(
                        "public artifact contains an absolute private locator"
                    )
                inspect_locators(child)
        elif isinstance(node, list):
            for child in node:
                inspect_locators(child)

    inspect_locators(value)


def current_implementation_binding(project_root: str | Path) -> dict[str, Any]:
    project = Path(project_root).resolve(strict=True)
    module = _acquisition_module()
    try:
        closure = module.implementation_binding(project)
    except Exception as exc:
        raise MuSiQueEvaluatorPortfolioError(
            "portfolio implementation is not the clean tracked HEAD closure"
        ) from exc
    return {
        "schema": IMPLEMENTATION_SCHEMA,
        "clean_HEAD_acquisition_closure": closure,
        "set_sha256": closure["set_sha256"],
    }


def fixed_programs() -> tuple[TypedRetrievalProgram, ...]:
    programs = tuple(enumerate_programs())
    if len(programs) != CANDIDATE_COUNT or any(row.type_issues() for row in programs):
        raise MuSiQueEvaluatorPortfolioError("typed candidate grammar drifted")
    if len({row.program_hash for row in programs}) != CANDIDATE_COUNT:
        raise MuSiQueEvaluatorPortfolioError("typed candidate hashes duplicate")
    return programs


def candidate_set_binding() -> dict[str, Any]:
    programs = fixed_programs()
    rows = [
        {
            "program_sha256": row.program_hash,
            "program_length": row.program_length,
            "seed_algorithm": row.seed_algorithm,
            "expansion_mode": row.expansion_mode,
        }
        for row in programs
    ]
    return {
        "candidate_count": len(rows),
        "candidate_set_sha256": stable_hash(rows),
        "capability_family_count": len(CAPABILITY_FAMILIES),
        "capability_families_sha256": stable_hash(CAPABILITY_FAMILIES),
    }


# The fixed objective is intentionally reused rather than reimplemented for the
# new domain.  These evidence objects already allow any non-empty number of
# supports; no two-support Hotpot assumption is present in their semantics.
GridItemEvidence = frozen_core.GridItemEvidence
ProgramGridEvidence = frozen_core.ProgramGridEvidence
FormationGridEvidence = frozen_core.FormationGridEvidence
PortfolioAssessment = frozen_core.PortfolioAssessment
fuse_rankings = frozen_core.fuse_rankings
canonical_behavior_programs = frozen_core.canonical_behavior_programs
select_portfolio = frozen_core.select_portfolio
incumbent_key = frozen_core.incumbent_key
challenger_key = frozen_core.challenger_key
_program_behavior_sha256 = frozen_core._program_behavior_sha256
_retained_p_behavior_sha256 = frozen_core._retained_p_behavior_sha256


def form_portfolio_policies_from_evidence(
    grid: FormationGridEvidence,
    *,
    expected_environment_ids: Sequence[str],
    retained_p_program_sha256: str,
) -> dict[str, Any]:
    try:
        old = frozen_core.form_portfolio_policies_from_evidence(
            grid,
            expected_environment_ids=expected_environment_ids,
            retained_p_program_sha256=retained_p_program_sha256,
        )
    except frozen_core.HotpotEvaluatorPortfolioError as exc:
        raise MuSiQueEvaluatorPortfolioError("portfolio formation failed closed") from exc
    body = dict(old)
    body.pop("formation_sha256", None)
    body["schema"] = f"{VERSION}_formation_core"
    body["source_family"] = "MuSiQue_official_DEV_residual"
    body["variable_support_count_supported"] = True
    return {**body, "formation_sha256": stable_hash(body)}


def exact_paired_sign_flip(deltas: Sequence[int]) -> dict[str, Any]:
    if not deltas or any(type(value) is not int for value in deltas):
        raise MuSiQueEvaluatorPortfolioError("paired deltas are malformed")
    observed = sum(deltas)
    magnitudes = [abs(value) for value in deltas if value]
    distribution: Counter[int] = Counter({0: 1})
    for magnitude in magnitudes:
        updated: Counter[int] = Counter()
        for subtotal, count in distribution.items():
            updated[subtotal + magnitude] += count
            updated[subtotal - magnitude] += count
        distribution = updated
    denominator = 1 << len(magnitudes)
    numerator = sum(count for subtotal, count in distribution.items() if subtotal >= observed)
    p_value = Fraction(numerator, denominator)
    return {
        "test": "one_sided_exact_paired_sign_flip_v1",
        "observed_net_support_hits": observed,
        "nonzero_pair_count": len(magnitudes),
        "p_value_numerator": p_value.numerator,
        "p_value_denominator": p_value.denominator,
        "p_value": float(p_value),
        "alpha_numerator": PROMOTION_ALPHA.numerator,
        "alpha_denominator": PROMOTION_ALPHA.denominator,
        "promoted": p_value <= PROMOTION_ALPHA,
        "sole_promotion_criterion": True,
    }


def _acquisition_module() -> Any:
    try:
        return importlib.import_module(
            ".musique_evaluator_portfolio_acquisition_v1", package=__package__
        )
    except (ImportError, AttributeError) as exc:
        raise MuSiQueEvaluatorPortfolioError("acquisition implementation unavailable") from exc


@dataclass(frozen=True)
class BlockCommitment:
    block: str
    count: int
    file_sha256: str
    item_commitment_set_sha256: str


def _live_design_binding(project: Path) -> dict[str, Any]:
    design, raw = _read_json(project / DESIGN_RELATIVE, "portfolio design")
    body = dict(design)
    declared = _require_sha256(body.pop("design_sha256", None), "portfolio design")
    mechanism = design.get("mechanism")
    promotion = design.get("promotion_contract")
    execution = design.get("execution_contract")
    cohort = design.get("cohort_contract")
    search = design.get("search_measurement_contract")
    terminal = design.get("terminal_policy")
    if (
        design.get("schema") != "musique_evaluator_portfolio_design_v1"
        or stable_hash(body) != declared
        or not isinstance(mechanism, Mapping)
        or mechanism.get("portfolio_size") != 2
        or mechanism.get("fold_count_per_environment") != 4
        or mechanism.get("environment_count_per_stage") != 2
        or mechanism.get("same_action_fallback") is not False
        or mechanism.get("candidate_program_count") != CANDIDATE_COUNT
        or mechanism.get("incumbent_id") != INCUMBENT_POLICY_ID
        or mechanism.get("challenger_id") != CHALLENGER_POLICY_ID
        or mechanism.get("retained_P_exact_hash_and_gold_free_behavior_aliases_excluded") is not True
        or mechanism.get("portfolio_action") != "deterministic_RRF_P_Q1_Q2_top5"
        or not isinstance(promotion, Mapping)
        or promotion.get("alpha_numerator") != 1
        or promotion.get("alpha_denominator") != 10
        or promotion.get("sole_promotion_criterion") is not True
        or not isinstance(execution, Mapping)
        or execution.get("A_hold_physical_work_units") != 288
        or execution.get("M_search_physical_work_units") != 192
        or execution.get("formation_physical_work_units_per_stage") != 4080
        or execution.get("formation_environment_barrier_count_per_stage") != 2
        or execution.get("formation_scoring_begins_after_both_environment_grids_join") is not True
        or not isinstance(cohort, Mapping)
        or {key: cohort.get(key) for key in (
            "A_form_0", "A_form_1", "F_search_0", "F_search_1", "A_hold", "M_search"
        )} != {
            "A_form_0": 24, "A_form_1": 24, "F_search_0": 24,
            "F_search_1": 24, "A_hold": 48, "M_search": 24,
        }
        or not isinstance(search, Mapping)
        or search.get("L5_improved_search_requires_exact_p_at_or_below_alpha") is not True
        or search.get("L5_improved_search_requires_positive_active_minus_incumbent_net") is not True
        or search.get("M_search_does_not_affect_or_rollback_evaluator_transition") is not True
        or not isinstance(terminal, Mapping)
        or terminal.get("additional_MuSiQue_same_source_evaluator_attempt_after_outcome") is not False
        or terminal.get("new_cohort_retry_replay_resample") is not False
    ):
        raise MuSiQueEvaluatorPortfolioError("portfolio design drifted")
    module = _acquisition_module()
    try:
        binding = module.portfolio_design_binding(project)
    except Exception as exc:
        raise MuSiQueEvaluatorPortfolioError("portfolio design custody drifted") from exc
    if (
        binding.get("schema") != design["schema"]
        or binding.get("design_sha256") != declared
        or binding.get("design_file_sha256") != _sha256_bytes(raw)
    ):
        raise MuSiQueEvaluatorPortfolioError("portfolio design binding drifted")
    return dict(binding)


def _load_acquisition_live(
    *, project: Path, path: str | Path, selection_secret_path: str | Path
) -> tuple[dict[str, Any], bytes, dict[str, BlockCommitment]]:
    module = _acquisition_module()
    try:
        receipt, rows = module.load_acquisition_binding_live(
            project=project,
            path=path,
            selection_secret_path=Path(selection_secret_path),
        )
    except Exception as exc:
        raise MuSiQueEvaluatorPortfolioError(
            "canonical live acquisition binding drifted"
        ) from exc
    raw = Path(path).absolute().read_bytes()
    expected_counts = {
        "A_form_0": 24,
        "A_form_1": 24,
        "F_search_0": 24,
        "F_search_1": 24,
        "A_hold": 48,
        "M_search": 24,
    }
    commitments = {
        row.block: BlockCommitment(
            block=row.block,
            count=row.count,
            file_sha256=_require_sha256(row.file_sha256, "block file"),
            item_commitment_set_sha256=_require_sha256(
                row.item_commitment_set_sha256, "block item commitments"
            ),
        )
        for row in rows
    }
    if (
        getattr(module, "BLOCK_COUNTS", None) != expected_counts
        or tuple(getattr(module, "BLOCK_ORDER", ())) != tuple(expected_counts)
        or {key: row.count for key, row in commitments.items()} != expected_counts
    ):
        raise MuSiQueEvaluatorPortfolioError("acquisition block contract drifted")
    design = receipt.get("portfolio_design_binding")
    try:
        live_implementation = module.implementation_binding(project)
        live_lineage = module.prior_study_lineage_binding(project)
    except Exception as exc:
        raise MuSiQueEvaluatorPortfolioError(
            "live acquisition implementation or lineage custody drifted"
        ) from exc
    if design != _live_design_binding(project):
        raise MuSiQueEvaluatorPortfolioError("acquisition design binding drifted")
    if receipt.get("implementation") != live_implementation:
        raise MuSiQueEvaluatorPortfolioError("acquisition implementation binding drifted")
    if receipt.get("retained_P_lineage") != live_lineage:
        raise MuSiQueEvaluatorPortfolioError("acquisition retained-P lineage drifted")
    _assert_public_safe(receipt)
    return dict(receipt), raw, commitments


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

    @property
    def item_commitment_sha256(self) -> str:
        return self.view.item_commitment_sha256


def _row_to_item(row: Mapping[str, Any], *, expected_block: str) -> StudyItem:
    corpus_raw = row.get("corpus")
    support_raw = row.get("support_indices")
    question = row.get("question")
    if (
        row.get("block") != expected_block
        or not isinstance(question, str)
        or not question.strip()
        or not isinstance(corpus_raw, list)
        or not isinstance(support_raw, list)
    ):
        raise MuSiQueEvaluatorPortfolioError("private MuSiQue row schema drifted")
    corpus: list[RetrievalParagraph] = []
    labels: list[int] = []
    for ordinal, paragraph in enumerate(corpus_raw):
        if not isinstance(paragraph, Mapping) or set(paragraph) != {
            "idx", "is_supporting", "text", "title"
        }:
            raise MuSiQueEvaluatorPortfolioError("MuSiQue paragraph schema drifted")
        idx = paragraph.get("idx")
        title = paragraph.get("title")
        text = paragraph.get("text")
        supporting = paragraph.get("is_supporting")
        if (
            type(idx) is not int
            or idx != ordinal
            or not isinstance(title, str)
            or not title.strip()
            or not isinstance(text, str)
            or not text.strip()
            or type(supporting) is not bool
        ):
            raise MuSiQueEvaluatorPortfolioError("MuSiQue paragraph value drifted")
        corpus.append(RetrievalParagraph(idx=idx, title=title, text=text))
        if supporting:
            labels.append(idx)
    if (
        len(corpus) < TOP_K
        or tuple(support_raw) != tuple(labels)
        or not labels
        or len(set(labels)) != len(labels)
    ):
        raise MuSiQueEvaluatorPortfolioError("MuSiQue support envelope drifted")
    return StudyItem(
        view=RetrievalItem(
            question=question,
            corpus=tuple(corpus),
            item_commitment_sha256=stable_hash(row),
        ),
        support_indices=tuple(labels),
    )


def _load_block(
    *, path: str | Path, expected_block: str, commitment: BlockCommitment
) -> tuple[StudyItem, ...]:
    module = _acquisition_module()
    rows = module.load_private_block(
        path,
        expected_block=expected_block,
        commitment=module.BlockCommitment(
            block=commitment.block,
            count=commitment.count,
            file_sha256=commitment.file_sha256,
            item_commitment_set_sha256=commitment.item_commitment_set_sha256,
        ),
    )
    items = tuple(_row_to_item(row, expected_block=expected_block) for row in rows)
    if len(items) != commitment.count or len(
        {row.item_commitment_sha256 for row in items}
    ) != commitment.count:
        raise MuSiQueEvaluatorPortfolioError("private block item closure drifted")
    return items


def _ranking(program: TypedRetrievalProgram, item: RetrievalItem) -> tuple[int, ...]:
    try:
        result = tuple(typed_retrieve(program, item.question, item.corpus))
    except (TypeError, ValueError) as exc:
        raise MuSiQueEvaluatorPortfolioError("typed retrieval failed") from exc
    if (
        len(result) != TOP_K
        or len(set(result)) != TOP_K
        or any(type(idx) is not int or not 0 <= idx < len(item.corpus) for idx in result)
    ):
        raise MuSiQueEvaluatorPortfolioError("retrieval ranking drifted")
    return result


def _load_p_lineage(
    *,
    project: Path,
    acquisition_receipt: Mapping[str, Any],
    p_formation_receipt_path: str | Path,
    p_frozen_program_path: str | Path,
    m1_freeze_path: str | Path,
    m1_report_path: str | Path,
) -> tuple[TypedRetrievalProgram, dict[str, Any]]:
    module = _acquisition_module()
    try:
        lineage = module.prior_study_lineage_binding(project)
    except Exception as exc:
        raise MuSiQueEvaluatorPortfolioError("retained P custody drifted") from exc
    if acquisition_receipt.get("retained_P_lineage") != lineage:
        raise MuSiQueEvaluatorPortfolioError("acquisition retained-P binding drifted")
    supplied = {
        "retained_P_formation_receipt": p_formation_receipt_path,
        "retained_P_frozen_program": p_frozen_program_path,
        "retained_P_measurement_freeze": m1_freeze_path,
        "retained_P_measurement_report": m1_report_path,
    }
    lineage_rows = {
        row["role"]: row for row in lineage.get("files", []) if isinstance(row, Mapping)
    }
    for role, path in supplied.items():
        row = lineage_rows.get(role)
        if (
            not isinstance(row, Mapping)
            or Path(path).resolve(strict=True) != (project / row["relative_path"]).resolve(strict=True)
            or _sha256_file(Path(path).resolve(strict=True)) != row.get("file_sha256")
        ):
            raise MuSiQueEvaluatorPortfolioError("retained P supplied path drifted")
    try:
        program, receipt, envelope = old_blocks.load_study_frozen_program(
            frozen_program_path=p_frozen_program_path,
            formation_receipt_path=p_formation_receipt_path,
            verify_live=True,
            implementation_root=project,
        )
    except old_blocks.MuSiQueStudyBlockError as exc:
        raise MuSiQueEvaluatorPortfolioError("retained P lineage drifted") from exc
    m1_freeze, m1_freeze_raw = _read_json(m1_freeze_path, "old M1 freeze")
    m1_report, m1_report_raw = _read_json(m1_report_path, "old M1 report")
    if (
        m1_report.get("valid") is not True
        or m1_report.get("raw_content_persisted") is not False
        or envelope.get("program_hash") != program.program_hash
        or program.program_hash not in {row.program_hash for row in fixed_programs()}
    ):
        raise MuSiQueEvaluatorPortfolioError("retained P evidence is invalid")
    del receipt, m1_freeze, m1_freeze_raw, m1_report_raw
    return program, dict(lineage)


def _grid_to_dict(grid: FormationGridEvidence) -> dict[str, Any]:
    return {
        "environment_ids": list(grid.environment_ids),
        "items": [
            [
                {
                    "item_commitment_sha256": item.item_commitment_sha256,
                    "p_ranking": list(item.p_ranking),
                    "support_indices": list(item.support_indices),
                }
                for item in environment
            ]
            for environment in grid.items
        ],
        "programs": [
            {
                "program_sha256": program.program_sha256,
                "program_length": program.program_length,
                "seed_algorithm": program.seed_algorithm,
                "expansion_mode": program.expansion_mode,
                "q_rankings": [
                    [None if row is None else list(row) for row in environment]
                    for environment in program.q_rankings
                ],
            }
            for program in grid.programs
        ],
    }


def _grid_from_dict(value: object) -> FormationGridEvidence:
    if not isinstance(value, Mapping):
        raise MuSiQueEvaluatorPortfolioError("formation grid is malformed")
    try:
        grid = FormationGridEvidence(
            environment_ids=tuple(value["environment_ids"]),
            items=tuple(
                tuple(
                    GridItemEvidence(
                        item_commitment_sha256=row["item_commitment_sha256"],
                        p_ranking=tuple(row["p_ranking"]),
                        support_indices=tuple(row["support_indices"]),
                    )
                    for row in environment
                )
                for environment in value["items"]
            ),
            programs=tuple(
                ProgramGridEvidence(
                    program_sha256=row["program_sha256"],
                    program_length=row["program_length"],
                    seed_algorithm=row["seed_algorithm"],
                    expansion_mode=row["expansion_mode"],
                    q_rankings=tuple(
                        tuple(None if ranking is None else tuple(ranking) for ranking in env)
                        for env in row["q_rankings"]
                    ),
                )
                for row in value["programs"]
            ),
        )
        return grid.validate()
    except (KeyError, TypeError, ValueError, frozen_core.HotpotEvaluatorPortfolioError) as exc:
        raise MuSiQueEvaluatorPortfolioError("formation grid is malformed") from exc


def formation_evidence_sha256(grid: FormationGridEvidence) -> str:
    return stable_hash(_grid_to_dict(grid.validate()))


def _submit_all_then_join(
    *,
    executor: concurrent.futures.Executor,
    function: Any,
    work_units: Sequence[Any],
) -> list[Any]:
    """Submit the complete barrier width before awaiting any terminal."""

    futures = tuple(executor.submit(function, unit) for unit in work_units)
    if len(futures) != len(work_units):
        raise MuSiQueEvaluatorPortfolioError("bulk submission width drifted")
    return [future.result() for future in futures]


def _evaluate_formation_grid(
    *,
    p_program: TypedRetrievalProgram,
    environment_ids: Sequence[str],
    environments: Sequence[Sequence[StudyItem]],
) -> tuple[FormationGridEvidence, dict[str, Any]]:
    """Execute the exact two-barrier, 4,080-terminal formation grid."""

    if (
        tuple(environment_ids) not in {A_FORM_ENVIRONMENTS, F_SEARCH_ENVIRONMENTS}
        or len(environments) != FORMATION_ENV_COUNT
        or any(len(rows) != FORMATION_ENV_ITEM_COUNT for rows in environments)
    ):
        raise MuSiQueEvaluatorPortfolioError("formation environments drifted")
    programs = fixed_programs()
    work_units = tuple(
        (environment_ordinal, item_ordinal, program_ordinal, item.view)
        for environment_ordinal, rows in enumerate(environments)
        for item_ordinal, item in enumerate(rows)
        for program_ordinal in range(-1, CANDIDATE_COUNT)
    )
    if len(work_units) != FORMATION_WORK_UNIT_COUNT:
        raise MuSiQueEvaluatorPortfolioError("formation grid width drifted")
    attempted = completed = 0
    lock = threading.Lock()

    def run_one(
        unit: tuple[int, int, int, RetrievalItem], barrier: threading.Barrier
    ) -> tuple[tuple[int, int, int], tuple[int, ...] | None]:
        nonlocal attempted, completed
        environment_ordinal, item_ordinal, program_ordinal, item = unit
        with lock:
            attempted += 1
        try:
            barrier.wait(timeout=180)
        except threading.BrokenBarrierError as exc:
            raise MuSiQueEvaluatorPortfolioError(
                "formation maximum-width barrier did not close"
            ) from exc
        try:
            ranking = _ranking(
                p_program if program_ordinal == -1 else programs[program_ordinal], item
            )
        except MuSiQueEvaluatorPortfolioError:
            if program_ordinal == -1:
                raise
            ranking = None
        with lock:
            completed += 1
        return (environment_ordinal, item_ordinal, program_ordinal), ranking

    terminals: list[tuple[tuple[int, int, int], tuple[int, ...] | None]] = []
    for environment_ordinal in range(FORMATION_ENV_COUNT):
        units = tuple(unit for unit in work_units if unit[0] == environment_ordinal)
        if len(units) != FORMATION_ENV_WORK_UNIT_COUNT:
            raise MuSiQueEvaluatorPortfolioError("formation environment width drifted")
        barrier = threading.Barrier(FORMATION_ENV_WORK_UNIT_COUNT)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=FORMATION_MAXIMUM_CONCURRENCY,
            thread_name_prefix=f"musique-portfolio-env{environment_ordinal}",
        ) as executor:
            terminals.extend(_submit_all_then_join(
                executor=executor,
                function=lambda unit: run_one(unit, barrier),
                work_units=units,
            ))
    direct = dict(terminals)
    if (
        attempted != FORMATION_WORK_UNIT_COUNT
        or completed != FORMATION_WORK_UNIT_COUNT
        or len(direct) != FORMATION_WORK_UNIT_COUNT
    ):
        raise MuSiQueEvaluatorPortfolioError("formation terminal closure incomplete")

    # Support labels enter the evidence object only after both retrieval grids join.
    item_grid = tuple(
        tuple(
            GridItemEvidence(
                item_commitment_sha256=item.item_commitment_sha256,
                p_ranking=direct[(environment_ordinal, item_ordinal, -1)],  # type: ignore[arg-type]
                support_indices=item.support_indices,
            ).validate()
            for item_ordinal, item in enumerate(rows)
        )
        for environment_ordinal, rows in enumerate(environments)
    )
    program_grid = tuple(
        ProgramGridEvidence(
            program_sha256=program.program_hash,
            program_length=program.program_length,
            seed_algorithm=program.seed_algorithm,
            expansion_mode=program.expansion_mode,
            q_rankings=tuple(
                tuple(
                    direct[(environment_ordinal, item_ordinal, program_ordinal)]
                    for item_ordinal in range(FORMATION_ENV_ITEM_COUNT)
                )
                for environment_ordinal in range(FORMATION_ENV_COUNT)
            ),
        ).validate(FORMATION_ENV_COUNT, FORMATION_ENV_ITEM_COUNT)
        for program_ordinal, program in enumerate(programs)
    )
    grid = FormationGridEvidence(
        environment_ids=tuple(environment_ids), items=item_grid, programs=program_grid
    ).validate(environment_ids)
    return grid, {
        "candidate_program_count": CANDIDATE_COUNT,
        "environment_count": FORMATION_ENV_COUNT,
        "environment_item_count": FORMATION_ENV_ITEM_COUNT,
        "item_count": FORMATION_ITEM_COUNT,
        "physical_work_unit_count": FORMATION_WORK_UNIT_COUNT,
        "retrieval_attempt_count": attempted,
        "retrieval_terminal_count": completed,
        "configured_maximum_concurrency": FORMATION_MAXIMUM_CONCURRENCY,
        "environment_barrier_count": FORMATION_ENV_COUNT,
        "environment_barrier_party_count": FORMATION_ENV_WORK_UNIT_COUNT,
        "all_terminals_joined_before_support_scoring": True,
        "variable_support_count_supported": True,
        "model_calls": 0,
        "external_network_calls": 0,
        "online_evaluator_calls": 0,
        "retries": 0,
        "replays": 0,
        "resamples": 0,
    }


def _formation_marker_path(project: Path, stage: str) -> Path:
    relative = {
        "A_form": A_FORM_CONSUMPTION_RELATIVE,
        "F_search": F_SEARCH_CONSUMPTION_RELATIVE,
    }.get(stage)
    if relative is None:
        raise MuSiQueEvaluatorPortfolioError("unknown formation stage")
    return project / relative


def _write_formation_marker(
    *,
    project: Path,
    stage: str,
    acquisition_file_sha256: str,
    output_cache_path: str | Path,
    output_receipt_path: str | Path,
) -> dict[str, Any]:
    marker_path = _formation_marker_path(project, stage)
    if marker_path.exists():
        raise MuSiQueEvaluatorPortfolioError(f"{stage} authorization already consumed")
    body = {
        "schema": CONSUMPTION_SCHEMA,
        "stage": stage,
        "acquisition_file_sha256": _require_sha256(
            acquisition_file_sha256, "acquisition file"
        ),
        "private_output_locator_sha256": stable_hash(
            {"absolute_output": str(Path(output_cache_path).absolute())}
        ),
        "public_output_locator_sha256": stable_hash(
            {"absolute_output": str(Path(output_receipt_path).absolute())}
        ),
        "marker_written_before_both_private_environment_blocks_open": True,
        "private_block_rows_opened_before_marker": 0,
        "replay_authorized": False,
        "raw_content_persisted": False,
    }
    marker = {**body, "consumption_sha256": stable_hash(body)}
    _assert_public_safe(marker)
    _write_json_exclusive(marker_path, marker)
    return marker


def _load_formation_marker(
    *,
    project: Path,
    stage: str,
    acquisition_file_sha256: str,
    private_cache_path: str | Path,
    public_receipt_path: str | Path,
) -> dict[str, Any]:
    marker, _raw = _read_json(_formation_marker_path(project, stage), "formation marker")
    body = dict(marker)
    declared = _require_sha256(body.pop("consumption_sha256", None), "consumption")
    expected = {
        "schema": CONSUMPTION_SCHEMA,
        "stage": stage,
        "acquisition_file_sha256": acquisition_file_sha256,
        "private_output_locator_sha256": stable_hash(
            {"absolute_output": str(Path(private_cache_path).absolute())}
        ),
        "public_output_locator_sha256": stable_hash(
            {"absolute_output": str(Path(public_receipt_path).absolute())}
        ),
        "marker_written_before_both_private_environment_blocks_open": True,
        "private_block_rows_opened_before_marker": 0,
        "replay_authorized": False,
        "raw_content_persisted": False,
    }
    if body != expected or stable_hash(body) != declared:
        raise MuSiQueEvaluatorPortfolioError("formation marker drifted")
    return marker


def _formation_source_binding(
    *,
    receipt: Mapping[str, Any],
    receipt_raw: bytes,
    commitments: Mapping[str, BlockCommitment],
    environment_ids: Sequence[str],
) -> dict[str, Any]:
    return {
        "acquisition_sha256": receipt["acquisition_sha256"],
        "acquisition_file_sha256": _sha256_bytes(receipt_raw),
        "private_pack_sha256": receipt["commitments"]["private_pack_sha256"],
        "portfolio_design_binding": receipt["portfolio_design_binding"],
        "environment_block_id_hashes": [
            stable_hash({"block": block}) for block in environment_ids
        ],
        "environment_block_file_sha256s": [
            commitments[block].file_sha256 for block in environment_ids
        ],
        "environment_item_commitment_set_sha256s": [
            commitments[block].item_commitment_set_sha256 for block in environment_ids
        ],
        "environment_item_counts": [commitments[block].count for block in environment_ids],
    }


def _form_stage(
    *,
    stage: str,
    project_root: str | Path,
    acquisition_receipt_path: str | Path,
    selection_secret_path: str | Path,
    environment_0_block_path: str | Path,
    environment_1_block_path: str | Path,
    p_formation_receipt_path: str | Path,
    p_frozen_program_path: str | Path,
    m1_freeze_path: str | Path,
    m1_report_path: str | Path,
    private_cache_output_path: str | Path,
    public_receipt_output_path: str | Path,
) -> dict[str, Any]:
    project = Path(project_root).resolve(strict=True)
    environment_ids = A_FORM_ENVIRONMENTS if stage == "A_form" else F_SEARCH_ENVIRONMENTS
    receipt, receipt_raw, commitments = _load_acquisition_live(
        project=project,
        path=acquisition_receipt_path,
        selection_secret_path=selection_secret_path,
    )
    p_program, lineage = _load_p_lineage(
        project=project,
        acquisition_receipt=receipt,
        p_formation_receipt_path=p_formation_receipt_path,
        p_frozen_program_path=p_frozen_program_path,
        m1_freeze_path=m1_freeze_path,
        m1_report_path=m1_report_path,
    )
    output_paths = {
        Path(private_cache_output_path).absolute(),
        Path(public_receipt_output_path).absolute(),
        _formation_marker_path(project, stage).absolute(),
    }
    if len(output_paths) != 3:
        raise MuSiQueEvaluatorPortfolioError("formation outputs and marker must be distinct")
    _require_private_boundary(private_cache_output_path)
    _prepare_output_parent(private_cache_output_path)
    _prepare_output_parent(public_receipt_output_path)
    marker = _write_formation_marker(
        project=project,
        stage=stage,
        acquisition_file_sha256=_sha256_bytes(receipt_raw),
        output_cache_path=private_cache_output_path,
        output_receipt_path=public_receipt_output_path,
    )
    items = (
        _load_block(
            path=environment_0_block_path,
            expected_block=environment_ids[0],
            commitment=commitments[environment_ids[0]],
        ),
        _load_block(
            path=environment_1_block_path,
            expected_block=environment_ids[1],
            commitment=commitments[environment_ids[1]],
        ),
    )
    grid, execution = _evaluate_formation_grid(
        p_program=p_program, environment_ids=environment_ids, environments=items
    )
    core = form_portfolio_policies_from_evidence(
        grid,
        expected_environment_ids=environment_ids,
        retained_p_program_sha256=p_program.program_hash,
    )
    source = _formation_source_binding(
        receipt=receipt,
        receipt_raw=receipt_raw,
        commitments=commitments,
        environment_ids=environment_ids,
    )
    cache_body = {
        "schema": PRIVATE_CACHE_SCHEMA,
        "stage": stage,
        "source_binding": source,
        "lineage_binding": lineage,
        "candidate_set_binding": candidate_set_binding(),
        "grid": _grid_to_dict(grid),
        "formation_core": core,
        "execution": execution,
        "raw_question_or_corpus_persisted": False,
    }
    cache = {**cache_body, "cache_sha256": stable_hash(cache_body)}
    _write_json_exclusive(private_cache_output_path, cache)
    cache_raw = Path(private_cache_output_path).absolute().read_bytes()
    receipt_body = {
        "schema": FORMATION_RECEIPT_SCHEMA,
        "stage": stage,
        "valid": True,
        "status": (
            "behavior_distinct_portfolio_frozen"
            if core["measurable_contrast"]
            else "terminal_no_behavior_distinct_portfolio"
        ),
        "implementation": current_implementation_binding(project),
        "source_binding": source,
        "portfolio_design_binding": receipt["portfolio_design_binding"],
        "lineage_binding": lineage,
        "candidate_set_binding": candidate_set_binding(),
        "formation_core": core,
        "private_cache_binding": {
            "file_sha256": _sha256_bytes(cache_raw),
            "cache_sha256": cache["cache_sha256"],
            "formation_evidence_sha256": formation_evidence_sha256(grid),
            "private_path_persisted_publicly": False,
            "item_level_evidence_persisted_publicly": False,
        },
        "authorization_consumption_sha256": marker["consumption_sha256"],
        "execution": execution,
        "A_hold_opened": False,
        "M_search_opened": False,
        "raw_content_persisted": False,
    }
    public = {**receipt_body, "receipt_sha256": stable_hash(receipt_body)}
    _assert_public_safe(public)
    _write_json_exclusive(public_receipt_output_path, public, mode=0o644)
    return public


def form_a_form_stage(**kwargs: Any) -> dict[str, Any]:
    return _form_stage(stage="A_form", **kwargs)


def form_f_search_stage(**kwargs: Any) -> dict[str, Any]:
    return _form_stage(stage="F_search", **kwargs)


def _load_formation_bundle(
    *,
    project: Path,
    private_cache_path: str | Path,
    public_receipt_path: str | Path,
    expected_stage: str,
    acquisition_receipt: Mapping[str, Any],
    acquisition_raw: bytes,
    commitments: Mapping[str, BlockCommitment],
    p_lineage: Mapping[str, Any],
    retained_p_program_sha256: str,
) -> tuple[FormationGridEvidence, dict[str, Any], dict[str, Any]]:
    public, public_raw = _read_json(public_receipt_path, "formation receipt")
    public_body = dict(public)
    declared = _require_sha256(public_body.pop("receipt_sha256", None), "formation receipt")
    cache, cache_raw = _read_json(private_cache_path, "private formation cache")
    cache_body = dict(cache)
    cache_declared = _require_sha256(cache_body.pop("cache_sha256", None), "formation cache")
    environment_ids = A_FORM_ENVIRONMENTS if expected_stage == "A_form" else F_SEARCH_ENVIRONMENTS
    source = _formation_source_binding(
        receipt=acquisition_receipt,
        receipt_raw=acquisition_raw,
        commitments=commitments,
        environment_ids=environment_ids,
    )
    marker = _load_formation_marker(
        project=project,
        stage=expected_stage,
        acquisition_file_sha256=_sha256_bytes(acquisition_raw),
        private_cache_path=private_cache_path,
        public_receipt_path=public_receipt_path,
    )
    grid = _grid_from_dict(cache.get("grid"))
    expected_core = form_portfolio_policies_from_evidence(
        grid,
        expected_environment_ids=environment_ids,
        retained_p_program_sha256=retained_p_program_sha256,
    )
    binding = public.get("private_cache_binding")
    if (
        public.get("schema") != FORMATION_RECEIPT_SCHEMA
        or public.get("stage") != expected_stage
        or public.get("valid") is not True
        or public.get("status") != (
            "behavior_distinct_portfolio_frozen"
            if expected_core["measurable_contrast"]
            else "terminal_no_behavior_distinct_portfolio"
        )
        or stable_hash(public_body) != declared
        or public.get("implementation") != current_implementation_binding(project)
        or public.get("source_binding") != source
        or public.get("portfolio_design_binding") != acquisition_receipt.get("portfolio_design_binding")
        or public.get("lineage_binding") != p_lineage
        or public.get("candidate_set_binding") != candidate_set_binding()
        or public.get("formation_core") != expected_core
        or public.get("authorization_consumption_sha256") != marker["consumption_sha256"]
        or public.get("A_hold_opened") is not False
        or public.get("M_search_opened") is not False
        or public.get("raw_content_persisted") is not False
        or cache.get("schema") != PRIVATE_CACHE_SCHEMA
        or cache.get("stage") != expected_stage
        or stable_hash(cache_body) != cache_declared
        or cache.get("source_binding") != source
        or cache.get("lineage_binding") != p_lineage
        or cache.get("candidate_set_binding") != candidate_set_binding()
        or cache.get("formation_core") != expected_core
        or public.get("execution") != cache.get("execution")
        or not isinstance(cache.get("execution"), Mapping)
        or cache["execution"].get("physical_work_unit_count") != FORMATION_WORK_UNIT_COUNT
        or cache["execution"].get("retrieval_terminal_count") != FORMATION_WORK_UNIT_COUNT
        or cache["execution"].get("environment_barrier_count") != FORMATION_ENV_COUNT
        or cache["execution"].get("environment_barrier_party_count") != FORMATION_ENV_WORK_UNIT_COUNT
        or cache["execution"].get("all_terminals_joined_before_support_scoring") is not True
        or not isinstance(binding, Mapping)
        or binding.get("file_sha256") != _sha256_bytes(cache_raw)
        or binding.get("cache_sha256") != cache_declared
        or binding.get("formation_evidence_sha256") != formation_evidence_sha256(grid)
    ):
        raise MuSiQueEvaluatorPortfolioError("formation artifact bundle drifted")
    _assert_public_safe(public)
    public_binding = {
        "stage": expected_stage,
        "public_receipt_file_sha256": _sha256_bytes(public_raw),
        "public_receipt_sha256": declared,
        "private_cache_file_sha256": _sha256_bytes(cache_raw),
        "private_cache_sha256": cache_declared,
        "formation_evidence_sha256": formation_evidence_sha256(grid),
        "formation_core_sha256": expected_core["formation_sha256"],
        "private_path_persisted_publicly": False,
    }
    return grid, public, public_binding


def _program_by_hash(program_sha256: str) -> TypedRetrievalProgram:
    target = _require_sha256(program_sha256, "program")
    matches = [row for row in fixed_programs() if row.program_hash == target]
    if len(matches) != 1:
        raise MuSiQueEvaluatorPortfolioError("selected program is outside grammar")
    return matches[0]


def _program_pair(core: Mapping[str, Any], role: str) -> tuple[TypedRetrievalProgram, ...]:
    selected = core.get(role)
    if not isinstance(selected, Mapping):
        raise MuSiQueEvaluatorPortfolioError("selected portfolio is malformed")
    hashes = selected.get("program_sha256s")
    if not isinstance(hashes, list) or len(hashes) != 2 or hashes[0] == hashes[1]:
        raise MuSiQueEvaluatorPortfolioError("selected portfolio width drifted")
    pair = tuple(_program_by_hash(value) for value in hashes)
    if tuple(sorted(row.program_hash for row in pair)) != tuple(hashes):
        raise MuSiQueEvaluatorPortfolioError("selected portfolio ordering drifted")
    if pair[0].seed_algorithm == pair[1].seed_algorithm and pair[0].expansion_mode == pair[1].expansion_mode:
        raise MuSiQueEvaluatorPortfolioError("selected portfolio repeats a capability family")
    return pair


def _artifact_bundles(
    *,
    project: Path,
    acquisition_receipt_path: str | Path,
    selection_secret_path: str | Path,
    p_formation_receipt_path: str | Path,
    p_frozen_program_path: str | Path,
    m1_freeze_path: str | Path,
    m1_report_path: str | Path,
    a_form_private_cache_path: str | Path,
    a_form_public_receipt_path: str | Path,
    f_search_private_cache_path: str | Path,
    f_search_public_receipt_path: str | Path,
) -> tuple[
    dict[str, Any], bytes, dict[str, BlockCommitment], TypedRetrievalProgram,
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]
]:
    receipt, raw, commitments = _load_acquisition_live(
        project=project,
        path=acquisition_receipt_path,
        selection_secret_path=selection_secret_path,
    )
    p_program, lineage = _load_p_lineage(
        project=project,
        acquisition_receipt=receipt,
        p_formation_receipt_path=p_formation_receipt_path,
        p_frozen_program_path=p_frozen_program_path,
        m1_freeze_path=m1_freeze_path,
        m1_report_path=m1_report_path,
    )
    _a_grid, a_public, a_binding = _load_formation_bundle(
        project=project,
        private_cache_path=a_form_private_cache_path,
        public_receipt_path=a_form_public_receipt_path,
        expected_stage="A_form",
        acquisition_receipt=receipt,
        acquisition_raw=raw,
        commitments=commitments,
        p_lineage=lineage,
        retained_p_program_sha256=p_program.program_hash,
    )
    _f_grid, f_public, f_binding = _load_formation_bundle(
        project=project,
        private_cache_path=f_search_private_cache_path,
        public_receipt_path=f_search_public_receipt_path,
        expected_stage="F_search",
        acquisition_receipt=receipt,
        acquisition_raw=raw,
        commitments=commitments,
        p_lineage=lineage,
        retained_p_program_sha256=p_program.program_hash,
    )
    return (
        receipt, raw, commitments, p_program, lineage,
        a_public, f_public, a_binding, f_binding,
    )


def _new_root(path: str | Path) -> Path:
    candidate = Path(path).absolute()
    if candidate.is_symlink() or not candidate.parent.resolve(strict=True).is_dir():
        raise MuSiQueEvaluatorPortfolioError("formal execution-root parent unavailable")
    return candidate


def _root_hash(path: str | Path) -> str:
    return stable_hash({"absolute_execution_root": str(_new_root(path))})


def _source_binding(
    receipt: Mapping[str, Any], raw: bytes, commitment: BlockCommitment
) -> dict[str, Any]:
    return {
        "acquisition_sha256": receipt["acquisition_sha256"],
        "acquisition_file_sha256": _sha256_bytes(raw),
        "private_pack_sha256": receipt["commitments"]["private_pack_sha256"],
        "portfolio_design_binding": receipt["portfolio_design_binding"],
        "measurement_block_id_hash": stable_hash({"block": commitment.block}),
        "measurement_block_file_sha256": commitment.file_sha256,
        "measurement_item_commitment_set_sha256": commitment.item_commitment_set_sha256,
        "measurement_item_count": commitment.count,
    }


def _anchor_execution_contract() -> dict[str, Any]:
    return {
        "physical_component_ids": list(ANCHOR_COMPONENT_IDS),
        "item_count": A_HOLD_ITEM_COUNT,
        "physical_work_unit_count": ANCHOR_WORK_UNIT_COUNT,
        "maximum_concurrency": ANCHOR_MAXIMUM_CONCURRENCY,
        "single_start_barrier_party_count": ANCHOR_WORK_UNIT_COUNT,
        "derived_arms": ["incumbent_portfolio", "challenger_portfolio"],
        "logical_retrieval_calls_per_compared_arm_item": 3,
        "all_terminals_join_before_offline_support_scoring": True,
        "variable_support_count_supported": True,
        "promotion_test": "one_sided_exact_paired_sign_flip_v1",
        "promotion_alpha_numerator": 1,
        "promotion_alpha_denominator": 10,
        "sole_promotion_criterion": True,
        "external_network_calls": 0,
        "online_evaluator_calls": 0,
        "retries": 0,
        "replays": 0,
        "resamples": 0,
    }


def _search_execution_contract() -> dict[str, Any]:
    return {
        "physical_component_ids": list(SEARCH_COMPONENT_IDS),
        "item_count": M_SEARCH_ITEM_COUNT,
        "physical_work_unit_count": SEARCH_WORK_UNIT_COUNT,
        "maximum_concurrency": SEARCH_MAXIMUM_CONCURRENCY,
        "single_start_barrier_party_count": SEARCH_WORK_UNIT_COUNT,
        "derived_arms": [
            "canonical_RAW", "retained_P", "incumbent_portfolio",
            "active_portfolio", "official_HippoRAG",
        ],
        "primary_comparison": "active_portfolio_minus_incumbent_portfolio",
        "secondary_comparisons": [
            "active_portfolio_minus_official_HippoRAG",
            "active_portfolio_minus_canonical_RAW",
            "active_portfolio_minus_retained_P",
        ],
        "logical_retrieval_calls_per_primary_arm_item": 3,
        "L5_achievement": "positive_net_and_exact_sign_flip_p_le_0_10",
        "M_search_does_not_change_evaluator_epoch": True,
        "all_terminals_join_before_runtime_postflight_and_offline_scoring": True,
        "external_network_calls": 0,
        "online_evaluator_calls": 0,
        "retries": 0,
        "replays": 0,
        "resamples": 0,
    }


def _aggregate_arm(
    *, arm_id: str, items: Sequence[StudyItem], rankings: Sequence[Sequence[int]]
) -> dict[str, Any]:
    if len(items) != len(rankings) or not items:
        raise MuSiQueEvaluatorPortfolioError("arm aggregate length drifted")
    hits = total = complete = coverage = 0
    item_hits: list[int] = []
    for item, ranking in zip(items, rankings):
        supports = frozenset(item.support_indices)
        row_hits = len(supports.intersection(ranking))
        item_hits.append(row_hits)
        hits += row_hits
        total += len(supports)
        complete += int(row_hits == len(supports))
        coverage += int(row_hits > 0)
    return {
        "arm_id": arm_id,
        "item_count": len(items),
        "support_hit_count": hits,
        "support_total": total,
        "support_recall_at_5_numerator": Fraction(hits, total).numerator,
        "support_recall_at_5_denominator": Fraction(hits, total).denominator,
        "complete_item_count": complete,
        "coverage_item_count": coverage,
        "item_hit_vector_sha256": stable_hash(item_hits),
    }


def _paired_arm(
    *,
    left: str,
    right: str,
    items: Sequence[StudyItem],
    arms: Mapping[str, Sequence[Sequence[int]]],
) -> dict[str, Any]:
    deltas = [
        len(frozenset(item.support_indices).intersection(arms[left][ordinal]))
        - len(frozenset(item.support_indices).intersection(arms[right][ordinal]))
        for ordinal, item in enumerate(items)
    ]
    return {
        "left_arm_id": left,
        "right_arm_id": right,
        "net_support_hit_count": sum(deltas),
        "gain_item_count": sum(value > 0 for value in deltas),
        "harm_item_count": sum(value < 0 for value in deltas),
        "tie_item_count": sum(value == 0 for value in deltas),
        "paired_delta_vector_sha256": stable_hash(deltas),
        "paired_test": exact_paired_sign_flip(deltas),
    }


def _l5_achievement(primary: Mapping[str, Any]) -> bool:
    paired_test = primary.get("paired_test")
    return bool(
        isinstance(paired_test, Mapping)
        and primary.get("net_support_hit_count", 0) > 0
        and paired_test.get("promoted") is True
    )


def _archive_transition(
    *,
    anchor_manifest_sha256: str,
    incumbent_hits: int,
    challenger_hits: int,
    support_total: int,
    item_count: int,
    promoted: bool,
) -> dict[str, Any]:
    anchor = _require_sha256(anchor_manifest_sha256, "anchor manifest")
    specs = {
        policy_id: EvaluatorSpec(
            id=policy_id,
            version=VERSION,
            implementation_hash=stable_hash({"module": VERSION}),
            criteria_hash=stable_hash({"policy": policy_id}),
            anchor_manifest_hash=anchor,
        )
        for policy_id in (INCUMBENT_POLICY_ID, CHALLENGER_POLICY_ID)
    }
    incumbent_epoch = EvaluatorEpoch(
        id=f"eval_epoch_0_{stable_hash(asdict(specs[INCUMBENT_POLICY_ID]))[:10]}",
        index=0,
        evaluator=specs[INCUMBENT_POLICY_ID],
    )
    archive = PolicyArchive()
    node = archive.create_node(
        active_hypothesis_ids=(), evaluator_epoch_id=incumbent_epoch.id,
        runtime_version=VERSION,
    )
    safe_items = tuple(stable_hash({"anchor": anchor, "ordinal": value}) for value in range(item_count))
    dependent = archive.record_score(
        archive_node_id=node.id, split="A_hold",
        evaluator_epoch_id=incumbent_epoch.id,
        metric="evaluator_dependent_portfolio_support", successes=incumbent_hits,
        total=support_total, item_ids=safe_items,
    )
    independent = archive.record_score(
        archive_node_id=node.id, split="A_hold",
        evaluator_epoch_id="fixed_source_support_v1",
        metric="independent_source_support", successes=challenger_hits,
        total=support_total, item_ids=safe_items,
    )
    if promoted:
        invalidated = archive.invalidate_evaluator_epoch(incumbent_epoch.id)
        next_epoch = EvaluatorEpoch(
            id=f"eval_epoch_1_{stable_hash(asdict(specs[CHALLENGER_POLICY_ID]))[:10]}",
            index=1, evaluator=specs[CHALLENGER_POLICY_ID],
            parent_epoch_id=incumbent_epoch.id,
        )
    else:
        invalidated = ()
        next_epoch = incumbent_epoch
    payload = {
        "promoted": promoted,
        "incumbent_epoch_id": incumbent_epoch.id,
        "next_epoch_id": next_epoch.id,
        "next_evaluator_id": next_epoch.evaluator.id,
        "parent_epoch_id": next_epoch.parent_epoch_id,
        "invalidated_score_record_ids": list(invalidated),
        "dependent_score_record_id": dependent.id,
        "dependent_score_valid_after_transition": archive.score_records[dependent.id].valid,
        "independent_source_score_record_id": independent.id,
        "independent_source_score_valid_after_transition": archive.score_records[independent.id].valid,
        "selective_invalidation_performed": promoted,
        "independent_source_record_retained": True,
        "archive_snapshot_sha256": stable_hash(archive.to_dict()),
    }
    return {**payload, "transition_sha256": stable_hash(payload)}


def build_a_hold_pre_run_freeze(
    *,
    project_root: str | Path,
    acquisition_receipt_path: str | Path,
    selection_secret_path: str | Path,
    p_formation_receipt_path: str | Path,
    p_frozen_program_path: str | Path,
    m1_freeze_path: str | Path,
    m1_report_path: str | Path,
    a_form_private_cache_path: str | Path,
    a_form_public_receipt_path: str | Path,
    f_search_private_cache_path: str | Path,
    f_search_public_receipt_path: str | Path,
    execution_root: str | Path,
    authorization_hash: str,
    output_path: str | Path,
) -> dict[str, Any]:
    project = Path(project_root).resolve(strict=True)
    (
        receipt, raw, commitments, p_program, lineage,
        a_public, f_public, a_binding, f_binding,
    ) = _artifact_bundles(
        project=project,
        acquisition_receipt_path=acquisition_receipt_path,
        selection_secret_path=selection_secret_path,
        p_formation_receipt_path=p_formation_receipt_path,
        p_frozen_program_path=p_frozen_program_path,
        m1_freeze_path=m1_freeze_path,
        m1_report_path=m1_report_path,
        a_form_private_cache_path=a_form_private_cache_path,
        a_form_public_receipt_path=a_form_public_receipt_path,
        f_search_private_cache_path=f_search_private_cache_path,
        f_search_public_receipt_path=f_search_public_receipt_path,
    )
    a_core = a_public["formation_core"]
    f_core = f_public["formation_core"]
    if a_core.get("measurable_contrast") is not True or f_core.get("measurable_contrast") is not True:
        raise MuSiQueEvaluatorPortfolioError("formation action coincides; A_hold must remain unopened")
    for core in (a_core, f_core):
        selected = {*core["incumbent"]["program_sha256s"], *core["challenger"]["program_sha256s"]}
        if p_program.program_hash in selected:
            raise MuSiQueEvaluatorPortfolioError("retained P entered Q portfolio")
    body = {
        "schema": ANCHOR_FREEZE_SCHEMA,
        "decision": "authorize_exact_A_hold_once_after_two_distinct_portfolios",
        "implementation": current_implementation_binding(project),
        "source_binding": _source_binding(receipt, raw, commitments[ANCHOR_BLOCK]),
        "portfolio_design_binding": receipt["portfolio_design_binding"],
        "lineage_binding": lineage,
        "A_form_binding": a_binding,
        "F_search_binding": f_binding,
        "anchor_actions": {
            "retained_P_program_sha256": p_program.program_hash,
            "incumbent": a_core["incumbent"],
            "challenger": a_core["challenger"],
            "equal_three_call_portfolio_grammar": True,
            "behavior_distinct": True,
        },
        "prospective_search_actions": {
            "incumbent": f_core["incumbent"],
            "challenger": f_core["challenger"],
            "measurable_contrast": True,
            "frozen_before_A_hold_open": True,
        },
        "candidate_set_binding": candidate_set_binding(),
        "execution_contract": _anchor_execution_contract(),
        "authorization_hash": _require_sha256(authorization_hash, "A_hold authorization"),
        "execution_root_sha256": _root_hash(execution_root),
        "ordering": {
            "A_hold_rows_read_while_freezing": 0,
            "A_hold_labels_read_while_freezing": 0,
            "M_search_rows_read_while_freezing": 0,
            "freeze_complete_before_A_hold_open": True,
        },
        "raw_content_persisted": False,
    }
    freeze = {**body, "freeze_sha256": stable_hash(body)}
    _assert_public_safe(freeze)
    _write_json_exclusive(output_path, freeze, mode=0o644)
    return freeze


def _load_anchor_freeze(
    path: str | Path, *, project: Path
) -> tuple[dict[str, Any], str]:
    freeze, raw = _read_json(path, "A_hold pre-run freeze")
    body = dict(freeze)
    declared = _require_sha256(body.pop("freeze_sha256", None), "A_hold freeze")
    if (
        freeze.get("schema") != ANCHOR_FREEZE_SCHEMA
        or stable_hash(body) != declared
        or freeze.get("implementation") != current_implementation_binding(project)
        or freeze.get("candidate_set_binding") != candidate_set_binding()
        or freeze.get("execution_contract") != _anchor_execution_contract()
        or freeze.get("raw_content_persisted") is not False
    ):
        raise MuSiQueEvaluatorPortfolioError("A_hold freeze drifted")
    _assert_public_safe(freeze)
    return freeze, _sha256_bytes(raw)


def _verify_anchor_inputs(
    *,
    project: Path,
    freeze: Mapping[str, Any],
    acquisition_receipt_path: str | Path,
    selection_secret_path: str | Path,
    p_formation_receipt_path: str | Path,
    p_frozen_program_path: str | Path,
    m1_freeze_path: str | Path,
    m1_report_path: str | Path,
    a_form_private_cache_path: str | Path,
    a_form_public_receipt_path: str | Path,
    f_search_private_cache_path: str | Path,
    f_search_public_receipt_path: str | Path,
) -> tuple[
    dict[str, Any], dict[str, BlockCommitment], TypedRetrievalProgram,
    tuple[TypedRetrievalProgram, ...], tuple[TypedRetrievalProgram, ...],
]:
    (
        receipt, raw, commitments, p_program, lineage,
        a_public, f_public, a_binding, f_binding,
    ) = _artifact_bundles(
        project=project,
        acquisition_receipt_path=acquisition_receipt_path,
        selection_secret_path=selection_secret_path,
        p_formation_receipt_path=p_formation_receipt_path,
        p_frozen_program_path=p_frozen_program_path,
        m1_freeze_path=m1_freeze_path,
        m1_report_path=m1_report_path,
        a_form_private_cache_path=a_form_private_cache_path,
        a_form_public_receipt_path=a_form_public_receipt_path,
        f_search_private_cache_path=f_search_private_cache_path,
        f_search_public_receipt_path=f_search_public_receipt_path,
    )
    a_core = a_public["formation_core"]
    f_core = f_public["formation_core"]
    expected_anchor = {
        "retained_P_program_sha256": p_program.program_hash,
        "incumbent": a_core["incumbent"],
        "challenger": a_core["challenger"],
        "equal_three_call_portfolio_grammar": True,
        "behavior_distinct": True,
    }
    expected_search = {
        "incumbent": f_core["incumbent"],
        "challenger": f_core["challenger"],
        "measurable_contrast": True,
        "frozen_before_A_hold_open": True,
    }
    if (
        freeze.get("source_binding") != _source_binding(receipt, raw, commitments[ANCHOR_BLOCK])
        or freeze.get("portfolio_design_binding") != receipt.get("portfolio_design_binding")
        or freeze.get("lineage_binding") != lineage
        or freeze.get("A_form_binding") != a_binding
        or freeze.get("F_search_binding") != f_binding
        or freeze.get("anchor_actions") != expected_anchor
        or freeze.get("prospective_search_actions") != expected_search
    ):
        raise MuSiQueEvaluatorPortfolioError("A_hold source or action binding drifted")
    return (
        receipt,
        commitments,
        p_program,
        _program_pair(a_core, "incumbent"),
        _program_pair(a_core, "challenger"),
    )


def execute_a_hold_formal(
    *,
    project_root: str | Path,
    pre_run_freeze_path: str | Path,
    acquisition_receipt_path: str | Path,
    selection_secret_path: str | Path,
    a_hold_block_path: str | Path,
    p_formation_receipt_path: str | Path,
    p_frozen_program_path: str | Path,
    m1_freeze_path: str | Path,
    m1_report_path: str | Path,
    a_form_private_cache_path: str | Path,
    a_form_public_receipt_path: str | Path,
    f_search_private_cache_path: str | Path,
    f_search_public_receipt_path: str | Path,
    execution_root: str | Path,
) -> dict[str, Any]:
    if _CLEAN_MODULE_CLI_ACTIVE is not True:
        raise MuSiQueEvaluatorPortfolioError(
            "formal A_hold is available only through clean module CLI"
        )
    project = Path(project_root).resolve(strict=True)
    freeze, freeze_file_hash = _load_anchor_freeze(pre_run_freeze_path, project=project)
    root = _new_root(execution_root)
    if freeze.get("execution_root_sha256") != _root_hash(root):
        raise MuSiQueEvaluatorPortfolioError("A_hold execution root drifted")
    if root.exists():
        raise MuSiQueEvaluatorPortfolioError("fresh A_hold root exists; replay forbidden")
    receipt, commitments, p_program, incumbent, challenger = _verify_anchor_inputs(
        project=project,
        freeze=freeze,
        acquisition_receipt_path=acquisition_receipt_path,
        selection_secret_path=selection_secret_path,
        p_formation_receipt_path=p_formation_receipt_path,
        p_frozen_program_path=p_frozen_program_path,
        m1_freeze_path=m1_freeze_path,
        m1_report_path=m1_report_path,
        a_form_private_cache_path=a_form_private_cache_path,
        a_form_public_receipt_path=a_form_public_receipt_path,
        f_search_private_cache_path=f_search_private_cache_path,
        f_search_public_receipt_path=f_search_public_receipt_path,
    )
    _require_private_boundary(root)
    _prepare_output_parent(root)
    os.mkdir(root, 0o700)
    attempted = completed = 0
    lock = threading.Lock()
    barrier = threading.Barrier(ANCHOR_WORK_UNIT_COUNT)
    stage = "authorization_consumption"
    try:
        consumption_body = {
            "schema": CONSUMPTION_SCHEMA,
            "stage": "A_hold",
            "authorization_hash": freeze["authorization_hash"],
            "freeze_sha256": freeze["freeze_sha256"],
            "freeze_file_sha256": freeze_file_hash,
            "execution_root_sha256": freeze["execution_root_sha256"],
            "replay_authorized": False,
            "raw_content_persisted": False,
        }
        consumption = {**consumption_body, "consumption_sha256": stable_hash(consumption_body)}
        _write_json_exclusive(root / ANCHOR_CONSUMPTION_FILENAME, consumption)
        stage = "exact_A_hold_open_after_consumption"
        items = _load_block(
            path=a_hold_block_path,
            expected_block=ANCHOR_BLOCK,
            commitment=commitments[ANCHOR_BLOCK],
        )
        if len(items) != A_HOLD_ITEM_COUNT:
            raise MuSiQueEvaluatorPortfolioError("A_hold item count drifted")
        programs = {
            "incumbent_P": p_program,
            "incumbent_Q1": incumbent[0],
            "incumbent_Q2": incumbent[1],
            "challenger_P": p_program,
            "challenger_Q1": challenger[0],
            "challenger_Q2": challenger[1],
        }
        work_units = tuple(
            (ordinal, component, item.view)
            for ordinal, item in enumerate(items)
            for component in ANCHOR_COMPONENT_IDS
        )
        stage = "maximum_width_six_component_retrieval"

        def run_one(
            unit: tuple[int, str, RetrievalItem]
        ) -> tuple[tuple[int, str], tuple[int, ...]]:
            nonlocal attempted, completed
            ordinal, component, item = unit
            with lock:
                attempted += 1
            try:
                barrier.wait(timeout=180)
            except threading.BrokenBarrierError as exc:
                raise MuSiQueEvaluatorPortfolioError("A_hold barrier did not close") from exc
            ranking = _ranking(programs[component], item)
            with lock:
                completed += 1
            return (ordinal, component), ranking

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=ANCHOR_MAXIMUM_CONCURRENCY,
            thread_name_prefix="musique-portfolio-anchor",
        ) as executor:
            terminals = _submit_all_then_join(
                executor=executor, function=run_one, work_units=work_units,
            )
        direct = dict(terminals)
        if attempted != ANCHOR_WORK_UNIT_COUNT or completed != ANCHOR_WORK_UNIT_COUNT or len(direct) != ANCHOR_WORK_UNIT_COUNT:
            raise MuSiQueEvaluatorPortfolioError("A_hold terminal closure incomplete")
        stage = "offline_support_scoring_after_join"
        arms: dict[str, list[tuple[int, ...]]] = {
            "incumbent_portfolio": [], "challenger_portfolio": []
        }
        for ordinal in range(len(items)):
            arms["incumbent_portfolio"].append(fuse_rankings(
                direct[(ordinal, "incumbent_P")],
                direct[(ordinal, "incumbent_Q1")],
                direct[(ordinal, "incumbent_Q2")],
            ))
            arms["challenger_portfolio"].append(fuse_rankings(
                direct[(ordinal, "challenger_P")],
                direct[(ordinal, "challenger_Q1")],
                direct[(ordinal, "challenger_Q2")],
            ))
        metrics = {
            arm: _aggregate_arm(arm_id=arm, items=items, rankings=rankings)
            for arm, rankings in arms.items()
        }
        paired = _paired_arm(
            left="challenger_portfolio", right="incumbent_portfolio",
            items=items, arms=arms,
        )
        transition = _archive_transition(
            anchor_manifest_sha256=commitments[ANCHOR_BLOCK].item_commitment_set_sha256,
            incumbent_hits=metrics["incumbent_portfolio"]["support_hit_count"],
            challenger_hits=metrics["challenger_portfolio"]["support_hit_count"],
            support_total=metrics["incumbent_portfolio"]["support_total"],
            item_count=len(items), promoted=paired["paired_test"]["promoted"],
        )
        private_body = {
            "schema": f"{VERSION}_A_hold_private_evidence",
            "freeze_sha256": freeze["freeze_sha256"],
            "source_binding": freeze["source_binding"],
            "item_rows": [
                {
                    "item_commitment_sha256": item.item_commitment_sha256,
                    "support_indices": list(item.support_indices),
                    "incumbent_portfolio_ranking": list(arms["incumbent_portfolio"][ordinal]),
                    "challenger_portfolio_ranking": list(arms["challenger_portfolio"][ordinal]),
                }
                for ordinal, item in enumerate(items)
            ],
            "raw_question_or_corpus_persisted": False,
        }
        private = {**private_body, "evidence_sha256": stable_hash(private_body)}
        private_path = root / ANCHOR_PRIVATE_EVIDENCE_FILENAME
        _write_json_exclusive(private_path, private)
        report_body = {
            "schema": ANCHOR_REPORT_SCHEMA,
            "valid": True,
            "freeze_sha256": freeze["freeze_sha256"],
            "freeze_file_sha256": freeze_file_hash,
            "source_binding": freeze["source_binding"],
            "portfolio_design_binding": freeze["portfolio_design_binding"],
            "A_form_binding": freeze["A_form_binding"],
            "F_search_binding": freeze["F_search_binding"],
            "anchor_actions": freeze["anchor_actions"],
            "prospective_search_actions": freeze["prospective_search_actions"],
            "arm_metrics": metrics,
            "challenger_minus_incumbent": paired,
            "evaluator_epoch_transition": transition,
            "private_evidence_binding": {
                "file_sha256": _sha256_file(private_path),
                "evidence_sha256": private["evidence_sha256"],
                "private_path_persisted_publicly": False,
                "item_level_evidence_persisted_publicly": False,
            },
            "execution": {
                "physical_component_ids": list(ANCHOR_COMPONENT_IDS),
                "item_count": len(items),
                "physical_work_unit_count": ANCHOR_WORK_UNIT_COUNT,
                "retrieval_attempt_count": attempted,
                "retrieval_terminal_count": completed,
                "configured_maximum_concurrency": ANCHOR_MAXIMUM_CONCURRENCY,
                "observed_start_barrier_party_count": barrier.parties,
                "all_work_units_released_from_single_start_barrier": True,
                "all_terminals_joined_before_support_scoring": True,
                "variable_support_count_supported": True,
                "external_network_calls": 0,
                "online_evaluator_calls": 0,
                "retries": 0, "replays": 0, "resamples": 0,
            },
            "M_search_opened": False,
            "M_search_authorized": transition["promoted"],
            "raw_content_persisted": False,
        }
        report = {**report_body, "report_sha256": stable_hash(report_body)}
        _assert_public_safe(report)
        _write_json_exclusive(root / ANCHOR_REPORT_FILENAME, report)
        return report
    except Exception as exc:
        failure_body = {
            "schema": FAILURE_SCHEMA, "stage": "A_hold", "valid": False,
            "freeze_sha256": freeze["freeze_sha256"], "failure_stage": stage,
            "error_type_sha256": stable_hash({"error_type": type(exc).__name__}),
            "authorization_consumed": (root / ANCHOR_CONSUMPTION_FILENAME).is_file(),
            "physical_work_unit_count": ANCHOR_WORK_UNIT_COUNT,
            "retrieval_attempt_count": attempted, "retrieval_terminal_count": completed,
            "retries": 0, "replays": 0, "resamples": 0,
            "replay_authorized": False, "raw_content_persisted": False,
        }
        try:
            _write_json_exclusive(
                root / ANCHOR_FAILURE_FILENAME,
                {**failure_body, "failure_sha256": stable_hash(failure_body)},
            )
        except Exception:
            pass
        raise MuSiQueEvaluatorPortfolioError("formal A_hold failed and cannot be replayed") from exc


def load_and_reverify_a_hold(
    *,
    project: Path,
    pre_run_freeze_path: str | Path,
    private_evidence_path: str | Path,
    report_path: str | Path,
    acquisition_receipt_path: str | Path,
    selection_secret_path: str | Path,
    p_formation_receipt_path: str | Path,
    p_frozen_program_path: str | Path,
    m1_freeze_path: str | Path,
    m1_report_path: str | Path,
    a_form_private_cache_path: str | Path,
    a_form_public_receipt_path: str | Path,
    f_search_private_cache_path: str | Path,
    f_search_public_receipt_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    freeze, freeze_file_hash = _load_anchor_freeze(pre_run_freeze_path, project=project)
    _receipt, commitments, _p, _inc, _chall = _verify_anchor_inputs(
        project=project,
        freeze=freeze,
        acquisition_receipt_path=acquisition_receipt_path,
        selection_secret_path=selection_secret_path,
        p_formation_receipt_path=p_formation_receipt_path,
        p_frozen_program_path=p_frozen_program_path,
        m1_freeze_path=m1_freeze_path,
        m1_report_path=m1_report_path,
        a_form_private_cache_path=a_form_private_cache_path,
        a_form_public_receipt_path=a_form_public_receipt_path,
        f_search_private_cache_path=f_search_private_cache_path,
        f_search_public_receipt_path=f_search_public_receipt_path,
    )
    private, private_raw = _read_json(private_evidence_path, "A_hold private evidence")
    private_body = dict(private)
    evidence_hash = _require_sha256(private_body.pop("evidence_sha256", None), "A_hold evidence")
    rows = private.get("item_rows")
    if (
        private.get("schema") != f"{VERSION}_A_hold_private_evidence"
        or private.get("freeze_sha256") != freeze["freeze_sha256"]
        or private.get("source_binding") != freeze["source_binding"]
        or private.get("raw_question_or_corpus_persisted") is not False
        or stable_hash(private_body) != evidence_hash
        or not isinstance(rows, list)
        or len(rows) != A_HOLD_ITEM_COUNT
    ):
        raise MuSiQueEvaluatorPortfolioError("A_hold private evidence drifted")
    items: list[StudyItem] = []
    arms: dict[str, list[tuple[int, ...]]] = {
        "incumbent_portfolio": [], "challenger_portfolio": []
    }
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "item_commitment_sha256", "support_indices",
            "incumbent_portfolio_ranking", "challenger_portfolio_ranking",
        }:
            raise MuSiQueEvaluatorPortfolioError("A_hold evidence row drifted")
        commitment = _require_sha256(row["item_commitment_sha256"], "A_hold item")
        supports = tuple(row["support_indices"])
        if not supports or any(type(value) is not int or value < 0 for value in supports) or len(set(supports)) != len(supports):
            raise MuSiQueEvaluatorPortfolioError("A_hold support vector drifted")
        dummy_corpus = tuple(
            RetrievalParagraph(idx=value, title="private", text="private")
            for value in range(max((*supports, TOP_K - 1)) + 1)
        )
        items.append(StudyItem(
            view=RetrievalItem(
                question="private", corpus=dummy_corpus,
                item_commitment_sha256=commitment,
            ),
            support_indices=supports,
        ))
        for arm, key in (
            ("incumbent_portfolio", "incumbent_portfolio_ranking"),
            ("challenger_portfolio", "challenger_portfolio_ranking"),
        ):
            ranking = tuple(row[key])
            if len(ranking) != TOP_K or len(set(ranking)) != TOP_K or any(type(value) is not int or value < 0 for value in ranking):
                raise MuSiQueEvaluatorPortfolioError("A_hold cached ranking drifted")
            arms[arm].append(ranking)
    if stable_hash([row.item_commitment_sha256 for row in items]) != commitments[ANCHOR_BLOCK].item_commitment_set_sha256:
        # Acquisition commits hashes of full rows, which are exactly the cached item commitments.
        raise MuSiQueEvaluatorPortfolioError("A_hold cached item commitment set drifted")
    metrics = {
        arm: _aggregate_arm(arm_id=arm, items=items, rankings=rankings)
        for arm, rankings in arms.items()
    }
    paired = _paired_arm(
        left="challenger_portfolio", right="incumbent_portfolio", items=items, arms=arms
    )
    transition = _archive_transition(
        anchor_manifest_sha256=commitments[ANCHOR_BLOCK].item_commitment_set_sha256,
        incumbent_hits=metrics["incumbent_portfolio"]["support_hit_count"],
        challenger_hits=metrics["challenger_portfolio"]["support_hit_count"],
        support_total=metrics["incumbent_portfolio"]["support_total"],
        item_count=len(items), promoted=paired["paired_test"]["promoted"],
    )
    report, report_raw = _read_json(report_path, "A_hold report")
    report_body = dict(report)
    report_hash = _require_sha256(report_body.pop("report_sha256", None), "A_hold report")
    evidence_binding = report.get("private_evidence_binding")
    execution = report.get("execution")
    if (
        report.get("schema") != ANCHOR_REPORT_SCHEMA
        or report.get("valid") is not True
        or stable_hash(report_body) != report_hash
        or report.get("freeze_sha256") != freeze["freeze_sha256"]
        or report.get("freeze_file_sha256") != freeze_file_hash
        or report.get("source_binding") != freeze["source_binding"]
        or report.get("portfolio_design_binding") != freeze["portfolio_design_binding"]
        or report.get("A_form_binding") != freeze["A_form_binding"]
        or report.get("F_search_binding") != freeze["F_search_binding"]
        or report.get("anchor_actions") != freeze["anchor_actions"]
        or report.get("prospective_search_actions") != freeze["prospective_search_actions"]
        or report.get("arm_metrics") != metrics
        or report.get("challenger_minus_incumbent") != paired
        or report.get("evaluator_epoch_transition") != transition
        or not isinstance(evidence_binding, Mapping)
        or evidence_binding.get("file_sha256") != _sha256_bytes(private_raw)
        or evidence_binding.get("evidence_sha256") != evidence_hash
        or not isinstance(execution, Mapping)
        or execution.get("physical_component_ids") != list(ANCHOR_COMPONENT_IDS)
        or execution.get("item_count") != A_HOLD_ITEM_COUNT
        or execution.get("physical_work_unit_count") != ANCHOR_WORK_UNIT_COUNT
        or execution.get("retrieval_attempt_count") != ANCHOR_WORK_UNIT_COUNT
        or execution.get("retrieval_terminal_count") != ANCHOR_WORK_UNIT_COUNT
        or execution.get("configured_maximum_concurrency") != ANCHOR_MAXIMUM_CONCURRENCY
        or execution.get("observed_start_barrier_party_count") != ANCHOR_WORK_UNIT_COUNT
        or execution.get("all_work_units_released_from_single_start_barrier") is not True
        or execution.get("all_terminals_joined_before_support_scoring") is not True
        or execution.get("variable_support_count_supported") is not True
        or any(execution.get(key) != 0 for key in (
            "external_network_calls", "online_evaluator_calls", "retries", "replays", "resamples"
        ))
        or report.get("M_search_authorized") is not transition["promoted"]
        or report.get("raw_content_persisted") is not False
    ):
        raise MuSiQueEvaluatorPortfolioError("A_hold report differs from exact evidence")
    _assert_public_safe(report)
    binding = {
        "pre_run_freeze_file_sha256": freeze_file_hash,
        "pre_run_freeze_sha256": freeze["freeze_sha256"],
        "private_evidence_file_sha256": _sha256_bytes(private_raw),
        "private_evidence_sha256": evidence_hash,
        "public_report_file_sha256": _sha256_bytes(report_raw),
        "public_report_sha256": report_hash,
        "anchor_transition_sha256": transition["transition_sha256"],
        "challenger_promoted": transition["promoted"],
        "recomputed_from_exact_private_evidence": True,
        "private_path_persisted_publicly": False,
    }
    return report, binding


def _promoted_transition_binding(transition: Mapping[str, Any]) -> dict[str, Any]:
    if (
        transition.get("promoted") is not True
        or transition.get("selective_invalidation_performed") is not True
        or transition.get("independent_source_record_retained") is not True
    ):
        raise MuSiQueEvaluatorPortfolioError(
            "challenger was not promoted; M_search must remain unopened"
        )
    for field in ("incumbent_epoch_id", "next_epoch_id", "next_evaluator_id"):
        if not isinstance(transition.get(field), str) or not transition[field]:
            raise MuSiQueEvaluatorPortfolioError("promoted transition identity drifted")
    return {
        "transition_sha256": _require_sha256(transition.get("transition_sha256"), "transition"),
        "incumbent_epoch_id": transition["incumbent_epoch_id"],
        "active_epoch_id": transition["next_epoch_id"],
        "active_evaluator_id": transition["next_evaluator_id"],
        "promoted": True,
    }


def _prepare_runtime(
    *,
    project: Path,
    runtime_python: str | Path,
    local_llm_model: str | Path,
    local_embedding_model: str | Path,
    base_binding_receipt_path: str | Path,
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


def build_m_search_pre_run_freeze(
    *,
    project_root: str | Path,
    acquisition_receipt_path: str | Path,
    selection_secret_path: str | Path,
    p_formation_receipt_path: str | Path,
    p_frozen_program_path: str | Path,
    m1_freeze_path: str | Path,
    m1_report_path: str | Path,
    a_form_private_cache_path: str | Path,
    a_form_public_receipt_path: str | Path,
    f_search_private_cache_path: str | Path,
    f_search_public_receipt_path: str | Path,
    a_hold_pre_run_freeze_path: str | Path,
    a_hold_private_evidence_path: str | Path,
    a_hold_report_path: str | Path,
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
    anchor_report, anchor_binding = load_and_reverify_a_hold(
        project=project,
        pre_run_freeze_path=a_hold_pre_run_freeze_path,
        private_evidence_path=a_hold_private_evidence_path,
        report_path=a_hold_report_path,
        acquisition_receipt_path=acquisition_receipt_path,
        selection_secret_path=selection_secret_path,
        p_formation_receipt_path=p_formation_receipt_path,
        p_frozen_program_path=p_frozen_program_path,
        m1_freeze_path=m1_freeze_path,
        m1_report_path=m1_report_path,
        a_form_private_cache_path=a_form_private_cache_path,
        a_form_public_receipt_path=a_form_public_receipt_path,
        f_search_private_cache_path=f_search_private_cache_path,
        f_search_public_receipt_path=f_search_public_receipt_path,
    )
    transition_binding = _promoted_transition_binding(anchor_report["evaluator_epoch_transition"])
    (
        receipt, raw, commitments, p_program, lineage,
        _a_public, f_public, a_binding, f_binding,
    ) = _artifact_bundles(
        project=project,
        acquisition_receipt_path=acquisition_receipt_path,
        selection_secret_path=selection_secret_path,
        p_formation_receipt_path=p_formation_receipt_path,
        p_frozen_program_path=p_frozen_program_path,
        m1_freeze_path=m1_freeze_path,
        m1_report_path=m1_report_path,
        a_form_private_cache_path=a_form_private_cache_path,
        a_form_public_receipt_path=a_form_public_receipt_path,
        f_search_private_cache_path=f_search_private_cache_path,
        f_search_public_receipt_path=f_search_public_receipt_path,
    )
    f_core = f_public["formation_core"]
    if f_core.get("measurable_contrast") is not True:
        raise MuSiQueEvaluatorPortfolioError("F_search contrast unavailable")
    capability, capability_raw = verify_capability_receipt(capability_receipt_path)
    prepared = _prepare_runtime(
        project=project, runtime_python=runtime_python,
        local_llm_model=local_llm_model, local_embedding_model=local_embedding_model,
        base_binding_receipt_path=base_binding_receipt_path,
        attestation_receipt_path=attestation_receipt_path,
    )
    body = {
        "schema": SEARCH_FREEZE_SCHEMA,
        "decision": "authorize_exact_promoted_portfolio_M_search_once",
        "implementation": current_implementation_binding(project),
        "source_binding": _source_binding(receipt, raw, commitments[SEARCH_BLOCK]),
        "portfolio_design_binding": receipt["portfolio_design_binding"],
        "lineage_binding": lineage,
        "A_form_binding": a_binding,
        "F_search_binding": f_binding,
        "A_hold_binding": anchor_binding,
        "evaluator_epoch_transition": transition_binding,
        "search_actions": {
            "retained_P_program_sha256": p_program.program_hash,
            "incumbent": f_core["incumbent"],
            "active": f_core["challenger"],
            "equal_three_call_portfolio_grammar": True,
            "incumbent_and_active_behavior_distinct": True,
            "frozen_on_F_search_before_A_hold": True,
        },
        "candidate_set_binding": candidate_set_binding(),
        "capability_binding": {
            "file_sha256": _sha256_bytes(capability_raw),
            "receipt_sha256": capability["receipt_sha256"],
            "bwrap_file_sha256": capability["bwrap_file_sha256"],
            "probe_contract_sha256": capability["probe_contract_sha256"],
            "fresh_probe_required_before_authorization": True,
        },
        "runtime_binding": prepared.safe_binding,
        "execution_contract": _search_execution_contract(),
        "authorization_hash": _require_sha256(authorization_hash, "M_search authorization"),
        "execution_root_sha256": _root_hash(execution_root),
        "ordering": {
            "M_search_rows_read_while_freezing": 0,
            "M_search_labels_read_while_freezing": 0,
            "A_hold_transition_reverified_before_freeze": True,
            "freeze_complete_before_M_search_open": True,
        },
        "raw_content_persisted": False,
    }
    freeze = {**body, "freeze_sha256": stable_hash(body)}
    _assert_public_safe(freeze)
    _write_json_exclusive(output_path, freeze, mode=0o644)
    return freeze


def _load_search_freeze(
    path: str | Path, *, project: Path
) -> tuple[dict[str, Any], str]:
    freeze, raw = _read_json(path, "M_search pre-run freeze")
    body = dict(freeze)
    declared = _require_sha256(body.pop("freeze_sha256", None), "M_search freeze")
    if (
        freeze.get("schema") != SEARCH_FREEZE_SCHEMA
        or stable_hash(body) != declared
        or freeze.get("implementation") != current_implementation_binding(project)
        or freeze.get("candidate_set_binding") != candidate_set_binding()
        or freeze.get("execution_contract") != _search_execution_contract()
        or freeze.get("raw_content_persisted") is not False
    ):
        raise MuSiQueEvaluatorPortfolioError("M_search freeze drifted")
    _assert_public_safe(freeze)
    return freeze, _sha256_bytes(raw)


def execute_m_search_formal(
    *,
    project_root: str | Path,
    pre_run_freeze_path: str | Path,
    acquisition_receipt_path: str | Path,
    selection_secret_path: str | Path,
    m_search_block_path: str | Path,
    p_formation_receipt_path: str | Path,
    p_frozen_program_path: str | Path,
    m1_freeze_path: str | Path,
    m1_report_path: str | Path,
    a_form_private_cache_path: str | Path,
    a_form_public_receipt_path: str | Path,
    f_search_private_cache_path: str | Path,
    f_search_public_receipt_path: str | Path,
    a_hold_pre_run_freeze_path: str | Path,
    a_hold_private_evidence_path: str | Path,
    a_hold_report_path: str | Path,
    capability_receipt_path: str | Path,
    runtime_python: str | Path,
    local_llm_model: str | Path,
    local_embedding_model: str | Path,
    base_binding_receipt_path: str | Path,
    attestation_receipt_path: str | Path,
    execution_root: str | Path,
) -> dict[str, Any]:
    if _CLEAN_MODULE_CLI_ACTIVE is not True:
        raise MuSiQueEvaluatorPortfolioError(
            "formal M_search is available only through clean module CLI"
        )
    project = Path(project_root).resolve(strict=True)
    freeze, freeze_file_hash = _load_search_freeze(pre_run_freeze_path, project=project)
    root = _new_root(execution_root)
    if freeze.get("execution_root_sha256") != _root_hash(root):
        raise MuSiQueEvaluatorPortfolioError("M_search execution root drifted")
    if root.exists():
        raise MuSiQueEvaluatorPortfolioError("fresh M_search root exists; replay forbidden")
    anchor_report, anchor_binding = load_and_reverify_a_hold(
        project=project,
        pre_run_freeze_path=a_hold_pre_run_freeze_path,
        private_evidence_path=a_hold_private_evidence_path,
        report_path=a_hold_report_path,
        acquisition_receipt_path=acquisition_receipt_path,
        selection_secret_path=selection_secret_path,
        p_formation_receipt_path=p_formation_receipt_path,
        p_frozen_program_path=p_frozen_program_path,
        m1_freeze_path=m1_freeze_path,
        m1_report_path=m1_report_path,
        a_form_private_cache_path=a_form_private_cache_path,
        a_form_public_receipt_path=a_form_public_receipt_path,
        f_search_private_cache_path=f_search_private_cache_path,
        f_search_public_receipt_path=f_search_public_receipt_path,
    )
    transition_binding = _promoted_transition_binding(anchor_report["evaluator_epoch_transition"])
    if freeze.get("A_hold_binding") != anchor_binding or freeze.get("evaluator_epoch_transition") != transition_binding:
        raise MuSiQueEvaluatorPortfolioError("promoted anchor binding drifted")
    (
        receipt, raw, commitments, p_program, lineage,
        _a_public, f_public, a_binding, f_binding,
    ) = _artifact_bundles(
        project=project,
        acquisition_receipt_path=acquisition_receipt_path,
        selection_secret_path=selection_secret_path,
        p_formation_receipt_path=p_formation_receipt_path,
        p_frozen_program_path=p_frozen_program_path,
        m1_freeze_path=m1_freeze_path,
        m1_report_path=m1_report_path,
        a_form_private_cache_path=a_form_private_cache_path,
        a_form_public_receipt_path=a_form_public_receipt_path,
        f_search_private_cache_path=f_search_private_cache_path,
        f_search_public_receipt_path=f_search_public_receipt_path,
    )
    f_core = f_public["formation_core"]
    expected_actions = {
        "retained_P_program_sha256": p_program.program_hash,
        "incumbent": f_core["incumbent"],
        "active": f_core["challenger"],
        "equal_three_call_portfolio_grammar": True,
        "incumbent_and_active_behavior_distinct": True,
        "frozen_on_F_search_before_A_hold": True,
    }
    if (
        freeze.get("source_binding") != _source_binding(receipt, raw, commitments[SEARCH_BLOCK])
        or freeze.get("portfolio_design_binding") != receipt.get("portfolio_design_binding")
        or freeze.get("lineage_binding") != lineage
        or freeze.get("A_form_binding") != a_binding
        or freeze.get("F_search_binding") != f_binding
        or freeze.get("search_actions") != expected_actions
    ):
        raise MuSiQueEvaluatorPortfolioError("M_search source or action binding drifted")
    incumbent = _program_pair(f_core, "incumbent")
    active = _program_pair(f_core, "challenger")
    capability, capability_raw = verify_capability_receipt(capability_receipt_path)
    expected_capability = {
        "file_sha256": _sha256_bytes(capability_raw),
        "receipt_sha256": capability["receipt_sha256"],
        "bwrap_file_sha256": capability["bwrap_file_sha256"],
        "probe_contract_sha256": capability["probe_contract_sha256"],
        "fresh_probe_required_before_authorization": True,
    }
    if freeze.get("capability_binding") != expected_capability:
        raise MuSiQueEvaluatorPortfolioError("M_search capability binding drifted")
    live_probe = _probe_bubblewrap()
    if (
        live_probe.get("bwrap_file_sha256") != capability["bwrap_file_sha256"]
        or live_probe.get("probe_contract_sha256") != capability["probe_contract_sha256"]
        or live_probe.get("probe_returncode") != 0
    ):
        raise MuSiQueEvaluatorPortfolioError("fresh bwrap preflight drifted")
    prepared = _prepare_runtime(
        project=project, runtime_python=runtime_python,
        local_llm_model=local_llm_model, local_embedding_model=local_embedding_model,
        base_binding_receipt_path=base_binding_receipt_path,
        attestation_receipt_path=attestation_receipt_path,
    )
    if freeze.get("runtime_binding") != prepared.safe_binding:
        raise MuSiQueEvaluatorPortfolioError("M_search runtime binding drifted")
    safe_runtime = prepared.safe_binding
    _require_private_boundary(root)
    _prepare_output_parent(root)
    os.mkdir(root, 0o700)
    attempted = completed = 0
    lock = threading.Lock()
    barrier = threading.Barrier(SEARCH_WORK_UNIT_COUNT)
    stage = "authorization_consumption"
    try:
        consumption_body = {
            "schema": CONSUMPTION_SCHEMA, "stage": "M_search",
            "authorization_hash": freeze["authorization_hash"],
            "freeze_sha256": freeze["freeze_sha256"],
            "freeze_file_sha256": freeze_file_hash,
            "execution_root_sha256": freeze["execution_root_sha256"],
            "replay_authorized": False, "raw_content_persisted": False,
        }
        _write_json_exclusive(
            root / SEARCH_CONSUMPTION_FILENAME,
            {**consumption_body, "consumption_sha256": stable_hash(consumption_body)},
        )
        stage = "exact_M_search_open_after_consumption"
        items = _load_block(
            path=m_search_block_path, expected_block=SEARCH_BLOCK,
            commitment=commitments[SEARCH_BLOCK],
        )
        if len(items) != M_SEARCH_ITEM_COUNT:
            raise MuSiQueEvaluatorPortfolioError("M_search item count drifted")
        work_units = tuple(
            (ordinal, component, item.view)
            for ordinal, item in enumerate(items)
            for component in SEARCH_COMPONENT_IDS
        )
        local_programs = {
            "incumbent_P": p_program, "incumbent_Q1": incumbent[0],
            "incumbent_Q2": incumbent[1], "active_P": p_program,
            "active_Q1": active[0], "active_Q2": active[1],
        }
        stage = "maximum_width_eight_component_retrieval"

        def run_one(
            unit: tuple[int, str, RetrievalItem]
        ) -> tuple[tuple[int, str], tuple[int, ...]]:
            nonlocal attempted, completed
            ordinal, component, item = unit
            with lock:
                attempted += 1
            try:
                barrier.wait(timeout=180)
            except threading.BrokenBarrierError as exc:
                raise MuSiQueEvaluatorPortfolioError("M_search barrier did not close") from exc
            if component == "canonical_RAW":
                ranking = tuple(row.idx for row in item.corpus[:TOP_K])
            elif component == "official_HippoRAG":
                ranking = tuple(prepared.retrieve(
                    question=item.question,
                    paragraphs=item.hipporag_paragraphs(),
                    work_root=root / f"official_item_{ordinal:02d}",
                ))
                if (
                    len(ranking) != TOP_K or len(set(ranking)) != TOP_K
                    or any(type(value) is not int or not 0 <= value < len(item.corpus) for value in ranking)
                ):
                    raise MuSiQueEvaluatorPortfolioError("official ranking drifted")
            else:
                ranking = _ranking(local_programs[component], item)
            with lock:
                completed += 1
            return (ordinal, component), ranking

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=SEARCH_MAXIMUM_CONCURRENCY,
            thread_name_prefix="musique-portfolio-search",
        ) as executor:
            terminals = _submit_all_then_join(
                executor=executor, function=run_one, work_units=work_units,
            )
        direct = dict(terminals)
        if attempted != SEARCH_WORK_UNIT_COUNT or completed != SEARCH_WORK_UNIT_COUNT or len(direct) != SEARCH_WORK_UNIT_COUNT:
            raise MuSiQueEvaluatorPortfolioError("M_search terminal closure incomplete")
        stage = "fresh_runtime_postflight_before_scoring"
        postflight = prepared.fresh_reverify()
        if postflight != safe_runtime:
            raise MuSiQueEvaluatorPortfolioError("M_search runtime postflight drifted")
        stage = "offline_support_scoring_after_join"
        arms: dict[str, list[tuple[int, ...]]] = {
            "canonical_RAW": [], "retained_P": [],
            "incumbent_portfolio": [], "active_portfolio": [],
            "official_HippoRAG": [],
        }
        for ordinal in range(len(items)):
            arms["canonical_RAW"].append(direct[(ordinal, "canonical_RAW")])
            arms["retained_P"].append(direct[(ordinal, "active_P")])
            arms["official_HippoRAG"].append(direct[(ordinal, "official_HippoRAG")])
            arms["incumbent_portfolio"].append(fuse_rankings(
                direct[(ordinal, "incumbent_P")], direct[(ordinal, "incumbent_Q1")],
                direct[(ordinal, "incumbent_Q2")],
            ))
            arms["active_portfolio"].append(fuse_rankings(
                direct[(ordinal, "active_P")], direct[(ordinal, "active_Q1")],
                direct[(ordinal, "active_Q2")],
            ))
        metrics = {
            arm: _aggregate_arm(arm_id=arm, items=items, rankings=rankings)
            for arm, rankings in arms.items()
        }
        primary = _paired_arm(
            left="active_portfolio", right="incumbent_portfolio", items=items, arms=arms
        )
        versus_official = _paired_arm(
            left="active_portfolio", right="official_HippoRAG", items=items, arms=arms
        )
        versus_raw = _paired_arm(
            left="active_portfolio", right="canonical_RAW", items=items, arms=arms
        )
        versus_p = _paired_arm(
            left="active_portfolio", right="retained_P", items=items, arms=arms
        )
        l5_achieved = _l5_achievement(primary)
        ranking_receipts = [
            {
                "ordinal_sha256": stable_hash({"ordinal": ordinal}),
                "component_id": component,
                "ranking_sha256": stable_hash({"retrieved_indices": list(ranking)}),
            }
            for (ordinal, component), ranking in sorted(direct.items())
        ]
        report_body = {
            "schema": SEARCH_REPORT_SCHEMA, "valid": True,
            "freeze_sha256": freeze["freeze_sha256"],
            "freeze_file_sha256": freeze_file_hash,
            "source_binding": freeze["source_binding"],
            "portfolio_design_binding": freeze["portfolio_design_binding"],
            "A_hold_binding": freeze["A_hold_binding"],
            "evaluator_epoch_transition": freeze["evaluator_epoch_transition"],
            "search_actions": freeze["search_actions"],
            "arm_metrics": metrics,
            "primary_active_minus_incumbent": primary,
            "secondary_active_minus_official_HippoRAG": versus_official,
            "secondary_active_minus_RAW": versus_raw,
            "secondary_active_minus_retained_P": versus_p,
            "L5_disposition": {
                "achievement_criterion": "positive_net_and_exact_sign_flip_p_le_0_10",
                "positive_net": primary["net_support_hit_count"] > 0,
                "exact_sign_flip_p_le_0_10": primary["paired_test"]["promoted"],
                "L5_achieved": l5_achieved,
                "M_search_used_for_epoch_transition": False,
                "followup_same_source_attempt_authorized": False,
                "statistical_superiority_claim_beyond_fixed_cohort": False,
                "compute_equivalence_claim_against_RAW_or_HippoRAG": False,
            },
            "execution": {
                "physical_component_ids": list(SEARCH_COMPONENT_IDS),
                "item_count": len(items), "physical_work_unit_count": SEARCH_WORK_UNIT_COUNT,
                "retrieval_attempt_count": attempted, "retrieval_terminal_count": completed,
                "configured_maximum_concurrency": SEARCH_MAXIMUM_CONCURRENCY,
                "observed_start_barrier_party_count": barrier.parties,
                "all_work_units_released_from_single_start_barrier": True,
                "all_terminals_joined_before_postflight_and_support_scoring": True,
                "variable_support_count_supported": True,
                "ranking_receipt_set_sha256": stable_hash(ranking_receipts),
                "external_network_calls": 0, "online_evaluator_calls": 0,
                "retries": 0, "replays": 0, "resamples": 0,
            },
            "runtime": {
                "capability_receipt_sha256": capability["receipt_sha256"],
                "fresh_bwrap_preflight_before_authorization": True,
                "official_arm_terminal_count": M_SEARCH_ITEM_COUNT,
                "official_arm_uses_frozen_local_LLM_OpenIE": True,
                "postflight_fresh_filesystem_attestation": True,
                "postflight_binding_sha256": postflight["binding_sha256"],
            },
            "raw_content_persisted": False,
        }
        report = {**report_body, "report_sha256": stable_hash(report_body)}
        _assert_public_safe(report)
        _write_json_exclusive(root / SEARCH_REPORT_FILENAME, report)
        return report
    except Exception as exc:
        failure_body = {
            "schema": FAILURE_SCHEMA, "stage": "M_search", "valid": False,
            "freeze_sha256": freeze["freeze_sha256"], "failure_stage": stage,
            "error_type_sha256": stable_hash({"error_type": type(exc).__name__}),
            "authorization_consumed": (root / SEARCH_CONSUMPTION_FILENAME).is_file(),
            "physical_work_unit_count": SEARCH_WORK_UNIT_COUNT,
            "retrieval_attempt_count": attempted, "retrieval_terminal_count": completed,
            "retries": 0, "replays": 0, "resamples": 0,
            "replay_authorized": False, "raw_content_persisted": False,
        }
        try:
            _write_json_exclusive(
                root / SEARCH_FAILURE_FILENAME,
                {**failure_body, "failure_sha256": stable_hash(failure_body)},
            )
        except Exception:
            pass
        raise MuSiQueEvaluatorPortfolioError("formal M_search failed and cannot be replayed") from exc


def formal_signatures_have_no_injection_surface() -> bool:
    forbidden = {
        "program", "programs", "retriever", "retrievers", "callable", "result",
        "results", "evidence", "items", "rankings", "support_indices",
    }
    anchor_freeze = set(inspect.signature(build_a_hold_pre_run_freeze).parameters)
    search_freeze = set(inspect.signature(build_m_search_pre_run_freeze).parameters)
    formal = set(inspect.signature(execute_a_hold_formal).parameters) | set(
        inspect.signature(execute_m_search_formal).parameters
    )
    return (
        not forbidden.intersection(anchor_freeze | search_freeze | formal)
        and "a_hold_block_path" not in anchor_freeze
        and "m_search_block_path" not in search_freeze
    )


def _add_lineage(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--acquisition-receipt", type=Path, required=True)
    parser.add_argument("--selection-secret", type=Path, required=True)
    parser.add_argument("--p-formation-receipt", type=Path, required=True)
    parser.add_argument("--p-frozen-program", type=Path, required=True)
    parser.add_argument("--m1-freeze", type=Path, required=True)
    parser.add_argument("--m1-report", type=Path, required=True)


def _add_formations(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--a-form-private-cache", type=Path, required=True)
    parser.add_argument("--a-form-public-receipt", type=Path, required=True)
    parser.add_argument("--f-search-private-cache", type=Path, required=True)
    parser.add_argument("--f-search-public-receipt", type=Path, required=True)


def _add_anchor(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--a-hold-freeze", type=Path, required=True)
    parser.add_argument("--a-hold-private-evidence", type=Path, required=True)
    parser.add_argument("--a-hold-report", type=Path, required=True)


def _add_runtime(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--capability-receipt", type=Path, required=True)
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--local-llm-model", type=Path, required=True)
    parser.add_argument("--local-embedding-model", type=Path, required=True)
    parser.add_argument("--base-binding-receipt", type=Path, required=True)
    parser.add_argument("--attestation-receipt", type=Path, required=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("form-a", "form-f"):
        command = commands.add_parser(name)
        _add_lineage(command)
        command.add_argument("--environment-0-block", type=Path, required=True)
        command.add_argument("--environment-1-block", type=Path, required=True)
        command.add_argument("--private-cache-output", type=Path, required=True)
        command.add_argument("--public-receipt-output", type=Path, required=True)
    freeze_anchor = commands.add_parser("freeze-a-hold")
    _add_lineage(freeze_anchor)
    _add_formations(freeze_anchor)
    freeze_anchor.add_argument("--execution-root", type=Path, required=True)
    freeze_anchor.add_argument("--authorization-hash", required=True)
    freeze_anchor.add_argument("--output", type=Path, required=True)
    run_anchor = commands.add_parser("run-a-hold")
    _add_lineage(run_anchor)
    _add_formations(run_anchor)
    run_anchor.add_argument("--pre-run-freeze", type=Path, required=True)
    run_anchor.add_argument("--a-hold-block", type=Path, required=True)
    run_anchor.add_argument("--execution-root", type=Path, required=True)
    freeze_search = commands.add_parser("freeze-m-search")
    _add_lineage(freeze_search)
    _add_formations(freeze_search)
    _add_anchor(freeze_search)
    _add_runtime(freeze_search)
    freeze_search.add_argument("--execution-root", type=Path, required=True)
    freeze_search.add_argument("--authorization-hash", required=True)
    freeze_search.add_argument("--output", type=Path, required=True)
    run_search = commands.add_parser("run-m-search")
    _add_lineage(run_search)
    _add_formations(run_search)
    _add_anchor(run_search)
    _add_runtime(run_search)
    run_search.add_argument("--pre-run-freeze", type=Path, required=True)
    run_search.add_argument("--m-search-block", type=Path, required=True)
    run_search.add_argument("--execution-root", type=Path, required=True)
    arguments = parser.parse_args(argv)
    base = {
        "project_root": arguments.project_root,
        "acquisition_receipt_path": arguments.acquisition_receipt,
        "selection_secret_path": arguments.selection_secret,
        "p_formation_receipt_path": arguments.p_formation_receipt,
        "p_frozen_program_path": arguments.p_frozen_program,
        "m1_freeze_path": arguments.m1_freeze,
        "m1_report_path": arguments.m1_report,
    }
    if arguments.command in {"form-a", "form-f"}:
        fn = form_a_form_stage if arguments.command == "form-a" else form_f_search_stage
        fn(
            **base,
            environment_0_block_path=arguments.environment_0_block,
            environment_1_block_path=arguments.environment_1_block,
            private_cache_output_path=arguments.private_cache_output,
            public_receipt_output_path=arguments.public_receipt_output,
        )
        return 0
    formations = {
        "a_form_private_cache_path": arguments.a_form_private_cache,
        "a_form_public_receipt_path": arguments.a_form_public_receipt,
        "f_search_private_cache_path": arguments.f_search_private_cache,
        "f_search_public_receipt_path": arguments.f_search_public_receipt,
    }
    if arguments.command == "freeze-a-hold":
        build_a_hold_pre_run_freeze(
            **base, **formations, execution_root=arguments.execution_root,
            authorization_hash=arguments.authorization_hash,
            output_path=arguments.output,
        )
        return 0
    global _CLEAN_MODULE_CLI_ACTIVE
    if arguments.command == "run-a-hold":
        _CLEAN_MODULE_CLI_ACTIVE = True
        try:
            execute_a_hold_formal(
                **base, **formations,
                pre_run_freeze_path=arguments.pre_run_freeze,
                a_hold_block_path=arguments.a_hold_block,
                execution_root=arguments.execution_root,
            )
        finally:
            _CLEAN_MODULE_CLI_ACTIVE = False
        return 0
    anchor = {
        "a_hold_pre_run_freeze_path": arguments.a_hold_freeze,
        "a_hold_private_evidence_path": arguments.a_hold_private_evidence,
        "a_hold_report_path": arguments.a_hold_report,
    }
    runtime = {
        "capability_receipt_path": arguments.capability_receipt,
        "runtime_python": arguments.runtime_python,
        "local_llm_model": arguments.local_llm_model,
        "local_embedding_model": arguments.local_embedding_model,
        "base_binding_receipt_path": arguments.base_binding_receipt,
        "attestation_receipt_path": arguments.attestation_receipt,
    }
    if arguments.command == "freeze-m-search":
        build_m_search_pre_run_freeze(
            **base, **formations, **anchor, **runtime,
            execution_root=arguments.execution_root,
            authorization_hash=arguments.authorization_hash,
            output_path=arguments.output,
        )
        return 0
    _CLEAN_MODULE_CLI_ACTIVE = True
    try:
        execute_m_search_formal(
            **base, **formations, **anchor, **runtime,
            pre_run_freeze_path=arguments.pre_run_freeze,
            m_search_block_path=arguments.m_search_block,
            execution_root=arguments.execution_root,
        )
    finally:
        _CLEAN_MODULE_CLI_ACTIVE = False
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
