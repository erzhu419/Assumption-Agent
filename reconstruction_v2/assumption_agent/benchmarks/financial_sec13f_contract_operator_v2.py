from __future__ import annotations

"""Typed, offline SEC-13F operator derived only from the public task contract.

The module is intentionally self-contained.  It does not import the period-out
pack, either reference oracle, the verifier, or any measurement artifact.  A
strict parser turns the public instruction into a typed plan; a stdlib-only
streaming executor then evaluates that plan before the benchmark verifier runs.
"""

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterator, Mapping, Sequence, TextIO
import unicodedata


OPERATOR_VERSION = "financial_sec13f_public_contract_operator_v2"
ASSET_VERSION = "financial_sec13f_public_contract_asset_v2"
PLAN_VERSION = "financial_sec13f_public_contract_plan_v2"
EXTRACTION_RECEIPT_VERSION = (
    "financial_sec13f_public_contract_extraction_receipt_v2"
)
QUERY_RECEIPT_VERSION = "financial_sec13f_public_contract_query_receipt_v2"
FORMATION_POLICY = "public_instruction_contract_only_v1"
PARSER_MODE = "anchored_public_sec13f_3q_4q_grammar_v1"
NUMERIC_ENGINE = "python_decimal_exact_until_json_boundary_v1"
EXCLUDED_INPUTS = (
    "benchmark_pack",
    "measurement_view",
    "selection_seed",
    "item_or_entity_inventory",
    "oracle_output",
    "gold_or_expected_answer",
    "verifier_content",
    "sealed_content",
    "model_or_online_service",
)

MAXIMUM_INSTRUCTION_BYTES = 64 * 1024
MAXIMUM_ENTITY_CHARACTERS = 256

COVERPAGE_COLUMNS = frozenset(
    {
        "ACCESSION_NUMBER",
        "REPORTCALENDARORQUARTER",
        "REPORTTYPE",
        "FILINGMANAGER_NAME",
    }
)
INFOTABLE_COLUMNS = frozenset(
    {
        "ACCESSION_NUMBER",
        "NAMEOFISSUER",
        "TITLEOFCLASS",
        "CUSIP",
        "VALUE",
    }
)

STOCK_TITLE_CLASSES = frozenset(
    {
        "ADR",
        "CAP STK CL A",
        "CAP STK CL C",
        "CL A",
        "CL A COM",
        "CL A NEW",
        "CL B",
        "CL B NEW",
        "CMN",
        "COM",
        "COM CL A",
        "COM NEW",
        "COM SHS",
        "COMM STK",
        "COMMON",
        "COMMON STOCK",
        "EQUITY",
        "FOREIGN STOCK",
        "ORD SHS",
        "SHS CL A",
        "SPONSORED ADR",
        "SPONSORED ADS",
        "STOCK",
        "CLASS A",
        "CLASS A COM",
    }
)

OPERATION_ORDER_BY_TEMPLATE: Mapping[str, tuple[str, ...]] = {
    "four_question_v1": (
        "current_aum",
        "current_stock_row_count",
        "positive_delta_cusip_rank",
        "current_holder_manager_rank",
    ),
    "three_question_v1": (
        "current_aum",
        "positive_delta_cusip_rank",
        "current_holder_manager_rank",
    ),
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ROOT_HEADER = re.compile(
    r"^You are a financial analyst comparing official SEC Form 13F data for "
    r"(.+?) against (.+?)\. The previous data is in `([^`]+)` and current "
    r"data is in `([^`]+)`\.$"
)
_NUMBERED_QUESTION = re.compile(r"(?m)^(\d+)\.\s+")


class FinancialSec13FContractError(RuntimeError):
    """The public contract, source data, or typed plan failed closed."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def payload_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    source = Path(path).expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise FinancialSec13FContractError(f"{label} is not a SHA-256 digest")
    return value


def _with_self_hash(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload)
    if field in result:
        raise FinancialSec13FContractError(f"{field} already exists")
    result[field] = payload_hash(result)
    return result


def _verify_self_hash(
    payload: Mapping[str, Any], *, field: str, label: str
) -> str:
    declared = _require_sha256(payload.get(field), f"{label} {field}")
    body = dict(payload)
    del body[field]
    if payload_hash(body) != declared:
        raise FinancialSec13FContractError(f"{label} self hash mismatch")
    return declared


def _read_json(path: str | Path, *, maximum_bytes: int = 8 * 1024 * 1024) -> dict[str, Any]:
    source = Path(path).expanduser().resolve(strict=True)
    if source.stat().st_size > maximum_bytes:
        raise FinancialSec13FContractError("JSON input exceeds its byte bound")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FinancialSec13FContractError("JSON input is unreadable") from exc
    if not isinstance(value, dict):
        raise FinancialSec13FContractError("JSON input must contain one object")
    return value


def _safe_write_target(path: str | Path) -> Path:
    unresolved = Path(path).expanduser()
    if unresolved.is_symlink():
        raise FinancialSec13FContractError("JSON output target may not be a symlink")
    parent = unresolved.parent
    if parent.is_symlink():
        raise FinancialSec13FContractError("JSON output parent may not be a symlink")
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise FinancialSec13FContractError("JSON output parent is invalid")
    target = parent.resolve(strict=True) / unresolved.name
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise FinancialSec13FContractError(
            "JSON output target is not a regular file"
        )
    return target


def _atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = _safe_write_target(path)
    raw = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(re.findall(r"[a-z0-9]+", text))


def normalize_title_class(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).upper()
    return " ".join(text.split())


def canonical_cusip(value: object) -> str:
    return "".join(str(value or "").upper().split())


def parse_sec_value(value: object) -> Decimal:
    text = str(value or "").strip().replace(",", "")
    if not text:
        raise FinancialSec13FContractError("SEC VALUE is empty")
    try:
        parsed = Decimal(text)
    except InvalidOperation as exc:
        raise FinancialSec13FContractError("SEC VALUE is not decimal") from exc
    if not parsed.is_finite() or parsed < 0:
        raise FinancialSec13FContractError(
            "SEC VALUE is negative or non-finite"
        )
    return parsed


def decimal_to_json_number(value: Decimal) -> int | float:
    if not value.is_finite():
        raise FinancialSec13FContractError("result is non-finite")
    if value == value.to_integral_value():
        return int(value)
    result = float(value)
    if not math.isfinite(result):
        raise FinancialSec13FContractError("result exceeds finite JSON range")
    return result


def _semantic_contract_text() -> str:
    classes = ", ".join(sorted(STOCK_TITLE_CLASSES))
    return (
        "Use only non-NOTICE filings at the latest REPORTCALENDARORQUARTER "
        "within each period. Manager and issuer matching is Unicode NFKC, "
        "case-insensitive, and punctuation-insensitive. Every selected fund "
        "has exactly one eligible accession per required period. AUM is the "
        "sum of VALUE over all rows for the selected current accession. "
        "Stock-only operations accept exactly these normalized TITLEOFCLASS "
        f"values: {classes}. For an issuer name, choose the matching CUSIP "
        "with greatest aggregate current VALUE (CUSIP ascending on a tie). "
        "Aggregate rows by CUSIP or normalized manager before ranking; rank "
        "VALUE descending and use canonical CUSIP or normalized manager "
        "ascending to break ties."
    )


def _contract_descriptor() -> dict[str, Any]:
    return {
        "identity_normalization": "unicode_nfkc_casefold_ascii_alnum_tokens_v1",
        "title_class_normalization": "unicode_nfkc_upper_whitespace_v1",
        "cusip_normalization": "upper_remove_whitespace_v1",
        "snapshot_policy": "global_latest_then_exclude_notice_no_fallback_v1",
        "manager_resolution": "exact_normalized_identity_unique_accession_v1",
        "issuer_resolution": "exact_identity_aggregate_value_cusip_ascending_tie_v1",
        "aggregation": "accession_cusip_or_normalized_manager_before_rank_v1",
        "ranking": "decimal_value_descending_identity_ascending_v1",
        "numeric_engine": NUMERIC_ENGINE,
        "stock_title_classes": sorted(STOCK_TITLE_CLASSES),
        "coverpage_columns": sorted(COVERPAGE_COLUMNS),
        "infotable_columns": sorted(INFOTABLE_COLUMNS),
        "templates": {
            key: list(value) for key, value in OPERATION_ORDER_BY_TEMPLATE.items()
        },
    }


def _template_grammar_descriptor() -> dict[str, Any]:
    return {
        "parser_mode": PARSER_MODE,
        "templates": {
            "four_question_v1": {
                "question_count": 4,
                "rank_top_k": [3, 5],
                "answer_keys": [
                    "q1_answer",
                    "q2_answer",
                    "q3_answer",
                    "q4_answer",
                ],
            },
            "three_question_v1": {
                "question_count": 3,
                "rank_top_k": [3, 5],
                "answer_keys": ["q1_answer", "q2_answer", "q3_answer"],
            },
        },
        "root_binding": "absolute_posix_instruction_roots_v1",
        "entity_matching": "anchored_question_span_then_exact_identity_v1",
    }


def build_contract_asset_v2(
    *,
    candidate_skill_source_receipt_hash: str,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    source_receipt = _require_sha256(
        candidate_skill_source_receipt_hash,
        "candidate skill source receipt hash",
    )
    operator_hash = sha256_file(Path(__file__))
    descriptor = _contract_descriptor()
    contract_hash = payload_hash(descriptor)
    grammar = _template_grammar_descriptor()
    grammar_hash = payload_hash(grammar)
    candidate_id = payload_hash(
        {
            "operator_version": OPERATOR_VERSION,
            "operator_source_sha256": operator_hash,
            "contract_hash": contract_hash,
            "template_grammar_hash": grammar_hash,
            "candidate_skill_source_receipt_hash": source_receipt,
        }
    )
    payload = _with_self_hash(
        {
            "asset_version": ASSET_VERSION,
            "operator_version": OPERATOR_VERSION,
            "formation_policy": FORMATION_POLICY,
            "contract_descriptor": descriptor,
            "contract_hash": contract_hash,
            "template_grammar_descriptor": grammar,
            "template_grammar_hash": grammar_hash,
            "operator_source_sha256": operator_hash,
            "candidate_skill_source_receipt_hash": source_receipt,
            "candidate_id": candidate_id,
            "excluded_inputs": list(EXCLUDED_INPUTS),
            "model_calls": 0,
            "online_calls": 0,
        },
        "manifest_hash",
    )
    if output_path is not None:
        _atomic_write_json(output_path, payload)
    return payload


def validate_contract_asset_payload_v2(
    asset: Mapping[str, Any],
) -> dict[str, Any]:
    expected_fields = {
        "asset_version",
        "operator_version",
        "formation_policy",
        "contract_descriptor",
        "contract_hash",
        "template_grammar_descriptor",
        "template_grammar_hash",
        "operator_source_sha256",
        "candidate_skill_source_receipt_hash",
        "candidate_id",
        "excluded_inputs",
        "model_calls",
        "online_calls",
        "manifest_hash",
    }
    if set(asset) != expected_fields:
        raise FinancialSec13FContractError("contract asset fields drifted")
    _verify_self_hash(asset, field="manifest_hash", label="contract asset")
    if (
        asset.get("asset_version") != ASSET_VERSION
        or asset.get("operator_version") != OPERATOR_VERSION
        or asset.get("formation_policy") != FORMATION_POLICY
        or asset.get("contract_descriptor") != _contract_descriptor()
        or asset.get("contract_hash") != payload_hash(_contract_descriptor())
        or asset.get("template_grammar_descriptor")
        != _template_grammar_descriptor()
        or asset.get("template_grammar_hash")
        != payload_hash(_template_grammar_descriptor())
        or asset.get("operator_source_sha256") != sha256_file(Path(__file__))
        or asset.get("model_calls") != 0
        or asset.get("online_calls") != 0
    ):
        raise FinancialSec13FContractError("contract asset semantics drifted")
    source_receipt = _require_sha256(
        asset.get("candidate_skill_source_receipt_hash"),
        "contract asset source receipt",
    )
    candidate_id = _require_sha256(asset.get("candidate_id"), "candidate id")
    expected_candidate = payload_hash(
        {
            "operator_version": OPERATOR_VERSION,
            "operator_source_sha256": asset["operator_source_sha256"],
            "contract_hash": asset["contract_hash"],
            "template_grammar_hash": asset["template_grammar_hash"],
            "candidate_skill_source_receipt_hash": source_receipt,
        }
    )
    if candidate_id != expected_candidate:
        raise FinancialSec13FContractError("candidate identity drifted")
    if asset.get("excluded_inputs") != list(EXCLUDED_INPUTS):
        raise FinancialSec13FContractError("contract asset exclusions drifted")
    return dict(asset)


def load_contract_asset_v2(path: str | Path) -> dict[str, Any]:
    return validate_contract_asset_payload_v2(_read_json(path))


def _safe_instruction_root(value: str, label: str) -> str:
    if (
        not value.startswith("/")
        or "\x00" in value
        or any(part in {"", ".", ".."} for part in value.split("/")[1:])
    ):
        raise FinancialSec13FContractError(f"{label} root is unsafe")
    return value


def _split_instruction(instruction: str) -> tuple[str, str, str]:
    raw = instruction.encode("utf-8")
    if not raw or len(raw) > MAXIMUM_INSTRUCTION_BYTES:
        raise FinancialSec13FContractError("instruction byte length is invalid")
    marker = "\n\nFrozen data semantics: "
    questions_marker = "\n\nQuestions:\n\n"
    if marker not in instruction or questions_marker not in instruction:
        raise FinancialSec13FContractError("instruction contract markers are absent")
    header, remainder = instruction.split(marker, 1)
    contract, questions = remainder.split(questions_marker, 1)
    if contract != _semantic_contract_text():
        raise FinancialSec13FContractError("public semantic contract drifted")
    return header, contract, questions


def _question_blocks(questions: str) -> tuple[str, ...]:
    format_marker = "\n\nWrite `/root/answers.json`"
    if format_marker not in questions:
        raise FinancialSec13FContractError("answer format contract is absent")
    question_text, format_tail = questions.split(format_marker, 1)
    matches = list(_NUMBERED_QUESTION.finditer(question_text))
    if len(matches) not in {3, 4}:
        raise FinancialSec13FContractError("instruction needs three or four questions")
    blocks: list[str] = []
    for index, match in enumerate(matches):
        if int(match.group(1)) != index + 1:
            raise FinancialSec13FContractError("question numbering is not contiguous")
        end = matches[index + 1].start() if index + 1 < len(matches) else len(question_text)
        block = " ".join(question_text[match.end() : end].split())
        if not block:
            raise FinancialSec13FContractError("question block is empty")
        blocks.append(block)
    expected_tail = (
        " with keys `q1_answer`, `q2_answer`, `q3_answer`, and `q4_answer` "
        "in that order. q1 and q2 are numbers; q3 and q4 are ordered JSON arrays.\n"
        if len(blocks) == 4
        else " with keys `q1_answer`, `q2_answer`, and `q3_answer` in that "
        "order. q1 is a number; q2 and q3 are ordered JSON arrays.\n"
    )
    if format_tail != expected_tail:
        raise FinancialSec13FContractError("answer format contract drifted")
    return tuple(blocks)


def _entity(value: str, label: str) -> tuple[str, str]:
    raw = value.strip()
    normalized = normalize_name(raw)
    if (
        not raw
        or len(raw) > MAXIMUM_ENTITY_CHARACTERS
        or not normalized
        or "\n" in raw
        or "\r" in raw
    ):
        raise FinancialSec13FContractError(f"{label} entity is invalid")
    return raw, normalized


def _match(pattern: str, block: str, label: str) -> re.Match[str]:
    match = re.fullmatch(pattern, block)
    if match is None:
        raise FinancialSec13FContractError(f"{label} question grammar drifted")
    return match


def build_contract_plan_v2(
    instruction: str,
    asset: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    asset = validate_contract_asset_payload_v2(asset)

    header, _, questions = _split_instruction(instruction)
    header_match = _ROOT_HEADER.fullmatch(header)
    if header_match is None:
        raise FinancialSec13FContractError("period/root header grammar drifted")
    previous_label, current_label, previous_root, current_root = header_match.groups()
    if not previous_label.strip() or not current_label.strip():
        raise FinancialSec13FContractError("period label is empty")
    previous_root = _safe_instruction_root(previous_root, "previous")
    current_root = _safe_instruction_root(current_root, "current")
    if previous_root == current_root:
        raise FinancialSec13FContractError("period roots must be distinct")

    blocks = _question_blocks(questions)
    template = "four_question_v1" if len(blocks) == 4 else "three_question_v1"
    aum_match = _match(
        r"What is the current-period AUM of (.+)\?", blocks[0], "AUM"
    )
    aum_raw, aum_normalized = _entity(aum_match.group(1), "AUM")
    operations: list[dict[str, Any]] = []

    def append_operation(
        *,
        index: int,
        operation: str,
        entity_raw: str,
        entity_normalized: str,
        top_k: int | None,
        entity_ref: str | None = None,
    ) -> None:
        row: dict[str, Any] = {
            "question_index": index,
            "answer_key": f"q{index}_answer",
            "operation": operation,
            "entity": entity_raw,
            "entity_normalized": entity_normalized,
            "entity_sha256": hashlib.sha256(entity_raw.encode("utf-8")).hexdigest(),
            "question_sha256": hashlib.sha256(
                blocks[index - 1].encode("utf-8")
            ).hexdigest(),
            "top_k": top_k,
            "entity_ref": entity_ref,
        }
        operations.append(row)

    append_operation(
        index=1,
        operation="current_aum",
        entity_raw=aum_raw,
        entity_normalized=aum_normalized,
        top_k=None,
    )
    offset = 0
    if template == "four_question_v1":
        count_match = _match(
            r"How many stock rows are held by (.+) in the current period\?",
            blocks[1],
            "stock count",
        )
        count_raw, count_normalized = _entity(count_match.group(1), "stock count")
        if count_normalized != aum_normalized:
            raise FinancialSec13FContractError(
                "stock count manager must reference the AUM manager"
            )
        append_operation(
            index=2,
            operation="current_stock_row_count",
            entity_raw=aum_raw,
            entity_normalized=aum_normalized,
            top_k=None,
            entity_ref="q1",
        )
        offset = 1

    increase_index = 2 + offset
    increase_match = _match(
        r"What are the top (3|5) CUSIPs with increased investment by (.+) "
        r"from the previous period to the current period, ranked by "
        r"dollar-value increase\?",
        blocks[increase_index - 1],
        "increase rank",
    )
    increase_raw, increase_normalized = _entity(
        increase_match.group(2), "increase rank"
    )
    append_operation(
        index=increase_index,
        operation="positive_delta_cusip_rank",
        entity_raw=increase_raw,
        entity_normalized=increase_normalized,
        top_k=int(increase_match.group(1)),
    )

    holder_index = 3 + offset
    holder_match = _match(
        r"Which top (3|5) fund managers hold (.+) in the current period, "
        r"ranked by aggregate position value\?",
        blocks[holder_index - 1],
        "holder rank",
    )
    issuer_raw, issuer_normalized = _entity(holder_match.group(2), "issuer")
    append_operation(
        index=holder_index,
        operation="current_holder_manager_rank",
        entity_raw=issuer_raw,
        entity_normalized=issuer_normalized,
        top_k=int(holder_match.group(1)),
    )

    plan = _with_self_hash(
        {
            "plan_version": PLAN_VERSION,
            "operator_version": OPERATOR_VERSION,
            "candidate_id": asset["candidate_id"],
            "asset_manifest_hash": asset["manifest_hash"],
            "contract_hash": asset["contract_hash"],
            "template_grammar_hash": asset["template_grammar_hash"],
            "operator_source_sha256": asset["operator_source_sha256"],
            "instruction_sha256": hashlib.sha256(
                instruction.encode("utf-8")
            ).hexdigest(),
            "period_label_hashes": {
                "previous": hashlib.sha256(previous_label.encode("utf-8")).hexdigest(),
                "current": hashlib.sha256(current_label.encode("utf-8")).hexdigest(),
            },
            "template_id": template,
            "previous_root": previous_root,
            "current_root": current_root,
            "operations": operations,
            "operation_set_hash": payload_hash(operations),
            "entity_lifecycle": (
                "ephemeral_integration_tmpdir_and_container_tmp_deleted_"
                "before_verifier_v1"
            ),
            "raw_instruction_persisted": False,
            "model_calls": 0,
            "online_calls": 0,
        },
        "plan_hash",
    )
    validated = validate_contract_plan_v2(plan, asset)
    receipt = _with_self_hash(
        {
            "receipt_version": EXTRACTION_RECEIPT_VERSION,
            "parser_mode": PARSER_MODE,
            "candidate_id": validated["candidate_id"],
            "asset_manifest_hash": validated["asset_manifest_hash"],
            "contract_hash": validated["contract_hash"],
            "template_grammar_hash": validated["template_grammar_hash"],
            "plan_hash": validated["plan_hash"],
            "instruction_sha256": validated["instruction_sha256"],
            "template_id": validated["template_id"],
            "question_hashes": [row["question_sha256"] for row in operations],
            "entity_hashes": [row["entity_sha256"] for row in operations],
            "semantic_assignment": [row["operation"] for row in operations],
            "raw_instruction_persisted": False,
            "raw_entity_persisted_in_receipt": False,
            "model_calls": 0,
            "online_calls": 0,
        },
        "receipt_hash",
    )
    return validated, receipt


def validate_contract_plan_v2(
    plan: Mapping[str, Any],
    asset: Mapping[str, Any],
) -> dict[str, Any]:
    asset = validate_contract_asset_payload_v2(asset)
    expected_fields = {
        "plan_version",
        "operator_version",
        "candidate_id",
        "asset_manifest_hash",
        "contract_hash",
        "template_grammar_hash",
        "operator_source_sha256",
        "instruction_sha256",
        "period_label_hashes",
        "template_id",
        "previous_root",
        "current_root",
        "operations",
        "operation_set_hash",
        "entity_lifecycle",
        "raw_instruction_persisted",
        "model_calls",
        "online_calls",
        "plan_hash",
    }
    if set(plan) != expected_fields:
        raise FinancialSec13FContractError("contract plan fields drifted")
    _verify_self_hash(plan, field="plan_hash", label="contract plan")
    for field in (
        "candidate_id",
        "asset_manifest_hash",
        "contract_hash",
        "template_grammar_hash",
        "operator_source_sha256",
        "instruction_sha256",
        "operation_set_hash",
    ):
        _require_sha256(plan.get(field), f"contract plan {field}")
    if (
        plan.get("plan_version") != PLAN_VERSION
        or plan.get("operator_version") != OPERATOR_VERSION
        or plan.get("operator_source_sha256") != sha256_file(Path(__file__))
        or plan.get("entity_lifecycle")
        != (
            "ephemeral_integration_tmpdir_and_container_tmp_deleted_"
            "before_verifier_v1"
        )
        or plan.get("raw_instruction_persisted") is not False
        or plan.get("model_calls") != 0
        or plan.get("online_calls") != 0
    ):
        raise FinancialSec13FContractError("contract plan boundary drifted")
    if (
        plan.get("candidate_id") != asset.get("candidate_id")
        or plan.get("asset_manifest_hash") != asset.get("manifest_hash")
        or plan.get("contract_hash") != asset.get("contract_hash")
        or plan.get("template_grammar_hash")
        != asset.get("template_grammar_hash")
        or plan.get("operator_source_sha256")
        != asset.get("operator_source_sha256")
    ):
        raise FinancialSec13FContractError("contract plan asset binding drifted")
    labels = plan.get("period_label_hashes")
    if not isinstance(labels, dict) or set(labels) != {"previous", "current"}:
        raise FinancialSec13FContractError("period label hashes are malformed")
    for value in labels.values():
        _require_sha256(value, "period label hash")
    previous_root = _safe_instruction_root(str(plan.get("previous_root")), "previous")
    current_root = _safe_instruction_root(str(plan.get("current_root")), "current")
    if previous_root == current_root:
        raise FinancialSec13FContractError("period roots must be distinct")
    template = plan.get("template_id")
    expected_operations = OPERATION_ORDER_BY_TEMPLATE.get(str(template))
    operations = plan.get("operations")
    if (
        expected_operations is None
        or not isinstance(operations, list)
        or len(operations) != len(expected_operations)
        or payload_hash(operations) != plan.get("operation_set_hash")
    ):
        raise FinancialSec13FContractError("typed operation set is malformed")
    for index, (row, expected_operation) in enumerate(
        zip(operations, expected_operations), start=1
    ):
        if not isinstance(row, dict):
            raise FinancialSec13FContractError("typed operation row is malformed")
        expected_row_fields = {
            "question_index",
            "answer_key",
            "operation",
            "entity",
            "entity_normalized",
            "entity_sha256",
            "question_sha256",
            "top_k",
            "entity_ref",
        }
        entity = row.get("entity")
        if (
            set(row) != expected_row_fields
            or row.get("question_index") != index
            or row.get("answer_key") != f"q{index}_answer"
            or row.get("operation") != expected_operation
            or not isinstance(entity, str)
            or not entity
            or len(entity) > MAXIMUM_ENTITY_CHARACTERS
            or row.get("entity_normalized") != normalize_name(entity)
            or hashlib.sha256(entity.encode("utf-8")).hexdigest()
            != row.get("entity_sha256")
        ):
            raise FinancialSec13FContractError("typed operation identity drifted")
        _require_sha256(row.get("question_sha256"), "question hash")
        if expected_operation in {
            "positive_delta_cusip_rank",
            "current_holder_manager_rank",
        }:
            if row.get("top_k") not in {3, 5} or row.get("entity_ref") is not None:
                raise FinancialSec13FContractError("rank operation scalar drifted")
        elif expected_operation == "current_stock_row_count":
            if row.get("top_k") is not None or row.get("entity_ref") != "q1":
                raise FinancialSec13FContractError("stock reference drifted")
            if row.get("entity_normalized") != operations[0].get(
                "entity_normalized"
            ):
                raise FinancialSec13FContractError("stock manager drifted")
        elif row.get("top_k") is not None or row.get("entity_ref") is not None:
            raise FinancialSec13FContractError("scalar attached to non-rank operation")
    return dict(plan)


def _parse_report_date(value: str) -> datetime:
    try:
        return datetime.strptime(value.strip().upper(), "%d-%b-%Y")
    except ValueError as exc:
        raise FinancialSec13FContractError(
            "REPORTCALENDARORQUARTER is not DD-MON-YYYY"
        ) from exc


def _rows(
    path: Path,
    required_columns: frozenset[str],
) -> Iterator[dict[str, str]]:
    if path.is_symlink() or not path.is_file():
        raise FinancialSec13FContractError(f"required SEC table is missing: {path.name}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        header = reader.fieldnames
        if (
            not isinstance(header, list)
            or len(header) != len(set(header))
            or not required_columns.issubset(header)
        ):
            raise FinancialSec13FContractError(
                f"SEC table header is incompatible: {path.name}"
            )
        for row in reader:
            if None in row:
                raise FinancialSec13FContractError(
                    f"SEC table row has excess fields: {path.name}"
                )
            yield {str(key): str(value or "") for key, value in row.items()}


@dataclass(frozen=True)
class _Snapshot:
    report_date: str
    accession_to_manager: Mapping[str, str]
    display_by_manager: Mapping[str, str]
    accessions_by_manager: Mapping[str, tuple[str, ...]]

    def unique_accession(self, manager: str) -> str:
        accessions = self.accessions_by_manager.get(manager, ())
        if len(accessions) != 1:
            raise FinancialSec13FContractError(
                "target manager is not uniquely resolvable"
            )
        return accessions[0]


def _load_snapshot(root: Path) -> _Snapshot:
    cover_path = root / "COVERPAGE.tsv"
    rows: list[tuple[datetime, str, str, str]] = []
    for row in _rows(cover_path, COVERPAGE_COLUMNS):
        accession = row["ACCESSION_NUMBER"].strip()
        display = row["FILINGMANAGER_NAME"].strip()
        if not accession or not display:
            continue
        rows.append(
            (
                _parse_report_date(row["REPORTCALENDARORQUARTER"]),
                accession,
                row["REPORTTYPE"].strip(),
                display,
            )
        )
    if not rows:
        raise FinancialSec13FContractError("COVERPAGE contains no usable filing")
    latest = max(row[0] for row in rows)
    accession_to_manager: dict[str, str] = {}
    displays: dict[str, set[str]] = defaultdict(set)
    accessions: dict[str, set[str]] = defaultdict(set)
    for date, accession, report_type, display in rows:
        if date != latest or "NOTICE" in report_type.upper():
            continue
        manager = normalize_name(display)
        if not manager:
            continue
        prior = accession_to_manager.get(accession)
        if prior is not None and prior != manager:
            raise FinancialSec13FContractError(
                "one accession maps to multiple normalized managers"
            )
        accession_to_manager[accession] = manager
        displays[manager].add(display)
        accessions[manager].add(accession)
    if not accession_to_manager:
        raise FinancialSec13FContractError(
            "latest SEC snapshot contains no non-NOTICE filing"
        )
    return _Snapshot(
        report_date=latest.strftime("%Y-%m-%d"),
        accession_to_manager=accession_to_manager,
        display_by_manager={
            manager: sorted(values, key=lambda value: (normalize_name(value), value))[0]
            for manager, values in displays.items()
        },
        accessions_by_manager={
            manager: tuple(sorted(values)) for manager, values in accessions.items()
        },
    )


def _resolve_runtime_root(
    supplied: str | Path,
    expected: str,
    label: str,
) -> Path:
    raw = Path(supplied).expanduser()
    if raw.is_symlink() or not raw.is_dir():
        raise FinancialSec13FContractError(f"{label} data root is invalid")
    resolved = raw.resolve(strict=True)
    expected_path = Path(expected).expanduser()
    if expected_path.is_symlink() or not expected_path.is_dir():
        raise FinancialSec13FContractError(
            f"instruction-bound {label} root is unavailable"
        )
    if resolved != expected_path.resolve(strict=True):
        raise FinancialSec13FContractError(f"{label} data root binding drifted")
    return resolved


def _input_file_receipts(
    previous: Path,
    current: Path,
) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    for role, root in (("previous", previous), ("current", current)):
        for table in ("COVERPAGE.tsv", "INFOTABLE.tsv"):
            path = root / table
            if path.is_symlink() or not path.is_file():
                raise FinancialSec13FContractError(
                    f"required SEC table is missing: {role}/{table}"
                )
            rows.append(
                {
                    "role": role,
                    "table": table,
                    "size_bytes": path.stat().st_size,
                    "file_sha256": sha256_file(path),
                }
            )
    return rows, payload_hash(rows)


def execute_contract_plan_v2(
    plan: Mapping[str, Any],
    previous_root: str | Path,
    current_root: str | Path,
    output_path: str | Path,
    receipt_path: str | Path | None = None,
    *,
    asset: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    validated = validate_contract_plan_v2(plan, asset)
    previous = _resolve_runtime_root(
        previous_root, validated["previous_root"], "previous"
    )
    current = _resolve_runtime_root(
        current_root, validated["current_root"], "current"
    )
    input_file_receipts, input_set_hash = _input_file_receipts(
        previous, current
    )
    previous_snapshot = _load_snapshot(previous)
    current_snapshot = _load_snapshot(current)
    operations = validated["operations"]
    by_operation = {row["operation"]: row for row in operations}

    aum_manager = by_operation["current_aum"]["entity_normalized"]
    increase_manager = by_operation["positive_delta_cusip_rank"][
        "entity_normalized"
    ]
    issuer = by_operation["current_holder_manager_rank"]["entity_normalized"]
    aum_accession = current_snapshot.unique_accession(aum_manager)
    previous_increase_accession = previous_snapshot.unique_accession(
        increase_manager
    )
    current_increase_accession = current_snapshot.unique_accession(
        increase_manager
    )

    aum = Decimal()
    stock_count = 0
    current_stock: dict[str, Decimal] = defaultdict(Decimal)
    issuer_values: dict[str, Decimal] = defaultdict(Decimal)
    for row in _rows(current / "INFOTABLE.tsv", INFOTABLE_COLUMNS):
        accession = row["ACCESSION_NUMBER"].strip()
        if accession not in current_snapshot.accession_to_manager:
            continue
        value = parse_sec_value(row["VALUE"])
        cusip = canonical_cusip(row["CUSIP"])
        if accession == aum_accession:
            aum += value
        stock = normalize_title_class(row["TITLEOFCLASS"])
        if stock in STOCK_TITLE_CLASSES:
            if accession == aum_accession:
                stock_count += 1
            if accession == current_increase_accession and cusip:
                current_stock[cusip] += value
        if normalize_name(row["NAMEOFISSUER"]) == issuer and cusip:
            issuer_values[cusip] += value

    if not issuer_values:
        raise FinancialSec13FContractError("target issuer is not exactly resolvable")
    target_cusip = min(
        issuer_values.items(), key=lambda row: (-row[1], row[0])
    )[0]

    previous_stock: dict[str, Decimal] = defaultdict(Decimal)
    for row in _rows(previous / "INFOTABLE.tsv", INFOTABLE_COLUMNS):
        accession = row["ACCESSION_NUMBER"].strip()
        if (
            accession not in previous_snapshot.accession_to_manager
            or accession != previous_increase_accession
            or normalize_title_class(row["TITLEOFCLASS"])
            not in STOCK_TITLE_CLASSES
        ):
            continue
        cusip = canonical_cusip(row["CUSIP"])
        if not cusip:
            continue
        previous_stock[cusip] += parse_sec_value(row["VALUE"])

    holder_values: dict[str, Decimal] = defaultdict(Decimal)
    for row in _rows(current / "INFOTABLE.tsv", INFOTABLE_COLUMNS):
        accession = row["ACCESSION_NUMBER"].strip()
        manager = current_snapshot.accession_to_manager.get(accession)
        if manager is None or canonical_cusip(row["CUSIP"]) != target_cusip:
            continue
        holder_values[manager] += parse_sec_value(row["VALUE"])

    deltas = {
        cusip: current_stock.get(cusip, Decimal())
        - previous_stock.get(cusip, Decimal())
        for cusip in set(current_stock) | set(previous_stock)
    }
    increase_top_k = by_operation["positive_delta_cusip_rank"]["top_k"]
    increases = [
        cusip
        for cusip, value in sorted(deltas.items(), key=lambda row: (-row[1], row[0]))
        if value > 0
    ][:increase_top_k]
    holder_top_k = by_operation["current_holder_manager_rank"]["top_k"]
    managers = [
        current_snapshot.display_by_manager[manager]
        for manager, value in sorted(
            holder_values.items(), key=lambda row: (-row[1], row[0])
        )
        if value > 0
    ][:holder_top_k]

    answers: dict[str, Any] = {}
    for row in operations:
        operation = row["operation"]
        if operation == "current_aum":
            answers[row["answer_key"]] = decimal_to_json_number(aum)
        elif operation == "current_stock_row_count":
            answers[row["answer_key"]] = stock_count
        elif operation == "positive_delta_cusip_rank":
            answers[row["answer_key"]] = increases
        elif operation == "current_holder_manager_rank":
            answers[row["answer_key"]] = managers
        else:  # pragma: no cover - plan validation closes this branch.
            raise FinancialSec13FContractError("unknown typed operation")

    unresolved_output = Path(output_path).expanduser()
    if unresolved_output.is_symlink():
        raise FinancialSec13FContractError("output target may not be a symlink")
    pre_output_exists = unresolved_output.is_file()
    pre_output_sha256 = (
        sha256_file(unresolved_output) if pre_output_exists else None
    )
    _atomic_write_json(unresolved_output, answers)
    output = _safe_write_target(unresolved_output)
    post_output_sha256 = sha256_file(output)

    receipt = _with_self_hash(
        {
            "receipt_version": QUERY_RECEIPT_VERSION,
            "operator_version": OPERATOR_VERSION,
            "candidate_id": validated["candidate_id"],
            "asset_manifest_hash": validated["asset_manifest_hash"],
            "contract_hash": validated["contract_hash"],
            "operator_source_sha256": validated["operator_source_sha256"],
            "plan_hash": validated["plan_hash"],
            "numeric_engine": NUMERIC_ENGINE,
            "input_file_receipts": input_file_receipts,
            "input_set_hash": input_set_hash,
            "pre_output_exists": pre_output_exists,
            "pre_output_sha256": pre_output_sha256,
            "post_output_sha256": post_output_sha256,
            "output_changed": (
                not pre_output_exists or pre_output_sha256 != post_output_sha256
            ),
            "answer_key_set_hash": payload_hash(list(answers)),
            "answers_payload_persisted_in_receipt": False,
            "raw_entity_persisted_in_receipt": False,
            "network_calls": 0,
            "model_calls": 0,
            "verifier_content_accessed": False,
            "gold_content_accessed": False,
            "pack_content_accessed": False,
        },
        "receipt_hash",
    )
    if receipt_path is not None:
        _atomic_write_json(receipt_path, receipt)
    return answers, receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    asset = subparsers.add_parser("build-asset")
    asset.add_argument("--skill-source-receipt-hash", required=True)
    asset.add_argument("--output", type=Path, required=True)
    execute = subparsers.add_parser("execute")
    execute.add_argument("--asset", type=Path, required=True)
    execute.add_argument("--plan", type=Path, required=True)
    execute.add_argument("--previous-root", type=Path, required=True)
    execute.add_argument("--current-root", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--receipt-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build-asset":
        payload = build_contract_asset_v2(
            candidate_skill_source_receipt_hash=args.skill_source_receipt_hash,
            output_path=args.output,
        )
    elif args.command == "execute":
        asset = load_contract_asset_v2(args.asset)
        _, payload = execute_contract_plan_v2(
            _read_json(args.plan),
            args.previous_root,
            args.current_root,
            args.output,
            args.receipt_output,
            asset=asset,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
