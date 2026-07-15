from __future__ import annotations

import ast
import copy
import csv
import json
from pathlib import Path
import random

import pytest

from assumption_agent.benchmarks import financial_sec13f_contract_operator_v2 as op


SOURCE_RECEIPT = "a" * 64


def _asset() -> dict[str, object]:
    return op.build_contract_asset_v2(
        candidate_skill_source_receipt_hash=SOURCE_RECEIPT
    )


def _instruction(
    previous: Path,
    current: Path,
    *,
    aum_manager: str = "Alpha Capital",
    increase_manager: str = "Beta Partners",
    issuer: str = "ACME, Inc.",
    four_questions: bool = True,
    increase_top_k: int = 5,
    manager_top_k: int = 3,
) -> str:
    header = (
        "You are a financial analyst comparing official SEC Form 13F data "
        "for 2026Q1 against 2025Q4. The previous data is in "
        f"`{previous}` and current data is in `{current}`.\n\n"
        f"Frozen data semantics: {op._semantic_contract_text()}\n\n"
        "Questions:\n\n"
    )
    if four_questions:
        questions = (
            f"1. What is the current-period AUM of {aum_manager}?\n\n"
            f"2. How many stock rows are held by {aum_manager} in the current period?\n\n"
            f"3. What are the top {increase_top_k} CUSIPs with increased "
            f"investment by {increase_manager} from the previous period to "
            "the current period, ranked by dollar-value increase?\n\n"
            f"4. Which top {manager_top_k} fund managers hold {issuer} in the "
            "current period, ranked by aggregate position value?\n\n"
            "Write `/root/answers.json` with keys `q1_answer`, `q2_answer`, "
            "`q3_answer`, and `q4_answer` in that order. q1 and q2 are numbers; "
            "q3 and q4 are ordered JSON arrays.\n"
        )
    else:
        questions = (
            f"1. What is the current-period AUM of {aum_manager}?\n\n"
            f"2. What are the top {increase_top_k} CUSIPs with increased "
            f"investment by {increase_manager} from the previous period to "
            "the current period, ranked by dollar-value increase?\n\n"
            f"3. Which top {manager_top_k} fund managers hold {issuer} in the "
            "current period, ranked by aggregate position value?\n\n"
            "Write `/root/answers.json` with keys `q1_answer`, `q2_answer`, and "
            "`q3_answer` in that order. q1 is a number; q2 and q3 are ordered "
            "JSON arrays.\n"
        )
    return header + questions


def _write_tsv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _write_period(
    root: Path,
    cover: list[dict[str, str]],
    info: list[dict[str, str]],
) -> None:
    _write_tsv(root / "COVERPAGE.tsv", sorted(op.COVERPAGE_COLUMNS), cover)
    _write_tsv(root / "INFOTABLE.tsv", sorted(op.INFOTABLE_COLUMNS), info)


def _cover(
    accession: str,
    manager: str,
    *,
    date: str,
    report_type: str = "13F-HR",
) -> dict[str, str]:
    return {
        "ACCESSION_NUMBER": accession,
        "REPORTCALENDARORQUARTER": date,
        "REPORTTYPE": report_type,
        "FILINGMANAGER_NAME": manager,
    }


def _info(
    accession: str,
    issuer: str,
    title: str,
    cusip: str,
    value: str,
) -> dict[str, str]:
    return {
        "ACCESSION_NUMBER": accession,
        "NAMEOFISSUER": issuer,
        "TITLEOFCLASS": title,
        "CUSIP": cusip,
        "VALUE": value,
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    previous = tmp_path / "previous"
    current = tmp_path / "current"
    previous_cover = [
        _cover("OLD-B", "Beta Partners", date="30-SEP-2025"),
        _cover("PB", "Ｂｅｔａ－Ｐａｒｔｎｅｒｓ", date="31-DEC-2025"),
        _cover("PN", "Noise Manager", date="31-DEC-2025", report_type="13F-NT NOTICE"),
    ]
    current_cover = [
        _cover("OLD-A", "Alpha Capital", date="31-DEC-2025"),
        _cover("A1", "ALPHA-CAPITAL", date="31-MAR-2026"),
        _cover("B1", "beta partners", date="31-MAR-2026"),
        _cover("G1", "Gamma & Co.", date="31-MAR-2026"),
        _cover("G2", "GAMMA & CO", date="31-MAR-2026"),
        _cover("D1", "Delta Fund", date="31-MAR-2026"),
        _cover("N1", "Notice Giant", date="31-MAR-2026", report_type="NOTICE"),
    ]
    previous_info = [
        _info("OLD-B", "Old", "COM", "0001", "999999"),
        _info("PB", "One", "COM", "0001", "10"),
        _info("PB", "Two", "ADR", "00 02", "2"),
        _info("PB", "Three", "COM", "0003", "10"),
        _info("PN", "Noise", "COM", "0001", "999999"),
    ]
    current_info = [
        _info("OLD-A", "Old", "COM", "Z", "999999"),
        _info("A1", "Alpha Asset", "COM", "A01", "1,000.10"),
        _info("A1", "Alpha Asset", "ADR", "A01", "2.20"),
        _info("A1", "Alpha Asset", "BOND", "A02", "3.70"),
        _info("B1", "One", "COM", "0001", "15"),
        _info("B1", "Two", "COM", "0002", "4"),
        _info("B1", "Two", "COM", "00 02", "3"),
        _info("B1", "Three", "COMMON STOCK", "0003", "5"),
        _info("A1", "ＡＣＭＥ， ＩＮＣ．", "BOND", "C1", "3"),
        _info("G1", "Acme Inc", "BOND", "C1", "6"),
        _info("G2", "ACME-INC", "BOND", "C1", "5"),
        _info("D1", "ACME INC", "BOND", "C1", "10"),
        _info("D1", "ACME INC", "BOND", "C2", "23"),
        _info("N1", "ACME INC", "BOND", "C2", "999999"),
    ]
    _write_period(previous, previous_cover, previous_info)
    _write_period(current, current_cover, current_info)
    return previous, current


def test_contract_operator_executes_exact_4q_semantics(tmp_path: Path) -> None:
    previous, current = _fixture(tmp_path)
    asset = _asset()
    plan, extraction = op.build_contract_plan_v2(
        _instruction(previous, current), asset
    )
    output = tmp_path / "answers.json"
    output.write_text('{"stale": true}\n', encoding="utf-8")
    receipt_path = tmp_path / "receipt.json"

    answers, receipt = op.execute_contract_plan_v2(
        plan,
        previous,
        current,
        output,
        receipt_path,
        asset=asset,
    )

    assert answers == {
        "q1_answer": 1009,
        "q2_answer": 2,
        "q3_answer": ["0001", "0002"],
        "q4_answer": ["GAMMA & CO", "Delta Fund", "ALPHA-CAPITAL"],
    }
    assert json.loads(output.read_text(encoding="utf-8")) == answers
    assert extraction["model_calls"] == extraction["online_calls"] == 0
    assert receipt["pre_output_exists"] is True
    assert receipt["output_changed"] is True
    assert receipt["answers_payload_persisted_in_receipt"] is False
    assert len(receipt["input_file_receipts"]) == 4
    assert receipt["input_set_hash"] == op.payload_hash(
        receipt["input_file_receipts"]
    )
    serialized = json.dumps(receipt, sort_keys=True)
    assert "q1_answer" not in serialized
    assert "Gamma" not in serialized
    assert "resolved_" not in serialized
    assert "holder_count" not in serialized
    assert receipt_path.is_file()


def test_three_question_template_and_root_binding(tmp_path: Path) -> None:
    previous, current = _fixture(tmp_path)
    asset = _asset()
    plan, _ = op.build_contract_plan_v2(
        _instruction(
            previous,
            current,
            four_questions=False,
            increase_top_k=3,
            manager_top_k=5,
        ),
        asset,
    )
    assert [row["operation"] for row in plan["operations"]] == [
        "current_aum",
        "positive_delta_cusip_rank",
        "current_holder_manager_rank",
    ]
    answers, _ = op.execute_contract_plan_v2(
        plan, previous, current, tmp_path / "three.json", asset=asset
    )
    assert list(answers) == ["q1_answer", "q2_answer", "q3_answer"]
    with pytest.raises(op.FinancialSec13FContractError, match="root binding"):
        op.execute_contract_plan_v2(
            plan, current, previous, tmp_path / "wrong.json", asset=asset
        )


def test_normalization_is_nfkc_exact_and_not_fuzzy() -> None:
    assert op.normalize_name("ＡＣＭＥ， ＩＮＣ．") == "acme inc"
    assert op.normalize_name("A.C.M.E.") == "a c m e"
    assert op.normalize_name("ACME") == "acme"
    assert op.normalize_name("A.C.M.E.") != op.normalize_name("ACME")
    assert op.normalize_title_class("  sponsored\u3000adr ") == "SPONSORED ADR"
    assert op.canonical_cusip(" 00\t01 ") == "0001"


def test_all_25_stock_classes_are_independent_rows(tmp_path: Path) -> None:
    previous = tmp_path / "previous"
    current = tmp_path / "current"
    cover_previous = [_cover("PB", "Beta", date="31-DEC-2025")]
    cover_current = [
        _cover("A", "Alpha", date="31-MAR-2026"),
        _cover("B", "Beta", date="31-MAR-2026"),
        _cover("H", "Holder", date="31-MAR-2026"),
    ]
    previous_info = [_info("PB", "Prior", "COM", "P", "1")]
    current_info = [
        _info("A", "Asset", title.lower(), f"S{index:02d}", "1")
        for index, title in enumerate(sorted(op.STOCK_TITLE_CLASSES))
    ]
    current_info.extend(
        [
            _info("A", "Asset", "SPONSORED ADREQUITY", "NO", "1"),
            _info("B", "Delta", "COM", "P", "2"),
            _info("H", "Issuer", "BOND", "I", "3"),
        ]
    )
    _write_period(previous, cover_previous, previous_info)
    _write_period(current, cover_current, current_info)
    asset = _asset()
    plan, _ = op.build_contract_plan_v2(
        _instruction(
            previous,
            current,
            aum_manager="Alpha",
            increase_manager="Beta",
            issuer="Issuer",
        ),
        asset,
    )
    answers, _ = op.execute_contract_plan_v2(
        plan, previous, current, tmp_path / "ontology.json", asset=asset
    )
    assert answers["q2_answer"] == 25


def test_latest_is_selected_before_notice_filter_without_fallback(tmp_path: Path) -> None:
    previous, current = _fixture(tmp_path)
    _write_period(
        current,
        [
            _cover("OLD", "Alpha Capital", date="31-MAR-2026"),
            _cover("NEW", "Alpha Capital", date="30-JUN-2026", report_type="NOTICE"),
        ],
        [_info("OLD", "ACME Inc", "COM", "C", "1")],
    )
    asset = _asset()
    plan, _ = op.build_contract_plan_v2(
        _instruction(previous, current), asset
    )
    with pytest.raises(op.FinancialSec13FContractError, match="no non-NOTICE"):
        op.execute_contract_plan_v2(
            plan, previous, current, tmp_path / "notice.json", asset=asset
        )


def test_target_manager_must_have_one_accession(tmp_path: Path) -> None:
    previous, current = _fixture(tmp_path)
    cover_rows = list(csv.DictReader(
        (current / "COVERPAGE.tsv").open(encoding="utf-8"), delimiter="\t"
    ))
    cover_rows.append(_cover("A2", "Alpha Capital", date="31-MAR-2026"))
    info_rows = list(csv.DictReader(
        (current / "INFOTABLE.tsv").open(encoding="utf-8"), delimiter="\t"
    ))
    info_rows.append(_info("A2", "Other", "COM", "X", "1"))
    _write_period(current, cover_rows, info_rows)
    asset = _asset()
    plan, _ = op.build_contract_plan_v2(
        _instruction(previous, current), asset
    )
    with pytest.raises(op.FinancialSec13FContractError, match="not uniquely"):
        op.execute_contract_plan_v2(
            plan, previous, current, tmp_path / "duplicate.json", asset=asset
        )


@pytest.mark.parametrize("bad", ["", "-1", "NaN", "Infinity", "not-a-number"])
def test_invalid_sec_values_fail_closed(tmp_path: Path, bad: str) -> None:
    previous, current = _fixture(tmp_path)
    rows = list(csv.DictReader(
        (current / "INFOTABLE.tsv").open(encoding="utf-8"), delimiter="\t"
    ))
    next(row for row in rows if row["ACCESSION_NUMBER"] == "A1")["VALUE"] = bad
    cover = list(csv.DictReader(
        (current / "COVERPAGE.tsv").open(encoding="utf-8"), delimiter="\t"
    ))
    _write_period(current, cover, rows)
    asset = _asset()
    plan, _ = op.build_contract_plan_v2(
        _instruction(previous, current), asset
    )
    with pytest.raises(op.FinancialSec13FContractError, match="SEC VALUE"):
        op.execute_contract_plan_v2(
            plan, previous, current, tmp_path / "bad.json", asset=asset
        )


def test_self_consistent_asset_semantic_forgery_fails_closed(
    tmp_path: Path,
) -> None:
    previous, current = _fixture(tmp_path)
    instruction = _instruction(previous, current)
    original = _asset()
    mutations = (
        ("candidate_id", "b" * 64),
        ("contract_hash", "c" * 64),
        ("template_grammar_hash", "d" * 64),
        ("excluded_inputs", ["gold_or_expected_answer"]),
    )
    for field, value in mutations:
        forged = copy.deepcopy(original)
        forged[field] = value
        body = dict(forged)
        body.pop("manifest_hash")
        forged["manifest_hash"] = op.payload_hash(body)
        with pytest.raises(op.FinancialSec13FContractError):
            op.build_contract_plan_v2(instruction, forged)
    forged = copy.deepcopy(original)
    forged["unexpected"] = True
    body = dict(forged)
    body.pop("manifest_hash")
    forged["manifest_hash"] = op.payload_hash(body)
    with pytest.raises(op.FinancialSec13FContractError, match="fields drifted"):
        op.build_contract_plan_v2(instruction, forged)


def test_output_and_receipt_symlinks_fail_closed(tmp_path: Path) -> None:
    previous, current = _fixture(tmp_path)
    asset = _asset()
    plan, _ = op.build_contract_plan_v2(
        _instruction(previous, current), asset
    )
    sentinel = tmp_path / "sentinel.txt"
    sentinel.write_text("untouched\n", encoding="utf-8")
    output_link = tmp_path / "answers-link.json"
    output_link.symlink_to(sentinel)
    with pytest.raises(op.FinancialSec13FContractError, match="symlink"):
        op.execute_contract_plan_v2(
            plan, previous, current, output_link, asset=asset
        )
    assert sentinel.read_text(encoding="utf-8") == "untouched\n"

    receipt_link = tmp_path / "receipt-link.json"
    receipt_link.symlink_to(sentinel)
    with pytest.raises(op.FinancialSec13FContractError, match="symlink"):
        op.execute_contract_plan_v2(
            plan,
            previous,
            current,
            tmp_path / "safe-answer.json",
            receipt_link,
            asset=asset,
        )
    assert sentinel.read_text(encoding="utf-8") == "untouched\n"


def test_instruction_and_plan_drift_fail_closed(tmp_path: Path) -> None:
    previous, current = _fixture(tmp_path)
    asset = _asset()
    instruction = _instruction(previous, current)
    with pytest.raises(op.FinancialSec13FContractError, match="contract drifted"):
        op.build_contract_plan_v2(
            instruction.replace("CUSIP ascending", "CUSIP descending"), asset
        )
    plan, _ = op.build_contract_plan_v2(instruction, asset)
    plan["operations"][0]["entity"] = "Fuzzy decoy"
    with pytest.raises(op.FinancialSec13FContractError, match="self hash"):
        op.validate_contract_plan_v2(plan, asset)


def test_row_order_does_not_change_answers(tmp_path: Path) -> None:
    previous, current = _fixture(tmp_path)
    asset = _asset()
    plan, _ = op.build_contract_plan_v2(
        _instruction(previous, current), asset
    )
    first, _ = op.execute_contract_plan_v2(
        plan, previous, current, tmp_path / "first.json", asset=asset
    )
    rng = random.Random(17)
    for root in (previous, current):
        rows = list(csv.DictReader(
            (root / "INFOTABLE.tsv").open(encoding="utf-8"), delimiter="\t"
        ))
        rng.shuffle(rows)
        cover = list(csv.DictReader(
            (root / "COVERPAGE.tsv").open(encoding="utf-8"), delimiter="\t"
        ))
        rng.shuffle(cover)
        _write_period(root, cover, rows)
    second, _ = op.execute_contract_plan_v2(
        plan, previous, current, tmp_path / "second.json", asset=asset
    )
    assert second == first


def test_operator_source_has_no_oracle_verifier_or_network_import() -> None:
    source_path = Path(op.__file__).resolve()
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    forbidden = {
        "requests",
        "httpx",
        "urllib",
        "replication_runtime",
        "offline_verifier",
        "oracle_pandas",
        "oracle_streaming",
    }
    assert not any(
        name == blocked or name.startswith(blocked + ".")
        for name in imported
        for blocked in forbidden
    )
    source = source_path.read_text(encoding="utf-8").casefold()
    assert "expected_output.json" not in source
    descriptor = json.dumps(_asset()["contract_descriptor"]).casefold()
    assert "measurement" not in descriptor
    assert "sealed" not in descriptor
    assert "oracle" not in descriptor
    assert "gold" not in descriptor
