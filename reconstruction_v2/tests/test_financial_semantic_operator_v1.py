from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from assumption_agent.benchmarks import financial_semantic_operator_v1 as operator
from assumption_agent.benchmarks.financial_semantic_integration_v1 import (
    _FinancialVerifierProxyV1,
)
from assumption_agent.benchmarks.financial_semantic_operator_v1 import (
    EMBEDDING_DIMENSION,
    FINANCIAL_SEMANTIC_ASSET_VERSION,
    FINANCIAL_SEMANTIC_OPERATOR_VERSION,
    FINANCIAL_SEMANTIC_PLAN_VERSION,
    OPERATION_ORDER,
    OPERATION_PROTOTYPES,
    build_financial_semantic_plan,
    execute_financial_semantic_plan,
    split_numbered_question_blocks,
)


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _semantic_asset() -> dict[str, object]:
    examples = [
        {
            "item_id": "formation",
            "question_index": index,
            "operation": operation,
            "text": f"formation {operation}",
            "text_sha256": hashlib.sha256(
                f"formation {operation}".encode()
            ).hexdigest(),
        }
        for index, operation in enumerate(OPERATION_ORDER, start=1)
    ]
    return {
        "asset_version": FINANCIAL_SEMANTIC_ASSET_VERSION,
        "operator_version": FINANCIAL_SEMANTIC_OPERATOR_VERSION,
        "candidate_id": "a" * 64,
        "manifest_hash": "b" * 64,
        "operator_source_sha256": "c" * 64,
        "minilm_runtime_asset_manifest_hash": "d" * 64,
        "qa_runtime_asset_manifest_hash": "e" * 64,
        "train_examples": examples,
    }


def _intent(text: str) -> int:
    lowered = text.lower()
    if "assets under management" in lowered or "aum" in lowered:
        return 0
    if "count" in lowered or "how many stocks" in lowered:
        return 1
    if "increase" in lowered:
        return 2
    if "manager" in lowered:
        return 3
    raise AssertionError(text)


def _encoder(texts: tuple[str, ...] | list[str]) -> np.ndarray:
    matrix = np.zeros((len(texts), EMBEDDING_DIMENSION), dtype=np.float32)
    for row, text in enumerate(texts):
        operation = next(
            (
                operation
                for operation in OPERATION_ORDER
                if operation in text
            ),
            None,
        )
        index = OPERATION_ORDER.index(operation) if operation else _intent(text)
        matrix[row, index] = 1.0
    return matrix


def test_train_semantic_decoder_builds_one_to_one_typed_plan() -> None:
    instruction = """Questions:
1. What is the AUM of Alpha Partners?
2. How many stocks are held by Alpha?
3. Which top two stocks received an investment increase by Beta Capital?
4. List top-3 managers invested in Acme Devices.

Format your answer as JSON.
"""
    answers = {
        "What full name follows 'AUM of'?": "Alpha Partners",
        "Which fund holds the stocks?": "Alpha",
        "Which fund increased investment?": "Beta Capital",
        "What company did the requested fund managers invest in?": (
            "Acme Devices"
        ),
    }

    plan, receipt = build_financial_semantic_plan(
        instruction=instruction,
        asset=_semantic_asset(),
        encoder=_encoder,
        qa=lambda question, _: (answers[question], 7.0),
    )

    assert [row["operation"] for row in plan["operations"]] == list(
        OPERATION_ORDER
    )
    assert plan["operations"][1]["entity"] == "Alpha Partners"
    assert plan["operations"][1]["entity_resolution"] == (
        "same_item_alias_to_q3_aum"
    )
    assert plan["operations"][2]["top_k"] == 2
    assert plan["operations"][3]["top_k"] == 3
    assert receipt["operator_created_raw_instruction_artifact"] is False
    assert receipt["online_calls"] == 0


def test_numbered_question_split_excludes_output_schema() -> None:
    blocks = split_numbered_question_blocks(
        "Questions:\n1. First request.\n\n2. Second request.\n\n"
        "3. Third request.\n\nFormat your answer as JSON with q1."
    )
    assert blocks == ("First request.", "Second request.", "Third request.")


def _write_table(root: Path, name: str, rows: list[dict[str, object]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(root / name, sep="\t", index=False)


def _plan() -> dict[str, object]:
    entities = (
        ("q3_aum", "Alpha Partners", None),
        ("q3_stock_count", "Alpha Partners", None),
        ("quarter_increase_rank", "Beta Capital", 2),
        ("q3_manager_rank", "Acme Devices", 2),
    )
    operations = [
        {
            "question_index": index,
            "answer_key": f"q{index}_answer",
            "operation": operation,
            "entity": entity,
            "entity_sha256": hashlib.sha256(entity.encode()).hexdigest(),
            "top_k": top_k,
            "question_block_sha256": f"{index}" * 64,
            "semantic_score": 1.0,
            "qa_score": 1.0,
            "entity_resolution": "direct_qa_span",
        }
        for index, (operation, entity, top_k) in enumerate(entities, start=1)
    ]
    plan: dict[str, object] = {
        "plan_version": FINANCIAL_SEMANTIC_PLAN_VERSION,
        "operator_version": FINANCIAL_SEMANTIC_OPERATOR_VERSION,
        "candidate_id": "a" * 64,
        "candidate_manifest_hash": "b" * 64,
        "instruction_sha256": "c" * 64,
        "question_block_count": 4,
        "operations": operations,
        "operation_set_hash": _hash(operations),
        "minilm_runtime_asset_manifest_hash": "d" * 64,
        "qa_runtime_asset_manifest_hash": "e" * 64,
        "operator_source_sha256": hashlib.sha256(
            Path(operator.__file__).read_bytes()
        ).hexdigest(),
        "online_calls": 0,
        "raw_instruction_persisted": False,
    }
    plan["plan_hash"] = _hash(plan)
    return plan


def test_typed_query_engine_reconciles_invented_13f_fixture(
    tmp_path: Path,
) -> None:
    q2 = tmp_path / "q2"
    q3 = tmp_path / "q3"
    cover_columns = [
        "ACCESSION_NUMBER",
        "REPORTCALENDARORQUARTER",
        "FILINGMANAGER_NAME",
        "REPORTTYPE",
    ]
    cover2 = [
        dict(
            zip(
                cover_columns,
                ("beta-q2", "30-JUN-2025", "Beta Capital LLC", "13F HOLDINGS REPORT"),
            )
        )
    ]
    cover3 = [
        dict(
            zip(
                cover_columns,
                ("alpha-q3", "30-SEP-2025", "Alpha Partners LLC", "13F HOLDINGS REPORT"),
            )
        ),
        dict(
            zip(
                cover_columns,
                ("beta-q3", "30-SEP-2025", "Beta Capital LLC", "13F HOLDINGS REPORT"),
            )
        ),
        dict(
            zip(
                cover_columns,
                ("m1-q3", "30-SEP-2025", "Manager One", "13F HOLDINGS REPORT"),
            )
        ),
        dict(
            zip(
                cover_columns,
                ("m2-q3", "30-SEP-2025", "Manager Two", "13F HOLDINGS REPORT"),
            )
        ),
        dict(
            zip(
                cover_columns,
                ("noise", "30-SEP-2025", "Alpha Partners Notice", "13F NOTICE"),
            )
        ),
    ]
    _write_table(q2, "COVERPAGE.tsv", cover2)
    _write_table(q3, "COVERPAGE.tsv", cover3)
    info_columns = (
        "ACCESSION_NUMBER",
        "NAMEOFISSUER",
        "TITLEOFCLASS",
        "CUSIP",
        "VALUE",
    )

    def info(*values: object) -> dict[str, object]:
        return dict(zip(info_columns, values))

    _write_table(
        q2,
        "INFOTABLE.tsv",
        [
            info("beta-q2", "Issuer A", "COM", "A", 10),
            info("beta-q2", "Issuer B", "COM", "B", 20),
        ],
    )
    _write_table(
        q3,
        "INFOTABLE.tsv",
        [
            info("alpha-q3", "Issuer D", "COM", "D", 100),
            info("alpha-q3", "Issuer E", "BOND", "E", 200),
            info("beta-q3", "Issuer A", "COM", "A", 40),
            info("beta-q3", "Issuer C", "COM", "C", 50),
            info("m1-q3", "ACME DEVICES", "COM", "X", 100),
            info("m2-q3", "ACME DEVICES", "COM", "X", 200),
        ],
    )
    output = tmp_path / "answers.json"
    answers, receipt = execute_financial_semantic_plan(
        plan=_plan(),
        q2_root=q2,
        q3_root=q3,
        output_path=output,
        receipt_path=tmp_path / "receipt.json",
    )

    assert answers == {
        "q1_answer": 300.0,
        "q2_answer": 1,
        "q3_answer": ["C", "A"],
        "q4_answer": ["Manager Two", "Manager One"],
    }
    assert json.loads(output.read_text()) == answers
    assert receipt["network_calls"] == 0
    assert receipt["verifier_content_accessed"] is False


def test_verifier_proxy_executes_typed_plan_before_test_command() -> None:
    order: list[str] = []

    class Delegate:
        def run(self, command: list[str], *_: object, **__: object) -> str:
            order.append("delegate:" + " ".join(command))
            return "completed"

    class Backend:
        def _execute_financial_plan_before_verifier_v1(
            self, *, delegate: object, container_name: str
        ) -> None:
            assert isinstance(delegate, Delegate)
            assert container_name == "container-1"
            order.append("typed-plan")

    proxy = _FinancialVerifierProxyV1(Delegate(), backend=Backend())
    result = proxy.run(
        ["docker", "exec", "container-1", "/bin/bash", "/tests/test.sh"]
    )

    assert result == "completed"
    assert order == [
        "typed-plan",
        "delegate:docker exec container-1 /bin/bash /tests/test.sh",
    ]
