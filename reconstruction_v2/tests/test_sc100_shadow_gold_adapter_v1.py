from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from assumption_agent.benchmarks.sc100_shadow_gold_adapter_v1 import (
    ShadowGoldValidationError,
    adapt_shadow_corpus,
    adapt_shadow_record,
    load_shadow_gold_jsonl,
)


CORPUS_ROOT = Path(__file__).parents[1] / "reference" / "synthetic_sc100_shadow_v1"
GOLD_PATH = CORPUS_ROOT / "gold.jsonl"


def _raw_rows() -> list[dict]:
    return [json.loads(line) for line in GOLD_PATH.read_text(encoding="utf-8").splitlines()]


def _raw(case_id: str) -> dict:
    return next(row for row in _raw_rows() if row["case_id"] == case_id)


def test_complete_frozen_corpus_adapts_without_inference() -> None:
    adapted = load_shadow_gold_jsonl(GOLD_PATH)
    assert len(adapted) == 24
    assert sum(row.oracle_gold is not None for row in adapted) == 18
    assert sum(row.rejection_reason is not None for row in adapted) == 6
    assert len({row.case_id for row in adapted}) == 24


def test_boundary_and_probe_semantics_are_preserved() -> None:
    rows = {row.case_id: row for row in load_shadow_gold_jsonl(GOLD_PATH)}
    assert rows["S03"].oracle_gold["claim"]["amount"] == "2500"
    assert rows["S03"].oracle_gold["questions"]["more_than_2500"] is False
    assert rows["S07"].oracle_gold["questions"]["more_than_12_claims"] is True
    assert rows["C02"].oracle_gold["signature"]["date"] == "2025-06-07"
    assert rows["C02"].oracle_gold["claim"]["end_date"] == "2025-05-30"
    assert rows["C04"].oracle_gold["venue"]["zip"] == "94610"
    assert rows["C04"].oracle_gold["defendant"]["zip"] == "95814"
    assert rows["C04"].oracle_gold["questions"]["venue_choice"] == 1
    assert rows["C05"].oracle_gold["plaintiff"]["phone"] == "3235550155"
    assert rows["C06"].oracle_gold["plaintiff"]["name"] == "Ana-María O'Neill"


def test_rejection_reason_is_exact_and_n02_precedence_is_frozen() -> None:
    row = adapt_shadow_record(_raw("N02"))
    assert row.oracle_gold is None
    assert row.rejection_reason == "attorney_fee_dispute"
    mutated = _raw("N02")
    mutated["reason_code"] = "unsupported_claim_type"
    with pytest.raises(ShadowGoldValidationError) as exc_info:
        adapt_shadow_record(mutated)
    assert "inconsistent:reason_code" in exc_info.value.codes


@pytest.mark.parametrize(
    ("mutator", "expected_code"),
    [
        (
            lambda row: row["plaintiff"]["address"].update({"county": "Alameda"}),
            "extra:plaintiff.address.county",
        ),
        (
            lambda row: row["claim"].update({"amount_cents": 250_001}),
            "invalid:claim.amount_cents_precision",
        ),
        (
            lambda row: row.update({"more_than_12_other_small_claims": True}),
            "inconsistent:more_than_12_other_small_claims",
        ),
        (
            lambda row: row["signature"].update({"name": "Someone Else"}),
            "inconsistent:signature.name",
        ),
        (
            lambda row: row["expected_form_semantics"].update({"more_than_2500": "yes"}),
            "inconsistent:expected_form_semantics",
        ),
    ],
)
def test_invalid_latent_rows_fail_closed(mutator, expected_code: str) -> None:
    row = deepcopy(_raw("S01"))
    mutator(row)
    with pytest.raises(ShadowGoldValidationError) as exc_info:
        adapt_shadow_record(row)
    assert expected_code in exc_info.value.codes


def test_contract_venue_cannot_be_silently_replaced_by_defendant_zip() -> None:
    row = deepcopy(_raw("C04"))
    row["venue"]["form_zip"] = row["defendant"]["address"]["zip"]
    row["expected_form_semantics"]["venue_selection"] = "defendant_residence_only"
    with pytest.raises(ShadowGoldValidationError) as exc_info:
        adapt_shadow_record(row)
    assert "inconsistent:expected_form_semantics" in exc_info.value.codes


def test_corpus_set_and_count_are_bound() -> None:
    rows = _raw_rows()
    with pytest.raises(ShadowGoldValidationError) as exc_info:
        adapt_shadow_corpus(rows[:-1])
    assert "corpus:case_id_set" in exc_info.value.codes
    assert "corpus:row_count" in exc_info.value.codes
