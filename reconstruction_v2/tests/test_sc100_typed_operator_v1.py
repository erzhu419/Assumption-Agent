from __future__ import annotations

from pathlib import Path

import pytest

from assumption_agent.benchmarks.sc100_typed_operator_v1 import (
    OPERATOR_VERSION,
    SC100OperatorError,
    compile_plan,
    execute,
    parse_instruction,
)


SYNTHETIC_OVER_LIMIT = """
Fill the California Small Claims Court form at /root/sc100-blank.pdf.
Case Description: I am Avery Stone. I live at 100 Cedar Way Apt 2,
Fremont, CA 94536. My phone number is 510-555-0101, and my email is
avery.stone@example.com. I want to sue Jordan Reed, who lives at 200 Birch
Road, Oakland, CA 94607. Their phone number is (510) 555-0102. They failed to
return my $3,100 security deposit under our signed room rental agreement after
move-out. This issue took place from 2026-01-02 to 2026-02-03. I requested
payment multiple times through text messages, but received no response. The
amount is documented in the signed room rental agreement. I am filing in the
location where the defendant lives. I have filed more than 12 other small
claims in California within the last 12 months. Use February 3, 2026 as the
declaration signing date.
"""


SYNTHETIC_FIRST_CASE = """
Fill the California Small Claims Court form at /root/sc100-blank.pdf.
Case Description: My name is Morgan Hale, and I live at 9 Pine Street,
San Jose, CA 95112. My phone number is 4085550103 and my email is
morgan.hale@example.org. I am suing Casey Park, who can be reached at
4085550104 and lives at 31 Palm Avenue Apt 4, Campbell, CA 95008. Under our
signed roommate sublease contract, they did not return my $1,200 security
deposit after moving out. The dispute period is from 2026-03-01 to 2026-04-05.
I sent text messages asking for repayment, but received no reply. The claimed
amount is documented in the signed roommate sublease contract. This is my first
time filing a small claims case. File in the venue where the defendant lives.
Please file it with date: April 5, 2026.
"""


def test_parse_and_compile_closed_grammar() -> None:
    facts = parse_instruction(SYNTHETIC_OVER_LIMIT)
    assert facts.plaintiff_name == "Avery Stone"
    assert facts.defendant_name == "Jordan Reed"
    assert facts.plaintiff_phone == "5105550101"
    assert facts.defendant_phone == "5105550102"
    assert facts.claim_amount == 3100
    assert facts.more_than_twelve_claims is True

    plan = compile_plan(facts)
    assert plan.operator_version == OPERATOR_VERSION
    assert len(plan.mutations) == 30
    assert len({mutation.field_name for mutation in plan.mutations}) == 30
    selected = {mutation.field_name: mutation.value for mutation in plan.mutations}
    assert selected[
        "SC-100[0].Page4[0].List9[0].Item9[0].Checkbox62[0]"
    ] == "1"
    assert selected[
        "SC-100[0].Page4[0].List10[0].li10[0].Checkbox63[0]"
    ] == "1"


def test_first_case_and_under_limit_select_no_options() -> None:
    facts = parse_instruction(SYNTHETIC_FIRST_CASE)
    assert facts.more_than_twelve_claims is False
    assert facts.claim_amount == 1200
    selected = {
        mutation.field_name: mutation.value for mutation in compile_plan(facts).mutations
    }
    assert selected[
        "SC-100[0].Page4[0].List9[0].Item9[0].Checkbox62[1]"
    ] == "2"
    assert selected[
        "SC-100[0].Page4[0].List10[0].li10[0].Checkbox63[1]"
    ] == "2"


@pytest.mark.parametrize(
    "replacement",
    [
        "I am suing the City of Oakland, a public entity",
        "This is an attorney fee dispute",
        "Use February 4, 2026 as the declaration signing date",
        "I have filed one prior case",
    ],
)
def test_unsupported_or_ambiguous_cases_fail_closed(replacement: str) -> None:
    if replacement.startswith("I am suing"):
        instruction = SYNTHETIC_OVER_LIMIT.replace(
            "I want to sue Jordan Reed", replacement
        )
    elif replacement.startswith("This is an attorney"):
        instruction = SYNTHETIC_OVER_LIMIT + replacement
    elif replacement.startswith("Use February"):
        instruction = SYNTHETIC_OVER_LIMIT.replace(
            "February 3, 2026", "February 4, 2026"
        )
    else:
        instruction = SYNTHETIC_OVER_LIMIT.replace(
            "more than 12", "one"
        )
    with pytest.raises(SC100OperatorError):
        parse_instruction(instruction)


def test_synthetic_pdf_write_and_reopen_reconciliation(tmp_path: Path) -> None:
    pytest.importorskip("fitz")
    project = Path(__file__).resolve().parents[1]
    blank = project / (
        "reference/self_evo_continual_20260707/repos/SkillLearnBench/tasks/"
        "court-form-filling/court-form-filling-3/environment/sc100-blank.pdf"
    )
    if not blank.is_file():
        pytest.skip("local public SC-100 blank is unavailable")

    output = tmp_path / "sc100-filled.pdf"
    receipt = execute(
        instruction=SYNTHETIC_FIRST_CASE,
        blank_pdf=blank,
        output_pdf=output,
    )
    assert output.is_file()
    assert receipt["operator_version"] == OPERATOR_VERSION
    assert receipt["page_count"] == 6
    assert receipt["field_count"] == 103
    assert receipt["mutation_count"] == 30
    assert receipt["source_unchanged"] is True
    assert receipt["field_structure_preserved"] is True
    assert receipt["exact_mutation_set_reconciled"] is True
    assert receipt["forbidden_fields_empty"] is True
    assert receipt["required_visible_values_present"] is True
    assert receipt["raw_case_text_persisted"] is False
