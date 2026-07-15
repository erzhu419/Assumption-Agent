from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from assumption_agent.benchmarks import sc100_typed_operator_v2 as _operator
from assumption_agent.benchmarks.sc100_typed_operator_v2 import (
    OPERATOR_VERSION,
    PUBLIC_BLANK_SHA256,
    compile_plan,
    execute,
    parse_instruction,
)


PROJECT = Path(__file__).resolve().parents[1]
PUBLIC_BLANK = PROJECT / (
    "reference/self_evo_continual_20260707/repos/SkillLearnBench/tasks/"
    "court-form-filling/court-form-filling-3/environment/sc100-blank.pdf"
)


def _instruction(
    *,
    amount: str = "2,500",
    claims: str = "I have filed 12 other small claims.",
    identities_reversed: bool = False,
    defendant_phone_first: bool = False,
    backup_phone: bool = True,
    venue: str = "File in the venue where the defendant lives.",
    request: str = (
        "I emailed a request asking the defendant to return the deposit, "
        "but no payment arrived."
    ),
    event: str = "The rental period was from May 1, 2026 through June 2, 2026.",
    signature: str = "Use July 3, 2026 as the declaration signing date.",
    extra: str = "",
) -> str:
    plaintiff_phone = "The plaintiff primary phone is (510) 555-1010"
    if backup_phone:
        plaintiff_phone += ", the backup phone is 510-555-1099"
    plaintiff = (
        "Plaintiff's name is Renée O'Neil-Santos, and the plaintiff address is "
        "18 Maple Lane Apt 3, Berkeley, CA 94704. "
        f"{plaintiff_phone}, and my email is renee@example.test."
    )
    if defendant_phone_first:
        defendant = (
            "The proposed defendant's name is Theo Marsh. The defendant phone is "
            "510-555-2020, and the defendant lives at 77 Oak Road, Oakland, CA 94607."
        )
    else:
        defendant = (
            "The proposed defendant's name is Theo Marsh, and the defendant address is "
            "77 Oak Road, Oakland, CA 94607. The defendant phone is 510-555-2020."
        )
    identities = f"{defendant} {plaintiff}" if identities_reversed else f"{plaintiff} {defendant}"
    return " ".join(
        part
        for part in (
            "Complete the California SC-100.",
            identities,
            "This is a rental security deposit claim.",
            (
                "Under our signed room rental agreement, the defendant did not return "
                f"my ${amount} security deposit after move-out."
            ),
            event,
            request,
            venue,
            claims,
            signature,
            extra,
        )
        if part
    )


def _rejection_reason(instruction: str) -> str:
    with pytest.raises(ValueError) as caught:
        parse_instruction(instruction)
    reason = getattr(caught.value, "reason_code", None)
    assert isinstance(reason, str)
    return reason


def _receipt_hash(receipt: dict[str, object]) -> str:
    body = dict(receipt)
    del body["receipt_hash"]
    encoded = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_role_anchored_parse_and_fixed_plan() -> None:
    facts = parse_instruction(_instruction())
    assert facts.plaintiff_name == "Renée O'Neil-Santos"
    assert facts.plaintiff_address.street == "18 Maple Lane Apt 3"
    assert facts.plaintiff_phone == "5105551010"
    assert facts.defendant_name == "Theo Marsh"
    assert facts.defendant_address.street == "77 Oak Road"
    assert facts.defendant_phone == "5105552020"
    assert facts.claim_amount == 2500
    assert facts.incident_start == "2026-05-01"
    assert facts.incident_end == "2026-06-02"
    assert facts.declaration_date == "2026-07-03"
    assert facts.venue_basis == "defendant_residence"
    assert facts.venue_zip == "94607"

    plan = compile_plan(facts)
    assert len(plan.mutations) == 30
    assert len({mutation.field_name for mutation in plan.mutations}) == 30
    assert len(plan.plan_hash) == 64


@pytest.mark.parametrize(
    ("reversed_roles", "phone_first"),
    [(True, False), (False, True), (True, True)],
)
def test_assignment_is_by_role_not_global_occurrence(
    reversed_roles: bool, phone_first: bool
) -> None:
    facts = parse_instruction(
        _instruction(
            identities_reversed=reversed_roles,
            defendant_phone_first=phone_first,
        )
    )
    assert facts.plaintiff_address.zip_code == "94704"
    assert facts.defendant_address.zip_code == "94607"
    assert facts.plaintiff_phone == "5105551010"
    assert facts.defendant_phone == "5105552020"


def test_my_name_and_i_am_suing_do_not_alias_roles() -> None:
    instruction = _instruction().replace(
        "Plaintiff's name is Renée O'Neil-Santos, and the plaintiff address is",
        "I am Renée O'Neil-Santos, and I live at",
    ).replace(
        "The proposed defendant's name is Theo Marsh, and the defendant address is",
        "I am suing Theo Marsh, who lives at",
    )
    facts = parse_instruction(instruction)
    assert facts.plaintiff_name == "Renée O'Neil-Santos"
    assert facts.defendant_name == "Theo Marsh"


def test_phone_prefix_cannot_be_reinterpreted_as_street() -> None:
    malformed = _instruction().replace(
        "The proposed defendant's name is Theo Marsh, and the defendant address is "
        "77 Oak Road, Oakland, CA 94607. The defendant phone is 510-555-2020.",
        "The proposed defendant's name is Theo Marsh. The defendant phone and address "
        "are 510 555 2020 Pine Road, Oakland, CA 94607.",
    )
    assert _rejection_reason(malformed) == "missing_or_ambiguous_required_fact"


def test_defendant_email_cannot_fill_plaintiff_email() -> None:
    defendant_only = _instruction().replace(
        "my email is renee@example.test",
        "the defendant email is theo@example.test",
    )
    assert _rejection_reason(defendant_only) == "missing_or_ambiguous_required_fact"
    both = _instruction().replace(
        "my email is renee@example.test",
        "my email is renee@example.test and the defendant email is theo@example.test",
    )
    assert parse_instruction(both).plaintiff_email == "renee@example.test"


def test_primary_phone_wins_over_backup() -> None:
    facts = parse_instruction(_instruction(backup_phone=True))
    assert facts.plaintiff_phone == "5105551010"


@pytest.mark.parametrize(
    ("amount", "selected_suffix", "selected_value"),
    [("2,499", "Checkbox63[1]", "2"), ("2,500", "Checkbox63[1]", "2"), ("2,501", "Checkbox63[0]", "1")],
)
def test_amount_boundary_compiles_exact_q10_choice(
    amount: str, selected_suffix: str, selected_value: str
) -> None:
    facts = parse_instruction(_instruction(amount=amount))
    selected = {
        mutation.field_name: mutation.value for mutation in compile_plan(facts).mutations
    }
    matching = {
        name: value for name, value in selected.items() if "Checkbox63" in name
    }
    assert len(matching) == 1
    name, value = next(iter(matching.items()))
    assert selected_suffix in name
    assert value == selected_value


@pytest.mark.parametrize(
    ("claims", "expected_over"),
    [
        ("This is my first small claims case.", False),
        ("I have filed 12 other small claims.", False),
        ("I have filed 13 other small claims.", True),
        ("I have filed more than 12 other small claims.", True),
    ],
)
def test_claim_count_boundary(claims: str, expected_over: bool) -> None:
    assert parse_instruction(_instruction(claims=claims)).more_than_twelve_claims is expected_over


def test_extra_demand_date_does_not_replace_event_or_signature_dates() -> None:
    facts = parse_instruction(
        _instruction(
            request=(
                "I emailed a written demand on June 10, 2026 asking the defendant "
                "to return the deposit."
            ),
            signature="Use July 8, 2026 as the declaration signing date.",
        )
    )
    assert (facts.incident_start, facts.incident_end) == ("2026-05-01", "2026-06-02")
    assert facts.declaration_date == "2026-07-08"


def test_contract_venue_is_typed_and_uses_distinct_zip() -> None:
    facts = parse_instruction(
        _instruction(
            venue=(
                "Use the contract venue: the contract was made, performed, and breached "
                "in San Mateo, CA 94401. Do not use defendant residence as the venue basis."
            )
        )
    )
    assert facts.venue_basis == "contract_made_performed_and_breached"
    assert facts.venue_zip == "94401"


def test_contract_venue_is_not_blocked_by_defendant_lives_identity_phrase() -> None:
    facts = parse_instruction(
        _instruction(
            defendant_phone_first=True,
            venue=(
                "Use the contract venue: the contract was made, performed, and breached "
                "in San Mateo, CA 94401."
            ),
        )
    )
    assert facts.venue_basis == "contract_made_performed_and_breached"
    assert facts.venue_zip == "94401"


def test_negated_public_and_attorney_statements_do_not_trigger_rejection() -> None:
    instruction = _instruction() + (
        " The defendant is not a public entity. This is not an attorney fee dispute."
    )
    assert parse_instruction(instruction).defendant_name == "Theo Marsh"


def test_structured_no_for_public_and_attorney_is_not_positive_signal() -> None:
    instruction = _instruction() + (
        " Public entity: no. Attorney fee dispute? No."
    )
    assert parse_instruction(instruction).plaintiff_name == "Renée O'Neil-Santos"


def test_named_municipality_is_public_entity() -> None:
    instruction = _instruction().replace("Theo Marsh", "City of Oakland")
    assert _rejection_reason(instruction) == "public_entity"


@pytest.mark.parametrize(
    "request_phrase",
    [
        "Payment was not requested before filing.",
        "I did not request payment from the defendant.",
    ],
)
def test_payment_not_requested_phrasings_bind_exact_reason(request_phrase: str) -> None:
    assert _rejection_reason(
        _instruction(request=request_phrase)
    ) == "payment_not_requested"


def test_unrelated_rent_amount_does_not_create_claim_conflict() -> None:
    facts = parse_instruction(_instruction(extra="Monthly rent was $1,200."))
    assert facts.claim_amount == 2500


def test_negated_signed_contract_fails_closed() -> None:
    instruction = _instruction().replace(
        "Under our signed room rental agreement,",
        "We never signed the room rental agreement, but",
    )
    assert _rejection_reason(instruction) == "missing_or_ambiguous_required_fact"


def test_unlabelled_date_range_cannot_pollute_event_dates() -> None:
    extra = "Letters were sent from April 1, 2026 through April 3, 2026."
    facts = parse_instruction(_instruction(extra=extra))
    assert (facts.incident_start, facts.incident_end) == ("2026-05-01", "2026-06-02")
    no_event = _instruction(event=extra)
    assert _rejection_reason(no_event) == "missing_or_ambiguous_required_fact"


def test_second_explicit_plaintiff_has_precedence_reason() -> None:
    assert _rejection_reason(
        _instruction(extra="The plaintiff's name is Bob Baker.")
    ) == "multiple_plaintiffs"


def test_second_plaintiff_precedes_payment_not_requested() -> None:
    instruction = _instruction(
        request="Payment was not requested before filing.",
        extra="The plaintiff's name is Bob Baker.",
    )
    assert _rejection_reason(instruction) == "multiple_plaintiffs"


@pytest.mark.parametrize(
    "extra",
    [
        (
            "Renée O'Neil-Santos, the plaintiff, also has address "
            "99 Ash Street, Berkeley, CA 94705."
        ),
        (
            "Renée O'Neil-Santos, the plaintiff, has another primary phone "
            "510-555-7777."
        ),
    ],
)
def test_second_explicit_role_fact_is_ambiguous(extra: str) -> None:
    assert _rejection_reason(
        _instruction(extra=extra)
    ) == "missing_or_ambiguous_required_fact"


def test_missing_claim_type_is_missing_not_explicitly_unsupported() -> None:
    instruction = _instruction().replace("security deposit", "deposit")
    assert _rejection_reason(instruction) == "missing_or_ambiguous_required_fact"


def test_non_california_precedes_explicit_unsupported_claim() -> None:
    instruction = _instruction().replace("CA 94704", "OR 97201") + (
        " This is a property damage claim, not a security-deposit dispute."
    )
    assert _rejection_reason(instruction) == "non_california_venue"


def test_two_explicit_venue_bases_are_ambiguous() -> None:
    instruction = _instruction(
        venue=(
            "File in the venue where the defendant lives. Also use the contract venue: "
            "the contract was made, performed, and breached in San Mateo, CA 94401."
        )
    )
    assert _rejection_reason(instruction) == "missing_or_ambiguous_required_fact"


def test_structured_labels_are_supported_without_global_position_assumptions() -> None:
    instruction = """
    Complete the California SC-100 for a rental security deposit claim.
    Claim amount: $2,501. Contract: signed roommate sublease contract.
    Rental start date: 2026-01-04; rental end date: 2026-02-06.
    Payment requested: yes, by email. Venue basis: defendant residence.
    Other small claims in the last 12 months: 13.
    Declaration date: 2026-02-09.
    Defendant: Noor Bennett, defendant address: 400 Elm Street, Richmond, CA 94801.
    Defendant phone: 510-555-3030.
    Plaintiff: Ana-María D'Arcy, plaintiff address: 22 Pine Road, Albany, CA 94706.
    Plaintiff primary phone: 510-555-4040. Email: ana@example.test.
    The defendant did not return the security deposit after move-out.
    """
    facts = parse_instruction(instruction)
    assert facts.plaintiff_name == "Ana-María D'Arcy"
    assert facts.defendant_name == "Noor Bennett"
    assert facts.claim_amount == 2501
    assert (facts.incident_start, facts.incident_end) == ("2026-01-04", "2026-02-06")
    assert facts.more_than_twelve_claims is True


def test_between_date_range_is_supported_but_demand_date_stays_separate() -> None:
    facts = parse_instruction(
        _instruction(
            event=(
                "The dispute period occurred between May 1, 2026 and June 2, 2026."
            ),
            request=(
                "I emailed a demand on June 9, 2026 asking the defendant to return "
                "the deposit."
            ),
        )
    )
    assert (facts.incident_start, facts.incident_end) == ("2026-05-01", "2026-06-02")


@pytest.mark.parametrize(
    ("mutator", "reason"),
    [
        (lambda text: text + " The defendant is a municipal housing authority.", "public_entity"),
        (lambda text: text + " This is an attorney-client fee dispute.", "attorney_fee_dispute"),
        (lambda text: text + " There are two plaintiffs.", "multiple_plaintiffs"),
        (
            lambda text: text.replace(
                "I emailed a request asking the defendant to return the deposit, but no payment arrived.",
                "I have not asked the defendant to pay or return the deposit.",
            ),
            "payment_not_requested",
        ),
        (lambda text: text + " The conflicting claimed amount is $2,700.", "conflicting_claim_amount"),
        (lambda text: text.replace("CA 94704", "OR 97201"), "non_california_venue"),
        (
            lambda text: text.replace(
                "This is a rental security deposit claim.",
                "This is a property damage claim, not a security-deposit dispute.",
            ),
            "unsupported_claim_type",
        ),
        (lambda text: text.replace("renee@example.test", ""), "missing_or_ambiguous_required_fact"),
    ],
)
def test_rejections_are_typed_and_fail_closed(
    mutator, reason: str, tmp_path: Path
) -> None:
    instruction = mutator(_instruction())
    assert _rejection_reason(instruction) == reason
    if PUBLIC_BLANK.is_file():
        output = tmp_path / f"{reason}.pdf"
        before = hashlib.sha256(PUBLIC_BLANK.read_bytes()).hexdigest()
        receipt = execute(instruction, PUBLIC_BLANK, output)
        assert receipt["action"] == "reject"
        assert receipt["reason_code"] == reason
        assert receipt["receipt_hash"] == _receipt_hash(receipt)
        assert not output.exists()
        assert hashlib.sha256(PUBLIC_BLANK.read_bytes()).hexdigest() == before


def test_rejection_precedence_is_stable() -> None:
    instruction = _instruction() + (
        " The defendant is a public entity. This is an attorney fee dispute. "
        "There are multiple plaintiffs."
    )
    assert _rejection_reason(instruction) == "public_entity"


def test_signature_before_event_end_is_rejected() -> None:
    assert _rejection_reason(
        _instruction(signature="Use May 20, 2026 as the declaration signing date.")
    ) == "missing_or_ambiguous_required_fact"


def test_execute_fill_is_bound_atomic_and_self_hashed(tmp_path: Path) -> None:
    pytest.importorskip("fitz")
    if not PUBLIC_BLANK.is_file():
        pytest.skip("local public SC-100 blank is unavailable")
    before = hashlib.sha256(PUBLIC_BLANK.read_bytes()).hexdigest()
    assert before == PUBLIC_BLANK_SHA256
    output = tmp_path / "filled.pdf"

    receipt = execute(
        instruction=_instruction(), blank_pdf=PUBLIC_BLANK, output_pdf=output
    )

    assert output.is_file()
    assert receipt["action"] == "fill"
    assert receipt["operator_version"] == OPERATOR_VERSION
    assert receipt["input_sha256"] == before
    assert receipt["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert receipt["mutation_count"] == 30
    assert receipt["atomic_publish"] is True
    assert receipt["temporary_cleanup_verified"] is True
    assert receipt["receipt_hash"] == _receipt_hash(receipt)
    assert hashlib.sha256(PUBLIC_BLANK.read_bytes()).hexdigest() == before


def test_execute_rejection_creates_no_output_and_is_self_hashed(tmp_path: Path) -> None:
    if not PUBLIC_BLANK.is_file():
        pytest.skip("local public SC-100 blank is unavailable")
    output = tmp_path / "must-not-exist.pdf"
    instruction = _instruction() + " The defendant is a public entity."

    receipt = execute(
        instruction=instruction, blank_pdf=PUBLIC_BLANK, output_pdf=output
    )

    assert receipt["action"] == "reject"
    assert receipt["reason_code"] == "public_entity"
    assert receipt["partial_output_created"] is False
    assert receipt["source_unchanged"] is True
    assert receipt["receipt_hash"] == _receipt_hash(receipt)
    assert not output.exists()


def test_non_string_instruction_rejects_without_write(tmp_path: Path) -> None:
    if not PUBLIC_BLANK.is_file():
        pytest.skip("local public SC-100 blank is unavailable")
    output = tmp_path / "non-string.pdf"
    receipt = execute(None, PUBLIC_BLANK, output)  # type: ignore[arg-type]
    assert receipt["action"] == "reject"
    assert receipt["reason_code"] == "missing_or_ambiguous_required_fact"
    assert receipt["receipt_hash"] == _receipt_hash(receipt)
    assert not output.exists()


def test_writer_failure_removes_temporary_and_final_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not PUBLIC_BLANK.is_file():
        pytest.skip("local public SC-100 blank is unavailable")
    output = tmp_path / "failed.pdf"

    def fail_after_partial(*, output_pdf: Path, **_kwargs):
        Path(output_pdf).write_bytes(b"partial")
        raise RuntimeError("invented writer failure")

    monkeypatch.setattr(_operator._writer, "apply_plan", fail_after_partial)
    with pytest.raises(RuntimeError, match="invented writer failure"):
        execute(_instruction(), PUBLIC_BLANK, output)
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_atomic_publish_never_clobbers_racing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not PUBLIC_BLANK.is_file():
        pytest.skip("local public SC-100 blank is unavailable")
    output = tmp_path / "raced.pdf"
    original_link = _operator.os.link

    def racing_link(source, destination):
        Path(destination).write_bytes(b"external-owner")
        return original_link(source, destination)

    monkeypatch.setattr(_operator.os, "link", racing_link)
    with pytest.raises(FileExistsError):
        execute(_instruction(), PUBLIC_BLANK, output)
    assert output.read_bytes() == b"external-owner"
    assert not any(path.name.endswith(".tmp.pdf") for path in tmp_path.iterdir())
