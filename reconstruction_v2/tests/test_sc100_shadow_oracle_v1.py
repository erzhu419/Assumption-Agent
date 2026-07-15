from __future__ import annotations

from dataclasses import replace
import json

import pytest

from assumption_agent.benchmarks.sc100_shadow_oracle_v1 import (
    BBoxWord,
    BUTTON_LOCATORS,
    BUTTON_ON_STATES,
    GoldValidationError,
    PdfSnapshot,
    TEXT_LOCATORS,
    WidgetSnapshot,
    build_receipt,
    _bbox_binding_failures,
    compare_document_snapshots,
    compare_widget_snapshots,
    compile_semantic_plan,
    qualify_sc100_shadow,
    _visible_text_failures,
)


REASON_TEXT = "The defendant did not return my security deposit."
CALCULATION_TEXT = "The roommate sublease contract required a $3,640 security deposit."


def _gold() -> dict:
    return {
        "plaintiff": {
            "name": "Avery Stone",
            "phone": "415-555-0137",
            "street": "41 Cedar Lane",
            "city": "Oakland",
            "state": "CA",
            "zip": "94607",
            "email": "avery@example.test",
        },
        "defendant": {
            "name": "Jordan Vale",
            "phone": "510-555-0184",
            "street": "900 Harbor Way",
            "city": "Oakland",
            "state": "CA",
            "zip": "94607",
        },
        "venue": {"zip": "94102"},
        "claim": {
            "amount": "3640",
            "start_date": "2025-03-14",
            "end_date": "2025-09-08",
            "contract": "roommate sublease contract",
        },
        "questions": {
            "asked_to_pay": True,
            "venue_choice": 1,
            "attorney_fee_dispute": False,
            "public_entity": False,
            "more_than_12_claims": False,
            "more_than_2500": True,
        },
        "signature": {"date": "2025-09-08", "name": "Avery Stone"},
    }


def _name(suffix: str) -> str:
    return "SC-100[0]" + suffix


def _blank_widgets() -> dict[str, WidgetSnapshot]:
    widgets: dict[str, WidgetSnapshot] = {}
    index = 0
    for suffix in TEXT_LOCATORS.values():
        name = _name(suffix)
        widgets[name] = WidgetSnapshot(
            full_name=name,
            page_index=1 + index // 9,
            rect=(10.0 + index, 20.0, 110.0 + index, 35.0),
            field_type="/Tx",
            field_flags=0,
            label="public label",
            value=None,
            appearance_state=None,
            appearance_states=(),
            appearance_digest=None,
        )
        index += 1
    for semantic, suffix in BUTTON_LOCATORS.items():
        name = _name(suffix)
        widgets[name] = WidgetSnapshot(
            full_name=name,
            page_index=2 if semantic.startswith(("q4", "q5", "q7", "q8")) else 3,
            rect=(10.0 + index, 40.0, 19.0 + index, 49.0),
            field_type="/Btn",
            field_flags=0,
            label="public label",
            value=None,
            appearance_state="/Off",
            appearance_states=(f"/{BUTTON_ON_STATES[semantic]}",),
            appearance_digest=f"button-ap-{semantic}",
        )
        index += 1
    forbidden = "SC-100[0].Page3[0].List8[0].item8[0].Date4[0]"
    widgets[forbidden] = WidgetSnapshot(
        full_name=forbidden,
        page_index=2,
        rect=(1.0, 1.0, 2.0, 2.0),
        field_type="/Tx",
        field_flags=0,
        label="date",
        value=None,
        appearance_state=None,
        appearance_states=(),
        appearance_digest=None,
    )
    return widgets


def _correct_filled(blank: dict[str, WidgetSnapshot], gold: dict) -> dict[str, WidgetSnapshot]:
    plan = compile_semantic_plan(gold)
    filled = dict(blank)
    for semantic, suffix in TEXT_LOCATORS.items():
        name = _name(suffix)
        if semantic in plan.text_values:
            value = plan.text_values[semantic]
        elif semantic == "plaintiff.phone":
            value = gold["plaintiff"]["phone"]
        elif semantic == "defendant.phone":
            value = gold["defendant"]["phone"]
        elif semantic == "claim.amount":
            value = gold["claim"]["amount"]
        elif semantic == "claim.reason":
            value = REASON_TEXT
        elif semantic == "claim.calculation":
            value = CALCULATION_TEXT
        else:
            raise AssertionError(f"unbound test locator: {semantic}")
        filled[name] = replace(
            blank[name],
            value=value,
            appearance_digest=f"text-ap-{semantic}",
            appearance_text=value,
            appearance_safe=True,
        )
    for semantic, suffix in BUTTON_LOCATORS.items():
        name = _name(suffix)
        state = plan.button_states[semantic]
        if state is not None:
            filled[name] = replace(blank[name], value=f"/{state}", appearance_state=f"/{state}")
    return filled


def test_correct_snapshot_canary_qualifies() -> None:
    gold = _gold()
    blank = _blank_widgets()
    decision = compare_widget_snapshots(blank, _correct_filled(blank, gold), gold)
    assert decision.qualified, decision.failure_codes
    assert len(decision.target_widget_names) == len(TEXT_LOCATORS) + len(BUTTON_LOCATORS)


def test_contract_venue_zip_is_independent_of_defendant_zip() -> None:
    gold = _gold()
    assert gold["venue"]["zip"] != gold["defendant"]["zip"]
    plan = compile_semantic_plan(gold)
    assert plan.text_values["venue.zip"] == "94102"
    blank = _blank_widgets()
    decision = compare_widget_snapshots(blank, _correct_filled(blank, gold), gold)
    assert decision.qualified, decision.failure_codes


def test_page2_page3_page4_captions_are_exact_targets() -> None:
    gold = _gold()
    blank = _blank_widgets()
    filled = _correct_filled(blank, gold)
    for semantic in ("caption.page2", "caption.page3", "caption.page4"):
        assert filled[_name(TEXT_LOCATORS[semantic])].value == gold["plaintiff"]["name"]
    name = _name(TEXT_LOCATORS["caption.page3"])
    filled[name] = replace(filled[name], value="Jordan Vale")
    decision = compare_widget_snapshots(blank, filled, gold)
    assert "text_value_mismatch:caption.page3" in decision.failure_codes


def test_extra_address_prefix_is_rejected() -> None:
    gold = _gold()
    blank = _blank_widgets()
    filled = _correct_filled(blank, gold)
    semantic = "plaintiff.street"
    name = _name(TEXT_LOCATORS[semantic])
    filled[name] = replace(filled[name], value="Apartment 7, 41 Cedar Lane")
    decision = compare_widget_snapshots(blank, filled, gold)
    assert f"text_value_mismatch:{semantic}" in decision.failure_codes


def test_wrong_amount_is_rejected_by_decimal_value() -> None:
    gold = _gold()
    blank = _blank_widgets()
    filled = _correct_filled(blank, gold)
    name = _name(TEXT_LOCATORS["claim.amount"])
    filled[name] = replace(filled[name], value="$3,641.00")
    decision = compare_widget_snapshots(blank, filled, gold)
    assert "amount_mismatch" in decision.failure_codes


def test_equivalent_phone_amount_and_free_text_formats_qualify() -> None:
    gold = _gold()
    blank = _blank_widgets()
    filled = _correct_filled(blank, gold)
    replacements = {
        "plaintiff.phone": "(415) 555-0137",
        "defendant.phone": "510.555.0184",
        "claim.amount": "$3,640.00",
        "claim.reason": "The defendant kept my security deposit and never refunded it.",
        "claim.calculation": (
            "Under the roommate sublease contract, the deposit balance is $3,640.00."
        ),
    }
    for semantic, value in replacements.items():
        name = _name(TEXT_LOCATORS[semantic])
        filled[name] = replace(filled[name], value=value, appearance_digest=f"variant-{semantic}")
        filled[name] = replace(filled[name], appearance_text=value, appearance_safe=True)
    decision = compare_widget_snapshots(blank, filled, gold)
    assert decision.qualified, decision.failure_codes


@pytest.mark.parametrize("question", ["q9", "q10"])
def test_q9_q10_single_bit_mutants_are_rejected(question: str) -> None:
    gold = _gold()
    blank = _blank_widgets()
    filled = _correct_filled(blank, gold)
    plan = compile_semantic_plan(gold)
    selected = next(key for key, state in plan.button_states.items() if key.startswith(question) and state)
    other = f"{question}.{'no' if selected.endswith('yes') else 'yes'}"
    selected_name = _name(BUTTON_LOCATORS[selected])
    other_name = _name(BUTTON_LOCATORS[other])
    other_state = BUTTON_ON_STATES[other]
    filled[selected_name] = replace(blank[selected_name])
    filled[other_name] = replace(
        blank[other_name], value=f"/{other_state}", appearance_state=f"/{other_state}"
    )
    decision = compare_widget_snapshots(blank, filled, gold)
    assert any(code.startswith("button_") and question in code for code in decision.failure_codes)


def test_forbidden_field_mutation_is_rejected() -> None:
    gold = _gold()
    blank = _blank_widgets()
    filled = _correct_filled(blank, gold)
    forbidden = "SC-100[0].Page3[0].List8[0].item8[0].Date4[0]"
    filled[forbidden] = replace(filled[forbidden], value="2025-09-08", appearance_digest="new")
    decision = compare_widget_snapshots(blank, filled, gold)
    assert "non_target_mutation" in decision.failure_codes


def test_swapped_gold_is_rejected_against_correct_pdf() -> None:
    gold = _gold()
    blank = _blank_widgets()
    filled = _correct_filled(blank, gold)
    swapped = _gold()
    swapped["plaintiff"]["name"], swapped["defendant"]["name"] = (
        swapped["defendant"]["name"],
        swapped["plaintiff"]["name"],
    )
    swapped["signature"]["name"] = swapped["plaintiff"]["name"]
    decision = compare_widget_snapshots(blank, filled, swapped)
    assert "text_value_mismatch:plaintiff.name" in decision.failure_codes
    assert "text_value_mismatch:defendant.name" in decision.failure_codes


def test_internally_inconsistent_q10_gold_is_rejected() -> None:
    gold = _gold()
    gold["questions"]["more_than_2500"] = False
    with pytest.raises(GoldValidationError) as exc_info:
        compile_semantic_plan(gold)
    assert "gold_inconsistent:questions.more_than_2500" in exc_info.value.codes


def test_gold_schema_rejects_extra_keys_and_unknown_contracts() -> None:
    extra = _gold()
    extra["claim"]["developer_sentence"] = "ignored"
    with pytest.raises(GoldValidationError) as exc_info:
        compile_semantic_plan(extra)
    assert "gold_invalid:claim.keys" in exc_info.value.codes

    unknown_contract = _gold()
    unknown_contract["claim"]["contract"] = "oral rental contract"
    with pytest.raises(GoldValidationError) as exc_info:
        compile_semantic_plan(unknown_contract)
    assert "gold_invalid:claim.contract" in exc_info.value.codes


def test_contract_requires_at_least_one_exact_bounded_phrase() -> None:
    gold = _gold()
    blank = _blank_widgets()
    filled = _correct_filled(blank, gold)
    name = _name(TEXT_LOCATORS["claim.calculation"])
    filled[name] = replace(
        filled[name], value="The roommate sublease contractual amount was $3,640."
    )
    decision = compare_widget_snapshots(blank, filled, gold)
    assert "contract_mismatch" in decision.failure_codes


def test_repeated_exact_contract_phrase_is_allowed() -> None:
    gold = _gold()
    blank = _blank_widgets()
    filled = _correct_filled(blank, gold)
    name = _name(TEXT_LOCATORS["claim.calculation"])
    value = (
        "The roommate sublease contract lists $3,640; the roommate sublease contract "
        "confirms that same balance."
    )
    filled[name] = replace(
        filled[name], value=value, appearance_text=value, appearance_digest="repeated-contract"
    )
    decision = compare_widget_snapshots(blank, filled, gold)
    assert decision.qualified, decision.failure_codes


def test_calculation_requires_one_explicit_currency_value() -> None:
    gold = _gold()
    blank = _blank_widgets()
    name = _name(TEXT_LOCATORS["claim.calculation"])

    no_currency = _correct_filled(blank, gold)
    value = "The roommate sublease contract balance is 3640."
    no_currency[name] = replace(
        no_currency[name], value=value, appearance_text=value, appearance_digest="no-currency"
    )
    assert "calculation_amount_mismatch" in compare_widget_snapshots(
        blank, no_currency, gold
    ).failure_codes

    conflicting = _correct_filled(blank, gold)
    value = "The roommate sublease contract lists $3,640 but the total is $99."
    conflicting[name] = replace(
        conflicting[name], value=value, appearance_text=value, appearance_digest="conflicting"
    )
    assert "calculation_amount_mismatch" in compare_widget_snapshots(
        blank, conflicting, gold
    ).failure_codes


def test_currency_space_is_an_equivalent_decimal_format() -> None:
    gold = _gold()
    blank = _blank_widgets()
    filled = _correct_filled(blank, gold)
    amount_name = _name(TEXT_LOCATORS["claim.amount"])
    filled[amount_name] = replace(
        filled[amount_name],
        value="$ 3,640.00",
        appearance_text="$ 3,640.00",
        appearance_digest="spaced-amount",
    )
    calculation_name = _name(TEXT_LOCATORS["claim.calculation"])
    value = "The roommate sublease contract requires $ 3,640.00."
    filled[calculation_name] = replace(
        filled[calculation_name],
        value=value,
        appearance_text=value,
        appearance_digest="spaced-calculation",
    )
    decision = compare_widget_snapshots(blank, filled, gold)
    assert decision.qualified, decision.failure_codes


def test_reason_requires_security_deposit_and_unreturned_semantics() -> None:
    gold = _gold()
    blank = _blank_widgets()
    filled = _correct_filled(blank, gold)
    name = _name(TEXT_LOCATORS["claim.reason"])
    filled[name] = replace(filled[name], value="A disagreement about a rental.")
    decision = compare_widget_snapshots(blank, filled, gold)
    assert "reason_semantics_mismatch" in decision.failure_codes


def test_reason_rejects_negated_withheld_but_accepts_yet_and_outstanding() -> None:
    gold = _gold()
    blank = _blank_widgets()
    name = _name(TEXT_LOCATORS["claim.reason"])

    negated = _correct_filled(blank, gold)
    value = "The security deposit was returned, not withheld."
    negated[name] = replace(
        negated[name], value=value, appearance_text=value, appearance_digest="negated"
    )
    assert "reason_semantics_mismatch" in compare_widget_snapshots(
        blank, negated, gold
    ).failure_codes

    for index, value in enumerate(
        (
            "The security deposit has yet to be returned.",
            "The security deposit remains outstanding.",
        )
    ):
        filled = _correct_filled(blank, gold)
        filled[name] = replace(
            filled[name],
            value=value,
            appearance_text=value,
            appearance_digest=f"valid-reason-{index}",
        )
        decision = compare_widget_snapshots(blank, filled, gold)
        assert decision.qualified, decision.failure_codes


def test_text_appearance_is_bound_to_each_field_value_and_safe_operators() -> None:
    gold = _gold()
    blank = _blank_widgets()
    filled = _correct_filled(blank, gold)
    plaintiff = _name(TEXT_LOCATORS["plaintiff.street"])
    defendant = _name(TEXT_LOCATORS["defendant.street"])
    plaintiff_text = filled[plaintiff].appearance_text
    defendant_text = filled[defendant].appearance_text
    filled[plaintiff] = replace(filled[plaintiff], appearance_text=defendant_text)
    filled[defendant] = replace(filled[defendant], appearance_text=plaintiff_text)
    decision = compare_widget_snapshots(blank, filled, gold)
    assert "text_ap_value_mismatch:plaintiff.street" in decision.failure_codes
    assert "text_ap_value_mismatch:defendant.street" in decision.failure_codes

    unsafe = _correct_filled(blank, gold)
    unsafe[plaintiff] = replace(unsafe[plaintiff], appearance_safe=False)
    assert "text_ap_unsafe:plaintiff.street" in compare_widget_snapshots(
        blank, unsafe, gold
    ).failure_codes


def test_every_button_ap_and_every_default_value_are_immutable() -> None:
    gold = _gold()
    blank = _blank_widgets()
    filled = _correct_filled(blank, gold)
    unselected = _name(BUTTON_LOCATORS["q4.no"])
    filled[unselected] = replace(filled[unselected], appearance_digest="mutated-ap")
    assert "button_ap_mutation:q4.no" in compare_widget_snapshots(
        blank, filled, gold
    ).failure_codes

    default_mutant = _correct_filled(blank, gold)
    forbidden = "SC-100[0].Page3[0].List8[0].item8[0].Date4[0]"
    default_mutant[forbidden] = replace(default_mutant[forbidden], default_value="2025-09-08")
    assert "field_structure_mismatch:widget" in compare_widget_snapshots(
        blank, default_mutant, gold
    ).failure_codes


def test_pdftotext_check_uses_free_text_semantics_and_actual_rendered_values() -> None:
    gold = _gold()
    blank = _blank_widgets()
    filled = _correct_filled(blank, gold)
    rendered_text = "\n".join(widget.value or "" for widget in filled.values())
    failures = _visible_text_failures(rendered_text, compile_semantic_plan(gold), filled)
    assert failures == ()


def test_missing_text_appearance_is_rejected() -> None:
    gold = _gold()
    blank = _blank_widgets()
    filled = _correct_filled(blank, gold)
    name = _name(TEXT_LOCATORS["plaintiff.name"])
    filled[name] = replace(filled[name], appearance_digest=None)
    decision = compare_widget_snapshots(blank, filled, gold)
    assert "text_ap_missing:plaintiff.name" in decision.failure_codes


def test_document_snapshot_rejects_page_image_and_security_mutations() -> None:
    blank = PdfSnapshot(
        page_count=6,
        page_geometry=(("0",),) * 6,
        page_content_digests=("content",) * 6,
        page_image_digests=((),) * 6,
        non_widget_annotation_digests=((),) * 6,
        xfa_digest="xfa",
        security_digest="security",
        widgets={},
        field_tree_digest="field-tree",
        page_resource_digests=("resources",) * 6,
    )
    filled = replace(
        blank,
        page_image_digests=(("new-full-page-image",),) + ((),) * 5,
        security_digest="new-javascript-or-attachment",
        field_tree_digest="new-field-tree",
        page_resource_digests=("new-resources",) + ("resources",) * 5,
    )
    failures = compare_document_snapshots(blank, filled)
    assert "page_images_changed" in failures
    assert "actions_or_attachments_changed" in failures
    assert "field_tree_changed" in failures
    assert "page_resources_changed" in failures


def test_poppler_bbox_words_must_bind_to_each_local_field() -> None:
    gold = _gold()
    blank = _blank_widgets()
    filled_widgets = _correct_filled(blank, gold)
    geometry = tuple(("0", "0", "612", "792", "0", "0", "612", "792", "0") for _ in range(6))
    snapshot = PdfSnapshot(
        page_count=6,
        page_geometry=geometry,
        page_content_digests=("content",) * 6,
        page_image_digests=((),) * 6,
        non_widget_annotation_digests=((),) * 6,
        xfa_digest="xfa",
        security_digest="security",
        widgets=filled_widgets,
    )
    words = []
    for semantic, suffix in TEXT_LOCATORS.items():
        widget = filled_widgets[_name(suffix)]
        left, bottom, right, top = widget.rect
        words.append(
            BBoxWord(
                page_index=widget.page_index,
                text=widget.appearance_text or "",
                x_min=left,
                y_min=792 - top,
                x_max=right,
                y_max=792 - bottom,
            )
        )
    assert _bbox_binding_failures(tuple(words), snapshot) == ()
    email_index = list(TEXT_LOCATORS).index("plaintiff.email")
    email_word = words[email_index]
    words[email_index] = replace(email_word, x_min=500, x_max=510)
    assert "poppler_bbox_mismatch:plaintiff.email" in _bbox_binding_failures(
        tuple(words), snapshot
    )


def test_qualification_always_returns_receipt_for_unserializable_gold_and_bad_paths() -> None:
    gold = _gold()
    gold["ignored_extra"] = object()
    receipt = qualify_sc100_shadow("/definitely/missing.pdf", "/also/missing.pdf", gold)
    assert receipt["qualified"] is False
    assert "gold_serialization_failed" in receipt["failure_codes"]
    assert "path_or_hash_failed" in receipt["failure_codes"]
    assert len(receipt["receipt_sha256"]) == 64


def test_receipt_contains_hashes_but_no_case_text() -> None:
    receipt = build_receipt(
        blank_sha256="a" * 64,
        filled_sha256="b" * 64,
        gold_sha256="c" * 64,
        failures=(),
        target_count=len(TEXT_LOCATORS) + len(BUTTON_LOCATORS),
        runtime={"pypdf": "5.1.0"},
    )
    encoded = json.dumps(receipt, sort_keys=True)
    assert receipt["qualified"] is True
    assert "Avery" not in encoded
    assert "Cedar" not in encoded
    assert len(receipt["receipt_sha256"]) == 64
