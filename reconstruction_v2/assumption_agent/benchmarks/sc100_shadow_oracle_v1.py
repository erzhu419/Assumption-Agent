"""Independent, offline oracle for synthetic California SC-100 shadow cases.

This module deliberately owns its field locators and semantic checks.  It does
not depend on the SC-100 action operator, benchmark verifiers, or benchmark
solutions.  The public blank form is the only PDF-side source of truth.

The top-level :func:`qualify_sc100_shadow` function is intended to run in the
frozen item image (pypdf 5.1.0, Poppler, and Pillow).  Pure snapshot helpers are
kept dependency-free so mutations can be tested on the host.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from numbers import Number
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import unicodedata
import xml.etree.ElementTree as ET


ORACLE_SCHEMA = "sc100-shadow-oracle-v1"
REQUIRED_PYPDF_VERSION = "5.1.0"
REQUIRED_POPPLER_VERSION = "24.02.0"
PUBLIC_BLANK_SHA256 = "ef3421b14ebf64dbf884566ff659b39776035ec5b6e6500be0af91e3cc15533c"
ALLOWED_CONTRACTS = frozenset({"room rental agreement", "roommate sublease contract"})
_GOLD_SECTION_KEYS: dict[str, frozenset[str]] = {
    "plaintiff": frozenset({"name", "phone", "street", "city", "state", "zip", "email"}),
    "defendant": frozenset({"name", "phone", "street", "city", "state", "zip"}),
    "venue": frozenset({"zip"}),
    "claim": frozenset({"amount", "start_date", "end_date", "contract"}),
    "questions": frozenset(
        {
            "asked_to_pay",
            "venue_choice",
            "attorney_fee_dispute",
            "public_entity",
            "more_than_12_claims",
            "more_than_2500",
        }
    ),
    "signature": frozenset({"date", "name"}),
}


# These suffixes are derived from the public labels and hierarchy in the blank
# Judicial Council SC-100 form.  Suffix matching preserves the public form's
# root name while avoiding an item-specific or producer-specific object id.
TEXT_LOCATORS: dict[str, str] = {
    "caption.page2": ".Page2[0].PxCaption[0].Plaintiff[0]",
    "caption.page3": ".Page3[0].PxCaption[0].Plaintiff[0]",
    "caption.page4": ".Page4[0].PxCaption[0].Plaintiff[0]",
    "plaintiff.name": ".Page2[0].List1[0].Item1[0].PlaintiffName1[0]",
    "plaintiff.phone": ".Page2[0].List1[0].Item1[0].PlaintiffPhone1[0]",
    "plaintiff.street": ".Page2[0].List1[0].Item1[0].PlaintiffAddress1[0]",
    "plaintiff.city": ".Page2[0].List1[0].Item1[0].PlaintiffCity1[0]",
    "plaintiff.state": ".Page2[0].List1[0].Item1[0].PlaintiffState1[0]",
    "plaintiff.zip": ".Page2[0].List1[0].Item1[0].PlaintiffZip1[0]",
    "plaintiff.email": ".Page2[0].List1[0].Item1[0].EmailAdd1[0]",
    "defendant.name": ".Page2[0].List2[0].item2[0].DefendantName1[0]",
    "defendant.phone": ".Page2[0].List2[0].item2[0].DefendantPhone1[0]",
    "defendant.street": ".Page2[0].List2[0].item2[0].DefendantAddress1[0]",
    "defendant.city": ".Page2[0].List2[0].item2[0].DefendantCity1[0]",
    "defendant.state": ".Page2[0].List2[0].item2[0].DefendantState1[0]",
    "defendant.zip": ".Page2[0].List2[0].item2[0].DefendantZip1[0]",
    "claim.amount": ".Page2[0].List3[0].PlaintiffClaimAmount1[0]",
    "claim.reason": ".Page2[0].List3[0].Lia[0].FillField2[0]",
    "claim.start_date": ".Page3[0].List3[0].Lib[0].Date2[0]",
    "claim.end_date": ".Page3[0].List3[0].Lib[0].Date3[0]",
    "claim.calculation": ".Page3[0].List3[0].Lic[0].FillField1[0]",
    "venue.zip": ".Page3[0].List6[0].item6[0].ZipCode1[0]",
    "signature.date": ".Page4[0].Sign[0].Date1[0]",
    "signature.name": ".Page4[0].Sign[0].PlaintiffName1[0]",
}


BUTTON_LOCATORS: dict[str, str] = {
    "q4.yes": ".Page3[0].List4[0].Item4[0].Checkbox50[0]",
    "q4.no": ".Page3[0].List4[0].Item4[0].Checkbox50[1]",
    "q5.1": ".Page3[0].List5[0].Lia[0].Checkbox5cb[0]",
    "q5.2": ".Page3[0].List5[0].Lib[0].Checkbox5cb[0]",
    "q5.3": ".Page3[0].List5[0].Lic[0].Checkbox5cb[0]",
    "q5.4": ".Page3[0].List5[0].Lid[0].Checkbox5cb[0]",
    "q5.5": ".Page3[0].List5[0].Lie[0].Checkbox5cb[0]",
    "q7.yes": ".Page3[0].List7[0].item7[0].Checkbox60[0]",
    "q7.no": ".Page3[0].List7[0].item7[0].Checkbox60[1]",
    "q8.yes": ".Page3[0].List8[0].item8[0].Checkbox61[0]",
    "q8.no": ".Page3[0].List8[0].item8[0].Checkbox61[1]",
    "q9.yes": ".Page4[0].List9[0].Item9[0].Checkbox62[0]",
    "q9.no": ".Page4[0].List9[0].Item9[0].Checkbox62[1]",
    "q10.yes": ".Page4[0].List10[0].li10[0].Checkbox63[0]",
    "q10.no": ".Page4[0].List10[0].li10[0].Checkbox63[1]",
}


BUTTON_ON_STATES: dict[str, str] = {
    "q4.yes": "1",
    "q4.no": "2",
    "q5.1": "1",
    "q5.2": "2",
    "q5.3": "3",
    "q5.4": "4",
    "q5.5": "5",
    "q7.yes": "1",
    "q7.no": "2",
    "q8.yes": "1",
    "q8.no": "2",
    "q9.yes": "1",
    "q9.no": "2",
    "q10.yes": "1",
    "q10.no": "2",
}


@dataclass(frozen=True)
class WidgetSnapshot:
    """Relevant, serialization-safe state for one AcroForm widget."""

    full_name: str
    page_index: int
    rect: tuple[float, float, float, float]
    field_type: str
    field_flags: int
    label: str
    value: str | None
    appearance_state: str | None
    appearance_states: tuple[str, ...]
    appearance_digest: str | None
    structure_digest: str = ""
    default_value: str | None = None
    appearance_text: str | None = None
    appearance_safe: bool = False


@dataclass(frozen=True)
class PdfSnapshot:
    """Document invariants that a form fill is not allowed to rewrite."""

    page_count: int
    page_geometry: tuple[tuple[str, ...], ...]
    page_content_digests: tuple[str, ...]
    page_image_digests: tuple[tuple[str, ...], ...]
    non_widget_annotation_digests: tuple[tuple[str, ...], ...]
    xfa_digest: str | None
    security_digest: str
    widgets: Mapping[str, WidgetSnapshot]
    field_tree_digest: str = ""
    page_resource_digests: tuple[str, ...] = ()


@dataclass(frozen=True)
class SemanticPlan:
    text_values: Mapping[str, str]
    button_states: Mapping[str, str | None]
    phone_digits: Mapping[str, str]
    amount: Decimal
    contract: str


@dataclass(frozen=True)
class SnapshotDecision:
    failure_codes: tuple[str, ...]
    target_widget_names: tuple[str, ...]

    @property
    def qualified(self) -> bool:
        return not self.failure_codes


@dataclass(frozen=True)
class BBoxWord:
    page_index: int
    text: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class GoldValidationError(ValueError):
    """Raised only for malformed or internally inconsistent semantic gold."""

    def __init__(self, codes: Iterable[str]):
        self.codes = tuple(sorted(set(codes)))
        super().__init__("invalid SC-100 semantic gold")


def _nfc(value: Any) -> str:
    return unicodedata.normalize("NFC", str(value)).replace("\r\n", "\n").replace("\r", "\n")


def _required_string(container: Mapping[str, Any], key: str, code: str, errors: list[str]) -> str:
    value = container.get(key)
    if not isinstance(value, str) or not value or value != value.strip():
        errors.append(f"gold_invalid:{code}")
        return ""
    normalized = _nfc(value)
    if any(ord(char) < 32 and char not in "\n\t" for char in normalized):
        errors.append(f"gold_invalid:{code}")
    return normalized


def _required_bool(container: Mapping[str, Any], key: str, code: str, errors: list[str]) -> bool:
    value = container.get(key)
    if type(value) is not bool:  # bool must not silently accept 0/1.
        errors.append(f"gold_invalid:{code}")
        return False
    return value


_MONEY_BODY = r"(?:0|[1-9]\d{0,2}(?:,\d{3})*|[1-9]\d*)(?:\.\d{1,2})?"
_MONEY_RE = re.compile(rf"(?:\$\s*)?{_MONEY_BODY}\Z")
_CURRENCY_MENTION_RE = re.compile(
    rf"(?<![\w.,])\$\s*({_MONEY_BODY})(?![\w,]|\.\d)"
)
_PHONE_RE = re.compile(r"[\d\s()+.\-]+\Z")


def _money_decimal(value: str) -> Decimal | None:
    if not _MONEY_RE.fullmatch(value):
        return None
    try:
        return Decimal(value.replace("$", "").replace(",", "").replace(" ", ""))
    except InvalidOperation:
        return None


def _currency_mentions(value: str) -> tuple[Decimal, ...]:
    mentions: list[Decimal] = []
    for match in _CURRENCY_MENTION_RE.finditer(value):
        parsed = _money_decimal(match.group(1))
        if parsed is not None:
            mentions.append(parsed)
    return tuple(mentions)


def _phone_digits(value: str) -> str | None:
    if not value or not _PHONE_RE.fullmatch(value):
        return None
    digits = "".join(char for char in value if char.isdigit())
    return digits if len(digits) >= 7 else None


def _contains_exact_phrase(text: str, phrase: str) -> bool:
    text = text.casefold()
    phrase = phrase.casefold()
    start = 0
    hits = 0
    while True:
        index = text.find(phrase, start)
        if index < 0:
            break
        before = text[index - 1] if index else ""
        after_index = index + len(phrase)
        after = text[after_index] if after_index < len(text) else ""
        if (not before or not before.isalnum()) and (not after or not after.isalnum()):
            hits += 1
        start = index + max(1, len(phrase))
    return hits >= 1


_UNRETURNED_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\bunreturned\b",
        r"\b(?:not|never)\s+(?:been\s+)?(?:return(?:ed)?|refund(?:ed)?)\b",
        r"\bdid(?:\s+not|n't)\s+(?:return|refund)\b",
        r"\b(?:has|have|had|was|were|is|are)\s+(?:not|never)\s+(?:been\s+)?(?:returned|refunded)\b",
        r"\b(?:fail(?:ed|ure)?|refus(?:ed|al))\s+to\s+(?:return|refund)\b",
        r"\b(?:has|have|had|is|are|was|were)?\s*yet\s+to\s+be\s+(?:returned|refunded)\b",
        r"\b(?:remain(?:s|ed)?\s+)?(?:still\s+)?outstanding\b",
    )
)


def _reason_has_required_semantics(value: str) -> bool:
    normalized = " ".join(_nfc(value).casefold().split())
    if not re.search(r"\bsecurity\b", normalized):
        return False
    if not re.search(r"\bdeposit\b", normalized):
        return False
    if any(pattern.search(normalized) for pattern in _UNRETURNED_PATTERNS):
        return True
    for match in re.finditer(r"\b(?:withheld|retained|kept)\b", normalized):
        prefix = normalized[max(0, match.start() - 24) : match.start()]
        if not re.search(r"\b(?:not|never|no)\s+(?:\w+\s+){0,2}$|n't\s+$", prefix):
            return True
    return False


def _calculation_has_required_semantics(value: str, amount: Decimal, contract: str) -> bool:
    normalized = " ".join(_nfc(value).split())
    mentions = _currency_mentions(normalized)
    return bool(mentions) and set(mentions) == {amount} and _contains_exact_phrase(
        normalized, contract
    )


def compile_semantic_plan(gold: Mapping[str, Any]) -> SemanticPlan:
    """Validate semantic gold and compile exact form values.

    Expected schema (all leaves are required)::

        plaintiff: name, phone, street, city, state, zip, email
        defendant: name, phone, street, city, state, zip
        venue: zip
        claim: amount, start_date, end_date, contract
        questions: asked_to_pay, venue_choice (1..5), attorney_fee_dispute,
                   public_entity, more_than_12_claims, more_than_2500
        signature: date, name

    Identity/address/date/signature values are exact.  Phone punctuation and
    currency formatting are presentation details: phones compare by digits and
    amounts compare by :class:`~decimal.Decimal` value.
    """

    errors: list[str] = []
    if not isinstance(gold, Mapping):
        raise GoldValidationError(("gold_invalid:root",))
    if set(gold) != set(_GOLD_SECTION_KEYS):
        errors.append("gold_invalid:root.keys")

    def section(name: str) -> Mapping[str, Any]:
        value = gold.get(name)
        if not isinstance(value, Mapping):
            errors.append(f"gold_invalid:{name}")
            return {}
        if set(value) != set(_GOLD_SECTION_KEYS[name]):
            errors.append(f"gold_invalid:{name}.keys")
        return value

    plaintiff = section("plaintiff")
    defendant = section("defendant")
    venue = section("venue")
    claim = section("claim")
    questions = section("questions")
    signature = section("signature")

    text_values = {
        "caption.page2": _required_string(plaintiff, "name", "plaintiff.name", errors),
        "caption.page3": _required_string(plaintiff, "name", "plaintiff.name", errors),
        "caption.page4": _required_string(plaintiff, "name", "plaintiff.name", errors),
        "plaintiff.name": _required_string(plaintiff, "name", "plaintiff.name", errors),
        "plaintiff.street": _required_string(plaintiff, "street", "plaintiff.street", errors),
        "plaintiff.city": _required_string(plaintiff, "city", "plaintiff.city", errors),
        "plaintiff.state": _required_string(plaintiff, "state", "plaintiff.state", errors),
        "plaintiff.zip": _required_string(plaintiff, "zip", "plaintiff.zip", errors),
        "plaintiff.email": _required_string(plaintiff, "email", "plaintiff.email", errors),
        "defendant.name": _required_string(defendant, "name", "defendant.name", errors),
        "defendant.street": _required_string(defendant, "street", "defendant.street", errors),
        "defendant.city": _required_string(defendant, "city", "defendant.city", errors),
        "defendant.state": _required_string(defendant, "state", "defendant.state", errors),
        "defendant.zip": _required_string(defendant, "zip", "defendant.zip", errors),
        "claim.start_date": _required_string(claim, "start_date", "claim.start_date", errors),
        "claim.end_date": _required_string(claim, "end_date", "claim.end_date", errors),
        "venue.zip": _required_string(venue, "zip", "venue.zip", errors),
        "signature.date": _required_string(signature, "date", "signature.date", errors),
        "signature.name": _required_string(signature, "name", "signature.name", errors),
    }
    phone_values = {
        "plaintiff.phone": _required_string(plaintiff, "phone", "plaintiff.phone", errors),
        "defendant.phone": _required_string(defendant, "phone", "defendant.phone", errors),
    }
    phone_digits: dict[str, str] = {}
    for semantic, value in phone_values.items():
        digits = _phone_digits(value)
        if digits is None:
            errors.append(f"gold_invalid:{semantic}")
            digits = ""
        phone_digits[semantic] = digits
    amount_value = _required_string(claim, "amount", "claim.amount", errors)
    contract = _required_string(claim, "contract", "claim.contract", errors)
    if contract not in ALLOWED_CONTRACTS:
        errors.append("gold_invalid:claim.contract")

    amount = _money_decimal(amount_value)
    if amount is None or amount < 0:
        errors.append("gold_invalid:claim.amount")
    if text_values["signature.name"] != text_values["plaintiff.name"]:
        errors.append("gold_inconsistent:signature.name")

    asked = _required_bool(questions, "asked_to_pay", "questions.asked_to_pay", errors)
    attorney = _required_bool(
        questions, "attorney_fee_dispute", "questions.attorney_fee_dispute", errors
    )
    public = _required_bool(questions, "public_entity", "questions.public_entity", errors)
    many = _required_bool(
        questions, "more_than_12_claims", "questions.more_than_12_claims", errors
    )
    over = _required_bool(questions, "more_than_2500", "questions.more_than_2500", errors)
    venue_choice = questions.get("venue_choice")
    if type(venue_choice) is not int or not 1 <= venue_choice <= 5:
        errors.append("gold_invalid:questions.venue_choice")
        venue_choice = 1
    if amount is not None and over != (amount > Decimal("2500")):
        errors.append("gold_inconsistent:questions.more_than_2500")

    button_states: dict[str, str | None] = {}
    for key in BUTTON_LOCATORS:
        button_states[key] = None
    button_states["q4.yes" if asked else "q4.no"] = BUTTON_ON_STATES[
        "q4.yes" if asked else "q4.no"
    ]
    q5_key = f"q5.{venue_choice}"
    button_states[q5_key] = BUTTON_ON_STATES[q5_key]
    for prefix, flag in (
        ("q7", attorney),
        ("q8", public),
        ("q9", many),
        ("q10", over),
    ):
        key = f"{prefix}.{'yes' if flag else 'no'}"
        button_states[key] = BUTTON_ON_STATES[key]

    if errors:
        raise GoldValidationError(errors)
    assert amount is not None
    return SemanticPlan(
        text_values=text_values,
        button_states=button_states,
        phone_digits=phone_digits,
        amount=amount,
        contract=contract,
    )


def _off(value: str | None) -> bool:
    return value is None or value in {"", "Off", "/Off"}


def _display_normalize(value: str | None) -> str:
    return " ".join(_nfc(value or "").split())


def _resolve_locator(widgets: Mapping[str, WidgetSnapshot], suffix: str) -> str | None:
    matches = [name for name in widgets if name.endswith(suffix)]
    return matches[0] if len(matches) == 1 else None


def compare_widget_snapshots(
    blank_widgets: Mapping[str, WidgetSnapshot],
    filled_widgets: Mapping[str, WidgetSnapshot],
    gold: Mapping[str, Any] | SemanticPlan,
) -> SnapshotDecision:
    """Compare exact semantic values and reject every undeclared field change."""

    failures: list[str] = []
    try:
        plan = gold if isinstance(gold, SemanticPlan) else compile_semantic_plan(gold)
    except GoldValidationError as exc:
        return SnapshotDecision(exc.codes, ())

    if set(blank_widgets) != set(filled_widgets):
        failures.append("field_structure_mismatch:names")

    all_names = sorted(set(blank_widgets) | set(filled_widgets))
    for name in all_names:
        before = blank_widgets.get(name)
        after = filled_widgets.get(name)
        if before is None or after is None:
            continue
        if (
            before.page_index,
            before.rect,
            before.field_type,
            before.field_flags,
            before.label,
            before.appearance_states,
            before.structure_digest,
            before.default_value,
        ) != (
            after.page_index,
            after.rect,
            after.field_type,
            after.field_flags,
            after.label,
            after.appearance_states,
            after.structure_digest,
            after.default_value,
        ):
            failures.append("field_structure_mismatch:widget")

    target_names: set[str] = set()
    text_name_by_semantic: dict[str, str] = {}
    for semantic, suffix in TEXT_LOCATORS.items():
        name = _resolve_locator(blank_widgets, suffix)
        if name is None:
            failures.append(f"locator_mismatch:{semantic}")
            continue
        target_names.add(name)
        text_name_by_semantic[semantic] = name
        actual = filled_widgets.get(name)
        if actual is None:
            continue
        value = actual.value or ""
        if semantic in plan.text_values:
            if actual.value != plan.text_values[semantic]:
                failures.append(f"text_value_mismatch:{semantic}")
        elif semantic in plan.phone_digits:
            if _phone_digits(value) != plan.phone_digits[semantic]:
                failures.append(f"phone_digits_mismatch:{semantic}")
        elif semantic == "claim.amount":
            if _money_decimal(value) != plan.amount:
                failures.append("amount_mismatch")
        elif semantic == "claim.reason":
            if not _reason_has_required_semantics(value):
                failures.append("reason_semantics_mismatch")
        elif semantic == "claim.calculation":
            mentions = _currency_mentions(value)
            if not mentions or set(mentions) != {plan.amount}:
                failures.append("calculation_amount_mismatch")
            if not _contains_exact_phrase(value, plan.contract):
                failures.append("contract_mismatch")
        else:  # Defensive exhaustiveness for a newly added public locator.
            failures.append(f"unbound_text_locator:{semantic}")
        if actual.appearance_state not in (None, "", "Off", "/Off"):
            failures.append(f"text_as_mismatch:{semantic}")
        if not actual.appearance_digest:
            failures.append(f"text_ap_missing:{semantic}")
        elif actual.appearance_digest == blank_widgets[name].appearance_digest:
            failures.append(f"text_ap_unchanged:{semantic}")
        if not actual.appearance_safe:
            failures.append(f"text_ap_unsafe:{semantic}")
        if _display_normalize(actual.appearance_text) != _display_normalize(actual.value):
            failures.append(f"text_ap_value_mismatch:{semantic}")

    for semantic, suffix in BUTTON_LOCATORS.items():
        name = _resolve_locator(blank_widgets, suffix)
        if name is None:
            failures.append(f"locator_mismatch:{semantic}")
            continue
        target_names.add(name)
        actual = filled_widgets.get(name)
        if actual is None:
            continue
        if actual.appearance_digest != blank_widgets[name].appearance_digest:
            failures.append(f"button_ap_mutation:{semantic}")
        expected = plan.button_states[semantic]
        if expected is None:
            if not _off(actual.value) or not _off(actual.appearance_state):
                failures.append(f"button_state_mismatch:{semantic}")
        else:
            if actual.value not in {expected, f"/{expected}"}:
                failures.append(f"button_v_mismatch:{semantic}")
            if actual.appearance_state not in {expected, f"/{expected}"}:
                failures.append(f"button_as_mismatch:{semantic}")
            normalized_states = {state.removeprefix("/") for state in actual.appearance_states}
            if expected not in normalized_states or not actual.appearance_digest:
                failures.append(f"button_ap_mismatch:{semantic}")

    # A correct set of target values is not enough: every field outside the
    # semantic plan must remain byte-semantically at its blank state.
    for name in sorted(set(blank_widgets) & set(filled_widgets) - target_names):
        before = blank_widgets[name]
        after = filled_widgets[name]
        if (
            before.value,
            before.appearance_state,
            before.appearance_digest,
        ) != (
            after.value,
            after.appearance_state,
            after.appearance_digest,
        ):
            failures.append("non_target_mutation")

    return SnapshotDecision(tuple(sorted(set(failures))), tuple(sorted(target_names)))


def compare_document_snapshots(blank: PdfSnapshot, filled: PdfSnapshot) -> tuple[str, ...]:
    """Reject page rewrites, rasterization, attachments, and action changes."""

    failures: list[str] = []
    if blank.page_count != filled.page_count:
        failures.append("page_count_mismatch")
    if blank.page_geometry != filled.page_geometry:
        failures.append("page_geometry_mismatch")
    if blank.page_content_digests != filled.page_content_digests:
        failures.append("page_content_rewritten")
    if blank.page_image_digests != filled.page_image_digests:
        failures.append("page_images_changed")
    if blank.non_widget_annotation_digests != filled.non_widget_annotation_digests:
        failures.append("non_widget_annotations_changed")
    if blank.xfa_digest != filled.xfa_digest:
        failures.append("xfa_changed")
    if blank.security_digest != filled.security_digest:
        failures.append("actions_or_attachments_changed")
    if blank.field_tree_digest != filled.field_tree_digest:
        failures.append("field_tree_changed")
    if blank.page_resource_digests != filled.page_resource_digests:
        failures.append("page_resources_changed")
    return tuple(failures)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object_digest(value: Any, *, skip_keys: frozenset[str] = frozenset()) -> str | None:
    if value is None:
        return None
    seen: set[tuple[int, int]] = set()

    def convert(obj: Any) -> Any:
        if hasattr(obj, "idnum") and hasattr(obj, "generation"):
            ref = (int(obj.idnum), int(obj.generation))
            if ref in seen:
                return {"cycle": True}
            seen.add(ref)
            return convert(obj.get_object())
        if hasattr(obj, "get_data") and isinstance(obj, Mapping):
            metadata = {
                str(key): convert(item)
                for key, item in sorted(obj.items(), key=lambda pair: str(pair[0]))
                # The decoded bytes are the semantic stream payload.  PDF
                # producers may losslessly rewrite a single filter as either
                # a name or a one-element array, so compression syntax is not
                # an invariant of a legitimate form fill.
                if str(key)
                not in {
                    "/Length",
                    "/Filter",
                    "/DecodeParms",
                    "/F",
                    "/FFilter",
                    "/FDecodeParms",
                }
                and str(key) not in skip_keys
            }
            return {"stream": _sha256_bytes(obj.get_data()), "dict": metadata}
        if isinstance(obj, Mapping):
            return {
                str(key): convert(item)
                for key, item in sorted(obj.items(), key=lambda pair: str(pair[0]))
                if str(key) not in skip_keys
            }
        if isinstance(obj, (list, tuple)):
            return [convert(item) for item in obj]
        if isinstance(obj, bytes):
            return {"bytes": obj.hex()}
        if isinstance(obj, Number) and not isinstance(obj, bool):
            number = Decimal(str(obj))
            normalized = format(number.normalize(), "f")
            return {"number": "0" if normalized in {"-0", ""} else normalized}
        if obj is None or isinstance(obj, bool):
            return obj
        return str(obj)

    payload = json.dumps(convert(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return _sha256_bytes(payload.encode("ascii"))


def _stable_dictionary_digest(
    value: Any, *, skip_keys: frozenset[str] = frozenset()
) -> str | None:
    if value is None:
        return None
    resolved = value.get_object() if hasattr(value, "get_object") else value
    if not isinstance(resolved, Mapping):
        return _object_digest(resolved, skip_keys=skip_keys)
    entries = {
        str(key): _object_digest(item, skip_keys=skip_keys)
        for key, item in sorted(resolved.items(), key=lambda pair: str(pair[0]))
        if str(key) not in skip_keys
    }
    return _canonical_hash(entries)


def _inherited(widget: Mapping[str, Any], key: str) -> Any:
    current: Any = widget
    visited: set[tuple[int, int]] = set()
    while isinstance(current, Mapping):
        if key in current:
            return current[key]
        parent = current.get("/Parent")
        if parent is None:
            return None
        if hasattr(parent, "idnum"):
            marker = (int(parent.idnum), int(parent.generation))
            if marker in visited:
                return None
            visited.add(marker)
        current = parent.get_object() if hasattr(parent, "get_object") else parent
    return None


def _full_field_name(widget: Mapping[str, Any]) -> str:
    parts: list[str] = []
    current: Any = widget
    visited: set[tuple[int, int]] = set()
    while isinstance(current, Mapping):
        if current.get("/T") is not None:
            parts.append(str(current["/T"]))
        parent = current.get("/Parent")
        if parent is None:
            break
        if hasattr(parent, "idnum"):
            marker = (int(parent.idnum), int(parent.generation))
            if marker in visited:
                break
            visited.add(marker)
        current = parent.get_object() if hasattr(parent, "get_object") else parent
    return ".".join(reversed(parts))


_SAFE_TEXT_AP_OPERATORS = frozenset(
    {
        b"q",
        b"Q",
        b"cm",
        b"w",
        b"J",
        b"j",
        b"M",
        b"d",
        b"m",
        b"l",
        b"c",
        b"v",
        b"y",
        b"h",
        b"re",
        b"S",
        b"s",
        b"f",
        b"F",
        b"f*",
        b"B",
        b"B*",
        b"b",
        b"b*",
        b"n",
        b"W",
        b"W*",
        b"BT",
        b"ET",
        b"Tc",
        b"Tw",
        b"Tz",
        b"TL",
        b"Tf",
        b"Tr",
        b"Ts",
        b"Td",
        b"TD",
        b"Tm",
        b"T*",
        b"Tj",
        b"TJ",
        b"'",
        b'"',
        b"g",
        b"G",
        b"rg",
        b"RG",
        b"k",
        b"K",
        b"BMC",
        b"BDC",
        b"EMC",
    }
)


def _pdf_text(value: Any) -> str | None:
    if isinstance(value, str):
        return _nfc(value)
    if isinstance(value, bytes):
        try:
            return _nfc(value.decode("utf-8"))
        except UnicodeDecodeError:
            return _nfc(value.decode("latin-1"))
    return None


def _parse_text_appearance(normal: Any, reader: Any) -> tuple[str | None, bool]:
    """Extract display text while rejecting external-resource operators."""

    try:
        from pypdf.generic import ContentStream

        stream = ContentStream(normal, reader)
        pieces: list[str] = []
        text_depth = 0
        graphics_depth = 0
        marked_depth = 0
        for operands, raw_operator in stream.operations:
            operator = (
                raw_operator
                if isinstance(raw_operator, bytes)
                else str(raw_operator).encode("ascii", errors="strict")
            )
            if operator not in _SAFE_TEXT_AP_OPERATORS:
                return None, False
            if operator == b"q":
                graphics_depth += 1
            elif operator == b"Q":
                graphics_depth -= 1
                if graphics_depth < 0:
                    return None, False
            elif operator == b"BT":
                text_depth += 1
            elif operator == b"ET":
                text_depth -= 1
                if text_depth < 0:
                    return None, False
            elif operator in {b"BMC", b"BDC"}:
                marked_depth += 1
            elif operator == b"EMC":
                marked_depth -= 1
                if marked_depth < 0:
                    return None, False
            elif operator == b"Tr":
                if not operands or int(operands[0]) not in {0, 1, 2}:
                    return None, False
            elif operator in {b"Tj", b"'", b'"'}:
                if text_depth <= 0 or not operands:
                    return None, False
                item = operands[-1]
                decoded = _pdf_text(item)
                if decoded is None:
                    return None, False
                pieces.append(decoded)
            elif operator == b"TJ":
                if text_depth <= 0 or not operands or not isinstance(operands[0], Sequence):
                    return None, False
                decoded_parts: list[str] = []
                for item in operands[0]:
                    if isinstance(item, Number):
                        continue
                    decoded = _pdf_text(item)
                    if decoded is None:
                        return None, False
                    decoded_parts.append(decoded)
                pieces.append("".join(decoded_parts))
        if text_depth or graphics_depth or marked_depth or not pieces:
            return None, False
        return _nfc(" ".join(piece for piece in pieces if piece)), True
    except Exception:
        return None, False


def _appearance_snapshot(
    widget: Mapping[str, Any], reader: Any, field_type: str
) -> tuple[tuple[str, ...], str | None, str | None, bool]:
    ap = widget.get("/AP")
    if ap is None:
        return (), None, None, False
    ap_object = ap.get_object() if hasattr(ap, "get_object") else ap
    normal = ap_object.get("/N") if isinstance(ap_object, Mapping) else None
    normal = normal.get_object() if hasattr(normal, "get_object") else normal
    states: tuple[str, ...] = ()
    if isinstance(normal, Mapping) and not hasattr(normal, "get_data"):
        states = tuple(sorted(str(key) for key in normal))
    appearance_text: str | None = None
    appearance_safe = False
    if field_type == "/Tx" and normal is not None and hasattr(normal, "get_data"):
        appearance_text, appearance_safe = _parse_text_appearance(normal, reader)
    return (
        states,
        _object_digest(ap_object, skip_keys=frozenset({"/P", "/Parent"})),
        appearance_text,
        appearance_safe,
    )


def _field_structure_digest(widget: Mapping[str, Any]) -> str:
    """Hash immutable widget/field keys while excluding fill-only state."""

    chain: list[dict[str, str | None]] = []
    current: Any = widget
    visited: set[tuple[int, int]] = set()
    immutable_keys = (
        "/Subtype",
        "/Type",
        "/T",
        "/TU",
        "/FT",
        "/Ff",
        "/Q",
        "/MaxLen",
        "/StructParent",
        "/MK",
        "/DV",
        "/A",
        "/AA",
    )
    while isinstance(current, Mapping):
        # Widget geometry is compared separately after a four-decimal
        # normalization.  Producers commonly round harmless coordinate noise
        # while regenerating an appearance stream.
        structural = {
            key: _object_digest(current.get(key), skip_keys=frozenset({"/P", "/Parent"}))
            for key in immutable_keys
            if current.get(key) is not None
        }
        chain.append(structural)
        parent = current.get("/Parent")
        if parent is None:
            break
        if hasattr(parent, "idnum"):
            marker = (int(parent.idnum), int(parent.generation))
            if marker in visited:
                break
            visited.add(marker)
        current = parent.get_object() if hasattr(parent, "get_object") else parent
    return _canonical_hash(chain)


def _field_tree_semantic_digest(acro: Mapping[str, Any] | None) -> str:
    fields = acro.get("/Fields") if isinstance(acro, Mapping) else None
    fields = fields.get_object() if hasattr(fields, "get_object") else fields
    records: list[dict[str, Any]] = []
    active: set[tuple[int, int]] = set()

    def walk(reference: Any, parent_name: str, position: str) -> None:
        marker: tuple[int, int] | None = None
        if hasattr(reference, "idnum"):
            marker = (int(reference.idnum), int(reference.generation))
            if marker in active:
                records.append({"path": position, "cycle": True})
                return
            active.add(marker)
        node = reference.get_object() if hasattr(reference, "get_object") else reference
        if not isinstance(node, Mapping):
            records.append({"path": position, "invalid": True})
            if marker is not None:
                active.discard(marker)
            return
        local_name = str(node.get("/T", ""))
        full_name = ".".join(part for part in (parent_name, local_name) if part)
        kids = node.get("/Kids") or ()
        kids = kids.get_object() if hasattr(kids, "get_object") else kids
        records.append(
            {
                "path": position,
                "name": full_name,
                "subtype": str(node.get("/Subtype", "")),
                "type": str(node.get("/FT", "")),
                "flags": str(node.get("/Ff", "")),
                "label": _string_value(node.get("/TU")) or "",
                "default": _string_value(node.get("/DV")),
                "kid_count": len(kids) if isinstance(kids, Sequence) else -1,
                "actions": _stable_dictionary_digest(
                    {"/A": node.get("/A"), "/AA": node.get("/AA")},
                    skip_keys=frozenset({"/P", "/Parent"}),
                ),
            }
        )
        if isinstance(kids, Sequence):
            for index, child in enumerate(kids):
                walk(child, full_name, f"{position}/{index}")
        if marker is not None:
            active.discard(marker)

    if isinstance(fields, Sequence):
        for index, field in enumerate(fields):
            walk(field, "", str(index))
    else:
        records.append({"fields_invalid": True})
    return _canonical_hash(records)


def _string_value(value: Any) -> str | None:
    if value is None:
        return None
    return _nfc(value)


def _page_geometry(page: Any) -> tuple[str, ...]:
    boxes: list[str] = []
    for name in ("mediabox", "cropbox"):
        box = getattr(page, name)
        boxes.extend(format(float(number), ".4f") for number in box)
    boxes.append(str(int(page.get("/Rotate", 0))))
    return tuple(boxes)


def _page_images(page: Any) -> tuple[str, ...]:
    resources = page.get("/Resources") or {}
    resources = resources.get_object() if hasattr(resources, "get_object") else resources
    xobjects = resources.get("/XObject") if isinstance(resources, Mapping) else None
    xobjects = xobjects.get_object() if hasattr(xobjects, "get_object") else xobjects
    results: list[str] = []
    if isinstance(xobjects, Mapping):
        for name, reference in sorted(xobjects.items(), key=lambda pair: str(pair[0])):
            obj = reference.get_object() if hasattr(reference, "get_object") else reference
            if isinstance(obj, Mapping) and str(obj.get("/Subtype")) == "/Image":
                semantic = {
                    "data": _sha256_bytes(obj.get_data()),
                    "width": str(obj.get("/Width", "")),
                    "height": str(obj.get("/Height", "")),
                    "bpc": str(obj.get("/BitsPerComponent", "")),
                    "colorspace": _object_digest(obj.get("/ColorSpace")),
                    "decode": _object_digest(obj.get("/Decode")),
                    "image_mask": str(obj.get("/ImageMask", "")),
                }
                results.append(f"{name}:{_canonical_hash(semantic)}")
    return tuple(results)


def _page_resource_digest(page: Any) -> str:
    resources = page.get("/Resources") or {}
    resources = resources.get_object() if hasattr(resources, "get_object") else resources
    if not isinstance(resources, Mapping):
        return _canonical_hash({"invalid": True})
    categories: dict[str, Any] = {}
    for category, value in sorted(resources.items(), key=lambda pair: str(pair[0])):
        resolved = value.get_object() if hasattr(value, "get_object") else value
        if isinstance(resolved, Mapping) and not hasattr(resolved, "get_data"):
            categories[str(category)] = {
                str(name): _object_digest(resource, skip_keys=frozenset({"/P", "/Parent"}))
                for name, resource in sorted(resolved.items(), key=lambda pair: str(pair[0]))
            }
        else:
            categories[str(category)] = _object_digest(
                resolved, skip_keys=frozenset({"/P", "/Parent"})
            )
    return _canonical_hash(categories)


def _security_surface(root: Mapping[str, Any], pages: Sequence[Any]) -> str:
    names = root.get("/Names")
    names = names.get_object() if hasattr(names, "get_object") else names
    embedded = names.get("/EmbeddedFiles") if isinstance(names, Mapping) else None
    javascript = names.get("/JavaScript") if isinstance(names, Mapping) else None
    actions: list[str | None] = [
        _object_digest(root.get("/OpenAction")),
        _object_digest(root.get("/A")),
        _object_digest(root.get("/AA")),
        _object_digest(javascript),
        _object_digest(embedded),
        _object_digest(root.get("/AF")),
    ]
    acro = root.get("/AcroForm")
    acro = acro.get_object() if hasattr(acro, "get_object") else acro
    if isinstance(acro, Mapping):
        actions.extend((_object_digest(acro.get("/A")), _object_digest(acro.get("/AA"))))
    for page in pages:
        actions.extend(
            (
                _object_digest(page.get("/A")),
                _object_digest(page.get("/AA")),
                _object_digest(page.get("/AF")),
            )
        )
        for reference in page.get("/Annots") or ():
            annotation = reference.get_object() if hasattr(reference, "get_object") else reference
            if isinstance(annotation, Mapping):
                actions.extend(
                    (
                        _object_digest(annotation.get("/A")),
                        _object_digest(annotation.get("/AA")),
                        _object_digest(annotation.get("/FS")),
                        _object_digest(annotation.get("/AF")),
                    )
                )
    payload = json.dumps(actions, separators=(",", ":"), ensure_ascii=True)
    return _sha256_bytes(payload.encode("ascii"))


def snapshot_pdf(path: str | Path) -> PdfSnapshot:
    """Read a PDF with pypdf and capture the oracle's structural state."""

    try:
        import pypdf
    except ImportError as exc:  # pragma: no cover - exercised in frozen image
        raise RuntimeError("pypdf unavailable") from exc
    if pypdf.__version__ != REQUIRED_PYPDF_VERSION:
        raise RuntimeError("unexpected pypdf version")

    reader = pypdf.PdfReader(str(path))
    widgets: dict[str, WidgetSnapshot] = {}
    non_widgets: list[tuple[str, ...]] = []
    page_content: list[str] = []
    page_images: list[tuple[str, ...]] = []
    page_resources: list[str] = []
    geometry: list[tuple[str, ...]] = []
    for page_index, page in enumerate(reader.pages):
        geometry.append(_page_geometry(page))
        contents = page.get_contents()
        page_content.append(_sha256_bytes(contents.get_data() if contents is not None else b""))
        page_images.append(_page_images(page))
        page_resources.append(_page_resource_digest(page))
        page_non_widgets: list[str] = []
        for reference in page.get("/Annots") or ():
            annotation = reference.get_object() if hasattr(reference, "get_object") else reference
            if str(annotation.get("/Subtype")) != "/Widget":
                digest = _stable_dictionary_digest(
                    annotation, skip_keys=frozenset({"/P", "/Parent"})
                )
                page_non_widgets.append(digest or "")
                continue
            name = _full_field_name(annotation)
            if not name or name in widgets:
                raise RuntimeError("ambiguous widget name")
            rect = tuple(round(float(item), 4) for item in annotation.get("/Rect", ()))
            if len(rect) != 4:
                raise RuntimeError("invalid widget rectangle")
            field_type = str(_inherited(annotation, "/FT") or "")
            states, ap_digest, appearance_text, appearance_safe = _appearance_snapshot(
                annotation, reader, field_type
            )
            widgets[name] = WidgetSnapshot(
                full_name=name,
                page_index=page_index,
                rect=rect,  # type: ignore[arg-type]
                field_type=field_type,
                field_flags=int(_inherited(annotation, "/Ff") or 0),
                label=_string_value(_inherited(annotation, "/TU")) or "",
                value=_string_value(_inherited(annotation, "/V")),
                appearance_state=_string_value(annotation.get("/AS")),
                appearance_states=states,
                appearance_digest=ap_digest,
                structure_digest=_field_structure_digest(annotation),
                default_value=_string_value(_inherited(annotation, "/DV")),
                appearance_text=appearance_text,
                appearance_safe=appearance_safe,
            )
        non_widgets.append(tuple(sorted(page_non_widgets)))

    root = reader.trailer["/Root"].get_object()
    acro = root.get("/AcroForm")
    acro = acro.get_object() if hasattr(acro, "get_object") else acro
    xfa = acro.get("/XFA") if isinstance(acro, Mapping) else None
    return PdfSnapshot(
        page_count=len(reader.pages),
        page_geometry=tuple(geometry),
        page_content_digests=tuple(page_content),
        page_image_digests=tuple(page_images),
        non_widget_annotation_digests=tuple(non_widgets),
        xfa_digest=_object_digest(xfa),
        security_digest=_security_surface(root, reader.pages),
        widgets=widgets,
        field_tree_digest=_field_tree_semantic_digest(acro),
        page_resource_digests=tuple(page_resources),
    )


def _command_version(command: str) -> str:
    completed = subprocess.run(
        [command, "-v"], check=False, capture_output=True, text=True, timeout=10
    )
    first_line = (completed.stderr or completed.stdout).splitlines()
    return first_line[0] if first_line else ""


def _poppler_text(path: Path, command: str) -> str:
    completed = subprocess.run(
        [command, "-layout", "-enc", "UTF-8", str(path), "-"],
        check=True,
        capture_output=True,
        timeout=30,
    )
    return completed.stdout.decode("utf-8", errors="strict")


def _poppler_bbox_words(path: Path, command: str) -> tuple[BBoxWord, ...]:
    completed = subprocess.run(
        [command, "-bbox-layout", "-enc", "UTF-8", str(path), "-"],
        check=True,
        capture_output=True,
        timeout=30,
    )
    root = ET.fromstring(completed.stdout.decode("utf-8", errors="strict"))
    words: list[BBoxWord] = []
    pages = [element for element in root.iter() if element.tag.rsplit("}", 1)[-1] == "page"]
    for page_index, page in enumerate(pages):
        for element in page.iter():
            if element.tag.rsplit("}", 1)[-1] != "word":
                continue
            words.append(
                BBoxWord(
                    page_index=page_index,
                    text=_nfc(element.text or ""),
                    x_min=float(element.attrib["xMin"]),
                    y_min=float(element.attrib["yMin"]),
                    x_max=float(element.attrib["xMax"]),
                    y_max=float(element.attrib["yMax"]),
                )
            )
    return tuple(words)


def _bbox_binding_failures(
    words: Sequence[BBoxWord], filled: PdfSnapshot
) -> tuple[str, ...]:
    failures: list[str] = []
    for semantic, suffix in TEXT_LOCATORS.items():
        name = _resolve_locator(filled.widgets, suffix)
        if name is None:
            failures.append(f"poppler_bbox_mismatch:{semantic}")
            continue
        widget = filled.widgets[name]
        page_height = float(filled.page_geometry[widget.page_index][3])
        left, bottom, right, top = widget.rect
        top_y = page_height - top
        bottom_y = page_height - bottom
        tolerance = 2.0
        local_words = [
            word.text
            for word in words
            if word.page_index == widget.page_index
            and left - tolerance <= (word.x_min + word.x_max) / 2 <= right + tolerance
            and top_y - tolerance <= (word.y_min + word.y_max) / 2 <= bottom_y + tolerance
        ]
        local_text = _display_normalize(" ".join(local_words))
        expected = _display_normalize(widget.appearance_text or widget.value)
        if not expected or expected not in local_text:
            failures.append(f"poppler_bbox_mismatch:{semantic}")
    return tuple(failures)


def _visible_text_failures(
    text: str,
    plan: SemanticPlan,
    filled_widgets: Mapping[str, WidgetSnapshot],
) -> tuple[str, ...]:
    flattened = " ".join(_nfc(text).split())
    failures: list[str] = []
    for semantic, value in plan.text_values.items():
        expected = " ".join(value.split())
        if expected not in flattened:
            failures.append(f"pdftotext_missing:{semantic}")
    for semantic, digits in plan.phone_digits.items():
        name = _resolve_locator(filled_widgets, TEXT_LOCATORS[semantic])
        actual = filled_widgets[name].value if name is not None else None
        rendered_value = " ".join((actual or "").split())
        if (
            not rendered_value
            or rendered_value not in flattened
            or _phone_digits(actual or "") != digits
        ):
            failures.append(f"pdftotext_missing:{semantic}")
    amount_name = _resolve_locator(filled_widgets, TEXT_LOCATORS["claim.amount"])
    amount_actual = filled_widgets[amount_name].value if amount_name is not None else None
    if (
        not amount_actual
        or _display_normalize(amount_actual) not in flattened
        or _money_decimal(amount_actual) != plan.amount
    ):
        failures.append("pdftotext_missing:claim.amount")
    for semantic in ("claim.reason", "claim.calculation"):
        name = _resolve_locator(filled_widgets, TEXT_LOCATORS[semantic])
        actual = filled_widgets[name].value if name is not None else None
        expected = " ".join((actual or "").split())
        if not expected or expected not in flattened:
            failures.append(f"pdftotext_missing:{semantic}")
    reason_name = _resolve_locator(filled_widgets, TEXT_LOCATORS["claim.reason"])
    reason = filled_widgets[reason_name].value if reason_name is not None else None
    if not _reason_has_required_semantics(reason or ""):
        failures.append("pdftotext_reason_semantics_mismatch")
    calculation_name = _resolve_locator(filled_widgets, TEXT_LOCATORS["claim.calculation"])
    calculation = filled_widgets[calculation_name].value if calculation_name is not None else None
    if not _calculation_has_required_semantics(calculation or "", plan.amount, plan.contract):
        failures.append("pdftotext_calculation_semantics_mismatch")
    return tuple(failures)


def _render_pages(path: Path, prefix: Path, command: str, dpi: int) -> None:
    subprocess.run(
        [command, "-png", "-r", str(dpi), str(path), str(prefix)],
        check=True,
        capture_output=True,
        timeout=60,
    )


def _crop_diff_pixels(
    blank_image: Any,
    filled_image: Any,
    rect: tuple[float, float, float, float],
    page_height_points: float,
    dpi: int,
) -> int:
    from PIL import ImageChops

    scale = dpi / 72.0
    left, bottom, right, top = rect
    pad = max(2, int(math.ceil(scale)))
    box = (
        max(0, int(math.floor(left * scale)) - pad),
        max(0, int(math.floor((page_height_points - top) * scale)) - pad),
        min(blank_image.width, int(math.ceil(right * scale)) + pad),
        min(blank_image.height, int(math.ceil((page_height_points - bottom) * scale)) + pad),
    )
    before = blank_image.convert("RGB").crop(box)
    after = filled_image.convert("RGB").crop(box)
    difference = ImageChops.difference(before, after)
    return sum(1 for pixel in difference.getdata() if max(pixel) >= 12)


def _pixel_visibility_failures(
    blank_path: Path,
    filled_path: Path,
    blank: PdfSnapshot,
    filled: PdfSnapshot,
    decision: SnapshotDecision,
    plan: SemanticPlan,
    *,
    pdftoppm_bin: str,
    dpi: int,
) -> tuple[str, ...]:
    from PIL import Image

    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="sc100-oracle-") as directory:
        root = Path(directory)
        _render_pages(blank_path, root / "blank", pdftoppm_bin, dpi)
        _render_pages(filled_path, root / "filled", pdftoppm_bin, dpi)
        images: dict[int, tuple[Any, Any]] = {}
        for page_index in range(blank.page_count):
            images[page_index] = (
                Image.open(root / f"blank-{page_index + 1}.png"),
                Image.open(root / f"filled-{page_index + 1}.png"),
            )
        selected_names: set[str] = set()
        for semantic, suffix in BUTTON_LOCATORS.items():
            if plan.button_states[semantic] is not None:
                name = _resolve_locator(blank.widgets, suffix)
                if name:
                    selected_names.add(name)
        text_names = {
            name
            for suffix in TEXT_LOCATORS.values()
            if (name := _resolve_locator(blank.widgets, suffix)) is not None
        }
        for name in sorted(text_names | selected_names):
            widget = filled.widgets[name]
            before_image, after_image = images[widget.page_index]
            page_height = float(blank.page_geometry[widget.page_index][3])
            changed = _crop_diff_pixels(
                before_image, after_image, widget.rect, page_height, dpi
            )
            if changed < 4:
                semantic = next(
                    (
                        key
                        for key, suffix in {**TEXT_LOCATORS, **BUTTON_LOCATORS}.items()
                        if name.endswith(suffix)
                    ),
                    "target",
                )
                failures.append(f"poppler_local_invisible:{semantic}")
        for before_image, after_image in images.values():
            before_image.close()
            after_image.close()
    return tuple(failures)


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return _sha256_bytes(payload.encode("ascii"))


def build_receipt(
    *,
    blank_sha256: str,
    filled_sha256: str,
    gold_sha256: str,
    failures: Iterable[str],
    target_count: int,
    runtime: Mapping[str, str],
) -> dict[str, Any]:
    """Build a receipt containing hashes and public check codes, never case text."""

    failure_codes = tuple(sorted(set(failures)))
    receipt: dict[str, Any] = {
        "schema": ORACLE_SCHEMA,
        "qualified": not failure_codes,
        "failure_codes": list(failure_codes),
        "bindings": {
            "blank_sha256": blank_sha256,
            "filled_sha256": filled_sha256,
            "semantic_gold_sha256": gold_sha256,
        },
        "target_widget_count": target_count,
        "runtime": dict(sorted(runtime.items())),
    }
    receipt["receipt_sha256"] = _canonical_hash(receipt)
    return receipt


def qualify_sc100_shadow(
    blank_pdf: str | Path,
    filled_pdf: str | Path,
    semantic_gold: Mapping[str, Any],
    *,
    pdftotext_bin: str = "pdftotext",
    pdftoppm_bin: str = "pdftoppm",
    render_dpi: int = 144,
) -> dict[str, Any]:
    """Run the frozen offline SC-100 qualification and return a redacted receipt."""

    failures: list[str] = []
    target_count = 0
    runtime: dict[str, str] = {}
    blank_hash = "unavailable"
    filled_hash = "unavailable"
    gold_hash = "unavailable"
    blank_path: Path | None = None
    filled_path: Path | None = None
    try:
        try:
            gold_hash = _canonical_hash(semantic_gold)
        except Exception:
            failures.append("gold_serialization_failed")
        try:
            blank_path = Path(blank_pdf)
            filled_path = Path(filled_pdf)
            blank_hash = _sha256_file(blank_path)
            filled_hash = _sha256_file(filled_path)
        except Exception:
            failures.append("path_or_hash_failed")
        if blank_hash != PUBLIC_BLANK_SHA256:
            failures.append("blank_binding_mismatch")

        if blank_path is not None and filled_path is not None:
            try:
                import pypdf
                import PIL

                runtime["pypdf"] = str(pypdf.__version__)
                runtime["pillow"] = str(PIL.__version__)
                runtime["pdftotext"] = _command_version(pdftotext_bin)
                runtime["pdftoppm"] = _command_version(pdftoppm_bin)
                if pypdf.__version__ != REQUIRED_PYPDF_VERSION:
                    failures.append("runtime_mismatch:pypdf")
                if REQUIRED_POPPLER_VERSION not in runtime["pdftotext"]:
                    failures.append("runtime_mismatch:pdftotext")
                if REQUIRED_POPPLER_VERSION not in runtime["pdftoppm"]:
                    failures.append("runtime_mismatch:pdftoppm")

                plan = compile_semantic_plan(semantic_gold)
                blank = snapshot_pdf(blank_path)
                filled = snapshot_pdf(filled_path)
                failures.extend(compare_document_snapshots(blank, filled))
                decision = compare_widget_snapshots(blank.widgets, filled.widgets, plan)
                failures.extend(decision.failure_codes)
                target_count = len(decision.target_widget_names)
                try:
                    visible_text = _poppler_text(filled_path, pdftotext_bin)
                    failures.extend(
                        _visible_text_failures(visible_text, plan, filled.widgets)
                    )
                except Exception:
                    failures.append("pdftotext_failed")
                try:
                    bbox_words = _poppler_bbox_words(filled_path, pdftotext_bin)
                    failures.extend(_bbox_binding_failures(bbox_words, filled))
                except Exception:
                    failures.append("poppler_bbox_failed")
                try:
                    failures.extend(
                        _pixel_visibility_failures(
                            blank_path,
                            filled_path,
                            blank,
                            filled,
                            decision,
                            plan,
                            pdftoppm_bin=pdftoppm_bin,
                            dpi=render_dpi,
                        )
                    )
                except Exception:
                    failures.append("poppler_render_failed")
            except GoldValidationError as exc:
                failures.extend(exc.codes)
            except Exception:
                failures.append("oracle_execution_failed")
    except Exception:
        failures.append("oracle_execution_failed")

    return build_receipt(
        blank_sha256=blank_hash,
        filled_sha256=filled_hash,
        gold_sha256=gold_hash,
        failures=failures,
        target_count=target_count,
        runtime=runtime,
    )


__all__ = [
    "BUTTON_LOCATORS",
    "BUTTON_ON_STATES",
    "GoldValidationError",
    "ORACLE_SCHEMA",
    "PUBLIC_BLANK_SHA256",
    "PdfSnapshot",
    "SemanticPlan",
    "SnapshotDecision",
    "TEXT_LOCATORS",
    "WidgetSnapshot",
    "build_receipt",
    "compare_document_snapshots",
    "compare_widget_snapshots",
    "compile_semantic_plan",
    "qualify_sc100_shadow",
    "snapshot_pdf",
]
