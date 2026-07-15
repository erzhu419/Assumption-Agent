"""Closed, executable SC-100 form-filling operator.

This module is deliberately narrow.  It accepts only the public case-description
grammar needed for a single-plaintiff, single-defendant California security-
deposit claim, compiles those facts to a fixed AcroForm mutation plan, writes a
fresh SC-100 blank with PyMuPDF, and then reopens and reconciles the result.

It does not inspect verifier files, item identifiers, historical solutions, or
model output.  Unsupported or ambiguous instructions fail closed before the PDF
is changed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable, Mapping, Sequence


OPERATOR_VERSION = "sc100_security_deposit_acroform_v1"
PDFTOTEXT_PATH = Path("/usr/bin/pdftotext")


class SC100OperatorError(ValueError):
    """Raised when the closed grammar or reconciliation contract is violated."""


@dataclass(frozen=True)
class Address:
    street: str
    city: str
    state: str
    zip_code: str


@dataclass(frozen=True)
class SC100CaseFacts:
    plaintiff_name: str
    plaintiff_address: Address
    plaintiff_phone: str
    plaintiff_email: str
    defendant_name: str
    defendant_address: Address
    defendant_phone: str
    claim_amount: int
    contract_kind: str
    incident_start: str
    incident_end: str
    declaration_date: str
    asked_to_pay: bool
    venue_is_defendant_residence: bool
    more_than_twelve_claims: bool
    public_entity: bool = False
    attorney_fee_dispute: bool = False


@dataclass(frozen=True)
class FieldMutation:
    field_name: str
    field_type: str
    value: str


@dataclass(frozen=True)
class SC100Plan:
    operator_version: str
    mutations: tuple[FieldMutation, ...]

    @property
    def plan_hash(self) -> str:
        return _stable_hash(
            {
                "operator_version": self.operator_version,
                "mutations": [asdict(mutation) for mutation in self.mutations],
            }
        )


_NAME = r"[A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){1,3}"
_ADDRESS = re.compile(
    r"(?P<street>\d{1,6}\s+[^,.]+?),\s*"
    r"(?P<city>[A-Za-z][A-Za-z .'-]+?),\s*"
    r"(?P<state>CA)\s+(?P<zip>\d{5})(?!\d)",
    re.IGNORECASE,
)
_EMAIL = re.compile(r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b", re.I)
_PHONE = re.compile(r"(?<!\d)(?:\+?1[ .()\-]*)?(\d{3})[ .()\-]*(\d{3})[ .\-]*(\d{4})(?!\d)")
_ISO_DATE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
_AMOUNT = re.compile(r"\$\s*([0-9][0-9,]*)\s+security deposit\b", re.I)
_CONTRACT = re.compile(
    r"\bsigned\s+(room rental agreement|roommate sublease contract)\b",
    re.I,
)
_NATURAL_DATE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+(\d{1,2}),\s*(20\d{2})\b",
    re.I,
)


# The only fields the operator can express.  No arbitrary field name or XFA
# stream operation can be supplied by an instruction.
_TEXT_FIELDS: Mapping[str, str] = {
    "caption_page2": "SC-100[0].Page2[0].PxCaption[0].Plaintiff[0]",
    "plaintiff_name": "SC-100[0].Page2[0].List1[0].Item1[0].PlaintiffName1[0]",
    "plaintiff_phone": "SC-100[0].Page2[0].List1[0].Item1[0].PlaintiffPhone1[0]",
    "plaintiff_street": "SC-100[0].Page2[0].List1[0].Item1[0].PlaintiffAddress1[0]",
    "plaintiff_city": "SC-100[0].Page2[0].List1[0].Item1[0].PlaintiffCity1[0]",
    "plaintiff_state": "SC-100[0].Page2[0].List1[0].Item1[0].PlaintiffState1[0]",
    "plaintiff_zip": "SC-100[0].Page2[0].List1[0].Item1[0].PlaintiffZip1[0]",
    "plaintiff_email": "SC-100[0].Page2[0].List1[0].Item1[0].EmailAdd1[0]",
    "defendant_name": "SC-100[0].Page2[0].List2[0].item2[0].DefendantName1[0]",
    "defendant_phone": "SC-100[0].Page2[0].List2[0].item2[0].DefendantPhone1[0]",
    "defendant_street": "SC-100[0].Page2[0].List2[0].item2[0].DefendantAddress1[0]",
    "defendant_city": "SC-100[0].Page2[0].List2[0].item2[0].DefendantCity1[0]",
    "defendant_state": "SC-100[0].Page2[0].List2[0].item2[0].DefendantState1[0]",
    "defendant_zip": "SC-100[0].Page2[0].List2[0].item2[0].DefendantZip1[0]",
    "claim_amount": "SC-100[0].Page2[0].List3[0].PlaintiffClaimAmount1[0]",
    "claim_reason": "SC-100[0].Page2[0].List3[0].Lia[0].FillField2[0]",
    "caption_page3": "SC-100[0].Page3[0].PxCaption[0].Plaintiff[0]",
    "incident_start": "SC-100[0].Page3[0].List3[0].Lib[0].Date2[0]",
    "incident_end": "SC-100[0].Page3[0].List3[0].Lib[0].Date3[0]",
    "calculation": "SC-100[0].Page3[0].List3[0].Lic[0].FillField1[0]",
    "venue_zip": "SC-100[0].Page3[0].List6[0].item6[0].ZipCode1[0]",
    "caption_page4": "SC-100[0].Page4[0].PxCaption[0].Plaintiff[0]",
    "declaration_date": "SC-100[0].Page4[0].Sign[0].Date1[0]",
    "signature_name": "SC-100[0].Page4[0].Sign[0].PlaintiffName1[0]",
}

_BUTTON_FIELDS: Mapping[str, tuple[str, str]] = {
    "asked_to_pay_yes": (
        "SC-100[0].Page3[0].List4[0].Item4[0].Checkbox50[0]",
        "1",
    ),
    "venue_defendant_residence": (
        "SC-100[0].Page3[0].List5[0].Lia[0].Checkbox5cb[0]",
        "1",
    ),
    "attorney_fee_no": (
        "SC-100[0].Page3[0].List7[0].item7[0].Checkbox60[1]",
        "2",
    ),
    "public_entity_no": (
        "SC-100[0].Page3[0].List8[0].item8[0].Checkbox61[1]",
        "2",
    ),
    "more_than_twelve_yes": (
        "SC-100[0].Page4[0].List9[0].Item9[0].Checkbox62[0]",
        "1",
    ),
    "more_than_twelve_no": (
        "SC-100[0].Page4[0].List9[0].Item9[0].Checkbox62[1]",
        "2",
    ),
    "over_2500_yes": (
        "SC-100[0].Page4[0].List10[0].li10[0].Checkbox63[0]",
        "1",
    ),
    "over_2500_no": (
        "SC-100[0].Page4[0].List10[0].li10[0].Checkbox63[1]",
        "2",
    ),
}

_BUTTON_GROUPS: tuple[tuple[str, ...], ...] = (
    (
        "SC-100[0].Page3[0].List4[0].Item4[0].Checkbox50[0]",
        "SC-100[0].Page3[0].List4[0].Item4[0].Checkbox50[1]",
    ),
    tuple(
        f"SC-100[0].Page3[0].List5[0].Li{suffix}[0].Checkbox5cb[0]"
        for suffix in ("a", "b", "c", "d", "e")
    ),
    (
        "SC-100[0].Page3[0].List7[0].item7[0].Checkbox60[0]",
        "SC-100[0].Page3[0].List7[0].item7[0].Checkbox60[1]",
    ),
    (
        "SC-100[0].Page3[0].List8[0].item8[0].Checkbox61[0]",
        "SC-100[0].Page3[0].List8[0].item8[0].Checkbox61[1]",
    ),
    (
        "SC-100[0].Page4[0].List9[0].Item9[0].Checkbox62[0]",
        "SC-100[0].Page4[0].List9[0].Item9[0].Checkbox62[1]",
    ),
    (
        "SC-100[0].Page4[0].List10[0].li10[0].Checkbox63[0]",
        "SC-100[0].Page4[0].List10[0].li10[0].Checkbox63[1]",
    ),
)

_FORBIDDEN_FIELD_FRAGMENTS = (
    ".Page1[0].",
    "PlaintiffName2[0]",
    "PlaintiffPhone2[0]",
    "PlaintiffAddress2[0]",
    "PlaintiffCity2[0]",
    "PlaintiffState2[0]",
    "PlaintiffZip2[0]",
    "EmailAdd2[0]",
    "DefendantName2[0]",
    "DefendantJob1[0]",
    "DefendantAddress2[0]",
    "DefendantCity2[0]",
    "DefendantState2[0]",
    "DefendantZip2[0]",
    "PlaintiffMailing",
    "DefendantMailing",
    ".List3[0].Lib[0].Date1[0]",
    ".List5[0].Lie[0].FillField55[0]",
    "Checkbox11[0]",
    "Checkbox14[0]",
    "Date4[0]",
    ".Sign[0].Date2[0]",
    ".Sign[0].PlaintiffName2[0]",
)


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _one_match(patterns: Iterable[re.Pattern[str]], text: str, label: str) -> str:
    values: list[str] = []
    for pattern in patterns:
        values.extend(match.group(1).strip() for match in pattern.finditer(text))
    unique = list(dict.fromkeys(values))
    if len(unique) != 1:
        raise SC100OperatorError(f"expected exactly one {label}; found {len(unique)}")
    return unique[0]


def _parse_name(instruction: str, *, plaintiff: bool) -> str:
    if plaintiff:
        patterns = (
            re.compile(rf"\bI am ({_NAME})\.", re.I),
            re.compile(rf"\bMy name is ({_NAME})(?:,|\.)", re.I),
            re.compile(rf"\bplaintiff is ({_NAME})(?:,|\.)", re.I),
        )
        label = "plaintiff name"
    else:
        patterns = (
            re.compile(rf"\b(?:want to sue|am suing|suing) ({_NAME})(?:,|\s+who\b)", re.I),
            re.compile(rf"\bdefendant is ({_NAME})(?:,|\.)", re.I),
        )
        label = "defendant name"
    return _one_match(patterns, instruction, label).title()


def _parse_addresses(instruction: str) -> tuple[Address, Address]:
    matches = list(_ADDRESS.finditer(instruction))
    if len(matches) != 2:
        raise SC100OperatorError(f"expected exactly two California addresses; found {len(matches)}")
    addresses = tuple(
        Address(
            street=match.group("street").strip(),
            city=match.group("city").strip(),
            state=match.group("state").upper(),
            zip_code=match.group("zip"),
        )
        for match in matches
    )
    return addresses[0], addresses[1]


def _parse_natural_dates(instruction: str) -> tuple[str, ...]:
    values: list[str] = []
    for match in _NATURAL_DATE.finditer(instruction):
        raw = " ".join(match.groups())
        parsed = datetime.strptime(raw.title(), "%B %d %Y")
        values.append(parsed.strftime("%Y-%m-%d"))
    return tuple(values)


def parse_instruction(instruction: str) -> SC100CaseFacts:
    """Parse the closed SC-100 case-description grammar or fail closed."""

    if not isinstance(instruction, str) or not instruction.strip():
        raise SC100OperatorError("instruction is empty")
    text = " ".join(instruction.split())

    plaintiff_name = _parse_name(text, plaintiff=True)
    defendant_name = _parse_name(text, plaintiff=False)
    plaintiff_address, defendant_address = _parse_addresses(text)

    phones = ["".join(match.groups()) for match in _PHONE.finditer(text)]
    if len(phones) != 2:
        raise SC100OperatorError(f"expected exactly two phone numbers; found {len(phones)}")
    emails = _EMAIL.findall(text)
    if len(emails) != 1:
        raise SC100OperatorError(f"expected exactly one email; found {len(emails)}")

    amount_match = _AMOUNT.search(text)
    if amount_match is None:
        raise SC100OperatorError("security-deposit amount is missing")
    amount = int(amount_match.group(1).replace(",", ""))
    if amount <= 0 or amount > 12_500:
        raise SC100OperatorError("claim amount is outside the closed SC-100 range")

    contract_match = _CONTRACT.search(text)
    if contract_match is None:
        raise SC100OperatorError("supported signed housing contract is missing")
    contract_kind = contract_match.group(1).lower()

    iso_dates = _ISO_DATE.findall(text)
    if len(iso_dates) != 2:
        raise SC100OperatorError(f"expected exactly two ISO dispute dates; found {len(iso_dates)}")
    incident_start, incident_end = iso_dates
    if incident_start > incident_end:
        raise SC100OperatorError("dispute start date is after end date")

    natural_dates = _parse_natural_dates(text)
    if len(set(natural_dates)) != 1:
        raise SC100OperatorError("expected exactly one unambiguous declaration date")
    declaration_date = natural_dates[0]
    if declaration_date != incident_end:
        raise SC100OperatorError("declaration date differs from the closed case end-date rule")

    lower = text.lower()
    asked_to_pay = any(
        phrase in lower
        for phrase in (
            "requested payment",
            "asking for repayment",
            "asked for repayment",
            "requested repayment",
            "requesting the deposit back",
        )
    )
    if not asked_to_pay or "text message" not in lower:
        raise SC100OperatorError("pre-suit payment request is not explicit")

    venue_is_defendant_residence = any(
        phrase in lower
        for phrase in (
            "where the defendant lives",
            "location where the defendant lives",
            "venue where the defendant lives",
        )
    )
    if not venue_is_defendant_residence:
        raise SC100OperatorError("defendant-residence venue is not explicit")

    has_more_than_twelve = "more than 12" in lower
    is_first_case = any(
        phrase in lower
        for phrase in ("first time filing", "first small claims case", "my first time")
    )
    if has_more_than_twelve == is_first_case:
        raise SC100OperatorError("small-claims history is ambiguous")

    if any(term in lower for term in ("public entity", "government agency", "city of ")):
        raise SC100OperatorError("public-entity cases are outside the closed grammar")
    if any(term in lower for term in ("attorney fee", "legal fee", "client-fee")):
        raise SC100OperatorError("attorney-fee cases are outside the closed grammar")

    return SC100CaseFacts(
        plaintiff_name=plaintiff_name,
        plaintiff_address=plaintiff_address,
        plaintiff_phone=phones[0],
        plaintiff_email=emails[0],
        defendant_name=defendant_name,
        defendant_address=defendant_address,
        defendant_phone=phones[1],
        claim_amount=amount,
        contract_kind=contract_kind,
        incident_start=incident_start,
        incident_end=incident_end,
        declaration_date=declaration_date,
        asked_to_pay=True,
        venue_is_defendant_residence=True,
        more_than_twelve_claims=has_more_than_twelve,
    )


def compile_plan(facts: SC100CaseFacts) -> SC100Plan:
    """Compile facts into the fixed AcroForm write vocabulary."""

    if facts.public_entity or facts.attorney_fee_dispute:
        raise SC100OperatorError("unsupported case type reached the compiler")
    if not facts.asked_to_pay or not facts.venue_is_defendant_residence:
        raise SC100OperatorError("required SC-100 preconditions are false")

    text_values = {
        "caption_page2": facts.plaintiff_name,
        "plaintiff_name": facts.plaintiff_name,
        "plaintiff_phone": facts.plaintiff_phone,
        "plaintiff_street": facts.plaintiff_address.street,
        "plaintiff_city": facts.plaintiff_address.city,
        "plaintiff_state": facts.plaintiff_address.state,
        "plaintiff_zip": facts.plaintiff_address.zip_code,
        "plaintiff_email": facts.plaintiff_email,
        "defendant_name": facts.defendant_name,
        "defendant_phone": facts.defendant_phone,
        "defendant_street": facts.defendant_address.street,
        "defendant_city": facts.defendant_address.city,
        "defendant_state": facts.defendant_address.state,
        "defendant_zip": facts.defendant_address.zip_code,
        "claim_amount": str(facts.claim_amount),
        "claim_reason": (
            f"Defendant did not return my ${facts.claim_amount} security deposit "
            "after move-out."
        ),
        "caption_page3": facts.plaintiff_name,
        "incident_start": facts.incident_start,
        "incident_end": facts.incident_end,
        "calculation": (
            f"${facts.claim_amount} security deposit documented in the signed "
            f"{facts.contract_kind}."
        ),
        "venue_zip": facts.defendant_address.zip_code,
        "caption_page4": facts.plaintiff_name,
        "declaration_date": facts.declaration_date,
        "signature_name": facts.plaintiff_name,
    }
    button_keys = [
        "asked_to_pay_yes",
        "venue_defendant_residence",
        "attorney_fee_no",
        "public_entity_no",
        "more_than_twelve_yes" if facts.more_than_twelve_claims else "more_than_twelve_no",
        "over_2500_yes" if facts.claim_amount > 2_500 else "over_2500_no",
    ]

    mutations = [
        FieldMutation(field_name=_TEXT_FIELDS[key], field_type="text", value=value)
        for key, value in text_values.items()
    ]
    mutations.extend(
        FieldMutation(
            field_name=_BUTTON_FIELDS[key][0],
            field_type="button",
            value=_BUTTON_FIELDS[key][1],
        )
        for key in button_keys
    )
    names = [mutation.field_name for mutation in mutations]
    if len(names) != len(set(names)):
        raise AssertionError("compiler emitted duplicate field mutations")
    return SC100Plan(operator_version=OPERATOR_VERSION, mutations=tuple(mutations))


def _widget_inventory(document: Any) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for page_number, page in enumerate(document):
        for widget in page.widgets() or ():
            name = str(widget.field_name or "")
            if not name or name in inventory:
                raise SC100OperatorError("blank form contains missing or duplicate widget names")
            as_state = document.xref_get_key(widget.xref, "AS")[1]
            raw_value = document.xref_get_key(widget.xref, "V")[1]
            appearance = document.xref_get_key(widget.xref, "AP")[1]
            on_state = None if widget.on_state() is None else str(widget.on_state())
            inventory[name] = {
                "page": page_number + 1,
                "type": str(widget.field_type_string or ""),
                "label": str(widget.field_label or ""),
                "value": "" if widget.field_value is None else str(widget.field_value),
                "on_state": on_state,
                "appearance_state": as_state,
                "raw_value": raw_value,
                "has_on_appearance": bool(
                    on_state is not None
                    and f"/{on_state}" in appearance
                    and "/Off" in appearance
                ),
            }
    return inventory


def _field_structure(inventory: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        name: {
            "page": row["page"],
            "type": row["type"],
            "label": row["label"],
            "on_state": row["on_state"],
            "has_on_appearance": row["has_on_appearance"],
        }
        for name, row in sorted(inventory.items())
    }


def _extract_pdf_text(path: Path) -> str:
    if not PDFTOTEXT_PATH.is_file():
        raise SC100OperatorError(f"pdftotext missing at {PDFTOTEXT_PATH}")
    completed = subprocess.run(
        [str(PDFTOTEXT_PATH), "-layout", str(path), "-"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise SC100OperatorError("pdftotext failed during reconciliation")
    return completed.stdout


def _normalized(value: str) -> str:
    return " ".join(value.lower().split())


def apply_plan(
    *,
    blank_pdf: Path,
    output_pdf: Path,
    plan: SC100Plan,
    facts: SC100CaseFacts,
) -> dict[str, Any]:
    """Write and reconcile a plan, returning a content-redacted receipt."""

    blank_pdf = blank_pdf.resolve()
    output_pdf = output_pdf.resolve()
    if blank_pdf == output_pdf:
        raise SC100OperatorError("blank and output paths must differ")
    if not blank_pdf.is_file():
        raise FileNotFoundError(blank_pdf)
    if output_pdf.exists():
        raise FileExistsError(output_pdf)

    try:
        import fitz
    except ImportError as exc:  # pragma: no cover - environment contract
        raise RuntimeError("PyMuPDF is required for the SC-100 operator") from exc

    input_sha = _sha256(blank_pdf)
    target = {mutation.field_name: mutation for mutation in plan.mutations}
    with fitz.open(blank_pdf) as document:
        if document.page_count != 6:
            raise SC100OperatorError("SC-100 blank must contain exactly six pages")
        before = _widget_inventory(document)
        if set(target) - set(before):
            raise SC100OperatorError("compiled field is absent from the blank form")
        for name, mutation in target.items():
            actual_type = str(before[name]["type"])
            expected_type = "Text" if mutation.field_type == "text" else "CheckBox"
            if actual_type != expected_type:
                raise SC100OperatorError(f"field type mismatch for {name}")
            if mutation.field_type == "button" and before[name]["on_state"] != mutation.value:
                raise SC100OperatorError(f"button on-state mismatch for {name}")
            if mutation.field_type == "button" and not before[name]["has_on_appearance"]:
                raise SC100OperatorError(f"button appearance dictionary mismatch for {name}")

        for page in document:
            for widget in page.widgets() or ():
                mutation = target.get(str(widget.field_name or ""))
                if mutation is None:
                    continue
                if mutation.field_type == "text":
                    widget.field_value = mutation.value
                    widget.update()
                else:
                    # PyMuPDF's Widget.update() regenerates a visually blank
                    # ZapfDingbats appearance for this hybrid XFA/AcroForm.
                    # Preserve the signed blank's /AP dictionary and select its
                    # existing named appearance directly.
                    document.xref_set_key(widget.xref, "AS", f"/{mutation.value}")
                    document.xref_set_key(widget.xref, "V", f"/{mutation.value}")
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        document.save(
            output_pdf,
            garbage=4,
            deflate=True,
            clean=False,
            no_new_id=True,
        )

    if _sha256(blank_pdf) != input_sha:
        output_pdf.unlink(missing_ok=True)
        raise SC100OperatorError("source blank changed during execution")

    with fitz.open(output_pdf) as reopened:
        if reopened.page_count != 6:
            raise SC100OperatorError("output page count changed")
        after = _widget_inventory(reopened)

    before_structure = _field_structure(before)
    after_structure = _field_structure(after)
    if before_structure != after_structure:
        raise SC100OperatorError("AcroForm field structure changed")

    changed_fields = {
        name
        for name in before
        if str(before[name]["value"]) != str(after[name]["value"])
    }
    if changed_fields != set(target):
        missing = sorted(set(target) - changed_fields)
        unexpected = sorted(changed_fields - set(target))
        raise SC100OperatorError(
            f"field mutation set mismatch: missing={missing}, unexpected={unexpected}"
        )
    for name, mutation in target.items():
        if str(after[name]["value"]).strip("/") != mutation.value.strip("/"):
            raise SC100OperatorError(f"reopened value mismatch for {name}")
        if mutation.field_type == "button":
            if after[name]["appearance_state"] != f"/{mutation.value}":
                raise SC100OperatorError(f"button appearance state mismatch for {name}")
            if after[name]["raw_value"] != f"/{mutation.value}":
                raise SC100OperatorError(f"button raw value mismatch for {name}")

    for group in _BUTTON_GROUPS:
        selected = [
            name
            for name in group
            if str(after[name]["value"]).strip("/") not in ("", "Off", "None")
        ]
        if len(selected) != 1 or selected[0] not in target:
            raise SC100OperatorError("button group does not have exactly one compiled selection")

    for name, row in after.items():
        if any(fragment in name for fragment in _FORBIDDEN_FIELD_FRAGMENTS):
            if str(row["value"]).strip("/") not in ("", "Off", "None"):
                raise SC100OperatorError(f"forbidden field was populated: {name}")

    output_text = _normalized(_extract_pdf_text(output_pdf))
    required_visible = (
        facts.plaintiff_name,
        facts.plaintiff_address.street,
        facts.plaintiff_address.city,
        facts.plaintiff_address.zip_code,
        facts.plaintiff_phone,
        facts.plaintiff_email,
        facts.defendant_name,
        facts.defendant_address.street,
        facts.defendant_address.city,
        facts.defendant_address.zip_code,
        facts.defendant_phone,
        str(facts.claim_amount),
        "security deposit",
        facts.incident_start,
        facts.incident_end,
        facts.contract_kind,
    )
    missing_visible = [value for value in required_visible if _normalized(value) not in output_text]
    if missing_visible:
        raise SC100OperatorError(
            "required visible values missing after pdftotext: "
            + _stable_hash(sorted(missing_visible))
        )

    receipt = {
        "operator_version": OPERATOR_VERSION,
        "operator_mode": "fixed_acroform_whitelist_no_xfa_no_raster",
        "xfa_dataset_claimed_updated": False,
        "input_sha256": input_sha,
        "output_sha256": _sha256(output_pdf),
        "plan_hash": plan.plan_hash,
        "fact_schema_hash": _stable_hash(asdict(facts)),
        "field_structure_hash": _stable_hash(before_structure),
        "field_count": len(before_structure),
        "mutation_count": len(target),
        "mutation_field_set_hash": _stable_hash(sorted(target)),
        "changed_field_set_hash": _stable_hash(sorted(changed_fields)),
        "page_count": 6,
        "source_unchanged": True,
        "field_structure_preserved": True,
        "exact_mutation_set_reconciled": True,
        "forbidden_fields_empty": True,
        "required_visible_value_count": len(required_visible),
        "required_visible_values_present": True,
        "raw_case_text_persisted": False,
        "receipt_hash": "",
    }
    receipt["receipt_hash"] = _stable_hash({**receipt, "receipt_hash": ""})
    return receipt


def execute(
    *,
    instruction: str,
    blank_pdf: Path,
    output_pdf: Path,
) -> dict[str, Any]:
    facts = parse_instruction(instruction)
    plan = compile_plan(facts)
    return apply_plan(blank_pdf=blank_pdf, output_pdf=output_pdf, plan=plan, facts=facts)


def _read_instruction(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruction", type=Path, required=True)
    parser.add_argument("--blank-pdf", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)

    receipt = execute(
        instruction=_read_instruction(args.instruction),
        blank_pdf=args.blank_pdf,
        output_pdf=args.output_pdf,
    )
    _write_json(args.receipt, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
