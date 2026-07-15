"""Role-anchored, closed SC-100 security-deposit operator.

The parser binds facts to explicit plaintiff/defendant evidence instead of
global occurrence order.  Unsupported or ambiguous cases return a redacted
rejection receipt before any output is created.  Accepted cases compile to the
same fixed AcroForm vocabulary as v1 and are written to a temporary PDF before
an atomic, no-clobber publish.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping
import unicodedata

from . import sc100_typed_operator_v1 as _writer


OPERATOR_VERSION = "sc100_role_anchored_acroform_v2"
REJECTION_RECEIPT_VERSION = "sc100-shadow-rejection-receipt-v1"
PUBLIC_BLANK_SHA256 = "ef3421b14ebf64dbf884566ff659b39776035ec5b6e6500be0af91e3cc15533c"
REJECTION_PRECEDENCE = (
    "public_entity",
    "attorney_fee_dispute",
    "multiple_plaintiffs",
    "payment_not_requested",
    "conflicting_claim_amount",
    "non_california_venue",
    "unsupported_claim_type",
    "missing_or_ambiguous_required_fact",
)


class SC100OperatorError(ValueError):
    """Infrastructure, blank-binding, or compiler failure."""


class _Reject(ValueError):
    def __init__(self, reason_code: str):
        if reason_code not in REJECTION_PRECEDENCE:
            raise AssertionError(f"unbound rejection reason: {reason_code}")
        self.reason_code = reason_code
        super().__init__(reason_code)


@dataclass(frozen=True)
class Address:
    street: str
    city: str
    state: str
    zip_code: str


@dataclass(frozen=True)
class _Located:
    value: str
    start: int
    end: int


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
    venue_basis: str
    venue_zip: str
    other_claim_count: int
    asked_to_pay: bool = True
    public_entity: bool = False
    attorney_fee_dispute: bool = False

    @property
    def more_than_twelve_claims(self) -> bool:
        return self.other_claim_count > 12


@dataclass(frozen=True)
class SC100Plan:
    mutations: tuple[_writer.FieldMutation, ...]

    @property
    def plan_hash(self) -> str:
        return _stable_hash(
            {
                "operator_version": OPERATOR_VERSION,
                "mutations": [asdict(mutation) for mutation in self.mutations],
            }
        )


_WORD = r"[^\W\d_]+(?:[’'\-][^\W\d_]+)*"
_NAME = rf"{_WORD}(?:\s+{_WORD}){{1,3}}"
_NAME_PATTERNS: Mapping[str, tuple[re.Pattern[str], ...]] = {
    "plaintiff": tuple(
        re.compile(pattern, re.I)
        for pattern in (
            rf"\bI am\s+(?!(?:suing|filing|asking|seeking)\b)"
            rf"(?P<value>{_NAME})(?=\s*[,.;]|\s+and\s+I\b)",
            rf"\bmy name is\s+(?P<value>{_NAME})(?=\s*[,.;]|\s+and\b)",
            rf"\b(?:the\s+)?plaintiff(?:'s name| name)?\s*(?:is|:)\s*"
            rf"(?!(?:not|no)\b)(?P<value>{_NAME})(?=\s*[,.;]|\s+and\b)",
            rf"\bplaintiff\s*[-–]\s*(?P<value>{_NAME})(?=\s*[,.;]|\s+address\b)",
        )
    ),
    "defendant": tuple(
        re.compile(pattern, re.I)
        for pattern in (
            rf"\b(?:I\s+)?(?:want to sue|am suing|sue)\s+(?P<value>{_NAME})(?=\s*[,.;]|\s+who\b)",
            rf"\b(?:the\s+)?(?:proposed\s+)?defendant(?:'s name| name)?\s*(?:is|:)\s*"
            rf"(?!(?:not|no)\b)(?P<value>{_NAME})(?=\s*[,.;]|\s+and\b)",
            rf"\bdefendant\s*[-–]\s*(?P<value>{_NAME})(?=\s*[,.;]|\s+address\b)",
        )
    ),
}

_ADDRESS = re.compile(
    r"(?<![\d-])(?P<street_no>\d{1,6})\s+"
    r"(?P<street>[^,\n]{1,80}?),\s*"
    r"(?P<city>[^,\n\d]{2,50}?),\s*"
    r"(?P<state>[A-Z]{2})\s+(?P<zip>\d{5})(?!\d)",
    re.I,
)
_PHONE = re.compile(
    r"(?<!\d)(?:\+?1[ .()\-]*)?(?P<a>\d{3})[ .()\-]*"
    r"(?P<b>\d{3})[ .\-]*(?P<c>\d{4})(?!\d)"
)
_EMAIL = re.compile(r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b", re.I)
_CURRENCY = re.compile(
    r"(?<![\w.])\$\s*(?P<number>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?)"
    r"(?![\w,]|\.\d)"
)
_ISO_DATE = r"20\d{2}-\d{2}-\d{2}"
_NATURAL_DATE = (
    r"(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2},\s*20\d{2}"
)
_DATE_TOKEN = rf"(?:{_ISO_DATE}|{_NATURAL_DATE})"
_EVENT_RANGE_PATTERNS = tuple(
    re.compile(pattern, re.I)
    for pattern in (
        rf"\b(?:dispute|issue|matter|event|tenancy|rental|claim)[^.;\n]{{0,45}}?"
        rf"(?:period|covers?|took place|occurred)?[^.;\n]{{0,25}}?\bfrom\s+"
        rf"(?P<start>{_DATE_TOKEN})\s+(?:to|through|thru|until|[-–])\s+(?P<end>{_DATE_TOKEN})",
        rf"\b(?:event|dispute|rental|tenancy)\s+start(?:ed)?\s*(?:is|:)?\s*"
        rf"(?P<start>{_DATE_TOKEN})[^.;\n]{{0,50}}?\b(?:event\s+)?end(?:ed)?\s*(?:is|:)?\s*(?P<end>{_DATE_TOKEN})",
        rf"\b(?:event|dispute|rental|tenancy|claim)(?:\s+period)?\s+"
        rf"start(?:ed)?\s*(?:is|:)?\s*(?P<start>{_DATE_TOKEN})\s*[,;/]\s*"
        rf"(?:through|end(?:ed)?)\s*(?:is|:)?\s*(?P<end>{_DATE_TOKEN})",
        rf"\b(?:dispute|issue|matter|event|tenancy|rental)(?:\s+period)?"
        rf"[^.;\n]{{0,45}}?\bbetween\s+(?P<start>{_DATE_TOKEN})\s+and\s+"
        rf"(?P<end>{_DATE_TOKEN})",
    )
)
_SIGNATURE_PATTERNS = tuple(
    re.compile(pattern, re.I)
    for pattern in (
        rf"\b(?:declaration(?: signing)?|signature|filing|file)\s+date\s*(?:is|:|of)?\s*(?P<date>{_DATE_TOKEN})",
        rf"\buse\s+(?P<date>{_DATE_TOKEN})\s+as\s+(?:the\s+)?(?:declaration(?: signing)?|signature|filing|file)\s+date",
        rf"\bfile\s+(?:it|the form)?\s*(?:with|on)\s+(?:date\s*:?)?\s*(?P<date>{_DATE_TOKEN})",
        rf"\bdated?\s+(?P<date>{_DATE_TOKEN})\s+for\s+(?:the\s+)?(?:declaration|signature|filing)",
    )
)

_PUBLIC_TERMS = (
    "public entity",
    "municipal housing authority",
    "government agency",
    "government entity",
    "housing authority",
    "school district",
)
_ATTORNEY_TERMS = (
    "attorney-client fee dispute",
    "attorney fee dispute",
    "attorney-fee dispute",
    "attorney fees",
    "attorney fee",
    "legal fees charged",
    "client fee dispute",
)
_MULTIPLE_PLAINTIFF_PATTERNS = tuple(
    re.compile(pattern, re.I)
    for pattern in (
        r"\b(?:two|2|multiple) plaintiffs\b",
        r"\bplaintiffs\s+are\b",
        r"\bjoint(?:ly)?\s+(?:plaintiffs|seek)\b",
        r"\bboth\s+[^.;]{0,80}\s+must\s+(?:appear|be listed)\s+as\s+plaintiffs\b",
        r"\b(?:another|second|additional)\s+plaintiff\b",
    )
)
_NOT_ASKED_PATTERNS = tuple(
    re.compile(pattern, re.I)
    for pattern in (
        r"\b(?:have|has|had)\s+not\s+asked\b",
        r"\bnever\s+(?:asked|requested|demanded)\b",
        r"\bnot\s+asked\b",
        r"\bno\s+(?:request|demand)\s+(?:was\s+)?made\b",
        r"\b(?:payment requested|asked to pay)\s*(?:is|:)?\s*no\b",
        r"\bpayment\s+(?:was|is)\s+not\s+requested\b",
        r"\bi\s+did\s+not\s+(?:request|demand|ask for)\s+payment\b",
    )
)
_ASKED_PATTERNS = tuple(
    re.compile(pattern, re.I)
    for pattern in (
        r"\b(?:asked|requested|demanded)\b[^.;]{0,90}\b(?:pay|payment|repay|repayment|return|refund|deposit)\b",
        r"\b(?:sent|mailed|emailed|texted)\b[^.;]{0,100}\b(?:request|demand|asking|deposit back|repayment)\b",
        r"\b(?:request|demand)\b[^.;]{0,100}\b(?:by|through|via)\b[^.;]{0,40}\b(?:text|email|letter|phone|telephone|message)\b",
        r"\b(?:payment requested|asked to pay)\s*(?:is|:)?\s*yes\b",
        r"\bpayment request\s*(?:is|:)?\s*(?:made|sent)\b",
    )
)


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value).strip()


def _normalize_space(value: str) -> str:
    return " ".join(_nfc(value).split())


def _unique(values: Iterable[str], reason: str = "missing_or_ambiguous_required_fact") -> str:
    unique = list(dict.fromkeys(_normalize_space(value) for value in values if value.strip()))
    if len(unique) != 1:
        raise _Reject(reason)
    return unique[0]


def _parse_date(value: str) -> str:
    value = _normalize_space(value)
    for pattern in ("%Y-%m-%d", "%B %d, %Y"):
        try:
            return datetime.strptime(value.title(), pattern).strftime("%Y-%m-%d")
        except ValueError:
            continue
    raise _Reject("missing_or_ambiguous_required_fact")


def _extract_name(text: str, role: str) -> _Located:
    matches: list[_Located] = []
    for pattern in _NAME_PATTERNS[role]:
        for match in pattern.finditer(text):
            matches.append(
                _Located(_normalize_space(match.group("value")), match.start("value"), match.end("value"))
            )
    values = list(dict.fromkeys(match.value for match in matches))
    if len(values) != 1:
        raise _Reject(
            "multiple_plaintiffs"
            if role == "plaintiff" and len(values) > 1
            else "missing_or_ambiguous_required_fact"
        )
    chosen = next(match for match in matches if match.value == values[0])
    if len(chosen.value.split()) < 2:
        raise _Reject("missing_or_ambiguous_required_fact")
    return chosen


def _sentence_bounds(text: str, position: int) -> tuple[int, int]:
    left = max(text.rfind(".", 0, position), text.rfind("\n", 0, position), text.rfind(";", 0, position))
    candidates = [index for index in (text.find(".", position), text.find("\n", position), text.find(";", position)) if index >= 0]
    right = min(candidates) if candidates else len(text)
    return left + 1, right


def _role_score(
    *, text: str, start: int, target: _Located, other: _Located, role: str, primary: bool = False
) -> int:
    left, right = _sentence_bounds(text, start)
    sentence = text[left:right].casefold()
    window = text[max(0, start - 220): min(len(text), start + 80)].casefold()
    target_name = target.value.casefold()
    other_name = other.value.casefold()
    score = 0
    if target_name in sentence:
        score += 80
    if other_name in sentence:
        score -= 70
    if target_name in window:
        score += 35
    if other_name in window:
        score -= 25
    if role == "plaintiff":
        if re.search(r"\b(?:plaintiff|my|I)\b", sentence, re.I):
            score += 30
        if re.search(r"\b(?:defendant|his|her)\b", sentence, re.I):
            score -= 25
    else:
        if re.search(r"\b(?:defendant|sue|suing|his|her|who)\b", sentence, re.I):
            score += 30
        if re.search(r"\b(?:plaintiff|my)\b", sentence, re.I):
            score -= 20
    target_distance = min(abs(start - target.start), abs(start - target.end))
    other_distance = min(abs(start - other.start), abs(start - other.end))
    if target_distance < other_distance:
        score += 20
    elif other_distance < target_distance:
        score -= 20
    local = text[max(0, start - 45):start].casefold()
    if primary and "primary" in local:
        score += 100
    if primary and any(term in local for term in ("backup", "alternate", "secondary")):
        score -= 100
    return score


def _choose_role_candidate(
    candidates: Iterable[_Located], *, text: str, target: _Located, other: _Located, role: str, primary: bool = False
) -> _Located:
    rows = [
        (
            _role_score(
                text=text,
                start=candidate.start,
                target=target,
                other=other,
                role=role,
                primary=primary,
            ),
            candidate,
        )
        for candidate in candidates
    ]
    if not rows:
        raise _Reject("missing_or_ambiguous_required_fact")
    rows.sort(key=lambda item: (-item[0], item[1].start))
    if rows[0][0] <= 0:
        raise _Reject("missing_or_ambiguous_required_fact")
    if len(rows) > 1 and rows[0][0] == rows[1][0] and rows[0][1].value != rows[1][1].value:
        raise _Reject("missing_or_ambiguous_required_fact")
    if (
        len(rows) > 1
        and rows[1][0] >= 60
        and rows[0][1].value != rows[1][1].value
    ):
        raise _Reject("missing_or_ambiguous_required_fact")
    return rows[0][1]


def _address_candidates(text: str) -> list[tuple[_Located, Address]]:
    rows: list[tuple[_Located, Address]] = []
    for match in _ADDRESS.finditer(text):
        street_tail = _normalize_space(match.group("street"))
        lowered = street_tail.casefold()
        if re.match(r"\d{3}\s+\d{4}\b", street_tail):
            continue
        if any(
            phrase in lowered
            for phrase in (
                "and lives at",
                "who lives at",
                "can be reached",
                "phone number",
                "telephone",
            )
        ):
            continue
        street = f"{match.group('street_no')} {street_tail}"
        address = Address(
            street=street,
            city=_normalize_space(match.group("city")),
            state=match.group("state").upper(),
            zip_code=match.group("zip"),
        )
        rows.append((_Located(street, match.start(), match.end()), address))
    return rows


def _role_addresses(text: str, plaintiff: _Located, defendant: _Located) -> tuple[Address, Address]:
    candidates = _address_candidates(text)
    if len(candidates) < 2:
        raise _Reject("missing_or_ambiguous_required_fact")
    p_location = _choose_role_candidate(
        (location for location, _ in candidates),
        text=text,
        target=plaintiff,
        other=defendant,
        role="plaintiff",
    )
    remaining = [(location, address) for location, address in candidates if location != p_location]
    d_location = _choose_role_candidate(
        (location for location, _ in remaining),
        text=text,
        target=defendant,
        other=plaintiff,
        role="defendant",
    )
    mapping = {location: address for location, address in candidates}
    if p_location == d_location:
        raise _Reject("missing_or_ambiguous_required_fact")
    return mapping[p_location], mapping[d_location]


def _role_phones(text: str, plaintiff: _Located, defendant: _Located) -> tuple[str, str]:
    candidates = [
        _Located("".join(match.group(key) for key in ("a", "b", "c")), match.start(), match.end())
        for match in _PHONE.finditer(text)
    ]
    if len(candidates) < 2:
        raise _Reject("missing_or_ambiguous_required_fact")
    p_phone = _choose_role_candidate(
        candidates,
        text=text,
        target=plaintiff,
        other=defendant,
        role="plaintiff",
        primary=True,
    )
    remaining = [candidate for candidate in candidates if candidate != p_phone]
    d_phone = _choose_role_candidate(
        remaining,
        text=text,
        target=defendant,
        other=plaintiff,
        role="defendant",
        primary=True,
    )
    return p_phone.value, d_phone.value


def _role_email(text: str, plaintiff: _Located, defendant: _Located) -> str:
    candidates = [
        _Located(match.group(0), match.start(), match.end())
        for match in _EMAIL.finditer(text)
    ]
    return _choose_role_candidate(
        candidates,
        text=text,
        target=plaintiff,
        other=defendant,
        role="plaintiff",
    ).value


def _claim_amounts(text: str) -> tuple[int, ...]:
    values: list[int] = []
    for match in _CURRENCY.finditer(text):
        left, right = _sentence_bounds(text, match.start())
        before = text[left:match.start()].casefold()
        after = text[match.end():right].casefold()
        before_clause = re.split(r"[,;]|\band\b", before)[-1]
        after_clause = re.split(r"[,;]|\band\b", after)[0]
        if not re.search(
            r"\b(?:security\s+deposit|deposit|claim(?:ed)?\s+amount|amount|"
            r"seek|seeks|seeking|owe|owes|owed)\b",
            before_clause,
        ) and not re.search(
            r"\b(?:security\s+deposit|deposit|claim(?:ed)?\s+amount|amount|"
            r"claimed|sought|owed)\b",
            after_clause,
        ):
            continue
        try:
            amount = Decimal(match.group("number").replace(",", ""))
        except InvalidOperation:
            continue
        if amount != amount.to_integral_value() or amount <= 0:
            raise _Reject("missing_or_ambiguous_required_fact")
        values.append(int(amount))
    return tuple(values)


def _has_unnegated_term(text: str, terms: Iterable[str]) -> bool:
    lower = text.casefold()
    for term in terms:
        for match in re.finditer(re.escape(term.casefold()), lower):
            prefix = lower[max(0, match.start() - 60):match.start()]
            suffix = lower[match.end():min(len(lower), match.end() + 30)]
            if re.search(
                r"\b(?:not|never|no|isn't|is\s+not|is\s+no)\b"
                r"(?:\W+\w+){0,4}\W*$",
                prefix,
            ) or re.match(r"\s*(?:\?|:|is)?\s*(?:no|false)\b", suffix):
                continue
            return True
    return False


def _pre_rejection(text: str) -> None:
    lower = text.casefold()
    if _has_unnegated_term(text, _PUBLIC_TERMS) or re.search(
        r"\b(?:city|county)\s+of\s+[A-Z][^,.;\n]{1,50}", text, re.I
    ):
        raise _Reject("public_entity")
    if _has_unnegated_term(text, _ATTORNEY_TERMS):
        raise _Reject("attorney_fee_dispute")
    if any(pattern.search(text) for pattern in _MULTIPLE_PLAINTIFF_PATTERNS):
        raise _Reject("multiple_plaintiffs")
    explicit_plaintiffs = {
        _normalize_space(match.group("value"))
        for pattern in _NAME_PATTERNS["plaintiff"]
        for match in pattern.finditer(text)
    }
    if len(explicit_plaintiffs) > 1:
        raise _Reject("multiple_plaintiffs")
    if any(pattern.search(text) for pattern in _NOT_ASKED_PATTERNS):
        raise _Reject("payment_not_requested")
    amounts = set(_claim_amounts(text))
    if len(amounts) > 1:
        raise _Reject("conflicting_claim_amount")
    states = {
        match.group(1).upper()
        for match in re.finditer(r"\b([A-Z]{2})\s+\d{5}\b", text)
    }
    if (
        any(state != "CA" for state in states)
        or
        "no event or party is connected to california" in lower
        or "all in portland, oregon" in lower
        or ("california sc-100" in lower and not re.search(r"\bCA\s+\d{5}\b", text))
    ):
        raise _Reject("non_california_venue")
    if (
        "not a rental security-deposit dispute" in lower
        or "not a security-deposit dispute" in lower
        or re.search(
            r"\b(?:property damage|unpaid wages?|personal injury|consumer debt)\s+claim\b",
            text,
            re.I,
        )
    ):
        raise _Reject("unsupported_claim_type")


def _parse_event_dates(text: str) -> tuple[str, str]:
    pairs: list[tuple[str, str]] = []
    for pattern in _EVENT_RANGE_PATTERNS:
        for match in pattern.finditer(text):
            pairs.append((_parse_date(match.group("start")), _parse_date(match.group("end"))))
    starts = [
        _parse_date(match.group("date"))
        for match in re.finditer(
            rf"\b(?:event|dispute|rental|tenancy)(?:\s+period)?\s+"
            rf"start(?:ed|\s+date)?\s*(?:is|:|on)?\s*(?P<date>{_DATE_TOKEN})",
            text,
            re.I,
        )
    ]
    ends = [
        _parse_date(match.group("date"))
        for match in re.finditer(
            rf"\b(?:event|dispute|rental|tenancy)(?:\s+period)?\s+"
            rf"end(?:ed|\s+date)?\s*(?:is|:|on)?\s*(?P<date>{_DATE_TOKEN})",
            text,
            re.I,
        )
    ]
    unique_starts = list(dict.fromkeys(starts))
    unique_ends = list(dict.fromkeys(ends))
    if unique_starts or unique_ends:
        if len(unique_starts) != 1 or len(unique_ends) != 1:
            raise _Reject("missing_or_ambiguous_required_fact")
        pairs.append((unique_starts[0], unique_ends[0]))
    unique = list(dict.fromkeys(pairs))
    if len(unique) != 1 or unique[0][0] > unique[0][1]:
        raise _Reject("missing_or_ambiguous_required_fact")
    return unique[0]


def _parse_signature_date(text: str) -> str:
    values: list[str] = []
    for pattern in _SIGNATURE_PATTERNS:
        values.extend(_parse_date(match.group("date")) for match in pattern.finditer(text))
    return _unique(values)


def _parse_venue(text: str, defendant_address: Address) -> tuple[str, str]:
    contract_patterns = (
        re.compile(
            r"\b(?:use|select|choose|file|filing|venue|courthouse)\b"
            r"[^.;\n]{0,180}\b(?:contract|agreement)\b[^.;\n]{0,140}"
            r"\b(?:made|performed|broken|breached)\b",
            re.I,
        ),
        re.compile(
            r"\b(?:contract|agreement)\b[^.;\n]{0,140}"
            r"\b(?:made|performed|broken|breached)\b[^.;\n]{0,180}"
            r"\b(?:venue|courthouse|file|filing)\b",
            re.I,
        ),
        re.compile(
            r"\b(?:contract|agreement)\b[^.;\n]{0,160}"
            r"\b(?:made|performed|broken|breached)\b[^.;\n]{0,140}"
            r"\b[A-Z]{2}\s+\d{5}\b\s*[.;]\s*"
            r"(?:use|select|choose|file|filing)[^.;\n]{0,60}"
            r"\b(?:there|that\s+(?:place|location|venue)|this\s+venue)\b",
            re.I,
        ),
    )
    contract_basis = any(pattern.search(text) for pattern in contract_patterns)
    residence_basis = bool(
        re.search(
            r"\b(?:file|filing|venue|courthouse|location)\b[^.;\n]{0,100}"
            r"\bwhere\s+the\s+defendant\s+lives\b",
            text,
            re.I,
        )
        or re.search(
            r"\b(?:defendant-residence venue|defendant residence is the venue)\b",
            text,
            re.I,
        )
        or re.search(
            r"\bvenue\s+(?:basis|selection)\s*(?:is|:)?\s*"
            r"defendant(?:'s)?\s+residence\b",
            text,
            re.I,
        )
    )
    if contract_basis and residence_basis:
        raise _Reject("missing_or_ambiguous_required_fact")
    if contract_basis:
        state_zip_patterns = (
            re.compile(
                r"\b(?:venue|courthouse|filing|file|contract|agreement)\b"
                r"[^.;\n]{0,220}?(?:in|at)\s+(?:[^\d,.;\n]{1,60},\s*)?"
                r"(?P<state>[A-Z]{2})\s+(?P<zip>\d{5})\b",
                re.I,
            ),
            re.compile(
                r"\b(?:contract|agreement)\b[^.;\n]{0,160}\b"
                r"(?:made|performed|broken|breached)\b[^.;\n]{0,120}"
                r"(?P<state>[A-Z]{2})\s+(?P<zip>\d{5})\b",
                re.I,
            ),
        )
        state_zip = [
            (match.group("state").upper(), match.group("zip"))
            for pattern in state_zip_patterns
            for match in pattern.finditer(text)
        ]
        if any(state != "CA" for state, _ in state_zip):
            raise _Reject("non_california_venue")
        values = [zip_code for state, zip_code in state_zip if state == "CA"]
        values.extend(
            match.group(1)
            for match in re.finditer(
                r"\b(?:venue|form)\s+zip(?: code)?\s*(?:is|:)?\s*(\d{5})\b",
                text,
                re.I,
            )
        )
        return "contract_made_performed_and_breached", _unique(values)
    if not residence_basis:
        raise _Reject("missing_or_ambiguous_required_fact")
    if defendant_address.state != "CA":
        raise _Reject("non_california_venue")
    return "defendant_residence", defendant_address.zip_code


def _parse_claim_count(text: str) -> int:
    lower = text.casefold()
    first = bool(
        re.search(
            r"\b(?:first time filing|first small claims case|my first time|"
            r"no other small claims|not filed any other small claims|"
            r"zero other small claims)\b",
            lower,
        )
    )
    over = bool(re.search(r"\bmore than\s+12\b[^.;]{0,60}\b(?:small )?claims\b", lower))
    exact = [
        int(match.group(1))
        for match in re.finditer(
            r"\b(?:i\s+(?:have\s+)?filed|plaintiff\s+has\s+filed|filed)\s+"
            r"(\d{1,3})\s+other\s+small\s+claims\b",
            lower,
        )
    ]
    exact.extend(
        int(match.group(1))
        for match in re.finditer(
            r"\b(?:other small claims(?: filed)?|prior small[- ]claims count)"
            r"(?:\s+in\s+(?:the\s+)?last\s+12\s+months)?\s*(?:is|:)?\s*(\d{1,3})\b",
            lower,
        )
    )
    values: list[int] = []
    if first:
        values.append(0)
    if over:
        values.append(13)
    values.extend(exact)
    if not values:
        raise _Reject("missing_or_ambiguous_required_fact")
    if over and exact and not all(value > 12 for value in exact):
        raise _Reject("missing_or_ambiguous_required_fact")
    if first and any(value != 0 for value in values):
        raise _Reject("missing_or_ambiguous_required_fact")
    if over:
        return max(values)
    return int(_unique(str(value) for value in values))


def parse_instruction(instruction: str) -> SC100CaseFacts:
    """Parse the role-anchored closed grammar or return a typed rejection."""

    if not isinstance(instruction, str) or not instruction.strip():
        raise _Reject("missing_or_ambiguous_required_fact")
    text = unicodedata.normalize("NFC", instruction).replace("\r\n", "\n").replace("\r", "\n")
    _pre_rejection(text)
    if not re.search(r"\bsecurity\s+deposit\b", text, re.I):
        raise _Reject("missing_or_ambiguous_required_fact")

    plaintiff = _extract_name(text, "plaintiff")
    defendant = _extract_name(text, "defendant")
    if plaintiff.value.casefold() == defendant.value.casefold():
        raise _Reject("missing_or_ambiguous_required_fact")
    plaintiff_address, defendant_address = _role_addresses(text, plaintiff, defendant)
    if plaintiff_address.state != "CA" or defendant_address.state != "CA":
        raise _Reject("non_california_venue")
    plaintiff_phone, defendant_phone = _role_phones(text, plaintiff, defendant)

    plaintiff_email = _role_email(text, plaintiff, defendant)
    amounts = set(_claim_amounts(text))
    if len(amounts) != 1:
        raise _Reject("missing_or_ambiguous_required_fact")
    amount = next(iter(amounts))
    if amount > 12_500:
        raise _Reject("missing_or_ambiguous_required_fact")

    contracts = list(
        dict.fromkeys(
            match.group(1).casefold()
            for match in re.finditer(
                r"\b(?:signed\s+)?(room rental agreement|roommate sublease contract)\b",
                text,
                re.I,
            )
        )
    )
    contract_kind = _unique(contracts)
    if re.search(
        rf"\b(?:not|never)\s+(?:been\s+)?signed\b[^.;\n]{{0,55}}"
        rf"\b{re.escape(contract_kind)}\b|"
        rf"\b{re.escape(contract_kind)}\b[^.;\n]{{0,55}}\b"
        rf"(?:was\s+not|is\s+not|never)\s+signed\b",
        text,
        re.I,
    ):
        raise _Reject("missing_or_ambiguous_required_fact")
    if not re.search(rf"\bsigned\b[^.;\n]{{0,45}}\b{re.escape(contract_kind)}\b|\b{re.escape(contract_kind)}\b[^.;\n]{{0,45}}\bsigned\b", text, re.I):
        raise _Reject("missing_or_ambiguous_required_fact")

    incident_start, incident_end = _parse_event_dates(text)
    declaration_date = _parse_signature_date(text)
    if declaration_date < incident_end:
        raise _Reject("missing_or_ambiguous_required_fact")
    if not any(pattern.search(text) for pattern in _ASKED_PATTERNS):
        raise _Reject("missing_or_ambiguous_required_fact")
    venue_basis, venue_zip = _parse_venue(text, defendant_address)
    other_claim_count = _parse_claim_count(text)

    return SC100CaseFacts(
        plaintiff_name=plaintiff.value,
        plaintiff_address=plaintiff_address,
        plaintiff_phone=plaintiff_phone,
        plaintiff_email=plaintiff_email,
        defendant_name=defendant.value,
        defendant_address=defendant_address,
        defendant_phone=defendant_phone,
        claim_amount=amount,
        contract_kind=contract_kind,
        incident_start=incident_start,
        incident_end=incident_end,
        declaration_date=declaration_date,
        venue_basis=venue_basis,
        venue_zip=venue_zip,
        other_claim_count=other_claim_count,
    )


def compile_plan(facts: SC100CaseFacts) -> SC100Plan:
    if facts.public_entity or facts.attorney_fee_dispute or not facts.asked_to_pay:
        raise SC100OperatorError("unsupported facts reached compiler")
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
            f"Defendant did not return the ${facts.claim_amount} security deposit after move-out."
        ),
        "caption_page3": facts.plaintiff_name,
        "incident_start": facts.incident_start,
        "incident_end": facts.incident_end,
        "calculation": (
            f"${facts.claim_amount} security deposit documented in the signed {facts.contract_kind}."
        ),
        "venue_zip": facts.venue_zip,
        "caption_page4": facts.plaintiff_name,
        "declaration_date": facts.declaration_date,
        "signature_name": facts.plaintiff_name,
    }
    button_keys = (
        "asked_to_pay_yes",
        "venue_defendant_residence",
        "attorney_fee_no",
        "public_entity_no",
        "more_than_twelve_yes" if facts.more_than_twelve_claims else "more_than_twelve_no",
        "over_2500_yes" if facts.claim_amount > 2_500 else "over_2500_no",
    )
    mutations = [
        _writer.FieldMutation(_writer._TEXT_FIELDS[key], "text", value)
        for key, value in text_values.items()
    ]
    mutations.extend(
        _writer.FieldMutation(
            _writer._BUTTON_FIELDS[key][0], "button", _writer._BUTTON_FIELDS[key][1]
        )
        for key in button_keys
    )
    names = [mutation.field_name for mutation in mutations]
    if len(mutations) != 30 or len(names) != len(set(names)):
        raise AssertionError("compiler must emit exactly 30 unique mutations")
    return SC100Plan(tuple(mutations))


def _as_v1_facts(facts: SC100CaseFacts) -> _writer.SC100CaseFacts:
    return _writer.SC100CaseFacts(
        plaintiff_name=facts.plaintiff_name,
        plaintiff_address=_writer.Address(**asdict(facts.plaintiff_address)),
        plaintiff_phone=facts.plaintiff_phone,
        plaintiff_email=facts.plaintiff_email,
        defendant_name=facts.defendant_name,
        defendant_address=_writer.Address(**asdict(facts.defendant_address)),
        defendant_phone=facts.defendant_phone,
        claim_amount=facts.claim_amount,
        contract_kind=facts.contract_kind,
        incident_start=facts.incident_start,
        incident_end=facts.incident_end,
        declaration_date=facts.declaration_date,
        asked_to_pay=True,
        venue_is_defendant_residence=True,
        more_than_twelve_claims=facts.more_than_twelve_claims,
    )


def _receipt_hash(payload: Mapping[str, Any]) -> str:
    return _stable_hash(dict(payload))


def _instruction_sha256(instruction: object) -> str:
    if isinstance(instruction, str):
        return hashlib.sha256(instruction.encode("utf-8")).hexdigest()
    return _stable_hash({"invalid_instruction_type": type(instruction).__name__})


def _rejection_receipt(
    *, instruction: object, reason_code: str, input_sha256: str
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "action": "reject",
        "operator_version": OPERATOR_VERSION,
        "rejection_receipt_version": REJECTION_RECEIPT_VERSION,
        "reason_code": reason_code,
        "input_sha256": input_sha256,
        "instruction_sha256": _instruction_sha256(instruction),
        "output_pdf": None,
        "source_unchanged": True,
        "partial_output_created": False,
        "raw_case_text_persisted": False,
    }
    receipt["receipt_hash"] = _receipt_hash(receipt)
    return receipt


def execute(instruction: str, blank_pdf: Path, output_pdf: Path) -> dict[str, Any]:
    """Execute one fill or return an exact, zero-write rejection receipt."""

    blank = Path(blank_pdf).resolve()
    output = Path(output_pdf).resolve()
    if blank == output:
        raise SC100OperatorError("blank and output paths must differ")
    if not blank.is_file():
        raise FileNotFoundError(blank)
    if output.exists():
        raise FileExistsError(output)
    input_sha = _sha256(blank)
    if input_sha != PUBLIC_BLANK_SHA256:
        raise SC100OperatorError("public blank binding mismatch")
    try:
        facts = parse_instruction(instruction)
    except _Reject as rejected:
        if output.exists() or _sha256(blank) != input_sha:
            raise SC100OperatorError("zero-write rejection contract violated")
        return _rejection_receipt(
            instruction=instruction, reason_code=rejected.reason_code, input_sha256=input_sha
        )

    plan = compile_plan(facts)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp.pdf", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        inner_plan = _writer.SC100Plan(
            operator_version=OPERATOR_VERSION, mutations=plan.mutations
        )
        inner = _writer.apply_plan(
            blank_pdf=blank,
            output_pdf=temporary,
            plan=inner_plan,
            facts=_as_v1_facts(facts),
        )
        if inner_plan.plan_hash != plan.plan_hash:
            raise SC100OperatorError("writer/compiler plan hash mismatch")
        inner_hash = inner.get("receipt_hash")
        if not isinstance(inner_hash, str) or _stable_hash(
            {**inner, "receipt_hash": ""}
        ) != inner_hash:
            raise SC100OperatorError("inner writer receipt hash mismatch")
        if _sha256(blank) != input_sha:
            raise SC100OperatorError("source changed during execution")
        output_sha = _sha256(temporary)
        if (
            inner.get("input_sha256") != input_sha
            or inner.get("output_sha256") != output_sha
            or inner.get("plan_hash") != plan.plan_hash
            or inner.get("mutation_count") != 30
            or any(
                inner.get(key) is not True
                for key in (
                    "source_unchanged",
                    "field_structure_preserved",
                    "exact_mutation_set_reconciled",
                    "forbidden_fields_empty",
                    "required_visible_values_present",
                )
            )
        ):
            raise SC100OperatorError("inner writer reconciliation mismatch")
        receipt: dict[str, Any] = {
            "action": "fill",
            "operator_version": OPERATOR_VERSION,
            "writer_version": _writer.OPERATOR_VERSION,
            "input_sha256": input_sha,
            "output_sha256": output_sha,
            "instruction_sha256": _instruction_sha256(instruction),
            "plan_hash": plan.plan_hash,
            "fact_schema_hash": _stable_hash(asdict(facts)),
            "mutation_count": 30,
            "text_mutation_count": 24,
            "button_mutation_count": 6,
            "inner_writer_receipt_hash": inner_hash,
            "source_unchanged": True,
            "atomic_publish": True,
            "publish_method": "same_filesystem_hard_link_no_clobber",
            "partial_output_created": False,
            "raw_case_text_persisted": False,
        }
        os.link(temporary, output)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        receipt["temporary_cleanup_verified"] = not temporary.exists()
        receipt["receipt_hash"] = _receipt_hash(receipt)
        return receipt
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


__all__ = [
    "Address",
    "OPERATOR_VERSION",
    "PUBLIC_BLANK_SHA256",
    "REJECTION_PRECEDENCE",
    "SC100CaseFacts",
    "SC100OperatorError",
    "SC100Plan",
    "compile_plan",
    "execute",
    "parse_instruction",
]
