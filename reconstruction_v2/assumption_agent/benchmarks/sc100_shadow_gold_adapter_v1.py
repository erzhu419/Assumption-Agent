"""Strict latent-gold adapter for the frozen synthetic SC-100 shadow corpus.

This module owns only the corpus schema translation.  It deliberately does not
import the candidate parser, form writer, benchmark tests, or the PDF oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence
import unicodedata


ADAPTER_SCHEMA = "sc100-shadow-gold-adapter-v1"

REQUIRED_POSITIVE_IDS = frozenset(f"S{index:02d}" for index in range(1, 13))
COVERAGE_PROBE_IDS = frozenset(f"C{index:02d}" for index in range(1, 7))
TRUE_NEGATIVE_IDS = frozenset(f"N{index:02d}" for index in range(1, 7))
ALL_CASE_IDS = REQUIRED_POSITIVE_IDS | COVERAGE_PROBE_IDS | TRUE_NEGATIVE_IDS

_TEMPLATE_BY_ID = {
    "S01": "T1", "S10": "T1",
    "S02": "T2", "S11": "T2",
    "S03": "T3", "S12": "T3",
    "S04": "T4", "S07": "T4",
    "S05": "T5", "S08": "T5",
    "S06": "T6", "S09": "T6",
}
_PROBE_DIMENSION_BY_ID = {
    "C01": "non_text_payment_request_email_and_letter",
    "C02": "signature_date_after_event_end",
    "C03": "extra_unambiguous_demand_date",
    "C04": "contract_venue_not_defendant_residence",
    "C05": "explicit_primary_and_backup_phone",
    "C06": "hyphen_apostrophe_unicode_names_and_interleaved_order",
}
_REJECTION_REASON_BY_ID = {
    "N01": "public_entity",
    "N02": "attorney_fee_dispute",
    "N03": "multiple_plaintiffs",
    "N04": "payment_not_requested",
    "N05": "conflicting_claim_amount",
    "N06": "non_california_venue",
}
_CONTRACT_TEXT = {
    "room_rental_agreement": "room rental agreement",
    "roommate_sublease_contract": "roommate sublease contract",
}
_BLANK_GROUPS = (
    "court_use_header",
    "second_plaintiff",
    "second_defendant",
    "optional_unstated_fields",
)

_BASE_FILL_KEYS = frozenset(
    {
        "case_id",
        "case_class",
        "expected_action",
        "plaintiff",
        "defendant",
        "claim",
        "payment_request",
        "venue",
        "attorney_fee_dispute",
        "public_entity",
        "other_small_claims_last_12_months",
        "more_than_12_other_small_claims",
        "claim_more_than_2500",
        "signature",
        "expected_form_semantics",
        "required_blank_groups",
    }
)
_ADDRESS_KEYS = frozenset({"street", "city", "state", "zip"})
_PHONE_RE = re.compile(r"\d{10}\Z")
_ZIP_RE = re.compile(r"\d{5}\Z")
_EMAIL_RE = re.compile(r"[^\s@]+@[^\s@]+\.[^\s@]+\Z")


class ShadowGoldValidationError(ValueError):
    """Raised when a latent record cannot be translated without inference."""

    def __init__(self, codes: Iterable[str]):
        self.codes = tuple(sorted(set(codes)))
        super().__init__("invalid synthetic SC-100 shadow gold")


@dataclass(frozen=True)
class AdaptedShadowRecord:
    case_id: str
    case_class: str
    expected_action: str
    oracle_gold: Mapping[str, Any] | None
    rejection_reason: str | None


def _nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def _mapping(value: Any, code: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"invalid:{code}")
        return {}
    return value


def _exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], code: str, errors: list[str]
) -> None:
    actual = frozenset(value)
    for key in sorted(expected - actual):
        errors.append(f"missing:{code}.{key}")
    for key in sorted(actual - expected):
        errors.append(f"extra:{code}.{key}")


def _string(value: Any, code: str, errors: list[str]) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        errors.append(f"invalid:{code}")
        return ""
    normalized = _nfc(value)
    if any(ord(char) < 32 for char in normalized):
        errors.append(f"invalid:{code}")
    return normalized


def _boolean(value: Any, code: str, errors: list[str]) -> bool:
    if type(value) is not bool:
        errors.append(f"invalid:{code}")
        return False
    return value


def _integer(value: Any, code: str, errors: list[str], *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        errors.append(f"invalid:{code}")
        return minimum
    return value


def _iso_date(value: Any, code: str, errors: list[str]) -> str:
    text = _string(value, code, errors)
    try:
        parsed = date.fromisoformat(text)
    except ValueError:
        errors.append(f"invalid:{code}")
        return "0001-01-01"
    if parsed.isoformat() != text:
        errors.append(f"invalid:{code}")
    return parsed.isoformat()


def _address(value: Any, code: str, errors: list[str]) -> dict[str, str]:
    address = _mapping(value, code, errors)
    _exact_keys(address, _ADDRESS_KEYS, code, errors)
    result = {
        key: _string(address.get(key), f"{code}.{key}", errors)
        for key in ("street", "city", "state", "zip")
    }
    if result["state"] != "CA":
        errors.append(f"inconsistent:{code}.state")
    if not _ZIP_RE.fullmatch(result["zip"]):
        errors.append(f"invalid:{code}.zip")
    return result


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _adapt_rejection(record: Mapping[str, Any], errors: list[str]) -> AdaptedShadowRecord:
    _exact_keys(
        record,
        frozenset({"case_id", "case_class", "expected_action", "reason_code"}),
        "record",
        errors,
    )
    case_id = _string(record.get("case_id"), "case_id", errors)
    if case_id not in TRUE_NEGATIVE_IDS:
        errors.append("inconsistent:case_id")
    if record.get("case_class") != "true_negative":
        errors.append("inconsistent:case_class")
    if record.get("expected_action") != "reject":
        errors.append("inconsistent:expected_action")
    reason = _string(record.get("reason_code"), "reason_code", errors)
    if reason != _REJECTION_REASON_BY_ID.get(case_id):
        errors.append("inconsistent:reason_code")
    if errors:
        raise ShadowGoldValidationError(errors)
    return AdaptedShadowRecord(case_id, "true_negative", "reject", None, reason)


def adapt_shadow_record(record: Mapping[str, Any]) -> AdaptedShadowRecord:
    """Validate one corpus record and translate fill rows to oracle semantics."""

    errors: list[str] = []
    if not isinstance(record, Mapping):
        raise ShadowGoldValidationError(("invalid:record",))
    if record.get("expected_action") == "reject" or record.get("case_class") == "true_negative":
        return _adapt_rejection(record, errors)

    case_id = _string(record.get("case_id"), "case_id", errors)
    case_class = _string(record.get("case_class"), "case_class", errors)
    expected_action = _string(record.get("expected_action"), "expected_action", errors)
    if case_id in REQUIRED_POSITIVE_IDS:
        expected_keys = _BASE_FILL_KEYS | {"template_id"}
        if case_class != "required_positive" or expected_action != "fill":
            errors.append("inconsistent:cohort_action")
        if record.get("template_id") != _TEMPLATE_BY_ID.get(case_id):
            errors.append("inconsistent:template_id")
    elif case_id in COVERAGE_PROBE_IDS:
        expected_keys = _BASE_FILL_KEYS | {"coverage_dimension"}
        if case_class != "coverage_probe" or expected_action != "coverage_probe_fill":
            errors.append("inconsistent:cohort_action")
        if record.get("coverage_dimension") != _PROBE_DIMENSION_BY_ID.get(case_id):
            errors.append("inconsistent:coverage_dimension")
    else:
        expected_keys = _BASE_FILL_KEYS
        errors.append("inconsistent:case_id")
    _exact_keys(record, frozenset(expected_keys), "record", errors)

    plaintiff = _mapping(record.get("plaintiff"), "plaintiff", errors)
    plaintiff_keys = {"name", "address", "phone_digits", "email"}
    if case_id == "C05":
        plaintiff_keys |= {"alternate_phone_digits", "form_phone_source"}
    _exact_keys(plaintiff, frozenset(plaintiff_keys), "plaintiff", errors)
    plaintiff_name = _string(plaintiff.get("name"), "plaintiff.name", errors)
    plaintiff_address = _address(plaintiff.get("address"), "plaintiff.address", errors)
    plaintiff_phone = _string(plaintiff.get("phone_digits"), "plaintiff.phone_digits", errors)
    if not _PHONE_RE.fullmatch(plaintiff_phone):
        errors.append("invalid:plaintiff.phone_digits")
    plaintiff_email = _string(plaintiff.get("email"), "plaintiff.email", errors)
    if not _EMAIL_RE.fullmatch(plaintiff_email):
        errors.append("invalid:plaintiff.email")
    if case_id == "C05":
        alternate = _string(
            plaintiff.get("alternate_phone_digits"), "plaintiff.alternate_phone_digits", errors
        )
        if not _PHONE_RE.fullmatch(alternate) or alternate == plaintiff_phone:
            errors.append("invalid:plaintiff.alternate_phone_digits")
        if plaintiff.get("form_phone_source") != "primary":
            errors.append("inconsistent:plaintiff.form_phone_source")

    defendant = _mapping(record.get("defendant"), "defendant", errors)
    _exact_keys(
        defendant,
        frozenset({"name", "address", "phone_digits", "entity_type"}),
        "defendant",
        errors,
    )
    defendant_name = _string(defendant.get("name"), "defendant.name", errors)
    defendant_address = _address(defendant.get("address"), "defendant.address", errors)
    defendant_phone = _string(defendant.get("phone_digits"), "defendant.phone_digits", errors)
    if not _PHONE_RE.fullmatch(defendant_phone):
        errors.append("invalid:defendant.phone_digits")
    if defendant.get("entity_type") != "private_individual":
        errors.append("inconsistent:defendant.entity_type")

    claim = _mapping(record.get("claim"), "claim", errors)
    _exact_keys(
        claim,
        frozenset(
            {
                "type", "amount_cents", "components", "additional_amounts_cents",
                "contract_type", "contract_signed", "event_start", "event_end",
            }
        ),
        "claim",
        errors,
    )
    if claim.get("type") != "security_deposit_return":
        errors.append("inconsistent:claim.type")
    amount_cents = _integer(claim.get("amount_cents"), "claim.amount_cents", errors, minimum=1)
    if amount_cents % 100:
        errors.append("invalid:claim.amount_cents_precision")
    additional_cents = _integer(
        claim.get("additional_amounts_cents"), "claim.additional_amounts_cents", errors
    )
    if additional_cents != 0:
        errors.append("inconsistent:claim.additional_amounts_cents")
    components = claim.get("components")
    if not isinstance(components, list) or len(components) != 1:
        errors.append("invalid:claim.components")
        components = []
    component_total = 0
    for index, raw_component in enumerate(components):
        component = _mapping(raw_component, f"claim.components[{index}]", errors)
        _exact_keys(
            component,
            frozenset({"type", "amount_cents"}),
            f"claim.components[{index}]",
            errors,
        )
        if component.get("type") != "security_deposit":
            errors.append(f"inconsistent:claim.components[{index}].type")
        component_total += _integer(
            component.get("amount_cents"), f"claim.components[{index}].amount_cents", errors
        )
    if component_total + additional_cents != amount_cents:
        errors.append("inconsistent:claim.component_total")
    contract_type = _string(claim.get("contract_type"), "claim.contract_type", errors)
    if contract_type not in _CONTRACT_TEXT:
        errors.append("invalid:claim.contract_type")
    if _boolean(claim.get("contract_signed"), "claim.contract_signed", errors) is not True:
        errors.append("inconsistent:claim.contract_signed")
    event_start = _iso_date(claim.get("event_start"), "claim.event_start", errors)
    event_end = _iso_date(claim.get("event_end"), "claim.event_end", errors)
    if event_start > event_end:
        errors.append("inconsistent:claim.event_dates")

    payment = _mapping(record.get("payment_request"), "payment_request", errors)
    payment_keys = {"asked", "channels", "demand_dates"}
    if case_id == "C01":
        payment_keys.add("text_message_used")
    _exact_keys(payment, frozenset(payment_keys), "payment_request", errors)
    asked = _boolean(payment.get("asked"), "payment_request.asked", errors)
    if not asked:
        errors.append("inconsistent:payment_request.asked")
    channels = payment.get("channels")
    allowed_channels = {"text_message", "email", "mailed_letter"}
    if (
        not isinstance(channels, list)
        or not channels
        or len(channels) != len(set(channels))
        or any(channel not in allowed_channels for channel in channels)
    ):
        errors.append("invalid:payment_request.channels")
        channels = []
    demand_dates = payment.get("demand_dates")
    if not isinstance(demand_dates, list):
        errors.append("invalid:payment_request.demand_dates")
        demand_dates = []
    parsed_demands = [
        _iso_date(value, f"payment_request.demand_dates[{index}]", errors)
        for index, value in enumerate(demand_dates)
    ]
    if len(parsed_demands) != len(set(parsed_demands)):
        errors.append("invalid:payment_request.demand_dates_duplicates")
    if case_id == "C01":
        if payment.get("text_message_used") is not False or "text_message" in channels:
            errors.append("inconsistent:payment_request.text_message_used")

    venue = _mapping(record.get("venue"), "venue", errors)
    basis = _string(venue.get("basis_code"), "venue.basis_code", errors)
    if basis == "defendant_residence":
        venue_keys = frozenset({"basis_code", "defendant_resides_at_listed_address", "form_zip"})
        venue_semantic = "defendant_residence_only"
    elif basis == "contract_made_performed_and_breached":
        venue_keys = frozenset(
            {
                "basis_code", "requested_county", "requested_city",
                "defendant_resides_at_listed_address",
                "defendant_residence_is_venue_basis", "form_zip",
            }
        )
        venue_semantic = "contract_made_performed_or_breached_only"
    else:
        venue_keys = frozenset({"basis_code", "defendant_resides_at_listed_address", "form_zip"})
        venue_semantic = ""
        errors.append("invalid:venue.basis_code")
    _exact_keys(venue, venue_keys, "venue", errors)
    if _boolean(
        venue.get("defendant_resides_at_listed_address"),
        "venue.defendant_resides_at_listed_address",
        errors,
    ) is not True:
        errors.append("inconsistent:venue.defendant_resides_at_listed_address")
    venue_zip = _string(venue.get("form_zip"), "venue.form_zip", errors)
    if not _ZIP_RE.fullmatch(venue_zip):
        errors.append("invalid:venue.form_zip")
    if basis == "defendant_residence" and venue_zip != defendant_address["zip"]:
        errors.append("inconsistent:venue.form_zip")
    if basis == "contract_made_performed_and_breached":
        _string(venue.get("requested_county"), "venue.requested_county", errors)
        _string(venue.get("requested_city"), "venue.requested_city", errors)
        if venue.get("defendant_residence_is_venue_basis") is not False:
            errors.append("inconsistent:venue.defendant_residence_is_venue_basis")

    attorney = _boolean(record.get("attorney_fee_dispute"), "attorney_fee_dispute", errors)
    public = _boolean(record.get("public_entity"), "public_entity", errors)
    if attorney or public:
        errors.append("inconsistent:fill_scope")
    claim_count = _integer(
        record.get("other_small_claims_last_12_months"),
        "other_small_claims_last_12_months",
        errors,
    )
    more_than_12 = _boolean(
        record.get("more_than_12_other_small_claims"),
        "more_than_12_other_small_claims",
        errors,
    )
    over_2500 = _boolean(record.get("claim_more_than_2500"), "claim_more_than_2500", errors)
    if more_than_12 != (claim_count > 12):
        errors.append("inconsistent:more_than_12_other_small_claims")
    if over_2500 != (amount_cents > 250_000):
        errors.append("inconsistent:claim_more_than_2500")

    signature = _mapping(record.get("signature"), "signature", errors)
    _exact_keys(signature, frozenset({"name", "date"}), "signature", errors)
    signature_name = _string(signature.get("name"), "signature.name", errors)
    signature_date = _iso_date(signature.get("date"), "signature.date", errors)
    if signature_name != plaintiff_name:
        errors.append("inconsistent:signature.name")
    if signature_date < event_end:
        errors.append("inconsistent:signature.date")

    expected = _mapping(record.get("expected_form_semantics"), "expected_form_semantics", errors)
    _exact_keys(
        expected,
        frozenset(
            {
                "asked_to_pay", "venue_selection", "attorney_fee_dispute",
                "public_entity", "more_than_12", "more_than_2500",
            }
        ),
        "expected_form_semantics",
        errors,
    )
    derived_expected = {
        "asked_to_pay": _yes_no(asked),
        "venue_selection": venue_semantic,
        "attorney_fee_dispute": _yes_no(attorney),
        "public_entity": _yes_no(public),
        "more_than_12": _yes_no(more_than_12),
        "more_than_2500": _yes_no(over_2500),
    }
    if dict(expected) != derived_expected:
        errors.append("inconsistent:expected_form_semantics")
    blank_groups = record.get("required_blank_groups")
    if not isinstance(blank_groups, list) or tuple(blank_groups) != _BLANK_GROUPS:
        errors.append("inconsistent:required_blank_groups")

    if errors:
        raise ShadowGoldValidationError(errors)
    oracle_gold = {
        "plaintiff": {
            "name": plaintiff_name,
            "phone": plaintiff_phone,
            **plaintiff_address,
            "email": plaintiff_email,
        },
        "defendant": {
            "name": defendant_name,
            "phone": defendant_phone,
            **defendant_address,
        },
        "venue": {"zip": venue_zip},
        "claim": {
            "amount": str(amount_cents // 100),
            "start_date": event_start,
            "end_date": event_end,
            "contract": _CONTRACT_TEXT[contract_type],
        },
        "questions": {
            "asked_to_pay": asked,
            # SC-100 item 5(a) covers both defendant residence and contract
            # made/performed/broken; the basis difference is expressed by ZIP.
            "venue_choice": 1,
            "attorney_fee_dispute": attorney,
            "public_entity": public,
            "more_than_12_claims": more_than_12,
            "more_than_2500": over_2500,
        },
        "signature": {"date": signature_date, "name": signature_name},
    }
    return AdaptedShadowRecord(case_id, case_class, expected_action, oracle_gold, None)


def adapt_shadow_corpus(records: Sequence[Mapping[str, Any]]) -> tuple[AdaptedShadowRecord, ...]:
    """Adapt the complete 24-row corpus and reject omissions or duplicates."""

    adapted: list[AdaptedShadowRecord] = []
    errors: list[str] = []
    for index, record in enumerate(records):
        try:
            adapted.append(adapt_shadow_record(record))
        except ShadowGoldValidationError as exc:
            errors.extend(f"row[{index}]:{code}" for code in exc.codes)
    ids = [row.case_id for row in adapted]
    if len(ids) != len(set(ids)):
        errors.append("corpus:duplicate_case_id")
    if set(ids) != ALL_CASE_IDS:
        errors.append("corpus:case_id_set")
    if len(records) != 24:
        errors.append("corpus:row_count")
    if errors:
        raise ShadowGoldValidationError(errors)
    return tuple(adapted)


def load_shadow_gold_jsonl(path: str | Path) -> tuple[AdaptedShadowRecord, ...]:
    """Load and adapt the frozen UTF-8 JSONL corpus without reading prompts."""

    rows: list[Mapping[str, Any]] = []
    for index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if not line:
            raise ShadowGoldValidationError((f"row[{index}]:blank_line",))
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ShadowGoldValidationError((f"row[{index}]:invalid_json",)) from exc
        if not isinstance(value, Mapping):
            raise ShadowGoldValidationError((f"row[{index}]:invalid_record",))
        rows.append(value)
    return adapt_shadow_corpus(rows)


__all__ = [
    "ADAPTER_SCHEMA",
    "ALL_CASE_IDS",
    "AdaptedShadowRecord",
    "ShadowGoldValidationError",
    "adapt_shadow_corpus",
    "adapt_shadow_record",
    "load_shadow_gold_jsonl",
]
