"""RFC 8785 JSON Canonicalization Scheme (JCS), version 1.

The public surface deliberately accepts only the JSON/I-JSON data model.  It
does not invoke user hooks, normalize Unicode, or silently serialize extended
Python types.  Numbers are serialized with ECMAScript's binary64
``Number::toString`` spelling, including its fixed/scientific thresholds.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Tuple, Union


JsonValue = Union[None, bool, int, float, str, List["JsonValue"], Dict[str, "JsonValue"]]


class RFC8785Error(ValueError):
    """Raised when input is outside RFC 8785's I-JSON domain."""


_SHORT_ESCAPES = {
    0x08: "\\b",
    0x09: "\\t",
    0x0A: "\\n",
    0x0C: "\\f",
    0x0D: "\\r",
}


def _is_unicode_noncharacter(codepoint: int) -> bool:
    return 0xFDD0 <= codepoint <= 0xFDEF or (codepoint & 0xFFFF) in (0xFFFE, 0xFFFF)


def _validate_unicode(value: str) -> None:
    for character in value:
        codepoint = ord(character)
        if 0xD800 <= codepoint <= 0xDFFF:
            raise RFC8785Error("JCS strings must not contain lone surrogate code points")
        if _is_unicode_noncharacter(codepoint):
            raise RFC8785Error("JCS strings must not contain Unicode noncharacters")


def _quote_string(value: str) -> str:
    _validate_unicode(value)
    output = ['"']
    for character in value:
        codepoint = ord(character)
        if character == '"':
            output.append('\\"')
        elif character == "\\":
            output.append("\\\\")
        elif codepoint in _SHORT_ESCAPES:
            output.append(_SHORT_ESCAPES[codepoint])
        elif codepoint <= 0x1F:
            output.append(f"\\u{codepoint:04x}")
        else:
            output.append(character)
    output.append('"')
    return "".join(output)


def _utf16_sort_key(value: str) -> bytes:
    _validate_unicode(value)
    # Big-endian bytes compare exactly like unsigned UTF-16 code units.
    return value.encode("utf-16-be")


def _binary64_to_ecmascript(value: float) -> str:
    """Return ECMAScript JSON's shortest representation of a binary64.

    Python 3's ``repr(float)`` and ECMAScript use the same shortest-roundtrip
    significant digits.  Their presentation thresholds differ, so the
    significant digits are parsed and then laid out using ECMAScript's
    ``1e-6 <= abs(x) < 1e21`` fixed-notation interval.
    """

    if not math.isfinite(value):
        raise RFC8785Error("NaN and Infinity are not permitted by JCS")
    if value == 0.0:
        return "0"

    negative = value < 0.0
    magnitude = -value if negative else value
    spelling = repr(magnitude).lower()

    if "e" in spelling:
        mantissa, exponent_text = spelling.split("e", 1)
        exponent = int(exponent_text)
    else:
        mantissa = spelling
        exponent = 0

    if "." in mantissa:
        integer_part, fractional_part = mantissa.split(".", 1)
    else:
        integer_part, fractional_part = mantissa, ""

    digits = integer_part + fractional_part
    decimal_point = len(integer_part) + exponent

    # repr(float) never supplies redundant leading or trailing significant
    # zeroes, apart from the conventional trailing ".0" for integral floats.
    if fractional_part == "0":
        digits = integer_part

    if 1e-6 <= magnitude < 1e21:
        if decimal_point <= 0:
            rendered = "0." + ("0" * -decimal_point) + digits
        elif decimal_point >= len(digits):
            rendered = digits + ("0" * (decimal_point - len(digits)))
        else:
            rendered = digits[:decimal_point] + "." + digits[decimal_point:]
    else:
        scientific_exponent = decimal_point - 1
        rendered = digits[0]
        if len(digits) > 1:
            rendered += "." + digits[1:]
        rendered += "e"
        rendered += "+" if scientific_exponent >= 0 else "-"
        rendered += str(abs(scientific_exponent))

    return ("-" if negative else "") + rendered


def _number_to_ecmascript(value: object) -> str:
    if type(value) is float:
        return _binary64_to_ecmascript(value)
    if type(value) is int:
        try:
            binary64 = float(value)
        except OverflowError as exc:
            raise RFC8785Error("integer is outside the finite binary64 domain") from exc
        if not math.isfinite(binary64) or int(binary64) != value:
            raise RFC8785Error("integer is not exactly representable as binary64")
        return _binary64_to_ecmascript(binary64)
    raise RFC8785Error("JCS number must be an exact Python int or binary64 float")


def _serialize(value: Any, active_containers: set[int]) -> str:
    value_type = type(value)
    if value is None:
        return "null"
    if value_type is bool:
        return "true" if value else "false"
    if value_type in (int, float):
        return _number_to_ecmascript(value)
    if value_type is str:
        return _quote_string(value)

    if value_type is list:
        identity = id(value)
        if identity in active_containers:
            raise RFC8785Error("cyclic arrays are not JSON values")
        active_containers.add(identity)
        try:
            return "[" + ",".join(_serialize(item, active_containers) for item in value) + "]"
        finally:
            active_containers.remove(identity)

    if value_type is dict:
        identity = id(value)
        if identity in active_containers:
            raise RFC8785Error("cyclic objects are not JSON values")
        active_containers.add(identity)
        try:
            keys = list(value.keys())
            if any(type(key) is not str for key in keys):
                raise RFC8785Error("JCS object member names must be strings")
            keys.sort(key=_utf16_sort_key)
            members = (
                _quote_string(key) + ":" + _serialize(value[key], active_containers)
                for key in keys
            )
            return "{" + ",".join(members) + "}"
        finally:
            active_containers.remove(identity)

    raise RFC8785Error(f"value type {value_type.__name__!r} is outside the JSON data model")


def canonicalize_rfc8785_v1(value: JsonValue) -> bytes:
    """Serialize an I-JSON value as canonical RFC 8785 UTF-8 bytes."""

    return _serialize(value, set()).encode("utf-8")


def canonicalize_rfc8785_text_v1(value: JsonValue) -> str:
    """Serialize an I-JSON value as canonical RFC 8785 Unicode text."""

    return _serialize(value, set())


def _pairs_without_duplicates(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RFC8785Error("JCS objects must not contain duplicate member names")
        result[key] = value
    return result


def _reject_non_json_constant(token: str) -> None:
    raise RFC8785Error(f"non-JSON numeric token {token!r} is not permitted")


def _decode_json_bytes(payload: bytes) -> JsonValue:
    if type(payload) is not bytes:
        raise TypeError("canonical JCS input must be exact bytes")
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise RFC8785Error("JCS input must be well-formed UTF-8") from exc
    try:
        return json.loads(
            text,
            parse_int=float,
            parse_float=float,
            parse_constant=_reject_non_json_constant,
            object_pairs_hook=_pairs_without_duplicates,
        )
    except RFC8785Error:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RFC8785Error("JCS input is not valid JSON") from exc


def decode_and_require_canonical_rfc8785_v1(payload: bytes) -> JsonValue:
    """Decode UTF-8 JSON and require exact byte-for-byte RFC 8785 spelling."""

    decoded = _decode_json_bytes(payload)
    if canonicalize_rfc8785_v1(decoded) != payload:
        raise RFC8785Error("JCS input is valid JSON but is not canonical")
    return decoded


__all__ = [
    "JsonValue",
    "RFC8785Error",
    "canonicalize_rfc8785_text_v1",
    "canonicalize_rfc8785_v1",
    "decode_and_require_canonical_rfc8785_v1",
]
