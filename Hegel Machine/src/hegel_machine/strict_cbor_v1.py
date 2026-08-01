"""Minimal deterministic CBOR for the Hegel Machine formal hashed core.

The :mod:`phase3_certificate_v1` module predates the v1.0.2 strict-acceptance
amendment and exposes a map-shaped, diagnostic certificate model.  This module
implements the deliberately smaller normative profile frozen by that
amendment.  It is dependency-free and accepts only integers, byte strings,
arrays, booleans, and null.  In particular, text, maps, tags, floats,
indefinite-length items, and non-shortest encodings are never accepted.

Decoder acceptance is intentionally stronger than successful parsing: the
decoded object is re-encoded and the original bytes must match exactly.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Final, TypeAlias


CBOR_PROFILE_ID: Final = "hegel-cbor-det-v1"
MAX_CBOR_ARGUMENT: Final = (1 << 64) - 1
MAX_CBOR_NESTING: Final = 64

StrictCborScalar: TypeAlias = int | bytes | bool | None
StrictCborValue: TypeAlias = StrictCborScalar | tuple["StrictCborValue", ...]


class StrictCborError(ValueError):
    """A stable fail-closed rejection from the strict CBOR profile."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _reject(code: str, detail: str) -> "None":
    raise StrictCborError(code, detail)


def _encode_head(major: int, argument: int) -> bytes:
    if type(argument) is not int or argument < 0 or argument > MAX_CBOR_ARGUMENT:
        _reject("REJECT_CBOR_INTEGER_RANGE", "CBOR argument is outside uint64")
    prefix = major << 5
    if argument <= 23:
        return bytes((prefix | argument,))
    if argument <= 0xFF:
        return bytes((prefix | 24, argument))
    if argument <= 0xFFFF:
        return bytes((prefix | 25,)) + argument.to_bytes(2, "big")
    if argument <= 0xFFFFFFFF:
        return bytes((prefix | 26,)) + argument.to_bytes(4, "big")
    return bytes((prefix | 27,)) + argument.to_bytes(8, "big")


def canonical_cbor_encode(value: object) -> bytes:
    """Encode one value under ``hegel-cbor-det-v1``.

    Python lists and tuples both denote a CBOR array.  The decoded form always
    uses tuples so callers cannot mutate a validated formal object afterward.
    """

    if value is False:
        return b"\xf4"
    if value is True:
        return b"\xf5"
    if value is None:
        return b"\xf6"
    if type(value) is int:
        if value >= 0:
            return _encode_head(0, value)
        argument = -1 - value
        return _encode_head(1, argument)
    if type(value) is bytes:
        return _encode_head(2, len(value)) + value
    if isinstance(value, (list, tuple)):
        encoded = bytearray(_encode_head(4, len(value)))
        for item in value:
            encoded.extend(canonical_cbor_encode(item))
        return bytes(encoded)
    if isinstance(value, str):
        _reject("REJECT_CBOR_TEXT", "text strings are forbidden in the formal core")
    if isinstance(value, dict):
        _reject("REJECT_CBOR_MAP", "maps are forbidden in the formal core")
    if isinstance(value, float):
        _reject("REJECT_CBOR_FLOAT", "floating point is forbidden in the formal core")
    _reject(
        "REJECT_CBOR_TYPE",
        f"unsupported formal CBOR value: {type(value).__name__}",
    )


class _Decoder:
    __slots__ = ("payload", "offset")

    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.offset = 0

    def _take(self, count: int) -> bytes:
        end = self.offset + count
        if end > len(self.payload):
            _reject("REJECT_TRUNCATED_CBOR", "item ends beyond the input")
        result = self.payload[self.offset:end]
        self.offset = end
        return result

    def _argument(self, additional: int) -> int:
        if additional <= 23:
            return additional
        widths = {24: 1, 25: 2, 26: 4, 27: 8}
        if additional == 31:
            _reject(
                "REJECT_INDEFINITE_CBOR",
                "indefinite-length items are forbidden",
            )
        width = widths.get(additional)
        if width is None:
            _reject("REJECT_RESERVED_CBOR", "reserved additional information")
        argument = int.from_bytes(self._take(width), "big")
        minimum = {1: 24, 2: 0x100, 4: 0x10000, 8: 0x100000000}[width]
        if argument < minimum:
            _reject(
                "REJECT_NONCANONICAL_CBOR",
                "integer or length did not use its shortest encoding",
            )
        return argument

    def item(self, nesting: int = 0) -> StrictCborValue:
        if nesting > MAX_CBOR_NESTING:
            _reject(
                "REJECT_CBOR_NESTING",
                f"formal CBOR nesting exceeds {MAX_CBOR_NESTING}",
            )
        if self.offset >= len(self.payload):
            _reject("REJECT_TRUNCATED_CBOR", "expected an item")
        initial = self._take(1)[0]
        major = initial >> 5
        additional = initial & 0x1F

        if major in {0, 1}:
            argument = self._argument(additional)
            return argument if major == 0 else -1 - argument
        if major == 2:
            length = self._argument(additional)
            return self._take(length)
        if major == 3:
            _reject("REJECT_CBOR_TEXT", "text strings are forbidden")
        if major == 4:
            length = self._argument(additional)
            return tuple(self.item(nesting + 1) for _ in range(length))
        if major == 5:
            _reject("REJECT_CBOR_MAP", "maps are forbidden")
        if major == 6:
            _reject("REJECT_CBOR_TAG", "CBOR tags are forbidden")
        if additional == 20:
            return False
        if additional == 21:
            return True
        if additional == 22:
            return None
        if additional == 23:
            _reject("REJECT_CBOR_UNDEFINED", "undefined is forbidden")
        if additional in {25, 26, 27}:
            _reject("REJECT_CBOR_FLOAT", "floating point is forbidden")
        if additional == 31:
            _reject("REJECT_INDEFINITE_CBOR", "break is forbidden")
        _reject("REJECT_CBOR_SIMPLE", "unapproved simple value")


def canonical_cbor_decode(payload: bytes) -> StrictCborValue:
    """Parse, validate, re-encode, and exactly compare one formal value."""

    if type(payload) is not bytes:
        raise TypeError("strict CBOR payload must be bytes")
    decoder = _Decoder(payload)
    value = decoder.item()
    if decoder.offset != len(payload):
        _reject("REJECT_TRAILING_CBOR", "trailing bytes after the first item")
    if canonical_cbor_encode(value) != payload:
        _reject("REJECT_NONCANONICAL_CBOR", "exact re-encoding differs")
    return value


def content_hash(domain: str, value: object) -> bytes:
    """Return ``SHA256(UTF8(domain) || 0x00 || CanonicalCBOR(value))``."""

    if not isinstance(domain, str) or not domain or "\x00" in domain:
        raise ValueError("content-hash domain must be nonempty text without NUL")
    return sha256(
        domain.encode("utf-8") + b"\x00" + canonical_cbor_encode(value)
    ).digest()


def content_hash_id(domain: str, value: object) -> str:
    return "sha256:" + content_hash(domain, value).hex()


def rfc6962_leaf_hash(record: object) -> bytes:
    return sha256(b"\x00" + canonical_cbor_encode(record)).digest()


def rfc6962_node_hash(left: bytes, right: bytes) -> bytes:
    if type(left) is not bytes or type(right) is not bytes:
        raise TypeError("RFC6962 child hashes must be bytes")
    if len(left) != 32 or len(right) != 32:
        raise ValueError("RFC6962 child hashes must be exactly 32 bytes")
    return sha256(b"\x01" + left + right).digest()


def _largest_power_of_two_less_than(value: int) -> int:
    if value <= 1:
        raise ValueError("RFC6962 split requires at least two leaves")
    return 1 << ((value - 1).bit_length() - 1)


def rfc6962_root(records: tuple[object, ...] | list[object]) -> bytes:
    """Compute the RFC6962 Merkle Tree Hash without duplicate-last padding."""

    if not isinstance(records, (tuple, list)):
        raise TypeError("RFC6962 records must be a list or tuple")
    if not records:
        return sha256(b"").digest()
    if len(records) == 1:
        return rfc6962_leaf_hash(records[0])
    split = _largest_power_of_two_less_than(len(records))
    return rfc6962_node_hash(
        rfc6962_root(records[:split]),
        rfc6962_root(records[split:]),
    )


def rfc6962_root_id(records: tuple[object, ...] | list[object]) -> str:
    return "sha256:" + rfc6962_root(records).hex()


__all__ = [
    "CBOR_PROFILE_ID",
    "MAX_CBOR_ARGUMENT",
    "MAX_CBOR_NESTING",
    "StrictCborError",
    "StrictCborValue",
    "canonical_cbor_decode",
    "canonical_cbor_encode",
    "content_hash",
    "content_hash_id",
    "rfc6962_leaf_hash",
    "rfc6962_node_hash",
    "rfc6962_root",
    "rfc6962_root_id",
]
