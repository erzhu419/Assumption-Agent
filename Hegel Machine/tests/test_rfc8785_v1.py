"""RFC 8785/JCS conformance and independent Node/V8 differential tests."""

from __future__ import annotations

import json
import math
import random
import struct
import subprocess
from pathlib import Path

import pytest

from hegel_machine.rfc8785_v1 import (
    RFC8785Error,
    canonicalize_rfc8785_text_v1,
    canonicalize_rfc8785_v1,
    decode_and_require_canonical_rfc8785_v1,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VECTORS_PATH = PROJECT_ROOT / "golden_vectors" / "phase2b_rfc8785_v1.json"
ORACLE_PATH = PROJECT_ROOT / "tools" / "phase2b_rfc8785_oracle_v1.mjs"
RANDOM_DIFFERENTIAL_SEED = 0x8785_2026_0814
RANDOM_FINITE_PATTERNS = 100_000
NODE_BATCH_SIZE = 2_000


def _vectors() -> dict[str, object]:
    return json.loads(VECTORS_PATH.read_text(encoding="utf-8"))


def _parse_as_binary64(text: str) -> object:
    return json.loads(text, parse_int=float, parse_float=float)


def _double_from_bits(hex_bits: str) -> float:
    return struct.unpack(">d", bytes.fromhex(hex_bits))[0]


def test_official_rfc8785_appendix_sample_and_utf8_bytes() -> None:
    vector = _vectors()["official_appendix_sample"]
    value = _parse_as_binary64(vector["input_json"])
    expected = vector["canonical"].encode("utf-8")

    assert canonicalize_rfc8785_v1(value) == expected
    assert canonicalize_rfc8785_text_v1(value) == vector["canonical"]
    assert expected.hex() == (
        "7b226c69746572616c73223a5b6e756c6c2c747275652c66616c73655d2c226e"
        "756d62657273223a5b3333333333333333332e333333333333332c31652b3330"
        "2c342e352c302e3030322c31652d32375d2c22737472696e67223a22e282ac24"
        "5c75303030665c6e4127425c225c5c5c5c5c222f227d"
    )


def test_official_utf16_code_unit_property_sorting() -> None:
    vector = _vectors()["official_property_sorting"]
    value = _parse_as_binary64(vector["input_json"])

    assert canonicalize_rfc8785_text_v1(value) == vector["canonical"]
    assert list(decode_and_require_canonical_rfc8785_v1(
        vector["canonical"].encode("utf-8")
    ).values()) == [
        "Carriage Return",
        "One",
        "Control",
        "Latin Small Letter O With Diaeresis",
        "Euro Sign",
        "Emoji: Grinning Face",
        "Hebrew Letter Dalet With Dagesh",
    ]


def test_official_appendix_b_all_binary64_vectors() -> None:
    vectors = _vectors()["official_appendix_b_numbers"]
    assert len(vectors) == 26
    assert sum("canonical" in vector for vector in vectors) == 24
    assert sum("error" in vector for vector in vectors) == 2

    for vector in vectors:
        value = _double_from_bits(vector["ieee754_hex"])
        if "canonical" in vector:
            assert canonicalize_rfc8785_text_v1(value) == vector["canonical"], vector
        else:
            with pytest.raises(RFC8785Error, match="NaN and Infinity"):
                canonicalize_rfc8785_v1(value)


def test_independent_literal_vectors() -> None:
    vectors = _vectors()["independent_literal_vectors"]
    assert len(vectors) == 5
    for vector in vectors:
        value = _parse_as_binary64(vector["input_json"])
        assert canonicalize_rfc8785_text_v1(value) == vector["canonical"], vector["name"]


def test_utf16_order_really_differs_from_python_codepoint_order() -> None:
    value = {"\ue000": 1, "\U00010000": 2}
    assert sorted(value) == ["\ue000", "\U00010000"]
    assert canonicalize_rfc8785_v1(value) == (
        "{\"\U00010000\":2,\"\ue000\":1}".encode("utf-8")
    )


def test_unicode_is_preserved_without_normalization_or_unneeded_escaping() -> None:
    composed = "é"
    decomposed = "e\N{COMBINING ACUTE ACCENT}"
    assert composed != decomposed
    assert canonicalize_rfc8785_text_v1([composed, decomposed, "/", "\u2028\u2029"]) == (
        '["é","é","/","\u2028\u2029"]'
    )


@pytest.mark.parametrize(
    "value",
    ["\ud800", "\udead", "\ufdd0", "\ufffe", "\U0001ffff", "\U0010ffff"],
)
def test_surrogates_and_ijson_noncharacters_are_rejected(value: str) -> None:
    with pytest.raises(RFC8785Error):
        canonicalize_rfc8785_v1(value)
    with pytest.raises(RFC8785Error):
        canonicalize_rfc8785_v1({value: None})


def test_programmatic_number_domain_and_ecmascript_thresholds() -> None:
    assert canonicalize_rfc8785_text_v1(-0.0) == "0"
    assert canonicalize_rfc8785_text_v1(1e-7) == "1e-7"
    assert canonicalize_rfc8785_text_v1(1e-6) == "0.000001"
    assert canonicalize_rfc8785_text_v1(1e20) == "100000000000000000000"
    assert canonicalize_rfc8785_text_v1(1e21) == "1e+21"
    assert canonicalize_rfc8785_text_v1(2**68) == "295147905179352830000"

    with pytest.raises(RFC8785Error, match="exactly representable"):
        canonicalize_rfc8785_v1(2**53 + 1)
    with pytest.raises(RFC8785Error, match="finite binary64"):
        canonicalize_rfc8785_v1(10**400)
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(RFC8785Error, match="NaN and Infinity"):
            canonicalize_rfc8785_v1(value)


def test_decode_and_require_canonical_accepts_only_exact_bytes() -> None:
    for text in _vectors()["canonical_acceptance"]:
        decoded = decode_and_require_canonical_rfc8785_v1(text.encode("utf-8"))
        assert canonicalize_rfc8785_v1(decoded) == text.encode("utf-8")

    for vector in _vectors()["canonical_rejection"]:
        with pytest.raises(RFC8785Error):
            decode_and_require_canonical_rfc8785_v1(vector["input_json"].encode("utf-8"))

    with pytest.raises(TypeError, match="exact bytes"):
        decode_and_require_canonical_rfc8785_v1(bytearray(b"null"))
    with pytest.raises(RFC8785Error, match="UTF-8"):
        decode_and_require_canonical_rfc8785_v1(b'"\xff"')
    with pytest.raises(RFC8785Error):
        decode_and_require_canonical_rfc8785_v1(b"\xef\xbb\xbfnull")


def test_duplicate_names_are_rejected_after_escape_processing() -> None:
    with pytest.raises(RFC8785Error, match="duplicate"):
        decode_and_require_canonical_rfc8785_v1(b'{"a":1,"\\u0061":2}')
    with pytest.raises(RFC8785Error, match="duplicate"):
        decode_and_require_canonical_rfc8785_v1(
            '{"😀":1,"\\ud83d\\ude00":2}'.encode("utf-8")
        )


def test_json_data_model_is_closed_and_cycles_fail() -> None:
    invalid_values = [b"bytes", (1, 2), {1: "not a string key"}, complex(1, 2)]
    for value in invalid_values:
        with pytest.raises(RFC8785Error):
            canonicalize_rfc8785_v1(value)

    cycle: list[object] = []
    cycle.append(cycle)
    with pytest.raises(RFC8785Error, match="cyclic"):
        canonicalize_rfc8785_v1(cycle)

    shared = [1]
    assert canonicalize_rfc8785_v1([shared, shared]) == b"[[1],[1]]"


def test_node12_v8_oracle_passes_frozen_vectors_without_python_import() -> None:
    source = ORACLE_PATH.read_text(encoding="utf-8")
    assert "hegel_machine" not in source
    assert "importlib" not in source

    completed = subprocess.run(
        ["node", str(ORACLE_PATH), "--check", str(VECTORS_PATH)],
        check=True,
        capture_output=True,
        text=True,
    )
    receipt = json.loads(completed.stdout)
    assert receipt["disposition"] == "PASS"
    assert receipt["oracle"] == "node-v8-json-stringify"
    assert int(receipt["node"].split(".", 1)[0].removeprefix("v")) >= 12
    assert receipt["serialization_cases"] == 7
    assert receipt["number_cases"] == 24
    assert receipt["rejection_cases"] == 2


def test_fixed_seed_100000_finite_binary64_values_match_node_v8() -> None:
    rng = random.Random(RANDOM_DIFFERENTIAL_SEED)
    finite: list[tuple[str, float]] = []
    while len(finite) < RANDOM_FINITE_PATTERNS:
        bits = rng.getrandbits(64)
        value = struct.unpack(">d", bits.to_bytes(8, "big"))[0]
        if math.isfinite(value):
            finite.append((f"{bits:016x}", value))

    checked = 0
    for start in range(0, len(finite), NODE_BATCH_SIZE):
        batch = finite[start : start + NODE_BATCH_SIZE]
        completed = subprocess.run(
            [
                "node",
                str(ORACLE_PATH),
                "--number-bits",
                *(hex_bits for hex_bits, _ in batch),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        oracle_rows = json.loads(completed.stdout)
        assert len(oracle_rows) == len(batch)
        for (hex_bits, value), oracle_row in zip(batch, oracle_rows, strict=True):
            python_spelling = canonicalize_rfc8785_text_v1(value)
            node_spelling = oracle_row.get("canonical")
            assert python_spelling == node_spelling, (
                f"binary64 differential mismatch: bits={hex_bits} "
                f"python={python_spelling!r} node={node_spelling!r}"
            )
            checked += 1

    assert checked == RANDOM_FINITE_PATTERNS
