# Phase-2B RFC 8785/JCS implementation v1

## Scope

`hegel_machine.rfc8785_v1` is the versioned, standard-library-only RFC 8785
canonical JSON primitive for the Phase-2B trusted wire.  This artifact closes
the canonicalizer implementation gap only.  It does not itself construct the
formal wire envelope, run a corpus, allocate campaign/attempt IDs, or satisfy
the full Phase-2B readiness DAG.

Public APIs are:

- `canonicalize_rfc8785_v1(value) -> bytes`;
- `canonicalize_rfc8785_text_v1(value) -> str`;
- `decode_and_require_canonical_rfc8785_v1(payload: bytes) -> JsonValue`.

## Frozen behavior

The implementation emits UTF-8 without inter-token whitespace and recursively
sorts raw object member names by unsigned UTF-16 code units.  It preserves
Unicode strings without normalization, uses ECMAScript JSON string escapes,
and rejects lone surrogates and the Unicode noncharacters excluded by I-JSON.
Duplicate names are rejected after JSON escape processing.

Numbers have the IEEE-754 binary64 data model.  Serialization uses the same
shortest-roundtrip significant digits as Python 3's float representation and
then applies ECMAScript's fixed/scientific presentation boundary:
`1e-6 <= abs(x) < 1e21` is fixed notation.  Negative zero is `0`; NaN and
infinities fail closed.  A programmatic Python integer is accepted only when
it is exactly representable as binary64, preventing silent precision loss.
The byte decoder parses every JSON number as binary64 and accepts input only
when reserialization is byte-for-byte identical.

## Qualification evidence

The frozen vectors contain the RFC 8785 Section 3.2.2 sample, the Section
3.2.3 UTF-16 ordering sample, and every Appendix B bit pattern: 24 accepted
number spellings plus the required NaN/Infinity failures.  Five independent
literal vectors exercise recursive sorting, the UTF-16/code-point ordering
difference, escapes, normalization preservation, and number thresholds.

The independent oracle is an `.mjs` program using only Node/V8 built-ins and
does not import the Python implementation.  On the qualification host,
Node `v12.22.9` with V8 `7.8.279.23-node.56` passed every oracle-addressed
frozen serialization and Appendix-B vector.  The additional canonical-input
acceptance and rejection vectors are exercised by the Python test suite.  A
fixed-seed differential run (`0x878520260814`) compared 100,000 finite random
binary64 bit patterns with exact equality and found no mismatch.  Focused
pytest repeats that full differential run.

Normative sources:

- RFC 8785: <https://www.rfc-editor.org/rfc/rfc8785.html>
- RFC 7493 (I-JSON): <https://www.rfc-editor.org/rfc/rfc7493.html>

This evidence supports the claim “RFC 8785 canonicalizer implementation
qualified against the frozen vectors and Node/V8 differential”.  It is not a
claim of formal proof, complete wire-v3 qualification, covert-audit passage,
Docker execution, or actual-corpus execution.
