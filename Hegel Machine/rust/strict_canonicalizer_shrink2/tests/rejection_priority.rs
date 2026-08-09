use hegel_strict_canonicalizer::{encode_strict_cbor_json, hex_decode};
use hegel_strict_canonicalizer_shrink2::{
    canonicalize_shrink2_source_json, decode_shrink2_canonical_ast,
};
use serde_json::{json, Value};

fn assert_source_error(source: Value, expected: &str) {
    let error = canonicalize_shrink2_source_json(&source)
        .expect_err("source rejection-priority vector must reject");
    assert_eq!(error.code, expected, "{source}");
}

fn assert_formal_error(formal: Value, expected: &str) {
    let payload = encode_strict_cbor_json(&formal).expect("formal vector must encode");
    let error = decode_shrink2_canonical_ast(&payload)
        .expect_err("formal rejection-priority vector must reject");
    assert_eq!(error.code, expected, "{formal}");
}

#[test]
fn source_parser_preserves_left_to_right_failure_priority() {
    let cases = [
        (json!(["top_level_AND"]), "REJECT_EMPTY_CONJUNCTION"),
        (
            json!(["sign", ["bit_at", 0]]),
            "REJECT_TYPE_MISMATCH",
        ),
        (
            json!(["add", ["set_size"], ["bit_at", 0]]),
            "REJECT_IMPLICIT_COERCION",
        ),
        (
            json!(["approx_equal", ["scalar_const", 1], ["bit_at", 0], 1]),
            "REJECT_TYPE_MISMATCH",
        ),
        (
            json!(["add", ["set_size"], ["not_in_old_dsl"]]),
            "REJECT_UNKNOWN_EXPRESSION",
        ),
        (
            json!(["add", ["sign", ["bit_at", 0]], ["not_in_old_dsl"]]),
            "REJECT_TYPE_MISMATCH",
        ),
        (
            json!(["aggregate", 99, 0, 0, [[0]]]),
            "REJECT_REGISTRY_INDEX_OUT_OF_RANGE",
        ),
        (
            json!(["aggregate", 0, 0, 0, [[99, "not-a-bool"]]]),
            "REJECT_MALFORMED_SOURCE_AST",
        ),
        (
            json!(["new_symbol_call", ["scalar_const", 99]]),
            "REJECT_NEW_SYMBOL_IN_OLD_DSL",
        ),
    ];
    for (source, expected) in cases {
        assert_source_error(source, expected);
    }
}

#[test]
fn formal_subtree_failure_precedes_later_siblings_and_fields() {
    let cases = [
        (json!([1, [0, 6]]), "REJECT_NEW_SYMBOL_IN_OLD_DSL"),
        (
            json!([1, [0, 6, -10]]),
            "REJECT_NEW_SYMBOL_IN_OLD_DSL",
        ),
        (
            json!([1, [1, 3, [0, 1, 0]]]),
            "REJECT_TYPE_MISMATCH",
        ),
        (
            json!([1, [2, 0, [1, 3, [0, 1, 0]], [0, 99]]]),
            "REJECT_TYPE_MISMATCH",
        ),
        (
            json!([1, [3, 0, [0, 0, 1], [0, 1, 0], 99]]),
            "REJECT_TYPE_MISMATCH",
        ),
        (
            json!([1, [0, 3, 99, 0, 0, [[0]]]]),
            "REJECT_REGISTRY_INDEX_OUT_OF_RANGE",
        ),
    ];
    for (formal, expected) in cases {
        assert_formal_error(formal, expected);
    }
}

#[test]
fn formal_cbor_failure_taxonomy_matches_the_frozen_wire_profile() {
    let cases = [
        ("", "REJECT_TRUNCATED_CBOR"),
        ("8201", "REJECT_TRUNCATED_CBOR"),
        ("1901", "REJECT_TRUNCATED_CBOR"),
        ("1c", "REJECT_RESERVED_CBOR"),
        ("f7", "REJECT_CBOR_UNDEFINED"),
        ("e0", "REJECT_CBOR_SIMPLE"),
    ];
    for (payload_hex, expected) in cases {
        let payload = hex_decode(payload_hex).expect("test vector hex must decode");
        let error = decode_shrink2_canonical_ast(&payload)
            .expect_err("formal CBOR failure vector must reject");
        assert_eq!(error.code, expected, "{payload_hex}");
    }
}
