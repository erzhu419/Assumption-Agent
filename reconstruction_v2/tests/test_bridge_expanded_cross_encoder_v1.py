from __future__ import annotations

import pytest

from reconstruction_v2.replication_runtime.bridge_expanded_cross_encoder_v1 import contract


def _documents(count: int = 32):
    return [{"content": f"document {index}", "ordinal": index} for index in range(count)]


def _item(count: int = 32):
    return {
        "documents": _documents(count),
        "mechanism_query": "mechanism query",
        "ordinal": 0,
        "relation_query": "relation query",
    }


def test_input_round_trip_at_minimum_and_maximum() -> None:
    for count in (contract.MINIMUM_DOCUMENT_COUNT, contract.MAXIMUM_DOCUMENT_COUNT):
        payload = contract.input_payload([_item(count)])
        parsed = contract.parse_input(contract.canonical_json_bytes(payload))
        assert len(parsed) == 1
        assert len(parsed[0].documents) == count


def test_input_rejects_pool_outside_bound() -> None:
    with pytest.raises(contract.BridgeExpandedCrossEncoderError):
        contract.input_payload([_item(contract.MINIMUM_DOCUMENT_COUNT - 1)])
    with pytest.raises(contract.BridgeExpandedCrossEncoderError):
        contract.input_payload([_item(contract.MAXIMUM_DOCUMENT_COUNT + 1)])


def test_input_rejects_document_ordinal_drift() -> None:
    value = _item()
    value["documents"][3]["ordinal"] = 4
    with pytest.raises(contract.BridgeExpandedCrossEncoderError):
        contract.input_payload([value])


def test_output_round_trip_preserves_separate_score_vectors() -> None:
    count = 40
    row = contract.output_item(
        ordinal=0,
        relation_scores_quantized=tuple(range(count)),
        mechanism_scores_quantized=tuple(range(count, 0, -1)),
    )
    payload = contract.output_payload([row])
    parsed = contract.parse_output(contract.canonical_json_bytes(payload))
    assert parsed["items"][0]["document_count"] == count
    assert parsed["items"][0]["relation_scores_quantized"] == list(range(count))


def test_output_rejects_shape_mismatch() -> None:
    with pytest.raises(contract.BridgeExpandedCrossEncoderError):
        contract.output_item(
            ordinal=0,
            relation_scores_quantized=[0] * 32,
            mechanism_scores_quantized=[0] * 31,
        )


def test_canonical_envelope_is_required() -> None:
    payload = contract.input_payload([_item()])
    raw = contract.canonical_json_bytes(payload)
    with pytest.raises(contract.BridgeExpandedCrossEncoderError):
        contract.parse_input(raw.replace(b'"items":', b'"items" :', 1))
