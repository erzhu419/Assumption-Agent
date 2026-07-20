from __future__ import annotations

import pytest

from replication_runtime.bright_cross_encoder_v1 import contract


def _items() -> list[dict[str, object]]:
    return [
        {
            "documents": [
                {"content": f"document {index}", "ordinal": index}
                for index in range(contract.CANDIDATE_COUNT)
            ],
            "mechanism_query": "causal mechanism query",
            "ordinal": 0,
            "relation_query": "relation comparison query",
        }
    ]


def test_input_round_trip() -> None:
    payload = contract.input_payload(_items())
    parsed = contract.parse_input(contract.canonical_json_bytes(payload))
    assert len(parsed) == 1
    assert parsed[0].documents[-1].ordinal == 31


def test_input_rejects_document_ordinal_drift() -> None:
    rows = _items()
    rows[0]["documents"][3]["ordinal"] = 4  # type: ignore[index]
    with pytest.raises(contract.BrightCrossEncoderError):
        contract.input_payload(rows)


def test_output_quantized_tie_break_and_round_trip() -> None:
    scores = [0] * contract.CANDIDATE_COUNT
    scores[7] = 2
    scores[4] = 2
    scores[2] = 1
    row = contract.output_item(ordinal=0, mean_logit_quantized=scores)
    assert row["ranked_ordinals"][:3] == [4, 7, 2]
    payload = contract.output_payload([row])
    assert contract.parse_output(contract.canonical_json_bytes(payload)) == payload


def test_output_rejects_noncanonical_ranking() -> None:
    row = contract.output_item(
        ordinal=0, mean_logit_quantized=list(range(contract.CANDIDATE_COUNT))
    )
    row["ranked_ordinals"] = list(reversed(row["ranked_ordinals"]))
    with pytest.raises(contract.BrightCrossEncoderError):
        contract.output_payload([row])
