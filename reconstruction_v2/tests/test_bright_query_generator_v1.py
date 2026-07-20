from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from replication_runtime.bright_query_generator_v1 import contract
from replication_runtime.bright_query_generator_v1 import worker


def _input_bytes(queries=("Why does a cold glass become wet?",)) -> bytes:
    return contract.canonical_json_bytes(
        {
            "items": [
                {"ordinal": index, "query": query}
                for index, query in enumerate(queries)
            ],
            "schema": contract.INPUT_SCHEMA,
        }
    )


def _completion() -> str:
    return json.dumps(
        {
            "entity_query": "cold glass water droplets",
            "relation_query": "glass temperature relation to surface moisture",
            "mechanism_query": "water vapor condensation mechanism on cold surfaces",
            "constraint_query": "humid air sealed room cold drinking glass conditions",
        },
        separators=(",", ":"),
    )


def test_canonical_input_and_exact_completion_contract() -> None:
    items = contract.parse_input(_input_bytes())
    assert items[0].ordinal == 0
    assert items[0].query.startswith("Why")
    expansions = contract.parse_completion(
        _completion(), original_query=items[0].query
    )
    assert len(expansions) == 4
    fenced = "```json\n" + _completion() + "\n```"
    assert contract.parse_completion(fenced, original_query=items[0].query) == expansions


def test_invalid_completion_is_auditable_empty_fallback_without_retry() -> None:
    row = contract.build_output_item(
        ordinal=0,
        completion='{"queries":["wrong schema"]}',
        completion_token_count=8,
        query="Why does a cold glass become wet?",
    )
    assert row["generation_valid"] is False
    assert row["expansions"] == []
    payload = contract.output_payload([row])
    assert contract.parse_output(contract.canonical_json_bytes(payload)) == payload


def test_contract_rejects_noncanonical_ordinals_and_duplicate_expansions() -> None:
    value = json.loads(_input_bytes())
    value["items"][0]["ordinal"] = 1
    with pytest.raises(contract.BrightQueryGeneratorError, match="ordinals"):
        contract.parse_input(contract.canonical_json_bytes(value))
    duplicate = json.dumps(
        {key: "same query" for key in contract.EXPANSION_KEYS},
        separators=(",", ":"),
    )
    with pytest.raises(contract.BrightQueryGeneratorError, match="distinct"):
        contract.parse_completion(duplicate, original_query="original")


class _Tokenizer:
    eos_token_id = 99
    pad_token_id = 99

    @staticmethod
    def apply_chat_template(messages, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return messages[-1]["content"]

    @staticmethod
    def decode(tokens, skip_special_tokens):
        assert skip_special_tokens is True
        return _completion()

    def __call__(self, prompts, **kwargs):
        import torch

        assert len(prompts) == 1
        return {
            "attention_mask": torch.tensor([[1, 1]]),
            "input_ids": torch.tensor([[7, 8]]),
        }


class _Model:
    @staticmethod
    def generate(**kwargs):
        import torch

        assert kwargs["do_sample"] is False
        return torch.tensor([[7, 8, 11, 12, 99]])


def test_generate_uses_completion_only_and_greedy_decode(monkeypatch) -> None:
    import torch

    monkeypatch.setattr(torch.Tensor, "to", lambda self, device: self)
    items = (SimpleNamespace(query="Why does a cold glass become wet?"),)
    payload = worker.generate(
        items=items,
        model=_Model(),
        tokenizer=_Tokenizer(),
        batch_size=1,
    )
    assert payload["items"][0]["generation_valid"] is True
    assert payload["items"][0]["completion_token_count"] == 2
