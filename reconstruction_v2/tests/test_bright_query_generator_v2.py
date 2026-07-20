from __future__ import annotations

import json
from types import SimpleNamespace

from replication_runtime.bright_query_generator_v2 import worker


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


def test_schedule_is_stable_and_bounds_every_non_singleton_batch() -> None:
    lengths = (300, 350, 400, 3858, 450, 800, 900, 200)
    schedule = worker.build_schedule(lengths)
    assert schedule == ((7, 0, 1, 2, 4), (5, 6), (3,))
    assert sorted(index for batch in schedule for index in batch) == list(range(8))
    for batch in schedule:
        padded = max(lengths[index] for index in batch) * len(batch)
        assert len(batch) == 1 or padded <= worker.PADDED_PROMPT_TOKEN_BUDGET


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

        if isinstance(prompts, str):
            return {"input_ids": [7, 8]}
        width = 2
        return {
            "attention_mask": torch.ones((len(prompts), width), dtype=torch.int64),
            "input_ids": torch.tensor([[7, 8]] * len(prompts)),
        }


class _Model:
    @staticmethod
    def generate(**kwargs):
        import torch

        assert kwargs["do_sample"] is False
        prefix = kwargs["input_ids"]
        suffix = torch.tensor([[11, 12, 99]] * prefix.shape[0])
        return torch.cat((prefix, suffix), dim=1)


def test_generation_restores_original_ordinals_and_reports_schedule(monkeypatch) -> None:
    import torch

    monkeypatch.setattr(torch.Tensor, "to", lambda self, device: self)
    items = tuple(
        SimpleNamespace(query=f"Why does query {index} happen?") for index in range(3)
    )
    payload, receipt = worker.generate(
        items=items,
        model=_Model(),
        tokenizer=_Tokenizer(),
    )
    assert [row["ordinal"] for row in payload["items"]] == [0, 1, 2]
    assert all(row["generation_valid"] for row in payload["items"])
    assert receipt["batch_sizes"] == [3]
    assert receipt["maximum_padded_prompt_tokens"] == 6
