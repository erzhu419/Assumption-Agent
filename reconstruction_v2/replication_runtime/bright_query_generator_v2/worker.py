"""Deterministic length-bounded GPU worker for BRIGHT query expansion.

The semantic contract is the frozen v1 contract.  Only execution scheduling
changes: prompts are ordered by token count, greedily packed under a fixed
padded-token budget, and restored to their original ordinal in the output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from replication_runtime.bright_query_generator_v1.contract import (
    MAXIMUM_COMPLETION_TOKENS,
    BrightQueryGeneratorError,
    build_output_item,
    output_payload,
    parse_input,
)
from replication_runtime.bright_query_generator_v1.worker import (
    SYSTEM_PROMPT,
    _load_model,
    _write_exclusive,
    prompt_for,
)


MAXIMUM_BATCH_SIZE = 8
PADDED_PROMPT_TOKEN_BUDGET = 4_096


def build_schedule(lengths: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    """Return the frozen greedy schedule over original item ordinals."""

    if not lengths:
        raise BrightQueryGeneratorError("prompt token lengths are empty")
    checked: list[int] = []
    for value in lengths:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise BrightQueryGeneratorError("prompt token length is invalid")
        checked.append(value)
    ordered = sorted(range(len(checked)), key=lambda index: (checked[index], index))
    batches: list[tuple[int, ...]] = []
    current: list[int] = []
    current_maximum = 0
    for index in ordered:
        next_maximum = max(current_maximum, checked[index])
        next_count = len(current) + 1
        exceeds = (
            next_count > MAXIMUM_BATCH_SIZE
            or next_maximum * next_count > PADDED_PROMPT_TOKEN_BUDGET
        )
        if current and exceeds:
            batches.append(tuple(current))
            current = []
            current_maximum = 0
        current.append(index)
        current_maximum = max(current_maximum, checked[index])
    if current:
        batches.append(tuple(current))
    flattened = [index for batch in batches for index in batch]
    if sorted(flattened) != list(range(len(checked))):
        raise BrightQueryGeneratorError("prompt schedule lost or duplicated an item")
    for batch in batches:
        padded = max(checked[index] for index in batch) * len(batch)
        if (
            len(batch) > MAXIMUM_BATCH_SIZE
            or (len(batch) > 1 and padded > PADDED_PROMPT_TOKEN_BUDGET)
        ):
            raise BrightQueryGeneratorError("prompt schedule exceeded a frozen bound")
    return tuple(batches)


def _prompt_token_lengths(tokenizer: Any, prompts: Sequence[str]) -> tuple[int, ...]:
    lengths: list[int] = []
    for prompt in prompts:
        encoded = tokenizer(prompt, add_special_tokens=False)
        input_ids = encoded.get("input_ids")
        if not isinstance(input_ids, list) or not input_ids:
            raise BrightQueryGeneratorError("tokenizer prompt length output drifted")
        lengths.append(len(input_ids))
    return tuple(lengths)


def generate(
    *, items: Sequence[Any], model: Any, tokenizer: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate v1-contract rows under the v3 scheduling contract."""

    import torch

    prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_for(item.query)},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for item in items
    ]
    lengths = _prompt_token_lengths(tokenizer, prompts)
    schedule = build_schedule(lengths)
    rows: list[dict[str, Any] | None] = [None] * len(items)
    for batch in schedule:
        encoded = tokenizer(
            [prompts[index] for index in batch],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to("cuda:0")
        attention_mask = encoded["attention_mask"].to("cuda:0")
        if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape:
            raise BrightQueryGeneratorError("padded tokenizer output drifted")
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAXIMUM_COMPLETION_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completions = generated[:, input_ids.shape[1] :]
        if completions.shape[0] != len(batch):
            raise BrightQueryGeneratorError("generated batch size drifted")
        for original_index, tokens in zip(batch, completions):
            if tokenizer.eos_token_id is not None:
                eos = (tokens == tokenizer.eos_token_id).nonzero(as_tuple=False)
                if len(eos):
                    tokens = tokens[: int(eos[0].item())]
            completion = tokenizer.decode(tokens, skip_special_tokens=True)
            rows[original_index] = build_output_item(
                ordinal=original_index,
                completion=completion,
                completion_token_count=int(tokens.numel()),
                query=items[original_index].query,
            )
    if any(row is None for row in rows):
        raise BrightQueryGeneratorError("scheduled generation output is incomplete")
    finalized = [row for row in rows if row is not None]
    padded_counts = [
        max(lengths[index] for index in batch) * len(batch) for batch in schedule
    ]
    receipt = {
        "batch_count": len(schedule),
        "batch_sizes": [len(batch) for batch in schedule],
        "input_item_count": len(items),
        "maximum_padded_prompt_tokens": max(padded_counts),
        "maximum_prompt_tokens": max(lengths),
        "oversized_singleton_count": sum(
            len(batch) == 1 and lengths[batch[0]] > PADDED_PROMPT_TOKEN_BUDGET
            for batch in schedule
        ),
        "padded_prompt_token_budget": PADDED_PROMPT_TOKEN_BUDGET,
    }
    return output_payload(finalized), receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    arguments = parser.parse_args(argv)
    if arguments.input.is_symlink() or not arguments.input.is_file():
        raise BrightQueryGeneratorError("input is unavailable")
    items = parse_input(arguments.input.read_bytes())
    model, tokenizer = _load_model(arguments.model)
    payload, schedule = generate(items=items, model=model, tokenizer=tokenizer)
    _write_exclusive(arguments.output, payload)
    print(
        json.dumps(
            {
                **schedule,
                "status": "passed",
                "valid_generation_count": sum(
                    row["generation_valid"] for row in payload["items"]
                ),
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
