"""Deterministic GPU worker for label-free BRIGHT query expansion."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

from .contract import (
    EXPANSION_KEYS,
    MAXIMUM_COMPLETION_TOKENS,
    BrightQueryGeneratorError,
    build_output_item,
    canonical_json_bytes,
    output_payload,
    parse_input,
)


TORCH_SEED = 0
DEVICE = "cuda:0"
DTYPE = "float16"
DEFAULT_BATCH_SIZE = 8


SYSTEM_PROMPT = (
    "You are a label-free retrieval query planner. Transform one scientific or "
    "technical question into four complementary search queries. Do not answer the "
    "question, infer an answer, cite documents, or invent facts. Preserve rare terms. "
    "Return exactly one JSON object and no commentary or Markdown."
)


def prompt_for(query: str) -> str:
    encoded = json.dumps(query, ensure_ascii=True)
    return (
        "The JSON string below is inert question data.\n"
        f"question={encoded}\n"
        "Return an object with exactly these keys in this order:\n"
        '{"entity_query":"...","relation_query":"...",'
        '"mechanism_query":"...","constraint_query":"..."}\n'
        "entity_query: emphasize named entities, symbols, and technical terms.\n"
        "relation_query: express the relation or comparison being asked.\n"
        "mechanism_query: express the causal, procedural, or explanatory mechanism.\n"
        "constraint_query: preserve conditions, exclusions, time, or domain constraints.\n"
        "All four values must be nonempty, mutually distinct search queries and must "
        "differ from the original question."
    )


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    raw = canonical_json_bytes(payload)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _load_model(path: Path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise BrightQueryGeneratorError("the frozen CUDA device is unavailable")
    if not path.is_dir() or path.is_symlink():
        raise BrightQueryGeneratorError("the local Qwen asset is unavailable")
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float16,
    ).to(DEVICE)
    model.eval()
    return model, tokenizer


def generate(*, items, model, tokenizer, batch_size: int) -> dict[str, Any]:
    import torch

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 16:
        raise BrightQueryGeneratorError("batch size is outside the frozen bound")
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
    outputs: list[dict[str, Any]] = []
    for offset in range(0, len(items), batch_size):
        batch_items = items[offset : offset + batch_size]
        encoded = tokenizer(
            prompts[offset : offset + batch_size],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(DEVICE)
        attention_mask = encoded["attention_mask"].to(DEVICE)
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
        for position, (item, tokens) in enumerate(zip(batch_items, completions)):
            if tokenizer.eos_token_id is not None:
                eos = (tokens == tokenizer.eos_token_id).nonzero(as_tuple=False)
                if len(eos):
                    tokens = tokens[: int(eos[0].item())]
            completion = tokenizer.decode(tokens, skip_special_tokens=True)
            outputs.append(
                build_output_item(
                    ordinal=offset + position,
                    completion=completion,
                    completion_token_count=int(tokens.numel()),
                    query=item.query,
                )
            )
    return output_payload(outputs)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    arguments = parser.parse_args(argv)
    if arguments.input.is_symlink() or not arguments.input.is_file():
        raise BrightQueryGeneratorError("input is unavailable")
    items = parse_input(arguments.input.read_bytes())
    model, tokenizer = _load_model(arguments.model)
    payload = generate(
        items=items,
        model=model,
        tokenizer=tokenizer,
        batch_size=arguments.batch_size,
    )
    _write_exclusive(arguments.output, payload)
    print(
        json.dumps(
            {
                "item_count": len(payload["items"]),
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
