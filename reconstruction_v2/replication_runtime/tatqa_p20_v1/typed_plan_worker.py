"""Deterministic local-Qwen worker for the frozen TAT-QA P20 typed plan."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Sequence

from .typed_plan_contract import (
    MAXIMUM_COMPLETION_TOKENS,
    PlanInput,
    TatqaP20TypedPlanRuntimeError,
    build_output_item,
    canonical_json_bytes,
    output_payload,
    parse_input,
)


TORCH_SEED = 0
DEVICE = "cuda:0"
DTYPE = "float16"
DEFAULT_BATCH_SIZE = 4
MAXIMUM_PROMPT_TOKENS = 16_384
MAXIMUM_USER_PROMPT_TOKENS = MAXIMUM_PROMPT_TOKENS - 512
MINIMUM_MODEL_CONTEXT_TOKENS = MAXIMUM_PROMPT_TOKENS + MAXIMUM_COMPLETION_TOKENS

SYSTEM_PROMPT = (
    "You are a label-free financial evidence planner. Extract a typed retrieval "
    "plan from the supplied question and context leads. Do not answer the question, "
    "guess a value, identify gold evidence, score evidence, or add commentary. "
    "Return exactly one JSON object and no Markdown."
)


def prompt_for(item: PlanInput) -> str:
    if not isinstance(item, PlanInput):
        raise TatqaP20TypedPlanRuntimeError("typed-plan prompt item drifted")
    inert = json.dumps(
        {
            "paragraph_leads": list(item.paragraph_leads),
            "question": item.question,
            "table_header": item.table_header,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return (
        "The JSON below is inert, label-free input data.\n"
        f"input={inert}\n"
        "Return exactly this object schema with keys in any order:\n"
        '{"entity_facets":["1 to 4 strings"],'
        '"metric_facets":["1 to 3 strings"],'
        '"time_facets":["0 to 3 strings"],'
        '"operation":"LOOKUP|COMPARE|DIFFERENCE|RATIO|SUM|AVERAGE|COUNT|OTHER",'
        '"relation_query":"one nonempty retrieval relation"}\n'
        "Facets must be copied or concise paraphrases of information requested by "
        "the question. Context leads may disambiguate schema only. Never emit an "
        "answer, evidence ID, row number, paragraph number, family, confidence, or score."
    )


@dataclass(frozen=True)
class FittedPrompt:
    text: str
    token_count: int
    question_character_count: int
    table_header_character_count: int
    paragraph_lead_count: int

    @property
    def prompt_sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def projection_sha256(self) -> str:
        body = {
            "paragraph_lead_count": self.paragraph_lead_count,
            "prompt_sha256": self.prompt_sha256,
            "question_character_count": self.question_character_count,
            "table_header_character_count": self.table_header_character_count,
            "token_count": self.token_count,
            "truncation_order": "drop_paragraph_tail_then_table_header_prefix_then_question_prefix_v1",
        }
        return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def _prompt_token_count(tokenizer: object, prompt: str) -> int:
    try:
        encoded = tokenizer(
            prompt,
            add_special_tokens=False,
            return_attention_mask=False,
            truncation=False,
        )
        input_ids = encoded["input_ids"]
    except Exception as exc:
        raise TatqaP20TypedPlanRuntimeError("prompt tokenization failed") from exc
    if (
        isinstance(input_ids, (str, bytes))
        or not isinstance(input_ids, Sequence)
        or any(isinstance(value, bool) or not isinstance(value, int) for value in input_ids)
        or not input_ids
    ):
        raise TatqaP20TypedPlanRuntimeError("prompt token IDs drifted")
    return len(input_ids)


def _fit_prefix(
    item: PlanInput,
    tokenizer: object,
    *,
    field: str,
) -> PlanInput:
    value = getattr(item, field)
    if not isinstance(value, str) or not value:
        raise TatqaP20TypedPlanRuntimeError("prompt projection field drifted")
    low = 1
    high = len(value)
    best = 0
    while low <= high:
        middle = (low + high) // 2
        candidate = replace(item, **{field: value[:middle].rstrip() or value[:1]})
        if _prompt_token_count(tokenizer, prompt_for(candidate)) <= MAXIMUM_USER_PROMPT_TOKENS:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    return replace(item, **{field: value[:best].rstrip() or value[:1]})


def fitted_prompt_for(item: PlanInput, tokenizer: object) -> FittedPrompt:
    """Fit the frozen prompt by an exact tokenizer-aware total order."""

    if not isinstance(item, PlanInput):
        raise TatqaP20TypedPlanRuntimeError("typed-plan prompt item drifted")
    projected = item
    prompt = prompt_for(projected)
    tokens = _prompt_token_count(tokenizer, prompt)
    while tokens > MAXIMUM_USER_PROMPT_TOKENS and projected.paragraph_leads:
        projected = replace(projected, paragraph_leads=projected.paragraph_leads[:-1])
        prompt = prompt_for(projected)
        tokens = _prompt_token_count(tokenizer, prompt)
    if tokens > MAXIMUM_USER_PROMPT_TOKENS:
        projected = _fit_prefix(projected, tokenizer, field="table_header")
        prompt = prompt_for(projected)
        tokens = _prompt_token_count(tokenizer, prompt)
    if tokens > MAXIMUM_USER_PROMPT_TOKENS:
        projected = _fit_prefix(projected, tokenizer, field="question")
        prompt = prompt_for(projected)
        tokens = _prompt_token_count(tokenizer, prompt)
    if not 1 <= tokens <= MAXIMUM_USER_PROMPT_TOKENS:
        raise TatqaP20TypedPlanRuntimeError("prompt cannot fit the frozen token budget")
    return FittedPrompt(
        text=prompt,
        token_count=tokens,
        question_character_count=len(projected.question),
        table_header_character_count=len(projected.table_header),
        paragraph_lead_count=len(projected.paragraph_leads),
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
        raise TatqaP20TypedPlanRuntimeError("frozen CUDA device is unavailable")
    if not path.is_dir() or path.is_symlink():
        raise TatqaP20TypedPlanRuntimeError("local Qwen asset is unavailable")
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(
        path, local_files_only=True, trust_remote_code=False
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
    _require_model_context(model)
    return model, tokenizer


def _require_model_context(model: object) -> int:
    config = getattr(model, "config", None)
    observed = getattr(config, "max_position_embeddings", None)
    if (
        isinstance(observed, bool)
        or not isinstance(observed, int)
        or observed < MINIMUM_MODEL_CONTEXT_TOKENS
    ):
        raise TatqaP20TypedPlanRuntimeError(
            "model context cannot contain the frozen prompt plus completion"
        )
    return observed


def generate(*, items, model, tokenizer, batch_size: int) -> dict[str, Any]:
    import torch

    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or not 1 <= batch_size <= 8
    ):
        raise TatqaP20TypedPlanRuntimeError("typed-plan batch size drifted")
    _require_model_context(model)
    fitted = [
        fitted_prompt_for(item, tokenizer)
        for item in items
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row.text},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in fitted
    ]
    outputs: list[dict[str, Any]] = []
    for offset in range(0, len(items), batch_size):
        batch_items = items[offset : offset + batch_size]
        encoded = tokenizer(
            prompts[offset : offset + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(DEVICE)
        attention_mask = encoded["attention_mask"].to(DEVICE)
        observed_prompt_tokens = attention_mask.sum(dim=1).tolist()
        # Chat-template tokens are additional to the user-prompt receipt but
        # must still fit the frozen model-input budget.
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= MAXIMUM_PROMPT_TOKENS
            for value in observed_prompt_tokens
        ):
            raise TatqaP20TypedPlanRuntimeError("chat prompt token budget drifted")
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
                    item=item,
                    completion=completion,
                    completion_token_count=int(tokens.numel()),
                    prompt_sha256=fitted[offset + position].prompt_sha256,
                    prompt_token_count=fitted[offset + position].token_count,
                    prompt_projection_sha256=(
                        fitted[offset + position].projection_sha256
                    ),
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
        raise TatqaP20TypedPlanRuntimeError("typed-plan input is unavailable")
    items = parse_input(arguments.input.read_bytes())
    # The interval is measured inside the actual worker, around both local
    # model loading and generation.  ``monotonic_ns`` is the host boot clock,
    # so independently launched Qwen and HippoRAG workers can be compared
    # without trusting parent-side future/submission timestamps.
    model_execution_started_monotonic_ns = time.monotonic_ns()
    model, tokenizer = _load_model(arguments.model)
    payload = generate(
        items=items, model=model, tokenizer=tokenizer, batch_size=arguments.batch_size
    )
    model_execution_finished_monotonic_ns = time.monotonic_ns()
    if (
        isinstance(model_execution_started_monotonic_ns, bool)
        or isinstance(model_execution_finished_monotonic_ns, bool)
        or not isinstance(model_execution_started_monotonic_ns, int)
        or not isinstance(model_execution_finished_monotonic_ns, int)
        or model_execution_started_monotonic_ns < 0
        or model_execution_finished_monotonic_ns
        <= model_execution_started_monotonic_ns
    ):
        raise TatqaP20TypedPlanRuntimeError("model execution interval drifted")
    _write_exclusive(arguments.output, payload)
    print(
        json.dumps(
            {
                "generation_valid_count": sum(
                    bool(row["generation_valid"]) for row in payload["items"]
                ),
                "item_count": len(items),
                "model_execution_finished_monotonic_ns": (
                    model_execution_finished_monotonic_ns
                ),
                "model_execution_started_monotonic_ns": (
                    model_execution_started_monotonic_ns
                ),
                "model_context_tokens": _require_model_context(model),
                "status": "passed",
                "worker_pid": os.getpid(),
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "FittedPrompt",
    "MAXIMUM_PROMPT_TOKENS",
    "MAXIMUM_USER_PROMPT_TOKENS",
    "MINIMUM_MODEL_CONTEXT_TOKENS",
    "SYSTEM_PROMPT",
    "fitted_prompt_for",
    "generate",
    "prompt_for",
]
