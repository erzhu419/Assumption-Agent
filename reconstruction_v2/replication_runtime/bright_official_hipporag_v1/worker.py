"""Private offline worker for candidate-restricted BRIGHT HippoRAG.

The official HippoRAG source is not modified.  Its local ``TransformersLLM``
backend omits ``finish_reason`` and returns prompt-plus-completion, while the
official OpenIE consumer requires ``finish_reason`` and completion-only text.
This worker installs the minimal backend compatibility boundary before any
indexing, then calls the official ``HippoRAG.index`` and ``HippoRAG.retrieve``
methods unchanged.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import threading
from types import MethodType
from typing import Any, Mapping, Sequence

from .contract import (
    CANDIDATE_COUNT,
    INPUT_SCHEMA,
    BrightOfficialHippoRAGError,
    canonical_json_bytes,
    output_payload,
    serialize_documents,
    stable_top_k,
    validate_input,
)


OPENIE_MAX_NEW_TOKENS = 96
TORCH_THREAD_COUNT = 2


def _load_input(path: Path) -> tuple[str, tuple[Any, ...]]:
    if path.is_symlink() or not path.is_file():
        raise BrightOfficialHippoRAGError("worker input is unavailable")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BrightOfficialHippoRAGError("worker input is invalid") from exc
    if (
        not isinstance(value, Mapping)
        or set(value) != {"documents", "query", "schema"}
        or value.get("schema") != INPUT_SCHEMA
        or canonical_json_bytes(value) != raw
    ):
        raise BrightOfficialHippoRAGError("worker input envelope drifted")
    return validate_input(value.get("query"), value.get("documents"))


def _install_completion_only_backend(core: object) -> None:
    """Adapt the official local backend's return contract, not its prompts."""

    llm = getattr(core, "llm_model", None)
    if llm is None or llm.__class__.__name__ != "TransformersLLM":
        raise BrightOfficialHippoRAGError("official local LLM backend drifted")
    model = getattr(llm, "model", None)
    tokenizer = getattr(llm, "tokenizer", None)
    llm_config = getattr(llm, "llm_config", None)
    if model is None or tokenizer is None or llm_config is None:
        raise BrightOfficialHippoRAGError("official local LLM state is incomplete")
    lock = threading.Lock()

    def compatible_infer(self: object, messages: Sequence[Mapping[str, str]], **kwargs: Any):
        import torch
        from hipporag.llm.transformers_llm import (
            convert_text_chat_messages_to_input_ids,
        )

        params = deepcopy(getattr(self, "llm_config").generate_params)
        params.update(kwargs)
        maximum = params.get("max_tokens", OPENIE_MAX_NEW_TOKENS)
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum != OPENIE_MAX_NEW_TOKENS:
            raise BrightOfficialHippoRAGError("OpenIE token budget drifted")
        input_ids = convert_text_chat_messages_to_input_ids(
            list(messages), getattr(self, "tokenizer")
        ).to(getattr(self, "model").device)
        attention_mask = torch.ones_like(input_ids)
        with lock, torch.inference_mode():
            generated = getattr(self, "model").generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=maximum,
                do_sample=False,
                pad_token_id=getattr(self, "tokenizer").eos_token_id,
            )
        completion_ids = generated[0, input_ids.shape[1] :]
        response = getattr(self, "tokenizer").decode(
            completion_ids, skip_special_tokens=True
        )
        finish_reason = "length" if completion_ids.numel() >= maximum else "stop"
        metadata = {
            "completion_tokens": int(completion_ids.numel()),
            "finish_reason": finish_reason,
            "prompt_tokens": int(input_ids.shape[1]),
        }
        return response, metadata, False

    llm.llm_config.generate_params["max_tokens"] = OPENIE_MAX_NEW_TOKENS
    llm.infer = MethodType(compatible_infer, llm)


def _build_core(*, save_dir: Path, llm_model: Path, embedding_model: Path) -> object:
    import torch
    from hipporag import HippoRAG
    from hipporag.utils.config_utils import BaseConfig

    torch.set_num_threads(TORCH_THREAD_COUNT)
    torch.manual_seed(0)
    config = BaseConfig(
        save_dir=str(save_dir),
        llm_name="Transformers/" + str(llm_model),
        embedding_model_name="Transformers/" + str(embedding_model),
        openie_mode="online",
        max_new_tokens=OPENIE_MAX_NEW_TOKENS,
        retrieval_top_k=CANDIDATE_COUNT,
        qa_top_k=10,
        force_index_from_scratch=True,
        save_openie=True,
        temperature=0,
        seed=0,
    )
    core = HippoRAG(global_config=config)
    _install_completion_only_backend(core)
    return core


def retrieve_with_core(
    *, core: object, query: str, documents: Sequence[Any]
) -> dict[str, Any]:
    serialized = serialize_documents(documents)
    mapping = {text: row.ordinal for text, row in zip(serialized, documents)}
    index = getattr(core, "index", None)
    retrieve = getattr(core, "retrieve", None)
    if not callable(index) or not callable(retrieve):
        raise BrightOfficialHippoRAGError("official core methods are unavailable")
    index(list(serialized))
    rows = retrieve([query], num_to_retrieve=CANDIDATE_COUNT)
    if not isinstance(rows, list) or len(rows) != 1:
        raise BrightOfficialHippoRAGError("official query result drifted")
    solution = rows[0]
    top = stable_top_k(
        retrieved_documents=getattr(solution, "docs", None),
        retrieved_scores=getattr(solution, "doc_scores", None),
        document_to_ordinal=mapping,
    )
    graph = getattr(core, "graph", None)
    vcount = getattr(graph, "vcount", None)
    ecount = getattr(graph, "ecount", None)
    if not callable(vcount) or not callable(ecount):
        raise BrightOfficialHippoRAGError("official graph counters are unavailable")
    return output_payload(
        top_ordinals=top,
        graph_nodes=int(vcount()),
        graph_edges=int(ecount()),
    )


def _write_output(path: Path, payload: Mapping[str, Any]) -> None:
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--index-root", required=True, type=Path)
    parser.add_argument("--llm-model", required=True, type=Path)
    parser.add_argument("--embedding-model", required=True, type=Path)
    arguments = parser.parse_args(argv)
    if arguments.index_root.exists() or arguments.index_root.is_symlink():
        raise BrightOfficialHippoRAGError("index root already exists")
    arguments.index_root.mkdir(mode=0o700)
    query, documents = _load_input(arguments.input)
    core = _build_core(
        save_dir=arguments.index_root,
        llm_model=arguments.llm_model,
        embedding_model=arguments.embedding_model,
    )
    payload = retrieve_with_core(core=core, query=query, documents=documents)
    _write_output(arguments.output, payload)
    print(
        json.dumps(
            {
                "graph_edge_count": payload["graph_edge_count"],
                "graph_node_count": payload["graph_node_count"],
                "status": "passed",
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
