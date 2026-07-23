"""Private one-query worker for candidate-restricted BIRCO HippoRAG.

The official HippoRAG source is not modified.  Its local ``TransformersLLM``
backend omits ``finish_reason`` and returns prompt-plus-completion, while the
official OpenIE consumer requires ``finish_reason`` and completion-only text.
This worker installs the same narrow compatibility boundary qualified for the
pinned core, then calls only ``HippoRAG.index`` and ``HippoRAG.retrieve``.

Each process accepts one item and requires a previously nonexistent index root;
eager stage executors and arm-level process pools remain controller concerns.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import threading
from types import MethodType
from typing import Any, Mapping, Sequence

from .contract import (
    INPUT_KEYS,
    INPUT_SCHEMA,
    MAX_CANDIDATE_COUNT,
    MIN_CANDIDATE_COUNT,
    BircoOfficialHippoRAGError,
    CandidateDocument,
    canonical_json_bytes,
    core_query_text,
    output_payload,
    serialize_documents,
    stable_permutation,
    validate_input,
)


OPENIE_MAX_NEW_TOKENS = 96
TORCH_THREAD_COUNT = 2
MAX_MODEL_ALIAS_CHARACTERS = 64
_MODEL_ALIAS_PATTERN = re.compile(
    rf"[A-Za-z0-9][A-Za-z0-9._-]{{0,{MAX_MODEL_ALIAS_CHARACTERS - 1}}}"
)


def _validate_model_alias(value: object, *, label: str) -> str:
    """Require a short cwd-local alias while preserving it for HippoRAG."""

    if not isinstance(value, str):
        raise BircoOfficialHippoRAGError(f"{label} alias is invalid")
    alias = value
    path = Path(alias)
    if (
        len(alias) > MAX_MODEL_ALIAS_CHARACTERS
        or ".." in alias
        or "/" in alias
        or "\\" in alias
        or path.is_absolute()
        or _MODEL_ALIAS_PATTERN.fullmatch(alias) is None
    ):
        raise BircoOfficialHippoRAGError(f"{label} alias is invalid")
    try:
        available = path.is_dir()
    except OSError:
        available = False
    if not available:
        raise BircoOfficialHippoRAGError(f"{label} alias is unavailable")
    return alias


def _load_input(
    path: Path,
) -> tuple[str, str, str, tuple[CandidateDocument, ...], str]:
    if path.is_symlink() or not path.is_file():
        raise BircoOfficialHippoRAGError("worker input is unavailable")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BircoOfficialHippoRAGError("worker input is invalid") from exc
    if (
        not isinstance(value, Mapping)
        or set(value) != INPUT_KEYS
        or value.get("schema") != INPUT_SCHEMA
        or canonical_json_bytes(value) != raw
    ):
        raise BircoOfficialHippoRAGError("worker input envelope drifted")
    return validate_input(
        value.get("work_id"),
        value.get("objective"),
        value.get("query"),
        value.get("documents"),
        value.get("common_projection_sha256"),
    )


def _install_completion_only_backend(core: object) -> None:
    """Adapt the official local backend's return contract, not its prompts."""

    llm = getattr(core, "llm_model", None)
    if llm is None or llm.__class__.__name__ != "TransformersLLM":
        raise BircoOfficialHippoRAGError(
            "official local LLM backend drifted"
        )
    model = getattr(llm, "model", None)
    tokenizer = getattr(llm, "tokenizer", None)
    llm_config = getattr(llm, "llm_config", None)
    if model is None or tokenizer is None or llm_config is None:
        raise BircoOfficialHippoRAGError(
            "official local LLM state is incomplete"
        )
    lock = threading.Lock()

    def compatible_infer(
        self: object,
        messages: Sequence[Mapping[str, str]],
        **kwargs: Any,
    ):
        import torch
        from hipporag.llm.transformers_llm import (
            convert_text_chat_messages_to_input_ids,
        )

        params = deepcopy(getattr(self, "llm_config").generate_params)
        params.update(kwargs)
        maximum = params.get("max_tokens", OPENIE_MAX_NEW_TOKENS)
        if (
            isinstance(maximum, bool)
            or not isinstance(maximum, int)
            or maximum != OPENIE_MAX_NEW_TOKENS
        ):
            raise BircoOfficialHippoRAGError("OpenIE token budget drifted")
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
        finish_reason = (
            "length" if completion_ids.numel() >= maximum else "stop"
        )
        metadata = {
            "completion_tokens": int(completion_ids.numel()),
            "finish_reason": finish_reason,
            "prompt_tokens": int(input_ids.shape[1]),
        }
        return response, metadata, False

    llm.llm_config.generate_params["max_tokens"] = OPENIE_MAX_NEW_TOKENS
    llm.infer = MethodType(compatible_infer, llm)


def _build_core(
    *,
    save_dir: Path,
    llm_model: str,
    embedding_model: str,
    candidate_count: int,
) -> object:
    if (
        isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or not MIN_CANDIDATE_COUNT
        <= candidate_count
        <= MAX_CANDIDATE_COUNT
    ):
        raise BircoOfficialHippoRAGError("candidate count is invalid")

    import torch
    from hipporag import HippoRAG
    from hipporag.utils.config_utils import BaseConfig

    torch.set_num_threads(TORCH_THREAD_COUNT)
    torch.manual_seed(0)
    config = BaseConfig(
        save_dir=str(save_dir),
        llm_name="Transformers/" + llm_model,
        embedding_model_name="Transformers/" + embedding_model,
        openie_mode="online",
        max_new_tokens=OPENIE_MAX_NEW_TOKENS,
        retrieval_top_k=candidate_count,
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
    *,
    core: object,
    work_id: str,
    objective: str,
    query: str,
    documents: Sequence[CandidateDocument],
    common_projection_sha256: str,
) -> dict[str, Any]:
    """Index and retrieve exactly one validated common-projection pool."""

    # Revalidate the public values even when this helper is called directly by
    # a controller or a qualification test rather than through ``_load_input``.
    (
        validated_work_id,
        validated_objective,
        validated_query,
        validated_documents,
        validated_projection_hash,
    ) = validate_input(
        work_id,
        objective,
        query,
        [
            {"ordinal": row.ordinal, "text": row.text}
            if isinstance(row, CandidateDocument)
            else row
            for row in documents
        ],
        common_projection_sha256,
    )
    serialized = serialize_documents(validated_documents)
    mapping = {
        text: row.ordinal
        for text, row in zip(serialized, validated_documents)
    }
    if len(mapping) != len(validated_documents):
        raise BircoOfficialHippoRAGError(
            "serialized candidate mapping is not unique"
        )
    index = getattr(core, "index", None)
    retrieve = getattr(core, "retrieve", None)
    if not callable(index) or not callable(retrieve):
        raise BircoOfficialHippoRAGError(
            "official core methods are unavailable"
        )
    index(list(serialized))
    candidate_count = len(serialized)
    official_query = core_query_text(
        objective=validated_objective, query=validated_query
    )
    rows = retrieve([official_query], num_to_retrieve=candidate_count)
    if not isinstance(rows, list) or len(rows) != 1:
        raise BircoOfficialHippoRAGError("official query result drifted")
    solution = rows[0]
    rank_ordinals = stable_permutation(
        retrieved_documents=getattr(solution, "docs", None),
        retrieved_scores=getattr(solution, "doc_scores", None),
        document_to_ordinal=mapping,
    )
    graph = getattr(core, "graph", None)
    vcount = getattr(graph, "vcount", None)
    ecount = getattr(graph, "ecount", None)
    if not callable(vcount) or not callable(ecount):
        raise BircoOfficialHippoRAGError(
            "official graph counters are unavailable"
        )
    return output_payload(
        work_id=validated_work_id,
        common_projection_sha256=validated_projection_hash,
        candidate_count=candidate_count,
        rank_ordinals=rank_ordinals,
        graph_nodes=vcount(),
        graph_edges=ecount(),
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
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--embedding-model", required=True)
    arguments = parser.parse_args(argv)
    llm_model = _validate_model_alias(
        arguments.llm_model, label="LLM model"
    )
    embedding_model = _validate_model_alias(
        arguments.embedding_model, label="embedding model"
    )
    if arguments.index_root.exists() or arguments.index_root.is_symlink():
        raise BircoOfficialHippoRAGError("index root already exists")
    arguments.index_root.mkdir(mode=0o700)
    work_id, objective, query, documents, projection_hash = _load_input(
        arguments.input
    )
    core = _build_core(
        save_dir=arguments.index_root,
        llm_model=llm_model,
        embedding_model=embedding_model,
        candidate_count=len(documents),
    )
    payload = retrieve_with_core(
        core=core,
        work_id=work_id,
        objective=objective,
        query=query,
        documents=documents,
        common_projection_sha256=projection_hash,
    )
    _write_output(arguments.output, payload)
    print(
        json.dumps(
            {
                "candidate_count": payload["candidate_count"],
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
