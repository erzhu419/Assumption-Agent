"""Private fresh-index worker for item-local official HippoRAG retrieval."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from replication_runtime.musique_official_hipporag_v1.worker import (
    _build_official_core as _build_musique_official_core,
)

from .contract import (
    INPUT_SCHEMA,
    EraserEvidenceInferenceOfficialHippoRAGError,
    canonical_json_bytes,
    exact_text_quotient,
    expand_ranked_quotient_to_top_five,
    validate_single_item,
)


def retrieve_ordinals_with_core(
    *,
    core: object,
    query: str,
    sentence_texts: Sequence[str],
) -> tuple[int, ...]:
    """Index one exact-text quotient once and return five logical ordinals."""

    validated_query, validated_sentences = validate_single_item(
        query, sentence_texts
    )
    quotient, document_to_ordinals = exact_text_quotient(validated_sentences)
    index = getattr(core, "index", None)
    retrieve = getattr(core, "retrieve", None)
    if not callable(index) or not callable(retrieve):
        raise EraserEvidenceInferenceOfficialHippoRAGError(
            "official core lacks index or retrieve"
        )
    # These are the original sentence strings.  Do not serialize or decorate.
    index(list(quotient))
    rows = retrieve(
        [validated_query],
        num_to_retrieve=len(quotient),
    )
    if not isinstance(rows, list) or len(rows) != 1:
        raise EraserEvidenceInferenceOfficialHippoRAGError(
            "official core returned an invalid item-local query batch"
        )
    ranked_documents = getattr(rows[0], "docs", None)
    return expand_ranked_quotient_to_top_five(
        retrieved_documents=ranked_documents,
        document_to_ordinals=document_to_ordinals,
        logical_sentence_count=len(validated_sentences),
    )


def _load_input(path: Path) -> tuple[str, list[str]]:
    if path.is_symlink() or not path.is_file():
        raise EraserEvidenceInferenceOfficialHippoRAGError(
            "worker input is unavailable"
        )
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EraserEvidenceInferenceOfficialHippoRAGError(
            "worker input is invalid"
        ) from exc
    if raw != canonical_json_bytes(payload) or not isinstance(payload, Mapping):
        raise EraserEvidenceInferenceOfficialHippoRAGError(
            "worker input is not canonical JSON"
        )
    if set(payload) != {"query", "schema", "sentence_texts"}:
        raise EraserEvidenceInferenceOfficialHippoRAGError(
            "worker input envelope is not exact"
        )
    query = payload.get("query")
    sentence_texts = payload.get("sentence_texts")
    if (
        payload.get("schema") != INPUT_SCHEMA
        or not isinstance(query, str)
        or not isinstance(sentence_texts, list)
        or any(not isinstance(text, str) for text in sentence_texts)
    ):
        raise EraserEvidenceInferenceOfficialHippoRAGError(
            "worker input schema drifted"
        )
    validated_query, validated_sentences = validate_single_item(
        query, sentence_texts
    )
    return validated_query, list(validated_sentences)


def _write_ordinals_only(path: Path, ordinals: Sequence[int]) -> None:
    raw = canonical_json_bytes(list(ordinals))
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except OSError as exc:
        raise EraserEvidenceInferenceOfficialHippoRAGError(
            "ordinal-only output cannot be created exclusively"
        ) from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _build_official_core(
    *,
    save_dir: Path,
    llm_model: Path,
    embedding_model: Path,
) -> object:
    """Thinly reuse the frozen MuSiQue official fresh-core factory."""

    return _build_musique_official_core(
        save_dir=save_dir,
        llm_model=llm_model,
        embedding_model=embedding_model,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--index-root", required=True, type=Path)
    parser.add_argument("--llm-model", required=True, type=Path)
    parser.add_argument("--embedding-model", required=True, type=Path)
    arguments = parser.parse_args(argv)
    if arguments.index_root.exists() or arguments.index_root.is_symlink():
        raise EraserEvidenceInferenceOfficialHippoRAGError(
            "item-local index root must not already exist"
        )
    arguments.index_root.mkdir(mode=0o700)
    query, sentence_texts = _load_input(arguments.input)
    core = _build_official_core(
        save_dir=arguments.index_root,
        llm_model=arguments.llm_model,
        embedding_model=arguments.embedding_model,
    )
    result = retrieve_ordinals_with_core(
        core=core,
        query=query,
        sentence_texts=sentence_texts,
    )
    _write_ordinals_only(arguments.output, result)
    print(
        json.dumps(
            {"retrieval_count": len(result), "status": "passed"},
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["retrieve_ordinals_with_core"]
