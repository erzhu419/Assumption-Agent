"""Single label-free HippoRAG availability screen for the P13 source epoch."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from reconstruction_v2.assumption_agent.benchmarks import (
    nanobeir_p12_completecase_availability_v1 as mature,
)
from reconstruction_v2.assumption_agent.benchmarks import (
    nanobeir_p12_acquisition_v1 as utilities,
)
from reconstruction_v2.assumption_agent.benchmarks import (
    fiqa_bridge_expansion_train_runtime_v1 as train,
)
from reconstruction_v2.assumption_agent.benchmarks import (
    bright_reasoning_retrieval_core_v1 as core,
)


SCHEMA = "nanobeir_p13_availability_result_v1"
ATTEMPT_SCHEMA = "nanobeir_p13_availability_attempt_v1"
PACK_SCHEMA = "nanobeir_p13_availability_private_pack_v1"
FREEZE_SCHEMA = "nanobeir_p13_availability_freeze_v1"
FAMILIES = ("NanoFiQA2018", "NanoNFCorpus", "NanoTouche2020")
FAMILY_QUERY_COUNTS = {
    "NanoFiQA2018": 50,
    "NanoNFCorpus": 50,
    "NanoTouche2020": 49,
}
ITEM_COUNT = sum(FAMILY_QUERY_COUNTS.values())
MINIMUM_ELIGIBLE_PER_FAMILY = 36
PROCESS_CONCURRENCY = 12
DOCUMENT_PROJECTION_CHARACTERS = 3000

SOURCE_ROOT_RELATIVE = Path("artifacts/nanobeir_p13_source_v1/dataset")
RUN_ROOT_RELATIVE = Path("artifacts/nanobeir_p13_availability_v1")
RESULT_RELATIVE = Path("manifests/nanobeir_p13_availability_result_v1.json")
FREEZE_RELATIVE = Path("manifests/nanobeir_p13_availability_freeze_v1.json")
IMPLEMENTATION_RELATIVE = Path(
    "assumption_agent/benchmarks/nanobeir_p13_availability_v1.py"
)
TEST_RELATIVE = Path("tests/test_nanobeir_p13_availability_v1.py")

PRECONDITIONS = {
    "candidate": {
        "relative": "manifests/nanobeir_p13_candidate_freeze_v1.json",
        "file_sha256": "64482d0e0d4647327da0a74f1e14844854291f73f439928788eeeba7e7c6a1b2",
        "self_sha256": "17f9865483cd3c4846db8a63c1047f8af6bdaa24b78ece09245f3e568e0457f0",
    },
    "design": {
        "relative": "manifests/nanobeir_p13_study_design_v2.json",
        "file_sha256": "a4c6bd7ca3e7fc7dfc85bb37dd9b1dad2167a302a12a07d2712b6d18e33dbc32",
        "self_sha256": "7d7230e3af8b1cc906e494851c52dc1dcc9ca04b4f247eeee7a3248494fb4e08",
    },
    "hardening": mature.PRECONDITIONS["hardening"],
    "source_access": {
        "relative": "manifests/nanobeir_p13_source_access_v2.json",
        "file_sha256": "887a202390d5e7e4e31a9e2756d56350b0e173e49783fbed0c57bc8a8c3037db",
        "self_sha256": "ad22b5c3cd8eb46208ca7d93ce5f6dac92812b2ca2ac69828a62b7b58c470eab",
    },
}

SOURCE_FILES = {
    "corpus/NanoFiQA2018-00000-of-00001.parquet": (
        "e6ff315a9fbd61e70c0f320c9ef04602b3d71794ee3ed422ea232ede87d314a0"
    ),
    "corpus/NanoNFCorpus-00000-of-00001.parquet": (
        "d50e7ac973d4367434b68c1e7eb54d7827b29d85aa54a1dde42883f05fbf7d95"
    ),
    "corpus/NanoTouche2020-00000-of-00001.parquet": (
        "5ae883dbf2cb6573672722741ad7b34761346bf99c2adaf85727ea69eb86b146"
    ),
    "queries/NanoFiQA2018-00000-of-00001.parquet": (
        "0529ff05670678d9896ccb60d45c35ce730d1dc6d6fed522e66fdec7d1291a92"
    ),
    "queries/NanoNFCorpus-00000-of-00001.parquet": (
        "e9a58c2e1f392a83b26eade3d9838f7448c8a6cdb34f7257f3475cb76024aec2"
    ),
    "queries/NanoTouche2020-00000-of-00001.parquet": (
        "d1673ebd1175a1b135e8641092e2ad87b5e76a526853d1f75e3c1c16fc621cd3"
    ),
}

REQUIRED_IMPLEMENTATION_RELATIVES = (
    IMPLEMENTATION_RELATIVE,
    TEST_RELATIVE,
    mature.IMPLEMENTATION_RELATIVE,
    *mature.REQUIRED_IMPLEMENTATION_RELATIVES[2:],
)


class P13AvailabilityError(RuntimeError):
    """The frozen P13 availability screen failed closed."""


class OneShotRefusal(P13AvailabilityError):
    """The formal P13 screen root or result is already consumed."""


def _verify_preconditions(base: Path) -> Mapping[str, Any]:
    loaded: dict[str, Any] = {}
    for name, binding in PRECONDITIONS.items():
        path = base / binding["relative"]
        if utilities.file_sha256(path) != binding["file_sha256"]:
            raise P13AvailabilityError(f"{name} manifest file drifted")
        value = mature._read_json(path, name)
        mature._verify_self(value, binding["self_sha256"])
        loaded[name] = value
    if loaded["source_access"].get("qualification", {}).get("source_passed") is not True:
        raise P13AvailabilityError("source qualification did not pass")
    if loaded["hardening"].get("status") != (
        "passed_upstream_fixed_comparator_qualified_for_future_new_studies_only"
    ):
        raise P13AvailabilityError("upstream hardening is not qualified")
    for relative, expected in SOURCE_FILES.items():
        path = base / SOURCE_ROOT_RELATIVE / relative
        if (
            path.is_symlink()
            or not path.is_file()
            or utilities.file_sha256(path) != expected
        ):
            raise P13AvailabilityError("pinned source file drifted")
    return loaded


def load_sources(base: Path) -> Mapping[str, mature.SourceFamily]:
    """Read corpus/query members only and apply the frozen shared empty filter."""

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise P13AvailabilityError("pyarrow is unavailable") from exc
    root = base / SOURCE_ROOT_RELATIVE
    output: dict[str, mature.SourceFamily] = {}
    excluded_counts: dict[str, int] = {}
    for family in FAMILIES:
        corpus = pq.read_table(
            root / "corpus" / f"{family}-00000-of-00001.parquet"
        )
        queries = pq.read_table(
            root / "queries" / f"{family}-00000-of-00001.parquet"
        )
        if corpus.column_names != ["_id", "text"]:
            raise P13AvailabilityError("corpus schema drifted")
        if queries.column_names != ["_id", "text"]:
            raise P13AvailabilityError("query schema drifted")
        ids: list[str] = []
        contents: list[str] = []
        excluded = 0
        for row in corpus.to_pylist():
            identifier = mature._required_text(row.get("_id"), "corpus ID")
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                excluded += 1
                continue
            if identifier in ids or "\x00" in text:
                raise P13AvailabilityError("corpus identity or text drifted")
            ids.append(identifier)
            contents.append(text[:DOCUMENT_PROJECTION_CHARACTERS])
        query_ids: list[str] = []
        query_texts: list[str] = []
        for row in queries.to_pylist():
            identifier = mature._required_text(row.get("_id"), "query ID")
            text = mature._required_text(row.get("text"), "query text")
            if identifier in query_ids or text in query_texts:
                raise P13AvailabilityError("query identity drifted")
            query_ids.append(identifier)
            query_texts.append(text)
        if (
            len(ids) < 32
            or len(ids) != len(set(ids))
            or len(query_ids) != FAMILY_QUERY_COUNTS[family]
        ):
            raise P13AvailabilityError("source family capacity drifted")
        excluded_counts[family] = excluded
        output[family] = mature.SourceFamily(
            tuple(ids), tuple(contents), tuple(query_ids), tuple(query_texts)
        )
    if excluded_counts != {
        "NanoFiQA2018": 27,
        "NanoNFCorpus": 0,
        "NanoTouche2020": 0,
    }:
        raise P13AvailabilityError("shared source filter drifted")
    return output


def build_screen_items(
    sources: Mapping[str, mature.SourceFamily],
    corpus_embeddings: Mapping[str, np.ndarray],
    query_embeddings: Mapping[str, np.ndarray],
) -> tuple[mature.ScreenItem, ...]:
    items: list[mature.ScreenItem] = []
    ordinal = 0
    for family in FAMILIES:
        source = sources[family]
        corpus_matrix = np.asarray(corpus_embeddings[family], dtype=np.float32)
        query_matrix = np.asarray(query_embeddings[family], dtype=np.float32)
        if corpus_matrix.shape != (len(source.ids), 384):
            raise P13AvailabilityError("corpus embedding shape drifted")
        if query_matrix.shape != (FAMILY_QUERY_COUNTS[family], 384):
            raise P13AvailabilityError("query embedding shape drifted")
        if not np.isfinite(corpus_matrix).all() or not np.isfinite(
            query_matrix
        ).all():
            raise P13AvailabilityError("embedding contains a nonfinite value")
        for family_ordinal, (query_id, query) in enumerate(
            zip(source.query_ids, source.queries)
        ):
            scores = train.quantized_scores(
                corpus_matrix, query_matrix[family_ordinal]
            )
            try:
                local = core.build_local_retrieval([scores])
            except core.BrightStudyCoreError as exc:
                raise P13AvailabilityError(str(exc)) from exc
            items.append(
                mature.ScreenItem(
                    ordinal=ordinal,
                    family=family,
                    family_ordinal=family_ordinal,
                    query_id=query_id,
                    query=query,
                    base_pool=local.candidate_rows,
                    raw_top10=local.raw_rows,
                )
            )
            ordinal += 1
    if len(items) != ITEM_COUNT:
        raise P13AvailabilityError("screen item count drifted")
    return tuple(items)


@contextmanager
def _patched_mature_screen() -> Iterator[None]:
    replacements = {
        "SCHEMA": SCHEMA,
        "ATTEMPT_SCHEMA": ATTEMPT_SCHEMA,
        "PACK_SCHEMA": PACK_SCHEMA,
        "FREEZE_SCHEMA": FREEZE_SCHEMA,
        "FAMILIES": FAMILIES,
        "ITEM_COUNT": ITEM_COUNT,
        "MINIMUM_ELIGIBLE_PER_FAMILY": MINIMUM_ELIGIBLE_PER_FAMILY,
        "PROCESS_CONCURRENCY": PROCESS_CONCURRENCY,
        "SOURCE_ROOT_RELATIVE": SOURCE_ROOT_RELATIVE,
        "RUN_ROOT_RELATIVE": RUN_ROOT_RELATIVE,
        "RESULT_RELATIVE": RESULT_RELATIVE,
        "FREEZE_RELATIVE": FREEZE_RELATIVE,
        "IMPLEMENTATION_RELATIVE": IMPLEMENTATION_RELATIVE,
        "TEST_RELATIVE": TEST_RELATIVE,
        "PRECONDITIONS": PRECONDITIONS,
        "SOURCE_FILES": SOURCE_FILES,
        "REQUIRED_IMPLEMENTATION_RELATIVES": REQUIRED_IMPLEMENTATION_RELATIVES,
        "load_sources": load_sources,
        "build_screen_items": build_screen_items,
        "_verify_preconditions": _verify_preconditions,
    }
    originals = {name: getattr(mature, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(mature, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(mature, name, value)


def run_formal(project_root: Path) -> Mapping[str, Any]:
    project_root = project_root.resolve(strict=True)
    base = project_root / "reconstruction_v2"
    root = base / RUN_ROOT_RELATIVE
    result_path = base / RESULT_RELATIVE
    if root.exists() or root.is_symlink():
        raise OneShotRefusal("P13 availability root already exists")
    if result_path.exists() or result_path.is_symlink():
        raise OneShotRefusal("P13 availability result already exists")
    with _patched_mature_screen():
        return mature.run_formal(project_root)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--formal", action="store_true")
    arguments = parser.parse_args(argv)
    if not arguments.formal:
        raise SystemExit("--formal is required")
    result = run_formal(arguments.project_root)
    print(
        json.dumps(
            {
                "eligibility_passed": result["eligibility_passed"],
                "self_sha256": result["self_sha256"],
                "status": result["status"],
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
