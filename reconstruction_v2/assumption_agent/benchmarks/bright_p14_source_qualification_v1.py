"""Offline structural qualification for the frozen fresh P14 BRIGHT source."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from reconstruction_v2.assumption_agent.benchmarks import (
    nanobeir_p13_acquisition_v1 as utilities,
)


SCHEMA = "bright_p14_source_qualification_result_v1"
ATTEMPT_SCHEMA = "bright_p14_source_qualification_attempt_v1"
FAMILIES = ("EARTH_SCIENCE", "PSYCHOLOGY", "SUSTAINABLE_LIVING")
SLUGS = {
    "EARTH_SCIENCE": "earth_science",
    "PSYCHOLOGY": "psychology",
    "SUSTAINABLE_LIVING": "sustainable_living",
}
MINIMUM_QUERY_COUNT = 72
DOCUMENT_CHARACTER_CAP = 3000

SOURCE_ROOT_RELATIVE = Path("artifacts/bright_p14_source_v1/dataset")
RUN_ROOT_RELATIVE = Path("artifacts/bright_p14_source_qualification_v1")
RESULT_RELATIVE = Path(
    "manifests/bright_p14_source_qualification_result_v1.json"
)
IMPLEMENTATION_RELATIVE = Path(
    "assumption_agent/benchmarks/bright_p14_source_qualification_v1.py"
)
TEST_RELATIVE = Path("tests/test_bright_p14_source_qualification_v1.py")

PRECONDITIONS = {
    "custody": {
        "relative": "manifests/bright_p14_source_custody_v1.json",
        "file_sha256": (
            "b0648fbae04a62f1f096238c36dcb7ff79b91140b4702d57969c672508e47484"
        ),
        "self_sha256": (
            "8c864e399b524122c12e2acfc21a790389a39f0022f27f3091e73138328f0566"
        ),
    },
    "design": {
        "relative": (
            "manifests/bright_p14_direct_completecase_study_design_v1.json"
        ),
        "file_sha256": (
            "4386e158c8b51fd24e924a32f573a7d717fad5567c28eefebbd0ae0109ca92bf"
        ),
        "self_sha256": (
            "8a4492ec320adb308174c9a26d6b380105e298d8dfccd2a495cfb0fadb9c33c5"
        ),
    },
}

SOURCE_FILES = {
    "documents/earth_science-00000-of-00001.parquet": {
        "sha256": "abcb2cd130d7d333b364bf5c89b7ff3829e0f88eb6a2de8232c3df1173eeb8a2",
        "size_bytes": 23084671,
    },
    "examples/earth_science-00000-of-00001.parquet": {
        "sha256": "5d29f108701111984eb91c93d3e340a784a99df47dc11f43783d1b994010a91d",
        "size_bytes": 184093,
    },
    "documents/psychology-00000-of-00001.parquet": {
        "sha256": "085d381739cb24b4227dfaf577f39d0adcad8b7b1ae74be028ac239d37be3c1d",
        "size_bytes": 11430533,
    },
    "examples/psychology-00000-of-00001.parquet": {
        "sha256": "404e7dff2a4528419df0bdc162541e92138e35b78918d82d3a04ade5b8f7876b",
        "size_bytes": 183889,
    },
    "documents/sustainable_living-00000-of-00001.parquet": {
        "sha256": "474628623cf9de252bd80a7d1b667aa5070e21b87e1dd33f6723db4d24121fdf",
        "size_bytes": 11720059,
    },
    "examples/sustainable_living-00000-of-00001.parquet": {
        "sha256": "61f97837a16b47a0d9953039cf0b6a53d0fc5deae96a34f839b7cb5e798eb117",
        "size_bytes": 218151,
    },
}


class P14SourceError(RuntimeError):
    """The frozen P14 source is unavailable or structurally invalid."""


class OneShotRefusal(P14SourceError):
    """The formal P14 source qualification was already consumed."""


@dataclass(frozen=True)
class SourceExample:
    item_id: str
    query: str
    excluded_ids: tuple[str, ...]
    gold_ids: tuple[str, ...]


@dataclass(frozen=True)
class SourceFamily:
    document_ids: tuple[str, ...]
    document_contents: tuple[str, ...]
    examples: tuple[SourceExample, ...]
    raw_document_count: int
    filtered_document_count: int
    over_cap_document_count: int


def _required_text(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\x00" in value
    ):
        raise P14SourceError(f"{name} is invalid")
    return value


def _text_list(value: object, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise P14SourceError(f"{name} is invalid")
    output = tuple(_required_text(item, name) for item in value)
    if len(output) != len(set(output)):
        raise P14SourceError(f"{name} is duplicated")
    return output


def _read_json(path: Path, name: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise P14SourceError(f"{name} is unavailable")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise P14SourceError(f"{name} is invalid") from exc
    if not isinstance(value, Mapping):
        raise P14SourceError(f"{name} is not an object")
    return value


def _verify_self(value: Mapping[str, Any], expected: str, name: str) -> None:
    body = dict(value)
    declared = body.pop("self_sha256", None)
    if declared != expected or utilities.stable_hash(body) != expected:
        raise P14SourceError(f"{name} self hash drifted")


def _verify_preconditions(base: Path) -> None:
    for name, binding in PRECONDITIONS.items():
        path = base / binding["relative"]
        if utilities.file_sha256(path) != binding["file_sha256"]:
            raise P14SourceError(f"{name} file drifted")
        _verify_self(_read_json(path, name), binding["self_sha256"], name)
    for relative, binding in SOURCE_FILES.items():
        path = base / SOURCE_ROOT_RELATIVE / relative
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != binding["size_bytes"]
            or utilities.file_sha256(path) != binding["sha256"]
        ):
            raise P14SourceError("source payload drifted")


def load_sources(base: Path) -> Mapping[str, SourceFamily]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise P14SourceError("pyarrow is unavailable") from exc
    root = base / SOURCE_ROOT_RELATIVE
    output: dict[str, SourceFamily] = {}
    for family in FAMILIES:
        slug = SLUGS[family]
        documents = pq.read_table(
            root / "documents" / f"{slug}-00000-of-00001.parquet"
        )
        examples = pq.read_table(
            root / "examples" / f"{slug}-00000-of-00001.parquet",
            columns=["query", "id", "excluded_ids", "gold_ids"],
        )
        if documents.column_names != ["id", "content"]:
            raise P14SourceError("document schema drifted")

        document_ids: list[str] = []
        document_contents: list[str] = []
        filtered_ids: set[str] = set()
        all_ids: set[str] = set()
        over_cap = 0
        for row in documents.to_pylist():
            identifier = _required_text(row.get("id"), "document ID")
            if identifier in all_ids:
                raise P14SourceError("document ID is duplicated")
            all_ids.add(identifier)
            content = row.get("content")
            if not isinstance(content, str) or not content.strip():
                filtered_ids.add(identifier)
                continue
            if "\x00" in content:
                raise P14SourceError("document content contains NUL")
            over_cap += len(content) > DOCUMENT_CHARACTER_CAP
            document_ids.append(identifier)
            document_contents.append(content[:DOCUMENT_CHARACTER_CAP])

        source_examples: list[SourceExample] = []
        seen_ids: set[str] = set()
        seen_queries: set[str] = set()
        for row in examples.to_pylist():
            item_id = _required_text(row.get("id"), "example ID")
            query = _required_text(row.get("query"), "query")
            excluded = _text_list(row.get("excluded_ids"), "excluded IDs")
            gold = _text_list(row.get("gold_ids"), "gold IDs")
            if not gold:
                raise P14SourceError("gold IDs are empty")
            if item_id in seen_ids or query in seen_queries:
                raise P14SourceError("example identity is duplicated")
            if any(identifier not in all_ids for identifier in gold):
                raise P14SourceError("gold ID is absent from source corpus")
            if any(identifier in filtered_ids for identifier in gold):
                raise P14SourceError("gold ID points to a filtered document")
            if set(excluded).intersection(gold):
                raise P14SourceError("excluded and gold IDs overlap")
            seen_ids.add(item_id)
            seen_queries.add(query)
            source_examples.append(
                SourceExample(item_id, query, excluded, gold)
            )
        if len(source_examples) < MINIMUM_QUERY_COUNT:
            raise P14SourceError("family query capacity is below 72")
        output[family] = SourceFamily(
            tuple(document_ids),
            tuple(document_contents),
            tuple(source_examples),
            documents.num_rows,
            len(filtered_ids),
            over_cap,
        )
    return output


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    payload = utilities.canonical_json_bytes(value) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _git_head(project_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def run_formal(project_root: Path) -> Mapping[str, Any]:
    project_root = project_root.resolve(strict=True)
    base = project_root / "reconstruction_v2"
    root = base / RUN_ROOT_RELATIVE
    result_path = base / RESULT_RELATIVE
    if root.exists() or root.is_symlink():
        raise OneShotRefusal("source qualification root already exists")
    if result_path.exists() or result_path.is_symlink():
        raise OneShotRefusal("source qualification result already exists")
    _verify_preconditions(base)
    root.mkdir(mode=0o700)
    marker = utilities.self_hashed(
        {
            "custody_self_sha256": PRECONDITIONS["custody"]["self_sha256"],
            "design_self_sha256": PRECONDITIONS["design"]["self_sha256"],
            "schema": ATTEMPT_SCHEMA,
        },
        field="attempt_sha256",
    )
    marker_path = root / "attempt.marker"
    utilities._write_json(marker_path, marker)
    sources = load_sources(base)
    aggregates: dict[str, Any] = {}
    for family in FAMILIES:
        source = sources[family]
        document_id_set = set(source.document_ids)
        present_excluded = sum(
            identifier in document_id_set
            for item in source.examples
            for identifier in item.excluded_ids
        )
        aggregates[family] = {
            "document_count_after_filter": len(source.document_ids),
            "document_count_before_filter": source.raw_document_count,
            "documents_over_3000_character_cap": (
                source.over_cap_document_count
            ),
            "filtered_document_count": source.filtered_document_count,
            "gold_reference_count": sum(
                len(item.gold_ids) for item in source.examples
            ),
            "maximum_query_characters": max(
                len(item.query) for item in source.examples
            ),
            "present_excluded_ID_count": present_excluded,
            "query_count": len(source.examples),
        }
    result = utilities.self_hashed(
        {
            "claim_boundary": {
                "action_evaluator_model_or_score_count": 0,
                "document_content_read_for_filter_and_projection_audit": True,
                "gold_ID_read_for_referential_integrity_only": True,
                "item_assignment_or_selection_count": 0,
                "network_call_count": 0,
                "reasoning_gold_answer_or_gold_ids_long_read": False,
                "selection_secret_created": False,
            },
            "family_aggregates": aggregates,
            "formal_binding": {
                "attempt_marker_sha256": utilities.file_sha256(marker_path),
                "formal_execution_commit": _git_head(project_root),
                "source_custody_self_sha256": PRECONDITIONS["custody"][
                    "self_sha256"
                ],
                "study_design_self_sha256": PRECONDITIONS["design"][
                    "self_sha256"
                ],
            },
            "qualification": {
                "all_families_have_at_least_72_valid_queries": True,
                "all_gold_IDs_reference_retained_documents": True,
                "shared_document_projection": (
                    "retain_nonempty_then_first_3000_Unicode_codepoints"
                ),
                "source_passed": True,
            },
            "recorded_date": "2026-07-21",
            "schema": SCHEMA,
            "status": "passed_source_ready_for_private_HMAC_acquisition",
        }
    )
    _write_exclusive(result_path, result)
    return result


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
