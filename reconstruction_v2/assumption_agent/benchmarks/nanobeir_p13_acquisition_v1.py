"""One-shot private HMAC acquisition for the frozen P13 study."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from reconstruction_v2.assumption_agent.benchmarks import (
    nanobeir_p12_completecase_acquisition_v1 as mature,
)
from reconstruction_v2.assumption_agent.benchmarks import (
    nanobeir_p13_availability_v1 as availability,
)


SCHEMA = "nanobeir_p13_acquisition_result_v1"
ATTEMPT_SCHEMA = "nanobeir_p13_acquisition_attempt_v1"
FREEZE_SCHEMA = "nanobeir_p13_acquisition_freeze_v1"
VIEW_SCHEMA = "nanobeir_p13_private_view_v1"
LABEL_SCHEMA = "nanobeir_p13_private_labels_v1"
HIPPO_SCHEMA = "nanobeir_p13_private_hipporag_v1"
FAMILIES = availability.FAMILIES
BLOCK_COUNTS = mature.BLOCK_COUNTS
RESERVE_COUNT = mature.RESERVE_COUNT
SELECTED_PER_FAMILY = mature.SELECTED_PER_FAMILY
LABEL_BLOCKS = mature.LABEL_BLOCKS

SOURCE_ROOT_RELATIVE = availability.SOURCE_ROOT_RELATIVE
AVAILABILITY_ROOT_RELATIVE = availability.RUN_ROOT_RELATIVE
AVAILABILITY_RESULT_RELATIVE = availability.RESULT_RELATIVE
RUN_ROOT_RELATIVE = Path("artifacts/nanobeir_p13_acquisition_v1")
INTERNAL_RESULT_RELATIVE = RUN_ROOT_RELATIVE / "internal.mature.result.json"
RESULT_RELATIVE = Path("manifests/nanobeir_p13_acquisition_result_v1.json")
FREEZE_RELATIVE = Path("manifests/nanobeir_p13_acquisition_freeze_v1.json")
IMPLEMENTATION_RELATIVE = Path(
    "assumption_agent/benchmarks/nanobeir_p13_acquisition_v1.py"
)
TEST_RELATIVE = Path("tests/test_nanobeir_p13_acquisition_v1.py")

PRECONDITIONS = {
    "candidate": availability.PRECONDITIONS["candidate"],
    "design": availability.PRECONDITIONS["design"],
    "source_access": availability.PRECONDITIONS["source_access"],
}

QREL_FILES = {
    "qrels/NanoFiQA2018-00000-of-00001.parquet": (
        "21bc1504b6d5efb1bf78236d08b5b0a81c352fb1aadb02d8959ba94fbf01e8ba"
    ),
    "qrels/NanoNFCorpus-00000-of-00001.parquet": (
        "d97ea8176db52aa04773f2459d02e20c582ed9aa694801201cd21e841a00f200"
    ),
    "qrels/NanoTouche2020-00000-of-00001.parquet": (
        "e1b7500589ed1356a38623d816e05073c5c7e48bcf6e9d263b22741b62940ed5"
    ),
}
SOURCE_FILES = availability.SOURCE_FILES
DOCUMENT_PROJECTION_CHARACTERS = availability.DOCUMENT_PROJECTION_CHARACTERS
DEPENDENCY_RELATIVES = (
    mature.IMPLEMENTATION_RELATIVE,
    availability.IMPLEMENTATION_RELATIVE,
    Path("assumption_agent/benchmarks/nanobeir_p12_acquisition_v1.py"),
    Path("assumption_agent/benchmarks/fiqa_bridge_expansion_train_runtime_v1.py"),
    Path("replication_runtime/bright_official_hipporag_v1/contract.py"),
)

canonical_json_bytes = mature.canonical_json_bytes
stable_hash = mature.stable_hash
file_sha256 = mature.file_sha256
self_hashed = mature.self_hashed
_write_exclusive = mature._write_exclusive
_write_json = mature._write_json
project_document = mature.project_document
verify_self_hash = mature.verify_self_hash
_verify_self_hash = mature._verify_self_hash


class P13AcquisitionError(RuntimeError):
    """The frozen P13 private acquisition failed closed."""


class OneShotRefusal(P13AcquisitionError):
    """The formal P13 acquisition root or result is already consumed."""


NanoBEIRP12AcquisitionError = P13AcquisitionError
NanoBEIRAcquisitionError = P13AcquisitionError


valid_cached_rank_sets = mature.valid_cached_rank_sets


def hmac_order(
    secret: bytes, family: str, query_ids: Sequence[str]
) -> tuple[str, ...]:
    if len(secret) != 32 or family not in FAMILIES:
        raise P13AcquisitionError("HMAC ordering input drifted")
    values = tuple(query_ids)
    if len(values) != len(set(values)):
        raise P13AcquisitionError("HMAC query IDs are duplicated")
    return tuple(
        sorted(
            values,
            key=lambda query_id: (
                hmac.new(
                    secret,
                    (family + "\n" + query_id).encode("utf-8"),
                    hashlib.sha256,
                ).digest(),
                query_id,
            ),
        )
    )


def allocate_blocks(
    secret: bytes, family: str, eligible_query_ids: Sequence[str]
) -> Mapping[str, tuple[str, ...]]:
    ordered = hmac_order(secret, family, eligible_query_ids)
    if len(ordered) < SELECTED_PER_FAMILY:
        raise P13AcquisitionError("eligible family capacity is below 36")
    selected = ordered[:SELECTED_PER_FAMILY]
    output: dict[str, tuple[str, ...]] = {}
    offset = 0
    for block, count in BLOCK_COUNTS:
        output[block] = selected[offset : offset + count]
        offset += count
    output["RESERVE"] = selected[offset : offset + RESERVE_COUNT]
    if offset + RESERVE_COUNT != SELECTED_PER_FAMILY:
        raise P13AcquisitionError("block allocation drifted")
    return output


@contextmanager
def _patched_mature_acquisition() -> Iterator[None]:
    replacements = {
        "SCHEMA": SCHEMA,
        "ATTEMPT_SCHEMA": ATTEMPT_SCHEMA,
        "FREEZE_SCHEMA": FREEZE_SCHEMA,
        "VIEW_SCHEMA": VIEW_SCHEMA,
        "LABEL_SCHEMA": LABEL_SCHEMA,
        "HIPPO_SCHEMA": HIPPO_SCHEMA,
        "FAMILIES": FAMILIES,
        "SOURCE_ROOT_RELATIVE": SOURCE_ROOT_RELATIVE,
        "AVAILABILITY_ROOT_RELATIVE": AVAILABILITY_ROOT_RELATIVE,
        "AVAILABILITY_RESULT_RELATIVE": AVAILABILITY_RESULT_RELATIVE,
        "RUN_ROOT_RELATIVE": RUN_ROOT_RELATIVE,
        "RESULT_RELATIVE": INTERNAL_RESULT_RELATIVE,
        "FREEZE_RELATIVE": FREEZE_RELATIVE,
        "IMPLEMENTATION_RELATIVE": IMPLEMENTATION_RELATIVE,
        "TEST_RELATIVE": TEST_RELATIVE,
        "PRECONDITIONS": PRECONDITIONS,
        "QREL_FILES": QREL_FILES,
        "SOURCE_FILES": SOURCE_FILES,
        "DEPENDENCY_RELATIVES": DEPENDENCY_RELATIVES,
        "availability": availability,
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
        raise OneShotRefusal("P13 acquisition root already exists")
    if result_path.exists() or result_path.is_symlink():
        raise OneShotRefusal("P13 acquisition result already exists")
    with _patched_mature_acquisition():
        internal = mature.run_formal(project_root)
    body = dict(internal)
    body.pop("self_sha256", None)
    body["candidate_name"] = "P13_P12_BRIDGE_SAFE_TYPED_QUERY_V1"
    body["internal_mature_result_file_sha256"] = file_sha256(
        base / INTERNAL_RESULT_RELATIVE
    )
    body["recorded_date"] = "2026-07-21"
    body["status"] = (
        "passed_99_item_private_acquisition_ready_for_P13_C_confirm_runtime"
    )
    result = self_hashed(body)
    _write_exclusive(result_path, canonical_json_bytes(result), mode=0o644)
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
