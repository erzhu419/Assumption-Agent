"""Final one-shot source-free feasibility before any new TAT-QA study.

P23 is not qualification or efficacy evidence.  It reuses the immutable P22
portable MiniLM correction and the P21 Qwen/HippoRAG public path, while
pre-registering the two deterministic OpenMP environment mutations observed
after model initialization.  Those names must have their exact known values,
are sealed in a self-hashed receipt, and are then removed before the original
P21 post-MiniLM and nested-launch contracts are revalidated.

This is the final feasibility attempt on this route.  The entry point has no
formal-source or attempt-root override.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
from typing import Iterator, Mapping, Sequence

from assumption_agent.benchmarks import (
    tatqa_p22_source_free_feasibility_v1 as base,
)


VERSION = "tatqa_p23_source_free_feasibility_v1"
MARKER_FILENAME = base.MARKER_FILENAME
SUCCESS_FILENAME = base.SUCCESS_FILENAME
FAILURE_FILENAME = base.FAILURE_FILENAME
PORTABLE_CAPABILITY_SCHEMA = (
    "tatqa_p23_portable_minilm_capability_receipt_snapshot_v1"
)
ENTRY_MODULE_NAME = (
    "assumption_agent.benchmarks.tatqa_p23_source_free_feasibility_v1"
)
NORMALIZER_CALLABLE_NAME = "_normalize_post_minilm_environment"
ENTRY_RELATIVE_PATH = (
    "assumption_agent/benchmarks/tatqa_p23_source_free_feasibility_v1.py"
)
TEST_RELATIVE_PATH = "tests/test_tatqa_p23_source_free_feasibility_v1.py"
P23_ADDITIONAL_IMPLEMENTATION_PATHS = frozenset(
    {ENTRY_RELATIVE_PATH, TEST_RELATIVE_PATH}
)
REQUIRED_IMPLEMENTATION_PATHS = (
    base.REQUIRED_IMPLEMENTATION_PATHS | P23_ADDITIONAL_IMPLEMENTATION_PATHS
)
REQUIRED_EVIDENCE_PATHS = base.REQUIRED_EVIDENCE_PATHS
SOURCE_ISOLATION_ROOTS = (
    "artifacts/tatqa_p21_official_source_v1",
    "artifacts/tatqa_p21_formal_v1",
    "artifacts/tatqa_p22_official_source_v1",
    "artifacts/tatqa_p22_formal_v1",
    "artifacts/tatqa_p23_official_source_v1",
    "artifacts/tatqa_p23_formal_v1",
)
SOURCE_ISOLATION_SENTINEL_NAME = ".p23-source-free-isolation-sentinel"
SOURCE_ISOLATION_SENTINEL_BYTES = b"P23 SOURCE-FREE ISOLATION SENTINEL V1\n"
SOURCE_ISOLATION_SENTINEL_PATHS = frozenset(
    f"{root}/{SOURCE_ISOLATION_SENTINEL_NAME}"
    for root in SOURCE_ISOLATION_ROOTS
)
REQUIRED_SNAPSHOT_PATHS = (
    REQUIRED_IMPLEMENTATION_PATHS
    | REQUIRED_EVIDENCE_PATHS
    | SOURCE_ISOLATION_SENTINEL_PATHS
)
EXPECTED_HOST_ROOT = Path("/home/erzhu419/p23_source_free_feasibility_20260723")
EXPECTED_PROJECT_ROOT = EXPECTED_HOST_ROOT / "runtime/reconstruction_v2"
EXPECTED_FEASIBILITY_ROOT = EXPECTED_HOST_ROOT / "attempt"
EXPECTED_WORK_ROOT = EXPECTED_FEASIBILITY_ROOT / "work"
EXPECTED_PUBLIC_CANARY_OUTPUT = EXPECTED_FEASIBILITY_ROOT / "public-canary.json"
EXPECTED_OUTER_UNIT = "p23-source-free-feasibility-c1-v1.service"
GIT_EXECUTABLE = Path("/home/erzhu419/p23_runtime_tools_20260723/git/bin/git")
GIT_EXECUTABLE_SHA256 = base.GIT_EXECUTABLE_SHA256
GIT_VERSION_STDOUT = base.GIT_VERSION_STDOUT
DETERMINISTIC_POST_MODEL_MUTATIONS = (
    ("KMP_DUPLICATE_LIB_OK", "True"),
    ("KMP_INIT_AT_FORK", "FALSE"),
)
NORMALIZATION_SCHEMA = (
    "tatqa_p23_post_minilm_environment_normalization_receipt_v1"
)


class TatqaP23SourceFreeFeasibilityError(RuntimeError):
    """The final independent source-free feasibility failed closed."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise TatqaP23SourceFreeFeasibilityError(
            "P23 receipt is not canonical JSON"
        ) from exc


def _semantic_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value).rstrip(b"\n")).hexdigest()


def _normalize_post_minilm_environment() -> dict[str, object]:
    """Accept exactly two known model-init mutations, seal, then remove them."""

    inherited_names = set(
        base.formal_runtime.USER_SYSTEMD_POST_RUNTIME_ENVIRONMENT_VARIABLE_ALLOWLIST
    )
    mutation_names = {name for name, _value in DETERMINISTIC_POST_MODEL_MUTATIONS}
    expected_names = inherited_names | mutation_names
    if set(os.environ) != expected_names:
        raise TatqaP23SourceFreeFeasibilityError(
            "post-MiniLM environment variable-name set drifted"
        )
    for key, value in base.formal_runtime.USER_SYSTEMD_ENTRY_SAFE_VALUES.items():
        expected = "" if key == "CUDA_VISIBLE_DEVICES" else value
        if os.environ.get(key) != expected:
            raise TatqaP23SourceFreeFeasibilityError(
                "post-MiniLM inherited safe value drifted"
            )
    if os.environ.get("CUDA_MODULE_LOADING") != "LAZY":
        raise TatqaP23SourceFreeFeasibilityError(
            "post-MiniLM CUDA module-loading value drifted"
        )
    for name, expected in DETERMINISTIC_POST_MODEL_MUTATIONS:
        if os.environ.get(name) != expected:
            raise TatqaP23SourceFreeFeasibilityError(
                "deterministic post-model environment mutation drifted"
            )

    observed_names_sha256 = _semantic_hash(sorted(expected_names))
    for name, _expected in DETERMINISTIC_POST_MODEL_MUTATIONS:
        del os.environ[name]
    if set(os.environ) != inherited_names:
        raise TatqaP23SourceFreeFeasibilityError(
            "post-MiniLM environment normalization did not reach P21 closure"
        )

    body: dict[str, object] = {
        "schema": NORMALIZATION_SCHEMA,
        "status": "verified_exact_deterministic_mutations_then_removed",
        "observed_variable_name_allowlist": sorted(expected_names),
        "observed_variable_name_set_sha256": observed_names_sha256,
        "accepted_exact_mutations": [
            {"added_name": name, "exact_value": value}
            for name, value in DETERMINISTIC_POST_MODEL_MUTATIONS
        ],
        "removed_before_nested_launch": [
            name for name, _value in DETERMINISTIC_POST_MODEL_MUTATIONS
        ],
        "normalized_variable_name_allowlist": sorted(inherited_names),
        "all_inherited_P21_safe_values_exact_before_normalization": True,
        "P21_post_minilm_phase_revalidation_required_after_normalization": True,
        "raw_environment_values_or_credentials_recorded": False,
        "provider_or_api_credentials_read": False,
        "formal_TAT_QA_source_or_rows_accessed": False,
        "external_network_calls": 0,
        "api_or_online_evaluator_calls": 0,
    }
    return {**body, "self_sha256": _semantic_hash(body)}


def _contract_bindings() -> dict[str, object]:
    return {
        "VERSION": VERSION,
        "PORTABLE_CAPABILITY_SCHEMA": PORTABLE_CAPABILITY_SCHEMA,
        "REQUIRED_IMPLEMENTATION_PATHS": REQUIRED_IMPLEMENTATION_PATHS,
        "REQUIRED_EVIDENCE_PATHS": REQUIRED_EVIDENCE_PATHS,
        "REQUIRED_SNAPSHOT_PATHS": REQUIRED_SNAPSHOT_PATHS,
        "REQUIRED_ENTRY_MODULE_NAME": ENTRY_MODULE_NAME,
        "EXPECTED_NORMALIZER_CALLABLE_NAME": NORMALIZER_CALLABLE_NAME,
        "SOURCE_ISOLATION_ROOTS": SOURCE_ISOLATION_ROOTS,
        "SOURCE_ISOLATION_SENTINEL_NAME": SOURCE_ISOLATION_SENTINEL_NAME,
        "SOURCE_ISOLATION_SENTINEL_BYTES": SOURCE_ISOLATION_SENTINEL_BYTES,
        "SOURCE_ISOLATION_SENTINEL_PATHS": SOURCE_ISOLATION_SENTINEL_PATHS,
        "EXPECTED_HOST_ROOT": EXPECTED_HOST_ROOT,
        "EXPECTED_PROJECT_ROOT": EXPECTED_PROJECT_ROOT,
        "EXPECTED_FEASIBILITY_ROOT": EXPECTED_FEASIBILITY_ROOT,
        "EXPECTED_WORK_ROOT": EXPECTED_WORK_ROOT,
        "EXPECTED_PUBLIC_CANARY_OUTPUT": EXPECTED_PUBLIC_CANARY_OUTPUT,
        "EXPECTED_OUTER_UNIT": EXPECTED_OUTER_UNIT,
        "GIT_EXECUTABLE": GIT_EXECUTABLE,
        "GIT_EXECUTABLE_SHA256": GIT_EXECUTABLE_SHA256,
        "GIT_VERSION_STDOUT": GIT_VERSION_STDOUT,
    }


@contextmanager
def _activate_p23_contract() -> Iterator[None]:
    """Activate one closed P23 contract in the inherited single-process engine."""

    bindings = _contract_bindings()
    originals = {name: getattr(base, name) for name in bindings}
    try:
        for name, value in bindings.items():
            setattr(base, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(base, name, value)


def run_source_free_feasibility(
    *,
    project_root: str | Path,
    typed_runtime_python: str | Path,
    hippo_runtime_python: str | Path,
    qwen_model: str | Path,
    minilm_asset_manifest: str | Path,
    minilm_model: str | Path,
    hippo_llm_model: str | Path,
    hippo_embedding_model: str | Path,
    hipporag_source: str | Path,
    hippo_attestation: str | Path,
    p21_runtime_fingerprint: str | Path,
    diagnostic_snapshot_commit: str,
) -> dict[str, object]:
    """Consume the final independent public-synthetic feasibility attempt."""

    with _activate_p23_contract():
        try:
            return base.run_source_free_feasibility(
                project_root=project_root,
                typed_runtime_python=typed_runtime_python,
                hippo_runtime_python=hippo_runtime_python,
                qwen_model=qwen_model,
                minilm_asset_manifest=minilm_asset_manifest,
                minilm_model=minilm_model,
                hippo_llm_model=hippo_llm_model,
                hippo_embedding_model=hippo_embedding_model,
                hipporag_source=hipporag_source,
                hippo_attestation=hippo_attestation,
                p21_runtime_fingerprint=p21_runtime_fingerprint,
                diagnostic_snapshot_commit=diagnostic_snapshot_commit,
                _post_minilm_environment_normalizer=(
                    _normalize_post_minilm_environment
                ),
            )
        except base.TatqaP22SourceFreeFeasibilityError as exc:
            raise TatqaP23SourceFreeFeasibilityError(
                "P23 source-free feasibility failed terminally"
            ) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--typed-runtime-python", required=True, type=Path)
    parser.add_argument("--hippo-runtime-python", required=True, type=Path)
    parser.add_argument("--qwen-model", required=True, type=Path)
    parser.add_argument("--minilm-asset-manifest", required=True, type=Path)
    parser.add_argument("--minilm-model", required=True, type=Path)
    parser.add_argument("--hippo-llm-model", required=True, type=Path)
    parser.add_argument("--hippo-embedding-model", required=True, type=Path)
    parser.add_argument("--hipporag-source", required=True, type=Path)
    parser.add_argument("--hippo-attestation", required=True, type=Path)
    parser.add_argument("--p21-runtime-fingerprint", required=True, type=Path)
    parser.add_argument("--diagnostic-snapshot-commit", required=True)
    return parser


def _main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_source_free_feasibility(
        project_root=args.project_root,
        typed_runtime_python=args.typed_runtime_python,
        hippo_runtime_python=args.hippo_runtime_python,
        qwen_model=args.qwen_model,
        minilm_asset_manifest=args.minilm_asset_manifest,
        minilm_model=args.minilm_model,
        hippo_llm_model=args.hippo_llm_model,
        hippo_embedding_model=args.hippo_embedding_model,
        hipporag_source=args.hipporag_source,
        hippo_attestation=args.hippo_attestation,
        p21_runtime_fingerprint=args.p21_runtime_fingerprint,
        diagnostic_snapshot_commit=args.diagnostic_snapshot_commit,
    )
    print(_canonical_bytes(result).decode("ascii"), end="")
    return 0


def main() -> int:
    return _main()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FAILURE_FILENAME",
    "MARKER_FILENAME",
    "SUCCESS_FILENAME",
    "TatqaP23SourceFreeFeasibilityError",
    "VERSION",
    "main",
    "run_source_free_feasibility",
]
