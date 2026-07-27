#!/usr/bin/env python3
"""One-shot CLI for the consumed HybridQA marginal-replacement diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import traceback


PROJECT_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_PACKAGE_ROOT))

from assumption_agent.benchmarks import (  # noqa: E402
    hybridqa_marginal_replacement_meta_development_v1 as diagnostic,
)


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        + b"\n"
    )


def write_exclusive(path: Path, value: object, *, mode: int = 0o600) -> str:
    raw = canonical_bytes(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, mode)
    try:
        written = 0
        while written < len(raw):
            written += os.write(descriptor, raw[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(raw).hexdigest()


def load_freeze(path: Path) -> tuple[dict[str, object], str]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if (
        not isinstance(value, dict)
        or raw != canonical_bytes(value)
        or value.get("schema")
        != "hybridqa_marginal_replacement_meta_development_freeze_v1"
    ):
        raise diagnostic.HybridQaMarginalMetaError("freeze envelope drifted")
    self_sha = diagnostic.verify_self_hash(value, "self_sha256")
    return value, self_sha


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_frozen_implementation(freeze: dict[str, object]) -> None:
    required = freeze.get("required_file_sha256s")
    runtime_resolved = freeze.get("runtime_python_resolved")
    runtime_sha = freeze.get("runtime_python_sha256")
    observed_runtime = Path(sys.executable).resolve(strict=True)
    if (
        not isinstance(required, dict)
        or not required
        or not isinstance(runtime_resolved, str)
        or not isinstance(runtime_sha, str)
        or str(observed_runtime) != runtime_resolved
        or sha256_file(observed_runtime) != runtime_sha
    ):
        raise diagnostic.HybridQaMarginalMetaError(
            "runtime or implementation binding drifted"
        )
    for relative, expected in sorted(required.items()):
        path = PROJECT_PACKAGE_ROOT / relative
        if (
            not isinstance(relative, str)
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or not isinstance(expected, str)
            or len(expected) != 64
            or sha256_file(path) != expected
        ):
            raise diagnostic.HybridQaMarginalMetaError(
                "required implementation file drifted"
            )


def validate_output_parent(output_root: Path) -> None:
    parent = output_root.parent
    metadata = parent.lstat()
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise diagnostic.HybridQaMarginalMetaError("output parent mode drifted")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acquisition-root", type=Path, required=True)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    arguments = parser.parse_args()

    freeze, freeze_self = load_freeze(arguments.freeze)
    validate_frozen_implementation(freeze)
    if (
        freeze.get("architecture_decision_self_sha256")
        != diagnostic.ARCHITECTURE_DECISION_SHA256
        or freeze.get("output_root") != str(arguments.output_root)
        or freeze.get("acquisition_root") != str(arguments.acquisition_root)
        or freeze.get("minilm_asset_manifest") != str(arguments.asset_manifest)
        or freeze.get("minilm_model_root") != str(arguments.model_root)
    ):
        raise diagnostic.HybridQaMarginalMetaError("CLI/freeze binding drifted")
    validate_output_parent(arguments.output_root)
    os.mkdir(arguments.output_root, mode=0o700)
    attempt = diagnostic.self_hashed(
        {
            "schema": f"{diagnostic.VERSION}_attempt",
            "version": diagnostic.VERSION,
            "status": "attempt_started",
            "architecture_decision_sha256": diagnostic.ARCHITECTURE_DECISION_SHA256,
            "implementation_freeze_self_sha256": freeze_self,
            "retry_replay_resample": 0,
        },
        "attempt_self_sha256",
    )
    write_exclusive(arguments.output_root / "attempt.json", attempt)

    try:
        corpus, views, labels = diagnostic.load_consumed_packs(
            arguments.acquisition_root
        )
        encoder = diagnostic.open_gpu_encoder(
            asset_manifest_path=arguments.asset_manifest,
            model_root=arguments.model_root,
        )
        index = diagnostic.build_portable_corpus_index(
            articles=corpus.articles,
            encoder=encoder,
        )
        items = diagnostic.form_items(
            corpus=corpus,
            views=views,
            labels=labels,
            index=index,
            encoder=encoder,
        )
        evaluation = diagnostic.evaluate_crossfit(
            items=items,
            graph=corpus.graph,
        )
        result = diagnostic.build_safe_result(
            corpus=corpus,
            items=items,
            evaluation=evaluation,
            encoder=encoder,
            freeze_self_sha256=freeze_self,
        )
        result_file_sha = write_exclusive(
            arguments.output_root / "result.safe.json",
            result,
        )
        terminal = diagnostic.self_hashed(
            {
                "schema": f"{diagnostic.VERSION}_terminal",
                "version": diagnostic.VERSION,
                "status": "complete",
                "decision": evaluation["decision"],
                "result_self_sha256": result["result_self_sha256"],
                "result_file_sha256": result_file_sha,
                "implementation_freeze_self_sha256": freeze_self,
                "retry_replay_resample": 0,
            },
            "terminal_self_sha256",
        )
        write_exclusive(arguments.output_root / "terminal.json", terminal)
        return 0
    except BaseException as exc:
        failure = diagnostic.self_hashed(
            {
                "schema": f"{diagnostic.VERSION}_terminal_failure",
                "version": diagnostic.VERSION,
                "status": "implementation_or_runtime_invalid",
                "exception_class": type(exc).__name__,
                "exception_message": str(exc)[:512],
                "traceback_sha256": hashlib.sha256(
                    traceback.format_exc().encode("utf-8")
                ).hexdigest(),
                "implementation_freeze_self_sha256": freeze_self,
                "retry_replay_resample": 0,
                "efficacy_inference_permitted": False,
            },
            "failure_self_sha256",
        )
        write_exclusive(arguments.output_root / "failure.safe.json", failure)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
