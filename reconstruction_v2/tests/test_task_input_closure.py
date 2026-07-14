from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

import pytest

from assumption_agent.benchmarks.task_input_closure import (
    TaskInputClosureError,
    build_closure_manifest,
    extract_download_urls,
    inspect_image_inputs,
    materialize_build_context,
    source_environment_hash,
    verify_closure_manifest,
)
from assumption_agent.models import stable_hash


_ORGANIZE_DOCKERFILE_EXCERPT = '''\
FROM ubuntu:24.04

WORKDIR /root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# This is the BuildKit-only block used by organize-messy-files.
RUN <<'EOF'
set -euo pipefail
mkdir -p /root/papers/all
cd /root/papers/all

urls="""
https://arxiv.org/pdf/1607.00266v1.pdf
https://arxiv.org/pdf/2603.01780.pdf
https://arxiv.org/pdf/2402.11651v2.pdf
"""

for url in $urls; do
    wget -q -O "$(basename "$url")" "$url"
done
EOF

COPY DAMOP.pptx paper_file_1.docx paper_file_2.docx /root/papers/all/
'''


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write_downloads(tmp_path: Path) -> tuple[tuple[str, ...], dict[str, Path]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    urls = (
        "https://arxiv.org/pdf/1607.00266v1.pdf",
        "https://arxiv.org/pdf/2603.01780.pdf",
    )
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    first.write_bytes(b"%PDF-1.7\nfirst fixture\n")
    second.write_bytes(b"%PDF-1.7\nsecond fixture\n")
    return urls, {urls[0]: first, urls[1]: second}


def _write_source_environment(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "Dockerfile").write_text(
        _ORGANIZE_DOCKERFILE_EXCERPT,
        encoding="utf-8",
    )
    for name in ("DAMOP.pptx", "paper_file_1.docx", "paper_file_2.docx"):
        (root / name).write_bytes(name.encode("ascii"))
    return root


def _build_manifest(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Path], Path]:
    urls, sources = _write_downloads(tmp_path / "downloads")
    source_environment = _write_source_environment(tmp_path / "source-environment")
    manifest = dict(
        build_closure_manifest(
            source_environment=source_environment,
            sources=sources,
            urls=urls,
            target_root="/root/papers/all",
            expected_file_count=2,
            expected_suffix_counts={".pdf": 2},
        )
    )
    return manifest, sources, source_environment


def _populate_object_store(
    object_store: Path,
    manifest: dict[str, Any],
    sources: dict[str, Path],
) -> None:
    object_store.mkdir(parents=True, exist_ok=True)
    for row in manifest["entries"]:
        destination = object_store / row["object_name"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(sources[row["url"]].read_bytes())


def _inspection_stdout(manifest: dict[str, Any]) -> str:
    return "\n".join(
        f"{row['sha256']}\t{row['size_bytes']}\t{Path(row['target_path']).name}"
        for row in manifest["entries"]
    )


def _inventory_stdout(manifest: dict[str, Any]) -> str:
    return "\n".join(Path(row["target_path"]).name for row in manifest["entries"])


def test_extracts_urls_from_organize_dockerfile_heredoc() -> None:
    assert extract_download_urls(_ORGANIZE_DOCKERFILE_EXCERPT) == (
        "https://arxiv.org/pdf/1607.00266v1.pdf",
        "https://arxiv.org/pdf/2603.01780.pdf",
        "https://arxiv.org/pdf/2402.11651v2.pdf",
    )


def test_closure_manifest_is_content_addressed_and_canonically_hashed(
    tmp_path: Path,
) -> None:
    urls, sources = _write_downloads(tmp_path)
    source_environment = _write_source_environment(tmp_path / "environment")

    first = dict(
        build_closure_manifest(
            source_environment=source_environment,
            sources=sources,
            urls=urls,
            target_root="/root/papers/all",
        )
    )
    reordered_sources = {urls[1]: sources[urls[1]], urls[0]: sources[urls[0]]}
    second = dict(
        build_closure_manifest(
            source_environment=source_environment,
            sources=reordered_sources,
            urls=urls,
            target_root="/root/papers/all",
        )
    )

    assert first == second
    assert first["target_root"] == "/root/papers/all"
    assert first["manifest_hash"] == stable_hash(
        {
            key: value
            for key, value in first.items()
            if key not in {"manifest_hash", "closure_hash"}
        }
    )
    assert first["closure_hash"] == first["manifest_hash"]
    assert first["source_environment_hash"] == source_environment_hash(
        source_environment
    )
    assert [row["url"] for row in first["entries"]] == list(urls)
    for row in first["entries"]:
        content = sources[row["url"]].read_bytes()
        assert row["sha256"] == _sha256(content)
        assert row["object_name"] == row["sha256"]
        assert row["size_bytes"] == len(content)
        assert Path(row["target_path"]).name == Path(row["url"]).name

    sources[urls[0]].write_bytes(b"%PDF-1.7\nchanged fixture\n")
    changed = build_closure_manifest(
        source_environment=source_environment,
        sources=sources,
        urls=urls,
        target_root="/root/papers/all",
    )
    assert changed["manifest_hash"] != first["manifest_hash"]
    assert changed["entries"][0]["sha256"] != first["entries"][0]["sha256"]


def test_closure_manifest_supports_a_safe_explicit_target_filename(
    tmp_path: Path,
) -> None:
    url = "https://registry.npmjs.org/d3/-/d3-6.7.0.tgz#package/dist/d3.min.js"
    source = tmp_path / "d3.v6.min.js"
    source.write_bytes(b"/*! d3.js v6.7.0 */\n")
    source_environment = _write_source_environment(tmp_path / "environment")

    manifest = build_closure_manifest(
        source_environment=source_environment,
        sources={url: source},
        urls=(url,),
        target_root="/root/data",
        target_filenames={url: "d3.v6.min.js"},
    )

    assert manifest["entries"][0]["target_path"] == "/root/data/d3.v6.min.js"
    with pytest.raises(TaskInputClosureError):
        build_closure_manifest(
            source_environment=source_environment,
            sources={url: source},
            urls=(url,),
            target_root="/root/data",
            target_filenames={url: "../d3.v6.min.js"},
        )


def test_materializes_legacy_compatible_build_context_with_plain_copy(
    tmp_path: Path,
) -> None:
    manifest, sources, source_context = _build_manifest(tmp_path)
    object_store = tmp_path / "objects"
    _populate_object_store(object_store, manifest, sources)

    # Runner preparation may add this directory; it is deliberately excluded.
    (source_context / "skills").mkdir()

    destination = tmp_path / "materialized-context"
    receipt = materialize_build_context(
        source_context=source_context,
        destination=destination,
        manifest=manifest,
        object_store=object_store,
    )

    dockerfile = (destination / "Dockerfile").read_text(encoding="utf-8")
    assert "RUN <<'EOF'" not in dockerfile
    assert "wget" not in dockerfile
    assert "https://arxiv.org/" not in dockerfile
    assert "COPY task-input-closure/ /root/papers/all/" in dockerfile
    assert (
        "COPY DAMOP.pptx paper_file_1.docx paper_file_2.docx "
        "/root/papers/all/"
    ) in dockerfile
    assert receipt["manifest_hash"] == manifest["manifest_hash"]
    assert receipt["source_environment_hash"] == manifest[
        "source_environment_hash"
    ]
    assert receipt["entry_count"] == len(manifest["entries"])
    assert receipt["download_heredoc_removed_count"] == 1
    assert receipt["runtime_download_required"] is False
    for row in manifest["entries"]:
        materialized = destination / "task-input-closure" / Path(
            row["target_path"]
        ).name
        assert materialized.read_bytes() == sources[row["url"]].read_bytes()


def test_materialization_rejects_source_dockerfile_url_drift(
    tmp_path: Path,
) -> None:
    manifest, sources, source_context = _build_manifest(tmp_path)
    object_store = tmp_path / "objects"
    _populate_object_store(object_store, manifest, sources)
    dockerfile = source_context / "Dockerfile"
    dockerfile.write_text(
        dockerfile.read_text(encoding="utf-8").replace(
            "1607.00266v1.pdf",
            "1607.00266v2.pdf",
        ),
        encoding="utf-8",
    )

    with pytest.raises(TaskInputClosureError, match="source environment hash mismatch"):
        materialize_build_context(
            source_context=source_context,
            destination=tmp_path / "materialized-context",
            manifest=manifest,
            object_store=object_store,
        )
    assert not (tmp_path / "materialized-context").exists()


def test_v1_manifest_without_source_binding_fails_closed(tmp_path: Path) -> None:
    manifest, _, _ = _build_manifest(tmp_path)
    manifest["manifest_version"] = "task_input_closure_manifest_v1"

    with pytest.raises(TaskInputClosureError, match="manifest version mismatch"):
        verify_closure_manifest(manifest, tmp_path / "objects")


@pytest.mark.parametrize("failure_mode", ["missing", "tampered"])
def test_closure_object_verification_fails_closed(
    tmp_path: Path,
    failure_mode: str,
) -> None:
    manifest, sources, _ = _build_manifest(tmp_path)
    object_store = tmp_path / "objects"
    _populate_object_store(object_store, manifest, sources)
    first_object = object_store / manifest["entries"][0]["object_name"]
    if failure_mode == "missing":
        first_object.unlink()
    else:
        first_object.write_bytes(b"%PDF-1.7\ntampered\n")

    with pytest.raises(TaskInputClosureError):
        verify_closure_manifest(manifest, object_store)


def test_parses_network_disabled_image_inspection_receipt(
    tmp_path: Path,
) -> None:
    manifest, _, _ = _build_manifest(tmp_path)
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        stdout = (
            _inspection_stdout(manifest)
            if any("sha256sum" in argument for argument in command)
            else _inventory_stdout(manifest)
        )
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=stdout,
            stderr="",
        )

    image_id = f"sha256:{'a' * 64}"
    receipt = inspect_image_inputs(
        image=image_id,
        expected_manifest=manifest,
        run=fake_run,
    )

    assert len(calls) == 2
    for command, kwargs in calls:
        assert command[:2] == ["docker", "run"]
        assert command[command.index("--pull") + 1] == "never"
        assert command[command.index("--network") + 1] == "none"
        assert image_id in command
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
    assert receipt["passed"] is True
    assert receipt["container_network"] == "none"
    assert receipt["manifest_hash"] == manifest["manifest_hash"]
    assert receipt["expected_file_count"] == len(manifest["entries"])
    assert receipt["observed_file_count"] == len(manifest["entries"])
    assert receipt["missing_count"] == 0
    assert receipt["content_mismatch_count"] == 0
    assert receipt["raw_content_persisted"] is False


def test_image_inspection_inventory_mismatch_fails_closed(tmp_path: Path) -> None:
    manifest, _, _ = _build_manifest(tmp_path)
    rows = _inspection_stdout(manifest).splitlines()
    _, size, name = rows[0].split("\t")
    rows[0] = f"{'0' * 64}\t{size}\t{name}"

    def fake_run(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="\n".join(rows),
            stderr="",
        )

    with pytest.raises(TaskInputClosureError):
        inspect_image_inputs(
            image=f"sha256:{'b' * 64}",
            expected_manifest=manifest,
            run=fake_run,
        )
