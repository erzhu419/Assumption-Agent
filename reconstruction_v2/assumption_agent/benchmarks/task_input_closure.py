from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlparse

from ..models import stable_hash


TASK_INPUT_CLOSURE_MANIFEST_VERSION = "task_input_closure_manifest_v2"
TASK_INPUT_CLOSURE_POLICY_VERSION = "pre_agent_public_input_closure_exact_v1"
TASK_INPUT_PREPARATION_VERSION = "skilllearn_public_task_inputs_v2"
TASK_INPUT_BUILD_CONTEXT_POLICY_VERSION = "content_addressed_copy_no_runtime_fetch_v1"
ORGANIZE_FAMILY = "organize-messy-files"
STOCK_FAMILY = "stock-data-visualization"
D3_VERSION = "6.7.0"
D3_TARBALL_URL = f"https://registry.npmjs.org/d3/-/d3-{D3_VERSION}.tgz"

_DOWNLOAD_URL = re.compile(r"https?://[^\s\"']+")
_SAFE_COMPONENT = re.compile(r"[a-zA-Z0-9._-]+")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_RUN_HEREDOC = re.compile(
    r"^\s*RUN\s+<<(?P<quote>['\"]?)(?P<delimiter>[A-Za-z_][A-Za-z0-9_]*)"
    r"(?P=quote)\s*$",
    re.IGNORECASE,
)


class TaskInputClosureError(RuntimeError):
    pass


def default_task_input_cache_root() -> Path:
    configured = os.environ.get("ASSUMPTION_TASK_INPUT_CACHE_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (
        Path.home()
        / ".cache"
        / "assumption-agent-v2"
        / "task-inputs"
    ).resolve()


def extract_download_urls(dockerfile_text: str) -> tuple[str, ...]:
    """Return unique HTTP(S) download targets in source order."""

    seen: set[str] = set()
    urls: list[str] = []
    for match in _DOWNLOAD_URL.finditer(dockerfile_text):
        url = match.group(0).rstrip("),;\\")
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return tuple(urls)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_environment_hash(root: Path) -> str:
    """Hash source content across filesystem transports, excluding skills."""

    root = Path(root).expanduser().resolve()
    if not root.is_dir() or not (root / "Dockerfile").is_file():
        raise TaskInputClosureError("source environment has no Dockerfile")
    rows: list[dict[str, Any]] = []
    for path in sorted(
        root.rglob("*"), key=lambda value: value.relative_to(root).as_posix()
    ):
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] == "skills":
            continue
        if path.is_symlink():
            raise TaskInputClosureError(
                "task input source environments may not contain symbolic links"
            )
        if path.is_dir():
            rows.append({"path": relative.as_posix(), "kind": "dir"})
        elif path.is_file():
            rows.append(
                {
                    "path": relative.as_posix(),
                    "kind": "file",
                    "size": path.stat().st_size,
                    "sha256": _file_sha256(path),
                }
            )
    return stable_hash(rows)


def _manifest_hash(payload: Mapping[str, Any]) -> str:
    return stable_hash(
        {
            key: value
            for key, value in payload.items()
            if key not in {"manifest_hash", "closure_hash"}
        }
    )


def build_closure_manifest(
    *,
    source_environment: Path,
    sources: Mapping[str, Path],
    urls: Sequence[str],
    target_root: str = "/root/papers/all",
    target_filenames: Mapping[str, str] | None = None,
    family: str = ORGANIZE_FAMILY,
    item_id: str = "local-item",
    expected_file_count: int | None = None,
    expected_suffix_counts: Mapping[str, int] | None = None,
) -> Mapping[str, Any]:
    if not target_root.startswith("/root/") or ".." in Path(target_root).parts:
        raise TaskInputClosureError("task input target root is unsafe")
    ordered_urls = tuple(dict.fromkeys(str(url) for url in urls))
    if not ordered_urls:
        raise TaskInputClosureError("task input closure has no URLs")
    entries: list[dict[str, Any]] = []
    seen_targets: set[str] = set()
    for url in ordered_urls:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise TaskInputClosureError("task input source URL is invalid")
        source = Path(sources[url]).resolve() if url in sources else None
        if source is None or not source.is_file():
            raise TaskInputClosureError("task input source file is missing")
        filename = str((target_filenames or {}).get(url) or Path(parsed.path).name)
        if not filename or not _SAFE_COMPONENT.fullmatch(filename):
            raise TaskInputClosureError("task input filename is unsafe")
        target_path = f"{target_root.rstrip('/')}/{filename}"
        if target_path in seen_targets:
            raise TaskInputClosureError("task input target path is duplicated")
        seen_targets.add(target_path)
        digest = _file_sha256(source)
        size = source.stat().st_size
        if size <= 0:
            raise TaskInputClosureError("task input object is empty")
        entries.append(
            {
                "url": url,
                "target_path": target_path,
                "sha256": digest,
                "size_bytes": size,
                "object_name": digest,
            }
        )
    payload: dict[str, Any] = {
        "manifest_version": TASK_INPUT_CLOSURE_MANIFEST_VERSION,
        "policy": TASK_INPUT_CLOSURE_POLICY_VERSION,
        "source_environment_hash": source_environment_hash(source_environment),
        "family_hash": stable_hash({"family": family}),
        "item_id_hash": stable_hash({"item_id": item_id}),
        "target_root": target_root,
        "entries": entries,
        "expected_file_count": expected_file_count,
        "expected_suffix_counts": dict(sorted((expected_suffix_counts or {}).items())),
        "test_or_solution_content_accessed": False,
        "raw_content_persisted": False,
    }
    payload["manifest_hash"] = _manifest_hash(payload)
    payload["closure_hash"] = payload["manifest_hash"]
    return payload


def _validated_entries(manifest: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    if manifest.get("manifest_version") != TASK_INPUT_CLOSURE_MANIFEST_VERSION:
        raise TaskInputClosureError("task input manifest version mismatch")
    if manifest.get("policy") != TASK_INPUT_CLOSURE_POLICY_VERSION:
        raise TaskInputClosureError("task input closure policy mismatch")
    if not _SHA256.fullmatch(str(manifest.get("source_environment_hash") or "")):
        raise TaskInputClosureError("task input source environment hash is missing")
    declared_hash = str(manifest.get("manifest_hash") or "")
    if (
        not _SHA256.fullmatch(declared_hash)
        or declared_hash != _manifest_hash(manifest)
    ):
        raise TaskInputClosureError("task input manifest hash mismatch")
    if manifest.get("closure_hash") != declared_hash:
        raise TaskInputClosureError("task input closure hash mismatch")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise TaskInputClosureError("task input manifest entries are missing")
    target_root = str(manifest.get("target_root") or "")
    if not target_root.startswith("/root/") or ".." in Path(target_root).parts:
        raise TaskInputClosureError("task input manifest target root is unsafe")
    seen_targets: set[str] = set()
    normalized: list[Mapping[str, Any]] = []
    for row in entries:
        if not isinstance(row, Mapping):
            raise TaskInputClosureError("task input manifest entry is malformed")
        target_path = str(row.get("target_path") or "")
        digest = str(row.get("sha256") or "")
        object_name = str(row.get("object_name") or "")
        size = row.get("size_bytes")
        if (
            not target_path.startswith(f"{target_root.rstrip('/')}/")
            or ".." in Path(target_path).parts
            or target_path in seen_targets
            or not _SHA256.fullmatch(digest)
            or object_name != digest
            or not isinstance(size, int)
            or size <= 0
        ):
            raise TaskInputClosureError("task input manifest entry is invalid")
        seen_targets.add(target_path)
        normalized.append(row)
    return tuple(normalized)


def validate_closure_source_environment_hash(
    manifest: Mapping[str, Any], expected_source_environment_hash: str
) -> None:
    if (
        not _SHA256.fullmatch(expected_source_environment_hash)
        or not _SHA256.fullmatch(
            str(manifest.get("source_environment_hash") or "")
        )
        or manifest.get("source_environment_hash") != expected_source_environment_hash
    ):
        raise TaskInputClosureError("task input source environment hash mismatch")


def verify_closure_manifest(
    manifest: Mapping[str, Any], object_store: Path
) -> None:
    object_store = Path(object_store).expanduser().resolve()
    for row in _validated_entries(manifest):
        object_path = (object_store / str(row["object_name"])).resolve()
        if object_store not in object_path.parents or not object_path.is_file():
            raise TaskInputClosureError("task input object is missing")
        if object_path.stat().st_size != int(row["size_bytes"]):
            raise TaskInputClosureError("task input object size mismatch")
        if _file_sha256(object_path) != row["sha256"]:
            raise TaskInputClosureError("task input object hash mismatch")


def _strip_download_heredoc(dockerfile_text: str) -> tuple[str, int]:
    lines = dockerfile_text.splitlines()
    output: list[str] = []
    stripped = 0
    index = 0
    while index < len(lines):
        match = _RUN_HEREDOC.match(lines[index])
        if match is None:
            output.append(lines[index])
            index += 1
            continue
        delimiter = match.group("delimiter")
        body: list[str] = []
        cursor = index + 1
        while cursor < len(lines) and lines[cursor].strip() != delimiter:
            body.append(lines[cursor])
            cursor += 1
        if cursor >= len(lines):
            raise TaskInputClosureError("Dockerfile heredoc is unterminated")
        body_text = "\n".join(body)
        if "http" not in body_text or "/root/papers/all" not in body_text:
            raise TaskInputClosureError(
                "unexpected Dockerfile heredoc cannot be removed"
            )
        output.append(
            "# task input download heredoc replaced by "
            "content-addressed offline closure"
        )
        stripped += 1
        index = cursor + 1
    return "\n".join(output).rstrip() + "\n", stripped


def _inject_closure_copy(dockerfile_text: str, target_root: str) -> str:
    copy_line = f"COPY task-input-closure/ {target_root.rstrip('/')}/"
    if copy_line in dockerfile_text:
        raise TaskInputClosureError("task input closure COPY already exists")
    lines = dockerfile_text.splitlines()
    marker = next(
        (
            index
            for index, line in enumerate(lines)
            if "task input download heredoc replaced" in line
        ),
        None,
    )
    if marker is not None:
        lines.insert(marker + 1, copy_line)
        return "\n".join(lines).rstrip() + "\n"
    workdir = next(
        (
            index
            for index, line in enumerate(lines)
            if line.strip().upper().startswith("WORKDIR ")
        ),
        None,
    )
    if workdir is None:
        raise TaskInputClosureError("Dockerfile has no WORKDIR insertion point")
    lines.insert(workdir + 1, copy_line)
    return "\n".join(lines).rstrip() + "\n"


def materialize_build_context(
    *,
    source_context: Path,
    destination: Path,
    manifest: Mapping[str, Any],
    object_store: Path,
) -> Mapping[str, Any]:
    source_context = Path(source_context).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    object_store = Path(object_store).expanduser().resolve()
    verify_closure_manifest(manifest, object_store)
    if not (source_context / "Dockerfile").is_file():
        raise TaskInputClosureError("source build context has no Dockerfile")
    validate_closure_source_environment_hash(
        manifest,
        source_environment_hash(source_context),
    )
    if destination != source_context:
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source_context, destination)
    closure_dir = destination / "task-input-closure"
    if closure_dir.exists():
        shutil.rmtree(closure_dir)
    closure_dir.mkdir(parents=True)
    seen_names: set[str] = set()
    for row in _validated_entries(manifest):
        filename = Path(str(row["target_path"])).name
        if filename in seen_names or not _SAFE_COMPONENT.fullmatch(filename):
            raise TaskInputClosureError("task input closure filename collision")
        seen_names.add(filename)
        shutil.copy2(object_store / str(row["object_name"]), closure_dir / filename)
    dockerfile = destination / "Dockerfile"
    original_text = dockerfile.read_text(encoding="utf-8")
    rewritten, stripped_count = _strip_download_heredoc(original_text)
    target_root = str(manifest["target_root"])
    if target_root == "/root/papers/all" and stripped_count != 1:
        raise TaskInputClosureError("organize task must contain one download heredoc")
    if target_root != "/root/papers/all" and stripped_count != 0:
        raise TaskInputClosureError(
            "unexpected download heredoc for dependency closure"
        )
    rewritten = _inject_closure_copy(rewritten, target_root)
    dockerfile.write_text(rewritten, encoding="utf-8")
    receipt: dict[str, Any] = {
        "policy": TASK_INPUT_BUILD_CONTEXT_POLICY_VERSION,
        "manifest_hash": manifest["manifest_hash"],
        "source_environment_hash": manifest["source_environment_hash"],
        "entry_count": len(manifest["entries"]),
        "target_root_hash": stable_hash({"target_root": target_root}),
        "download_heredoc_removed_count": stripped_count,
        "dockerfile_hash": _file_sha256(dockerfile),
        "runtime_download_required": False,
        "test_or_solution_content_accessed": False,
        "raw_content_persisted": False,
    }
    receipt["receipt_hash"] = stable_hash(receipt)
    return receipt


def _parse_inventory_output(stdout: str) -> tuple[tuple[str, int, str], ...]:
    rows: list[tuple[str, int, str]] = []
    for raw in stdout.splitlines():
        parts = raw.split("\t", 2)
        if len(parts) != 3:
            raise TaskInputClosureError(
                "task input image inspection output is malformed"
            )
        digest, size_text, name = parts
        try:
            size = int(size_text)
        except ValueError as exc:
            raise TaskInputClosureError("task input image size is malformed") from exc
        if not _SHA256.fullmatch(digest) or size <= 0 or not name:
            raise TaskInputClosureError("task input image evidence is invalid")
        rows.append((digest, size, name))
    return tuple(rows)


def inspect_image_inputs(
    *,
    image: str,
    expected_manifest: Mapping[str, Any],
    run: Callable[..., Any] = subprocess.run,
    image_id: str | None = None,
) -> Mapping[str, Any]:
    entries = _validated_entries(expected_manifest)
    paths = [str(row["target_path"]) for row in entries]
    script = (
        "set -eu; for p in \"$@\"; do "
        "test -f \"$p\"; h=$(sha256sum \"$p\" | cut -d' ' -f1); "
        "s=$(stat -c %s \"$p\"); n=$(basename \"$p\"); "
        "printf '%s\\t%s\\t%s\\n' \"$h\" \"$s\" \"$n\"; done"
    )
    completed = run(
        [
            "docker",
            "run",
            "--rm",
            "--pull",
            "never",
            "--network",
            "none",
            image,
            "sh",
            "-c",
            script,
            "sh",
            *paths,
        ],
        capture_output=True,
        text=True,
    )
    if int(getattr(completed, "returncode", 1)) != 0:
        raise TaskInputClosureError("task input image inspection failed")
    observed = _parse_inventory_output(str(getattr(completed, "stdout", "") or ""))
    expected = tuple(
        sorted(
            (
                str(row["sha256"]),
                int(row["size_bytes"]),
                Path(str(row["target_path"])).name,
            )
            for row in entries
        )
    )
    if tuple(sorted(observed)) != expected:
        raise TaskInputClosureError("task input image closure mismatch")

    target_root = str(expected_manifest["target_root"])
    inventory = run(
        [
            "docker",
            "run",
            "--rm",
            "--pull",
            "never",
            "--network",
            "none",
            image,
            "sh",
            "-c",
            "find \"$1\" -maxdepth 1 -type f -printf '%f\\n' | sort",
            "sh",
            target_root,
        ],
        capture_output=True,
        text=True,
    )
    if int(getattr(inventory, "returncode", 1)) != 0:
        raise TaskInputClosureError("task input image inventory failed")
    names = tuple(
        name
        for name in str(getattr(inventory, "stdout", "") or "").splitlines()
        if name
    )
    expected_count = expected_manifest.get("expected_file_count")
    if expected_count is not None and len(names) != int(expected_count):
        raise TaskInputClosureError("task input image file count mismatch")
    suffix_counts = {
        suffix: sum(name.lower().endswith(suffix.lower()) for name in names)
        for suffix in (expected_manifest.get("expected_suffix_counts") or {})
    }
    if suffix_counts != dict(expected_manifest.get("expected_suffix_counts") or {}):
        raise TaskInputClosureError("task input image suffix inventory mismatch")
    receipt: dict[str, Any] = {
        "policy": TASK_INPUT_CLOSURE_POLICY_VERSION,
        "image_id": image_id or image,
        "manifest_hash": expected_manifest["manifest_hash"],
        "expected_tree_hash": stable_hash(expected),
        "observed_tree_hash": stable_hash(tuple(sorted(observed))),
        "expected_file_count": expected_count,
        "observed_file_count": len(names),
        "expected_suffix_counts": dict(
            expected_manifest.get("expected_suffix_counts") or {}
        ),
        "observed_suffix_counts": suffix_counts,
        "missing_count": 0,
        "content_mismatch_count": 0,
        "container_network": "none",
        "test_or_solution_content_accessed": False,
        "raw_content_persisted": False,
        "passed": True,
    }
    receipt["receipt_hash"] = stable_hash(receipt)
    return receipt


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _validate_cached_source(
    cache_root: Path, source_receipt: Mapping[str, Any]
) -> Path | None:
    digest = str(source_receipt.get("sha256") or "")
    size = source_receipt.get("size_bytes")
    if not _SHA256.fullmatch(digest) or not isinstance(size, int) or size <= 0:
        return None
    path = cache_root / "objects" / "sha256" / digest
    if (
        not path.is_file()
        or path.stat().st_size != size
        or _file_sha256(path) != digest
    ):
        return None
    return path


def _download_source(
    url: str,
    *,
    cache_root: Path,
    attempts: int,
    cache_only: bool = False,
) -> tuple[str, Path, bool]:
    source_key = stable_hash({"url": url})
    source_receipt_path = cache_root / "sources" / f"{source_key}.json"
    if source_receipt_path.is_file():
        try:
            source_receipt = json.loads(source_receipt_path.read_text(encoding="utf-8"))
            declared_receipt_hash = str(source_receipt.get("receipt_hash") or "")
            expected_receipt_hash = stable_hash(
                {
                    key: value
                    for key, value in source_receipt.items()
                    if key != "receipt_hash"
                }
            )
            cached = _validate_cached_source(cache_root, source_receipt)
            if (
                cached is not None
                and source_receipt.get("url") == url
                and declared_receipt_hash == expected_receipt_hash
            ):
                return url, cached, False
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    if cache_only:
        raise TaskInputClosureError(
            "task input source is missing from cache-only preparation"
        )

    objects = cache_root / "objects" / "sha256"
    objects.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        temporary: Path | None = None
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "assumption-agent-v2-offline-preparation/1"},
            )
            digest = hashlib.sha256()
            size = 0
            with urllib.request.urlopen(request, timeout=90) as response:
                status = int(getattr(response, "status", 200) or 200)
                if status < 200 or status >= 300:
                    raise TaskInputClosureError(f"download returned HTTP {status}")
                with tempfile.NamedTemporaryFile(
                    "wb", dir=objects, delete=False
                ) as handle:
                    temporary = Path(handle.name)
                    prefix = b""
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        if len(prefix) < 8:
                            prefix += chunk[: 8 - len(prefix)]
                        digest.update(chunk)
                        size += len(chunk)
                        handle.write(chunk)
            if size <= 0 or (
                url.lower().endswith(".pdf") and not prefix.startswith(b"%PDF-")
            ):
                raise TaskInputClosureError("downloaded task input has invalid content")
            sha256 = digest.hexdigest()
            object_path = objects / sha256
            if object_path.exists():
                if (
                    object_path.stat().st_size != size
                    or _file_sha256(object_path) != sha256
                ):
                    raise TaskInputClosureError(
                        "existing content-addressed object is corrupt"
                    )
                temporary.unlink(missing_ok=True)
            else:
                temporary.replace(object_path)
            receipt = {
                "url": url,
                "sha256": sha256,
                "size_bytes": size,
                "source_key": source_key,
            }
            receipt["receipt_hash"] = stable_hash(receipt)
            _atomic_json(source_receipt_path, receipt)
            return url, object_path, True
        except (OSError, urllib.error.URLError, TaskInputClosureError) as exc:
            last_error = exc
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            if attempt < attempts:
                time.sleep(float(min(8, 2 ** (attempt - 1))))
    raise TaskInputClosureError(
        f"task input download failed: {type(last_error).__name__}"
    )


def _store_bytes(cache_root: Path, content: bytes) -> Path:
    if not content:
        raise TaskInputClosureError("task input object is empty")
    digest = hashlib.sha256(content).hexdigest()
    destination = cache_root / "objects" / "sha256" / digest
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != content:
            raise TaskInputClosureError("content-addressed object collision")
        return destination
    with tempfile.NamedTemporaryFile(
        "wb", dir=destination.parent, delete=False
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(destination)
    return destination


def _extract_d3_bundle(cache_root: Path, tarball: Path) -> Path:
    try:
        with tarfile.open(tarball, "r:gz") as archive:
            package = json.loads(
                archive.extractfile("package/package.json").read().decode("utf-8")
            )
            if package.get("name") != "d3" or package.get("version") != D3_VERSION:
                raise TaskInputClosureError("D3 package identity mismatch")
            bundle = archive.extractfile("package/dist/d3.min.js").read()
    except (
        AttributeError,
        KeyError,
        OSError,
        tarfile.TarError,
        json.JSONDecodeError,
    ) as exc:
        raise TaskInputClosureError("D3 package archive is malformed") from exc
    if len(bundle) < 100_000 or b"6.7.0" not in bundle[:512]:
        raise TaskInputClosureError("D3 bundle validation failed")
    return _store_bytes(cache_root, bundle)


def _write_item_manifest(
    cache_root: Path, family: str, item_id: str, payload: Mapping[str, Any]
) -> None:
    if not _SAFE_COMPONENT.fullmatch(family) or not _SAFE_COMPONENT.fullmatch(item_id):
        raise TaskInputClosureError("task input closure item identity is unsafe")
    _atomic_json(cache_root / "closures" / family / f"{item_id}.json", payload)


def load_task_input_closure(
    cache_root: Path, family: str, item_id: str
) -> Mapping[str, Any]:
    if not _SAFE_COMPONENT.fullmatch(family) or not _SAFE_COMPONENT.fullmatch(item_id):
        raise TaskInputClosureError("task input closure item identity is unsafe")
    path = (
        Path(cache_root).expanduser().resolve()
        / "closures"
        / family
        / f"{item_id}.json"
    )
    if not path.is_file():
        raise TaskInputClosureError("task input closure manifest is missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskInputClosureError("task input closure manifest is malformed") from exc
    if not isinstance(payload, Mapping):
        raise TaskInputClosureError("task input closure manifest must be an object")
    if payload.get("family_hash") != stable_hash({"family": family}) or payload.get(
        "item_id_hash"
    ) != stable_hash({"item_id": item_id}):
        raise TaskInputClosureError("task input closure identity mismatch")
    verify_closure_manifest(payload, Path(cache_root) / "objects" / "sha256")
    return payload


def family_requires_task_input_closure(family: str) -> bool:
    return family in {ORGANIZE_FAMILY, STOCK_FAMILY}


def prepare_skilllearn_task_inputs(
    *,
    benchmark_root: Path,
    cache_root: Path | None = None,
    parallel_workers: int = 16,
    attempts: int = 5,
    cache_only: bool = False,
) -> Mapping[str, Any]:
    benchmark_root = Path(benchmark_root).expanduser().resolve()
    cache_root = (cache_root or default_task_input_cache_root()).expanduser().resolve()
    if parallel_workers <= 0 or attempts <= 0:
        raise ValueError("task input preparation bounds must be positive")
    tasks_root = benchmark_root / "tasks"
    organize: dict[str, tuple[str, ...]] = {}
    for task in sorted((tasks_root / ORGANIZE_FAMILY).glob(f"{ORGANIZE_FAMILY}-*")):
        dockerfile = task / "environment" / "Dockerfile"
        if not dockerfile.is_file():
            continue
        urls = tuple(
            url
            for url in extract_download_urls(dockerfile.read_text(encoding="utf-8"))
            if urlparse(url).netloc == "arxiv.org" and ".pdf" in urlparse(url).path
        )
        if len(urls) != 100:
            raise TaskInputClosureError(
                f"{task.name} must declare exactly 100 PDF inputs"
            )
        organize[task.name] = urls
    if len(organize) != 6:
        raise TaskInputClosureError("organize task inventory is incomplete")
    stock_items = tuple(
        task.name
        for task in sorted((tasks_root / STOCK_FAMILY).glob(f"{STOCK_FAMILY}-*"))
        if (task / "environment" / "Dockerfile").is_file()
    )
    if len(stock_items) != 5:
        raise TaskInputClosureError("stock task inventory is incomplete")

    download_urls = tuple(
        sorted(
            {url for urls in organize.values() for url in urls}
            | {D3_TARBALL_URL}
        )
    )
    sources: dict[str, Path] = {}
    downloaded_source_count = 0
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=parallel_workers
    ) as executor:
        futures = {
            executor.submit(
                _download_source,
                url,
                cache_root=cache_root,
                attempts=attempts,
                cache_only=cache_only,
            ): url
            for url in download_urls
        }
        for future in concurrent.futures.as_completed(futures):
            url, path, downloaded = future.result()
            sources[url] = path
            if downloaded:
                downloaded_source_count += 1

    for item_id, urls in organize.items():
        manifest = build_closure_manifest(
            source_environment=(
                tasks_root / ORGANIZE_FAMILY / item_id / "environment"
            ),
            sources=sources,
            urls=urls,
            target_root="/root/papers/all",
            family=ORGANIZE_FAMILY,
            item_id=item_id,
            expected_file_count=103,
            expected_suffix_counts={".pdf": 100, ".docx": 2, ".pptx": 1},
        )
        _write_item_manifest(cache_root, ORGANIZE_FAMILY, item_id, manifest)

    d3_bundle = _extract_d3_bundle(cache_root, sources[D3_TARBALL_URL])
    d3_url = f"{D3_TARBALL_URL}#package/dist/d3.min.js"
    for item_id in stock_items:
        manifest = build_closure_manifest(
            source_environment=(
                tasks_root / STOCK_FAMILY / item_id / "environment"
            ),
            sources={d3_url: d3_bundle},
            urls=[d3_url],
            target_root="/root/data",
            target_filenames={d3_url: "d3.v6.min.js"},
            family=STOCK_FAMILY,
            item_id=item_id,
        )
        _write_item_manifest(cache_root, STOCK_FAMILY, item_id, manifest)

    manifest_records = [
        (family, item_id, load_task_input_closure(cache_root, family, item_id))
        for family, item_ids in (
            (ORGANIZE_FAMILY, tuple(organize)),
            (STOCK_FAMILY, stock_items),
        )
        for item_id in item_ids
    ]
    manifests = [manifest for _, _, manifest in manifest_records]
    object_paths = {
        str(row["object_name"])
        for manifest in manifests
        for row in manifest["entries"]
    }
    closure_ledger = sorted(
        (
            {
                "item_id_hash": stable_hash({"item_id": item_id}),
                "family_hash": stable_hash({"family": family}),
                "source_environment_hash": manifest["source_environment_hash"],
                "closure_hash": manifest["closure_hash"],
                "object_count": len(manifest["entries"]),
                "object_hashes": sorted(
                    str(entry["object_name"])
                    for entry in manifest["entries"]
                ),
                "object_set_hash": stable_hash(
                    sorted(
                        str(entry["object_name"])
                        for entry in manifest["entries"]
                    )
                ),
            }
            for family, item_id, manifest in manifest_records
        ),
        key=lambda row: (str(row["family_hash"]), str(row["item_id_hash"])),
    )
    receipt: dict[str, Any] = {
        "preparation_version": TASK_INPUT_PREPARATION_VERSION,
        "policy": TASK_INPUT_CLOSURE_POLICY_VERSION,
        "benchmark_source_environment_set_hash": stable_hash(
            sorted(
                str(row["source_environment_hash"])
                for row in closure_ledger
            )
        ),
        "organize_item_count": len(organize),
        "stock_item_count": len(stock_items),
        "closure_count": len(manifests),
        "closure_ledger_item_count": len(closure_ledger),
        "closure_ledger": closure_ledger,
        "closure_ledger_hash": stable_hash(closure_ledger),
        "download_source_count": len(download_urls),
        "content_object_count": len(object_paths),
        "closure_set_hash": stable_hash(
            sorted(str(manifest["manifest_hash"]) for manifest in manifests)
        ),
        "object_set_hash": stable_hash(sorted(object_paths)),
        "parallel_workers": parallel_workers,
        "maximum_attempts": attempts,
        "cache_only_preparation": cache_only,
        "cached_source_count": len(download_urls) - downloaded_source_count,
        "downloaded_source_count": downloaded_source_count,
        "online_preparation_performed": downloaded_source_count > 0,
        "trial_runtime_download_required": False,
        "test_or_solution_content_accessed": False,
        "raw_content_persisted": False,
        "passed": True,
    }
    receipt["receipt_hash"] = stable_hash(receipt)
    _atomic_json(cache_root / "preparation_receipt.json", receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare SkillLearn public task inputs"
    )
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument(
        "--cache-root", type=Path, default=default_task_input_cache_root()
    )
    parser.add_argument("--parallel-workers", type=int, default=16)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    receipt = prepare_skilllearn_task_inputs(
        benchmark_root=args.benchmark_root,
        cache_root=args.cache_root,
        parallel_workers=args.parallel_workers,
        attempts=args.attempts,
        cache_only=args.cache_only,
    )
    if args.output:
        _atomic_json(args.output.expanduser().resolve(), receipt)
    else:
        print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
