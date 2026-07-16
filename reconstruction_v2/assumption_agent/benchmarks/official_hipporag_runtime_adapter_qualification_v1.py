"""Non-scoring qualification for the official HippoRAG runtime adapter.

The qualification uses only a deterministic, locally generated hierarchical
fixture.  It never opens an external benchmark pack and never computes a
benchmark score.  The official package is imported normally from a dedicated
runtime, then exercised through initialization, indexing, retrieval, and QA
with local Transformer models while network syscalls are traced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
import unicodedata
from typing import Any, Mapping, Sequence

from ..models import stable_hash


QUALIFICATION_VERSION = "official-hipporag-runtime-adapter-qualification-v1"
CUSTOM_BENCHMARK_SCHEMA = "synthetic-hierarchical-custom-benchmark-v1"
ANSWER_SCHEMA = "hipporag-custom-answer-v1"
CORE_INPUT_SCHEMA = "hipporag-core-synthetic-input-v1"
OFFICIAL_COMMIT = "ef2f14c4f254f11ac29f9395f262466ad1bb4d10"
OFFICIAL_REMOTE = "https://github.com/OSU-NLP-Group/HippoRAG.git"


class OfficialHippoRAGQualificationError(RuntimeError):
    """Raised when the qualification contract cannot be audited."""


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OfficialHippoRAGQualificationError(f"{field} must be non-empty text")
    return value.strip()


def normalize_answer(value: str) -> str:
    """Normalize one answer without introducing language-specific semantics."""

    if not isinstance(value, str):
        raise TypeError("answer must be text")
    folded = unicodedata.normalize("NFKC", value).casefold()
    characters = [
        " " if unicodedata.category(character)[0] in {"P", "Z"} else character
        for character in folded
    ]
    return " ".join("".join(characters).split())


def normalize_multi_answers(values: Sequence[str]) -> tuple[str, ...]:
    """Normalize aliases, remove duplicates, and return a stable ordering."""

    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("answers must be a sequence of text aliases")
    normalized = {normalize_answer(value) for value in values}
    normalized.discard("")
    if not normalized:
        raise OfficialHippoRAGQualificationError("no normalized answer aliases remain")
    return tuple(sorted(normalized))


def build_synthetic_fixture() -> dict[str, Any]:
    """Create a tiny deterministic hierarchy with no external data ancestry."""

    seed = hashlib.sha256(QUALIFICATION_VERSION.encode("utf-8")).hexdigest()
    leaf_a = hashlib.sha256(f"{seed}:leaf-a".encode("utf-8")).hexdigest()[:12]
    leaf_b = hashlib.sha256(f"{seed}:leaf-b".encode("utf-8")).hexdigest()[:12]
    return {
        "schema": CUSTOM_BENCHMARK_SCHEMA,
        "provenance": {
            "generator": QUALIFICATION_VERSION,
            "external_corpus_rows": 0,
            "external_question_rows": 0,
            "model_generated_rows": 0,
        },
        "corpus": [
            {
                "node_id": "root",
                "parent_id": None,
                "title": "Synthetic root",
                "text": "The synthetic hierarchy contains two branches.",
            },
            {
                "node_id": "branch-a",
                "parent_id": "root",
                "title": "Synthetic branch A",
                "text": "Branch A contains leaf A.",
            },
            {
                "node_id": "leaf-a",
                "parent_id": "branch-a",
                "title": "Synthetic leaf A",
                "text": f"Leaf A has locally generated signal {leaf_a}.",
            },
            {
                "node_id": "leaf-b",
                "parent_id": "root",
                "title": "Synthetic leaf B",
                "text": f"Leaf B has locally generated signal {leaf_b}.",
            },
        ],
        "questions": [
            {
                "question_id": "synthetic-q1",
                "query": "Which locally generated signal belongs to synthetic leaf B?",
                "accepted_aliases": [leaf_b, leaf_b.upper(), f"  {leaf_b}  "],
            }
        ],
    }


def _validated_nodes(payload: Mapping[str, Any]) -> tuple[list[dict[str, str | None]], dict[str, str]]:
    if payload.get("schema") != CUSTOM_BENCHMARK_SCHEMA:
        raise OfficialHippoRAGQualificationError("custom benchmark schema mismatch")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping) or any(
        provenance.get(field) != 0
        for field in ("external_corpus_rows", "external_question_rows", "model_generated_rows")
    ):
        raise PermissionError("fixture provenance is not fully synthetic")
    raw_nodes = payload.get("corpus")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise OfficialHippoRAGQualificationError("synthetic corpus is empty")

    nodes: list[dict[str, str | None]] = []
    by_id: dict[str, dict[str, str | None]] = {}
    for index, raw in enumerate(raw_nodes):
        if not isinstance(raw, Mapping):
            raise OfficialHippoRAGQualificationError("corpus row must be an object")
        node_id = _require_text(raw.get("node_id"), f"corpus[{index}].node_id")
        if node_id in by_id:
            raise OfficialHippoRAGQualificationError("duplicate node_id")
        parent_raw = raw.get("parent_id")
        parent_id = None if parent_raw is None else _require_text(parent_raw, "parent_id")
        row: dict[str, str | None] = {
            "node_id": node_id,
            "parent_id": parent_id,
            "title": _require_text(raw.get("title"), "title"),
            "text": _require_text(raw.get("text"), "text"),
        }
        nodes.append(row)
        by_id[node_id] = row

    paths: dict[str, str] = {}

    def resolve(node_id: str, active: frozenset[str] = frozenset()) -> str:
        if node_id in paths:
            return paths[node_id]
        if node_id in active:
            raise OfficialHippoRAGQualificationError("hierarchy cycle detected")
        node = by_id[node_id]
        parent_id = node["parent_id"]
        if parent_id is None:
            path = node_id
        else:
            if parent_id not in by_id:
                raise OfficialHippoRAGQualificationError("unknown parent_id")
            path = f"{resolve(parent_id, active | {node_id})}/{node_id}"
        paths[node_id] = path
        return path

    for node_id in by_id:
        resolve(node_id)
    return nodes, paths


def documents_from_custom_json(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Flatten a validated hierarchy into deterministic HippoRAG documents."""

    nodes, paths = _validated_nodes(payload)
    return tuple(
        "\n".join(
            (
                f"Node-ID: {node['node_id']}",
                f"Hierarchy: {paths[str(node['node_id'])]}",
                f"Title: {node['title']}",
                f"Content: {node['text']}",
            )
        )
        for node in nodes
    )


def questions_from_custom_json(payload: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    _validated_nodes(payload)
    raw_questions = payload.get("questions")
    if not isinstance(raw_questions, list) or not raw_questions:
        raise OfficialHippoRAGQualificationError("synthetic questions are empty")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_questions):
        if not isinstance(raw, Mapping):
            raise OfficialHippoRAGQualificationError("question row must be an object")
        question_id = _require_text(raw.get("question_id"), f"questions[{index}].question_id")
        if question_id in seen:
            raise OfficialHippoRAGQualificationError("duplicate question_id")
        seen.add(question_id)
        aliases_raw = raw.get("accepted_aliases")
        if not isinstance(aliases_raw, list) or any(not isinstance(item, str) for item in aliases_raw):
            raise OfficialHippoRAGQualificationError("accepted_aliases must be text")
        rows.append(
            {
                "question_id": question_id,
                "query": _require_text(raw.get("query"), "query"),
                "normalized_aliases": normalize_multi_answers(aliases_raw),
                "source_aliases": tuple(aliases_raw),
            }
        )
    return tuple(rows)


def write_answer_json(
    path: Path,
    *,
    question_ids: Sequence[str],
    predictions: Mapping[str, str],
) -> str:
    """Write predictions only; accepted aliases never enter ``answer.json``."""

    ordered_ids = tuple(_require_text(value, "question_id") for value in question_ids)
    if len(set(ordered_ids)) != len(ordered_ids):
        raise OfficialHippoRAGQualificationError("duplicate answer question_id")
    if set(predictions) != set(ordered_ids):
        raise OfficialHippoRAGQualificationError("prediction IDs do not match questions")
    rows = [
        {
            "question_id": question_id,
            "answer": _require_text(predictions[question_id], "prediction"),
        }
        for question_id in ordered_ids
    ]
    payload = {"schema": ANSWER_SCHEMA, "answers": rows}
    raw = (json.dumps(payload, sort_keys=True, ensure_ascii=True, indent=2) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return _sha256_bytes(raw)


def _git_bytes(repository: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise OfficialHippoRAGQualificationError(
            f"git provenance command failed: {arguments[0]}"
        )
    return completed.stdout


def official_source_binding(repository: Path) -> dict[str, Any]:
    """Bind the official source without opening benchmark or result folders."""

    repository = repository.resolve()
    commit = _git_bytes(repository, "rev-parse", f"{OFFICIAL_COMMIT}^{{commit}}").decode().strip()
    if commit != OFFICIAL_COMMIT:
        raise OfficialHippoRAGQualificationError("official commit mismatch")
    remote = _git_bytes(repository, "remote", "get-url", "origin").decode().strip()
    if remote.rstrip("/") != OFFICIAL_REMOTE.rstrip("/"):
        raise OfficialHippoRAGQualificationError("official remote mismatch")
    package_tree = _git_bytes(
        repository, "rev-parse", f"{OFFICIAL_COMMIT}:src/hipporag"
    ).decode().strip()
    tracked_status = _git_bytes(
        repository, "status", "--porcelain", "--untracked-files=no"
    ).decode().strip()
    if tracked_status:
        raise OfficialHippoRAGQualificationError("official checkout has tracked modifications")
    files: dict[str, str] = {}
    for relative in (
        "requirements.txt",
        "setup.py",
        "src/hipporag/__init__.py",
        "src/hipporag/HippoRAG.py",
    ):
        raw = _git_bytes(repository, "show", f"{OFFICIAL_COMMIT}:{relative}")
        files[relative] = _sha256_bytes(raw)
    tree_names = _git_bytes(
        repository,
        "ls-tree",
        "-r",
        "--name-only",
        OFFICIAL_COMMIT,
        "--",
        "src/hipporag",
    ).decode().splitlines()
    python_rows = []
    for relative in sorted(name for name in tree_names if name.endswith(".py")):
        raw = _git_bytes(repository, "show", f"{OFFICIAL_COMMIT}:{relative}")
        python_rows.append(
            {
                "path": relative.removeprefix("src/hipporag/"),
                "sha256": _sha256_bytes(raw),
            }
        )
    return {
        "remote": remote,
        "commit": commit,
        "package_tree_object": package_tree,
        "source_file_sha256": files,
        "python_source_file_count": len(python_rows),
        "python_source_tree_sha256": stable_hash(python_rows),
        "runtime_archive_allowlist": ["README.md", "LICENSE", "setup.py", "src/hipporag"],
        "official_checkout_tracked_status_clean": True,
        "runtime_archive_target_commit": commit,
    }


_CORE_PROBE_PROGRAM = r"""
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import sys
import tempfile

from hipporag import HippoRAG
import hipporag
from hipporag.utils.config_utils import BaseConfig

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    probe_input = json.load(handle)
assert probe_input["schema"] == "hipporag-core-synthetic-input-v1"
documents = probe_input["documents"]
queries = probe_input["queries"]
assert documents and queries and all(isinstance(value, str) for value in documents + queries)

def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None

with tempfile.TemporaryDirectory(prefix="official-hipporag-core-") as working_dir:
    config = BaseConfig(
        save_dir=working_dir,
        llm_name="Transformers/" + sys.argv[2],
        embedding_model_name="Transformers/" + sys.argv[3],
        openie_mode="online",
        max_new_tokens=4,
        retrieval_top_k=len(documents),
        qa_top_k=len(documents),
        force_index_from_scratch=True,
        save_openie=True,
    )
    rag = HippoRAG(global_config=config)
    rag.llm_model.llm_config.generate_params["max_tokens"] = 4
    rag.index(documents)
    retrieved = rag.retrieve(queries, num_to_retrieve=len(documents))
    qa_payload = rag.rag_qa(queries)
    qa_rows = qa_payload[0]
    answers = [row.answer for row in qa_rows]

module_file = Path(hipporag.__file__).resolve()
core_file = module_file.with_name("HippoRAG.py")
python_rows = [
    {
        "path": str(path.relative_to(module_file.parent)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    for path in sorted(module_file.parent.rglob("*.py"))
]
python_tree_sha256 = hashlib.sha256(
    json.dumps(
        python_rows,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
payload = {
    "status": "passed",
    "core_class_module": HippoRAG.__module__,
    "config_class_module": BaseConfig.__module__,
    "installed_init_sha256": hashlib.sha256(module_file.read_bytes()).hexdigest(),
    "installed_core_sha256": hashlib.sha256(core_file.read_bytes()).hexdigest(),
    "installed_python_source_file_count": len(python_rows),
    "installed_python_source_tree_sha256": python_tree_sha256,
    "network_namespace": os.readlink("/proc/self/ns/net"),
    "document_count": len(documents),
    "query_count": len(queries),
    "retrieval_result_count": len(retrieved),
    "retrieved_document_counts": [len(row.docs) for row in retrieved],
    "qa_result_count": len(qa_rows),
    "qa_answer_sha256": [hashlib.sha256(value.encode("utf-8")).hexdigest() for value in answers],
    "runtime_versions": {
        name: package_version(name)
        for name in (
            "hipporag", "torch", "transformers", "sentence-transformers",
            "python-igraph", "openai", "litellm", "gritlm", "networkx",
            "pydantic", "tenacity", "tiktoken", "vllm"
        )
    },
}
print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
""".strip()


def _parse_last_json_object(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise OfficialHippoRAGQualificationError("official core emitted no JSON receipt")


def _trace_summary(trace_prefix: Path) -> dict[str, Any]:
    paths = sorted(trace_prefix.parent.glob(trace_prefix.name + "*"))
    if not paths:
        raise OfficialHippoRAGQualificationError("network trace was not produced")
    file_rows: list[dict[str, Any]] = []
    syscall_lines = 0
    connect_lines = 0
    external_connect_lines = 0
    for path in paths:
        raw = path.read_bytes()
        text = raw.decode("utf-8", errors="replace")
        lines = [line for line in text.splitlines() if line.strip()]
        syscall_lines += len(lines)
        connect_lines += sum("connect(" in line for line in lines)
        external_connect_lines += sum(
            "connect(" in line and ("AF_INET" in line or "AF_INET6" in line)
            for line in lines
        )
        file_rows.append({"sha256": _sha256_bytes(raw), "line_count": len(lines)})
    return {
        "trace_file_count": len(paths),
        "network_syscall_line_count": syscall_lines,
        "connect_syscall_line_count": connect_lines,
        "external_connect_attempt_line_count": external_connect_lines,
        "trace_bundle_sha256": stable_hash(file_rows),
    }


def run_official_core_probe(
    *,
    runtime_python: Path,
    local_llm_model: Path,
    local_embedding_model: Path,
    documents: Sequence[str],
    queries: Sequence[str],
    source_binding: Mapping[str, Any],
    timeout_seconds: int = 300,
) -> dict[str, Any]:
    """Exercise the unmodified official core with offline local models."""

    # Preserve the venv launcher symlink: resolving it would execute the base
    # interpreter directly and silently drop the venv's site-packages.
    runtime_python = runtime_python.absolute()
    local_llm_model = local_llm_model.resolve()
    local_embedding_model = local_embedding_model.resolve()
    if not runtime_python.is_file() or not os.access(runtime_python, os.X_OK):
        raise OfficialHippoRAGQualificationError("runtime Python is unavailable")
    if not local_llm_model.is_dir() or not local_embedding_model.is_dir():
        raise OfficialHippoRAGQualificationError("offline model assets are unavailable")

    with tempfile.TemporaryDirectory(prefix="official-hipporag-probe-") as directory:
        root = Path(directory)
        probe_input_path = root / "core_input.json"
        probe_input = {
            "schema": CORE_INPUT_SCHEMA,
            "documents": list(documents),
            "queries": list(queries),
        }
        probe_input_path.write_text(
            json.dumps(probe_input, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        home = root / "home"
        home.mkdir()
        cache = root / "cache"
        cache.mkdir()
        runtime_tmp = root / "tmp"
        runtime_tmp.mkdir()
        trace_prefix = root / "network.trace"
        environment = {
            "PATH": f"{runtime_python.parent}:/usr/bin:/bin",
            "HOME": str(home),
            "HF_HOME": str(cache),
            "TMPDIR": str(runtime_tmp),
            "TMP": str(runtime_tmp),
            "TEMP": str(runtime_tmp),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "TOKENIZERS_PARALLELISM": "false",
        }
        command = [
            "/usr/bin/strace",
            "-ff",
            "-qq",
            "-e",
            "trace=network",
            "-o",
            str(trace_prefix),
            "/usr/bin/bwrap",
            "--unshare-net",
            "--die-with-parent",
            "--new-session",
            "--ro-bind",
            "/",
            "/",
            "--dev",
            "/dev",
            "--bind",
            str(root),
            str(root),
            str(runtime_python),
            "-I",
            "-c",
            _CORE_PROBE_PROGRAM,
            str(probe_input_path),
            str(local_llm_model),
            str(local_embedding_model),
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=timeout_seconds,
        )
        trace = _trace_summary(trace_prefix)
        if completed.returncode != 0:
            exception_markers = sorted(
                set(
                    re.findall(
                        r"([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception))(?=:)",
                        completed.stderr,
                    )
                )
            )
            raise OfficialHippoRAGQualificationError(
                "official core probe failed; "
                f"returncode={completed.returncode}; "
                f"exception_markers={exception_markers}; "
                f"stderr_sha256={_sha256_bytes(completed.stderr.encode('utf-8'))}"
            )
        core = _parse_last_json_object(completed.stdout)

    source_files = source_binding.get("source_file_sha256")
    if not isinstance(source_files, Mapping):
        raise OfficialHippoRAGQualificationError("source file binding is missing")
    if core.get("installed_init_sha256") != source_files.get("src/hipporag/__init__.py"):
        raise OfficialHippoRAGQualificationError("installed package init differs from official source")
    if core.get("installed_core_sha256") != source_files.get("src/hipporag/HippoRAG.py"):
        raise OfficialHippoRAGQualificationError("installed core differs from official source")
    installed_python_tree_matches = bool(
        core.get("installed_python_source_file_count")
        == source_binding.get("python_source_file_count")
        and core.get("installed_python_source_tree_sha256")
        == source_binding.get("python_source_tree_sha256")
    )
    if not installed_python_tree_matches:
        raise OfficialHippoRAGQualificationError(
            "installed Python source tree differs from official commit"
        )
    parent_network_namespace = os.readlink("/proc/self/ns/net")
    network_namespace_isolated = core.get("network_namespace") != parent_network_namespace
    if not network_namespace_isolated:
        raise PermissionError("official core did not run in an isolated network namespace")
    return {
        "status": core.get("status"),
        "core_class_module": core.get("core_class_module"),
        "config_class_module": core.get("config_class_module"),
        "document_count": core.get("document_count"),
        "query_count": core.get("query_count"),
        "retrieval_result_count": core.get("retrieval_result_count"),
        "retrieved_document_counts": core.get("retrieved_document_counts"),
        "qa_result_count": core.get("qa_result_count"),
        "qa_answer_sha256": core.get("qa_answer_sha256"),
        "runtime_versions": core.get("runtime_versions"),
        "installed_python_source_tree_matches_commit": True,
        "network_namespace_isolated": True,
        "stdout_sha256": _sha256_bytes(completed.stdout.encode("utf-8")),
        "stderr_sha256": _sha256_bytes(completed.stderr.encode("utf-8")),
        "network_trace": trace,
    }


def build_qualification_receipt(
    *,
    official_repository: Path,
    runtime_python: Path,
    local_llm_model: Path,
    local_embedding_model: Path,
) -> dict[str, Any]:
    source_binding = official_source_binding(official_repository)
    fixture = build_synthetic_fixture()
    documents = documents_from_custom_json(fixture)
    questions = questions_from_custom_json(fixture)
    with tempfile.TemporaryDirectory(prefix="hipporag-adapter-") as directory:
        answer_path = Path(directory) / "answer.json"
        predictions = {
            str(row["question_id"]): str(row["source_aliases"][0])
            for row in questions
        }
        answer_sha256 = write_answer_json(
            answer_path,
            question_ids=[str(row["question_id"]) for row in questions],
            predictions=predictions,
        )
        answer_payload = json.loads(answer_path.read_text(encoding="utf-8"))
        answer_mode = stat.S_IMODE(answer_path.stat().st_mode)
        core = run_official_core_probe(
            runtime_python=runtime_python,
            local_llm_model=local_llm_model,
            local_embedding_model=local_embedding_model,
            documents=documents,
            queries=[str(row["query"]) for row in questions],
            source_binding=source_binding,
        )

    answer_shape_valid = bool(
        answer_payload.get("schema") == ANSWER_SCHEMA
        and isinstance(answer_payload.get("answers"), list)
        and len(answer_payload["answers"]) == len(questions)
        and all(set(row) == {"question_id", "answer"} for row in answer_payload["answers"])
        and answer_mode == 0o600
    )
    runtime_versions = core.get("runtime_versions")
    if not isinstance(runtime_versions, Mapping):
        raise OfficialHippoRAGQualificationError("runtime versions are missing")
    local_model_files = [
        {
            "path": str(path.relative_to(local_llm_model)),
            "sha256": _sha256_file(path),
        }
        for path in local_llm_model.rglob("*")
        if path.is_file()
    ]
    local_model_files.sort(key=lambda row: row["path"])
    receipt: dict[str, Any] = {
        "schema": QUALIFICATION_VERSION,
        "decision": "qualified_non_scoring_runtime_adapter",
        "qualified": bool(
            answer_shape_valid
            and core.get("status") == "passed"
            and core.get("installed_python_source_tree_matches_commit") is True
            and core.get("network_namespace_isolated") is True
        ),
        "scope": {
            "benchmark_performance_claim": False,
            "homologous_baseline_claim": False,
            "adapter_contract_only": True,
            "official_core_path_exercised": ["import", "initialize", "index", "retrieve", "qa"],
        },
        "source_binding": source_binding,
        "synthetic_fixture": {
            "schema": fixture["schema"],
            "fixture_sha256": stable_hash(fixture),
            "document_count": len(documents),
            "documents_sha256": stable_hash(documents),
            "question_count": len(questions),
            "queries_sha256": stable_hash(tuple(row["query"] for row in questions)),
            "normalized_multi_answers_sha256": stable_hash(
                tuple(row["normalized_aliases"] for row in questions)
            ),
            "external_rows": 0,
        },
        "adapter": {
            "json_to_documents": "passed",
            "hierarchy_preserved": all("Hierarchy:" in document for document in documents),
            "multi_answer_normalization": "passed",
            "answer_json_writer": "passed" if answer_shape_valid else "failed",
            "answer_json_sha256": answer_sha256,
            "answer_json_mode": oct(answer_mode),
            "answer_json_contains_predictions_only": answer_shape_valid,
        },
        "official_core": core,
        "dependency_boundary": {
            "runtime_kind": "dedicated_overlay_venv",
            "official_declared_openai_pin": "1.91.1",
            "runtime_openai_version": runtime_versions.get("openai"),
            "declared_openai_pin_satisfied": runtime_versions.get("openai") == "1.91.1",
            "official_declared_vllm_pin": "0.6.6.post1",
            "runtime_vllm_version": runtime_versions.get("vllm"),
            "vllm_required_for_exercised_path": False,
            "local_llm_asset_sha256": stable_hash(local_model_files),
            "local_embedding_asset_sha256": stable_hash(
                sorted(
                    (
                        str(path.relative_to(local_embedding_model)),
                        _sha256_file(path),
                    )
                    for path in local_embedding_model.rglob("*")
                    if path.is_file()
                )
            ),
        },
        "safety": {
            "external_benchmark_rows_read": 0,
            "external_result_rows_read": 0,
            "api_credentials_forwarded": False,
            "online_model_calls": 0,
            "online_evaluator_calls": 0,
            "network_namespace_isolated": core.get("network_namespace_isolated"),
            "external_network_transport_possible": False,
            "external_connect_attempt_lines": core.get("network_trace", {}).get(
                "external_connect_attempt_line_count"
            ),
            "official_source_modified": False,
            "import_stubs_used": False,
            "monkeypatches_used": False,
            "performance_gate_created": False,
        },
        "limitations": [
            "This qualifies adapter and runtime execution only; it is not a benchmark comparison.",
            "The tiny local causal model is used only to traverse official code paths; answer quality is not evaluated.",
            "The official openai==1.91.1 declaration is not satisfied by the 1.91.0 runtime overlay.",
            "vLLM is not installed because the exercised Transformers path imports it lazily and does not require it.",
        ],
    }
    receipt["qualification_sha256"] = stable_hash(receipt)
    return receipt


def _write_receipt(path: Path, payload: Mapping[str, Any]) -> None:
    raw = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-repository", required=True, type=Path)
    parser.add_argument("--runtime-python", required=True, type=Path)
    parser.add_argument("--local-llm-model", required=True, type=Path)
    parser.add_argument("--local-embedding-model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    receipt = build_qualification_receipt(
        official_repository=arguments.official_repository,
        runtime_python=arguments.runtime_python,
        local_llm_model=arguments.local_llm_model,
        local_embedding_model=arguments.local_embedding_model,
    )
    _write_receipt(arguments.output, receipt)
    print(
        json.dumps(
            {
                "qualified": receipt["qualified"],
                "qualification_sha256": receipt["qualification_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0 if receipt["qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
