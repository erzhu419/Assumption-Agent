from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import stat
import subprocess
from types import ModuleType, SimpleNamespace
import sys
import sysconfig
import tempfile
import threading

import numpy as np
import pytest

from assumption_agent.benchmarks import hitab_p1_public_canary_v1 as canary
from replication_runtime.birco_official_hipporag_v1 import (
    contract as hippo_contract,
)
from replication_runtime.bright_query_generator_v1 import (
    contract as planner_contract,
)
from replication_runtime.hitab_p1_formal_v1 import runner
from replication_runtime.hitab_p1_formal_v1 import dependency_closure


_REAL_CURRENT_HIPPO_ATTESTATION_LINKAGE = (
    runner._validate_current_hipporag_attestation_linkage
)
_SYNTHETIC_LEGACY_HIPPO_ROOT = Path(
    "/synthetic/legacy/HippoRAG"
)
_SYNTHETIC_CLEAN_SOURCE_BYTES = b"# synthetic clean HippoRAG source\n"
_SYNTHETIC_CLEAN_SOURCE_TREE_SHA256 = runner.stable_hash(
    [
        {
            "path": "src/hipporag/__init__.py",
            "sha256": hashlib.sha256(
                _SYNTHETIC_CLEAN_SOURCE_BYTES
            ).hexdigest(),
            "size_bytes": len(_SYNTHETIC_CLEAN_SOURCE_BYTES),
        }
    ]
)


@pytest.fixture(autouse=True)
def _synthetic_current_attestation_linkage(monkeypatch):
    """Synthetic closure tests do not claim the production P17 asset trees."""

    monkeypatch.setattr(
        runner,
        "_validate_current_hipporag_attestation_linkage",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runner, "EXPECTED_HIPPORAG_SOURCE_CLEAN_FILE_COUNT", 1
    )
    monkeypatch.setattr(
        runner,
        "EXPECTED_HIPPORAG_SOURCE_CLEAN_SIZE_BYTES",
        len(_SYNTHETIC_CLEAN_SOURCE_BYTES),
    )
    monkeypatch.setattr(
        runner,
        "EXPECTED_HIPPORAG_SOURCE_CLEAN_TREE_SHA256",
        _SYNTHETIC_CLEAN_SOURCE_TREE_SHA256,
    )
    monkeypatch.setattr(
        runner,
        "_validate_reusable_hipporag_attestation",
        lambda *_args, **_kwargs: {
            "embedding_tree_sha256": "1" * 64,
            "hipporag_origin_file_sha256": "2" * 64,
            "hipporag_origin_path": str(
                _SYNTHETIC_LEGACY_HIPPO_ROOT
                / "src/hipporag/__init__.py"
            ),
            "llm_tree_sha256": "3" * 64,
            "source_tree_sha256": "4" * 64,
        },
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(runner.canonical_bytes(value))


def _self_hashed(body: dict[str, object]) -> dict[str, object]:
    value = dict(body)
    value["self_sha256"] = runner.stable_hash(value)
    return value


def _unit_text(*, mode: str, project: Path, python: Path) -> str:
    common = (
        "[Unit]\n"
        f"Description=synthetic HiTab {mode} unit\n\n"
        "[Service]\n"
        "Type=oneshot\n"
        f"WorkingDirectory={project}\n"
        "UMask=0077\n"
    )
    if mode == "canary":
        environment = dict(runner._COMMON_OFFLINE_ENVIRONMENT)
        environment["PYTHONPATH"] = str(project)
        arguments = [
            "canary",
            "--implementation-freeze",
            str(
                project
                / "manifests/hitab_p1_implementation_freeze_v1.json"
            ),
            "--output",
            str(project.parent / "receipts/source_free_canary.json"),
        ]
        network = (
            "RestrictAddressFamilies=AF_UNIX\n"
            "IPAddressDeny=any\n"
        )
    elif mode == "formal":
        environment = dict(runner._COMMON_OFFLINE_ENVIRONMENT)
        environment["PYTHONPATH"] = str(project)
        arguments = [
            "formal",
            "--execution-freeze",
            str(project / "manifests/hitab_p1_execution_freeze_v1.json"),
        ]
        network = (
            "RestrictAddressFamilies=AF_UNIX\n"
            "IPAddressDeny=any\n"
        )
    else:
        environment = dict(runner._ACQUISITION_ENVIRONMENT)
        environment["PYTHONPATH"] = str(project)
        arguments = [
            "acquire",
            "--acquisition-freeze",
            str(
                project
                / "manifests/hitab_p1_source_acquisition_freeze_v1.json"
            ),
        ]
        network = "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6\n"
    command_rows = [
        "/usr/bin/env",
        "-i",
        *(f"{key}={value}" for key, value in sorted(environment.items())),
        str(python),
        "-S",
        "-B",
        "-c",
        runner.OUTER_ENTRYPOINT_SCRIPT,
        *arguments,
    ]
    command = "ExecStart=" + shlex.join(command_rows) + "\n"
    return (
        common
        + command
        + "CPUQuota=800%\n"
        + "MemoryMax=40G\n"
        + "TasksMax=64\n"
        + "KillMode=control-group\n"
        + "Restart=no\n"
        + network
        + "TimeoutStartSec=infinity\n"
    )


def _runtime_binding(
    tmp_path: Path,
    *,
    role: str,
    project: Path,
    files_by_label: dict[str, Path],
    required_distributions: dict[str, tuple[str, str]],
    required_project_imports: dict[str, str],
) -> tuple[dict[str, object], dict[str, object]]:
    venv = tmp_path / "runtimes" / role
    executable = venv / "bin" / "python"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(f"synthetic-{role}-python\n".encode("ascii"))
    executable.chmod(0o755)
    pyvenv = venv / "pyvenv.cfg"
    pyvenv.write_text("version = 3.10.12\n", encoding="ascii")
    stdlib = venv / "lib" / "python3.10"
    stdlib.mkdir(parents=True)
    dependency_root = tmp_path / "dependencies" / role
    dependency_root.mkdir(parents=True)

    probe: dict[str, object] = {}
    for module, (distribution, version) in sorted(
        required_distributions.items()
    ):
        origin = (
            dependency_root
            / module.replace(".", "/")
            / "__init__.py"
        )
        origin.parent.mkdir(parents=True, exist_ok=True)
        origin.write_text(
            f"# synthetic {module} {version}\n", encoding="ascii"
        )
        probe[module] = {
            "distribution": distribution,
            "origin_path": str(origin),
            "origin_receipt": dependency_closure.regular_file_receipt(origin),
            "version": version,
        }
    for module, label in sorted(required_project_imports.items()):
        origin = files_by_label[label]
        probe[module] = {
            "distribution": None,
            "origin_path": str(origin),
            "origin_receipt": dependency_closure.regular_file_receipt(origin),
            "version": None,
        }
    for module in sorted(runner.REQUIRED_STDLIB_IMPORTS):
        origin = stdlib / f"{module}.py"
        origin.write_text(f"# synthetic stdlib {module}\n", encoding="ascii")
        probe[module] = {
            "distribution": None,
            "origin_path": str(origin),
            "origin_receipt": dependency_closure.regular_file_receipt(origin),
            "version": None,
        }
    (stdlib / "pathlib.py").write_text(
        "# synthetic unprobed stdlib transitive\n", encoding="ascii"
    )
    python_binding = {
        "executable_path": str(executable),
        "lexical_symlink_target": None,
        "pyvenv_cfg": {
            "path": str(pyvenv),
            "receipt": dependency_closure.regular_file_receipt(pyvenv),
        },
        "python_version": "3.10.12",
        "resolved_target": {
            "path": str(executable),
            "receipt": dependency_closure.regular_file_receipt(executable),
        },
        "stdlib_root": {
            "path": str(stdlib),
            "tree_receipt": dependency_closure.tree_receipt(stdlib),
        },
    }
    dependency_binding = {
        "import_probe": probe,
        "ordered_roots": [
            {
                "path": str(dependency_root),
                "tree_receipt": dependency_closure.tree_receipt(
                    dependency_root
                ),
            }
        ],
    }
    return python_binding, dependency_binding


def _implementation_freeze(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    project = tmp_path / "study/formal_v1/reconstruction_v2"
    project.mkdir(parents=True)
    source_project = Path(runner.__file__).resolve().parents[2]
    files: dict[str, object] = {}
    files_by_label: dict[str, Path] = {}
    outer_executable = (
        tmp_path / "runtimes" / "outer" / "bin" / "python"
    )
    for label, relative in sorted(runner.REQUIRED_PROJECT_FILES.items()):
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if label == "hitab_canary_unit":
            path.write_text(
                _unit_text(
                    mode="canary",
                    project=project,
                    python=outer_executable,
                ),
                encoding="ascii",
            )
        elif label == "hitab_acquire_unit":
            path.write_text(
                _unit_text(
                    mode="acquire",
                    project=project,
                    python=outer_executable,
                ),
                encoding="ascii",
            )
        elif label == "hitab_formal_unit":
            path.write_text(
                _unit_text(
                    mode="formal",
                    project=project,
                    python=outer_executable,
                ),
                encoding="ascii",
            )
        else:
            path.write_bytes((source_project / relative).read_bytes())
        files_by_label[label] = path
        files[label] = {
            "relative_path": relative,
            "sha256": runner.file_sha256(path),
        }
    models: dict[str, object] = {}
    for label in sorted(runner.REQUIRED_MODEL_LABELS):
        root = tmp_path / "models" / label
        root.mkdir(parents=True)
        (root / "weights.bin").write_bytes(
            hashlib.sha256(label.encode("ascii")).digest()
        )
        models[label] = {
            "path": str(root),
            "tree_sha256": runner.model_tree_sha256(root),
        }
    minilm_manifest = tmp_path / "assets" / "minilm.json"
    minilm_manifest.parent.mkdir()
    minilm_manifest.write_bytes(b'{"synthetic":true}\n')
    outer_python, outer_dependency = _runtime_binding(
        tmp_path,
        role="outer",
        project=project,
        files_by_label=files_by_label,
        required_distributions=runner.OUTER_REQUIRED_DISTRIBUTIONS,
        required_project_imports=runner.OUTER_REQUIRED_PROJECT_IMPORTS,
    )
    hippo_python, hippo_dependency = _runtime_binding(
        tmp_path,
        role="hippo_child",
        project=project,
        files_by_label=files_by_label,
        required_distributions=runner.HIPPORAG_REQUIRED_DISTRIBUTIONS,
        required_project_imports=runner.HIPPORAG_REQUIRED_PROJECT_IMPORTS,
    )
    clean_source_root = (
        project.parent.parent / "runtime/hipporag_clean/HippoRAG"
    )
    clean_source_file = clean_source_root / "src/hipporag/__init__.py"
    clean_source_file.parent.mkdir(parents=True)
    clean_source_file.write_bytes(_SYNTHETIC_CLEAN_SOURCE_BYTES)
    clean_source_files = tuple(
        path
        for path in clean_source_root.rglob("*")
        if path.is_file()
    )
    body: dict[str, object] = {
        "dependency_closure": {
            "hippo_child": hippo_dependency,
            "outer": outer_dependency,
        },
        "files": files,
        "hippo_worker": {
            "file_label": "hippo_worker",
            "module": runner.HIPPORAG_WORKER_MODULE,
        },
        "implementation_revision": runner.IMPLEMENTATION_REVISION,
        "hippo_source_projection": {
            "clean_root": str(clean_source_root),
            "file_count": len(clean_source_files),
            "legacy_attested_root": str(
                _SYNTHETIC_LEGACY_HIPPO_ROOT
            ),
            "projection_policy": (
                runner.HIPPORAG_SOURCE_PROJECTION_POLICY
            ),
            "size_bytes": sum(
                path.stat().st_size for path in clean_source_files
            ),
            "tree_receipt": dependency_closure.tree_receipt(
                clean_source_root
            ),
            "tree_sha256": runner.model_tree_sha256(
                clean_source_root
            ),
        },
        "minilm_asset_manifest": {
            "path": str(minilm_manifest),
            "sha256": runner.file_sha256(minilm_manifest),
        },
        "models": models,
        "project_root": str(project),
        "python": {
            "hippo_child": hippo_python,
            "outer": outer_python,
        },
        "runtime_policy": runner.RUNTIME_POLICY,
        "schema": runner.IMPLEMENTATION_FREEZE_SCHEMA,
        "study_id": runner.STUDY_ID,
    }
    value = _self_hashed(body)
    path = tmp_path / "implementation.freeze.json"
    _write_json(path, value)
    return path, value


class _Planner:
    def __call__(self, canonical_input: bytes) -> bytes:
        item = planner_contract.parse_input(canonical_input)[0]
        completion = json.dumps(
            {
                "constraint_query": "use the synthetic 2024 cells",
                "entity_query": "North and South synthetic regions",
                "mechanism_query": "identify the larger percentage",
                "relation_query": "compare displayed renewable shares",
            },
            ensure_ascii=True,
            separators=(",", ":"),
        )
        row = planner_contract.build_output_item(
            ordinal=0,
            completion=completion,
            completion_token_count=24,
            query=item.query,
        )
        return planner_contract.canonical_json_bytes(
            planner_contract.output_payload((row,))
        )


class _Scorer:
    def __call__(self, pairs):
        return tuple((index % 17) / 7.0 for index, _row in enumerate(pairs))


def _vector(text: str) -> np.ndarray:
    raw = hashlib.sha256(text.encode("utf-8")).digest() * 12
    row = np.frombuffer(raw, dtype=np.uint8)[:384].astype(np.float32)
    row -= np.float32(127.5)
    row /= np.float32(np.linalg.norm(row.astype(np.float64)))
    return row


class _Encoder:
    def encode(self, texts):
        return np.stack([_vector(text) for text in texts]).astype(np.float32)


class _SyntheticHippo:
    def __init__(self) -> None:
        self.barrier = threading.Barrier(2, timeout=10)

    def __call__(
        self,
        canonical_input: bytes,
        *,
        physical_gpu: int,
        cpu_thread_limit: int,
        launch_ack,
    ) -> bytes:
        assert physical_gpu in {0, 1}
        assert cpu_thread_limit == 4
        value = json.loads(canonical_input.decode("ascii"))
        checked = hippo_contract.validate_input(
            value["work_id"],
            value["objective"],
            value["query"],
            value["documents"],
            value["common_projection_sha256"],
        )
        launch_ack()
        self.barrier.wait()
        return hippo_contract.canonical_json_bytes(
            hippo_contract.output_payload(
                work_id=checked[0],
                common_projection_sha256=checked[4],
                candidate_count=len(checked[3]),
                rank_ordinals=tuple(range(len(checked[3]))),
                graph_nodes=12,
                graph_edges=11,
            )
        )


def _synthetic_bindings(
    _implementation: runner.FrozenImplementation, _root: Path
) -> runner.ProductionBindings:
    cache_body: dict[str, object] = {
        "model_offload_or_reload": False,
        "physical_gpu": 0,
        "schema": "hitab_p1_gpu0_unused_cuda_cache_release_v1",
        "study_id": runner.STUDY_ID,
        "torch_cuda_empty_cache_called": True,
    }

    def release_cache() -> dict[str, object]:
        return _self_hashed(cache_body)

    return runner.ProductionBindings(
        planner_runner=_Planner(),
        cross_encoder_scorer=_Scorer(),
        minilm_encoder=_Encoder(),
        hippo_runner=_SyntheticHippo(),
        gpu0_cache_releaser=release_cache,
    )


def _noop_runtime_preparer(
    _implementation: runner.FrozenImplementation,
    *,
    verify_hippo_child: bool,
) -> dict[str, object]:
    return {"synthetic": True, "verify_hippo_child": verify_hippo_child}


def test_freeze_verifies_files_models_worker_python_and_exact_policy(
    tmp_path: Path,
) -> None:
    path, _value = _implementation_freeze(tmp_path)
    frozen = runner.load_implementation_freeze(path)
    assert set(frozen.files) == runner.REQUIRED_FILE_LABELS
    assert set(frozen.models) == runner.REQUIRED_MODEL_LABELS
    assert frozen.runtime_policy == runner.RUNTIME_POLICY
    assert (
        frozen.outer_runtime.executable
        != frozen.hippo_runtime.executable
    )

    (frozen.models["planner"] / "weights.bin").write_bytes(b"drift")
    with pytest.raises(
        runner.HitabP1ProductionRuntimeError, match="model tree planner drifted"
    ):
        runner.load_implementation_freeze(path)


def test_freeze_accepts_lexical_outer_symlink_to_detached_base_runtime(
    tmp_path: Path,
) -> None:
    path, value = _implementation_freeze(tmp_path)
    project = Path(value["project_root"])
    original = Path(
        value["python"]["outer"]["executable_path"]
    )
    lexical = tmp_path / "study-runtime" / "outer_venv" / "bin" / "python"
    lexical.parent.mkdir(parents=True)
    lexical.symlink_to(original)
    pyvenv = lexical.parent.parent / "pyvenv.cfg"
    pyvenv.write_text(
        f"home = {original.parent}\n"
        "include-system-site-packages = false\n"
        "version = 3.10.12\n",
        encoding="ascii",
    )

    outer = value["python"]["outer"]
    outer["executable_path"] = str(lexical)
    outer["lexical_symlink_target"] = str(original)
    outer["pyvenv_cfg"] = {
        "path": str(pyvenv),
        "receipt": dependency_closure.regular_file_receipt(pyvenv),
    }
    for label in (
        "hitab_acquire_unit",
        "hitab_canary_unit",
        "hitab_formal_unit",
    ):
        unit = project / runner.REQUIRED_PROJECT_FILES[label]
        text = unit.read_text(encoding="ascii")
        assert str(original) in text
        unit.write_text(
            text.replace(str(original), str(lexical)),
            encoding="ascii",
        )
        value["files"][label]["sha256"] = runner.file_sha256(unit)
    value.pop("self_sha256")
    value = _self_hashed(value)
    _write_json(path, value)

    frozen = runner.load_implementation_freeze(path)
    assert frozen.outer_runtime.executable == lexical
    assert frozen.outer_runtime.resolved_target == original
    assert frozen.outer_runtime.stdlib_root == original.parent.parent / (
        "lib/python3.10"
    )


def test_freeze_rejects_nonlocal_or_drifted_full_source_projection(
    tmp_path: Path,
) -> None:
    nonlocal_path, nonlocal_value = _implementation_freeze(
        tmp_path / "nonlocal"
    )
    nonlocal_value["hippo_source_projection"]["clean_root"] = str(
        tmp_path / "outside-study/HippoRAG"
    )
    nonlocal_value.pop("self_sha256")
    _write_json(nonlocal_path, _self_hashed(nonlocal_value))
    with pytest.raises(
        runner.HitabP1ProductionRuntimeError,
        match="exact study-local mapping",
    ):
        runner.load_implementation_freeze(nonlocal_path)

    drift_path, drift_value = _implementation_freeze(tmp_path / "drift")
    clean_root = Path(
        drift_value["hippo_source_projection"]["clean_root"]
    )
    (clean_root / "src/hipporag/non_origin.py").write_text(
        "DRIFT = True\n", encoding="ascii"
    )
    with pytest.raises(
        runner.HitabP1ProductionRuntimeError,
        match="full clean source closure drifted",
    ):
        runner.load_implementation_freeze(drift_path)

    empty_cache_path, empty_cache_value = _implementation_freeze(
        tmp_path / "empty-cache"
    )
    empty_cache_root = Path(
        empty_cache_value["hippo_source_projection"]["clean_root"]
    )
    (empty_cache_root / "src/hipporag/__pycache__").mkdir()
    empty_cache_value["hippo_source_projection"]["tree_receipt"] = (
        dependency_closure.tree_receipt(empty_cache_root)
    )
    empty_cache_value.pop("self_sha256")
    _write_json(
        empty_cache_path,
        _self_hashed(empty_cache_value),
    )
    with pytest.raises(
        runner.HitabP1ProductionRuntimeError,
        match="full clean source closure drifted",
    ):
        runner.load_implementation_freeze(empty_cache_path)


def test_committed_units_use_study_local_outer_runtime_symlink() -> None:
    project = Path(runner.__file__).resolve().parents[2]
    expected = (
        "/home/erzhu419/hitab_p1_20260726/runtime/"
        "outer_venv/bin/python"
    )
    old_copied_interpreter = (
        "/home/erzhu419/p19_runtime_assets_20260723/"
        "typed_venv/bin/python"
    )
    for label in (
        "hitab_acquire_unit",
        "hitab_canary_unit",
        "hitab_formal_unit",
    ):
        unit = project / runner.REQUIRED_PROJECT_FILES[label]
        text = unit.read_text(encoding="ascii")
        assert expected in text
        assert old_copied_interpreter not in text


def test_freeze_rejects_stdlib_zip_structural_pyvenv_and_project_bytecode(
    tmp_path: Path,
) -> None:
    stdlib_path, stdlib_value = _implementation_freeze(
        tmp_path / "stdlib"
    )
    stdlib_root = Path(
        stdlib_value["python"]["outer"]["stdlib_root"]["path"]
    )
    (stdlib_root / "pathlib.py").write_text(
        "# drifted unprobed stdlib bytes\n", encoding="ascii"
    )
    with pytest.raises(
        runner.HitabP1ProductionRuntimeError,
        match="outer stdlib tree drifted",
    ):
        runner.load_implementation_freeze(stdlib_path)

    zip_path, zip_value = _implementation_freeze(tmp_path / "zip")
    zip_root = Path(
        zip_value["python"]["outer"]["stdlib_root"]["path"]
    )
    (zip_root.parent / "python310.zip").write_bytes(b"shadow")
    with pytest.raises(
        runner.HitabP1ProductionRuntimeError,
        match="automatic Python zip path drifted",
    ):
        runner.load_implementation_freeze(zip_path)

    pyvenv_path, pyvenv_value = _implementation_freeze(
        tmp_path / "pyvenv"
    )
    replacement = tmp_path / "pyvenv" / "other-pyvenv.cfg"
    replacement.write_text("version = 3.10.12\n", encoding="ascii")
    pyvenv_value["python"]["outer"]["pyvenv_cfg"] = {
        "path": str(replacement),
        "receipt": dependency_closure.regular_file_receipt(replacement),
    }
    pyvenv_value.pop("self_sha256")
    pyvenv_value = _self_hashed(pyvenv_value)
    _write_json(pyvenv_path, pyvenv_value)
    with pytest.raises(
        runner.HitabP1ProductionRuntimeError,
        match="structural pyvenv binding drifted",
    ):
        runner.load_implementation_freeze(pyvenv_path)

    bytecode_path, bytecode_value = _implementation_freeze(
        tmp_path / "bytecode"
    )
    project = Path(bytecode_value["project_root"])
    cache = project / "__pycache__"
    cache.mkdir()
    (cache / "shadow.pyc").write_bytes(b"unbound")
    with pytest.raises(
        runner.HitabP1ProductionRuntimeError,
        match="unbound Python bytecode",
    ):
        runner.load_implementation_freeze(bytecode_path)


def test_current_hipporag_assets_must_equal_reused_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "HippoRAG"
    origin = source_root / "src" / "hipporag" / "__init__.py"
    origin.parent.mkdir(parents=True)
    origin.write_text("# attested synthetic HippoRAG\n", encoding="ascii")
    source_hash = runner.model_tree_sha256(source_root)
    origin_hash = runner.file_sha256(origin)
    monkeypatch.setattr(
        runner, "EXPECTED_HIPPORAG_SOURCE_CLEAN_FILE_COUNT", 1
    )
    monkeypatch.setattr(
        runner,
        "EXPECTED_HIPPORAG_SOURCE_CLEAN_SIZE_BYTES",
        origin.stat().st_size,
    )
    monkeypatch.setattr(
        runner,
        "EXPECTED_HIPPORAG_SOURCE_CLEAN_TREE_SHA256",
        source_hash,
    )
    runtime = SimpleNamespace(
        import_probe={
            "hipporag": {
                "origin_path": str(origin),
                "origin_receipt": {"content_sha256": origin_hash},
            }
        },
        ordered_roots=(source_root / "src",),
    )
    attestation = {
        "embedding_tree_sha256": "1" * 64,
        "hipporag_origin_file_sha256": origin_hash,
        "hipporag_origin_path": str(
            tmp_path / "legacy-runtime/HippoRAG/src/hipporag/__init__.py"
        ),
        "llm_tree_sha256": "2" * 64,
        "source_tree_sha256": "3" * 64,
    }
    _REAL_CURRENT_HIPPO_ATTESTATION_LINKAGE(
        attestation,
        runtime=runtime,
        model_hashes={
            "hippo_embedding": "1" * 64,
            "hippo_llm": "2" * 64,
        },
        clean_source_root=source_root,
    )
    with pytest.raises(
        runner.HitabP1ProductionRuntimeError,
        match="source or model binding is not attested",
    ):
        _REAL_CURRENT_HIPPO_ATTESTATION_LINKAGE(
            attestation,
            runtime=runtime,
            model_hashes={
                "hippo_embedding": "1" * 64,
                "hippo_llm": "3" * 64,
            },
            clean_source_root=source_root,
        )

    cache = origin.parent / "__pycache__"
    cache.mkdir()
    (cache / "__init__.cpython-310.pyc").write_bytes(b"nonportable")
    monkeypatch.setattr(
        runner, "EXPECTED_HIPPORAG_SOURCE_CLEAN_FILE_COUNT", 2
    )
    monkeypatch.setattr(
        runner,
        "EXPECTED_HIPPORAG_SOURCE_CLEAN_SIZE_BYTES",
        origin.stat().st_size + len(b"nonportable"),
    )
    monkeypatch.setattr(
        runner,
        "EXPECTED_HIPPORAG_SOURCE_CLEAN_TREE_SHA256",
        runner.model_tree_sha256(source_root),
    )
    with pytest.raises(
        runner.HitabP1ProductionRuntimeError,
        match="frozen clean projection",
    ):
        _REAL_CURRENT_HIPPO_ATTESTATION_LINKAGE(
            attestation,
            runtime=runtime,
            model_hashes={
                "hippo_embedding": "1" * 64,
                "hippo_llm": "2" * 64,
            },
            clean_source_root=source_root,
        )

    (cache / "__init__.cpython-310.pyc").unlink()
    cache.rmdir()
    alias = origin.parent / "hardlinked_alias.py"
    os.link(origin, alias)
    monkeypatch.setattr(
        runner, "EXPECTED_HIPPORAG_SOURCE_CLEAN_FILE_COUNT", 2
    )
    monkeypatch.setattr(
        runner,
        "EXPECTED_HIPPORAG_SOURCE_CLEAN_SIZE_BYTES",
        2 * origin.stat().st_size,
    )
    monkeypatch.setattr(
        runner,
        "EXPECTED_HIPPORAG_SOURCE_CLEAN_TREE_SHA256",
        runner.model_tree_sha256(source_root),
    )
    with pytest.raises(
        runner.HitabP1ProductionRuntimeError,
        match="frozen clean projection",
    ):
        _REAL_CURRENT_HIPPO_ATTESTATION_LINKAGE(
            attestation,
            runtime=runtime,
            model_hashes={
                "hippo_embedding": "1" * 64,
                "hippo_llm": "2" * 64,
            },
            clean_source_root=source_root,
        )


def test_module_cache_scan_bypasses_lazy_module_getattr() -> None:
    class LazyModule(ModuleType):
        def __getattr__(self, name: str) -> object:
            if name == "__cached__":
                raise AssertionError("lazy __cached__ access executed")
            raise AttributeError(name)

    module = LazyModule("lazy_cache_sentinel")
    assert runner._invalid_module_cache_paths((module,)) == []
    module.__dict__["__cached__"] = "/unexpected/cache.pyc"
    assert runner._invalid_module_cache_paths((module,)) == [
        "/unexpected/cache.pyc"
    ]


def test_v3_outer_excludes_sentence_transformers_and_tmp_modules() -> None:
    assert runner.IMPLEMENTATION_REVISION == (
        "direct_transformers_minilm_v3_child_cwd_sanitized"
    )
    assert (
        "sentence_transformers"
        not in runner.OUTER_REQUIRED_DISTRIBUTIONS
    )
    assert (
        runner.HIPPORAG_REQUIRED_DISTRIBUTIONS[
            "sentence_transformers"
        ]
        == ("sentence-transformers", "3.1.1")
    )

    regular = ModuleType("tmp_regular")
    regular.__file__ = "/tmp/frozen-bypass/module.py"
    package = ModuleType("tmp_package")
    package.__path__ = ["/var/tmp/frozen-bypass/package"]
    assert runner._temporary_module_paths((regular, package)) == [
        "/tmp/frozen-bypass/module.py",
        "/var/tmp/frozen-bypass/package",
    ]


def test_child_probe_and_bootstrap_remove_every_cwd_alias(
    tmp_path: Path,
) -> None:
    model_cwd = tmp_path / "model-cwd"
    model_cwd.mkdir()
    model_cwd_alias = tmp_path / "model-cwd-alias"
    model_cwd_alias.symlink_to(model_cwd, target_is_directory=True)
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPYCACHEPREFIX": "/dev/null",
            "PYTHONPATH": os.pathsep.join(
                [str(model_cwd), ".", str(model_cwd_alias)]
            ),
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-S",
            "-B",
            "-c",
            runner._IMPORT_PROBE_SCRIPT,
            json.dumps({"json": None}, separators=(",", ":")),
        ],
        cwd=model_cwd,
        check=False,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode == 0, completed.stderr.decode(
        "utf-8", errors="replace"
    )
    probe_paths = json.loads(completed.stdout)["sys_path"]
    assert probe_paths
    assert all(Path(path).is_absolute() for path in probe_paths)
    assert all(
        Path(path).resolve() != model_cwd.resolve()
        for path in probe_paths
    )

    project = tmp_path / "project"
    project.mkdir()
    asset = tmp_path / "asset"
    asset.mkdir()
    (model_cwd / "smollm2").symlink_to(
        asset, target_is_directory=True
    )
    output = tmp_path / "bootstrap-paths.json"
    (project / "synthetic_child_worker.py").write_text(
        "import json,sys\n"
        "from pathlib import Path\n"
        "output=Path(sys.argv[1])\n"
        "expected=Path(sys.argv[2]).resolve()\n"
        "if Path('smollm2').resolve()!=expected:"
        " raise RuntimeError('relative model path drifted')\n"
        "output.write_text(json.dumps(sys.path),encoding='ascii')\n",
        encoding="ascii",
    )
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(project), str(model_cwd), ".", str(model_cwd_alias)]
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-S",
            "-B",
            "-c",
            runner._HIPPO_CHILD_BOOTSTRAP_SCRIPT,
            str(project),
            str(Path(sysconfig.get_path("stdlib")).resolve()),
            "synthetic_child_worker",
            str(output),
            str(asset),
        ],
        cwd=model_cwd,
        check=False,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode == 0, completed.stderr.decode(
        "utf-8", errors="replace"
    )
    bootstrap_paths = json.loads(output.read_text(encoding="ascii"))
    assert bootstrap_paths
    assert all(Path(path).is_absolute() for path in bootstrap_paths)
    assert all(
        Path(path).resolve() != model_cwd.resolve()
        for path in bootstrap_paths
    )


def test_child_probe_cache_scan_bypasses_lazy_module_getattr() -> None:
    prefix = (
        "import sys,types\n"
        "class LazyCacheModule(types.ModuleType):\n"
        " def __getattr__(self,name):\n"
        "  if name=='__cached__':"
        " raise RuntimeError('lazy cache access executed')\n"
        "  raise AttributeError(name)\n"
        "sys.modules['lazy_cache_sentinel']="
        "LazyCacheModule('lazy_cache_sentinel')\n"
    )
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPYCACHEPREFIX": "/dev/null",
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-S",
            "-B",
            "-c",
            prefix + runner._IMPORT_PROBE_SCRIPT,
            json.dumps({"json": None}, separators=(",", ":")),
        ],
        check=False,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode == 0, completed.stderr.decode(
        "utf-8", errors="replace"
    )
    assert json.loads(completed.stdout)["rows"]["json"]["version"] is None


def test_live_probe_rejects_unlisted_python_path(tmp_path: Path) -> None:
    freeze_path, value = _implementation_freeze(tmp_path)
    frozen = runner.load_implementation_freeze(freeze_path)
    runtime = frozen.outer_runtime
    rows = {
        module: {
            "content_sha256": expected["origin_receipt"][
                "content_sha256"
            ],
            "origin_path": expected["origin_path"],
            "version": expected["version"],
        }
        for module, expected in runtime.import_probe.items()
    }
    probe = {
        "dont_write_bytecode": True,
        "invalid_cached": [],
        "no_site": 1,
        "pycache_prefix": "/dev/null",
        "python_version": runtime.python_version,
        "resolved_executable": str(runtime.resolved_target),
        "rows": rows,
        "stdlib_root": str(runtime.stdlib_root),
        "sys_path": [
            str(frozen.project_root),
            str(runtime.stdlib_root),
            *map(str, runtime.ordered_roots),
            str(tmp_path / "unfrozen-root"),
        ],
    }
    with pytest.raises(
        runner.HitabP1ProductionRuntimeError,
        match="unfrozen Python path",
    ):
        runner._validate_live_probe(
            runtime, probe, project_root=frozen.project_root
        )


def test_outer_entrypoint_uses_one_canonical_runner_module(
    tmp_path: Path,
) -> None:
    _freeze, value = _implementation_freeze(tmp_path)
    project = Path(value["project_root"])
    environment = {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(project),
        "PYTHONPYCACHEPREFIX": "/dev/null",
    }
    completed = subprocess.run(
        [
            str(Path(sys.executable)),
            "-S",
            "-B",
            "-c",
            runner.OUTER_ENTRYPOINT_SCRIPT,
            "--help",
        ],
        cwd=project,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0
    assert b"found in sys.modules" not in completed.stderr
    assert b"RuntimeWarning" not in completed.stderr


def _hippo_input() -> bytes:
    documents = [
        {"ordinal": index, "text": f"synthetic unit {index}"}
        for index in range(10)
    ]
    objective = "Retrieve five synthetic units."
    query = "Which synthetic units are relevant?"
    projection = hippo_contract.common_projection_sha256(
        objective=objective, query=query, documents=documents
    )
    return hippo_contract.canonical_json_bytes(
        {
            "common_projection_sha256": projection,
            "documents": documents,
            "objective": objective,
            "query": query,
            "schema": hippo_contract.INPUT_SCHEMA,
            "work_id": "synthetic-work",
        }
    )


def _fresh_runner(
    tmp_path: Path, fake_run
) -> runner.HippoFreshProcessRunner:
    project = tmp_path / "project"
    project.mkdir(parents=True, exist_ok=True)
    worker = project / "worker.py"
    worker.write_text("# synthetic frozen worker\n", encoding="ascii")
    llm = tmp_path / "llm"
    embedding = tmp_path / "embedding"
    for root, value in ((llm, b"llm"), (embedding, b"embedding")):
        root.mkdir()
        (root / "weights").write_bytes(value)
    dependency_root = tmp_path / "hippo-dependencies"
    dependency_root.mkdir()
    stdlib_root = tmp_path / "hippo-stdlib"
    stdlib_root.mkdir()
    clean_source_root = tmp_path / "hippo-source/HippoRAG"
    clean_source = clean_source_root / "src/hipporag/__init__.py"
    clean_source.parent.mkdir(parents=True)
    clean_source.write_bytes(_SYNTHETIC_CLEAN_SOURCE_BYTES)
    return runner.HippoFreshProcessRunner(
        project_root=project,
        runtime_root=Path(
            tempfile.mkdtemp(
                prefix=f"hitab-hippo-{tmp_path.name}-", dir="/tmp"
            )
        ),
        python_executable=Path(sys.executable),
        dependency_roots=(dependency_root,),
        stdlib_root=stdlib_root,
        hippo_source_root=clean_source_root,
        hippo_source_tree_receipt=dependency_closure.tree_receipt(
            clean_source_root
        ),
        hippo_source_file_count=1,
        hippo_source_size_bytes=len(_SYNTHETIC_CLEAN_SOURCE_BYTES),
        hippo_source_tree_sha256=(
            _SYNTHETIC_CLEAN_SOURCE_TREE_SHA256
        ),
        worker_module=runner.HIPPORAG_WORKER_MODULE,
        worker_file=worker,
        worker_file_sha256=runner.file_sha256(worker),
        llm_model_root=llm,
        llm_model_tree_sha256=runner.model_tree_sha256(llm),
        embedding_model_root=embedding,
        embedding_model_tree_sha256=runner.model_tree_sha256(embedding),
        subprocess_runner=fake_run,
    )


def test_hippo_runner_rechecks_full_source_before_every_launch(
    tmp_path: Path,
) -> None:
    subprocess_called = False
    acknowledged = False

    def fake_run(*_args: object, **_kwargs: object) -> SimpleNamespace:
        nonlocal subprocess_called
        subprocess_called = True
        return SimpleNamespace(returncode=0)

    fresh = _fresh_runner(tmp_path, fake_run)
    source = (
        tmp_path
        / "hippo-source/HippoRAG/src/hipporag/non_origin.py"
    )
    source.write_text("DRIFT = True\n", encoding="ascii")

    def acknowledge() -> None:
        nonlocal acknowledged
        acknowledged = True

    with pytest.raises(
        runner.HitabP1ProductionRuntimeError,
        match="clean source drifted before child launch",
    ):
        fresh(
            _hippo_input(),
            physical_gpu=0,
            cpu_thread_limit=4,
            launch_ack=acknowledge,
        )
    assert subprocess_called is False
    assert acknowledged is False


def test_hippo_runner_is_env_i_offline_fresh_two_lane_and_one_per_gpu(
    tmp_path: Path,
) -> None:
    barrier = threading.Barrier(2, timeout=10)
    lock = threading.Lock()
    active = {0: 0, 1: 0}
    maximum = {0: 0, 1: 0}
    commands: list[list[str]] = []
    launch_acks: list[int] = []

    def acknowledge(gpu: int) -> None:
        with lock:
            launch_acks.append(gpu)

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        assert command[:2] == ["/usr/bin/env", "-i"]
        executable_index = command.index(str(Path(sys.executable)))
        assignments = {
            row.split("=", 1)[0]: row.split("=", 1)[1]
            for row in command[2:executable_index]
        }
        assert assignments["HF_HUB_OFFLINE"] == "1"
        assert assignments["TRANSFORMERS_OFFLINE"] == "1"
        assert assignments["OMP_NUM_THREADS"] == "4"
        assert assignments["PYTHONPYCACHEPREFIX"] == "/dev/null"
        assert len(assignments["PYTHONPATH"].split(os.pathsep)) == 2
        assert command[executable_index + 1 : executable_index + 4] == [
            "-S",
            "-B",
            "-c",
        ]
        assert runner.HIPPORAG_WORKER_MODULE in command
        assert not any(
            "API" in key or "KEY" in key or "PROXY" in key
            for key in assignments
        )
        assert kwargs["env"] == {}
        gpu = int(assignments["CUDA_VISIBLE_DEVICES"])
        with lock:
            assert launch_acks.count(gpu) == 1
            commands.append(command)
            active[gpu] += 1
            maximum[gpu] = max(maximum[gpu], active[gpu])
        try:
            barrier.wait()
            os.write(kwargs["stdout"], b"synthetic safe stdout\n")
            os.write(kwargs["stderr"], b"synthetic safe stderr\n")
            input_path = Path(command[command.index("--input") + 1])
            output_path = Path(command[command.index("--output") + 1])
            index_path = Path(command[command.index("--index-root") + 1])
            index_path.mkdir(mode=0o700)
            (index_path / "ephemeral.bin").write_bytes(b"index")
            value = json.loads(input_path.read_text(encoding="ascii"))
            output_path.write_bytes(
                hippo_contract.canonical_json_bytes(
                    hippo_contract.output_payload(
                        work_id=value["work_id"],
                        common_projection_sha256=value[
                            "common_projection_sha256"
                        ],
                        candidate_count=len(value["documents"]),
                        rank_ordinals=tuple(range(len(value["documents"]))),
                        graph_nodes=10,
                        graph_edges=9,
                    )
                )
            )
            output_path.chmod(0o600)
            return SimpleNamespace(returncode=0)
        finally:
            with lock:
                active[gpu] -= 1

    fresh = _fresh_runner(tmp_path, fake_run)
    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs = tuple(
            pool.map(
                lambda gpu: fresh(
                    _hippo_input(),
                    physical_gpu=gpu,
                    cpu_thread_limit=4,
                    launch_ack=lambda: acknowledge(gpu),
                ),
                (0, 1),
            )
        )
    assert len(outputs) == len(commands) == 2
    assert sorted(launch_acks) == [0, 1]
    assert maximum == {0: 1, 1: 1}
    attempts = sorted(fresh.runtime_root.glob("attempt-*"))
    assert len(attempts) == 2
    for attempt in attempts:
        assert not (attempt / "index.private").exists()
        terminal_path = attempt / "attempt.terminal.private.json"
        terminal = json.loads(terminal_path.read_text(encoding="ascii"))
        assert terminal["status"] == "validated_success_index_removed"
        assert terminal["subprocess_timeout_seconds"] is None
        assert terminal["stdout"]["total_size_bytes"] > 0
        assert terminal["stderr"]["total_size_bytes"] > 0
        for private in (
            "input.private.json",
            "output.private.json",
            "stdout.private.log",
            "stderr.private.log",
            "attempt.terminal.private.json",
        ):
            assert (
                stat.S_IMODE((attempt / private).stat().st_mode) == 0o600
            )

    entered = threading.Event()
    release = threading.Event()

    def blocking_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        entered.set()
        assert release.wait(10)
        input_path = Path(command[command.index("--input") + 1])
        output_path = Path(command[command.index("--output") + 1])
        index_path = Path(command[command.index("--index-root") + 1])
        index_path.mkdir(mode=0o700)
        (index_path / "ephemeral.bin").write_bytes(b"index")
        value = json.loads(input_path.read_text(encoding="ascii"))
        output_path.write_bytes(
            hippo_contract.canonical_json_bytes(
                hippo_contract.output_payload(
                    work_id=value["work_id"],
                    common_projection_sha256=value[
                        "common_projection_sha256"
                    ],
                    candidate_count=len(value["documents"]),
                    rank_ordinals=tuple(range(len(value["documents"]))),
                    graph_nodes=10,
                    graph_edges=9,
                )
            )
        )
        output_path.chmod(0o600)
        return SimpleNamespace(returncode=0)

    guarded = _fresh_runner(tmp_path / "guarded", blocking_run)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(
            guarded,
            _hippo_input(),
            physical_gpu=0,
            cpu_thread_limit=4,
            launch_ack=lambda: None,
        )
        assert entered.wait(10)
        with pytest.raises(
            runner.HitabP1ProductionRuntimeError,
            match="more than one HippoRAG process",
        ):
            guarded(
                _hippo_input(),
                physical_gpu=0,
                cpu_thread_limit=4,
                launch_ack=lambda: None,
            )
        release.set()
        first.result()

    def failing_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        index_path = Path(command[command.index("--index-root") + 1])
        index_path.mkdir(mode=0o700)
        (index_path / "failure-evidence.bin").write_bytes(b"preserve")
        os.write(kwargs["stderr"], b"private synthetic failure\n")
        return SimpleNamespace(returncode=7)

    failed = _fresh_runner(tmp_path / "failed", failing_run)
    with pytest.raises(
        runner.HitabP1ProductionRuntimeError,
        match="exited unsuccessfully",
    ):
        failed(
            _hippo_input(),
            physical_gpu=1,
            cpu_thread_limit=4,
            launch_ack=lambda: None,
        )
    failed_attempt = next(failed.runtime_root.glob("attempt-*"))
    assert (failed_attempt / "index.private").is_dir()
    failed_terminal = json.loads(
        (failed_attempt / "attempt.terminal.private.json").read_text(
            encoding="ascii"
        )
    )
    assert failed_terminal["status"] == "terminal_failure_no_retry"
    assert failed_terminal["returncode"] == 7
    assert failed_terminal["ephemeral_index_preserved"] is True
    assert failed_terminal["stderr"]["total_size_bytes"] > 0


def test_source_free_canary_full_synthetic_path_is_exclusive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    implementation_path, implementation = _implementation_freeze(tmp_path)
    output = tmp_path / "receipts" / "canary.json"
    failed_prepare_output = tmp_path / "failed_prepare" / "canary.json"
    binding_calls = 0

    def forbidden_bindings(
        _implementation: runner.FrozenImplementation, _root: Path
    ) -> runner.ProductionBindings:
        nonlocal binding_calls
        binding_calls += 1
        return _synthetic_bindings(_implementation, _root)

    def failed_prepare(
        _implementation: runner.FrozenImplementation,
        *,
        verify_hippo_child: bool,
    ) -> dict[str, object]:
        assert verify_hippo_child is True
        raise RuntimeError("synthetic dependency failure before canary")

    with pytest.raises(RuntimeError, match="before canary"):
        runner.run_source_free_canary_once(
            implementation_freeze_path=implementation_path,
            output_path=failed_prepare_output,
            binding_builder=forbidden_bindings,
            runtime_preparer=failed_prepare,
        )
    assert binding_calls == 0
    assert not failed_prepare_output.exists()
    failed_attempt_path = (
        failed_prepare_output.parent
        / "source_free_canary.attempt.private.json"
    )
    assert failed_attempt_path.is_file()
    failed_attempt = json.loads(
        failed_attempt_path.read_text(encoding="ascii")
    )
    assert failed_attempt["schema"] == runner.CANARY_ATTEMPT_SCHEMA
    assert (
        failed_attempt[
            "retry_replay_resample_provider_model_candidate_or_gate_change_count"
        ]
        == 0
    )
    assert (
        failed_attempt["implementation_freeze_file_sha256"]
        == runner.file_sha256(implementation_path)
    )
    with pytest.raises(
        runner.HitabP1ProductionRuntimeError, match="already exists"
    ):
        runner.run_source_free_canary_once(
            implementation_freeze_path=implementation_path,
            output_path=failed_prepare_output,
            binding_builder=forbidden_bindings,
            runtime_preparer=_noop_runtime_preparer,
        )
    assert binding_calls == 0

    failed_binding_output = tmp_path / "failed_binding" / "canary.json"
    failed_binding_calls = 0

    def failed_binding(
        _implementation: runner.FrozenImplementation, _root: Path
    ) -> runner.ProductionBindings:
        nonlocal failed_binding_calls
        failed_binding_calls += 1
        raise RuntimeError("synthetic canary binding failure")

    with pytest.raises(RuntimeError, match="canary binding failure"):
        runner.run_source_free_canary_once(
            implementation_freeze_path=implementation_path,
            output_path=failed_binding_output,
            binding_builder=failed_binding,
            runtime_preparer=_noop_runtime_preparer,
        )
    assert failed_binding_calls == 1
    with pytest.raises(
        runner.HitabP1ProductionRuntimeError, match="already exists"
    ):
        runner.run_source_free_canary_once(
            implementation_freeze_path=implementation_path,
            output_path=failed_binding_output,
            binding_builder=failed_binding,
            runtime_preparer=_noop_runtime_preparer,
        )
    assert failed_binding_calls == 1

    contaminated_binding_output = (
        tmp_path / "contaminated_binding" / "canary.json"
    )
    contaminated_name = "hitab_v2_tmp_binding_sentinel"

    def contaminated_binding(
        _implementation: runner.FrozenImplementation, _root: Path
    ) -> runner.ProductionBindings:
        module = ModuleType(contaminated_name)
        module.__file__ = "/tmp/hitab-v2-binding/generated.py"
        sys.modules[contaminated_name] = module
        return _synthetic_bindings(_implementation, _root)

    try:
        with pytest.raises(
            runner.HitabP1ProductionRuntimeError,
            match="shared temporary space",
        ):
            runner.run_source_free_canary_once(
                implementation_freeze_path=implementation_path,
                output_path=contaminated_binding_output,
                binding_builder=contaminated_binding,
                runtime_preparer=_noop_runtime_preparer,
            )
    finally:
        sys.modules.pop(contaminated_name, None)
    assert not contaminated_binding_output.exists()

    failed_canary_output = tmp_path / "failed_canary" / "canary.json"
    canary_calls = 0

    def failed_canary(**_kwargs: object) -> dict[str, object]:
        nonlocal canary_calls
        canary_calls += 1
        raise RuntimeError("synthetic real canary failure")

    from assumption_agent.benchmarks import (
        hitab_p1_public_canary_v1 as public_canary,
    )

    with monkeypatch.context() as scoped:
        scoped.setattr(public_canary, "run_public_canary", failed_canary)
        with pytest.raises(
            runner.HitabP1ProductionRuntimeError,
            match="production canary failed",
        ):
            runner.run_source_free_canary_once(
                implementation_freeze_path=implementation_path,
                output_path=failed_canary_output,
                binding_builder=_synthetic_bindings,
                runtime_preparer=_noop_runtime_preparer,
            )
    assert canary_calls == 1
    with pytest.raises(
        runner.HitabP1ProductionRuntimeError, match="already exists"
    ):
        runner.run_source_free_canary_once(
            implementation_freeze_path=implementation_path,
            output_path=failed_canary_output,
            binding_builder=_synthetic_bindings,
            runtime_preparer=_noop_runtime_preparer,
        )
    assert canary_calls == 1

    contaminated_canary_output = (
        tmp_path / "contaminated_canary" / "canary.json"
    )
    contaminated_canary_name = "hitab_v2_tmp_canary_sentinel"
    original_public_canary = public_canary.run_public_canary

    def contaminated_canary(**kwargs: object) -> dict[str, object]:
        value = original_public_canary(**kwargs)
        module = ModuleType(contaminated_canary_name)
        module.__file__ = "/var/tmp/hitab-v2-canary/generated.py"
        sys.modules[contaminated_canary_name] = module
        return value

    try:
        with monkeypatch.context() as scoped:
            scoped.setattr(
                public_canary,
                "run_public_canary",
                contaminated_canary,
            )
            with pytest.raises(
                runner.HitabP1ProductionRuntimeError,
                match="production canary failed",
            ):
                runner.run_source_free_canary_once(
                    implementation_freeze_path=implementation_path,
                    output_path=contaminated_canary_output,
                    binding_builder=_synthetic_bindings,
                    runtime_preparer=_noop_runtime_preparer,
                )
    finally:
        sys.modules.pop(contaminated_canary_name, None)
    assert not contaminated_canary_output.exists()

    receipt = runner.run_source_free_canary_once(
        implementation_freeze_path=implementation_path,
        output_path=output,
        binding_builder=_synthetic_bindings,
        runtime_preparer=_noop_runtime_preparer,
    )
    assert receipt["qualified"] is True
    assert receipt["source_or_HiTab_rows_accessed"] is False
    assert receipt["online_or_API_call_count"] == 0
    assert (
        receipt["implementation_freeze_self_sha256"]
        == implementation["self_sha256"]
    )
    assert receipt["canary"]["repeat_count"] == 2
    assert receipt["canary"]["repeat_exact"] is True
    assert len(receipt["canary_attempt_self_sha256"]) == 64
    with pytest.raises(
        runner.HitabP1ProductionRuntimeError, match="already exists"
    ):
        runner.run_source_free_canary_once(
            implementation_freeze_path=implementation_path,
            output_path=output,
            binding_builder=_synthetic_bindings,
            runtime_preparer=_noop_runtime_preparer,
        )


def _source_fixture(
    tmp_path: Path,
) -> tuple[dict[str, Path], dict[str, object]]:
    linux_root = Path(
        tempfile.mkdtemp(prefix=f"hitab-source-{tmp_path.name}-", dir="/tmp")
    )
    source_paths: dict[str, Path] = {}
    identities: dict[str, object] = {}
    for index, key in enumerate(("TRAIN", "DEV", "TEST", "TABLES")):
        path = linux_root / "source" / f"{key.casefold()}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = (f"frozen-{key}-{index}\n").encode("ascii")
        path.write_bytes(raw)
        path.chmod(0o600)
        blob = hashlib.sha1(
            f"blob {len(raw)}\0".encode("ascii") + raw
        ).hexdigest()
        identity = {
            "git_blob_sha1": blob,
            "raw_newline_count": (
                raw.count(b"\n") if key != "TABLES" else None
            ),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        }
        source_paths[key] = path
        identities[key] = identity
    receipt = _self_hashed(
        {
            "file_count": 4,
            "files": identities,
            "json_decode_count": 0,
            "network_attempt_count": 4,
            "parallel_transport_count": 4,
            "retry_resume_range_mirror_or_provider_switch_count": 0,
            "schema": "hitab_p1_source_download_receipt_v1",
            "source_identity_commitment": runner.stable_hash(identities),
            "status": "four_exact_sources_acquired_once",
            "study_id": runner.STUDY_ID,
            "test_json_decode_count": 0,
            "version": "hitab_p1_source_acquisition_v1",
        }
    )
    return source_paths, receipt


def test_acquire_stage_consumes_committed_authorization_once_without_decode(
    tmp_path: Path,
) -> None:
    implementation_path, implementation = _implementation_freeze(tmp_path)
    canary_path = tmp_path / "receipts" / "canary.json"
    runner.run_source_free_canary_once(
        implementation_freeze_path=implementation_path,
        output_path=canary_path,
        binding_builder=_synthetic_bindings,
        runtime_preparer=_noop_runtime_preparer,
    )
    canary_value = json.loads(canary_path.read_text(encoding="ascii"))
    source_root = tmp_path / "fresh_source"
    control_root = tmp_path / "acquisition_control"
    acquisition = _self_hashed(
        {
            "canary_receipt": {
                "file_sha256": runner.file_sha256(canary_path),
                "path": str(canary_path),
                "self_sha256": canary_value["self_sha256"],
            },
            "control_root": str(control_root),
            "implementation_freeze": {
                "file_sha256": runner.file_sha256(implementation_path),
                "path": str(implementation_path),
                "self_sha256": implementation["self_sha256"],
            },
            "json_decode_count": 0,
            "network_attempt_count": 4,
            "parallel_transport_count": 4,
            "retry_resume_range_mirror_or_provider_switch_count": 0,
            "schema": runner.ACQUISITION_FREEZE_SCHEMA,
            "source_root": str(source_root),
            "study_id": runner.STUDY_ID,
        }
    )
    acquisition_path = tmp_path / "acquisition.freeze.json"
    _write_json(acquisition_path, acquisition)
    _paths, safe_receipt = _source_fixture(tmp_path / "downloaded")

    def fake_acquire(
        frozen: runner.FrozenAcquisition,
    ) -> dict[str, object]:
        assert frozen.source_root == source_root
        frozen.control_root.mkdir(parents=True, mode=0o700)
        _write_json(
            frozen.control_root / "source_download.attempt.private.json",
            _self_hashed({"consumed": True}),
        )
        _write_json(
            frozen.control_root / "source_download.receipt.safe.json",
            safe_receipt,
        )
        return safe_receipt

    acquisition_calls = 0

    def forbidden_acquire(
        _frozen: runner.FrozenAcquisition,
    ) -> dict[str, object]:
        nonlocal acquisition_calls
        acquisition_calls += 1
        return safe_receipt

    def failed_prepare(
        _implementation: runner.FrozenImplementation,
        *,
        verify_hippo_child: bool,
    ) -> dict[str, object]:
        assert verify_hippo_child is False
        raise RuntimeError("synthetic dependency failure before network")

    with pytest.raises(RuntimeError, match="before network"):
        runner.run_source_acquisition_once(
            acquisition_freeze_path=acquisition_path,
            acquisition_runner=forbidden_acquire,
            runtime_preparer=failed_prepare,
        )
    assert acquisition_calls == 0

    result = runner.run_source_acquisition_once(
        acquisition_freeze_path=acquisition_path,
        acquisition_runner=fake_acquire,
        runtime_preparer=_noop_runtime_preparer,
    )
    assert result["json_decode_count"] == 0
    assert result["test_json_decode_count"] == 0
    assert result["network_attempt_count"] == 4
    with pytest.raises(
        runner.HitabP1ProductionRuntimeError, match="already consumed"
    ):
        runner.run_source_acquisition_once(
            acquisition_freeze_path=acquisition_path,
            acquisition_runner=fake_acquire,
            runtime_preparer=_noop_runtime_preparer,
        )


def test_formal_consumes_verified_canary_and_execution_freeze_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    implementation_path, implementation = _implementation_freeze(tmp_path)
    canary_path = tmp_path / "receipts" / "canary.json"
    runner.run_source_free_canary_once(
        implementation_freeze_path=implementation_path,
        output_path=canary_path,
        binding_builder=_synthetic_bindings,
        runtime_preparer=_noop_runtime_preparer,
    )
    canary_value = json.loads(canary_path.read_text(encoding="ascii"))
    source_paths, source_receipt = _source_fixture(tmp_path)
    source_receipt_path = tmp_path / "source_control" / "receipt.json"
    _write_json(source_receipt_path, source_receipt)
    formal_root = tmp_path / "formal_work"
    execution_body: dict[str, object] = {
        "acquisition_factory": {
            "attribute": runner.ACQUISITION_FACTORY_ATTRIBUTE,
            "file_label": runner.ACQUISITION_FACTORY_FILE_LABEL,
            "module": runner.ACQUISITION_FACTORY_MODULE,
        },
        "canary_receipt": {
            "file_sha256": runner.file_sha256(canary_path),
            "path": str(canary_path),
            "self_sha256": canary_value["self_sha256"],
        },
        "formal_work_root": str(formal_root),
        "implementation_freeze": {
            "file_sha256": runner.file_sha256(implementation_path),
            "path": str(implementation_path),
            "self_sha256": implementation["self_sha256"],
        },
        "retry_replay_resample_provider_model_candidate_or_gate_change_count": 0,
        "runtime_policy": runner.RUNTIME_POLICY,
        "schema": runner.EXECUTION_FREEZE_SCHEMA,
        "source_files": {
            key: {
                "git_blob_sha1": source_receipt["files"][key][
                    "git_blob_sha1"
                ],
                "path": str(source_paths[key]),
                "sha256": source_receipt["files"][key]["sha256"],
                "size_bytes": source_receipt["files"][key]["size_bytes"],
            }
            for key in ("TRAIN", "DEV", "TEST", "TABLES")
        },
        "source_receipt": {
            "file_sha256": runner.file_sha256(source_receipt_path),
            "path": str(source_receipt_path),
            "self_sha256": source_receipt["self_sha256"],
            "source_identity_commitment": source_receipt[
                "source_identity_commitment"
            ],
        },
        "study_id": runner.STUDY_ID,
    }
    execution = _self_hashed(execution_body)
    execution_path = tmp_path / "execution.freeze.json"
    _write_json(execution_path, execution)
    acquisition = object()
    calls = 0

    source_validation_calls = 0
    original_source_validator = runner._validate_source_download_receipt

    def recording_source_validator(*args, **kwargs):
        nonlocal source_validation_calls
        source_validation_calls += 1
        return original_source_validator(*args, **kwargs)

    monkeypatch.setattr(
        runner,
        "_validate_source_download_receipt",
        recording_source_validator,
    )

    def failed_runtime_prepare(
        _implementation: runner.FrozenImplementation,
        *,
        verify_hippo_child: bool,
    ) -> dict[str, object]:
        assert verify_hippo_child is True
        raise RuntimeError("synthetic dependency failure before source")

    with pytest.raises(RuntimeError, match="before source"):
        runner.run_formal_once(
            execution_freeze_path=execution_path,
            binding_builder=_synthetic_bindings,
            acquisition_factory_loader=lambda _execution: acquisition,
            controller_runner=lambda **_kwargs: {"forbidden": True},
            runtime_preparer=failed_runtime_prepare,
        )
    assert source_validation_calls == 0
    assert not formal_root.exists()

    def fake_controller(**kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        assert kwargs["acquisition"] is acquisition
        assert kwargs["execution_binding_sha256"] == execution["self_sha256"]
        assert kwargs["work_root"] == formal_root
        return {"safe": True}

    terminal = runner.run_formal_once(
        execution_freeze_path=execution_path,
        binding_builder=_synthetic_bindings,
        acquisition_factory_loader=lambda _execution: acquisition,
        controller_runner=fake_controller,
        runtime_preparer=_noop_runtime_preparer,
    )
    assert terminal == {"safe": True}
    assert source_validation_calls == 1
    assert calls == 1
    with pytest.raises(
        runner.HitabP1ProductionRuntimeError, match="already exists"
    ):
        runner.run_formal_once(
            execution_freeze_path=execution_path,
            binding_builder=_synthetic_bindings,
            acquisition_factory_loader=lambda _execution: acquisition,
            controller_runner=fake_controller,
            runtime_preparer=_noop_runtime_preparer,
        )
    assert calls == 1

    failure_root = tmp_path / "binding_failure_formal_work"
    failure_body = dict(execution_body)
    failure_body["formal_work_root"] = str(failure_root)
    failure_execution = _self_hashed(failure_body)
    failure_path = tmp_path / "binding_failure.execution.freeze.json"
    _write_json(failure_path, failure_execution)
    factory_calls = 0
    failed_binding_calls = 0

    def forbidden_factory(_execution: runner.FrozenExecution) -> object:
        nonlocal factory_calls
        factory_calls += 1
        return object()

    def failed_binding(
        _implementation: runner.FrozenImplementation, _root: Path
    ) -> runner.ProductionBindings:
        nonlocal failed_binding_calls
        failed_binding_calls += 1
        raise RuntimeError("synthetic source-free binding failure")

    with pytest.raises(RuntimeError, match="source-free binding failure"):
        runner.run_formal_once(
            execution_freeze_path=failure_path,
            binding_builder=failed_binding,
            acquisition_factory_loader=forbidden_factory,
            controller_runner=fake_controller,
            runtime_preparer=_noop_runtime_preparer,
        )
    assert factory_calls == 0
    assert failed_binding_calls == 1
    assert (
        failure_root / "production_execution.claim.json"
    ).is_file()
    assert {
        path.name for path in failure_root.iterdir()
    } == {"production_execution.claim.json"}
    assert not (
        source_receipt_path.parent / "initial_selection.attempt.private.json"
    ).exists()
    with pytest.raises(
        runner.HitabP1ProductionRuntimeError, match="already exists"
    ):
        runner.run_formal_once(
            execution_freeze_path=failure_path,
            binding_builder=failed_binding,
            acquisition_factory_loader=forbidden_factory,
            controller_runner=fake_controller,
            runtime_preparer=_noop_runtime_preparer,
        )
    assert failed_binding_calls == 1
    assert factory_calls == 0


@pytest.mark.parametrize(
    "unit_name",
    (
        "hitab_p1_source_free_canary_unit_v1.service",
        "hitab_p1_source_acquisition_unit_v1.service",
        "hitab_p1_formal_unit_v1.service",
    ),
)
def test_user_units_freeze_resource_and_network_policy(unit_name: str) -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "manifests" / unit_name).read_text(encoding="utf-8")
    assert "Restart=no" in text
    assert "CPUQuota=800%" in text
    assert "MemoryMax=40G" in text
    assert "TasksMax=64" in text
    assert "KillMode=control-group" in text
    assert "TimeoutStartSec=infinity" in text
    assert "/usr/bin/env -i " in text
    assert " -S -B -c " in text
    assert runner.OUTER_ENTRYPOINT_SCRIPT in text
    assert " -m replication_runtime.hitab_p1_formal_v1.runner " not in text
    if "source_acquisition" in unit_name:
        assert "IPAddressDeny=any" not in text
        assert "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6" in text
        assert " acquire --acquisition-freeze " in text
        assert "CUDA_VISIBLE_DEVICES" not in text
    else:
        assert "IPAddressDeny=any" in text
        assert "RestrictAddressFamilies=AF_UNIX" in text
        assert "HF_HUB_OFFLINE=1" in text
        assert "TRANSFORMERS_OFFLINE=1" in text
        assert "CUDA_VISIBLE_DEVICES=0 " in text
        assert "CUDA_VISIBLE_DEVICES=0,1" not in text
    assert "ASSUMPTION_V2_API" not in text
    assert "ruoli" not in text.casefold()
