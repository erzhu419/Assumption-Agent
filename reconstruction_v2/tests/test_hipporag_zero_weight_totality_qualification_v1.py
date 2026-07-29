from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytest

from reconstruction_v2.assumption_agent.benchmarks import (
    hipporag_zero_weight_totality_qualification_v1 as qualification,
)
from reconstruction_v2.replication_runtime.hipporag_upstream_hardening_v1 import (
    backport as upstream_hardening,
)
from reconstruction_v2.replication_runtime.hipporag_zero_weight_totality_v1 import (
    backport,
    landlock_exec,
    repair_qualification,
    synthetic_worker,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE = PROJECT_ROOT / "reconstruction_v2"
REPO = BASE / qualification.BASELINE_REPO_RELATIVE
SOURCE = REPO / qualification.BASELINE_SOURCE_WITHIN_REPO


def _method_from_source(raw: bytes) -> object:
    tree = ast.parse(raw.decode("utf-8"))
    node = next(
        item
        for item in ast.walk(tree)
        if isinstance(item, ast.FunctionDef) and item.name == "get_top_k_weights"
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "Dict": Dict,
        "Tuple": Tuple,
        "compute_mdhash_id": lambda content, prefix: prefix + content,
        "np": np,
    }
    exec(compile(module, "HippoRAG.get_top_k_weights", "exec"), namespace)
    return namespace["get_top_k_weights"]


def test_design_self_hash_and_future_only_boundary() -> None:
    design = json.loads((BASE / qualification.DESIGN_RELATIVE).read_text())
    assert design["schema"] == qualification.DESIGN_SCHEMA
    assert qualification.verify_self_hash(design) == qualification.DESIGN_SELF_SHA256
    assert design["execution_contract"]["process_concurrency"] == 12
    assert design["execution_contract"]["label_or_qrel_open_count"] == 0
    assert design["prospective_boundary"][
        "current_FiQA_DEV_or_NanoBEIR_P12_cohort_resume_or_score"
    ] is False


def test_exact_one_edit_patch_hash_and_ast_shape() -> None:
    qualified = upstream_hardening.apply_fixed_backport(SOURCE.read_bytes())
    patched = backport.apply_totality_hardening(qualified)
    patch = backport.unified_patch_bytes(qualified, patched)
    assert hashlib.sha256(qualified).hexdigest() == backport.INPUT_SOURCE_SHA256
    assert hashlib.sha256(patched).hexdigest() == backport.PATCHED_SOURCE_SHA256
    assert hashlib.sha256(patch).hexdigest() == backport.UNIFIED_PATCH_SHA256
    before = qualified.decode("utf-8")
    after = patched.decode("utf-8")
    assert before.count("assert np.count_nonzero(all_phrase_weights)") == 1
    assert "assert np.count_nonzero(all_phrase_weights)" not in after
    assert after.count("np.all(np.isfinite(all_phrase_weights))") == 1
    assert after.count("selected_nonzero_phrase_ids") == 2
    compile(patched, "HippoRAG.py", "exec")


def test_old_assertion_fails_but_frozen_totality_fixture_passes() -> None:
    qualified = upstream_hardening.apply_fixed_backport(SOURCE.read_bytes())
    patched = backport.apply_totality_hardening(qualified)
    old_method = _method_from_source(qualified)
    new_method = _method_from_source(patched)

    class OldHippoRAG:
        get_top_k_weights = old_method

    class NewHippoRAG:
        get_top_k_weights = new_method

    with pytest.raises(AssertionError):
        synthetic_worker.exercise_fixture(
            OldHippoRAG, lambda content, prefix: prefix + content
        )
    value = synthetic_worker.exercise_fixture(
        NewHippoRAG, lambda content, prefix: prefix + content
    )
    assert value["allowed_values_unchanged"] is True
    assert value["rejected_cases"] == ["nonfinite", "unselected_nonzero"]


def test_totality_patch_rejects_source_drift() -> None:
    qualified = bytearray(
        upstream_hardening.apply_fixed_backport(SOURCE.read_bytes())
    )
    qualified[0] ^= 1
    with pytest.raises(backport.HippoRAGZeroWeightTotalityError):
        backport.apply_totality_hardening(bytes(qualified))


def test_frozen_cached_fixture_sets() -> None:
    observed = qualification.verify_frozen_artifact_sets(BASE)
    assert observed["input_count"] == 60
    assert observed["index_file_count"] == 360
    assert observed["success_output_count"] == 58
    assert observed["failure_fixture_count"] == 2
    assert (
        observed["input_set_sha256"]
        == "d7fbed92d0a9101e69925868bf3ca8bcc803628d3b23c79341a8f0d99382f310"
    )
    assert (
        observed["index_set_sha256"]
        == "c02f726b0a2e1a36d50da21a3c983c32acbb05e49325904f6911934cc8e3599e"
    )


def test_offline_runtime_accepts_resolved_venv_python_symlink() -> None:
    runtime = BASE / qualification.RUNTIME_PYTHON_RELATIVE
    assert runtime.is_symlink()
    assert runtime.resolve(strict=True).is_file()
    qualification.verify_offline_runtime_assets(BASE)


def test_nested_cached_worker_root_materializes_source_parent(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cached" / "source" / "item_000"
    qualification._prepare_writable_root(root)
    assert root.is_dir()
    assert (root / "home").is_dir()
    assert (root / "hf").is_dir()
    assert (root / "tmp").is_dir()


def test_repair_materializes_patched_shadow_import_tree(tmp_path: Path) -> None:
    package = tmp_path / "source" / "hipporag"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "from .HippoRAG import HippoRAG\n", encoding="utf-8"
    )
    qualified = upstream_hardening.apply_fixed_backport(SOURCE.read_bytes())
    input_source = package / "HippoRAG.py"
    input_source.write_bytes(qualified)
    result = repair_qualification._materialize_source(
        input_source, tmp_path / "work"
    )
    import_source = result["patched_import_root"] / "hipporag/HippoRAG.py"
    assert import_source.read_bytes() == backport.apply_totality_hardening(
        qualified
    )
    assert result["patched_source_sha256"] == backport.PATCHED_SOURCE_SHA256
    assert result["patched_import_file_count"] == 2
    public = repair_qualification._public_source_binding(result)
    assert "patched_path" not in public
    assert "patched_import_root" not in public


def test_landlock_worker_environment_is_exact_and_secret_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUOLI_API_KEY", "must-not-propagate")
    patched = tmp_path / "patched"
    runtime = tmp_path / "runtime"
    writable = tmp_path / "worker"
    environment = repair_qualification._worker_environment(
        writable_root=writable,
        patched_import_root=patched,
        runtime_root=runtime,
    )
    assert set(environment) == landlock_exec.ENVIRONMENT_KEYS
    assert "RUOLI_API_KEY" not in environment
    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    assert environment["PYTHONPATH"].split(":")[0] == str(patched)
    assert environment["TRANSFORMERS_OFFLINE"] == "1"


def test_canonical_json_rejects_nonfinite() -> None:
    with pytest.raises(qualification.HippoRAGTotalityQualificationError):
        qualification.canonical_json_bytes({"value": float("nan")})


def test_self_hash_round_trip_and_tamper_detection() -> None:
    value = qualification.self_hashed({"schema": "fixture", "value": 1})
    assert qualification.verify_self_hash(value) == value["self_sha256"]
    tampered = dict(value)
    tampered["value"] = 2
    with pytest.raises(qualification.HippoRAGTotalityQualificationError):
        qualification.verify_self_hash(tampered)


def test_one_shot_refusal_precedes_freeze_read(tmp_path: Path) -> None:
    project = tmp_path / "project"
    base = project / "reconstruction_v2"
    (base / qualification.RUN_ROOT_RELATIVE).mkdir(parents=True)
    with pytest.raises(qualification.OneShotRefusal):
        qualification.run_formal(project)
