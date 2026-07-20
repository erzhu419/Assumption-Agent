from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from reconstruction_v2.assumption_agent.benchmarks import (
    hipporag_upstream_hardening_qualification_v1 as qualification,
)
from reconstruction_v2.replication_runtime.hipporag_upstream_hardening_v1 import (
    backport,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE = PROJECT_ROOT / "reconstruction_v2"
REPO = BASE / qualification.BASELINE_REPO_RELATIVE
SOURCE = REPO / qualification.BASELINE_SOURCE_WITHIN_REPO


def test_design_self_hash_and_no_dev_replay() -> None:
    design = json.loads((BASE / qualification.DESIGN_RELATIVE).read_text())
    assert design["schema"] == qualification.DESIGN_SCHEMA
    assert qualification.verify_self_hash(design) == qualification.DESIGN_SELF_SHA256
    boundary = design["authorization_boundary"]
    assert boundary["current_FiQA_DEV_cohort_input_index_or_query_access"] is False
    assert boundary["current_FiQA_DEV_cohort_replay_or_recovery"] is False
    assert boundary["FiQA_DEV_or_TEST_label_access"] is False
    assert design["result_policy"]["labels_scores_or_performance_claims"] == 0


def test_exact_backport_hash_and_ast_shape() -> None:
    baseline = SOURCE.read_bytes()
    patched = backport.apply_fixed_backport(baseline)
    patch = backport.unified_patch_bytes(baseline, patched)
    assert hashlib.sha256(patched).hexdigest() == backport.PATCHED_SOURCE_SHA256
    assert hashlib.sha256(patch).hexdigest() == backport.UNIFIED_PATCH_SHA256
    tree = ast.parse(patched.decode("utf-8"))
    graph_search = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "graph_search_with_fact_entities"
    )
    source = ast.get_source_segment(patched.decode("utf-8"), graph_search)
    assert source is not None
    assert source.count("phrases_and_ids.add((phrase, phrase_id))") == 1
    assert source.count("where=number_of_occurs != 0") == 1
    compile(patched, "HippoRAG.py", "exec")


def test_pinned_upstream_contains_exact_fix() -> None:
    completed = subprocess.run(
        [
            "git",
            "show",
            f"{backport.UPSTREAM_COMMIT}:src/hipporag/HippoRAG.py",
        ],
        cwd=REPO,
        check=True,
        capture_output=True,
    )
    backport.verify_upstream_contains_backport(completed.stdout)


def test_frozen_train_artifact_sets() -> None:
    observed = qualification.verify_frozen_artifact_sets(BASE)
    assert observed["input_count"] == 12
    assert observed["output_count"] == 12
    assert observed["cached_index_file_count"] == 72
    assert observed["input_set_sha256"] == qualification.FROZEN_INPUT_SET_SHA256
    assert observed["output_set_sha256"] == qualification.FROZEN_OUTPUT_SET_SHA256
    assert observed["cached_index_set_sha256"] == qualification.FROZEN_INDEX_SET_SHA256


def test_canonical_json_rejects_nonfinite() -> None:
    with pytest.raises(qualification.HippoRAGQualificationError):
        qualification.canonical_json_bytes({"value": float("nan")})


def test_self_hash_round_trip_and_tamper_detection() -> None:
    value = qualification.self_hashed({"schema": "fixture", "value": 1})
    assert qualification.verify_self_hash(value) == value["self_sha256"]
    tampered = dict(value)
    tampered["value"] = 2
    with pytest.raises(qualification.HippoRAGQualificationError):
        qualification.verify_self_hash(tampered)


def test_one_shot_refusal_precedes_freeze_read(tmp_path: Path) -> None:
    project = tmp_path / "project"
    base = project / "reconstruction_v2"
    (base / qualification.RUN_ROOT_RELATIVE).mkdir(parents=True)
    with pytest.raises(qualification.OneShotRefusal):
        qualification.run_formal(project)
