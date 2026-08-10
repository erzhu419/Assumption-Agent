from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import ast as python_ast

from hegel_machine import phase3_q0_quotient_contract_v1 as contract
from hegel_machine.strict_cbor_v1 import canonical_cbor_decode, content_hash


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = PROJECT_ROOT / "tools/phase3_q0_python_oracle_entrypoint_v1.py"


def test_isolated_python_endpoint_is_exact_and_non_authoritative() -> None:
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-B", str(ENTRYPOINT)],
        cwd=PROJECT_ROOT,
        env={
            "PATH": os.environ.get("PATH", ""),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        },
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert completed.stdout.count("\n") == 1
    payload = json.loads(completed.stdout)

    assert payload["schema_version"] == "hegel-q0-python-micro-oracle/1"
    assert payload["terminal_status"] == contract.Q0_ENDPOINT_PASS_STATUS
    assert payload["canonical_syntax_count"] == 537
    assert payload["behavior_class_count"] == 69
    assert payload["frontier_point_count"] == 122
    assert payload["syntax_continuation_bank_point_count"] == 251
    assert payload["quotient_continuation_bank_point_count"] == 251
    assert payload["projection_manifest_root"] == (
        "sha256:2f39aa248f1305eeaf20a724f6d690cf2b13003f86620d09d2753815831f7ad1"
    )
    assert payload["semantic_binding_root"] == (
        "sha256:b7ec5e860a007469b8a1b3930f17c130f59a800d2a832dfd438d18a75538ff99"
    )
    assert payload["endpoint_state_root"] == (
        "sha256:d33e54dd99e6cbe8aacc541fc0877af9657a553be58523670cce5c474006d4d2"
    )

    endpoint_object = canonical_cbor_decode(
        bytes.fromhex(payload["endpoint_state_cbor_hex"])
    )
    assert len(endpoint_object) == 43
    assert content_hash(
        contract.ENDPOINT_STATE_ROOT_DOMAIN,
        endpoint_object,
    ).hex() == payload["endpoint_state_root"].removeprefix("sha256:")
    assert len(payload["syntax_coverage"]) == 27
    assert len(payload["direct_coverage"]) == 27
    assert payload["target_truth_accessed"] is False
    assert payload["split_accessed"] is False
    assert payload["role_evaluation_performed"] is False
    assert payload["formal_roots_generated"] is False
    assert payload["authority_claimed"] is False
    assert payload["python_source_root"].startswith("sha256:")


def test_entrypoint_source_has_no_target_truth_or_split_import() -> None:
    tree = python_ast.parse(ENTRYPOINT.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in python_ast.walk(tree):
        if isinstance(node, python_ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, python_ast.ImportFrom) and node.module:
            imports.add(node.module)
    assert not any(
        token in module
        for module in imports
        for token in ("phase3_dsl_v1", "target", "truth", "split")
    )
