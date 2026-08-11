from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/hegel_machine/phase3_q05b_negative_vectors_v1.py"


def test_negative_vector_import_surface_is_target_blind_and_exact() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    relative_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 1
    }
    assert relative_modules == {
        None,
        "strict_cbor_v1",
    }
    imported_aliases = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module is None
        for alias in node.names
    }
    assert imported_aliases == {
        "phase3_q05b_host_replay_v1",
        "phase3_q1_archive_projection_v1",
        "phase3_q1_capacity_preflight_v1",
        "phase3_q1_external_sort_profile_v1",
        "phase3_q1_formal_archive_contract_v1",
        "phase3_q1_partition_snapshot_v1",
        "phase3_q1_qualification_wire_v1",
        "phase3_q1_semantic_coverage_v1",
    }


def test_negative_vector_corpus_executes_in_empty_package() -> None:
    script = r'''
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType

root = Path(sys.argv[1])
package_root = root / "src/hegel_machine"
package = ModuleType("hegel_machine")
package.__path__ = [str(package_root)]
package.__package__ = "hegel_machine"
sys.modules["hegel_machine"] = package
vectors = importlib.import_module("hegel_machine.phase3_q05b_negative_vectors_v1")
corpus = vectors.run_q05b_negative_vector_corpus_v1()

try:
    vectors.Q05BNegativeVectorRowV1(b"bad", True, b"FAIL", b"FAIL", b"r" * 32)
except vectors.Q05BNegativeVectorError:
    bool_alias_rejected = True
else:
    bool_alias_rejected = False

forbidden = {
    "hegel_machine.__init__",
    "hegel_machine.phase3_dsl_v1",
    "hegel_machine.phase3_m25_rows_v1",
    "hegel_machine.phase3_m25_split_v1",
    "hegel_machine.phase3_m25_formal_static_basis_v1",
}
loaded = sorted(name for name in sys.modules if name.startswith("hegel_machine"))
value = {
    "bool_alias_rejected": bool_alias_rejected,
    "category_roots": [[category, root.hex()] for category, root in corpus.category_roots],
    "closed_authority": [
        item.hex() if type(item) is bytes else list(item) if type(item) is tuple else item
        for item in corpus.canonical_object()[-1]
    ],
    "corpus_root": corpus.corpus_root.hex(),
    "forbidden_loaded": sorted(forbidden.intersection(loaded)),
    "rows": [
        [
            row.vector_id.decode("ascii"),
            row.category,
            row.expected_failure.decode("ascii"),
            row.observed_failure.decode("ascii"),
            row.evidence_root.hex(),
        ]
        for row in corpus.rows
    ],
}
print(json.dumps(value, sort_keys=True, separators=(",", ":")))
'''
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", script, str(ROOT)],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=900,
    )
    assert completed.stderr == ""
    value = json.loads(completed.stdout)
    rows = value["rows"]
    assert len(rows) == 11
    assert [row[0] for row in rows] == sorted(row[0] for row in rows)
    assert {row[1] for row in rows} == {13, 18}
    assert all(type(row[1]) is int for row in rows)
    assert all(row[2] == row[3] for row in rows)
    assert sum(row[2] == "NO_FAILURE" for row in rows) == 1
    assert all(len(row[4]) == 64 for row in rows)

    by_id = {row[0]: row for row in rows}
    assert by_id["p13-boundary-accepted-exact-16mib"][3] == "NO_FAILURE"
    assert by_id["p13-boundary-reject-plus-one"][3] == "INCONCLUSIVE_RESOURCE_LIMIT"
    assert by_id["p13-cbor-bool-is-not-uint-alias"][3] == "REJECT_Q05B_UINT"
    assert by_id["p13-cbor-noncanonical-uint"][3] == "REJECT_NONCANONICAL_CBOR"
    for vector_id in (
        "p18-candidate-gap-external-manifest-hash",
        "p18-candidate-gap-framed-reorder",
        "p18-candidate-gap-record-set-duplicate",
    ):
        assert by_id[vector_id][3] == "FAIL_Q05B_HOST_STREAM_REPLAY"

    assert value["category_roots"][0][0] == 13
    assert value["category_roots"][1][0] == 18
    assert all(len(row[1]) == 64 for row in value["category_roots"])
    assert len(value["corpus_root"]) == 64
    assert value["bool_alias_rejected"] is True
    assert value["forbidden_loaded"] == []
    assert value["closed_authority"] == [
        "71315f7374617465",
        "4e4f545f52554e",
        "71315f676174655f636f756e74",
        0,
        "71315f676174655f6d61736b",
        0,
        "71315f6f75747075745f736c6f7473",
        [None] * 8,
        "63657274696669636174655f616374697665",
        False,
    ]
