from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
TOOL_SPEC = importlib.util.spec_from_file_location(
    "phase3_q3_bounded_quotient_certificate_tool_test_core",
    ROOT / "tools/phase3_q3_bounded_quotient_certificate_v1.py",
)
assert TOOL_SPEC is not None and TOOL_SPEC.loader is not None
tool = importlib.util.module_from_spec(TOOL_SPEC)
sys.modules[TOOL_SPEC.name] = tool
TOOL_SPEC.loader.exec_module(tool)
wire = tool.wire


def test_core_project_import_surface_is_strict_cbor_only() -> None:
    path = ROOT / "src/hegel_machine/phase3_q3_bounded_quotient_certificate_v1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    project_imports: set[str] = set()
    forbidden_calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level:
            project_imports.add(node.module or "")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                forbidden_calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                forbidden_calls.add(node.func.attr)
    assert project_imports == {"strict_cbor_v1"}
    assert not {
        "__import__",
        "eval",
        "exec",
        "import_module",
        "Popen",
        "run",
        "system",
    }.intersection(forbidden_calls)
    source = path.read_text(encoding="utf-8")
    for forbidden in (
        "phase3_q2_role_membership_v1",
        "phase3_q2_campaign_v1",
        "phase3_q2_archive_reader_v1",
        "phase3_m25_rows_v1",
        "phase3_m25_split_v1",
        "subprocess",
        "docker",
        "socket",
        "urllib",
        "requests",
        "httpx",
        "aiohttp",
    ):
        assert forbidden not in source


def test_actual_q2_artifact_replays_with_producer_free_decoder() -> None:
    payload = (ROOT / "artifacts/phase3_q2_role_membership_v1.cbor").read_bytes()
    artifact = wire.decode_sealed_q2_artifact_v1(payload)

    assert artifact.status == "Q2_ROLE_MEMBERSHIP_PASS"
    assert "sha256:" + artifact.artifact_root.hex() == (
        "sha256:10d2c1407253c85b41ab113c61bd77012ef84bc45f1b62bdd254fabb816ae790"
    )
    outside, null = artifact.role_results
    assert (outside.evaluated_class_count, outside.match_count) == (418, 0)
    assert outside.result_root_id == (
        "sha256:df7db06682bbe7d92ca7da69b25e70be7f9967dd24703950347e1fdf505095e6"
    )
    assert (null.evaluated_class_count, null.match_count) == (89, 1)
    assert null.result_root_id == (
        "sha256:e081d8a871e2b727538bdf9996c477634e81d45957d8162bfe35bede0ca7bdfd"
    )
def test_q3_candidate_is_unsigned_internal_admission_only() -> None:
    contract = tool.state.load_contract(tool.DEFAULT_CONTRACT)
    prereg = tool.state.load_preregistration(
        Path(contract.value["preregistration_path"])
    )
    replay = tool._replay_q2(contract, prereg)
    candidate = replay.candidate
    decoded = wire.decode_q3_certificate_candidate_v1(candidate.canonical_bytes)

    assert decoded == candidate
    assert replay.preflight_receipt["q3_action_audit"] == tool.state.ZERO_ACTION_AUDIT
    value = tool.state.raw_sha256(candidate.canonical_bytes)
    assert value == replay.preflight_receipt[
        "expected_candidate_artifact_raw_sha256"
    ]
    assert candidate.canonical_object()[15] == wire.Q3_SUCCESS_STATUS.encode("ascii")
    assert candidate.canonical_object()[16] == wire.Q3_CLAIM_LEVEL.encode("ascii")
    assert candidate.canonical_object()[-10:] == (False,) * 10
