from __future__ import annotations

import json
from pathlib import Path

import pytest

from hegel_machine.phase3_m25_wire_v1 import decode_formal_object
from hegel_machine.phase3_m3_bounded_enumerator_cli_v1 import main


ROOTS = (
    "11" * 32,
    "22" * 32,
    "33" * 32,
)


def _args(output: Path | None = None) -> list[str]:
    result = [
        "--enumerate-prefix",
        "--child-dsl-spec-root",
        ROOTS[0],
        "--operator-semantics-root",
        ROOTS[1],
        "--identifier-registry-root",
        ROOTS[2],
        "--diagnostic-canonical-budget",
        "10",
        "--diagnostic-raw-application-cap",
        "20000",
    ]
    if output is not None:
        result.extend(("--output-directory", str(output)))
    return result


def _read_framed(path: Path) -> list[bytes]:
    payload = path.read_bytes()
    offset = 0
    rows = []
    while offset < len(payload):
        length = int.from_bytes(payload[offset : offset + 4], "big")
        offset += 4
        rows.append(payload[offset : offset + length])
        offset += length
    assert offset == len(payload)
    return rows


def test_binding_material_is_public_and_role_free(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--binding-material"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["implementation_id"] == 1
    assert not report["target_roles_evaluated"]
    assert not report["split_material_accessed"]
    assert not report["secrets_accessed"]


def test_small_diagnostic_cli_writes_replayable_exclusive_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "python-enumerator"
    assert main(_args(output)) == 0
    stdout_report = json.loads(capsys.readouterr().out)
    disk_report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert stdout_report == disk_report
    assert disk_report["claim_level"] == "NON_FORMAL_DIAGNOSTIC_TEST_PROFILE"
    assert disk_report["closure_status"] == "DSL_TOO_LARGE"
    assert disk_report["raw_operator_application_count"] == 14
    assert disk_report["program_record_count"] == 10
    schemas = (
        ("canonical_program_records.cborframed", "CanonicalProgramRecordV2", 10),
        ("program_chunk_manifests.cborframed", "ProgramChunkManifestV2", 1),
        ("bucket_accounting_records.cborframed", "BucketAccountingRecordV1", 175),
    )
    for name, schema, expected_count in schemas:
        rows = _read_framed(output / name)
        assert len(rows) == expected_count
        for row in rows:
            decode_formal_object(row, expected_name=schema)
    with pytest.raises(FileExistsError):
        main(_args(output))


def test_cli_requires_both_diagnostic_limits() -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--enumerate-prefix",
                "--child-dsl-spec-root",
                ROOTS[0],
                "--operator-semantics-root",
                ROOTS[1],
                "--identifier-registry-root",
                ROOTS[2],
                "--diagnostic-canonical-budget",
                "10",
            ]
        )
