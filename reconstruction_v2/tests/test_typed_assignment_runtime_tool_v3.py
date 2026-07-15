from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pytest

from assumption_agent.benchmarks import typed_assignment_runtime_tool_v3 as runtime


PUBLIC_INSTRUCTION = """
Organize every paper in /root/papers/all into these five sibling folders:
LLM, trapped_ion_and_qc, black_hole, DNA, and music_history.
Use the paper's contents to decide. If a paper does not fit into any other 4
folders, put it into the last one.
"""


def _write_docx(path: Path, text: str) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "word/document.xml",
            (
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<w:document xmlns:w="urn:test"><w:body>'
                f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>"
                "</w:body></w:document>"
            ),
        )


def _prepared_runtime(tmp_path: Path, texts: tuple[str, ...] = ("DNA binding",)):
    task_root = tmp_path / "papers"
    source = task_root / "all"
    source.mkdir(parents=True)
    for index, text in enumerate(texts):
        _write_docx(source / f"paper-{index}.docx", text)
    instruction = tmp_path / "instruction.txt"
    instruction.write_text(PUBLIC_INSTRUCTION, encoding="utf-8")
    sidecar = tmp_path / "sidecar"
    receipt = runtime.prepare_assignment_runtime(
        task_root=task_root,
        source_dir=source,
        public_instruction_file=instruction,
        sidecar_dir=sidecar,
    )
    receipt_file_sha256 = runtime.sha256_file(
        sidecar / runtime.DEFAULT_PREPARE_RECEIPT_FILENAME
    )
    evidence = json.loads(
        (sidecar / runtime.DEFAULT_EVIDENCE_FILENAME).read_text(encoding="utf-8")
    )
    return task_root, source, sidecar, receipt, receipt_file_sha256, evidence


def _write_plan(
    sidecar: Path,
    evidence: dict[str, object],
    *,
    assignments: list[dict[str, object]],
) -> Path:
    path = sidecar / runtime.DEFAULT_PLAN_FILENAME
    path.write_text(
        json.dumps(
            {
                "contract_hash": evidence["contract_hash"],
                "evidence_set_hash": evidence["evidence_set_hash"],
                "assignments": assignments,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _positive_assignment(
    evidence_row: dict[str, object], destination: str
) -> dict[str, object]:
    evidence_items = evidence_row["evidence"]
    assert isinstance(evidence_items, list) and evidence_items
    evidence_item = evidence_items[0]
    assert isinstance(evidence_item, dict)
    return {
        "file_id": evidence_row["file_id"],
        "destination": destination,
        "basis": "positive_content_evidence",
        "evidence_ids": [evidence_item["evidence_id"]],
    }


def test_public_default_parses_exact_other_four_last_one_wording() -> None:
    destinations, public_default = runtime.parse_public_destination_spec(
        PUBLIC_INSTRUCTION
    )

    assert destinations == runtime.PUBLIC_ORGANIZE_DESTINATIONS
    assert public_default == "music_history"


def test_prepare_extracts_bounded_openxml_and_emits_host_safe_receipt(
    tmp_path: Path,
) -> None:
    _, _, sidecar, receipt, _, evidence = _prepared_runtime(
        tmp_path, ("DNA " * 5000, "large language model reasoning")
    )

    assert receipt["file_count"] == 2
    assert receipt["evidence_count"] == 2
    assert receipt["extraction_unavailable_count"] == 0
    assert receipt["host_safe_receipt"] is True
    serialized_receipt = json.dumps(receipt, sort_keys=True)
    assert "paper-0.docx" not in serialized_receipt
    assert "large language model" not in serialized_receipt
    assert receipt["receipt_hash"] == runtime._payload_hash(
        {key: value for key, value in receipt.items() if key != "receipt_hash"}
    )
    files = evidence["files"]
    assert len(files) == 2
    assert len(files[0]["evidence"][0]["text"]) <= (
        runtime.DEFAULT_MAX_EXTRACTED_CHARACTERS
    )
    assert files[0]["evidence"][0]["truncated"] is True
    for filename in (
        runtime.DEFAULT_EVIDENCE_FILENAME,
        runtime.DEFAULT_PRE_MANIFEST_FILENAME,
        runtime.DEFAULT_PLAN_SCHEMA_FILENAME,
        runtime.DEFAULT_PREPARE_STATE_FILENAME,
        runtime.DEFAULT_PREPARE_RECEIPT_FILENAME,
    ):
        assert (sidecar / filename).is_file()


def test_pdf_first_pages_extraction_is_bounded_and_uses_offline_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"not-needed-by-mocked-pdftotext")
    captured: list[list[str]] = []

    class Completed:
        returncode = 0

    def fake_run(command, **kwargs):
        captured.append(list(command))
        Path(command[-1]).write_text("title " + "abstract " * 5000, encoding="utf-8")
        return Completed()

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)
    text, truncated, status = runtime._extract_pdf_text(
        pdf,
        pdftotext_binary="/usr/bin/pdftotext",
        pdf_pages=2,
        timeout_seconds=3.0,
        maximum_characters=300,
    )

    assert status == "ok"
    assert truncated is True
    assert len(text) == 300
    assert captured[0][1:5] == ["-f", "1", "-l", "2"]
    assert "http" not in " ".join(captured[0])


def test_valid_total_plan_is_applied_transactionally_and_reconciled(
    tmp_path: Path,
) -> None:
    task_root, source, sidecar, _, receipt_sha, evidence = _prepared_runtime(
        tmp_path, ("DNA binding protein", "history of baroque music")
    )
    rows = evidence["files"]
    assignments = [
        _positive_assignment(rows[1], "music_history"),
        _positive_assignment(rows[0], "DNA"),
    ]
    _write_plan(sidecar, evidence, assignments=assignments)

    receipt = runtime.apply_assignment_plan(
        sidecar_dir=sidecar,
        expected_prepare_receipt_sha256=receipt_sha,
    )

    assert list(source.iterdir()) == []
    assert (task_root / "DNA" / "paper-0.docx").is_file()
    assert (task_root / "music_history" / "paper-1.docx").is_file()
    assert all((task_root / name).is_dir() for name in runtime.PUBLIC_ORGANIZE_DESTINATIONS)
    assert receipt["mode"] == "apply_and_reconcile"
    assert receipt["assignment_count"] == 2
    assert receipt["reopened_file_count"] == 2
    assert receipt["all_destination_content_hashes_match"] is True
    assert receipt["source_filenames_in_receipt"] is False

    second = runtime.reconcile_assignment_runtime(
        sidecar_dir=sidecar,
        expected_prepare_receipt_sha256=receipt_sha,
    )
    assert second["mode"] == "reconcile_existing"
    assert second["final_task_manifest_hash"] == receipt["final_task_manifest_hash"]


@pytest.mark.parametrize(
    "mutation",
    ("missing_file", "duplicate_file", "foreign_evidence", "bad_default"),
)
def test_plan_rejects_non_bijection_or_unlicensed_evidence(
    tmp_path: Path, mutation: str
) -> None:
    _, source, sidecar, _, receipt_sha, evidence = _prepared_runtime(
        tmp_path, ("DNA binding", "music history")
    )
    rows = evidence["files"]
    assignments = [
        _positive_assignment(rows[0], "DNA"),
        _positive_assignment(rows[1], "music_history"),
    ]
    if mutation == "missing_file":
        assignments.pop()
    elif mutation == "duplicate_file":
        assignments[1]["file_id"] = assignments[0]["file_id"]
    elif mutation == "foreign_evidence":
        assignments[0]["evidence_ids"] = assignments[1]["evidence_ids"]
    else:
        assignments[0] = {
            "file_id": rows[0]["file_id"],
            "destination": "DNA",
            "basis": "public_default",
            "evidence_ids": [],
        }
    _write_plan(sidecar, evidence, assignments=assignments)

    with pytest.raises(runtime.TypedAssignmentRuntimeError):
        runtime.apply_assignment_plan(
            sidecar_dir=sidecar,
            expected_prepare_receipt_sha256=receipt_sha,
        )

    assert len(list(source.iterdir())) == 2


def test_agent_task_tree_or_sidecar_mutation_is_rejected_before_moves(
    tmp_path: Path,
) -> None:
    task_root, source, sidecar, _, receipt_sha, evidence = _prepared_runtime(
        tmp_path
    )
    row = evidence["files"][0]
    _write_plan(
        sidecar,
        evidence,
        assignments=[_positive_assignment(row, "DNA")],
    )
    (task_root / "agent-created.txt").write_text("mutation", encoding="utf-8")

    with pytest.raises(
        runtime.TypedAssignmentRuntimeError,
        match="task tree changed before plan application",
    ):
        runtime.apply_assignment_plan(
            sidecar_dir=sidecar,
            expected_prepare_receipt_sha256=receipt_sha,
        )
    assert (source / "paper-0.docx").is_file()

    (task_root / "agent-created.txt").unlink()
    (sidecar / "unregistered.json").write_text("{}", encoding="utf-8")
    with pytest.raises(
        runtime.TypedAssignmentRuntimeError,
        match="unregistered file",
    ):
        runtime.apply_assignment_plan(
            sidecar_dir=sidecar,
            expected_prepare_receipt_sha256=receipt_sha,
        )
    assert (source / "paper-0.docx").is_file()


def test_failed_post_move_reconciliation_rolls_back_exact_pre_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_root, source, sidecar, _, receipt_sha, evidence = _prepared_runtime(
        tmp_path, ("DNA binding", "LLM inference")
    )
    assignments = [
        _positive_assignment(evidence["files"][0], "DNA"),
        _positive_assignment(evidence["files"][1], "LLM"),
    ]
    _write_plan(sidecar, evidence, assignments=assignments)

    def fail_reconciliation(**_kwargs):
        raise runtime.TypedAssignmentRuntimeError("injected reconcile failure")

    monkeypatch.setattr(runtime, "_reconcile_final_tree", fail_reconciliation)
    with pytest.raises(
        runtime.TypedAssignmentRuntimeError, match="injected reconcile failure"
    ):
        runtime.apply_assignment_plan(
            sidecar_dir=sidecar,
            expected_prepare_receipt_sha256=receipt_sha,
        )

    assert sorted(path.name for path in source.iterdir()) == [
        "paper-0.docx",
        "paper-1.docx",
    ]
    assert all(not (task_root / name).exists() for name in runtime.PUBLIC_ORGANIZE_DESTINATIONS)


def test_public_default_requires_empty_evidence_and_exact_default_destination(
    tmp_path: Path,
) -> None:
    task_root, source, sidecar, _, receipt_sha, evidence = _prepared_runtime(
        tmp_path
    )
    row = evidence["files"][0]
    _write_plan(
        sidecar,
        evidence,
        assignments=[
            {
                "file_id": row["file_id"],
                "destination": "music_history",
                "basis": "public_default",
                "evidence_ids": [],
            }
        ],
    )

    receipt = runtime.apply_assignment_plan(
        sidecar_dir=sidecar,
        expected_prepare_receipt_sha256=receipt_sha,
    )

    assert receipt["public_default_assignment_count"] == 1
    assert (task_root / "music_history" / "paper-0.docx").is_file()
    assert not any(source.iterdir())
