from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import stat
import subprocess
import sys
import tarfile
from typing import Any

import pytest

from assumption_agent.benchmarks import finqa_fresh_source_qualification_v1 as audit


PROJECT = Path(__file__).resolve().parents[1]
MODULE = (
    PROJECT
    / "assumption_agent"
    / "benchmarks"
    / "finqa_fresh_source_qualification_v1.py"
)
CUSTODY = PROJECT / "manifests" / "finqa_graph_evaluator_source_custody_v1.json"

SECRET_MARKERS = (
    "REPORT_DO_NOT_LEAK",
    "QUESTION_DO_NOT_LEAK",
    "PRE_DO_NOT_LEAK",
    "POST_DO_NOT_LEAK",
    "CELL_DO_NOT_LEAK",
    "GOLD_VALUE_DO_NOT_LEAK",
    "PROGRAM_DO_NOT_LEAK",
    "ANSWER_DO_NOT_LEAK",
    "UNKNOWN_FIELD_DO_NOT_LEAK",
    "EXTRA_MEMBER_DO_NOT_LEAK",
    "DECOY_MEMBER_DO_NOT_LEAK",
    "PRIVATE_TEST_DO_NOT_LEAK",
)


def _entry(
    report: str,
    index: int,
    *,
    pre_text: list[str] | None = None,
    post_text: list[str] | None = None,
    table: list[list[str]] | None = None,
    gold: dict[str, str] | None = None,
    program: str = "add(10, 20)",
    program_re: str | None = None,
    exe_ans: Any = 30,
) -> dict[str, Any]:
    pre = ["PRE_DO_NOT_LEAK"] if pre_text is None else pre_text
    post = ["POST_DO_NOT_LEAK"] if post_text is None else post_text
    rows = (
        [
            ["", "2020", "2021"],
            ["metric", "10", "20"],
            ["ragged", "CELL_DO_NOT_LEAK"],
        ]
        if table is None
        else table
    )
    evidence = (
        {
            "text_0": pre[0],
            "table_1": audit.table_row_to_text(rows[0], rows[1]),
        }
        if gold is None
        else gold
    )
    return {
        "id": f"{report}-{index}",
        "pre_text": pre,
        "post_text": post,
        "table": rows,
        "qa": {
            "question": f"QUESTION_DO_NOT_LEAK_{index}?",
            "program": program,
            "program_re": program if program_re is None else program_re,
            "gold_inds": evidence,
            "exe_ans": exe_ans,
            "answer": "ANSWER_DO_NOT_LEAK",
            "explanation": "GOLD_VALUE_DO_NOT_LEAK",
        },
        "UNKNOWN_FIELD_DO_NOT_LEAK": "UNKNOWN_VALUE_DO_NOT_LEAK",
    }


def _datasets() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normal_report = "REPORT_DO_NOT_LEAK-ALPHA/2020/page_1.pdf"
    normal_first = _entry(normal_report, 0)
    normal_second = _entry(
        normal_report,
        1,
        gold={
            "table_0": audit.table_row_to_text(
                normal_first["table"][0], normal_first["table"][0]
            ),
            "text_1": "GOLD_VALUE_DO_NOT_LEAK_mismatch",
        },
        exe_ans="yes",
    )
    row_fingerprint_table = [
        ["", "a", "b"],
        ["disclosed", "102,400 / 619,314", "0"],
        ["filler", "1", "2"],
    ]
    train = [
        normal_first,
        normal_second,
        _entry("ETR/2016/page_23.pdf", 0),
        _entry("GRMN/2006/page_91.pdf", 0),
        _entry(
            "REPORT_DO_NOT_LEAK-PROGRAM/2020/page_2.pdf",
            0,
            program=" divide( 102400, 619314 ) ",
        ),
        _entry(
            "REPORT_DO_NOT_LEAK-ROW/2020/page_3.pdf",
            0,
            table=row_fingerprint_table,
            gold={
                "text_0": "PRE_DO_NOT_LEAK",
                "table_1": audit.table_row_to_text(
                    row_fingerprint_table[0], row_fingerprint_table[1]
                ),
            },
        ),
        _entry("REPORT_DO_NOT_LEAK-OVERLAP/2020/page_4.pdf", 0),
        _entry(
            "REPORT_DO_NOT_LEAK-LOWGOLD/2020/page_5.pdf",
            0,
            gold={"text_0": "PRE_DO_NOT_LEAK"},
        ),
    ]
    dev = [
        _entry("REPORT_DO_NOT_LEAK-DEV/2021/page_1.pdf", 0),
        _entry("REPORT_DO_NOT_LEAK-OVERLAP/2020/page_4.pdf", 1),
    ]
    return train, dev


def _write_member(bundle: tarfile.TarFile, name: str, payload: bytes) -> None:
    member = tarfile.TarInfo(name=name)
    member.size = len(payload)
    member.mtime = 0
    bundle.addfile(member, io.BytesIO(payload))


def _archive(
    tmp_path: Path,
    *,
    train: Any | None = None,
    dev: Any | None = None,
    train_raw: bytes | None = None,
    dev_raw: bytes | None = None,
    include_decoys: bool = True,
) -> Path:
    default_train, default_dev = _datasets()
    train_payload = default_train if train is None else train
    dev_payload = default_dev if dev is None else dev
    archive = tmp_path / "finqa-synthetic.tar.gz"
    with tarfile.open(archive, mode="w:gz") as bundle:
        _write_member(
            bundle,
            audit.EXPECTED_MEMBERS["train"],
            json.dumps(train_payload, ensure_ascii=False).encode("utf-8")
            if train_raw is None
            else train_raw,
        )
        _write_member(
            bundle,
            audit.EXPECTED_MEMBERS["dev"],
            json.dumps(dev_payload, ensure_ascii=False).encode("utf-8")
            if dev_raw is None
            else dev_raw,
        )
        if include_decoys:
            _write_member(
                bundle,
                "DECOY_MEMBER_DO_NOT_LEAK/dataset/train.json",
                b"not JSON and must not be opened",
            )
            _write_member(
                bundle,
                f"{audit.ARCHIVE_ROOT}/dataset/test.json",
                b"PRIVATE_TEST_DO_NOT_LEAK not JSON",
            )
            _write_member(
                bundle,
                f"{audit.ARCHIVE_ROOT}/dataset/private_test.json",
                b"PRIVATE_TEST_DO_NOT_LEAK not JSON",
            )
            _write_member(
                bundle,
                f"{audit.ARCHIVE_ROOT}/code/evaluate/test.json",
                b"PRIVATE_TEST_DO_NOT_LEAK not JSON",
            )
            _write_member(
                bundle,
                "EXTRA_MEMBER_DO_NOT_LEAK.txt",
                b"EXTRA_MEMBER_CONTENT_DO_NOT_LEAK",
            )
    return archive


def _without_hash(receipt: dict[str, Any]) -> dict[str, Any]:
    body = dict(receipt)
    body.pop("qualification_sha256", None)
    return body


def _rehash(receipt: dict[str, Any]) -> None:
    receipt.pop("qualification_sha256", None)
    receipt["qualification_sha256"] = hashlib.sha256(
        json.dumps(
            receipt,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _contains_list(value: Any) -> bool:
    if isinstance(value, list):
        return True
    if isinstance(value, dict):
        return any(_contains_list(item) for item in value.values())
    return False


def test_official_table_serializer_branches_are_frozen() -> None:
    assert audit.table_row_to_text(
        ["", " FY  2020 ", "FY\t2021"],
        [" net  income ", "  10 ", "20"],
    ) == "the net income of FY 2020 is 10 ; the net income of FY\t2021 is 20 ;"
    assert audit.table_row_to_text(
        [" unit  USD ", "2020", "2021"], ["revenue", "10"]
    ) == "unit USD the revenue of 2020 is 10 ;"
    assert audit.table_row_to_text(
        ["", "2020"], ["revenue", "10", "ignored"]
    ) == "the revenue of 2020 is 10 ;"
    assert audit.remove_space(" a\t b   c ") == "a\t b c"


def test_formal_constants_bind_the_frozen_custody_without_source_rows() -> None:
    raw = CUSTODY.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == audit.FORMAL_CUSTODY_MANIFEST_FILE_SHA256
    payload = json.loads(raw)
    body = dict(payload)
    declared = body.pop("custody_sha256")
    assert declared == audit.FORMAL_CUSTODY_SHA256
    assert hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest() == declared
    assert audit.FORMAL_CUSTODY_COMMIT == "6321cfe2"
    assert audit.FORMAL_ARCHIVE_SHA256.startswith("eec31e")


def test_synthetic_qualification_is_aggregate_only_and_exact(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    receipt = audit.build_qualification(archive, CUSTODY)
    serialized = json.dumps(receipt, ensure_ascii=False, sort_keys=True)
    for marker in SECRET_MARKERS:
        assert marker.casefold() not in serialized.casefold()
    assert not _contains_list(receipt)
    assert receipt["schema"] == audit.SCHEMA
    assert receipt["selection_status"] == "not_performed"
    assert receipt["selection_secret_opened_or_generated"] is False
    assert receipt["performance_or_retrieval_scoring_performed"] is False
    assert receipt["qualification_sha256"] == hashlib.sha256(
        json.dumps(
            _without_hash(receipt),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    archive_receipt = receipt["archive"]
    assert archive_receipt["exact_train_dev_members_opened"] == 2
    assert archive_receipt["test_private_or_evaluate_test_members_opened"] == 0
    assert archive_receipt["official_members"]["train"][
        "expected_relative_path"
    ] == audit.EXPECTED_MEMBERS["train"]

    train_nodes = receipt["addressable_graph"]["split_aggregates"]["train"]
    assert train_nodes["entry_count"] == 8
    assert train_nodes["table_0_candidate_count"] == 8
    assert train_nodes["empty_header_0_entry_count"] == 8
    assert train_nodes["entry_with_ragged_table_count"] == 7
    assert receipt["addressable_graph"]["ragged_rows_use_zip_truncation"] is True
    assert receipt["addressable_graph"]["duplicate_content_nodes_are_not_deduplicated"] is True

    train_gold = receipt["gold_mapping_diagnostics"]["split_aggregates"]["train"]
    assert train_gold["table_0_gold_unit_count"] == 1
    assert train_gold["gold_value_canonical_mismatch_count"] == 1
    assert train_gold["program_fingerprint_match_entry_count"] == 1
    assert train_gold["table_row_fingerprint_match_entry_count"] == 1

    overlap = receipt["report_grouping_and_duplicates"][
        "cross_split_report_overlap"
    ]
    assert overlap["cross_split_exact_report_overlap_count"] == 1
    train_exclusions = receipt["custody_exclusions"]["split_aggregates"]["train"]
    assert train_exclusions["official_example_denylist_report_count"] == 1
    assert train_exclusions["paper_figure_semantic_denylist_report_count"] == 1
    assert train_exclusions["program_content_fingerprint_report_count"] == 1
    assert train_exclusions["table_row_content_fingerprint_report_count"] == 1
    assert train_exclusions["union_custody_excluded_report_count"] == 4

    train_eligible = receipt["selection_eligibility"]["split_aggregates"]["train"]
    dev_eligible = receipt["selection_eligibility"]["split_aggregates"]["dev"]
    assert train_eligible[
        "formal_eligible_entry_count_before_one_question_per_report_cap"
    ] == 2
    assert train_eligible["formal_eligible_report_count_one_question_cap"] == 1
    assert dev_eligible["formal_eligible_report_count_one_question_cap"] == 1
    assert receipt["formal_capacity_decision"]["all_minimums_met"] is False
    assert receipt["status"] == audit.STATUS_DIAGNOSTIC


def test_public_cli_uses_clean_subprocess_on_synthetic_archive(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(MODULE),
            "--archive",
            str(archive),
            "--custody-manifest",
            str(CUSTODY),
        ],
        cwd=PROJECT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    receipt = json.loads(completed.stdout)
    assert audit._validate_child_receipt(receipt) == receipt
    assert completed.stdout.count("qualification_sha256") == 1
    for marker in SECRET_MARKERS:
        assert marker.casefold() not in completed.stdout.casefold()


def test_parent_output_is_atomic_exclusive_public_json_on_synthetic_archive(
    tmp_path: Path,
) -> None:
    archive = _archive(tmp_path)
    output = tmp_path / "qualification.json"
    command = [
        sys.executable,
        str(MODULE),
        "--archive",
        str(archive),
        "--custody-manifest",
        str(CUSTODY),
        "--output",
        str(output),
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    original = output.read_bytes()
    assert json.loads(original) == json.loads(completed.stdout)
    assert stat.S_IMODE(output.stat().st_mode) == 0o644
    assert not list(tmp_path.glob(".qualification.json.*.tmp"))

    repeated = subprocess.run(
        command,
        cwd=PROJECT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert repeated.returncode != 0
    assert output.read_bytes() == original
    assert not list(tmp_path.glob(".qualification.json.*.tmp"))


def test_worker_rejects_output_and_formal_parent_requires_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _archive(tmp_path)
    output = tmp_path / "must-not-exist.json"
    worker = subprocess.run(
        [
            sys.executable,
            str(MODULE),
            "--_aggregate-worker",
            "--archive",
            str(archive),
            "--custody-manifest",
            str(CUSTODY),
            "--output",
            str(output),
        ],
        cwd=PROJECT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert worker.returncode != 0
    assert not output.exists()

    def forbidden_run(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("formal parent must reject before qualification")

    monkeypatch.setattr(audit, "run_clean_qualification", forbidden_run)
    with pytest.raises(SystemExit):
        audit.main(
            [
                "--archive",
                str(archive),
                "--custody-manifest",
                str(CUSTODY),
                "--formal",
            ]
        )
    assert not output.exists()

    def failed_qualification(*_args: Any, **_kwargs: Any) -> Any:
        raise audit.FinqaQualificationError("synthetic parent failure")

    monkeypatch.setattr(audit, "run_clean_qualification", failed_qualification)
    with pytest.raises(audit.FinqaQualificationError, match="synthetic parent failure"):
        audit.main(
            [
                "--archive",
                str(archive),
                "--custody-manifest",
                str(CUSTODY),
                "--formal",
                "--output",
                str(output),
            ]
        )
    assert not output.exists()


def test_output_failure_leaves_no_file_or_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "failed.json"

    def fail_link(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("synthetic publication failure")

    monkeypatch.setattr(audit.os, "link", fail_link)
    with pytest.raises(OSError, match="synthetic publication failure"):
        audit._write_json_exclusive(output, {"schema": audit.SCHEMA})
    assert not output.exists()
    assert not list(tmp_path.glob(".failed.json.*.tmp"))


def test_clean_launcher_is_isolated_and_hides_worker_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _archive(tmp_path)
    captured: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        captured.update(kwargs)
        return subprocess.CompletedProcess(
            command,
            7,
            stdout="",
            stderr="QUESTION_DO_NOT_LEAK must not be forwarded",
        )

    monkeypatch.setattr(audit.subprocess, "run", fake_run)
    with pytest.raises(audit.FinqaQualificationError, match="clean aggregate worker failed"):
        audit.run_clean_qualification(archive, CUSTODY, enforce_formal_bindings=True)
    assert captured["command"][1] == "-I"
    assert captured["command"][-1] == "--formal"
    assert "--output" not in captured["command"]
    assert captured["stdin"] is subprocess.DEVNULL
    assert captured["close_fds"] is True
    assert set(captured["env"]) == {"PATH", "PYTHONHASHSEED", "LC_ALL"}


def test_only_exact_train_and_dev_members_are_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _archive(tmp_path)
    opened: list[str] = []
    original = tarfile.TarFile.extractfile

    def recording_extractfile(
        self: tarfile.TarFile, member: tarfile.TarInfo | str
    ) -> Any:
        opened.append(member.name if isinstance(member, tarfile.TarInfo) else member)
        return original(self, member)

    monkeypatch.setattr(tarfile.TarFile, "extractfile", recording_extractfile)
    audit.build_qualification(archive, CUSTODY)
    assert opened == [
        audit.EXPECTED_MEMBERS["train"],
        audit.EXPECTED_MEMBERS["dev"],
    ]
    assert not set(opened) & audit.FORBIDDEN_DATA_MEMBERS


def test_duplicate_exact_member_is_rejected_but_basename_decoy_is_not_selected(
    tmp_path: Path,
) -> None:
    train, dev = _datasets()
    raw_train = json.dumps(train).encode("utf-8")
    archive = tmp_path / "duplicate.tar.gz"
    with tarfile.open(archive, mode="w:gz") as bundle:
        _write_member(bundle, audit.EXPECTED_MEMBERS["train"], raw_train)
        _write_member(bundle, audit.EXPECTED_MEMBERS["train"], raw_train)
        _write_member(
            bundle,
            audit.EXPECTED_MEMBERS["dev"],
            json.dumps(dev).encode("utf-8"),
        )
    with pytest.raises(audit.FinqaQualificationError, match="duplicate exact official"):
        audit.build_qualification(archive, CUSTODY)


def test_missing_exact_member_is_not_satisfied_by_a_basename_decoy(tmp_path: Path) -> None:
    train, dev = _datasets()
    archive = tmp_path / "missing-exact.tar.gz"
    with tarfile.open(archive, mode="w:gz") as bundle:
        _write_member(
            bundle,
            audit.EXPECTED_MEMBERS["train"],
            json.dumps(train).encode("utf-8"),
        )
        _write_member(
            bundle,
            "DECOY_MEMBER_DO_NOT_LEAK/dataset/dev.json",
            json.dumps(dev).encode("utf-8"),
        )
    with pytest.raises(audit.FinqaQualificationError, match="lacks an exact official"):
        audit.build_qualification(archive, CUSTODY)


def test_duplicate_json_object_key_is_rejected(tmp_path: Path) -> None:
    _train, dev = _datasets()
    raw = (
        b'[{"id":"REPORT_DO_NOT_LEAK/2020/page_1.pdf-0",'
        b'"id":"REPORT_DO_NOT_LEAK/2020/page_1.pdf-1"}]'
    )
    archive = _archive(tmp_path, train_raw=raw, dev=dev, include_decoys=False)
    with pytest.raises(audit.FinqaQualificationError, match="duplicate JSON object key"):
        audit.build_qualification(archive, CUSTODY)


def test_duplicate_full_entry_id_is_rejected_across_splits(tmp_path: Path) -> None:
    duplicate = _entry("REPORT_DO_NOT_LEAK/2020/page_1.pdf", 0)
    archive = _archive(
        tmp_path,
        train=[duplicate],
        dev=[deepcopy(duplicate)],
        include_decoys=False,
    )
    with pytest.raises(audit.FinqaQualificationError, match="duplicate full entry id"):
        audit.build_qualification(archive, CUSTODY)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.__setitem__("id", None), "required id"),
        (lambda row: row.__setitem__("pre_text", None), "required pre_text"),
        (lambda row: row.__setitem__("post_text", [1]), "required post_text"),
        (lambda row: row.__setitem__("table", []), "table must be a nonempty"),
        (lambda row: row.__setitem__("table", [[""], []]), "each table row"),
        (lambda row: row.__setitem__("qa", None), "required qa"),
        (lambda row: row["qa"].__setitem__("question", None), "required question"),
        (lambda row: row["qa"].__setitem__("program", None), "required program"),
        (lambda row: row["qa"].__setitem__("program_re", None), "required program_re"),
        (lambda row: row["qa"].__setitem__("exe_ans", True), "unsupported public type"),
        (lambda row: row["qa"].__setitem__("gold_inds", None), "gold_inds must"),
    ],
)
def test_required_public_schema_rejects_wrong_types(
    tmp_path: Path, mutation: Any, message: str
) -> None:
    train, dev = _datasets()
    train = [deepcopy(train[0])]
    mutation(train[0])
    archive = _archive(tmp_path, train=train, dev=dev, include_decoys=False)
    with pytest.raises(audit.FinqaQualificationError, match=message):
        audit.build_qualification(archive, CUSTODY)


def test_root_and_nonfinite_json_are_rejected(tmp_path: Path) -> None:
    _train, dev = _datasets()
    archive = _archive(tmp_path, train={}, dev=dev, include_decoys=False)
    with pytest.raises(audit.FinqaQualificationError, match="root must be a JSON list"):
        audit.build_qualification(archive, CUSTODY)

    bad = _entry("REPORT_DO_NOT_LEAK/2020/page_1.pdf", 0)
    bad["qa"]["exe_ans"] = float("nan")
    archive = _archive(
        tmp_path,
        train=[bad],
        dev=dev,
        include_decoys=False,
    )
    with pytest.raises(audit.FinqaQualificationError, match="non-finite constant"):
        audit.build_qualification(archive, CUSTODY)


@pytest.mark.parametrize(
    ("gold", "message"),
    [
        ({"text_00": "x", "table_1": "y"}, "canonical node id"),
        ({"table_-1": "x", "text_0": "y"}, "canonical node id"),
        ({"evidence_0": "x", "text_0": "y"}, "canonical node id"),
        ({"table_99": "x", "text_0": "y"}, "table gold key is out of bounds"),
        ({"text_99": "x", "table_1": "y"}, "text gold key is out of bounds"),
        ({"text_0": None, "table_1": "y"}, "bind text keys to text"),
    ],
)
def test_gold_node_ids_are_canonical_and_bounded(
    tmp_path: Path, gold: dict[str, Any], message: str
) -> None:
    _train, dev = _datasets()
    train = [_entry("REPORT_DO_NOT_LEAK/2020/page_1.pdf", 0, gold=gold)]
    archive = _archive(tmp_path, train=train, dev=dev, include_decoys=False)
    with pytest.raises(audit.FinqaQualificationError, match=message):
        audit.build_qualification(archive, CUSTODY)


def test_table_0_gold_empty_text_lists_and_extra_fields_are_supported(
    tmp_path: Path,
) -> None:
    table = [
        ["label", "2020", "2021"],
        ["a", "1", "2"],
        ["b", "3", "4"],
        ["c", "5", "6"],
        ["d", "7", "8"],
    ]
    train = [
        _entry(
            "REPORT_DO_NOT_LEAK-WITH-HYPHEN/2020/page-x.pdf",
            12,
            pre_text=[],
            post_text=[],
            table=table,
            gold={
                "table_0": audit.table_row_to_text(table[0], table[0]),
                "table_4": audit.table_row_to_text(table[0], table[4]),
            },
            exe_ans="no",
        )
    ]
    dev = [_entry("REPORT_DO_NOT_LEAK-DEV/2021/page_1.pdf", 0)]
    archive = _archive(tmp_path, train=train, dev=dev, include_decoys=False)
    receipt = audit.build_qualification(archive, CUSTODY)
    train_nodes = receipt["addressable_graph"]["split_aggregates"]["train"]
    train_gold = receipt["gold_mapping_diagnostics"]["split_aggregates"]["train"]
    assert train_nodes["nonempty_header_0_entry_count"] == 1
    assert train_nodes["addressable_node_occurrence_count"] == 5
    assert train_gold["table_0_gold_unit_count"] == 1
    assert train_gold["gold_value_exact_canonical_match_count"] == 2
    assert audit.parse_report_id(
        "REPORT_DO_NOT_LEAK-WITH-HYPHEN/2020/page-x.pdf-12"
    ) == ("REPORT_DO_NOT_LEAK-WITH-HYPHEN/2020/page-x.pdf", 12)
    with pytest.raises(audit.FinqaQualificationError, match="canonical report suffix"):
        audit.parse_report_id("REPORT_DO_NOT_LEAK/page.pdf-01")


def test_denylist_and_content_fingerprint_are_frozen() -> None:
    assert audit.report_matches_paper_figure_denylist("GRMN/2006/page_91.pdf")
    assert audit.report_matches_paper_figure_denylist("garmin/2006/page_91.pdf")
    assert not audit.report_matches_paper_figure_denylist("GRMN/2007/page_91.pdf")
    assert audit.program_matches_disclosed_fingerprint(" divide( 102400, 619314 ) ")
    assert not audit.program_matches_disclosed_fingerprint("divide(619314, 102400)")
    assert audit.row_matches_disclosed_fingerprint(
        ["ratio", "102,400", "/", "619,314"]
    )
    assert not audit.row_matches_disclosed_fingerprint(
        ["ratio", "102,400", "619,314"]
    )


def test_report_disjoint_capacity_uses_one_question_per_report(tmp_path: Path) -> None:
    train = [
        _entry(f"TRAIN-REPORT-{index}/2020/page_1.pdf", 0)
        for index in range(audit.FORMAL_MINIMUM_DISTINCT_ELIGIBLE_REPORTS["train"])
    ]
    dev = [
        _entry(f"DEV-REPORT-{index}/2021/page_1.pdf", 0)
        for index in range(audit.FORMAL_MINIMUM_DISTINCT_ELIGIBLE_REPORTS["dev"])
    ]
    train.append(_entry("TRAIN-REPORT-0/2020/page_1.pdf", 1))
    archive = _archive(tmp_path, train=train, dev=dev, include_decoys=False)
    receipt = audit.build_qualification(archive, CUSTODY)
    decision = receipt["formal_capacity_decision"]
    assert decision["all_minimums_met"] is True
    assert decision["distinct_eligible_report_minimums"]["train"][
        "observed_distinct_eligible_reports"
    ] == 192
    assert decision["distinct_eligible_report_minimums"]["dev"][
        "observed_distinct_eligible_reports"
    ] == 64
    assert receipt["selection_eligibility"]["split_aggregates"]["train"][
        "report_with_multiple_questions_count"
    ] == 1


def test_formal_byte_mismatch_stops_before_archive_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _archive(tmp_path)

    def forbidden_archive_parse(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("row parse must not run after a byte binding mismatch")

    monkeypatch.setattr(audit, "_read_archive", forbidden_archive_parse)
    with pytest.raises(audit.FinqaQualificationError, match="archive byte binding"):
        audit.build_qualification(archive, CUSTODY, enforce_formal_bindings=True)

    archive_hash, archive_size = audit._sha256_path(archive)
    monkeypatch.setattr(audit, "FORMAL_ARCHIVE_SHA256", archive_hash)
    monkeypatch.setattr(audit, "FORMAL_ARCHIVE_SIZE", archive_size)
    payload = json.loads(CUSTODY.read_text(encoding="utf-8"))
    payload["recorded_date"] = "2099-01-01"
    body = dict(payload)
    body.pop("custody_sha256", None)
    payload["custody_sha256"] = hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    altered = tmp_path / "altered-custody.json"
    altered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(audit.FinqaQualificationError, match="custody manifest file"):
        audit.build_qualification(archive, altered, enforce_formal_bindings=True)


def test_child_receipt_validation_rejects_tampering_and_row_channels(
    tmp_path: Path,
) -> None:
    receipt = audit.build_qualification(_archive(tmp_path), CUSTODY)
    bad_hash = deepcopy(receipt)
    bad_hash["qualification_sha256"] = "0" * 64
    with pytest.raises(audit.FinqaQualificationError, match="hash mismatch"):
        audit._validate_child_receipt(bad_hash)

    private_string = deepcopy(receipt)
    private_string["leak"] = "QUESTION_DO_NOT_LEAK"
    _rehash(private_string)
    with pytest.raises(audit.FinqaQualificationError, match="private string"):
        audit._validate_child_receipt(private_string)

    row_array = deepcopy(receipt)
    row_array["leak"] = [audit.SCHEMA]
    _rehash(row_array)
    with pytest.raises(audit.FinqaQualificationError, match="row arrays"):
        audit._validate_child_receipt(row_array)

    operation = deepcopy(receipt)
    operation["qualification_operations"][
        "concrete_item_or_report_identifiers_emitted"
    ] = 1
    _rehash(operation)
    with pytest.raises(audit.FinqaQualificationError, match="violates redaction"):
        audit._validate_child_receipt(operation)
