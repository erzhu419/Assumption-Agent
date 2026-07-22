from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from assumption_agent.benchmarks import frames_p1_source_qualification_v1 as qualification


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_CUSTODY_PATH = PROJECT_ROOT / "manifests/frames_p1_source_custody_v1.json"
REAL_BRIGHT_DISPOSITION_PATH = (
    PROJECT_ROOT
    / "manifests/bright_p17_postterminal_view_exposure_disposition_v1.json"
)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _semantic_hash(value: object) -> str:
    return hashlib.sha256(_canonical(value).rstrip(b"\n")).hexdigest()


def _git_blob_sha1(raw: bytes) -> str:
    return hashlib.sha1(f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def fixed_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    originals = (
        qualification.EXPECTED_SOURCE_SIZE_BYTES,
        qualification.EXPECTED_SOURCE_GIT_BLOB_SHA1,
        qualification.EXPECTED_CUSTODY_SELF_SHA256,
    )
    repository = tmp_path / "repository"
    project = repository / "reconstruction_v2"
    qualifier_copy = (
        project
        / "assumption_agent/benchmarks/frames_p1_source_qualification_v1.py"
    )
    test_copy = project / "tests/test_frames_p1_source_qualification_v1.py"
    qualifier_copy.parent.mkdir(parents=True)
    test_copy.parent.mkdir(parents=True)
    qualifier_copy.write_bytes(Path(qualification.__file__).read_bytes())
    test_copy.write_bytes(Path(__file__).read_bytes())
    monkeypatch.setattr(qualification, "GIT_TOP", repository)
    monkeypatch.setattr(qualification, "QUALIFIER_PATH", qualifier_copy)
    monkeypatch.setattr(qualification, "TEST_PATH", test_copy)
    monkeypatch.setattr(
        qualification, "SOURCE_PATH", project / "artifacts/source/test.tsv"
    )
    monkeypatch.setattr(
        qualification,
        "DOWNLOAD_RECEIPT_PATH",
        project / "manifests/frames_p1_source_download_receipt_v1.json",
    )
    monkeypatch.setattr(
        qualification,
        "CUSTODY_PATH",
        project / "manifests/frames_p1_source_custody_v1.json",
    )
    monkeypatch.setattr(
        qualification,
        "BRIGHT_DISPOSITION_PATH",
        project
        / "manifests/bright_p17_postterminal_view_exposure_disposition_v1.json",
    )
    monkeypatch.setattr(
        qualification,
        "FREEZE_PATH",
        project / "manifests/frames_p1_source_qualification_freeze_v1.json",
    )
    monkeypatch.setattr(
        qualification, "MARKER_PATH", project / "artifacts/marker.json"
    )
    monkeypatch.setattr(
        qualification, "FAILURE_PATH", project / "artifacts/failure.json"
    )
    monkeypatch.setattr(
        qualification, "RESULT_PATH", project / "manifests/result.json"
    )
    qualification.BRIGHT_DISPOSITION_PATH.parent.mkdir(parents=True)
    qualification.BRIGHT_DISPOSITION_PATH.write_bytes(
        REAL_BRIGHT_DISPOSITION_PATH.read_bytes()
    )
    yield
    (
        qualification.EXPECTED_SOURCE_SIZE_BYTES,
        qualification.EXPECTED_SOURCE_GIT_BLOB_SHA1,
        qualification.EXPECTED_CUSTODY_SELF_SHA256,
    ) = originals


def _row(row_id: int, family_index: int) -> dict[str, str]:
    types = (
        "Temporal reasoning",
        "Numerical reasoning | Multiple constraints",
        "Multiple constraints | Post processing",
    )[family_index % 3]
    links = [
        f"https://en.wikipedia.org/wiki/Synthetic_Page_{row_id}_A#section",
        f"https://en.wikipedia.org/wiki/Synthetic_Page_{row_id}_B",
    ]
    row = {column: "" for column in qualification.EXPECTED_COLUMNS}
    row.update(
        {
            "Unnamed: 0": str(row_id),
            "Prompt": f"PRIVATE_PROMPT_{row_id}",
            "Answer": f"PRIVATE_ANSWER_{row_id}",
            "wikipedia_link_1": links[0],
            "wikipedia_link_2": links[1],
            "reasoning_types": types,
            "wiki_links": repr(links),
        }
    )
    return row


def _write_custody_and_freeze(raw: bytes) -> None:
    source_size = len(raw)
    source_git_blob = _git_blob_sha1(raw)
    qualification.EXPECTED_SOURCE_SIZE_BYTES = source_size
    qualification.EXPECTED_SOURCE_GIT_BLOB_SHA1 = source_git_blob

    custody = json.loads(REAL_CUSTODY_PATH.read_text("ascii"))
    custody.pop("self_sha256")
    custody["official_git_blob_sha1"] = source_git_blob
    custody["official_source_size_bytes"] = source_size
    custody_self = _semantic_hash(custody)
    qualification.EXPECTED_CUSTODY_SELF_SHA256 = custody_self
    qualification.CUSTODY_PATH.write_bytes(
        _canonical({**custody, "self_sha256": custody_self})
    )

    repository = qualification.GIT_TOP
    subprocess.run(
        ["/usr/bin/git", "init", "-q", str(repository)], check=True
    )
    subprocess.run(
        ["/usr/bin/git", "-C", str(repository), "config", "user.name", "test"],
        check=True,
    )
    subprocess.run(
        [
            "/usr/bin/git",
            "-C",
            str(repository),
            "config",
            "user.email",
            "test@example.invalid",
        ],
        check=True,
    )
    tracked = [
        qualification.QUALIFIER_PATH,
        qualification.TEST_PATH,
        qualification.CUSTODY_PATH,
        qualification.BRIGHT_DISPOSITION_PATH,
    ]
    subprocess.run(
        [
            "/usr/bin/git",
            "-C",
            str(repository),
            "add",
            "--",
            *[str(path.relative_to(repository)) for path in tracked],
        ],
        check=True,
    )
    subprocess.run(
        [
            "/usr/bin/git",
            "-C",
            str(repository),
            "commit",
            "-q",
            "-m",
            "synthetic implementation freeze",
        ],
        check=True,
    )
    implementation_commit = subprocess.run(
        ["/usr/bin/git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()

    bindings = {
        "qualifier": {
            "relative_path": (
                "assumption_agent/benchmarks/"
                "frames_p1_source_qualification_v1.py"
            ),
            "sha256": _file_sha256(qualification.QUALIFIER_PATH),
        },
        "tests": {
            "relative_path": "tests/test_frames_p1_source_qualification_v1.py",
            "sha256": _file_sha256(qualification.TEST_PATH),
        },
        "source_custody": {
            "relative_path": "manifests/frames_p1_source_custody_v1.json",
            "sha256": _file_sha256(qualification.CUSTODY_PATH),
        },
        "bright_exposure_disposition": {
            "relative_path": (
                "manifests/"
                "bright_p17_postterminal_view_exposure_disposition_v1.json"
            ),
            "sha256": _file_sha256(qualification.BRIGHT_DISPOSITION_PATH),
        },
    }
    body = {
        "schema": qualification.FREEZE_SCHEMA,
        "status": "frozen_before_persisted_source_download_and_formal_qualification",
        "implementation_commit": implementation_commit,
        "file_bindings": bindings,
        "source_repository_object": {
            "dataset_repository": qualification.EXPECTED_REPOSITORY,
            "dataset_revision": qualification.EXPECTED_REVISION,
            "dataset_file": qualification.EXPECTED_DATASET_FILE,
            "git_blob_sha1": source_git_blob,
            "size_bytes": source_size,
        },
        "pre_freeze_public_viewer_validation": {
            "row_id_interval_half_open": [0, 90],
            "header_matches_expected": True,
            "reasoning_types_pipe_delimited_scalar_rows": 90,
            "wiki_links_python_list_string_rows": 90,
            "first_10_plus_11th_scalar_columns_match_rows": 90,
            "question_answer_or_URL_values_output_count": 0,
        },
        "pre_freeze_nonsemantic_source_stream": {
            "attempt_count": 1,
            "reason": (
                "raw_URL_returned_content_instead_of_expected_git_pointer"
            ),
            "persisted": False,
            "row_or_cell_semantically_parsed": False,
            "question_answer_URL_or_row_value_output_count": 0,
            "candidate_metric_quota_or_parser_adaptation_from_content": False,
        },
        "formal_source_file_present_at_freeze": False,
        "source_download_receipt_present_at_freeze": False,
        "formal_qualification_attempt_count_at_freeze": 0,
        "model_action_or_score_count_at_freeze": 0,
        "online_evaluator_or_API_calls_at_freeze": 0,
    }
    qualification.FREEZE_PATH.write_bytes(
        _canonical({**body, "self_sha256": _semantic_hash(body)})
    )


def _write_source(
    *,
    wrong_header: bool = False,
    all_temporal: bool = False,
    noncanonical_row_id: bool = False,
    formation_collisions: bool = False,
) -> None:
    qualification.SOURCE_PATH.parent.mkdir(parents=True)
    columns = list(qualification.EXPECTED_COLUMNS)
    if wrong_header:
        columns[-1] = "wrong_links"
    with qualification.SOURCE_PATH.open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for row_id in range(qualification.EXPECTED_ROW_COUNT):
            row = _row(row_id, 0 if all_temporal else row_id)
            if noncanonical_row_id and row_id == 7:
                row["Unnamed: 0"] = "007"
            if formation_collisions and row_id == 100:
                row["Prompt"] = "PRIVATE_PROMPT_0"
            if formation_collisions and row_id == 101:
                row["Answer"] = "PRIVATE_ANSWER_1"
            if formation_collisions and row_id == 102:
                links = [
                    "https://en.wikipedia.org/wiki/Synthetic_Page_2_A#other",
                    "https://en.wikipedia.org/wiki/Synthetic_Page_2_B",
                ]
                row["wikipedia_link_1"] = links[0]
                row["wikipedia_link_2"] = links[1]
                row["wiki_links"] = repr(links)
            if wrong_header:
                row["wrong_links"] = row.pop("wiki_links")
            writer.writerow({key: row[key] for key in columns})
    raw = qualification.SOURCE_PATH.read_bytes()
    _write_custody_and_freeze(raw)
    body = {
        "schema": qualification.DOWNLOAD_RECEIPT_SCHEMA,
        "status": "downloaded_exact_pinned_official_source",
        "dataset_repository": qualification.EXPECTED_REPOSITORY,
        "dataset_revision": qualification.EXPECTED_REVISION,
        "dataset_file": qualification.EXPECTED_DATASET_FILE,
        "pinned_download_url": qualification.EXPECTED_DOWNLOAD_URL,
        "fixed_local_source_path": (
            "artifacts/frames_p1_official_source_v1/test.tsv"
        ),
        "source_file_sha256": hashlib.sha256(raw).hexdigest(),
        "source_git_blob_sha1": _git_blob_sha1(raw),
        "source_size_bytes": len(raw),
        "source_content_semantically_opened_during_download": False,
        "external_network_use": "pinned_source_download_only",
        "online_evaluator_or_API_calls": 0,
    }
    qualification.DOWNLOAD_RECEIPT_PATH.write_bytes(
        _canonical({**body, "self_sha256": _semantic_hash(body)})
    )


def _assert_self_hash(path: Path) -> None:
    value = json.loads(path.read_text("ascii"))
    declared = value.pop("self_sha256")
    assert declared == _semantic_hash(value)


def test_success_reports_only_aggregate_disjoint_capacity() -> None:
    _write_source()

    result = qualification.run_source_qualification()

    assert result["qualified"] is True
    assert result["row_count"] == 824
    assert result["eligible_pre_disjoint_counts_by_family"] == {
        "constraint_postprocess": 241,
        "structured": 242,
        "temporal": 241,
    }
    assert result["deterministic_disjoint_pool_counts_by_family"] == {
        "constraint_postprocess": 48,
        "structured": 48,
        "temporal": 48,
    }
    assert result["ineligible_counts"] == {
        "exposed_formation_interval": 100,
        "gold_link_count_outside_2_through_5": 0,
        "noncanonical_or_duplicate_normalized_gold_link": 0,
        "formation_prompt_collision": 0,
        "formation_answer_collision": 0,
        "formation_gold_page_collision": 0,
    }
    raw = qualification.RESULT_PATH.read_text("ascii")
    assert "PRIVATE_PROMPT" not in raw
    assert "PRIVATE_ANSWER" not in raw
    assert "Synthetic_Page" not in raw
    assert json.loads(raw) == result
    assert qualification.MARKER_PATH.is_file()
    assert not qualification.FAILURE_PATH.exists()
    assert qualification.MARKER_PATH.stat().st_mode & 0o777 == 0o600
    assert qualification.RESULT_PATH.stat().st_mode & 0o777 == 0o600
    _assert_self_hash(qualification.MARKER_PATH)
    _assert_self_hash(qualification.RESULT_PATH)


def test_header_drift_consumes_attempt_and_writes_no_row_values() -> None:
    _write_source(wrong_header=True)

    with pytest.raises(
        qualification.FramesP1SourceQualificationError,
        match="failed terminally",
    ):
        qualification.run_source_qualification()

    failure = json.loads(qualification.FAILURE_PATH.read_text("ascii"))
    assert failure["failure_stage"] == (
        "header_cell_grammar_and_aggregate_disjoint_capacity"
    )
    assert failure["question_answer_url_or_row_id_values_output_count"] == 0
    assert not qualification.RESULT_PATH.exists()
    _assert_self_hash(qualification.FAILURE_PATH)
    with pytest.raises(
        qualification.FramesP1SourceQualificationError,
        match="already consumed",
    ):
        qualification.run_source_qualification()


def test_custody_drift_is_post_marker_terminal() -> None:
    _write_source()
    custody = json.loads(qualification.CUSTODY_PATH.read_text("ascii"))
    custody["dataset_revision"] = "0" * 40
    body = dict(custody)
    body.pop("self_sha256")
    custody["self_sha256"] = _semantic_hash(body)
    qualification.CUSTODY_PATH.write_bytes(_canonical(custody))

    with pytest.raises(qualification.FramesP1SourceQualificationError):
        qualification.run_source_qualification()

    assert qualification.MARKER_PATH.is_file()
    failure = json.loads(qualification.FAILURE_PATH.read_text("ascii"))
    assert failure["failure_stage"] == "frozen_custody_and_implementation_binding"


def test_bound_source_detects_path_replacement_after_parse() -> None:
    _write_source()
    download = qualification._validate_download_receipt()
    replacement = qualification.SOURCE_PATH.with_suffix(".replacement")
    replacement.write_bytes(qualification.SOURCE_PATH.read_bytes())

    with pytest.raises(
        qualification.FramesP1SourceQualificationError,
        match="changed during qualification",
    ):
        with qualification._open_bound_source(download) as handle:
            assert handle.readline()
            os.replace(replacement, qualification.SOURCE_PATH)


def test_capacity_failure_is_terminal_result_not_retryable_exception() -> None:
    _write_source(all_temporal=True)

    result = qualification.run_source_qualification()

    assert result["qualified"] is False
    assert result["status"] == "terminal_FRAMES_aggregate_disjoint_capacity_failed"
    assert result["deterministic_disjoint_pool_counts_by_family"] == {
        "constraint_postprocess": 0,
        "structured": 0,
        "temporal": 48,
    }
    assert qualification.RESULT_PATH.is_file()
    assert not qualification.FAILURE_PATH.exists()


def test_noncanonical_row_id_is_terminal() -> None:
    _write_source(noncanonical_row_id=True)

    with pytest.raises(qualification.FramesP1SourceQualificationError):
        qualification.run_source_qualification()

    assert qualification.FAILURE_PATH.is_file()
    assert not qualification.RESULT_PATH.exists()


def test_formation_content_collisions_are_excluded_aggregately() -> None:
    _write_source(formation_collisions=True)

    result = qualification.run_source_qualification()

    assert result["qualified"] is True
    counts = result["ineligible_counts"]
    assert counts["formation_prompt_collision"] == 1
    assert counts["formation_answer_collision"] == 1
    assert counts["formation_gold_page_collision"] == 1
    raw = qualification.RESULT_PATH.read_text("ascii")
    assert "PRIVATE_" not in raw
    assert "Synthetic_Page" not in raw


def test_family_precedence_and_pipe_grammar_are_fixed() -> None:
    assert qualification._family(("Temporal reasoning", "Numerical reasoning")) == (
        "temporal"
    )
    assert qualification._family(
        ("Tabular reasoning", "Multiple constraints")
    ) == "structured"
    assert qualification._family(("Post processing",)) == "constraint_postprocess"
    assert qualification._reasoning_types(
        "Numerical reasoning | Multiple constraints"
    ) == ("Numerical reasoning", "Multiple constraints")
    with pytest.raises(qualification.FramesP1SourceQualificationError):
        qualification._reasoning_types("Numerical reasoning|Multiple constraints")


def test_link_canonicalization_is_strict_and_fragment_stable() -> None:
    assert qualification._normalized_canonical_links(
        (
            "https://en.wikipedia.org/wiki/A#section",
            "https://en.wikipedia.org/wiki/B",
        )
    ) == (
        "https://en.wikipedia.org/wiki/A",
        "https://en.wikipedia.org/wiki/B",
    )
    invalid = [
        (
            "https://en.wikipedia.org/wiki/A#one",
            "https://en.wikipedia.org/wiki/A#two",
        ),
        ("https://en.wikipedia.org/w/index.php?search=A",),
        ("https://en.wikipedia.org:not-a-port/wiki/A",),
        ("https://en.wikipedia.org/wiki/Special:Search",),
        ("https://en.wikipedia.org/wiki/Special%3ASearch",),
        (" https://en.wikipedia.org/wiki/A",),
        ("https://en.wikipedia.org/wiki/A%ZZ",),
        ("https://en.wikipedia.org/wiki/../w/index.php",),
        ("https://en.wikipedia.org/wiki/%2e%2e/w/index.php",),
        ("https://en.wikipedia.org/wiki/A\\B",),
        ("https://en.wikipedia.org/wiki/A%5CB",),
    ]
    for links in invalid:
        assert qualification._normalized_canonical_links(links) is None


@pytest.mark.parametrize("count", [10, 11, 12, 15])
def test_redundant_link_columns_cover_all_official_boundaries(count: int) -> None:
    links = [f"https://en.wikipedia.org/wiki/Boundary_{index}" for index in range(count)]
    row = {column: "" for column in qualification.EXPECTED_COLUMNS}
    for index, link in enumerate(links[:10], start=1):
        row[f"wikipedia_link_{index}"] = link
    if count == 11:
        row["wikipedia_link_11+"] = links[10]
    if count > 11:
        row["wikipedia_link_11+"] = repr(links[10:])
    qualification._validate_redundant_link_columns(row, links)
    if count > 11:
        row["wikipedia_link_11+"] = "arbitrary-nonempty-not-the-11th-link"
        with pytest.raises(qualification.FramesP1SourceQualificationError):
            qualification._validate_redundant_link_columns(row, links)


@pytest.mark.parametrize("fake_commit", ["0" * 40, "f" * 40])
def test_freeze_rejects_zero_or_nonexistent_commit(fake_commit: str) -> None:
    _write_source()
    freeze = json.loads(qualification.FREEZE_PATH.read_text("ascii"))
    freeze["implementation_commit"] = fake_commit
    body = dict(freeze)
    body.pop("self_sha256")
    freeze["self_sha256"] = _semantic_hash(body)
    qualification.FREEZE_PATH.write_bytes(_canonical(freeze))

    with pytest.raises(qualification.FramesP1SourceQualificationError):
        qualification.run_source_qualification()

    assert qualification.MARKER_PATH.is_file()
    assert qualification.FAILURE_PATH.is_file()
    assert not qualification.RESULT_PATH.exists()


def test_git_replace_ref_cannot_change_frozen_commit_binding() -> None:
    _write_source()
    freeze = json.loads(qualification.FREEZE_PATH.read_text("ascii"))
    original = freeze["implementation_commit"]
    empty_tree = subprocess.run(
        ["/usr/bin/git", "-C", str(qualification.GIT_TOP), "mktree"],
        check=True,
        input=b"",
        stdout=subprocess.PIPE,
    ).stdout.decode("ascii").strip()
    replacement = subprocess.run(
        [
            "/usr/bin/git",
            "-C",
            str(qualification.GIT_TOP),
            "commit-tree",
            empty_tree,
            "-m",
            "replacement object must be ignored",
        ],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout.decode("ascii").strip()
    subprocess.run(
        [
            "/usr/bin/git",
            "-C",
            str(qualification.GIT_TOP),
            "replace",
            original,
            replacement,
        ],
        check=True,
    )

    result = qualification.run_source_qualification()

    assert result["qualified"] is True
    assert result["qualification_freeze_self_sha256"] == freeze["self_sha256"]


def test_cli_has_no_source_result_family_or_quota_override() -> None:
    options = {
        action.dest
        for action in qualification._parser()._actions
        if action.dest != "help"
    }
    assert options == set()


def test_main_returns_two_for_capacity_terminal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        qualification,
        "run_source_qualification",
        lambda: {"qualified": False, "status": "terminal"},
    )
    assert qualification._main([]) == 2
    assert json.loads(capsys.readouterr().out)["qualified"] is False
