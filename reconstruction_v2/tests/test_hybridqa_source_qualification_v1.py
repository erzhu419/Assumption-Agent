from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest

from assumption_agent.benchmarks import hybridqa_source_qualification_v1 as qualification


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def _raw_row(
    qid: str,
    table_id: str,
    *,
    answer: str | None,
    question: str,
) -> dict[str, str]:
    row = {
        "question": question,
        "question_id": qid,
        "question_postag": "NN VB",
        "table_id": table_id,
    }
    if answer is not None:
        row["answer-text"] = answer
    return row


def _traced(row: dict[str, str], nodes: list[list[object]]) -> dict[str, object]:
    return {**row, "answer-node": nodes}


def _table(table_id: str, *, linked: bool) -> dict[str, object]:
    links = ["/private_link"] if linked else []
    return {
        "data": [
            [["private_cell", links], ["second_cell", []]],
            [["another_cell", []], ["last_cell", []]],
        ],
        "header": [["first_header", []], ["second_header", []]],
        "intro": "private intro",
        "section_text": "private section",
        "section_title": "private section title",
        "title": "private title",
        "uid": table_id,
        "url": "private URL",
    }


def _fixture() -> tuple[dict[str, object], list[tuple[str, object, object]]]:
    train = _raw_row(
        "private_train_qid",
        "private_table_one",
        answer="private_cell",
        question="private train question",
    )
    dev = _raw_row(
        "private_dev_qid",
        "private_table_one",
        answer="private_cell",
        question="private dev question",
    )
    test = _raw_row(
        "private_test_qid",
        "private_table_two",
        answer=None,
        question="private test question",
    )
    payloads: dict[str, object] = {
        "train": [train],
        "train_traced": [
            _traced(
                train,
                [
                    ["private_cell", [0, 0], None, "table"],
                    ["private passage answer", [0, 0], "/private_link", "passage"],
                ],
            )
        ],
        "dev": [dev],
        "dev_traced": [
            _traced(dev, [["private_cell", [0, 0], None, "table"]])
        ],
        "test": [test],
        "dev_reference": {
            "reference": {"private_dev_qid": "private_cell"},
            "table": ["private_dev_qid"],
            "passage": [],
        },
    }
    corpus = [
        (
            "private_table_one",
            _table("private_table_one", linked=True),
            {"/private_link": "private passage body"},
        ),
        (
            "private_table_two",
            _table("private_table_two", linked=False),
            {},
        ),
        # A pinned paper/source table may be unused by all QA splits.
        (
            "private_unused_table",
            _table("private_unused_table", linked=False),
            {},
        ),
    ]
    return payloads, corpus


def _qualify(
    payloads: dict[str, object],
    corpus: list[tuple[str, object, object]],
) -> dict[str, object]:
    return qualification.qualify_decoded_sources(
        payloads,
        corpus,
        expected_qa_counts={"train": 1, "dev": 1, "test": 1},
        expected_corpus_count=3,
        expected_dev_reference_partition={
            "table": 1,
            "passage": 0,
            "computed": 0,
        },
        qualification_code_sha256=HASH_A,
        qa_file_set_sha256=HASH_B,
        corpus_file_set_sha256=HASH_C,
    )


def test_synthetic_success_is_aggregate_only_and_allows_unused_tables() -> None:
    payloads, corpus = _fixture()
    receipt = _qualify(payloads, corpus)

    assert receipt["status"] == "synthetic_or_nonformal_aggregate_diagnostic"
    assert receipt["qa"] == {
        "train_row_count": 1,
        "dev_row_count": 1,
        "test_row_count": 1,
        "question_id_count": 3,
        "question_ids_unique_within_splits": True,
        "question_id_splits_pairwise_disjoint": True,
        "train_traced_raw_exact_match": True,
        "dev_traced_raw_exact_match": True,
        "train_empty_answer_node_row_count": 0,
        "dev_empty_answer_node_row_count": 0,
        "referenced_table_count": 2,
    }
    assert receipt["corpus"]["unused_table_count"] == 1
    assert receipt["answer_nodes"] == {
        "answer_node_count": 3,
        "table_source_count": 2,
        "passage_source_count": 1,
        "sources_coordinates_and_links_valid": True,
    }
    assert receipt["safeguards"]["selection_secret_created_or_read_count"] == 0
    serialized = json.dumps(receipt, sort_keys=True)
    for private_value in (
        "private_train_qid",
        "private_dev_qid",
        "private_test_qid",
        "private_table_one",
        "private question",
        "private_cell",
        "/private_link",
        "private passage body",
        "private title",
    ):
        assert private_value not in serialized


def test_traced_raw_field_disagreement_fails_closed() -> None:
    payloads, corpus = _fixture()
    payloads["train_traced"][0]["question"] = "changed private question"
    with pytest.raises(
        qualification.HybridQaSourceQualificationError,
        match="traced/raw fields disagree",
    ):
        _qualify(payloads, corpus)


def test_source_native_empty_answer_nodes_are_counted_not_rejected() -> None:
    payloads, corpus = _fixture()
    payloads["train_traced"][0]["answer-node"] = []

    receipt = _qualify(payloads, corpus)

    assert receipt["qa"]["train_empty_answer_node_row_count"] == 1
    assert receipt["answer_nodes"] == {
        "answer_node_count": 1,
        "table_source_count": 1,
        "passage_source_count": 0,
        "sources_coordinates_and_links_valid": True,
    }


def test_source_native_empty_request_passage_is_counted_not_rejected() -> None:
    payloads, corpus = _fixture()
    corpus[0][2]["/private_link"] = ""

    receipt = _qualify(payloads, corpus)

    assert receipt["corpus"]["empty_request_entry_count"] == 1
    assert receipt["corpus"]["empty_request_link_reference_count"] == 1


def test_question_id_overlap_between_splits_fails_closed() -> None:
    payloads, corpus = _fixture()
    payloads["test"][0]["question_id"] = "private_train_qid"
    with pytest.raises(
        qualification.HybridQaSourceQualificationError,
        match="split question IDs overlap",
    ):
        _qualify(payloads, corpus)


def test_dataset_table_must_exist_in_exact_table_request_pair_set() -> None:
    payloads, corpus = _fixture()
    payloads["test"][0]["table_id"] = "private_absent_table"
    with pytest.raises(
        qualification.HybridQaSourceQualificationError,
        match="dataset table ID is absent",
    ):
        _qualify(payloads, corpus)


def test_unresolvable_cell_link_fails_closed() -> None:
    payloads, corpus = _fixture()
    corpus[0][1]["data"][0][0][1] = ["/missing_link"]
    with pytest.raises(
        qualification.HybridQaSourceQualificationError,
        match="not exactly resolvable",
    ):
        _qualify(payloads, corpus)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payloads: payloads["train_traced"][0]["answer-node"][0].__setitem__(
                1, [99, 0]
            ),
            "outside table data",
        ),
        (
            lambda payloads: payloads["train_traced"][0]["answer-node"][1].__setitem__(
                2, "/other_link"
            ),
            "does not resolve at its cell",
        ),
        (
            lambda payloads: payloads["train_traced"][0]["answer-node"][0].__setitem__(
                3, "unknown"
            ),
            "source is invalid",
        ),
    ],
)
def test_answer_node_source_coordinate_and_link_are_strict(mutate, message: str) -> None:
    payloads, corpus = _fixture()
    mutate(payloads)
    with pytest.raises(qualification.HybridQaSourceQualificationError, match=message):
        _qualify(payloads, corpus)


def test_table_answer_node_must_equal_exact_cell() -> None:
    payloads, corpus = _fixture()
    payloads["train_traced"][0]["answer-node"][0][0] = "not the cell"
    with pytest.raises(
        qualification.HybridQaSourceQualificationError,
        match="does not resolve exactly to its cell",
    ):
        _qualify(payloads, corpus)


def test_dev_reference_is_exact_and_partitioned() -> None:
    payloads, corpus = _fixture()
    payloads["dev_reference"]["table"] = []
    with pytest.raises(
        qualification.HybridQaSourceQualificationError,
        match="partition counts drifted",
    ):
        _qualify(payloads, corpus)


def test_strict_json_rejects_duplicate_keys_and_nonfinite_numbers() -> None:
    with pytest.raises(
        qualification.HybridQaSourceQualificationError,
        match="duplicate JSON keys",
    ):
        qualification._decode_strict_json(b'{"x":1,"x":2}', label="synthetic")
    with pytest.raises(
        qualification.HybridQaSourceQualificationError,
        match="non-finite",
    ):
        qualification._decode_strict_json(b'{"x":NaN}', label="synthetic")


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_git_custody_pins_commit_tree_file_set_and_clean_checkout(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "source"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "synthetic@example.invalid")
    _git(repo, "config", "user.name", "Synthetic Test")
    tracked = repo / "tracked.txt"
    tracked.write_text("synthetic\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-q", "-m", "synthetic")
    commit = _git(repo, "rev-parse", "HEAD")
    tree = _git(repo, "rev-parse", "HEAD^{tree}")

    custody = qualification._verify_git_checkout(
        repo,
        expected_commit=commit,
        expected_tree=tree,
        repository_label="synthetic",
    )
    assert custody["commit"] == commit
    assert custody["tree"] == tree
    assert custody["tracked_file_count"] == 1
    assert len(custody["tracked_file_set_sha256"]) == 64

    tracked.write_text("changed\n", encoding="utf-8")
    with pytest.raises(
        qualification.HybridQaSourceQualificationError,
        match="not clean",
    ):
        qualification._verify_git_checkout(
            repo,
            expected_commit=commit,
            expected_tree=tree,
            repository_label="synthetic",
        )


def test_corpus_uid_and_rectangular_cell_schema_are_strict() -> None:
    payloads, corpus = _fixture()
    corpus_bad_uid = deepcopy(corpus)
    corpus_bad_uid[0][1]["uid"] = "different"
    with pytest.raises(
        qualification.HybridQaSourceQualificationError,
        match="uid/filename stem drifted",
    ):
        _qualify(payloads, corpus_bad_uid)

    payloads, corpus = _fixture()
    corpus[0][1]["data"][0].pop()
    with pytest.raises(
        qualification.HybridQaSourceQualificationError,
        match="row width drifted",
    ):
        _qualify(payloads, corpus)


def test_formal_constants_bind_official_commits_trees_and_counts() -> None:
    assert qualification.FORMAL_HYBRIDQA_COMMIT == (
        "db22fda8c5951438fade3c69d75b350335ba93b3"
    )
    assert qualification.FORMAL_HYBRIDQA_TREE == (
        "1e1ef6a6168ef6c6cf362264d8f7b75859ce8fdf"
    )
    assert qualification.FORMAL_WIKITABLES_COMMIT == (
        "dc066e1a6d5281511d8b73a6107d5ad2824cc2b2"
    )
    assert qualification.FORMAL_WIKITABLES_TREE == (
        "b4f2d5e0eeb2d18cf95bf6e6a583bc499c53b68c"
    )
    assert qualification.FORMAL_QA_COUNTS == {
        "train": 62_682,
        "dev": 3_466,
        "test": 3_463,
    }
    assert qualification.FORMAL_CORPUS_COUNT == 15_316
    assert qualification.FORMAL_DEV_REFERENCE_PARTITION == {
        "table": 1_349,
        "passage": 2_025,
        "computed": 92,
    }


def test_module_exposes_no_formal_cli_or_persistence_path() -> None:
    assert not hasattr(qualification, "main")
    assert not hasattr(qualification, "FORMAL_OUTPUT_RELATIVE")
    assert not hasattr(qualification, "FORMAL_MARKER_RELATIVE")
