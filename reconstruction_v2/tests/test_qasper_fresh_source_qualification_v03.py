from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from typing import Any

import pytest

from assumption_agent.benchmarks import qasper_fresh_source_qualification_v03 as audit


PROJECT = Path(__file__).resolve().parents[1]
MODULE = (
    PROJECT
    / "assumption_agent"
    / "benchmarks"
    / "qasper_fresh_source_qualification_v03.py"
)

SECRET_MARKERS = (
    "PAPER_ID_DO_NOT_LEAK",
    "QUESTION_ID_DO_NOT_LEAK",
    "QUESTION_TEXT_DO_NOT_LEAK",
    "BODY_ALPHA_DO_NOT_LEAK",
    "CAPTION_DO_NOT_LEAK",
    "ANSWER_DO_NOT_LEAK",
    "EVIDENCE_MISSING_DO_NOT_LEAK",
    "REFERENCE_PATH_DO_NOT_LEAK",
    "UNKNOWN_FIELD_DO_NOT_LEAK",
    "ARCHIVE_PREFIX_DO_NOT_LEAK",
    "EXTRA_MEMBER_DO_NOT_LEAK",
    "FILLER_ID_DO_NOT_LEAK",
    "DENY_BODY_DO_NOT_LEAK",
    "DENY_QUESTION_DO_NOT_LEAK",
    "HIGHLIGHTED_EVIDENCE_DO_NOT_LEAK",
)


def _annotation(
    evidence: list[str], *, unanswerable: bool = False, ordinal: int = 0
) -> dict[str, Any]:
    return {
        "answer": {
            "unanswerable": unanswerable,
            "extractive_spans": ["ANSWER_DO_NOT_LEAK"],
            "yes_no": None,
            "free_form_answer": "ANSWER_DO_NOT_LEAK",
            "evidence": evidence,
        },
        "annotation_id": f"ANNOTATION_DO_NOT_LEAK_{ordinal}",
        "worker_id": "WORKER_DO_NOT_LEAK",
        "highlighted_evidence": ["HIGHLIGHTED_EVIDENCE_DO_NOT_LEAK"],
    }


def _qa(
    ordinal: int,
    question: str,
    evidence: list[str],
    *,
    question_id: str | None = None,
    unanswerable: bool = False,
) -> dict[str, Any]:
    return {
        "question": question,
        "question_id": question_id or f"QUESTION_ID_DO_NOT_LEAK_{ordinal}",
        "nlp_background": "BACKGROUND_DO_NOT_LEAK",
        "topic_background": "TOPIC_DO_NOT_LEAK",
        "paper_read": "READ_DO_NOT_LEAK",
        "search_query": "QUERY_DO_NOT_LEAK",
        "answers": [_annotation(evidence, unanswerable=unanswerable, ordinal=ordinal)],
    }


def _paper(
    suffix: str,
    questions: list[dict[str, Any]],
    *,
    title: str,
    body: list[str],
) -> dict[str, Any]:
    return {
        "title": title,
        "abstract": [f"ABSTRACT_DO_NOT_LEAK_{suffix}"],
        "full_text": [
            {
                "section_name": f"SECTION_DO_NOT_LEAK_{suffix}",
                "paragraphs": body,
            }
        ],
        "qas": questions,
        "figures_and_tables": [
            {
                "caption": f"CAPTION_DO_NOT_LEAK_{suffix}",
                "file": f"FILE_DO_NOT_LEAK_{suffix}",
            }
        ],
        "UNKNOWN_FIELD_DO_NOT_LEAK": "UNKNOWN_VALUE_DO_NOT_LEAK",
    }


def _datasets() -> tuple[dict[str, Any], dict[str, Any]]:
    alpha = "BODY_ALPHA_DO_NOT_LEAK"
    beta = "BODY_BETA_DO_NOT_LEAK"
    gamma = "BODY_GAMMA_DO_NOT_LEAK"
    delta = "BODY_DELTA_DO_NOT_LEAK"
    epsilon = "BODY_EPSILON_DO_NOT_LEAK"
    collision_question = "QUESTION_TEXT_DO_NOT_LEAK collision target?"
    exposed_question = "QUESTION_TEXT_DO_NOT_LEAK disclosed target?"
    train_qas = [
        _qa(1, collision_question, [alpha, beta]),
        _qa(2, exposed_question, [], unanswerable=True),
        _qa(
            3,
            "QUESTION_TEXT_DO_NOT_LEAK float reference?",
            ["FLOAT SELECTED: CAPTION_DO_NOT_LEAK_train", alpha, beta],
        ),
        _qa(
            4,
            "QUESTION_TEXT_DO_NOT_LEAK caption is not a node?",
            [alpha, "CAPTION_DO_NOT_LEAK_train"],
        ),
        _qa(
            5,
            "QUESTION_TEXT_DO_NOT_LEAK only one distinct node?",
            [alpha, alpha],
        ),
        _qa(
            6,
            "QUESTION_TEXT_DO_NOT_LEAK valid two paragraph reference?",
            [gamma, delta],
        ),
    ]
    train: dict[str, Any] = {}
    for index in range(audit.TRAIN_SOURCE_INSERTION_EXCLUSION_COUNT):
        filler_body = [
            f"FILLER_BODY_DO_NOT_LEAK_{index}_{node}"
            for node in range(5 if index == 0 else 1)
        ]
        filler_qas = (
            [
                _qa(
                    98,
                    "FILLER_FIRST16_QUESTION_DO_NOT_LEAK?",
                    filler_body[:2],
                )
            ]
            if index == 0
            else []
        )
        train[f"FILLER_ID_DO_NOT_LEAK_{index}"] = _paper(
            f"filler_{index}",
            filler_qas,
            title=f"FILLER_TITLE_DO_NOT_LEAK_{index}",
            body=filler_body,
        )
    denied_body = [f"DENY_BODY_DO_NOT_LEAK_{index}" for index in range(5)]
    train[sorted(audit.PUBLIC_EXAMPLE_PAPER_ID_DENYLIST)[0]] = _paper(
        "denied",
        [
            _qa(
                99,
                "DENY_QUESTION_DO_NOT_LEAK?",
                denied_body[:2],
            )
        ],
        title="DENY_TITLE_DO_NOT_LEAK",
        body=denied_body,
    )
    train["PAPER_ID_DO_NOT_LEAK_train"] = _paper(
            "train",
            train_qas,
            title="TITLE_DO_NOT_LEAK_train",
            body=[alpha, alpha, beta, gamma, delta, epsilon],
    )

    dev_body = [
        "DEV_BODY_ONE_DO_NOT_LEAK",
        "DEV_BODY_TWO_DO_NOT_LEAK",
        "DEV_BODY_THREE_DO_NOT_LEAK",
        "DEV_BODY_FOUR_DO_NOT_LEAK",
        "DEV_BODY_FIVE_DO_NOT_LEAK",
    ]
    dev = {
        "PAPER_ID_DO_NOT_LEAK_dev": _paper(
            "dev",
            [
                _qa(7, collision_question, dev_body[:2]),
                _qa(
                    8,
                    "QUESTION_TEXT_DO_NOT_LEAK dev valid reference?",
                    dev_body[2:4],
                ),
            ],
            title="TITLE_DO_NOT_LEAK_dev",
            body=dev_body,
        )
    }
    return train, dev


def _write_member(bundle: tarfile.TarFile, name: str, payload: bytes) -> None:
    member = tarfile.TarInfo(name=name)
    member.size = len(payload)
    member.mtime = 0
    bundle.addfile(member, io.BytesIO(payload))


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    train, dev = _datasets()
    archive = tmp_path / "qasper-synthetic.tgz"
    with tarfile.open(archive, mode="w:gz") as bundle:
        _write_member(
            bundle,
            "ARCHIVE_PREFIX_DO_NOT_LEAK/qasper-train-v0.3.json",
            json.dumps(train, ensure_ascii=False).encode("utf-8"),
        )
        _write_member(
            bundle,
            "ARCHIVE_PREFIX_DO_NOT_LEAK/qasper-dev-v0.3.json",
            json.dumps(dev, ensure_ascii=False).encode("utf-8"),
        )
        _write_member(
            bundle,
            "EXTRA_MEMBER_DO_NOT_LEAK.txt",
            b"EXTRA_MEMBER_CONTENT_DO_NOT_LEAK",
        )
    reference = tmp_path / "REFERENCE_PATH_DO_NOT_LEAK.md"
    reference.write_text(
        "A prior disclosure contains: question_text_do_not_leak   DISCLOSED TARGET?",
        encoding="utf-8",
    )
    return archive, reference


def _without_hash(receipt: dict[str, Any]) -> dict[str, Any]:
    body = dict(receipt)
    body.pop("qualification_sha256")
    return body


def test_synthetic_qualification_is_exact_and_aggregate_only(tmp_path: Path) -> None:
    archive, reference = _fixture(tmp_path)
    receipt = audit.build_qualification(archive, reference)
    serialized = json.dumps(receipt, ensure_ascii=False, sort_keys=True)
    for marker in SECRET_MARKERS:
        assert marker not in serialized
        assert marker.casefold() not in serialized.casefold()
    for public_example_id in audit.PUBLIC_EXAMPLE_PAPER_ID_DENYLIST:
        assert public_example_id not in serialized

    assert receipt["schema"] == audit.SCHEMA
    assert receipt["selection_status"] == "not_performed"
    assert receipt["selection_secret_opened_or_generated"] is False
    assert receipt["qualification_sha256"] == hashlib.sha256(
        json.dumps(
            _without_hash(receipt),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    assert receipt["split_counts"] == {
        "train": {"paper_count": 18, "question_count": 8},
        "dev": {"paper_count": 1, "question_count": 2},
    }
    train_nodes = receipt["label_free_content_graph"]["split_aggregates"]["train"]
    assert train_nodes["body_paragraph_occurrence_count"] == 31
    assert train_nodes["unique_exact_nonempty_body_content_node_count"] == 30
    assert train_nodes["exact_duplicate_body_content_occurrence_count"] == 1
    assert train_nodes["figure_or_table_caption_occurrence_count"] == 18
    assert train_nodes["unique_exact_nonempty_caption_text_count"] == 18
    assert train_nodes["paper_with_5_to_128_unique_body_content_nodes_count"] == 3
    assert receipt["label_free_content_graph"][
        "figure_or_table_caption_is_a_content_node"
    ] is False

    train_gold = receipt["gold_evidence_scoreability"]["split_aggregates"]["train"]
    assert train_gold["annotation_count"] == 8
    assert train_gold["answerable_annotation_count"] == 7
    assert train_gold["unanswerable_annotation_count"] == 1
    assert train_gold["raw_empty_evidence_group_count"] == 1
    assert train_gold["float_selected_evidence_string_count"] == 1
    assert train_gold["post_float_empty_gold_group_count"] == 1
    assert train_gold["exact_deduplicated_node_match_count"] == 13
    assert train_gold["exact_missing_node_match_count"] == 1
    assert train_gold["deduplicated_node_ambiguous_match_count"] == 0
    assert train_gold["matched_node_multiple_occurrence_count"] == 5
    assert train_gold["scoreable_nonempty_gold_group_count"] == 6
    assert train_gold["unscoreable_nonempty_gold_group_count"] == 1
    assert train_gold[
        "text_only_all_exact_reference_ge1_distinct_body_node_count"
    ] == 5
    assert train_gold[
        "text_only_all_exact_reference_ge2_distinct_body_node_count"
    ] == 4
    assert receipt["field_schema"]["highlighted_evidence_presence"][
        "answer_annotation_field_presence_count"
    ] == 10
    assert receipt["field_schema"]["highlighted_evidence_presence"][
        "used_as_primary_gold_count"
    ] == 0

    collision = receipt["collision_audit"]
    assert collision["cross_split"]["normalized_question_overlap_count"] == 1
    assert collision["cross_split"]["exact_paper_id_overlap_count"] == 0
    train_eligible = receipt["selection_eligibility"]["split_aggregates"]["train"]
    assert train_eligible["label_free_5_to_128_node_candidate_count"] == 8
    assert train_eligible["paper_or_question_collision_exclusion_count"] == 1
    assert train_eligible["public_example_denylist_exclusion_question_count"] == 1
    assert train_eligible[
        "train_source_insertion_prefix_exclusion_question_count"
    ] == 1
    assert train_eligible["custody_exclusion_union_question_count"] == 2
    assert train_eligible[
        "exact_normalized_exposed_question_substring_exclusion_count"
    ] == 1
    assert train_eligible["exposure_clean_label_free_candidate_count"] == 4
    assert train_eligible["structural_label_ge1_question_count"] == 2
    assert train_eligible["structural_label_ge2_question_count"] == 1
    assert train_eligible["formal_eligible_paper_count_one_question_cap"] == 1
    source_exclusion = receipt["custody_source_exclusions"]["split_aggregates"][
        "train"
    ]
    assert source_exclusion == {
        "source_paper_count": 18,
        "public_example_denylist_matched_paper_count": 1,
        "train_source_insertion_prefix_excluded_paper_count": 16,
        "overlap_between_two_custody_exclusions_paper_count": 0,
        "union_custody_excluded_paper_count": 17,
    }
    dev_eligible = receipt["selection_eligibility"]["split_aggregates"]["dev"]
    assert dev_eligible["paper_or_question_collision_exclusion_count"] == 1
    assert dev_eligible["structural_label_ge2_question_count"] == 1
    assert dev_eligible["formal_eligible_paper_count_one_question_cap"] == 1


def test_public_cli_uses_clean_child_and_emits_only_one_json_receipt(
    tmp_path: Path,
) -> None:
    archive, reference = _fixture(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(MODULE),
            "--archive",
            str(archive),
            "--reference",
            str(reference),
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


def test_clean_launcher_is_isolated_and_does_not_forward_worker_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, reference = _fixture(tmp_path)
    captured: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        captured.update(kwargs)
        return subprocess.CompletedProcess(
            command,
            9,
            stdout="",
            stderr="QUESTION_TEXT_DO_NOT_LEAK must never be forwarded",
        )

    monkeypatch.setattr(audit.subprocess, "run", fake_run)
    with pytest.raises(audit.QasperQualificationError, match="clean aggregate worker failed"):
        audit.run_clean_qualification(
            archive, reference, enforce_formal_bindings=True
        )
    assert captured["command"][1] == "-I"
    assert captured["command"][-1] == "--formal"
    assert captured["stdin"] is subprocess.DEVNULL
    assert captured["close_fds"] is True
    assert set(captured["env"]) == {"PATH", "PYTHONHASHSEED", "LC_ALL"}


def test_archive_must_contain_exactly_one_member_for_each_official_split(
    tmp_path: Path,
) -> None:
    train, _ = _datasets()
    archive = tmp_path / "bad.tgz"
    raw = json.dumps(train).encode()
    with tarfile.open(archive, mode="w:gz") as bundle:
        _write_member(bundle, "a/qasper-train-v0.3.json", raw)
        _write_member(bundle, "b/qasper-train-v0.3.json", raw)
    reference = tmp_path / "reference.md"
    reference.write_text("disclosed", encoding="utf-8")
    with pytest.raises(audit.QasperQualificationError, match="duplicate official"):
        audit.build_qualification(archive, reference)


def test_normalization_and_float_heuristic_are_frozen() -> None:
    assert audit.normalize_text("  Ｃafé\tStraße １２ ") == "café strasse 12"
    assert "FLOAT SELECTED: Figure 1".startswith(audit.FLOAT_EVIDENCE_PREFIX)
    assert not "float selected: Figure 1".startswith(audit.FLOAT_EVIDENCE_PREFIX)


def test_distinct_nodes_are_not_unioned_across_references(tmp_path: Path) -> None:
    train, dev = _datasets()
    paper = train["PAPER_ID_DO_NOT_LEAK_train"]
    question = paper["qas"][4]
    question["answers"] = [
        _annotation(["BODY_ALPHA_DO_NOT_LEAK"], ordinal=50),
        _annotation(["BODY_BETA_DO_NOT_LEAK"], ordinal=51),
    ]
    archive = tmp_path / "multi-reference.tgz"
    with tarfile.open(archive, mode="w:gz") as bundle:
        _write_member(
            bundle,
            audit.EXPECTED_MEMBERS["train"],
            json.dumps(train).encode(),
        )
        _write_member(
            bundle,
            audit.EXPECTED_MEMBERS["dev"],
            json.dumps(dev).encode(),
        )
    reference = tmp_path / "reference.md"
    reference.write_text("no disclosed question here", encoding="utf-8")
    receipt = audit.build_qualification(archive, reference)
    train_gold = receipt["gold_evidence_scoreability"]["split_aggregates"]["train"]
    # The two one-node references remain two one-node references; they never
    # become a two-node gold set by unioning annotators.
    assert train_gold[
        "question_with_text_only_all_exact_reference_ge2_count"
    ] == 4


def test_global_title_and_full_text_duplicate_clusters_are_excluded(
    tmp_path: Path,
) -> None:
    train, dev = _datasets()
    title_a_body = [f"TITLE_A_BODY_DO_NOT_LEAK_{index}" for index in range(5)]
    title_b_body = [f"TITLE_B_BODY_DO_NOT_LEAK_{index}" for index in range(5)]
    train["TITLE_DUPLICATE_A_DO_NOT_LEAK"] = _paper(
        "title_a",
        [_qa(201, "TITLE_DUPLICATE_QUESTION_A_DO_NOT_LEAK?", title_a_body[:2])],
        title="  Shared   Collision Title  ",
        body=title_a_body,
    )
    train["TITLE_DUPLICATE_B_DO_NOT_LEAK"] = _paper(
        "title_b",
        [_qa(202, "TITLE_DUPLICATE_QUESTION_B_DO_NOT_LEAK?", title_b_body[:2])],
        title="shared collision title",
        body=title_b_body,
    )
    shared_body = [f"SHARED_FULL_TEXT_DO_NOT_LEAK_{index}" for index in range(5)]
    train["FULL_TEXT_DUPLICATE_TRAIN_DO_NOT_LEAK"] = _paper(
        "shared_full_text",
        [_qa(203, "FULL_TEXT_DUPLICATE_TRAIN_QUESTION_DO_NOT_LEAK?", shared_body[:2])],
        title="UNIQUE_TRAIN_STRUCTURE_TITLE_DO_NOT_LEAK",
        body=shared_body,
    )
    dev["FULL_TEXT_DUPLICATE_DEV_DO_NOT_LEAK"] = _paper(
        "shared_full_text",
        [_qa(204, "FULL_TEXT_DUPLICATE_DEV_QUESTION_DO_NOT_LEAK?", shared_body[:2])],
        title="UNIQUE_DEV_STRUCTURE_TITLE_DO_NOT_LEAK",
        body=shared_body,
    )
    archive = tmp_path / "paper-collisions.tgz"
    with tarfile.open(archive, mode="w:gz") as bundle:
        _write_member(
            bundle, audit.EXPECTED_MEMBERS["train"], json.dumps(train).encode()
        )
        _write_member(bundle, audit.EXPECTED_MEMBERS["dev"], json.dumps(dev).encode())
    reference = tmp_path / "reference.md"
    reference.write_text("no matching disclosure", encoding="utf-8")
    receipt = audit.build_qualification(archive, reference)
    paper_collision = receipt["collision_audit"][
        "paper_duplicate_class_exclusion"
    ]
    title = paper_collision["duplicate_key_classes"]["normalized_title"]
    assert title["global_duplicate_class_count"] == 1
    assert title["within_train_duplicate_class_count"] == 1
    structure = paper_collision["duplicate_key_classes"][
        "canonical_label_free_full_text_structure_content"
    ]
    assert structure["global_duplicate_class_count"] == 1
    assert structure["cross_split_duplicate_class_count"] == 1
    clusters = paper_collision["transitive_union_clusters"]
    assert clusters["duplicate_cluster_count"] == 2
    assert clusters["paper_count_in_duplicate_clusters"] == 4
    assert clusters["cross_split_duplicate_cluster_count"] == 1
    assert clusters["within_train_only_duplicate_cluster_count"] == 1
    train_eligible = receipt["selection_eligibility"]["split_aggregates"]["train"]
    dev_eligible = receipt["selection_eligibility"]["split_aggregates"]["dev"]
    assert train_eligible["paper_or_question_collision_exclusion_count"] == 4
    assert dev_eligible["paper_or_question_collision_exclusion_count"] == 2
    assert train_eligible["formal_eligible_paper_count_one_question_cap"] == 1
    assert dev_eligible["formal_eligible_paper_count_one_question_cap"] == 1
    serialized = json.dumps(receipt, sort_keys=True)
    assert "SHARED_FULL_TEXT_DO_NOT_LEAK" not in serialized
    assert "paper_label_free_commitment_sha256" not in serialized


def test_formal_byte_bindings_fail_before_any_dataset_row_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, reference = _fixture(tmp_path)

    def forbidden_row_open(_path: Path) -> Any:
        raise AssertionError("formal mismatch must stop before row parse")

    monkeypatch.setattr(audit, "_read_archive", forbidden_row_open)
    with pytest.raises(audit.QasperQualificationError, match="archive byte binding"):
        audit.build_qualification(
            archive, reference, enforce_formal_bindings=True
        )

    archive_hash, archive_size = audit._sha256_path(archive)
    monkeypatch.setattr(audit, "FORMAL_ARCHIVE_SHA256", archive_hash)
    monkeypatch.setattr(audit, "FORMAL_ARCHIVE_SIZE", archive_size)
    with pytest.raises(
        audit.QasperQualificationError, match="disclosed-reference binding"
    ):
        audit.build_qualification(
            archive, reference, enforce_formal_bindings=True
        )


def test_frozen_qualifier_rejects_official_reader_nullable_section_name(
    tmp_path: Path,
) -> None:
    """Postfailure diagnostic: the frozen parser is narrower than official code.

    The official Qasper baseline reader explicitly accepts a null
    ``section_name``.  This synthetic test records that the already-committed
    formal qualifier did not, without reopening the official source archive.
    It is diagnostic only and does not authorize a Qasper replay.
    """

    train, dev = _datasets()
    train["PAPER_ID_DO_NOT_LEAK_train"]["full_text"][0]["section_name"] = None
    archive = tmp_path / "nullable-section-name.tgz"
    with tarfile.open(archive, mode="w:gz") as bundle:
        _write_member(
            bundle, audit.EXPECTED_MEMBERS["train"], json.dumps(train).encode()
        )
        _write_member(
            bundle, audit.EXPECTED_MEMBERS["dev"], json.dumps(dev).encode()
        )
    reference = tmp_path / "reference.md"
    reference.write_text("no matching disclosure", encoding="utf-8")
    with pytest.raises(
        audit.QasperQualificationError, match="section name must be text"
    ):
        audit.build_qualification(archive, reference)
