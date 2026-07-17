from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import stat
import zipfile

import pytest

from assumption_agent.benchmarks import fever_fixed_p_itemlocal_acquisition_v1 as f
from assumption_agent.benchmarks import fever_fixed_p_itemlocal_runner_v1 as runner


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()


def _paper_row(label: str, index: int) -> dict[str, object]:
    page = f"Gold_{label}_{index}"
    return {
        "id": index if label == "SUPPORTS" else 10_000 + index,
        "verifiable": "VERIFIABLE",
        "label": label,
        "claim": f"common claim token {label.casefold()} {index}",
        "evidence": [[[index, index, page, 0]]],
    }


def _write_paper(path: Path, *, supports: int = 64, refutes: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        *(_paper_row("SUPPORTS", index) for index in range(supports)),
        *(_paper_row("REFUTES", index) for index in range(refutes)),
    ]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    path.chmod(0o600)


def _wiki_page(page_id: str, sentence: str) -> dict[str, str]:
    return {"id": page_id, "text": sentence, "lines": f"0\t{sentence}\n"}


def _write_wiki(path: Path, *, supports: int = 64, refutes: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pages = [
        *(
            _wiki_page(
                f"Gold_SUPPORTS_{index}",
                f"common claim token supports {index} gold evidence",
            )
            for index in range(supports)
        ),
        *(
            _wiki_page(
                f"Gold_REFUTES_{index}",
                f"common claim token refutes {index} gold evidence",
            )
            for index in range(refutes)
        ),
        *(
            _wiki_page(
                f"Negative_{index}",
                f"common claim token supports refutes distractor {index}",
            )
            for index in range(40)
        ),
    ]
    raw = "".join(
        json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in pages
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("wiki-pages/", b"")
        archive.writestr("wiki-pages/wiki-001.jsonl", raw)
    path.chmod(0o600)


def _write_license(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"synthetic license fixture\n")
    path.chmod(0o600)


def _patch_synthetic_assets(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    expectations: dict[str, dict[str, object]] = {}
    for asset_id, relative in {
        "paper_test": f.PAPER_TEST_RELATIVE,
        "wiki_pages": f.WIKI_ZIP_RELATIVE,
        "license": f.LICENSE_RELATIVE,
    }.items():
        path = root / relative
        expectations[asset_id] = {
            "relative_path": relative,
            "size_bytes": path.stat().st_size,
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    monkeypatch.setattr(f, "ASSET_EXPECTATIONS", expectations)


def _synthetic_source(root: Path, *, supports: int = 64, refutes: int = 64) -> None:
    _write_paper(root / f.PAPER_TEST_RELATIVE, supports=supports, refutes=refutes)
    _write_wiki(root / f.WIKI_ZIP_RELATIVE, supports=supports, refutes=refutes)
    _write_license(root / f.LICENSE_RELATIVE)


def test_public_design_and_source_manifests_are_exact_committed_bindings() -> None:
    bindings = f.verify_public_source_bindings(PROJECT_ROOT)
    assert set(bindings) == {"source_custody", "source_access", "design"}
    assert bindings["source_custody"]["file_sha256"] == f.SOURCE_CUSTODY_FILE_SHA256
    assert bindings["source_access"]["file_sha256"] == f.SOURCE_ACCESS_FILE_SHA256
    assert bindings["design"]["file_sha256"] == f.DESIGN_FILE_SHA256


def test_paper_selection_is_label_exact_hmac_deterministic_and_wiki_blind(
    tmp_path: Path,
) -> None:
    paper = tmp_path / "paper.jsonl"
    _write_paper(paper)
    candidates, row_count = f.load_paper_candidates(paper)
    assert row_count == len(candidates) == 128
    first, stats = f.select_candidates(candidates, b"a" * 32)
    again, _ = f.select_candidates(candidates, b"a" * 32)
    other, _ = f.select_candidates(candidates, b"b" * 32)
    assert first == again
    assert [row.candidate.item_id_hash for row in first] != [
        row.candidate.item_id_hash for row in other
    ]
    assert stats["selected_label_counts"] == {"SUPPORTS": 64, "REFUTES": 64}
    assert stats["selected_evidence_set_cardinality_histogram"] == {"1": 128}
    assert all(len(row.selected_set) == 1 for row in first)


def test_evidence_set_is_fixed_before_resolution_and_annotated_alternatives_excluded() -> None:
    raw = _paper_row("SUPPORTS", 0)
    raw["evidence"] = [
        [[1, 1, "Page_A", 0], [1, 2, "Page_A", 0]],
        [[2, 3, "Page_B", 1]],
        [[3, 4, None, None]],
    ]
    candidate = f._parse_paper_row(raw, 0)
    assert candidate is not None
    assert candidate.eligible_sets == (
        (f.EvidenceRef("Page_A", 0),),
        (f.EvidenceRef("Page_B", 1),),
    )
    assert candidate.all_annotated_refs == {
        f.EvidenceRef("Page_A", 0),
        f.EvidenceRef("Page_B", 1),
    }
    selected, _ = f.select_candidates(
        tuple(
            f.PaperCandidate(
                source_row_ordinal=index,
                exact_id=index,
                item_id_hash=_canonical_hash(index),
                source_label="SUPPORTS" if index < 64 else "REFUTES",
                claim=f"claim {index}",
                identity_commitment_sha256=_canonical_hash({"row": index}),
                eligible_sets=candidate.eligible_sets,
                all_annotated_refs=candidate.all_annotated_refs,
            )
            for index in range(128)
        ),
        b"c" * 32,
    )
    assert all(row.selected_set in candidate.eligible_sets for row in selected)


def test_two_pass_full_wiki_bm25_index_and_private_pack_contract(tmp_path: Path) -> None:
    paper = tmp_path / "paper.jsonl"
    wiki = tmp_path / "wiki.zip"
    _write_paper(paper)
    _write_wiki(wiki)
    candidates, _ = f.load_paper_candidates(paper)
    selected, _ = f.select_candidates(candidates, b"d" * 32)
    statistics, resolved, wiki_stats = f.scan_wiki_pass1(
        wiki_zip=wiki,
        selected=selected,
        identity_ledger_path=tmp_path / "identity.sqlite3",
    )
    assert wiki_stats["chosen_reference_count"] == 128
    assert wiki_stats["chosen_reference_resolved_count"] == 128
    negatives = f.mine_hard_negatives_pass2(
        wiki_zip=wiki,
        selected=selected,
        statistics=statistics,
    )
    assert {len(rows) for rows in negatives.values()} == {31}
    for item_i, rows in negatives.items():
        excluded = selected[item_i].candidate.all_annotated_refs
        assert all(
            f.EvidenceRef(sentence.page_id, sentence.line_number) not in excluded
            for _score, sentence in rows
        )
    action, labels, stats = f.build_private_packs(
        selected=selected,
        resolved=resolved,
        hard_negatives=negatives,
        statistics=statistics,
        secret=b"d" * 32,
    )
    assert action["item_count"] == labels["item_count"] == 128
    assert stats["selected_label_counts"] == {"SUPPORTS": 64, "REFUTES": 64}
    assert "source_label" not in json.dumps(action)
    for action_row, label_row in zip(action["items"], labels["items"], strict=True):
        assert [row["doc_id"] for row in action_row["documents"]] == list(range(32))
        assert sorted(action_row["bm25_rank"]) == list(range(32))
        assert len(action_row["bm25_scores"]) == 32
        assert label_row["action_item_sha256"] == action_row["action_item_sha256"]
        assert 1 <= len(label_row["gold_indices"]) <= 5


def test_secret_marker_precedes_single_urandom_and_source_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze = {"implementation_freeze_sha256": hashlib.sha256(b"freeze").hexdigest()}
    freeze_binding = {
        "file_sha256": hashlib.sha256(b"freeze-file").hexdigest(),
        "git_blob_sha1": hashlib.sha1(b"freeze-blob").hexdigest(),
        "current_git_HEAD": hashlib.sha1(b"head").hexdigest(),
    }
    monkeypatch.setattr(f, "_require_artifact_ignored", lambda _root: None)
    monkeypatch.setattr(
        f,
        "load_committed_implementation_freeze",
        lambda _root: (freeze, freeze_binding),
    )
    calls: list[bool] = []

    def urandom(size: int) -> bytes:
        marker = tmp_path / f.SECRET_MARKER_RELATIVE
        assert marker.is_file()
        assert size == 32
        calls.append(True)
        return b"s" * 32

    monkeypatch.setattr(f.os, "urandom", urandom)
    custody = f.create_selection_custody(tmp_path)
    assert calls == [True]
    assert custody["selection_secret_commitment_sha256"] == hashlib.sha256(
        b"s" * 32
    ).hexdigest()
    assert stat.S_IMODE((tmp_path / f.SELECTION_SECRET_RELATIVE).stat().st_mode) == 0o600
    assert custody["source_bytes_opened_listed_decoded_or_parsed"] == 0


def test_formal_acquire_marker_precedes_first_parse_and_receipt_is_aggregate_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _synthetic_source(tmp_path)
    _patch_synthetic_assets(monkeypatch, tmp_path)
    monkeypatch.setattr(f, "_require_artifact_ignored", lambda _root: None)
    custody = {
        "selection_custody_sha256": hashlib.sha256(b"custody").hexdigest(),
        "selection_secret_commitment_sha256": hashlib.sha256(b"z" * 32).hexdigest(),
        "implementation_freeze_sha256": hashlib.sha256(b"freeze").hexdigest(),
    }
    custody_binding = {
        "file_sha256": hashlib.sha256(b"custody-file").hexdigest(),
        "git_blob_sha1": hashlib.sha1(b"custody-blob").hexdigest(),
        "verified_at_git_HEAD": hashlib.sha1(b"head").hexdigest(),
    }
    monkeypatch.setattr(
        f,
        "load_committed_selection_custody",
        lambda _root: (custody, custody_binding, b"z" * 32),
    )
    original_paper = f.load_paper_candidates
    original_pass1 = f.scan_wiki_pass1
    observations: list[str] = []

    def checked_paper(path: Path):
        assert (tmp_path / f.ACQUISITION_MARKER_RELATIVE).is_file()
        observations.append("paper")
        return original_paper(path)

    def checked_pass1(**kwargs):
        assert (tmp_path / f.ACQUISITION_MARKER_RELATIVE).is_file()
        observations.append("wiki")
        return original_pass1(**kwargs)

    monkeypatch.setattr(f, "load_paper_candidates", checked_paper)
    monkeypatch.setattr(f, "scan_wiki_pass1", checked_pass1)
    receipt = f.acquire(tmp_path)
    assert observations == ["paper", "wiki"]
    assert receipt["status"] == f.ACQUISITION_STATUS
    assert receipt["safety"]["source_labels_and_evidence_read_only_by_acquisition"] is True
    assert receipt["safety"]["label_pack_opened_by_action_runner_before_action_barrier"] is False
    assert receipt["wiki_aggregate"]["full_wiki_pass_count"] == 2
    action_path = tmp_path / f.ACTION_PACK_RELATIVE
    label_path = tmp_path / f.LABEL_PACK_RELATIVE
    assert stat.S_IMODE(action_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(label_path.stat().st_mode) == 0o600
    commitments = receipt["commitments"]
    loaded_action = runner.load_action_pack(
        action_path,
        expected_file_sha256=commitments["action_pack_file_sha256"],
        expected_item_commitment_set_sha256=commitments[
            "action_item_commitment_set_sha256"
        ],
    )
    loaded_labels = runner.load_label_pack(
        label_path,
        expected_file_sha256=commitments["label_pack_file_sha256"],
        expected_item_commitment_set_sha256=commitments[
            "label_item_commitment_set_sha256"
        ],
    )
    assert len(loaded_action.items) == len(loaded_labels.items) == 128
    public = json.dumps(receipt, ensure_ascii=True, sort_keys=True)
    for forbidden in (
        "Gold_SUPPORTS",
        "common claim",
        '\"source_label\":',
        '\"sentence_text\":',
    ):
        assert forbidden not in public


def test_capacity_failure_is_post_marker_terminal_and_never_opens_wiki(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _synthetic_source(tmp_path, supports=63, refutes=64)
    _patch_synthetic_assets(monkeypatch, tmp_path)
    monkeypatch.setattr(f, "_require_artifact_ignored", lambda _root: None)
    custody = {
        "selection_custody_sha256": hashlib.sha256(b"custody").hexdigest(),
        "selection_secret_commitment_sha256": hashlib.sha256(b"q" * 32).hexdigest(),
        "implementation_freeze_sha256": hashlib.sha256(b"freeze").hexdigest(),
    }
    binding = {
        "file_sha256": hashlib.sha256(b"file").hexdigest(),
        "git_blob_sha1": hashlib.sha1(b"blob").hexdigest(),
        "verified_at_git_HEAD": hashlib.sha1(b"head").hexdigest(),
    }
    monkeypatch.setattr(
        f,
        "load_committed_selection_custody",
        lambda _root: (custody, binding, b"q" * 32),
    )
    wiki_calls: list[bool] = []
    monkeypatch.setattr(
        f,
        "scan_wiki_pass1",
        lambda **_kwargs: wiki_calls.append(True),
    )
    with pytest.raises(f.FeverAcquisitionError, match="capacity insufficient"):
        f.acquire(tmp_path)
    assert wiki_calls == []
    assert (tmp_path / f.ACQUISITION_MARKER_RELATIVE).is_file()
    failure = json.loads(
        (tmp_path / f.ACQUISITION_FAILURE_RELATIVE).read_text(encoding="ascii")
    )
    assert failure["status"] == "terminal_infrastructure_invalid_no_replay"
    assert failure["failure_class"] == "source_capacity_insufficient"
    with pytest.raises(f.FeverAcquisitionError, match="replay is forbidden"):
        f.acquire(tmp_path)


def test_cli_and_formal_entrypoints_have_no_source_or_output_path_overrides() -> None:
    assert set(inspect.signature(f.create_implementation_freeze).parameters) == {"project"}
    assert set(inspect.signature(f.create_selection_custody).parameters) == {"project"}
    assert set(inspect.signature(f.acquire).parameters) == {"project"}
    source = Path(f.__file__).read_text(encoding="utf-8")
    for forbidden in ("--paper-test", "--wiki", "--secret-path", "--output"):
        assert forbidden not in source
